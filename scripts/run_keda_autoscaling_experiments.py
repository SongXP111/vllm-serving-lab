#!/usr/bin/env python3
"""Step 6 Live Kubernetes Queue-Driven Autoscaling (KEDA + Prometheus + HPA) Harness.

Runs 4 live experiments on a local `kind` cluster (`llm-serving` namespace) where:
  Gateway (`gateway_queue_depth`) -> Prometheus -> KEDA ScaledObject -> HPA -> Deployment/mock-vllm
  -> Headless Service DNS -> Gateway dynamic `effective_max_inflight` capacity!

Experiments:
  1. `exp1_fixed_replica_burst`: Fixed 1 replica baseline on a 55s burst workload
     (high queue buildup, high P99 queue wait/E2E, significant 503 rejections).
  2. `exp2_autoscaled_burst_and_stabilization`: KEDA ScaledObject active (`min=1, max=4`)
     replaying the EXACT same 55s burst workload -> proves control lag during initial spike,
     scale-up rescue (`1 -> 4`), lower queue/P99/rejections, followed by HPA
     `scaleDown.stabilizationWindowSeconds: 45` anti-flapping hold and graceful step-down.
  3. `exp3_slow_cold_start_control_lag`: Increases `STARTUP_DELAY_SEC=15.0` to measure
     slow cold-start control-lag amplification (`spec_replicas` vs `ready_replicas` gap
     and temporary overshoot while pods load weights).
  4. `exp4_metrics_outage_fallback`: Scales Prometheus to 0 to simulate metrics outage ->
     proves KEDA `fallback` (`failureThreshold: 2, replicas: 2`) protects capacity instead
     of scaling down to 1/0, and recovers automatically when Prometheus returns.

Note: Measures Mock control-plane autoscaling behavior, NOT multi-GPU inference TPS gains.
"""
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

NAMESPACE = "llm-serving"
GATEWAY_URL = "http://localhost:30080"
SCALED_OBJECT_YAML = "k8s/keda-scaledobject.yaml"
OUT_DIR = Path("results/k8s_autoscaling")

# Shared deterministic burst workload parameters for Fixed vs Elastic comparison
BURST_DURATION_SEC = 55.0
BURST_INTERVAL_SEC = 0.30  # ~3.33 req/s (> 1 replica's ~1.48 req/s cap, < 4 replicas' ~5.9 req/s cap)
BURST_MAX_TOKENS = 24
BURST_TOKEN_DELAY_MS = 55.0  # ~1.35s per request


def run_kubectl(args: List[str], timeout: float = 45.0) -> Tuple[int, str, str]:
    cmd = ["kubectl", "-n", NAMESPACE] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 2)


def get_deployment_replicas(name: str = "mock-vllm") -> Dict[str, int]:
    rc, out, _ = run_kubectl(["get", "deployment", name, "-o", "json"])
    if rc != 0 or not out:
        return {"spec": 0, "ready": 0, "available": 0}
    data = json.loads(out)
    spec_r = int(data.get("spec", {}).get("replicas", 0) or 0)
    status = data.get("status", {})
    return {
        "spec": spec_r,
        "ready": int(status.get("readyReplicas", 0) or 0),
        "available": int(status.get("availableReplicas", 0) or 0),
    }


def get_keda_cli_snapshots() -> Dict[str, str]:
    _, hpa_out, hpa_err = run_kubectl(["get", "hpa"])
    _, so_out, so_err = run_kubectl(["get", "scaledobject"])
    _, dep_out, dep_err = run_kubectl(["get", "deployment", "mock-vllm", "prometheus"])
    return {
        "hpa": hpa_out or hpa_err,
        "scaledobject": so_out or so_err,
        "deployment": dep_out or dep_err,
    }


def get_scaledobject_conditions() -> Dict[str, Any]:
    rc, out, _ = run_kubectl(["get", "scaledobject", "mock-vllm-queue-scaler", "-o", "json"])
    if rc != 0 or not out:
        return {"exists": False}
    data = json.loads(out)
    status = data.get("status", {})
    conds = {
        c.get("type"): {"status": c.get("status"), "reason": c.get("reason")}
        for c in (status.get("conditions") or [])
    }
    return {
        "exists": True,
        "hpa_name": status.get("hpaName"),
        "conditions": conds,
    }


def query_prometheus(promql: str) -> Optional[float]:
    """Query in-cluster Prometheus through the K8s API service proxy (no port-forward needed)."""
    raw_path = (
        f"/api/v1/namespaces/{NAMESPACE}/services/prometheus:9090/proxy"
        f"/api/v1/query?query={quote(promql, safe='')}"
    )
    proc = subprocess.run(
        ["kubectl", "get", "--raw", raw_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15.0,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        payload = json.loads(proc.stdout)
        results = payload.get("data", {}).get("result", [])
        if not results:
            return None
        return float(results[0]["value"][1])
    except Exception:
        return None


async def wait_for_clean_single_replica(
    client: httpx.AsyncClient,
    *,
    expect_scaledobject: bool,
    wait_prom_zero: bool = True,
) -> None:
    """Wait until mock-vllm has exactly 1 Ready replica and Gateway/Prometheus are idle."""
    for _ in range(90):
        dep = get_deployment_replicas("mock-vllm")
        try:
            r_adm = await client.get(f"{GATEWAY_URL}/admin/admission")
            adm = r_adm.json()
            r_rdy = await client.get(f"{GATEWAY_URL}/readyz")
            rdy = r_rdy.json()
        except Exception:
            await asyncio.sleep(1.0)
            continue

        prom_val = query_prometheus("max_over_time(gateway_queue_depth[30s])")
        prom_ok = (not wait_prom_zero) or (prom_val is not None and prom_val == 0.0)
        so_ok = True
        if expect_scaledobject:
            so = get_scaledobject_conditions()
            so_ok = (
                so.get("exists", False)
                and so.get("conditions", {}).get("Ready", {}).get("status") == "True"
            )
        if (
            dep["spec"] == 1
            and dep["ready"] == 1
            and rdy.get("available_replicas") == 1
            and adm.get("in_flight", 0) == 0
            and adm.get("queue_depth", 0) == 0
            and prom_ok
            and so_ok
        ):
            return
        await asyncio.sleep(2.0)
    raise RuntimeError(
        f"Timed out waiting for clean 1-replica baseline: dep={get_deployment_replicas('mock-vllm')}"
    )


async def send_one_sse_request(
    client: httpx.AsyncClient,
    req_id: str,
    sent_offset_sec: float,
    max_tokens: int = BURST_MAX_TOKENS,
    token_delay_ms: float = BURST_TOKEN_DELAY_MS,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    try:
        async with client.stream(
            "POST",
            f"{GATEWAY_URL}/v1/chat/completions",
            headers={"X-Request-ID": req_id},
            json={
                "model": "mock-qwen2.5-7b-fp8",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Step 6 KEDA queue-driven autoscaling burst benchmark shared system prompt "
                            "for evaluating Gateway queue depth, P99 latency, and rejection rates."
                        ),
                    },
                    {"role": "user", "content": f"Burst request {req_id}"},
                ],
                "max_tokens": max_tokens,
                "token_delay_ms": token_delay_ms,
                "stream": True,
            },
        ) as resp:
            status = resp.status_code
            routed_replica = resp.headers.get("X-Routed-Replica")
            queue_wait_hdr = resp.headers.get("X-Gateway-Queue-Wait-Ms")
            queue_wait_ms = float(queue_wait_hdr) if queue_wait_hdr else 0.0
            raw_chunks: List[bytes] = []
            async for chunk in resp.aiter_bytes():
                if ttft_ms is None and chunk:
                    ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                raw_chunks.append(chunk)
            body_text = b"".join(raw_chunks).decode("utf-8", errors="ignore")
            done_ok = "data: [DONE]" in body_text and "mid_stream_error" not in body_text
            ok = status == 200 and done_ok
            err_type = None
            if not ok:
                try:
                    err_obj = json.loads(body_text).get("error", {})
                    err_type = err_obj.get("type") or f"http_{status}"
                except Exception:
                    err_type = f"http_{status}"
            return {
                "request_id": req_id,
                "sent_offset_sec": round(sent_offset_sec, 2),
                "ok": ok,
                "status_code": status,
                "error_type": err_type,
                "routed_replica": routed_replica,
                "queue_wait_ms": round(queue_wait_ms, 2),
                "ttft_ms": ttft_ms if ok else None,
                "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
    except Exception as exc:
        return {
            "request_id": req_id,
            "sent_offset_sec": round(sent_offset_sec, 2),
            "ok": False,
            "status_code": 0,
            "error_type": type(exc).__name__,
            "routed_replica": None,
            "queue_wait_ms": 0.0,
            "ttft_ms": None,
            "e2e_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


def summarize_requests(records: List[Dict[str, Any]], label: str = "all") -> Dict[str, Any]:
    total = len(records)
    ok_recs = [r for r in records if r["ok"]]
    rej_503 = [r for r in records if r["status_code"] == 503]
    other_err = [r for r in records if not r["ok"] and r["status_code"] != 503]
    qw = [r["queue_wait_ms"] for r in ok_recs]
    ttfts = [r["ttft_ms"] for r in ok_recs if r["ttft_ms"] is not None]
    e2es = [r["e2e_ms"] for r in ok_recs]
    return {
        "window": label,
        "total_requests": total,
        "succeeded_200": len(ok_recs),
        "rejected_503": len(rej_503),
        "other_errors": len(other_err),
        "success_rate_pct": round(len(ok_recs) / max(1, total) * 100.0, 2),
        "rejection_rate_503_pct": round(len(rej_503) / max(1, total) * 100.0, 2),
        "queue_wait_avg_ms": round(sum(qw) / max(1, len(qw)), 2) if qw else 0.0,
        "queue_wait_p50_ms": percentile(qw, 50),
        "queue_wait_p99_ms": percentile(qw, 99),
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p99_ms": percentile(ttfts, 99),
        "e2e_p50_ms": percentile(e2es, 50),
        "e2e_p99_ms": percentile(e2es, 99),
    }


async def drive_burst_and_observe(
    client: httpx.AsyncClient,
    *,
    prefix: str,
    duration_sec: float,
    interval_sec: float = BURST_INTERVAL_SEC,
    poll_interval_sec: float = 2.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Send fixed-rate burst traffic for `duration_sec` while recording K8s + Gateway timeline."""
    t_start = time.monotonic()
    records: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    burst_done = asyncio.Event()

    async def observer():
        while not burst_done.is_set():
            elapsed = round(time.monotonic() - t_start, 2)
            dep = await asyncio.to_thread(get_deployment_replicas, "mock-vllm")
            prom_q = await asyncio.to_thread(
                query_prometheus, "max_over_time(gateway_queue_depth[30s])"
            )
            try:
                adm = (await client.get(f"{GATEWAY_URL}/admin/admission")).json()
                qd = int(adm.get("queue_depth", 0))
                infl = int(adm.get("in_flight", 0))
                eff_cap = int(adm.get("effective_max_inflight", 0))
            except Exception:
                qd, infl, eff_cap = -1, -1, -1
            timeline.append(
                {
                    "t_sec": elapsed,
                    "gateway_queue_depth": qd,
                    "gateway_in_flight": infl,
                    "gateway_effective_max_inflight": eff_cap,
                    "prom_max_queue_30s": prom_q,
                    "spec_replicas": dep["spec"],
                    "ready_replicas": dep["ready"],
                }
            )
            await asyncio.sleep(poll_interval_sec)

    obs_task = asyncio.create_task(observer())
    pending = set()
    idx = 0
    while True:
        offset = time.monotonic() - t_start
        if offset >= duration_sec:
            break
        idx += 1
        task = asyncio.create_task(
            send_one_sse_request(client, f"{prefix}-{idx:03d}", sent_offset_sec=offset)
        )
        pending.add(task)
        task.add_done_callback(lambda fut: (records.append(fut.result()), pending.discard(fut)))
        await asyncio.sleep(interval_sec)

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    burst_done.set()
    await obs_task
    records.sort(key=lambda r: r["sent_offset_sec"])
    return records, timeline


async def run_exp1_fixed_baseline(client: httpx.AsyncClient) -> Dict[str, Any]:
    print("\n=== [Exp 1] Fixed 1-Replica Baseline (No KEDA Autoscaling) ===")
    run_kubectl(["delete", "scaledobject", "mock-vllm-queue-scaler", "--ignore-not-found"])
    run_kubectl(["set", "env", "deployment/mock-vllm", "STARTUP_DELAY_SEC=3.0"])
    run_kubectl(["scale", "deployment/mock-vllm", "--replicas=1"])
    run_kubectl(["rollout", "status", "deployment/mock-vllm", "--timeout=90s"])
    await wait_for_clean_single_replica(client, expect_scaledobject=False, wait_prom_zero=False)

    cli_before = get_keda_cli_snapshots()
    records, timeline = await drive_burst_and_observe(
        client, prefix="fixed", duration_sec=BURST_DURATION_SEC, poll_interval_sec=2.5
    )
    cli_after = get_keda_cli_snapshots()

    q_vals = [p["gateway_queue_depth"] for p in timeline if p["gateway_queue_depth"] >= 0]
    overall = summarize_requests(records, "all_0_55s")
    early = summarize_requests([r for r in records if r["sent_offset_sec"] < 20.0], "early_0_20s")
    late = summarize_requests([r for r in records if r["sent_offset_sec"] >= 20.0], "late_20_55s")

    result = {
        "experiment": "exp1_fixed_1_replica_baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "duration_sec": BURST_DURATION_SEC,
            "interval_sec": BURST_INTERVAL_SEC,
            "target_rps": round(1.0 / BURST_INTERVAL_SEC, 2),
            "max_tokens": BURST_MAX_TOKENS,
            "token_delay_ms": BURST_TOKEN_DELAY_MS,
        },
        "queue_depth_peak": max(q_vals) if q_vals else 0,
        "queue_depth_avg": round(sum(q_vals) / max(1, len(q_vals)), 2),
        "max_spec_replicas": max(p["spec_replicas"] for p in timeline),
        "max_ready_replicas": max(p["ready_replicas"] for p in timeline),
        "summary_overall": overall,
        "summary_early_0_20s": early,
        "summary_late_20_55s": late,
        "timeline": timeline,
        "kubectl_snapshots": {"before": cli_before, "after": cli_after},
    }
    (OUT_DIR / "exp1_fixed_replica_burst.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Exp 1 done: ok={overall['succeeded_200']}/{overall['total_requests']} "
        f"(503={overall['rejection_rate_503_pct']}%), peak_q={result['queue_depth_peak']}, "
        f"qw_p99={overall['queue_wait_p99_ms']}ms, e2e_p99={overall['e2e_p99_ms']}ms"
    )
    return result


async def run_exp2_autoscaled_and_stabilization(client: httpx.AsyncClient) -> Dict[str, Any]:
    print("\n=== [Exp 2] KEDA Queue-Driven Autoscaling (1 -> 4) & Scale-Down Stabilization ===")
    run_kubectl(["scale", "deployment/mock-vllm", "--replicas=1"])
    rc_so, out_so, err_so = run_kubectl(["apply", "-f", SCALED_OBJECT_YAML])
    print(f"Applied ScaledObject (rc={rc_so}): {out_so or err_so}")
    await wait_for_clean_single_replica(client, expect_scaledobject=True, wait_prom_zero=True)

    cli_snapshots: Dict[str, Any] = {"before_burst": get_keda_cli_snapshots()}

    # Phase 2A: Replay the EXACT same 55s burst!
    records, burst_timeline = await drive_burst_and_observe(
        client, prefix="auto", duration_sec=BURST_DURATION_SEC, poll_interval_sec=2.5
    )
    cli_snapshots["at_burst_end"] = get_keda_cli_snapshots()

    # Phase 2B: Observe Scale-Down Stabilization Window (45s hold) & graceful step-down back to 1!
    print("Burst finished. Observing scaleDown.stabilizationWindowSeconds (45s) and graceful scale-down...")
    t_burst_end = time.monotonic()
    peak_spec = max(p["spec_replicas"] for p in burst_timeline)
    peak_ready = max(p["ready_replicas"] for p in burst_timeline)
    stabilization_timeline: List[Dict[str, Any]] = []
    t_queue_zero: Optional[float] = None
    t_prom_zero: Optional[float] = None
    t_first_scale_down: Optional[float] = None
    t_back_to_min: Optional[float] = None
    captured_hold_snap = False

    for _ in range(45):  # up to ~180s
        await asyncio.sleep(4.0)
        elapsed_post = round(time.monotonic() - t_burst_end, 2)
        dep = get_deployment_replicas("mock-vllm")
        prom_q = query_prometheus("max_over_time(gateway_queue_depth[30s])")
        adm = (await client.get(f"{GATEWAY_URL}/admin/admission")).json()
        qd = int(adm.get("queue_depth", 0))
        eff_cap = int(adm.get("effective_max_inflight", 0))

        if t_queue_zero is None and qd == 0:
            t_queue_zero = elapsed_post
        if t_prom_zero is None and prom_q is not None and prom_q == 0.0:
            t_prom_zero = elapsed_post
        if (
            t_prom_zero is not None
            and not captured_hold_snap
            and elapsed_post - t_prom_zero >= 15.0
            and dep["spec"] >= peak_spec
        ):
            cli_snapshots["during_stabilization_hold"] = get_keda_cli_snapshots()
            captured_hold_snap = True
        if t_first_scale_down is None and dep["spec"] < peak_spec:
            t_first_scale_down = elapsed_post
            cli_snapshots["first_scale_down_step"] = get_keda_cli_snapshots()

        stabilization_timeline.append(
            {
                "post_burst_sec": elapsed_post,
                "gateway_queue_depth": qd,
                "gateway_effective_max_inflight": eff_cap,
                "prom_max_queue_30s": prom_q,
                "spec_replicas": dep["spec"],
                "ready_replicas": dep["ready"],
            }
        )
        if dep["spec"] == 1 and dep["ready"] == 1 and t_prom_zero is not None:
            t_back_to_min = elapsed_post
            break

    cli_snapshots["after_scale_down"] = get_keda_cli_snapshots()

    q_vals = [p["gateway_queue_depth"] for p in burst_timeline if p["gateway_queue_depth"] >= 0]
    first_scale_up_pt = next((p for p in burst_timeline if p["spec_replicas"] > 1), None)
    first_ready_up_pt = next((p for p in burst_timeline if p["ready_replicas"] > 1), None)
    overall = summarize_requests(records, "all_0_55s")
    early = summarize_requests([r for r in records if r["sent_offset_sec"] < 20.0], "early_0_20s")
    late = summarize_requests([r for r in records if r["sent_offset_sec"] >= 20.0], "late_20_55s")

    hold_sec = (
        round(t_first_scale_down - t_prom_zero, 2)
        if (t_first_scale_down is not None and t_prom_zero is not None)
        else None
    )

    result = {
        "experiment": "exp2_keda_autoscaled_burst_and_stabilization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queue_depth_peak": max(q_vals) if q_vals else 0,
        "queue_depth_avg": round(sum(q_vals) / max(1, len(q_vals)), 2),
        "peak_spec_replicas": peak_spec,
        "peak_ready_replicas": peak_ready,
        "t_hpa_first_scale_up_sec": first_scale_up_pt["t_sec"] if first_scale_up_pt else None,
        "t_first_new_pod_ready_sec": first_ready_up_pt["t_sec"] if first_ready_up_pt else None,
        "summary_overall": overall,
        "summary_early_0_20s_control_lag": early,
        "summary_late_20_55s_post_scale": late,
        "scale_down_stabilization": {
            "configured_stabilization_window_sec": 45,
            "t_post_burst_queue_zero_sec": t_queue_zero,
            "t_post_burst_prom_window_zero_sec": t_prom_zero,
            "t_post_burst_first_scale_down_sec": t_first_scale_down,
            "observed_hold_after_prom_zero_sec": hold_sec,
            "t_post_burst_returned_to_min_1_sec": t_back_to_min,
            "timeline": stabilization_timeline,
        },
        "burst_timeline": burst_timeline,
        "kubectl_snapshots": cli_snapshots,
    }
    (OUT_DIR / "exp2_autoscaled_burst_and_stabilization.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Exp 2 done: peak_replicas={peak_spec}/{peak_ready}, "
        f"ok={overall['succeeded_200']}/{overall['total_requests']} "
        f"(503={overall['rejection_rate_503_pct']}%, late_503={late['rejection_rate_503_pct']}%), "
        f"qw_p99={overall['queue_wait_p99_ms']}ms, late_qw_p99={late['queue_wait_p99_ms']}ms, "
        f"stabilization_hold={hold_sec}s, back_to_1={t_back_to_min}s"
    )
    return result


async def run_exp3_slow_cold_start(client: httpx.AsyncClient) -> Dict[str, Any]:
    print("\n=== [Exp 3] Slow Cold Start Amplification (STARTUP_DELAY_SEC=15.0) ===")
    run_kubectl(["set", "env", "deployment/mock-vllm", "STARTUP_DELAY_SEC=15.0"])
    run_kubectl(["rollout", "status", "deployment/mock-vllm", "--timeout=120s"])
    await wait_for_clean_single_replica(client, expect_scaledobject=True, wait_prom_zero=True)

    records, timeline = await drive_burst_and_observe(
        client, prefix="cold", duration_sec=35.0, poll_interval_sec=2.0
    )
    cli_at_end = get_keda_cli_snapshots()

    t_q_trigger = next((p["t_sec"] for p in timeline if p["gateway_queue_depth"] >= 2), None)
    t_hpa_scale = next((p["t_sec"] for p in timeline if p["spec_replicas"] > 1), None)
    t_pod_ready = next((p["t_sec"] for p in timeline if p["ready_replicas"] > 1), None)
    max_spec_ready_gap = max((p["spec_replicas"] - p["ready_replicas"]) for p in timeline)
    cold_gap = (
        round(t_pod_ready - t_hpa_scale, 2)
        if (t_pod_ready is not None and t_hpa_scale is not None)
        else None
    )

    overall = summarize_requests(records, "cold_start_burst_35s")
    result = {
        "experiment": "exp3_slow_cold_start_control_lag",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configured_startup_delay_sec": 15.0,
        "t_queue_crossed_threshold_sec": t_q_trigger,
        "t_hpa_requested_scale_sec": t_hpa_scale,
        "t_first_new_pod_ready_sec": t_pod_ready,
        "cold_start_readiness_lag_sec": cold_gap,
        "max_spec_minus_ready_gap_pods": max_spec_ready_gap,
        "peak_spec_replicas": max(p["spec_replicas"] for p in timeline),
        "peak_ready_replicas": max(p["ready_replicas"] for p in timeline),
        "queue_depth_peak": max(p["gateway_queue_depth"] for p in timeline),
        "summary": overall,
        "timeline": timeline,
        "kubectl_at_burst_end": cli_at_end,
    }
    (OUT_DIR / "exp3_slow_cold_start_control_lag.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Exp 3 done: hpa_scale_at={t_hpa_scale}s, pod_ready_at={t_pod_ready}s "
        f"(cold_lag={cold_gap}s, max_unready_gap={max_spec_ready_gap} pods), "
        f"503={overall['rejection_rate_503_pct']}%"
    )

    # Restore STARTUP_DELAY_SEC=3.0 before Exp 4
    run_kubectl(["set", "env", "deployment/mock-vllm", "STARTUP_DELAY_SEC=3.0"])
    run_kubectl(["rollout", "status", "deployment/mock-vllm", "--timeout=120s"])
    return result


async def run_exp4_metrics_outage_fallback(client: httpx.AsyncClient) -> Dict[str, Any]:
    print("\n=== [Exp 4] Prometheus Outage -> KEDA Fallback (replicas=2) & Recovery ===")
    # Ensure we start cleanly at minReplicaCount=1 (delete & re-apply ScaledObject briefly
    # if still cooling down from Exp 3 so we don't waste 90s waiting for stabilization).
    run_kubectl(["delete", "scaledobject", "mock-vllm-queue-scaler", "--ignore-not-found"])
    run_kubectl(["scale", "deployment/mock-vllm", "--replicas=1"])
    run_kubectl(["apply", "-f", SCALED_OBJECT_YAML])
    await wait_for_clean_single_replica(client, expect_scaledobject=True, wait_prom_zero=True)

    snap_before = get_keda_cli_snapshots()
    cond_before = get_scaledobject_conditions()

    # Inject Prometheus outage!
    print("Scaling Prometheus to 0 replicas to simulate metrics outage...")
    t_outage = time.monotonic()
    run_kubectl(["scale", "deployment/prometheus", "--replicas=0"])

    outage_timeline: List[Dict[str, Any]] = []
    t_fallback_active: Optional[float] = None
    t_scaled_to_fallback_2: Optional[float] = None
    for _ in range(25):  # up to 75s
        await asyncio.sleep(3.0)
        elapsed = round(time.monotonic() - t_outage, 2)
        dep = get_deployment_replicas("mock-vllm")
        so = get_scaledobject_conditions()
        fb_status = so.get("conditions", {}).get("Fallback", {}).get("status")
        ready_status = so.get("conditions", {}).get("Ready", {}).get("status")
        outage_timeline.append(
            {
                "t_sec": elapsed,
                "phase": "prometheus_down",
                "fallback_condition": fb_status,
                "ready_condition": ready_status,
                "spec_replicas": dep["spec"],
                "ready_replicas": dep["ready"],
            }
        )
        if t_fallback_active is None and fb_status == "True":
            t_fallback_active = elapsed
        if t_scaled_to_fallback_2 is None and dep["spec"] == 2 and dep["ready"] == 2:
            t_scaled_to_fallback_2 = elapsed
        if t_fallback_active is not None and t_scaled_to_fallback_2 is not None:
            break

    snap_during_outage = get_keda_cli_snapshots()
    cond_during_outage = get_scaledobject_conditions()

    # Restore Prometheus!
    print("Restoring Prometheus to 1 replica and verifying recovery out of fallback...")
    t_restore = time.monotonic()
    run_kubectl(["scale", "deployment/prometheus", "--replicas=1"])
    run_kubectl(["rollout", "status", "deployment/prometheus", "--timeout=90s"])

    t_recovered_from_fallback: Optional[float] = None
    for _ in range(25):
        await asyncio.sleep(3.0)
        elapsed_r = round(time.monotonic() - t_restore, 2)
        dep = get_deployment_replicas("mock-vllm")
        so = get_scaledobject_conditions()
        fb_status = so.get("conditions", {}).get("Fallback", {}).get("status")
        ready_status = so.get("conditions", {}).get("Ready", {}).get("status")
        outage_timeline.append(
            {
                "t_sec_after_restore": elapsed_r,
                "phase": "prometheus_restored",
                "fallback_condition": fb_status,
                "ready_condition": ready_status,
                "spec_replicas": dep["spec"],
                "ready_replicas": dep["ready"],
            }
        )
        if fb_status == "False" and ready_status == "True":
            t_recovered_from_fallback = elapsed_r
            break

    snap_after_recovery = get_keda_cli_snapshots()
    cond_after_recovery = get_scaledobject_conditions()

    result = {
        "experiment": "exp4_prometheus_outage_keda_fallback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configured_fallback": {"failureThreshold": 2, "replicas": 2},
        "t_fallback_activated_sec": t_fallback_active,
        "t_scaled_to_fallback_replicas_sec": t_scaled_to_fallback_2,
        "t_recovered_after_prometheus_restore_sec": t_recovered_from_fallback,
        "conditions": {
            "before_outage": cond_before,
            "during_outage": cond_during_outage,
            "after_recovery": cond_after_recovery,
        },
        "kubectl_snapshots": {
            "before_outage": snap_before,
            "during_outage": snap_during_outage,
            "after_recovery": snap_after_recovery,
        },
        "timeline": outage_timeline,
    }
    (OUT_DIR / "exp4_metrics_outage_fallback.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Exp 4 done: fallback_active_at={t_fallback_active}s, "
        f"fallback_replicas_2_ready_at={t_scaled_to_fallback_2}s, "
        f"recovered_at={t_recovered_from_fallback}s"
    )
    return result


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=30.0) as client:
        ready = await client.get(f"{GATEWAY_URL}/readyz")
        assert ready.status_code == 200, f"Gateway not ready: {ready.text}"
        print(f"Gateway ready on {GATEWAY_URL}: {ready.json()}")

        prom_ping = query_prometheus("gateway_queue_depth")
        assert prom_ping is not None, "Prometheus is not scraping gateway_queue_depth!"
        print(f"Prometheus scrape verified: gateway_queue_depth={prom_ping}")

        e1 = await run_exp1_fixed_baseline(client)
        e2 = await run_exp2_autoscaled_and_stabilization(client)
        e3 = await run_exp3_slow_cold_start(client)
        e4 = await run_exp4_metrics_outage_fallback(client)

        summary = {
            "cluster": "kind-llm-serving",
            "namespace": NAMESPACE,
            "disclaimer": (
                "Measures Mock Replica control-plane autoscaling (Gateway queue -> Prometheus -> "
                "KEDA -> HPA -> Deployment -> Headless DNS), NOT multi-GPU inference TPS scaling."
            ),
            "fixed_vs_autoscaled_comparison": {
                "fixed_1_replica_overall": {
                    "replicas": 1,
                    "queue_depth_peak": e1["queue_depth_peak"],
                    "queue_depth_avg": e1["queue_depth_avg"],
                    "rejection_rate_503_pct": e1["summary_overall"]["rejection_rate_503_pct"],
                    "queue_wait_p99_ms": e1["summary_overall"]["queue_wait_p99_ms"],
                    "e2e_p99_ms": e1["summary_overall"]["e2e_p99_ms"],
                    "late_window_20_55s_503_pct": e1["summary_late_20_55s"]["rejection_rate_503_pct"],
                    "late_window_20_55s_queue_wait_p99_ms": e1["summary_late_20_55s"]["queue_wait_p99_ms"],
                },
                "autoscaled_1_to_4_overall": {
                    "peak_replicas": e2["peak_ready_replicas"],
                    "queue_depth_peak": e2["queue_depth_peak"],
                    "queue_depth_avg": e2["queue_depth_avg"],
                    "rejection_rate_503_pct": e2["summary_overall"]["rejection_rate_503_pct"],
                    "queue_wait_p99_ms": e2["summary_overall"]["queue_wait_p99_ms"],
                    "e2e_p99_ms": e2["summary_overall"]["e2e_p99_ms"],
                    "early_window_0_20s_control_lag_503_pct": e2["summary_early_0_20s_control_lag"]["rejection_rate_503_pct"],
                    "late_window_20_55s_post_scale_503_pct": e2["summary_late_20_55s_post_scale"]["rejection_rate_503_pct"],
                    "late_window_20_55s_queue_wait_p99_ms": e2["summary_late_20_55s_post_scale"]["queue_wait_p99_ms"],
                },
            },
            "scale_down_stabilization": {
                "configured_window_sec": 45,
                "observed_hold_after_prom_zero_sec": e2["scale_down_stabilization"]["observed_hold_after_prom_zero_sec"],
                "returned_to_min_1_sec": e2["scale_down_stabilization"]["t_post_burst_returned_to_min_1_sec"],
            },
            "slow_cold_start_15s": {
                "hpa_scale_at_sec": e3["t_hpa_requested_scale_sec"],
                "first_new_pod_ready_at_sec": e3["t_first_new_pod_ready_sec"],
                "cold_start_readiness_lag_sec": e3["cold_start_readiness_lag_sec"],
                "max_spec_minus_ready_gap_pods": e3["max_spec_minus_ready_gap_pods"],
                "rejection_rate_503_pct": e3["summary"]["rejection_rate_503_pct"],
            },
            "prometheus_outage_fallback": {
                "fallback_activated_sec": e4["t_fallback_activated_sec"],
                "scaled_to_fallback_2_sec": e4["t_scaled_to_fallback_replicas_sec"],
                "recovered_after_restore_sec": e4["t_recovered_after_prometheus_restore_sec"],
            },
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n=== ALL 4 STEP 6 KEDA AUTOSCALING EXPERIMENTS COMPLETE ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
