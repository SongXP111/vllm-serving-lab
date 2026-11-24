#!/usr/bin/env python3
"""Step 5 Live Kubernetes Fault Injection & Lifecycle Drill Harness.

Runs 3 mandatory fault drills against a real local `kind` cluster running
`llm-gateway` (NodePort 30080) + 3 `mock-vllm` replicas (Headless Service):
1. Drill 1 (`kill_backend`): Hard-kill a backend Pod (`--grace-period=0 --force`)
   during live SSE streaming traffic; record success/failure rate, Endpoint IP
   changes, recovery time to 3 Ready replicas, and Gateway/Pod logs.
2. Drill 2 (`rolling_update`): Execute a live Rolling Update (`APP_VERSION=v2`
   + `kubectl rollout status`) while continuous SSE streaming traffic flows;
   verify `preStop` + `SIGTERM` Graceful Drain achieves near/full zero-drop
   rollout and transitions responses smoothly from `v1` -> `v2`.
3. Drill 3 (`cold_start_and_readiness`):
   - Part A: Increase `STARTUP_DELAY_SEC=12.0` to verify `startupProbe` shields
     slow model loading from `livenessProbe` kills (`restartCount == 0`).
   - Part B: Toggle 1 Pod unready (`/admin/readiness {"ready": false}`) to prove
     `readinessProbe` removes the Pod from Endpoints & Gateway traffic WITHOUT
     restarting the container (`restartCount == 0`), and automatically restores
     it when marked ready again.

Outputs verifiable JSON evidence artifacts into `results/k8s/`.
"""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

NAMESPACE = "llm-serving"
GATEWAY_URL = "http://localhost:30080"
OUT_DIR = Path("results/k8s")


def run_kubectl(args: List[str], timeout: float = 60.0) -> Tuple[int, str, str]:
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


def get_endpoint_ips(service_name: str = "mock-vllm-headless") -> List[str]:
    rc, out, _ = run_kubectl(["get", "endpoints", service_name, "-o", "json"])
    if rc != 0 or not out:
        return []
    data = json.loads(out)
    ips: List[str] = []
    for subset in data.get("subsets") or []:
        for addr in subset.get("addresses") or []:
            ip = addr.get("ip")
            if ip:
                ips.append(ip)
    return sorted(ips)


def get_backend_pods() -> List[Dict[str, Any]]:
    rc, out, _ = run_kubectl(["get", "pods", "-l", "app=mock-vllm", "-o", "json"])
    if rc != 0 or not out:
        return []
    items = json.loads(out).get("items", [])
    pods: List[Dict[str, Any]] = []
    for it in items:
        meta = it.get("metadata", {})
        status = it.get("status", {})
        c_statuses = status.get("containerStatuses") or []
        ready = bool(c_statuses and c_statuses[0].get("ready"))
        restarts = int(c_statuses[0].get("restartCount", 0)) if c_statuses else 0
        pods.append(
            {
                "name": meta.get("name"),
                "phase": status.get("phase"),
                "pod_ip": status.get("podIP"),
                "ready": ready,
                "restarts": restarts,
                "terminating": meta.get("deletionTimestamp") is not None,
            }
        )
    return pods


def get_gateway_logs(tail: int = 25) -> List[str]:
    rc, out, _ = run_kubectl(["logs", "deployment/llm-gateway", f"--tail={tail}"])
    return out.splitlines() if rc == 0 and out else []


async def send_one_sse_request(
    client: httpx.AsyncClient,
    req_id: str,
    max_tokens: int = 10,
    token_delay_ms: float = 35.0,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
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
                            "Kubernetes lifecycle fault drill shared system prompt "
                            "with enough tokens to test prefix routing and graceful drain."
                        ),
                    },
                    {"role": "user", "content": f"Drill request {req_id}"},
                ],
                "max_tokens": max_tokens,
                "token_delay_ms": token_delay_ms,
                "stream": True,
            },
        ) as resp:
            status = resp.status_code
            routed_replica = resp.headers.get("X-Routed-Replica")
            backend_req_id = resp.headers.get("X-Backend-Request-ID")
            raw_chunks: List[bytes] = []
            async for chunk in resp.aiter_bytes():
                raw_chunks.append(chunk)
            body_text = b"".join(raw_chunks).decode("utf-8", errors="ignore")
            mid_stream_err = "mid_stream_error" in body_text
            done_received = "data: [DONE]" in body_text
            ok = status == 200 and done_received and not mid_stream_err
            return {
                "request_id": req_id,
                "ts": round(time.time(), 3),
                "ok": ok,
                "status_code": status,
                "mid_stream_error": mid_stream_err,
                "done_received": done_received,
                "routed_replica": routed_replica,
                "backend_request_id": backend_req_id,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "error": "mid_stream_error" if mid_stream_err else (None if ok else f"http_{status}"),
            }
    except Exception as exc:
        return {
            "request_id": req_id,
            "ts": round(time.time(), 3),
            "ok": False,
            "status_code": 0,
            "mid_stream_error": False,
            "done_received": False,
            "routed_replica": None,
            "backend_request_id": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_drill_1_kill_backend(client: httpx.AsyncClient) -> Dict[str, Any]:
    print("\n=== [Drill 1] Hard-Kill Backend Pod under Live Streaming Traffic ===")
    endpoints_before = get_endpoint_ips()
    pods_before = [
        p
        for p in get_backend_pods()
        if p["ready"] and not p["terminating"] and p["pod_ip"] in endpoints_before
    ]
    assert pods_before, f"No ready non-terminating backend pods found: {get_backend_pods()}"
    target_pod = pods_before[0]["name"]
    target_ip = pods_before[0]["pod_ip"]
    print(f"Initial endpoints: {endpoints_before}; target kill pod: {target_pod} ({target_ip})")

    stop_traffic = asyncio.Event()
    traffic_records: List[Dict[str, Any]] = []
    endpoint_timeline: List[Dict[str, Any]] = [
        {"phase": "before_kill", "t_sec": 0.0, "endpoints": endpoints_before}
    ]

    async def traffic_driver():
        idx = 0
        pending = set()
        while not stop_traffic.is_set():
            idx += 1
            t = asyncio.create_task(
                send_one_sse_request(
                    client, f"drill1-{idx:03d}", max_tokens=10, token_delay_ms=35.0
                )
            )
            pending.add(t)
            t.add_done_callback(lambda fut: (traffic_records.append(fut.result()), pending.discard(fut)))
            await asyncio.sleep(0.25)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    driver_task = asyncio.create_task(traffic_driver())
    await asyncio.sleep(1.5)  # warm steady traffic

    # Inject hard pod kill!
    t_kill = time.monotonic()
    rc, kill_out, kill_err = run_kubectl(
        ["delete", "pod", target_pod, "--grace-period=0", "--force"]
    )
    print(f"Killed {target_pod} (rc={rc}): {kill_out or kill_err}")

    # Poll Endpoints until killed IP disappears and 3 Ready IPs (including replacement) recover
    disappeared_at: Optional[float] = None
    recovered_at: Optional[float] = None
    last_eps = endpoints_before
    for _ in range(90):
        await asyncio.sleep(0.5)
        elapsed = round(time.monotonic() - t_kill, 2)
        eps = get_endpoint_ips()
        if eps != last_eps:
            endpoint_timeline.append({"phase": "transition", "t_sec": elapsed, "endpoints": eps})
            last_eps = eps
        if disappeared_at is None and target_ip not in eps:
            disappeared_at = elapsed
        if disappeared_at is not None and len(eps) == 3 and target_ip not in eps:
            recovered_at = elapsed
            break

    # Run 3s of post-recovery traffic to verify 100% healthy steady state
    t_post_start = time.time()
    await asyncio.sleep(3.0)
    stop_traffic.set()
    await driver_task

    total = len(traffic_records)
    succeeded = sum(1 for r in traffic_records if r["ok"])
    failed = total - succeeded
    post_recs = [r for r in traffic_records if r["ts"] >= t_post_start]
    post_ok = sum(1 for r in post_recs if r["ok"])

    disc_snap = (await client.get(f"{GATEWAY_URL}/admin/discovery")).json()
    result = {
        "drill": "drill_1_kill_backend_pod",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "killed_pod": target_pod,
        "killed_ip": target_ip,
        "endpoints_before": endpoints_before,
        "endpoints_after": get_endpoint_ips(),
        "endpoint_removal_sec": disappeared_at,
        "full_recovery_to_3_ready_sec": recovered_at,
        "endpoint_timeline": endpoint_timeline,
        "traffic_summary": {
            "total_requests": total,
            "succeeded": succeeded,
            "failed": failed,
            "success_rate_pct": round(succeeded / max(1, total) * 100.0, 2),
            "failure_rate_pct": round(failed / max(1, total) * 100.0, 2),
            "post_recovery_total": len(post_recs),
            "post_recovery_success_rate_pct": round(
                post_ok / max(1, len(post_recs)) * 100.0, 2
            ),
            "failed_samples": [r for r in traffic_records if not r["ok"]][:5],
        },
        "gateway_discovery_events": disc_snap.get("events", [])[-6:],
        "gateway_logs_tail": get_gateway_logs(15),
    }
    (OUT_DIR / "drill1_kill_backend.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Drill 1 done: success={succeeded}/{total} ({result['traffic_summary']['success_rate_pct']}%), "
        f"post_recovery={post_ok}/{len(post_recs)}, recovery_sec={recovered_at}s"
    )
    return result


async def run_drill_2_rolling_update(client: httpx.AsyncClient) -> Dict[str, Any]:
    print("\n=== [Drill 2] Rolling Update (v1 -> v2) under Live SSE Streaming Traffic ===")
    endpoints_before = get_endpoint_ips()
    stop_traffic = asyncio.Event()
    traffic_records: List[Dict[str, Any]] = []
    endpoint_timeline: List[Dict[str, Any]] = [
        {"phase": "before_rollout", "t_sec": 0.0, "endpoints": endpoints_before}
    ]

    async def traffic_driver():
        idx = 0
        pending = set()
        while not stop_traffic.is_set():
            idx += 1
            t = asyncio.create_task(
                send_one_sse_request(
                    client, f"drill2-{idx:03d}", max_tokens=12, token_delay_ms=35.0
                )
            )
            pending.add(t)
            t.add_done_callback(lambda fut: (traffic_records.append(fut.result()), pending.discard(fut)))
            await asyncio.sleep(0.25)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    driver_task = asyncio.create_task(traffic_driver())
    await asyncio.sleep(1.5)

    t_rollout = time.monotonic()
    rc, set_out, set_err = run_kubectl(
        ["set", "env", "deployment/mock-vllm", "APP_VERSION=v2"]
    )
    print(f"Triggered rolling update APP_VERSION=v2 (rc={rc}): {set_out or set_err}")

    # Watch endpoints concurrently while `kubectl rollout status` runs in a worker thread
    rollout_done = asyncio.Event()
    rollout_status_output = ""

    async def watch_rollout_status():
        nonlocal rollout_status_output
        rc_s, out_s, err_s = await asyncio.to_thread(
            run_kubectl, ["rollout", "status", "deployment/mock-vllm", "--timeout=120s"], 130.0
        )
        rollout_status_output = out_s or err_s
        rollout_done.set()

    watcher_task = asyncio.create_task(watch_rollout_status())
    last_eps = endpoints_before
    while not rollout_done.is_set():
        await asyncio.sleep(0.75)
        eps = get_endpoint_ips()
        if eps != last_eps:
            endpoint_timeline.append(
                {
                    "phase": "rolling",
                    "t_sec": round(time.monotonic() - t_rollout, 2),
                    "endpoints": eps,
                }
            )
            last_eps = eps
    await watcher_task
    rollout_duration_sec = round(time.monotonic() - t_rollout, 2)

    await asyncio.sleep(2.0)
    stop_traffic.set()
    await driver_task

    total = len(traffic_records)
    succeeded = sum(1 for r in traffic_records if r["ok"])
    failed = total - succeeded
    v1_count = sum(
        1 for r in traffic_records if (r.get("backend_request_id") or "").find("-v1-") >= 0
    )
    v2_count = sum(
        1 for r in traffic_records if (r.get("backend_request_id") or "").find("-v2-") >= 0
    )

    disc_snap = (await client.get(f"{GATEWAY_URL}/admin/discovery")).json()
    result = {
        "drill": "drill_2_rolling_update_zero_drop_drain",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rollout_status_output": rollout_status_output,
        "rollout_duration_sec": rollout_duration_sec,
        "endpoints_before": endpoints_before,
        "endpoints_after": get_endpoint_ips(),
        "endpoint_timeline": endpoint_timeline,
        "traffic_summary": {
            "total_requests": total,
            "succeeded": succeeded,
            "failed": failed,
            "success_rate_pct": round(succeeded / max(1, total) * 100.0, 2),
            "failure_rate_pct": round(failed / max(1, total) * 100.0, 2),
            "served_by_v1_count": v1_count,
            "served_by_v2_count": v2_count,
            "failed_samples": [r for r in traffic_records if not r["ok"]][:5],
        },
        "gateway_discovery_events": disc_snap.get("events", [])[-8:],
        "gateway_logs_tail": get_gateway_logs(15),
    }
    (OUT_DIR / "drill2_rolling_update.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Drill 2 done: success={succeeded}/{total} ({result['traffic_summary']['success_rate_pct']}%), "
        f"v1={v1_count}, v2={v2_count}, rollout_sec={rollout_duration_sec}s"
    )
    return result


async def run_drill_3_cold_start_and_readiness(client: httpx.AsyncClient) -> Dict[str, Any]:
    print("\n=== [Drill 3] Cold Start (startupProbe) & Readiness Isolation (readinessProbe) ===")

    # Part 3A: Slow Cold Start (12s) -> startupProbe protects from livenessProbe kills!
    t_cold = time.monotonic()
    run_kubectl(["set", "env", "deployment/mock-vllm", "STARTUP_DELAY_SEC=12.0"])
    cold_observations: List[Dict[str, Any]] = []
    rollout_done = asyncio.Event()

    async def wait_cold_rollout():
        await asyncio.to_thread(
            run_kubectl, ["rollout", "status", "deployment/mock-vllm", "--timeout=180s"], 190.0
        )
        rollout_done.set()

    w_task = asyncio.create_task(wait_cold_rollout())
    while not rollout_done.is_set():
        await asyncio.sleep(2.0)
        pods_now = get_backend_pods()
        unready_loading = [p for p in pods_now if not p["ready"] and not p["terminating"]]
        cold_observations.append(
            {
                "t_sec": round(time.monotonic() - t_cold, 2),
                "ready_endpoints_count": len(get_endpoint_ips()),
                "loading_unready_pods": [
                    {"name": p["name"], "restarts": p["restarts"]} for p in unready_loading
                ],
            }
        )
    await w_task
    cold_rollout_sec = round(time.monotonic() - t_cold, 2)

    # Wait briefly for any old Terminating pods from Part 3A to exit cleanly,
    # then select an ACTIVE Ready Pod whose IP is actually present in Endpoints!
    ready_pods: List[Dict[str, Any]] = []
    eps_before_unready: List[str] = []
    for _ in range(30):
        eps_before_unready = get_endpoint_ips()
        pods_now = get_backend_pods()
        ready_pods = [
            p
            for p in pods_now
            if p["ready"] and not p["terminating"] and p["pod_ip"] in eps_before_unready
        ]
        terminating_pods = [p for p in pods_now if p["terminating"]]
        if len(ready_pods) == 3 and not terminating_pods:
            break
        await asyncio.sleep(0.5)

    assert ready_pods, f"No ready pods found after Part 3A: {get_backend_pods()}"
    max_restarts_during_cold_start = max(p["restarts"] for p in ready_pods)

    # Part 3B: Force 1 Active Ready Pod unready via `/admin/readiness` -> prove
    # Readiness removes Pod from Endpoints WITHOUT restarting container (`restarts == 0`),
    # then restores it cleanly back into Endpoints!
    target_pod_info = ready_pods[0]
    target_pod = target_pod_info["name"]
    target_ip = target_pod_info["pod_ip"]
    restarts_before = target_pod_info["restarts"]
    assert target_ip in eps_before_unready, (
        f"Selected pod IP {target_ip} not in endpoints {eps_before_unready}"
    )

    print(f"Toggling {target_pod} ({target_ip}) readiness -> False via kubectl exec...")
    t_unready = time.monotonic()
    rc_off, out_off, err_off = run_kubectl(
        [
            "exec",
            target_pod,
            "--",
            "python",
            "-c",
            "import urllib.request; req = urllib.request.Request('http://127.0.0.1:8000/admin/readiness', data=b'{\"ready\":false}', headers={'Content-Type':'application/json'}, method='POST'); print(urllib.request.urlopen(req).read().decode())",
        ]
    )
    print(f"Readiness OFF response (rc={rc_off}): {out_off or err_off}")

    removed_from_endpoints_sec: Optional[float] = None
    for _ in range(30):
        await asyncio.sleep(0.5)
        eps = get_endpoint_ips()
        if target_ip not in eps:
            removed_from_endpoints_sec = round(time.monotonic() - t_unready, 2)
            break

    # Wait 2s for Gateway DNS discovery to sync, then send 12 requests:
    # 100% must succeed and 0 must route to the unready pod!
    await asyncio.sleep(2.0)
    eps_during_unready = get_endpoint_ips()
    unready_window_reqs = [
        await send_one_sse_request(client, f"drill3-unready-{i:02d}", max_tokens=6, token_delay_ms=20.0)
        for i in range(12)
    ]
    unready_pod_hits = sum(
        1 for r in unready_window_reqs if target_pod in (r.get("backend_request_id") or "")
    )
    unready_window_ok = sum(1 for r in unready_window_reqs if r["ok"])

    # Verify Pod was NOT restarted by Kubelet!
    pod_during_unready = next(
        (p for p in get_backend_pods() if p["name"] == target_pod), {}
    )
    restarts_while_unready = pod_during_unready.get("restarts", -1)

    # Restore readiness -> True and measure recovery back into Endpoints!
    print(f"Restoring {target_pod} readiness -> True...")
    t_restore = time.monotonic()
    rc_on, out_on, err_on = run_kubectl(
        [
            "exec",
            target_pod,
            "--",
            "python",
            "-c",
            "import urllib.request; req = urllib.request.Request('http://127.0.0.1:8000/admin/readiness', data=b'{\"ready\":true}', headers={'Content-Type':'application/json'}, method='POST'); print(urllib.request.urlopen(req).read().decode())",
        ]
    )
    print(f"Readiness ON response (rc={rc_on}): {out_on or err_on}")
    restored_to_endpoints_sec: Optional[float] = None
    for _ in range(30):
        await asyncio.sleep(0.5)
        eps = get_endpoint_ips()
        if target_ip in eps and len(eps) == 3:
            restored_to_endpoints_sec = round(time.monotonic() - t_restore, 2)
            break

    eps_after_restore = get_endpoint_ips()

    # Reset STARTUP_DELAY_SEC back to 3.0s and APP_VERSION back to v1 for tidy cluster state
    run_kubectl(["set", "env", "deployment/mock-vllm", "STARTUP_DELAY_SEC=3.0", "APP_VERSION=v1"])

    result = {
        "drill": "drill_3_cold_start_and_readiness_isolation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "part_3a_cold_start_startup_probe": {
            "configured_startup_delay_sec": 12.0,
            "cold_rollout_duration_sec": cold_rollout_sec,
            "max_pod_restarts_during_12s_cold_start": max_restarts_during_cold_start,
            "observed_unready_loading_phase": any(
                len(o["loading_unready_pods"]) > 0 for o in cold_observations
            ),
            "timeline_samples": cold_observations[:8],
        },
        "part_3b_readiness_probe_isolation": {
            "target_pod": target_pod,
            "target_ip": target_ip,
            "endpoints_before": eps_before_unready,
            "endpoints_during_unready": eps_during_unready,
            "endpoints_after_restore": eps_after_restore,
            "removed_from_endpoints_sec": removed_from_endpoints_sec,
            "restored_to_endpoints_sec": restored_to_endpoints_sec,
            "container_restarts_before": restarts_before,
            "container_restarts_while_unready": restarts_while_unready,
            "readiness_never_restarted_container": restarts_while_unready == restarts_before,
            "requests_during_unready_total": len(unready_window_reqs),
            "requests_during_unready_succeeded": unready_window_ok,
            "requests_routed_to_unready_pod": unready_pod_hits,
        },
        "gateway_logs_tail": get_gateway_logs(15),
    }
    (OUT_DIR / "drill3_cold_start_and_readiness.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Drill 3 done: cold_start_restarts={max_restarts_during_cold_start}, "
        f"removed_sec={removed_from_endpoints_sec}s, unready_pod_hits={unready_pod_hits}, "
        f"restored_sec={restored_to_endpoints_sec}s"
    )
    return result


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Verify Gateway is reachable and sees 3 ready pods
        ready = await client.get(f"{GATEWAY_URL}/readyz")
        assert ready.status_code == 200, f"Gateway not ready: {ready.text}"
        print(f"Gateway ready on {GATEWAY_URL}: {ready.json()}")

        d1 = await run_drill_1_kill_backend(client)
        d2 = await run_drill_2_rolling_update(client)
        d3 = await run_drill_3_cold_start_and_readiness(client)

        summary = {
            "cluster": "kind-llm-serving",
            "namespace": NAMESPACE,
            "drill1_kill_backend_summary": {
                "success_rate_pct": d1["traffic_summary"]["success_rate_pct"],
                "post_recovery_success_rate_pct": d1["traffic_summary"][
                    "post_recovery_success_rate_pct"
                ],
                "recovery_sec": d1["full_recovery_to_3_ready_sec"],
            },
            "drill2_rolling_update_summary": {
                "success_rate_pct": d2["traffic_summary"]["success_rate_pct"],
                "served_by_v1": d2["traffic_summary"]["served_by_v1_count"],
                "served_by_v2": d2["traffic_summary"]["served_by_v2_count"],
                "rollout_sec": d2["rollout_duration_sec"],
            },
            "drill3_cold_start_and_readiness_summary": {
                "cold_start_restarts": d3["part_3a_cold_start_startup_probe"][
                    "max_pod_restarts_during_12s_cold_start"
                ],
                "readiness_never_restarted_container": d3[
                    "part_3b_readiness_probe_isolation"
                ]["readiness_never_restarted_container"],
                "requests_routed_to_unready_pod": d3[
                    "part_3b_readiness_probe_isolation"
                ]["requests_routed_to_unready_pod"],
                "removed_sec": d3["part_3b_readiness_probe_isolation"][
                    "removed_from_endpoints_sec"
                ],
                "restored_sec": d3["part_3b_readiness_probe_isolation"][
                    "restored_to_endpoints_sec"
                ],
            },
        }
        (OUT_DIR / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("\n=== ALL 3 K8S FAULT DRILLS COMPLETE ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
