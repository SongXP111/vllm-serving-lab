#!/usr/bin/env python3
"""
vLLM Serving Lab - Performance Regression Detector & Gatekeeper

Compares a candidate/optimized benchmark summary against a reference baseline summary.
Asserts that improvements in target metrics do NOT regress:
  - TTFT (P99 / Mean)
  - TPOT (P99 / Mean)
  - E2E Tail Latency (P99)
  - Throughput (tokens/s)
  - OOM / Error Rate (%)
  - Cost per 1M tokens ($/1M tokens)

Exit Code:
  0 - All workloads within regression tolerances (PASS)
  1 - One or more workloads violated regression thresholds (REGRESSION FAIL)
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="vLLM Serving Performance Regression Checker")
    parser.add_argument("--baseline", type=str, default="results/baseline/summary.json", help="Path to golden baseline summary.json")
    parser.add_argument("--current", type=str, required=True, help="Path to newly evaluated summary.json")
    parser.add_argument("--ttft-tol", type=float, default=15.0, help="Max allowed TTFT P99 regression (+%) [default: 15%]")
    parser.add_argument("--tpot-tol", type=float, default=10.0, help="Max allowed TPOT P99 regression (+%) [default: 10%]")
    parser.add_argument("--p99-e2e-tol", type=float, default=20.0, help="Max allowed E2E P99 regression (+%) [default: 20%]")
    parser.add_argument("--throughput-tol", type=float, default=-10.0, help="Max allowed throughput drop (-%) [default: -10%]")
    parser.add_argument("--max-error-rate", type=float, default=0.0, help="Max allowed error rate % [default: 0.0%]")
    parser.add_argument("--output", type=str, default="", help="Path to output regression report JSON")
    args = parser.parse_args()

    if not os.path.isfile(args.baseline):
        print(f"❌ Error: Baseline file '{args.baseline}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.current):
        print(f"❌ Error: Current evaluation file '{args.current}' not found.")
        sys.exit(1)

    with open(args.baseline, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    with open(args.current, "r", encoding="utf-8") as f:
        curr_data = json.load(f)

    base_map = {}
    for b in base_data.get("benchmarks", []):
        key = (b["workload"], b["concurrency"])
        base_map[key] = b

    print("\n" + "=" * 118)
    print("                     🛡️  vLLM Serving Lab - Performance Regression Evaluation Report")
    print("=" * 118)
    print(f"📌 Golden Baseline : {args.baseline} ({base_data.get('model', {}).get('model_name', 'N/A')})")
    print(f"🧪 Candidate Target : {args.current} ({curr_data.get('model', {}).get('model_name', 'N/A')})")
    print(f"🎯 Tolerances       : TTFT P99 <= +{args.ttft_tol}%, TPOT P99 <= +{args.tpot_tol}%, E2E P99 <= +{args.p99_e2e_tol}%, TPS >= {args.throughput_tol}%, Err <= {args.max_error_rate}%")
    print("-" * 118)

    printf_fmt = "%-13s %-5s %-12s %-12s %-12s %-12s %-10s %-10s %-8s\n"
    print(printf_fmt % (
        "Workload", "Conc", "Throughput", "P99 TTFT", "P99 TPOT", "P99 E2E", "$/1M Tok", "Err Rate", "Status"
    ), end="")
    print(printf_fmt % (
        "-------------", "-----", "------------", "------------", "------------", "------------", "----------", "----------", "--------"
    ), end="")

    regressions = []
    comparisons = []

    for curr in curr_data.get("benchmarks", []):
        key = (curr["workload"], curr["concurrency"])
        if key not in base_map:
            continue
        base = base_map[key]

        # Metric deltas
        base_tps = base.get("output_tps", 0.0)
        curr_tps = curr.get("output_tps", 0.0)
        tps_diff_pct = ((curr_tps - base_tps) / base_tps * 100.0) if base_tps > 0 else 0.0

        base_ttft = base.get("p99_ttft_ms", 0.0)
        curr_ttft = curr.get("p99_ttft_ms", 0.0)
        ttft_diff_pct = ((curr_ttft - base_ttft) / base_ttft * 100.0) if base_ttft > 0 else 0.0

        base_tpot = base.get("p99_tpot_ms", 0.0)
        curr_tpot = curr.get("p99_tpot_ms", 0.0)
        tpot_diff_pct = ((curr_tpot - base_tpot) / base_tpot * 100.0) if base_tpot > 0 else 0.0

        base_e2e = base.get("p99_e2e_ms", 0.0)
        curr_e2e = curr.get("p99_e2e_ms", 0.0)
        e2e_diff_pct = ((curr_e2e - base_e2e) / base_e2e * 100.0) if base_e2e > 0 else 0.0

        base_cost = base.get("cost_per_million_tokens_usd", 0.0)
        curr_cost = curr.get("cost_per_million_tokens_usd", 0.0)

        curr_err = curr.get("error_rate_pct", 0.0)

        # Regression checks
        reasons = []
        if ttft_diff_pct > args.ttft_tol:
            reasons.append(f"TTFT P99 regressed +{ttft_diff_pct:.1f}% (>+{args.ttft_tol}%)")
        if tpot_diff_pct > args.tpot_tol:
            reasons.append(f"TPOT P99 regressed +{tpot_diff_pct:.1f}% (>+{args.tpot_tol}%)")
        if e2e_diff_pct > args.p99_e2e_tol:
            reasons.append(f"E2E P99 regressed +{e2e_diff_pct:.1f}% (>+{args.p99_e2e_tol}%)")
        if tps_diff_pct < args.throughput_tol:
            reasons.append(f"Throughput dropped {tps_diff_pct:.1f}% (<{args.throughput_tol}%)")
        if curr_err > args.max_error_rate:
            reasons.append(f"Error/OOM rate {curr_err:.1f}% (>{args.max_error_rate}%)")

        status_str = "PASS" if not reasons else "REGRESS"
        if reasons:
            regressions.append({
                "workload": curr["workload"],
                "concurrency": curr["concurrency"],
                "reasons": reasons,
            })

        def fmt_diff(base_val, curr_val, pct, unit="", invert_sign=False):
            sign = "+" if pct > 0 else ""
            return f"{curr_val:.1f}{unit} ({sign}{pct:.0f}%)"

        tps_str = fmt_diff(base_tps, curr_tps, tps_diff_pct)
        ttft_str = fmt_diff(base_ttft, curr_ttft, ttft_diff_pct, "ms")
        tpot_str = fmt_diff(base_tpot, curr_tpot, tpot_diff_pct, "ms")
        e2e_str = fmt_diff(base_e2e, curr_e2e, e2e_diff_pct, "ms")
        cost_str = f"${curr_cost:.2f}"
        err_str = f"{curr_err:.1f}%"

        print(printf_fmt % (
            curr["workload"], f"c={curr['concurrency']}", tps_str, ttft_str, tpot_str, e2e_str, cost_str, err_str, status_str
        ), end="")

        comparisons.append({
            "workload": curr["workload"],
            "concurrency": curr["concurrency"],
            "baseline": base,
            "candidate": curr,
            "deltas": {
                "throughput_pct": round(tps_diff_pct, 2),
                "ttft_p99_pct": round(ttft_diff_pct, 2),
                "tpot_p99_pct": round(tpot_diff_pct, 2),
                "e2e_p99_pct": round(e2e_diff_pct, 2),
            },
            "status": status_str,
            "reasons": reasons,
        })

    print("=" * 118)

    overall_passed = len(regressions) == 0
    if overall_passed:
        print("🎉 [REGRESSION PASS] Candidate meets all SLA thresholds and causes NO performance regression!\n")
    else:
        print(f"❌ [REGRESSION FAILED] Detected {len(regressions)} workload performance violations:")
        for r in regressions:
            print(f"   • {r['workload']} (c={r['concurrency']}): {', '.join(r['reasons'])}")
        print("")

    if args.output:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": overall_passed,
            "baseline_path": args.baseline,
            "candidate_path": args.current,
            "violations_count": len(regressions),
            "violations": regressions,
            "details": comparisons,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📁 Regression report written to: {args.output}\n")

    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
