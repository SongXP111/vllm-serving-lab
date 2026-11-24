"""Multi-Instance Router with Filter -> Score -> Pick pipeline (M2).

Implements four switchable routing policies:
1. `random`           : Uniform random selection among filtered healthy replicas.
2. `least-load`       : Minimize combined load (`inflight + backend_waiting_queue`
                        normalized by `max_inflight`, adjusted for KV pressure).
3. `prefix-aware`     : Maximize estimated token-prefix affinity (chained block hash
                        matching vLLM APC), breaking cold ties by least load.
4. `cache+load-aware` : Jointly weights estimated token-prefix affinity and replica
                        load/KV headroom, with hotspot damping so a single warm
                        replica is never overwhelmed under skewed traffic.

IMPORTANT SEMANTIC CONTRACT:
The prefix index in this module (`EstimatedPrefixCacheIndex`) is a Gateway-side
*estimate* of KV prefix affinity derived from routing history + bounded LRU.
It is NEVER reported or treated as verified GPU KV cache hits. Ground-truth
cache hit verification must come from real vLLM Prometheus `/metrics` synced
into `ReplicaRegistry.update_from_vllm_metrics(...)`.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import random
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from gateway.registry import ReplicaRegistry, ReplicaState


class RoutingPolicy(str, Enum):
    """Switchable routing policies supported by the M2 InstanceRouter."""

    RANDOM = "random"
    LEAST_LOAD = "least-load"
    PREFIX_AWARE = "prefix-aware"
    CACHE_LOAD_AWARE = "cache+load-aware"

    @classmethod
    def from_str(cls, raw: Optional[str]) -> "RoutingPolicy":
        if not raw:
            return cls.LEAST_LOAD
        normalized = raw.strip().lower().replace("_", "-")
        aliases = {
            "random": cls.RANDOM,
            "least-load": cls.LEAST_LOAD,
            "leastload": cls.LEAST_LOAD,
            "prefix-aware": cls.PREFIX_AWARE,
            "prefix": cls.PREFIX_AWARE,
            "cache+load-aware": cls.CACHE_LOAD_AWARE,
            "cache-load-aware": cls.CACHE_LOAD_AWARE,
            "cache-and-load": cls.CACHE_LOAD_AWARE,
            "hybrid": cls.CACHE_LOAD_AWARE,
        }
        if normalized not in aliases:
            valid = ", ".join(p.value for p in cls)
            raise ValueError(f"Unknown routing policy '{raw}'. Valid policies: {valid}")
        return aliases[normalized]


class NoHealthyReplicaError(Exception):
    """Raised when Filter stage eliminates all replicas (no healthy backend)."""


# Lightweight regex pseudo-tokenizer (word/number/CJK char/punctuation) so the
# Gateway hot path computes token-prefix blocks without loading a heavy tokenizer.
_PSEUDO_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s\w]", re.UNICODE)


def extract_prompt_text(body: Dict[str, Any]) -> str:
    """Extract canonical prompt text from an OpenAI chat/completions request body."""
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            content = msg.get("content", "")
            if isinstance(content, list):
                # OpenAI multimodal content blocks
                text_chunks = [
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content_str = "".join(text_chunks)
            else:
                content_str = str(content)
            parts.append(f"<|{role}|>{content_str}")
        return "\n".join(parts)

    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(p) for p in prompt)
    return ""


def compute_prefix_block_hashes(
    prompt_text: str,
    *,
    block_size: int = 16,
    namespace: str = "",
) -> List[str]:
    """Compute vLLM-style chained rolling block hashes over pseudo-tokens.

    Mirrors vLLM Automatic Prefix Caching:
      hash(B_0) = H(namespace || tokens[0:block_size])
      hash(B_k) = H(hash(B_{k-1}) || tokens[k*block_size : (k+1)*block_size])

    Any trailing partial block with fewer than `block_size` tokens is excluded,
    matching vLLM's full-block APC indexing semantics.
    """
    if not prompt_text or block_size <= 0:
        return []
    tokens = _PSEUDO_TOKEN_RE.findall(prompt_text)
    num_full_blocks = len(tokens) // block_size
    if num_full_blocks == 0:
        return []

    hashes: List[str] = []
    parent_hash = hashlib.blake2b(namespace.encode("utf-8"), digest_size=8).hexdigest()
    for idx in range(num_full_blocks):
        block_tokens = tokens[idx * block_size : (idx + 1) * block_size]
        payload = f"{parent_hash}|{' '.join(block_tokens)}".encode("utf-8")
        block_hash = hashlib.blake2b(payload, digest_size=8).hexdigest()
        hashes.append(block_hash)
        parent_hash = block_hash
    return hashes


class EstimatedPrefixCacheIndex:
    """Gateway-side bounded LRU estimator of per-replica KV prefix affinity.

    Tracks which chained token-prefix block hashes were recently routed to each
    replica. Evicts oldest blocks once a replica's tracked block count exceeds
    its capacity hint, approximating vLLM's finite GPU KV cache LRU eviction.

    NOTE: This produces *estimated* affinity only — never treat it as verified
    vLLM GPU cache hit counts.
    """

    def __init__(self, default_max_blocks_per_replica: int = 1024) -> None:
        self.default_max_blocks = max(1, default_max_blocks_per_replica)
        self._replica_lru: Dict[str, "OrderedDict[str, float]"] = {}

    def _get_lru(self, replica_id: str) -> "OrderedDict[str, float]":
        lru = self._replica_lru.get(replica_id)
        if lru is None:
            lru = OrderedDict()
            self._replica_lru[replica_id] = lru
        return lru

    def estimate_affinity(
        self, replica_id: str, block_hashes: Sequence[str]
    ) -> Tuple[int, float]:
        """Return `(estimated_matched_blocks, estimated_affinity_ratio)` in [0.0, 1.0].

        Counts the longest contiguous prefix match starting from block 0.
        """
        total = len(block_hashes)
        if total == 0:
            return 0, 0.0
        lru = self._replica_lru.get(replica_id)
        if not lru:
            return 0, 0.0

        matched = 0
        for h in block_hashes:
            if h in lru:
                matched += 1
            else:
                break
        return matched, round(matched / total, 4)

    def record_routed_prefix(
        self,
        replica_id: str,
        block_hashes: Sequence[str],
        *,
        max_blocks: Optional[int] = None,
    ) -> None:
        """Record that `block_hashes` were routed to `replica_id` and apply LRU cap."""
        if not block_hashes:
            return
        cap = max(1, max_blocks if max_blocks is not None else self.default_max_blocks)
        lru = self._get_lru(replica_id)
        now = time.monotonic()
        # Insert/refresh from leaf -> root or root -> leaf; refreshing in reverse
        # order keeps earlier foundation prefix blocks fresher than leaf blocks
        # when evicting from the oldest end (mirroring vLLM leaf-first tie-breaking).
        for h in reversed(block_hashes):
            lru[h] = now
            lru.move_to_end(h)
        while len(lru) > cap:
            lru.popitem(last=False)

    def tracked_blocks_count(self, replica_id: str) -> int:
        lru = self._replica_lru.get(replica_id)
        return len(lru) if lru is not None else 0

    def clear_replica(self, replica_id: str) -> None:
        self._replica_lru.pop(replica_id, None)


@dataclass
class RoutingContext:
    """Per-request inputs fed through the Filter -> Score -> Pick pipeline."""

    request_id: str
    model: str
    prompt_text: str
    block_hashes: List[str]
    policy: RoutingPolicy


@dataclass
class RoutingDecision:
    """Structured outcome of a single Filter -> Score -> Pick scheduling cycle."""

    replica: ReplicaState
    policy: RoutingPolicy
    total_prefix_blocks: int
    estimated_matched_blocks: int
    estimated_prefix_affinity: float
    candidate_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    filtered_out: Dict[str, str] = field(default_factory=dict)
    capacity_fallback_used: bool = False

    def to_summary(self) -> Dict[str, Any]:
        return {
            "replica_id": self.replica.replica_id,
            "endpoint": self.replica.endpoint,
            "policy": self.policy.value,
            "total_prefix_blocks": self.total_prefix_blocks,
            "estimated_matched_blocks": self.estimated_matched_blocks,
            "estimated_prefix_affinity": self.estimated_prefix_affinity,
            "capacity_fallback_used": self.capacity_fallback_used,
            "candidate_scores": self.candidate_scores,
            "filtered_out": self.filtered_out,
        }


# =============================================================================
# Filter -> Score -> Pick Plugin Interfaces & Concrete Implementations
# =============================================================================


class FilterPlugin(Protocol):
    name: str

    def filter(
        self, replicas: Sequence[ReplicaState], rctx: RoutingContext
    ) -> Tuple[List[ReplicaState], Dict[str, str]]: ...


class HealthFilter:
    """Eliminates unhealthy or circuit-broken replicas."""

    name = "health-filter"

    def filter(
        self, replicas: Sequence[ReplicaState], rctx: RoutingContext
    ) -> Tuple[List[ReplicaState], Dict[str, str]]:
        now = time.monotonic()
        kept: List[ReplicaState] = []
        dropped: Dict[str, str] = {}
        for r in replicas:
            if r.is_available(now):
                kept.append(r)
            else:
                reason = "unhealthy" if not r.healthy else "circuit_open"
                dropped[r.replica_id] = reason
        return kept, dropped


class CapacityFilter:
    """Eliminates replicas that hit `max_inflight` or critical KV pressure."""

    name = "capacity-filter"

    def __init__(self, kv_overload_threshold: float = 0.95) -> None:
        self.kv_overload_threshold = kv_overload_threshold

    def filter(
        self, replicas: Sequence[ReplicaState], rctx: RoutingContext
    ) -> Tuple[List[ReplicaState], Dict[str, str]]:
        kept: List[ReplicaState] = []
        dropped: Dict[str, str] = {}
        for r in replicas:
            if r.inflight >= r.max_inflight:
                dropped[r.replica_id] = (
                    f"inflight_saturated({r.inflight}/{r.max_inflight})"
                )
            elif r.kv_cache_usage_perc >= self.kv_overload_threshold:
                dropped[r.replica_id] = (
                    f"kv_cache_overloaded({r.kv_cache_usage_perc:.2f})"
                )
            else:
                kept.append(r)
        return kept, dropped


class LoadScorer:
    """Normalized [0.0, 1.0] headroom score based on inflight, waiting queue, and KV usage."""

    name = "load-scorer"

    def score(
        self, candidates: Sequence[ReplicaState], rctx: RoutingContext
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for r in candidates:
            # Soft saturation curve so load_ratio > 1.0 during fallback still ranks
            # less-overloaded replicas ahead of heavily-overloaded ones.
            ratio = r.load_ratio()
            if ratio <= 1.0:
                queue_headroom = 1.0 - ratio
            else:
                queue_headroom = 0.0 / ratio  # 0.0 when saturated
                # Preserve slight ordering via inverse load under fallback
                queue_headroom = max(0.0, 0.05 / ratio)
            kv_headroom = 1.0 - 0.5 * max(0.0, min(1.0, r.kv_cache_usage_perc))
            scores[r.replica_id] = round(max(0.0, min(1.0, queue_headroom * kv_headroom)), 4)
        return scores


class EstimatedPrefixAffinityScorer:
    """Normalized [0.0, 1.0] score based on estimated chained token-prefix matches."""

    name = "estimated-prefix-affinity-scorer"

    def __init__(self, prefix_index: EstimatedPrefixCacheIndex) -> None:
        self.prefix_index = prefix_index

    def score_with_details(
        self, candidates: Sequence[ReplicaState], rctx: RoutingContext
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        scores: Dict[str, float] = {}
        matched_counts: Dict[str, int] = {}
        for r in candidates:
            matched, ratio = self.prefix_index.estimate_affinity(
                r.replica_id, rctx.block_hashes
            )
            scores[r.replica_id] = ratio
            matched_counts[r.replica_id] = matched
        return scores, matched_counts


@dataclass
class RoutingStats:
    """Observability metrics for routing decisions and estimated prefix affinity."""

    decisions_total: int = 0
    no_healthy_replica_errors: int = 0
    capacity_fallbacks: int = 0
    decisions_by_policy: Dict[str, int] = field(default_factory=dict)
    decisions_by_replica: Dict[str, int] = field(default_factory=dict)
    # Explicitly labeled as ESTIMATED prefix affinity (not real vLLM cache hits!)
    estimated_affinity_sum: float = 0.0
    estimated_affinity_evaluated_requests: int = 0
    estimated_affinity_nonzero_requests: int = 0
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)


class InstanceRouter:
    """Orchestrates Filter -> Score -> Pick across multiple vLLM replicas."""

    def __init__(
        self,
        registry: ReplicaRegistry,
        *,
        policy: RoutingPolicy = RoutingPolicy.LEAST_LOAD,
        prefix_block_size: int = 16,
        prefix_weight: float = 0.55,
        load_weight: float = 0.45,
        hotspot_load_threshold: float = 0.75,
        kv_hotspot_threshold: float = 0.85,
        kv_overload_threshold: float = 0.95,
        default_prefix_cache_blocks: int = 1024,
        rng_seed: Optional[int] = None,
        recent_decisions_limit: int = 200,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.prefix_block_size = max(1, prefix_block_size)
        self.prefix_weight = max(0.0, prefix_weight)
        self.load_weight = max(0.0, load_weight)
        self.hotspot_load_threshold = hotspot_load_threshold
        self.kv_hotspot_threshold = kv_hotspot_threshold
        self.recent_decisions_limit = max(10, recent_decisions_limit)

        self._rng = random.Random(rng_seed)
        self.prefix_index = EstimatedPrefixCacheIndex(
            default_max_blocks_per_replica=default_prefix_cache_blocks
        )
        self.health_filter = HealthFilter()
        self.capacity_filter = CapacityFilter(
            kv_overload_threshold=kv_overload_threshold
        )
        self.load_scorer = LoadScorer()
        self.prefix_scorer = EstimatedPrefixAffinityScorer(self.prefix_index)
        self.stats = RoutingStats()

    def set_policy(self, policy: RoutingPolicy | str) -> RoutingPolicy:
        resolved = (
            policy
            if isinstance(policy, RoutingPolicy)
            else RoutingPolicy.from_str(policy)
        )
        self.policy = resolved
        return self.policy

    def select_replica(
        self,
        *,
        request_id: str,
        body: Dict[str, Any],
        policy_override: Optional[str] = None,
    ) -> RoutingDecision:
        """Run Filter -> Score -> Pick to choose the target backend replica."""
        active_policy = (
            RoutingPolicy.from_str(policy_override)
            if policy_override
            else self.policy
        )
        model = str(body.get("model", ""))
        prompt_text = extract_prompt_text(body)
        block_hashes = compute_prefix_block_hashes(
            prompt_text,
            block_size=self.prefix_block_size,
            namespace=model,
        )
        rctx = RoutingContext(
            request_id=request_id,
            model=model,
            prompt_text=prompt_text,
            block_hashes=block_hashes,
            policy=active_policy,
        )

        all_replicas = self.registry.all_replicas()
        filtered_out: Dict[str, str] = {}

        # Stage 1A: Health Filter (hard requirement)
        healthy_candidates, health_dropped = self.health_filter.filter(
            all_replicas, rctx
        )
        filtered_out.update(health_dropped)
        if not healthy_candidates:
            self.stats.no_healthy_replica_errors += 1
            raise NoHealthyReplicaError(
                f"No healthy backend replicas available (total={len(all_replicas)})"
            )

        # Stage 1B: Capacity Filter (soft fallback to healthy candidates if all saturated)
        candidates, cap_dropped = self.capacity_filter.filter(healthy_candidates, rctx)
        capacity_fallback_used = False
        if candidates:
            filtered_out.update(cap_dropped)
        else:
            # All healthy replicas are at or above capacity hints; fall back to
            # healthy set so admitted requests can still be scored & routed.
            candidates = healthy_candidates
            capacity_fallback_used = True
            self.stats.capacity_fallbacks += 1

        # Stage 2: Score every surviving candidate on both Load & Estimated Prefix Affinity
        load_scores = self.load_scorer.score(candidates, rctx)
        prefix_scores, matched_blocks = self.prefix_scorer.score_with_details(
            candidates, rctx
        )

        candidate_scores: Dict[str, Dict[str, float]] = {}
        combined_scores: Dict[str, float] = {}

        for r in candidates:
            rid = r.replica_id
            l_score = load_scores[rid]
            p_score = prefix_scores[rid]

            if active_policy == RoutingPolicy.RANDOM:
                total = 1.0
            elif active_policy == RoutingPolicy.LEAST_LOAD:
                total = l_score
            elif active_policy == RoutingPolicy.PREFIX_AWARE:
                total = p_score
            else:
                # `cache+load-aware`: jointly weight estimated prefix affinity
                # and load/KV headroom, attenuating affinity when a replica enters
                # hotspot territory so traffic spills over before saturation.
                eff_prefix = p_score
                load_ratio = r.load_ratio()
                if (
                    load_ratio >= self.hotspot_load_threshold
                    or r.kv_cache_usage_perc >= self.kv_hotspot_threshold
                ):
                    pressure = max(load_ratio, r.kv_cache_usage_perc)
                    damping = max(0.0, 1.0 - pressure)
                    eff_prefix = p_score * damping
                weight_sum = self.prefix_weight + self.load_weight
                if weight_sum > 0:
                    total = (
                        self.prefix_weight * eff_prefix + self.load_weight * l_score
                    ) / weight_sum
                else:
                    total = l_score

            total = round(total, 6)
            combined_scores[rid] = total
            candidate_scores[rid] = {
                "load_score": l_score,
                "estimated_prefix_affinity": p_score,
                "combined_score": total,
            }

        # Stage 3: Pick
        if active_policy == RoutingPolicy.RANDOM:
            chosen = self._rng.choice(candidates)
        else:
            max_score = max(combined_scores.values())
            top_candidates = [
                r
                for r in candidates
                if abs(combined_scores[r.replica_id] - max_score) <= 1e-9
            ]
            if len(top_candidates) == 1:
                chosen = top_candidates[0]
            else:
                # Tie-breaker: prefer lowest effective load first (spreads cold
                # prompts across idle replicas cleanly), then fewest tracked
                # prefix blocks (coldest cache, like llm-d `no-hit-lru`), then RNG.
                min_load = min(r.effective_queue_load() for r in top_candidates)
                least_loaded_ties = [
                    r for r in top_candidates if r.effective_queue_load() == min_load
                ]
                if len(least_loaded_ties) > 1 and active_policy in (
                    RoutingPolicy.PREFIX_AWARE,
                    RoutingPolicy.CACHE_LOAD_AWARE,
                ):
                    min_blocks = min(
                        self.prefix_index.tracked_blocks_count(r.replica_id)
                        for r in least_loaded_ties
                    )
                    least_loaded_ties = [
                        r
                        for r in least_loaded_ties
                        if self.prefix_index.tracked_blocks_count(r.replica_id)
                        == min_blocks
                    ]
                chosen = self._rng.choice(least_loaded_ties)

        chosen_affinity = prefix_scores[chosen.replica_id]
        chosen_matched = matched_blocks[chosen.replica_id]

        # Record estimated prefix affinity on chosen replica (for future routing estimates)
        if block_hashes:
            self.prefix_index.record_routed_prefix(
                chosen.replica_id,
                block_hashes,
                max_blocks=chosen.kv_cache_capacity_blocks,
            )

        decision = RoutingDecision(
            replica=chosen,
            policy=active_policy,
            total_prefix_blocks=len(block_hashes),
            estimated_matched_blocks=chosen_matched,
            estimated_prefix_affinity=chosen_affinity,
            candidate_scores=candidate_scores,
            filtered_out=filtered_out,
            capacity_fallback_used=capacity_fallback_used,
        )
        self._record_stats(decision)
        return decision

    def _record_stats(self, decision: RoutingDecision) -> None:
        self.stats.decisions_total += 1
        p_key = decision.policy.value
        r_key = decision.replica.replica_id
        self.stats.decisions_by_policy[p_key] = (
            self.stats.decisions_by_policy.get(p_key, 0) + 1
        )
        self.stats.decisions_by_replica[r_key] = (
            self.stats.decisions_by_replica.get(r_key, 0) + 1
        )
        if decision.total_prefix_blocks > 0:
            self.stats.estimated_affinity_evaluated_requests += 1
            self.stats.estimated_affinity_sum += decision.estimated_prefix_affinity
            if decision.estimated_prefix_affinity > 0.0:
                self.stats.estimated_affinity_nonzero_requests += 1

        self.stats.recent_decisions.append(decision.to_summary())
        if len(self.stats.recent_decisions) > self.recent_decisions_limit:
            self.stats.recent_decisions.pop(0)

    def snapshot(self) -> Dict[str, Any]:
        eval_cnt = self.stats.estimated_affinity_evaluated_requests
        avg_est_affinity = (
            round(self.stats.estimated_affinity_sum / eval_cnt, 4)
            if eval_cnt > 0
            else 0.0
        )
        est_hit_req_ratio = (
            round(self.stats.estimated_affinity_nonzero_requests / eval_cnt, 4)
            if eval_cnt > 0
            else 0.0
        )
        return {
            "active_policy": self.policy.value,
            "supported_policies": [p.value for p in RoutingPolicy],
            "weights": {
                "prefix_weight": self.prefix_weight,
                "load_weight": self.load_weight,
                "hotspot_load_threshold": self.hotspot_load_threshold,
                "kv_hotspot_threshold": self.kv_hotspot_threshold,
            },
            "prefix_estimator": {
                "block_size_tokens": self.prefix_block_size,
                "is_estimated_only": True,
                "note": (
                    "Gateway prefix affinity is an estimate from routing history; "
                    "validate real KV cache hits via ReplicaRegistry observed vLLM metrics."
                ),
                "tracked_blocks_by_replica": {
                    r.replica_id: self.prefix_index.tracked_blocks_count(r.replica_id)
                    for r in self.registry.all_replicas()
                },
            },
            "stats": {
                "decisions_total": self.stats.decisions_total,
                "no_healthy_replica_errors": self.stats.no_healthy_replica_errors,
                "capacity_fallbacks": self.stats.capacity_fallbacks,
                "decisions_by_policy": dict(self.stats.decisions_by_policy),
                "decisions_by_replica": dict(self.stats.decisions_by_replica),
                "estimated_affinity_evaluated_requests": eval_cnt,
                "avg_estimated_prefix_affinity": avg_est_affinity,
                "estimated_affinity_nonzero_ratio": est_hit_req_ratio,
            },
            "registry": self.registry.snapshot(),
        }
