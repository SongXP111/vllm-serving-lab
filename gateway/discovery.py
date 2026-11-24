"""Kubernetes Headless Service Dynamic Endpoint Discovery for M2/M3 Gateway.

Resolves a Kubernetes Headless Service (`clusterIP: None`) DNS name periodically
to obtain the exact set of Ready Pod IPs, synchronizing them into `ReplicaRegistry`:
- New Ready Pod IP appears in DNS -> register new `ReplicaState` in `ReplicaRegistry`.
- Existing Pod IP disappears from DNS (failed readinessProbe, killed, or Terminating)
  -> immediately mark `healthy=False` so ZERO new requests are routed to it, while
  existing in-flight SSE streams drain cleanly; once `inflight == 0`, unregister it
  and clear its stale prefix-affinity entries from `EstimatedPrefixCacheIndex`.
"""
import asyncio
from collections import deque
import logging
import socket
import time
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set

from gateway.registry import ReplicaRegistry, ReplicaState
from gateway.router import InstanceRouter

logger = logging.getLogger("gateway.discovery")

DNSResolverFn = Callable[[str, int], Awaitable[Set[str]]]


async def default_dns_resolver(service_host: str, port: int) -> Set[str]:
    """Resolve all IPv4 pod addresses for a Kubernetes Headless Service."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(
            service_host,
            port,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
        return {info[4][0] for info in infos if info and info[4]}
    except socket.gaierror:
        # NXDOMAIN when 0 Ready endpoints exist in the Headless Service
        return set()


class HeadlessDNSReplicaDiscoverer:
    """Periodically syncs Ready Pod IPs from a K8s Headless Service into ReplicaRegistry."""

    def __init__(
        self,
        *,
        registry: ReplicaRegistry,
        router: InstanceRouter,
        service_host: str,
        backend_port: int = 8000,
        per_replica_max_inflight: int = 4,
        poll_interval_seconds: float = 1.0,
        resolver: Optional[DNSResolverFn] = None,
        max_events: int = 100,
    ) -> None:
        self.registry = registry
        self.router = router
        self.service_host = service_host.strip()
        self.backend_port = backend_port
        self.per_replica_max_inflight = max(1, per_replica_max_inflight)
        self.poll_interval_seconds = max(0.2, poll_interval_seconds)
        self._resolver = resolver or default_dns_resolver

        self.managed_ids: Set[str] = set()
        self.draining_ids: Set[str] = set()
        self.last_sync_ts: Optional[float] = None
        self.last_resolved_ips: List[str] = []
        self.events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    @staticmethod
    def _replica_id_for_ip(ip: str) -> str:
        return f"pod-{ip.replace('.', '-')}"

    async def sync_once(self) -> Dict[str, Any]:
        """Run one reconciliation pass between DNS Ready IPs and ReplicaRegistry."""
        ips = await self._resolver(self.service_host, self.backend_port)
        now = time.time()
        self.last_sync_ts = now
        sorted_ips = sorted(ips)
        self.last_resolved_ips = sorted_ips

        desired_ids = {self._replica_id_for_ip(ip): ip for ip in sorted_ips}
        added: List[str] = []
        removed: List[str] = []
        reaped: List[str] = []

        # 1. Register newly Ready Pod IPs (or re-enable if it came back Ready)
        for rid, ip in desired_ids.items():
            existing = self.registry.get(rid)
            if existing is None:
                self.registry.register(
                    ReplicaState(
                        replica_id=rid,
                        endpoint=f"http://{ip}:{self.backend_port}",
                        max_inflight=self.per_replica_max_inflight,
                    )
                )
                self.managed_ids.add(rid)
                self.draining_ids.discard(rid)
                added.append(rid)
            else:
                self.managed_ids.add(rid)
                if rid in self.draining_ids or not existing.healthy:
                    self.registry.set_health(rid, True)
                    self.draining_ids.discard(rid)
                    added.append(rid)

        # 2. Mark disappeared Pod IPs unhealthy immediately (stops NEW requests!),
        #    and unregister once any in-flight streams finish draining.
        for rid in list(self.managed_ids):
            if rid not in desired_ids:
                existing = self.registry.get(rid)
                if existing is None:
                    self.managed_ids.discard(rid)
                    self.draining_ids.discard(rid)
                    continue
                if existing.healthy:
                    self.registry.set_health(rid, False)
                    self.draining_ids.add(rid)
                    removed.append(rid)
                if existing.inflight == 0:
                    self.registry.unregister(rid)
                    self.router.prefix_index.clear_replica(rid)
                    self.managed_ids.discard(rid)
                    self.draining_ids.discard(rid)
                    reaped.append(rid)

        if added or removed or reaped:
            event = {
                "timestamp": round(now, 3),
                "service_host": self.service_host,
                "ready_ips": sorted_ips,
                "added": added,
                "marked_unready": removed,
                "unregistered": reaped,
            }
            self.events.append(event)
            logger.info("K8s endpoint sync change: %s", event)

        return {
            "service_host": self.service_host,
            "ready_ips": sorted_ips,
            "added": added,
            "marked_unready": removed,
            "unregistered": reaped,
        }

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.sync_once()
            except Exception as exc:
                logger.warning("K8s DNS discovery error for %s: %s", self.service_host, exc)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval_seconds
                )
            except asyncio.TimeoutError:
                continue

    def start(self) -> None:
        if self._task is None:
            self._stop_event.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "service_host": self.service_host,
            "backend_port": self.backend_port,
            "poll_interval_seconds": self.poll_interval_seconds,
            "last_sync_ts": self.last_sync_ts,
            "ready_ips": list(self.last_resolved_ips),
            "draining_replicas": sorted(self.draining_ids),
            "events": list(self.events),
        }
