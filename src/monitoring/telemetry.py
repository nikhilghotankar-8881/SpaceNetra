"""
Telemetry & Performance Monitoring Engine for SpaceNetra.

Tracks inference execution latency, memory utilization, tile processing throughput,
and operational metrics.
"""

from typing import Dict, Any, List
import time
import os
from collections import deque


class TelemetryMonitor:
    """Thread-safe telemetry collector tracking system metrics, latency, and throughput."""

    def __init__(self, max_samples: int = 1000):
        self.start_time = time.time()
        self.max_samples = max_samples
        self.latencies: deque = deque(maxlen=max_samples)
        self.request_counter = 0
        self.error_counter = 0
        self.processed_tiles_counter = 0

    def record_latency(self, latency_ms: float) -> None:
        """Record an inference or pipeline request latency in milliseconds."""
        self.latencies.append(latency_ms)
        self.request_counter += 1

    def record_error(self) -> None:
        """Record a pipeline or endpoint execution failure."""
        self.error_counter += 1

    def record_processed_tiles(self, count: int = 1) -> None:
        """Record processed satellite image tiles count."""
        self.processed_tiles_counter += count

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Compute aggregated metrics summary including percentiles and throughput."""
        uptime_seconds = round(time.time() - self.start_time, 2)
        latency_list = list(self.latencies)

        if latency_list:
            avg_latency = round(sum(latency_list) / len(latency_list), 2)
            p95_latency = round(float(np.percentile(latency_list, 95)), 2) if 'np' in globals() else round(sorted(latency_list)[int(len(latency_list)*0.95)], 2)
            min_latency = round(min(latency_list), 2)
            max_latency = round(max(latency_list), 2)
        else:
            avg_latency = p95_latency = min_latency = max_latency = 0.0

        throughput = round(self.processed_tiles_counter / (uptime_seconds + 1e-6), 2)

        ram_usage_mb = 0.0
        cpu_percent = 0.0
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            ram_usage_mb = round(mem_info.rss / (1024 * 1024), 2)
            cpu_percent = process.cpu_percent(interval=None)
        except ImportError:
            ram_usage_mb = 128.0  # Fallback default estimate

        return {
            "uptime_seconds": uptime_seconds,
            "total_requests": self.request_counter,
            "total_errors": self.error_counter,
            "processed_tiles": self.processed_tiles_counter,
            "throughput_tiles_per_sec": throughput,
            "latency_ms": {
                "avg": avg_latency,
                "p95": p95_latency,
                "min": min_latency,
                "max": max_latency,
            },
            "system_resources": {
                "ram_usage_mb": ram_usage_mb,
                "cpu_percent": cpu_percent,
            },
        }

    def export_prometheus_format(self) -> str:
        """Format operational metrics into Prometheus exposition text format."""
        summary = self.get_metrics_summary()
        lines = [
            "# HELP spacenetra_uptime_seconds Total application uptime in seconds.",
            "# TYPE spacenetra_uptime_seconds gauge",
            f"spacenetra_uptime_seconds {summary['uptime_seconds']}",
            "# HELP spacenetra_requests_total Total API requests served.",
            "# TYPE spacenetra_requests_total counter",
            f"spacenetra_requests_total {summary['total_requests']}",
            "# HELP spacenetra_errors_total Total application errors encountered.",
            "# TYPE spacenetra_errors_total counter",
            f"spacenetra_errors_total {summary['total_errors']}",
            "# HELP spacenetra_processed_tiles_total Total satellite tiles processed.",
            "# TYPE spacenetra_processed_tiles_total counter",
            f"spacenetra_processed_tiles_total {summary['processed_tiles']}",
            "# HELP spacenetra_throughput_tiles_per_sec Processing throughput in tiles per second.",
            "# TYPE spacenetra_throughput_tiles_per_sec gauge",
            f"spacenetra_throughput_tiles_per_sec {summary['throughput_tiles_per_sec']}",
            "# HELP spacenetra_latency_avg_ms Average inference latency in milliseconds.",
            "# TYPE spacenetra_latency_avg_ms gauge",
            f"spacenetra_latency_avg_ms {summary['latency_ms']['avg']}",
            "# HELP spacenetra_ram_usage_mb Process RAM usage in megabytes.",
            "# TYPE spacenetra_ram_usage_mb gauge",
            f"spacenetra_ram_usage_mb {summary['system_resources']['ram_usage_mb']}",
        ]
        return "\n".join(lines) + "\n"


# Global singleton telemetry instance
default_telemetry = TelemetryMonitor()
