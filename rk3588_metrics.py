#!/usr/bin/env python3
"""
RK3588 runtime metrics for Kokoro / CosyVoice RKNN pipelines.

Collects:
  - RTF   Real-Time Factor = inference_time / audio_duration (<1 is faster than realtime)
  - TTFT  Time To First Token (first speech token latency, ms)
  - CPU   Average process CPU over the run, 0-100% of all cores
  - MEM   Process RSS memory (MB)
  - NPU   Per-core load from /sys/kernel/debug/rknpu/load (requires root/debug group)
  - VRAM  NPU SRAM used (MB) from /sys/kernel/debug/rknpu/mm (if present)

NPU/VRAM need read access to debugfs:
  sudo chmod 644 /sys/kernel/debug/rknpu/load
  sudo chmod 644 /sys/kernel/debug/rknpu/mm   # optional, not on all kernels
  # or: sudo python3 benchmark.py ...
  # or: sudo usermod -aG debug $USER  (re-login)
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional

RKNPU_LOAD_PATH = "/sys/kernel/debug/rknpu/load"
RKNPU_MM_PATH = "/sys/kernel/debug/rknpu/mm"
NPU_CORE_PATTERN = re.compile(r"Core(\d+):\s*(\d+)%")

_npu_access_denied = False
_npu_mm_missing = False


@dataclass
class RuntimeMetrics:
    rtf: float = 0.0
    ttft_ms: float = 0.0
    cpu_pct: float = 0.0
    mem_mb: float = 0.0
    npu_pct: float = 0.0
    vram_mb: float = 0.0
    audio_sec: float = 0.0
    infer_sec: float = 0.0
    npu_cores_peak: List[int] = field(default_factory=list)
    npu_available: bool = False
    npu_permission_denied: bool = False
    vram_available: bool = False
    npu_model_ddr_mb: float = 0.0  # fallback: .rknn file size when SRAM mm unavailable

    def to_dict(self):
        d = asdict(self)
        return d


def npu_permission_denied() -> bool:
    return _npu_access_denied


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except PermissionError:
        return None
    except OSError:
        return None


def read_npu_cores() -> Optional[List[int]]:
    """Read per-core NPU load [%]. Returns None if unavailable."""
    global _npu_access_denied
    if _npu_access_denied:
        return None

    text = _read_text(RKNPU_LOAD_PATH)
    if text is None:
        if os.path.exists(RKNPU_LOAD_PATH):
            _npu_access_denied = True
        return None

    core_map = {
        int(m.group(1)): int(m.group(2))
        for m in NPU_CORE_PATTERN.finditer(text)
    }
    if not core_map:
        return None
    max_core = max(core_map)
    return [core_map.get(i, 0) for i in range(max_core + 1)]


def read_npu_load_pct() -> float:
    cores = read_npu_cores()
    if not cores:
        return 0.0
    return sum(cores) / len(cores)


def npu_sram_configured() -> bool:
    """True if kernel exposes NPU on-chip SRAM via debugfs."""
    return os.path.exists(RKNPU_MM_PATH)


def read_npu_model_file_mb(path: str) -> float:
    """RKNN model file size on disk — DDR footprint fallback when SRAM mm is absent."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def diagnose_npu_sram() -> str:
    """One-line hint why 显存 may read N/A on this board."""
    if npu_sram_configured():
        return "NPU SRAM debugfs 可用: /sys/kernel/debug/rknpu/mm"
    return (
        "NPU SRAM 未分配: 内核无 /sys/kernel/debug/rknpu/mm。"
        "常见原因: DTS 把片上 SRAM 分给了 rkvdec 等模块，未分给 RKNPU。"
        "运行 dmesg | grep -i sram 可确认。"
    )


def format_vram_line(m: RuntimeMetrics) -> str:
    if m.vram_available and m.vram_mb > 0:
        return f"显存: {m.vram_mb:.0f}MB"
    if m.npu_model_ddr_mb > 0:
        return f"显存: {m.npu_model_ddr_mb:.0f}MB"
    if not npu_sram_configured():
        return "显存: N/A (内核未分配NPU SRAM)"
    return "显存: N/A (无权限读取 rknpu/mm)"


def read_npu_sram_used_mb() -> float:
    """Parse `SRAM total size: 978944, used: 524288, free: 454656`."""
    global _npu_mm_missing
    if _npu_mm_missing:
        return 0.0

    text = _read_text(RKNPU_MM_PATH)
    if text is None:
        if not os.path.exists(RKNPU_MM_PATH):
            _npu_mm_missing = True
        return 0.0

    m = re.search(r"used:\s*(\d+)", text)
    if not m:
        return 0.0
    return int(m.group(1)) / (1024 * 1024)


def read_process_rss_mb(pid: Optional[int] = None) -> float:
    pid = pid or os.getpid()
    status = _read_text(f"/proc/{pid}/status")
    if not status:
        return 0.0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            kb = int(line.split()[1])
            return kb / 1024.0
    return 0.0


class MetricsCollector:
    """Collect timing + system metrics around one TTS inference."""

    def __init__(self, pid: Optional[int] = None):
        self.pid = pid or os.getpid()
        self._t0: Optional[float] = None
        self._t_first_token: Optional[float] = None
        self._proc = None
        self._cpu_count = 1
        self._cpu_initialized = False
        self._npu_cores_peak: List[int] = []
        self._cpu_peak = 0.0
        self._cpu_sum = 0.0
        self._cpu_n = 0

        try:
            import psutil

            self._proc = psutil.Process(self.pid)
            self._cpu_count = psutil.cpu_count(logical=True) or 1
        except ImportError:
            pass

    def _read_cpu_percent(self) -> float:
        """Process CPU normalized to 0-100% of all cores."""
        if self._proc is None:
            return 0.0
        raw = self._proc.cpu_percent(interval=None)
        return raw / self._cpu_count

    def mark_start(self):
        self._t0 = time.perf_counter()
        self._t_first_token = None
        self._npu_cores_peak = []
        self._cpu_peak = 0.0
        self._cpu_sum = 0.0
        self._cpu_n = 0
        if self._proc is not None:
            self._proc.cpu_percent(interval=None)  # prime baseline
            self._cpu_initialized = True

    def mark_first_token(self):
        if self._t0 is not None and self._t_first_token is None:
            self._t_first_token = time.perf_counter()

    def sample(self) -> RuntimeMetrics:
        cpu_pct = self._read_cpu_percent() if self._cpu_initialized else 0.0
        self._cpu_peak = max(self._cpu_peak, cpu_pct)
        self._cpu_sum += cpu_pct
        self._cpu_n += 1

        npu_cores = read_npu_cores()
        npu_pct = 0.0
        if npu_cores:
            if not self._npu_cores_peak:
                self._npu_cores_peak = list(npu_cores)
            else:
                for i, v in enumerate(npu_cores):
                    if i < len(self._npu_cores_peak):
                        self._npu_cores_peak[i] = max(self._npu_cores_peak[i], v)
                    else:
                        self._npu_cores_peak.append(v)
            npu_pct = sum(npu_cores) / len(npu_cores)

        vram_mb = read_npu_sram_used_mb()
        return RuntimeMetrics(
            cpu_pct=cpu_pct,
            mem_mb=read_process_rss_mb(self.pid),
            npu_pct=npu_pct,
            vram_mb=vram_mb,
            npu_cores_peak=list(self._npu_cores_peak),
            npu_available=npu_cores is not None,
            npu_permission_denied=npu_permission_denied(),
            vram_available=vram_mb > 0 or os.path.exists(RKNPU_MM_PATH),
        )

    def finish(
        self,
        audio_duration_sec: float,
        inference_time_sec: Optional[float] = None,
        peak: Optional[RuntimeMetrics] = None,
    ) -> RuntimeMetrics:
        infer_sec = inference_time_sec
        if infer_sec is None and self._t0 is not None:
            infer_sec = time.perf_counter() - self._t0
        infer_sec = infer_sec or 0.0

        ttft_ms = 0.0
        if self._t0 is not None and self._t_first_token is not None:
            ttft_ms = (self._t_first_token - self._t0) * 1000.0

        rtf = infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0

        snap = peak or self.sample()
        snap.rtf = round(rtf, 2)
        snap.ttft_ms = round(ttft_ms, 0)
        cpu_avg = self._cpu_sum / self._cpu_n if self._cpu_n else snap.cpu_pct
        snap.cpu_pct = round(cpu_avg, 0)
        snap.mem_mb = round(snap.mem_mb, 0)
        snap.npu_pct = round(snap.npu_pct, 0)
        snap.vram_mb = round(snap.vram_mb, 0)
        snap.audio_sec = round(audio_duration_sec, 2)
        snap.infer_sec = round(infer_sec, 2)
        if self._npu_cores_peak:
            snap.npu_cores_peak = self._npu_cores_peak
        return snap


def format_metrics_overlay(m: RuntimeMetrics) -> str:
    """Format like the reference UI overlay."""
    if m.npu_available and m.npu_cores_peak:
        npu_str = "/".join(f"{v}%" for v in m.npu_cores_peak)
        npu_line = f"NPU: {npu_str}"
    elif m.npu_permission_denied:
        npu_line = "NPU: N/A (无权限，请 sudo python3 benchmark.py)"
    else:
        npu_line = f"NPU: {m.npu_pct:.0f}%"

    vram_line = format_vram_line(m)

    return (
        f"RTF: {m.rtf:.2f}\n"
        f"CPU: {m.cpu_pct:.0f}%\n"
        f"内存: {m.mem_mb:.0f}MB\n"
        f"{npu_line}\n"
        f"{vram_line}\n"
        f"TTFT: {m.ttft_ms:.0f}ms"
    )


def format_metrics_json(m: RuntimeMetrics) -> str:
    import json
    return json.dumps(m.to_dict(), ensure_ascii=False)


class BackgroundSampler:
    """Poll NPU/CPU/MEM during long inference phases."""

    def __init__(self, interval_sec: float = 0.2, pid: Optional[int] = None):
        self.interval_sec = interval_sec
        self.collector = MetricsCollector(pid=pid)
        self._peak = RuntimeMetrics()
        self._thread = None
        self._stop = False

    def start(self):
        import threading

        self._stop = False
        self.collector.mark_start()

        def _loop():
            while not self._stop:
                s = self.collector.sample()
                self._peak.mem_mb = max(self._peak.mem_mb, s.mem_mb)
                self._peak.npu_pct = max(self._peak.npu_pct, s.npu_pct)
                self._peak.vram_mb = max(self._peak.vram_mb, s.vram_mb)
                if s.npu_cores_peak:
                    if not self._peak.npu_cores_peak:
                        self._peak.npu_cores_peak = list(s.npu_cores_peak)
                    else:
                        for i, v in enumerate(s.npu_cores_peak):
                            if i < len(self._peak.npu_cores_peak):
                                self._peak.npu_cores_peak[i] = max(
                                    self._peak.npu_cores_peak[i], v
                                )
                            else:
                                self._peak.npu_cores_peak.append(v)
                self._peak.npu_available = self._peak.npu_available or s.npu_available
                self._peak.npu_permission_denied = s.npu_permission_denied
                self._peak.vram_available = self._peak.vram_available or s.vram_available
                time.sleep(self.interval_sec)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def mark_first_token(self):
        self.collector.mark_first_token()

    def stop(self, audio_duration_sec: float, inference_time_sec: float) -> RuntimeMetrics:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)
        result = self.collector.finish(
            audio_duration_sec=audio_duration_sec,
            inference_time_sec=inference_time_sec,
            peak=self._peak,
        )
        result.npu_available = self._peak.npu_available
        result.npu_permission_denied = self._peak.npu_permission_denied
        result.vram_available = self._peak.vram_available
        result.npu_cores_peak = self._peak.npu_cores_peak
        return result
