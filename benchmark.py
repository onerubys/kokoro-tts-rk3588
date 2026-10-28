#!/usr/bin/env python3
"""
Kokoro RK3588 benchmark — in-process, models loaded once.

Metrics (rk3588_metrics convention):
  RTF   = infer_time / audio_time  (<1 = faster than realtime)
  TTFT  = first decoder window done (warm; excludes model init)
  CPU   = average process CPU over the run (0-100% of all cores)
  NPU   = peak per-core load from /sys/kernel/debug/rknpu/load

Example:
  python3 benchmark.py --cores 0_1_2
  sudo $(which python3) benchmark.py --cores 0_1_2 --append   # NPU/显存监控需要 root
  python3 benchmark.py --text "Hello world." --log logs/benchmark.log
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from rk3588_metrics import (
    BackgroundSampler,
    diagnose_npu_sram,
    format_metrics_overlay,
    npu_permission_denied,
    npu_sram_configured,
    read_npu_model_file_mb,
)

from infer import (
    DEPOP_CTX,
    DEPOP_OVL,
    IStft,
    RknnDecoder,
    SAMPLES_PER_FR,
    SR,
    T_FIX_DEFAULT,
    depop_append,
    load_voice,
    load_vocab,
    phonemes_to_ids,
    write_wav,
)

LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Console-only logging; benchmark.log gets concise report via write_report()."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(stream_handler)


def write_report(log_file: Optional[str], report: str, append: bool = False) -> None:
    """Write concise benchmark summary to log file."""
    if not log_file:
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        if append and log_path.exists() and log_path.stat().st_size > 0:
            f.write("\n")
        f.write(report)
        if not report.endswith("\n"):
            f.write("\n")


def synthesize(
    enc,
    har_sess,
    dec,
    istft,
    phonemes: str,
    vocab: dict,
    voice: str,
    speed: float,
    t_fix: int,
    on_first_audio=None,
):
    """Run encoder → har → decoder(NPU) → iSTFT. Returns (audio, stage_ms dict)."""
    ids = phonemes_to_ids(phonemes, vocab)
    if not ids:
        raise ValueError("no valid phonemes")

    input_ids = np.array([[0, *ids, 0]], np.int64)
    ref_s = load_voice(voice, len(ids))
    speed_arr = np.array([speed], np.float32)

    t0 = time.perf_counter()
    asr, f0, n, s, pred_dur = enc.run(
        None, {"input_ids": input_ids, "ref_s": ref_s, "speed": speed_arr}
    )
    t_enc = (time.perf_counter() - t0) * 1000

    t_total_frames = int(asr.shape[2])
    trail = int(pred_dur[-1])
    t_emit = max(1, t_total_frames - trail)

    audio_full = np.empty(0, np.float32)
    t_har = t_dec = t_ist = t_join = 0.0
    n_win = 0
    ctx = DEPOP_CTX
    step = t_fix - 2 * ctx
    emit_offset = ctx * SAMPLES_PER_FR
    half = DEPOP_OVL // 2

    for emit_start in range(0, t_emit, step):
        emit_end = min(emit_start + step, t_emit)
        win_start, win_end = emit_start - ctx, emit_end + ctx
        a_lo, a_hi = max(0, win_start), min(t_total_frames, win_end)
        lpad = a_lo - win_start
        rpad = win_end - a_hi
        rpad += t_fix - (a_hi - a_lo + lpad + rpad)

        a = asr[:, :, a_lo:a_hi]
        f = f0[:, 2 * a_lo : 2 * a_hi]
        n_slice = n[:, 2 * a_lo : 2 * a_hi]
        if lpad or rpad:
            a = np.pad(a, ((0, 0), (0, 0), (lpad, rpad)))
            f = np.pad(f, ((0, 0), (2 * lpad, 2 * rpad)))
            n_slice = np.pad(n_slice, ((0, 0), (2 * lpad, 2 * rpad)))
        a = np.ascontiguousarray(a, np.float32)
        f = np.ascontiguousarray(f, np.float32)
        n_slice = np.ascontiguousarray(n_slice, np.float32)

        t0 = time.perf_counter()
        har = har_sess.run(None, {"F0": f})[0].astype(np.float32)
        t_har += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        spec, phase = dec(a, f, n_slice, s, har)
        t_dec += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        audio = istft(spec, phase)
        t_ist += (time.perf_counter() - t0) * 1000

        emit_samples = (emit_end - emit_start) * SAMPLES_PER_FR
        left_ext = half if emit_start > 0 else 0
        right_ext = half if emit_end < t_emit else 0
        piece = audio.reshape(-1)[
            emit_offset - left_ext : emit_offset + emit_samples + right_ext
        ].astype(np.float32, copy=False)

        t0 = time.perf_counter()
        audio_full = depop_append(audio_full, piece)
        t_join += (time.perf_counter() - t0) * 1000
        n_win += 1

        if n_win == 1 and on_first_audio is not None:
            on_first_audio()

    return audio_full, {
        "enc_ms": t_enc,
        "har_ms": t_har,
        "dec_ms": t_dec,
        "istft_ms": t_ist,
        "join_ms": t_join,
        "total_ms": t_enc + t_har + t_dec + t_ist + t_join,
        "windows": n_win,
        "audio_sec": audio_full.size / SR,
    }


def load_models(encoder, har, decoder, cores, t_fix):
    logger.info("Loading models (one-time)...")
    t0 = time.perf_counter()
    enc = ort.InferenceSession(encoder, providers=["CPUExecutionProvider"])
    har_sess = ort.InferenceSession(har, providers=["CPUExecutionProvider"])
    dec = RknnDecoder(decoder, core_mask=cores)
    istft = IStft()

    warm_a = np.zeros((1, 512, t_fix), np.float32)
    warm_fn = np.zeros((1, 2 * t_fix), np.float32)
    warm_s = np.zeros((1, 128), np.float32)
    warm_har = np.zeros((1, 22, 2 * t_fix * 300 // 5 + 1), np.float32)
    _ = dec(warm_a, warm_fn, warm_fn, warm_s, warm_har)
    init_ms = (time.perf_counter() - t0) * 1000
    logger.info("init + warmup: %.0f ms", init_ms)
    return enc, har_sess, dec, istft, init_ms


def _format_run_npu(m) -> str:
    if m.npu_available and m.npu_cores_peak:
        return "/".join(f"{v}%" for v in m.npu_cores_peak)
    if m.npu_permission_denied:
        return "N/A(无权限)"
    return f"{m.npu_pct:.0f}%"


def _format_run_vram(m) -> str:
    if m.vram_available and m.vram_mb > 0:
        return f"{m.vram_mb:.0f}MB"
    if m.npu_model_ddr_mb > 0:
        return f"{m.npu_model_ddr_mb:.0f}MB"
    return "N/A"


def build_report(results: list, avg) -> str:
    """Concise summary for benchmark.log (per-run + average only)."""
    lines = ["--- per-run ---"]
    for i, (m, s) in enumerate(results, 1):
        lines.append(
            f"Run {i}: RTF={m.rtf:.3f} TTFT={m.ttft_ms:.0f}ms "
            f"CPU={m.cpu_pct:.0f}% NPU={_format_run_npu(m)} "
            f"Mem={m.mem_mb:.0f}MB VRAM={_format_run_vram(m)} "
            f"decoder={s['dec_ms']:.0f}ms audio={s['audio_sec']:.2f}s"
        )
    lines += ["", "--- average (warm) ---", format_metrics_overlay(avg)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Kokoro RK3588 in-process benchmark")
    ap.add_argument(
        "--phonemes",
        default="həlˈO wˈɜɹld ðɪs ɪz ɐ pɚfˈɔɹməns tˈɛst",
        help="IPA phonemes (skips slow G2P; use --text for G2P latency test)",
    )
    ap.add_argument("--text", default=None, help="English text (includes G2P in TTFT)")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cores", default="0_1_2", choices=RknnDecoder.CORE_MASKS)
    ap.add_argument("--out", default="wavs/bench_out.wav")
    ap.add_argument("--encoder", default="onnx/kokoro_encoder.onnx")
    ap.add_argument("--har", default="onnx/har_generator.onnx")
    ap.add_argument("--decoder", default="onnx/kokoro_decoder.rknn")
    ap.add_argument("--vocab", default="Kokoro-82M/config.json")
    ap.add_argument("--t-fix", type=int, default=T_FIX_DEFAULT)
    ap.add_argument(
        "--log",
        type=str,
        default="logs/benchmark.log",
        help="Write concise summary to this file (default: logs/benchmark.log)",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to --log instead of overwriting",
    )
    ap.add_argument(
        "--no-log",
        action="store_true",
        help="Disable file logging (console only)",
    )
    args = ap.parse_args()

    log_file = None if args.no_log else args.log
    setup_logging()

    vocab = load_vocab(args.vocab)

    phonemes = args.phonemes
    t_g2p_ms = 0.0
    if args.text:
        from infer import text_to_phonemes

        t0 = time.perf_counter()
        phonemes = text_to_phonemes(args.text)
        t_g2p_ms = (time.perf_counter() - t0) * 1000
        logger.info("[g2p] %.0f ms  %r → %r", t_g2p_ms, args.text, phonemes)

    enc, har_sess, dec, istft, init_ms = load_models(
        args.encoder, args.har, args.decoder, args.cores, args.t_fix
    )

    model_ddr_mb = read_npu_model_file_mb(args.decoder)
    if not npu_sram_configured():
        logger.warning(diagnose_npu_sram())
        logger.warning(
            "使用 RKNN 模型文件大小 %.0fMB 作为显存 fallback（NPU 实际占用 DDR，非片上 SRAM）",
            model_ddr_mb,
        )

    logger.info("=" * 50)
    logger.info("Kokoro RK3588 Benchmark (warm, in-process)")
    logger.info("=" * 50)
    logger.info("phonemes : %r", phonemes)
    logger.info("cores    : %s", args.cores)
    logger.info("runs     : %d", args.runs)

    results = []
    audio = np.empty(0, np.float32)
    for i in range(args.runs):
        sampler = BackgroundSampler(interval_sec=0.05)
        first_audio = {"marked": False}

        def mark_first():
            if not first_audio["marked"]:
                sampler.mark_first_token()
                first_audio["marked"] = True

        sampler.start()
        t0 = time.perf_counter()
        audio, stages = synthesize(
            enc,
            har_sess,
            dec,
            istft,
            phonemes,
            vocab,
            args.voice,
            1.0,
            args.t_fix,
            on_first_audio=mark_first,
        )
        infer_sec = time.perf_counter() - t0
        metrics = sampler.stop(
            audio_duration_sec=stages["audio_sec"],
            inference_time_sec=infer_sec,
        )

        if t_g2p_ms:
            metrics.ttft_ms += t_g2p_ms
        metrics.npu_model_ddr_mb = model_ddr_mb

        results.append((metrics, stages))
        if metrics.npu_available and metrics.npu_cores_peak:
            npu_str = "/".join(f"{v}%" for v in metrics.npu_cores_peak)
        elif metrics.npu_permission_denied:
            npu_str = "N/A(无权限)"
        else:
            npu_str = f"{metrics.npu_pct:.0f}%"
        vram_str = f"{metrics.vram_mb:.0f}MB" if metrics.vram_available else "N/A"
        logger.info(
            "Run %d: RTF=%.3f, TTFT=%.0fms, CPU=%.0f%%, NPU=%s, "
            "Mem=%.0fMB, VRAM=%s, decoder=%.0fms",
            i + 1,
            metrics.rtf,
            metrics.ttft_ms,
            metrics.cpu_pct,
            npu_str,
            metrics.mem_mb,
            vram_str,
            stages["dec_ms"],
        )

    dec.close()
    write_wav(args.out, audio, sr=SR)
    logger.info("wrote %s", args.out)

    avg = results[0][0]
    for key in ("rtf", "ttft_ms", "cpu_pct", "mem_mb", "npu_pct", "vram_mb"):
        setattr(avg, key, sum(getattr(m, key) for m, _ in results) / len(results))
    avg.npu_model_ddr_mb = model_ddr_mb

    last_stages = results[-1][1]

    logger.info("")
    logger.info("=" * 50)
    logger.info(
        "FINAL RESULTS (avg of %d warm runs, init %.0fms excluded)",
        args.runs,
        init_ms,
    )
    logger.info("=" * 50)
    for line in format_metrics_overlay(avg).splitlines():
        logger.info(line)
    logger.info("")
    logger.info("Stage breakdown (last run):")
    logger.info("  encoder (CPU) : %7.1f ms", last_stages["enc_ms"])
    logger.info(
        "  har gen (CPU) : %7.1f ms  (%d windows)",
        last_stages["har_ms"],
        last_stages["windows"],
    )
    logger.info("  decoder (NPU) : %7.1f ms", last_stages["dec_ms"])
    logger.info("  iSTFT   (CPU) : %7.1f ms", last_stages["istft_ms"])
    logger.info("  audio         : %.2f s", last_stages["audio_sec"])
    logger.info("=" * 50)

    if npu_permission_denied() or (not avg.npu_available and avg.npu_pct == 0):
        logger.warning(
            "NPU/显存监控读数为 0 — 是 debugfs 权限问题，不是 NPU 未运行。\n"
            "  方案1 (推荐): sudo $(which python3) benchmark.py --cores 0_1_2 --append\n"
            "  方案2: sudo chmod 644 /sys/kernel/debug/rknpu/load\n"
            "  方案3: sudo usermod -aG debug $USER  (重新登录后生效)\n"
            "decoder (NPU) 耗时已证明 RKNN 在 NPU 上推理。"
        )

    report = build_report(results, avg)
    write_report(log_file, report, append=args.append)
    if log_file:
        logger.info("summary saved to %s", log_file)


if __name__ == "__main__":
    main()
