"""
scripts/benchmark_inference.py
==============================
추론 속도 프로파일링. 어디서 시간이 걸리는지 측정.

Usage:
    python -m scripts.benchmark_inference --pt-path pt_files/runs/20260610-004/best_model.pt
    python -m scripts.benchmark_inference --pt-path pt_files/runs/20260610-004/best_model.pt --use-onnx
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.realtime import LiteSeq2SeqCNNGRU_AttnPool, load_model_from_pt

N_WARMUP = 10
N_BENCH  = 200
INPUT_STEPS  = 400
OUTPUT_STEPS = 40


def bench_pytorch(model, x_mean, x_std, device, n=N_BENCH):
    x_np = np.random.randn(1, 4, INPUT_STEPS).astype(np.float32)
    x_t  = torch.from_numpy(x_np).to(device)
    xn   = x_t.clone()
    xn[:, :3, :] = (x_t[:, :3, :] - torch.from_numpy(x_mean).to(device)) / \
                   (torch.from_numpy(x_std).to(device) + 1e-8)

    # warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            model(xn)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(xn).cpu().numpy()
        times.append((time.perf_counter() - t0) * 1000)

    return times


def bench_onnx(onnx_path, x_mean, x_std, n=N_BENCH):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    x_np = np.random.randn(1, 4, INPUT_STEPS).astype(np.float32)
    xn   = x_np.copy()
    xn[:, :3, :] = (x_np[:, :3, :] - x_mean) / (x_std + 1e-8)

    for _ in range(N_WARMUP):
        sess.run(None, {inp_name: xn})

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: xn})
        times.append((time.perf_counter() - t0) * 1000)

    return times


def report(label, times):
    import statistics
    print(f"\n[{label}]  n={len(times)}")
    print(f"  mean  : {statistics.mean(times):.2f} ms")
    print(f"  median: {statistics.median(times):.2f} ms")
    print(f"  p95   : {sorted(times)[int(len(times)*0.95)]:.2f} ms")
    print(f"  min   : {min(times):.2f} ms")
    print(f"  max   : {max(times):.2f} ms")
    print(f"  target: 5 ms  -> {'OK' if statistics.median(times) < 5 else 'TOO SLOW'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt-path", required=True)
    p.add_argument("--onnx-path", default=None,
                   help="ONNX 파일 경로. 없으면 pt-path 옆에 자동 생성")
    p.add_argument("--use-onnx", action="store_true",
                   help="ONNX Runtime 벤치마크 실행 (onnxruntime 필요)")
    p.add_argument("--export-only", action="store_true",
                   help="ONNX export만 하고 종료")
    args = p.parse_args()

    device = "cpu"
    pt_path = Path(args.pt_path)
    onnx_path = Path(args.onnx_path) if args.onnx_path else pt_path.with_suffix(".onnx")

    print(f"[LOAD] {pt_path}")
    model, x_mean, x_std, y_mean, y_std = load_model_from_pt(
        pt_path, device=device, in_ch=3, output_steps=OUTPUT_STEPS
    )

    # PyTorch benchmark
    pt_times = bench_pytorch(model, x_mean, x_std, device)
    report("PyTorch CPU", pt_times)

    # ONNX export
    if args.use_onnx or args.export_only:
        dummy = torch.randn(1, 4, INPUT_STEPS)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes=None,
            opset_version=17,
        )
        print(f"\n[EXPORT] ONNX saved -> {onnx_path}  ({onnx_path.stat().st_size/1024:.1f} KB)")

        if args.export_only:
            return

        ort_times = bench_onnx(onnx_path, x_mean, x_std)
        report("ONNX Runtime CPU", ort_times)

        import statistics
        speedup = statistics.median(pt_times) / statistics.median(ort_times)
        print(f"\n  Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
