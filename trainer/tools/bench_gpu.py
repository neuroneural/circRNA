"""Raw GPU throughput of the model, no data pipeline: samples/s per batch size.

    python -m tools.bench_gpu --shape 64 64 64            # from trainer/
    python -m tools.bench_gpu --shape 53 63 52 --channels 3 --amp

Rows append to <paths.logdir>/<name>/results.csv, beside the experiments.
train.py's samples/s below these numbers is time lost to loading and overhead.
"""

import argparse
import csv
import os
import socket
import time

import torch
from omegaconf import OmegaConf

from src.models.resnet3d import ResNet3D, default_HPs

COLUMNS = ["time", "host", "gpu", "shape", "channels", "amp", "batch",
           "samples_per_s", "ms_per_step", "peak_gib"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast")
    parser.add_argument("--name", default="bench_gpu", help="folder under paths.logdir")
    args = parser.parse_args()

    # same log root as train.py
    rundir = os.path.join(OmegaConf.load("conf/config.yaml").paths.logdir, args.name)
    os.makedirs(rundir, exist_ok=True)
    results = os.path.join(rundir, "results.csv")

    torch.backends.cudnn.benchmark = True
    params = default_HPs(OmegaConf.create({}))
    params.in_channels = args.channels
    model = ResNet3D(params).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    print(f"shape={args.shape} channels={args.channels} amp={args.amp} -> {results}")

    for batch in args.batches:
        x = torch.randn(batch, args.channels, *args.shape, device="cuda")
        y = torch.randint(0, 2, (batch,), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(args.steps + 5):
                if step == 5:  # warmup done, cudnn has picked kernels
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                    loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"batch {batch:4d}: OOM")
            break

        elapsed = time.perf_counter() - start
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(),
            "shape": "x".join(map(str, args.shape)),
            "channels": args.channels,
            "amp": args.amp,
            "batch": batch,
            "samples_per_s": round(batch * args.steps / elapsed, 1),
            "ms_per_step": round(1000 * elapsed / args.steps, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }

        # one row at a time, so a killed job keeps what it measured
        new = not os.path.isfile(results)
        with open(results, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"batch {batch:4d}: {row['samples_per_s']:7.0f} samples/s  "
            f"{row['ms_per_step']:6.1f} ms/step  peak {row['peak_gib']:5.1f} GiB"
        )


if __name__ == "__main__":
    main()
