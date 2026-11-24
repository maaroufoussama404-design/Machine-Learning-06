import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from train import make_loaders

def main():
    project = Path(__file__).resolve().parent

    with open(project / "configs" / "config.yaml", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    worker_values = [0, 1, 2, 4, 8]
    sample_limit = 10000
    results = []

    for workers in worker_values:
        config = {**base_config, "data": dict(base_config["data"])}
        config["data"]["num_workers"] = workers

        train_loader, _ = make_loaders(config)

        count = 0
        first_batch_time = None
        start = time.perf_counter()

        print(f"\nTesting num_workers={workers}...")

        iterator = iter(train_loader)

        while count < sample_limit:
            try:
                images, labels = next(iterator)
            except StopIteration:
                break

            if first_batch_time is None:
                first_batch_time = time.perf_counter() - start

            count += len(images)

        elapsed = time.perf_counter() - start
        throughput = count / elapsed

        results.append({
            "num_workers": workers,
            "samples": count,
            "time_seconds": round(elapsed, 3),
            "first_batch_seconds": round(first_batch_time or 0, 3),
            "samples_per_second": round(throughput, 2)
        })

        print(f"Time: {elapsed:.2f}s | Throughput: {throughput:.2f} images/s")

        del iterator, train_loader

    output = project / "results"
    output.mkdir(exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output / "worker_benchmark.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["num_workers"], df["samples_per_second"], marker="o")
    plt.xlabel("Number of DataLoader Workers")
    plt.ylabel("Throughput (images/second)")
    plt.title("DataLoader Worker Performance")
    plt.xticks(worker_values)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output / "worker_benchmark.png", dpi=150)
    plt.close()

    print("\n--- WORKER BENCHMARK RESULTS ---")
    print(df.to_string(index=False))
    print("\nSaved worker_benchmark.csv and worker_benchmark.png")

if __name__ == "__main__":
    main()
