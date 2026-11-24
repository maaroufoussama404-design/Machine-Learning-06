import time
import yaml
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.profiler import profile, ProfilerActivity, record_function

from train import CNN, make_loaders, make_optimizer, set_seed

def main():
    project = Path(__file__).resolve().parent
    config_path = project / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cpu")

    train_loader, _ = make_loaders(config)
    model = CNN(config).to(device)
    optimizer = make_optimizer(model, config["training"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    timings = {
        "data_loading": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0
    }

    iterator = iter(train_loader)
    print("Profiling one complete training epoch...")

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:

        for step in range(len(train_loader)):
            start = time.perf_counter()
            with record_function("data_loading"):
                images, labels = next(iterator)
            timings["data_loading"] += time.perf_counter() - start

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            start = time.perf_counter()
            with record_function("forward"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            timings["forward"] += time.perf_counter() - start

            start = time.perf_counter()
            with record_function("backward"):
                loss.backward()
            timings["backward"] += time.perf_counter() - start

            start = time.perf_counter()
            with record_function("optimizer"):
                optimizer.step()
            timings["optimizer"] += time.perf_counter() - start

            prof.step()

    output_dir = project / "results"
    output_dir.mkdir(exist_ok=True)

    total = sum(timings.values())

    rows = []
    for operation, seconds in timings.items():
        rows.append({
            "operation": operation,
            "time_seconds": round(seconds, 4),
            "percentage": round(seconds / total * 100, 2)
        })

    df = pd.DataFrame(rows)
    df.loc[len(df)] = {
        "operation": "total",
        "time_seconds": round(total, 4),
        "percentage": 100.0
    }

    df.to_csv(output_dir / "profiler_breakdown.csv", index=False)
    prof.export_chrome_trace(str(output_dir / "profiler_trace.json"))

    print("\n--- PROFILING BREAKDOWN ---")
    print(df.to_string(index=False))

    print("\n--- PYTORCH PROFILER SUMMARY ---")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=15
    ))

    print("\nSaved:")
    print(output_dir / "profiler_breakdown.csv")
    print(output_dir / "profiler_trace.json")


if __name__ == "__main__":
    main()
