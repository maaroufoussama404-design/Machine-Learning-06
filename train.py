import argparse
import copy
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--optimizer", choices=["Adam", "SGD"])
    parser.add_argument("--scheduler",
                        choices=["none", "StepLR", "CosineAnnealingLR",
                                 "ReduceLROnPlateau"])
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = config["model"]
        self.network = nn.Sequential(
            nn.Conv2d(model["input_channels"], model["conv1_channels"],
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(model["conv1_channels"], model["conv2_channels"],
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(model["conv2_channels"] * 7 * 7,
                      model["num_classes"])
        )

    def forward(self, x):
        return self.network(x)


def make_loaders(config):
    data = config["data"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((data["mean"],), (data["std"],))
    ])

    train_set = datasets.FashionMNIST(
        root=data["root"], train=True,
        download=data["download"], transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data["root"], train=False,
        download=data["download"], transform=transform
    )

    generator = torch.Generator()
    generator.manual_seed(config["seed"])

    common = {
        "batch_size": data["batch_size"],
        "num_workers": data["num_workers"],
        "pin_memory": data["pin_memory"],
        "persistent_workers": data["num_workers"] > 0
    }

    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, **common
    )
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, test_loader


def make_optimizer(model, training):
    if training["optimizer"] == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=training["learning_rate"],
            momentum=training["momentum"],
            weight_decay=training["weight_decay"]
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"]
    )


def make_scheduler(optimizer, training, epochs):
    name = training["scheduler"]
    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.5
        )
    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    return None


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_epoch(model, loader, optimizer, criterion, device,
                accumulation_steps):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        group_start = (index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, len(loader) - group_start)
        (loss / group_size).backward()

        if (index + 1) % accumulation_steps == 0 or index + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    training = config["training"]
    data = config["data"]

    overrides = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    for key, value in {
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "accumulation_steps": args.accumulation_steps
    }.items():
        if value is not None:
            training[key] = value

    if data["batch_size"] < 1 or data["num_workers"] < 0:
        raise ValueError("Invalid batch size or worker count.")
    if training["accumulation_steps"] < 1:
        raise ValueError("accumulation_steps must be at least 1.")

    set_seed(config["seed"])
    device = torch.device(config["output"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    train_loader, test_loader = make_loaders(config)
    model = CNN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, training)
    scheduler = make_scheduler(optimizer, training, training["epochs"])

    project_root = config_path.parent.parent
    checkpoint_dir = project_root / config["checkpoint"]["directory"]
    results_dir = project_root / config["output"]["results_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_accuracy = 0.0
    history = []

    resume_path = args.checkpoint
    if args.resume and not resume_path:
        resume_path = str(checkpoint_dir / "latest.pt")

    if resume_path:
        state = torch.load(resume_path, map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_accuracy = state.get("best_accuracy", 0.0)
        history = state.get("history", [])
        if "python_rng" in state:
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
        print(f"Resuming from epoch {start_epoch + 1}")

    tracking = config["tracking"]
    tracking_uri = tracking["tracking_uri"]

    if tracking_uri.startswith("sqlite:///"):
        database_path = project_root / tracking_uri.replace("sqlite:///", "", 1)
        tracking_uri = "sqlite:///" + database_path.resolve().as_posix()
        database_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tracking_path = project_root / tracking_uri
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = tracking_path.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(tracking["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({
            "model": config["model"]["name"],
            "dataset": data["name"],
            "batch_size": data["batch_size"],
            "num_workers": data["num_workers"],
            "epochs": training["epochs"],
            "optimizer": training["optimizer"],
            "learning_rate": training["learning_rate"],
            "scheduler": training["scheduler"],
            "accumulation_steps": training["accumulation_steps"],
            "seed": config["seed"],
            "device": str(device)
        })
        mlflow.log_artifact(str(config_path))

        for epoch in range(start_epoch, training["epochs"]):
            train_loss, train_accuracy = train_epoch(
                model, train_loader, optimizer, criterion, device,
                training["accumulation_steps"]
            )
            test_loss, test_accuracy = evaluate(
                model, test_loader, criterion, device
            )

            if scheduler is not None:
                if training["scheduler"] == "ReduceLROnPlateau":
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            learning_rate = optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "learning_rate": learning_rate
            }
            history.append(record)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "learning_rate": learning_rate
            }, step=epoch + 1)

            best_accuracy = max(best_accuracy, test_accuracy)

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_accuracy": best_accuracy,
                "history": history,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state()
            }
            torch.save(state, checkpoint_dir / "latest.pt")
            if test_accuracy >= best_accuracy:
                torch.save(state, checkpoint_dir / "best.pt")

            print(
                f"Epoch {epoch + 1}/{training['epochs']} | "
                f"train loss {train_loss:.4f} | "
                f"test loss {test_loss:.4f} | "
                f"test accuracy {test_accuracy:.4f}"
            )

        history_path = results_dir / "training_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        with (results_dir / "training_summary.json").open("w") as f:
            json.dump({
                "best_test_accuracy": best_accuracy,
                "completed_epochs": len(history),
                "device": str(device)
            }, f, indent=2)

        if tracking["log_artifacts"]:
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(results_dir / "training_summary.json"))
            mlflow.log_artifact(str(checkpoint_dir / "best.pt"))

        print(f"Best test accuracy: {best_accuracy:.4f}")
        print(f"Results: {results_dir}")
        print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()

