# Lab 6 – Training Pipelines and Project Proposal

This repository contains the implementation and documentation for Lab 6.

## Environment
- Python 3.12
- PyTorch
- Torchvision
- MLflow
- FashionMNIST
- CPU execution

## Model
The project uses the CNN architecture developed in Lab 3.

## Project Structure

- `configs/` – YAML training configuration.
- `results/` – Experimental results, plots, and analysis.
- `checkpoints/` – Local model checkpoints.
- `train.py` – YAML-driven training pipeline.
- `profiler_lab.py` – PyTorch profiling.
- `worker_benchmark.py` – DataLoader worker benchmark.
- `proposal.md` – Individual project proposal.

## Completed Experimental Evidence

- YAML-driven training and MLflow tracking.
- Training checkpoints.
- PyTorch profiler trace and timing breakdown.
- DataLoader worker benchmark.

## Additional Analysis

Some requested experiments are documented using theoretical analysis rather than measured results. These limitations are explicitly stated in the corresponding files under `results/`.

## Running Training

Set the MLflow file-store environment variable if using the local file-based tracking backend:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
python train.py --config configs/config.yaml
```

## Project Proposal

See `proposal.md`.
