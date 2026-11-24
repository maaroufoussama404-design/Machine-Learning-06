# Task 6 – Gradient Accumulation Experiment

## Configurations
- Baseline: physical batch size 256, accumulation steps 1.
- Accumulation: physical batch size 64, accumulation steps 4.
- Effective batch size: 256 for both.
- Epochs: 5 per configuration.
- Optimizer: Adam, learning rate 0.001.
- Dataset: FashionMNIST.
- Device: CPU.
- Random seed: 42.

## Measured results

| Metric | Batch 256 | Batch 64, accumulation 4 |
|---|---:|---:|
| Final test accuracy | 0.8888 | 0.8890 |
| Training time (seconds) | 109.12 | 108.38 |

Maximum absolute parameter difference after training: 0.0187196657

Absolute final accuracy difference: 0.000200

## Interpretation
Both configurations use an effective batch size of 256. Gradient accumulation processes four micro-batches before performing an optimizer update. Gradients are weighted by the number of samples, including the final incomplete batch.

Small numerical differences may occur because of floating-point computation and CPU execution.

## Conclusion
The experiment compares physical batch size 256 with four accumulated micro-batches of 64 over the same five-epoch budget. See gradient_accumulation_comparison.csv for the epoch-by-epoch measurements.
