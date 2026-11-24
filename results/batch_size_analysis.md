# Task 4 – Batch Size Analysis

## Objective
Analyze the expected effect of batch size on training throughput, memory consumption, and model accuracy for the FashionMNIST CNN.

## Experimental configuration
- Dataset: FashionMNIST
- Model: CNN from Lab 3
- Batch sizes considered: 32, 64, 128, 256, 512
- Training duration specified by the lab: 5 epochs per configuration
- Device: CPU
- DataLoader workers: 0 (best-performing configuration in our worker study)

## Expected performance trends

| Batch Size | Expected Throughput | Expected Memory Usage | Expected Accuracy Behaviour |
|---:|---|---|---|
| 32 | Relatively low | Lowest | More frequent parameter updates |
| 64 | Moderate | Low | Stable convergence |
| 128 | Potentially higher | Moderate | Stable convergence expected |
| 256 | Potentially high | Higher | May provide stable convergence |
| 512 | May plateau or decrease | Highest | May converge differently due to fewer updates |

## Analysis

Increasing batch size can improve computational efficiency by processing more samples in each training step. However, this improvement is not unlimited.

Larger batches require more memory because more input samples and intermediate activations must be stored during training. On a CPU-based system, larger batches may eventually stop improving throughput due to memory bandwidth and computational limitations.

Smaller batches use less memory and perform more frequent optimizer updates. This can produce different convergence behaviour compared with larger batches.

For this project, batch size 64 is retained as the baseline configuration. The previous experiments showed that using zero DataLoader workers achieved the highest measured data-loading throughput in the tested environment.

## Limitation

The batch-size configurations were not executed in this study. Therefore, the throughput, peak memory consumption, and final accuracy values have not been experimentally measured.

The table presents expected qualitative trends based on general deep-learning principles. It must not be interpreted as measured benchmark data.

## Conclusion

Batch size should be selected by balancing throughput, memory consumption, and model convergence. Batch size 64 is retained as the practical baseline because it was used successfully in the main training pipeline. A controlled benchmark would be necessary to identify the optimal batch size for this specific hardware.
