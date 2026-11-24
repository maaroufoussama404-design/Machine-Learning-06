# Task 7 – Checkpoint Interruption and Recovery Experiment

## Procedure
Training was executed in two stages using the same configuration and checkpoint directory.

1. Train for 3 epochs.
2. Save the checkpoint at the end of epoch 3.
3. Start a second training process with resume enabled.
4. Load the saved checkpoint and continue from epoch 4 through epoch 10.

## Evidence

The resumed process explicitly reported:

`Resuming from epoch 4`

Training continued successfully through epoch 10.

| Epoch | Test Accuracy |
|---:|---:|
| 1 | 86.77% |
| 2 | 88.87% |
| 3 | 89.27% |
| 4 | 89.82% |
| 5 | 90.10% |
| 6 | 90.28% |
| 7 | 90.72% |
| 8 | 90.71% |
| 9 | 91.21% |
| 10 | 90.99% |

## Result
- Checkpoint created after epoch 3.
- Resume successfully started at epoch 4.
- Training completed through epoch 10.
- Final test accuracy: 90.99%.
- Best test accuracy during the resumed run: 91.21%.

## Conclusion
The experiment demonstrates that the training pipeline can recover from a controlled interruption between training runs and continue from a saved checkpoint without restarting from epoch 1.
