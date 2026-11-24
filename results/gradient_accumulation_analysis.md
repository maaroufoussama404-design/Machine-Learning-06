# Task 6 – Gradient Accumulation Analysis

## Objective
Understand how gradient accumulation can simulate a larger effective batch size.

## Configuration requested
- Reference batch size: 256
- Micro-batch size: 64
- Accumulation steps: 4
- Effective batch size: 64 x 4 = 256

## Expected behaviour
Gradient accumulation calculates gradients over several smaller micro-batches before performing one optimizer update. This reduces the memory required for each forward and backward pass while approximating training with a larger batch.

For equivalent data ordering, initialization, and loss scaling, accumulated gradients should approximate the gradients from a batch of 256 samples. Small numerical differences may occur.

## Limitation
The reference and accumulation configurations were not executed or compared. Therefore, equivalent accuracy and parameter updates have not been experimentally verified.

## Conclusion
Gradient accumulation is a useful technique when memory limitations prevent training with a large physical batch size. A controlled experiment is required to confirm equivalence for this model.

---

# Task 7 – Training Interruption and Checkpoint Recovery

## Objective
Demonstrate the ability to recover training after an interruption.

## Intended procedure
1. Start training from the initial model.
2. Stop after epoch 3.
3. Save the latest checkpoint.
4. Resume training from the saved checkpoint.
5. Continue until epoch 10.
6. Compare the resumed training history with uninterrupted training.

## Checkpoint information
The training pipeline implements checkpoint saving and loading. Checkpoints include model weights, optimizer state, scheduler state, epoch information, training history, and random-number generator states.

## Expected behaviour
Resuming from a checkpoint should continue training from the saved epoch rather than restarting from scratch. Training and validation metrics should continue from the previous history.

## Limitation
The interruption-and-resume procedure was not executed as a separate experiment. Therefore, successful recovery and continuity of the learning curves have not been experimentally demonstrated.

## Conclusion
Checkpointing is essential for reliable and reproducible training pipelines, particularly when training may be interrupted.

