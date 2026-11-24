# Task 5 – Optimizer and Learning Rate Scheduler Analysis

## Objective
Compare the expected behaviour of Adam and SGD with momentum, combined with three learning-rate scheduling strategies.

## Configurations considered

Optimizers:
- Adam
- SGD with momentum (0.9)

Schedulers:
- StepLR
- CosineAnnealingLR
- ReduceLROnPlateau

The requested experimental budget is 10 epochs per combination.

## Theoretical comparison

| Optimizer | Scheduler | Expected Behaviour |
|---|---|---|
| Adam | StepLR | Adaptive updates with periodic learning-rate reductions |
| Adam | CosineAnnealingLR | Adaptive updates with gradual learning-rate decay |
| Adam | ReduceLROnPlateau | Reduces learning rate when validation performance stops improving |
| SGD | StepLR | Momentum-based updates with periodic learning-rate reductions |
| SGD | CosineAnnealingLR | Momentum-based optimization with gradual learning-rate decay |
| SGD | ReduceLROnPlateau | Reduces learning rate when validation performance plateaus |

## Analysis

Adam adapts the learning rate for individual model parameters using estimates of first and second moments of gradients. It often converges quickly and is relatively robust to initial learning-rate choices.

SGD with momentum uses accumulated gradients to accelerate optimization in consistent directions and reduce oscillations. It can generalize well but may require more careful learning-rate tuning.

StepLR reduces the learning rate at fixed intervals. It is simple and predictable.

CosineAnnealingLR gradually decreases the learning rate following a cosine curve, allowing larger updates early in training and smaller updates later.

ReduceLROnPlateau monitors a validation metric and reduces the learning rate when improvement stops. Its behaviour depends on validation performance and scheduler parameters.

## Expected outcome

Adam is expected to provide faster initial convergence for this CNN, while SGD with momentum may require additional tuning.

CosineAnnealingLR provides a smooth learning-rate decay, StepLR provides scheduled reductions, and ReduceLROnPlateau adapts to observed validation performance.

No optimizer-scheduler combination can be declared the winner without experimental results.

## Limitation

The six configurations were not executed. The comparisons and expected outcomes are theoretical and are not measured training results.

## Conclusion

Optimizer and scheduler selection depends on convergence speed, stability, and final generalization performance. Adam with CosineAnnealingLR is a reasonable candidate for further experimentation, but this combination has not been experimentally established as the best configuration.
