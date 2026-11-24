# Project Proposal

## Title
Lightweight Fashion Item Classification for Edge Devices Using CNN Compression

## Student Information
- Name: Oussama Maarouf
- Student ID: 69815

## Problem Statement
Fashion image classification can support applications such as smart retail and automated clothing categorization. Edge devices have limited computational resources, memory, and processing capacity. This project aims to develop and optimize a lightweight CNN capable of classifying clothing images efficiently.

## Dataset
FashionMNIST: 70,000 grayscale images across 10 clothing categories, each measuring 28 × 28 pixels.

## Model
The CNN architecture developed in the previous laboratory will serve as the baseline model.

## Evaluation Metrics
- Classification accuracy
- Inference latency
- Model artifact size
- Performance before and after compression

## Deployment Constraints
- Maximum target inference latency: 50 ms
- Maximum model artifact size: 5 MB
- Target environment: CPU-based edge device

## Optimization Plan
1. Structured pruning to reduce model complexity.
2. Quantization to reduce model size and potentially improve inference speed.
3. Knowledge distillation if additional optimization is needed.

The optimized model will be compared with the baseline to evaluate the trade-off between accuracy, latency, and model size.

## Expected Outcome
A compact image-classification model that maintains competitive accuracy while meeting the deployment constraints.
