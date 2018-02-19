# Candle: 
PyTorch utilities for pruning and quantization.

## Notes:
- When designing architectures, it's helpful to interleave batch normalization with quantized layers. This reduces covariate shift, which can be quite extreme for binary weights.
- It also helps to "prime" weights by pre-training with full-precision using a soft quantize penalty (see `quantized_loss`). This loss should increase according to a schedule. Sometimes, this will make the difference between convergence and divergence.
- Having a good pruning schedule is quite important. On the simple MNIST dataset, an absolute difference of 0.5-1 point can occur, depending on the pruning schedule used.
- Model compression varies from architecture to architecture per task. For MNIST, a simple conv net with 1.7M parameters can be pruned to ~1.2% of the original model size without loss in accuracy (99.5%).
