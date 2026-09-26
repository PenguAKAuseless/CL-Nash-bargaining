# E7d -- curvature block: per-player against batched forward-only

NVIDIA GeForce RTX 3090, reduced ResNet-18 (713076 parameters, width 16), 32 examples per player, FP32 (TF32 off). Disagreement is the largest entry-wise difference from the two-sided block, relative to its largest entry.

| players | two-sided (s) | forward-only (s) | batched (s) | batched vs forward-only | batched vs two-sided | disagreement forward-only | disagreement batched |
|---|---|---|---|---|---|---|---|
| 2 | 0.228 | 0.056 | 0.031 | 1.8x | 7.3x | 1.1e-07 | 1.1e-07 |
| 4 | 0.733 | 0.219 | 0.062 | 3.6x | 11.9x | 2.3e-07 | 2.3e-07 |
| 6 | 1.633 | 0.481 | 0.104 | 4.6x | 15.6x | 1.0e-07 | 1.0e-07 |
| 8 | 2.957 | 0.871 | 0.229 | 3.8x | 12.9x | 1.5e-07 | 1.5e-07 |
| 10 | 4.659 | 1.369 | 0.289 | 4.7x | 16.1x | 1.7e-07 | 1.7e-07 |
