# E7c -- curvature block: default path against the forward-only path

Reduced ResNet-18 (1109240 parameters), 64 examples per player, one curvature block per timing. Both paths compute the same array; the rightmost column is the measured disagreement between them.

| players | default (s) | forward-only (s) | speedup | max relative disagreement |
|---|---|---|---|---|
| 2 | 0.197 | 0.094 | 2.10x | 1.09e-05 |
| 4 | 0.717 | 0.343 | 2.09x | 7.77e-06 |
| 6 | 1.561 | 0.726 | 2.15x | 1.11e-05 |
| 8 | 2.727 | 1.271 | 2.14x | 1.28e-05 |
| 10 | 4.395 | 2.036 | 2.16x | 1.17e-05 |

Median speedup: **2.14x**, at a worst relative disagreement of 1.3e-05 (the two paths are algebraically identical; see tests/test_methods_torch.py).
