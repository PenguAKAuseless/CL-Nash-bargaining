# E7b -- reduced dimension and solve cost at real buffer scale

Backbone with 1109240 parameters, 100-class head, 5 absent players in the buffer. dim V is measured as the numerical rank of the actual Jacobian rows plus one direction per player gradient; the solve cost is one dense damped-Newton iteration at that dimension. The rank is taken at the smallest relative singular-value threshold at which a duplicated copy of one player's rows adds no rank (the control noise floor column is the largest singular value that copy introduces), so float32 noise cannot hide a rank deficiency.

| examples/player | basis rows | bound n+sum k_i | dim V measured | rank tol | smallest sv / largest | control noise floor | basis build (s) | solve dim timed | s / Newton iter | solve peak mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 500 | 505 | 505 | 1e-06 | 1.54e-02 | 4.88e-09 | 3.2 | 505 | 1.295 | 42.9 |
| 2 | 1000 | 1005 | 1005 | 1e-06 | 7.31e-03 | 4.51e-09 | 5.6 | 1005 | 1.833 | 169.8 |
| 4 | 2000 | 2005 | 2005 | 1e-06 | 4.32e-03 | 6.55e-09 | 15.6 | 2005 | 3.434 | 675.5 |
| 8 | 4000 | 4005 | 4005 | 1e-06 | 2.43e-03 | 6.99e-09 | 31.2 | 4000 | 3.668 | 2688.3 |

## Projected to larger per-player samples

| examples/player | bound n+sum k_i | dimension timed | s / Newton iter | peak mem (MB) |
|---|---|---|---|---|
| 8 | 4005 | 4005 | 3.794 | 2695.0 |
| 32 | 16005 | 6000 | 13.280 | 6048.5 |
| 128 | 64005 | 6000 | 13.873 | 6048.5 |

The measured dim V is what decides whether the exact reduction is usable in training: it is independent of the parameter count, as Proposition "Representer property" states, but it grows linearly in buffered examples times classes, and a dense Newton step is cubic in it.
