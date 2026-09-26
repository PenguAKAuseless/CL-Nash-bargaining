# E4b -- buffer curvature against the TRUE Hessian

All quantities matrix free: the true Hessian acts by double backward, the buffer estimator by the same weighted Gauss-Newton product the training loop builds its curvature block from, and extreme eigenvalues come from Lanczos on those actions. The last two columns are the unweighted function-space operator J^T J of E4, for comparison.

| task | Hess. examples | buffer | lambda_min(H) | lambda_max(H) | ||hat_H||_op | ||hat_H - H||_op | vs zeta used | zeta floor implied | s | ||J^T J||_op | ||J^T J - H||_op |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 512 | 220 | -1.189e+01 | 2.264e+01 | 2.521e+01 | 1.834e+01 | 1834.0x | 1.189e+01 | 40 | 2.512e+02 | 2.456e+02 |
| 1 | 512 | 148 | -1.019e+01 | 3.725e+01 | 3.093e+01 | 1.630e+01 | 1630.4x | 1.019e+01 | 42 | 6.236e+02 | 5.952e+02 |
| 2 | 512 | 144 | -1.378e+01 | 5.186e+01 | 6.001e+01 | 2.668e+01 | 2667.8x | 1.378e+01 | 38 | 1.394e+03 | 1.362e+03 |
| 3 | 512 | 134 | -9.628e+00 | 4.479e+01 | 4.461e+01 | 1.722e+01 | 1721.7x | 9.628e+00 | 38 | 1.327e+03 | 1.297e+03 |
| 4 | 512 | 150 | -9.827e+00 | 4.114e+01 | 4.006e+01 | 1.846e+01 | 1845.6x | 9.827e+00 | 42 | 2.129e+03 | 2.103e+03 |
| 5 | 512 | 99 | -1.125e+01 | 4.554e+01 | 5.613e+01 | 3.154e+01 | 3153.6x | 1.125e+01 | 39 | 3.294e+03 | 3.272e+03 |
| 6 | 512 | 97 | -1.052e+01 | 3.885e+01 | 5.056e+01 | 3.624e+01 | 3624.3x | 1.052e+01 | 47 | 3.900e+03 | 3.878e+03 |
| 7 | 512 | 94 | -1.235e+01 | 5.055e+01 | 4.267e+01 | 3.045e+01 | 3045.2x | 1.235e+01 | 40 | 5.505e+03 | 5.472e+03 |
| 8 | 512 | 80 | -1.756e+01 | 3.720e+01 | 5.502e+01 | 4.178e+01 | 4177.8x | 1.756e+01 | 43 | 7.449e+03 | 7.430e+03 |

Largest measured ||hat_H - H||_op relative to the zeta = 0.01 used on real streams: **4177.8x**.

Boundaries with an indefinite true Hessian: **9/9**; the most negative eigenvalue seen sets a hard lower bound on any admissible curvature-uncertainty floor.

Claim (the zeta actually used bounds the real curvature error) holds: **False**
