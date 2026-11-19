# E-cert -- the certificate on real streams

| seed | max active | n steps measured | coverage (calibrated) | coverage (no calibration) | coverage (no probe slice, replay-cal.) | coverage (full task loss) | median tightness (units of tau) | fraction beta* binds | median resolve gap |
|---|---|---|---|---|---|---|---|---|---|
| 0 | None | 75050 | 0.9613 | 0.3996 | 0.9569 | 0.9872 | 8.7611 | 0.9999 | 24.5185 |

"median resolve gap" (Remark "Admission control is cheap, not optimal"): at steps where beta*<1, the relative extra magnitude re-solving with tau tightened by beta* would have bought over the cheap backtrack beta*Delta, compared in the REDUCED beta-space (see run_one's scope note).


Evaluation criterion: coverage materially below 1 out-of-sample invalidates the certificate; coverage near 1 with tightness of order 10 or more makes it valid but uninformative; matching replay-slice and probe-slice calibration makes the probe slice unnecessary.


NOT implemented in this pass: the horizon-amortised budget schedule tau_i^(t)=B_i/(T_i-t) (Corollary "horizon"; only the constant schedule tau_i^(t)=knob is measured). The "first-order utilities, d_i=0" ablation is a separate run (`run_one(method="nashmtl")`), not read off this trajectory, since it changes the actual update applied.
