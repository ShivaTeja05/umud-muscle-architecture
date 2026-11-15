# Literature check and staged experiments (2026-11-10)

## Literature (searched 2026-11-10; adds to ledger I1/I4)
- No published method reports error below ours on comparable images. Best published: PA 0.8-1.8 deg, FL 2.3-5.2 mm, MT 0.6-1.5 mm (DL_Track_US, TimTrack, UltraTimTrack, SMA). All were already benchmarked or ported (ledger I1, C9, I2).
- Franchi et al. 2020 (biceps femoris, 5 cm probe): trigonometric FL equations read 6-24 % longer than extended-field-of-view truth; manual linear extrapolation is closest. Agrees with our LB loss for coupling FL (C19).
- UMUD paper (Cronin et al. 2026): benchmark = 35 architecture images, six expert operators; protocol is in its Appendix B, which I could not obtain.
- No paper on learned extrapolation of fascicles beyond the field of view, on cross-device generalisation, or on a muscle-ultrasound foundation model with a usable licence.

## Staged experiments (each has a cheap first readout and a kill rule)
| # | Experiment | Stage 0 / early readout | Kill rule | Status |
|---|---|---|---|---|
| 1 | Split-matched VL as Telemed FL yardstick (v34) | retrodiction of 3 LB facts from saved rows | < 3/3 | KILLED 1/3 |
| 2 | Better deep reference line for PA (v35 refline) | swap in the expert's line on VL from caches | no MAE gain | KILLED: 1.83 -> 1.82 deg (strip), 2.25 -> 2.33 (fusion) |
| 3 | v33 PA network trained on the angle itself | fold 0 only, read out-of-fold PA MAE at epochs 4 and 8 against production on the same images | not below production at epoch 8 -> stop, other 4 folds never run | CPU smoke running; GPU needs user OK |
| 4 | Telemed yardstick from hand labels of the 96 non-test Telemed frames | first 12 frames: which FL head is closest | sequential; stop when one head's interval excludes the others | needs the user's labels |
| 5 | T12 fascicle-length level, one group constant (probing, not model) | one submission gives the sign | - | needs user OK; campaign value -0.00016 is under its own threshold |

## r00 numbers (v35 refline, VL n 244)
Our deep line vs the expert's: median +0.46 deg, sd 2.02, MAE 1.50, corr 0.88. PA MAE by |line difference|: <1 deg 1.54; 1-2 deg 1.69; >=2 deg 2.22. Swapping in the expert's line does not lower the raw heads' MAE, and makes production worse (1.76 -> 1.88): the two errors partly cancel (corr -0.40).
