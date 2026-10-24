# UMUD Rules And Reproducibility Audit

Last checked: 2026-09-14

Authoritative pages:

- Rules: https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/rules
- Data: https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data
- Organizer reproducibility clarification: https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/discussion/739660

## Current Compliance

| Area | Status | Action |
| --- | --- | --- |
| Competition acceptance | Confirmed | Kaggle shows that this account has accepted the competition rules. |
| Personal eligibility | User confirmation | Entrant must be a registered account holder, meet the local age-of-majority rule, not be in an excluded jurisdiction, and comply with local/employer policies. |
| Team size | Compliant | Maximum team size is 10. Do not use multiple accounts. |
| Submission limit | Compliant | Maximum five submissions per day; select at most two final submissions. |
| Competition data | Compliant locally | Data is CC BY-NC-SA 4.0 and excluded by `.gitignore`; never redistribute it to non-participants. |
| External data | Allowed with conditions | Use only public, equally accessible data at no or reasonable cost; record URL, version, license, and preprocessing. |
| Pretrained weights | Allowed, licensing to verify | ImageNet ResNet-34 is public and external models are permitted, but retain a from-scratch fallback and obtain clarification that the winning weight artifact can be relicensed under the required terms. Record version, source, and checksum. |
| Code sharing | Compliant | Keep private sharing within the Kaggle team. Public sharing is allowed. |
| Winner license | Corrected | Project metadata and `LICENSE` now declare GPL-3.0-only, the required winner license. |
| Reproducibility | In progress | A top-three entry needs public source, dependency lock, documentation, and a complete train/inference path. |
| Novelty | In progress | External code or models must be incorporated into an improved or novel solution, not submitted unchanged. |

The current software dependencies use permissive OSI licenses compatible with
GPL distribution (`torch`, OpenCV, `timm`, and `segmentation-models-pytorch`).
Their exact versions and transitive licenses still need to be frozen before a
final release.

## Test Annotation Risk

The foundational rules prohibit hand labeling or human prediction of test or
validation records. An organizer discussion later described declared manual
test annotation as permitted. Because the foundational rules state that they
control conflicts, this repository will not use hand-labeled test targets for
training, calibration, or submission unless Kaggle and the sponsor provide an
unambiguous written ruling. This keeps the main track reproducible and
prize-safe.

Discussion: https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/discussion/690868

## Data Provenance Register

| Source | Intended use | License/status |
| --- | --- | --- |
| UMUD competition train/test | Primary training and inference | CC BY-NC-SA 4.0; competition use only |
| UMUD static expert benchmark | External model selection, never fit-and-score on the same fold | CC BY 4.0; https://osf.io/xbawc/ |
| UMUD dynamic expert benchmark | Sequence validation and temporal modeling | CC BY 4.0; https://osf.io/xbawc/ |
| DL_Track_US data/code | Candidate external pretraining and implementation reference | Public; code Apache-2.0; https://github.com/PaulRitsche/DL_Track_US |
| ImageNet ResNet-34 weights | Encoder initialization | Public pretrained weights distributed through `segmentation-models-pytorch`/`timm` |

Before adding any other dataset, verify that it is longitudinal B-mode muscle
ultrasound, publicly available to all competitors, legally reusable, and useful
for the target muscles or scanner domains. Record its checksum and citation.

## Release Checklist

- Keep competition images, labels, generated masks, and submissions out of Git.
- Pin Python and package versions and export the Kaggle runtime details.
- Save seeds, split manifests, duplicate hashes, checkpoints, and SHA-256 sums.
- Publish scripts that reproduce preprocessing, training, inference, and CSV validation.
- Document all external data and pretrained models in the final method report.
- Do not use leaderboard-derived blending as the final official method.
