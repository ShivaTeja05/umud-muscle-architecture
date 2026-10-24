# External data scout: public muscle-ultrasound data with expert architecture labels

Checked 2026-10-21 by the datascout agent. Nothing in this file has been downloaded into the project. No dataset was used, no model was run, and nothing was submitted or committed.

**What was fetched to `<scratchpad>/datascout/`:**
- repository trees, READMEs, licence pages, and OSF, Zenodo, figshare and Dryad API listings;
- four TimTrack label spreadsheets of 15–45 KB each, opened only to read their column layout;
- full texts from Europe PMC.

**One local read-only check:** I measured the width of each test image's field of view with `umud.geometry.ultrasound_metadata`. It ran one image at a time, used no torch, and peaked at 0.11 GB RSS. The results are in §2.

**How the rules apply** (`research/rules_audit.md`; organiser threads 690868 and 737782):
- External data must be **public and equally accessible at no cost**, and must be **declared**: URL, version, licence and preprocessing.
- The winner's code is GPL-3.0. CC0, CC BY 4.0 and Apache-2.0 data are fine with attribution.
- Data that is "available on request" is **not** equally accessible.
- Unlicensed GitHub content is public but carries no reuse grant, so use it only for internal validation and never ship it in the final pipeline.

---

## 0. Bottom line

1. **No public dataset found has labels on the test scanners × muscles.** The test combinations are Siemens Juniper, Philips Lumify, or Telemed ArtUS with VL, GM or RF, plus CP subjects. Searches of Zenodo, figshare, Dryad, OSF, Mendeley, Kaggle and Hugging Face, plus the UMUD index of 15 datasets, turned up no labelled public images of RF, Lumify or CP.
   - The two studies that match the test domain both come from the organiser's lab in Basel. Ritsche 2022 used Lumify + Juniper on RF, VL and GM. Minghetti 2026 used Juniper on VL in CP. Both are **"data on request"**, and both are probably the **source cohorts of the test images**. **Do not request their raw data.**
   - Their **published tables** are free and CC BY, and they are the most useful external information we have; see §2.
2. **The most valuable new things are small:**
   - (a) Ritsche 2022's Table 1, a Lumify/Juniper × RF/VL/GM prior analysed by test-rater Paul Ritsche. It reproduces our Lumify and Siemens cell medians once the field-of-view geometry is accounted for (§2).
   - (b) About 20 KB of raw manual-tracking files that decide which PA convention the dynamic benchmark uses (§1, rank 2).
   - (c) The TimTrack bundle: about 80 MB, 339 images, 3 raters each, CC0/CC BY. It would be a second out-of-domain, per-image, multi-rater PA benchmark.
3. Every larger source (FALLMUD, UltraTimTrack, GasFasTrack, AnkleImage, NeuAge) serves only for segmentation pretraining or domain adaptation. That needs GPU, and the quota is exhausted until the weekly reset. Their benefit is uncertain.

## 1. Ranked by expected benefit to the current pipeline

| # | Source | What we would gain | Effort / where | Rules |
|---|---|---|---|---|
| 1 | **Ritsche et al. 2022, Front Physiol 13:981862**, Table 1 means (no images) | A Lumify and Juniper × GM/RF/VL prior, analysed by **Paul Ritsche** (test rater) with SMA. Checks the v4 Lumify choices: the RF gate, the GM muscle move, and critic fix 1 (§2) | 0 (numbers below) | Public CC BY article: OK as a hand-set prior, declare it. Raw data on request: **not OK** |
| 2 | **RUB_ultrasound_study** manual-tracking `.mat` files, 3 raters, on the **DL_Track calf-raise video = our dynamic benchmark** | Raw drawn line endpoints for the fascicle **and** the deep aponeurosis in each frame. This settles whether the dynamic "PA" is measured against the horizontal. The dynamic term is 15% of S_sel and about 70% of C7's edge over C8/C9 (critic4) | About 20 KB of files; parse with scipy | No licence, so internal validation only. The per-frame values themselves are already CC BY via UMUD |
| 3 | **TimTrack example datasets** (van der Zee & Kuo 2022), 339 images × 3 observers | A second out-of-domain multi-rater PA benchmark: VL n=153 (9 subjects), GM Telemed + Philips HD11 n=30 (15 subjects), GM Telemed SmartUs n=100 (1 subject). Gives the fascicle angle against the horizontal (scale-free), the aponeurosis angles, and thickness in px. Tests per-image PA against shrinkage outside the Esaote VL set | About 79 MB raw. Our masks need torch, so run in a **Kaggle CPU kernel** (no GPU quota) | CC0 (datasets 1–2) and CC BY 4.0 (3–4): OK |
| 4 | **UltraTimTrack GM dataset** (figshare 26502868) | Unlabelled **Telemed ArtUs EXT-1H** GM frames, the same scanner family as the test Telemed clips (45% of rows), from 8 subjects. Labels are only algorithm estimates. For self-supervised or pseudo-label adaptation, or AdaBN, of the apo/fasc nets | 8.7 GB; download individual 40 MB videos in a GPU kernel after the reset | CC BY 4.0: OK |
| 5 | **FALLMUD** (CEA / Kaggle mirror) | 812 images with fascicle + aponeurosis masks. Cunningham 504 = GM/SOL on Hitachi Aloka SSD-5000. Cronin about 308 = the DL_Track 2020 set, probably already in competition train | 125 MB; GPU training | **Licence unclear**: none on the CEA page, CC BY 4.0 in the UMUD index, CC BY-NC 4.0 on the Kaggle mirror. Ask CEA first |
| 6 | **GasFasTrack** (Drazan 2019, Zenodo 2598553) | GM on Telemed SmartUs, 5 subjects. Manual FL and **PA against the deep aponeurosis**, 3 observers × 3 days, 18 frames per subject (`validation_data.zip`). Mostly maximal contractions | 1.9 GB labels + 3.6–7 GB of video per subject; Kaggle only | CC BY 4.0: OK |
| 7 | **DL_Track_US training data** (OSF 7mjsc `Train_Data_DL_Track.7z`; UMUD xbawc "Example Dataset Structure") | About 570 apo + 310 fasc images with masks, probably ⊂ competition train. Use it for hash dedupe, and possibly to recover native-resolution originals of the stretched training images | 522 MB or 680 MB | Apache-2.0: OK |
| 8 | NeuAge Young/Old (OSF bf8ut / x7nfw) | 552 + 1408 Esaote VL/QF images, **unlabelled**. The labelled subset is ExpDrawnData, which we already use as `vl_labels.csv` | Large; GPU | CC BY 4.0: OK |
| 9 | AnkleImage (BioStudies S-BIAD1482) | 69,440 ultrafast frames, TA/SOL/GM/GL, S-Sharp Prodigy. Labels are UltraTrack-tracked (semi-automatic) | Very large | CC BY 4.0 (UMUD index); the article is CC BY-NC-ND |
| 10 | Rosa et al. 2021 soleus (Dryad vq83bk3rp) | Soleus on Telemed ArtUs EXT-1H, 6 subjects × 5 tasks, 60 fps `.mat`. No documented expert labels | 8.8 GB | CC0: OK |
| 11 | DeepMTJ test set (OSF wgy4d / GitHub luuleitner/deepMTJ) | 1344 images of the **muscle–tendon junction**, Telemed ArtUs / Aixplorer / Esaote, including impaired subjects and children. MTJ points only | 464 MB | CC BY 4.0, but irrelevant labels |
| 12 | DeepACSA (OSF a3u4v, Zenodo 5799204) + UMUD RF CSA benchmark | **Transverse** panoramic CSA masks: RF/VL/GM/GL on Juniper/Esaote/Aixplorer | 546 MB / 11 MB | Apache-2.0, but wrong plane |

**Already in hand, not new:**
- UMUD MultExpData static benchmark: 35 images × 7 raters.
- Dynamic GM calf raise: 167 frames, 3 raters.
- ExpDrawnData / "Example Annotated Images": about 250 VL Esaote images (we parsed 244), expert Fabio Sarto.
- DL_Track_US v0.3 weights.

---

## 2. Published priors from the organiser's lab, and a field-of-view check

**Ritsche et al. 2022**, Front Physiol 13:981862, PMC9478722, CC BY:
- **Subjects and devices:** 36 healthy active adults, imaged at 50% muscle length with a Lumify L12-4 (37 mm by the paper) and an ACUSON **Juniper 12L3** (54/56 mm).
- **Analysis:** PR (Paul Ritsche) collected and analysed every image with SMA, semi-automatic and visually checked; the mean of at least two images per parameter.
- **Data:** available "upon request". German UI, Basel, RF/VL/GM on Lumify + Juniper: this is very probably the cohort behind the 58 Lumify test PNGs.

Table 1 means (PA° / FL mm / MT mm):

| muscle | Lumify | Juniper |
|---|---|---|
| GM | 21.7 / 55.0 / 18.6 | 21.4 / 55.1 / 18.9 |
| RF | 15.2 / 87.5 / 21.2 | 16.4 / 82.7 / 21.4 |
| VL | 14.3 / 81.2 / 21.2 | 14.6 / 79.9 / 21.1 |

**Width of the field of view in the test images.** This is the viewport width divided by the vertical px/cm from `ultrasound_metadata`; script: `<scratchpad>/datascout/scripts/fov_width.py`.

| family | n | FOV width (cm, vertical scale applied to x) | probe aperture | implied x-scale / y-scale `c` |
|---|---|---|---|---|
| Siemens Juniper | 91 | **5.74** (5.74–5.77) | 12L3: 54–56 mm | 1.03–1.06 (slight horizontal stretch) |
| Philips Lumify | 58 | **2.86** (2.84–2.87) at every depth (px/cm 100–201) | L12-4: 34–34.5 mm (Philips/MSF), 37 mm (paper), 38.4 mm (one Philips page) | **0.75–0.84 (horizontal compression)** |
| Telemed 1200×800 | 90 | 5.79 | unknown (60 mm if LV8-5N60) | ~0.97? |
| Telemed 1088×644 | 50 | 6.40 | unknown | ~1.07? |
| legacy 853 / 512 | 12 / 8 | 6.41 or 5.74 / 5.98 | unknown | – |

**Why this looks like compression and not zoom:**
- The Lumify width is constant at 2.86 cm while the depth scale halves.
- A zoomed or cropped view would change width with depth; a full aperture drawn with non-square pixels would not.
- The vertical scale is right: our Lumify MT medians (18.5 / 21.8 / 19.6 for GM / RF / VL) match the paper (18.6 / 21.2 / 21.2).

**Mapping the paper's native values into PNG pixel geometry.** The organiser's raters put the vertical bar on both axes (thread 748126), so their labels live in PNG pixel geometry. The mapping is `PA' = atan(tan PA / c)` and `FL' = FL·sqrt(c²cos²PA + sin²PA)`:

| Lumify | prior at c=0.77 | prior at c=0.83 | C8 | **C9** | C7 | raw cell | prior if pixels were square (c=1) |
|---|---|---|---|---|---|---|---|
| GM | 27.2° / 44.4 mm | 25.6 / 47.0 | 24.8 / 46.7 | **26.4 / 44.8** | 26.8 / 44.8 | 26.4 / 44.8 | 21.7 / 55.0 |
| RF | 19.4 / 69.2 | 18.1 / 73.7 | 19.1 / 69.4 | **19.1 / 69.7** (gated) | 19.5 / 69.7 | 18.0 / 82.8 (ungated D_C9: FL 84) | 15.2 / 87.5 |
| VL | 18.2 / 64.0 | 17.1 / 68.2 | 18.9 / 64.7 | **18.6 / 65.0** | 18.9 / 65.0 | 18.6 / 65.1 | 14.3 / 81.2 |

| Siemens (C9 PA/FL) | prior at c=1.025 | prior at c=1.063 | ours (C9) |
|---|---|---|---|
| GM | 20.9 / 56.3 | 20.2 / 58.1 | 19.5 / 58.6 |
| RF | 16.0 / 84.6 | 15.5 / 87.5 | 14.8 / 87.3 |
| VL | 14.3 / 81.8 | 13.8 / 84.6 | 14.6 / 89.1 |

**Reading:**
1. **One factor taken from the probe aperture alone (nothing tuned) moves the paper's Lumify means onto our Lumify cell medians.** Averaged across muscles, the gap is about 0.5–1.5° and 1–4 mm. The PA-implied `c` is 0.80 for GM, 0.79 for RF (C9) and 0.76 for VL, which matches the FOV-implied 0.77–0.83. Siemens agrees to within 1.4° and 0–3 mm, except VL FL (+4.5 to +7 mm).
   - So our Lumify and Siemens cell centres look physically right **in the geometry the labels use**.
   - Pulling Lumify toward the raw literature means (c=1) would add about 4.5° PA and about 13 mm FL of bias.
2. **Supports gating Lumify RF FL.** The prior is 69–74 mm against the gated 69.7; the ungated 84 mm (D_C9_ungated) is 10–15 mm beyond it.
3. **Supports the C9/C7 Lumify GM move.** The prior range is 25.6–27.2°. C8's 24.8° lies below it, C9's 26.4° inside it.
4. **Undecided for critic fix 1** (Lumify RF PA to its raw 18.0°). The prior is 18.1° at c=0.83 and 19.4° at c=0.77, and the stake is about ±0.002. Lumify VL PA may be 0.4–1.5° high, which is within SMA-vs-manual noise.
5. **Main caveat:** if the raters had labelled square-pixel originals rather than these PNGs, our Lumify PA would be about 4.5° too high and FL 10–18 mm too short.
   - That would cost about 0.12 of the total score: 58/309 × (4.5/6 + 13/12)/3.
   - The host's statement (vertical scale on both axes, on the provided images) rules this out, and the only cheap way to test it is the public LB, which the team rules forbid. I recommend not acting on it.
   - Still worth asking the host in thread 748126 whether Lumify labels were drawn on the 1200×800 PNGs.
6. **Not from this paper:** these are SMA values on healthy adults, not the FIJI two-rater protocol, and the test may include CP subjects. Use them as a **sanity band**, not as shrinkage targets, unless the benchmarks say otherwise.

**Minghetti et al. 2026**, BMC Musculoskelet Disord 27:531, PMC13281589:
- **Subjects and protocol:** Juniper 12L3; VL at 50% femur length; 9 CP + 18 typically developing adolescents, about 20 years old; both legs, 3 images each.
- **Values are DL_Track_US v0.2.1 automatic**, visually inspected, not manual expert labels. The critic4 comparison "Siemens VL below the Basel study (PA 15.8–23.8°)" is therefore against DL_Track output, not rater style. Treat it as weak.
- **Data:** on request, so **not usable**. It is probably the source of the "Rechts VL 50%" Siemens block, IMG_00001–035. **Do not request.**

---

## 3. Source details

### 3.1 TimTrack example datasets (rank 3)
- **URL:** https://github.com/timvanderzee/ultrasound-automated-algorithm, folder `Example images/` (`raw/` 360 files / 79 MB; `analyzed/` 341 / 44.5 MB). Paper: van der Zee & Kuo 2022, PLoS One e0265752.
- **Licence:** CC0-1.0 for the code, datasets 1–2 and the single images. Datasets 3–4 are CC BY 4.0 (Drazan, Zenodo 2598553; Seynnes & Cronin, Mendeley dpmf9bz8pt); cite the originals.

| dataset | muscle | scanner | images / subjects | condition |
|---|---|---|---|---|
| 1 | **VL** | GE Logiq E9, 5 cm 11 MHz | 153 / 9 (`S1..S11`, png) | isometric knee torque |
| 2 | GL | GE Logiq E9 | 56 / 1 | jump + range of motion |
| 3 | **GM** | **Telemed SmartUs** LV7.5/60/128Z-2 | 100 / 1 (jpg) | isokinetic 30–500°/s + max isometric; PA up to ~48° |
| 4 | **GM** | **Telemed LogicScan 128** LV7.5/60/96Z (SampleA, png) + **Philips HD11XE** (SampleB, bmp) | 15 + 15 / 15 | rest |

**Population:** healthy adults.

**Labels:**
- **Raters:** 3 independent observers (biomechanics staff, ≥1 year of ultrasound and manual tracking), in ImageJ.
- **Files:** `raw/TimTrack paper/dataset N/datasetN_for_manual_estimation[_obsK].xlsx`.
  - d1: one sheet per observer (TZ/EM/BH), 162 rows. Columns: subject, leg, name, `alpha`, `betha`, `spoint`, `dpoint`, `height`.
  - d2, d3, d4: one file per observer.
  - d2: columns `image, alpha, betha, spoint, dpoint, height`.
  - d3 and d4: columns `Alpha (deg), Beta (deg), Height (pixels)`, plus `Gamma` in d4 and in d3's first sheet.
- **Meaning, per `auto_ultrasound.m`:**
  - `alpha` = fascicle angle against the image horizontal;
  - `betha` = superficial aponeurosis angle;
  - `gamma` = deep aponeurosis angle;
  - `height` = thickness in **pixels**;
  - TimTrack's own PA is `alpha − betha`, i.e. against the superficial aponeurosis.
- **PA against the deep aponeurosis** (the UMUD convention) = `alpha − gamma`, available only where Gamma exists. FL is not labelled (TimTrack computes `height/sin(alpha−betha)`). Units are pixels; the mm scale would have to come from each image's ruler.

**Use:**
- (a) Score our fascicle-orientation outputs (seg fragments, `fasc_orient_v1` field) against the 3-rater mean of `alpha` on 339 out-of-domain images. This is scale-free.
- (b) Score PA against the deep aponeurosis on d4 (Telemed + Philips GM, 30 images × 3).
- (c) Use the spread between raters as a noise floor.
- (d) This is the cheapest way to replicate the VL-Esaote result that per-image PA is no better than a constant, on new scanners, before tuning PA shrinkage again.

Masks would come from a Kaggle **CPU** kernel (U-Net++ r34 384 on about 340 images). That does not draw on the GPU quota.

### 3.2 RUB_ultrasound_study manual tracking (rank 2)
- **URL:** https://github.com/timvanderzee/RUB_ultrasound_study, `data/ultrasound/calf_raise/manual/Manual_Tracking_{BR,PT,TZ}.mat` (3.1 / 3.2 / 2.7 KB). There is also `TA_video/manual/*`, and DL_Track and UltraTimTrack outputs for the same video.
- **Video:** the DL_Track_US `calf_raise.mp4` (OSF 7mjsc, Apache-2.0). It is our UMUD dynamic benchmark: Telemed, 167 frames, every second frame analysed by 3 raters with a custom MATLAB script.
- **Paper:** van der Zee et al. 2025, PeerJ CS 11:e2636, CC BY. It reports 84 equidistant frames (every 2nd) and 3 observers.
- **Label format** (`processing/ultrasound/ManualTracking.m`):
  - `FasData.FAngle` = `atan2d` of the drawn fascicle line, i.e. against the **image horizontal**.
  - `FasData.FLength` is in px, measured on a frame padded by 200 px on each side.
  - `FasData.pts` holds the endpoints.
  - `ApoData.Angle` and `ApoData.pts` hold the drawn deep aponeurosis line.
  - A superficial line is drawn too but not saved.
- **Use:** compare the dynamic xlsx PA with `FAngle` and with `FAngle − ApoData.Angle`. This settles the "+10° vs deep aponeurosis" question in FACTS. Then either re-score the dynamic S_sel term with the correct convention or drop it.
- **Licence:** none in the repository. Use it only to decode the convention of the CC BY UMUD values; do not train on it or redistribute it.

### 3.3 UltraTimTrack GM dataset (rank 4)
- **Where:** https://doi.org/10.6084/m9.figshare.26502868.v1, CC BY 4.0, 8.66 GB. Paper: PeerJ CS 2025 e2636; code at github.com/timvanderzee/RUB_ultrasound_study (unlicensed).
- **Acquisition:** GM, left leg; **Telemed ArtUs EXT-1H + LV8-5N60-A2**, 60×50 mm FOV, 33–100 fps; 8 healthy young adults (4 women); ramp, sinusoid and passive ankle tasks.
- **Files:**
  - about 100 mp4 videos (6–120 MB each);
  - `fascicle_tracking_estimates.zip` (394 MB): per-frame **algorithm** FL and angle from UltraTrack, TimTrack and UltraTimTrack;
  - `dynamometry.zip` (258 MB).
- **No expert labels.**
- **Use:** in-domain unlabelled Telemed ArtUs frames for adaptation, and weak orientation priors from TimTrack. Also: the native FOV is known (60 mm), so its exports show how Telemed renders pixels.

### 3.4 FALLMUD (rank 5)
- **Where:**
  - https://kalisteo.cea.fr/index.php/fallmud/ (page `?p=3568`; zip 124.67 MB through the WPDM link `https://kalisteo.cea.fr/index.php/download/fallmud_dataset/`; the certificate chain is broken, so use `curl -k` or a browser);
  - Kaggle mirror `angeliqueloesch/fallmud`, about 135 MB.
- **Paper:** Michard et al. 2021, AW-Net, ACM BCB.
- **Contents:** 812 images in two author folders, each with **one fascicle mask and one aponeurosis mask**.
  - Cunningham et al. 2018 (J Imaging 4:29): 504 jpg, GM + soleus, Hitachi Aloka SSD-5000, 8 subjects, video frames at 25 fps.
  - Cronin: about 308 images, which is the DL_Track 2020 fascicle set (VL/GM/TA/SOL, 4 devices), **probably already in competition train**.
- **Licence is ambiguous:**
  - the CEA page states none;
  - the UMUD index says CC BY 4.0;
  - the Kaggle mirror says CC BY-NC 4.0, which could be a problem for a prize entry.
  - Ask CEA (the AW-Net authors) before using it.
- **Use:** extra masks for GM aponeurosis/fascicle segmentation. Hash- and correlation-dedupe the Cronin part against competition train first.

### 3.5 GasFasTrack (Drazan, Hullfish & Baxter 2019) (rank 6)
- **Where:** https://zenodo.org/records/2598553, CC BY 4.0, about 30 GB.
  - Subject zips are 3.6–7.0 GB each.
  - `validation_data.zip` is 1.92 GB: all tracked data, including manual.
  - `Sample Videos.zip` 591 MB; `matlab.zip` 0.5 MB.
- **Acquisition:** GM, **Telemed SmartUs** LV7.5/60/128Z-2, 60 Hz; 5 healthy adults; maximal isometric and isokinetic contractions at 30, 120, 210 and 500°/s. Raw data are Telemed `.tvd`, also converted to AVI/MP4.
- **Labels** (PeerJ 7:e7120; DUSTrack Sci Rep 2026 used the same data):
  - FL = point-to-point distance between the fascicle's intersections with both aponeuroses, extrapolated if needed;
  - **PA against the deep aponeurosis**;
  - 3 raters × 3 days, 18 frames per participant, stored as MATLAB.
- **Use:** multi-rater Telemed-family GM PA/FL. The protocol is close to UMUD's, but contraction states differ from rest. TimTrack dataset 3 already holds 100 frames of one subject from this set, so use that first.

### 3.6 DL_Track_US / LabTrainData (rank 7)
- **OSF 7mjsc:** `v0.2.1/training_images/Train_Data_DL_Track.7z`, 522 MB (https://osf.io/download/u2584/).
- **OSF xbawc "Example Dataset Structure":** `apo_images.zip` 305 MB (x53sw), `fasc_images.zip` 368 MB (5xvd4), `apo_masks.zip` 4.4 MB (jtn86), `fasc_masks.zip` 7.1 MB (gjexq). Apache-2.0, plus the DL_Track data-usage policy (research use, cite, no re-identification).
- **Contents:** about 570 apo and 310 fasc images; VL/GM/TA/SOL; Aloka Alpha-10, Telemed Echo Blaster 128, Philips HD11, Telemed ArtUs EXT-1H. Binary masks.
- **Use:** dedupe and provenance. Native-resolution originals could explain the 1200×800 stretch factors. Little new label content.

### 3.7 Others, briefly
- **NeuAge_YOUNG / NeuAge_OLD** (OSF bf8ut / x7nfw, CC BY 4.0): Esaote MyLab70, 47 mm probe; VL longitudinal static + QF panoramic; 46 / 88 subjects; 552 / 1408 images; no labels.
- **AnkleImage_2024** (https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/482/S-BIAD1482/Files/; Sci Data 2024, PMC11680872): S-Sharp Prodigy at 1000 Hz; TA/SOL/GM/GL; 10 healthy subjects; PA/FL/MT from UltraTrack optical-flow tracking, sometimes partial.
- **Rosa et al. 2021** (Dryad doi:10.5061/dryad.vq83bk3rp, CC0, 8.77 GB): soleus, Telemed ArtUs EXT-1H, `.mat` sequences for UltraTrack.
- **DeepMTJ** (CC BY 4.0): only the 1344-image test set is public; labels are MTJ points from 4 raters.
- **DeepACSA** (OSF a3u4v / Zenodo 5799204, Apache-2.0): transverse panoramic CSA, including Siemens Juniper RF/VL.
- **UMUD RF CSA benchmark** (OSF xbawc, 11 MB): 30 panoramic RF images.
- **SimpleMuscleArchitectureAnalysis** (Mendeley dpmf9bz8pt v2, CC BY 4.0, 53 MB):
  - SampleA: GM, Telemed LogicScan 128 EXT-1Z, 15 subjects.
  - SampleB: GM, Philips HD11XE, 15 scans.
  - Sample C: a TA sequence.
  - The two authors' manual values are **not** published; TimTrack dataset 4 provides 3-observer labels for A and B.

## 4. Not usable, or avoid
- **Ritsche 2022 raw images and Minghetti 2026 raw images.** Both are "on request", so not equally accessible. They are probably the test cohorts, which would contaminate the test set. Use the published tables only, and declare them as literature priors.
- **Kaggle `dreaddevelopment/umud-seg-corpus-v1`** (2.73 GB, labelled "CC0"): images only, "assembled for the UMUD challenge", with no documented provenance. It is probably re-hosted competition or DL_Track images (licence laundering of CC BY-NC-SA data). Do not use.
- **Kaggle `nagatakengo/umud-challenge-dataset`** (1.6 GB): a competition-data mirror, not external.
- **Records with tables only** (no images): La Trobe GM/GL reliability (2,145 digitised images but only `Raw Data.xlsx`); figshare architecture tables (basketballers, DMD, master runners, VL stretching); Dryad VL pregnancy and VL jumping.
- **Not found anywhere public:** CP muscle-architecture images, Philips Lumify muscle images, labelled longitudinal RF images.

## 5. Suggested next steps (none done; each download needs the user's OK)
1. **Now, at no cost:** keep the §2 table as a pre-registered sanity band for the Lumify and Siemens cells.
   - Use it as evidence against ungating Lumify RF FL, and for the C9 Lumify GM move.
   - Do not use it as a shrinkage target.
   - Ask the host in thread 748126 whether the Lumify labels were drawn on the provided 1200×800 PNGs.
2. **Tiny:** fetch the three `Manual_Tracking_*.mat` files (about 9 KB in total) to `<scratchpad>` and read them with `scipy.io.loadmat`, or h5py if they are v7.3.
   - Decide the PA convention of the dynamic benchmark.
   - If it is measured against the horizontal, re-score the dynamic S_sel term before the C7 / C9 decision.
3. **Small, CPU only:** run the TimTrack bundle (about 80 MB) through a Kaggle **CPU** kernel with our apo/fasc models. Then score PA (`alpha − gamma` where available; otherwise the fascicle angle against the horizontal, compared with our fragment angle) and MT in px against the 3-rater means.
4. **After the GPU reset, only if the Telemed risk stays the priority:** sample 10–20 UltraTimTrack videos (Telemed ArtUs) for unlabelled adaptation. FALLMUD comes only after its licence is clarified.

## Sources
- TimTrack: https://github.com/timvanderzee/ultrasound-automated-algorithm ; https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265752
- RUB study code: https://github.com/timvanderzee/RUB_ultrasound_study ; UltraTimTrack paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/ ; data: https://doi.org/10.6084/m9.figshare.26502868.v1
- GasFasTrack: https://zenodo.org/records/2598553 ; https://doi.org/10.7717/peerj.7120 ; DUSTrack: https://doi.org/10.1038/s41598-026-42795-3 , https://github.com/praneethnamburi/DUSTrack (MIT)
- SMA data: https://data.mendeley.com/datasets/dpmf9bz8pt/2 ; paper https://doi.org/10.1371/journal.pone.0229034
- FALLMUD: https://kalisteo.cea.fr/index.php/fallmud/ ; https://www.kaggle.com/datasets/angeliqueloesch/fallmud
- UMUD paper (Ritsche, Sarto et al. 2026, BMC Med Imaging 26:139): https://doi.org/10.1186/s12880-026-02170-0 (PMC12998083); UMUD repository and index entries: https://github.com/PaulRitsche/UMUD (`webapp/database_entries/*.json`); app: https://universalmuscledatabase.streamlit.app/
- OSF nodes: https://osf.io/xbawc/ (UMUD data repository), https://osf.io/7mjsc/ (DL_Track_US), https://osf.io/a3u4v/ (DeepACSA), https://osf.io/bf8ut/ and https://osf.io/x7nfw/ (NeuAge), https://osf.io/wgy4d/ (DeepMTJ)
- Lumify/Juniper agreement study: https://doi.org/10.3389/fphys.2022.981862 (PMC9478722)
- CP / Juniper VL study: https://doi.org/10.1186/s12891-026-09829-3 (PMC13281589)
- AnkleImage: https://doi.org/10.6019/S-BIAD1482 ; Rosa 2021: https://doi.org/10.5061/dryad.vq83bk3rp
- L12-4 aperture: Philips L12-4 product pages (34 mm, one page 38.4 mm); MSF catalogue (field of view 34.5 mm)
