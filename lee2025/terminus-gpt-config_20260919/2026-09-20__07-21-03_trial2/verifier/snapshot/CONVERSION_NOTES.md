# Dataset Conversion Notes

## Overview
- **Dataset**: Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping (provided paper/code/data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124
- Imports succeeded.

Directory contents:
- `.manifest` (file)
- `code` (directory)
- `CONVERSION_NOTES.md` (file)
- `data` (directory)
- `decoder.py` (file)
- `docker-compose.yaml` (file)
- `Dockerfile` (file)
- `methods.txt` (file)
- `paper.pdf` (file)
- `train_decoder.py` (file)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | `code/georepca1/src/utils.py:61` | LOADING | Loads an animal's preprocessed joblib dictionary; alternatively loads MATLAB with `mat73.loadmat` and converts `envs`, `position`, and `trace` to NumPy arrays. |
| `generate_behav_dict` | `code/georepca1/src/utils.py:130` | LOADING | Collects each animal's environments and position arrays for behavior/model analyses. |
| `get_env_mat` | `code/georepca1/src/utils.py:215` | PROCESSING | Converts each named arena geometry to a binary 3x3 matrix (1 accessible, 0 omitted/blocked). |
| `get_rate_maps` | `code/georepca1/src/utils.py:313` | PROCESSING | Spatially bins position and rise events into 15x15 maps, smooths numerator and occupancy with Gaussian sigma 1.5 bins, occupancy-normalizes, and scales to events/s at 30 fps. |
| `get_split_half` / `get_shuffle_split_half` | `code/georepca1/src/utils.py:356/393` | CURATION | Computes odd/even-minute split-half map reliability and shuffled null distributions. |
| `get_place_cells` | `code/georepca1/src/utils.py:415` | CURATION | Labels cells whose split-half reliability exceeds the shuffled threshold (`nsims=500`, alpha 0.05). |
| `clean_rate_maps` | `code/georepca1/src/utils.py:463` | CURATION | Masks spatial bins inconsistent with each named environment geometry. |
| `get_masked_maps` | `code/georepca1/src/utils.py:482` | CURATION | Applies registration/environment masks to precomputed maps for downstream RSM analyses. |
| `fit_decoder` / `test_decoder` | `code/georepca1/src/utils.py:1776/1806` | PROCESSING | Fits/tests Gaussian Naive Bayes position decoders after temporal aggregation (default 3 frames). |
| `decode_position_within` | `code/georepca1/src/utils.py:1845` | PROCESSING/CURATION | Within-day position decoding with running-frame and active-cell selection, K-fold CV, and snapping to valid arena bins. |

### Notes
- Repository documentation says each animal file contains `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, and `trace`. `trace` is already a rise-extracted binary calcium-event series (1 = significant event), so delta-F/F must **not** be recomputed for this conversion.
- `position` is x-y by frame for each recording day/session. `trace` stores the same temporal frames and registered cells; unregistered cells on a day are represented as NaN. `SFPs` and `centroids` likewise retain cross-day registration.
- `blocked` encodes blocked partition indices in row-major 3x3 order `[[0,1,2],[3,4,5],[6,7,8]]`; -1 means none blocked. `get_env_mat` provides equivalent named-geometry occupancy matrices.
- Reference sampling is 30 frames/s. Rate-map analyses use 15x15 spatial bins and Gaussian sigma 1.5 bins (2.5 cm as described by README). They sum event traces and occupancy separately before division.
- Reference position decoding uses 15x15 bins, Gaussian-smoothed speed (sigma 5 frames), speed >5 spatial units/s, cells with >5 events during selected running frames, 5-fold non-shuffled `KFold`, and 3-frame (0.1 s) temporal aggregation in GNB fitting/testing. The decoder predicts x and y jointly and reports Euclidean error.
- Place-cell classification exists for specific map analyses, but the raw decoder function does not restrict to place cells; it only applies its active-cell threshold. This distinction will be preserved when planning the requested neural decoder conversion.
- `clean_rate_maps`/valid-bin snapping concerns map-level analyses. For requested framewise 3x3 position classes, inaccessible bins should naturally have no valid samples; geometry remains a separate static decoder input.
- README and source consistently describe preprocessed calcium-event data rather than electrophysiology, so no spike-quality metrics or electrophysiological unit filtering apply.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is approximately 15 GB. It contains seven animal datasets in both compressed joblib (extensionless, used by reference `load_dat`) and duplicate MATLAB v7.3 `.mat` formats, plus `behav_dict` and precomputed analysis/model results. No data README is present; `/app/code/README.md` documents fields.
- Animal IDs: `QLAK-CA1-08`, `QLAK-CA1-30`, `QLAK-CA1-50`, `QLAK-CA1-51`, `QLAK-CA1-56`, `QLAK-CA1-74`, `QLAK-CA1-75`.
- Each joblib root is `{animal_id: dataset}` with fields:
  - `SFPs`: float64 `(35,35,n_registered_cells,n_days)` spatial footprints; NaN when a cell is absent on a day.
  - `blocked`: length-`n_days` nested list of blocked 3x3 partition indices; `[-1]` denotes no blocked partition.
  - `centroids`: float64 `(n_registered_cells,2,n_days)`; NaN for absent cells.
  - `envs`: Unicode `(n_days,1)` named geometries.
  - `maps`: `sampling (15,15,n_days)`, and `smoothed`/`unsmoothed (15,15,n_registered_cells,n_days)`.
  - `position`: float64 `(n_days,2,n_frames)` x-y coordinates, range approximately 0--75.
  - `trace`: float64 `(n_days,n_registered_cells,n_frames)` binary 0/1 rise events; an absent cell is NaN at every frame of that day.
- Neural and position frame counts match exactly for every session. Registration masks at first, middle, and last frames are identical for every animal/day, showing NaNs encode absent cells rather than invalid temporal periods. All positions are finite and all finite sampled traces are exactly 0 or 1.
- Frame rate is 30 Hz. Each animal has a fixed session length of 71,866--72,219 frames (39.926--40.122 min); lengths differ slightly across animals but not days within an animal.
- Environment schedule comprises ten shapes (`square`, `o`, `t`, `u`, `rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`) repeated across sequences. Across data, square has 27 sessions and every other shape has 20 (207 total).
- `n_registered_cells` is the cross-day union per animal (515--952). Day-present neurons are selected by finite trace registration and number 113--564/session. Summing cross-day unions gives 5,413 unique animal-specific cell identities; summing day-present neuron counts gives 69,744 session-neuron instances.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5,413 cross-day registered identities across animals; 69,744 day-present session-neuron instances |
| Neurons / session | 113--564; animal-level mean session count 213.9--405.0; overall approximately 337 |
| Subjects | 7 |
| Sessions / subject | 21 for QLAK-CA1-51; 31 for each other subject |
| Sessions (total) | 207 recording days |
| Native trials | Continuous day sessions; no native trial boundaries |
| Planned 1-min trials available | 8,187 complete non-overlapping minutes at 1,800 frames/min; 166,419 trailing frames would remain across sessions |
| Frames (total) | 14,903,019 aligned 30 Hz frames |
| Recording duration | 8,279.455 min total; approximately 40 min/session |
| Coordinate range | x,y approximately [0,75] |
| Finite trace values | Binary {0,1} |
| Mean population event rate | Animal means 0.212--0.260 events/s/cell |

### Per-animal sizes
| Animal | Sessions | Registered union | Mean present/session | Present range | Frames/session |
|--------|----------|------------------|----------------------|---------------|----------------|
| QLAK-CA1-08 | 31 | 515 | 213.9 | 153--254 | 71,866 |
| QLAK-CA1-30 | 31 | 875 | 381.0 | 336--422 | 71,866 |
| QLAK-CA1-50 | 31 | 942 | 400.9 | 214--564 | 71,866 |
| QLAK-CA1-51 | 21 | 554 | 230.0 | 113--323 | 72,219 |
| QLAK-CA1-56 | 31 | 862 | 380.9 | 251--529 | 72,091 |
| QLAK-CA1-74 | 31 | 713 | 312.3 | 258--405 | 72,060 |
| QLAK-CA1-75 | 31 | 952 | 405.0 | 263--535 | 72,071 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (registered identities) | Mean 773 ± 68 SE/animal; range 515--952 | Paper Results: “mean number of cells per animal = 773 ± 68 SE, minimum ... 515, maximum ... 952.” |
| Neurons / session | Not numerically stated | Cells tracked across sessions; day absence represented in source data. |
| Subjects | 7 inferred/confirmed by deposited animal files | Seven C57BL/6 animal IDs in deposited data; paper resource table specifies C57BL/6 mice. |
| Sessions / subject | Up to 31; one animal has 21 in deposited data | Results: geometry sequence repeated “up to three times”; Methods: one session/day. |
| Sessions (total) | 207 in deposited data | Consistent with six ×31 plus one ×21. |
| Session duration | 40 min | Methods: “All sessions were 40 min, and one session was recorded per day.” |
| Neural/behavior time bin | 1/30 s native frame | Methods: behavior and cellular imaging streams acquired simultaneously at 30 Hz and timestamped. |
| Arena | 75×75 cm, conceptual 3×3 grid of 25×25 cm partitions | Figure 1/Methods. |
| Rate-map spatial bin | 5×5 cm (15×15 grid); 2.5 cm Gaussian kernel | Paper map methods/README/reference code (`n_bins=15`, sigma 1.5 bins). |
| Reward rate | N/A | Free navigation; no trial rewards described. |
| Place-cell criterion | Split-half map correlation >99th percentile of 1,000 circular position shuffles | STAR Methods. |
| Position decoder | 5-fold Gaussian Naive Bayes; flat prior; held-out Euclidean error | STAR Methods. |

### Processing Details
- Miniscope calcium video and overhead behavioral video were acquired simultaneously at 30 Hz and timestamped for post-hoc alignment.
- Source arena was 75×75 cm; inserted 25 cm walls create named geometries on an imagined 3×3 grid. Geometry order was randomized between mice but repeated within a mouse, beginning/ending with square.
- Calcium processing: motion correction, cell segmentation/transient extraction; median-subtracted calcium derivative smoothed with Gaussian SD 5 frames, noise estimated from negative derivative values via a half-normal model, noise z-scored, then thresholded at z>2.5. Result is a binary rising-phase vector and is the “firing rate” used in all analyses.
- Position is head location from DeepLabCut. Cross-session cell registration used surface landmarks, spatial footprints, and centroids.
- Rate maps divide summed events by occupancy and use 15×15 bins; reference code scales binary events by 30 Hz to events/s.
- Paper's position decoder uses spatially binned position, 5-fold held-out Gaussian Naive Bayes, equal position prior, sklearn variance smoothing 1e-9, and Euclidean error. Reference code additionally documents running speed >5 cm/s, Gaussian speed smoothing sigma 5 frames, >5 events/cell, and 3-frame aggregation.

### Curation Steps

**Neuron curation rules**:
- Motion correction and spatial footprints were manually inspected; lens artifacts were manually removed before deposited preprocessing.
- Cells are registered across sessions. A cell absent on a day is all-NaN and must not be included in that session.
- Paper explicitly says reliability results “motivated the inclusion of all cells in subsequent analyses.” Therefore place-cell filtering is analysis-specific, not a general dataset filter.

**Trial curation rules**:
- Native recordings are continuous 40-minute sessions, not event-aligned trials.
- No invalid-time mask or omitted days is described. Deposited position is finite and day-present neural traces are finite throughout.
- Requested downstream format requires non-overlapping one-minute trials; handling of sub-minute session tails will be planned in Step 5.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| 2-D animal position | Paper reports significant decreasing error over sessions (ANOVA p<0.0001, F=7.9845) and says late-session accuracy reaches the maximum in recent work. Exact cm values are figure-only/not extractable from text. |

### Source quotes relevant to conversion
- “All sessions were 40 min, and one session was recorded per day.”
- “The final binarized rising-phase vector was then set to 1 whenever this z-scored ... vector exceeded 2.5, and 0 otherwise.”
- “All analyses were conducted using the binary vector ... treating this vector as if it were the firing rate.”
- “Behavioral and cellular imaging streams [were acquired] at 30 Hz ... timestamped for post-hoc alignment.”

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Registered cell count | Loads cross-day registered union | Mean 773.29, range 515--952 | Mean 773 ±68 SE, range 515--952 | Exact match (rounding). Use only finite/day-present cells per target session to avoid NaNs, retaining all such cells. |
| Recording length/rate | Defaults `fps=30`; split-half logic assumes 20-min halves | 39.926--40.122 min, animal-specific fixed frame count | 40 min at 30 Hz | Minor acquisition-frame variation around nominal duration. Use actual aligned frames; construct complete exact 1-min chunks. |
| Position alignment | `position` and `trace` loaded together | Exact frame-dimension equality all 207 sessions | Streams simultaneously acquired/timestamped | Preserve same frame indices for neural and output; no interpolation needed. |
| Neural representation | Uses deposited `trace` | Finite values exactly {0,1} | z>2.5 rising-phase binary vector used as firing rate | Use deposited trace unchanged before time binning; do not compute dF/F/deconvolution. |
| Place-cell shuffles | `get_place_cells` defaults `nsims=500`, alpha=.05; helper defaults 1,000 | Precomputed SHR outputs available | 1,000 shuffles, 99th percentile | Paper criterion governs if identifying place cells, but requested decoder and paper population analyses include all cells, so no place-cell filter is applied. |
| Decoder spatial bins | Reference decoder defaults 15×15 | Maps are 15×15; raw coordinates 0--75 | Published decoder is spatially binned; task explicitly requires 3×3 | Task requirement overrides only output discretization. Map each coordinate directly to 25 cm grid bins. |
| Decoder temporal bin | Reference decoder aggregates 3 frames (0.1 s) | Native 30 Hz | Acquisition 30 Hz; paper does not explicitly state aggregation | Use reference decoder's 3-frame binning for neural/output to match decoding processing and reduce dataset size, unless validator behavior indicates otherwise. |
| Geometry | `get_env_mat` maps names to accessible 3×3 matrices | `blocked` indices exactly match names | Arena uses blocked partitions in 3×3 design | Construct static 9-vector with 1 accessible/0 blocked directly from `blocked`; cross-check against `get_env_mat`. |
| Numeric expected decoder accuracy | Code reports Euclidean error | `within_decoding` is precomputed | Paper gives trend/significance, no extractable numeric table | Requested validator reports categorical accuracy, so compare against 1/9 chance and verify robustly above chance rather than equating to paper's cm error. |

### Final Understanding
- Seven mice contributed 207 continuous, approximately 40-minute CA1 calcium imaging sessions at 30 Hz.
- The deposited joblib arrays are already motion-corrected, artifact-curated, cross-session registered, behavior-aligned, and binarized into significant rising events.
- Every recording day is one target “session.” All finite/day-present CA1 cells are retained; all-NaN absent registered cells are removed independently per day.
- Neural and position streams require identical slicing only, not resampling/interpolation. Requested one-minute trials will be contiguous chunks within each day.
- Static arena geometry is represented by the nine accessible/blocked partitions. Time-varying output is one categorical 3×3 position class in row-major order.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day, present_cells, :]` | `neural[session][trial]` | Select cells finite on that day; Gaussian smooth time axis sigma=3 native frames; average non-overlapping groups of 3; transpose to neuron×time; slice exact 600-bin minutes | `fit_decoder`, `test_decoder` | float32, 100 ms bins; all curated/day-present CA1 cells retained. |
| `blocked[day]` / `envs[day]` | `input[session][trial]` | 9-element row-major binary accessibility vector: initialize ones, set blocked indices to zero | `get_env_mat` | Static 1-D input repeated for each minute. Values match reference geometry matrices. |
| `position[day, x/y, :]` | `output[session][trial]` | Average matching 3-frame groups; clip x,y to [0,75]; divide by 25 and floor/clip to bins 0--2; class=`y_bin*3+x_bin`; slice 600-bin minutes | `fit_decoder`, `test_decoder`; task-required 3×3 override | Shape `(1,600)`, integer classes 0--8. Rare tracking points in blocked bins snap to nearest accessible bin. |
| Animal ID | `subjects`, `subject_idx` | Sorted unique IDs; day-session maps to its animal index | `load_dat` | 7 subjects full data. |
| Recording site | `brain_regions`, `brain_region_idx` | `brain_regions=['CA1']`; zero vector per session neuron set | Paper/data ID | All recordings are dorsal CA1. |
| Environment/day/frame info | `metadata.session_info` | Per-session animal, source day, environment, native frames, retained neurons, trials, and discarded tail | N/A | Enables provenance and tail accounting. |

### Key Decisions
1. **Target session definition**: Each original recording day is one target session. This preserves a constant simultaneously recorded neuron set and the paper's within-session decoder framing.
2. **One-minute trials**: Use contiguous, non-overlapping exact minutes. At 100 ms/bin each trial has 600 timepoints. Drop only the final sub-minute tail of each day; never pad or create a short “one-minute” trial.
3. **Temporal processing**: Match reference position decoder: Gaussian-filter neural traces with sigma 3 native frames, then average neural and position over stride-3 windows. Process the whole day before trial slicing to avoid artificial filter boundaries.
4. **Neuron curation**: Keep every cell registered/present on that day and remove only all-NaN absent cells. Do not restrict to place cells because the paper explicitly includes all cells and the reference position decoder uses an activity threshold rather than place-cell labels. Deposited cells are already manually artifact-curated.
5. **Spatial discretization/orientation**: Use 25 cm bins and row-major `y*3+x`. Direct testing against all original blocked masks found 445/14,903,019 frames (0.0030%) in blocked classes versus 17.12% under the wrong x-major convention, decisively confirming orientation.
6. **Blocked-bin artifacts**: Snap the very rare blocked output to the nearest accessible 3×3 cell after temporal averaging, matching the spirit of reference `decode_position_within`, which snaps actual/predicted positions to valid map bins. This avoids contradictory geometry/output labels caused by boundary/tracking noise.
7. **Geometry encoding**: Use 1=accessible and 0=blocked, matching `get_env_mat`. Input names explicitly state accessibility to prevent semantic ambiguity.
8. **Neural scale/dtype**: Store smoothed mean binary-event activity as float32. Values represent event probability per 100 ms frame grouping, not events/s; this exactly follows the reference decoder's filtering/pooling rather than rate-map scaling.
9. **Outputs**: One time-varying categorical output named `position_bin`; output labels describe row-major locations (`bottom-left` through `top-right`, based on increasing source y). Numeric values are 0--8.
10. **Alignment**: Apply identical 3-frame windows and minute slices to neural and position arrays. No interpolation is required because source dimensions match exactly.

### Planned Sanity Checks
- [x] Directly loaded every original joblib and verified neural-position frame equality and finite registration masks at first/middle/last frames.
- [x] Verified finite trace values are binary {0,1} and coordinates span approximately 0--75.
- [x] Verified data registered-cell statistics exactly match paper (mean ~773, range 515--952).
- [x] Tested both flattening orientations against blocked partitions; `y*3+x` leaves only 0.0030% raw tracking artifacts.
- [ ] `np.allclose` converted neural spot-check against independently smoothed/pooled original trace.
- [ ] `np.allclose` geometry vector against independently constructed original `blocked` mask and reference `get_env_mat`.
- [ ] `np.allclose` output classes against independently pooled/discretized original position.
- [ ] Check every converted trial is `(n_neurons,600)`, input is `(9,)`, output is `(1,600)`, finite, and aligned.
- [ ] Check every output is 0--8 and never a blocked class for its trial geometry.
- [ ] Reconcile retained trial/frame/session/neuron counts against direct source-derived totals.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py` with required invocation and mutually exclusive `--full` (default behavior) / `--sample` modes plus `--show-processing`.
- Script discovers the seven source joblib animal files, loads one animal at a time, processes each day independently, and validates all target shapes/values before pickle serialization.
- Implemented reference decoder processing: Gaussian smoothing sigma=3 native frames, non-overlapping 3-frame mean pooling, and identical behavior pooling.
- Implemented day-present cell filtering, static 3×3 geometry, row-major 3×3 position output, nearest-valid correction, exact-minute trial slicing, metadata provenance, and processing plots.
- `python3 -m py_compile /app/convert_data.py` passed.
- Smoke test (`--sample`, two sessions) completed without errors: 78 trials, 30.23 MiB, 9.44 s total; source load 8.93 s and processing 0.16--0.21 s/session.

Code inefficiencies identified:
- Compressed source loading is slower than per-session processing.
- Full output is intrinsically large because it retains all day-present cells across 8,187 one-minute trials.

Code speedups added:
- Sequential per-animal loading with garbage collection bounds source-memory use.
- Vectorized Gaussian filtering, reshape-based pooling, discretization, and blocked-bin correction.
- float32 neural matrices and int8 inputs/outputs minimize target size.
- Session-wide filtering occurs once before slicing, avoiding repeated trial computations and filter-boundary artifacts.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 338 session-neuron instances |
| Neurons / session | 185, 153 |
| Subjects | 7 in vocabulary; sample sessions reference QLAK-CA1-08 |
| Sessions / subject | 2 sampled day-sessions |
| Trials (total) | 78 |
| Trials / session | 39, 39 |
| Trial shape/time | 600 bins ×100 ms = 60 s |
| Neural range | finite float32; nonnegative smoothed mean event activity |
| Input range | [0,1], static 9-element accessibility mask |
| Output distribution | [0.104, 0.062, 0.135, 0.102, 0.044, 0.104, 0.104, 0.117, 0.229] for classes 0--8 |
| Output range | integer classes [0,8] |
| Blocked-output violations | 0 |

### Processing Plots Review
- Created `processing_QLAK-CA1-08_day00.png` and `processing_QLAK-CA1-08_day01.png`.
- Plots include raw x-y trajectory, 3×3 accessibility mask, raw binary events, temporally processed neural activity, aligned position classes, and class counts colored by accessibility.
- Square session shows all partitions accessible. O-shaped session shows center blocked and zero center output samples. No alignment or discretization anomaly was detected.

### Format Validation
- Required `verification_sample_out.txt` created by `train_decoder.py --verify-only`.
- Errors: None.
- Warnings: None.
- Validator confirms 600 timepoints/trial, 185/153 neurons, input range 0--1, output range 0--8, all nine classes overall, and CA1 region assignment.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| One source load for multiple days | Avoids repeated ~9--23 s decompression per session |
| Vectorized whole-session filtering/pooling | ~0.2 s/session without plotting |
| float32/int8 output | Approximately halves neural storage versus float64 |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Source loading | ~9--23 s/animal | ~2 min for seven animals |
| Session conversion | ~0.2 s (0.6--1.0 s with plot) | <1 min for 207 sessions without plots |
| Full save | Estimated several minutes for ~8 GB | Total conservatively <10 min, below 15-min threshold |

Required sample conversion completed in 10.56 s and produced a 30.23 MiB pickle.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| position_bin | 0.4376 | 0.3659 |

- Uniform chance is 0.1111; validation accuracy is 3.29× chance.
- Training loss decreased monotonically from 2.279124 (epoch 1) to 1.717718 (epoch 200); test loss was 1.797686.
- Training/validation accuracy ratio is 1.20, below the 1.5× overfitting concern threshold.
- `train_decoder_sample_out.txt` was created and training finished successfully.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6.17 GiB (6,320.40 MiB)
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created
- Full conversion runtime: 170.32 s after final dtype correction.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total registered identities | Mean 773 ±68 SE/animal; range 515--952 | Cross-day registration | 5,413 union sum; mean 773.29, range 515--952 | Provenance retained; day-present cells selected | Yes |
| Mean neurons/session | Not stated | Day-specific finite cells | 336.93 | 336.93 | Yes |
| Session-neuron sum | Not stated | Exclude all-NaN absent cells | 69,744 | 69,744 | Yes |
| Neurons/session range | Not stated | Day-specific finite cells | 113--564 | 113--564 | Yes |
| Subjects | 7 deposited mice | Animal files | 7 | 7 | Yes |
| Sessions | One/day, up to three sequences | Each day array | 207 | 207 | Yes |
| Trials (total) | Continuous 40-min sessions | N/A | 8,187 complete exact minutes | 8,187 | Yes |
| Trials/session | Nominal 40-min sessions | Actual frames | 39 or 40 | 93 sessions ×39; 114 ×40 | Yes |
| Time bins/trial | Task requires 1 min | 3-frame pooling at 30 Hz | 600 expected | 600 | Yes |
| Input range | 3×3 blocked geometry | `get_env_mat`: 0/1 | [0,1] | [0,1] float32 | Yes |
| Output distribution | Not reported for task bins | N/A | Raw aligned position | [0.09966,0.09853,0.13512,0.07541,0.05723,0.07685,0.11570,0.14121,0.20029] | Plausible; geometry-dependent |
| Output range | Task: 9 classes | N/A | 3×3 bins | integer 0--8 | Yes |
| Geometry/output violations | Should be none | Valid-map snapping | 152 pooled tracking artifacts identified | 0 after correction | Yes |

### Validation outcome and iteration
- Initial full validator run exposed repeated warnings that static inputs were int8 rather than expected float32. This was a formatting inefficiency, not a semantic mismatch.
- Fixed `geometry_vector` to emit float32, regenerated both sample and full pickles, and reran both validators.
- Final sample and full logs report: “Data format is valid, no errors or warnings.”
- Full validator confirms 207 sessions, 8,187 trials, 600 timepoints/trial, seven subjects, one CA1 region, and 69,744 session-neuron instances.
- No data loss beyond explicitly documented trailing sub-minute frames; all complete one-minute segments are retained.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Final `verification_full_out.txt` begins “Data format is valid, no errors or warnings” and ends “Data verification complete.” Initial int8-input warnings were fixed by emitting float32 static inputs and regenerating/revalidating both pickles.
2. **Independent original-data `np.allclose` checks**: `/app/cache/critical_checks.py` directly loaded original `QLAK-CA1-08` and the converted pickle without importing conversion functions. Sessions/trials `(0,0)`, `(0,5)`, and `(1,38)` were independently reconstructed. Neural Gaussian smoothing + 3-frame pooling passed `np.allclose(rtol=1e-6, atol=1e-7)`; geometry and output passed exact `np.allclose(rtol=0, atol=0)`.
3. **Neural sanity check**: Direct source traces were selected with original finite registration masks, independently Gaussian-filtered at sigma 3, mean-pooled by 3, and compared across full selected trials. All values matched and were finite float32.
4. **Input sanity check**: Original nested `blocked` values were independently converted to a nine-element float32 accessibility mask. Exact equality passed; all converted outputs index accessible entries.
5. **Output sanity check**: Original x-y position was independently 3-frame averaged, mapped with `floor(coord/25)`, clipped, flattened as `y*3+x`, and valid-bin corrected. Exact equality passed for beginning, middle, and last sample trials.
6. **Key statistics**: Source-derived and converted counts match exactly: 7 subjects, 207 sessions, 8,187 complete minutes, 69,744 session-neuron instances, 113--564 neurons/session. Paper registered-cell mean/range also match exactly after rounding.
7. **Edge cases**: Checked coordinate value 75 clipping to bin 2; `blocked=-1`; 39-vs-40-trial sessions; sub-minute tails (all 0--1,799 native frames); first/last trial slicing; animal boundary and 21-session animal; all-NaN absent cells; and rare blocked tracking points. No off-by-one failure found.

### Reference Code Comparison
| Processing step | Conversion implementation | Reference implementation | Comparison/result |
|-----------------|---------------------------|--------------------------|-------------------|
| Data loading | `joblib.load(file)[animal]` | `load_dat` joblib branch | Same source files and nested animal dictionary. |
| Neuron/trial filtering | Finite/day-present cell mask; all complete minute chunks | Decoder uses day cells with >5 running events; paper later includes all registered cells | All day-present curated cells retained, justified by paper and downstream general neural decoder. Only incomplete tails removed to satisfy exact 1-min trials. |
| Temporal alignment | Identical frame windows for trace/position before slicing | Simultaneous timestamped streams; decoder applies pooling to each | Same aligned source indices; no interpolation or shift. |
| Binning | Neural Gaussian sigma=3 then stride-3 average; position stride-3 average | `fit_decoder`/`test_decoder`: same `gaussian_filter1d(...sigma=3)` and `AvgPool1d(3,3)` | Logic matched. Requested 3×3 output replaces reference 15×15 spatial output only. |
| Input construction | `blocked` → row-major 9-vector, 1 accessible/0 blocked | `get_env_mat` named geometry matrices | Exact semantics; direct orientation test strongly supports row-major mapping. |
| Output construction | pooled x-y → 25 cm bins → `y*3+x`; nearest valid correction | Average-pools coordinates, casts to integer; `decode_position_within` snaps to nearest valid spatial bin | Same temporal aggregation and valid-bin principle; coarser 3×3 classes required by task. |

### Issues Found and Resolved
- **Input dtype warning**: Geometry was initially int8. Validator expected float32 and warned it would convert during training. Changed geometry creation to float32, regenerated sample/full outputs, and reran all validation. Final logs have no warnings.
- **Rare blocked-bin tracking artifacts**: Raw orientation analysis found 445/14,903,019 native frames in blocked bins, primarily one bit-donut partition. After temporal pooling 152 bins required correction. Snapped to nearest accessible class, consistent with reference valid-map snapping; final global check confirms zero contradictions.
- **No further mismatch**: Independent checks and all global structural assertions pass. `/app/cache/critical_checks_out.txt` records the successful results.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Training loss decreased smoothly from 2.327492 (epoch 1) to 1.159876 (epoch 200).
- Test loss: 1.136346.
- Device: CUDA.
- Split: 6,531 training trials and 1,656 validation trials.
- `--plot-samples` completed and generated sample/prediction plots.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| position_bin | 0.6925 | 0.6053 | Uniform chance 0.1111; validation is 5.45× chance; train/validation ratio 1.14. |

- Full 200-epoch execution finished successfully; `/app/train_decoder_full_out.txt` is complete.
- Accuracy greatly exceeds chance and the sample result (0.3659), consistent with correct temporal alignment and the larger full training set.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Accuracy | Expectation from Paper |
|----------|-------------------|------------------------|
| position_bin | Validation balanced accuracy 0.6053; training 0.6925 | Paper uses a different metric: 15×15 Gaussian-Naive-Bayes Euclidean error (cm), reports significant improvement over sessions (ANOVA p<0.0001, F=7.9845), but no numeric mean accuracy/error in extractable text. |

### Checks Performed
1. **Accuracy vs chance**: Uniform chance is 1/9=0.1111. Validation accuracy 0.6053 is 5.45× chance and training accuracy 0.6925 is 6.23× chance. Validation greatly exceeds both chance and the 1.5×-chance review threshold.
2. **Accuracy comparison to paper**: The paper reports Euclidean error in centimeters for a 15×15 Gaussian Naive Bayes decoder, whereas the required validator reports balanced classification accuracy for nine 3×3 classes using its neural architecture. These values cannot be numerically equated. The deposited reference `within_decoding` results were inspected to verify the source decoder is successful and improves over sessions; our strong above-chance result is directionally consistent. No categorical accuracy is reported in the paper.
3. **Train-validation gap**: Ratio 0.6925/0.6053=1.144 and absolute gap=0.0872. This is below the >1.5× concern criterion and does not indicate severe overfitting or leakage.
4. **Raw output verification**: Step 10 independently checked original position for three specific trials `(session,trial)=(0,0),(0,5),(1,38)` and exact `np.allclose` passed.
5. **Temporal alignment**: Neural and output were reconstructed from identical source-frame windows; independent allclose passed. Processing plots and full `sample_trials.png`/`predictions.png` provide visual alignment checks. No shift/interpolation exists in code.
6. **Output variation**: All nine classes occur globally. Fractions are [0.09966, 0.09853, 0.13512, 0.07541, 0.05723, 0.07685, 0.11570, 0.14121, 0.20029]; no class is near 99%. Every session contains multiple accessible classes.
7. **Neural filtering**: Confirmed deposited artifact-curated binary events, day-present cell filtering, no accidental all-NaN neurons, and reference-matched Gaussian smoothing/3-frame pooling. Paper supports inclusion of all cells.
8. **Loss behavior**: Full loss decreases smoothly over all 200 epochs (2.327492→1.159876); test loss 1.136346 is comparable/slightly lower, supporting stable generalization.

### Paper/reference metric context
- Paper's only text-reported positional decoder statistic is the session effect: ANOVA p<0.0001, F=7.9845, plus the qualitative statement that late recordings reached contemporary maximum decoding accuracy.
- The deposited `within_decoding` file contains per-animal, per-session, five-fold Euclidean errors, confirming the reference metric differs from this task's categorical balanced accuracy.
- Therefore no same-unit paper accuracy exists for a direct table entry; this is a metric/task-resolution difference rather than an excuse for low performance. Our 0.6053 categorical validation accuracy is robustly high.

### Issues Found and Resolved
- No new conversion issue was identified by accuracy review.
- Input dtype warnings had already been resolved in Step 10, and the final full decoder used the warning-free regenerated pickle.
- Accuracy is high, all classes vary appropriately, alignment checks pass, and the generalization gap is acceptable; no iteration of conversion logic is warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading example, field specification, processing summary, statistics, and decoder results.
- [x] `cache/` folder created.
- [x] Investigation scripts and extracted summaries moved to `cache/`.
- [x] `cache/README_CACHE.md` documents cached files.
- [x] Required conversion, validation, and training logs retained at `/app`.
- [x] `CONVERSION_NOTES.md` reviewed and completed through all workflow steps.
- [x] Final required-file and status audit passed.

