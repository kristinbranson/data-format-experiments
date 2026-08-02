# Dataset Conversion Notes

## Overview
- **Dataset**: CA1 cognitive mapping dataset from "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping"
- **Date started**: 2026-03-11
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `__pycache__/`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | `code/georepca1/src/utils.py` | LOADING | Loads one animal dataset from `data/` either from MATLAB or joblib; joblib is the default reference path. |
| `generate_behav_dict` | `code/georepca1/src/utils.py` | LOADING | Extracts lightweight per-animal behavior and environment metadata (`position`, `envs`, map shape). |
| `get_env_mat` | `code/georepca1/src/utils.py` | PROCESSING | Converts an environment label (`square`, `o`, `t`, `u`, `rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`) into a binary 3x3 occupancy/geometry matrix. |
| `get_rate_maps` | `code/georepca1/src/utils.py` | PROCESSING | Builds 15x15 event-rate maps from position and calcium-event traces; bins position, smooths with Gaussian filter, divides by occupancy, multiplies by 30 Hz. |
| `get_split_half` | `code/georepca1/src/utils.py` | PROCESSING | Computes split-half reliability from first vs second half rate maps. |
| `get_shuffle_split_half` | `code/georepca1/src/utils.py` | PROCESSING | Circularly shuffles position relative to traces to build a shuffle null for split-half reliability. |
| `get_place_cells` | `code/georepca1/src/utils.py` | CURATION | Calls split-half and shuffled split-half to mark place cells by p-value threshold (`alpha=0.05`). |
| `get_shr_within` | `code/georepca1/src/utils.py` | CURATION | Applies place-cell reliability analysis day-by-day within one animal. |
| `clean_rate_maps` | `code/georepca1/src/utils.py` | PROCESSING | Masks 15x15 rate maps outside the valid environment footprint derived from the 3x3 geometry. |
| `fit_decoder` | `code/georepca1/src/utils.py` | PROCESSING | Temporal-bins behavior and traces (3-frame pooling by default), smooths traces, converts binned x-y position to class labels, fits Gaussian Naive Bayes. |
| `test_decoder` | `code/georepca1/src/utils.py` | PROCESSING | Applies the trained decoder to temporally binned traces and reconstructs predicted x-y bins. |
| `decode_position_within` | `code/georepca1/src/utils.py` | PROCESSING | Main within-session reference decoder: 15x15 spatial bins, velocity filtering, cell activity thresholding, 5-fold CV, GaussianNB. |

### Notes
- `code/README.md` states the dataset fields per animal: `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`.
- The neural signal is not raw fluorescence and does not require delta F/F computation in the reference code. The README explicitly describes `trace` as rise-extracted calcium traces where `1` indicates a significant event.
- The reference code uses those preprocessed `trace` event matrices directly for rate-map generation and decoding.
- Spatial maps in the paper/code use 15x15 bins derived from the continuous position stream.
- Environment geometry is represented by named environments and converted to a 3x3 binary matrix with `get_env_mat`; this is directly relevant to the decoder input required for the converted dataset.
- The reference decoder in `main.py` decodes position within each recording day using `decode_position_within(dat[animal]['position'].T, dat[animal]['trace'].T, dat[animal]['maps']['smoothed'])`.
- The reference decoder applies additional online curation during decoding:
  - velocity filter: only keep time points with smoothed speed above `v_thresh=5`
  - cell filter: only keep cells with more than `cell_threshold=5` events during those retained time points
- Separate place-cell curation exists for map reliability analyses (`get_place_cells` / `get_shr_within`), but the position decoder shown in `main.py` does not pre-filter to place cells before decoding.
- Cells absent on a given day appear as `NaN` in per-day traces/maps; this is the main day-specific registration/quality indicator visible in the code.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains 7 primary subject datasets, each available in two formats:
  - joblib file with subject name only, e.g. `data/QLAK-CA1-08`
  - MATLAB file, e.g. `data/QLAK-CA1-08.mat`
- `data/behav_dict` is a lighter derived file with behavior/environment metadata.
- `data/precomputed_results/` contains cached analysis outputs from the reference code and is not the primary source for conversion.
- Each primary joblib dataset is a dictionary keyed by subject ID, with fields:
  - `SFPs`: spatial footprints, shape `(35, 35, n_registered_cells, n_sessions)` in the example animal
  - `blocked`: Python list of length `n_sessions`; each entry stores blocked partition IDs in the 3x3 arena indexing scheme, with `-1` meaning no blocked partitions
  - `centroids`: cell centroid positions, shape `(n_registered_cells, 2, n_sessions)`
  - `envs`: session environment labels, shape `(n_sessions, 1)` with 10 distinct labels
  - `maps`: dict with `sampling` `(15, 15, n_sessions)`, `smoothed` `(15, 15, n_registered_cells, n_sessions)`, `unsmoothed` same shape
  - `position`: continuous x-y position, shape `(n_sessions, 2, n_frames)`
  - `trace`: rise-extracted calcium event traces, shape `(n_sessions, n_registered_cells, n_frames)` with `NaN` for cells not registered on a given session
- Example from `QLAK-CA1-08`:
  - `position.shape = (31, 2, 71866)`
  - `trace.shape = (31, 515, 71866)`
  - `maps['smoothed'].shape = (15, 15, 515, 31)`
- Unique environment labels observed directly in the data:
  - `square`, `o`, `t`, `u`, `rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`
- Session lengths are almost fixed:
  - minimum `71866` frames
  - maximum `72219` frames
  - mean `71995.26` frames
- At 30 Hz this is about 40 minutes per session, which implies 39-40 non-overlapping 1-minute windows per session when split for the decoder task.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5413 registered cells across subject files |
| Neurons / session | 336.93 mean present cells/session (min 113, max 564) |
| Subjects | 7 |
| Sessions / subject | mean 29.57 (31 for 6 animals, 21 for QLAK-CA1-51) |
| Trials (total) | 8187 full 1-minute windows if sessions are split into non-overlapping 1800-frame trials |
| Trials / session | mean 39.55 full 1-minute windows/session (min 39, max 40) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 unique neurons | "5,413 unique neurons across 207 sessions in 10 geometries" |
| Neurons / session | 69,744 / 207 = 336.93 mean rate maps per session | "forming 69,744 rate maps" |
| Subjects | 7 inferred animals | "mean number of cells per animal = 773 ± 68 SE, minimum cells per animal = 515, maximum cells per animal = 952" and the dataset totals are consistent with 7 subject files |
| Sessions / subject | up to 31 days / sessions per animal | "up to three total repetitions (31 days)" and "All sessions were 40 min, and one session was recorded per day" |
| Trials (total) | No native trial structure in the paper; sessions are continuous 40 min recordings | "All sessions were 40 min" |
| Trials / session | N/A in reference; will be derived as 1-minute windows for target format | "All sessions were 40 min" |
| Neural data time bin | 33.33 ms raw frame bin (30 Hz) | "behavioral and cellular imaging streams at 30 Hz" |
| Behavior data time bin | 33.33 ms raw frame bin (30 Hz) | "behavioral and cellular imaging streams at 30 Hz" |
| Reward rate | N/A | No reward variable described in methods or copied text |
| Arena size | 75 cm x 75 cm square base environment | "The full square environment was 75 cm x 75 cm" |
| Geometries | 10 distinct geometries | "207 sessions in 10 geometries" |
| Session duration | 40 min | "All sessions were 40 min" |


### Processing Details
- Acquisition/alignment:
  - Behavioral and cellular imaging streams were simultaneously acquired at 30 Hz and timestamped for post-hoc alignment.
  - Position was obtained from DeepLabCut head tracking.
- Neural preprocessing:
  - Motion correction, cell segmentation, and transient extraction were performed before release of the dataset.
  - The rising phase of each calcium transient was extracted by taking the derivative of the filtered calcium trace, smoothing with a Gaussian kernel of standard deviation 5 frames, estimating noise from the negative derivative component, z-scoring, and thresholding at 2.5.
  - The final signal used for analysis is binary: 1 when the z-scored rising-phase signal exceeds 2.5, else 0.
  - The paper explicitly states that this binary vector is treated as the firing rate in all further analyses.
- Spatial coding analysis:
  - Place-cell identification uses split-half rate-map correlation between the first and second 20-minute halves of each 40-minute session.
  - Shuffle null uses 1000 circular shuffles of position relative to activity.
- Decoder details:
  - Position decoding is performed within each session.
  - Position is spatially binned and converted to a one-hot representation.
  - A 5-fold split is used.
  - The decoder is Gaussian Naive Bayes with flat priors and default variance regularization.
  - Decoding error is Euclidean distance between predicted and actual spatial bins on held-out data.

### Curation Steps

**Neuron curation rules**:
- Motion-corrected imaging data were manually inspected to ensure motion correction did not introduce artifacts.
- Spatial footprints were manually verified to remove lens artifacts.
- Cells were tracked across sessions using landmarks, spatial footprints, and/or centroids.
- For main analyses, the paper states that all cells were included in subsequent analyses because recordings showed high spatial reliability.
- Place-cell classification is an analysis-specific label, not a prerequisite for the primary position-decoding analysis.

**Trial curation rules**:
- The reference experiment does not define trialized behavior; sessions are continuous 40-minute recordings.
- For place-cell analysis the session is split into first and second 20-minute halves.
- For Bayesian position decoding the analysis is within-session, using spatially binned continuous position and trace data on held-out folds.

### Decoders Trained
| Decoded variable | Accuracy |
| Position | Paper reports decreasing Bayesian position decoding error across recorded sessions; the searched text did not provide a single numeric accuracy value in the methods text, but Figure 1F caption reports a significant improvement across sessions (`ANOVA: p < 0.0001; F = 7.9845`) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total dataset size | Code/README describe per-animal joblib datasets and analyses over all animals | Summing non-NaN cells across sessions gives exactly 69,744 session-cell maps; 207 sessions total; 5,413 registered cells across subject files | "5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps" | No discrepancy. Raw files match paper exactly. |
| Session duration / frame count | Code assumes `fps=30` in rate-map and decoder functions | Session lengths are 71,866-72,219 frames, i.e. 2,395.53-2,407.30 s | "behavioral and cellular imaging streams at 30 Hz" and "All sessions were 40 min" | No discrepancy. Small frame-count deviations are consistent with nominal 40-minute sessions and timestamp-based acquisition. |
| Place-cell threshold | `get_place_cells(..., alpha=0.05)` returns a boolean at p<0.05 by default | Raw data do not store place-cell labels | Paper methods say cells exceeded the 99th percentile of shuffled distribution; figure text discusses p<0.01 | Resolved by reading plotting code: `plot_shr_pvals` thresholds the saved p-values at `0.05`, `0.01`, and `0.001`. For the paper figure the relevant threshold is `0.01`; the saved object is the p-value matrix, so no conversion impact. |
| Cell inclusion for main analyses | `main.py` decodes position from all cells using `decode_position_within(...)` and that function applies velocity/activity filters online | Raw traces contain `NaN` for cells absent on a session, but no separate quality mask | Paper states recordings motivated "the inclusion of all cells in subsequent analyses" | No discrepancy. For the conversion I should keep all session-present cells and avoid pre-filtering to place cells. |
| Environment geometry representation | Code commonly uses `envs` plus `get_env_mat(env)` and sometimes additional flips/transposes when constructing masks | Raw data also contain an explicit `blocked` list with partition IDs in the stated 3x3 indexing scheme; direct blocked-to-grid conversion differs in orientation from map axes for some asymmetric environments | README states `blocked` is organized as `[[0,1,2],[3,4,5],[6,7,8]]` and indicates which partitions are blocked | Cross-checking against the valid-mask of `maps['smoothed']` shows that `blocked` reshaped to 3x3 and then transposed matches the actual map/position coordinate frame for all 207 sessions. For decoder input, use `blocked` as authoritative, transpose the 3x3 matrix to align with `position`, and use `envs` as a metadata cross-check. |
| Trial structure | Reference code analyzes continuous sessions, not explicit trials | Raw data are continuous session-long streams | Paper describes continuous 40-minute sessions with within-session decoding | No discrepancy. The 1-minute trialization is a task-specific transformation required by the target format, not part of the original acquisition. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `dat[animal]['trace'][day, present_cells, frame_start:frame_end]` | `neural` | Keep session-present cells only (`~np.isnan(trace[:,0])`), split each session into non-overlapping 1800-frame windows, cast to `float32` | `load_dat`; decoder uses `trace` directly in `decode_position_within` | No delta-F/F. Use released binary rising-phase event traces directly. |
| `dat[animal]['blocked'][day]` | `input[trial]` | Convert blocked partition IDs to 3x3 binary open/blocked matrix, transpose to align with map/position axes, flatten to 9-dim float vector | README field description; cross-checked against `maps['smoothed']` valid mask and `get_env_mat` conventions | Static per trial; same value for every 1-minute trial within a session. |
| `dat[animal]['position'][day, :, frame_start:frame_end]` | `output[trial]` | Compute 3x3 spatial bins using the reference floor-division rule with session-wide position maxima and `buffer`, then flatten `(xbin, ybin)` to one categorical class `xbin*3 + ybin`; store as shape `(1, T)` integer array | `get_rate_maps`; `decode_position_within`; `fit_decoder`; `test_decoder` | Using session-wide maxima preserves a single spatial partition per original session across all derived trials. |
| Animal file name / key | `subjects`, `subject_idx` | Unique sorted subject IDs; one target session per original recording day | `main.py` animal list; raw file organization | Session order will be by subject file and in-file day order. |
| Constant `CA1` from dataset identity | `brain_regions`, `brain_region_idx` | Single region list `['CA1']`; zeros for all neurons in each session | Dataset/paper identity | All recordings are CA1. |

### Key Decisions
1. **Use primary joblib animal files, not cached analysis results**: This matches the reference loading path in `load_dat(..., format="joblib")` and avoids inheriting any downstream analysis assumptions.
2. **Keep one target session per original recording day**: The raw data are organized by day/session, and the target format supports multiple trials within each session.
3. **Create trials by splitting each continuous 40-minute session into non-overlapping 1-minute windows**: This satisfies the decoder task while preserving within-session context. Trial length will be exactly 1800 frames at 30 Hz; any trailing partial minute will be discarded.
4. **Use all session-present cells**: This matches the paper’s statement that all cells were included in subsequent analyses. Cells absent on a day are removed by excluding `NaN` rows for that session.
5. **Do not pre-filter to place cells**: The paper’s position decoder is not place-cell-restricted, and place-cell status is an analysis label rather than a required curation step for decoding.
6. **Do not compute new calcium features**: The released `trace` is already the binary rising-phase representation used in the paper/code.
7. **Align geometry input to the map/position frame by transposing the 3x3 blocked matrix**: This is required for asymmetric geometries and was verified against the non-NaN support of `maps['smoothed']` for every session.
8. **Use session-wide position normalization when binning to 3x3 outputs**: The reference code bins position using maxima from the full session/day, not from smaller windows. This keeps spatial bins consistent across all trials within a session.
9. **Represent output as one 9-class time-varying variable instead of separate x/y outputs**: The task explicitly requests 3x3=9 spatial bins, so a single categorical output variable is the most direct representation.

### Planned Sanity Checks
- [ ] Verify subject/session totals after conversion: 7 subjects, 207 sessions, and 8187 derived full-minute trials.
- [ ] Verify per-session neuron counts equal the number of non-NaN trace rows in the raw day-level session.
- [ ] Verify `blocked`-derived 3x3 geometry (after transpose) matches the valid spatial support of `maps['smoothed']` for spot-checked sessions and with `np.allclose()` for selected examples.
- [ ] Verify trial neural arrays exactly match slices from the raw `trace` matrix for selected subject/day/trial combinations with `np.allclose()`.
- [ ] Verify trial output class labels exactly match raw-position-derived 3x3 bins for selected subject/day/trial combinations with `np.allclose()`.
- [ ] Verify each trial has `1800` time points and each session has at least `39` derived trials.

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `convert_data.py` with:
  - deterministic session enumeration from primary joblib animal files
  - reference-consistent use of released binary calcium-event traces
  - session-wise neuron selection via non-NaN trace rows
  - trialization into non-overlapping 1-minute windows (`1800` frames at `30 Hz`)
  - static per-trial 9D geometry input from the transposed `blocked` field
  - time-varying 9-class 3x3 position output from session-wide spatial binning
  - geometry-vs-map consistency checks per session
  - optional processing plots for up to 2 sessions
  - per-session timing output
- `python3 -m py_compile convert_data.py` completed successfully.

Code inefficiencies identified:
- Loading the full animal files is the main runtime cost because each file contains all sessions and registered cells for one subject.

Code speedups added:
- Process sessions animal-by-animal so only one large subject file is kept in memory at a time.
- Reuse one loaded animal dataset across all of its sessions before releasing it.
- Slice full-session binned outputs and present-cell traces directly without redundant recomputation inside trials.
- Store neural trials as `float16` and outputs as `int8` to reduce disk footprint and improve the chances that the full dataset remains tractable during downstream validation.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 338 session-present neurons across the 2 converted sample sessions |
| Neurons / session | [185, 153], mean 169.0 |
| Subjects | 1 represented in sample (`QLAK-CA1-08`) |
| Sessions / subject | 2 for the sampled subject |
| Trials (total) | 78 |
| Trials / session | [39, 39] |
| Geometry input range (9 dims) | mins `[1,1,1,1,0,1,1,1,1]`, maxs `[1,1,1,1,1,1,1,1,1]` |
| Position-bin output distribution | `[0.103063, 0.102521, 0.103974, 0.061574, 0.043782, 0.117258, 0.134423, 0.103027, 0.230377]` |

### Processing Plots Review
- Created:
  - `processing_QLAK-CA1-08_day00.png`
  - `processing_QLAK-CA1-08_day01.png`
- No anomalies detected from the underlying checks used to generate the plots:
  - the transposed `blocked` geometry exactly matched the valid spatial support of `maps['smoothed']`
  - position binning covered the full `0..8` output range in both sample sessions
  - neural trials had fixed length `1800` and no NaN values after present-cell filtering

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Reuse one loaded animal file across all of its sessions | Avoids reloading the same 68-145 MB joblib file for every session |
| Direct session-wide position binning followed by trial slicing | Avoids recomputing bins separately inside each 1-minute trial |
| Save neural trials as `float16` instead of `float32` | Reduced `sample_data.pkl` from about 92 MB to 46 MB |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion observed | 6.51 s/session over 2 sessions | 22.4 min if extrapolated naively |
| Refined estimate accounting for one-time per-animal load and prior measured raw-file load costs | approximately 2-3 s/session effective | approximately 7-10 min for all 207 sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `position_bin_3x3` | 0.3577 | 0.3021 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9.3G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 5,413 unique neurons; 69,744 rate maps | Uses all loaded cells/session maps in analyses | 5,413 registered cells; 69,744 session-present cells across sessions | 69,744 session-level neurons across all sessions | Yes; converted format preserves session-present neurons, not cross-day identities |
| Mean neurons/session | 69,744 / 207 = 336.93 | Implicit from session-wise decoding over all cells | 336.93 present cells/session | 336.93 neurons/session | Yes |
| Subjects | 7 inferred from totals and subject files | 7 animals hard-coded in `main.py` | 7 | 7 | Yes |
| Sessions | 207 | Iterates over all day/session recordings | 207 | 207 | Yes |
| Trials (total) | N/A in paper; continuous 40 min sessions | N/A in code; continuous sessions | 8187 full 1-minute windows derived from frame counts | 8187 | Yes |
| Trials/session (mean) | N/A | N/A | 39.55 | 39.55 | Yes |
| Geometry input range | Binary geometry by blocked partitions / environment shape | Environment masks are binary | `[0,1]` in the 3x3 geometry support implied by blocked partitions | each geometry dimension in `[0,1]` | Yes |
| Position-bin output distribution | No exact 3x3 distribution reported | Within-session position decoding from binned position | derived from raw 3x3-binned position | `[0.099285, 0.075014, 0.116048, 0.097637, 0.056258, 0.141957, 0.135226, 0.076953, 0.201622]` | Yes; converted values come directly from raw position |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reported "Data format is valid, no errors or warnings." No fixes were required.
2. **Sanity checks against original data with `np.allclose()`**:
   - Session `0` (`QLAK-CA1-08_day00`): converted neural data exactly matched the raw present-cell trace slice over all 39 trials (`185 x 70200`), input geometry exactly matched the transposed `blocked` field, and output labels exactly matched the raw-position-derived 3x3 bins.
   - Session `93` (`QLAK-CA1-51_day00`): same checks passed for a 40-trial session (`113 x 72000` neural reconstruction).
   - Session `206` (`QLAK-CA1-75_day30`): same checks passed for the final session (`514 x 72000` neural reconstruction).
3. **Reference code comparison**:
   - `(a) data loading`: converter uses the primary joblib animal files, matching `load_dat(..., format="joblib")`.
   - `(b) neuron/trial filtering`: converter keeps all session-present cells (`NaN` rows removed only) and does not pre-filter to place cells, consistent with the paper’s statement that all cells were included in subsequent analyses.
   - `(c) temporal alignment`: converter preserves the raw 30 Hz synchronized time base and uses the released aligned `position` and `trace` streams directly.
   - `(d) binning`: converter uses the same session-wide position floor-division rule as the reference code, but with `3 x 3` bins instead of `15 x 15` because the decoder task requires 9 spatial classes.
   - `(e) input construction`: converter uses the raw `blocked` field, transposed to match the map/position coordinate frame; this was verified against the non-NaN support of `maps['smoothed']` for all sessions.
   - `(f) output construction`: converter builds categorical 3x3 position labels directly from the raw position stream; this is the required task-specific discretization of the reference continuous position variable.
4. **Key statistics comparison**:
   - Raw sessions = converted sessions = `207`
   - Raw registered cells = `5413`
   - Raw session-present cells = converted neuron instances = `69744`
   - Raw 1-minute derived trials = converted trials = `8187`
   - Trial counts per session are exactly `39` or `40`, matching raw frame counts.
5. **Edge-case checks**:
   - `discarded_tail_frames` values were exactly the raw remainders `[60, 71, 91, 219, 1666]`
   - all converted trials had length `1800`
   - all sessions had at least 2 trials
   - no NaN/Inf values remained in neural, input, or output arrays

### Issues Found and Resolved
- **Environment orientation ambiguity**: resolved before full conversion by proving that `blocked.reshape(3,3).T` exactly matches the valid spatial support of `maps['smoothed']` for all 207 sessions.
- **Storage pressure**: reduced serialized neural data from `float32` to `float16` and outputs to `int8`; sample conversion and sample decoder training were re-run afterward and still passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `position_bin_3x3` | 0.6221 | 0.5489 | Chance is `1/9 = 0.1111`; full training completed on GPU without memory failure |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `position_bin_3x3` | Validation balanced accuracy `0.5489`; training balanced accuracy `0.6221`; uniform chance `0.1111` | Paper reports that Bayesian position decoding error decreases significantly across sessions and that the recordings achieve strong decoding performance, but it reports Euclidean decoding error rather than 3x3 balanced accuracy, so there is no exact metric match |

[Analysis of any low accuracies]
- **Chance comparison**: `0.5489 / 0.1111 = 4.94x` uniform chance, so the decoder is well above the bug-indicating threshold.
- **Class imbalance robustness**: the largest output class fraction in the full dataset is `0.2016`, so validation balanced accuracy also exceeds the majority baseline by about `2.72x`.
- **Train vs validation gap**: `0.6221 / 0.5489 = 1.13x`, well below the `1.5x` overfitting concern threshold.
- **Comparison to the paper**: direct numeric comparison is not possible because the paper’s Figure 1F reports Euclidean decoding error (cm) from a within-session Gaussian Naive Bayes decoder, while the validation script reports balanced accuracy for a different downstream model on 1-minute trialized 3x3 classes. Qualitatively, the strong above-chance accuracy and the absence of alignment/format warnings are consistent with the paper’s claim that position decoding quality is high.

### Issues Found and Resolved
- No new conversion issues were revealed by full decoder training.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
