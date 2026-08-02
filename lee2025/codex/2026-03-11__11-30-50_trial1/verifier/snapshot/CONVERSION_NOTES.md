# Dataset Conversion Notes

## Overview
- **Dataset**: CA1 geometry remapping dataset from "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping"
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

Environment checks:
- `python3` available: `3.13.12`
- `numpy` import successful: `2.3.5`
- `torch` import successful: `2.6.0+cu124`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | `code/georepca1/src/utils.py` | LOADING | Loads one animal’s preprocessed dataset from `data/<animal>` joblib or original MATLAB file; converts key list-like fields to NumPy only for MATLAB path. |
| `save_dat` / `mat2joblib` | `code/georepca1/src/utils.py` | LOADING | Saves converted joblib copies of per-animal datasets. |
| `generate_behav_dict` | `code/georepca1/src/utils.py` | PROCESSING | Creates a lightweight behavior dictionary with `position`, `envs`, and map shape metadata for all animals. |
| `get_env_mat` | `code/georepca1/src/utils.py` | PROCESSING | Converts categorical environment names (`square`, `o`, `t`, `u`, `rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`) into binary 3x3 occupancy matrices. |
| `get_rate_maps` | `code/georepca1/src/utils.py` | PROCESSING | Builds 15x15 occupancy-normalized event-rate maps from framewise position and rise-event traces at 30 fps, with Gaussian smoothing. |
| `get_split_half` / `get_shuffle_split_half` | `code/georepca1/src/utils.py` | CURATION | Computes split-half reliability and shuffled null distributions for spatial tuning. |
| `get_place_cells` | `code/georepca1/src/utils.py` | CURATION | Classifies place cells using split-half reliability significance (`alpha=0.05`, shuffle-based). |
| `get_shr_within` | `code/georepca1/src/utils.py` | CURATION | Applies place-cell reliability calculation to each recording day within an animal. |
| `fit_decoder` | `code/georepca1/src/utils.py` | PROCESSING | Temporally bins data with 3-frame average pooling, smooths traces, and fits a flat-prior Gaussian naive Bayes decoder on binned 2D position classes. |
| `test_decoder` | `code/georepca1/src/utils.py` | PROCESSING | Applies the fitted decoder, reconstructs 2D position bins, and computes squared decoding error. |
| `decode_position_within` | `code/georepca1/src/utils.py` | PROCESSING | Performs 5-fold cross-validated within-session position decoding; filters frames by velocity and cells by activity before decoding. |

### Notes
- The repository README states the dataset contains preprocessed calcium imaging data, not electrophysiology. The `trace` field is already a rise-extracted event matrix where `1` marks significant calcium events. No code computes `dF/F`; for this conversion the reference neural signal is the provided event trace.
- The joblib data files are the reference loading path used throughout `main.py`. The analysis code expects per-animal files in `data/` and works from already registered cells across days.
- `main.py` shows the authors’ within-session position decoder call:
  `decode_position_within(dat[animal]['position'].T, dat[animal]['trace'].T, dat[animal]['maps']['smoothed'])`.
  This indicates:
  1. position is stored per day as 2 x time and transposed to time x 2 for processing,
  2. trace is stored per day as cells x time and transposed to time x cells,
  3. the authors use precomputed smoothed maps as spatial masks/templates during decoding.
- Decoder-specific reference settings from `decode_position_within`:
  1. spatial discretization is `n_bins=15` internally for the paper’s decoder,
  2. temporal binning is `temporal_bin_size=3` frames via `AvgPool1d`,
  3. velocity filter uses Gaussian-smoothed speed with `v_filt_size=5` and threshold `v_thresh=5`,
  4. cells must exceed `cell_threshold=5` events in included frames,
  5. cross-validation uses `n_fold=5`.
- Arena geometry is represented in code via `envs` string labels mapped by `get_env_mat`, not by the `blocked` field directly. This is likely the cleanest reference representation for the decoder input requested here.
- Place-cell filtering exists in the repository for specific analyses, but the position decoder in `main.py` does not restrict to place cells before decoding. The decoder instead relies on activity and movement thresholds inside `decode_position_within`.
- Registration quality is encoded by `NaN` traces/maps for cells absent on a given day. Several functions treat `NaN` entries as unregistered cells rather than applying an additional cell-quality filter.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains 7 per-animal joblib files, 7 matching MATLAB `.mat` files, one `behav_dict` joblib summary, and a `precomputed_results/` folder with analysis outputs from the reference repository.
- Each per-animal joblib file is a dictionary keyed by animal ID and contains:
  - `trace`: dense array of shape `(n_days, n_registered_cells, n_frames)` with rise-extracted calcium events. Unregistered cells on a day are represented by `NaN` across time.
  - `position`: dense array of shape `(n_days, 2, n_frames)` with x-y position sampled at the same frame rate as `trace`.
  - `envs`: array of shape `(n_days, 1)` with categorical environment labels.
  - `blocked`: Python list of length `n_days`; each element is a list containing an array of blocked partition indices in the 3x3 environment layout. `-1` denotes no blocked partitions.
  - `maps`: dictionary with precomputed rate maps:
    - `sampling`: `(15, 15, n_days)`
    - `smoothed`: `(15, 15, n_registered_cells, n_days)`
    - `unsmoothed`: `(15, 15, n_registered_cells, n_days)`
  - `SFPs`: `(35, 35, n_registered_cells, n_days)` centered spatial footprints.
  - `centroids`: `(n_registered_cells, 2, n_days)` ROI centroids.
- Native recordings are continuous sessions, not trialized data. Session durations are approximately 39.9 to 40.1 minutes at 30 Hz depending on animal.
- Environment order is a repeated geometry sequence. Example from `QLAK-CA1-08`:
  `square, o, t, u, rectangle, +, i, l, bit donut, glenn` repeated three times, plus a final `square`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5,413 registered cells across animals |
| Neurons / session | 336.93 mean valid cells per session (min 113, max 564); registered-cell counts per animal = [515, 875, 942, 554, 862, 713, 952] |
| Subjects | 7 mice (`QLAK-CA1-08`, `QLAK-CA1-30`, `QLAK-CA1-50`, `QLAK-CA1-51`, `QLAK-CA1-56`, `QLAK-CA1-74`, `QLAK-CA1-75`) |
| Sessions / subject | 31 for six mice; 21 for `QLAK-CA1-51` |
| Trials (total) | No native trial structure in source data; recordings are continuous per-session time series |
| Trials / session | N/A in source data (continuous ~40 min sessions to be segmented later) |

Additional raw-data observations:
- Total sessions: 207
- Total valid cell-by-session entries (non-NaN registrations): 69,744
- Session frame counts: `{71866, 72060, 72071, 72091, 72219}`
- Session durations at 30 Hz: `{39.93, 40.03, 40.04, 40.05, 40.12}` minutes
- Example blocked-partition encoding from `QLAK-CA1-08`: `-1`, `4`, `[3,5,6,8]`, `[4,5]`, `[0,3,6]`, `[0,2,6,8]`, ...

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 unique neurons | `"5,413 unique neurons across 207 sessions"` |
| Neurons / session | Mean 773 +/- 68 SE cells per animal; min 515, max 952 across animals | `"mean number of cells per animal = 773 +/- 68 SE"` |
| Subjects | 7 mice | `"Naive male (4) and female (3) mice"` |
| Sessions / subject | Up to 31 days / 3 repeated sequences; one mouse contributes only 21 sessions in released data | `"up to three total repetitions (31 days)"` |
| Trials (total) | No native trials; sessions are continuous 40 min recordings | `"All sessions were 40 min"` |
| Trials / session | N/A in source data | `"one session was recorded per day"` |
| Neural data time bin | 33.3 ms per frame (30 Hz acquisition); paper analyses often pool 3 frames for decoding and use 5 cm spatial bins for maps | `"streams at 30 Hz"` |
| Behavior data time bin | 33.3 ms per frame, aligned with neural recording | `"behavioral and cellular imaging streams at 30 Hz"` |
| Reward rate | N/A; free exploration task, no reward manipulation described | No reward variable described in Methods or Results |
| Geometry conditions | 10 geometries in a repeated sequence | `"10 geometrically distinct environments"` |
| Arena size | 75 x 75 cm square base arena; 3 x 3 partitions | `"75 x 75 cm"` and `"3 x 3 grid space"` |
| Rate maps | 15 x 15 bins corresponding to 5 cm x 5 cm locations | `"5cm x 5cm grid of locations"` |
| Rate-map count | 69,744 rate maps | `"forming 69,744 rate maps"` |


### Processing Details
- Neural and behavioral streams were acquired simultaneously at 30 Hz and timestamped for post-hoc alignment.
- Sessions are single-day 40 min free-exploration recordings.
- Calcium traces were preprocessed before release: motion correction, cell segmentation, transient extraction, then rising-phase binarization.
- Rising-phase extraction in Methods:
  1. Differentiate median-subtracted calcium trace.
  2. Smooth derivative with Gaussian kernel (`sd = 5` frames).
  3. Estimate derivative noise from the negative half distribution.
  4. Z-score derivative relative to that noise estimate.
  5. Threshold at `z > 2.5` to set a binary rising-event vector.
- The resulting binary event vector is treated as the firing rate in all later analyses.
- Position is derived from DeepLabCut head tracking.
- Rate maps are constructed on a 5 cm x 5 cm grid and smoothed with an isotropic Gaussian kernel with 5 cm standard deviation.
- Place-cell detection compares first vs second 20 min session halves using 1000 circular shuffles of position.
- Position decoding uses a 5-fold split on spatially binned position and trace data with Gaussian Naive Bayes, flat priors, and Euclidean distance as the error metric.

### Curation Steps

**Neuron curation rules**:
- Motion-corrected movies were manually inspected for artifact-free correction.
- Spatial footprints were manually verified to remove lens artifacts.
- Cells were tracked across sessions using landmarks plus footprints/centroids.
- For reported main analyses, all cells were retained after demonstrating high spatial reliability; the Results explicitly say this motivated inclusion of all cells in subsequent analyses.
- Place-cell labels were computed for descriptive analyses, but all registered cells were used for the paper’s main downstream analyses including within-session decoding.

**Trial curation rules**:
- No native trial structure exists in the source dataset.
- For place-cell analysis, each 40 min session is divided into first and second 20 min halves.
- For within-session decoding in code, low-velocity frames and low-activity cells are excluded inside the decoder (`v_thresh=5`, `cell_threshold=5` events), consistent with the reference implementation even though these constants are not spelled out in Methods text.

### Decoders Trained
| Decoded variable | Accuracy |
| Within-session animal position | Paper reports Bayesian position decoding error in cm, not categorical accuracy; decoding error decreases significantly across sessions (`ANOVA p < 0.0001, F = 7.9845`) |
| Animal identity from RSM | Not decodable above useful levels; paper states identity could not be decoded from RSMs |
| Across-animal / across-sequence RSM prediction | Reported as accurate in Figures 2H-2I qualitatively; not directly relevant to current conversion target |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Place-cell threshold | `get_place_cells(..., alpha=0.05)` default in `utils.py` | Stored outputs in `*_shr` are raw p-values, not only booleans | Place cells defined at 99th percentile / `p < 0.01` | Not a real mismatch for main figures: the code saves p-values, and `plot_shr_pvals()` applies thresholds `[0.05, 0.01, 0.001]` later. For conversion, place-cell labels are not needed. |
| Sessions per subject | Most code examples assume 31-day sequences | Raw data contain six 31-session mice and one 21-session mouse (`QLAK-CA1-51`) | `"up to three total repetitions (31 days)"` | Consistent with paper wording: not every mouse completed all 31 days. Keep all 207 released sessions. |
| Environment order across mice | Paper says geometry sequence randomized across mice | Raw `envs` orders differ by mouse but repeat within mouse across sequences | `"randomized for each mouse"` | Consistent. No action needed. |
| Mean cells per animal | Paper reports `773 +/- 68 SE`, min 515, max 952 | Raw registered-cell counts per animal are `[515, 875, 942, 554, 862, 713, 952]`, mean `773.29`, SE `68.50` | Same reported summary in Results | Exact match up to rounding. Confirms raw data align with paper. |
| Total neurons / sessions / rate maps | Code and paper operate on a longitudinal registered-cell dataset | Raw data sum to 5,413 registered cells, 207 sessions, and 69,744 valid cell-by-session maps | `"5,413 unique neurons across 207 sessions ... 69,744 rate maps"` | Exact match. Confirms loading interpretation is correct. |
| Geometry representation source | Reference helper `get_env_mat(env)` maps `envs` label to a 3x3 template | Raw `blocked` lists directly encode blocked partition indices; for some shapes (`t`, `l`, `bit donut`) orientation differs from `get_env_mat`, and `glenn` does not exactly match the helper template | Paper describes blocked partitions in a 3x3 design | For the decoder input, use raw `blocked` as canonical geometry because it directly encodes blocked partitions per session. Keep `envs` as metadata / human-readable labels. This preserves actual session geometry and avoids helper-template simplifications. |

Final consistency summary:
- The core counts in paper, code, and raw data agree exactly: 7 mice, 207 sessions, 5,413 registered cells, 69,744 valid rate maps.
- The code and Methods agree on the main processing path: 30 Hz aligned acquisition, binary rising-event traces, 15 x 15 rate maps, shuffle-based place-cell reliability, and 5-fold Gaussian naive Bayes position decoding.
- The only conversion-relevant discrepancy is geometry representation. The raw `blocked` field is more faithful for constructing decoder inputs than the abstract `envs -> get_env_mat()` helper.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day, valid_cells, frames]` | `neural[session][trial]` | Keep only session-valid cells (`~np.isnan(trace[day, :, 0])`). Split each session into 40 consecutive 1-minute chunks. Convert each chunk from 30 Hz binary events to 100 ms bins via non-overlapping 3-frame averaging. | `load_dat`; paper/code use binary rising-event traces directly; temporal pooling logic from `fit_decoder` / `test_decoder` | No `dF/F` computation. No place-cell filter. No extra neuron-quality filter beyond removing unregistered (`NaN`) cells on that day. |
| `blocked[day]` | `input[session][trial]` | Convert blocked-partition indices into a length-9 binary open-mask vector in raw partition order (`1=open`, `0=blocked`); store as static 1D trial input. | README data description; paper 3 x 3 partition design | Use raw `blocked`, not `envs -> get_env_mat`, because raw field preserves actual blocked partitions without helper-template ambiguities. |
| `position[day, :, frames]` | `output[session][trial]` | Split into same 1-minute chunks as neural. Average x/y position within each 3-frame bin (100 ms), then discretize to 3 x 3 bins using session-wise maxima and the same integer-floor logic as reference code. Encode as one categorical output variable with values `0..8`, shape `(1, T)`. | `decode_position_within`, `fit_decoder`, `test_decoder`, `get_rate_maps` | Bin index convention will follow README partition order: `bin = x_bin * 3 + y_bin` with `x_bin,y_bin in {0,1,2}`. |
| Animal ID from filename / top-level dict key | `subjects`, `subject_idx` | Session order will be animals in sorted filename order, days in native order within each animal | `main.py` animal list; raw data organization | Produces 207 sessions total. |
| Session-valid neurons | `brain_region_idx[session]` | Length = number of valid cells in that session; all zeros | Paper and code are CA1-only | `brain_regions = ['CA1']`. |
| `envs[day]` | metadata / optional debug info | Preserve as human-readable session geometry label in metadata/session notes | `load_dat`, README | Not used as decoder input. |

### Key Decisions
1. **Neural signal**: use the released binarized rising-phase calcium-event traces directly.
Rationale: both Methods and code treat this binary vector as the firing rate in all later analyses; recomputing fluorescence-derived signals would be inconsistent.

2. **Neuron inclusion**: keep all valid registered cells for each session and remove only unregistered (`NaN`) cells on that day.
Rationale: the paper explicitly says the observed reliability motivated inclusion of all cells in subsequent analyses; the within-session decoder in `main.py` also uses all cells.

3. **Trialization**: impose 40 one-minute trials per session using session start as time zero.
Rationale: paper sessions are nominally 40 min. For recordings shorter than 72,000 frames, the 40th trial is shorter; for recordings longer than 72,000 frames, discard the small tail beyond 40 min. This preserves the experimental session duration while minimizing data loss.

4. **Temporal bin size**: store data at 100 ms resolution using non-overlapping 3-frame bins.
Rationale: the reference decoder temporally bins data in 3-frame windows. Using 100 ms bins matches this scale, reduces computation substantially for the validator, and preserves alignment across neural and behavior streams.

5. **Geometry input source**: use raw `blocked` rather than `envs -> get_env_mat`.
Rationale: Step 4 showed helper templates are an abstract geometry representation and do not exactly match raw blocked partitions for all shapes. The raw blocked-partition field is the direct decoder input requested by the user.

6. **Position discretization**: use session-wise maximum-based spatial binning, matching the reference code’s flooring rule, but with 3 bins instead of 15.
Rationale: the paper’s code bins position by dividing by `(session_max + buffer) / n_bins`. Reusing the same logic preserves coordinate handling while adapting to the requested 3 x 3 output grid.

7. **Output representation**: use one categorical, time-varying output dimension (`position_bin_3x3`) rather than two separate x/y outputs.
Rationale: the task explicitly requests 9 spatial bins. A single 9-class output matches the decoder objective directly and avoids reconstructing classes from separate marginals.

8. **Static input representation**: store geometry as a 1D length-9 vector per trial.
Rationale: the validator will automatically tile 1D inputs across time. Static 1D storage is compact and faithful because geometry is constant within each trial.

### Planned Sanity Checks
- [ ] Neural check: for selected `(session, trial, neuron, timebin)`, verify converted value equals the mean of the corresponding raw 3-frame event segment after session/day cell filtering.
- [ ] Input check: for selected sessions, verify converted 9-element geometry vector exactly matches the raw `blocked` partition indices from original files.
- [ ] Output check: for selected `(session, trial, timebin)`, verify converted class equals the session-wise-max discretization of the corresponding raw averaged x-y position.
- [ ] Count check: converted dataset must contain 207 sessions, 40 trials/session, and per-session neuron counts matching raw non-NaN cell counts.
- [ ] Summary-stat check: total registered cells (5,413), valid session-cell maps (69,744), and per-animal cell statistics must match paper/data.
- [ ] Distribution check: output classes should exclude blocked bins for a given session geometry except for negligible edge effects from binning boundaries; investigate any large violations.
- [ ] Alignment check: within a spot-checked trial, neural, input, and output arrays must have identical time dimensions after 100 ms binning.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `convert_data.py` with the following behavior:
- CLI: `python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]`
- Streams animal files one at a time from `data/` to limit memory use.
- Uses raw binarized CA1 event traces, removes only session-invalid (`NaN`) cells.
- Splits each session into nominal 40 one-minute chunks.
- Temporally bins neural and position streams into 100 ms bins via non-overlapping 3-frame averaging.
- Converts raw blocked partitions into a static 9-element geometry input vector.
- Converts x-y position into one categorical 3x3 output class per 100 ms bin.
- Stores session-level metadata including original frame counts, usable frame counts, geometry labels, and trial counts.
- Saves `processing_<session_id>.png` for up to 2 sessions when `--show-processing` is requested.
- Prints per-session and per-animal timing information.

Quick verification:
- `python3 -m py_compile convert_data.py` passed.

Code inefficiencies identified:
- Full-data runtime will be dominated by decompressing the 7 large joblib animal files.

Code speedups added:
- Per-animal streaming instead of loading all animals at once.
- Vectorized 3-frame temporal binning with reshape/mean.
- Static per-trial inputs stored as 1D arrays so the validator can tile them automatically.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 338 session-valid neurons across 2 sessions |
| Neurons / session | mean 169.0; session counts = [185, 153] |
| Subjects | 1 active subject in sample (`QLAK-CA1-08`) |
| Sessions / subject | 2 |
| Trials (total) | 80 |
| Trials / session | 40 |
| `open_partition_4` range | [0, 1] |
| other `open_partition_*` ranges in sample | mostly [1, 1] because sample covers `square` and `o` sessions only |
| `position_bin_3x3` distribution | [0.102, 0.100, 0.103, 0.060, 0.043, 0.116, 0.140, 0.104, 0.231] across bins 0-8 |

### Processing Plots Review
- `processing_QLAK-CA1-08_day00.png`: square geometry input is fully open; raw trajectory fills the full arena; trial duration panel shows 39 full 60 s trials plus one shorter final trial (~55.5 s), matching the slightly short raw recording.
- `processing_QLAK-CA1-08_day01.png`: geometry input shows only center partition blocked; raw trajectory shows the expected central hole; binned position classes avoid the blocked center for long periods, indicating geometry/output alignment is sensible.
- No obvious temporal misalignment between neural preview, trial boundaries, and binned outputs in the plotted sessions.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| 100 ms temporal binning + per-animal streaming | Keeps projected full conversion comfortably below 15 minutes and reduces decoder training load substantially |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion (`--sample --show-processing`) | ~0.73 s/session after load; ~12 s for 2 sessions total | ~3-4 min for full conversion, conservatively < 10 min including file decompression overhead |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `position_bin_3x3` | 0.3755 | 0.2986 |

Additional observations:
- Chance level for the 9-class position output is `1/9 = 0.1111`.
- Training loss decreased monotonically from `2.279852` at epoch 1 to `1.785706` at epoch 200.
- Test loss was `1.861886`.
- Sample decoder performance is comfortably above chance, supporting the current alignment / discretization choices.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6.3G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 5,413 unique registered cells | Same animal files loaded by `load_dat` | 5,413 registered cells across animals | Target format stores session-valid neurons; raw source summary preserved in notes/metadata, and verification reports 69,744 session-neuron instances | Partial by design |
| Mean neurons/session | Not stated directly; implied by 69,744 / 207 = 336.93 valid cell-maps/session | Uses all valid cells/session | 336.93 valid cells/session | 336.93 neurons/session | Yes |
| Subjects | 7 | 7-animal list in `main.py` | 7 | 7 | Yes |
| Sessions | 207 | 207 from released data files | 207 | 207 | Yes |
| Trials (total) | N/A in paper (continuous sessions) | N/A | Continuous sessions only | 8,280 derived 1-min trials (= 207 x 40) | By conversion design |
| Trials/session (mean) | N/A | N/A | Continuous sessions only | 40.0 | By conversion design |
| Geometry input range | 0/1 blocked-open partitions in 3 x 3 design | `get_env_mat` / raw blocked partitions | 0 or 1 per partition | partition 0-8 ranges consistent with binary geometry inputs; partition 7 always open in released geometries | Yes |
| Output class range | Paper decoder uses spatial bins and Euclidean error | 2D position bins via integer flooring | x/y trajectories span arena | 9 classes, range [0, 8] | Yes |
| `position_bin_3x3` distribution | Paper reports qualitative decoding improvement, not 9-class fractions | Not explicitly reported | Derived from raw trajectories | [0.099, 0.075, 0.116, 0.098, 0.056, 0.142, 0.135, 0.077, 0.202] | Reasonable / consistent with arena occupancy |

Step 9 verification summary:
- `train_decoder.py converted_data.pkl --verify-only` reported no errors or warnings.
- Converted dataset summary: 207 sessions, 8,280 trials, 40 trials/session, mean 336.93 neurons/session, time bin size 100 ms.
- Trial lengths are exactly as expected from the source recordings after 3-frame binning: full trials have 600 bins and shortened final trials in the three slightly short-recorded animals have 555 bins.
- Session counts by mouse match the source data exactly: six mice with 31 sessions and one mouse (`QLAK-CA1-51`) with 21 sessions.
- Spot-check statistics from conversion logs match Step 2 raw valid-cell counts for all animals.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Re-read `verification_full_out.txt` after the latest full rebuild. Result: no errors, no warnings.
2. **Raw-vs-converted sanity checks with `np.allclose()`**: Compared original raw data against `converted_data.pkl` for 8 sessions spanning all animals, including both shortened-final-trial sessions (`QLAK-CA1-08`, `QLAK-CA1-30`, `QLAK-CA1-50`) and full-final-trial sessions (`QLAK-CA1-51`, `QLAK-CA1-56`, `QLAK-CA1-74`, `QLAK-CA1-75`). Result: input geometry, neural 3-frame means, output class construction, and final-trial lengths all matched.
3. **Reference code comparison**:
   - (a) Data loading: conversion loads the same per-animal joblib files used by `load_dat(..., format='joblib')`.
   - (b) Neuron filtering: conversion keeps all session-valid cells (non-NaN registrations), consistent with the paper’s inclusion of all reliable recorded cells in main analyses.
   - (c) Temporal alignment: conversion keeps the released neural and position streams aligned frame-for-frame and applies a common 3-frame binning to both.
   - (d) Binning: conversion uses the same session-wise max-based flooring logic as the reference code, adapted from 15 x 15 bins to the requested 3 x 3 grid.
   - (e) Input construction: unlike the plotting/model helper `get_env_mat`, conversion uses raw `blocked` partitions because this is the direct source representation requested by the decoder task.
   - (f) Output construction: conversion produces one categorical position-bin output over time, consistent with the paper’s position-decoding objective.
4. **Key statistics comparison**: Confirmed again that 7 mice, 207 sessions, 69,744 valid session-cell entries, and 336.93 mean neurons/session match paper/data summaries.
5. **Edge-case checks**:
   - Short recordings: final trial length is 555 bins for the three slightly short 39.93 min recordings and 600 bins elsewhere.
   - Input/output geometry coherence: measured fraction of time bins assigned to blocked partitions.
   - Result after fix: `206 / 4,963,815 = 4.15e-05`, consistent with negligible bin-boundary effects rather than a systematic mismatch.

### Issues Found and Resolved
- **Geometry input orientation bug**: Initial conversion stored the 9-element geometry vector in the raw blocked-index layout, but output classes used `x_bin * 3 + y_bin`. This caused asymmetric geometries to be misregistered, with `~17.15%` of output time bins appearing in blocked partitions. Resolution: transpose the 3 x 3 open-mask before flattening in `blocked_to_open_vector()`, rebuild `converted_data.pkl`, rerun full verification, and rerun sanity checks. After the fix, blocked-bin occupancy dropped to `4.15e-05` and all raw spot-checks passed.
- **Full-mode inefficiency**: The first full conversion attempt did an unnecessary preload over all animal files before conversion. Resolution: removed the redundant `select_sessions()` full-data pass and reran the full conversion. Runtime returned to the expected few-minute range.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| position_bin_3x3 | 0.6378 | 0.5637 | Chance = 0.1111; validation is 5.07x chance; `predictions.png` and `sample_trials.png` saved by `--plot-samples`. |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| position_bin_3x3 | Validation balanced accuracy 0.5637; training balanced accuracy 0.6378; chance 0.1111 | Paper reports cross-validated within-session Bayesian position decoding with low Euclidean error on finer 15x15 bins rather than categorical balanced accuracy. Strong above-chance decoding on coarse 3x3 bins is consistent with the paper's conclusion that CA1 activity robustly encodes position. |

Validation accuracy is comfortably above the Step 12 thresholds: 0.5637 is 5.07x chance and therefore well above the 1.5x-chance investigation threshold.

The train/validation gap is modest: 0.6378 / 0.5637 = 1.13x, so there is no sign of severe overfitting or leakage.

Paper comparison is necessarily qualitative because the reference analysis uses a different metric and target:
- reference: Gaussian naive Bayes, 5-fold CV, 15x15 spatial bins, Euclidean decoding error
- converted dataset validation: provided downstream decoder, 3x3 spatial bins, balanced classification accuracy

Given that our task is easier spatially (9 bins instead of 225) but uses a different decoder, exact numeric agreement is not expected. The important consistency check is that decoding remains strongly above chance after matching the paper's event-trace loading and temporal binning.

### Issues Found and Resolved
- No new conversion issue was revealed by full decoder training. The Step 10 geometry transpose fix remained validated on the full run.
- No output fell below chance or below the 1.5x-chance investigation threshold.
- No problematic train/validation gap was observed.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Notes:
- No standalone scratch analysis scripts were created during the conversion, so `cache/` only contains `README_CACHE.md`.
- Required outputs (`converted_data.pkl`, validation logs, decoder logs, and processing figures) were retained at the project root for inspection.
