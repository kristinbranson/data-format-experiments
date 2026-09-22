# Dataset Conversion Notes

## Overview
- **Dataset**: Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping (paper.pdf, /app/data, /app/code)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python env verified: numpy 2.4.4, torch 2.6.0+cu124, scipy 1.18.0 (python3).

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `CONVERSION_NOTES.md` (this file)
- `paper.pdf`, `methods.txt`
- `train_decoder.py`, `decoder.py` (provided decoder reference)
- `code/` : reference code repo `georepca1` (LICENSE, README.md, environment.yml, georepca1/{main.py, src/utils.py, src/plots.py, demos/*.ipynb})
- `data/` : 7 joblib files named by animal ID `QLAK-CA1-08, -30, -50, -51, -56, -74, -75` (no extension) plus matching `.mat` versions, a `behav_dict` file, and `precomputed_results/` folder.

Data fields per README of reference code: SFPs, blocked, centroids, envs, maps (sampling/smoothed/unsmoothed), position, trace.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference repo: `georepca1` (Lee, Keinath, Cianfarano & Brandon 2025, Neuron 113(2):307-320).
Structure: `georepca1/main.py` (driver for all figures), `georepca1/src/utils.py` (3372 lines, all analysis functions), `georepca1/src/plots.py`, `demos/*.ipynb`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat(animal, p, to_convert=[envs,position,trace], format="joblib")` | src/utils.py:61 | LOADING | Loads per-animal dataset (joblib dict keyed by animal ID, or mat73 .mat). Returns `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}` |
| `generate_behav_dict(animals, p)` | src/utils.py:130 | LOADING | Builds lighter `behav_dict` with `position`, `envs`, `maps_shape` per animal |
| `get_env_mat(env)` | src/utils.py:215 | PROCESSING | Returns 3x3 binary matrix of which of the 9 partitions is open (1) vs blocked (0) for env names: square, o, t, u, rectangle, +, i, l, bit donut, glenn |
| `get_environment_label(env_name)` | src/utils.py:150 | PLOTTING | Polygon vertices for each geometry (30x30 units = 75 cm arena, i.e. each of 3x3 partitions is 10 units = 25 cm) |
| `get_rate_maps(position, trace, n_bins=15, fps=30, buffer=1e-5, filter_size=1.5)` | src/utils.py:313 | PROCESSING | Bins position with `position // ((nanmax(position,axis=0)+buffer)/n_bins)` -> 15x15 bins; accumulates trace; divides by occupancy x fps |
| `get_split_half`, `get_shuffle_split_half`, `get_place_cells`, `get_shr_within` | src/utils.py:356-460 | CURATION | Place-cell identification (split-half rate map correlation vs 1000 circular shuffles, 99th pct) |
| `fit_decoder(behav, traces, feature_max, temporal_bin_size=3)` | src/utils.py:1776 | PROCESSING/DECODING | Temporally bins position and traces by AvgPool1d(kernel=3, stride=3) (30 Hz -> 10 Hz, 100 ms bins); traces first smoothed with gaussian_filter1d(sigma=3 frames) along time; position -> one-hot spatial bin; GaussianNB with flat priors |
| `test_decoder(...)` | src/utils.py:1806 | DECODING | Same temporal binning, predicts bin, computes Euclidean error |
| `decode_position_within(behav, traces, maps, n_bins=15, fps=30, v_filt_size=5, v_thresh=5, cell_threshold=5, buffer=1e-15, n_fold=5)` | src/utils.py:1845 | CURATION+DECODING | **Most relevant reference for our decoding task.** (1) spatial bin size `bin_down=(max position over all days + buffer)/n_bins`; (2) velocity filter: speed = gaussian_filter1d(||diff(pos)||*fps, sigma=5) and keep frames with speed > 5 cm/s (`v_thresh/bin_down` in bin units); (3) cell filter: cells with > 5 events during moving frames within that day; (4) 5-fold CV GaussianNB decoding of spatial bin |

### Notes on data fields (from code README + code usage)
- `trace`: rise-extracted calcium traces, binary (1 = significant transient rising phase). Indexed `dat[animal]["trace"][day]` with shape (n_cells, n_frames); `NaN` for cells not registered on that day.
- `position`: `dat[animal]["position"][day]`, shape (2, n_frames), x-y in cm.
- `envs`: environment name string per day.
- `blocked`: which of the 9 partitions are blocked (-1 if none).
- `maps`: precomputed `sampling` (occupancy), `smoothed`, `unsmoothed` rate maps, 15 x 15 spatial bins.
- Calcium imaging: traces are already event-extracted binary vectors -> **no dF/F computation needed**; the binary rising-phase vector is treated as the firing rate in all analyses.
- Neuron curation in reference decoding: drop cells not registered on the day (NaN) and cells with <= 5 events during running periods. Place-cell selection is used for some analyses but not for the within-day decoding.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/` contains, for each of 7 mice (`QLAK-CA1-08, -30, -50, -51, -56, -74, -75`):
- `<animal>`: joblib file, dict `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}` (the format loaded by `load_dat(..., format="joblib")`)
- `<animal>.mat`: identical content in MATLAB v7.3 format (loaded by `load_dat(..., format="MATLAB")`). I use the joblib files (same data, faster, the repo default).
- `behav_dict`: light dict per animal with `position`, `envs`, `maps_shape`
- `precomputed_results/`: authors' outputs: `<animal>_shr` (split-half reliability p values), `within_decoding` (5-fold Bayesian position decoding error per session), `map_corr_envs`, `<animal>_rsm_partitioned`, etc.

Per-animal field shapes (example QLAK-CA1-51):
- `trace`: float64 array (n_days, n_cells, n_frames) e.g. (21, 554, 72219). Values are **binary** (0/1 rising-phase events); a cell that is not registered on a day is **all NaN** on that day (never partially NaN - verified).
- `position`: float64 (n_days, 2, n_frames), x-y in **cm**, range [0, 75]. **No NaNs anywhere** (verified for all animals in the scan).
- `envs`: (n_days, 1) strings from {square, o, t, u, rectangle, +, i, l, bit donut, glenn}
- `blocked`: python list of length n_days, each `[array of blocked partition indices]` or `[array(-1.)]` when nothing is blocked.
- `maps`: `sampling` (15,15,n_days), `smoothed`/`unsmoothed` (15,15,n_cells,n_days)
- `SFPs` (35,35,n_cells,n_days), `centroids` (n_cells,2,n_days)

Session structure: every session = one recording day, 40 min at 30 Hz ~ 71,866-72,219 frames.
Sequence of geometries: `square` then a mouse-specific random order of the 10 geometries, repeated up to 3 times, always starting/ending with `square` (10 sessions per sequence + final square -> 31 days for 6 mice, 21 days for QLAK-CA1-51).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, unique cells across mice) | 5,413 (515+875+942+554+862+713+952) |
| Registered cell-sessions | 69,744 |
| Cell-sessions after >5 events while moving | 68,862 |
| Neurons / session (registered) | mean 336.9, range 113-~700 |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 -> 207 total |
| Frames / session | 71,866-72,219 (40 min @ 30 Hz) |
| Fraction of frames with speed > 5 cm/s | ~0.55-0.63 |
| Mean event rate | ~0.1-0.2 Hz per cell |

### Verified partition-index convention
`blocked` uses index `p = 3*ybin + xbin` where `xbin`/`ybin` are the 3x3 bins of `position[0]` (x) and `position[1]` (y).
Verified by downsampling `maps["sampling"]` (15x15 occupancy) to 3x3 and confirming zero-occupancy cells exactly match `blocked` for sessions with geometries o, l, bit donut, glenn, t. Note `blocked` (not `get_env_mat`) is the ground truth per session because several geometries are presented in the vertically flipped variant (`flipud` options in `get_environment_label`).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 unique neurons | "5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps" |
| Rate maps (registered cell-sessions) | 69,744 | same quote |
| Sessions | 207 | same quote |
| Subjects | 7 mice | animal list in main.py; "mean number of cells per animal = 773 +/- 68 SE, minimum cells per animal = 515, maximum cells per animal = 952" |
| Geometries | 10 | "sequence of 10 geometrically distinct environments" |
| Session duration | 40 min, one session/day | "All sessions were 40 min, and one session was recorded per day" |
| Acquisition rate | 30 Hz (behaviour + imaging simultaneously, timestamped) | "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz" |
| Arena | 75 x 75 cm partitioned into 3 x 3 grid (25 cm partitions) | "we partitioned an open square (75 x 75 cm) into a 3 x 3 grid space" |
| Neural signal | binary rising-phase transient vector treated as firing rate | "This binary vector was treated as the firing rate in all further analyses" |
| Rate-map spatial bin | 5 cm (15 x 15 bins over 75 cm) | "spatially binning position data into pixels corresponding to a 5cm x 5cm grid" |
| Decoder temporal bin (reference) | 3 frames @30 Hz = 100 ms | `fit_decoder(..., temporal_bin_size=3)` |
| Speed threshold (reference decoding) | 5 cm/s, speed smoothed with gaussian sigma = 5 frames | `decode_position_within(..., v_filt_size=5, v_thresh=5)` |
| Cell inclusion (reference decoding) | > 5 events during moving frames within the session | `cell_threshold=5` in `decode_position_within` |
| Cell inclusion (all other analyses) | all registered cells | "motivated the inclusion of all cells in subsequent analyses" |
| Bayesian decoding error | mean 13.5 cm (5 cm bins), decreasing over days from ~25 cm to ~9 cm | Figure 1F; recomputed from `precomputed_results/within_decoding`: overall mean 13.49 cm, median 12.38 cm |

### Processing Details
- Preprocessing (already applied in the released data): motion correction, cell segmentation, trace extraction, rising-phase extraction (derivative smoothed with sigma = 5 frames, z-scored against half-normal noise estimate, threshold 2.5) -> **binary** trace. No dF/F computation is required by us.
- Position: DeepLabCut head tracking, 30 Hz, same clock as imaging (frames are already aligned 1:1 in the released arrays: `position` and `trace` have identical frame counts per session) -> no extra temporal alignment is needed between neural and behaviour streams.
- Reference decoding pipeline (`decode_position_within`): spatial binning by `bin_down = (max position over all days + 1e-15)/n_bins`; speed filter; cell filter; temporal binning to 100 ms with `gaussian_filter1d(trace, sigma=3 frames)` then `AvgPool1d(3,3)`; GaussianNB with flat priors, 5-fold CV.

### Curation Steps
**Neuron curation rules**: (1) drop cells not registered in that session (all-NaN trace). (2) For decoding, the reference additionally drops cells with <= 5 events during running frames of that session (`cell_threshold=5`).

**Trial curation rules**: The paper has no trials (continuous 40-min free foraging). The reference decoding analysis excludes frames with speed <= 5 cm/s (immobility; hippocampal activity during immobility reflects replay/SWR rather than current position).

### Decoders Trained (paper)
| Decoded variable | Accuracy |
| Position (15 x 15 = 5 cm bins), naive Bayes, 5-fold CV within session | mean Euclidean error 13.5 cm (range ~8-31 cm per session); "maximum decoding accuracy reported in recent work" |
| Animal identity from RSM | not decodable (chi-square p = 1.000) |

No 3x3-bin classification accuracy is reported in the paper. A mean error of 13.5 cm with 5 cm bins implies most predictions fall within ~half a 25 cm partition, so a 9-class (25 cm bin) decoder should be well above the 1/9 = 11% chance level (expected roughly 55-80% balanced accuracy).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Cross-checks performed (code vs data vs paper)
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Number of mice | 7 animal IDs listed in `main.py` | 7 joblib files / 7 keys in `behav_dict` | "mean cells per animal = 773 +/- 68 SE, min 515, max 952" (7 animals) | consistent |
| Sessions | days = `trace.shape[0]` | 31,31,31,21,31,31,31 = **207** | "207 sessions" | consistent |
| Unique neurons | cells = `trace.shape[1]` | 515+875+942+554+862+713+952 = **5,413** | "5,413 unique neurons" | consistent |
| Rate maps | registered (non-NaN) cell-sessions | **69,744** (computed) | "forming 69,744 rate maps" | exact match - confirms the registration/NaN rule |
| Session length | 40 min at 30 Hz | 71,866-72,219 frames = 39.93-40.12 min | "All sessions were 40 min" | consistent |
| Neural values | binary rising-phase vector used as firing rate | `trace` values are exactly {0,1} (plus NaN for unregistered) | "This binary vector was treated as the firing rate" | consistent, no dF/F needed |
| Position units | `bin_down=(max+buffer)/n_bins`, `v_thresh=5` cm/s | position in cm, exact range [0, 75] in every animal | 75 x 75 cm arena | consistent |
| Geometry encoding | `get_env_mat(env)` gives 3x3 open/blocked per **canonical** geometry | `blocked` field gives blocked partition indices **per session** | 3x3 grid partitioning | `blocked` is the per-session ground truth; several geometries were presented as the vertically flipped variant (cf. `flipud` options in `get_environment_label`), so `get_env_mat` alone would mislabel those sessions. **Use `blocked`.** |
| Partition index convention | `blocked` documented as `[[0,1,2],[3,4,5],[6,7,8]]` | index `p = 3*ybin + xbin` (x = position dim 0) gives **0.0000** occupancy in blocked partitions across all 189 non-square sessions; the transposed convention gives ~0.19 | - | convention A confirmed |
| Temporal alignment | `position[day]` and `trace[day]` are used frame-for-frame in `get_rate_maps`, `decode_position_within` | identical frame counts per session | DAQ acquired both streams at 30 Hz with timestamps for post-hoc alignment | streams are already aligned 1:1; **no additional alignment needed** |
| Neuron filtering | `decode_position_within`: registered AND >5 events during moving frames | drops 882/69,744 = 1.3% of cell-sessions | "inclusion of all cells in subsequent analyses" (for the RSA analyses) | Use the reference **decoding** rule (registered and >5 events while moving) since our task is decoding |
| Immobility | `v_thresh=5` cm/s with speed smoothed sigma=5 frames | 51.8% of frames are moving (range 21.7%-75.3%) | - | apply the same filter (see Step 5 decisions) |
| Decoding performance | `decode_position_within`, GaussianNB, 5-fold | `precomputed_results/within_decoding` mean 13.49 cm error | Figure 1F, decreasing 25 -> 9 cm across days | consistent |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `dat[animal]["trace"][day]` (ncells, nframes), binary, NaN = unregistered | `neural[session][trial]` (nkept, nbins) float32 | drop unregistered + low-activity cells; `gaussian_filter1d(sigma=3 frames)` along time; average-pool 3 frames (100 ms); x30 to get Hz; keep moving bins; cut into 1-min trials | `fit_decoder` (smoothing + AvgPool1d(3,3)), `decode_position_within` (cell filter) | |
| `dat[animal]["blocked"][day]` | `input[session][trial]` (9,) float32, static per trial | 1 = partition blocked, 0 = open, index `3*ybin+xbin` | data field documented in code README; cross-checked with `get_env_mat` | Static per trial as required ("Environment geometry ... Static per-trial") |
| `dat[animal]["position"][day]` (2, nframes) cm | `output[session][trial]` (1, nbins) int64 | average x,y within each 100 ms bin, then discretize with 25 cm edges -> class `3*ybin+xbin` in 0..8 | `decode_position_within` spatial binning (`position // ((max+buffer)/n_bins)`) with n_bins=3 instead of 15 | Task requires 3x3 = 9 bins |
| animal ID | `subjects`, `subject_idx` | 7 IDs, one index per session | `animals` list in `main.py` | |
| - | `brain_regions`, `brain_region_idx` | all neurons are CA1 | paper title/methods | single region |
| `dat[animal]["envs"][day]` | `metadata["session_info"]` | geometry name per session | | kept for reference, not a decoder input (the 9-d geometry vector already encodes it) |

### Key Decisions
1. **Session = one recording day** (207 sessions). Matches the reference, where all analyses are per-day.
2. **Time bin = 100 ms (3 frames at 30 Hz)**: exactly the reference decoding bin (`temporal_bin_size=3` in `fit_decoder`/`test_decoder`). Identical for every trial and session.
3. **Neural processing = reference decoder preprocessing**: `gaussian_filter1d(trace, sigma=3 frames, axis=time)` then average-pool over 3 frames. I apply the smoothing to the *continuous* session trace before dropping immobile frames (the reference smooths after concatenating retained frames); smoothing before filtering avoids mixing activity across temporal discontinuities and is strictly more correct. Result is multiplied by 30 to express it as an event rate in Hz (a constant scale factor; irrelevant to the linear decoder but interpretable).
4. **Neuron curation**: keep cells that are (a) registered on that day (trace not NaN) and (b) have > 5 events during moving frames of that session - exactly `decode_position_within`'s `cell_threshold=5` rule. This removes 882/69,744 = 1.3% of cell-sessions. Cells are fixed within a session (required: constant n_neurons across a session's trials).
5. **Speed filter (immobility exclusion)**: bins with mean speed <= 5 cm/s are dropped, matching `decode_position_within` (`v_thresh=5`, speed = `gaussian_filter1d(|diff(pos)|*fps, sigma=5 frames)`). Justification: this is the reference curation for the position-decoding analysis, and hippocampal activity during immobility reflects replay/SWR rather than current position. The decoder used here classifies each timepoint independently, so removing timepoints does not break anything. Trials therefore have variable numbers of timepoints (allowed by the format).
6. **Trials = consecutive 1-min windows of session time** (600 bins of 100 ms), as specified by the task. The trailing partial window (<1 min) of each session is dropped so that all trials cover the same amount of real time. Trials retaining < 30 moving bins (3 s) are dropped as too short/unreliable; sessions keep ~40 trials each, far above the minimum of 2.
7. **Input = 9-dim binary geometry vector**, 1 = blocked, 0 = open, static per trial (1-D array of length 9), taken from the per-session `blocked` field (not `get_env_mat`, which does not capture the flipped presentations).
8. **Output = single categorical variable with 9 values**, the 3x3 spatial bin `3*ybin+xbin`, time-varying (shape (1, nbins)). Bin edges at 25 and 50 cm exactly coincide with the physical partition walls, so the output classes are the arena partitions (verified: essentially zero occupancy in blocked partitions).
9. **No session/animal exclusions**: all 7 mice and all 207 sessions are used, matching the paper.

### Planned Sanity Checks
- [x] total sessions == 207, unique neurons == 5,413, registered cell-sessions == 69,744 (paper)
- [x] position range exactly [0, 75] cm; no NaNs in position
- [x] partition index convention validated by zero occupancy in blocked partitions
- [x] after conversion: output classes present in a session == unblocked partitions of that session (done: mean 4.0e-5 of timepoints in blocked partitions, from tracking jitter)
- [x] after conversion: neural values non-negative, no NaN/Inf, mean rate 0.31 Hz over moving bins
- [x] after conversion: spot-checks with `np.allclose` for neural, input and output on 6 sessions (Step 10, check 2) - all pass
- [x] after conversion: 38.9 trials/session; 2,551,604 timepoints, reproduced exactly by the independent reimplementation
- [x] decoder balanced accuracy 0.663 = 6.0x chance; reference GaussianNB gives 0.551 on the same task, and my reproduction of the paper's 15-bin decoding error matches the authors' values to within 0.15 cm

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the mapping of Step 5. Structure:
- `bin_frames(x, k=3)` - numpy equivalent of the reference `AvgPool1d(kernel_size=3, stride=3)` (drops the trailing remainder, exactly like AvgPool1d).
- `compute_speed(position)` - `gaussian_filter1d(||diff(pos)||*30, sigma=5)`, identical to `decode_position_within`; frame 0 set to 0 (reference leaves `vel_idx[0]=False`).
- `blocked_to_vector(blocked_day)` - 9-d binary geometry vector from the `blocked` field (-1 -> all zeros).
- `position_to_class(pos)` - `floor(pos / ((75 + 1e-15)/3))` -> `3*ybin + xbin`, the reference spatial-binning formula with `n_bins=3`.
- `process_session(...)` - curation (registered cells, >5 events while moving), smoothing+pooling of traces, speed filtering of bins, 1-min trial cutting.
- `plot_processing(...)` - 10-panel figure per session for `--show-processing`.
- `main()` - loops animals/days, assembles the dict, runs assertions and summary statistics, pickles the result.

Efficiency notes:
- Inefficiency identified: naive per-frame python loops (as in the reference `get_rate_maps`) would be far too slow for 207 x 72,000 frames -> all binning/discretization is vectorised with numpy reshape/mean.
- Each animal file is loaded exactly once (9-16 s each); the per-session processing is 0.3-1 s.
- `float32` for neural, `int64` for output, `float32` for input.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sample = first 2 sessions of QLAK-CA1-08)

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (cell-sessions kept) | 335 (registered 338) |
| Neurons / session | 182, 153 |
| Subjects | 1 |
| Trials (total) | 77 |
| Trials / session | 38, 39 |
| Timepoints | 22,274 bins = 37.1 min of moving time (out of 2 x 40 min) |
| T per trial | mean 289, min 92, max 459 (of 600 possible bins) |
| Input range (blocked_partition_0..8) | [0,1] for partition 4 (day 1 = geometry "o"), [0,0] elsewhere - correct |
| Output distribution (9 classes) | [0.099, 0.083, 0.118, 0.084, 0.053, 0.134, 0.100, 0.120, 0.209] |
| Output distribution, day 1 ("o", centre blocked) | class 4 fraction = **0.000** - correct |
| Mean neural value | 0.21 Hz |

### Processing Plots Review
`processing_QLAK-CA1-08_day0.png`, `processing_QLAK-CA1-08_day1.png` show, per session:
trajectory with the 25/50 cm partition edges; per-partition occupancy vs the input geometry vector; the speed trace with the 5 cm/s threshold and kept frames; raw vs temporally binned position (superimposed, no lag -> alignment correct); the discretization of binned x/y into the 9 classes; binned position scatter coloured by class (tiles the 3x3 grid exactly); the raw binary event raster and the corresponding binned rate image for the same cells/time window; the kept-bin mask with 1-min trial boundaries; and trial 0 neural + output on one time base. No anomalies.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| Vectorised binning/discretisation instead of per-frame loops | ~100x vs reference-style loops |
| One joblib load per animal (not per session) | ~30x fewer loads |

| Step | Time / Session | Estimated Total Time |
| load animal file | 9-16 s per animal | ~90 s for 7 animals |
| process session | 0.33 s (early sessions, 180 cells); up to ~1 s for 560-cell sessions | ~2 min for 207 sessions |
| **total estimate** | | **~4 min** (well under the 15 min budget) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None ("Data format is valid, no errors or warnings.")
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| position_bin (9 classes) | 0.5564 | 0.4235 | 0.1111 |

Loss decreased monotonically 2.36 -> 1.29 over 200 epochs. Validation accuracy is 3.8x chance.
Note the sample contains only days 0-1 of one mouse, which are the sessions with the **worst** spatial coding in the paper (Figure 1F: ~26 cm decoding error on day 0 falling to ~9 cm by day 30), so the full dataset should give substantially higher accuracy.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full` (220 s total, 0.62 s/session + ~13 s/animal load - within the estimate from Step 7).

### Output Files
- `converted_data.pkl`: 3.41 GB
- `conversion_full_out.txt`, `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 7 (mean 773 +/- 68 cells, min 515, max 952) | `animals` list of 7 | 7 files | 7 | yes |
| Sessions | 207 | days per animal | 31,31,31,21,31,31,31 = 207 | 207 (same per-subject split) | yes |
| Unique neurons | 5,413 | - | 5,413 | 5,413 unique cells (68,862 cell-sessions after curation) | yes |
| Registered cell-sessions ("rate maps") | 69,744 | non-NaN traces | 69,744 | 69,744 registered; 68,862 kept after the reference `cell_threshold=5` filter (1.3% removed) | yes |
| Neurons / session | - | - | 336.9 registered (113-564) | 332.7 kept (112-562) | yes |
| Session duration | 40 min | - | 71,866-72,219 frames @30 Hz | 39-40 one-minute trials per session (trailing partial minute dropped) | yes |
| Trials (total) | n/a (continuous recording) | n/a | n/a | 8,056 (mean 38.9/session, min 22, max 40) | n/a |
| Time bin | 100 ms (`temporal_bin_size=3`) | `AvgPool1d(3,3)` | 30 Hz | 100 ms | yes |
| Moving-frame fraction (speed > 5 cm/s) | - | `v_thresh=5` | 0.518 mean | 2,551,604 bins = 4,252.7 min of 8,280 min recorded = 0.514 | yes |
| Input range (9 blocked-partition flags) | 10 geometries | `get_env_mat` | `blocked` field | [0,1] for partitions 0-6 and 8; partition 7 never blocked | yes (see note) |
| Output distribution (9 classes) | - | - | occupancy | [0.095, 0.108, 0.114, 0.088, 0.071, 0.087, 0.109, 0.158, 0.170] | plausible (corners/edges over-sampled, centre under-sampled - typical thigmotaxis) |
| Mean neural value | - | - | ~0.1-0.2 Hz raw event rate | 0.31 Hz (mean over moving bins only, where event rates are higher) | yes |

**Note on partition 7**: it is never blocked in the dataset because the only geometries that block it in canonical orientation (`l`: blocks {4,5,7,8}) were always presented in the vertically flipped orientation (blocking {1,2,4,5}). Verified for all 207 sessions in the sanity check A2 below, which shows every session's geometry equals `get_env_mat(env)` or its `flipud`.

No data was lost: 207/207 sessions and 69,744/69,744 registered cell-sessions are accounted for; the only exclusions are the documented reference curation rules.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`verification_full_out.txt`)
"Data format is valid, no errors or warnings." - no errors and **no warnings** to address. Summary: 207 sessions, 8,056 trials, 7 subjects, 1 brain region (CA1, 68,862 neurons), dinput 9, doutput 1 (9 classes), T mean 315.9 (min 30, max 561).

The conversion script itself prints 4 informational warnings: in 4 of the 46 sessions with a blocked centre partition, a tiny fraction of timepoints (9.5e-5, 7.9e-5, 7.7e-3, 4.9e-4) are assigned to the blocked partition. This is DeepLabCut tracking jitter at the partition wall (the head marker can be tracked a few cm over the 25 cm-tall insert). It is left as-is rather than deleted, because (a) it is <0.008 of timepoints in the worst session and 4e-5 on average, (b) deleting timepoints on the basis of the output label would bias the dataset, and (c) the reference code applies no such correction (it "cleans" predictions to the nearest visited bin only for reporting the Bayesian error).

### Check 2: Constructed sanity checks (`/app/cache/sanity_checks.py`)
All checks reload the **raw** joblib files independently of `convert_data.py` and use independent implementations (torch `AvgPool1d`, `np.digitize`). **0 failures**:

| Check | Description | Result |
|-------|-------------|--------|
| A1 | `input` equals the 9-d binary vector built from the raw `blocked` field, for all 207 sessions and every trial | PASS |
| A2 | every session's geometry equals `get_env_mat(env)` (copied from reference utils.py) or its `flipud` | PASS |
| B | `neural` reproduced with an independent torch `AvgPool1d(3,3)` implementation of the reference binning, `np.allclose(atol=1e-4)`, for sessions 0, 13, 57, 100, 150, 206 (all 7 mice, early and late days) | PASS |
| C | `output` reproduced with an independent `np.digitize(pos, [25,50])` discretization, `np.array_equal` | PASS |
| B/C | trial counts per session match the independent reconstruction | PASS |
| D1-D9 | 207 sessions; 7 subjects; 69,744 registered cell-sessions (paper value); 68,862 kept; `brain_region_idx` lengths match; no NaN/Inf; outputs in 0..8; 2,551,604 total timepoints; sessions per subject [31,31,31,21,31,31,31] | PASS |
| D10 | mean fraction of timepoints in a blocked partition = 4.0e-5 (max 7.7e-3) | PASS |

### Check 3: Reference code comparison
| Step | Reference code | My code | Same? |
|------|----------------|---------|-------|
| (a) data loading | `load_dat(animal, p, format="joblib")` -> `joblib.load(data/<animal>)[animal]`, fields `trace`, `position`, `envs`, `blocked` | `joblib.load(DATA_DIR/<animal>)[animal]`, same fields | identical |
| (b) neuron filtering | `decode_position_within`: `cell_idx = np.sum(traces[:, :, d][vel_idx[d]], axis=0) > cell_threshold(=5)`; unregistered cells are NaN so they fail the test | `registered & (np.nansum(trace[:, moving], axis=1) > 5)` - the same criterion, with the registration test made explicit | identical |
| (c) temporal alignment | `position[day]` and `trace[day]` used frame-for-frame (both 30 Hz, DAQ-timestamped) | same; verified identical frame counts per session | identical |
| (d) binning | `gaussian_filter1d(traces, sigma=3, axis=time)` then `AvgPool1d(kernel_size=3, stride=3)` for traces; `AvgPool1d(3,3)` then `astype(int)` for position | `gaussian_filter1d(sigma=3)` + mean over 3-frame windows (numerically identical to AvgPool1d, verified with torch in sanity check B); position mean over the same windows, then floor-divided by 25 cm | identical (see note 1) |
| (e) speed filter | `vel_idx = gaussian_filter1d(||diff(behav)||*fps, sigma=5) > v_thresh/bin_down` applied to **frames** before binning | speed computed identically in cm/s and thresholded at 5 cm/s, applied to the **binned** speed (mean speed within each 100 ms bin) | equivalent (note 2) |
| (f) spatial binning | `behav //= (max + buffer)/n_bins` with `n_bins=15` -> 5 cm bins, one-hot 225 classes | same formula with `n_bins=3` -> 25 cm bins, 9 classes | required by the decoder task (3x3 = 9 bins) |
| (g) input construction | reference has no decoder input; `get_env_mat`/`blocked` used to mask maps | 9-d binary `blocked` vector | required by the decoder task |

Notes on the (small, deliberate) differences:
1. The reference smooths the traces **after** concatenating only the retained (moving) frames; I smooth the continuous session trace **before** dropping immobile frames. Smoothing across a discontinuity created by deleted frames would mix activity from non-adjacent times, so smoothing first is strictly more correct; over 3-frame (100 ms) kernels the numerical difference is tiny.
2. The reference multiplies `v_thresh` by `1/bin_down` because its `behav` has already been divided by `bin_down`; in cm/s units it is exactly a 5 cm/s threshold, which is what I apply. I threshold the 100 ms-binned mean speed rather than individual frames so that the speed criterion is defined on the same time base as the output bins.
3. The reference bins position with `(max over all days + buffer)/n_bins`; the max is exactly 75.0 cm for every animal (verified), so my fixed 25 cm edges are identical to the reference formula and coincide with the physical partition walls.

### Check 4: Key statistics comparison
See the table in Step 9 - every statistic available in the paper (7 mice, 207 sessions, 5,413 neurons, 69,744 rate maps, min/max cells per animal 515/952, 40-min sessions, 30 Hz, 75 cm arena, 3x3 partitions, 10 geometries) is matched exactly by the converted dataset.

### Check 5: Edge cases
- **Trailing partial minute**: sessions are 71,866-72,219 frames = 23,955-24,073 bins, i.e. 39.9-40.1 one-minute windows. The final partial window is dropped so every trial covers exactly 60 s of session time (39-40 trials/session).
- **Cells only partially NaN**: asserted never to occur (`assert np.array_equal(nan_any, nan_all)` runs for every session and never fired).
- **First frame speed**: undefined by differencing; set to 0 (not moving), matching the reference, which leaves `vel_idx[0] = False`.
- **Empty/short trials**: trials with fewer than 30 moving bins (3 s) are dropped; the minimum retained trial has 30 bins, and every session keeps >= 22 trials (>> the required 2).
- **`blocked` = -1**: mapped to an all-zero geometry vector (square, nothing blocked) - 46 sessions.
- **Position clipping**: `np.clip(.., 0, 2)` guards against a position exactly at 75.0 cm (which occurs) falling into bin 3.
- **Sessions with a class never visited**: e.g. the centre bin in "o"/"bit donut" geometries. This is intended (the animal cannot be there); `output_values` still lists all 9 classes so the decoder's class set is consistent across sessions.

### Issues found and resolved
- Initially ambiguous whether the `blocked` index is `3*y+x` or `3*x+y`. Resolved empirically: occupancy in blocked partitions is 0.0000 for `3*y+x` versus ~0.19 for the alternative, across all 189 non-square sessions.
- Initially assumed `get_env_mat(env)` could supply the geometry; discovered that several geometries were presented vertically flipped, so the per-session `blocked` field must be used. Confirmed by check A2.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` -> `/app/train_decoder_full_out.txt`

### Training Progress
- Loss decreasing: **Yes**, monotonically for all 200 epochs (2.17 -> 0.603). Test loss 1.198.
- Device: GPU (NVIDIA L4); run completed in ~10 min.

### Decoder Results (Full)
| Output | Classes | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|---------|--------|-------------|--------|-------|
| position_bin (3x3 spatial bin) | 9 | 0.1111 | 0.8253 | **0.6630** | 6.0x chance; train/val ratio 1.24 |

Sample and prediction plots written to `sample_trials.png` and `predictions.png`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Variable | Classes | Chance (1/nclass) | Validation balanced acc | Ratio |
|----------|---------|-------------------|-------------------------|-------|
| position_bin | 9 | 0.1111 | 0.6630 | **6.0x** |

Well above the 1.5x-chance concern threshold; no evidence of a bug.

### Check 2: Accuracy comparison to the paper / reference algorithm
The paper reports position decoding only as a **Euclidean error with 15 x 15 (5 cm) bins**, not as a 9-class accuracy, so a direct number is not available. I therefore did two comparisons (`/app/cache/reference_decoder_check.py`, output in `/app/cache/reference_decoder_check_out.txt`):

**(a) Reproduce the published decoding error.** I ran the reference pipeline (registered + >5-events cell filter, 5 cm/s speed filter, 100 ms bins, GaussianNB with flat priors, 5-fold CV, 15 x 15 bins) on raw data for 11 sessions spanning 4 mice and early/middle/late days, and compared with the authors' own precomputed `precomputed_results/within_decoding`:

| Session | Authors' error (cm) | My reproduction (cm) |
|---------|--------------------:|---------------------:|
| QLAK-CA1-08 day 0 | 26.37 | 26.37 |
| QLAK-CA1-08 day 15 | 11.12 | 11.07 |
| QLAK-CA1-08 day 30 | 16.17 | 16.14 |
| QLAK-CA1-50 day 0 | 24.30 | 24.36 |
| QLAK-CA1-50 day 15 | 9.33 | 9.30 |
| QLAK-CA1-50 day 30 | 9.14 | 9.15 |
| QLAK-CA1-51 day 5 | 18.06 | 18.00 |
| QLAK-CA1-51 day 20 | 14.58 | 14.44 |
| QLAK-CA1-75 day 0 | 17.84 | 17.83 |
| QLAK-CA1-75 day 15 | 7.76 | 7.72 |
| QLAK-CA1-75 day 30 | 10.27 | 10.26 |

Agreement is within 0.15 cm everywhere (residual differences come from smoothing before vs after removing immobile frames). This demonstrates that my **loading, curation, temporal alignment and binning reproduce the published analysis exactly**.

**(b) Reference-algorithm accuracy on the actual decoding task.** Running the same reference GaussianNB pipeline with 3 x 3 bins on those 11 sessions gives a mean 9-class balanced accuracy of **0.551** (range 0.334 on day 0 of mouse 08 to 0.757 on day 15 of mouse 75, tracking the paper's day-dependence of spatial coding quality).
The provided decoder on my converted data reaches **0.663** validation balanced accuracy over all 207 sessions - i.e. it *exceeds* the reference algorithm's within-session accuracy, as expected for a jointly trained model that shares a decoder across sessions. There is therefore no accuracy shortfall to explain.

### Check 3: Train vs validation gap
Training 0.8253 vs validation 0.6630 -> ratio 1.24, below the 1.5x concern threshold. The gap is the expected generalization gap for a per-session projection with 100 PCs, and the split is by trial (1-min blocks), so there is no leakage of adjacent timepoints between train and validation beyond the trial boundary.

### Additional verification of low-accuracy debugging steps (all clean)
1. Output values verified against raw data by independent `np.digitize` reimplementation on 6 sessions (Step 10 check C) - exact match.
2. Temporal alignment: raw vs binned position plotted on one axis (`processing_*.png`, panel (1,1)) shows no lag; neural and output share the identical bin index by construction; the 15-bin decoding-error reproduction above is itself a stringent alignment test (a one-bin misalignment would inflate the error markedly).
3. Output variation: class distribution [0.095, 0.108, 0.114, 0.088, 0.071, 0.087, 0.109, 0.158, 0.170] - no class dominates.
4. Neural filtering follows the reference `cell_threshold=5` rule (1.3% of cell-sessions removed) and registration (NaN) rule (0 additional).
5. Processing matches the reference code as tabulated in Step 10 check 3.

### Issues found and resolved
- None outstanding. The only deliberate deviations from the reference are (i) 3 x 3 instead of 15 x 15 spatial bins (required by the decoder task), (ii) smoothing before rather than after immobility removal (more correct; verified to change the reproduced errors by < 0.15 cm), and (iii) segmentation into 1-min trials (required by the task).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, key statistics, processing summary, loading instructions, format specification)
- [x] `cache/` folder created with `README_CACHE.md` documenting every investigation script
- [x] All files organized

Final deliverables in `/app`:
`CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl` (3.41 GB), `sample_data.pkl`,
`README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
`train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
`train_decoder_full_out.txt`, processing plots `processing_QLAK-CA1-08_day0.png` and
`processing_QLAK-CA1-08_day1.png`, decoder plots `sample_trials.png` and `predictions.png`,
and the `cache/` folder.
