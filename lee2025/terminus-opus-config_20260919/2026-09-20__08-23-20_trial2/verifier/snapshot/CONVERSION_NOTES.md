# Dataset Conversion Notes

## Overview
- **Dataset**: CA1 recordings - "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping"
- **Date**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: python3 with numpy 2.4.4, torch 2.6.0+cu124 (CUDA available), joblib, scipy, sklearn available.

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `CONVERSION_NOTES.md` (this file)
- `code/` : georepca1 reference repository (README.md, environment.yml, georepca1/main.py, georepca1/src/utils.py, georepca1/src/plots.py, demos/*.ipynb)
- `data/` : 15 GB. Per-animal joblib files (no extension) and MATLAB `.mat` twins for 7 animals:
  QLAK-CA1-08, -30, -50, -51, -56, -74, -75; plus `behav_dict` (position+envs only) and `precomputed_results/`
  (rsm_partitioned, *_shr split-half p-values, within_decoding, etc.)
- `methods.txt`, `paper.pdf`
- `decoder.py`, `train_decoder.py` (validation/decoder scripts)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat(animal, p, format)` | src/utils.py:61 | LOADING | Loads per-animal dict (joblib or .mat via mat73). Returns `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}` |
| `mat2joblib` / `save_dat` | src/utils.py:87-107 | LOADING | Convert .mat -> joblib; the joblib and .mat files contain identical data |
| `generate_behav_dict` | src/utils.py:130 | LOADING | Makes lightweight dict with `position`, `envs`, `maps_shape` for all animals -> data/behav_dict |
| `get_env_mat(env)` | src/utils.py:215 | PROCESSING | Maps environment name (square, o, t, u, rectangle, +, i, l, bit donut, glenn) -> binary 3x3 matrix; 0 = blocked/omitted partition. Ordering [[0,1,2],[3,4,5],[6,7,8]] |
| `get_environment_label` | src/utils.py:150 | PLOTTING | Polygon vertices of each geometry |
| `get_rate_maps(position, trace, n_bins=15, fps=30, buffer=1e-5, filter_size=1.5)` | src/utils.py:313 | PROCESSING | Bins position into 15x15 by integer division by `(nanmax(position,axis=0)+buffer)/n_bins`; sums binary trace per bin; divides by occupancy; multiplies by fps=30 |
| `get_split_half`, `get_shuffle_split_half`, `get_place_cells`, `get_shr_within` | src/utils.py:356-462 | CURATION | Split-half rate-map reliability vs 1000 circular shuffles; place cells = p < alpha (paper: 99th pct) |
| `clean_rate_maps(maps, envs)` | src/utils.py:463 | PROCESSING | NaNs pixels outside the environment; mask = `np.fliplr(get_env_mat(env).T)` expanded 3x3 -> 15x15 (5x5 pixels per partition). This defines the orientation convention between position bins and the 3x3 partition matrix |
| `fit_decoder(behav, traces, feature_max, temporal_bin_size=3)` | src/utils.py:1776 | PROCESSING/DECODE | Temporal binning: `AvgPool1d(kernel_size=3, stride=3)` on position & traces (30 Hz -> 10 Hz, 100 ms bins); traces first smoothed with `gaussian_filter1d(sigma=3 frames)` along time; position bins -> one-hot class; GaussianNB with flat priors |
| `test_decoder` | src/utils.py:1806 | DECODE | Same binning on test set; error = Euclidean distance in cm between decoded and true bin |
| `decode_position_within(behav, traces, maps, n_bins=15, fps=30, v_filt_size=5, v_thresh=5, cell_threshold=5, n_fold=5)` | src/utils.py:1845 | CURATION/DECODE | Position scaled with a single `bin_down = (max over days & axes + buffer)/n_bins`; speed = `gaussian_filter1d(|diff(position)|*fps, sigma=5)` and frames with speed <= 5 cm/s are DROPPED; cells kept if summed events during moving frames > 5; 5-fold CV within day |

### Notes
- Neural data are already preprocessed: `trace` is the **binarized rising-phase** vector (1 = significant calcium event) - no dF/F computation needed.
- Cells not registered on a given day appear as **NaN** for that day (both in `trace` and `maps`), so per-session neuron selection = non-NaN cells.
- The paper/reference decoding pipeline uses 100 ms temporal bins (3 frames at 30 Hz) with a sigma=3-frame Gaussian smoothing of the binary traces, which is directly reusable for the neural decoder here.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` contains, for each of 7 mice (`QLAK-CA1-08, -30, -50, -51, -56, -74, -75`):
- `<animal>` : joblib dump of `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}`
- `<animal>.mat` : identical content in MATLAB v7.3 format (loadable with mat73). The joblib file is what
  `load_dat(..., format="joblib")` reads, so I use it (identical data, ~5x faster to load).
- `behav_dict` : light-weight dict with `position`, `envs`, `maps_shape` for all animals
- `precomputed_results/` : paper results (`*_shr` split-half p-values, `within_decoding` Bayesian decoding errors,
  `*_rsm_partitioned`, etc.)

Fields per animal (example QLAK-CA1-51, 21 sessions, 554 cells, 72,219 frames):
| Field | Shape | Meaning |
|-------|-------|---------|
| `trace` | (n_days, n_cells, n_frames) float64 | **binarized rising-phase calcium events** (0/1). NaN for the whole session if a cell was not registered that day |
| `position` | (n_days, 2, n_frames) float64 | x,y head position in cm, range [0, 75]; no NaNs anywhere |
| `envs` | (n_days, 1) str | geometry name: square, o, t, u, rectangle, +, i, l, bit donut, glenn |
| `blocked` | list of n_days | indices (0-8) of blocked partitions in the 3x3 grid, `-1` if none |
| `maps.smoothed` / `.unsmoothed` | (15,15,n_cells,n_days) | rate maps (events/frame), NaN outside visited bins / unregistered cells |
| `maps.sampling` | (15,15,n_days) | dwell time per 5x5 cm bin **in seconds** (frame counts / 30) |
| `SFPs` | (35,35,n_cells,n_days) | spatial footprints |
| `centroids` | (n_cells,2,n_days) | ROI centroids |

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, unique cells summed over animals) | 5,413 (515+875+942+554+862+713+952) |
| Registered neurons / session | mean 336.9, min 113, max 564 (total cell-sessions = **69,744**) |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 -> **207 sessions** |
| Frames / session | 71,866 - 72,219 (40 min @ 30 Hz) |
| Trials (total, if 1-min non-overlapping trials) | 207 sessions x 39-40 = **8,247** |
| Position range | 0 - 75 cm in both x and y |
| trace values | strictly {0, 1} (plus whole-session NaN when unregistered) |

### Partition (3x3) indexing convention - determined empirically
Partition index `p = 3*floor(y/25) + floor(x/25)` (x,y in cm).
Checked against the `blocked` field for all 207 sessions: only 0.003% of frames fall in a partition listed as
blocked (tracking noise), i.e. the convention is correct. Equivalently, `blocked` =
`np.flatnonzero(np.flipud(get_env_mat(env)).ravel()==0)`, consistent with the mask used in the reference
`clean_rate_maps` (`np.fliplr(get_env_mat(env).T)` on the (x,y)-indexed 15x15 maps).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 | "5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps" |
| Cell-sessions (rate maps) | 69,744 | same |
| Neurons / animal | mean 773 +/- 68 SE, min 515, max 952 | "(mean number of cells per animal = 773 +/- 68 SE, minimum cells per animal = 515, maximum cells per animal = 952)" |
| Subjects | 7 (implied: 207 sessions, up to 3 sequences of 10 geometries + squares) | |
| Sessions / subject | 31 (one animal 21) | "one session was recorded per day"; "sequence repeated up to three times" |
| Session duration | 40 min | "All sessions were 40 min, and one session was recorded per day" |
| Sampling rate | 30 Hz | "DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz ... all recorded frames were timestamped for post-hoc alignment" |
| Arena | 75 x 75 cm square, 3x3 partitions of 25 cm | "partitioned an open square (75 x 75 cm) into a 3 x 3 grid space" |
| Spatial bin (rate maps) | 5 x 5 cm (15x15 bins) | "spatially binning position data into pixels corresponding to a 5cm x 5cm grid" |
| Neural signal | binary rising-phase vector treated as firing rate | "The final binarized rising-phase vector was then set to 1 ... This binary vector was treated as the firing rate in all further analyses" |
| Place cells | split-half reliability > 99th pct of 1000 circular shuffles | "Place cells were identified as those whose split-half rate map correlation exceeded the 99th percentile" |
| Bayesian decoding | 5-fold CV within session, error = Euclidean distance between predicted and true 5 cm bin | "we performed a 5-fold split of spatially binned position and trace data ... Decoding error ... Euclidean distance" |

### Processing Details
- Neural data are **already fully preprocessed** (motion correction, CNMF-E segmentation, rising-phase
  binarization). No dF/F computation is required; the binary event vector *is* the firing rate.
- Cells registered across days with CellReg; a cell not registered on a day is NaN for the whole session.
- Temporal alignment: behaviour and calcium are recorded by the same DAQ at 30 Hz and timestamp-aligned by the
  authors, so `position[d][:, t]` and `trace[d][:, t]` are already sample-for-sample aligned. No further
  alignment is needed (verified by exactly reproducing the stored rate maps, see Step 4).
- Reference *decoder* temporal binning (`fit_decoder`): traces smoothed with `gaussian_filter1d(sigma=3 frames)`
  along time, then `AvgPool1d(kernel_size=3, stride=3)` -> **100 ms bins**; position averaged over the same bins
  and then floor-divided into spatial bins.

### Curation Steps
**Neuron curation rules**:
- Include only cells registered in that session (non-NaN) - this is what produces the 69,744 rate maps.
- Paper: "These results ... motivated the inclusion of **all cells** in subsequent analyses" (no place-cell or
  reliability threshold is applied for the main analyses).
- The reference Bayesian decoder (`decode_position_within`) additionally keeps only cells with > 5 events during
  running frames; in this dataset that would drop only 882/69,744 = 1.3% of cell-sessions (computed directly).

**Trial curation rules**:
- The paper has no trial structure (continuous 40 min free foraging). For this task sessions are cut into
  non-overlapping 1-min trials; a trailing partial minute is dropped.
- The reference decoder drops frames with speed <= 5 cm/s (`v_thresh=5`, speed smoothed with sigma=5 frames).
  Note that the paper's rate maps themselves are built from **all** frames (verified by exact reproduction),
  so speed filtering is specific to their Bayesian decoding analysis.

### Decoders Trained (reference)
| Decoded variable | Accuracy |
| Position (15x15 = 5 cm bins), Gaussian naive Bayes, 5-fold CV within session | Mean Euclidean error per animal: 16.5, 12.8, 11.4, 17.6, 15.2, 11.4, 10.9 cm (from `precomputed_results/within_decoding`); error decreases across days (ANOVA p<0.0001, F=7.98), reaching ~8-10 cm in late sessions |

An error of ~11-17 cm against a 25 cm partition width implies that a 3x3 (25 cm) position decoder should be well
above the 1/9 = 11% chance level (roughly 50-80% correct), which is the expectation for this conversion.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Cross-checks performed (code vs data vs paper)
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset size | 7 animals in `main.py` `animals` list | 7 files, 207 sessions, 5,413 cells, 69,744 registered cell-sessions | "5,413 unique neurons across 207 sessions ... 69,744 rate maps" | **Exact match** |
| Cells per animal | - | 515, 875, 942, 554, 862, 713, 952 (mean 773.3) | mean 773 +/- 68 SE, min 515, max 952 | **Exact match** |
| Session length | `fps=30` | 71,866-72,219 frames = 39.9-40.1 min | 40 min at 30 Hz | Match |
| Neural signal | `trace` used directly as rate | values in {0,1} | binarized rising phase | Match; no dF/F needed |
| Spatial binning | `get_rate_maps`: `position // ((nanmax(position,axis=0)+1e-5)/15)` | stored `maps.unsmoothed` are exactly reproduced with a **global** bin width of (75+1e-5)/15 = 5 cm and **no** fps multiplication | 5 cm pixels | Stored maps = events/frame using a fixed 0-75 cm grid. `get_rate_maps` as published uses the per-session max and multiplies by fps, so it only reproduces the stored maps up to a factor of 30 and a per-session scaling; the stored maps (and the 75 cm arena) show the intended binning is the **fixed 5 cm grid over 0-75 cm**. I therefore use the fixed grid (25 cm for the 3x3 partitions). |
| `maps.sampling` | described as "occupancy" | equals frame counts / 30 | - | It is dwell time in seconds, not a probability |
| Blocked partitions | `get_env_mat` 3x3 matrices, `blocked` = indices | `blocked` matches low-occupancy partitions with index `3*floor(y/25)+floor(x/25)` in all 207 sessions | "organized in the following way - [[0,1,2],[3,4,5],[6,7,8]]" | Convention resolved: `blocked` indexes `np.flipud(get_env_mat(env)).ravel()` |
| Cell inclusion | `decode_position_within` uses >5 events on moving frames | filter would remove 1.3% of cell-sessions | "motivated the inclusion of all cells" | Use all registered cells (keeps the 69,744 statistic); documented in Step 5 |
| Temporal alignment | position and trace indexed by the same frame index | reproducing stored rate maps from `position`+`trace` works to 1 frame in 72,219 | frames timestamped and aligned by DAQ | No extra alignment needed |

### Sanity check already run (details in Step 10)
Recomputing `maps['unsmoothed']` for animal QLAK-CA1-51, day 0 from raw `position` and `trace` with a fixed
5 cm grid reproduces the stored maps exactly for cell 0 and to <1e-5 for other cells (the only difference is the
single frame where x == 75.0 exactly, which the authors' per-session `nanmax + buffer` binning places in the last
bin). This confirms (a) position/trace alignment, (b) the x-then-y axis order of the maps, and (c) the 5 cm grid.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `dat[animal]["trace"][day][cell, frame]` (binary events, 30 Hz) | `neural[session][trial]` (n_reg_cells, 600) | keep registered (non-NaN) cells -> `gaussian_filter1d(sigma=3 frames)` along time over the whole session -> `AvgPool(kernel=3, stride=3)` (100 ms bins) -> x30 to get events/s | `fit_decoder`/`test_decoder` (utils.py:1776/1806) use exactly this smoothing + pooling | float32, units = Hz |
| `dat[animal]["blocked"][day]` (indices of blocked partitions, -1 = none) | `input[session][trial]` (9,) static | binary vector b[p]=1 if partition p blocked | `get_env_mat` / `clean_rate_maps` define the same 3x3 layout | matches "Decoder Inputs: environment geometry ... static per trial" |
| `dat[animal]["position"][day][:, frame]` (x,y cm) | `output[session][trial]` (1, 600) | average x,y within each 100 ms bin -> `p = 3*clip(floor(y/25),0,2) + clip(floor(x/25),0,2)` -> snap to nearest open partition if p is blocked | `fit_decoder` pools position the same way; `decode_position_within` snaps positions to valid bins | 9 categories |
| `dat[animal]["envs"][day]` | `metadata["session_info"][session]["env"]` | geometry name string | `get_env_mat` | also stored per session |
| animal ID | `subjects`, `subject_idx` | 7 mice | `main.py` `animals` list | |
| - | `brain_regions`, `brain_region_idx` | all cells are CA1 | paper title/methods | single region |

### Key Decisions
1. **Session = animal-day (207 total)**: the paper's unit of analysis ("one session was recorded per day"); all 207 kept.
2. **Trial = 1 non-overlapping minute** (1800 frames @30 Hz), as required by the task. A trailing partial minute is kept only if >= 30 s long
   (all sessions are 39.9-40.1 min, so every session yields 40 trials: 39 full 60 s trials plus a final trial of
   55.5-60 s; 8,280 trials in total). Every session therefore has >= 2 trials.
3. **Time bin = 100 ms (3 frames)**: exactly the reference decoder's `temporal_bin_size=3` with `AvgPool1d`.
   600 bins per trial.
4. **Neural pre-processing = reference decoder's**: `gaussian_filter1d(trace, sigma=3 frames, axis=time)` then
   3-frame average pooling. Smoothing is done once on the **whole continuous session** before cutting into trials,
   so there are no filter edge artifacts at trial boundaries. The result is multiplied by 30 to express the rate in
   events/s (a constant scaling; makes the values interpretable and avoids the "all values are 0 or 1" error that
   raw binary data would trigger in `verify_data_format`).
5. **Neuron curation: keep every cell registered in that session** (non-NaN trace), i.e. 69,744 cell-sessions, the
   number quoted in the paper. The paper explicitly states its spatial-coding checks "motivated the inclusion of all
   cells in subsequent analyses". I deliberately do **not** apply the `cell_threshold=5` event criterion used inside
   `decode_position_within`, because (a) the paper includes all cells, (b) it would drop only 1.3% of cell-sessions,
   and (c) that criterion is defined over whole-session running frames, which is not meaningful for 1-min trials.
6. **No speed filter**: `decode_position_within` drops frames with speed <= 5 cm/s, but the paper's rate maps (which
   I reproduce exactly) use all frames, and the task requires contiguous time-varying outputs within fixed 1-min
   trials. Dropping ~40% of the frames would make trials non-contiguous and discard data, so all frames are kept.
   (Speed is *not* added as a decoder input because the task specifies the input is the environment geometry.)
7. **Output = single 9-class variable** "position_bin", time-varying at 100 ms, with the same partition indexing as
   the dataset's `blocked` field (`p = 3*row + col` on `np.flipud(get_env_mat(env))`, i.e. p = 3*floor(y/25)+floor(x/25)).
   This makes input (blocked partitions) and output (occupied partition) use one consistent 0-8 labelling.
8. **Blocked-partition tracking noise**: 0.003% of frames are tracked inside a partition that is walled off. Such
   time bins are re-assigned to the nearest **open** partition (Euclidean distance from the binned x,y position to
   the open partition centres), mirroring the reference decoder's step that snaps actual/predicted positions to
   valid bins. This avoids creating classes with 1-2 samples that would distort balanced accuracy.
9. **Input is static per trial** (shape (9,)) as specified by the task: geometry is constant within a session.
10. **dtype**: neural/input float32, output int64.

### Planned Sanity Checks
- [x] Total sessions == 207, subjects == 7, unique cells == 5,413, registered cell-sessions == 69,744 (paper).
- [x] Cells/animal min 515, max 952, mean 773 (paper).
- [x] Re-derive `maps["unsmoothed"]` from raw position+trace and compare with the stored maps (`np.allclose`).
- [ ] Compare converted neural rates against raw traces for specific (session, trial, neuron, timepoint) values.
- [ ] Compare converted output bins against raw position for specific (session, trial, timepoint) values.
- [ ] Check converted `input` equals the `blocked` field for every session; check that the decoded output bins are
      never a blocked partition.
- [ ] Check occupancy distribution over the 9 partitions is roughly uniform over open partitions and 0 for blocked.
- [ ] Check trials/session in {39, 40} and T == 600 for every trial.
- [ ] Decoder accuracy well above 1/9 chance, consistent with the paper's 11-17 cm mean Bayesian decoding error
      (25 cm partitions).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the plan of Step 5. Structure:
- `env_blocked_vector(env)` - 3x3 geometry -> length-9 binary "blocked" vector (dataset indexing).
- `bin_time_series(x, kernel=3)` - vectorised equivalent of `torch.nn.AvgPool1d(kernel_size=3, stride=3)`
  used by the reference `fit_decoder`/`test_decoder`.
- `position_to_partition(pos, blocked)` - 25 cm discretization + snapping of the rare bins that land inside a
  walled-off partition to the nearest open partition.
- `trial_slices(n_bins)` - non-overlapping 1-minute trials; a trailing partial trial is kept only if >= 30 s.
- `process_session(...)` - neuron curation (registered = non-NaN), gaussian smoothing (sigma=3 frames) of the
  **whole continuous session** followed by 3-frame average pooling and x30 scaling to events/s, behaviour binning,
  and trial cutting. Asserts that the geometry-derived blocked vector equals the dataset's `blocked` field.
- `plot_processing(...)` - 6-panel figure per session for `--show-processing` (raw raster, processed rates,
  raw-vs-processed alignment for one neuron, raw/binned position with partition and trial boundaries, resulting
  discrete output, occupancy vs blocked input).
- `main()` - assembles the target dictionary, runs whole-dataset assertions, prints summary statistics, pickles.

Code inefficiencies identified: the per-frame python loops of the reference `get_rate_maps` /
`get_transition_matrix`; these are not needed for the conversion and were replaced by vectorised reshape/mean.

Code speedups added: fully vectorised smoothing + pooling (scipy `gaussian_filter1d` on the (n_cells, n_frames)
matrix, then a reshape-mean), one joblib load per animal (the 15 GB `.mat` files are never touched), and
`float32` neural storage.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sample = QLAK-CA1-51, the first 2 sessions).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Trials (total) | 80 (40 per session) |
| T per trial | 600 bins (= 60 s at 100 ms) |
| Neurons (total cell-sessions) | 256 |
| Neurons / session | 113, 143 |
| Subjects | 1 |
| Input range (blocked_partition_0..8) | [0,0] except partition 4 which is [0,1] (session 2 = "o" geometry, centre blocked) |
| Output distribution (9 partitions) | [0.086, 0.070, 0.096, 0.096, 0.010, 0.096, 0.156, 0.120, 0.270] |
| Neural rate range | [0, 30] events/s, mean 0.281 |
| Bins snapped out of blocked partitions | 0 |

### Processing Plots Review
`processing_QLAK-CA1-51_day00_square.png` and `..._day01_o.png`:
- raw 30 Hz event raster and the 100 ms rate image show the same event structure at the same times;
- the single-neuron overlay shows every processed rate peak sitting on the raw event times (no temporal shift);
- binned position tracks raw position exactly, partition boundaries (25/50 cm) and 60 s trial boundaries line up
  with the steps in the discrete output;
- occupancy bars are zero exactly for the partition flagged as blocked in the input vector (centre for "o").
No anomalies.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| vectorised smoothing/pooling instead of per-frame loops | ~100x on the neural step |
| joblib (not .mat) loading | ~5x on I/O |

| Step | Time / Session | Estimated Total Time |
| load animal file | 6.5 s per animal (7 animals) | ~1.5 min |
| process session | 0.9 s (small animal) - ~2 s (large animal) | ~6 min |
| pickle write | - | ~1-2 min (about 7 GB) |
| **total** | | **< 10 min** (well under the 15 min budget) |

### Format verification (`/app/verification_sample_out.txt`)
"Data format is valid, no errors or warnings." All T = 600, input dim 9, output dim 1 with 9 categories.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| position_partition (9 classes) | 0.3910 | 0.3087 | 0.1111 |

Loss decreased monotonically (2.27 -> 1.72 over 200 epochs); validation accuracy is 2.8x chance.

**Context**: the sample is QLAK-CA1-51 sessions 0 and 1. In the authors' own precomputed Bayesian decoding
(`precomputed_results/within_decoding`) these two sessions have the *worst* decoding error of the whole dataset
(31.0 and 30.9 cm; typical late sessions are 8-11 cm), because few cells are registered early in the experiment
(113 and 143) and spatial reliability increases over days (paper Figures 1E-F). A modest accuracy on this sample
is therefore expected, and the full dataset should score considerably higher.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full` (187 s total: ~89 s joblib loading,
~93 s processing = 0.45 s/session, 5 s pickle write), matching the Step 7 estimate.

### Output Files
- `converted_data.pkl`: 6.73 GB
- `conversion_full_out.txt`, `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total unique neurons | 5,413 | 7 animals in the `animals` list | 515+875+942+554+862+713+952 = 5,413 | 5,413 | YES |
| Registered cell-sessions (= rate maps) | 69,744 | non-NaN cells per day | 69,744 | 69,744 | YES |
| Mean neurons/session | 69,744/207 = 336.9 | - | 336.9 (min 113, max 564) | 336.93 (min 113, max 564) | YES |
| Neurons/animal | mean 773 +/- 68 SE, min 515, max 952 | - | mean 773.3, min 515, max 952 | same | YES |
| Subjects | 7 | 7 | 7 | 7 | YES |
| Sessions | 207 | - | 31,31,31,21,31,31,31 | 207, same split per subject | YES |
| Trials (total) | n/a (continuous 40 min) | n/a | 71,866-72,219 frames/session | 8,280 (40/session) | consistent |
| Time bin | n/a | temporal_bin_size=3 frames @30 Hz | 30 Hz frames | 100 ms | YES |
| Session duration | 40 min | fps=30 | 39.9-40.1 min | 39.9-40.1 min (all frames kept) | YES |
| Input (blocked) range | 10 geometries, 0-4 blocked partitions | `get_env_mat` | `blocked` field | [0,1] per partition; partition 7 never blocked (correct - no geometry blocks bottom-centre) | YES |
| Output distribution (9 partitions) | not reported; centre avoided | - | occupancy in maps.sampling | [0.099,0.098,0.135,0.075,0.057,0.077,0.116,0.141,0.201] | consistent (thigmotaxis) |
| Neural rate | binary events treated as firing rate | rate = events x fps | mean event rate ~0.29 Hz | [0,30] Hz, mean 0.286 Hz | YES |

No data lost: all 7 animals, 207 sessions, 69,744 registered cell-sessions and 99.997% of recorded frames are
present (only the last 0-2 frames of a session that cannot fill a 100 ms bin are dropped; no quality exclusions).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
"Data format is valid, no errors or warnings." There are **no errors and no warnings** to address.
The summary confirms 207 sessions / 8,280 trials / 7 subjects / 1 brain region (CA1, 69,744 neurons),
input dim 9, output dim 1 with all 9 categories used, T = 600 for the 39 full trials of each session and
555-600 for the final trial (session lengths are 71,866-72,219 frames).

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`, `/app/cache/crosscheck_maps.py`)
These scripts load the **raw joblib files** and the converted pickle and compare them; no conversion code is
imported. 81 checks over 9 randomly-chosen sessions from 3 animals, **0 failures**:
1. **Neural**: `np.allclose(converted_neural[s][t], pooled(gaussian_filter1d(raw_trace[registered], sigma=3,
   axis=1))*30)` for 3 random trials in each session, atol 1e-5. A hand-computed single element
   (trial 5, neuron 3, time bin 10) also matches.
2. **Input**: `converted_input[s][t]` equals the binary vector built from the raw `blocked` field for every trial
   of every checked session, and equals the vector derived independently from `get_env_mat(envs[day])`.
3. **Output**: `converted_output[s][t]` equals `3*floor(y_binned/25)+floor(x_binned/25)` recomputed from the raw
   position for all time bins except the 0.003% deliberately snapped out of blocked partitions; and no output
   label is ever a partition flagged as blocked in the input.
4. **End-to-end cross-check** (`crosscheck_maps.py`): the per-partition mean firing rate recomputed from the
   **converted** neural and output streams matches the dwell-time-weighted aggregation of the authors **stored**
   15x15 rate maps (`maps["unsmoothed"]`x30): r = 0.9995-0.9999, e.g. QLAK-CA1-30 day 0 stored 0.1942 Hz vs
   converted 0.1941 Hz. This jointly tests neural/behaviour temporal alignment, partition indexing and map axes.
5. **Rate-map reproduction** (`/app/cache/check_maps*.py`, Step 4): `maps["unsmoothed"]` is reproduced from raw
   `position` and `trace` with a fixed 5 cm grid, differing for a single frame in 72,219 (x == 75.0 exactly).

### Check 3: Reference code comparison
| Stage | Reference code | My code | Same? |
|-------|----------------|---------|-------|
| (a) Loading | `load_dat(animal, p, format="joblib")` -> `joblib.load(data/<animal>)[animal]` | identical | YES |
| (b) Neuron filtering | unregistered (NaN) cells excluded everywhere; `decode_position_within` also requires >5 events on running frames | keep all registered (non-NaN) cells; the >5-event rule not applied | Intentional: the paper includes all cells; the rule would drop 1.3% of cell-sessions and is defined on whole-session running frames, which does not transfer to 1-min trials |
| (c) Temporal alignment | `position[d][:, t]` and `trace[d][:, t]` share the DAQ frame index | same index, no shifts | YES |
| (d) Binning | `fit_decoder`: `gaussian_filter1d(sigma=3)` along time then `AvgPool1d(3,3)`; position pooled identically then floor-divided | identical (vectorised reshape-mean == AvgPool1d), plus a x30 scaling to events/s | YES (constant scale only) |
| (e) Input construction | `get_env_mat(env)` and the `blocked` field, 3x3 layout [[0,1,2],[3,4,5],[6,7,8]] | binary 9-vector, asserted equal to `blocked` for every session | YES |
| (f) Output construction | position binned to 15x15 and snapped to visited bins; partitions = 5x5-pixel blocks (`clean_rate_maps`) | position binned by 25 cm (exactly 3x3 blocks of that grid), snapped to nearest open partition | YES at the 3x3 resolution the task requires |
| Spatial grid origin | `get_rate_maps` uses the per-session `nanmax` | fixed 0-75 cm grid | Intentional: the stored rate maps are reproduced exactly by the fixed grid, the arena is physically 75 cm, and a fixed grid is stable across sessions. Per-session maxima differ by <3% so <1% of labels are affected |
| Speed filter | `decode_position_within` drops frames <= 5 cm/s | not applied | Intentional: the paper rate maps use all frames, trials must be contiguous 1-min blocks (Step 5, decision 6) |

### Check 4: Key statistics comparison
See the Step 9 table: every statistic available in the paper (5,413 neurons, 207 sessions, 69,744 rate maps,
cells/animal mean 773 / min 515 / max 952, 40-min sessions, 30 Hz, 75 cm arena, 3x3 partitions, 10 geometries)
is reproduced exactly. Additional checks: the 10 geometries and their blocked partitions match `get_env_mat` for
all 207 sessions (asserted in the conversion), and each animal sequence starts and ends with `square` and repeats
the same 10-geometry sequence (3 repeats; 2 for QLAK-CA1-51), exactly as the paper describes.

### Check 5: Edge cases
- **Unregistered cells** are asserted to be NaN for the *entire* session before being dropped (never partial).
- **Position exactly at the arena edge** (x or y == 75.0 cm, present in every animal): `np.clip` keeps it in
  row/column 2 rather than creating a 4th bin.
- **Trailing partial trial**: sessions are 39 full 60 s trials plus 55.5-60 s; the remainder is kept as a final
  trial (>= 30 s). Frames that cannot fill a 100 ms bin (0-2 per session) are dropped.
- **Tracking noise inside blocked partitions** (152 of 4,967,611 bins = 0.003%) is snapped to the nearest open
  partition, so no class is created from a handful of spurious samples.
- **Sessions with < 2 trials**: none (all 40); asserted in the conversion script.
- **Smoothing at trial edges**: smoothing is applied to the continuous session *before* cutting trials, so no
  trial boundary artefacts are introduced.

### Issues Found and Resolved
- *Documentation mismatch*: the Step 5 plan initially said the trailing partial minute is dropped, while the
  implementation keeps it when >= 30 s. The note was corrected to match the code (keeping it preserves data and
  the format allows variable T).
- No other issues found; no re-conversion required.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, ~4 min).

### Training Progress
- Loss decreasing: **Yes**, monotonically - 2.05 (epoch 1) -> 1.42 (20) -> 1.03 (50) -> 0.87 (130) -> 0.822 (200).
  Test loss 1.333.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|--------|-------|
| position_partition (9 classes) | 0.7487 | 0.6051 | 0.1111 | 5.4x chance |

Sample plots `sample_trials.png` and `predictions.png` show the decoded partition sequence closely tracking the
true sequence within held-out trials.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Variable | Classes | Chance | Validation balanced acc | Ratio |
|----------|---------|--------|-------------------------|-------|
| position_partition | 9 | 0.111 | 0.605 | **5.4x** |

Far above the 1.5x-chance flag, so there is no indication of a conversion bug.

### Check 2: Accuracy comparison to the paper
The paper reports position decoding as a **Euclidean error in cm** on the 15x15 (5 cm) grid, not as an accuracy,
so I made two comparisons:

| Source | Method | Metric | Value |
|--------|--------|--------|-------|
| Paper / `precomputed_results/within_decoding` | Gaussian naive Bayes, flat priors, 5-fold CV within session, 15x15 bins | mean Euclidean error | 10.9-17.6 cm per animal (early sessions 25-31 cm, late 8-11 cm) |
| **This conversion + the paper own method** (`/app/cache/ref_bayes_check.py`: GaussianNB, flat priors, 5-fold CV within session, applied to my converted neural data and 3x3 labels) | GNB, 9 classes | balanced accuracy | **0.491** over 15 random sessions (0.552 for day >= 15) |
| **This conversion + the provided decoder** | shared low-rank projection + logistic decoder, 80/20 trial split | balanced accuracy | **0.605** |

The provided decoder therefore **exceeds** the paper own method applied to the same converted data
(0.605 vs 0.491), and both are far above chance (0.111). This is the expected scale: a mean error of 11-17 cm in
a 75 cm arena with 25 cm partitions means most time bins fall in the correct or an immediately adjacent
partition. There is no decoding accuracy reported in the paper that this conversion falls short of.

### Check 3: Train vs validation gap
Training 0.749 vs validation 0.605 -> ratio **1.24**, below the 1.5x flag. The residual gap is the expected
per-session overfitting of a model that learns one projection per session (207 sessions, up to 564 neurons).
The split is by trial within session, so there is no leakage of time points between train and validation
(trials are disjoint 1-minute blocks).

### Additional debugging performed (all negative - no problems found)
1. **Output values verified against the raw data** for 9 random sessions x 3 trials (Step 10, Check 2).
2. **Temporal alignment plotted** for single trials (`processing_*.png`, `sample_trials.png`, `predictions.png`):
   the processed rates sit exactly on the raw event times and the discrete output steps coincide with the 25 cm
   crossings of the raw position trace.
3. **Output variation**: the most common class holds 20% of time bins and all 9 classes occur; no degenerate
   distribution and balanced accuracy is a meaningful metric.
4. **Neural filtering** follows the reference (registered cells only; smoothed, 100 ms-pooled binarized
   rising-phase events).
5. **Per-session pattern matches the paper**: sessions where the authors own decoder performs worst
   (QLAK-CA1-51 days 0-1, 31 cm error) also give the lowest accuracy here (sample run: 0.309), and late,
   cell-rich sessions give the highest (QLAK-CA1-50 day 21, 519 cells: GNB balanced accuracy 0.694). The
   accuracy therefore varies with data quality exactly as the paper reports (Figures 1E-F).

### Issues Found and Resolved
- None in this round; no re-conversion was required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, key statistics, how to load, output format specification,
      processing summary)
- [x] `cache/` folder created with all exploration / validation scripts and `cache/README_CACHE.md`
      documenting each of them
- [x] All files organised

### Deliverables
| File | Description |
|------|-------------|
| `CONVERSION_NOTES.md` | This document - decisions, checks, validation results |
| `README.md` | User-facing documentation |
| `convert_data.py` | Conversion script (`--full`, `--sample`, `--show-processing`) |
| `converted_data.pkl` | Full converted dataset (6.73 GB, 207 sessions) |
| `sample_data.pkl` | 2-session sample |
| `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt` | Sample logs |
| `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt` | Full-dataset logs |
| `processing_QLAK-CA1-51_day00_square.png`, `processing_QLAK-CA1-51_day01_o.png` | Per-step processing plots |
| `sample_trials.png`, `predictions.png` | Decoder sample trials and predictions |
| `cache/` | Exploration and independent validation scripts (+ `README_CACHE.md`) |

### Final summary
207 sessions from 7 mice, 8,280 one-minute trials, 69,744 registered CA1 cell-sessions (exactly the paper
figure), 100 ms time bins. Decoding the occupied 3x3 arena partition from CA1 activity plus the static geometry
input reaches a validation balanced accuracy of **0.605** (chance 0.111), exceeding the paper own Gaussian naive
Bayes method applied to the same converted data (0.491).
