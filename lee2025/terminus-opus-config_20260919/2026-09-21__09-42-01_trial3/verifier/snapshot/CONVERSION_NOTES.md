# Dataset Conversion Notes

## Overview
- **Dataset**: Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping (paper.pdf, /app/data, /app/code)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124 (CUDA available), scipy 1.18.0, pandas 3.0.5.

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `CONVERSION_NOTES.md` (this file)
- `code/` -> reference repo `georepca1` (README.md, environment.yml, georepca1/{main.py, src/utils.py, src/plots.py, demos/*.ipynb})
- `data/` -> 7 per-animal joblib files (QLAK-CA1-08, -30, -50, -51, -56, -74, -75), the same 7 as `.mat`,
  `behav_dict`, `precomputed_results/`  (15 GB total)
- `decoder.py`, `train_decoder.py` (provided decoder harness)
- `methods.txt`, `paper.pdf`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference repo: `georepca1` (Lee, Keinath, Cianfarano & Brandon 2025, Neuron 113(2):307-320).
Files: `code/README.md`, `code/environment.yml`, `code/georepca1/main.py`,
`code/georepca1/src/utils.py` (3372 lines), `code/georepca1/src/plots.py`, demos/*.ipynb.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat(animal, p, format)` | src/utils.py:61 | LOADING | Loads per-animal joblib (or .mat via mat73) -> dict {animal: {SFPs, blocked, centroids, envs, maps, position, trace}} |
| `save_dat` / `mat2joblib` | src/utils.py:87,100 | LOADING | Convert .mat -> joblib |
| `generate_behav_dict(animals,p)` | src/utils.py:130 | LOADING | Makes lightweight dict with position, envs, maps_shape for all animals (saved as `data/behav_dict`) |
| `get_environment_label(env_name)` | src/utils.py:150 | PROCESSING | Polygon vertices for each of 10 geometries (plotting) |
| `get_env_mat(env)` | src/utils.py:215 | PROCESSING | **Binary 3x3 matrix of environment geometry** (1 = open partition, 0 = blocked). 10 names: square, o, t, u, rectangle, +, i, l, bit donut, glenn |
| `get_rate_maps(position, trace, n_bins=15, fps=30, buffer=1e-5, filter_size=1.5)` | src/utils.py:313 | PROCESSING | Spatial binning: `position // ((nanmax(position,axis=0)+buffer)/n_bins)`; rate = events/occupancy*fps; 2D gaussian smoothing sigma=1.5 bins (=2.5cm, 5cm/bin) |
| `get_split_half`, `get_shuffle_split_half`, `get_place_cells`, `get_shr_within` | 356/393/415/442 | CURATION | Place-cell identification via split-half rate-map correlation vs 1000 circular shuffles |
| `clean_rate_maps(maps, envs)` | src/utils.py:463 | PROCESSING | NaNs out pixels outside geometry. **Mapping from 3x3 env matrix to 15x15 map: `np.fliplr(get_env_mat(env).T)` then each of 3x3 cells expanded to 5x5** |
| `fit_decoder(behav, traces, feature_max, temporal_bin_size=3)` | src/utils.py:1776 | PROCESSING/DECODE | Temporal binning of behav+traces with `AvgPool1d(kernel=3,stride=3)` (30Hz -> 10Hz, 100 ms bins); traces first smoothed with `gaussian_filter1d(sigma=3, axis=0)`; position one-hot over 15x15 bins; GaussianNB with flat priors |
| `test_decoder(...)` | src/utils.py:1806 | DECODE | Predict, reverse one-hot, Euclidean error |
| `decode_position_within(behav, traces, maps, n_bins=15, fps=30, v_filt_size=5, v_thresh=5, cell_threshold=5, buffer=1e-15, n_fold=5)` | src/utils.py:1845 | CURATION+DECODE | **The reference decoding pipeline.** (1) spatial bin_down = (max position over all days + buffer)/15; (2) velocity computed as `gaussian_filter1d(norm(diff(behav)*fps), sigma=5)` and thresholded at 5 cm/s -> `vel_idx`; (3) cells kept if `sum(trace[vel_idx]) > 5` events -> `cell_idx`; (4) 5-fold CV GaussianNB |

### Notes
- Data fields per animal (from code/README.md):
  - **SFPs** (dimx, dimy, n_cells, n_days): spatial footprints, NaN if not registered that day
  - **blocked**: indices (0-8) of blocked partitions in the 3x3 design, ordered [[0,1,2],[3,4,5],[6,7,8]]; -1 if none blocked
  - **centroids** (n_cells, xy, n_days)
  - **envs**: string name of geometry per day
  - **maps**: {sampling (occupancy: xbins,ybins,days), smoothed (xbins,ybins,cells,days), unsmoothed}
  - **position**: per day, (xy, n_frames)
  - **trace**: rise-extracted calcium, binary (1 = significant event), NaN for the whole day if the cell is not registered that day
- Neural signal to use = the **binarized rising-phase vector** (`trace`), treated as firing rate (per Methods).
  No dF/F computation needed - already done by authors.
- Neuron curation available: (a) NaN = unregistered on a day (must drop or handle), (b) reference decoder
  drops cells with <=5 events during movement periods, (c) place cells (split-half reliability) - used for
  figures but NOT for the decoder.
- Sampling: 30 Hz for both behavior and calcium; sessions are 40 min.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/` contains, for each of 7 mice (`QLAK-CA1-08, -30, -50, -51, -56, -74, -75`):
- `<animal>`  : joblib file, loads to `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}`
- `<animal>.mat` : identical content as MATLAB v7.3/HDF5 (`h5py`-readable, arrays transposed)
Plus `behav_dict` (position+envs+maps_shape for all animals) and `precomputed_results/`
(`<animal>_shr` split-half p-values, `within_decoding` = reference Bayesian decoding errors,
`<animal>_rsm_partitioned`, `map_corr_envs`, `pv_corr_pixelwise`, `riab/`).

Per-animal fields (verified on QLAK-CA1-51):
| Field | Shape | Notes |
|-------|-------|-------|
| `trace` | (n_days, n_cells, T) float64 | **binary 0/1** rising-phase events @30 Hz. If a cell is not registered on a day, the whole (cell, day) row is NaN (verified: NaN count per cell-day is either 0 or T) |
| `position` | (n_days, 2, T) float64 | x,y in **cm**, range [0, 75], no NaNs anywhere (checked all animals) |
| `envs` | (n_days, 1) <U9 | geometry name per day |
| `blocked` | list of n_days, each `[array]` | indices (0-8) of blocked partitions, `-1` if none |
| `maps.sampling` | (15,15,n_days) | occupancy in **seconds** |
| `maps.smoothed` / `maps.unsmoothed` | (15,15,n_cells,n_days) | event rate maps in **events/frame**; NaN outside visited bins and for unregistered cells |
| `SFPs` | (35,35,n_cells,n_days) | spatial footprints |
| `centroids` | (n_cells,2,n_days) | |

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 7 (QLAK-CA1-08/30/50/51/56/74/75) |
| Sessions (days) | 31 for all animals except QLAK-CA1-51 (21) -> **207 total** |
| Sessions / subject | 31,31,31,21,31,31,31 |
| Neurons (total, unique/registered across days) | 515+875+942+554+862+713+952 = **5,413** |
| Neurons / session (registered, non-NaN) | mean 337 (min 113, max 564) |
| Registered cell-sessions (= number of rate maps) | **69,744** |
| Frames / session | 71,866 - 72,219 @30 Hz = 2395-2407 s (~40 min) |
| Trials | none natively; free exploration. Will be cut into 1-min trials (Decoder Task) |
| Geometries | 10 (square, o, t, u, rectangle, +, i, l, bit donut, glenn), 3 repeats of the sequence per animal (2 for -51), always starting and ending with square |

### Verified conventions (sanity checks run in Step 2)
1. `maps['sampling'][:,:,d]` == my occupancy (frames/30) computed with a **global** bin size
   `(75 + buffer)/15 = 5 cm` and index `[xbin, ybin]` (max abs diff = 0.033 s = one frame).
   Per-day-max binning does NOT reproduce it -> **bins are absolute 5 cm bins spanning 0-75 cm**.
2. `maps['unsmoothed'][:,:,c,d]` == sum(trace in bin)/occupancy(frames) for every registered cell
   (max abs diff <0.004 = 1 frame; NaN patterns identical). This validates
   **exact frame-by-frame alignment of `trace` and `position`** (no lag) and the bin indexing.
3. `blocked` partition index p = **3*ybin + xbin** with 25-cm partitions: occupancy inside
   blocked partitions is 0.000-0.006 for all 21 sessions of QLAK-CA1-51, whereas the alternative
   (3*xbin + ybin) gives up to 0.71. Consistent with `clean_rate_maps`, which builds the map
   mask as `np.fliplr(get_env_mat(env).T)` i.e. mask[x,y] = env_mat[2-y, x].
4. Total registered cell-sessions = **69,744** = number of rate maps quoted in the paper.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 | "5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps" |
| Rate maps (cell-sessions) | 69,744 | same |
| Sessions | 207 | same |
| Subjects | 7 (4 male, 3 female) | "Naive male (4) and female (3) mice (C57Bl/6, Charles River)" |
| Mean cells / animal | 773 | "(mean number of cells per an-imal = 773...)" (5413/7 = 773.3) |
| Sessions / subject | up to 31 (3 sequences of 10 geometries + final square) | "The same sequence was repeated up to three times within animals" |
| Session length | 40 min, 1 session/day | "All sessions were 40 min, and one session was recorded per day" |
| Acquisition rate | 30 Hz (behaviour + calcium simultaneously, timestamped) | "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz" |
| Arena | 75 x 75 cm square partitioned into 3 x 3 grid | "We partitioned an open square (75 x 75 cm) into a 3 x 3 grid space" |
| Spatial bin | 5 cm^2 (15 x 15 map) | "5 cm 2 spatial bins @ 2 Hz" (transition matrices); rate maps are 15x15 over 75 cm |
| Neural signal | binarized rising phase of calcium transients, treated as firing rate | "This binary vector was treated as the firing rate in all further analyses" |
| Position | DeepLabCut head tracking | "Position data were generated from tracking the head with DeepLabCut" |
| Decoding | Gaussian Naive Bayes, 5-fold, flat prior, error in cm | "we performed a 5-fold split of spatially binned position and trace data and transformed the binned positions to a one-hot vector" |
| Decoding error | ~8-31 cm per session (mean over folds), decreasing across days (ANOVA p<0.0001, F=7.9845) | Figure 1F; `precomputed_results/within_decoding` |

### Processing Details
- Neural: no dF/F needed; the distributed `trace` is already the binarized rising-phase vector.
- Temporal alignment: behaviour and calcium are acquired by the same DAQ at 30 Hz and were already
  frame-aligned by the authors (confirmed by sanity check 2 in Step 2).
- Temporal binning (reference decoder): `AvgPool1d(kernel=3, stride=3)` on 30 Hz data = **100 ms bins**,
  traces first smoothed with `gaussian_filter1d(sigma=3 frames = 100 ms)`.
- Spatial binning: absolute bins of (75 cm)/n_bins.

### Curation Steps
**Neuron curation rules** (from `decode_position_within`):
- cells with `sum(events during movement) > 5` in that session are kept (`cell_threshold=5`).
  Unregistered cells (all-NaN) automatically fail this test.
- Place-cell selection (split-half reliability) is used for figures but explicitly NOT for decoding:
  the paper states the results "motivated the inclusion of all cells in subsequent analyses".

**Trial curation rules**:
- No trials in the original experiment (40-min free exploration). The reference decoder excludes
  periods when the animal moves slower than **5 cm/s** (`v_thresh=5`, speed from
  `gaussian_filter1d(|diff(position)|*fps, sigma=5 frames)`).

### Decoders Trained (reference)
| Decoded variable | Accuracy |
| position (15x15 bins, GNB, within session) | mean Euclidean error 8.2-31.0 cm per session; group mean ~13 cm; chance for uniform 75x75 arena ~39 cm |
(No classification accuracies are reported in the paper; only Euclidean decoding error.)

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Spatial bin size | `get_rate_maps` divides by `(nanmax(position)+buffer)/n_bins` (per session); `decode_position_within` divides by the max over **all** days | Position is normalised to exactly [0,75] cm in x and y for **every** animal and day, so both give the identical 5 cm bin | "5 cm^2 spatial bins", 75x75 cm arena | No discrepancy in practice. I use the absolute bin size 75/n_bins. Verified: reproduces `maps['sampling']` and `maps['unsmoothed']` exactly |
| Rate map units | `get_rate_maps` multiplies by fps (Hz) | stored `maps['unsmoothed']` = events/frame (my recomputation x 1/30) | - | The stored maps were made without the fps factor. Irrelevant for the conversion (I use the raw `trace`), but it confirmed my binning |
| Env matrix orientation | `get_env_mat(env)` is the matrix as drawn in the figures; `clean_rate_maps` applies `np.fliplr(mat.T)` before masking a 15x15 map, i.e. mask[x,y] = mat[2-y,x] | `blocked` indices == flat indices of the zeros of `flipud(get_env_mat(env))` for **all 207 sessions** (0 mismatches) | partitions "organized in the following way - [[0,1,2],[3,4,5],[6,7,8]]" | Partition index = **3*ybin + xbin** in position coordinates. Verified independently: occupancy inside `blocked` partitions is ~0 (<=0.6%), whereas the transposed hypothesis gives up to 71% |
| Number of sessions | animals list of 7 | 31,31,31,21,31,31,31 = 207 | 207 sessions | consistent |
| Number of neurons | - | 5,413 unique; 69,744 registered cell-sessions | 5,413 neurons / 69,744 rate maps | consistent |
| Speed filtering | `decode_position_within` uses v_thresh = 5 cm/s on `gaussian_filter1d(speed, sigma=5 frames)` | - | Not mentioned in the paper text | Follow the reference code (it is the code that produced Figure 1F) |
| Cell filtering for decoding | `decode_position_within`: keep cells with >5 events during movement | Unregistered cells are all-NaN and are removed by the same test | "motivated the inclusion of all cells in subsequent analyses" (i.e. no place-cell selection) | Use the reference decoder's rule: drop unregistered cells and cells with <=5 events while moving. No place-cell selection |
| Temporal binning | `fit_decoder`/`test_decoder`: gaussian_filter1d(sigma=3 frames) then AvgPool1d(3) -> 100 ms | 30 Hz frames | "acquired ... at 30 Hz" | Use 100 ms bins |

Understanding of the three sources is consistent.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day]` (n_cells, T) binary events @30 Hz | `neural[session][trial]` (n_neurons, n_bins) float32 | drop unregistered (all-NaN) cells; drop cells with <=5 events during movement; `gaussian_filter1d(sigma=3 frames)` along time; `AvgPool1d(kernel=3,stride=3)`; x30 -> events/s | `fit_decoder`, `decode_position_within` | 100 ms bins; x30 only rescales to Hz (no effect on a linear decoder, keeps values O(0.1-3) so the L1 penalty on the projection is not dominant) |
| `blocked[day]` (indices 0-8, -1 = none) | `input[session][trial]` (9,) float32 | binary vector, 1 = partition blocked | `get_env_mat` / README | static per trial, as required by the Decoder Task |
| `position[day]` (2, T) in cm | `output[session][trial]` (1, n_bins) int | average x,y within each 100 ms bin, then bin = `floor(pos/25)` clipped to [0,2]; class = 3*ybin + xbin | `fit_decoder` (pools behaviour the same way), `decode_position_within`, `blocked` convention | 3x3 = 9 classes as required by the Decoder Task |
| `envs[day]` | `metadata['session_info']` | geometry name per session | | also stored per-trial geometry name |
| animal ID | `subjects`, `subject_idx` | | | 7 mice |
| - | `brain_regions` = ['CA1'], `brain_region_idx` | all neurons are CA1 | paper | |

### Key Decisions
1. **Neural signal = the distributed binarised rising-phase `trace`**: the paper states this binary
   vector "was treated as the firing rate in all further analyses". No dF/F computation is needed.
2. **Temporal bin = 100 ms** with a gaussian smoothing of sigma = 3 frames before pooling: exactly the
   reference decoder's `fit_decoder`/`test_decoder` preprocessing. It also removes the
   "neural data is all 0/1" error raised by the verification harness.
3. **Trials = consecutive 60 s blocks** of each session (Decoder Task). The final partial block is kept
   only if it is at least 30 s long (all sessions have a ~55 s remainder, so it is kept); this avoids
   discarding ~2.5% of the data. ~40 trials/session, well above the minimum of 2.
4. **Speed filter, >5 cm/s**: timepoints where the smoothed speed does not exceed 5 cm/s are dropped,
   exactly as `decode_position_within` does for its Bayesian position decoder (`v_thresh=5`,
   `v_filt_size=5` frames). Position coding in CA1 is only well defined during locomotion, and
   immobility periods are dominated by replay/SWR activity that is not about current position.
   Trials keep ~60% of their bins; trial lengths therefore differ, which the format allows.
5. **Neuron curation** = the reference decoder's rule: keep cells registered on that day
   (non-NaN) **and** with >5 events during the movement periods (`cell_threshold=5`).
   No place-cell selection - the paper explicitly includes all cells.
6. **Output = one categorical variable with 9 values** (3x3 partitions), time-varying, using the
   same partition indexing as the dataset's `blocked` field (index = 3*ybin + xbin). Positions that
   fall inside a blocked partition (<=0.6% of frames, tracking noise near partition walls) are kept
   as-is rather than being re-assigned, so the output is a faithful discretisation of the tracked position.
7. **Input = 9-dim binary "blocked" vector**, static per trial, exactly as specified by the Decoder Task.
8. **Sessions kept**: all 207. No session/animal is excluded (all pass the quality checks of the paper).

### Planned Sanity Checks
- [x] Recomputed occupancy == `maps['sampling']` (validates spatial binning)
- [x] Recomputed unsmoothed rate maps == `maps['unsmoothed']` (validates trace/position alignment)
- [x] `blocked` == zeros of `flipud(get_env_mat(env))` for all 207 sessions
- [x] Total registered cell-sessions == 69,744 (paper)
- [ ] Converted totals: 7 subjects, 207 sessions, cells/session in [113, 564]
- [ ] Output class distribution matches the occupancy of the 3x3 partitions computed directly from raw position
- [ ] Input vector equals the raw `blocked` field for spot-checked sessions
- [ ] Neural values in a spot-checked trial equal the smoothed+pooled raw trace (np.allclose)
- [ ] No NaN/Inf; neural not all 0/1; time dimensions equal across neural/input/output

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the conversion. Structure:
- `env_blocked_vector(env)` - 9-dim binary blocked-partition vector from the geometry name
  (re-derived from `utils.get_env_mat` + `utils.clean_rate_maps`).
- `open_animal` / `read_meta` / `read_session` - **lazy** per-session reads from the MATLAB v7.3
  files with `h5py`. The joblib files load a whole animal (up to 17 GB expanded) at once; the
  HDF5 route reads one session (~0.4 s) and keeps peak memory at ~1 GB. Verified byte-identical
  to the joblib contents.
- `compute_speed` - copy of the speed computation of `utils.decode_position_within`.
- `pool_mean` - vectorised equivalent of `torch.nn.AvgPool1d(kernel_size=3, stride=3)` used by
  `utils.fit_decoder`.
- `position_to_class` - 3x3 discretisation, class = 3*ybin + xbin.
- `process_session` - neuron curation, smoothing, binning, speed filtering, 60 s trial cutting.
- `plot_processing` - the `--show-processing` figure (6 panels, see Step 7).
- `main` - `--full` (default) / `--sample` / `--show-processing`, prints per-session timing.

Code inefficiencies identified:
- Loading whole-animal joblib files (6-40 s each, >10 GB RAM). -> replaced by per-session HDF5 reads.
- `AvgPool1d` through torch tensors. -> replaced by a numpy reshape-mean (identical result).
- Python loops over frames for binning (as in the reference `get_rate_maps`). -> not needed here;
  all operations are vectorised.

Code speedups added: lazy HDF5 session reads, numpy reshape pooling, in-place float32 casting.
Total runtime for all 207 sessions is ~3 min, so no parallelism was necessary.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Sample = 2 sessions from 2 animals chosen to give input variation:
`QLAK-CA1-51 day 0` (square, nothing blocked) and `QLAK-CA1-56 day 9` (t, partitions 3,5,6,8 blocked).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (of 7 listed) |
| Neurons (total) | 445 (112 + 333) |
| Neurons / session | 112, 333 (registered 113, 336) |
| Trials (total) | 80 |
| Trials / session | 40, 40 |
| Timepoints / trial | mean 344, min 134, max 468 (of 600 possible; the rest removed by the speed filter) |
| Input range (blocked_partition_0..8) | [0,0],[0,0],[0,0],[0,1],[0,0],[0,1],[0,1],[0,0],[0,1] |
| Output distribution (pooled) | [0.141, 0.110, 0.122, 0.037, 0.125, 0.047, 0.075, 0.208, 0.134] |
| Output distribution, session 2 (t) | [0.185,0.162,0.132,**0.000**,0.229,**0.000**,**0.000**,0.293,**0.000**] |

The zero-occupancy classes of the `t` session are exactly the partitions flagged as blocked in the
input vector (3, 5, 6, 8) - a strong cross-check that the input and output share one indexing.

### Processing Plots Review
`processing_QLAK-CA1-51_day00.png`, `processing_QLAK-CA1-56_day09.png`, 6 panels each:
1. raw binary events (30 Hz), 2. smoothed + 100 ms pooled rates, 3. speed trace with the 5 cm/s
threshold and the kept bins, 4. raw vs binned position with the 25/50 cm partition borders
(the two traces overlie exactly - no temporal shift), 5. output class vs the class recomputed
from the binned position (identical step functions), 6. trajectory with the partition grid and
ids, the output class histogram and the input vector as a 3x3 image. No anomalies.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| lazy per-session HDF5 reads instead of whole-animal joblib | ~6-40 s -> 0.3-0.6 s per animal-load, and >10 GB less RAM |
| numpy reshape-mean instead of torch AvgPool1d | minor, avoids tensor round trip |

| Step | Time / Session | Estimated Total Time |
| read session from .mat | 0.43 s | 1.5 min for 207 |
| process session | 0.46 s | 1.6 min for 207 |
| **total** | **0.89 s** | **~3.1 min** (sample sessions have fewer cells than average - allowing for the mean of 337 cells/session the realistic estimate is 4-6 min) |

### Independent Sanity Checks (`/app/cache/sanity_check.py`)
Re-derives everything from the raw `.mat` files without importing the conversion code:
- neural of every trial `np.allclose` to the recomputation: **True** for both sessions
- scalar spot checks, e.g. `neural[trial 5, neuron 0, t 210] = 29.995012 vs 29.995014`: **allclose**
- `input` equals the raw `blocked` field of the file and is constant across trials: **True**
- output class fractions vs those computed directly from the raw 30 Hz moving frames:
  max difference 0.0016
- fraction of output samples inside blocked partitions: 0.00000
- FAILURES: 0

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
Loss decreased monotonically from 3.1608 (epoch 1) to 1.1589 (epoch 200); test loss 1.7832.

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| position_3x3 | 0.5477 | 0.4203 | 0.1111 |

3.8x chance on only two sessions, both of which are early recordings (day 0 and day 9) that the
paper shows have the *worst* spatial coding (Figure 1F: decoding error falls from ~31 cm on the
first sessions to ~9 cm at the end). Accuracy on the full dataset is expected to be higher.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full` -> 245 s (4.1 min),
close to the 3-4 min estimated in Step 7.

### Output Files
- `converted_data.pkl`: 3.44 GB
- `conversion_full_out.txt`: created (one line per session)
- `verification_full_out.txt`: created - **"Data format is valid, no errors or warnings."**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 7 (4 M, 3 F) | 7 animal IDs in `main.py` | 7 files | 7 | YES |
| Sessions | 207 | - | 31+31+31+21+31+31+31 = 207 | 207 (31,31,31,21,31,31,31) | YES |
| Unique neurons | 5,413 | - | 515+875+942+554+862+713+952 = 5,413 | 5,413 (per-animal cell counts in the files) | YES |
| Registered cell-sessions (rate maps) | 69,744 | - | 69,744 | 68,862 kept after the reference cell filter (>5 events while moving) = 98.7% | expected difference, see below |
| Mean cells / animal | 773 | - | 773.3 | 773.3 | YES |
| Mean cells / session | - | - | 336.9 registered | 332.7 | YES (98.7%) |
| Geometries | 10 | 10 in `get_env_mat` | square 27, others 20 each | same | YES |
| Session length | 40 min | - | 71,866-72,219 frames @30 Hz = 39.9-40.1 min | 40 x 60 s trials/session | YES |
| Trials | n/a (continuous) | n/a | n/a | 8,147 (mean 39.4/session) | n/a |
| Time bin | - | 100 ms (AvgPool1d(3) @30 Hz) | 33.3 ms frames | 100 ms | YES |
| Input range (blocked_partition_0..8) | 0/1 | 0/1 | 0/1 | [0,1] for all except partition 7, which is never blocked in any of the 10 geometries ([0,0]) | YES |
| Output distribution (9 partitions) | - | - | occupancy of the 3x3 partitions | [0.095, 0.108, 0.114, 0.088, 0.070, 0.087, 0.109, 0.158, 0.171] | YES (matches raw occupancy to <0.002 per session) |
| Decoding error (reference) | 8.2-31.0 cm/session | `within_decoding` | same | n/a (this decoder is a 9-way classifier) | - |

**Why 68,862 < 69,744**: 882 cell-sessions (1.3%) are registered on a day but produce <= 5
calcium events while the animal is running, and are removed by the reference decoder's
`cell_threshold = 5` rule in `utils.decode_position_within`. Keeping them would add rows that are
exactly zero for the whole session.

**Trials per session**: 40 for most sessions (a 71,866-frame session gives 39 full 60 s trials plus
a 55-s remainder that is kept). A few sessions have fewer because a trial in which the mouse ran for
less than 3 s is dropped (the minimum of 22 trials is QLAK-CA1-74 day 21). Every session has >= 22
trials, far above the required 2.

### Data integrity spot checks (`/app/cache/sanity_check.py` on the full pickle)
4 randomly chosen sessions (QLAK-CA1-30 day 24 glenn, QLAK-CA1-51 day 12 l,
QLAK-CA1-74 day 28 glenn, QLAK-CA1-56 day 16 +): neural allclose to an independent recomputation
from the raw `.mat` files for **every trial**, inputs equal the raw `blocked` field, output
distributions within 0.0017 of the raw 30 Hz moving-frame occupancy, 0.00000 of the output samples
fall in blocked partitions. **FAILURES: 0**

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
`Data format is valid, no errors or warnings.` - there is nothing to fix. Every structural
assertion of `verify_data_format` passes: 207 sessions in `neural`/`input`/`output`,
`subject_idx` int64 in range, `brain_region_idx` length == n_neurons for every session,
no NaN/Inf, neural is float and not all 0/1, input/output time dimensions equal the neural one,
outputs are whole numbers, `input_names`/`output_names`/`output_values` have the right lengths.

### Check 2: Independent sanity checks (`/app/cache/sanity_check.py`)
The script re-loads the raw `.mat` files with `h5py` and recomputes everything from scratch
(it never imports `convert_data.py`). On 4 randomly chosen sessions of the full dataset:

| Check | Criterion | Result |
|-------|-----------|--------|
| NEURAL, every trial | `np.allclose(converted, recomputed)` | True for all trials of all 4 sessions |
| NEURAL scalar spot check | `neural[trial 5, neuron 3, t 10]` | allclose (0.000000 vs 0.000000) |
| NEURAL non-zero spot check | argmax element of trial 5, e.g. `neural[trial5, neuron4, t37] = 30.000000` | allclose |
| INPUT | equals the raw `blocked` field of the MATLAB file; identical for every trial | True |
| INPUT | raw `blocked` == zeros of `flipud(get_env_mat(env))` | True |
| OUTPUT | class fractions vs those computed from the raw 30 Hz moving frames | max difference 0.0017 |
| OUTPUT | fraction of samples in blocked partitions | 0.00000 |
| **Total failures** | | **0** |

Earlier (Step 2) sanity checks against the dataset's own derived products:
- recomputed occupancy == `maps['sampling']` (1-frame precision) -> spatial binning correct
- recomputed rate maps == `maps['unsmoothed']` for every registered cell, identical NaN patterns
  -> `trace` and `position` are frame-aligned with **zero** lag
- `blocked` consistent with `get_env_mat` for **all 207 sessions**
- registered cell-sessions == 69,744 == the paper's rate-map count

### Check 3: Reference code comparison
| Stage | Reference (`/app/code/georepca1/src/utils.py`) | My `convert_data.py` | Same? |
|-------|---------------------------------------------|----------------------|-------|
| (a) loading | `load_dat` -> joblib / `mat73.loadmat`, fields `trace`, `position`, `envs`, `blocked` | `open_animal`/`read_session` -> `h5py` on the same `.mat` files, same fields | YES (verified byte-identical to the joblib arrays for a test session) |
| (b) neuron filtering | `decode_position_within`: `cell_idx = sum(traces[vel_idx]) > cell_threshold(=5)` | `keep_cells = registered & (events_moving > 5)` | YES (`registered` is implied in the reference: unregistered cells are NaN and `np.sum` of NaN is not > 5) |
| (b) trial/timepoint filtering | `vel_idx = gaussian_filter1d(norm(diff(behav)*fps), sigma=5) > v_thresh(=5)` | `compute_speed` is a line-by-line copy; `moving = speed > 5` | YES |
| (c) temporal alignment | frames of `trace` and `position` are used with the same index, no shift | identical; verified by reproducing `maps['unsmoothed']` | YES |
| (d) temporal binning | `fit_decoder`: `gaussian_filter1d(traces, sigma=3, axis=0)` then `AvgPool1d(kernel=3, stride=3)`; behaviour pooled with the **same** `AvgPool1d` | `gaussian_filter1d(sigma=3, axis=0)` then `pool_mean(k=3)` for neural, `pool_mean` for position and speed | YES (`pool_mean` is a numpy reimplementation of `AvgPool1d`) |
| (d) spatial binning | `behav / ((max + buffer)/n_bins)` then `floor` | `floor(position / (75/3))` clipped to [0,2] | YES - position is exactly [0,75] in every session, so both give identical bins (verified against `maps['sampling']`) |
| (e) input construction | `get_env_mat(env)` (+ `clean_rate_maps` orientation) | `env_blocked_vector(env)` = `flipud(get_env_mat(env)) == 0` | YES, and checked against the dataset's own `blocked` field for all 207 sessions |
| (f) output construction | one-hot of the binned position; predictions and ground truth snapped to the nearest bin belonging to the environment (`true_bins[np.argmin(actual_norms)]`) | class = 3*ybin + xbin; the 0.005% of samples that tracking noise puts inside a blocked partition are snapped to the nearest open partition | YES (3x3 analogue of the reference cleaning) |

**Deliberate differences, and why**
1. *Grid resolution 3x3 instead of 15x15.* Required by the Decoder Task ("3 x 3 = 9 spatial bins").
2. *Trials.* The experiment is 40 min of continuous exploration; the Decoder Task requires 1-min
   trials, so each session is cut into consecutive 60 s blocks.
3. *Units.* Neural data are multiplied by 30 to be in events/s. A pure rescaling; it does not
   change what a linear decoder can represent but keeps the values O(1) relative to the L1 penalty.
4. *Decoder model.* The provided harness (PCA projection + multinomial logistic regression per
   timepoint) replaces the reference Gaussian Naive Bayes. The processing of the data streams is
   unchanged.
5. *No place-cell selection.* The paper explicitly "motivated the inclusion of all cells in
   subsequent analyses".

### Check 4: Key statistics comparison
See the table in Step 9 - subjects (7), sessions (207, split 31/31/31/21/31/31/31), unique neurons
(5,413), mean cells/animal (773), geometries (10, square x27 and 20 each of the others),
session length (40 min) and registered cell-sessions (69,744) all match the paper and the raw data.
The only number that differs is the count of neuron-sessions actually used, 68,862 (98.7%),
because the reference decoder's `cell_threshold = 5` removes 882 essentially silent cell-sessions.
The output distribution across the 9 partitions,
`[0.095, 0.108, 0.114, 0.088, 0.070, 0.087, 0.109, 0.158, 0.171]`, matches the raw occupancy of the
moving periods to better than 0.002 in every session checked. The paper reports no other
distribution statistics.

### Check 5: Edge cases (`/app/cache/edge_checks.py`)
| Edge case | Result |
|-----------|--------|
| Sessions with < 2 trials | 0 (minimum is 22 trials, median 40) |
| All-zero or non-finite trials | 0 |
| Trial length | min 30 bins (the enforced minimum of 3 s of running), mean 316, max 561 |
| Neuron count consistency `brain_region_idx` vs neural rows | matches for all 207 sessions |
| Output classes per session == 9 - (number of blocked partitions) | 207/207 |
| Trailing frames | `pool_mean` drops at most 2 frames (67 ms) of a 40-min session |
| Final partial trial | 71,866 frames = 23,955 bins = 39 full 600-bin trials + 555 bins (55.5 s); kept because it exceeds the 50% threshold, so most sessions give 40 trials |
| dtypes | neural float32, input float32, output int64 - no dtype warnings |

### Issues Found and Resolved (iteration log)
**Iteration 1.** `edge_checks.py` found 4 sessions (QLAK-CA1-08 day 13 `u`, QLAK-CA1-30 day 6 `u`,
QLAK-CA1-51 day 4 `bit donut`, QLAK-CA1-74 day 9 `u`) whose output contained a class that the
geometry declares blocked - 127 of 2,573,752 samples (0.005%), i.e. DeepLabCut tracking noise a few
centimetres inside a partition wall. This is exactly the situation the reference
`decode_position_within` cleans up by snapping positions to the nearest bin that belongs to the
environment. *Fix*: `position_to_class` now snaps such samples to the centre of the nearest open
partition. *Re-check*: full conversion, verification, sanity checks and edge checks were all re-run;
"sessions where #classes != 9-#blocked" went from 4 to **0**, all other statistics were unchanged
(207 sessions, 8,147 trials, 68,862 neuron-sessions), and `sanity_check.py` still reports
**0 failures**. The effect on decoder accuracy was negligible (0.6617 both before and after),
as expected for 0.005% of the samples.

No other issues were found; all checks above were re-run after the fix.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, NVIDIA L4), ~7 min.

### Training Progress
- Loss decreasing: **Yes**, monotonically: 2.19 (epoch 1) -> 0.985 (10) -> 0.720 (80) ->
  0.660 (120) -> 0.6039 (200). Test loss 1.2119.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | x chance | Notes |
|--------|-------------|--------|--------|----------|-------|
| position_3x3 | 0.8251 | **0.6617** | 0.1111 | 5.96 | 9-way classification of the 3x3 partition |

Outputs `sample_trials.png` and `predictions.png` were produced by the harness.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Output | Classes | Chance (uniform) | Validation balanced acc | Ratio |
|--------|---------|------------------|-------------------------|-------|
| position_3x3 | 9 | 0.1111 | 0.6617 | **5.96x** |

Far above chance and far above the 1.5x threshold, so no bug is indicated.

### Check 2: Accuracy comparison to the paper
The paper reports **no classification accuracies** - its only decoding metric is the mean Euclidean
error of a Gaussian Naive Bayes decoder on the 15x15 grid (Figure 1F, and the per-session values
stored in `data/precomputed_results/within_decoding`). I therefore built two comparisons
(`/app/cache/gnb_benchmark.py`), both run on **my converted neural data**:

| Comparison | Paper / reference | This conversion | Verdict |
|------------|-------------------|-----------------|---------|
| Mean 15x15 Bayesian decoding error over 15 random sessions | 12.99 cm (`within_decoding`) | 13.71 cm | within 0.7 cm (5%) |
| Per-session 15x15 error, e.g. QLAK-CA1-50 day 0 / day 29 | 24.30 / 8.01 cm | 25.14 / 8.45 cm | tracks the reference session-by-session |
| 3x3 balanced accuracy with the **reference** GNB decoder | - | 0.596 | - |
| 3x3 balanced accuracy with the **provided** decoder | - | **0.662** | provided decoder beats the reference method by 0.066 |

The residual 0.7 cm is explained: the reference additionally snaps both the true and the predicted
15x15 bin to the nearest bin that belongs to the environment (`true_bins[np.argmin(...)]` in
`decode_position_within`), which can only reduce the error; my benchmark does not. Reproducing the
paper's own decoding error to within 5% from the converted data is the strongest available evidence
that the neural, position and geometry streams are loaded, filtered, aligned and binned correctly.

The per-session decoding error in the paper falls from ~31 cm on the first recordings to ~8 cm at
the end (Figure 1F). The same trend is present in the converted data (the sample of two early
sessions reaches only 0.42 balanced accuracy, while the full dataset reaches 0.66).

### Check 3: Train vs validation gap
Training 0.8251 vs validation 0.6617, ratio **1.25** (< 1.5). There is mild but unremarkable
overfitting, which is expected: each session has its own 100-PC projection fitted on ~10,000
timepoints, and the split is by trial so there is no leakage (a trial is either entirely in the
training set or entirely in the validation set, and the 1-min blocks are contiguous so temporally
adjacent samples are not split across the two sets).

### Additional debugging steps performed
1. Output values verified against the raw files for 4 random sessions (Step 10, check 2).
2. Temporal alignment verified two ways: by reproducing the dataset's own rate maps from
   `trace` + `position` (exact), and visually in panels 4-5 of the `--show-processing` figures.
3. Output variation: the least frequent class holds 7.0% and the most frequent 17.1% of the samples,
   so no class dominates.
4. Neural filtering follows `decode_position_within` exactly (see the Step 10 comparison table).
5. Processing matches the reference code (same table).

### Issues Found and Resolved
- None beyond the out-of-geometry sample issue already fixed and re-verified in Step 10.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created - dataset description, how to load the data, the full output-format
      specification, key statistics, a processing summary and the file list.
- [x] `cache/` folder created with `README_CACHE.md` documenting `sanity_check.py`,
      `edge_checks.py`, `gnb_benchmark.py` and `paper.txt`.
- [x] Stale artefacts removed (`__pycache__`, an obsolete processing figure from an earlier
      sample-session choice).
- [x] All required outputs present in `/app`:
      `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
      `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
      `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
      `train_decoder_full_out.txt`, plus the `--show-processing` figures
      `processing_QLAK-CA1-51_day00.png` and `processing_QLAK-CA1-56_day09.png` and the harness
      figures `sample_trials.png` / `predictions.png`.

### Final summary
| | |
|---|---|
| Subjects / sessions / trials | 7 / 207 / 8,147 |
| Neuron-sessions | 68,862 (of 69,744 registered; 5,413 unique neurons) |
| Time bin | 100 ms |
| Decoder input | 9 binary blocked-partition flags, static per trial |
| Decoder output | 3x3 position, 9 classes, time-varying |
| Validation balanced accuracy | **0.6617** (chance 0.1111, 5.96x) |
| Reference-method cross-check | paper's Bayesian 15x15 decoding error reproduced to within 0.7 cm |
