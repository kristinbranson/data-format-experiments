# Dataset Conversion Notes

## Overview
- **Dataset**: Lee, Keinath, Cianfarano & Brandon (2025), *"Identifying representational structure in CA1 to
  benchmark theoretical models of cognitive mapping"*, Neuron 113(2):307-320.
  Miniscope 1-photon calcium imaging of dorsal CA1 in freely-moving mice during a geometric-deformation paradigm.
  Sources: `/app/paper.pdf`, `/app/methods.txt`, `/app/code` (github.com/jquinnlee/georepca1), `/app/data` (Zenodo 13993254).
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format — decode the animal's position (3x3 = 9 spatial bins)
  from CA1 population activity, with the environment geometry (which partitions are blocked) as decoder input.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `paper.pdf` (20 pages), `methods.txt` (excerpted methods)
- `code/` — `README.md`, `environment.yml`, `georepca1/{main.py, src/utils.py, src/plots.py, demos/*.ipynb}`
- `data/` — 7 animal files in **two** formats each: joblib (`QLAK-CA1-XX`) and MATLAB v7.3/HDF5 (`QLAK-CA1-XX.mat`),
  plus `behav_dict` and `precomputed_results/` (`within_decoding`, `*_shr`, `*_rsm_partitioned`, `map_corr_envs`, ...)
- `decoder.py`, `train_decoder.py` (provided decoder reference)

Environment verified: numpy 2.4.4, torch 2.6.0+cu124 (CUDA available, NVIDIA L4 23 GB), scipy 1.18.0,
h5py 3.16.0, pandas 3.0.5, joblib. 1006 GB RAM, 128 CPUs.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat(animal, p, format)` | `src/utils.py:61` | LOADING | Loads one animal from `.mat` (mat73) or joblib. Returns `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}` |
| `mat2joblib` / `save_dat` | `src/utils.py:87,100` | LOADING | The joblib files in `/app/data` are just the `.mat` files re-saved; contents are identical |
| `get_env_mat(env)` | `src/utils.py:215` | PROCESSING | 3x3 binary "open" matrix for each of the 10 geometry names |
| `get_rate_maps(position, trace, n_bins=15, fps=30, filter_size=1.5)` | `src/utils.py:313` | PROCESSING | Bins position by integer division, accumulates binary trace per bin, divides by occupancy. Confirms `position`/`trace` are frame-aligned at 30 Hz and that maps are indexed `[xbin, ybin]` |
| `get_split_half`, `get_shuffle_split_half`, `get_place_cells`, `get_shr_within` | `src/utils.py:356-462` | CURATION (not used for decoding) | Place-cell identification by split-half rate-map reliability vs 1000 circular shuffles |
| `clean_rate_maps(maps, envs)` | `src/utils.py:463` | PROCESSING | Masks rate-map pixels outside the geometry using `np.fliplr(get_env_mat(env).T)` upsampled 3x3 -> 15x15 |
| **`decode_position_within(behav, traces, maps, n_bins=15, fps=30, v_filt_size=5, v_thresh=5, cell_threshold=5, buffer=1e-15, n_fold=5)`** | `src/utils.py:1845` | **THE reference analysis for our task** | 5-fold cross-validated naive-Bayes decoding of position within each session |
| `fit_decoder(behav, traces, feature_max, temporal_bin_size=3)` | `src/utils.py:1776` | PROCESSING | `gaussian_filter1d(traces, sigma=3, axis=0)` then `AvgPool1d(kernel=3, stride=3)` on traces **and** position -> 10 Hz; position one-hot encoded; `GaussianNB(priors=flat)` |
| `test_decoder(...)` | `src/utils.py:1806` | PROCESSING | Same temporal binning applied to test data; error = Euclidean distance between decoded and true spatial bin (in cm) |
| `get_all_decoding_within` | `src/utils.py:1969` | — | Loads `results/within_decoding` (provided in `data/precomputed_results/`) |

### Notes — the reference decoding pipeline (`decode_position_within`, called in `main.py:41-48`)
Called as `decode_position_within(dat[animal]['position'].T, dat[animal]['trace'].T, dat[animal]['maps']['smoothed'])`,
i.e. `behav` is `(T, 2, n_days)` and `traces` is `(T, n_cells, n_days)`.

Steps, in order:
1. `bin_down = (global max over x, y and days + buffer) / n_bins` -> position in bin units. Position spans exactly
   0-75 cm, so `bin_down = 5 cm` for `n_bins=15`.
2. **Velocity filter**: `v = gaussian_filter1d(||diff(pos)||*fps, sigma=5)` computed at 30 Hz;
   frames with `v <= 5 cm/s` are **excluded** (`vel_idx`). `vel_idx[0] = False` (first frame always dropped).
3. **Cell filter**: `cell_idx = traces[vel_idx].sum(axis=0) > 5` — a cell is kept for that session only if it has
   **more than 5 binarised events during running**.
4. 5-fold `KFold` split (not shuffled) of the retained frames.
5. Per fold: `gaussian_filter1d(traces, sigma=3, axis=0)` then `AvgPool1d(3,3)` -> **100 ms / 10 Hz bins**;
   position average-pooled the same way and floored to integer bins.
6. `GaussianNB` with flat priors; error = Euclidean distance (cm) between predicted and true bin.

**Calcium processing**: none needed in this conversion. `trace` in the distributed dataset is **already** the
final binarised "rising phase of transient" vector (values in {0, 1}, NaN for cells not registered that day).
dF/F extraction, motion correction, CNMF-E segmentation, transient extraction and z-scoring at 2.5 SD were all
performed upstream in MATLAB by the authors (methods.txt, "Data preprocessing"). **No dF/F needs to be computed.**

**Place-cell filtering is NOT applied** for decoding: the paper states the spatial-coding quality results
"motivated the inclusion of **all** cells in subsequent analyses".

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/QLAK-CA1-{08,30,50,51,56,74,75}` (joblib) and `.mat` (MATLAB v7.3 = HDF5). The two are identical in
content; the `.mat` is used in this conversion because HDF5 allows **lazy, per-day reads** (a whole animal
decompressed from joblib is 9-25 GB of float64 in RAM).

Per animal, fields (joblib orientation):
| Field | Shape | Meaning |
|---|---|---|
| `trace` | `(n_days, n_cells, T)` | Binarised calcium-transient rising phases, {0,1}; **all-NaN** for a cell not registered that day |
| `position` | `(n_days, 2, T)` | DeepLabCut head position in cm, `[x, y]`, range exactly 0-75 cm |
| `envs` | `(n_days, 1)` str | Geometry name: `square, o, t, u, rectangle, +, i, l, bit donut, glenn` |
| `blocked` | list of `n_days` | Indices (0-8) of blocked partitions in the 3x3 grid, `[[0,1,2],[3,4,5],[6,7,8]]`; `-1` = none blocked (square) |
| `maps` | `sampling (15,15,n_days)`, `smoothed/unsmoothed (15,15,n_cells,n_days)` | Precomputed occupancy (seconds) and rate maps, indexed `[xbin, ybin, ...]` |
| `SFPs` | `(35,35,n_cells,n_days)` | Spatial footprints (not needed) |
| `centroids` | `(n_cells,2,n_days)` | ROI centroids (not needed) |

In the HDF5 file every dimension order is reversed relative to the joblib arrays
(e.g. `f['maps/sampling'][d]` == joblib `maps['sampling'][:, :, d].T`).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects (mice) | 7 |
| Sessions (days) total | **207** (31 days x 6 animals + 21 days for QLAK-CA1-51) |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Unique neurons (total, sum of per-animal cell registries) | **5,413** (515, 875, 942, 554, 862, 713, 952) |
| Neurons / animal | mean 773.3, min 515, max 952 |
| Registered cell-sessions ("rate maps") | **69,744** (mean 336.9 / session) |
| Frames / session | 71,866 - 72,219 @ 30 Hz = 2,395 - 2,407 s ≈ 40 min |
| Geometries | 10, each repeated once per sequence, up to 3 sequences, each sequence starts/ends with `square` |
| Trials | not defined in the source data — created here as 1-min blocks (see Step 5) |

Per-animal geometry order is a fixed random permutation repeated 3x (2x for `-51`), always starting/ending with
`square`, exactly as described in the paper.

`blocked` vs `get_env_mat`: verified `blocked` lists the flattened zeros of `np.flipud(get_env_mat(env))` for all
10 geometries, i.e. `blocked` indexes the grid as `3*row + col` in the convention drawn in the paper.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 | "5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps" |
| Sessions | 207 | same |
| Rate maps (registered cell-sessions) | 69,744 | same |
| Subjects | 7 | 7 animal IDs in `main.py`; "mean number of cells per animal = 773 ± 68 SE" over 7 animals |
| Neurons / animal | mean 773 ± 68 SE, min 515, max 952 | "mean number of cells per animal = 773 ± 68 SE, minimum cells per animal = 515, maximum cells per animal = 952" |
| Sessions / subject | 31 (up to 3 sequences of 10 geometries + final square) | "(D) ... up to three total repetitions (31 days)" |
| Session duration | 40 min, one session/day | "All sessions were 40 min, and one session was recorded per day" |
| Acquisition rate | 30 Hz, behaviour and imaging simultaneously timestamped | "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz ... all recorded frames were timestamped for post-hoc alignment" |
| Arena | 75 x 75 cm square partitioned into a 3 x 3 grid; 25 cm partition walls | "We partitioned an open square (75 x 75 cm) into a 3 x 3 grid space" |
| Neural data time bin (decoding) | 3 frames = 100 ms | `fit_decoder(..., temporal_bin_size=3)` with `fps=30` |
| Behaviour data time bin (decoding) | same 100 ms (same `AvgPool1d`) | `fit_decoder` |
| Spatial bins (paper decoding) | 15 x 15 (5 cm) | `decode_position_within(n_bins=15)`; "spatially binning position data into pixels corresponding to a 5 cm x 5 cm grid" |
| Running-speed threshold | 5 cm/s, velocity smoothed with sigma = 5 frames | `decode_position_within(v_thresh=5, v_filt_size=5)` |
| Cell inclusion for decoding | > 5 events while running | `cell_threshold=5` |
| Firing rate definition | binarised transient rising phase | "This binary vector was treated as the firing rate in all further analyses" |
| Place cells | split-half reliability > 99th pct of 1000 shuffles — **but all cells used for analyses** | "motivated the inclusion of all cells in subsequent analyses" |
| Bayesian decoding error (15x15 bins) | mean 13.49 cm over 207 sessions (range 6.53-31.01 cm per session) | `data/precomputed_results/within_decoding`, the exact numbers behind Figure 1F |

### Processing Details
- Temporal alignment: imaging and behaviour are acquired **on the same DAQ at 30 Hz and timestamp-aligned by the
  authors**; `position[d]` and `trace[d]` have identical frame counts and are index-for-index aligned.
  There is no trial structure and no stimulus event in this experiment — a session is 40 min of free foraging.
- Temporal binning for decoding: gaussian smoothing (sigma = 3 frames) then average pooling by 3 frames -> 100 ms.
- Spatial binning: `floor(position / bin_size)` with `bin_size = 75 cm / n_bins`.

### Curation Steps
**Neuron curation rules** (as used by the reference decoder):
1. Cell must be registered on that session (trace not all-NaN).
2. Cell must have > 5 binarised events during running frames in that session.
(No place-cell selection — the paper explicitly includes all cells.)

**Trial curation rules**: the source data has no trials. The reference excludes **frames** with running speed
<= 5 cm/s. Frame 0 of each session is always excluded (no velocity estimate).

**Session/subject curation**: none — all 7 animals and all 207 sessions are used in the paper.

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Position, 15x15 (5 cm) bins, Gaussian naive Bayes, 5-fold CV within session | Mean Euclidean error **13.49 cm** (Figure 1F); significant decrease across days (ANOVA p<0.0001, F=7.9845) |

The paper reports decoding **error in cm**, not classification accuracy, and never decodes into a 3x3 grid.
A 13.5 cm mean error is ~0.54 of a 25 cm bin, so a well-converted 3x3 decode should be well above the 1/9 = 11.1%
chance level (an expectation of roughly 60-80% for the naive-Bayes reference; the provided decoder is a linear
read-out of 100 PCs, so somewhat lower is plausible).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Number of neurons / sessions / rate maps | 7 animals in `main.py` | 5,413 cells, 207 days, 69,744 registered cell-days | "5,413 unique neurons across 207 sessions ... 69,744 rate maps" | **Exact match** — confirms cell registry = `trace` second axis and "registered" = trace not all-NaN |
| Position -> 3x3 bin index | `clean_rate_maps` masks maps with `np.fliplr(get_env_mat(env).T)` | occupancy is **exactly zero** in every `blocked` partition when `bin = 3*floor(y/25) + floor(x/25)` | `blocked` "organized in the following way - [[0,1,2],[3,4,5],[6,7,8]]" | Resolved: `bin = 3*ybin + xbin`. Algebraically `fliplr(E.T)[x,y] == flipud(E)[y,x]`, i.e. the map mask and the `blocked` index agree under this convention. Empirically verified on every geometry (see Step 10) |
| Spatial bin size | `decode_position_within` uses per-dataset max position | `position` spans exactly 0-75 cm overall, but **individual sessions do not**: e.g. `rectangle` sessions have `min(x) = 25` and several sessions have `max < 74` | arena is 75 x 75 cm | Use a **fixed 25 cm bin size from the 75 cm arena**, not a per-session max. Verified: reproducing the stored occupancy map (`maps/sampling`) requires `bin = 75/15`; a per-session max is off by up to 0.77 s of occupancy, while 75/15 matches to 1 frame (0.033 s) |
| Rate-map scaling | `get_rate_maps` multiplies by `fps` | stored `maps['unsmoothed']` are 30x smaller than `get_rate_maps` output (correlation 0.9999) | — | Irrelevant for this conversion (we never use the stored rate maps), but it confirms frame-by-frame `trace`/`position` alignment |
| Calcium processing | no dF/F code in the repo | `trace` values are exactly {0, 1} (or NaN) | binarisation described in methods, done upstream in MATLAB | Nothing to compute; use `trace` as the firing rate, as the paper does |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day]` (n_cells, T) @30 Hz | `neural[session][trial]` (n_neurons, T_trial) | drop unregistered cells (all-NaN) -> drop cells with <= 5 events while running -> `gaussian_filter1d(sigma=3 frames)` -> average-pool 3 frames (100 ms) -> keep running bins -> cut into 1-min blocks | `decode_position_within` (cell filter), `fit_decoder` (smoothing + pooling) | float32 |
| `position[day]` (2, T) @30 Hz | `output[session][trial]` (1, T_trial) | average-pool 3 frames -> `bin = 3*clip(floor(y/25),0,2) + clip(floor(x/25),0,2)` | `fit_decoder` (pooling + one-hot of the 2-D bin), `decode_position_within` (`bin_down`) | int, 9 categories |
| `blocked[day]` | `input[session][trial]` (9,) | 9-dim binary vector, 1 = partition blocked; `-1` -> all zeros | `get_env_mat` / dataset README | static per trial |
| `envs[day]` | `metadata['session_info']` | geometry name string | `get_env_mat` | also used to cross-check `blocked` |
| animal ID | `subjects` / `subject_idx` | 7 IDs | `main.py` animal list | |
| — | `brain_regions` = `['CA1']`, `brain_region_idx` = zeros | all cells are dorsal CA1 | methods | |

### Key Decisions
1. **Session = recording day.** 207 sessions. Cell identity is only stable within a day in the format required
   here (neuron count must be constant within a session but may differ between sessions), and the geometry — the
   decoder input — changes day to day.
2. **Trial = contiguous 60 s block of the session** (600 bins of 100 ms), as specified in the Decoder Task.
   ~40 trials per session. The trailing partial block is kept if it covers >= 10 s of recording, otherwise dropped
   (nothing else is discarded, so essentially no data is lost).
3. **Time bin = 100 ms (3 frames @30 Hz)** with gaussian pre-smoothing sigma = 3 frames — exactly
   `fit_decoder`'s `temporal_bin_size=3`. This also makes the neural values continuous rather than binary, which
   the validator requires ("Neural data lacks variability" error if all values are 0/1).
   *Difference from the reference*: the reference smooths **after** discarding immobility frames, which convolves
   across temporal discontinuities. Here smoothing and pooling are done on the intact 30 Hz session and the
   speed criterion is applied afterwards, which is strictly more correct and otherwise identical.
4. **Exclude immobility (speed <= 5 cm/s)** using the reference's velocity estimate
   (`gaussian_filter1d(|diff(pos)|*30, sigma=5)` at 30 Hz, downsampled to 100 ms bins by majority vote).
   This is the reference decoder's own curation rule, and during immobility CA1 activity reflects
   replay/sharp-wave-ripple content rather than the animal's current location. The provided decoder is memoryless
   (per-timepoint linear read-out of neural PCs + input), so removing timepoints does not disturb it.
   Trials retaining < 30 bins (3 s) after this filter are dropped.
5. **Neuron curation = registered on the day AND > 5 events while running** (the reference rule). This removes
   only ~0.7% of registered cell-sessions. No place-cell selection, per the paper.
6. **Output = 9 position bins, time-varying**, one output variable with 9 categories. Blocked bins simply never
   occur in the sessions where they are blocked; all 9 occur across the dataset.
7. **Input = 9-dim binary "blocked" vector, static per trial**, exactly as the Decoder Task specifies. It tells
   the decoder which of the 9 output classes are reachable in that session.
8. **All 7 animals and all 207 sessions are used** — the paper applies no session- or animal-level exclusion.
9. **Read from the `.mat` (HDF5) files** rather than joblib, for lazy per-day access; content is byte-identical
   (the joblib files were produced from the `.mat` by `mat2joblib`). Verified in Step 10 Check 2.

### Planned Sanity Checks
- [ ] Totals equal the paper: 7 subjects, 207 sessions, 5,413 unique cells, 69,744 registered cell-sessions.
- [ ] Occupancy of every `blocked` partition is exactly 0 under `bin = 3*ybin + xbin`, for all 207 sessions.
- [ ] Reproduce the stored `maps['sampling']` occupancy map from `position` (validates position binning).
- [ ] Reproduce the stored `maps['unsmoothed']` rate maps (up to the known 1/fps scale) from `position` + `trace`
      (validates neural/behaviour temporal alignment).
- [ ] Re-derive converted `neural`, `input` and `output` for spot-checked (session, trial, time, neuron) entries
      straight from the raw `.mat`, compared with `np.allclose`.
- [ ] h5py-read day slices equal the joblib arrays used by the reference code.
- [ ] Trial time coverage: sum of trial durations ~= session duration x fraction of running frames.
- [ ] Output distribution: 9 classes, no NaN, all within 0-8; blocked bins absent in the right sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Runs as `python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]`.

Structure:
- `read_session(f, day)` — lazy per-day read from the MATLAB v7.3/HDF5 file (`position`, `trace`, `envs`, `blocked`).
- `blocked_from_env(env)` — re-derives the blocked list from the reference `get_env_mat`; asserted against the
  stored `blocked` for every one of the 207 sessions (a built-in consistency check that runs on every conversion).
- `running_speed(position)` — the reference's velocity estimate, in cm/s.
- `position_to_bin(pos_cm)` — `3*clip(floor(y/25),0,2) + clip(floor(x/25),0,2)`.
- `process_session(...)` — neuron curation -> temporal smoothing/binning -> speed masking -> 1-min trial split.
- `plot_processing(...)` — 7-panel figure covering every processing step (`--show-processing`).
- `process_animal(task)` / `main()` — one worker process per animal (`multiprocessing.Pool`), results merged and
  sorted by `(animal, day)`.

Code inefficiencies identified:
- Loading a whole animal from joblib materialises 9-25 GB of float64 in RAM (`trace` is `n_days x n_cells x T`).
- `gaussian_filter1d` on float64 is ~10% slower than on float32 and doubles memory traffic.
- Naive per-frame python loops (as in the reference `get_rate_maps` / `fit_decoder` one-hot construction) are slow.

Code speedups added:
- Lazy per-day HDF5 reads instead of whole-animal joblib loads (memory stays ~0.3 GB per worker).
- `trace` cast to float32 at read time; smoothing and pooling done with reshape+`mean` instead of `torch.AvgPool1d`.
- Position binning and one-hot-free integer bin indices computed vectorised (`np.floor`/`np.clip`), no python loop.
- 7 animal-level worker processes (`--nproc`, default 7).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sample = QLAK-CA1-08 day 5, geometry `+`, and QLAK-CA1-30 day 5, geometry `i` — two subjects, two geometries).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 2 |
| Sessions | 2 |
| Sessions / subject | 1 |
| Neurons (total cell-sessions) | 551 |
| Neurons / session | mean 275.5, min 180, max 371 (from 186 and 380 registered cells) |
| Trials (total) | 76 |
| Trials / session | 40, 36 |
| Timepoints / trial | mean 277.9, min 46, max 445 (of a maximum 600) |
| Fraction of bins running | 57.5% (`+`), 31.9% (`i`) |
| Input `blocked_*` range | [0, 1]; 4 blocked for `+`, 2 for `i` |
| Output `position_bin` range | [0, 8] |
| Output distribution (pooled) | [0.039, 0.171, 0.056, 0.088, 0.184, 0.132, 0.058, 0.205, 0.067] |
| Output distribution (`+` session) | [0, 0.206, 0, 0.136, 0.214, 0.205, 0, 0.240, 0] — exactly zero in blocked bins 0,2,6,8 |
| Output distribution (`i` session) | [0.110, 0.108, 0.159, 0, 0.131, 0, 0.162, 0.142, 0.188] — exactly zero in blocked bins 3,5 |

### Processing Plots Review
`processing_QLAK-CA1-08_day05.png`, `processing_QLAK-CA1-30_day05.png`. Seven panels covering raw 30 Hz position,
running speed + retained frames, raw binarised events, smoothed/pooled `neural`, the discretisation check
(continuous `x/25`, `y/25` overlaid on the integer output bin), trial boundaries + retained 100 ms bins, and the
occupancy of the 9 output bins against the `input` geometry vector. No anomalies:
- the output bin steps exactly when `floor(x/25)` or `floor(y/25)` changes, with no lag relative to the position
  traces (temporal alignment is correct);
- retained bins coincide with the shaded running periods in the speed panel;
- occupancy is identically zero in every partition flagged blocked in `input`.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| HDF5 lazy per-day reads instead of joblib whole-animal load | avoids ~9 s + 9-25 GB per animal, and re-reading |
| float32 traces, reshape/mean pooling instead of torch AvgPool1d | ~2x on the filter/pool step |
| 7 worker processes (one per animal) | ~6x wall-clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| read + process one session | 1.7-2.1 s | 207 x ~2.0 s = ~7 min serial |
| with 7 animal workers (longest animal = 31 sessions) | — | **~1.5 min** |
| pickle write (~3 GB) | — | ~1 min |
| **Full conversion estimate** | | **< 5 min** (well under the 15 min budget; no further optimisation needed) |

### Verification (`/app/verification_sample_out.txt`)
`Data format is valid, no errors or warnings.` Input/output ranges and distributions as tabulated above.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| position_bin | 0.7621 | **0.6211** | 0.1111 |

Loss decreased monotonically over all 200 epochs (2.2757 -> 0.9905; test loss 1.2157).
Validation accuracy is 5.6x chance on only two sessions, so the conversion is sound.
(Re-run after the Step-10 fix; the pre-fix run gave 0.7540 / 0.5863 — the fix touches 0.006% of samples,
the difference is run-to-run variation in the decoder's random train/test split.)

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full` — **44 s wall clock** (7 worker processes;
39.8 s processing + 4.3 s pickling), well inside the estimate from Step 7.

### Output Files
- `converted_data.pkl`: **3.44 GB**, 207 sessions, 8,148 trials, 2,569,722 timepoints (71.4 h of running)
- `verification_full_out.txt`: created — `Data format is valid, no errors or warnings.`

### Converted dataset statistics
| Statistic | Value |
|---|---|
| Subjects | 7 |
| Sessions | 207 (31/31/31/21/31/31/31) |
| Trials | 8,148 (mean 39.36/session, min 22, max 40; 161 of 207 sessions have the full 40) |
| Neurons / session | mean 332.7, min 112, max 562; 68,862 cell-sessions |
| Registered -> kept cells | 69,744 -> 68,862 (882 dropped, **1.26%**, by the >5-events-while-running rule) |
| Timepoints / trial | mean 315.4, min 30, max 561 (of a possible 600) |
| Fraction of bins retained (running) | mean 51.8%, min 21.7%, max 75.3% |
| Geometries | square x27, each of the other 9 x20 = 207 |
| Mean event rate | 0.101 events/s/cell |
| Samples snapped to nearest open bin | 152 of 2,569,722 (**0.006%**) |

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 7 | 7 animal IDs in `main.py` | 7 files | 7 | YES |
| Sessions | 207 | — | 31+31+31+21+31+31+31 = 207 | 207 | YES |
| Unique neurons | 5,413 | — | 5,413 | 5,413 (`n_cells_in_registry`) | YES |
| Neurons / animal | mean 773 ± 68 SE, min 515, max 952 | — | 515/875/942/554/862/713/952 | mean 773.3 ± 68 SE, min 515, max 952 | YES |
| Registered cell-sessions ("rate maps") | 69,744 | — | 69,744 | 69,744 (`n_registered_cells`) | YES |
| Session duration | 40 min | — | 71,866-72,219 frames @30 Hz = 39.9-40.1 min | 2,395-2,407 s -> 40 trials of 60 s | YES |
| Geometries | 10, sequence repeated up to 3x, starts/ends with square | `get_env_mat` lists 10 | 10 names, fixed per-animal permutation repeated 3x (2x for -51) | square x27, others x20 | YES |
| Neural time bin | 100 ms | `fit_decoder(temporal_bin_size=3)` @ 30 fps | — | 100 ms | YES |
| Behaviour time bin | 100 ms | same `AvgPool1d` | — | 100 ms | YES |
| Speed threshold | — | `v_thresh=5` cm/s, sigma 5 frames | 51.8% of frames running | 51.8% of bins retained | YES |
| Cell threshold | "inclusion of all cells" (no place-cell filter) | `cell_threshold=5` events while running | — | 1.26% of cell-sessions dropped, no place-cell filter | YES |
| Arena / bin size | 75 x 75 cm, 3 x 3 grid | `bin_down = max/n_bins` | position range exactly [0, 75] cm | 25 cm bins | YES |
| Input range | blocked partitions per geometry | `get_env_mat` | `blocked` field | [0, 1], mean 2.31 blocked/trial | YES |
| Output range/distribution | — | — | occupancy zero in blocked bins | [0..8], [.095 .108 .114 .088 .070 .087 .109 .158 .171] | YES |
| Bayesian decoding error, 15x15 | mean 13.49 cm (Fig 1F) | `decode_position_within` | `precomputed_results/within_decoding` | reproduced **exactly** on 12 spot-checked sessions (Step 10 Check 3) | YES |

No inconsistencies remain.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
`Data format is valid, no errors or warnings.` — nothing to fix. Every structural field
(`subject_idx` dtype/range, `brain_region_idx` length per session, matching trial counts across
neural/input/output, matching T between neural and output, integer-valued outputs, no NaN/Inf,
consistent `dinput`/`doutput`, `input_names`/`output_names`/`output_values` lengths) passes.

One benign observation from the summary: `blocked_x1y0` has range `[0, 0]` across the whole dataset.
This is a property of the experiment, not a bug: the union of blocked partitions over the 10 geometries
is `{0,1,2,3,4,5,6,8}` — the bottom-centre partition (index 7 in the dataset's convention) is open in
every geometry. It is kept in the input vector so that the 9 inputs align one-to-one with the 9 output
bins.

### Check 2: Sanity checks against the original files
`/app/sanity_checks.py` (output in `/app/sanity_checks_out.txt`). It imports nothing from
`convert_data.py` and re-derives everything from the raw `.mat`/joblib files. **All checks pass.**

| # | Stream | Check | Result |
|---|---|---|---|
| 0 | all | HDF5 `.mat` reads == joblib arrays that `utils.load_dat` returns (position, trace incl. NaNs, envs), days 0/5/30 | PASS |
| 1 | output | Position binning reproduces the dataset's own stored occupancy map `maps['sampling']` (15x15) for days 0/5/17 | PASS, max diff 0.033 s (one frame) of up to 118 s |
| 2 | neural | Rate maps rebuilt from raw `position`+`trace` match the stored `maps['unsmoothed']`, cells 0/63/143 | PASS, Pearson r = 1.000000 — proves frame-by-frame neural/behaviour alignment |
| 3 | output/input | Occupancy is **exactly zero** in every blocked partition of all 207 sessions, and the `input` vector flags exactly the `blocked` partitions | PASS (480/480 blocked partitions) |
| 3b | output | Every open partition is visited in every session | PASS (207/207) |
| 4 | neural | Full independent re-implementation, `np.allclose(..., atol=1e-6)` on **every trial** of 3 sessions (QLAK-CA1-08 d5/d30, QLAK-CA1-75 d12) | PASS, 0 mismatches |
| 4b | neural | Explicit element spot-check `neural[trial 5, neuron 3, t 10]` = 0.68481296 | PASS |
| 4c | output | `output[trial 5, t 10]` and element-wise equality over every trial | PASS, 0 mismatches |
| 4d | input | `input` == 9-dim indicator of the raw `blocked` field, every trial | PASS |
| 5 | totals | 7 subjects, 207 sessions, 5,413 neurons, 69,744 rate maps, 773 ± 68 SE cells/animal | PASS |
| 6 | structure | Consistent shapes, finite values, outputs in 0-8, 30 <= T <= 600 | PASS |
| 6b | coverage | Retained time (71.38 h) == recorded time x running fraction (71.46 h) to 0.12% | PASS |

A further built-in check runs on **every** conversion: for all 207 sessions the stored `blocked` list is
asserted equal to `np.flatnonzero(np.flipud(get_env_mat(env)) == 0)` re-derived from the reference
`get_env_mat` and the geometry name.

### Check 3: Reference code comparison
`/app/reference_check.py` copies `fit_decoder`, `test_decoder` and the `distances` path of
`decode_position_within` **verbatim** from `georepca1/src/utils.py` and runs them on the raw joblib data.
(The only edit is `np.where(...)[0]` -> `np.where(...)[0][0]`, forced by numpy >= 1.25 no longer allowing a
length-1 array to be assigned to a scalar element; the value is identical.)

> Reference decoder at n_bins=15 versus the authors' own stored `within_decoding` (= Figure 1F):
>
> | animal | day | env | reference error | stored error | diff |
> |---|---|---|---|---|---|
> | QLAK-CA1-08 | 0 | square | 26.373 | 26.373 | 0.000 |
> | QLAK-CA1-08 | 5 | + | 14.848 | 14.848 | 0.000 |
> | QLAK-CA1-08 | 15 | + | 11.124 | 11.124 | 0.000 |
> | QLAK-CA1-08 | 25 | + | 12.251 | 12.251 | 0.000 |
> | QLAK-CA1-08 | 30 | square | 16.172 | 16.172 | 0.000 |
> | QLAK-CA1-50 | 0/10/20/30 | square | 24.298 / 12.558 / 12.050 / 9.143 | identical | 0.000 |
> | QLAK-CA1-51 | 3/12/20 | u/l/square | 28.098 / 13.956 / 14.575 | identical | 0.000 |
>
> **12/12 sessions reproduce the published values to the last printed digit.** This establishes that the
> velocity filter, cell filter, temporal smoothing/pooling and spatial binning used in this conversion are
> the reference ones.

Step-by-step comparison of `convert_data.py` against the reference:

| Stage | Reference (`utils.py`) | This conversion | Same? |
|---|---|---|---|
| (a) data loading | `load_dat` -> joblib/`mat73`; `position` (n_days,2,T), `trace` (n_days,n_cells,T) | `read_session` -> h5py on the same `.mat`; identical arrays (Check 0) | YES (equivalent, lazier) |
| (b) neuron filtering | registered = non-NaN; `cell_idx = traces[vel_idx].sum(0) > 5` | same two rules, same threshold | YES |
| (b') place cells | `get_place_cells` exists but is **not** applied to decoding; paper includes all cells | not applied | YES |
| (c) temporal alignment | `position[d]` and `trace[d]` are frame-aligned at 30 Hz by the authors' DAQ timestamps; no re-alignment | no re-alignment; verified by rate-map reproduction (Check 2) | YES |
| (d) binning | `gaussian_filter1d(traces, sigma=3, axis=0)` then `AvgPool1d(3,3)`; position `AvgPool1d(3,3)` then `int()` | identical, implemented as reshape+`mean` | YES, with one deliberate difference (below) |
| (e) input construction | reference has no decoder input; `blocked`/`get_env_mat` define the geometry | 9-dim binary blocked vector | N/A — required by the Decoder Task |
| (f) output construction | `behav / bin_down` floored, one-hot over `n_bins x n_bins`; predicted/true bins snapped to the nearest valid bin of `temp_maps` | `3*floor(y/25) + floor(x/25)` with `n_bins=3`; samples in a blocked partition snapped to the nearest open partition | YES in kind; `n_bins` 15 -> 3 is required by the Decoder Task |

**Deliberate differences, and why:**
1. **`n_bins = 3` instead of 15.** Required by the Decoder Task ("3 x 3 = 9 spatial bins").
2. **Bin size fixed at 75 cm / n_bins instead of `(max position + buffer) / n_bins`.** The reference takes the
   maximum over *all* days of an animal, which for this dataset equals 75 cm anyway; taking it per session
   would be wrong because walled-off geometries do not span the arena. Reproducing the dataset's stored
   occupancy maps requires exactly 75 cm (Check 1). Same numbers, safer derivation.
3. **Smoothing applied before, not after, discarding immobility frames.** The reference calls
   `gaussian_filter1d(traces, sigma=3)` on the already-subsetted (and already fold-split) frames, so the
   kernel bridges temporal discontinuities and fold boundaries. Here the session is smoothed and pooled
   intact at 30 Hz and the speed criterion is applied to the resulting 100 ms bins, which is strictly more
   correct and produces the same quantity everywhere away from a discontinuity.
4. **Trials.** The source data has no trials; 1-minute blocks are imposed by the Decoder Task.
5. **Decoder-input vector.** Required by the Decoder Task; the reference decoder has no side input.

### Check 4: Key statistics comparison
See the table in Step 9. Every number that the paper, the reference code or the reference data provides —
7 subjects, 207 sessions, 5,413 neurons, 69,744 rate maps, 773 ± 68 SE / 515 / 952 cells per animal, 40 min
sessions at 30 Hz, 10 geometries in a thrice-repeated per-animal sequence, 100 ms decoding bins, 5 cm/s
speed threshold, >5-event cell threshold, and the Figure 1F decoding errors — matches. No discrepancy
required investigation beyond the four resolved in Step 4.

### Check 5: Edge cases
| Edge case | Handling |
|---|---|
| Cells not registered on a session (NaN traces) | dropped per session; asserted that a registered cell never has partial NaNs |
| `blocked = -1` (open square) | filtered to an empty list -> all-zero input vector |
| Position exactly at 75.0 cm (the arena edge) | `np.clip(..., 0, 2)`; the reference relies on a `1e-5` buffer that is lost to float rounding at 75.0, so clipping is the safer equivalent |
| Frame 0 has no velocity estimate | speed set to 0, so frame 0 is always excluded — matches `vel_idx[d, 1:] = ...` |
| Session length not a multiple of 3 frames | the trailing 1-2 frames are dropped (<= 67 ms of 40 min) |
| Session length not a multiple of 600 bins | trailing partial block kept as a shorter trial if it spans >= 10 s; 71,866 frames give 39 full 60 s trials + a 55 s trial |
| Trial left with very little running | trials with < 30 bins (3 s) dropped; this is why 46 sessions have < 40 trials (min 22) |
| Sessions with < 2 usable trials | would be dropped (the validator needs 2 trials per session) — **none occur**, all 207 sessions are kept |
| Head tracked a centimetre past a partition wall | 152 samples (0.006%) snapped to the nearest open bin, mirroring the reference's `temp_maps` cleaning. Found by Check 3 in the first full run, which failed on 8 of 480 blocked partitions; fixed and re-run |
| Animal QLAK-CA1-51 has 21 days, not 31 | handled generically (`n_days` read from the file) |

### Issues Found and Resolved
- **Iteration 1.** First full conversion: sanity Check 3 failed — 4 sessions had 1-120 timepoints in a
  blocked partition (127 of 2,569,722 samples). Traced to DeepLabCut tracking the head slightly past a
  25 cm partition wall in the raw data (e.g. `(31.4, 25.1)` cm in QLAK-CA1-51 day 4), plus one case created
  by the 3-frame average of a boundary crossing. **Fix**: `snap_to_open_bins`, the analogue of the
  reference's snap-to-`temp_maps` cleaning in `decode_position_within`. Re-ran the sample conversion,
  sample verification, the full conversion and **all** Step-10 checks: all pass.
- No other issues found.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, NVIDIA L4).

### Training Progress
- Loss decreasing: **Yes**, monotonically for all 200 epochs: 2.2679 (ep 1) -> 1.2028 (ep 100) ->
  0.9499 (ep 200). Test loss 0.9526, essentially equal to the training loss.

### Decoder Results (Full)
| Output | Classes | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|---|---|-------------|--------|-------|
| position_bin | 9 | 0.1111 | 0.7677 | **0.6759** | 6.08x chance; train/val ratio 1.14 |

`predictions.png` shows the decoded bin (dashed) tracking the true bin (solid) closely across four
randomly chosen held-out trials.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Output | Validation balanced acc | Chance (1/9) | Ratio |
|---|---|---|---|
| position_bin | 0.6759 | 0.1111 | **6.08x** |

Far above chance and above the 1.5x threshold; no indication of a conversion bug.

### Check 2: Accuracy comparison to the paper
The paper reports position decoding as **Euclidean error in cm on a 15 x 15 grid**, never as accuracy and
never on a 3 x 3 grid, so a like-for-like number does not exist. Two bridging comparisons were therefore
constructed:

| Comparison | Source | Value |
|---|---|---|
| Paper Figure 1F mean decoding error (15x15) | `precomputed_results/within_decoding` | 13.49 cm (per-session range 6.53-31.01 cm) |
| Reference decoder re-run on the raw data (15x15) | `/app/reference_check.py`, 12 sessions | **identical to the published values, diff = 0.000 cm** |
| **Reference decoder (the paper's GaussianNB) re-run at n_bins=3** | `/app/reference_check.py` | balanced accuracy 0.548 (QLAK-CA1-08), 0.532 (QLAK-CA1-50), 0.470 (QLAK-CA1-51) |
| Paper's GaussianNB applied to **this converted dataset**, 5-fold within session | 15 random sessions | 0.555 |
| Per-session PCA(100) + logistic regression on this converted dataset | 15 random sessions | 0.625 |
| **Provided decoder on the full converted dataset** | `train_decoder_full_out.txt` | **0.676** |

The provided decoder **exceeds** the paper's own method on the same data (0.676 vs 0.555), as expected
since it pools all 207 sessions to learn a shared read-out rather than fitting each session alone. There
is no accuracy shortfall to explain. A 13.5 cm mean error on the 5 cm grid corresponds to roughly half a
25 cm bin, which is quantitatively consistent with ~68% correct on the 3 x 3 grid.

### Check 3: Train vs validation gap
Training 0.7677 vs validation 0.6759, ratio **1.14** (threshold 1.5). No overfitting or leakage. Trials are
split by whole 1-minute blocks, and the 100 ms bins within a block are never split across train and test,
so there is no within-trial leakage. The residual gap is the expected generalisation gap of a
session-specific 100-dimensional projection.

### Additional checks run
- **Output has enough variation**: the 9 classes hold 7.0%-17.1% of timepoints each; no class dominates.
- **Temporal alignment**: `processing_*.png` panel 5 overlays the continuous `x/25`, `y/25` traces on the
  integer output bin; the bin steps exactly at the crossings, no lag. Independently, rebuilding rate maps
  from raw `position` + `trace` reproduces the dataset's stored maps at r = 1.000000.
- **Neural filtering matches the reference**: verified by the exact reproduction of Figure 1F.
- **Per-session accuracy is not driven by a few sessions**: the 15-session spot check spans 0.37-0.83
  balanced accuracy, tracking cell count and recording day exactly as the paper's Figure 1F does
  (early days worse, later days better).

### Issues Found and Resolved
- None in this step. (The one issue in the whole conversion — samples tracked inside a partition wall — was
  found and fixed in Step 10.)

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — dataset description, how to load and use the data, output format
      specification, key statistics, file inventory
- [x] `CONVERSION_NOTES.md` complete — all decisions and their rationale, every correctness check,
      validation tables, and the one issue found and how it was resolved
- [x] `cache/` folder created with the investigation/verification scripts and their logs
      (`sanity_checks.py`, `sanity_checks_out.txt`, `reference_check.py`) and `cache/README_CACHE.md`
- [x] Directory cleaned (`__pycache__` removed)

### Final file inventory
| File | Purpose |
|---|---|
| `convert_data.py` | conversion script (`--full` / `--sample` / `--show-processing`) |
| `converted_data.pkl` | full converted dataset, 3.44 GB |
| `sample_data.pkl` | 2-session sample, 21 MB |
| `CONVERSION_NOTES.md` | this file |
| `README.md` | user-facing documentation |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | format-verification logs |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_QLAK-CA1-08_day05.png`, `processing_QLAK-CA1-30_day05.png` | per-step diagnostic figures |
| `sample_trials.png`, `predictions.png` | decoder input/output and held-out predictions |
| `cache/` | verification scripts + logs, documented in `cache/README_CACHE.md` |
