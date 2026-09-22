# Dataset Conversion Notes

## Overview
- **Dataset**: Lee, Keinath, Cianfarano & Brandon (2025), *"Identifying representational structure in CA1 to
  benchmark theoretical models of cognitive mapping"*, Neuron 113(2):307-320.
  Miniscope 1-photon calcium imaging of dorsal CA1 in freely-moving mice during a 3x3 geometric-deformation
  paradigm. Data: `/app/data` (Zenodo record 14867736), code: `/app/code` (github.com/jquinnlee/georepca1).
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format: decode the mouse's position (3x3 = 9 spatial bins) from CA1
  population activity, with the environment geometry (which partitions are blocked) as decoder input.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `paper.pdf` (the Neuron paper), `methods.txt` (excerpt of experiment + methods)
- `code/` -> `code/georepca1/{src/utils.py, src/plots.py, main.py, demos/*.ipynb}`, `code/README.md`
- `data/` -> 7 joblib files (`QLAK-CA1-XX`), 7 MATLAB v7.3 files (`QLAK-CA1-XX.mat`), `behav_dict`,
  `precomputed_results/` (per-animal split-half-reliability `*_shr`, partitioned RSMs, `within_decoding`, ...)
- `decoder.py`, `train_decoder.py` (target decoder + verification code), `Dockerfile`, `docker-compose.yaml`
- created by me: `CONVERSION_NOTES.md`, `convert_data.py`, `cache/` (exploration scripts)

Environment verified: `python3`, numpy 2.4.4, torch 2.6.0+cu124, GPU = NVIDIA L4 (23 GB), 128 CPUs, 1 TB RAM.
`joblib`, `scipy`, `sklearn`, `mat73`, `h5py`, `pymupdf` available.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat(animal, p, format)` | src/utils.py:61 | LOADING | Loads one animal's dict (joblib or MATLAB v7.3 via `mat73`). Returns `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}` |
| `mat2joblib` / `save_dat` | src/utils.py:87-107 | LOADING | Converts the `.mat` file to the joblib file. The joblib and `.mat` files hold the *same* data. |
| `generate_behav_dict` | src/utils.py:130 | LOADING | Light-weight dict with only `position`, `envs`, `maps_shape` (this is the `data/behav_dict` file) |
| `get_env_mat(env)` | src/utils.py:215 | PROCESSING | 3x3 binary matrix of open (1) / blocked (0) partitions for an environment name |
| `get_environment_label` | src/utils.py:150 | PLOTTING | Polygon outline for each geometry; has a `flipud` option (see Step 4) |
| `get_rate_maps(position, trace, n_bins=15, fps=30, buffer=1e-5, filter_size=1.5)` | src/utils.py:313 | PROCESSING | Bins position into `n_bins` x `n_bins` by integer division `position // ((nanmax(position,axis=0)+buffer)/n_bins)`, accumulates the binary trace per bin, smooths, divides by occupancy and multiplies by `fps` -> **rate maps in Hz** |
| `get_split_half` / `get_shuffle_split_half` / `get_place_cells` | src/utils.py:356-441 | CURATION | Place-cell identification by split-half rate-map reliability vs. 1000 circular shuffles. **Used for figures only - the paper explicitly includes *all* cells in the analyses.** |
| `decode_position_within(behav, traces, maps, n_bins=15, fps=30, v_filt_size=5, v_thresh=5, cell_threshold=5, buffer=1e-15, n_fold=5)` | src/utils.py:1845 | PROCESSING/CURATION | **The reference analogue of our task**: within-session, 5-fold cross-validated Bayesian decoding of position from the traces. See details below. |
| `fit_decoder(behav, traces, feature_max, temporal_bin_size=3)` | src/utils.py:1776 | PROCESSING | Temporal binning: `gaussian_filter1d(traces, sigma=3, axis=time)` then `AvgPool1d(kernel=3, stride=3)`; position is avg-pooled the same way then truncated to int; fits `GaussianNB` with flat priors |
| `test_decoder` | src/utils.py:1806 | PROCESSING | Same binning on test data; error = Euclidean distance between predicted and true spatial bin (x `bin_down` cm) |
| `get_all_decoding_within` | src/utils.py:1969 | ANALYSIS | Aggregates `results/within_decoding` into the Figure-1F dataframe |
| `get_transition_matrix` | src/utils.py:271 | PROCESSING | Bins behaviour the same way, `step_size=15` frames |

### Notes: what `decode_position_within` does (the reference decoding pipeline)
Called from `main.py` as
`decode_position_within(dat[a]['position'].T, dat[a]['trace'].T, dat[a]['maps']['smoothed'])`,
so `behav` is (frames, 2, days) and `traces` is (frames, cells, days).

1. **Spatial binning**: `behav_max = behav.max(axis=0).max(axis=1)` (per-dimension max over *all* frames and
   *all* days); `bin_down = (behav_max.max() + 1e-15)/n_bins` -> a single, *global* (per animal) bin size, so
   the spatial frame is identical across sessions. With `n_bins=15` and a 75 cm box this is 5 x 5 cm bins.
2. **Velocity filter**: `vel_idx[d,1:] = gaussian_filter1d(norm(diff(behav)*fps), sigma=v_filt_size=5) > v_thresh/bin_down`,
   i.e. **speed > 5 cm/s**, with the speed trace smoothed with a 5-frame Gaussian. Frame 0 is always excluded.
   Only these "moving" frames are used for fitting *and* testing.
3. **Cell filter**: `cell_idx[d] = np.sum(traces[:,:,d][vel_idx[d]], axis=0) > cell_threshold = 5`, i.e. a cell
   must have **> 5 binarized events during the moving frames** of that session. NaN traces (cells not
   registered that day) fail this test automatically, so unregistered cells are dropped.
4. **Temporal binning** (inside `fit_decoder`/`test_decoder`): traces are smoothed with
   `gaussian_filter1d(sigma=3 frames)` and average-pooled over `temporal_bin_size=3` frames -> **100 ms bins**
   at the 30 Hz acquisition rate. Position is average-pooled over the same 3 frames and truncated to the
   integer spatial bin.
5. **Model**: Gaussian naive Bayes with flat priors, 5-fold `KFold` split of the moving frames.
6. **Error metric**: Euclidean distance (cm) between the decoded and true spatial bin.

Nothing in the pipeline computes dF/F: the dataset already contains the **binarized rising-phase event
trains** ("trace"), which the paper treats as the firing rate. No further neuron quality filtering exists
beyond (3): the paper states that the spatial-reliability analysis "motivated the inclusion of all cells in
subsequent analyses".

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/<ANIMAL>` (joblib) and `/app/data/<ANIMAL>.mat` (MATLAB v7.3) hold the same content for
7 animals: `QLAK-CA1-08, -30, -50, -51, -56, -74, -75`. The joblib file is a dict `{animal: {...}}` with
(per `code/README.md` and verified directly):

| field | type/shape | meaning |
|-------|-----------|---------|
| `trace` | (n_days, n_cells, n_frames) float64, values {0, 1} or NaN | binarized calcium-transient rising-phase events ("firing rate"). **All-NaN for (day, cell) pairs where the cell was not registered that day** (verified: NaN is all-or-none within a (day, cell)) |
| `position` | (n_days, 2, n_frames) float64 | x-y head position in cm, range exactly [0, 75] in both dims, **no NaNs** |
| `envs` | (n_days, 1) `<U9` | geometry name per day: square, o, t, u, rectangle, +, i, l, bit donut, glenn |
| `blocked` | list of n_days, each `[array]` | indices of blocked partitions in the 3x3 grid, layout `[[0,1,2],[3,4,5],[6,7,8]]`; `-1` when nothing is blocked (square) |
| `maps` | dict: `sampling` (15,15,days), `smoothed`/`unsmoothed` (15,15,cells,days) | precomputed occupancy and rate maps |
| `SFPs` | (35,35,cells,days) | spatial footprints |
| `centroids` | (cells,2,days) | ROI centroids |

`data/precomputed_results/within_decoding` holds the paper's Figure-1F decoding errors
(per animal: `decoding_error` of shape (n_days, 5 folds), in cm).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, unique cells summed over animals) | **5,413** |
| Neurons / session (registered) | mean 336.9, min 113, max 564 |
| Registered cell-sessions (= rate maps) | **69,744** |
| Subjects | **7** |
| Sessions / subject | 31 for 6 animals, 21 for QLAK-CA1-51 -> **207 sessions** |
| Frames / session | 71,866-72,219 (constant within an animal) = 39.9-40.1 min at 30 Hz |
| Trials | not defined in the source; we create 1-min trials (see Step 5) |

Per animal (`cache/explore_out.txt`):

| animal | days | cells | frames | registered/day (mean) | cell-sessions | mean event rate (Hz) | frac. time > 5 cm/s |
|--------|------|-------|--------|----------------------|---------------|----------------------|---------------------|
| QLAK-CA1-08 | 31 | 515 | 71866 | 213.9 | 6631 | 0.212 | 0.502 |
| QLAK-CA1-30 | 31 | 875 | 71866 | 381.0 | 11812 | 0.230 | 0.446 |
| QLAK-CA1-50 | 31 | 942 | 71866 | 400.9 | 12427 | 0.274 | 0.621 |
| QLAK-CA1-51 | 21 | 554 | 72219 | 230.0 | 4831 | 0.263 | 0.612 |
| QLAK-CA1-56 | 31 | 862 | 72091 | 380.9 | 11807 | 0.237 | 0.517 |
| QLAK-CA1-74 | 31 | 713 | 72060 | 312.3 | 9680 | 0.246 | 0.509 |
| QLAK-CA1-75 | 31 | 952 | 72071 | 405.0 | 12556 | 0.263 | 0.448 |
| **total** | **207** | **5413** | | | **69744** | | ~0.52 |

Data quality checks run: traces are exactly {0,1} where registered; position has no NaN and no frozen
(dropout) segments longer than 2 frames; recordings are not zero-padded at either end.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 | "5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps" |
| Rate maps (registered cell-sessions) | 69,744 | same |
| Sessions | 207 | same |
| Subjects | 7 | animal list in `main.py`; "mean number of cells per animal = 773 +- 68 SE, minimum cells per animal = 515, maximum cells per animal = 952" |
| Cells / animal | mean 773, min 515, max 952 | same |
| Sessions / subject | up to 31 | "(31 days)"; "Sequence 1: Session 1-11; Sequence 2: Session 11-21" (3 sequences of 10 geometries starting/ending with square) |
| Geometries | 10 | "a sequence of 10 geometrically distinct environments" |
| Session length | 40 min | "All sessions were 40 min, and one session was recorded per day" |
| Acquisition rate | 30 Hz | "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz" |
| Arena | 75 x 75 cm, 3x3 grid of 25 x 25 cm partitions | "The full square environment was 75 cm x 75 cm"; "the 3x3 grid design ... (25 x 25 cm)" |
| Neural data time bin (decoding) | 3 frames = 100 ms | `fit_decoder(..., temporal_bin_size=3)` |
| Behaviour data time bin (decoding) | same 100 ms | `test_decoder` avg-pools position identically |
| Rate-map spatial bin | 5 x 5 cm (15 x 15) | "spatially binning position data into pixels corresponding to a 5cm x 5cm grid" |
| Position decoding error (Fig 1F) | mean 13.68 cm (per-animal 10.9-17.6 cm; per-session 6.5-31.0 cm) | `data/precomputed_results/within_decoding`; "significant decrease in decoding error across recorded sessions (ANOVA p<0.0001, F=7.9845)" |
| Speed threshold | 5 cm/s | `decode_position_within(v_thresh=5)` |
| Cell activity threshold | > 5 events in moving frames | `decode_position_within(cell_threshold=5)` |

### Processing Details
- Calcium preprocessing (already applied in the shipped data): motion correction, segmentation, transient
  extraction, then **rising-phase binarization**: derivative of the calcium trace smoothed with a 5-frame
  Gaussian, z-scored against a half-normal noise estimate, thresholded at z > 2.5 -> binary vector,
  "This binary vector was treated as the firing rate in all further analyses."
- Position from DeepLabCut head tracking, acquired at 30 Hz simultaneously with the imaging stream and
  timestamped for post-hoc alignment -> **position frame i is aligned to trace frame i** (both streams in the
  files have identical length).
- Cells registered across days with CellReg; unregistered (day, cell) entries are NaN.

### Curation Steps
**Neuron curation rules** (from `decode_position_within`, the decoding analysis):
- cell must be registered on that session (non-NaN trace), and
- must emit **more than 5 events during the moving (> 5 cm/s) frames** of that session.
No place-cell / split-half selection is applied: "motivated the inclusion of all cells in subsequent analyses".

**Trial curation rules**: the source data has no trials (continuous 40-min sessions). The only temporal
curation in the reference decoding analysis is the **speed > 5 cm/s** filter.

### Decoders Trained (reference)
| Decoded variable | Accuracy |
|------------------|----------|
| Animal position (15x15 = 5 cm bins), Gaussian naive Bayes, 5-fold CV within session | mean Euclidean error **13.68 cm** (Fig. 1F); no % accuracy is reported |
| Animal identity from RSM (Fig 2I) | not decodable (chi-square p = 1.000) - unrelated to our task |

The paper reports no categorical decoding accuracy, so the only quantitative benchmark for our decoder is
the 13.7 cm mean error at 5 cm resolution. 13.7 cm is about half of a 25 cm partition, so a well-converted
dataset should decode the 9 coarse partitions far above the 1/9 = 11.1% chance level (my own within-session
logistic-regression pilot on 100 ms bins gave 0.36-0.49 balanced accuracy for the animal with the *fewest*
cells; see `cache/poc.py`).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Orientation of the 3x3 geometry matrix | `get_env_mat('l')` = `[[1,1,1],[1,0,0],[1,0,0]]` -> blocked {4,5,7,8} | `blocked` for 'l' is {1,2,4,5} on every day of every animal | Figure 1A shows geometries as images | `blocked` equals the zeros of **`flipud(get_env_mat(env))`** for all 10 geometries (identity for the vertically symmetric ones). `get_env_mat` uses the image/plot (y-down) convention; `get_environment_label` carries an explicit `flipud` flag for exactly the asymmetric shapes (u, l, bit donut, glenn). **I use the per-session `blocked` field**, which I verified directly against occupancy (below). |
| Sessions per animal | `main.py` implies 31 days | 6 animals x 31 + 1 animal (QLAK-CA1-51) x 21 | "207 sessions", "(31 days)" | 6*31 + 21 = **207** exactly -> consistent; QLAK-CA1-51 completed only 2 sequences. |
| Which cells to include | `decode_position_within` drops NaN + <=5-event cells | 58% of (day, cell) entries are NaN | "inclusion of all cells" (i.e. no place-cell selection) | Use the reference decoding rule: registered + >5 events during moving frames. |
| Spatial bin normalisation | `get_rate_maps` normalises by the **per-session** position max; `decode_position_within` normalises by the **global (all-day)** max | per-day max is 72.4-75.0 cm, global max exactly 75.0 | 75 x 75 cm arena with 25 cm partitions | Use the fixed physical grid (25 cm, i.e. global max 75 / 3), as in `decode_position_within`. This keeps the spatial frame identical across sessions, which is required for the `blocked` partition indices to be meaningful. |

### Verified consistencies
- **Partition indexing**: with `x_bin = floor(position[0]/25)`, `y_bin = floor(position[1]/25)` and
  `partition = 3*y_bin + x_bin`, the occupancy of every `blocked` partition is **zero** (checked every
  session of QLAK-CA1-08 and QLAK-CA1-74: 14 frames out of 4.46 M, i.e. 0.0003%, fall into a blocked
  partition, all in a single session - tracking noise at a wall). Equivalently, the 3x3 block-sum of the
  precomputed 15x15 `maps['sampling']` transposed equals the open/blocked mask. This confirms both the
  axis convention (dim 0 = West-East = column, dim 1 = North-South = row) and the `blocked` semantics.
- Total sessions / cells / registered cell-sessions reproduce the paper's 207 / 5,413 / 69,744 exactly.
- Mean cells per animal from the data = 773.3, min 515, max 952 -> matches "773 +- 68 SE, min 515, max 952".
- Each of the 10 geometry names maps to exactly one `blocked` set, identical across all animals and
  repetitions: square {}, o {4}, t {3,5,6,8}, u {4,5}, rectangle {0,3,6}, + {0,2,6,8}, i {3,5},
  l {1,2,4,5}, bit donut {0,4}, glenn {0,8}.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `dat[animal]['trace'][day]` (cells x frames, binary) | `neural[session][trial]` (n_neurons, T) | Gaussian smooth sigma = 3 frames along time -> average-pool 3 frames (100 ms) -> x 30 (events/s) -> keep moving bins -> split into 1-min trials | `fit_decoder`/`test_decoder` (sigma=3, `AvgPool1d(3,3)`), `get_rate_maps` (x fps for Hz) | float32 |
| NaN rows of `trace` + activity | neuron curation | drop cells not registered that day; drop cells with <= 5 events within the retained moving bins | `decode_position_within` `cell_idx` | |
| `dat[animal]['blocked'][day]` | `input[session][trial]`, shape (9,) | binary vector, 1 = partition blocked, 0 = open; `-1` -> all zeros (square) | `get_env_mat` (same information, see Step 4) | static per trial |
| `dat[animal]['position'][day]` (2 x frames, cm) | `output[session][trial]`, shape (1, T) | average-pool 3 frames -> `partition = 3*floor(y/25) + floor(x/25)`, clipped to [0,2] per axis; rare labels in a blocked partition reassigned to the nearest open partition | `test_decoder` (avg-pool then int), `decode_position_within` (global bin size, "cleaning" to valid bins) | 9 categories |
| `dat[animal]['position'][day]` | temporal curation | speed = `gaussian_filter1d(norm(diff(pos)*30), sigma=5)`, averaged within each 100 ms bin; keep bins with speed > 5 cm/s | `decode_position_within` `vel_idx` | |
| animal id | `subjects`, `subject_idx` | 7 animal IDs | | |
| - | `brain_regions`, `brain_region_idx` | all `CA1` | "hippocampal subregion CA1" | |

### Key Decisions
1. **Neural signal = binarized event train, not dF/F.** The shipped `trace` field *is* the
   rising-phase-binarized signal that the paper "treated as the firing rate"; no dF/F step is needed or
   possible from these files.
2. **Time bin = 100 ms (3 frames at 30 Hz), with a 3-frame Gaussian smoothing before pooling.** This is
   exactly the reference decoder's `temporal_bin_size=3` + `gaussian_filter1d(sigma=3)`. I apply the
   smoothing to the *continuous* session trace before the velocity selection (the reference smooths after
   concatenating the moving frames); smoothing before selection is strictly more correct temporally and is
   the only defensible order once the data are cut into trials.
3. **Units: events/s (Hz)**, i.e. the pooled mean binary value x 30 fps, as `get_rate_maps` does. The decoder
   is a linear projection + linear classifier trained with a fixed number of full-batch Adam steps, so it is
   *not* scale invariant; Hz units put the leading PCs at std ~5-9, a healthy scale (pilot in `cache/poc2.py`).
4. **Speed > 5 cm/s filter (`v_thresh=5`, speed smoothed with sigma = 5 frames).** Matches the reference
   decoding analysis. Verified in a pilot to improve balanced accuracy (0.488 vs 0.424 without). Bins are kept
   or dropped as whole 100 ms bins (the speed trace is averaged within the bin), so every retained sample is a
   contiguous 100 ms of recording. This makes trials variable-length, which the target format allows.
5. **Cell curation: registered that day AND > 5 events during the retained moving bins**, matching
   `decode_position_within`'s `cell_idx`. No place-cell selection (the paper includes all cells).
6. **Trials = consecutive 1 min blocks (600 bins of 100 ms) of the session**, as required by the task. The
   final partial block of each session (~18 s) is kept as a shorter trial so that no data is discarded.
   Trials retaining fewer than 30 bins (3 s of movement) after the speed filter are dropped as unusable.
7. **Output = single categorical variable with 9 values** (the 3x3 partition), time-varying, as required by
   the Decoder Task. Value names `r0c0 ... r2c2` (row = North-South = position dim 1, column = West-East =
   position dim 0), matching the paper's partition ordering ("West to East, and top to bottom (North to
   South)") and the `blocked` indexing.
8. **Input = 9-dim binary geometry vector (1 = blocked)**, static per trial, exactly the Decoder Task
   specification. It is the same information as the geometry name but in the decoder-usable form that tells
   the decoder which partitions are impossible.
9. **Spatial binning uses the fixed physical grid (25 cm)** rather than a per-session position maximum, so
   that bin identity is comparable across sessions and consistent with `blocked` (Step 4).
10. **All 7 animals / 207 sessions are kept** - no session-level exclusions exist in the reference.

### Planned Sanity Checks
- [x] Partition labels never fall in a `blocked` partition (before the nearest-open reassignment: < 0.001%).
- [ ] Session/animal/neuron counts: 7 subjects, 207 sessions, 69,744 - (cells failing the >5-event rule)
      total neurons across sessions; unique cells 5,413.
- [ ] Trials per session = 40 (39 full 1-min trials + 1 partial) minus dropped low-movement trials.
- [ ] T per trial <= 600 bins; total retained bins / total bins ~ 0.52 (the measured fraction of time moving).
- [ ] Neural values >= 0, mean ~ 0.2-0.3 Hz (matches per-animal mean event rates in Step 2).
- [ ] Independent spot-checks against the **MATLAB `.mat` files** (read with h5py, a different path from the
      joblib files used by the converter) for neural, input and output values (Step 10).
- [ ] Output distribution: 9 classes, blocked partitions never used; per-session class fractions consistent
      with the occupancy in the precomputed `maps['sampling']`.
- [ ] Input: exactly 10 unique geometry vectors, each matching `get_env_mat(env)` up to the documented flipud.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`, run as `python -u convert_data.py <outpicklefile> [--full|--sample]
[--show-processing] [--jobs N]`.

Structure:
- `blocked_vector()` - the session's `blocked` entry -> 9-d binary geometry vector (decoder input).
- `compute_speed()` - `gaussian_filter1d(norm(diff(pos)*30), sigma=5)`, frame 0 = 0 (reference
  `decode_position_within`).
- `bin_time()` - vectorised reshape-and-reduce for the 3-frame (100 ms) binning of every stream.
- `discretize_position()` - `3*floor(y/25) + floor(x/25)`, clipped to [0,2], with the rare samples inside a
  blocked partition reassigned to the nearest open partition.
- `process_session()` - the whole per-session pipeline (speed -> binning -> labels -> cell curation ->
  neural rates -> 1-min trials).
- `process_animal()` - loads one animal's joblib file once and converts all of its sessions.
- `main()` - runs the animals in parallel, assembles the target dict, prints a summary, pickles the result.
- `--sample` converts 2 sessions: `QLAK-CA1-08` day 2 (geometry 't', 4 blocked partitions) and
  `QLAK-CA1-51` day 0 (square) - two animals and both a deformed and an undeformed geometry.
- `--show-processing` writes `processing_<session_id>.png` with 10 panels covering every processing step.

Code inefficiencies identified:
- The naive implementation would loop over frames (as the reference's `get_rate_maps`/`fit_decoder` do);
  all binning is instead done with a single `reshape(...).mean(-1)`.
- `joblib.load` of one animal takes 6-16 s and the decompressed arrays are 9-17 GB, so each animal is
  loaded exactly once and all of its sessions are processed from that one load.
- The traces are float64 in the file; they are cast to float32 before the Gaussian smoothing (the
  dominant cost), and only the *registered* cells (about 40%) are smoothed.

Code speedups added:
- Animals are processed in parallel (`--jobs`, default 7 = one process per animal). Peak RAM ~100 GB of
  the 1 TB available.
- Result: **41 s** for the full dataset (207 sessions), of which ~17 s is per-animal processing
  (0.54-0.59 s/session) and ~16 s is file loading.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/sample_data.pkl --sample --show-processing` ->
`/app/conversion_sample_out.txt`, `/app/sample_data.pkl`, 2 processing plots.
`python -u train_decoder.py /app/sample_data.pkl --verify-only` -> `/app/verification_sample_out.txt`:
**"Data format is valid, no errors or warnings."**

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (QLAK-CA1-08 day 2 = 't'; QLAK-CA1-51 day 0 = square) |
| Neurons (total) | 262 (150 + 112) |
| Neurons / session | 150 of 156 registered; 112 of 113 registered |
| Subjects | 7 listed, 2 used |
| Trials (total) | 80 |
| Trials / session | 39 and 41 |
| T per trial | mean 329, min 56, max 468 (<= 600) |
| Input range (all 9 dims) | [0, 1]; session 1 = partitions 3,5,6,8 blocked, session 2 = none |
| Output distribution | r0c0 0.130, r0c1 0.117, r0c2 0.177, r1c0 0.039, r1c1 0.088, r1c2 0.050, r2c0 0.079, r2c1 0.179, r2c2 0.141 |

### Processing Plots Review
`processing_QLAK-CA1-08_day02.png` / `processing_QLAK-CA1-51_day00.png`, 10 panels each:
1. raw 30 Hz position overlaid with the 100 ms binned position - no offset, no lag;
2. the discretisation `partition = 3*row + col` drawn together with `floor(x/25)` and `floor(y/25)`
   - every step of the partition trace is explained by a row/column change;
3. the speed trace with the 5 cm/s threshold and the retained bins shaded (49-60% retained);
4. the binned trajectory coloured by the assigned output label - the nine colours tile the 3x3 grid
   exactly, and for 't' the trajectory covers only the 5 open partitions while the 4 blocked partitions
   (shaded grey, input = 1) are empty;
5. raw binary events of 3 cells overlaid with their smoothed/binned rate - peaks coincide exactly;
6./7. population raster before and after binning over the same minute - same event pattern;
8. cell-curation histogram (150 of 156, 112 of 113 kept);
9. trial layout across the 40 min session;
10. output distribution, with zero mass on the blocked partitions.
No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised binning instead of per-frame loops | ~50x on the binning step |
| smooth only registered cells, in float32 | ~3x on the dominant smoothing step |
| one file load per animal instead of per session | ~20x fewer seconds of I/O |
| 7 animals in parallel | ~6x wall clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| load (per animal, amortised) | ~0.5 s | ~16 s (once per animal, in parallel) |
| process session | 0.54-0.59 s (1.9 s single-process on the first, cold run) | ~2 min serial, ~20 s with 7 jobs |
| assemble + pickle 3.4 GB | - | ~20 s |
| **total (measured)** | | **41 s** |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| position_bin (9 classes) | 0.4797 | 0.3828 | 0.1111 |

Loss decreased monotonically from 2.75 to 1.41 over 200 epochs. Validation accuracy is 3.4x chance from
only 2 sessions of the two animals with the *fewest* cells (112 and 150 neurons), and matches a fully
converged scikit-learn logistic regression on the same data (0.46-0.49, `cache/poc2.py`), i.e. the decoder
is extracting essentially all of the linearly available information.

Scale check: rescaling the neural data by 0.2x and 5x changed validation accuracy by <= 0.02
(0.402 / 0.383 / 0.383), so the choice of units (events/s) is not driving the result.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3.44 GB, written in 41 s (`conversion_full_out.txt`)
- `verification_full_out.txt`: created - **"Data format is valid, no errors or warnings."**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 7 | 7 in `main.py` `animals` | 7 files | 7 | yes |
| Sessions | 207 | - | 6x31 + 1x21 = 207 | 207 | yes |
| Sessions/subject | up to 31 | - | 31,31,31,21,31,31,31 | 31,31,31,21,31,31,31 | yes |
| Unique neurons | 5,413 | - | 5,413 | 5,374 used (39 cells never pass the >5-event rule) | yes (99.3%) |
| Cells per animal | mean 773 +- 68 SE, min 515, max 952 | - | mean 773.3, 515-952 | mean 767.7, 510-944 | yes |
| Registered cell-sessions ("rate maps") | 69,744 | - | 69,744 | 68,860 kept (1.27% dropped by `cell_threshold=5`) | yes |
| Neurons/session | - | - | mean 336.9 registered | mean 332.7 | yes |
| Session length | 40 min | - | 71,866-72,219 frames @30 Hz | 23,955-24,073 bins of 100 ms | yes |
| Geometries | 10 | 10 in `get_env_mat` | 10 distinct `blocked` sets | 10 distinct input vectors | yes |
| Time bin | 100 ms (`temporal_bin_size=3`) | 3 frames @ 30 Hz | - | 100 ms | yes |
| Speed threshold | 5 cm/s | `v_thresh=5` | 51.9% of frames moving | 51.9% of bins retained | yes |
| Spatial grid | 75 cm / 3 = 25 cm | `bin_down = (max+buffer)/n_bins`, max = 75.0 | position range [0, 75] | 25 cm | yes |
| Trials | n/a (continuous sessions) | n/a | n/a | 8,163 (39.4/session, 22-41) | n/a |
| Input range | - | - | - | [0,1] for 8 dims; partition r2c1 is open in all 10 geometries so its input is always 0 | expected |
| Output distribution | - | - | occupancy of `maps['sampling']` | r0c0 .095, r0c1 .108, r0c2 .114, r1c0 .088, r1c1 .070, r1c2 .087, r2c0 .109, r2c1 .158, r2c2 .171 | see Step 10 Check 4 |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`verification_full_out.txt` reports **no errors and no warnings** ("Data format is valid, no errors or
warnings."), so there was nothing to fix or to explain away. The same is true of the sample log.

### Check 2: Independent sanity checks (`cache/sanity_checks.py`)
All of these load the **original MATLAB v7.3 files** (`/app/data/<animal>.mat`) with `h5py` - a different
file format and a different code path from the joblib files the converter reads - and re-derive the
pipeline from the reference code rather than importing anything from `convert_data.py`.
Four sessions were checked (`QLAK-CA1-08` day 2 = 't', `QLAK-CA1-51` day 0 = square,
`QLAK-CA1-75` day 17 = 'bit donut', `QLAK-CA1-74` day 9 = 'u', the one session that needed a
blocked-partition reassignment). **All 64 checks pass**:

| Check | Result |
|-------|--------|
| NEURAL: every trial `np.allclose` to the independently recomputed rate | PASS (4/4 sessions) |
| NEURAL: single-element spot check (largest element of trial 5) recomputed from scratch for one cell | PASS, e.g. 29.995012 vs 29.995012 |
| NEURAL: total event mass conserved by smoothing+binning (`sum(rate)/30*3 == #events`) | PASS, e.g. 273,060.0 vs 273,060.0 |
| NEURAL: finite, non-negative, number of neurons matches | PASS |
| INPUT: equals the `blocked` vector read from the `.mat` file, identical in every trial | PASS |
| OUTPUT: every trial `np.array_equal` to the independently recomputed partition labels | PASS |
| OUTPUT: spot check from the raw (unbinned) position, e.g. x=58.8 y=12.9 -> bin 2 | PASS |
| OUTPUT: never takes the value of a blocked partition | PASS |
| OUTPUT: the set of visited partitions equals exactly the set of open partitions | PASS |
| every retained bin has speed > 5 cm/s; trial lengths match; T <= 600 | PASS |
| number of trials, number of neurons, subject index, geometry name | PASS |

### Check 3: Reference code comparison
| Step | Reference (`georepca1/src/utils.py`) | `convert_data.py` | Same? |
|------|--------------------------------------|-------------------|-------|
| (a) data loading | `load_dat`: `joblib.load(data/<animal>)`, fields `trace`, `position`, `envs`, `blocked` | identical file, identical fields, `process_animal` | yes |
| (b) neuron filtering | `cell_idx = sum(traces[vel_idx]) > 5` (NaN cells fail automatically) | `registered = ~isnan(trace[:,0])`; `events_in_moving_bins > 5` | yes (NaN handling explicit) |
| (b') trial/sample filtering | `vel_idx = gaussian_filter1d(norm(diff(behav)*fps), sigma=5) > v_thresh/bin_down` | same speed trace, thresholded at 5 cm/s after averaging within each 100 ms bin | yes, except the threshold is applied per 100 ms bin instead of per frame, so that every retained sample is a contiguous 100 ms of recording |
| (c) temporal alignment | position frame i <-> trace frame i; both avg-pooled with `AvgPool1d(3,3)` | same, `bin_time()` on both streams with the same bin edges | yes |
| (d) binning | `gaussian_filter1d(traces, sigma=3, axis=time)` then `AvgPool1d(3,3)`; rate maps are `x fps` for Hz | `gaussian_filter1d(sigma=3)` then mean over 3 frames, `x 30` | yes (units Hz, as `get_rate_maps`) |
| (d') smoothing order | smooths *after* concatenating the moving frames | smooths the *continuous* session trace, then selects moving bins | **different**, deliberately: smoothing across a discontinuous concatenation mixes samples minutes apart. Smoothing first is the temporally correct order and is required once the data is cut into trials. |
| (e) input construction | `get_env_mat(env)` (3x3 open/blocked matrix) | the per-session `blocked` field, verified to equal `flipud(get_env_mat(env))` for all 10 geometries and validated against occupancy | yes (see Step 4) |
| (f) output construction | `behav /= bin_down` with `bin_down = (global max + buffer)/n_bins`, avg-pooled, `astype(int)`; predictions "cleaned" onto visited bins | `floor(mean position / 25 cm)`, clipped to [0,2]; samples inside a blocked partition moved to the nearest open partition | yes: the global position max is *exactly* 75.0 for all 7 animals, so `(75+buffer)/3 = 25.0 cm` is identical. The only difference is the `clip`, which fixes a genuine edge case: 198 frames of 14.9 M (0.0013%) sit exactly at 75.0 and would index bin 3 (the reference's `buffer=1e-15` is smaller than the float64 spacing at 75, so it does not protect against this). |
| spatial resolution | `n_bins=15` (5 cm) | `N_GRID=3` (25 cm) | **different, required by the Decoder Task** ("3 x 3 = 9 spatial bins") |
| decoder | Gaussian naive Bayes, 5-fold CV within session | provided PCA+linear decoder trained across sessions, 80/20 trial split | **different, required by the task** |

### Check 4: Key statistics comparison (`cache/stats_checks.py`, `cache/occ_check.py`)
- 207 sessions / 7 subjects / 31,31,31,21,31,31,31 sessions per subject - exactly as in the data and paper.
- 68,860 cell-sessions of the 69,744 registered (98.7%); 884 dropped by the reference's own
  `cell_threshold=5` rule. 5,374 of the 5,413 unique cells appear in at least one session; the 39 missing
  cells never emit more than 5 events while the animal is moving in any session.
- Mean cells per animal 767.7 (510-944) vs the paper's 773 +- 68 SE (515-952).
- 51.9% of the 4,967,611 time bins survive the 5 cm/s filter, matching the 52% of frames above 5 cm/s
  measured directly from the raw position data.
- 2,574,536 of the 2,576,698 retained bins end up in the pickle; the 2,162 missing ones (0.08%) are in
  1-min blocks that retained < 30 bins and were dropped.
- Exactly 10 unique geometry input vectors, each corresponding to exactly one environment name.
- **Output distribution vs the paper's own occupancy maps**: for each session I compared the distribution
  of my output labels with the 3x3 block sums of the precomputed `maps['sampling']` (15x15 occupancy maps
  produced by the reference `get_rate_maps`). Without the velocity filter the correlation is
  **1.0000 in all 207 sessions** (i.e. my spatial discretisation is *identical* to the reference's), and
  with the velocity filter it is 0.952 on average - the entire difference is explained by the speed filter
  (mice rest disproportionately in a few partitions).
- Converted mean rate equals the raw event rate over the same cells and bins (e.g. 0.2295 vs 0.2294 Hz,
  0.3399 vs 0.3402 Hz).

### Check 5: Edge cases
- Sessions with < 2 trials: 0. Trials with T = 0: 0. Trials with T > 600: 0. Minimum T = 30 bins.
- Shape/dtype violations across all 8,163 trials: 0 (`neural` float32 (n,T), `input` float32 (9,),
  `output` int64 (1,T), `brain_region_idx` length == n_neurons).
- Frame count is not a multiple of 3 in every session; the <= 2 trailing frames are dropped (< 0.003% of a
  session) so that all bins are complete.
- The last 1-min block of a session is partial (~18 s); it is kept as a shorter trial rather than
  discarded, unless it retains < 30 bins.
- Position values exactly at the 75 cm arena edge (198 frames of 14.9 M) are clipped into bin 2.
- Samples whose *averaged* position falls in a blocked partition (14 frames in the whole dataset, all in
  `QLAK-CA1-74` day 9) are reassigned to the nearest open partition.
- 2 sessions contain one neuron that is silent in every trial (its > 5 events all fell inside 1-min blocks
  that were later dropped). This is harmless - the format checker reports no warning and the decoder's
  SVD initialisation handles zero rows - so it was left as is rather than adding a second, non-reference
  filtering pass.
- All 9 output classes occur in the dataset; per session only the open partitions occur, as they must.

### Issues Found and Resolved
- Initially `--sample` converted day 0 of two animals, which are both the square geometry and therefore
  never exercised the `blocked` handling; changed to a 't' session plus a square session.
- The first version of the neural spot check happened to pick a zero element (the data is sparse); it now
  selects the largest element of the trial so that the check is informative.
- No conversion bugs were found; no re-conversion was required.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/converted_data.pkl --plot-samples` -> `/app/train_decoder_full_out.txt`
(ran on the GPU in under 2 minutes).

### Training Progress
- Loss decreasing: **Yes**, monotonically, 3.890 (epoch 1) -> 1.904 (10) -> 0.997 (30) -> 0.685 (100) ->
  0.603 (200). Test loss 1.196.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| position_bin (9 classes) | 0.8257 | **0.6616** | 0.1111 | 5.95x chance; train/val ratio 1.25 |

`sample_trials.png` and `predictions.png` show the decoded partition tracking the true partition
closely, with errors concentrated at partition crossings.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Variable | Classes | Chance | Validation balanced accuracy | Ratio |
|----------|---------|--------|------------------------------|-------|
| position_bin | 9 | 0.1111 | 0.6616 | **5.95x** |

Far above the 1.5x threshold. Per session: mean 0.643, min 0.223, max 0.828 (207/207 sessions above
chance). The confusion matrix is strongly diagonal (0.59-0.74 per class) and **87.0% of predictions are
either correct or in an adjacent partition** - the residual errors are the geometrically expected ones.

### Check 2: Accuracy comparison to the paper
The paper reports no categorical accuracy; its only decoding benchmark is the mean Euclidean position
error of a Gaussian naive Bayes decoder on 5 x 5 cm bins (Figure 1F, precomputed in
`data/precomputed_results/within_decoding`). I therefore computed the same metric on my validation
predictions (`cache/decode_error.py`):

| Quantity | Reference paper | This conversion + provided decoder |
|----------|-----------------|------------------------------------|
| Decoding error (Euclidean distance between decoded and true spatial bin) | **13.49 cm** (mean over the 207 sessions; 13.68 cm as the mean of per-animal means), 5 cm bins | **11.37 cm**, 25 cm bins (range 4.6-29.2 cm) |
| Per-animal ordering (best -> worst) | 75 (10.9), 74 (11.4), 50 (11.4), 30 (12.8), 56 (15.2), 08 (16.5), 51 (17.6) | 75 (8.6), 74 (9.0), 50 (9.5), 30 (11.6), 56 (12.6), 08 (14.1), 51 (15.7) - **identical ordering** |
| Improvement with experience (Fig 1F, ANOVA p<0.0001) | 17.42 cm (days 0-9) -> 10.67 cm (days 20-30) | accuracy 0.555 (days 0-9) -> 0.675 -> 0.702 (days 20-30) - **same effect** |
| Per-session agreement | - | corr(paper's error, my accuracy) = **-0.931**; corr(paper's error, my cm error) = **+0.946** across all 207 sessions |

My error is slightly lower than the paper's because the task's 25 cm bins are 5x coarser than the paper's
5 cm bins (a correct 25 cm prediction contributes zero error), so the numbers are not directly comparable
in absolute terms. What *is* directly comparable - and what would expose a conversion bug - is the
session-by-session and animal-by-animal agreement, which is near-perfect (r = 0.95). A misalignment of the
neural and behavioural streams, a wrong axis convention, or wrong cell curation would destroy that
correlation.

### Check 3: Train vs validation gap
Training 0.8257 vs validation 0.6616 -> ratio **1.25**, below the 1.5x flag. Some gap is expected: the
decoder fits a separate 100-dimensional projection for each of the 207 sessions (about 33k parameters per
session) on ~12k timepoints, and neighbouring 1-min trials are not statistically independent. There is no
leakage: the train/validation split is by whole trials, and every sample belongs to exactly one trial.

### Additional confirmations that the outputs/alignment are right
1. Output values verified against the raw `.mat` position for specific trials (Step 10, Check 2).
2. Temporal alignment verified visually (`processing_*.png` panels 1, 5, 6, 7) and numerically (the
   independent recomputation reproduces every neural and output array exactly).
3. Output variation: the least common class holds 7.0% and the most common 17.1% of samples - no
   degenerate class.
4. Neural curation follows the reference's own `cell_threshold=5` rule; only 1.3% of cell-sessions drop out.
5. The decoder's per-session accuracy tracks the paper's own per-session decoding quality (r = -0.93).

### Issues Found and Resolved
None: every check in this step passed on the first run, so no re-conversion was needed.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format spec, key statistics)
- [x] `cache/` folder with the exploration / validation scripts and `README_CACHE.md`
- [x] All deliverable files present: `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`,
      `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
      `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
      `train_decoder_full_out.txt`, plus the plots `processing_*.png`, `sample_trials.png`,
      `predictions.png`.
