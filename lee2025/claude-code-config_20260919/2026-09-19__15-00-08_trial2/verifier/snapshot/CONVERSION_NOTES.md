# Dataset Conversion Notes

## Overview
- **Dataset**: CA1 miniscope calcium imaging in a 3x3-partitioned geometric-deformation paradigm.
  Lee, Keinath, Cianfarano & Brandon (2025) *Identifying representational structure in CA1 to benchmark
  theoretical models of cognitive mapping*, Neuron 113(2):307-320. Data: Zenodo 10.5281/zenodo.13993254.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format — decode the mouse's position (3x3 = 9 spatial bins)
  from CA1 population activity, with the environment geometry (which partitions are blocked) as decoder input.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents (`/app`):
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `paper.pdf` (Lee et al. 2025, Neuron), `methods.txt` (excerpted methods)
- `code/` — reference code base (`georepca1`: `src/utils.py`, `src/plots.py`, `main.py`, `demos/*.ipynb`)
- `data/` — 7 animal files in joblib format (`QLAK-CA1-XX`) + identical MATLAB `.mat` versions,
  `behav_dict` (position/envs only), `precomputed_results/` (paper's saved analysis outputs)
- `decoder.py`, `train_decoder.py` — provided decoder/validation code
- `CONVERSION_NOTES.md` (this file)

Environment verified: `python3` with numpy 2.4.4, torch 2.6.0+cu124, scipy 1.18.0, h5py 3.16.0,
joblib, sklearn, matplotlib. Machine: 128 CPUs, ~1 TB RAM, NVIDIA L4 (23 GB).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat(animal, p, format)` | `src/utils.py:61` | LOADING | Loads per-animal dataset, either the MATLAB `.mat` (via `mat73.loadmat`) or the pre-converted `joblib` file. Returns `{animal: {SFPs, blocked, centroids, envs, maps, position, trace}}`. |
| `mat2joblib` / `save_dat` | `src/utils.py:87,100` | LOADING | Converts `.mat` → joblib. The joblib files shipped in `data/` are exactly this conversion, so loading either gives identical arrays. |
| `generate_behav_dict` | `src/utils.py:130` | LOADING | Builds the lighter `behav_dict` with only `position`, `envs`, `maps_shape`. |
| `get_env_mat(env)` | `src/utils.py:215` | PROCESSING | Maps the environment string name → 3x3 binary matrix, 0 = omitted/blocked partition. |
| `get_environment_label` | `src/utils.py:150` | PLOTTING | Polygon outline of each geometry (has a `flipud` option — orientation conventions differ between plotting and the `blocked` field). |
| `get_rate_maps(position, trace, n_bins=15, fps=30, buffer=1e-5, filter_size=1.5)` | `src/utils.py:313` | PROCESSING | Spatially bins position with `position // ((nanmax(position,axis=0)+buffer)/n_bins)`, accumulates `trace` into `rate_maps[:, x, y]`, divides by occupancy, ×fps. Confirms the **[x_bin, y_bin] indexing convention** of the rate maps. |
| `get_split_half` / `get_shuffle_split_half` / `get_place_cells` / `get_shr_within` | `src/utils.py:356-462` | CURATION (not used for decoding) | Place-cell identification via split-half rate-map reliability vs. 1000 circular shuffles. **Note: place-cell status is *not* used to select cells for the decoding analysis** (paper: "motivated the inclusion of all cells in subsequent analyses"). |
| `fit_decoder(behav, traces, feature_max, temporal_bin_size=3)` | `src/utils.py:1776` | PROCESSING | The paper's decoder front-end. Temporal binning: `gaussian_filter1d(traces, sigma=temporal_bin_size, axis=0)` then `AvgPool1d(kernel=stride=temporal_bin_size)`; position is average-pooled the same way and truncated to int. Position → one-hot class index via `empty_map[x, y]` flattened row-major, i.e. **class = x_bin * n_bins + y_bin**. Gaussian naive Bayes with flat priors. |
| `test_decoder` | `src/utils.py:1806` | PROCESSING | Same binning on test data; error = Euclidean distance between predicted and true bin centres × bin size (cm). |
| `decode_position_within(behav, traces, maps, n_bins=15, fps=30, v_filt_size=5, v_thresh=5, cell_threshold=5, buffer=1e-15, n_fold=5)` | `src/utils.py:1845` | CURATION + PROCESSING | **The paper's within-session position-decoding pipeline.** This is the single most relevant reference function. It: (1) computes `bin_down = (global max position over time, both axes and all days + buffer) / n_bins` → one isotropic spatial bin size shared by all sessions of an animal; (2) computes speed as `gaussian_filter1d(norm(diff(position)*fps), sigma=v_filt_size=5 frames)` and keeps only frames with **speed > 5 cm/s** (frame 0 always excluded); (3) keeps only cells with **> 5 events during the retained (moving) frames** (`cell_threshold=5`); (4) 5-fold cross-validation with `fit_decoder`/`test_decoder`. |
| `get_all_decoding_within` | `src/utils.py:1969` | ANALYSIS | Aggregates the saved `within_decoding` results (used for Figure 1F). |

### Notes
- **No dF/F computation is needed.** `trace` in the distributed dataset is already the final
  *rise-extracted binarized* signal (values are exactly {0, 1}, verified). The methods state this binary
  vector "was treated as the firing rate in all further analyses". So the neural stream for the decoder is
  this binary event train, temporally binned.
- Cells that were not registered on a given day are stored as **all-NaN** along the time axis for that day
  (verified: `np.isnan(trace).all(axis=time)` is identical to `np.isnan(trace[:, :, 0])`). These must be
  dropped per session.
- `blocked` is not used anywhere in the reference code (geometry is handled through the `envs` string names
  and `get_env_mat`), but it encodes exactly the decoder input we need.
- The reference decoding uses `n_bins=15` (5 cm bins). Our task specifies 3x3 = 9 bins, so `n_bins=3`
  (25 cm partitions) — this matches the physical 3x3 partition design of the apparatus.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` contains, for each of 7 animals (`QLAK-CA1-08/30/50/51/56/74/75`):
- `QLAK-CA1-XX` — joblib pickle, `{animal_id: {...}}` (used; loads in 9–23 s)
- `QLAK-CA1-XX.mat` — the identical data as MATLAB v7.3 (source of the joblib file)

Per-animal dictionary fields (n_days = sessions for that animal, T ≈ 72,000 frames @ 30 Hz = 40 min):

| Field | Shape / type | Meaning |
|---|---|---|
| `trace` | (n_days, n_cells, T) float64 | Binarized rising-phase calcium events, {0,1}; **NaN for the whole day if the cell was not registered that day** |
| `position` | (n_days, 2, T) float64 | x,y head position in **cm**, range exactly [0, 75]; no NaNs anywhere |
| `envs` | (n_days, 1) `<U9` | Geometry name per day: square, o, t, u, rectangle, +, i, l, bit donut, glenn |
| `blocked` | list of n_days | Indices of blocked partitions in the 3x3 grid laid out `[[0,1,2],[3,4,5],[6,7,8]]`; `-1` if none (square) |
| `maps` | dict: `sampling` (15,15,n_days), `smoothed`/`unsmoothed` (15,15,n_cells,n_days) | Pre-computed occupancy and event-rate maps |
| `SFPs` | (35,35,n_cells,n_days) | Spatial footprints |
| `centroids` | (n_cells,2,n_days) | Footprint centroids |

`data/precomputed_results/within_decoding` holds the paper's Figure-1F decoding errors
(per animal: `decoding_error` (n_days, 5 folds)).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects (mice) | 7 |
| Sessions (animal-days) | **207** (31,31,31,21,31,31,31) |
| Sessions / subject | 21–31 (mean 29.6) |
| Neurons (total, unique registered cells) | **5,413** (515, 875, 942, 554, 862, 713, 952) |
| Neurons / session (registered, non-NaN) | mean **337.0** (= 69,744 / 207) |
| Cell × session pairs (= rate maps) | **69,744** |
| Session duration | 71,866 – 72,219 frames @ 30 Hz = 39.93 – 40.12 min |
| Trials / session (1-min, complete) | 39 or 40 (see Step 5) |
| Mean event rate per cell | 0.21–0.27 Hz (median ≈ 0.19 Hz) |
| Fraction of frames with speed > 5 cm/s | 0.45–0.62 per animal (≈ 0.52 overall) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 | "5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps" |
| Sessions | 207 | same |
| Rate maps (cell × session) | 69,744 | same |
| Geometries | 10 | same |
| Subjects | 7 | animal list in `main.py`; "mean number of cells per animal = 773 ± 68 SE, minimum cells per animal = 515, maximum cells per animal = 952" (5,413/7 = 773.3) |
| Sessions / subject | up to 31 | "a repeating sequence of randomly ordered geometries across days ... for up to three total repetitions (31 days)" |
| Session duration | 40 min | "All sessions were 40 min, and one session was recorded per day" |
| Acquisition rate | 30 Hz | "The DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz" |
| Arena | 75 × 75 cm, 3 × 3 grid | "We partitioned an open square (75 × 75 cm) into a 3 × 3 grid space"; partition walls 25 cm tall |
| Neural signal | binary rising-phase vector | "The final binarized rising-phase vector was then set to 1 whenever this z-scored vector exceeded 2.5 ... This binary vector was treated as the firing rate in all further analyses." |
| Neural data time bin (reference decoding) | 3 frames = 100 ms | `fit_decoder(..., temporal_bin_size=3)` |
| Behaviour data time bin | same as neural (pooled jointly) | `fit_decoder` pools position with the same `AvgPool1d` |
| Reward rate | N/A | Free foraging, no task/reward structure |
| Position decoding error (Fig 1F, 15×15 = 5 cm bins) | mean **13.49 ± 0.32 cm** (n = 207 sessions) | computed from `data/precomputed_results/within_decoding`; paper: "significant decrease in position decoding error across recordings, achieving the maximum decoding accuracy reported in recent work" |

### Processing Details
- **Temporal alignment**: calcium and behaviour streams are acquired by the same DAQ at 30 Hz and were
  timestamped for post-hoc alignment; the distributed `trace` and `position` arrays are already
  frame-by-frame aligned (identical T for both within a day). No further alignment is required.
- **Temporal binning** (reference decoder): traces smoothed with `gaussian_filter1d(sigma = temporal_bin_size)`
  along time, then non-overlapping average pooling with kernel = stride = `temporal_bin_size` frames;
  position average-pooled identically and then discretized.
- **Spatial binning**: `bin_down = (max position across both axes and all days + buffer) / n_bins`, i.e. an
  isotropic bin size shared across all sessions of an animal (= 25 cm for n_bins = 3).

### Curation Steps
**Neuron curation rules** (as used by the paper's decoding analysis, `decode_position_within`):
1. Drop cells not registered on that day (all-NaN traces).
2. Drop cells with ≤ 5 events during the retained (running) frames of that session (`cell_threshold=5`).
3. **No place-cell selection** — the paper explicitly includes all cells ("motivated the inclusion of all
   cells in subsequent analyses").

**Trial curation rules**: the original experiment has no trials (40 min of continuous free foraging).
The decoder task defines trials as 1-minute segments. Reference-derived sample curation:
- Only frames with running speed > 5 cm/s are used for position decoding (`v_thresh=5`,
  speed smoothed with `gaussian_filter1d(sigma=5 frames)`).

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Position (15×15 = 5 cm bins), Gaussian naive Bayes, 5-fold CV | mean Euclidean error 13.49 cm (chance for uniform occupancy in a 75 cm box ≈ 39 cm) |
No classification accuracy for a 3×3 discretization is reported in the paper, so there is no direct
accuracy target; the 13.5 cm mean error implies most predictions fall within the correct 25 cm partition
or an adjacent one.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Orientation of the 3×3 geometry matrix | `get_env_mat(env)` returns a 3×3 matrix with 0 = blocked | `blocked` indices equal `flipud(get_env_mat(env)) == 0`, flattened row-major (verified for every session of every animal) | README: partitions numbered `[[0,1,2],[3,4,5],[6,7,8]]` | `get_env_mat` is in *plot* orientation (row 0 = bottom, as used for the polygon outlines with `flipud=True` in `plots.py`); `blocked` is in data/array orientation. **I use `blocked` directly**, and verify it against actual occupancy (see Step 10 sanity checks). |
| Mapping from position to partition index | `fit_decoder` builds class = `x_bin * n_bins + y_bin` (row = x) ; `plots.py` displays `maps[...].T` | Occupancy is exactly zero in partition `p` iff `p ∈ blocked` when partition index = `y_bin * 3 + x_bin` | README partition layout | Use **class = y_bin * 3 + x_bin**, which is the indexing in which `blocked` (and therefore the paper's partition numbering) is expressed. This is the transpose of `fit_decoder`'s internal class index — an arbitrary relabelling of the 9 classes that leaves decoding accuracy unchanged, but makes the decoder *input* (geometry) and *output* (partition) share one index. |
| Spatial resolution | reference uses `n_bins=15` | — | 3×3 partition design of the apparatus | Task specifies 3×3 = 9 bins → `n_bins=3`, which coincides with the physical partition grid (25 cm bins). |
| Temporal bin | reference uses 3 frames (100 ms) | events are sparse (≈0.24 Hz/cell) | — | Increased to 15 frames (500 ms) — see Step 5 "Key decisions" for the empirical justification. |
| Number of days per animal | `main.py` implies up to 31 | 31 for 6 animals, 21 for QLAK-CA1-51 | "up to three total repetitions (31 days)" | Consistent — animal 51 completed only 2 sequences (21 days). |
| `.mat` vs joblib | `load_dat` supports both | both present in `data/` | — | Use the joblib files: `mat2joblib` shows they are a lossless conversion of the `.mat` files, and they load far faster. Verified by spot-checking values against the `.mat` file (Step 10). |

Everything else is consistent: 7 animals × (31,31,31,21,31,31,31) days = 207 sessions; 5,413 cells;
69,744 registered cell-sessions; all numbers reproduce the paper exactly.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day, cell, t]` (binary events) | `neural[session][trial]` (n_neurons, T) | drop unregistered (NaN) cells → drop cells with ≤5 events in retained frames → `gaussian_filter1d(sigma=15 frames)` along time → average-pool 15 frames (500 ms) → split into 1-min trials → keep bins with speed > 5 cm/s | `decode_position_within`, `fit_decoder` | float32 |
| `blocked[day]` | `input[session][trial]` (9,) | 9-dim binary vector, 1 = partition blocked; constant within a session | (`get_env_mat` equivalent) | static per trial, 1-D array |
| `position[day, :, t]` | `output[session][trial]` (1, T) | average-pool 15 frames → `floor(pos / 25 cm)`, clipped to [0,2] per axis → class = `y_bin*3 + x_bin` | `fit_decoder`, `decode_position_within` | int, 9 categories |
| animal ID | `subjects`, `subject_idx` | 7 IDs | `main.py` animal list | |
| — | `brain_regions`, `brain_region_idx` | all cells are dorsal CA1 | paper | `['CA1']`, all zeros |
| `envs[day]`, day index | `metadata['session_info']` | geometry name, day index, sequence number | | |

### Key Decisions
1. **Session = one animal-day (40-min recording).** 207 sessions, matching the paper's unit of analysis.
   Different days have different registered cells, so neuron count varies per session (allowed by the format).
2. **Trials = consecutive, non-overlapping 60 s segments** (1800 frames at 30 Hz), as specified by the task.
   The trailing partial segment (< 60 s: 219, 91, 60, 71 frames for 5 animals; 1,666 frames ≈ 55 s for
   QLAK-CA1-30 and -50) is dropped so that every trial is a full minute. Maximum data loss 2.3 % for two
   animals, < 0.3 % for the rest. Result: 40 trials/session (39 for animals 30 and 50).
3. **Neural signal = the distributed binarized rising-phase `trace`.** No dF/F is needed: the pipeline in
   the paper already produced this binary event vector and "treated [it] as the firing rate in all further
   analyses".
4. **Temporal binning = 500 ms (15 frames), with the reference's smooth-then-average-pool recipe**
   (`gaussian_filter1d(sigma = bin) → AvgPool1d(kernel = stride = bin)`), so the neural value in a bin is a
   smoothed event *rate* (mean of the binary vector; multiplied by 30 to express it in Hz).
   *Why 500 ms rather than the reference's 100 ms*: calcium events are very sparse (≈0.24 Hz/cell), so a
   100 ms bin contains ~0.024 events/cell. The provided decoder is a linear read-out of 100 PCs evaluated
   per time bin, not a Bayesian accumulator, so per-bin SNR matters. A direct test (PCA + multinomial
   logistic regression, per session, 80/20 split, identical everything else) gave balanced accuracy:

   | session | 100 ms | 500 ms | 1 s | 2 s |
   |---|---|---|---|---|
   | 51 day 0 (square) | 0.254 | 0.331 | 0.379 | 0.308 |
   | 51 day 5 (rectangle) | 0.464 | **0.595** | 0.560 | 0.552 |
   | 51 day 12 (l) | 0.654 | **0.767** | 0.709 | 0.635 |

   500 ms is at or near the optimum and is still short relative to the time the mouse spends crossing a
   25 cm partition, so no positional information is smeared away. It also keeps the converted dataset to
   < 1 GB and the 200-epoch decoder training to a few minutes.
5. **Speed filter > 5 cm/s, as in `decode_position_within`.** Speed = `gaussian_filter1d(‖Δposition‖·30, sigma=5 frames)`;
   frame 0 is excluded (as in the reference, where `vel_idx[0]` stays False). The filter is applied *after*
   binning (a 500 ms bin is kept if its mean smoothed speed exceeds 5 cm/s) so that binning never averages
   across a temporal gap — this is a strict improvement over the reference, which pools after filtering and
   therefore mixes non-contiguous frames. Trials therefore have variable length (≈ 60 of 120 bins), which
   the target format explicitly allows. Empirically the filter also *improves* balanced accuracy
   (e.g. 0.652 → 0.757 on animal 51 day 18), because position coding is unreliable during immobility.
6. **Cell curation**: drop unregistered (NaN) cells, then drop cells with ≤ 5 events among the retained
   running frames (`cell_threshold = 5` in `decode_position_within`). No place-cell selection — the paper
   explicitly includes all cells.
7. **Spatial discretization**: `bin_down = (max position over both axes and all days of the animal + 1e-5)/3`,
   exactly the `decode_position_within` rule with `n_bins = 3`. The maximum is 75.0 cm for every animal, so
   `bin_down` = 25 cm — the true physical partition size. Bins are `floor(pos/bin_down)` clipped to [0, 2]
   (the clip guards the single-frame edge case `pos == 75.0` exactly).
8. **Output = a single categorical variable with 9 values** (partition index `y*3 + x`), time-varying at the
   500 ms bin resolution. The 3×3 partitions are exactly the arena's physical partitions, so the class index
   is directly comparable with the geometry input.
9. **Input = 9-dim binary geometry vector** (1 = partition blocked), static per trial, taken from the
   `blocked` field. `square` (`blocked == -1`) → all zeros. This is the "environment geometry" the task asks
   for and it shares its index with the output classes.
10. **Empty/degenerate trials are dropped**: a trial with fewer than 10 retained bins (5 s of running) is
    discarded; a session is kept only if ≥ 2 trials survive (required for the train/validation split).
11. **Trial alignment**: trials have no experimental alignment event (continuous free foraging), so trial t
    starts at `t*60 s` from the start of the recording; `off_start = 0`, `off_end = 60`.

### Planned Sanity Checks
- [x] Total sessions = 207, subjects = 7, unique cells = 5,413, registered cell-sessions = 69,744 (paper).
- [ ] For every session, occupancy of a partition is exactly 0 iff that partition is listed in `blocked`
      (ties the output discretization to the geometry input and to the arena orientation).
- [ ] Number of distinct geometries = 10, each appearing 2–3 times per animal, sequences starting/ending
      with `square`.
- [ ] Spot-check neural values: reload the raw joblib (and the `.mat`) and recompute one trial's binned
      trace for a named cell, compare with `np.allclose`.
- [ ] Spot-check output: recompute the partition index of a specific trial/timepoint from raw positions.
- [ ] Spot-check input: compare the 9-dim geometry vector against `blocked` and against
      `flipud(get_env_mat(env))` for all sessions.
- [ ] Neural values non-negative, finite; output in [0, 8]; input in {0, 1}.
- [ ] Mean event rate after binning ≈ 0.21–0.27 Hz per animal (matches raw-data estimate).
- [ ] Decoding accuracy well above chance (1/9 = 0.111) and broadly consistent with the paper's 13.5 cm
      mean error at 5 cm resolution.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Usage:

```
python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
```

Structure:
- `parse_blocked` — `blocked` field → 9-dim binary geometry vector (the decoder input).
- `compute_speed` — running speed exactly as `decode_position_within`
  (`gaussian_filter1d(‖Δposition‖·fps, sigma=5 frames)`, element 0 set to 0/excluded).
- `bin_time` — reshape-and-mean; the numpy equivalent of the reference's
  `AvgPool1d(kernel=stride=bin)`, applied *within* 1-minute trials.
- `convert_session` — the whole per-session pipeline (8 numbered stages, in the order:
  registered cells → speed → position→partition → geometry → sample mask → cell activity
  filter → neural smoothing/binning → per-trial assembly).
- `convert_animal` — loads one animal's joblib archive, computes the animal-wide
  `bin_down`, loops over days.
- `build_dataset` / `print_summary` — target-format assembly and statistics.
- `_plot_processing` — the 7-panel `--show-processing` figure.

Code inefficiencies identified:
- The reference filters traces with `gaussian_filter1d(traces, sigma, axis=0)` on a
  `(T, n_cells)` array — a strided pass over memory. Keeping the native `(n_cells, T)`
  layout and filtering `axis=-1` is ~15 % faster and avoids a transpose copy.
- Loading a whole animal decompresses ~17 GB of float64; only one animal is held at a
  time and each is loaded exactly once (no per-session file I/O).

Code speedups added:
- float32 for the neural stream throughout (halves memory and filter cost).
- All binning done with a single `reshape(...).mean(-1)` instead of per-trial loops.
- Cell-activity counting done with one boolean-mask sum over the whole session.

Measured: 0.57–1.07 s of processing per session, 6–17 s to load each animal.
Full conversion = **4.7 min** wall clock, well under the 15-minute budget, so no
multiprocessing was needed (it would have multiplied the 17 GB peak per worker).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl` (4.5 MB),
`processing_QLAK-CA1-08_day00_square.png`, `processing_QLAK-CA1-30_day03_l.png`.

The two sample sessions come from two different animals and two different geometries
(square = nothing blocked, and `l` = partitions 1,2,4,5 blocked), so the sample exercises
subject indexing, the geometry input and the "blocked partitions are never visited" logic.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (1 session each) |
| Neurons (total, after curation) | 529 (181 + 348) |
| Registered cell-sessions | 541 |
| Trials (total) | 77 of 78 possible (38 + 39) |
| Trials / session | 38.5 |
| Time bins kept | 4,339 (46.4 % of all 500 ms bins) |
| Mean bins / trial | 56.4 of 120 |
| Input range (9 geometry dims) | [0, 1] — `[0,0]` for the square session, `[1,1]` on partitions 1,2,4,5 for the `l` session |
| Output range | [0, 8] |
| Output distribution | [0.142, 0.035, 0.069, 0.096, 0.054, 0.076, 0.134, 0.151, 0.243] |
| Mean neural value | 0.18 Hz (unweighted mean of per-trial means over these 2 sessions) |

### Processing Plots Review
The 7 panels were inspected for `QLAK-CA1-30_day03_l`:
1. Raw position over the full 40 min with 1-min trial boundaries and the 25 / 50 cm
   partition borders — position stays in [0, 75], no dropouts.
2. One trial at 30 Hz vs the 500 ms binned position, with the derived x/y bin index
   overlaid: the bin index steps **exactly** where the position trace crosses a border.
   No lead or lag ⇒ no temporal misalignment and correct discretization.
3. Speed: raw smoothed trace, binned values on top of it, threshold line, and green
   shading on exactly the bins above 5 cm/s (80/120 in this trial).
4. + 5. Raw binary events for 30 cells at 30 Hz, and the same cells after smoothing and
   500 ms pooling: every raster tick lines up with a bright bin at the same time.
6. The output partition index for the trial, with kept samples marked — it is consistent
   with the x/y bin traces in panel 2 (e.g. x bin 0, y bin 2 → partition 6).
7. Occupancy per partition for the kept samples, with the geometry input overlaid:
   partitions 1, 2, 4, 5 are flagged BLOCKED (input = 1) and contain **0 samples**,
   the other five contain thousands. Input and output indices are therefore aligned.

No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| filter along the native contiguous time axis instead of transposing | ~15 % of filter time |
| float32 neural stream | ~2x memory, ~5 % time |
| vectorized reshape-based binning instead of per-trial loops | large (avoids 8,000 python-level loops) |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| load animal archive | (6–17 s per animal, 7 animals) | ~100 s |
| per-session processing | 0.75–1.6 s | ~210 s |
| pickle write | — | ~1 s |
| **estimated total** | | **~5–6 min** (actual: 4.7 min) |

Format check: `python -u /app/train_decoder.py /app/sample_data.pkl --verify-only`
→ `/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**
Input ranges [0,1] and output fractions are as expected; the `l` session has exactly
0.000 of its samples in each of its four blocked partitions.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None

### Training Progress
Loss fell monotonically from 2.887 (epoch 1) to 0.437 (epoch 200); test loss 1.620.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| position_partition | 0.8806 | 0.5494 | 0.1111 |

Validation accuracy is 4.9x chance on only two sessions. The train/validation gap is
expected here: 100 PCs are fitted per session from just ~30 training trials.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full` → `/app/conversion_full_out.txt`
(284 s), then `python -u /app/train_decoder.py /app/converted_data.pkl --verify-only`
→ `/app/verification_full_out.txt`.

### Output Files
- `converted_data.pkl`: 709.3 MB
- `verification_full_out.txt`: created — **valid, no errors, no warnings**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 7 (implied; "mean cells per animal = 773 ± 68 SE") | 7 in `main.py` animal list | 7 files | 7 | ✅ |
| Sessions | 207 | — | 207 (31+31+31+21+31+31+31) | 207 | ✅ |
| Total unique neurons | 5,413 | — | 5,413 | 5,413 | ✅ |
| Cells per animal | min 515, max 952, mean 773 | — | 515…952, mean 773.3 | same | ✅ |
| Registered cell-sessions ("rate maps") | 69,744 | — | 69,744 | 69,744 | ✅ |
| Mean neurons/session (registered) | — | — | 336.9 | 336.9 | ✅ |
| Mean neurons/session (after >5-event filter) | — | `cell_threshold=5` | — | 332.9 (68,911 total; 833 = 1.2 % dropped) | ✅ |
| Geometries | 10 | 10 in `get_env_mat` | 10 | 10 | ✅ |
| Sequences per animal | up to 3 (31 days) | `n_seq = n_days // n_shapes` | 3 (2 for CA1-51) | 3 / 2 | ✅ |
| Session duration | 40 min | — | 39.93–40.12 min | 39/40 complete 1-min trials | ✅ |
| Sampling rate | 30 Hz | `fps=30` | T/duration = 30 Hz | 30 Hz → 500 ms bins | ✅ |
| Arena / partition size | 75 × 75 cm, 3 × 3 | `bin_down` rule | position range exactly [0, 75] | bin_down = 25.000003 cm | ✅ |
| Speed threshold | — | 5 cm/s | 45–62 % of frames above it | 53.9 % of bins kept | ✅ |
| Trials (total) | n/a (no trials in the experiment) | — | — | 8,019 (mean 38.74/session) | n/a |
| Input range | geometry ∈ {blocked, open} | `get_env_mat` ∈ {0,1} | `blocked` indices | [0,1], and == `flipud(get_env_mat)==0` on all 207 sessions | ✅ |
| Output distribution | not reported | — | occupancy | [.095 .108 .114 .088 .070 .088 .108 .158 .171] | ✅ plausible |
| Position decoding error (5 cm bins) | 13.49 cm over 207 sessions | `decode_position_within` | saved `within_decoding` | reproduced **exactly** (see Step 10 Check 3) | ✅ |

No data was lost: every animal, every session and every registered cell-session is
accounted for. The only discarded material is (a) the trailing < 60 s of each recording
that cannot form a complete 1-minute trial, (b) samples excluded by the reference's own
running-speed and cell-activity criteria, (c) 28 time bins (0.003 %) tracked inside
physically walled-off partitions, and (d) trials/sessions left with too little data
(see Step 10 Check 4 for the exact accounting).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
`Data format is valid, no errors or warnings.` — **zero errors and zero warnings**, so
there is nothing left to fix or excuse. Specifically, the verifier found:
no NaN/Inf, no non-integer outputs, no dimension mismatches, correct dtypes (`float32`
neural/input, `int64` output), `brain_region_idx` lengths matching the neuron counts,
consistent `dinput = 9` / `doutput = 1`, and no all-zero trials.

One observation that is *not* a problem: `geometry_partition_7_blocked` has range
[0.0, 0.0]. Partition 7 (x = 1, y = 2) is never occluded in any of the 10 geometries
(`square` none; `o` 4; `l` 1,2,4,5; `u` 4,5; `bit donut` 0,4; `rectangle` 0,3,6;
`+` 0,2,6,8; `glenn` 0,8; `i` 3,5; `t` 3,5,6,8), so that input dimension is constant by
design. It is kept so that the input vector is indexed identically to the output classes.

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py` → `/app/cache/sanity_checks_out.txt`)
Every check re-derives its quantity **from the original data files** (the joblib archives
and, for one animal, the original `.mat`), without importing anything from
`convert_data.py`, and compares with `np.allclose` / `np.array_equal`. **All 33 checks pass.**

| # | Check | Result |
|---|---|---|
| 0 | 7 subjects, 207 sessions, 69,744 registered cell-sessions, 10 geometries (paper) | PASS |
| 1 | **Neural**: for 4 randomly chosen sessions (one per animal), independently redo the entire pipeline from `data/<animal>` and compare *every* value of *every* trial with `np.allclose(atol=1e-6)`; also compare trial counts and neuron counts | PASS |
| 1 | **Neural** spot-check of one explicitly named element, e.g. `QLAK-CA1-74_day28_glenn` `neural[trial=5, neuron=6, t=20] = 26.283426 Hz` | PASS |
| 1 | **Output**: every partition index of every trial of those sessions matches the independent recomputation exactly (`np.array_equal`) | PASS |
| 2 | **Input**: for all 207 sessions the 9-dim geometry vector equals `flipud(get_env_mat(env)) == 0` (the reference code's own geometry definition) *and* equals the dataset's `blocked` field, and is identical on every trial of the session | PASS |
| 3 | **Output**: no sample anywhere in the dataset falls inside a blocked partition (0 offenders out of 528,802) | PASS |
| 3 | All 9 classes present; output dtype integer | PASS |
| 4 | Shapes/dtypes consistent; neural finite and non-negative; 10 ≤ T ≤ 120 for every trial; `subject_idx` agrees with `session_info` | PASS |
| 5 | The joblib archive is byte-for-byte equivalent to the original MATLAB file (`position` and `trace` of day 7 of QLAK-CA1-51 compared through `h5py`) | PASS |

### Check 3: Reference code comparison
`/app/cache/reference_comparison.py` contains **verbatim copies** of `fit_decoder`,
`test_decoder` and `decode_position_within` from `code/georepca1/src/utils.py` (the only
edits: `np.where(...)[0]` → `np.where(...)[0][0]`, required because numpy 2 no longer
casts a 1-element array to a scalar; and removal of the coherence-map block, which does
not touch the returned `distances`). Running it on **all days of two animals** reproduces
the paper's own saved Figure-1F decoding errors **exactly**:

| Animal | Sessions | Reproduced mean error | Saved `within_decoding` | Max per-day difference |
|---|---|---|---|---|
| QLAK-CA1-51 | 21 | 17.567 cm | 17.567 cm | **0.0000 cm** |
| QLAK-CA1-75 | 31 | 10.893 cm | 10.893 cm | **0.0000 cm** |

This is an end-to-end proof that my data loading, stream alignment and curation are
identical to the reference implementation.

Step-by-step comparison of my script against the reference:

| Stage | Reference | `convert_data.py` | Same? |
|---|---|---|---|
| (a) loading | `load_dat(animal, p, format="joblib")` → `joblib.load(data/<animal>)[animal]` | identical call | ✅ |
| (b) neuron filtering | unregistered cells are NaN and are removed implicitly; `cell_idx = sum(traces[vel_idx]) > cell_threshold(=5)` | explicit `~isnan(trace[:,0])` mask, then `raw[:, frame_keep].sum(1) > 5` | ✅ |
| (b) neuron filtering | the paper's place-cell test (`get_place_cells`) is **not** applied before decoding | not applied | ✅ |
| (c) temporal alignment | `trace` and `position` share one 30 Hz frame index (hardware-timestamped by the same DAQ); no further alignment | same; trials are cut on the same frame index for all three streams | ✅ |
| (c) sample selection | `vel_idx = gaussian_filter1d(‖Δbehav‖·fps, sigma=5) > v_thresh/bin_down`, index 0 left False | same formula in cm/s (`speed > 5`), bin 0 of the session excluded | ✅ (applied after binning, see below) |
| (d) binning (neural) | `AvgPool1d(k=s=3)` over `gaussian_filter1d(traces, sigma=3, axis=0)` | `mean` over 15-frame blocks of `gaussian_filter1d(trace, sigma=15, axis=-1)` | ⚠️ bin size 15 instead of 3 frames — justified in Step 5 decision 4 |
| (d) binning (position) | same pooling, then `.astype(int)` after dividing by `bin_down` | same: mean position per bin, then `floor(pos/bin_down)` clipped to [0,2] | ✅ (clip added, see Check 5) |
| (d) spatial grid | `bin_down = (max over time, axes and days + buffer)/n_bins` | identical, with `n_bins = 3` instead of 15 (task requirement) | ✅ |
| (e) input construction | reference never builds a decoder input (its Bayes decoder uses neural data only); geometry appears via `get_env_mat` | 9-dim binary vector from `blocked`, verified identical to `flipud(get_env_mat(env))==0` | ✅ |
| (f) output construction | class index = `x_bin*n_bins + y_bin` (`fit_decoder`'s `empty_map` flattening) | class = `y_bin*3 + x_bin` | ⚠️ transposed labelling, chosen so the output index matches the dataset's own partition numbering and hence the geometry input. A relabelling of 9 classes; it cannot change any accuracy. |

Two deliberate deviations, both documented above and in Step 5: the temporal bin size
(500 ms instead of 100 ms, required to give the provided per-timepoint linear decoder a
usable SNR from a 0.24 Hz event train) and the class-index transpose (required so that
decoder input and output share an index). One refinement: the speed filter is applied
*after* binning rather than before, so that no time bin ever averages across a temporal
gap — the reference pools already-filtered frames, which silently mixes non-contiguous
samples. Neither affects agreement with the reference results above.

### Check 4: Key statistics comparison
See the table in Step 9 — every statistic available in the paper (subjects, sessions,
neurons, cells per animal, rate maps, geometries, sequences, session length, sampling
rate, arena size) is reproduced exactly. Full accounting of the data:

| Quantity | Value | Explanation |
|---|---|---|
| frames in the raw recordings | 14,903,019 (207 sessions × 71,866–72,219) | 100 % |
| frames in complete 1-min trials | 14,736,600 (**98.88 %**) | trailing partial minute dropped: 1,666 frames/session for CA1-30 and CA1-50 (39 trials), ≤ 219 for the other five animals (40 trials) |
| 500 ms bins | 982,440 | 8,187 possible trial-minutes |
| bins dropped: speed ≤ 5 cm/s | 453,142 (46.1 %) | reference criterion |
| bins dropped: inside a blocked partition | 28 (0.003 %) | DeepLabCut artifacts; 428 of the 443 offending raw frames come from a single session (CA1-51 day 4) |
| bins dropped with their trial (< 10 bins left) | 468 bins in 168 trials of 8,187 (2.05 % of trials, 0.05 % of bins) | minutes the mouse spent almost entirely immobile (mean 2.8 running bins each) |
| **bins retained** | **528,802 (53.8 %)** | 73.4 h of running, in 8,019 trials |
| registered cell-sessions | 69,744 | matches the paper |
| cell-sessions dropped: ≤ 5 events while running | 833 (1.19 %) | reference criterion |
| **neurons retained** | **68,911** | mean 332.9/session |
| sessions dropped | 0 | every one of the 207 sessions survives |
| subjects dropped | 0 | |

Output class distribution [.095 .108 .114 .088 .070 .088 .108 .158 .171]: mildly
non-uniform, as expected for a freely foraging mouse in a partitioned box (the centre
partition 4 is the least occupied because it is blocked in 5 of the 10 geometries and is
the least thigmotaxic location; partitions 7 and 8 are never/rarely blocked). The
decoder is scored with *balanced* accuracy and is trained with `balanced_loss=True`, so
this imbalance does not inflate the score.

### Check 5: Edge cases
| Edge case | Handling |
|---|---|
| Position exactly at the arena maximum (75.0 cm) | `bin_down = (75 + 1e-5)/3`, so `floor(75/bin_down) = 2`; a `np.clip(..., 0, 2)` guard is applied anyway. Without the clip, the *reference* formula puts a sample at exactly `n_bins`, which is out of range — this really does happen (see next row). |
| The reference's own `behav_max` logic is fragile | Running `decode_position_within` on a *subset* of an animal's days raises `IndexError` because the bin grid is derived from the subset's maximum. My conversion always uses the animal's global maximum (75.0 cm for all 7 animals), so the grid is the true 25 cm partition grid. Documented in `cache/reference_comparison.py`. |
| Boundary frames in the `rectangle` sessions | The mouse's minimum x is exactly 25.0 cm, which the buffered `bin_down` rounds into the walled-off column. Exactly 1 frame per rectangle session; removed by the "unreachable partition" rule. |
| Frame 0 has no defined speed | Excluded, matching the reference (`vel_idx[0]` stays False). |
| Recordings are not an exact multiple of 60 s | `n_frames // 1800` complete trials; the remainder is dropped. |
| Different session lengths across animals (71,866–72,219 frames) | Handled per session; gives 39 or 40 trials. |
| Cells unregistered on a day (NaN traces) | Dropped per session. Verified that NaNs are always whole-session (`isnan(trace).all(axis=time) == isnan(trace[:,:,0])`), and confirmed globally because no NaN survives into the converted data (`gaussian_filter1d` would have propagated any partial NaN and the verifier would have flagged it). |
| A session where every cell fails the activity filter | `convert_session` returns None; never triggered. |
| A trial with almost no running | Dropped below 10 bins (164 of 8,183); no session ends up with fewer than 2 trials, so none is dropped. |
| Determinism | The conversion contains no RNG; re-running produces a byte-identical summary. |

### Issues found and resolved
1. **Orientation ambiguity between `get_env_mat` and `blocked`** — resolved by testing
   both against actual occupancy on all 207 sessions; `blocked` is in array orientation
   and equals `flipud(get_env_mat(env)) == 0`. Adopted the `blocked` convention for both
   input and output.
2. **Samples tracked inside walled-off partitions** (443 raw frames dataset-wide) —
   originally passed through as label noise; now dropped, and counted in the summary.
3. **`sequence` metadata off by one on the closing square day** — originally
   `day // 10 + 1`, which labelled day 30 as a 4th sequence. Changed to match the
   reference's square-to-square definition (`get_rsm_partitioned_sequences`:
   `n_seq = n_days // n_shapes`), so day 30 belongs to sequence 3. The full conversion
   was re-run and all checks in this step re-executed afterwards; every other statistic
   was unchanged.
4. **numpy 2 incompatibility in the reference code** — `np.where(...)[0]` assigned to a
   scalar element now raises; fixed in my verbatim copy with `[0][0]`, which is what
   numpy 1.x did implicitly. (This is in the comparison script only, not the conversion.)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.

### Training Progress
- Loss decreasing: **Yes**, monotonically — 3.185 (epoch 1) → 0.2915 (epoch 200).
  Test loss 1.111.
- Trained on 6,382 trials, validated on 1,637 held-out trials (every session contributes
  to both sides of the split).

### Decoder Results (Full)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|--------|-------------|--------|-------|
| position_partition (9 classes) | 0.1111 | 0.9511 | **0.7157** | 6.4x chance; train/val ratio 1.33 |

`predictions.png` shows the predicted partition (dashed) tracking the true partition
(solid) closely on held-out trials, with errors concentrated at partition transitions.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

Breakdowns below come from `/app/cache/analyze_decoder.py` → `/app/cache/analyze_decoder_out.txt`
(the same `train_validate_decoder` call with the same hyperparameters, instrumented to
report per-animal, per-geometry and per-class results). Because the train/test split and the torch initialisation are randomly seeded on each invocation, that run scored 0.7127 overall where the logged `train_decoder.py` run scored 0.7157; the two agree to within run-to-run noise.

### Check 1: Accuracy vs chance
| Output | Classes | Chance (uniform) | Chance (majority class) | Validation balanced acc | Ratio to chance |
|---|---|---|---|---|---|
| position_partition | 9 | 0.1111 | 0.171 | **0.7157** | **6.4x** |

Far above both chance definitions, and above 1.5x chance for **every** class
individually — per-class recall ranges from 0.612 (partition 4, the centre, the rarest
class) to 0.791 (partition 8):

| partition | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| recall | .773 | .667 | .753 | .689 | .612 | .709 | .738 | .682 | .791 |

The confusion matrix is dominated by **adjacent-partition** confusions (e.g. true 0 →
predicted 1 in 8.0 % and → 3 in 7.3 %, while 0 → 8, the opposite corner, is 1.3 %),
which is the signature of a genuine spatial code rather than a labelling bug.
Only 6 of 108,146 held-out predictions land in a partition that is walled off on that
session (0.006 %), confirming the geometry input is doing its job.

### Check 2: Accuracy comparison to the paper
The paper reports no classification accuracy — only Bayesian decoding **error in cm**
(Figure 1F). I therefore made two comparisons.

**(2a) Same metric, same decoder, same resolution.** I ran the paper's own Gaussian
naive-Bayes decoder (verbatim `decode_position_within`) at the task's 3 × 3 resolution:

| Animal | Paper's decoder, 3x3 balanced acc | This conversion + provided decoder | Δ |
|---|---|---|---|
| QLAK-CA1-51 (paper's *worst* animal, 17.57 cm) | 0.4938 | **0.6525** | +0.159 |
| QLAK-CA1-75 (paper's *best* animal, 10.89 cm) | 0.6854 | **0.7550** | +0.070 |

The converted data supports **higher** accuracy than the paper's own method on the same
animals, so there is no evidence of information lost in conversion. The gain is expected:
the provided decoder is trained jointly across all 207 sessions (the paper's is fit
per session, per fold) and uses 500 ms rather than 100 ms bins.

**(2b) Same metric as the paper (Euclidean error), per session.** Converting the
decoder's 9-class predictions into partition centres:

| | Paper (Bayes, 15x15 = 5 cm bins) | This work (9 classes, 25 cm bins) |
|---|---|---|
| mean error over 207 sessions | 13.49 cm | 8.44 cm |

These are not directly comparable (coarser bins cannot produce small errors but also
cannot produce large ones), so the informative statistic is the **agreement across
sessions and animals**:

| Comparison | Statistic |
|---|---|
| per-session error, ours vs paper's saved `within_decoding` (n = 207) | Pearson **r = 0.924**, p = 8e-88 |
| per-session balanced accuracy vs paper's error (n = 207) | Spearman **r = −0.804**, p = 3e-48 |
| per-animal error, ours vs paper's (n = 7) | Pearson **r = 0.987** |
| per-animal balanced accuracy vs paper's error (n = 7) | Spearman **r = −0.964**, p = 0.0005 |

Sessions and animals the paper decodes well are exactly the ones this decoder decodes
well. Per animal:

| Animal | Paper error (cm) | Our validation bal. acc | Our error (cm) |
|---|---|---|---|
| QLAK-CA1-08 | 16.49 | 0.6778 | 10.03 |
| QLAK-CA1-30 | 12.83 | 0.7169 | 8.23 |
| QLAK-CA1-50 | 11.43 | 0.7254 | 7.45 |
| QLAK-CA1-51 | 17.57 | 0.6525 | 11.00 |
| QLAK-CA1-56 | 15.16 | 0.6938 | 9.33 |
| QLAK-CA1-74 | 11.38 | 0.7231 | 7.55 |
| QLAK-CA1-75 | 10.89 | 0.7550 | 6.54 |

Per geometry, accuracy is lowest in the open `square` (0.620) and highest in the most
subdivided geometries (`+` 0.753, `t` 0.748) — as expected, since walls both restrict
the reachable set and drive the remapping that makes positions more discriminable.

### Check 3: Train vs validation gap
Training balanced accuracy 0.9511, validation 0.7157 → ratio **1.33 < 1.5**. There is
some overfitting (100 free PCs per session fitted from ~31 training trials each), but it
is within the acceptable range and there is no data leakage: the split is by whole
1-minute trials, and because the speed filter removes whole bins rather than shifting
anything, no timepoint appears in both sets. Trials are contiguous, non-overlapping
segments of the recording, so no sample is duplicated across trials either.

### Issues found and resolved
- None new. All four issues listed in Step 10 were fixed before this step; the full
  conversion, verification, sanity checks and decoder training were all re-run on the
  final pickle, and all statistics above come from that final run.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created
- [x] `cache/` folder created with `README_CACHE.md`
- [x] All files organized
