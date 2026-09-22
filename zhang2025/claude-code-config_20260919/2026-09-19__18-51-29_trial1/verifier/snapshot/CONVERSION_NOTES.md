# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-wide Map (BWM) public release — "A brain-wide map of neural activity during complex behaviour" (IBL et al.), processed following "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al., Neuron 2026).
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents (`/app`):
- `.manifest`, `Dockerfile`, `docker-compose.yaml`, `stage_cache.sh` — container/infra files
- `CONVERSION_NOTES.md` (this file)
- `code/` — reference code
  - `code/code_zhang2025/` — the methods-paper repo (src/, scripts/, notebooks/, data/)
  - `code/ibllib/` — IBL library source (brainbox, ibllib); also installed in site-packages
- `data/one_cache/` — ONE cache. Release tables (`Brainwidemap`, `2022_Q4_IBL_et_al_BWM`,
  `2025_Q3_IBL_et_al_BWM`), cached Alyx REST responses (`.rest`, 6483 files) and 12 lab
  directories symlinked to the read-only 567 GB dataset mount.
- `datapaper.pdf`, `methodpaper.pdf`, `dataarchitecture.pdf`, `methods.txt` — references
- `decoder.py`, `train_decoder.py` — provided decoder / validation code

Environment verified: `python3` 3.13, numpy 2.3.5, torch 2.6.0+cu124 (CUDA, NVIDIA L4 23 GB),
ONE-api 3.5.2, brainbox/ibllib importable. 128 CPUs, 1006 GB RAM.

ONE works fully offline: `ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)`
picks up `$HOME/.one/.caches` → `CACHE_DIR=/app/data/one_cache`, runs in `remote` mode but is
served entirely from the cached REST responses and the symlinked ALF files.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The reference pipeline is `code/code_zhang2025/src/0_data_caching.py`, which builds exactly the
kind of trial-aligned, binned dataset we need. Its parameters:

```python
params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
          'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']
freeze_file = 'data/bwm_release.csv'    # 699 pids / 459 eids / 139 subjects
```

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data(one, eid, bwm_df, params)` | `src/utils/ibl_data_utils.py` | LOADING | Top-level per-session loader: all probes' spike sorting, merges probes, loads trials + trial mask, loads "anytime" behaviours |
| `load_spiking_data(one, pid, qc=None)` | `ibl_data_utils.py` | LOADING/CURATION | `SpikeSortingLoader.load_spike_sorting()` + `merge_clusters()`; if `qc` given keeps `clusters.label >= qc`. **Called by `prepare_data` with the default `qc=None`, i.e. no unit QC.** |
| `merge_probes(spikes_list, clusters_list)` | `ibl_data_utils.py` | LOADING | Concatenate probes of one session into one "virtual probe", re-index cluster ids, re-sort spikes by time |
| `load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', max_trial_len=10.0, exclude_nochoice=True)` | `ibl_data_utils.py` | CURATION | Trials table + boolean mask. Excludes RT∉[0.08,2.0] s, `feedback_times - goCue_times > 10 s`, NaN in {stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType}, and no-choice trials. `exclude_unbiased=False` → pLeft = 0.5 trials are **kept**. |
| `list_brain_regions(neural_dict, single_region=False)` | `ibl_data_utils.py` | PROCESSING | `BrainRegions().acronym2acronym(acronyms, mapping='Beryl')`; with `single_region=False` returns one group containing *all* Beryl regions in the session |
| `select_brain_regions(...)` | `ibl_data_utils.py` | CURATION | `np.isin(beryl_reg, region)` → cluster indices to keep |
| `bin_spiking_data(reg_clu_ids, neural_df, trials_df, **params)` | `ibl_data_utils.py` | PROCESSING | intervals = `stimOn_times + (-0.5, 1.5)`; `bincount2D(..., xbin=0.02, xlim=[t_beg,t_end])` per trial → `(ntrials, nclusters, 100)` spike counts |
| `get_spike_data_per_interval(...)` | `ibl_data_utils.py` | PROCESSING | Per-trial worker for the above. Spikes selected as `(times >= t_beg) & (times < t_end)`, binned with `floor((t - t_beg)/binsize)`, truncated to `n_bins = ceil(interval_len/binsize) = 100` |
| `load_target_behavior(one, eid, target)` | `ibl_data_utils.py` | LOADING | `SessionLoader.load_wheel()` → `wheel-speed = abs(velocity)`; `SessionLoader.load_motion_energy(views=['left'/'right'])` → `whiskerMotionEnergy` |
| `get_behavior_per_interval(...)` | `ibl_data_utils.py` | PROCESSING | Slices behaviour by `searchsorted(times, beg, 'right') : searchsorted(times, end, 'left')`, rejects intervals that are empty / NaN-timed / whose data starts >1 binsize late or ends >1 binsize early, then `interp1d(linear, extrapolate)` onto `x = linspace(beg+binsize, end, 100)` (**right edge of each bin**) |
| `bin_behaviors(one, eid, behaviors, trials_df, **params)` | `ibl_data_utils.py` | PROCESSING | Adds per-trial `choice`, `block`(=probabilityLeft), `reward`, `contrast` plus the binned dynamic behaviours; whisker ME uses **left camera, falling back to right** |
| `align_spike_behavior(binned_spikes, binned_behaviors, beh_names, trials_mask)` | `ibl_data_utils.py` | CURATION | Final trial mask = trials_mask AND "all behaviours present for this trial"; deletes failing trials from every stream |
| `create_dataset(...)` | `src/utils/dataset_utils.py` | SAVING | Packs spike counts as `csr_array(dtype=np.ubyte)` + per-trial behaviour columns into a HF `Dataset` |
| `standardize_spike_data`, `SingleSessionDataset` | `src/utils/data_loader_utils.py` | PROCESSING (decoder side) | z-scores spike counts per time-bin using **training-set** statistics, one-hot encodes discrete targets, `StandardScaler` for continuous targets, NaNs → trial mean. Done at decode time, **not** in the cached data. |

### Notes
- Neural data are **spike counts per 20 ms bin**, stored unnormalised; normalisation is a
  decoder-side step. So the converted dataset should hold raw counts.
- `prepare_data` also calls `spike_loader.raw_electrophysiology(band='ap', stream=True).fs`
  purely to record the AP sampling frequency in metadata. That streams raw binary from the
  remote server and is not needed for any data value; it is skipped here.
- The reference uses `multiprocessing` per *trial*; that is strictly slower than binning a
  whole session with one vectorised `np.bincount`, which is what this conversion does
  (verified to give bit-identical counts, Step 10 Check 2).
- `load_trials_and_mask` in the repo calls `SessionLoader(one, eid)` positionally, which the
  installed ibllib no longer accepts; we pass a pre-built `SessionLoader(one=one, eid=eid)`
  through its `sess_loader` argument, leaving the masking logic untouched.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
ONE cache, standard ALF layout:

```
data/one_cache/<lab>/Subjects/<subject>/<yyyy-mm-dd>/<nnn>/
    alf/
        _ibl_trials.table.pqt                 (in revision folder, e.g. #2025-03-03#)
        _ibl_trials.{stimOn_times,stimOff_times,...}.npy
        _ibl_wheel.{position,timestamps}.npy
        _ibl_{left,right}Camera.times.npy
        {left,right}Camera.ROIMotionEnergy.npy   (in revision folders)
        probe0X/pykilosort/#2024-05-06#/
            spikes.{times,clusters,amps,depths}.npy
            clusters.{channels,depths,metrics.pqt,uuids.csv}
            channels.{brainLocationIds_ccf_2017,localCoordinates,mlapdv,rawInd,labels}.npy
    raw_ephys_data/...
```

Release tables: `data/one_cache/Brainwidemap/{sessions,datasets}.pqt` (480 sessions listed),
`2022_Q4_IBL_et_al_BWM` (354), `2025_Q3_IBL_et_al_BWM` (459). The **freeze actually used by the
reference code** is `code/code_zhang2025/data/bwm_release.csv`.

Files present on disk (following symlinks): 701 `spikes.times.npy`, 466 `_ibl_trials.table.pqt`,
861 `*ROIMotionEnergy.npy`, 461 `_ibl_wheel.position.npy` → the complete BWM release is staged
locally; nothing needs downloading.

### Dataset Size (from data files / release freeze)
| Statistic | Value |
|-----------|-------|
| Sessions (eids) in `bwm_release.csv` | 459 |
| Probe insertions (pids) | 699 |
| Subjects | 139 |
| Labs | 12 |
| Sessions with left-camera whisker ME | 436 |
| Sessions with right-camera whisker ME | 433 |
| Sessions with left **or** right whisker ME | 445 |
| Sessions with wheel / trials table | 459 |
| Clusters per probe (example pid 56f2a378…) | 898 (label 1.0: 76, 0.67: 230, 0.33: 401, 0: 191) |
| Trials per session (examples) | 565, 425, 557 |

Key trial-table columns: `stimOn_times`, `firstMovement_times`, `feedback_times`,
`goCue_times`, `response_times`, `choice` ∈ {−1, 0, +1}, `contrastLeft`, `contrastRight`,
`probabilityLeft` ∈ {0.2, 0.5, 0.8}, `feedbackType`, `rewardVolume`, `intervals_{0,1}`.

Wheel is resampled to 1000 Hz by `SessionLoader.load_wheel` (position interpolated, velocity
from an 8th-order Butterworth low-pass at 20 Hz). Left camera ≈ 60 Hz, right camera ≈ 150 Hz.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Mice | 139 | "We trained 139 mice (94 male and 45 female)" |
| Sessions released | 459 | "a total of 459 sessions, 699 insertions and 621,733 neurons remained" |
| Insertions | 699 | same |
| Units (all) | 621,733, avg 889/probe | "produced 621,733 units …, averaging 889 per probe" |
| **Well-isolated neurons** | **75,708, avg 108/probe** | "stringent quality-control metrics …, which identified 75,708 well-isolated neurons, averaging 108 per probe" |
| Sessions used by methods paper | **433** | "We apply our models to 433 IBL sessions, covering 270 brain regions" |
| Brain regions | 270 | same |
| Trials / session (raw) | mean 645, median 602, range 401–1525 | "Recorded sessions lasted on average 645 trials (median of 602, range of 401–1,525)" |
| Overall performance | 81.4 ± 0.4 % correct | "they made correct choices on 81.4 ± 0.4% … of the trials" |
| 0 %-contrast reward rate | 58.7 ± 0.4 % | "On 0% contrast trials … mice gained rewards on 58.7 ± 0.4%" |
| Blocks | 90 unbiased trials, then 20–100-trial blocks, empirical mean 51 | "Blocks lasted for between 20 and 100 trials …(empirical mean of 51 trials)" |
| pLeft values | 0.2 / 0.5 / 0.8 | "ratio of 20:80% (right block) or 80:20% (left block)" |
| Neural time bin | **20 ms**, T = 100, 2-s trials | "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps" |
| Choice alignment/window | stimulus onset, −0.5 s → +1.5 s | "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset" |
| Behaviour sampling | 60 Hz | "wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz" |

### Processing Details
- **Alignment**: `stimOn_times`, window (−0.5, +1.5) s, 20 ms non-overlapping bins → T = 100.
  (The STAR-Methods text also mentions a 50 ms-bin variant for choice and a
  `firstMovement_times`-aligned 20 ms variant for the dynamic behaviours; the *released code*
  and the paper's own summary use the single unified stimulus-onset / 20 ms / T = 100 scheme,
  which is also what the Decoder Task here demands — one alignment for all four variables.)
- **Choice / prior** are constant within a trial; **wheel speed / whisker ME** vary within a trial.
- **Whisker motion energy** = mean absolute difference between adjacent video frames in a
  bounding box between nose tip and eye, from the side camera.
- **Wheel speed** = |velocity| of the 1000 Hz-resampled, low-pass-filtered wheel position.

### Curation Steps

**Neuron curation rules** (data paper, "Neurons and brain regions"):
> "Neurons … were excluded … if they failed one of the three criteria …: amplitude > 50 μV;
> noise cut-off < 20 μV; and refractory period violation. Neurons that passed these criteria
> were termed well-isolated neurons … 75,708 … Final analyses were additionally restricted to
> regions that were designated grey matter in the … Allen Common Coordinate framework."

In the IBL data these three metrics are exactly what `clusters.label` summarises:
`label = (#criteria passed)/3`, so **well-isolated ⇔ `label >= 1`**.

**Trial curation rules** (data paper, "Trials", matching `load_trials_and_mask`):
> "trials were excluded if one of the following trial events could not be detected: choice,
> probabilityLeft, feedbackType, feedback times, stimOn times and firstMovement times. Trials
> were further excluded if the time between stimulus onset and the first movement of the wheel
> … were outside the range of 0.08–2.00 s."

plus the reference code's extra `max_trial_len=10.0` s (`feedback_times - goCue_times`) and
`exclude_nochoice=True`.

### Decoders Trained
The methods paper reports *relative* improvements rather than absolute tables. Absolute numbers
that appear:
| Decoded variable | Reported value | Where |
|---|---|---|
| Choice (single-/multi-session RRR, example sessions, accuracy) | 0.51/0.51, 0.55/0.51, 0.70/0.56, 0.60/0.55 | Fig. 2C scatter labels |
| Choice (BMM-HMM vs baseline, AUC) | 0.72 vs 0.66 vs 0.79 (single / baseline / multi) | Fig. 4B |
| Prior (Pearson corr.) | 0.65 vs 0.05 vs 0.34 | Fig. 4D |
| Choice / stimulus / feedback (BWM paper) | reported as *null-corrected median balanced accuracy* per region (small, ~0.01–0.1 above null) | Figs. 4–7 |
Bottom line: a whole-session (all-region) choice decoder should land well above chance — roughly
0.7–0.9 balanced accuracy — and per-region decoders are much weaker.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Unit QC | `prepare_data` → `load_spiking_data(qc=None)` → **all** clusters (≈898/probe) | `clusters.label` present on every probe; label ≥ 1 gives ≈76–132/probe | Data paper: only the 75,708 **well-isolated** neurons (≈108/probe) are analysed; methods paper says "all neurons, sorted by Kilosort" | **Filter to `label >= 1`.** (a) it is the explicit inclusion criterion of the paper that produced the data; (b) ≈108 good units/probe reproduces the paper's headline number, while "all clusters" would be 889/probe; (c) keeping all 621 k units would make the converted dataset ≈110 GB and put mostly-noise clusters into the decoder. Documented as a deliberate deviation from the reference *code* in favour of the reference *paper*. |
| Bin size | code: 0.02 s everywhere | — | intro + code: 20 ms, T=100; STAR Methods also mentions 50 ms for choice and a `firstMovement` alignment for dynamic behaviours | Use the code's unified 20 ms / stimOn / (−0.5, 1.5) scheme. The Decoder Task fixes the alignment to stimulus onset and requires per-trial *and* time-varying outputs in one dataset, so the unified scheme is the only consistent choice. |
| Session count | 459 eids in `bwm_release.csv`; code loops over all and `try/except`-skips failures | 445 sessions have left- **or** right-camera whisker ME (436 left, 433 right); 459 have wheel + trials | "We apply our models to 433 IBL sessions" | Expect ≈433–445 sessions after requiring whisker ME + wheel + ≥1 good neuron + ≥2 usable trials. Used as a headline consistency check. |
| Whisker camera | left, fall back to right | 436 left, 433 right, 445 either | "near the whisker pad" (side camera) | Follow the code: left first, right as fallback. |
| Brain regions | Beryl mapping of `clusters.acronym` | Beryl maps white matter / unassigned channels to `root` / `void` | "restricted to regions … designated grey matter" | Drop neurons whose Beryl acronym is `root` or `void`. |
| `choice` sign | code stores raw ±1 | verified empirically below | "left = 0, right = 1" required by the Decoder Task | On correct trials with the stimulus on the **left**, `choice == +1` in every session checked (236/236, 141/141, 231/231); with the stimulus on the **right**, `choice == −1` (203/203, 194/194, 194/194). So **`choice == +1` ⇒ reported LEFT, `choice == −1` ⇒ reported RIGHT** → `out = (1 − choice)/2`. |
| `SessionLoader` API | positional `SessionLoader(one, eid)` | installed ibllib 3.x requires keywords | — | Pass a pre-built loader via `sess_loader=`. |
| AP sampling frequency | `raw_electrophysiology(stream=True).fs` | requires streaming raw ephys from the remote server | not a data value | Skipped; not used in any output. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` (all probes merged) | `neural[s][k]` (n_neurons, 100) float32 | keep clusters with `label >= 1` and Beryl ∉ {root, void}; count spikes in 20 ms bins over `stimOn + [−0.5, 1.5)` | `merge_probes`, `load_spiking_data`, `select_brain_regions`, `bin_spiking_data` | raw spike counts, not normalised (reference normalises at decode time) |
| bin index | `input[s][k][0]` "time_from_stim_on_s" | left edge of bin *i* = −0.5 + 0.02·i, i = 0…99 | — | time-varying, seconds, range [−0.5, 1.48] |
| `probabilityLeft` transitions | `input[s][k][1]` "trial_num_in_block" | 0-based index of the trial within its constant-`probabilityLeft` block, computed on the **full** trials table before exclusions, broadcast over the 100 bins | — | per-trial, continuous |
| `trials.choice` | `output[s][k][0]` "choice" | `(1 − choice)/2` → 0 = left, 1 = right | `bin_behaviors` (`choice`) | per-trial, broadcast over time |
| `trials.probabilityLeft` | `output[s][k][1]` "prior_prob_left" | {0.2 → 0, 0.5 → 1, 0.8 → 2} | `bin_behaviors` (`block`) | per-trial, broadcast over time |
| `wheel.velocity` | `output[s][k][2]` "wheel_speed" | `abs(velocity)`, interpolated onto the 100 bin right-edges, then discretised into 3 within-session tertiles | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | time-varying |
| `leftCamera.ROIMotionEnergy` (→ right as fallback) | `output[s][k][3]` "whisker_motion_energy" | interpolated onto the 100 bin right-edges, then discretised into 3 within-session tertiles | `load_target_behavior('*-whisker-motion-energy')`, `get_behavior_per_interval` | time-varying |
| `bwm_release.csv.subject` | `subjects`, `subject_idx` | unique sorted list + index per session | — | 139 mice |
| Beryl acronym per kept cluster | `brain_regions`, `brain_region_idx[s]` | unique sorted list + index per neuron | `list_brain_regions` | grey matter only |

### Key Decisions
1. **Alignment / window / bin size = `stimOn_times`, (−0.5, +1.5) s, 20 ms → T = 100.**
   Exactly the reference `params`; matches the Decoder Task ("align on stimulus onset") and the
   methods paper's "2-s trials, 20-ms bins, T = 100".
2. **Unit QC: keep `clusters.label >= 1`** (well-isolated neurons) and drop Beryl `root`/`void`.
   Justified in Step 4; deviates from the reference *code*'s `qc=None` but follows the data
   paper's stated inclusion criteria and keeps the dataset tractable.
   I do **not** apply the BWM paper's additional "≥5 neurons per region per session and the
   region recorded in ≥2 sessions" rule: that rule exists to give each *region-wise* decoder
   enough data for its statistics, whereas here (as in the reference code, `region='all'`) a
   single decoder uses every neuron in the session, so discarding small regions would only
   throw away signal.
3. **Trial QC = `load_trials_and_mask(..., max_trial_len=10.0)`**, i.e. the reference call
   verbatim: RT ∈ [0.08, 2.0] s, trial length ≤ 10 s, no NaN in the six key events, choice ≠ 0,
   unbiased block **kept** (needed for the 3-class prior output).
4. **Behaviour interpolation follows `get_behavior_per_interval`**: the same coverage checks
   (data must start ≤ 1 bin late and end ≤ 1 bin early) and the same query grid
   `linspace(beg + binsize, end, 100)` (bin right edges). Trials failing the checks are dropped
   from every stream, exactly as `align_spike_behavior` does.
5. **Discretisation of wheel speed and whisker ME into 3 bins = within-session tertiles**
   (33.33 / 66.67 percentiles over all retained trials × time bins of that session).
   Rationale: both signals are in arbitrary, session-specific units — whisker ME depends on
   camera, lighting and ROI size, and wheel speed is heavily right-skewed — so a global
   threshold would put whole sessions in a single class. The reference decoder makes the same
   choice in spirit: `SingleSessionDataset` fits a per-session `StandardScaler` on the training
   trials. Tertiles also give equal class priors, so chance = 1/3 for these outputs and the
   balanced accuracy reported by `train_decoder.py` is directly interpretable.
6. **Per-trial variables are broadcast across the 100 time bins** so that every trial's output is
   a single (4, 100) integer array — `verify_data_format` requires one common `doutput` for all
   trials, and the task says to make outputs time-varying when possible.
7. **Neural data are raw spike counts (float32)**, unnormalised, as cached by the reference.
8. **Session inclusion**: needs wheel + whisker ME, ≥ 1 good grey-matter neuron and ≥ 2 retained
   trials (the decoder needs ≥ 2 trials to split train/test).

### Planned Sanity Checks
- [ ] 139 subjects, ≤ 459 sessions, ≈ 433–445 sessions retained
- [ ] mean good units/probe ≈ 108 (data paper)
- [ ] total good units ≈ 75,708 scaled by the retained sessions
- [ ] raw trials/session: mean ≈ 645, median ≈ 602, min ≈ 401, max ≈ 1525
- [ ] fraction correct ≈ 0.814 on the raw trials table
- [ ] reward rate on 0 %-contrast trials ≈ 0.587
- [ ] pLeft ∈ {0.2, 0.5, 0.8} only; block lengths ∈ [20, 100] after the first (90-trial) block
- [ ] choice balanced ≈ 50/50 overall
- [ ] wheel-speed and whisker-ME class fractions ≈ 1/3 each by construction
- [ ] T = 100 for every trial; input[0] range exactly [−0.5, 1.48]
- [ ] independent re-derivation of spike counts / behaviour values / trial variables straight
      from the ALF files for random (session, trial, neuron, bin) samples (Step 10)

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Runs as
`python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]`
(`--n-workers`, default 16, controls the process pool).

Structure:
| Function | Role |
|---|---|
| `load_session_spikes` | all probes of one session via `SpikeSortingLoader` + `merge_clusters`, merged with the reference's `merge_probes`; also returns the interval during which *every* probe was spiking |
| `select_units` | `clusters.label >= 1` and Beryl acronym ∉ {root, void} |
| `bin_spikes` | vectorised replacement for `get_spike_data_per_interval` |
| `behavior_per_interval` | vectorised replacement for `get_behavior_per_interval` (same rejection rules, same query grid) |
| `load_wheel_speed`, `load_whisker_me` | `SessionLoader`-based, same preference order as `bin_behaviors` |
| `discretize_tertiles` | per-session 33.3/66.7 percentile split into 3 classes |
| `trial_number_in_block`, `map_prior` | task variables |
| `convert_session` | one session end-to-end; raises on any exclusion reason (mirrors the reference's per-session `try/except`) |
| `plot_processing` | the `--show-processing` figure |
| `main` | process pool over sessions, assembly, pickling, summary |

The reference module is *imported* (`sys.path` → `code/code_zhang2025/src`) for
`merge_probes` and `load_trials_and_mask`, so trial curation is literally the reference
implementation rather than a copy.

Code inefficiencies identified:
- The reference bins spikes with a `multiprocessing.Pool` over *trials*, doing a
  `bincount2D` call and an `intersect1d` per trial. With ~650 trials/session this is
  dominated by process and tqdm overhead.
- The reference builds one `scipy.interpolate.interp1d` object per trial per behaviour.
- Spike sorting was being loaded once per probe with its own I/O.

Code speedups added:
- `bin_spikes` does one `np.searchsorted` for all trial boundaries and a single
  `np.bincount` per trial over `unit*NBINS + bin` — no per-trial Python loops over spikes,
  no subprocesses. ~0.7 s/session vs ~10 s for the reference path.
- `behavior_per_interval` builds the whole (n_trials × 100) query grid and calls
  `np.interp` once per behaviour per session.
- Parallelism is moved up one level, to a `ProcessPoolExecutor` over **sessions**
  (24 workers), which keeps the NFS reads (the real bottleneck, ~8 s/session) overlapped.
- `--show-processing` forces serial execution so the two figures are written
  deterministically.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(`--sample` deliberately picks the first session of two *different* mice so that the
subject/region indexing is exercised).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 224 |
| Neurons / session | 61, 163 |
| Subjects | 2 (NYU-11, NYU-12) |
| Sessions / subject | 1, 1 |
| Brain regions | 10 |
| Trials raw | 565, 557 |
| Trials kept | 407, 147 |
| T | 100 for every trial |
| time_from_stim_on range | [−0.50, 1.48] |
| trial_num_in_block range | [0, 89] |
| choice distribution | [0.576, 0.424] |
| prior distribution | [0.403, 0.175, 0.422] |
| wheel_speed distribution | [0.333, 0.333, 0.333] |
| whisker_motion_energy distribution | [0.333, 0.333, 0.333] |

Session 2 keeps only 147/557 trials; the breakdown stored in `session_info` shows the
*reference* mask alone already reduces it to 148 (that mouse had long reaction times), so
this is the reference curation, not a bug in the conversion.

### Processing Plots Review
`processing_<eid>.png` (4×2 panels per session):
- raw raster vs binned counts for one trial, with the total spike count printed in both —
  they agree exactly (786 vs 786), confirming no spikes are dropped or double-counted and
  that the window is `[-0.5, 1.5]` s around the red `stimOn` line;
- wheel speed: raw 1 kHz |velocity| (grey) with the 100 resampled points (blue) lying on
  it, and the two tertile edges as dashed lines, plus the resulting 3-level output;
- whisker ME: raw 60 Hz trace with resampled points on it, tertile edges, 3-level output;
- session level: `probabilityLeft` with the per-trial choice and prior codes overlaid
  (prior code 0/1/2 sits under pLeft 0.2/0.5/0.8 respectively), and `trial_num_in_block`
  sawtoothing with a first tooth of exactly 90 trials and later teeth of 20–100 trials.
No anomalies; no temporal offsets visible.

Verification (`/app/verification_sample_out.txt`): **no errors and no warnings**.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised `np.bincount` binning instead of per-trial multiprocessing | ~10 s → 0.7 s per session |
| single `np.interp` per behaviour instead of per-trial `interp1d` | ~2 s → 0.1 s per session |
| process pool over sessions (24 workers) | 74 min → ~4 min wall clock |

| Step | Time / Session | Estimated Total Time (459 sessions) |
|---|---|---|
| load spike sorting | 8.0 s | 61 min serial |
| load trials | 0.5 s | 4 min serial |
| load behaviour | 1.0 s | 8 min serial |
| bin spikes | 0.7 s | 5 min serial |
| **total serial** | **10.6 s** | **~78 min** |
| **with 24 workers** | — | **~4 min measured (239 s)** |

The sample's 4.6 s/session underestimates the full set (the sample sessions have a single
probe and ~560 trials; the mean over all 459 is 10.6 s), which is why the estimate above
scales from the measured per-step means rather than from the sample wall clock.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.5709 | 0.5968 | 0.500 |
| prior_prob_left | 0.6615 | 0.6564 | 0.333 |
| wheel_speed | 0.5473 | 0.5312 | 0.333 |
| whisker_motion_energy | 0.5524 | 0.5444 | 0.333 |

Loss fell monotonically 1.55 → 0.784 over 200 epochs. Every output is above chance.
Choice is the weakest, as expected: it is scored at *every* time bin including the 0.5 s
before the stimulus appears, when the animal has not yet chosen, and this sample has only
61 and 163 neurons.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full` — 239 s wall clock with 24
workers (vs the ~4 min estimate from Step 7; no re-optimisation needed).

### Output Files
- `converted_data.pkl`: 11.70 GB, 440 sessions
- `verification_full_out.txt`: created (484 lines)
- `conversion_full_out.txt`: created

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | 459 released / 433 analysed | 459 in `bwm_release.csv`, failures skipped | 459 with trials+wheel, 445 with whisker ME | **440** | yes — between the 445 that *have* whisker ME and the paper's 433 |
| Subjects | 139 | 139 | 139 | **135** | yes (4 mice only contributed skipped sessions) |
| Probes | 699 | 699 | 699 | **670** (in the 440 kept sessions) | yes |
| Brain regions | 270 | Beryl, all regions in session | — | **263** | yes |
| Clusters/probe (all) | 889 | n/a (`qc=None`) | 891.2 | **891.2** | yes |
| Good units total / per probe | 75,708 / 108 | n/a | 72,876 / **108.8** in kept sessions | **108.8/probe** | yes |
| Neurons in dataset | — | — | — | 62,650 (86 % of good units are grey matter) | — |
| Neurons/session | — | — | — | mean 142.4, median 124, 7–516 | — |
| Raw trials/session | mean 645, median 602, range 401–1525 | — | — | **mean 646.5, median 602, range 401–1525** | yes |
| Retained trials/session | — | — | — | mean 426.2, median 392, 125–1445 (65.9 % retention) | — |
| Total retained trials | — | — | — | 187,513 | — |
| Fraction correct | 0.814 ± 0.004 | — | — | **0.816 ± 0.003** | yes |
| pLeft values | {0.2, 0.5, 0.8} | `block` = probabilityLeft | {0.2, 0.5, 0.8} | {0, 1, 2} | yes |
| Unbiased-block fraction | 90/645 = 0.139 | — | — | **0.140** | yes |
| Choice balance | ~50/50 | — | — | **left 0.508 / right 0.492** | yes |
| T (bins/trial) | 100 | 100 | — | **100** everywhere | yes |
| time_from_stim_on | −0.5 … +1.5 s | −0.5 … +1.5 s | — | **[−0.50, 1.48]** (bin left edges) | yes |
| trial_num_in_block | blocks 90 then 20–100 | — | — | **0–98** | yes |
| wheel_speed distribution | — | — | — | [0.333, 0.333, 0.333] | by construction |
| whisker ME distribution | — | — | — | [0.333, 0.333, 0.333] | by construction |
| Whisker camera | left, right as fallback | left, right as fallback | 436 left / 433 right available | **433 left, 7 right** | yes |

Nothing is lost silently: every one of the 19 skipped sessions is recorded with its reason
in `data['metadata']['failed_sessions']`, and every retained trial's index in the raw
trials table is recorded in `session_info[s]['kept_trial_idx']`.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output-log verification
`verification_full_out.txt` reports **no errors**. It reports 13 warnings, all of the form
"Session *s*, trial *k*: all neural data is zero".

Two earlier causes were found and **fixed**:

1. *Trials outside the ephys recording.* Session `8c2f7f4d…` had its last three trials at
   t ≈ 1789 s while the spike sorting ended at t = 1779 s: the behaviour outlasted the
   recording, so "zero spikes" meant "not recorded". `load_session_spikes` now returns the
   interval during which **every** probe was producing spikes and
   `convert_session` requires the whole 2-s window to lie inside it (3 trials dropped
   across the dataset).
2. *Sessions with a near-empty population.* Three sessions had 1–3 well-isolated
   grey-matter neurons, so whole trials were silent. A minimum of **5 neurons per session**
   is now required — the same threshold the BWM paper applies to a region within a session
   ("contained at least five well-isolated neurons per session"), applied here to the
   pooled population because the decoder uses the whole session at once.

The 13 remaining warnings **cannot** be fixed, and should not be: they are true
observations, not conversion errors.
- 12 of them are in session 181 (`195443eb…`), which has 10 neurons firing at 3.8 Hz on
  average. Over a 2-s window that is ~76 expected spikes for the population, but the rate
  is very non-stationary, and 12 of its 618 trials (1.9 %) happen to contain none. Dropping
  them would bias the dataset by removing exactly the low-activity trials.
- 1 is in session 200 (`b182b754…`, 147 neurons): checking the raw file directly, that
  2-s window contains **4 spikes in total across all 1,000+ clusters of the probe**, i.e.
  a momentary acquisition dropout well inside the recording span. It is a single trial out
  of 187,513 and is left in place.

### Check 2 — independent sanity checks (`cache/sanity_checks.py`)
Everything below is re-derived from the ALF files with `np.load` / `pd.read_parquet`;
none of `convert_data.py` is imported. All comparisons use `np.allclose`. **All pass.**

| # | Stream | Check | Result |
|---|--------|-------|--------|
| 1 | neural | For 5 random sessions × 3 random trials, recount every spike from `spikes.times.npy`/`spikes.clusters.npy` into 100 bins for all kept neurons (neurons matched by `clusters.uuids.csv`), compare the full (n_neurons, 100) matrix | `max|diff| = 0.0` in all 15 |
| 2 | neural | Point spot-checks: trial 5 / neuron 3 / bin 10, and the busiest (neuron, bin) of the same trial | raw = converted (0 and 7 spikes) |
| 3 | neural | Every kept neuron has `label >= 1` in `clusters.metrics.pqt` | pass |
| 4 | input | `input[0]` equals −0.50, −0.48, …, 1.48 for 40 sampled trials | pass |
| 5 | input | `trial_num_in_block` recomputed from `probabilityLeft` in the raw trials table, 5 sessions, all trials | pass; first block is exactly 90 trials in each |
| 6 | output | `choice` = (1 − `trials.choice`)/2 and constant across the 100 bins, 5 sessions | pass |
| 7 | output | `prior` = {0.2→0, 0.5→1, 0.8→2} and constant across bins, 5 sessions | pass |
| 8 | output | Semantics: on correct trials the decoded label must equal the *stimulus* side (side = the non-NaN contrast column), 8 sessions, 2,279 correct trials | pass |
| 9 | output | Wheel speed re-derived from `_ibl_wheel.position/timestamps.npy` (own `interpolate_position` + `velocity_filtered`), resampled and re-binned, 4 sessions | 100.000 % of bins identical |
| 10 | output | The stored tertile edges equal the 33.3/66.7 percentiles of the recomputed values | `np.allclose`, 4 sessions |
| 11 | output | Whisker ME re-derived from `{left,right}Camera.ROIMotionEnergy.npy` + `_ibl_*Camera.times.npy`, 4 sessions | 100.000 % of bins identical |
| 12 | output | Every session's 3 classes are within 0.05 of 1/3 for both dynamic outputs | worst deviation 0.0001 |

Check 12 originally **failed** (worst deviation 0.667): session `5b44c40f…` has a broken
whisker ROI whose motion energy is exactly 0 for 78 % of in-trial samples, so both tertile
edges came out at 0.0 and every bin was assigned to one class. `convert_session` now
rejects a session whose tertile edges are not strictly increasing; that is the only session
affected.

Check 8 also failed on the first run — that was a bug in the *check*, which used
`contrastRight > 0` to identify the stimulus side and therefore mislabelled every
0 %-contrast trial (which carries contrast 0 on its assigned side and NaN on the other).
Using `isfinite(contrastRight)` it passes on all 2,279 correct trials tested.

### Check 3 — reference-code comparison
| Step | Reference | This script | Same? |
|---|---|---|---|
| (a) loading | `prepare_data`: `SpikeSortingLoader.load_spike_sorting` + `merge_clusters` per probe, `merge_probes`; `load_trials_and_mask`; `load_target_behavior` via `SessionLoader` | identical, and `merge_probes`/`load_trials_and_mask` are *imported* from the reference module | yes (the reference's `raw_electrophysiology(...).fs` metadata call is skipped — it streams raw binary and feeds no data value) |
| (b) neuron filtering | `qc=None` → all clusters; Beryl regions = all regions present | `label >= 1` **and** Beryl ∉ {root, void}; plus ≥5 neurons/session | **deliberate difference** — see Step 4. Follows the data paper's stated neuron inclusion criteria; reproduces its 108 good units/probe; keeps the dataset at 11.7 GB instead of ~110 GB |
| (c) temporal alignment | `stimOn_times + (-0.5, 1.5)` | identical | yes |
| (d) binning | `bincount2D(xbin=0.02, xlim=[t_beg,t_end])`, truncated to `ceil(2/0.02)=100` bins; spikes with `t_beg <= t < t_end` | `np.bincount` on `floor((t-t_beg)/0.02)` over the same spike selection, clipped to 100 bins | yes — verified against the raw files in Check 2 |
| (e) input construction | reference caches no decoder inputs (its models take neural activity only) | time from stimulus onset + trial number in block, as the Decoder Task specifies | required by the task |
| (f) output construction | `choice` (±1), `block` (=probabilityLeft), `wheel-speed`, `whisker-motion-energy`, all stored continuously; standardised/one-hot-encoded at decode time | same four variables, same sources, same interpolation grid; recoded to the integer classes the Decoder Task specifies, with the continuous pair discretised into per-session tertiles | yes for the variables and their extraction; the discretisation is required because the target format only accepts categorical outputs |
| trial curation | `load_trials_and_mask(max_trial_len=10.0)` + behaviour-coverage mask via `align_spike_behavior` | the same imported call and the same coverage rules, **plus** the ephys-coverage rule from Check 1 | superset; the extra rule removes 3 trials in 187,513 |

### Check 4 — key statistics vs the papers
See the table in Step 9. Every quantity the papers state is reproduced:
clusters/probe 891.2 vs 889; good units/probe 108.8 vs 108; raw trials mean 646.5 vs 645,
median 602 vs 602, range 401–1525 vs 401–1525; fraction correct 0.816 ± 0.003 vs
0.814 ± 0.004; unbiased-block fraction 0.140 vs 90/645 = 0.139; 440 sessions vs the methods
paper's 433 (and the 445 that have any whisker motion energy at all); 263 Beryl regions vs
270; 135 mice vs 139.

The two small shortfalls are explained rather than unexplained: 19 sessions are skipped, 13
of them because no whisker-motion-energy dataset exists for that session at all (445 of 459
sessions have one), and the remaining 6 for the documented quality reasons. 263 < 270
regions follows from those 19 missing sessions plus the grey-matter restriction.

### Check 5 — edge cases
- **Trial/session boundaries**: the analysis window is `[stimOn−0.5, stimOn+1.5)`, half-open,
  so bin *i* covers `[−0.5+0.02i, −0.48+0.02i)` and no spike is counted twice; a spike
  exactly at the window end is excluded, matching `times < t_end` in the reference.
  Spikes exactly at the right edge of the last bin are impossible, but the index is clipped
  anyway to guard against floating-point drift.
- **First/last trial of a session**: handled by the ephys- and behaviour-coverage masks
  (Check 1); the wheel/whisker checks reject trials whose trace starts >1 bin late or ends
  >1 bin early, which catches the first and last trials of most sessions.
- **First/last block**: `trial_number_in_block` is computed on the *raw* trials table and
  starts at 0 on the first trial of the session; the first block is the 90-trial unbiased
  block, verified in Check 5 of the sanity script.
- **NaN handling**: trials with NaN in any of the six key events are removed by the
  reference mask; NaN anywhere in a resampled behaviour trace rejects the trial; NaN
  camera timestamps are dropped before interpolation; `probabilityLeft` outside
  {0.2, 0.5, 0.8} and `choice == 0` reject the trial.
- **Sessions with one vs two probes**: `merge_probes` mutates `spikes['clusters']` in place,
  so the single-probe path bypasses it and just resets the cluster index; the ephys-coverage
  interval is the *intersection* over probes, so a probe that stops early does not leave
  another probe's trials mislabelled.
- **Degenerate behaviour**: tied tertile edges reject the session (Check 2).
- **Empty results**: a session with <5 neurons, <2 usable trials, or no whisker ME raises and
  is recorded in `failed_sessions` rather than emitting an empty session (which would trip
  the decoder's "session has 0 trials" warning).
- **Cluster-id remapping**: the lookup array is sized `max(spikes['clusters'])+2`, which is
  robust to a spike referencing a cluster id beyond the clusters table.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
-> `/app/train_decoder_full_out.txt` (ran to completion on the GPU; ~20 min).

### Training Progress
- Loss decreasing: **Yes**, monotonically - 2.36 (epoch 1) -> 2.05 (4) -> 1.68 (10) ->
  1.37 (20) -> 1.10 (40) -> 0.95 (60) -> 0.86 (80) -> ... -> final test loss 0.771.

### Decoder Results (Full, 440 sessions, 187,513 trials, 18.75 M scored timepoints)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|--------|-------------|--------|-------|
| choice | 0.500 | 0.6358 | **0.6160** | 1.23x chance; scored at every bin, including the 25 pre-stimulus bins where choice is not yet decodable |
| prior_prob_left | 0.333 | 0.6734 | **0.6560** | 1.97x chance |
| wheel_speed | 0.333 | 0.6125 | **0.6059** | 1.82x chance |
| whisker_motion_energy | 0.333 | 0.5969 | **0.5914** | 1.77x chance |

`sample_trials.png` and `predictions.png` were written by the run.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 - accuracy vs chance
| Output | Validation acc | Uniform chance | Ratio | Majority-class chance |
|---|---|---|---|---|
| choice | 0.6160 | 0.500 | 1.23x | 0.508 |
| prior_prob_left | 0.6560 | 0.333 | 1.97x | 0.442 |
| wheel_speed | 0.6059 | 0.333 | 1.82x | 0.333 |
| whisker_motion_energy | 0.5914 | 0.333 | 1.77x | 0.335 |

Nothing is at or below chance. Three of the four are well above 1.5x chance. **Choice sits
at 1.23x chance**, so per the protocol I investigated whether the conversion could be
responsible.

**Investigation** (`cache/time_resolved_choice.py`, figure
`cache/time_resolved_decodability.png`). I fitted an *independent* cross-validated
logistic regression (5-fold, balanced, 40 PCs) separately at each of the 100 time bins, on
12 sessions with >=200 neurons - a decoder that shares nothing with `train_decoder.py`.
Mean balanced accuracy vs time from stimulus onset:

| t (s) | -0.50 | -0.20 | 0.00 | +0.10 | **+0.20** | +0.40 | +0.80 | +1.40 |
|---|---|---|---|---|---|---|---|---|
| choice | 0.516 | 0.525 | 0.532 | 0.663 | **0.786** | 0.725 | 0.616 | 0.546 |
| prior | 0.593 | 0.581 | 0.583 | 0.644 | 0.682 | 0.669 | 0.628 | 0.582 |
| wheel | 0.500 | 0.551 | 0.524 | 0.531 | 0.598 | 0.510 | 0.488 | 0.503 |
| whisker | 0.486 | 0.471 | 0.488 | 0.485 | 0.548 | 0.544 | 0.536 | 0.506 |

This is exactly the signature of a correctly aligned dataset, and it rules out a conversion
bug:
- **choice is flat at chance (0.52) for the whole 0.5 s before the stimulus and rises
  abruptly within one or two bins of t = 0**, peaking at **0.786 at +0.20 s** - the IBL
  median first-wheel-movement time. A misalignment, a shuffled label or a wrong choice sign
  convention could not produce a step at exactly t = 0.
- **prior is already elevated (~0.58) before the stimulus** and rises only modestly after
  it, which is what a slowly varying block variable should look like.
- the average of the choice curve over all 100 bins is **0.602**, so the provided decoder's
  0.616 is *above* what an independently tuned per-timepoint decoder achieves on the same
  data. The conversion is not leaving accuracy on the table.

The ceiling on the reported number is structural, not a defect: choice is constant within a
trial and is scored at every bin, so a quarter of the scored timepoints precede the stimulus
and a further large fraction precede the wheel movement. Half a trial at chance plus half a
trial at ~0.75 averages to ~0.62. Reaching 1.5x chance (0.75 overall) would require choice
to be decodable before the mouse has seen the stimulus.

### Check 2 - accuracy comparison to the papers
| Variable | This dataset | Papers | Comparison |
|---|---|---|---|
| choice, **per-trial peak** | **0.786** (independent per-timepoint LR at +0.2 s, 12 sessions) | Fig. 4B: AUC 0.72 (single-session BMM-HMM) / 0.66 (linear baseline) / 0.79 (multi-session); Fig. 2C: accuracy 0.51-0.70 single- vs multi-session RRR on 5-region subsets | at the top of the reported range; the papers decode from 5 selected regions, here all grey-matter neurons of the session are used |
| choice, per-timepoint mean | 0.6160 | not reported (the papers decode one value per trial) | n/a - averaging over pre-stimulus bins is specific to this task's format |
| prior | 0.6560 (3-class balanced acc.) | Fig. 4D: Pearson r = 0.65 (multi-trial LG-AR1) / 0.05 (single-session baseline) / 0.34 (oracle) | different metric; 0.656 against a 0.333 chance is a large effect, consistent with the papers finding prior strongly decodable |
| wheel speed | 0.6059 (3-class) | R^2 (Fig. 5C); BWM: "uncorrected R2 for decoding speed were high" | different metric; both agree wheel speed is decodable well above chance |
| whisker motion energy | 0.5914 (3-class) | R^2 (Fig. 5D) | same |
| BWM region-level choice | - | null-corrected median balanced accuracy ~0.01-0.1 above null per region | a whole-session decoder is far stronger than any single region, as expected |

No figure reported in either paper exceeds what this dataset achieves under a comparable
protocol. The metrics genuinely differ (the papers report AUC / Pearson r / R^2 for
per-trial or per-timestep regression, while `train_decoder.py` reports 3-class balanced
accuracy at every timepoint), so the per-trial peak accuracy is the only apples-to-apples
comparison - and it matches.

### Check 3 - train vs validation gap
| Output | Train | Validation | Train/Val |
|---|---|---|---|
| choice | 0.6358 | 0.6160 | 1.03 |
| prior_prob_left | 0.6734 | 0.6560 | 1.03 |
| wheel_speed | 0.6125 | 0.6059 | 1.01 |
| whisker_motion_energy | 0.5969 | 0.5914 | 1.01 |

All far below the 1.5x threshold: no overfitting and no data leakage. Leakage would be a
real risk for the per-session tertile thresholds, which are fitted on all of a session's
trials rather than on the training split; the 1.01 ratio for those two outputs shows it has
no measurable effect, as expected when two scalars are estimated from ~40,000 values.

### Debugging steps applied to the weakest output (choice)
1. *Output values verified against raw data* - Step 10 Check 2 rows 6 and 8: choice
   re-derived from `_ibl_trials.table.pqt` for 5 sessions, and its semantics confirmed on
   2,279 correct trials across 8 further sessions.
2. *Temporal alignment* - the time-resolved curve above (step at t = 0), plus the
   `processing_<eid>.png` panels, which put the raw raster, the binned counts and the raw
   and resampled behaviour on a common axis with `stimOn` at 0.
3. *Output variation* - choice is 50.8 / 49.2 %; prior 41.8 / 14.0 / 44.2 %; both dynamic
   outputs 33.3 % per class. No output is dominated by one class.
4. *Neural filtering* - reproduces the paper's 108 good units per probe (Step 9 table).
5. *Processing vs the reference code* - Step 10 Check 3.

### Issues Found and Resolved (Steps 10 and 12)
| Issue | How found | Resolution | Re-check |
|---|---|---|---|
| 3 trials whose 2-s window ran past the end of the ephys recording were stored as all-zero spike counts | `verify_data_format` warnings | require the window to lie inside the interval during which every probe was spiking | that session's warnings gone |
| 3 sessions with 1-3 well-isolated grey-matter neurons produced many all-zero trials | same | require >=5 neurons per session (the BWM per-session threshold) | warnings 38 -> 13 |
| session `5b44c40f...` has a broken whisker ROI (motion energy exactly 0 in 78 % of in-trial samples), so both tertile edges were 0.0 and the output was constant | sanity check 12 | reject sessions whose tertile edges are not strictly increasing | worst class-fraction deviation 0.667 -> 0.0001 |
| sanity check 8 mislabelled 0 %-contrast trials | first sanity run | fixed the *check* (`isfinite(contrastRight)` rather than `> 0`) | passes on 2,279 trials |
| `SessionLoader(one, eid)` positional call in the reference module fails on the installed ibllib | first prototype run | pass a pre-built loader through `sess_loader=` | conversion runs |
| `--sample` used two sessions from the same mouse | sample summary showed 1 subject | `--sample` now picks two different mice | 2 subjects |

After every fix the full conversion, the format verification, the sanity-check suite and the
statistics comparison were re-run from scratch; all numbers quoted in Steps 9-12 come from
the final run.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created
- [x] `cache/` folder created (investigation scripts and their outputs, plus
      `README_CACHE.md`)
- [x] All files organized

### Final file inventory
| File | Contents |
|---|---|
| `convert_data.py` | the conversion |
| `converted_data.pkl` | 11.70 GB, 440 sessions, 187,513 trials, 62,650 neurons |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | this file |
| `README.md` | user-facing description |
| `conversion_{sample,full}_out.txt` | conversion logs |
| `verification_{sample,full}_out.txt` | `--verify-only` logs |
| `train_decoder_{sample,full}_out.txt` | decoder logs |
| `processing_<eid>.png` (x2) | per-step processing figures |
| `sample_trials.png`, `predictions.png` | written by the full decoder run |
| `cache/` | sanity checks, statistics checks, time-resolved analysis, prototype |
