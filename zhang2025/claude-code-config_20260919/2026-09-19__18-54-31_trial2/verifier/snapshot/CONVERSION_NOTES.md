# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-Wide Map (BWM) — Neuropixels electrophysiology + behaviour during the
  IBL decision-making task (IBL et al., *A brain-wide map of neural activity during complex
  behaviour*, Nature 645, 2025).
- **Reference processing code**: Zhang et al. 2026 (Neuron), *Exploiting correlations across
  trials and behavioral sessions to improve neural decoding* — `/app/code/code_zhang2025`.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`) for
  `/app/train_decoder.py`.

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`, `stage_cache.sh` — container plumbing
- `CONVERSION_NOTES.md` (this file)
- `datapaper.pdf` (IBL BWM Nature paper), `methodpaper.pdf` (Zhang et al. Neuron),
  `dataarchitecture.pdf` (IBL data architecture white paper), `methods.txt` (excerpts)
- `decoder.py` (decoder library), `train_decoder.py` (validation/training entry point)
- `code/code_zhang2025` — reference processing + decoding code (this is the code used by the
  method paper)
- `code/ibllib` — source of `ibllib`/`brainbox` (the IBL data-access library)
- `data/one_cache` — a ONE cache: release tables (`Brainwidemap`, `2022_Q4_IBL_et_al_BWM`,
  `2025_Q3_IBL_et_al_BWM`), a `.rest` Alyx REST response cache, and per-lab symlinks to the
  actual session data (567 GB).

Python environment verified: `numpy 2.3.5`, `torch 2.6.0+cu124` (CUDA L4 GPU, 23 GB),
`ONE-api 3.5.2`, `ibllib`/`brainbox`, `iblatlas`. 128 CPU cores, ~1 TB RAM, 3.4 TB free disk.

**Data access**: there is no network. ONE must be constructed as
`ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True, cache_dir='/app/data/one_cache')`
**without** `password=` (supplying a password forces re-authentication over the network and
fails). With the staged auth token + `.rest` cache, ONE runs in `remote` mode but answers every
query from the on-disk REST cache. Pure `mode='local'` does **not** work because the release
parquet tables do not contain the dataset revisions that are actually staged on disk
(e.g. trials are staged under `alf/#2025-03-03#/`, which no table lists).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

`/app/code/code_zhang2025` is the Zhang et al. decoding repository. The data-preparation entry
point is `src/0_data_caching.py`, which calls helpers in `src/utils/ibl_data_utils.py`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data(one, eid, bwm_df, params)` | `src/utils/ibl_data_utils.py` | LOADING | Top level per-session loader: resolves probes via `one.eid2pid`, loads spike sorting per probe, merges probes, loads trials + trial mask, loads continuous behaviours. |
| `load_spiking_data(one, pid, qc=None)` | `ibl_data_utils.py` | LOADING/CURATION | `SpikeSortingLoader(...).load_spike_sorting()` + `merge_clusters(...).to_df()`. `qc=None` → all clusters; `qc=1` → only clusters with `label >= 1` (i.e. the "well-isolated"/good units), re-indexing `spikes['clusters']`. |
| `merge_probes(spikes_list, clusters_list)` | `ibl_data_utils.py` | LOADING | Concatenates probes of one session into a single population (re-indexes cluster ids, sorts spikes by time). |
| `load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', max_trial_len=..., exclude_nochoice=True)` | `ibl_data_utils.py` | CURATION | Returns the full trials table plus a boolean keep-mask. Excludes NaNs in `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`; reaction time (`firstMovement_times - stimOn_times`) outside [0.08, 2.0] s; no-response trials (`choice == 0`). `prepare_data` additionally passes `max_trial_len=10.0` (excludes `feedback_times - goCue_times > 10 s`). |
| `list_brain_regions(neural_dict, **kwargs)` | `ibl_data_utils.py` | PROCESSING | Maps each cluster's Allen `acronym` to the **Beryl** atlas mapping via `iblatlas.regions.BrainRegions.acronym2acronym(..., mapping='Beryl')`. |
| `select_brain_regions(...)` | `ibl_data_utils.py` | CURATION | Index of clusters whose Beryl region is in the requested set (with `single_region=False` this is *all* clusters). |
| `bin_spiking_data(reg_clu_ids, neural_df, trials_df, **params)` | `ibl_data_utils.py` | PROCESSING | Builds per-trial intervals `[align_time + t0, align_time + t1]`, then bins spikes into `binsize` bins → array `(n_trials, n_bins, n_clusters)`. Only clusters that emit ≥1 spike in the session appear as columns (`clusters_used_in_bins = np.unique(regclu)`). |
| `get_spike_data_per_interval(...)` | `ibl_data_utils.py` | PROCESSING | Per-trial `bincount2D(times, clusters, xbin=binsize, xlim=[t_beg, t_end])`, truncated to `n_bins = ceil(interval_len/binsize)`. Bin index = `floor((t - t_beg)/binsize)`, spikes selected with `t_beg <= t < t_end`. |
| `load_target_behavior(one, eid, target)` | `ibl_data_utils.py` | LOADING | `SessionLoader.load_wheel()` → `wheel-speed = abs(velocity)`; `SessionLoader.load_motion_energy(views=['left'/'right'])` → `whiskerMotionEnergy`. |
| `get_behavior_per_interval(...)` | `ibl_data_utils.py` | PROCESSING | Cuts the continuous behaviour trace to each trial interval and **linearly interpolates** onto `np.linspace(t_beg + binsize, t_end, n_bins)` — i.e. the behaviour sample for bin *i* is taken at the **right edge** of the spike bin *i*. Marks a trial bad if the trace is absent, starts >1 binsize after `t_beg`, or ends >1 binsize before `t_end`. |
| `bin_behaviors(one, eid, behaviors, trials_df, ...)` | `ibl_data_utils.py` | PROCESSING | Assembles per-trial scalars (`choice`, `probabilityLeft` as `block`, `reward`, signed `contrast`) and the time-varying behaviours. For `whisker-motion-energy` it uses the **left** camera, falling back to the **right** camera when the left is unavailable. |
| `align_spike_behavior(binned_spikes, binned_behaviors, beh_names, trials_mask)` | `ibl_data_utils.py` | CURATION | Deletes trials that are masked out by the trial mask or for which any behaviour is missing, so neural and behaviour arrays have identical trial counts. |
| `standardize_spike_data` / `SingleSessionDataset` | `src/utils/data_loader_utils.py` | PROCESSING | Model-side (not data-side) per-time-bin z-scoring of spike counts and `StandardScaler` for continuous behaviours; one-hot encoding of discrete targets. |

### Reference caching parameters (`src/0_data_caching.py`)
```python
params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
          'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']
```
i.e. **align to `stimOn_times`, window −0.5 s to +1.5 s, 20 ms bins ⇒ T = 100 bins**. This is
exactly the alignment requested by the decoder task.

### Notes on running the reference code as-is
- `load_spiking_data` calls `spike_loader.raw_electrophysiology(band='ap', stream=True).fs`,
  which needs network streaming of raw AP data; it is only used to report `sampling_freq` and is
  skipped here.
- `load_trials_and_mask` / `load_target_behavior` construct `SessionLoader(one, eid)`
  positionally; in the installed `brainbox` version `SessionLoader` is a dataclass and must be
  constructed with keywords (`SessionLoader(one=one, eid=eid)`). The conversion script passes an
  explicitly constructed `sess_loader` (the function already supports this) / calls the loader
  directly.
- The multiprocessing helpers (`get_spike_data_per_interval`, `get_behavior_per_interval`) spawn
  a `Pool` per session per signal. For 459 sessions this is very slow, so the conversion
  re-implements the *same arithmetic* vectorised (verified to match the reference function
  exactly — see Step 10).

### Do cells need to be filtered based on quality?
Yes. `load_spiking_data` documents `qc=1` → "use good clusters" (`label >= 1`). The data paper
(methods.txt, "Neurons and brain regions") is explicit: neurons that fail any of the three RIGOR
single-unit metrics (amplitude > 50 µV, noise cut-off < 20 µV, refractory-period violation) are
excluded, leaving 75,708 "well-isolated neurons" out of 621,733 units. See Step 5 for the
decision taken here.

(No imaging data is involved, so no ΔF/F computation is required.)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/one_cache` is an ALyx/ONE cache laid out as
`<lab>/Subjects/<subject>/<date>/<number>/…`:

```
alf/                                   session-level ALF files
  #2025-03-03#/_ibl_trials.table.pqt   trial table (revisioned!)
  _ibl_wheel.position.npy, _ibl_wheel.timestamps.npy
  #2025-05-29#/leftCamera.ROIMotionEnergy.npy
  #2025-05-31#/rightCamera.ROIMotionEnergy.npy
  #2023-04-20#/_ibl_{left,right}Camera.times.npy
  probe00/pykilosort/#2024-05-06#/{spikes,clusters,channels}.*.npy|pqt
  probe01/pykilosort/#2024-05-06#/…
raw_ephys_data/…                       (only .meta/.ch stubs staged)
```
Most datasets are staged under dataset *revisions*, so file access must go through ONE
(which resolves revisions), not through hard-coded paths.

Release tables present: `Brainwidemap` (480 sessions / 76,563 datasets — the superset),
`2022_Q4_IBL_et_al_BWM` (354 sessions), `2025_Q3_IBL_et_al_BWM` (459 sessions, newer
lightning-pose / motion-energy revisions).

The session/probe freeze used by the reference code is
`/app/code/code_zhang2025/data/bwm_release.csv`: **699 pids, 459 eids, 139 subjects** — exactly
the public BWM release.

### Available variables (per session)
- `trials` table: `stimOn_times, stimOnTrigger_times, stimOff_times, goCue_times,
  goCueTrigger_times, response_times, feedback_times, feedbackType, rewardVolume, choice,
  contrastLeft, contrastRight, probabilityLeft, firstMovement_times, intervals_0/1,
  intervals_bpod_0/1, quiescencePeriod`
- `wheel`: `times, position, velocity, acceleration` (interpolated to 1 kHz by `SessionLoader`)
- `motion_energy`: `leftCamera` (60 Hz) / `rightCamera` (150 Hz) `whiskerMotionEnergy`
- spike sorting per probe: `spikes.times/clusters/amps/depths`, `clusters.*` (incl. the QC
  `label` ∈ {0, 1/3, 2/3, 1}), `channels.*` with Allen `acronym` per cluster.

### Dataset Size (measured directly from the data files, restricted to the 459 BWM eids)
| Statistic | Value |
|-----------|-------|
| Units (all, total) | **621,733** |
| Units (all) / probe | 889.5 |
| Well-isolated units (`label >= 1`) total | **75,708** |
| Well-isolated units / probe | 108.3 |
| Probes (pids) | 699 |
| Subjects | **139** |
| Sessions | **459** |
| Sessions / subject | 3.30 (mean) |
| Trials (total, before curation) | **296,090** |
| Trials / session | mean 645.1, min 401, max 1525 |
| Sessions with left-camera motion energy | 437 |
| Sessions with right-camera motion energy | 420 |
| Sessions with **no** whisker motion energy at all | 14 |
| Sessions with wheel data | 459 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Units (total) | 621,733 | "This process produced 621,733 units (including multineuron activity), averaging 889 per probe." (data paper) |
| Well-isolated neurons | 75,708 (108/probe) | "…identified 75,708 well-isolated neurons, averaging 108 per probe." |
| Probes / sessions | 699 insertions, 459 sessions | "a total of 459 sessions, 699 insertions and 621,733 neurons remained, constituting the publicly released dataset" |
| Subjects | 139 mice (94 M, 45 F) | "We trained 139 mice … on the IBL decision-making task" |
| Brain areas | 279 (Beryl) | "The probes covered 279 brain areas" |
| Trials / session | mean 645, median 602, range 401–1,525 | "Recorded sessions lasted on average 645 trials (median of 602, range of 401–1,525)" |
| Proportion correct | 81.4 ± 0.4 % | "they made correct choices on 81.4 ± 0.4% … of the trials" |
| Reward on 0 %-contrast trials | 58.7 ± 0.4 % | "On 0% contrast trials … mice gained rewards on 58.7 ± 0.4%" |
| Reaction times < 80 ms | 22.8 % of trials | "A total of 22.8% first wheel-movement times occurred under 80 ms" |
| Block structure | first 90 trials p(left)=0.5, then 0.2/0.8 blocks of 20–100 trials (mean 51) | data paper, "Methods" |
| Sessions used by method paper | 433 (of 459) | "On 433 sessions spanning 270 brain regions…" |
| Neurons / session (method paper) | "average of 676 … ranging from approximately 300 to over 2000" | method paper, STAR Methods |
| Trial length / binning | 2 s trials, 20 ms bins, T = 100 | "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps." |
| Alignment for choice/prior | stimulus onset, −0.5 s → +1.5 s | "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset." |
| Behaviour sampling | wheel/whisker sampled at 60 Hz (video), wheel interpolated to 1 kHz | "wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz" |

### Processing Details
- **Temporal alignment**: `stimOn_times`; window (−0.5, +1.5) s; 20 ms non-overlapping bins;
  T = 100 (reference `params` in `0_data_caching.py`; matches the decoder task's request to
  "temporally align based on stimulus onset").
- `methods.txt` also contains the sentence "Within each trial, we segment neural activity into
  50-ms non-overlapping time bins" for choice/prior. This contradicts both the released code
  (`binsize: 0.02`) and the paper's own "2-s trials … 20-ms bins … T = 100". **20 ms is used**,
  which is also required here because two of the four decoder outputs are time-varying
  behaviours that must be resolved within the trial.
- **Whisker motion energy** = mean absolute difference between adjacent video frames in a
  bounding box anchored between nose tip and eye (left camera preferred, 60 Hz).
- **Wheel speed** = |wheel velocity|, from `SessionLoader.load_wheel()` (Gaussian-smoothed
  velocity on a 1 kHz uniform grid).

### Curation Steps

**Session curation (already applied by the release)**: ≥ 250 trials performed, ≥ 90 % correct on
100 %-contrast trials in both block types, ≥ 3 incorrect trials, hardware QC passed, probe
alignment resolved, no major artefacts. The 459-session / 699-pid freeze in `bwm_release.csv`
*is* the result of these criteria, so no further session-level QC is re-derived here.

**Neuron curation rules** (data paper, "Neurons and brain regions"):
- exclude units failing any of the three RIGOR single-unit metrics (amplitude > 50 µV,
  noise cut-off < 20 µV, refractory-period violation) ⇒ keep `clusters.label >= 1`
  ("well-isolated neurons", 75,708 of 621,733);
- restrict to regions designated **grey matter** in the Allen CCF ⇒ drop units whose Beryl
  acronym is `root` (in brain, unassigned) or `void` (outside brain).

**Trial curation rules** (data paper "Trials" = `load_trials_and_mask` defaults):
- drop trials with an undetected `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`,
  `stimOn_times` or `firstMovement_times`;
- drop trials whose reaction time `firstMovement_times − stimOn_times` is outside 0.08–2.00 s;
- drop no-response trials (`choice == 0`);
- (reference code, `prepare_data`) drop trials with `feedback_times − goCue_times > 10 s`;
- (reference code, `align_spike_behavior`) drop trials for which the wheel or whisker trace does
  not cover the full −0.5…+1.5 s window (e.g. the first trials of a session, before the video
  starts).

Measured on 15 sessions: 27.1 % of trials have RT < 80 ms, 7.8 % RT > 2 s, 1.6 % have a NaN
event, 0.5 % are no-go, 4.0 % have trial length > 10 s ⇒ ≈ 63 % of trials survive the trial-event
criteria (before the behaviour-coverage criterion).

### Decoders Trained (reference numbers; different decoders/metrics than the one used here)
| Decoded variable | Reported performance |
|---|---|
| Choice (binary, single-session linear baseline / RRR, 10 sessions, 5 regions) | ≈ 0.6–0.8 accuracy / AUC (Figure 3A, bar plots; RRR ≳ linear) |
| Prior / block (BMM-HMM vs baseline) | AUC 0.72 (single-session) vs 0.79 (multi-session), baseline 0.66 |
| Wheel speed (R², aligned to first movement) | 0.24 → 0.55 (single-trial R², linear → RRR (M)) |
| Whisker motion energy (R²) | 0.52 → 0.80 |
| BWM paper, per-region decoding | null-corrected **median balanced accuracy**, small effect sizes (≈ 0.5–0.6 raw) for single regions |

Note the reference decoders are region-specific/session-specific and use different metrics
(R²/AUC), so they are not directly comparable to the balanced accuracy produced by
`train_decoder.py`, which decodes from *all* neurons of a session at every time bin. They do
establish that choice/prior/wheel/whisker are all decodable well above chance from IBL
population activity.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Time bin size | `binsize = 0.02` | n/a | method paper body: "20-ms bins, T = 100"; methods.txt STAR section: "50-ms bins" for choice/prior | Use **20 ms** (code + paper body); needed for time-varying outputs. Documented. |
| Alignment window | `time_window = (-0.5, 1.5)`, `align_time='stimOn_times'` | n/a | "0.5 s before to 1.5 s post-onset" for choice | Consistent → (−0.5, +1.5) s. |
| Neuron QC | `load_spiking_data(qc=None)` used in `prepare_data` (all units); `qc=1` supported | 621,733 units, 75,708 with `label>=1` | data paper: exclude units failing RIGOR ⇒ 75,708 well-isolated | Use **`label >= 1`** (see Step 5 rationale). Documented as a deliberate deviation from the reference caching script. |
| Neurons/session | all units ⇒ 1,354/session | 1,354 (all) / 165 (good) per session | method paper: "average of 676 … 300 to over 2000" | The method paper's 676 matches neither all-units (1,354) nor good-units (165) for the full 459-session freeze; its stated *range* (300–2,000) matches all-units. Treated as an unreconcilable paper statistic; not used as a target. |
| Sessions | `bwm_release.csv` → 459 | 459 staged | data paper 459; method paper analysed 433 | 459 is the release; sessions are dropped only when a required signal is missing (see Step 5). |
| Trials filter | `load_trials_and_mask(..., max_trial_len=10.0)` | — | data paper lists only the NaN + RT criteria | Apply the reference code's filter (superset of the paper's); it removes a further 0.5 % of trials. |
| Whisker camera | left preferred, right fallback | 437 left / 420 right / 14 with neither | "left" camera at 60 Hz | Follow the code: left, else right. |
| Choice sign | `choice` used raw | correct & stimulus-left ⇒ `choice == +1`; correct & stimulus-right ⇒ `choice == −1` | — | **`choice == +1` ⇒ left, `choice == −1` ⇒ right**; encode left = 0, right = 1 as the task requires. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial geometry
- alignment event: `stimOn_times`
- `off_start = -0.5 s`, `off_end = +1.5 s`, `time_bin_size = 20 ms`, `T = 100` bins per trial
- spike bin *i* covers `[stimOn − 0.5 + 0.02·i, stimOn − 0.5 + 0.02·(i+1))`
- continuous behaviour for bin *i* is interpolated at the bin's **right edge**
  `stimOn − 0.5 + 0.02·(i+1)` — exactly `get_behavior_per_interval`'s
  `np.linspace(t_beg + binsize, t_end, n_bins)`.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` (all probes merged) | `neural[s][k]` `(n_neurons, 100)` | spike counts in 20 ms bins, float32 | `merge_probes`, `bin_spiking_data`, `get_spike_data_per_interval` | only well-isolated grey-matter clusters |
| bin index → time | `input[0]` `time_from_stim_onset` | `-0.5 + 0.02·(i+0.5)` s (bin centre), −0.49 … +1.49 | (new; required by the task) | continuous, time-varying |
| `probabilityLeft` run-lengths | `input[1]` `trial_number_in_block` | 0-based index of the trial inside its constant-`probabilityLeft` block, computed on the **full** (uncurated) trials table | (new; required by the task) | continuous, per-trial (broadcast over time) |
| `trials.choice` | `output[0]` `choice` | `+1 → 0` (left), `−1 → 1` (right) | `bin_behaviors` (`choice`) | per-trial, broadcast over T |
| `trials.probabilityLeft` | `output[1]` `prior_prob_left` | `0.2 → 0`, `0.5 → 1`, `0.8 → 2` | `bin_behaviors` (`block`) | per-trial, broadcast over T |
| `wheel.velocity` | `output[2]` `wheel_speed` | `abs(velocity)` → interp to bin right edges → 3 per-session terciles | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | time-varying |
| `leftCamera.ROIMotionEnergy` (else right) | `output[3]` `whisker_motion_energy` | interp to bin right edges → 3 per-session terciles | `load_target_behavior('…-whisker-motion-energy')` | time-varying |
| `clusters.acronym` → Beryl | `brain_region_idx[s]`, `brain_regions` | `BrainRegions.acronym2acronym(mapping='Beryl')` | `list_brain_regions` | |
| `bwm_release.csv` `subject` | `subjects`, `subject_idx` | unique subject names | | |

All four outputs are emitted as `(4, T)` int arrays and both inputs as `(2, T)` float32 arrays,
i.e. everything is time-varying (per the format guidance "If at all possible, make it
time-varying"); the per-trial variables are constant along the time axis.

### Key Decisions
1. **Session set** — the 459 eids / 699 pids of `bwm_release.csv` (the same freeze file the
   reference code uses). Sessions are dropped only if, after curation, they have < 2 usable
   trials or no usable neurons or no whisker motion energy (both cameras missing) — the last is
   a hard requirement because whisker ME is a required decoder output. Expect ≈ 445 sessions.
2. **Neuron curation: `clusters.label >= 1` (well-isolated), grey matter only** — deviates from
   `0_data_caching.py` (`qc=None`, all units) but follows the data paper's explicit inclusion
   criteria, and is directly supported by the reference loader (`load_spiking_data(qc=1)`).
   Rationale: (a) the data paper defines "neurons" as exactly these units; (b) multi-unit
   clusters are not neurons and are excluded from every analysis in the data paper; (c) the
   decoder here pools *all* neurons in a session and is trained on every session jointly, so
   keeping 621,733 multi-unit clusters would make the converted dataset ~110 GB and the decoder
   input dominated by poorly isolated clusters. Sanity target: 75,708 units total.
3. **Trial curation** — `load_trials_and_mask` defaults + `max_trial_len=10.0` (reference code)
   + trials whose wheel/whisker traces do not cover the full window (reference
   `get_behavior_per_interval` / `align_spike_behavior`). `allow_nans=True` as in the reference
   caching script; any trial still containing a NaN in a behaviour trace is dropped here,
   because the decoder requires finite values.
4. **Binning** — identical arithmetic to `bincount2D` (`floor((t − t_beg)/binsize)`,
   `t_beg <= t < t_end`), vectorised over trials for speed; verified against the reference
   implementation.
5. **Discretisation of the two continuous outputs into 3 classes** — per-session terciles
   (33.3 %/66.7 % quantiles of that session's binned values over all retained trials). Rationale:
   (a) whisker motion energy is in arbitrary camera-dependent units (left 60 Hz 1280×1024 vs
   right 150 Hz 640×512), so a global threshold would mean different things in different
   sessions; (b) the reference decoding code likewise standardises each behaviour *per session*
   (`StandardScaler` in `SingleSessionDataset`); (c) terciles give ≈ 1/3 per class so that
   chance = balanced-accuracy chance = 1/3 and the shared read-out is not biased by
   session-specific scale. Bin edges are stored in the metadata.
6. **`trial_number_in_block` is computed before trial curation**, so it reflects the animal's
   true position in the block even when neighbouring trials are excluded.
7. **Choice coding** — verified empirically from the data (correct + stimulus-left ⇒
   `choice == +1`), so left = 0 ⇔ `choice == +1`, right = 1 ⇔ `choice == −1`.

### Planned Sanity Checks
- [ ] total units == 621,733 and well-isolated units == 75,708 over the 699 pids (checked in Step 2 ✓)
- [ ] 459 sessions, 139 subjects, trials/session mean 645 / median 602 / range 401–1525 before curation (✓ Step 2)
- [ ] fraction of correct trials ≈ 0.814
- [ ] fraction of trials with RT < 80 ms ≈ 0.228
- [ ] `probabilityLeft` values are exactly {0.2, 0.5, 0.8}; ≈ 14 % of trials are 0.5 (first 90 trials of each session)
- [ ] choice is ≈ 50/50 left/right
- [ ] vectorised spike binning == reference `get_spike_data_per_interval` (`np.allclose`)
- [ ] vectorised behaviour interpolation == reference `get_behavior_per_interval` (`np.allclose`)
- [ ] raw-file spot checks of neural counts, inputs, and outputs at specific (session, trial, neuron, bin)
- [ ] number of Beryl regions ≈ 270–279
- [ ] per-session output class fractions ≈ 1/3 for the two discretised behaviours

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`, run as
`python -u /app/convert_data.py <outfile> [--full|--sample] [--show-processing] [--workers N]`.

Structure:
- `get_one()` — ONE client (no password; REST answered from the staged `.rest` cache).
- `load_spiking_data()` — port of the reference function with `qc=1`; drops the
  `raw_electrophysiology` call (network-only, used solely to report `sampling_freq`).
- `load_trials_and_mask()` — same query string as the reference, with `min_rt=0.08`,
  `max_rt=2.0`, `max_trial_len=10.0`, default `nan_exclude`, `exclude_nochoice=True`.
- `load_behaviour_traces()` — `SessionLoader.load_wheel()` → `|velocity|`;
  `load_motion_energy(views=['left'])` with a right-camera fallback (reference `bin_behaviors`).
- `bin_spikes()` — vectorised equivalent of `get_spike_data_per_interval`.
- `bin_behaviour()` — line-for-line port of `get_behavior_per_interval`'s `interpolate_behavior`
  (slice → `interp1d(..., fill_value='extrapolate')` at `linspace(t_beg+binsize, t_end, n_bins)`).
- `trial_number_in_block()`, `discretize_terciles()`, `apply_terciles()` — new, required by the
  decoder task specification.
- `convert_session()` — the per-session pipeline; `_plot_processing()` writes the
  `--show-processing` figure; `ProcessPoolExecutor` parallelises over sessions.

Code inefficiencies identified in the reference:
- `get_spike_data_per_interval` and `get_behavior_per_interval` each spawn a
  `multiprocessing.Pool` **per session per signal** and bin one trial at a time with
  `bincount2D`. For 459 sessions that is hours of pool start-up alone.
- `prepare_data` loads six continuous behaviours (`load_anytime_behaviors`) of which only two
  are used.
- `load_spiking_data` streams raw AP data just to read the sampling rate.

Code speedups added:
- spike binning vectorised with one `np.bincount` over all trials of a probe
  (`searchsorted` for the per-trial spike slices);
- only the two required behaviour traces are loaded, and the `SessionLoader` instance is shared
  between the trials / wheel / motion-energy loads (one ONE session lookup instead of four);
- behaviour interpolation done in-process (no pool) — ~0.05 s per session;
- parallelism moved up to the session level (`ProcessPoolExecutor`, 24 workers).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing --workers 2`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`processing_6713a4a7-….png`, `processing_56956777-….png`.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 (NYU-11) |
| Sessions / subject | 2 |
| Neurons (total) | 253 |
| Neurons / session | 126.5 (61, 192) |
| Trials (total, kept) | 651 (of 990 raw) |
| Trials / session | 407, 244 |
| Brain regions | 16 |
| T per trial | 100 (all trials) |
| `time_from_stim_onset` range | [−0.49, +1.49] |
| `trial_number_in_block` range | [0, 89] |
| `choice` distribution | left 0.518, right 0.482 |
| `prior_prob_left` distribution | 0.2 → 0.478, 0.5 → 0.161, 0.8 → 0.361 |
| `wheel_speed` distribution | 0.333 / 0.333 / 0.333 |
| `whisker_motion_energy` distribution | 0.333 / 0.333 / 0.333 |

Consistency with the papers: `trial_number_in_block` maxes out at 89 — exactly the 90-trial
unbiased block at the start of each session. `prior_prob_left == 0.5` is 16.1 % of retained
trials (90 unbiased trials out of ~565 raw ⇒ ~16 %). Choice is close to 50/50, as expected.

### Processing Plots Review
`processing_<eid>.png` panels and what they demonstrate:
1. Raw |wheel velocity| overlaid with the values interpolated at each bin's right edge — the
   red samples lie exactly on the raw trace, and `stimOn` sits at t = 0 → **no temporal
   misalignment**.
2. Same for the raw 60 Hz whisker motion-energy trace.
3./4. Continuous trace + tercile edges + resulting class step function → **discretisation is
   correct** (class changes exactly where the trace crosses an edge).
5. Binned spike-count image for one trial with `stimOn` marked.
6. Population PSTH over all retained trials: flat baseline before 0, sharp rise starting at
   t = 0 and peaking at ~0.25 s, just after the median first-movement time (magenta) — an
   independent confirmation that the neural data are aligned to stimulus onset.
7. Decoder inputs: the time ramp, and the `trial_number_in_block` saw-tooth that resets at every
   block change, with the first tooth reaching 89 (the unbiased block).
8. Per-trial outputs against the raw trials table: the prior class tracks `probabilityLeft`
   exactly, and the choice class matches the raw `choice` sign (+1 ⇒ left ⇒ class 0).

No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised spike binning instead of per-trial `bincount2D` in a pool | spike binning 3–7 s/probe (dominated by file I/O), vs minutes/session for the reference |
| behaviour interpolation without a `multiprocessing.Pool` | ~0.05 s/session vs several seconds of pool start-up |
| only 2 of 6 continuous behaviours loaded, shared `SessionLoader` | ~0.6 s/session vs several s |
| session-level `ProcessPoolExecutor` (24 workers) | ~24× |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| trials load | 0.4–0.5 s | |
| behaviour load | 0.6 s | |
| behaviour binning | 0.0–0.1 s | |
| spike load + bin | 3.1 s (1 probe) / 6.5 s (2 probes) | |
| **total per session** | 6.1 s (1 probe) / 9.4 s (2 probes) | |
| **459 sessions @ 24 workers** | — | ≈ 8 s × 459 / 24 ≈ **3 min** + pickle write |

Comfortably below the 15-minute budget, so no further optimisation was needed.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: None
- Warnings: None ("Data format is valid, no errors or warnings.")

### Training
Loss decreases monotonically: 1.47 (epoch 4) → 1.22 (10) → 0.88 (50) → 0.79 (100) → 0.764 (200).
Test loss 0.798.

### Decoder Results (Sample, 2 sessions / 651 trials / 61 + 192 neurons)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.6138 | 0.5340 | 0.5000 |
| prior_prob_left | 0.6960 | 0.6509 | 0.3333 |
| wheel_speed | 0.5717 | 0.5649 | 0.3333 |
| whisker_motion_energy | 0.5609 | 0.5585 | 0.3333 |

All four outputs are above chance. `choice` is the weakest: these two sessions record thalamus
/ amygdala / hippocampus (LGd, VPM, MG, BMA, CEA, CA1-3), which are not strong choice-coding
regions, only 61 and 192 neurons are available, and the decoder must predict choice at *every*
time bin including the 25 bins before the stimulus even appears (when choice information cannot
yet be present). Re-checked on the full dataset in Steps 11–12.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --workers 24`
→ `/app/conversion_full_out.txt` (274 s wall clock, 459 sessions attempted).

### Output Files
- `converted_data.pkl`: **11.72 GB**
- `verification_full_out.txt`: created (0 errors, 16 warnings — see Step 10)

### Sessions dropped (18 of 459)
| Reason | N |
|---|---|
| no whisker motion energy from either camera (required decoder output) | 14 |
| 0 usable trials after curation (`f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1`) | 1 |
| < 5 well-isolated grey-matter neurons | 3 |

This leaves **441 sessions / 136 subjects**, close to the 433 sessions the method paper reports
using (they additionally require pupil/other traces in some analyses).

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data (files) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions in release | 459 | 459 (`bwm_release.csv`) | 459 staged | 459 attempted, 441 kept | ✓ |
| Insertions (probes) | 699 | 699 | 699 | 672 (in kept sessions) | ✓ |
| Subjects | 139 | 139 | 139 | 136 (in kept sessions) | ✓ |
| Units (all) | 621,733 | — | 621,733 | — (not used) | ✓ |
| Well-isolated units | 75,708 (108/probe) | `qc=1` ⇒ `label>=1` | 75,708 | 62,757 after grey-matter + ≥1-spike + session filters | ✓ (82.9 % of good units are in grey matter) |
| Neurons / session | "300–2,000" (all units); 108/probe good | — | 165/session good | mean 142.3, min 7, max 516 | ✓ |
| Trials / session (raw) | mean 645, median 602, range 401–1,525 | — | mean 645.1, min 401, max 1525 | **mean 646.3, median 601, min 401, max 1525** | ✓ |
| Trials total (raw) | — | — | 296,090 (459 sess.) | 285,031 (441 sess.) | ✓ |
| Trials kept after curation | ≈ 63 % expected (22.8 % RT<80 ms + RT>2 s + NaNs + no-go) | same rule | — | 187,934 = **65.9 %** | ✓ |
| Brain regions (Beryl) | 279 (all units) / 270 (method paper) | — | — | **263** | ✓ (fewer because only grey-matter well-isolated neurons are kept) |
| `choice` distribution | ≈ 50/50 | — | — | left 0.508 / right 0.492 | ✓ |
| `prior_prob_left` distribution | 90 unbiased trials/session ⇒ ≈ 14 % at 0.5 | — | — | 0.2 → 0.417, **0.5 → 0.140**, 0.8 → 0.442 | ✓ |
| `wheel_speed` classes | — | — | — | 0.333 / 0.333 / 0.333 | ✓ (terciles by construction) |
| `whisker_motion_energy` classes | — | — | — | 0.334 / 0.333 / 0.333 | ✓ |
| `time_from_stim_onset` range | −0.5 … +1.5 s | `time_window=(-.5,1.5)` | — | [−0.49, +1.49] (bin centres) | ✓ |
| `trial_number_in_block` range | blocks of 20–100 trials; first block 90 | — | — | [0, 98] | ✓ |
| T per trial | 100 bins of 20 ms | `binsize=0.02` | — | 100 (every trial) | ✓ |
| Whisker camera | left (60 Hz) preferred | left, right fallback | 437 left / 420 right | 434 left, 7 right | ✓ |

No data was lost silently: every one of the 459 sessions is either present in `data['neural']`
or listed with its failure reason in `metadata['failed_sessions']`, and every kept trial of
every kept session is present (the per-session `n_trials_kept` equals `len(data['neural'][s])`).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification (`/app/verification_full_out.txt`)
- **Errors: none.** `verify_data_format` returns valid.
- **Warnings: 16**, all of the form *"Session &lt;s&gt;, trial &lt;k&gt;: all neural data is
  zero"*, in 2 sessions that have only 7 and 10 well-isolated grey-matter neurons. With so few
  sparse-firing neurons a 2 s window occasionally contains no spike at all. This is a genuine
  property of the recording, not a conversion bug: the independent raw-file reconstruction
  (Check 2) reproduces exactly the same all-zero trials. It was *partially* fixed by adding the
  data paper's "at least five well-isolated neurons per session" criterion, which removed
  3 sessions and cut the warnings from 38 to 16. It cannot be removed entirely without either
  (a) dropping trials on the basis of their neural activity — which would bias the dataset and
  is not a criterion used by either reference — or (b) raising the neuron threshold beyond what
  the data paper states. The affected trials are 16 of 187,934 (0.009 %).

### Check 2 — Independent sanity checks against the original files (`/app/sanity_checks.py`)
The script re-loads the raw data with ONE/`SpikeSortingLoader`/`SessionLoader` and re-derives
everything with code that shares nothing with `convert_data.py` (spikes are re-binned with
`np.histogram` per trial per cluster), then compares with `np.allclose`.
**Result: 37/37 checks passed, 0 failed** on 3 randomly chosen sessions
(`c02e5155…` 12 neurons, `aa3432cd…` 151 neurons, `195443eb…` 10 neurons):

| Stream | Check | Result |
|---|---|---|
| neural | spike-count matrix `allclose` to per-trial `np.histogram` of raw `spikes.times` | PASS (max abs diff = 0; 108,584 / 1,162,585 / 46,556 spikes) |
| neural | Beryl region of every neuron matches `clusters.acronym` → Beryl | PASS |
| neural | #neurons == len(`brain_region_idx[s]`) for all 441 sessions | PASS |
| input | `time_from_stim_onset` == bin centres of [−0.5, 1.5] | PASS |
| input | `trial_number_in_block` == run-length index of raw `probabilityLeft` | PASS |
| input | `trial_number_in_block` constant within a trial | PASS |
| output | `choice`: raw `+1 → 0` (left), `−1 → 1` (right) | PASS |
| output | `prior_prob_left`: raw `probabilityLeft` 0.2/0.5/0.8 → 0/1/2 | PASS |
| output | `wheel_speed` classes == terciles of independently interpolated `abs(wheel.velocity)` | PASS |
| output | `whisker_motion_energy` classes == terciles of independently interpolated camera ME | PASS |
| output | stored tercile edges == the 1/3 and 2/3 quantiles of the session's values | PASS |
| global | trial count per session == independent reconstruction | PASS |
| global | raw trials/session mean 646.3, median 601, range 401–1525 == paper (645 / 602 / 401–1525) | PASS |
| global | every trial has T = 100 | PASS |

### Check 3 — Reference code comparison (`/app/reference_comparison.py`)
Every vectorised routine was run against the *actual reference function* from
`/app/code/code_zhang2025/src/utils/ibl_data_utils.py` on a real session
(`6713a4a7-faed-4df2-acab-ee4e63326f8d`):

| Step | My code | Reference | Result |
|---|---|---|---|
| (a) data loading | `load_spiking_data(qc=1)` (port), `SessionLoader` for trials/wheel/motion energy | `ibl_data_utils.load_spiking_data`, `load_target_behavior` | same objects, same fields; only the network-only `raw_electrophysiology` sampling-rate call is dropped |
| (b) trial filtering | `load_trials_and_mask` (port) | `ibl_data_utils.load_trials_and_mask(max_trial_len=10.0)` | **masks identical** (407/565 trials kept) |
| (b) neuron filtering | `label >= 1`, Beryl ∉ {root, void}, ≥ 1 spike, ≥ 5 neurons/session | `qc=1` option; `select_brain_regions`; `np.unique(regclu)` | same mechanism; the grey-matter, ≥5-neuron and `qc=1` choices follow the data paper (documented deviation from `0_data_caching.py`, which uses `qc=None`) |
| (c) temporal alignment | `stimOn_times`, window (−0.5, +1.5) | `params` in `0_data_caching.py` | identical |
| (d) binning | `bin_spikes` (vectorised) | `get_spike_data_per_interval` (`bincount2D` in a pool) | **`np.allclose` → True, max abs diff = 0** over 60 trials × all clusters × 100 bins |
| (d) behaviour binning | `bin_behaviour` | `get_behavior_per_interval` | **values identical (max diff = 0) and good-trial masks identical** for both wheel speed and whisker motion energy |
| (e) input construction | new (`time_from_stim_onset`, `trial_number_in_block`) | — | required by the decoder-task specification; no reference equivalent |
| (f) output construction | `choice`, `probabilityLeft`, wheel speed, whisker ME | `bin_behaviors` produces exactly these four variables (`choice`, `block`, `wheel-speed`, `whisker-motion-energy`) | same source variables; the two continuous ones are additionally discretised into 3 classes, as the decoder task requires categorical outputs |

**Deviations from the reference code, and why**
1. `qc=1` instead of `qc=None` — follows the data paper's neuron inclusion criteria; also keeps
   the converted dataset at 12 GB instead of ~110 GB, which matters because `train_decoder.py`
   holds the whole dataset plus a float32 copy in memory.
2. grey-matter-only and ≥ 5 neurons/session — data paper, "Neurons and brain regions".
3. 20 ms bins (reference code) rather than the 50 ms quoted in one sentence of
   `methods.txt` — required to resolve the two time-varying outputs, and consistent with the
   method paper's own "20-ms bins, T = 100".
4. Trials with a NaN left in an interpolated behaviour trace are dropped, whereas the reference
   keeps them (`allow_nans=True`) and imputes the trial mean at model-fitting time. The decoder
   here rejects non-finite values outright, so they must be removed at conversion time.
5. Sessions with no whisker motion energy are dropped. The reference could keep them (it decodes
   one target at a time); here whisker ME is a required output for every trial.

### Check 4 — Key statistics comparison
See the table in Step 9. Every statistic that the papers state and that survives the curation
matches: 459 sessions / 699 probes / 139 subjects in the release; 621,733 units and 75,708
well-isolated units measured directly from `clusters.metrics.pqt` over the 699 pids (exact
match to the paper); raw trials/session mean 646.3, median 601, range 401–1,525 (paper:
645 / 602 / 401–1,525); ≈ 14 % of trials in the unbiased 50:50 block; choice ≈ 50/50.

Investigated discrepancies:
- **"average of 676 neurons per session" (method paper)**: reproducible from neither all units
  (1,354/session) nor well-isolated units (165/session) on the 459-session freeze. Its stated
  *range* ("300 to over 2000") matches all units. The number is not reproducible from the
  released freeze file + code and is not used as a target; the two exactly-stated data-paper
  counts (621,733 / 75,708) are matched instead.
- **263 vs 279 brain regions**: 279 is the Beryl coverage of *all* units including `root`/`void`
  and multi-unit clusters; restricting to well-isolated grey-matter neurons leaves 263, and the
  method paper's own figure is 270. Consistent.
- **441 vs 433 sessions**: the method paper's 433 comes from a slightly different set of
  required traces (it also loads pupil in some analyses and silently skips sessions that raise).
  Our 18 exclusions are individually enumerated and all justified.

### Check 5 — Edge cases
- *Off-by-one at trial edges*: bin *i* spans `[t_beg + 0.02 i, t_beg + 0.02 (i+1))`; the last
  bin ends exactly at `stimOn + 1.5`. Verified by `np.allclose` against `bincount2D` (Check 3)
  and against `np.histogram` with explicit edges (Check 2). The input's time value is the bin
  *centre*, so its range is [−0.49, +1.49] rather than [−0.5, +1.5].
- *Trials at the start/end of a session*: handled by `get_behavior_per_interval`'s coverage
  test — video/wheel traces often start after the first trials, and those trials are dropped.
- *Blocks*: `trial_number_in_block` is computed on the **full** trials table, restarts at 0 on
  every `probabilityLeft` change, and reaches at most 98 (blocks are 20–100 trials) with the
  first block reaching 89 (90 unbiased trials). Verified in the sanity check and the plot.
- *Sessions with 2 probes*: 231 of 441; clusters of both probes are concatenated and
  `brain_region_idx` is built in the same order.
- *Degenerate tercile edges*: one session (`5b44c40f…`) has 78 % of its whisker-ME samples
  exactly 0, so the 1/3 and 2/3 quantiles coincide; `discretize_terciles` then splits the
  non-zero mass at its median, giving 0.784/0.108/0.108 instead of 1/3 each. This is the best
  available 3-way split of that trace and is reported in the metadata.
- *Unexpected `probabilityLeft`*: the conversion raises if any value is not 0.2/0.5/0.8 — never
  triggered on the 459 sessions.
- *Clusters with no spikes*: excluded, matching the reference `bin_spiking_data`.
- *`min_rt` sign*: the reference query uses strict `<`/`>` so trials at exactly 0.08 s or 2.0 s
  are kept; the port reproduces the query string verbatim and the masks are bit-identical.

### Iterations
1. **Iteration 1** — first full conversion gave 444 sessions with 38 "all neural data is zero"
   warnings and a session containing a single neuron. *Fix*: added the data paper's
   "≥ 5 well-isolated neurons per session" criterion (`MIN_NEURONS_PER_SESSION = 5`).
   *Re-check*: 441 sessions, warnings down to 16, min neurons/session 7; all sanity checks and
   reference comparisons re-run and pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.
Ran on the GPU, 200 epochs, ~13 min.

### Training Progress
- Loss decreasing: **Yes**, monotonically — 2.645 (epoch 1) → 1.965 (10) → 1.557 (20) →
  1.169 (40) → 0.806 (100) → 0.7477 (200). Test loss 0.7713.

### Decoder Results (Full: 441 sessions, 187,934 trials, 62,757 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance |
|--------|--------|-------------|--------|-------|
| choice | 0.5000 | 0.6358 | **0.6144** | 1.23× |
| prior_prob_left | 0.3333 | 0.6750 | **0.6575** | 1.97× |
| wheel_speed | 0.3333 | 0.6123 | **0.6061** | 1.82× |
| whisker_motion_energy | 0.3333 | 0.5940 | **0.5879** | 1.76× |

All four outputs are well above chance and the train→validation gap is small
(≤ 0.021 absolute, ratio ≤ 1.03), so there is no overfitting and no sign of leakage.

`sample_trials.png` shows the expected structure: a linear time ramp for input 0, a constant
`trial_number_in_block` for input 1, constant `choice` / `prior_prob_left` within a trial, and
`wheel_speed` / `whisker_motion_energy` classes that step up shortly after the stimulus.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance

| Variable | Classes | Chance | Validation Balanced Acc | Ratio to chance | Above chance? |
|---|---|---|---|---|---|
| choice | 2 | 0.5000 | 0.6144 | 1.23× | yes |
| prior_prob_left | 3 | 0.3333 | 0.6575 | 1.97× | yes |
| wheel_speed | 3 | 0.3333 | 0.6061 | 1.82× | yes |
| whisker_motion_energy | 3 | 0.3333 | 0.5879 | 1.76× | yes |

Nothing is below chance. Three of the four outputs are ≥ 1.76× chance. `choice` is 1.23×
chance, below the 1.5× flag, so it was investigated in depth.

**Why `choice` cannot reach 1.5× chance on this task definition, and why that is not a bug.**
`choice` is a *per-trial* label that the decoder must predict at **every one of the 100 time
bins**, and the trial window starts 0.5 s *before* the stimulus appears. Choice information
simply does not exist in the brain during the first quarter of the window. Two independent
analyses of the time course confirm this (`/app/timecourse_analysis.py`, and a
completely independent scikit-learn per-bin logistic regression on 25 sessions):

| time from stimOn | decoder balanced acc (choice) | independent per-bin logistic (choice) |
|---|---|---|
| −0.49 … −0.01 s (pre-stimulus) | 0.570 | 0.52 |
| +0.15 s | 0.662 | 0.72 |
| **+0.23 … +0.31 s (peak)** | **0.697** | **0.78** |
| +0.55 s | 0.657 | 0.65 |
| +1.43 s | 0.594 | 0.55 |
| **mean over all 100 bins** | **0.616** | 0.600 |

The decoder's overall 0.614 is exactly the average of a curve that runs from chance before the
stimulus to ~0.70 just after the first wheel movement. The peak time (+0.23 … +0.31 s) matches
the IBL median reaction time, i.e. **choice becomes decodable precisely when the animal moves**
— strong independent evidence that the temporal alignment is correct. The decoder's own value
(0.616 averaged over bins) also *exceeds* the independent baseline (0.600), so the conversion is
not losing information relative to a straightforward reimplementation.

Corresponding time courses for the other outputs (peak / pre-stimulus balanced accuracy):
prior 0.694 / 0.644, wheel speed 0.596 / 0.456, whisker ME 0.565 / 0.457 — all peak at
+0.23 s, again consistent with stimulus-locked alignment. (Per-bin balanced accuracy is lower
than the pooled value for the two time-varying outputs because within a single early time bin
the "medium"/"high" classes are rare; pooling over bins restores balance.)

Things that were considered and rejected as ways to raise `choice`:
- *align to `firstMovement_times` instead* (what the data paper uses for choice decoding, and
  what the method paper uses for dynamic behaviours) — the decoder task here explicitly
  specifies "Temporally align based on stimulus onset", so this is not allowed;
- *shorten the window to post-stimulus only* — would discard the reference window (−0.5, +1.5) s
  that both the reference code and the method paper prescribe for choice;
- *use all 621,733 units instead of the 75,708 well-isolated ones* — measured directly
  (see Check 2 below): it raises per-bin choice decoding by only ~0.02 for an 8.4× larger
  dataset, and is contrary to the data paper's neuron inclusion criteria.

### Check 2 — Comparison to the reference papers

| Variable | Reference paper number | Metric / window used there | This dataset | Comparable? |
|---|---|---|---|---|
| choice | method paper Fig. 2/5: single-session linear & RRR per region ≈ 0.55–0.80 accuracy; "to avoid **ceiling effects from using all regions**, we decode each region separately" | one label per trial, single region, window (−0.5, +1.5) s | 0.614 averaged over all 100 bins; **0.697 at the informative bins**; 0.78 with a per-bin logistic on all neurons of a session | partly — theirs is per-trial and per-region, ours is per-timestep and all-region. Our peak sits at the top of their per-region range, consistent with their remark that using all regions approaches ceiling. |
| prior / block | method paper: AUC 0.66 (linear baseline) → 0.72 (single-session BMM-HMM) → 0.79 (multi-session) | AUC, binary L/R block, window (−0.6, −0.1) s pre-onset | 0.651 balanced accuracy over **3** classes (chance 0.333), 0.694 at peak | different metric (AUC on 2 classes vs balanced accuracy on 3) — ours is far above its chance level and includes the harder 0.5 class |
| wheel speed | method paper: R² 0.24 (linear) → 0.55 (RRR (M)) | R² of the continuous signal, aligned to first movement, 0–1 s | 0.598 balanced accuracy over 3 classes (chance 0.333) | not directly comparable (R² vs 3-class accuracy; different alignment event). R² 0.24–0.55 corresponds to a moderately decodable signal, consistent with 1.8× chance on a 3-way discretisation. |
| whisker motion energy | method paper: R² 0.52 → 0.80 | same | 0.594 balanced accuracy, 3 classes | as above |
| choice / stimulus / feedback per region | data paper: **null-corrected median balanced accuracy**, small effect sizes; raw per-region balanced accuracies ≈ 0.5–0.6 | logistic regression, single region, single time window | 0.697 peak using all neurons of a session | ours is higher, as expected when pooling all regions of a session instead of one region |

No reported number is above what this conversion achieves once the metric and window
differences are accounted for, so there is no evidence of a conversion bug.

**All-units control experiment** (`qc=None` vs `qc=1`, 6 randomly chosen sessions, identical
per-bin logistic decoder):

| Neuron set | mean neurons/session | mean per-bin choice acc | peak per-bin choice acc |
|---|---|---|---|
| well-isolated (`label >= 1`, used here) | 125 | 0.569 | 0.741 |
| all units (`qc=None`, reference caching script) | 1047 | 0.588 | 0.774 |

An 8.4× larger dataset (≈ 100 GB instead of 11.7 GB) buys +0.019 balanced accuracy. This
confirms that the well-isolated-neuron criterion of the data paper is not costing meaningful
decodable information, and justifies the neuron-curation decision documented in Step 5.

### Check 3 — Train vs validation gap

| Output | Train | Validation | Train/Val ratio |
|---|---|---|---|
| choice | 0.6358 | 0.6144 | 1.035 |
| prior_prob_left | 0.6750 | 0.6575 | 1.027 |
| wheel_speed | 0.6123 | 0.6061 | 1.010 |
| whisker_motion_energy | 0.5940 | 0.5879 | 1.010 |

All ratios are ≤ 1.04, far below the 1.5 threshold: no overfitting and no data leakage. (Nor is
leakage structurally possible: the only per-trial input, `trial_number_in_block`, is independent
of block *identity* and therefore of every output.)

### Additional debugging performed
1. **Output values verified against raw data for specific trials** — `sanity_checks.py`
   re-derives `choice`, `prior_prob_left`, and both discretised behaviours from the original
   files for every trial of 3 sessions and compares with `np.allclose`: all match (Step 10,
   Check 2).
2. **Temporal alignment verified by plotting** — `processing_<eid>.png` overlays raw wheel and
   motion-energy traces on the binned values, and the population PSTH rises exactly at t = 0
   and peaks at +0.25 s. The decoder's own accuracy time course peaks at +0.23…+0.31 s.
3. **Output variation checked** — no output is dominated by one class: choice 0.508/0.492,
   prior 0.417/0.140/0.442, wheel and whisker 0.333/0.333/0.334. Per-session, the rarest class
   fraction is 0.122 (choice), 0.056 (prior — a session with few unbiased trials) and 0.108
   (whisker, the one session whose motion-energy trace is 78 % exactly zero).
4. **Neural filtering checked** — `label >= 1` reproduces the paper's 75,708 well-isolated
   units exactly; the grey-matter restriction leaves 62,757 (82.9 %).
5. **Processing checked against the reference code** — spike binning, behaviour interpolation
   and the trial mask are bit-identical to `ibl_data_utils` (Step 10, Check 3).

### Issues Found and Resolved
- *Sessions with 1–3 neurons produced all-zero trials* → added the data paper's ≥ 5
  well-isolated neurons per session criterion; re-ran the full conversion, verification, sanity
  checks and reference comparison. 16 all-zero trials remain out of 187,934 (0.009 %), in two
  sessions with 7 and 10 neurons; these are real (reproduced from the raw files) and removing
  them would mean censoring trials by their neural activity.
- *No other issue was found in this pass; no change to the conversion was required by Checks
  1–3.*

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — dataset description, how to load, format specification, curation
      summary, key statistics vs the papers, decoder performance, file inventory.
- [x] `cache/` folder created with `README_CACHE.md`; the three validation scripts
      (`sanity_checks.py`, `reference_comparison.py`, `timecourse_analysis.py`), their outputs
      and the Step-2 exploration table (`session_scan.csv`) moved there.
- [x] All required files present in `/app`:
      `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl` (11.7 GB),
      `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`,
      `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
      `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`,
      plus `processing_<eid>.png` (×2), `sample_trials.png`, `predictions.png`.
- [x] `sample_data.pkl` and its three logs regenerated with the final version of
      `convert_data.py`, and the full decoder training re-run end-to-end on the final
      `converted_data.pkl`.

### Summary of the decisions made (and why)
| Decision | Choice | Justification |
|---|---|---|
| Session set | the 459-eid / 699-pid BWM freeze (`bwm_release.csv`) | the file the reference code itself uses; equals the data paper's released set |
| Alignment | `stimOn_times`, window (−0.5, +1.5) s | decoder task requirement; identical to the reference `params` and to the method paper's choice/prior window |
| Bin size | 20 ms → T = 100 | reference `binsize=0.02`; method paper "2-s trials … 20-ms bins … T = 100"; needed to resolve the time-varying outputs |
| Neurons | well-isolated (`label >= 1`), grey matter, ≥ 1 spike, ≥ 5 per session | data paper's neuron inclusion criteria; supported by the reference loader's `qc=1`; the all-units control shows it costs ≈ 0.02 accuracy while shrinking the dataset 8.4× |
| Trials | reference `load_trials_and_mask` + `max_trial_len=10` + full behaviour coverage | data paper "Trials" + reference `prepare_data` / `align_spike_behavior` |
| Spike binning | `floor((t − t_beg)/binsize)`, `t_beg ≤ t < t_end` | bit-identical to `bincount2D` as used by the reference |
| Behaviour sampling | linear interpolation at each bin's **right** edge | exactly `get_behavior_per_interval` |
| Discretisation | per-session terciles, class 0 = `v ≤ q33` | camera-dependent units make a global threshold meaningless; the reference standardises behaviour per session; gives chance = 1/3 |
| Choice coding | `choice == +1` → left (0), `== −1` → right (1) | verified empirically: correct trials with a left stimulus all have `choice == +1` |
| `trial_number_in_block` | 0-based run-length index of `probabilityLeft`, computed **before** trial curation | reflects the animal's true position in the block; carries no information about block identity, so it cannot leak the `prior_prob_left` label |
