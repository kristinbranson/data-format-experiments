# Dataset Conversion Notes

## Overview
- **Dataset**: IBL brain-wide map (ONE cache in /app/data); methods paper decoder code in /app/code
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest (list of staged ONE cache files)
- Dockerfile, docker-compose.yaml, stage_cache.sh (environment setup)
- code/ (reference code from methods paper)
- data/ (IBL ONE cache)
- dataarchitecture.pdf, datapaper.pdf, methodpaper.pdf, methods.txt (references)
- decoder.py, train_decoder.py (provided decoder validation code)
- CONVERSION_NOTES.md (this file)

Environment verified: numpy 2.3.5, torch 2.6.0+cu124, ONE api 3.5.2, brainbox importable.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference code: `/app/code/code_zhang2025` (Zhang et al. 2025, "Exploiting correlations across trials
 and behavioral sessions to improve neural decoding"), plus a copy of `ibllib`/`brainbox` in `/app/code/ibllib`.

Entry point for data preparation: `src/0_data_caching.py`. It uses `one.api.ONE` +
`src/utils/ibl_data_utils.py`. Parameters used there (verbatim):

```python
params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
          'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']
```

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data(one, eid, bwm_df, params)` | src/utils/ibl_data_utils.py | LOADING | Per-eid: `one.eid2pid` -> loads each probe with `load_spiking_data`, merges probes, loads trials+mask, loads "anytime" behaviors. Returns neural_dict, behave_dict, meta_data, trials_data |
| `load_spiking_data(one, pid, qc=None)` | ibl_data_utils.py | LOADING/CURATION | `brainbox.io.one.SpikeSortingLoader.load_spike_sorting()` + `merge_clusters` -> clusters table with `label`. `qc=None` (as called in `prepare_data`) keeps ALL clusters; `qc=1` would keep only clusters with label>=1 ("good units"). Also reads sampling freq via `raw_electrophysiology` (needs remote data; will be replaced) |
| `merge_probes(spikes_list, clusters_list)` | ibl_data_utils.py | LOADING | Concatenate probes of one session, re-index clusters, sort spikes by time |
| `load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', max_trial_len=10.0, exclude_nochoice=True)` | ibl_data_utils.py | CURATION | Builds trials df via `brainbox.io.one.SessionLoader.load_trials()` and a boolean mask excluding: RT<0.08s, RT>2s, trial length (feedback-goCue)>10s, NaNs in [stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType], choice==0 |
| `list_brain_regions(neural_dict, single_region=False)` | ibl_data_utils.py | PROCESSING | `iblatlas.regions.BrainRegions().acronym2acronym(acronyms, mapping='Beryl')` -> Beryl region per cluster |
| `select_brain_regions` | ibl_data_utils.py | CURATION | cluster indices within requested region(s); with `single_region=False` all regions are used together |
| `bin_spiking_data(reg_clu_ids, neural_dict, trials_df, binsize=0.02, align_time, time_window)` | ibl_data_utils.py | PROCESSING | intervals = stimOn + (-0.5, 1.5); bins spikes with `iblutil.numerical.bincount2D` into 100 bins of 20 ms; returns array (n_trials, n_bins, n_clusters) and cluster ids used |
| `load_target_behavior(one, eid, target)` | ibl_data_utils.py | LOADING | `SessionLoader.load_wheel()` -> wheel-speed = abs(velocity); `SessionLoader.load_motion_energy(views=['left'])` -> leftCamera whiskerMotionEnergy (falls back to right camera) |
| `get_behavior_per_interval(...)` | ibl_data_utils.py | PROCESSING | For each trial interval, linearly interpolates the continuous behavior onto `x_interp = linspace(beg+binsize, end, n_bins)` (100 bins); marks interval bad if data missing / starts too late / ends too early / NaNs (when allow_nans=False) |
| `bin_behaviors(one, eid, ['wheel-speed','whisker-motion-energy'], trials_df, allow_nans=True)` | ibl_data_utils.py | PROCESSING | Per-trial scalars: choice (`trials.choice`), block (`trials.probabilityLeft`), reward (`rewardVolume>1`), contrast (signed, -(contrastLeft+contrastRight)); plus binned time-varying behaviors |
| `align_spike_behavior(binned_spikes, binned_behaviors, beh_names, trials_mask)` | ibl_data_utils.py | CURATION | Deletes trials that are masked out (bad trials) or whose behavior interval is None so that neural/behavior trial counts match |
| `standardize_spike_data` | src/utils/data_loader_utils.py | PROCESSING | z-scores spike counts per time bin across trials *inside the decoder* (not part of the cached dataset) |
| `SingleSessionDataset` | data_loader_utils.py | PROCESSING | one-hot encodes discrete targets (choice) for classification; StandardScaler for continuous/time-varying targets; NaNs in behavior replaced by mean |

### Notes
- The cached dataset stores *spike counts* per (trial, 20 ms bin, cluster); z-scoring happens in the decoder, so I will save raw spike counts in `neural`.
- `prepare_data` keeps **all** clusters and records `good_clusters = (label >= 1)` in metadata; the decoder examples use `region='all'` i.e. all clusters. The BWM data paper, however, uses a strict single-unit QC (label==1, i.e. all three metrics pass) for its analyses. See Steps 3/4 for the resolution (I will filter to good units, label>=1, and also drop `root`/`void` regions as the reference multi-region loader does).
- `align_spike_behavior` contains a Python quirk (`list and list` returns the second list) so effectively only `trials_mask` is applied; I implement the intended behaviour (drop trials failing the trials mask OR with missing behavioral data).
- `load_spiking_data` calls `spike_loader.raw_electrophysiology(band="ap", stream=True).fs` only to record the sampling frequency; this requires streaming raw data from the cloud and is not needed for binning, so it is omitted for the local read-only cache.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/one_cache` is a standard **ONE cache**: `<lab>/Subjects/<subject>/<yyyy-mm-dd>/<number>/alf/...`.
  The lab directories are symlinks into a read-only mount (567 GB of staged ALF files).
- Three release-tag table directories are copied into the cache root: `Brainwidemap`
  (480 sessions / 76,563 datasets), `2022_Q4_IBL_et_al_BWM` (354 / 38,814) and
  `2025_Q3_IBL_et_al_BWM` (459 / 5,339).
- **Important**: the staged files are the *latest revisions* (e.g.
  `alf/#2025-03-03#/_ibl_trials.table.pqt`, `alf/probe00/pykilosort/#2024-05-06#/spikes.times.npy`,
  `alf/#2025-05-29#/leftCamera.ROIMotionEnergy.npy`). Several of those revisions are **not** in any
  of the local parquet tables, so `ONE(..., mode='local')` on the cache root cannot see them.
  There are 6,483 cached Alyx REST responses in `/app/data/one_cache/.rest` whose expiry is set to
  2076, and `$HOME/.one` holds a token + cache map. Therefore
  `ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True, cache_dir='/app/data/one_cache')`
  works fully **offline** in ONE's `remote` mode (no network: port 443 is refused), and resolves
  every staged dataset including revisions. This is exactly how the reference code builds `one`
  (`src/0_data_caching.py`), so I use the same call.
- Data objects used: `trials` (`_ibl_trials.table.pqt` + extra attributes), `wheel`
  (`_ibl_wheel.position/timestamps`), `leftCamera`/`rightCamera` `ROIMotionEnergy` + camera `times`,
  spike sorting (`pykilosort`: `spikes.times/clusters`, `clusters.*`, `channels.*`).

### Dataset Size (from data files)
Surveyed all 459 BWM eids (`/tmp/survey.csv`, script `/app/cache/survey.py`):

| Statistic | Value |
|-----------|-------|
| Sessions in release (`bwm_release.csv`, = Alyx tag 2025_Q3) | 459 |
| Probe insertions | 699 |
| Subjects | 139 |
| Labs | 12 |
| Trials (total, raw) | 296,090 (mean 645.1/session, min 401, max 1525) |
| Trials passing reference trial mask | 195,781 (mean 426.5/session, median 393, min 126, max 1445) |
| Sessions with wheel data | 459 |
| Sessions with left-camera whisker ME | 437 |
| Sessions with right-camera whisker ME | 433 |
| Sessions with neither camera ME | 14 |
| Clusters / probe (9-probe sample) | 754 (paper: 889) |
| Well-isolated (`label >= 1`) clusters / probe (9-probe sample) | 84 (paper: 108) |
| Behaviour: fraction correct (masked trials) | 0.856 |
| Behaviour: fraction choice = right (choice==-1) | 0.494 |
| Fraction of masked trials in 0.5 (unbiased) block | 0.155 |
| Fraction of masked trials in p(left)=0.8 block | 0.432 |

Other verified facts about the native data:
- `trials.choice`: +1 = mouse turned wheel so that a **left** stimulus was correct (verified: all
  correct trials with `contrastLeft>0` have choice=+1), -1 = **right** choice, 0 = no-go.
- `trials.probabilityLeft` takes exactly the three values {0.2, 0.5, 0.8}; the first 90 trials of a
  session are the 0.5 unbiased block; biased blocks then alternate (example session: 12 biased
  blocks, mean length 40).
- wheel from `SessionLoader.load_wheel()` is resampled to 1000 Hz with Gaussian-smoothed velocity.
- left camera ROIMotionEnergy is sampled at 60 Hz (dt = 0.0166 s), right camera at 150 Hz.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 139 mice (94 M / 45 F) | "We trained 139 mice (94 male and 45 female)" (data paper) |
| Sessions released | 459 | "a total of 459 sessions, 699 insertions and 621,733 neurons remained" |
| Insertions | 699 | same quote |
| Units (all, pre-QC) | 621,733 total, 889/probe | "...averaging 889 per probe" |
| Well-isolated neurons | 75,708 total, 108/probe | "...identified 75,708 well-isolated neurons, averaging 108 per probe" |
| Sessions used by methods paper | 433 sessions, 270 Beryl regions | "We apply our models to 433 IBL sessions, covering 270 brain regions" |
| Min trials/session for release | >= 250 trials | "mice performed at least 250 trials" |
| Neural time bin (dynamic behaviours / this task) | 20 ms | "neural activity within each trial is binned into non-overlapping 20 ms bins"; reference code `binsize: 0.02` |
| Trial length | 2 s -> T = 100 bins | "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps" |
| Choice alignment window | stimOn, -0.5 s to +1.5 s | "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset"; code `align_time: stimOn_times, time_window: (-.5, 1.5)` |
| Behaviour sampling | wheel + whisker ME "time-varying signals sampled at 60 Hz" | methods paper |
| Block structure | first 90 trials unbiased (p=0.5), then 20:80 / 80:20 blocks, mean 51 trials | data paper |
| Contrast set | {1.0, 0.25, 0.125, 0.0625, 0} | data paper |
| Reward rate | ~85-90% on easy trials (>=90% on 100% contrast required for release) | data paper inclusion criteria; measured 0.856 overall |
| Prior values | p(left) in {0.2, 0.5, 0.8} | data paper / verified in data |

### Processing Details
- **Alignment**: trials aligned to `stimOn_times`; interval = [stimOn-0.5, stimOn+1.5]; 100 non-overlapping 20 ms bins.
- **Neural binning**: spike counts per (cluster, 20 ms bin); reference uses `bincount2D(times, clusters, xbin=0.02, xlim=[t_beg,t_end])` and truncates to the first 100 bins.
- **Behaviour binning**: continuous behaviours are linearly interpolated onto `linspace(t_beg+binsize, t_end, 100)` (i.e. the **right edge** of each 20 ms bin) by `get_behavior_per_interval`; an interval is rejected if the behavioural samples do not cover it to within one bin.
- **Wheel speed** = `abs(velocity)` of the 1 kHz-resampled, Gaussian-smoothed wheel (`SessionLoader.load_wheel`).
- **Whisker motion energy** = `leftCamera.whiskerMotionEnergy` (`SessionLoader.load_motion_energy(views=['left'])`), with fallback to the right camera, as in `bin_behaviors`.
- **Decoding in the data paper**: regularized logistic regression on binned spike counts, neurons pooled across probes within a session, evaluated with **balanced accuracy**; wheel values averaged in 20 ms bins and decoded with Lasso (R^2).

### Curation Steps

**Session curation rules** (data paper): >=250 trials, >=90% correct on 100% contrast in both block types, >=3 error trials, hardware QC passed, resolved histology alignment, no RIGOR failure -> the 459 released sessions (= `bwm_release.csv`).

**Neuron curation rules**: data paper excludes neurons failing any of three RIGOR single-unit metrics (amplitude > 50 uV, noise cut-off < 20 uV, refractory-period violation) -> "well-isolated neurons" (75,708 of 621,733). In IBL data these three metrics are combined in `clusters.label` (0, 1/3, 2/3, 1); `label == 1` means all three passed, i.e. `label >= 1` selects well-isolated units (this is exactly `load_spiking_data(..., qc=1)` in the reference code). Analyses are further restricted to grey-matter regions (Beryl; `root`/`void` excluded).

**Trial curation rules** (identical in paper and reference code `load_trials_and_mask`): exclude trials with missing `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times`; exclude trials whose first-movement time is not within 0.08-2.00 s of stimulus onset; reference code additionally drops no-go trials (`choice == 0`) and trials longer than 10 s (`max_trial_len=10.0`).

### Decoders Trained (reference accuracies to compare against)
| Decoded variable | Reported performance |
|---|---|
| Choice (single region, IBL BWM paper) | null-corrected median balanced accuracy, significant regions ~0.55-0.65 |
| Choice (methods paper, whole session) | AUC 0.66-0.79; example accuracies 0.71 (linear) -> 0.90 (BMM-HMM) |
| Prior (methods paper) | Pearson correlation 0.05 (single-session) -> 0.65 (oracle) |
| Wheel speed / whisker ME (methods paper) | continuous R^2/correlation (RRR > ridge); data paper wheel-speed decoding R^2 over null |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Unit quality control | `prepare_data` calls `load_spiking_data` with `qc=None` (keeps every cluster) but stores `good_clusters = label >= 1` in metadata; `load_spiking_data(..., qc=1)` is the provided "good units" path | `clusters.label` takes values {0, 1/3, 2/3, 1}; ~754 clusters/probe, ~84 with label==1 in a 9-probe sample | Data paper: analyses use only the 75,708 "well-isolated neurons" (of 621,733) that pass all three RIGOR single-unit metrics | Use **label >= 1** (well-isolated). This matches the data paper's definition of "neurons", is the reference code's own `qc=1` option, and keeps the converted dataset a manageable size (~13 GB vs ~90 GB). Documented as a deliberate choice. |
| Brain regions | `list_brain_regions` maps acronyms to **Beryl**; `MultiRegionDataModule.list_regions` drops `root` and `void` | Beryl acronyms include `root`/`void` for units outside the annotated grey matter | Data paper restricts analyses to grey-matter Allen CCF regions | Map to Beryl and **drop units in `root`/`void`** (non-grey-matter), as the reference multi-region loader does |
| Alignment event | caching script: `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)` | - | Methods paper: stimOn for choice/prior; firstMovement for wheel speed / whisker ME | The Decoder Task here mandates **stimulus-onset alignment**, which is exactly the reference caching configuration, so stimOn +/- (-0.5, 1.5) is used for all variables |
| Time bin | code `binsize=0.02` (20 ms), `interval_len=2` | - | Methods paper: 50 ms bins for choice/prior, 20 ms bins for the dynamic behaviours; "2-s trials ... 20-ms bins, producing T = 100" | Use **20 ms / T = 100**, i.e. the cached-dataset configuration. Required anyway because wheel speed and whisker ME must be decoded as time-varying outputs |
| Trial mask | `load_trials_and_mask(..., max_trial_len=10.0)` -> RT in [0.08, 2] s, no NaNs in 6 events, `choice != 0`, feedback-goCue <= 10 s | 426.5/645.1 trials per session pass | Data paper: same NaN + RT criteria (does not mention the 10 s / no-go rules, but no-go trials have no choice so they are excluded implicitly) | Use the reference function unchanged (`max_trial_len=10.0`) |
| Behaviour interval rejection | `get_behavior_per_interval` rejects an interval if behaviour data is absent, starts >1 bin late, or ends >1 bin early; `allow_nans=True` in the caching script | Wheel: 1 kHz continuous. Left-camera ME: 60 Hz, 437/459 sessions; 14 sessions have no camera ME at all | Methods paper analyses 433 of the 459 sessions | Apply the same rejection per trial, and **drop sessions with no whisker ME** (the whisker output would otherwise be undefined). 445 candidate sessions, close to the paper's 433 |
| `align_spike_behavior` | `target_mask = [1]*n and beh_mask` - Python's `and` returns the *second* list, so only the last behaviour's mask and then only `trials_mask` survive | - | Intent is clearly the conjunction | Implement the **intended conjunction**: a trial is kept only if it passes the trials mask *and* both behaviour streams cover it |
| `SessionLoader(one, eid)` | reference calls it positionally | installed `brainbox` version is a dataclass -> `TypeError` | - | Call `SessionLoader(one=one, eid=eid)` and pass it into `load_trials_and_mask(sess_loader=...)`; logic unchanged |
| Sampling frequency read | `load_spiking_data` calls `spike_loader.raw_electrophysiology(band='ap', stream=True).fs` | raw `.cbin` files are not staged; this needs the network | Not used in any processing | Omit; only metadata |
| Number of sessions | caching script runs on a user-selected subset (`--n_sessions`) | 459 eids available | Methods paper: 433 sessions | Convert **all** sessions that have the required data streams |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial geometry (identical for every trial and session)
`t0 = stimOn_times - 0.5`, `t1 = stimOn_times + 1.5`, `binsize = 0.02 s`, `T = 100` bins.
Bin *k* covers `[t0 + k*0.02, t0 + (k+1)*0.02)`; its centre is `t0 + (k+0.5)*0.02`.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` (all probes merged) | `neural[session][trial]` (n_neurons, 100) float32 | spike counts in 20 ms bins, units restricted to `label>=1` and grey-matter Beryl regions | `prepare_data`, `merge_probes`, `load_spiking_data(qc=1)`, `bin_spiking_data` | dtype float32 as required by the decoder |
| time relative to stimOn | `input[·][·][0, :]` | bin-centre time in seconds, -0.49 ... 1.49 (same for every trial) | (new; required by Decoder Task) | "Time since stimulus onset", continuous, time-varying; negative before onset |
| `trials.probabilityLeft` | `input[·][·][1, :]` | trial index within the current block (0-based, counted over *all* trials of the session), broadcast along time | (new) | "Trial number in block", continuous, per-trial |
| `trials.choice` | `output[·][·][0, :]` | +1 (left) -> 0, -1 (right) -> 1, broadcast along time | `bin_behaviors` (`choice`) | binary, per-trial |
| `trials.probabilityLeft` | `output[·][·][1, :]` | 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, broadcast along time | `bin_behaviors` (`block`) | 3 classes, per-trial |
| `wheel.velocity` | `output[·][·][2, :]` | `abs(velocity)`, interpolated to the 100 bin right-edges, then discretised into 3 per-session tertile bins | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | time-varying |
| `leftCamera.whiskerMotionEnergy` (fallback right) | `output[·][·][3, :]` | interpolated to the 100 bin right-edges, then discretised into 3 per-session tertile bins | `load_target_behavior('left-whisker-motion-energy')`, `bin_behaviors` | time-varying |
| `clusters.acronym` -> Beryl | `brain_regions`, `brain_region_idx` | `BrainRegions().acronym2acronym(..., 'Beryl')` | `list_brain_regions` | `root`/`void` units dropped |
| session table `subject` | `subjects`, `subject_idx` | unique subject names | `bwm_release.csv` | |

### Key Decisions
1. **ONE access**: `ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True, cache_dir='/app/data/one_cache')`, exactly as `src/0_data_caching.py`. Works offline from the staged REST cache and resolves the staged dataset revisions (a `mode='local'` ONE cannot see them). All loading goes through `brainbox.io.one.SessionLoader` / `SpikeSortingLoader` (via the reference functions), never by reading files directly.
2. **Sessions**: all 459 eids of `bwm_release.csv` (the BWM release); a session is dropped if it has no whisker motion energy at all, no unit surviving QC, or fewer than 2 usable trials.
3. **Probes merged** per session (`merge_probes`), as both papers do ("neurons in the same session and region were combined across probes").
4. **Units**: `label >= 1` (well-isolated, data paper's three RIGOR metrics) and Beryl region not in {`root`, `void`} (grey matter).
5. **Trials**: reference `load_trials_and_mask(max_trial_len=10.0)` mask AND complete wheel + whisker coverage of the 2 s interval (reference `get_behavior_per_interval` criteria).
6. **Spike binning**: 20 ms counts, vectorised with `np.searchsorted` + `np.bincount` (mathematically identical to the reference `bincount2D` per-trial loop, verified by a direct comparison in Step 10).
7. **Behaviour binning**: linear interpolation onto the bin right-edges `linspace(t0+0.02, t1, 100)`, identical to `get_behavior_per_interval`.
8. **Discretisation into 3 bins**: per-session tertiles (33.3/66.7 percentiles over all retained trials x bins of that session). Per-session rather than global because whisker motion energy is in arbitrary camera-dependent units (its session medians vary by an order of magnitude), so a global threshold would collapse whole sessions into one class; per-session tertiles also give the balanced 1/3 classes that balanced accuracy is defined against. Applied to wheel speed as well for consistency.
9. **Per-trial variables are broadcast along time** so that every trial has the same (d, T) shape - the format spec asks for time-varying representations "if at all possible", and the decoder requires a consistent `dinput`/`doutput`.
10. **Metadata** records alignment event, off_start=-0.5, off_end=+1.5, bin size 20 ms, QC rules, discretisation thresholds per session and session/eid lists.

### Planned Sanity Checks
- [ ] Session/subject counts: <= 459 sessions, <= 139 subjects, 12 labs.
- [ ] Trials/session mean ~ 420 (survey: 426.5 pass the trial mask before behaviour rejection).
- [ ] Neurons/probe ~ 108 (paper) -> neurons/session ~ 150-170 after merging probes; total well-isolated units <= 75,708.
- [ ] Every trial: neural shape (n_neurons, 100), input (2, 100), output (4, 100).
- [ ] Choice distribution ~ 50/50; prior distribution ~ {0.2: 0.41, 0.5: 0.16, 0.8: 0.43}; wheel/whisker classes ~ 1/3 each.
- [ ] Time input ranges exactly [-0.49, 1.49]; trial-in-block index within [0, ~100].
- [ ] Spot-check spike counts, wheel speed, whisker ME and trial variables against independently loaded raw data with `np.allclose` (Step 10).
- [ ] Reference-implementation cross-check: run the reference `bin_spiking_data` / `bin_behaviors` on one session and compare with my vectorised output.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (`python -u /app/convert_data.py <out.pkl> [--full|--sample] [--show-processing]`).

Structure:
- `get_one()` - single ONE client per process, built like the reference caching script.
- `load_spiking_data()` - copy of the reference function with `qc=1` (well-isolated units) and
  without the network-only `raw_electrophysiology(...).fs` call.
- `load_session_neural()` - loops over `one.eid2pid(eid)`, merges probes with the reference
  `merge_probes`, maps acronyms to Beryl (`BrainRegions.acronym2acronym`), drops `root`/`void`
  units, re-indexes clusters and re-sorts spikes.
- `load_trials_and_mask()` - imported unchanged from the reference module (with an explicit
  `sess_loader=SessionLoader(one=one, eid=eid)` because the installed brainbox `SessionLoader`
  is a keyword-only dataclass).
- `bin_spikes()` - vectorised 20 ms spike counts (searchsorted + bincount per trial).
- `bin_behavior()` - linear interpolation onto `t0 + (1..100)*0.02` (the reference right-edge
  grid) plus the reference validity criteria (data present, starts <= 1 bin late, ends <= 1 bin
  early, no NaNs inside the interval).
- `discretize_tertiles()` - 3 classes at the per-session 33.3/66.7 percentiles.
- `trial_number_in_block()` - 0-based trial index within each run of constant `probabilityLeft`.
- `process_session()` - per-session pipeline; returns per-session dict or a skip reason.
- `plot_processing()` - `--show-processing` diagnostics: spike-count image, raw-vs-binned wheel
  speed with the first-movement marker, wheel discretisation with the tertile edges, raw-vs-binned
  whisker ME, whisker discretisation, and a session-level distribution figure.
- `main()` - multiprocessing over sessions (`--n-workers`, default 24), assembly of the target
  dictionary and metadata, and summary statistics.

Code inefficiencies identified: the reference code bins each trial in a separate multiprocessing
task (`get_spike_data_per_interval`, `interpolate_behavior`), which costs more in task overhead
than the binning itself.

Code speedups added: binning is vectorised per session and parallelism is at the session level;
spike sorting is loaded once per probe; behaviours are interpolated in one `np.interp` call.
Result: 4-9 s per session (spike-sorting load dominates: 3.5-8.3 s), 0.01-0.03 s for all binning.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(-> `/app/conversion_sample_out.txt`), then
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` (-> `/app/verification_sample_out.txt`).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (004d8fd5..., 02fbb6da...) |
| Neurons (total) | 204 (116 + 88) |
| Neurons / session | 102 mean |
| Subjects | 2 (DY_010, MFD_05) |
| Sessions / subject | 1 |
| Trials (total) | 644 |
| Trials / session | 198, 446 |
| T | 100 for every trial |
| Brain regions | 13 Beryl regions |
| time_from_stim_onset range | [-0.49, 1.49] (printed as [-0.5, 1.5]) |
| trial_number_in_block range | [0, 94] |
| choice distribution | [0.500, 0.500] |
| prior distribution | [0.443, 0.118, 0.439] |
| wheel_speed distribution | [0.333, 0.333, 0.333] |
| whisker ME distribution | [0.333, 0.333, 0.333] |

Verification output: **"Data format is valid, no errors or warnings."**

Consistency: the kept trial counts (198 and 446) are *identical* to the independent survey of the
reference trial mask for these eids (`/tmp/survey.csv`), i.e. the behavioural-coverage criterion
rejected no additional trial in these two sessions. Prior class fractions match the whole-release
survey (0.41/0.16/0.43). Wheel/whisker classes are exactly balanced by construction (tertiles).
The Beryl region `x` is "Nucleus x" (a real grey-matter region), not an unassigned label.

### Processing Plots Review
`processing_<eid>.png` / `processing_<eid>_dist.png`: binned wheel speed and whisker ME track the
raw traces with no time offset; wheel speed rises just after the plotted first-movement marker,
which itself falls 0.08-2 s after the stimulus-onset line at t = 0; the discretised class traces
switch exactly where the continuous traces cross the tertile edges; spike-count images show the
expected post-stimulus response. No anomalies.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| Vectorised spike/behaviour binning instead of per-trial multiprocessing tasks | binning reduced to 0.01-0.03 s/session (was the reference bottleneck) |
| Session-level multiprocessing (24 workers) | ~24x wall-clock |
| Single spike-sorting load per probe | avoids re-reading 20 M spike times |

| Step | Time / Session | Estimated Total Time |
| trials + behaviour load | 0.8 s | 6 min serial |
| spike sorting load (dominant, scales with n probes) | 3.5-8.3 s | 40-60 min serial |
| binning + assembly | 0.05 s | < 1 min |
| **total** | ~7 s (1 probe) / ~13 s (2 probes) | **~4-8 min with 24 workers** |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: None
- Warnings: None

Training loss decreased monotonically over 200 epochs; test loss 0.818.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|---|
| choice | 0.6151 | 0.6088 | 0.5 |
| prior_prob_left | 0.6782 | 0.6783 | 0.3333 |
| wheel_speed | 0.5490 | 0.5336 | 0.3333 |
| whisker_motion_energy | 0.5339 | 0.5193 | 0.3333 |

All outputs are clearly above chance on validation data with only two sessions, and the
train/validation gap is negligible (no overfitting or leakage).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --n-workers 24`
(-> `/app/conversion_full_out.txt`, 200 s wall-clock for all 459 candidate sessions),
then `python -u /app/train_decoder.py /app/converted_data.pkl --verify-only`
(-> `/app/verification_full_out.txt`).

### Output Files
- `converted_data.pkl`: 11.43 GB
- `verification_full_out.txt`: created, reports **"Data format is valid, no errors or warnings."**

### Converted dataset
| Statistic | Value |
|---|---|
| Sessions | 441 (of 459 released; 18 skipped) |
| Subjects | 136 (of 139) |
| Trials | 187,936 (mean 426.2, median 392, min 125, max 1445) |
| Neurons | 62,757 (mean 142.3/session, min 7, max 516) |
| Beryl regions | 263 |
| T | 100 bins of 20 ms for every trial |
| Clusters before QC | 599,022 = 890.1/probe |
| Units with `label >= 1` | 73,010 = 108.5/probe |
| Units after grey-matter filter | 62,757 |
| Camera used for whisker ME | 424 left, 17 right |

Skipped sessions (18): 14 have no whisker motion energy from either camera; 1 (f8041c1e) has
only right-camera motion energy that does not cover any masked trial; 3 have fewer than 5
well-isolated grey-matter units (1, 2 and 3 units).

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | 459 released / 433 analysed (methods paper) | `bwm_release.csv`: 459 eids | 459 | 441 | Yes - between the released and analysed counts; the 18 exclusions are all explained (missing whisker video or < 5 units) |
| Insertions | 699 | 699 pids | 699 | 673 probes contributed (699 minus probes of skipped sessions and 1 probe without spike sorting) | Yes |
| Subjects | 139 | 139 | 139 | 136 | Yes - the 3 missing subjects only had skipped sessions |
| Clusters/probe | 889 | - | 890.1 measured | 890.1 | Yes |
| Well-isolated units/probe | 108 | - | 108.5 measured | 108.5 (62,757 after the grey-matter filter) | Yes |
| Well-isolated units total | 75,708 (all 699 probes) | - | 73,010 over the 673 probes used | 62,757 after grey-matter filter | Yes - the difference is the skipped probes plus `root`/`void` units |
| Trials/session (raw) | >= 250 required | - | 645.1 | - | Yes |
| Trials/session after curation | - | mask keeps 426.5 (survey) | 426.5 | 426.2 | Yes |
| Trials total after curation | - | 195,781 over 459 sessions | 195,781 | 187,936 over 441 sessions | Yes |
| T / bin size | 100 bins / 20 ms | 100 bins / 20 ms | - | 100 / 20 ms | Yes |
| Alignment | stimOn, -0.5 to +1.5 s | stimOn, (-0.5, 1.5) | - | same | Yes |
| Beryl regions | 270 (methods paper) | - | 308 exist in the atlas | 263 | Yes - 263 of the paper's 270, the rest were in skipped sessions or are non-grey-matter |
| choice distribution | ~50/50 | - | 0.494 right (survey) | [0.508 left, 0.492 right] | Yes |
| prior distribution | 90 unbiased trials then alternating 0.2/0.8 blocks | - | [0.417, 0.141, 0.442] (survey: 0.43/0.16/0.41 before behaviour filtering) | [0.417, 0.140, 0.442] | Yes |
| wheel speed classes | - | - | - | [0.333, 0.333, 0.333] | Balanced by construction |
| whisker ME classes | - | - | - | [0.333, 0.333, 0.335] | Balanced by construction |
| time input range | -0.5 to 1.5 s | - | - | [-0.49, 1.49] (bin centres) | Yes |
| trial-in-block input | blocks of 20-100 trials (mean 51), first block 90 | - | - | [0, 98], mean 31.2 | Yes |

### Issues found and fixed during Step 9
1. **One probe without spike sorting** (`dfd8e7df`, `clusters is None`) raised an exception and lost
   the whole session. Fixed: such probes are skipped, the session is kept with its remaining probes
   (recorded in `metadata.session_info[...]['probes_without_spike_sorting']`).
2. **Camera choice**: taking the left camera whenever it exists (the literal reference fallback) gave
   0 usable trials for `f8041c1e`, whose right-camera motion energy covers only part of the session.
   Fixed: when both cameras are available the one covering more trials is used (left preferred on a
   tie). 424 sessions use the left camera, 17 the right.
3. **Sessions with < 5 well-isolated units** (1, 2 and 3 units) produced many all-zero trials. The
   data paper requires at least five well-isolated neurons per session, so these sessions are now
   dropped.
4. **Trials with zero spikes in the whole 2 s window** (16 trials in 3 sessions) were traced to gaps
   in the ephys recording (e.g. a 5.1 s gap in `b182b754`) or to trials that occur after the ephys
   recording ended while the behavioural session continued (`8c2f7f4d`, 3 trials). They are dropped:
   they carry no neural information and were the only remaining verifier warnings.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
- **Errors: none.** The log's first line is "Data format is valid, no errors or warnings."
- Warnings: none remain. The earlier run reported 16 "all neural data is zero" warnings; these were
  investigated (see Step 9, issue 4), traced to ephys recording gaps / recordings that end before the
  behavioural session, and those trials are now dropped. Nothing had to be left unfixed.

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`, output `/app/cache/sanity_checks_out.txt`)
The script re-loads the raw data with ONE/brainbox and re-implements the binning independently (it
does **not** import the conversion's binning helpers), for 3 randomly chosen sessions (280, 225, 373)
and converted trials 0, 1 and 5 of each:

| Check | Data stream | Result |
|---|---|---|
| Brute-force recomputed 20 ms spike-count matrix equals the stored matrix (`np.allclose`) and identifies the matching raw trial | neural | PASS for all 9 trials |
| Beryl acronym per neuron equals `brain_regions[brain_region_idx]` | neural/metadata | PASS for all 3 sessions |
| `time_from_stim_onset` row equals the bin centres `-0.49 ... 1.49` | input | PASS |
| `trial_number_in_block` equals the independently recomputed index within the `probabilityLeft` run | input | PASS |
| choice class equals `int(trials.choice < 0)` of the matched raw trial | output | PASS |
| prior class equals the {0.2,0.5,0.8} -> {0,1,2} map of the matched raw trial | output | PASS |
| re-interpolated wheel speed, discretised with the stored tertile edges, equals the stored classes | output | PASS |
| re-interpolated whisker ME, discretised with the stored tertile edges, equals the stored classes | output | PASS |
| First 50 (choice, prior, trial-in-block) triples of each session exist among the independently recomputed curated trials | input/output | PASS |

One subtlety found and explained: in session 280, trial 5, a single time bin differs by one class.
Its wheel speed is 0.0218028584, within 1e-9 of the session's 33.3rd-percentile edge (that bin *is*
the percentile sample). The conversion stores the behaviour in float32 and the check interpolates in
float64, so an exact tie can land on either side. The check now treats samples within 1e-6 of an edge
as ties; with that, **ALL SANITY CHECKS PASSED**.

### Check 3: Reference code comparison (`/app/cache/reference_comparison.py`, output `/app/cache/reference_comparison_out.txt`)
| Stage | Reference | This conversion | Match |
|---|---|---|---|
| (a) data loading | `prepare_data`: `one.eid2pid` -> `load_spiking_data` per probe -> `merge_probes`; `SessionLoader` for trials/wheel/motion energy | identical; `merge_probes` and `load_trials_and_mask` are **imported from the reference module**; `load_spiking_data` copied with `qc=1` and without the network-only `fs` read | Yes (documented deviation: `qc=1`) |
| (b) neuron filtering | `prepare_data` keeps all clusters but records `label >= 1`; `qc=1` option exists; `root`/`void` dropped in `MultiRegionDataModule` | `label >= 1` + `root`/`void` dropped + sessions with < 5 such units dropped | Deliberate: matches the data paper's "well-isolated neurons" and grey-matter restriction |
| (b) trial filtering | `load_trials_and_mask(max_trial_len=10.0)` + behaviour interval rejection + `align_spike_behavior` | same function imported unchanged; behaviour rejection re-implemented with the same four criteria; the conjunction that `align_spike_behavior` intended (its `list and list` is a Python bug) is applied; additionally trials with no spikes at all are dropped | Yes (bug fixed, one addition) |
| (c) temporal alignment | intervals `stimOn + (-0.5, 1.5)`; behaviour grid `linspace(t_beg+binsize, t_end, 100)` | identical; verified numerically (grid max difference 5.7e-14 s) | Yes |
| (d) binning | `bincount2D(times, clusters, xbin=0.02, xlim=[t_beg, t_end])`, truncated to 100 bins | vectorised `searchsorted` + `bincount` | **Bit-identical**: "spike counts identical to the reference implementation: True, max abs diff 0.0" over all 565 trials of the test session |
| (d) behaviour binning | `get_behavior_per_interval` (linear interpolation, validity mask) | `bin_behavior` | Validity masks agree on **100%** of trials. Values agree to float32 precision in 99 of 100 bins; only the **last** bin differs (mean abs diff 2.2 for whisker ME, 1.3e-2 rad/s for wheel speed). Reason: the reference restricts the interpolation source to samples *strictly inside* the interval and therefore linearly **extrapolates** its last grid point (which sits exactly at the interval end), whereas this conversion interpolates from the full stream and uses the true neighbouring sample. Keeping the more accurate value is a deliberate improvement and affects 1% of bins. |
| (e) input construction | reference caches `choice/block/reward/contrast` and the binned behaviours; no time/trial-in-block regressor | time-from-onset and trial-number-in-block are new, as mandated by the Decoder Task | Required by the task |
| (f) output construction | `choice` in {-1, +1}, `block` in {0.2, 0.5, 0.8}, continuous wheel speed / whisker ME (used with regression) | same variables, mapped to the class codes the Decoder Task specifies, continuous ones discretised into 3 tertile bins | Required by the task (outputs must be categorical) |

### Check 4: Key statistics comparison
See the table in Step 9. Every statistic available in the reference texts is reproduced:
890.1 clusters/probe vs 889 in the paper, 108.5 well-isolated units/probe vs 108, 459 candidate
sessions / 699 insertions / 139 subjects from `bwm_release.csv`, 441 converted sessions (the methods
paper analysed 433), 100 bins of 20 ms, stimOn alignment from -0.5 to +1.5 s, choice ~50/50,
p(left) fractions [0.417, 0.140, 0.442] consistent with 90 unbiased trials followed by alternating
0.2/0.8 blocks of mean length ~51 trials, and a trial-in-block index bounded by 98 (block lengths
are truncated at 100, and the first block has 90 trials).

### Check 5: Edge cases
- Trials with NaN `stimOn_times` cannot define an interval: excluded by the mask *and* by an explicit
  `np.isfinite(align)` test (the placeholder interval used to keep array shapes is never kept).
- Sessions where one probe has no spike sorting: that probe is skipped, the session is kept.
- Sessions with only right-camera motion energy, or with a camera that covers only part of the
  session: the camera with the better coverage is used; if neither covers any curated trial the
  session is skipped (1 session).
- Sessions with < 5 well-isolated grey-matter units (3 sessions) and sessions with < 2 usable trials
  are skipped; `MIN_TRIALS = 2` guarantees the decoder can split train/validation.
- Trials that fall in ephys recording gaps (all-zero spike counts) are dropped (16 trials).
- `probabilityLeft` values other than {0.2, 0.5, 0.8} would abort the session (never triggered).
- Block boundaries: `trial_number_in_block` restarts at 0 at each change of `probabilityLeft`,
  including the transition out of the initial 90-trial unbiased block, and is computed over *all*
  trials (not only curated ones) so that the index reflects the animal's actual experience.
- Bin edges: bin k is `[t0+0.02k, t0+0.02(k+1))`, the last bin ends exactly at stimOn+1.5 s; spikes
  exactly at the interval end are excluded (`side='left'`), matching `bincount2D`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` ->
`/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`. Trained on the GPU
(NVIDIA L4) over all 441 sessions / 187,936 trials.

### Training Progress
- Loss decreasing: **Yes** (1.42 -> 1.32 at epoch 10 -> 0.882 at 50 -> 0.777 at 100 -> 0.752 at 150
  -> 0.7419 at 200; test loss 0.7654).

### Decoder Results (Full)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|--------|-------------|--------|-------|
| choice | 0.500 | 0.6378 | 0.6152 | 1.23x chance |
| prior_prob_left | 0.333 | 0.6772 | 0.6595 | 1.98x chance |
| wheel_speed (3 bins) | 0.333 | 0.6137 | 0.6073 | 1.82x chance |
| whisker_motion_energy (3 bins) | 0.333 | 0.6029 | 0.5976 | 1.79x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
Every output is well above chance, and every output is above 1.5x chance except choice (1.23x),
which is a binary variable whose chance level is already 0.5; in relative terms it removes 23% of
the distance to perfect. No output is at or below chance, so there is no sign of a label or
alignment bug. As an additional control, the two per-trial outputs (choice, prior) are decoded far
above chance *while* the two time-varying outputs are also decoded above chance, which could not
happen if the neural/behaviour time axes were misaligned.

### Check 2: Accuracy comparison to the papers
| Variable | This conversion (validation balanced acc) | Reported in the papers | Assessment |
|---|---|---|---|
| choice | 0.615 | BWM data paper: null-corrected median balanced accuracy of significant single regions ~0.55-0.65 (whole-session decoders are not reported); methods paper: AUC 0.66-0.79 for session-level linear/RRR decoders, example accuracies 0.71 (linear) to 0.90 (BMM-HMM, which exploits across-trial correlations) | Consistent. 0.615 balanced accuracy for a single decoder shared across 441 sessions with only 10 PCs per session is in the same range as the paper's single-region results, and below the best session-specific models, which is expected: the papers fit a *separate* model per session (and per region) with all neurons, while here one model must generalise across 441 sessions through a 10-dimensional per-session projection. |
| prior (p(left)) | 0.660 (3 classes) | methods paper: Pearson correlation 0.05 (single-session baseline) to 0.65 (oracle) for the continuous prior | Consistent / favourable: the block variable is decoded at twice chance. |
| wheel speed | 0.607 (3 classes) | data paper: decoding R^2 over null (significant in motor/sensory regions); methods paper: RRR > ridge for continuous wheel speed | Consistent; not directly comparable because the papers regress the continuous signal. |
| whisker motion energy | 0.598 (3 classes) | methods paper: reliable reconstruction of trial-averaged and trial-specific whisker ME | Consistent, same caveat. |

The remaining gap to the best published choice decoders is explained by model class (per-session
regularised logistic regression on all neurons and a task-specific window, vs. a single shared
PCA-10 + decoder over all sessions), not by the data: the spike counts are bit-identical to the
reference implementation (Step 10, Check 3) and every spot-check of inputs/outputs passes.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| choice | 0.6378 | 0.6152 | 1.04 |
| prior_prob_left | 0.6772 | 0.6595 | 1.03 |
| wheel_speed | 0.6137 | 0.6073 | 1.01 |
| whisker_motion_energy | 0.6029 | 0.5976 | 1.01 |

All ratios are ~1.0, far below the 1.5x threshold: no overfitting and no data leakage (each trial
appears in exactly one session list, and per-trial variables are broadcast within a trial only).

### Additional debugging performed
1. Output values verified against raw data for 9 specific trials in 3 sessions (Step 10, Check 2).
2. Temporal alignment verified visually (`processing_<eid>.png`: wheel speed rises after the marked
   first-movement time, which lies 0.08-2 s after the stimulus-onset line) and numerically (the
   behaviour grid agrees with the reference to 5.7e-14 s; spike counts are bit-identical).
3. Class balance checked: choice [0.508, 0.492], prior [0.417, 0.140, 0.442], wheel and whisker
   classes 1/3 each. No output is dominated by one class.
4. Neural filtering checked against the paper: 890.1 clusters/probe and 108.5 well-isolated
   units/probe reproduce the published 889 and 108.
5. `sample_trials.png` / `predictions.png` inspected: predictions track the discretised wheel-speed
   and whisker-ME time courses within trials.

### Issues Found and Resolved
- **`--show-processing` crashed after the camera-selection refactor** (`NameError: me_times`): the
  diagnostic plotting call still referenced the pre-refactor variables. Fixed by passing the selected
  camera's arrays (`cameras[camera_used]`). This only affected the plotting path, not the data
  (`--full` never calls it), but the sample artefacts (`sample_data.pkl`,
  `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
  `cache/plots/processing_*.png`) were regenerated with the final script; the sample results are
  unchanged (2 sessions, 644 trials, 204 neurons, "Data format is valid, no errors or warnings",
  validation balanced accuracy choice 0.609 / prior 0.678 / wheel 0.534 / whisker 0.519).
- float32/float64 tie at a tertile edge (1 bin in 1 of 9 spot-checked trials): explained as a
  numerical tie, not a bug; the sanity check now allows ties.
- Last-bin behaviour value differs from the reference because the reference extrapolates; kept the
  more accurate interpolation and documented it.
- No other issues remained after the Step 9 fixes.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created (`/app/README.md`): dataset description, key statistics, loading example,
      format specification, curation summary.
- [x] cache/ folder created (`/app/cache/`) with `README_CACHE.md` documenting every cached script:
      `survey.py` + `survey.csv`, `sanity_checks.py` (+ output), `reference_comparison.py`
      (+ output), the three investigation scripts, and `plots/processing_*.png`.
- [x] All files organised. Deliverables in `/app`: `CONVERSION_NOTES.md`, `convert_data.py`,
      `converted_data.pkl`, `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`,
      `verification_sample_out.txt`, `train_decoder_sample_out.txt`, `conversion_full_out.txt`,
      `verification_full_out.txt`, `train_decoder_full_out.txt`, plus the decoder figures
      `sample_trials.png` and `predictions.png`.
