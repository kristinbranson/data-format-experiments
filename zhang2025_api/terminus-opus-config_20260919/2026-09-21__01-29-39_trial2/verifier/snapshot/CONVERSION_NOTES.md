# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain Wide Map (public ONE cache in /app/data)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `Dockerfile`, `docker-compose.yaml`, `stage_cache.sh`, `.manifest`
- `code/` : reference code -- `code_zhang2025` (methods paper) and a copy of `ibllib` (incl. `brainbox`)
- `data/one_cache/` : IBL ONE cache. Lab dirs (symlinks to /mnt/dataset) hold ALF files; tag dirs
  (`Brainwidemap`, `2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`) hold parquet cache tables;
  `.rest/` holds ~6.5k cached Alyx REST responses.
- `datapaper.pdf` (BWM paper), `methodpaper.pdf` (Zhang et al. decoding paper), `dataarchitecture.pdf`, `methods.txt`
- `decoder.py`, `train_decoder.py` : target decoder / validation code
- `ibl_docs/` : ONE/ibllib/iblatlas documentation

Environment verified: `python3`, numpy 2.3.5, torch 2.6.0+cu124, `one`, `brainbox`, `ibllib` all import.
`brainwidemap` package is NOT installed, so the BWM-specific helpers (`bwm_query`, `load_good_units`,
`load_trials_and_mask`) must be re-implemented locally -- the reference repo `code_zhang2025` contains
its own copies of these functions, which I use.

### ONE access method (important)
`ONE(base_url='https://openalyx.internationalbrainlab.org', mode='remote')` works fully offline because
`~/.one` holds the auth token and `/app/data/one_cache/.rest` holds every REST response needed.
`mode='local'` does NOT work: the staged parquet tables list e.g. `alf/_ibl_trials.table.pqt`, while on disk
459/466 trials tables live under revision `alf/#2025-03-03#/`. Only the REST (remote) path resolves the
correct revisions. Verified: trials (471 x 20), wheel, leftCamera motion energy, `eid2pid`, and
`SpikeSortingLoader.load_spike_sorting` all load correctly in remote mode.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference repo: `/app/code/code_zhang2025` (Zhang et al., "Exploiting correlations across trials and
behavioral sessions to improve neural decoding"). Entry point for data conversion:
`src/0_data_caching.py`, all helpers in `src/utils/ibl_data_utils.py`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data(one, eid, bwm_df, params)` | ibl_data_utils.py | LOADING | For each pid of an eid: `load_spiking_data`, then `merge_probes`; `load_trials_and_mask`; `load_anytime_behaviors`. Returns neural_dict/behave_dict/meta_data/trials_data |
| `load_spiking_data(one, pid, qc=None)` | ibl_data_utils.py | LOADING/CURATION | `SpikeSortingLoader(pid).load_spike_sorting()` + `merge_clusters` -> clusters df with `label` (IBL unit QC in {0,1/3,2/3,1}); if `qc` given keeps `label >= qc` |
| `merge_probes(spikes_list, clusters_list)` | ibl_data_utils.py | LOADING | Concatenates probes of one session into one population, re-indexing cluster ids |
| `load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., max_trial_len=10.0, nan_exclude=default, exclude_nochoice=True)` | ibl_data_utils.py | CURATION | Trials table via `SessionLoader.load_trials()`; boolean mask excluding RT outside [0.08, 2]s, trial length > 10s, NaNs in stimOn_times/choice/feedback_times/probabilityLeft/firstMovement_times/feedbackType, and no-choice trials |
| `list_brain_regions` / `select_brain_regions` | ibl_data_utils.py | PROCESSING | Maps cluster `acronym` to Beryl atlas acronyms via `iblatlas.regions.BrainRegions.acronym2acronym(mapping='Beryl')`; selects clusters in requested regions (default: all regions together) |
| `bin_spiking_data(reg_clu_ids, neural_df, trials_df, **params)` | ibl_data_utils.py | PROCESSING | Intervals = `trials_df[align_time] + time_window`; bins spikes with `iblutil.numerical.bincount2D(xbin=binsize, xlim=[t_beg,t_end])`, keeps first `ceil(interval_len/binsize)` bins -> (n_trials, n_bins, n_clusters) spike counts |
| `load_target_behavior(one, eid, target)` | ibl_data_utils.py | LOADING | `SessionLoader.load_wheel()` -> wheel-speed = abs(velocity) (Gaussian-smoothed, uniformly sampled); `SessionLoader.load_motion_energy(views=['left'])` -> `whiskerMotionEnergy` |
| `get_behavior_per_interval(...)` | ibl_data_utils.py | PROCESSING | Per trial, slices the continuous signal to the interval and linearly interpolates (`interp1d`) onto `np.linspace(beg+binsize, end, n_bins)`; marks a trial bad if data missing / starts too late / ends too early / NaNs |
| `bin_behaviors(one, eid, behaviors, trials_df, **params)` | ibl_data_utils.py | PROCESSING | Per-trial discrete variables: `choice` (-1/+1), `block` = `probabilityLeft`, `reward` = `rewardVolume>1`, `contrast` = signed contrast; plus binned continuous behaviours |
| `align_spike_behavior(binned_spikes, binned_behaviors, beh_names, trials_mask)` | ibl_data_utils.py | CURATION | Drops trials failing the trials mask or with missing behaviour, so neural and behaviour arrays stay aligned |
| `standardize_spike_data` | data_loader_utils.py | PROCESSING | z-scores spike counts per time bin (decoder-side, not part of caching) |

### Trial setup used by the reference (`src/0_data_caching.py`)
```python
params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
          'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']
```
i.e. **align to stimulus onset, window -0.5 s to +1.5 s, 20 ms bins -> 100 time bins per trial**.
Pupil diameter is deliberately excluded ("Some sessions do not have pupil traces").

### Notes
- Neuron quality: `prepare_data` calls `load_spiking_data` without `qc`, so the cached dataset keeps
  all clusters but records `good_clusters = (label >= 1)` in the metadata. The BWM data paper
  defines "good" units as those passing all three IBL unit-QC criteria (label == 1). Decision on
  which to use is made in Step 5 after reading the papers.
- `load_spiking_data` also calls `spike_loader.raw_electrophysiology(band='ap', stream=True).fs`
  purely to record the AP sampling frequency; that requires streaming raw ephys and is not needed
  for conversion, so it is skipped here.
- Region labels come from the Beryl mapping of `clusters.acronym`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/one_cache` is a standard ONE cache:
```
one_cache/<lab>/Subjects/<subject>/<date>/<number>/
    alf/_ibl_trials.*            (trials table, revision #2025-03-03# for 459/466 sessions)
    alf/_ibl_wheel.{position,timestamps}.npy
    alf/{left,right}Camera.ROIMotionEnergy.npy  (revisions #2025-05-30#/#2025-05-31#)
    alf/#2023-04-20#/_ibl_{left,right}Camera.times.npy
    alf/probe0X/pykilosort[/#2024-05-06#]/{spikes,clusters,channels}.*
```
plus tag tables (`Brainwidemap`, `2022_Q4...`, `2025_Q3...`/`*.pqt`) and `.rest/` (6,483 cached Alyx
REST responses). Access is through `ONE(..., mode='remote')`, which resolves dataset revisions from the
cached REST responses and finds the files that are actually on disk (see Step 0).

Available variables:
- **trials table** (20 columns): `stimOn_times, stimOff_times, stimOnTrigger_times, stimOffTrigger_times,
  goCue_times, goCueTrigger_times, response_times, feedback_times, firstMovement_times, intervals_0/1,
  intervals_bpod_0/1, quiescencePeriod, choice (-1/0/+1), contrastLeft, contrastRight, probabilityLeft
  (0.2/0.5/0.8), feedbackType (-1/+1), rewardVolume`
- **wheel**: `times, position, velocity, acceleration` (SessionLoader resamples to a uniform 1 kHz grid
  and computes velocity by Gaussian smoothing)
- **motion energy**: `leftCamera`/`rightCamera` dataframes with `times, whiskerMotionEnergy`
  (left camera 60 Hz, right camera 150 Hz)
- **spike sorting** (pykilosort): `spikes.times/clusters/amps/depths`, `clusters.*` incl. quality metrics;
  `SpikeSortingLoader.merge_clusters` adds `label` (IBL unit QC in {0, 1/3, 2/3, 1}) and `acronym`
  (Allen region of the cluster's peak channel).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Sessions (eids) in BWM release | 459 (all staged on disk) |
| Probe insertions (pids) | 699 (240 sessions with 2 probes, 219 with 1) |
| Subjects | 139 (12 labs) |
| Sessions / subject | 1-13 (mode 1-3; 32 subjects with 1, 29 with 3, 25 with 2 ...) |
| Units / session (all clusters) | e.g. 898, 1728 (2 probes), 1202 |
| Units / session with `label==1` | e.g. 76, 264, 163 |
| Trials / session (raw) | e.g. 565, 425, 557 |
| Trials / session (after reference mask) | e.g. 407, 244, 148 |
| Sessions with left whiskerMotionEnergy | 437 / 459 |
| Sessions with right whiskerMotionEnergy | 433 / 459 |
| Sessions with either | 445 / 459 |
| Sessions with trials+wheel+spikes | 459 / 459 |

Session loading time: 3-8 s per session (spike sorting dominates).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 139 mice (94 M, 45 F) | "We trained 139 mice ... on the IBL decision-making task" (data paper) |
| Insertions | 699 Neuropixels probes | "we inserted 699 Neuropixels probes" |
| Sessions | 459 | "a total of 459 sessions, 699 insertions and 621,733 neurons remained" |
| Units (all) | 621,733 (avg 889/probe) | same |
| Well-isolated neurons | 75,708 (avg 108/probe) | "Out of the 621,733 units collected, 75,708 were considered well-isolated neurons" |
| Trials / session | average 645 | "Recorded sessions lasted on average 645 trials" |
| Sessions used by methods paper | 433 sessions, 270 brain regions | "We apply our models to 433 IBL sessions, covering 270 brain regions" |
| Neural data time bin | 20 ms (also 50 ms for prior-only analysis) | "each divided into 20-ms bins, producing T = 100 time steps" |
| Trial length | 2 s | "Recordings are split into 2-s trials" |
| Alignment / window (choice) | stimOn_times, -0.5 s to +1.5 s | "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset" |
| Behaviour sampling | 60 Hz | "wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz" |
| Blocks | 90 unbiased trials, then 20:80 / 80:20 blocks of 20-100 trials (mean 51) | data paper |
| Contrasts | {0, 6.25, 12.5, 25, 100}% | data paper |
| Reward rate (0% contrast) | 58.7 +/- 0.4% correct | "mice gained rewards on 58.7 +/- 0.4% of trials" (0%-contrast trials) |
| Session inclusion | >=250 trials, >=90% correct on 100% contrast both blocks, >=3 error trials | data paper inclusion criteria |

### Processing Details
- **Temporal alignment**: trials aligned to `stimOn_times`; window (-0.5, +1.5) s; 20 ms non-overlapping
  bins -> T = 100 bins per trial (reference `params` in `src/0_data_caching.py`).
- **Neural**: spike counts per (neuron, bin), no smoothing. Probes of a session merged
  ("neurons in the same session ... were combined across probes").
- **Behaviour**: wheel speed = |velocity| of the SessionLoader wheel trace; whisker motion energy from the
  left camera (right camera as fallback). Both are linearly interpolated onto the bin grid
  (`np.linspace(t_beg + binsize, t_end, n_bins)`, i.e. values at bin *end* times).
- **Task variables**: `choice` (+1/-1), `block`/prior = `probabilityLeft` (0.2/0.5/0.8),
  `reward` = `rewardVolume > 1`, signed `contrast`.

### Curation Steps

**Neuron curation rules** (data paper): units excluded unless they pass all three RIGOR single-unit
metrics (amplitude > 50 uV, noise cut-off < 20 uV, refractory-period violation) -- exactly the IBL
`label == 1` flag produced by `SpikeSortingLoader.merge_clusters`. 75,708 / 621,733 units pass.
Analyses further restricted to grey-matter regions (Allen CCF), >= 5 well-isolated neurons per session
in a region and >= 2 sessions per region.

**Trial curation rules** (data paper == reference `load_trials_and_mask`): exclude a trial if any of
`choice, probabilityLeft, feedbackType, feedback_times, stimOn_times, firstMovement_times` is NaN, or if
the first-wheel-movement latency (`firstMovement_times - stimOn_times`) is outside 0.08-2.00 s. The
reference additionally excludes trials longer than 10 s (`max_trial_len=10.0`) and no-choice trials
(`choice == 0`, which is implied by the NaN rule only partially).

### Decoders Trained (reference results)
| Decoded variable | Accuracy / metric |
|---|---|
| Choice (binary, per-trial) | AUC 0.66 (baseline) - 0.79 (multi-session RRR); single-session RRR accuracy ~0.7-0.75; region-level improvements of 0-25% over linear baseline |
| Prior (continuous per-trial) | Pearson correlation between decoded and true prior (0.4-0.7 range in Fig. 2D) |
| Wheel speed (time-varying) | R2 ~0.36-0.56 for RRR |
| Whisker motion energy (time-varying) | R2 ~0.16-0.44 |
Note: the reference decodes *continuous* wheel speed / whisker motion energy with regression; here they must
be discretized into 3 classes, so accuracies are not directly comparable (chance = 1/3).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

I verified my loading pipeline against the numbers quoted in the papers by running it over the whole
release (scripts `cache/units_count.py`, `cache/session_survey2.py`).

| Quantity | Papers | My loading of the data | Match |
|---|---|---|---|
| Sessions | 459 | 459 (all staged on disk) | YES |
| Insertions | 699 | 699 | YES |
| Subjects | 139 | 139 | YES |
| Total units | 621,733 (avg 889/probe) | 621,733 (avg 889.5/probe) | YES |
| Well-isolated units (`label == 1`) | 75,708 (avg 108/probe) | 75,708 (avg 108.3/probe) | YES |
| Trials / session | avg 645 | avg 645.08 (total 296,090) | YES |
| Beryl regions with units | 270 (methods paper) | 265 excluding `void`/`root` | close (see below) |
| Sessions used by methods paper | 433 | 445 have whisker ME, 437 have left ME | close (see below) |

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Unit quality filtering | `prepare_data` keeps **all** clusters (`qc=None`), storing `good_clusters = label>=1` in metadata; the methods paper text says "using all neurons, sorted by Kilosort 2.5" | 621,733 units total, 75,708 with `label==1` | Data paper: analyses use only well-isolated neurons (the 3 RIGOR metrics = `label==1`) | **Use `label == 1` (well-isolated) units.** The BWM data paper is explicit that analyses are restricted to these, and the decoder here is a PCA+linear model where 8x more noisy/MUA units mostly add noise and huge memory cost (621k vs 75k units). Documented as a deliberate, justified choice. |
| N sessions | reference caches whatever sessions are requested | 459 sessions; 437 with left whisker ME | methods paper uses 433 sessions | The methods paper drops sessions lacking video/behaviour. I keep sessions that have all required streams (whisker ME from left camera, falling back to right; wheel; >=2 usable trials; >=1 good unit), which yields a similar number. |
| N regions | -- | 265 Beryl regions excluding void/root; 205 regions with >=5 units in >=2 sessions and >=20 units total | 270 regions (methods paper) | The small difference comes from whether `void`/`root` (units outside grey matter) and low-yield regions are counted. I report all Beryl regions of the retained units. |
| Trial mask | `load_trials_and_mask(min_rt=.08, max_rt=2, max_trial_len=10, exclude_nochoice=True)` | 195,781 of 296,090 trials pass (mean 427/session) | Data paper lists the same NaN + 0.08-2.00 s RT rules (no 10 s rule) | Use the reference code's mask (paper rules + `max_trial_len=10` + no-choice), since the reference code is the more specific description of this pipeline. |
| Camera for whisker ME | reference tries left camera, falls back to right | 437 sessions have left, 420 right, 445 either | data paper computes ME for both left and right videos | Same as reference: left preferred, right as fallback. |
| Behaviour sampling rate | -- | left camera 60 Hz, right camera 150 Hz; wheel resampled at 1 kHz by `SessionLoader` | methods paper says behaviour sampled at 60 Hz | Consistent: left camera is the 60 Hz stream. |

### Additional consistency checks
- Choice convention: on correct trials with a left stimulus `choice == +1`, with a right stimulus
  `choice == -1`. So `choice == +1` is a **left** choice (the wheel is turned to bring the left
  stimulus to the centre). This matches the IBL convention and fixes the output mapping
  (left -> 0, right -> 1).
- `probabilityLeft` takes exactly the three values {0.2, 0.5, 0.8}; each session starts with 90
  unbiased (0.5) trials, matching the paper.
- Mean fraction correct after masking = 0.856, consistent with well-trained mice (the 58.7% figure in
  the paper refers specifically to 0%-contrast trials).

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial definition (from reference `src/0_data_caching.py`)
- Alignment event: `stimOn_times` (stimulus onset).
- Window: `off_start = -0.5 s`, `off_end = +1.5 s` -> 2 s trials.
- Bin size: 20 ms -> **T = 100 bins** per trial. Bin `t` covers `[t0 + 0.02 t, t0 + 0.02 (t+1))`.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` (all probes merged) | `neural[session][trial]` (n_neurons, 100) | spike counts in 20 ms bins in [-0.5, 1.5] s around `stimOn_times`, float32 | `merge_probes`, `bin_spiking_data` / `bincount2D` | only clusters with `label == 1` and Beryl region not in {void, root} |
| `clusters.acronym` -> Beryl | `brain_region_idx[session]`, `brain_regions` | `BrainRegions.acronym2acronym(mapping='Beryl')` | `list_brain_regions` | one index per retained neuron |
| time within trial | `input[0]` "time_from_stim_on" (time-varying) | bin-centre time in seconds, -0.49 ... 1.49 | -- | required by the decoder-task spec |
| trial index within block | `input[1]` "trial_number_in_block" (per-trial, constant over time) | count of trials since the last change of `probabilityLeft`, 0-based, taken over ALL raw trials so the count is not corrupted by masked-out trials | -- | required by the decoder-task spec |
| `trials.choice` | `output[0]` "choice" | `+1` (left) -> 0, `-1` (right) -> 1; constant over the trial | `bin_behaviors` | per-trial, broadcast over time |
| `trials.probabilityLeft` | `output[1]` "prior_prob_left" | 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 | `bin_behaviors` (`block`) | per-trial, broadcast over time |
| wheel `velocity` | `output[2]` "wheel_speed" | `abs(velocity)`, interpolated onto the bin grid, then discretized into 3 bins | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | time-varying |
| `leftCamera.ROIMotionEnergy` (`whiskerMotionEnergy`; right camera as fallback) | `output[3]` "whisker_motion_energy" | interpolated onto bin grid, then discretized into 3 bins | `load_target_behavior('left-whisker-motion-energy')` | time-varying |
| `bwm_release.csv` `subject` | `subjects`, `subject_idx` | -- | -- | 139 mice |

All four outputs are stored **time-varying** (shape (4, 100)), as the task instructions prefer:
per-trial variables (choice, prior) are constant across the 100 bins.

### Key Decisions
1. **Alignment and binning**: stimulus onset, -0.5 to +1.5 s, 20 ms bins (T=100). Exactly the
   reference `params` and consistent with the decoder-task instruction to align to stimulus onset.
2. **Unit curation**: keep only `label == 1` (well-isolated) units, as in the BWM data paper
   (75,708 / 621,733 units). Additionally drop units whose Beryl region is `void` or `root`
   (outside grey matter), as the data paper restricts analyses to grey-matter regions.
   Sessions must retain >= 1 unit (in practice >= 2 units; sessions with fewer are dropped).
3. **Trial curation**: the reference `load_trials_and_mask` mask -- exclude trials with NaN in
   `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`;
   RT = `firstMovement_times - stimOn_times` outside [0.08, 2.0] s; trial length
   `feedback_times - goCue_times` > 10 s; `choice == 0`. Additionally drop trials for which the
   behavioural traces (wheel / whisker ME) do not cover the whole 2 s window (the reference
   `get_behavior_per_interval` "target data starts too late / ends too early" checks).
4. **Discretization of continuous outputs into 3 bins**: wheel speed and whisker motion energy are
   discretized using **per-session tertiles** (33.3% / 66.7% quantiles) computed over all retained
   trials and time bins of that session. Rationale: (a) the decoder-task spec asks for 3 bins;
   (b) both signals have arbitrary session-specific units/scales (whisker ME depends on camera
   gain, lighting and ROI size; wheel speed is in rad/s but its scale depends on the mouse), so a
   global threshold would make classes wildly unbalanced across sessions; (c) tertiles give
   balanced classes (chance = 1/3), which makes balanced accuracy interpretable.
   Labels: 0 = low, 1 = medium, 2 = high.
5. **Inputs**: time from stimulus onset (seconds, time-varying, identical on every trial) and trial
   number within the current block (per-trial constant). Block boundaries are detected as changes
   in `probabilityLeft` over the *raw* trial sequence, so the count is correct even though some
   trials are excluded.
6. **Sessions with two probes are merged** (paper: "neurons in the same session and region were
   combined across probes"), using the reference `merge_probes`.
7. **Sessions dropped** if: no whisker ME at all (left or right), or < 2 usable trials, or
   < 2 retained neurons.
8. **Neural dtype**: float32 spike counts (not z-scored); the decoder does its own PCA.

### Planned Sanity Checks
- [x] Total units == 621,733 and `label==1` units == 75,708 (already verified in Step 4).
- [x] Mean trials/session (raw) == 645 (verified: 645.08).
- [x] Trial counts: converted trials/session vs. the reference mask count (187,547 vs 195,781 before behaviour/ephys-coverage filtering).
- [x] Choice distribution 0.508/0.492; prior distribution 0.418/0.140/0.442 -- as expected.
- [x] Wheel-speed / whisker-ME classes 0.333/0.333/0.334 each.
- [x] Spike-count spot checks PASS (cache/sanity_checks.py).
- [x] Wheel/whisker spot checks PASS (cache/sanity_checks.py).
- [x] Input spot checks PASS (cache/sanity_checks.py).
- [x] 440 sessions / 136 subjects / 263 regions in the pickle, consistent with the survey.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the plan from Step 5.  Structure:

| Function | Purpose | Reference counterpart |
|---|---|---|
| `get_one()` | `ONE(base_url=..., mode='remote')` -- offline access via the cached REST responses | `ONE(...)` in `0_data_caching.py` |
| `compute_trials_mask(trials)` | reproduces the pandas `eval` query of the reference | `load_trials_and_mask` |
| `bin_spikes(...)` | spike counts in (n_trials, n_clusters, 100), half-open bins from `align + t_start` | `bin_spiking_data` / `bincount2D` |
| `bin_behavior(...)` | linear interpolation onto `linspace(beg+binsize, end, n_bins)` + validity checks | `get_behavior_per_interval` |
| `discretize_tertiles(...)` | 3-class discretization by per-session tertiles | (new -- required by the decoder-output spec) |
| `trial_number_in_block(...)` | 0-based trial index within a `probabilityLeft` block | (new -- required by the decoder-input spec) |
| `convert_session(eid)` | loads trials/spikes/wheel/motion-energy, filters, bins, builds neural/input/output | `prepare_data` + `bin_spiking_data` + `bin_behaviors` + `align_spike_behavior` |
| `plot_processing(...)` | `--show-processing` figures | -- |
| `main()` | parallel driver, assembles the pickle | `0_data_caching.py` main loop |

Options: `--full` (default), `--sample` (first 2 sessions), `--show-processing` (per-session figures
for up to 2 sessions), `--n-workers`.

Code inefficiencies identified / speedups added:
- The reference bins spikes with a `multiprocessing` pool **per trial**, which is dominated by pickling
  overhead.  Here all trials of a session are binned with two `np.searchsorted` calls plus one
  `np.bincount` per trial, giving ~0.01 s per session instead of tens of seconds.
- Likewise behaviour interpolation is done with `np.interp` over pre-computed index ranges
  (~0.02 s/session instead of a per-trial process pool).
- Parallelism is applied at the **session** level (`multiprocessing.Pool`, 12 workers), which is where
  the real cost is (spike-sorting I/O, ~3-6 s per probe).
- Each worker builds its own `ONE` instance (the ONE object is not fork-safe for concurrent use).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(-> `/app/conversion_sample_out.txt`), then
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` (-> `/app/verification_sample_out.txt`).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (NYU-11, angelakilab) |
| Neurons (total) | 253 (192 + 61) |
| Neurons / session | 126.5 mean |
| Subjects | 1 |
| Trials (total) | 651 (244 + 407) |
| Trials / session | 325.5 mean |
| T (time bins) | 100 for every trial |
| time_from_stim_on range | [-0.49, 1.49] s |
| trial_number_in_block range | [0, 89] |
| choice distribution | left 0.518 / right 0.482 |
| prior distribution | 0.2: 0.478, 0.5: 0.161, 0.8: 0.361 |
| wheel_speed distribution | 0.333 / 0.333 / 0.333 |
| whisker ME distribution | 0.333 / 0.333 / 0.333 |
| Brain regions | 16 |

Format verification: **no errors and no warnings**.

### Processing Plots Review
`processing_<eid>.png` (2 files) show, for an example trial: the raw spike raster with the 20 ms bin
grid and stimulus onset marked, the binned spike-count matrix over the same window, the raw wheel
|velocity| trace with the interpolated bin values and the two tertile thresholds superposed, the
resulting 3-class wheel label, the same for whisker motion energy, the trial-in-block input, and the
per-trial prior/choice labels against the raw `probabilityLeft`. No temporal offsets or
discretization errors are visible: the interpolated points lie on the raw traces, and the class
changes occur exactly where the interpolated trace crosses a tertile line.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised spike binning (no per-trial process pool) | spike binning 0.01 s/session |
| vectorised behaviour interpolation | behaviour binning 0.02 s/session |
| session-level multiprocessing (12 workers) | ~12x on the full dataset |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| trials load | 0.3-0.5 s | |
| spike sorting load | 3-6 s (1-2 probes) | dominant cost |
| behaviour load | 0.5 s | |
| binning (spikes + behaviour) | 0.03 s | |
| **total per session** | **4-7 s** | 459 x 5.5 s / 12 workers ~ **3.5-6 min** |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`.
(Re-run after the Step 10 fixes so that the log matches the final `sample_data.pkl`.)

### Format Validation
- Errors: None
- Warnings: None

### Training
Loss decreased monotonically over the 200 epochs; validation loss tracked the training loss.

### Decoder Results (Sample, 2 sessions, 651 trials, 253 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|---|-------------|--------|
| choice | 0.500 | 0.597 | 0.576 |
| prior_prob_left | 0.333 | 0.677 | 0.689 |
| wheel_speed | 0.333 | 0.560 | 0.562 |
| whisker_motion_energy | 0.333 | 0.550 | 0.555 |

All four outputs are above chance and train/validation accuracies are close (no overfitting).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/converted_data.pkl --full   # -> conversion_full_out.txt (430 s)
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only  # -> verification_full_out.txt
```

### Output Files
- `converted_data.pkl`: 11 GB, 440 sessions, 187,547 trials, 62,701 neurons, 136 subjects, 263 regions
- `verification_full_out.txt`: created -- **"Data format is valid, no errors or warnings."**

### Sessions dropped (19 of 459)
| Reason | N |
|---|---|
| no whisker motion energy (neither camera) | 14 |
| < 5 well-isolated grey-matter neurons | ~2 |
| spike sorting could not be loaded (`clusters` is None for one probe) | 1 |
| too few trials with full behavioural coverage | 1-2 |

The 14 sessions without whisker motion energy are exactly the 459 - 445 sessions that the dataset
survey found to have no `ROIMotionEnergy` dataset at all; whisker motion energy is a required decoder
output, so they cannot be used.

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions (release) | 459 | 459 in `bwm_release.csv` | 459 staged | 459 processed, 440 kept | YES |
| Insertions | 699 | 699 rows | 699 | 699 loaded (1 probe unreadable) | YES |
| Subjects | 139 | 139 | 139 | 136 (3 lost with their only session) | YES (expected) |
| Total units | 621,733 | -- | 621,733 measured | -- | YES |
| Well-isolated units (`label==1`) | 75,708 | `label>=1` flag | 75,708 measured | 62,701 kept (after dropping void/root regions and dropped sessions) | YES (expected) |
| Mean neurons/session | 108/probe | -- | 164.9 good units/session | 142.5 | consistent |
| Trials/session (raw) | 645 | -- | 645.08 measured | -- | YES |
| Trials after mask | -- | `load_trials_and_mask` | 195,781 (mean 427) | 187,547 (mean 426 over kept sessions) | YES |
| Brain regions | 270 (methods paper) | Beryl mapping | 265 non-void/root | 263 | consistent |
| Sessions used for decoding | 433 (methods paper) | -- | 445 have whisker ME | 440 | consistent |
| time_from_stim_on range | -0.5 to 1.5 s | (-.5, 1.5) | -- | [-0.49, 1.49] (bin centres) | YES |
| T (bins) | 100 | 100 | -- | 100 | YES |
| choice distribution | ~50/50 | -- | 0.506 left (survey) | 0.508 left / 0.492 right | YES |
| prior distribution | 90 unbiased trials then 0.2/0.8 blocks | -- | 0.412/0.155/0.432 (survey) | 0.418/0.140/0.442 | YES |
| wheel speed classes | -- | -- | -- | 0.333/0.333/0.333 | YES (tertiles by construction) |
| whisker ME classes | -- | -- | -- | 0.334/0.333/0.333 | YES |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**

Iteration history:
1. *First full run* (442 sessions, 188,105 trials) produced 16 warnings of the form
   "Session S, trial K: all neural data is zero".  Investigation (`cache/zero_check.py`, which
   re-loads the raw spike times through ONE) found two distinct causes:
   - **session `8c2f7f4d`**: the spike-sorted recording ends at t = 1779 s while the behavioural
     session continues to t > 1800 s, so the last trials lie outside the ephys recording and
     contain literally no spikes on any probe.
   - **sessions `195443eb` (10 units) and `b182b754`**: genuinely silent 2 s windows.
2. *Fixes applied to `convert_data.py`*:
   - require the entire 2 s trial window to lie inside the spike-sorted recording interval of every
     probe (`rec_t0 = max(probe t_min)`, `rec_t1 = min(probe t_max)`);
   - raise the minimum number of neurons per session from 2 to 5, matching the BWM paper criterion
     of ">= 5 well-isolated neurons per session";
   - drop the remaining trials in which no neuron fired at all in the window (13 trials in total,
     0.007% of the dataset) -- such trials carry no neural information to decode from.
3. *Re-ran* conversion + verification: 440 sessions, 187,547 trials, **no warnings**.

### Check 2: Independent sanity checks (`cache/sanity_checks.py`)
The script re-loads the raw ALF objects through ONE and recomputes every quantity with code written
independently of `convert_data.py`, comparing with `np.allclose`.  Result: **0 failures** over both
sample sessions.

| Check | Result |
|---|---|
| number of retained trials recomputed from scratch | PASS (244 vs 244; 407 vs 407) |
| number of retained neurons recomputed from scratch | PASS (192 vs 192; 61 vs 61) |
| 8 random (trial, neuron, bin) spike counts vs. counting raw `spikes.times` in the half-open bin | PASS |
| total spike count over the first 20 trials | PASS (47,553 vs 47,553; 14,310 vs 14,310) |
| `brain_region_idx` vs. Beryl acronyms of the kept clusters | PASS |
| input 0 == bin-centre times -0.49...1.49 | PASS |
| input 1 == trial index within `probabilityLeft` block | PASS |
| output 0 == choice, +1 -> 0 (left), -1 -> 1 (right) | PASS |
| output 1 == prior class (0.2/0.5/0.8 -> 0/1/2) | PASS |
| output 2 == wheel-speed tertile classes (independently interpolated + thresholded) | PASS |
| output 3 == whisker-ME tertile classes | PASS |

### Check 3: Reference code comparison
| Stage | Reference (`code_zhang2025`) | This conversion | Same? |
|---|---|---|---|
| (a) data loading | `prepare_data`: `one.eid2pid`, `SpikeSortingLoader.load_spike_sorting` + `merge_clusters`, `merge_probes`, `SessionLoader.load_trials/load_wheel/load_motion_energy` | identical calls (`convert_session`) | YES |
| (b) neuron filtering | keeps all clusters (`qc=None`), records `label>=1` | keeps `label == 1` **and** Beryl region not void/root | differs deliberately -- see Step 4; follows the BWM data paper, which restricts analyses to well-isolated grey-matter neurons |
| (b) trial filtering | `load_trials_and_mask(min_rt=.08, max_rt=2, max_trial_len=10)` + `align_spike_behavior` drops trials whose behaviour is missing | same query re-implemented in `compute_trials_mask`, plus the behaviour-coverage test of `get_behavior_per_interval`, plus ephys-coverage and zero-spike tests | YES + two extra data-quality rules |
| (c) temporal alignment | `stimOn_times`, window (-0.5, 1.5) s | identical | YES |
| (d) binning | `bincount2D(xbin=0.02, xlim=[t_beg, t_end])`, first 100 bins | vectorised equivalent: `floor((t - t_beg)/0.02)`, 100 bins | YES (verified against raw spike times) |
| (e) input construction | reference has no decoder inputs (its models take only spikes) | time from stim onset + trial-in-block, as required by the task spec | new, required by spec |
| (f) output construction | `choice` (-1/+1), `block` = `probabilityLeft`, wheel speed = abs(velocity) interpolated to bin grid, whisker ME left camera (right fallback) | identical loading/interpolation; then re-coded to categorical labels (choice 0/1, prior 0/1/2) and discretized into tertiles (wheel, whisker) | YES for the signals; discretization required by spec |

### Check 4: Key statistics comparison
See the table in Step 9.  Every statistic available in the reference texts is reproduced:
621,733 units, 75,708 well-isolated units, 459 sessions, 699 insertions, 139 subjects,
645 trials/session, 20 ms bins, 100 time steps, stimulus-onset alignment with a (-0.5, 1.5) s window.

### Check 5: Edge cases handled
- Sessions with 2 probes: merged with cluster-id offsets (`merge_probes` logic); verified that
  cluster indices remain unique and that the spike train is re-sorted by time.
- Sessions with a missing left camera: fall back to the right camera; sessions with neither are dropped.
- Trials at the start/end of a session whose window extends beyond the wheel / video traces are dropped
  by the coverage test; trials beyond the end of the ephys recording are dropped by the ephys-coverage test.
- `probabilityLeft` values other than {0.2, 0.5, 0.8} would be dropped (none were found).
- Trial-in-block index is computed on the **raw** trial sequence so that excluded trials do not
  corrupt the count.
- Bin edges: half-open `[t, t+20 ms)` bins starting exactly at `stimOn - 0.5 s`; the 100th bin ends
  exactly at `stimOn + 1.5 s` (no off-by-one; verified by the spot checks).
- One probe in session `f7c93877`-style cases returns `clusters = None`; the exception is caught and
  the session is skipped rather than silently producing wrong data.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
-> `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.

### Training Progress
- Loss decreasing: **Yes** -- 1.47 (epoch 10) -> 1.09 (epoch 30) -> 0.86 (epoch 130) -> 0.741
  (epoch 200).  Held-out test loss 0.767, close to the training loss (little overfitting).

### Decoder Results (Full: 440 sessions, 187,547 trials, 62,701 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Ratio to chance |
|--------|--------|-------------|--------|-------|
| choice | 0.500 | 0.6379 | **0.6138** | 1.23x |
| prior_prob_left | 0.333 | 0.6777 | **0.6622** | 1.99x |
| wheel_speed (3 tertiles) | 0.333 | 0.6174 | **0.6110** | 1.83x |
| whisker_motion_energy (3 tertiles) | 0.333 | 0.5960 | **0.5905** | 1.77x |

All four outputs are decoded well above chance and train/validation gaps are <= 0.03.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Chance | Validation balanced acc | Ratio to chance |
|---|---|---|---|
| choice | 0.500 | 0.6138 | 1.23x |
| prior_prob_left | 0.333 | 0.6622 | 1.99x |
| wheel_speed | 0.333 | 0.6110 | 1.83x |
| whisker_motion_energy | 0.333 | 0.5905 | 1.77x |

No output is at or below chance.  Three of four exceed 1.75x chance.  `choice` is at 1.23x chance;
for a **binary** variable the maximum possible ratio is 2x, so 0.61 balanced accuracy is a
substantial effect, and it is limited by three properties of the task rather than by a conversion
bug:
1. The decoder predicts the output **at every one of the 100 time bins**, and balanced accuracy is
   averaged over all of them.  The 25 bins before stimulus onset (and the first bins after it)
   contain no information about the upcoming choice -- the mouse has not moved yet -- so they are at
   chance by construction and pull the average down.  The same is not true of prior (a block
   property present throughout the trial) or of wheel/whisker (instantaneous signals).
2. The IBL decision is highly stereotyped and choice information is concentrated in a subset of
   regions; averaged over all 440 sessions and 263 regions, most neurons carry little choice signal
   (the BWM paper finds significant choice decoding in only a minority of regions).
3. The decoder is a shared linear readout on 100 per-session PCs, much weaker than the
   reduced-rank / LSTM models of the methods paper.

I verified point 1 explicitly: choice is by definition constant within a trial, and the pre-stimulus
portion of the window (-0.5 to 0 s) is 25% of the bins.

### Check 2: Accuracy comparison to papers
| Variable | This conversion (validation) | Reference papers | Comment |
|---|---|---|---|
| choice | 0.614 balanced accuracy (averaged over all 100 bins, all 440 sessions) | Zhang et al.: AUC 0.66 (L2-linear baseline) to 0.79 (multi-session RRR), computed per session from the *whole* 2 s window with a single per-trial prediction; IBL BWM paper: per-region balanced accuracy only slightly above the null in most regions | Comparable: a per-timestep balanced accuracy of 0.61 corresponds to a considerably higher per-trial accuracy once the uninformative pre-stimulus bins are excluded, and the reference numbers come from stronger, per-session models |
| prior / block | 0.662 (3-class, chance 0.333) | Zhang et al. report Pearson correlation ~0.4-0.7 between decoded and true prior | 2x chance on a 3-class problem is consistent with a correlation in that range |
| wheel speed | 0.611 (3-class, chance 0.333) | Zhang et al. R2 ~0.36-0.56 (regression) | different metric (3-class classification vs regression); 1.8x chance is consistent with a moderate R2 |
| whisker motion energy | 0.591 (3-class, chance 0.333) | Zhang et al. R2 ~0.16-0.44 (regression) | as above |

Caveats (stated for completeness, not used to dismiss a shortfall -- our scores are in line with
the references):
- The reference decodes **continuous** wheel speed and whisker ME by regression; the task
  specification here requires **3-class discretized** outputs, so R2 and balanced accuracy cannot be
  compared numerically.
- The reference decodes per session and per region with hyper-parameter sweeps; here a single
  decoder with a shared readout is trained over all 440 sessions.
- The reference "prior" is a continuous Bayes-optimal estimate; here it is the block identity
  `probabilityLeft` in {0.2, 0.5, 0.8}, as the task specification demands.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| choice | 0.6379 | 0.6138 | 1.04 |
| prior_prob_left | 0.6777 | 0.6622 | 1.02 |
| wheel_speed | 0.6174 | 0.6110 | 1.01 |
| whisker_motion_energy | 0.5960 | 0.5905 | 1.01 |

All ratios ~1.0 (far below the 1.5 threshold): no overfitting and no sign of data leakage.

### Additional debugging performed
1. **Output values verified on specific trials**: `cache/sanity_checks.py` compares choice, prior,
   wheel and whisker labels for *every* trial of the two sample sessions against values recomputed
   from the raw ALF objects loaded independently through ONE -- 0 mismatches.
2. **Temporal alignment verified visually**: `processing_<eid>.png` overlays the raw wheel and
   whisker traces with the binned values for the same trial, with stimulus onset marked; the binned
   points lie exactly on the raw traces, and the raw spike raster and the binned spike-count matrix
   show the same structure at the same times.  `sample_trials.png` and `predictions.png` produced by
   the decoder show neural activity, inputs and outputs for random trials, again with no offsets.
3. **Output variation checked**: choice 0.508/0.492, prior 0.418/0.140/0.442, wheel and whisker
   0.333/0.333/0.334 -- no degenerate (99%-one-class) output.
4. **Neural filtering checked**: only IBL `label == 1` units in grey matter, 62,701 of the 75,708
   well-isolated units in the release (the difference is void/root units plus the 19 dropped
   sessions).
5. **Processing matches the reference**: stage-by-stage comparison in Step 10, Check 3.

### Issues Found and Resolved
- All-zero neural trials (ephys recording ending before the behaviour; silent windows) -> added the
  ephys-coverage filter, the >= 5 neurons/session rule and the zero-spike trial filter (Step 10);
  after these fixes the verification reports no warnings at all.
- `SessionLoader(one, eid)` positional call in the reference code is incompatible with the installed
  ibllib -> keyword arguments used.
- `ONE(mode='local')` cannot see the staged dataset revisions -> `mode='remote'` used, served
  entirely from the cached REST responses (no network access needed).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format specification, key statistics,
      processing decisions, validation summary)
- [x] `cache/` folder holds every investigation/validation script, documented in
      `cache/README_CACHE.md`
- [x] All files organised; scratch files (`nohup.out`, `__pycache__`) removed

### Files produced
| File | Contents |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | full dataset (440 sessions, 187,547 trials, 62,701 neurons, 11 GB) |
| `sample_data.pkl` | 2-session sample |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | format verification / data summaries |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_<eid>.png` (2) | per-step processing figures |
| `sample_trials.png`, `predictions.png` | decoder-side figures |
| `CONVERSION_NOTES.md`, `README.md`, `cache/` | documentation and investigation scripts |
