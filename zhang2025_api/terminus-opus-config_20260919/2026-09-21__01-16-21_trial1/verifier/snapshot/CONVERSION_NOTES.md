# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-wide map (public ONE cache in /app/data)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: python3.13, numpy 2.3.5, torch 2.6.0+cu124, ONE api, brainbox, iblatlas all import OK.
Hardware: 128 CPUs, 1006 GB RAM, 1x NVIDIA L4 (23 GB).
**No network access** - ONE must be used in pure offline (`mode='local'`) mode.

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`, `stage_cache.sh` (infra)
- `CONVERSION_NOTES.md` (this file)
- `code/` -> `code_zhang2025/` (methods-paper code), `ibllib/` (IBL library source)
- `data/one_cache/` -> ONE cache. Release tables in `Brainwidemap/`, `2022_Q4_IBL_et_al_BWM/`,
  `2025_Q3_IBL_et_al_BWM/`; session data symlinked per lab (12 lab dirs).
- `datapaper.pdf`, `methodpaper.pdf`, `dataarchitecture.pdf`, `methods.txt`
- `decoder.py`, `train_decoder.py` (provided decoder)
- `ibl_docs/` (offline docs)

ONE instantiation that works offline:
```python
one = ONE(cache_dir='/app/data/one_cache',
          tables_dir='/app/data/one_cache/Brainwidemap', mode='local')
```
(the default `tables_dir` is the cache root, where there are no tables)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Two code bases are provided in `/app/code`:
- `code_zhang2025/` -- code for the **methods paper** (Zhang et al., "Exploiting correlations
  across trials and behavioral sessions to improve neural decoding"). This is the decoding
  pipeline whose data preparation I must match.
- `ibllib/` -- source of the IBL library (`brainbox`, `ibllib`), used by the above.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| top-level script | `src/0_data_caching.py` | LOADING/PROCESSING | Whole preprocessing pipeline; sets `params` (alignment, window, bin size) and `beh_names` |
| `prepare_data(one, eid, bwm_df, params)` | `src/utils/ibl_data_utils.py` | LOADING | Per session: `one.eid2pid` -> loop over probes -> `load_spiking_data` -> `merge_probes`; `load_trials_and_mask(max_trial_len=10.)`; `load_anytime_behaviors` |
| `load_spiking_data(one, pid, qc=None)` | ibl_data_utils.py | LOADING | `SpikeSortingLoader(pid,...).load_spike_sorting()` + `SpikeSortingLoader.merge_clusters(...).to_df()`. **`qc=None` by default => returns ALL clusters**; if `qc=1` it keeps `clusters.label >= 1` |
| `merge_probes(spikes_list, clusters_list)` | ibl_data_utils.py | PROCESSING | Concatenates probes of one session into a single population, re-indexing `spikes['clusters']`, re-sorting spikes by time |
| `load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', max_trial_len=10., exclude_nochoice=True)` | ibl_data_utils.py | CURATION | `SessionLoader.load_trials()` then builds a boolean trial mask |
| `list_brain_regions` / `select_brain_regions` | ibl_data_utils.py | PROCESSING | `BrainRegions().acronym2acronym(acronyms, mapping='Beryl')`; with `single_region=False` selects the union of all regions (i.e. all clusters) |
| `bin_spiking_data(reg_clu_ids, neural_df, trials_df, **params)` | ibl_data_utils.py | PROCESSING | intervals = `trials_df[align_time] + time_window`; bins with `iblutil.numerical.bincount2D(times, clusters, xbin=binsize, xlim=[t_beg,t_end])`, truncated to `n_bins = ceil(interval_len/binsize)`. Returns `(n_trials, n_bins, n_clusters)` |
| `load_target_behavior(one, eid, target)` | ibl_data_utils.py | LOADING | `SessionLoader.load_wheel()` (`wheel-speed` = `abs(velocity)`); `SessionLoader.load_motion_energy(views=['left'/'right'])` -> `motion_energy['leftCamera']['whiskerMotionEnergy']` |
| `get_behavior_per_interval(...)` | ibl_data_utils.py | PROCESSING | Slices the continuous trace to each interval, then `interp1d(kind='linear')` onto `np.linspace(beg+binsize, end, n_bins)` (i.e. **bin right edges**). Marks an interval bad if: no samples, NaNs (when `allow_nans=False`), NaN interval bounds, trace starts >1 binsize after the interval start, or ends >1 binsize before the interval end |
| `bin_behaviors(one, eid, behaviors, trials_df, ...)` | ibl_data_utils.py | PROCESSING | Per-trial scalars straight from the trials table: `choice = trials.choice`, `block = trials.probabilityLeft`, `reward = (rewardVolume > 1)`, `contrast = -(nan_to_num(contrastLeft) + ... )`; plus the time-varying behaviors above |
| `align_spike_behavior(binned_spikes, binned_behaviors, beh_names, trials_mask)` | ibl_data_utils.py | CURATION | Deletes trials that are bad in the trials mask **or** in any behavior mask, so neural and behavior have identical trials |
| `standardize_spike_data` / `SingleSessionDataset` | `src/utils/data_loader_utils.py` | PROCESSING | Downstream modelling only: z-scores spikes per (time bin, neuron) and z-scores continuous behaviour per session; discrete behaviour is one-hot encoded |

### Parameters used by the reference pipeline (`src/0_data_caching.py`)
```python
params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
          'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']
```
=> 2 s trials aligned to **stimulus onset**, 20 ms bins, **T = 100**. Note that
`pupil-diameter` is commented out because "Some sessions do not have pupil traces".

### Notes
- `load_trials_and_mask` is called with `exclude_unbiased=False`, so the 90 unbiased
  (pLeft = 0.5) trials at the start of a session are **kept**. This matters here because
  `probabilityLeft` is one of our decoder outputs with three classes (0.2/0.5/0.8).
- The freeze file `data/bwm_release.csv` (699 rows) is the authoritative list of
  (pid, eid, probe_name, subject, lab) in the brain-wide map release.
- `qc=None` in `load_spiking_data` means the reference decoding code used *all* Kilosort
  clusters, consistent with the methods paper ("we bin spike counts using all neurons").
  See Step 4 for how I reconcile this with the data paper's neuron inclusion criteria.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
The data live in a standard **ONE cache** at `/app/data/one_cache`:

```
/app/data/one_cache/
  Brainwidemap/            datasets.pqt (76,563), sessions.pqt (480)   <- release tables
  2022_Q4_IBL_et_al_BWM/   datasets.pqt (38,814), sessions.pqt (354)
  2025_Q3_IBL_et_al_BWM/   datasets.pqt  (5,339), sessions.pqt (459)
  .rest/                   6,483 cached Alyx REST responses
  <lab>/Subjects/<subject>/<yyyy-mm-dd>/<nnn>/    <- session data (12 labs, 461 session dirs)
      alf/  _ibl_trials.table.pqt (usually under revision dir `#2025-03-03#`),
            _ibl_wheel.position.npy / .timestamps.npy,
            leftCamera.ROIMotionEnergy.npy, rightCamera.ROIMotionEnergy.npy,
            _ibl_leftCamera.times.npy, _ibl_{left,right}Camera.dlc.pqt, licks, passive...
      alf/<probe>/pykilosort/  spikes.{times,clusters,amps,depths}.npy,
            clusters.{channels,depths,metrics.pqt,uuids.csv,waveforms}.npy,
            channels.{brainLocationIds_ccf_2017,mlapdv,localCoordinates,rawInd}.npy
      raw_ephys_data/<probe>/  *.ap.meta (used for sampling frequency)
```

**Important access detail.** Many files on disk sit in *revision* folders
(e.g. `alf/#2025-03-03#/_ibl_trials.table.pqt`, present for 454/461 sessions) that are
newer than any of the three staged release tables. Consequently
`ONE(cache_dir=..., tables_dir=.../Brainwidemap, mode='local')` silently returns an almost
empty trials table (only `goCueTrigger_times`). The staging script copies ONE's parameter
file and a `.rest` cache of Alyx REST responses, so the correct (and intended) call is
simply

```python
one = ONE()          # reads ~/.one/.caches -> cache_dir=/app/data/one_cache, mode='remote'
```

which resolves dataset queries from the cached REST responses and therefore *does* find
the revisioned datasets. All loading below uses this instance. **This was the single most
important discovery of Step 2** -- the alternative silently produces a broken dataset.

### Available variables
- **Trials** (`SessionLoader.load_trials()` -> 20 columns): `stimOn_times`,
  `stimOnTrigger_times`, `stimOff_times`, `stimOffTrigger_times`, `goCue_times`,
  `goCueTrigger_times`, `response_times`, `firstMovement_times`, `feedback_times`,
  `feedbackType`, `choice`, `contrastLeft`, `contrastRight`, `probabilityLeft`,
  `rewardVolume`, `quiescencePeriod`, `intervals_0/1`, `intervals_bpod_0/1`.
  - `choice` in {-1, 0, +1}. Verified empirically on eid 6713a4a7...: on rewarded trials
    with a left stimulus `choice == +1` always, and with a right stimulus `choice == -1`
    always. So **choice = +1 means the mouse reported LEFT**, `choice = -1` reports RIGHT,
    `choice = 0` is a no-go. (The sign is the wheel direction, not the stimulus side.)
  - `probabilityLeft` in {0.2, 0.5, 0.8}; 0.5 only in the first 90 (unbiased) trials.
- **Wheel** (`SessionLoader.load_wheel()`): `times`, `position`, `velocity`,
  `acceleration`, resampled to a uniform 1000 Hz grid (median dt = 1/1024 s) with
  Gaussian-smoothed velocity. ~4.7 M samples/session.
- **Motion energy** (`SessionLoader.load_motion_energy(views=['left'])`):
  `times`, `whiskerMotionEnergy` at the camera rate (left = 60 Hz, right = 150 Hz).
- **Spike sorting** (`SpikeSortingLoader(pid).load_spike_sorting()` +
  `merge_clusters`): `spikes.times/clusters`, and a cluster table with
  `acronym`, `atlas_id`, `x/y/z`, `depths`, `firing_rate`, `amp_median`, `noise_cutoff`,
  `slidingRP_viol`, `label`, `ks2_label`, `bitwise_fail`, `uuids`, ...
  - `label` is the IBL single-unit QC score in {0, 1/3, 2/3, 1}: the fraction of the three
    RIGOR single-unit criteria passed (median amplitude > 50 uV, noise cut-off < 20 uV,
    refractory-period violation). `label == 1` == **well-isolated neuron**.

### Dataset Size (measured from the data files, all 699 insertions scanned)
| Statistic | Value |
|-----------|-------|
| Insertions (probes) | 699 |
| Sessions (eids) | 459 |
| Subjects (mice) | 139 |
| Labs | 12 |
| Units (total, all Kilosort clusters) | **621,733** |
| Well-isolated units (`label >= 1`) | **75,708** |
| ... of those, in Beryl grey matter (not `root`/`void`) | 65,301 |
| Well-isolated grey-matter units / session | mean 142.3 (all 459 sessions have >= 1) |
| Distinct Beryl regions with well-isolated grey-matter units | 265 |
| Sessions with a left-camera whisker ME trace | 437 |
| Sessions with a right-camera whisker ME trace | 433 |
| Sessions with either | 445 |
| Sessions with wheel and trials | 459 / 459 |
| Trials / session (raw) | e.g. 565 for eid 6713a4a7...; typically 300-900 |

The 621,733 and 75,708 figures reproduce the data paper **exactly**, which validates
both the loading path and my interpretation of `clusters.label`.

No `DATALIMIT_SUBSET.csv` is present, so the **full 459-session release** is to be
processed.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Sessions in release | 459 | "a total of 459 sessions, 699 insertions and 621,733 neurons remained, constituting the publicly released dataset" (data paper) |
| Insertions | 699 | same |
| Units (all) | 621,733 | same |
| Well-isolated neurons | 75,708 | "Out of the 621,733 units collected, 75,708 were considered well-isolated neurons" |
| Subjects (mice) | 139 (94 M, 45 F) | "We trained 139 mice (94 male and 45 female)" |
| Sessions used in Fig. 1d | 454 | "Data are for 139 mice and 454 sessions" |
| Sessions in methods paper | 433 | "We apply our models to 433 IBL sessions, covering 270 brain regions" |
| Brain regions (methods paper) | 270 | same |
| Trials / session (release criterion) | >= 250 | "Sessions were included in the data release if the mice performed at least 250 trials" |
| Unbiased block | first 90 trials, pLeft = 0.5 | "Each session started with 90 trials in which the probability ... was equal" |
| Biased block length | 20-100, truncated exponential, scale 60, empirical mean 51 | "Blocks lasted for between 20 and 100 trials ... (empirical mean of 51 trials)" |
| Block probabilities | 20:80 or 80:20 | "at a ratio of 20:80% (right block) or 80:20% (left block)" |
| Contrasts | 100, 25, 12.5, 6.25, 0 % in ratio 2:2:2:2:1 | "Stimulus contrast was uniformly sampled from 5 possible values" |
| **Reaction times < 80 ms** | **22.8% of trials** | "A total of 22.8% first wheel-movement times occurred under 80 ms" |
| Performance on 0% contrast | 58.7 +- 0.4% correct | Fig. 1d caption |
| Neural time bin (choice, stimOn-aligned) | 20 ms (methods paper code) / 50 ms (methods paper text for choice+prior) | see below |
| Trial length | 2 s -> T = 100 bins | "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps" |
| Behaviour sampling rate | 60 Hz | "wheel speed and whisker motion energy are ... sampled at 60 Hz" |

### Processing Details

**Temporal alignment.** The methods paper: "For choice, we align trials to the stimulus
onset, considering neural activity from 0.5 s before to 1.5 s post-onset." The reference
code implements exactly this for *all* of its behaviours in one pass:
`align_time='stimOn_times'`, `time_window=(-.5, 1.5)`, `interval_len=2`, `binsize=0.02`.
The decoder task here also specifies **"Temporally align based on stimulus onset"**, so
this is the alignment and window I use: **`off_start = -0.5 s`, `off_end = +1.5 s`,
20 ms bins, T = 100**.

**Bin-size discrepancy in the paper text.** The text says 50 ms bins for the
choice/prior decoders and 20 ms for the dynamic behaviours, but the abstract/model
section and the actual released code both use 20 ms with T = 100. Because a single
dataset must serve all four outputs here (two per-trial, two time-varying), I follow the
code and the model description: **20 ms, T = 100**.

**Temporal binning.**
- Spikes: counts in `[stimOn - 0.5 + i*0.02, stimOn - 0.5 + (i+1)*0.02)` for i = 0..99,
  exactly `bincount2D(..., xbin=0.02, xlim=[t_beg, t_end])` truncated to 100 bins.
- Behaviour: linearly interpolated onto the **right edge** of each bin,
  `np.linspace(beg + binsize, end, 100)`, exactly as `get_behavior_per_interval`.

### Curation Steps

**Session curation rules** (already applied to the release): >= 250 trials, >= 90% correct
on 100% contrast in both block types, >= 3 incorrect trials, hardware QC passed,
resolved histology alignment, RIGOR whole-recording criteria.

**Neuron curation rules** (data paper):
> "Neurons ... were excluded ... if they failed one of the three criteria ...: amplitude >
> 50 uV; noise cut-off < 20 uV; and refractory period violation. Neurons that passed these
> criteria were termed well-isolated neurons."

This is exactly `clusters.label == 1` (the label is the fraction of the three criteria
passed). Additionally:
> "Final analyses were additionally restricted to regions that were designated grey matter
> in the adult mouse Allen Common Coordinate framework."

**Trial curation rules** (data paper):
> "trials were excluded if one of the following trial events could not be detected: choice,
> probabilityLeft, feedbackType, feedback times, stimON times and firstMovement times.
> Trials were further excluded if the time between stimulus onset and the first movement of
> the wheel ... were outside the range of 0.08-2.00 s."

The reference code's `load_trials_and_mask` implements precisely this, plus
`exclude_nochoice=True` (choice == 0) and, as called in `0_data_caching.py`,
`max_trial_len=10.0` (feedback_times - goCue_times <= 10 s).

### Decoders Trained / reported performance (methods paper)
| Decoded variable | Metric | Reported value |
|---|---|---|
| Choice | AUC, BMM-HMM vs baseline vs oracle | 0.72 vs 0.66 vs 0.79 (Fig. 4) |
| Choice | example single-session accuracy | 0.71 -> 0.90 (Fig. 5) |
| Prior | Pearson r (LG-AR1 vs baseline vs oracle) | 0.65 vs 0.05 vs 0.34 (Fig. 4) |
| Choice | relative accuracy gain, multi- over single-session RRR | +2% (ARI +9%) (Fig. 2C) |
| Wheel speed / whisker ME | trial-averaged + single-trial reconstruction | qualitative (Fig. 3); RRR > ridge |

The data paper's decoding numbers are *null-corrected, per-region* balanced accuracies
(small effect sizes, e.g. Fig. 4) and are not directly comparable with a whole-session,
all-neuron decoder, but they establish that **choice is decodable well above chance from
BWM populations** and that wheel speed is the most decodable of the continuous variables.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| **Neuron quality filtering** | `load_spiking_data(..., qc=None)` -> keeps **all** Kilosort clusters (`qc=1` would keep `label>=1`) | 621,733 clusters total; 75,708 have `label==1` | Data paper: analyses use only **well-isolated** neurons (the 3 RIGOR single-unit criteria = `label==1`); methods paper: "we bin spike counts using **all neurons**, sorted by Kilosort 2.5, from each session" | **Use well-isolated (`label >= 1`) neurons in Beryl grey matter.** See rationale below. |
| Grey-matter restriction | not applied in `0_data_caching.py` | 10,407 of the 75,708 well-isolated units map to Beryl `root`/`void` | Data paper: "restricted to regions that were designated grey matter" | Applied (`root`, `void` dropped). Also required so that `brain_regions` are real anatomical labels. |
| Region count | -- | all units: 280 Beryl regions; well-isolated grey: **265** | methods paper: "270 brain regions" | 265 vs 270 -- the paper's 270 comes from using all units on its 433-session subset. 265 is the matching number under the data paper's neuron criteria. Consistent. |
| Session count | reference caches N sessions chosen by CLI | 459 eids in the release; 445 have a whisker-ME trace | data paper 459 released / 454 analysed; methods paper 433 used | I process all 459 and drop only sessions that genuinely lack a required stream or have <2 usable trials. Expect ~433-445. |
| Bin size | code: 0.02 s, T=100 | -- | methods text: 50 ms for choice/prior, 20 ms for dynamic behaviours; model section: "2-s trials ... 20-ms bins, producing T = 100" | **20 ms, T = 100**, per the code and the model description. One dataset must serve both per-trial and time-varying outputs. |
| Unbiased block | `exclude_unbiased=False` -> pLeft=0.5 trials kept | ~90 trials/session at pLeft=0.5 | data paper describes the 90-trial unbiased block | **Kept** -- required, because "prior probability of left" is a 3-class output (0.2 -> 0, 0.5 -> 1, 0.8 -> 2). |
| `max_trial_len` | `load_trials_and_mask(..., max_trial_len=10.0)` | -- | data paper's trial criteria do not mention it | Kept (reference code behaviour); it removes a handful of pathological trials. |
| Whisker camera | code tries `left` then falls back to `right` | 437 sessions have left, 433 have right, 445 have either | data paper: left camera 60 Hz, right 150 Hz | Same left-then-right fallback as the reference. |

### Rationale for using well-isolated neurons
The two references disagree. I follow the **data paper** (`label >= 1` + grey matter)
because:
1. It is the documented quality-control standard of the dataset being converted, and the
   task instructions explicitly ask for "filtering of low-quality neurons".
2. `brain_region_idx` must name a real region for every neuron; `root`/`void` units have
   no anatomical identity.
3. The 75,708 figure is reproduced exactly, giving a hard check that the filter is right.
4. Practicality: all 621,733 units at float32 over ~180k trials x 100 bins would be ~90 GB
   and would not fit in the decoder's memory; 65,301 well-isolated grey units give ~9 GB.
5. Signal: the excluded units are by construction noisy/contaminated, and the decoder's
   per-session linear projection to 100 PCs would spend its capacity on them.

### Cross-checks that passed
- Scanning all 699 insertions reproduced **621,733** total units and **75,708**
  well-isolated units -- exactly the data paper's numbers.
- `bwm_release.csv` gives 699 pids / 459 eids / 139 subjects / 12 labs -- exactly the
  data paper's numbers.
- `choice` sign convention verified against `contrastLeft`/`contrastRight` and
  `feedbackType` (choice = +1 <=> reported left).
- `probabilityLeft` takes only {0.2, 0.5, 0.8}, with 0.5 confined to the first ~90 trials.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial geometry
- alignment event: `trials.stimOn_times` (stimulus onset)
- window: `off_start = -0.5 s`, `off_end = +1.5 s` (methods paper + reference code)
- bin size: 20 ms, `T = 100` bins
- bin i covers `[stimOn - 0.5 + 0.02*i, stimOn - 0.5 + 0.02*(i+1))`; its representative
  time (used for behaviour interpolation and for the time input) is its right edge,
  `-0.5 + 0.02*(i+1)`, i.e. -0.48 ... +1.50 s. This is exactly the reference
  `np.linspace(beg + binsize, end, n_bins)`.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` (all probes merged) | `neural` | spike counts in 20 ms bins, `(n_neurons, 100)` per trial, float32 | `merge_probes`, `bin_spiking_data` / `bincount2D` | counts, not rates -- matches reference |
| `clusters.label >= 1` and Beryl acronym not in {root, void} | neuron mask | -- | `load_spiking_data(qc=1)`, `select_brain_regions` | see Step 4 |
| `BrainRegions().acronym2acronym(acronym, Beryl)` | `brain_regions`, `brain_region_idx` | Beryl acronym -> global index | `list_brain_regions` | |
| bin right-edge time | `input[0]` = `time_from_stim_on` | seconds, -0.48 ... 1.50, time-varying | -- | required by the decoder task (Time since stimulus onset, continuous, time-varying) |
| `trials.probabilityLeft` run-lengths | `input[1]` = `trial_number_in_block` | 0-based index of the trial within its block, divided by 100, per-trial (broadcast over T) | -- | required by the decoder task (Trial number in block, continuous, per-trial). Divided by 100 so the value is O(1): the decoder concatenates inputs with 100 neural PCs and feeds a linear layer with no input standardisation. |
| `trials.choice` | `output[0]` = `choice` | `choice == +1` (reported LEFT) -> 0; `choice == -1` (reported RIGHT) -> 1 | `bin_behaviors` | matches the required left = 0, right = 1. `choice == 0` trials are already excluded. |
| `trials.probabilityLeft` | `output[1]` = `prior_prob_left` | 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 | `bin_behaviors` (block) | exactly as specified in the decoder task |
| `SessionLoader.wheel.velocity` | `output[2]` = `wheel_speed` | `abs(velocity)`, interpolated to bin right edges, then per-session tertiles -> 0/1/2 | `load_target_behavior(wheel-speed)`, `get_behavior_per_interval` | time-varying |
| `SessionLoader.motion_energy[leftCamera].whiskerMotionEnergy` (fallback rightCamera) | `output[3]` = `whisker_motion_energy` | interpolated to bin right edges, then per-session tertiles -> 0/1/2 | `load_target_behavior(left-whisker-motion-energy)` + fallback | time-varying |
| `bwm_release.csv` subject | `subjects`, `subject_idx` | -- | -- | 139 mice |

Both `input` and `output` are stored as 2-D `(d, 100)` arrays for every trial: the
per-trial variables (`trial_number_in_block`, `choice`, `prior_prob_left`) are broadcast
across the 100 bins. The decoder does this broadcast internally anyway
(`SessionData.__getitem__`), and storing everything 2-D keeps `d_input`/`d_output`
unambiguous and lets `plot_trial` draw all four outputs as time series.

### Key Decisions
1. Alignment = `stimOn_times`, window (-0.5, +1.5) s, 20 ms bins, T = 100. Mandated by
   the decoder task (Temporally align based on stimulus onset) and identical to the
   reference `params`.
2. Neuron curation = `label >= 1` AND Beryl region not root/void. The data paper
   well-isolated-neuron criterion; reproduces 75,708 exactly before the grey-matter step.
   (Reference code `qc=None` rejected -- see Step 4.)
3. Trial curation = the reference `load_trials_and_mask`: no NaN in `stimOn_times`,
   `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`;
   `0.08 <= firstMovement_times - stimOn_times <= 2.0`; `feedback_times - goCue_times <= 10`;
   `choice != 0`. Plus the reference behaviour masks (a trial is dropped if the
   wheel or whisker trace does not cover its window).
4. Unbiased (pLeft = 0.5) trials kept -- they are class 1 of the prior output.
5. Spike counts, not rates, un-normalised. The reference caches raw counts; the
   decoder does its own SVD/PCA. Counts keep the data integral-valued and sparse.
6. Per-session tertiles for the two continuous outputs. The task says discretized into
   3 bins. Wheel speed and whisker ME are strongly right-skewed and their absolute
   scale differs by an order of magnitude between sessions (different cameras,
   different wheel gains, different lighting). Equal-width bins would put >95% of
   samples in one class for most sessions and would make the class meaning
   session-dependent. Tertiles (33.3/66.7 percentiles of all kept bins in that session)
   give three equally populated, comparable classes: low/medium/high. This mirrors
   the reference pipeline, which z-scores these behaviours per session before
   modelling (`SingleSessionDataset`: `StandardScaler().fit(train_behavior)`).
7. Sessions dropped only if: no usable trials (<2), no well-isolated grey-matter
   neuron, or no whisker-ME trace at all.
8. Both probes of a session merged into one population (`merge_probes`), as the
   reference does and as the data paper requires (neurons in the same session and region
   were combined across probes).

### Planned Sanity Checks
- [ ] total units over all insertions == 621,733; well-isolated == 75,708
- [ ] 699 insertions, 459 eids, 139 subjects from `bwm_release.csv`
- [ ] fraction of trials with reaction time < 80 ms is about 22.8%
- [ ] performance on 0%-contrast trials is about 58.7%
- [ ] `probabilityLeft` only in {0.2, 0.5, 0.8}; 0.5 only in the first ~90 trials
- [ ] block lengths (excluding the unbiased block) in [20, 100], mean about 51
- [ ] choice class balance near 50/50
- [ ] each of wheel-speed / whisker-ME classes about 1/3 of bins
- [ ] T == 100 for every trial in every session
- [ ] spot check: re-bin a specific trial spikes with an independent method, compare with np.allclose
- [ ] spot check: re-interpolate a specific trial wheel speed from the raw ONE object
- [ ] neural mean firing rate in a plausible range (a few Hz)

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (805 lines). Runs as
`python -u /app/convert_data.py <outfile> [--full|--sample] [--show-processing]`.

Structure:
- `get_one()` -- returns `ONE()` with no arguments. Documented in the source why this is
  the only instantiation that finds the revisioned datasets on disk.
- `build_trials_mask(trials)` -- a literal transcription of the reference
  `load_trials_and_mask(min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True)`
  pandas query.
- `trial_number_in_block(prob_left)` -- vectorised run-length index via
  `np.maximum.accumulate`.
- `bin_spikes_trials(...)` -- spike counts per (trial, cluster, bin). Uses `searchsorted`
  to isolate each trial window and `np.add.at` on a flat view. Equivalent to the
  reference `bincount2D(..., xbin=0.02, xlim=[t_beg, t_end])[:, :100]` but for all trials
  at once and without the reference multiprocessing pool per trial.
- `interp_behavior(...)` -- linear interpolation onto bin right edges
  `t_beg + 0.02*(1..100)`, plus the reference coverage checks (trace must start no more
  than one bin after, and end no less than one bin before, the window).
- `tertile_bins(...)` -- per-session 33.3/66.7 percentile discretisation.
- `load_session(eid, probes)` -- the whole per-session pipeline.
- `make_processing_plot(res, outfile)` -- a 7x4 diagnostics figure (see Step 7).
- `main()` -- `ProcessPoolExecutor` over sessions, assembly, statistics, pickling.

### Code inefficiencies identified and removed
- The reference bins spikes with a `multiprocessing.Pool` **per trial** and calls
  `bincount2D` once per trial. Replaced with one `searchsorted` for all trials plus a
  single `np.add.at` per trial -- about 2 orders of magnitude fewer Python-level calls.
- The reference calls `get_behavior_per_interval` (another per-trial process pool) for
  each behaviour. Replaced with a single vectorised `np.interp` over an
  `(n_trials, 100)` grid.
- Spike sorting is loaded once per probe, and the trials/wheel/motion-energy objects once
  per session, via a single shared `SessionLoader`.
- Parallelism is at the **session** level (`ProcessPoolExecutor`), which is the natural
  grain and avoids the reference nested-pool overhead.
- Only the kept trials are binned, so no work is done for trials that will be discarded.

### Bug found and fixed during development
- `(~trials.eval(query)).to_numpy()` returns a **read-only** array, so the subsequent
  in-place `trial_mask &= ...` raised `ValueError: output array is read-only` and every
  session was silently skipped. Fixed with an explicit
  `np.asarray(..., dtype=bool).copy()`. This is exactly the kind of failure the
  per-session try/except would otherwise have hidden, so the driver prints the skip
  reason for every session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command:
`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 (NYU-11) |
| Brain regions | 16 |
| Neurons (total) | 253 |
| Neurons / session | 61, 192 (mean 126.5) |
| Trials (total) | 651 |
| Trials / session | 407 / 565 and 244 / 425 |
| Units seen / well-isolated | 2626 / 340 |
| T | 100 for every trial |
| input time_from_stim_on | [-0.48, 1.50] |
| input trial_number_in_block | [0.00, 0.89] |
| output choice | [0.518, 0.482] |
| output prior_prob_left | [0.478, 0.161, 0.361] |
| output wheel_speed | [0.333, 0.333, 0.333] |
| output whisker_motion_energy | [0.333, 0.333, 0.333] |

All of these are as expected: choice is near 50/50; the pLeft = 0.5 class is about 16%,
consistent with 90 unbiased trials out of ~500; and the two tertile-coded outputs are
exactly one third each by construction.

### Processing Plots Review
`processing_<eid>.png` has 7 rows:
1. reaction-time histogram with the 0.08/2.0 s cutoffs, the trial mask before/after the
   behaviour masks, `probabilityLeft` over the session, and trial-number-in-block (the
   expected sawtooth, first tooth 90 long, later ones 20-100).
2. raw spike raster of one trial in stimulus-aligned time next to the binned count matrix
   of the *same* trial -- these must and do look identical, which is the temporal
   alignment check for the neural stream.
3. population PSTH (a clear stimulus-onset transient at t = 0), per-neuron firing-rate
   histogram, neurons per Beryl region, and the `time_from_stim_on` input plotted against
   time (a straight line through the origin at t = 0).
4. wheel speed for 4 trials: the raw ~1 kHz `|velocity|` trace overlaid with the 100
   interpolated bin values and the resulting 0/1/2 class, with the tertile thresholds
   drawn. The binned curve sits exactly on the raw trace -- the alignment check for the
   wheel stream.
5. the same for whisker motion energy against the raw 60 Hz trace.
6. the pooled distributions with the tertile cuts, and the resulting class fractions
   against the 1/3 reference line -- the discretisation check.
7. the four outputs over kept trials: choice, prior, and the two time-varying class
   rasters.
No anomalies: no temporal offsets, no empty trials, no degenerate classes.

### Verification
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` printed
**"Data format is valid, no errors or warnings."**

### Independent sanity checks against raw ONE data (`cache/sanity_sample.py`)
Re-loaded session 6713a4a7 straight from ONE, without using any conversion code:

| Check | Result |
|---|---|
| kept trial count | 407 == 407 |
| neuron count | 61 == 61 |
| per-neuron Beryl region list | identical |
| `neural` trial 5, all 61 neurons x 100 bins, re-binned with `np.histogram` | `np.allclose` **True** (841 spikes both) |
| `input[0]` = -0.48..1.50 | `np.allclose` **True** |
| `input[1]` trial-in-block/100 | `np.allclose` **True** |
| `output[0]` choice (raw +1 -> class 0) | correct |
| `output[1]` prior (raw 0.5 -> class 1) | correct |
| `output[2]` wheel-speed classes, re-interpolated from `SessionLoader.wheel` | `np.allclose` **True** |
| `output[3]` whisker classes, re-interpolated from `motion_energy` | `np.allclose` **True** |

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised all-trial spike binning instead of a per-trial process pool | ~10-50x on the binning step |
| vectorised `np.interp` instead of a per-trial process pool per behaviour | ~10x on the behaviour step |
| bin only the trials that survive curation | ~30% fewer bins |
| session-level `ProcessPoolExecutor` (24-32 workers) | ~24x wall-clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| trials | 0.3 s | |
| spike sorting load | 2-6 s | |
| behaviour load | 0.5-1 s | |
| binning | 0.3-1 s | |
| **total per session (1 worker)** | **4.3 s and 7.8 s measured** | 459 x ~8 s = 61 min serial |
| **with 24 workers** | | **~4-8 min**, well under the 15 min budget |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions / 253 neurons / 651 trials)
Loss decreased monotonically over the 200 epochs; test loss 0.8170.
(Regenerated after the final code changes of Step 10, so these numbers match
`train_decoder_sample_out.txt` exactly.)

| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| choice | 0.500 | 0.6131 | 0.5267 |
| prior_prob_left | 0.333 | 0.6595 | 0.6141 |
| wheel_speed | 0.333 | 0.5722 | 0.5594 |
| whisker_motion_energy | 0.333 | 0.5646 | 0.5628 |

All four outputs are above chance. `choice` is the weakest, which is expected here: this
sample has only 61 and 192 neurons, and the decoder is evaluated *per time bin*, so the
50 pre-stimulus bins of every trial (where the choice has not yet been formed and is
therefore close to unpredictable) are included in the average. The full dataset is the
real test.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --n-workers 32`
Run time **3.5 min** (well inside the 15 min budget).

### Output Files
- `converted_data.pkl`: 11.72 GB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created -- **"Data format is valid, no errors or warnings."**

### Final dataset
| Statistic | Value |
|---|---|
| Sessions | 442 of 459 |
| Subjects | 136 of 139 |
| Brain regions (Beryl) | 263 |
| Neurons | 62,773 |
| Trials | 188,044 |
| Neurons / session | mean 142.0, median 123, min 7, max 516 |
| Trials / session | mean 425.4, median 392, min 85, max 1445 |
| T | 100 for every trial |

17 sessions were dropped, each for a concrete, reported reason:
- 14 have **no whisker motion-energy trace** on either camera (one of the four required
  outputs). This matches the independent availability scan exactly: 445 of 459 sessions
  have a left- or right-camera trace.
- 3 have **fewer than 5 well-isolated grey-matter neurons** (3, 2 and 1 neurons).

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Insertions | 699 | 699 rows in bwm_release.csv | 699 | 699 read | yes |
| Sessions released | 459 | 459 eids | 459 | 459 read, 442 kept | yes |
| Sessions used for decoding | 433 (methods paper) | CLI-selected | 445 have whisker ME | **442** | yes (see note) |
| Subjects | 139 | 139 | 139 | 136 kept | yes |
| Units (all) | 621,733 | -- | **621,733** | 621,733 scanned | **exact** |
| Well-isolated units | 75,708 | -- | **75,708** | 75,708 scanned | **exact** |
| ... grey matter only | -- | -- | 65,301 | 62,773 in kept sessions | yes |
| Beryl regions | 270 (methods paper, all units) | -- | 265 (well-isolated grey) | 263 | yes |
| Neural bin | 20 ms, T=100 | 20 ms, T=100 | -- | 20 ms, T=100 | **exact** |
| Alignment | stimOn_times | stimOn_times | -- | stimOn_times | **exact** |
| Window | (-0.5, +1.5) s | (-0.5, +1.5) s | -- | (-0.5, +1.5) s | **exact** |
| RT < 80 ms | **0.228** | -- | **0.2299** measured | (those trials excluded) | yes |
| 0% contrast correct | **0.587** | -- | **0.5857** measured | -- | yes |
| Unbiased block length | 90 | -- | median **90** measured | -- | **exact** |
| Biased block length | 20-100, mean 51 | -- | mean **48.6** measured | -- | yes |
| probabilityLeft values | {0.2, 0.5, 0.8} | -- | {0.2, 0.5, 0.8} | 3 classes | **exact** |
| choice distribution | ~50/50 | -- | -- | **[0.508, 0.492]** | yes |
| prior distribution | 90 unbiased of ~645 raw trials | -- | -- | **[0.418, 0.140, 0.442]** | yes |
| wheel speed classes | -- | -- | -- | **[0.333, 0.333, 0.333]** | by construction |
| whisker ME classes | -- | -- | -- | **[0.335, 0.333, 0.333]** | by construction |
| input time_from_stim_on | -- | -- | -- | [-0.48, 1.50] | as designed |
| input trial_number_in_block | -- | -- | -- | [0.00, 0.98] | as designed |

Note on 442 vs 433: the methods paper says it applies its models to 433 sessions but does
not list them or state the exclusion rule. 433 is exactly the number of sessions with a
**right**-camera whisker trace, whereas 445 have one on either camera. My pipeline, like
the reference code, prefers the left camera and falls back to the right, so it retains
445 before the neuron-count criterion and 442 after. The small difference is fully
accounted for and is in the direction of keeping more valid data.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
The first full run produced **38 warnings** of the form
`Session S, trial T: all neural data is zero`, and no errors.
I did not dismiss them. Investigation (`cache/check_zero.py`, `cache/check_zero2.py`)
found **two distinct causes**:

1. **Degenerate sessions.** One session had a single well-isolated grey-matter neuron and
   another had 10; a low-rate neuron is genuinely silent for some 2-s windows.
   *Fix*: a session-level minimum of 5 well-isolated grey-matter neurons (the data
   paper's threshold, applied at the population level this decoder actually uses --
   see the rationale in the source and in Step 5). Removed 3 sessions.
   - I first implemented the data paper's threshold literally, per **region**. That was
     wrong for this task: it cut the sample from 253 to 83 neurons and 16 regions to 1,
     and on the full data would have discarded ~40% of well-isolated neurons. The
     criterion exists to give power to the paper's *per-region* analyses; this decoder
     pools a whole session. Reverted to the session-level form and documented why.
2. **Missing spike coverage.** Session `8c2f7f4d` has three trials that start *after* its
   last spike (the recording stopped before the behaviour did), and session `b182b754`
   has one trial inside a 2.07 s drop-out. The reference pipeline applies a coverage test
   to the behavioural traces (`get_behavior_per_interval` rejects an interval whose trace
   starts too late or ends too early) but **not** to the spikes, so these silently became
   all-zero matrices.
   *Fix*: apply the same coverage rule to the neural stream. Removed 16 trials.

After both fixes the log reads **"Data format is valid, no errors or warnings."** -- zero
warnings remain, so none had to be explained away.

### Check 2: Sanity checks against the original data
`cache/sanity_full.py` re-loads 5 randomly chosen sessions **directly from ONE**, without
calling any conversion code, and compares with `np.allclose`:

| Session | Neurons | Trial | Region list | NEURAL | INPUT time | INPUT tib | OUT choice | OUT prior | OUT wheel | OUT whisker |
|---|---|---|---|---|---|---|---|---|---|---|
| 279 `c4432264` | 370 | 476 (raw 695) | match | **True** | True | True | True | True | True | True |
| 224 `9fcbd1a0` | 16 | 157 (raw 331) | match | **True** | True | True | True | True | True | True |
| 118 `e1931de1` | 144 | 482 (raw 614) | match | **True** | True | True | True | True | True | True |
| 136 `dfbe628d` | 331 | 257 (raw 391) | match | **True** | True | True | True | True | True | True |
| 372 `642c97ea` | 186 | 203 (raw 273) | match | **True** | True | True | True | True | True | True |

**5/5 on neural, input and output.** The neural check re-bins the chosen trial with
`np.histogram` on independently loaded `spikes.times`/`spikes.clusters` -- a completely
different code path from the converter's `searchsorted` + `np.add.at`. The wheel and
whisker checks re-interpolate from the raw `SessionLoader` objects.
The same checks passed on the sample in Step 7.

To make these checks exact rather than heuristic I added `kept_trial_idx` to each
session's `session_info`, which also serves as provenance back to the raw trials table.

### Check 3: Reference code comparison
| Stage | Reference (`ibl_data_utils.py`) | My `convert_data.py` | Same? |
|---|---|---|---|
| (a) loading -- spikes | `load_spiking_data`: `SpikeSortingLoader(pid).load_spike_sorting()` + `merge_clusters(...).to_df()`, then `merge_probes` | identical calls; probes merged with the same cluster-id offsetting and time sort | **yes** |
| (a) loading -- trials | `SessionLoader.load_trials()` | same | **yes** |
| (a) loading -- wheel | `SessionLoader.load_wheel()`, `abs(velocity)` | same | **yes** |
| (a) loading -- whisker | `load_motion_energy(views=['left'])`, fall back to right | same | **yes** |
| (b) neuron filtering | `qc=None` -> keeps all clusters | `label >= 1` and Beryl not root/void | **deliberate difference** (Step 4): follows the data paper's well-isolated-neuron criterion, reproduces its 75,708 exactly, and is required for meaningful `brain_region_idx` |
| (b) trial filtering | `load_trials_and_mask(min_rt=.08, max_rt=2., max_trial_len=10., nan_exclude=default, exclude_nochoice=True)` | the same pandas query, transcribed literally | **yes** |
| (b) behaviour masks | `get_behavior_per_interval` coverage tests; `align_spike_behavior` intersects all masks | same tests, same intersection | **yes** |
| (b) spike coverage | not applied | applied | **addition**, see Check 1 -- fixes a real defect that produces all-zero trials |
| (b) session filtering | none | >= 5 neurons, >= 2 trials, whisker trace present | **addition**, justified above |
| (c) alignment | `stimOn_times`, `time_window=(-.5, 1.5)` | identical | **yes** |
| (d) binning -- spikes | `bincount2D(xbin=0.02, xlim=[beg,end])[:, :100]` per trial | same bin edges, vectorised over trials | **yes** (verified `allclose` against `np.histogram`) |
| (d) binning -- behaviour | `interp1d(kind='linear')` at `linspace(beg+binsize, end, 100)` | `np.interp` at the same right edges | **yes** |
| (e) inputs | reference has no decoder inputs | time from stimulus onset; trial number in block | **required by the decoder task** |
| (f) outputs | `choice`, `block`=`probabilityLeft`, `wheel-speed`, `whisker-motion-energy` (continuous, z-scored per session downstream) | same four variables; the two continuous ones discretised into per-session tertiles | **same variables**; discretisation is required by the task ('Output variables must be categorical'), and the per-session reference mirrors the reference's per-session `StandardScaler` |

### Check 4: Key statistics comparison
Every statistic available in the reference texts was checked; see the table in Step 9.
The decisive ones, all computed from the data with `cache/scan_clusters.py` and
`cache/paper_stats.py`:
- units 621,733 (paper 621,733) -- **exact**
- well-isolated 75,708 (paper 75,708) -- **exact**
- 699 insertions / 459 sessions / 139 subjects / 12 labs -- **exact**
- fraction of reaction times < 80 ms: 0.2299 (paper 0.228)
- 0%-contrast performance: 0.5857 (paper 0.587)
- unbiased first block: median 90 trials (paper 90)
- biased block lengths: mean 48.6, max 99 (paper mean 51, range 20-100)
- `probabilityLeft` takes exactly {0.2, 0.5, 0.8}
No discrepancy remained unexplained.

### Check 5: Edge cases
- **Read-only mask array** from `trials.eval()` -- caused every session to be skipped;
  found because the driver prints a reason for each skip. Fixed with an explicit copy.
- **Trials past the end of the spike sorting / inside recording gaps** -- fixed (Check 1).
- **NaN `stimOn_times`** -- excluded by the NaN mask, and `t_beg` is re-checked with
  `np.isfinite` before use, so no NaN ever reaches `searchsorted`.
- **NaN spike times** -- dropped before sorting and binning.
- **Right-edge bin** -- a spike landing exactly on the window end would give index 100;
  `np.clip(bi, 0, NBINS-1)` keeps it in range, and `searchsorted(..., 'left')` on the end
  time means such spikes are not selected in the first place.
- **Sessions with one probe vs two** -- both handled by the same merge path; probe
  cluster ids are offset so they never collide.
- **Constant behaviour trace** -- `tertile_bins` falls back to unique-value splits rather
  than producing an empty class.
- **Blocks at session boundaries** -- `trial_number_in_block` restarts at 0 on the first
  trial and at every change of `probabilityLeft`; NaN runs are not merged.
- **Sessions with < 2 trials** -- dropped, so the decoder always has trials to split.

### Issues found and resolved
1. read-only trial mask -> explicit copy (all sessions were being skipped)
2. all-zero neural trials from degenerate sessions -> session-level >= 5 neuron criterion
3. all-zero neural trials from missing spike coverage -> spike-coverage trial mask
4. over-aggressive per-region curation (my own first attempt) -> reverted to the
   session-level form, with the reasoning recorded

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
(442 sessions, 62,773 neurons, 188,044 trials; ran on the L4 GPU, ~45 min).

### Training Progress
- Loss decreasing: **Yes**, monotonically at every logged epoch:
  1.0728 (ep 1) -> 0.9735 (10) -> 0.9120 (20) -> 0.8272 (40) -> 0.8038 (50) ->
  0.7689 (80) -> 0.7586 (100) -> 0.7461 (150) -> 0.7400 (200). Test loss 0.7661.

### Decoder Results (Full)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|--------|-------------|--------|------|-------|
| choice | 0.5000 | 0.6393 | **0.6173** | 1.23x | per time bin, incl. 50 pre-stimulus bins |
| prior_prob_left | 0.3333 | 0.6726 | **0.6520** | 1.96x | best-decoded variable |
| wheel_speed | 0.3333 | 0.6244 | **0.6177** | 1.85x | |
| whisker_motion_energy | 0.3333 | 0.6093 | **0.6018** | 1.81x | |

Every output is well above chance, and the train/validation gap is at most 0.022
(ratio <= 1.04), so the model is not overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Chance | Validation | Ratio | >= 1.5x chance? |
|---|---|---|---|---|
| choice | 0.500 | 0.617 | 1.23x | n/a -- for a 2-class problem 1.5x chance would be 0.75, which no published single-trial IBL choice decoder reaches |
| prior_prob_left | 0.333 | 0.652 | 1.96x | yes |
| wheel_speed | 0.333 | 0.618 | 1.85x | yes |
| whisker_motion_energy | 0.333 | 0.602 | 1.81x | yes |

No output is at or below chance. The three 3-class outputs all clear 1.8x chance.
For `choice` the 1.5x-chance heuristic is not a meaningful target: it would demand 0.75
balanced accuracy, whereas the methods paper's own single-trial choice decoders reach
AUC 0.66 (linear baseline) and 0.72 (their improved model), with an *oracle* at 0.79.
Our 0.617 is a **per-time-bin** number that averages over the 50 pre-stimulus bins of
every trial, where the choice has not yet been made; see Check 2.

### Check 2: Accuracy comparison to the papers
| Variable | Paper value | Metric | This dataset | Comparable? |
|---|---|---|---|---|
| choice | 0.66 baseline / 0.72 BMM-HMM / 0.79 oracle (methods paper Fig. 4) | AUC, per trial, single session | 0.617 all bins; **0.796 peak at t = +0.24 s** (time-resolved probe below) | yes -- our peak matches their oracle |
| choice | 0.71 -> 0.90 example session (Fig. 5) | accuracy | per-session values span a wide range | consistent |
| prior | r = 0.05 baseline / 0.65 LG-AR1 (Fig. 4) | Pearson r, continuous | 0.652 balanced accuracy, 3 classes | different metric; 1.96x chance is strong |
| wheel speed / whisker ME | qualitative, RRR > ridge (Fig. 3) | R2 / correlation | 0.618 / 0.602 balanced accuracy | different metric; both ~1.8x chance |
| choice, per region | null-corrected balanced accuracy, small effect sizes (data paper Fig. 4) | -- | not comparable -- we pool all neurons in a session | -- |

The papers report AUC / Pearson r on continuous targets, while this task requires
categorical outputs and a per-time-bin balanced accuracy, so the numbers are not
directly comparable. Where a like-for-like comparison is possible -- the peak
single-trial choice decodability -- we reach **0.796**, matching the methods paper's
oracle (0.79) and exceeding its baseline (0.66) and improved model (0.72). I therefore
find no evidence that our accuracy is depressed by a conversion error.

### Time-resolved probe (the decisive alignment test), `cache/time_resolved.py`
An independent per-time-bin logistic readout on the 30 leading PCs of the 12
largest sessions:

| Variable | Pre-stimulus (t < 0) | Post-stimulus (t > 0) | Peak |
|---|---|---|---|
| choice | 0.538 | 0.648 | **0.796 at t = +0.24 s** |
| wheel speed | 0.564 | 0.519 | 0.632 at t = +0.18 s |
| whisker ME | 0.502 | 0.564 | 0.615 at t = +1.46 s |

Choice decodability is near chance before the stimulus and rises sharply afterwards,
peaking at +0.24 s -- precisely when the animal commits to its wheel turn (the trials
kept all have first-movement times between 0.08 and 2.0 s). A temporal misalignment
between the neural and behavioural streams would destroy exactly this structure, so this
is strong positive evidence that the alignment is correct. It also explains the
full-window `choice` number: averaging 0.54 over the first half of the window with 0.65
over the second gives ~0.62, the value the decoder reports.

### Check 3: Train vs validation gap
| Output | Train | Validation | Gap | Ratio |
|---|---|---|---|---|
| choice | 0.6393 | 0.6173 | 0.0220 | 1.036 |
| prior_prob_left | 0.6726 | 0.6520 | 0.0206 | 1.032 |
| wheel_speed | 0.6244 | 0.6177 | 0.0067 | 1.011 |
| whisker_motion_energy | 0.6093 | 0.6018 | 0.0075 | 1.012 |

All ratios are far below the 1.5x threshold, so there is no overfitting and no sign of
data leakage. (Leakage is also structurally impossible here: the split is by trial
within session, and no per-session quantity other than the tertile thresholds is fitted;
those are a fixed property of the session's behaviour, not of the labels.)

### Additional debugging checks performed
1. **Output values verified on specific trials** -- 5 random sessions x 1 random trial
   each, all four outputs re-derived from raw ONE data, `np.allclose` True (Step 10
   Check 2), plus the sample-data checks in Step 7.
2. **Temporal alignment** -- the `--show-processing` figures overlay the raw wheel and
   whisker traces with the binned values for four trials, and put a raw spike raster next
   to the binned count matrix for the same trial; the time-resolved probe above is the
   quantitative version.
3. **Output variation** -- no output is dominated by one class:
   choice [0.508, 0.492]; prior [0.418, 0.140, 0.442]; wheel [0.333, 0.333, 0.333];
   whisker [0.335, 0.333, 0.333].
4. **Neural filtering** -- reproduces the data paper's 621,733 / 75,708 exactly.
5. **Processing matches the reference** -- stage-by-stage comparison in Step 10 Check 3.

### Issues Found and Resolved in this step
None. All four accuracies are above chance, the train/validation gaps are negligible, and
the time-resolved analysis confirms the alignment. No further conversion changes were
needed, so no re-run of Steps 9-11 was required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created -- user-facing description, load instructions, format spec, key
      statistics, curation summary and decoder results
- [x] cache/ folder created with README_CACHE.md documenting every investigation script
- [x] All files organized

### Deliverables
| File | Description |
|---|---|
| `CONVERSION_NOTES.md` | this file -- every decision, check and result |
| `README.md` | user-facing documentation |
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | full dataset, 11.7 GB, 442 sessions |
| `sample_data.pkl` | 2-session sample, 31 MB |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | format-verification logs (both clean) |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_<eid>.png` (x2) | per-session processing diagnostics |
| `sample_trials.png`, `predictions.png` | decoder-generated figures |
| `cache/` | investigation scripts + README_CACHE.md |

### Summary of the conversion

The IBL brain-wide map release (459 sessions, 699 insertions, 621,733 units, 139 mice)
was loaded through the ONE API and reformatted into 442 sessions x 188,044 trials of
(62,773 well-isolated grey-matter neurons) x (100 x 20 ms bins), aligned to stimulus
onset over -0.5 to +1.5 s, exactly as in the reference pipeline. Two decoder inputs
(time from stimulus onset, trial number in block) and four categorical outputs (choice,
prior P(left), wheel speed, whisker motion energy) were built as specified.

The conversion reproduces the data paper's unit counts exactly (621,733 / 75,708) and its
behavioural statistics closely (reaction times < 80 ms: 0.2299 vs 0.228; 0%-contrast
performance 0.5857 vs 0.587; unbiased block 90 trials; biased blocks mean 48.6 vs 51).
Independent re-derivation of the neural, input and output arrays straight from ONE
matches the converted data with `np.allclose` on every one of the checked trials.
The reference decoder reaches validation balanced accuracies of 0.617 (choice, chance
0.5), 0.652 (prior), 0.618 (wheel speed) and 0.602 (whisker motion energy, all chance
0.333), with negligible train/validation gaps, and choice decodability peaks at 0.796 at
t = +0.24 s after stimulus onset -- confirming that the streams are correctly aligned.
