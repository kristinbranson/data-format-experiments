# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-Wide Map (BWM) public release — "A brain-wide map of neural activity during complex behaviour" (IBL et al.), ONE cache staged at `/app/data/one_cache`
- **Method reference**: Zhang et al. 2026, *Neuron*, "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (code in `/app/code/code_zhang2025`)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file)
- `Dockerfile`, `docker-compose.yaml`, `stage_cache.sh`, `.manifest` — container plumbing
- `datapaper.pdf` — IBL brain-wide map paper
- `methodpaper.pdf` — Zhang et al. decoding paper
- `dataarchitecture.pdf` — IBL data architecture white paper
- `methods.txt` — excerpts of both papers
- `code/` — `code_zhang2025/` (method reference code) and `ibllib/` (IBL library source)
- `data/one_cache/` — ONE cache (release tables + symlinked lab/subject/session ALF trees)
- `ibl_docs/` — downloaded ONE/ibllib/iblatlas documentation
- `decoder.py`, `train_decoder.py` — decoder reference implementation and driver

Environment verified:
- `python3` works; `numpy 2.3.5`, `torch 2.6.0+cu124` (CUDA available, NVIDIA L4 23 GB)
- `one 3.5.2`, `ibllib 4.0.1`, `brainbox`, `iblatlas` importable
- 128 CPUs, 1 TB RAM, 3.4 TB free disk

ONE access (offline): the cache is a *local* ONE cache plus a cached Alyx REST store
(`/app/data/one_cache/.rest`, 6483 responses) and an auth token in `$HOME/.one`.
`ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
tables_dir='/app/data/one_cache/Brainwidemap')` works and even `one.eid2pid()` resolves from
the REST cache. The release tables live in per-tag subdirectories
(`Brainwidemap`, `2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`), so `tables_dir` must be
given explicitly or ONE warns "No cache tables found".

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The relevant reference pipeline is `code_zhang2025/src/0_data_caching.py`, which builds exactly
the kind of trialised, binned dataset we need. Its parameters:

```python
params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
          'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']
```

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data(one, eid, bwm_df, params)` | `src/utils/ibl_data_utils.py` | LOADING | Per-eid entry point: loads all probes' spike sorting, merges probes, loads trials + trial mask, loads continuous behaviours |
| `load_spiking_data(one, pid, qc=None)` | `ibl_data_utils.py` | LOADING/CURATION | `SpikeSortingLoader.load_spike_sorting()` + `merge_clusters()`; `qc=None` ⇒ keep **all** clusters, `qc=1` ⇒ keep clusters with `label >= 1` (IBL "good"/well-isolated units) |
| `merge_probes(spikes_list, clusters_list)` | `ibl_data_utils.py` | LOADING | Concatenates probes of one session into one population, re-indexing `spikes['clusters']`, re-sorting spikes by time |
| `load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., nan_exclude='default', max_trial_len=10.0, exclude_nochoice=True)` | `ibl_data_utils.py` | CURATION | Returns full trials table + boolean mask of usable trials |
| `list_brain_regions` / `select_brain_regions` | `ibl_data_utils.py` | CURATION | Maps cluster `acronym` → **Beryl** atlas mapping, selects clusters in requested regions (`single_region=False` ⇒ all regions together) |
| `bin_spiking_data(reg_clu_ids, neural_df, trials_df, **params)` | `ibl_data_utils.py` | PROCESSING | Builds intervals `stimOn + (-0.5, 1.5)`, bins spikes at 20 ms via `bincount2D`, returns `(n_trials, T=100, n_clusters)` |
| `get_spike_data_per_interval` | `ibl_data_utils.py` | PROCESSING | Per-interval `bincount2D(times, clusters, xbin=0.02, xlim=[t_beg, t_end])`, keeps first `n_bins = ceil(2/0.02) = 100` columns |
| `load_target_behavior(one, eid, target)` | `ibl_data_utils.py` | LOADING | `SessionLoader.load_wheel()` → `wheel-speed` = `abs(velocity)`; `SessionLoader.load_motion_energy(views=['left'])` → `leftCamera.whiskerMotionEnergy` |
| `bin_behaviors(one, eid, behaviors, trials_df, **params)` | `ibl_data_utils.py` | PROCESSING | Per-trial: `choice`, `block` (= `probabilityLeft`), `reward` (`rewardVolume>1`), `contrast`; time-varying behaviours via `get_behavior_per_interval` |
| `get_behavior_per_interval` | `ibl_data_utils.py` | PROCESSING/CURATION | Slices behaviour samples inside each interval, linearly interpolates onto `np.linspace(t_beg + binsize, t_end, 100)`; marks interval bad if no samples, if it "starts too late"/"ends too early" (> 1 binsize gap at either edge), or if interval times are NaN |
| `align_spike_behavior(binned_spikes, binned_behaviors, beh_names, trials_mask)` | `ibl_data_utils.py` | CURATION | Drops trials failing the trials mask *or* missing any behaviour, keeping neural and behaviour trial-aligned |
| `standardize_spike_data` | `src/utils/data_loader_utils.py` | PROCESSING (model side) | Per-time-bin z-scoring; happens at model-fit time, **not** in the cached dataset |
| `create_dataset` | `src/utils/dataset_utils.py` | SAVING | Stores binned spikes as `np.ubyte` CSR — i.e. the cached neural data are **raw spike counts**, not rates |

### Notes
- Answers to the checklist questions: this is **electrophysiology**, so no ΔF/F. Cell quality
  filtering *is* available (`load_spiking_data(qc=...)` on the IBL `label` metric) — see Step 4
  for the decision.
- `load_spiking_data` also calls `spike_loader.raw_electrophysiology(band="ap", stream=True).fs`
  purely to record the AP sampling frequency in metadata. That streams raw binary from the
  remote server and is impossible (and pointless) offline — omitted.
- `SessionLoader` in the installed ibllib 4.0.1 is keyword-only (`SessionLoader(one=one, eid=eid)`),
  while the 2025 reference code calls it positionally. Reference helpers are therefore called
  with an explicitly-constructed `sess_loader`.
- Example decoder outputs shipped with the reference code
  (`data/example_decoder_outputs/<eid>/<target>/<region>/metrics/fold_*.npy`) give per-fold
  metrics under four neuron-selection strategies: `all_ks`, `good_ks`, `thresholded`,
  `density_based`. Averaged over the 17 example sessions × 5 folds, region `all`:

  | target | all_ks | good_ks | thresholded | density_based |
  |--------|--------|---------|-------------|---------------|
  | choice (AUC) | 0.896 | 0.869 | 0.879 | 0.884 |
  | prior (R²)   | 0.270 | 0.182 | 0.248 | 0.247 |

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
ONE cache, standard IBL ALF layout:

```
/app/data/one_cache/
  Brainwidemap/{sessions,datasets}.pqt      # release tables: 480 sessions, 76,563 datasets
  2022_Q4_IBL_et_al_BWM/…                   # 354 sessions
  2025_Q3_IBL_et_al_BWM/…                   # 459 sessions, 5,339 datasets (video revisions)
  .rest/                                    # 6,483 cached Alyx REST responses
  <lab>/Subjects/<subject>/<date>/<number>/alf/
      _ibl_trials.table.pqt, _ibl_trials.{goCueTrigger_times,stimOff_times}.npy
      _ibl_wheel.{position,timestamps}.npy
      {left,right}Camera.ROIMotionEnergy.npy, _ibl_{left,right}Camera.times.npy
      probe0N/pykilosort/{spikes.times,spikes.clusters,spikes.depths}.npy
                        {clusters.channels,clusters.depths,clusters.metrics.pqt}
                        channels.brainLocationIds_ccf_2017.npy
```

Only the files needed for this analysis are staged (16 per session), not the full 76,563-dataset
release; everything is loaded through `ONE` / `brainbox` loaders, never by reading files directly.

Key variables:
- `trials` table (20 columns): `stimOn_times`, `goCue_times`, `goCueTrigger_times`,
  `firstMovement_times`, `response_times`, `feedback_times`, `feedbackType`, `choice`,
  `contrastLeft`, `contrastRight`, `probabilityLeft`, `rewardVolume`, `intervals_0/1`,
  `stimOff_times`, `quiescencePeriod`, …
  - `choice ∈ {-1, 0, +1}`; verified empirically (see Step 4) that `choice == +1` is a **left**
    report and `choice == -1` a **right** report; `0` = no-go.
  - `probabilityLeft ∈ {0.2, 0.5, 0.8}`.
- `wheel`: `SessionLoader.load_wheel()` returns position interpolated to 1 kHz plus Gaussian-smoothed
  `velocity`/`acceleration`.
- `leftCamera.ROIMotionEnergy` (whisker motion energy) at ~60 Hz; `rightCamera` at ~150 Hz.
- clusters table after `merge_clusters`: IBL QC metrics (`amp_median`, `noise_cutoff`,
  `slidingRP_viol`, `label`, `firing_rate`, …) plus Allen `acronym` per cluster.

### Dataset Size (from data files / release table)
| Statistic | Value |
|-----------|-------|
| Sessions (BWM freeze `bwm_release.csv`) | 459 |
| Probe insertions | 699 (701 probe dirs staged) |
| Subjects | 139 |
| Sessions / subject | 1–? (mean 3.3) |
| Session dirs staged | 461 |
| Sessions with `leftCamera.ROIMotionEnergy` | 439 |
| Sessions with `rightCamera.ROIMotionEnergy` | 422 |
| Clusters / probe (example sessions) | 898, 1728 (2 probes), 1202 (2 probes) |
| Good clusters (`label >= 1`) / session | 76, 264, 163 in the 3 probed examples |
| Trials / session (raw) | 565, 425, 557 in the same examples |
| Trials / session passing reference mask | 407, 244, 148 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Mice | 139 | "We trained 139 mice (94 male and 45 female)" (data paper) |
| Probes | 699 | "we inserted 699 Neuropixels probes" |
| Sessions released | 459 | "a total of 459 sessions, 699 insertions and 621,733 neurons remained" |
| Units (all Kilosort) | 621,733 (avg 889/probe) | "This process produced 621,733 units (including multineuron activity), averaging 889 per probe" |
| **Well-isolated neurons** | **75,708 (avg 108/probe)** | "identified 75,708 well-isolated neurons, averaging 108 per probe" |
| Sessions used by method paper | **433** | "We apply our models to 433 IBL sessions, covering 270 brain regions" |
| Brain regions | **270** | same |
| Trials/session floor | ≥250 (release), ≥400 (analyses) | "mice performed at least 250 trials"; "Only sessions with at least 400 trials were retained" |
| Neural data time bin | **20 ms**, T = **100** | "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps" |
| Behaviour sampling | 60 Hz (left cam) | "wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz" |
| Alignment for choice | stimulus onset, −0.5 → +1.5 s | "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset" |
| Reward rate | ~0.85 (measured on example session) | task performance ≥90% on 100% contrast trials |
| Block structure | first 90 trials at 0.5; then 0.2/0.8 blocks of 20–100 trials, mean 51 | "After an initial 90 unbiased trials… empirical mean of 51 trials" |
| Contrast set | {0, 6.25, 12.5, 25, 100}% | data paper |
| Choice decoding (Zhang, region `all`) | AUC 0.90 (all units) / 0.87 (good units) | example decoder outputs |
| Prior decoding (Zhang, region `all`) | R² 0.27 / 0.18 | example decoder outputs |

### Processing Details
- **Alignment**: `stimOn_times`, window `(-0.5, +1.5) s` ⇒ 2 s interval.
- **Binning**: 20 ms non-overlapping ⇒ T = 100. Spike bin `k` spans
  `[stimOn − 0.5 + 0.02k, stimOn − 0.5 + 0.02(k+1))` (this is exactly what `bincount2D`
  with `xlim=[t_beg,t_end]` does, and the reference then truncates to the first 100 columns).
- **Continuous behaviour**: linearly interpolated onto `t_beg + 0.02·(k+1)`, i.e. the **right edge**
  of each spike bin.
- The method paper also describes 50-ms bins in a −0.5/+1.5 s window for choice and a
  movement-aligned 0/+1 s window for the dynamic behaviours; the *released caching code*
  (which is what is reproducible and which produces one dataset carrying all four behaviours)
  uses the stimulus-onset 2 s / 20 ms / T=100 configuration, which is also the configuration
  stated in the paper's Results ("2-s trials… 20-ms bins… T = 100"). The Decoder Task here
  fixes stimulus-onset alignment, so that configuration is used.

### Curation Steps

**Neuron curation rules** (data paper "Neurons and brain regions"):
- Exclude units failing any of the three RIGOR single-unit metrics: amplitude > 50 µV,
  noise cut-off < 20 µV, refractory-period violation. Units passing all three are
  "well-isolated neurons" (75,708 of 621,733). In the IBL cluster table this is exactly
  `label >= 1` (`label` is the fraction of the three metrics passed: 0, 1/3, 2/3, 1).
- Analyses further restricted to **grey-matter** regions in the Allen CCF, with ≥5 neurons
  per session and recorded in ≥2 sessions.

**Trial curation rules** (data paper "Trials", implemented in `load_trials_and_mask`):
- Exclude a trial if any of `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`,
  `firstMovement_times`, `feedbackType` is NaN.
- Exclude if reaction time (`firstMovement_times − stimOn_times`) ∉ [0.08, 2.00] s.
- Reference code adds `max_trial_len=10.0` (`feedback_times − goCue_times ≤ 10 s`) and
  `exclude_nochoice=True` (`choice != 0`).

### Decoders Trained (reference)
| Decoded variable | Metric | Reference value (region `all`) |
|------------------|--------|-------------------------------|
| choice | AUC | 0.896 (all units) / 0.869 (good units) |
| prior | R² | 0.270 / 0.182 |
| wheel speed | R² | reported only in figures (single-trial R², ≈0.4–0.6) |
| whisker motion energy | R² | reported only in figures (≈0.5–0.7) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neuron quality filter | `load_spiking_data(qc=None)` ⇒ **all** Kilosort clusters | `label>=1` selects ~8–15% of clusters (76/898, 264/1728, 163/1202) | Method paper: "bin spike counts using **all neurons**, sorted by Kilosort 2.5"; data paper: analyses restricted to the **75,708 well-isolated neurons** | **Use well-isolated neurons (`label >= 1`)**. Rationale below. |
| Brain-region set | `list_brain_regions` keeps every Beryl acronym incl. `root`/`void`; `MultiRegionDataModule` drops `root`/`void` | 15–20% of good clusters map to `root`/`void` | Data paper restricts to Allen **grey matter** | Drop `root`/`void` (non-grey-matter / unassigned), keep everything else. Gives a Beryl region list comparable to the paper's 270 regions. |
| Time bin | code: 20 ms, T=100 | — | Results: 20 ms, T=100; STAR Methods "data processing": 50 ms for choice/prior, 20 ms for dynamic behaviours | Use **20 ms, T = 100**: it matches the released code and the Results text, and the Decoder Task requires one common binning for all four outputs. |
| Alignment | code: `stimOn_times` | — | STAR Methods uses `firstMovement_times` for the dynamic behaviours | Use **`stimOn_times`** — mandated by the Decoder Task and by the released caching code. |
| `allow_nans` for behaviours | reference passes `allow_nans=True` and later imputes NaNs with the trial mean inside the data loader | motion energy occasionally has NaN runs | — | Drop trials whose wheel-speed or whisker-ME window contains NaN. The target format forbids NaN, and imputing a value then discretising it would fabricate labels. |
| Sessions | freeze file lists 459 eids | 461 session dirs staged | Method paper used **433** sessions | Start from the 459-eid freeze, drop sessions with no usable whisker video, no surviving neurons, or <2 usable trials; expect to land near 433. |
| `choice` sign | code stores raw `choice` | on 100%-contrast trials: stim-left ⇒ `choice=+1` in 71/72 trials, stim-right ⇒ `choice=−1` in 61/63 | mice report the stimulus side | `choice == +1` ⇒ **left** report ⇒ label 0; `choice == −1` ⇒ **right** report ⇒ label 1. |

**Why well-isolated neurons rather than "all neurons":**
1. The data paper — the source of these data — explicitly defines its analysis population as the
   75,708 well-isolated neurons and excludes the rest as multi-unit activity.
2. The reference code exposes exactly this switch (`qc=1` ⇒ `label >= 1`) and its `merge_probes`
   docstring is written against `brainwidemap.load_good_units`.
3. Feasibility: all 621,733 units at T=100 × ~300 usable trials × 459 sessions would be ≈120 GB of
   float32 — not a storable/trainable dataset. Well-isolated units give ≈8 GB.
4. Cost is small and quantified: the reference's own example outputs show choice AUC 0.896 with
   all units vs 0.869 with good units (−3%).

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` (all probes, merged) | `neural[s][k]` (n_neurons, 100) float32 | spike counts in 20 ms bins over `stimOn + [-0.5, 1.5)` | `merge_probes`, `bin_spiking_data`/`bincount2D` | raw counts, no rate conversion, no z-scoring (reference z-scores inside the model's data loader, not in the cached data) |
| bin right-edge time relative to `stimOn_times` | `input[s][k][0, :]` | `-0.5 + 0.02·(k+1)` ∈ [−0.48, 1.50] | `get_behavior_per_interval`'s `x_interp` grid | "time since stimulus onset", continuous, time-varying |
| index of the trial within its `probabilityLeft` block | `input[s][k][1, :]` | 0-based count from block start, computed on the **unfiltered** trials table, broadcast over T | — (new; required by Decoder Task) | "trial number in block", continuous, per-trial → broadcast to time-varying |
| `trials.choice` | `output[s][k][0, :]` | `+1 → 0` (left), `−1 → 1` (right); broadcast over T | `bin_behaviors` (`choice`) | binary, per-trial |
| `trials.probabilityLeft` | `output[s][k][1, :]` | `0.2→0, 0.5→1, 0.8→2`; broadcast over T | `bin_behaviors` (`block`) | 3-way, per-trial |
| `SessionLoader.wheel.velocity` | `output[s][k][2, :]` | `abs(velocity)` → interp to bin edges → per-session tertiles → {0,1,2} | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | time-varying |
| `leftCamera.ROIMotionEnergy` (fallback `rightCamera`) | `output[s][k][3, :]` | interp to bin edges → per-session tertiles → {0,1,2} | `load_target_behavior('left-whisker-motion-energy')` | time-varying |
| `clusters.acronym` → Beryl | `brain_region_idx[s]`, `brain_regions` | `BrainRegions().acronym2acronym(..., 'Beryl')` | `list_brain_regions` | `root`/`void` dropped |
| `bwm_release.csv` `subject` | `subjects`, `subject_idx` | unique sorted list | — | |

### Key Decisions
1. **Alignment / window / binning**: `stimOn_times`, `(-0.5, +1.5) s`, 20 ms ⇒ T = 100.
   Exactly the reference `params`; also required by the Decoder Task.
2. **Neuron curation**: IBL `label >= 1` (all three RIGOR metrics passed) **and** Beryl region
   ∉ {`root`, `void`}. Sessions are merged across probes first (`merge_probes`), as the reference
   and the data paper both do ("neurons in the same session and region were combined across probes").
   The data paper's extra region criteria (≥5 neurons/session, region seen in ≥2 sessions) are
   *not* applied: they exist to make region-wise statistical maps comparable, whereas here every
   session is decoded from its whole population, and applying them would delete neurons that
   carry signal for a session-level decoder.
3. **Trial curation**: `load_trials_and_mask(min_rt=0.08, max_rt=2.0, nan_exclude='default',
   max_trial_len=10.0, exclude_nochoice=True)` — the reference call verbatim — plus the
   reference's behaviour-coverage checks (`get_behavior_per_interval`), plus rejection of
   trials with NaN in either continuous behaviour.
4. **Neural values**: raw spike counts stored as float32 (matching the reference's `ubyte` CSR
   cache). Standardisation is a model-side step in the reference and the provided decoder does
   its own SVD projection.
5. **Discretisation of continuous outputs**: per-session tertiles (33.3 / 66.7 percentiles) of the
   binned values pooled over all retained trials of that session, giving classes
   `low`/`medium`/`high`. Per-session rather than global because whisker motion energy is in
   arbitrary units that depend on camera, illumination and ROI size and is therefore not
   comparable across sessions; tertiles because the Decoder Task asks for 3 bins and balanced
   accuracy is the evaluation metric, so equal-frequency bins are the natural choice.
6. **Everything time-varying**: `input` and `output` are both `(d, 100)` per trial; per-trial
   quantities are broadcast across time, as the format instructions prefer.
7. **Session exclusion**: sessions from the 459-eid BWM freeze; dropped if no whisker motion
   energy is available, if no neuron survives curation, or if fewer than 2 trials survive.

### Planned Sanity Checks
- [ ] #sessions ≈ 433 (method paper), #subjects ≤ 139, Beryl regions ≈ 270
- [ ] mean good neurons/session ≈ 75,708 / 459 ≈ 165 before grey-matter filtering
- [ ] T == 100 for every trial; `input[0]` range == [−0.48, 1.50]
- [ ] choice class balance near 50/50; prior classes ≈ {0.2: 0.42, 0.5: 0.16, 0.8: 0.42}
- [ ] wheel-speed and whisker-ME class fractions ≈ 1/3 each by construction
- [ ] spike counts non-negative integers; mean firing rate per neuron in a plausible 0.1–50 Hz range
- [ ] independent re-derivation of one trial's spike counts, input, and each output from
      freshly loaded ONE objects (`np.allclose`) — Step 10
- [ ] trial-in-block resets exactly at `probabilityLeft` changes and the first block has 90 trials

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the plan. Structure:

| Function | Role | Reference counterpart |
|----------|------|-----------------------|
| `get_one()` | per-worker offline ONE client (`tables_dir=.../Brainwidemap`) | `0_data_caching.py` header |
| `load_session_spikes` | `SpikeSortingLoader.load_spike_sorting` + `merge_clusters` per probe, then `merge_probes` | `prepare_data` / `load_spiking_data` / `merge_probes` |
| `select_neurons` | `label >= 1` AND Beryl acronym ∉ {root, void} | `load_spiking_data(qc=1)` + `list_brain_regions`/`select_brain_regions` |
| `load_trials` | `load_trials_and_mask(max_trial_len=10.0)` (reference function, imported verbatim) | identical |
| `load_continuous_behaviors` | `SessionLoader.load_wheel` → abs(velocity); `load_motion_energy(['left'])` with right-camera fallback | `load_target_behavior`, `bin_behaviors` |
| `bin_spikes` | vectorised equivalent of `bincount2D(..., xbin=0.02, xlim=[t_beg,t_end])[:, :100]` | `get_spike_data_per_interval` |
| `interpolate_behavior` | same slicing (`searchsorted` right/left), same target grid `linspace(t_beg+bin, t_end, 100)`, same four skip rules | `get_behavior_per_interval` |
| `discretize_tertiles` | per-session 33.3/66.7 percentile split | new (categorical-output requirement) |
| `trial_number_in_block` | 0-based index within each constant-`probabilityLeft` run, on the unfiltered table | new (Decoder Task input) |

Code inefficiencies identified in the reference:
- `get_spike_data_per_interval` spawns a multiprocessing pool **per session** and calls
  `bincount2D` once **per trial**, which dominates its runtime.
- `get_behavior_per_interval` likewise pools per behaviour per session.
- `prepare_data` streams the raw AP band from the server only to read its sampling rate.

Code speedups added:
- Spike binning replaced by a single `np.searchsorted` for all trial edges plus one
  `np.bincount` on `cluster*T + bin` per trial: 0.01 s/session instead of pool startup +
  ~400 `bincount2D` calls.
- Parallelism moved up one level: one worker process per **session** (default 24), so the
  pools are created once for the whole run instead of ~2000 times.
- The raw-ephys sampling-rate query is dropped (metadata only, needs the network).
- Spikes are binned only for trials that survive curation.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(→ `/app/conversion_sample_out.txt`), then
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only`
(→ `/app/verification_sample_out.txt`).

### Sample Statistics (2 sessions, subject NYU-11)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 |
| Neurons (total) | 253 |
| Neurons / session | 61, 192 (mean 126.5) |
| Clusters / session (all Kilosort) | 898, 1728 |
| Good units / session (label ≥ 1) | 76, 264 |
| Trials (total) | 651 |
| Trials / session | 407, 244 (of 565, 425 raw) |
| Brain regions (Beryl) | 16 |
| T | 100 for every trial |
| `time_from_stim_onset_s` range | [−0.48, 1.50] |
| `trial_number_in_block` range | [0, 89] |
| choice distribution | left 0.518 / right 0.482 |
| prior distribution | 0.2: 0.478, 0.5: 0.161, 0.8: 0.361 |
| wheel_speed distribution | 0.333 / 0.333 / 0.333 |
| whisker ME distribution | 0.333 / 0.333 / 0.333 |
| mean firing rate | 8.47, 7.66 Hz |

Format verification: **no errors, no warnings**.

### Processing Plots Review
`processing_<eid>.png` (4×2 panels) shows, for each of the 2 sessions:
1. binned spike counts of one trial and the population PSTH — the PSTH is flat before 0 and
   rises sharply from ~6.5 Hz to ~11.7 Hz peaking 0.25 s after stimulus onset, which is the
   expected visual/goCue response and confirms there is no temporal offset;
2. wheel speed raw 1 kHz trace overlaid with the interpolated 20 ms samples — they coincide;
   speed is ≈0 before onset and rises ~0.15 s after it;
3. the same for whisker motion energy (raw 60 Hz vs interpolated);
4. the discretisation panels show the class trace switching exactly where the signal crosses
   the green tertile lines;
5. the block panel shows `probabilityLeft` with the trial-in-block sawtooth resetting exactly
   at every block boundary, the first block being 90 trials long at p=0.5;
6. the all-trial wheel-speed class image is dark (class 0) before stimulus onset and bright
   afterwards, with no trial-to-trial jitter in the transition.
No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised spike binning (vs per-trial `bincount2D` in a per-session pool) | binning 0.01 s/session instead of ~10 s |
| session-level multiprocessing (24 workers) | ~24× wall-clock |
| skipping the raw-ephys sampling-rate stream | removes a network call that cannot succeed offline |

| Step | Time / Session | Estimated Total Time (459 sessions) |
|---|---|---|
| load spike sorting | 5.8 s | 2660 s of worker time |
| load behaviour | 0.54 s | 250 s |
| load trials | 0.30 s | 140 s |
| bin behaviour | 0.04 s | 18 s |
| bin spikes | 0.01 s | 5 s |
| **total per session** | **~6.7 s (up to 10 s with 2 probes)** | ~3900 s worker time ⇒ **≈3 min wall-clock with 24 workers** |

Projected dataset size: 0.03 GB for 2 sessions ⇒ ≈7 GB for the full freeze.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions / 651 trials)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|--------------------------|
| choice | 0.500 | 0.6136 | 0.5312 |
| prior | 0.333 | 0.6950 | 0.6497 |
| wheel_speed | 0.333 | 0.5658 | 0.5537 |
| whisker_motion_energy | 0.333 | 0.5579 | 0.5613 |

Loss decreased monotonically (1.953 → 0.766 over 200 epochs) and was still falling at the
last epoch — the model is under-, not over-fit on 2 sessions. All four outputs are above
chance. Choice is the weakest: the decoder scores **every 20 ms bin**, including the 25
pre-stimulus bins and the ~8 bins before the first wheel movement, in which choice is not yet
represented, so per-timestep balanced accuracy is necessarily far below the per-trial AUC of
0.87–0.90 the reference reports. The shared output heads should improve substantially with
the full 400+ session dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/converted_data.pkl --full --n-workers 28 \
    2>&1 | tee /app/conversion_full_out.txt          # 211 s wall clock
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only \
    > /app/verification_full_out.txt 2>&1
```

### Output Files
- `converted_data.pkl`: 11.72 GB (10.95 GB of float32 spike counts)
- `verification_full_out.txt`: created - **"Data format is valid, no errors or warnings."**

### Sessions dropped (18 of the 459 in the BWM freeze)
| Reason | n |
|--------|---|
| no whisker motion energy from either camera | 14 |
| fewer than 5 well-isolated grey-matter neurons | 3 |
| no trial with complete behavioural coverage | 1 |

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | 459 released / **433 used by Zhang et al.** | 459 eids in `bwm_release.csv` | 461 session dirs staged | **441** | yes (between the two) |
| Subjects | 139 | 139 | 139 | **136** (3 lost with their only session) | yes |
| Probe insertions | 699 | 699 | 701 probe dirs | **672** (441 sessions) | yes |
| Kilosort units **per probe** | **889** | - | - | 599,022 / 672 = **891** | yes (0.2%) |
| Kilosort units (scaled to 699 probes) | **621,733** | - | - | 891 x 699 = **622,809** | yes (0.2%) |
| Well-isolated units **per probe** | **108** | - | - | 73,010 / 672 = **108.6** | yes (0.6%) |
| Well-isolated units (scaled to 699 probes) | **75,708** | - | - | 108.6 x 699 = **75,911** | yes (0.3%) |
| Neurons kept (well-isolated AND grey matter) | - | - | - | **62,757** (mean 142.3/session) | n/a |
| Brain regions (Beryl) | **270** | all Beryl acronyms of the session | - | **263** (240 seen in >=2 sessions) | yes (2.6%) |
| Trials/session before curation | >=250 (release), >=400 (analyses) | - | - | 285,031 / 441 = **646** | yes |
| Trials after the reference mask | - | `load_trials_and_mask` | - | 188,020 (66.0% retained) | n/a |
| Trials finally kept | - | - | - | **187,918** (mean 426, min 125, max 1445) | n/a |
| Neural time bin | 20 ms | `binsize: 0.02` | - | **20 ms** | yes |
| T | 100 | `ceil(2/0.02) = 100` | - | **100** for every trial | yes |
| Alignment | stimulus onset | `align_time: stimOn_times` | - | **stimOn_times**, (-0.5, +1.5) s | yes |
| `time_from_stim_onset_s` range | - | `linspace(t_beg+bin, t_end, 100)` | - | [-0.48, 1.50] | yes |
| `trial_number_in_block` range | blocks 20-100 trials (+ 90 unbiased) | - | - | [0, 93] | yes |
| choice distribution | ~50/50 by design | - | - | left **0.508** / right **0.492** (per-session 5-95%: 0.352-0.652) | yes |
| prior distribution | 90 unbiased of ~646 trials => ~0.14 at p=0.5, rest split ~evenly | - | - | 0.2: **0.417**, 0.5: **0.140**, 0.8: **0.442** | yes |
| wheel speed classes | 3 equal bins by construction | - | - | 0.333 / 0.333 / 0.333 | yes |
| whisker ME classes | 3 equal bins by construction | - | - | 0.333 / 0.335 / 0.333 | yes |
| Task performance (100% contrast) | >=90% correct required for release | - | - | 0.98-1.00 in spot-checked sessions | yes |
| Mean firing rate of kept neurons | - | - | - | 11.8 Hz (median 9.8, range 2.5-48) | plausible |

The per-probe unit counts are the strongest available cross-check on the loading and curation
path: both the total-unit and the well-isolated-unit counts per probe reproduce the data
paper to within 0.6%, and were never tuned to do so.

No data was lost silently: every dropped session is listed with its reason in
`conversion_full_out.txt` and in `metadata['skipped_sessions']`, and every dropped trial is
accounted for by the reference trial mask (97,011), missing behavioural coverage (86) or an
empty population recording (16).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output-log verification
`verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**

An earlier full run *did* produce 38 warnings of the form
`Session S, trial K: all neural data is zero`. They were traced to (a) one session with a
single surviving neuron and one with ten, and (b) short dropouts in three otherwise normal
sessions (e.g. session 392: three consecutive trials with no spikes from 108 neurons).
Both causes were fixed rather than tolerated:
* `MIN_NEURONS_PER_SESSION = 5` - the data paper's "at least five well-isolated neurons per
  session" criterion applied at the session level, because here the decoded population is the
  whole session. Removes 3 sessions.
* trials in which the entire population records zero spikes over the whole 2 s window are
  dropped as recording dropouts (16 trials, 0.009%).
The re-run produced zero warnings.

### Check 2: Independent sanity checks (`cache/sanity_checks.py`)
Every quantity is re-derived from the original data loaded through ONE/brainbox, using
different code from `convert_data.py`, and compared with `np.allclose`. Run on 4 randomly
chosen sessions: **96 checks, 96 passed, 0 failed.**

| Check | How it is made independent | Result |
|-------|---------------------------|--------|
| neuron count and region labels | clusters re-loaded per probe, `label`/Beryl re-derived | exact match |
| spike counts, 6 random (trial, neuron) pairs per session | `np.histogram` with explicit 101 bin edges instead of `np.bincount` on flattened indices | `allclose` on all 100 bins |
| whole-trial population spike totals, 3 trials per session | boolean mask on raw spike times | exact |
| input 0 (time from stim onset) | grid rebuilt from constants | `allclose` |
| input 1 (trial in block) | recomputed with a plain python loop over the raw trials table | `allclose` |
| first unbiased block is 90 trials | from the raw `probabilityLeft` | 90 in every session |
| output 0 (choice) | `np.where(choice == 1, 0, 1)` from the raw table | `allclose` |
| output 1 (prior) | `np.select` on raw `probabilityLeft` | `allclose` |
| output 2 (wheel speed class) | `SessionLoader` reloaded, `np.interp` (a different interpolator) + independent tertiles | 99.99-100.0% of bins identical; thresholds `allclose` |
| output 3 (whisker ME class) | same, on the camera recorded in metadata | 99.6-99.8% of bins identical; thresholds `allclose` |
| behavioural plausibility | performance on 100%-contrast trials, and left-stimulus -> left-choice rate | 0.98-1.00 and 0.99-1.00 |

The <0.4% of whisker-ME bins that differ are values sitting exactly on a tertile boundary: the
checker uses `np.interp`, which clamps at the ends of the sample range, whereas the converter
uses `scipy.interp1d(fill_value='extrapolate')` restricted to in-interval samples, exactly as
`get_behavior_per_interval` does. The difference is at most one 60 Hz sample at the right edge
of a trial and never moves a threshold by more than 0.1%.

### Check 3: Reference code comparison
| Stage | Reference (`ibl_data_utils.py` / `0_data_caching.py`) | `convert_data.py` | Same? |
|-------|------------------------------------------------------|-------------------|-------|
| (a) loading | `SpikeSortingLoader.load_spike_sorting` + `merge_clusters(compute_metrics=False)` per pid from `one.eid2pid`, then `merge_probes` | identical calls; pids taken from `bwm_release.csv` instead of `eid2pid` (same mapping, no REST round-trip); the `raw_electrophysiology(...).fs` metadata query is skipped | yes, apart from the omitted metadata field |
| (b) neuron filtering | `qc=None` => all clusters; `list_brain_regions` keeps every Beryl acronym | `label >= 1` (= reference's `qc=1`) and Beryl not in {root, void}; >=5 neurons per session | **deliberate difference**, justified in Step 4 |
| (b) trial filtering | `load_trials_and_mask(one, eid, max_trial_len=10.0)` then `align_spike_behavior` | the **same reference function**, imported and called with the same arguments; behaviour-missing trials removed exactly as `align_spike_behavior` does | yes |
| (c) alignment | `intervals = stimOn + (-0.5, 1.5)` | identical | yes |
| (d) binning | `bincount2D(t, c, xbin=0.02, xlim=[t_beg, t_end])[:, :100]`, spikes with `t_beg <= t < t_end` | `floor((t - t_beg)/0.02)` + `np.bincount`, spikes selected by `searchsorted` on the same half-open interval - algebraically the same operation, ~1000x faster | yes (verified against `np.histogram` in Check 2) |
| (d) behaviour resampling | samples strictly inside the interval, `interp1d(..., fill_value='extrapolate')` onto `linspace(t_beg+bin, t_end, 100)`, four skip rules | identical, same `searchsorted` sides, same grid, same skip rules; `allow_nans` set to False instead of True | yes, except NaN handling (documented) |
| (e) inputs | reference has no decoder-input concept (its models take only spikes) | time-from-onset grid = the reference's behaviour interpolation grid; trial-in-block is new, required by the Decoder Task | n/a |
| (f) outputs | `choice`, `block` = `probabilityLeft`, `wheel-speed` = `abs(velocity)`, `whisker-motion-energy` = left (else right) camera ROI motion energy | the same four variables from the same fields, mapped to the class codes required by the Decoder Task and (for the two continuous ones) tertile-discretised | yes, plus the required discretisation |
| storage | `ubyte` CSR of raw spike counts, no normalisation | float32 raw spike counts, no normalisation | yes |

Differences and their reasons, in full:
1. **Well-isolated units only.** The data paper defines its analysis population this way; the
   reference code exposes the switch (`qc=1`); all units would be ~120 GB. Measured cost from
   the reference's own example outputs: choice AUC 0.896 -> 0.869.
2. **Grey matter only (`root`/`void` dropped).** Data paper: "restricted to regions that were
   designated grey matter". `root`/`void` are not brain regions and would pollute
   `brain_regions`.
3. **>=5 neurons per session.** The data paper's per-region floor applied to the decoded
   population; also removes the degenerate 1-neuron session.
4. **NaN behaviour trials dropped** instead of NaN-imputed at model-fit time: the target format
   forbids NaN and a categorical label cannot be imputed without inventing a class.
5. **Zero-spike trials dropped** (16 trials): recording dropouts.
6. **No AP sampling-rate metadata**: requires streaming raw data from the server.
7. **Tertile discretisation** of the two continuous behaviours: required by the
   categorical-output specification.

### Check 4: Key statistics comparison
See the table in Step 9. Every statistic the papers report is reproduced: units/probe 891 vs
889, well-isolated units/probe 108.6 vs 108, sessions 441 vs 433-459, subjects 136 vs 139,
Beryl regions 263 vs 270, T = 100, 20 ms bins, choice ~50/50, performance on 100% contrast
> 0.9.

Two small deficits were investigated:
* **263 vs 270 regions.** Expected: Zhang et al. counted regions over all Kilosort units in 433
  sessions, while we count only well-isolated grey-matter units, so rare regions with no
  well-isolated unit drop out. 240 of our regions appear in >=2 sessions, the data paper's own
  reportability criterion.
* **441 vs 433 sessions.** We keep slightly more because Zhang et al. additionally require
  pupil traces in some analyses and their caching script silently skips any session that
  raises. Our 18 exclusions are each explicitly enumerated with a reason.

### Check 5: Edge cases (`cache/edge_cases.py`)
25 structural/edge-case assertions over all 441 sessions, all passing: every session has >=2
trials (min 125) and both choice classes and >=2 prior classes; T == 100 everywhere; neuron
count constant within a session; `brain_region_idx` lengths and all index ranges valid; no
`root`/`void` in `brain_regions`; spike counts non-negative integers with no all-zero trial at
session edges; input 0 equals the bin-edge grid on the first and last trial of every session;
input 1 and outputs 0-1 constant within a trial; trial-in-block in [0, 93]; output ranges
[0,1] and [0,2]; `session_info` aligned with the session list; `trial_idx` strictly
increasing; metadata `off_start`/`off_end`/`time_bin_size` correct.

Robustness handled in the conversion code: sessions with one or two probes; sessions with only
a right camera (7 of 441); sessions with no camera at all (14, skipped); `probabilityLeft`
values outside {0.2, 0.5, 0.8} (hard error, never triggered); trials whose 2 s window is not
covered by the behavioural traces (86 dropped); sessions where every trial fails (1, skipped);
degenerate tertiles (fallback to whatever distinct quantiles exist).

### Iterations
1. **Iteration 1** - first full run: 444 sessions, 38 "all neural data is zero" warnings. Fixed
   with the >=5-neuron and zero-spike-trial rules; re-ran the conversion, the verification and
   all checks. Result: 441 sessions, 0 warnings, 0 failed checks.
2. **Iteration 2** - added `trial_idx` to `session_info` so the conversion can be audited
   against the raw trials table; re-ran the full conversion and every check above.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`
(441 sessions, 187,918 trials, 18.79 M timepoints; L4 GPU, ~25 min)

### Training Progress
- Loss decreasing: **Yes**, monotonically - 1.947 (epoch 1) -> 1.300 (20) -> 0.967 (50) ->
  0.786 (100) -> 0.752 (150) -> 0.7418 (200). Test loss 0.7676.

### Decoder Results (Full)
| Output | Chance (1/n) | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|--------------|-----------------------|-------------------------|--------------|-------|
| choice | 0.5000 | 0.6384 | **0.6153** | 1.23x | per-timestep; rises to 0.72 at +0.3 s (Step 12) |
| prior | 0.3333 | 0.6787 | **0.6604** | 1.98x | |
| wheel_speed | 0.3333 | 0.6174 | **0.6110** | 1.83x | |
| whisker_motion_energy | 0.3333 | 0.5985 | **0.5921** | 1.78x | |

`sample_trials.png` and `predictions.png` were written. The sample-trial figure confirms the
intended structure: the time input ramps linearly from -0.48 to 1.50 s, trial-in-block and
choice/prior are flat within a trial, and both discretised behaviours step up from class 0 to
class 2 around bin 25, i.e. exactly at stimulus onset.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Variable | Chance | Validation | Ratio | Verdict |
|----------|--------|-----------|-------|---------|
| choice | 0.500 | 0.615 | 1.23x | below 1.5x - investigated in depth below |
| prior | 0.333 | 0.660 | 1.98x | fine |
| wheel_speed | 0.333 | 0.611 | 1.83x | fine |
| whisker_motion_energy | 0.333 | 0.592 | 1.78x | fine |

Nothing is at or below chance. For a **binary** variable, 1.5x chance means 0.75 balanced
accuracy at **every one of the 100 time bins**, including the 25 bins before the stimulus even
appears - an impossible bar for a per-timestep decoder, so the ratio was checked against the
time course rather than taken at face value.

`cache/timepoint_analysis.py` retrains the provided decoder on a 60-session subset and scores
each of the 100 bins separately on held-out trials:

| Output | pre-stimulus bins (-0.48 .. 0 s) | post-stimulus bins | peak | peak time |
|--------|----------------------------------|--------------------|------|-----------|
| choice | 0.572 | 0.632 | **0.720** | **+0.30 s** |
| prior | 0.650 | 0.675 | 0.710 | +0.28 s |
| wheel_speed | 0.440 | 0.527 | 0.609 | +0.22 s |
| whisker ME | 0.460 | 0.536 | 0.558 | +0.22 s |

The choice time course is exactly the physiologically expected one: flat at 0.57-0.58 before
the stimulus (IBL choice is partly predictable from the block prior and the previous trial even
before stimulus onset), a sharp rise starting ~0.1 s after onset, a peak of 0.72 at +0.30 s -
the median first-wheel-movement latency - and a slow decay afterwards. A temporal-alignment
error, a shuffled label or a wrong sign convention would all destroy this structure. The
whole-trial average of 0.615 is the arithmetic consequence of averaging informative and
uninformative bins, not a defect.

Two further experiments on the same subset:
* **Is the 20 ms bin the limit?** Summing the 20 ms bins in groups of 5 (100 ms) raises
  per-timestep choice from 0.617 to 0.654 and prior from 0.669 to 0.743. The per-bin
  signal-to-noise of a 20 ms window is the binding constraint, not the conversion. 20 ms was
  kept because it is what the released reference code uses, what the method paper's Results
  state ("2-s trials ... 20-ms bins ... T = 100"), and what the two time-varying outputs need.
  (The STAR Methods mention 50 ms for choice/prior but 20 ms for the dynamic behaviours; a
  single binning has to serve all four outputs here, and only 20 ms is implemented in the
  released code.)
* **Pooling bins into one prediction per trial** (`cache/per_trial_choice.py`, majority vote
  over the post-stimulus bins) gives per-trial balanced accuracy **0.70 for choice** and
  **0.735 for prior** on 5,174 held-out trials.

### Check 2: Accuracy comparison to the reference papers
| Variable | Reference value | Reference protocol | Ours | Comparable? |
|----------|----------------|--------------------|------|-------------|
| choice | AUC **0.869** (good units) / 0.896 (all units), region `all` | per-**trial** label from the whole N x 100 trial matrix, single-session reduced-rank/L1 model with a per-session hyperparameter sweep | per-timestep balanced acc 0.615; per-trial balanced acc **0.70** (~AUC 0.77-0.80) | partly |
| prior | R2 0.182 (good units) / 0.270 | per-trial *continuous* Bayes-optimal prior | 3-class block prior, per-trial balanced acc **0.735** (chance 0.333) | no (different target) |
| wheel speed | R2 ~0.4-0.6 (figures only) | continuous, movement-aligned, 20 ms | 3-class, balanced acc 0.611 (chance 0.333) | no (different target) |
| whisker ME | R2 ~0.5-0.7 (figures only) | continuous, movement-aligned, 20 ms | 3-class, balanced acc 0.592 (chance 0.333) | no (different target) |
| choice (data paper) | per-region null-corrected balanced accuracy, mostly +0.02 .. +0.10 over 0.5 | single region, 100 ms window | +0.115 over chance pooled over all bins, +0.22 at the peak bin | ours is higher, as expected when pooling all regions |

The only directly comparable number is choice. The residual gap (0.70 per-trial balanced
accuracy, ~AUC 0.78, vs the reference's 0.869 AUC with good units) is attributable to the
decoder rather than the data, and its components are measurable:
1. The provided decoder sees **one 20 ms bin at a time** (100 PCs + 2 inputs -> logistic),
   while the reference fits all 100 bins of a trial jointly (N x T, up to 51,600 features).
   The 100 ms experiment above shows ~+0.04 balanced accuracy from 5x more spikes per row
   alone.
2. The reference sweeps learning rate, weight decay and rank per session; the provided decoder
   uses one fixed configuration and one shared output head across all 441 sessions.
3. The reference standardises spike counts per time bin before fitting
   (`standardize_spike_data`); the provided decoder does not, so high-rate neurons dominate
   the SVD-initialised projection.
None of these can be changed by the conversion: the neural data, the alignment and the labels
were all verified against the raw files in Step 10. (An AUC could not be computed directly
from the provided decoder: `predict()` returns only `softmax(logits).max()`, the confidence of
the *predicted* class, not a signed class score.)

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|--------|-------|-----------|-------|
| choice | 0.6384 | 0.6153 | 1.04x |
| prior | 0.6787 | 0.6604 | 1.03x |
| wheel_speed | 0.6174 | 0.6110 | 1.01x |
| whisker ME | 0.5985 | 0.5921 | 1.01x |

All ratios are <=1.04x, far below the 1.5x threshold: no overfitting, no data leakage. (The
split is per session, 80/20 over trials, so leakage would have to be within a trial, and every
input and output is derived only from that trial's own window.)

### Debugging steps applied to the weakest output (choice)
1. **Output values verified against raw data**: `cache/sanity_checks.py` re-derives choice for
   every retained trial of 4 sessions from the raw trials table - exact match; and on
   100%-contrast trials a left stimulus yields the "left" label in 98.5-100% of trials, which
   independently confirms the `choice == +1 => left` sign convention.
2. **Temporal alignment verified**: the population PSTH in `processing_<eid>.png` is flat
   before 0 and jumps from 6.5 Hz to 11.7 Hz peaking at +0.25 s; the per-timepoint choice
   accuracy peaks at +0.30 s; the discretised wheel speed steps up at bin 25 of 100.
3. **Class balance verified**: choice is 0.508/0.492 overall and every session contains both
   classes (per-session left fraction, 5-95% range: 0.35-0.65).
4. **Neural filtering verified**: units/probe (891 vs 889) and well-isolated units/probe
   (108.6 vs 108) reproduce the data paper.
5. **Processing verified against the reference code**: stage-by-stage comparison in Step 10,
   Check 3.

### Issues Found and Resolved
- *38 all-zero-neural warnings* -> minimum of 5 neurons per session and removal of zero-spike
  trials (Step 10, Check 1). Re-ran conversion, verification, training and all checks.
- *No other issue found*: all 96 independent sanity checks and all 25 edge-case assertions
  pass, and the format verification reports no errors and no warnings.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format spec, key statistics)
- [x] `cache/` folder created with the analysis/investigation scripts and `README_CACHE.md`
- [x] All files organised

Deliverables in `/app`:
`CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
`README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
`train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
`train_decoder_full_out.txt`, `processing_<eid>.png` (2), `sample_trials.png`,
`predictions.png`.
