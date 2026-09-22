# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain Wide Map (BWM) public release, served from a local ONE cache at `/app/data/one_cache`
- **Reference papers**: IBL et al. 2025 *A brain-wide map of neural activity during complex behaviour* (Nature) = "datapaper"; Zhang et al. 2026 *Exploiting correlations across trials and behavioral sessions to improve neural decoding* (Neuron) = "methodpaper"
- **Reference code**: `/app/code/code_zhang2025` (Zhang et al. caching + decoding pipeline)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file)
- `Dockerfile`, `docker-compose.yaml`, `stage_cache.sh`, `.manifest` — container/env plumbing
- `code/` — `code_zhang2025` (method-paper code) and `ibllib` (library source copy)
- `data/one_cache/` — the ONE cache (release tables + symlinked lab/Subjects trees)
- `datapaper.pdf`, `methodpaper.pdf`, `dataarchitecture.pdf`, `methods.txt` — references
- `ibl_docs/` — downloaded ONE/ibllib/iblatlas documentation
- `decoder.py`, `train_decoder.py` — provided decoder + validation harness

Environment check: `python3` works; `numpy 2.3.5`, `torch 2.6.0+cu124`, `ONE-api 3.5.2`,
`ibllib 4.0.1`, `iblatlas 1.2.0` all import. Hardware: 128 CPUs, 1006 GB RAM, 3.4 TB disk,
1x NVIDIA L4 (23 GB).

**ONE access**: the cache is used offline through cached Alyx REST responses
(`/app/data/one_cache/.rest`) plus an auth token staged into `$HOME/.one`. The working
recipe is

```python
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          silent=True, cache_dir='/app/data/one_cache')   # -> mode='remote'
```

Using `mode='local'` + `one.load_cache(tag='Brainwidemap')` does **not** work: that
release table was built 2025-02-20 and lists `alf/_ibl_trials.table.pqt`, whereas the
staged files live under the newer revision folder `alf/#2025-03-03#/`. With the local
table, `SessionLoader.load_trials()` silently returns a 1-column frame
(`goCueTrigger_times` only). In `remote` mode (served from the `.rest` cache) all 20
trial columns load correctly. This was verified explicitly before writing any conversion
code.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference pipeline entry point: `/app/code/code_zhang2025/src/0_data_caching.py`.
It sets

```python
params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
          'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']
```

and then calls, in order: `prepare_data` → `list_brain_regions` → `select_brain_regions`
→ `bin_spiking_data` → `bin_behaviors` → `align_spike_behavior`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `src/utils/ibl_data_utils.py` | LOADING | `one.eid2pid(eid)`; loads each probe with `load_spiking_data`; `merge_probes`; `load_trials_and_mask(max_trial_len=10.0)`; `load_anytime_behaviors` |
| `load_spiking_data` | `src/utils/ibl_data_utils.py` | LOADING/CURATION | `SpikeSortingLoader.load_spike_sorting()` + `merge_clusters(...).to_df()`. `qc=None` (default) keeps every cluster; `qc=1` keeps `label >= 1` ("good"/well-isolated) |
| `merge_probes` | `src/utils/ibl_data_utils.py` | LOADING | concatenates probes of one session into a single cluster/spike set, re-indexing `spikes['clusters']`, re-sorting spikes by time |
| `load_trials_and_mask` | `src/utils/ibl_data_utils.py` | CURATION | `SessionLoader.load_trials()`; builds a boolean mask excluding RT<0.08 s, RT>2 s, `feedback_times-goCue_times > max_trial_len`, NaN in {stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType}, and `choice == 0` |
| `list_brain_regions` / `select_brain_regions` | `src/utils/ibl_data_utils.py` | PROCESSING | `BrainRegions().acronym2acronym(..., mapping='Beryl')`; with `single_region=False` returns one group with **all** Beryl regions |
| `bin_spiking_data` | `src/utils/ibl_data_utils.py` | PROCESSING | intervals = `stimOn_times + (-0.5, 1.5)`; per-trial `bincount2D(times, clusters, xbin=0.02, xlim=[t_beg,t_end])`, truncated to `n_bins = ceil(2/0.02) = 100` |
| `get_spike_data_per_interval` | `src/utils/ibl_data_utils.py` | PROCESSING | the per-trial worker for the above (multiprocessing) |
| `load_target_behavior` | `src/utils/ibl_data_utils.py` | LOADING | `SessionLoader.load_wheel()` → `wheel-speed = abs(velocity)`; `SessionLoader.load_motion_energy(views=['left'/'right'])` → `whiskerMotionEnergy` |
| `get_behavior_per_interval` | `src/utils/ibl_data_utils.py` | PROCESSING | slices the behavioural trace to the trial interval and `interp1d`-resamples onto `np.linspace(beg+binsize, end, n_bins)` (i.e. the **right edge of each 20 ms bin**); marks the interval bad if the trace starts >1 bin late or ends >1 bin early, if the interval time is NaN, or (when `allow_nans=False`) if it contains NaNs |
| `bin_behaviors` | `src/utils/ibl_data_utils.py` | PROCESSING | builds `choice`, `block` (=`probabilityLeft`), `reward` (=`rewardVolume>1`), `contrast` per trial, plus the interpolated dynamic traces; called with `allow_nans=True` |
| `align_spike_behavior` | `src/utils/ibl_data_utils.py` | CURATION | drops trials where any behaviour is `None` **and** trials failing `trials_mask` |
| `standardize_spike_data` | `src/utils/data_loader_utils.py` | PROCESSING | per-time-bin z-scoring; belongs to the *model*, not the dataset (their cached HF dataset stores raw spike counts as `np.ubyte`) |

### Notes
- The cached dataset the reference writes contains **raw spike counts** per (trial, time-bin,
  neuron), stored sparsely as unsigned bytes. Standardisation happens at model-fit time.
  Our target format wants neural activity per trial as `(n_neurons, n_timepoints)`, so we
  store the same raw spike counts (as `float32`), transposed.
- `prepare_data` calls `load_spiking_data` **without** `qc`, i.e. it keeps all Kilosort
  clusters including MUA. The data paper instead uses only well-isolated neurons
  (`label >= 1`). See Step 4 for how this discrepancy is resolved.
- `load_spiking_data` also calls `spike_loader.raw_electrophysiology(band="ap", stream=True).fs`
  purely to record the AP sampling rate in metadata. That streams raw data over the network
  and is not needed for binning, so it is not reproduced here.
- The decoding scripts (`1_/2_/3_decode_*.py`, `utils/data_loader_utils.py`) show how the
  cached data is consumed: `choice` as a classification target, `wheel-speed` /
  `whisker-motion-energy` as **time-varying** (length-T) regression targets, `block`/prior
  per trial. `MultiRegionDataModule.list_regions` excludes `root` and `void`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/one_cache/` is a standard ONE cache:
- release tables copied in: `Brainwidemap/` (480 sessions, 76,563 datasets, built 2025-02-20),
  `2022_Q4_IBL_et_al_BWM/` (354 sessions), `2025_Q3_IBL_et_al_BWM/` (459 sessions — a
  supplementary release of re-computed ROI motion energy / lightning-pose)
- twelve lab directories symlinked to the read-only dataset mount, each
  `lab/Subjects/<subject>/<date>/<number>/`. **461 session directories** on disk.
- `.rest/` — 6,483 cached Alyx REST responses, which is what makes `mode='remote'` work offline.

Files staged per session (example `angelakilab/Subjects/NYU-11/2020-02-18/001`):

| Object | Path |
|---|---|
| trials | `alf/#2025-03-03#/_ibl_trials.table.pqt` (+ `stimOff_times`, `stimOnTrigger_times`, …), `alf/_ibl_trials.goCueTrigger_times.npy`, `alf/_ibl_trials.quiescencePeriod.npy`, `alf/_ibl_trials.intervals_bpod.npy` |
| wheel | `alf/_ibl_wheel.position.npy`, `alf/_ibl_wheel.timestamps.npy` |
| camera times | `alf/#2023-04-20#/_ibl_{left,right}Camera.times.npy` |
| whisker motion energy | `alf/#2025-05-29#/leftCamera.ROIMotionEnergy.npy`, `alf/#2025-05-31#/rightCamera.ROIMotionEnergy.npy` |
| spike sorting | `alf/probeNN/pykilosort/#2024-05-06#/spikes.{times,clusters,amps,depths}.npy`, `clusters.{metrics.pqt,channels,depths,amps,uuids,…}`, `channels.*` |

`_ibl_trials.table.pqt` count on disk: 466 (459 at revision `#2025-03-03#`, 5 at
`#2024-07-15#`, 2 unrevisioned) — consistent with the 459-session public release.

No DLC/pose or pupil datasets are staged, which is why pupil diameter (commented out in the
reference caching script as well) is not usable here.

### Dataset Size (from data files / release table `bwm_release.csv`)
| Statistic | Value |
|-----------|-------|
| Probe insertions (pids) | 699 |
| Sessions (eids) | 459 |
| Subjects (mice) | 139 |
| Labs | 12 |
| Sessions / subject | mean 3.30, median 3, range 1–13 |
| Probes / session | 1 (219 sessions) or 2 (240 sessions) |
| Clusters / probe (example probe00 of `6713a4a7…`) | 898 total, 76 with `label >= 1` |
| Trials / session (example) | 565 |

Trials table columns available: `choice, contrastLeft, contrastRight, probabilityLeft,
feedbackType, rewardVolume, stimOn_times, stimOff_times, goCue_times, goCueTrigger_times,
response_times, feedback_times, firstMovement_times, intervals_0, intervals_1,
intervals_bpod_0/1, quiescencePeriod, stimOnTrigger_times, stimOffTrigger_times`.

Wheel: ~1 kHz uniformly resampled `times/position/velocity/acceleration`.
Motion energy: left camera 60 Hz, right camera 150 Hz, single `whiskerMotionEnergy` column.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Units (all, incl. MUA) | 621,733, avg 889/probe | datapaper: "produced 621,733 units … averaging 889 per probe" |
| Well-isolated neurons | 75,708, avg 108/probe | datapaper: "identified 75,708 well-isolated neurons, averaging 108 per probe" |
| Probes | 699 | datapaper |
| Sessions (public release) | 459 | datapaper: "a total of 459 sessions, 699 insertions and 621,733 neurons … constituting the publicly released dataset" |
| Subjects | 139 (94 M, 45 F) | datapaper: "We trained 139 mice" |
| Sessions used by method paper | 433 | methodpaper: "We apply our models to 433 IBL sessions" |
| Brain regions (method paper) | 270 | methodpaper: "covering 270 brain regions" (Beryl) |
| Brain areas (data paper) | 279 | datapaper abstract |
| Trials / session | mean 645, median 602, range 401–1,525 | datapaper: "Recorded sessions lasted on average 645 trials (median of 602, range of 401–1,525)" |
| Fraction correct | 81.4 ± 0.4% | datapaper: "they made correct choices on 81.4 ± 0.4% … of the trials" |
| Fraction correct, 0% contrast | 58.7 ± 0.4% | datapaper |
| First-movement time < 80 ms | 22.8% of trials | datapaper Fig. 1c caption |
| Unbiased block length | first 90 trials, pLeft = 0.5 | datapaper |
| Biased block length | 20–100 trials, truncated exponential, scale 60, empirical mean 51 | datapaper |
| Block probabilities | pLeft ∈ {0.2, 0.5, 0.8} | datapaper |
| Contrasts | {0, 6.25, 12.5, 25, 100}% in ratio 2:2:2:2:1 | datapaper |
| Neural data time bin | 20 ms, T = 100, 2 s trials | methodpaper: "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps" |
| Alignment (choice) | stimulus onset, −0.5 s to +1.5 s | methodpaper Data processing; reference code `params` |
| Behaviour sampling | 60 Hz (left camera), wheel ~1 kHz | methodpaper: "wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz" |

### Processing Details
- **Temporal alignment**: `stimOn_times`, window `(-0.5, +1.5)` s, 20 ms bins → T = 100.
  (The STAR-Methods prose mentions 50-ms bins for the choice/prior *decoders*; the released
  caching code and the model description both use 20 ms / T = 100, and the task here needs
  time-resolved behaviour, so 20 ms is used. See Step 4.)
- **Neural**: raw spike counts per bin, all neurons of a session pooled across probes.
- **Behaviour**: wheel speed = `|velocity|` of the ~1 kHz resampled wheel; whisker motion
  energy from the left camera (right camera as fallback); both linearly interpolated onto
  the right edge of each 20 ms bin.

### Curation Steps

**Neuron curation rules** (datapaper): exclude units failing any RIGOR single-unit metric —
amplitude > 50 µV, noise cut-off < 20 µV, refractory-period violation. Survivors are
"well-isolated neurons" (621,733 → 75,708). In `ibllib` this is exactly the
`clusters['label'] >= 1` flag produced by `SpikeSortingLoader.merge_clusters`.
Analyses further restrict to **grey-matter** regions.

**Trial curation rules** (datapaper, and `load_trials_and_mask` in the reference code):
exclude a trial if any of `choice, probabilityLeft, feedbackType, feedback_times,
stimOn_times, firstMovement_times` is undetected (NaN), or if
`firstMovement_times − stimOn_times ∉ [0.08, 2.00] s`. The reference code additionally
drops `choice == 0` (no response) and trials with `feedback_times − goCue_times > 10 s`.

**Session curation**: release-level criteria (≥250 trials, ≥90% correct on 100% contrast,
≥3 error trials, hardware QC, resolved histology) are already applied to the 459 released
sessions. The method paper ends up with 433 sessions, i.e. ~26 are lost, consistent with
missing/unusable video (whisker motion energy) data.

### Decoders Trained (values reported in the reference papers)
| Decoded variable | Metric | Reported value |
|---|---|---|
| Choice (region-level, method paper Fig. 4/5) | AUC | ~0.66 (linear) → 0.72 (RRR) → 0.79 (combined), example session |
| Prior | Pearson r | 0.05 (single-session) → 0.34 / 0.65 (multi-session / oracle) |
| Wheel speed | R² | 0.24 (linear) → 0.49–0.55 (RRR) |
| Whisker motion energy | R² | 0.52–0.55 → 0.75–0.80 |
| Choice/stimulus/feedback (data paper) | balanced accuracy | region-level effect sizes, small (a few % above 0.5) for most regions |

These are *region-level, single-session* numbers, so they are a lower bound for what a
whole-session decoder should achieve; they are not directly comparable to the
per-timepoint balanced accuracies produced by `/app/train_decoder.py`.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Reference code says | Data shows | Papers say | Resolution |
|-------|--------------------|------------|------------|------------|
| Time bin for choice/prior decoding | `binsize = 0.02` in `0_data_caching.py` | n/a | methodpaper STAR Methods prose: "50-ms non-overlapping time bins" for choice/prior; methodpaper Results + Fig. 1: "2-s trials, each divided into 20-ms bins, producing T = 100" | **20 ms, T = 100.** The released caching code, the model description and the abstract-level description all use 20 ms/T = 100; the 50 ms sentence describes a separate per-trial decoding experiment. 20 ms is also required here because wheel speed and whisker motion energy must be decoded *per timepoint*, and the reference itself uses 20 ms for those. |
| Alignment event | `align_time = 'stimOn_times'`, window `(-0.5, 1.5)` | `trials.stimOn_times` present in all sessions | choice aligned to stimulus onset −0.5…+1.5 s; dynamic behaviours aligned to first movement onset, 0…+1 s | **Stimulus onset, −0.5…+1.5 s**, as the task specification here mandates ("Temporally align based on stimulus onset") and as the reference caching code does for *all* behaviours simultaneously. |
| Neuron quality control | `prepare_data` calls `load_spiking_data(one, pid, …)` with the default `qc=None`, i.e. **all** Kilosort clusters; metadata records `good_clusters = label >= 1` but never filters on it | `clusters['label']` ∈ {0, 1/3, 2/3, 1}; 890 clusters/probe, 108.5 of them with `label >= 1` | datapaper: analyses use only the 75,708 "well-isolated neurons" (RIGOR single-unit metrics); methodpaper: "bin spike counts using all neurons … from each session" | **Keep `label >= 1` (well-isolated neurons).** Rationale: (a) this is the inclusion criterion the *data* paper applies to every analysis of this dataset and the standard `brainwidemap.load_good_units` behaviour; (b) the method paper's phrase "all neurons" contrasts with its *region-restricted* decoders, not with QC; (c) including all 621k multi-unit clusters would make the converted dataset ~10x larger (≈120 GB) for units that the data paper explicitly says are not separable neurons. Documented as a deliberate departure from `0_data_caching.py`. |
| Grey matter | `list_brain_regions` keeps every Beryl acronym incl. `root`/`void`; but `MultiRegionDataModule.list_regions` drops `root` and `void` | ~15% of good clusters map to `root`/`void` | datapaper: "restricted to regions that were designated grey matter in the adult mouse Allen CCF" | **Drop `root`/`void`.** Consistent with the data paper and with the reference code's own region listing. |
| Trial exclusion | `load_trials_and_mask(one, eid, max_trial_len=10.0)` → RT ∈ [0.08, 2] s, no NaN in 6 events, `choice != 0`, `feedback_times − goCue_times <= 10 s` | 34% of raw trials excluded | datapaper lists exactly the 6 NaN events and the 0.08–2.00 s reaction-time window; 22.8% of trials have RT < 80 ms | **Use `load_trials_and_mask` verbatim** (imported from the reference module), including `max_trial_len=10.0`, which is the reference code's only addition. |
| `SessionLoader` call signature | reference calls `SessionLoader(one, eid)` positionally | ibllib 4.0.1 `SessionLoader` is a keyword-only dataclass | — | Pass an explicit `sess_loader=SessionLoader(one=one, eid=eid)` into `load_trials_and_mask`. Pure API adaptation, no change in behaviour. |
| AP sampling frequency | `load_spiking_data` calls `raw_electrophysiology(band='ap', stream=True).fs` | requires streaming raw ephys from the network | — | Skipped: it only fills a metadata field and is never used downstream. |
| NaNs in behaviour | `bin_behaviors(..., allow_nans=True)`; the model later imputes NaNs with the trial average | NaNs do occur in whisker ME and (rarely) wheel | — | Use the reference's `allow_nans=False` branch (reject the trial). The converted outputs are class labels, so there is no sensible "NaN class", and `verify_data_format` errors on NaNs. |
| ONE cache access | reference builds `ONE(..., password='international', cache_dir=...)` (auto mode) | `mode='local'` + `load_cache(tag='Brainwidemap')` cannot see the staged dataset revisions | — | Default (remote) mode against the staged `.rest` response cache and auth token; verified to return all 20 trial columns (Step 0). |
| Session list | reference reads `data/bwm_release.csv` (the BWM freeze file) | 459 eids / 699 pids / 139 subjects on disk | datapaper: 459 sessions, 699 insertions, 139 mice | **Use `bwm_release.csv`**, exactly as the reference does. |

After these resolutions the three sources agree: same session list, same trial mask, same
alignment/binning, same behavioural signals; the only substantive addition is the data
paper's neuron QC + grey-matter restriction, which the method-paper code omits.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (ONE object.attribute) | Target field | Transform | Reference code function(s) | Notes |
|---|---|---|---|---|
| `spikes.times`, `spikes.clusters` (all probes) + `clusters.label`, `clusters.acronym` | `neural[session][trial]` (n_neurons, 100) float32 | pool probes → keep `label>=1` & Beryl ∉ {root, void} → count spikes in 100 × 20 ms bins over `stimOn_times + (-0.5, 1.5)` | `prepare_data`, `load_spiking_data`, `merge_probes`, `list_brain_regions`, `select_brain_regions`, `bin_spiking_data` | spike **counts**, not rates, matching the reference cached dataset |
| bin index | `input[…][0]` | signed bin-centre time, −0.49 … +1.49 s | — (task spec: "time since stimulus onset, continuous, time-varying") | constant grid, identical for every trial |
| `trials.probabilityLeft` | `input[…][1]` | 0-based index of the trial within its run of constant `probabilityLeft`, computed on the **unfiltered** trials table, broadcast over the 100 bins | — (task spec: "trial number in block, continuous, per-trial") | block structure must come from all trials, not the filtered subset |
| `trials.choice` | `output[…][0]` | `+1 → 0` (left), `−1 → 1` (right); `0` (no response) already excluded | `bin_behaviors` (`choice`) | direction verified empirically, see sanity checks |
| `trials.probabilityLeft` | `output[…][1]` | `0.2 → 0`, `0.5 → 1`, `0.8 → 2` | `bin_behaviors` (`block`) | exactly as required by the task spec |
| `wheel.velocity` (SessionLoader, ~1 kHz resampled) | `output[…][2]` | `abs(velocity)` → interpolate onto the right edge of each 20 ms bin → per-session tertiles → {0,1,2} | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | time-varying |
| `leftCamera.ROIMotionEnergy` (`whiskerMotionEnergy`, 60 Hz; right camera fallback) | `output[…][3]` | interpolate onto bin right edges → per-session tertiles → {0,1,2} | `load_target_behavior('left-whisker-motion-energy')`, fallback logic in `bin_behaviors` | time-varying |
| `clusters.acronym` → Beryl | `brain_regions`, `brain_region_idx` | `BrainRegions().acronym2acronym(..., mapping='Beryl')` | `list_brain_regions` | |
| `bwm_release.csv` `subject` | `subjects`, `subject_idx` | unique sorted subject names | `0_data_caching.py` | |

### Key Decisions
1. **Alignment = `stimOn_times`, window (−0.5, +1.5) s, 20 ms bins (T = 100).** Exactly the
   reference `params`; also what the task specification requires.
2. **Spike counts, unnormalised.** The reference caches raw counts (`np.ubyte`) and
   z-scores only inside the model (`standardize_spike_data`). `train_decoder.py` does its
   own SVD-based projection, so raw counts are the right thing to store.
3. **Neuron QC: `label >= 1` + grey matter.** See Step 4. Sanity check: this must give
   ≈108 neurons per probe and ≈62–63k neurons in total.
4. **Trial QC: reference `load_trials_and_mask(max_trial_len=10.0)`**, imported directly
   from the reference module so the logic cannot drift.
5. **Behaviour-coverage QC.** A trial is dropped when the wheel or whisker trace does not
   cover the whole 2 s window (the reference's "target data starts too late / ends too
   early" tests) or contains NaN.
6. **Ephys-coverage QC (addition).** A trial is dropped when its window intersects a
   >0.5 s gap in the *pooled unfiltered* spike train or falls outside the recorded
   interval. Necessary because some sessions keep running the behavioural rig after the
   ephys stops; those trials would otherwise be all-zero.
7. **Session QC: ≥5 well-isolated grey-matter neurons** (the data paper's per-session
   neuron minimum), a usable whisker trace, and ≥2 usable trials.
8. **Per-session tertiles for the two continuous outputs.** Both signals are in
   session-specific units (motion energy is an uncalibrated pixel difference that depends
   on camera resolution/frame rate, illumination and ROI placement; wheel speed depends on
   how vigorously the individual mouse turns). Per-session tertiles make chance exactly
   1/3 in every session and give the *shared* decoder a comparable target everywhere. A
   single global threshold would push whole sessions into one class.
9. **Both inputs and all four outputs are stored time-varying (d, 100)**, per-trial
   variables being broadcast along time, as the format instructions prefer.
10. **Session order** follows `bwm_release.csv`, so the conversion is deterministic.

### Planned Sanity Checks
- [x] all clusters/probe ≈ 889 (datapaper)
- [x] well-isolated clusters/probe ≈ 108 (datapaper)
- [x] raw trials/session: mean ≈ 645, median ≈ 602, range 401–1525 (datapaper)
- [x] fraction correct ≈ 0.814 (datapaper)
- [x] 459 eids / 699 pids / 139 subjects in the freeze file (datapaper)
- [x] number of retained sessions in the ballpark of the method paper's 433
- [x] number of Beryl regions in the ballpark of the method paper's 270
- [x] total well-isolated grey-matter neurons ≈ the data paper's canonical 62,857
- [x] `probabilityLeft` only ever 0.2 / 0.5 / 0.8, first block = 90 trials at 0.5
- [x] population PSTH shows a stimulus-locked transient starting at t = 0
- [x] wheel speed rises at `firstMovement_times` when trials are sorted by reaction time
- [x] tertile classes each hold ≈1/3 of timepoints
- [x] element-wise re-derivation of neural/input/output from the raw data (Step 10)

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`; runs as
`python -u /app/convert_data.py <out.pkl> [--full|--sample] [--show-processing] [--n-workers N]`.

Structure:
- `get_one()` / `get_brain_regions()` — per-process singletons.
- `load_session_spikes` — brainbox `SpikeSortingLoader` per probe + reference `merge_probes`.
- `select_neurons` — `label >= 1` and Beryl ∉ {root, void}.
- `bin_spiking_data_fast` — vectorised replacement for the reference
  `bin_spiking_data`/`get_spike_data_per_interval`, bin-for-bin identical.
- `neural_data_available` — ephys-coverage mask.
- `load_dynamic_behaviour` — reference `load_target_behavior` logic (wheel speed, whisker ME).
- `bin_behaviour_per_trial` — faithful re-implementation of `get_behavior_per_interval`.
- `discretize_tertiles`, `trial_number_in_block`.
- `convert_session` — the per-session pipeline; `_worker` wraps it so a single bad session
  cannot kill the run.
- `plot_processing` — the `--show-processing` figure.
- `main` — `ProcessPoolExecutor` fan-out over sessions, assembly, summary, pickle.

The reference `load_trials_and_mask` and `merge_probes` are **imported** from
`/app/code/code_zhang2025/src/utils/ibl_data_utils.py` rather than re-implemented.

Code inefficiencies identified in the reference:
- `get_spike_data_per_interval` spawns a multiprocessing pool and calls `bincount2D`
  once per trial (≈600 pool tasks per session), each of which re-derives a cluster
  index with `np.intersect1d`.
- `get_behavior_per_interval` likewise uses a pool per behaviour per session.
- `load_anytime_behaviors` loads six behavioural streams, four of which
  (wheel velocity, right whisker ME, both pupil diameters) the caching script never uses.
- `load_spiking_data` streams raw AP data just to read a sampling rate.

Code speedups added:
- Spike binning vectorised with `searchsorted` + a single `np.bincount` per trial:
  ~0.02 s per session instead of seconds of pool overhead.
- Spikes are binned **only for the trials that survive curation**.
- Parallelism moved up to the session level (`ProcessPoolExecutor`, 32 workers), which is
  where the real cost (loading 200–500 MB `spikes.*.npy` per probe) lies.
- Only the two behaviours actually needed are loaded; no raw-ephys streaming.
- Spike counts are carried as `uint8` from the workers (the reference stores `np.ubyte`
  too) and expanded to `float32` once, in the parent.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`/app/processing_<eid>.png` (2 files).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 253 |
| Neurons / session | 61, 192 (mean 126.5) |
| Subjects | 1 (NYU-11) |
| Sessions / subject | 2 |
| Trials (total) | 651 |
| Trials / session | 407, 244 |
| All clusters / probe | 875.3 (datapaper: 889) |
| Good clusters / probe | 113.3 (datapaper: 108) |
| Fraction correct | 0.854 |
| `time_from_stimulus_onset` range | [−0.49, 1.49] |
| `trial_number_in_block` range | [0, 89] |
| choice distribution | left 0.518, right 0.482 |
| prior distribution | 0.2 → 0.478, 0.5 → 0.161, 0.8 → 0.361 |
| wheel speed distribution | 0.333 / 0.333 / 0.333 |
| whisker ME distribution | 0.333 / 0.333 / 0.333 |

### Processing Plots Review
`processing_6713a4a7-….png` (7 rows of panels) shows:
1. Population PSTH: flat baseline before 0, sharp rise **exactly at t = 0**, peak at
   ~0.25 s — the alignment to stimulus onset is correct.
2. Raw spike raster vs binned matrix for the same trial, plus per-bin totals computed the
   two ways overlaying exactly.
3. Wheel speed: the resampled points sit on the raw ~1 kHz trace; the two tertile
   thresholds bracket it sensibly; the pre-stimulus quiescence period is flat and falls in
   the "low" class.
4. Whisker motion energy: resampled trace tracks the 60 Hz camera signal.
5. Class-balance bar chart: all six classes at 1/3.
6. Inputs: time input is a straight ramp through 0 at stimulus onset; trial-in-block is
   constant within the trial.
7. Block structure: `trial_number_in_block` resets exactly where `probabilityLeft`
   changes, the first block is 90 trials long at pLeft = 0.5; prior and choice outputs
   overlay the trials table exactly; wheel-speed heat map sorted by reaction time shows
   movement onset following the plotted `firstMovement_times` curve — i.e. neural and
   behavioural streams are aligned to the same clock.

No anomalies found.

### Run Time Estimates
| Speed-ups implemented | Time saving |
|---|---|
| vectorised spike binning (vs pool + per-trial `bincount2D`) | ~5–10 s/session |
| binning only retained trials | ~35% of binning work |
| session-level `ProcessPoolExecutor` (32 workers) | ~20x wall clock |
| loading 2 behaviours instead of 6, no raw-ephys streaming | ~10 s/session |

| Step | Time / session | Estimated total (441 sessions) |
|---|---|---|
| `eid2pid` | 0.02 s | 9 s |
| spike sorting load (dominant; 1–2 probes) | 3–30 s | ~2.0 h serial |
| trials load | 0.3 s | 2 min |
| behaviour load | 0.5 s | 4 min |
| binning (spikes + behaviour) | 0.3 s | 2 min |
| **serial total** | ~6–30 s | ~2.2 h |
| **with 32 workers** | — | **~3.5 min** (measured: 203 s including the 19 s pickle write) |

Sample verification (`/app/verification_sample_out.txt`): **"Data format is valid, no
errors or warnings."**

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions / 651 trials / 61–192 neurons)
| Output | Training balanced acc | Validation balanced acc | Chance |
|--------|------|------|------|
| choice | 0.612 | 0.531 | 0.500 |
| prior_probability_left | 0.693 | 0.652 | 0.333 |
| wheel_speed | 0.568 | 0.558 | 0.333 |
| whisker_motion_energy | 0.558 | 0.562 | 0.333 |

Loss fell monotonically from 1.44 to 0.768 over 200 epochs (held-out test loss 0.799);
every output is above chance.
Choice is the weakest, as expected for two sessions whose neurons sit mostly in BMA, CEA,
LGd and VPM rather than in the motor/decision areas where choice decodes best.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --n-workers 32`
→ `/app/conversion_full_out.txt` (203 s wall clock).

### Output Files
- `converted_data.pkl`: 11.72 GB
- `verification_full_out.txt`: created (`--verify-only`)

### Sessions not converted (18 of 459)
| Reason | Count |
|---|---|
| no whisker motion energy dataset available (neither camera) | 14 |
| fewer than 5 well-isolated grey-matter neurons | 3 |
| 0 usable trials after curation | 1 |

The method paper reports 433 usable sessions; 441 here. The difference is the same order
as the sessions each pipeline loses to missing video, and we additionally keep a handful
of sessions that the reference may have dropped for pupil/DLC reasons.

### Consistency Check
| Statistic | Reference papers | Reference code | Reference data (`bwm_release.csv` / ONE) | Converted data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Probe insertions | 699 | 699 rows | 699 | 673 used | ✓ (26 in the 18 dropped sessions) |
| Sessions in release | 459 | 459 eids | 459 | 441 converted | ✓ |
| Sessions used for decoding | 433 (methodpaper) | — | — | 441 | ✓ same ballpark |
| Subjects | 139 | 139 | 139 | 136 | ✓ (3 lost with their only session) |
| Sessions / subject | — | 3.30 mean | 3.30 | 3.24 | ✓ |
| Units per probe (all clusters) | 889 | — | 890.1 | 890.1 | ✓ |
| Well-isolated units per probe | 108 | — | 108.5 | 108.5 | ✓ |
| Well-isolated units, total | 75,708 (459 sessions) | — | 73,010 in the 441 converted sessions | — | ✓ (75,708 × 673/699 = 72,892 expected) |
| Neurons after grey-matter restriction | 62,857 ("canonical dataset") | — | — | **62,757** | ✓ |
| Brain regions (Beryl) | 270 (methodpaper) / 279 areas (datapaper) | — | — | 263 | ✓ |
| Regions with ≥20 neurons | 201 canonical regions | — | — | 210 | ✓ |
| Raw trials/session | mean 645, median 602, range 401–1525 | — | — | mean 646.3, median 601, range 401–1525 | ✓ |
| Trials retained after curation | ~66% expected (22.8% RT<80 ms + RT>2 s + NaN + no-choice) | — | — | 66.0% pass the mask; 426.1/session kept | ✓ |
| Trials (total) | — | — | — | 187,901 | — |
| Fraction correct | 81.4 ± 0.4% | — | — | 81.6% | ✓ |
| T (timepoints) | 100 | 100 | — | 100 | ✓ |
| Bin size | 20 ms | 0.02 s | — | 20 ms | ✓ |
| pLeft values | {0.2, 0.5, 0.8} | — | {0.2, 0.5, 0.8} | {0, 1, 2} | ✓ |
| Unbiased block | first 90 trials | — | 90 | 90 | ✓ |
| `time_from_stimulus_onset` range | −0.5…1.5 s | (−0.5, 1.5) | — | [−0.49, 1.49] (bin centres) | ✓ |
| `trial_number_in_block` range | blocks 20–100 trials (+90 unbiased) | — | — | [0, 98] | ✓ |
| choice distribution | ~balanced | — | — | left 0.508 / right 0.492 | ✓ |
| prior distribution | 90 unbiased of ~645 ⇒ ~14% pLeft = 0.5, rest split | — | — | 0.417 / 0.140 / 0.442 | ✓ |
| wheel speed distribution | — | — | — | 0.333 / 0.333 / 0.333 | ✓ by construction |
| whisker ME distribution | — | — | — | 0.333 / 0.333 / 0.335 | ✓ by construction |
| Mean firing rate | — | — | — | 11.4 sp/s per well-isolated neuron | plausible |
| Camera used for whisker ME | left (60 Hz) preferred | left, right fallback | — | left 434, right 7 | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification (`/app/verification_full_out.txt`)
- **Errors: none.** `verify_data_format` returns valid.
- **Warnings: 12**, all of the form
  `Session 360, trial N: all neural data is zero`, for session index 360
  (eid `195443eb-08e9-4a18-a7e1-d105b2ce1429`).

  Investigated and **not fixable without discarding valid data**: that session has only
  10 well-isolated grey-matter neurons, eight of which fire at ~1.4 sp/s (firing rates
  0.48, 1.25, 1.28, 1.37, 1.40, 1.43, 1.44, 1.46, 5.22, 13.0 sp/s). I checked the
  *pooled, unfiltered* spike train in exactly those 12 trial windows: it contains
  1,501–2,742 spikes, i.e. the electrophysiology is present and healthy — the recording is
  not missing, the ten well-isolated units simply happen to be silent for those 2 s
  (12 of 618 trials, 1.9%, all in the first quarter of the session where these units are
  least active). Dropping them would throw away genuine, correctly-converted data, so they
  are kept.

  The three *other* causes of all-zero trials found in the first full run **were** real
  defects and were fixed (see "Issues found and resolved" below), which removed 26 of the
  original 38 warnings, and the earlier 1-neuron session that produced most of them.

### Check 2 — Independent sanity checks (`/app/cache/sanity_checks.py`)
This script never imports `convert_data.py`. It reloads every quantity from the original
data through `ONE`, `SessionLoader`, `SpikeSortingLoader` and `BrainRegions`, and compares
with `np.allclose()`. Four randomly chosen sessions (indices 118, 224, 279, 372 —
subjects NYU-21, PL031, ZFM-01576, NR_0031; 16–370 neurons; 243–586 trials).

**Result: 64/64 checks passed.** Per session:

*Neural*
- `n_neurons` equals an independently recomputed `label >= 1` + grey-matter selection.
- `brain_region_idx` decoded back to acronyms equals the Beryl mapping of those clusters.
- For 5 random trials per session, the **entire** (n_neurons × 100) count matrix is
  rebuilt from raw `spikes.times`/`spikes.clusters` and matches exactly.
- A single-element spot check, e.g. *S279: neural[trial 493, neuron 177, bin 42] == 2*,
  computed by counting spikes of one cluster in one 20 ms window.

*Input*
- `input[0]` equals the bin-centre grid `−0.49 … 1.49`.
- `input[1]` equals the trial-in-block index recomputed from `trials.probabilityLeft`, and
  is constant within each trial.

*Output*
- `output[0]` equals `(trials.choice < 0)`; and, independently, on every one of the four
  sessions `choice == +1` occurs **only** on correct left-stimulus trials and
  `choice == −1` **only** on correct right-stimulus trials, confirming the
  left = 0 / right = 1 mapping.
- `output[1]` equals the `probabilityLeft → {0,1,2}` recode.
- `output[2]`/`output[3]` are reproduced bin-for-bin by re-slicing the raw wheel /
  motion-energy traces, re-running `interp1d(..., fill_value='extrapolate')` onto
  `linspace(beg+0.02, end, 100)` and re-applying the recorded per-session thresholds; the
  thresholds themselves are reproduced to `rtol=1e-6` by recomputing the 1/3 and 2/3
  quantiles.
- The number of trials kept is reproduced exactly by re-deriving the whole curation chain
  (trial mask, behaviour coverage, ephys coverage) from scratch.

### Check 3 — Reference code comparison
| Stage | Reference (`code_zhang2025`) | This conversion | Same? |
|---|---|---|---|
| (a) session list | `pd.read_csv('data/bwm_release.csv')` | identical file, `pd.unique(bwm_df.eid)` | **yes** |
| (a) probe→session | `one.eid2pid(eid)`, loop over pids | identical | **yes** |
| (a) spike loading | `SpikeSortingLoader.load_spike_sorting()` + `merge_clusters(...).to_df()` | identical | **yes** |
| (a) probe merge | `merge_probes(spikes_list, clusters_list)` | **the same function, imported** | **yes** |
| (a) trials | `load_trials_and_mask(one, eid, max_trial_len=10.0)` | **the same function, imported**, with an explicit `sess_loader` (ibllib 4 API) | **yes** |
| (a) behaviour | `SessionLoader.load_wheel()` → `abs(velocity)`; `load_motion_energy(views=['left'])` → `whiskerMotionEnergy`, right-camera fallback | identical | **yes** |
| (b) neuron filtering | none (`qc=None`) | `label >= 1` and Beryl ∉ {root, void} | **no — deliberate**, see Step 4. Matches the *data* paper. |
| (b) trial filtering | `align_spike_behavior`: trials mask ∧ behaviour present | same, plus the ephys-coverage test and NaN rejection | **superset**, both additions documented |
| (b) session filtering | `try/except` around each session | same, plus ≥5 neurons / ≥2 trials | **superset** |
| (c) alignment | `intervals = trials[align_time] + time_window`, `align_time='stimOn_times'`, `time_window=(-0.5,1.5)` | identical | **yes** |
| (d) spike binning | `bincount2D(t, c, xbin=0.02, xlim=[beg,end])[:, :100]` | `floor((t-beg)/0.02)`, keep `0 <= b < 100`, `np.bincount` | **yes — bin-for-bin identical**, verified in Check 2 |
| (d) behaviour binning | slice to (beg, end), reject if data starts >1 bin late / ends >1 bin early, `interp1d(..., 'extrapolate')` onto `linspace(beg+bin, end, 100)` | identical, with `allow_nans=False` | **yes** |
| (e) inputs | reference has no decoder "inputs" (its models take X only) | time-from-onset + trial-in-block, per this task's spec | n/a |
| (f) outputs | `choice` (raw ±1), `block` (raw pLeft), `wheel-speed` / `whisker-motion-energy` (raw continuous, z-scored at fit time) | same four variables, recoded to the categorical encoding this task's spec demands | **yes** (same source variables, required recoding) |
| storage dtype | `csr_array(..., dtype=np.ubyte)` raw counts | `float32` raw counts | equivalent |
| normalisation | `standardize_spike_data` applied inside the model, not the dataset | none in the dataset; `train_decoder.py` does its own projection | **yes** |

### Check 4 — Key statistics comparison
See the table in Step 9. Every statistic that the papers report is matched:
clusters/probe 890.1 vs 889; well-isolated/probe 108.5 vs 108; raw trials/session
646.3 / 601 / 401–1525 vs 645 / 602 / 401–1525; fraction correct 0.816 vs 0.814;
grey-matter well-isolated neurons 62,757 vs the canonical 62,857; 441 sessions vs 433;
263 Beryl regions vs 270; 136 of 139 subjects. No discrepancy needed further work.

### Check 5 — Edge cases
- **Trial 0 / last trial.** `trial_number_in_block` starts at 0 for the first trial
  (verified: the first block is exactly 90 trials of pLeft = 0.5, matching the task
  design). The last trials of a session are dropped when the ephys ends first
  (`8c2f7f4d…`: 3 such trials).
- **NaN `stimOn_times`** — the window becomes NaN; guarded in `bin_spiking_data_fast`,
  `neural_data_available` and `bin_behaviour_per_trial`, and excluded by the trials mask.
- **Bin edges.** A spike landing exactly on `t_end` maps to bin index 100 and is
  discarded, reproducing the reference's `[:, :n_bins]` truncation. Spike counts are
  clipped at 255 before the `uint8` cast (never reached in practice: the maximum count
  in a 20 ms bin over the whole dataset is far below that, since the reference itself
  stores `np.ubyte`).
- **Sessions with one probe vs two** — both go through `merge_probes`, so cluster
  re-indexing is identical in both cases.
- **Sessions with no left camera** — the right-camera fallback is used (7 sessions);
  14 sessions have neither and are dropped with an explicit message.
- **Degenerate tertiles** — `np.searchsorted(thresholds, v, side='right')` still yields
  valid labels if the two thresholds coincide; in practice all 441 sessions came out at
  0.333/0.333/0.333–0.335.
- **Worker failures** — `_worker` catches any exception and reports it, so one bad
  session cannot abort the run; all 18 failures are listed in
  `metadata['failed_sessions']`.

### Issues found and resolved (iteration log)
1. **`mode='local'` silently loaded an empty trials table.** Found in Step 0 before any
   conversion code existed; fixed by using the default remote mode against the staged
   REST cache. Re-checked: all 20 trial columns load.
2. **`SessionLoader(one, eid)` positional call in the reference raises under ibllib 4.0.1.**
   Fixed by passing `sess_loader=` explicitly.
3. **38 "all neural data is zero" warnings in the first full run.** Root-caused to three
   different things: (i) a 5 s dropout in the spike train of `b182b754…`; (ii) the ephys
   of `8c2f7f4d…` stopping 12 s before the behaviour did, leaving 3 trials with no neural
   data; (iii) a session with a single well-isolated neuron. Fixed by adding
   `neural_data_available` (ephys-coverage test, 33 trials dropped dataset-wide) and the
   ≥5-neuron session criterion. Re-ran the full conversion and all checks: warnings went
   from 38 to 12, and the remaining 12 were verified to be genuine sparse firing
   (Check 1).
4. After the fix the full pipeline was re-run end to end (conversion → `--verify-only` →
   sanity checks → decoder), and all of Checks 1–5 were repeated on the new file.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `/app/sample_trials.png`, `/app/predictions.png`.
Ran on the GPU (NVIDIA L4); ~9 min for 200 epochs over 441 sessions / 187,901 trials /
18.8 M timepoints.

### Training Progress
- Loss decreasing: **yes**, monotonically every epoch:
  1.47 (epoch 1) → 1.209 (10) → 1.062 (20) → 0.899 (40) → 0.821 (60) → 0.765 (100) →
  0.7378 (200). Held-out test loss 0.7649, close to the final training loss (0.7378),
  i.e. almost no overfitting.

### Decoder Results (Full)
| Output | #classes | Chance | Training balanced acc | Validation balanced acc | Val / chance | Notes |
|--------|---|---|-------------|--------|---|-------|
| choice | 2 | 0.500 | 0.6391 | **0.6166** | 1.23x | brain-wide, per timepoint, incl. 0.5 s of pre-stimulus baseline where choice is by construction undecodable |
| prior_probability_left | 3 | 0.333 | 0.6829 | **0.6630** | 1.99x | |
| wheel_speed | 3 | 0.333 | 0.6166 | **0.6092** | 1.83x | |
| whisker_motion_energy | 3 | 0.333 | 0.5999 | **0.5927** | 1.78x | |

`predictions.png` shows the decoder's per-timepoint predictions overlaying the true
wheel-speed and whisker-ME class traces closely, including the rise from "low" before the
stimulus to "medium/high" after movement onset.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance
| Variable | Validation balanced acc | Chance (1/K) | Ratio | Verdict |
|---|---|---|---|---|
| choice | 0.6166 | 0.5000 | 1.23x | above chance; see the analysis below for why 1.5x is not attainable for a binary variable here |
| prior_probability_left | 0.6630 | 0.3333 | 1.99x | well above chance |
| wheel_speed | 0.6092 | 0.3333 | 1.83x | well above chance |
| whisker_motion_energy | 0.5927 | 0.3333 | 1.78x | well above chance |

No output is at or below chance, so there is no sign of an inverted label, a shuffled
trial order or a temporal misalignment.

**Investigating choice (the only output below 1.5x chance).** 1.5x chance for a binary
variable means 0.75 balanced accuracy at *every* timepoint of the window, which the
biology rules out: the window starts 0.5 s *before* the stimulus, and the mouse has not
yet decided. To quantify this I fitted an independent, 5-fold cross-validated logistic
regression at each of the 100 time bins on 20 per-session PCs, for 30 randomly chosen
sessions with >=100 neurons and >=300 trials (`/app/cache/timecourse.py`,
`/app/cache/decodability_timecourse.png`):

| Variable | chance | mean over window | pre-stimulus (t<0) | post-stimulus (t>=0) | peak (time) |
|---|---|---|---|---|---|
| choice | 0.500 | 0.587 | **0.525** | 0.608 | **0.751 at +0.25 s** |
| prior_probability_left | 0.333 | 0.580 | 0.559 | 0.587 | 0.635 at +0.23 s |
| wheel_speed | 0.333 | 0.528 | 0.540 | 0.525 | 0.607 at +0.19 s |
| whisker_motion_energy | 0.333 | 0.533 | 0.477 | 0.552 | 0.574 at +0.59 s |

Two things follow. (1) Choice decodability is flat at ~0.52 for the whole pre-stimulus
half-second, jumps **exactly at t = 0**, and peaks at 0.75 around the mean first-movement
time — an independent confirmation that the neural and trial-event streams are aligned to
the same clock, and a demonstration that the whole-window average is diluted by ~25 bins
in which the variable genuinely carries no information. (2) The provided decoder
(0.617) *beats* this per-session per-timepoint benchmark (0.587) on choice, and beats it
on every other output too (0.663 vs 0.580; 0.609 vs 0.528; 0.593 vs 0.533) — exactly what
a model sharing structure across 441 sessions should do. If the conversion were scrambling
anything, the shared decoder would be worse than, not better than, the per-session
baseline.

### Check 2 — Comparison with accuracies reported in the papers
The papers never report the quantity `train_decoder.py` reports (per-timepoint balanced
accuracy of a shared decoder over a -0.5..1.5 s window), so the comparison has to be made
against the nearest published quantity.

| Variable | Paper value | What it measures | This conversion | Assessment |
|---|---|---|---|---|
| choice | AUC 0.66 (L2 linear baseline), 0.72 (RRR), 0.79 (combined) — methodpaper Fig. 5A, best regions, 10 sessions | **whole-trial** decoding from 2 s of activity, single region | per-timepoint balanced acc 0.617 over the whole window; **0.751 peak** in my per-timepoint benchmark | consistent: the peak-time, whole-population number lands between the paper's linear baseline and its best model |
| choice (data paper) | region-level effect sizes of a few % above 0.5 balanced accuracy (Fig. 4) | per-region, whole-trial, null-corrected | 0.617 brain-wide | higher, as expected when all of a session's neurons are pooled |
| prior | Pearson r 0.05 (single-session linear) / 0.34 (multi-session) / 0.65 (oracle) — methodpaper Fig. 4 | continuous prior regression | balanced acc 0.663 vs chance 0.333 for the 3-class block variable | not directly comparable (different target and metric); the discretized block variable is easier than the continuous prior |
| wheel speed | R^2 0.24 (linear) → 0.49-0.55 (RRR) — methodpaper Fig. 5C | continuous regression, aligned to **first movement**, 0..1 s | balanced acc 0.609 vs chance 0.333 (3 tertile classes) | consistent in spirit: an R^2 of ~0.5 on a continuous signal corresponds to comfortably-above-chance tertile classification |
| whisker motion energy | R^2 0.52-0.55 → 0.75-0.80 — methodpaper Fig. 5D | continuous regression, aligned to first movement, 0..1 s | balanced acc 0.593 vs chance 0.333 | same |

Two structural differences make the paper's dynamic-behaviour numbers an upper bound that
this dataset cannot match, and both are forced by the task specification here:
1. The paper aligns wheel speed / whisker ME to **first movement onset** over 0..1 s — the
   window in which those signals are largest and most stereotyped. This task requires
   alignment to **stimulus onset** over -0.5..1.5 s, which includes the quiescence period.
2. The paper's dynamic decoders are **causal filters over W = 10 preceding bins**
   (data paper) or full reduced-rank models over the whole 2 s (method paper), whereas
   `train_decoder.py` predicts each bin from that bin's activity alone.
Neither difference is a conversion defect.

### Check 3 — Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| choice | 0.6391 | 0.6166 | 1.04 |
| prior_probability_left | 0.6829 | 0.6630 | 1.03 |
| wheel_speed | 0.6166 | 0.6092 | 1.01 |
| whisker_motion_energy | 0.5999 | 0.5927 | 1.01 |

All ratios are far below the 1.5x flag; training loss (0.7378) and held-out test loss
(0.7649) nearly coincide. There is no overfitting and, equally, no sign of leakage (a
leak would show as validation accuracy *matching* training accuracy at an implausibly high
level, which is not the case).

### Additional debugging steps run (from the checklist)
1. **Output values verified against raw data on specific trials** — Step 10 Check 2:
   `output[0]`, `output[1]` re-derived from the trials table and `output[2]`, `output[3]`
   re-derived by re-interpolating the raw wheel / camera traces, for 4 sessions and every
   trial in them; all `np.allclose`.
2. **Temporal alignment plotted for single trials** — `processing_<eid>.png` panels
   (PSTH rising at t = 0; wheel-speed heat map sorted by reaction time tracking
   `firstMovement_times`), plus the per-timepoint decodability time course above.
3. **Output variation** — no output is dominated by one class: choice 0.508/0.492,
   prior 0.417/0.140/0.442, wheel and whisker 0.333/0.333/0.333.
4. **Neural filtering verified** — 108.5 well-isolated clusters per probe against the data
   paper's 108; 62,757 grey-matter neurons against its canonical 62,857.
5. **Processing matched to the reference** — Step 10 Check 3 table; the spike-binning and
   behaviour-interpolation code was verified to reproduce the reference conventions
   bin-for-bin.

### Issues found and resolved
None in this round: all four outputs are above chance, the train/validation gap is
negligible, and the only sub-1.5x-chance output (choice) was traced to the pre-stimulus
part of the mandated window rather than to a defect. No change to `convert_data.py` was
needed, so Steps 9-11 did not have to be re-run.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format spec, key statistics)
- [x] `cache/` folder created, with `README_CACHE.md` describing every file in it
- [x] Investigation/analysis scripts moved to `/app/cache/`
- [x] All required deliverables present:
  `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
  `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
  `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
  `train_decoder_full_out.txt`, plus `processing_<eid>.png` x2, `sample_trials.png`,
  `predictions.png`.
