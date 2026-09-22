# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-Wide Map (BWM) public Neuropixels release — mice performing the
  IBL decision-making task. Staged as a ONE cache at `/app/data/one_cache`.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`, `stage_cache.sh` — container plumbing
- `code/code_zhang2025/` — reference code for the methods paper (Zhang et al. 2026, *Neuron*)
- `code/ibllib/` — IBL library source (reference for loaders)
- `data/one_cache/` — ONE cache: release tables (`Brainwidemap`, `2022_Q4_IBL_et_al_BWM`,
  `2025_Q3_IBL_et_al_BWM`), `.rest` REST cache, and 12 lab directories symlinked to the
  read-only dataset mount
- `datapaper.pdf`, `methodpaper.pdf`, `dataarchitecture.pdf`, `methods.txt` — references
- `decoder.py`, `train_decoder.py` — decoder to validate against

Environment verified: `python3` with numpy 2.3.5, torch 2.6.0+cu124 (CUDA, NVIDIA L4 23 GB),
pandas 3.0.5, scipy 1.18.0, ONE-api 3.5.2, ibllib 4.0.1, iblatlas 1.2.0.
Host: 128 CPUs, 1006 GB RAM, 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference repo: `/app/code/code_zhang2025` (Zhang et al., "Exploiting correlations across
trials and behavioral sessions to improve neural decoding"). The data-preparation entry
point is `src/0_data_caching.py`; all loading/processing lives in
`src/utils/ibl_data_utils.py`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `src/utils/ibl_data_utils.py` | LOADING | Per-eid driver: loads every probe's spike sorting, merges probes, loads trials + mask, loads continuous behaviours |
| `load_spiking_data` | `ibl_data_utils.py` | LOADING / CURATION | `SpikeSortingLoader.load_spike_sorting()` + `merge_clusters()`; optional `qc` threshold on `clusters['label']` (docstring: "If use good clusters, set qc to 1") |
| `merge_probes` | `ibl_data_utils.py` | LOADING | Concatenates spikes/clusters of the probes of one session, re-indexes `spikes['clusters']`, re-sorts by time |
| `load_trials_and_mask` | `ibl_data_utils.py` | CURATION | Trials table + boolean inclusion mask (reaction time 0.08–2 s, no NaN in 6 events, `choice != 0`, optional `max_trial_len`) |
| `list_brain_regions` / `select_brain_regions` | `ibl_data_utils.py` | PROCESSING | Maps cluster acronyms to the **Beryl** atlas; with `single_region=False` returns *all* clusters (no region filtering) |
| `bin_spiking_data` → `get_spike_data_per_interval` | `ibl_data_utils.py` | PROCESSING | Per-trial interval `[align + w0, align + w1]`, `bincount2D(..., xbin=binsize, xlim=[t_beg,t_end])`, keeps first `ceil(interval_len/binsize)` bins |
| `load_target_behavior` | `ibl_data_utils.py` | LOADING | `SessionLoader.load_wheel()` → `wheel-speed = abs(velocity)`; `SessionLoader.load_motion_energy()` → `whiskerMotionEnergy` |
| `get_behavior_per_interval` | `ibl_data_utils.py` | PROCESSING | Linear interpolation of the behaviour trace onto `np.linspace(t_beg+binsize, t_end, n_bins)` (right edges of the neural bins); rejects intervals not covered by the trace or containing NaN (when `allow_nans=False`) |
| `bin_behaviors` | `ibl_data_utils.py` | PROCESSING | Builds per-trial `choice`, `block` (= `probabilityLeft`), `reward`, `contrast` and the time-varying behaviours |
| `align_spike_behavior` | `ibl_data_utils.py` | CURATION | Drops trials that fail the trial mask / behaviour masks |
| `standardize_spike_data` | `src/utils/data_loader_utils.py` | PROCESSING | Per-time-bin z-scoring — applied **in the model's data loader**, not in the cached dataset |

### Notes
- **Binning parameters** are fixed once, in `0_data_caching.py`:
  `{'interval_len': 2, 'binsize': 0.02, 'single_region': False,
    'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` → T = 100 bins of 20 ms,
  aligned to stimulus onset. This is exactly the alignment the decoder task asks for.
- **Behaviours cached**: `['choice', 'reward', 'block', 'wheel-speed',
  'whisker-motion-energy']` (pupil excluded — "Some sessions do not have pupil traces").
- **Whisker motion energy** uses the *left* camera, falling back to the *right* camera
  when the left is unavailable (`bin_behaviors`).
- **Electrophysiology QC**: the caching script calls `load_spiking_data` without `qc`, so
  the reference caches *all* units; but it stores `good_clusters = (label >= 1)` in the
  metadata, and `load_spiking_data`'s docstring documents `qc=1` as the good-unit setting.
  See Step 4/5 for the decision taken here.
- **No dF/F** is involved: this is electrophysiology (spike times), not imaging.
- The model-side loader z-scores spikes per time bin and standardises continuous
  behaviours; those are *model* steps, not dataset steps, and the target format here wants
  raw neural activity plus categorical outputs, so they are not applied.
- Bug noticed in `align_spike_behavior`: `target_mask = target_mask and beh_mask` uses
  Python's `and` on lists, which returns the *second* operand whole — so only the last
  behaviour's mask (and then only `trials_mask`) actually survives. Fixed here by
  AND-ing the masks element-wise (required anyway: the target format forbids NaN).
- Bug noticed in `merge_probes`: `cluster_max = clusters.index.max() + 1` replaces rather
  than accumulates the offset. Harmless for this release (no session has more than 2
  probes — 240 sessions have 2, 219 have 1), but corrected here.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
A standard ONE cache:

```
/app/data/one_cache/
  Brainwidemap/{sessions,datasets}.pqt        480 sessions / 76,563 dataset records
  2022_Q4_IBL_et_al_BWM/{...}.pqt             354 sessions / 38,814 records
  2025_Q3_IBL_et_al_BWM/{...}.pqt             459 sessions / 5,339 records (motion energy only)
  <lab>/Subjects/<subject>/<date>/<number>/
      alf/_ibl_wheel.{position,timestamps}.npy
      alf/#2025-03-03#/_ibl_trials.table.pqt          <-- revision folder
      alf/#2025-05-29#/leftCamera.ROIMotionEnergy.npy <-- revision folder
      alf/#2023-04-20#/_ibl_{left,right}Camera.times.npy
      alf/<probe>/pykilosort/#2024-05-06#/{spikes,clusters,channels}.*
```

`bwm_release.csv` (shipped with the reference code) enumerates the release:
**699 pids, 459 eids, 139 subjects, 12 labs** — identical to the data paper.

**Problem found and solved**: the shipped release tables are *stale* relative to the staged
files. `alf/_ibl_trials.table.pqt` is listed with `default_revision=False` and does not
exist on disk; all 459 sessions carry the trials table only inside revision folders
(454 × `#2025-03-03#`, 5 × both `#2024-07-15#` and `#2025-03-03#`). Loading through the
shipped table silently returns a 1-column trials frame (`goCueTrigger_times` only) instead
of raising. The conversion therefore rebuilds the `datasets` cache table by walking each
session's `alf` tree (`build_dataset_table`), so ONE resolves exactly the files present,
newest revision first. ONE's offline `eid2pid` also needs the network, so probe IDs come
from `bwm_release.csv` instead.

### Available variables
- `trials` (20 columns): `stimOn_times`, `stimOff_times`, `goCue_times`,
  `goCueTrigger_times`, `response_times`, `feedback_times`, `firstMovement_times`,
  `intervals_0/1`, `intervals_bpod_0/1`, `choice`, `contrastLeft`, `contrastRight`,
  `probabilityLeft`, `feedbackType`, `rewardVolume`, `quiescencePeriod`,
  `stimOnTrigger_times`, `stimOffTrigger_times`
- `wheel`: `times`, `position`, `velocity`, `acceleration` (uniformly resampled to 1 kHz
  and Butterworth-filtered by `SessionLoader.load_wheel`)
- `leftCamera`/`rightCamera`: `ROIMotionEnergy` (whisker pad) + `times` (60 / 150 Hz)
- spike sorting (pykilosort, revision `#2024-05-06#`): `spikes.times/clusters/amps/depths`,
  `clusters.metrics` (22 QC columns incl. `label`), `clusters.channels/depths/uuids`,
  `channels.*` with Allen CCF `acronym`

### Dataset Size (from data files, before any curation)
| Statistic | Value |
|-----------|-------|
| Neurons (total, all units) | **621,733** |
| Neurons (total, `label == 1`) | **75,708** |
| Neurons / probe (all / good) | 889.5 / 108.3 |
| Neurons / session (all / good) | 1354.5 / 164.9 (median good 142, min 2) |
| Subjects | 139 |
| Sessions | 459 |
| Probes | 699 (240 sessions × 2 probes, 219 × 1) |
| Labs | 12 |
| Sessions with whisker motion energy | 445 (436 left camera, 433 right, union 445) |
| Trials / session | ~400–600 raw |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total, all units) | 621,733 | "This process produced 621,733 units (including multineuron activity), averaging 889 per probe" (data paper) |
| Neurons (well-isolated) | 75,708 | "which identified 75,708 well-isolated neurons, averaging 108 per probe" (data paper) |
| Subjects | 139 | "We trained 139 mice (94 male and 45 female)" (data paper) |
| Probes | 699 | "we inserted 699 Neuropixels probes" |
| Sessions (release) | 459 | "a total of 459 sessions, 699 insertions and 621,733 neurons remained, constituting the publicly released dataset" |
| Sessions (decoding) | **433** | "We apply our models to 433 IBL sessions, covering 270 brain regions" (methods paper) |
| Brain regions | 270 (methods paper) / 279 (data paper, Allen areas) | "covering 270 brain regions"; "The probes covered 279 brain areas" |
| Neural data time bin | **20 ms**, T = 100 | "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps" |
| Alignment / window (choice) | stimulus onset, −0.5 → +1.5 s | "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset" |
| Behaviour sampling | 60 Hz | "wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz" |
| Reaction times < 80 ms | 22.8 % | "A total of 22.8% first wheel-movement times occurred under 80 ms" (Fig. 1c, data paper) |
| Performance, 0 % contrast | 58.7 ± 0.4 % correct | data paper Fig. 1d |
| Block structure | 90 unbiased trials, then 20–100-trial blocks, mean 51 | "Blocks lasted for between 20 and 100 trials … (empirical mean of 51 trials)" |
| Prior values | 0.2 / 0.5 / 0.8 | "the prior probability … at a ratio of 20:80% … or 80:20%" |

### Processing Details
- **Temporal alignment**: `stimOn_times`; window (−0.5 s, +1.5 s); 100 non-overlapping
  20 ms bins. (The methods paper also mentions a 50 ms binning for its
  *prior* decoder over a −0.6 → −0.1 s window, and a first-movement alignment for the
  *dynamic* behaviours; the decoder task here fixes stimulus-onset alignment, and the
  reference caching script itself uses stimOn / (−0.5, 1.5) / 20 ms for everything it
  caches, so that is what is used.)
- **Neural regressors**: raw spike counts per bin, all neurons of the session pooled
  across probes ("neurons in the same session and region were combined across probes").
- **Behaviour resampling**: linear interpolation onto the right edge of each 20 ms bin.

### Curation Steps

**Neuron curation rules** (data paper, "Neurons and brain regions"):
> "Neurons … were excluded … if they failed one of the three criteria …: amplitude > 50 μV;
> noise cut-off < 20 μV; and refractory period violation. Neurons that passed these criteria
> were termed well-isolated neurons … Out of the 621,733 units collected, 75,708 were
> considered well-isolated neurons."

In the data this is exactly `clusters['label'] == 1` (label is the fraction of the three
RIGOR metrics passed) — **verified numerically: 621,733 units, 75,708 with label == 1.**

**Trial curation rules** (data paper, "Trials"; identical to the reference
`load_trials_and_mask` defaults):
> "trials were excluded if one of the following trial events could not be detected: choice,
> probabilityLeft, feedbackType, feedback times, stimOn times and firstMovement times.
> Trials were further excluded if the time between stimulus onset and the first movement of
> the wheel … were outside the range of 0.08–2.00 s."

plus, from the reference caching script, `max_trial_len=10.0`
(`feedback_times - goCue_times <= 10 s`) and `exclude_nochoice=True` (`choice != 0`).

**Session curation**: release criteria (≥ 250 trials, ≥ 90 % correct on 100 % contrast,
≥ 3 error trials, hardware QC, resolved histology) are already applied upstream — the 459
released sessions all pass. The methods paper's 433 sessions is the subset that survives
"sessions excluded from decoding because of missing behavioral data".

### Decoders Trained (reported in the papers)
| Decoded variable | Reported performance |
|---|---|
| Choice (per-session, all regions, RRR) | accuracy 0.51–0.91 (multi-session), 0.51–0.73 (single-session); Fig. 2C |
| Choice (per-region, BWM) | null-corrected **median balanced accuracy**, mostly 0.50–0.62 per region |
| Prior | AUC 0.66 (single-trial baseline) → 0.72 / 0.79 (BMM-HMM variants); correlation 0.05–0.65 |
| Wheel speed | R² 0.24–0.75 per session (per-timestep regression) |
| Whisker motion energy | R² 0.49–0.80 per session (per-timestep regression) |

These are different metrics/architectures (ridge / reduced-rank regression on continuous
targets, region-restricted) from the balanced accuracy over 3 discretised classes computed
by `train_decoder.py`, so they set expectations of *direction and rough magnitude*, not
exact values: choice modestly above 0.5, prior clearly above 1/3, and the two continuous
behaviours well above 1/3 (they are the best-decoded variables in both papers).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neuron QC | `0_data_caching.py` calls `load_spiking_data` with default `qc=None` → all 621,733 units | `clusters['label'] >= 1` selects exactly 75,708 units | Data paper: analyses use the 75,708 **well-isolated** neurons; methods paper: "all neurons, sorted by Kilosort 2.5" | **Keep `label == 1` (well-isolated).** The data paper states the criterion explicitly and quantitatively, the number reproduces exactly (a hard sanity check), `load_spiking_data` exposes `qc=1` for precisely this, and the reference still records `good_clusters` in its metadata. Documented as a deliberate deviation from `qc=None`. |
| Number of sessions | code processes whatever is requested | 459 released; 445 have whisker motion energy | methods paper: 433 sessions | Sessions without usable whisker motion energy cannot supply a required output and are dropped; the resulting count is compared against 433/445 in Step 9. |
| Trial-length cap | reference passes `max_trial_len=10.0` | removes ~1 extra trial/session | data paper does not mention it | Kept (follows the reference decoding pipeline); its effect is negligible (≈0.2 % of trials). |
| NaNs in behaviour | reference caches with `allow_nans=True`, then mean-imputes in the model loader | some trials have NaN wheel/ME inside the window | — | Target format forbids NaN, so such trials are dropped instead of imputed (`allow_nans=False` path of the same reference function). |
| Region filtering | `single_region=False` → **all** clusters, incl. `root`/`void` | ~20 % of good units map to `root` | data paper restricts *region-level* analyses to grey matter with ≥5 neurons/session | No region filtering: the decoder here is a whole-session ("region = all") decoder, exactly the reference's `region='all'` setting. Region labels are still recorded per neuron. |
| Trials table revision | ONE table lists un-revised path | only revision folders exist on disk | — | `datasets` cache table rebuilt from the filesystem (Step 2). |
| Camera for whisker ME | left, falling back to right | 436 sessions have left, 433 right, 445 either | "whisker motion energy near the whisker pad" | Follow the reference: left first, right as fallback. |

Everything else (alignment event, window, bin size, trial mask, probe merging, Beryl
region mapping) is consistent across code, data and papers.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` (label == 1, probes merged) | `neural[s][k]` (n_neurons, 100) | spike counts in 20 ms bins, `floor((t − (stimOn−0.5))/0.02)` | `load_spiking_data`, `merge_probes`, `bin_spiking_data` | float32 counts |
| bin index | `input[0]` `time_from_stim_on` | bin centre time, −0.49 … +1.49 s | (new; the decoder task asks for it) | time-varying |
| `trials.probabilityLeft` | `input[1]` `trial_number_in_block` | 0-based position inside the run of constant `probabilityLeft`, computed over **all** trials of the session | (new) | per-trial, broadcast over time |
| `trials.choice` | `output[0]` `choice` | `+1 → 0` (left), `−1 → 1` (right) | `bin_behaviors` | verified against `feedbackType`/`contrastLeft`/`contrastRight` |
| `trials.probabilityLeft` | `output[1]` `prior_prob_left` | 0.2→0, 0.5→1, 0.8→2 | `bin_behaviors` (`block`) | per-trial |
| `wheel.velocity` | `output[2]` `wheel_speed` | `abs()`, interpolate to bin right edges, digitise at within-session tertiles | `load_target_behavior('wheel-speed')`, `get_behavior_per_interval` | time-varying |
| `leftCamera.ROIMotionEnergy` (else right) | `output[3]` `whisker_motion_energy` | interpolate to bin right edges, digitise at within-session tertiles | `load_target_behavior('left-whisker-motion-energy')` | time-varying |
| `bwm_release.csv.subject` | `subjects`, `subject_idx` | unique sorted | — | |
| `clusters.acronym` → Beryl | `brain_regions`, `brain_region_idx` | `BrainRegions().acronym2acronym(..., mapping='Beryl')` | `list_brain_regions` | |

### Key Decisions
1. **Alignment / binning**: `stimOn_times`, window (−0.5, +1.5) s, 100 × 20 ms bins —
   identical to the reference caching parameters and to the decoder task's requirement to
   "temporally align based on stimulus onset".
2. **Neuron curation**: keep only `clusters['label'] == 1` (well-isolated). Reproduces the
   data paper's 75,708 exactly; see the Step 4 table for why this differs from the
   reference script's `qc=None`.
3. **No region filtering**: whole-session decoding, matching the reference's `region='all'`.
4. **Trial curation**: reference `load_trials_and_mask` defaults + `max_trial_len=10.0`,
   then additionally drop trials whose wheel or whisker trace does not cover the window or
   contains NaN inside it (the format forbids NaN).
5. **Choice coding**: IBL `choice == +1` means the mouse reported the **left** stimulus
   (verified: on correct trials with a left stimulus, `choice` is always +1). Mapped to
   left = 0, right = 1 as the task specifies.
6. **Discretisation of the two continuous outputs**: within-session tertiles
   (equal-occupancy 3-way split over all retained trials × timepoints of that session).
   Rationale: whisker motion energy is in arbitrary units that depend on camera resolution
   and frame rate (left 1280×1024 @ 60 Hz vs right 640×512 @ 150 Hz) and on lighting, so no
   fixed threshold transfers across sessions; wheel speed is strongly zero-inflated and
   varies in scale across animals. Equal-occupancy edges give the same class semantics
   (low / medium / high for this session) everywhere and make chance exactly 1/3 for the
   balanced-accuracy metric that `train_decoder.py` reports. The edges actually used are
   stored per session in `metadata['session_info']`.
7. **Time-varying wherever possible**: all four outputs are emitted as (4, 100) arrays,
   with the two per-trial variables constant across the 100 bins; inputs likewise (2, 100).
8. **Neural values**: raw spike counts (float32), not rates or z-scores — the reference
   caches counts and z-scoring is a model-side step.

### Planned Sanity Checks
- [x] Total units == 621,733 and `label == 1` units == 75,708 (data paper)
- [x] 459 eids / 699 pids / 139 subjects / 12 labs in the release table
- [x] `choice == +1` ⟺ left stimulus on correct trials
- [x] `probabilityLeft ∈ {0.2, 0.5, 0.8}` only
- [ ] Fraction of reaction times < 80 ms ≈ 22.8 % (data paper Fig. 1c)
- [ ] Mean block length ≈ 51 trials; first block 90 trials at p = 0.5
- [ ] Number of Beryl regions ≈ 270 (methods paper)
- [ ] Sessions retained ≈ 433–445
- [ ] Population PSTH shows a stimulus-locked transient at t = 0 (alignment)
- [ ] Binned spike counts reproduce the raw raster for a spot-checked trial
- [ ] Binned wheel / whisker traces reproduce the raw traces
- [ ] Output class fractions: wheel and whisker ≈ 1/3 each by construction

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Runs as
`python -u /app/convert_data.py <out.pkl> [--full|--sample] [--show-processing]`.

Implementation notes:
- `build_dataset_table` rebuilds ONE's `datasets` table from the staged filesystem
  (see Step 2) — ~7 s for all 459 sessions.
- `load_spiking_data` / `merge_probes` / `load_trials_and_mask` are transcriptions of the
  reference functions (the `raw_electrophysiology(...).fs` call is dropped: it needs the
  network and only fed a metadata field).
- `bin_spiking_data` is a vectorised rewrite of
  `get_spike_data_per_interval`: the reference spawns a multiprocessing pool and calls
  `bincount2D` once per trial. Here spike indices are found with two `np.searchsorted`
  calls over the whole session and each trial is one `np.bincount` on
  `cluster * 100 + floor((t − t_beg)/0.02)` — the identical bin definition.
- `bin_behavior` is a transcription of `get_behavior_per_interval` including all four
  skip conditions, with `allow_nans=False`.
- Sessions are converted in a `ProcessPoolExecutor` (24 workers).

Code inefficiencies identified:
- Reference `get_spike_data_per_interval` / `get_behavior_per_interval` create a process
  pool *per session per signal* and a tqdm bar per trial — dominant cost for 459 sessions.

Code speedups added:
- Vectorised spike binning (no per-trial pool): 500-trial session bins in < 0.2 s.
- Parallelism moved up to the session level, 24 workers.
- Spike sorting loaded once per probe; `float32`/`int8` storage.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`processing_<eid>.png` × 2.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 340 |
| Neurons / session | 76, 264 (mean 170) |
| Subjects | 1 (NYU-11) |
| Sessions / subject | 2 |
| Trials (total) | 651 |
| Trials / session | 407, 244 (mean 325.5) |
| T per trial | 100 (all trials) |
| `time_from_stim_on` range | [−0.49, 1.49] |
| `trial_number_in_block` range | [0, 89] |
| `choice` distribution | left 0.518 / right 0.482 |
| `prior_prob_left` distribution | 0.2 → 0.478, 0.5 → 0.161, 0.8 → 0.361 |
| `wheel_speed` distribution | 0.333 / 0.333 / 0.333 |
| `whisker_motion_energy` distribution | 0.333 / 0.333 / 0.333 |

Trial attrition for the two sample sessions (565 → 407 and 425 → 244) is dominated by the
BWM reaction-time criterion (87 and 79 trials with RT < 80 ms; 59 and 98 with RT > 2 s),
not by the behaviour masks, which dropped **0** trials in both sessions.
`max_trial_len=10 s` removed 1 extra trial per session.

### Processing Plots Review
`processing_<eid>.png` panels (raster vs binned counts, population PSTH, raw vs resampled
wheel / whisker traces, discretisation scatter, per-trial variables, class fractions):
- The population PSTH is flat before t = 0 and rises sharply at t = 0 to a peak at
  ≈ 0.22 s — the alignment to stimulus onset is correct and there is no off-by-one-bin
  shift.
- Binned spike counts visually match the raw raster (same rows active at the same times).
- Resampled wheel speed and whisker motion energy overlay the raw traces.
- The discretisation scatter shows three non-overlapping value bands separated by the two
  tertile edges.
- `trial_number_in_block` is a sawtooth that resets exactly where `probabilityLeft`
  changes; the first block runs to 89 (the 90-trial unbiased block).
- Class fractions for the two discretised outputs are 0.33/0.33/0.33 as intended.
No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| Vectorised spike binning instead of per-trial multiprocessing `bincount2D` | ~100× on the binning step |
| Session-level `ProcessPoolExecutor` (24 workers) instead of per-trial pools | ~20× wall clock |
| Filesystem-built ONE dataset table (one 7 s scan, shared by fork) | avoids a per-session table query |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| ONE cache build | — | 7 s (once) |
| trials | 0.25 s | |
| behaviour load + bin | ~1.5 s | |
| spike sorting load (per probe) | ~1.5–3 s | |
| spike binning | ~0.2 s | |
| **total per session (serial)** | **~4.5 s** (1 probe) / ~7 s (2 probes) | 459 × ~6 s ≈ 46 min serial |
| **with 24 workers** | | **≈ 3–5 min + pickling** |

Estimated output size: 459 sessions × ~340 trials × ~165 neurons × 100 bins × 4 B
≈ 10 GB, well within the 15-minute budget and available RAM/disk.

### Format verification (`/app/verification_sample_out.txt`)
- Errors: **None**
- Warnings: **None**

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`

Loss decreased monotonically: 1.99 (epoch 1) → 0.744 (epoch 200); test loss 0.776.

### Decoder Results (Sample, 2 sessions)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|-------------------------|
| choice | 0.500 | 0.6189 | 0.5415 |
| prior_prob_left | 0.333 | 0.7124 | 0.6709 |
| wheel_speed | 0.333 | 0.5829 | 0.5723 |
| whisker_motion_energy | 0.333 | 0.5709 | 0.5670 |

All four outputs are above chance on held-out trials. Choice is the weakest, which is
expected: these two sessions record hippocampus / thalamus / amygdala (CA1, DG, LGd, MG,
CEA, BMA, VPM), regions where the BWM paper's choice decoding is close to chance, and the
shared decoder only has two sessions of data here.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full` →
`/app/conversion_full_out.txt`.  Run time **162 s** (459 sessions, 24 workers,
0.4 s/session) — well inside the 15-minute budget and close to the Step 7 estimate.
Verification: `python -u /app/train_decoder.py /app/converted_data.pkl --verify-only`
→ `/app/verification_full_out.txt`.

### Output Files
- `converted_data.pkl`: **12.96 GB**
- `verification_full_out.txt`: created — **"Data format is valid, no errors or warnings."**

### Sessions dropped (17 of 459), every one accounted for
| Reason | N | Evidence |
|---|---|---|
| No whisker motion-energy file at all | 14 | 445 of 459 eids have a `{left,right}Camera.ROIMotionEnergy` record in the release tables; 459 − 445 = 14 |
| Whisker ME present but camera timestamps unusable | 1 | `f8041c1e-…`: right-camera times span 1515–6852 s while the trials span 29–3941 s, so no trial window is covered |
| Fewer than 5 well-isolated neurons | 2 | `d16a9a8d-…` (3 neurons), `3a3ea015-…` (2 neurons) |

442 sessions retained. The methods paper reports **433** sessions after "sessions excluded
from decoding because of missing behavioral data"; the 9-session difference is explained by
the newer motion-energy revisions in this staged release (`#2025-05-29#` / `#2025-06-01#`),
which cover 436 sessions on the left camera where the release the paper used covered fewer.

### Trial attrition (over the 442 kept sessions)
| Stage | Trials | Note |
|---|---|---|
| raw | 285,682 | mean 646 / session; min 401 (release criterion ≥ 400 ✓) |
| after BWM trial mask | 188,469 | 66.0 % retained; dominated by RT < 80 ms (22.8 %) and RT > 2 s (9.6 %) |
| after behaviour coverage mask | 188,383 | −86 trials (0.05 %) |
| after no-spike (dropout) mask | **188,367** | −16 trials |

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions in release | 459 | 459 eids in `bwm_release.csv` | 459 session dirs on disk | 459 processed, 442 kept | ✓ |
| Sessions decoded | 433 (methods paper) | — | 445 with ME | 442 | ✓ (see above) |
| Probes | 699 | 699 pids | 699 spike-sorting dirs | 699 loaded | ✓ |
| Subjects | 139 | 139 | 139 | 136 (3 lost with the 17 dropped sessions) | ✓ |
| Labs | 12 | 12 | 12 | 12 | ✓ |
| Units (all) | 621,733 | — | **621,733** | 599,335 in kept sessions | ✓ exact on the release |
| Well-isolated neurons | 75,708 | `label >= 1` | **75,708** | 73,039 in kept sessions | ✓ exact on the release |
| Units / probe | 889 | — | 889.5 | — | ✓ |
| Good units / probe | 108 | — | 108.3 | — | ✓ |
| Neurons / session (good) | — | — | 164.9 | 165.2 | ✓ |
| Trials / session (raw) | ≥ 400 retained | — | min 401, mean 645 | min 401, mean 646 | ✓ |
| Trials / session (used) | — | — | — | mean 426, median 392, 125–1445 | — |
| RT < 80 ms | **22.8 %** | excluded by mask | **22.8 %** | excluded | ✓ exact |
| Performance at 0 % contrast | 58.7 ± 0.4 % | — | **58.6 ± 0.3 %** | — | ✓ |
| First (unbiased) block length | 90 | — | 90 in all 459 sessions | — | ✓ |
| Biased block length | 20–100, mean 51 | — | 20–99, mean 50.2 | — | ✓ |
| probabilityLeft values | 0.2 / 0.5 / 0.8 | `block` | {0.2, 0.5, 0.8} | {0,1,2} | ✓ |
| Brain regions (Beryl) | 270 over 433 sessions | Beryl mapping | 268 (good units, 459 sessions) / 281 (all units) | **265** | ✓ |
| Time bins | 100 × 20 ms | 100 × 20 ms | — | 100 × 20 ms, all trials | ✓ |
| Alignment | stimOn_times | `stimOn_times` | — | stimOn_times | ✓ |
| `time_from_stim_on` range | −0.5 … 1.5 s | (−0.5, 1.5) | — | [−0.49, 1.49] (bin centres) | ✓ |
| `trial_number_in_block` range | blocks ≤ 100 trials | — | 0–99 | [0, 98] | ✓ |
| choice distribution | ~balanced | — | — | left 0.508 / right 0.492 | ✓ |
| prior distribution | 90 unbiased of ~645 ⇒ ~14 % at 0.5 | — | — | 0.417 / 0.140 / 0.442 | ✓ |
| wheel / whisker distribution | — | — | — | 0.333 / 0.333 / 0.333 each | ✓ by construction |

Top regions by neuron count in the converted data — root 10,094, CP 2,890, MRN 2,590,
PO 2,235, CA1 1,783, LP 1,669, SCm 1,605, APN 1,543, MOp 1,414 — match the high-yield
regions the BWM paper highlights.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output-log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**

The first full run *did* emit 16 warnings of the form "Session s, trial k: all neural data
is zero". Investigated with the raw files:
- `8c2f7f4d-…`: the spike train ends at 1779.4 s but the last three trials have stimulus
  onsets at 1790–1800 s — the ephys stream stopped before the behavioural session did.
- `b182b754-…`: a 4.6 s gap in the spike train at t ≈ 185 s swallows one trial.
- `195443eb-…` (10 neurons): 13 scattered trials with no spikes at all.

These are recording dropouts, not conversion errors, but a trial with an identically-zero
population carries no neural information. A `binned.sum() > 0` trial mask was added — the
neural counterpart of the reference's own "target data ends too early" behaviour check —
removing 16 of 188,383 trials (0.008 %). Two sessions with 2 and 3 well-isolated neurons
were also dropped under a `MIN_NEURONS = 5` rule (the BWM paper's "at least five
well-isolated neurons per session"). After both changes there are **no warnings left**.

### Check 2 — independent sanity checks (`/app/cache/sanity_checks.py`)
52 checks over 4 sessions spanning the size range (11 / 143 / 288 / 474 neurons, 1 and 2
probes). Every check re-derives the quantity from the raw ALF `.npy` / `.pqt` files with
plain numpy/pandas — **none of `convert_data.py`'s loading, binning or discretisation code
is used** (only `get_session_table`, which just locates the session directory).
**52/52 passed** (`/app/cache/sanity_checks_out.txt`):

| Stream | Check | Result |
|---|---|---|
| trials | trial-mask count recomputed from `_ibl_trials.table.pqt` == `n_trials_after_trialmask` | exact, 4/4 |
| neural | neuron count from `clusters.metrics.pqt` (`label >= 1`) == `neural[s][k].shape[0]` | exact, 4/4 |
| neural | **every** converted trial reproduced bin-for-bin from `spikes.times.npy` / `spikes.clusters.npy` (1,192 trials total) | exact, 4/4 |
| neural | `np.allclose` on trial 5 of each session | pass |
| input | `input[0]` == bin-centre time vector | `np.allclose` pass |
| input | `input[1]` == trial index in block recomputed from the raw `probabilityLeft` column | exact |
| input | first block of every session is the 90-trial `p = 0.5` unbiased block | pass |
| output | `output[0]` == `(choice == −1)` from the raw parquet | exact |
| output | on correct trials with a left stimulus, `output[0] == 0` | pass (150 / 177 / 105 / 107 trials) |
| output | `output[1]` == `probabilityLeft` code | exact |
| output | `output[2]` classes recomputed from `_ibl_wheel.position.npy` via `brainbox.behavior.wheel.interpolate_position` + `velocity_filtered` | agreement 0.9994–0.9999 |
| output | wheel tertile edges == the 1/3 and 2/3 quantiles of the independently computed values | `np.allclose(rtol=2e-3)` pass |
| output | `output[3]` classes recomputed from `{left,right}Camera.ROIMotionEnergy.npy` + camera times | agreement **1.000** |

The only non-exact agreements are the wheel labels (0.06 % of timepoints). Cause: the
independent path resamples the wheel with a slightly different time grid than
`SessionLoader.load_wheel` (which is what the conversion uses), so a handful of values sit
on the opposite side of a tertile edge. The tertile edges themselves agree to < 0.1 %.

### Check 3 — reference code comparison
| Stage | Reference (`ibl_data_utils.py` / `0_data_caching.py`) | This conversion | Same? |
|---|---|---|---|
| (a) loading, spikes | `SpikeSortingLoader.load_spike_sorting()` + `merge_clusters()`; `merge_probes` | identical call sequence (`load_spiking_data`, `merge_probes`) | ✓ (`merge_probes` offset bug corrected; no effect — no session has > 2 probes) |
| (a) loading, trials | `load_trials_and_mask(one, eid, max_trial_len=10.0)` | transcribed verbatim, same defaults, same `max_trial_len` | ✓ |
| (a) loading, behaviour | `SessionLoader.load_wheel()` → `abs(velocity)`; `load_motion_energy(['left'])` with right fallback | identical | ✓ |
| (b) neuron filtering | `qc=None` → all 621,733 units | `label >= 1` → 75,708 well-isolated units | **deliberate difference** (Step 4) |
| (b) region filtering | `single_region=False` → no filtering | none | ✓ |
| (b) trial filtering | `load_trials_and_mask` + `align_spike_behavior` | same mask, plus element-wise AND of the behaviour masks (the reference's `and` on lists silently discards all but the last) and the no-spike mask | ✓ + fixes |
| (b) session filtering | `try/except` skip on any error | explicit: no ME, unusable camera times, < 5 neurons, < 2 trials | ✓ (made explicit and reported) |
| (c) temporal alignment | `intervals = trials[align_time] + time_window`, `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)` | identical | ✓ |
| (d) binning, neural | `bincount2D(t, c, xbin=0.02, xlim=[t_beg, t_end])[:, :100]` ⇒ bin `floor((t − t_beg)/0.02)`, left-closed | `np.bincount` on `cluster*100 + floor((t − t_beg)/0.02)` — same bin definition, vectorised | ✓ |
| (d) binning, behaviour | `interp1d(linear, extrapolate)` at `linspace(t_beg + binsize, t_end, 100)` = bin right edges, with 4 coverage/NaN skip rules | transcribed verbatim, `allow_nans=False` | ✓ |
| (e) input construction | reference has no decoder-input stream (it caches behaviours only) | `time_from_stim_on` (bin centres), `trial_number_in_block` | new, per the decoder task |
| (f) output construction | `choice` (±1), `block` = `probabilityLeft`, `wheel-speed`, `whisker-motion-energy` — all kept continuous/raw | same four variables, recoded to the categorical codes the task specifies | ✓ + task-required discretisation |
| post-processing | `standardize_spike_data` (per-bin z-score), `StandardScaler` on behaviour — inside the *model's* data loader | not applied | ✓ (model-side step; target format wants raw activity and categorical outputs) |

Reasons for each difference are given in Step 4 (neuron QC), Step 5 (discretisation,
input construction) and Check 1 (no-spike mask, MIN_NEURONS).

### Check 4 — key statistics comparison (`/app/cache/paper_stats.py`)
Recomputed from the raw files for **all 459 released sessions**, independently of the
conversion (`/app/cache/paper_stats_out.txt`):

| Statistic | Paper | Measured | Match |
|---|---|---|---|
| sessions / probes / subjects / labs | 459 / 699 / 139 / 12 | 459 / 699 / 139 / 12 | ✓ |
| total units | 621,733 | **621,733** | ✓ exact |
| well-isolated neurons | 75,708 | **75,708** | ✓ exact |
| units per probe | 889 | 889.5 | ✓ |
| good units per probe | 108 | 108.3 | ✓ |
| min trials per session | ≥ 400 | 401 | ✓ |
| RT < 80 ms | 22.8 % | **22.8 %** | ✓ exact |
| 0 %-contrast performance | 58.7 ± 0.4 % | 58.6 ± 0.3 % | ✓ |
| unbiased first block | 90 trials | 90 in every session | ✓ |
| biased block length | 20–100, mean 51 | 20–99, mean 50.2 | ✓ |
| probabilityLeft values | 0.2 / 0.5 / 0.8 | {0.2, 0.5, 0.8} | ✓ |
| Beryl regions | 270 (433 sessions) | 268 good-unit / 281 all-unit (459 sessions); 265 in the converted file | ✓ |

The exact reproduction of 621,733 / 75,708 / 22.8 % confirms that the correct spike-sorting
revision, the correct QC definition and the correct trial criteria are being used.
The region count also shows the QC choice costs almost nothing in coverage
(268 vs 281 Beryl regions).

### Check 5 — edge cases
| Edge case | Handling | Verified |
|---|---|---|
| Stale ONE release tables (trials only in a revision folder) | dataset table rebuilt from disk; a silent 1-column trials frame would otherwise be returned | trials frames have all 20 columns for all 459 sessions |
| `eid2pid` needs the network | probe IDs taken from `bwm_release.csv` | 699/699 probes loaded |
| Multi-probe sessions | `merge_probes` with an accumulated offset | 2-probe sessions spot-checked in Check 2 (s320, s299, s12) |
| NaN spike times | dropped before binning (`np.isfinite`) | none present in this release |
| Spike times not sorted after merge | re-sorted with a stable argsort before `searchsorted` | binning matches raw recomputation exactly |
| Bin-edge off-by-one | `floor((t − t_beg)/binsize)`, clipped to `[0, 100)`; first bin starts exactly at `stimOn − 0.5`, last ends at `stimOn + 1.5` | independent recomputation reproduces every trial |
| Trial 0 / last trial of a block | `trial_number_in_block` computed over the *full* trials table before exclusion, so the first block still runs 0…89 even when its first trials are excluded | checked against raw `probabilityLeft` |
| Behaviour trace not covering the window | the reference's 4 skip rules (`starts too late` / `ends too early` / NaN / empty) | 86 trials dropped |
| Ephys ends before the behaviour does / recording gap | no-spike trial mask | 16 trials dropped, 0 warnings left |
| Session with < 2 usable trials | dropped with an explicit reason | 0 such sessions after the other rules |
| Degenerate (near-constant) behaviour trace | `discretize_tertiles` falls back to quantiles of the unique values, then to an eps-separated edge | no session needed the fallback (all edges strictly increasing) |
| `probabilityLeft` outside {0.2, 0.5, 0.8} | session rejected with a reported reason | never triggered |
| Sessions with only right-camera video | right-camera fallback | 7 of 442 sessions use the right camera |

### Iterations
1. **Iteration 1** — first full run (444 sessions) produced 16 "all neural data is zero"
   warnings. Root-caused to ephys dropouts (above), added the no-spike trial mask and
   `MIN_NEURONS = 5`; re-ran conversion + verification + all checks. Result: 442 sessions,
   **no errors and no warnings**, all 52 sanity checks still pass.
2. **Iteration 2** — added the per-session exclusion counters (`n_units_total`,
   `n_rt_short`, …) to `metadata['session_info']` and re-ran the conversion, verification,
   sample outputs and decoder training so every artefact refers to the same pickle.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.
Trained on GPU (NVIDIA L4), 200 epochs, 442 sessions / 188,367 trials / 73,039 neurons.

### Training Progress
- Loss decreasing: **Yes**, monotonically — 2.467 (epoch 1) → 1.222 (10) → 0.850 (50) →
  0.750 (100) → 0.736 (200). Test loss 0.760.

### Decoder Results (Full)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|--------|-----------------------|-------------------------|--------------|-------|
| choice | 0.500 | 0.6446 | **0.6224** | 1.24× | peaks at 0.708 at t = +0.27 s (see Step 12) |
| prior_prob_left | 0.333 | 0.6875 | **0.6676** | 2.00× | decodable throughout the trial, as expected for a slow block variable |
| wheel_speed | 0.333 | 0.6196 | **0.6130** | 1.84× | |
| whisker_motion_energy | 0.333 | 0.5990 | **0.5934** | 1.78× | |

`sample_trials.png` shows the two time-varying outputs rising together ~0.2 s after
stimulus onset in every sampled trial, and the two per-trial outputs flat — exactly the
expected structure.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — accuracy vs chance
No output is at or below chance. Three of four exceed 1.75× chance. Only `choice`
(1.24× chance) is under the 1.5× flag, which for a **binary** variable would mean a
balanced accuracy of 0.75 — higher than any pooled, brain-wide, stimulus-onset-aligned
choice decoding reported in either paper. Investigated with a time-resolved analysis
(`/app/cache/timecourse_diagnostic.py`, `/app/cache/accuracy_vs_time.png`), which retrains
the same decoder and scores each of the 100 time bins separately:

| t from stimulus onset | choice | prior | wheel speed | whisker ME |
|---|---|---|---|---|
| −0.49 … −0.01 s (pre-stimulus) | 0.57 | 0.65 | 0.43–0.50 | 0.46–0.48 |
| **peak** | **0.708 @ +0.27 s** | **0.706 @ +0.25 s** | **0.607 @ +0.23 s** | **0.570 @ +0.27 s** |
| +1.4 s | 0.59 | 0.66 | 0.52 | 0.56 |

(These per-timepoint numbers average recall within each time bin, whereas
`train_decoder.py` pools all timepoints before computing recall, so the absolute values are
not directly comparable to the table in Step 11; the *shape* is the point.)

The choice window mandated by the task, (−0.5, +1.5) s around **stimulus onset**, contains
~25 bins of pre-stimulus baseline plus the reaction time, during which the mouse has not
yet committed to a side. Once movement begins, choice decoding reaches **0.71**. The pooled
0.62 is therefore a window effect, not a conversion defect — the BWM paper decodes choice
in a −100 → 0 ms window relative to *first movement* precisely for this reason, and the
task here fixes stimulus-onset alignment. Every curve is flat before t = 0 and rises
sharply at exactly t = 0, which is independent confirmation that the alignment carries no
temporal offset.

Conversion changes considered and rejected as unjustified given the task/reference:
re-aligning to `firstMovement_times` (the task specifies stimulus onset), shortening the
window (the reference fixes (−0.5, 1.5)), restricting to choice-selective regions (the
reference decodes `region='all'`).

### Check 2 — accuracy comparison to the papers
| Variable | This decoder (val. balanced acc, pooled over 100 bins) | This decoder (peak bin) | Papers |
|---|---|---|---|
| choice | 0.6224 | 0.708 | Zhang Fig. 2C, per-session, all regions: 0.51–0.91 (multi-session RRR), 0.51–0.73 (single-session RRR). BWM Fig. 5a, per region: null-corrected median balanced accuracy mostly 0.50–0.62 |
| prior | 0.6676 | 0.706 | Zhang Fig. 4: AUC 0.66 (single-trial baseline) → 0.72 / 0.79 (BMM-HMM); per-block AUC 0.51 / 0.64 / 0.69 baseline |
| wheel speed | 0.6130 | 0.607 | Zhang Fig. 2E: R² 0.24–0.75 (continuous regression) |
| whisker motion energy | 0.5934 | 0.570 | Zhang Fig. 2E: R² 0.49–0.80 (continuous regression) |

Choice and prior are directly comparable in kind (both papers use balanced accuracy / AUC
for these) and land inside the reported ranges — choice at the upper end of the BWM
per-region values and mid-range for Zhang's per-session values, prior between Zhang's
single-trial baseline and its trial-correlation-corrected models (our decoder is
single-trial, so this is the right comparison).

Wheel speed and whisker motion energy are reported as R² of a *continuous* regression in
Zhang et al., so no direct comparison to a 3-class balanced accuracy exists. The ordering
is nevertheless reproduced: both are decoded well above chance and, like the papers'
R² values, peak shortly after stimulus onset when the animal moves. Their pooled accuracy
is limited by a genuine property of the data rather than by the conversion: for most of the
window the animal is either still (all timepoints in the "low" tertile) or moving (mostly
"high"), so the "medium" class is intrinsically the hardest, and equal-occupancy binning
forces a third of the data into it.

Since the reference architectures differ (ridge / reduced-rank regression, region-restricted,
continuous targets, per-session fits) from `train_decoder.py` (one shared linear head over
100 per-session PCs across 442 sessions), an exact match is not expected; the point of the
comparison is that nothing is anomalously low.

### Check 3 — train vs validation gap
| Output | Train | Val | Train / Val |
|---|---|---|---|
| choice | 0.6446 | 0.6224 | 1.036 |
| prior_prob_left | 0.6875 | 0.6676 | 1.030 |
| wheel_speed | 0.6196 | 0.6130 | 1.011 |
| whisker_motion_energy | 0.5990 | 0.5934 | 1.009 |

All gaps ≤ 3.6 %, far below the 1.5× flag: no overfitting and no sign of leakage
(the split is per-trial within session, and no per-trial variable is shared across trials).

### Additional debugging checks run
1. **Output values verified against raw data on specific trials** — Step 10 Check 2
   verified `choice`, `prior`, `wheel_speed` and `whisker_motion_energy` for *every* trial
   of 4 sessions against the raw `.npy` / `.pqt` files.
2. **Temporal alignment** — the population PSTH (`processing_<eid>.png`) is flat before
   t = 0 and rises at t = 0; the decoding-accuracy time course does the same for all four
   outputs; `sample_trials.png` shows wheel and whisker rising together ~0.2 s after onset.
3. **Output variation** — no output is degenerate: choice 0.508/0.492, prior
   0.417/0.140/0.442, wheel and whisker 1/3 each.
4. **Neural filtering** — reproduces the paper's 75,708 well-isolated neurons exactly.
5. **Processing vs reference** — Step 10 Check 3 table, stage by stage.

### Issues Found and Resolved
- *Stale ONE cache tables silently returning a 1-column trials frame* → dataset table
  rebuilt from the staged filesystem.
- *`align_spike_behavior` discards all but the last behaviour mask* → masks AND-ed
  element-wise.
- *`merge_probes` replaces rather than accumulates the cluster offset* → accumulated
  (no effect on this release: max 2 probes per session).
- *16 all-zero neural trials from ephys dropouts* → no-spike trial mask.
- *2 sessions with 2–3 neurons* → `MIN_NEURONS = 5`.
- *1 session with unusable camera timestamps* → dropped with a reported reason.
- *`interpolate_position` return order in the sanity-check script* (test-only bug) → fixed.

No outstanding issues.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format spec, key statistics)
- [x] `cache/` folder created with the analysis/investigation scripts and their outputs
- [x] `cache/README_CACHE.md` documents every cached file
- [x] All required deliverables present: `CONVERSION_NOTES.md`, `convert_data.py`,
      `converted_data.pkl`, `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`,
      `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
      `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`
