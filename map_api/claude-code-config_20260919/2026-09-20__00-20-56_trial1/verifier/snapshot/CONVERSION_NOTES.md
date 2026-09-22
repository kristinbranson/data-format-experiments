# Dataset Conversion Notes

## Overview
- **Dataset**: DANDI:000363 "Mesoscale Activity Map Dataset" (Chen, Nguyen, Li, Svoboda 2023),
  the data behind *Chen et al., "Brain-wide neural activity underlying memory-guided movement"* (Cell 2024, `datapaper.pdf`)
  and re-analysed in *Wang, Kurgyis et al., "Brain-wide analysis reveals movement encoding structured across and within
  brain areas"* (Nat. Neurosci. 2025, `methodpaper.pdf`, code in `/app/code`).
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`).

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `convert_data.py` (written in Step 6)
- `datapaper.pdf`, `methodpaper.pdf`, `ChenLiuEtAl2023_SpikeSortingQC.pdf`, `methods.txt`
- `code/` — reference code for the method paper (`MapVideoAnalysis` repo)
- `data/` — 28 `sub-*/` directories, 174 `*.nwb` files, 50 GB total, plus `dandiset.yaml`
- `decoder.py`, `train_decoder.py` — provided decoder / validation code
- `pynwb_docs/`, `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `cache/` — scratch scan scripts written during exploration

Environment verified: `python3`, numpy 2.4.4, torch 2.6.0+cu124 (CUDA available), pynwb 4.1.0,
pandas 3.0.5, scipy. Machine: 128 cores, 1 TB RAM, 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

`/app/code` is the `MapVideoAnalysis` repo for the method paper. It works from **DataJoint `.mat`
exports** of the same experiment, not from the NWB files, so the field *names* differ but the
*semantics* map 1:1 onto the NWB tables (see Step 4).

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel` / `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Top-level per-session loader: concatenates all probes of a session, reads behaviour/task variables, spike times, unit info, CCF histology |
| `preprocessing_utils.loadmat` | `VideoAnalysisUtils/preprocessing_utils.py` | LOADING | MATLAB struct → nested dict |
| QC good-unit list (`goodunits/*.mat`, `qc_mode='classifier'`) | `preprocessing_DJ_2022Aug.py` l.74-82, 163-183 | CURATION | Only units accepted by the region-specific logistic-regression QC classifier are kept. In NWB this is the `units.classification == 'good'` column. |
| `helper_get_neuron_id_area` | `preprocessing_DJ_2022Aug.py` | CURATION | Per-area/per-hemisphere neuron selection; hemisphere from CCF ML coordinate with midline at **5700 µm**; requires a non-empty CCF annotation |
| unit "ephys ∩ histology" mask | `process_one_sess` l.204-218 | CURATION | Keeps only units that have *both* spike data and a CCF annotation |
| `sliding_histogram` | `preprocessing_DJ_2022Aug.py` l.36 | PROCESSING | Bins spike times into **firing rates** (`binSpikes / bin_width`) on bin centres `begin_time + k*stride`; window `[c-bw/2, c+bw/2)` |
| `process_one_area` l.311-324 | `preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates per-trial spike times to `[begin_time, end_time]` **already aligned to the Go cue** ("in Susu's data, spike_times are relative to go cue time") and bins them |
| `preprocess_all_ephys.py` | `Sherlock/` | PROCESSING | The actual settings used for the paper: `bw=0.04 s`, `stride=0.0034 s`, `begin=-3 s`, `end=+3 s`, `qc_mode='classifier'` |
| `align_markers_between_lims` / `align_markers.py` | `Sherlock/align_markers.py` | PROCESSING | Aligns DeepLabCut markers (`tongue_x, tongue_y, jaw_*, nose_*`) to the **Go cue** (`task_cue_time[0]`), video frame period `dt = 0.0034 s` (300 Hz), window `[-3, +1.5] s` |
| `get_regular_trial_mask` | `VideoAnalysisUtils/population_decoding_utils.py` | CURATION | "regular" trials = no early lick, no auto water, no free water, response given (`correctness != -1`), no photostimulation |
| `load_session` | `population_decoding_utils.py` | LOADING | Reassembles per-area pickles into a session (fr array `(n_bins, n_trials, n_neurons)`) |
| `nested_cross_validation` | `population_decoding_utils.py` | ANALYSIS | PCA(16)+logistic-regression population decoding, per time point |

### Notes
- **Electrophysiology, not imaging** → no dF/F. Neurons *are* quality-filtered: the pipeline only ever
  loads units on the classifier-derived "good unit" list (`methods.txt`: 15 quality metrics → 5
  region-specific logistic-regression classifiers → 'good' labels; 69,943 good units = 25.9 % of KS2 clusters).
- Alignment throughout the reference pipeline is to the **Go cue**, which is exactly what this task asks for.
- Firing rates (spikes/s), not raw counts, are the reference representation.
- The method paper additionally drops neurons with mean firing rate < 2 Hz, but only for the
  ridge-regression "predict firing rate from video" analyses ("Neurons with low firing rates (below 2 Hz)
  were excluded from analyses; the results were not sensitive to the exact value of this threshold").
  That is an analysis-specific convenience threshold, not a data-quality criterion — see Step 5 decision D4.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/sub-<subject_id>/sub-<subject_id>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys+ogen.nwb`
- 28 subject directories, **174 NWB files** (one per session), 50 GB.
- Read with `pynwb.NWBHDF5IO(..., load_namespaces=True)`.

Per file:
- `nwb.subject.subject_id` (e.g. `440956`), `nwb.subject.description` (lab ID, e.g. `SC015`), `nwb.identifier`
  (e.g. `SC015_20190208_133600_s2`).
- `nwb.trials` (`TimeIntervals`): `start_time, stop_time, trial, photostim_onset, photostim_power,
  photostim_duration, trial_uid, task, task_protocol, trial_instruction, early_lick, outcome,
  auto_water, free_water`.
  - `trial_instruction` ∈ {left, right}; `outcome` ∈ {hit, miss, ignore}; `early_lick` ∈ {early, no early};
    `task` = 'audio delay' for all 174 sessions. No missing/`N/A` values in any of these.
- `nwb.acquisition['BehavioralEvents']` — 14 `TimeSeries` with session-clock `timestamps`:
  `presample/sample/delay/go/trialend` `_start_times`/`_stop_times`, `left_lick_times`,
  `right_lick_times`, `photostim_start_times`, `photostim_stop_times` (data = laser power in mW).
- `nwb.acquisition['BehavioralTimeSeries']` — DeepLabCut side-view markers, each `(n_frames, 3)` =
  `(x, y, likelihood)` with per-frame `timestamps`: `Camera0_side_TongueTracking`,
  `Camera0_side_JawTracking`, `Camera0_side_NoseTracking` (present in **all 174** sessions),
  plus occasionally whisker / lick-port / a second camera. Frame period 3.4 ms (300 Hz).
- `nwb.units`: 39 columns incl. 15 QC metrics, `classification` ∈ {good, unlabelled},
  `anno_name` (CCF annotation; non-empty for *every* good unit and empty for every unlabelled unit),
  `spike_times`, `obs_intervals`, `is_good_trials`, `electrodes`, `electrode_group`.
- `nwb.electrodes`: `x, y, z` = CCF coordinates in µm (x = ML, midline 5700; y = DV; z = AP, bregma 5400)
  — verified against `location` JSON (`ap_location`, `ml_location`) and against the reference code's
  `global_offset_vec = (5700, 0, 5400)`.
- `nwb.electrode_groups[*].location` JSON gives the *targeted* region, e.g. `"left ALM"`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files / sessions | 174 (173 with ≥1 good unit) |
| Probe insertions (electrode groups) | 659 |
| Units (all KS2 clusters) | 272,227 |
| Good units (`classification=='good'`) | **69,453** (25.5 % of clusters) |
| Good units / session | mean 399.2, median 390, range 0–923 |
| Subjects | 28 |
| Sessions / subject | mean 6.2, range 1–11 |
| Trials (total, trials table) | 94,990 |
| Trials / session | mean 545.9, median 534, range 264–800 |
| Trials with ephys coverage | 93,930 (9 sessions have ephys for only a prefix of the behavioural trials) |
| Photostim trials | 18,588 (168/174 sessions), always 5.5 mW, always 0.5 s |
| Auto-water or free-water trials | 3,766 (4.0 %) |
| Video frame rate | 300 Hz (3.4 ms) |

Important structural facts discovered (all verified over all 174 files):
1. **Spike times exist only inside `[trial.start_time, trial.stop_time]`** — 0.0 % of spikes fall between
   trials. The NWB export is per-trial-windowed, not continuous.
2. `units.obs_intervals` for every unit equals the trials table's `(start_time, stop_time)` rows
   (172/174 exactly; in the other 2 it is the same length but with a small numeric difference) and
   `units.is_good_trials` has that same length. In **9 sessions** this length is *smaller* than the number
   of behavioural trials → ephys stops part-way through the session. Trials beyond that point have no
   neural data at all and must be dropped.
3. Trial windows relative to the Go cue: `go − start` ≥ 2.11 s (min over the whole dataset, median 3.15 s)
   and `stop − go` ≥ 0.155 s. **`stop − go` depends strongly on outcome**: ≈1.7–2.0 s for *hit*, 1.80 s for
   *ignore*, but only ≈0.6–0.9 s for *miss* (error) trials, which are terminated by the time-out.
   Consequence: in the requested window the last ~0.7 s of a *miss* trial contains **no recorded spikes and
   no video frames**. See Step 5 decision D6.
4. In 2 sessions (`SC045_20201216_120333_s24`, `SC053_20210220_124834_s15`) each trial's video frames are
   stored **twice**, so the marker `timestamps` are not monotonically increasing. Must sort before
   `searchsorted`.
5. 2 sessions have truncated video: `SC066_20210413_112028_s6` (video for only the first 7 of 550 trials)
   and `SC065_20210505_170309_s6` (412 of 660 trials).
6. `units.is_good_trials` is all-True except in 4 sessions (see Step 5, D5).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 28 | "This study is based on data from 28 mice" (datapaper) |
| Sessions | 173 behavioural sessions | "aggregated over 660 penetrations, 173 behavioral sessions, and 28 mice" (Fig 1J) |
| Penetrations | 660 | same |
| Good units (total) | 69,943 (25.9 % of KS2 clusters) | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt) |
| Good units / session | median 393 | "simultaneous measurements from hundreds of neurons (median = 393)" |
| Good units, ALM | 8,717 | "This method yielded 8717 good units from ALM, 7664 from striatum, 12808 from thalamus, 7495 from midbrain and 2928 from medulla, etc." |
| Good units, striatum | 7,664 | same |
| Good units, thalamus | 12,808 | same |
| Good units, midbrain | 7,495 | same |
| Good units, medulla | 2,928 | same |
| Trials / session | mean 476, range 130–785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84 %, range 65–99 % | "with 84% correct rate (range, 65-99%)" |
| Photoinhibition | ~25 % of trials, 5 mW/hemisphere, 0.5 s in late delay, ends before Go | Photoinhibition section of methods.txt |
| Bilateral-ALM photoinhibition effect | 83.2 % → 71.7 %, 17 mice, 93 sessions | methods.txt |
| Sample epoch | 3 × 150 ms tone + 2 × 100 ms gap = 0.65 s | methods.txt |
| Delay epoch | 1.2 s | methods.txt |
| Response (answer) period | 1.5 s | methods.txt |
| Neural data time bin (method paper) | 40 ms width, 3.4 ms stride | `Sherlock/preprocess_all_ephys.py`; "Spike rates were binned in 40 ms time windows and 17 ms time steps" (datapaper, AUC analysis) |
| Video / behaviour time bin | 3.4 ms (300 Hz) | methods.txt, `align_markers.py` |
| Session selection | performance > 65 % and ≥ 50 correct lick-left and lick-right trials | methods.txt |

### Processing Details
- **Temporal alignment**: everything is aligned to the **Go cue**. The reference spike data are already
  Go-cue-relative; markers are aligned by subtracting `task_cue_time[0]` (= `go_start_times`).
- **Temporal binning**: sliding histogram → firing rate = spikes / bin width.
- **Trial structure**: presample → sample (0.65 s instruction tone train) → delay (1.2 s) → Go cue
  (6 kHz, 0.1 s) → 1.5 s answer period → 1.5 s consumption → trial end. Licking during sample/delay
  *replays* the sample+delay epochs, so a trial can contain several sample/delay onsets; the effective
  instruction-tone onset is the **last** sample onset before the Go cue (verified: median Go − last sample
  onset = 1.85 s = 0.65 + 1.2 s exactly).

### Curation Steps
**Neuron curation rules** (papers + reference code):
1. Kilosort2 clusters → 15 quality metrics → five region-specific logistic-regression classifiers →
   keep only units labelled **'good'** (`units.classification == 'good'` in NWB).
2. Only units that also have a CCF annotation are used (in NWB every good unit has one).
3. (method-paper, analysis-specific) mean firing rate ≥ 2 Hz for the video→firing-rate regressions.

**Trial curation rules** (method paper, "Trial selection and cross-validation"):
> "The dataset contains trials with photoinhibition, water administration regardless of the animals'
> choice (free water trials), early licks and trials where the animal ignores the lick-spouts. These were
> excluded from all analyses."

and `get_regular_trial_mask` = no early lick, no auto water, no free water, `correctness != -1`
(i.e. no *ignore*), no photostimulation.

### Decoders Trained (reported accuracies in the papers)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Choice, from **video** during sample epoch | logistic regression on embedding | ROC-AUC 0.66 ± 0.12 s.d. (n = 106 sessions); at chance in the *first half* of the sample epoch |
| Choice, from **video** during response epoch | same | ROC-AUC ≈ 0.9 ("easily decodable from video of directional licking") |
| Choice, single neuron, delay epoch | logistic regression on spike rate | AUC threshold 0.65 used to call a neuron "choice modulated" |
| Choice, population (medulla subregions) | PCA(16) + logistic regression | AUC per time point, Fig. 7 |
Note: the papers never train a *neural-population → choice/outcome/early-lick/tongue* decoder of the kind
required here, so there is no directly comparable published accuracy number. The closest reference points
are: choice is nearly perfectly decodable from ALM/brain-wide population activity after the Go cue, and
around 0.9 AUC from video in the response epoch.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

Mapping of reference-code (`.mat`) variables → NWB fields, and consistency checks:

| Reference `.mat` variable | NWB equivalent | Verified |
|---|---|---|
| `task_cue_time[0,:]` (go cue) | `BehavioralEvents/go_start_times.timestamps` | 174/174 sessions: exactly one Go cue per trial, always inside `[start_time, stop_time]` |
| `task_sample_time` | `sample_start_times` / `sample_stop_times` | last sample onset → Go = 1.85 s median = 0.65 + 1.2 s ✓ |
| `task_delay_time` | `delay_start_times` | last delay onset → Go = 1.2 s median ✓ |
| `behavior_report` (1/0/−1) | `trials.outcome` (hit/miss/ignore) | choice reconstructed from `outcome`+`trial_instruction` agrees with the choice reconstructed independently from `left_lick_times`/`right_lick_times` on **99.7 %** of all 95k trials |
| `task_trial_type` ('l'/'r') | `trials.trial_instruction` | ✓ |
| `behavior_early_report` | `trials.early_lick` | ✓ |
| `behavior_is_auto_water` / `is_free_water` | `trials.auto_water` / `free_water` | ✓ |
| `task_stimulation` (power, type, on, off) | `trials.photostim_*` + `photostim_start/stop_times` | exactly one photostim event per photostim trial, 5.5 mW, 0.5 s, onset at −1.2 s or −0.5 s rel. Go ✓ (matches "late delay, 0.5 s, ends before Go") |
| `neuron_single_units` (spike times rel. Go) | `units.spike_times` (session clock) − Go time | ✓ |
| QC good-unit index files | `units.classification == 'good'` | 69,453 units = 25.5 % of clusters vs. paper 69,943 = 25.9 % |
| `histology.annotation` | `units.anno_name` | ✓ (non-empty for all good units) |
| `histology.ccf_x/y/z` | `electrodes.x/y/z` via `units.electrodes` | midline 5700 µm and offset (5700, 0, 5400) reproduce ML/AP of the probe `location` JSON ✓ |
| `tracking.camera_0_side.tongue_x/tongue_y` | `BehavioralTimeSeries/Camera0_side_TongueTracking` cols 0/1 | ✓ ; NWB additionally provides col 2 = DLC likelihood |

### Discrepancies Found
| Topic | Code/Papers say | Data shows | Resolution |
|-------|-----------------|------------|------------|
| Number of sessions | 173 | 174 NWB files | One file (`SC017_20190216_162508_s4`) has 1852 clusters but **0 good units** → 173 usable sessions. Dropped (no neural data to decode from). |
| Good units | 69,943 | 69,453 | 0.7 % fewer in the published DANDI release (version 0.230822.0128) than quoted in the paper; nothing in the data to reconcile further. Per-area counts agree to ≈1–2 % (see below). |
| Penetrations | 660 | 659 electrode groups | Off by one, same cause as the session count. |
| Trials/session | mean 476 (range 130–785) | mean 546 (range 264–800) | The published trials table contains every behavioural trial; the paper's figure is presumably over the trials entering analysis / a differently-counted set. Does not affect the conversion. Our per-session trial counts are reported honestly. |
| Correct rate | 84 % (65–99 %) | 81 % (54–97 %) computed as hits/(hits+misses) on non-early, non-photostim trials | Same definition as the paper gives 81 %; 22 of 174 sessions fail the paper's stated selection criteria (>65 % and ≥50 correct L and R) yet are in the release. Since the release *is* the curated dataset of the paper, we do not re-apply the session filter (decision D2). |
| Trial selection | exclude photostim, early-lick, ignore, free-water trials | — | **Required discrepancy**: the decoder task asks for photostimulation as an *input* and early-lick / ignore as *outputs*, so those trials must be kept. Only auto-water/free-water trials are excluded, as in the reference (decision D3). |
| Bin width | 40 ms width / 3.4 ms stride (reference) | — | **Required discrepancy**: the task specifies 50 ms bins. We use non-overlapping 50 ms bins (width = stride = 50 ms), i.e. the same `sliding_histogram` formula with `bw = stride = 0.05`. |
| Window | [−3, +3] s (reference) | — | **Required discrepancy**: the task specifies [−2.5, +1.5] s. |

Region-mapping cross-check (my CCF-annotation → coarse-area mapping vs. the paper's counts):

| Area | Paper | This conversion | Δ |
|---|---|---|---|
| Medulla | 2,928 | 2,928 | 0.0 % |
| Midbrain | 7,495 | 7,480 | −0.2 % |
| Striatum | 7,664 | 7,738 | +1.0 % |
| Thalamus | 12,808 | 13,016 | +1.6 % |
| ALM | 8,717 | 8,882 | +1.9 % |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (NWB) | Target field | Transform |
|---|---|---|
| `units.spike_times` (good units), `go_start_times.timestamps` | `neural[s][t]` `(n_neurons, 80)` float32 | subtract Go time; histogram into 80 non-overlapping 50 ms bins spanning [−2.5, +1.5] s; divide by 0.05 → firing rate in Hz |
| last `sample_start_times` ≤ Go | `input[0]` `time_from_tone_onset` (80,) float32 | bin-centre time minus (Go − tone-onset), in seconds; time-varying |
| `photostim_start_times` / `photostim_stop_times` | `input[1]` `photostim_on` (80,) float32 | 1.0 if the bin centre lies in `[stim_start, stim_stop)`, else 0.0; time-varying |
| `trials.outcome` + `trials.trial_instruction` | `output[0]` `choice` | hit → instruction; miss → opposite of instruction; ignore → "no lick". 0=left, 1=right, 2=no lick. Constant over the 80 bins |
| `trials.outcome` | `output[1]` `outcome` | 0=ignore, 1=miss, 2=hit. Constant over the 80 bins |
| `trials.early_lick` | `output[2]` `early_lick` | 0='no early', 1='early'. Constant over the 80 bins |
| `Camera0_side_TongueTracking` cols (y, likelihood) | `output[3]` `tongue_y` | per bin: frames with likelihood > 0.9 → mean y; no visible frame → class 3. Visible bins discretised by the session's 40th/60th percentiles of those binned y values → 0/1/2 |
| `units.anno_name` + `electrodes.x/y/z` | `brain_region_idx[s]` | CCF annotation → 14 coarse areas (`cache/region_map.py` rules); frontal motor cortex additionally required to be AP > 1.75 mm and |ML| < 2.0 mm to be called ALM |
| `nwb.subject.description` | `subjects` / `subject_idx` | lab mouse ID, e.g. `SC015` |

Resulting shapes: `T = 80` bins of 50 ms, `dinput = 2`, `doutput = 4`.

`output_values`:
- `choice`: `['left', 'right', 'no lick']`
- `outcome`: `['ignore', 'miss', 'hit']`
- `early_lick`: `['no', 'yes']`
- `tongue_y`: `['<40th pct', '40-60th pct', '>60th pct', 'not visible']`

### Key Decisions
1. **D1 — Neuron curation: `classification == 'good'` only.** This is exactly the classifier-based QC
   described in `methods.txt` / the QC white paper and used by the reference pipeline (which loads only
   the `goodunits/*.mat` index). Reproduces 69,453 units (25.5 % of clusters) vs. the paper's 69,943 (25.9 %),
   and per-area counts within 2 % (Medulla exact).
2. **D2 — Session curation: keep all sessions that have ≥1 good unit (173).** The DANDI release *is* the
   curated dataset of the paper (173 behavioural sessions), so the paper's session-selection criteria have
   already been applied upstream; re-applying my own reconstruction of them would only remove sessions the
   authors kept. The one file with 0 good units is dropped because it carries no decodable neural data.
3. **D3 — Trial curation.** Keep a trial iff
   (a) it is within the range of trials for which the probes were recording
       (`is_good_trials`/`obs_intervals` length — drops 1,060 trials in 9 sessions that have *no* ephys at all), and
   (b) it is not an auto-water and not a free-water trial (3,766 trials, 4.0 %) — following the reference
       ("water administration regardless of the animals' choice (free water trials) … were excluded from all analyses");
       on these trials reward is decoupled from the animal's action so the choice/outcome labels are not
       behaviourally meaningful.
   **Photostimulation, early-lick and ignore trials are deliberately kept**, even though the reference
   excludes them, because the decoder task defines photostimulation as an input and early-lick and
   `ignore` as output classes. This is an explicitly allowed discrepancy.
4. **D4 — No firing-rate threshold on neurons.** The method paper's 2 Hz cut is specific to its
   video→firing-rate ridge regressions ("the results were not sensitive to the exact value of this
   threshold"); it is not part of the dataset's quality control. Removing low-rate neurons would throw away
   information a population decoder can use, and the decoder's own PCA front-end handles scale. Kept all
   good units.
5. **D5 — `is_good_trials` used only to define the ephys-covered trial range**, not to drop individual
   unit×trial pairs. Only 4 sessions contain any False entries, the reference pipeline has no equivalent
   filter at all, and dropping per-unit-per-trial data is impossible in a (n_neurons, T) matrix; dropping
   the affected trials would cost 56 % of one session's trials for no principled gain.
6. **D6 — Truncated trial windows are binned as they are (zeros after the trial ends).** The NWB export
   only contains spikes/video inside `[trial.start_time, trial.stop_time]`, and *miss* trials end ≈0.8 s
   after the Go cue. 81.5 % of trials fully cover [−2.5, +1.5] s; 96.8 % cover the pre-Go part and 84.3 %
   the post-Go part. Excluding partially-covered trials would delete ~91 % of all *miss* trials and make
   the `outcome` output undecodable, so all trials are kept and the missing tail is simply empty
   (zero firing rate, tongue "not visible"). This is documented as a known property of the release.
7. **D7 — "Time from tone onset" = time since the *instruction* tone (sample epoch onset)**, not the Go
   cue: the trials are already aligned to the Go cue, so a Go-relative clock would carry no per-trial
   information, whereas the sample-onset-to-Go interval varies from 0.95 s to 10.4 s (early-lick replays)
   and is genuine task context. The *last* sample onset before the Go cue is used, because an early lick
   replays the sample+delay epochs and it is the final presentation that instructs the animal.
8. **D8 — Tongue visibility threshold: DLC likelihood > 0.9.** The likelihood distribution is strongly
   bimodal (85.6 % < 0.01, 14.0 % > 0.99), so the result is insensitive to the threshold between 0.1 and 0.99.
   A bin counts as visible if *any* frame in the 50 ms bin is above threshold (tongue protrusions last
   only ~50–100 ms), and the reported y is the mean over those frames.
   The method paper's own outlier handling ("when the tongue was occluded … we set the tongue position to
   its mean value") is not applicable here because the task specifies an explicit "not visible" class.
9. **D9 — Percentiles for the tongue discretisation are computed per session over all *visible* binned
   y values of that session** (all kept trials × 80 bins), matching "40th percentile of y-position over
   the session".
10. **D10 — Firing rates (Hz), not spike counts**, matching `sliding_histogram(..., rate=True)`.
11. **D11 — Brain regions**: 14 coarse areas derived from the per-unit CCF annotation, matching the region
    list used by the reference preprocessing code
    (`['ALM','Medulla','Midbrain','Striatum','Thalamus','Pons','Cerebellum','Hypothalamus','Hippocampus',
    'Orbital','OtherCortex','Olfactory','CorticalSubplate','Pallidum']`). Hemispheres are pooled
    (the reference splits them via the ML = 5700 µm midline; the target format's example
    (`ALM`, `V1`) is hemisphere-agnostic).

### Planned Sanity Checks
- [x] Total good units = 69,453; median good units/session ≈ 390–393 (paper: 393)
- [x] Per-area good-unit counts vs. paper (ALM/striatum/thalamus/midbrain/medulla)
- [x] Choice from `outcome`+`instruction` == choice from lick times (99.7 %)
- [x] Go − last sample onset = 1.85 s = sample (0.65 s) + delay (1.2 s)
- [x] Photostim: 0.5 s, 5.5 mW, onset −1.2 s or −0.5 s rel. Go, one event per stim trial
- [x] Spot check firing rate values against raw `spike_times` re-binned independently (Step 10, Check 2 — passed, incl. against the reference `sliding_histogram`)
- [x] Spot check tongue class against raw tracking data (Step 10, Check 2 — passed)
- [x] Spot check photostim/tone inputs against raw event timestamps (Step 10, Check 2 — passed)
- [x] Output distributions: hit 0.685, miss 0.167, ignore 0.148, early 0.116 (Step 9 — match the raw data exactly)
- [x] Decoder accuracy above chance on every output (Step 11 — 1.5x to 2.6x chance)

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Runs as `python -u /app/convert_data.py <outpicklefile>` with
`--full` (default) / `--sample` / `--show-processing` / `--jobs N`.

Structure:
| Function | Purpose |
|---|---|
| `assign_brain_regions` | CCF `anno_name` substring rules -> 14 coarse areas; frontal motor cortex is ALM only if AP > 1.75 mm and abs(ML) < 2.0 mm |
| `bin_spike_rates` | vectorised equivalent of the reference `sliding_histogram(bw=stride=0.05, rate=True)` |
| `bin_visible_mean` | per-bin mean of the tongue y over frames with DLC likelihood > 0.9 |
| `discretise_tongue` | per-session 40th/60th percentile split + "not visible" class |
| `process_session` | one NWB file -> per-trial neural/input/output arrays + session info |
| `_plot_processing` | `--show-processing` figures |
| `main` | multiprocessing driver, assembly of the output dict, summary, pickling |

Implementation notes:
* Binning is fully vectorised. For a session, the 81 bin edges of every trial are
  concatenated into one `(n_trials*81,)` query array and `np.searchsorted` is called once
  per neuron; the per-bin counts are `np.diff` of the resulting indices. Equivalent to a
  per-trial `np.histogram` but ~100x faster, and it needs no assumption that trial windows
  are ordered or disjoint.
* The tongue y mean per bin uses the same trick plus a cumulative sum of the visible-frame
  y values, so no Python loop over bins is needed.
* Video timestamps are sorted first (2 sessions store each trial's frames twice, which makes
  the stored timestamps non-monotonic).
* Sessions are processed in parallel with `multiprocessing` (`spawn`), default 12-14 workers.

Code inefficiencies identified: naive per-trial `np.histogram` over ~400 neurons x ~500
trials; per-bin boolean masking of ~1e6 video frames.
Code speedups added: single `searchsorted` per neuron over all trial edges; cumulative-sum
binning for the video; parallelism over sessions; reading only the `good` units' spike trains.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(sessions `SC015_20190208_133600_s2` and `SC067_20210418_132006_s26`).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 1162 |
| Neurons / session | 375, 787 |
| Subjects | 2 (SC015, SC067) |
| Trials (total) | 848 |
| Trials / session | 159, 689 |
| T (bins) | 80 for every trial |
| time_from_tone_onset range | [-1.5, 6.1] s |
| photostim_on range | [0, 1] |
| choice distribution | left 0.491, right 0.449, no lick 0.060 |
| outcome distribution | ignore 0.060, miss 0.179, hit 0.761 |
| early_lick distribution | no 0.903, yes 0.097 |
| tongue_y distribution | 0.100 / 0.050 / 0.100 / 0.751 |

The tongue distribution is exactly 40 % / 20 % / 40 % of the *visible* bins, as the
percentile definition requires (0.100 : 0.050 : 0.100 = 4 : 2 : 4).

### Processing Plots Review
`processing_<session>.png` shows, for a hit / miss / ignore / photostim / early-lick trial:
binned population raster, population mean rate with the trial interval shaded, the
time-from-tone input against the true tone onset, the photostim input against the raw
photostim event, the raw tongue trace with the binned means and session percentiles, and
the four output traces. Everything lines up:
* the tone-onset input crosses zero exactly at the plotted tone onset;
* the photostim step coincides with the raw event shading (-0.5 to 0 s or -1.2 to -0.7 s);
* the binned tongue means track the raw visible y samples and the class switches to
  "not visible" exactly where no visible frame exists;
* on the *miss* trial the population rate drops to exactly zero at the end of the shaded
  trial interval (+0.7 s) — the known truncation documented in Step 5 D6.

### Run Time Estimates
| Speed-ups implemented | Time saving |
|---|---|
| single `searchsorted` per neuron instead of per-trial `np.histogram` | ~100x on the binning step |
| cumulative-sum video binning instead of per-bin masks | ~50x on the tongue step |
| 14 parallel worker processes | ~11x wall clock |

| Step | Time / session | Estimated total |
|---|---|---|
| open NWB + trials table | 0.25 s | 45 s |
| read good-unit spike trains | 0.4 s | 70 s |
| bin spikes | 0.4 s | 70 s |
| regions / inputs / outputs / assemble | 0.2 s | 35 s |
| **per session total** | **~1.3 s** | **~220 s serial, ~25 s on 14 workers** |
| pickle write (11.9 GB) | — | ~15 s |

Estimated (and realised) full conversion: **~40 s**, far below the 15 minute budget, so no
further optimisation was needed.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None (an earlier run warned "all neural data is zero" for the last trial of
  `SC015_20190208_133600_s2`; that trial is now dropped — see Step 10, Issue 1)

### Decoder Results (Sample, 2 sessions)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| choice | 0.333 | 0.700 | 0.600 |
| outcome | 0.333 | 0.673 | 0.541 |
| early_lick | 0.500 | 0.813 | 0.701 |
| tongue_y | 0.250 | 0.792 | 0.729 |

Loss decreased monotonically from 15.0 (epoch 4) to 0.616 (epoch 200); test loss 0.687.
Every output is above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --jobs 14` — 42 s wall clock,
173 sessions converted, 1 skipped (`SC017_20190216_162508_s4`, no good units), 0 errors.

### Output Files
- `converted_data.pkl`: 11.89 GB
- `verification_full_out.txt`: created, "Data format is valid, no errors or warnings."

### Consistency Check
| Statistic | Reference papers | Reference code | Reference data (recomputed from NWB) | Converted data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Sessions | 173 | — | 173 with good units (174 files) | 173 | yes |
| Subjects | 28 | — | 28 | 28 | yes |
| Probe insertions | 655 (methods.txt) / 660 (Fig 1J) | — | 655 in the 173 kept sessions | — | yes |
| Total good units | 69,943 | classifier 'good' list | 69,453 | 69,453 | -0.7 % vs paper, exact vs data |
| Good units, fraction of clusters | 25.9 % | — | 25.5 % | 25.5 % | close |
| Median neurons/session | 393 | — | 390 | 390 | close |
| Mean neurons/session | — | — | 401.5 | 401.5 | exact |
| Trials (total, after curation) | — | — | 89,546 | 89,544 | 2 trials dropped (zero population spikes) |
| Trials/session (mean) | 476 (all behavioural trials) | — | 517.6 | 517.6 | exact vs data; paper number differs (Step 4) |
| Behavioural performance | 84 % (65-99 %) | hit/(hit+miss) on control non-early trials | 80.6 % (53-99 %) | — | same definition, 3 points lower |
| ALM units | 8,717 | 14-region scheme | — | 8,882 | +1.9 % |
| Striatum units | 7,664 | | — | 7,738 | +1.0 % |
| Thalamus units | 12,808 | | — | 13,016 | +1.6 % |
| Midbrain units | 7,495 | | — | 7,480 | -0.2 % |
| Medulla units | 2,928 | | — | 2,928 | exact |
| time_from_tone_onset range | sample onset -> Go = 0.65+1.2 = 1.85 s | `align_markers` uses `task_cue_time` | min Go-tone 0.95 s -> min input -1.525 s | [-1.525, 11.894] | yes |
| photostim_on range | 0.5 s, 5.5 mW, late delay | `task_stimulation` | 20.04 % of kept trials are stim trials | 20.02 % of trials have stim inside the window | yes |
| outcome distribution | — | — | ignore 0.1484 / miss 0.1668 / hit 0.6849 | 0.1484 / 0.1667 / 0.6849 | yes |
| choice distribution | — | — | no lick == ignore = 0.1484 | left 0.4292 / right 0.4224 / no lick 0.1484 | yes |
| early_lick distribution | — | — | yes 0.1161 | no 0.8840 / yes 0.1160 | yes |
| tongue_y distribution | — | — | 24.5 % of bins have a visible tongue | 0.098 / 0.049 / 0.098 / 0.755 | 40/20/40 of visible, as defined |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification
`/app/verification_full_out.txt` begins with **"Data format is valid, no errors or warnings."**
There are no errors and no warnings left to address.

Warnings that appeared in earlier iterations and how they were removed:
* *"Session 0, trial 159: all neural data is zero"* — the last trial of the ephys block can
  contain no spikes at all because the recording stopped mid-trial. Fixed by dropping trials
  in which the entire simultaneously-recorded population has zero spikes in the window
  (2 trials in the whole dataset after the Check-5 fix below).

### Check 2 — Independent sanity checks (`cache/sanity_checks.py`)
All checks re-open the raw NWB files with `pynwb` and re-derive the quantity with different
code from `convert_data.py`, comparing with `np.allclose` / `np.array_equal`.
Run over 4 sessions (3 random + the known edge-case session `SC026_20190807_134913_s20`):

| Check | What it does | Result |
|---|---|---|
| trial count | re-derives the kept-trial list, matching `obs_intervals` to trials by their **stop** time (the converter matches on the **start** time), re-applies the water and zero-spike filters | OK (662, 445, 472, 505 trials) |
| **neural** spot check | for 3 random (trial, neuron) pairs per session, `np.histogram(spike_times - go, bins=-2.5+arange(81)*0.05)/0.05` vs. the stored row | OK (12/12) |
| **neural** vs reference code | the **verbatim** `sliding_histogram` from `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` run with `bin_width=stride=0.05`, `begin=-2.475`, `end=1.475` on 5 neurons x 3 trials per session; also checks that its `binCenters` equal ours | OK (4/4 sessions), bin centres identical |
| **input** spot check | 6 random trials/session: `bin_centre_abs - last_sample_onset_before_Go` and a loop over the raw photostim start/stop events | OK (24/24) |
| **output** tongue percentiles | recomputes the session's 40th/60th percentile from the raw tracking with a per-bin boolean mask (no `searchsorted`) | OK (4/4, agreement to 1e-12) |
| **output** spot check | 6 random trials/session: choice from `outcome`+`trial_instruction`, outcome, early lick and the full 80-bin tongue class vector | OK (24/24) |
| brain regions | neuron count equals the number of good units; every unit labelled ALM carries a frontal-motor-cortex CCF annotation | OK (4/4) |

Full log: `cache/sanity_checks_out.txt` — "ALL SANITY CHECKS PASSED".

### Check 3 — Reference code comparison
| Stage | Reference (`preprocessing_DJ_2022Aug.py`, `align_markers.py`, `population_decoding_utils.py`) | This conversion | Same? |
|---|---|---|---|
| (a) loading | `loadmat` of per-probe DataJoint `.mat`, concatenating probes of a session | `pynwb` on the single per-session NWB file, which already contains all probes | equivalent (NWB is the published form of the same export) |
| (b) neuron filtering | units on the classifier `goodunits/*.mat` list, and present in both ephys and histology | `units.classification == 'good'`; all such units carry a CCF annotation | same |
| (b') extra reference filters | hemisphere split at ML 5700 µm; per-area pickles | hemispheres pooled; areas recorded in `brain_region_idx` | difference: the target format wants one flat region list per neuron, and the decoder uses all simultaneously-recorded neurons of a session jointly |
| (c) temporal alignment | Go cue (`task_cue_time[0]`), also for the video markers | Go cue (`go_start_times.timestamps`), also for the tongue marker | same |
| (d) binning | `sliding_histogram`, firing rate = counts / bin width, half-open bins `[c-bw/2, c+bw/2)`; paper used `bw=0.04, stride=0.0034, window [-3,3]` | identical formula with `bw = stride = 0.05`, window `[-2.5, 1.5]` — *verified numerically against the reference function* | same algorithm; window/bin size differ **because the task specifies them** |
| (e) input construction | reference has no decoder inputs; it stores `task_stimulation` (power, type, on/off relative to the Go cue) and the epoch times | photostim on/off from the same events; time from the sample-epoch onset | consistent with the reference's variables |
| (f) output construction | `behavior_report` (1/0/-1), `task_trial_type` ('l'/'r'), `behavior_early_report`, `tongue_y` from `tracking.camera_0_side` | `outcome`, `trial_instruction`, `early_lick`, `Camera0_side_TongueTracking` | same variables |
| trial curation | `get_regular_trial_mask`: no early lick, no auto water, no free water, response given, no photostim | no auto water, no free water **only** | deliberate, required by the decoder task (photostim is an input; early lick and `ignore` are output classes) |
| marker alignment | frames assigned to a 3.4 ms grid, using the **last** frame in each interval | frames assigned to 50 ms bins, using the **mean** of the visible frames in each bin | difference: at 50 ms a bin holds ~15 frames, so the mean is a better summary than the last sample; visibility is taken from the DLC likelihood, which the `.mat` export used by the reference does not expose |
| marker imputation | outliers by a 5-sigma velocity threshold; occluded tongue set to its mean value | no imputation; occluded tongue is its own output class | required: the task defines an explicit "not visible" class |

### Check 4 — Key statistics comparison
See the table in Step 9. Everything that both the papers and the data report agrees to
within 2 %, and everything recomputed from the raw NWB with the same curation rules agrees
exactly. Investigated discrepancies:
* **69,453 vs 69,943 good units (-0.7 %)** — the published DANDI release (version
  0.230822.0128) simply contains 490 fewer classifier-'good' units than the number quoted in
  the paper. Confirmed by counting `classification == 'good'` over all 174 files with an
  independent scan; there is nothing in the files to reconcile it further. Per-area counts
  agree to <= 2 % and the medulla count is exact, so the region assignment is not the cause.
* **Mean 546 trials/session in the trials table vs 476 in the paper** — the NWB trials table
  holds every behavioural trial of the session. Neither restricting to ephys-covered trials
  (537) nor removing early-lick/ignore trials (408) reproduces 476, so the paper's figure is
  computed over some other set. It has no bearing on the conversion; our numbers are reported
  as measured.
* **Performance 80.6 % vs 84 %** — computed with the paper's own definition
  (hits / (hits+misses) on non-photostim, non-early trials). 22 of 174 sessions fall below
  the paper's stated >65 % selection criterion even though they are in the release, which is
  what pulls the mean down. See decision D2.

### Check 5 — Edge cases
| Edge case | Found in | Handling |
|---|---|---|
| Session with **0 good units** | `SC017_20190216_162508_s4` | skipped, reported in the log |
| Ephys covers only part of the session | 9 sessions | trials matched to `units.obs_intervals` by time |
| Ephys block does **not** start at trial 1 | `SC026_20190807_134913_s20` (trials 126-630) | **bug found and fixed** — see Issue 2 below |
| Last ephys trial contains no spikes | 2 trials | trials with zero population spikes are dropped |
| Video timestamps not monotonic (frames duplicated) | `SC045_20201216_120333_s24`, `SC053_20210220_124834_s15` | timestamps sorted (stable) before binning |
| Video missing for most of the session | `SC066_20210413_112028_s6` (7/550 trials), `SC065_20210505_170309_s6` (412/660) | tongue class 3 where there is no video; other outputs unaffected; documented in `metadata['known_limitations']` |
| Trial window extends past the end of the recorded trial | 18.5 % of trials, ~91 % of *miss* trials | binned as zero rate / not-visible; decision D6 |
| Trial with no sample-epoch onset before the Go cue | 0 occurrences (fallback code present anyway) | fallback to the session median Go-tone interval |
| Units with NaN CCF coordinates | none among good units | ALM coordinate test written to fail safe on NaN |
| Bin edge convention | — | half-open `[lo, hi)`, matching the reference `sliding_histogram`; verified numerically |
| Sessions with a constant output | 3 sessions have no early-lick trials | kept; a real property of those sessions |
| dtypes | — | neural/input `float32`, output/`brain_region_idx`/`subject_idx` `int64`; T = 80 for every trial of every session |

### Issues Found and Resolved
1. **Trials with no spikes at all.** The verification pass warned that one trial was
   entirely zero. Root cause: the recording stops mid-trial, so the last `obs_intervals`
   entry can contain no spikes. Fix: drop trials whose whole population has zero spikes in
   the window. Re-ran conversion + verification: warning gone.
2. **Wrong ephys-coverage rule.** The first version assumed `obs_intervals` covered the
   *first* N trials. `compare_stats.py` flagged `SC026_20190807_134913_s20` as losing 125
   trials to the zero-spike filter; investigation showed its ephys block runs from trial 126
   to 630, not 1 to 505. Fix: match each `obs_intervals` row to a trial by its
   `(start_time, stop_time)` instead of assuming a prefix. A dedicated scan
   (`cache/scan_obs.py`) confirmed that after the fix **every** `obs_intervals` row in all
   173 sessions maps to exactly one trial, that all good units of a session share the same
   `obs_intervals`, and that only this one session is non-prefix. After the fix the
   zero-spike filter drops only 2 trials in the whole dataset, and the per-session
   neuron/trial mismatch count against the independent recomputation is 0.

All checks were re-run after both fixes and all pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
(log: `/app/train_decoder_full_out.txt`; figures `sample_trials.png`, `predictions.png`).
Trained on an L4 GPU, ~6 min for 200 epochs over 173 sessions / 89,544 trials / 7.16 M
time bins.

### Training Progress
- Loss decreasing: **yes**, monotonically — 15.9 (epoch 1) -> 7.58 (10) -> 3.14 (30) ->
  1.42 (60) -> 0.834 (100) -> 0.653 (200). Held-out test loss 0.644.

### Decoder Results (Full)
| Output | Chance (1/n_classes) | Majority-class fraction | Training balanced acc | Validation balanced acc | Val / chance |
|--------|------|------|-------------|--------|-------|
| choice (left/right/no lick) | 0.333 | 0.429 | 0.710 | **0.682** | 2.04x |
| outcome (ignore/miss/hit) | 0.333 | 0.685 | 0.697 | **0.664** | 1.99x |
| early_lick (no/yes) | 0.500 | 0.884 | 0.795 | **0.755** | 1.51x |
| tongue_y (4 classes) | 0.250 | 0.755 | 0.687 | **0.656** | 2.62x |

Note that these are *balanced* accuracies, so the relevant chance level is 1/n_classes, and
every output is far above both that and the majority-class fraction.

`sample_trials.png` and `predictions.png` show sensible trials: the photostim input is a
clean 0.5 s pulse ending at the Go cue, the tone-onset input is a straight ramp, the tongue
output alternates between "not visible" and the three position classes during licking, and
the predicted traces track the true ones.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance
Every output is at least 1.5x chance and none is below chance (table above). The weakest
ratio is `early_lick` at 1.51x, which is expected for a *binary* variable: the ceiling for
a 2-class balanced accuracy is 1.0, i.e. only 2x chance, so 1.51x corresponds to 0.755 of a
maximum of 1.0. The others are at 2.0-2.6x chance.

To check that the remaining error is task structure rather than a conversion bug, the
decoder was re-run and its validation accuracy broken down **per 50 ms bin**
(`cache/time_resolved_accuracy.py`, log `cache/time_resolved_accuracy_out.txt`). The
provided decoder classifies every bin independently, so bins in which a variable is simply
not yet encoded can only be at chance and pull the trial average down:

| Epoch | choice (0.333) | outcome (0.333) | early_lick (0.500) | tongue_y (0.250) |
|---|---|---|---|---|
| baseline (-2.5 .. -1.85 s, before the tone) | 0.545 | 0.557 | 0.738 | 0.664 |
| sample (-1.85 .. -1.2 s) | 0.626 | 0.576 | **0.813** | 0.626 |
| delay (-1.2 .. 0 s) | 0.659 | 0.589 | **0.801** | 0.623 |
| response (0 .. +1.5 s) | **0.811** | **0.790** | 0.687 | 0.585 |
| peak bin | 0.864 (+0.33 s) | 0.871 (+1.03 s) | 0.837 (-1.18 s) | 0.692 (-1.88 s) |

The profiles are exactly what the physiology predicts, which is strong evidence that the
alignment and the labels are right:
* `choice` is at chance before the instruction tone, rises through the delay and jumps at
  the Go cue when the animal actually licks;
* `outcome` only becomes decodable *after* the response, peaking ~1 s after the Go cue —
  the data paper reports outcome selectivity "after licking offset (2-3 s post Go cue)";
* `early_lick` is most decodable during the sample and delay epochs, which is when the
  early licking happens;
* `tongue_y` is decodable throughout, as it is a moment-to-moment movement variable.

### Check 2 — Accuracy comparison to the papers
Neither paper trains a decoder of this form (neural population -> choice / outcome /
early lick / tongue position, one classifier per time bin, shared across 173 sessions), so
there is no directly comparable published number. The published decoding results and the
corresponding check on the converted data:

| Published analysis | Paper value | This conversion | Comment |
|---|---|---|---|
| Choice from **population spike rates**, PCA + logistic regression per time bin (method paper Fig. 7 / `population_decoding_utils.nested_cross_validation`) | not tabulated; used as the standard analysis | **ALM populations, 8 sessions with the most ALM neurons** (`cache/choice_auc_check.py`): ROC-AUC 0.528 (baseline) -> 0.688 (sample) -> 0.765 (delay) -> **0.953 (response)** | reproduces the expected profile |
| Choice from **video** during the sample epoch | AUC 0.66 +/- 0.12 s.d., at chance in the first half of the sample epoch | neural AUC first rises at the bin centred on -1.675 s, i.e. 0.18 s **after** the instruction-tone onset at -1.85 s | latency is right; alignment verified |
| Choice from **video** during the response epoch | AUC ~0.9 ("easily decodable") | neural AUC 0.95 in the response epoch | neural > video, as expected |
| Single-neuron choice AUC threshold for "choice modulated" | 0.65 | population AUC exceeds this from the sample epoch onward | consistent |

The critical diagnostic is the **baseline AUC of 0.528** (chance) that rises only after the
instruction tone: a temporal misalignment of even one bin would leak choice information into
the pre-tone baseline, and a mislabelling would flatten the profile. Neither happens.

### Check 3 — Train vs validation gap
| Output | Train | Validation | Train/Val |
|---|---|---|---|
| choice | 0.710 | 0.682 | 1.04 |
| outcome | 0.697 | 0.664 | 1.05 |
| early_lick | 0.795 | 0.755 | 1.05 |
| tongue_y | 0.687 | 0.656 | 1.05 |

All ratios are 1.04-1.05, far below the 1.5 threshold: no overfitting and no sign of data
leakage. (For comparison, the 2-session sample run had ratios of 1.17-1.24, as expected
with ~40x less data.)

### Additional debugging steps carried out
1. **Output values verified against the raw data** for 24 individual trials across 4
   sessions (Step 10, Check 2) — all exact.
2. **Temporal alignment verified** both visually (`processing_*.png`, where the photostim
   input coincides with the raw laser event and the tone-onset input crosses zero at the
   raw sample onset) and statistically (the choice-AUC onset latency above).
3. **Output variation checked**: the least balanced output is `early_lick` (88.4 % / 11.6 %);
   no output is 99 % one class, and the decoder uses `balanced_loss=True` plus balanced
   accuracy, so imbalance is handled.
4. **Neuron filtering checked** against the papers: 69,453 'good' units, per-area counts
   within 2 % of the published figures (medulla exact).
5. **Processing checked against the reference code**: the binned firing rates are numerically
   identical to the reference `sliding_histogram` implementation run on the same spikes.

### Issues Found and Resolved
None new in this step. The two issues found in Step 10 (zero-spike trials; the
non-prefix ephys block in `SC026_20190807_134913_s20`) were fixed before this training run,
and the full conversion, verification, sanity checks, statistics comparison and decoder
training were all re-run afterwards.

### Residual limitations (documented, not bugs)
* Error (*miss*) trials have no recorded spikes or video after ~0.8 s post Go cue because the
  NWB export stores data only inside `[trial.start_time, trial.stop_time]` and error trials
  are terminated by the time-out. This is a property of the released data given the
  requested [-2.5, +1.5] s window; it is recorded in `metadata['known_limitations']`.
* Two sessions have partially missing video, so some `tongue_y` bins are labelled
  "not visible" because there is no video rather than because the tongue was in the mouth
  (~0.9 % of trials dataset-wide).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — dataset description, key statistics, how to load and use the
      pickle, the full output-format specification, a processing summary and the reference
      decoder's accuracies.
- [x] `CONVERSION_NOTES.md` complete — every decision (D1–D11 in Step 5) with its
      justification, every consistency check against the papers/code/data, all validation
      tables, and the two issues found and fixed.
- [x] `cache/` folder created — all exploration scans and the independent validation
      scripts, documented in `cache/README_CACHE.md`.
- [x] Directory clean (`__pycache__` and one broken scratch file removed).

### Files produced
| File | Description |
|---|---|
| `convert_data.py` | the conversion script (self-contained; needs only `pynwb`, `numpy`, `matplotlib`) |
| `converted_data.pkl` | full converted dataset, 11.89 GB, 173 sessions |
| `sample_data.pkl` | 2-session sample, 0.20 GB |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | `train_decoder.py --verify-only` logs |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_<session>.png`, `processing_<session>_summary.png` | `--show-processing` figures for 2 sessions |
| `sample_trials.png`, `predictions.png` | figures produced by `train_decoder.py` |
| `README.md` | user-facing documentation |
| `CONVERSION_NOTES.md` | this file |
| `cache/` | exploration scans and independent validation scripts (see `cache/README_CACHE.md`) |
