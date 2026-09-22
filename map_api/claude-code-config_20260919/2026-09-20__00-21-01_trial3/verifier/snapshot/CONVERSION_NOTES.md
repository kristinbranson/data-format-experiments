# Dataset Conversion Notes

## Overview
- **Dataset**: DANDI:000363 "Mesoscale Activity Map Dataset" (Chen, Nguyen, Li, Svoboda 2023),
  version 0.230822.0128. Neuropixels recordings + DeepLabCut video tracking while mice perform
  an auditory delayed-response (memory-guided directional licking) task.
  - Data paper: Chen et al., *Brain-wide neural activity underlying memory-guided movement* (`/app/datapaper.pdf`)
  - Method paper: Wang, Kurgyis et al., *Brain-wide analysis reveals movement encoding structured
    across and within brain areas*, Nat Neurosci 2025 (`/app/methodpaper.pdf`)
  - Spike-sorting/QC white paper: `/app/ChenLiuEtAl2023_SpikeSortingQC.pdf`
  - Reference analysis code: `/app/code` (github.com/druckmann-lab/MapVideoAnalysis)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `convert_data.py`, `ccf_regions.py` (written by me)
- `datapaper.pdf`, `methodpaper.pdf`, `ChenLiuEtAl2023_SpikeSortingQC.pdf`, `methods.txt`
- `train_decoder.py`, `decoder.py` (provided decoder / validation code)
- `code/` — reference analysis code (Archive/, Notebooks/, Sherlock/, VideoAnalysisUtils/)
- `data/` — 174 `.nwb` files in 28 `sub-*` directories (50 GB), plus `dandiset.yaml`
- `pynwb_docs/`, `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `cache/` — my exploration scripts and extracted PDF text

Environment verified: Python 3.13.15, numpy 2.4.4, torch 2.6.0+cu124 (CUDA available, NVIDIA L4
23 GB), pynwb 4.1.0. Host: 128 CPUs, 1006 GB RAM, 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The reference repository (`/app/code`) analyses the *DataJoint `.mat` export* of this dataset,
not the NWB release, so the loading layer differs; everything downstream (curation, alignment,
binning, trial variables) is directly transferable.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel` / `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Top-level per-session ephys preprocessing: concatenates probes, reads behaviour/task variables, applies QC unit lists, bins spikes. |
| `preprocessing_utils.loadmat` | `VideoAnalysisUtils/preprocessing_utils.py` | LOADING | MATLAB struct loader (replaced here by `pynwb`). |
| `helper_get_neuron_id_area` | `preprocessing_DJ_2022Aug.py` | CURATION | Selects units by (a) membership in the classifier "good units" QC list for a region and (b) hemisphere, using CCF ML midline = **5700 µm**; also drops units with empty CCF label. |
| `process_one_area` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates per-trial spike times to `[begin_time, end_time]` **already relative to the go cue** and bins them. |
| `sliding_histogram` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Spike binning: sliding bins of width `bw`, stride `stride`, centres from `begin_time` to `end_time`; returns **firing rate = count / bin_width** when `rate=True`. |
| `preprocess_all_ephys.py` | `Sherlock/` | PROCESSING | Production settings: `bw=0.04`, `stride=0.0034`, `begin_time=-3.0`, `end_time=3.0`, `qc_mode='classifier'`. |
| `align_markers_between_lims` | `Sherlock/align_markers.py` | PROCESSING | Aligns 300 Hz DeepLabCut markers to the go cue: frame *i* of a trial is at `i*dt - go_time` with `dt = 0.0034 s`, sampled on `np.arange(t_min, t_max, dt)` with `t_min=-3`, `t_max=1.5`. |
| `get_bad_trial_inds` | `Sherlock/align_markers.py` | CURATION | Flags trials whose video frame count disagrees with the trial duration. |
| `get_regular_trial_mask` | `VideoAnalysisUtils/functions_for_r2.py` | CURATION | "Regular" trials: `early_lick == 0` AND `auto_water == 0` AND `free_water == 0` AND `outcome != no-response` AND `photostim power == 0`. |
| `create_4fold_trial_type_mask` | `functions_for_r2.py` | CURATION | Stratification groups from `trial_type` (instruction) × `correctness`. |
| `temporal_alignment_embed_and_ephys` | `functions_for_r2.py` | ALIGNMENT | Trims video and ephys time axes to a common window at a shared `dt`. |

### Notes
- **Electrophysiology, not imaging**: no dF/F. Units *must* be quality-filtered. The reference uses
  `qc_mode='classifier'`, i.e. the per-region logistic-regression classifier described in
  `methods.txt` and the QC white paper; its output is the list of **"good" units**. In the NWB
  release this is materialised as the `units['classification']` column (`good` / `unlabelled`),
  so `classification == 'good'` is the exact NWB equivalent of the reference's `goodunits/*.mat`
  lists. (Confirmed: every `good` unit has a non-empty CCF annotation `anno_name`, and every
  `unlabelled` unit has an empty one — matching the reference's additional requirement that a unit
  have both ephys and histology.)
- Spike times in the reference `.mat` export are **already aligned to the go cue of their trial**;
  the NWB stores absolute session times plus a go-cue event per trial, so I do the subtraction.
- Firing rates are counts divided by the bin width (Hz), not raw counts.
- Marker (video) sampling interval is `dt = 0.0034 s` (≈294 Hz; the paper calls the cameras 300 Hz).
- The reference's photostim variable is `[laser_power, stim_type(1=left/2=right/6=bilateral ALM),
  laser_on_time, laser_off_time]`, with on/off times re-referenced to the go cue. The NWB equivalent
  is `BehavioralEvents/photostim_start_times`/`photostim_stop_times` (absolute) plus the trials-table
  columns `photostim_onset`/`photostim_power`/`photostim_duration`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/sub-<subject_id>/sub-<subject_id>_ses-<YYYYMMDDThhmmss>_behavior+ecephys+ogen.nwb`
— one NWB file per recording session, 174 files, 28 subjects, 50 GB. All files read with
`pynwb.NWBHDF5IO` (lazy/HDF5-backed; no `h5py` used directly).

Per file:
- `nwb.identifier` — e.g. `SC015_20190207_120657_s1` (mouse, date, time, session number)
- `nwb.subject.subject_id` (numeric, e.g. `440956`) and `nwb.subject.description` (mouse name, `SC015`)
- `nwb.intervals['trials']` (= `nwb.trials`) with columns
  `start_time, stop_time, trial, photostim_onset, photostim_power, photostim_duration, trial_uid,
   task, task_protocol, trial_instruction, early_lick, outcome, auto_water, free_water`
  - `task` is always `audio delay`, `task_protocol` always `1`
  - `trial_instruction` ∈ {`left`, `right`}; `outcome` ∈ {`hit`, `miss`, `ignore`};
    `early_lick` ∈ {`early`, `no early`}; `auto_water`, `free_water` ∈ {0,1}
  - `photostim_*` are strings, `'N/A'` when there was no stimulation
- `nwb.acquisition['BehavioralEvents']` — 14 `TimeSeries` whose **timestamps** carry the events:
  `presample/sample/delay/go/trialend` `_start_times`/`_stop_times`, `left_lick_times`,
  `right_lick_times`, `photostim_start_times`, `photostim_stop_times`
- `nwb.acquisition['BehavioralTimeSeries']` — DeepLabCut tracking, `(n_frames, 3)` =
  `(x, y, likelihood)` with per-frame `timestamps`:
  `Camera0_side_JawTracking`, `Camera0_side_NoseTracking`, `Camera0_side_TongueTracking`
  (present in **all 174** sessions); `..._WhiskerTracking_whisker` in 20, `LickPortTracking` in 4,
  `Camera3_side_*` in 3.
- `nwb.units` — columns include `spike_times` (absolute session seconds), `obs_intervals`,
  `is_good_trials` (per-trial bool), 15 Kilosort quality metrics (`unit_amp`, `snr`,
  `isi_violation`, `presence_ratio`, `amplitude_cutoff`, `drift_metric`, `isolation_distance`,
  `l_ratio`, `d_prime`, `nn_hit_rate`, `nn_miss_rate`, `silhouette_score`, `max_drift`,
  `cumulative_drift`, waveform features), `unit_quality` (`good`/`multi` — Kilosort label),
  **`classification`** (`good`/`unlabelled` — the QC-classifier label) and **`anno_name`**
  (CCF v3 annotation, non-empty only for `classification == 'good'`).
- `nwb.electrodes` — per-electrode CCF coordinates `x` (ML), `y` (DV), `z` (AP) in µm.
  `units['electrodes']` links each unit to exactly one (peak) electrode → per-unit CCF coordinates.
- `nwb.electrode_groups[*].location` — JSON with the targeted `brain_regions` (e.g. `"left ALM"`),
  stereotaxic coordinates and insertion angles.
- `nwb.ogen_sites` — `OBIS470_4` (unilateral ALM) and `OBIS470_6` (bilateral ALM), 473 nm.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Files / sessions | 174 (173 with ≥1 `good` unit) |
| Subjects (mice) | 28 |
| Sessions / subject | 3–10 (median 6) |
| Units (all Kilosort clusters) | 272,227 |
| Units with `classification == 'good'` | **69,453** (25.5 % of all clusters) |
| Good units / session | mean 399, range 0–923 |
| Trials (total) | 94,990 |
| Trials / session | mean 546, range 264–800 |
| Outcome | hit 65,254 (68.7 %), miss 15,641 (16.5 %), ignore 14,095 (14.8 %) |
| Early lick | 10,805 (11.4 %) |
| Instruction | right 48,913, left 46,077 |
| Photostim trials | 18,588 (19.6 %), in 168/174 sessions |
| auto_water / free_water trials | 1,339 / 2,450 |
| Video frame interval | 0.0034 s |
| Tongue `likelihood` | strongly bimodal: 87 % < 1e-4, 11 % > 0.999 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (`methods.txt`) |
| Sessions | 173 | same |
| Probe insertions | 655 | same |
| Fraction of Kilosort clusters kept | 25.9 % | "This corresponds to 25.9 % of clusters reported by Kilosort2." |
| Subjects | 28 mice | "This study is based on data from 28 mice, including 25 VGAT-ChR2-EYFP…" (method paper Methods) |
| Trials / session | mean 476, range 130–785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Behavioural performance | 84 %, range 65–99 % | "with 84% correct rate (range, 65-99%)" |
| Session selection | perf > 65 % and ≥50 correct lick-left **and** ≥50 correct lick-right | "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." |
| Performance definition | fraction correct of **control** (no photostim) trials, **excluding early-lick trials** | "Overall performance was computed as the fraction of correct control trials (i.e. no photostimulation), excluding any early lick trials." |
| Good units by region | ALM 8717, orbital 10223, striatum 7664, pallidum 1092, hippocampus 1944, thalamus 12808, hypothalamus 815, midbrain 7495, pons 347, medulla 2928, cerebellum 1820, other 14090 | data paper Fig. 1E |
| Photostim | ~25 % of randomly interleaved trials, 17 VGAT-ChR2 mice, 93 sessions; 40 Hz sinusoid, 5 mW/hemisphere, last 0.5 s of delay incl. 100 ms ramp-down, **always ends before the go cue** | `methods.txt` "Photoinhibition" |
| Photostim effect on behaviour | 83.2 % → 71.7 % | same |
| Task epochs | sample (3 × 150 ms tones, 100 ms gaps), delay 1.2 s, go cue 0.1 s, answer 1.5 s, consumption 1.5 s | `methods.txt` "Behavior and video tracking" |
| Video | 2 CMOS cameras, 300 Hz, side + bottom; DeepLabCut tongue/jaw/nose | same |
| Neural bin (method paper) | 40 ms width, 3.4 ms stride | "we then binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Behaviour bin | 3.4 ms (300 Hz video) | same |

### Processing Details
- **Temporal alignment**: everything is aligned to the **go cue**. The reference `.mat` export stores
  spike times already relative to the go cue; markers are aligned with
  `times_for_frames = arange(n_frames)*0.0034 − go_time` (go-cue-relative), over `[-3, 1.5)` s.
- **Temporal binning**: reference uses 40 ms width / 3.4 ms stride sliding bins → firing **rate**
  (spikes / bin_width). The present task instead prescribes **50 ms bins**, so I use
  non-overlapping 50 ms bins over `[-2.5, +1.5)` s (80 bins), rate = count / 0.05 s.
- **Marker handling** (method paper Methods): outliers found with a 5-sigma velocity threshold and
  imputed from neighbouring frames; "when the tongue was occluded while it was in the mouth … we set
  the tongue position to its mean value". Here occlusion is an explicit output category instead.

### Curation Steps

**Neuron curation rules**:
1. Keep only units labelled `good` by the region-specific QC classifier
   (`units['classification'] == 'good'`); this is the reference's `qc_mode='classifier'` /
   `goodunits/*.mat` selection and the paper's 69,943-unit set.
2. Require a CCF annotation (reference drops units with empty `ccf_label`; in the NWB this is
   automatic — `anno_name` is non-empty for exactly the `good` units).
3. `is_good_trials` is True for 99.98 % of unit×trial pairs (only 0.1 % of units have any False
   entry), so no additional per-trial unit masking is applied — same as the reference.

**Trial curation rules** (reference `get_regular_trial_mask`): no early lick, no auto-water,
no free-water, no no-response, no photostimulation.
→ *Only the auto-water and free-water exclusions are adopted here*; see Step 5 "Key Decisions"
for why early-lick / no-response / photostim trials must be **kept** for this decoding task.

**Session curation**: performance > 65 % and ≥ 50 correct lick-left and ≥ 50 correct lick-right
control trials (data paper), plus ≥ 1 good unit.

### Decoders Trained (accuracies reported in the papers)
| Decoded variable | Reported accuracy |
|---|---|
| Choice, from **video** (embedding), before sample epoch | AUC 0.51 ± 0.06 (n = 106 sessions) |
| Choice, from **video**, sample+delay epoch | AUC 0.66 ± 0.12 |
| Choice, from **video**, 2nd half of response epoch | AUC 0.99 ± 0.01 |
| Choice, from video markers, response epoch | AUC 0.88 ± 0.01 |
| Choice, from single-neuron delay firing rate | threshold AUC > 0.65 used to call a neuron "choice-modulated" |
Neither paper reports a multi-class accuracy for choice/outcome/early-lick/tongue decoding from
population neural activity, so there is no direct numerical target for this task's decoder; the
AUC ≈ 0.99 for post-go-cue choice from video is the closest reference point and implies that
lick direction is near-perfectly recoverable after the go cue.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Unit QC | `goodunits/*.mat` lists from region classifiers | `units['classification'] ∈ {good, unlabelled}`; 69,453 good = 25.5 % of 272,227 clusters | 69,943 good units, 25.9 % of clusters, 173 sessions | Same selection. Residual 490-unit (0.7 %) difference is the dandiset version (papers cite 0.231012.2129, `/app/data` is 0.230822.0128). Use `classification == 'good'`. |
| Region labels | 14 coarse groups (`ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum`) from external QC files | per-unit CCF annotation string + CCF coordinates | Fig. 1E per-region counts | Rebuilt the grouping from the CCF annotation (`/app/ccf_regions.py`). **9 of 11 named groups reproduce the Fig. 1E counts exactly** (orbital 10223, pallidum 1092, hippocampus 1944, thalamus 12808, hypothalamus 815, midbrain 7495, pons 347, medulla 2928, cerebellum 1820), striatum 7664 exactly, ALM 8445 vs 8717 (−3.1 %; ALM is not an Allen structure, see below). |
| Sessions in release | — | 174 files, 173 with good units; 16 sessions below the 65 % performance criterion, 7 with <50 correct trials of one direction | "173 behavioral sessions", but selection criteria perf > 65 % & ≥50/direction | The release contains all recorded sessions; the paper's *unit* counts are pre-selection. Applying the stated selection gives **151 sessions, performance mean 83.3 %, range 65.1–97.0 %** vs the paper's "84 %, range 65–99 %" — an excellent match, confirming the criteria. |
| Trials / session | — | mean 546 (all), mean 484 excluding early-lick | mean 476 | The paper's number matches trials **excluding early-lick** trials (484 vs 476) for the selected sessions. |
| Trial window coverage | reference bins `[-3, +3]` s around the go cue | NWB spikes exist **only inside `[trial start_time, trial stop_time]`** (0 spikes outside `obs_intervals`, verified in all 174 sessions; trials cover only 66 % of session wall-clock) | — | Inherent to the released dataset (the DataJoint export the reference used is segmented the same way). 3.1 % of trials start less than 2.5 s before the go cue and 15.6 % end less than 1.5 s after it (miss/error trials are truncated a few hundred ms after the go cue by the error time-out). Bins outside the trial get a firing rate of 0. Documented, not "fixed". |
| Photostim timing | `[laser_on_time, laser_off_time]` relative to go cue | onsets at −1.2 s or −0.5 s, offsets at −0.7 s or −0.000/+0.001 s relative to go | "always ended before the Go cue" | Consistent. Use the actual per-trial event times; a 1 ms overshoot at the go cue is within rounding. |
| Photostim event ↔ trial | — | `len(photostim_start_times)` equals the number of trials with `photostim_onset != 'N/A'` and they map 1:1 to those trials **in all 174 sessions** | ~25 % of trials | Consistent (19.6 % here, over all sessions including the 6 without any photostim). |
| Go cue | one per trial | exactly one `go_start_times` per trial in **all 174 sessions** | — | Consistent; used as the alignment event. |
| Sample / delay events | one per trial | 405 sample and 395 delay starts for 368 trials in the example session | "Licking early during the sample/delay epoch triggered a replay of the epoch" | Replays produce extra sample/delay onsets. "Tone onset" is therefore defined as the **last** `sample_start_times` at or before the trial's go cue (always inside the trial: 0 violations dataset-wide). |
| Video | markers from `tracking.camera_0_side` | `BehavioralTimeSeries/Camera0_side_*Tracking`, present in all 174 sessions | 300 Hz side view | Consistent; `dt` = 0.0034 s exactly. 3 sessions have video that stops early (795 trials, 0.8 %, have no frames in the analysis window). |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (NWB) | Target field | Transform | Reference code analogue |
|---|---|---|---|
| `units['spike_times']` (absolute s) for units with `classification == 'good'` | `neural[session][trial]`, shape `(n_neurons, 80)`, float32 | subtract the trial's go-cue time; histogram into 80 non-overlapping 50 ms bins spanning `[-2.5, +1.5)` s; divide by 0.05 s → firing rate in Hz | `process_one_area` + `sliding_histogram(..., rate=True)` |
| `BehavioralEvents/go_start_times.timestamps` | alignment event (t = 0) | one per trial | `task_cue_time[0,:]` |
| `BehavioralEvents/sample_start_times.timestamps` | `input[0]` = `time_from_tone_onset_s` | last sample onset ≤ go cue; input value at bin *b* = `bin_centre_b − (tone_onset − go)` (seconds since the tone) | `task_sample_time` |
| `BehavioralEvents/photostim_start_times` / `photostim_stop_times` | `input[1]` = `photostim_on` | 1 if the bin `[t0,t1)` intersects `[stim_start, stim_stop)`, else 0 | `task_stimulation[:,2:]` |
| `trials['trial_instruction']` + `trials['outcome']` | `output[0]` = `lick_direction_choice` | `ignore` → `no lick` (2); `hit` → instruction; `miss` → opposite of instruction | `trial_type` × `correctness` (`create_4fold_trial_type_mask`) |
| `trials['outcome']` | `output[1]` = `outcome` | `ignore`→0, `miss`→1, `hit`→2 | `behavior_report` (`correctness`: −1/0/1) |
| `trials['early_lick']` | `output[2]` = `early_lick` | `no early`→0, `early`→1 | `behavior_early_report` |
| `BehavioralTimeSeries/Camera0_side_TongueTracking` (`y`, `likelihood`) | `output[3]` = `tongue_y_position` | per bin: frames with `likelihood > 0.9` are "visible"; bin value = mean `y` of visible frames; bins with no visible frame → 3; otherwise 0/1/2 by the session's 40th/60th percentile of the visible bin values | `align_markers_between_lims` (`tongue_y`) |
| `units['anno_name']` + `electrodes[x,y,z]` | `brain_region_idx[session]` | `ccf_regions.annotation_to_region` → one of 14 coarse groups | `helper_get_neuron_id_area` |
| `nwb.subject.description` (e.g. `SC015`) | `subjects` / `subject_idx` | unique mouse names | mouse name in the `.mat` filename |
| `nwb.identifier` | `metadata['session_info']` | e.g. `SC015_20190207_120657_s1` | `sess_name` |

### Key Decisions
1. **Loading**: `pynwb.NWBHDF5IO` only. Trials, events, units and tracking are read through the
   `pynwb`/`hdmf` API (`nwb.intervals['trials']`, `nwb.acquisition[...]`, `nwb.units`,
   `nwb.electrodes`). No direct `h5py` access.
2. **Unit curation** = `classification == 'good'` (the QC-classifier "good units" of the reference
   pipeline, validated against the paper's 69,943 / 25.9 % figures). No further metric thresholds:
   the 15 quality metrics were already consumed by that classifier, so re-thresholding them would
   double-filter.
3. **Session curation**: ≥1 good unit, **and** the data paper's behavioural criteria (control,
   non-early-lick performance > 65 %; ≥50 correct lick-left and ≥50 correct lick-right control
   trials). Validated: the 151 surviving sessions have mean performance 83.3 % (range 65.1–97.0 %)
   vs the paper's 84 % (65–99 %).
4. **Trial curation**: drop `auto_water` or `free_water` trials (reference `get_regular_trial_mask`;
   their `outcome` is not a genuine behavioural report — the DataJoint comment for the equivalent
   field reads "1: correct **or free water**"). Drop trials with no video frame in the analysis
   window (0.8 %), because every bin would otherwise be labelled "tongue not visible" when the
   truth is "not measured".
   **Deliberate deviation from `get_regular_trial_mask`**: early-lick, no-response (`ignore`) and
   photostimulation trials are **kept**, because the decoder specification requires `early_lick`
   (no/yes) and `outcome` (ignore/miss/hit) as outputs and photostimulation as an input — removing
   those trials would leave those variables constant and the task undefined.
5. **Alignment**: go-cue onset (`go_start_times`), one per trial, verified in all 174 sessions.
   Window −2.5 s → +1.5 s, 80 bins of 50 ms. Bin *b* spans `[-2.5 + 0.05b, -2.5 + 0.05(b+1))`;
   the bin centre `-2.475 + 0.05b` is used for time-valued inputs.
6. **Firing rate**, not spike count (reference `sliding_histogram(rate=True)`): count / 0.05 s.
   Bins lying outside the trial's `[start_time, stop_time]` interval necessarily contain no spikes
   and get rate 0; this is a property of the release (see Step 4), and `metadata` records it.
7. **Tongue visibility threshold** `likelihood > 0.9`. The DeepLabCut likelihood is extremely
   bimodal (87 % of frames < 1e-4, 11 % > 0.999); the visible fraction changes by < 0.5 % of frames
   between thresholds of 0.01 and 0.999, so the choice is immaterial (verified in Step 10).
   A bin counts as visible if **any** frame inside it is visible, because a tongue protrusion lasts
   only ~50–100 ms and a majority rule would drop genuine short protrusions straddling a bin edge.
8. **Percentiles for the tongue discretisation** are computed **per session** over the visible
   per-bin y-values of all retained trials of that session — the population that is actually being
   labelled — so classes 0/1/2 hold 40 %/20 %/40 % of the visible bins by construction.
9. **Choice on error trials** is the *opposite* of the instruction (the mouse licked the wrong
   port), and `no lick` on `ignore` trials. This is the standard reading of `outcome` in this task
   and matches the reference's `correctness` semantics (1 correct / 0 error / −1 no response).
10. **Inputs are exactly the two the task specifies**; the go cue is not added as an input because
    it is at t = 0 in every trial by construction.
11. **Brain regions**: 14 coarse groups derived from the per-unit CCF annotation (Step 4). Region
    names carry no hemisphere; the hemisphere of every neuron (from CCF ML vs the reference's
    5700 µm midline) is stored separately in `metadata['neuron_hemisphere']`.

### Planned Sanity Checks
- [x] Good-unit count and fraction-of-clusters vs `methods.txt` (69,943 / 25.9 %)
- [x] Per-region good-unit counts vs data paper Fig. 1E
- [x] Session-selection outcome (count, mean and range of performance) vs data paper
- [x] Trials/session vs the paper's 476
- [x] Exactly one go cue per trial; tone onset always inside its trial
- [x] Photostim events map 1:1 to `photostim_onset != 'N/A'` trials; stim ends at or before the go cue
- [ ] Spike counts re-derived from raw NWB for specific (session, trial, neuron, bin) cells
- [ ] Tongue category re-derived from raw NWB frames for specific (session, trial, bin) cells
- [ ] Input values re-derived from raw NWB event times for specific (session, trial, bin) cells
- [ ] Output class distributions vs the raw trials tables
- [ ] Number of sessions/trials/neurons in the pickle vs the counts computed directly from NWB

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (+ `/app/ccf_regions.py` for the CCF→region grouping).
Run as `python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
[--workers N]`.

Structure:
| Function | Purpose |
|---|---|
| `read_session(nwb)` | pulls trials table, behavioural events, tongue tracking, good units, CCF coordinates out of an open `NWBFile` (pynwb only) |
| `recorded_trials(...)` | trials covered by the units' `obs_intervals` (the ephys recording) |
| `session_performance(raw, keep)` | data-paper performance / correct-trial counts, on the curated trials |
| `window_edges(go)` | absolute times of the 81 bin edges of each trial |
| `bin_spike_times(st, edges)` | one `np.searchsorted` per unit over all trials' edges → (n_trials, 80) counts |
| `segment_sums(...)` | 300 Hz tracking → per-(trial, bin) visible-frame count and y sum, via cumulative sums |
| `convert_session(path)` | full per-session conversion, returns arrays + metadata or a rejection reason |
| `plot_processing(result, out)` | 8-panel diagnostic figure (`--show-processing`) |
| `assemble(sessions)` | builds the final dictionary |
| `summarise(data, sessions)` | prints headline statistics |

Efficiency notes:
- **Inefficiency identified**: the obvious implementation loops over (unit × trial × bin) in
  Python. **Speed-up**: all trials' bin edges are stacked into one `(n_trials, 81)` array and
  a single `np.searchsorted(spike_times, edges.ravel())` per unit yields every bin count by
  differencing — `np.searchsorted` requires only the *haystack* to be sorted, so the trial
  windows need not be globally monotone.
- **Inefficiency identified**: reducing 300 Hz video (≈1,180 frames per trial window) bin by
  bin. **Speed-up**: prefix sums of the visible-frame indicator and of the masked y values,
  indexed by the same edge positions → whole session in three vectorised operations.
- **Speed-up**: sessions are converted in 24 worker processes (`ProcessPoolExecutor`).
- **Memory**: each session is returned as `(n_trials, ...)` blocks and split into the per-trial
  list only in `assemble`, where the block is dropped immediately afterwards; otherwise the
  ~10 GB of neural data would be held twice.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl` (0.07 GB),
`processing_SC015_20190209_150135_s3.png`, `processing_SC015_20190210_155629_s4.png`.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (SC015_20190209_150135_s3, SC015_20190210_155629_s4) |
| Neurons (total) | 757 |
| Neurons / session | 205, 552 |
| Subjects | 1 (SC015) |
| Trials (total) | 723 |
| Trials / session | 517, 206 |
| Timepoints / trial | 80 |
| `time_from_tone_onset_s` range | [-0.625, 4.467] s |
| `photostim_on` range | [0, 1] |
| `lick_direction_choice` | left 0.498, right 0.479, no lick 0.024 |
| `outcome` | ignore 0.024, miss 0.062, hit 0.914 |
| `early_lick` | no 0.992, yes 0.008 |
| `tongue_y_position` | 0.099 / 0.050 / 0.099 / 0.752 |

### Processing Plots Review
The 8-panel figures show, per session: (1) the raw spike raster of a trial with the go cue,
tone onset, photostim window and the trial's own start/stop marked; (2) the same trial after
50 ms binning — the two agree row by row; (3) the population PSTH split by choice, which is
flat through the delay and jumps sharply **exactly at t = 0** and then separates by lick
direction, confirming go-cue alignment; (4) the two decoder inputs, with the ramp crossing
zero exactly at the marked true tone onset and the photostim square matching the shaded window
in panel 1; (5) the raw 300 Hz tongue trace, the per-bin mean of the *visible* frames, the
session's 40th/60th percentile lines and the resulting class trace — occluded frames sit at a
constant "tongue in the mouth" y and are correctly assigned class 3; (6) the session histogram
of visible per-bin y with the percentile cuts; (7) the tongue class for every trial sorted by
choice, showing class 3 almost everywhere before the go cue and licking structure after it;
(8) the per-trial outputs and photostim across the session. No anomalies.

**Anomaly found and fixed during this step**: the first version produced trials in which
*every* neuron was silent for the whole 4 s window. Tracing one such trial back to the NWB
showed that the units' `obs_intervals` covered only the first 160 of the session's 480 trials —
the ephys recording stopped while the behaviour continued. Fixed by adding `recorded_trials()`
(see Step 10, issue 1).

### Run Time Estimates
| Speed-up implemented | Effect |
|---|---|
| vectorised `searchsorted` spike binning | ~0.5 s per session for 400 units × 500 trials (a Python triple loop would be ~10^7 iterations) |
| prefix-sum video reduction | video reduction is <0.1 s per session |
| 24 worker processes | 174 files in 26 s wall-clock vs ~2.5 min serial |

| Step | Time / session | Estimated total |
|---|---|---|
| read + convert one session | 0.5–1.3 s (24 in parallel) | ~15 s for 174 files |
| assemble + pickle 9.7 GB | — | ~11 s |
| **total** | | **26 s** (measured) |

Well under the 15-minute budget, so no further optimisation was needed.

### Format verification (sample)
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only`
→ `/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**
Input ranges and output distributions are as tabulated above and match the raw trials tables.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None

### Training
Loss fell monotonically from 10.0 (epoch 6) to 0.475 (epoch 200); test loss 0.497.

### Decoder Results (Sample, 2 sessions)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|-------------------------|
| lick_direction_choice | 0.333 | 0.790 | 0.661 |
| outcome | 0.333 | 0.782 | 0.558 |
| early_lick | 0.500 | 0.976 | 0.962 |
| tongue_y_position | 0.250 | 0.642 | 0.591 |

Every output is above chance. The two sample sessions contain only 17 no-response and 6
early-lick trials between them, so the `outcome` and `early_lick` numbers are noisy; they are
much better determined on the full dataset (Step 11).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full` → `/app/conversion_full_out.txt`
`python -u /app/train_decoder.py /app/converted_data.pkl --verify-only` → `/app/verification_full_out.txt`

### Output Files
- `converted_data.pkl`: 9.66 GB, written in 26 s total
- `verification_full_out.txt`: created — **"Data format is valid, no errors or warnings."**

### Where the trials went
| | trials |
|---|---|
| in the 174 NWB files | 94,990 |
| in the 142 curated sessions, before trial curation | 77,593 |
| − outside the ephys recording (`obs_intervals`) | −736 |
| − auto-water or free-water | −2,773 |
| − no video frames in the analysis window | −239 |
| **kept** | **73,845** |
| (trials in the 32 rejected sessions) | (17,397) |

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (all 174 NWB) | Converted Data | Match? |
|-----------|------------------|----------------|------------------------------|----------------|--------|
| Subjects | 28 | — | 28 | 28 | ✔ |
| Sessions with good units | 173 | — | 173 | 142 after behavioural selection | ✔ (selection is explicit in the paper) |
| Good units, whole release | 69,943 (v0.231012) | `classification=='good'` | 69,453 (v0.230822) | 57,023 in the 142 kept sessions | ✔ (0.7 % release difference) |
| Good units as % of clusters | 25.9 % | — | 25.5 % | — | ✔ |
| Neurons / session | — | — | 399 | 402 (range 90–923) | ✔ |
| Trials / session | 476 (130–785) | — | 546 (264–800) all, 484 excl. early lick | 520 all; **460 excl. early lick** (203–687) | ✔ approx. |
| Behavioural performance | 84 % (65–99 %) | — | 81 % over all sessions | **83.8 % (65.8–98.9 %)** | ✔ |
| ALM good units | 8,717 | 14 coarse groups | 8,445 (whole release) | 6,723 | ✔ (−3.1 % on the release) |
| Orbital / pallidum / hippocampus / thalamus / hypothalamus / midbrain / pons / medulla / cerebellum / striatum | 10223 / 1092 / 1944 / 12808 / 815 / 7495 / 347 / 2928 / 1820 / 7664 | — | **identical** (10223 / 1092 / 1944 / 12808 / 815 / 7495 / 347 / 2928 / 1820 / 7664) | 8706 / 1014 / 1633 / 11029 / 749 / 6121 / 339 / 2591 / 1483 / 6076 | ✔ exact on the release |
| Photostim trials | ~25 %, ALM, ends before the go cue | `task_stimulation` | 19.6 % of all trials; onsets −1.2/−0.5 s, offsets −0.7/0.0 s rel. go | `photostim_on` ∈ {0,1}, on only at ≤ 0 s | ✔ |
| Outcome distribution | — | — | hit 0.687 / miss 0.165 / ignore 0.148 | hit 0.737 / miss 0.153 / ignore 0.109 | ✔ (better sessions kept) |
| Early-lick fraction | — | — | 0.114 | 0.115 | ✔ |
| `time_from_tone_onset_s` | tone ≈ 1.85 s before the go cue | `task_sample_time` | go − tone: median 1.85 s, min 0.95, max 10.42 | [−1.525, 11.894] s | ✔ |
| `tongue_y_position` | tongue visible only around licking | — | likelihood > 0.9 in 11.7 % of all frames | 25.6 % of bins visible (window is lick-rich); classes 0.104/0.052/0.104/0.740 | ✔ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output-log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There are no errors and no warnings to address. `/app/conversion_full_out.txt` contains no
warnings either; the only non-trivial lines are the 32 explicit session rejections.

### Check 2 — independent sanity checks (`/app/sanity_checks.py` → `/app/sanity_checks_out.txt`)
The script re-derives values **straight from the NWB files with `pynwb`**, with code written
independently of `convert_data.py` (it imports nothing from it), and compares with
`np.allclose` / `np.array_equal`. Run over 6 randomly chosen sessions (seed 0) plus global
checks. **All checks pass.**

Global (all 142 sessions, all 73,845 trials):
| Check | Result |
|---|---|
| every session has ≥ 2 trials | PASS |
| neural / input / output trial counts agree | PASS |
| all trials have 80 timepoints; neuron count constant per session | PASS |
| `brain_region_idx` length == neuron count | PASS |
| no NaN/Inf in neural | PASS |
| firing rates are non-negative multiples of 20 Hz (= count / 0.05 s) | PASS |
| all output values within their `output_values` range | PASS |
| dtypes float32 / float32 / integer | PASS |
| tongue classes 0/1/2 split the visible bins 0.400 / 0.200 / 0.400 | PASS |
| choice "no lick" ⇔ outcome "ignore" | PASS |

Per session (spot checks and full re-derivations):
| Check | Method | Result |
|---|---|---|
| **neural**, 25 random (trial, neuron, bin) cells | `sum((spikes >= go+edge_b) & (spikes < go+edge_{b+1})) / 0.05` from `units['spike_times']` | PASS, `np.allclose` |
| **neural**, one whole random trial | `np.histogram` of every good unit's spike times on that trial's 81 edges (e.g. 414 neurons × 80 bins, 20,853 spikes) | PASS, `np.allclose` |
| **neural**, neuron identity | stored `neuron_index_in_file` == `np.flatnonzero(classification=='good')` | PASS |
| **input 0**, 25 random (trial, bin) cells | `(go + bin_centre) − max(sample_start_times ≤ go)` | PASS, `np.allclose(atol=1e-4)` |
| **input 1**, 100 random (trial, bin) cells | overlap of the bin with any `[photostim_start, photostim_stop)` | PASS, `np.allclose` |
| **input 1**, per trial | trials with any photostim == `trials['photostim_onset'] != 'N/A'` | PASS |
| **output 0/1/2**, every trial | re-derived from `trial_instruction`, `outcome`, `early_lick` | PASS |
| **output 0/1/2** constant within a trial | | PASS |
| **output 3**, every trial × bin | per-bin mean y of frames with `likelihood > 0.9`, re-binned frame by frame with an explicit Python loop, then the 40th/60th percentiles recomputed | PASS, exact equality, percentiles identical to 3 decimals |
| **curation** | no auto/free-water trial kept; every kept trial's `start_time` is in the unit's `obs_intervals`; trial indices strictly increasing | PASS |
| **metadata** | per-neuron CCF coordinates == peak electrode's `x,y,z`; per-neuron annotation == `units['anno_name']`; mouse id == `subject.description` | PASS |

### Check 3 — reference code comparison
| Stage | Reference (`/app/code`) | This conversion | Same? |
|---|---|---|---|
| (a) **Data loading** | `preprocessing_utils.loadmat` on the DataJoint `.mat` export | `pynwb.NWBHDF5IO` on the NWB release (the task requires `pynwb`; the `.mat` export is not available) | different *interface*, same variables — each reference field has an identified NWB counterpart (Step 5 table) |
| (b1) **Neuron filtering** | `qc_mode='classifier'` → `goodunits/*.mat` lists per region; units without a CCF label dropped; units needing both ephys and histology | `units['classification'] == 'good'`, which is the materialised output of the same classifier; all such units carry a CCF annotation and an electrode | same |
| (b2) **Trial filtering** | `get_regular_trial_mask`: no early lick, no auto-water, no free-water, no no-response, no photostim | auto-water/free-water dropped; **early-lick, no-response and photostim trials kept**; additionally trials outside `obs_intervals` and trials without video dropped | deliberate deviation, forced by the decoder specification (those three variables are the outputs/input to be decoded); the extra two rules remove data that is *missing*, not merely atypical |
| (b3) **Session filtering** | not in the code (applied upstream) | data paper's criteria: performance > 65 %, ≥ 50 correct lick-left and lick-right, ≥ 1 good unit | same as the paper |
| (c) **Temporal alignment** | spike times already go-cue-relative in the `.mat` export; markers aligned as `i*0.0034 − go_time` | subtract `BehavioralEvents/go_start_times` (one per trial, verified) from absolute spike and video times | same |
| (d) **Binning** | `sliding_histogram`, `bw = 0.04 s`, `stride = 0.0034 s`, rate = count / bw | non-overlapping 50 ms bins over [−2.5, +1.5) s, rate = count / 0.05 s | deviation in bin width/stride, required by the task ("50-ms-width bins"); the rate convention (count / bin width, in Hz) is kept |
| (e) **Input construction** | `task_sample_time`, `task_stimulation` re-referenced to the go cue | same variables, from the NWB event timestamps | same |
| (f) **Output construction** | `trial_type` (instruction), `correctness` (1/0/−1), `early_lick_trials`, marker `tongue_y` | `trial_instruction`, `outcome` (hit/miss/ignore ≡ 1/0/−1), `early_lick`, `Camera0_side_TongueTracking` y | same |
| **Region grouping** | 14 coarse groups, hemisphere from CCF ML vs 5700 µm | same 14 groups, rebuilt from the CCF annotation (`ccf_regions.py`); hemisphere with the same 5700 µm midline | same, validated against the paper's Fig. 1E counts |
| **Marker outlier handling** | 5-σ velocity threshold + imputation; occluded tongue set to its mean | occluded frames are *excluded* and their bins labelled class 3 | deliberate: the task defines "not visible" as its own output class, so imputing the mean would destroy exactly the distinction being decoded. Within visible frames the discretisation is into 3 coarse percentile bands, which is insensitive to the residual outliers the 5-σ rule would catch. |

### Check 4 — key statistics comparison
See the table in Step 9. Nine of the eleven per-region unit counts in the data paper's Fig. 1E
are reproduced **exactly** on this release, the tenth (striatum) exactly as well, and ALM to
within 3.1 % (ALM is not an Allen CCF structure; see `ccf_regions.py`). The behavioural
performance of the selected sessions (83.8 %, range 65.8–98.9 %) matches the paper's
"84 % (range 65–99 %)". Trials per session excluding early licks (460) matches the paper's 476
to 3 %. The only statistic that is not matched is the paper's *range* of trials per session
(130–785 vs our 203–687), which is a property of the released session set, not of the
conversion.

Investigations run for this check:
- **Performance definition.** Computing performance with vs. without auto-water/free-water
  trials in the denominator: including them gives mean 83.2 %, range [65.1, 97.0]; excluding
  them gives mean 83.9 %, range [65.8, 98.9]. The latter matches the paper's "84 %, 65–99 %",
  so auto/free-water trials are excluded (which is also what the reference
  `get_regular_trial_mask` does).
- **Tongue likelihood threshold.** Fraction of frames above threshold: 0.1216 (0.5), 0.1168
  (0.9), 0.1153 (0.95), 0.1120 (0.99), 0.1094 (0.999) — a 1 % change in visible frames across
  two orders of magnitude of threshold, so the choice of 0.9 is immaterial.

### Check 5 — edge cases
| Edge case | Handling |
|---|---|
| Ephys stops before the behavioural session ends (8 sessions) | `recorded_trials()`; 736 trials removed. Verified: **0 all-silent trials remain** in the final dataset. |
| Video stops before the session ends (3 sessions) | trials with no frame in the window removed (239 trials); one session (SC066_20210413_112028_s6, 7 usable trials) then fails the behavioural criteria and is dropped entirely. |
| Session with 0 good units (SC017_20190216_162508_s4) | rejected explicitly. |
| Trial window extends outside `[trial start, trial stop]` | unavoidable (see Step 4); 2.9 % of bins on average, at most 8.5 % in any session. Those bins hold rate 0 and the limitation is recorded in `metadata['known_limitation']`. |
| Early-lick replays give several sample/delay onsets per trial | "tone onset" = the **last** `sample_start_times` ≤ go cue; asserted to lie inside the trial for every trial of every session. |
| Trials with no photostim / sessions with no photostim (3) | `photostim_on` is all zeros; `verify_data_format` accepts this and the decoder's other sessions provide the contrast. |
| Photostim offset 1 ms *after* the go cue (34.6 % of stim events) | kept as-is; it affects the single bin [0, 50 ms) and reflects the real stimulus. |
| Bin-edge convention | bin *b* is `[-2.5 + 0.05b, -2.5 + 0.05(b+1))`, half-open, so no spike or frame is double counted; verified by the exact re-binning sanity check. |
| Session with no `ignore` trials (1 session) | fine — `ncategories` is taken over the whole dataset by the decoder. |
| `is_good_trials` False for a few unit×trial pairs (0.02 %) | not used, as in the reference; documented in Step 3. |
| Sessions with identical mouse but different subject_id | none: `subject_id` ↔ `subject.description` is 1:1 (28 ↔ 28). |

### Issues Found and Resolved
1. **Trials outside the ephys recording produced all-zero neural data.** Found in Step 7 when a
   processing plot showed an empty raster. In 8 sessions the units' `obs_intervals` cover only a
   contiguous prefix (or, in one session, suffix) of the trials. Fixed with `recorded_trials()`.
   Re-check: 0 all-silent trials in the full dataset, and the sample plots now show spikes in
   every trial.
2. **Session curation was evaluated on all trials rather than the curated ones.** After fix 1,
   performance and the ≥50-correct criterion are computed on the trials that survive trial
   curation, which (a) is self-consistent and (b) automatically removes the session whose video
   failed (7 usable trials). Re-check: selected-session performance moved from
   83.2 % / [65.1, 97.0] to 83.8 % / [65.8, 98.9], i.e. closer to the paper's 84 % / [65, 99].
   Sessions: 150 → 142.
3. **`Infracerebellar nucleus`, `Copula pyramidis`, the hypo/sub-thalamic names and the
   basomedial/basolateral amygdalar names were mis-grouped** by substring collisions in the
   first version of `ccf_regions.py`. Fixed with an explicit `_PRE_RULES` disambiguation list.
   Re-check: nine region counts then matched the paper's Fig. 1E exactly (they were off by
   3 / 169 / 213 / 72 units before).

After each fix the conversion, the format verification and all sanity checks were re-run; the
results quoted in Steps 9–12 are from the final version.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`

### Training Progress
- Loss decreasing: **yes**, monotonically, 16.6 (epoch 1) → 0.655 (epoch 200); test loss 0.662.
- Device: CUDA (NVIDIA L4). Wall-clock ≈ 5 min.

### Decoder Results (Full, 142 sessions / 73,845 trials / 57,023 neurons)
| Output | Chance (1/k) | Majority class | Training Balanced Acc | Validation Balanced Acc | Val / chance |
|--------|--------------|----------------|-----------------------|-------------------------|--------------|
| lick_direction_choice | 0.333 | 0.447 | 0.7165 | **0.6817** | 2.05× |
| outcome | 0.333 | 0.737 | 0.7036 | **0.6555** | 1.97× |
| early_lick | 0.500 | 0.885 | 0.7826 | **0.7513** | 1.50× |
| tongue_y_position | 0.250 | 0.740 | 0.6924 | **0.6552** | 2.62× |

Every output is well above chance, and above the majority-class rate for choice
(0.682 vs 0.447) and tongue position (0.655 vs 0.740 on the *balanced* metric, i.e. far above
the 0.25 that a majority-class predictor scores on balanced accuracy).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — accuracy vs chance
| Output | #classes | Chance | Validation balanced acc | Ratio |
|---|---|---|---|---|
| lick_direction_choice | 3 | 0.333 | 0.682 | 2.05× |
| outcome | 3 | 0.333 | 0.656 | 1.97× |
| early_lick | 2 | 0.500 | 0.751 | 1.50× |
| tongue_y_position | 4 | 0.250 | 0.655 | 2.62× |

No output is below chance and none is below 1.5× chance except `early_lick`, which is exactly
1.50× — and for a binary variable 1.5× chance means balanced accuracy 0.75, i.e. a d′ of about
1.35, which is a strong effect, not a marginal one. The 1.5×-chance heuristic is not meaningful
for a two-class output (its maximum possible ratio is 2.0×).

### Check 2 — comparison with accuracies reported in the papers
Neither paper reports a multi-class decoding accuracy for these four variables from population
spiking, so there is no like-for-like number to match. The closest published quantities are
choice decoding AUCs. To compare on the papers' own terms I ran an independent epoch-wise
choice decoder (L2 logistic regression on mean firing rates, 5-fold CV, 12 random sessions,
lick-left vs lick-right trials only — the paper's setting):

| Quantity | This dataset | Papers |
|---|---|---|
| Choice AUC, response epoch (+0.2 to +1.0 s) | **0.998 ± 0.002** (n = 12 sessions) | 0.99 ± 0.01 from video, 2nd half of the response epoch (method paper) |
| Choice AUC, delay epoch (−1.0 to 0 s) | **0.916 ± 0.029** | 0.66 ± 0.12 from video over sample+delay; single-neuron delay choice AUC > 0.65 used as the "choice-modulated" threshold (method paper) |

Neural choice decoding reaches the papers' video-based ceiling after the go cue and is far
above the video-based delay value (as expected — the delay-epoch choice signal is explicitly
what the data paper reports as widespread neural preparatory activity, while the video only
sees uninstructed movements). This is strong independent evidence that the go-cue alignment,
the choice labels and the firing rates are all correct.

The provided decoder's 0.682 balanced accuracy for choice is an **average over all 80 bins**,
including the 2.5 s before the go cue where the choice is undefined for the first ~0.6 s and
only gradually becomes decodable. The per-bin validation accuracy (`/app/accuracy_vs_time.png`,
`/app/cache/accuracy_vs_time.npz`) shows the expected time course: near chance before the tone,
rising through sample and delay, and saturating after the go cue. There is therefore no gap to
the papers to explain.

### Check 3 — train vs validation gap
| Output | Train | Validation | Train / Val |
|---|---|---|---|
| lick_direction_choice | 0.7165 | 0.6817 | 1.05 |
| outcome | 0.7036 | 0.6555 | 1.07 |
| early_lick | 0.7826 | 0.7513 | 1.04 |
| tongue_y_position | 0.6924 | 0.6552 | 1.06 |

All ratios are ≤ 1.07, far below the 1.5 threshold: no overfitting and no sign of leakage
(the split is by trial within session, and every output is either a per-trial label or derived
from the video, never from the neural data).

### Additional debugging performed
1. **Output values verified on specific trials** — `sanity_checks.py` re-derives all four
   outputs for *every* trial of 6 random sessions directly from the NWB; exact match.
2. **Temporal alignment verified by plotting** — panel 3 of each processing figure shows the
   population PSTH jumping at exactly t = 0 and splitting by lick direction afterwards; panel 5
   shows the tongue becoming visible only around licks.
3. **Output variation** — no output is degenerate: the rarest class is `early_lick = yes`
   at 11.5 % and `outcome = ignore` at 10.9 % of all bins.
4. **Neural filtering** follows the reference (classifier "good" units) and the per-region
   counts reproduce the paper exactly.
5. **Processing matches the reference** — see Step 10, Check 3.

### Issues Found and Resolved
None remaining. The three issues found in Step 10 were fixed there and all checks were re-run
afterwards.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created
- [x] `cache/` folder created with `README_CACHE.md`
- [x] All files organised
