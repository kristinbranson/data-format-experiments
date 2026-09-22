# Dataset Conversion Notes

## Overview
- **Dataset**: DANDI:000363 "Mesoscale Activity Map Dataset" (Chen, Nguyen, Li, Svoboda 2023).
  Neuropixels recordings + DeepLabCut orofacial tracking during an auditory delayed-response
  (memory-guided directional licking) task. 174 NWB files (`/app/data/sub-*/*.nwb`, 50 GB).
- **Papers**:
  - `datapaper.pdf` — Chen et al., "Brain-wide neural activity underlying memory-guided movement" (Cell 2024).
  - `methodpaper.pdf` — Wang, Kurgyis et al., "Brain-wide analysis reveals movement encoding
    structured across and within brain areas" (Nat Neurosci 2025).
  - `ChenLiuEtAl2023_SpikeSortingQC.pdf` — spike sorting / QC white paper.
- **Reference code**: `/app/code` (MapVideoAnalysis repository from the method paper).
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`).

---

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `convert_data.py` (to be written)
- `datapaper.pdf`, `methodpaper.pdf`, `ChenLiuEtAl2023_SpikeSortingQC.pdf`, `methods.txt`
- `code/` — reference analysis code (`VideoAnalysisUtils/`, `Sherlock/`, `Notebooks/`, `Archive/`)
- `data/` — 28 subject directories `sub-<id>/`, 174 `*.nwb` files, `dandiset.yaml`
- `decoder.py`, `train_decoder.py` — provided decoder / validation code
- `Dockerfile`, `docker-compose.yaml`, `.manifest`

Environment verified: `python3` 3.13, numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0, pynwb 4.1.0.
Hardware: 128 CPUs, ~1 TB RAM, NVIDIA L4 (23 GB).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` | `VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Load the DataJoint-exported `.mat` session/probe files |
| `process_all_sess_parallel` / `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Concatenate all probes of a session; pull behavior/task fields, spike times, unit info, QC metrics, CCF histology |
| `helper_get_neuron_id_area` | `preprocessing_DJ_2022Aug.py` | CURATION | Keep only units on the QC "good unit" list for a region; split hemisphere at CCF ML = 5700 µm; drop units without a CCF label |
| `process_one_area` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Truncate per-trial spike times to `[begin_time, end_time]` **relative to go cue** and bin them |
| `sliding_histogram` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Sliding-window spike counts → firing rate (`counts / bin_width`) |
| `preprocess_all_ephys.py` | `Sherlock/` | PROCESSING | The actual settings used for the paper: `bw = 0.04 s`, `stride = 0.0034 s`, `begin_time = -3.0 s`, `end_time = 3.0 s`, `qc_mode = 'classifier'` |
| `align_markers_between_lims` | `Sherlock/align_markers.py` | PROCESSING | Align DeepLabCut markers (incl. `tongue_x`, `tongue_y`) to the go cue, `t_min = -3`, `t_max = 1.5`, frame period `dt = 0.0034 s` |
| `get_bad_trial_inds` | `Sherlock/align_markers.py` | CURATION | Flags trials whose video frame count disagrees with the trial duration |
| `load_session` | `VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Reassemble a preprocessed session (`fr`, `ccf_coordinate`, `ccf_label`, trial variables) |
| `get_regular_trial_mask` | `population_decoding_utils.py` | CURATION | "Regular" trials = no early lick, no auto-water, no free-water, response given (`correctness != -1`), no photostimulation |
| `nested_cross_validation` | `population_decoding_utils.py` | ANALYSIS | Population decoding of binary trial labels (PCA-16 + logistic regression, 5 outer folds) |
| `check_fr` | `preprocessing_utils.py` | CURATION | Drop neurons with zero across-trial variance |
| `get_period` | `preprocessing_utils.py` | REFERENCE | Epoch definitions relative to go cue: sample `[-1.9, -1.2]`, delay `[-1.2, 0.0]`, post-go `[0.0, 1.0]` |

### Notes
- The reference pipeline consumes **DataJoint `.mat` exports**, not the published NWB files. Every
  field it uses has a direct NWB counterpart (table below in Step 4), so the logic transfers.
- **Time base**: in the `.mat` export the per-trial spike times are already expressed relative to
  the go cue. In NWB, `units/spike_times` are absolute session times, so we subtract the trial's go
  cue time ourselves. This is the same alignment.
- The reference truncation window (`-3 s` … `+3/3.5 s` around the go cue) is *wider than a typical
  trial* (median trial start is 3.15 s before, and median trial stop 1.8 s after the go cue),
  confirming that per-trial spike trains in the export are not clipped at NWB `start_time`/
  `stop_time`. We therefore slice spikes by absolute session time, which reproduces this.
- Neuron curation in the reference = units on the classifier-based "good unit" list **and** having a
  CCF annotation. In NWB this is `units/classification == 'good'` (`units/anno_name` is then always
  non-empty — verified: 0 of 69,453 good units lack an annotation).
- `get_regular_trial_mask` is the *analysis* mask used for the figures of the method paper. It cannot
  be applied here: the decoder task explicitly requires photostimulation as an input and
  early-lick / no-response (ignore) trials as outputs. See Step 5 for what we do keep.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data/sub-<subject_id>/sub-<subject_id>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys+ogen.nwb`
— one NWB (HDF5) file per recording session, 174 files, 28 subjects, 50 GB total.

Relevant contents of each file:

| NWB path | Contents |
|----------|----------|
| `identifier` | e.g. `SC015_20190207_120657_s1` (mouse name + date + time + session no.) |
| `general/subject/subject_id` | numeric DANDI subject id, e.g. `440956` |
| `intervals/trials/` | `id`, `start_time`, `stop_time`, `trial_instruction` (`left`/`right`), `outcome` (`hit`/`miss`/`ignore`), `early_lick` (`early`/`no early`), `auto_water`, `free_water`, `photostim_onset`/`_duration`/`_power` (strings, `'N/A'` if none), `task` (all `audio delay`), `task_protocol` (all 1) |
| `acquisition/BehavioralEvents/` | `presample/sample/delay/go/trialend_{start,stop}_times` (TimeSeries with session-time `timestamps`), `left_lick_times`, `right_lick_times`, `photostim_{start,stop}_times` |
| `acquisition/BehavioralTimeSeries/` | `Camera0_side_{Jaw,Nose,Tongue}Tracking`, each `data` = (n_frames, 3) = (x, y, DLC likelihood), `timestamps` = session time, 300 Hz (dt = 0.0034 s) |
| `units/` | `spike_times` + `spike_times_index` (ragged, absolute session time), `classification` (`good`/`unlabelled`), `anno_name` (CCF structure name), `unit_quality`, `is_good_trials` (n_units × n_trials), 15 QC metrics, `electrodes` (index into electrode table) |
| `general/extracellular_ephys/electrodes` | `x`, `y`, `z` = CCF coordinates (ML, DV, AP) in µm, `location` = JSON with the targeted `brain_regions` (e.g. `"left ALM"`) |

### Dataset Size (from data files, all 174 sessions)
| Statistic | Value |
|-----------|-------|
| NWB files / sessions | 174 |
| Subjects (mice) | 28 |
| Sessions / subject | 1–11 (mean 6.2) |
| Units (all clusters) | 272,227 |
| Units `classification == 'good'` | **69,453** (25.5 % of all clusters) |
| Good units / session | mean 399, range 29–1000+ |
| Trials (total) | 94,990 |
| Trials / session | mean 546, range 264–800 |
| Trials excluding no-response (`ignore`) | mean 465, range 104–785 |
| Photostimulation trials | 18,588 (19.6 %), 168/174 sessions |
| Auto-water trials | 1,339 (1.4 %) |
| Free-water trials | 2,450 (2.6 %) |
| Video frame rate | 300 Hz (dt = 0.0034 s), present in all 174 sessions |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (`methods.txt`) |
| Fraction of Kilosort2 clusters kept | 25.9 % | "This corresponds to 25.9 % of clusters reported by Kilosort2." |
| Sessions | 173 | same sentence |
| Probe insertions | 655 (660 in Fig. 1J) | same sentence |
| Subjects | 28 | Fig. 1J caption: "aggregated over 660 penetrations, 173 behavioral sessions, and 28 mice" |
| Units per brain area | ALM 8717, orbital 10223, striatum 7664, pallidum 1092, thalamus 12808, midbrain 7495, medulla 2928, pons 347, cerebellum 1820, hypothalamus 815, hippocampus 1944, other cortex 7993, olfactory 4137, cortical subplate 1960 (Σ = 69,943) | datapaper Fig. 1J / Fig. 2 |
| Trials per session | mean 476, range 130–785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84 %, range 65–99 % | "with 84% correct rate (range, 65-99%)" |
| Session inclusion | performance > 65 %, ≥ 50 correct lick-left and ≥ 50 correct lick-right trials | "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." |
| Performance definition | fraction correct of **control** (no-photostim) trials, **excluding early-lick trials** | "Overall performance was computed as the fraction of correct control trials (i.e. no photostimulation), excluding any early lick trials." |
| Sample epoch | 3 tones × 150 ms with 100 ms gaps = 0.65 s | `methods.txt` behavior section |
| Delay epoch | 1.2 s | "The sample epoch was followed by a 1.2 s delay epoch." |
| Go cue | 6 kHz, 0.1 s | `methods.txt` |
| Answer period | 1.5 s | "During the response epoch (answer period: 1.5 s)" |
| Photoinhibition | ~25 % of trials, ALM, 5 mW/hemisphere, 0.5 s incl. 100 ms ramp-down, **always ends before the go cue** | `methods.txt` photoinhibition section |
| Photoinhibition effect | performance 83.2 % → 71.7 % (17 mice, 93 sessions) | `methods.txt` |
| Neural binning (method paper) | 40 ms width, 3.4 ms stride | "we binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Video | 300 Hz side view, DeepLabCut markers for jaw / nose / tongue | method paper Methods |
| Tongue occlusion | "When the tongue was occluded while it was in the mouth, as was typically the case before the response epoch, we set the tongue position to its mean value." | method paper Methods |
| Sessions used for video analyses | 105–106 | method paper Results |

### Processing Details
- **Temporal alignment**: everything is aligned to the **go cue** (`task_cue_time` in the export;
  `go_start_times` in NWB). The reference marker alignment also uses the go cue with a window of
  −3 … +1.5 s. Our task window is −2.5 … +1.5 s.
- **Temporal binning**: reference 40 ms / 3.4 ms sliding; the Decoder Task here mandates **50 ms
  bins** → 80 non-overlapping bins over the 4 s window. Firing rate = spike count / 0.05 s (Hz),
  exactly the reference's `sliding_histogram(..., rate=True)` convention.
- **Trial structure**: presample → sample (0.65 s) → delay (1.2 s) → go cue → response (1.5 s) →
  trial end. Licking during sample/delay triggers a *replay* of the epoch, so a trial can have more
  than one `sample_start_times` / `delay_start_times` event (5.8 % of trials).

### Curation Steps

**Neuron curation rules (reference)**:
1. Unit must be labelled `good` by the region-specific logistic-regression QC classifier
   (`classification == 'good'`; this is the "8717 from ALM …" set).
2. Unit must have a CCF histology annotation (`anno_name` non-empty) — automatically satisfied.
3. (Method paper, analysis-level only) brain areas with < 10 neurons in a session were dropped from
   *per-area* analyses; not applicable here because we pool all neurons of a session.

**Trial curation rules (reference)**:
- `get_regular_trial_mask`: no early lick, no auto-water, no free-water, response given, no
  photostimulation — used for the figure analyses.
- Data paper: "Early lick trials and no response trials were excluded for analysis."

**Session curation rules (reference)**: performance > 65 % and ≥ 50 correct lick-left and
lick-right trials.

### Decoders Trained (reference)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Trial type / choice from **video** (pre-sample) | embedding + logistic regression | AUC 0.51 ± 0.06 (chance) |
| Trial type / choice from **video** (2nd half sample + delay) | same | AUC 0.66 ± 0.12 |
| Trial type / choice from **video** (2nd half of response epoch) | same | AUC 0.99 ± 0.01 |
| Choice from **neural activity** (medulla population, delay) | PCA + logistic regression | reported per-session AUC, Fig. 7 |

No paper reports a neural decoder for outcome, early lick or tongue position, so the only
quantitative expectation is that **choice decoding after the go cue should be near-perfect**
(video reaches AUC 0.99; neural recordings include ALM, which is strongly choice-selective).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### NWB ↔ DataJoint-export field mapping (reference code → NWB)
| Reference `.mat` field | NWB equivalent | Verified |
|---|---|---|
| `task_cue_time[0]` (go cue) | `acquisition/BehavioralEvents/go_start_times.timestamps` | exactly one per trial, inside `[start_time, stop_time]`, in all 174 sessions |
| `behavior_report` (1 correct / 0 error / −1 no response) | `intervals/trials/outcome` (`hit` / `miss` / `ignore`) | outcome value sets match |
| `task_trial_type` (`l`/`r`) | `intervals/trials/trial_instruction` (`left`/`right`) | — |
| `behavior_early_report` | `intervals/trials/early_lick` (`early` / `no early`) | — |
| `behavior_is_auto_water`, `behavior_is_free_water` | `intervals/trials/auto_water`, `free_water` | — |
| `task_stimulation` = [power, type, on, off] | `intervals/trials/photostim_{power,onset,duration}` + `BehavioralEvents/photostim_{start,stop}_times` | per-session stim-trial counts equal event counts in all 174 sessions |
| `behavior_lick_times` / `behavior_lick_directions` | `BehavioralEvents/left_lick_times`, `right_lick_times` | — |
| `task_sample_time` | `BehavioralEvents/sample_start_times` | median onset = −1.85 s re. go cue in **all** 174 sessions (= 0.65 s sample + 1.2 s delay ✓) |
| `neuron_single_units` (spike times re. go cue) | `units/spike_times` (absolute) − go cue time | — |
| `neuron_unit_info`, `neuron_unit_quality_control` | `units/*` QC metric columns | — |
| QC "good unit" index lists (`goodunits/*.mat`) | `units/classification == 'good'` | reproduces the paper's per-area counts exactly (below) |
| `histology.annotation` | `units/anno_name` | all 293 distinct names resolve in the Allen CCF ontology |
| `histology.ccf_x/y/z` | electrode `x`/`y`/`z` of the unit's peak electrode | ML midline 5700 µm and bregma offset (5700, 0, 5400) of `population_decoding_utils.get_ventral_medial_mask` are consistent with these coordinates |
| `tracking.camera_0_side.tongue_x/y` | `BehavioralTimeSeries/Camera0_side_TongueTracking.data[:, 0:2]`, likelihood in column 2 | — |

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Number of sessions | – | 174 NWB files | 173 behavioral sessions | The published dandiset has one extra file; all statistics below are computed on the 174 files and still match the paper. Not an error on our side. |
| Total good units | – | 69,453 | 69,943 | 0.7 % fewer. The entire difference (490) is in cortex (see per-area table below); every subcortical area matches **exactly**. Different dandiset version; nothing to fix. |
| Trials/session | – | mean 546 (all trials), mean 465 (excl. no-response) | mean 476, range 130–785 | Resolved: the paper's number counts trials **excluding no-response trials** and only over sessions passing the > 65 % performance criterion. Applying exactly that gives **mean 478.6, range 130–785** and **performance 83.7 %, range 65.8–98.9 %** — matching "476 (130–785)" and "84 % (65–99 %)". |
| Neural bin | 40 ms / 3.4 ms stride | – | 40 ms / 3.4 ms | Decoder Task mandates 50 ms bins → deviation required by the task specification. |
| Trial exclusions | `get_regular_trial_mask` removes early-lick, no-response and photostim trials | – | "Early lick trials and no response trials were excluded for analysis" | Cannot be applied: these are exactly the variables the decoder must predict / receive. See Step 5 decision 4. |
| Tongue occlusion | – | DLC likelihood is strongly bimodal (10.5 % of frames > 0.9, ~89.5 % < 1e-3) | "set the tongue position to its mean value" when occluded | The Decoder Task instead defines an explicit 4th class "not visible", which supersedes the reference's imputation. Threshold 0.9 (any threshold in 1e-3 … 0.99 changes the visible fraction by < 0.3 %). |

### Consistency check: per-area good-unit counts
Coarse areas were reconstructed from `units/anno_name` using the **Allen CCF structure graph**
(downloaded to `cache/allen_structure_graph.json`, mapping code in `cache/region_map.py`):
a unit belongs to the first matching ancestor among MY, P, MB, CB, TH, HY, STR, PAL, HPF, OLF,
CTXsp, ORB, Isocortex.

| Area | Paper (Fig. 1J) | This conversion (all 174 sessions) | Match |
|---|---|---|---|
| Thalamus | 12,808 | **12,808** | exact |
| Orbital | 10,223 | **10,223** | exact |
| Striatum | 7,664 | **7,664** | exact |
| Midbrain | 7,495 | **7,495** | exact |
| Olfactory | 4,137 | **4,137** | exact |
| Medulla | 2,928 | **2,928** | exact |
| Cortical subplate | 1,960 | **1,960** | exact |
| Hippocampus | 1,944 | **1,944** | exact |
| Cerebellum | 1,820 | **1,820** | exact |
| Pallidum | 1,092 | **1,092** | exact |
| Hypothalamus | 815 | **815** | exact |
| Pons | 347 | **347** | exact |
| ALM + other cortex | 8,717 + 7,993 = 16,710 | 16,220 | −490 (dandiset version) |
| **Total** | 69,943 | 69,453 | −490 |

This is a decisive confirmation that `classification == 'good'` is the paper's QC criterion and that
the CCF-ontology grouping reproduces the paper's area definitions.

**ALM**: the reference splits ALM out of cortex with a voxel mask (`ALM_voxels_symmetric.npy`, not
distributed). Reconstructing it as *cortical units more than 2.0 mm anterior to bregma*
(CCF AP coordinate `z < 3400 µm`) yields **8,728** units versus the paper's 8,717 (0.1 %
difference); all such units also fall within |ML| < 2.2 mm of the midline, i.e. the ALM location
(AP 2.5, ML 1.5) of `methods.txt`. Thresholds of 1.9 / 2.1 mm give 9,095 / 8,072, so 2.0 mm is
clearly the intended boundary.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial window and binning
- Alignment event: **go cue onset** = `BehavioralEvents/go_start_times.timestamps[trial]`.
- Window: `off_start = -2.5 s`, `off_end = +1.5 s` → 4.0 s.
- 50 ms non-overlapping bins → **80 time points**; bin *k* covers `[-2.5 + 0.05k, -2.45 + 0.05k)`
  relative to the go cue; bin centre = `-2.475 + 0.05k`.

### Variable Mapping
| Source (NWB) | Target field | Transform | Reference analogue |
|---|---|---|---|
| `units/spike_times` (ragged, absolute) − go cue | `neural[s][t]` (n_neurons, 80), float32 | histogram into 80 × 50 ms bins, divide by 0.05 → Hz | `sliding_histogram(..., rate=True)` |
| `units/classification == 'good'` | neuron mask | keep only `good` | classifier QC lists |
| `units/anno_name` → Allen ontology → coarse area (+ ALM by AP > 2.0 mm) | `brain_region_idx[s]` | 14 coarse areas | `helper_get_neuron_id_area` |
| `general/subject/subject_id` | `subjects`, `subject_idx` | one entry per mouse | — |
| last `sample_start_times` before the go cue | `input[s][t][0]` — `time_from_tone_onset` (s) | `bin_centre − tone_onset`, continuous | `task_sample_time` |
| `trials/photostim_onset` (+ `start_time`) and `photostim_duration` | `input[s][t][1]` — `photostim` | 1 if the bin centre lies in `[onset, onset+duration)`, else 0 | `task_stimulation[:, 2:4]` |
| `trials/outcome` + `trials/trial_instruction` | `output[s][t][0]` — `choice` | `ignore`→`no lick`(2); `hit`→instruction; `miss`→opposite instruction | `behavior_report` + `task_trial_type` |
| `trials/outcome` | `output[s][t][1]` — `outcome` | `ignore`→0, `miss`→1, `hit`→2 | `behavior_report` |
| `trials/early_lick` | `output[s][t][2]` — `early_lick` | `no early`→0, `early`→1 | `behavior_early_report` |
| `Camera0_side_TongueTracking` (y, likelihood) | `output[s][t][3]` — `tongue_y` | see below | `align_markers.py` (`tongue_y`) |

All inputs and outputs are **time-varying**, shape (2, 80) and (4, 80). Per-trial variables
(choice, outcome, early lick) are held constant across the 80 bins, as allowed and encouraged by the
target format ("If at all possible, make it time-varying").

### Tongue y discretisation (per the Decoder Task)
1. A video frame counts as *tongue visible* if `tongue_likelihood > 0.9` (DLC default cut-off; the
   likelihood is strongly bimodal so this is not a sensitive choice).
2. For each 50 ms bin, take the frames whose timestamps fall in the bin. If at least one is visible,
   the bin's y = mean y over the **visible** frames; otherwise the bin is *not visible* → class 3.
   (Tongue protrusions last ~50–100 ms, so requiring a majority of the ~15 frames per bin would
   discard the onset and offset of most licks.)
3. Per-session 40th and 60th percentiles are computed over **all visible bins of that session**
   (this is the "y-position over the session"), then
   `0: y < p40`, `1: p40 ≤ y ≤ p60`, `2: y > p60`, `3: not visible`.

### Key Decisions
1. **Neuron curation = `classification == 'good'`.** Validated by reproducing the paper's per-area
   unit counts exactly (Step 4 table). No further QC-metric thresholds are applied, because the
   classifier already consumes all 15 metrics (white paper / `methods.txt`).
2. **`units/is_good_trials`**: only 0.14 % of (unit, trial) pairs and 565 / 69,453 good units are
   affected. The target format requires a rectangular (n_neurons, 80) array per trial, so per-trial
   unit exclusion is impossible; we keep all good units and note the negligible contamination.
3. **Session curation = the data paper's criteria**: control-trial performance > 65 % and ≥ 50
   correct lick-left and ≥ 50 correct lick-right trials → **151 of 174 sessions, 28 mice**. This
   reproduces the paper's trial and performance statistics exactly (Step 4).
4. **Trial curation**: we keep *all* trials of the selected sessions **except** auto-water and
   free-water trials (3.7 %). Rationale: on those trials the animal is given water irrespective of
   its action, so `outcome`/`choice` do not reflect a decision — the reference excludes them in
   `get_regular_trial_mask`. Early-lick, no-response (ignore) and photostimulation trials are
   *deliberately kept*, because the Decoder Task requires early lick and "ignore"/"no lick" as output
   classes and photostimulation as an input; excluding them would make those classes empty.
5. **Spikes are sliced in continuous session time**, not clipped at NWB `start_time`/`stop_time`.
   The reference does the same (its ±3 s window exceeds the trial duration). 15.6 % of trials stop
   less than 1.5 s after the go cue, so clipping would inject artefactual zero firing rates.
6. **Video frames are likewise taken by absolute session time.** The camera pauses for ~0.5 s
   between trials, so a few bins genuinely have no frames; those become class 3 ("not visible"),
   which is the only available representation of missing tracking.
7. **Firing rates in Hz** (count / 0.05 s), float32 — matches `sliding_histogram(rate=True)`.
8. **Brain regions**: the 14 coarse areas of datapaper Fig. 1J
   (`ALM, OtherCortex, Orbital, Olfactory, CorticalSubplate, Hippocampus, Striatum, Pallidum,
   Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum`). Hemisphere is *not* split
   (the paper's summary statistics are not hemisphere-split either).

### Planned Sanity Checks
- [x] Good-unit counts per coarse area == datapaper Fig. 1J (done in Step 4).
- [x] Session selection reproduces "476 trials (130–785), 84 % correct (65–99 %)".
- [x] Median tone onset == −1.85 s re. go cue in every session.
- [x] Photostim trial count == photostim event count in every session; all stimulation ends at or
      before the go cue.
- [x] Choice derived from `outcome`+`trial_instruction` agrees with the first lick after the go cue
      (99.3–100 % over spot-checked sessions).
- [x] Tongue visibility is ≈ 0 before the go cue and jumps to ~0.4 immediately after it → video and
      ephys clocks agree.
- [ ] Spot-check spike counts, input and output values against the raw NWB (Step 10, Check 2).
- [ ] Population PSTH aligned to go cue shows the expected response-epoch increase.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`. Run as
`python -u /app/convert_data.py <outpickle> [--full|--sample] [--show-processing] [--njobs N]`.

Structure:

| Function | Purpose |
|---|---|
| `load_ontology`, `assign_brain_regions` | Allen CCF structure graph → 14 coarse areas + ALM carve-out |
| `read_trial_table` | trials table, go cue times, tone (sample-epoch) onsets |
| `session_performance`, `session_passes` | data-paper session inclusion criteria (+ ≥1 good unit, ≥2 usable trials) |
| `observed_trial_mask`, `trial_mask` | trials with ephys coverage, minus auto-/free-water trials |
| `load_good_units` | `classification == 'good'` units, their spike trains, CCF coordinates and regions |
| `bin_spikes` | 80 × 50 ms bins aligned to the go cue → Hz |
| `build_inputs` | time-from-tone-onset ramp and photostim indicator |
| `choice_codes`, `bin_tongue`, `discretize_tongue`, `build_outputs` | the four output variables |
| `plot_processing` | the `--show-processing` diagnostic figure |
| `process_session`, `main` | per-session driver and multiprocessing pool |

**Efficiency**: binning is fully vectorised — for each unit one `np.searchsorted` of the
`ntrials × 81` bin edges into its sorted spike train, then `np.diff`.  The tongue loop is
vectorised per trial with `np.bincount`.  Sessions are processed in parallel with a
`multiprocessing` pool (`--njobs`, default 16).

Code inefficiencies identified and removed:
- a naive per-(unit, trial, bin) `np.sum(mask)` (the reference `sliding_histogram`
  pattern) would be ~10^4 times slower; replaced by the searchsorted/diff formulation.
- per-trial firing-rate matrices are produced by slicing one preallocated
  `(ntrials, nneurons, 80)` block rather than by concatenating per-trial temporaries.

Measured: **0.2–2 s per session**, 22 s wall clock for all 150 sessions with 24 workers.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`processing_SC015_20190208_133600_s2.png`, `processing_SC015_20190209_150135_s3.png`.

### Sample Statistics (2 sessions of mouse SC015)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 (SC015) |
| Neurons (total) | 580 |
| Neurons / session | 375, 205 |
| Trials (total) | 676 |
| Trials / session | 159, 517 |
| Timepoints / trial | 80 (all trials) |
| `time_from_tone_onset` range | [-0.63, 5.48] s |
| `photostim` range | [0, 1], on in 3.2 % of all bins |
| `choice` distribution | [0.516, 0.462, 0.022] (left, right, no lick) |
| `outcome` distribution | [0.022, 0.084, 0.893] (ignore, miss, hit) |
| `early_lick` distribution | [0.978, 0.022] |
| `tongue_y` distribution | [0.103, 0.052, 0.103, 0.742] |

The tongue fractions confirm the discretisation: of the 25.8 % visible bins, exactly
40 % / 20 % / 40 % fall into classes 0 / 1 / 2.

### Processing Plots Review
The figure shows, per session: raw spike ticks against the binned rate (and an
independent `np.histogram` re-binning with max |difference| = 0); the population PSTH
(flat before, sharp rise right after the go cue); a z-scored neuron × time map;
the tone-onset ramp (zero crossing at −1.85 s); the photostim raster (always ending at
or before the go cue) cross-checked against the raw `photostim_start_times`; the raw
300 Hz tongue trace with the per-session percentile lines and the resulting class digits;
P(tongue visible) overlaid on the lick rate (both rise exactly at the go cue); the output
class fractions; the choice-versus-first-lick confusion matrix (agreement 1.000 in both
sessions); the per-region neuron counts; and three final `neural[trial]` matrices.

**Anomalies found and fixed during this review** (see also Step 10):
1. *All-zero neural matrices for 321 of 480 trials in session SC015_…_s2.* Cause: in 9 of
   the 174 sessions the electrophysiology only covers a contiguous block of the
   behavioral session. Fixed by masking trials with `units/obs_intervals`.
2. *One remaining all-zero trial* — the last trial listed in `obs_intervals` extended past
   the end of the recording. Fixed by dropping any trial in which not one neuron fires.
3. *Misleading photostim cross-check panel* — the raw events were being assigned to the
   preceding trial's go cue. Plot-only bug, fixed.

### Run Time Estimates
| Speed-ups implemented | Time savings |
|---|---|
| vectorised searchsorted binning instead of per-bin masks | ~10^3–10^4× on the binning step |
| single preallocated `(ntrials, nneurons, 80)` block | avoids a second full copy |
| `multiprocessing` pool over sessions (24 workers) | ~12× wall clock |

| Step | Time / session | Estimated total (150 sessions) |
|---|---|---|
| trials + go cues | 0.05 s | 8 s |
| unit loading (spike times) | 0.06–0.6 s | 30 s |
| spike binning | 0.1–0.8 s | 60 s |
| inputs + outputs (incl. video) | 0.16–0.5 s | 40 s |
| **total, serial** | **0.2–2 s** | **~2.5 min** |
| **total, 24 workers** | — | **22 s** (measured) |

Well under the 15-minute budget, so no further optimisation was needed.

### Verification
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only`
→ `/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**
All trials have T = 80; input and output ranges as listed above.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: None
- Warnings: None

### Training
Loss decreased monotonically from ~9.2 (epoch 4) to 0.52 (epoch 200); test loss 0.665.

### Decoder Results (Sample, 2 sessions)
| Output | Chance | Training balanced acc. | Validation balanced acc. |
|--------|--------|------------------------|--------------------------|
| choice | 0.333 | 0.778 | **0.622** |
| outcome | 0.333 | 0.780 | **0.549** |
| early_lick | 0.500 | 0.919 | **0.804** |
| tongue_y | 0.250 | 0.648 | **0.582** |

Every output is above chance. The sample contains only 676 trials of one mouse and very
few "no lick" / "ignore" / "early" trials (2.2 % each), so these numbers are expected to
be much lower than on the full dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --njobs 24`
→ `/app/conversion_full_out.txt` (43.4 s total, 22.2 s of conversion).

### Output Files
- `converted_data.pkl`: **10.21 GB**, 150 sessions, 77,521 trials, 59,749 neurons
- `verification_full_out.txt`: created, **no errors, no warnings**

### Session accounting
| | Sessions |
|---|---|
| NWB files in `/app/data` | 174 |
| excluded: performance ≤ 65 % | 23 |
| excluded: no unit passed quality control (`SC017_20190216_162508_s4`) | 1 |
| **converted** | **150** |

(No session was excluded by the "≥ 50 correct trials per direction" criterion — every
session that passes the performance criterion also passes that one.)

### Consistency Check
| Statistic | Reference papers | Reference data (all 174 NWB) | Converted data (150 sessions) | Match? |
|-----------|------------------|------------------------------|-------------------------------|--------|
| Subjects | 28 | 28 | 28 | ✔ |
| Sessions | 173 (published) / 151 pass the paper's criteria | 174 | 150 | ✔ |
| Good units, total | 69,943 | 69,453 | 59,749 (150 of 174 sessions) | ✔ (−0.7 % dandiset version) |
| Good units / session | ~404 | 399 | 398 | ✔ |
| Trials/session, excl. no-response, selected sessions | 476 (130–785) | 478.6 (130–785) | — | ✔ exact range |
| Behavioral trials/session (selected) | — | 544.4 | 543.9 | ✔ |
| Converted trials/session | — | — | 516.8 (159–796) | — |
| Performance (selected sessions) | 84 % (65–99 %) | 83.7 % (65.8–98.9 %) | 83.6 % (65.8–98.9 %) | ✔ |
| Photostim trials | "~25 %, randomly interleaved" | 19.6 % | 20.5 % | ✔ approx |
| ALM units | 8,717 | 8,728 | 7,720 | ✔ (0.1 % on the full set) |
| Thalamus units | 12,808 | **12,808** | 11,329 | ✔ exact on the full set |
| Orbital units | 10,223 | **10,223** | 9,226 | ✔ exact |
| Striatum units | 7,664 | **7,664** | 6,225 | ✔ exact |
| Midbrain units | 7,495 | **7,495** | 6,416 | ✔ exact |
| Olfactory units | 4,137 | **4,137** | 3,471 | ✔ exact |
| Medulla units | 2,928 | **2,928** | 2,866 | ✔ exact |
| Cortical subplate | 1,960 | **1,960** | 773 | ✔ exact |
| Hippocampus | 1,944 | **1,944** | 1,680 | ✔ exact |
| Cerebellum | 1,820 | **1,820** | 1,820 | ✔ exact |
| Pallidum | 1,092 | **1,092** | 1,021 | ✔ exact |
| Hypothalamus | 815 | **815** | 755 | ✔ exact |
| Pons | 347 | **347** | 339 | ✔ exact |
| `time_from_tone_onset` | tone at −1.85 s re. go cue | median −1.85 s in all 174 sessions | range [−1.53, 11.89] s, median 0-crossing at −1.85 s | ✔ |
| `photostim` | 0.5 s, ends before the go cue | onsets at −1.2 or −0.5 s | 1 in 2.56 % of bins, never after t = 0 | ✔ |
| `choice` | — | — | [0.443, 0.438, 0.120] | — |
| `outcome` | — | — | [0.120, 0.153, 0.728] | — |
| `early_lick` | — | — | [0.885, 0.115] | — |
| `tongue_y` | — | — | [0.102, 0.051, 0.102, 0.746] | ✔ 40/20/40 split of visible bins |

Additional data-quality statistics of the converted dataset:
- neurons with no spike in any trial: **0 / 59,749**
- trials with no spike from any neuron: **0** (1 dropped during conversion)
- per-neuron mean rate: median 4.45 Hz, mean 9.19 Hz, 99th pct 71.9 Hz
- P(tongue visible) = 0.074 before the go cue, 0.541 after it

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output log verification
`/app/verification_full_out.txt`: `Data format is valid, no errors or warnings.`
There is nothing left to address. (The only oddity is that the `Other` brain-region entry
has 0 neurons; it is kept deliberately as a fallback bucket in case an annotation ever
fails to resolve in the Allen ontology — all 293 annotations in this release resolve.)

### Check 2 — independent sanity checks against the raw NWB files
`cache/sanity_checks.py` re-derives every quantity directly from the HDF5 datasets with
independently written code (it does **not** import `convert_data.py`) and compares with
`np.allclose`. Run over 8 randomly chosen sessions (seed 0), 5 random spot checks each:

| Check | What is compared | Result |
|---|---|---|
| Trial correspondence | rebuilt keep-mask (auto/free-water + `obs_intervals` + silent-trial rule) vs number of converted trials | match in all 8 sessions |
| Neurons | `len(classification == 'good')` vs `neural[s][t].shape[0]` | match |
| **Neural** | `np.histogram(spike_times − go_cue, bins=edges)/0.05` for a random (trial, neuron) vs `neural[s][t][n]` | `allclose` ✔ (40 spot checks) |
| **Input 0** | `bin_centre − (last sample_start before the go cue − go cue)` | `allclose` ✔ |
| **Input 1** | `[trial start + photostim_onset, +photostim_duration)` | `allclose` ✔ |
| **Output 0–2** | choice from `outcome` × `trial_instruction`, outcome code, early-lick flag | `allclose` ✔ |
| **Output 3** | full re-binning of the 300 Hz tongue trace, re-computed session percentiles, re-discretised — compared for **all trials × all 80 bins** | `allclose` ✔ |
| Tongue percentiles | recomputed p40/p60 vs `metadata['session_info'][s]['tongue_percentile_values']` | `allclose` ✔ |
| Brain regions | every `ALM` unit has CCF AP > 2.0 mm, no `OtherCortex` unit does | ✔ |

`0 failures`.

Additional cross-checks built into the conversion itself:
- `read_trial_table` raises if the number of go cues differs from the number of trials.
- `observed_trial_mask` raises if `obs_intervals` differ between the first and the last
  unit of a session, or if they do not line up exactly with the trial table.
- `main` asserts that mouse name ↔ DANDI subject id is a 1:1 mapping.
- the `--show-processing` figure re-bins one trial with `np.histogram` (max |diff| = 0)
  and re-derives choice from the raw lick event times (agreement 1.000).

### Check 3 — reference code comparison
| Stage | Reference (`preprocessing_DJ_2022Aug.py`, `align_markers.py`, `population_decoding_utils.py`) | This conversion | Same? |
|---|---|---|---|
| (a) Loading | `loadmat` of DataJoint `.mat` exports, probes concatenated per session | `h5py` on the published NWB; NWB stores all probes of a session in one `units` table, so no concatenation is needed | equivalent |
| (b) Neuron filtering | units on the classifier "good unit" list, with a CCF annotation, hemisphere split at ML 5700 µm | `units/classification == 'good'` (all such units have an annotation); hemisphere not split | equivalent; hemisphere pooling is a deliberate choice (papers' summary statistics are not hemisphere-split, and the format asks for region names such as "ALM") |
| (c) Alignment | spike times relative to `task_cue_time` (go cue); markers aligned to the go cue | spike/video times minus `go_start_times` | same |
| (d) Binning | `sliding_histogram`, `bw = 0.04 s`, `stride = 0.0034 s`, rate = counts/bw | 50 ms width, 50 ms stride, rate = counts/0.05 s | **differs by task mandate** (Decoder Task requires 50 ms bins); the rate convention is identical |
| (e) Window | −3 … +3 s (ephys) / −3 … +1.5 s (markers) re. go cue | −2.5 … +1.5 s | **differs by task mandate** |
| (f) Trial filtering | `get_regular_trial_mask`: no early lick, no auto-water, no free-water, response given, no photostim | auto-/free-water excluded; early-lick, no-response and photostim trials **kept** | **differs by task mandate** — those three are decoder outputs/inputs; excluding them would leave the `no lick`, `ignore` and `early = yes` classes empty |
| Session filtering | data paper: performance > 65 %, ≥ 50 correct per direction | identical | same |
| Trial coverage | implicit (the export only contains recorded trials) | explicit via `units/obs_intervals` | equivalent |
| (g) Inputs | reference uses task variables only as labels | `time_from_tone_onset` from `task_sample_time`/`sample_start_times`; `photostim` from `task_stimulation`/`photostim_onset`+`duration` | same source variables |
| (h) Tongue | `align_markers.py` takes the last frame in each 3.4 ms window; occluded tongue imputed with the session mean | mean of the *visible* frames in each 50 ms bin; occlusion is its own class | **differs by task mandate** — the Decoder Task defines "not visible" as a 4th class, which is incompatible with mean-imputation |

### Check 4 — key statistics comparison
See the table in Step 9. Every per-area unit count of datapaper Fig. 1J is reproduced
**exactly** on the full 174-session dataset (thalamus 12,808; orbital 10,223; striatum
7,664; midbrain 7,495; olfactory 4,137; medulla 2,928; cortical subplate 1,960;
hippocampus 1,944; cerebellum 1,820; pallidum 1,092; hypothalamus 815; pons 347), ALM to
within 0.1 % (8,728 vs 8,717), and the behavioral statistics ("476 trials, range
130–785", "84 % correct, range 65–99 %") are reproduced to 478.6 / 130–785 and
83.7 % / 65.8–98.9 %. The only residual discrepancy is 490 cortical units that this
release of the dandiset does not contain (69,453 vs 69,943 units in total).

### Check 5 — edge cases
| Edge case | Handling |
|---|---|
| Trials whose NWB `[start, stop]` is shorter than the −2.5 … +1.5 s window (3 % at the start, 16 % at the end) | spikes and video frames are sliced in continuous session time, so nothing is lost or zero-padded |
| Camera pauses during the inter-trial interval | affected bins have no frames → class 3 ("not visible"), the only representation available |
| Sessions where the ephys covers only part of the behavioral session (9 sessions) | `units/obs_intervals` mask |
| Last `obs_intervals` trial extending past the end of the recording | trials with zero spikes from every neuron are dropped (1 trial dataset-wide) |
| Session with zero good units (`SC017_20190216_162508_s4`) | excluded |
| Sessions with < 2 usable trials | excluded (`MIN_TRIALS_PER_SESSION`) |
| Early-lick replays giving several `sample_start_times` per trial | the last one before the go cue is used; 5.2 % of trials then have their tone onset before the window start, which simply makes `time_from_tone_onset` start above 0 |
| Trial with no preceding `sample_start` | falls back to the nominal go cue − 1.85 s (never triggered: 0 of 94,990 trials) |
| Bin-edge convention | half-open `[edge_k, edge_{k+1})` everywhere, identical to the reference's `(t >= lo) & (t < hi)`; verified against `np.histogram` |
| Photostim strings `'N/A'` | treated as "no stimulation"; the trials-table count equals the `photostim_start_times` event count in all 174 sessions |
| Session with no visible tongue at all | `discretize_tongue` returns all-3 instead of failing (never triggered) |
| `units/is_good_trials` False | affects 0.14 % of (unit, trial) pairs and 565/69,453 units; not actionable because the format requires a rectangular `(n_neurons, 80)` matrix per trial. Documented, not fixed. |

### Iterations
1. First sample run → all-zero neural matrices → added the `obs_intervals` mask → re-ran
   sample conversion, verification, plots. Re-ran all checks.
2. Second sample run → one residual all-zero trial → added the silent-trial rule →
   re-ran. Re-ran all checks.
3. First full run → crash on a session with zero good units → added the "≥ 1 good unit"
   and "≥ 2 usable trials" session criteria → re-ran the full conversion, verification and
   all sanity checks. All pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.
Trained on 61,958 trials, validated on 15,563 held-out trials, device `cuda`.

### Training Progress
Loss decreasing: **Yes** — 19.83 → 15.28 → 12.51 → … → 0.913 (epoch 100) → 0.657
(epoch 200); test loss 0.675, i.e. essentially no generalisation gap.

### Decoder Results (Full, 150 sessions)
| Output | Chance (1/k) | Majority class | Training balanced acc. | Validation balanced acc. | Val / chance |
|--------|--------------|----------------|------------------------|--------------------------|--------------|
| choice (left/right/no lick) | 0.333 | 0.443 | 0.7208 | **0.6766** | 2.03× |
| outcome (ignore/miss/hit) | 0.333 | 0.728 | 0.7077 | **0.6481** | 1.94× |
| early_lick (no/yes) | 0.500 | 0.885 | 0.7843 | **0.7479** | 1.50× |
| tongue_y (4 classes) | 0.250 | 0.746 | 0.6841 | **0.6597** | 2.64× |

Every output is well above both uniform chance and the majority-class rate (balanced
accuracy of a majority-class predictor is 1/k by construction).

### `predictions.png` review
For the plotted held-out trials the predicted traces are at chance during the first ~50
bins (= up to the go cue) and lock onto the correct class immediately after bin 50
(t = 0). This is the expected signature: `choice`, `outcome` and `early_lick` are
properties of the whole trial but can only be read out of the neural activity once the
information is present in the brain, so pooling all 80 bins necessarily caps the
attainable balanced accuracy.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — accuracy vs chance
| Output | Validation balanced acc. | Chance | Ratio | Verdict |
|---|---|---|---|---|
| choice | 0.6766 | 0.3333 | 2.03× | ✔ |
| outcome | 0.6481 | 0.3333 | 1.94× | ✔ |
| early_lick | 0.7479 | 0.5000 | 1.50× | at the 1.5× line — investigated below |
| tongue_y | 0.6597 | 0.2500 | 2.64× | ✔ |

Nothing is below chance. `early_lick` is intrinsically the hardest of the four, and its
ceiling is structural rather than a conversion defect:
- an "early lick" is a lick anywhere in the sample **or** delay epoch, and an early lick
  triggers a *replay* of that epoch. The neural correlate therefore often falls **before**
  the −2.5 s window boundary (5.2 % of trials have their tone onset before the window
  starts, and those are exactly the replayed, i.e. early-lick, trials). Widening the
  window is not possible — it is fixed by the Decoder Task.
- for a binary variable the ratio to chance is bounded by 2.0, so 1.50× corresponds to
  the same "distance from chance" as 2.0× would for a 4-class variable.

### Check 2 — accuracy comparison to the papers
No paper reports a neural decoder for outcome, early lick or tongue position, so the only
directly comparable quantity is **choice**. The method paper reports choice decoding from
*video* (embedding method, logistic regression, AUC of ROC), per epoch. To compare
like-with-like I reproduced that analysis on the converted **neural** data
(`cache/time_resolved_decoding.py`: PCA-16 + logistic regression, 5-fold stratified CV,
independently at each of the 80 bins; 6 randomly chosen sessions, left vs right trials):

| Epoch | Method paper (video → choice) | This conversion (neural → choice) |
|---|---|---|
| Pre-sample (t < −1.85 s) | AUC 0.51 ± 0.06 s.d. | **0.538 ± 0.035** |
| 2nd half of sample + delay (−1.5 … 0 s) | AUC 0.66 ± 0.12 s.d. | **0.752 ± 0.062** |
| 2nd half of the response epoch (t > 0.75 s) | AUC 0.99 ± 0.01 s.d. | **0.910 ± 0.032** |

Interpretation: the pre-sample value sits at chance, exactly as the paper requires — this
is a strong alignment check, because any leakage of trial identity into the pre-stimulus
period (e.g. an off-by-one trial shift) would push it far above 0.5. The delay-epoch value
is *higher* than the paper's video-based number, as expected, since the recordings include
ALM, whose delay activity is the canonical choice-selective signal of this dataset. The
response-epoch value (0.91) is below the video's 0.99, which is also expected: the video
literally shows the tongue moving left or right, whereas single 50 ms bins of ~400 neurons
are a noisier read-out, and (unlike the paper) we include photostimulation, early-lick and
error trials in this analysis. The saved curve is `cache/choice_auc_vs_time.png`.

### Check 3 — train vs validation gap
| Output | Train | Validation | Train / Val |
|---|---|---|---|
| choice | 0.7208 | 0.6766 | 1.065 |
| outcome | 0.7077 | 0.6481 | 1.092 |
| early_lick | 0.7843 | 0.7479 | 1.049 |
| tongue_y | 0.6841 | 0.6597 | 1.037 |

All far below the 1.5× threshold — no overfitting and no sign of leakage (a leak would
show up as a *high* validation accuracy, and specifically as an above-chance pre-stimulus
AUC in Check 2, which it does not).

### Further debugging steps performed
1. **Output values verified on raw data** — `cache/sanity_checks.py`, 8 sessions × 5
   random trials per output, plus a full trial × bin comparison of the tongue classes.
   0 failures (Step 10, Check 2).
2. **Temporal alignment verified** — three independent checks, all confirming that the
   video, lick-event, photostim and spike clocks agree with the go cue:
   P(tongue visible) rises from 0.074 to 0.541 exactly at t = 0 and tracks the lick rate
   bin for bin; the population PSTH is flat before and rises sharply after t = 0; choice
   AUC is at chance before the tone and rises through the delay epoch.
3. **Output variation checked** — the least frequent classes are `no lick` (12.0 %),
   `ignore` (12.0 %), `early = yes` (11.5 %) and `tongue = 40–60th pctile` (5.1 %); none
   is degenerate, and `train_decoder.py` runs with `balanced_loss=True`.
4. **Neuron filtering re-checked** against the paper's per-area unit counts — exact match
   on 12 of 14 areas (Step 10, Check 4).
5. **Processing re-checked** against the reference code, step by step (Step 10, Check 3).

### Investigated and deliberately not adopted: rescaling the neural array
The provided decoder is a linear projection (SVD-initialised) plus a linear read-out,
trained with a fixed learning rate and L1 penalty, and it never standardises the neural
input — so its accuracy depends on the *units* of the neural array even though the
information content does not. Sweeping a global scale factor on a 12-session subset
(`cache/scale_sweep.py`, `cache/scale_test.py`):

| Neural array | choice | outcome | early_lick | tongue_y |
|---|---|---|---|---|
| firing rate in Hz (**shipped**) | 0.7015 | 0.6623 | 0.7314 | 0.6600 |
| × 0.2 | 0.7191 | 0.6771 | 0.7373 | 0.6833 |
| × 0.05 (= spike count per 50 ms bin) | 0.7255 | 0.6794 | 0.7408 | 0.6932 |
| × 0.01 (arbitrary unit) | 0.7273 | 0.6846 | 0.7439 | 0.6964 |
| per-neuron z-score | 0.7156 | 0.6394 | 0.7227 | 0.6760 |

Smaller units help by 1–3 points, and the gain saturates. **We ship firing rates in Hz**
anyway, because (a) that is exactly what the reference pipeline produces
(`sliding_histogram(..., rate=True)` = counts / bin width) and the task requires matching
the reference's processing, (b) Hz is the standard, interpretable unit and is documented
in `metadata['neural_units']`, and (c) the 1–3 point difference is an artefact of one
decoder's optimiser settings, not a property of the data — rescaling by an arbitrary
constant to suit a particular optimiser would be tuning the dataset to the metric. The
measurement is reported here so that the choice is explicit and reversible (multiply
`neural` by 0.05 to obtain spike counts).

### Issues found and resolved (cumulative)
| Issue | Resolution |
|---|---|
| 321/480 trials of `SC015_…_s2` (and large blocks in 8 other sessions) had all-zero firing rates | mask trials with `units/obs_intervals` |
| one residual all-zero trial (last `obs_intervals` trial past the end of the recording) | drop trials in which no neuron fires at all |
| conversion crashed on `SC017_20190216_162508_s4` (0 good units) | added "≥ 1 good unit" and "≥ 2 usable trials" session criteria |
| photostim cross-check panel assigned events to the previous trial | plot-only fix |
| `assign_brain_regions` contained a redundant, misleading condition for ALM | simplified to `group == 'OtherCortex' and ap_mm > 2.0` |

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — dataset description, curation summary, how to load, full
      format specification, key statistics, decoder performance.
- [x] `cache/` folder created with `README_CACHE.md` documenting every file.
- [x] All analysis / validation / exploration scripts moved to `cache/`.

### Final file inventory
| File | Contents |
|---|---|
| `CONVERSION_NOTES.md` | this document |
| `README.md` | user-facing documentation |
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | 10.21 GB, 150 sessions, 77,521 trials, 59,749 neurons |
| `sample_data.pkl` | 2 sessions, 676 trials |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | `--verify-only` logs (no errors, no warnings) |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_SC015_20190208_133600_s2.png`, `processing_SC015_20190209_150135_s3.png` | `--show-processing` figures |
| `sample_trials.png`, `predictions.png` | produced by `train_decoder.py --plot-samples` |
| `cache/` | Allen CCF ontology, sanity checks, statistics and exploration scripts (see `cache/README_CACHE.md`) |

### Summary of the decisions taken (and why)
1. **Alignment to the go cue**, window −2.5 … +1.5 s, 80 × 50 ms bins — mandated by the
   Decoder Task; the reference aligns to the same event.
2. **Firing rates in Hz** = spike count / 50 ms — the reference's convention
   (`sliding_histogram(rate=True)`).
3. **Neurons**: `units/classification == 'good'`, which reproduces the paper's per-area
   unit counts exactly.
4. **Brain areas**: Allen CCF ontology grouping into the 14 coarse areas of datapaper
   Fig. 1J, with ALM carved out of cortex at AP > 2.0 mm (reproduces 8,728 vs 8,717).
5. **Sessions**: data-paper criteria (> 65 % performance, ≥ 50 correct per direction),
   reproducing "476 trials (130–785), 84 % correct (65–99 %)"; plus ≥ 1 good unit.
6. **Trials**: all except auto-/free-water, unobserved (`obs_intervals`) and spike-free
   trials — early-lick, no-response and photostim trials are kept because they *are* the
   decoder's outputs/inputs.
7. **Inputs**: continuous time-since-tone-onset and a binary photostim indicator, both as
   the Decoder Task specifies.
8. **Outputs**: all four as time-varying class indices; tongue y discretised per session
   at the 40th/60th percentiles of visible bins, with a separate "not visible" class
   (DeepLabCut likelihood ≤ 0.9), superseding the reference's mean-imputation of
   occluded frames as the Decoder Task requires.
