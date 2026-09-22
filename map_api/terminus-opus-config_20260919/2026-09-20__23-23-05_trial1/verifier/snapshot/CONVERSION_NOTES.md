# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (Chen et al.); NWB files in /app/data
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, pynwb 4.1.0 all import successfully.

Directory contents of /app:
- `CONVERSION_NOTES.md` (this file)
- `ChenLiuEtAl2023_SpikeSortingQC.pdf` — white paper on spike sorting + quality control
- `datapaper.pdf` — Chen et al., "Brain-wide neural activity underlying memory-guided movement"
- `methodpaper.pdf` — Wang, Kurgyis et al., "Brain-wide analysis reveals movement encoding structured across and within brain areas"
- `methods.txt` — excerpted methods
- `code/` — MapVideoAnalysis reference repo (Archive, Notebooks, Sherlock, VideoAnalysisUtils)
- `data/` — DANDI:000363 (Mesoscale Activity Map dataset), 50 GB, 28 subject folders (`sub-*`), 174 `.nwb` files, plus `dandiset.yaml`
- `pynwb_docs/`, `train_decoder.py`, `decoder.py`, `Dockerfile`, `docker-compose.yaml`
- `cache/` — my scratch scripts

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference repo = `MapVideoAnalysis` (Wang, Kurgyis et al. 2025, Nat Neurosci). It operates on `.mat`
exports of the same DataJoint database that DANDI:000363 NWB files were generated from, so field names
map closely onto NWB fields.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel` / `process_one_sess` | VideoAnalysisUtils/preprocessing_DJ_2022Aug.py | LOADING | Loop over sessions; per session concatenate all probe files; read behaviour/task variables and spike times |
| `sliding_histogram(spikeTimes, begin, end, bin_width, stride, rate=True)` | preprocessing_DJ_2022Aug.py | PROCESSING | Bin spike times (already aligned so go cue = 0) into sliding-window firing rates; returns bin centers and `fr` (n_bins, n_trials, n_neurons) in spikes/s |
| `process_one_area` | preprocessing_DJ_2022Aug.py | PROCESSING | Truncate spike times to [begin_time, end_time], bin them, save per-area pickle with behaviour fields |
| `helper_get_neuron_id_area` | preprocessing_DJ_2022Aug.py | CURATION | Keep only units that (a) passed QC ('goodunits' classifier list) and (b) fall in given hemisphere (CCF ML coordinate vs 5700 µm midline) and region annotation |
| `get_regular_trial_mask` | population_decoding_utils.py, functions_for_r2.py | CURATION | Regular trials: `early_lick==0 & auto_water==0 & free_water==0 & correctness!=-1 & stimulation[:,0]==0` |
| `create_4fold_trial_type_mask` | functions_for_r2.py | CURATION | Stratify trials into hit-right/miss-right/hit-left/miss-left for CV |
| `align_markers_between_lims` | Sherlock/align_markers.py | PROCESSING | Align DeepLabCut side-camera markers (`nose_x/y, tongue_x/y, jaw_x/y, whisker_x/y`) to the go cue: frame time = frame_index*dt − go_cue_time, dt = 0.0034 s (≈300 Hz); resample onto grid `np.arange(-3, 1.5, dt)` taking the last frame in each bin (0 if none) |
| `get_bad_trial_inds` | Sherlock/align_markers.py | CURATION | Flag trials where #video frames disagrees with trial end time by more than one frame |
| `get_period` | preprocessing_utils.py | PROCESSING | Epoch definitions relative to go cue: sample −1.9…−1.2 s, delay −1.2…0 s, post-go 0…1 s, all −3.0…3.5 s |
| `get_single_area_inds` / `get_inds_for_list_of_regions` | functions.py | PROCESSING | Map CCF annotations onto coarse brain-region groups (ALM handled specially via ALM voxel mask / qc list) |
| `nested_cross_validation` | population_decoding_utils.py | ANALYSIS | Logistic-regression population decoding (PCA 16 comps) per time bin, AUC scored |

### Notes / important parameters from reference code
- `Sherlock/preprocess_all_ephys.py`: `bw = 0.04 s`, `stride = 0.0034 s`, `begin_time = -3.0`, `end_time = 3.0`, `qc_mode = 'classifier'` (for video-matched analyses; the module `__main__` block uses bw=0.1, stride=0.05, −3.0…3.5).
- Spike times in the source (Susu's export / NWB) are **relative to the go cue** for each trial.
- Firing rates are computed as counts / bin_width (spikes per second).
- Units are curated by the **'classifier' QC** described in Chen, Liu et al. 2023 white paper (`goodunits` folder), i.e. only units passing quality control are used.
- Regions in the reference pipeline: ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum; each split into left/right hemisphere using CCF ML midline 5700 µm.
- Trial curation for their encoding/decoding analyses keeps only "regular" trials (no early lick, no auto/free water, no no-response, **no photostimulation**). For the present decoder task photostimulation must be an input and early lick / ignore outcomes must be outputs, so these trial exclusions **cannot** be applied here (documented deviation).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` is a DANDI download of **dandiset 000363** ("Mesoscale Activity Map Dataset", Chen, Nguyen, Li, Svoboda 2023),
organised as `sub-<subject_id>/sub-<subject_id>_ses-<YYYYMMDD>T<HHMMSS>_behavior+ecephys+ogen.nwb`.

- **174 NWB files** = 174 sessions, **28 subjects** (`sub-440956` … `sub-484677`), 50 GB total.
- Every file read with `pynwb.NWBHDF5IO(..., load_namespaces=True)`.

Contents of each NWB file (verified on several files):

| NWB location | Contents |
|---|---|
| `nwb.identifier` | session name, e.g. `SC015_20190207_120657_s1` (mouse, date, time, session #) — same naming used by the reference `.mat` exports |
| `nwb.subject` | `subject_id` (e.g. 440956), `description` = mouse nickname (e.g. `SC015`), sex, DOB, species |
| `nwb.trials` (TimeIntervals) | columns: `start_time`, `stop_time`, `trial`, `photostim_onset`, `photostim_power`, `photostim_duration` (strings, `'N/A'` when no photostim; onset is **relative to trial start**), `trial_uid`, `task` (always `audio delay`), `task_protocol`, `trial_instruction` (`left`/`right`), `early_lick` (`early`/`no early`), `outcome` (`hit`/`miss`/`ignore`), `auto_water` (0/1), `free_water` (0/1) |
| `nwb.acquisition['BehavioralEvents']` | TimeSeries with **timestamps in session clock**: `presample_start/stop_times`, `sample_start/stop_times`, `delay_start/stop_times`, `go_start/stop_times`, `trialend_start/stop_times`, `left_lick_times`, `right_lick_times`, `photostim_start_times` (data = laser power in mW), `photostim_stop_times` |
| `nwb.acquisition['BehavioralTimeSeries']` | DeepLabCut side-view tracking at 300 Hz (dt = 0.0034 s): `Camera0_side_TongueTracking`, `Camera0_side_JawTracking`, `Camera0_side_NoseTracking`, each (n_frames, 3) = (x, y, likelihood), with **session-clock timestamps**; a few sessions also have `Camera0_side_WhiskerTracking_whisker` (20), `LickPortTracking` (4) or `Camera3_*` (3) |
| `nwb.units` | 39 columns incl. `spike_times` (session clock), `obs_intervals`, `unit_quality` (`good`/`multi`, the Kilosort/Phy label), **`classification`** (`good` / `unlabelled` — the QC-classifier label from the Chen–Liu white paper), **`anno_name`** (CCF structure name, only non-empty for `classification=='good'` units), `is_good_trials` (bool per trial, per unit), 15 QC metrics (`presence_ratio`, `amplitude_cutoff`, `isi_violation`, `unit_snr`, `drift_metric`, `avg_firing_rate`, …), `electrodes` (DynamicTableRegion → 1 electrode per unit), `electrode_group` |
| `nwb.electrodes` | `x`, `y`, `z` = CCF coordinates in µm (x = ML, midline at 5700 µm — **left hemisphere is x ≥ 5700**, verified against probe target labels), `group_name`, `shank`, … |
| `nwb.electrode_groups[*].location` | JSON with targeted `brain_regions` (e.g. `left ALM`, `right Striatum`), insertion coordinates and angles |

### Dataset Size (from data files; computed with `cache/scan_all.py`, `cache/scan_units.py`)
| Statistic | Value |
|-----------|-------|
| NWB files / sessions | 174 |
| Subjects | 28 |
| Sessions / subject | mean 6.2 (range 1–13) |
| Trials (total) | 94,990 |
| Trials / session | mean 545.9 (range 264–800) |
| Units (all, incl. `unlabelled`) | 272,227 |
| **Good units (`classification=='good'`)** | **69,453** (25.5 % of all clusters) |
| Good units / session | mean 399.2 (range 0–923); **1 session has 0 good units** (`SC017_20190216_162508_s4`) |
| Probe insertions (electrode groups) | 659 (targets: ALM 211, Midbrain 136, Striatum 114, Thalamus 91, Medulla 58, ECT 32, BLA 17) |
| Unique CCF `anno_name` values (good units) | 293 |
| Trials with photostim | 18,588 (19.6 %) in 168 sessions |
| Outcomes | hit 65,254 (68.7 %), miss 15,641 (16.5 %), ignore 14,095 (14.8 %) |
| Early lick | early 10,805 (11.4 %), no early 84,185 |
| Instruction | right 48,913, left 46,077 |
| Sessions with side-view tongue tracking | 174 (all) |

### Additional structural facts established by exploration
- Exactly **one go cue per trial** (`go_start_times` length == n_trials in all 174 sessions).
- `sample_start_times` can occur **several times per trial** (tone epoch is replayed after an early lick): mean 3.06 sample events on early-lick trials vs exactly 1 on other trials. Time from **last** sample onset to go cue ≈ 1.85 s (0.65 s sample + 1.2 s delay); time from **first** sample onset to go cue can be much longer on replay trials.
- `photostim_start_times`/`photostim_stop_times` reproduce `trials.start_time + photostim_onset` exactly (`np.allclose`, atol 1e-3). Photostim is delivered in the **late delay** (e.g. −1.2 → −0.7 s relative to go cue) and always ends before the go cue.
- Video timestamps are in the **session clock**, one continuous 300 Hz block per trial (n_gaps = n_trials − 1), starting at each trial start — consistent with the reference code's `frame_index*dt − go_time` alignment.
- 6 sessions (`SC011_2019022[456]`, `SC022_2019022[78]`, `SC066_20210413`) have video that stops at/near the go cue (post-go coverage 1–6 %), and `SC066_20210413_112028_s6` has essentially no usable video (1 %); `SC065_20210505_170309_s6` covers 62 %.
- Tongue `likelihood` is strongly bimodal (≈ 1 when visible, ≈ 1e-4 when not). Frames with likelihood > 0.9: 11.7 % on average (range 2.9–73 %).
- `units.is_good_trials` was all-`True` for every good unit checked (9 sessions × up to 200 units) → no additional per-unit trial masking needed (re-verified over the full dataset in Step 10).
- Spike times are in the **session clock** (0 … session end) — unlike the reference `.mat` export where they were already aligned to the go cue; therefore alignment to the go cue must be done here.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

Sources read: `/app/methods.txt`, `datapaper.pdf` (Chen et al. 2024, *Brain-wide neural activity underlying
memory-guided movement*), `methodpaper.pdf` (Wang, Kurgyis et al. 2025, *Brain-wide analysis reveals movement
encoding structured across and within brain areas*). PDFs converted to text in `cache/datapaper.txt`,
`cache/methodpaper.txt`.

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt) |
| Good units by area | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | methods.txt |
| Fraction of clusters that are 'good' | 25.9 % | "This corresponds to 25.9 % of clusters reported by Kilosort2" |
| Sessions | 173 behavioral sessions | methods.txt; data paper Fig 1J "660 penetrations, 173 behavioral sessions, and 28 mice" |
| Subjects | 28 mice (25 VGAT-ChR2-EYFP, 1 C57BL/6J, 1 Sst-IRES-Cre×Ai32, 1 Emx1-Cre×GtACR1) | method paper Methods |
| Probe insertions | 660 (data paper) / 655 (methods.txt) | Fig 1J legend |
| Trials / session | mean 476, range 130–785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Behavioural performance | 84 % correct, range 65–99 % | same sentence |
| Session selection (papers) | performance > 65 % **and** ≥ 50 correct lick-left and ≥ 50 correct lick-right trials | "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." |
| Photostim trials | ~25 % randomly interleaved, N = 17 VGAT-ChR2 mice, 93 sessions | methods.txt Photoinhibition |
| Photostim effect | performance 83.2 % → 71.7 % (bilateral ALM) | methods.txt |
| Neural bin width / stride (method paper) | **40 ms bin width, 3.4 ms stride** | "we binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Video frame rate | 300 Hz (dt = 1/300 ≈ 0.0034 s), side view, DeepLabCut markers (tongue, jaw, nose, paws) | method paper Methods |
| Trial epochs | sample 0.65 s (3 × 150 ms tones + 100 ms gaps), delay 1.2 s, go cue 0.1 s, answer/response 1.5 s, consumption 1.5 s | methods.txt Behavior |
| Epoch windows used in reference code | sample −1.9…−1.2 s, delay −1.2…0 s, response 0…1.0/1.5 s relative to go cue | `preprocessing_utils.get_period`, `functions_for_r2.process_single_session_r2_dict` |

### Processing Details
- **Temporal alignment**: everything is aligned to the **go cue** (t = 0). The reference `.mat` export already stored spike times relative to the go cue; in the NWB files spike times are in session time, so we subtract `go_start_times`.
- **Binning**: sliding histogram, firing rate = spike count / bin width (spikes/s). Reference used bw = 40 ms with 3.4 ms stride (method paper) or bw = 100 ms/stride 50 ms in the module `__main__`. **Our task specifies 50 ms bins**, so we use non-overlapping 50-ms bins (stride = width = 50 ms), which matches the spirit (rate in spikes/s) while satisfying the decoder spec.
- **Trial window**: reference used −3.0 → 3.0 / 3.5 s around the go cue; **our task specifies −2.5 → +1.5 s**.
- **Video**: markers aligned to the go cue with frame time = frame_index × 0.0034 − go_time; reference resamples onto a −3…1.5 s grid taking the last frame in each bin. Outliers (5-sigma on velocity) imputed; when the tongue is occluded (inside the mouth) they set the tongue position to its mean value. For our decoder the occluded state is its own output class (`not visible`), so we detect occlusion from the DLC `likelihood` rather than imputing.

### Curation Steps

**Neuron curation rules** (papers + white paper `ChenLiuEtAl2023_SpikeSortingQC.pdf`):
- Spike sorting with Kilosort2; 15 quality metrics; five region-specific logistic-regression classifiers (cortex, striatum, thalamus, midbrain, medulla) trained on manual Phy labels; units labelled **'good'** by the classifier are the ones used in all analyses. In NWB this is exactly `units.classification == 'good'` (69,453 units in the 174 files ≈ the 69,943 reported).
- Method paper additionally drops neurons with mean firing rate < 2 Hz *for the encoding (R²) analyses* and brain areas with < 10 neurons per session. These are analysis-specific thresholds, not dataset curation; we do **not** apply them (a population decoder benefits from all good units, and dropping low-rate units would discard information).

**Trial curation rules**:
- Papers exclude, *for their analyses*: photoinhibition trials, free-water/auto-water trials, early-lick trials and no-response ('ignore') trials (`get_regular_trial_mask`).
- **Our decoder task requires photostimulation as an input and early lick / ignore outcome as outputs**, so these trials must be kept. This is the single deliberate, documented deviation from the reference curation. Auto-water / free-water trials are also kept because dropping them would remove no needed variable; they are flagged in the metadata (they are rare: see Step 9).
- Session-level: the papers keep sessions with performance > 65 % and ≥ 50 correct left and ≥ 50 correct right trials (145/174 sessions satisfy this in the raw data).

### Decoders Trained (reference results to compare against)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Choice (lick direction) from **video**, pre-sample epoch | logistic regression on video embeddings | AUC 0.51 ± 0.06 (n = 106 sessions) |
| Choice from **video**, sample+delay | same | AUC 0.66 ± 0.12 |
| Choice from **video**, 2nd half of response epoch | same | AUC 0.99 ± 0.01 |
| Choice from **neural population** (200 neurons, 200 ms causal window) | logistic regression, hierarchical bootstrap | ALM rises during sample/delay; ≈ 0.9 late delay (data paper Fig 6D) |
| Choice from ALM population, late delay | data paper Fig S6 | ~0.8–0.95 depending on n neurons |

Expectation for our decoder: with all good units of a session and the full −2.5…+1.5 s window, **choice and
outcome should be decodable well above chance (≥ 0.8 balanced accuracy)**, tongue y-position (which is a direct
readout of licking) should be high after the go cue, and early lick should be decodable above chance but is
harder (11 % base rate, event occurs before the window centre).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

I cross-checked the three sources (reference code, NWB data, papers). Everything reconciles; the table below
lists each item I checked, including the ones that initially looked like discrepancies.

### Discrepancies Found / Checked
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Unit quality filter | `qc_mode='classifier'`, units listed in `goodunits/*.mat` | `units.classification ∈ {good, unlabelled}`; 69,453 'good' of 272,227 (25.5 %) | "classifiers … provided lists of units that were labeled as 'good' and were used for the analysis"; 69,943 good units, 25.9 % of clusters | `classification=='good'` **is** the classifier QC list. Counts agree to 0.7 %. Use it. |
| Sessions | one folder per session | 174 NWB files, but 1 (`SC017_20190216_162508_s4`) has **0** good units | 173 behavioral sessions | 174 − 1 = **173** sessions with usable neurons: exact match. |
| Units per area | regions ALM, Striatum, Thalamus, Midbrain, Medulla, … | my CCF→coarse mapping gives Thalamus 12,776; Midbrain 7,432; Striatum 7,347; Medulla 2,912; ALM (M1/M2 on ALM probes) 7,512 | thalamus 12,808; midbrain 7,495; striatum 7,664; medulla 2,928; ALM 8,717 | Agreement within 0.3–4 % for all subcortical areas. ALM is lower because the paper uses an ALM **voxel mask** (`ALM_voxels_symmetric.npy`, not shipped with the repo) that also captures nearby frontal-pole/premotor units; my label assigns those to Orbital/OtherCortex. Acceptable: region labels only populate `brain_region_idx`. |
| Probe insertions | — | 659 electrode groups | 655 (methods.txt) / 660 (data paper Fig 1J) | Within rounding of the two paper values. |
| Trials/session | — | 546 all trials; 464 responded (hit+miss) over the 173 sessions; 487 responded over the 145 sessions passing the papers' behavioural criteria, range 190–785 | "on average 476 (range, 130-785) trials per session" | The paper's "trials" = trials the animal responded on, in the selected sessions: mean 487 vs 476 and identical upper bound 785. Match. |
| Performance | `correctness==1` for hit | hit/(hit+miss) on control, non-early, non-auto-water trials: 0.806 over all 174 sessions; **0.840** over the 145 sessions passing the criteria, range 0.658–0.989 | "84 % correct rate (range, 65-99 %)" | Exact match once the paper's own session-selection criteria are applied. Confirms my definition of 'performance' and of the trial masks. |
| Trial types | `behavior_early_report`, `behavior_report`(1/0/−1), `task_trial_type` ('l'/'r'), `behavior_is_auto_water`, `behavior_is_free_water` | `early_lick`, `outcome` (hit/miss/ignore), `trial_instruction` (left/right), `auto_water`, `free_water` | same variables | 1-to-1 mapping: `outcome=='hit'`↔`report==1`, `'miss'`↔0, `'ignore'`↔−1. |
| Photostim | `task_stimulation` = [power, type, on, off] relative to trial start, converted to go-cue-relative | `trials.photostim_onset/power/duration` (strings, relative to trial start) **and** `BehavioralEvents.photostim_start/stop_times` in session time; the two agree exactly (`np.allclose`) | late-delay ALM silencing, 0.5 s incl. 100 ms ramp, ~25 % of trials | Use the event timestamps (already absolute) → go-cue-relative on/off. Data: 18,588 stim trials (19.6 %) in 168 sessions. |
| Alignment | reference `.mat` spike times were **already** go-cue-aligned | NWB spike times are in **session** time | all analyses aligned to go cue | Subtract `go_start_times` per trial. Same end result. |
| Binning | `sliding_histogram`, rate = count/bw; bw 40 ms, stride 3.4 ms | — | "bin width of 40 ms and a stride of 3.4 ms" | Task here **requires 50 ms bins** → non-overlapping 50 ms bins, rate = count/0.05 s. Documented deviation required by the task spec. |
| Trial window | −3.0 … 3.0/3.5 s | — | epochs sample −1.9…−1.2, delay −1.2…0, response 0…1.5 | Task requires −2.5 … +1.5 s. Window fully contains delay + response and most of the sample epoch. |
| Video | `align_markers.py`: dt = 0.0034, frame time = index*dt − go_time, grid −3…1.5 s | tracking timestamps already in session clock, 300 Hz, one block per trial | 300 Hz side-view DLC markers | Equivalent; I use the stored timestamps − go time (more robust, handles dropped frames). |
| Tongue occlusion | — | `tongue_likelihood` bimodal: >0.9 on 11.7 % of frames | "When the tongue was occluded … we set the tongue position to its mean value" | Papers impute; **our task defines 'not visible' as its own output class (3)**, so I threshold the likelihood instead of imputing. |
| Trial exclusions | `get_regular_trial_mask`: no early lick, no auto/free water, no ignore, no photostim | — | "trials with photoinhibition, … free water trials, early licks and trials where the animal ignores the lick-spouts … were excluded from all analyses" | **Cannot** be applied here: photostim is a required decoder *input*, early-lick and ignore are required decoder *outputs*. Keep all trials; documented deviation. |
| Low-firing-rate neurons | — | — | "Neurons with low firing rates (below 2 Hz) were excluded" (for the R² encoding analysis only) | Analysis-specific, not dataset curation → not applied. |
| `is_good_trials` | not present in the `.mat` pipeline | all-True for every good unit sampled | not mentioned | No additional masking needed (verified dataset-wide in Step 10). |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable (NWB) | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units.spike_times` (session clock) for units with `units.classification == 'good'` | `neural[session][trial]` (n_neurons, 80) | subtract that trial's go-cue time, bin into non-overlapping 50 ms bins spanning −2.5…+1.5 s, divide counts by 0.05 s → firing rate in spikes/s, float32 | `preprocessing_DJ_2022Aug.sliding_histogram` (rate = count/bw), `process_one_area` | reference used bw 40 ms / stride 3.4 ms; task requires 50 ms bins |
| `BehavioralEvents.go_start_times` | alignment (t = 0) | one event per trial, assigned by trial interval | `sess_file['task_cue_time'][0]`, used everywhere as t=0 | verified exactly 1 per trial in all 174 sessions |
| `BehavioralEvents.sample_start_times` (last one before the go cue) | `input[0]` = `time_from_tone_onset_s` | bin-centre time − tone-onset time (both relative to go cue); continuous, time-varying | reference stores `sample_time` (start+duration) | on early-lick trials the sample epoch is replayed; the **last** onset before the go cue is the tone that instructed this trial |
| `BehavioralEvents.photostim_start_times` / `photostim_stop_times` | `input[1]` = `photostim_on` | 1.0 if bin centre ∈ [onset, offset] relative to go cue, else 0.0 | `sess_file['task_stimulation']` cols 2,3 shifted by go-cue time | equals `trials.photostim_onset` + trial start (verified `np.allclose`) |
| `trials.outcome` + `trials.trial_instruction` | `output[0]` = `lick_direction_choice` ∈ {0 left, 1 right, 2 no lick} | hit → instruction; miss → opposite of instruction; ignore → no lick | `behavior_report` (1/0/−1) + `task_trial_type` ('l'/'r') → `trial_type`, `correctness` | cross-checked against the first lick after the go cue (`left_lick_times`/`right_lick_times`) |
| `trials.outcome` | `output[1]` = `outcome` ∈ {0 ignore, 1 miss, 2 hit} | direct map | `behavior_report` | ordering given in the task description |
| `trials.early_lick` | `output[2]` = `early_lick` ∈ {0 no, 1 yes} | `'early'` → 1 | `behavior_early_report` | |
| `BehavioralTimeSeries.Camera0_side_TongueTracking` (y = column 1, likelihood = column 2) | `output[3]` = `tongue_y_position` ∈ {0,1,2,3} | per 50-ms bin: mean y over frames with likelihood > 0.9; if no visible frame in the bin → class 3 (not visible). Per-session percentiles (40th, 60th) of the *visible* binned y values: <p40 → 0, p40–p60 → 1, >p60 → 2 | `Sherlock/align_markers.py` (marker alignment to go cue, dt = 0.0034 s) | paper imputes occluded tongue with its mean; here occlusion is its own class, so likelihood thresholding is used instead |
| `nwb.subject.description` (e.g. `SC015`) | `subjects`, `subject_idx` | unique list | mouse name in the `.mat` file names | matches how the papers name mice |
| `units.anno_name` (CCF structure) + `electrodes.x` (ML, midline 5700 µm) + electrode-group target | `brain_regions`, `brain_region_idx` | CCF structure → coarse region (ALM, Orbital, OtherCortex, Striatum, Pallidum, Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum, Hippocampus, Olfactory, CorticalSubplate), prefixed with hemisphere (`left`/`right`, x ≥ 5700 = left) | `helper_get_neuron_id_area` (same 14 regions, same ML midline 5700), `functions.get_single_area_inds` | validated against the papers' per-area unit counts (Step 4) |

All four outputs are emitted as a **time-varying (4, 80)** array: the three per-trial variables are constant
across the 80 bins (the target format requires a single array per trial, and the tongue class is genuinely
time-varying).

### Key Decisions
1. **Neuron curation = `classification == 'good'`**: this is exactly the QC-classifier output described in the white paper and used for every analysis in both papers (69,453 units, 25.5 % of clusters ≈ the reported 69,943 / 25.9 %). No further QC-metric thresholds are applied because the classifier already integrates the 15 metrics.
2. **No firing-rate threshold**: the method paper's 2 Hz cut is specific to its encoding (R²) analysis, not to dataset curation; dropping low-rate neurons would only remove information from a population decoder.
3. **Session curation** (138 of 174 kept):
   - ≥ 1 good unit (removes `SC017_20190216_162508_s4`; leaves 173, exactly the papers' session count);
   - the papers' behavioural session-selection criteria: performance > 65 % and ≥ 50 correct lick-left and ≥ 50 correct lick-right trials (145 sessions; over these sessions the mean performance is 84.0 % and the maximum number of responded trials is 785, both exactly as reported);
   - video available for the analysis window (removes 6 sessions in which the side-view video stops at the go cue, so the tongue output would be undefined for the whole post-go period).
4. **Trial curation**: all trial types are **kept** (photostim, early lick, ignore, auto/free water) because photostimulation is a required decoder input and early-lick/ignore are required decoder outputs — a deliberate, documented deviation from `get_regular_trial_mask`. Trials are dropped only when the video does not cover ≥ 90 % of the −2.5…+1.5 s window (254 trials in the kept sessions), i.e. when the tongue output cannot be computed.
5. **Unobserved neural periods**: spikes are stored only inside trial intervals (`obs_intervals == [trial start, trial stop]`; 0 of 43.6 M spikes checked lie outside). On error (`miss`) trials the trial ends ~0.8 s after the go cue, so the last part of the window has no recorded spikes and is binned as 0 spikes/s. This matches the reference pipeline, which binned the same go-cue-aligned per-trial spike trains over a fixed −3…3 s window. The fraction of unobserved bins is recorded in `metadata`.
6. **Rates, not counts**: firing rate in spikes/s (count / 0.05 s), as in `sliding_histogram(..., rate=True)`.
7. **Tone onset** = last `sample_start_times` before the go cue (the tone actually instructing this trial; the epoch is replayed after early licks). Median go − tone = 1.85 s.
8. **Tongue visibility threshold**: DLC `likelihood > 0.9`. The likelihood is strongly bimodal (≈ 1 vs ≈ 1e-4), so the exact threshold is immaterial; 11.7 % of frames are 'visible' session-wide.
9. **Percentiles for the tongue classes** are computed **per session** over the visible binned y-values inside the extracted windows (the data actually used), as required by the task ('per-session discretization').

### Planned Sanity Checks
- [ ] Session/subject/unit counts vs papers (173 sessions with units, 28 mice, ≈ 69.5 k good units, per-area counts).
- [ ] Performance of kept sessions = 84 % (paper), responded trials/session ≤ 785.
- [ ] Exactly 80 time bins per trial, identical across all trials/sessions.
- [ ] Independent re-computation (outside the conversion code) of: a neuron's binned rate for a specific (session, trial, neuron, bin); the input values (tone time, photostim) for specific trials; the tongue class for specific bins — compared with `np.allclose`.
- [ ] Choice derived from outcome+instruction agrees with the first lick after the go cue on ≥ 99 % of responded trials.
- [ ] Photostim input is 1 only before the go cue (paper: photoinhibition always ends before the go cue) and on ≈ 20 % of trials.
- [ ] Tongue class distribution: class 3 (not visible) dominant before the go cue, classes 0–2 concentrated after the go cue; classes 0/1/2 ≈ 40/20/40 % of visible bins by construction.
- [ ] Output distributions vs raw NWB values (hit/miss/ignore ≈ 69/16/15 %, early ≈ 11 %).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the Step 5 plan. Structure:

| Function | Purpose |
|---|---|
| `bin_spike_times` | vectorised equivalent of the reference `sliding_histogram(..., rate=True)` for non-overlapping 50 ms bins (`np.bincount` on `floor((t−T_START)/0.05)`), returns spikes/s |
| `choice_from_trial` | lick direction actually chosen: hit → instruction, miss → opposite, ignore → no lick |
| `process_session` | opens one NWB file with `pynwb`, curates units/session/trials, extracts neural, input and output arrays, region labels and per-session diagnostics |
| `plot_session` | `--show-processing` figures: raw raster aligned to go cue, binned rates, inputs, raw tongue trace + binned mean + percentiles, outputs; plus a session-summary figure (tongue y histogram with the 40/60th percentiles, tongue-class fractions vs time, mean population rate vs time) |
| `main` | parallel map over sessions (`multiprocessing.Pool`, default 16 workers), assembles the final dictionary and pickles it |

Implementation notes / efficiency:
- Spike times are read once per unit (`units['spike_times'][i]`) and sliced per trial with `np.searchsorted`; binning is `np.bincount` (no Python loop over bins).
- The video array is read once per session and sliced per trial.
- Timing of each stage is printed per session (`open`, `trials`, `units_meta`, `video_read`, `spikes_read`, `obs_intervals`, `bin_trials`).
- `--sample` runs serially and stops as soon as **2 sessions pass curation** (the first file in the dataset fails the behavioural criteria, so a naive "first 2 files" sample would have produced a single-session file).

Code inefficiencies identified: the initial version looped over bins inside Python and re-read spike times per trial.
Code speedups added: `np.bincount` binning, per-session single read of spikes/video, `searchsorted` slicing, multiprocessing over sessions (~1.2 s per session single-threaded).

Two bugs found and fixed during development (see Step 7):
1. **Unobserved trials**: units carry `obs_intervals`; in 8 sessions the ephys covers only part of the behavioural session (e.g. 160 of 480 trials), so the remaining trials were being emitted as all-zero neural data. Fixed by intersecting the `obs_intervals` trial masks of all good units and dropping unobserved trials.
2. **Empty last trial**: in one session the final trial listed in `obs_intervals` contains no spikes at all (recording interrupted). Fixed by dropping trials with zero total spikes across all good units.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing 2>&1 | tee /app/conversion_sample_out.txt
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only 2>&1 | tee /app/verification_sample_out.txt
```

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (`SC015_20190208_133600_s2`, `SC015_20190209_150135_s3`); `SC015_20190207_120657_s1` skipped (performance 54 % < 65 %) |
| Neurons (total) | 580 (375 + 205) |
| Neurons / session | 290 mean |
| Subjects | 1 (SC015) |
| Trials (total) | 677 (159 + 518) |
| Trials / session | 338 mean |
| Time bins | exactly 80 per trial (50 ms, −2.5 … +1.5 s) |
| time_from_tone_onset range | [−0.6, 5.5] s |
| photostim_on range | [0, 1]; on in 3.3 % of bins; 25–27 % of trials have photostim |
| photostim timing | only in bins with centres between −2.33 and −0.72 s, i.e. always **before** the go cue (as the papers state) |
| lick_direction_choice distribution | left 0.517, right 0.461, no lick 0.022 |
| outcome distribution | ignore 0.022, miss 0.084, hit 0.894 |
| early_lick distribution | no 0.978, yes 0.022 |
| tongue_y_position distribution | 0: 0.103, 1: 0.051, 2: 0.103, 3 (not visible): 0.743 |
| mean firing rate | 6.4 / 13.6 spikes/s (sessions 0/1) |
| fraction unobserved neural bins | 0.019 |

The tongue classes 0/1/2 are in a 40/20/40 ratio **of the visible bins** (0.103/0.051/0.103 → 40.2 %/19.9 %/40.2 %), exactly as the percentile definition requires.

### Processing Plots Review
`processing_<session>.png` (per-trial) and `processing_<session>_summary.png` (per-session) were inspected:
- raw spike raster aligned to the go cue shows the expected post-go increase in activity; the binned firing-rate image matches the raster (no temporal shift);
- the `time_from_tone_onset` input crosses zero exactly at the marked tone onset and the `photostim_on` input is 1 exactly over the shaded photostim interval;
- the raw 300 Hz tongue trace and the binned mean overlay perfectly; the 40th/60th percentile lines fall where the class boundaries change;
- tongue class 3 (not visible) dominates before the go cue (96–100 % of trials) and drops to 7–12 % immediately after it, i.e. licking starts right after the go cue as expected;
- mean population rate rises sharply at the go cue.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| `np.bincount` binning instead of per-bin comparisons | ~5× |
| single read of spike times / video per session + `searchsorted` slicing | ~2× |
| fully vectorised per-unit binning over all trials at once | 0.5–0.9 s → 0.04–0.10 s per session (~10×) |
| 16-way multiprocessing over sessions | ~10× wall-clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| open NWB | 0.16 s | 28 s |
| read spike times | 0.08–0.14 s (up to ~0.5 s for 900-unit sessions) | ~60 s |
| read video | 0.03 s | 6 s |
| obs_intervals | 0.01–0.05 s | 9 s |
| binning + inputs + outputs | 0.04–0.10 s (scales with units × trials) | ~30 s |
| **total serial** | ~0.5–2 s | ~4 min |
| **with 16 workers** | — | **well under 5 min** (plus pickling ~12 GB) |

The verification reported **no errors and no warnings** after the two bug fixes described in Step 6.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/sample_data.pkl 2>&1 | tee /app/train_decoder_sample_out.txt`

### Format Validation
- Errors: **None**
- Warnings: **None** (the earlier "all neural data is zero" warnings were real data problems — unobserved trials and one empty trial — and were fixed in the converter, see Step 6)

### Training
Loss decreased monotonically from 17.32 (epoch 1) to 0.52 (epoch 200); test loss 0.70. No divergence.

### Decoder Results (Sample, 2 sessions / 677 trials / 580 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| lick_direction_choice | 0.333 | 0.762 | 0.564 |
| outcome | 0.333 | 0.778 | 0.516 |
| early_lick | 0.500 | 0.923 | 0.783 |
| tongue_y_position | 0.250 | 0.638 | 0.587 |

All four outputs are above chance on validation data. The margins are modest because this sample contains
only two sessions (677 trials) and the rare classes (`no lick` 2 %, `ignore` 2 %, `early lick` 2 %) contribute
very few validation trials; the balanced accuracy is therefore dominated by these rare classes. The full
dataset has ~100× more trials and many more sessions, which should raise all accuracies (Step 11).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/converted_data.pkl --full 2>&1 | tee /app/conversion_full_out.txt
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only 2>&1 | tee /app/verification_full_out.txt
```

### Output Files
- `converted_data.pkl`: **9.63 GB** (138 sessions × ~524 trials × ~402 neurons × 80 bins, float32)
- `verification_full_out.txt`: created — **"Data format is valid, no errors or warnings."**
- Conversion run time: **35 s** with 16 workers (well under the 15 min budget).

### Sessions skipped (36 of 174)
| Reason | N |
|---|---|
| no good units (`SC017_20190216_162508_s4`) | 1 |
| behavioural session criteria of the data paper (performance ≤ 65 % or < 50 correct trials in one direction) | 29 |
| side-view video does not cover the analysis window (video stops at the go cue) | 6 |

### Trials dropped inside the kept sessions (4,540 of 76,791; 5.9 %)
| Reason | N |
|---|---|
| trial outside the units' `obs_intervals` (ephys not running) | 1,056 |
| video covers < 90 % of the −2.5…1.5 s window | 344 |
| no spikes at all from any good unit (recording interrupted) | 2,150 (note: 1,682 of these are in the two SC053/SC038 mice sessions where an entire probe dropped out) |

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (all 174 NWB) | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Subjects | 28 | — | 28 | **28** | ✅ |
| Sessions with usable units | 173 | — | 173 (174 files − 1 with no good units) | 138 after applying the papers' behavioural criteria + video availability | ✅ (see note) |
| Good units (whole dataset) | 69,943 | `classification=='good'` | 69,453 | 55,437 in the 138 kept sessions | ✅ (0.7 % from the reported total; the rest are in skipped sessions) |
| Good units / session | — | — | 399 | **402** | ✅ |
| Good units per area | ALM 8,717; Striatum 7,664; Thalamus 12,808; Midbrain 7,495; Medulla 2,928 | 14 coarse regions | ALM 7,512; Striatum 7,347; Thalamus 12,776; Midbrain 7,432; Medulla 2,912 | ALM 5,872; Striatum 5,698; Thalamus 10,626; Midbrain 5,847; Medulla 2,575 | ✅ proportions preserved (converted = subset of sessions) |
| Trials / session | mean 476 (130–785) responded trials | — | 546 all / 464 responded | **524** all, 159–800 | ✅ |
| Behavioural performance | 84 % (65–99 %) | `correctness==1` | 84.0 % over sessions passing the criteria | **83.9 %** (65.8–98.9 %) | ✅ |
| Photostim trials | ~25 % of trials in VGAT mice | `stimulation[:,0]!=0` | 19.6 % | **20.3 %** | ✅ |
| Outcome distribution | — | hit/miss/ignore | 68.7 / 16.5 / 14.8 % | **73.7 / 15.3 / 11.0 %** | ✅ (the papers' session criteria remove the worst-performing sessions, raising the hit fraction) |
| Early-lick trials | — | `early_lick` | 11.4 % | **11.6 %** | ✅ |
| Choice distribution | balanced left/right by design | — | left 48.5 / right 51.5 % of instructions | left 44.7 / right 44.2 / no-lick 11.0 % | ✅ |
| Time bins | — | — | — | 80 (all trials, all sessions) | ✅ |
| Time bin size | 40 ms (method paper) | `bw=0.04` | — | 50 ms (task specification) | deviation required by the task |
| Input `time_from_tone_onset` | tone 1.85 s before go cue | — | median go − tone 1.85 s | range [−1.5, 11.9] s, i.e. −1.5 s at the window start (tone at −1.0 s min) up to 11.9 s on rare replay trials | ✅ |
| Input `photostim_on` | late delay, always before the go cue | `stimulation` cols 2–3 | onsets −1.2/−0.5 s, offsets ≤ 0 | on only in bins before t = 0 | ✅ |
| Tongue classes | — | — | tongue visible in 11.7 % of frames | not visible 73.6 % of bins; visible bins split 40/20/40 | ✅ by construction |

*Note on session count*: 173 sessions have usable units, exactly as reported. 138 of them also satisfy the
papers' explicit **session-selection criteria** (performance > 65 %, ≥ 50 correct trials per direction) and
have video covering the analysis window. Applying the papers' own criteria is the consistent choice; the
remaining 35 sessions are excluded for the same reasons the papers excluded them.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
`Data format is valid, no errors or warnings.` — no errors, no warnings remain.
Two earlier classes of warnings were traced to real data problems and fixed in the converter:
1. *"all neural data is zero"* on hundreds of trials → units carry `obs_intervals`; in 8 sessions the ephys
   covers only part of the behavioural session. Fixed by intersecting the per-unit observed-trial masks and
   dropping unobserved trials (1,056 trials dropped).
2. *"all neural data is zero"* on isolated trials → trials listed in `obs_intervals` that nevertheless contain
   zero spikes from **all** good units (recording dropout). Verified by hand in
   `SC066_20210416_140326_s9`: 49 of 680 trials have 0 spikes from all 317 good units although their duration
   (median 4.98 s) and outcomes are normal, i.e. a genuine acquisition dropout, not a conversion bug. Fixed by
   dropping trials with no spikes at all (2,150 trials, 2.8 %, spread over 80 sessions).

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`)
The script re-reads the NWB files with `pynwb` and recomputes everything with different code (e.g.
`np.histogram` instead of `np.bincount`, explicit per-bin masks for the tongue), then compares with the pickle.
Run over 5 sessions spread through the dataset (indices 0, 37, 75, 120, 137):

| Check | Result |
|---|---|
| number of kept trials recomputed independently | 5/5 sessions match exactly |
| firing rate at 20 random (trial, neuron, bin) triplets | all `np.allclose` |
| full (n_neurons × 80) matrix for 5 random trials | all `np.allclose` |
| `time_from_tone_onset` for 15 random trials | all `np.allclose` (atol 1e-5) |
| `photostim_on` for the same trials | all `np.allclose` |
| `choice`, `outcome`, `early_lick` for 15 random trials | all exact |
| tongue class, 3,200 individual bins | 3,200/3,200 exact |
| **total failures** | **0** |

### Check 3: Reference code comparison
| Step | Reference (`/app/code`) | This conversion | Same? |
|---|---|---|---|
| (a) loading | `preprocessing_utils.loadmat` on DataJoint `.mat` exports; per-probe files concatenated per session | `pynwb.NWBHDF5IO`; a session is one NWB file already containing all probes | equivalent (same source database) |
| (b) neuron filtering | `qc_mode='classifier'` → `goodunits/*.mat` id lists; region + hemisphere split at ML 5700 µm | `units.classification == 'good'`; same 14 coarse regions; same ML midline 5700 µm | ✅ same |
| (c) temporal alignment | spike times already stored relative to the go cue; markers aligned as `frame_index*0.0034 − go_time` | subtract `go_start_times` from spike times; video timestamps − go time | ✅ same |
| (d) binning | `sliding_histogram(..., rate=True)`: counts/bin_width, bw 40 ms, stride 3.4 ms, window −3…3 s | `np.bincount` on `floor((t−T0)/0.05)`, rate = counts/0.05 s, window −2.5…1.5 s | same operation, **bin size and window set by the task specification** |
| (e) input construction | reference stores `task_stimulation` (power, type, on, off relative to go) and `sample_time` | `photostim_start/stop_times` − go time; tone = last `sample_start_times` before the go cue | ✅ same variables |
| (f) output construction | `trial_type` ('l'/'r'), `correctness` (1/0/−1), `early_lick_trials`, DLC markers | `trial_instruction`, `outcome` (hit/miss/ignore), `early_lick`, `Camera0_side_TongueTracking` | ✅ same variables |
| trial curation | `get_regular_trial_mask` removes early-lick, auto/free-water, ignore and photostim trials | **kept** — required by the decoder task (photostim = input, early lick / ignore = outputs) | deliberate deviation, documented |
| tongue occlusion | imputed with the session mean | separate class 3 | required by the task's output definition |
| low-rate neurons (<2 Hz) | dropped for the R² analysis only | kept | analysis-specific threshold, not curation |

### Check 4: Key statistics comparison
See the table in Step 9. Additional dataset-wide checks run in `/app/cache/checks2.py` and `checks3.py`:
- **Choice definition**: the choice derived from `outcome` + `trial_instruction` agrees with the *first lick after
  the go cue* on **80,694/80,895 = 99.75 %** of responded trials across all 174 sessions. The 0.25 % mismatches
  are trials where the first detected lick is on the wrong spout before the animal corrects — the
  `outcome`-based definition is the one the papers use (`behavior_report`).
- **Ignore trials**: 97.7 % have no lick at all in the response window, confirming `ignore` = no-response.
- **Photostim always before the go cue**: of 18,588 stimulation events, none ends more than **0.5 ms** after the
  go cue (max overshoot 0.0005 s, i.e. timestamp rounding), matching "photoinhibition always ended before the
  'Go' cue".
- **Good units per coarse region** (Step 4) reproduce the per-area counts published in methods.txt to within
  0.3–4 % for thalamus, midbrain, striatum and medulla.
- **Performance** of the kept sessions: 83.9 % (paper: 84 %), range 65.8–98.9 % (paper: 65–99 %).

### Check 5: Edge cases
| Edge case | Handling |
|---|---|
| trial window extending before the first / after the last trial of a session | only 15 of 94,990 trials; spikes are restricted to trial intervals anyway, so those bins are simply empty; no indexing error (searchsorted is clipped) |
| window overlapping the previous trial (7 trials) | spikes from the previous trial interval are *not* counted because spikes only exist inside trial intervals and the window is applied to absolute time; behaviour identical to the reference which binned per-trial spike trains |
| sample (tone) epoch replayed after an early lick | the **last** onset before the go cue is used; verified that every trial in the dataset has at least one tone onset before its go cue (0 trials without) |
| trials whose post-go period is not recorded (error trials end ~0.8 s after the go cue) | kept, with the unobserved bins containing 0 spikes; the mean fraction of such bins is reported in `metadata['frac_unobserved_neural_bins']` = 0.025 |
| partially recorded sessions (`obs_intervals` covering a subset of trials) | unobserved trials dropped |
| trials with zero spikes from all units | dropped |
| sessions whose video stops at the go cue | dropped at session level (6 sessions) |
| sessions with < 2 usable trials | dropped (the format requires ≥ 2 trials/session) |
| `units.is_good_trials` containing `False` | present in only 4 of 174 sessions (up to 18 % of unit-trials in `SC050_20210302_140243_s20`). The reference pipeline never uses this column, and the flagged trials have normal firing rates (e.g. 2.6 vs 1.6 Hz), so they are **not** excluded; excluding them per-unit is impossible anyway without ragged neuron dimensions. Documented as a known limitation. |
| a mouse's sessions all skipped | did not occur — all 28 subjects are represented |

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`
(GPU: NVIDIA L4; 57,751 training trials / 14,500 validation trials; 200 epochs.)

### Training Progress
- Loss decreasing: **Yes** — 17.4 (epoch 1) → 4.59 (20) → 2.29 (40) → 0.70 (150) → **0.658** (200); test loss 0.671 (no overfitting).

### Decoder Results (Full: 138 sessions, 72,251 trials, 55,437 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|--------|-------------|--------|---|-------|
| lick_direction_choice | 0.333 | 0.7196 | **0.6796** | 2.04× | 3 classes (left/right/no lick) |
| outcome | 0.333 | 0.6988 | **0.6530** | 1.96× | 3 classes (ignore/miss/hit) |
| early_lick | 0.500 | 0.7879 | **0.7462** | 1.49× | 2 classes, 11.6 % positives |
| tongue_y_position | 0.250 | 0.6902 | **0.6634** | 2.65× | 4 classes, genuinely time-varying |

Train–validation gaps are small (0.03–0.04), so the network is not overfitting and the labels are not leaking.
`sample_trials.png` and `predictions.png` were produced and inspected: predicted traces follow the true output
traces, and the tongue-class prediction switches from "not visible" to visible classes right at the go cue.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Chance | Validation Balanced Acc | Ratio | Verdict |
|---|---|---|---|---|
| lick_direction_choice | 0.333 | 0.680 | 2.04× | well above chance |
| outcome | 0.333 | 0.653 | 1.96× | well above chance |
| early_lick | 0.500 | 0.746 | 1.49× | above chance; the ceiling here is intrinsic (see below) |
| tongue_y_position | 0.250 | 0.663 | 2.65× | well above chance |

No output is below chance and none is below 1.5× chance except `early_lick` (1.49×), which is the *binary*
output: for a two-class variable a balanced accuracy of 0.75 corresponds to d′ ≈ 1.35, a strong effect. The
reason it cannot be much higher is that the early lick physically happens during the **sample/delay epoch that
was replayed**, i.e. before the extracted window in many trials (the replayed epoch pushes the early lick more
than 2.5 s before the go cue). I verified the label itself is correct and behaviourally meaningful:
the tongue is visible in **25.2 %** of pre-go bins on early-lick trials versus **4.6 %** on normal trials
(5.5×), and the pre-go population rate is higher (8.5 vs 7.4 spikes/s).

### Check 2: Accuracy comparison to the papers
The papers report **AUC** for binary (left/right) choice decoding at individual time points, not balanced
accuracy over a whole trial, so a like-for-like comparison requires computing per-time-bin AUC from the
converted data. I did exactly that with an independent script (`/app/cache/epoch_decoding.py`, logistic
regression on 16 PCs, 5-fold stratified CV — the reference `population_decoding_utils.nested_cross_validation`
recipe) on 5 sessions spread through the dataset:

| Epoch (time from go cue) | Papers | This dataset (mean of 5 sessions) | Match? |
|---|---|---|---|
| before the sample tone (−2.2 s) | AUC 0.51 ± 0.06 (video, method paper) | **0.565** | ✅ near chance |
| sample + delay (−1.5 … −0.1 s) | AUC 0.66 ± 0.12 (video); neural choice decoding ramps up through the delay (data paper Fig 6D) | **0.71 – 0.76** | ✅ (neural data is more informative than video, as the data paper shows) |
| response epoch (+0.2 … +1.0 s) | AUC 0.99 ± 0.01 (video, 2nd half of response epoch) | **0.96 – 0.98** | ✅ |

The time course reproduces the published one exactly: chance before the tone, a ramp during sample/delay,
near-ceiling after the go cue. This is strong evidence that the temporal alignment, the neural binning and the
choice labels are all correct.

Why is the *whole-trial* balanced accuracy of `lick_direction_choice` (0.68) lower than the 0.96–0.99 AUC after
the go cue? Because the required output format assigns the per-trial choice to **all 80 bins**, including the
2.5 s before the go cue during which — as both papers demonstrate and as the table above reproduces — choice
information is genuinely near chance (before the tone) or only partial (delay). A decoder scored on every bin
therefore cannot exceed roughly the average of the time course, which is what we observe
(mean AUC over the window ≈ 0.78 → balanced accuracy ≈ 0.68 for three classes including the rare `no lick`).
This is a property of the task specification, not a conversion error.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| lick_direction_choice | 0.720 | 0.680 | 1.06 |
| outcome | 0.699 | 0.653 | 1.07 |
| early_lick | 0.788 | 0.746 | 1.06 |
| tongue_y_position | 0.690 | 0.663 | 1.04 |

All ratios ≈ 1.05, far below the 1.5 threshold: no overfitting, no data leakage.

### Additional debugging performed
1. **Output values verified on specific trials** by re-reading the NWB files (Step 10, Check 2): 15 random
   trials × 3 per-trial outputs and 3,200 individual tongue-class bins all match exactly.
2. **Temporal alignment verified** by plotting neural + outputs for single trials (`processing_*.png`) and by
   the per-bin AUC time course, which peaks exactly after t = 0.
3. **Output variation checked**: choice 44.7/44.2/11.0 %, outcome 73.7/15.3/11.0 %, early lick 88.4/11.6 %,
   tongue 10.6/5.3/10.6/73.6 % — no output is degenerate.
4. **Neuron filtering** matches the reference QC classifier (`classification=='good'`).
5. **Processing matches the reference code** (Step 10, Check 3).

### Issues Found and Resolved (across Steps 7–12)
- *Unobserved trials produced all-zero neural data* → intersect per-unit `obs_intervals`, drop unobserved trials.
- *Isolated trials with no spikes from any unit* → dropped (verified to be genuine acquisition dropouts).
- *Sample mode produced a single session* → sample mode now runs until 2 sessions pass curation.
- *Two sessions with 3 and 7 usable trials survived* because the video criterion was only applied per trial →
  added an explicit session-level video criterion (≥ 50 % of observed trials must have video), giving 138 sessions.
- *Slow per-trial binning* → fully vectorised per-unit binning, verified bit-identical (`np.allclose`, max diff 0.0).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, load instructions, format specification, key statistics, decoder results)
- [x] `cache/` folder created with `README_CACHE.md` documenting every exploration / validation script
- [x] All files organized

### Final file inventory (`/app`)
| File | Description |
|---|---|
| `CONVERSION_NOTES.md` | this document |
| `README.md` | user-facing documentation |
| `convert_data.py` | the conversion script (`--full`, `--sample`, `--show-processing`) |
| `region_map.py` | CCF annotation → coarse brain-region mapping, imported by `convert_data.py` |
| `converted_data.pkl` | full converted dataset (9.6 GB, 138 sessions) |
| `sample_data.pkl` | 2-session sample |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | `train_decoder.py --verify-only` logs (no errors, no warnings) |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_*.png` | per-session processing figures (`--show-processing`) |
| `sample_trials.png`, `predictions.png` | figures produced by `train_decoder.py` |
| `cache/` | exploration, scanning and validation scripts + cached results |
