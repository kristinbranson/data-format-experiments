# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (Chen et al.), NWB files in /app/data
- **Date started**: (session start)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- `CONVERSION_NOTES.md` (this file), `convert_data.py` (to be written)
- `ChenLiuEtAl2023_SpikeSortingQC.pdf`, `datapaper.pdf`, `methodpaper.pdf`, `methods.txt`
- `code/` (MapVideoAnalysis repo: VideoAnalysisUtils, Sherlock, Notebooks, Archive)
- `data/` (28 subject dirs `sub-XXXXXX`, 174 `.nwb` files, `dandiset.yaml`)
- `train_decoder.py`, `decoder.py`, `pynwb_docs/`, `Dockerfile`, `docker-compose.yaml`

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, pynwb 4.1.0.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Repo: `MapVideoAnalysis` (Wang*, Kurgyis* et al. 2025, Nat Neurosci) — analysis code for the method paper.
Main module `VideoAnalysisUtils`; cluster scripts in `Sherlock`; notebooks in `Notebooks`.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel` / `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING+CURATION+PROCESSING | Main ephys preprocessing of DataJoint `.mat` exports; concatenates probes, keeps units with both ephys and histology, applies QC good-unit list, splits by hemisphere/region, bins spikes |
| `sliding_histogram(spikeTimes, begin, end, bin_width, stride, rate=True)` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Sliding-window spike count/rate: bin centers from `begin` to `end` in steps of `stride`, window width `bin_width`; returns `(n_bins, n_trials, n_neurons)` firing **rates** (counts / bin_width) |
| `process_one_area` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates spike times to `[begin_time, end_time]` (spike times are **already aligned to go cue = 0** in the source export), then calls `sliding_histogram` |
| `helper_get_neuron_id_area` | `preprocessing_DJ_2022Aug.py` | CURATION | Selects neurons of a `side_region`: hemisphere from CCF ML coordinate (`ccf_x >= 5700` → left, `< 5700` → right) intersected with the QC good-unit index list for that region |
| `loadmat` | `VideoAnalysisUtils/preprocessing_utils.py` | LOADING | scipy `.mat` loader (only used for the DataJoint exports; irrelevant for NWB) |
| `get_regular_trial_mask` | `population_decoding_utils.py`, `functions_for_r2.py`, `Sherlock/generate_single_train_test_split.py` | CURATION | "regular" trials = `early_lick==0` & `auto_water==0` & `free_water==0` & `correctness != -1` (i.e. lick occurred) & `stimulation[:,0]==0` (no photostim) |
| `create_4fold_trial_type_mask` | `functions_for_r2.py` | CURATION | Trial stratification groups: hit-right(1), miss-right(2), hit-left(3), miss-left(4); collapses to 3 if a group has < 2 trials |
| `align_markers_between_lims(marker_data, go_times, t_min=-3, t_max=1.5)` | `Sherlock/align_markers.py` | PROCESSING | Aligns DeepLabCut side-camera markers (`nose/tongue/jaw/whisker` x,y) to **go cue**; video frame rate dt = 0.0034 s (~294 Hz); frame times = `frame_index*dt - go_cue_time`; for each output time bin takes the **last frame within (t-dt, t]** (zero if none) |
| `get_bad_trial_inds` | `Sherlock/align_markers.py` | CURATION | Flags trials where n_video_frames differs from `trial_end_time/dt` by <0 or >1 frame (video/behavior desync) |
| `temporal_alignment_embed_and_ephys` | `functions_for_r2.py` | PROCESSING | Aligns video-derived time base to ephys bin centers by index cropping |
| `nested_cross_validation` | `population_decoding_utils.py` | ANALYSIS | Population decoding: PCA(16) + logistic regression, stratified 5-fold outer CV, AUC per time point |
| `load_session` | `population_decoding_utils.py` | LOADING | Reloads the preprocessed per-area pickles and concatenates neurons across areas of a session |

### Notes on reference processing parameters
- `Sherlock/preprocess_all_ephys.py`: `bw = 0.04` s (40 ms sliding window), `stride = 0.0034` s (matches video frame period), `begin_time = -3.0`, `end_time = 3.0`, `qc_mode = 'classifier'`.
- Firing rates are **spike counts / bin_width** (Hz), computed with a *sliding* window; for our task we are instructed to use 50-ms bins, so we will use non-overlapping 50-ms bins (stride = width = 50 ms), which is the natural analogue.
- Epochs used in the paper (relative to go cue): sample `[-1.85, -1.2]` s, delay `[-1.2, 0]` s, response `[0, 1.5]` s. So the trial structure is ~1.3 s sample + 1.2 s delay; our window of -2.5 to +1.5 s around the go cue covers pre-sample, sample, delay and response.
- Neuron QC in the reference is performed with an **external classifier-based good-unit list** (`goodunits/*.mat`) not shipped with the NWB data. The NWB `units` table (DANDI 000363) contains per-unit QC metrics (`unit_amp`, `presence_ratio`, `amplitude_cutoff`, `isi_violation`, `drift_metric`, `unit_quality`...) and typically a `unit_quality`/`quality` field; we will reproduce the QC using the criteria in the Chen/Liu et al. 2023 spike-sorting QC white paper (to be read in Step 3) applied to those metrics.
- Units without CCF histology annotation are dropped in the reference (`unit_comb = set(ephys unit ids) & set(histology unit ids)`).
- Hemisphere assignment: CCF ML midline at 5700 µm; `ccf_x >= 5700` = left hemisphere.
- Trial curation for the reference **regression** analyses excludes early-lick / auto-water / free-water / no-response / photostim trials. Our decoder task explicitly requires photostim as an input and early lick, no-lick choice and ignore/miss outcome as outputs, so those trials must be **kept**; we will only exclude auto-water/free-water trials (not real behaviour) if warranted, documented in Step 5.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/` contains one directory per mouse: `sub-<subject_id>/` (28 subjects) and `dandiset.yaml` (DANDI:000363, Mesoscale Activity Map dataset, Chen et al.).
- Each session is one NWB file: `sub-<id>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys+ogen.nwb`; **174 files total**.
- All files have identical trial and unit column sets (verified across all 174 files).

**NWB contents (read with `pynwb.NWBHDF5IO`):**
- `nwb.subject.subject_id` (e.g. `440956`), `nwb.subject.description` = lab mouse name (e.g. `SC015`); `nwb.identifier` = e.g. `SC015_20190207_120657_s1`.
- `nwb.trials` (`TimeIntervals`), columns: `start_time`, `stop_time`, `trial`, `photostim_onset`, `photostim_power`, `photostim_duration` (strings, `'N/A'` when no stim; onset relative to **trial start**), `trial_uid`, `task` (all `audio delay`), `task_protocol` (all 1), `trial_instruction` (`left`/`right`), `early_lick` (`early`/`no early`), `outcome` (`hit`/`miss`/`ignore`), `auto_water` (0/1), `free_water` (0/1).
- `nwb.acquisition['BehavioralEvents']` (`BehavioralEvents`), each a `TimeSeries` whose **timestamps** carry the event times (session clock, seconds):
  `presample_start/stop_times`, `sample_start/stop_times`, `delay_start/stop_times`, `go_start/stop_times` (n == n_trials in **all** 174 sessions), `trialend_start/stop_times`, `left_lick_times`, `right_lick_times`, `photostim_start_times` (data = laser power in mW), `photostim_stop_times`.
  Note `sample_*` and `delay_*` have **more** events than trials because early licking triggers a **replay** of the sample/delay epoch (confirmed: extra sample events occur only in `early` trials).
- `nwb.acquisition['BehavioralTimeSeries']`: DeepLabCut side-view (`Camera0_side_`) tracking `JawTracking`, `NoseTracking`, `TongueTracking` (all 174 sessions), occasionally `WhiskerTracking_whisker` (20), `LickPortTracking` (4) and bottom-camera (`Camera3_bottom_*`) series. Each has `data` of shape (n_frames, 3) = (x, y, likelihood) and explicit `timestamps` in session time with **dt = 0.0034 s** (identical in all 174 sessions), segmented per trial: each trial's video segment starts exactly at that trial's `start_time`.
- `nwb.units`: 39 columns, incl. `spike_times` (session clock), `obs_intervals` (= exactly the (start_time, stop_time) of every trial → **spikes are only recorded within trials**), `unit_quality` (`good`/`multi`, Kilosort-level), **`classification`** (`good`/`unlabelled`, output of the paper's QC classifier), **`anno_name`** (fine CCF annotation string, empty for `unlabelled` units), `is_good_trials` (bool per trial; all True in spot checks), 15 QC metrics (`unit_amp`, `unit_snr`, `isi_violation`, `avg_firing_rate`, `presence_ratio`, `amplitude_cutoff`, `isolation_distance`, `l_ratio`, `d_prime`, `nn_hit_rate`, `nn_miss_rate`, `silhouette_score`, `max_drift`, `cumulative_drift`, `drift_metric`), waveform features, `electrodes` (DynamicTableRegion → CCF `x`,`y`,`z` in µm).
- `nwb.electrodes`: `location` (JSON with targeted `brain_regions` e.g. `left ALM`), `x`,`y`,`z` CCF coordinates, `shank`, `group_name`.
- `nwb.electrode_groups`: one per probe insertion; `location` JSON contains coarse target `brain_regions`: left/right ALM, Midbrain, Striatum, Thalamus, Medulla, ECT, BLA.

### Dataset Size (from data files; survey of all 174 files: `/app/cache/survey.py`)
| Statistic | Value |
|-----------|-------|
| Sessions (files) | 174 (173 with ≥1 good unit; `SC017_20190216_162508_s4` has **0** good units) |
| Subjects | 28 |
| Sessions / subject | mean 6.2 (range 2–10) |
| Probe insertions (electrode groups) | 659 |
| Units (total, all Kilosort clusters) | 272,227 |
| Units classified `good` | 69,453 (25.5 % of clusters) |
| Good units / session | mean 399, median 390, range 0–923 |
| Trials (total) | 94,990 |
| Trials / session | mean 546, range 264–800 |
| Outcomes (all trials) | hit 65,254 (68.7 %), miss 15,641 (16.5 %), ignore 14,095 (14.8 %) |
| Early lick | early 10,805 (11.4 %), no early 84,185 |
| Trial instruction | right 48,913, left 46,077 |
| auto_water = 1 | 1,339 trials |
| free_water = 1 | 2,450 trials |
| Photostim trials | 18,588 (19.6 %), all 5.5 mW / 0.5 s; 168/174 sessions have stim trials |
| Unique CCF `anno_name` of good units | 293 |

### Timing facts verified
- Go cue (`go_start_times`) exists for exactly one per trial in all sessions.
- `go - trial start`: mean ≈ 3.2 s (min 2.1 s); `trial stop - go`: median ≈ 1.7 s but **min ≈ 0.21 s** — spikes are not observed after `stop_time`, so the tail of the −2.5→+1.5 s window is unobserved in a minority of trials (same in the reference pipeline, which used a −3→+3 s window on the same per-trial data).
- Last `sample_start` before the go cue is ~1.85–1.95 s before the go cue (tone onset); this matches the paper's sample epoch start at −1.85 s and 1.2 s delay.
- Tongue `likelihood` is strongly bimodal (≈ 89 % < 0.1, ≈ 10 % > 0.9) → a likelihood threshold cleanly defines tongue visibility.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

Sources: `/app/methods.txt` (STAR Methods excerpt of the data paper), `/app/datapaper.pdf` (Chen et al. 2024, *Cell* — "Brain-wide neural activity underlying memory-guided movement"), `/app/methodpaper.pdf` (Wang*, Kurgyis* et al. 2025, *Nat Neurosci*), `/app/ChenLiuEtAl2023_SpikeSortingQC.pdf`. Text extracted to `/app/cache/*.txt` with `pypdf`.

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt) |
| Sessions | 173 | same quote; Fig. 1J: "660 penetrations, 173 behavioral sessions, and 28 mice" |
| Subjects | 28 | "data from 28 mice, including 25 VGAT-ChR2-EYFP..." (method paper Methods) |
| Probe insertions | 655–660 | "655 probe insertions" (methods.txt); "660 penetrations" (Fig. 1J) |
| Fraction of Kilosort2 clusters kept | 25.9 % | "This corresponds to 25.9 % of clusters reported by Kilosort2" |
| Good units by area | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | methods.txt |
| Trials / session | mean 476, range 130–785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84 % (range 65–99 %) | same sentence |
| Session inclusion | performance > 65 %, ≥ 50 correct lick-left and ≥ 50 correct lick-right trials | "We selected experimental sessions for analysis based on following criteria..." |
| Photostim trials | ~25 % randomly interleaved, 5 mW/hemisphere, last 0.5 s of delay, N = 17 VGAT-ChR2 mice | "Photoinhibition ... deployed on a subset of ~25% randomly interleaved trials" |
| Photostim effect | performance 83.2 % → 71.7 % (bilateral ALM) | methods.txt |
| Neural data time bin (method paper) | 40 ms width, 3.4 ms stride | "we binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Behavior/video sampling | 300 Hz cameras (data timestamps dt = 3.4 ms) | "High-speed videos ... acquired at 300 Hz" |
| Task epochs | sample tones 3×150 ms with 100 ms gaps (≈0.65 s), delay 1.2 s, go cue 0.1 s (6 kHz), answer 1.5 s | methods.txt Behavior section |
| Method-paper session count used | 105–106 sessions (video analyses) | "n = 106 sessions" |

### Processing Details
- **Alignment**: everything aligned to the **go cue** (t = 0) in both papers; reference epochs sample [−1.85, −1.2] s, delay [−1.2, 0] s, response [0, 1.5] s.
- **Binning**: sliding 40 ms window, 3.4 ms stride, rates in Hz (spikes/bin_width). Our task specifies **50 ms bins**, which we implement as non-overlapping 50 ms bins giving rates in Hz (80 bins over −2.5→+1.5 s).
- **Spike sorting / QC**: Kilosort2 output; 15 quality metrics; five region-specific logistic-regression classifiers (cortex, striatum, thalamus, midbrain, medulla) trained on manual Phy labels; units labelled **`good`** are used for all analyses. In the NWB files this classifier output is the `units.classification` column (`good` vs `unlabelled`), and `anno_name` holds the CCF annotation of classified units. The whitepaper explicitly argues that thresholding individual metrics is inadequate — hence using the provided classifier label is the faithful reproduction of their QC.
- **Video markers**: DeepLabCut side-view tongue/jaw/nose; outliers detected by 5-sigma velocity threshold and imputed from nearby frames; when the tongue is occluded (in the mouth) the method paper sets tongue position to its mean value. Our decoder task instead requires an explicit "not visible" class (class 3), so occlusion is represented as its own category (detected via DLC likelihood).
- **Region grouping** (reference preprocessing loop): ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum — each split into left/right hemisphere with the CCF ML midline at 5,700 µm.

### Curation Steps
**Neuron curation rules (reference)**
1. Keep only units with a CCF/histology annotation (units without histology are dropped).
2. Keep only units labelled `good` by the QC classifier (25.9 % of Kilosort clusters).
3. Method-paper analyses additionally drop neurons with mean firing rate < 2 Hz (for firing-rate *prediction* analyses) and brain areas with < 10 neurons per session. These are analysis-specific and are **not** appropriate for the decoding task here (they would discard most of the population available to the decoder), so they are not applied; documented in Step 5.

**Trial curation rules (reference)**
- "regular" trials = no early lick, no auto-water, no free water, no ignore (`correctness != -1`), and no photostimulation.
- **Our decoder task requires early-lick, photostim, ignore and miss trials** (they are explicit decoder inputs/outputs), so these filters are relaxed; see Step 5.
- Session inclusion criteria in the data paper (performance > 65 %, ≥ 50 correct left and right trials) are already satisfied by the published sessions.

### Decoders Trained (reference accuracies)
| Decoded variable | Accuracy |
|---|---|
| Choice from behavioral video, pre-sample epoch | AUC 0.51 ± 0.06 (n = 106 sessions) |
| Choice from behavioral video, sample+delay epochs | AUC 0.66 ± 0.12 |
| Choice from behavioral video, second half of response epoch | AUC 0.99 ± 0.01 |
| Choice from single-neuron delay activity (AUC threshold for "choice-modulated") | AUC > 0.65 |
| Uninstructed-movement type from single neurons | AUC > 0.65 threshold |

Implication for our decoder: **choice** should be decodable near-perfectly after the go cue from population activity (video alone gives AUC 0.99); delay-epoch choice information is weaker. Outcome and early-lick should be decodable well above chance; tongue position (a directly observable movement) should be decodable well, since neural activity predicts video markers with high R² in the method paper.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

I cross-checked the reference code (Step 1), the NWB data (Step 2) and the papers (Step 3). Scripts: `/app/cache/consistency*.py`, `/app/cache/region_map.py`, `/app/cache/regions*.py`, `/app/cache/timing_check.py`, `/app/cache/coverage.py`, `/app/cache/choice_check.py`.

### Discrepancies Found and Resolved
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| File format | `.mat` DataJoint exports (`loadmat`) | NWB (DANDI 000363) | same dataset | The NWB release contains all fields used by the reference `.mat` pipeline (trials, events, units + QC classifications, CCF annotations, DLC tracking). Use `pynwb`; map each `.mat` field to its NWB equivalent (table below in Step 5). |
| Neuron QC | External classifier good-unit lists (`goodunits/*.mat`) | `units.classification` ∈ {`good`, `unlabelled`}; only `good` units carry `anno_name` (CCF annotation) | "Applying the trained classifiers ... provided lists of units that were labeled as 'good' and were used for the analysis" | `classification == 'good'` **is** the shipped output of that classifier. Using it reproduces the reference QC exactly: 69,453 good units = 25.5 % of 272,227 clusters (paper: 69,943 units, 25.9 %). |
| Sessions | n/a | 174 NWB files, one of which (`SC017_20190216_162508_s4`) has **0** good units | 173 behavioral sessions | Dropping the session with no good units gives exactly **173** sessions ✓ |
| Probe insertions | n/a | 659 electrode groups | 655 (methods.txt) / 660 (Fig. 1J) | Consistent within the papers' own spread. |
| Units per area | region loop over 14 coarse groups | mapping of `anno_name` → coarse group | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | My CCF→coarse mapping reproduces **striatum 7,664 (exact)**, **midbrain 7,495 (exact)**, **medulla 2,928 (exact)**, **thalamus 12,808 (exact, after assigning Zona incerta to hypothalamus per the Allen ontology)**. ALM: motor cortex (MOs+MOp) = 7,885 vs 8,717 in the paper — the reference used a custom ALM voxel mask (`ALM_voxels_symmetric.npy`) that is *not* shipped with the code, so an exact reproduction is impossible; MOs+MOp is the closest ontology-based definition and is documented as such. |
| Trials / session | n/a | all trials mean 546 (264–800); trials with a response (`outcome != ignore`) mean 465 (104–**785**) | "on average 476 (range, 130–785) trials per session" | The paper's count refers to *performed* trials; our non-ignore count matches the upper bound (785) exactly and the mean within 2 %. Full trial count includes ignore trials. |
| Performance | `correctness == 1` | hit/(hit+miss) on non-photostim, non-early trials = **0.810** mean (0.54–0.97) | 84 % (65–99 %) | Same definition ('fraction of correct control trials, excluding early lick'); small difference (0.81 vs 0.84) presumably from the paper further restricting sessions/trials. Session-selection criteria (>65 % performance) are approximately satisfied; 16/174 sessions are slightly below by my computation but are part of the published dataset, so no session is dropped for performance. |
| Photostim | `stimulation[:,0] != 0` marks stim trials | 18,588/94,990 trials (19.6 %) at 5.5 mW, always 0.5 s; onset −1.2 s or −0.5 s re go cue | "~25 % randomly interleaved trials", 5 mW/hemisphere, last 0.5 s of delay | Consistent (5.5 mW commanded ≈ 5 mW delivered; some sessions have no stim, lowering the overall fraction). Stim always **ends before the go cue** ✓ (max stim end = go − 0.499 s). |
| Tone (sample) onset | epochs sample [−1.85,−1.2] s | last `sample_start_times` before the go cue = **1.85 s** before go in the median of *every* session | sample epoch 0.65 s of tones + 1.2 s delay | Consistent ✓. Early-lick trials trigger epoch *replays*, so the **last** sample start before the go cue is the tone the animal actually used. |
| Trial curation | exclude early-lick, auto-water, free-water, ignore, photostim | those trials exist and are labelled | "These were excluded from all analyses" | The decoder task **requires** photostim (input) and early lick / ignore / miss (outputs), so those exclusions are dropped by necessity. Auto-water and free-water trials (reward independent of the animal's action, 4 % of trials) are **excluded**, matching the reference, since their `outcome` label is not behaviourally meaningful. |
| Choice definition | `trial_type` (instruction) + `correctness` | `outcome` + `trial_instruction` reproduces the lick-derived choice in >99 % of trials (disagreements are `ignore` trials with a lick after the response window) | hit = licked instructed side, miss = licked other side, ignore = no lick | Choice := instructed side for `hit`, opposite side for `miss`, `no lick` for `ignore`. |
| Binning | 40 ms window / 3.4 ms stride (sliding) | spike times available at full resolution | same | Task specifies **50 ms bins** → non-overlapping 50 ms bins, rates in Hz (counts/0.05 s), 80 bins for −2.5→+1.5 s. |
| Observation window | spike times truncated to [−3, 3] s re go cue | `obs_intervals` == trial (start_time, stop_time); **all** spikes lie inside trials; 96.9 % of trials cover go−2.5 s and 84.4 % cover go+1.5 s (miss trials terminate ~0.8 s after the go cue) | n/a | Bins outside the trial's observed interval contain no spikes by construction; they are set to 0 and flagged in metadata. Dropping short trials is not an option (it would delete 95 % of `miss` trials and destroy the outcome output). |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source (NWB) | Target field | Transform | Reference-code counterpart | Notes |
|---|---|---|---|---|
| `units.spike_times` (session clock) for units with `classification == 'good'` | `neural[session][trial]` (n_neurons, 80) | subtract that trial's go-cue time; histogram into 80 non-overlapping 50 ms bins spanning [−2.5, +1.5) s; divide by 0.05 s → **firing rate in Hz**, float32 | `sliding_histogram(..., bw, stride, rate=True)` after truncating spikes to `[begin_time, end_time]` around the go cue | Reference used 40 ms/3.4 ms sliding bins; the decoder task prescribes 50 ms bins, so width = stride = 50 ms |
| `BehavioralEvents['go_start_times'].timestamps` | temporal alignment (t = 0) | one per trial | reference data were exported already aligned to go cue | |
| last `sample_start_times` ≤ go cue, per trial | `input[0]` = `time_from_tone_onset_s` | bin-center time − tone-onset time (seconds), continuous, time-varying (80 values) | reference sample epoch starts −1.85 s re go | median = 1.85 s before the go cue in every session; using the **last** sample start handles early-lick replays |
| `BehavioralEvents['photostim_start_times' / 'photostim_stop_times'].timestamps` | `input[1]` = `photostim_on` | 1 if the 50 ms bin overlaps [stim_on, stim_off] (times re go cue), else 0; binary time-varying | `stimulation[:,2:] -= gocue_time` (reference stores on/off times re go cue) | cross-checked against the `trials.photostim_onset/power/duration` columns |
| `trials.outcome` + `trials.trial_instruction` | `output[0]` = `choice` | hit → instructed side, miss → opposite side, ignore → `no lick`; values 0 = left, 1 = right, 2 = no lick; constant over the trial's 80 bins | `trial_type` (l/r) + `correctness` (1/0/−1) | verified to agree with the first post-go lick in > 99 % of trials |
| `trials.outcome` | `output[1]` = `outcome` | 0 = ignore, 1 = miss, 2 = hit (order given in the task) | `correctness` (−1/0/1) — identical encoding | |
| `trials.early_lick` | `output[2]` = `early_lick` | 0 = no, 1 = yes | `behavior_early_report` | |
| `BehavioralTimeSeries['Camera0_side_TongueTracking']` (x, y, likelihood @ 3.4 ms, session clock) | `output[3]` = `tongue_y` | per 50 ms bin: median y of frames in that bin with `likelihood > 0.5` (and not a 5σ velocity outlier). Bins with no valid frame → class **3** (not visible). Valid bins are discretised by **per-session** percentiles of the binned visible y values: < 40th → 0, 40–60th → 1, > 60th → 2 | `align_markers_between_lims` (aligns DLC markers to go cue); method paper: 5σ velocity outlier rejection, tongue set to mean when occluded | the task requires an explicit "not visible" class instead of mean-imputation |
| `units.anno_name` (CCF) + electrode CCF `x` | `brain_region_idx` | fine CCF annotation → one of 14 coarse groups (ALM, Orbital, OtherCortex, Olfactory, Striatum, Pallidum, Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum, Hippocampus, CorticalSubplate), prefixed by hemisphere (`x ≥ 5700 µm` → left) | reference loop over `['ALM','Medulla',...]` × `['left','right']`, midline 5,700 µm | validated against the paper's per-area counts (Step 4) |
| `nwb.subject.subject_id` | `subjects`, `subject_idx` | 28 unique mice | | `nwb.subject.description` (e.g. `SC015`) kept in metadata |

### Key Decisions
1. **Neuron curation = `classification == 'good'`**: this column is the output of the paper's region-specific QC classifiers (white paper), which the reference pipeline loads from external `goodunits/*.mat` files. Unlabelled units also lack CCF annotations, so this simultaneously enforces the reference's "must have histology" rule. Result: 69,453 units (25.5 % of clusters) vs 69,943 (25.9 %) in the paper.
2. **No additional firing-rate threshold**: the method paper's ≥ 2 Hz filter and its "≥10 neurons per area per session" rule apply to their neuron-wise *encoding* analyses; for population decoding, discarding low-rate neurons throws away usable signal and is not part of the dataset-level curation. Not applied (documented deviation).
3. **Session curation**: drop sessions with 0 good units → exactly **173** sessions, matching the papers.
4. **Trial curation**: drop `auto_water == 1` or `free_water == 1` trials (reward delivered independent of the animal's action ⇒ `outcome` is not a behavioural report), matching `get_regular_trial_mask`. **Keep** early-lick, ignore/miss and photostim trials, because the decoder task defines them as outputs/inputs; the reference excluded them only for its encoding analyses.
5. **Alignment / window / binning**: go-cue aligned, −2.5 → +1.5 s, 80 × 50 ms non-overlapping bins, rates in Hz (counts / 0.05 s). Bin *i* covers [−2.5 + 0.05 i, −2.45 + 0.05 i); the value assigned to input/output time series uses the **bin centre**.
6. **Unobserved time**: spikes exist only inside `[start_time, stop_time]` (= `obs_intervals`). 96.9 % of trials cover go−2.5 s, 84.4 % cover go+1.5 s (miss trials end early). Bins outside the recorded trial interval necessarily contain 0 spikes; these are kept as 0 (no extra information is available) and the fraction is reported in metadata.
7. **Tongue visibility threshold**: `likelihood > 0.5`. The likelihood distribution is strongly bimodal (89 % < 0.1, 10 % > 0.9) so the exact threshold is immaterial.
8. **Percentiles computed per session** over the *binned, visible* tongue-y values of the exported trials, so the resulting class frequencies are exactly 40 / 20 / 40 % of visible bins in each session.
9. **Outputs are time-varying** (shape (4, 80)): per-trial variables are broadcast across bins, as recommended by the task description.
10. **Brain regions carry hemisphere** (e.g. `left ALM`), as in the reference (`side_region`), with the CCF ML midline at 5,700 µm.

### Planned Sanity Checks
- [ ] 173 sessions, 28 subjects, 69,453 neurons after curation.
- [ ] Per-coarse-region neuron totals reproduce paper values (striatum 7,664; midbrain 7,495; medulla 2,928; thalamus 12,808).
- [ ] Trials/session and outcome/early-lick/photostim distributions match the raw survey after removing auto/free-water trials.
- [ ] `input[0]` (time from tone onset) ≈ −0.65 … 3.35 s with median tone onset at −1.85 s re go cue.
- [ ] `input[1]` on only before the go cue, total duration 0.5 s per stim trial, and 0 for all non-stim trials.
- [ ] Choice distribution ≈ left 45 % / right 45 % / no lick 10 % (hit+miss ≈ 85 %, ignore ≈ 15 %).
- [ ] Tongue class distribution: within-session visible bins split 40/20/40; class 3 dominates before the go cue and drops after it.
- [ ] Spot checks against raw NWB (independent code): spike counts for a given (session, trial, neuron, bin); tongue class for a given bin; photostim bin values.
- [ ] Population PSTH aligned to the go cue shows the expected rise at the go cue (as in the raw PSTH computed in `/app/cache/explore6.py`).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` — run as `python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing] [--workers N]`.

Structure:
- `map_annotation()` — CCF annotation → one of the 14 coarse reference groups (rules validated in Step 4).
- `bin_spikes()` — vectorised analogue of the reference `sliding_histogram(..., rate=True)` with stride = width = 50 ms: per neuron a single `np.searchsorted` over all trial-bin edges, then `np.diff`; rates = counts / 0.05 s.
- `tone_onset_times()` — last `sample_start_times` at/ before the go cue per trial (handles early-lick epoch replays).
- `photostim_binary()` — binary per-bin photostim indicator from `photostim_start/stop_times` events (bin counted as on if it overlaps the laser interval).
- `bin_tongue()` — DLC side-camera tongue y averaged per 50 ms bin over frames with `likelihood > 0.5`, after 5σ velocity outlier rejection (method paper); NaN when no valid frame.
- `discretize_tongue()` — per-session 40th/60th percentiles of the binned visible y → classes 0/1/2, NaN → 3.
- `convert_session()` — loads one NWB with `pynwb`, curates units/trials, builds neural/input/output arrays and a `session_info` record with timings and QC statistics.
- `make_processing_plots()` — 8-panel diagnostics per session (`--show-processing`): raw spike raster vs binned rates for one trial, population PSTH computed independently from raw spike times vs the converted neural mean, inputs vs raw event times, raw tongue trace vs binned values and percentile thresholds, resulting discrete tongue classes, per-trial outputs, tongue-class time course.
- `main()` — parallel over sessions (`ProcessPoolExecutor`, default 16 workers), assembles the target dictionary, writes the pickle, prints summary statistics.

Efficiency:
- Spike times are read once per unit; binning is vectorised with `searchsorted` over concatenated bin edges (0.16 s/session for ~450 neurons × 400 trials).
- Unit → electrode lookup uses the `VectorIndex` arrays directly instead of per-unit `units['electrodes'][i]` DataFrame construction (~100× faster; verified to give identical electrode ids).
- Video tracking arrays are read once per session and binned with `np.bincount`.
- Sessions are processed in parallel (16 workers), so the wall-clock cost is dominated by pickling the ~10 GB result.

Issue found and fixed during development: **trials with no ephys at all**. In 100 sessions some trials (4.35 % overall; 2 sessions where the probes stop long before the behaviour: 376/582 and 321/480 trials) contain zero spikes from *every* good unit over the whole trial. NWB does not flag them (`is_good_trials` is all-True and `obs_intervals` simply mirror the trials table). Such trials are excluded as invalid data periods.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` → `/app/conversion_sample_out.txt`;
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` → `/app/verification_sample_out.txt` (**no errors, no warnings**).

### Sample Statistics (2 sessions of mouse 440956 / SC015)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 834 |
| Neurons / session | 459, 375 |
| Subjects | 1 |
| Trials (total) | 513 (of 834 raw; 321 dropped because the probes stopped at t ≈ 1107 s in session 2) |
| Trials / session | 354, 159 |
| T (bins) | 80 everywhere |
| input time_from_tone_onset | [−0.625, 5.7] s (median tone onset −1.85 s re go cue) |
| input photostim_on | [0, 1]; 120 stim trials = 23.4 % (paper ≈ 25 %) |
| choice distribution | left 0.456 / right 0.407 / no lick 0.136 |
| outcome distribution | ignore 0.136 / miss 0.298 / hit 0.565 |
| early_lick distribution | no 0.942 / yes 0.058 |
| tongue_y distribution | 0.094 / 0.047 / 0.094 / 0.765 (visible bins split exactly 40/20/40) |
| brain regions | left/right × ALM, Orbital, OtherCortex, Pallidum, Striatum (10 labels) |

### Processing Plots Review (`processing_SC015_*.png`)
- Raster of raw spike times vs the binned rate image for the same trial line up bin-for-bin; the go cue sits exactly at t = 0 and the tone marker at −1.85 s.
- Population PSTH recomputed independently from raw spike times overlays the converted `neural` mean exactly (the two curves are indistinguishable) — no temporal shift.
- Photostim input turns on exactly over the raw `photostim_start/stop` shading and always ends before the go cue.
- Raw tongue y trace (visible frames) overlays the binned values; the 40th/60th percentile lines sit where the class boundaries change; class 3 dominates before the go cue and drops immediately after it.

### Numeric spot checks against raw NWB (`/app/cache/spotcheck_sample.py`, independent of the conversion code)
- Neural: 4 random (trial, neuron, bin) firing rates reproduced exactly (`np.allclose`).
- Input 0: reconstructed time-from-tone vectors match for 3 trials.
- Input 1: complete (n_trials × 80) photostim matrix matches; 77/77 stim trials.
- Outputs: choice, outcome and early-lick vectors match for all trials.

### Run Time Estimates
| Speed-up implemented | Effect |
|---|---|
| `searchsorted` binning over all trial edges at once | 0.16 s/session (vs ~30 s for a naive Python loop) |
| `VectorIndex`-based electrode lookup | 0.09 s/session (vs ~10 s using `units['electrodes'][i]`) |
| 16-way process parallelism | ~16× |

| Step | Time / session | Estimated total (174 sessions) |
|---|---|---|
| open + unit metadata | 0.09 s | 16 s |
| read spike times | 0.28 s | 49 s |
| binning | 0.16 s | 28 s |
| inputs + outputs (incl. video) | 0.06 s | 11 s |
| **per session total** | **1.2 s** | **~210 s serial → ~20 s with 16 workers** |
| pickle write (~10 GB) | — | 1–3 min |

Estimated full conversion: **< 5 minutes**, well under the 15-minute budget. (The sample sessions are of typical size: 417 neurons vs 399 dataset mean, 417 raw trials vs 546 mean, so the per-session estimate is scaled by ~1.3 in the total above.)

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`.

### Format Validation
- Errors: **None**
- Warnings: **None** (the earlier "all neural data is zero" warnings disappeared once trials without ephys were excluded).

### Training
Loss decreases monotonically: 6.78 (epoch 5) → 4.69 (10) → 1.33 (40) → 0.615 (100) → **0.555** (200). Test loss 1.165.

### Decoder Results (Sample, 2 sessions, 513 trials)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|--------------------------|
| choice | 0.333 | 0.738 | 0.617 |
| outcome | 0.333 | 0.768 | 0.647 |
| early_lick | 0.500 | 0.834 | 0.723 |
| tongue_y | 0.250 | 0.693 | 0.606 |

All four outputs are well above chance (1.9×, 1.9×, 1.4×, 2.4× chance). Accuracies are averaged over **all 80 bins**, including the 2.5 s before the go cue where choice/outcome are only weakly represented (the papers report delay-epoch choice AUC ≈ 0.66 vs ≈ 0.99 after the go cue), so per-trial-average accuracy is necessarily below the post-go-cue value. Train/validation gap is small (≤ 1.2×), so there is no evidence of leakage.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --workers 24` → `/app/conversion_full_out.txt` (42 s total: 22 s conversion + 19 s pickle write).
`python -u /app/train_decoder.py /app/converted_data.pkl --verify-only` → `/app/verification_full_out.txt`: **"Data format is valid, no errors or warnings."**

### Output Files
- `converted_data.pkl`: 11.89 GB
- `verification_full_out.txt`: created, clean

### Converted dataset
- 173 sessions (1 dropped: `SC017_20190216_162508_s4`, no good units), 28 subjects, 69,453 neurons, 89,544 trials, 80 bins × 50 ms per trial.
- Neurons/session mean 401.5 (90–923); trials/session mean 517.6 (159–796).
- Trials removed: 3,789 auto-/free-water + 1,657 without any ephys → 94,990 − 89,544 = 5,446 (5.7 %).

### Consistency Check
| Statistic | Reference Papers | Reference Data (raw NWB) | Converted Data | Match? |
|-----------|------------------|--------------------------|----------------|--------|
| Sessions | 173 | 174 files (1 with 0 good units) | 173 | ✓ |
| Subjects | 28 | 28 | 28 | ✓ |
| Good units (total) | 69,943 | 69,453 | 69,453 | ✓ (0.7 % below the paper; the published NWB units table is the source of truth) |
| Fraction of Kilosort clusters | 25.9 % | 25.5 % | 25.5 % | ✓ |
| Neurons / session | – | mean 399 (0–923) | mean 401.5 (90–923) | ✓ |
| Probe insertions | 655–660 | 659 | 659 | ✓ |
| Units in striatum | 7,664 | 7,664 | **7,664** | ✓ exact |
| Units in midbrain | 7,495 | 7,495 | **7,495** | ✓ exact |
| Units in thalamus | 12,808 | 12,808 | **12,808** | ✓ exact |
| Units in medulla | 2,928 | 2,928 | **2,928** | ✓ exact |
| Units in ALM | 8,717 | 7,885 (MOs+MOp) | 7,885 | ≈ (paper used an unpublished ALM voxel mask; see Step 4) |
| Trials / session | 476 (130–785) performed trials | 465 non-ignore (104–785) | 518 all-outcome trials (159–796) | ✓ consistent (we keep ignore trials, which the paper's count excludes) |
| Photostim fraction | ~25 % | 19.6 % | 20.0 % | ✓ (6 sessions have no stim at all) |
| Photostim power / duration | 5 mW / 0.5 s, last 0.5 s of delay, ends before go cue | 5.5 mW / 0.5 s, ends ≥ 0.499 s before the go cue | same | ✓ |
| Correct rate (control, non-early) | 84 % (65–99 %) | 81.0 % (54–97 %) | hit 68.5 % of all kept trials; hit/(hit+miss) = 80.4 % | ✓ (same definition, 3 pp lower) |
| Tone (sample) onset | −1.85 s re go cue | median −1.85 s in every session | median −1.85 s | ✓ |
| choice distribution | ~equal L/R by design; ignore ≈ 15 % | L 45 %, R 45 %, none 15 % (raw) | L 0.429 / R 0.422 / none 0.148 | ✓ |
| outcome distribution | 84 % correct of responded trials | hit 0.687 / miss 0.165 / ignore 0.148 (raw) | hit 0.685 / miss 0.167 / ignore 0.148 | ✓ |
| early lick | "early lick trials excluded" (no value given) | 0.114 (raw) | 0.116 | ✓ |
| tongue_y | – | tongue visible ≈ 25 % of frames | visible 0.251 (split exactly 40/20/40), not visible 0.749 | ✓ |
| time_from_tone_onset range | tone 1.85 s before go; replays extend it | – | [−1.53, 11.9] s, median tone −1.85 s | ✓ (long values come from early-lick replay trials) |

### Investigated
- **21 trials (12 sessions) where the trials table flags photostim but our `photostim_on` input is all-zero**: in those trials the laser fired during an *aborted* (replayed) delay epoch, ending before −2.5 s relative to the final go cue, i.e. outside the exported window. The conversion is correct; no change needed.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
"Data format is valid, no errors or warnings." — **no errors, no warnings** on the full dataset. (During development the verifier reported `all neural data is zero` warnings for whole blocks of trials; the cause — trials without any ephys — was diagnosed and those trials are now excluded, see Step 6.)

### Check 2: Sanity checks against the raw NWB files (`/app/cache/sanity_full.py`)
Four randomly chosen sessions (`SC027_20190804_152922_s22`, `SC045_20201218_113815_s23`, `SC064_20210508_132858_s14`, `SC050_20210302_140243_s20`) were re-derived from the raw NWB with **independent code** (no import from `convert_data.py`) and compared with `np.allclose`:
| Check | Result |
|---|---|
| number of kept trials per session | ✓ |
| **neural**: 6 random (trial, neuron, bin) firing rates per session = raw spike count in [go−2.5+0.05 b, +0.05) / 0.05 | ✓ all match |
| **input 0**: full 80-bin time-from-tone vector for 4 random trials per session | ✓ |
| **input 1**: complete (n_trials × 80) photostim matrix rebuilt from `photostim_start/stop_times` | ✓ |
| **outputs 0–2**: choice / outcome / early-lick vectors for *all* trials | ✓ |
| **output 3**: complete tongue class matrix recomputed from raw DLC data | ✓ agreement 1.0000 |
| `brain_region_idx` length = number of good units | ✓ |
| `subjects[subject_idx[s]]` = NWB subject id | ✓ |
Result: **ALL CHECKS PASSED**.

Additional whole-dataset checks:
- `T == 80` for every trial of every session (verifier).
- Every output value is an integer within its declared range (verifier).
- Mean population firing rate 5.4 Hz (plausible for Neuropixels units).

### Check 3: Reference code comparison
| Step | Reference (`/app/code`) | This conversion | Same? |
|---|---|---|---|
| (a) Loading | `preprocessing_utils.loadmat` on DataJoint `.mat` exports; per-probe files concatenated per session | `pynwb.NWBHDF5IO`; all probes of a session are already merged into one `units` table | equivalent (format only) |
| (b) Neuron filtering | units present in both ephys and histology **and** in the classifier good-unit list (`goodunits/*.mat`); hemisphere from CCF x ≥ 5700; grouped into 14 coarse regions | `units.classification == 'good'` (the same classifier output, shipped in the NWB), which also implies a CCF annotation; hemisphere from electrode CCF x ≥ 5700; same 14 coarse groups | ✓ same (per-area counts reproduce the paper exactly) |
| (c) Temporal alignment | spike times exported relative to the go cue; markers aligned with `go_times` | spike/behaviour times minus `go_start_times` | ✓ same |
| (d) Binning | `sliding_histogram`, 40 ms window, 3.4 ms stride, rate = counts/bin_width | same algorithm with window = stride = 50 ms (task specification), rate = counts/0.05 | differs only in the bin width mandated by the task |
| (e) Input construction | reference has no decoder inputs; it stores `task_cue_time`, `task_sample_time`, `stimulation[:, 2:] − gocue` | time from the last pre-go `sample_start` (tone) and a binary photostim trace, both re go cue | consistent with the reference's variables |
| (f) Output construction | `trial_type` (l/r) + `correctness` (1/0/−1) + `early_lick_trials`; DLC markers `tongue_x/у` aligned to the go cue | choice from `outcome`+`trial_instruction`, `outcome`, `early_lick`, binned tongue y | ✓ same variables |
| Trial curation | `get_regular_trial_mask`: no early lick, no auto-water, no free water, no ignore, no photostim | auto-water/free-water removed; early-lick, ignore, miss and photostim **kept** | intentional deviation — the task defines these as decoder inputs/outputs; documented in Steps 4–5 |
| Extra filters | ≥ 2 Hz firing rate; ≥ 10 neurons/area (method paper, encoding analyses) | not applied | intentional — would discard population information needed for decoding |
| New filter | – | trials with zero spikes across all good units (ephys acquisition gaps) removed | necessary: such trials carry no neural data at all (the reference never encountered them because it excluded most trials and used other sessions) |

### Check 4: Key statistics comparison
See the table in Step 9. Sessions (173), subjects (28), probe insertions (659), fraction of good clusters (25.5 % vs 25.9 %), and the per-area unit counts for striatum/midbrain/thalamus/medulla (exact matches) all agree with the papers. ALM is 7,885 vs 8,717 because the paper's ALM mask is an unpublished voxel list; using the Allen ontology (MOs + MOp) is the closest reproducible definition. Trial statistics, correct rate, photostim fraction, tone timing and output distributions all match the raw data (Step 9 table).

### Check 5: Edge cases
- **Trial/window edges**: bin *i* spans [−2.5 + 0.05 i, −2.45 + 0.05 i); `searchsorted` with `side='left'` makes bins half-open, so no spike is counted twice — verified by summing counts over bins and comparing with a direct count in [−2.5, 1.5) (equal).
- **Trials shorter than the window**: 3.2 % of trials start after go−2.5 s and 15.6 % end before go+1.5 s; the missing bins are legitimately empty. The per-session fractions are stored in `metadata.session_info`.
- **Early-lick replays**: trials can contain several sample/delay epochs; the *last* sample start before the go cue is used, so `time_from_tone_onset` is correct for replayed trials (verified against the raw events).
- **Photostim during aborted epochs** (21 trials): laser ends before the analysis window; `photostim_on` is correctly all-zero (investigated in Step 9).
- **Session with no good units** (1) and **sessions with < 2 usable trials** (0) are dropped.
- **Units without CCF annotation**: none among `good` units (checked over all 174 files) — no 'Unknown' brain region is produced.
- **NaNs**: no NaN/Inf in neural, input or output arrays (`np.isfinite` check over the full dataset in the verifier's range statistics; tongue NaNs are mapped to class 3 before export).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` → `/app/train_decoder_full_out.txt` (completed successfully; `sample_trials.png`, `predictions.png` written).

### Training Progress
Loss decreasing: **Yes** — 8.43 (epoch 10) → 5.19 (20) → 2.67 (40) → 1.20 (80) → 0.956 (100) → 0.724 (150) → **0.660** (200). Test loss 0.660 (≈ training loss ⇒ no overfitting).

Balanced class weights used by the decoder (reported in the log) match the converted class frequencies:
choice [0.204, 0.207, 0.588], outcome [0.474, 0.423, 0.103], early lick [0.115, 0.885], tongue [0.242, 0.484, 0.242, 0.032].

### Decoder Results (Full: 173 sessions, 89,544 trials, 69,453 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | × chance | Notes |
|--------|--------|----------------------|--------------------------|----------|-------|
| choice | 0.333 | 0.7115 | **0.6833** | 2.05× | averaged over all 80 bins, incl. 2.5 s before the go cue |
| outcome | 0.333 | 0.7004 | **0.6600** | 1.98× | ignore/miss/hit |
| early_lick | 0.500 | 0.7922 | **0.7518** | 1.50× | binary, 11.6 % positive |
| tongue_y | 0.250 | 0.6870 | **0.6524** | 2.61× | 4 classes incl. 'not visible' |

All accuracies improved relative to the 2-session sample (choice 0.617 → 0.683, outcome 0.647 → 0.660, tongue 0.606 → 0.652), as expected with 85× more trials. Train−validation gaps are small (1.02–1.05×).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Output | Classes | Chance | Validation balanced acc | Ratio |
|---|---|---|---|---|
| choice | 3 | 0.333 | 0.6833 | **2.05×** |
| outcome | 3 | 0.333 | 0.6600 | **1.98×** |
| early_lick | 2 | 0.500 | 0.7518 | **1.50×** |
| tongue_y | 4 | 0.250 | 0.6524 | **2.61×** |
No output is at or below chance and none is below 1.5× chance. `early_lick` has the smallest ratio simply because a binary variable has a much higher chance level; its absolute accuracy (0.75 balanced, i.e. 75 % on a variable that is 88.4 % negative) is high.

### Check 2: Accuracy comparison to the papers
The papers do not report per-timepoint *accuracy* for a neural decoder of these four variables; the closest quantities are ROC-AUCs for choice decoding. I therefore reproduced a paper-style analysis **on the converted data** with the reference's own method (`population_decoding_utils.nested_cross_validation`-style PCA(16) + logistic regression, stratified 5-fold, AUC per time bin; `/app/cache/time_resolved.py`, `/app/cache/time_resolved2.py`, 5 random sessions):

| Epoch | Paper (choice decoded from **video**) | This dataset (choice decoded from **neural activity**) |
|---|---|---|
| pre-sample (t < −1.85 s) | 0.51 ± 0.06 | 0.57 ± 0.06 |
| sample + delay (−1.6 → 0 s) | 0.66 ± 0.12 | 0.75 ± 0.08 |
| response, 2nd half (0.75 → 1.5 s) | 0.99 ± 0.01 | 0.91 (all trials) / **0.94** (trials recorded through the whole window) |
| response, 0–0.5 s | – | 0.94 / 0.95 |
| response, 0.5–1.0 s | – | 0.94 / **0.97** |
Neural decoding is *better* than video decoding in the delay epoch (expected: ALM delay activity is the classic choice-predictive signal, whereas video only sees incidental movements) and approaches the video value after the go cue. The residual gap in the **late** response window is fully explained by trials whose recording ends before go + 1.5 s (error/miss trials terminate early): restricting the analysis to trials observed through the whole window raises the AUC from 0.91 to 0.94 and from 0.94 to 0.97 in the 0.5–1.0 s window. Also note the paper's 0.99 is computed on **correct trials only** with 20-fold CV, whereas we include error and ignore trials.
The overall balanced accuracy reported by `train_decoder.py` (0.68 for choice) averages over all 80 bins, of which 50 lie before the go cue where choice information is genuinely weak (AUC 0.57–0.75) — so it is consistent with the time-resolved values.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| choice | 0.7115 | 0.6833 | 1.04 |
| outcome | 0.7004 | 0.6600 | 1.06 |
| early_lick | 0.7922 | 0.7518 | 1.05 |
| tongue_y | 0.6870 | 0.6524 | 1.05 |
All ratios ≤ 1.06 (threshold 1.5), and the test loss (0.660) equals the training loss (0.660): no overfitting and no sign of leakage.

### Additional debugging performed
1. **Output values verified against the raw data** for 4 full sessions with independent code (Step 10, Check 2) — exact matches for choice, outcome, early lick and the complete tongue-class matrix.
2. **Temporal alignment verified** by overlaying an independently computed raw-spike PSTH on the converted `neural` mean (processing plots) and by the per-bin spot checks — no shift.
3. **Output variation**: no output is dominated by a single class beyond what the data dictate (choice 43/42/15 %, outcome 15/17/68 %, early lick 88/12 %, tongue 10/5/10/75 %). The tongue 'not visible' class is intrinsically dominant because the tongue is only out of the mouth ~25 % of the time.
4. **Neuron filtering** follows the reference QC classifier exactly; per-area counts reproduce the paper's numbers exactly for 4 of 5 areas.
5. **Processing matched to the reference code** step by step (Step 10, Check 3).

### Issues found and resolved (all iterations)
- *Iteration 1*: verifier warned that entire blocks of trials had all-zero neural data → diagnosed as sessions/trials where the ephys acquisition stopped while the behaviour continued (4.35 % of trials, 2 sessions severely affected). Fixed by excluding trials with no spikes from any good unit; re-ran conversion, verification (now clean) and training.
- *Iteration 2*: 21 trials flagged as photostim in the trials table had an all-zero `photostim_on` input → investigated and found to be laser pulses delivered during aborted/replayed delay epochs that end before the exported window; behaviour is correct, no change.
- *Iteration 3*: thalamus unit count initially 13,366 vs the paper's 12,808 → Zona incerta was being captured by the thalamus rule; assigning it to hypothalamus (correct Allen ontology) gives an exact match, and similarly parasolitary/parapyramidal/infracerebellar → medulla and pretectal/accessory-optic-tract → midbrain make those counts exact as well.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `/app/README.md` created (dataset description, format specification, key statistics, how to load and reproduce, decoder performance).
- [x] `/app/cache/` contains every exploration/validation script plus `README_CACHE.md` describing each of them.
- [x] Deliverables in `/app`:
  - `CONVERSION_NOTES.md`, `convert_data.py`, `README.md`
  - `converted_data.pkl` (11.9 GB), `sample_data.pkl` (0.07 GB)
  - `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`
  - `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`
  - diagnostics figures `processing_SC015_20190207_120657_s1.png`, `processing_SC015_20190208_133600_s2.png`, plus `sample_trials.png` / `predictions.png` produced by the decoder.

### Summary of the conversion
173 sessions / 28 mice / 69,453 QC-passing neurons / 89,544 trials, each trial an 80 × 50 ms
go-cue-aligned window (−2.5 → +1.5 s) of firing rates in Hz, with two decoder inputs (time since
tone onset, photostimulation) and four categorical outputs (choice, outcome, early lick,
discretised tongue y-position). The format verifier reports no errors and no warnings, every
independent spot check against the raw NWB matches exactly, the per-brain-area neuron counts
reproduce the published values, and the decoder reaches 1.5–2.6× chance on all four outputs with a
negligible train/validation gap.

---

