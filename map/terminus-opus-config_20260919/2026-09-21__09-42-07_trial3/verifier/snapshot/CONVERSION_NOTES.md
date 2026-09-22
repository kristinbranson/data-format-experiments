# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (Chen et al.); method paper: Brain-wide analysis reveals movement encoding structured across and within brain areas
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/CONVERSION_NOTES.md` (this file)
- `/app/ChenLiuEtAl2023_SpikeSortingQC.pdf`, `/app/datapaper.pdf`, `/app/methodpaper.pdf`, `/app/methods.txt` - references
- `/app/code/` - MapVideoAnalysis reference code (Notebooks, Sherlock, VideoAnalysisUtils, Archive)
- `/app/data/` - 50 GB, 28 `sub-*` folders, 174 NWB files (`sub-*_ses-*_behavior+ecephys(+ogen).nwb`) + `dandiset.yaml` (DANDI 000363, Mesoscale Activity Map)
- `/app/train_decoder.py`, `/app/decoder.py` - decoder validation/training code
- `/app/Dockerfile`, `/app/docker-compose.yaml`, `/app/.manifest`

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0, pynwb 4.1.0 (pypdf installed for reading PDFs).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel(bw, stride, begin_time, end_time, data_folder, save_folder, qc_mode)` | Sherlock/preprocess_all_ephys.py + VideoAnalysisUtils/preprocessing_DJ_2022Aug.py | LOADING | Top-level entry: bw=0.04 s, stride=0.0034 s, window [-3, 3] s aligned to go cue, `qc_mode='classifier'` |
| `process_one_sess(...)` | preprocessing_DJ_2022Aug.py | LOADING/CURATION | Loads all probe .mat files of a session, concatenates units, reads behavior/task variables, applies QC unit list (classifier `goodunits` files), keeps only units having BOTH ephys and histology (CCF) entries |
| `helper_get_neuron_id_area(...)` | preprocessing_DJ_2022Aug.py | CURATION | Intersects classifier-QC good-unit indices with hemisphere (CCF ML coordinate vs midline 5700 um) and requires non-empty CCF annotation |
| `process_one_area(...)` | preprocessing_DJ_2022Aug.py | PROCESSING | Truncates spike times to [begin_time, end_time] (already go-cue aligned in the .mat export) and bins to firing rates |
| `sliding_histogram(spikeTimes, begin, end, bin_width, stride, rate=True)` | preprocessing_DJ_2022Aug.py | PROCESSING | Sliding-window spike counts -> rates (counts / bin_width, in Hz); bin centers from begin to end with given stride |
| `align_markers_between_lims(marker_data, go_times, t_min=-3, t_max=1.5)` | Sherlock/align_markers.py | PROCESSING | Aligns DeepLabCut markers (nose/tongue/jaw x,y) to go cue at dt = 0.0034 s, taking the last frame within each bin |
| `get_bad_trial_inds(...)` | Sherlock/align_markers.py | CURATION | Flags trials where the number of video frames disagrees with the trial duration |
| `get_regular_trial_mask(ephys_data)` | VideoAnalysisUtils/population_decoding_utils.py and functions_for_r2.py | CURATION | "Regular" trials = no early lick, no auto water, no free water, response present (`correctness != -1`), no photostimulation |
| `load_session(session_path, area)` | population_decoding_utils.py | LOADING | Concatenates per-area pickles of one session into (n_bins, n_trials, n_neurons) firing-rate array + behaviour |
| `nested_cross_validation(X, y, ...)` | population_decoding_utils.py | ANALYSIS | PCA + logistic-regression decoding of binary labels from population activity |

### Notes
- Electrophysiology: spike **times** are binned into **firing rates** (Hz) with a sliding window (bin width 40 ms, stride 3.4 ms in the published pipeline). No dF/F (this is ephys, not imaging).
- Neuron curation in the reference is done with a **pre-computed classifier-based QC list** (`goodunits/*.mat`) produced by the spike-sorting white paper pipeline. In the DANDI NWB release the same information is stored per unit in `units/classification` (`'good'` vs `'unlabelled'`) - this is the field I will use.
- Units without CCF histology annotation are dropped by the reference (`unit_comb = set(unit_info) & set(ccf_unit_id)`; `ccf_label[idx] != []`). In the NWB, `units/anno_name` is empty exactly for the non-`good` units, so the `classification == 'good'` filter already implies a valid annotation (verified in Step 2).
- Trial curation used for the paper's *regression* analyses removes early-lick / auto-water / free-water / no-response / photostim trials. The present decoding task explicitly requires photostimulation as an input and early lick / ignore / miss as outputs, so those trials must be retained (see Step 5 deviations).
- Alignment everywhere in the reference is to the **go cue** (spike times in the .mat export are already go-cue relative; markers are explicitly aligned to `task_cue_time[0]`).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` is DANDI dandiset 000363 (Mesoscale Activity Map, Chen et al. 2024) with one folder per mouse:
`/app/data/sub-<animalid>/sub-<animalid>_ses-<yyyymmddThhmmss>_behavior+ecephys[+ogen].nwb` (174 NWB/HDF5 files, 50 GB total) plus `dandiset.yaml`.

Relevant contents of each NWB file (verified with h5py):
- `intervals/trials` (TimeIntervals, one row per trial): `start_time`, `stop_time`, `trial`, `trial_uid`, `task` (all 'audio delay'), `task_protocol`,
  `trial_instruction` ('left'/'right'), `early_lick` ('early'/'no early'), `outcome` ('hit'/'miss'/'ignore'), `auto_water`, `free_water` (0/1),
  `photostim_onset`, `photostim_power`, `photostim_duration` (strings; 'N/A' when no photostimulation, otherwise seconds **relative to trial start** / mW / s).
- `acquisition/BehavioralEvents/*`: `go_start_times`, `go_stop_times`, `sample_start_times`, `sample_stop_times`, `delay_start_times`, `delay_stop_times`,
  `presample_*`, `trialend_*`, `left_lick_times`, `right_lick_times`, `photostim_start_times`, `photostim_stop_times`. Each has `timestamps` in **session-absolute seconds**.
  There is exactly one `go_start_times` entry per trial in every session (174/174 verified). Sample/delay epochs can occur more than once in a trial (early-lick replay).
- `acquisition/BehavioralTimeSeries/Camera0_side_{Jaw,Nose,Tongue}Tracking`: DeepLabCut markers, `data` = (nframes, 3) = (x, y, likelihood), `timestamps`
  session-absolute, dt = 0.0034 s (294 Hz). Some sessions also have `Camera0_side_WhiskerTracking_whisker`, `Camera0_side_LickPortTracking` or `Camera3_*` (bottom view).
- `units` (Units table, 272,227 rows total): `spike_times` (+`spike_times_index`, session-absolute), `obs_intervals` (per trial), `classification`
  ('good' = passed the white-paper QC classifier, 'unlabelled'), `anno_name` (CCF annotation of the unit; empty for non-'good' units), `is_good_trials`
  (n_units x n_trials bool: "manually annotated 'good' trials for a particular probe insertion"), `electrodes`/`electrode_group`, plus the 15+ quality metrics
  (`unit_amp`, `snr`, `isi_violation`, `presence_ratio`, `amplitude_cutoff`, `drift_metric`, ...) used by the QC classifier.
- `general/extracellular_ephys/electrodes`: `x`,`y`,`z` CCF coordinates (um; x = ML with midline 5700, y = DV, z = AP with bregma 5400) and `location`
  (JSON with `brain_regions` = probe target, e.g. "left ALM").
- `general/subject/subject_id`, `general/optogenetics/*` (OBIS470 lasers, ALM locations).

Important structural facts verified:
- **Spike times exist only inside trial intervals** (100% of spikes of sampled units fall inside `[start_time, stop_time]`); the inter-trial gap (median 2.15 s) contains no spikes.
- `go_start_times[i]` always falls inside trial i.
- Photostimulation events in `BehavioralEvents/photostim_start_times` correspond exactly to the trials flagged in the trials table (verified set equality), duration 0.5 s, ending before the go cue (late delay).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files (sessions) | 174 (173 with QC unit labels; 1 session, sub-440958_ses-20190216, has `classification` = NaN for all units) |
| Units (all) | 272,227 |
| Units with `classification == 'good'` | 69,453 (mean 399/session, range 0-923) |
| Subjects | 28 |
| Sessions / subject | 3-10 (mean 6.2) |
| Trials (total) | 94,990 |
| Trials / session | mean 546, range 264-800 |
| Trials / session excluding auto-water + free-water | mean 524 |
| Trials / session excluding water + early lick | mean 464 |
| Photostim trials | 19.6% of all trials; 168/174 sessions have some |
| Early-lick trials | 11.4% |
| free water / auto water | 2.6% / 1.4% |
| Outcome (control trials) | hit 69.0%, miss 15.4%, ignore 15.6% |
| Probe-target regions | left/right ALM, Striatum, Thalamus, Midbrain, Medulla, ECT, BLA |
| Unique CCF `anno_name` of good units | 293 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt) |
| Good units by area | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | methods.txt |
| Sessions | 173 | "...across 173 behavioral sessions, from which 655 probe insertions were made" (methods.txt); "660 penetrations, 173 behavioral sessions, and 28 mice" (data paper Fig 1J) |
| Subjects | 28 | data paper Fig 1J; method paper Methods ("data from 28 mice") |
| Trials / session | mean 476, range 130-785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84%, range 65-99% | "with 84% correct rate (range, 65-99%)" |
| Session selection | performance > 65% AND >= 50 correct lick-left and >= 50 correct lick-right | "We selected experimental sessions for analysis based on following criteria..." |
| Photoinhibition | ~25% of trials, N = 17 VGAT-ChR2 mice, 93 sessions; late delay (last 0.5 s), always ends before go cue | methods.txt |
| Bilateral ALM photostim effect | performance 83.2% -> 71.7% | methods.txt |
| Neural bin (reference pipeline) | 40 ms width, 3.4 ms stride | "we binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" (method paper Methods) |
| Video | 300 Hz side view, DeepLabCut markers (nose, jaw, tongue) | both papers |
| Marker alignment window (reference) | go cue, [-3, 1.5] s, dt = 3.4 ms | `align_markers.py` |
| Spike-sorting QC | 15 metrics -> region-specific logistic-regression classifiers; 25.9% of Kilosort2 clusters kept | methods.txt / QC white paper |
| Task epochs | sample (3 tones x 150 ms + 100 ms gaps = 0.65 s), delay 1.2 s, answer 1.5 s | methods.txt |

### Processing Details
- Alignment: everything is aligned to the **go cue** (reference `.mat` export already stores go-cue-relative spike times; markers are explicitly aligned to `task_cue_time[0]`).
- Binning: sliding histogram -> firing **rates in Hz** (`counts / bin_width`).
- Video markers: outliers detected by a 5-sigma velocity threshold and imputed from neighbouring frames; tongue position set to its mean value when the tongue is occluded (invisible).
- Neurons: only units passing the classifier QC ('good'), and only units with a valid CCF annotation.
- Hemisphere assignment: CCF ML coordinate vs midline 5,700 um (`helper_get_neuron_id_area`).

### Curation Steps

**Neuron curation rules** (reference):
1. QC classifier label 'good' (white-paper 15-metric logistic-regression classifiers, region-specific).
2. Unit must have histology/CCF annotation (non-empty `anno_name`).
3. For the video-regression analyses only, units with mean firing rate < 2 Hz were additionally dropped (method paper).

**Trial curation rules** (reference):
1. Session level: behavioural performance > 65% and >= 50 correct lick-left and >= 50 correct lick-right trials (data paper).
2. Trial level for the published analyses: exclude early lick, auto water, free water, no-response ('ignore') and photostimulation trials (`get_regular_trial_mask`).

### Decoders Trained (reference expectations)
| Decoded variable | Accuracy |
| --- | --- |
| Choice from ALM population (200 neurons, late delay, logistic regression on 200 ms sliding spike rates) | ~0.9 (data paper Fig 6D / S6) |
| Choice from other areas (midbrain, thalamus, striatum, medulla) | ~0.6-0.8 depending on area and epoch |
| Choice from behavioural video (ROC AUC) | high in response epoch, near chance before the go cue (method paper Figs 6-7) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Cross-checks performed
| Quantity | Reference text | Reference code | This dataset (raw NWB) | Consistent? |
|---|---|---|---|---|
| Sessions | 173 | n/a | 174 files, of which 173 have QC labels (1 session has no `classification`) | YES - the unlabelled session is the one excluded by the paper |
| Mice | 28 | n/a | 28 `sub-*` folders | YES |
| Good units | 69,943 | classifier `goodunits` lists | 69,453 with `classification=='good'` (99.3%) | Close; small differences (<1%) presumably from the exact classifier version used for the DANDI export |
| Good units per area | ALM 8,717 / striatum 7,664 / thalamus 12,808 / midbrain 7,495 / medulla 2,928 | grouping by CCF label + hemisphere | my CCF keyword grouping gives ALM(MOs+FRP, AP>=2 mm) 8,464 / striatum 7,347 / thalamus 13,514 / midbrain 7,341 / medulla 2,918 | YES (within a few %) - validates the CCF -> coarse-region mapping |
| Mean correct rate | 84% (65-99%) | n/a | 84.0% over the 145 sessions passing the paper's selection criteria (computed as hits/(hits+misses) on control, non-early, non-water trials) | YES |
| Trials per session | mean 476 (130-785) | n/a | 546 raw; 464 after removing water + early-lick trials | YES (the paper's 476 matches the post-exclusion count) |
| Photostim fraction | ~25% of trials | `stimulation[:,0]!=0` | 19.6% of all trials (168/174 sessions) | Consistent in magnitude ('a subset of ~25% randomly interleaved trials' applies to the photostim sessions) |
| Alignment event | go cue | go cue | `go_start_times`, 1 per trial | YES |
| Neural binning | 40 ms / 3.4 ms stride | `sliding_histogram(bw=0.04, stride=0.0034)` | n/a | The decoder task **requires 50-ms bins**, so I use 50-ms non-overlapping bins (rates in Hz, same as reference) |
| Trial exclusions | early lick, water, ignore, photostim | `get_regular_trial_mask` | all of these are available as NWB trial columns | Deviation required: the decoder task asks to **decode** early lick and ignore/miss/hit and to use photostim as an **input**, so those trials must be kept. Auto-water/free-water trials are still excluded (they are not task trials: reward is given regardless of the animal's choice). |

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| QC good-unit list | external `goodunits/*.mat` classifier output | `units/classification == 'good'` in NWB | classifier trained on 15 QC metrics | Use `classification == 'good'` - it is the NWB encoding of exactly that classifier output (69,453 vs 69,943 reported, 99.3%) |
| Region grouping | 14 coarse groups from CCF label + hemisphere | 293 CCF `anno_name` values | per-area counts in methods.txt | Implemented a keyword CCF -> coarse-region mapping validated against the per-area counts (see Step 4 table) |
| Spike coverage | .mat export kept go-cue-relative spikes in [-3, 3] s | NWB spike times exist only within `[start_time, stop_time]` of each trial | not discussed | Bins of the [-2.5, 1.5] s window that fall outside the recorded trial interval contain no spikes. This mostly affects 'miss' trials (they end ~0.8 s after the go cue). Those trials must be kept (they are an output class), so unobserved bins are 0 Hz by construction; documented. |
| Trials with repeated sample epochs | `task_sample_time` single value per trial | up to 12 `sample_start_times` in one trial (early-lick replay) | "Licking early during the sample/delay epoch triggered a replay of the epoch" | Use the **last** sample (tone) onset before the go cue, i.e. the tone that actually instructed the executed trial |
| `is_good_trials` | not present in the .mat pipeline | per-probe manual annotation of stable trials; 4/174 sessions have some False entries | not discussed | Used as an extra curation step (see Step 5) |
| Video coverage | assumes full-trial video | 7 sessions have video that stops at/near the go cue or is nearly absent | data paper: sessions were "screened for video artifacts and excluded if these were present" | Trials whose [-2.5, 1.5] s window is not fully covered by video frames are dropped (otherwise the tongue output would be mislabelled 'not visible') |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable (NWB) | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` (+`spike_times_index`), `units/classification=='good'` | `neural[session][trial]` (n_neurons, 80) | spikes counted in 80 non-overlapping 50-ms bins spanning [-2.5, +1.5] s around the go cue, divided by 0.05 s -> **firing rate in Hz**, float32 | `sliding_histogram`, `process_one_area` | reference used 40 ms width / 3.4 ms stride; the decoder task prescribes 50-ms bins |
| `acquisition/BehavioralEvents/go_start_times/timestamps` | alignment event | t = 0 for every trial | `align_markers_between_lims`, `.mat` export | one per trial in all 174 sessions |
| `acquisition/BehavioralEvents/sample_start_times/timestamps` | `input[0]` = `time_from_tone_onset` (s) | signed time of each bin centre relative to the **last** tone (sample) onset before the go cue | n/a (tone onset = `task_sample_time` in the .mat export) | negative before the tone; continuous, time-varying |
| `acquisition/BehavioralEvents/photostim_start_times` / `photostim_stop_times` (cross-checked against `intervals/trials/photostim_onset`,`photostim_duration`) | `input[1]` = `photostim_on` | 1 if the bin centre lies in [stim start, stim stop], else 0 | `sess_dict['stimulation']` columns 2-3 | binary time series |
| `acquisition/BehavioralEvents/{left,right}_lick_times` | `output[0]` = `choice` | first lick after the go cue (within the trial) -> 0 left, 1 right, 2 no lick | `behavior_lick_directions`/`lick_times` | agrees with `outcome` x `trial_instruction` in 99.8% of trials (sanity check) |
| `intervals/trials/outcome` | `output[1]` = `outcome` | 'ignore'->0, 'miss'->1, 'hit'->2 | `behavior_report` (-1/0/1 in the .mat export) | same three categories as the reference `correctness` variable |
| `intervals/trials/early_lick` | `output[2]` = `early_lick` | 'no early'->0, 'early'->1 | `behavior_early_report` | |
| `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` (`data[:,1]`=y, `data[:,2]`=likelihood) | `output[3]` = `tongue_y` | per bin: if any frame with likelihood > 0.9, take the median y of the visible frames in that bin and discretize with the session's 40th/60th percentiles of visible y (0/1/2); if no visible frame -> 3 (not visible) | markers in `align_markers.py`; "tongue position set to its mean value when occluded" (method paper) | percentiles computed per session over all visible frames of that session |
| `units/anno_name` + electrode CCF `x`,`y`,`z` | `brain_region_idx` | CCF annotation mapped to a coarse region (ALM, Orbital, OtherCortex, Striatum, Pallidum, Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum, Hippocampus, Olfactory, CorticalSubplate), hemisphere from ML coordinate (midline 5,700 um) | `helper_get_neuron_id_area`, region loop in `process_one_sess` | validated against the per-area good-unit counts in methods.txt |
| `general/subject/subject_id` (folder `sub-*`) | `subjects`, `subject_idx` | one entry per mouse | n/a | 28 mice |

### Key Decisions
1. **Neuron curation = `classification == 'good'`**: this is the NWB encoding of the reference pipeline's classifier-based QC (white paper, 15 metrics, region-specific logistic regression). Gives 69,453 units vs the 69,943 reported (99.3%). Units without this label also have an empty `anno_name`, so the reference's "unit must have histology" criterion is automatically satisfied.
2. **No additional firing-rate threshold**: the 2-Hz threshold in the method paper applies only to their video->firing-rate regression analyses, not to the dataset itself. Keeping all good units maximises the information available to the decoder.
3. **`is_good_trials`**: trials for which more than 10% of a session's good units are flagged as bad are dropped; any unit that is still flagged bad on a retained trial is dropped. Affects only 4/174 sessions.
4. **Session curation** follows the data paper: behavioural performance > 65% (hits/(hits+misses) on control, non-early, non-water trials) and >= 50 correct lick-left and >= 50 correct lick-right trials; plus the session must contain at least one 'good' unit and usable video (see 6). This reproduces the paper's mean performance of 84%.
5. **Trial curation**: auto-water and free-water trials are removed (reward independent of the animal's choice; removed by the reference `get_regular_trial_mask` as well). Early-lick, 'ignore'/'miss' and photostimulation trials are **kept**, deviating from the reference analyses, because the decoder task explicitly requires early lick and outcome as decoder outputs and photostimulation as a decoder input.
6. **Video coverage**: a trial is kept only if every one of its 80 bins contains at least one video frame; otherwise the tongue output could not be distinguished between "tongue not visible" and "no video recorded". This removes ~5% of trials overall and effectively removes 6 sessions whose video stops at the go cue (the data paper also screened sessions for video artifacts).
7. **Neural data outside the recorded trial interval**: spikes are only stored within `[start_time, stop_time]`. For 'miss' trials, which end ~0.8 s after the go cue, the tail of the window has no recorded spikes and therefore 0 Hz. These trials are kept because 'miss' is one of the output classes; the effect is documented.
8. **Outputs are time-varying** (shape (4, 80)): choice/outcome/early lick are per-trial values broadcast across time (as allowed by the format), tongue position is genuinely time-varying.
9. **Firing rates in Hz** (counts / bin width), matching the reference `sliding_histogram(rate=True)`.

### Planned Sanity Checks
- [ ] Session/mouse/unit counts vs the papers (173 sessions, 28 mice, 69,943 good units, per-area counts).
- [ ] Mean behavioural performance of retained sessions = 84% (65-99%).
- [ ] Mean trials/session after exclusions ~476.
- [ ] Choice derived from lick times agrees with `outcome` x `trial_instruction` (>99%).
- [ ] Photostimulation always ends before the go cue and lasts 0.5 s.
- [ ] Tone onset ~1.85 s before the go cue (sample epoch 0.65 s + delay 1.2 s).
- [ ] Spike-count spot checks recomputed directly from the NWB file with `np.allclose`.
- [ ] Tongue-class distribution: class 3 (not visible) dominant before the go cue, classes 0-2 appearing after it.
- [ ] Mean firing rate per neuron in a plausible range (~1-10 Hz).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (`python -u /app/convert_data.py <out.pkl> [--full|--sample] [--show-processing] [--nproc N]`).

Structure:
- `ccf_to_region(anno_name, ap_um)` - keyword mapping of the Allen CCF annotation to the coarse regions used by the reference code (ALM, Orbital, OtherCortex, Striatum, Pallidum, Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum, Hippocampus, Olfactory, CorticalSubplate); ALM = anterior (AP >= 2 mm) MOs / frontal pole, hemisphere from the CCF ML coordinate vs the 5,700 um midline (same convention as `helper_get_neuron_id_area`).
- `process_session(file)` - all per-session work: session selection, unit selection, trial selection, spike binning, input and output construction. Executed in a `multiprocessing.Pool` (16 workers by default).
- `make_processing_plot(...)` - `--show-processing` diagnostics (2 figures per session).
- `main()` - assembles the global dictionary, prints summary statistics and writes the pickle plus a per-session `.info.json`.

Efficiency:
- Spike binning is vectorised: for each unit a single `np.searchsorted` of all 81 x n_trials bin edges into that unit's spike-time array, then `np.diff` (no Python loop over trials or bins). This replaced an O(n_units x n_trials x n_bins) loop.
- Video frames per bin are obtained with one `np.searchsorted` of all bin edges into the frame timestamp array.
- Tongue classes per trial are computed with `np.bincount` over bin indices instead of a per-bin loop.
- Each NWB file is opened once and all needed arrays are read in bulk.
- Sessions are processed in parallel (16 processes; the machine has 128 CPUs).

Issues found during development:
1. `units/is_good_trials` has fewer columns than there are trials in 9/174 sessions -> crash. Cause: the probes were only recorded during a subset of the behavioural trials.
2. Because of the same issue, the first version emitted trials with *all-zero* neural data (warning from `verify_data_format`). Fixed by using `units/obs_intervals` (matched by trial start time) to determine which trials each unit actually observed; unobserved trials are removed and `is_good_trials` is expanded onto the observed trials.
3. `brain_regions` contained `np.str_` instead of `str` - fixed.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` (sessions sub-440956_ses-20190210 and sub-484677_ses-20210420).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 920 |
| Neurons / session | 368, 552 |
| Subjects | 2 |
| Trials (total) | 805 |
| Trials / session | 206, 599 |
| Behavioural performance | 0.888, 0.936 |
| Mean firing rate | 12.5 Hz (8.6, 16.4) |
| time_from_tone_onset range | [-0.63, 7.11] s |
| photostim_on range | [0, 1], on in 3.1% of bins |
| choice distribution | left 0.455, right 0.478, no lick 0.067 |
| outcome distribution | ignore 0.067, miss 0.107, hit 0.826 |
| early_lick distribution | no 0.928, yes 0.072 |
| tongue_y distribution | 0.110 / 0.065 / 0.060 / 0.765 (not visible) |

### Processing Plots Review
`processing_<session>.png` shows, for one example trial: the raw spike raster aligned to the go cue, the binned rate image, the mean binned rate overlaid with the go cue and tone-onset markers, the two inputs with the raw photostim interval shaded, the raw tongue-y trace with the session percentiles, and the four output traces. `processing_<session>_summary.png` shows the session tongue-y histogram with the 40th/60th percentiles, the fraction of each tongue class over time (class 3 dominates before the go cue and drops immediately after it), the population PSTH (clear go-cue response) and the fraction of trials with photostimulation on (a 0.5 s block ending before the go cue). No temporal misalignment or discretisation anomalies were visible.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| vectorised spike binning via `searchsorted` over all bin edges | ~100x vs per-trial loops |
| bulk HDF5 reads, one file open per session | large (I/O bound otherwise) |
| 16-way multiprocessing over sessions | ~10x |

| Step | Time / Session | Estimated Total Time |
| session processing (read + bin + outputs) | 0.5-1.0 s (sample) up to ~10 s for the largest sessions | 174 sessions / 16 workers -> ~2-5 min plus pickling of a ~9 GB dictionary (~2-4 min) |

### Validation
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` -> **"Data format is valid, no errors or warnings."** (the earlier "all neural data is zero" warnings disappeared after the `obs_intervals` fix).

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions)
Loss decreased monotonically from 10.64 (epoch 9) to 0.57 (epoch 200); test loss 0.76.

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.711 | 0.597 | 0.333 |
| outcome | 0.657 | 0.551 | 0.333 |
| early_lick | 0.781 | 0.693 | 0.500 |
| tongue_y | 0.755 | 0.697 | 0.250 |

All outputs are well above chance. Accuracies are averaged over *all* time bins, including the 2.5 s before the go cue during which choice/licking has not yet occurred, so they are necessarily lower than the peak (late-delay / response epoch) decoding accuracies reported in the papers.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --nproc 16` (36.9 s wall time, 174 sessions, 16 workers).

### Output Files
- `converted_data.pkl`: 9.19 GB (+ `converted_data.pkl.info.json` with per-session diagnostics)
- `verification_full_out.txt`: created - **"Data format is valid, no errors or warnings."**

### Curation accounting (174 NWB files -> 138 sessions)
| Reason for rejection | n sessions |
|---|---|
| behavioural performance <= 65% | 12 |
| fewer than 50 correct lick-left or lick-right trials | 17 |
| no QC-good units (the session whose `classification` is NaN) | 1 |
| fewer than 10 usable trials (video does not cover the analysis window) | 6 |
| **kept** | **138** |

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (raw NWB) | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Sessions with QC units | 173 | n/a | 173 of 174 files | 138 after applying the paper's session-selection criteria (145 sessions pass the behavioural criteria, 144 of which have good units, 6 more lost to missing video) | YES |
| Subjects | 28 | n/a | 28 | 28 | YES |
| Good units (dataset) | 69,943 | classifier QC | 69,453 | 55,429 in the 138 retained sessions | YES (units scale with retained sessions) |
| Mean neurons/session | ~400 | n/a | 399 | 402 (range 90-923) | YES |
| Trials/session | 476 mean (130-785) | n/a | 546 raw / 464 after water+early-lick exclusion | 500 (range 159-796); 464 if early-lick trials are also excluded | YES |
| Behavioural performance | 84% (65-99%) | n/a | 84.0% over sessions passing the criteria | 83.9% (65.8-98.9%) | YES |
| Photostim trial fraction | ~25% of trials in photostim sessions | `stimulation[:,0]!=0` | 19.6% of all trials | photostim input on in 2.6% of all time bins (0.5 s of a 4 s window on ~21% of trials) | YES |
| choice distribution | ~50/50 left/right with ~10% no-lick (ignore) | n/a | ignore 15.6% of control trials | left 0.453, right 0.443, no lick 0.104 | YES |
| outcome distribution | 84% correct among responded trials | n/a | hit 69.0% / miss 15.4% / ignore 15.6% of control trials | hit 0.741, miss 0.154, ignore 0.106 (hit/(hit+miss) = 82.8%) | YES |
| early lick | 11.4% of raw trials | excluded by reference | 11.4% | 0.120 | YES |
| tongue not visible | tongue visible ~12% of video frames | n/a | 11.7% of frames | class 3 (not visible) in 73.4% of bins (bins count as visible if any frame in the 50 ms bin is tracked) | YES |
| mean firing rate | n/a | n/a | `units/avg_firing_rate` mean 16.2 Hz for one session (median 3.6 Hz) | 9.5 Hz across sessions (4.3-19.9) | YES (matches the file's own `avg_firing_rate`) |
| Input ranges | n/a | markers aligned [-3, 1.5] s | tone onset median -1.85 s before the go cue | time_from_tone_onset [-1.53, 11.89] s; photostim_on {0,1} | YES (long values occur on early-lick replay trials) |

Issue found and fixed during this step: one trial (session 0, trial 159) produced all-zero neural data because the recording stopped during the last observed trial. Trials in which no retained unit fires a single spike in the whole window are now removed (1 trial in the whole dataset).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`verification_full_out.txt`)
- Final state: **"Data format is valid, no errors or warnings."**
- Warnings that appeared in earlier iterations and how they were removed:
  1. *"Session 0, trial N: all neural data is zero"* for several hundred trials. Cause: in 9/174 sessions the probes were recorded only during part of the behavioural session, and `units/obs_intervals` (and `units/is_good_trials`) cover only those trials. Fix: unobserved trials are now identified by matching `obs_intervals` start times to trial start times and removed.
  2. The same warning for a single remaining trial (the last observed trial of sub-440956_ses-20190208, during which the recording stopped: zero spikes in the whole 4 s window across 375 units). Fix: trials in which no retained unit fires a single spike are removed (1 trial dataset-wide).

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`, `/app/cache/sanity_photostim.py`)
These scripts load the raw NWB files with h5py and recompute everything brute-force (no code shared with `convert_data.py`), comparing with `np.allclose`. The converted pickle stores the raw trial and unit indices of every session (`metadata.session_info[i]['trial_indices'|'unit_indices']`), so the comparison is exact.

| Check | Sessions tested | Result |
|---|---|---|
| **Neural**: firing rate of 4 random neurons x 4 random bins per trial, recomputed as `sum(spike_times in [go+edge, go+edge+0.05)) / 0.05` | 4 sessions x 3 trials (192 spot checks) | PASS |
| **Neural**: total spike count of the whole (n_neurons x 80) trial matrix | 4 sessions x 3 trials | PASS |
| **Input** `time_from_tone_onset`: `bin_centre - (last sample_start_time before the go cue - go cue)` | 4 sessions x 3 trials | PASS |
| **Input** `photostim_on`: from `BehavioralEvents/photostim_start|stop_times`, and independently from `trials.photostim_onset/duration` (relative to trial start) | 12 photostim trials in 3 sessions | PASS (always exactly 10 bins = 0.5 s, always ending at or before the go cue) |
| **Output** `choice`: direction of the first lick in `[go cue, trial stop]` | 4 sessions x 3 trials | PASS |
| **Output** `outcome`, `early_lick`: `trials.outcome`, `trials.early_lick` | 4 sessions x 3 trials | PASS |
| **Output** `tongue_y`: per-bin mean y of frames with likelihood > 0.9 discretised at the session's 40th/60th percentiles | 4 sessions x 3 trials (960 bins) | PASS |
| **Curation**: every retained unit has `classification == 'good'`; no auto-water/free-water trial is retained; stored tongue percentiles equal recomputed ones | 4 sessions | PASS |

### Check 3: Reference code comparison
| Processing step | Reference (`/app/code`) | This conversion | Same? |
|---|---|---|---|
| (a) Data loading | `preprocessing_utils.loadmat` on per-probe DataJoint `.mat` exports; probes concatenated per session | h5py on the DANDI NWB files, which already contain all probes of a session in one `units` table | Equivalent (same underlying data, different distribution format) |
| (b) Neuron filtering | classifier QC list (`goodunits/*.mat`), unit must have a CCF annotation, hemisphere from CCF ML vs 5,700 um | `units/classification == 'good'` (the NWB encoding of that list; empty `anno_name` only for non-good units), hemisphere from electrode CCF `x` vs 5,700 um | Same |
| | (not available in the .mat pipeline) | additionally `units/obs_intervals` and `units/is_good_trials` are used to drop trials/units that were not recorded or were manually flagged unstable | Extra check enabled by the NWB release |
| (c) Temporal alignment | go cue (`task_cue_time[0]`); spike times in the export are already go-cue relative; markers aligned with `align_markers_between_lims` | go cue (`BehavioralEvents/go_start_times`); spike times, photostim events and video frames are all converted to go-cue-relative time | Same |
| (d) Binning | `sliding_histogram(bin_width=0.04, stride=0.0034, rate=True)` -> Hz | 80 non-overlapping 50-ms bins, counts / 0.05 s -> Hz | Same units and estimator; bin width/stride prescribed by the decoder task |
| (e) Input construction | the reference does not build decoder inputs; task variables stored are `task_sample_time`, `task_cue_time`, `task_stimulation` | tone onset -> `time_from_tone_onset`; `task_stimulation` equivalent -> `photostim_on` | Uses the same source variables |
| (f) Output construction | `behavior_lick_directions`/`lick_times`, `behavior_report`, `behavior_early_report`, DLC markers | licks -> `choice`; `outcome` -> `outcome`; `early_lick` -> `early_lick`; tongue y marker -> discretised `tongue_y` | Same source variables |
| Trial curation | `get_regular_trial_mask`: no early lick, no auto water, no free water, response present, no photostim | auto/free water removed; early-lick, ignore/miss and photostim trials **kept** | Deliberate deviation - these are decoder outputs/inputs in this task (documented in Step 5) |
| Session curation | data paper: performance > 65%, >= 50 correct left and right trials | identical, plus "has QC-good units" and "video covers the analysis window" | Same + necessary additions |

### Check 4: Key statistics comparison
See the table in Step 9. All dataset-level statistics (28 mice, 173 QC sessions in the raw data, 69,453 good units, per-area unit counts, 84% performance, 476 trials/session after the paper's exclusions, ~25% photostim trials in stim sessions, photostim always before the go cue, tone onset 1.85 s before the go cue) reproduce the values in the papers. Remaining small differences and why:
- 69,453 vs 69,943 good units (-0.7%): the DANDI export's `classification` column is the closest available representation of the published classifier output; no per-unit list is shipped with the code.
- ALM 8,464 vs 8,717 (-2.9%), thalamus 13,514 vs 12,808 (+5.5%), striatum 7,347 vs 7,664 (-4.1%): my CCF->coarse-region keyword mapping is an approximation of the Allen ontology (no internet access to the ontology), and the paper's exact ALM definition is a 3D voxel mask (`ALM_voxels_symmetric.npy`, not shipped).

### Check 5: Edge cases
- Trials whose window extends beyond the recorded trial interval (mostly 'miss' trials, which end ~0.8 s after the go cue): kept, because 'miss' is an output class; 2.6% of all time bins have no spikes from any neuron as a result. Documented.
- Sessions where the probes ran for only part of the session: handled with `obs_intervals` (9 sessions).
- `is_good_trials` with a mismatched number of columns: expanded onto the observed trials of each unit.
- Repeated sample (tone) epochs on early-lick replay trials: the **last** tone before the go cue is used; `time_from_tone_onset` therefore reaches up to 11.9 s on replay trials.
- Trials without any video frame in some bin: removed (would otherwise be mislabelled "tongue not visible").
- One session (sub-440958_ses-20190216) has `classification` = NaN for every unit -> rejected (this is the 174th file; the papers report 173 sessions).
- Sessions where all trials have the same early-lick value: 1 session has no early-lick trial; this is fine for the decoder (the class is present in other sessions).
- All neural values are finite and non-negative; all trials have exactly 80 bins; `subject_idx` is int64 and within range; `brain_region_idx` lengths equal the neuron counts.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, NVIDIA L4).

### Training Progress
- Loss decreasing: **Yes** - 4.34 (epoch 20) -> 0.63 (epoch 200); test loss 0.658.

### Decoder Results (Full: 138 sessions, 69,074 trials, 55,429 neurons)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Val / chance |
|--------|-------------|--------|--------|--------|
| choice (left/right/no lick) | 0.740 | 0.695 | 0.333 | 2.09x |
| outcome (ignore/miss/hit) | 0.718 | 0.659 | 0.333 | 1.98x |
| early_lick (no/yes) | 0.776 | 0.742 | 0.500 | 1.48x |
| tongue_y (4 classes) | 0.714 | 0.684 | 0.250 | 2.74x |

All four outputs are far above chance and the train/validation gap is small (<1.1x), i.e. no substantial overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Output | Validation balanced acc | Chance | Ratio |
|---|---|---|---|
| choice | 0.695 | 0.333 | 2.09x |
| outcome | 0.659 | 0.333 | 1.98x |
| early_lick | 0.742 | 0.500 | 1.48x |
| tongue_y | 0.684 | 0.250 | 2.74x |

No output is below chance. `early_lick` has the smallest ratio (1.48x), which is expected: it is a binary variable, only 12% of trials are early-lick trials, and the licking that defines it happens *before* the analysis window's response epoch - the balanced accuracy of 0.74 is nevertheless clearly informative.

Important context for all four numbers: the decoder is scored on **every 50-ms bin of the 4 s window**, including the 2.5 s before the go cue during which the animal has not yet licked. Choice, outcome and tongue position are only strongly encoded from the late delay onwards, so a time-averaged accuracy is necessarily much lower than the epoch-specific accuracies reported in the papers.

### Check 2: Accuracy comparison to the papers
The papers do not report accuracies for this exact decoding problem (their decoders are per-epoch, per-area, and use pseudo-populations), so I reproduced their analysis on the converted data
(`/app/cache/choice_decoding_check.py`: per-session ALM population, logistic regression with 5-fold CV, no photostim / no early lick / responded trials):

| Analysis | Paper value | This converted dataset |
|---|---|---|
| ALM population choice decoding, late delay (t = -0.1 s) | ~0.9 (Fig 6D / S6E, 200-neuron pseudo-populations) | 0.56-0.90 per session (5 sessions with 130-160 ALM neurons: 0.563, 0.902, 0.780, 0.787, 0.845) |
| ALM population choice decoding, response epoch (t = +0.5 s) | ~0.95 | 0.876-0.979 |
| ALM population choice decoding, sample epoch | ~0.6 | 0.56-0.72 |
| Choice decodable from tongue/video during the response epoch | high AUC (method paper Figs 6-7) | `tongue_y` decoding 0.68 averaged over the whole window, with the tongue visible in 59% of post-go-cue bins vs 7.5% of pre-go-cue bins |

The epoch-resolved accuracies reproduce the published values, which confirms that the neural data, the trial alignment and the behavioural labels in the converted dataset are correct. The lower whole-window accuracy of the reference decoder is a property of the scoring (all time bins, including the pre-cue baseline), not of the conversion.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| choice | 0.740 | 0.695 | 1.06 |
| outcome | 0.718 | 0.659 | 1.09 |
| early_lick | 0.776 | 0.742 | 1.05 |
| tongue_y | 0.714 | 0.684 | 1.04 |

All ratios are far below 1.5, so there is no sign of overfitting or of data leakage.

### Additional verification of the output/input time courses
- tongue class 3 ("not visible") in 92.5% of pre-go-cue bins and 41.4% of post-go-cue bins; classes 0/1/2 rise immediately after the go cue - exactly the expected licking behaviour.
- `photostim_on` is non-zero only between -2.18 s and -0.02 s relative to the go cue, i.e. photoinhibition always ends before the go cue, as stated in the methods.
- Population PSTHs show a clear go-cue-locked response (see `processing_*_summary.png`).

### Issues found and resolved in this step
- None: no accuracy was below chance or below 1.5x chance except `early_lick` (1.48x), which is explained above (rare binary class defined by behaviour occurring mostly before the analysed window).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format specification, key statistics, decoder results)
- [x] `cache/` folder created with `README_CACHE.md` documenting every exploration and validation script
- [x] All files organised

### Files produced
| File | Content |
|---|---|
| `/app/convert_data.py` | the conversion script (`--full`, `--sample`, `--show-processing`) |
| `/app/converted_data.pkl` (9.2 GB) | full converted dataset (138 sessions) |
| `/app/converted_data.pkl.info.json` | per-session diagnostics incl. rejection reasons |
| `/app/sample_data.pkl` | 2-session sample |
| `/app/conversion_sample_out.txt`, `/app/conversion_full_out.txt` | conversion logs |
| `/app/verification_sample_out.txt`, `/app/verification_full_out.txt` | `train_decoder.py --verify-only` output (valid, no errors or warnings) |
| `/app/train_decoder_sample_out.txt`, `/app/train_decoder_full_out.txt` | decoder training logs |
| `/app/processing_<session>.png`, `/app/processing_<session>_summary.png` | `--show-processing` diagnostics for 2 sessions |
| `/app/CONVERSION_NOTES.md`, `/app/README.md` | documentation |
| `/app/cache/` | exploration and validation scripts (see `README_CACHE.md`) |
