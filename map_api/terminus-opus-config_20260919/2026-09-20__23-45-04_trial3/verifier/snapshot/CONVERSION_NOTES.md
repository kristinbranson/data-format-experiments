# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (Chen et al.); NWB files in /app/data
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, pynwb 4.1.0.

Directory contents of /app:
- `CONVERSION_NOTES.md` (this file)
- `ChenLiuEtAl2023_SpikeSortingQC.pdf` - spike sorting/QC white paper
- `datapaper.pdf` - Chen et al., Brain-wide neural activity underlying memory-guided movement
- `methodpaper.pdf` - Wang, Kurgyis et al. 2025, Brain-wide analysis reveals movement encoding...
- `methods.txt` - excerpted methods
- `code/` - reference repo (VideoAnalysisUtils, Sherlock, Notebooks, Archive)
- `data/` - 29 subject folders (`sub-XXXXXX`), 174 `.nwb` files, 50 GB total
- `decoder.py`, `train_decoder.py` - decoder reference code
- `pynwb_docs/`, `Dockerfile`, `docker-compose.yaml`, `.manifest`

Data files are NWB (`sub-<id>_ses-<datetime>_behavior+ecephys+ogen.nwb`).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_all_sess_parallel(bw, stride, begin_time, end_time, ...)` | Sherlock/preprocess_all_ephys.py -> VideoAnalysisUtils/preprocessing_DJ_2022Aug.py | LOADING | Top-level driver. Called with `bw=0.04 s`, `stride=0.0034 s`, `begin_time=-3`, `end_time=3`, `qc_mode='classifier'` |
| `process_one_sess` | preprocessing_DJ_2022Aug.py | LOADING/CURATION | Concatenates all probes of a session, loads behavior/task variables, reads QC 'goodunits' list, splits neurons by region x hemisphere |
| `sliding_histogram(spikeTimes, begin_time, end_time, bin_width, stride, rate=True)` | preprocessing_DJ_2022Aug.py | PROCESSING | Counts spikes in sliding bins of width `bw` strided by `stride`, bin *centers* from begin_time to end_time; divides by bin width to give firing **rates** (Hz) |
| `process_one_area` | preprocessing_DJ_2022Aug.py | PROCESSING | Truncates spike times to [begin_time, end_time] (already go-cue aligned) and bins them -> `fr` (n_bins, n_trials, n_neurons) |
| `helper_get_neuron_id_area` | preprocessing_DJ_2022Aug.py | CURATION | Intersects QC-good unit ids with hemisphere mask (CCF ML midline = 5700 um) and requires a non-empty CCF annotation |
| `get_regular_trial_mask(ephys_data)` | population_decoding_utils.py, functions_for_r2.py, Sherlock/generate_single_train_test_split.py | CURATION | Trial mask: `early_lick==0 & auto_water==0 & free_water==0 & correctness!=-1 & stimulation[:,0]==0` |
| `load_session` | population_decoding_utils.py | LOADING | Loads per-area pickles of a session and concatenates neurons along axis 2 |
| `align_markers_between_lims(marker_data, go_times, t_min=-3, t_max=1.5)` | Sherlock/align_markers.py | PROCESSING | Aligns DeepLabCut video markers (`nose/tongue/jaw/whisker _x/_y` from `tracking.camera_0_side`) to the go cue; video frame period `dt = 0.0034 s` (300 Hz); frame times = frame_index*dt - go_cue_time; within each output bin takes the **last** frame in [t-dt, t) |
| `get_bad_trial_inds` | Sherlock/align_markers.py | CURATION | Flags trials where the number of video frames is inconsistent with the trial end time |

### Notes
- The reference code operates on **DataJoint .mat exports**, not on NWB. No `pynwb` usage anywhere in `/app/code`. I must therefore map each .mat field to the equivalent NWB field (Step 2/4).
- Neural processing chain: spike times per trial relative to **go cue** -> truncate to window -> sliding-window spike counts -> divide by bin width -> **firing rate in Hz**.
- The reference used overlapping bins (bw=40 ms, stride=3.4 ms) to match the 300 Hz video rate. The decoder task here specifies **50 ms bins**, so I will use non-overlapping 50 ms bins (bw = stride = 50 ms) - a required deviation.
- Reference window was -3 to 3 s about the go cue; the decoder task specifies **-2.5 to +1.5 s**, another required deviation.
- Reference epochs relative to the go cue (functions_for_r2.py): sample [-1.85, -1.2] s, delay [-1.2, 0] s, response [0, 1.5] s. So the tone (sample) onset is ~1.85 s before the go cue in a standard trial (sample 0.65 s + delay 1.2 s).
- Reference trial curation drops early-lick, auto-water, free-water, no-response and photostim trials. The present decoder task **requires** photostim as an input and early-lick / no-lick / ignore as outputs, so those exclusions cannot be applied here (documented deviation); I keep the auto-water/free-water exclusion because those trials are not the animal's own choice.
- Neuron curation in the reference relies on pre-computed classifier 'goodunits' files. In the NWB release the equivalent information must come from the units table (Step 2).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/sub-<subject_id>/sub-<id>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys+ogen.nwb`
- 174 NWB files, 28 subjects, 50 GB. DANDI:000363 (Mesoscale Activity Map dataset, Chen et al. 2023).
- Loaded exclusively with `pynwb.NWBHDF5IO`.

Per-file contents (verified on several files):
- `nwb.identifier`: e.g. `SC015_20190207_120657_s1` -> mouse name, date, time, session number (same naming as the reference .mat exports `map-export_SC015_20190207_120657_s1_p1.mat`).
- `nwb.subject`: `subject_id` (numeric, e.g. 440956), `description` (mouse name, e.g. SC015), sex, DOB.
- `nwb.intervals['trials']` (TimeIntervals, one row per trial): `start_time`, `stop_time`, `trial`, `photostim_onset` (str, s from trial start, or 'N/A'), `photostim_power` (str, mW), `photostim_duration` (str, s), `trial_uid`, `task` (all 'audio delay'), `task_protocol`, `trial_instruction` ('left'/'right'), `early_lick` ('early'/'no early'), `outcome` ('hit'/'miss'/'ignore'), `auto_water` (0/1), `free_water` (0/1).
- `nwb.acquisition['BehavioralEvents']`: `presample_*`, `sample_start/stop_times`, `delay_start/stop_times`, `go_start/stop_times`, `trialend_*`, `left_lick_times`, `right_lick_times`, `photostim_start/stop_times` (data = power in mW).
  - `go_start_times` has exactly one entry per trial in **all 174 sessions** (verified).
  - `sample_*`/`delay_*` can have **more** entries than trials: early-lick trials trigger a **replay** of the sample/delay epoch, so several sample epochs can occur in one trial. The **last** sample onset before the go cue is the tone onset that actually preceded that go cue (go - last sample onset = 1.85 s = 0.65 s sample + 1.2 s delay in standard trials).
- `nwb.acquisition['BehavioralTimeSeries']`: `Camera0_side_{Nose,Jaw,Tongue}Tracking`, each (n_frames, 3) = (x, y, likelihood) with explicit `timestamps` in session time, frame period 0.0034 s (300 Hz), identical to the reference `dt=0.0034`. Frames exist **only inside trials** (~71-76% of session time). All 174 sessions have tongue tracking.
- `nwb.units` (n=1000-3000 per session): QC metrics (`unit_amp`, `unit_snr`, `isi_violation`, `avg_firing_rate`, `presence_ratio`, `amplitude_cutoff`, `drift_metric`, ...), `unit_quality` ('good'/'multi'), **`classification`** ('good'/'unlabelled' = output of the region-specific QC classifiers described in the methods / white paper), **`anno_name`** (CCF annotation of the unit), `is_good_trials` (bool per trial), `spike_times` (session time), `obs_intervals` (== trial start/stop), `electrodes` (1 electrode per unit).
- `nwb.electrodes`: `location` = JSON string with the probe insertion target (`brain_regions` = '{left,right} {ALM,Striatum,Thalamus,Midbrain,Medulla,ECT,BLA}'), plus CCF `x` (ML, midline 5700 um), `y` (DV), `z` (AP; bregma at 5400 um, so AP_mm = (5400 - z)/1000).

**Critical structural facts**
1. Spike times exist **only within trial intervals** (0 spikes in inter-trial gaps; `obs_intervals` == trial start/stop). Verified: 210,882 spikes inside trials, 0 outside, for 30 good units of one session.
2. Trials start ~3.2 s before the go cue (min 3.12 s) and end 1.28-1.78 s after it, so the requested window go-2.5 .. go+1.5 s is fully observed for **81.6%** of trials. Coverage by outcome: hit 97.2%, ignore 94.4%, **miss 4.9%** (miss trials end ~0.95 s after the go cue, because the trial ends 1.5 s after the last lick / immediately if no lick and no reward consumption).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files / sessions | 174 (173 with >0 good units) |
| Units (all) | 272,227 |
| Units, `classification == 'good'` | 69,453 |
| Good units / session | mean 399, range 0-923 |
| Subjects | 28 |
| Sessions / subject | 3-10 (median 6) |
| Trials (total) | 94,990 |
| Trials / session | mean 546, range 264-800 |
| Outcome | hit 65,254 (68.7%), miss 15,641 (16.5%), ignore 14,095 (14.8%) |
| Early lick | early 10,805 (11.4%) |
| Instruction | right 48,913, left 46,077 |
| Photostim trials | 18,588 (19.6%), in 168/174 sessions |
| Auto-water trials | 1,339;  free-water 2,450 |
| Sessions with tongue tracking | 174/174 |

Session `SC017_20190216_162508_s4` (sub-441666) has **0 good units** -> dropped. Remaining **173 sessions**, exactly the number quoted in the methods.

### CCF region grouping
`anno_name` has 293 distinct values for good units. I wrote a keyword mapper (`region_map.coarse_region`, later folded into `convert_data.py`) onto the **same 14 coarse groups the reference uses** (ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum). Resulting good-unit counts (vs. paper):

| Region | Mine | Paper |
|--------|------|-------|
| ALM (motor ctx) | 7,885 | 8,717 |
| Striatum | 7,736 | 7,664 |
| Thalamus | 13,021 | 12,808 |
| Midbrain | 7,374 | 7,495 |
| Medulla | 2,856 | 2,928 |
| Orbital | 10,223 | - |
| OtherCortex | 8,385 | - |
| Olfactory | 4,136 | - |
| Hippocampus | 1,895 | - |
| CorticalSubplate | 1,888 | - |
| Cerebellum | 1,823 | - |
| Pallidum | 1,092 | - |
| Hypothalamus | 602 | - |
| Pons | 537 | - |
| **Total** | **69,453** | **69,943** |

All 293 annotations are mapped (0 unknown).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt / datapaper) |
| Good units by area | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | methods.txt |
| Sessions | 173 | "173 behavioral sessions, from which 655 probe insertions were made" (methods.txt); "660 penetrations, 173 behavioral sessions, and 28 mice" (datapaper Fig 1J) |
| Subjects | 28 | "This study is based on data from 28 mice" (methodpaper Methods) |
| Trials / session | mean 476, range 130-785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84%, range 65-99% | "with 84% correct rate (range, 65-99%)" |
| Session selection | performance > 65% and >= 50 correct lick-left and lick-right trials | methods.txt |
| Photostim trials | ~25% randomly interleaved, 17 VGAT-ChR2 mice | methods.txt |
| Photostim timing | last 0.5 s of delay incl. 100 ms ramp-down; "photoinhibition always ended before the Go cue" | methods.txt |
| Task epochs | sample 0.65 s (3x150 ms tone + 100 ms gaps), delay 1.2 s, go cue 0.1 s, answer period 1.5 s | methods.txt |
| Neural data time bin | 40 ms width, 3.4 ms stride | "we binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" (methodpaper) |
| Behavior/video time bin | 300 Hz (3.4 ms) | methods.txt, methodpaper |
| Spike sorting QC | 15 metrics -> 5 region-specific logistic-regression classifiers -> 'good' units | methods.txt + white paper |

Measured in the NWB files (Step 2) for comparison: 69,453 good units / 174 files (173 with >0 good units) / 28 subjects; response trials (hit+miss) per session mean 465, range 104-785 (paper: 476, 130-785); control-trial performance hit/(hit+miss) = 81.0% mean, range 54-97% (paper: 84%, 65-99%); photostim on 19.6% of trials in 168 sessions (paper "~25% of trials").

### Processing Details
- **Alignment**: everything is aligned to the **go cue**. In the reference .mat export spike times were *already* go-cue-aligned; in NWB they are in session time so I subtract `go_start_times[trial]`.
- **Binning**: reference used sliding windows (bw 40 ms, stride 3.4 ms) so the neural sampling grid matched the 300 Hz video. The decoder task here mandates **50 ms bins**, so I use non-overlapping 50 ms bins (spike counts / 0.05 s = rate in Hz).
- **Window**: reference -3 .. +3 s (ephys) and -3 .. +1.5 s (video markers). Task mandates **-2.5 .. +1.5 s**.
- **Epoch structure relative to go cue**: sample [-1.85, -1.2] s, delay [-1.2, 0] s, response [0, 1.5] s (`functions_for_r2.py`). Verified in NWB: go - (last sample onset) = 1.85 s exactly for non-early-lick trials.
- **Video**: DeepLabCut markers (tongue/jaw/nose x, y + likelihood) at 300 Hz; reference assigns to each output time bin the **last** frame falling in that bin.

### Curation Steps

**Neuron curation rules**:
- Reference: only units on the classifier-derived 'goodunits' lists, which must also have a CCF annotation and fall in one of 14 region x 2 hemisphere groups. In NWB this is `units.classification == 'good'` (+ non-empty `anno_name`, which holds for 100% of good units).
- Method-paper R2 analyses additionally dropped neurons with firing rate < 2 Hz; that is specific to their regression analysis (not a data-quality criterion) so it is **not** applied here (dropping low-rate neurons would throw away decodable signal).

**Trial curation rules**:
- Reference `get_regular_trial_mask`: early_lick == 0, auto_water == 0, free_water == 0, outcome != no-response, no photostim.
- For this decoder task, early lick, outcome (incl. ignore/no-response) and photostim are **required** inputs/outputs, so those three exclusions must be dropped (documented deviation). Auto-water and free-water trials are excluded (as in the reference) because the animal receives reward independent of choice, making choice/outcome labels not reflect the animal's decision.

### Decoders Trained (reference accuracies)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Choice from **video** (pre-sample) | embedding decoder | AUC 0.51 +- 0.06 (n=106 sessions) |
| Choice from **video** (sample+delay) | embedding decoder | AUC 0.66 +- 0.12 |
| Choice from **video** (late response epoch) | embedding decoder | AUC 0.99 +- 0.01 |
| Choice from **neural population** (ALM, late delay) | logistic regression on pseudo-populations, 200 ms causal windows | ~0.8-0.9 accuracy (datapaper Fig 6D, S6) |
| Single-neuron choice modulation | logistic regression | AUC > 0.65 threshold |

Expectation for this task: choice and outcome should be decodable well above chance from population activity (especially after the go cue, when directional licking occurs); tongue y-position is strongly reflected in the video-driven neural activity (method paper: much of neural variance is movement-related), so it should also be well above chance.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Mapping of reference (.mat/DataJoint) fields to NWB fields
| Reference .mat field | NWB equivalent | Verified |
|---|---|---|
| `task_cue_time[0,:]` (go cue) | `acquisition['BehavioralEvents'].go_start_times.timestamps` | yes: exactly 1 per trial in all 174 sessions |
| `task_sample_time` | `sample_start/stop_times` | yes: 0.65 s duration; **more events than trials** because early-lick trials replay the sample epoch |
| `task_delay_time` | `delay_start/stop_times` | yes: 1.2 s in standard trials |
| `behavior_early_report` | `trials.early_lick` ('early'/'no early') | yes |
| `behavior_report` (1 correct / 0 error / -1 no response) | `trials.outcome` ('hit'/'miss'/'ignore') | yes |
| `behavior_is_auto_water`, `behavior_is_free_water` | `trials.auto_water`, `trials.free_water` | yes |
| `behavior_lick_times`, `behavior_lick_directions` | `left_lick_times`, `right_lick_times` | yes: lick-derived choice agrees 100% with outcome x instruction |
| `task_trial_type` ('l'/'r') | `trials.trial_instruction` | yes |
| `task_stimulation` [power, type, on, off] | `trials.photostim_onset/power/duration` (strings, rel. trial start) and `photostim_start/stop_times` | yes: table onset == event time - trial start (exact) |
| spike times per trial (go-cue aligned) | `units.spike_times` (session time) - subtract go cue | yes |
| `neuron_unit_quality_control` metrics | `units.<metric>` columns | yes |
| `goodunits` classifier lists | `units.classification == 'good'` | yes: totals match the paper per area |
| `histology.annotation` (CCF label) | `units.anno_name` | yes |
| `histology.ccf_x/y/z` | `electrodes.x/y/z` via `units.electrodes` (1 electrode per unit) | yes |
| `tracking.camera_0_side.{tongue,jaw,nose}_{x,y}` | `acquisition['BehavioralTimeSeries'].Camera0_side_*Tracking` (x, y, likelihood) | yes, 300 Hz |
| `trial_end_time` | `trials.stop_time` | yes |

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Number of sessions | n/a | 174 NWB files | 173 behavioral sessions | One file (`SC017_20190216_162508_s4`) has **0 good units**; excluding it gives 173. Confirmed independently by the probe-count distribution: data {2 probes: 2 sessions, 3: 53, 4: 99, 5: 20}; paper {2: 2, 3: 53, 4: 98, 5: 20}; the dropped session has 4 probes -> exact match. |
| Probe insertions | n/a | 659 electrode groups | 655 (methods) / 660 (Fig 1J) | Consistent within the papers' own spread; no action. |
| Good units | classifier lists | 69,453 | 69,943 | 0.7% difference, consistent with a slightly different dandiset version / the 0-good-unit session. Per-area counts match well (ALM 7,885 vs 8,717; striatum 7,736 vs 7,664; thalamus 13,021 vs 12,808; midbrain 7,374 vs 7,495; medulla 2,856 vs 2,928), so the QC field and region mapping are correct. |
| Trials / session | n/a | 546 (all), 465 response trials | 476 | The paper's "trials performed" = trials where the mouse responded (hit+miss): data mean 465, range 104-785 vs paper 476, range 130-785. Consistent. |
| Correct rate | n/a | 81.0% (control trials, hit/(hit+miss)) | 84% (range 65-99%) | Paper additionally applied session-selection criteria (perf > 65%, >= 50 correct each direction); 152/174 sessions pass those. Restricting to them raises the mean. The DANDI release includes all 173 sessions, so I keep them all (see Step 5 decision). |
| Photostim fraction | `stimulation[:,0]==0` for control | 19.6% of trials, 168/174 sessions | "~25% randomly interleaved trials, N = 17 VGAT-ChR2 mice" | The ~25% applies to the photostim-mouse subset/sessions; averaged over all sessions (incl. 6 with none) 19.6% is consistent. |
| Trial exclusions | early lick / auto water / free water / no response / photostim all excluded | - | same | Cannot exclude early-lick, ignore (no-response), or photostim trials here: they are explicitly required decoder outputs/inputs. Auto-water and free-water trials **are** excluded (reward independent of choice). |
| Bin size | 40 ms bw / 3.4 ms stride | - | same | Task mandates 50 ms bins -> non-overlapping 50 ms bins. |
| Window | -3 .. +3 s | trials cover ~-3.2 .. +1.3-1.8 s about the go cue | -3 .. +1.5 s for video | Task mandates -2.5 .. +1.5 s. Data are observed only inside trials, so late bins of short trials (mostly 'miss') are unobserved -> zero-filled (see Step 5). |

After these resolutions the reference code, the NWB data, and the papers are mutually consistent.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial window / binning
- Alignment event: **go cue onset** = `BehavioralEvents.go_start_times.timestamps[trial]` (1 per trial, verified in all sessions).
- Window: **-2.5 s to +1.5 s** -> 80 non-overlapping bins of **50 ms**; bin i covers [go-2.5+0.05i, go-2.5+0.05(i+1)), centre at -2.475+0.05i.

### Variable Mapping
| Source Variable (NWB) | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units.spike_times` (session time), `units.classification=='good'` | `neural` | subtract go cue, count spikes in each 50 ms bin, divide by 0.05 -> **rate in Hz**, float32 | `sliding_histogram`, `process_one_area` | Reference used bw=40 ms/stride=3.4 ms; 50 ms non-overlapping is mandated by the task |
| `sample_start_times` (last one before the go cue) | `input[0]` = `time_from_tone_onset` | bin-centre time minus tone onset, in seconds (signed, continuous) | - | Early-lick trials replay the sample epoch; the **last** sample onset before the go cue is the tone that preceded that go cue. Verified: go - last sample onset = 1.85 s in standard trials |
| `photostim_start_times`/`photostim_stop_times` (cross-checked with `trials.photostim_onset/duration`) | `input[1]` = `photostim` | 1 if the stim interval overlaps the bin, else 0 | `stimulation` columns in `preprocessing_DJ_2022Aug` | Photoinhibition is in the last 0.5 s of the delay, i.e. ~[-1.21,-0.71] s re. go cue |
| `left_lick_times` / `right_lick_times` after the go cue | `output[0]` = `choice` | 0=left, 1=right, 2=no lick (direction of the **first** lick after the go cue) | `behavior_lick_times/directions` | Verified to agree 100% with `outcome` x `trial_instruction` |
| `trials.outcome` | `output[1]` = `outcome` | 0=ignore, 1=miss, 2=hit | `behavior_report` (-1/0/1) | Same 3 classes, same order as the -1/0/1 coding of the reference |
| `trials.early_lick` | `output[2]` = `early_lick` | 0='no early', 1='early' | `behavior_early_report` | |
| `Camera0_side_TongueTracking` (x, y, likelihood) @300 Hz | `output[3]` = `tongue_y_position` | per bin: mean y over frames with likelihood > 0.9; class 0 = < 40th pct, 1 = 40-60th pct, 2 = > 60th pct of the session's binned visible y values, 3 = no visible frame in the bin | `align_markers_between_lims` | Percentiles computed **per session** as specified |
| `units.anno_name` + `electrodes.x` (ML, midline 5,700 um) | `brain_region_idx` / `brain_regions` | CCF annotation -> one of the reference's 14 coarse groups, prefixed with hemisphere: e.g. `left ALM` | `helper_get_neuron_id_area` (side split at ML 5700), region list in `process_one_sess` | |
| `subject.subject_id` | `subjects`, `subject_idx` | | | 28 mice |

Outputs are stored as a (4, 80) int array per trial: choice / outcome / early lick are constant across time (they are per-trial variables, as specified), tongue-y class is time-varying.

### Key Decisions
1. **Neuron curation = `classification == 'good'`**: this column is the output of the region-specific QC classifiers described in the methods and white paper, i.e. exactly the `goodunits` lists the reference code loads. Per-area totals reproduce the paper's (see Step 2), confirming the identification. Units without a CCF annotation would be dropped (none exist among good units).
2. **No firing-rate threshold**: the method paper's 2 Hz cut is specific to their video->activity regression, not a quality criterion; removing low-rate neurons would discard decodable signal.
3. **Trial curation**: drop `auto_water` or `free_water` trials (1,339 + 2,450 trials), as in the reference `get_regular_trial_mask`, because on those trials reward is delivered independently of the animal's choice, so choice/outcome labels do not reflect a decision. **Keep** early-lick, ignore (no-response) and photostim trials, which the reference excluded, because the decoder task explicitly requires early lick and outcome=ignore as outputs and photostimulation as an input.
4. **Session curation**: drop sessions with 0 good units (1 session: `SC017_20190216_162508_s4`) -> 173 sessions, matching the paper. No further session selection: the DANDI release already corresponds to the paper's 173 selected sessions (probe-count distribution matches exactly).
5. **Partially observed windows**: spikes and video frames exist only inside trial intervals (`obs_intervals` == trial start/stop; 0 spikes in the ITI). Trials begin ~3.2 s before the go cue (so -2.5 s is covered in 96.9% of trials) but end 1.3-1.8 s after it, so +1.5 s is covered in only 84.6% of trials, and in only 4.9% of **miss** trials (miss trials terminate early). Excluding partially observed trials would delete the entire `miss` output class, so all trials are kept and **unobserved bins are zero-filled** (no spikes recorded -> rate 0; tongue class 3 = not visible). This is recorded in `metadata`.
6. **Tongue visibility threshold 0.9** on the DLC likelihood: the likelihood distribution is strongly bimodal (frac > 0.5 = 0.1059, > 0.9 = 0.1052, > 0.99 = 0.1045 in an example session), so the result is insensitive to the exact threshold.
7. **Per-bin aggregation of the tongue marker**: the reference took the *last* video frame in each (3.4 ms) bin. With 50 ms bins there are ~15 frames per bin, so I average the visible frames in the bin, which is less noisy and keeps the 'visible' information; percentiles are computed on the same binned quantity so the class proportions are exactly 40/20/40 among visible bins.
8. **Hemisphere-resolved region names** (`left ALM`, `right Thalamus`, ...) following the reference's `side_region` areas; hemisphere from CCF ML with the reference's midline of 5,700 um.

### Planned Sanity Checks
- [ ] 173 sessions, 28 subjects, 69,453-ish good units after conversion; per-region unit counts close to the paper's (ALM ~8.7k, striatum ~7.7k, thalamus ~12.8k, midbrain ~7.5k, medulla ~2.9k).
- [ ] Trial count = 94,990 - (auto_water | free_water) trials.
- [ ] Output distributions: hit ~69%, miss ~16%, ignore ~15%; early lick ~11%; choice left ~= choice right; tongue classes ~40/20/40 of visible bins.
- [ ] `time_from_tone_onset` at the go-cue bin == 1.85 s for standard trials.
- [ ] Photostim input is 1 only in [-1.25, -0.65] s re. go cue, in ~19.6% of trials, 0 in sessions/trials with `photostim_onset == 'N/A'`.
- [ ] Independent re-computation (loading the NWB file separately) of: a spike count for a specific (session, trial, neuron, bin); the tongue class for a specific (trial, bin); the choice/outcome/early-lick for specific trials -> `np.allclose`.
- [ ] Population PSTH aligned to the go cue shows the expected response-epoch increase; tongue-visible fraction peaks after the go cue.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the plan of Step 5.

Structure:
- `coarse_region(anno)` - CCF annotation -> one of the reference's 14 coarse region groups.
- `process_session(path, make_plots)` - all per-session work, opened once with `pynwb.NWBHDF5IO`:
  1. read the trials table, behavioural events, units and electrodes;
  2. trial mask (no auto/free water, go cue inside trial, ephys coverage);
  3. unit mask (`classification == 'good'` and a mappable CCF annotation), hemisphere from `electrodes.x` vs the reference ML midline of 5,700 um;
  4. neural: `np.searchsorted` of each unit's spike times against the (n_trials x 81) matrix of bin edges, clipped to the trial interval -> counts -> /0.05 s;
  5. inputs: time from the last sample (tone) onset preceding the go cue; photostim overlap per bin;
  6. outputs: choice from first lick after the go cue, outcome, early lick, and tongue-y class from the 300 Hz DLC trace using cumulative sums;
- `plot_session` - `--show-processing` figures;
- `main` - multiprocessing pool over sessions, assembly, summary statistics, pickle.

Efficiency notes:
- Every operation is vectorised over trials; the only python loop over units is the spike-time lookup (unavoidable, ragged storage), which uses one `searchsorted` per unit for all trials at once.
- Tongue binning uses cumulative sums of (y, visible) so all (trial x bin) means are computed with two `searchsorted` calls.
- Sessions are processed in parallel (default 16 workers; each worker opens one NWB file).
- Timing per stage is recorded in the returned dict and reported.

Bugs found and fixed during development (see Step 7):
1. `photostim` array was float and used `|=` -> switched to a boolean accumulator.
2. **Trials without ephys coverage**: `units.obs_intervals` shows that in 8/173 sessions the recording stops before the behavioural session ends; those trials were being emitted as all-zero firing rates. Now the trial mask requires every good unit to have an `obs_intervals` entry for the trial.
3. **One trailing trial with zero spikes**: `obs_intervals` may list one final trial in which the recording had already stopped (verified: last spike 1107.44 s, trial 159 spans 1109.4-1114.5 s and contains 0 spikes from 375 units). Trials with zero spikes across all simultaneously recorded good units are now dropped.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing 2>&1 | tee /app/conversion_sample_out.txt
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only 2>&1 | tee /app/verification_sample_out.txt
```

### Sample Statistics (2 sessions of mouse SC015 / 440956)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 834 (459 + 375) |
| Neurons / session | 417 mean |
| Subjects | 1 |
| Trials (total) | 513 (354 + 159) |
| Trials / session | 354, 159 (the second session's ephys stops after 160 trials) |
| Timepoints / trial | 80 (50 ms bins, -2.5..+1.5 s) |
| time_from_tone_onset range | [-0.625, 5.722] s |
| photostim range | [0, 1]; 23.4% of trials, 3.2% of bins |
| choice distribution | left 0.456, right 0.407, no lick 0.136 |
| outcome distribution | ignore 0.136, miss 0.298, hit 0.565 |
| early_lick distribution | no 0.942, yes 0.058 |
| tongue_y_position distribution | 0.082 / 0.041 / 0.082 / 0.796 (not visible) |

Numeric sanity checks on the sample pickle:
- Bin centres run -2.475 .. +1.475 s; the bin nearest the go cue is index 49 (centre -0.025 s) and `time_from_tone_onset` there equals **1.825 s = 1.85 - 0.025** for standard trials (larger for early-lick trials with replayed epochs). Correct.
- Photostim occupies 10-11 consecutive bins (0.50-0.55 s), matching the 0.5 s laser; onsets fall between -2.28 and -1.17 s relative to the go cue (the late values are the standard "last 0.5 s of the delay"; earlier ones occur on early-lick trials whose delay was replayed). Always **before** the go cue, as the methods state.
- Tongue classes among visible bins are 0.3999 / 0.2002 / 0.3999 - exactly the specified 40/20/40 split.
- Mean firing rate 5.4 Hz, max 360 Hz (plausible for 50 ms bins).

### Processing Plots Review
`processing_SC015_20190207_120657_s1.png`, `processing_SC015_20190208_133600_s2.png` (+ `_summary`): for one hit, one miss, one ignore and one photostim trial each they show (1) the raw spike raster relative to the go cue with the tone-onset and trial-end markers, (2) the 50 ms binned rates, (3) the two inputs with the laser interval shaded, (4) the raw 300 Hz tongue-y trace, the binned means and the session 40/60 percentiles, (5) the four outputs with the individual lick times overlaid. No misalignment: licks start right after the go cue, the tongue becomes visible exactly when licks occur, the photostim input covers the shaded laser interval, and rates are zero only after the trial-end marker.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| One `searchsorted` per unit for all trial x bin edges | ~100x vs per-trial loops |
| Cumulative-sum binning of the 300 Hz video | ~50x vs per-bin masks |
| 16-way multiprocessing over sessions | ~16x |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| Full session conversion (459 units, 368 trials) | ~2.8 s wall (single worker) | 174 sessions / 16 workers x ~5 s (larger sessions) ~= 1-2 min |

The largest sessions have ~2.3x more units and ~2x more trials than the sample ones, so even a 5x margin keeps the full conversion far below the 15-minute budget.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/sample_data.pkl 2>&1 | tee /app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None (after the two ephys-coverage fixes of Step 6)

### Training
Loss decreased monotonically: 1.138 (epoch 50) -> 0.930 -> 0.801 -> 0.719 -> 0.666 -> 0.632 -> ... -> 0.544 (epoch 200). Test loss 0.975.

### Decoder Results (Sample, 2 sessions, 513 trials; final code)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.744 | 0.621 | 0.333 |
| outcome | 0.767 | 0.650 | 0.333 |
| early_lick | 0.835 | 0.720 | 0.500 |
| tongue_y_position | 0.753 | 0.621 | 0.250 |

All four outputs are decoded well above chance (1.4-2.5x) already with only two sessions.
(The sample files were regenerated after the final two fixes - answer-period choice window and
provenance metadata - so `sample_data.pkl`, `conversion_sample_out.txt`,
`verification_sample_out.txt` and `train_decoder_sample_out.txt` all come from the final script.)

--------|-------------|--------|--------|
| choice | 0.738 | 0.616 | 0.333 |
| outcome | 0.765 | 0.642 | 0.333 |
| early_lick | 0.832 | 0.731 | 0.500 |
| tongue_y_position | 0.753 | 0.619 | 0.250 |

All four outputs are decoded well above chance (1.5-2.5x) already with only two sessions.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/converted_data.pkl --full --nproc 24 2>&1 | tee /app/conversion_full_out.txt
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only 2>&1 | tee /app/verification_full_out.txt
```

### Output Files
- `converted_data.pkl`: 11.89 GB, written in 24.3 s of conversion (24 workers) + 20.5 s of pickling.
- `verification_full_out.txt`: created; **"Data format is valid, no errors or warnings."**

### Conversion log highlights
- 173 sessions converted, 1 skipped (`sub-440958_ses-20190216T162508` = `SC017_20190216_162508_s4`, 0 good units), 0 errors.
- 1,060 trials dropped for lack of ephys coverage (8 sessions), 2 trials dropped for zero spikes across all units.
- 3,789 auto-water/free-water trials excluded by the trial mask.
- 0 trials needed the tone-onset fallback.

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (raw NWB) | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total good neurons | 69,943 | n/a (classifier lists) | 69,453 | 69,453 | yes (0.7% below the paper) |
| ALM / Striatum / Thalamus / Midbrain / Medulla neurons | 8,717 / 7,664 / 12,808 / 7,495 / 2,928 | 14 coarse groups | 7,885 / 7,736 / 13,021 / 7,374 / 2,856 | same | yes (+-10%) |
| Mean neurons/session | - | - | 399 (incl. 0-unit session) | 401.5 (range 90-923) | yes |
| Subjects | 28 | - | 28 | 28 | yes |
| Sessions | 173 | - | 174 files, 173 with good units | 173 | yes |
| Probe insertions | 655-660 | - | 659 (distribution 2:2, 3:53, 4:99, 5:20) | - | yes (dropping the 0-unit session gives exactly the paper's 2:2, 3:53, 4:98, 5:20) |
| Trials (total) | - | - | 94,990 | 89,544 | expected: -3,789 auto/free water, -1,062 no ephys, -620 in the dropped session (+ go-cue sanity) |
| Trials/session (mean) | 476 responded (130-785) | - | 546 all / 465 responded | 517.6 (159-796) | yes: 517.6 x (hit+miss fraction 0.851) = 441 responded trials |
| Correct rate | 84% (65-99%) | - | 81.0% control | hit/(hit+miss) = 0.684/0.851 = 80.4% | yes (paper applies extra session selection) |
| Photostim trials | ~25% | `stimulation[:,0]==0` for control | 19.6% | 20.0% | yes |
| Early lick | - | excluded by reference | 11.4% | 11.6% | yes |
| choice distribution | - | - | - | left 0.432 / right 0.422 / no lick 0.146 | balanced, as expected for the randomized task |
| outcome distribution | - | - | hit .687 / miss .165 / ignore .148 | hit 0.685 / miss 0.167 / ignore 0.148 | yes |
| tongue classes | - | - | - | 0.096 / 0.048 / 0.096 / 0.760 | exactly 40/20/40 among visible bins |
| time_from_tone_onset | tone 1.85 s before go cue | - | - | [-1.525, 11.894] s, 1.825 s at the go-cue bin | yes (long values = early-lick trials with replayed sample/delay epochs) |
| Bins per trial | - | - | - | 80 (50 ms, -2.5..+1.5 s) | as specified |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."** - no errors and no warnings remain.

Warnings that appeared in earlier iterations and how they were removed:
| Warning | Cause | Fix |
|---|---|---|
| `Session 1, trial N: all neural data is zero` (321 trials) | In 8/173 sessions the ephys recording stops before the behavioural session ends; `units.obs_intervals` covers only the first N trials | Trial mask now requires an `obs_intervals` entry for the trial in **every** good unit (1,060 trials dropped) |
| `Session 1, trial 159: all neural data is zero` (1 trial) | `obs_intervals` lists one extra trailing trial after the last recorded spike (last spike 1107.44 s, trial spans 1109.39-1114.47 s) | Trials with zero spikes summed over all good units are dropped (2 trials dataset-wide) |

### Check 2: Independent sanity checks
`/app/cache/sanity_checks.py` reloads the NWB files with `pynwb` and recomputes every stream with *different* code (explicit boolean masks rather than `searchsorted`/cumulative sums), then compares with `np.allclose`/`np.array_equal`. Run on sessions 0, 60, 120 and 172 (all four spanning different mice and years). Output: `/app/cache/sanity_checks_out.txt`, **0 checks failed**:

| Check | Result |
|---|---|
| go-cue times stored in metadata == `go_start_times` at the stored trial rows | PASS |
| n neurons per session == number of `classification == 'good'` units | PASS (e.g. 268 vs 268, 368 vs 368) |
| NEURAL: random (trial, neuron, bin) firing rates == (spikes in [edge_lo, edge_hi) clipped to the trial) / 0.05 | PASS (24 random cells) |
| NEURAL: total spike count in the whole 4 s window for a trial == sum(rates)*0.05 | PASS (e.g. 5,303 vs 5,303; 22,841 vs 22,841) |
| INPUT 0: `time_from_tone_onset` == (go - last sample onset before go) + bin centre | PASS |
| INPUT 1: photostim == bins overlapping a laser on/off interval, for **every** trial | PASS |
| INPUT 1: photostim trials == `trials.photostim_onset != 'N/A'` | PASS (e.g. 136 vs 136) |
| OUTPUT 1: outcome == `trials.outcome` | PASS |
| OUTPUT 2: early lick == `trials.early_lick` | PASS |
| OUTPUT 0/1/2 constant across time within a trial | PASS |
| OUTPUT 0: choice == direction of the first lick in the answer period | PASS |
| OUTPUT 0: choice consistent with outcome x instruction | PASS (0-3 mismatched trials per session, see below) |
| OUTPUT 3: tongue class == recomputed (mean y over visible frames vs session percentiles) | PASS (24 random cells) |
| OUTPUT 3: 40/20/40 split among visible bins | PASS (0.400/0.200/0.400) |
| brain-region hemisphere == CCF ML >= 5,700 um | PASS |
| brain-region ALM == motor-cortex annotations | PASS |

**Issue found and fixed by these checks**: the choice cross-check initially failed in one session. Investigation (`/app/cache/check_choice.py`) showed 4 'ignore' trials in which a lick was detected 19-41 ms after the go cue (continuation of ongoing licking) or at 1.75 s (after the answer period), which the behavioural state machine did not count as a response. I therefore restricted the choice lick search to the **1.5 s answer period** defined in the methods ("During the response epoch (answer period: 1.5 s) mice reported the instruction by licking one of the two lick ports"), which also matches the decoding window. Dataset-wide this reduced the disagreement between lick-derived choice and outcome x instruction from 526/94,990 (0.55%) to 318/94,990 (0.33%). The residual 0.33% are trials where the licks recorded in the NWB events differ from the behavioural report (e.g. a single lick that did not trigger the state machine); the lick-derived definition is the more direct measure of what the animal did and is what the reference code uses (`behavior_lick_times` / `behavior_lick_directions`).

### Check 3: Reference code comparison
| Step | Reference code | My code | Same? |
|---|---|---|---|
| (a) Data loading | `preprocessing_utils.loadmat` of DataJoint `.mat` exports; per-probe files concatenated per session | `pynwb.NWBHDF5IO`; the NWB file already contains all probes of a session | Equivalent (field-by-field mapping verified in Step 4) |
| (b) Neuron filtering | `goodunits` classifier lists x non-empty CCF annotation x hemisphere/region membership (`helper_get_neuron_id_area`) | `units.classification == 'good'` x mappable `anno_name` x hemisphere from `electrodes.x` (same 5,700 um midline) | Same. Per-region totals reproduce the paper |
| (c) Temporal alignment | spike times already relative to `task_cue_time` (go cue); markers aligned by subtracting `go_times` | subtract `go_start_times[trial]` from spike times and video timestamps | Same |
| (d) Binning | `sliding_histogram`, bw 40 ms, stride 3.4 ms, counts / bin width -> Hz | non-overlapping 50 ms bins (task requirement), counts / 0.05 s -> Hz | Same estimator, different bin width (required by the task) |
| (e) Input construction | reference stores `stimulation` [power, type, on, off] relative to the go cue; epoch times from `task_sample_time`/`task_delay_time` | photostim binary per bin from the laser on/off events; time from the tone (sample) onset preceding the go cue | Same source variables; the decoder task dictates the binary/continuous representation |
| (f) Output construction | `behavior_lick_times`/`lick_directions`, `behavior_report`, `behavior_early_report`, tongue marker from `tracking.camera_0_side` | same variables from the NWB equivalents; tongue y discretized per the task spec | Same, plus the task-mandated discretization |
| Trial curation | `get_regular_trial_mask`: early_lick==0, auto_water==0, free_water==0, correctness!=-1, no stim | auto_water==0 & free_water==0 & ephys coverage | Deliberate deviation: early lick / no-response / photostim are required decoder outputs and inputs, so they must be kept; documented in Step 5 |
| Window | -3 .. +3 s (ephys), -3 .. +1.5 s (video) | -2.5 .. +1.5 s | Task requirement |

### Check 4: Key statistics comparison
See the table in Step 9. Every statistic available in the papers is reproduced: 173 sessions, 28 mice, ~69.5k good units (paper 69.9k), per-region unit counts within 10%, probe-count distribution exactly equal after dropping the 0-unit session, 476-vs-441 responded trials per session, 84%-vs-80% correct rate, ~25%-vs-20% photostim trials.

### Check 5: Edge cases handled
- Sessions with 0 good units (1) -> skipped.
- Sessions/trials where the ephys recording stops early (8 sessions, 1,062 trials) -> dropped.
- Early-lick trials with **replayed** sample/delay epochs: the tone onset is taken as the last sample onset *before* the go cue (405 sample events for 368 trials in one example session), and photostim can then occur earlier than the canonical -1.2 s.
- Trials whose window extends beyond the trial end (mostly `miss`): bin edges are clipped to the trial interval so spikes/video from the neighbouring trial can never leak in; the unobserved bins are empty (rate 0, tongue class 3).
- Trials whose window starts before the trial start (3.1% of trials): same clipping.
- Sessions without any photostimulation (6) -> `photostim_start_times` absent or empty is handled.
- Video frames with low DLC likelihood -> excluded from the per-bin mean; a bin with no visible frame gets class 3.
- Sessions with fewer than 10 visible tongue bins would get all class 3 (no such session occurs).
- `units.electrodes` is a ragged `VectorIndex`; the unit -> electrode mapping is taken through `vi.target.data` (exactly one electrode per unit, verified).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`
(ran on the GPU, 200 epochs, ~7 min)

### Training Progress
- Loss decreasing: **Yes** - 9.89 (epoch 6) -> 1.85 (50) -> 0.869 (100) -> 0.703 (150) -> 0.658 (200).
- Test loss 0.6505, slightly **below** the final training loss -> no overfitting.

### Decoder Results (Full: 173 sessions, 89,544 trials, 69,453 neurons)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Val / chance |
|--------|-------------|--------|--------|--------|
| choice (left/right/no lick) | 0.7032 | 0.6769 | 0.3333 | 2.03x |
| outcome (ignore/miss/hit) | 0.6969 | 0.6638 | 0.3333 | 1.99x |
| early_lick (no/yes) | 0.7992 | 0.7522 | 0.5000 | 1.50x |
| tongue_y_position (4 classes) | 0.6848 | 0.6597 | 0.2500 | 2.64x |

Note that these are **per-time-point** balanced accuracies averaged over the whole -2.5 .. +1.5 s window, which includes the pre-tone baseline period where choice/outcome are not yet represented in the brain and therefore cannot be decoded above chance. See Step 12 for the time-resolved control analysis.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Variable | Classes | Chance | Validation balanced acc | Ratio |
|---|---|---|---|---|
| choice | 3 | 0.333 | 0.6769 | 2.03x |
| outcome | 3 | 0.333 | 0.6638 | 1.99x |
| early_lick | 2 | 0.500 | 0.7522 | 1.50x |
| tongue_y_position | 4 | 0.250 | 0.6597 | 2.64x |

No output is at or below chance. `early_lick` has the lowest ratio (1.50x) simply because it is a binary variable (chance 0.5); in absolute terms 0.75 balanced accuracy on an 11.6%/88.4% imbalanced variable is a strong result. Also note that early licking happens during the sample/delay epoch, so the evidence in the window is genuinely limited to the first ~2 s.

### Check 2: Accuracy comparison to the papers
The papers do not decode exactly these four variables from neural activity over a whole -2.5..1.5 s window, so a direct number-for-number comparison is not possible. The closest published quantities are the **time-resolved** population choice decoders. I therefore repeated that analysis on the converted data (`/app/cache/timeresolved_choice.py`: per 50 ms bin, PCA(50) + L2 logistic regression, 5-fold stratified CV, left vs right responded trials, 6 sessions spread over the dataset):

| Epoch | Converted data (balanced acc, 6 sessions) | Papers |
|---|---|---|
| pre-sample (< -1.85 s) | 0.540 | AUC 0.51 +- 0.06 for video-based choice prediction (methodpaper); "at chance" |
| sample (-1.85 .. -1.2 s) | 0.630 | AUC 0.66 +- 0.12 for sample+delay (methodpaper); choice information "developed in the sample epoch" (datapaper Fig 6D) |
| delay (-1.2 .. 0 s) | 0.634 | ramping choice information through the delay; ALM pseudo-population accuracy ~0.7-0.9 with 200 neurons (datapaper Fig 6D/S6) |
| response (0 .. 1.5 s) | 0.845 (max 0.91-0.94 per session) | "close to perfect performance (0.99 +- 0.01)" for video-based decoding after the go cue; datapaper neural decoders saturate high |

The converted data reproduce the published pattern: chance before the tone, a rise during sample/delay, and near-saturation in the response epoch. Individual sessions reach 0.91-0.94. Our per-session values are below the published pseudo-population numbers because the paper pools 200 neurons of one area across mice/sessions by hierarchical bootstrapping and uses 200 ms causal windows, whereas we decode single sessions in single 50 ms bins.

Why the whole-window numbers reported by `train_decoder.py` (0.66-0.75) are lower than the epoch-resolved peaks: the balanced accuracy is averaged over **all 80 time bins**, and ~30 of them precede the tone, where choice, outcome and early lick are not yet represented in the brain at all (0.54 measured above). Averaging chance-level early bins with 0.85-0.94 late bins yields ~0.68, exactly what the decoder reports. This is expected, not a bug.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| choice | 0.7032 | 0.6769 | 1.04 |
| outcome | 0.6969 | 0.6638 | 1.05 |
| early_lick | 0.7992 | 0.7522 | 1.06 |
| tongue_y_position | 0.6848 | 0.6597 | 1.04 |

All ratios are ~1.05 (far below the 1.5 threshold) and the test loss (0.6505) is below the final training loss (0.658): no overfitting and no data leakage.

### Additional diagnostics performed
1. **Temporal alignment**: the `--plot-samples` figure `predictions.png` and the `--show-processing` figures show neural activity, inputs and outputs on the same time axis; licking, tongue visibility and the post-go activity increase all start at t=0, and the photostim input coincides with the shaded laser interval.
2. **Output variation**: no output is dominated by one class beyond what the task implies (choice 0.43/0.42/0.15, outcome 0.15/0.17/0.68, early lick 0.88/0.12, tongue 0.10/0.05/0.10/0.76). The tongue 'not visible' class is necessarily dominant because the tongue is only out during licking (~24% of bins), which matches the physiology.
3. **Neuron filtering**: verified again that the neuron set is exactly `classification == 'good'` and reproduces the paper's per-region counts.
4. **Processing vs reference**: see the table in Step 10 Check 3.

### Issues Found and Resolved (cumulative)
- All-zero neural trials caused by ephys recordings that stop before the behavioural session ends -> trials without `obs_intervals` coverage in every good unit are dropped (1,060 trials), plus 2 trailing trials with zero spikes.
- Choice defined from licks over the whole trial disagreed with the behavioural report on 0.55% of trials -> restricted to the 1.5 s answer period (0.33% residual), as defined in the methods.
- `photostim` boolean accumulation bug (float `|=`).
- Tone onset on early-lick trials: the sample epoch is replayed, so the **last** sample onset before the go cue is used.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format specification, key statistics, decoder performance, processing summary)
- [x] `cache/` folder created with all exploration/validation scripts and `README_CACHE.md`
- [x] All required outputs present:
  - `/app/CONVERSION_NOTES.md`, `/app/README.md`
  - `/app/convert_data.py`
  - `/app/converted_data.pkl` (11.9 GB), `/app/sample_data.pkl`
  - `/app/conversion_sample_out.txt`, `/app/verification_sample_out.txt`, `/app/train_decoder_sample_out.txt`
  - `/app/conversion_full_out.txt`, `/app/verification_full_out.txt`, `/app/train_decoder_full_out.txt`
  - processing plots `processing_SC015_*.png` (+ `_summary`), decoder plots `sample_trials.png`, `predictions.png`
