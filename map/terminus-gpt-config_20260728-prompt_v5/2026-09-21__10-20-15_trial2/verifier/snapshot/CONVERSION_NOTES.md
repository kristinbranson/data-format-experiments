# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- ChenLiuEtAl2023_SpikeSortingQC.pdf
- Dockerfile
- code
- data
- datapaper.pdf
- decoder.py
- docker-compose.yaml
- methodpaper.pdf
- methods.txt
- train_decoder.py


---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| preprocess_all_ephys (script) | Sherlock/preprocess_all_ephys.py | PROCESSING | Main entry point for generating preprocessed ephys session pickle files |
| count_neurons_per_file (script) | Sherlock/count_neurons_per_file.py | LOADING | Loads preprocessed ephys pickles and counts neurons from `fr` arrays |
| collect_trial_type_labels_for_fig6 (script) | Sherlock/collect_trial_type_labels_for_fig6.py | CURATION | Matches trial labels to sessions and filters trials using `correctness` |

### Notes
- `/app/code/Sherlock/preprocess_all_ephys.py` appears to be the main ephys preprocessing entry point used to create per-session preprocessed pickle files for downstream analyses.
- Downstream Sherlock scripts consistently load these preprocessed ephys pickles and access fields such as `fr` and `correctness`, indicating trial-aligned firing-rate arrays and trial outcome labels are stored in the preprocessed representation.
- `/app/code/Sherlock/count_neurons_per_file.py` counts neurons using `ephys_data['fr'].shape[2]`, implying `fr` is organized with neurons on axis 2 in the reference preprocessed files.
- `/app/code/Sherlock/collect_trial_type_labels_for_fig6.py` matches trial UIDs between behavioral labels and preprocessed ephys data, then filters to successful trials using `new_ephys_data['correctness'][trial_inds]`, showing correctness/outcome is part of the standard processed trial metadata.
- `VideoAnalysisUtils` contains reusable preprocessing and decoding utilities used by the Sherlock scripts; these are likely the main source of alignment/binning helper functions.
- Initial conclusion: the reference workflow first converts raw session data into standardized preprocessed per-session pickle files, then performs decoding/analysis from those pickles rather than directly from NWB each time.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
Data are organized as NWB files in subject-specific folders under `/app/data/sub-<subject>/`. Each session is one `*_behavior+ecephys+ogen.nwb` file containing trial and unit tables plus acquisition groups `BehavioralEvents` and `BehavioralTimeSeries`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272227 |
| Neurons / session | mean 1564.52 (min 493, max 3191) |
| Subjects | 28 |
| Sessions / subject | varies by subject (see notes; total 174 sessions) |
| Trials (total) | 94990 |
| Trials / session | mean 545.92 (min 264, max 800) |
Sessions per subject from filenames: sub-440956: 4, sub-440957: 4, sub-440958: 5, sub-440959: 8, sub-441666: 5, sub-442571: 5, sub-449141: 6, sub-455219: 4, sub-455220: 6, sub-456772: 10, sub-456773: 4, sub-456774: 3, sub-460432: 4, sub-460434: 4, sub-460436: 4, sub-479121: 6, sub-479149: 7, sub-480133: 9, sub-480134: 7, sub-480135: 7, sub-480927: 10, sub-480928: 9, sub-484672: 6, sub-484673: 9, sub-484674: 9, sub-484675: 6, sub-484676: 7, sub-484677: 6.

Representative NWB contents observed:
- Example file: `sub-440956_ses-20190207T120657_behavior+ecephys+ogen.nwb`
- Acquisition groups: `BehavioralEvents`, `BehavioralTimeSeries`
- Trial columns: ['start_time', 'stop_time', 'trial', 'photostim_onset', 'photostim_power', 'photostim_duration', 'trial_uid', 'task', 'task_protocol', 'trial_instruction', 'early_lick', 'outcome', 'auto_water', 'free_water']
- Unit columns: ['unit', 'sampling_rate', 'unit_quality', 'unit_posx', 'unit_posy', 'unit_amp', 'unit_snr', 'isi_violation', 'avg_firing_rate', 'drift_metric', 'left_trials_drift_metric', 'right_trials_drift_metric', 'presence_ratio', 'amplitude_cutoff', 'isolation_distance', 'l_ratio', 'd_prime', 'nn_hit_rate', 'nn_miss_rate', 'silhouette_score', 'max_drift', 'cumulative_drift', 'duration', 'halfwidth', 'pt_ratio', 'repolarization_slope', 'recovery_slope', 'spread', 'velocity_above', 'velocity_below', 'classification', 'anno_name', 'is_good_trials', 'spike_times', 'obs_intervals', 'electrodes', 'electrode_group', 'waveform_mean', 'waveform_sd']
- Behavioral event streams: ['delay_start_times', 'delay_stop_times', 'go_start_times', 'go_stop_times', 'left_lick_times', 'photostim_start_times', 'photostim_stop_times', 'presample_start_times', 'presample_stop_times', 'right_lick_times', 'sample_start_times', 'sample_stop_times', 'trialend_start_times', 'trialend_stop_times']
- Behavioral time series streams: ['Camera0_side_JawTracking', 'Camera0_side_NoseTracking', 'Camera0_side_TongueTracking']

Sessions per subject from filenames: sub-484672: 2, sub-484673: 9, sub-484674: 9, sub-484675: 6, sub-484676: 7, sub-484677: 5.

Representative NWB contents observed:
- Acquisition groups: `BehavioralEvents`, `BehavioralTimeSeries`
- Trial columns, unit columns, and behavioral stream names printed in terminal for later mapping in Steps 4-5.


---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units in full paper dataset; 16,030 units in currently inspected NWB subset | "Overall, the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" | 
| Neurons / session | mean 421.8 (min 171, max 840) | |
| Subjects | 17 VGAT-ChR2-EYFP mice for photoinhibition subset; full NWB release contains 28 subjects | "N = 17 VGAT-ChR2-EYFP mice" |
| Sessions / subject | varies by subject (see notes; total 174 sessions) | |
| Trials (total) | 21773 | |
| Trials / session | paper: mean 476 (range 130-785); NWB subset currently observed mean 572.97 (min 381, max 742) | "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Neural data time bin | spikes acquired at 30 kHz; analysis binning to be determined from code | "Data from up to five simultaneously recorded probes were acquired at 30 kHz." |
| Behavior data time bin | | |
| Reward rate | 84% correct control-trial performance on average | "Mice performed on average 476 trials per session with 84% correct rate" | 
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| ... | | | | 


### Processing Details
- Task: auditory delayed-response licking task with sample tones (3 or 12 kHz), 1.2 s delay, 0.1 s Go cue, 1.5 s answer period, and 1.5 s consumption period.
- Early licks during sample/delay triggered replay of the epoch; early-lick and no-response trials were excluded in many paper analyses.
- Photoinhibition occurred on a subset of ~25% randomly interleaved trials and ended before the Go cue; this supports constructing a time-varying photostim input aligned to Go cue.
- Neural recordings were acquired at 30 kHz with Neuropixels probes; spike sorting used Kilosort2 with post hoc quality-control classifiers described in the white paper.

### Curation Steps

**Neuron curation rules**:
- Use units labeled as `good` by the reference quality-control pipeline/classifiers when that information is available in the NWB/unit metadata.
- Reference text reports 69,943 good units across 173 behavioral sessions in the full paper dataset.

**Trial curation rules**:
- Paper analyses often excluded early-lick and no-response trials and selected sessions with >65% control-trial performance plus at least 50 correct left and 50 correct right trials.
- For the decoder conversion, we must inspect whether all NWB sessions satisfy these criteria and whether any extra NWB file explains the 174-file vs 173-session discrepancy.

### Decoders Trained
| Decoded variable | Accuracy |
| Not yet extracted from papers/code | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Reference code appears built around preprocessed sessions from the original project | Local `/app/data` currently contains 174 NWB files across 28 subjects | Methods text states 173 behavioral sessions for the full paper dataset | Treat local data release as the authoritative source for conversion; investigate whether one NWB file is extra, auxiliary, or fails paper inclusion criteria during later filtering |
| Unit count | Reference papers analyze `good` units after QC classifiers | Raw NWB unit table count in local files is lower than full-paper total and may include only a subset/release-specific sessions | Methods text states 69,943 good units across 173 sessions | Need to inspect NWB unit metadata for quality labels and use those rather than raw counts when building the converted dataset |
| Trial counts/session | Code downstream often filters by `correctness` and selected trial sets | Local NWB files expose all trials, likely including early/no-response/photostim trials | Methods text reports average 476 trials/session and session inclusion criteria | Expect converted trial counts to differ before applying paper-style trial/session filters; final decisions to be documented in mapping step |
| Behavioral variables | Code references `correctness`, trial UIDs, and preprocessed `fr` arrays | NWB acquisition contains go/sample/delay/photostim/lick events and tongue/jaw/nose tracking streams | Methods text describes Go cue, sample tones, delay, licks, photoinhibition, DeepLabCut tracking | Sources are consistent on the available variable families; exact field mapping to be resolved in Step 5 |


---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| NWB `units` spike times for good units | neural | Bin spikes in 50 ms bins from -2.5 s to +1.5 s around Go cue; convert to firing rates or spike counts consistently across sessions | Sherlock/preprocess_all_ephys.py; downstream scripts use preprocessed `fr` | Need exact good-unit flag from NWB unit columns; if unavailable, document fallback |
| Trial-relative time from tone onset | input[0] | Time-varying continuous input aligned to Go cue; likely represent as per-bin elapsed time since sample/tone onset within trial window | Consistent with task timing from methods; no exact helper identified yet | Tone onset occurs before Go cue by sample + delay timing; use trial event timestamps from NWB |
| Photostimulation on/off | input[1] | Binary time-varying indicator from photostim start/stop event streams over aligned bins | BehavioralEvents photostim_start/stop_times | Photostim ends before Go cue per methods |
| Choice | output[0] | Per-trial categorical label: left / right / no lick | collect_trial_type_labels_for_fig6.py references trial labels and correctness | Determine exact NWB trial column encoding |
| Outcome | output[1] | Per-trial categorical label: ignore / miss / hit | trial metadata and/or correctness columns | Need exact mapping from NWB trial columns |
| Early lick | output[2] | Per-trial categorical label: no / yes | NWB trial metadata | Important because papers often excluded these trials |
| Tongue y-position | output[3] | Time-varying categorical output from tongue tracking y coordinate discretized by session percentiles; 3 = not visible | BehavioralTimeSeries `Camera0_side_TongueTracking` | Need to identify coordinate/confidence columns from timeseries array |

### Key Decisions
1. **Use Go cue onset as alignment event**: Required by decoder task and supported by explicit `go_start_times`/`go_stop_times` event streams in NWB.
2. **Retain trial-level outputs even if papers excluded some trial types**: Decoder task explicitly requires predicting no-lick/early-lick/outcome variables, so exclusion rules from paper analyses may need to be relaxed for conversion while still documented.
3. **Use paper/code curation when possible for units and sessions**: Prefer good-unit labels and paper-style session filters where compatible with decoder-task requirements.

### Planned Sanity Checks
- [ ] Compare binned spike counts for one trial/session against direct raw spike-time histogram around Go cue
- [ ] Compare photostim binary vector against raw photostim start/stop times for one trial
- [ ] Compare tongue visibility/discretization against raw tracking rows for one trial
- [ ] Verify choice/outcome/early-lick labels on three hand-checked trials

---

## Step 6: Script Development
**Status**: IN PROGRESS

[Implementation notes]

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: IN PROGRESS

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 16030 |
| Neurons / session | mean 421.8 (min 171, max 840) |
| Subjects | 28 |
| Sessions / subject | varies by subject (see notes; total 174 sessions) |
| Trials (total) | 21773 |
| Trials / session | mean 572.97 (min 381, max 742) |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
[Notes on any anomalies]

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: NOT STARTED

### Format Validation
- Errors: [None / List]
- Warnings: [None / List]

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| <Output 1> | | |
| <Output 2> | | |
| ... | | |

---

## Step 9: Full Conversion and Validation
**Status**: NOT STARTED

### Output Files
- `converted_data.pkl`: [size]
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | Session count | Reference code appears built around preprocessed sessions from the original project | Local `/app/data` currently contains 174 NWB files across 28 subjects | Methods text states 173 behavioral sessions for the full paper dataset | Treat local data release as the authoritative source for conversion; investigate whether one NWB file is extra, auxiliary, or fails paper inclusion criteria during later filtering |
| Unit count | Reference papers analyze `good` units after QC classifiers | Raw NWB unit table count in local files is lower than full-paper total and may include only a subset/release-specific sessions | Methods text states 69,943 good units across 173 sessions | Need to inspect NWB unit metadata for quality labels and use those rather than raw counts when building the converted dataset |
| Trial counts/session | Code downstream often filters by `correctness` and selected trial sets | Local NWB files expose all trials, likely including early/no-response/photostim trials | Methods text reports average 476 trials/session and session inclusion criteria | Expect converted trial counts to differ before applying paper-style trial/session filters; final decisions to be documented in mapping step |
| Behavioral variables | Code references `correctness`, trial UIDs, and preprocessed `fr` arrays | NWB acquisition contains go/sample/delay/photostim/lick events and tongue/jaw/nose tracking streams | Methods text describes Go cue, sample tones, delay, licks, photoinhibition, DeepLabCut tracking | Sources are consistent on the available variable families; exact field mapping to be resolved in Step 5 |

| Mean neurons/session | Session count | Reference code appears built around preprocessed sessions from the original project | Local `/app/data` currently contains 174 NWB files across 28 subjects | Methods text states 173 behavioral sessions for the full paper dataset | Treat local data release as the authoritative source for conversion; investigate whether one NWB file is extra, auxiliary, or fails paper inclusion criteria during later filtering |
| Unit count | Reference papers analyze `good` units after QC classifiers | Raw NWB unit table count in local files is lower than full-paper total and may include only a subset/release-specific sessions | Methods text states 69,943 good units across 173 sessions | Need to inspect NWB unit metadata for quality labels and use those rather than raw counts when building the converted dataset |
| Trial counts/session | Code downstream often filters by `correctness` and selected trial sets | Local NWB files expose all trials, likely including early/no-response/photostim trials | Methods text reports average 476 trials/session and session inclusion criteria | Expect converted trial counts to differ before applying paper-style trial/session filters; final decisions to be documented in mapping step |
| Behavioral variables | Code references `correctness`, trial UIDs, and preprocessed `fr` arrays | NWB acquisition contains go/sample/delay/photostim/lick events and tongue/jaw/nose tracking streams | Methods text describes Go cue, sample tones, delay, licks, photoinhibition, DeepLabCut tracking | Sources are consistent on the available variable families; exact field mapping to be resolved in Step 5 |

| Subjects | 17 VGAT-ChR2-EYFP mice for photoinhibition subset; full NWB release contains 28 subjects | "N = 17 VGAT-ChR2-EYFP mice" | | | |
| Sessions | Session count | Reference code appears built around preprocessed sessions from the original project | Local `/app/data` currently contains 174 NWB files across 28 subjects | Methods text states 173 behavioral sessions for the full paper dataset | Treat local data release as the authoritative source for conversion; investigate whether one NWB file is extra, auxiliary, or fails paper inclusion criteria during later filtering |
| Unit count | Reference papers analyze `good` units after QC classifiers | Raw NWB unit table count in local files is lower than full-paper total and may include only a subset/release-specific sessions | Methods text states 69,943 good units across 173 sessions | Need to inspect NWB unit metadata for quality labels and use those rather than raw counts when building the converted dataset |
| Trial counts/session | Code downstream often filters by `correctness` and selected trial sets | Local NWB files expose all trials, likely including early/no-response/photostim trials | Methods text reports average 476 trials/session and session inclusion criteria | Expect converted trial counts to differ before applying paper-style trial/session filters; final decisions to be documented in mapping step |
| Behavioral variables | Code references `correctness`, trial UIDs, and preprocessed `fr` arrays | NWB acquisition contains go/sample/delay/photostim/lick events and tongue/jaw/nose tracking streams | Methods text describes Go cue, sample tones, delay, licks, photoinhibition, DeepLabCut tracking | Sources are consistent on the available variable families; exact field mapping to be resolved in Step 5 |

| Trials (total) | 21773 | | | | |
| Trials/session (mean) | Session count | Reference code appears built around preprocessed sessions from the original project | Local `/app/data` currently contains 174 NWB files across 28 subjects | Methods text states 173 behavioral sessions for the full paper dataset | Treat local data release as the authoritative source for conversion; investigate whether one NWB file is extra, auxiliary, or fails paper inclusion criteria during later filtering |
| Unit count | Reference papers analyze `good` units after QC classifiers | Raw NWB unit table count in local files is lower than full-paper total and may include only a subset/release-specific sessions | Methods text states 69,943 good units across 173 sessions | Need to inspect NWB unit metadata for quality labels and use those rather than raw counts when building the converted dataset |
| Trial counts/session | Code downstream often filters by `correctness` and selected trial sets | Local NWB files expose all trials, likely including early/no-response/photostim trials | Methods text reports average 476 trials/session and session inclusion criteria | Expect converted trial counts to differ before applying paper-style trial/session filters; final decisions to be documented in mapping step |
| Behavioral variables | Code references `correctness`, trial UIDs, and preprocessed `fr` arrays | NWB acquisition contains go/sample/delay/photostim/lick events and tongue/jaw/nose tracking streams | Methods text describes Go cue, sample tones, delay, licks, photoinhibition, DeepLabCut tracking | Sources are consistent on the available variable families; exact field mapping to be resolved in Step 5 |

| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: NOT STARTED

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

### Training Progress
- Loss decreasing: [Yes/No]

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| <Output 1> | | |
| <output 2> | | |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
