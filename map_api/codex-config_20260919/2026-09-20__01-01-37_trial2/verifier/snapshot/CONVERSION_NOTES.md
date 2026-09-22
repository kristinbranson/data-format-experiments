# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (NWB files in `/app/data`)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `datapaper.pdf`, `methodpaper.pdf`, `ChenLiuEtAl2023_SpikeSortingQC.pdf`, `methods.txt`
- `code/`, `data/`, `pynwb_docs/`
- `decoder.py`, `train_decoder.py`, `CONVERSION_NOTES.md`

Environment check: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124; PyNWB 4.1.0. The required notes-file checkpoint (`ls -la /app/CONVERSION_NOTES.md`) passed.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING/CURATION | Loads per-probe MATLAB exports, combines probes, intersects ephys with histology, loads classifier-QC unit indices, and partitions curated units by side/region. |
| `sliding_histogram` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Counts spikes in half-open bins and optionally divides by bin width; reference run uses 40-ms windows at 3.4-ms stride. |
| `process_one_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Uses already go-cue-relative spikes, clips the analysis window, bins firing rates, and saves behavior plus anatomy. |
| `helper_get_neuron_id_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Applies classifier-QC IDs, histology availability, annotation agreement, and hemisphere (CCF ML midpoint 5700). |
| `load_session` | `VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Concatenates processed region files and exposes firing rates, CCF metadata, trial variables, and ALM indices. |
| `get_regular_trial_mask` | `VideoAnalysisUtils/population_decoding_utils.py` | CURATION | For the paper's regular-trial analyses excludes early lick, auto/free water, no response, and stimulation trials. |
| `align_markers_between_lims` | `Sherlock/align_markers.py` | PROCESSING | Aligns video markers to go cue at 3.4-ms frame spacing, using the last sample in each interval and sorting by trial number. |

### Notes
The repository is the analysis accompanying Wang et al., *Brain-wide analysis reveals movement encoding structured across and within brain areas*. Neural data are electrophysiology, so dF/F is not applicable. The principal preprocessing script (`Sherlock/preprocess_all_ephys.py`) invokes classifier QC, a -3 to +3 s window, 40-ms smoothing/bin width, and 3.4-ms stride. For this task, the required nonoverlapping 50-ms bins override the reference bin width/stride while preserving its half-open counting convention. Raw spike times are already relative to go cue. Reference marker processing includes `tongue_y`, uses side-camera trial numbers, aligns by subtracting absolute go-cue time, and treats incomplete video trials as bad. The task explicitly asks to predict early lick, outcome, and photostimulation, so those trials/conditions cannot be removed using `get_regular_trial_mask`; retaining them is a necessary decoder-task difference. Classifier-QC selection remains applicable and should be reproduced from NWB quality fields if available.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` is DANDI:000363 version 0.230822.0128 (53.6 GB): 174 NWB files arranged as one directory per subject (`sub-<id>/sub-<id>_ses-<timestamp>_behavior+ecephys[+ogen].nwb`) plus `dandiset.yaml`. All NWB inspection used `pynwb.NWBHDF5IO`; no direct HDF5 loading was used.

Each NWB file is one recording session. `nwb.trials` contains start/stop time, trial number/UID, photostimulation onset/power/duration, task/protocol, instructed side, early-lick label, outcome, and auto/free-water flags. `nwb.units` contains absolute spike times, `classification` (`good` or `unlabelled`), extensive sorting-QC metrics, trial-validity masks, electrode links, and `anno_name` CCF anatomy. Curated units are exactly `classification == "good"` in the inspected data and carry anatomical annotations. Acquisition `BehavioralEvents` contains go/sample/delay/trial-end, left/right lick, and photostimulation event timestamps; `BehavioralTimeSeries` contains side-camera jaw, nose, and tongue x/y/likelihood with explicit timestamps. Tongue tracking exists in all 174 sessions.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw units; 69,453 classifier-curated (`classification == good`) |
| Neurons / session | raw 493-3,191 (mean 1,564.5); curated 0-923 (mean 399.2) |
| Subjects | 28 |
| Sessions / subject | 3-10 |
| Trials (total) | 94,990 |
| Trials / session | 264-800 (mean 545.9) |

Native trial-label totals: instruction right 48,913 / left 46,077; outcome ignore 14,095 / miss 15,641 / hit 65,254; early lick no 84,185 / yes 10,805. All trials are `audio delay`, protocol 1. Some sessions have zero curated units and will require session exclusion because a neural decoder cannot use them; the exact count and identities will be established during mapping/implementation. The dataset manifest independently reports 174 files and 28 subjects, matching the PyNWB inventory.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units (~70,000) | Data paper Methods: classifier-labeled good units; 25.9% of KS2 clusters. |
| Neurons / session | median 393 | Data paper Results. |
| Subjects | 28 | Both papers. |
| Sessions / subject | not tabulated | — |
| Trials (total) | not tabulated | — |
| Trials / session | mean 476, range 130-785 in selected analyses | Data paper Methods. |
| Neural data time bin | 40-ms width, 3.4-ms stride in movement paper; 200-ms width/10-ms step for population choice decoding | Method paper Methods; data paper population decoder. |
| Behavior data time bin | video at 300 Hz (~3.4 ms) | Both papers. |
| Reward rate | 84% correct, range 65-99%, on selected control/non-early sessions/trials | Data paper Methods. |
| Photoinhibition prevalence | ~25% randomly interleaved trials in 17 mice | Data paper Methods. |
| Behavioral choice decoding | video AUC 0.51 pre-sample; 0.74 sample+delay; 0.99 post-go | Method paper Results, 106 sessions; this is video-to-choice and not directly comparable to neural decoder balanced accuracy. |


### Processing Details
The task is an auditory delayed-response task: 3 or 12 kHz instruction tones (three 150-ms tones separated by 100 ms), a 1.2-s delay, then a 100-ms go cue; the response window is 1.5 s. Early licking triggers replay of the sample/delay epoch. Neural spikes are Kilosort2 output. Reference movement processing uses 40-ms firing-rate windows at 3.4-ms stride; the decoder specification instead mandates 50-ms bins and -2.5 to +1.5 s around go cue. Videos are side/bottom at 300 Hz; the movement paper uses side view. Marker outliers were defined by five-sigma frame velocity and imputed from nearby frames; when tongue was occluded/in the mouth, position was set to its mean. Here the explicit fourth tongue class (“not visible”) requires preserving missing/low-confidence visibility instead of mean imputation. Photoinhibition occupies the final 0.5 s of delay and ends at go cue (including ramp-down).

### Curation Steps

**Neuron curation rules**:
Use the released region-specific logistic-regression classifier result (`units.classification == "good"`). The classifiers use 15 quality metrics and were trained on manual good/unlabelled labels by major region. The newer movement paper additionally excludes mean firing rates below 2 Hz for video-to-neural prediction, but this is analysis-specific and would discard valid classifier-QC neural inputs; it is not adopted for this general decoder conversion.

**Trial curation rules**:
The papers' specific analyses usually exclude photoinhibition, free/auto-water, early-lick, and ignore trials; selected behavioral analyses also require >65% control performance and ≥50 correct trials per side. Those exclusions conflict with required decoder variables/classes, so all native trials with valid go-cue alignment are retained. Any session with zero curated neurons must be excluded. The full -2.5/+1.5 window lies within the NWB trials around go cue and will be checked explicitly.

### Decoders Trained
| Decoded variable | Accuracy |
| Neural population choice (data paper) | Time-resolved figure; no single tabulated accuracy; logistic decoder with 200 neurons, 200-ms bin, 10-ms step. |
| Video-to-choice (method paper) | ROC AUC 0.51 pre-sample, 0.74 sample+delay, 0.99 after go (not directly comparable). |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Curated units | Classifier-QC IDs used | 69,453 `classification=good` | 69,943 good units | Use archive-native classifier labels. The paper value differs by 490 (likely version/reporting transposition); the archive exactly reproduces 173 nonempty sessions and ~70k/25.5% of 272,227 clusters. |
| Sessions | Process every source session with curated units | 174 NWBs, one has zero good units; 173 usable | 173 behavioral sessions | Exclude only `sub-440958_ses-20190216T162508...`, yielding the published 173 sessions. |
| Trial filtering | `get_regular_trial_mask` removes early, ignore, stimulation, free/auto water | NWB explicitly represents all these classes | Papers remove them for specific analyses | Retain them because they are requested decoder inputs/outputs; this is a necessary task-specific difference. |
| Firing-rate binning | Reference half-open 40-ms windows/3.4-ms stride | Absolute spike times plus go events support arbitrary bins | 40 ms/3.4 ms (movement); other bins analysis-dependent | Use required nonoverlapping 50-ms half-open bins; divide counts by 0.05 s. |
| Trial window versus NWB interval | Raw reference spikes are go-relative and may extend before nominal trial start | 17,467 trials have go-2.5 earlier than NWB `trials.start_time`, especially replay/early trials | Early licking can replay epochs | Do not treat the behavioral interval boundary as missing neural/video data; use continuous acquisition/spikes and exact requested go window. |
| Tongue handling | Last frame per aligned interval; paper imputes marker outliers/occluded tongue | Tongue likelihood is strongly bimodal; all sessions contain timestamps/x/y/likelihood | Occluded tongue set to mean for continuous regressions | For this categorical task preserve occlusion as class 3 using DLC likelihood <0.9; use reference last-sample alignment for visible y. |

Final understanding: one NWB equals one session; use all 173 sessions containing classifier-good, anatomically annotated units, all 94,370 trials therein, exact go events, continuous behavioral timestamps, and task-mandated bins/labels. All 174 NWBs have one go event per trial. `anno_name` provides 293 fine CCF labels with no empty labels among curated units. Insertion-region totals are not comparable to paper anatomical-area totals because a probe traverses multiple annotated structures.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units.spike_times`, `units.classification`; `BehavioralEvents/go_start_times` | `neural` | Keep classifier `good`; histogram absolute spikes in 80 half-open 50-ms bins spanning go-2.5 to go+1.5; divide by 0.05 to Hz; float32 `(neurons,80)` | `sliding_histogram`, `process_one_area` | Task-required width/stride/window override reference analysis bins. |
| `BehavioralEvents/sample_start_times`; trial boundaries | `input[0]` | Select first tone/sample onset within each native trial; for every bin center store absolute bin-center time minus tone onset (seconds) | raw `task_sample_time`; marker alignment convention | Continuous/time-varying as explicitly requested. |
| trial photostimulation onset/duration | `input[1]` | Binary 1 when a 50-ms bin center lies in `[stim_on, stim_off)`, otherwise 0 | reference aligns stimulation by subtracting go time | Preserves stimulated trials rather than filtering. |
| `trials.trial_instruction` + `trials.outcome` | `output[0]` choice | hit→instructed side; miss→opposite side; ignore→no lick; broadcast across 80 bins | reference `lick_directions`, correctness | Values: left=0, right=1, no lick=2; validate against post-go lick events. |
| `trials.outcome` | `output[1]` outcome | Map ignore=0, miss=1, hit=2; broadcast across time | reference `correctness` | Required order. |
| `trials.early_lick` | `output[2]` early lick | Map no early=0, early=1; broadcast across time | reference `early_lick_trials` | Retained as requested. |
| `BehavioralTimeSeries/Camera0_side_TongueTracking` | `output[3]` tongue y | At each 50-ms bin center use the last tracking frame at/before center (reference convention). DLC likelihood <0.9→3 not visible. Compute session q40/q60 on all native visible y values; visible y <q40→0, q40≤y≤q60→1, y>q60→2. | `align_markers_between_lims`; movement-paper marker methods | Likelihood is bimodal, making 0.9 robust; percentiles exclude invisible frames. |
| `subject.subject_id` | `subjects`, `subject_idx` | Stable sorted unique strings and per-session indices | — | 28 subjects expected. |
| `units.anno_name` | `brain_regions`, `brain_region_idx` | Stable sorted union of fine CCF annotation strings and per-unit indices | CCF helpers | Preserves maximal native anatomical specificity. |

### Key Decisions
1. **Session/trial curation**: Exclude only the one zero-curated-unit session; retain every trial because filtering requested classes would make the decoder targets impossible. Require ≥2 trials (all sessions far exceed this).
2. **Choice derivation**: Outcome plus instruction is deterministic under this two-alternative task and handles no-response cleanly; spot-check against lick-event direction.
3. **Time representation**: Although generic format guidance suggests binary onset series, the Decoder Task explicitly requests continuous “time from tone onset in seconds”; the explicit task controls.
4. **Mixed outputs**: Store all outputs as `(4,80)`; repeat per-trial labels over time so tongue y can remain time-varying and dimensions stay rectangular.
5. **Validity masks**: Do not apply `units.is_good_trials` as an additional undocumented filter; reference released preprocessing uses session-level classifier QC. Check finite values and spike/bin consistency instead.
6. **Outliers/visibility**: The reference's continuous-regression imputation would erase the requested not-visible class. Use native likelihood for visibility and native y only when visible; five-sigma velocity outliers are rare and will be checked in sample plots/statistics.
7. **Edge convention**: bins are `[edge_i, edge_{i+1})`, centers are edge+25 ms, exactly 80 points; metadata bin size is 50 ms, offsets -2.5/+1.5 s.

### Planned Sanity Checks
- [ ] Direct PyNWB-versus-pickle `np.allclose` check for spike histogram on chosen session/trial/neuron.
- [ ] Direct PyNWB-versus-pickle `np.allclose` check for tone-time and photostimulation rows.
- [ ] Direct PyNWB-versus-pickle `np.allclose` check for four output rows, including tongue alignment/discretization.
- [ ] Verify 173 sessions, 28 subjects, 94,370 trials, and 69,453 curated units before any later issue-driven exclusions.
- [ ] Verify exactly 80 finite time bins, firing rates are nonnegative multiples of 20 Hz, photostim is binary, and categorical ranges match metadata.
- [ ] Compare choice to first response-epoch lick direction on at least three hit/miss/ignore trials and summarize mismatch rate.
- [ ] Confirm per-session tongue thresholds and class occupancy; visually overlay raw y/likelihood, aligned samples, thresholds, and categorical output.
- [ ] Plot neural raster/rates, tone-relative time, photostim interval, trial labels, and tongue processing for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with `--full` (default), `--sample`, and `--show-processing`. It uses only `pynwb.NWBHDF5IO` for NWB access, validates cue/event counts and anatomy, bins spikes with vectorized `numpy.searchsorted`, constructs aligned inputs/outputs, records per-session diagnostics, and writes protocol-5 pickle output. A two-session smoke run completed without errors (848 trials, 834 curated neurons, 0.113-GB pickle).

Code inefficiencies identified:
The dominant unavoidable cost is materializing neuron × trial × 80 firing rates. Per-unit Python spike counting and repeated event scans would be unnecessarily slow.

Code speedups added:
For each unit, `searchsorted` evaluates all trial-bin edges at once; behavioral alignment is vectorized; one NWB is opened once; session cubes are computed once and exposed as trial matrices. Smoke processing took ~0.5-0.6 s/session plus serialization, projecting comfortably below 15 minutes.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 session-units |
| Neurons / session | [459, 375] |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 527 |
| Trials / session | [368, 159] |
| Time from tone onset range | [-0.6, 5.7] s |
| Photostimulation range | [0, 1] |
| Choice distribution | [0.452, 0.408, 0.140] |
| Outcome distribution | [0.140, 0.298, 0.562] |
| Early lick distribution | [0.943, 0.057] |
| Tongue y distribution | [0.053, 0.029, 0.061, 0.857] |

### Processing Plots Review
Both processing figures show go-aligned firing rates, linear tone-relative time, a late-delay photostimulation interval ending before go where applicable, constant per-trial labels, and time-varying tongue visibility/categories. No temporal offset or bin-count anomaly remains. Representative-trial selection was improved to include the most tongue-visible, stimulated, and early-lick trials.

Initial validation revealed hundreds of trailing all-zero trials in session 2. Direct PyNWB inspection showed that eight NWBs contain more behavioral-table rows than each unit's `is_good_trials` vector/recording interval (for example, 480 behavior rows but 160 recorded entries); spikes stopped with the recording. The converter now uses the common per-unit validity-vector length as the source ephys trial count, and removes rare population-all-zero windows (one final session-2 entry in the sample). Across usable sessions the native recorded-entry count is 93,310 rather than 94,370 behavioral rows. Re-conversion reduced the sample from 848 apparent trials to 527 valid neural trials. Re-validation reports no errors or warnings.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Vectorized spike edge searches and single-pass NWB loading | Smoke/sample processing ~0.5 s/session before plotting/serialization. |

| Step | Time / Session | Estimated Total Time |
| Session processing | 0.48 s/session (sample mean) | ~83 s for 173 sessions |
| Plotting + serialization + validation allowance | ~1.0 s/session equivalent | ~3-5 min total |

Exact required files `/app/sample_data.pkl`, `/app/conversion_sample_out.txt`, `/app/verification_sample_out.txt`, and two processing plots exist. Sample pickle size is 0.074 GB.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None after correcting trailing non-neural behavioral rows and one population-all-zero trailing window.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction choice | 0.7123 | 0.6135 |
| Outcome | 0.7522 | 0.6499 |
| Early lick | 0.8420 | 0.7549 |
| Tongue y-position | 0.6657 | 0.5159 |

Training completed on CUDA in ~8 s. Loss decreased monotonically from 16.5767 to 0.5935 (test loss 0.7364). Every validation balanced accuracy exceeded uniform chance (choice/outcome 0.3333, early 0.5, tongue 0.25), with no >1.5× train/validation gap.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11,837,281,595 bytes (11.837 GB decimal)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 reported (~70k) | classifier good units | 69,453 | 69,453 | Archive exact; paper differs by 490 |
| Mean neurons/session | median 393 | — | mean 401.46, median to check | mean 401.46 | Consistent hundreds/session |
| Subjects | 28 | 28 | 28 | 28 | Yes |
| Sessions | 173 | 173 | 174 NWBs / 173 with good units | 173 | Yes |
| Trials (total) | not reported | analysis-dependent | 93,310 ephys trial entries | 90,734 neural-nonzero trials | 2,576 invalid/no-spike windows excluded |
| Trials/session (mean) | selected analysis mean 476, range 130-785 | exclusions analysis-specific | recorded entries mean 539.4 | mean 524.5, range 159-800 | Reasonable; all requested classes retained |
| Time from tone onset range | task timing 1.2-s delay after sample | first trial sample event | native event derived | [-1.5, 11.9] | Long values arise from early-lick replay; retained |
| Photostimulation range | binary presence, ~25% trials in subset | stimulation intervals | onset/duration fields | [0,1] | Yes |
| Choice distribution | not full-data reported | lick direction | event/labels | [0.428,0.422,0.149] | Balanced L/R, plausible ignores |
| Outcome distribution | selected control correct 84% | correctness | native outcomes | [ignore .149, miss .166, hit .684] | Lower than selected 84% as required early/stim/all sessions retained |
| Early lick distribution | excluded in paper analyses | early flag | native labels | [no .884, yes .116] | Plausible |
| Tongue distribution | mostly occluded before response | marker likelihood | native side video | [.062,.032,.065,.841] | Consistent with brief post-go visibility |

Full conversion completed in 178.7 s, below the estimate and optimization threshold. Verification reports valid format with no errors or warnings. Choice inferred from instruction/outcome agrees with first response-epoch lick events on 90,449/90,734 trials (99.69%); rare mismatches reflect complex/missing lick-event sequences and will be spot-checked in Step 10.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` says the format is valid with no errors or warnings. All 173 sessions have ≥159 trials and 90-923 neurons; all arrays are finite.
2. **Independent neural check**: `/app/sanity_checks.py` loaded the original first NWB directly with PyNWB (without importing conversion code), selected converted session 0/trial 5/curated neuron 3, independently applied `np.histogram` to absolute spikes and go-relative edges, divided by 0.05, and passed `np.allclose` against the pickle.
3. **Independent input check**: The same script independently selected the first sample event in the native trial and calculated center-relative seconds, plus the trial-relative photostimulation interval; both rows passed `np.allclose`.
4. **Independent output check**: Native instruction/outcome/early fields and independently timestamp-aligned tongue y/likelihood with session percentiles all passed `np.allclose`. Explicit native hit/miss/ignore trials 2/15/0 map to output pairs (choice,outcome) (right,hit), (right,miss), and (no lick,ignore).
5. **Reference loading comparison**: Reference loads per-probe MATLAB exports then combines probes. This script loads the released combined NWB session via PyNWB. Both access Kilosort2 spikes, task events, video markers, and CCF anatomy; using released NWB avoids re-parsing old MATLAB containers.
6. **Reference neuron/trial filtering comparison**: Both retain only region-classifier `good` units. Paper regular-trial masks exclude required early/stim/ignore classes, so this conversion necessarily retains them. One zero-good session and native intervals with no population spikes are excluded. The newer movement paper's 2-Hz cutoff is specific to video→neural regression and is not applied.
7. **Reference temporal alignment comparison**: Both use go onset as zero. Raw reference exports already store go-relative spikes; NWB stores absolute spikes and absolute go events, so subtracting/adding go produces the same alignment.
8. **Reference binning comparison**: Both use half-open spike intervals and convert counts to Hz. Required nonoverlapping 50-ms bins override movement paper 40-ms/3.4-ms sliding rates and data-paper decoder 200-ms/10-ms bins. Every converted value is exactly a nonnegative multiple of 20 Hz.
9. **Reference input/output comparison**: Tone and photostim derive from the same task timing fields used by reference preprocessing. Choice/correctness/early correspond to reference `lick_directions`, `correctness`, and `early_lick_trials`. Tongue follows the reference last-frame alignment; likelihood-based class 3 intentionally replaces occlusion imputation because visibility is a required output.
10. **Key statistics**: Converted median neurons/session is 390 versus paper 393; mean 401.46, range 90-923. Subjects/sessions are exactly 28/173. The exact archive unit count is 69,453 versus published 69,943 (~0.7% discrepancy documented as archive-version/reporting difference). Output ranges and distributions are valid and scientifically plausible.
11. **Edge cases**: Checked exact 81 edges/80 bins, final edge +1.5 exclusion convention, event-count bounds, sessions with behavioral rows after ephys ends, the zero-good session, all-zero neural windows, empty anatomy, low-confidence/NaN tongue samples, and minimum trial counts. No off-by-one mismatch remains.

### Issues Found and Resolved
- **Trailing behavioral rows after ephys**: Eight files had trial tables longer than unit validity vectors; truncated to the common recorded-entry length.
- **Population-all-zero neural periods**: 2,576 entries across 94 sessions had no spikes from any curated unit (typically trailing after acquisition stopped); excluded as invalid neural periods. This yields 90,734 usable trials and removes all validator warnings.
- **Choice-event mismatch**: Derived labels agree with first response-window lick events for 99.69% of trials. Instruction/outcome is the authoritative trial result; 285 complex/missing event sequences are not used to overwrite native outcome-derived choice.
- **Paper/archive unit discrepancy**: No conversion fix is warranted: the released NWB classifier labels are authoritative and internally yield the exact reported 173 usable sessions and near-identical median units/session.

All checks were rerun after the trial-period fixes. `/app/sanity_checks_out.txt` records all three `np.allclose` groups passing.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes; 19.1680 at epoch 1, 8.0158 at epoch 10, 0.8676 at epoch 100, 0.6627 at epoch 200; test loss 0.6566.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction choice | 0.7055 | 0.6727 | uniform chance 0.3333 |
| Outcome | 0.6958 | 0.6580 | uniform chance 0.3333 |
| Early lick | 0.8018 | 0.7592 | uniform chance 0.5000 |
| Tongue y-position | 0.6725 | 0.6204 | uniform chance 0.2500 |

The exact required command completed on CUDA with 72,533 train and 18,201 validation trials. Sample/prediction plots were created by `--plot-samples`. No OOM fallback was needed.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| Lick direction choice | 0.6727 balanced accuracy | Uniform chance 0.3333 (2.02×); data-paper 200-neuron choice decoder is time-resolved without a tabulated aggregate value; movement-paper video choice AUC is 0.51 pre-sample, 0.74 sample+delay, 0.99 post-go. |
| Outcome | 0.6580 | Uniform chance 0.3333 (1.97×); no directly comparable reported decoder accuracy. |
| Early lick | 0.7592 | Uniform chance 0.5000 (1.52×); no directly comparable reported decoder accuracy. |
| Tongue y-position | 0.6204 | Uniform chance 0.2500 (2.48×); papers predict continuous markers/neural activity using R², not these percentile/visibility classes. |

All outputs exceed the requested 1.5×-chance investigation threshold. Train/validation ratios are 1.05, 1.06, 1.06, and 1.08—well below the 1.5 overfitting threshold. The closest output is early lick at 1.518× chance; it was explicitly investigated: native labels were checked on three concrete trials, class prevalence is 11.6% yes (not degenerate), go alignment and sample timing passed direct PyNWB comparisons, classifier-QC filtering matches the papers, and full loss/accuracy generalize. No conversion change is justified.

Paper comparisons were investigated rather than treated as architecture differences: the method-paper choice numbers decode binary instructed choice from video at particular epochs/correct-trial selections, whereas this task decodes three classes (including no lick) from neural activity over the full requested window. The achieved 0.673 balanced accuracy is consistent with the reported rise from near chance before sample to high accuracy during movement. The data paper's neural population curve likewise rises substantially above chance but provides no single numerical aggregate to match. The continuous video-to-neural R² values are not categorical accuracy measurements and cannot be numerically equated to the requested outcomes.

### Issues Found and Resolved
- **Potential early-lick weakness**: Raw labels/alignment/variation/filtering rechecked; 0.7592 validation balanced accuracy and 1.06 train/validation ratio support correctness; no issue found.
- **Potential choice discrepancy with event licks**: 99.69% agreement and explicit hit/miss/ignore checks support native outcome/instruction mapping; rare complex event sequences retained under authoritative trial labels.
- **Potential leakage/overfit**: Small train-validation gaps for all four outputs; inputs contain only specified time/stimulation variables, and validation uses held-out trials.

All Step 12 checks passed without a new issue requiring reconversion.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

`/app/README.md` documents loading, reproduction, structure, processing, statistics, and full decoder results. Independent investigation code/output were moved to `/app/cache/` and documented in `README_CACHE.md`. All eleven required output/code/log files exist; processing and decoder plots also exist. Final logs end in successful validation/training, and this notes file contains decisions, discrepancies, both critical reviews, sanity checks, and accuracy tables.
