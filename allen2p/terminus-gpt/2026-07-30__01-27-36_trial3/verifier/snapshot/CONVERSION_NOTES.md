# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-30
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .
- ..
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code
- data
- decoder.py
- docker-compose.yaml
- methods.txt
- paper.pdf
- train_decoder.py
- tutorials
- whitepaper.pdf

Python/package verification:
- python3 OK
- numpy 2.3.5
- torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| BehaviorOphysExperiment.from_lims / from_nwb / from_json | code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py | LOADING | Construct visual behavior ophys experiment object from data source |
| BehaviorOphysExperiment.stimulus_presentations | code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py | LOADING | Access time-aligned stimulus presentation table |
| BehaviorOphysExperiment.trials | code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py | LOADING | Access trial table with go/catch/aborted/auto-rewarded and outcomes |
| BehaviorOphysExperiment.running_speed | code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py | LOADING | Access running speed time series aligned to timestamps |
| BehaviorOphysExperiment.eye_tracking / pupil data accessors | code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py | LOADING | Access pupil/eye tracking time series |
| BehaviorOphysExperiment.dff_traces | code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py | LOADING | Access per-cell dF/F traces for neural activity |
| BehaviorOphysExperiment.ophys_timestamps | code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py | PROCESSING | Common ophys time base for temporal alignment |
| BehaviorSession.trials | code/allensdk/brain_observatory/behavior/behavior_session.py | LOADING | Trial definitions and behavioral annotations |
| BehaviorSession.stimulus_presentations | code/allensdk/brain_observatory/behavior/behavior_session.py | LOADING | Image presentation identity and timing table |

### Notes
- Repository is AllenSDK; relevant API for this task is under `allensdk.brain_observatory.behavior`.
- Visual Behavior ophys data appear to be accessed through `BehaviorOphysExperiment`, which exposes aligned tables/streams for trials, stimulus presentations, running speed, eye tracking, dF/F traces, and ophys timestamps.
- Likely conversion path: load each ophys experiment, use `ophys_timestamps` as common time base, segment by `trials`, intersect with `stimulus_presentations`, and extract neural/behavior streams on the ophys clock.
- Need further inspection of underlying data object properties and any quality/curation fields in later steps.


---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Raw dataset is in `data/visual-behavior-ophys-1.1.0/`.
- Main neural data are NWB files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`, one file per available ophys experiment.
- Metadata tables are in `data/visual-behavior-ophys-1.1.0/project_metadata/` and include `ophys_experiment_table.csv`, `ophys_session_table.csv`, `behavior_session_table.csv`, and `ophys_cells_table.csv`.
- Sample NWB inspection showed task-relevant groups for this conversion: `processing/ophys`, `processing/running/speed`, `processing/rewards`, `stimulus/presentation`, and NWB trial intervals.
- Important note: project metadata describe a larger release than the locally available NWB subset; conversion must operate on the locally available NWB files only.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 25475 |
| Neurons / session | 89.701 |
| Subjects | 21 |
| Sessions / subject | 13.524 |
| Trials (total) | 171887 |
| Trials / session | 605.236 |


---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Trial interval | 750 ms image-presentation interval | "By image presentation interval we refer to the 750 ms interval beginning with each image presentation." |
| Omission interval handling | Use the 750 ms after omission time | "For image omissions we used the 750 ms following the time of the omission, when the image should have been presented." |
| Lick bout segmentation | 700 ms inter-lick interval | "Licks were segmented into licking bouts using an inter-lick interval of 700 ms." |
| Neural data representation used in paper analyses | Detected calcium events | "For all analysis of neural data we used the detected calcium events... This process produces, for each cell, a set of calcium events each with a time and magnitude." |
| Decoder granularity in paper | Image-by-image | "To decode task signals on an image by image basis..." |
| Decoder targets in paper | image changes vs repeats; hits vs misses | "...predict either image changes versus repeats (change decoder), or hits versus misses (hit decoder)." |

### Processing Details
- Behavioral analyses in the paper assign events to 750 ms image presentation intervals.
- Omitted images are treated as their own 750 ms interval after the expected image time.
- Licks are grouped into bouts using a 700 ms inter-lick interval, and bout start/end are assigned to image intervals.
- The paper's neural analyses use detected calcium events, not raw fluorescence; this is a critical reference point for choosing the neural signal in conversion.
- The paper's decoding analyses are image-by-image, whereas the present task requires trial segmentation based on experiment-defined trials; this must be reconciled carefully while preserving reference-aligned signal processing.

### Curation Steps

**Neuron curation rules**:
- Paper excerpt indicates use of detected calcium events; exact cell-quality filters still need confirmation from code/SDK and possibly whitepaper.

**Trial curation rules**:
- Task instructions require including Go and Catch trials and excluding Aborted and Auto-rewarded trials.
- Need exact reference text/code definitions for these trial labels and outcome fields.

### Decoders Trained
| Decoded variable | Accuracy |
| image change vs repeat | [not yet extracted from text] |
| hit vs miss | [not yet extracted from text] |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset scale | AllenSDK/project metadata tables describe a larger project release | Local `behavior_ophys_experiments/` folder contains 284 NWB files, 21 subjects, 25,475 cells in available subset | Paper/methods may refer to broader dataset/analysis subset | Convert only locally available NWB sessions; document that local data are a subset of the full release |
| Neural signal choice | AllenSDK exposes dF/F traces and event-related data accessors | NWB files contain ophys processing groups consistent with traces/events | Paper states neural analyses used detected calcium events | Prefer event-based neural representation if accessible cleanly from local NWB/SDK; otherwise verify what reference code uses and justify any fallback |
| Temporal unit of analysis | AllenSDK provides trials and stimulus presentation tables | NWB contains both trial intervals and image-presentation timing | Paper decoder is image-by-image using 750 ms image intervals | For this task, segment by experiment-defined trials as required, but preserve reference-aligned ophys timestamp alignment and image-interval labeling within trials |
| Trial inclusion | AllenSDK trial tables likely contain labels for go/catch/aborted/auto-rewarded | NWB trials table available for each session | Task explicitly requires Go and Catch only, excluding Aborted and Auto-rewarded | Use trial table fields to filter to valid Go/Catch trials and define trial outcomes from the same source |
| Behavioral alignment | AllenSDK exposes running speed, eye tracking, stimulus presentations, and ophys timestamps | NWB contains running/reward/stimulus groups on session time bases | Methods assign behavior to image presentation intervals and use 750 ms windows | Align all time-varying streams to ophys timestamps, then derive per-trial outputs from aligned streams/image intervals |

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| ophys events or dF/F traces per cell on ophys clock | neural | Slice into trial windows on common ophys timestamps; matrix shape (n_neurons, n_timepoints) | BehaviorOphysExperiment.events or dff_traces; ophys_timestamps | Prefer detected calcium events to match paper; use dF/F only if events are inaccessible/impractical |
| none required by decoder task | input | Use empty input vectors/arrays consistently across trials | N/A | Decoder inputs are specified as none for this task |
| stimulus_presentations.image_name over non-grey image periods | output[0] image identity | Convert to categorical time-varying labels on ophys bins within each trial | BehaviorOphysExperiment.stimulus_presentations | Label only during image presentation periods; grey periods may use a dedicated background/blank class if needed for full time series consistency |
| image identity transitions | output[1] image change | Binary time-varying series, 1 immediately after image identity changes, else 0 | stimulus_presentations + ophys_timestamps | Must reflect change events within trial-aligned time series |
| running_speed | output[2] running speed bin | Interpolate/align to ophys timestamps, discretize into 5 equal-percentile bins over valid samples | BehaviorOphysExperiment.running_speed | Time-varying categorical output |
| pupil diameter / eye tracking pupil area-equivalent | output[3] pupil diameter bin | Align to ophys timestamps, derive diameter-like measure if needed, discretize into 5 equal-percentile bins over valid samples | BehaviorOphysExperiment.eye_tracking | Handle missing data carefully; likely use valid samples only for percentile computation |
| trials outcome fields (hit/miss/false alarm/correct reject etc.) | output[4] trial outcome | Static per-trial categorical label repeated or stored as per-trial vector | BehaviorOphysExperiment.trials | Include only Go/Catch, exclude Aborted and Auto-rewarded |
| metadata mouse_id | subjects / subject_idx | Map unique mouse IDs to subject list and per-session indices | metadata / experiment table | Session order follows converted session list |
| metadata targeted_structure | brain_regions / brain_region_idx | Map region name to index for all neurons in session | metadata / experiment table | All neurons in a session likely share targeted_structure |

### Key Decisions
1. **Session source**: Convert only locally available NWB experiment files, not the full project metadata release, because only those sessions are actually present.
2. **Temporal alignment**: Use `ophys_timestamps` as the master clock for all neural and behavioral/stimulus outputs.
3. **Trial segmentation**: Use experiment-defined trial intervals from the trial table; include Go and Catch trials, exclude Aborted and Auto-rewarded trials.
4. **Neural signal**: Prefer detected calcium events to match the paper; verify exact access pattern in SDK/local NWB. If unavailable, use dF/F traces with clear documentation.
5. **Decoder inputs**: Keep decoder `input` empty/zero-width because task specifies no inputs.
6. **Continuous output discretization**: Compute five equal-percentile bins for running speed and pupil diameter using valid pooled samples, then apply bin edges within all sessions.
7. **Output representation**: Make image identity, image change, running bin, and pupil bin time-varying; make trial outcome static per trial.

### Planned Sanity Checks
- [ ] Compare one trial's raw stimulus presentation times to converted image-identity labels on ophys timestamps.
- [ ] Compare one trial's raw running-speed samples to aligned/discretized output bins.
- [ ] Compare one trial's raw pupil signal to aligned/discretized output bins.
- [ ] Compare one neuron's raw event/dF/F segment to converted neural trial slice using `np.allclose()` on a spot check.
- [ ] Verify counts of included/excluded trial types directly from raw trial table versus converted dataset.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented initial conversion script `convert_data.py` using AllenSDK `BehaviorOphysExperiment` loaded from local NWB files via `pynwb`. Added trial filtering, ophys-timestamp alignment, image identity/change outputs, running and pupil discretization, and subject/region metadata. Fixed loader to pass NWBFile rather than file path, and added robust stimulus interval end-time handling.

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 22 |
| Neurons / session | 11.0 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 656 |
| Trials / session | [365, 291] |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
Verification passed with no errors or warnings after switching neural data to dF/F and labeling image identity over 750 ms image-presentation intervals. Image identity distribution became balanced across image classes with omitted fraction ~3.7% and blank ~0.1%.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: Non-fatal HDMF namespace warning; pandas FutureWarning inside AllenSDK stimulus processing

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| image_identity | 0.2323 | 0.1472 |
| image_change | 0.6815 | 0.6569 |
| running_speed_bin | 0.2375 | 0.2285 |
| pupil_diameter_bin | 0.2438 | 0.2354 |
| trial_outcome | 0.2829 | 0.2462 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 14G
- `verification_full_out.txt`: created and passed verification

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | | | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
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
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| image_identity | 0.3758 | 0.3438 | strong above-chance decoding |
| image_change | 0.6985 | 0.6377 | strong above-chance decoding |
| running_speed_bin | 0.2952 | 0.2853 | above chance |
| pupil_diameter_bin | 0.3595 | 0.3477 | above chance |
| trial_outcome | 0.4454 | 0.3233 | above chance |

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
