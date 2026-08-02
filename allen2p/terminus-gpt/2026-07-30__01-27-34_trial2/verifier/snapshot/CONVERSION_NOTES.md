# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-30
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code/
- data/
- decoder.py
- docker-compose.yaml
- methods.txt
- paper.pdf
- train_decoder.py
- tutorials/
- whitepaper.pdf

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| BehaviorOphysExperiment.from_nwb / from_lims | code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py | LOADING | Construct ophys experiment object with behavior, ophys traces/events, timestamps, metadata |
| BehaviorSession.from_json / from_nwb | code/allensdk/brain_observatory/behavior/behavior_session.py | LOADING | Construct behavior session and assemble trials, stimulus presentations, running speed, eye tracking |
| Trials.from_stimulus_file / from_nwb | code/allensdk/brain_observatory/behavior/data_objects/trials/trials.py | PROCESSING | Build trial table including go/catch, aborted, auto_rewarded and response/outcome fields |
| Presentations.from_stimulus_file / from_nwb | code/allensdk/brain_observatory/behavior/data_objects/stimuli/presentations.py | PROCESSING | Build stimulus presentation table with image identity, timing, omissions, and change flags |
| RunningSpeed.from_stimulus_file / from_multiple_stimulus_files | code/allensdk/brain_observatory/behavior/data_objects/running_speed/running_speed.py | PROCESSING | Load running speed aligned via stimulus/sync timestamps, optionally filtered |
| EyeTrackingTable.from_nwb / from_data_file | code/allensdk/brain_observatory/behavior/data_objects/eye_tracking/eye_tracking_table.py | PROCESSING | Load eye tracking table including pupil-related measures and timestamps |
| DFFTraces.from_nwb | behavior data_objects cell specimen module | LOADING | Load fluorescence dF/F traces per ROI on ophys timestamps |
| Events.from_nwb | behavior data_objects cell specimen module | LOADING | Load deconvolved/detected event traces per ROI on ophys timestamps |
| CellSpecimens.from_nwb | behavior data_objects cell specimen module | CURATION | Load ROI/cell metadata and valid cell specimen selection for experiment |

### Notes
- AllenSDK behavior/ophys objects are assembled from NWB/JSON/LIMS and already encode alignment between behavior and ophys streams through sync/timestamps.
- Running speed loading explicitly supports filtered and unfiltered versions; likely use filtered running_speed for decoder output unless references indicate otherwise.
- Trials table contains the fields needed to include Go and Catch while excluding Aborted and Auto-rewarded trials.
- Stimulus presentations table appears to be the source for image identity over time and image change flags.
- Ophys neural signal for this dataset is calcium imaging; likely use SDK-provided dF/F traces or events rather than recomputing from raw fluorescence unless reference code indicates otherwise.
- Need Step 2 to determine what precomputed files are actually present in data/ and whether NWB files, CSVs, or pickles are supplied.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Dataset root: `data/visual-behavior-ophys-1.1.0`
- Contains project-level metadata CSV tables plus per-experiment/session data files (to be inspected further in later steps).
- Key metadata tables: `behavior_session_table.csv`, `ophys_session_table.csv`, `ophys_experiment_table.csv`.
- `ophys_session_table.csv` includes mouse/session identifiers, session_type, behavior_type, image_set, experience_level, and list-valued `ophys_experiment_id`.
- Visual Behavior Ophys dataset structure distinguishes behavior sessions, ophys sessions, and ophys experiments (multiple experiments can belong to one ophys session).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 22 |
| Neurons / session | mean=11; sessions: 13, 9 |
| Subjects | 107 |
| Sessions / subject | behavior mean=44.69; ophys session mean=6.57 |
| Trials (total) | 8515 image-presentation trials |
| Trials / session | 4093, 4422 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | TBD after inspecting per-experiment cell tables/files | | 
| Neurons / session | TBD after inspecting per-experiment cell tables/files | |
| Subjects | 107 | |
| Sessions / subject | behavior mean=44.69; ophys session mean=6.57 | |
| Trials (total) | TBD after inspecting trial-level files | |
| Trials / session | TBD after inspecting trial-level files | |
| Neural data time bin | 750 ms image-presentation interval for image-by-image analyses in paper; native ophys frame timing also available | "By image presentation interval we refer to the 750 ms interval beginning with each image presentation." (methods.txt) |
| Behavior data time bin | 750 ms image-presentation interval for behavioral event assignment | "We performed all of our behavioral analysis after assigning behavioral events to each image presentation interval... the 750 ms interval beginning with each image presentation." (methods.txt) |
| Reward rate | | | | 
| Trial segmentation unit | 750 ms image presentation interval | "By image presentation interval we refer to the 750 ms interval beginning with each image presentation." (methods.txt) | 
| Neural signal used in paper | detected calcium events | "For all analysis of neural data we used the detected calcium events..." (methods.txt) |
| Lick bout threshold | 700 ms inter-lick interval | "Licks were segmented into licking bouts using an inter-lick interval of 700 ms." (methods.txt) | 


### Processing Details
- Behavioral events were assigned to each 750 ms image presentation interval.
- For omitted images, the 750 ms interval after the omission time was used.
- Licks were segmented into bouts using a 700 ms inter-lick interval threshold.
- Neural analyses in the paper used detected calcium events rather than raw fluorescence or recomputed dF/F.
- The paper states that familiar image set sessions from the multiplane calcium imaging rig were used for most neural analyses; for this conversion we will include all available Visual Behavior task sessions unless later reference constraints indicate stronger filtering is required.
- Decoder analyses in the paper were image-by-image, suggesting stimulus-aligned segmentation is the natural unit for trialing.

### Curation Steps

**Neuron curation rules**:
- Use SDK-provided valid cell/ROI tables and whichever neural signal the reference analysis used (detected calcium events per methods text).
- Additional filtering rules still need confirmation from code/data inspection.

**Trial curation rules**:
- Include Go and Catch trials.
- Exclude Aborted and Auto-rewarded trials, per task instructions.
- Trial/image segmentation should follow image presentation intervals and trial definitions in the SDK tables.

### Decoders Trained
| Decoded variable | Accuracy |
| image changes vs repeats | accuracy not numerically copied yet; decoder described in methods |
| hits vs misses | accuracy not numerically copied yet; decoder described in methods |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal choice | SDK exposes both `dff_traces` and `events` in `BehaviorOphysExperiment` | Example NWB contains both, aligned to ophys timestamps | methods.txt says neural analyses used detected calcium events | Use `events` as primary neural signal for conversion; keep note that dF/F was available but not chosen |
| Trial definition | SDK provides `trials` table with `go`, `catch`, `aborted`, `auto_rewarded` columns | Example session has these columns and 503 trials | Task instructions require Go and Catch, exclude Aborted and Auto-rewarded | Filter trials using SDK trial table flags |
| Time-varying stimulus labels | SDK provides `stimulus_presentations` with `image_name`, `is_change`, `omitted`, `trials_id`, start/end times | Example session has 13,808 stimulus presentations | methods assign behavior to 750 ms image presentation intervals | Use stimulus presentations to construct time-varying image identity and image-change labels within trials |
| Temporal alignment basis | SDK exposes `ophys_timestamps`, running speed timestamps, eye tracking timestamps | Example session has all streams available | User task explicitly says align based on ophys timestamp | Resample/assign all outputs onto ophys timestamps within each trial |
| Session granularity | Metadata distinguishes behavior sessions, ophys sessions, and ophys experiments | Example NWB is per ophys experiment with ROI-level data | Paper discusses imaging planes/experiments within sessions | Treat each ophys experiment NWB as one decoder session because neural populations are experiment-specific |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `BehaviorOphysExperiment.events['events']` per ROI | neural | Stack per-cell event traces into `(n_neurons, n_timepoints)` trial matrices by slicing on ophys timestamps between trial start/stop | `BehaviorOphysExperiment.from_nwb_path`, `Events.from_nwb` | Primary neural signal because methods text says detected calcium events were used |
| none | input | Use zero-dimensional input arrays per trial (no decoder inputs requested) | n/a | `input_names = []`; each trial input can be `np.zeros((0, T), dtype=float32)` |
| `stimulus_presentations.image_name` | output[0] image identity | Map each ophys timestamp to active image presentation interval; encode categorical integer time series | `Presentations.from_nwb` | Time-varying; use non-grey image presented during each interval |
| `stimulus_presentations.is_change` | output[1] image change | Binary time series on ophys timestamps, 1 during image presentation intervals immediately after an image change, else 0 | `Presentations.from_nwb` | Time-varying |
| `running_speed.speed` with `running_speed.timestamps` | output[2] running speed bin | Interpolate or assign running speed onto ophys timestamps, then discretize globally into 5 equal-frequency percentile bins | `RunningSpeed.from_stimulus_file/from_nwb` via experiment object | Time-varying categorical output |
| eye tracking pupil measure (`pupil_area` or diameter-derived value) with `eye_tracking.timestamps` | output[3] pupil diameter bin | Convert to pupil diameter proxy (prefer geometric diameter from area: `2*sqrt(area/pi)` if direct diameter absent), align to ophys timestamps, exclude blink/missing frames as needed, discretize into 5 equal-frequency percentile bins | `EyeTrackingTable.from_nwb` | Time-varying categorical output |
| `trials` outcome flags (`hit`, `miss`, `false_alarm`, `correct_reject`) | output[4] trial outcome | Static per-trial categorical label derived from mutually exclusive outcome flags | `Trials.from_nwb` | Include only Go/Catch, exclude Aborted/Auto-rewarded |
| `metadata['mouse_id']` | subjects / subject_idx | Unique mouse IDs mapped to subject index per experiment/session | experiment metadata | One subject per experiment session |
| `metadata['targeted_structure']` | brain_regions / brain_region_idx | Map targeted structure string to region index for all neurons in experiment | experiment metadata and/or cell specimen table | Single region per experiment expected |

### Key Decisions
1. **Neural signal = detected events**: methods.txt explicitly states neural analyses used detected calcium events, so use `events` rather than dF/F.
2. **Session unit = ophys experiment**: each NWB/experiment has its own neural population and aligned behavior tables; this matches decoder session semantics.
3. **Temporal basis = ophys timestamps**: all trial matrices will be sampled on native ophys timestamps as required by the task.
4. **Trial segmentation = SDK trials table**: start/stop from `trials`; keep only rows with `go` or `catch`, excluding `aborted` and `auto_rewarded`.
5. **Image labels from stimulus presentations**: assign image identity and change labels by overlap of ophys timestamps with stimulus presentation intervals.
6. **Running/pupil binning global across included data**: compute percentile bin edges over all valid timepoints in processed sessions to ensure comparable categorical outputs across sessions.
7. **Pupil variable choice**: derive diameter from pupil area if no direct diameter column exists; mask likely blinks and missing values before binning/alignment.
8. **Static trial outcome representation**: store outcome as a per-trial scalar categorical output; decoder format permits static outputs.

### Planned Sanity Checks
- [ ] For one raw NWB experiment, verify converted neural trial slice equals the original event trace values at selected neuron/time indices using `np.allclose()`.
- [ ] For one trial, verify converted image identity and image-change labels match `stimulus_presentations` intervals on the same ophys timestamps.
- [ ] For one trial, verify converted running-speed bins correspond to raw running speed ordering after alignment to ophys timestamps.
- [ ] For one trial, verify converted pupil bins come from the chosen raw pupil measure after blink masking and alignment.
- [ ] Verify trial inclusion/exclusion exactly matches raw `trials` flags (`go`/`catch` included; `aborted`/`auto_rewarded` excluded).

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `convert_data.py` with CLI `python -u convert_data.py <outpicklefile>` and options `--full`, `--sample`, `--show-processing`.
- Loads each NWB via `BehaviorOphysExperiment.from_nwb_path`.
- Uses detected calcium `events` as neural signal.
- Filters trials to include Go/Catch and exclude Aborted/Auto-rewarded.
- Aligns image labels, running speed, and pupil-derived diameter to ophys timestamps within each trial.
- Discretizes running speed and pupil into 5 global percentile bins.
- Saves decoder-format pickle and optional processing plots.

Code inefficiencies identified:
- Current implementation iterates over stimulus presentations per trial and loads experiments serially.

Code speedups added:
- Uses direct slicing on native ophys timestamps.
- Stores neural traces as float32.
- Reuses one pass over sessions to collect global binning statistics.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | TBD after inspecting per-experiment cell tables/files |
| Neurons / session | TBD after inspecting per-experiment cell tables/files |
| Subjects | 107 |
| Sessions / subject | behavior mean=44.69; ophys session mean=6.57 |
| Trials (total) | TBD after inspecting trial-level files |
| Trials / session | TBD after inspecting trial-level files |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
- Initial full-trial segmentation produced invalid unlabeled image-identity periods during gray screens.
- Revised conversion uses image-presentation intervals as trials, eliminating invalid labels and matching the paper's image-by-image analysis.
- Processing plots for up to 2 sessions were generated and showed aligned neural/activity/output traces over 7-8 ophys-frame image windows.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| float32 neural arrays + direct timestamp slicing | reduced memory and simple indexing |
| one-pass global bin edge collection | avoided second raw-data load |

| Step | Time / Session | Estimated Total Time |
| sample conversion | ~22 s/session | ~104 minutes for 284 sessions before optimization |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: Verifier accepted structure; initial invalid image labels were fixed by switching to image-presentation trialing.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| image_identity | 0.1431 | 0.1456 |
| image_change | 0.5115 | 0.5081 |
| running_speed_bin | 0.2099 | 0.2080 |
| pupil_diameter_bin | 0.2094 | 0.2094 |
| trial_outcome | 0.2556 | 0.2524 |

---

## Step 9: Full Conversion and Validation
**Status**: IN PROGRESS

### Output Files
- `converted_data.pkl`: 4.7G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | TBD from papers | SDK/data-derived | 42147 | 42147 | yes(data) |
| Mean neurons/session | TBD from papers | SDK/data-derived | 148.405 | 148.405 | yes(data) |
| Subjects | 107 behavior-session mice in metadata | SDK/data-derived | 38 included ophys subjects | 38 | partial (ophys subset) |
| Sessions | TBD from papers | SDK/data-derived | 284 | 284 | yes(data) |
| Trials (total) | image-presentation based; paper analyzes image-by-image | SDK/data-derived | 963758 | 963758 | yes(data) |
| Trials/session (mean) | image-presentation based | SDK/data-derived | 3393.514 | 3393.514 | yes(data) |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: IN PROGRESS

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
| image_identity | 0.1630 | 0.1615 | above chance (0.0625) |
| image_change | 0.5563 | 0.5514 | above chance (0.5000) |
| running_speed_bin | 0.2504 | 0.2485 | above chance (0.2000) |
| pupil_diameter_bin | 0.2686 | 0.2675 | above chance (0.2000) |
| trial_outcome | 0.3010 | 0.2937 | above chance (0.2500) |

---

## Step 12: Critical Review 2
**Status**: IN PROGRESS

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| | |

All validation accuracies are above chance. Image identity is substantially above its 16-class chance level; running speed, pupil, and trial outcome are modestly above chance; image change is slightly above chance. No output is below chance, so there is no immediate evidence of gross misalignment or label corruption.

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
