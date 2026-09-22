# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (provided NWB data)
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verified: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124. Checkpoint verified with `ls -la /app/CONVERSION_NOTES.md`.

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `Dockerfile`
- `code/`
- `data/`
- `datapaper.pdf`
- `decoder.py`
- `docker-compose.yaml`
- `methodpaper.pdf`
- `methods.txt`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` | `code/VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Loads MATLAB structs/cells recursively into Python dictionaries/lists. |
| `process_one_sess` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING / CURATION | Groups probes by session, concatenates units, retains units with ephys and histology, applies classifier-derived good-unit lists, and separates units by hemisphere/major region. |
| `helper_get_neuron_id_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Intersects classifier QC indices with histology-bearing units and assigns hemisphere at Allen CCF ML coordinate 5700. |
| `sliding_histogram` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Counts spikes in half-open bins centered on requested times and divides by bin width to produce Hz. |
| `process_one_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Uses spike times already relative to go cue, clips them to the analysis window, computes firing rates, and saves trial/task/unit metadata. |
| `load_session` | `code/VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Concatenates preprocessed region files into a session-wide firing-rate array and neuron annotations. |
| `get_regular_trial_mask` | `code/VideoAnalysisUtils/functions_for_r2.py` and `population_decoding_utils.py` | CURATION | For the paper's video analyses excludes early-lick, auto-water, free-water, no-response, and stimulation trials. |
| `create_4fold_trial_type_mask` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Stratifies trials by instructed side and correct/error outcome, pooling rare error groups when needed. |
| `align_embedding_vecs_between_lims` | `code/Sherlock/align_embed_vecs.py` | PROCESSING | Aligns video-derived samples to go cue and uses the most recent frame in each target interval. |

### Notes
- This is electrophysiology, not calcium imaging; delta-F/F is inapplicable.
- The repository's canonical ephys preprocessing used classifier QC (`goodunits`) rather than thresholding the scalar QC columns directly. The supplied NWB units table must therefore be checked for the exported good-unit subset/quality fields in Step 2.
- Raw spike times in the original exported data were already go-cue-relative. The reference used 40-ms windows slid every 3.4 ms for the video paper. This conversion instead must use non-overlapping 50-ms bins over exactly -2.5 to +1.5 s because the decoder specification overrides that binning.
- The paper's `regular` mask cannot be reused wholesale: stimulation, early lick, and no-response are explicitly required decoder variables/classes. Only trials invalid for alignment or required source streams should be removed.
- Behavioral conventions in preprocessing: `correctness` is 1 correct/free-water, 0 error, -1 no response; instructed `trial_type` is mapped left=1/right=0. Actual lick choice must come from response/lick fields rather than instructed trial type.
- Repository README identifies the data as DANDI 000363 and states that archived code may be stale; primary preprocessing utilities and current Sherlock scripts were prioritized.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/dandiset.yaml` describes DANDI 000363 version 0.230822.0128 (53.6 GB, 28 mice, NWB).
- 174 NWB 2.x HDF5 files are grouped in 28 `sub-<id>/` directories; one file is one recording session.
- `/intervals/trials` has 264-800 rows/session and columns: start/stop time, 1-based trial number, globally unique trial UID, task, task protocol, instructed side, early-lick label, outcome, auto/free-water flags, and photostimulation onset/power/duration.
- `/acquisition/BehavioralEvents` provides absolute timestamps for presample, sample (auditory tone), delay, go cue, trial end, left/right licks, and photostimulation starts/stops. Go events have one row per NWB trial. Extra sample/delay events can occur, so trial association must be interval-based rather than positional for those streams.
- `/acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` exists in every session and contains `(x, y, DeepLabCut likelihood)` at nominal 3.4-ms camera intervals with absolute timestamps. Jaw and nose tracks also exist but are not requested. Camera 3 duplicates exist in three files and are not the canonical side camera.
- `/units` contains ragged absolute spike timestamps, unit/probe metadata, scalar QC metrics, `classification` (`good`/`unlabelled`), fine Allen CCF `anno_name`, and a neuron-by-trial `is_good_trials` stability mask. `classification == "good"` is the exported classifier-curated population and has nonempty anatomy.
- Spikes, behavioral events, video timestamps, and trial intervals share the same absolute session clock. Example session extent was 0-2491.8 s, with unit spikes and observation intervals on that clock.
- Tongue likelihood is strongly bimodal (example: only 10.5% of frames >=0.9), supporting the standard DLC likelihood threshold to define visibility; the precise threshold is checked against the methods in Step 3.
- Available requested variables are complete in all files: neural spikes, go events, auditory sample/tone events, photo-stimulation events/columns, instructed side, actual left/right lick timestamps, outcome, early lick, and tongue tracking.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw units; 69,453 classifier-`good` units |
| Neurons / session | Good units: 90-923 among usable sessions; median 390; mean 401.46 |
| Subjects | 28 |
| Sessions / subject | 3-10 (174 total files) |
| Trials (total) | 94,990 rows in NWB trial tables; 93,310 trials overlap ephys observation intervals in 173 sessions with curated units |
| Trials / session | Recorded/ephys-overlapping: 160-800; median 530; mean 539.36 |

One session (`sub-440958_ses-20190216T162508`) has 620 trials but `classification` and `anno_name` entirely NaN, hence zero classifier-curated/anatomically assigned units. It cannot form a valid neural decoder session and is a candidate for exclusion. Eight usable sessions have trial-table rows after all units' observation intervals end (1,060 rows total); these are behavior-only trailing trials and cannot be decoded from neural data. Four otherwise usable sessions contain some `is_good_trials=False` unit/trial pairs; no recorded trial has zero valid curated units. The trial-stability policy is deferred to Steps 3-5.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units | Data paper STAR Methods / QC white paper: “69,943 good units recorded across 173 behavioral sessions.” | 
| Neurons / session | Hundreds; median text extraction is truncated | Data paper Results: each session yielded simultaneous measurements from hundreds of neurons. |
| Subjects | 28 | Data paper Fig. 1 and method paper Methods. |
| Sessions / subject | Not explicitly tabulated | 173 sessions across 28 mice. |
| Trials (total) | Not explicitly reported | Per-session statistic given below. |
| Trials / session | Mean 476, range 130-785 (analysis-selected trials/sessions) | Data paper Methods. |
| Neural data time bin | 40-ms width, 3.4-ms stride (method paper); other data-paper analyses use 200-ms causal windows / 10-ms step | Method paper “binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms.” |
| Behavior data time bin | Video acquired at 300 Hz; code uses nominal 3.4-ms frames | Both papers and reference code. |
| Reward/correct rate | Mean 84%, range 65-99% in selected control/non-early trials | Data paper Methods. |
| Photostimulation fraction | Approximately 25%, randomly interleaved in 17 VGAT-ChR2 mice | Data paper Methods. |
| Task timing | 0.65-s sample, 1.2-s delay, 0.1-s go cue, 1.5-s answer period | Data/method papers; sample comprises three 150-ms tones separated by 100 ms. |
| Curated cluster fraction | 25.9% of Kilosort2 clusters | Data paper / QC white paper. |


### Processing Details
- Trial epochs in the papers are aligned to go cue. Canonical time landmarks are approximately sample `[-1.85,-1.2]` s, delay `[-1.2,0]` s, and response `[0,1.5]` s.
- The video paper bins firing rates with overlapping 40-ms windows at 3.4-ms stride. The requested decoder explicitly overrides this with 50-ms-width bins and a -2.5 to +1.5 s window.
- Reference code treats bins as half-open intervals. It aligns video/markers by go cue and selects the last source frame in each target interval. This supports sampling behavioral state at each decoder bin center from the nearest preceding camera frame.
- The method paper uses side-view video only. DeepLabCut tracks tongue, jaw, and nose. It removes marker velocity outliers (>5 SD) and imputes them; occluded tongue positions were set to the mean for its continuous regression. The present categorical output instead explicitly requires an unobserved class, so low-confidence tongue samples must remain class 3 rather than be imputed.
- Photoinhibition is typically during the last 0.5 s of delay and ends before go cue; representing the actual event intervals preserves exceptional timing rather than hard-coding this nominal window.

### Curation Steps

**Neuron curation rules**:
Kilosort2 clusters were classified using five region-specific logistic-regression classifiers trained on 15 QC metrics and blinded manual labels. Individual-metric thresholding is specifically discouraged because good and unlabelled distributions overlap. The NWB `classification == "good"` flag is therefore the authoritative inclusion rule. The method paper additionally excludes <2-Hz neurons only for video-prediction analyses; this task decodes behavior from neural activity and the original data paper's broad curated population is the applicable rule, so no extra firing-rate cutoff is planned.

**Trial curation rules**:
The papers' movement analyses excluded photoinhibition, free/auto-water, early-lick, and ignore/no-response trials. Those exclusions cannot be copied here because photostimulation, early lick, no lick, and ignore are explicitly required inputs/outputs. Trials will instead be retained unless they lack the go-aligned window, required behavior stream, or valid neural recording. The NWB `is_good_trials` probe-stability mask requires special handling in Steps 4-5.

### Decoders Trained
| Decoded variable | Accuracy |
| Neural population choice | Data paper Fig. 6: chance before instruction and roughly 0.7-0.85 near go depending on area (read graphically; 200-neuron pseudo-populations, not directly comparable architecture). |
| Choice from behavioral video | Method paper: ROC-AUC 0.51 +/- 0.06 before sample, 0.66 +/- 0.12 over late sample/delay, 0.99 +/- 0.01 late response (106 sessions). |
| Outcome, early lick, tongue-y class from neural activity | Not reported in either reference paper. |

The QC white paper reports regional classifier ROC-AUC >0.9 and reiterates that classifier output—not ad hoc scalar thresholds—defines good units.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Curated units | Load classifier-produced `goodunits`; do not threshold individual QC metrics | `units/classification == "good"` gives 69,453 units in 173 usable sessions | 69,943 units, 173 sessions, 25.9% of Kilosort clusters | Use the release's explicit classifications. The 490-unit paper/release discrepancy cannot be corrected from available fields and is attributed to the published snapshot/export; never synthesize or reclassify units. |
| Session with no QC/anatomy | Requires a classifier QC list and histology | One of 174 files has all-NaN classification/anatomy | 173 analyzed sessions | Exclude that file; exact session count then matches. |
| Neural trial extent | Raw MATLAB spike-time arrays define recorded trials | Eight files contain 1,060 trailing behavior-table rows after every unit's observation intervals; `is_good_trials.shape[1]` and ragged `obs_intervals` both identify the ephys trial count | Paper analyzes trials with neural recordings | Exclude behavior-only trailing rows. This corrected the preliminary Step 2 count from 94,370 to 93,310 recorded trials. |
| Within-recording validity | Legacy exported preprocessing does not expose this mask | Four sessions have some false neuron/trial stability flags; 509 trials are not valid for every curated neuron | QC paper rejects unstable recordings/periods | Require a trial to be valid for every retained curated neuron, yielding a rectangular, artifact-free session tensor while retaining the published unit population. |
| Trial exclusions | Video paper `regular` mask removes stimulation, early lick, water, and ignore trials | These labels/events are present and internally consistent | Same exclusions for the papers' questions | Do not apply those exclusions because they would delete required decoder classes/inputs. Exclude only neural/alignment invalidity. |
| Binning | 40-ms sliding window at 3.4-ms stride in method paper code | Absolute spikes support arbitrary edges | User requires 50-ms width from -2.5 to +1.5 s | Use 80 non-overlapping half-open bins, divide counts by 0.05 s. |
| Tone onset under early-lick replay | Raw code aligns task streams but does not export sample onset | `sample_start_times` can contain 1-14 epochs per trial; extra entries are concentrated in early-lick trials | Early licking replays the sample/delay epoch | Use the first `sample_start_times` event between trial start and go: “tone onset” denotes the trial's initial instruction onset. Interval matching avoids positional mismatch while preserving the genuinely longer elapsed time on replay trials. |
| Tongue occlusion | Method paper imputes occluded tongue to mean for continuous movement-to-neural regression | DLC likelihood is bimodal and available for every frame | Requested output explicitly has class 3 “not visible” | Preserve occlusion as class 3 using DLC likelihood <0.9; compute y percentiles only from visible samples. |
| Trial statistics | `regular` masks are analysis-specific | Recorded data: 93,310 trials; 63,805 hit, 15,458 miss, 14,047 ignore; 10,708 early; 19.47% photostim | Mean 476 trials and 84% correct after paper-specific selection/exclusions | Broader counts/rates are expected because required classes cannot be excluded. Reapplying the available regular mask gives pooled 81.6% hit among hit/miss; the paper's 84% reflects its analysis selection/version. |

Final cross-source understanding: all clocks in NWB are absolute; go alignment is done by subtracting each trial's go timestamp. Units are selected only by the provided regional classifier label. Ephys trial extent comes from observation intervals, and a retained trial must pass the curated units' stability mask. Requested rare/invalid-behavior labels remain valid decoding targets, so paper-specific behavioral trial exclusions are intentionally not used.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `/units/spike_times`, `_index` | `neural` | Select `classification==good`; count absolute spikes in 80 half-open 50-ms bins with edges `go + [-2.5,...,1.5]`; divide by 0.05 to Hz; float32 `(neurons,80)` | `sliding_histogram`, `process_one_area` | No smoothing/overlap because decoder task specifies 50-ms width. |
| First `/acquisition/BehavioralEvents/sample_start_times` within trial start-to-go | `input[0]` | At every bin center, `absolute_bin_center - tone_onset`; float32 seconds | Event alignment logic in preprocessing/marker alignment | Continuous, time-varying as explicitly requested. |
| Trial `photostim_onset` + start time and `photostim_duration` | `input[1]` | 1 where bin center is in half-open stimulation interval, else 0 | Raw `task_stimulation` alignment in `process_one_sess` | Uses actual interval, not nominal task timing. |
| `outcome` + `trial_instruction` | `output[0]` choice | ignore -> no lick; hit -> instructed direction; miss -> opposite direction | Trial-type/correctness logic in `create_4fold_trial_type_mask` | Codes left=0, right=1, no lick=2. This agrees with first response-period lick events on 99.64% of retained trials; trial labels are authoritative for exceptional/ambiguous lick trains. |
| `/intervals/trials/outcome` | `output[1]` outcome | Map ignore=0, miss=1, hit=2 | Reference `correctness`: -1/0/1 | Per-trial label broadcast across 80 timepoints to coexist with tongue time series. |
| `/intervals/trials/early_lick` | `output[2]` early lick | `no early`=0, `early`=1 | `early_lick_trials` in preprocessing | Per-trial label broadcast across time. |
| Camera0 side TongueTracking `(y, likelihood)` | `output[3]` tongue y-position | Session-wide 40th/60th percentiles from frames with likelihood >=0.9; sample nearest preceding frame at each bin center; visible y maps to 0/1/2, likelihood <0.9 maps 3 | Reference uses side camera, marker alignment by preceding frame; method paper DLC processing | Percentiles exclude invisible coordinates because those coordinates are tracking artifacts. Boundary convention: `<q40`, `q40<=y<=q60`, `>q60`. |
| `/general/subject/subject_id` / folder ID | `subjects`, `subject_idx` | Stable sorted subject identifiers and integer indices | N/A | Folder and NWB IDs will be asserted consistent. |
| `/units/anno_name` for good units | `brain_regions`, `brain_region_idx` | Global sorted list of fine Allen CCF annotations; integer lookup per unit | `helper_get_neuron_id_area`; reference CCF pooling | Fine actual recording location is preferred to probe target; all curated units have annotation. |
| Trial/unit observation and stability fields | trial/session curation | Exclude all-NaN-QC session; restrict to first `is_good_trials.shape[1]` recorded trials; retain trials for which all curated units are good | QC paper stability policy | Leaves 173 sessions, 69,453 units and 92,801 trials before any alignment-edge rejection. |

### Key Decisions
1. **Mixed outputs are all 2D**: choice/outcome/early labels are broadcast through time, producing integer `(4,80)` arrays so the time-varying tongue output can share one target array. This preserves their per-trial semantics.
2. **Do not apply the paper's regular-trial mask**: those exclusions directly conflict with required photostimulation, early-lick, ignore, and no-lick variables.
3. **Classifier flag is final QC**: do not recreate regional logistic classifiers or add a 2-Hz filter; the latter was specific to predicting neural activity from video.
4. **Output numerical types**: neural and inputs are float32; outputs and indices are integer. Float32 avoids precision loss and validator warnings despite the approximately 12-GB full neural payload.
5. **Window convention**: metadata offsets are bin edges (-2.5,+1.5); sample times are centers (-2.475,...,+1.475); all state sampling and plotted alignment uses centers.
6. **All available experimental variables considered**: task/protocol, instruction, auto/free water, lick events, sample/delay/go/trial events, jaw/nose/tongue tracks, and photostimulation fields were inventoried. Only the decoder-specified variables are emitted; instruction/outcome support actual choice derivation, and auto/free-water flags are retained in per-session metadata counts rather than used as exclusions.
7. **Session order/sample mode**: lexicographic source path order; `--sample` takes the first two usable sessions deterministically.

### Planned Sanity Checks
- [ ] Directly reload raw NWB spikes for selected `(session,trial,unit)` triples and require `np.allclose(raw_count/0.05, converted_rate)`.
- [ ] Directly reconstruct tone-time and photostimulation state from raw events/columns for selected trials and require `np.allclose` to converted inputs.
- [ ] Directly reconstruct choice/outcome/early and raw tongue frame class for selected trials/timepoints and require `np.allclose` to outputs.
- [ ] Assert 80 bins, finite values, rates nonnegative and multiples of 20 Hz, consistent neuron count within session, and at least two retained trials.
- [ ] Compare 173 sessions, 28 subjects, 69,453 units, and 92,801 pre-edge-filter trials to source inventory; explain paper-release differences.
- [ ] Compare outcome/early/stimulation distributions to independently scanned raw counts and visually inspect aligned neural, inputs, and tongue discretization for two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with deterministic `--full` (default), `--sample`, and `--show-processing` modes. Syntax compilation and CLI help completed successfully. The code validates subject IDs, anatomy, observation-interval counts, tone-event containment, video density, shapes, finiteness, and the 20-Hz quantization implied by 50-ms spike counts.

Code inefficiencies identified:
Naive unit x trial loops and repeated HDF5 access would dominate runtime; loading all 53.6 GB or materializing per-spike trial assignments would inflate memory. Full output is intrinsically about 12 GB as float32.

Code speedups added:
For each unit, one vectorized `searchsorted` call bins all trial edges. Small metadata arrays are read once/session, raw spikes are sliced once/unit, and only one session-rate tensor is added at a time. Trial arrays are views of the session buffers until pickle serialization. Timing and per-session MiB are printed.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 session-units |
| Neurons / session | [459, 375] |
| Subjects | 1 represented (28-name global vocabulary) |
| Sessions / subject | 2 for subject 440956 |
| Trials (total) | 527 |
| Trials / session | [368, 159] |
| Time from tone onset range | [-0.6, 5.7] s |
| Photostimulation range | [0, 1] |
| Choice distribution | [0.4516, 0.4080, 0.1404] left/right/no lick |
| Outcome distribution | [0.1404, 0.2979, 0.5617] ignore/miss/hit |
| Early-lick distribution | [0.9431, 0.0569] no/yes |
| Tongue distribution | [0.0526, 0.0290, 0.0610, 0.8574] low/middle/high/not visible |
| Neural range / mean | 0-360 Hz / 5.705 Hz |

### Processing Plots Review
Both required plots were visually inspected. Neural heatmaps show plausible sparse 20-Hz increments and a population response around go; tone markers occur at the expected -1.85 s on example trials; stimulation is binary; per-trial outputs are constant; tongue classes change only on visible frames and respect the plotted session thresholds. No temporal shift or percentile inversion was observed.

Initial validation warned that the last trial of sample session 2 was entirely zero. Direct raw inspection found every curated spike train ended at 1107.44 s while that trial began at 1109.39 s, despite the NWB observation table including it. The converter now excludes all-zero population windows as missing neural coverage. Re-conversion leaves 159 rather than 160 trials and validation reports no errors or warnings.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Vectorized all-trial `searchsorted` per unit | Avoids about 37 million Python unit/trial loops in full mode |
| Read each unit spike slice and session metadata once | Avoids repeated 53.6-GB source I/O |

| Step | Time / Session | Estimated Total Time |
| Inventory | 2.80 s total | 2.80 s |
| Convert + plots (sample) | 1.13 s/session; plots add overhead | Without plots, conservatively 5-7 min based on 11.06-GiB neural payload |
| Pickle write | 0.08 s for 0.069 GiB | Approximately 15-30 s for estimated 11.1-GiB pickle |

`sample_data.pkl` is 73,704,536 bytes. Neural/input/output dtypes are float32/float32/int8, all trials have 80 timepoints, and `/app/verification_sample_out.txt` ends with successful completion and no warnings.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None after excluding one source off-by-one all-zero terminal trial

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Lick direction choice | 0.7275 | 0.6198 |
| Outcome | 0.7569 | 0.6527 |
| Early lick | 0.8343 | 0.7554 |
| Tongue y-position | 0.6741 | 0.5240 |

Training completed on GPU. Mean loss decreased monotonically from 13.8644 (epoch 1) to 0.5784 (epoch 200); test loss was 0.7256. Every validation balanced accuracy exceeds uniform chance (0.3333, 0.3333, 0.5, 0.25 respectively), including the rare early-lick class.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 10.966 GiB (173 sessions)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 | classifier `goodunits` | 69,453 `classification==good` | 69,453 | Data/release match; paper differs by 490 |
| Mean neurons/session | not reported | computed from good units | 401.46 | 401.46 | Yes |
| Subjects | 28 | all subjects | 28 | 28 | Yes |
| Sessions | 173 | 173 session list | 173 usable of 174 files | 173 | Yes |
| Trials (total) | mean 476/session after paper exclusions | analysis-dependent filters | 94,990 all-file table; 93,310 usable-session ephys-leading | 90,253 | Expected broader task-specific curation |
| Trials/session (mean) | 476 after excluding stim/early/ignore/water | analysis-dependent | 545.92 all-file table; 539.36 usable table; 530.17 ephys-leading | 521.69 | Expected task difference |
| Tone elapsed-time range (s) | sample onset precedes go, usually about -1.85 s | aligned event timestamps | [-1.5, 11.9] after conversion filters | [-1.5, 11.9] | Yes |
| Photostimulation range | binary; about 25% stimulated trials | trial onset/duration | [0,1] | [0,1] | Yes |
| Choice distribution | correct performance 84% (not class fraction) | instruction + outcome | left/right/no lick | [0.428,0.422,0.150] | Internally consistent |
| Outcome distribution | correct performance 84% under paper filters | -1/0/1 correctness | ignore/miss/hit | [0.150,0.166,0.684] | Broader requested trials explain difference |
| Early-lick distribution | early trials excluded from paper analyses | label field | no/yes | [0.884,0.116] | Yes |
| Tongue class distribution | not reported | side-camera tracking | requested 4 classes | [0.062,0.032,0.065,0.841] | Plausible; invisibility includes occlusion and temporal gaps |

The full conversion took 177.14 s, well below the 15-minute limit. The validator took 21.65 s and reported: “Data format is valid, no errors or warnings.” It independently confirmed 80 bins for every trial, 2 inputs, 4 outputs, finite/ranged values, 28 subjects, 293 regions, and the counts above. Spot inspection included the first, middle, and final sessions and confirmed matching list lengths, neuron counts, trial shapes, finite values, and region-vector lengths.

The first full attempt exposed that several NWB files store camera timestamps only in within-trial segments rather than densely through inter-trial gaps. Treating every gap as invalid caused one session to have fewer than two trials. It was corrected so a source frame is contemporaneous when the preceding frame is within 10 ms and missing/noncontemporaneous frames map to tongue class 3. Critical Review 1 then found that requiring even one video frame still contradicted that policy. Removing this unintended filter recovered 754 neurally valid trials, mainly in two sessions. The final conversion contains all 92,801 stable trials initially eligible by time events, minus 2,548 entire-population all-zero 4-s windows beyond actual spike coverage, for 90,253 trials.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output logs**: `verification_full_out.txt` says “Data format is valid, no errors or warnings.” It confirms 173 sessions, 90,253 trials, fixed 80-bin time axes, valid ranges and all categorical values. No warning requires waiver.
2. **Independent raw `np.allclose` checks**: `cache/raw_sanity_checks.py` does not import the converter. It directly reloads four NWBs (first, last, and both camera-segmented edge cases), independently reconstructs trial eligibility, then checks first/middle/last converted trials. Exact/allclose comparisons cover raw spike histograms for three neurons over all 80 bins, tone elapsed time, photostimulation state, choice, outcome, early lick, tongue class, and complete brain-region vectors. All checks passed; output is in `cache/raw_sanity_out.txt`.
3. **Neural sanity**: every rate is finite, nonnegative, and a multiple of 20 Hz; no retained 4-s population array is entirely zero. Direct raw histogram comparisons passed with zero tolerance.
4. **Input sanity**: shapes are exactly `(2,80)`, photostimulation is binary, and elapsed-time traces advance by 0.05 s/bin. Raw input comparisons passed. The extrema (-1.525 to 11.8943 s before validator rounding) were traced to legitimate raw trials: one has tone-to-go -0.95 s and another unusually long -10.4193 s; neither is a mapping error.
5. **Output sanity**: shapes are `(4,80)`, scalar outputs are constant within trial, requested ranges are exact, and raw label/tongue comparisons passed. The session percentile convention and class boundaries were rechecked.
6. **Key statistics**: raw/converted reconciliation is 94,990 rows over all 174 files; excluding the 620-row no-good-unit file gives 94,370 rows; ephys-leading length gives 93,310; all-good-unit stability gives 92,801; population spike coverage gives 90,253. Units reconcile to 69,453 in 173 sessions and 28 mice. Paper differences are explained by paper-version unit counts and deliberate task-specific trial inclusion.
7. **Edge cases**: checked shorter stability matrices, all-NaN unit-QC session, false stability entries, optional numeric `N/A` byte strings, long sample-to-go intervals, terminal/interleaved all-zero neural windows, camera segment gaps, absent video windows, low-likelihood/NaN tongue coordinates, half-open bin endpoints, percentile equality, and sessions with very small trial counts. Every retained session has at least two trials (minimum 159 after the video-policy fix; maximum 800).

### Reference Code Comparison
| Stage | Reference | Conversion | Assessment |
|-------|-----------|------------|------------|
| Data loading | MATLAB/NWB-derived absolute event and spike timestamps | Direct h5py reads of corresponding NWB arrays | Same source semantics |
| Neuron/trial filtering | Regional classifier `goodunits`; trial filters depend on analysis | `classification=='good'`, per-unit stability and real spike coverage; retain stim/early/ignore/water required here | Same QC; decoder-task trial exception justified |
| Temporal alignment | Subtract go cue; interval-aligned markers | Absolute go plus -2.5..+1.5 s edges/centers | Same alignment |
| Binning | `sliding_histogram`, half-open windows; method analysis used 40 ms/3.4 ms | Vectorized half-open nonoverlapping 50-ms bins | Width/stride intentionally follow decoder specification |
| Input construction | Sample and laser events retained as task timing | First within-trial sample onset elapsed time; trial-relative laser onset/duration at centers | Direct event mapping |
| Output construction | Correctness/instruction labels and preceding video-frame alignment | Choice derived from outcome/instruction; direct outcome/early; preceding tongue frame with visibility class | Same labels/alignment, categorical task transform |

### Issues Found and Resolved
- **Iteration 1 — missing video**: the first full attempt rejected camera timestamp gaps; changed gaps to class 3 and reran.
- **Iteration 2 — entirely absent trial video**: aggregate review found 791 pre-neural-filter trials were still being dropped for having no camera frame. Removed the video-presence filter, recovering 754 trials with real spikes; the other 37 also lacked population spikes. Sample/full conversion, sample/full validation, all aggregate checks, and all raw comparisons were rerun and passed.
- **Raw optional-number parser**: the first independent audit expected float NaN but NWB uses byte string `N/A`. This was a test-script issue; parsing was corrected and all checks rerun.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Mean training loss fell from 18.095259 (epoch 1) to 0.659051 (epoch 200); test loss was 0.655269. Training used CUDA with 72,149 training and 18,104 validation trials and completed normally.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Lick direction choice | 0.7063 | 0.6760 | chance 0.3333 |
| Outcome | 0.6944 | 0.6624 | chance 0.3333 |
| Early lick | 0.7992 | 0.7521 | chance 0.5000 |
| Tongue y-position | 0.6771 | 0.6241 | chance 0.2500 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Expectation from papers |
|----------|------------------------------|--------|--------------------|-------------------------|
| Lick direction choice | 0.6760 | 0.3333 | 2.03x | Data paper neural binary choice is graphically ~0.70-0.85 near go depending on area; method paper video-to-binary-choice AUC is 0.51 pre-sample, 0.66 late sample/delay, 0.99 late response |
| Outcome | 0.6624 | 0.3333 | 1.99x | Not reported |
| Early lick | 0.7521 | 0.5000 | 1.50x (just above threshold) | Not reported; reference analyses exclude early trials |
| Tongue y-position | 0.6241 | 0.2500 | 2.50x | Not reported as a categorical neural decoder; method paper predicts neural activity from continuous video instead |

All outputs are above chance and none is below the requested 1.5x-chance investigation threshold (early lick is 1.5042x). Choice is 0.024 below the lowest graphical near-go data-paper estimate, so it was investigated rather than treated as equivalent: three raw trials in each of four sessions were rechecked; choice derivation from instruction/outcome was exact; all labels vary substantially; go alignment and spike bins matched raw data; and sample/prediction plots showed no time shift. The task here is intrinsically broader than the paper's plotted binary, correct/control-trial, fixed-time classifier: it includes a third no-lick class plus miss, early, and photostimulation trials and reports balanced accuracy over the full time-varying target. Those explicit target/population differences—not an unexplained conversion discrepancy—account for the small difference. The method paper's 0.99 late-response number uses visible movement video to decode binary choice during the response itself, while this decoder uses neural activity and three choice classes across the full window; its late sample/delay video AUC (0.66) is close to the achieved 0.676.

Train/validation ratios were 1.045 (choice), 1.048 (outcome), 1.063 (early lick), and 1.085 (tongue), all far below the 1.5x overfitting criterion. Test loss (0.655269) was also slightly lower than final training loss (0.659051). Visual review of `sample_trials.png` confirmed fixed alignment and categorical traces, while `predictions.png` showed predictions tracking scalar task labels and time-varying tongue visibility without a systematic lag.

### Issues Found and Resolved
- No new conversion issue was found. The required low-accuracy debugging checks—raw labels, alignment plots, class variation, QC filtering, and reference-processing comparison—had all passed. Therefore no reconversion/retraining iteration was warranted.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading example, mappings, curation, statistics, and reproduction commands
- [x] cache/ folder created with independent audit script/output and README_CACHE.md
- [x] All required files are present; analysis artifacts are in cache; conversion, validation, training, processing, and prediction outputs are retained at project root
