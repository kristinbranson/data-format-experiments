# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P dataset (provided project files)
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`
- `tutorials/`
- `whitepaper.pdf`

Environment check: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124 imported successfully. Checkpoint confirmed with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `BehaviorOphysExperiment.from_nwb` | `code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py` | LOADING | Constructs synchronized behavior + ophys experiment from NWB; loads trials, stimuli, running, eye tracking, timestamps, cells, DFF and events. |
| `OphysTimestamps.from_nwb` | `code/allensdk/brain_observatory/behavior/data_objects/timestamps/ophys_timestamps.py` | LOADING | Reads frame timestamps from the DFF response series. |
| `OphysTimestamps.validate` | same | CURATION | Truncates extra Scientifica sentinel timestamps to DFF length and errors when traces exceed timestamps. |
| `CellSpecimens.__init__` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/cell_specimens.py` | CURATION | Excludes `valid_roi == False` by default and filters/reorders every trace table to the curated cell table. |
| `DFFTraces.from_nwb` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/traces/dff_traces.py` | LOADING | Reads already-computed DFF from NWB and transposes time-by-ROI storage to ROI traces. |
| `Events.from_nwb` / `filter_events_array` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/events.py` | PROCESSING | Reads detected event traces and creates SDK `filtered_events` with a causal half-Gaussian (scale 2/31 s, 20 samples). |
| `Trials.from_nwb` | `code/allensdk/brain_observatory/behavior/data_objects/trials/trials.py` | LOADING | Reads the NWB trial table and preserves trial outcome/timing columns. |
| `Trial._get_trial_data` | `code/allensdk/brain_observatory/behavior/data_objects/trials/trial.py` | PROCESSING | Defines mutually exclusive go/catch/aborted/auto-rewarded and hit/miss/false-alarm/correct-reject logic. |
| `Trials._get_trial_bounds` | same trials module | PROCESSING | Defines contiguous native trials from each trial-start to the next trial-start (last to session end). |
| `BehaviorSession.stimulus_presentations` | `code/allensdk/brain_observatory/behavior/behavior_session.py` | LOADING | Exposes presentation timing, image identity, omissions and `is_change`; active behavior is the `change_detection` block. |
| `get_running_df` | `code/allensdk/brain_observatory/behavior/data_objects/running_speed/running_processing.py` | PROCESSING | Recomputes cm/s from encoder voltage, corrects wraps/outliers, and applies the default 3rd-order 4-Hz Butterworth low-pass filter (described publicly as 10-Hz filtering). |
| `process_eye_tracking_data` / `filter_on_blinks` | `code/allensdk/brain_observatory/behavior/eye_tracking_processing.py` | PROCESSING | Computes pupil area, marks area outliers/failed fits plus two adjacent frames as likely blinks, and replaces affected pupil values with NaN. |

### Notes
- The supplied ophys NWBs contain synchronized streams, so use `BehaviorOphysExperiment.from_nwb` semantics (or direct NWB reads that reproduce them) rather than reconstructing synchronization.
- DFF does **not** need to be recomputed: it is stored in NWB after the Allen processing pipeline. Detected events are also stored; the SDK derives only the optional smoothed `filtered_events` view at load time.
- Ophys frame timestamps are the required master timebase. Behavioral streams have their own synchronized timestamps and therefore need interpolation/nearest-appropriate sampling onto ophys frames.
- Default cell curation is `valid_roi == True`; there is no electrophysiology unit-quality filtering because this is two-photon calcium imaging.
- Trial category definitions imply the requested retained set is exactly rows with `go | catch`, equivalently excluding aborted and auto-rewarded trials. Outcome classes are hit, miss, false alarm, and correct reject.
- Running speed should use the already processed `running_speed`, not `raw_running_speed`. Pupil values affected by blinks are NaN in the processed `pupil_area`; downstream interpolation must not silently use `pupil_area_raw`.
- Step 1 was restricted to `/app/code` as required.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` contains 284 per-imaging-plane NWB 2.0 files (246.79 GiB total; 223 MiB to 3.23 GiB each).
- `/app/data/visual-behavior-ophys-1.1.0/project_metadata/` contains four CSV tables: 1,936 experiments, 703 ophys sessions, 4,782 behavior sessions, and 133,066 experiment-cell rows for the full release. The locally downloaded NWBs form a complete 284-experiment subset of those tables.
- The 284 experiment files correspond to 247 unique ophys/behavior sessions. The difference is seven multiscope sessions represented by multiple simultaneous imaging planes; each NWB is a distinct neural population and is therefore a natural target-format session, although behavioral trials repeat across simultaneous planes.
- NWB `processing/ophys/dff/traces` and `processing/ophys/event_detection` store time-by-cell float64 arrays with shared ophys timestamps. Across files there are 48,284--149,508 frames per experiment and 4--666 valid cells. All downloaded NWB cell-table entries have `valid_roi=True`.
- NWB `intervals/trials` stores start/stop/change time, initial/change image, go/catch/aborted/auto-rewarded flags and the four retained outcomes. `go | catch` gives 85,230 eligible experiment-trials; raw tables also contain 85,842 aborted and 815 auto-rewarded trials.
- Image presentations are interval tables with image name/index, start/stop times, omission, `is_change`, `is_sham_change`, activity and trial ID. Each image set has eight identities plus `omitted`; image-set A and B labels differ. Natural-movie and spontaneous blocks are also present but fall outside trial/task intervals.
- `processing/running/speed` stores synchronized processed cm/s and timestamps; the raw/unfiltered stream is also present.
- `acquisition/EyeTracking` stores synchronized pupil/eye/corneal ellipse area, width, height, angle and timestamps. Three experiment files (795953296, 806456687, 833631914) lack eye tracking; other sessions contain up to 272,837 samples.
- Subjects and experiment metadata are in both NWB (`general/subject`, `general/metadata`) and CSV. The selected experiments cover 38 mice, VISp (261 experiments) and VISl (23), and three Cre lines (Slc17a7: 153, Sst: 85, Vip: 46).
- No README is present under `/app/data`; organization and variable semantics are supplied by the NWB schema and project metadata tables.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 42,147 experiment-cell observations; 16,337 unique tracked `cell_specimen_id` values |
| Neurons / session | 4--666; mean 148.40; median 66 |
| Subjects | 38 |
| Sessions / subject | 4--45 experiment NWBs; mean 7.47 (247 unique behavior/ophys sessions overall) |
| Trials (total) | 171,887 raw experiment-trials; 85,230 requested go/catch trials; outcome counts sum exactly to retained count |
| Trials / session | raw 399--1,241 (mean 605.24, median 618); retained 39--412 (mean 300.11, median 293) |

Additional raw-data checks: retained outcomes are hit 15,936, miss 58,602, false alarm 932, and correct reject 9,760. Ophys median frame intervals span 32.29--93.22 ms (single-plane ~31 Hz; multiplane ~10.7 Hz). Every session has running data; 281/284 have eye tracking. Direct NWB counts match the project metadata counts exactly.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (paper neural subset) | 8,619 excitatory + 470 Sst + 1,239 Vip = 10,328 | Paper: familiar, multiplane neural analysis contained these three class counts. | 
| Neurons / session | Not tabulated | Paper reports imaging-plane/session counts by class, not per-session cell distributions. |
| Subjects | 82 behavioral; neural subset 9 excitatory, 6 Sst, 9 Vip mice | Paper: “376 imaging sessions from 82 mice”; familiar multiplane neural subset counts reported separately. |
| Sessions / subject | Behavioral mean 376/82 = 4.59; neural imaging planes: 21 excitatory, 15 Sst, 21 Vip | Paper abstract/results and neural dataset description. |
| Trials (total) | Not tabulated | Trial generation and categories are described, but no global trial count is given. |
| Trials / session | At least 100 hit/miss changes required during transition-ready training; imaging session total not tabulated | Whitepaper transition criteria. |
| Neural data time bin | Native ~31 Hz single-plane or ~11 Hz/plane multiplane; paper interpolated event-triggered activity to 30 Hz | Whitepaper synchronization; paper Neural data methods. |
| Behavior data time bin | Images 250 ms followed by 500 ms gray (750 ms presentation interval); eye/behavior acquired ~30 Hz | Whitepaper task and synchronization sections. |
| Reward rate | No global rate; SDK engagement threshold is 2 rewards/min, while paper used 1 reward/120 s OR 1 lick bout/10 s | Whitepaper behavior metrics; paper Task engagement methods. |
| Catch frequency | Approximately 12.5% in later natural-image sessions | Whitepaper: balanced transition sampling pushed catch probability to ~12.5%. |
| Omission frequency | 5% of repeat presentations; changes and immediately preceding images never omitted | Whitepaper/paper task description. |
| Change timing | Truncated exponential 2.25--8.25 s, actual mean ~4.2 s after flash alignment | Whitepaper trial structure. |
| Response window | 150--750 ms after non-display-lag-compensated change/sham-change time | Whitepaper behavior metrics. |


### Processing Details
- All clocks were hardware synchronized through one 100-kHz digital I/O board. Ophys movies were acquired at ~31 Hz (single plane) or ~11 Hz/plane (multiplane); eye and behavior at ~30 Hz.
- The study used **detected calcium events**, not recomputed DFF, because event inference removes slow GCaMP6f decay. FastLZero fits exponentially decaying events with L0 regularization; minimum-event/noise factors were 2.0 at 31 Hz and 2.6 at 11 Hz.
- For event-triggered analyses, paper code isolated events at native ophys timestamps and linearly interpolated to a common 30-Hz relative timebase. Running speed was processed the same way. This is the clearest reference precedent for a common decoder bin size.
- Trial-native task time is a continuous 250-ms image / 500-ms gray sequence. The paper often analyzed 750-ms image-presentation intervals and used a 50-ms visual latency for response-window summaries, but the requested conversion requires full trials and time-varying image/change labels.
- Running uses the processed SDK trace after voltage unwrap, wrap clipping, z-score >=10 transient removal, cm/s conversion and low-pass Butterworth filtering.
- The pupil’s major ellipse axis reflects pupil diameter. Processed pupil values exclude failed/outlier fits: missing pupil/eye fits or area z-score >3, plus two frames on each side, are marked likely blinks and set to NaN. Raw outlier values should not be substituted.
- Paper decoding used neural activity in the first 400 ms after presentations, random forests, and 5-fold cross-validation. Change-vs-repeat accuracy in Figure 6A rises from roughly 52% to 65% as neuron count increases; hit-vs-miss Figure 6C is roughly 52%--74% depending on class, strategy, and neuron count. Exact numeric values are not tabulated in the text.

### Curation Steps

**Neuron curation rules**:
Use released, valid ROIs after the Allen segmentation, overlap/union filtering, demixing, neuropil subtraction, event inference, and experiment/container QC. The whitepaper reports extensive experiment QC (sync integrity, motion, z-drift, task performance, hardware failure, etc.); the NWBs already passed release QC. The paper additionally selected familiar, multiplane recordings for its primary neural analyses, but that restriction served its strategy question rather than the requested general decoder.

**Trial curation rules**:
The reference paper explicitly used active behavioral sessions and did not analyze passive viewing. Per the decoder specification, keep go and catch trials and exclude aborted/reset and auto/free-reward trials. Outcomes are hit, miss, false alarm, and correct rejection. Trial times and category flags should be accepted from the synchronized released NWB rather than re-derived.

### Decoders Trained
| Decoded variable | Accuracy |
| Image change vs immediately preceding repeat | Figure 6A approximately 52%--65% across 5--80 cells; no numeric table |
| Hit vs miss | Figure 6C approximately 52%--74% across cell classes/strategies/counts; no numeric table |

Reference-selection caveat: the paper's behavioral cohort (376 sessions/82 mice) exceeds the locally supplied subset (247 unique recordings/38 mice). Its main behavioral analysis used all **active** sessions across Familiar, Novel 1, and Novel >1 and both image-set assignments; passive sessions were expressly not analyzed. Its primary neural subset was narrower (familiar, multiplane only). For this task, active-session selection matches both the task framing and behavioral outcomes, while retaining all experience levels, rigs, areas, depths, and valid cells maximizes appropriate decoder data.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neural representation | NWB exposes DFF, raw detected `events`, and SDK-only smoothed `filtered_events` | Every file has DFF and event detection on identical ophys timestamps | Study used detected calcium events to remove slow GCaMP decay | Use raw released event magnitudes, not DFF and not the visualization-only half-Gaussian filtered events. |
| Cell filtering | `CellSpecimens` defaults to `valid_roi=True` only | All 42,147 local experiment-cell rows are valid; table and trace dimensions agree | Release includes segmentation/demixing/ROI/experiment QC; paper uses released cells | Preserve all released valid cells; add no unreferenced activity threshold. |
| Session selection | Metadata distinguishes active behavior and passive viewing | 202/284 experiments are active; 82 passive. Passive outcomes are largely forced misses/correct rejects and cannot represent animal choice | Paper used all active sessions across experience levels and expressly excluded passive | Keep active experiments only. |
| Missing pupil | SDK returns no eye table when absent | 3 active files lack EyeTracking entirely | Eye pipeline describes filtering but no valid imputation for absent recordings | Exclude these 3 experiments because pupil diameter is a required decoder output. Final planned set: 199 experiments, 171 unique recordings, 38 mice. |
| Pupil variable | SDK stores ellipse half-axes and processed pupil area; area = pi * max(width,height)^2 | Direct equality check confirms area uses the major half-axis; blink frames are NaN | Whitepaper says major ellipse axis reflects pupil diameter and filtered values remove blink/outlier fits | Define diameter as `2 * max(pupil_width, pupil_height)` in pixels, preserving the requested physical variable and blink filtering. |
| Common timebase | SDK exposes native ~31/11-Hz ophys timestamps and synchronized behavior timestamps | Native frame interval differs by rig (32.29--93.22 ms) | Paper linearly interpolated calcium events and running onto a common 30-Hz event-relative series | Generate a 30-Hz grid within each native trial, anchored to synchronized trial start; interpolate neural/running/pupil from their timestamps. This retains ophys-based alignment and provides identical bins across rigs. |
| Running filter label | SDK prose calls it 10-Hz low-pass; current code constructs a 3rd-order Butterworth with `Wn=4, fs=60` | NWB provides both stored `speed` and `speed_unfiltered` | Whitepaper specifies processed/default low-pass speed | Use stored `processing/running/speed`; do not re-filter, avoiding version-label ambiguity. |
| Trial definition | NWB trial flags/times are produced with display lag correction and native go/catch logic | 51,992 go/catch trials across active experiments; every retained row has exactly one of four outcomes | Task defines go/catch and four outcomes; request excludes aborted/free reward | Trust NWB start/stop/change times and retain exactly `go | catch`; no performance/engagement filter was requested. |
| Change alignment | Stimulus table provides exact `is_change` and `is_sham_change` onset times | Across all 202 active experiments, every go change and catch sham change matches trial `change_time` exactly; no boundary violations | Change occurs at image onset; catch is a sham without identity change | Set image-change=1 only for the single 30-Hz bin containing a true `is_change` onset; catch sham changes remain 0. |
| Stimulus cadence | Presentation table has exact start/stop and image identity | Active images median 250.2 ms and gray gap median 500.4 ms; ~4,802 active presentations/experiment | Papers specify 250-ms image + 500-ms gray and 5% omissions | Label image only on `[start_time, stop_time)`; label all gaps and omissions as `gray`. |
| Dataset counts | SDK metadata supports experiment, session, container hierarchy | Local subset: 284 experiments/247 recordings/38 mice; planned active+eye: 199 experiments/171 recordings/38 mice, 29,168 cells, 51,075 trials | Paper behavioral cohort is larger (376 sessions/82 mice); primary neural subset is familiar multiplane only | Treat the supplied files as the available cohort. Use each imaging plane/experiment as a decoder session, consistent with the paper’s imaging-plane sampling unit. Do not impose the paper’s question-specific familiar/multiplane restriction. |

Final cross-source checks: direct NWB and metadata counts match exactly; every requested trial is within the ophys timestamp range; retained outcome one-hot sums are exact; trial change times and stimulus annotations agree with zero offset; and native stimulus cadence matches the papers. The remaining transform choices (percentile scope, interpolation through short invalid spans, categorical encoding, and edge handling) are specified in Step 5.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| NWB `processing/ophys/event_detection/data` + timestamps | `neural[session][trial]` | Select valid released cells; linearly interpolate raw event magnitudes at 30-Hz trial-bin centers; transpose to neuron x time; float32 | `BehaviorOphysExperiment.events`, `OphysTimestamps.from_nwb` | Raw detected events match the paper; every trial matrix has the experiment's complete valid neural population. |
| No decoder inputs requested | `input[session][trial]` | Empty float32 array with shape `(0, n_timepoints)` | N/A | Preserves target nesting/time shape without leaking any requested output into decoder inputs; `input_names=[]`. |
| Active image-presentation `image_name`, `start_time`, `stop_time`, `omitted` | `output[...][0, :]` | At each bin center label one of 16 global image identities only while an image is physically on screen; gray gaps and omissions map to `gray` | `BehaviorSession.stimulus_presentations` | Global classes: gray, im000, im031, im035, im045, im054, im061, im062, im063, im065, im066, im069, im073, im075, im077, im085, im106. |
| Active presentation `is_change`, `start_time` | `output[...][1, :]` | Binary 1 at the first 30-Hz sample at/after each true image-change onset; 0 otherwise | stimulus presentations / trial `change_time` | Catch `is_sham_change` is intentionally 0 because identity does not change. |
| NWB `processing/running/speed/{data,timestamps}` | `output[...][2, :]` | Linear interpolation of processed cm/s; session-wise quintile boundaries on retained trial samples; codes 0--4 | `RunningSpeed`, `get_running_df` | Do not clamp legitimate negative filtered values or re-filter. |
| NWB blink-filtered pupil ellipse `width`, `height`, eye timestamps | `output[...][3, :]` | Diameter = `2*max(width,height)`; interpolate gaps <=30 eye frames, interpolate to 30-Hz bin centers, then session-wise quintiles; codes 0--4 | `process_eye_tracking_data`, `filter_on_blinks` | Gaps >1 s are invalid periods; trials overlapping them are excluded rather than fabricating long stretches. Edge gaps are not extrapolated. |
| Trial flags `hit`, `miss`, `false_alarm`, `correct_reject` | `output[...][4, :]` | Map mutually exclusive class to 0--3 and repeat across all trial timepoints | `Trial._get_trial_data`, `Trials.from_nwb` | Repetition is required to combine one static label with four time-varying variables in one `(5,T)` output matrix. |
| NWB trial `start_time`, `stop_time`, `go`, `catch`, `aborted`, `auto_rewarded` | trial nesting | Active sessions only; select exactly `go | catch`; create complete 1/30-s bins whose centers are `start + (k+0.5)/30` | `Trials` / NWB table | Native variable-duration trials (7.02--12.56 s) become 210--376 timepoints; final sample count is known only after long-pupil-gap exclusion. |
| Metadata `mouse_id` | `subjects`, `subject_idx` | String IDs, sorted unique list, per-experiment index | experiment metadata table | Planned cohort retains all 38 mice. |
| Metadata `targeted_structure` | `brain_regions`, `brain_region_idx` | Sorted region names (`VISl`, `VISp`); repeat experiment region index for every neuron | `BehaviorOphysExperiment.metadata` | Each per-plane NWB records one structure. |

### Key Decisions
1. **Session unit and cohort**: Each NWB imaging plane is one decoder session. Keep all active experiments with eye tracking across Familiar, Novel 1, Novel >1, areas, depths, rigs, and cell classes; exclude passive viewing and the three files with no eye stream. This follows paper task selection without importing its question-specific familiar/multiplane restriction.
2. **Neural activity**: Use released raw detected calcium event magnitudes. DFF is less appropriate because the paper explicitly used inferred events; SDK `filtered_events` adds smoothing intended for visualization only.
3. **Time grid**: Use 30 Hz (`33.333333 ms`) because target bins must match across sessions and the paper linearly interpolated both events and running to 30 Hz. Bin centers ensure only complete bins are included. Source alignment remains the synchronized ophys timestamp axis.
4. **Pupil diameter**: Use twice the major fitted half-axis rather than pupil area. Short blink/outlier gaps (<=1 s) are linearly interpolated only between valid processed samples; any go/catch trial intersecting a longer invalid interval is removed. The 1-s rule limits interpolation to brief blink-scale gaps while retaining about 95.8% of otherwise eligible trials in a pre-conversion scan.
5. **Percentile scope**: Compute running and pupil quintile edges separately within each session, using only kept-trial samples. Pupil pixels are rig/animal specific, and session-local quantiles remove calibration offsets; applying the same rule to running yields comparable behavioral-state ranks and balanced targets. Store each session's physical-unit edges in metadata.
6. **Stimulus labels**: Use 17 global identity values (16 named images plus gray). Omissions mean no image was presented, so they are gray rather than a seventeenth image. Natural-movie/spontaneous blocks are outside active trials.
7. **Change pulse**: Mark one sample—the first sample at or after a true change onset. Do not mark sham catch changes, trial onset, image-to-gray offset, gray-to-repeat onset, or omissions.
8. **Static outcome**: Repeat outcome along time to support a single output matrix and allow the validator/decoder to consume mixed static and time-varying outputs. Codes are hit=0, miss=1, false alarm=2, correct reject=3.
9. **Missing values**: Running has no NaNs. Pupil uses no raw outlier values and no global/zero fill. Sessions without the stream are excluded; trials with long invalid periods are excluded; any unexpected nonfinite neural/running/output value is a hard error.
10. **Precision and size**: Neural/input intermediates are float32; categorical outputs use compact integer dtype. Estimated neural payload is ~7.5 GiB before pickle overhead, feasible in available storage/memory. Direct HDF5 reads avoid loading masks/projections and should keep conversion under 15 minutes.
11. **Offsets/metadata**: `temporal_alignment_event='native trial start on synchronized ophys clock'`, `off_start=0.0`, `off_end=None` because stop time varies. Metadata will contain source IDs, trial/cell counts, rig/frame-rate, region, experience, cell class, excluded-session/trial rationale, bin edges, and categorical codebooks.
12. **Sample mode**: Select two deterministic eligible experiments with adequate cells and all four outcome types, preferably spanning image sets A/B, so sample validation exercises every category and processing branch.

### Planned Sanity Checks
- [ ] For fixed session/trial/neuron/time samples, independently read raw event values and use `np.allclose` against the conversion's linear interpolation.
- [ ] Independently read raw running and pupil timestamps/values, reproduce short-gap fill and interpolation, and use `np.allclose` before checking stored quintile codes.
- [ ] Independently reconstruct image identity/change samples from raw presentation intervals and compare exactly (`np.allclose` on integer arrays).
- [ ] Assert each retained trial is active, go XOR catch, not aborted/auto-rewarded, in ophys bounds, has exactly one outcome, and has >=2 bins.
- [ ] Assert session/trial nesting counts agree across neural/input/output; neuron counts agree with cell table; all matrices share T; all values are finite and categorical ranges match codebooks.
- [ ] Check global counts against raw metadata (199 pre-trial-filter sessions, 38 mice, 29,168 valid cell observations, 51,075 pre-pupil-filter trials) and report exact long-gap exclusions.
- [ ] Verify empirical running/pupil output frequencies are near 20% per class and explain deviations from ties at quantile edges.
- [ ] Verify observed image labels are exactly the planned 16 identities plus gray, true-change pulses are one-bin wide, and every go trial has its true change represented.
- [ ] `--show-processing` plots will show native vs 30-Hz timestamps/events, trial boundaries, image/gray state, one-bin change pulses, processed continuous running/pupil with quintile edges/codes, missing-pupil gaps/trial filtering, and outcome distributions for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required CLI (`--full`, `--sample`, `--show-processing`). It directly reads only necessary NWB datasets, applies all Step 5 selection/alignment/curation rules, performs strict assertions, writes atomically with pickle protocol 5, reports per-session and total timings, and emits eight-panel processing figures for up to two sessions. `python3 -m py_compile` and `--help` both completed successfully.

Code inefficiencies identified:
Loading AllenSDK session objects would materialize projections, masks, tables, and pandas object arrays that are irrelevant to conversion. Per-neuron interpolation loops and repeated trial-level HDF5 reads would also multiply I/O. A full converted neural payload is estimated at ~7.5 GiB, so redundant copies matter.

Code speedups added:
Direct `h5py` access; one sequential read per required stream/session; float32 `read_direct` for event matrices; vectorized timestamp search/interpolation across all retained trial samples; one concatenated alignment followed by contiguous trial slicing; compact int16 outputs; and no image masks/projections. Sessions are processed sequentially to avoid simultaneous decompression of multi-GiB NWBs and unbounded memory.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 408 session-neurons |
| Neurons / session | [208, 200] |
| Subjects | 2 |
| Sessions / subject | 1 each |
| Trials (total) | 344 retained; 2/346 candidates removed for long pupil gaps |
| Trials / session | [151, 193] |
| Timepoints / trial | 217--375; session-mean 244.9 and 247.2 |
| Input range | No decoder inputs; `(0,T)` |
| Image identity distribution | gray 0.667; each of 16 images 0.016--0.026 overall |
| Image change distribution | [0.9964, 0.0036] |
| Running quintile distribution | [0.2000, 0.2000, 0.2000, 0.2000, 0.2000] |
| Pupil quintile distribution | [0.2000, 0.2000, 0.2000, 0.2000, 0.2000] |
| Trial outcome distribution (time-weighted) | [hit 0.681, miss 0.204, false alarm 0.059, correct reject 0.056] |

### Processing Plots Review
Reviewed `processing_792813858.png` and `processing_809501118.png`. Native ophys intervals cluster near 32.3 ms and the target line is 33.33 ms; raw event dots and interpolated traces coincide without shifts; image flashes alternate with gray at the expected cadence; true change is exactly one sample and coincides with the new image; running and pupil are smooth and synchronized; quintile thresholds are ordered; pupil raw/filled traces agree outside invalid samples; and curation/outcome panels agree with logged counts. No temporal or discretization anomaly was found.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Direct stream-only HDF5 reads + vectorized session interpolation | Sample processing was 0.48 s and 0.42 s/session; avoids SDK/mask/projection loading |
| float32 event read/interpolation and compact output arrays | Sample pickle is only 67 MiB; write took 0.07 s |

| Step | Time / Session | Estimated Total Time |
| Stream processing (sample) | 0.45 s mean | ~90 s for 199 sessions before size adjustment |
| Size-adjusted processing | Sample neural payload vs estimated full payload gives ~115x scale | ~105--180 s |
| Pickle serialization | 0.07 s for 67 MiB | ~8--20 s for ~7.5 GiB |
| Full conversion estimate | N/A | approximately 2--4 minutes, well below 15 minutes |

Required commands completed. `/app/sample_data.pkl` is 67 MiB; `/app/conversion_sample_out.txt` and `/app/verification_sample_out.txt` exist. The provided verifier reported “Data format is valid, no errors or warnings.” Manual structure checks confirmed finite float32 neural data, `(0,T)` inputs, `(5,T)` integer outputs, aligned trial dimensions, and correct metadata/codebook lengths.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Image identity (chance 0.0588) | 0.3929 | 0.3998 |
| Image change (chance 0.5000) | 0.6711 | 0.6328 |
| Running speed quintile (chance 0.2000) | 0.2628 | 0.2492 |
| Pupil diameter quintile (chance 0.2000) | 0.2364 | 0.2169 |
| Trial outcome (chance 0.2500) | 0.3382 | 0.3117 |

`/app/train_decoder_sample_out.txt` was created and training completed on CUDA. Mean normalized loss decreased monotonically from 1.6371 (epoch 1) to 1.5032 (epoch 200); test loss was 1.5558. Every validation balanced accuracy is above uniform chance. The train/validation accuracies are close, with no evidence of leakage or severe overfitting.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 7.346 GiB (7.4G filesystem display)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Paper question-specific neural subset: 10,328 | Valid ROI filter | 29,168 active+eye session-neurons | 29,168 | Yes for selected local cohort; paper subset intentionally narrower |
| Mean neurons/session | Not tabulated | Cell table defines released cells | 146.57 (4--666) | 146.57 (4--666) | Yes |
| Subjects | Paper behavior cohort 82 | Metadata `mouse_id` | 38 in supplied selected files | 38 | Yes for supplied cohort |
| Sessions | Paper behavior cohort 376 recordings | Active-session selection; imaging plane is neural unit | 199 eligible experiment planes / 171 unique recordings after missing-eye exclusion | 199 experiment sessions | Yes |
| Trials (total) | Not tabulated; go/catch only required | `go | catch`, exclude abort/auto | 51,075 candidates; 2,166 overlap long pupil-invalid periods | 48,909 | Yes; exact accounting |
| Trials/session | Not tabulated | Native trial table | 39--409 candidates | 39--403 retained | Yes after documented filter |
| Timepoints/trial | 30-Hz reference interpolation | N/A | Native duration 7.02--12.56 s | 210--376 complete 30-Hz bins | Yes |
| Brain regions | V1 and LM | `targeted_structure` | VISp 29,006 cells; VISl 162 | same | Yes |
| Image identity distribution | 250-ms image + 500-ms gray predicts ~1/3 image, ~2/3 gray | presentation intervals | 16 identities + gray | gray 0.669; each image 0.019--0.022 | Yes |
| Image change distribution | one change per retained trial, variable trial length | exact `is_change` onset | 48,909 pulses over ~12.4M bins | no-change 0.997, change 0.003 | Yes |
| Running quintiles | Five percentile bins requested | stored processed speed | session-wise quantiles | [0.200,0.200,0.200,0.200,0.200] | Yes |
| Pupil quintiles | Five percentile bins requested | blink-filtered pupil fits | session-wise quantiles | [0.200,0.200,0.200,0.200,0.200] | Yes |
| Trial outcomes | Four task outcomes | mutually exclusive SDK flags | kept raw trial counts | hit 15,209; miss 27,582; false alarm 886; correct reject 5,232 | Yes; sums to 48,909 |

Full conversion took 88.09 s (81.40 s processing, 6.69 s serialization), substantially below the estimate and optimization threshold. The verifier completed successfully with no structural errors. It emitted 2,429 warnings for all-zero neural trials. These represent valid silent intervals in sparse detected-event data, concentrated in small-cell planes; filtering them would violate trial selection and bias activity. Step 10 formally checks raw-data equality and quantifies this warning before final acceptance.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Full verification log**: Read all of `/app/verification_full_out.txt`. There are no errors. The only warnings are 2,429 all-zero neural trials across 87 sessions. Independent scan found that warning sessions have only 4--120 cells. Three representative warnings (experiments 792815735, 958741222, 940852112, including the largest 120-cell warning case) were independently reconstructed from raw event-detection arrays and matched exactly with `np.allclose`; the raw event data are truly silent in those trial windows. These warnings cannot and should not be “fixed”: FastLZero event traces are sparse, and selecting trials based on neural activity would bias the dataset and violate go/catch trial selection. DFF substitution or visualization smoothing would contradict the paper.
2. **Independent neural sanity check**: `/app/cache/sanity_checks.py` did not import conversion code. For experiment 792813858, converted trial 5, it independently read raw event timestamps/data and used per-neuron `np.interp`. Full `(208,240)` arrays matched with `np.allclose(rtol=1e-5, atol=1e-6)`; maximum absolute difference was `2.98e-08` (float32 rounding only).
3. **Independent input sanity check**: From the raw trial start/stop, the independent script reconstructed 240 complete 30-Hz bins and compared the specified no-input representation to converted `(0,240)`. Shape and `np.allclose` passed. This is the only scientifically correct input check because the task explicitly specifies no decoder inputs.
4. **Independent output sanity check**: For the same raw trial, the script independently rebuilt all five rows from the raw presentation, running, pupil, and trial tables. Image identity, true-change pulse, running quintile, pupil-diameter quintile, and repeated outcome each matched exactly (`np.allclose` true for every row).
5. **All-session raw checks**: For every converted session/trial, raw trial IDs were unique, go/catch counts reconciled, timepoint count equaled `floor((stop-start)*30)`, first/last centers were strictly inside native bounds, output outcome was constant, and change-pulse count was exactly one for hit/miss and zero for false alarm/correct reject. Raw cell/event dimensions matched `brain_region_idx`, and every cell-table ROI was valid.
6. **Pupil exclusion re-derivation**: Independently reconstructed short and residual pupil gaps for all 199 sessions and compared exact removed raw trial-ID sets. Every set matched. Global accounting passed: 51,075 candidate go/catch - 2,166 pupil-invalid = 48,909 retained.
7. **Reference-code comparison**: Compared each major processing stage as detailed below; no mismatches remain.
8. **Statistics comparison**: Local cohort statistics exactly match raw NWB/CSV values after documented filters. Retained catch fraction is 12.509%, matching the whitepaper's ~12.5%; active omission fraction is 3.52% of all presentations (consistent with nominal 5% only among eligible repeats, since changes and pre-change flashes are protected); image duration median is 250.2 ms; ophys rates span 10.73--30.97 Hz; trial bins cover 7.00--12.53 s; and running/pupil bins are each 20.0% per quintile. Paper cohort counts differ because supplied files are a 38-mouse subset and its primary neural analysis further restricts to familiar multiscope recordings.
9. **Edge-case audit**: Verified initial/final bin handling, variable trial duration, no partial end bins, no out-of-range interpolation, catch sham changes, omissions-as-gray, A/B image codebooks, edge/long pupil gaps, sessions with only four cells, simultaneous multiplane experiments, missing-eye sessions, and sessions lacking rare outcomes. All retained sessions have 39--403 trials, safely above the two-trial minimum.

### Major Processing Logic Comparison
| Stage | Converter | Reference code/text | Comparison |
|-------|-----------|---------------------|------------|
| Data loading | Direct NWB HDF5 datasets | `BehaviorOphysExperiment.from_nwb` constructs the same released streams | Same underlying NWB values; direct access avoids irrelevant masks/projections. |
| Neuron/trial filtering | Released valid ROIs; active+eye experiments; go/catch; long pupil-invalid trials removed | `CellSpecimens` valid ROI filter; paper active-session selection; `Trial` category logic | Same referenced curation plus required pupil validity rule, fully documented and independently checked. |
| Temporal alignment | Hardware-synchronized timestamps; 30-Hz bin centers inside native trials | `OphysTimestamps`; paper linearly interpolates events/running to 30 Hz | Same source clock and interpolation precedent; center/complete-bin convention removes edge ambiguity. |
| Binning | Linear event/running/pupil interpolation; session quintiles | Paper 30-Hz interpolation; decoder asks for five percentile bins | Matches applicable reference; categorical quintiles are required downstream transform. |
| Input construction | `(0,T)` float32 | Decoder task: “No inputs” | Exact. |
| Output construction | Presentation identity/true change, processed running, blink-filtered pupil diameter, SDK outcome | SDK stimulus/trial/behavior tables; whitepaper pupil major axis and trial definitions | Same source variables and definitions; gray/omission and sham-change edge cases explicit. |

### Issues Found and Resolved
- **Validator all-zero warnings**: Verified as genuine raw FastLZero silence, limited to <=120-cell planes. No conversion change is scientifically justified; warning retained and explained.
- **Paper-vs-local cohort counts**: Not a conversion loss. The provided local NWBs contain 38 mice, while the paper reports a larger behavioral cohort and a narrower familiar-multiscope neural subset. All local selection counts reconcile exactly.
- **No failed check or converter mismatch was found**, so the iteration protocol did not require a conversion rerun. The independent audit log is `/app/cache/sanity_checks_out.txt`; all checks end with `ALL_INDEPENDENT_SANITY_CHECKS_PASSED`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Training loss decreased monotonically from 1.632168 at epoch 1 to 1.579239 at epoch 200; held-out test loss was 1.599742.
- Execution: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt` completed successfully on CUDA. It used 39,052 training trials and 9,857 validation trials and created `sample_trials.png` and `predictions.png`.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Image identity | 0.1965 | 0.1921 | 17 classes; chance 0.0588; 3.27x chance on validation |
| Image change | 0.5519 | 0.5309 | Binary one-bin pulse; chance 0.5000; 1.06x chance |
| Running speed quintile | 0.2266 | 0.2226 | Five classes; chance 0.2000; 1.11x chance |
| Pupil diameter quintile | 0.2254 | 0.2207 | Five classes; chance 0.2000; 1.10x chance |
| Trial outcome | 0.2938 | 0.2654 | Four classes; chance 0.2500; 1.06x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation Balanced Accuracy | Chance | Ratio to Chance | Expectation from Papers / Review |
|----------|------------------------------|--------|-----------------|----------------------------------|
| Image identity | 0.1921 | 0.0588 | 3.27x | Not decoded in the reference paper; strong above-chance result is expected from visually responsive cortex. |
| Image change | 0.5309 | 0.5000 | 1.06x | Figure 6A is approximately 0.52--0.65 across cell types/counts. The achieved score lies in that range despite the harder required label being positive for only one 33-ms bin rather than a selected 400-ms change-vs-previous-repeat example. |
| Running speed quintile | 0.2226 | 0.2000 | 1.11x | Not decoded in the paper. Modest above-chance cortical prediction is plausible for five instantaneous behavioral bins. |
| Pupil diameter quintile | 0.2207 | 0.2000 | 1.10x | Not decoded in the paper. Modest above-chance cortical prediction is plausible for five instantaneous arousal bins. |
| Trial outcome | 0.2654 | 0.2500 | 1.06x | Figure 6C's *binary hit-vs-miss* decoder is approximately 0.52--0.74 (chance 0.50). This is not the same target: the required output has all four outcomes and is scored at every trial timepoint, including pre-change activity that cannot reveal a future choice. Raw audit below rules out label/alignment errors. |

All five validation accuracies exceed their uniform balanced-accuracy chance levels. Only image identity exceeds the workflow's conservative 1.5x-chance screening threshold, so all four other outputs received the full debugging audit rather than being accepted solely from the aggregate score.

The paper reports only two classification analyses: change-vs-repeat and hit-vs-miss. Its plots, rather than tables, provide the approximate ranges above. Its classifier is a random forest on the first 400 ms after selected image presentations, uses 5-fold image-example cross-validation, and balances two selected categories. The supplied validator instead learns shared linear categorical heads from instantaneous event activity across every trial bin. The change result is nevertheless inside the paper's plotted range. Direct numerical comparison of 0.2654 four-class outcome accuracy to binary hit/miss accuracy would be misleading; relative improvement over the appropriate chance level and the independent label reconstruction are the meaningful checks.

### Required Low-Accuracy Debugging
1. **Three specific raw trials**: `/app/cache/critical_review2.py` independently reconstructed neural activity and every output row from raw NWBs without importing the converter. Experiment/trial pairs `(792813858, 2)`, `(809501118, 26)`, and `(958741222, 5)` cover hit, false alarm, and correct reject examples across image sets/recordings. All 15 output-row comparisons were exact (`np.allclose` with zero tolerance) and all three complete neural matrices passed `np.allclose(rtol=1e-5, atol=1e-6)`.
2. **Temporal alignment plot**: `/app/cache/critical_review2_alignment.png` overlays mean raw detected events, image identity, exact one-bin change labels, running quintile, pupil quintile, and raw synchronized change onset. Visual review shows the hit change pulse begins at the first bin center at/after the raw change onset; catch trials correctly contain no true-change pulse; image periods and gray periods alternate without shift.
3. **Output variation**: Every declared class occurs. Exact time-bin counts are image identity `[8335350, 237998, 242930, 240100, 240823, 240153, 272367, 276144, 274643, 271842, 276008, 272649, 239467, 245104, 273353, 277583, 238271]`; image change `[12411994, 42791]`; running `[2490994, 2490924, 2490910, 2490925, 2491032]`; pupil `[2490994, 2490924, 2490909, 2490925, 2491033]`; outcome `[3822065, 7071388, 215594, 1345738]`. The sparse positive change class is intentional and the loss/metric are class-balanced.
4. **Neural filtering**: Rechecked Step 10's all-session assertions: released valid ROIs only, raw FastLZero events, active sessions, synchronized ophys clock, no activity-dependent trial removal. The 2,429 silent trials exactly match raw event arrays and are not a processing artifact.
5. **Reference processing**: Rechecked raw event interpolation, trial selection, task flags, processed running stream, pupil major-axis definition, 30-Hz alignment, and presentation tables against the SDK/methods mapping in Step 10. No mismatch was found.
6. **Train/validation gap**: Train/validation accuracy ratios are 1.023 (image), 1.040 (change), 1.018 (running), 1.021 (pupil), and 1.107 (outcome). All are far below the 1.5x overfitting threshold; there is no evidence of leakage or material overfitting.

### Issues Found and Resolved
- **Four outputs below 1.5x chance**: Investigated with all five prescribed checks. Labels, clocks, class variation, filters, and reference transforms all passed. The modest margins are explained by instantaneous decoding of weak behavioral/state signals and, for change/outcome, by the output definitions required here. No converter change is justified.
- **Paper outcome accuracy appears numerically higher**: Resolved as a target/evaluation mismatch mandated by this task, not dismissed without testing. Paper: binary hit/miss on selected 400-ms post-change windows with a random forest. This dataset: four outcomes across complete variable-length trials, scored per bin by the provided linear decoder. Three raw label reconstructions and global outcome counts passed.
- **Iteration result**: No conversion bug was discovered, so rerunning the 7.35-GiB conversion/training would produce no corrective change. The audit log `/app/cache/critical_review2_out.txt` ends with `ALL_CRITICAL_REVIEW2_RAW_CHECKS_PASSED`.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading instructions, format, statistics, caveats, and reproduction commands
- [x] cache/ folder created with `README_CACHE.md`
- [x] Investigation scripts, audit logs, temporary paper renders, and bytecode moved to cache; primary code, datasets, required logs, and diagnostic plots remain in `/app`
