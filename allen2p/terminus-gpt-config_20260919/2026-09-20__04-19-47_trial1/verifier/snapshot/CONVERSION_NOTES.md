# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- numpy 2.4.4
- torch 2.6.0+cu124
- Key package imports succeeded.
- `/app/CONVERSION_NOTES.md` existence verified with `ls -la`.

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
| `BehaviorOphysExperiment.from_nwb` | `allensdk/brain_observatory/behavior/behavior_ophys_experiment.py` | LOADING | Construct an ophys experiment from NWB, including behavior and imaging streams. |
| `BehaviorSession.from_nwb` | `allensdk/brain_observatory/behavior/behavior_session.py` | LOADING | Load trials, stimulus presentations, running, rewards, licks, and filtered eye tracking. |
| `BehaviorOphysExperiment.dff_traces` | `behavior_ophys_experiment.py` | LOADING | Return precomputed per-cell dF/F traces; dF/F does not need to be recomputed. |
| `BehaviorOphysExperiment.events` | `behavior_ophys_experiment.py` | PROCESSING | Return precomputed event traces and filtered events where available. |
| `BehaviorOphysExperiment.ophys_timestamps` | `behavior_ophys_experiment.py` | LOADING | Imaging-frame timestamps used as the required common temporal axis. |
| `BehaviorOphysExperiment.cell_specimen_table` | `behavior_ophys_experiment.py` | CURATION | Cell metadata and ROI/cell identifiers associated with neural rows. |
| `Trials.from_nwb` and trial properties | `data_objects/trials/trials.py` | LOADING | Load trial boundaries and go/catch/aborted/auto-rewarded plus hit/miss/false-alarm/correct-reject flags. |
| `StimulusPresentations.from_nwb` | behavior data objects | LOADING | Load image identity and stimulus presentation start/stop times. |
| `RunningSpeed.from_nwb(..., filtered=True)` | `data_objects/running_speed/running_speed.py` | PROCESSING | Load the SDK-filtered running-speed time series and timestamps. |
| `EyeTrackingTable.from_nwb` | `data_objects/eye_tracking/eye_tracking_table.py` | PROCESSING | Load eye/pupil measurements after outlier filtering (z threshold and frame dilation parameters). |
| experiment table `passed_only=True` | `behavior_project_cache/tables/experiments_table.py` | CURATION | Default cache behavior excludes experiments that failed release/QC criteria. |

### Notes
- The repository is the AllenSDK and includes dedicated Visual Behavior optical physiology/NWB documentation.
- The ophys experiment object joins behavior and imaging data. Neural rows are keyed by cell specimen/ROI metadata and columns follow ophys frame timestamps.
- Use released precomputed dF/F (or supplied event traces if the source data specifically contains those), not a new dF/F computation. The final choice must follow the actual packaged data and paper methods.
- Trial outcome is represented by mutually meaningful boolean fields: hit/miss for go trials and false alarm/correct reject for catch trials. Aborted trials remain in the native table but are explicitly excluded from non-aborted performance calculations. The decoder task additionally requires excluding auto-rewarded trials.
- Stimulus presentations provide image identity on presentation intervals; omitted/gray intervals must be treated according to their explicit labels rather than guessed.
- Running speed and pupil/eye streams have their own timestamps and must be sampled/aligned to ophys timestamps. SDK-filtered streams should be preferred to reproducing filters ad hoc.
- Eye tracking uses z-score outlier detection and temporal dilation around outlier frames in the SDK loader; missing filtered samples must not be silently converted into a biological category.
- Cache experiment tables default to `passed_only=True`, documenting release/QC curation. Any additional curation in the supplied subset will be checked against metadata and paper methods.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Root package: `/app/data/visual-behavior-ophys-1.1.0` (Allen Visual Behavior Ophys release 1.1.0), plus a project manifest JSON.
- 284 `behavior_ophys_experiment_<id>.nwb` HDF5/NWB files (one imaging experiment/plane per file) and four project metadata CSVs: behavior sessions, ophys sessions, ophys experiments, and ophys cells.
- Total package size is approximately 247 GB (291 files including metadata/manifest).
- The 284 experiment files map to 247 unique ophys sessions and 247 behavior sessions. Plane/experiment counts per session: 239 sessions with 1, one with 3, one with 4, two with 5, and four with 7. Behavioral tables are duplicated across planes belonging to the same session; unique behavioral trial statistics must count one representative file per `ophys_session_id`.
- Neural arrays are `processing/ophys/dff/traces/data` and `processing/ophys/event_detection/data`, shaped time x cells, with matching ophys timestamps. Example shape: 140,204 x 13; experiment frame counts range 48,284--149,508.
- Cell metadata are in `processing/ophys/image_segmentation/cell_specimen_table`, including IDs and `valid_roi`.
- Trial table fields include start/stop/change times, initial/change image names, go, catch, aborted, auto-rewarded, hit, miss, false alarm, correct reject, reward and response fields.
- Running has filtered and unfiltered speed plus timestamps. Eye/pupil data are under `acquisition/EyeTracking`; filtered pupil `area`, raw area, width, height and eye-camera timestamps are present in 281/284 experiment files. Three experiment files lack actual eye-tracking arrays.
- Stimulus presentation intervals and image templates are stored in NWB stimulus/interval structures; image presentation timestamps differ from the ophys clock and therefore require timestamp alignment.
- Native array dtypes include float64 traces/timestamps, boolean trial flags/valid ROI, and integer image indices/IDs.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 42,147 valid ROI rows across 284 experiment files |
| Neurons / session | Per experiment: min 4, median 66, mean 148.40, max 666 valid ROIs; session-level concatenation will be computed during conversion |
| Subjects | 38 mice |
| Sessions / subject | 247 ophys sessions / 38 mice (mean 6.50; exact distribution retained in metadata) |
| Trials (total) | 148,231 native rows counted once per unique ophys session; 74,476 eligible go/catch trials; auto-rewarded trials are a separate mutually exclusive category |
| Trials / session | Native: min 399, median 586, mean 600.13, max 1,241; eligible counts to be calculated in conversion |

### Available Variables and Distributions
- Regions: 261 experiment files in VISp and 23 in VISl.
- Cre lines: 153 Slc17a7-IRES2-Cre, 85 Sst-IRES-Cre, 46 Vip-IRES-Cre experiment files.
- Experience: 150 Familiar, 38 Novel 1, 96 Novel >1 experiment files.
- Unique-session trial flags: go 65,128 (hit 14,194; miss 50,934), catch 9,348 (false alarm 846; correct reject 8,502), aborted 73,080, auto-rewarded 675. The decoder-required eligible count is go + catch = 74,476; auto-rewarded and aborted are separate mutually exclusive flags.
- Important edge cases: multi-plane session duplication; three files without pupil arrays; variable frame and trial counts; very small cell counts in some planes; timestamps differ across ophys, stimulus/running, and eye streams.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not stated as one total in machine-readable text | Paper analyzes excitatory, Sst, and Vip cells with imaging-plane/cell hierarchical sampling. |
| Neurons / session | Variable; decoder iterates neuron count per imaging plane | “For each imaging plane we sampled neurons…” |
| Subjects | 82 mice in the broader behavioral cohort | “This dataset contains behavior from 376 imaging sessions from 82 mice.” |
| Sessions / subject | 376 imaging sessions / 82 mice in broader cohort | Same paper statement; supplied subset is smaller. |
| Trials (total) | Not reported as a single text value | Trial/image analyses described methodologically. |
| Trials / session | Not reported as a single text value | Native data provide exact counts. |
| Neural data time bin | Ophys acquisition timestamps; paper interpolated event-triggered traces to 30 Hz | “linearly interpolating onto a common 30hz timeseries.” |
| Behavior data time bin | Image-presentation intervals are 750 ms | “the 750 ms interval beginning with each image presentation.” |
| Reward rate | Not reported as one fraction | Outcomes and strategy metrics vary by session. |
| Change response window | 250–750 ms after image change | Methods describe licks in this response window as hits for go trials and false alarms for catch trials. |
| Neural image response windows | 50–800 ms full interval; 50–425 and 425–800 ms halves; 150–250 ms narrow excitatory window | Paper methods; 50 ms delay accounts for propagation to visual cortex. |
| Eye camera | 30 fps, 33 ms exposure | Whitepaper behavior-monitoring hardware section. |
| Published decoder window | First 400 ms after each stimulus presentation | Paper decoding-results text. |

### Processing Details
- The task presents 250 ms natural images separated by 500 ms gray, yielding one 750 ms image interval. Image changes occur without an intervening gray period at the changed presentation boundary; omissions retain a nominal 750 ms interval.
- Behavioral events were assigned to image presentation intervals. Licks were grouped into bouts with a 700 ms inter-lick threshold for the paper’s behavioral analysis.
- The paper used detected calcium events for all neural analyses, producing per-cell events with times and magnitudes. Therefore event-detection traces, not recomputed dF/F, are the reference-consistent neural representation for this conversion.
- Running event-triggered traces were linearly interpolated to a common 30 Hz axis. For this task, the common axis must specifically be native ophys timestamps; interpolation of running/pupil to those timestamps preserves that intent without arbitrary fixed bins.
- Eye processing uses DeepLabCut perimeter points and ellipse fits for pupil, eye, and corneal reflection. The SDK’s filtered pupil area is the reference-processed pupil-size measurement. Diameter can be derived as equivalent-circle diameter `2*sqrt(area/pi)`, which is monotonic with area and has physical “diameter” semantics.
- Published random-forest decoders predicted image change versus repeat and hit versus miss using the first 400 ms after presentations and 5-fold cross-validation. The current required decoder has additional image, running, pupil, and four-class outcome targets and a different architecture, so published values are qualitative rather than direct numeric benchmarks.

### Curation Steps

**Neuron curation rules**:
- Use released valid ROIs/cells and the event-detection output; do not rerun segmentation or event detection.
- Whitepaper experiment QC includes z-drift exclusion above 10 µm, residual-motion review, data-stream integrity checks, and temporal synchronization checks. Project cache defaults to passed experiments.

**Trial curation rules**:
- Required conversion includes go and catch only and excludes aborted and auto-rewarded categories.
- Go outcomes are hit/miss; catch outcomes are false alarm/correct reject.
- Whitepaper release QC required peak d-prime at least 1.0 and checked task performance.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Image change versus preceding repeat | Reported graphically by neuron count/cell type; no single numerical text value extractable |
| Hit versus miss | Reported graphically by neuron count/cell type; no single numerical text value extractable |

The broader paper cohort (376 sessions, 82 mice) does not equal the supplied subset (247 sessions, 38 mice); this discrepancy is investigated in Step 4 rather than forcing the supplied files to match a non-identical cohort.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Cohort size | Full release metadata has 1,936 experiments | Supplied subset has 284 experiments, 247 sessions, 38 mice | Broader paper behavior cohort has 376 sessions, 82 mice | Convert every supplied usable experiment; do not fabricate absent paper-cohort files. Report subset statistics explicitly. |
| Session unit | SDK distinguishes experiment (plane) and ophys session | Eight sessions contain 3–7 planes; plane clocks are staggered by up to 69.91 ms | Paper decoding and bootstrap unit is imaging plane | Treat each NWB experiment/plane as a target “session”; this avoids invalid direct concatenation and matches reference analysis. |
| Trial categories | SDK exposes go/catch/aborted/auto-rewarded | Every one of 148,231 unique-session rows has exactly one category | Task requires go+catch and excludes aborted+auto-rewarded | Eligible mask is exactly `go | catch`, giving 74,476 unique-session trials. No subtraction of auto-rewarded from go/catch. |
| Neural representation | SDK exposes dF/F and event detection | Both arrays share ophys timestamps | Paper used detected calcium events for all neural analyses | Use event-detection magnitudes. |
| Pupil availability | SDK allows missing eye tracking | 281/284 experiments have pupil arrays; three missing experiments are distinct sessions | Required decoder output includes pupil diameter | Exclude the three experiments because the required target cannot be constructed without unjustified imputation. |
| Temporal axis | Streams have independent timestamps | Ophys, running/stimulus, and eye clocks differ | Paper resampled event-triggered behavior to 30 Hz; task explicitly requires ophys timestamps | Keep native ophys samples and align stimulus by intervals and continuous behavior by interpolation. |
| Stimulus table path | NWB dynamic tables can have stimulus-specific names | Active natural-image group contains image_name, omitted, is_change, start/stop | Analyses use 750 ms presentation intervals (250 ms image + 500 ms gray) | Discover active natural-image interval group by columns; assign identity only on actual 250 ms non-gray start/stop intervals and gray otherwise. |

### Final Reconciled Understanding
- The supplied subset consists of released/QC-passed experiment NWBs covering VISp/VISl, three Cre lines, and familiar/novel experience levels. Full metadata contains absent projects/files and is contextual only.
- Release QC incorporates task-performance, motion/z-drift, sync, hardware, and stress checks. Use valid ROI rows and do not invent further signal-amplitude thresholds.
- Each experiment/plane has its own event matrix and ophys timestamps and will be independently segmented by its copied behavior trials. This duplicates behavioral trials only where neural planes differ, which is intentional and consistent with paper decoding by plane.
- Trial boundaries are native start/stop times. Eligible categories are mutually exclusive go and catch. Outcomes map to hit, miss, false alarm, and correct reject.
- The three missing-eye experiments will be explicitly listed in metadata as exclusions.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/event_detection/data` | `neural` | Select valid ROI columns; linearly interpolate event magnitudes to a 30 Hz grid anchored to the first ophys timestamp in each native trial | `BehaviorOphysExperiment.events`, `ophys_timestamps` | Output is cells x time, float32. Detected events match paper methods. |
| none | `input` | Empty float32 array of shape 0 x time | N/A | Decoder Task explicitly specifies no inputs. |
| active natural-image presentation `image_name`, start/stop, omitted | `output[0]` | Gray by default; assign one of 16 image identities only during the actual ~250 ms non-gray presentation; omitted remains gray | `StimulusPresentations.from_nwb` | 17 classes including gray. |
| presentation `is_change` and start time | `output[1]` | One-bin pulse at first 30 Hz sample at/after each true image-change onset | stimulus presentation table | Binary; omitted/sham changes are not true image changes. |
| filtered `processing/running/speed` | `output[2]` | Linear interpolation to trial grid, then session-wise quintile discretization | `RunningSpeed.from_nwb(filtered=True)` | Five labels Q1–Q5; percentile edges computed from all selected eligible-trial samples in that experiment. |
| filtered pupil ellipse `area` and eye timestamps | `output[3]` | Equivalent-circle diameter `2*sqrt(area/pi)`; interpolate across finite positive eye samples; session-wise quintiles | `EyeTrackingTable.from_nwb` | Monotonic conversion gives diameter semantics; three experiments with no pupil stream excluded. |
| trial hit/miss/false_alarm/correct_reject | `output[4]` | Four-class label repeated at every time bin | `Trials` outcome properties | Semantically static per trial; repeated because validator requires a single 2-D matrix when other outputs vary in time. |
| `mouse_id` | `subjects`, `subject_idx` | String IDs and integer lookup | experiment metadata | One subject index per experiment session. |
| `targeted_structure` | `brain_regions`, `brain_region_idx` | Region name and repeated index for all valid cells in plane | experiment/cell metadata | VISp or VISl. |

### Key Decisions
1. **Session unit is one experiment/imaging plane**: Matches paper decoding and avoids merging multiscope planes whose timestamps are staggered.
2. **Common bin is 1/30 s (33.333 ms)**: Papers use a common 30 Hz time series and target format requires equal bin size across sessions. Native single-plane and multiscope plane sampling rates differ, so retaining raw frame indices would violate that requirement.
3. **Grid anchor**: For each trial, start at the first 30 Hz grid point at or after the native trial start, relative to that experiment’s first ophys timestamp; stop before native trial stop. This avoids including pre-trial samples and is explicitly tied to the ophys clock.
4. **Neural interpolation**: Linear interpolation is consistent with paper event-triggered trace processing. No smoothing, normalization, or deconvolution is added.
5. **Image identity**: Class 0 is gray (including inter-image gray and omitted presentations); real images are globally sorted classes 1–16. This follows “image presented during the non-grey screen.”
6. **Image change**: A single sample pulse is “right after” a true identity-change onset and avoids labeling the full changed-image interval as an event.
7. **Behavior quintiles**: Compute percentile edges independently per experiment from samples retained in eligible trials, preventing between-rig calibration and mouse-size differences from dominating labels. Duplicate quantile edges are handled with `searchsorted`, though continuous streams should normally yield all five classes.
8. **Pupil gaps**: Interpolate using only finite positive filtered pupil values. Do not turn missing samples into a category. Exclude only sessions lacking the stream entirely.
9. **Trial filtering**: Include exactly `go | catch`; categories are mutually exclusive, so this already excludes aborted and auto-rewarded rows.
10. **ROI filtering**: Use `valid_roi`; no extra amplitude threshold because files already passed release QC.
11. **Variable duration**: Preserve native start/stop boundaries, yielding variable time lengths but a fixed bin duration. All supplied usable experiments have at least 39 eligible trials.

### Planned Sanity Checks
- [ ] Compare converted neural values at selected trial grid points to direct linear interpolation from original HDF5 using `np.allclose`.
- [ ] Compare empty decoder input shapes to expected `(0, T)` using shape and `np.allclose` on empty arrays.
- [ ] Compare image/change/running/pupil/outcome rows for selected trials to values independently reconstructed from original HDF5 using `np.allclose`.
- [ ] Verify every neural/output trial has identical time length, fixed 1/30 s grid spacing, finite values, and valid class ranges.
- [ ] Verify trial counts equal direct `(go | catch).sum()` for every included experiment.
- [ ] Verify outcome exclusivity and image category vocabulary.
- [ ] Compare subjects, sessions, neurons, regions, outcome distributions, and trial counts with Steps 2–4.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements direct, reproducible NWB/HDF5 conversion and supports `--full` (default), `--sample`, and `--show-processing`. It validates ROI/trace dimensions, trial counts, output lengths, finiteness, and minimum trial count. The script compiles and completed a two-session end-to-end smoke test.

Code inefficiencies identified:
- Loading complete 247 GB NWBs through high-level SDK objects would add unnecessary overhead.
- Per-neuron interpolation loops and repeated loading of source arrays would be costly.
- Output size is inherently large because all event cells and native trial durations are retained at 30 Hz.

Code speedups added:
- Direct HDF5 access reads only required datasets.
- Neural interpolation is vectorized over all cells and each source dataset is loaded once per experiment.
- Trial stimulus lookup uses sorted interval searches.
- Float32 neural/input data and compact integer categories reduce memory and pickle size.
- Smoke test: 2 sessions, 444 trials, 221 neurons in 2.3 s; 60.1 MB output.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 221 |
| Neurons / session | 89, 132 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 444 |
| Trials / session | 39, 405 |
| Trial timepoints | min 218, median 242, max 378 at 30 Hz |
| Image identity distribution | gray 0.670; each of 8 images present approximately 0.037–0.044 |
| Image change distribution | no change 0.9966, one-bin change pulse 0.0034 |
| Running quintiles | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Pupil quintiles | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Trial outcome distribution (time-weighted) | hit 0.0601, miss 0.8128, false alarm 0.0097, correct reject 0.1173 |

### Processing Plots Review
`processing_775614751.png` and `processing_778644591.png` were created. Plots jointly display event activity, non-gray image intervals, single-bin change pulses, continuous running/pupil traces, and their quintile labels on the same 30 Hz trial axis. Shapes and numerical inspection show no offset or length anomalies.

### Format Validation
- `/app/verification_sample_out.txt`: “Data format is valid, no errors or warnings.”
- All neural/output values are finite; each trial has matching neural/input/output time length.
- All specified class ranges are present globally. A single session need not contain every trial outcome, which is expected.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Direct HDF5 and vectorized interpolation | Formal sample completed in 2.4 s |
| Float32 neural arrays and compact labels | Sample output limited to 60.08 MB |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion | 0.4–1.1 s computation plus serialization | Approximately 5–10 minutes conservatively for 284 heterogeneous files and a multi-GB pickle; below 15 minutes |
| Verification | Seconds for 60 MB sample | Several minutes for full output |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: Verification produced none. Training emitted sklearn `y_pred contains classes not in y_true` because the small held-out split lacked some globally defined classes while the model predicted them; this is a small-sample class-coverage warning, not a format defect.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| Image identity | 0.1885 | 0.1820 |
| Image change | 0.5825 | 0.5102 |
| Running speed quintile | 0.2471 | 0.2433 |
| Pupil diameter quintile | 0.2568 | 0.2555 |
| Trial outcome | 0.3543 | 0.3075 |

Uniform chances are 0.0588, 0.5000, 0.2000, 0.2000, and 0.2500 respectively. Every validation score is above chance. Loss decreased monotonically from approximately 1.62 to 1.5358 over 200 epochs; test loss was 1.5361. Image identity is especially strong at over 3× chance. The sparse one-bin image-change task is only slightly above balanced chance in the sample and will be reassessed with the full cohort.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 13.683 GB (13G displayed by `ls`)
- `conversion_full_out.txt`: created; conversion finished in 282.2 s (4.7 min)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | No single total stated | released valid ROIs | 42,147 over all 284 files | 41,871 after excluding 3 pupil-missing sessions | Yes |
| Mean neurons/session | Variable by plane | valid ROI rows | 148.40 over all files | 149.01; min 4, median 67, max 666 | Yes |
| Subjects | 82 in broader paper cohort | metadata IDs | 38 in supplied files | 38 | Yes for supplied subset |
| Sessions | 376 in broader behavior cohort | experiment is imaging plane | 284 experiment files | 281 usable; 3 documented exclusions | Yes |
| Trials (total) | Not stated | go or catch | 84,313 experiment-level eligible trials among usable files | 84,313 | Yes |
| Trials/session | Not stated | native trial table | min 39, median 293, mean 300.05, max 412 | same | Yes |
| Image identity | 250 ms image + 500 ms gray | presentation start/stop | 16 image names plus gray | gray 0.66975; each image ~0.0198–0.0216 | Yes |
| Image change | change onset event | `is_change` | one onset on go/change trials | pulse fraction 0.003403 | Yes |
| Running quintiles | 30 Hz interpolation in paper | filtered speed | continuous | each class 0.2000 | Yes by definition |
| Pupil quintiles | filtered ellipse fits | filtered pupil area | continuous with NaN gaps | each class 0.2000 | Yes by definition |
| Trial outcome | hit/miss/FA/CR | mutually exclusive flags | four outcomes | time-weighted [0.18233, 0.69202, 0.01039, 0.11526] | Yes |
| Regions | VISp and VISl in supplied subset | targeted structure | VISp/VISl | 41,633 / 238 neurons | Yes |

All shape/finiteness spot checks passed. Verification reported no structural errors. It warned that some trials contain all-zero event data; because detected-event traces are extremely sparse, this warning is investigated against original NWBs in Step 10 rather than suppressing or filtering it without evidence.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Full verifier reported no structural errors. Its only warnings were 3,915 all-zero neural trials. Direct reconstruction from original event arrays confirmed these are genuine zero-event intervals (4.64% of 84,313 trials), expected for sparse detected events and low-cell-count planes. Filtering them would violate the required trial selection and bias against quiescent neural periods, so they are retained.
2. **Independent original-data `np.allclose` checks**: `/app/cache/sanity_checks.py` independently loads original HDF5 without importing conversion code. For first, middle, and last converted sessions it reconstructed neural interpolation, empty inputs, all five outputs, running/pupil quintile edges, trial counts, and neuron counts. Trials 0, 5, and last in each session passed `np.allclose`; first ten warning trials were independently verified exactly zero. Report: `/app/cache/sanity_checks_out.txt`.
3. **Reference code comparison**:
   - Loading: direct HDF5 paths correspond to SDK BehaviorOphysExperiment/BehaviorSession data objects.
   - Neuron/trial filtering: released `valid_roi`; exact `go | catch`; SDK/paper outcome booleans.
   - Temporal alignment: all streams sampled on a 30 Hz grid anchored to ophys timestamps; independent boundary audit had zero failures.
   - Binning: linear interpolation matches paper event-triggered 30 Hz processing; no smoothing or normalization added.
   - Input construction: explicit empty `(0,T)` arrays because task specifies no decoder inputs.
   - Output construction: presentation intervals, true change onset, filtered behavior, and trial outcome directly follow native variables. The repeated outcome row is solely a mixed-output format accommodation.
4. **Key statistics comparison**: Raw usable outcomes and converted labels match exactly: hit 15,682; miss 58,051; false alarm 921; correct reject 9,659. There are 73,733 go and 10,580 catch trials. All 73,733 go trials have exactly one change pulse and all catch trials have zero. Subjects, sessions, neurons, regions, class vocabularies, and counts match Step 9.
5. **Edge cases/off-by-one**: Tested every converted trial against the ceil-based start/stop grid formula; zero length mismatches. Checked first/middle/last sessions and first/fifth/last trials. Three entirely missing pupil sessions are documented exclusions; finite gaps in available pupil streams are interpolated from filtered valid samples. Multiplane clocks are not incorrectly concatenated.

### Issues Found and Resolved
- **Initial eligible-count arithmetic**: Initially subtracted auto-rewarded from go+catch. Raw exclusivity audit showed all four trial categories are mutually exclusive; corrected to `go | catch` before script development.
- **Mixed static/time-varying outputs**: Trainer accepts a trial output as wholly 1-D or 2-D. Repeated static outcome across time to preserve semantics while supporting the four time-varying rows.
- **All-zero warnings**: Investigated, reproduced from original arrays, and retained for scientific fidelity. No conversion fix is warranted.
- **Stimulus dynamic-table name**: It is image-set-specific rather than a fixed `stimulus_presentations` path. Script discovers it by required columns.

All checks passed after these resolutions; no unresolved conversion issue remains.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, monotonically from 1.632334 at epoch 1 to 1.577781 at epoch 200.
- Test loss: 1.588543.
- Full script completed successfully and generated sample/prediction plots.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Image identity | 0.1904 | 0.1864 | 3.17× uniform chance (0.0588) |
| Image change | 0.5472 | 0.5287 | Above binary chance (0.5000) |
| Running speed quintile | 0.2258 | 0.2233 | Above five-class chance (0.2000) |
| Pupil diameter quintile | 0.2275 | 0.2237 | Above five-class chance (0.2000) |
| Trial outcome | 0.2922 | 0.2728 | Above four-class chance (0.2500) |

The training run completed all 200 epochs without GPU-memory fallback or runtime error. Full log: `/app/train_decoder_full_out.txt`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Ratio to Chance | Expectation from Papers |
|----------|---------------------|--------|-----------------|-------------------------|
| Image identity | 0.1864 | 0.0588 | 3.169× | Not decoded in paper; strong above-chance result |
| Image change | 0.5287 | 0.5000 | 1.057× | Paper change-vs-repeat decoder is shown graphically, uses balanced selected presentations and first 400 ms, unlike required one-bin onset labels |
| Running quintile | 0.2233 | 0.2000 | 1.116× | Not decoded in paper |
| Pupil quintile | 0.2237 | 0.2000 | 1.119× | Not decoded in paper |
| Trial outcome | 0.2728 | 0.2500 | 1.091× | Paper hit-vs-miss binary decoder is shown graphically; required target has four outcomes and all trial timepoints |

### Accuracy Comparison to Papers
The paper reports change-vs-repeat and hit-vs-miss decoding curves as figures rather than extractable numerical table values. Those tasks are not directly comparable: the paper selects changed presentations and matched preceding repeats or changed hit/miss presentations, and uses neural activity in the first 400 ms. This conversion must instead label every 30 Hz point, use a one-bin change event “right after” onset, and represent four outcomes over full variable-length trials. No paper accuracy exists for 17-class image identity or five-class running/pupil. Accordingly, the paper provides qualitative evidence that change/outcome information is decodable, which the above-chance full results satisfy, but no defensible numeric threshold to copy.

### Train vs Validation Gap
| Variable | Train / Validation Ratio |
|----------|--------------------------|
| Image identity | 1.021 |
| Image change | 1.035 |
| Running | 1.011 |
| Pupil | 1.017 |
| Outcome | 1.071 |

All ratios are far below the 1.5× overfitting threshold; there is no concerning train/validation gap.

### Low-Accuracy Debugging Checks
1. **Raw output values**: Checked specific trials `(session,trial)=(0,0),(140,5),(280,130)` directly in original NWBs. Miss/correct-reject/hit labels and go/catch change-pulse counts matched exactly.
2. **Temporal alignment**: Processing plots place neural, stimulus, change, continuous behavior, and discretized labels on one ophys-anchored grid. Independent reconstruction and exhaustive boundary checks passed.
3. **Variation**: Global distributions contain every class. Change is intentionally sparse (0.3403% of time bins), while percentile outputs are exactly balanced.
4. **Neural filtering**: Uses released valid ROIs and paper-specified detected calcium events. All-zero intervals were independently confirmed in source events.
5. **Reference matching**: Loading, filtering, event processing, interpolation, and trial/outcome definitions were compared in Step 10.

The lower chance ratios for four targets do not indicate misalignment: a one-bin onset precedes delayed calcium dynamics; five quintiles are deliberately difficult fine-grained behavior labels; and static outcome labels include long pre-change periods. Changing these representations to inflate accuracy would violate the explicit decoder-output definitions. Image identity’s 3.17×-chance result and the absence of train/validation gaps provide strong positive alignment evidence.

### Issues Found and Resolved
- No new conversion issue was found. Every requested debugging check passed.
- The apparent low change score was investigated exhaustively; onset pulses exactly match all 73,733 raw go trials and no catch trials, so labels were retained rather than broadened into a biologically delayed response window.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset description, loading instructions, output semantics, statistics, and reproduction commands.
- [x] cache/ folder created with README_CACHE.md.
- [x] Investigation script, sanity-check report, and extracted reference text moved to cache.
- [x] All required conversion, validation, training, plot, and documentation files verified present.
- [x] All workflow steps documented and marked COMPLETE.
