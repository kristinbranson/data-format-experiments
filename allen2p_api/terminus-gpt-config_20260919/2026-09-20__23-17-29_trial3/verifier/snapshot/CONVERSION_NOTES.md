# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior Ophys (AllenSDK cache)
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment checks:
- Python 3, NumPy, and PyTorch imported successfully (see terminal log).
- Mandatory checkpoint passed: `/app/CONVERSION_NOTES.md` exists.

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `allensdk_docs/`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`
- `tutorials/`
- `whitepaper.pdf`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_s3_cache` / `from_local_cache` | `behavior_project_cache.py` | LOADING | Construct the required AllenSDK project cache and access manifest tables. |
| `get_behavior_ophys_experiment` | `behavior_project_cache.py` | LOADING | Return a `BehaviorOphysExperiment` for an ophys experiment ID; this is the only planned session loader. |
| `BehaviorOphysExperiment.dff_traces` | `behavior_ophys_experiment.py` | LOADING | Expose SDK-provided per-cell dF/F traces indexed by cell specimen ID. |
| `BehaviorOphysExperiment.ophys_timestamps` | `behavior_ophys_experiment.py` | LOADING | Expose imaging-frame timestamps used as the master temporal grid. |
| `BehaviorOphysExperiment.cell_specimen_table` | `behavior_ophys_experiment.py` | CURATION | Expose segmented ROI/cell metadata and IDs aligned to traces. |
| `BehaviorSession.stimulus_presentations` | `behavior_session.py` | LOADING | Expose image presentation identity and start/stop times. |
| `BehaviorSession.trials` | `behavior_session.py` | CURATION | Expose trial boundaries, image change time, go/catch and outcome flags. |
| `BehaviorSession.running_speed` | `behavior_session.py` | PROCESSING | Expose SDK-processed running speed with timestamps. |
| `BehaviorSession.eye_tracking` | `behavior_session.py` | PROCESSING | Expose eye/pupil ellipse metrics, timestamps, and blink masking. |
| `DFFTraces.from_nwb` | `data_objects/cell_specimens/traces/dff_traces.py` | LOADING | Load already-computed dF/F through SDK and retain valid ROI IDs; no manual dF/F computation is needed. |
| `OphysTimestamps.from_nwb` | `data_objects/timestamps/ophys_timestamps.py` | LOADING | Load frame times and validate count against trace length. |

### Notes
- `/app/code` is the AllenSDK repository. Its Visual Behavior Ophys documentation explicitly recommends loading and interacting with NWB-backed data through AllenSDK; conversion will not directly open NWB files.
- The public object model is project cache -> `BehaviorOphysExperiment` -> tabular/array properties. Neural traces, stimuli, trials, running, and eye tracking are therefore accessed through one synchronized SDK session object.
- Neural signal choice: use the released `dff_traces`. The SDK reads a precomputed `DfOverF` stream and aligns it to valid ROI/cell IDs. Recomputing delta-F/F from fluorescence would diverge from released/reference processing.
- Imaging-frame `ophys_timestamps` are the required master timebase. Other timestamped streams must be sampled/aligned to these timestamps rather than aligned by row number.
- Trial outcome fields available from `Trials` include hit, miss, false alarm, correct reject, aborted, and auto-rewarded; SDK performance calculations explicitly distinguish non-aborted trials. Task requirements further mandate keeping only go and catch while excluding aborted and auto-rewarded trials.
- `stimulus_presentations` supplies image identity and presentation intervals. Trial records supply `change_time`; both are needed because image identity is time-varying while trial outcome is static.
- Running speed is an SDK-produced timestamped table. Eye tracking includes pupil ellipse measurements and `likely_blink`; blink-invalid pupil values are represented as missing by SDK and must not be treated as real diameters.
- Cell order must always be taken from the SDK's cell specimen index and used consistently for dF/F matrices and brain-region labels.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is an AllenSDK Visual Behavior Ophys S3-cache layout containing a release manifest/metadata and 284 locally cached experiment NWB resources. File names were inventoried only at filesystem level; scientific contents were never opened directly.
- All scientific exploration used `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir="/app/data")` and `get_behavior_ophys_experiment`.
- The local resources comprise 202 active-behavior experiments and 82 passive-viewing experiments. Passive sessions are available but excluded because the requested decoder task requires task trials and trial outcomes.
- Native SDK experiment fields inspected: `dff_traces` (DataFrame, one precomputed array per valid cell), `ophys_timestamps` (1-D float seconds), `cell_specimen_table` (ROI/cell metadata), `trials` (trial bounds and outcome flags), `stimulus_presentations` (presentation bounds, image identity, change/omission fields and stimulus block), `running_speed` (`timestamps`, `speed`), and `eye_tracking` (timestamped pupil/eye ellipse metrics and blink flag).
- Stimulus tables contain multiple blocks in the current release. The SDK warning and table content require selecting only rows whose `stimulus_block_name` contains `change_detection`.
- Example active experiment 775614751: ophys sampling interval 32.31 ms (~31 Hz); running has 287,868 samples and eye tracking 144,962 samples. Eye table includes `pupil_width`, `pupil_height`, `pupil_area`, and `likely_blink`.
- Data types are numeric NumPy/pandas arrays for traces/timestamps/behavior; categorical/string columns for image and metadata; boolean/nullable-boolean trial and validity flags.
- API census completed for all active experiments in 265 s with zero failures.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 29,444 valid SDK cell specimens across 202 active experiments |
| Neurons / session | mean 145.76, range 4-666 per ophys experiment |
| Subjects | See metadata summary below (unique `mouse_id`; active subset) |
| Sessions / subject | Computed from active experiment metadata; sessions represented at ophys-experiment/plane level |
| Trials (total) | 138,638 native trial rows; 51,992 eligible go/catch non-aborted, non-auto-rewarded trials |
| Trials / session | eligible mean 257.39, range 39-409 per active experiment |
| Active experiments | 202 (all loaded successfully) |
| Passive experiments excluded | 82 |
| Go / catch eligible trials | 45,477 / 6,515 |
| Trial outcomes | hit 15,936; miss 29,541; false alarm 931; correct reject 5,584 |
| Aborted / auto-rewarded native rows | 85,831 / 815 |

### Compact Active-Subset Metadata Summary
```text
local_experiments=284 active_experiments=202 passive_experiments=82
active_mice=38 active_ophys_sessions=174 active_behavior_sessions=174 containers=44
project_code {'VisualBehavior': 168, 'VisualBehaviorMultiscope': 34}
session_type {'OPHYS_1_images_A': 55, 'OPHYS_3_images_A': 55, 'OPHYS_4_images_B': 46, 'OPHYS_6_images_B': 46}
experience_level {'Familiar': 110, 'Novel 1': 38, 'Novel >1': 54}
targeted_structure {'VISl': 17, 'VISp': 185}
cre_line {'Slc17a7-IRES2-Cre': 107, 'Sst-IRES-Cre': 62, 'Vip-IRES-Cre': 33}
planes_per_ophys_session {1: 168, 3: 1, 5: 2, 7: 3}
```

### Interpretation of “session”
The atomic neural resource is an `ophys_experiment_id` (one imaging plane with its own cells and timestamps). An `ophys_session_id` can contain multiple simultaneous planes in multiscope data. Counts above are explicitly experiment-level; Step 4-5 will resolve whether to combine simultaneously acquired planes or preserve SDK experiment units based on paper/reference consistency and common timestamp support.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Dataset scope | Active imaging sessions in V1 and LM, image sets A/B, familiar and novel experience levels | Paper methods: “For our behavioral analysis we used all active behavioral sessions from mice in the V1 and LM datasets across all image set experience levels.” |
| Passive sessions | Excluded from paper analyses | Methods: passive viewing “was not analyzed here.” |
| Neural analysis subset in this paper | Primarily familiar-image, multiplane recordings | Methods: “Except where noted we used familiar image set sessions collected on the multiplane calcium imaging rig.” This is paper-analysis-specific, whereas the decoder task requests the full Visual Behavior task. |
| Image presentation interval | 750 ms | Methods define each image-presentation interval as the 750 ms beginning with image presentation. |
| Stimulus timing | Image displayed 250 ms followed by 500 ms gray; changes separated by multiple presentations | Technical whitepaper task description. |
| Ophys sampling | Approximately 31 Hz for single-plane data; multiplane effective plane rate is lower | Whitepaper/acquisition documentation; local example median interval was 32.31 ms. |
| Reference behavior resampling | 30 Hz | Paper methods: event-triggered running traces were linearly interpolated “onto a common 30hz timeseries.” |
| Reward/outcome categories | Hit, miss, false alarm, correct reject; aborted and auto-rewarded are distinct flags | Whitepaper and SDK trial definitions. |
| Local eligible outcome fractions | hit 30.65%, miss 56.82%, false alarm 1.79%, correct reject 10.74% | AllenSDK census of 51,992 eligible experiment-level trials; included here for comparison because papers do not tabulate this exact full-subset distribution. |
| Paper decoder targets | change vs repeat; hit vs miss | Paper Methods, Decoding analysis. |
| Paper decoder evaluation | 5-fold CV random forest, percent test images correct, summarized over imaging planes | Paper Methods, Decoding analysis. |
| Expected decoding result | Change vs repeat decodable above chance for excitatory, Vip, and Sst populations; exact numerical values are only graphical in Figure 6 | Paper Results: “Changes vs. repeats could be decoded equally well for all cell classes...” |
| Neural-choice relation | For excitatory cells, decoder prediction/choice correlation >2x stronger in visual- than timing-strategy sessions | Paper Results surrounding Figure 6B. |

### Processing Details
- **Task structure**: mice perform a go/no-go visual change-detection task. Natural images occur in 750 ms cycles (250 ms image, 500 ms gray). Go trials contain an identity change; catch trials have no change. Outcome depends on licking in the post-change response window.
- **Behavior annotation**: the paper assigns behavioral events to 750 ms image-presentation intervals. Licks are grouped into bouts with a 700 ms inter-lick threshold for its strategy analyses. The requested decoder instead uses SDK trial outcomes directly and continuous time-varying running/pupil streams.
- **Neural acquisition/alignment**: ophys frame timestamps are the authoritative imaging clock. The paper interpolated event-triggered traces to 30 Hz for cross-recording averaging. Because the target explicitly requires alignment based on ophys timestamp, conversion should retain each experiment’s ophys frames and use a common approximately 30 Hz bin/grid.
- **Released dF/F computation**: the whitepaper describes motion correction, ROI segmentation/filtering, demixing/neuropil correction, a 600 s median-filter baseline, noise-floor normalization, and constrained detrending using a 3.33 s median filter. SDK `dff_traces` are therefore the reference processed product and must not be recomputed.
- **Paper neural metric difference**: this paper used detected calcium-event magnitudes for its neural analyses. The decoder specification asks for neural activity generally and the SDK exposes released dF/F; dF/F preserves dense framewise activity and is preferable for time-varying decoding. This intentional difference will be documented in mapping.
- **Running**: paper event analyses linearly interpolated running to 30 Hz. For conversion, SDK running speed will likewise be interpolated to ophys timestamps before percentile discretization.
- **Pupil**: SDK eye tracking provides ellipse width/height/area and a blink validity flag. Pupil diameter should be derived consistently from the ellipse (equivalent-area diameter) or an explicitly justified diameter field, with blink/missing periods excluded before interpolation/binning.
- **Decoder performance**: no exact numerical accuracy is stated in searchable paper text; Figure 6 reports curves graphically. The defensible expectation is above-chance change decoding for each cell class rather than an invented number.

### Curation Steps

**Neuron curation rules**:
- Use only cells/ROIs returned in the SDK `cell_specimen_table` and aligned `dff_traces`; these have passed the released ROI filtering pipeline.
- Preserve imaging-plane experiment boundaries unless Step 4 determines simultaneous planes can be combined on a rigorously common ophys grid.
- Keep both VISp (V1) and VISl (LM), all three cre lines, and all requested active experience levels.
- Do not independently redo segmentation, ROI filtering, neuropil correction, dF/F, or session matching.

**Trial curation rules**:
- Active-behavior sessions only; exclude passive viewing.
- Keep go and catch trials only.
- Exclude aborted and auto-rewarded trials exactly as required.
- Use SDK `start_time`/`stop_time` and outcome flags; use presentation records only from `stimulus_block_name` containing `change_detection`.
- Reject/handle trials with no ophys samples or unusable required streams explicitly, recording counts.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Change vs repeat (paper RF) | Above chance/equally decodable across excitatory, Vip, Sst; exact values only in Figure 6 curves |
| Hit vs miss (paper RF) | Evaluated by 5-fold CV; exact numerical value not stated in extracted text |
| Requested five-output neural decoder | To be measured in Steps 8 and 11 |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session unit | SDK resource is one `ophys_experiment_id` per imaging plane; one `ophys_session_id` may contain several planes | 202 active experiments map to 174 ophys sessions; simultaneous planes have identical trials but plane-specific timestamps | Paper defines an imaging plane (area/depth/session) as the neural sampling unit and reports decoder summaries over planes | Use each ophys experiment/imaging plane as one target session. This gives internally native cell matrices and avoids artificial cross-plane interpolation. |
| Multiplane timing | Each experiment exposes its own `ophys_timestamps` | Seven-plane example has identical trials but ~10.7 Hz timestamps offset by up to ~23 ms between planes | Paper analyzes each plane and interpolates event traces only for population summaries | Do not merge planes. Align every stream to that experiment's native ophys timestamps. |
| Neural representation | SDK exposes released dF/F and detected events | Both streams exist; dF/F is dense and frame-aligned | Whitepaper defines dF/F pipeline; paper’s specific analyses used detected events | Use SDK dF/F because target requires framewise neural activity and because it is a documented released product. Note this intentional decoder-driven difference. |
| Sampling grid | SDK has native plane timestamps (single-plane ~31 Hz, multiplane ~10.7 Hz) | Rates differ by acquisition type | Paper interpolated event/running traces to common 30 Hz for cross-recording analysis | Preserve one native sample per ophys timestamp within each experiment; report nominal bin size. The target requires alignment based on ophys timestamp, and upsampling 10.7 Hz data to 30 Hz adds no information. Time-bin consistency means each session uses its own essentially uniform native frame interval; if validator requires one scalar, metadata reports the dataset convention and per-session rates. |
| Stimulus table | Current SDK table contains multiple stimulus blocks and uses `start_time`/`end_time` | Change-detection rows are explicitly labeled; spontaneous/fingerprint blocks also occur | Paper concerns change-detection task only | Restrict to `stimulus_block_name` containing `change_detection`; use `end_time` for visible-image offset. |
| Trial/change alignment | Trials carry `change_time`; presentations carry `is_change` and onset | Representative go-trial change times equal nearest change-presentation onset exactly (100% within 20 ms; observed delta 0) | Task defines go changes at image onset | Use SDK trial boundaries for segmentation and presentation onsets for time-varying image/change labels. |
| Trial categories | SDK includes aborted/auto-rewarded plus go/catch and outcomes | Eligible flags are mutually consistent: hit+miss are go, false alarm+correct reject are catch | Whitepaper uses the same four outcomes and distinguishes aborted/auto-rewarded | Eligible mask is `(go OR catch) AND NOT aborted AND NOT auto_rewarded`; static outcome is one of four labels. |
| Passive data | Manifest includes active and passive resources | 82 of 284 local experiments are passive | Paper states passive was not analyzed; decoder requires task trials | Exclude all passive experiments. |
| Regions/names | SDK structures are `VISp`, `VISl` | Active subset includes both | Paper calls these V1 and LM | Preserve SDK-standard names `VISp` and `VISl` in `brain_regions`, documenting aliases V1/LM. |
| Dataset scope | Paper neural figures often restrict to familiar multiscope data | Local active cache also includes single-plane and Novel 1/Novel >1 | Decoder says collect the Visual Behavior task; paper behavioral scope includes all active levels | Include all 202 active experiments. A familiar-multiscope restriction would discard requested task data and most sessions without being required. |

### Final Consistent Understanding
1. Load every locally available active Visual Behavior ophys experiment through `VisualBehaviorOphysProjectCache`; never open NWB directly.
2. Treat each imaging plane/ophys experiment as one decoder session, matching the paper's neural analysis unit.
3. Use SDK-curated cells and released dF/F in SDK cell order.
4. Segment on SDK trial `start_time` to `stop_time`; retain only non-aborted, non-auto-rewarded go/catch trials.
5. Align dF/F, image state, changes, running, and pupil using each plane's `ophys_timestamps`.
6. Build stimulus labels only from the `change_detection` block. SDK trial change times and presentation changes are empirically synchronized.
7. Use four static outcomes: hit, miss, false alarm, correct reject.
8. Preserve all active experience levels, VISp/VISl regions, and three cre lines. Passive sessions are excluded.

### Checks Performed
- Loaded all 202 active experiments via SDK: zero failures.
- Compared all trial core columns across seven simultaneous planes: exact equality.
- Compared plane timestamps in the same session: equal nominal rate but plane-specific offsets, validating plane separation.
- Compared trial `change_time` to presentation onset: exact agreement in representative data.
- Verified local metadata counts against paper scope and SDK project/session labels.
- Verified released dF/F processing against whitepaper and SDK implementation.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| SDK `dff_traces.dff` + `ophys_timestamps` | `neural[session][trial]` | Stack in `cell_specimen_table` order; interpolate each trace from native ophys timestamps to a 100 ms grid anchored to trial start; float32 `(n_cells,T)` | `BehaviorOphysExperiment.dff_traces`, `.ophys_timestamps` | Released, curated dF/F; common 10 Hz bins satisfy identical bin size across single-/multiplane experiments without materially upsampling 10.7 Hz planes. |
| No decoder inputs requested | `input[session][trial]` | Empty float32 array `(0,T)` | N/A | Experimental/behavioral variables are decoder outputs by task specification. |
| Change-detection `image_name`, `start_time`, `end_time`, `omitted` | output 0: `image_identity` | Assign image category only while a real image is visibly on screen; gray and omitted intervals are category 0 `none/gray`; actual image names are global categories | `stimulus_presentations` | Filter `stimulus_block_name` containing `change_detection`; presentation `end_time` marks image offset. |
| Change-detection `is_change` / trial `change_time` | output 1: `image_change` | Binary 1 in the single 100 ms bin containing each real image-change onset, else 0 | `stimulus_presentations`, `trials.change_time` | Event representation; SDK alignment check showed exact agreement. Catch trials remain all-zero. |
| `running_speed.timestamps`, `.speed` | output 2: `running_speed_bin` | Linear interpolation to grid; discretize finite aligned values with global 0/20/40/60/80/100 percentile edges into classes 0-4 | `BehaviorSession.running_speed` | Global edges keep class meaning consistent across sessions and approximate equal global occupancy. Negative SDK speeds are retained before binning. |
| `eye_tracking.timestamps`, `pupil_width`, `pupil_height`, `likely_blink` | output 3: `pupil_diameter_bin` | Diameter = `sqrt(width*height)` (geometric mean full-axis diameter, equivalent to `2*sqrt(area/pi)`); invalidate blinks/nonfinite values; interpolate only across short gaps, then global quintile bin to 0-4 | `BehaviorSession.eye_tracking` | Do not bridge long missing/blink periods. Trials/timestamps lacking valid pupil cannot receive a true quintile and will be excluded at session/trial level as documented below. |
| Trial flags `hit`, `miss`, `false_alarm`, `correct_reject` | output 4: `trial_outcome` | Static integer vector `(1,)`: hit=0, miss=1, false_alarm=2, correct_reject=3 | `BehaviorSession.trials` | Exactly one outcome must be true for every retained trial. Decoder may broadcast static output over trial time if required by loader. |
| `mouse_id` | `subjects`, `subject_idx` | Sorted unique string IDs and integer lookup | experiment table / metadata | 38 expected subjects in full active subset. |
| `targeted_structure` | `brain_regions`, `brain_region_idx` | Global `['VISp','VISl']`; repeat session region index for every cell | experiment table / metadata | SDK names retained; aliases are V1 and LM. |
| Experiment/session metadata | `metadata.session_info` | Per-session dict with experiment, ophys/behavior session, mouse, region, depth, cre line, experience, native frame rate, trial/cell counts, exclusions | cache table + experiment metadata | Ensures provenance and allows auditing plane-level sessions. |

### Key Decisions
1. **Session unit is one ophys experiment/imaging plane**: this matches SDK trace/timestamp resources and the paper’s decoding unit. Simultaneous planes are not merged because their timestamps are offset.
2. **Uniform temporal bin is 100 ms (10 Hz)**: required format says all sessions need the same bin size. This is just below the slowest observed native plane rate (~10.7 Hz), avoids information-free upsampling, and supports all time-varying outputs. Grid sample centers are `start_time + 0.05 + 0.1*k` while `< stop_time`.
3. **Variable trial lengths are preserved**: trials are segmented by experimental SDK `start_time`/`stop_time`, not forced into an arbitrary change-centered window. Metadata therefore uses `off_start=None`, `off_end=None`, and alignment event “trial start on ophys clock.”
4. **Ophys-based alignment**: the common trial grid is defined in absolute seconds and all streams are sampled onto it; neural interpolation uses native `ophys_timestamps`, satisfying the explicit temporal-alignment requirement.
5. **Image category during gray/omission**: use `none/gray` because decoder output asks for identity “during the non-grey screen.” Omitted flashes present no image and belong to the same no-image category. Real global image labels are the union of image sets A/B and are encoded consistently across sessions.
6. **Change is an onset event**: exactly one bin immediately containing/after a real identity change is 1. This avoids labeling the entire changed-image presentation as an event.
7. **Continuous-output bins are global quintiles**: percentile edges are learned in a first pass over finite aligned samples from all included sessions, then fixed in a second conversion pass. Global rather than per-session bins preserve comparable class semantics. Duplicate percentile edges are checked and would be handled explicitly.
8. **Pupil diameter**: use geometric mean of SDK full ellipse width and height, an orientation-invariant equivalent-area diameter. SDK `pupil_area` is equivalent but widths/heights make the diameter definition explicit.
9. **Missing pupil**: mark likely blinks invalid; interpolate only between nearest valid samples when the gap is <=0.5 s. Exclude a trial if any requested 100 ms pupil sample remains invalid, because inventing a sixth “missing” category violates the requested five bins. Exclude an experiment only if fewer than two fully valid eligible trials remain. Record all exclusions.
10. **Running gaps**: interpolate within source timestamp support only; reject trials with missing aligned running samples rather than extrapolating.
11. **Cell curation**: retain exactly SDK cells and dF/F rows, after asserting index equality and finite usable traces. No manual re-curation or dF/F recomputation.
12. **Trial filtering**: `(go OR catch) AND NOT aborted AND NOT auto_rewarded`, exactly one of the four outcomes, at least one grid bin, full neural/running/pupil support, and at least two retained trials/session.
13. **Static outcome representation**: store a one-element integer array per trial as allowed by the target schema. If validator requires time-varying rows, conversion will broadcast it only after sample validation demonstrates necessity.

### Output Schema
- `input_names = []`; each input trial has shape `(0,T)`.
- `output_names = ['image_identity','image_change','running_speed_bin','pupil_diameter_bin','trial_outcome']`.
- Time-varying outputs are represented together as a `(5,T)` integer array for validator compatibility; the static trial outcome is repeated across T while metadata marks it static. This is semantically equivalent to a per-trial scalar and avoids ragged mixed output containers.
- `output_values`: image labels (`none/gray` plus sorted global image names), `['no_change','change']`, five running quintile labels, five pupil quintile labels, and `['hit','miss','false_alarm','correct_reject']`.
- `metadata.time_bin_size = 100.0` ms; `temporal_alignment_event = 'trial start, with bins defined in absolute ophys timestamp coordinates'`; `off_start = None`; `off_end = None`.

### Planned Sanity Checks
- [ ] Assert all retained experiments are active and all loads use `VisualBehaviorOphysProjectCache`.
- [ ] Assert every session has >=2 trials, each neural trial is `(n_cells,T)`, input `(0,T)`, output `(5,T)`, and T agrees.
- [ ] Assert 100 ms spacing for every trial grid and bounds are within SDK trial and ophys support.
- [ ] `np.allclose` converted neural spot samples to direct SDK interpolation for at least three sessions/trials/cells.
- [ ] `np.allclose` converted running classes to independently interpolated SDK speed + saved percentile edges.
- [ ] `np.allclose` converted pupil classes to independently computed diameter/interpolation + saved edges.
- [ ] Assert converted change bins coincide with direct SDK presentation `is_change` onset bins.
- [ ] Assert image labels are `none/gray` outside visible real-image intervals and exact SDK identity inside.
- [ ] Assert outcome constant over time and exactly matches one direct SDK trial flag.
- [ ] Compare raw/retained counts and all exclusion reasons to 202 experiments, 29,444 cells, 51,992 initially eligible trials, 38 mice, and VISp/VISl metadata.
- [ ] Check global running and pupil class fractions are approximately 20% each; investigate deviations >2 percentage points.
- [ ] Check image, change, outcome distributions and reward/outcome fractions against Step 2/reference expectations.
- [ ] Verify no NaN/inf remains in neural or categorical outputs and category codes are within declared ranges.
- [ ] Plot native versus aligned neural/running/pupil and stimulus overlays for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with:
- Required CLI modes `--full` (default), `--sample`, and `--show-processing`.
- Scientific loading exclusively through `VisualBehaviorOphysProjectCache`.
- Active-experiment discovery, SDK trial/cell curation, 100 ms ophys-clock alignment, stimulus labels, behavior interpolation, global percentile discretization, categorical encoding, structural assertions, metadata, timing logs, and processing plots.
- Syntax compilation and CLI help checks passed.

Code inefficiencies identified:
- AllenSDK experiment construction dominates runtime (about 1-2 seconds/session in the exploration census).
- Two-pass disk loading was avoided by retaining compact converted trial records in memory until global percentile edges are computed.
- Python loops remain over sessions/trials/presentations because trial lengths and presentation intervals are ragged; heavy neural interpolation is vectorized over cells.

Code speedups added:
- Vectorized neural interpolation for all neurons in a trial.
- One SDK load per experiment and one in-memory percentile/construction pass.
- Float32 neural/behavior arrays and int16 categorical outputs.
- No redundant raw-file access, dF/F recomputation, or cross-plane resampling.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 231 |
| Neurons / session | 89, 142 (mean 115.5) |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 204 retained (39 and 165) |
| Trials / session | mean 102; range 39-165 |
| Trial length | 72-118 bins; mean session lengths 78.8 and 81.6 bins |
| Image identity range | 0-16; `none/gray` fraction 0.661 |
| Image change distribution | no-change 0.9892, change 0.0108 |
| Running quintile distribution | [0.2000, 0.2000, 0.2000, 0.2000, 0.2000] |
| Pupil quintile distribution | [0.2000, 0.2000, 0.2000, 0.2000, 0.2000] |
| Trial outcome time-bin distribution | hit 0.5869, miss 0.2799, false alarm 0.0595, correct reject 0.0737 |
| Neural dtype / output dtype | float32 / int16 |
| Input shape | `(0,T)` as required by no-input task |
| Output shape | `(5,T)` |

### Format Validation
- `/app/sample_data.pkl` created (8.5 MiB).
- `/app/verification_sample_out.txt` created by the required `--verify-only` command.
- Validator completed with no errors or warnings and all declared categorical ranges match observed values.
- Manual inspection confirmed session/trial list lengths, neuron dimensions, region-index lengths, subject indexing, metadata, and time dimensions are consistent.
- Session-specific percentile occupancy is intentionally not equal because global percentile edges preserve consistent physical class semantics; pooled occupancy is equal by construction.

### Processing Plots Review
- `processing_775614751.png` and `processing_788490510.png` were created.
- Plots contain aligned dF/F heatmaps, visible-image codes, one-bin change events, continuous running/pupil with quintile thresholds, and resulting categorical traces on the same trial-relative 100 ms axis.
- No plotting or alignment anomalies were reported. Change labels are sparse as expected (~1.1% of bins), and gray occupancy (~66%) is consistent with 250 ms images in 750 ms cycles.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Vectorized interpolation over all neurons | Avoids per-cell interpolation loops |
| Single SDK load per experiment | Avoids a second I/O pass for global quantiles |
| Four-process full-mode experiment processing | Expected ~3-4x reduction in SDK/processing wall time |
| Float32/int16 compact storage | Reduced memory and pickle I/O |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Serial sample conversion | 5.45 s/session plus ~1 s finalization | ~18.5 min for 202 sessions |
| Four-worker full conversion | Estimated 1.4-2.0 s/session effective | ~5-7 min processing plus serialization, under 15 min |

### Exclusions
- Session 775614751 retained all 39 eligible trials.
- Session 788490510 excluded 25/190 eligible trials due to pupil gaps longer than 0.5 s; 165 remained.
- Both sessions exceed the required minimum of two trials.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None.
- Warnings: One sklearn warning during accuracy scoring: `y_pred contains classes not in y_true`. This is expected for a small two-session validation split in which some rare image/outcome classes are absent from validation ground truth; declared ranges and full sample data contain the classes. No conversion or tensor warning occurred.

### Training Progress
- Training completed all 200 epochs.
- Loss decreased monotonically from approximately 1.65 initially to 1.2516 at epoch 200.
- Test loss: 1.5723.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Above chance? |
|--------|-----------------------|-------------------------|--------|---------------|
| image_identity | 0.4663 | 0.3293 | 0.0588 | Yes (5.60x chance validation) |
| image_change | 0.8471 | 0.6104 | 0.5000 | Yes |
| running_speed_bin | 0.2947 | 0.2614 | 0.2000 | Yes |
| pupil_diameter_bin | 0.3742 | 0.2619 | 0.2000 | Yes |
| trial_outcome | 0.4379 | 0.2659 | 0.2500 | Yes |

All requested outputs exceeded chance on validation. Image identity and change decoding provide especially strong evidence that neural/stimulus alignment is correct. Running and pupil exceed five-class chance despite only two sessions. Trial outcome is modestly above chance and will be reassessed on the full dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 2.47 GiB (2,466.6 MiB reported at write time)
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; validator finished with `Data verification complete.`

### Conversion Performance
- Four-worker processing completed all 202 candidate experiments in 293.5 s; complete construction and serialization took 299.4 s (~5.0 min), well under 15 min.
- No SDK experiment load or conversion exceptions occurred.
- 199 sessions retained; three experiments had fewer than two pupil-valid trials and were excluded under the predeclared QC rule.

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | Paper does not tabulate exact supplied-cache total | SDK valid cell specimens | 29,444 across 202 active experiments | 29,168 across 199 retained experiments | Yes after documented 276-cell/3-session pupil QC exclusion |
| Mean neurons/session | Imaging-plane unit | One experiment/plane | 145.76 (range 4-666) before stream QC | 146.57; validator range includes expected planes | Yes |
| Subjects | Active V1/LM cohort | SDK `mouse_id` | 38 | 38 | Exact |
| Sessions | Active imaging planes | `ophys_experiment_id` | 202 candidate active experiments | 199 retained, 3 excluded for unusable pupil | Explained |
| Trials (eligible total) | Go/catch; aborted/auto-rewarded excluded | SDK flags | 51,992 before stream QC | 48,112 after complete-stream QC | 92.5% retained; 3,880 exclusions documented, principally pupil gaps |
| Trials/session | Variable experimental trials | SDK `start_time`/`stop_time` | mean 257.39 before stream QC | mean 241.77 retained | Consistent after QC |
| Brain regions | V1 and LM | `VISp`, `VISl` | Both present | VISp 29,006 neurons; VISl 162 | Exact naming/scope |
| Image/gray occupancy | 250 ms image + 500 ms gray predicts ~1/3 image, ~2/3 gray | Presentation intervals | Expected | gray 0.6652, real images 0.3348 | Excellent |
| Change distribution | Sparse trial event | `is_change` | Expected sparse | change 0.01035 of bins | Sensible |
| Running bins | Five equal percentile bins | Planned global quantiles | N/A | each 0.200000 | Exact |
| Pupil bins | Five equal percentile bins | Planned global quantiles | N/A | each 0.200000 | Exact |
| Trial outcomes | Four categories | SDK trial flags | initial eligible bin-weighted fractions near hit .307, miss .568, FA .018, CR .107 | bin-weighted hit .3073, miss .5672, FA .0174, CR .1081 | Very close |

### Full Validator Results
- No `error`, `failed`, `traceback`, or validator warning appears in `/app/verification_full_out.txt`.
- Output ranges exactly match declarations: image 0-16, change 0-1, running 0-4, pupil 0-4, outcome 0-3.
- Validator brain-region total: 29,168 neurons (VISp 29,006; VISl 162).
- Manual first/middle/last session spot checks confirmed finite float32 neural arrays, `(0,T)` inputs, `(5,T)` categorical outputs, consistent T, and legal code ranges.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` begins “Data format is valid, no errors or warnings” and ends “Data verification complete.” Grep found no actual error, failure, traceback, or warning. The conversion log contains repeated AllenSDK `UpdatedStimulusPresentationTableWarning`; this is informational and explicitly addressed by filtering `stimulus_block_name` for `change_detection` as instructed by the warning.
2. **Independent source-to-converted sanity checks**: `/app/critical_review.py` independently loaded source experiments through `VisualBehaviorOphysProjectCache` and did not import conversion functions. It reconstructed first/middle/last retained trials in first/middle/last converted sessions (nine trials total). `np.allclose` passed for neural dF/F interpolation, image codes, change events, running bins, pupil bins, and outcomes. Trial counts/order also matched. Maximum neural absolute differences were 4.58e-8 to 1.30e-7, consistent with float32 rounding. Report: `/app/critical_review_out.txt`.
3. **Reference-code comparison**: each major stage was compared as detailed below; no unresolved mismatch remains.
4. **Key-statistics comparison**: all available paper/code/data statistics were compared in Step 9. Gray/image occupancy matches 500/250 ms timing; global quintiles are exact; subjects/regions and pre-QC experiment/cell/trial totals reconcile exactly with source metadata and exclusions.
5. **Edge-case review**: checked trial first/last bin centers, support bounds, variable trial lengths, first/middle/last trial ordering, static outcome constancy, category bounds, finite values, session minimum trial count, zero-input shapes, low-cell planes, multiplane timestamps, omitted stimuli, blink gaps, and source-stream support.

### Major Processing-Step Comparison
| Stage | Conversion implementation | Reference SDK/paper implementation | Comparison/result |
|-------|---------------------------|------------------------------------|-------------------|
| Data loading | `VisualBehaviorOphysProjectCache.from_s3_cache` and `get_behavior_ophys_experiment` only | SDK documentation recommends project cache/session objects | Exact; no direct NWB access |
| Neuron filtering | SDK `cell_specimen_table` index aligned to SDK `dff_traces` | Whitepaper ROI segmentation/filtering; SDK returns valid cells | Exact released curation; no redundant filtering |
| Trial filtering | go/catch, not aborted, not auto-rewarded, unique four-way outcome, complete streams | SDK flags and task requirement | Exact; additional complete-stream QC required by five outputs |
| Temporal alignment | Absolute 100 ms trial grid; interpolate from `ophys_timestamps` and behavior timestamps | Paper interpolates event/running traces to common grid; target mandates ophys alignment and same bin size | Consistent; 10 Hz avoids artificial oversampling of 10.7 Hz multiplane data |
| Neural binning | Linear sample of released dF/F at 100 ms centers | Whitepaper defines released dF/F; paper-specific analysis used events | Intentional decoder-driven difference, documented and justified |
| Input construction | `(0,T)` because task specifies no decoder inputs | Target specification | Exact |
| Image output | Real identity only during SDK visible interval; `none/gray` otherwise | 250 ms image + 500 ms gray; current SDK multi-block table | Exact task semantics; omissions correctly no-image |
| Change output | One 100 ms event bin at SDK `is_change` onset | SDK `change_time` equals presentation onset; target says “right after a change” | Exact; direct checks passed |
| Running output | SDK speed interpolated, global quintiles | Paper interpolates running to common 30 Hz; task requests five percentile bins | Same alignment principle; common 10 Hz grid required by slowest imaging planes |
| Pupil output | SDK blink-masked ellipse diameter, short-gap interpolation, global quintiles | SDK eye table marks blink-invalid fits; task requests pupil diameter quintiles | Consistent and conservative |
| Outcome output | Four SDK flags, constant across each trial | SDK/whitepaper outcome definitions | Exact; direct checks passed |

### Key Statistics and Edge Cases
- Candidate/retained reconciliation: 202 active experiments and 29,444 cells before complete-stream QC; 199 experiments and 29,168 cells after excluding experiments 795953296, 806456687, and 833631914, whose eligible trials all lacked valid pupil coverage. Thus exactly 3 sessions and 276 cells are accounted for.
- Trial reconciliation: 51,992 initially eligible trials and 48,112 retained (92.54%). Exclusions are recorded per session in metadata and principally reflect blink/missing pupil gaps >0.5 s. No trials were silently lost.
- All retained sessions have 39-393 trials, well above the required minimum of two.
- Native rates range 10.726-30.950 Hz, but every converted bin is exactly 100 ms by construction.
- Trial grids use centers at start+50 ms through the last center strictly before stop. Independent checks passed both first-center-after-start and last-center-before-stop criteria.
- `omitted` presentations are encoded as `none/gray`, not an image; changes on omitted rows are not labeled as real image changes.
- Global percentile edges are strictly increasing. All five classes have exactly 20% pooled occupancy up to single-bin rounding.
- Low-cell multiplane sessions (minimum four cells) are retained because cells are SDK-valid and the target imposes no minimum neuron count.

### Issues Found and Resolved
- **Initial planning ambiguity about native frame rates**: Step 4 considered preserving native ~31/~10.7 Hz rates, but the target explicitly requires the same bin size across sessions. Step 5 corrected this before implementation by choosing a uniform 100 ms grid. Rechecks confirm exact shape and alignment consistency.
- **Current SDK stimulus table has multiple blocks**: resolved by the SDK-prescribed `change_detection` filter. Gray/image occupancy and direct image checks validate the fix.
- **Three sessions have unusable pupil streams**: excluded because assigning invented quintiles would violate the requested five-bin output. All IDs/counts are recorded; all 38 subjects remain represented.
- **Large full pickle (2.47 GiB)**: expected from preserving dense dF/F for 48,112 trials and 29,168 plane-level neurons. Validator loads it successfully; float32 and int16 already minimize size without lossy compression.

### Final Recheck
`ALL_INDEPENDENT_CHECKS_PASS True`; full validator passes; no unresolved issue remains.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Full command completed successfully on CUDA with `--plot-samples`.
- Training/test split: 38,406 / 9,706 trials.
- Loss decreasing: Yes, smoothly from 1.6410 at epoch 1 to 1.3103 at epoch 200.
- Test loss: 1.4597.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| image_identity | 0.4072 | 0.3722 | 0.0588 | 6.33x chance; strong temporal/stimulus alignment evidence |
| image_change | 0.7507 | 0.6457 | 0.5000 | Above chance; paper also reports decodable changes |
| running_speed_bin | 0.3957 | 0.3713 | 0.2000 | 1.86x chance |
| pupil_diameter_bin | 0.4423 | 0.4104 | 0.2000 | 2.05x chance |
| trial_outcome | 0.4399 | 0.2923 | 0.2500 | Above chance but modest; investigated in Step 12 |

The script finished normally and `/app/train_decoder_full_out.txt` contains the complete run.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Ratio to Chance | Expectation from Papers |
|----------|--------------------:|-------:|----------------:|-------------------------|
| image_identity | 0.3722 | 0.0588 | 6.33x | Not directly decoded in paper; strong above-chance result expected from visual cortex |
| image_change | 0.6457 | 0.5000 | 1.29x | Paper reports change vs repeat is decodable across all cell classes; no exact text value |
| running_speed_bin | 0.3713 | 0.2000 | 1.86x | No directly comparable paper decoder; locomotion modulation is expected |
| pupil_diameter_bin | 0.4104 | 0.2000 | 2.05x | No directly comparable paper decoder; arousal modulation is expected |
| trial_outcome | 0.2923 | 0.2500 | 1.17x | Paper hit decoder predicts hit vs miss only on image changes, a simpler binary task |

### Check 1: Accuracy vs Chance
- Every output is above chance. Image identity, running, and pupil exceed 1.5x chance.
- **Image change investigation**: 1.29x chance. Converted change is intentionally a one-bin event and only 1.035% of all bins. The validator performs framewise binary decoding over whole trials; the paper instead balances each image change against the immediately preceding repeat, concatenates image-window activity, and uses a random forest. Thus the tasks and chance-adjusted difficulty differ substantially. Independent raw checks for three trials and nine broader Step 10 checks showed exact event-vector agreement. Image identity at the same timestamps is decoded at 6.33x chance, ruling out broad temporal misalignment.
- **Trial outcome investigation**: 1.17x chance. Outcome is static over a long trial but the validator trains/evaluates each local frame, while behavior outcome depends primarily on change/response-period activity. The four-way task includes rare false alarms (1.82% of retained trials) and correct rejects (10.70%), unlike the paper's binary hit/miss-at-change decoder. Three direct raw trials and nine Step 10 trials exactly match SDK flags. There is ample class variation: trial counts [14,979 hit, 27,107 miss, 878 false alarm, 5,148 correct reject]. No conversion change is justified.

### Check 2: Accuracy Comparison to Papers
- The paper’s exact decoder accuracies are graphical in Figure 6 and not stated numerically in extracted text; inventing values would be inappropriate.
- Qualitative comparison matches: both paper and converted dataset show above-chance change information in neural activity.
- Algorithms/tasks differ in explicitly material ways: paper uses paired changes/repeats or hit/miss changes, detected calcium events, per-plane random forests, and 5-fold CV; provided validator uses dF/F, whole-trial framewise multi-output neural network decoding. The full result is therefore compared qualitatively rather than as a claimed numeric replication.
- The strong image/running/pupil results and above-chance change/outcome results are consistent with correctly aligned visual and behavioral signals.

### Check 3: Train vs Validation Gap
| Output | Train | Validation | Train/Validation | Assessment |
|--------|------:|-----------:|-----------------:|------------|
| image_identity | 0.4072 | 0.3722 | 1.094 | Small gap |
| image_change | 0.7507 | 0.6457 | 1.163 | Small gap |
| running_speed_bin | 0.3957 | 0.3713 | 1.066 | Small gap |
| pupil_diameter_bin | 0.4423 | 0.4104 | 1.078 | Small gap |
| trial_outcome | 0.4399 | 0.2923 | 1.505 | Threshold-level gap investigated |

The outcome ratio barely exceeds 1.5. Investigation found no data leakage: train/test split is performed by the provided decoder, labels come directly from SDK flags, outcomes are constant within each trial, and all four classes occur. The likely cause is limited generalization of a framewise model for a trial-level behavioral event plus rare catch outcomes. Altering correct labels to improve validator accuracy would be scientifically unjustified.

### Targeted Raw Checks
- `/app/check_three_trials.py` loaded sessions 0, 99, and 198 directly with AllenSDK and checked converted trials 0, 101, and 127.
- Raw outcomes were miss, hit, hit and exactly matched converted codes 1, 0, 0; each converted outcome was constant over time.
- Each raw trial had one image-change event and each converted vector had the same one event at the same bin (`np.allclose=True`).
- Step 10 additionally checked nine trials for all five outputs and neural data.

### Issues Found and Resolved
- No conversion bug was found in low-accuracy outputs.
- Sparse framewise change representation is required by “right after a change”; broadening it merely to increase accuracy would violate semantics.
- Static four-way outcome is required by the target; reducing to paper-style hit/miss would omit catch outcomes and violate instructions.
- All checks were rerun conceptually against the unchanged, validated conversion; no reconversion was necessary.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset description, loading instructions, format, statistics, and validation results
- [x] cache/ folder created
- [x] `cache/README_CACHE.md` documents investigation artifacts
- [x] Analysis scripts and temporary census files moved to cache
- [x] All required deliverables verified
- [x] CONVERSION_NOTES.md completed through both critical reviews
