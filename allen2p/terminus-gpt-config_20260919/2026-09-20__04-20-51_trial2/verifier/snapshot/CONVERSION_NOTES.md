# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P (provided project data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

- Verified `/app/CONVERSION_NOTES.md` exists before exploration.
- Environment: Python 3.13.15, NumPy 2.4.4, PyTorch 2.6.0+cu124; imports succeeded.

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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `BehaviorOphysExperiment.from_nwb` | `behavior_ophys_experiment.py` | LOADING | Constructs a complete ophys session from NWB-backed data objects. |
| `Trials.from_nwb` / trial parsing | `data_objects/trials/trials.py`, `trial.py` | LOADING / CURATION | Loads trial start/stop/change timing and derives go, catch, aborted, auto-rewarded, hit, miss, false-alarm, correct-reject and response latency fields. |
| `StimulusPresentations.from_nwb` | `data_objects/stimuli/presentations.py` | LOADING / PROCESSING | Loads presentation intervals, sorts by `start_time`, and derives `is_change` and flashes-since-change when requested. |
| `OphysTimestamps.from_nwb` / `validate` | `data_objects/timestamps/ophys_timestamps.py` | LOADING / CURATION | Loads imaging-frame timestamps and reconciles timestamp count with trace frame count (clips excess timestamps; errors on insufficient timestamps). |
| `DFFTraces.from_nwb` | `data_objects/cell_specimens/traces/dff_traces.py` | LOADING | Reads already processed per-cell delta-F/F traces from NWB; conversion should not recompute dF/F. |
| `Events.from_nwb` | `data_objects/cell_specimens/events.py` | LOADING / PROCESSING | Reads event-detection output and exposes event and filtered-event traces aligned to ophys frames. |
| `CellSpecimens.from_nwb` | `data_objects/cell_specimens/cell_specimens.py` | LOADING / CURATION | Builds the cell specimen/ROI table and aligns traces/events by cell ROI/specimen identifiers. |
| `RunningSpeed.from_nwb` | `data_objects/running_speed/running_speed.py` | LOADING | Loads timestamped running speed table. |
| `EyeTrackingTable.from_nwb` | `data_objects/eye_tracking/eye_tracking_table.py` | LOADING / PROCESSING | Loads processed eye tracking, including pupil measurements and blink flags. |

### Notes
- `/app/code` is an AllenSDK source checkout (1309 files, 121 MB). The SDK documentation recommends loading the Visual Behavior Optical Physiology NWB files through AllenSDK.
- The experiment is two-photon calcium imaging during a go/no-go visual image-change detection task.
- Native public interfaces expose `ophys_timestamps`, `dff_traces`, `events`, `cell_specimen_table`, `trials`, `stimulus_presentations`, `running_speed`, and `eye_tracking`.
- dF/F is already computed and stored in NWB. It does **not** need to be recomputed. Inferred events are also available as a separate processed representation; the final neural choice will be resolved against the paper/methods in Steps 3-5.
- Trial type and outcome flags are derived by the SDK from behavioral logs. Required curation can directly select `(go | catch) & ~aborted & ~auto_rewarded`; outcomes are represented by hit/miss/false_alarm/correct_reject.
- Stimulus timing and behavioral timing are in the common session clock. For this task all streams will ultimately be sampled at `ophys_timestamps`, as explicitly required.
- Stimulus presentations include image identity, presentation start/stop and `is_change`; omitted flashes receive special postprocessing in SDK and should remain distinguishable from actual images/gray screen.
- Ophys timestamp validation is important for avoiding frame-count mismatches: the SDK trims excess timestamps to the number of trace frames and rejects too few timestamps.
- Running speed and eye tracking remain timestamped continuous streams; SDK does not automatically force them onto ophys frames, so explicit temporal interpolation/resampling is needed for decoder outputs.
- No electrophysiology quality filtering applies. Ophys cell/ROI validity and any paper-specific curation must be honored rather than inventing ephys criteria.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains a local subset of Allen Visual Behavior Ophys release 1.1.0: 284 `behavior_ophys_experiment_<id>.nwb` files, four project metadata CSV tables, a project manifest JSON, a checksum/path JSON, and `_manifest_last_used.txt`.
- NWB is the native format. Each file is one ophys experiment/imaging plane and includes session/mouse metadata, trials, stimulus presentations/templates, ophys dF/F and event data, running data, and processed eye tracking when available.
- Metadata tables: `behavior_session_table.csv`, `ophys_session_table.csv`, `ophys_experiment_table.csv`, and `ophys_cells_table.csv`. The manifest catalogs the full release (6,015 remote data files); only the 284 supplied local NWBs are in scope.
- Native dF/F dataset path is `processing/ophys/dff/traces/data`, shaped `(n_ophys_frames, n_cells)`. Across files it has 48,284–149,508 frames and 4–666 cells. Conversion must transpose trial slices to required `(neurons, time)`.
- Running speed is timestamped and present in all 284 files (`processing/running/speed/{data,timestamps}`); unfiltered speed is also present.
- Trial tables contain start/stop/change times and `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, and `correct_reject` flags. Stimulus presentations provide image timing/identity; natural-image template sets contain eight images.
- A few acquisition sessions have multiple simultaneously acquired imaging-plane experiment NWBs. Their behavioral trial tables have identical counts. Native per-experiment recording units are retained as distinct decoder sessions because each has its own neuron population; acquisition-session-level counts are reported separately to avoid double-counting behavior.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Local NWB ophys experiments | 284 |
| Unique ophys / behavior acquisition sessions | 247 / 247 |
| Subjects | 38 mice |
| Experiments / acquisition session | 1 for 239 sessions; 3 for 1; 4 for 1; 5 for 2; 7 for 4 |
| Neurons (experiment-specific cell ROI rows) | 42,147 |
| Unique tracked cell specimen IDs | 16,337 |
| Neurons / experiment | mean 148.40; median 66; min 4; max 666 |
| Ophys frames (total across experiments) | 35,701,771 |
| Frames / experiment | mean 125,710; min 48,284; max 149,508 |
| Raw trials (unique acquisition sessions) | 148,231 |
| Raw trials / acquisition session | mean 600.13; min 399; max 1,241 |
| Raw trial rows across experiment NWBs | 171,887 (behavior duplicated for multi-plane sessions) |
| Required go+catch trials (unique acquisition sessions) | 74,476 = 65,128 go + 9,348 catch |
| Required go+catch trial rows across experiment NWBs | 85,230 = 74,538 go + 10,692 catch |
| Excluded trials (unique acquisition sessions) | 73,080 aborted + 675 auto-rewarded |
| Brain regions (experiments) | VISp: 261; VISl: 23 |
| Cre lines (experiments) | Slc17a7: 153; Sst: 85; Vip: 46 |
| Experience levels (experiments) | Familiar: 150; Novel >1: 96; Novel 1: 38 |

### Data Integrity Checks
- All 284 local experiment IDs occur in `ophys_experiment_table.csv`.
- NWB dF/F cell dimensions sum to 42,147, exactly matching local experiment rows in `ophys_cells_table.csv`.
- All 284 files contain timestamped filtered running speed.
- For every multi-plane acquisition session, trial aggregate counts match across its experiment files (zero discrepancies).
- Trial classes partition the unique raw trials exactly: go + catch + aborted + auto-rewarded = 148,231.
- Corrected an exploration-only axis-label mistake immediately: NWB stores dF/F as time × cells, not cells × time.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote / Context |
|-----------|-------|------------------------|
| Trial design | 87.5% go, 12.5% catch among planned valid transitions | Whitepaper: “GO trials comprise 87.5% ... CATCH trials comprise 12.5%”. Local retained trial rows closely match this design. |
| Stimulus cadence | 750 ms image-presentation interval | Paper Methods: behavioral events assigned to each 750 ms image presentation interval. |
| Neural acquisition rate | 31 Hz single-plane; 11 Hz multi-plane | Whitepaper event-detection section distinguishes Scientifica 31 Hz and Multiscope 11 Hz experiments. |
| Eye / behavior acquisition | 30 Hz | Whitepaper acquisition QC section: eye tracking and behavior recorded at 30 Hz. |
| Imaging planes / session | Up to 8 for multi-plane | Whitepaper: up to 8 experiments per session; fewer when planes fail QC. |
| Behavioral strategy model | Mean cross-validated AUC 0.83 across 382 sessions | Paper Fig. 2; this is a behavioral model, not the neural decoder requested here. |
| Neural decoder window | First 400 ms after image presentation | Paper Fig. 6 caption. |
| Neural decoder accuracy | Plotted, no single numeric value in text | Paper Fig. 6 reports percent-correct change-vs-repeat and hit-vs-miss random-forest results by plane/cell type. |
| Local subset size | 38 mice, 247 acquisition sessions, 284 experiments, 42,147 ROI rows | Directly measured in Step 2; references describe the broader released dataset rather than this supplied subset. |

### Processing Details
- Task: mice lick after an image-identity change in a go/no-go task. Image presentations last 250 ms and are separated by 500 ms gray periods, yielding 750 ms presentation intervals. Change times are sampled from a truncated exponential schedule.
- Reference neural analyses use **detected calcium events**, each with time and magnitude, rather than recomputing dF/F. The NWBs provide frame-aligned event traces and dF/F. To best match the paper, filtered detected events will be the decoder neural signal unless a data-integrity check shows them unavailable.
- Paper neural decoding concatenated event activity in the first 400 ms after each image presentation and used 5-fold cross-validated random forests. This task instead mandates whole-trial time series and a provided neural decoder, so the representation can match while the segmentation/model necessarily differ.
- Behavior was assigned to 750 ms image-presentation intervals. For this conversion, image identity and change labels will be constructed from exact stimulus-presentation intervals and sampled on ophys timestamps.
- Single-plane experiments are approximately 31 Hz and multi-plane experiments approximately 11 Hz. The target requires one common bin size; explicit resampling onto a common ophys-time grid will therefore be needed and justified in Step 5.
- Eye tracking and behavior were acquired near 30 Hz. Running and pupil streams must be interpolated to ophys-aligned bins before percentile discretization.

### Curation Steps

**Experiment / neuron curation rules**:
- Whitepaper experiment QC includes visual review of movie quality, motion and z-drift; experiments with z-drift above 10 µm, hardware/stimulus failures, or possible epileptiform events could be excluded.
- Segmentation excludes cells within 3 µm of FOV boundaries. ROI filtering removes duplicate ROIs (>70% overlap), union ROIs, non-somatic objects, and invalid ROIs. Neuropil subtraction failures led to removal of about 1% of ROIs.
- Invalid ROIs are not included in final matched cells; supplied NWB experiment/cell tables represent released post-QC data, so additional ad hoc activity thresholds would double-filter and are not justified.
- Multi-plane sessions can have fewer than eight experiment files because failed planes are omitted during release QC.

**Trial curation rules**:
- Use SDK/NWB trial flags. Include go and catch; exclude aborted and auto-rewarded exactly as required.
- Outcome mapping: hit/miss for go and false alarm/correct reject for catch.
- Passive sessions were not analyzed in the reference paper’s active behavioral analyses. Their applicability to required go/catch trial conversion must be resolved empirically in Step 4 because the local files still contain trial tables.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Image change vs repeat | Reported graphically as percent correct by cell class/plane; exact scalar not stated |
| Hit vs miss | Reported graphically as percent correct by cell class/plane; exact scalar not stated |
| Behavioral strategy dynamic model | Mean AUC 0.83 (not a neural decoder and not directly comparable) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session unit | SDK loads one `BehaviorOphysExperiment` NWB at a time | 284 experiments map to 247 acquisition sessions; trial tables duplicate across sister planes | Up to 8 planes/experiments per session; failed planes omitted | Treat each experiment as a decoder session because it has a distinct simultaneous neuron population; report acquisition-level behavior without duplicates. |
| Passive sessions | SDK exposes trial-like tables for passive files | Passive rows are almost entirely miss/correct-reject because no task responses occur | Passive viewing was not analyzed for active behavior/neural task analyses | Exclude all passive experiments. Their nominal outcomes are not genuine task outcomes and would create label artifacts. |
| Neural representation | NWB provides dF/F and detected events | Event and dF/F arrays match in all 284 files | Paper neural analyses use detected calcium events | Use detected event magnitude at ophys timestamps; do not recompute dF/F. |
| Frame rate | Ophys timestamp spacing varies by rig | Active: 168 experiments at ~30.94 Hz; 34 at ~10.73 Hz | 31 Hz Scientifica, 11 Hz Multiscope | Agreement is exact. Resample to a common target bin in Step 5. |
| Trial proportions | Trial flags partition tables exactly | Active retained trials are 87.47% go and 12.53% catch | Designed 87.5% go / 12.5% catch | Agreement. Use explicit flags, not inferred transition identities. |
| Pupil availability | SDK eye table can be absent | Eye tracking exists in 281/284 experiments; 3 active experiments lack it | Eye tracking acquired at 30 Hz but QC/missingness can occur | Mandatory pupil output cannot be fabricated; exclude experiment/session units without a valid eye stream (final exact rule in Step 5). |
| Pupil invalid samples | SDK exposes blink flag and ellipse axes | Mean 99.1% of pupil rows finite; mean blink fraction 3.46% | Eye processing includes QC/blink handling | Treat blink/nonfinite samples as missing and temporally interpolate only within valid support; do not label blinks as a diameter category. |
| Ophys timestamp exploratory check | Explicit SDK path is frame timestamps | A broad investigation glob accidentally selected unrelated timestamps and appeared as 1 Hz | Reference predicts 11/31 Hz | Fixed check to use `processing/ophys/dff/traces/timestamps`; observed 10.73/30.94 Hz. Conversion code will use exact path only. |

### Final Consistent Understanding
- The supplied NWBs are already post-release-QC, including experiment and ROI filtering. No extra neuron activity threshold is warranted.
- Use active task experiments only, detected calcium events as neural data, NWB trial flags for curation/outcomes, stimulus-presentation timing for image outputs, and running/eye timestamp streams aligned to ophys time.
- Go/catch and outcome identities are exact in all 174 active acquisition sessions: `hit + miss = go`, `false_alarm + correct_reject = catch`, and trial classes partition every row.
- Event and dF/F shapes match in all files. Event timestamps equal ophys/dF/F timestamps in checked files.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/event_detection/data` | `neural` | Select trial interval, resample event magnitudes into 100 ms bins, transpose time×cell to cell×time, float32 | `Events.from_nwb`, `OphysTimestamps.from_nwb` | Matches paper use of detected calcium events. |
| none | `input` | Empty float32 array `(0, T)` | Decoder supports zero input dimensions | Task explicitly specifies no decoder inputs. |
| stimulus presentation `image_name`, start/stop | output 0: image identity | Assign each ophys-aligned bin the presented image category; gray gaps category 0 | `StimulusPresentations.from_nwb` | Eight images pooled by identity labels plus gray; omitted periods remain gray because no image is shown. |
| stimulus presentation `is_change` / trial `change_time` | output 1: image change | Binary impulse in first bin at/after true change presentation onset | `is_change_event`, trial parsing | One-bin event “right after” change, not a sustained label. |
| `processing/running/speed` data/timestamps | output 2: running speed bin | Linear interpolation to bin centers, then session-specific quintiles 0–4 | `RunningSpeed.from_nwb` | Quintile edges fit using all valid retained-session time samples, avoiding trial-specific rank leakage. |
| EyeTracking pupil ellipse axes/timestamps + blink | output 3: pupil diameter bin | Mark blink/nonfinite invalid; derive equivalent diameter from ellipse (`sqrt(width*height)` because the stored values are ellipse diameters), interpolate valid values, session-specific quintiles 0–4 | `EyeTrackingTable.from_nwb` | Equivalent circular diameter uses both pupil axes and is robust to ellipse orientation. |
| trial flags `hit`, `miss`, `false_alarm`, `correct_reject` | output 4: trial outcome | Encode 0–3 and broadcast across T | `Trials.from_nwb` | Broadcast is required because target array mixes four time-varying outputs with one static output. |
| mouse ID | `subjects`, `subject_idx` | Sorted string IDs and per-experiment index | NWB `general/subject/subject_id` / metadata CSV | Only retained active experiments. |
| `targeted_structure` | `brain_regions`, `brain_region_idx` | Regions `VISp`, `VISl`; repeat experiment region for every neuron | experiment metadata table | Each NWB is one imaging plane/region. |

### Key Decisions
1. **Session unit = ophys experiment/imaging plane**: each file has a distinct simultaneous neuron population, matching paper decoding by plane. Multi-plane behavioral rows are intentionally repeated across distinct neural sessions.
2. **Curation**: exclude all passive experiments; retain only `(go | catch) & ~aborted & ~auto_rewarded`; exclude the three active single-plane experiments with no eye stream because pupil is mandatory and cannot be reconstructed. Keep all released post-QC ROIs and require at least two retained trials.
3. **Neural signal**: use released detected event magnitudes, not dF/F, matching the paper. No additional smoothing or activity threshold.
4. **Common bin = 100 ms**: this is supported by both 31 Hz and 11 Hz experiments (about 3 and 1 native frames/bin), preserves 250 ms image flashes and 400 ms reference decoding windows, and keeps variable trial lengths manageable. Bin centers begin at trial start + 50 ms and stop before trial stop; event magnitudes are averaged over native samples in each bin (nearest interpolation fallback only for empty bins).
5. **Native variable trial boundaries**: retain full SDK/NWB start-to-stop intervals (7.02–12.61 s), yielding variable T but identical 100 ms bin width. Alignment event is trial start (`off_start=0`, `off_end=None`).
6. **Image categories**: use a global deterministic vocabulary with gray as 0 and the released natural-image names thereafter. Gray includes inter-flash gaps and omission intervals because requested identity is the image shown during non-gray screen.
7. **Change impulse**: use actual change time/is_change onset and mark exactly one 100 ms bin. Catch trials contain no identity change and remain zero.
8. **Continuous discretization**: compute quintile edges independently for each experiment/session from valid aligned values; collapse duplicate quantile edges safely if needed, though variation is expected. Store categorical labels 0–4 with output names `0-20%` ... `80-100%`.
9. **Missing behavior samples**: linear interpolate only between valid samples. Edge values may use nearest valid sample within the stream. Drop a trial if running or pupil has no valid support over its interval; do not create a missing category because outputs must remain five percentile bins.
10. **Outcome representation**: categories are hit, miss, false alarm, correct reject. Broadcasting a static outcome over time is semantically equivalent and required by the trainer’s homogeneous output matrix.
11. **Storage**: event/neural float32; categorical outputs int64; empty input float32. Use direct h5py reads of released NWB paths, reproducing the SDK paths while avoiding costly full PyNWB object construction.

### Planned Sanity Checks
- [ ] Direct raw-vs-converted neural comparison using `np.allclose` for selected experiment/trial/neuron/bin means.
- [ ] Confirm each converted trial is contained in raw start/stop and bin count is `floor((stop-start)/0.1)`.
- [ ] Direct raw stimulus-vs-converted image identity and one-bin change impulse checks with `np.allclose`.
- [ ] Direct raw running/pupil interpolation and quintile-label checks with `np.allclose` on selected trials.
- [ ] Verify output ranges: image vocabulary, change {0,1}, running/pupil {0..4}, outcome {0..3}.
- [ ] Verify every go trial outcome is hit/miss and every catch is false alarm/correct reject.
- [ ] Check quintile distributions are approximately balanced session-wide and go fraction remains ~87.5%.
- [ ] Check no NaN/Inf, all neural/output time dimensions agree, all sessions have >=2 trials.
- [ ] Compare retained mice/experiments/cells/trials to independently computed raw counts.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py` with required positional output path and `--full` (default), `--sample`, and `--show-processing` modes.
- Syntax compilation and CLI help completed without errors.
- Implemented direct, documented NWB HDF5 path loading equivalent to the SDK data objects, active/eye/trial curation, 100 ms event binning, stimulus construction, behavior interpolation, quintile discretization, outcome broadcasting, metadata assembly, internal assertions, timing output, and processing plots.
- Processing plots show neural events, image/change timing, continuous and discretized running/pupil streams, and static outcome for a representative trial.

Code inefficiencies identified:
- Full PyNWB/AllenSDK object construction would load unnecessary images/templates and incur high overhead for 199 large NWBs.
- Per-bin boolean masks over all native timestamps would scale poorly.

Code speedups added:
- Stream one NWB at a time with h5py and read only required datasets.
- Use `np.searchsorted` for trial/frame and stimulus timing lookup.
- Store neural arrays as float32 and categorical arrays as int64.
- Build each experiment once and close its file before proceeding; no redundant full-file reads during conversion.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions / experiments | 2 |
| Neurons (total) | 231 |
| Neurons / session | 89, 142 |
| Subjects | 1 |
| Trials (total) | 229 |
| Trials / session | 39, 190 |
| Time bins / trial | 72–125 (100 ms each) |
| Neural event magnitude range | [0, 0.71093] |
| Image identity distribution | gray 0.662; 16 images collectively 0.338 |
| Image change distribution | no-change 0.9893; change 0.0107 |
| Running quintiles | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Pupil quintiles | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Outcome distribution (time-weighted) | hit 0.595; miss 0.274; false alarm 0.053; correct reject 0.078 |

### Format Validation
- `/app/train_decoder.py --verify-only` reported: “Data format is valid, no errors or warnings.”
- All required keys, session/trial nesting, neuron/time dimensions, categorical ranges, and metadata passed.
- Image occupancy (~1/3 image, ~2/3 gray) matches the 250 ms image + 500 ms gray reference cadence.
- Exactly one change bin occurs on each sample go trial; catch trials remain zero.

### Processing Plots Review
- `processing_775614751.png` and `processing_788490510.png` were created (117/126 KB).
- Neural event heatmaps have sparse nonnegative detected events without clipping artifacts.
- Image labels alternate with gray at expected cadence; change impulses coincide with image transitions.
- Continuous running/pupil traces and their quintile steps are temporally coincident with no visible shift.
- Static outcome is constant across each plotted trial.

### Run Time Estimates
| Speed-up Implemented | Time Savings |
|----------------------|--------------|
| Direct h5py selective reads | Avoids loading large stimulus templates/projections |
| Searchsorted binning | Avoids frame×bin masks |
| One-file streaming | Bounds conversion memory |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Core session processing | 0.34–0.61 s in sample | ~2 minutes for 199 sessions plus I/O |
| Whole sample including vocabulary scan/plots/write | 5.6 s / 2 sessions | conservatively <10 minutes full; below 15-minute optimization threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None.
- Warnings: sklearn reported `y_pred contains classes not in y_true` for one metric. This arises because the two-session validation split lacks at least one rare class while the model can predict it; it is not a format error and all declared classes occur in the full sample aggregate.

### Training Progress
- Loss decreased monotonically from approximately 1.636 to 1.550 over 200 epochs; test loss was 1.592.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance |
|--------|-----------------------|-------------------------|----------------|
| image identity | 0.2693 | 0.2431 | 0.0588 |
| image change | 0.7219 | 0.6567 | 0.5000 |
| running speed bin | 0.2147 | 0.2035 | 0.2000 |
| pupil diameter bin | 0.2459 | 0.2179 | 0.2000 |
| trial outcome | 0.3137 | 0.2857 | 0.2500 |

All sample validation accuracies are above chance. Image identity is >4× chance and change is clearly decodable, supporting temporal alignment. Running/pupil/outcome are modestly above chance on only two sessions and will be reassessed on full data.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 2,868,423,136 bytes (2.671 GiB)
- `conversion_full_out.txt`: created; conversion completed in 149.3 s
- `verification_full_out.txt`: created; “Data format is valid, no errors or warnings” and “Data verification complete”

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Subjects | supplied subset not stated | NWB subject ID | 38 retained | 38 | Yes |
| Sessions | plane-level neural analysis | one ophys experiment object | 199 active experiments with eye | 199 | Yes |
| Total neurons | released post-QC ROIs | event/cell table | 29,168 retained experiment ROIs | 29,168 | Yes |
| Neurons/session | variable by post-QC plane | event array cells | 4–666, mean 146.57 | 4–666, mean 146.57 | Yes |
| Trials (total) | include go/catch only | explicit trial flags | 51,075 file-level retained rows | 51,075 | Yes |
| Trials/session | variable | trial table | 39–409, mean 256.66 | 39–409, mean 256.66 | Yes |
| Go fraction | designed 87.5% | go flag | 44,672 / 51,075 = 87.46% | 44,672 one-bin changes | Yes |
| Outcomes (trial counts) | hit/miss/FA/CR semantics | outcome flags | 15,682 / 28,990 / 920 / 5,483 | same underlying trial labels; verifier reports time-weighted broadcast fractions | Yes |
| Frame rates | 31 Hz or 11 Hz | ophys timestamps | 30.93–30.95 or 10.73 Hz | resampled to common 100 ms bins | Yes |
| Image occupancy | 250 ms image + 500 ms gray predicts ~1/3 image | presentation timestamps | cadence present | gray 0.6653, images 0.3347 | Yes |
| Running bins | five percentiles requested | timestamped speed | continuous cm/s | each category ~0.200 | Yes |
| Pupil bins | five percentiles requested | eye ellipse + blink | processed eye data | each category ~0.200 | Yes |
| Brain regions | V1 and LM | VISp and VISl | VISp/VISl | 29,006 VISp / 162 VISl neurons | Yes |

### Full Output Statistics
- Trial lengths: 70–125 bins (mean 84.64, median 80) at exactly 100 ms/bin.
- Image-change distribution: 0.989667 no-change, 0.010333 change.
- Time-weighted outcome distribution: hit 0.3029, miss 0.5714, false alarm 0.0172, correct reject 0.1085.
- Spot aggregate checks against independently loaded raw NWBs matched sessions, neurons, trials, outcomes, frame-rate modes, and regions exactly.
- No trials were lost after the planned curation: every raw retained trial in the 199 included experiments appears in converted metadata/counts.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: searched `verification_full_out.txt` for error, warning, invalid, and failed messages. None were present. The verifier explicitly reported valid format and successful completion.
2. **Independent raw-data sanity checks**: `/app/cache/independent_sanity.py` loads raw NWBs directly and does not import conversion code. For experiments 775614751, 940352367, and 1086048031 it used `np.allclose()` to compare:
   - neural 100 ms event-bin means/interpolation;
   - zero-dimensional input shapes;
   - image identity from raw template indices/onsets;
   - one-bin image-change events from raw change times;
   - running interpolation plus stored quintile edges;
   - blink-filtered pupil equivalent diameter interpolation plus quintile edges;
   - raw outcome flags and broadcast converted outcomes;
   - trial bin counts from raw start/stop times.
   Every comparison returned `True`. A global pass over all 51,075 converted trials confirmed finite arrays, matching time dimensions, valid ranges, static outcomes, and >=2 trials/session.
3. **Reference code comparison**:

| Processing step | Conversion logic | Reference SDK / methods comparison | Result |
|-----------------|------------------|------------------------------------|--------|
| Data loading | direct h5py released NWB paths | SDK `from_nwb` reads the same NWB groups | Equivalent; avoids unrelated large objects |
| Neuron filtering | keep all released event-array ROIs | `CellSpecimens.from_nwb`; whitepaper release QC already removes invalid/duplicate/edge ROIs | Match; no unjustified double filtering |
| Trial filtering | active only; `(go|catch)&~aborted&~auto_rewarded` | SDK trial flags and task requirement; paper excludes passive | Match |
| Temporal alignment | exact event, stimulus, running, and eye timestamps on session clock | SDK exposes each timestamp stream; task mandates ophys alignment | Match, with explicit common-grid resampling required by target |
| Binning | 100 ms means of detected event magnitudes | paper uses detected events and first 400 ms decoder window; native rates are 11/31 Hz | Reasoned target-specific difference to obtain common bin size |
| Input construction | `(0,T)` empty array | task says no decoder inputs | Exact |
| Output construction | presentation identity, one-bin change, speed/pupil quintiles, outcome broadcast | same raw SDK variables; discretization/broadcast required by target/trainer | Exact or task-required transform |

4. **Key statistics comparison**: independent raw scan exactly matched 199 sessions, 38 mice, 29,168 neurons, 51,075 trials, and trial outcomes. Go fraction 87.46% matches the whitepaper’s 87.5% design. Image occupancy 33.47% matches the 250/750 ms duty cycle. Quintiles are balanced to numerical precision. Frame-rate modes match 11/31 Hz. Brain regions match VISp/VISl.
5. **Edge cases / off-by-one review**:
   - Trial bins use `floor((stop-start)/0.1)` and centers from start+50 ms, guaranteeing every center is before stop.
   - Half-open neural bins `[edge_j, edge_{j+1})` avoid double-counting frames.
   - Change uses first bin at/after onset; direct raw comparisons passed.
   - Stimulus visibility uses `[onset,onset+0.25)`; inter-flash and omission periods map to gray.
   - Empty 11 Hz event bins use local linear interpolation; nonempty bins use means.
   - Blinks/nonfinite pupil samples are excluded before interpolation; three sessions with no eye stream are excluded rather than fabricated.
   - Variable trial lengths and minimum/maximum neuron counts passed the global check.
   - Multi-plane duplicate behavior is intentional because each plane is a separate simultaneously recorded neuron population.

### Issues Found and Resolved
- **Exploration dF/F axis label**: initially reversed frame/cell labels; corrected immediately and verified against the cell table.
- **Exploration frame-rate path**: a broad timestamp glob selected an unrelated series and suggested 1 Hz; corrected to the exact ophys path, yielding 10.73/30.94 Hz as expected. Conversion always used the exact event timestamp path.
- **Passive pseudo-outcomes**: discovered passive tables were nearly all misses/correct rejects; excluded passive files in agreement with the paper.
- **Missing eye streams**: three active single-plane sessions had no eye data; excluded because pupil is mandatory and no sister plane could supply it.
- **No conversion mismatches remained after the independent re-check.**

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Device: CUDA.
- Split: 40,786 training trials; 10,289 validation/test trials.
- Loss decreasing: Yes, monotonically from 1.6342 to 1.5665 over 200 epochs.
- Test loss: 1.5861.
- Script finished successfully and generated sample/prediction plots.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance | Notes |
|--------|-----------------------|-------------------------|----------------|-------|
| image identity | 0.2331 | 0.2303 | 0.0588 | 3.92× chance |
| image change | 0.6501 | 0.6354 | 0.5000 | 1.27× chance |
| running speed bin | 0.2326 | 0.2294 | 0.2000 | 1.15× chance |
| pupil diameter bin | 0.2255 | 0.2210 | 0.2000 | 1.11× chance |
| trial outcome | 0.3062 | 0.2740 | 0.2500 | 1.10× chance |

All outputs exceed chance. Image identity is strongly decodable; image change is substantially above chance despite being a one-bin sparse event. Continuous behavior quintiles and outcome are modestly above chance. Train/validation values are close for every output.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Ratio to Chance | Paper Expectation |
|----------|---------------------|--------|-----------------|-------------------|
| Image identity | 0.2303 | 0.0588 | 3.92× | Not decoded in the paper; strong above-chance result expected from visual cortex |
| Image change | 0.6354 | 0.5000 | 1.27× | Paper reports above-chance change-vs-repeat decoding graphically, no scalar value |
| Running speed quintile | 0.2294 | 0.2000 | 1.15× | Not reported |
| Pupil diameter quintile | 0.2210 | 0.2000 | 1.11× | Not reported |
| Trial outcome | 0.2740 | 0.2500 | 1.10× | Related paper hit-vs-miss decoder reported graphically, not the four-outcome task here |

### Check 1: Accuracy vs Chance
- Every validation balanced accuracy is above chance.
- Image identity is far above the 1.5× threshold.
- Change, running, pupil, and outcome are below 1.5× chance and were therefore investigated fully rather than assumed correct from accuracy alone.
- Image change is a deliberately sparse one-bin event (1.03% of time points), yet reaches 0.635 balanced accuracy, supporting correct temporal alignment.
- Five equal-percentile behavior bins are intrinsically harder than coarse locomotion/state labels; nevertheless both behavior outputs exceed chance.

### Check 2: Accuracy Comparison to Papers
- The paper’s Figure 6 reports cross-validated random-forest **percent correct** for change-vs-repeat and hit-vs-miss graphically by cell class, strategy, and neuron count. No single numerical accuracy is stated in text.
- The paper uses only the first 400 ms after each image presentation, balanced presentation subsets, and a random forest. The required decoder uses full variable-length trials, one-bin change labels, five simultaneous outputs, and a neural network. Therefore a precise scalar equality cannot be computed.
- The textual conclusions are consistent: image-change information is decodable and false-alarm/choice-related signals are weaker. The achieved change accuracy is above chance, while four-way outcome is modestly above chance.
- The paper’s AUC 0.83 is for a behavioral strategy model across sessions, not neural decoding, and was not used as a neural accuracy benchmark.

### Check 3: Train vs Validation Gap
| Output | Train / Validation Ratio | Absolute Gap | Overfitting? |
|--------|--------------------------|--------------|--------------|
| Image identity | 1.012 | 0.0028 | No |
| Image change | 1.023 | 0.0147 | No |
| Running speed | 1.014 | 0.0032 | No |
| Pupil diameter | 1.020 | 0.0045 | No |
| Trial outcome | 1.118 | 0.0322 | No |

No ratio approaches the 1.5× criterion and no data-leakage/overfitting signature is present.

### Required Low-Accuracy Debugging
1. **Raw output checks**: three experiments/trials (775614751, 940352367, 1086048031) were independently loaded from NWB. Image, change, running-bin, pupil-bin, and outcome arrays all matched conversion with `np.allclose()`.
2. **Temporal alignment plots**: processing plots show event activity and output streams on the same 100 ms trial-start grid. Flash/gray cadence and change timing are synchronized.
3. **Variation**: running/pupil categories are each ~20%; all 17 image categories and four outcomes occur globally. Change is sparse by specification, not a conversion imbalance.
4. **Neural filtering**: detected calcium events and released post-QC ROIs match the paper/whitepaper; no unfiltered fluorescence or invalid ROIs were introduced.
5. **Reference processing**: active-session curation, trial flags, event representation, exact timestamp streams, and blink filtering were rechecked against SDK and methods.

### Issues Found and Resolved
- No conversion bug was found during accuracy review.
- Modest behavior/outcome accuracy is supported by exact raw comparisons, low train/validation gaps, and the difficulty of five-bin/four-class full-trial decoding. Changing labels or filtering solely to inflate accuracy would depart from the requested outputs and reference processing.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] `cache/README_CACHE.md` documents investigation artifacts
- [x] Conversion script, datasets, validation logs, training logs, and diagnostic plots retained at documented locations
- [x] CONVERSION_NOTES.md reviewed for decisions, checks, statistics, and decoder results
