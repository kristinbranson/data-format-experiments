# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, finds locally present NWB files, removes passive experiments, and opens each retained file directly with `h5py`; it does not use the SDK cache or combine planes.

ii. `exp_table = pd.read_csv(table_path)`; `available_files = {... experiment_dir.glob("behavior_ophys_experiment_*.nwb")}`; `exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)]`; `exp_table = exp_table[~exp_table["passive"]]`; `with h5py.File(session.filepath, "r") as f:`

iii. The notes say direct NWB access was chosen because the local AllenSDK/NWB high-level loader failed, and active files were selected because passive viewing is outside task performance.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s from retained experiment rows; each experiment receives the corresponding global subject index.

ii. `subjects = sorted({session.mouse_id for session in kept_sessions})`; `subject_idx[idx - 1] = subject_to_idx[session.mouse_id]`

iii. The notes identify `mouse_id` as the animal identifier and validate subject/metadata consistency against the experiment table.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` NWB (one imaging plane) is treated as a converted session, even when several experiments share an `ophys_session_id`.

ii. `for row in exp_table.itertuples(index=False): sessions.append(SessionMeta(ophys_experiment_id=..., ophys_session_id=..., filepath=...))`; `for idx, session in enumerate(kept_sessions, start=1):`

iii. The agent justified this as matching `BehaviorOphysExperiment` granularity and providing one neuron set/region per output session.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`; each retained trial spans its raw `start_time` through `stop_time` and is represented by 30 Hz bin centers.

ii. `trials = get_trial_table(f)`; `centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))`

iii. The agent states that using the processed NWB trial table mirrors Allen trial construction and avoids ad hoc segmentation.

## 1-e. How are trials filtered based on quality controls?

i. It explicitly retains go or catch trials and excludes aborted and auto-rewarded trials. Empty/invalid time windows are skipped, sessions need at least two usable trials, passive sessions and sessions without eye tracking are excluded.

ii. `trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])]`; `if centers.size == 0: continue`; `if len(neural_trials) < 2: raise RuntimeError(...)`

iii. The notes tie go/catch filtering to Allen's contingent-trial definition and the prompt; eye tracking is required because pupil is a requested output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB event-detection `timestamps` and `data`, not dF/F.

ii. `event_group = f["processing"]["ophys"]["event_detection"]`; `events = np.asarray(event_group["data"][:], dtype=np.float32)`

iii. The agent chose event magnitudes because the paper analyzes extracted calcium events and the signals were already processed upstream.

## 2-b. How is the `neural` data processed?

i. The time-by-cell event matrix is linearly interpolated onto trial-specific 30 Hz centers and transposed to cells by time. No smoothing or normalization is added.

ii. `neural_trial = linear_resample_matrix(ophys_time, events, centers)`; `return interp.T.astype(np.float32, copy=False)`

iii. The notes say raw rather than filtered events preserve the embedded event-detection result, while vectorized interpolation gives a common grid efficiently.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit ROI-quality mask is applied in the converter. The number of cells is taken from `cell_specimen_table`, relying on cells stored in each NWB; all-zero event trials are retained.

ii. `n_cells = len(cell_table["cell_specimen_id"])`; `brain_region_idx = np.full(n_cells, region_index, dtype=np.int64)`

iii. The notes rely on Allen's upstream valid-ROI/event pipeline and report that all-zero trials were genuine source data, not conversion errors.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute ophys timestamps are interpolated at centers beginning half a 30 Hz bin after trial start and ending before trial stop; metadata calls the alignment event “trial start.”

ii. `centers = start_time + (np.arange(n_bins) + 0.5) * DT`; `idx_hi = np.searchsorted(src_time, dst_time, side="left")`; `"temporal_alignment_event": "trial start"`

iii. The agent says absolute-time alignment makes ophys time the reference shared by all streams while preserving full variable-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Every session is resampled to 30 Hz (33.333 ms), including interpolation of neural events from their native 11/31 Hz rates.

ii. `DT = 1.0 / 30.0`; `TIME_BIN_MS = DT * 1000.0`; `"time_bin_size": TIME_BIN_MS`

iii. The agent argued a common 30 Hz grid satisfies the stated common-bin requirement and resembles paper event-triggered interpolation and behavior/eye rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It uses task stimulus-presentation rows: `image_name`, `omitted`, start/stop times, trial ID, and block name, rather than trial initial/change image columns.

ii. `columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", "stimulus_block_name", ...]`

iii. The notes say presentation tables provide true per-flash state and allow restriction to change-detection blocks.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Global sorted image labels are integer-coded with `gray` first. Each trial starts gray; bins inside a non-omitted presentation receive that image, while inter-stimulus and omitted bins remain gray.

ii. `image_identity = np.full(..., image_value_to_idx["gray"])`; `image_identity[mask] = image_value_to_idx[str(row.image_name)]`; `image_values = ["gray"] + ...`

iii. The agent explicitly added gray so non-image periods occupying trial time would be represented rather than mislabeled as an image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are converted to masks on the exact same 30 Hz `centers` used for neural interpolation.

ii. `mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))`; `image_identity[mask] = ...`

iii. The agent's absolute-time/common-grid rationale is that using identical centers guarantees alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from stimulus-presentation `is_change`, start/stop times, and trial membership.

ii. `trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])]`; `if bool(row.is_change): image_change[mask] = 1`

iii. The agent favored processed presentation annotations as Allen's direct per-flash change definition.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero trace is created and bins during every presentation labeled `is_change` are set to one.

ii. `image_change = np.zeros(centers.shape[0], dtype=np.int64)`; `image_change[mask] = 1`

iii. No further processing was considered necessary because the NWB already contains the change annotation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: zero outside change presentations and one inside them; no numerical threshold is applied.

ii. `image_change = np.zeros(...)`; `if bool(row.is_change): image_change[mask] = 1`

iii. The notes describe the requested indicator as a categorical binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The same presentation masks and 30 Hz centers used alongside neural data define the change bins.

ii. `mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))`

iii. The diagnostic plots and raw spot checks were used to justify change-flash alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `processing/running/speed` timestamps and speed data.

ii. `running_group = f["processing"]["running"]["speed"]`; `speed = np.asarray(running_group["data"][:], dtype=np.float64)`

iii. This is the Allen-processed filtered running-speed stream in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to 30 Hz trial centers; global quantile edges are computed in pass 1 and applied in pass 2.

ii. `running_trial = linear_resample_vector(running_time, running_speed, centers)`; `running_edges = compute_quantile_edges(..., 5)`; `running_bin = digitize_with_edges(...)`

iii. Global bins give consistent semantics and roughly balanced classes across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins use the four internal global quantile edges; values are clipped to the observed edge range and labeled 0–4.

ii. `edges = np.quantile(values, np.linspace(0.0, 1.0, nbins + 1))`; `bins = np.searchsorted(edges[1:-1], clipped, side="right")`

iii. This directly implements the requested five equal-percentile categories.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is interpolated at exactly the same trial centers as neural events.

ii. `neural_trial = linear_resample_matrix(..., centers)`; `running_trial = linear_resample_vector(..., centers)`

iii. The agent cites hardware-synchronized absolute timestamps and a shared target grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking timestamps plus pupil ellipse `width` and `height`, defining diameter as twice the larger dimension.

ii. `width = ...["pupil_tracking"]["width"][:]`; `height = ...["height"][:]`; `diameter = 2.0 * np.maximum(width, height)`

iii. The notes say the requested term “diameter” motivated deriving a geometric measure instead of using pupil width alone.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaNs are linearly filled over eye time, diameter is interpolated to 30 Hz trial centers, and global five-quantile binning is applied. Blink flags are not actually read by the code.

ii. `diameter = fill_nan_by_time(timestamps, diameter)`; `pupil_trial = linear_resample_vector(...)`; `pupil_bin = digitize_with_edges(...)`

iii. The agent intended within-session interpolation for blink-related missingness and excluded sessions lacking eye tracking entirely.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global equal-percentile categories (0–4) are computed from all retained trial samples.

ii. `pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)`; `pupil_bin = digitize_with_edges(...)`

iii. Global quantiles were chosen for consistent, balanced decoder classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Filled pupil diameter is linearly interpolated at the same 30 Hz centers as neural events.

ii. `pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)`

iii. Shared absolute timestamps and centers are the stated alignment guarantee.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. `if bool(trial_row["hit"]): return 0` (followed by miss, false alarm, and correct reject checks).

iii. These are the SDK's mutually exclusive contingent-trial outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes map in fixed order to 0–3 and the static value is repeated over every time bin in its trial; missing labels raise an error.

ii. `outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)`; `raise ValueError("Trial has no valid outcome label")`

iii. Repetition keeps all outputs in a consistent `(n_output, T)` representation.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing EyeTracking causes a session to be skipped in pass 1; pupil NaNs are interpolated (all-NaN raises); invalid trial bounds are skipped; duplicate quantiles are separated by epsilon; interpolation outside source range uses endpoint values; malformed outcome labels error. Pass 2 otherwise has no per-session exception recovery.

ii. `except KeyError ... skip`; `values[~finite] = np.interp(...)`; `if centers.size == 0: continue`; `edges = np.maximum.accumulate(edges + ... * eps)`

iii. The notes prioritize retaining usable samples through within-session interpolation while requiring pupil availability for the requested output.

## 9-a. What are the most time-consuming steps of the code?

i. Disk I/O and loading full NWB/event matrices dominate; every retained file is opened in both passes, and neural interpolation/serialization add cost in pass 2.

ii. `with h5py.File(session.filepath, "r") as f:` appears in `collect_global_statistics` and `convert_session`; `events = np.asarray(event_group["data"][:], ...)`

iii. The notes explicitly identify full conversion as I/O-heavy and report large planes as the slowest cases.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial loops, stimulus-presentation loops, string decoding, metadata-row construction, and per-session processing remain iterative. Neural interpolation itself is vectorized across cells.

ii. `for trial_idx, trial in trials.iterrows():`; `for row in trial_presentations.itertuples(index=False):`; `for value in values:`

iii. The agent specifically documented vectorized `searchsorted`/broadcasting for event interpolation as the important speedup; irregular trial lengths make outer loops convenient.

## 9-c. What processing does the code repeat multiple times?

i. Each retained NWB is read twice. Trial filtering, trial-bin construction, running/pupil loading and interpolation are repeated in pass 1 for quantiles and pass 2 for output construction; presentations are also read in both passes.

ii. Both `collect_global_statistics` and `convert_session` call `get_trial_table`, `get_task_presentations`, `get_running_data`, `get_pupil_data`, `build_trial_bins`, and `linear_resample_vector`.

iii. The agent acknowledges the two-pass reread as the cost of computing dataset-global bin edges without holding the whole dataset in memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass-1 resampled running/pupil arrays are retained only long enough to calculate edges and then discarded; pass-1 presentation tables are used only to collect image names. Optional plots compute extra raw windows and figures that are not part of the dataset. Metadata fields such as session type/experience/project code are parsed but not emitted individually.

ii. `running_values.append(running_trial)`; `pupil_values.append(pupil_trial)`; `image_names.update(...)`; `if show_processing ... make_processing_plot(...)`

iii. These costs support global coding/QC or diagnostics, but their intermediate arrays and plots are not consumed by decoder training.
