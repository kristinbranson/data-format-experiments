# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 fixed-delay `Ephys_Behavior` sessions and one ALM probe per session, then opens each `data_structure_*.mat` with `h5py` and each matching motion-energy file with SciPy. It does not load the randomized-delay folder or the other 32 author-selected sessions.

ii. `SESSIONS = (("JEB6", "2021-04-18", 2), ..., ("JEB19", "2023-04-18", 1))`; `with h5py.File(data_path, "r") as handle:`; `mat = io.loadmat(path, simplify_cells=True, variable_names=["me"])`

iii. The trajectory says the session order and probes were taken from the Figure 8 and `load<animal>_ALMVideo.m` scripts. The final response characterizes the result as the “Figure 8 pipeline”; no justification is given for excluding data outside those 12 sessions despite the requirement to convert all relevant data.

## 1-b. How are the data split into subjects?

i. Subject IDs are the animal strings in the hard-coded session tuples. Unique subjects preserve first appearance, and each session indexes that list.

ii. `subjects = list(dict.fromkeys(session[0] for session in SESSIONS))`; `[subjects.index(animal) for animal, _, _ in SESSIONS]`

iii. The agent treated the animal component of the author-curated session names as the reliable subject identifier.

## 1-c. How are the data split into sessions?

i. Each hard-coded `(animal, date, probe)` tuple and corresponding MAT file becomes one session-level list element. Only 12 fixed-delay sessions are represented.

ii. `for session_index, (animal, date, probe) in enumerate(SESSIONS, start=1):`; `neural_sessions.append(neural_trials)`

iii. The trajectory repeatedly describes these as the sessions used by Figure 8 and reports validation of 12 sessions.

## 1-d. How are the data split into trials?

i. Behavioral arrays and go-cue length define trial rows; spike trial IDs are 1-based and converted to zero-based. Kept trial indices are used consistently to extract neural, input, and output arrays.

ii. `result["ntrials"] = np.asarray([len(result["go_cue"])])`; `trial_ids = np.flatnonzero(keep)`; `neural_trials.append(neural_all[trial].copy())`

iii. The agent relied on the explicit Bpod trial structure and spike trial labels rather than reconstructing boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials with early licks or photostimulation are removed; ignore trials are retained. It does not remove behavioral trials occurring after electrophysiology recording ended.

ii. `keep = ~behavior["early"] & ~behavior["stim"]`

iii. Comments and the final response say early/stimulation exclusions follow the paper, while ignores are kept because outcome explicitly requires an ignore class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected `obj/clu` probe’s cluster `trial`, `trialtm`, and `quality` fields, plus `obj/bp/ev/goCue` for alignment.

ii. `spike_trial = _array(_deref(handle, probe_group["trial"], unit))`; `spike_time = _array(...["trialtm"]...)`; `quality = _string(...["quality"]...)`

iii. The agent identified these as the repository fields needed to reproduce ALM spike-rate processing.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 10 ms bins, divided by 0.01 s to obtain Hz, and causally smoothed with a modified 15-bin Gaussian kernel and copied-prefix boundary handling. No normalization or baseline subtraction is applied.

ii. `counts = np.zeros((ntrials, len(edges) - 1))`; `np.add.at(counts, ...)`; `rates = _causal_smooth(counts / DT)`

iii. The docstring claims this matches `mySmooth.m` and the Figure 8 pipeline, including causal smoothing and the repository’s reflect behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes exact, case-sensitive quality labels `garbage`, `gabrga`, `noisy`, and `real?`, then keeps units whose equally condition-averaged mean rate exceeds 1 Hz. It omits the reference’s `poor` exclusion and lower-casing, and uses only one selected probe per session.

ii. `exclusions = {"garbage", "gabrga", "noisy", "real?"}`; `condition_means = [rates[mask].mean(axis=0) ...]`; `if mean_rate > LOW_FR_HZ:`

iii. The agent explicitly says exact case-sensitive matching reproduces `findClusters.m`, and condition-equal averaging reproduces `removeLowFRClusters.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike’s trial-relative time is shifted by that trial’s go-cue time before binning.

ii. `aligned = spike_time - go_cue[spike_trial - 1]`

iii. The agent understood `trialtm` and `goCue` to share the behavior clock, so one subtraction suffices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It uses 10 ms bins and 550 bin centers from −2.995 through 2.495 s (window metadata −3.0 to 2.5 s). Raw spikes are rebinned directly to this grid; video traces are linearly interpolated to its centers.

ii. `TMIN = -3.0`; `TMAX = 2.5`; `DT = 0.01`; `edges = np.arange(TMIN, TMAX + DT / 2, DT)`

iii. The agent chose the Figure 8 100 Hz convention and reported “550 × 10 ms bins from −3.0 to 2.5 s.”

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from conversion constants rather than a raw per-trial field; go cue defines the zero represented by the common grid.

ii. `time = (edges[:-1] + DT / 2).astype(np.float32)`

iii. The agent regarded time as the sole continuous, time-varying decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are computed from uniform edges and copied as a `(1, 550)` array for every retained trial.

ii. `input_trials.append(time[None, :].copy())`

iii. A shared deterministic time vector ensures identical input dimensions across trials.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the centers of the same edges used to bin go-cue-aligned spikes.

ii. `bin_index = np.floor((aligned - TMIN) / DT)`; `time = (edges[:-1] + DT / 2)`

iii. The agent used one common target grid for neural and all decoder variables.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, and `bp.no` (although `L` and `miss` are loaded but not directly needed by the final expression).

ii. `for name in ("R", "L", "hit", "miss", "no", ...)`; `lick[(behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])] = 1`

iii. The comment cites `getPrevChoice.m`: right choice is a right hit or left miss; no-response trials have no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Default class 0 is left, right-hit or left-miss becomes 1, and `no` becomes 2; the per-trial class is repeated through time.

ii. `lick = np.zeros(len(keep), dtype=np.int8)`; `lick[...] = 1`; `lick[behavior["no"]] = 2`; `trial_output[0] = lick[trial]`

iii. The agent justified the mapping using instructed side plus hit/miss semantics and an explicit no-choice class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived directly from `obj.bp.autowater`.

ii. `context = behavior["autowater"].astype(np.int8)`

iii. The agent interpreted autowater as WC and all other trials as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Boolean false is encoded 0/DR and true 1/WC, then repeated across all time bins.

ii. `context = behavior["autowater"].astype(np.int8)  # 0 DR, 1 WC`; `trial_output[1] = context[trial]`

iii. This follows the ordering the agent selected in `output_values` (`["DR", "WC"]`).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit` and `bp.no`, with the default representing incorrect/miss; `bp.miss` is loaded but not used in assignment.

ii. `outcome = np.zeros(len(keep), dtype=np.int8)`; `outcome[behavior["hit"]] = 1`; `outcome[behavior["no"]] = 2`

iii. The agent assumes hit, miss, and no-response are mutually exclusive and exhaustive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Classes are 0 incorrect, 1 correct, and 2 ignore, repeated across time.

ii. `outcome[behavior["hit"]] = 1`; `outcome[behavior["no"]] = 2`; `trial_output[2] = outcome[trial]`

iii. Ignore trials were deliberately retained because the requested target explicitly includes that category.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side camera (`camera=0`) `tongue` feature’s `frameTimes` and x/y columns of `ts`; feature names select the coordinate column. It also uses behavioral go cues and SGLX/behavior bit starts for clock correction. Likelihood is not read or applied.

ii. `_load_velocity(handle, behavior, time, camera=0, feature="tongue")`; `tracking[:, 0, feature_index]`; `tracking[:, 1, feature_index]`

iii. The code comment calls the side-view tongue the “primary” repository-standard feature. The agent did not justify omitting the bottom `top_tongue` view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-view x and y are linearly interpolated to 10 ms bin centers; Euclidean magnitude of per-bin `np.gradient` is computed. Tongue coordinates are not nearest-filled, smoothed, divided by elapsed time, likelihood-filtered, or combined/scale-normalized across two views.

ii. `xpos = _interp_trace(...)`; `ypos = _interp_trace(...)`; `xvel, yvel = np.gradient(xpos), np.gradient(ypos)`; `speed[trial] = np.hypot(xvel, yvel)`

iii. Metadata describes Euclidean speed and side-view tongue. The trajectory only reports that tongue visibility is brief and distributions look internally consistent.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile is computed from visible finite samples of retained trials within each session. Values below it are 0, values at/above it are 1, and unavailable bins are 2.

ii. `threshold = float(np.percentile(values[usable], 50))`; `output[usable & (values < threshold)] = 0`; `output[usable & (values >= threshold)] = 1`

iii. The agent follows the prompt’s per-session median and preserves a not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video-clock offset (medians of bit starts) and the trial go cue are subtracted from frame times, then coordinates are interpolated at neural bin centers.

ii. `return float(np.median(video_bit_start) / sampling_rate - np.median(bit_start))`; `aligned_frames = frames - offset - behavior["go_cue"][trial]`

iii. The agent calls median a robust equivalent of MATLAB mode and uses the common target grid for exact output/neural length agreement.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom camera (`camera=1`) `top_paw` frame times and x/y tracking, plus bit starts and go cue for alignment. Likelihood is not used.

ii. `_load_velocity(handle, behavior, time, camera=1, feature="top_paw")`

iii. The agent cites Figure 1’s `top_paw_yvel_view2` and chooses the reliably tracked top paw.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Coordinates are interpolated, NaNs nearest-filled, differentiated per grid index, each derivative has a median-difference drift estimate subtracted, and x/y magnitudes are combined. The original pre-fill finite mask determines visibility. No 5 ms coordinate smoothing is applied and derivatives are not divided by time.

ii. `xpos, ypos = _nearest_fill(xpos), _nearest_fill(ypos)`; `xvel -= np.nanmedian(np.diff(xpos))`; `speed[trial] = np.hypot(xvel, yvel)`

iii. Comments say filling and drift removal follow `findVelocity.m`, while independently correcting each coordinate fixes an apparent repository typo.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Visible finite samples from kept trials are split at the session median into 0/1; unavailable bins are 2.

ii. `paw_class, paw_threshold = _median_discretize(paw_speed, paw_visible, keep)`

iii. This directly implements the prompt’s threshold and not-visible class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frames are corrected by the session offset and trial go cue, then interpolated onto the common 10 ms centers.

ii. `aligned_frames = frames - offset - behavior["go_cue"][trial]`; `_interp_trace(aligned_frames, ..., target_time)`

iii. The agent uses the relevant camera’s own frames and the same target grid as spikes.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads `me.data` from the standalone session motion-energy MAT file and uses side-camera frame times, bit starts, sampling rate, and go cues for alignment.

ii. `raw = mat["me"]["data"]`; `group = _video_group(handle, 0)`

iii. The standalone files provide the precomputed per-frame motion signal used by the paper.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each trial trace is linearly interpolated onto the 10 ms target centers. No additional smoothing or spatial computation is performed.

ii. `values[trial] = _interp_trace(aligned_frames, trace, target_time)`

iii. The agent treats motion energy as already reduced to one value per frame.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Available finite samples of kept trials are split at their session median into 0/1; unavailable bins are 2 (“no video”).

ii. `motion_class, motion_threshold = _median_discretize(motion_energy, motion_available, keep)`

iii. This implements the requested per-session 50th-percentile split and explicit no-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the bitcode-derived session offset and trial go cue, then linearly interpolated onto neural bin centers.

ii. `aligned_frames = frames - offset - behavior["go_cue"][trial]`; `_interp_trace(aligned_frames, trace, target_time)`

iii. Motion energy has one sample per side-camera frame, so the agent uses camera 0 timing.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Video trials with nonfinite dropped-frame metadata are skipped; interpolation gaps/out-of-coverage samples remain NaN and become class 2. Paw NaNs are nearest-filled for velocity but visibility is recorded before filling. Lengths are truncated to the shorter frame/trace pair. The code does not handle v5 data files or post-ephys behavioral trials.

ii. `if np.size(dropped) and not np.all(np.isfinite(dropped)): continue`; `n = min(len(source_time), len(values))`; `output = np.full(values.shape, 2, dtype=np.int8)`

iii. The agent sought to keep trials while representing unavailable video explicitly. It described silencing drift estimation for wholly missing trials after validation exposed warnings.

## 11-a. What are the most time-consuming steps of the code?

i. The code itself does not profile stages. Likely costs are HDF5 dereferencing, per-unit spike accumulation/convolution, and repeated per-trial video interpolation; the observed 12-session conversion was run repeatedly during debugging.

ii. `for unit in range(...)`; `for trial in range(ntrials):`; `signal.convolve(...)`

iii. The trajectory reports completion and repeated regeneration but gives no timing analysis or explicit optimization rationale.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Unit and trial loops remain. Spike accumulation within a unit is vectorized with `np.add.at`, and convolution covers all trials for that unit. Ragged HDF5/video records make cross-trial vectorization difficult; final per-trial packaging could be reduced but is not a major cost.

ii. `for unit in range(...)`; `np.add.at(counts, ...)`; `for trial in range(ntrials):`; `for trial in trial_ids:`

iii. No explicit trajectory justification discusses vectorization; the structure reflects pragmatic handling of ragged MATLAB cells.

## 11-c. What processing does the code repeat multiple times?

i. `_video_offset` is recomputed separately for tongue, paw, and motion energy; video groups/features and trial frames are dereferenced in separate passes. The full conversion itself was rerun several times after fixes.

ii. `offset = _video_offset(handle, behavior)` appears in both `_load_velocity` and `_load_motion_energy`, with `_load_velocity` called twice.

iii. The agent prioritized separate reusable loaders and validation; it gives no justification for recomputing the session-constant offset.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `L`, `miss`, and `autowater` alongside other fields (some are only indirectly needed), computes/stores unit quality counts and thresholds only as metadata, calculates visibility/speeds/classes for all source trials before discarding early/stimulation trials, and creates many per-trial copies. Unlike the reference, direct HDF5 access avoids loading the entire MAT tree.

ii. `tongue_speed, tongue_visible = _load_velocity(...)` before `trial_ids = np.flatnonzero(keep)`; `neural_trials.append(neural_all[trial].copy())`; `unit_quality_counts = {...}`

iii. The trajectory emphasizes validation and metadata rather than efficiency; it provides no explicit reason for processing video on trials known to be excluded.
