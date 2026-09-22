# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 Figure 8 sessions, all under `Ephys_Behavior`, and directly reads selected HDF5 fields with `h5py`; motion-energy files are loaded with SciPy. It therefore does not load all available author-curated sessions, particularly those under `RandomizedDelay_Ephys_Behavior`.

ii.
```python
DATA_ROOT = Path("/app/data/Ephys_Behavior")
SESSION_SPECS = (
    SessionSpec("JEB6", "2021-04-18", (2,)),
    ...
    SessionSpec("JEB19", "2023-04-18", (1,)),
)
with h5py.File(spec.data_path, "r") as handle:
    bp = load_behavior(handle)
```

iii. The notes say this is the “exact 12-session Figure 8 order” and justify it as the cohort used for the paper's context analysis. The agent reports 12 sessions, seven subjects, and 3,116 retained trials.

## 1-b. How are the data split into subjects?

i. The subject is the `animal` field in each hard-coded `SessionSpec`. Unique animal strings are sorted, and each session receives the corresponding integer index.

ii.
```python
subjects = sorted({spec.animal for spec in selected})
subject_to_index = {subject: index for index, subject in enumerate(subjects)}
"subject_idx": np.array([subject_to_index[spec.animal] for spec in selected], dtype=np.int64),
```

iii. The notes treat the seven explicit animal IDs in the selected loaders as authoritative and document a discrepancy with the paper's report of six mice.

## 1-c. How are the data split into sessions?

i. Each hard-coded animal/date pair is one session and one element in the output session lists. Only the 12 Figure 8 sessions are included.

ii.
```python
for index, spec in enumerate(selected, start=1):
    session, diagnostics = convert_session(spec)
    converted.append(session)
...
"neural": [session["neural"] for session in converted],
```

iii. The agent says the Figure 8 loaders define the relevant cohort and probe IDs. It did not adopt the human reference's broader 44-session author-curated list.

## 1-d. How are the data split into trials?

i. Raw behavioral arrays define trials. Retained zero-based raw trial indices are generated from a Boolean mask, and spike, tracking, motion, and static-label arrays are indexed by those same IDs. Each retained trial becomes one matrix in each session list.

ii.
```python
keep_trials = ~bp["early"] & ~bp["stim"]
raw_trials = np.flatnonzero(keep_trials)
neural_trials = [neural[:, trial, :].copy() for trial in range(len(raw_trials))]
```

iii. The notes report independent checks that raw behavioral fields and retained indices reproduce the converted trial lists exactly.

## 1-e. How are trials filtered based on quality controls?

i. Trials with early licking or enabled photostimulation are removed. Hit, miss, and ignore trials are retained. Unlike the human solution, the agent does not remove late behavioral trials after electrophysiology recording ended.

ii.
```python
keep_trials = ~bp["early"] & ~bp["stim"]
raw_trials = np.flatnonzero(keep_trials)
```

iii. The paper excludes early/stimulation trials, while ignore trials are retained because the requested outputs include an ignore/no-lick class. The notes explicitly claim “no trial lost beyond this rule.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected probes in `obj/clu`: cluster `quality`, per-spike `trial`, and `trialtm`, plus behavioral `obj/bp/ev/goCue` for alignment.

ii.
```python
quality_refs = matlab_cell_refs(group["quality"])
trial_refs = matlab_cell_refs(group["trial"])
trialtm_refs = matlab_cell_refs(group["trialtm"])
aligned = cluster["trial_times"][use] - bp["goCue"][spike_trials]
```

iii. The notes state these fields reproduce the released alignment and selected-probe workflow.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 5 ms bins, divided by 0.005 to obtain Hz, and passed through a 15-sample causalized Gaussian filter. There is no z-scoring or baseline correction.

ii.
```python
np.add.at(counts[unit], (mapped_trials[in_window], bins[in_window]), 1)
return reference_smooth(counts / DT)
...
kernel[: n // 2] = 0
kernel /= kernel.sum()
```

iii. The agent argues this exactly recreates the released `mySmooth`/`gausswin(15)` implementation and says it verified the filter against explicit convolution. This differs from the human conversion's symmetric 14 ms Gaussian smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labeled `garbage`, `gabrga`, `noisy`, or `real?` are excluded; `poor` units are retained. A strict greater-than-1-Hz threshold is then calculated from the mean of seven condition PSTHs spanning −3 to +2.5 seconds, rather than from the final retained-trial window.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
...
mean_rates[unit] = np.nanmean(np.stack(condition_psths))
keep = mean_rates > LOW_FR_HZ
```

iii. The notes justify this as the Figure 8 context-workflow filter, reporting 521 retained units and 214 well-isolated units. The human solution additionally excludes `poor` and applies the rate cutoff to the final window.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time, then binned on the common −2.5 to +2.5 second grid.

ii.
```python
aligned = cluster["trial_times"][use] - bp["goCue"][spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The notes identify this as the released `alignSpikes` sign convention and report independent raw-spike agreement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 1,000 non-overlapping 5 ms bins from −2.5 to +2.5 seconds. Raw spikes are binned directly; video streams are interpolated onto the bin centers.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.005
TIME = (np.arange(TMIN, TMAX, DT) + DT / 2).astype(np.float32)
```

iii. The agent says the grid matches the paper's 200 Hz time base.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is analytically defined from the common window and 5 ms bin centers; raw go-cue values are used to align data to that coordinate system but do not vary the resulting input vector.

ii.
```python
TIME = (np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2).astype(np.float32)
input_trials = [TIME[None, :].copy() for _ in raw_trials]
```

iii. The notes say every input vector exactly matched the analytical bin centers.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The left edge of each 5 ms interval is generated and shifted by half a bin to form centers; this same row is copied for every trial.

ii.
```python
np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
```

iii. The agent notes that zero lies between the two central bins, as expected for even-count bin centers.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input contains the centers of the exact edges used for go-cue-aligned spike binning, so its columns correspond one-to-one with neural columns.

ii.
```python
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
input_trials = [TIME[None, :].copy() for _ in raw_trials]
```

iii. The notes report exact grid and shape checks for all trials.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is inferred from `bp.R`, `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
if bp["no"][raw]:
    lick = 2
elif bp["hit"][raw]:
    lick = 1 if bp["R"][raw] else 0
elif bp["miss"][raw]:
    lick = 0 if bp["R"][raw] else 1
```

iii. The notes say actual choice must be inferred: hits use the instructed side, misses use the opposite side, and ignores have no lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Left, right, and none are encoded as 0, 1, and 2 and broadcast across all 1,000 time bins.

ii.
```python
static[kept] = (lick, context, outcome)
out[:3] = static[trial, :, None]
```

iii. The notes report exact raw-field checks on hit, miss, and no-response trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from the per-trial `bp.autowater` flag.

ii.
```python
context = 0 if bp["autowater"][raw] else 1
```

iii. The notes identify autowater trials as water-cued (WC) and all others as delayed-response (DR).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The Boolean flag is relabeled WC=0 and DR=1 and repeated across time.

ii.
```python
context = 0 if bp["autowater"][raw] else 1
out[:3] = static[trial, :, None]
```

iii. The agent reports that converted context counts match direct source-field counts.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses the mutually exclusive per-trial `bp.hit`, `bp.miss`, and `bp.no` flags.

ii.
```python
outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
...
outcome = 0 if bp["miss"][raw] else (1 if bp["hit"][raw] else 2)
```

iii. The notes say ignores are retained to meet the requested three-class decoder target.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss, hit, and no-response trials become incorrect=0, correct=1, and ignore=2, repeated over time.

ii.
```python
outcome = 0 if bp["miss"][raw] else (1 if bp["hit"][raw] else 2)
```

iii. The agent validates that the three source flags are exhaustive and reports direct raw-data agreement.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The agent uses only the side-camera `tongue` x/y trajectory and its frame times. It does not use the bottom-camera `top_tongue` trajectory used by the human solution. Go cue and bitcode fields provide synchronization.

ii.
```python
tongue[kept_trial], tongue_visible[kept_trial], tongue_x, tongue_y = feature_speed(
    handle, side, int(raw_trial), "tongue", aligned_side_time, True
)
```

iii. The notes call this the side-camera tongue coordinate used by the selected reference workflow; no justification is given for discarding the second tongue view beyond matching that chosen workflow.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-camera x/y coordinates, already NaN where tracking is invalid, are linearly interpolated to 5 ms centers. `np.gradient` is applied in samples (not seconds), and the x/y magnitude is calculated. There is no 5 ms Gaussian position smoothing, no contiguous-run differentiation, no second-camera normalization, and no two-view averaging.

ii.
```python
x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
y = interpolate_with_nans(aligned_time, trajectory[:, 1, feature_index], TIME)
x_velocity = np.gradient(x_for_velocity)
y_velocity = np.gradient(y_for_velocity)
speed = np.hypot(x_velocity, y_velocity)
```

iii. The agent says this matches its reading of the released function and independently reconstructs its own output, but its procedure materially differs from the human reference processing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One session-wide median is computed over visible finite tongue-speed samples. Visible samples below it are 0 and samples at or above it are 1; invisible samples are 2.

ii.
```python
tongue_threshold = float(np.nanmedian(tongue[tongue_visible]))
classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
```

iii. The notes cite the task's per-session 50th-percentile requirement and verify near-50/50 visible class balance.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session bitcode-derived video offset and the trial go cue are subtracted from frame times, after which positions are interpolated to the neural bin centers.

ii.
```python
aligned_side_time = side_frames - offset - bp["goCue"][raw_trial]
x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
```

iii. The notes report raw-frame spot checks and synchronized plots. The clock correction matches the reference, though interpolation differs from reference bin averaging.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity comes from bottom-camera `top_paw` x/y trajectories, frame times, and go-cue/bitcode synchronization fields.

ii.
```python
paw[kept_trial], paw_visible[kept_trial], paw_x, paw_y = feature_speed(
    handle, bottom, int(raw_trial), "top_paw", aligned_bottom_time, False
)
```

iii. The notes say `top_paw` is the reliably tracked paw used by the released analysis.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated onto 5 ms centers; gaps are nearest-filled for differentiation; gradients are baseline-corrected (including the released function's x-baseline-for-y behavior), combined as speed, and finally remasked at originally invisible centers. This differs from the human solution's likelihood-filtered, per-run Gaussian smoothing and time-based differentiation followed by bin averaging.

ii.
```python
x_for_velocity, y_for_velocity = fill_nearest(x), fill_nearest(y)
x_velocity = np.gradient(x_for_velocity)
y_velocity = np.gradient(y_for_velocity)
x_velocity = x_velocity - baseline_derivative[0]
y_velocity = y_velocity - baseline_derivative[0]
speed = np.hypot(x_velocity, y_velocity)
speed[~visible] = np.nan
```

iii. The agent justifies the baseline operation, including the apparent x/y typo, as exact reproduction of the released function.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide median over visible paw speed defines below=0 and at/above=1; invisible bins are 2.

ii.
```python
paw_threshold = float(np.nanmedian(paw[paw_visible]))
classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
```

iii. The notes cite the explicit 50th-percentile instruction and verify session-level balance.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are corrected by the session offset and trial go cue, then positions are interpolated to the neural bin centers before speed is computed.

ii.
```python
aligned_bottom_time = bottom_frames - offset - bp["goCue"][raw_trial]
x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
```

iii. The notes report exact reconstruction against source frames. The clock alignment agrees with the reference, but its resampling order does not.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the paired `motionEnergy_<animal>_<date>.mat` file, synchronized with side-camera frame times, bitcode offset, and go cue. Its loader handles one or two nested `data` wrappers, but not the human solution's fully general repeated unwrapping.

ii.
```python
motion = loadmat(path, simplify_cells=True)["me"]
data = motion["data"]
if isinstance(data, dict):
    data = data["data"]
```

iii. The notes state the standalone file is the reference source and that trial counts are checked.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Raw per-frame energy is linearly interpolated to bin centers, then all internal and edge NaNs are nearest-filled. No smoothing or differentiation is applied.

ii.
```python
interpolated = interpolate_with_nans(motion_times, trial_motion, TIME)
motion[kept_trial] = fill_nearest(interpolated)
```

iii. The agent says the spatial motion-energy calculation was already performed upstream and that nearest filling matches its selected workflow. This changes missing bins that the human solution leaves missing.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session median over finite samples defines below=0 and at/above=1; nonfinite/no-video bins are 2.

ii.
```python
valid_motion = np.isfinite(motion)
motion_threshold = float(np.nanmedian(motion[valid_motion]))
```

iii. The notes cite the requested per-session median and report nearly exact 50/50 class balance.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Normally side-camera frame times are offset- and go-cue-corrected and energy is interpolated to neural centers. If frame times are unusable or counts mismatch, the agent constructs a nominal 400 Hz time base with a 0.5-second shift. Nearest filling then extends values across missing target bins.

ii.
```python
motion_times = aligned_side_time
...
motion_times = np.arange(1, len(trial_motion) + 1) / 400
motion_times = motion_times - 0.5 - bp["goCue"][raw_trial]
```

iii. The notes say this fallback was added after finding an all-NaN-frame-time trial and attribute it to `loadMotionEnergy.m`; it recovers the trace instead of assigning class 2.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Load-time inconsistencies generally raise errors. Missing tongue samples remain class 2. Paw gaps are nearest-filled only while computing derivatives and then remasked. Motion gaps and out-of-range target bins are nearest-filled. Motion with unusable timestamps receives a synthetic 400 Hz clock. Missing features produce all-NaN outputs.

ii.
```python
if feature not in names:
    nan = np.full(N_TIME, np.nan, dtype=np.float64)
    return nan, np.zeros(N_TIME, dtype=bool), nan, nan
...
motion[kept_trial] = fill_nearest(interpolated)
```

iii. The notes emphasize preserving requested class 2 for DLC visibility, while treating the nominal-clock motion fallback as a correction required by released code. The human solution avoids fabricating values for missing frames.

## 11-a. What are the most time-consuming steps of the code?

i. Per-session HDF5 field/reference reads, variable-length per-trial DLC processing, and pickle output dominate; the agent optimized away full nested MATLAB-object materialization.

ii.
```python
with h5py.File(spec.data_path, "r") as handle:
    ...
for kept_trial, raw_trial in enumerate(raw_trials):
```

iii. The notes report about 43 seconds for the selected full conversion and say naive full-object loading was much slower and more memory intensive.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent retains loops over clusters, trials, and output construction because MATLAB references and camera traces are variable length. Within a trace it vectorizes interpolation, gradients, spike accumulation, smoothing, and discretization.

ii.
```python
for unit, cluster in enumerate(clusters):
    ...
    np.add.at(counts[unit], (...), 1)
for kept_trial, raw_trial in enumerate(raw_trials):
```

iii. The notes say variable-length DLC arrays require the trial loop, while transformations within each trace are vectorized.

## 11-c. What processing does the code repeat multiple times?

i. The code rereads side-camera trajectory data once to get timestamps and again inside `feature_speed`; it similarly reads bottom trajectory/frame times and then rereads the trajectory for paw processing. It also copies the identical input time row per trial and builds output matrices in a trial loop.

ii.
```python
side_trajectory, side_frames = trajectory_for_trial(handle, side, int(raw_trial))
del side_trajectory
...
feature_speed(handle, side, int(raw_trial), ...)
```

iii. The notes do not acknowledge this repeated read; they instead emphasize that session-wide offsets and thresholds are computed once.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `side_trajectory` is loaded and immediately deleted, and `feature_speed` reloads it. Diagnostic mean rates, unit details, sample continuous traces, and figures are retained only for metadata/validation, not decoder inputs. In normal conversion, the clearest discarded work is the duplicate trajectory load.

ii.
```python
side_trajectory, side_frames = trajectory_for_trial(handle, side, int(raw_trial))
del side_trajectory  # Loaded again feature-wise; retained here for timestamps.
```

iii. The notes present diagnostics as sanity checks and argue that direct field-level loading avoids the much larger amount of unused processing incurred by materializing the complete MATLAB object.
