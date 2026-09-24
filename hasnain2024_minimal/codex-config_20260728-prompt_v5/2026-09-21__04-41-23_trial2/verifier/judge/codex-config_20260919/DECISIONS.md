# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 subject/date/probe records, reads only `Ephys_Behavior/data_structure_<subject>_<date>.mat` with `h5py`, and reads the matching motion-energy file with SciPy. It does not load the other listed ephys sessions or the `RandomizedDelay_Ephys_Behavior` folder.

ii.
```python
DATA_SUBDIR = "Ephys_Behavior"
SESSION_SPECS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    # ... 11 more ...
]
sessions = [process_session(data_root, spec) for spec in SESSION_SPECS]
with h5py.File(session_path, "r") as mat:
```

iii. The trajectory says the agent believed the decoder targets called for only the two-context sessions and that the hand-picked list approximately reproduced the paper's reported 522 ALM units. It explicitly chose the context-session pipeline instead of sweeping every ephys file.

## 1-b. How are the data split into subjects?

i. Subject IDs are taken directly from each hard-coded session specification. Unique subjects are accumulated in first-appearance order, and every session receives the corresponding integer index.

ii.
```python
if sess["subject"] not in subject_to_idx:
    subject_to_idx[sess["subject"]] = len(subjects)
    subjects.append(sess["subject"])
subject_idx.append(subject_to_idx[sess["subject"]])
```

iii. The trajectory gives no separate rationale beyond treating the hand-written session records as authoritative.

## 1-c. How are the data split into sessions?

i. Each record in `SESSION_SPECS` and its one `data_structure` file becomes one session in each top-level list. This produces 12 fixed-delay sessions.

ii.
```python
session_tag = f"{spec['subject']}_{spec['date']}"
session_path = data_root / DATA_SUBDIR / f"data_structure_{session_tag}.mat"
return {"subject": spec["subject"], "date": spec["date"], ...}
```

iii. The agent says the MATLAB context analyses use a specific hand-written session list and took that as the correct inclusion rule.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` determines the trial count. Behavior arrays, cluster spike-trial labels, trajectory cell entries, and motion-energy cell entries are indexed by the same zero-based Python trial index. Each retained index becomes one trial matrix.

ii.
```python
"ntrials": int(read_h5_numeric(bp["Ntrials"])),
valid_idx = np.flatnonzero(valid_trials)
for kept_trial_pos, trial_idx in enumerate(valid_idx):
    neural_trials.append(valid_neural[:, kept_trial_pos, :])
```

iii. The trajectory indicates direct HDF5 inspection was used to map the MATLAB trial layout; it gives no further justification.

## 1-e. How are trials filtered based on quality controls?

i. Trials flagged `early` or `stim.enable` are removed. Hit, miss, and ignore trials are retained. The code does not remove behavioral trials occurring after electrophysiology recording ended.

ii.
```python
valid_trials = ~beh["early"] & ~beh["stim_enable"]
valid_idx = np.flatnonzero(valid_trials)
```

iii. The agent states that this mirrors the paper's pipeline and deliberately retains correct, incorrect, and ignore outcomes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the selected `obj.clu` probe's `trial`, `trialtm`, `quality`, and optional `site`/`channel` fields, plus `obj.bp.ev.goCue` for alignment.

ii.
```python
trial_refs = np.asarray(clu["trial"][()]).squeeze()
trialtm_refs = np.asarray(clu["trialtm"][()]).squeeze()
aligned_times = spike_trialtm - go_cue[spike_trials]
```

iii. The agent traced the repository's spike-alignment path and described it as the paper's go-cue-aligned ALM pipeline.

## 2-b. How is the `neural` data processed?

i. Per unit and trial, aligned spikes are histogrammed into 10 ms bins, divided by 0.01 s to obtain Hz, and passed through a custom one-sided Gaussian-window convolution with a reflected prefix. No normalization or baseline subtraction is applied.

ii.
```python
counts, _ = np.histogram(spk, bins=edges)
rates = counts.astype(np.float64) / DT
trialdat[out_idx, tr, :] = my_smooth(rates, SMOOTH, BOUNDARY)
# my_smooth zeros the first half of gausswin(15) before convolution
```

iii. The agent says this is the paper's “causal Gaussian filter (window 15, reflect boundary)” and rebuilt trial-by-trial firing rates rather than using pre-exported arrays.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units labeled `garbage`, misspelled `gabrga`, `noisy`, or `real?` (case-insensitive) are removed. A second `>1 Hz` filter is computed from the unweighted mean of seven condition PSTHs. Unlike the reference, `poor` is not rejected.

ii.
```python
return label not in {"garbage", "gabrga", "noisy", "real?"}
psth_by_cond = np.stack(psth_by_cond, axis=2)
mean_frs = psth_by_cond.mean(axis=(1, 2))
use = mean_frs > 1.0
```

iii. The agent cites the paper's `>1 Hz` inclusion rule and used condition masks copied from the MATLAB workflow. The trajectory does not justify retaining `poor` units or condition-weighting the rate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One is subtracted from MATLAB's spike trial numbers, then the go-cue time for each spike's trial is subtracted from `trialtm` before binning.

ii.
```python
spike_trials = spike_trials - 1
aligned_times = spike_trialtm - go_cue[spike_trials]
```

iii. The agent explicitly traced `alignSpikes.m` and chose the required go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported resolution is 10 ms with 550 bins from −3.0 to +2.5 s. Raw spikes are newly binned at that resolution; video streams are linearly interpolated to the same bin centers.

ii.
```python
TIME_MIN = -3.0
TIME_MAX = 2.5
DT = 1.0 / 100.0
edges = np.arange(TIME_MIN, TIME_MAX + DT, DT)
```

iii. The final trajectory calls 10 ms and this window part of the paper's context-session pipeline, but it does not explain the departure from the repository's 5 ms, −2.5 to +2.5 s parameters.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a generated set of bin centers, conceptually relative to each trial's `bp.ev.goCue`; no raw value other than the chosen alignment event enters the saved axis.

ii.
```python
def get_time_axis():
    edges = np.arange(TIME_MIN, TIME_MAX + DT, DT)
    return edges[:-1] + DT / 2.0
```

iii. The agent chose a fixed axis because every stream was to be go-cue aligned.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code builds uniform 10 ms edges and adds half a bin, yielding centers from −2.995 to +2.495 s. The same row is copied to every retained trial.

ii.
```python
taxis = get_time_axis().astype(np.float32)
input_template = taxis[None, :]
input_trials.append(input_template.copy())
```

iii. No distinct justification beyond a fixed go-cue-aligned grid appears in the trajectory.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input contains the centers of exactly the edges used to histogram aligned spikes, so its columns correspond one-to-one with neural columns.

ii.
```python
edges = get_edges()
counts, _ = np.histogram(spk, bins=edges)
taxis = get_time_axis()
```

iii. The agent reports fixed 550-bin trials and verified shape consistency with the decoder.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived directly from per-trial `bp.ev.lickL` and `bp.ev.lickR` timestamps and `bp.ev.goCue`.

ii.
```python
data["lickL_refs"] = np.asarray(bp["ev"]["lickL"][()]).squeeze()
data["lickR_refs"] = np.asarray(bp["ev"]["lickR"][()]).squeeze()
```

iii. The final report says lick direction is the first post-go lick; no deeper rationale is recorded.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Licks before the go cue are discarded. The earlier of the first left and right post-go timestamps sets left (0) or right (1); if neither occurs, or the first times tie, the result is none (2). It is repeated across all time bins.

ii.
```python
post_l = lick_l[lick_l >= go_cue[trial_idx] - 1e-9]
post_r = lick_r[lick_r >= go_cue[trial_idx] - 1e-9]
if first_l < first_r: lick_dir[trial_idx] = 0
elif first_r < first_l: lick_dir[trial_idx] = 1
```

iii. The agent regarded the first post-go lick as the behavioral direction to decode.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived solely from `obj.bp.autowater`.

ii.
```python
"autowater": read_h5_numeric(bp["autowater"]).astype(bool),
```

iii. The agent identified water-cued trials through this field while inspecting which sessions contain both contexts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater=True` maps to WC (0), otherwise DR (1), and the per-trial value is repeated over time.

ii.
```python
context = np.where(beh["autowater"], 0, 1).astype(np.int64)
out[1, :] = context[trial_idx]
```

iii. The trajectory treats `autowater` as the discriminator for the two-context dataset.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses the per-trial `obj.bp.hit` and `obj.bp.miss` flags; `no` is loaded but not needed for assignment.

ii.
```python
"hit": read_h5_numeric(bp["hit"]).astype(bool),
"miss": read_h5_numeric(bp["miss"]).astype(bool),
"no": read_h5_numeric(bp["no"]).astype(bool),
```

iii. The agent says outcome comes from `hit/miss/no` and chose to retain all three outcome types.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. All trials start as ignore (2); misses become incorrect (0), and hits become correct (1). The class is repeated over time.

ii.
```python
outcome = np.full((beh["ntrials"],), 2, dtype=np.int64)
outcome[beh["miss"]] = 0
outcome[beh["hit"]] = 1
```

iii. The agent explicitly retained correct, incorrect, and ignore trials to populate the requested three classes.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses x/y trajectories for seven named tongue features across views 1 and 2, trajectory `frameTimes`, dropped-frame markers, `bp.ev.goCue`, `bp.ev.bitStart`, and SpikeGLX bitcode start/sample rate for clock correction. It does not use the trajectory likelihood channel.

ii.
```python
TONGUE_FEATURES = [(1, "tongue"), (1, "left_tongue"), ...,
                   (2, "bottomleft_tongue")]
x = ts[:, 0, feat_idx]
y = ts[:, 1, feat_idx]
```

iii. The final report says the outputs use the MATLAB video alignment and feature-velocity logic. The trajectory does not justify expanding from the principal tongue features to seven features.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Each feature's x/y is linearly interpolated to the 10 ms grid. Gradients are taken per sample (not divided by seconds), NaN gradients are set to zero, and Euclidean speed is computed. Visible feature speeds are averaged at each bin. There is no likelihood cutoff, smoothing, contiguous-run handling, or per-view scale normalization.

ii.
```python
x_interp = interp_with_nans(old_t, x, taxis)
xvel, yvel = compute_velocity(x_proc, y_proc, is_tongue=True)
speed = np.sqrt(xvel**2 + yvel**2)
composite_speed = speed_sum / np.maximum(visible_counts, 1)
```

iii. The agent characterized this as matching the MATLAB kinematic logic, but the trajectory supplies no justification for these detailed departures.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The threshold is the 50th percentile of all visible tongue-speed bins among retained trials in that session. Visible values below it are 0, values at or above it are 1, and invisible bins are 2.

ii.
```python
threshold = float(np.nanpercentile(keep_values[keep_visible], 50))
classes[present] = (keep_values[present] >= threshold).astype(np.int64)
```

iii. The agent explicitly cites session-wise median discretization required by the prompt.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session clock offset is computed as the mode of SpikeGLX bitcode starts divided by sampling rate minus the mode of behavioral bit starts. Each frame time is corrected by this offset and the trial's go cue, then x/y is interpolated directly to neural bin centers.

ii.
```python
vidshift = matlab_mode(bitcode_starts) / fs - matlab_mode(bit_start)
old_t = frame_times - vidshift - align_time
x_interp = interp_with_nans(old_t, x, taxis)
```

iii. The agent inspected `findVideoOffset.m` and says it used the same video alignment as the MATLAB code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses x/y trajectories for both `top_paw` and `bottom_paw` from view 2, plus the same frame-time and clock-alignment fields as tongue velocity. Likelihood is ignored.

ii.
```python
PAW_FEATURES = [(2, "top_paw"), (2, "bottom_paw")]
paw_speed, paw_visible = compute_composite_speed(mat, beh, PAW_FEATURES)
```

iii. The trajectory only says it follows the repository's feature-velocity logic; it does not justify combining both paws.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated to 10 ms and nearest-filled. Gradients are computed; the median x-position difference is subtracted from both x and y gradients, gaps in gradients are nearest-filled, and per-feature speeds are averaged wherever visible.

ii.
```python
x_proc = fill_nearest_1d(x_proc)
basederiv_x = np.nanmedian(np.diff(np.column_stack([xpos, ypos]), axis=0)[:, 0])
xvel = xvel - basederiv_x
yvel = yvel - basederiv_x
```

iii. No detailed rationale is recorded beyond the claim that it mirrors MATLAB velocity processing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The retained session's visible paw bins are split at their median: below is 0, at/above is 1, and bins where neither paw was originally finite are 2.

ii.
```python
paw_classes, paw_threshold = discretize_with_visibility(
    paw_speed, paw_visible, valid_trials, absent_code=2)
```

iii. The agent cites the requested per-session median discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Frame times receive the same session bitcode correction and trial go-cue subtraction as tongue frames; interpolated positions share the neural bin-center grid.

ii.
```python
old_t = frame_times - vidshift - align_time
y_interp = interp_with_nans(old_t, y, taxis)
```

iii. The agent states video and spikes are both aligned to `goCue` and verified their common 550-bin shape.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Values come from `motionEnergy_<subject>_<date>.mat` (`me.data`), while timestamps come from view 1 trajectory frames. Bitcode and go-cue fields provide alignment.

ii.
```python
motion_file = sio.loadmat(motion_path, squeeze_me=True, struct_as_record=False)
motion_trials = np.asarray(motion_file["me"].data, dtype=object).reshape(-1)
bundle = load_trial_video_bundle(mat, traj_groups[0], trial_idx)
```

iii. The agent traced `loadMotionEnergy.m` and used the standalone motion files associated with its selected sessions.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each trial trace is linearly interpolated from view-1 frame times to 10 ms bin centers. Interpolation gaps within a trial are then nearest-filled, though the pre-fill finite mask controls visibility. No smoothing is applied.

ii.
```python
trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
visible[trial_idx, :] = np.isfinite(trial_aligned)
aligned[trial_idx, :] = fill_nearest_1d(trial_aligned)
```

iii. The agent says it follows the MATLAB motion/video alignment; no rationale for interpolation and nearest filling is stated.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Visible bins in retained trials are split at the session median; below is 0, at/above is 1, and nonvisible/no-video bins are 2.

ii.
```python
motion_classes, motion_threshold = discretize_with_visibility(
    motion_energy, motion_visible, valid_trials, absent_code=2)
```

iii. The agent cites the prompt's per-session 50th-percentile rule.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. View-1 frame times are corrected by the session video offset and trial go cue, then motion energy is interpolated to exactly the neural bin centers.

ii.
```python
old_t = frame_times - vidshift - beh["goCue"][trial_idx]
trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
```

iii. The agent states that the same go-cue/video-clock alignment was used for all movement signals.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion files yield all class 2. A NaN dropped-frame marker causes a missing video bundle. Missing frame times are synthesized as 400 Hz times. Paw positions/velocities and motion traces are nearest-filled; tongue NaNs become zero velocity after gradient calculation. Missing site/channel values default to zero. Sessions with fewer than two retained trials raise an error.

ii.
```python
if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
    frame_times = (np.arange(ts.shape[0]) + 1.0) / 400.0
if not motion_path.exists(): return None, None
xvel[np.isnan(xvel)] = 0.0
```

iii. The trajectory mentions guarding an all-NaN velocity edge case and accommodating `channel` in place of `site`. It gives no scientific justification for synthesizing times or filling missing samples.

## 11-a. What are the most time-consuming steps of the code?

i. The apparent expensive work is nested unit/trial spike histogramming plus repeated trial/feature HDF5 reads, interpolation, gradients, and convolution. The trajectory reports that the full conversion took long because it rebuilt smoothed firing rates and aligned video features session by session.

ii.
```python
for out_idx, unit_idx in enumerate(keep_unit_indices):
    for tr in np.unique(spike_trials):
        counts, _ = np.histogram(spk, bins=edges)
for trial_idx in range(ntrials):
    for feat_out_idx, (view, feat_name) in enumerate(feature_specs):
```

iii. The agent explicitly attributed runtime to rebuilding neural and video features rather than relying on exported arrays, but provided no profiling.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike counting loops over units and then trials; smoothing loops over array columns; video processing loops over trials and features; output assembly loops over retained trials. Spike counting could use a 2-D histogram per unit, smoothing could operate along an axis, and some feature aggregation/output assembly could be batched.

ii.
```python
for col in range(arr_filt.shape[1]):
    out[:, col] = np.convolve(arr_filt[:, col], kern, mode="same")
for kept_trial_pos, trial_idx in enumerate(valid_idx):
```

iii. The trajectory does not discuss vectorization; it merely observes session-by-session rebuilding as the reason for runtime.

## 11-c. What processing does the code repeat multiple times?

i. `compute_composite_speed` is called separately for tongue and paw and each call recomputes video offset, reloads trajectory groups, scans feature names, loads trial bundles, and interpolates tracks. Motion energy computes the offset and loads view-1 bundles a third time. `get_time_axis()` is also rebuilt frequently.

ii.
```python
tongue_speed, tongue_visible = compute_composite_speed(mat, beh, TONGUE_FEATURES)
paw_speed, paw_visible = compute_composite_speed(mat, beh, PAW_FEATURES)
motion_energy, motion_visible = load_motion_energy_aligned(mat, motion_path, beh)
```

iii. The agent gives no justification for these repetitions; they follow from implementing each requested output in a separate function.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `L`, `sample`, `delay`, `no`, and lick arrays not used by most downstream operations; computes and returns electrode `sites` but discards them; builds seven condition PSTHs solely for the unit-rate filter; computes metadata offsets unrelated to conversion; and nearest-fills values whose bins are later overwritten with class 2 according to the original visibility mask.

ii.
```python
neural_all_trials, sites = load_neural_session(...)
"sample": read_h5_numeric(bp["ev"]["sample"]),
"delay": read_h5_numeric(bp["ev"]["delay"]),
"bit_start_offset_s": float(np.median(beh["bitStart"] - beh["goCue"])),
```

iii. The trajectory does not identify discarded work; it focused on successful schema verification and decoder training.
