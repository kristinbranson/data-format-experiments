# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reconstructs the paper cohort by parsing the released Figure 3 script for animal IDs and each animal's `load*_ALMVideo.m` file for dates and selected probes. It obtains 44 session specifications across the fixed and randomized folders, then loads each session with `mat73` and falls back to `scipy.io.loadmat`. Motion energy is loaded separately when available, otherwise from `obj.me`.

ii.
```python
fixed_animals = parse_loader_animals(FIXED_SCRIPT, marker="fixmeta")
randomized_animals = parse_loader_animals(RANDOMIZED_SCRIPT, marker="randmeta")
...
obj = load_mat_file(spec.data_path)["obj"]
...
def load_mat_file(path):
    try:
        return mat73.loadmat(str(path))
    except Exception:
        return scipy.io.loadmat(str(path), simplify_cells=True)
```

iii. The notes say the released loaders, rather than raw-file enumeration, define the paper-comparable cohort and probe selections; one extra raw session is commented out. Both MATLAB layouts must be supported.

## 1-b. How are the data split into subjects?

i. The subject is the animal ID parsed from each loader name. Subjects are added on first encounter and each retained session gets the corresponding integer `subject_idx`; the result has 14 subjects.

ii.
```python
subject_idx = subject_to_idx.setdefault(spec.animal, len(subject_to_idx))
converted["subject_idx"].append(subject_idx)
converted["subjects"] = [subject for subject, _ in
    sorted(subject_to_idx.items(), key=lambda kv: kv[1])]
```

iii. The notes treat loader-derived animal IDs as authoritative and report agreement with the 14-subject raw neural cohort.

## 1-c. How are the data split into sessions?

i. Each loader date/probe statement becomes one `SessionSpec` and one output session. Its data path is `data_structure_<animal>_<date>.mat` in the cohort folder. A processed session becomes one element of each top-level session list, unless it fails minimum-unit or minimum-trial checks.

ii.
```python
m_date = re.search(r"date = '([^']+)'", line)
m_probe = re.search(r"probe = (\[[^\]]+\]|\d+);", line)
...
sessions.append(SessionSpec(cohort=cohort, folder=folder,
                            animal=animal, date=current_date, probes=probes))
```

iii. This follows the paper loaders and yields the expected 25 fixed plus 19 randomized sessions.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the trial count. Behavioral arrays are flattened to that trial axis; spike `trial` IDs are converted from 1-based to 0-based; video and motion-energy structures are normalized into per-trial lists. Retained trials are emitted individually as neural, input, and output arrays.

ii.
```python
n_trials = int(gget(bp, "Ntrials", 0) or 0)
trial_ids = np.asarray(unit["trial"], dtype=np.int64).reshape(-1) - 1
...
for tr in keep_trials:
    neural_trials.append(trialdat[:, tr, :])
    input_trials.append(taxis[np.newaxis, :])
    output_trials.append(trial_output)
```

iii. The notes identify Bpod trial fields, per-spike trial IDs, and trial-indexed trajectory cells as the native trial definitions.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be hit, miss, or no-response, must not be early-lick or stimulation trials, and must have at least one spike from some selected quality-passing unit (`neural_covered`). Sessions must retain at least two trials. This last condition removes late behavior-only tails, but technically tests observed spikes rather than recording coverage.

ii.
```python
neural_covered[trial_ids] = True
keep_trials = np.flatnonzero(
    (hit | miss | no) & ~early & ~stim & neural_covered
)
if keep_trials.size < MIN_TRIALS_PER_SESSION:
    return None
```

iii. The notes justify excluding perturbation and early-lick trials from reference trial definitions, retaining ignores because the decoder requires them, and removing trials without neural coverage. They report 13,762 retained trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the loader-selected probe entries in `obj.clu`: cluster `quality`, spike `trial`, and `trialtm`, plus `bp.ev.goCue` for alignment.

ii.
```python
units = extract_selected_units(obj, spec.probes)
trial_ids = np.asarray(unit["trial"], dtype=np.int64) - 1
trial_times = np.asarray(unit["trialtm"], dtype=np.float64)
aligned_times = trial_times - go_times[trial_ids]
```

iii. The agent states that the released loaders determine probes and that raw spike times should be aligned to the requested go cue.

## 2-b. How is the `neural` data processed?

i. Aligned spikes are histogrammed in 10 ms bins, converted to Hz, and smoothed per unit/trial with a custom one-sided (causal) 15-bin Gaussian kernel and reflected prefix. No z-scoring or baseline subtraction is done.

ii.
```python
DT = 0.01
counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
rates = counts.astype(np.float64) / DT
trialdat[unit_idx, tr] = my_smooth_1d(rates, boundary_condition="reflect")
...
kernel[: window_bins // 2] = 0.0
```

iii. The notes claim this matches `mySmooth(...,15,'reflect')` and figure-level decoder scripts more closely than generic defaults, while preserving general-purpose firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes quality labels `garbage`, `gabrga`, `noisy`, and `real?`. It constructs condition PSTHs, averages across time and conditions, retains units above 1 Hz, and drops sessions with fewer than 10 retained units. It does not exclude `poor` units.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
mean_fr = np.nanmean(psth, axis=(1, 2))
keep_units = mean_fr > LOW_FR
if trialdat.shape[0] < MIN_UNITS_PER_SESSION:
    return None
```

iii. The notes cite `findClusters`, `removeLowFRClusters`, the paper's 1 Hz rule, and a paper session-inclusion rule of at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time, then histogrammed on the common −2.5 to +2.5 s grid.

ii.
```python
aligned_times = trial_times - go_times[trial_ids]
counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
```

iii. The agent identifies go-cue alignment as consistent across the decoder task, reference scripts, and raw event tables.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The saved resolution is 10 ms (500 bins over five seconds). Raw spikes are directly rebinned into those bins; video-derived streams are interpolated onto their centers.

ii.
```python
TMIN, TMAX, DT = -2.5, 2.5, 0.01
edges = np.arange(TMIN, TMAX + DT * 0.5, DT)
time_axis = edges[:-1] + DT / 2.0
```

iii. The notes explicitly choose 10 ms and claim it follows figure scripts/tutorial paths.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common time axis relative to `bp.ev.goCue`, not a separately sampled raw variable.

ii.
```python
go_times = event_array(ev, ALIGN_EVENT)
edges, taxis = session_time_axis()
```

iii. The notes say the decoder task specifies this as the only input and the axis is the same as neural time.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers from −2.495 through +2.495 s are computed at 0.01 s spacing, cast to float32, and repeated as a `(1, T)` array for every trial.

ii.
```python
time_axis = edges[:-1] + DT / 2.0
input_trials.append(taxis[np.newaxis, :].astype(np.float32, copy=False))
```

iii. This creates a consistent time-varying decoder input on the chosen neural grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact histogram edges used for go-cue-aligned spikes, so corresponding columns represent the same bins.

ii.
```python
edges, taxis = session_time_axis()
counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
```

iii. The notes describe the time vector as repeated identically and aligned to go cue.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.ev.lickL`, `bp.ev.lickR`, and `bp.ev.goCue`, selecting actual post-go-cue lick events rather than deriving direction from instructed side and outcome.

ii.
```python
lick_l = as_list(gget(ev, "lickL"))
lick_r = as_list(gget(ev, "lickR"))
post_go_choice(lick_l[tr], lick_r[tr], go_times[tr])
```

iii. The notes argue that `bp.L/R` encode instructed side, whereas event timestamps record the animal's realized response.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It finds the first left and right lick strictly after go cue, chooses the earlier one (`left=0`, `right=1`), or assigns `none=2`; the scalar is broadcast across time.

ii.
```python
if np.isinf(t_left) and np.isinf(t_right): return 2
return 0 if t_left < t_right else 1
...
np.full(taxis.size, lick_direction[tr], dtype=np.int64)
```

iii. The agent says this naturally flips miss trials relative to instructed side and better represents actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived directly from per-trial `bp.autowater`.

ii.
```python
autowater = bool_array(bp, "autowater", n_trials)
```

iii. The notes conclude that context should come directly from this field, not from the specialized 12-session figure subset.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is encoded `WC=0`; all other trials are `DR=1`; the value is broadcast over time.

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int64)
```

iii. This is described as direct trial-wise relabeling consistent with reference condition logic.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = bool_array(bp, "hit", n_trials)
miss = bool_array(bp, "miss", n_trials)
no = bool_array(bp, "no", n_trials)
```

iii. The notes identify these as the standard mutually exclusive outcome fields.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss is `incorrect=0`, hit is `correct=1`, and no-response/default is `ignore=2`; it is broadcast over time.

ii.
```python
outcome = np.full(n_trials, 2, dtype=np.int64)
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
```

iii. Ignore trials are intentionally retained because the requested decoder output includes that category.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only side-view `obj.traj` coordinates and frame times, preferring feature `tongue` with `left_tongue`/`right_tongue` fallbacks. Go cue and SpikeGLX/behavior bit starts provide alignment. Unlike the reference, it does not combine the bottom-view tongue.

ii.
```python
build_speed_trace(obj, view_index=0,
    feature_names=["tongue", "left_tongue", "right_tongue"],
    go_times=go_times, taxis=taxis)
```

iii. The notes say this reuses reference kinematic synchronization while preserving visibility; they selected actual available tongue feature names.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Coordinates are linearly interpolated directly to the 10 ms grid, a nearest-neighbor visibility mask is applied, `np.gradient` is taken with unit sample spacing, nonfinite derivative values are set to zero, and x/y derivatives are combined by Euclidean magnitude. There is no reference-style 5 ms Gaussian smoothing, derivative with real frame times, bin averaging, camera normalization, or two-view averaging.

ii.
```python
x_interp = linear_interp(frame_times, x, taxis)
vis_interp = nearest_interp(frame_times, frame_visible.astype(float), taxis) >= 0.5
x_interp[~vis_interp] = np.nan
x_vel = np.gradient(x_interp)
x_vel[~np.isfinite(x_vel)] = 0.0
speed[trial_idx] = np.sqrt(x_vel ** 2 + y_vel ** 2)
```

iii. The agent claims it follows `findPosition`/`findVelocity` and intentionally retains a pre-fill missingness mask for the requested category.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The median is computed over finite, visible samples from retained trials within each session. Visible values below it are 0, values at or above it are 1, and invisible bins are 2.

ii.
```python
threshold = float(np.nanpercentile(values, 50))
visible_and_low = visible & np.isfinite(signal) & (signal < threshold)
visible_and_high = visible & np.isfinite(signal) & ~visible_and_low
out[visible_and_low] = 0; out[visible_and_high] = 1
```

iii. This directly implements the requested per-session 50th-percentile split and separate `not visible` class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The session video offset is computed from SpikeGLX bit-code starts and behavioral `bitStart`; offset and trial go cue are subtracted from frame times, then coordinates are interpolated to neural bin centers.

ii.
```python
return matlab_mode(bitcode_start) / float(fs) - matlab_mode(bit_start)
...
frame_times = trial_frame_times(...) - vidshift - go_times[trial_idx]
x_interp = linear_interp(frame_times, x, taxis)
```

iii. The notes cite `findVideoOffset` and the reference interpolation/alignment logic.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-view `obj.traj`, preferring `top_paw` and falling back to `bottom_paw`, plus frame-time synchronization fields and go cue.

ii.
```python
build_speed_trace(obj, view_index=1,
    feature_names=["top_paw", "bottom_paw"],
    go_times=go_times, taxis=taxis)
```

iii. The agent cites published use of `top_paw_yvel_view2` as the closest precedent and adds the fallback for availability.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. After interpolation, missing positions are nearest-filled. A median frame-to-frame derivative is calculated, but the same x-component baseline scalar is subtracted from both x and y gradients. Derivatives are filled again and combined as speed. This differs substantially from the human processing.

ii.
```python
stacked = np.column_stack([x_filled, y_filled])
basederiv = np.nanmedian(np.diff(stacked, axis=0), axis=0)
baseline = basederiv[0] if np.isfinite(basederiv[0]) else 0.0
x_vel = np.gradient(x_filled) - baseline
y_vel = np.gradient(y_filled) - baseline
```

iii. The notes characterize this as reuse of reference non-tongue missing-data behavior and computation of scalar speed magnitude.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Finite visible values from retained trials are split at their session median: below is 0, at/above is 1, and not visible is 2.

ii.
```python
paw_disc, paw_thresh = discretize_session_signal(
    paw_speed, paw_visible, 2, keep_trials)
```

iii. This directly follows the requested per-session percentile categories.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are shifted by the common video offset and trial go cue, then positions are interpolated to neural bin centers.

ii.
```python
frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]
x_interp = linear_interp(frame_times, x, taxis)
```

iii. The notes say all streams share the same synchronized time base.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It prefers the standalone `motionEnergy_<animal>_<date>.mat` `me` structure and falls back to `obj.me`; side-camera frame times, synchronization fields, and go cue provide timing.

ii.
```python
if spec.motion_path.exists():
    me_struct = load_mat_file(spec.motion_path).get("me")
else:
    me_struct = gget(obj, "me", None)
```

iii. The notes cite the reference `loadMotionEnergy` path and explain that some randomized sessions lack separate data or usable video.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already-reduced per-frame trace is linearly interpolated onto neural bin centers and any remaining nonfinite values are nearest-filled. No additional spatial reduction or smoothing is applied.

ii.
```python
interp = linear_interp(frame_times, raw, taxis)
interp = fill_nearest_1d(interp)
aligned[trial_idx] = interp.astype(np.float32)
```

iii. The notes say reference alignment is reused and the task-required median split replaces the paper's manual bimodal movement threshold.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Valid values in retained trials are split at the per-session median: below is 0, at/above is 1, and missing/no-video is 2.

ii.
```python
me_disc, me_thresh = discretize_session_signal(
    motion_energy, motion_valid, 2, keep_trials)
```

iii. This follows the decoder specification rather than the paper's movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are shifted by the session video offset and trial go cue, then linearly interpolated onto the 10 ms neural centers.

ii.
```python
frame_times = trial_frame_times(side_trials[trial_idx], raw.size) \
              - vidshift - go_times[trial_idx]
interp = linear_interp(frame_times, raw, taxis)
```

iii. The notes identify shared video synchronization and the common neural time base as required reference behavior.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Helpers normalize heterogeneous MATLAB structures and absent fields. Missing video trials are normally left in missing class 2, but absent/all-NaN frame times are replaced with a synthetic 400 Hz clock; interpolation extrapolates endpoint values outside observed time; paw and motion energy are nearest-filled. Invalid `NdroppedFrames` trials are skipped. Missing motion-energy files fall back to `obj.me` or all class 2.

ii.
```python
if frame_times is None or np.isnan(frame_times).all():
    return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
...
fill_value=(values[valid][0], values[valid][-1])
...
interp = fill_nearest_1d(interp)
```

iii. The notes emphasize explicit normalization of old/new layouts and preservation of required missing categories, though the implementation also invents or fills some missing samples.

## 11-a. What are the most time-consuming steps of the code?

i. The agent identifies MATLAB loading/nested-structure normalization and per-session video alignment as expensive. Its sample took about 5–6 seconds per session and it estimated roughly four minutes for 44 sessions.

ii.
```python
obj = load_mat_file(spec.data_path)["obj"]
processed = process_session(spec, obj, ...)
```

iii. The notes say repeated Python branching through MATLAB structs and per-trial feature parsing are the main risks; full loading and processing are necessarily session-wise.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural counting loops over every unit and then every unique trial, although a 2D histogram can count all trials for a unit at once. Video speed and motion-energy routines also loop over trials; ragged frame arrays make those harder, but feature lookup and interpolation setup could be reduced. Output assembly loops over retained trials.

ii.
```python
for unit_idx, unit in enumerate(units):
    ...
    for tr in np.unique(trial_ids):
        counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
...
for trial_idx in range(n_trials):
```

iii. The notes claim preallocation and once-per-session normalization keep runtime near-linear, but do not discuss the readily vectorizable spike trial loop.

## 11-c. What processing does the code repeat multiple times?

i. `find_video_offset(obj)` is recomputed independently for tongue, paw, and motion energy. `get_trials_view` is reconstructed separately for both speed calls and motion energy, feature names are parsed within every video trial, and the time input array is recast for every retained trial.

ii.
```python
vidshift = find_video_offset(obj)  # in build_speed_trace, called twice
...
vidshift = find_video_offset(obj)  # again in build_motion_energy_trace
feat_idx = feature_index(trial_view, chosen_feature)
```

iii. The notes intended trajectory layouts to be normalized once per session, but the implementation still repeats these computations across streams/trials.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes eight condition PSTHs solely to reduce them to one mean firing rate per unit. It computes diagnostic continuous kinematic traces and thresholds, then saves only categorical values. Optional plots use those traces only when requested. It also loads full MATLAB objects, including fields never used.

ii.
```python
psth = np.zeros((trialdat.shape[0], taxis.size, len(condition_indices)))
...
mean_fr = np.nanmean(psth, axis=(1, 2))
...
tongue_speed, ... = build_speed_trace(...)
tongue_disc, ... = discretize_session_signal(...)
```

iii. The notes justify PSTHs as reference-compatible low-FR filtering and continuous signals as necessary to obtain session thresholds; nevertheless, the full PSTH tensor and much of the loaded object do not enter the saved dataset.
