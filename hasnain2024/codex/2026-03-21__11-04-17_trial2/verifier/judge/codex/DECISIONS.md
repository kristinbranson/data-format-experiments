# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not scan the whole raw dataset dynamically. It hard-codes a 12-session two-context ALM subset in `CONTEXT_SESSION_SPECS`, then loads one session `.mat` plus its paired motion-energy `.mat` per entry. Neural/session/trial data are loaded session-by-session inside `convert_session()`. In `CONVERSION_NOTES.md`, the agent justifies this as following the Figure 8 two-context reference loader path rather than the full raw archive.

ii. ```python
CONTEXT_SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 1),
    SessionSpec("JEB7", "2021-04-29", 0),
    ...
]

obj = mat73.loadmat(spec.data_path)["obj"]
me = load_motion_energy(spec)
```

iii. Justification in notes/trajectory: it decided the decoder should use the paper's two-context ALM subset because `behavioral_context` only varies there, and explicitly cites `Figure8a_thru_c.m` and the `loadJEB*/loadEKH*/loadJGR*` loaders as the reference subset.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from `SessionSpec.animal`, stored per session as `subject`, then unique animal IDs are sorted into `subjects`; `subject_idx` maps each converted session to that subject list.

ii. ```python
result = {
    "session_id": spec.session_id,
    "subject": spec.animal,
    ...
}

subjects = sorted({sess["subject"] for sess in converted_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in converted_sessions], dtype=np.int64),
```

iii. The notes say this follows the figure-loader animal IDs directly. The agent also documents that this yields 7 animal IDs although the paper text says 6 mice, and says it resolved that discrepancy in favor of the code/raw data.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` corresponds to one converted session. `convert_session()` returns per-session `neural`, `input`, `output`, and metadata, and `build_dataset()` preserves that session list order.

ii. ```python
for sess_idx, spec in enumerate(session_specs):
    converted_sessions.append(convert_session(spec, make_plot=make_plot))

dataset = {
    "neural": [sess["neural"] for sess in converted_sessions],
    "input": [sess["input"] for sess in converted_sessions],
    "output": [sess["output"] for sess in converted_sessions],
    ...
}
```

iii. The notes explicitly say it is mirroring the 12-session Figure 8 loader set and treating each selected file/probe pair as one session.

## 1-d. How are the data split into trials?

i. Trials are defined by raw trial indices in `obj["bp"]`. The code first finds session-level valid trial indices, then keeps a subset with a valid first post-alignment lick, then stores one neural matrix / input array / output array per kept trial.

ii. ```python
valid_trials = select_valid_trials(obj)
...
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
    if lick_dir is None:
        continue
    kept_trials.append(int(trial_idx))
...
for local_trial_idx, trial_idx in enumerate(kept_trials):
    neural_trial = neural_by_trial[:, local_trial_idx, :]
    ...
    session_neural.append(neural_trial.astype(np.float32))
```

iii. The notes describe this as keeping decoder-ready trials only: hit/miss, non-early, non-ignore, non-stim, with a defined lick direction.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `early`, `no`, and `stim.enable`, then keeping only `hit` or `miss` trials. A second pass drops trials with no first lick after the alignment event.

ii. ```python
def select_valid_trials(obj: dict) -> np.ndarray:
    early = ensure_1d_numeric(bp["early"]) != 0
    no = ensure_1d_numeric(bp["no"]) != 0
    hit = ensure_1d_numeric(bp["hit"]) != 0
    miss = ensure_1d_numeric(bp["miss"]) != 0
    stim_enable = ensure_1d_numeric(bp["stim"]["enable"]) != 0
    valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
```

iii. In `CONVERSION_NOTES.md`, the agent says this was chosen to match paper/code omission of early and ignore trials, avoid stimulation trials, and keep outcome well-defined for decoding.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-unit spike times and trial assignments on the selected probe, aligned using `bp.ev.goCue`.

ii. ```python
clu = obj["clu"][spec.probe_index]
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
...
trialdat = bin_unit_spikes(
    clu["trialtm"][unit_idx],
    np.asarray(clu["trial"][unit_idx], dtype=np.int64),
    align_times,
    kept_trials,
)
```

iii. The notes tie this directly to the reference `alignSpikes` + `getSeq` pipeline and explicitly note these are ephys spike trains, not imaging data.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to `goCue`, histogrammed into uniform bins from `TMIN=-2.5` to `TMAX=2.5` with `DT=0.005`, converted to firing rates by dividing by `DT`, then smoothed with a causal Gaussian-like kernel (`SMOOTH_N=15`, `BCTYPE="reflect"`). The final trial representation is neurons by time.

ii. ```python
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
rates = aligned_counts / DT
rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
...
neural_trial = neural_by_trial[:, local_trial_idx, :]
```

iii. The notes say this is meant to match `getSeq.m` and use smoothed single-trial firing rates because the paper's decoding scripts use `obj.trialdat`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code keeps only units whose `quality` label is not in `{"garbage","gabrga","noisy","real?"}` and whose mean firing rate across the full session alignment window exceeds 1 Hz.

ii. ```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
...
quality_keep = np.asarray([keep_quality(q) for q in unit_quality], dtype=bool)
...
mean_fr = mean_firing_rate_window(...)
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. The notes justify this from `findClusters(..., {'all'})` and the paper's statement that analyses used units with firing rates exceeding 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every neural spike train is aligned to `bp.ev.goCue`. For WC trials, the agent assumes the stored `goCue` field is a goCue-equivalent water-presentation event and uses it unchanged.

ii. ```python
ALIGN_EVENT = "goCue"
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
```

iii. The notes and trajectory explicitly call this out as a major decision: the user required universal go-cue alignment, so the agent inspected WC trials and decided the recorded `goCue` could be treated as the corresponding WC action-triggering event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 5 ms bins (`DT = 0.005`) over a 5 s window. No later temporal rebinning is applied.

ii. ```python
DT = 0.005
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
...
"time_bin_size": 5.0,
```

iii. The notes justify 5 ms as paper-consistent even though the chosen Figure 8 reference script uses 10 ms bins; the agent says it resolved that in favor of a 5 ms base binning.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a separate continuous raw signal. It is a synthetic aligned time axis defined relative to the chosen alignment event (`bp.ev.goCue`) and the fixed conversion constants.

ii. ```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
...
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The notes say this is a target-format adaptation: the reference code trains decoders per aligned time bin, while the requested decoder format explicitly asked for time from go cue as input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No signal processing is applied beyond constructing a fixed per-trial vector of bin centers from `-2.5` to `2.5` s.

ii. ```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The agent's notes explicitly describe this as a deliberate formatting adaptation rather than a transformation copied from the reference code.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the exact same `TIME_AXIS` as neural binning, so each input timepoint corresponds one-for-one to each neural bin center.

ii. ```python
time_input = TIME_AXIS[None, :].astype(np.float32)
...
neural_trial = neural_by_trial[:, local_trial_idx, :]
```

iii. The notes say the input axis was made identical to the neural alignment grid to satisfy the decoder format.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from the per-trial lick event lists `bp.ev.lickL` and `bp.ev.lickR`, not from instructed side (`bp.R` / `bp.L`).

ii. ```python
lick_l = event_list_to_array(obj["bp"]["ev"]["lickL"][trial_idx])
lick_r = event_list_to_array(obj["bp"]["ev"]["lickR"][trial_idx])
```

iii. The notes justify this because the user asked for actual lick direction, and for error trials that differs from the instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code finds the first lick after the alignment time and labels left as `0`, right as `1`. Trials with no post-alignment lick are dropped. The label is then repeated across all time bins.

ii. ```python
lick_l = lick_l[lick_l >= go_time]
lick_r = lick_r[lick_r >= go_time]
...
return 0 if first_l < first_r else 1
...
np.full(TIME_AXIS.size, lick_dir, dtype=np.int64),
```

iii. The notes and trajectory explicitly state this was a conscious choice over using instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `bp.autowater`.

ii. ```python
autowater = ensure_1d_numeric(bp["autowater"])
...
0 if autowater[trial_idx] != 0 else 1
```

iii. The notes say this matches the paper/code convention `autowater` = WC and `~autowater` = DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code binarizes `autowater` into WC=`0`, DR=`1`, then repeats the label across the whole trial.

ii. ```python
trial_labels.append(
    (
        lick_dir,
        0 if autowater[trial_idx] != 0 else 1,
        1 if hit[trial_idx] != 0 else 0,
    )
)
```

iii. The notes say this is the exact context variable used throughout the reference code.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived from `bp.hit`, with `bp.miss` defining the complementary incorrect trials after trial filtering; ignore/no-response trials are already excluded.

ii. ```python
hit = ensure_1d_numeric(bp["hit"])
...
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
```

iii. The notes cite `getOutcome.m`, which returns `bp.hit` and sets ignore trials to `NaN`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as correct=`1` if `hit!=0`, otherwise incorrect=`0` for the kept miss trials, then repeated across all time bins.

ii. ```python
1 if hit[trial_idx] != 0 else 0,
...
np.full(TIME_AXIS.size, outcome_label, dtype=np.int64),
```

iii. The notes describe this as a direct binary decoder target derived from the paper/code outcome logic.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It is derived from the side-view DLC tongue trajectory stored in `obj["traj"][0]`, feature name `"tongue"`, using x/y positions only for that feature.

ii. ```python
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
```

iii. The notes say the agent deliberately collapsed the requested tongue variable to one scalar speed stream, choosing a single tongue feature rather than the full multi-feature tongue representation in the reference kinematics code.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code interpolates tongue x/y positions onto the neural time axis, computes x/y velocity with `np.gradient`, converts that to scalar speed magnitude, and keeps track of visibility from finite x/y positions.

ii. ```python
tongue_visible = np.isfinite(tongue_x) & np.isfinite(tongue_y)
tongue_vx, tongue_vy = compute_velocity(tongue_x, tongue_y, "tongue")
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. The notes justify scalar speed magnitude as the simplest single decoder output corresponding to “tongue velocity”.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The threshold is the 50th percentile of tongue speed over visible timepoints from kept trials in the session. Invisible timepoints are forced into the low bin.

ii. ```python
tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
...
tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. The trajectory shows the agent changed this after a verifier issue, specifically to use only visible tongue timepoints for the median so the split would not collapse.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are interpolated onto `TIME_AXIS` after subtracting video offset and trial alignment time, so the binned tongue labels share the neural bin centers.

ii. ```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

iii. The notes cite the reference `getKinematicsFromVideo` / `findPosition` / `findVelocity` path as the alignment model.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It is derived from two bottom-view DLC paw features, `"top_paw"` and `"bottom_paw"`, in `obj["traj"][1]`.

ii. ```python
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
```

iii. The notes say this was a deliberate reduction of the richer paw tracking into one requested scalar paw-movement output.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw feature is aligned to the common time axis, x/y velocity is computed, speed magnitude is taken, then the two paw speeds are averaged where finite to form one scalar trace.

ii. ```python
paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
paw_speeds.append(np.sqrt(paw_vx**2 + paw_vy**2))
...
np.divide(np.nansum(paw_stack, axis=0), paw_counts, out=paw_speed, where=paw_counts > 0)
```

iii. The notes justify scalar speed magnitude plus averaging as the most defensible collapse for the requested single paw-velocity output.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The threshold is the session median of `paw_speed` across all kept trials/timepoints, then each timepoint is binarized as below vs at/above that threshold.

ii. ```python
paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
...
(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64),
```

iii. The notes say this follows the user's required median split for continuous movement variables.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. As with tongue, paw positions are interpolated onto the same aligned time grid after subtracting video offset and `goCue` alignment time.

ii. ```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
...
paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
```

iii. The notes say all video-derived outputs were aligned to the same neural-centered time base.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is derived from the companion `motionEnergy_<animal>_<date>.mat` files, specifically `me.data`.

ii. ```python
me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
me = me_mat["me"]
return {
    "data": np.atleast_1d(me.data),
    "moveThresh": float(me.moveThresh),
}
```

iii. The notes explicitly identify `loadMotionEnergy.m` as the reference function and say the raw motion-energy files are the source.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, motion energy is interpolated from video-frame timestamps onto the neural `TIME_AXIS` after subtracting video offset and alignment time, then missing edge values are filled with nearest-neighbor interpolation.

ii. ```python
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

iii. The notes say this was intended to match the alignment logic in `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The agent ignores the provided manual `me.moveThresh` for the decoder output and instead thresholds aligned motion energy at the 50th percentile within each session, per the decoder instructions.

ii. ```python
me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))
...
(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64),
```

iii. The notes explicitly frame this as a user-required discretization that differs from the paper's movement/non-movement thresholding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated onto the same `TIME_AXIS` as the neural data, using the same `goCue`-centered alignment.

ii. ```python
motion_energy = align_motion_energy(obj, me, align_times, TIME_AXIS)
```

iii. The notes say the goal was to place motion energy on the identical trial-centered neural time base.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses several pragmatic fallbacks: if frame times are missing it reconstructs them as a 400 Hz grid; non-tongue missing positions/velocities are nearest-filled; tongue missing velocities are set to zero; missing aligned motion-energy edges are nearest-filled; trials with no valid lick direction are dropped; sessions with too few valid trials or no surviving units raise errors.

ii. ```python
if frame_times is None:
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
...
x[~good] = np.interp(idx[~good], idx[good], x[good])
...
tempx(isnan) -> 0.0  # via np.where for tongue velocities
...
if lick_dir is None:
    continue
```

iii. The notes explicitly mention these as edge-case handling choices and compare them to the reference behavior of reconstructing timestamps and filling missing values.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are per-unit spike binning/smoothing and per-trial interpolation of video features and motion energy, because both are implemented with Python loops over units and/or trials.

ii. ```python
for unit_idx, use_unit in enumerate(quality_keep):
    ...
    trialdat = bin_unit_spikes(...)

for trial_idx in range(n_trials):
    ...
    xpos[:, trial_idx] = np.interp(...)
```

iii. The notes' runtime section also identifies kinematic extraction and spike conversion as the dominant work, and says `np.add.at` was one of the key speed-ups.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalarized: `my_smooth()` loops over columns; `align_feature_positions()`, `compute_velocity()`, and `align_motion_energy()` loop over trials; unit processing loops over units; paw features are processed one feature at a time.

ii. ```python
for col in range(x_filt.shape[1]):
    out[:, col] = np.convolve(...)
...
for trial_idx in range(n_trials):
    ...
for unit_idx, use_unit in enumerate(quality_keep):
    ...
```

iii. The notes mention the agent already vectorized spike binning but left these other loops in place.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly computes aligned/interpolated traces feature-by-feature and trial-by-trial, recomputes nearest-fill passes column-by-column, and recomputes per-trial static output vectors as full-length constant time series.

ii. ```python
tongue_x, tongue_y = align_feature_positions(...)
...
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(...)
...
np.full(TIME_AXIS.size, lick_dir, dtype=np.int64)
```

iii. The notes say only a few requested features were kept to limit repeated DLC processing, implying the agent recognized this duplication.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `me.moveThresh` but does not use it, computes aligned kinematic/motion-energy traces for all raw trials before subselecting kept trials, optionally creates plots not needed by downstream decoders, and expands per-trial static labels into time-varying constant arrays even though the information is trial-level.

ii. ```python
return {
    "data": np.atleast_1d(me.data),
    "moveThresh": float(me.moveThresh),
}
...
tongue_speed[:, kept_trials]
...
np.full(TIME_AXIS.size, context_label, dtype=np.int64)
```

iii. The notes acknowledge that the median-binned movement outputs are a target-format adaptation, and their runtime notes show the agent tried to reduce but not eliminate unnecessary feature processing.
