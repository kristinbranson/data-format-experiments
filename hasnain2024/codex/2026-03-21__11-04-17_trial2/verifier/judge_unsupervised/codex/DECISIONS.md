# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not scan the whole raw dataset dynamically. It hard-codes a 12-session two-context ALM subset in `CONTEXT_SESSION_SPECS`, then loads one `data_structure_<animal>_<date>.mat` file and one matching `motionEnergy_<animal>_<date>.mat` file per listed session. In `main()`, it iterates through that fixed list and converts each session independently.

ii. 
```python
CONTEXT_SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 1),
    SessionSpec("JEB7", "2021-04-29", 0),
    ...
    SessionSpec("JEB19", "2023-04-18", 0),
]

obj = mat73.loadmat(spec.data_path)["obj"]
me = load_motion_energy(spec)

for sess_idx, spec in enumerate(session_specs):
    converted_sessions.append(convert_session(spec, make_plot=make_plot))
```

iii. In `CONVERSION_NOTES.md` Step 4-5, the agent says it intentionally restricted conversion to the Figure 8 two-context ALM subset because the requested decoder needs both neural activity and behavioral context (`DR` vs `WC`), which the behavior-only sessions lack and the randomized-delay sessions do not vary in the same way.

## 1-b. How are the data split into subjects?

i. Each hard-coded `SessionSpec` carries one mouse ID in `animal`. After conversion, the dataset creates `subjects` as the sorted unique set of those IDs and `subject_idx` as one index per session.

ii. 
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe_index: int

subjects = sorted({sess["subject"] for sess in converted_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in converted_sessions], dtype=np.int64),
```

iii. The notes say the subject split should follow the figure-loader metadata (`anm` in the MATLAB loaders), so the Python port preserves mouse identity at the session level rather than inferring it from file names later.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` is treated as one session, identified by `<animal>_<date>`. The output lists (`neural`, `input`, `output`, `brain_region_idx`) each store one top-level entry per `SessionSpec`, in the same hard-coded order.

ii. 
```python
@property
def session_id(self) -> str:
    return f"{self.animal}_{self.date}"

result = {
    "session_id": spec.session_id,
    "subject": spec.animal,
    "neural": session_neural,
    "input": session_input,
    "output": session_output,
    ...
}

"neural": [sess["neural"] for sess in converted_sessions],
"input": [sess["input"] for sess in converted_sessions],
"output": [sess["output"] for sess in converted_sessions],
```

iii. The justification in Step 5 says the agent mirrored the Figure 8 session-loader list, where each loader entry corresponds to one session and one selected ALM probe.

## 1-d. How are the data split into trials?

i. Within each session, the code starts from all behavioral trials in `obj["bp"]["Ntrials"]`, computes a boolean validity mask, and then creates a smaller `kept_trials` list that is used to subset neural, input, and output trial data. Trial order is preserved from the raw session.

ii. 
```python
n_trials = int(obj["bp"]["Ntrials"])

valid_trials = select_valid_trials(obj)

kept_trials = []
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
    if lick_dir is None:
        continue
    kept_trials.append(int(trial_idx))

for local_trial_idx, trial_idx in enumerate(kept_trials):
    neural_trial = neural_by_trial[:, local_trial_idx, :]
    ...
    session_neural.append(neural_trial.astype(np.float32))
```

iii. The notes describe trial splitting as "keep non-stim, non-early, non-ignore neural trials with a valid lick outcome," so trial identity is preserved until explicit curation removes unwanted trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages. First, `select_valid_trials()` keeps only `~early`, `~no`, `~stim.enable`, and `(hit | miss)` trials. Second, `convert_session()` drops any remaining trial that lacks a detectable first post-go-cue lick direction. Sessions with fewer than 2 kept trials raise an error.

ii. 
```python
def select_valid_trials(obj: dict) -> np.ndarray:
    early = ensure_1d_numeric(bp["early"]) != 0
    no = ensure_1d_numeric(bp["no"]) != 0
    hit = ensure_1d_numeric(bp["hit"]) != 0
    miss = ensure_1d_numeric(bp["miss"]) != 0
    stim_enable = ensure_1d_numeric(bp["stim"]["enable"]) != 0
    valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
    return np.flatnonzero(valid)

if lick_dir is None:
    continue

if kept_trials.size < 2:
    raise RuntimeError(...)
```

iii. Step 5 says this was chosen to match the paper/code exclusion of early, ignore, and stimulation trials while keeping outcome defined. The extra "must have a lick direction" filter is justified in the notes as necessary to populate the requested lick-direction output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the selected probe's cluster spike-time fields: `obj["clu"][probe]["trialtm"]` and `obj["clu"][probe]["trial"]`, aligned using `obj["bp"]["ev"]["goCue"]`.

ii. 
```python
clu = obj["clu"][spec.probe_index]
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])

mean_fr = mean_firing_rate_window(
    clu["trialtm"][unit_idx],
    np.asarray(clu["trial"][unit_idx], dtype=np.int64),
    align_times,
)

trialdat = bin_unit_spikes(
    clu["trialtm"][unit_idx],
    np.asarray(clu["trial"][unit_idx], dtype=np.int64),
    align_times,
    kept_trials,
)
```

iii. The notes explicitly say the intended reference representation is electrophysiology, not imaging, and that the decoder scripts operate on `obj.trialdat`, which is ultimately built from aligned spike times.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the code subtracts the per-trial go-cue time from every spike, bins spikes into 5 ms bins from `-2.5` s to `+2.5` s, divides by `DT` to convert counts to rates, and applies a causal Gaussian-like smoothing kernel with `SMOOTH_N = 15` and `BCTYPE = "reflect"`. The final per-trial matrices are stored as `(n_neurons, n_timepoints)` float32 arrays.

ii. 
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
SMOOTH_N = 15
BCTYPE = "reflect"

aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
rates = aligned_counts / DT
rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
```

iii. The notes justify this as a Python port of `alignSpikes` plus `getSeq` plus the reference `mySmooth`, while also choosing the paper/default 5 ms base bin rather than the 10 ms bin used in the Figure 8 script.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies two neural QC filters. First, it drops any unit whose `quality` string is in `{"garbage", "gabrga", "noisy", "real?"}`. Second, it computes a mean firing rate over the aligned analysis window and drops units with `mean_fr <= 1.0` Hz.

ii. 
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0

def keep_quality(quality: str) -> bool:
    ...
    return quality not in QUALITY_EXCLUDE

if mean_fr <= LOW_FR_HZ:
    continue
```

iii. Step 4-5 says this was chosen to mimic `findClusters(..., {'all'})` plus `removeLowFRClusters(..., 1)` and to match the paper's statement that analyses used units with firing rates above 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural data are aligned to `bp.ev.goCue`. For every spike, the code subtracts the go-cue time of that spike's trial before binning. This same alignment anchor is used for all sessions and both contexts.

ii. 
```python
ALIGN_EVENT = "goCue"
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])

aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
```

iii. The notes say this followed both the user instruction ("Go cue onset") and the Figure 8 script (`params.alignEvent = 'goCue'`). For WC trials, the agent believed the stored `goCue` field acts as the water-presentation-equivalent event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 5 ms bins (`DT = 0.005`) on a uniform trial grid. No additional temporal rebinning is applied after that initial binning.

ii. 
```python
DT = 0.005
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)

"time_bin_size": 5.0,
```

iii. In the notes, the agent acknowledges that the Figure 8 script uses 10 ms bins but resolves the discrepancy in favor of 5 ms because `getDefaultParams.m` and parts of the paper mention 5 ms bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw recorded variable. It is a synthetic time vector derived from the chosen global analysis window (`TMIN`, `TMAX`, `DT`) after deciding to align the session to `bp.ev.goCue`.

ii. 
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0

time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. Step 5 says the user requested exactly one decoder input, "time from go cue onset in seconds," so the agent chose the aligned common time axis itself as that input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code creates mid-bin timestamps from `-2.5` to `+2.5` seconds in 5 ms steps and repeats the same 1-by-T vector for every trial of every session. There is no trial-specific warping or event-dependent transformation beyond the common alignment convention.

ii. 
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0

for local_trial_idx, trial_idx in enumerate(kept_trials):
    ...
    time_input = TIME_AXIS[None, :].astype(np.float32)
    session_input.append(time_input)
```

iii. The notes justify this as the simplest faithful way to expose "time from go cue" to the decoder while keeping all trials on the same grid as neural activity.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is perfectly co-registered with neural data because both use the same `TIME_AXIS`. Neural spikes are histogrammed into that axis, and the input for every trial is just that same axis copied into the dataset.

ii. 
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
...
trialdat = bin_unit_spikes(...)
...
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The notes explicitly state that the decoder input should share the neural time base so each time bin corresponds to the same aligned moment across inputs and outputs.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj["bp"]["ev"]["lickL"]` and `obj["bp"]["ev"]["lickR"]`, not from the instructed trial labels such as `bp.R`/`bp.L`.

ii. 
```python
lick_l = event_list_to_array(obj["bp"]["ev"]["lickL"][trial_idx])
lick_r = event_list_to_array(obj["bp"]["ev"]["lickR"][trial_idx])
```

iii. Step 5 says this was intentional because the task asks for "lick direction," and the agent wanted actual behavioral direction, especially on error trials where actual lick side can differ from instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code keeps only licks at or after the aligned `go_time`, takes the earliest left and earliest right lick time, and labels the trial `0` if the left lick comes first and `1` otherwise. Trials with no post-go lick on either side are dropped entirely.

ii. 
```python
lick_l = lick_l[lick_l >= go_time]
lick_r = lick_r[lick_r >= go_time]
first_l = lick_l[0] if lick_l.size else math.inf
first_r = lick_r[0] if lick_r.size else math.inf
if math.isinf(first_l) and math.isinf(first_r):
    return None
return 0 if first_l < first_r else 1
```

iii. The trajectory shows the agent later sanity-checked this against raw data (`LICK_MATCH True`). The notes justify the choice as a behavior-first definition of direction rather than a cue-based one.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj["bp"]["autowater"]`.

ii. 
```python
autowater = ensure_1d_numeric(bp["autowater"])
...
0 if autowater[trial_idx] != 0 else 1
```

iii. The notes say this directly matches the reference code's use of `autowater` to separate water-cued versus delayed-response trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The agent converts each trial to a constant binary label: `WC = 0` when `autowater != 0`, otherwise `DR = 1`. It then repeats that label across all time bins for the trial.

ii. 
```python
trial_labels.append(
    (
        lick_dir,
        0 if autowater[trial_idx] != 0 else 1,
        1 if hit[trial_idx] != 0 else 0,
    )
)

np.full(TIME_AXIS.size, context_label, dtype=np.int64)
```

iii. Step 5 says the repetition across time was a formatting choice so all outputs share the same `(n_output, n_timepoints)` structure while preserving per-trial context labels.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived mainly from `obj["bp"]["hit"]`, with `obj["bp"]["miss"]` and `obj["bp"]["no"]` used in filtering so that only hit/miss trials remain and ignore trials are excluded.

ii. 
```python
hit = ensure_1d_numeric(bp["hit"]) != 0
miss = ensure_1d_numeric(bp["miss"]) != 0
no = ensure_1d_numeric(bp["no"]) != 0
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
...
1 if hit[trial_idx] != 0 else 0
```

iii. The notes cite the reference helper `getOutcome`, which uses `bp.hit` and sets ignore trials to `NaN`. The Python port instead removes ignore trials earlier and then uses `hit` as the binary outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. After filtering to hit/miss trials, the code assigns `correct = 1` if `bp.hit` is nonzero and `incorrect = 0` otherwise. That trial-level label is repeated across all time bins.

ii. 
```python
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
...
trial_labels.append(
    (
        lick_dir,
        ...,
        1 if hit[trial_idx] != 0 else 0,
    )
)
...
np.full(TIME_AXIS.size, outcome_label, dtype=np.int64)
```

iii. The justification in the notes is that outcome must remain well-defined for every retained trial, so no-response trials are excluded before outcome construction.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-view DeepLabCut trajectory in `obj["traj"][0]`, specifically the feature named `"tongue"`, using its `ts` coordinate array and `frameTimes`.

ii. 
```python
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
```

iii. The notes say the agent reduced the many available kinematic features to just the ones needed for the requested outputs, and used a single tongue feature because the task asked for one tongue-velocity output stream rather than the full kinematic feature set.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code aligns tongue x/y positions to the common neural time axis using video frame times and a video offset, computes x/y gradients over that aligned trace, replaces undefined tongue velocity samples with zeros, and then collapses x/y into scalar speed magnitude `sqrt(vx^2 + vy^2)`.

ii. 
```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
ypos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 1], left=np.nan, right=np.nan)

tongue_vx, tongue_vy = compute_velocity(tongue_x, tongue_y, "tongue")
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. The notes and trajectory justify this as a simplified scalar version of the reference kinematic pipeline. The trajectory also shows a later fix: because tongue visibility is sparse, the threshold was recalculated from visible-only timepoints.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The tongue threshold is the 50th percentile of tongue speed, but only over visible tongue samples from kept trials. Each time bin is then labeled `1` if speed is at or above that threshold and the tongue is visible; invisible bins are forced into the low bin (`0`).

ii. 
```python
tongue_visible = np.isfinite(tongue_x) & np.isfinite(tongue_y)
tongue_visible_kept = tongue_visible[:, kept_trials]
if np.any(tongue_visible_kept):
    tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
else:
    tongue_threshold = 0.0

tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. The trajectory records the reason explicitly: using all timepoints made the tongue median collapse to zero in one sample session because invisible periods dominated, so the agent switched to visible-only thresholding and mapped invisible periods to the low category.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue position is interpolated onto the same `TIME_AXIS` used for neural data after subtracting both the session's video offset and each trial's go-cue time. Velocity is then computed on that aligned grid.

ii. 
```python
vidshift = compute_vidshift(obj)
shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
...
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
```

iii. The notes tie this directly to the reference `loadMotionEnergy` / `getKinematicsFromVideo` alignment strategy: bring video-derived signals onto the same trial-centered neural time base.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-view DLC trajectories in `obj["traj"][1]`, using the features `"top_paw"` and `"bottom_paw"`.

ii. 
```python
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
```

iii. The notes say this was a deliberate reduction of the available paw tracking features to the minimal pair needed to produce one paw-velocity output.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw feature is aligned to the neural time base, converted to x/y velocity with the same reference-style `gradient` calculation, collapsed to scalar speed magnitude, and then the two paw speeds are averaged wherever either paw feature is finite.

ii. 
```python
paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
paw_speeds.append(np.sqrt(paw_vx**2 + paw_vy**2))
...
paw_stack = np.stack(paw_speeds, axis=0)
paw_counts = np.sum(np.isfinite(paw_stack), axis=0)
paw_speed = np.full(paw_counts.shape, np.nan, dtype=np.float64)
np.divide(np.nansum(paw_stack, axis=0), paw_counts, out=paw_speed, where=paw_counts > 0)
```

iii. Step 5 says the scalar-speed collapse was chosen because the user asked for one categorical paw-velocity output rather than separate x/y regressors.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The threshold is the 50th percentile of all aligned paw-speed values from kept trials in the session. Time bins at or above the threshold are labeled `1`; lower bins are `0`.

ii. 
```python
paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
...
(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64)
```

iii. The notes say this follows the decoder instruction literally: a per-session median split for continuous movement outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned the same way as tongue trajectories: `frameTimes - vidshift - align_times[trial_idx]` are interpolated onto `TIME_AXIS`, so the resulting paw-speed trace lives on the same time bins as neural activity.

ii. 
```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
ypos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 1], left=np.nan, right=np.nan)
```

iii. The notes justify this as a direct port of the reference video-alignment logic so kinematics and neural activity are co-registered trial by trial.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically `me.data`, with timing anchored by `obj["traj"][0]["frameTimes"]` or, if necessary, the number of video frames in `obj["traj"][0]["ts"]`.

ii. 
```python
me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
me = me_mat["me"]
return {
    "data": np.atleast_1d(me.data),
    "moveThresh": float(me.moveThresh),
}
```

iii. The notes say this mirrors the reference `loadMotionEnergy` path for ephys sessions, where motion energy lives in a companion MATLAB file rather than inside the session object.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code interpolates each trial's continuous motion-energy trace from video time into the common neural time axis after subtracting a video offset and the go-cue alignment time. Missing edge values are filled by nearest-neighbor interpolation. It does not recompute motion energy from video frames and does not use the stored manual `moveThresh` for the final output.

ii. 
```python
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

iii. The notes justify this as using the paper's already-computed motion-energy variable while changing only the final discretization step to match the decoder task's required median split.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The threshold is the 50th percentile of all aligned motion-energy values from kept trials in the session. Each time bin is then categorized as `0` below threshold or `1` at/above threshold.

ii. 
```python
me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))
...
(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64)
```

iii. The notes say this was driven by the decoder instructions, even though the paper's own move/non-move analyses use a manually chosen per-session threshold (`me.moveThresh`) for a different purpose.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Like kinematics, motion energy is aligned by converting video timestamps into trial-centered time relative to the per-trial `goCue`, then interpolating onto `TIME_AXIS`, which is the same grid used for neural bins.

ii. 
```python
vidshift = compute_vidshift(obj)
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
```

iii. The notes explicitly cite the reference `loadMotionEnergy` function as the model for this alignment step.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are handled with fallbacks rather than strict failure, except when too much essential data is lost. If `frameTimes` are absent or all-NaN, the code synthesizes them at 400 Hz; if position or motion-energy traces have NaNs, non-tongue traces are filled with nearest values; tongue velocity NaNs become zero and invisible tongue bins are set to the low category; trials with no first post-go lick are dropped; sessions with too few valid trials or no surviving units raise errors.

ii. 
```python
if frame_times is None:
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
...
if frame_times.size == 0 or np.all(np.isnan(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

if "tongue" not in feat_name:
    xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])
else:
    xv = np.where(np.isfinite(xv), xv, 0.0)

if lick_dir is None:
    continue
if kept_trials.size < 2:
    raise RuntimeError(...)
```

iii. The notes and trajectory say the aim was to match the paper's special handling of missing kinematic values, especially for tongue visibility, while keeping the converted dataset usable by the decoder verifier.

## 11-a. What are the most time-consuming steps of the code?

i. The slowest steps are the per-session `mat73.loadmat()` reads, the trial-by-trial interpolation of kinematic and motion-energy traces, and the per-unit firing-rate/binning loops. These dominate runtime because they repeatedly traverse large MATLAB/HDF5 arrays in Python.

ii. 
```python
obj = mat73.loadmat(spec.data_path)["obj"]

for trial_idx in range(n_trials):
    ...
    xpos[:, trial_idx] = np.interp(...)

for unit_idx, use_unit in enumerate(quality_keep):
    ...
    mean_fr = mean_firing_rate_window(...)
    trialdat = bin_unit_spikes(...)
```

iii. The notes explicitly call out full-session `mat73` loading and trial-by-trial kinematic interpolation as the main inefficiencies. The runtime logs in `conversion_full_out.txt` also show that each session spends several seconds in this conversion path.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalarized and could have been vectorized further: the per-column smoothing convolution in `my_smooth`, the per-trial interpolation in `align_feature_positions` and `align_motion_energy`, the per-trial velocity loop in `compute_velocity`, the per-unit mean-FR/binning loop in `convert_session`, and the per-trial output assembly loop.

ii. 
```python
for col in range(x_filt.shape[1]):
    out[:, col] = np.convolve(...)

for trial_idx in range(n_trials):
    ...

for trial_idx in range(xpos.shape[1]):
    ...

for unit_idx, use_unit in enumerate(quality_keep):
    ...

for local_trial_idx, trial_idx in enumerate(kept_trials):
    ...
```

iii. Step 6 says the agent vectorized spike binning with `np.add.at` but knowingly left the kinematic alignment path as Python loops. So the code reflects partial, not complete, vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly redoes similar interpolation work feature by feature and trial by trial: tongue alignment, top-paw alignment, bottom-paw alignment, and motion-energy alignment are all separate passes over the same per-trial video timing information. It also loops over each unit once to estimate FR and again to bin spikes, so neural preprocessing traverses unit data multiple times.

ii. 
```python
tongue_x, tongue_y = align_feature_positions(...)
...
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(...)
...
motion_energy = align_motion_energy(obj, me, align_times, TIME_AXIS)

mean_fr = mean_firing_rate_window(...)
...
trialdat = bin_unit_spikes(...)
```

iii. The notes describe this as a conscious simplicity tradeoff: only a few features are extracted, but each is still processed in its own pass rather than through a shared vectorized alignment routine.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several things the downstream decoder does not use: `moveThresh` is loaded but ignored; `kept_unit_indices` is accumulated but never returned; plotting summaries, histograms, and per-session count summaries are only for debugging/documentation; and helper code like `fill_nearest_2d()` is defined but unused. More broadly, the decoder only consumes `neural`, `input`, `output`, and metadata, so the extra diagnostic bookkeeping is discarded.

ii. 
```python
return {
    "data": np.atleast_1d(me.data),
    "moveThresh": float(me.moveThresh),
}

kept_unit_indices = []
...
kept_unit_indices.append(unit_idx)

def fill_nearest_2d(x: np.ndarray) -> np.ndarray:
    ...

"thresholds": {...},
"summary": {...},
```

iii. The notes frame these extras as support for the required sanity checks, plots, and human-readable documentation rather than for the final decoder input itself.
