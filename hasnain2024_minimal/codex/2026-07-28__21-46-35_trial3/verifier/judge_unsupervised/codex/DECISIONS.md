# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not discover sessions dynamically. It hard-codes a 12-session Figure 8 roster in `SESSION_SPECS`, then for each session opens one electrophysiology/behavior HDF5 `.mat` file and one separate `motionEnergy_*.mat` file. Within each session file it reads behavior (`obj/bp`), event times (`obj/bp/ev`), stimulation flags (`obj/bp/stim`), spike data (`obj/clu` for the selected probe), synchronization data (`obj/sglx`), and tracked video trajectories (`obj/traj`).

ii. 
```python
DATA_DIR = ROOT / "data" / "Ephys_Behavior"

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

def load_session(spec: SessionSpec) -> dict:
    with h5py.File(spec.data_path, "r") as f:
        bp_group = f["obj/bp"]
        ev_group = bp_group["ev"]
        stim_group = bp_group["stim"]
        ...
        fs = read_h5_scalar(f["obj/sglx/fs"])
        bitcode_bitstart = read_h5_vector(f["obj/sglx/bitcode/bitstart"])
        ...
        traj_root = f["obj/traj"]
        side_group = f[h5_ref_at(traj_root, 0)]
        bottom_group = f[h5_ref_at(traj_root, 1)]

        raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
```

iii. The notes say the roster was taken from `code/Scripts/Figure 8/Figure8a_thru_c.m` plus the per-animal loader files, because the decoder task requires the two-context ALM cohort rather than other datasets.

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from the hard-coded `SessionSpec.animal` strings. After all sessions are loaded, the agent creates `subjects` as the ordered unique animal IDs and `subject_idx` as the subject index for each session. This yields 7 packaged subject IDs, even though the paper text says 6 mice.

ii. 
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int

def build_dataset() -> dict:
    sessions = [load_session(spec) for spec in SESSION_SPECS]

    subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
    subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
    ...
    "subjects": subjects,
    "subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in sessions], dtype=np.int64),
```

iii. The notes explicitly justify preserving the packaged subject IDs despite the manuscript mismatch: the agent says it kept the IDs present in the provided files rather than collapsing or renaming mice without evidence.

## 1-c. How are the data split into sessions?

i. Sessions are exactly the hard-coded `SESSION_SPECS` entries. Each `SessionSpec` defines one session by animal, date, and selected probe, and `build_dataset()` loads one dataset entry per `SessionSpec`.

ii. 
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

def build_dataset() -> dict:
    sessions = [load_session(spec) for spec in SESSION_SPECS]
```

iii. The notes say this roster was copied from the Figure 8 context-analysis pipeline and per-animal loader scripts.

## 1-d. How are the data split into trials?

i. Trials are split by session-level trial indices. The agent reads `bp["Ntrials"]`, builds a boolean inclusion mask, and then uses `included_trials = np.flatnonzero(included_mask)`. Neural, input, and output trial lists contain one entry per included trial, in that order.

ii. 
```python
bp = {
    "Ntrials": int(round(read_h5_scalar(bp_group["Ntrials"]))),
    ...
}

included_mask = trial_selector(bp)
included_trials = np.flatnonzero(included_mask)

session_neural = [
    np.asarray(neural_arr[:, :, tr].T, dtype=np.float32)
    for tr in range(neural_arr.shape[2])
]
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. No separate rationale is given beyond mirroring the Figure 8 session-wise loading and then exporting only usable trials.

## 1-e. How are trials filtered based on quality controls?

i. The exported dataset keeps trials where `hit` or `miss` is true, `early` is false, and `stim_enable` is false. Ignore/no-response trials are excluded. There is no extra end-of-session truncation.

ii. 
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)
```

iii. The notes justify this as “hit | miss”, `stim.enable == 0`, `early == 0`, and say ignore/no-response trials were excluded because the requested outcome target is binary correct/incorrect and the paper methods say ignore trials were omitted from analyses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the spike cluster tables for the selected probe. Specifically, the code uses cluster quality labels, per-spike trial assignments, and per-spike times from `obj/clu`, plus per-trial `goCue` times from `obj/bp/ev`.

ii. 
```python
clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
qds = clu_group["quality"]
...
trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
trialtm = np.asarray(trialtm_ds[()], dtype=np.float64).reshape(-1)
aligned = trialtm - align_times[session_trial]
```

iii. The notes say the neural pipeline follows the Figure 8 context-analysis code: ALM ephys sessions only, go-cue alignment, 10 ms bins, causal smoothing, and low-FR filtering.

## 2-b. How is the `neural` data processed?

i. For each kept cluster, the code subtracts each spike’s trial-specific go-cue time, bins aligned spikes into 10 ms bins over `[-3.0, 2.5]` s, converts counts to rates by dividing by `DT`, and applies a causal Gaussian smoothing kernel of width 15 with reflect padding. The exported trial matrices are smoothed single-trial firing rates.

ii. 
```python
def build_edges_and_time() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    ...

def build_neural_trials(...):
    out = np.zeros((n_time, included_trials.size), dtype=np.float64)
    for col, tr in enumerate(included_trials):
        spike_mask = session_trial == tr
        ...
        counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
        out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. The notes say this matches the repository’s `mySmooth.m` and Figure 8 parameters: `goCue` alignment, `dt = 1/100`, `[-3.0, 2.5]` s, `smooth = 15`, reflect padding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first rejects clusters with quality labels in `{"garbage", "gabrga", "noisy", "real?"}`. It then computes a mean firing rate from smoothed PSTHs averaged across the Figure 8 condition set and removes clusters with mean rate `<= 1 Hz`.

ii. 
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR = 1.0

quality = read_h5_string(f, h5_ref_at(qds, clu_idx)).strip().lower()
if quality in BAD_QUALITIES:
    continue

mean_fr = compute_psth_mean_fr(
    session_trial=session_trial,
    aligned_spikes=aligned,
    condition_masks=condition_masks,
    edges=edges,
)
if mean_fr <= LOW_FR:
    continue
```

iii. The notes explicitly cite `findClusters(..., {'all'})` for the quality filter and Figure 8 / context scripts for the `> 1 Hz` low-FR threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike is aligned by subtracting the trial’s `goCue` time, so all neural trial matrices are centered on go-cue onset. The exported `metadata["temporal_alignment_event"]` is `"Go cue onset"`.

ii. 
```python
ALIGN_EVENT = "goCue"
...
aligned = trialtm - align_times[session_trial]
...
"metadata": {
    ...
    "temporal_alignment_event": "Go cue onset",
    "off_start": float(TMIN),
    "off_end": float(TMAX),
}
```

iii. The notes say `goCue` was taken directly from the reference Figure 8 pipeline.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural and aligned video-derived outputs use 10 ms bins. Neural spikes are binned directly into 10 ms bins; video and motion-energy traces are interpolated onto the same 10 ms neural time axis. There is no additional rebinning beyond that interpolation/resampling.

ii. 
```python
DT = 1 / 100

def build_edges_and_time() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time

"time_bin_size": float(DT * 1000),
```

iii. The notes explicitly state “10 ms neural/video bins” and `dt = 1/100` s.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a separate raw continuous variable. The code synthesizes it from the fixed window parameters `TMIN`, `TMAX`, and `DT`, with the conceptual anchor being the chosen alignment event `goCue`.

ii. 
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100

def build_edges_and_time() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
```

iii. The notes describe this as a decoder input chosen to represent time from go cue as a `1 x 550` time series for each trial.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes bin centers spanning `[-3.0, 2.5]` seconds at 10 ms spacing, then duplicates that same 1D time vector for every trial in a session.

ii. 
```python
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The notes say the input is one continuous variable, `time_from_go_cue_seconds`, stored as a `1 x 550` time series for each trial.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the exact same `time` vector as the neural bin centers, so each input sample corresponds one-to-one with a neural time bin.

ii. 
```python
edges, time = build_edges_and_time()
...
session_neural = [...]
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. No extra rationale is given beyond using the same aligned neural time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the behavioral trial labels `bp["R"]` and `bp["hit"]`. The code infers actual response direction from instructed side plus correctness; it does not read actual lick timestamps from `lickL`/`lickR`.

ii. 
```python
def lick_direction_from_trial(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
    if hit:
        return int(is_right_trial)
    return int(not is_right_trial)
```

iii. The notes justify this choice as “actual response direction, not instructed side,” and summarize it as: correct-right plus incorrect-left become right, while correct-left plus incorrect-right become left.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code converts each included trial into a binary label and then repeats that constant label across all 550 time bins for the trial.

ii. 
```python
lick_dir = lick_direction_from_trial(bp, tr)
...
tr_out = np.vstack(
    [
        np.full(tongue_speed.shape[0], lick_dir, dtype=np.int16),
        ...
    ]
)
```

iii. The notes say this was chosen to produce the requested per-trial decoder target while using actual response direction rather than instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context comes from `bp["autowater"]`.

ii. 
```python
context = 0 if bp["autowater"][tr] else 1
```

iii. The notes explicitly map `WC = 0`, `DR = 1` and identify `autowater` as the context indicator.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code converts `autowater` to a binary context label per trial and repeats that label across all time bins for the trial.

ii. 
```python
tr_out = np.vstack(
    [
        ...,
        np.full(tongue_speed.shape[0], context, dtype=np.int16),
        ...
    ]
)
```

iii. The notes say the context target follows the paper’s DR/WC split and uses `autowater` to encode it.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["hit"]`, after restricting exported trials to hit-or-miss trials.

ii. 
```python
outcome = 1 if bp["hit"][tr] else 0
```

iii. The notes justify the binary outcome target as `incorrect = 0`, `correct = 1`, with ignore trials removed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps hit trials to 1 and miss trials to 0, then repeats that constant value across all time bins in the trial.

ii. 
```python
tr_out = np.vstack(
    [
        ...,
        np.full(tongue_speed.shape[0], outcome, dtype=np.int16),
        ...
    ]
)
```

iii. The notes describe this as the requested binary correct/incorrect decoder target.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from bottom-camera tracked tongue landmarks. The code locates `top_tongue` and `bottom_tongue` in `obj/traj` feature names, reads their `ts` position arrays and `frameTimes`, aligns them to go cue, computes x/y velocities, averages top and bottom tongue velocities into a tongue-tip estimate, and then takes speed magnitude.

ii. 
```python
feat_indices = {
    "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
    "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
    ...
}
...
top_tongue_xvel, top_tongue_yvel = velocities["top_tongue"]
bottom_tongue_xvel, bottom_tongue_yvel = velocities["bottom_tongue"]
tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
```

iii. The notes justify this as using the same aligned DLC trajectories as the repository and reducing them to bottom-view tongue-tip speed.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue feature, the code interpolates positions onto the neural time axis, computes velocity with `np.gradient`, leaves tongue NaNs as zeros, averages the two tongue velocities, converts to speed magnitude, and replaces remaining NaNs with zero.

ii. 
```python
def aligned_feature_velocity(xpos: np.ndarray, ypos: np.ndarray, feat_name: str) -> tuple[np.ndarray, np.ndarray]:
    ...
    xvel[:, tr] = np.gradient(tsinterp[:, 0])
    yvel[:, tr] = np.gradient(tsinterp[:, 1])
    ...
    else:
        xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0
        yvel[:, tr][np.isnan(yvel[:, tr])] = 0.0

...
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
```

iii. The notes say this follows the repository’s tongue-visibility convention that invisible tongue samples get zero velocity.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code does **not** use the literal per-session median over all exported tongue-speed bins. Instead it computes the 50th percentile only over positive tongue-speed bins, then labels each sample as `1` if `>= threshold` and `0` otherwise.

ii. 
```python
positive_tongue = tongue_use[tongue_use > 0]
if positive_tongue.size:
    tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
else:
    tongue_thresh = 0.0
...
(tongue_speed[:, tr] >= tongue_thresh).astype(np.int16),
```

iii. The notes and trajectory explicitly justify this deviation: using all bins makes the median zero because invisible-tongue samples were set to zero, which would collapse the requested `>= threshold` classifier into a constant all-ones label.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are aligned with `frameTimes - vidshift - goCue`, then interpolated onto `taxis = time + ADVANCE_MOVEMENT`, where `time` is the 10 ms neural bin-center axis. That makes tongue velocity sample-for-sample aligned with neural data.

ii. 
```python
taxis = time + ADVANCE_MOVEMENT
...
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
...
session_output, thresholds = derive_session_outputs(
    ...
    tongue_speed=tongue_speed,
    ...
)
```

iii. The notes explicitly cite the repository’s video alignment rule: `frameTimes - vidshift - goCue`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera `top_paw` and `bottom_paw` tracked positions in `obj/traj`, using their `ts` arrays and `frameTimes`.

ii. 
```python
feat_indices = {
    ...,
    "top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
    "bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
}
```

iii. The notes say paw speed was derived from the aligned DLC trajectories used by the repository’s kinematics code and then reduced to a single scalar speed.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code interpolates paw positions to the neural time axis, computes x/y velocities with baseline subtraction for non-tongue features, converts each paw landmark to speed magnitude, averages the two paw speeds, and replaces remaining NaNs with zero.

ii. 
```python
top_paw_xvel, top_paw_yvel = velocities["top_paw"]
bottom_paw_xvel, bottom_paw_yvel = velocities["bottom_paw"]
top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The notes justify this as using the same aligned DLC trajectories while reducing them to a scalar paw-speed target for the decoder.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The code uses the literal per-session 50th percentile over all exported paw-speed bins. Values below the threshold are `0`; values at or above it are `1`.

ii. 
```python
paw_thresh = float(np.nanpercentile(paw_use, 50))
...
(paw_speed[:, tr] >= paw_thresh).astype(np.int16),
```

iii. The notes say paw velocity uses the “literal per-session 50th percentile over all exported bins.”

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw positions are aligned with the same `frameTimes - vidshift - goCue` transform and interpolated onto the same 10 ms `taxis` used for neural data.

ii. 
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
...
velocities[feat_name] = aligned_feature_velocity(xpos, ypos, feat_name)
```

iii. The notes group paw processing with the same aligned DLC kinematics pipeline used for other movement variables.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<animal>_<date>.mat` files, specifically `me.data` and `me.moveThresh`.

ii. 
```python
def load_motion_energy(path: Path) -> tuple[np.ndarray, float]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
    return trial_arrays, float(me.moveThresh)
```

iii. The trajectory says the agent chose the separate `motionEnergy_*.mat` files because the embedded motion-energy field was not consistently present, matching the repository loader.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the code interpolates motion-energy samples from video frame time onto the 10 ms neural time axis after subtracting `vidshift` and the trial’s `goCue`. It then nearest-fills edge NaNs; if a trial becomes entirely NaN, the helper effectively returns zeros.

ii. 
```python
def aligned_motion_energy(...):
    out = np.full((taxis.size, n_trials), np.nan, dtype=np.float64)
    for tr in range(n_trials):
        frame_times = frame_times_by_trial[tr]
        me_trial = np.asarray(raw_motion_energy[tr], dtype=np.float64).reshape(-1)
        out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
        out[:, tr] = fill_nearest_1d(out[:, tr])
    return out
```

iii. The notes explicitly say this matches `findVideoOffset.m` and `loadMotionEnergy.m`: interpolate onto the neural axis, then nearest-fill edge NaNs.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded at the per-session 50th percentile over all exported aligned motion-energy bins, then binarized with the requested `< median` / `>= median` rule.

ii. 
```python
me_thresh = float(np.nanpercentile(me_use, 50))
...
(motion_energy[:, tr] >= me_thresh).astype(np.int16),
```

iii. The notes say motion energy uses the literal per-session median over all exported bins.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It is aligned with `frameTimes - vidshift - goCue` and then resampled onto the same 10 ms neural time axis.

ii. 
```python
motion_energy = aligned_motion_energy(
    frame_times_by_trial=frame_times_by_trial,
    raw_motion_energy=raw_motion_energy,
    align_times=bp[ALIGN_EVENT],
    vidshift=vidshift,
    taxis=taxis,
)
```

iii. The notes explicitly cite the same video-alignment formula as for the other movement streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses several ad hoc fallbacks. Missing/invalid tongue velocity samples are zeroed; non-tongue position and velocity NaNs are nearest-filled; motion-energy interpolation NaNs are nearest-filled and all-NaN traces become zeros; trials with NaN dropped-frame metadata are skipped for feature extraction; if a session has fewer than 2 usable trials or no surviving units, it raises an error. It does not implement every MATLAB fallback, such as reconstructing missing `frameTimes` from sample count inside the main kinematic path.

ii. 
```python
def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    ...
    if idx.size == 0:
        return np.zeros_like(x)

if np.isnan(ndropped):
    continue

if "tongue" not in feat_name:
    xpos[:, tr] = fill_nearest_1d(xpos[:, tr])
    ypos[:, tr] = fill_nearest_1d(ypos[:, tr])
...
else:
    xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0
    yvel[:, tr][np.isnan(yvel[:, tr])] = 0.0
```

iii. The notes mention matching the repository’s nearest-fill and tongue-zeroing behavior, and the trajectory highlights the tongue-zero convention as the reason the naive median threshold collapsed.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive work is repeated per-session, per-trial interpolation and per-unit spike processing: motion-energy interpolation, feature-position interpolation, feature-velocity computation, PSTH/mean-FR calculation for every cluster, and building trial-aligned neural matrices by histogramming spikes separately for every kept trial.

ii. 
```python
for tr in range(n_trials):
    ...
    out[:, tr] = interp_with_nan(...)

for feat_name, feat_idx in feat_indices.items():
    xpos, ypos = aligned_feature_position(...)
    velocities[feat_name] = aligned_feature_velocity(xpos, ypos, feat_name)

for clu_idx in range(qds.shape[0]):
    ...
    mean_fr = compute_psth_mean_fr(...)
    ...
    neural_by_trial.append(build_neural_trials(...))
```

iii. The agent did not document runtime hotspots directly; this follows from the structure of the implementation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: per-trial interpolation in `aligned_motion_energy`, per-trial loops in `aligned_feature_position` and `aligned_feature_velocity`, per-condition PSTH construction in `compute_psth_mean_fr`, per-trial histogramming in `build_neural_trials`, and the per-cluster loop that repeatedly scans spikes and smooths PSTHs.

ii. 
```python
for tr in range(n_trials):
    ...

for tr in range(xpos.shape[1]):
    ...

for cond_mask in condition_masks:
    ...

for col, tr in enumerate(included_trials):
    ...

for clu_idx in range(qds.shape[0]):
    ...
```

iii. No explicit justification was given; this is an evaluation of the code structure.

## 11-c. What processing does the code repeat multiple times?

i. It repeats interpolation and gradient computations separately for each tracked feature, repeats similar per-trial histogramming once for low-FR PSTHs and again for exported single-trial neural data, and repeats the same constant time input vector for every trial instead of storing it once per session.

ii. 
```python
for feat_name, feat_idx in feat_indices.items():
    xpos, ypos = aligned_feature_position(...)
    velocities[feat_name] = aligned_feature_velocity(xpos, ypos, feat_name)

mean_fr = compute_psth_mean_fr(...)
...
neural_by_trial.append(
    build_neural_trials(...)
)

session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The agent did not call this out, but it is visible in the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some values only for sanity logging or not at all downstream: `raw_move_thresh`, `prefilter_unit_count`, `kept_cluster_indices`, and `cluster_ids`; it loads side-camera data even though exported outputs only use bottom-camera tongue/paw plus motion energy; it computes all seven Figure 8 condition masks even though the exported dataset only keeps hit/miss trials; and it reads behavioral fields like `L`, `sample`, and `delay` without using them in the final dataset.

ii. 
```python
raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
...
kept_cluster_indices = []
prefilter_unit_count = 0
...
cluster_ids = []
...
condition_masks = compute_condition_masks(bp)
...
"raw_motion_energy_threshold": float(raw_move_thresh),
```

iii. No explicit rationale was given beyond using these values for debugging/sanity checks and mirroring Figure 8 condition logic.
