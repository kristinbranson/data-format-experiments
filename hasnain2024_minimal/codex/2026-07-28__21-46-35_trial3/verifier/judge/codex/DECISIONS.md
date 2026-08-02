# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes a 12-session Figure 8 ALM two-context roster, then loads one electrophysiology `.mat` file and one motion-energy `.mat` file per session. The main session file is opened with `h5py` and read field-by-field from the MATLAB HDF5 structure; motion energy is loaded separately with `scipy.io.loadmat`.

ii.
```python
DATA_DIR = ROOT / "data" / "Ephys_Behavior"

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

with h5py.File(spec.data_path, "r") as f:
    bp_group = f["obj/bp"]
    ...

raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
```

iii. `CONVERSION_NOTES.md` says the roster was taken from `code/Scripts/Figure 8/Figure8a_thru_c.m` plus the per-animal loader files in `code/DataLoadingScripts/Recording and video/`, because the decoder required the two-context cohort.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `animal` field in each `SessionSpec`. After all sessions are loaded, the script builds an ordered unique subject list and a per-session `subject_idx`.

ii.
```python
subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}

"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in sessions], dtype=np.int64),
```

iii. The notes say the agent preserved the packaged subject IDs directly from the roster, even though the manuscript text says six mice while the packaged roster yields seven IDs.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` is treated as one session. `build_dataset()` calls `load_session(spec)` once per roster entry and stores each returned session as one element of `neural`, `input`, and `output`.

ii.
```python
def build_dataset() -> dict:
    sessions = [load_session(spec) for spec in SESSION_SPECS]

    data = {
        "neural": [sess["neural"] for sess in sessions],
        "input": [sess["input"] for sess in sessions],
        "output": [sess["output"] for sess in sessions],
        ...
    }
```

iii. The stated justification is that the Figure 8 context-analysis pipeline defines the session roster explicitly, so the converter should follow that session list rather than auto-discovering files.

## 1-d. How are the data split into trials?

i. Trial-level arrays come from `obj/bp` fields with length `Ntrials`. Included trial indices are computed with `trial_selector(bp)`, and neural/output arrays are then built one trial at a time using those integer trial indices. Spike times are assigned to trials through the cluster `trial` field.

ii.
```python
included_mask = trial_selector(bp)
included_trials = np.flatnonzero(included_mask)

session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1

for col, tr in enumerate(included_trials):
    spike_mask = session_trial == tr
    ...
```

iii. The notes say exported trials were limited to the hit/miss non-early non-stimulation trials needed for the decoder outputs.

## 1-e. How are trials filtered based on quality controls?

i. Exported trials are restricted to hit or miss trials with `early == 0` and `stim.enable == 0`. Ignore/no-response trials and stimulation trials are excluded; WC and DR trials are both retained through `autowater`.

ii.
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)
```

iii. `CONVERSION_NOTES.md` says this was chosen because the requested binary `outcome` target is correct/incorrect, and the paper methods say early and ignore trials were omitted from analyses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the selected probe’s cluster spike-time fields: per-spike trial IDs (`trial`), within-trial spike times (`trialtm`), per-trial go-cue times (`bp.ev.goCue`) for alignment, and cluster quality labels for filtering.

ii.
```python
clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
qds = clu_group["quality"]

trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
...
aligned = trialtm - align_times[session_trial]
```

iii. The notes describe the neural export as reference-matched Figure 8 ALM activity with the same alignment and filtering used by the repository.

## 2-b. How is the `neural` data processed?

i. For each kept unit, aligned spike times are histogrammed into 10 ms bins over `[-3.0, 2.5]` s relative to go cue, converted to firing rate by dividing by `DT`, and smoothed with a causal Gaussian kernel matching `mySmooth.m`. The saved per-trial matrices are smoothed firing rates, not raw spike counts.

ii.
```python
counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")

session_neural = [
    np.asarray(neural_arr[:, :, tr].T, dtype=np.float32)
    for tr in range(neural_arr.shape[2])
]
```

iii. The notes explicitly say “smoothed single-trial firing rates” with 10 ms bins, a 15-sample causal Gaussian, and reflect padding to match repository preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied. First, clusters with quality labels in `{"garbage", "gabrga", "noisy", "real?"}` are excluded. Second, surviving units are removed if their mean firing rate across the Figure 8 condition PSTHs is `<= 1 Hz`.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
if quality in BAD_QUALITIES:
    continue
...
mean_fr = compute_psth_mean_fr(...)
if mean_fr <= LOW_FR:
    continue
```

iii. The notes justify this as matching `findClusters(..., {'all'})` plus the Figure 8 / context-script `> 1 Hz` low-firing-rate filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the trial’s `goCue` time, and then only the `[-3.0, 2.5]` s window relative to go cue is binned and saved.

ii.
```python
ALIGN_EVENT = "goCue"
...
aligned = trialtm - align_times[session_trial]
...
edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
```

iii. The notes say the alignment event was taken directly from the Figure 8 code and from the decoder instruction itself: go cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural and aligned behavioral traces use 10 ms bins (`DT = 1/100` s). Neural data are produced directly in that bin size; there is no second rebinning step after the initial histogramming.

ii.
```python
DT = 1 / 100
...
edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. The notes state this was copied from the Figure 8 context-analysis parameters.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not taken from a raw recorded signal. It is a synthetic per-bin time axis derived from the converter constants `TMIN`, `TMAX`, and `DT`, with go cue serving as the zero point through the alignment choice.

ii.
```python
def build_edges_and_time() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
```

iii. The notes say the only decoder input is `time_from_go_cue_seconds`, stored as a `1 x 550` time series per trial.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs bin centers from `[-3.0, 2.5]` s and copies the same `1 x 550` vector into every trial of a session.

ii.
```python
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The justification in the notes is simply that the decoder asked for time from go cue onset as the single input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the exact same bin centers as the neural data, so the input time axis is sample-aligned to the neural matrices with identical length and spacing.

ii.
```python
edges, time = build_edges_and_time()
...
session_neural = [...]
session_input = [np.asarray(time[None, :], dtype=np.float32) ...]
```

iii. The notes say both neural and movement streams were aligned to go cue with the same 10 ms grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp["R"]` plus the hit/miss outcome. `R` is treated as the instructed right-trial flag, and miss trials are flipped to infer the actual response direction.

ii.
```python
def lick_direction_from_trial(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
    if hit:
        return int(is_right_trial)
    return int(not is_right_trial)
```

iii. The notes explicitly justify this as “actual response direction, not instructed side.”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code maps trials to a binary label: hit-right and miss-left become `1` (right), while hit-left and miss-right become `0` (left). That per-trial label is then repeated across all time bins in the output matrix.

ii.
```python
lick_dir = lick_direction_from_trial(bp, tr)
...
np.full(tongue_speed.shape[0], lick_dir, dtype=np.int16),
```

iii. The notes call this “choice logic” and describe the same hit/miss flip rule.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp["autowater"]`, which the reference code and tutorial use as the WC-versus-DR indicator.

ii.
```python
context = 0 if bp["autowater"][tr] else 1
```

iii. The notes state the mapping directly: `WC = 0`, `DR = 1`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code converts `autowater == 1` to WC (`0`) and `autowater == 0` to DR (`1`), then repeats that binary label across all bins in the trial output.

ii.
```python
context = 0 if bp["autowater"][tr] else 1
...
np.full(tongue_speed.shape[0], context, dtype=np.int16),
```

iii. The notes justify this with the repository convention that `autowater` is the proxy for water-cued blocks.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["hit"]` after the trial set has already been restricted to hit or miss trials.

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
```

iii. The notes say the requested target was binary `incorrect` versus `correct`, so hit/miss trials were used and ignore trials were excluded.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hits are encoded as `1` and misses as `0`, then the per-trial label is repeated over time in the exported output matrix.

ii.
```python
np.full(tongue_speed.shape[0], outcome, dtype=np.int16),
```

iii. The notes state the same mapping: `incorrect = 0`, `correct = 1`.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from bottom-camera DeepLabCut tongue positions: `top_tongue` and `bottom_tongue`, read from `obj.traj{2}.ts` and aligned with `frameTimes`.

ii.
```python
feat_indices = {
    "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
    "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
    ...
}
...
xy = read_feature_xy(f, view_group, tr, feat_idx)
```

iii. The notes say tongue speed was computed from bottom-view tongue-tip velocity using the average of `top_tongue` and `bottom_tongue`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The bottom-view tongue positions are linearly interpolated onto the neural time axis, no nearest-fill is applied to tongue positions, gradients are taken per coordinate, NaN tongue velocities are set to zero, and the top/bottom tongue-point velocities are averaged before taking Euclidean speed.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
...
xvel[:, tr] = np.gradient(tsinterp[:, 0])
yvel[:, tr] = np.gradient(tsinterp[:, 1])
...
xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0
yvel[:, tr][np.isnan(yvel[:, tr])] = 0.0
...
tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
```

iii. The notes justify this as using the same aligned DLC trajectories as the repository kinematics code, then turning them into a decoder-specific speed target.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code does **not** use the literal per-session median over all exported bins. Instead, it computes the 50th percentile only over strictly positive tongue-speed values, then labels each bin as `0` below threshold and `1` at or above threshold.

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

iii. The notes justify this deviation explicitly: because invisible tongue samples are zeroed by the kinematics pipeline, using the median over all bins would produce a zero threshold and collapse the requested `>= threshold` target to an all-ones label.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are aligned by interpolating `frameTimes - vidshift - goCue` onto the same 10 ms `taxis` used for neural data, so the derived speed trace is bin-for-bin aligned with neural activity.

ii.
```python
taxis = time + ADVANCE_MOVEMENT
...
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The notes say video alignment matched `findVideoOffset.m` and `loadMotionEnergy.m`, and that tongue traces were derived from those aligned DLC trajectories.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DeepLabCut positions for `top_paw` and `bottom_paw`.

ii.
```python
feat_indices = {
    ...
    "top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
    "bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
}
```

iii. The notes say paw speed was computed as the mean speed of `top_paw` and `bottom_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Bottom-view paw positions are interpolated onto the neural time axis, non-tongue NaNs are nearest-filled, coordinate velocities are computed by gradient, a baseline derivative is subtracted, nearest-fill is applied again to the velocity traces, speed is taken per paw point, and the two paw-point speeds are averaged.

ii.
```python
if "tongue" not in feat_name:
    xpos[:, tr] = fill_nearest_1d(xpos[:, tr])
    ypos[:, tr] = fill_nearest_1d(ypos[:, tr])
...
xvel[:, tr] = np.gradient(tsinterp[:, 0])
yvel[:, tr] = np.gradient(tsinterp[:, 1])
if "tongue" not in feat_name:
    xvel[:, tr] = xvel[:, tr] - basederiv[0]
    yvel[:, tr] = yvel[:, tr] - basederiv[0]
...
top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
```

iii. The notes justify this as following the repository’s kinematics path up to aligned continuous traces, then collapsing to a decoder-specific speed summary.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The threshold is the per-session 50th percentile over all exported paw-speed bins from included trials. Values below threshold are `0`; values at or above threshold are `1`.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_use, 50))
...
(paw_speed[:, tr] >= paw_thresh).astype(np.int16),
```

iii. The notes say `paw_velocity_bin` used the literal per-session 50th percentile requested by the decoder task.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw positions are interpolated from bottom-camera frame times after subtracting the video offset and go-cue time, onto the same `taxis` as the neural data. The derived paw-speed bins therefore share the neural time grid exactly.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The notes say paw velocities came from the same aligned DLC trajectories used by the repository kinematics code.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically `me.data`. Alignment uses side-camera `frameTimes` from the main HDF5 session file plus the go-cue times and video offset.

ii.
```python
def load_motion_energy(path: Path) -> tuple[np.ndarray, float]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
    return trial_arrays, float(me.moveThresh)
```

iii. The notes say motion energy followed the repository exactly up to the aligned continuous trace.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each trial’s motion-energy trace is interpolated onto the neural time axis using side-camera frame times corrected by `vidshift` and go cue, then edge/internal NaNs are nearest-filled. The script ignores the paper’s manual `moveThresh` for the exported decoder target and keeps the continuous aligned trace until the final median split.

ii.
```python
out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
out[:, tr] = fill_nearest_1d(out[:, tr])
```

iii. The notes justify this as matching `loadMotionEnergy.m` for alignment/interpolation, then applying decoder-specific discretization afterward.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The threshold is the per-session 50th percentile over all exported motion-energy bins from included trials. Values below threshold are `0`; values at or above threshold are `1`.

ii.
```python
me_thresh = float(np.nanpercentile(me_use, 50))
...
(motion_energy[:, tr] >= me_thresh).astype(np.int16),
```

iii. The notes say `motion_energy_bin` used the literal per-session 50th percentile requested by the decoder task, rather than the paper’s manual movement threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The trace is aligned with `frameTimes - vidshift - goCue` and interpolated to the same `taxis` as neural activity, producing a bin-matched time series.

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

iii. The notes explicitly cite `findVideoOffset.m` and `loadMotionEnergy.m` as the alignment reference.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data are handled differently by signal type. Motion energy and non-tongue DLC traces are nearest-filled after interpolation. Tongue positions are left missing through interpolation, but tongue velocity NaNs are set to zero so invisible tongue frames become zero speed. If a non-tongue trace has no valid samples at all, the helper `fill_nearest_1d` returns zeros. Trials with `NdroppedFrames` equal to NaN are skipped at the position-loading stage.

ii.
```python
if idx.size == 0:
    return np.zeros_like(x)
...
if np.isnan(ndropped):
    continue
...
if "tongue" not in feat_name:
    xpos[:, tr] = fill_nearest_1d(xpos[:, tr])
    ypos[:, tr] = fill_nearest_1d(ypos[:, tr])
...
else:
    xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0
    yvel[:, tr][np.isnan(yvel[:, tr])] = 0.0
```

iii. The notes justify the nearest-fill and tongue-zeroing as matching the repository’s motion-energy and kinematics behavior, and they call out the tongue-zero issue as the reason for the custom tongue threshold.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are the nested per-session/per-unit/per-trial spike binning and smoothing, plus the per-feature/per-trial video interpolation and velocity computation. The low-FR pass is especially costly because it computes condition PSTHs for every cluster before the single-trial export computes another full set of histograms.

ii.
```python
for clu_idx in range(qds.shape[0]):
    ...
    mean_fr = compute_psth_mean_fr(...)
    ...
    neural_by_trial.append(
        build_neural_trials(...)
    )

for feat_name, feat_idx in feat_indices.items():
    xpos, ypos = aligned_feature_position(...)
    velocities[feat_name] = aligned_feature_velocity(xpos, ypos, feat_name)
```

iii. The trajectory and notes do not analyze runtime in detail; this is inferred directly from the structure of `convert_data.py`.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are straightforward vectorization targets: column-wise convolution in `my_smooth`, per-trial interpolation/filling in `aligned_motion_energy` and `aligned_feature_position`, per-trial velocity computation in `aligned_feature_velocity`, per-condition PSTH construction in `compute_psth_mean_fr`, and per-trial spike histograms in `build_neural_trials`.

ii.
```python
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")

for tr in range(n_trials):
    ...

for tr in range(xpos.shape[1]):
    ...

for cond_mask in condition_masks:
    ...

for col, tr in enumerate(included_trials):
    ...
```

iii. No explicit justification was documented by the agent; this is an implementation-level observation from the final code.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several operations: it bins and smooths spikes once for low-FR filtering and again for exported single trials; it interpolates each DLC feature independently even when they share the same frame-time grid; and it repeatedly scans trial feature-name lists to locate feature indices.

ii.
```python
mean_fr = compute_psth_mean_fr(...)
...
build_neural_trials(...)

for feat_name, feat_idx in feat_indices.items():
    xpos, ypos = aligned_feature_position(...)
```

iii. The notes mention the low-FR filter and the aligned DLC derivations, but they do not discuss the repeated work explicitly.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads and stores several values that do not affect the exported decoder tensors: `sample` and `delay` are loaded but unused, `raw_move_thresh` is only reported in session sanity metadata, and `kept_cluster_indices`, `prefilter_unit_count`, and `cluster_ids` are accumulated but never used after assignment. It also keeps constants like `SIDE_FEATURES` and `BOTTOM_FEATURES` that are not consumed anywhere.

ii.
```python
SIDE_FEATURES = [...]
BOTTOM_FEATURES = [...]
...
"sample": read_h5_vector(ev_group["sample"]),
"delay": read_h5_vector(ev_group["delay"]),
...
kept_cluster_indices = []
prefilter_unit_count = 0
cluster_ids = []
...
"raw_motion_energy_threshold": float(raw_move_thresh),
```

iii. The notes only use `raw_motion_energy_threshold` for sanity reporting; the rest of these discarded computations are evident from the final script rather than from an explicit written justification.
