# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from HDF5-format `.mat` files located in `data/Ephys_Behavior/` using `h5py`. Each session has a `data_structure_{animal}_{date}.mat` file for neural/behavioral data and a `motionEnergy_{animal}_{date}.mat` file for motion energy (loaded via `scipy.io.loadmat`). A hardcoded roster of 12 sessions (`SESSION_SPECS`) defines which files to load, matching the Figure 8 context-task analysis pipeline from the reference MATLAB code.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    # ... 12 sessions total
]

def load_session(spec: SessionSpec) -> dict:
    with h5py.File(spec.data_path, "r") as f:
        bp_group = f["obj/bp"]
        # ... reads behavioral, neural, and video data
    raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
```

iii. The AI states in CONVERSION_NOTES.md: "The session roster was taken from the reference MATLAB context-analysis pipeline in `code/Scripts/Figure 8/Figure8a_thru_c.m` plus the per-animal loader files in `code/DataLoadingScripts/Recording and video/`."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `animal` field in each `SessionSpec`. Unique subject names are collected in order across sessions using `OrderedDict.fromkeys`, and each session is mapped to a subject index.

ii.
```python
subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
```

iii. The AI preserved the subject IDs present in the data files (7 unique: JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19), noting that the paper reports 6 mice for the context cohort. The AI chose not to collapse or rename subjects without evidence.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_SPECS` defines one session by animal name, date, and probe number. The `load_session` function processes one session at a time, and `build_dataset` iterates over all 12 specs.

ii.
```python
def build_dataset() -> dict:
    sessions = [load_session(spec) for spec in SESSION_SPECS]
```

iii. The roster matches the Figure 8 script which loads sessions via `loadJEB6_ALMVideo`, `loadJEB7_ALMVideo`, etc. The AI verified 12 sessions matching the paper's reported count.

## 1-d. How are the data split into trials?

i. Each session's total trial count comes from `obj/bp/Ntrials` in the HDF5 file. Neural spike data is organized per-cluster with trial indices (`clu.trial`) and spike times (`clu.trialtm`). Video data is organized per-trial with frame times and tracking coordinates.

ii.
```python
bp = {
    "Ntrials": int(round(read_h5_scalar(bp_group["Ntrials"]))),
    # ...
}
session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
```

iii. Trial splitting follows the data structure directly: each HDF5 file contains all trials for a session, and the AI reads the total count and per-trial fields.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by `trial_selector`: only `hit` or `miss` trials are kept, excluding `early` lick trials and trials with stimulation enabled (`stim.enable`). Ignore/no-response trials are excluded.

ii.
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)
```

iii. The AI states: "Retained hit and miss trials with stim.enable == 0 and early == 0; ignore/no trials excluded." This follows the paper's methods: "early lick and ignore trials...were omitted from all analyses."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times stored in each cluster's `trialtm` (spike times relative to trial start) and `trial` (trial index) fields in the `obj/clu` HDF5 group for the specified probe.

ii.
```python
clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
```

iii. This matches the reference code's `alignSpikes.m` which accesses `obj.clu{prbnum}(clu).trialtm` and `obj.clu{prbnum}(clu).trial`.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, then binned into 10 ms bins (edges from -3.0 to 2.5 s). Binned spike counts are converted to firing rates (counts/dt) and smoothed with a causal Gaussian kernel (window size 15, reflect boundary condition).

ii.
```python
def build_neural_trials(...):
    counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
    out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. This matches the reference `getSeq.m`: `obj.trialdat{prbnum}(:,i,j) = mySmooth(N./params.dt, params.smooth, params.bctype)` with `bctype='reflect'` from Figure 8.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded. (2) Remaining clusters with mean firing rate <= 1 Hz (averaged across PSTH conditions and time) are excluded.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}

quality = read_h5_string(f, h5_ref_at(qds, clu_idx)).strip().lower()
if quality in BAD_QUALITIES:
    continue

mean_fr = compute_psth_mean_fr(...)
if mean_fr <= LOW_FR:
    continue
```

iii. Quality filtering matches `findClusters.m` with `{'all'}` (excludes garbage, gabrga, noisy, real?). Low-FR threshold of 1 Hz matches Figure 8's `params.lowFR = 1`. The mean FR computation matches `removeLowFRClusters.m`: `meanFRs = mean(mean(obj.psth{prbnum},3,'omitnan'),'omitnan')`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time for each spike's trial: `aligned = trialtm - goCue[trial]`. Spikes are then histogrammed into time bins relative to the go cue.

ii.
```python
ALIGN_EVENT = "goCue"
aligned = trialtm - align_times[session_trial]
counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
```

iii. Matches the reference `alignSpikes.m`: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event` and the instruction to align to "Go cue onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (dt = 1/100 s), producing 550 time bins over the [-3.0, 2.5] s window. No temporal rebinning is applied; spikes are binned directly at 10 ms resolution.

ii.
```python
DT = 1 / 100
TMIN = -3.0
TMAX = 2.5
edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
```

iii. Matches Figure 8a_thru_c.m: `params.dt = 1/100; params.tmin = -3; params.tmax = 2.5`. The metadata records `time_bin_size: 10.0` ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time bin edges, not from a raw data variable. It represents the center of each time bin relative to the go cue onset.

ii.
```python
edges, time = build_edges_and_time()
# time = edges[:-1] + DT / 2  (bin centers)
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The time axis is constructed from the alignment parameters (TMIN, TMAX, DT) and represents seconds from go cue onset, matching the instruction for "Time from go cue onset in seconds."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin edges are generated from TMIN to TMAX with step DT. Bin centers are computed as `edges[:-1] + DT/2`. The same time vector is replicated for every trial with shape `(1, 550)`.

ii.
```python
def build_edges_and_time():
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
```

iii. No complex processing is needed; the time axis is a direct arithmetic construction from the alignment parameters.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector uses the same bin centers as the neural data, so they are inherently aligned. Both share the same 550-timepoint axis.

ii.
```python
session_input = [
    np.asarray(time[None, :], dtype=np.float32)  # same 'time' as neural bins
    for _ in range(neural_arr.shape[2])
]
```

iii. Since both neural and input data use the same time axis constructed from the same edges, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from two fields: `bp.hit` (whether the trial was correct) and `bp.R` (whether the trial was a right-instruction trial).

ii.
```python
def lick_direction_from_trial(bp, trial_idx):
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
    if hit:
        return int(is_right_trial)
    return int(not is_right_trial)
```

iii. The AI states: "lick_direction: actual response direction, not instructed side." On correct (hit) trials, the animal licked the instructed side. On incorrect (miss) trials, the animal licked the opposite side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For hit trials, lick direction equals the instructed side (R=1→right, L=0→left). For miss trials, lick direction is the opposite of the instructed side. The result is broadcast to all 550 time bins as a per-trial constant.

ii.
```python
np.full(tongue_speed.shape[0], lick_dir, dtype=np.int16)
```

iii. The AI's logic: "Correct right trials and incorrect left trials were labeled right. Correct left trials and incorrect right trials were labeled left." This captures the actual animal response direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `bp.autowater` field.

ii.
```python
context = 0 if bp["autowater"][tr] else 1
```

iii. In the reference code, `autowater=1` corresponds to WC (water-cued) trials and `autowater=0` to DR (delayed-response) trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=true → WC=0, autowater=false → DR=1. The value is broadcast to all time bins as a per-trial constant.

ii.
```python
context = 0 if bp["autowater"][tr] else 1
np.full(tongue_speed.shape[0], context, dtype=np.int16)
```

iii. Matches the instruction specification: "WC = 0, DR = 1."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `bp.hit` field.

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
```

iii. Since only hit and miss trials are included (no ignore trials), `hit=1` maps to correct and `hit=0` (i.e., miss) maps to incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: hit → correct=1, miss → incorrect=0. Broadcast to all time bins as a per-trial constant.

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
np.full(tongue_speed.shape[0], outcome, dtype=np.int16)
```

iii. Matches the instruction specification: "incorrect = 0, correct = 1."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `top_tongue` and `bottom_tongue` DLC tracking features from the bottom camera view (`obj/traj`, view index 1). The raw data are x,y position time series stored per-trial in `traj.ts` with associated `frameTimes`.

ii.
```python
feat_indices = {
    "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
    "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
}
```

iii. The AI chose `top_tongue` and `bottom_tongue` from the bottom view to represent the tongue tip, consistent with the reference code's use of these features.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps: (1) Interpolate x,y positions from video frame times to the neural time axis using video offset correction. (2) Compute velocity as `np.gradient` of position (no fillmissing for tongue). (3) Set NaN velocities to 0. (4) Average x-velocities and y-velocities of top_tongue and bottom_tongue. (5) Compute speed as `sqrt(vx^2 + vy^2)`. (6) Replace remaining NaN with 0.

ii.
```python
top_tongue_xvel, top_tongue_yvel = velocities["top_tongue"]
bottom_tongue_xvel, bottom_tongue_yvel = velocities["bottom_tongue"]
tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
```

iii. The AI states: "Tongue speed was computed from the bottom-view tongue-tip velocity using the average of top_tongue and bottom_tongue." The velocity computation and NaN→0 handling matches `findVelocity.m`. The averaging of two features is the AI's own construction to produce a single "tongue speed."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session thresholding at the 50th percentile of *positive* tongue speed values across all included trial bins. Values >= threshold → 1 (high), values < threshold → 0 (low).

ii.
```python
positive_tongue = tongue_use[tongue_use > 0]
if positive_tongue.size:
    tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
else:
    tongue_thresh = 0.0
```

iii. The AI justifies using positive-only values: "Using the median over all bins collapses the threshold to zero in every session" because tongue velocity is zero when the tongue is not visible. "Restricting the threshold computation to positive tongue-speed bins preserves the reference-aligned trace while yielding a usable categorical target."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are corrected with the video offset (`vidshift = mode(bitcode.bitstart)/fs - mode(bp.ev.bitStart)`) and aligned to the go cue. Positions are interpolated onto the neural time axis (`taxis = time + advance_movement` where `advance_movement=0`).

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. Matches `findPosition.m`: `tsinterp = interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)` and `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `top_paw` and `bottom_paw` DLC tracking features from the bottom camera view.

ii.
```python
"top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
"bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
```

iii. The paper methods state "paws were tracked using only the bottom view," consistent with the AI's use of bottom-view paw features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing steps: (1) Interpolate x,y positions to neural time axis with video offset correction. (2) Fill NaN positions with nearest value (non-tongue features). (3) Compute velocity via `np.gradient`, subtract baseline derivative (`nanmedian(diff)`). (4) Fill NaN velocities with nearest value. (5) Compute speed for each paw as `sqrt(vx^2 + vy^2)`. (6) Average top_paw and bottom_paw speeds. (7) Replace NaN with 0.

ii.
```python
top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The velocity computation (gradient, baseline subtraction, fillmissing for non-tongue) matches `findVelocity.m`. Averaging top_paw and bottom_paw is the AI's construction to produce a single "paw speed."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session thresholding at the 50th percentile of all paw speed values across all included trial bins. Values >= threshold → 1, values < threshold → 0.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_use, 50))
(paw_speed[:, tr] >= paw_thresh).astype(np.int16)
```

iii. Matches the instruction: "50th percentile" per session, "0: < 50th percentile, 1: >= 50th percentile."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue velocity: video offset correction and interpolation onto the neural time axis.

ii.
```python
xpos, ypos = aligned_feature_position(f=f, view_group=bottom_group, ...,
    align_times=bp[ALIGN_EVENT], vidshift=vidshift, taxis=taxis)
```

iii. Matches the reference `findPosition.m` alignment approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_{animal}_{date}.mat` files. The raw data is in `me.data` (a cell array of per-trial motion energy vectors) and `me.moveThresh` (per-session movement threshold, not used for the decoder).

ii.
```python
def load_motion_energy(path: Path) -> tuple[np.ndarray, float]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
    return trial_arrays, float(me.moveThresh)
```

iii. Matches the reference `loadMotionEnergy.m` which loads the same files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy vectors are interpolated from video frame times to the neural time axis using the video offset correction, then edge NaNs are filled with the nearest value.

ii.
```python
def aligned_motion_energy(...):
    for tr in range(n_trials):
        out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
        out[:, tr] = fill_nearest_1d(out[:, tr])
```

iii. Matches `loadMotionEnergy.m`: `me.newdata(:,trix) = interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)` followed by `me.data = fillmissing(me.data,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session thresholding at the 50th percentile of all motion energy values across all included trial bins. Values >= threshold → 1, values < threshold → 0.

ii.
```python
me_thresh = float(np.nanpercentile(me_use, 50))
(motion_energy[:, tr] >= me_thresh).astype(np.int16)
```

iii. Matches the instruction: "50th percentile" per session.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the same video offset and go cue alignment as other video features, interpolated onto the neural time axis.

ii.
```python
out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
```

iii. Matches `loadMotionEnergy.m`: `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) For non-tongue video features, NaN positions are filled with nearest value (`fill_nearest_1d`), matching MATLAB `fillmissing('nearest')`. (2) For tongue features, NaN positions are left as NaN; NaN velocities are set to 0. (3) For motion energy, NaN values are filled with nearest value after interpolation. (4) Dropped-frame trials (NdroppedFrames = NaN) are skipped for video feature extraction. (5) Sessions with fewer than 2 usable trials or no surviving units raise errors. (6) The `interp_with_nan` function skips NaN x/y values during interpolation.

ii.
```python
if "tongue" not in feat_name:
    xpos[:, tr] = fill_nearest_1d(xpos[:, tr])
    ypos[:, tr] = fill_nearest_1d(ypos[:, tr])
# ...
if "tongue" not in feat_name:
    xvel[:, tr] = xvel[:, tr] - basederiv[0]
    # ...
    xvel[:, tr] = fill_nearest_1d(xvel[:, tr])
else:
    xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0
```

iii. The handling is consistent with the reference code's approach in `findPosition.m` and `findVelocity.m`.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Reading and parsing HDF5 files, especially dereferencing cell arrays of references for feature names, frame times, and tracking data. (2) Per-unit spike processing: loading spike times, computing PSTHs for 7 conditions (for low-FR filtering), and building single-trial firing rates. (3) Per-trial video feature alignment: interpolating DLC tracking data from video frame times to the neural time axis for each feature and trial.

ii.
```python
for clu_idx in range(qds.shape[0]):  # iterate all clusters
    # read quality, load spikes, compute PSTH mean FR, build neural trials
    session_trial, _trialtm, aligned = load_trial_spikes(...)
    mean_fr = compute_psth_mean_fr(...)
    neural_by_trial.append(build_neural_trials(...))
```

iii. The nested loops over clusters, conditions, and trials dominate runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The per-trial spike histogramming in `build_neural_trials` iterates trial-by-trial; batch histogram computation could be faster. (2) The per-column convolution in `my_smooth` loops over columns. (3) The per-trial video interpolation in `aligned_feature_position` and `aligned_motion_energy`. (4) The per-condition PSTH computation in `compute_psth_mean_fr`.

ii.
```python
for col, tr in enumerate(included_trials):  # per-trial histogram
    counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
    out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")

for j in range(x_filt.shape[1]):  # per-column convolution
    out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
```

iii. These loops follow the structure of the MATLAB reference code, which also processes trial-by-trial and column-by-column.

## 11-c. What processing does the code repeat multiple times?

i. (1) Feature name lookups (`find_feat_index`) scan trials repeatedly to find the same feature indices. (2) Video offset (`vidshift`) is computed once per session but frame times are read in multiple places (once for motion energy, again for each DLC feature). (3) The `read_h5_string` helper is called for every cluster quality label individually. (4) Condition masks are computed once but PSTH computation iterates all conditions for every single unit.

ii.
```python
# Frame times read for motion energy
frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
# Frame times read again per trial in aligned_feature_position
frame_times = read_frame_times(f, view_group, tr)
```

iii. The repeated frame time reads are not strictly necessary; they could be cached.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `compute_condition_masks` function creates 7 condition masks that are only used for low-FR filtering PSTHs, not for the final output. (2) Video features (tongue, paw) are computed for ALL trials in the session, but only the included trials are used in the output. (3) The `motion_energy_path` loads `moveThresh` which is stored in sanity metadata but not used for thresholding (the decoder uses 50th percentile instead). (4) The `compute_condition_masks` includes conditions with and without early-trial filtering, but early trials are excluded from the final dataset.

ii.
```python
# Condition masks computed but only used for low-FR filtering
condition_masks = compute_condition_masks(bp)
# Motion energy computed for all trials
motion_energy = aligned_motion_energy(..., align_times=bp[ALIGN_EVENT], ...)
# Only included trials used in output
tongue_use = tongue_speed[:, included_trials]
```

iii. Processing all trials for video features before filtering is inherited from the reference code's structure, where video and neural processing happen in separate stages.
