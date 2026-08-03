# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the full paper/reference dataset. It hard-codes a 12-session two-context subset in `CONTEXT_SESSION_SPECS`, reads each session from `data/Ephys_Behavior/data_structure_<session>.mat` with `mat73.loadmat`, and reads motion energy from a paired `motionEnergy_<session>.mat` with `scipy.io.loadmat`. Trials are then taken from the loaded session object rather than from a separate session-discovery pass.

ii. 
```python
CONTEXT_SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 1),
    ...
    SessionSpec("JEB19", "2023-04-18", 0),
]
```

```python
@property
def data_path(self) -> Path:
    return Path("data/Ephys_Behavior") / f"data_structure_{self.session_id}.mat"

@property
def motion_energy_path(self) -> Path:
    return Path("data/Ephys_Behavior") / f"motionEnergy_{self.session_id}.mat"
```

```python
obj = mat73.loadmat(spec.data_path)["obj"]
me = load_motion_energy(spec)
```

iii. The notes say the AI intentionally restricted conversion to the paper’s “two-context ALM ephys subset” and specifically to the “12-session context-analysis loader set,” because context varies there and behavior-only sessions lack neural data.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is taken from the `animal` field in each hard-coded `SessionSpec`. At dataset assembly, unique subject names are sorted into `subjects`, and each session gets a `subject_idx` pointing into that list.

ii. 
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe_index: int
```

```python
result = {
    "session_id": spec.session_id,
    "subject": spec.animal,
    ...
}
```

```python
subjects = sorted({sess["subject"] for sess in converted_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in converted_sessions], dtype=np.int64),
```

iii. The justification is implicit in the code and metadata: sessions are named by animal and date, and the AI chose to carry the animal id from the session specification directly rather than rely on possibly inconsistent metadata fields inside the MATLAB files.

## 1-c. How are the data split into sessions?

i. One `SessionSpec` corresponds to one session. The AI treats each hard-coded `<animal, date, probe_index>` entry as a separate session and converts only those 12 fixed-delay two-context sessions from `Ephys_Behavior`.

ii. 
```python
session_specs = CONTEXT_SESSION_SPECS[:2] if use_sample else CONTEXT_SESSION_SPECS
...
for sess_idx, spec in enumerate(session_specs):
    make_plot = args.show_processing and sess_idx < 2
    converted_sessions.append(convert_session(spec, make_plot=make_plot))
```

```python
"session_subset": "Figure 8 two-context ALM session list from reference code",
```

iii. The notes explicitly justify this as using the figure-specific two-context ALM subset instead of the full raw archive, because the requested context output requires sessions containing both DR and WC trials.

## 1-d. How are the data split into trials?

i. Trials are treated as trial indices into `obj["bp"]` and related per-trial arrays. The AI first builds a boolean mask over all trials, converts that to trial indices with `np.flatnonzero`, and then iterates those trial indices when computing labels and per-trial arrays.

ii. 
```python
def select_valid_trials(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    ...
    valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
    return np.flatnonzero(valid)
```

```python
valid_trials = select_valid_trials(obj)
...
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
    ...
    kept_trials.append(int(trial_idx))
```

iii. The justification is mostly implicit. The notes say the raw session object stores behavior, trajectories, and spikes in per-trial structures, so trial identity can be carried by the original trial index without reconstructing boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early-lick trials, ignore/no-response trials, and photostimulation trials, and it keeps only hit or miss trials. It then applies an additional filter by discarding trials that do not have a detectable first lick after the alignment event.

ii. 
```python
early = ensure_1d_numeric(bp["early"]) != 0
no = ensure_1d_numeric(bp["no"]) != 0
hit = ensure_1d_numeric(bp["hit"]) != 0
miss = ensure_1d_numeric(bp["miss"]) != 0
stim_enable = ensure_1d_numeric(bp["stim"]["enable"]) != 0
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
```

```python
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
    if lick_dir is None:
        continue
```

iii. The notes justify this as matching the paper’s omission of early and ignore trials and keeping outcome well-defined. The lick-direction drop is justified there as using “actual first post-alignment lick side” rather than instructed side.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the selected probe’s cluster structure, specifically per-unit `trialtm` and `trial` arrays from `obj["clu"][spec.probe_index]`, together with `bp.ev.goCue` for alignment. Unit `quality` labels are also used for curation.

ii. 
```python
clu = obj["clu"][spec.probe_index]
...
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
```

```python
mean_fr = mean_firing_rate_window(
    clu["trialtm"][unit_idx],
    np.asarray(clu["trial"][unit_idx], dtype=np.int64),
    align_times,
)
```

```python
trialdat = bin_unit_spikes(
    clu["trialtm"][unit_idx],
    np.asarray(clu["trial"][unit_idx], dtype=np.int64),
    align_times,
    kept_trials,
)
```

iii. The notes say the AI followed the ephys path from the reference code: align spikes to `goCue`, bin them on a common grid, then smooth and curate units by quality and firing rate.

## 2-b. How is the `neural` data processed?

i. The AI bins aligned spikes into 5 ms bins from `TMIN` to `TMAX`, converts counts to firing rates by dividing by `DT`, and smooths those rates with a custom one-sided Gaussian-like kernel (`my_smooth`) using `SMOOTH_N = 15` and `BCTYPE = "reflect"`.

ii. 
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
SMOOTH_N = 15
BCTYPE = "reflect"
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
```

```python
aligned_counts = np.zeros((kept_trials_0based.size, TIME_AXIS.size), dtype=np.float64)
np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
rates = aligned_counts / DT
rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
```

iii. The notes justify this as using “smoothed single-trial firing rates” because the paper’s decoders operate on `obj.trialdat`. They also cite 5 ms base binning and reference-style smoothing as the intended preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units whose `quality` string is not exactly one of `{"garbage", "gabrga", "noisy", "real?"}` and whose mean firing rate over the full `[-2.5, 2.5]` window exceeds `1 Hz`. It does not lowercase the quality label before matching, and it does not exclude `poor`.

ii. 
```python
LOW_FR_HZ = 1.0
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
```

```python
def keep_quality(quality: str) -> bool:
    if quality is None:
        quality = ""
    quality = str(quality).strip()
    return quality not in QUALITY_EXCLUDE
```

```python
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. The notes justify the 1 Hz threshold as matching the paper rather than the looser default helper threshold. They also state that junk-quality units should be excluded, but the exact drop list implemented in code is the four-value set above.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each spike’s trial-specific `goCue` time from the raw `trialtm` value, so neural time is expressed relative to `bp.ev.goCue`.

ii. 
```python
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
```

```python
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
ALIGN_EVENT = "goCue"
```

iii. The notes explicitly justify a universal `goCue` alignment, including WC trials, because the stored WC `goCue` values were interpreted as a goCue-equivalent water-presentation event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 5 ms bins over a `[-2.5, 2.5)` s window. No later temporal rebinning is applied; the stored neural matrices remain on that 5 ms grid.

ii. 
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
```

```python
"time_bin_size": 5.0,
```

iii. The notes justify 5 ms as the paper-consistent base bin size and say any decoder-ready outputs should share that same common time base.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw per-trial variable. The AI defines it from the chosen alignment convention and the fixed binning constants `TMIN`, `TMAX`, and `DT`, producing a common `TIME_AXIS` relative to go cue.

ii. 
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
```

```python
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The notes justify this as the single decoder input requested by the user, repeated for every trial on the same time grid as the neural data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI simply constructs the bin centers once and reuses them for every trial; there is no additional per-trial transformation.

ii. 
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
...
time_input = TIME_AXIS[None, :].astype(np.float32)
session_input.append(time_input)
```

iii. The notes describe this as a uniform decoder input that keeps all trials on the same `(1, n_timepoints)` format.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `TIME_AXIS` used for the input is also the grid used to bin neural spikes, so the input axis and neural axis are exactly shared.

ii. 
```python
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The justification is implicit in the implementation and also stated in the notes: all streams were intended to live on one common aligned time base.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the raw lick event times `bp.ev.lickL` and `bp.ev.lickR`, not from the instructed side fields. For each trial it looks at lick events occurring at or after `goCue`.

ii. 
```python
def get_first_lick_direction(obj: dict, trial_idx: int, go_time: float) -> int | None:
    lick_l = event_list_to_array(obj["bp"]["ev"]["lickL"][trial_idx])
    lick_r = event_list_to_array(obj["bp"]["ev"]["lickR"][trial_idx])
    lick_l = lick_l[lick_l >= go_time]
    lick_r = lick_r[lick_r >= go_time]
```

iii. The notes explicitly justify this as using the “actual first post-alignment lick side from `lickL` / `lickR`” rather than the instructed side, because the prompt asked for lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI finds the first post-go-cue lick on the left and right ports and assigns `0` for left if the left lick comes first, otherwise `1` for right. Trials with no post-go-cue lick are discarded entirely, so there is no “no lick” category.

ii. 
```python
first_l = lick_l[0] if lick_l.size else math.inf
first_r = lick_r[0] if lick_r.size else math.inf
if math.isinf(first_l) and math.isinf(first_r):
    return None
return 0 if first_l < first_r else 1
```

```python
if lick_dir is None:
    continue
```

iii. The notes justify this as making lick direction behaviorally literal and compatible with the requested binary output.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp.autowater` flag.

ii. 
```python
autowater = ensure_1d_numeric(bp["autowater"])
```

iii. The notes explicitly map `autowater` to context, matching the paper/code distinction between WC and DR trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater != 0` as `WC = 0` and `autowater == 0` as `DR = 1`, then repeats that label across all time bins of the trial.

ii. 
```python
trial_labels.append(
    (
        lick_dir,
        0 if autowater[trial_idx] != 0 else 1,
        1 if hit[trial_idx] != 0 else 0,
    )
)
```

```python
np.full(TIME_AXIS.size, context_label, dtype=np.int64),
```

iii. The notes justify this as the exact context variable used throughout the paper and reference code, with the prompt’s required coding `WC = 0`, `DR = 1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes from the hit/miss behavioral flags. The valid-trial selection requires `(hit | miss)`, and the final label uses `hit` to distinguish correct from incorrect among those kept trials.

ii. 
```python
hit = ensure_1d_numeric(bp["hit"]) != 0
miss = ensure_1d_numeric(bp["miss"]) != 0
...
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
```

```python
1 if hit[trial_idx] != 0 else 0,
```

iii. The notes justify this as keeping only trials with well-defined outcome labels and excluding ignore/no-response trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI encodes outcome as binary: `1` for hit/correct and `0` otherwise within the kept hit-or-miss trial set. Ignore trials are removed instead of being given their own category.

ii. 
```python
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
```

```python
trial_labels.append(
    (
        lick_dir,
        0 if autowater[trial_idx] != 0 else 1,
        1 if hit[trial_idx] != 0 else 0,
    )
)
```

iii. The notes explicitly justify dropping ignore trials so the output matches the prompt’s binary incorrect/correct specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera trajectory `obj["traj"][0]` for the feature named `"tongue"`, plus `frameTimes`, the bitcode-based video offset, and `bp.ev.goCue` for temporal alignment. The bottom-camera tongue feature is not used in the final code.

ii. 
```python
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
```

```python
view_dict = obj["traj"][view_index]
feat_index = find_feature_index(view_dict, feat_name)
...
frame_times = trial.get("frameTimes")
ts = trial.get("ts")
```

iii. The notes justify extracting only the kinematic features needed for the requested outputs. Earlier planning notes mentioned combining tongue information, but the final code only uses the side-view `"tongue"` feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI interpolates tongue x/y positions directly onto the 5 ms neural time grid, computes per-bin velocity by `np.gradient` on those interpolated coordinates, takes the speed magnitude, and never explicitly applies a DLC likelihood threshold or the reference’s within-run smoothing. Visibility is inferred afterward from whether interpolated x/y are finite.

ii. 
```python
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
ypos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 1], left=np.nan, right=np.nan)
```

```python
tongue_visible = np.isfinite(tongue_x) & np.isfinite(tongue_y)
tongue_vx, tongue_vy = compute_velocity(tongue_x, tongue_y, "tongue")
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

```python
else:
    xv = np.where(np.isfinite(xv), xv, 0.0)
    yv = np.where(np.isfinite(yv), yv, 0.0)
```

iii. The notes justify the overall approach as producing a scalar tongue speed and then median-thresholding it per session. They also document a later fix: the tongue threshold is computed only from visible time points because invisible periods otherwise dominated the median.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes a per-session 50th percentile using only visible tongue-speed values from kept trials, then emits a binary trace where a bin is `1` if the tongue speed is at or above that threshold and the tongue is visible, otherwise `0`.

ii. 
```python
tongue_visible_kept = tongue_visible[:, kept_trials]
if np.any(tongue_visible_kept):
    tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
else:
    tongue_threshold = 0.0
```

```python
tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. The notes explicitly justify the visible-only median as a repair for a degenerate threshold caused by long invisible periods, and they say invisible bins are intentionally pushed into the low bin instead of receiving a third class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a per-session video clock offset from bitcode, subtracts that offset and the per-trial `goCue` from each frame time, and then interpolates positions to the same 5 ms `TIME_AXIS` used by the neural data.

ii. 
```python
def compute_vidshift(obj: dict) -> float:
    bit_start = mode_value(obj["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_value(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
    return float(vid_file_offset - bit_start)
```

```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
...
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

iii. The notes justify this as matching the reference’s video-offset idea while keeping all streams on one shared goCue-centered time base.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera tracked features, `"top_paw"` and `"bottom_paw"`, from `obj["traj"][1]`, again using `frameTimes`, the bitcode-derived video offset, and `bp.ev.goCue`.

ii. 
```python
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
    paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
    paw_speeds.append(np.sqrt(paw_vx**2 + paw_vy**2))
```

iii. The notes justify this as extracting only requested paw features, and the planning notes specifically say the AI planned to average speed magnitude across `top_paw` and `bottom_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw feature is aligned by interpolation to the neural grid, converted to x/y velocity with `np.gradient`, reduced to speed magnitude, and then the two paw speeds are averaged wherever finite. For non-tongue features, missing interpolated positions and velocities are nearest-filled.

ii. 
```python
if "tongue" not in feat_name:
    xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])
    ypos[:, trial_idx] = fill_nearest_1d(ypos[:, trial_idx])
```

```python
if "tongue" not in feat_name:
    xv = xv - basederiv[0]
    yv = yv - basederiv[0]
    xv = fill_nearest_1d(xv)
    yv = fill_nearest_1d(yv)
```

```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_counts = np.sum(np.isfinite(paw_stack), axis=0)
paw_speed = np.full(paw_counts.shape, np.nan, dtype=np.float64)
np.divide(np.nansum(paw_stack, axis=0), paw_counts, out=paw_speed, where=paw_counts > 0)
```

iii. The notes justify the scalar-speed reduction and averaging of the two paw features as a way to produce the single requested paw-velocity output stream.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI computes a session-wide median over the aligned paw-speed array and then labels each time bin as `0` or `1` according to whether it falls below or at/above that threshold. Because it uses a raw comparison, bins with `NaN` paw speed become `0`.

ii. 
```python
paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
```

```python
(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64),
```

iii. The notes justify median splitting as following the prompt’s required discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw tracking is aligned exactly like tongue tracking: subtract the per-session video offset and the trial’s `goCue`, then interpolate onto the same 5 ms grid as the neural data.

ii. 
```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
```

```python
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
```

iii. The justification is the same shared-time-base argument used throughout the notes.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the companion `motionEnergy_<session>.mat` file, specifically from `me.data`, and paired to session video timing using side-camera `frameTimes` from `obj["traj"][0]`.

ii. 
```python
def load_motion_energy(spec: SessionSpec) -> dict:
    me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
    me = me_mat["me"]
    return {
        "data": np.atleast_1d(me.data),
        "moveThresh": float(me.moveThresh),
    }
```

```python
view_dict = obj["traj"][0]
...
me_trial = np.asarray(me["data"][trial_idx], dtype=np.float64).reshape(-1)
```

iii. The notes justify this as using the aligned motion-energy stream from the companion file rather than recomputing motion energy from raw video frames.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligns each trial’s motion-energy trace to go cue using video timing, interpolates it to the 5 ms neural grid, nearest-fills missing bins, and then applies a session-median threshold.

ii. 
```python
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

iii. The notes justify the interpolation step as putting motion energy on the same neural time axis; the median split is justified as following the prompt.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes a session-wide 50th percentile over the aligned motion-energy array and outputs a binary trace where each bin is `1` if it is at or above that threshold and `0` otherwise.

ii. 
```python
me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))
```

```python
(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64),
```

iii. The notes justify this directly from the user instruction that motion energy be discretized by per-session median split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the bitcode-derived video offset and the trial’s `goCue`, then the motion-energy trace is interpolated onto the neural `TIME_AXIS`.

ii. 
```python
vidshift = compute_vidshift(obj)
view_dict = obj["traj"][0]
...
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
```

iii. The notes and validation log justify this as a direct raw-data alignment check that matched the AI’s own reconstruction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed video data are usually repaired rather than left missing. If `frameTimes` are missing or all `NaN`, the AI synthesizes them as `(1..N)/400`. For non-tongue features it nearest-fills missing aligned positions and velocities. Motion energy is also nearest-filled after interpolation. Trials with no post-go-cue lick are dropped rather than retained with a special label.

ii. 
```python
if frame_times is None:
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
...
if frame_times.size == 0 or np.all(np.isnan(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
```

```python
if "tongue" not in feat_name:
    xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])
    ypos[:, trial_idx] = fill_nearest_1d(ypos[:, trial_idx])
```

```python
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

```python
if lick_dir is None:
    continue
```

iii. The notes justify some of this as keeping the decoder outputs binary and usable, especially for tongue thresholding, but they do not present a strong paper-based justification for nearest-filling or synthetic frame times.

## 11-a. What are the most time-consuming steps of the code?

i. The likely hotspots in the AI code are session loading with `mat73`, repeated per-trial interpolation of video and motion-energy traces onto `TIME_AXIS`, per-trial velocity computation, and the per-unit spike-binning loop. Optional plot generation is also extra work when enabled.

ii. 
```python
obj = mat73.loadmat(spec.data_path)["obj"]
```

```python
for trial_idx in range(n_trials):
    ...
    xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

```python
for unit_idx, use_unit in enumerate(quality_keep):
    ...
    trialdat = bin_unit_spikes(...)
```

iii. The notes mention `np.add.at` spike binning as a speed-up and estimate about `~7.4 s/session`, but they do not explicitly analyze hotspots beyond that.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be reduced or restructured: the per-trial interpolation loop in `align_feature_positions`, the per-trial loop in `compute_velocity`, the per-unit firing-rate loop in `convert_session`, the per-feature paw loop, and the final per-trial assembly loop over outputs.

ii. 
```python
for trial_idx in range(n_trials):
    ...
```

```python
for trial_idx in range(xpos.shape[1]):
    ...
```

```python
for unit_idx, use_unit in enumerate(quality_keep):
    ...
```

```python
for local_trial_idx, trial_idx in enumerate(kept_trials):
    ...
```

iii. The notes explicitly mention only one optimization here, “Vectorized spike binning with `np.add.at`,” implying the rest of these loops were left as-is.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several things: `compute_vidshift(obj)` is called once before feature alignment and again inside `align_motion_energy`; each feature is aligned separately with its own full per-trial interpolation loop; and the final output arrays repeat constant per-trial labels across all 1000 bins.

ii. 
```python
vidshift = compute_vidshift(obj)
...
motion_energy = align_motion_energy(obj, me, align_times, TIME_AXIS)
```

```python
def align_motion_energy(...):
    vidshift = compute_vidshift(obj)
```

```python
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(...)
```

```python
np.full(TIME_AXIS.size, lick_dir, dtype=np.int64),
np.full(TIME_AXIS.size, context_label, dtype=np.int64),
np.full(TIME_AXIS.size, outcome_label, dtype=np.int64),
```

iii. The notes do not present these as redundant, though they do justify repeating static labels across time so every output shares the same `(n_output, n_timepoints)` structure.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI code includes several computations or objects that are not needed for the final converted dataset: optional matplotlib plotting machinery, an unused `fill_nearest_2d` helper, an unused `EDGES` constant, an unused `kept_unit_indices` list, and per-session `thresholds`/`summary` dictionaries that are stored in intermediate results but not included in the final dataset. It also computes both `top_paw` and `bottom_paw` streams even though the target format only needs one paw-velocity output.

ii. 
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

```python
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
...
def fill_nearest_2d(x: np.ndarray) -> np.ndarray:
```

```python
kept_unit_indices = []
...
kept_unit_indices.append(unit_idx)
```

```python
result = {
    ...
    "thresholds": {...},
    "summary": {...},
}
```

iii. The notes justify plotting and summaries as validation aids, but they are not part of the downstream decoder input file.
