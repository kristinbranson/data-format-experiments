# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 12-session Figure 8 ALM cohort, all from `Ephys_Behavior`, instead of loading the full 44-session cohort across both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`. Session `data_structure_*.mat` files are opened with `h5py`, so the loader assumes the session files are MATLAB v7.3/HDF5. Motion energy is read separately from `motionEnergy_*.mat` with `scipy.io.loadmat`.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", "Ephys_Behavior", 2),
    ...
    SessionSpec("JEB19", "2023-04-21", "Ephys_Behavior", 1),
]
```

```python
def process_session(spec: SessionSpec):
    with h5py.File(spec.data_path, "r") as h5:
        bp = load_bp_fields(h5)
        ...
```

```python
def read_motion_energy(path: Path):
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    me = loaded["me"]
    data = me.data
    if not isinstance(data, np.ndarray) and hasattr(data, "data"):
        data = data.data
    return data, float(me.moveThresh)
```

iii. In the trajectory, the AI said it was targeting the “Figure 8 two-context ALM ephys cohort” and later summarized the final dataset as “12 sessions, 7 mice, 3116 kept trials” (step 141). Earlier, it explicitly concluded that the “main session files are MATLAB v7.3/HDF5” and therefore should be read through `h5py` rather than `scipy.io.loadmat` (step 14).

## 1-b. How are the data split into subjects (mice)?

i. The AI uses the `animal` field of each hard-coded `SessionSpec` as the subject id, then builds `subjects` and `subject_idx` in first-appearance order across the session list.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    folder: str
    probe: int
```

```python
subjects = []
subject_lookup = {}
subject_idx = []
for spec in SESSION_SPECS:
    if spec.animal not in subject_lookup:
        subject_lookup[spec.animal] = len(subjects)
        subjects.append(spec.animal)
    subject_idx.append(subject_lookup[spec.animal])
```

iii. The trajectory justification is implicit: the AI chose a hand-curated session list for the Figure 8 cohort and carried the animal id directly in that list (steps 92 and 141). I did not find a more explicit subject-splitting justification.

## 1-c. How are the data split into sessions?

i. The AI treats each `SessionSpec` as one session and each `data_structure_<animal>_<date>.mat` file as one session. It does not search both task folders dynamically; it only processes the 12 fixed-delay `Ephys_Behavior` entries listed in `SESSION_SPECS`.

ii.
```python
class SessionSpec:
    ...
    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"

    @property
    def data_path(self) -> Path:
        return DATA_ROOT / self.folder / f"data_structure_{self.session_id}.mat"
```

```python
def build_dataset():
    sessions = [process_session(spec) for spec in SESSION_SPECS]
```

iii. The AI justified this as using “the paper’s two-context ALM ephys cohort from the Figure 8 pipeline” (step 141). Its plan also explicitly said it would “load the 12 Figure 8 sessions” (step 92).

## 1-d. How are the data split into trials?

i. Trials are defined by the per-trial arrays in `obj.bp`, indexed from `0..Ntrials-1`. After filtering, `valid_trials = np.flatnonzero(valid_mask)` is used as the trial list for outputs and inputs. Neural spikes are split into trials by `cluster["trial"]`, histogrammed per trial, then filtered by the same boolean `valid_mask`.

ii.
```python
def load_bp_fields(h5: h5py.File) -> dict[str, np.ndarray]:
    bp = h5["obj"]["bp"]
    ...
    return {
        "Ntrials": int(np.asarray(bp["Ntrials"])[0, 0]),
        "hit": np.asarray(bp["hit"]).reshape(-1).astype(bool),
        ...
    }
```

```python
valid_mask = (~bp["early"]) & (~bp["stim_enable"])
valid_trials = np.flatnonzero(valid_mask)
```

```python
trial_ids = np.asarray(cluster["trial"], dtype=np.int64)
aligned_times = trialtm - bp["goCue"][trial_ids - 1]
counts = align_and_histogram(trial_ids, aligned_times, n_trials)
cluster_trials = counts[valid_mask] / DT
```

iii. The AI’s trajectory consistently treated the dataset as already trial-structured, and it focused its work on deciding which trials to keep rather than reconstructing boundaries (for example steps 89, 92, and 141). I did not find a separate explicit argument beyond that.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops early-lick and photostimulation trials using `~bp["early"] & ~bp["stim_enable"]`. It does not implement the reference solution’s extra cut for trials that extend past the end of the neural recording.

ii.
```python
valid_mask = (~bp["early"]) & (~bp["stim_enable"])
valid_trials = np.flatnonzero(valid_mask)
```

iii. In the trajectory, the AI explicitly said “Trials are filtered with `~early & ~stim.enable`” (step 141). It also checked `haveEphys` and `haveVid` but concluded it would “drop missing-neural trials and keep missing-video trials” only if needed; for the 12-session cohort it found no missing flags and did not add another trial filter (steps 89 to 91).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the chosen probe’s cluster array: each cluster contributes `quality`, `trial`, and `trialtm`. The go-cue times `bp["goCue"]` are used to align spike times before binning.

ii.
```python
clusters.append(
    {
        "quality": quality,
        "trial": read_ref_array(h5, probe_group["trial"][clu_idx, 0]).astype(np.int64),
        "trialtm": read_ref_array(h5, probe_group["trialtm"][clu_idx, 0]).astype(np.float64),
    }
)
```

```python
trial_ids = np.asarray(cluster["trial"], dtype=np.int64)
trialtm = np.asarray(cluster["trialtm"], dtype=np.float64)
aligned_times = trialtm - bp["goCue"][trial_ids - 1]
```

iii. The trajectory justification matches the code: the AI said it had pinned down the neural side around “probe assignments,” “spike binning/smoothing,” and `goCue` alignment, and later summarized that it “matched the repository’s `goCue` alignment” (steps 92 and 141).

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, counted into 10 ms bins from `-3.0` to `2.5` s, converted to firing rate by dividing by `DT`, and smoothed with a custom causal half-Gaussian (`my_smooth`) using a 15-bin window and reflect padding. For unit filtering, the AI first builds seven condition-average PSTHs and uses their mean to estimate mean firing rate; for retained units it separately smooths per-trial firing rates.

ii.
```python
DT = 0.01
TMIN = -3.0
TMAX = 2.5
SMOOTH_WINDOW = 15
BCTYPE = "reflect"
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

```python
kernel = gausswin(n)
kernel[: len(kernel) // 2] = 0.0
kernel /= kernel.sum()
```

```python
psth = counts[cond_mask].sum(axis=0) / n_cond / DT
psth = my_smooth(psth, SMOOTH_WINDOW, BCTYPE)
...
cluster_trials = counts[valid_mask] / DT
cluster_trials = my_smooth(cluster_trials.T, SMOOTH_WINDOW, BCTYPE).T
```

iii. The AI stated that it had settled on “10 ms bins, causal Gaussian smoothing, Figure 8 trial conditions, and the same low-FR filter” (step 72), and it repeated that summary in the final message (step 141).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI excludes clusters whose lower-cased `quality` label is in `{"garbage", "gabrga", "noisy", "real?"}`. It then keeps only clusters whose average across seven smoothed condition-average PSTHs exceeds `1.0` Hz. It does not exclude `poor` units and does not trim trials after recording end.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0
```

```python
quality = str(cluster["quality"]).strip().lower()
if quality in QUALITY_EXCLUDE:
    continue
...
mean_fr = float(np.mean(np.stack(psths, axis=1)))
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. In the trajectory, the AI described this as using “the same low-FR filter” and “excluding garbage/noisy clusters” (steps 72 and 141). I did not find evidence that it noticed the reference code’s extra `poor` exclusion or the recording-end trial cut.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the go cue of that spike’s trial: `trialtm - bp["goCue"][trial_ids - 1]`.

ii.
```python
trial_ids = np.asarray(cluster["trial"], dtype=np.int64)
trialtm = np.asarray(cluster["trialtm"], dtype=np.float64)
aligned_times = trialtm - bp["goCue"][trial_ids - 1]
```

iii. The AI repeatedly described the neural stream as `goCue`-aligned, including in its final summary (step 141). This is one of the clearest decisions in the trajectory.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 10 ms bins (`DT = 0.01`) over a `[-3.0, 2.5]` s window, yielding 550 time bins. The AI does not rebin a finer neural representation later; 10 ms is its native output resolution.

ii.
```python
DT = 0.01
TMIN = -3.0
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

iii. The AI explicitly justified “10 ms bins” in the trajectory (step 72) and later reported that all sessions had `T = 550` during verification (steps 115, 128, and 139).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the AI’s global time grid relative to the go cue event. It is not read from a stored per-trial raw field; instead it is constructed from `TMIN`, `TMAX`, and `DT` to represent time from `bp.ev.goCue`.

ii.
```python
DT = 0.01
TMIN = -3.0
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

```python
inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]
```

iii. The trajectory justification is indirect: the AI framed the entire conversion around `goCue` alignment and said it “matched the repository’s `goCue` alignment” (step 141). I did not find a separate justification for treating time itself as a constructed variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes a single `TIME` vector of bin centers from `-3.0` to `2.5` s in 10 ms steps and copies that same 1-by-550 vector into every kept trial.

ii.
```python
TIME = EDGES[:-1] + (DT / 2.0)
...
inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]
```

iii. The AI’s stated rationale was to use the same 10 ms `goCue`-aligned binning grid for all streams (steps 72 and 141).

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input and neural data share the same `TIME`/`EDGES` grid. Neural spike counts are accumulated into `EDGES`, and the input uses the corresponding bin centers in `TIME`.

ii.
```python
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

```python
bins = np.searchsorted(EDGES, aligned_times, side="right") - 1
...
inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]
```

iii. The AI’s trajectory consistently described the neural and behavioral channels as sharing one aligned grid; the final summary names `goCue` alignment and 10 ms bins together (step 141).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial behavioral flags `R`, `L`, `hit`, `miss`, and `no`.

ii.
```python
def trial_choice_code(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    if bp["no"][trial_idx]:
        return 2
    if (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx]):
        return 1
    if (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx]):
        return 0
```

iii. The trajectory did not contain a separate detailed justification for this output, but the AI’s final summary says it used the two-context cohort and decoded lick direction from the behavioral trial structure (step 141). The use of explicit `no` trials is visible in the code rather than in a separate reasoning note.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI maps trials to three classes: `2` for `no` response, `1` for right licks (`R&hit` or `L&miss`), and `0` for left licks (`L&hit` or `R&miss`). It then repeats that per-trial category across all time bins of the trial.

ii.
```python
lick_code = trial_choice_code(bp, trial_idx)
...
trial_output = np.empty((len(OUTPUT_NAMES), TIME.size), dtype=np.int64)
trial_output[0] = lick_code
```

iii. The AI did not justify this separately in the trajectory, but it matches the behavioral logic it used everywhere else in the converter. No conflicting rationale appeared.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp["autowater"]`.

ii.
```python
context_code = 0 if bp["autowater"][trial_idx] else 1
```

iii. The trajectory explicitly tied the chosen cohort to the “two-context” task and said “context is taken from `autowater` in this two-context cohort” (step 141).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly remaps `autowater=True` to `WC` (`0`) and `False` to `DR` (`1`), then repeats that class across all time bins of the trial.

ii.
```python
trial_output[1] = context_code
```

```python
context_code = 0 if bp["autowater"][trial_idx] else 1
```

iii. The trajectory justification is the same as 5-a: the AI chose the two-context cohort and explicitly said it used `autowater` as the context indicator (step 141).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["miss"]`, `bp["hit"]`, and `bp["no"]`.

ii.
```python
def trial_outcome_code(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    if bp["miss"][trial_idx]:
        return 0
    if bp["hit"][trial_idx]:
        return 1
    if bp["no"][trial_idx]:
        return 2
```

iii. I did not find a separate written justification in the trajectory; this is an implementation-level choice. The final summary confirms that outcome was one of the decoded behavioral variables (step 141).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `miss -> 0` (`incorrect`), `hit -> 1` (`correct`), and `no -> 2` (`ignore`), then repeats that class across all time bins of the trial.

ii.
```python
outcome_code = trial_outcome_code(bp, trial_idx)
...
trial_output[2] = outcome_code
```

iii. The trajectory does not give a separate argument beyond choosing to keep ignore trials as a third class in the decoded outputs; that decision is implicit in the code and final dataset summary.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DeepLabCut coordinates in `obj.traj` from both cameras and from several tongue landmarks per view. On the side view it uses any of `tongue`, `left_tongue`, and `right_tongue`; on the bottom view it uses any of `top_tongue`, `topleft_tongue`, `bottom_tongue`, and `bottomleft_tongue`. Alignment additionally uses video `frameTimes`, the SpikeGLX/behavior bit-start offset, and `goCue`.

ii.
```python
tongue_indices = {
    0: [feat_names[0].index(name) for name in ["tongue", "left_tongue", "right_tongue"] if name in feat_names[0]],
    1: [
        feat_names[1].index(name)
        for name in ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]
        if name in feat_names[1]
    ],
}
```

```python
coords = side_ts[:, :2, feat_idx]
coords_interp = interpolate_coords(coords, side_ft, align_time, vidshift)
speed, visible = feature_speed(coords_interp, tongue_feature=True)
```

iii. The AI explicitly justified avoiding “guessing at one landmark” and said it wanted “a scalar summary that’s consistent across all 12 sessions” rather than “an arbitrary landmark” (steps 78 and 141).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI linearly interpolates each tongue landmark’s x/y coordinates onto the 10 ms `TIME` grid after correcting frame times by video shift and go cue. It computes velocity with `np.gradient` on the interpolated coordinates, converts that to speed, averages across whichever tongue landmarks are visible at each time bin, and does not apply the reference likelihood threshold, per-run smoothing, or per-view percentile normalization.

ii.
```python
def interpolate_coords(coords: np.ndarray, frame_times: np.ndarray, align_time: float, vidshift: float) -> np.ndarray:
    interp = interp1d(
        frame_times - vidshift - align_time,
        coords,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return np.asarray(interp(TIME), dtype=np.float64)
```

```python
if tongue_feature:
    vel = np.gradient(coords_interp, axis=0)
    vel[~np.isfinite(vel)] = 0.0
...
speed = np.sqrt((vel ** 2).sum(axis=1))
```

```python
agg_tongue, agg_tongue_visible = aggregate_speeds(trial_tongue_speeds, trial_tongue_visible)
```

iii. The trajectory shows the main justification: the AI wanted a “defensible single time series” and explicitly chose aggregation across multiple tracked tongue landmarks instead of one landmark (step 78). I did not find any trajectory evidence that it intended to preserve the reference likelihood thresholding or smoothing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After assembling all kept-trial tongue speeds, the AI computes a session-wide 50th percentile over bins marked visible, then thresholds each visible bin into `0` below median or `1` at/above median. Bins marked not visible are set to `2`.

ii.
```python
tongue_thresh = float(np.nanpercentile(tongue_series[tongue_visible], 50))
...
trial_output[3] = np.where(
    ~tongue_visible[out_idx],
    2,
    (tongue_series[out_idx] >= tongue_thresh).astype(np.int64),
)
```

iii. The AI’s final verification summary highlighted that tongue was mostly “not visible,” and it accepted that distribution as plausible for this task (step 116). The percentile split itself follows the prompt rather than an explicit extra argument in the trajectory.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue coordinates are aligned by computing a session-wide video shift from SpikeGLX bitcode and behavior `bitStart`, subtracting that shift and the trial’s `goCue` from `frameTimes`, and then interpolating onto the same 10 ms `TIME` grid used by the neural data.

ii.
```python
def get_video_shift(h5: h5py.File, bp: dict[str, np.ndarray]) -> float:
    fs = float(np.asarray(h5["obj"]["sglx"]["fs"]).reshape(-1)[0])
    bitstart = np.asarray(h5["obj"]["sglx"]["bitcode"]["bitstart"]).reshape(-1)
    return matlab_mode(bitstart) / fs - matlab_mode(bp["bitStart"])
```

```python
interp = interp1d(
    frame_times - vidshift - align_time,
    coords,
    ...
)
return np.asarray(interp(TIME), dtype=np.float64)
```

iii. The AI explicitly said it was checking and using the video alignment shift, and later summarized the conversion as matching the repository’s `goCue` alignment for the Figure 8 cohort (steps 89 and 141).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DeepLabCut coordinates for both `top_paw` and `bottom_paw`, plus bottom-camera `frameTimes`, the video shift from bitcode/`bitStart`, and `goCue`.

ii.
```python
paw_indices = {
    1: [feat_names[1].index(name) for name in ["top_paw", "bottom_paw"] if name in feat_names[1]],
}
```

```python
coords = bottom_ts[:, :2, feat_idx]
coords_interp = interpolate_coords(coords, bottom_ft, align_time, vidshift)
speed, visible = feature_speed(coords_interp, tongue_feature=False)
```

iii. The AI justified this indirectly with the same argument it used for tongue: it wanted a consistent scalar summary instead of picking one “arbitrary landmark” (step 78). I did not find a separate paw-specific justification.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI linearly interpolates bottom-camera paw coordinates to the 10 ms grid, nearest-fills missing coordinates, computes gradients, subtracts a baseline derivative term, converts to speed, and averages across `top_paw` and `bottom_paw` if both are present. It does not use a likelihood threshold or the reference per-run Gaussian smoothing.

ii.
```python
coords_filled = fill_nearest(coords_interp)
...
vel = np.gradient(coords_filled, axis=0)
base_deriv = np.nanmedian(np.diff(coords_filled, axis=0), axis=0)
vel[:, 0] = vel[:, 0] - base_deriv[0]
vel[:, 1] = vel[:, 1] - base_deriv[0]
vel = fill_nearest(vel)
speed = np.sqrt((vel ** 2).sum(axis=1))
```

```python
if trial_paw_speeds:
    agg_paw = np.mean(np.stack(trial_paw_speeds, axis=0), axis=0)
    agg_paw_visible = np.ones(TIME.size, dtype=bool)
```

iii. The trajectory justification is the same as for 7-b: the AI wanted one scalar paw summary over the 12-session cohort and did not want to commit to a single landmark without checking feature availability (step 78). It did not justify omitting the reference smoothing/likelihood handling.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI computes a session-wide 50th percentile over the bins it considers visible, thresholds visible bins into `0` or `1`, and reserves `2` for bins it marks not visible. In practice, because `agg_paw_visible` is set to all-ones whenever any paw trace exists on a trial, paw output is almost always binary and rarely uses class `2`.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_series[paw_visible], 50))
```

```python
if trial_paw_speeds:
    agg_paw = np.mean(np.stack(trial_paw_speeds, axis=0), axis=0)
    agg_paw_visible = np.ones(TIME.size, dtype=bool)
...
trial_output[4] = np.where(
    ~paw_visible[out_idx],
    2,
    (paw_series[out_idx] >= paw_thresh).astype(np.int64),
)
```

iii. The trajectory does not justify this visibility handling explicitly. The only related comment is that the AI was monitoring class-balance pathologies during validation (step 118).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw coordinates are aligned exactly the same way as tongue coordinates: bottom-camera `frameTimes` are shifted by the session-wide video offset and by the trial’s `goCue`, then interpolated to the neural `TIME` grid.

ii.
```python
coords_interp = interpolate_coords(coords, bottom_ft, align_time, vidshift)
```

```python
interp = interp1d(
    frame_times - vidshift - align_time,
    coords,
    ...
)
return np.asarray(interp(TIME), dtype=np.float64)
```

iii. The AI’s trajectory justification is the same shared video/neural alignment argument used for tongue and motion energy (steps 89 and 141).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is read from the session’s `motionEnergy_<session>.mat` file, and aligned using the side-camera frame times from `obj.traj`, plus the video shift and `goCue`.

ii.
```python
def read_motion_energy(path: Path):
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    me = loaded["me"]
    data = me.data
    ...
    return data, float(me.moveThresh)
```

```python
me_trace = np.asarray(motion_energy_data[trial_idx]).reshape(-1)
if side_ts is None:
    motion = np.full(TIME.size, np.nan, dtype=np.float64)
else:
    motion = interpolate_motion_energy(me_trace, side_ft, align_time, vidshift)
```

iii. The trajectory shows the AI inspecting motion-energy file structure and fixing its loader when it mistakenly unwrapped NumPy’s own `.data` buffer instead of a MATLAB wrapper (steps 101 to 103). That is the clearest motion-energy-specific justification.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates each trial’s motion-energy trace to the 10 ms `TIME` grid and then nearest-fills missing bins. It does not use the reference bin-wise frame averaging approach.

ii.
```python
def interpolate_motion_energy(me_trace: np.ndarray, frame_times: np.ndarray, align_time: float, vidshift: float) -> np.ndarray:
    if frame_times.size != me_trace.size:
        frame_times = np.arange(1, me_trace.size + 1, dtype=np.float64) / 400.0
    interp = interp1d(
        frame_times - vidshift - align_time,
        me_trace,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))
```

iii. I did not find a separate trajectory justification for interpolation versus binning. The AI’s broader justification was to put all time-varying outputs on the same 10 ms aligned grid (steps 72 and 141).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes a session-wide 50th percentile over finite motion-energy bins and thresholds bins into `0` or `1`; bins left non-finite are assigned `2`. Because `fill_nearest` makes most interpolated bins finite, the `2` class is effectively limited to trials where side-camera video is absent.

ii.
```python
me_thresh = float(np.nanpercentile(me_series[me_visible], 50))
...
trial_output[5] = np.where(
    ~me_visible[out_idx],
    2,
    (me_series[out_idx] >= me_thresh).astype(np.int64),
)
```

iii. The trajectory does not discuss this thresholding separately. Its only explicit rationale is that it was checking for class-balance problems during decoder validation (step 118).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy frame times are aligned using the same session-wide video shift and per-trial `goCue` subtraction used for kinematics, then interpolated to the neural `TIME` grid.

ii.
```python
motion = interpolate_motion_energy(me_trace, side_ft, align_time, vidshift)
```

```python
interp = interp1d(
    frame_times - vidshift - align_time,
    me_trace,
    ...
)
return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))
```

iii. The trajectory justification is the same shared alignment argument used for the other video-derived outputs (steps 89 and 141).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally fills through missing video data instead of leaving gaps. If `frameTimes` are missing or all-NaN, it synthesizes a 400 Hz frame-time grid; missing coordinates are nearest-filled for paw and missing motion-energy bins are nearest-filled after interpolation. For tongue, NaN bins remain “not visible.” Trials are usually kept rather than dropped.

ii.
```python
if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
```

```python
def fill_nearest(x: np.ndarray) -> np.ndarray:
    ...
    interp = interp1d(
        idx[valid],
        x[valid, col],
        kind="nearest",
        bounds_error=False,
        fill_value=(x[valid, col][0], x[valid, col][-1]),
    )
```

```python
return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))
```

iii. The trajectory explicitly says the AI checked `haveEphys`/`haveVid` to decide whether to keep missing-video trials and later cleaned up an all-NaN warning rather than changing the missing-data policy (steps 89 to 91, 107, and 129 to 130).

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps appear to be the repeated per-trial video interpolation and feature extraction in `compute_behavioral_outputs`, plus reading each session twice with `h5py` (once in `process_session`, once again in `compute_behavioral_outputs`). The trajectory also treats video interpolation over 12 sessions as the expected runtime bottleneck.

ii.
```python
with h5py.File(spec.data_path, "r") as h5:
    bp = load_bp_fields(h5)
    ...
outputs, thresholds = compute_behavioral_outputs(spec, bp, valid_trials)
```

```python
with h5py.File(spec.data_path, "r") as h5:
    vidshift = get_video_shift(h5, bp)
    ...
    for trial_idx in valid_trials:
        ...
```

iii. The AI wrote that a full run was “expected with per-trial video interpolation across 12 sessions” (step 111). It did not mention file I/O as the main bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized: the per-cluster/per-condition loop used only to estimate mean firing rate, the per-time loop inside `aggregate_speeds`, and many of the per-trial feature loops in `compute_behavioral_outputs`. The spike histogramming itself is already partly vectorized through `np.add.at`.

ii.
```python
for cluster in clusters:
    ...
    for cond_mask in condition_masks:
        ...
```

```python
for t in range(TIME.size):
    mask = vis_stack[:, t]
    if np.any(mask):
        out[t] = float(np.mean(speed_stack[mask, t]))
```

```python
for trial_idx in valid_trials:
    ...
    for feat_idx in tongue_indices[0]:
        ...
    for feat_idx in tongue_indices[1]:
        ...
    for feat_idx in paw_indices[1]:
        ...
```

iii. I did not find an explicit trajectory justification for leaving these loops unvectorized. The trajectory mainly focused on correctness and decoder validation rather than optimization.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats work in a few places. It opens each session file twice, once for neural extraction and once again for video-derived outputs. For neural QC it computes seven condition-average PSTHs for each cluster, smooths them, uses them only to estimate mean firing rate, and then separately smooths the per-trial neural data for the same cluster.

ii.
```python
with h5py.File(spec.data_path, "r") as h5:
    bp = load_bp_fields(h5)
    ...
outputs, thresholds = compute_behavioral_outputs(spec, bp, valid_trials)
```

```python
for cond_mask in condition_masks:
    ...
    psth = counts[cond_mask].sum(axis=0) / n_cond / DT
    psth = my_smooth(psth, SMOOTH_WINDOW, BCTYPE)
...
cluster_trials = counts[valid_mask] / DT
cluster_trials = my_smooth(cluster_trials.T, SMOOTH_WINDOW, BCTYPE).T
```

iii. The trajectory gives one indirect justification: the AI said it was intentionally using “Figure 8 trial conditions” and the “same low-FR filter” (step 72), which explains why it chose to build condition-average PSTHs before filtering units. It did not explicitly justify reopening the file or re-smoothing.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded computation is the set of seven smoothed condition-average PSTHs per cluster, which are used only to compute a scalar mean firing-rate filter and are never stored in the output. The loader also reads `sample` and `delay` fields into `bp`, but they are never used downstream, and `read_motion_energy` returns `moveThresh` even though nothing uses it.

ii.
```python
return {
    ...
    "sample": np.asarray(ev["sample"]).reshape(-1),
    "delay": np.asarray(ev["delay"]).reshape(-1),
    ...
}
```

```python
psths = []
for cond_mask in condition_masks:
    ...
mean_fr = float(np.mean(np.stack(psths, axis=1)))
if mean_fr <= LOW_FR_HZ:
    continue
```

```python
return data, float(me.moveThresh)
```

iii. The trajectory never presents these as intentional downstream products. The only explicit rationale is that the AI wanted the Figure 8 condition structure for its low-firing-rate filter (step 72).
