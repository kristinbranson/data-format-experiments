# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only curated ALM sessions by parsing the authors' `load*_ALMVideo.m` manifest files under `DataLoadingScripts/Recording and video`, then locating each session in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`. Each selected session is loaded once with a mixed MATLAB loader: HDF5/v7.3 files go through `load_session_v73`, and older v5 files go through `load_session_v5`.

ii.
```python
def parse_manifest_sessions() -> list[SessionSpec]:
    sessions: list[SessionSpec] = []
    for manifest in sorted(MANIFEST_ROOT.glob("load*_ALMVideo.m")):
        ...
        for date, probe_text in zip(dates, probes):
            probe_list = parse_probe_list(probe_text)
            for cohort in ("Ephys_Behavior", "RandomizedDelay_Ephys_Behavior"):
                data_path = DATA_ROOT / cohort / f"data_structure_{subject}_{date}.mat"
                if data_path.exists():
                    sessions.append(SessionSpec(...))
                    break

def load_raw_session(path: Path) -> dict:
    if h5py.is_hdf5(path):
        return load_session_v73(path)
    return load_session_v5(path)
```

iii. The justification in `CONVERSION_NOTES.md` is that the raw download contains extra cohorts and sessions not used in the paper, so the manifest files are the authoritative definition of the ALM dataset. The notes also justify the dual loader because `/app/data` mixes MATLAB v7.3/HDF5 and v5 files.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the manifest/file naming convention: the subject is the `subject` field in `SessionSpec`, which is the animal ID before the date, and later `subjects` is the sorted unique list of those IDs with `subject_idx` per session.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    ...

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"
```
```python
subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
...
"subject_idx": np.array([subject_to_idx[sess["subject"]] for sess in processed_sessions], dtype=np.int64),
```

iii. The notes justify using the curated manifest and session filename structure because metadata fields inside raw files are heterogeneous and the manifest already encodes the intended session identity.

## 1-c. How are the data split into sessions?

i. One manifest entry becomes one session. Each `SessionSpec` corresponds to one `data_structure_<subject>_<date>.mat` file plus its associated motion-energy file, and each processed session becomes one element in `neural`, `input`, and `output`.

ii.
```python
for date, probe_text in zip(dates, probes):
    probe_list = parse_probe_list(probe_text)
    for cohort in ("Ephys_Behavior", "RandomizedDelay_Ephys_Behavior"):
        data_path = DATA_ROOT / cohort / f"data_structure_{subject}_{date}.mat"
        if data_path.exists():
            sessions.append(SessionSpec(...))
            break
```
```python
"neural": [sess["neural"] for sess in processed_sessions],
"input": [sess["input"] for sess in processed_sessions],
"output": [sess["output"] for sess in processed_sessions],
```

iii. In the notes, the AI explicitly chose the 44 curated ALM ephys sessions from the manifests rather than every file in `/app/data`, because the paper/code operate on that curated subset.

## 1-d. How are the data split into trials?

i. Trials are split by raw trial index from `bp`: the session loader reads per-trial behavior arrays of length `Ntrials`, and later `candidate_trials` is an array of raw trial indices. Neural and behavioral outputs are indexed by those raw trial numbers rather than reconstructing trial boundaries from spikes or video.

ii.
```python
bp_out = {
    "Ntrials": ntrials,
    "hit": bp["hit"][()].reshape(-1).astype(bool),
    "miss": bp["miss"][()].reshape(-1).astype(bool),
    ...
    "ev": {
        "goCue": ev["goCue"][()].reshape(-1),
        ...
    },
}
```
```python
candidate_trials = np.flatnonzero(~raw["bp"]["early"] & np.isfinite(raw["bp"]["ev"]["goCue"]))
...
for raw_trial_idx in valid_trials.tolist():
    ...
    out[0, :] = lick_direction_value(raw["bp"], raw_trial_idx)
```

iii. The notes describe the native data as already organized trial-wise in `obj.bp`, `obj.traj`, and cluster `trial`/`trialtm` fields, so the AI treated the raw trial index as the canonical split.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials with finite `goCue` and excludes early-lick trials. It does not exclude photostimulation trials. After neural binning, it also removes any kept trial whose neural matrix is entirely zero, interpreting those as behavior continuing after the recording stopped. Sessions with fewer than 2 surviving trials or fewer than 10 units are skipped entirely.

ii.
```python
candidate_trials = np.flatnonzero(~raw["bp"]["early"] & np.isfinite(raw["bp"]["ev"]["goCue"]))
```
```python
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if nonzero_trial_mask.size and not np.all(nonzero_trial_mask):
    candidate_trials = candidate_trials[nonzero_trial_mask]
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_trial_mask.tolist()) if keep]
```
```python
if len(neural_trials) < 2 or len(keep_unit_idx) < 10:
    info["skipped"] = True
    return None, info
```

iii. The notes justify early-lick exclusion from the paper and the all-zero-trial removal as a necessary fix for two JEB24 sessions where behavioral trials outlasted the selected neural recording. The metadata explicitly records `trial_exclusion` as `"early_lick_only"`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the selected probe(s) in `raw["clu_probes"]`, specifically each unit's `trial`, `trialtm`, and `quality` fields, together with `bp.ev.goCue` for alignment.

ii.
```python
units.append(
    {
        "quality": quality,
        "tm": tm,
        "trialtm": trialtm,
        "trial": trial,
        "site": int(site[0]) if site.size else -1,
    }
)
```
```python
trial_idx = unit["trial"] - 1
aligned = unit["trialtm"] - align_times[trial_idx]
```

iii. The notes frame this as reference-style ALM probe selection plus go-cue alignment from the same raw cluster fields used in the authors' code.

## 2-b. How is the `neural` data processed?

i. For each selected unit, spike times are aligned by subtracting the trial's `goCue`, binned into a common `[-2.5, 2.5)` window at 10 ms resolution, converted from counts to Hz, then smoothed with a custom causal half-Gaussian kernel of length 15 samples and reflect padding. Surviving units are stored as `float32`.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 100  # 10 ms; matches most figure/decoder scripts
SMOOTH_N = 15
```
```python
bin_idx = np.floor((aligned[mask] - TMIN) / DT).astype(int)
np.add.at(counts, (mapped_trials, bin_idx), 1.0)
fr = smooth_causal_reflect((counts / DT).T, SMOOTH_N, BCTYPE).T
```
```python
kern = gaussian(n, std=std)
kern[: len(kern) // 2] = 0.0
kern = kern / kern.sum()
```

iii. The notes and trajectory justify this as matching "most figure/decoder scripts" rather than `getDefaultParams.m`, so the AI deliberately chose a 10 ms grid and causal smoothing for the exported trial tensors.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are first restricted to the manifest-selected probe(s). Within those probes, any unit whose free-text `quality` label lower-cases to `garbage`, `gabrga`, `noisy`, or `real?` is excluded. After binning/smoothing, units with mean firing rate `<= 1 Hz` over the exported window are dropped.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0

def matlab_quality_ok(quality: str) -> bool:
    return quality.strip().lower() not in QUALITY_EXCLUDE
```
```python
for unit in probe_units:
    if matlab_quality_ok(unit["quality"]):
        units.append(unit)
```
```python
mean_fr = float(np.nanmean(fr))
if mean_fr > LOW_FR_HZ:
    keep_idx.append(unit_idx)
```

iii. The notes justify this as "reference-matched" quality exclusion plus the paper's `>1 Hz` rule for "all other analyses". The notes also explicitly mention choosing all non-garbage/non-noisy ALM units rather than only single units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each spike to go cue by subtracting `bp.ev.goCue` from `trialtm` on the same trial, then bins the result on the common trial grid.

ii.
```python
align_times = raw["bp"]["ev"]["goCue"]
trial_idx = unit["trial"] - 1
aligned = unit["trialtm"] - align_times[trial_idx]
```

iii. The notes justify go-cue alignment as consistent with the task instructions and the reference alignment functions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 10 ms bins over `[-2.5, 2.5)` seconds, producing 500 time points per trial. This is a fresh binning choice imposed during conversion; raw spikes are event times.

ii.
```python
DT = 1 / 100  # 10 ms; matches most figure/decoder scripts

def build_time_axis() -> np.ndarray:
    edges = np.arange(TMIN, TMAX + DT, DT)
    return edges[:-1] + DT / 2
```

iii. The AI's notes explicitly justify 10 ms by citing "most analysis scripts" and the decoder pipeline, despite also recording that the default params file uses 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw variable directly. The AI defines it from the exported common time grid around the raw `goCue` event, using the centers of the 10 ms bins.

ii.
```python
def build_time_axis() -> np.ndarray:
    edges = np.arange(TMIN, TMAX + DT, DT)
    return edges[:-1] + DT / 2
```
```python
inp = time_axis[np.newaxis, :].astype(np.float32)
```

iii. The notes justify this as the only decoder input requested by the task, built on the same go-cue-aligned trial grid as the neural data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The only processing is constructing a shared centered time vector once per session from `TMIN`, `TMAX`, and `DT`, then repeating that `1 x T` array for every valid trial.

ii.
```python
time_axis = build_time_axis()
...
for raw_trial_idx in valid_trials.tolist():
    inp = time_axis[np.newaxis, :].astype(np.float32)
    inputs.append(inp)
```

iii. The notes describe this as a task-specific export choice rather than a transformation present in the reference code.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same bin grid used for neural spike binning, so each input time point corresponds to the same 10 ms interval as the neural data at that column.

ii.
```python
time_axis = build_time_axis()
neural_trials, keep_unit_idx = bin_session_neural(raw, units, candidate_trials, time_axis)
...
inp = time_axis[np.newaxis, :].astype(np.float32)
```

iii. The notes justify this by emphasizing a common go-cue-centered trial tensor for neural, input, and output streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the per-trial raw behavior flags `R`, `L`, `hit`, `miss`, and `no`.

ii.
```python
def lick_direction_value(bp: dict, trial_idx: int) -> int:
    if bp["no"][trial_idx]:
        return 2
    right_choice = (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx])
    left_choice = (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx])
```

iii. The notes justify this as decoding the animal's actual lick side, not merely the instructed side, by combining side and outcome flags.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI maps trials to three classes: `0` left, `1` right, `2` none. `no` trials go to class 2, hits give the instructed side, and misses flip the instructed side to the opposite lick direction.

ii.
```python
if bp["no"][trial_idx]:
    return 2
right_choice = (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx])
left_choice = (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx])
if left_choice:
    return 0
if right_choice:
    return 1
return 2
```

iii. The notes explicitly justify this as reconstructing actual choice from `R/L` and `hit/miss/no`, then repeating the per-trial label across all time bins.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the raw per-trial `autowater` flag.

ii.
```python
def context_value(bp: dict, trial_idx: int) -> int:
    return 1 if bp["autowater"][trial_idx] else 0
```

iii. The notes justify using `autowater` directly because the reference trial logic distinguishes delayed-response from water-cued trials via this field.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI encodes behavioral context as a binary per-trial label repeated across time: `0` for delayed-response (`autowater == 0`) and `1` for water-cued (`autowater == 1`). The metadata names the classes in that order as `["DR", "WC"]`.

ii.
```python
OUTPUT_VALUES = [
    ["left", "right", "none"],
    ["DR", "WC"],
    ...
]
```
```python
def context_value(bp: dict, trial_idx: int) -> int:
    return 1 if bp["autowater"][trial_idx] else 0
```

iii. The Step 5 mapping notes explicitly state `autowater==0 -> DR` and `autowater==1 -> WC`, and justify labeling DR-only and randomized-delay sessions as all-DR for decoder purposes.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `miss` and `hit`, with the remaining trials implicitly treated as ignore trials.

ii.
```python
def outcome_value(bp: dict, trial_idx: int) -> int:
    if bp["miss"][trial_idx]:
        return 0
    if bp["hit"][trial_idx]:
        return 1
    return 2
```

iii. The notes justify this as the standard three-way mapping of incorrect/correct/ignore from the same raw behavioral flags used elsewhere.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI encodes outcome as `0` incorrect, `1` correct, and `2` ignore, then repeats that categorical value across the whole trial.

ii.
```python
out[2, :] = outcome_value(raw["bp"], raw_trial_idx)
```
```python
if bp["miss"][trial_idx]:
    return 0
if bp["hit"][trial_idx]:
    return 1
return 2
```

iii. The notes justify keeping ignore as a third class because the task asks for it explicitly, even though some paper analyses drop ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera tongue trajectory only: `raw["traj"][0]` feature `"tongue"`, specifically its tracked `ts` coordinates and `frameTimes`, together with the session-wide video offset from `bp.ev.bitStart`, `sglx.bitcode.bitstart`, and `sglx.fs`, and the per-trial `goCue`.

ii.
```python
tongue_x, tongue_y, tongue_vis = align_feature(raw, time_axis, align_times, 0, "tongue")
```
```python
vidshift = find_video_offset(raw)
...
ts = np.asarray(trial["ts"][:, :2, feat_idx], dtype=float)
frame_times = np.asarray(trial["frameTimes"], dtype=float).reshape(-1)
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
y = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 1], time_axis + ADVANCE_MOVEMENT)
```

iii. The notes justify preserving a `not visible` class from raw visibility and aligning video with the same event-offset logic as the reference, but they do not justify using only the side-view tongue after earlier planning notes discussed broader tongue handling.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI first interpolates tongue `x` and `y` positions from camera frame times onto the neural 10 ms grid while preserving NaN gaps. It then computes per-bin tongue speed as `sqrt(gradient(x)^2 + gradient(y)^2)` on that aligned grid, without explicit likelihood thresholding or temporal smoothing. Finally it discretizes the resulting speed by the session median over visible bins.

ii.
```python
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
y = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 1], time_axis + ADVANCE_MOVEMENT)
vis = np.isfinite(x) & np.isfinite(y)
```
```python
xvel = np.gradient(xx)
yvel = np.gradient(yy)
speed[:, tr_idx] = np.sqrt(xvel ** 2 + yvel ** 2)
```
```python
tongue_speed = compute_speed(tongue_x, tongue_y, "tongue")
tongue_cat, tongue_thr = discretize_visible_signal(tongue_speed, tongue_vis)
```

iii. The notes justify the final median split and visibility-aware class `2`, but do not provide an explicit rationale for omitting the reference camera-space smoothing and two-view normalization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After computing tongue speed, the AI thresholds it per session at the 50th percentile over bins marked visible and finite. Visible bins below threshold become class `0`, visible bins at or above threshold become class `1`, and invisible/NaN bins remain class `2`.

ii.
```python
def discretize_visible_signal(values: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, float]:
    out = np.full(values.shape, 2, dtype=np.int64)
    valid_vals = values[visible & np.isfinite(values)]
    ...
    thresh = float(np.nanpercentile(valid_vals, 50))
    low_mask = visible & np.isfinite(values) & (values < thresh)
    high_mask = visible & np.isfinite(values) & ~low_mask
    out[low_mask] = 0
    out[high_mask] = 1
```

iii. The notes explicitly justify a session-specific 50th-percentile split because the decoder task asked for that discretization, and class `2` is justified as the required `not_visible` state.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI subtracts a session-wide video offset and the trial's go cue from each frame time, then interpolates tongue coordinates directly onto the same 10 ms time axis used for the neural data.

ii.
```python
def find_video_offset(raw: dict) -> float:
    bit_start = mode_float(raw["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_float(raw["sglx_bitstart"]) / raw["sglx_fs"]
    return vid_file_offset - bit_start
```
```python
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
y = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 1], time_axis + ADVANCE_MOVEMENT)
```

iii. The notes justify using the same event-offset logic as `findVideoOffset.m` and aligning all exported streams onto a common go-cue-centered grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera paw features, `"top_paw"` and `"bottom_paw"`, using each feature's `ts` coordinates and `frameTimes`, again aligned via `goCue` and the session video offset.

ii.
```python
paw_feats = ["top_paw", "bottom_paw"]
for feat in paw_feats:
    x, y, vis = align_feature(raw, time_axis, align_times, 1, feat)
    paw_speeds.append(compute_speed(x, y, feat))
    paw_vis.append(vis)
```

iii. The notes justify using bottom-camera paw tracking generally, but the code itself expands that to both paw features and later combines them.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI aligns both paw markers to the 10 ms neural grid, fills missing coordinates by nearest-neighbor interpolation, computes per-bin speed from coordinate gradients, then combines the two paw markers by taking the maximum finite speed at each bin. For non-tongue features it also subtracts a crude baseline derivative and fills missing gradients.

ii.
```python
if not is_tongue:
    x = fill_nearest_1d(x)
    y = fill_nearest_1d(y)
```
```python
if not is_tongue:
    diffs = np.column_stack([np.diff(xx), np.diff(yy)])
    ...
    xvel = fill_nearest_1d(xvel)
    yvel = fill_nearest_1d(yvel)
speed[:, tr_idx] = np.sqrt(xvel ** 2 + yvel ** 2)
```
```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_finite = np.isfinite(paw_stack)
paw_speed = np.max(np.where(paw_finite, paw_stack, -np.inf), axis=0)
paw_speed[~paw_finite.any(axis=0)] = np.nan
```

iii. The notes justify visibility-aware paw categories and bottom-view paw tracking, but do not give a separate rationale for the two-paw max combination or the gradient-on-interpolated-grid implementation.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded per session with the same helper as tongue velocity: class `0` below the median of visible finite values, class `1` at or above that median, and class `2` where neither paw marker is visible.

ii.
```python
paw_visible = np.any(np.stack(paw_vis, axis=0), axis=0)
paw_cat, paw_thr = discretize_visible_signal(paw_speed, paw_visible)
```

iii. The notes justify the 50th-percentile session split because it is required by the decoder specification, and the visibility-aware class `2` because the task asks for a `not visible` category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. As with tongue velocity, the AI subtracts the session video offset and the trial go cue from each bottom-camera frame time, then interpolates onto the same 10 ms neural time grid before computing speed.

ii.
```python
x, y, vis = align_feature(raw, time_axis, align_times, 1, feat)
```
```python
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
y = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 1], time_axis + ADVANCE_MOVEMENT)
```

iii. The notes justify a single common aligned trial tensor for neural and movement outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from a separate file beside each session, `motionEnergy_<subject>_<date>.mat`. The loader unwraps possible nested `.data` fields and returns one numeric trace per trial.

ii.
```python
def load_motion_energy(spec: SessionSpec) -> tuple[list[np.ndarray] | None, float | None]:
    me_path = spec.data_path.with_name(f"motionEnergy_{spec.subject}_{spec.date}.mat")
    if not me_path.exists():
        return None, None
    me = sio.loadmat(me_path, struct_as_record=False, squeeze_me=True)["me"]
    ...
    while hasattr(payload, "_fieldnames") and "data" in payload._fieldnames:
        ...
        payload = next_payload
    ...
    for trial_arr in np.ravel(trial_source):
        data.append(np.asarray(trial_arr, dtype=float).reshape(-1))
```

iii. The notes justify this by pointing out that motion-energy files appear in multiple nested layouts and should be normalized into one representation at load time.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligns each per-frame motion-energy trace to go cue using side-camera frame times and the session video offset, interpolates it onto the 10 ms neural grid, fills missing bins by nearest-neighbor, and then discretizes the aligned values by the per-session median over finite bins.

ii.
```python
sig = interp_numeric(tt, y, time_axis + ADVANCE_MOVEMENT)
aligned[:, tr_idx] = fill_nearest_1d(sig)
```
```python
valid = np.isfinite(me_aligned)
vals = me_aligned[valid]
thresh = float(np.nanpercentile(vals, 50))
out[valid & (me_aligned < thresh)] = 0
out[valid & (me_aligned >= thresh)] = 1
```

iii. The notes justify the 50th-percentile discretization as task-specific and describe the aligned motion-energy trace as a decoder-oriented export built on the reference offset logic.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded per session at the median of finite aligned values. Class `0` is below threshold, class `1` is at or above threshold, and class `2` is reserved for missing/no-video cases.

ii.
```python
def discretize_motion_energy(me_aligned: np.ndarray | None) -> tuple[np.ndarray, float]:
    if me_aligned is None:
        return None, float("nan")
    out = np.full(me_aligned.shape, 2, dtype=np.int64)
    valid = np.isfinite(me_aligned)
    ...
    thresh = float(np.nanpercentile(vals, 50))
```

iii. The notes explicitly justify this as matching the decoder-task requirement of a session-median split with a special missing-video class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses side-camera frame times, shifted by the session video offset and the trial's go cue, then interpolated onto the same 10 ms trial grid used by the neural data.

ii.
```python
trial = raw["traj"][0][tr_idx]
frame_times = np.asarray(trial["frameTimes"], dtype=float).reshape(-1)
...
tt = frame_times - vidshift - align_times[tr_idx]
sig = interp_numeric(tt, y, time_axis + ADVANCE_MOVEMENT)
```

iii. The notes justify this as using the same reference-style video offset as the other video-derived outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several raw-data irregularities in code: mixed MAT-file formats; HDF5 cell arrays with different orientations; sessions missing `obj.ex`; `ex.probe.loc` stored as a direct char array; `site` versus `channel`; nested motion-energy structs; and trials extending past the neural recording, which are removed if their binned neural matrix is all zeros. For video, empty `frameTimes` are replaced with a nominal 400 Hz timeline, non-tongue NaNs are nearest-filled, and if `NdroppedFrames` is NaN the feature is skipped, leaving bins invisible/NaN.

ii.
```python
def h5_cell_ref(cell_ds, idx: int):
    refs = np.asarray(cell_ds[()]).reshape(-1, order="F")
```
```python
if frame_times.size == 0:
    frame_times = (np.arange(ts.shape[0]) + 1) / 400.0
```
```python
if np.isnan(ndropped):
    continue
```
```python
if nonzero_trial_mask.size and not np.all(nonzero_trial_mask):
    candidate_trials = candidate_trials[nonzero_trial_mask]
```

iii. The notes devote substantial space to these edge cases and justify them as necessary to get all 44 curated sessions through validation. The zero-neural-trial removal is singled out in Step 10 as a post-verification fix for two JEB24 sessions.

## 11-a. What are the most time-consuming steps of the code?

i. The AI's notes identify session loading and the overall per-session conversion loop as the main runtime cost; full conversion is reported at roughly 130 seconds for 44 sessions.

ii.
```python
for idx, spec in enumerate(session_specs, start=1):
    session_data, info = convert_one_session(spec, show_processing=do_plot)
```
```python
def load_raw_session(path: Path) -> dict:
    if h5py.is_hdf5(path):
        return load_session_v73(path)
    return load_session_v5(path)
```

iii. `CONVERSION_NOTES.md` explicitly states that the mixed-format loading path dominated runtime and records per-session timing estimates around 3 seconds.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain that could be reduced or cached: the per-unit loop in `bin_session_neural`, the per-spike membership/map construction inside that loop, the per-trial loops in `align_feature`, `compute_speed`, `align_motion_energy`, and the per-trial construction of identical input arrays in `build_session_outputs`.

ii.
```python
for unit_idx, unit in enumerate(units):
    ...
    valid_mask = np.array([t in trial_map for t in trial_idx], dtype=bool)
    ...
    mapped_trials = np.array([trial_map[t] for t in trial_idx[mask]], dtype=int)
```
```python
for tr_idx in range(ntrials):
    ...
```
```python
for raw_trial_idx in valid_trials.tolist():
    inp = time_axis[np.newaxis, :].astype(np.float32)
```

iii. The notes emphasize that the code is "vectorized within trials wherever possible," but they also acknowledge a session-level looping structure and do not claim these loops were eliminated.

## 11-c. What processing does the code repeat multiple times?

i. The code recomputes some session-constant quantities repeatedly: `parse_manifest_sessions()` is called twice in `main`, `find_video_offset(raw)` is recomputed separately inside each `align_feature` call and again in `align_motion_energy`, and the exact same `time_axis[np.newaxis, :]` input array is rebuilt for every trial.

ii.
```python
print(f"Curated sessions available: {len(parse_manifest_sessions())}")
...
print(f"Estimated total time at this rate for full curated set: {np.mean(timings) * len(parse_manifest_sessions()):.2f}s")
```
```python
def align_feature(...):
    vidshift = find_video_offset(raw)
    ...

def align_motion_energy(...):
    ...
    vidshift = find_video_offset(raw)
```
```python
for raw_trial_idx in valid_trials.tolist():
    inp = time_axis[np.newaxis, :].astype(np.float32)
```

iii. The notes do not foreground these repetitions, but they are visible in the implementation and are not required by the task.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader reads several raw fields that never affect the exported dataset, such as cluster `tm`, `site/channel`, `lickL`, and `lickR`. It also tracks per-unit brain-region strings session-by-session, but later collapses the exported dataset to `brain_regions = ["ALM"]` with all-zero `brain_region_idx`. The code also computes threshold metadata used only for notes/plots rather than the decoder input/output arrays.

ii.
```python
"lickL": [h5_read_cell_numeric_1d(f, ev["lickL"], i) for i in range(ntrials)],
"lickR": [h5_read_cell_numeric_1d(f, ev["lickR"], i) for i in range(ntrials)],
```
```python
units.append(
    {
        "quality": quality,
        "tm": tm,
        "trialtm": trialtm,
        "trial": trial,
        "site": int(site[0]) if site.size else -1,
    }
)
```
```python
unit_regions = [unit_regions[i] for i in keep_unit_idx.tolist()]
...
brain_regions = ["ALM"]
...
"brain_region_idx": [np.zeros(sess["neural"][0].shape[0], dtype=np.int64) for sess in processed_sessions],
```

iii. The notes mainly justify the extra loading as part of making the mixed MATLAB schemas uniform, but these specific values are not used downstream in the saved decoder dataset.
