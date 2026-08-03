# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent did not load all available paper sessions. It hard-coded a 12-session Figure 8 context-task roster, all from `/app/data/Ephys_Behavior`, and loaded each session's `data_structure_<animal>_<date>.mat` with direct `h5py` access. Motion energy was loaded separately from `motionEnergy_<animal>_<date>.mat` with `scipy.io.loadmat`.

ii.
```python
DATA_DIR = ROOT / "data" / "Ephys_Behavior"

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]
...
with h5py.File(spec.data_path, "r") as f:
    ...
raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
```

iii. `CONVERSION_NOTES.md` says the agent chose the two-context ALM cohort because the decoder needed behavioral context, and that the roster came from the Figure 8 MATLAB pipeline plus loader files.

## 1-b. How are the data split into subjects (mice)?

i. Each `SessionSpec` carries an `animal` string, and that field becomes the subject id. Subjects are collected in first-seen order with `OrderedDict.fromkeys`, not sorted.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int
...
subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
...
"subject": spec.animal,
```

iii. The notes justify this as preserving the subject ids present in the packaged files, even though the paper text reports fewer mice than the resulting subject-id count.

## 1-c. How are the data split into sessions?

i. One `SessionSpec` entry is one session. Each hard-coded `(animal, date, probe)` tuple becomes one element of `neural`, `input`, and `output`.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
]
...
sessions = [load_session(spec) for spec in SESSION_SPECS]
...
"neural": [sess["neural"] for sess in sessions],
"input": [sess["input"] for sess in sessions],
"output": [sess["output"] for sess in sessions],
```

iii. The stated justification is that Figure 8's context-task roster is the relevant analysis subset.

## 1-d. How are the data split into trials?

i. Trials are represented by the raw Bpod per-trial vectors in `bp`. The code builds a boolean inclusion mask over all trial indices and then uses `np.flatnonzero` to obtain the kept trial numbers. Neural and output arrays are later constructed by iterating over those trial indices.

ii.
```python
bp = {
    "Ntrials": int(round(read_h5_scalar(bp_group["Ntrials"]))),
    "hit": read_h5_vector(bp_group["hit"]).astype(np.int16),
    ...
}
...
included_mask = trial_selector(bp)
included_trials = np.flatnonzero(included_mask)
...
for col, tr in enumerate(included_trials):
    spike_mask = session_trial == tr
```

iii. There is no separate explicit justification beyond mirroring the session object's per-trial arrays and then filtering them.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only hit-or-miss trials and drops early-lick and photostimulation trials. Ignore/no-response trials are excluded entirely. It does not implement the reference's extra drop of trials occurring after the recording stopped.

ii.
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)
```

iii. `CONVERSION_NOTES.md` says ignore/no trials were excluded because the requested outcome target was binary correct/incorrect and because the paper omitted ignore trials from analyses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the chosen probe's cluster table: spike trial ids (`trial`), spike times relative to trial start (`trialtm`), and cluster quality labels (`quality`). `bp.ev.goCue` provides the alignment event.

ii.
```python
clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
qds = clu_group["quality"]
...
trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
...
aligned = trialtm - align_times[session_trial]
```

iii. The notes say the converter follows the reference context-analysis pipeline's spike binning and go-cue alignment.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the code aligns spikes to go cue, bins them in 10 ms bins over `[-3.0, 2.5]` s, converts counts to rates, and smooths with a causal half-Gaussian implemented by `my_smooth`. It also computes a condition-averaged smoothed PSTH to decide whether the unit passes the low-rate filter.

ii.
```python
DT = 1 / 100
SMOOTH = 15
...
counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
psth = my_smooth(counts / trix.size / DT, SMOOTH, "reflect")
...
out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. The notes justify this as reproducing the Figure 8 pipeline: `dt = 1/100`, `[-3.0, 2.5]`, and `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent drops clusters whose lowercase quality string is in `{"garbage", "gabrga", "noisy", "real?"}` and then keeps only units whose condition-averaged smoothed PSTH has mean firing rate above 1 Hz. It does not drop `poor` units.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR = 1.0
...
quality = read_h5_string(f, h5_ref_at(qds, clu_idx)).strip().lower()
if quality in BAD_QUALITIES:
    continue
...
mean_fr = compute_psth_mean_fr(...)
if mean_fr <= LOW_FR:
    continue
```

iii. `CONVERSION_NOTES.md` cites the repository's `findClusters(..., {'all'})` behavior and the Figure 8 scripts' `> 1 Hz` filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go-cue time of the spike's trial: `aligned = trialtm - goCue[trial]`.

ii.
```python
session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
trialtm = np.asarray(trialtm_ds[()], dtype=np.float64).reshape(-1)
aligned = trialtm - align_times[session_trial]
```

iii. The notes explicitly say the alignment event is `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 ms bins (`DT = 1/100`) across a 5.5 s window, yielding 550 time points. Neural and movement streams are put directly onto that 10 ms grid; there is no later rebinning step.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
...
edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. The notes justify 10 ms as matching the Figure 8 context scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input time series is not read from a raw data field. It is synthesized from the chosen alignment window and bin size, implicitly centered on raw `goCue` because all streams are aligned to that event.

ii.
```python
def build_edges_and_time() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    ...
    time = edges[:-1] + DT / 2
    return edges, time
...
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The only stated justification is that the decoder input is time from go cue, so the code exports the shared bin centers.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes bin edges from `TMIN`, `TMAX`, and `DT`, converts them to bin centers, and repeats the resulting 1D time axis for every kept trial.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
if not np.isclose(edges[-1], TMAX):
    edges = np.append(edges, TMAX)
time = edges[:-1] + DT / 2
...
np.asarray(time[None, :], dtype=np.float32)
```

iii. No deeper justification is given beyond exporting the common aligned time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the neural time grid. The same `edges` and `time` arrays are used to bin spikes and to populate the per-trial input.

ii.
```python
edges, time = build_edges_and_time()
...
counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
...
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The alignment is implied by sharing one grid for all exported streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp["hit"]` and `bp["R"]` on the already filtered hit/miss trials. `bp["miss"]` is only used indirectly by the trial filter.

ii.
```python
def lick_direction_from_trial(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
    if hit:
        return int(is_right_trial)
    return int(not is_right_trial)
```

iii. The notes say the agent exported actual response direction rather than instructed side, using hit/miss outcome plus trial side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. On a hit trial, the code returns the instructed side (`R` as 1, not-`R` as 0). On a miss trial, it returns the opposite side. Ignore trials are not represented because they were filtered out earlier.

ii.
```python
if hit:
    return int(is_right_trial)
return int(not is_right_trial)
...
np.full(tongue_speed.shape[0], lick_dir, dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` explicitly describes this "actual response direction" logic and ties the omission of ignore trials to the binary target request.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from the Bpod `autowater` flag.

ii.
```python
"autowater": read_h5_vector(bp_group["autowater"]).astype(np.int16),
...
context = 0 if bp["autowater"][tr] else 1
```

iii. The notes explain this as `WC = 0` for autowater trials and `DR = 1` otherwise.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It is a direct binary relabeling: `autowater == 1` becomes `0` (`WC`), else `1` (`DR`).

ii.
```python
context = 0 if bp["autowater"][tr] else 1
...
np.full(tongue_speed.shape[0], context, dtype=np.int16)
```

iii. The notes state this mapping explicitly.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["hit"]` on the already filtered hit/miss trials. Miss trials are identified implicitly as the remaining included trials.

ii.
```python
included_mask = trial_selector(bp)
...
outcome = 1 if bp["hit"][tr] else 0
```

iii. The notes justify binary outcome by saying the requested decoder target was `incorrect` versus `correct`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as `1` for hits and `0` otherwise. Because ignore trials were removed earlier, "otherwise" effectively means miss.

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
...
np.full(tongue_speed.shape[0], outcome, dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` explicitly says ignore/no-response trials were excluded so that exported outcome stayed binary.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from bottom-camera DeepLabCut features `top_tongue` and `bottom_tongue`, their `frameTimes`, and the video/behavior alignment fields `bp.ev.bitStart`, `sglx.bitcode.bitstart`, `sglx.fs`, and `bp.ev.goCue`.

ii.
```python
feat_indices = {
    "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
    "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
    ...
}
...
vidshift = find_video_offset(bp["bitStart"], bitcode_bitstart, fs)
...
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The notes say the agent used "the same aligned DLC trajectories" and then reduced them to a bottom-view tongue-tip speed.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code interpolates `top_tongue` and `bottom_tongue` positions onto the 10 ms neural grid, takes discrete gradients of x and y on that grid, averages the two tongue features' x- and y-velocities, converts the result to speed magnitude, and replaces NaNs with zeros. It does not use a likelihood cutoff, per-run smoothing, or side-view normalization.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
...
xvel[:, tr] = np.gradient(tsinterp[:, 0])
yvel[:, tr] = np.gradient(tsinterp[:, 1])
...
top_tongue_xvel, top_tongue_yvel = velocities["top_tongue"]
bottom_tongue_xvel, bottom_tongue_yvel = velocities["bottom_tongue"]
tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
```

iii. The trajectory says the main design choice was collapsing multiple tongue landmarks into one scalar; the notes say this was chosen as the least arbitrary decoder-friendly reduction.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The threshold is the 50th percentile of strictly positive tongue-speed bins from the exported session. Each bin is then labeled `0/1` by whether it falls below or above that threshold.

ii.
```python
positive_tongue = tongue_use[tongue_use > 0]
if positive_tongue.size:
    tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
else:
    tongue_thresh = 0.0
...
(tongue_speed[:, tr] >= tongue_thresh).astype(np.int16)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this non-reference choice as preventing the tongue target from collapsing to a constant all-ones label after zero-filling invisible bins.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by a session-wide video offset, shifted by each trial's go cue, and interpolated directly onto the same 10 ms grid used for neural data.

ii.
```python
def find_video_offset(bit_start: np.ndarray, bitcode_bitstart: np.ndarray, fs: float) -> float:
    return matlab_mode(bitcode_bitstart) / fs - matlab_mode(bit_start)
...
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The notes say this matches `findVideoOffset.m`, and the trajectory repeatedly describes "aligned DLC trajectories" on the neural time axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DeepLabCut features `top_paw` and `bottom_paw`, plus the same frame-time and video-offset fields used for tongue alignment.

ii.
```python
feat_indices = {
    ...
    "top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
    "bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
}
```

iii. The notes explicitly say paw speed was computed as the mean speed of `top_paw` and `bottom_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, the code interpolates x/y position to the 10 ms grid, takes discrete gradients, subtracts a per-trial median derivative baseline for non-tongue features, fills missing values by nearest-neighbor interpolation, converts each paw's x/y velocity to speed, averages the two paw speeds, and zero-fills any remaining NaNs.

ii.
```python
xvel[:, tr] = np.gradient(tsinterp[:, 0])
yvel[:, tr] = np.gradient(tsinterp[:, 1])
if "tongue" not in feat_name:
    xvel[:, tr] = xvel[:, tr] - basederiv[0]
    yvel[:, tr] = yvel[:, tr] - basederiv[0]
    xvel[:, tr] = fill_nearest_1d(xvel[:, tr])
    yvel[:, tr] = fill_nearest_1d(yvel[:, tr])
...
top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The trajectory says the agent chose to collapse multiple paw landmarks into one decoder scalar instead of trying to keep the full tracking geometry.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The threshold is the per-session median of all exported paw-speed bins. Each time bin is then labeled `0/1` by whether it is below or above that threshold.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_use, 50))
...
(paw_speed[:, tr] >= paw_thresh).astype(np.int16)
```

iii. The notes describe this as the literal per-session 50th percentile over exported bins.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The paw trajectories are aligned exactly like the tongue trajectories: video-offset correction, subtraction of per-trial go cue, then interpolation onto the neural time axis.

ii.
```python
vidshift = find_video_offset(bp["bitStart"], bitcode_bitstart, fs)
...
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The notes group paw and tongue together as "aligned DLC trajectories used by the repository's kinematics code."

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file's `me.data` per-trial traces, together with side-camera `frameTimes` and the same offset/alignment variables used for video streams.

ii.
```python
def load_motion_energy(path: Path) -> tuple[np.ndarray, float]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
    return trial_arrays, float(me.moveThresh)
...
frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
```

iii. While one trajectory update incorrectly said motion energy was embedded in the session object, the final notes and the code both use the standalone motion-energy file.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code interpolates each per-frame motion-energy trace onto the 10 ms neural grid and then nearest-fills NaN gaps to produce a dense continuous signal before thresholding.

ii.
```python
out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
out[:, tr] = fill_nearest_1d(out[:, tr])
```

iii. `CONVERSION_NOTES.md` says this was meant to match `loadMotionEnergy.m` and preserve a usable aligned motion-energy trace.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The threshold is the per-session median over all exported motion-energy bins. Each bin becomes `0/1` depending on whether it is below or above that threshold.

ii.
```python
me_thresh = float(np.nanpercentile(me_use, 50))
...
(motion_energy[:, tr] >= me_thresh).astype(np.int16)
```

iii. The notes describe this as literal per-session median binarization requested by the decoder spec.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The side-camera frame times are corrected by the session's video offset, shifted by trial go cue, and interpolated onto the common neural time axis.

ii.
```python
frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
...
out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
```

iii. The notes explicitly mention video-offset correction followed by alignment to the neural time axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data are usually filled rather than preserved as missing. Motion energy is nearest-filled after interpolation; non-tongue positions and velocities are nearest-filled; all-NaN vectors become zeros; tongue NaNs are explicitly replaced by zero speed. A trial with missing `NdroppedFrames` is skipped for that feature, which then remains NaN until later filling/zeroing.

ii.
```python
def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    ...
    if idx.size == 0:
        return np.zeros_like(x)
...
out[:, tr] = fill_nearest_1d(out[:, tr])
...
if np.isnan(ndropped):
    continue
...
xvel[:, tr] = fill_nearest_1d(xvel[:, tr])
yvel[:, tr] = fill_nearest_1d(yvel[:, tr])
...
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The only explicit justification is the tongue-threshold note: zero-filling invisible tongue bins made the raw median unusable, which is why the threshold was changed to positive bins only.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are the repeated HDF5 reads plus the nested per-trial/per-feature interpolation and gradient calculations for movement streams, followed by the per-cluster histogramming and smoothing used both for low-FR filtering and trial-wise neural matrices.

ii.
```python
for tr in range(n_trials):
    frame_times = read_frame_times(f, view_group, tr)
    xy = read_feature_xy(f, view_group, tr, feat_idx)
    interp_xy = interp_with_nan(...)
...
for clu_idx in range(qds.shape[0]):
    ...
    mean_fr = compute_psth_mean_fr(...)
    ...
    neural_by_trial.append(build_neural_trials(...))
```

iii. The agent did not explicitly profile runtime in the notes; this follows from the code structure and from the saved full-conversion run.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are left scalarized: per-trial loops in `aligned_motion_energy`, `aligned_feature_position`, `aligned_feature_velocity`, `build_neural_trials`, and `derive_session_outputs`; and the per-cluster loop that calls both `compute_psth_mean_fr` and `build_neural_trials`.

ii.
```python
for tr in range(n_trials):
    ...
for tr in range(xpos.shape[1]):
    ...
for col, tr in enumerate(included_trials):
    ...
for clu_idx in range(qds.shape[0]):
    ...
for tr in included_trials:
    ...
```

iii. There is no explicit justification in the notes; the code simply leaves these loops in place.

## 11-c. What processing does the code repeat multiple times?

i. The code rereads or recomputes several session-wide items multiple times: feature names are scanned trial-by-trial in `find_feat_index`; frame times are read once for motion energy and again for every aligned feature; cluster spikes are histogrammed once for the mean-FR filter and again for every kept trial matrix.

ii.
```python
def find_feat_index(...):
    for trial_idx in range(n_trials):
        names = read_feat_names(f, view_group, trial_idx)
...
frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
...
frame_times = read_frame_times(f, view_group, tr)
...
mean_fr = compute_psth_mean_fr(...)
...
neural_by_trial.append(build_neural_trials(...))
```

iii. The agent did not document these repeats as deliberate; they are evident from the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items are computed or loaded but never used downstream: `SIDE_FEATURES` and `BOTTOM_FEATURES`; Bpod fields `L`, `sample`, and `delay`; `raw_move_thresh`; and bookkeeping lists/counters such as `kept_cluster_indices`, `prefilter_unit_count`, and `cluster_ids`.

ii.
```python
SIDE_FEATURES = [...]
BOTTOM_FEATURES = [...]
...
"L": read_h5_vector(bp_group["L"]).astype(np.int16),
"sample": read_h5_vector(ev_group["sample"]),
"delay": read_h5_vector(ev_group["delay"]),
...
raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
...
kept_cluster_indices = []
prefilter_unit_count = 0
cluster_ids = []
```

iii. No justification was given for these extra reads or temporary structures.
