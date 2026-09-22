# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 12-session cohort and loads only `data_structure_<subject>_<date>.mat` files from `/app/data/Ephys_Behavior`. It uses `h5py` for those session files and `scipy.io.loadmat` only for the matching motion-energy file. It does not discover sessions from both task folders and does not implement the reference solution's mixed v7.3/v5 session loader.

ii. 
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

SESSIONS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    ...
    {"subject": "JEB19", "date": "2023-04-18", "probe": 1},
]
...
data_path = DATA_DIR / f"data_structure_{subject}_{date}.mat"
with h5py.File(data_path, "r") as h5file:
    ...
me_raw = loadmat(me_path, simplify_cells=True)["me"]["data"]
```

iii. In the trajectory, the AI said it had "pinned down" a published 12-session DR+WC ALM-video cohort and explicitly planned to restrict the converter to that set. It also described the implementation as a self-contained HDF5 reader plus a v5 motion-energy loader.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the hard-coded `subject` field in each `SESSIONS` entry. During assembly, subjects are added in first-seen order and `subject_idx` records the index assigned to each session.

ii. 
```python
SESSIONS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    ...
]
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
...
"subjects": subjects,
"subject_idx": np.asarray(all_subject_idx, dtype=np.int64),
```

iii. The trajectory justification was that the converter should follow the published session cohort, so the subject identities were carried in the session list itself. There was no separate discussion of sorting or parsing the subject name from filenames.

## 1-c. How are the data split into sessions?

i. One hard-coded dictionary entry in `SESSIONS` becomes one session. The script loops over those 12 entries, converts each one independently, and appends one element per session to `neural`, `input`, and `output`.

ii. 
```python
def convert_session(session_info: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    subject = str(session_info["subject"])
    date = str(session_info["date"])
    probe_num = int(session_info["probe"])
    data_path = DATA_DIR / f"data_structure_{subject}_{date}.mat"
    ...

for session_info in SESSIONS:
    session_data, session_meta = convert_session(session_info)
    all_neural.append(session_data["neural"])
    all_input.append(session_data["input"])
    all_output.append(session_data["output"])
```

iii. In the trajectory, the AI repeatedly said it was using "the first 12 `Ephys_Behavior` ALM-video sessions" as the DR+WC cohort and treating those as the relevant sessions for this task.

## 1-d. How are the data split into trials?

i. Trials are defined by array position in the Bpod fields. The script builds a boolean `keep_mask` from per-trial fields such as `stim/enable` and `early`, converts that to integer trial indices with `np.flatnonzero`, and uses those indices everywhere else (`go_cue`, neural spike assignment, behavior labels, and video outputs).

ii. 
```python
stim_enable = as_bool_1d(bp["stim/enable"])
early = as_bool_1d(bp["early"])
autolearn = (
    as_bool_1d(bp["autolearn"]) if "autolearn" in bp.keys() else np.zeros_like(early, dtype=bool)
)
keep_mask = (~stim_enable) & (~early) & (~autolearn)
keep_trials = np.flatnonzero(keep_mask)
...
trial_lookup = np.full(go_cue.size, -1, dtype=np.int32)
trial_lookup[keep_trials] = np.arange(keep_trials.size, dtype=np.int32)
```

iii. The trajectory did not give a separate trial-splitting rationale beyond saying the code would work from filtered control trials and align everything to `goCue`.

## 1-e. How are trials filtered based on quality controls?

i. The AI removes photostimulation trials, early-lick trials, and additionally `autolearn` trials when the field exists. It does not implement the reference solution's extra removal of behavior trials that extend past the end of the recording.

ii. 
```python
stim_enable = as_bool_1d(bp["stim/enable"])
early = as_bool_1d(bp["early"])
autolearn = (
    as_bool_1d(bp["autolearn"]) if "autolearn" in bp.keys() else np.zeros_like(early, dtype=bool)
)
keep_mask = (~stim_enable) & (~early) & (~autolearn)
keep_trials = np.flatnonzero(keep_mask)
```

iii. In its plan and final summary, the AI justified this as keeping only "control/non-early trials" and explicitly described the trial filter as `~stim.enable & ~early`, with `~autolearn` added if present.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the selected probe's cluster table inside `obj/clu`, specifically `quality`, `trial`, and `trialtm`, together with `bp.ev.goCue` for alignment.

ii. 
```python
clu_group = h5file[h5file["obj/clu"][probe_num - 1, 0]]
qualities = clu_group["quality"]
trials = clu_group["trial"]
trial_times = clu_group["trialtm"]
...
aligned_times = clu_trial_times - go_cue[clu_trials]
```

iii. The trajectory said the core neural preprocessing was "spikes aligned to `goCue`" with published unit filtering, which is consistent with these fields being the raw inputs.

## 2-b. How is the `neural` data processed?

i. For each kept unit, spikes are aligned to `goCue`, restricted to a `[-3.0, 2.5)` s window, counted into 5 ms bins, converted to firing rates by dividing by `DT`, and then smoothed with a custom causal-looking Gaussian convolution that uses leading reflected padding.

ii. 
```python
TMIN = -3.0
TMAX = 2.5
DT = 1.0 / 200.0
...
bin_idx = np.floor((aligned_times - TMIN) / DT).astype(np.int32)
counts = np.zeros((keep_trials.size, NT), dtype=np.float64)
if mapped_trials.size:
    np.add.at(counts, (mapped_trials, bin_idx), 1.0)

rates = smooth_causal_reflect(counts / DT).astype(np.float32)
```

iii. The trajectory explicitly said the neural pipeline would recreate "goCue alignment, 5 ms binning on `[-3.0, 2.5]` s, causal Gaussian smoothing with reflected boundary." That is the clearest statement of the AI's intended processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code drops clusters whose `quality` string is exactly one of `garbage`, `gabrga`, `noisy`, or `real?`, then removes units whose mean smoothed firing rate is not above 1 Hz. It does not lowercase the label and does not exclude `poor`.

ii. 
```python
for clu_idx in range(qualities.shape[0]):
    quality = str(read_matlab_any(h5file, qualities[clu_idx, 0]) or "").strip()
    if quality in {"garbage", "gabrga", "noisy", "real?"}:
        continue
    ...
    mean_fr = float(rates.mean())
    if mean_fr > LOW_FR_HZ:
        kept_unit_data.append(rates)
```

iii. In the trajectory, the AI justified this as following the paper's published unit filter, describing it as "garbage/noisy clusters removed" and "FR > 1 Hz." The final code only partially matches that broader description.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the `goCue` time of the spike's own trial from `trialtm`. No interpolation or cross-stream offset correction is applied to the neural data.

ii. 
```python
clu_trials = as_1d_float(read_matlab_any(h5file, trials[clu_idx, 0])).astype(np.int64) - 1
clu_trial_times = as_1d_float(read_matlab_any(h5file, trial_times[clu_idx, 0]))
aligned_times = clu_trial_times - go_cue[clu_trials]
```

iii. The trajectory repeatedly described neural preprocessing as "aligned to `goCue`," and did not mention any extra correction for the neural stream.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 5 ms bins (`DT = 1/200`). The binning grid spans `[-3.0, 2.5]`, yielding 1100 bins. There is no additional neural rebinning beyond assigning spikes to that grid and smoothing the resulting rate trace.

ii. 
```python
DT = 1.0 / 200.0
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
NT = TIME_BINS.size
```

iii. The trajectory explicitly justified the choice as "5 ms binning on `[-3.0, 2.5]` s" to match what it believed the paper used.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw data field. It is constructed from the script's global time grid `TIME_BINS`, which is intended to represent time relative to the `goCue` alignment event.

ii. 
```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
...
input_trials = [TIME_BINS[np.newaxis, :].astype(np.float32) for _ in range(keep_trials.size)]
```

iii. The trajectory treated this as a decoder-format choice built on top of the `goCue`-aligned time grid rather than as a field loaded from the source files.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The processing is simply constructing a 1D vector of bin centers from `TMIN`, `TMAX`, and `DT`, then reusing that same vector for every trial in the session.

ii. 
```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
...
input_trials = [TIME_BINS[np.newaxis, :].astype(np.float32) for _ in range(keep_trials.size)]
```

iii. The trajectory did not give a separate justification beyond saying the converter would align everything to `goCue` on a common 5 ms grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same global time grid that the neural spike counts are binned onto, so alignment is by construction: each input bin index corresponds to the same `TIME_BINS` interval used for the neural rates.

ii. 
```python
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
...
bin_idx = np.floor((aligned_times - TMIN) / DT).astype(np.int32)
...
input_trials = [TIME_BINS[np.newaxis, :].astype(np.float32) for _ in range(keep_trials.size)]
```

iii. The trajectory said the converter would put neural and video streams onto a common 5 ms grid aligned to `goCue`; the input is that grid's bin centers.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp.R`, `bp.L`, `bp.hit`, and `bp.miss`. The code also reads `bp.no`, although it does not use it directly in the final computation.

ii. 
```python
r = as_bool_1d(bp["R"])
l = as_bool_1d(bp["L"])
hit = as_bool_1d(bp["hit"])
miss = as_bool_1d(bp["miss"])
no = as_bool_1d(bp["no"])
```

iii. The trajectory explicitly called out the ambiguity around whether `R/L` reflected instructed side or actual response and said it was checking the authors' labeling before finalizing lick direction. The resulting code treats `R/L` together with `hit/miss` as the ingredients for the actual lick choice.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code derives actual choice from instructed side and trial outcome. A right choice is `(R & hit) | (L & miss)`, a left choice is `(L & hit) | (R & miss)`, and any remaining kept trial stays in the `none` class.

ii. 
```python
lick_direction = np.full(keep_trials.size, 2, dtype=np.int64)
right_choice = (r & hit) | (l & miss)
left_choice = (l & hit) | (r & miss)
lick_direction[left_choice[keep_trials]] = 0
lick_direction[right_choice[keep_trials]] = 1
```

iii. In the trajectory, the AI said it needed to resolve whether `R/L` was instruction or response because that determined how to build the requested lick-direction output. This code reflects the "actual choice" interpretation.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp.autowater` field.

ii. 
```python
autowater = as_bool_1d(bp["autowater"])
```

iii. The trajectory described context as coming from `autowater`, consistent with the paper's WC versus DR task structure.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code directly relabels `autowater` trials as `0` (`WC`) and all other kept trials as `1` (`DR`).

ii. 
```python
context = np.where(autowater[keep_trials], 0, 1).astype(np.int64)  # WC, DR
```

iii. The trajectory justified this as deriving context from `autowater` for the alternating DR/WC cohort.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial `bp.hit` and `bp.miss` flags. The script also reads `bp.no` but does not need it because the default class is `ignore`.

ii. 
```python
hit = as_bool_1d(bp["hit"])
miss = as_bool_1d(bp["miss"])
no = as_bool_1d(bp["no"])
```

iii. The trajectory described outcome as coming from `hit/miss/no`, but the implemented code only uses `hit` and `miss` explicitly and infers `ignore` as the fallback.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code assigns `0` to miss trials (`incorrect`), `1` to hit trials (`correct`), and leaves all other kept trials as `2` (`ignore`).

ii. 
```python
outcome = np.full(keep_trials.size, 2, dtype=np.int64)
outcome[miss[keep_trials]] = 0
outcome[hit[keep_trials]] = 1
```

iii. The trajectory justification was that the decoder outputs needed explicit categorical values for incorrect, correct, and ignore outcomes.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from tracked tongue feature coordinates in `obj.traj`, specifically `featNames`, `ts`, and `frameTimes`, plus `bp.ev.goCue` and the session-wide video/behavior offset computed from `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, and `bp.ev.bitStart`. The code searches side-view features `["tongue", "left_tongue", "right_tongue"]` and bottom-view features `["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]`.

ii. 
```python
feat_names = np.asarray(read_matlab_any(h5file, view_group["featNames"][trial_idx, 0])).reshape(-1)
ts = np.asarray(read_matlab_any(h5file, view_group["ts"][trial_idx, 0]), dtype=np.float64)
frame_times = as_1d_float(read_matlab_any(h5file, view_group["frameTimes"][trial_idx, 0]))
frame_times = frame_times - video_shift - align_time
...
side_tongue_speed, side_tongue_visible, _ = get_feature_positions(..., ["tongue", "left_tongue", "right_tongue"], ...)
bottom_tongue_speed, bottom_tongue_visible, _ = get_feature_positions(..., ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"], ...)
```

iii. The trajectory justified tongue velocity as coming from the video streams aligned to `goCue`, and described the implementation as preserving visibility states while resampling video onto the neural grid.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code linearly interpolates visible tongue positions onto the neural time grid within contiguous finite segments, marks bins as visible when interpolated `x/y` are finite, fills missing edges by nearest value for the purpose of taking derivatives, computes speed from `np.gradient` on the interpolated positions, averages side and bottom tongue speeds over bins where either view is visible, and does not normalize the two views to a common scale before averaging.

ii. 
```python
interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
visible = np.all(np.isfinite(interp_xy), axis=1)

filled_xy = interp_xy.copy()
filled_xy[:, 0] = fill_nearest_1d(filled_xy[:, 0])
filled_xy[:, 1] = fill_nearest_1d(filled_xy[:, 1])
...
vel = np.gradient(filled_xy, axis=0)
speed = np.sqrt((vel**2).sum(axis=1))
...
tongue_speed[out_trial_idx], tongue_visible[out_trial_idx] = average_over_visible(
    tongue_speeds, tongue_vis
)
```

iii. In the trajectory, the AI justified this generally as "resampling the video streams onto the 5 ms neural grid" while preserving `not_visible` states. It did not separately justify omitting the likelihood cutoff or view normalization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After the full session's tongue-speed matrix is computed, the code takes the 50th percentile across visible tongue-speed bins and uses that as the session threshold. Visible bins below threshold are class `0`, visible bins at or above threshold are class `1`, and invisible bins remain class `2`.

ii. 
```python
tongue_threshold = float(np.nanpercentile(tongue_speed[tongue_visible], 50))
...
tongue_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
tongue_disc[tongue_visible & (tongue_speed < tongue_threshold)] = 0
tongue_disc[tongue_visible & (tongue_speed >= tongue_threshold)] = 1
```

iii. The trajectory explicitly said the movement outputs would use per-session median discretization while preserving `not_visible` / `no_video` categories.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code first aligns camera frame times to the behavior clock by subtracting a session-wide video shift and then subtracting that trial's `goCue`. It then interpolates tongue positions directly onto the neural `TIME_BINS` grid, so the final tongue-speed trace is time-locked to the same bins as the neural data.

ii. 
```python
def get_video_shift(h5file: h5py.File, bit_start: np.ndarray) -> float:
    bitcode_start = as_1d_float(h5file["obj/sglx/bitcode/bitstart"])
    fs = float(np.asarray(h5file["obj/sglx/fs"]).squeeze())
    return float(np.nanmedian(bitcode_start / fs) - np.nanmedian(bit_start))

frame_times = frame_times - video_shift - align_time
interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
```

iii. The trajectory justified the video alignment as following the paper's video-offset correction and said the expensive part of the conversion was resampling the video streams onto the 5 ms neural grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-view paw coordinates in `obj.traj`. The code searches both `top_paw` and `bottom_paw` features in the bottom-view camera, using the same frame-time and offset machinery as for the tongue.

ii. 
```python
paw_speed_trial, paw_visible_trial, _ = get_feature_positions(
    h5file,
    bottom_view,
    int(trial_idx),
    ["top_paw", "bottom_paw"],
    align_time,
    video_shift,
)
```

iii. The trajectory grouped paw velocity with the other video outputs and did not separately justify using both paw features rather than a single one.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The processing is the same helper pipeline used for tongue features: interpolate visible coordinate segments onto `TIME_BINS`, fill missing edges by nearest value before taking derivatives, compute speed from `np.gradient`, average over whichever paw feature is visible, and then discretize later at a session-wide median.

ii. 
```python
interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
visible = np.all(np.isfinite(interp_xy), axis=1)
...
vel = np.gradient(filled_xy, axis=0)
baseline_deriv = np.nanmedian(np.diff(filled_xy, axis=0), axis=0)
if "tongue" not in feature_name:
    vel[:, 0] = vel[:, 0] - baseline_deriv[0]
    vel[:, 1] = vel[:, 1] - baseline_deriv[0]
speed = np.sqrt((vel**2).sum(axis=1))
```

iii. The trajectory justification was only the general claim that the script would discretize tongue speed, paw speed, and motion energy after resampling them onto the neural grid.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The code computes the 50th percentile over visible paw-speed bins for the session and uses it to split bins into `0` (below threshold), `1` (at or above threshold), and `2` (not visible).

ii. 
```python
paw_threshold = float(np.nanpercentile(paw_speed[paw_visible], 50))
...
paw_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
paw_disc[paw_visible & (paw_speed < paw_threshold)] = 0
paw_disc[paw_visible & (paw_speed >= paw_threshold)] = 1
```

iii. The trajectory explicitly said the movement outputs would use per-session median thresholds with preserved missing-data categories.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Like tongue velocity, paw coordinates are placed on the behavior clock by subtracting the session video shift and the trial's `goCue`, then interpolated onto the neural bin centers `TIME_BINS`.

ii. 
```python
frame_times = frame_times - video_shift - align_time
interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
```

iii. The trajectory treated paw alignment as part of the same general video-to-neural resampling pipeline as tongue and motion energy.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from a separate `motionEnergy_<subject>_<date>.mat` file, specifically from `loadmat(... )["me"]["data"]`, and combined with side-camera frame times plus the same video-shift and `goCue` alignment variables.

ii. 
```python
me_path = DATA_DIR / f"motionEnergy_{subject}_{date}.mat"
...
me_raw = loadmat(me_path, simplify_cells=True)["me"]["data"]
...
frame_times = as_1d_float(read_matlab_any(h5file, side_view["frameTimes"][trial_idx, 0]))
frame_times = frame_times - video_shift - align_time
```

iii. The trajectory explicitly discussed a "motion-energy fallback" and described motion energy as one of the video-derived outputs aligned onto the neural grid.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each kept trial, the code aligns side-camera frame times, optionally fabricates fallback frame times if the recorded `frameTimes` are unusable, linearly interpolates the motion-energy trace onto `TIME_BINS`, fills edge gaps with nearest values, records which bins ended up finite, and then later discretizes by a session-wide median threshold.

ii. 
```python
if (
    frame_times.size != me_trial.size
    or frame_times.size < 2
    or np.isfinite(frame_times).sum() < 2
):
    side_ts = np.asarray(read_matlab_any(h5file, side_view["ts"][trial_idx, 0]), dtype=np.float64)
    side_ts = np.transpose(side_ts, (2, 1, 0))
    fallback_times = np.arange(1, side_ts.shape[0] + 1, dtype=np.float64) / 400.0
    fallback_times = fallback_times - 0.5 - align_time
    if fallback_times.size == me_trial.size and fallback_times.size >= 2:
        frame_times = fallback_times

aligned_me = interp_motion(frame_times, me_trial, TIME_BINS)
available = np.isfinite(aligned_me)
motion_energy[out_trial_idx] = aligned_me
motion_available[out_trial_idx] = available
```

iii. The trajectory justification focused on removing spurious `no_video` cases. It said the MATLAB loader had a fallback for frame-time mismatches and that the AI was mirroring that behavior.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The code takes the 50th percentile over all available motion-energy bins within a session. Available bins below threshold become `0`, available bins at or above threshold become `1`, and unavailable bins remain `2`.

ii. 
```python
me_threshold = float(np.nanpercentile(motion_energy[motion_available], 50))
...
me_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
me_disc[motion_available & (motion_energy < me_threshold)] = 0
me_disc[motion_available & (motion_energy >= me_threshold)] = 1
```

iii. The trajectory explicitly justified this as a per-session median split with a dedicated `no_video` state.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session video shift and the trial `goCue` from side-camera frame times, then interpolating the per-frame motion-energy trace directly onto the neural `TIME_BINS` grid. If the recorded frame times are missing or inconsistent, the code can substitute synthetic equally spaced frame times.

ii. 
```python
frame_times = frame_times - video_shift - align_time
...
aligned_me = interp_motion(frame_times, me_trial, TIME_BINS)
```

iii. The trajectory justification was that the converter should follow the paper's video-offset logic and use the MATLAB-style fallback when bad frame times would otherwise create artificial `no_video` bins.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally keeps trials and tries to repair or route around missing video timing. For tracked features, it interpolates within contiguous visible segments, uses nearest-value filling at the edges before taking derivatives, and marks bins with no interpolated coordinates as not visible. For motion energy, it substitutes synthetic frame times when `frameTimes` are missing or invalid and then interpolates onto the neural grid. Trials are not dropped for missing video data unless the upstream filters already remove them for other reasons.

ii. 
```python
def interpolate_visible_segments(src_t: np.ndarray, src_xy: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    ...
    for grp in groups:
        ...
        out[seg_mask, dim] = np.interp(target_t[seg_mask], seg_t, src_xy[grp, dim])

def fill_nearest_1d(y: np.ndarray) -> np.ndarray:
    ...
    y[: good[0]] = y[good[0]]
    y[good[-1] + 1 :] = y[good[-1]]
    return y
...
if (
    frame_times.size != me_trial.size
    or frame_times.size < 2
    or np.isfinite(frame_times).sum() < 2
):
    ...
    frame_times = fallback_times
```

iii. In the trajectory, the AI justified these choices as mirroring the MATLAB fallback for motion energy and as preserving `not_visible` / `no_video` classes while preventing a small number of bad timing vectors from producing spurious missing-data outputs.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming work in this script is the per-session neural and video resampling pipeline: iterating over clusters to bin spikes and smooth rates, and iterating over kept trials and tracked features to interpolate video streams onto the 5 ms grid. The code also reopens each session file a second time inside `compute_video_outputs`, which adds I/O overhead.

ii. 
```python
for clu_idx in range(qualities.shape[0]):
    ...
    np.add.at(counts, (mapped_trials, bin_idx), 1.0)
    rates = smooth_causal_reflect(counts / DT).astype(np.float32)
...
for out_trial_idx, trial_idx in enumerate(keep_trials):
    side_tongue_speed, side_tongue_visible, _ = get_feature_positions(...)
    bottom_tongue_speed, bottom_tongue_visible, _ = get_feature_positions(...)
    paw_speed_trial, paw_visible_trial, _ = get_feature_positions(...)
```

iii. The trajectory explicitly described the conversion as expensive because "each session is rebinning spikes and resampling the video streams onto the 5 ms neural grid."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain non-vectorized: the per-cluster loop in `compute_neural_session`, the per-trial loop in `compute_video_outputs`, the per-feature loop inside `get_feature_positions`, and the per-trial loop in `build_trial_outputs`. Some ragged video structures limit full vectorization, but the implementation still leaves more Python-level looping than the reference solution.

ii. 
```python
for clu_idx in range(qualities.shape[0]):
    ...

for out_trial_idx, trial_idx in enumerate(keep_trials):
    ...

for feature_name in feature_names:
    ...

for trial_idx in range(lick_direction.size):
    outputs.append(np.vstack([...]))
```

iii. The trajectory did not explicitly discuss vectorization. Its only performance commentary was that the resampling-heavy conversion was expected to take time.

## 11-c. What processing does the code repeat multiple times?

i. The script repeats some work. It opens each session `.mat` file once in `convert_session` for neural and behavior extraction and again in `compute_video_outputs` for video processing. It also recreates the same per-trial `TIME_BINS` input array for every kept trial and repeatedly searches feature names inside `get_feature_positions`.

ii. 
```python
with h5py.File(data_path, "r") as h5file:
    ...
    neural_trials, brain_region_idx = compute_neural_session(...)
    lick_direction, context, outcome = compute_behavior_labels(...)

tongue_disc, paw_disc, me_disc, thresholds = compute_video_outputs(session_info, keep_trials)
...
input_trials = [TIME_BINS[np.newaxis, :].astype(np.float32) for _ in range(keep_trials.size)]
```

iii. The trajectory did not identify or justify this repeated work. Its updates treated the implementation as a single end-to-end conversion pass.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some processing whose results are not used downstream: it reads `bp.L` and `bp.no` even though the final lick/outcome logic does not need `no` and only uses `L` indirectly in boolean combinations; `get_feature_positions` returns a third all-ones boolean array that callers ignore; it computes filled coordinates and baseline-corrected velocities even for bins that later remain invisible; and it stores per-session threshold metadata that the decoder does not consume.

ii. 
```python
r = as_bool_1d(bp["R"])
l = as_bool_1d(bp["L"])
hit = as_bool_1d(bp["hit"])
miss = as_bool_1d(bp["miss"])
no = as_bool_1d(bp["no"])
...
return mean_speed, any_visible, np.ones(TIME_BINS.shape, dtype=bool)
...
"thresholds": thresholds,
```

iii. The trajectory did not explicitly justify these extra computations. They appear to be incidental to the implementation rather than deliberate outputs needed by the downstream decoder.
