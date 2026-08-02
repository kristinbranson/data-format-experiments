# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 12-session alternating-context ALM cohort in `SESSION_SPECS`, then iterates over those session specs and opens one MATLAB `data_structure_<animal>_<date>.mat` file per session from `/app/data/Ephys_Behavior`. Within each file it loads behavior, cluster/spike data, trajectory data, and motion-energy data, then converts the session into session-level `neural`, `input`, and `output` trial lists.

ii. ```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    {"animal": "JEB7", "date": "2021-04-29", "probes": [1]},
    ...
]

DATA_DIR = "/app/data"
EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")

def build_session_payload(spec: dict) -> dict:
    data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
    with h5py.File(data_path, "r") as f:
        behavior = load_behavior(f)
        probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
        neural_all, kept_quality, single_flags = bin_spikes_for_session(behavior, probes)
        raw_motion_energy, manual_motion_thresh = load_motion_energy(spec["animal"], spec["date"], behavior)
        motion_energy = align_motion_energy(f, behavior, raw_motion_energy)
        tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
        paw_feature = choose_paw_feature(f)
        paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. In `CONVERSION_NOTES.md`, the AI says it intentionally targeted the “alternating delayed-response / water-cued ALM electrophysiology cohort” and that the session list follows `Figure8a_thru_c.m` and `Figure8d.m`. The trajectory also states it “pinned the reference processing path and the likely target cohort: the 12 alternating-context ALM sessions used in the context analyses.”

## 1-b. How are the data split into subjects?

i. Subjects are defined only by the `animal` string in each hard-coded session spec. The AI builds a unique subject list in first-seen order and stores a session-to-subject index in `subject_idx`.

ii. ```python
def build_dataset() -> tuple[dict, dict]:
    subjects = []
    subject_lookup = OrderedDict()
    ...
    for spec in SESSION_SPECS:
        payload = build_session_payload(spec)
        sid = spec["animal"]
        if sid not in subject_lookup:
            subject_lookup[sid] = len(subject_lookup)
            subjects.append(sid)
        ...
        data["subject_idx"].append(subject_lookup[sid])
```

iii. The justification in `CONVERSION_NOTES.md` is that the cohort is the Figure 8 loader subset. The same notes explicitly acknowledge a mismatch with the methods text: the loader subset yields 12 sessions across 7 animal IDs, while `methods.txt` says 12 sessions from 6 mice. The AI chose to preserve the code-defined IDs rather than reconcile that discrepancy.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_SPECS` is treated as one session. `build_dataset()` loops over the specs and appends one converted session at a time to `data["neural"]`, `data["input"]`, `data["output"]`, and `data["brain_region_idx"]`.

ii. ```python
for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    ...
    data["neural"].append(payload["neural_trials"])
    data["input"].append(payload["input_trials"])
    data["output"].append(payload["output_trials"])
    data["subject_idx"].append(subject_lookup[sid])
    data["brain_region_idx"].append(payload["brain_region_idx"])
```

iii. The AI’s notes say the session list “follows the loader combination used in `code/Scripts/Figure 8/Figure8a_thru_c.m` and `code/Scripts/Figure 8/Figure8d.m`,” so the split is justified as mirroring that analysis cohort rather than discovering sessions dynamically.

## 1-d. How are the data split into trials?

i. Within each session, the AI treats the raw behavior field `Ntrials` as the master trial count. It builds trial-aligned arrays with shape `(time, ntrials)` or `(time, ntrials, ...)`, computes a `use_trials` index from a boolean analysis mask, and then emits one trial object per kept trial into the session lists.

ii. ```python
def bin_spikes_for_session(behavior: dict, probes: list[dict]):
    ntrials = behavior["Ntrials"]
    ...
    trial_counts = np.zeros((TIME_AXIS.size, ntrials), dtype=np.float64)
    ...

use_trials = np.flatnonzero(analysis_trial_mask(behavior))
...
for local_idx, trial_idx in enumerate(use_trials):
    neural_trials.append(neural_selected[:, :, local_idx].astype(np.float32))
    input_trials.append(TIME_AXIS[None, :].astype(np.float32))
    ...
    output_trials.append(output.astype(np.int64))
```

iii. The trajectory says the AI first verified that “the behavioral event fields are directly available per trial” and then built the converter around those session MAT files, so trial splitting is justified as using the raw trial structure already present in the data object.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials satisfying `(hit or miss) and not early and not no-response and not stim-enable`. This mask is applied after neural and kinematic streams are aligned and before trials are written into the final dataset.

ii. ```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])

use_trials = np.flatnonzero(analysis_trial_mask(behavior))
if use_trials.size < 2:
    raise RuntimeError(f"{spec['animal']} {spec['date']} has fewer than 2 usable trials")
```

iii. `CONVERSION_NOTES.md` says the decoder dataset excludes `early`, `no`, and `stim.enable` trials and claims this matches recurring repository filters `~early`, `~no`, and `~stim.enable`. The notes also say the remaining trials include both correct and incorrect trials in both contexts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-cluster spike assignments stored in the MATLAB cluster objects: cluster quality labels, spike trial numbers, and spike times relative to trial. Specifically, the AI uses `quality`, `trial`, and `trialtm` from `obj.clu{probe}` plus the per-trial alignment event `behavior["ev"]["goCue"]`.

ii. ```python
def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
    return [
        {
            "quality": qualities[i],
            "trial": np.asarray(trials[i], dtype=np.int64) - 1,
            "trialtm": np.asarray(trialtm[i], dtype=np.float64),
        }
        for i in range(len(qualities))
    ]
```

iii. The notes say the conversion reproduces the repository’s “neural data” path: load clusters, align to `goCue`, bin at 10 ms, smooth, and apply the same low-firing-rate rule used by the Figure 8 scripts.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times to go-cue-aligned single-trial firing-rate traces. For each kept cluster and each trial, it subtracts the trial’s `goCue` time, bins spikes into 10 ms bins over `[-3.0, 2.5]` s, divides by `DT` to get rate, and applies a causal Gaussian smoother matching MATLAB `mySmooth(..., 15, 'reflect')`.

ii. ```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15
BCTYPE = "reflect"
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)

aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
trial_counts = np.zeros((TIME_AXIS.size, ntrials), dtype=np.float64)
...
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. `CONVERSION_NOTES.md` explicitly states: alignment event `goCue`, window `[-3.0 s, 2.5 s]`, bin size `10 ms`, and smoothing with a causal Gaussian kernel `N=15`, matching the reference `mySmooth(..., 15, 'reflect')`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI uses two neural QC stages. First it keeps all cluster qualities except `garbage`, `gabrga`, `noisy`, and `real?`. Then it computes condition-averaged PSTHs and removes units whose mean PSTH firing rate is not greater than 1 Hz.

ii. ```python
def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in {"garbage", "gabrga", "noisy", "real?"}

...
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
for ci, mask in enumerate(conditions):
    trix = np.flatnonzero(mask)
    if trix.size:
        psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
mean_fr = float(np.mean(psth))
if mean_fr > LOW_FR_HZ:
    kept_neural.append(trial_counts)
```

iii. The notes say the AI matched the Figure 8 low-FR filter by building single-trial smoothed firing rates, forming the same 7 context conditions, averaging PSTHs by condition, and dropping units whose mean across time and conditions is not above 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every neuron’s spike times are aligned per trial to the go cue by subtracting that trial’s `behavior["ev"]["goCue"]` before binning.

ii. ```python
ALIGN_EVENT = "goCue"
...
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
```

iii. Both the instructions and the AI’s notes emphasize go-cue alignment. The notes say “Alignment event: `goCue`,” and the trajectory says the AI “confirmed the MATLAB processing path” uses `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 ms bins. No later temporal rebinning is applied inside `convert_data.py`; all emitted neural, input, and output time series stay on that common 10 ms axis.

ii. ```python
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
"time_bin_size": float(DT * 1000.0),
```

iii. `CONVERSION_NOTES.md` lists bin size `10 ms`, and the trajectory says the AI had “go-cue alignment, 10 ms bins, causal smoothing” pinned down before implementation.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from a separate raw channel. It is derived from the chosen alignment event (`goCue`) and the fixed conversion window/binning constants (`TMIN`, `TMAX`, `DT`) used to define the common aligned time axis.

ii. ```python
ALIGN_EVENT = "goCue"
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. The notes justify this as “one time-varying input channel equal to the aligned neural time axis.” The trajectory likewise says the decoder input would be “time from go cue.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI constructs the midpoint of each 10 ms time bin over the aligned window and then uses that same vector as the sole input channel for every kept trial.

ii. ```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. `CONVERSION_NOTES.md` says the input is the aligned neural time axis, so the justification is simply to present go-cue-relative time directly to the decoder in the requested format.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is perfectly aligned by construction, because the same `TIME_AXIS` used for neural binning is also written as the input for every trial.

ii. ```python
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The notes say the input channel is “equal to the aligned neural time axis,” so the alignment rationale is explicit.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the raw per-trial behavior masks `R`, `L`, `hit`, and `miss`.

ii. ```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says the AI intentionally defined lick direction as actual lick direction rather than instructed side: right if `R & hit` or `L & miss`, left if `L & hit` or `R & miss`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI converts the logical rule above into a binary label, then repeats that single-trial label across all time bins so the output has shape `(n_output, T)` for each trial.

ii. ```python
lick_dir = actual_lick_direction(behavior)[use_trials]
...
output = np.vstack(
    [
        repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size),
        ...
    ]
)
```

iii. The notes say all outputs are saved as time-varying arrays because the training code prefers `(d_output, T)` format. The trajectory also indicates the AI was optimizing for the provided decoder pipeline.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the raw `autowater` trial flag.

ii. ```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` maps `0 = WC` and `1 = DR`. Its justification is that the repository uses `autowater` as the DR/WC separator; `WorkingWithDataObjs.m` also describes `autowater` as a proxy for water-cued versus delayed-response blocks.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI inverts `autowater` so that WC becomes `0` and DR becomes `1`, then tiles the resulting per-trial label across time bins.

ii. ```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
...
repeat_labels(np.asarray([context[local_idx]], dtype=np.int64), TIME_AXIS.size),
```

iii. The notes explicitly record the label convention `0 = WC`, `1 = DR`, and say all outputs are time-tiled for decoder compatibility.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the raw `hit` mask after trial filtering.

ii. ```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. `CONVERSION_NOTES.md` justifies this as `0 = incorrect`, `1 = correct`, with incorrect corresponding to misses among the retained hit/miss trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI treats `hit` as a binary correctness label, converts it to integers, and repeats the label across all time bins for each retained trial.

ii. ```python
outcome = behavior["hit"][use_trials].astype(np.int64)
...
repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size),
```

iii. The notes say the remaining trials include both correct and incorrect trials and that outputs are tiled over time to fit the decoder format.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from side-view trajectory data in `obj.traj` for the feature named `"tongue"`, using per-trial frame times and the go-cue event times.

ii. ```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
...
frame_times = np.asarray(read_numeric_dataset(f[frame_refs[trial_idx]]), dtype=np.float64)
ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), dtype=np.float64)
coords = ts[:, 0:2, feat_idx]
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The trajectory says the AI concluded the repository “interpolates onto the neural time axis, fills non-tongue gaps with nearest values, and sets invisible tongue velocity to zero,” then chose the tongue stream as the source for this decoder output.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI linearly interpolates tongue x/y positions onto the common aligned time axis, computes x/y velocity with `np.gradient`, sets invisible-tongue velocity samples to zero, and then collapses x/y into a scalar speed magnitude `sqrt(xvel^2 + yvel^2)`.

ii. ```python
def compute_velocity_from_position(xpos: np.ndarray, ypos: np.ndarray, feature_name: str):
    ...
    xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
    yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
    ...
    else:
        xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
        yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0

...
tongue_xvel, tongue_yvel = compute_velocity_from_position(tongue_xpos, tongue_ypos, "tongue")
tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
```

iii. `CONVERSION_NOTES.md` says tongue velocity is “scalar tongue speed from the reference x/y tongue velocity components” and explicitly notes the invisible-tongue periods are set to zero “as in the MATLAB code.”

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI thresholds tongue speed per session at the 50th percentile, but with an extra non-reference rule: if the all-sample median is zero, it recomputes the median using only positive tongue-speed samples.

ii. ```python
def percentile_threshold(values: np.ndarray, drop_zeros_if_needed: bool = False) -> float:
    ...
    thresh = float(np.nanpercentile(vals, 50))
    if drop_zeros_if_needed and thresh <= 0:
        pos = vals[vals > 0]
        if pos.size:
            thresh = float(np.nanpercentile(pos, 50))
    return thresh

...
tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
...
(tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :]
```

iii. The trajectory says the AI changed the rule after seeing nearly constant labels: “if the all-sample median is zero the threshold is recomputed from positive tongue-speed samples only.” The same rationale is documented in `CONVERSION_NOTES.md`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are aligned per trial by subtracting `video_offset` and that trial’s `goCue`, then interpolated directly onto the shared neural `TIME_AXIS`. The resulting tongue-velocity labels therefore have the same bin centers as the neural data.

ii. ```python
vidshift = get_video_offset(f, behavior)
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear", ...)
vals = interp(time_axis)
...
tongue_selected = tongue_speed[:, use_trials]
```

iii. The notes explicitly say “Video alignment matches the reference code: `frameTimes - video_offset - goCue`,” and the trajectory says the AI had pinned that alignment rule before writing the converter.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-view trajectory data in `obj.traj`, using whichever paw feature the AI picks from the first trial’s feature list: `"top_paw"` if present, otherwise `"bottom_paw"`.

ii. ```python
def choose_paw_feature(f: h5py.File) -> str:
    traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
    view_group = f[traj_refs[1]]
    first_feats = deref_string_list(f, f[np.array(view_group["featNames"]).reshape(-1, order="F")[0]])
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name

...
paw_feature = choose_paw_feature(f)
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. `CONVERSION_NOTES.md` says paw velocity comes from the `top_paw` bottom-view marker. The trajectory shows this was part of the AI’s attempt to choose “the right streams” for movement outputs from the DLC data.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates the paw x/y position onto the common aligned time axis, nearest-fills non-tongue gaps, computes x/y velocity with `np.gradient`, subtracts the baseline x-derivative from both velocity components, and converts the result to scalar speed magnitude.

ii. ```python
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
...
if "tongue" not in feature_name:
    xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
    ypos[:, trial_idx] = fill_nearest(ypos[:, trial_idx])

...
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
if "tongue" not in feature_name:
    xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
    yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]

...
paw_speed = np.sqrt(paw_xvel**2 + paw_yvel**2)
```

iii. The notes say the AI intended to reproduce the repository’s non-tongue logic, including nearest-fill and the MATLAB quirk of subtracting `basederiv(1)` from both `xvel` and `yvel`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI thresholds scalar paw speed per session at the 50th percentile over all kept session samples, then emits `0` for below-threshold bins and `1` for above-or-equal bins.

ii. ```python
paw_thresh = percentile_threshold(paw_selected)
...
(paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :]
```

iii. `CONVERSION_NOTES.md` says paw velocity is binarized with the per-session 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned exactly like tongue trajectories: `frameTimes - video_offset - goCue`, then interpolated onto the shared `TIME_AXIS` used by the neural data.

ii. ```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear", ...)
vals = interp(time_axis)
...
paw_selected = paw_speed[:, use_trials]
```

iii. The notes explicitly justify video alignment with the same reference `findVideoOffset` rule and say the code resamples movement streams onto the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the companion MATLAB file `motionEnergy_<animal>_<date>.mat`, specifically the stored `me.data` array and `me.moveThresh`, together with frame times from `obj.traj` and behavior event times for alignment.

ii. ```python
def load_motion_energy(animal: str, date: str, behavior: dict) -> tuple[np.ndarray, float]:
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    ...
    move_thresh = float(dat.moveThresh)
    return raw, move_thresh
```

iii. `CONVERSION_NOTES.md` says motion energy “comes from `motionEnergy_Animal_Date.mat`,” matching the repository loader.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI resamples trial-wise raw motion-energy traces onto the shared neural time axis using linear interpolation, aligns with `frameTimes - video_offset - goCue`, and nearest-fills missing edge values.

ii. ```python
def align_motion_energy(f: h5py.File, behavior: dict, raw_motion_energy) -> np.ndarray:
    ...
    shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
    valid = np.isfinite(shifted_t) & np.isfinite(me_trial)
    ...
    interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear", ...)
    aligned[:, trial_idx] = interp(TIME_AXIS)
    aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. The notes justify this by saying motion energy is “resampled onto the neural time axis, and is nearest-filled at the edges,” matching the reference loader behavior.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI thresholds aligned motion energy per session at the 50th percentile of all kept samples, producing a binary time-varying output.

ii. ```python
me_thresh = percentile_threshold(me_selected)
...
(me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :]
```

iii. `CONVERSION_NOTES.md` says motion energy is “binarized with the per-session 50th percentile.” The stored `manual_motion_energy_move_thresh` is preserved only in metadata, not used for output labeling.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned per trial by subtracting `video_offset` and the trial’s `goCue`, then interpolated directly onto the same `TIME_AXIS` used for neural activity.

ii. ```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
...
aligned[:, trial_idx] = interp(TIME_AXIS)
```

iii. The notes explicitly say motion energy alignment matches the reference `frameTimes - video_offset - goCue` path.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several fallbacks: if `sglx.bitcode` is absent it assumes `video_offset = 0.5`; if frame times are missing it synthesizes them at 400 Hz from trajectory length; if a trial’s dropped-frame metadata are all `NaN` it skips that trial’s kinematics; non-tongue gaps are nearest-filled; all-missing non-tongue traces become zeros after fill; tongue missing velocity samples are set to zero; percentile thresholds fall back to `0.0` if no finite data exist.

ii. ```python
def get_video_offset(f: h5py.File, behavior: dict) -> float:
    if "sglx" not in f["obj"] or "bitcode" not in f["obj"]["sglx"]:
        return 0.5

...
except Exception:
    ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), dtype=np.float64)
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0

...
if ndropped.size and np.isnan(ndropped).all():
    continue

...
if not mask.any():
    return np.zeros_like(x)

...
tempx[isnan] = 0.0

...
if vals.size == 0:
    return 0.0
```

iii. The notes justify several of these as mirroring repository behavior: nearest-fill for non-tongue features, zero velocity for invisible tongue, and fallback frame times at 400 Hz. The `0.5` second default offset is not separately justified beyond matching the repository’s catch/fallback style.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the nested per-neuron/per-trial spike binning and smoothing in `bin_spikes_for_session`, the per-trial interpolation of tongue/paw positions and motion energy onto `TIME_AXIS`, and the subsequent per-trial velocity computations.

ii. ```python
for probe in probes:
    for clu in probe:
        ...
        for t in unique_trials:
            counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
            trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)

for trial_idx in range(ntrials):
    ...
    interp = interp1d(...)
    vals = interp(time_axis)

for trial_idx in range(xpos.shape[1]):
    ...
    xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
    yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
```

iii. There is no explicit performance justification in the notes beyond trying to match the MATLAB path. The trajectory shows the AI focused on reproducing reference processing rather than optimizing runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The following loops could be vectorized or batched: per-column smoothing in `my_smooth`, per-neuron/per-trial histogramming in `bin_spikes_for_session`, per-trial interpolation loops in `load_traj_feature_series` and `align_motion_energy`, and per-trial velocity loops in `compute_velocity_from_position`.

ii. ```python
for col in range(x.shape[1]):
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")

for probe in probes:
    for clu in probe:
        ...
        for t in unique_trials:
            ...

for trial_idx in range(ntrials):
    ...

for trial_idx in range(xpos.shape[1]):
    ...
```

iii. The AI did not give a separate optimization rationale. The code structure itself shows a direct MATLAB-to-Python translation with several scalar loops preserved.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly computes the same alignment/interpolation pattern for tongue, paw, and motion energy; repeatedly fills missing values with nearest-neighbor logic; and repeatedly tiles per-trial scalar labels across all 550 time bins.

ii. ```python
tongue_xpos, tongue_ypos = load_traj_feature_series(...)
paw_xpos, paw_ypos = load_traj_feature_series(...)
motion_energy = align_motion_energy(...)

repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size)
repeat_labels(np.asarray([context[local_idx]], dtype=np.int64), TIME_AXIS.size)
repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The notes justify the repeated time-tiling only by compatibility with the provided decoder format. No other repetition is explicitly justified.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several items that the decoder does not use: full condition-averaged PSTHs built only to apply the low-FR filter, `kept_quality` and single-unit counts for metadata, `manual_motion_energy_move_thresh`, `event_times_sec_relative_to_go_cue`, and raw x/y position plus x/y velocity components that are immediately collapsed into scalar speeds and then discarded. It also tiles trial-constant labels across time, which is convenient for the trainer but redundant for those variables.

ii. ```python
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
...
session_stats = {
    ...
    "manual_motion_energy_move_thresh": manual_motion_thresh,
    "quality_labels_kept": kept_quality,
    "event_times_sec_relative_to_go_cue": {...},
}

tongue_xvel, tongue_yvel = compute_velocity_from_position(...)
paw_xvel, paw_yvel = compute_velocity_from_position(...)
tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
paw_speed = np.sqrt(paw_xvel**2 + paw_yvel**2)
```

iii. The notes justify most of this as sanity checking and documentation, not as required decoder inputs. The trajectory shows the AI prioritized traceability and validation artifacts in addition to producing a decoder-ready pickle.
