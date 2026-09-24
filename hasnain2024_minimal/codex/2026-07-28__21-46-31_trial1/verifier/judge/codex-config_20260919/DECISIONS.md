# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 12 fixed-delay sessions in `SESSION_SPECS`, reads each `data_structure_<animal>_<date>.mat` from `Ephys_Behavior` with `h5py`, and loads its separate motion-energy file with `scipy.io.loadmat`. It does not load the remaining sessions or the randomized-delay folder.

ii.
```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    ...
    {"animal": "JEB19", "date": "2023-04-18", "probes": [1]},
]
EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")
with h5py.File(data_path, "r") as f:
    behavior = load_behavior(f)
```

iii. The trajectory says this was a deliberate choice to target the 12-session alternating-context ALM cohort used by the Figure 8 scripts. The AI preferred that explicit loader subset over forcing agreement with broader paper-level counts.

## 1-b. How are the data split into subjects?

i. The `animal` field in each hard-coded session identifies the subject. Unique animals are accumulated in first-seen order and each session receives its index. This produces seven subjects for the selected sessions.

ii.
```python
sid = spec["animal"]
if sid not in subject_lookup:
    subject_lookup[sid] = len(subject_lookup)
    subjects.append(sid)
data["subject_idx"].append(subject_lookup[sid])
```

iii. The AI treated the animal IDs encoded in the Figure 8 loader/session filenames as authoritative and explicitly noted that its selected subset has seven IDs despite methods text saying six mice.

## 1-c. How are the data split into sessions?

i. Each entry of `SESSION_SPECS`, corresponding to one animal/date MATLAB file, becomes one session element in `neural`, `input`, and `output`.

ii.
```python
for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    data["neural"].append(payload["neural_trials"])
    data["input"].append(payload["input_trials"])
    data["output"].append(payload["output_trials"])
```

iii. The AI justified the 12 sessions as the exact cohort used for the reference repository's Figure 8 context analyses.

## 1-d. How are the data split into trials?

i. Raw trial vectors are read from `obj.bp`; spike records already contain zero-based trial indices. After a Boolean trial mask is applied, each retained trial is appended separately to the three session lists.

ii.
```python
use_trials = np.flatnonzero(analysis_trial_mask(behavior))
for local_idx, trial_idx in enumerate(use_trials):
    neural_trials.append(neural_selected[:, :, local_idx].astype(np.float32))
    input_trials.append(TIME_AXIS[None, :].astype(np.float32))
    output_trials.append(output.astype(np.int64))
```

iii. The AI followed the trial indexing already present in Bpod and cluster data and required at least two usable trials per session for decoder evaluation.

## 1-e. How are trials filtered based on quality controls?

i. It retains only hit or miss trials and removes early, no-response, and photostimulation trials. It does not apply the reference's recording-end cutoff.

ii.
```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])
```

iii. The AI said `~early`, `~no`, and `~stim.enable` recur in repository analyses and chose to keep both correct and incorrect trials in both contexts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: each cluster's `quality`, spike `trial`, and `trialtm`, plus `obj.bp.ev.goCue` for alignment.

ii.
```python
qualities = deref_string_list(f, probe_group["quality"])
trials = deref_numeric_list(f, probe_group["trial"])
trialtm = deref_numeric_list(f, probe_group["trialtm"])
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The AI aimed to reproduce the repository's aligned single-trial firing-rate path using the explicitly selected ALM probe for each session.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed per cluster and trial, divided by 10 ms to obtain Hz, and smoothed with a causalized 15-sample Gaussian window using reflected prepending. No normalization or baseline subtraction is applied.

ii.
```python
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
...
kern = gausswin(n)
kern[: n // 2] = 0
```

iii. The AI described this as matching `mySmooth(..., 15, 'reflect')` and the Figure 8 processing, including a causal Gaussian kernel.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It excludes clusters whose case-sensitive stripped label is exactly `garbage`, `gabrga`, `noisy`, or `real?`. It then computes seven condition PSTHs and keeps units whose mean over all time/conditions exceeds 1 Hz.

ii.
```python
return quality not in {"garbage", "gabrga", "noisy", "real?"}
...
mean_fr = float(np.mean(psth))
if mean_fr > LOW_FR_HZ:
    kept_neural.append(trial_counts)
```

iii. The AI explicitly sought to match the MATLAB Figure 8 condition-PSTH path and paper's greater-than-1-Hz inclusion rule; it accepted a one-unit discrepancy rather than forcing counts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's go-cue time subtracted before binning.

ii.
```python
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The AI selected `goCue` because the task explicitly requires go-cue alignment and the reference code uses `alignEvent = goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 10 ms bins over −3.0 to +2.5 seconds (550 bins). Spikes are directly histogrammed into this grid; camera streams are linearly interpolated onto its bin centers.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
```

iii. The AI asserted that this was the Figure 8 reference grid and recorded `time_bin_size` as 10 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived trial-by-trial from a raw variable; it is the module-level `TIME_AXIS` defined relative to zero, with `bp.ev.goCue` establishing the same zero for neural/video alignment.

ii.
```python
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The AI regarded the common aligned grid itself as the requested continuous input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Adjacent time-bin edges are averaged to form centers, and the identical one-row array is copied into every trial.

ii.
```python
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The AI chose bin centers so the input has exactly the same number of samples as neural firing rates.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Neural histograms use `TIME_EDGES`, while the input is the corresponding centers, so each input sample labels the matching neural bin and zero denotes go-cue onset.

ii.
```python
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The AI emphasized a common time axis across neural, input, and behavioral streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses Bpod instructed-right/left (`R`, `L`) and hit/miss flags.

ii.
```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. The AI reasoned that actual direction is the instructed side on hits and the opposite side on misses.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Right is encoded 1 for `R & hit` or `L & miss`; all retained alternatives are left (0). The per-trial code is repeated through time. No `none` class exists because no-response trials were dropped.

ii.
```python
lick_dir = actual_lick_direction(behavior)[use_trials]
repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The AI described this as actual rather than instructed lick direction and limited labels to the response trials it retained.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from the per-trial `obj.bp.autowater` flag.

ii.
```python
"autowater": np.asarray(read_numeric_dataset(bp["autowater"]), dtype=bool)
```

iii. The AI interpreted autowater as water-cued context and its complement as delayed response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is inverted and cast to integer, yielding WC=0 and DR=1, then repeated at every time bin.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
repeat_labels(np.asarray([context[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The mapping follows the output label order `['WC', 'DR']`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Although hit, miss, and no are loaded, the saved outcome is derived from `hit` after the trial mask has removed no-response trials.

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. The AI retained only hit/miss trials, making hit sufficient to distinguish correct from incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit is cast to 1 (correct), while retained misses become 0 (incorrect), and the label is repeated through time. There is no ignore class.

ii.
```python
OUTPUT_VALUES = [..., ["incorrect", "correct"], ...]
repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The AI deliberately excluded `no` trials based on repository analysis masks, despite the requested three-category output.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-view `obj.traj` entries for the feature named `tongue`: `featNames`, `frameTimes`, `NdroppedFrames`, and x/y values from `ts`. It also uses bitcode timing and go-cue times. Likelihood values are present in `ts` but ignored.

ii.
```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
coords = ts[:, 0:2, feat_idx]
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The AI said it followed reference position interpolation and velocity logic, but selected only the side-view tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-camera x/y positions are linearly interpolated to the 10 ms neural centers. `np.gradient` is applied with respect to sample index, missing gradients are set to zero, and speed is `sqrt(xvel²+yvel²)`. No likelihood cutoff, time-scaled derivative, second camera, or cross-view normalization is used.

ii.
```python
vals = interp(time_axis)
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
```

iii. The AI claimed tongue invisibility becomes zero as in the MATLAB code and used a scalar magnitude from x/y velocity components.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over retained samples; if it is nonpositive, the median is recomputed over positive samples. Values below it are 0 and values at/above it are 1. No not-visible category 2 is produced.

ii.
```python
tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
(tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :]
```

iii. The AI added the positive-only fallback because zero-filled invisible periods otherwise made several sessions' labels constant; it prioritized decoder usefulness.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video offset is calculated from SpikeGLX and Bpod bit starts; each frame time is transformed as `frameTimes - offset - goCue`, then positions are interpolated directly to neural bin centers.

ii.
```python
return matlab_mode(bitstart) / fs - matlab_mode(behavior["ev"]["bitStart"])
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], coords[valid, dim], ...)
```

iii. The AI explicitly intended to reproduce `findVideoOffset` and use one common time axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-view trajectory `featNames`, `frameTimes`, `NdroppedFrames`, and x/y `ts`, preferring `top_paw` and falling back to `bottom_paw`; bitcode and go cue provide alignment.

ii.
```python
for name in ("top_paw", "bottom_paw"):
    if name in first_feats:
        return name
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The AI described the output as scalar speed from a bottom-view paw marker, normally `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are linearly interpolated to 10 ms centers and nearest-filled. Gradients are computed per sample; a median x/y displacement is calculated, but its x component is subtracted from both velocity components. Speed is the Euclidean magnitude.

ii.
```python
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0]) - basederiv[0]
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1]) - basederiv[0]
paw_speed = np.sqrt(paw_xvel**2 + paw_yvel**2)
```

iii. The AI intentionally reproduced what it viewed as the MATLAB implementation, including subtracting `basederiv(1)` from both axes.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. All finite retained-session samples are split at their 50th percentile into 0 below and 1 at/above. No not-visible category is generated.

ii.
```python
paw_thresh = percentile_threshold(paw_selected)
(paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :]
```

iii. The AI followed the prompt's per-session median split but filled missing data beforehand.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-view frame times are corrected by the same session offset and trial go cue, then x/y are interpolated to the neural centers.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
vals = interp(time_axis)
```

iii. The AI used the common video-offset correction and common neural time axis for all streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads `me.data` and `me.moveThresh` from `motionEnergy_<animal>_<date>.mat`; side-camera frame times and bitcode/go-cue timing align it. `moveThresh` is recorded only as metadata.

ii.
```python
dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
raw = dat.data
move_thresh = float(dat.moveThresh)
```

iii. The AI used the standalone motion-energy files as the reference source and retained the manual threshold for diagnostics.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each already-computed trace is linearly interpolated from camera frames to 10 ms neural centers, then edge gaps are nearest-filled. No further spatial or temporal feature extraction is done.

ii.
```python
interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear", ...)
aligned[:, trial_idx] = interp(TIME_AXIS)
aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. The AI reasoned that the file already contains reduced motion energy and only alignment/resampling remains.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Finite retained-session samples are split at the session median into categories 0 and 1. There is no category 2 for no video.

ii.
```python
me_thresh = percentile_threshold(me_selected)
(me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :]
```

iii. The AI followed the requested 50th-percentile threshold but removed missingness by filling.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video offset and trial go cue, and the values are interpolated to `TIME_AXIS`.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
aligned[:, trial_idx] = interp(TIME_AXIS)
```

iii. The AI used the same bitcode-derived offset and go-cue-centered grid as for tracked positions.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing `stim.enable` becomes all false; absent SpikeGLX timing gets a hard-coded 0.5 s offset; malformed/missing trajectory trials remain NaN initially; tongue gaps become zero velocity, paw and motion gaps are nearest-filled (fully missing paw becomes zeros), and missing motion frame times are synthesized at 400 Hz. Sessions with fewer than two usable trials or no neurons raise errors.

ii.
```python
out["stim_enable"] = np.zeros(out["Ntrials"], dtype=bool)
if "sglx" not in f["obj"] ...: return 0.5
frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
if not mask.any(): return np.zeros_like(x)
```

iii. The trajectory says these choices remove runtime warnings and make saved output deliberately complete; in particular it treated invisible tongue velocity as zero and nearest-filled non-tongue streams.

## 11-a. What are the most time-consuming steps of the code?

i. The likely dominant steps are HDF5 traversal plus nested cluster/trial spike histogramming and per-trial trajectory interpolation. The AI did not profile or explicitly identify a bottleneck in its final notes.

ii.
```python
for probe in probes:
    for clu in probe:
        for t in unique_trials:
            counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
for trial_idx in range(ntrials):
    interp = interp1d(...)
```

iii. The trajectory focused on correctness checks and decoder runs, not timing measurements; it regenerated the full conversion several times.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cluster inner trial histogram loop could use a 2-D histogram over trial and time. Some output assembly/repetition and per-trial gradient operations could also be batched once rectangular arrays exist; irregular HDF5 camera records still require some trial iteration.

ii.
```python
for t in unique_trials:
    mask = trial_idx == t
    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
for local_idx, trial_idx in enumerate(use_trials):
    output = np.vstack([...])
```

iii. The AI gave no explicit vectorization justification; its implementation favors direct correspondence with MATLAB/session logic.

## 11-c. What processing does the code repeat multiple times?

i. `get_video_offset` is recomputed independently for motion energy, tongue, and paw in each session. Feature-name structures are dereferenced per trial, and the same time/input and scalar trial labels are repeatedly allocated for every trial.

ii.
```python
vidshift = get_video_offset(f, behavior)  # in load_traj_feature_series
vidshift = get_video_offset(f, behavior)  # in align_motion_energy
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The AI's notes claimed a common alignment method, but the code does not cache the session offset; no rationale for this repetition was recorded.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes seven condition PSTHs only to reduce them to a firing-rate filter; loads `reward`; computes/retains quality and single-unit diagnostics not used by the decoder; reads manual motion thresholds but does not use them for labels; calculates `basederiv[1]` but never applies it; and creates extensive metadata statistics.

ii.
```python
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
mean_fr = float(np.mean(psth))
raw_motion_energy, manual_motion_thresh = load_motion_energy(...)
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
```

iii. The AI used these extras for matching the Figure 8 filter, documenting the one-unit discrepancy, sanity checking cohort totals, and recording diagnostics, even though decoder training does not consume them.
