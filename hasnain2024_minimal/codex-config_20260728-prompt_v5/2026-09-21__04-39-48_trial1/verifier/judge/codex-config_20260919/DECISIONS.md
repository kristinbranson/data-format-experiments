# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 Figure 8 two-context sessions and one probe per session. It reads each `data_structure_*.mat` as HDF5 with `h5py`, and reads the companion motion-energy v5 MAT file with SciPy. It does not load the other author-selected fixed- or randomized-delay sessions.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", "Ephys_Behavior", 2),
    ...
    SessionSpec("JEB19", "2023-04-21", "Ephys_Behavior", 1),
]
sessions = [process_session(spec) for spec in SESSION_SPECS]
with h5py.File(spec.data_path, "r") as h5:
    bp = load_bp_fields(h5)
```

iii. The trajectory says the fixed-delay directory mixes DR and two-context experiments, and that the agent chose the 12-session Figure 8 cohort because those sessions contain substantial WC blocks. It chose `h5py` because the main files are MATLAB v7.3 and SciPy for the older motion-energy files.

## 1-b. How are the data split into subjects?

i. Subject IDs are explicitly stored as `SessionSpec.animal`; unique subjects are accumulated in first-appearance order and each session receives the corresponding integer index. The selected subset has seven mice.

ii.
```python
if spec.animal not in subject_lookup:
    subject_lookup[spec.animal] = len(subjects)
    subjects.append(spec.animal)
subject_idx.append(subject_lookup[spec.animal])
```

iii. The agent relied on the animal identifiers embedded in the session-loader-derived specifications. It did not discuss an alternative subject split.

## 1-c. How are the data split into sessions?

i. Each hard-coded `SessionSpec` and corresponding `data_structure_<animal>_<date>.mat` is one output session. Only 12 fixed-delay Figure 8 sessions are included.

ii.
```python
def session_id(self) -> str:
    return f"{self.animal}_{self.date}"
sessions = [process_session(spec) for spec in SESSION_SPECS]
```

iii. The trajectory explicitly says the agent followed the Figure 8 context cohort instead of mixing in DR-only or randomized-delay sessions.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the trial count. Boolean Bpod vectors are indexed by zero-based trial index; spike `clu.trial` values are treated as one-based and converted during spike binning. Each retained trial becomes one matrix in each session list.

ii.
```python
"Ntrials": int(np.asarray(bp["Ntrials"])[0, 0]),
valid_trials = np.flatnonzero(valid_mask)
np.add.at(counts, (trial_ids[valid] - 1, bins[valid]), 1.0)
neural_trials = [trial_array[trial_idx] for trial_idx in range(trial_array.shape[0])]
```

iii. The agent inspected the raw HDF5 organization and used its existing per-trial Bpod, cluster, trajectory, and motion-energy indexing rather than reconstructing trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained when they are neither early-lick nor photostimulation trials. The code does not remove behavior trials after the electrophysiology recording has ended and does not use `haveEphys`/`haveVid` flags.

ii.
```python
valid_mask = (~bp["early"]) & (~bp["stim_enable"])
valid_trials = np.flatnonzero(valid_mask)
```

iii. The trajectory identifies early-trial exclusion as the main-analysis rule and says it checked missing-data flags before deciding to keep missing-video trials as categorical missingness. No justification is given for omitting the recording-end filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probe's `obj.clu` entries: `quality`, spike `trial`, and within-trial spike time `trialtm`, together with `bp.ev.goCue` for alignment.

ii.
```python
"trial": read_ref_array(h5, probe_group["trial"][clu_idx, 0]),
"trialtm": read_ref_array(h5, probe_group["trialtm"][clu_idx, 0]),
aligned_times = trialtm - bp["goCue"][trial_ids - 1]
```

iii. The agent traced the paper's spike pipeline and probe assignments and reported that go-cue alignment and cluster data were the relevant neural inputs.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed at 10 ms, converted to Hz, and smoothed with a custom causal half-Gaussian window of 15 bins with reflected leading padding. Stored values are `float32`; there is no normalization or baseline subtraction.

ii.
```python
DT = 0.01
cluster_trials = counts[valid_mask] / DT
cluster_trials = my_smooth(cluster_trials.T, SMOOTH_WINDOW, BCTYPE).T
kernel[: len(kernel) // 2] = 0.0
```

iii. The trajectory says the agent believed the Figure 8 pipeline used 10 ms bins and causal Gaussian smoothing and attempted to reproduce the MATLAB helper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labeled `garbage`, misspelled `gabrga`, `noisy`, or `real?` are excluded. A unit is retained only if the mean of seven condition PSTHs exceeds 1 Hz. The label `poor` is not excluded.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
mean_fr = float(np.mean(np.stack(psths, axis=1)))
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. The trajectory states that it followed the repository's “all non-garbage units above 1 Hz” rule and the Figure 8 condition construction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time before histogramming.

ii.
```python
aligned_times = trialtm - bp["goCue"][trial_ids - 1]
```

iii. The agent explicitly identified `obj.bp.ev.goCue` as the repository's alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 550 non-overlapping 10 ms bins centered from −2.995 through 2.495 seconds. Raw spikes are rebinned into this grid; videos are linearly interpolated to the bin centers.

ii.
```python
DT = 0.01
TMIN = -3.0
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

iii. The agent repeatedly stated that it inferred 10 ms and `[−3.0, 2.5]` from the Figure 8 code path.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw field beyond the semantic choice of `goCue`; it is the module-level `TIME` vector constructed from fixed constants.

ii.
```python
TIME = EDGES[:-1] + (DT / 2.0)
inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]
```

iii. The agent regarded the shared bin centers relative to go cue as the required decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are computed by adding half a 10 ms bin to each left edge; the identical one-row vector is copied for every retained trial.

ii.
```python
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

iii. The agent wanted a consistent time-varying input tensor for all trials.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. `TIME` is both the spike-bin-center axis and the interpolation target for behavior, so input column `t` corresponds directly to neural column `t`.

ii.
```python
bins = np.searchsorted(EDGES, aligned_times, side="right") - 1
inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]
```

iii. The trajectory says it checked that the time-varying channels and neural bins lined up.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from Bpod `R`, `L`, `hit`, `miss`, and `no` fields.

ii.
```python
if bp["no"][trial_idx]: return 2
if (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx]): return 1
```

iii. The code infers the actual lick from instructed side plus success/failure and assigns no-response trials to `none`; the trajectory gives no additional justification.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct right and incorrect left trials become right (1); correct left and incorrect right trials become left (0); `no` becomes none (2). The per-trial code is repeated across all time bins.

ii.
```python
trial_output[0] = lick_code
OUTPUT_VALUES[0] = ["left", "right", "none"]
```

iii. The agent chose repeated labels so all outputs form a single time-by-output tensor.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived solely from `bp.autowater`.

ii.
```python
"autowater": np.asarray(bp["autowater"]).reshape(-1).astype(bool)
context_code = 0 if bp["autowater"][trial_idx] else 1
```

iii. The trajectory says the repository treats `autowater` as the WC-context flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater=True` maps to WC (0), otherwise DR (1), repeated over the trial's time axis.

ii.
```python
OUTPUT_VALUES[1] = ["WC", "DR"]
trial_output[1] = context_code
```

iii. The Figure 8 sessions were selected specifically because they contain substantial blocks of both contexts.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes from Bpod `miss`, `hit`, and `no`.

ii.
```python
if bp["miss"][trial_idx]: return 0
if bp["hit"][trial_idx]: return 1
if bp["no"][trial_idx]: return 2
```

iii. No separate trajectory justification was recorded; these fields directly encode the requested outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss maps to incorrect (0), hit to correct (1), and no response to ignore (2), repeated through time.

ii.
```python
OUTPUT_VALUES[2] = ["incorrect", "correct", "ignore"]
trial_output[2] = outcome_code
```

iii. The agent followed the requested three categories and its uniform output tensor design.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses x/y coordinates from several DLC landmarks in both `obj.traj` views: side `tongue`, `left_tongue`, `right_tongue`; bottom `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`. It also uses frame times, video/behavior bit starts, sampling rate, and go cue. Likelihood is present in `ts` but is not explicitly read or thresholded.

ii.
```python
tongue_indices = {0: [... "tongue", "left_tongue", "right_tongue" ...],
                  1: [... "top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue" ...]}
coords = side_ts[:, :2, feat_idx]
```

iii. The trajectory says the paper did not define one canonical decoder landmark, so the agent chose a cross-session scalar summary across available tongue landmarks and views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Landmark coordinates are linearly interpolated to the 10 ms grid. Gradients are taken in samples (not divided by seconds), nonfinite gradient values are set to zero, Euclidean speed is computed, and visible landmarks are averaged at each time point. There is no coordinate smoothing, likelihood threshold, or cross-camera scale normalization.

ii.
```python
coords_interp = interpolate_coords(coords, side_ft, align_time, vidshift)
vel = np.gradient(coords_interp, axis=0)
vel[~np.isfinite(vel)] = 0.0
speed = np.sqrt((vel ** 2).sum(axis=1))
out[t] = float(np.mean(speed_stack[mask, t]))
```

iii. The agent wanted one defensible scalar summary and to preserve visibility handling. Its trajectory does not justify omitting smoothing, physical-time differentiation, or camera-scale normalization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single session median over values marked visible is used: below median is 0, at/above is 1, and not visible is 2.

ii.
```python
tongue_thresh = float(np.nanpercentile(tongue_series[tongue_visible], 50))
trial_output[3] = np.where(~tongue_visible[out_idx], 2,
                           (tongue_series[out_idx] >= tongue_thresh).astype(np.int64))
```

iii. The 50th-percentile session threshold and missing category come directly from the task. The agent noted that high missingness was plausible and retained it after decoder validation.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The session video shift is computed from ephys and behavior bit starts. Each frame time is transformed as `frameTimes - vidshift - goCue`, then coordinates are linearly interpolated at `TIME`.

ii.
```python
return matlab_mode(bitstart) / fs - matlab_mode(bp["bitStart"])
frame_times - vidshift - align_time
interp(TIME)
```

iii. The agent traced the repository's bitcode-based clock correction and aimed to put all channels on the neural grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-view x/y coordinates for both `top_paw` and `bottom_paw`, plus frame times, clock-alignment fields, and go cue.

ii.
```python
paw_indices = {1: [feat_names[1].index(name)
                   for name in ["top_paw", "bottom_paw"] if name in feat_names[1]]}
```

iii. The agent inspected available DLC names and chose to summarize multiple paw landmarks; it did not justify using two distinct paws rather than the reliably tracked `top_paw` alone.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Coordinates are linearly interpolated and nearest-filled. Their gradients are computed; a median drift estimate is subtracted, but the x component of that estimate is mistakenly subtracted from both x and y gradients. Speeds from both paw landmarks are averaged. If any paw exists, every bin is marked visible.

ii.
```python
coords_filled = fill_nearest(coords_interp)
base_deriv = np.nanmedian(np.diff(coords_filled, axis=0), axis=0)
vel[:, 0] = vel[:, 0] - base_deriv[0]
vel[:, 1] = vel[:, 1] - base_deriv[0]
agg_paw = np.mean(np.stack(trial_paw_speeds, axis=0), axis=0)
agg_paw_visible = np.ones(TIME.size, dtype=bool)
```

iii. The agent sought a single scalar feature and used filling/drift correction to stabilize it. The trajectory supplies no paper-based justification for these operations.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session median of values marked visible is used; values below/above map to 0/1 and invisible bins to 2. Because visibility is forced true whenever paw features exist, class 2 is generally suppressed.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_series[paw_visible], 50))
trial_output[4] = np.where(~paw_visible[out_idx], 2,
                           (paw_series[out_idx] >= paw_thresh).astype(np.int64))
```

iii. The median split follows the prompt; the all-visible behavior is not justified.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times receive the same bitcode shift and go-cue subtraction as tongue coordinates, then are linearly interpolated at neural bin centers.

ii.
```python
coords_interp = interpolate_coords(coords, bottom_ft, align_time, vidshift)
```

iii. The shared clock correction and grid were chosen to align all time-varying outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It uses each trial's precomputed trace from `motionEnergy_<session>.mat`, the side-camera frame times, video/behavior bit starts, sampling rate, and go cue. The file's `moveThresh` is read but ignored.

ii.
```python
motion_energy_data, _ = read_motion_energy(spec.motion_energy_path)
me_trace = np.asarray(motion_energy_data[trial_idx]).reshape(-1)
```

iii. The agent found the companion files to be MATLAB v5 and normalized an extra wrapper encountered during execution.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The precomputed trace is linearly interpolated to `TIME`; if its length differs from frame times, synthetic 400 Hz times are used. Missing interpolation results are nearest-filled.

ii.
```python
if frame_times.size != me_trace.size:
    frame_times = np.arange(1, me_trace.size + 1) / 400.0
return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))
```

iii. The agent treated motion energy as already reduced to one value per frame and only resampled it. It did not justify nearest-filling rather than preserving missing video.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session median over values marked visible defines classes 0 and 1; class 2 is used when the side video is absent. Nearest filling makes interpolated traces appear visible across the full output window.

ii.
```python
me_thresh = float(np.nanpercentile(me_series[me_visible], 50))
trial_output[5] = np.where(~me_visible[out_idx], 2,
                           (me_series[out_idx] >= me_thresh).astype(np.int64))
```

iii. The threshold and no-video class follow the prompt; the trajectory does not justify the effective erasure of partial missingness.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video shift and trial go cue, and the trace is interpolated onto the neural bin centers.

ii.
```python
motion = interpolate_motion_energy(me_trace, side_ft, align_time, vidshift)
```

iii. The agent used the common bitcode correction and target grid to align the output.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing frame times are replaced with a synthetic 400 Hz sequence. Coordinates are interpolated; paw coordinates/velocities and motion energy are nearest-filled. Completely missing tongue/paw data becomes class 2; absent side video makes motion energy class 2. The code has early returns for wholly nonfinite coordinates, and it fixed an accidental NumPy `.data` unwrap found during testing.

ii.
```python
frame_times = np.arange(1, ts.shape[0] + 1) / 400.0
coords_filled = fill_nearest(coords_interp)
if not np.any(visible):
    return np.full(TIME.size, np.nan), visible
```

iii. The agent intended to keep missing-video trials using the requested categorical missing labels, and added all-NaN guards after observing warnings. It favored filling for partial gaps.

## 11-a. What are the most time-consuming steps of the code?

i. The observed slow stage is per-trial video coordinate interpolation and feature processing across all sessions; file reading is also substantial. Neural cluster histogramming/smoothing adds nested work.

ii.
```python
for trial_idx in valid_trials:
    ...
    coords_interp = interpolate_coords(...)
```

iii. While the converter ran, the agent explicitly attributed its runtime to per-trial video interpolation across the sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. `aggregate_speeds` loops over every time bin even though masked means can be vectorized. Cluster-by-condition PSTH construction, per-column convolution/filling, per-trial output assembly, and feature loops also offer partial batching, though ragged video trials limit full vectorization.

ii.
```python
for t in range(TIME.size):
    mask = vis_stack[:, t]
    if np.any(mask):
        out[t] = float(np.mean(speed_stack[mask, t]))
```

iii. The trajectory does not discuss vectorization; it accepted the runtime once conversion completed.

## 11-c. What processing does the code repeat multiple times?

i. Each session's HDF5 file is opened once for Bpod/clusters and again for video. Every unit recomputes seven PSTHs solely to obtain a rate filter. Each feature separately interpolates the same trial/view time base, and each trial receives a fresh copy of identical `TIME`.

ii.
```python
with h5py.File(spec.data_path, "r") as h5: ...
outputs, thresholds = compute_behavioral_outputs(spec, bp, valid_trials)  # reopens file
for cond_mask in condition_masks:
    ...
inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]
```

iii. The agent did not identify these repetitions; its implementation separates neural and behavioral processing for clarity.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Seven smoothed condition PSTHs are computed for every cluster and then discarded after reducing them to one scalar mean-rate criterion. `sample`, `delay`, motion-energy `moveThresh`, and visibility arrays are loaded/computed but not saved directly. Detailed session thresholds are retained only as metadata.

ii.
```python
psths.append(psth)
mean_fr = float(np.mean(np.stack(psths, axis=1)))
motion_energy_data, _ = read_motion_energy(spec.motion_energy_path)
```

iii. The agent believed the condition PSTHs reproduced the Figure 8 low-firing-rate filter. It did not discuss discarded fields or simpler equivalent rate computation.
