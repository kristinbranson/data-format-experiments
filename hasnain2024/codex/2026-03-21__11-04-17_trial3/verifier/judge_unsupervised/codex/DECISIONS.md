# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not scan every raw file blindly. It reconstructs the reference session list by parsing the MATLAB loader scripts for fixed-delay and randomized-delay ephys sessions, then loads each selected session `.mat` file with `pymatreader.read_mat`. Motion-energy files are loaded separately per session when needed.

ii.
```python
FIXED_DELAY_LOADERS = [
    "loadJEB6_ALMVideo.m", ..., "loadJEB19_ALMVideo.m",
]
RANDOMIZED_DELAY_LOADERS = [
    "loadJEB11_ALMVideo.m", ..., "loadJEB24_ALMVideo.m",
]

def get_reference_sessions() -> List[SessionSpec]:
    sessions: List[SessionSpec] = []
    for loader in FIXED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
    for loader in RANDOMIZED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))
    return sessions

obj = read_mat(spec.data_path)["obj"]
```

iii. In Step 5 and Step 10 of `CONVERSION_NOTES.md`, the agent says it followed the same loader files used by `loadSessionData.m` so session inclusion would match the released MATLAB code rather than the raw directory contents.

## 1-b. How are the data split into subjects?

i. Each `SessionSpec` stores a `subject` parsed from the loader filename, and `build_dataset()` constructs `subjects`/`subject_idx` by session order.

ii.
```python
subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
subject = subject_match.group(1)

if spec.subject not in subject_order:
    subject_order.append(spec.subject)
subject_idx.append(subject_order.index(spec.subject))
```

iii. The notes say the agent wanted subject assignment to come from the reference loader metadata, not from ad hoc path parsing later in the pipeline.

## 1-c. How are the data split into sessions?

i. One parsed loader entry becomes one session. Each session is processed independently by `process_session(spec, ...)`, and the final dataset stores one list entry per session in `neural`, `input`, and `output`.

ii.
```python
sessions.append(
    SessionSpec(
        subject=subject,
        date=current_date,
        probe=current_probe,
        folder=folder,
        task=task,
    )
)

for spec in session_specs:
    result = process_session(spec, show_processing=do_plot)
```

iii. The notes explicitly justify this as matching `loadSessionData.m`, where one `meta` entry corresponds to one session.

## 1-d. How are the data split into trials?

i. Within each session, trials are defined from the raw session trial count, filtered by `trial_mask`, and stored as one `(neurons, time)` matrix per kept trial. Inputs and outputs are also stored one object per kept trial.

ii.
```python
trial_mask = build_trial_mask(obj)
keep_trials = np.flatnonzero(trial_mask)

neural_trials = [np.zeros((n_neurons, time_centers.size), dtype=np.float32) for _ in range(n_trials)]

for tr in range(n_trials):
    input_trials.append(time_centers[None, :].astype(np.float32))
    output_trials.append(np.vstack([...]))
```

iii. In the notes, the agent describes this as converting the MATLAB `trialdat` convention into the required list-of-trials format while keeping session boundaries intact.

## 1-e. How are trials filtered based on quality controls?

i. Trials are first filtered to exclude stimulation, early-lick, and no-response trials. The code then applies an additional ad hoc trim: if kept neurons have no spikes beyond some raw trial index, later behavior-valid trials are dropped for lack of neural coverage.

ii.
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid

max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
refined_trial_mask = trial_mask & neural_coverage_mask
```

iii. Step 5 says this matches the reference “no-stim / non-early / non-no-response” logic. Step 9 and Step 10 justify the extra neural-coverage trim as a fix for two JEB24 sessions with behavior-valid trials after the last neural trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from per-cluster trial-local spike times and trial indices in `obj.clu{probe}` together with the trial go-cue times in `obj.bp.ev.goCue`.

ii.
```python
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
trials = to_vector(probe["trial"][neuron_index], int)
trialtm = to_vector(probe["trialtm"][neuron_index], float)
```

iii. The notes map `obj.clu{probe}.trialtm`, `obj.clu{probe}.trial`, and `obj.bp.ev.goCue` directly onto the converted neural representation.

## 2-b. How is the `neural` data processed?

i. The agent aligns spikes to go cue, bins them at 5 ms from `-2.5` to `2.5` s, converts counts to Hz, and applies a causal Gaussian smooth with reflect padding.

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
SMOOTH = 15
BCTYPE = "reflect"

aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts = np.zeros((edges.size - 1, keep_trials_0based.size), dtype=np.float32)
rates = counts / DT
return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
```

iii. Step 5 says this was chosen to match `alignSpikes`/`getSeq` while using the paper’s 5 ms decoding bin instead of the 10 ms setting seen in some scripts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in three stages: bad quality labels are removed, neurons with mean firing rate `<= 1 Hz` are removed, and sessions with fewer than 10 kept units are skipped.

ii.
```python
bad = {"garbage", "gabrga", "noisy", "real?"}
return np.array([q not in bad for q in cleaned], dtype=bool)

if mean_fr > LOW_FR:
    kept_indices.append(int(neuron_index))

if len(selected_neurons) < MIN_UNITS_PER_SESSION:
    log(f"Skipping {spec.session_id}: only {len(selected_neurons)} units after filtering")
```

iii. The notes tie these choices to `findClusters(..., 'all')`, `removeLowFRClusters`, and the paper statement that sessions required at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset. For each spike, the code subtracts the go-cue time of that spike’s trial before binning.

ii.
```python
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
aligned = trialtm[mask] - align_times[trials[mask] - 1]
```

iii. Both the notes and the reference-script excerpts emphasize `params.alignEvent = 'goCue'`, which the agent states was required by the decoder task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 5 ms bins. There is no later rebinning stage; all aligned data streams are interpolated or binned directly onto that axis.

ii.
```python
DT = 0.005

def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
```

iii. Step 3 and Step 4 justify the 5 ms choice from the paper’s decoding methods, even though several MATLAB scripts use 10 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not taken from a raw vector in the file. It is derived from the common bin-center axis used after go-cue alignment, so it depends indirectly on the go-cue event definition and the chosen `TMIN/TMAX/DT`.

ii.
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The notes say the decoder input should be “the common aligned time axis” shared with the neural representation.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code creates a fixed time vector of bin centers from `-2.4975` to `2.4975` s and copies that same vector into every kept trial.

ii.
```python
def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0

for tr in range(n_trials):
    input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. In Step 10, the agent says it spot-checked that this input exactly matched the reconstructed neural time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is perfectly aligned by construction because it uses the same `time_centers` array that indexes the neural binning.

ii.
```python
time_centers = make_time_centers(time_edges)
rates = binned_neuron_trials(..., time_edges)
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The notes explicitly justify this as using “the same bin centers as the neural data.”

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from `obj.bp.R` after trial filtering. Left is implicitly encoded by `R == 0` on the remaining trials.

ii.
```python
bp = obj["bp"]
R = to_vector(bp["R"], float).astype(int)
lick_direction = R[keep_trials]
```

iii. Step 5 says the mapping is `obj.bp.R / obj.bp.L -> left=0, right=1`, with no-response trials removed first.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The only processing is trial filtering and converting the per-trial value into a time-repeated categorical row for each trial.

ii.
```python
lick_direction = R[keep_trials]
np.full(time_centers.size, lick_direction[tr], dtype=np.int64)
```

iii. The notes justify this as keeping choice well-defined by removing `no` and `early` trials before encoding left/right.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`.

ii.
```python
autowater = to_vector(bp["autowater"], float).astype(int)
context = 1 - autowater[keep_trials]
```

iii. The notes state that `autowater` is the reference code’s proxy for WC vs DR context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The raw `autowater` polarity is inverted so that WC becomes `0` and DR becomes `1`, then repeated across all time bins within a trial.

ii.
```python
context = 1 - autowater[keep_trials]
np.full(time_centers.size, context[tr], dtype=np.int64)
```

iii. Step 5 says this inversion was required because the task definition asked for `WC = 0, DR = 1`, opposite the raw `autowater` interpretation.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit` and checked against `obj.bp.miss` after the no-response trials have been excluded.

ii.
```python
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)
outcome = hit[keep_trials]
```

iii. The notes say the intended mapping was miss/incorrect = `0`, hit/correct = `1`, with ignore trials excluded rather than merged.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code uses `hit` directly as the binary label and asserts that `hit + miss == 1` on all kept trials.

ii.
```python
outcome = hit[keep_trials]
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(...)
```

iii. Step 5 and Step 10 justify this as preserving the hit/miss interpretation while removing no-response trials first.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It is derived from DeepLabCut trajectories in `obj.traj` for multiple tongue-related features from both camera views: `tongue`, `left_tongue`, `right_tongue`, `top_tongue`, `topleft_tongue`, `bottom_tongue`, and `bottomleft_tongue`.

ii.
```python
TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}
```

iii. The notes say this was a deliberate reduction from the richer reference kinematic representation because the decoder task asked for one tongue-velocity output.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each selected tongue feature, the code interpolates x/y position onto the neural time axis, computes x/y velocity with `np.gradient`, converts that to speed magnitude, and then averages speed across all available tongue features.

ii.
```python
xpos, ypos = aligned_position(obj, view_index, feat_name, time_centers, align_times, vidshift)
xvel, yvel = feature_velocity(xpos, ypos, feat_name)
speed = np.sqrt(np.square(xvel) + np.square(yvel))
...
stacked = np.stack(speed_components, axis=2)
agg = np.divide(summed, np.maximum(count, 1), dtype=np.float32)
```

iii. Step 5 explicitly says the agent chose to “collapse the multi-feature representation to a single tongue-speed trace using aggregate speed across the relevant tracked points.”

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The stated decision is to use a per-session 50th-percentile threshold over the continuous tongue-speed trace and label below-median as `0`, at-or-above-median as `1`. The implementation instead sorts all values and forces exactly half the samples to 1, ignoring the `threshold` argument.

ii.
```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)
tongue_bin = discretize_trace(tongue_speed, tongue_thr)

def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
```

iii. The notes repeatedly justify the decision as following the task’s per-session median binning requirement, not the paper’s manual movement threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is aligned by subtracting video offset and trial go-cue time from each feature’s frame times, then interpolating onto `time_centers`.

ii.
```python
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, axis=0, kind="linear", ...)
xy_aligned = interp(taxis)
```

iii. The notes say the agent copied the reference pattern used by `findPosition`/`getKinematicsFromVideo`, namely interpolation onto the neural time axis after video-offset correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It is derived from bottom-camera DeepLabCut trajectories for the `top_paw` and `bottom_paw` features in `obj.traj`.

ii.
```python
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}
```

iii. The notes say paws were taken only from the bottom view because that matches the reference tracking setup.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw processing mirrors tongue processing except paw positions/velocities use nearest-value filling and baseline-subtracted x/y velocities before speed magnitude is computed and averaged across the two paw features.

ii.
```python
if "tongue" not in feat_name:
    xv = xv - base[0]
    yv = yv - base[1]
    xv = fill_nearest_1d(xv)
    yv = fill_nearest_1d(yv)

speed = np.sqrt(np.square(xvel) + np.square(yvel))
```

iii. In Step 5, the agent justifies this as reusing the reference velocity logic and then reducing it to one paw-speed scalar because the decoder task asks for one paw-velocity output.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The stated decision is session-median thresholding. The actual code again ignores the numeric threshold and enforces a rank-based 50/50 split.

ii.
```python
paw_thr = percentile_threshold(paw_speed, 50.0)
paw_bin = discretize_trace(paw_speed, paw_thr)
```

iii. The notes justify the intended decision exactly the same way as tongue velocity: task-required median discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned to the neural bins by correcting frame times with video offset and go-cue time, then interpolating onto `time_centers`.

ii.
```python
old_time = frame_times - vidshift - float(align_times[trial])
xy_aligned = interp(taxis)
```

iii. The notes cite the same reference alignment logic used for other kinematic features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `motionEnergy_<subject>_<date>.mat` `me.data` when present, or `obj.me` for embedded motion-energy data, plus trial video frame times and go-cue times for alignment.

ii.
```python
if spec.motion_energy_path.exists():
    raw_me = read_mat(spec.motion_energy_path).get("me")
...
if "me" in obj:
    raw_me = obj["me"]
```

iii. The notes explain that this mirrors the released code path where ephys sessions use external motion-energy files and behavior-only sessions can embed motion energy in `obj.me`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code unwraps nested `me.data` structures if needed, interpolates each per-trial trace onto the neural time axis using corrected frame times, fills edge NaNs with nearest values, and converts remaining NaNs/Infs to zero.

ii.
```python
return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": move_thresh}

old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, me_trial, kind="linear", ...)
aligned[:, trial] = fill_nearest_1d(aligned[:, trial])
return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. Step 9 and Step 10 justify the nested-struct handling as a fix discovered for JEB15, and the rest as an attempt to reproduce `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The intended decision is per-session median thresholding, but the implementation again uses the rank-based `discretize_trace()` helper rather than the computed threshold.

ii.
```python
me_thr = percentile_threshold(motion_energy, 50.0)
me_bin = discretize_trace(motion_energy, me_thr)
```

iii. The notes say this deviation from the paper’s manual `moveThresh` was intentional because the decoder task required 50th-percentile binning.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Each motion-energy trace is aligned by subtracting video offset and the per-trial go-cue time from frame times, then interpolating onto the neural bin centers.

ii.
```python
if frame_times.size != me_trial.size:
    old_time = frame_times - 0.5 - float(align_times[trial])
else:
    old_time = frame_times - vidshift - float(align_times[trial])
```

iii. The notes explicitly reference `loadMotionEnergy.m` and `findVideoOffset.m` as the justification for this alignment rule.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses several fallbacks: default 400 Hz frame times when `frameTimes` are missing or mismatched, a fallback video offset of `0.5` s if bitcode-based alignment fails, nearest-value filling for non-tongue kinematics and motion energy, zero-filling for tongue-velocity NaNs, recursive unwrapping of nested motion-energy structs, zero output if motion energy is absent, and dropping late trials with no neural coverage.

ii.
```python
if frame_times is None:
    return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
...
return 0.5
...
x[mask] = x[nearest[mask]]
...
xv = np.nan_to_num(xv, nan=0.0)
...
if raw_me is None:
    return np.zeros((time_centers.size, n_trials), dtype=np.float32)
```

iii. Step 9 and Step 10 document these as fixes for real edge cases in JEB15 and JEB24 plus fallbacks meant to mimic the MATLAB code’s handling of absent `frameTimes` and edge NaNs.

## 11-a. What are the most time-consuming steps of the code?

i. The agent identifies large `.mat` loads, kinematic interpolation, and per-neuron spike binning as the runtime bottlenecks. The code structure supports that: every session loads a large MATLAB file, interpolates multiple video features trial-by-trial, and bins spikes neuron-by-neuron.

ii.
```python
obj = read_mat(spec.data_path)["obj"]
...
tongue_speed_all, tongue_feats = aggregate_speed(...)
paw_speed_all, paw_feats = aggregate_speed(...)
...
for out_idx, selected in enumerate(selected_neurons):
    rates = binned_neuron_trials(...)
```

iii. Step 6 says “Session loading from large MATLAB files is the main cost” and “Kinematic interpolation and per-neuron spike binning dominate runtime.”

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the trial loop in `aligned_position()`, the trial loop in `feature_velocity()`, the neuron loop in `select_neurons()`, and the nested neuron/trial writes used to assemble `neural_trials`.

ii.
```python
for trial in range(n_trials):
    ...

for trial in range(xpos.shape[1]):
    ...

for neuron_index in np.flatnonzero(quality_mask):
    ...

for out_idx, selected in enumerate(selected_neurons):
    ...
    for tr in range(n_trials):
        neural_trials[tr][out_idx, :] = rates[:, tr]
```

iii. The notes mention speedups but still acknowledge kinematic interpolation and per-neuron binning as dominant costs, implying these loops remained largely scalar.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats feature interpolation separately for tongue and paw groups, rescans feature names repeatedly, may rerun `select_neurons()` after neural-coverage trimming, and rebuilds per-trial constant input/output arrays in Python loops.

ii.
```python
tongue_speed_all, tongue_feats = aggregate_speed(...)
paw_speed_all, paw_feats = aggregate_speed(...)

selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
...
selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
```

iii. Step 6 says the agent optimized by restricting feature groups and using a two-pass neural path, which implies it was already aware of repeated work in the original design.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is computing numeric median thresholds and then not using them in `discretize_trace()`. The script also computes full continuous tongue/paw/motion traces even though the downstream decoder only consumes the binarized versions, and it builds plotting-related continuous summaries when `--show-processing` is enabled.

ii.
```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)
...
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")

tongue_speed_all, tongue_feats = aggregate_speed(...)
paw_speed_all, paw_feats = aggregate_speed(...)
motion_energy_all = aligned_motion_energy(...)
```

iii. The notes frame the continuous traces as needed for validation and plotting, but for the saved decoder outputs only the binary categories survive.
