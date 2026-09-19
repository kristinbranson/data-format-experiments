# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not glob raw data folders. It parses the MATLAB loader scripts in `/app/code/DataLoadingScripts/Recording and video/` to recover the reference-selected sessions and probes, builds one `SessionSpec` per session, and then loads each selected `data_structure_<subject>_<date>.mat` file with `pymatreader.read_mat`. Motion-energy files are loaded separately, per session, if present.

ii. 
```python
def parse_loader_file(loader_name: str, folder: str, task: str) -> List[SessionSpec]:
    ...
    for raw_line in path.read_text().splitlines():
        ...
        if "datapth = fullfile" in line and current_date is not None and current_probe is not None:
            sessions.append(SessionSpec(...))

def get_reference_sessions() -> List[SessionSpec]:
    sessions = []
    for loader in FIXED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
    for loader in RANDOMIZED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))

obj = read_mat(spec.data_path)["obj"]
```

iii. In `CONVERSION_NOTES.md`, the agent says session inclusion should follow the reference loader files rather than scanning raw folders heuristically, and that parsing those loaders avoids pulling in extra raw files while matching the authors’ analyzed session set.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity comes from the loader filename/session specification: each `SessionSpec` stores `subject`, and sessions are assigned to subjects by that field. In the final dataset, `subjects` is the order in which new subjects first appear during session iteration, and `subject_idx` points each session to that ordered list.

ii. 
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str

...
if spec.subject not in subject_order:
    subject_order.append(spec.subject)
subject_idx.append(subject_order.index(spec.subject))
```

iii. The notes justify this by saying the reference loader files already encode the analyzed session list and animal IDs, so carrying `subject` forward from the parsed loader definitions is the cleanest way to preserve the paper/code session organization.

## 1-c. How are the data split into sessions?

i. One parsed `SessionSpec` is treated as one session. Each selected session becomes one element of `neural`, `input`, and `output` in the final dataset. The agent distinguishes fixed-delay and randomized-delay sessions by which loader file and raw subfolder produced the `SessionSpec`.

ii. 
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    probe: Tuple[int, ...]
    folder: str
    task: str

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"

for spec in session_specs:
    result = process_session(spec, show_processing=do_plot)
    ...
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
```

iii. The notes explicitly say session inclusion should match the reference ephys loaders: 25 fixed-delay sessions plus the 19 randomized-delay sessions listed in the authors’ loader scripts.

## 1-d. How are the data split into trials?

i. Trials are taken directly from per-trial arrays in `obj["bp"]`. A boolean `trial_mask` is built over all trials, and `keep_trials = np.flatnonzero(trial_mask)` defines the retained trial indices for every stream. Neural spikes stay attached to trials through `probe["trial"]`; video and motion-energy streams are iterated trial-by-trial using the per-trial entries in `obj["traj"]` and motion-energy lists.

ii. 
```python
trial_mask = build_trial_mask(obj)
keep_trials = np.flatnonzero(trial_mask)

trials = to_vector(probe["trial"][neuron_index], int)
...
for trial in range(n_trials):
    trial_view = get_view_trial(view, trial)
```

iii. The notes describe the raw object as trial-structured already: `obj.bp` contains one value per trial for task variables, `obj.clu` stores a `trial` index per spike, and `obj.traj` stores per-trial camera data.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials where `stim.enable == 0`, `early == 0`, and `no == 0`. After neuron selection, it also trims away any behavior-valid trials whose trial number is larger than the last trial containing spikes in the kept neurons. Sessions with fewer than two remaining trials are skipped.

ii. 
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid

...
neural_coverage_mask = (np.arange(trial_mask.size, dtype=np.int32) + 1) <= max_trial_with_spikes
refined_trial_mask = trial_mask & neural_coverage_mask
```

iii. The notes justify the first filter as keeping “control/non-stimulation trials with valid behavioral labels,” and say the no-response exclusion was chosen to keep outputs “unambiguous.” They justify the second filter as an edge-case fix for trials that continue after neural recording has stopped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the selected probe entries in `obj["clu"]`, specifically each neuron's `trial`, `trialtm`, and `quality` fields, together with `obj["bp"]["ev"]["goCue"]` for alignment. Probe-location metadata from `obj["ex"]["probe"]` are also read, but only for region labels.

ii. 
```python
trials = to_vector(probe["trial"][neuron_index], int)
trialtm = to_vector(probe["trialtm"][neuron_index], float)
qualities = [normalize_string(q) for q in probe["quality"]]
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
```

iii. The notes say the reference neural path is spike alignment to go cue, 5 ms binning, smoothing, and quality/FR curation, using the probe(s) selected by the reference loaders.

## 2-b. How is the `neural` data processed?

i. The agent aligns spike times to go cue, bins them into 5 ms bins from `-2.5` to `+2.5` s, divides counts by bin width to get firing rates, and applies a custom causal half-Gaussian smoothing filter with a 15-bin window and `reflect` padding. It does not z-score or baseline-subtract neural data.

ii. 
```python
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
counts = np.zeros((edges.size - 1, keep_trials_0based.size), dtype=np.float32)
np.add.at(counts, (bin_index[valid], keep_index[valid]), 1.0)
rates = counts / DT
return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)

def causal_gaussian_smooth(x: np.ndarray, n: int, bctype: str = BCTYPE) -> np.ndarray:
    kern = gaussian_window(n)
    kern[: n // 2] = 0.0
    kern /= kern.sum()
```

iii. The notes say the script “reproduces the reference neural processing steps needed here” and repeatedly describe the smoothing choice as “causal Gaussian smoothing” matched to the released code path.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent filters neurons in two stages: it removes clusters whose `quality` lower-cases to `garbage`, `gabrga`, `noisy`, or `real?`, then keeps only neurons with mean firing rate `> 1 Hz` over the aligned window and retained trials. It also skips sessions with fewer than 10 remaining units.

ii. 
```python
def quality_keep_mask(qualities: Sequence[str]) -> np.ndarray:
    cleaned = [normalize_string(q).lower() for q in qualities]
    bad = {"garbage", "gabrga", "noisy", "real?"}
    return np.array([q not in bad for q in cleaned], dtype=bool)

...
mean_fr = neuron_mean_fr(probe, int(neuron_index), align_times, trial_mask)
if mean_fr > LOW_FR:
    ...

if len(selected_neurons) < MIN_UNITS_PER_SESSION:
    log(f"Skipping {spec.session_id}: only {len(selected_neurons)} units after filtering")
    return None
```

iii. The notes justify this as matching the released-code quality filter plus the paper’s `>1 Hz` inclusion threshold, and add the paper-level session rule that analyzed sessions should have at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is converted from trial-relative time to go-cue-relative time by subtracting that spike’s trial-specific `goCue` time. No interpolation is used for spikes.

ii. 
```python
aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The notes explicitly say the decoder should use go-cue alignment to match the task and the main reference alignment path.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses a 5 ms time step over `[-2.5, 2.5]` s, yielding 1000 bins per trial. Spikes are directly binned to that resolution; video-derived streams are resampled onto that grid rather than kept at native frame rate.

ii. 
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5

def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
```

iii. The notes justify 5 ms as the paper’s decoder bin size and the correct common time axis for neural, input, and output streams.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a dedicated raw array. It is the synthetic common time axis implied by the chosen alignment window and 5 ms bins around go cue.

ii. 
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
...
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The notes describe the decoder input as the common aligned time axis itself, derived from the same bin centers used for neural data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The agent constructs bin edges from `TMIN`, `TMAX`, and `DT`, converts them to bin centers, and repeats the same `1 x 1000` vector for every trial.

ii. 
```python
def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)

def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0
```

iii. The notes justify this as the decoder’s required “time from go cue” input and a direct consequence of the chosen common binning grid.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same bin-center grid used for neural spike binning, so trial inputs and neural arrays share the exact same 1000 aligned time bins.

ii. 
```python
rates = counts / DT
return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
...
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The notes explicitly say the input should be “derived from the same bin centers as the neural data.”

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. In the implemented code, lick direction is derived only from `obj["bp"]["R"]` after filtering out `no` trials. `hit` and `miss` are read, but only for validating the outcome encoding, not for computing lick direction itself.

ii. 
```python
R = to_vector(bp["R"], float).astype(int)
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)

lick_direction = R[keep_trials]
```

iii. The notes justify removing no-response trials so “behavioral labels” stay unambiguous, and later sanity-check `lick_direction` directly against raw `bp.R`, indicating the agent treated instructed side as the lick-direction label.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is almost no processing: after trial filtering, the retained `R` values are copied directly into the output and repeated across all time bins of each trial. This produces a binary label with `0` meaning left and `1` meaning right, with no separate “none” class.

ii. 
```python
lick_direction = R[keep_trials]

output_trials.append(
    np.vstack(
        [
            np.full(time_centers.size, lick_direction[tr], dtype=np.int64),
            ...
        ]
    )
)
```

iii. The notes justify this indirectly through the decision to drop no-response trials; once `no` trials are removed, the agent treats `R` as sufficient to define a binary lick-direction target.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj["bp"]["autowater"]`.

ii. 
```python
autowater = to_vector(bp["autowater"], float).astype(int)
```

iii. The notes explicitly identify `autowater` as the code’s proxy for context and say the raw polarity must be remapped for the decoder output.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater == 1` to WC and `autowater == 0` to DR by computing `1 - autowater`. The resulting scalar is then repeated across time bins for each kept trial.

ii. 
```python
context = 1 - autowater[keep_trials]
...
np.full(time_centers.size, context[tr], dtype=np.int64)
```

iii. The notes justify this as an explicit relabeling to the decoder’s desired coding: WC=`0`, DR=`1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. In the implemented code, outcome is effectively derived from `obj["bp"]["hit"]` alone, after first dropping all `no` trials and checking that `hit + miss == 1` on the remaining trials. `miss` is only used as a consistency check.

ii. 
```python
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)

outcome = hit[keep_trials]
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(...)
```

iii. The notes justify excluding no-response trials and then treating the kept-trial outcome as a binary correct/incorrect variable.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code copies `hit` directly as the output label, so `1` means correct and `0` means incorrect. There is no retained `ignore` class because `no` trials were removed upstream.

ii. 
```python
outcome = hit[keep_trials]
...
np.full(time_centers.size, outcome[tr], dtype=np.int64)
```

iii. The notes say this was done to keep behavioral labels unambiguous after removing no-response trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj["traj"]` camera trajectories. The agent uses several tongue-related tracked features, not just one per camera: side-view `tongue`, `left_tongue`, `right_tongue` and bottom-view `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`. It also uses frame times, `goCue`, and the estimated video offset.

ii. 
```python
TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}

xy = ts[:, :2, feat_index]
old_time = frame_times - vidshift - float(align_times[trial])
```

iii. The notes justify this as a task-driven aggregation choice: the decoder specification asks for exactly one tongue-velocity output, so the agent says it collapsed the relevant tongue-tracking channels to a single scalar speed trace.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each selected tongue feature, the code linearly interpolates x/y positions onto the common 5 ms time grid, computes `np.gradient` of the interpolated positions, converts x/y derivatives to speed magnitude, and averages speeds across all available tongue-related features. For tongue features, NaNs in velocity are replaced by zeros rather than kept as missing.

ii. 
```python
interp = interp1d(old_time, xy, axis=0, kind="linear", bounds_error=False, fill_value=np.nan)
xy_aligned = interp(taxis)

xv = np.gradient(tsinterp[:, 0]).astype(np.float32)
yv = np.gradient(tsinterp[:, 1]).astype(np.float32)
...
else:
    xv = np.nan_to_num(xv, nan=0.0)
    yv = np.nan_to_num(yv, nan=0.0)

speed = np.sqrt(np.square(xvel) + np.square(yvel))
stacked = np.stack(speed_components, axis=2)
agg = np.divide(summed, np.maximum(count, 1), dtype=np.float32)
```

iii. The notes say the script “reproduces reference video/motion processing” and that it intentionally aggregates only the feature groups needed by the decoder instead of rebuilding the paper’s full kinematic feature matrix.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code computes a 50th-percentile value for the session, but the actual discretization ignores that threshold and instead sorts all bins and forces exactly half of them to class `1` and half to class `0`. There is no class `2` for not-visible bins.

ii. 
```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)
tongue_bin = discretize_trace(tongue_speed, tongue_thr)

def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

iii. The notes claim the movement outputs are “discretized by per-session median as required by the decoder task,” but the code actually implements an exact half-split by rank.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The agent estimates one session-wide video offset from bitcode timing, subtracts that offset and the trial’s `goCue` from frame times, then interpolates positions directly onto the same 5 ms bin-center axis used for neural data.

ii. 
```python
def compute_video_offset(obj: dict) -> float:
    bit_start = robust_mode(obj["bp"]["ev"]["bitStart"])
    vid_file_offset = robust_mode(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
    return float(vid_file_offset - bit_start)

old_time = frame_times - vidshift - float(align_times[trial])
xy_aligned = interp(taxis)
```

iii. The notes justify this as matching the reference video-offset correction and using a common aligned neural/video time axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is also derived from `obj["traj"]`. The agent uses two paw features from the bottom camera, `top_paw` and `bottom_paw`, plus frame times, `goCue`, and the video offset.

ii. 
```python
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}
```

iii. The notes justify this with the same aggregation logic as tongue velocity: the decoder wants one scalar paw-velocity output rather than the paper’s fuller feature set.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw feature is linearly interpolated onto the 5 ms grid, differentiated with `np.gradient`, baseline-adjusted by subtracting the median frame-to-frame difference, nearest-filled over NaNs, converted to speed magnitude, and then averaged across the two paw features.

ii. 
```python
xv = np.gradient(tsinterp[:, 0]).astype(np.float32)
yv = np.gradient(tsinterp[:, 1]).astype(np.float32)
if "tongue" not in feat_name:
    xv = xv - base[0]
    yv = yv - base[1]
    xv = fill_nearest_1d(xv)
    yv = fill_nearest_1d(yv)

speed = np.sqrt(np.square(xvel) + np.square(yvel))
```

iii. The notes describe this generally as reference-style video interpolation and per-feature velocity calculation, then aggregation to the single paw-speed trace required by the decoder task.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. As with tongue velocity, the code computes a nominal session median but actually applies an exact half-split by sorted rank, producing only classes `0` and `1` and never class `2` for not visible.

ii. 
```python
paw_thr = percentile_threshold(paw_speed, 50.0)
paw_bin = discretize_trace(paw_speed, paw_thr)
```

iii. The notes justify this as session-median binning, but the implementation is the same rank-based split described in 7-c.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned exactly like tongue trajectories: frame times are corrected by a session-wide video offset, then shifted by trial-specific `goCue`, then interpolated onto the neural 5 ms time grid.

ii. 
```python
old_time = frame_times - vidshift - float(align_times[trial])
xy_aligned = interp(taxis)
```

iii. The notes justify using the same shared video/neural time axis for all movement-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from external `motionEnergy_<session>.mat` files when they exist; otherwise the code falls back to `obj["me"]` if present. The per-trial motion-energy traces are paired with side-camera frame times from `obj["traj"][0]`, plus `goCue` and the video offset.

ii. 
```python
if spec.motion_energy_path.exists():
    raw_me = read_mat(spec.motion_energy_path).get("me")
    ...
if "me" in obj:
    raw_me = obj["me"]
```

iii. The notes justify this as robust handling of the dataset’s mixed motion-energy layouts while preserving the reference session set.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code unwraps nested motion-energy structs, linearly interpolates each trial’s framewise motion-energy trace onto the common 5 ms grid, nearest-fills missing values, converts any remaining NaNs to zero, and then binarizes the result.

ii. 
```python
me_trial = np.asarray(me_data[trial], dtype=np.float32).reshape(-1)
...
interp = interp1d(old_time, me_trial, kind="linear", bounds_error=False, fill_value=np.nan)
aligned[:, trial] = interp(taxis).astype(np.float32)
aligned[:, trial] = fill_nearest_1d(aligned[:, trial])

return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The notes say the script “reproduces reference video/motion processing” and that movement outputs use aligned continuous traces before discretization.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The code again computes a 50th-percentile value but actually assigns classes by exact median split over sorted bins, with only `0` and `1` categories and no `2` class for missing/no-video data.

ii. 
```python
me_thr = percentile_threshold(motion_energy, 50.0)
me_bin = discretize_trace(motion_energy, me_thr)
```

iii. The notes justify this as per-session median binning, but the implementation is the same threshold-ignoring rank split used for the other movement outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. For each trial, the agent pairs motion-energy values with side-camera frame times, subtracts the session-wide video offset and the trial’s `goCue`, and linearly interpolates the result onto the same 5 ms neural grid.

ii. 
```python
view = obj["traj"][0]
...
frame_times = get_frame_times(trial_view, ts.shape[0])
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, me_trial, kind="linear", bounds_error=False, fill_value=np.nan)
aligned[:, trial] = interp(taxis).astype(np.float32)
```

iii. The notes justify this as using the same offset-corrected aligned time axis for motion energy and the other video-derived streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or awkward data by filling or falling back rather than by preserving a missing-data category. Missing/invalid frame times are replaced with a synthetic 400 Hz clock; failure to compute video offset falls back to `0.5` s; non-tongue NaNs are nearest-filled; tongue NaNs in velocity become zeros; missing motion-energy arrays yield all-zero traces; aligned motion-energy NaNs are nearest-filled and then zero-filled. The code also recursively unwraps nested motion-energy structs and drops late trials with no neural coverage.

ii. 
```python
if frame_times is None:
    return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
if arr.size != n_frames or not np.isfinite(arr).any():
    return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0

except Exception:
    pass
return 0.5

x[mask] = x[nearest[mask]]
...
xv = np.nan_to_num(xv, nan=0.0)
...
if raw_me is None:
    return np.zeros((time_centers.size, n_trials), dtype=np.float32)
```

iii. The notes explicitly justify only two of these cases: recursive motion-energy unwrapping and dropping behavior-valid trials after neural recording ended. The broader fill/fallback strategy is implicit in the implementation and appears aimed at avoiding NaNs and passing decoder-format verification.

## 11-a. What are the most time-consuming steps of the code?

i. According to the notes, loading the large MATLAB session files is the main cost, with kinematic interpolation and per-neuron spike binning as the other dominant costs. This matches the structure of the code, which repeatedly interpolates video features and loops over selected neurons to build trial matrices.

ii. 
```python
obj = read_mat(spec.data_path)["obj"]
...
for view_index, feat_names in feature_map.items():
    for feat_name in feat_names:
        xpos, ypos = aligned_position(...)
        ...
for out_idx, selected in enumerate(selected_neurons):
    rates = binned_neuron_trials(...)
```

iii. `CONVERSION_NOTES.md` explicitly says “Session loading from large MATLAB files is the main cost” and that “Kinematic interpolation and per-neuron spike binning dominate runtime.”

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several large Python loops that could in principle be reduced: per-trial loops in `aligned_position` and `feature_velocity`, per-feature loops in `aggregate_speed`, and nested loops that copy each neuron’s rates into each trial matrix. Some of these are partly constrained by ragged raw video arrays, but the implementation is not aggressively vectorized.

ii. 
```python
for trial in range(n_trials):
    trial_view = get_view_trial(view, trial)
    ...

for trial in range(xpos.shape[1]):
    tsinterp = np.column_stack([xpos[:, trial], ypos[:, trial]])
    ...

for out_idx, selected in enumerate(selected_neurons):
    ...
    for tr in range(n_trials):
        neural_trials[tr][out_idx, :] = rates[:, tr]
```

iii. The notes do not give a detailed vectorization analysis. They instead emphasize high-level speedups: two-pass neuron selection, parsing loader files once, and only computing the movement feature groups needed for the decoder.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats some work. It recomputes time edges/centers inside every `process_session` call, may rerun `select_neurons` after neural-coverage trimming, and independently interpolates multiple related tongue/paw features before averaging them. It also computes percentile thresholds that the discretizer does not actually use.

ii. 
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)

selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
...
selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)

tongue_thr = percentile_threshold(tongue_speed, 50.0)
tongue_bin = discretize_trace(tongue_speed, tongue_thr)
```

iii. The notes justify the explicit two-pass neural logic as a speed/memory optimization: first estimate firing rates to decide which neurons to keep, then build trial matrices only for kept neurons. They do not discuss the unused threshold values.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are not used downstream. The clearest bug is that the session percentile thresholds are computed and stored but never used by `discretize_trace`. The script also loads/unwraps `moveThresh` from motion-energy files but never uses it, optionally imports plotting machinery for diagnostics, and computes/stores continuous movement traces only to immediately collapse them to binary outputs for the saved dataset.

ii. 
```python
move_thresh = np.nan
...
if "moveThresh" in cursor:
    move_thresh = cursor["moveThresh"]
...
return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": move_thresh}

tongue_thr = percentile_threshold(tongue_speed, 50.0)
...
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
```

iii. The notes only partially acknowledge this. They mention that the script intentionally computes only the output traces needed by the decoder, but they do not call out that `moveThresh` and the computed percentile values are discarded by the actual saved-data path.
