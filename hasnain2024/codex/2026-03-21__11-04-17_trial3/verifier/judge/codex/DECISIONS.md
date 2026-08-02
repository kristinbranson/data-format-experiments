# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not scan every raw file directly. It first reconstructs the analyzed session list by parsing the reference MATLAB loader files, then loads each selected session `.mat` file with `pymatreader.read_mat`. This yields the fixed-delay and randomized-delay ephys sessions encoded by the reference loaders.

ii. ```python
def parse_loader_file(loader_name: str, folder: str, task: str) -> List[SessionSpec]:
    ...
    for raw_line in path.read_text().splitlines():
        ...
        if "datapth = fullfile" in line and current_date is not None and current_probe is not None:
            sessions.append(SessionSpec(subject=subject, date=current_date, probe=current_probe, folder=folder, task=task))

def get_reference_sessions() -> List[SessionSpec]:
    sessions: List[SessionSpec] = []
    for loader in FIXED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
    for loader in RANDOMIZED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))
    return sessions

obj = read_mat(spec.data_path)["obj"]
```

iii. In `CONVERSION_NOTES.md`, the agent says session inclusion should follow the analysis loaders rather than raw folder contents, especially to resolve the randomized-delay session-count discrepancy. The trajectory also shows it auditing loader parsing until it recovered the intended 44 sessions.

## 1-b. How are the data split into subjects?

i. The data are split into subjects by animal identifier extracted from the MATLAB loader filename, e.g. `loadJEB15_ALMVideo.m -> JEB15`. Sessions are grouped by first-seen subject order when building `subjects` and `subject_idx`.

ii. ```python
subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
subject = subject_match.group(1)
...
"subjects": subject_order,
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The notes explicitly treat the loader metadata as the reference definition of the analyzed mice, and the trajectory mentions using those loader files to resolve paper-versus-data count discrepancies.

## 1-c. How are the data split into sessions?

i. Each `(subject, date, probe)` entry parsed from the loader scripts becomes one `SessionSpec`. A session is therefore one reference loader entry, not one raw folder. If a session uses two probes, both probes are pooled into the same output session rather than split apart.

ii. ```python
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
...
for probe_num in spec.probe:
    ...
    selected.append(SelectedNeuron(probe_num=probe_num, neuron_index=int(neuron_index), region_label=region_label))
```

iii. The notes and trajectory both emphasize matching `loadSessionData.m`, including the later fix for `JEB15` multi-probe sessions after the agent discovered the parser had initially dropped `[1 2]` probe specifications.

## 1-d. How are the data split into trials?

i. Trials are taken from the raw behavioral trial axis within each selected session. After filtering, each kept trial becomes one list element in `neural`, `input`, and `output`, with neural data shaped `(neurons, time)` and inputs/outputs shaped `(features, time)`.

ii. ```python
trial_mask = build_trial_mask(obj)
keep_trials = np.flatnonzero(trial_mask)
...
n_trials = keep_trials.size
neural_trials = [np.zeros((n_neurons, time_centers.size), dtype=np.float32) for _ in range(n_trials)]
...
for tr in range(n_trials):
    input_trials.append(time_centers[None, :].astype(np.float32))
    output_trials.append(np.vstack([...]))
```

iii. The notes say the agent wanted “all valid trials” after reference-style curation rather than condition-averaged PSTHs, because the target decoder format is session-by-trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding stimulation trials, early-lick trials, and no-response trials. After neuron selection, the code also trims any behavior-valid trials occurring after the last raw trial index that still has spikes in the selected neural data.

ii. ```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid
...
max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
neural_coverage_mask = (np.arange(trial_mask.size, dtype=np.int32) + 1) <= max_trial_with_spikes
trial_mask = trial_mask & neural_coverage_mask
```

iii. `CONVERSION_NOTES.md` says the agent intentionally excluded `stim.enable`, `early`, and `no` trials to keep output definitions unambiguous. The trajectory shows a later patch after verify-only revealed all-zero neural trials at the end of two `JEB24` sessions; it then trimmed those late behavior-only tails.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from per-cluster spike trial indices and spike times in `obj.clu[probe]["trial"]` and `obj.clu[probe]["trialtm"]`, aligned to `obj.bp.ev.goCue`.

ii. ```python
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
...
trials = to_vector(probe["trial"][neuron_index], int)
trialtm = to_vector(probe["trialtm"][neuron_index], float)
aligned = trialtm[mask] - align_times[trials[mask] - 1]
```

iii. The notes explicitly map `obj.clu{probe}.trialtm`, `obj.clu{probe}.trial`, and `obj.bp.ev.goCue` to the converted `neural` field, citing `alignSpikes` and `getSeq`.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, binned from `-2.5` to `2.5` s in 5 ms bins, converted to firing rates by dividing counts by `DT`, and smoothed with a causal Gaussian kernel intended to mirror MATLAB `mySmooth(..., 15, 'reflect')`.

ii. ```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
SMOOTH = 15
...
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts = np.zeros((edges.size - 1, keep_trials_0based.size), dtype=np.float32)
np.add.at(counts, (bin_index[valid], keep_index[valid]), 1.0)
rates = counts / DT
return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
```

iii. The notes say the agent chose 5 ms bins because the paper’s decoding description uses 5 ms bins, even though some MATLAB figure scripts use 10 ms. The trajectory also says it was trying to reproduce the reference smoothing/alignment logic directly instead of inventing a new pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered in three layers: only loader-selected probes are used; clusters with bad quality labels are excluded; remaining neurons must exceed 1 Hz mean firing rate; sessions with fewer than 10 kept units are skipped.

ii. ```python
bad = {"garbage", "gabrga", "noisy", "real?"}
...
for neuron_index in np.flatnonzero(quality_mask):
    mean_fr = neuron_mean_fr(probe, int(neuron_index), align_times, trial_mask)
    if mean_fr > LOW_FR:
        selected.append(...)
...
if len(selected_neurons) < MIN_UNITS_PER_SESSION:
    log(f"Skipping {spec.session_id}: only {len(selected_neurons)} units after filtering")
    return None
```

iii. The notes directly cite `findClusters`, `removeLowFRClusters`, and the paper’s “>1 Hz” and “at least 10 units” rules. The trajectory shows the agent using those same thresholds during implementation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to auditory go-cue onset by subtracting `bp.ev.goCue` from each spike time on the corresponding trial before binning.

ii. ```python
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
...
aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The notes repeatedly state that go cue was chosen because both the user instructions and the main reference analyses use go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 5 ms bins. Neural spikes are directly binned at that resolution; no later temporal rebinning is applied.

ii. ```python
DT = 0.005
...
def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
```

iii. The notes say the 5 ms choice came from the paper’s decoding methods and was preferred over generic 10 ms examples in other scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the chosen go-cue alignment plus the synthetic binning grid, not from a dedicated stored raw variable. The raw ingredient is `obj.bp.ev.goCue`; the actual time axis is built from `TMIN`, `TMAX`, and `DT`.

ii. ```python
def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)

def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0
```

iii. The notes describe the decoder input as the common aligned time axis used by `getSeq`-style processing, rather than a separate sensor stream.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code creates bin centers spanning `[-2.5, 2.5)` relative to go cue, then uses the same vector for every trial in the session.

ii. ```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
...
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The notes say the time input should exactly match the neural time axis for every session and trial.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is perfectly aligned by construction: the same `time_centers` vector is used both for neural binning and for the decoder input.

ii. ```python
rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
...
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The planned sanity checks in the notes explicitly include verifying that `input[0]` exactly matches the converted time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` on kept trials, with `R=1` treated as right and `R=0` as left.

ii. ```python
bp = obj["bp"]
R = to_vector(bp["R"], float).astype(int)
...
lick_direction = R[keep_trials]
```

iii. The notes say left/right choice comes from the raw `R`/`L` trial labels; the implementation uses `R` directly because it already encodes the binary class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code just subsets the per-trial binary label after trial filtering and then repeats that static label across all time bins for each kept trial.

ii. ```python
lick_direction = R[keep_trials]
...
np.full(time_centers.size, lick_direction[tr], dtype=np.int64)
```

iii. The notes treat lick direction as a per-trial categorical output, not a time-varying trace.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`.

ii. ```python
autowater = to_vector(bp["autowater"], float).astype(int)
context = 1 - autowater[keep_trials]
```

iii. The notes explicitly identify `autowater` as the code’s proxy for context: raw `autowater==1` is water-cued and raw `autowater==0` is delayed-response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code remaps raw `autowater` polarity to the requested output encoding `WC=0, DR=1`, then repeats that per-trial label across time bins.

ii. ```python
context = 1 - autowater[keep_trials]
...
np.full(time_centers.size, context[tr], dtype=np.int64)
```

iii. The notes call this out as an intentional remapping so the saved output values match the user instruction rather than the raw MATLAB polarity.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit` and checked against `obj.bp.miss`.

ii. ```python
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)
...
outcome = hit[keep_trials]
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(...)
```

iii. The notes say the desired outcome is miss versus hit, with no-response trials removed instead of being merged into “incorrect”.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. After trial filtering removes no-response trials, the code uses `hit` as a binary correct/incorrect label and validates that hit and miss are mutually exclusive on the retained trials. The label is then repeated across time.

ii. ```python
outcome = hit[keep_trials]
if not np.all((outcome == 0) | (outcome == 1)):
    raise ValueError(...)
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(...)
```

iii. The trajectory and notes both justify excluding `no` trials so the outcome variable remains a clean hit-vs-miss binary.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC tongue feature trajectories in `obj.traj` from both views. The code uses the tongue-related feature lists hard-coded in `TONGUE_FEATURES`.

ii. ```python
TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}
...
tongue_speed_all, tongue_feats = aggregate_speed(obj, TONGUE_FEATURES, time_centers, align_times, vidshift)
```

iii. The notes say this was a required deviation from the reference feature set because the decoder task asks for exactly one tongue-velocity output instead of the full kinematic tensor.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue feature, the code interpolates x/y positions onto the neural time axis, takes gradients to get x/y velocity, converts that to Euclidean speed, and then averages speed across all available tongue features to produce one scalar tongue-speed trace per trial.

ii. ```python
xy_aligned = interp(taxis)
...
xv = np.gradient(tsinterp[:, 0]).astype(np.float32)
yv = np.gradient(tsinterp[:, 1]).astype(np.float32)
...
speed = np.sqrt(np.square(xvel) + np.square(yvel))
...
agg = np.divide(summed, np.maximum(count, 1), dtype=np.float32)
```

iii. The notes explain that the agent first followed the reference interpolation and velocity logic, then collapsed multiple tongue channels to one speed trace because of the decoder schema.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code computes the 50th percentile value for the session but does not actually threshold by that value. Instead, it performs a rank-based median split so exactly half the samples are labeled `1`, even when many values tie at the median.

ii. ```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)
...
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

iii. The trajectory explicitly says the agent changed from literal thresholding to rank-based splitting after sample verification revealed tongue bins collapsing to one class because of ties at the median.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is aligned by interpolating raw video trajectories onto the same go-cue-centered time axis used for neural binning, after correcting for video offset.

ii. ```python
vidshift = compute_video_offset(obj)
old_time = frame_times - vidshift - float(align_times[trial])
xy_aligned = interp(taxis)
...
tongue_speed_all, tongue_feats = aggregate_speed(obj, TONGUE_FEATURES, time_centers, align_times, vidshift)
```

iii. The notes repeatedly reference `findVideoOffset`, `findPosition`, and `getKinematicsFromVideo` as the intended alignment model.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DLC paw features defined in `PAW_FEATURES`, specifically `top_paw` and `bottom_paw`.

ii. ```python
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}
...
paw_speed_all, paw_feats = aggregate_speed(obj, PAW_FEATURES, time_centers, align_times, vidshift)
```

iii. The notes say paws are tracked only in the bottom view and are collapsed to one scalar paw-speed output for the decoder.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code uses the same interpolation and velocity pipeline as for tongue, but for paw features. Non-tongue features are baseline-subtracted and nearest-filled before Euclidean speed is computed and averaged across the selected paw points.

ii. ```python
if "tongue" not in feat_name:
    xv = xv - base[0]
    yv = yv - base[1]
    xv = fill_nearest_1d(xv)
    yv = fill_nearest_1d(yv)
...
speed = np.sqrt(np.square(xvel) + np.square(yvel))
```

iii. The notes describe this as reference-style kinematic preprocessing followed by a task-driven collapse to one paw-speed trace.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Like tongue velocity, paw velocity is nominally assigned a session median but is actually binarized by rank order, not by comparing to the percentile value.

ii. ```python
paw_thr = percentile_threshold(paw_speed, 50.0)
paw_bin = discretize_trace(paw_speed, paw_thr)
```

iii. The trajectory says the agent intentionally used the same rank-based split for all three continuous movement outputs to avoid degenerate one-class bins.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are interpolated onto the same go-cue-centered `time_centers` axis used for the neural data, with the same video-offset correction.

ii. ```python
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, axis=0, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
```

iii. The notes say the behavioral streams were intentionally interpolated to the neural axis rather than left at native video frame rate.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motionEnergy_*.mat` session files when present, or from embedded `obj.me` otherwise. In both cases the actual trace comes from the `me.data` field.

ii. ```python
if spec.motion_energy_path.exists():
    raw_me = read_mat(spec.motion_energy_path).get("me")
    ...
    return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": move_thresh}
if "me" in obj:
    raw_me = obj["me"]
    ...
```

iii. The notes identify `loadMotionEnergy` as the reference function and mention the later fix for nested `me.data` structs discovered during the full run.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy traces are unwrapped from MATLAB structures, interpolated onto the neural time axis using video-frame times and video offset correction, nearest-filled at edges, and then converted to a binary output.

ii. ```python
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, me_trial, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
aligned[:, trial] = interp(taxis).astype(np.float32)
aligned[:, trial] = fill_nearest_1d(aligned[:, trial])
```

iii. The notes say the agent followed the reference motion-energy alignment logic but replaced the paper’s manual move threshold with the task-mandated sessionwise binarization rule.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. As with tongue and paw, the code computes the session median but then uses rank-based bin splitting rather than literal `< median` versus `>= median` thresholding.

ii. ```python
me_thr = percentile_threshold(motion_energy, 50.0)
me_bin = discretize_trace(motion_energy, me_thr)
```

iii. The trajectory says this was a deliberate choice made after the sample verifier exposed degenerate class balance under literal median thresholding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It is aligned to go cue by interpolating each trial’s motion-energy trace onto the shared neural time axis after subtracting the same go-cue alignment time and video offset.

ii. ```python
taxis = time_centers + ADVANCE_MOVEMENT
old_time = frame_times - vidshift - float(align_times[trial])
aligned[:, trial] = interp(taxis).astype(np.float32)
```

iii. The notes explicitly compare this to `loadMotionEnergy.m` and list motion-energy alignment as a required sanity-check target.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses several fallbacks: it reconstructs frame times as `1/400` s if missing; it nearest-fills NaNs for non-tongue positions, non-tongue velocities, and motion energy; tongue NaN velocities are set to zero; missing motion-energy data return zeros; sessions or trials with insufficient neural support are skipped or trimmed.

ii. ```python
if frame_times is None:
    return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
...
x[mask] = x[nearest[mask]]
...
xv = np.nan_to_num(xv, nan=0.0)
yv = np.nan_to_num(yv, nan=0.0)
...
if raw_me is None:
    return np.zeros((time_centers.size, n_trials), dtype=np.float32)
```

iii. The trajectory documents multiple defensive patches: handling single-dict `clu` objects, nested motion-energy structs, and late trials with missing neural coverage. The notes also emphasize copying the reference code’s habit of filling missing video values rather than dropping whole sessions immediately.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive work is per-session raw MATLAB loading, per-feature/per-trial interpolation for kinematics and motion energy, and per-neuron/per-trial spike binning followed by assembling `neural_trials`.

ii. ```python
obj = read_mat(spec.data_path)["obj"]
...
for trial in range(n_trials):
    ...
    interp = interp1d(...)
...
for out_idx, selected in enumerate(selected_neurons):
    rates = binned_neuron_trials(...)
    for tr in range(n_trials):
        neural_trials[tr][out_idx, :] = rates[:, tr]
```

iii. The trajectory repeatedly comments on session runtime and identifies large file loads and later per-session processing as the main contributors to the 7.5-minute full pass.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization targets: the per-column convolution loop in `causal_gaussian_smooth`, the per-trial loops in `aligned_position` and `aligned_motion_energy`, and the nested neuron-by-trial copy loop when building `neural_trials`.

ii. ```python
for col in range(x_filt.shape[1]):
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
...
for trial in range(n_trials):
    ...
for out_idx, selected in enumerate(selected_neurons):
    ...
    for tr in range(n_trials):
        neural_trials[tr][out_idx, :] = rates[:, tr]
```

iii. The trajectory does not present these as design choices, but it does show the agent monitoring runtime rather than refactoring for vectorization once the full pass stayed within estimate.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats similar interpolation/velocity work separately for every tongue and paw feature, computes neuron selection twice if trial trimming changes the mask, and separately recomputes aligned feature streams for tongue, paw, and motion energy even though they share the same aligned time grid.

ii. ```python
tongue_speed_all, tongue_feats = aggregate_speed(...)
paw_speed_all, paw_feats = aggregate_speed(...)
motion_energy_all = aligned_motion_energy(...)
...
selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
...
selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
```

iii. The trajectory shows one explicit repeated pass: after discovering late trials without neural coverage, the agent recomputes neuron selection with the refined trial mask.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is computing percentile thresholds that are not actually used by `discretize_trace`. The code also expands static trial labels (`lick_direction`, `context`, `outcome`) to full time-varying arrays even though the values do not change within a trial, and it records extra plotting/session metadata not needed by the downstream decoder.

ii. ```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)
...
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
...
np.full(time_centers.size, lick_direction[tr], dtype=np.int64)
np.full(time_centers.size, context[tr], dtype=np.int64)
np.full(time_centers.size, outcome[tr], dtype=np.int64)
```

iii. The trajectory explains why the threshold values remain in metadata even after switching to rank-based splitting: the agent wanted to preserve the nominal 50th-percentile quantity for documentation and diagnostic plots.
