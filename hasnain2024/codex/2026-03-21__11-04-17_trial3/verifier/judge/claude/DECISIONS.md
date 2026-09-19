# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI dynamically parses the MATLAB loader scripts (`load<ANM>_ALMVideo.m`) at runtime to extract session names, dates, and probe assignments. It uses `pymatreader.read_mat()` to load each `.mat` file. The session list is built by regex-parsing each loader file for `meta(end).date` and `meta(end).probe` entries, collecting them into `SessionSpec` dataclass instances.

ii.
```python
def parse_loader_file(loader_name: str, folder: str, task: str) -> List[SessionSpec]:
    path = LOADER_ROOT / loader_name
    subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
    ...
    date_re = re.compile(r"meta\(end\)\.date = '([^']+)'")
    probe_re = re.compile(r"meta\(end\)\.probe = (\[[^\]]+\]|[0-9]+)")
    ...

def get_reference_sessions() -> List[SessionSpec]:
    sessions: List[SessionSpec] = []
    for loader in FIXED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
    for loader in RANDOMIZED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))
    return sessions
```

iii. The AI parsed the loader scripts to find which sessions were included in the paper's analysis, rather than hardcoding the session list or globbing data directories.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the `SessionSpec.subject` field, which is parsed from the loader filename (e.g., `loadJEB6_ALMVideo.m` -> subject `JEB6`). Subject order is maintained in insertion order during dataset construction.

ii.
```python
subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
subject = subject_match.group(1)
...
if spec.subject not in subject_order:
    subject_order.append(spec.subject)
subject_idx.append(subject_order.index(spec.subject))
```

iii. Subject identity comes from the loader filenames, which is correct since each loader file is named after a mouse.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` represents one session, identified by `<subject>_<date>`. Sessions are loaded one at a time from their respective data folders (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). The AI finds 44 sessions across the two folders.

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

    @property
    def data_path(self) -> Path:
        return DATA_ROOT / self.folder / f"data_structure_{self.subject}_{self.date}.mat"
```

iii. Sessions are defined by the loader scripts, one `.mat` file per session.

## 1-d. How are the data split into trials?

i. Trials are indexed by position in the `obj.bp` arrays. The total number of trials per session comes from `obj.bp.Ntrials`. A boolean mask is built to filter valid trials, and the kept trial indices (0-based) are used to index into all data arrays.

ii.
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid

keep_trials = np.flatnonzero(trial_mask)
```

iii. The AI splits trials based on position in the Bpod arrays, which is the standard trial indexing.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) photostimulation trials (`stim.enable`) are removed, (2) early-lick trials (`early`) are removed, and (3) no-response/ignore trials (`no`) are removed. Additionally, trials beyond the last trial with neural coverage are dropped. Sessions with fewer than 2 valid trials or fewer than 10 units are skipped entirely.

ii.
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid
```

```python
if len(selected_neurons) < MIN_UNITS_PER_SESSION:
    log(f"Skipping {spec.session_id}: only {len(selected_neurons)} units after filtering")
    return None
```

iii. The AI's CONVERSION_NOTES.md documents excluding early-lick, photostim, and no-response trials. The no-response exclusion is documented as following behavioral analyses in the paper that require enough correct trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu` (spike-sorted clusters). Each cluster contains `trial` (1-based trial index for each spike), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). Go cue times from `obj.bp.ev.goCue` are used for alignment.

ii.
```python
probes = normalize_probe_container(obj.get("clu"))
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
...
trials = to_vector(probe["trial"][neuron_index], int)
trialtm = to_vector(probe["trialtm"][neuron_index], float)
```

iii. Same raw variables as the reference code's spike processing pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 5ms bins spanning -2.5 to +2.5s from the go cue, converted to firing rates (Hz), then smoothed with a **causal** Gaussian window of length 15 bins. The causal smoothing zeros out the first half of the kernel, so only past and current bins contribute.

ii.
```python
def causal_gaussian_smooth(x: np.ndarray, n: int, bctype: str = BCTYPE) -> np.ndarray:
    ...
    kern = gaussian_window(n)
    kern[: n // 2] = 0.0
    kern /= kern.sum()

    out = np.empty_like(x_filt, dtype=np.float64)
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    ...

def binned_neuron_trials(...) -> np.ndarray:
    ...
    counts = np.zeros((edges.size - 1, keep_trials_0based.size), dtype=np.float32)
    np.add.at(counts, (bin_index[valid], keep_index[valid]), 1.0)
    rates = counts / DT
    return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
```

iii. The AI chose causal smoothing to avoid information leakage from future time bins. The reference code uses symmetric (two-sided) Gaussian smoothing via `gaussian_filter1d`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) cluster quality labels are normalized and checked against a bad set of `{garbage, gabrga, noisy, real?}` - note that `poor` is NOT in the exclusion list. (2) Units with mean firing rate <= 1 Hz are excluded.

ii.
```python
def quality_keep_mask(qualities: Sequence[str]) -> np.ndarray:
    cleaned = [normalize_string(q).lower() for q in qualities]
    bad = {"garbage", "gabrga", "noisy", "real?"}
    return np.array([q not in bad for q in cleaned], dtype=bool)
...
mean_fr = neuron_mean_fr(probe, int(neuron_index), align_times, trial_mask)
if mean_fr > LOW_FR:
    kept_indices.append(int(neuron_index))
```

iii. The quality label exclusion list matches the reference `findClusters.m` exactly. The AI does not include `poor` in the drop set.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time (`trialtm`) is aligned to the go cue by subtracting the go cue time for that spike's trial. Spikes are then binned into 5ms bins from -2.5 to +2.5s.

ii.
```python
aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. This matches the reference alignment approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins spanning -2.5 to +2.5 s from go cue, yielding 1000 time bins per trial. No rebinning is applied - spikes are directly binned at 5 ms resolution.

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5

def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
```

iii. Matches the reference's `params.dt = 1/200` and `params.tmin/tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is derived from the bin centers of the time grid, which is defined by the go cue alignment window (-2.5 to +2.5 s at 5 ms resolution).

ii.
```python
time_centers = make_time_centers(time_edges)
...
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The input is the time axis itself, constructed from the binning parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as the center of each 5 ms bin: `edges[:-1] + DT/2`.

ii.
```python
def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0
```

iii. No special processing needed - it's a deterministic time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same grid as the neural binning, so they are inherently aligned.

ii.
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
# Same edges used for neural binning and input construction
```

iii. Both use the same 5 ms grid from -2.5 to +2.5 s.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived directly from `obj.bp.R`, which indicates the instructed lick direction (right=1, left=0).

ii.
```python
R = to_vector(bp["R"], float).astype(int)
...
lick_direction = R[keep_trials]
```

iii. The AI uses the instructed side (`R`) directly as the lick direction, rather than deriving the actual lick direction from the combination of instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The value of `R` is used directly: 0 for left, 1 for right. Only two classes are used (no "no lick" class). This is because ignore/no-response trials are already filtered out in the trial mask.

ii.
```python
lick_direction = R[keep_trials]
...
output_trials.append(
    np.vstack([
        np.full(time_centers.size, lick_direction[tr], dtype=np.int64),
        ...
    ])
)
```

Output values defined as:
```python
"output_values": [
    ["left", "right"],
    ...
]
```

iii. Since ignore trials are removed, all remaining trials have either a hit or a miss, making `R` equivalent to lick direction on hit trials and opposite on miss trials. However, `R` represents the *instructed* side, not the actual lick direction. On miss trials, `R` gives the wrong answer for actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `obj.bp.autowater`: autowater=1 means WC context (coded as 0), autowater=0 means DR context (coded as 1).

ii.
```python
autowater = to_vector(bp["autowater"], float).astype(int)
context = 1 - autowater[keep_trials]
```

iii. This is a direct relabeling of the autowater flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Simple inversion: `context = 1 - autowater`, so WC=0 and DR=1.

ii.
```python
context = 1 - autowater[keep_trials]
```

Output values:
```python
["WC", "DR"]
```

iii. Matches the reference encoding (WC=0, DR=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `obj.bp.hit` only. Hit=1 means correct, hit=0 means incorrect.

ii.
```python
hit = to_vector(bp["hit"], float).astype(int)
outcome = hit[keep_trials]
```

iii. Since ignore trials are filtered out, all remaining trials are either hits or misses, so `hit` alone suffices for a 2-class outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct use of the hit flag as outcome. Only two classes: incorrect=0, correct=1. No ignore class.

ii.
```python
outcome = hit[keep_trials]
...
"output_values": [
    ...
    ["incorrect", "correct"],
    ...
]
```

The code also validates:
```python
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(f"{spec.session_id}: hit/miss are not mutually exclusive on kept trials")
```

iii. With ignore trials excluded, every remaining trial is exactly a hit or a miss, so this is internally consistent.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` tracking data. The AI uses multiple tongue features from both cameras: side camera features `[tongue, left_tongue, right_tongue]` and bottom camera features `[top_tongue, topleft_tongue, bottom_tongue, bottomleft_tongue]`.

ii.
```python
TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}
...
tongue_speed_all, tongue_feats = aggregate_speed(obj, TONGUE_FEATURES, time_centers, align_times, vidshift)
```

iii. The AI included all available tongue-related DLC features from both cameras, aggregating them by averaging.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue feature: (1) x,y positions are interpolated (linear interp1d) from the camera frame times onto the 5ms time grid, (2) velocity is computed via `np.gradient` on the interpolated positions, (3) speed is the Euclidean norm of x and y velocity. For tongue features, NaN velocities are replaced with 0. All tongue feature speeds are averaged together.

ii.
```python
def aligned_position(obj, view_index_one_based, feat_name, time_centers, align_times, vidshift):
    ...
    interp = interp1d(old_time, xy, axis=0, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
    xy_aligned = interp(taxis)
    ...

def feature_velocity(xpos, ypos, feat_name):
    ...
    xv = np.gradient(tsinterp[:, 0]).astype(np.float32)
    yv = np.gradient(tsinterp[:, 1]).astype(np.float32)
    if "tongue" not in feat_name:
        xv = xv - base[0]
        yv = yv - base[1]
        ...
    else:
        xv = np.nan_to_num(xv, nan=0.0)
        yv = np.nan_to_num(yv, nan=0.0)
    ...
```

iii. The AI interpolates positions to the neural time grid first, then differentiates. The reference instead computes velocity at the native frame rate within contiguous valid runs, then bins.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI's `discretize_trace` function sorts all values and assigns the top 50% to class 1 and bottom 50% to class 0. There is no "not visible" (class 2) category, since NaN values are replaced with 0 before discretization.

ii.
```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

And NaN filling:
```python
agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The AI uses a rank-based 50/50 split rather than a median threshold, and has only 2 classes instead of the 3 specified in the instructions ("not visible" is missing).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Camera frame times are corrected using the video offset (computed from bitcode alignment), then the go cue time is subtracted. Positions are then linearly interpolated onto the same 5ms grid used for neural data.

ii.
```python
def compute_video_offset(obj: dict) -> float:
    bit_start = robust_mode(obj["bp"]["ev"]["bitStart"])
    vid_file_offset = robust_mode(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
    return float(vid_file_offset - bit_start)
...
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, axis=0, kind="linear", ...)
xy_aligned = interp(taxis)
```

iii. The video offset computation follows the reference's `findVideoOffset.m`. The alignment approach (interpolation vs binning) differs from the reference.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj` tracking data, using both `top_paw` and `bottom_paw` features from the bottom camera (view 2).

ii.
```python
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}
...
paw_speed_all, paw_feats = aggregate_speed(obj, PAW_FEATURES, time_centers, align_times, vidshift)
```

iii. The AI uses both paw features from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same processing pipeline as tongue velocity: positions are interpolated onto the 5ms grid, velocity computed via `np.gradient`, speed as Euclidean norm. For non-tongue features, baseline drift (median of frame-to-frame differences) is subtracted, and NaN values are filled with nearest-neighbor interpolation.

ii.
```python
def feature_velocity(xpos, ypos, feat_name):
    ...
    if "tongue" not in feat_name:
        xv = xv - base[0]
        yv = yv - base[1]
        xv = fill_nearest_1d(xv)
        yv = fill_nearest_1d(yv)
    ...
```

iii. The baseline subtraction and nearest-fill for paw features follow the reference code's `findVelocity.m` approach for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same rank-based 50/50 split as tongue velocity, with only 2 classes (no "not visible" class).

ii.
```python
paw_bin = discretize_trace(paw_speed, paw_thr)
```

iii. Same approach as tongue velocity discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same interpolation-based alignment as tongue velocity: frame times corrected by video offset and go cue, then linearly interpolated onto the 5ms grid.

ii. Same as 7-d.

iii. Same alignment method for all camera-derived features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from external `motionEnergy_<anm>_<date>.mat` files. Falls back to embedded `obj.me` if the external file doesn't exist.

ii.
```python
def load_motion_energy_raw(obj: dict, spec: SessionSpec) -> Optional[dict]:
    if spec.motion_energy_path.exists():
        raw_me = read_mat(spec.motion_energy_path).get("me")
        ...
    if "me" in obj:
        raw_me = obj["me"]
        ...
```

iii. Matches the reference's approach of loading external motion energy files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values (one per camera frame) are linearly interpolated onto the 5ms grid using `interp1d`. NaN values are filled with nearest-neighbor interpolation, then remaining NaNs are set to 0.

ii.
```python
def aligned_motion_energy(obj, raw_me, time_centers, align_times, vidshift):
    ...
    interp = interp1d(old_time, me_trial, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
    aligned[:, trial] = interp(taxis).astype(np.float32)
    aligned[:, trial] = fill_nearest_1d(aligned[:, trial])
    ...
    return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The reference bins motion energy by averaging frames in each bin, while the AI interpolates. The reference also preserves NaN for bins with no video data.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same rank-based 50/50 split, 2 classes only (no "no video" class).

ii.
```python
me_bin = discretize_trace(motion_energy, me_thr)
```

iii. Same as tongue and paw discretization. The instructions specify a "no video" class (2), which is not implemented.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera frame times corrected by the video offset and go cue, then interpolated onto the 5ms grid.

ii.
```python
frame_times = get_frame_times(trial_view, ts.shape[0])
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, me_trial, kind="linear", ...)
```

iii. Same interpolation-based alignment as other camera streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Missing/NaN frame times cause position arrays to be filled with NaN, which propagates to velocity as NaN, then gets replaced with 0 (tongue) or nearest-neighbor fill (paw). (2) Motion energy NaN values are filled with nearest-neighbor, then remaining NaN set to 0. (3) Sessions with fewer than 10 units or fewer than 2 valid trials are skipped entirely. (4) DLC tracking with mismatched dimensions is skipped.

ii.
```python
def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    ...
    x[mask] = x[nearest[mask]]
    return x

# For tongue:
xv = np.nan_to_num(xv, nan=0.0)

# For paw:
xv = fill_nearest_1d(xv)
```

iii. The AI fills missing data rather than preserving it as a separate category. The reference preserves NaN and maps it to a "not visible" output class.

## 11-a. What are the most time-consuming steps of the code?

i. Loading and parsing the MATLAB files is the most time-consuming step. The AI uses `pymatreader.read_mat()` which must parse potentially large HDF5 files. Additionally, per-neuron trial binning with `np.add.at` in a loop over neurons is relatively slow.

ii.
```python
obj = read_mat(spec.data_path)["obj"]
...
for out_idx, selected in enumerate(selected_neurons):
    rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
```

iii. File I/O dominates, consistent with the reference.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron loop for binning spikes (`binned_neuron_trials` called in a loop over `selected_neurons`), the per-trial loop for position interpolation (`aligned_position`), and the per-trial velocity computation (`feature_velocity`). The reference vectorizes spike counting with `np.histogram2d` across all trials at once.

ii.
```python
for out_idx, selected in enumerate(selected_neurons):
    probe = probes[selected.probe_num - 1]
    rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
    ...
    for tr in range(n_trials):
        neural_trials[tr][out_idx, :] = rates[:, tr]

for trial in range(n_trials):
    trial_view = get_view_trial(view, trial)
    ...
```

iii. The per-neuron spike binning loop is less efficient than the reference's single histogram2d call.

## 11-c. What processing does the code repeat multiple times?

i. The `get_view_trial` function is called multiple times for the same trial across different features (e.g., tongue features and paw features from the same camera view). Frame times are also recomputed for each feature. The `normalize_string` function is called repeatedly on the same values.

ii.
```python
for feat_name in feat_names:
    xpos, ypos = aligned_position(obj, view_index, feat_name, time_centers, align_times, vidshift)
    # Each call re-reads the view data and frame times
```

iii. The repeated view/frame-time access is redundant but not a major performance bottleneck compared to file I/O.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads the entire `obj` structure via `read_mat`, materializing fields like `spkWavs`, `tm`, and other metadata that are never used. The `percentile_threshold` function is computed but not actually used by `discretize_trace` (which ignores its threshold argument). The AI also computes velocities for all 7 tongue features when only a few are present in any given session.

ii.
```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1  # threshold parameter is ignored!
    return out.reshape(traces.shape)
```

iii. The threshold computation is wasted since discretization uses rank-based splitting instead.
