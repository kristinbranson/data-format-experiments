# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reconstructs the cohort by parsing the released MATLAB loader scripts rather than hard-coding the 44 sessions. It reads `Figure3h.m` to find the animals, reads each `load<animal>_ALMVideo.m` file to recover session dates and probe choices, then loads each `data_structure_<animal>_<date>.mat` with `mat73`, falling back to `scipy.io.loadmat`. Motion-energy files are loaded separately when needed.

ii.
```python
def get_session_specs(sample: bool) -> list[SessionSpec]:
    fixed_animals = parse_loader_animals(FIXED_SCRIPT, marker="fixmeta")
    randomized_animals = parse_loader_animals(RANDOMIZED_SCRIPT, marker="randmeta")
    ...

def parse_loader_sessions(animal: str, cohort: str, folder: str) -> list[SessionSpec]:
    text = (LOADER_DIR / f"load{animal}_ALMVideo.m").read_text()
    ...
    if m_probe and current_date:
        probes = tuple(int(x) for x in re.findall(r"\d+", m_probe.group(1)))
        sessions.append(SessionSpec(...))

def load_mat_file(path: Path) -> dict[str, Any]:
    try:
        return mat73.loadmat(str(path))
    except Exception:
        return scipy.io.loadmat(str(path), simplify_cells=True)
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI says it wanted a loader-defined cohort that mirrors the released figure scripts rather than raw-folder enumeration, and it notes that mixed MATLAB layouts require both readers.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the animal id carried in each parsed `SessionSpec`. The converter assigns one subject index per unique `spec.animal` in encounter order and stores that index per saved session.

ii.
```python
subject_idx = subject_to_idx.setdefault(spec.animal, len(subject_to_idx))
...
converted["subjects"] = [subject for subject, _ in sorted(subject_to_idx.items(), key=lambda kv: kv[1])]
converted["subject_idx"] = np.asarray(converted["subject_idx"], dtype=np.int64)
```

iii. The notes explicitly justify using the loader/file naming metadata because session membership comes from `load*_ALMVideo.m`, and the trajectory shows the agent treating the animal id as the reliable subject key.

## 1-c. How are the data split into sessions?

i. One parsed `SessionSpec` becomes one session. Each saved session corresponds to one `data_structure_<animal>_<date>.mat` file plus its selected probe list and, if present, its matching motion-energy file in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    cohort: str
    folder: str
    animal: str
    date: str
    probes: tuple[int, ...]

    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"
```

```python
for sess_num, spec in enumerate(specs, start=1):
    obj = load_mat_file(spec.data_path)["obj"]
    processed = process_session(spec, obj, ...)
```

iii. In Steps 4 and 10, the AI says it intentionally mirrors the loader-defined fixed/randomized cohorts, not all raw files, because that is what the released paper code actually analyzes.

## 1-d. How are the data split into trials?

i. Trials are indexed directly from `bp.Ntrials`, with trial ids `0..Ntrials-1`. Per-trial behavioral labels come from whole-trial boolean arrays; neural spikes use `unit["trial"] - 1`; video and motion-energy traces are processed trial-by-trial using the same trial index.

ii.
```python
n_trials = int(gget(bp, "Ntrials", 0) or 0)
...
trial_ids = np.asarray(unit["trial"], dtype=np.int64).reshape(-1) - 1
...
for tr in keep_trials:
    neural_trials.append(trialdat[:, tr, :].astype(np.float32, copy=False))
```

iii. The AI’s notes repeatedly describe one Bpod trial as the atomic unit and mention a late-session neural-coverage edge case, but not any need to reconstruct trial boundaries from timestamps.

## 1-e. How are trials filtered based on quality controls?

i. The saved trials are those satisfying `(hit | miss | no) & ~early & ~stim & neural_covered`. This removes early-lick and stimulation trials and also removes trials after the last trial with any neural spikes.

ii.
```python
neural_covered = np.zeros(n_trials, dtype=bool)
...
neural_covered[trial_ids] = True
...
keep_trials = np.flatnonzero((hit | miss | no) & ~early & ~stim & neural_covered)
if keep_trials.size < MIN_TRIALS_PER_SESSION:
    return None
```

iii. In Step 10, the AI explicitly says it added `neural_covered` after verify warnings revealed late behavioral-only trials in two `JEB24` sessions. The notes also say the `early` and `stim` exclusions were taken from the reference trial logic.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu`, specifically each unit’s `trial` and `trialtm` vectors after selecting the requested probes and dropping bad quality labels. Alignment uses `bp.ev.goCue`.

ii.
```python
units = extract_selected_units(obj, spec.probes)
...
trial_ids = np.asarray(unit["trial"], dtype=np.int64).reshape(-1) - 1
trial_times = np.asarray(unit["trialtm"], dtype=np.float64).reshape(-1)
...
aligned_times = trial_times - go_times[trial_ids]
```

iii. The notes in Steps 2, 5, and 10 say the neural mapping is probe-restricted `obj.clu` plus go-cue alignment, matching the raw ephys structure the agent found during exploration.

## 2-b. How is the `neural` data processed?

i. For each unit and trial, spikes are aligned to go cue, binned on a common `[-2.5, 2.5]` grid with `DT = 0.01`, converted to Hz by dividing by `DT`, then smoothed with a custom causal Gaussian kernel. Low-firing units are filtered after computing condition-averaged PSTHs.

ii.
```python
DT = 0.01
SMOOTH_BINS = 15
...
counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
rates = counts.astype(np.float64) / DT
trialdat[unit_idx, tr] = my_smooth_1d(rates, boundary_condition="reflect")
```

```python
def make_causal_kernel(window_bins: int) -> np.ndarray:
    ...
    kernel = gaussian(window_bins, std=std).astype(np.float64)
    kernel[: window_bins // 2] = 0.0
    kernel /= kernel.sum()
```

iii. The AI justifies this in `CONVERSION_NOTES.md` Steps 5 and 10 by pointing to figure-script defaults it believed were closer to the intended decoder path: `alignEvent='goCue'`, `dt=1/100`, and a causal `mySmooth(...,15,'reflect')`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first drops units with quality labels in `{"garbage", "gabrga", "noisy", "real?"}`. It then computes condition-averaged PSTHs across eight condition masks and drops units whose mean PSTH firing rate is not greater than `1 Hz`. Entire sessions with fewer than 10 retained units are discarded.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR = 1.0
MIN_UNITS_PER_SESSION = 10
...
if normalize_string(unit["quality"]).lower() in BAD_QUALITIES:
    continue
```

```python
condition_indices = low_fr_condition_indices(bp, n_trials)
...
mean_fr = np.nanmean(psth, axis=(1, 2))
keep_units = mean_fr > LOW_FR
trialdat = trialdat[keep_units]
if trialdat.shape[0] < MIN_UNITS_PER_SESSION:
    return None
```

iii. In Steps 5 and 10, the AI says this mirrors `findClusters('all')` plus the paper’s `>1 Hz` rule and session inclusion threshold, and the trajectory shows it explicitly choosing the figure-script defaults for low-FR filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go-cue time of the spike’s own trial, then histogramming those aligned times on the common time grid.

ii.
```python
go_times = event_array(ev, ALIGN_EVENT)
...
aligned_times = trial_times - go_times[trial_ids]
counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
```

iii. The notes and trajectory both state that `goCue` is the common alignment event across the decoder task and the figure scripts, including WC trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses `DT = 0.01`, so the converted dataset has 10 ms bins over `[-2.5, 2.5]` for 500 bins total. Spikes are binned directly into this grid; video and motion-energy traces are interpolated or resampled onto the same grid rather than stored at frame rate.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
...
def session_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT)
    time_axis = edges[:-1] + DT / 2.0
    return edges, time_axis
```

iii. The notes say this 10 ms choice was deliberate because the agent read the figure scripts as using `dt=1/100` for single-trial analyses and decided to prioritize that path.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw per-trial variable. It is generated from the session-wide aligned time axis defined relative to `goCue`.

ii.
```python
edges, taxis = session_time_axis()
go_times = event_array(ev, ALIGN_EVENT)
...
input_trials.append(taxis[np.newaxis, :].astype(np.float32, copy=False))
```

iii. In Steps 5 and 10, the AI says the decoder task only required time from go cue, so it used the aligned neural time axis itself as the sole decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The input is the bin-center vector of the common session grid, repeated identically for every kept trial in the session as a `(1, T)` array.

ii.
```python
def session_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT)
    time_axis = edges[:-1] + DT / 2.0
    return edges, time_axis
...
input_trials.append(taxis[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The AI justifies this in the notes as the minimal task-driven input construction: since the decoder input specification only asked for time from go cue, no extra raw behavioral variable was added.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same `taxis` used to bin the spikes. Every input trial stores those same bin centers, and every neural trial is histogrammed on `edges` derived from the same `DT`, `TMIN`, and `TMAX`.

ii.
```python
edges, taxis = session_time_axis()
...
counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
...
input_trials.append(taxis[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The notes explicitly say the saved input is the aligned neural time axis, so no further alignment step is needed.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the raw lick-event arrays `bp.ev.lickL` and `bp.ev.lickR`, together with `bp.ev.goCue`, by identifying the first post-go-cue lick side. It does not derive it from trial side plus outcome.

ii.
```python
lick_l = as_list(gget(ev, "lickL"))
lick_r = as_list(gget(ev, "lickR"))
lick_direction = np.array(
    [post_go_choice(lick_l[tr], lick_r[tr], go_times[tr]) for tr in range(n_trials)],
    dtype=np.int64,
)
```

```python
def post_go_choice(lick_l: Any, lick_r: Any, go_time: float) -> int:
    left = event_list_after(lick_l, go_time)
    right = event_list_after(lick_r, go_time)
    ...
```

iii. The trajectory explicitly says `bp.L`/`bp.R` are trial-side variables and that “actual lick direction should come from the first post-go-cue lick event.”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, it finds the earliest left lick and earliest right lick strictly after the go cue. If neither exists, the class is `2` (`none`). Otherwise it returns `0` for left if the first left lick precedes the first right lick, else `1` for right. That scalar label is then broadcast across all time bins of the trial.

ii.
```python
def post_go_choice(lick_l: Any, lick_r: Any, go_time: float) -> int:
    left = event_list_after(lick_l, go_time)
    right = event_list_after(lick_r, go_time)
    t_left = left.min() if left.size else np.inf
    t_right = right.min() if right.size else np.inf
    if np.isinf(t_left) and np.isinf(t_right):
        return 2
    return 0 if t_left < t_right else 1
```

```python
np.full(taxis.size, lick_direction[tr], dtype=np.int64)
```

iii. The notes and trajectory justify this as using the realized choice rather than the instructed side, especially on miss trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived directly from the per-trial `bp.autowater` flag.

ii.
```python
autowater = bool_array(bp, "autowater", n_trials)
...
context = np.where(autowater, 0, 1).astype(np.int64)
```

iii. In Steps 4, 5, and 10, the AI says `autowater` is the authoritative WC-vs-DR trial label, even outside the dedicated context-only figure subset.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It is a direct relabeling: `autowater=True` becomes `0` (`WC`), and `False` becomes `1` (`DR`). The trial label is repeated across all time bins.

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int64)
...
np.full(taxis.size, context[tr], dtype=np.int64)
```

iii. The notes justify this as the simplest task-consistent mapping and say the reference trial logic already distinguishes `autowater` from non-`autowater` trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes from the per-trial `bp.hit`, `bp.miss`, and `bp.no` flags.

ii.
```python
hit = bool_array(bp, "hit", n_trials)
miss = bool_array(bp, "miss", n_trials)
no = bool_array(bp, "no", n_trials)
```

iii. The notes say the decoder task required an explicit ignore class, so the agent kept `no` trials instead of dropping them as some behavioral analyses do.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as `0=incorrect` on miss trials, `1=correct` on hit trials, and `2=ignore` on `no` trials. The trial label is then repeated across all bins.

ii.
```python
outcome = np.full(n_trials, 2, dtype=np.int64)
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
```

```python
np.full(taxis.size, outcome[tr], dtype=np.int64)
```

iii. In the notes, the AI calls this a task-driven extension of the paper logic so that ignore trials remain available for decoding.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses `obj.traj` from the side camera only (`view_index=0`) and looks for one of the features `tongue`, `left_tongue`, or `right_tongue`. It uses the per-trial `ts` arrays, per-trial `frameTimes`, and the session video offset plus `goCue` to align those trajectories.

ii.
```python
tongue_speed, tongue_visible, tongue_feature = build_speed_trace(
    obj, view_index=0, feature_names=["tongue", "left_tongue", "right_tongue"], go_times=go_times, taxis=taxis
)
```

```python
ts = np.asarray(gget(trial_view, "ts", None), dtype=np.float64)
...
frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]
```

iii. The notes say the converter should “extend the reference kinematic extraction only where the decoder task forces it,” but the trajectory also shows the AI explicitly choosing concrete tongue features from the side view for reproducibility.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates tongue x and y coordinates onto the neural time grid, derives a visibility mask from finite x/y values, sets invisible bins to `NaN`, computes velocity by taking `np.gradient` of the interpolated x/y traces, replaces non-finite tongue gradients with zero, converts to speed magnitude, and then median-splits the session’s visible values.

ii.
```python
x_interp = linear_interp(frame_times, x, taxis)
y_interp = linear_interp(frame_times, y, taxis)
vis_interp = nearest_interp(frame_times, frame_visible.astype(float), taxis) >= 0.5

x_interp[~vis_interp] = np.nan
y_interp[~vis_interp] = np.nan

if feat_is_tongue:
    x_vel = np.gradient(x_interp)
    y_vel = np.gradient(y_interp)
    x_vel[~np.isfinite(x_vel)] = 0.0
    y_vel[~np.isfinite(y_vel)] = 0.0
...
speed[trial_idx] = np.sqrt(x_vel ** 2 + y_vel ** 2).astype(np.float32)
```

iii. The notes justify the median split as required by the decoder task. They also say the code is reusing the reference alignment strategy while preserving a missing-visibility class, though the implementation is a custom interpolation-based path.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After building continuous tongue speed, the AI computes the 50th percentile across visible tongue-speed values from the kept trials in that session. Visible values below threshold become `0`, visible values at or above threshold become `1`, and invisible values become `2`.

ii.
```python
def discretize_session_signal(
    signal: np.ndarray,
    visible: np.ndarray,
    missing_code: int,
    kept_trials: np.ndarray,
) -> tuple[np.ndarray, float]:
    ...
    threshold = float(np.nanpercentile(values, 50))
    visible_and_low = visible & np.isfinite(signal) & (signal < threshold)
    visible_and_high = visible & np.isfinite(signal) & ~visible_and_low
    out[visible_and_low] = 0
    out[visible_and_high] = 1
```

iii. The notes explicitly state that the per-session median split was chosen because the decoder instructions requested a 50th-percentile discretization.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting a session-wise video offset and the per-trial go cue, then the positions are interpolated onto the same `taxis` used for neural binning. Velocity is computed on that aligned neural-time grid.

ii.
```python
vidshift = find_video_offset(obj)
...
frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]
...
x_interp = linear_interp(frame_times, x, taxis)
y_interp = linear_interp(frame_times, y, taxis)
```

iii. The notes and trajectory both say video is synchronized to the neural/behavior clock using the bitcode offset and then aligned to `goCue`; the AI considered this the key reference-consistent part of the video pipeline.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera (`view_index=1`) using `top_paw` as the preferred feature and `bottom_paw` as a fallback, again from `ts`, `frameTimes`, the session video offset, and `goCue`.

ii.
```python
paw_speed, paw_visible, paw_feature = build_speed_trace(
    obj, view_index=1, feature_names=["top_paw", "bottom_paw"], go_times=go_times, taxis=taxis
)
```

iii. In the notes, the AI says it added paw extraction as a decoder-specific extension and chose the bottom-view paw markers because that is where paw tracking is available.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates paw x/y coordinates onto the neural time base, derives visibility from finite coordinates, fills missing positions by nearest-value interpolation, estimates a baseline derivative from the filled trace, subtracts that baseline from `np.gradient` velocities, nearest-fills any non-finite gradients, computes speed magnitude, and then median-splits visible values per session.

ii.
```python
x_interp = linear_interp(frame_times, x, taxis)
y_interp = linear_interp(frame_times, y, taxis)
vis_interp = nearest_interp(frame_times, frame_visible.astype(float), taxis) >= 0.5
...
x_filled = fill_nearest_1d(x_interp)
y_filled = fill_nearest_1d(y_interp)
...
basederiv = np.nanmedian(np.diff(stacked, axis=0), axis=0)
baseline = basederiv[0] if np.isfinite(basederiv[0]) else 0.0
x_vel = np.gradient(x_filled) - baseline
y_vel = np.gradient(y_filled) - baseline
```

iii. The notes justify the median split as task-required and describe this whole path as a reuse of the reference synchronization/interpolation idea for kinematics.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The paw signal is thresholded exactly like tongue velocity: session-wise median over visible values from kept trials, with categories `0=low`, `1=high`, and `2=not visible`.

ii.
```python
paw_disc, paw_thresh = discretize_session_signal(paw_speed, paw_visible, 2, keep_trials)
```

iii. The notes again point to the decoder instructions’ required per-session 50th-percentile threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are corrected by the session video offset and per-trial go cue, then interpolated onto the common neural `taxis`.

ii.
```python
frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]
...
x_interp = linear_interp(frame_times, x, taxis)
y_interp = linear_interp(frame_times, y, taxis)
```

iii. The notes justify this by saying all video-derived signals should share the same go-cue-centered time base as the neural data.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from `motionEnergy_<animal>_<date>.mat` when that file exists; otherwise it falls back to `obj.me`. The per-trial motion-energy vectors are then aligned using side-camera frame times, the session video offset, and `goCue`.

ii.
```python
def load_motion_energy_source(obj: Any, spec: SessionSpec) -> tuple[list[Any] | None, bool]:
    if spec.motion_path.exists():
        me_struct = load_mat_file(spec.motion_path).get("me")
        return motion_energy_data_list(me_struct), True
    me_struct = gget(obj, "me", None)
    if me_struct is None:
        return None, False
    return motion_energy_data_list(me_struct), True
```

iii. The notes say separate motion-energy files are the normal source in the ephys cohorts but that the code handles mixed layouts and fallback locations because the raw dataset contains multiple storage formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates each per-frame motion-energy trace onto the neural time grid, nearest-fills remaining gaps, and then discretizes the aligned continuous values by a per-session median split. It does not recompute motion energy from video frames.

ii.
```python
frame_times = trial_frame_times(side_trials[trial_idx], raw.size) - vidshift - go_times[trial_idx]
interp = linear_interp(frame_times, raw, taxis)
interp = fill_nearest_1d(interp)
aligned[trial_idx] = interp.astype(np.float32)
```

iii. In the notes, the AI says motion energy should reuse the reference alignment source and only add the task-required median discretization.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It uses the same `discretize_session_signal` helper as tongue and paw: per-session 50th percentile over valid motion-energy values from kept trials, with missing/invalid bins assigned code `2`.

ii.
```python
me_disc, me_thresh = discretize_session_signal(motion_energy, motion_valid, 2, keep_trials)
```

iii. The notes explicitly say the median split was chosen because the decoder specification overrode the paper’s manual motion threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video offset and trial go cue, and the motion-energy trace is then interpolated onto the same neural `taxis`.

ii.
```python
side_trials = get_trials_view(obj, 0)
vidshift = find_video_offset(obj)
...
frame_times = trial_frame_times(side_trials[trial_idx], raw.size) - vidshift - go_times[trial_idx]
interp = linear_interp(frame_times, raw, taxis)
```

iii. The notes justify this by saying motion energy should be synchronized exactly like the other video-derived streams and share the common go-cue-centered neural time base.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly keeps sessions and trials and substitutes fallback behavior. If `frameTimes` are missing or all `NaN`, it fabricates times as `1..n_frames / 400`. If a video trial is marked invalid by `NdroppedFrames`, it leaves that trial’s speed or motion-energy arrays as missing so they discretize to class `2`. For paw and motion energy it nearest-fills gaps after interpolation; for tongue it zeros non-finite gradients after interpolation. It also added a `neural_covered` mask so late behavioral-only trials are dropped.

ii.
```python
def trial_frame_times(trial_view: dict[str, Any], n_frames: int) -> np.ndarray:
    frame_times = gget(trial_view, "frameTimes", None)
    if frame_times is None:
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if frame_times.size == 0 or np.isnan(frame_times).all():
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
    return frame_times
```

```python
x_filled = fill_nearest_1d(x_interp)
y_filled = fill_nearest_1d(y_interp)
...
interp = fill_nearest_1d(interp)
...
keep_trials = np.flatnonzero((hit | miss | no) & ~early & ~stim & neural_covered)
```

iii. In Step 10, the AI explicitly justifies `neural_covered` as a bug fix after validator warnings. The notes also frame the rest of the missing-data behavior as preserving trials for decoder use wherever possible.

## 11-a. What are the most time-consuming steps of the code?

i. The AI’s own notes say file loading and repeated nested-structure normalization are the expensive parts. From the code, the other heavy steps are the nested per-unit/per-trial spike histogramming loop and the per-trial interpolation of video/motion-energy streams.

ii.
```python
obj = load_mat_file(spec.data_path)["obj"]
...
for unit_idx, unit in enumerate(units):
    ...
    for tr in np.unique(trial_ids):
        counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
```

```python
for trial_idx in range(n_trials):
    ...
    x_interp = linear_interp(frame_times, x, taxis)
    y_interp = linear_interp(frame_times, y, taxis)
```

iii. `CONVERSION_NOTES.md` Step 6 says “MATLAB-style nested structs/cells produce expensive Python-side branching” and that parsing and normalization were optimized once per session where possible.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several obvious Python loops in place: the per-unit and per-trial spike histogramming loop, per-trial video interpolation loops, per-trial feature-name lookup, and per-trial assembly of `neural_trials`, `input_trials`, and `output_trials`. The code does not use a session-wide `histogram2d` spike counter or vectorized frame binning.

ii.
```python
for unit_idx, unit in enumerate(units):
    ...
    for tr in np.unique(trial_ids):
        tr_mask = trial_ids == tr
        counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
```

```python
for trial_idx in range(n_trials):
    trial_view = trials_view[trial_idx]
    ...
    feat_idx = feature_index(trial_view, chosen_feature)
```

iii. The notes acknowledge some efficiency concerns but justify the overall implementation on practical runtime grounds, saying the sample run was fast enough that further parallelization was unnecessary.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several session-level operations across streams: `find_video_offset(obj)` is recomputed separately for tongue, paw, and motion energy; `get_trials_view(obj, ...)` is rebuilt separately for each stream; feature lookup happens trial-by-trial; and full `taxis` arrays are copied into every trial entry.

ii.
```python
vidshift = find_video_offset(obj)
...
tongue_speed, ... = build_speed_trace(...)
paw_speed, ... = build_speed_trace(...)
motion_energy, ... = build_motion_energy_trace(...)
```

```python
trials_view = get_trials_view(obj, view_index)
...
side_trials = get_trials_view(obj, 0)
```

iii. The AI’s notes claim it normalized layouts “once per session before inner-loop processing,” but the actual implementation still repeats several stream-specific setup steps because each output is built independently.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes continuous tongue speed, paw speed, and motion-energy arrays for every bin even though only the discretized categories are saved. It also computes optional diagnostic plots and histograms that are not part of the converted dataset, and it stores detailed per-session thresholds/feature names that are not needed by the downstream decoder itself.

ii.
```python
tongue_speed, tongue_visible, tongue_feature = build_speed_trace(...)
paw_speed, paw_visible, paw_feature = build_speed_trace(...)
motion_energy, motion_valid = build_motion_energy_trace(...)

tongue_disc, tongue_thresh = discretize_session_signal(...)
paw_disc, paw_thresh = discretize_session_signal(...)
me_disc, me_thresh = discretize_session_signal(...)
```

```python
if show_processing:
    plot_processing(...)
```

iii. The notes justify this as useful for validation and debugging: Step 6 says the script should emit processing diagnostics, and Step 10 documents continuous-trace spot checks against the raw files.
