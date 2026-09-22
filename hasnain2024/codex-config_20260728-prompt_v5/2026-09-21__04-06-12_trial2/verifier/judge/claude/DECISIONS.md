# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the MATLAB loader scripts (`Figure3h.m` and `load<ANM>_ALMVideo.m`) to dynamically discover sessions and their probe assignments, then loads each session's `.mat` file using either `mat73` (for v7.3/HDF5) or `scipy.io.loadmat` (for older format). Motion energy is loaded from separate `motionEnergy_*.mat` files.

ii.
```python
def load_mat_file(path: Path) -> dict[str, Any]:
    try:
        return mat73.loadmat(str(path))
    except Exception:
        return scipy.io.loadmat(str(path), simplify_cells=True)

def parse_loader_animals(script_path: Path, marker: str) -> list[str]:
    text = script_path.read_text()
    animals: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("%"):
            continue
        match = re.search(rf"{re.escape(marker)}\s*=\s*load([A-Za-z0-9]+)_ALMVideo\(", line)
        if match:
            animals.append(match.group(1))
    return animals
```

iii. The AI chose to programmatically parse the MATLAB loader scripts rather than hard-coding sessions. The CONVERSION_NOTES state: "The released paper code defines the paper-comparable cohorts explicitly through the `load*_ALMVideo.m` scripts."

## 1-b. How are the data split into subjects?

i. The subject (animal) is extracted from the session specification's `animal` field, which is parsed from the loader script filenames (e.g., `loadJEB6_ALMVideo.m` yields `JEB6`). Subjects are assigned indices in the order they are encountered.

ii.
```python
subject_to_idx: dict[str, int] = {}
# ...
subject_idx = subject_to_idx.setdefault(spec.animal, len(subject_to_idx))
# ...
converted["subjects"] = [subject for subject, _ in sorted(subject_to_idx.items(), key=lambda kv: kv[1])]
```

iii. The AI uses the animal name from the loader script parsing, which matches the animal ID in the filenames.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` object represents one session, defined by animal, date, cohort, folder, and probes. The AI parses `load<ANM>_ALMVideo.m` to extract date+probe pairs, yielding 44 sessions (25 fixed-delay, 19 randomized-delay). Each session becomes one element of the output lists.

ii.
```python
def parse_loader_sessions(animal: str, cohort: str, folder: str) -> list[SessionSpec]:
    text = (LOADER_DIR / f"load{animal}_ALMVideo.m").read_text()
    sessions: list[SessionSpec] = []
    current_date: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("%"):
            continue
        m_date = re.search(r"date = '([^']+)'", line)
        if m_date:
            current_date = m_date.group(1)
        m_probe = re.search(r"probe = (\[[^\]]+\]|\d+);", line)
        if m_probe and current_date:
            probes = tuple(int(x) for x in re.findall(r"\d+", m_probe.group(1)))
            sessions.append(SessionSpec(...))
            current_date = None
    return sessions
```

iii. From CONVERSION_NOTES: "The conversion will therefore use the union of the fixed-delay and randomized-delay ephys loader sessions (44 sessions total before downstream filtering)."

## 1-d. How are the data split into trials?

i. Trials are defined by the `Ntrials` field in `obj.bp`. Each trial has associated behavioral flags, event times, and neural/video data indexed by trial number. The AI reads `Ntrials` and uses trial indices 0 through `n_trials-1`.

ii.
```python
n_trials = int(gget(bp, "Ntrials", 0) or 0)
# ...
keep_trials = np.flatnonzero((hit | miss | no) & ~early & ~stim & neural_covered)
```

iii. The AI uses the standard trial table structure from `obj.bp`, consistent with the reference code.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three criteria: (1) must be hit, miss, or no-response (ignore) — not other categories, (2) must not be early-lick trials, (3) must not be photostimulation trials, and (4) must have neural coverage (at least one spike from any unit). Sessions must have at least 2 valid trials and at least 10 units.

ii.
```python
hit = bool_array(bp, "hit", n_trials)
miss = bool_array(bp, "miss", n_trials)
no = bool_array(bp, "no", n_trials)
early = bool_array(bp, "early", n_trials)
stim = stim_enable_array(bp, n_trials)
keep_trials = np.flatnonzero((hit | miss | no) & ~early & ~stim & neural_covered)
if keep_trials.size < MIN_TRIALS_PER_SESSION:
    return None
```

iii. From CONVERSION_NOTES: "Exclude `stim.enable` and `early` trials, but keep hit/miss/no trials...Ignore/no-response trials are retained because the decoder task requires the `ignore` outcome class."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu` (spike-sorted clusters), specifically `trial` (spike trial IDs), `trialtm` (spike times relative to trial start), and `quality` (curation label). The go cue times `bp.ev.goCue` provide the alignment event.

ii.
```python
units = extract_selected_units(obj, spec.probes)
# For each unit:
trial_ids = np.asarray(unit["trial"], dtype=np.int64).reshape(-1) - 1
trial_times = np.asarray(unit["trialtm"], dtype=np.float64).reshape(-1)
aligned_times = trial_times - go_times[trial_ids]
```

iii. The AI states this follows the reference pipeline: "load `obj` -> select clusters by quality -> align spike times to an event -> bin and smooth spikes."

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned in 10 ms bins over [-2.5, 2.5] s (500 bins), converted to firing rates (Hz), and smoothed with a **causal** Gaussian kernel of window length 15 bins. The causal kernel zeros out the first half of the Gaussian window.

ii.
```python
DT = 0.01
SMOOTH_BINS = 15

def make_causal_kernel(window_bins: int) -> np.ndarray:
    std = (window_bins - 1) / (2.0 * 2.5)
    kernel = gaussian(window_bins, std=std).astype(np.float64)
    kernel[: window_bins // 2] = 0.0
    kernel /= kernel.sum()
    return kernel

# Per unit, per trial:
counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
rates = counts.astype(np.float64) / DT
trialdat[unit_idx, tr] = my_smooth_1d(rates, boundary_condition="reflect")
```

iii. The CONVERSION_NOTES state the AI follows the "figure scripts (`Figure3h.m`, `Figure8a_thru_c.m`) more closely than the generic defaults." The AI uses causal smoothing described as matching `mySmooth(...,15,'reflect')`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) quality label filtering excludes clusters labeled `garbage`, `gabrga`, `noisy`, or `real?`; (2) firing rate filtering removes units whose mean condition-averaged PSTH firing rate is below 1 Hz. Sessions with fewer than 10 units after filtering are skipped.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR = 1.0
MIN_UNITS_PER_SESSION = 10

# Quality filter:
if normalize_string(unit["quality"]).lower() in BAD_QUALITIES:
    continue

# FR filter using condition-averaged PSTHs:
condition_indices = low_fr_condition_indices(bp, n_trials)
psth = np.zeros((trialdat.shape[0], taxis.size, len(condition_indices)), dtype=np.float32)
for cond_idx, trial_idx in enumerate(condition_indices):
    if trial_idx.size == 0:
        continue
    psth[:, :, cond_idx] = np.nanmean(trialdat[:, trial_idx, :], axis=1)
mean_fr = np.nanmean(psth, axis=(1, 2))
keep_units = mean_fr > LOW_FR
```

iii. The AI does not include `poor` in the quality drop list, stating it follows `findClusters('all')`. The FR filter uses condition-averaged PSTHs rather than overall mean rate, which attempts to mirror the reference `removeLowFRClusters.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's time relative to trial start (`trialtm`) is shifted by subtracting the go cue time for that trial (`go_times[trial_id]`), giving spike times in seconds from go cue onset.

ii.
```python
go_times = event_array(ev, ALIGN_EVENT)  # ALIGN_EVENT = "goCue"
aligned_times = trial_times - go_times[trial_ids]
```

iii. This matches the reference `alignSpikes.m` approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (`DT = 0.01`), producing 500 time bins over the [-2.5, 2.5] s window. No rebinning is applied after the initial binning.

ii.
```python
DT = 0.01
TMIN = -2.5
TMAX = 2.5

def session_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT)
    time_axis = edges[:-1] + DT / 2.0
    return edges, time_axis
```

iii. The CONVERSION_NOTES state this choice follows "the released figure-script configuration more closely than the generic 5 ms / 200 Hz defaults." The AI interpreted `getDefaultParams.m` `dt=1/200` as 10ms (1/100) based on its reading of the figure scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself, constructed from `TMIN`, `TMAX`, and `DT`. It is the bin centers of the neural binning grid.

ii.
```python
edges, taxis = session_time_axis()
# ...
input_trials.append(taxis[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The time axis is defined by the conversion parameters, not derived from a raw data variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as the bin centers of an evenly-spaced grid from -2.5 to 2.5 s with step DT=0.01 s.

ii.
```python
edges = np.arange(TMIN, TMAX + DT * 0.5, DT)
time_axis = edges[:-1] + DT / 2.0
```

iii. No processing of raw data is involved.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is the same grid used for binning spikes, so alignment is by construction.

ii.
```python
counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
# Same edges used for both neural binning and time axis
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp.ev.lickL` and `bp.ev.lickR` (actual lick event times) and `bp.ev.goCue` (go cue time). The first post-go-cue lick determines direction.

ii.
```python
lick_l = as_list(gget(ev, "lickL"))
lick_r = as_list(gget(ev, "lickR"))
lick_direction = np.array(
    [post_go_choice(lick_l[tr], lick_r[tr], go_times[tr]) for tr in range(n_trials)],
    dtype=np.int64,
)
```

iii. The CONVERSION_NOTES state: "Use trial-event licks for lick direction instead of `bp.L` / `bp.R`: `bp.L` and `bp.R` encode the trial side, not the animal's realized first post-go-cue lick."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the function finds the earliest lick event after the go cue from both `lickL` and `lickR`. Whichever side has the earlier lick determines direction: left=0, right=1, none=2 (if neither side has a post-go-cue lick).

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

iii. The AI chose actual lick events rather than instructed side + outcome to determine lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii.
```python
autowater = bool_array(bp, "autowater", n_trials)
context = np.where(autowater, 0, 1).astype(np.int64)
```

iii. The CONVERSION_NOTES state: "Context is carried by `autowater`."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater trials become WC (0), non-autowater trials become DR (1).

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int64)
```

iii. Straightforward relabeling matching the instruction's WC/DR encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = bool_array(bp, "hit", n_trials)
miss = bool_array(bp, "miss", n_trials)
no = bool_array(bp, "no", n_trials)
outcome = np.full(n_trials, 2, dtype=np.int64)
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
```

iii. Three-class encoding: incorrect=0 (miss), correct=1 (hit), ignore=2 (no response).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping from the three mutually exclusive behavioral flags to the three output classes.

ii.
```python
outcome = np.full(n_trials, 2, dtype=np.int64)
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
```

iii. This matches the instruction specification exactly.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side camera's DLC tracking (`obj.traj[0]`), specifically the `tongue` feature (with fallbacks to `left_tongue` and `right_tongue`). Frame times (`frameTimes`) and the video offset from `sglx.bitcode.bitstart` / `sglx.fs` / `bp.ev.bitStart` are used for alignment.

ii.
```python
tongue_speed, tongue_visible, tongue_feature = build_speed_trace(
    obj, view_index=0, feature_names=["tongue", "left_tongue", "right_tongue"],
    go_times=go_times, taxis=taxis
)
```

iii. The AI uses only the side camera (view_index=0) for tongue tracking, unlike the reference which uses both side and bottom cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial: (1) extract x,y coordinates from the tracked feature, (2) get frame times corrected by video offset and go cue, (3) filter by visibility (finite x,y), (4) interpolate x,y to the neural time axis using linear interpolation, (5) mark interpolated visibility using nearest-neighbor interpolation, (6) compute velocity as gradient of interpolated positions, (7) compute speed as sqrt(dx^2 + dy^2). For tongue specifically, non-finite velocity values are set to 0. Then discretize per session at 50th percentile: <threshold=0, >=threshold=1, not visible=2.

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

speed[trial_idx] = np.sqrt(x_vel ** 2 + y_vel ** 2).astype(np.float32)
```

iii. The AI's approach interpolates positions to the neural time axis before computing velocity, rather than computing velocity at frame resolution and then binning. This is a different processing order than the reference.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Within each session, the 50th percentile of visible tongue speed values (from kept trials only) is computed. Values below the threshold get 0, at or above get 1, and not-visible bins get 2.

ii.
```python
def discretize_session_signal(signal, visible, missing_code, kept_trials):
    out = np.full(signal.shape, missing_code, dtype=np.int64)
    keep_signal = signal[kept_trials]
    keep_visible = visible[kept_trials]
    values = keep_signal[keep_visible]
    values = values[np.isfinite(values)]
    threshold = float(np.nanpercentile(values, 50))
    visible_and_low = visible & np.isfinite(signal) & (signal < threshold)
    visible_and_high = visible & np.isfinite(signal) & ~visible_and_low
    out[visible_and_low] = 0
    out[visible_and_high] = 1
    return out, threshold
```

iii. Matches the instruction specification for per-session 50th percentile thresholding.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (computed from bitcode synchronization) and the go cue time, then tongue positions are linearly interpolated onto the neural time axis.

ii.
```python
vidshift = find_video_offset(obj)
frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]
x_interp = linear_interp(frame_times, x, taxis)
```

iii. The AI uses the same video offset computation as the reference (`findVideoOffset.m`) but interpolates to the neural time axis rather than binning frame-level values.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera's DLC tracking (`obj.traj[1]`), specifically the `top_paw` feature (with fallback to `bottom_paw`).

ii.
```python
paw_speed, paw_visible, paw_feature = build_speed_trace(
    obj, view_index=1, feature_names=["top_paw", "bottom_paw"],
    go_times=go_times, taxis=taxis
)
```

iii. Matches the reference's use of `top_paw` from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Similar to tongue but with different missing-data handling: (1) interpolate x,y to neural time axis, (2) for non-tongue features, nearest-fill missing values, (3) compute baseline drift correction by subtracting the median derivative, (4) compute velocity as gradient of filled positions, (5) compute speed magnitude.

ii.
```python
# For non-tongue (paw):
x_filled = fill_nearest_1d(x_interp)
y_filled = fill_nearest_1d(y_interp)
stacked = np.column_stack([x_filled, y_filled])
basederiv = np.nanmedian(np.diff(stacked, axis=0), axis=0)
baseline = basederiv[0] if np.isfinite(basederiv[0]) else 0.0
x_vel = np.gradient(x_filled) - baseline
y_vel = np.gradient(y_filled) - baseline
```

iii. The AI applies a baseline drift correction for paw velocity that is not present in the reference solution.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile threshold on visible values from kept trials. Below=0, above=1, not visible=2.

ii.
```python
paw_disc, paw_thresh = discretize_session_signal(paw_speed, paw_visible, 2, keep_trials)
```

iii. Matches instruction specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then positions linearly interpolated to the neural time axis.

ii.
```python
frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]
x_interp = linear_interp(frame_times, x, taxis)
```

iii. Same approach as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_*.mat` files (or from `obj.me` as fallback). Frame times come from the side camera (`obj.traj[0]`).

ii.
```python
def load_motion_energy_source(obj, spec):
    if spec.motion_path.exists():
        me_struct = load_mat_file(spec.motion_path).get("me")
        return motion_energy_data_list(me_struct), True
    me_struct = gget(obj, "me", None)
    if me_struct is None:
        return None, False
    return motion_energy_data_list(me_struct), True
```

iii. The AI tries the standalone file first, falling back to `obj.me`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Raw motion energy values (one per frame) are linearly interpolated to the neural time axis, then nearest-filled for any remaining gaps.

ii.
```python
interp = linear_interp(frame_times, raw, taxis)
interp = fill_nearest_1d(interp)
aligned[trial_idx] = interp.astype(np.float32)
```

iii. The AI uses interpolation rather than bin-averaging as the reference does.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw: per-session 50th percentile threshold on valid values from kept trials. Below=0, above=1, no video=2.

ii.
```python
me_disc, me_thresh = discretize_session_signal(motion_energy, motion_valid, 2, keep_trials)
```

iii. Matches instruction specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by video offset and go cue, then motion energy is linearly interpolated to the neural time axis.

ii.
```python
frame_times = trial_frame_times(side_trials[trial_idx], raw.size) - vidshift - go_times[trial_idx]
interp = linear_interp(frame_times, raw, taxis)
```

iii. Same video offset and alignment approach as the other camera-derived signals.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Trials where video is invalid (`NdroppedFrames` is all NaN) are skipped. (2) Trials where frameTimes are missing or all NaN get synthetic frame times at 400 Hz. (3) For tongue, non-finite velocity values are set to 0. (4) For paw, positions are nearest-filled before velocity computation. (5) Motion energy is nearest-filled after interpolation. (6) Trials past the end of neural recording are excluded via `neural_covered`. (7) Sessions with too few units (<10) or trials (<2) are skipped.

ii.
```python
def trial_frame_times(trial_view, n_frames):
    frame_times = gget(trial_view, "frameTimes", None)
    if frame_times is None:
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
    # ...
    if frame_times.size == 0 or np.isnan(frame_times).all():
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0

# For tongue:
x_vel[~np.isfinite(x_vel)] = 0.0

# For paw:
x_filled = fill_nearest_1d(x_interp)
```

iii. The AI uses nearest-fill interpolation for missing values in non-tongue features, which matches the reference MATLAB code's behavior for non-tongue features. The tongue preserves NaN/visibility masks.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the MATLAB files. The full conversion takes about 213 seconds for 44 sessions (~5 s/session). The per-trial neural histogram loop is also potentially slow since it iterates over individual trials within each unit.

ii.
```python
obj = load_mat_file(spec.data_path)["obj"]
# ...
for tr in np.unique(trial_ids):
    tr_mask = trial_ids == tr
    counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
```

iii. The CONVERSION_NOTES estimate ~5 s/session, and total conversion is under the 15-minute threshold.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike histogramming loop inside the unit loop could be vectorized using `np.histogram2d` (as the reference does). The per-trial video processing loop also iterates trial by trial through feature extraction and interpolation.

ii.
```python
# Current per-trial loop:
for tr in np.unique(trial_ids):
    tr_mask = trial_ids == tr
    counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
    rates = counts.astype(np.float64) / DT
    trialdat[unit_idx, tr] = my_smooth_1d(rates, boundary_condition="reflect")
```

iii. The reference solution uses a single `np.histogram2d` call per unit to bin all trials at once, avoiding the inner trial loop.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`find_video_offset`) is computed once per session in the main processing flow, but it is called separately for tongue/paw (`build_speed_trace`) and motion energy (`build_motion_energy_trace`), leading to redundant computation. The trial view parsing (`get_trials_view`) is also called multiple times for the same view.

ii.
```python
# Called in build_speed_trace for tongue:
vidshift = find_video_offset(obj)
# Called again in build_speed_trace for paw:
vidshift = find_video_offset(obj)
# Called again in build_motion_energy_trace:
vidshift = find_video_offset(obj)
```

iii. The video offset is a session constant but is recomputed 3 times per session.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `low_fr_condition_indices` function computes elaborate condition-specific trial masks (8 different conditions) for the firing rate filter, which is more complex than needed — the reference simply checks mean rate across all trials. The code also loads the full MATLAB file structure including unused fields. The `fill_nearest_1d` interpolation fills gaps that are then masked as "not visible" anyway for the tongue case.

ii.
```python
def low_fr_condition_indices(bp, n_trials):
    masks = [
        hit | miss | no,
        right & hit & ~stim & ~autowater & ~early,
        left & hit & ~stim & ~autowater & ~early,
        # ... 8 total conditions
    ]
    return [np.flatnonzero(mask) for mask in masks]
```

iii. The condition-averaged PSTH approach mirrors the reference's `removeLowFRClusters.m` but goes beyond what's strictly needed for a simple firing rate threshold.
