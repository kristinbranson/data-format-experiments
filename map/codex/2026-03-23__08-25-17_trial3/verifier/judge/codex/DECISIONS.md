# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files directly with `h5py`, not `pynwb`. It first finds session files with a sorted glob over `data/sub-*/*.nwb`, opens each file once in `get_nwb_files()` to prefilter for sessions with at least one `units/classification == "good"` unit, and then opens each retained file again in `process_session()` to read trials, events, units, spikes, and video.

ii.
```python
def get_nwb_files(sample_only: bool) -> list[Path]:
    files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
    valid = []
    for path in files:
        with h5py.File(path, "r") as f:
            good = decode_str_array(f["units/classification"]) == "good"
            if np.any(good):
                valid.append(path)
```

```python
with h5py.File(path, "r") as f:
    trials = f["intervals/trials"]
    go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
    spike_times_flat = f["units/spike_times"][()]
    tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose NWB-native loading via `h5py` for lower overhead and explicit ragged-array access, and that it intentionally excluded the single zero-good-unit session up front so the retained analyzed set matched the 173-session paper count.

## 1-b. How are the data split into subjects?

i. The AI treats the parent folder name of each NWB file, such as `sub-440956`, as the subject identifier. It does not read `nwb.subject.subject_id`. During dataset assembly it creates `subjects` in first-seen session order and fills `subject_idx` from that mapping.

ii.
```python
session_id = path.stem
subject_id = path.parent.name
```

```python
subjects = []
subject_to_idx = {}
...
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The notes say it would “use exact subject IDs (`sub-xxxxx`),” arguing that subject folders already encode the subject grouping and that session order should follow the sorted NWB paths.

## 1-c. How are the data split into sessions?

i. The AI uses one NWB file as one session. Session order follows the sorted file list, and each session id is just `path.stem`.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
session_id = path.stem
...
"session_ids": [r.session_id for r in results],
```

iii. The notes explicitly state that the NWB files are session-level source files and that one raw session corresponds to one session in the converted output.

## 1-d. How are the data split into trials?

i. The AI starts from the NWB trials table and assumes one `go_start_times` timestamp per trial. It checks that `len(go_times_all)` matches the raw trial count, then uses the boolean `valid_trial_mask` to select the retained subset of those trial rows.

ii.
```python
trials = f["intervals/trials"]
n_trials_raw = len(trials["id"])
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
if len(go_times_all) != n_trials_raw:
    raise ValueError(...)
```

```python
start_times = start_times_all[valid_trial_mask]
stop_times = stop_times_all[valid_trial_mask]
go_times = go_times_all[valid_trial_mask]
trial_instruction = decode_str_array(trials["trial_instruction"])[valid_trial_mask]
```

iii. In the notes, the AI says `go_start_times` is the only reliable one-per-trial event stream and should be the unique per-trial anchor because sample and delay event streams contain replay-related extras.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials whose full neural window `[go-2.5 s, go+1.5 s]` falls inside at least one `units/obs_intervals` interval from the first good unit. After spike binning, it further drops any trial that is all-zero across all retained units and bins. It does not explicitly filter `free_water` trials.

ii.
```python
obs_intervals = get_ragged_row(
    f["units/obs_intervals"],
    obs_intervals_index,
    int(good_unit_indices[0]),
)
valid_trial_mask = compute_valid_trial_mask(obs_intervals=obs_intervals, go_times=go_times_all)
```

```python
def compute_valid_trial_mask(obs_intervals: np.ndarray, go_times: np.ndarray) -> np.ndarray:
    trial_start = go_times + REL_START_S
    trial_end = go_times + REL_END_S
    starts = obs_intervals[:, 0][None, :]
    ends = obs_intervals[:, 1][None, :]
    return np.any((trial_start[:, None] >= starts) & (trial_end[:, None] <= ends), axis=1)
```

```python
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
if not np.all(nonzero_trial_mask):
    firing_rates = firing_rates[:, nonzero_trial_mask, :]
    ...
```

iii. The notes justify this as a fix for an initial bug that trusted the full behavioral table and produced many all-zero neural trials; the AI says the `obs_intervals` prefilter and defensive all-zero removal were added to remove out-of-recording trials and “raw session-wide silent gaps.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units/spike_times` for units with `units/classification == "good"`, using `acquisition/BehavioralEvents/go_start_times` to place trial-aligned bin edges.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_unit_indices = np.flatnonzero(classification == "good")
...
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
...
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
```

iii. The notes say this was the closest NWB-native analogue of the paper’s externally generated `goodunits` files while preserving go-cue-centered alignment.

## 2-b. How is the `neural` data processed?

i. For each good unit, the AI bins absolute spike times into 50 ms non-overlapping bins from `-2.5 s` to `+1.5 s` relative to go cue, counts spikes with `np.searchsorted`, and converts counts to firing rates in Hz. It stores the result as `float16`.

ii.
```python
def bin_spikes_to_firing_rates(...):
    flat_edges = trial_edges_abs.reshape(-1)
    ...
    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
        counts = np.diff(edge_idx, axis=1)
        firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
```

iii. The notes say this preserved the reference processing principle of go-cue-centered firing rates while changing only the requested time binning to 50 ms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered by keeping only units whose `units/classification` string equals `"good"`. Sessions with zero such units are excluded before conversion.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    raise ValueError(f"Session {session_id} has no good units")
```

```python
for path in files:
    with h5py.File(path, "r") as f:
        good = decode_str_array(f["units/classification"]) == "good"
        if np.any(good):
            valid.append(path)
```

iii. The notes repeatedly justify this as the paper-consistent QC decision available from the NWB release, even though the total good-unit count differs from the paper by 490 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go cue onset. It creates absolute bin edges by adding the fixed relative edge grid to each trial’s go cue time and bins spikes against those edges directly.

ii.
```python
REL_START_S = -2.5
REL_END_S = 1.5
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
...
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

```python
edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
counts = np.diff(edge_idx, axis=1)
```

iii. The notes say all relevant NWB streams share the same absolute clock, so alignment is done entirely by re-expressing everything relative to go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins over a 4 s go-centered window, giving 80 time bins per trial. No additional temporal rebinning or overlap is applied.

ii.
```python
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
```

iii. The notes explicitly call this a required deviation from the paper’s 40 ms / 3.4 ms preprocessing because the user instructions mandated 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI does not derive this input from any raw tone event variable. Instead it assumes a fixed tone onset at `-1.85 s` relative to go cue from the task structure and combines that constant with the shared relative bin centers.

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
...
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The notes justify this by saying the raw `sample_start_times` stream contains replay-related extra events and that a canonical task-defined tone onset is “more consistent with the task definition.”

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes a single 80-bin vector equal to `REL_CENTERS + 1.85` seconds and tiles that identical vector across every trial. There is no per-trial lookup of tone onset.

ii.
```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The notes say the task structure is enough to define sample onset and avoids ambiguity from replayed sample events.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI aligns this input by using the exact same go-cue-centered 50 ms bin centers as the neural data. The input is just a deterministic function of `REL_CENTERS`.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The notes emphasize that both decoder inputs are meant to be fully time-varying arrays defined on the same 80-bin grid as the neural activity.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trials-table strings `photostim_onset` and `photostim_duration`, together with `start_time` and the trial’s `go_time` to convert into go-relative coordinates.

ii.
```python
photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
...
input_stim = build_photostim_matrix(
    photostim_onset_str=photostim_onset_str,
    photostim_duration_str=photostim_duration_str,
    start_times=start_times,
    go_times=go_times,
)
```

iii. The notes say this matches the NWB representation, where photostim timing is stored relative to trial start and must be shifted into the go-centered frame used by the decoder.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI parses `'N/A'` onset and duration strings into `NaN`, converts valid onset and duration values to seconds relative to go cue, and then marks each 50 ms bin center as 1 if it lies in `[stim_on, stim_off)`, else 0.

ii.
```python
def parse_optional_float_array(strings: np.ndarray) -> np.ndarray:
    out = np.full(strings.shape, np.nan, dtype=np.float64)
    for i, value in enumerate(strings):
        if value == "N/A":
            continue
        out[i] = float(value)
```

```python
onset_trial = parse_optional_float_array(photostim_onset_str)
duration = parse_optional_float_array(photostim_duration_str)
go_minus_start = go_times - start_times
onset_rel_go = onset_trial - go_minus_start
offset_rel_go = onset_rel_go + duration
...
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
stim[trial_idx, mask] = 1.0
```

iii. The notes say it kept stimulation trials because photostimulation is an explicit decoder input, and it chose a binary time-varying representation on the shared bin grid.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation by converting onset and offset into go-relative seconds and comparing them directly to the same `REL_CENTERS` used for the neural bins.

ii.
```python
go_minus_start = go_times - start_times
onset_rel_go = onset_trial - go_minus_start
offset_rel_go = onset_rel_go + duration
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
```

iii. The notes explicitly describe this as converting trial-start coordinates into go-centered coordinates so neural and stimulation time series live on one axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `trial_instruction`, `outcome`, and also global left/right lick event timestamps. For hit trials it uses the instructed side; for miss trials the opposite side; for ignore trials it tries to infer a side from actual lick events and otherwise falls back to the instructed side.

ii.
```python
left_lick_times = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()].astype(np.float64)
right_lick_times = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()].astype(np.float64)
choice_code = build_choice_array(
    trial_instruction=trial_instruction,
    outcome_code=outcome_code,
    start_times=start_times,
    go_times=go_times,
    stop_times=stop_times,
    left_lick_times=left_lick_times,
    right_lick_times=right_lick_times,
)
```

```python
choice[hit_mask] = instructed[hit_mask]
choice[miss_mask] = 1 - instructed[miss_mask]
for trial_idx in ignore_trials:
    choice[trial_idx] = lick_choice_with_fallback(...)
```

iii. The notes justify this as an edge-case policy for ignore trials: because most ignore trials have no post-go lick, choice should use a documented fallback, preferably lick-derived and otherwise instructed side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as only two classes, left `0` and right `1`, with no separate no-lick class. It repeats the resulting per-trial label across all 80 time bins.

ii.
```python
choice = np.zeros(len(trial_instruction), dtype=np.int16)
instructed = np.where(trial_instruction == "left", 0, 1).astype(np.int16)
...
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_code[trial_idx], dtype=np.int16),
        ...
    ]
)
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "40_to_60pct", "gt_60pct"],
],
```

iii. The notes frame the ignore-trial fallback as necessary because the target output should still be a left/right choice label even when the animal did not lick in the response window.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trials-table `outcome` column.

ii.
```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
```

iii. The notes treat this as a direct mapping because the raw NWB table already stores the same three categories requested by the decoder task.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats that per-trial code across all 80 bins.

ii.
```python
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

```python
np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16)
```

iii. The notes say this followed the task instructions exactly and used time-repeated labels so all outputs could share one `(n_output, n_timepoints)` shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trials-table `early_lick` column.

ii.
```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
```

iii. The notes describe this as a direct per-trial flag already present in the NWB trial metadata.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats that label across all 80 bins.

ii.
```python
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
...
np.full(N_BINS, early_code[trial_idx], dtype=np.int16)
```

iii. The notes say this matched the requested output coding and the reference code’s use of early-lick trial metadata.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position comes from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using its `data[:, 1]` y-coordinate, `data[:, 2]` tracking likelihood, `data[:, 0]` x-coordinate for velocity-based cleaning, and the accompanying `timestamps`.

ii.
```python
tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. The notes say it used the side-camera tongue tracking series because that is the relevant video stream described in the methods.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first cleans the full-session tongue track with a five-sigma frame-to-frame velocity rule, linearly interpolates over outlier frames, and replaces all low-likelihood frames with the session mean visible y-position. It then aligns a single carried-forward y value to each bin center for each trial.

ii.
```python
speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
...
y[outlier_mask] = np.interp(...)
```

```python
visible_mask = np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
mean_y = float(np.nanmean(y[visible_mask])) if np.any(visible_mask) else float(np.nanmean(y))
occluded_mask = ~visible_mask
y[occluded_mask] = mean_y
```

```python
abs_centers = go_times[:, None] + REL_CENTERS[None, :]
idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
return cleaned_y[idx]
```

iii. The notes justify this as “method-paper-inspired” preprocessing: use a five-sigma velocity rule, impute occluded tongue positions to mean visible y, and align by last-frame-carried-forward rather than interpolation between bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After alignment, the AI takes the 40th and 60th percentiles over all retained aligned tongue-y samples in the session, then assigns classes 0 for `< q40`, 1 for `q40..q60`, and 2 for `> q60`. It does not keep a separate “not visible” category.

ii.
```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

```python
"output_values": [
    ...,
    ["lt_40pct", "40_to_60pct", "gt_60pct"],
],
```

iii. The notes say it intentionally replaced an earlier `np.digitize` implementation with explicit threshold comparisons to avoid ties collapsing the middle class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue y-position to the neural data by evaluating the cleaned tongue track at each neural bin center. It converts every trial’s relative bin centers to absolute timestamps using the go cue and uses last-frame-carried-forward via `np.searchsorted(..., side="right") - 1`.

ii.
```python
def align_tongue_y(timestamps: np.ndarray, cleaned_y: np.ndarray, go_times: np.ndarray) -> np.ndarray:
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

iii. The notes say this was chosen to mirror the reference code’s marker alignment style more closely than linear interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses a mix of exclusion and imputation. Sessions with no good units are excluded; trials outside `obs_intervals` or with all-zero neural matrices after binning are excluded; missing or low-likelihood tongue frames are imputed rather than marked missing; optional photostim strings use `NaN` for `'N/A'`. Text fields are decoded by converting any non-bytes entry to `str(x)`.

ii.
```python
def decode_str_array(ds: h5py.Dataset) -> np.ndarray:
    ...
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        elif isinstance(x, np.bytes_):
            out.append(x.astype(str))
        else:
            out.append(str(x))
```

```python
if len(good_unit_indices) == 0:
    raise ValueError(...)
...
valid_trial_mask = compute_valid_trial_mask(...)
...
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
```

```python
occluded_mask = ~visible_mask
y[occluded_mask] = mean_y
```

iii. The notes say this behavior came from trying to eliminate obvious neural coverage artifacts while keeping decoder-required trials and making tongue outputs dense enough to validate and train cleanly.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are repeated NWB file I/O, loading full spike and tongue arrays, and the per-unit spike binning loop. The code also does an initial file scan that opens every session before the main conversion pass.

ii.
```python
for path in files:
    with h5py.File(path, "r") as f:
        good = decode_str_array(f["units/classification"]) == "good"
```

```python
tongue_data = tongue_ts["data"][()]
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
```

```python
for i, unit_idx in enumerate(good_unit_indices):
    spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
    edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
```

iii. The notes explicitly call out NWB loading, spike binning, and large array handling as the runtime-dominant work; later full-run notes report a 7.24 minute conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several small-to-medium loops could have been vectorized further: `parse_optional_float_array`, the ignore-trial loop in `build_choice_array`, the per-trial photostim loop in `build_photostim_matrix`, and the per-trial assembly loop that builds `input_trials` and `output_trials`. The per-unit spike loop is harder to remove because spike trains are ragged.

ii.
```python
for i, value in enumerate(strings):
    if value == "N/A":
        continue
    out[i] = float(value)
```

```python
for trial_idx in ignore_trials:
    choice[trial_idx] = lick_choice_with_fallback(...)
```

```python
for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
    stim[trial_idx, mask] = 1.0
```

iii. The notes say the heavy operations were already vectorized in NumPy, but these residual loops remain in the implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several steps. It opens every NWB file once in `get_nwb_files()` and again in `process_session()`. It also reparses photostim onset strings after trial assembly just to build diagnostics, recomputing `go_minus_start` and `onset_rel_go`. Trial-level arrays are materialized once in batch form and then rewrapped again as Python lists of per-trial matrices.

ii.
```python
def get_nwb_files(sample_only: bool) -> list[Path]:
    ...
    for path in files:
        with h5py.File(path, "r") as f:
            ...
```

```python
with h5py.File(path, "r") as f:
    ...
```

```python
onset_trial = parse_optional_float_array(photostim_onset_str)
go_minus_start = go_times - start_times
onset_rel_go = onset_trial - go_minus_start
photostim_onsets_rel_go = onset_rel_go[np.isfinite(onset_rel_go)]
```

iii. The notes mainly emphasize avoiding repeated heavy computation, but the final code still duplicates this file scan and some per-session bookkeeping work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some data only for diagnostics or optional plots rather than the final converted dataset. Examples include `SessionResult.region_names`, the preview slices of raw and cleaned tongue traces, photostim-onset histograms, and per-session plotting support. These are not part of the saved pickle that downstream decoding uses.

ii.
```python
return SessionResult(
    ...
    region_names=sorted(set(region_labels.tolist())),
    ...
    diagnostics=diagnostics,
)
```

```python
diagnostics = {
    "tracking_preview_time": tongue_timestamps[:preview_n] - go_times[0],
    "tracking_preview_raw": tongue_y[:preview_n],
    "tracking_preview_clean": cleaned_tongue_y[:preview_n],
    "aligned_tongue_y_trial": aligned_tongue_y[0],
    "photostim_onsets_rel_go": photostim_onsets_rel_go,
    ...
}
```

```python
if args.show_processing and len(results) <= 2:
    plot_name = f"processing_{result.session_id}.png"
    make_session_plot(result, Path(plot_name))
```

iii. The notes say these were added for sanity checking, speed investigation, and manual visualization rather than for the exported decoder dataset itself.
