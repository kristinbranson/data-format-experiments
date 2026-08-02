# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads raw NWB files directly from `data/sub-*/*.nwb` with `h5py`, not the paper code's pre-exported `.mat` session files. It scans all subject folders, keeps only NWB files that contain at least one unit with `classification == "good"`, then opens each NWB file and reads trial tables, behavioral events, behavioral time series, and unit tables from HDF5 paths.

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
    classification = decode_str_array(f["units/classification"])
    go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
    tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
    spike_times_flat = f["units/spike_times"][()]
```

iii. In `CONVERSION_NOTES.md`, the agent says it chose NWB loading because the local dataset is a DANDI-style NWB release and because `h5py` gave lower overhead and explicit control over ragged arrays. It explicitly notes that this differs from the reference code's `.mat` loading path.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the parent directory name of each NWB file, e.g. `sub-440956`. During dataset assembly, the agent builds a unique `subjects` list and a `subject_idx` array mapping each processed session to its subject.

ii.
```python
session_id = path.stem
subject_id = path.parent.name
```

```python
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The notes say the NWB layout is one folder per subject and one NWB file per session, so using the folder name was the agent's direct mapping for mouse identity.

## 1-c. How are the data split into sessions?

i. Each included NWB file is treated as one session. Session order follows the sorted file list returned by `get_nwb_files`, and `path.stem` becomes the session identifier.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
for i, path in enumerate(files, start=1):
    result = process_session(path)
```

```python
session_id = path.stem
```

iii. The notes state that the local NWB release stores one session per NWB file, so the agent used that as the session boundary instead of reconstructing multi-probe sessions from multiple `.mat` files as in the reference code.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table, with one row per behavioral trial. The agent also requires `go_start_times` to have the same count as the trial table and then uses the per-trial arrays after masking invalid trials.

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
n_trials = len(go_times)
```

iii. The trajectory shows the agent inspected the NWB schema and concluded that `go_start_times` was the only reliable one-per-trial alignment stream because sample and delay streams can include replayed epochs.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not use the paper's regular-trial mask. Instead it keeps trials that have full neural coverage for the requested `[-2.5, 1.5)` window according to one good unit's `obs_intervals`, then drops any remaining trial whose binned activity is all zero across all good units and bins.

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
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
if not np.all(nonzero_trial_mask):
    firing_rates = firing_rates[:, nonzero_trial_mask, :]
    ...
```

iii. The notes say this was added after `train_decoder.py --verify-only` found zero-neural trials. The agent documented that it intentionally retained stimulation, early-lick, ignore, and miss trials because those variables were required decoder inputs or outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `units/spike_times` and `units/spike_times_index` for spikes, `units/classification` for the good-unit mask, and `acquisition/BehavioralEvents/go_start_times/timestamps` for trial alignment. `units/anno_name` is used for region labels, not for firing rate calculation.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
...
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. In the notes, the agent maps the NWB spike tables to the reference code's `neuron_single_units` representation and says `classification == "good"` is the closest NWB-native analogue of the external QC files used by the paper code.

## 2-b. How is the `neural` data processed?

i. The agent bins absolute spike times into 50 ms bins from 2.5 s before to 1.5 s after go cue, counts spikes with `np.searchsorted`, and divides by `BIN_SIZE_S` to store firing rates in Hz as `float16`.

ii.
```python
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
```

```python
edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
counts = np.diff(edge_idx, axis=1)
firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
```

iii. The notes say the agent intentionally changed from the paper's 40 ms / 3.4 ms preprocessing to the user-requested 50 ms bins while preserving go-cue-centered alignment and firing-rate conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered only by `units/classification == "good"`. Sessions with zero such units are excluded. The code does not reproduce the reference pipeline's external `goodunits` files or histology intersection.

ii.
```python
good = decode_str_array(f["units/classification"]) == "good"
if np.any(good):
    valid.append(path)
```

```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    raise ValueError(...)
```

iii. The notes explicitly justify this as the closest directly available NWB-native QC signal and document the residual mismatch with the paper's reported good-unit count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go cue onset. For each retained trial, the code adds the relative bin edges `[-2.5, 1.5]` to that trial's absolute go-cue time and bins spikes in that per-trial go-centered window.

ii.
```python
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
...
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. Both the notes and trajectory say the agent chose go cue as the unique per-trial temporal anchor because the instruction required it and the reference code and papers are also go-cue-centered.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, with 80 bins spanning 4 s total. No rebinning from a finer reference grid is applied; the code bins directly from raw spike times into the target bin size.

ii.
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))
```

```python
firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
```

iii. The notes say this was a required deviation from the paper code because the decoder task explicitly requested 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is not derived from a raw per-trial sample-event variable. The agent derives it from the fixed task structure using the relative go-centered bin centers and a hard-coded canonical tone onset `-1.85 s` from go cue.

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
```

```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The trajectory shows the agent rejected the raw sample-event streams because replayed sample/delay epochs made them irregular, and it chose the canonical `-1.85 s` offset from the task structure described in `methods.txt`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The code computes a single per-bin vector equal to `bin_center - (-1.85)` seconds and tiles that same vector across all trials in a session. This produces a continuous time-from-tone signal from about `-0.6` to `3.3` s within the go-centered window.

ii.
```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The notes say this was chosen to avoid ambiguous replayed sample events while remaining consistent with the canonical task timing in the paper.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined on the same `REL_CENTERS` vector used for neural binning, so it is aligned one-to-one with the neural time axis for every trial.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
...
input_trial = np.vstack([input_time[trial_idx], input_stim[trial_idx]]).astype(np.float16)
```

iii. The notes say both decoder inputs were intentionally made fully time-varying arrays with the same 80-bin structure as the neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset`, `intervals/trials/photostim_duration`, `intervals/trials/start_time`, and `acquisition/BehavioralEvents/go_start_times/timestamps`.

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

iii. The notes say the trial table stored photostim timing relative to trial start, so the agent converted it to go-centered timing to match the paper code's convention.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The code parses string-valued onset and duration fields, treats `"N/A"` as missing, converts onset and offset from trial-start coordinates into go-relative coordinates, then fills a binary `0/1` vector over bin centers for each trial.

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
stim[trial_idx, mask] = 1.0
```

iii. The notes justify this as the NWB equivalent of the reference preprocessing, which subtracts go-cue time from stimulation timing.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned by evaluating whether each neural bin center falls within the go-relative stimulation interval for that same trial.

ii.
```python
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
stim[trial_idx, mask] = 1.0
```

iii. The notes say the agent deliberately represented photostimulation as a full time-varying input on the same grid as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `intervals/trials/trial_instruction`, `intervals/trials/outcome`, `intervals/trials/start_time`, `intervals/trials/stop_time`, `go_start_times`, and the global left- and right-lick event streams.

ii.
```python
trial_instruction = decode_str_array(trials["trial_instruction"])[valid_trial_mask]
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
left_lick_times = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()].astype(np.float64)
right_lick_times = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()].astype(np.float64)
```

iii. The notes say this was the agent's NWB reconstruction of the reference code's `trial_type`, `lick_times`, and `lick_directions` behavior variables.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code maps left to `0` and right to `1`. For hit trials it uses the instructed side, for miss trials it uses the opposite side, and for ignore trials it tries to infer the first post-go lick side, then the first lick side anywhere in the trial, and finally falls back to the instructed side if no lick is present.

ii.
```python
instructed = np.where(trial_instruction == "left", 0, 1).astype(np.int16)
hit_mask = outcome_code == 2
miss_mask = outcome_code == 1
ignore_mask = outcome_code == 0

choice[hit_mask] = instructed[hit_mask]
choice[miss_mask] = 1 - instructed[miss_mask]
```

```python
for trial_idx in ignore_trials:
    choice[trial_idx] = lick_choice_with_fallback(...)
```

iii. The notes justify the ignore-trial fallback because most ignore trials had no post-go lick, so the agent documented a fallback rule instead of leaving choice undefined.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the NWB trial-table column `intervals/trials/outcome`.

ii.
```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

iii. The notes state that this mapping followed the decoder task specification exactly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code converts outcome strings to integer classes `ignore=0`, `miss=1`, `hit=2`, then repeats the trial label across all 80 time bins when forming the per-trial output matrix.

ii.
```python
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

```python
np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16)
```

iii. The notes say all outputs were stored as time-varying arrays for shape consistency, even when the value is constant across the trial.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This output does not exist in the agent's converted dataset. The question appears to be a template typo. For `outcome`, the code simply repeats the per-trial outcome label across the same 80 neural bins, so alignment is trivial rather than event-based.

ii.
```python
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_code[trial_idx], dtype=np.int16),
        np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16),
        np.full(N_BINS, early_code[trial_idx], dtype=np.int16),
        tongue_disc[trial_idx].astype(np.int16),
    ]
)
```

iii. The agent's notes repeatedly describe the output set as `choice`, `outcome`, `early_lick`, and `tongue_y_position`; there is no reward-zone variable.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes from the NWB trial-table column `intervals/trials/early_lick`.

ii.
```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
```

iii. The notes say this matched the trial-table encoding observed across the NWB files.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `"no early"` to `0` and `"early"` to `1`, then repeats that value across all bins for the trial.

ii.
```python
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
...
np.full(N_BINS, early_code[trial_idx], dtype=np.int16)
```

iii. The notes justify retaining early-lick trials because early lick itself is a requested decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position comes from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data`, specifically the `x`, `y`, and likelihood columns, plus that time series' timestamps.

ii.
```python
tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. The notes say all sessions had side-view tongue tracking and that these columns matched the expected `(x, y, likelihood)` schema.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The code cleans tracking by detecting large frame-to-frame velocity outliers using a 5-sigma threshold, linearly interpolating over those outliers, and replacing low-likelihood frames with the mean visible `y` value. It then samples the cleaned `y` signal at trial-aligned bin centers.

ii.
```python
speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
...
y[occluded_mask] = mean_y
```

```python
cleaned_tongue_y, tongue_clean_diag = clean_tongue_tracking(...)
aligned_tongue_y = align_tongue_y(...)
```

iii. The notes and trajectory say this was chosen to mirror the method paper's video-cleaning rules: five-sigma velocity outlier handling and mean imputation for occluded tongue positions.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After alignment, the code computes the 40th and 60th percentiles of all aligned tongue-y values in the included trials of a session. It assigns class `0` for values below `q40`, class `1` for values between `q40` and `q60`, and class `2` for values above `q60`.

ii.
```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

iii. The notes say the agent originally used `np.digitize`, then changed to explicit threshold logic so the middle category would not collapse when the percentile edges tied.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The code aligns tongue position to go cue by evaluating the previous video frame at each neural bin center (`go_time + REL_CENTERS`) using `np.searchsorted(..., side="right") - 1`. This is a last-frame-carried-forward alignment.

ii.
```python
abs_centers = go_times[:, None] + REL_CENTERS[None, :]
idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
idx = np.clip(idx, 0, len(timestamps) - 1)
return cleaned_y[idx]
```

iii. The notes explicitly say the agent chose last-frame-carried-forward because that was the closest match to the reference alignment scripts.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases explicitly: string `"N/A"` photostim values become `NaN` and then all-zero photostim bins; scalar spike-time or string arrays are coerced to arrays; tongue outliers are interpolated; low-likelihood tongue frames are set to mean visible `y`; sessions with zero good units are excluded; trials with no neural coverage or all-zero neural activity are removed; ignore-trial choice falls back to trial licks or instruction.

ii.
```python
if value == "N/A":
    continue
```

```python
if np.sum(keep_mask) >= 2:
    x[outlier_mask] = np.interp(...)
    y[outlier_mask] = np.interp(...)
else:
    x[outlier_mask] = np.nanmean(x)
    y[outlier_mask] = np.nanmean(y)
```

```python
if len(good_unit_indices) == 0:
    raise ValueError(...)
...
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
```

iii. The notes frame these as defensive fixes found during validation, especially the `obs_intervals` filter and the post-binning all-zero-trial removal.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning across all good units and trials. The code even records a timing around `bin_spikes_to_firing_rates`, and the notes say this dominated per-session runtime.

ii.
```python
t_neural = time.perf_counter()
firing_rates = bin_spikes_to_firing_rates(...)
...
neural_time_s = time.perf_counter() - t_neural
```

```python
"binning_time_s": float(neural_time_s),
```

iii. In the notes, the agent says the main runtime bottleneck was spike binning, while tracking cleanup and array assembly were smaller contributors.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over units during spike binning, over valid photostim trials when painting the binary matrix, over ignore trials when inferring choice, and over trials when packaging lists of `input_trials` and `output_trials`.

ii.
```python
for i, unit_idx in enumerate(good_unit_indices):
    spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
    ...
```

```python
for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
    stim[trial_idx, mask] = 1.0
```

```python
for trial_idx in range(n_trials):
    input_trial = np.vstack(...)
    output_trial = np.vstack(...)
```

iii. The notes say the heavy numerical parts were already mostly vectorized, but these loops remained because of ragged spike trains and per-trial assembly.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly decodes string-valued HDF5 arrays, repeatedly applies trial masks to many parallel arrays after each filtering step, and repeatedly allocates per-trial stacked matrices after the main computations are already complete.

ii.
```python
trial_instruction = decode_str_array(trials["trial_instruction"])[valid_trial_mask]
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
```

```python
if not np.all(nonzero_trial_mask):
    start_times = start_times[nonzero_trial_mask]
    stop_times = stop_times[nonzero_trial_mask]
    go_times = go_times[nonzero_trial_mask]
    ...
```

iii. The notes mention that the code had to propagate filters through many trial-level arrays and that it reuses common bin-center arrays where possible to limit redundant work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores diagnostics and processing plots that are not used by the downstream decoder. It also computes `region_names`, raw-unit counts, and several tracking-preview arrays inside `SessionResult`, but these are only for documentation and plotting. The `gc.collect()` call and retained diagnostics do not affect the saved decoder dataset.

ii.
```python
diagnostics = {
    "tracking_preview_time": ...,
    "tracking_preview_raw": ...,
    "tracking_preview_clean": ...,
    "aligned_tongue_y_trial": ...,
    "photostim_onsets_rel_go": ...,
    ...
}
```

```python
return SessionResult(
    ...,
    region_names=sorted(set(region_labels.tolist())),
    n_raw_units=int(len(classification)),
    diagnostics=diagnostics,
)
```

iii. The notes say these diagnostics were added for sanity checking and review rather than for the final decoder input format.
