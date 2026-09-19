# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files directly with `h5py`, not `pynwb`. It finds files with a sorted glob over `data/sub-*/*.nwb`, then pre-opens every file once in `get_nwb_files()` to keep only sessions with at least one `classification == "good"` unit. Each retained file is reopened in `process_session()`, where trials, units, behavioral events, and tongue tracking are read from HDF5 groups on demand.

ii. ```python
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
```

iii. In `CONVERSION_NOTES.md`, the AI says it used `h5py` for “lower overhead and explicit access to ragged arrays,” and that excluding the single zero-good-unit session up front matched the paper’s 173 analyzed sessions.

## 1-b. How are the data split into subjects?

i. The AI uses the NWB file’s parent directory name, e.g. `sub-440956`, as the subject identifier for each session. Subject indices are then assigned in first-seen order while assembling the output.

ii. ```python
session_id = path.stem
subject_id = path.parent.name
```

```python
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The notes state that the dataset is organized DANDI-style as `sub-<subject_id>/...`, so the folder name was treated as the canonical subject id. The AI did not use `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session order is the sorted path order, and each session id is the filename stem rather than `nwb.identifier`.

ii. ```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
session_id = path.stem
```

iii. The notes explicitly say “Each session is a single NWB file,” so the file boundary is the session boundary. The AI also wanted a deterministic order from sorting the paths.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table as the behavioral trial list and checks that the number of go cues matches the number of trial rows. After that, it subsets trials with a validity mask before further processing.

ii. ```python
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
```

iii. In the notes, the AI records that `go_start_times` is the only reliable one-event-per-trial stream, while `sample` and `delay` event streams can repeat because early licks replay epochs.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages. First, the AI keeps only trials whose full `[-2.5 s, +1.5 s]` go-aligned window falls inside the `obs_intervals` of the first good unit. Second, after spike binning it drops any residual trial whose firing rates are all zero across all good units and all bins. It does not explicitly filter `free_water`.

ii. ```python
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

iii. The notes justify this as a neural-coverage filter: the AI says sample verification exposed “real recording-coverage” gaps, so it switched to `obs_intervals` pre-filtering and kept a defensive all-zero-trial removal step.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` for units marked `classification == "good"`, together with `BehavioralEvents/go_start_times` to define trial-aligned windows.

ii. ```python
classification = decode_str_array(f["units/classification"])
good_unit_indices = np.flatnonzero(good_mask)
...
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The notes describe this as preserving the reference unit curation and go-cue-centered alignment while changing only the requested bin width.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into non-overlapping 50 ms bins over `[-2.5, 1.5]` around go cue, counts spikes with `np.searchsorted`, and divides by 0.05 s to produce firing rates. It uses `float16` storage and does not smooth, normalize, or baseline-subtract.

ii. ```python
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
```

```python
edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
counts = np.diff(edge_idx, axis=1)
firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
```

iii. The notes say the script must do “efficient spike binning from absolute NWB timestamps” and keep the explicit task requirement of 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are kept. A session is excluded if it has zero such units.

ii. ```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    raise ValueError(f"Session {session_id} has no good units")
```

iii. The notes say this is the “closest NWB-native analogue” of the authors’ external `goodunits` classifier output and explains the 173-session subset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go cue by adding a fixed vector of relative bin edges to each trial’s absolute go-cue timestamp. Spikes are binned directly on that absolute-time grid.

ii. ```python
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The notes repeatedly state that go cue is the unique per-trial anchor and that all streams should be centered on it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms. The AI constructs 80 non-overlapping bins spanning 4 s and does not apply any additional rebinning.

ii. ```python
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
```

iii. The notes explicitly frame this as a required deviation from the reference paper’s 40 ms / 3.4 ms preprocessing because the decoder task asked for 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI does not derive this input from a raw tone-onset event stream. Instead it uses a fixed constant, `TONE_ONSET_REL_GO_S = -1.85`, representing the canonical tone onset relative to go cue from task structure.

ii. ```python
TONE_ONSET_REL_GO_S = -1.85
```

```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The notes justify this by saying the raw sample-event stream contains replay-related extra events, so the AI chose the canonical `0.65 s` sample plus `1.2 s` delay structure instead of per-trial sample timestamps.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes a per-bin value by subtracting the fixed tone-onset offset from the go-centered bin centers, then tiles the same 80-value vector across all trials. There is no trial-specific tone inference.

ii. ```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The notes say this was chosen to avoid ambiguity from replayed sample epochs and to keep a single canonical task timing definition.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by construction: the AI uses the same 80 go-centered bin centers as the neural representation and stores one time value per neural bin.

ii. ```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The notes describe both inputs as fully time-varying arrays on the same bin grid as the spike data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `Photostimulation` is derived from the trials-table strings `photostim_onset` and `photostim_duration`, plus `start_time` and per-trial `go` times to convert trial-relative timing into go-relative timing.

ii. ```python
photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
```

```python
onset_trial = parse_optional_float_array(photostim_onset_str)
duration = parse_optional_float_array(photostim_duration_str)
go_minus_start = go_times - start_times
onset_rel_go = onset_trial - go_minus_start
offset_rel_go = onset_rel_go + duration
```

iii. The notes say the NWB table stores photostim timing relative to trial start, so it must be converted into the go-centered coordinates used everywhere else.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI parses `"N/A"` as `NaN`, computes stimulus onset and offset relative to go cue, then marks a bin as 1 if its center lies inside `[onset, offset)` and 0 otherwise.

ii. ```python
def parse_optional_float_array(strings: np.ndarray) -> np.ndarray:
    out = np.full(strings.shape, np.nan, dtype=np.float64)
    for i, value in enumerate(strings):
        if value == "N/A":
            continue
        out[i] = float(value)
```

```python
stim = np.zeros((len(go_times), N_BINS), dtype=np.float16)
for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
    stim[trial_idx, mask] = 1.0
```

iii. The notes call this a “binary per-bin input” and explicitly justify keeping stimulation trials because photostimulation is a decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI expresses photostimulation onset and offset relative to go cue and compares them against the same go-centered bin centers used for neural data.

ii. ```python
go_minus_start = go_times - start_times
onset_rel_go = onset_trial - go_minus_start
offset_rel_go = onset_rel_go + duration
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
```

iii. The notes explicitly state that photostimulation timing is converted into “go-centered coordinates” to match the neural alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `trial_instruction` and `outcome`, but it also consults the global left/right lick event streams for `ignore` trials. Hits are mapped to the instructed side, misses to the opposite side, and ignores use the first post-go lick if present, else the first lick anywhere in the trial, else the instructed side.

ii. ```python
def build_choice_array(
    trial_instruction: np.ndarray,
    outcome_code: np.ndarray,
    start_times: np.ndarray,
    go_times: np.ndarray,
    stop_times: np.ndarray,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> np.ndarray:
```

```python
choice[hit_mask] = instructed[hit_mask]
choice[miss_mask] = 1 - instructed[miss_mask]
...
choice[trial_idx] = lick_choice_with_fallback(...)
```

iii. The notes say the AI considered ignore trials an important edge case and chose a “lick-derived side when available, otherwise instructed side” fallback because many ignore trials have no post-go lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded only as `0 = left` and `1 = right`; there is no explicit `no lick` class. The trial-level label is repeated across all 80 bins in the output array.

ii. ```python
data = {
    ...
    "output_values": [
        ["left", "right"],
```

```python
output_trial = np.vstack(
    [
        np.full(N_BINS, choice_code[trial_idx], dtype=np.int16),
        ...
    ]
)
```

iii. The justification in the notes focuses on the ignore-trial fallback and on keeping all outputs in a uniform `(n_output, 80)` per-trial array. The notes do not justify dropping the `no lick` category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `Outcome` is taken directly from the trials-table `outcome` column.

ii. ```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
```

iii. The notes identify `outcome` as a native trial-table variable with the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome strings to `0 = ignore`, `1 = miss`, `2 = hit` and repeats the per-trial label across all bins.

ii. ```python
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

```python
np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16)
```

iii. The notes say this is a direct categorical remapping required by the decoder output format.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. `Early lick` is taken directly from the trials-table `early_lick` column.

ii. ```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
```

iii. The notes identify `early_lick` as an explicit per-trial table field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats the per-trial label across all bins.

ii. ```python
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
```

```python
np.full(N_BINS, early_code[trial_idx], dtype=np.int16)
```

iii. The notes justify this as the requested binary categorical output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `Tongue y-position` comes from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the `x`, `y`, and `likelihood` columns plus the series timestamps.

ii. ```python
tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. The notes state that all sessions contain this side-camera tongue tracking stream and that its columns are `(x, y, likelihood)`.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI performs several processing steps that differ from the reference solution. It detects frame-to-frame velocity outliers using `x` and `y` and linearly interpolates over them; uses a likelihood threshold of `0.9`; imputes low-likelihood frames to the mean visible `y`; aligns the cleaned `y` trace to each go-centered bin center via last-frame-carried-forward; then computes session percentiles from the aligned trial-bin values.

ii. ```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
```

```python
speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
...
x[outlier_mask] = np.interp(...)
y[outlier_mask] = np.interp(...)
...
mean_y = float(np.nanmean(y[visible_mask])) ...
y[occluded_mask] = mean_y
```

```python
aligned_tongue_y = align_tongue_y(...)
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
```

iii. The notes say this was intended to mirror “method-paper-inspired velocity-outlier interpolation” and “last-frame-carried-forward alignment,” while using only bins that actually enter the converted dataset.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI thresholds per-session aligned tongue values using the 40th and 60th percentiles of `aligned_tongue_y.reshape(-1)`. Values below `q40` become class 0, above `q60` become class 2, and everything else becomes class 1. It does not create a separate “not visible” class.

ii. ```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

```python
"output_values": [
    ...
    ["lt_40pct", "40_to_60pct", "gt_60pct"],
],
```

iii. The notes say this explicit threshold logic was added after `np.digitize` collapsed the middle class when percentile cutoffs tied.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data by evaluating the cleaned continuous tongue trace at each go-centered neural bin center. For each bin center, it uses the most recent preceding camera frame (`searchsorted(..., side="right") - 1`) and copies that `y` value into the trial-bin matrix.

ii. ```python
def align_tongue_y(
    timestamps: np.ndarray,
    cleaned_y: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

iii. The notes say this was chosen to match the reference repository’s marker-alignment style, which the AI interpreted as last-frame-carried-forward rather than interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by excluding sessions with no good units, excluding trials that fail the `obs_intervals` neural-coverage check, dropping residual all-zero neural trials, interpolating tongue-tracking outliers, and imputing low-likelihood tongue frames to the mean visible `y`. It does not preserve a dedicated missing/hidden tongue category.

ii. ```python
if len(good_unit_indices) == 0:
    raise ValueError(f"Session {session_id} has no good units")
```

```python
valid_trial_mask = compute_valid_trial_mask(obs_intervals=obs_intervals, go_times=go_times_all)
...
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
```

```python
x[outlier_mask] = np.interp(...)
y[outlier_mask] = np.interp(...)
...
y[occluded_mask] = mean_y
```

iii. The notes justify these choices as making the converted set contain only trials with full neural coverage while cleaning tongue data with a five-sigma velocity rule and imputation inspired by the method paper.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are opening and reading NWB files, loading large ragged spike buffers and tongue-tracking arrays, and the per-unit spike-binning loop. The code also records per-session spike-binning runtime in diagnostics.

ii. ```python
with h5py.File(path, "r") as f:
    ...
    tongue_data = tongue_ts["data"][()]
    ...
    spike_times_flat = f["units/spike_times"][()]
```

```python
t_neural = time.perf_counter()
firing_rates = bin_spikes_to_firing_rates(...)
neural_time_s = time.perf_counter() - t_neural
```

iii. The notes say runtime was dominated by conversion I/O and spike binning, and that `obs_intervals` pre-filtering plus compact dtypes kept the full run under the 15-minute target.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over files, units, valid photostim trials, ignore trials, per-trial output assembly, and per-unit brain-region indexing. The most expensive loop is the per-unit spike-binning loop; the per-trial photostim and output-packing loops could also be vectorized further.

ii. ```python
for i, unit_idx in enumerate(good_unit_indices):
    spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
    edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
```

```python
for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
    stim[trial_idx, mask] = 1.0
```

```python
for trial_idx in range(n_trials):
    input_trial = np.vstack([input_time[trial_idx], input_stim[trial_idx]]).astype(np.float16)
    output_trial = np.vstack([...])
```

iii. The notes emphasize that spike binning was already vectorized across trials and that some simpler loops were left in place because the overall runtime was acceptable.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work. It opens every NWB file once in `get_nwb_files()` just to inspect `units/classification`, then reopens retained files in `process_session()`. It also reparses photostim onset strings a second time for diagnostics after already building the photostim matrix.

ii. ```python
for path in files:
    with h5py.File(path, "r") as f:
        good = decode_str_array(f["units/classification"]) == "good"
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

iii. The notes partly justify this by describing the up-front session filter and by keeping extra diagnostic summaries for review, but they do not present it as an optimization.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several diagnostic-only values that are not saved into `converted_data.pkl`: preview arrays for tongue tracking, per-session timing diagnostics, raw unit counts, `region_names`, and optional plotting outputs. It also constructs `SessionResult` bookkeeping fields that are thrown away after dataset assembly.

ii. ```python
diagnostics = {
    "tracking_preview_time": tongue_timestamps[:preview_n] - go_times[0],
    "tracking_preview_raw": tongue_y[:preview_n],
    "tracking_preview_clean": cleaned_tongue_y[:preview_n],
    ...
    "binning_time_s": float(neural_time_s),
}
```

```python
return SessionResult(
    ...
    region_names=sorted(set(region_labels.tolist())),
    n_raw_units=int(len(classification)),
    diagnostics=diagnostics,
)
```

```python
if args.show_processing and len(results) <= 2:
    plot_name = f"processing_{result.session_id}.png"
    make_session_plot(result, Path(plot_name))
```

iii. The notes justify these as validation and review aids: the AI repeatedly mentions processing plots, runtime summaries, and extra sanity-check diagnostics used during conversion, even though they are not part of the final dataset.
