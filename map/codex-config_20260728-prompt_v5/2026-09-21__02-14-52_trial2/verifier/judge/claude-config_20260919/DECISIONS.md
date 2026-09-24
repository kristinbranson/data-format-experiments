# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is a DANDI-style NWB bundle: one `.nwb` file per session under `/app/data/sub-<id>/`. The AI walks the whole `/app/data` tree with `os.walk`, collects every path ending in `.nwb`, and sorts the list (174 files). It then makes a **second pass** over every file (`discover_curated_sessions` → `session_has_good_units`) that opens each file purely to read `units/classification`, keeping only sessions that contain at least one `"good"` unit (173 of 174). Each surviving file is then opened a third time in `process_session`, where trials, behavioural events, camera tracking, units and electrode metadata are read. Files are read with raw **`h5py`**, not `pynwb`: the AI reads the HDF5 groups directly (`intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `units`, `general/extracellular_ephys/electrodes`), and re-implements NWB ragged-array indexing itself via `ragged_rows()` using the `*_index` offset datasets.

ii.
```python
def list_session_files() -> list[str]:
    files = []
    for dirpath, _, filenames in os.walk(DATA_ROOT):
        for name in filenames:
            if name.endswith(".nwb"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def session_has_good_units(path: str) -> bool:
    with h5py.File(path, "r") as h5:
        cls = decode_vector(h5["units"]["classification"][:])
    return "good" in cls


def discover_curated_sessions() -> list[str]:
    files = list_session_files()
    curated = [path for path in files if session_has_good_units(path)]
    return curated
```

```python
def ragged_rows(flat: np.ndarray, index: np.ndarray, row: int) -> np.ndarray:
    start = 0 if row == 0 else int(index[row - 1])
    end = int(index[row])
    return flat[start:end]
```

```python
with h5py.File(path, "r") as h5:
    trials = h5["intervals"]["trials"]
    ev = h5["acquisition"]["BehavioralEvents"]
    tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
    units = h5["units"]
    electrode_locations = decode_vector(h5["general"]["extracellular_ephys"]["electrodes"]["location"][:])
```

iii. From CONVERSION_NOTES Step 6: "Implemented raw NWB loading with `h5py` only, so the conversion stays close to the provided source data and avoids extra dependency / serialization layers." Step 2 documents that the bundle is 174 NWB files across 28 `sub-*` folders, one file per session, and Step 4 records the deliberate decision to "Use NWB as the raw source, but reproduce the logical processing from the `.mat` pipeline" because the reference code was written for DataJoint-exported `.mat` files that are not shipped here.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the **parent directory name** of each NWB file, i.e. the literal string `sub-440956` (including the `sub-` prefix), rather than from the NWB `subject/subject_id` field (which holds the bare numeric string `440956`). At assembly, `subjects` is the sorted set of unique directory names and `subject_idx` is one index per session into that list. This yields 28 subjects with 3–10 sessions each.

ii.
```python
def get_subject_id(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

```python
subjects = sorted({sess["subject"] for sess in session_dicts})
subject_to_idx = {sub: i for i, sub in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array([subject_to_idx[sess["subject"]] for sess in session_dicts], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "NWB subject folder name `sub-xxxxx` → `subjects`, `subject_idx` — Preserve as subject IDs — One subject index per kept session." Step 2 documents that the data directory contains "28 subject folders named `sub-<id>/`", so the folder name is treated as the canonical animal id. Step 3/4 cross-check this against the data paper's "This study is based on data from 28 mice".

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is performed. The session identifier is derived from the **filename** by stripping the modality suffix, e.g. `sub-440956_ses-20190207T120657`. Session order in the output follows the sorted file list (so chronological within subject, because the filename embeds the acquisition timestamp). One session is dropped at the session level: the single file with zero `classification == "good"` units, giving 173 sessions in the output. Per-session `session_id`, `subject`, `n_good_units` and `n_kept_trials` are recorded in `metadata['session_info']`.

ii.
```python
def get_session_id(path: str) -> str:
    return os.path.basename(path).replace("_behavior+ecephys+ogen.nwb", "").replace("_behavior+ecephys.nwb", "")
```

```python
curated_sessions = discover_curated_sessions()
print(f"[setup] curated sessions with good units: {len(curated_sessions)}")
...
for path in curated_sessions:
    session_id = get_session_id(path)
    session_data, session_summary = process_session(path, make_plot=session_id in plot_ids)
```

```python
"session_info": [
    {
        "session_id": summary["session_id"],
        "subject": summary["subject"],
        "n_good_units": summary["n_good_units"],
        "n_kept_trials": summary["n_kept_trials"],
    }
    for summary in session_summaries
],
```

iii. CONVERSION_NOTES Step 2: "174 NWB files total, one NWB file per session", with the naming convention `sub-<subject>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys(+ogen).nwb`. Step 4 resolves the session-count discrepancy: "One NWB session, `sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`, has 1,852 units but zero `classification == "good"` units and all classifier labels are `nan`. Excluding this session recovers the paper's 173-session count." Step 5 Key Decision 1: "Use 173 sessions, not all 174 raw NWBs".

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial, read as parallel arrays (`start_time`, `stop_time`, `outcome`, `early_lick`, `trial_instruction`, `auto_water`, `free_water`, `photostim_onset`, `photostim_duration`). The per-trial go cue is taken as the *i*-th entry of `BehavioralEvents/go_start_times/timestamps`, i.e. the code assumes a strict one-go-cue-per-trial-row correspondence and indexes `go_times[i]` by trial-table row index. No assertion is made that `len(go_times) == len(trial_start)`. Trial-phase events whose counts can exceed the trial count (`sample_start_times`) are *not* indexed positionally; they are resolved by time search instead (see 3-a).

ii.
```python
trials = h5["intervals"]["trials"]
trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
outcome_text = decode_vector(trials["outcome"][:])
early_text = decode_vector(trials["early_lick"][:])
trial_instruction = decode_vector(trials["trial_instruction"][:])
auto_water = np.asarray(trials["auto_water"][:], dtype=np.int64)
free_water = np.asarray(trials["free_water"][:], dtype=np.int64)
photo_onset_text = decode_vector(trials["photostim_onset"][:])
photo_dur_text = decode_vector(trials["photostim_duration"][:])

ev = h5["acquisition"]["BehavioralEvents"]
go_times = np.asarray(ev["go_start_times"]["timestamps"][:], dtype=np.float64)
```

```python
for i in range(len(trial_start)):
    ...
    go = float(go_times[i])
    edges_abs = go + BIN_EDGES_REL
```

iii. CONVERSION_NOTES Step 2 documents that the trial table columns are "consistent across all 174 files" and lists them. Step 4 explicitly flags that `sample_start_times` / `delay_start_times` event counts "can exceed trial count" because "early licks trigger replay of sample/delay epochs", and therefore resolves the tone by time search rather than by row index — implying that `go_start_times` was treated as the one stream that is reliably one-per-trial.

## 1-e. How are trials filtered based on quality controls?

i. Six filters are applied, in order:

1. **Ephys coverage** — only trials whose `(start_time, stop_time)` pair exactly matches an interval in the first good unit's `units/obs_intervals` are kept. Matching is done on an integer key at 0.1 ms precision, and an unmatched `obs_interval` raises an error. (removes 1,060 trials)
2. **`auto_water == 1` or `free_water == 1`** excluded. (removes 3,764 trials)
3. **Full camera coverage** — the trial is dropped unless the entire `[go-2.5, go+1.5)` window lies inside the session's tongue-tracking timestamp range. (removes 765 trials)
4. **Tone present** — the trial is dropped if there is no `sample_start` event in `[trial_start, go]`. (removes 0)
5. **All-zero neural** — after binning, a trial whose whole firing-rate matrix is zero is dropped. (removes 2 trials)
6. A session with fewer than 2 surviving trials **raises `ValueError`** (aborting the run) rather than being skipped; this never triggered.

Early-lick, `ignore`, and photostim trials are deliberately **kept**. Net: 94,370 → 88,779 trials over the 173 retained sessions (−5.9%).

ii.
```python
def time_key(value: float) -> int:
    """Stable key for matching NWB trial intervals stored at 0.1 ms precision."""
    return int(round(float(value) * 10000.0))


def map_obs_intervals_to_trial_indices(trial_start, trial_stop, obs_intervals) -> np.ndarray:
    trial_lookup = {
        (time_key(start), time_key(stop)): idx
        for idx, (start, stop) in enumerate(zip(trial_start, trial_stop, strict=False))
    }
    mapped = []
    for start, stop in obs_intervals:
        key = (time_key(start), time_key(stop))
        if key not in trial_lookup:
            raise ValueError(f"Could not map obs interval {(start, stop)} onto trial table.")
        mapped.append(trial_lookup[key])
    return np.asarray(mapped, dtype=np.int64)
```

```python
# `obs_intervals` gives the subset of behavioral trials with ephys coverage.
coverage_obs = ragged_rows(obs_intervals_flat, obs_intervals_index, good_units[0])
recorded_trial_idx = map_obs_intervals_to_trial_indices(trial_start, trial_stop, coverage_obs)
recorded_trial_set = set(recorded_trial_idx.tolist())
```

```python
for i in range(len(trial_start)):
    if i not in recorded_trial_set:
        continue
    if auto_water[i] == 1 or free_water[i] == 1:
        continue
    go = float(go_times[i])
    edges_abs = go + BIN_EDGES_REL
    if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]:
        continue
    sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
    if sample_start is None:
        continue
```

```python
if not np.any(neural):
    dropped_all_zero_neural += 1
    continue
```

```python
if len(candidate_trial_idx) < 2:
    raise ValueError(f"{session_id}: fewer than 2 trials remain after filtering")
...
if len(trial_keep) < 2:
    raise ValueError(f"{session_id}: fewer than 2 trials remain after neural validation")
```

iii. CONVERSION_NOTES Step 5 Key Decisions 5–7: keep photostim trials ("Decoder inputs explicitly require photostimulation state"); keep early-lick and ignore trials ("Decoder outputs explicitly require `early lick` and `outcome` including `ignore`, so excluding them would remove requested labels"); exclude auto/free water ("These are atypical task contingencies that the reference analyses usually exclude, and they are not represented in the requested decoder inputs/outputs. Keeping them would add label noise the decoder cannot account for"). The auto/free-water exclusion is traced in Step 1 to the reference function `get_regular_trial_mask`, which "defines 'regular trials' as no early lick, no auto water, no free water, no no-response, and no stimulation" — the AI adopts the subset of that mask not contradicted by the decoder spec. The `obs_intervals` and all-zero-neural filters were added reactively: Step 7 records that "Initial sample verification exposed a block of all-zero neural trials in session 2. Investigation showed these were trials outside the session's usable ephys coverage." The camera-coverage filter is justified in Step 10 check (b) as required "because the decoder uses both neural and tongue streams", and its cost is explicitly investigated in the Step 10 edge-case check: "Low-trial session `sub-484676_ses-20210413T112028`: … only `7` survive because `515` fail the tongue-window coverage requirement and `28` are auto/free-water; confirms the low converted trial count is due to missing video coverage, not a conversion bug."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a single concatenated session-absolute spike-time vector) together with `units/spike_times_index` (per-unit end offsets), restricted to the rows where `units/classification == "good"`. `BehavioralEvents/go_start_times/timestamps` supplies the per-trial alignment time that places the bin edges. `units/anno_name` plus `units/electrodes` → `general/extracellular_ephys/electrodes/location` supply the per-neuron region label that accompanies the neural matrix.

ii.
```python
units = h5["units"]
classification = decode_vector(units["classification"][:])
anno_name = decode_vector(units["anno_name"][:])
spike_flat = np.asarray(units["spike_times"][:], dtype=np.float64)
spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
obs_intervals_flat = np.asarray(units["obs_intervals"][:], dtype=np.float64)
obs_intervals_index = np.asarray(units["obs_intervals_index"][:], dtype=np.int64)
electrodes = np.asarray(units["electrodes"][:], dtype=np.int64)
electrode_locations = decode_vector(h5["general"]["extracellular_ephys"]["electrodes"]["location"][:])

good_units = [i for i, c in enumerate(classification) if c == "good"]
```

```python
# Pre-slice spike times once per good unit
unit_spike_times = [ragged_rows(spike_flat, spike_index, unit_idx) for unit_idx in good_units]
```

iii. CONVERSION_NOTES Step 1: "Neural modality is electrophysiology, not imaging. No `dF/F` computation is involved anywhere in the reference loading path." Step 5 maps "NWB `units` rows with `classification == "good"` and session-level `spike_times`" → `neural`. Spike times are the only neural representation in the file, so firing rates must be computed from them.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin **firing rates in Hz**. For each kept trial the 81 absolute bin edges are formed as `go + BIN_EDGES_REL`; for each good unit, `np.searchsorted(spikes, edges, side="left")` gives the running spike count at each edge and `np.diff` gives the count per bin; counts are divided by the 0.05 s bin width. No smoothing, no normalisation, no baseline subtraction, no sliding/overlapping windows. Result per trial is `(n_good_units, 80)` `float32`.

ii.
```python
def bin_spike_rates(spike_times_abs: np.ndarray, trial_edges_abs: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(spike_times_abs, trial_edges_abs, side="left")
    counts = np.diff(idx)
    return counts.astype(np.float32) / BIN_SIZE
```

```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    go = float(go_times[trial_idx])
    edges_abs = go + BIN_EDGES_REL

    # Neural
    neural = np.empty((len(good_units), NBINS), dtype=np.float32)
    for unit_row, spikes_abs in enumerate(unit_spike_times):
        neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. CONVERSION_NOTES Step 6: "Implemented spike binning as firing rates (`spike count / 0.05 s`) from absolute session spike times using `np.searchsorted` against absolute trial bin edges." Step 5 maps this to the reference functions `process_one_area` / `sliding_histogram`, which in the reference repo also return `binSpikes / bin_width`. Step 10 check (d) records the deliberate deviation: reference uses "40 ms width with 3.4 ms stride"; the conversion uses "50 ms non-overlapping bins over `[-2.5, 1.5)` s — intentional task-required deviation; underlying spike counting / event alignment logic preserved."

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single unit-level filter: keep units with `units/classification == "good"`. No thresholds are applied to the individual QC metrics stored in the file (`unit_snr`, `isi_violation`, `presence_ratio`, `amplitude_cutoff`, `drift_metric`, …), and `unit_quality` is explicitly *not* used. A session with no `"good"` unit is excluded at discovery time. `units/is_good_trials` is explicitly rejected as a filter. This retains 69,453 of 272,227 units (25.5%), mean 401.5 per session, across 173 sessions.

ii.
```python
good_units = [i for i, c in enumerate(classification) if c == "good"]
if not good_units:
    raise ValueError(f"{session_id}: no classifier-good units")
```

```python
def session_has_good_units(path: str) -> bool:
    with h5py.File(path, "r") as h5:
        cls = decode_vector(h5["units"]["classification"][:])
    return "good" in cls
```

```python
"unit_filter": "units.classification == good",
"session_filter": "sessions with at least one units.classification == good",
```

iii. CONVERSION_NOTES Step 4: "`unit_quality == "good"` is too permissive (154,948 units). `classification == "good"` yields 69,453 units, close to the paper's 69,943. Therefore `classification` is the correct NWB analog of the paper's good-unit list." Step 1 traces this to the reference preprocessing, which "loads a session-specific `goodunits` file (`qc_mode='classifier'`) and uses those approved neuron indices" rather than recomputing metrics. Step 5 Key Decisions 2–4 restate this and reject `is_good_trials`: "It is almost always true for classifier-good units in sampled sessions, and the reference code did not use a per-unit per-trial mask in preprocessing." Step 3 cross-checks the yield against the QC white paper's "25.9 % of clusters reported by Kilosort2". The residual 490-unit gap versus the paper's 69,943 is documented and attributed to "a release/version difference or small conversion discrepancy between the DANDI NWB export and the exact paper-analysis snapshot."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go cue onset**, taken as `BehavioralEvents/go_start_times/timestamps[trial_idx]`. All NWB streams share one session-absolute clock, so no resampling, interpolation or per-stream offset correction is applied: the fixed relative edge grid is simply added to each trial's go-cue time to produce absolute edges, and spikes are binned against those absolute edges.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_EDGES_REL = np.linspace(WINDOW_START, WINDOW_END, NBINS + 1, dtype=np.float64)
```

```python
go = float(go_times[trial_idx])
edges_abs = go + BIN_EDGES_REL

neural = np.empty((len(good_units), NBINS), dtype=np.float32)
for unit_row, spikes_abs in enumerate(unit_spike_times):
    neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

```python
"temporal_alignment_event": "Go cue onset",
"off_start": WINDOW_START,
"off_end": WINDOW_END,
```

iii. CONVERSION_NOTES Step 1: "Reference code explicitly subtracts per-trial go-cue time from lick times and stimulation on/off times, then stores `gocue_time` separately. This confirms trialwise temporal alignment is centered on go cue in the preprocessed representation." Step 10 check (c): "reference: go-cue-centered alignment for licks, stimulation, neural data, and markers; conversion: all trial tensors are aligned to `go_start_times` — assessment: matched." The instructions also mandate go-cue alignment directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, giving exactly **80 non-overlapping bins** spanning `[-2.5, +1.5)` s relative to the go cue, identical for every trial and session. There is no rebinning step: spike times are binned once, directly at 50 ms, straight from the raw spike-time vector — there is no intermediate finer representation that is later aggregated. The tongue stream, which is natively at ~3.4 ms, *is* downsampled onto the same 50 ms grid (by last-frame-in-bin, see 8-b). `metadata['time_bin_size']` is set to `50.0` ms and `metadata['bin_centers_rel_go_s']` stores the 80 bin centres.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES_REL = np.linspace(WINDOW_START, WINDOW_END, NBINS + 1, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

```python
"time_bin_size": 50.0,
"off_start": WINDOW_START,
"off_end": WINDOW_END,
"n_timepoints": NBINS,
"bin_centers_rel_go_s": BIN_CENTERS_REL.astype(np.float32),
```

iii. CONVERSION_NOTES Step 6: "Implemented per-trial go-cue-centered extraction over `[-2.5, 1.5)` s with exactly 80 non-overlapping 50 ms bins." Step 4 records the reasoning for departing from the paper's binning: "Methodpaper explicitly states 40 ms bins with 3.4 ms stride. For consistency checks, match reference-style 40 ms / 3.4 ms processing when comparing to paper/code. For the final converted decoder dataset, adapt to the user-required 50 ms bins while preserving the same underlying alignment and QC choices where possible." Step 5 Key Decision 12: "Respect reference timing, then adapt binning/window only where required."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times/timestamps` (the sample-epoch / tone onsets) together with the trial's go cue and the trial's `start_time`. For each trial, the tone used is the **latest `sample_start` in the closed interval `[trial_start, go]`**. A trial with no such event is dropped.

ii.
```python
def last_event_before(times: np.ndarray, lo: float, hi: float) -> float | None:
    """Latest event in [lo, hi], else None."""
    idx = np.searchsorted(times, hi, side="right") - 1
    if idx < 0:
        return None
    t = float(times[idx])
    if t < lo or t > hi:
        return None
    return t
```

```python
sample_start_times = np.asarray(ev["sample_start_times"]["timestamps"][:], dtype=np.float64)
...
sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
if sample_start is None:
    continue
...
candidate_sample_onsets_rel.append(sample_start - go)
```

iii. CONVERSION_NOTES Step 4: "The extra sample/delay events are consistent with replayed epochs after early licks. For each trial, the definitive tone/sample onset should be the **last** `sample_start` before that trial's go cue, not simply the first event in the trial." Step 5 mapping table: "Use the **last** sample start before go to handle replayed epochs after early licks." The additional `>= trial_start` bound prevents borrowing a tone from the preceding trial. Step 7 validated this: "final sample onset histogram is sharply concentrated at `-1.85 s`, with a very small tail of earlier replayed sample epochs", matching the paper's 0.65 s sample + 1.2 s delay structure.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone time is expressed relative to the go cue once (`tone_on_rel = sample_start - go`, normally ≈ −1.85 s). The input is then a continuous, time-varying row: the go-cue-relative bin centre minus that offset, i.e. seconds elapsed since tone onset at each bin centre. Stored as `float32` in row 0 of the `(2, 80)` input array. Full-dataset range is `[-1.5, 11.9]` s.

ii.
```python
tone_on_rel = float(candidate_sample_onsets_rel[kept_idx])
input_arr = np.empty((2, NBINS), dtype=np.float32)
input_arr[0] = (BIN_CENTERS_REL - tone_on_rel).astype(np.float32)
```

```python
INPUT_NAMES = ["time_from_tone_onset_s", "photostimulation_on"]
```

iii. CONVERSION_NOTES Step 5 mapping table: "Compute signed time-from-tone-onset at each 50 ms bin center: `t_rel - sample_onset_rel`." Step 6: "Implemented tone onset reconstruction as the last `sample_start` before each go cue." The instructions specify this input as "continuous, time-varying", so it is emitted as a per-bin value rather than a per-trial scalar or a binary event marker.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the input row is evaluated on `BIN_CENTERS_REL`, the centres of exactly the same go-cue-relative 80-bin grid whose edges (`BIN_EDGES_REL`) are used to bin the spikes. Bin *k* of the input therefore covers the same interval as bin *k* of the firing rates, with no separate alignment step. The per-trial tone offsets are carried in `candidate_sample_onsets_rel` and indexed by the same `kept_idx` that indexes the trial being binned.

ii.
```python
BIN_EDGES_REL = np.linspace(WINDOW_START, WINDOW_END, NBINS + 1, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    go = float(go_times[trial_idx])
    edges_abs = go + BIN_EDGES_REL          # neural bin edges
    ...
    tone_on_rel = float(candidate_sample_onsets_rel[kept_idx])
    input_arr[0] = (BIN_CENTERS_REL - tone_on_rel).astype(np.float32)
```

iii. CONVERSION_NOTES Step 10 check (e): "reference: sample/tone and stimulation timing represented relative to go cue; conversion: `time_from_tone_onset_s` comes from the last `sample_start` before go — assessment: matched in source variables and alignment, adapted to required target format." Step 10 sanity check 2 verified the full `(2, 80)` input matrix for a trial against an independent recomputation from the raw NWB, with `np.allclose == True` and max absolute difference `0.0`, explicitly to confirm "the converted trial index is correctly offset relative to raw trial index after filtering".

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials-table columns `photostim_onset` and `photostim_duration`, both stored as **strings** with the sentinel `'N/A'` on unstimulated trials, plus `start_time` (the onsets are measured from trial start) and the go cue (to place them on the bin axis). The AI planned to prefer the `photostim_start_times` / `photostim_stop_times` event streams with the trial table as fallback, but the shipped code uses the trial table exclusively.

ii.
```python
photo_onset_text = decode_vector(trials["photostim_onset"][:])
photo_dur_text = decode_vector(trials["photostim_duration"][:])
```

```python
if photo_onset_text[i] != "N/A" and photo_dur_text[i] != "N/A":
    stim_on = float(trial_start[i]) + float(photo_onset_text[i])
    stim_off = stim_on + float(photo_dur_text[i])
else:
    stim_on = math.nan
    stim_off = math.nan
...
candidate_photo_intervals_abs.append((stim_on, stim_off))
```

iii. CONVERSION_NOTES Step 5 mapping table: "`photostim_start_times` / `photostim_stop_times` (fallback: trial table `photostim_onset` + `photostim_duration`) → `input[1]` … Kept as time-varying because decoder input explicitly requires it"; Step 6 records the shipped implementation: "Implemented photostimulation input from trial-table onset/duration fields as a time-varying binary input at 50 ms bin centers." Step 2 documents the prevalence: "photostim trials are present in 18,588 / 94,990 trials (`19.57%`)", cross-checked in Step 3 against the paper's "typically 25%".

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0.0/1.0) time series over the 80 bins: a bin is 1 when its **absolute** centre `go + BIN_CENTERS_REL` lies in `[stim_on, stim_off)`. Unstimulated trials (NaN bounds) are short-circuited to an all-zero row rather than relying on NaN comparison. Stored as `float32` in row 1 of the input array. Because photoinhibition is delivered in the last 0.5 s of the delay and always ends before the go cue, the 1s fall in the pre-zero part of the window.

ii.
```python
stim_on, stim_off = candidate_photo_intervals_abs[kept_idx]
if math.isnan(stim_on):
    input_arr[1] = 0.0
else:
    abs_centers = go + BIN_CENTERS_REL
    input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off)).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Keep photostimulation trials: Decoder inputs explicitly require photostimulation state, so these trials must remain." The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)", so a per-bin binary series is used rather than a per-trial flag. Step 3 recorded the expected timing from the paper — "We silenced ALM activity during the late delay epoch (last 0.5 s) … Thus, photoinhibition always ended before the 'Go' cue" — and Step 5 planned the sanity check "Spot-check photostim binary input against raw event times, including that stimulation ends before go cue."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset/offset are kept in absolute session time and compared against the absolute bin centres `go + BIN_CENTERS_REL` — the centres of the same go-cue-anchored grid used for the spike edges. No interpolation or offset correction; both streams already share the NWB global clock.

ii.
```python
abs_centers = go + BIN_CENTERS_REL
input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off)).astype(np.float32)
```

```python
stim_on = float(trial_start[i]) + float(photo_onset_text[i])   # trial-start-relative -> absolute
stim_off = stim_on + float(photo_dur_text[i])
```

iii. CONVERSION_NOTES Step 1 notes that the reference code "explicitly subtracts per-trial go-cue time from … stimulation on/off times", i.e. stimulation is represented on the go-cue axis; Step 10 check (e) assesses the conversion as "matched in source variables and alignment". The Step 10 input sanity check (`np.allclose == True`, max abs diff `0.0` on the full `(2, 80)` matrix) covered the photostim row and is described as confirming "photostimulation timing".

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column in the file. The AI derives choice from the **actual lick events**: `BehavioralEvents/left_lick_times/timestamps` and `right_lick_times/timestamps`, evaluated in the 1.5 s response window after each trial's `go_start_time`. `trial_instruction` is read but is *not* used for choice (it is unused in the final code path). This differs from the human reference, which derives choice arithmetically from `trial_instruction × outcome`.

ii.
```python
left_lick_times = np.asarray(ev["left_lick_times"]["timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(ev["right_lick_times"]["timestamps"][:], dtype=np.float64)
```

```python
choice = derive_choice_per_trial(go_times, left_lick_times, right_lick_times)
```

iii. CONVERSION_NOTES Step 4: "NWB trials table has `trial_instruction` and session-level left/right lick events, but no explicit per-trial choice column … Reconstruct per-trial choice from the first left/right lick after `go_start` within the 1.5 s response window. In checks across sample sessions this matched hits perfectly, matched misses as opposite-to-instruction essentially perfectly, and mapped ignore trials to no-lick." Step 1 notes the reference `.mat` export carried `behavior_lick_directions` / `behavior_lick_times`, so licks are the reference's own source for choice; Step 5 maps this as "Reference `.mat` used `lick_directions`; NWB reconstruction validated in Step 4."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial the left and right lick trains are masked to `[go, go + 1.5)`; the side of the **earliest** lick in that window sets the class. Coding is `0 = left`, `1 = right`, `2 = no lick` (the default when neither train has an event in the window). The per-trial integer is then broadcast across all 80 bins into row 0 of an `(4, 80)` `int64` output array. `RESPONSE_WINDOW = 1.5` matches the paper's response epoch. Full-dataset distribution: `[0.430, 0.421, 0.149]`, whose no-lick fraction matches the `ignore` fraction exactly.

ii.
```python
RESPONSE_WINDOW = 1.5

def derive_choice_per_trial(go_times, left_lick_times, right_lick_times) -> np.ndarray:
    """Choice from the first lick after go cue within the 1.5 s response window."""
    choice = np.full(len(go_times), 2, dtype=np.int64)  # default no lick
    for i, go in enumerate(go_times):
        end = go + RESPONSE_WINDOW
        left = left_lick_times[(left_lick_times >= go) & (left_lick_times < end)]
        right = right_lick_times[(right_lick_times >= go) & (right_lick_times < end)]
        if len(left) == 0 and len(right) == 0:
            continue
        if len(left) > 0 and (len(right) == 0 or left[0] < right[0]):
            choice[i] = 0
        elif len(right) > 0 and (len(left) == 0 or right[0] < left[0]):
            choice[i] = 1
    return choice
```

```python
output_arr = np.empty((4, NBINS), dtype=np.int64)
output_arr[0] = choice[trial_idx]
```

```python
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ...
]
```

iii. CONVERSION_NOTES Step 5 planned categorical mapping "`0 = left`, `1 = right`, `2 = no lick`", following the instructions' "(left, right, no lick)" ordering. Key Decision 8: "Use all outputs as time-varying `(d_output, T)` arrays: Choice, outcome, and early-lick labels will be broadcast across the 80 bins so they can coexist with time-varying tongue-y output in a single consistent tensor shape" — which also satisfies the instruction's preference to "If at all possible, make it time-varying".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`. No derivation.

ii.
```python
outcome_text = decode_vector(trials["outcome"][:])
```

iii. CONVERSION_NOTES Step 2 confirms `outcome` takes exactly the values `hit`, `miss`, `ignore` across the whole dataset, matching the three classes the instructions ask for, so the column is used as-is. Step 5 mapping table: "Trial table `outcome` → `output[1]` outcome".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a fixed dictionary to `0 = ignore`, `1 = miss`, `2 = hit` and broadcast across all 80 bins into row 1 of the output array. Full-dataset distribution `[0.149, 0.167, 0.683]`; the derived control (no-early, no-photostim, responded) hit rate is 0.8147, against the paper's reported 84%.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int64)
```

```python
output_arr[1] = outcome[trial_idx]
```

```python
OUTPUT_VALUES = [
    ...
    ["ignore", "miss", "hit"],
    ...
]
```

iii. CONVERSION_NOTES Step 5 mapping table: "Map `ignore->0`, `miss->1`, `hit->2`; broadcast across all 80 bins … Preserve ignore trials because decoder output explicitly includes them." The code ordering follows the instructions' "(ignore, miss, hit)". The dictionary lookup will raise `KeyError` on any unexpected string, which acts as an implicit assertion; Step 2 established that no other values occur.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the trials-table `early_lick` column, which holds the strings `'no early'` and `'early'`. No derivation.

ii.
```python
early_text = decode_vector(trials["early_lick"][:])
```

iii. CONVERSION_NOTES Step 2 confirms `early_lick` takes exactly `early` / `no early` across the dataset. Step 5 mapping table: "Trial table `early_lick` → `output[2]` early lick … Keep early-lick trials because decoder output explicitly includes them." This is a conscious departure from the data paper's "Early lick trials and no response trials were excluded for analysis" (Step 3), overridden because the decoder spec requires the label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped through a fixed dictionary to `0 = no`, `1 = yes` and broadcast across all 80 bins into row 2. Full-dataset distribution `[0.884, 0.116]`. The early lick itself occurs during the sample or delay epoch, so the generating event falls inside the −2.5 s pre-go portion of the window even though the label is per-trial.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_text], dtype=np.int64)
```

```python
output_arr[2] = early[trial_idx]
```

```python
OUTPUT_VALUES = [
    ...
    ["no", "yes"],
    ...
]
```

iii. CONVERSION_NOTES Step 5: "early lick: `0 = no`, `1 = yes`", following the instructions' "(no, yes)" ordering, and Key Decision 8 (broadcast per-trial labels across bins for a uniform tensor shape).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` and whose `timestamps` are session-absolute. The AI read the series' own `description` to establish the channel layout as `('tongue_x', 'tongue_y', 'tongue_likelihood')`: column 1 is the y-position used for the output, column 2 is the DeepLabCut likelihood used for visibility, and column 0 (x) is used only inside the velocity-outlier detector. Present in all 174 sessions at ~3.4 ms (≈294 Hz) sampling.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
```

```python
xy = np.asarray(data[:, :2], dtype=np.float64)
likelihood = np.asarray(data[:, 2], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 2: "all 174 files have `Camera0_side_TongueTracking` … description says it stores `('tongue_x', 'tongue_y', 'tongue_likelihood')`; timestamps are explicit and sampled every 3.4 ms." Step 4 resolves the marker-set discrepancy between sources ("Datapaper text says tongue/jaw/nose; methodpaper says jaw/paws/tongue; code aligns nose/tongue/jaw/whisker") with: "For the required output, only tongue y-position matters. Use the NWB tongue tracking stream directly and align it in the same go-cue-centered manner as the reference marker code." Step 4 also confirms "NWB tracking timestamps increment by 0.0034 s", matching the reference code's `dt=0.0034`.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four stages, all at session scope first, then per trial:

1. **Outlier rejection** — a 5-sigma rule on the frame-to-frame speed of the (x, y) tracking point; both frames straddling a flagged transition are marked bad.
2. **Visibility** — a frame counts as visible only if x and y are finite, the likelihood is finite and `>= 0.9`, and it is not a velocity outlier. Session-wide visible fraction is ~16%.
3. **Session thresholds** — the 40th and 60th quantiles of the **visible raw-frame** y values give `q40`, `q60` (see 8-c).
4. **Per-bin sample-and-hold** — for each 50 ms bin the *last* frame whose timestamp falls in that bin is taken; its y value is discretised. No averaging or interpolation is performed.

Result is written as `int64` into row 3 of the output array. Full-dataset distribution `[0.063, 0.032, 0.066, 0.839]`.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
TONGUE_VELOCITY_SIGMA = 5.0


def velocity_outlier_mask(xy: np.ndarray, dt: float) -> np.ndarray:
    """Flag tracking outliers using a 5-sigma velocity rule."""
    if len(xy) < 3:
        return np.zeros(len(xy), dtype=bool)
    diffs = np.diff(xy, axis=0)
    speed = np.linalg.norm(diffs, axis=1) / max(dt, 1e-9)
    mu = float(np.mean(speed))
    sigma = float(np.std(speed))
    thresh = mu + TONGUE_VELOCITY_SIGMA * sigma
    bad = np.zeros(len(xy), dtype=bool)
    if sigma == 0.0:
        return bad
    flagged = np.where(speed > thresh)[0]
    bad[flagged] = True
    bad[flagged + 1] = True
    return bad
```

```python
def build_tongue_session_stats(timestamps, data):
    xy = np.asarray(data[:, :2], dtype=np.float64)
    likelihood = np.asarray(data[:, 2], dtype=np.float64)
    dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0034
    outliers = velocity_outlier_mask(xy, dt)
    visible = np.isfinite(xy[:, 1]) & np.isfinite(likelihood) & (~outliers) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    visible_y = xy[visible, 1]
    if len(visible_y) == 0:
        raise ValueError("No visible tongue samples remain after filtering.")
    q40, q60 = np.quantile(visible_y, [0.4, 0.6])
    return xy, likelihood, visible, float(q40), float(q60)
```

iii. CONVERSION_NOTES Step 3 records the methodpaper's marker cleaning: "behavioral markers were tracked from video and cleaned by 5-sigma velocity-based outlier rejection" — the AI reproduces this rule rather than inventing one. Step 5 Key Decision 10: "Use a conservative tongue visibility rule: Preliminary likelihood distributions are strongly bimodal (mostly near 0 or 1), so a high threshold such as 0.9 is appropriate and robust." Key Decision 11: "Reproduce marker alignment in spirit, then downsample: Reference marker code aligns with sample-and-hold from the last frame in each interval. I will keep that rule when collapsing 3.4 ms tracking to 50 ms bins" — traced in Step 1 to `align_markers_between_lims`, which "per bin copies the last frame with timestamp in `[t-dt, t)`". Note this deliberately departs from the methodpaper's occlusion handling ("tongue position was imputed to its mean value"), because the decoder spec defines an explicit "not visible" class instead (Key Decision 9).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, per the instructions. The class edges `q40`/`q60` are the 40th and 60th **percentiles of the visible raw y frames within that session** (not of the binned values, and not global across sessions). For each bin, the last frame in the bin is examined: if that frame is not visible (or the bin contains no frame at all) the bin is class `3`; otherwise `y < q40 → 0`, `q40 <= y <= q60 → 1`, `y > q60 → 2`. The default fill for the whole 80-bin row is `3`, so anything not positively classified falls through to "not visible". Because the percentiles are taken over the same quantity that is discretised (individual frame y values), the visible classes come out near the intended 40/20/40 split: `0.063 / 0.032 / 0.066` of all bins, i.e. 39% / 20% / 41% of visible bins.

ii.
```python
TONGUE_PCT_SOURCE = "visible frames"   # q40, q60 = np.quantile(visible_y, [0.4, 0.6])
```

```python
def discretize_tongue_bins(frame_timestamps, tongue_xy, tongue_likelihood,
                           tongue_visible_mask, q40, q60, trial_edges_abs) -> np.ndarray:
    """Reference-style sample-and-hold: use the last frame within each bin."""
    out = np.full(NBINS, 3, dtype=np.int64)
    frame_idx = np.searchsorted(frame_timestamps, trial_edges_abs)
    for b in range(NBINS):
        start = int(frame_idx[b])
        end = int(frame_idx[b + 1])
        if end <= start:
            continue
        last = end - 1
        if not tongue_visible_mask[last]:
            continue
        y = float(tongue_xy[last, 1])
        if y < q40:
            out[b] = 0
        elif y <= q60:
            out[b] = 1
        else:
            out[b] = 2
    return out
```

```python
OUTPUT_VALUES = [
    ...
    ["<40th percentile", "40th-60th percentile", ">60th percentile", "not visible"],
]
```

iii. CONVERSION_NOTES Step 5 Key Decision 9: "Use visible-frame percentiles for tongue discretization: The task definition includes a dedicated 'not visible' class, so percentiles should be computed from visible tongue samples only, not from imputed/occluded values." The per-session scope and the 40/60 split are taken verbatim from the instructions' "per-session discretization". Step 7 reviewed the discretisation visually: "tongue becomes visible primarily after go cue in the example trial, and the binned tongue classes switch between visible bins (`0/1/2`) and not-visible bins (`3`) in a way consistent with the raw tracking points."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps share the NWB session-absolute clock with spikes and events, so the same absolute edge array `edges_abs = go + BIN_EDGES_REL` that bins the spikes is used to bin the frames: `np.searchsorted(frame_timestamps, edges_abs)` yields the frame index bounding each bin, so bin *k* of the tongue output covers exactly the same interval as bin *k* of the firing rates. No interpolation or offset correction. Additionally, any trial whose window is not fully inside the session's camera timestamp range is dropped upstream (see 1-e), so no trial is emitted with partially-absent video at the session edges — though within-session gaps (the camera is trial-gated and off during the inter-trial interval) still produce "not visible" bins.

ii.
```python
go = float(go_times[trial_idx])
edges_abs = go + BIN_EDGES_REL       # same edges used for bin_spike_rates

tongue_disc = discretize_tongue_bins(
    tongue_timestamps, tongue_xy, tongue_likelihood, tongue_visible_mask,
    tongue_q40, tongue_q60, edges_abs,
)
output_arr[3] = tongue_disc
```

```python
frame_idx = np.searchsorted(frame_timestamps, trial_edges_abs)
```

```python
if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]:
    continue
```

iii. CONVERSION_NOTES Step 5: align the tongue stream "in the same go-cue-centered manner as the reference marker code" (`align_markers.py`, which aligns markers to go cue on a fixed grid by carrying forward the last frame in each bin). Step 7 processing-plot review: "no obvious temporal offset between go-cue alignment and the onset of response-period tongue movement." Step 10 sanity check 2 recomputed the full `(4, 80)` output matrix — including the tongue row — independently from the raw NWB and got `np.allclose == True`, max abs difference `0.0`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases are handled:

- **Session never quality-controlled** (`classification` all `nan`): the session yields no `"good"` units and is excluded at discovery, recovering the paper's 173-session count.
- **Trials outside the ephys recording block**: excluded via `units/obs_intervals`; the mapping is strict and raises if an interval cannot be matched to a trial row.
- **Residual trials with no spikes at all**: caught after binning by `not np.any(neural)` and dropped (2 trials dataset-wide).
- **Trials with incomplete camera coverage or no tone event**: dropped.
- **Units with an empty `anno_name`** (no histology label): fall back to the insertion-target region parsed out of the JSON-encoded electrode `location`, and finally to the literal `"Unknown"`, so no unit is ever emitted with a blank region label.
- **Occluded/untracked tongue frames**: represented as the explicit class `3` rather than imputed.

The one gap: sessions that end up with fewer than 2 trials **raise `ValueError`**, which would abort the whole conversion rather than skip the session. This never fired on this dataset (minimum retained count is 7 trials).

ii.
```python
good_units = [i for i, c in enumerate(classification) if c == "good"]
if not good_units:
    raise ValueError(f"{session_id}: no classifier-good units")
```

```python
for unit_idx in good_units:
    unit_region = anno_name[unit_idx].split(",")[0].strip()
    if not unit_region:
        target = parse_target_region(electrode_locations[int(electrodes[unit_idx])])
        unit_region = target if target else "Unknown"
    region_labels.append(unit_region)
```

```python
def parse_target_region(location_json: str) -> str:
    try:
        return json.loads(location_json).get("brain_regions", "").strip()
    except Exception:
        return location_json.strip()
```

```python
if not np.any(neural):
    dropped_all_zero_neural += 1
    continue
```

```python
out = np.full(NBINS, 3, dtype=np.int64)   # default "not visible"
```

iii. CONVERSION_NOTES Step 10 "Issues Found and Resolved": "All-zero neural trials in partially covered sessions — Cause: some behavioral trials were present in the NWB trial table even though the ephys recording block did not cover them, and one additional trial survived initial filters despite having no spikes anywhere in the decoder window. Resolution: restricted trials to the raw `units.obs_intervals` coverage subset and dropped any residual all-zero-neural trial after binning. Re-check result: both sample and full verification logs are clean." Step 4 documents the unlabelled-session case and its resolution. Step 5 Key Decision 9 gives the rationale for representing occlusion as a class rather than imputing. Step 9 gives a complete trial-loss accounting (94,370 → 1,060 → 3,764 → 765 → 2 → 88,779) and concludes "no unexplained trial or unit loss remains."

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented the code with a `time.perf_counter()` per session and printed it, and measured the full run at **298.40 s** for 173 sessions (~1.7 s/session, ranging ~1.7 s to 21.5 s). Its stated bottleneck is the neural binning: "The main cost is repeated per-trial spike binning for every good unit; this scales with `n_trials * n_units`." Secondary costs it names are the per-trial tongue discretisation loop and the extra startup pass in which `discover_curated_sessions` opens all 174 files to inspect `classification`. In practice the run is also heavily I/O-bound — each session materialises the whole `spike_times` buffer (up to ~11.5 M doubles) and the whole `(n_frames, 3)` tongue array into memory with `[:]` — which the AI's notes do not call out separately, though the observed per-session spread tracks unit count as expected. Pickling the 11.7 GB result adds further time outside the per-session timers.

ii.
```python
def process_session(path: str, make_plot: bool = False) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    ...
    elapsed = time.perf_counter() - t0
    print(
        f"[session] {session_id}: kept {len(trial_keep)} trials, "
        f"{len(good_units)} good units in {elapsed:.2f}s"
    )
```

```python
spike_flat = np.asarray(units["spike_times"][:], dtype=np.float64)
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 6 "Code inefficiencies identified" lists the three items above. Step 7 estimated "Upper bound `~12.7 min` by scaling with total `good_units * recorded_trials` workload" from a 2-session sample, and Step 9 confirms "Full conversion runtime was `298.40 s` for all 173 sessions, well below the conservative `~12 min` upper-bound estimate from Step 7", so no further optimisation was pursued against the instructions' 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops remain that are vectorizable:

1. **The neural binning double loop** — `for trial: for unit: np.searchsorted(...)`, i.e. `n_trials × n_units` separate `searchsorted` calls on 81 edges each (≈ 88,779 × 400 ≈ 35 M calls dataset-wide). The trial dimension is fully collapsible: all 81×`n_trials` edges could be flattened into one array and searched once per unit, cutting the call count by ~500×. The AI did not do this; it only hoisted the ragged slicing out of the loop.
2. **`discretize_tongue_bins`** — an 80-iteration Python loop per trial, plus one `np.searchsorted` over the whole session frame-timestamp array per trial. The last-frame-per-bin rule is expressible as pure array indexing (`frame_idx[1:] - 1`, masked where the bin is empty), and the trial dimension could likewise be flattened.
3. **`derive_choice_per_trial`** — loops over every trial and builds two full boolean masks over the *entire* session lick arrays each iteration, making it `O(n_trials × n_licks)`. Two `np.searchsorted` calls on the sorted lick trains would make it `O(n_trials log n_licks)`.

The AI identified (1) and (2) but not (3), and vectorized none of them, on the grounds that the measured runtime was already inside budget.

ii.
```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    ...
    neural = np.empty((len(good_units), NBINS), dtype=np.float32)
    for unit_row, spikes_abs in enumerate(unit_spike_times):
        neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

```python
for b in range(NBINS):
    start = int(frame_idx[b])
    end = int(frame_idx[b + 1])
    if end <= start:
        continue
```

```python
for i, go in enumerate(go_times):
    end = go + RESPONSE_WINDOW
    left = left_lick_times[(left_lick_times >= go) & (left_lick_times < end)]
    right = right_lick_times[(right_lick_times >= go) & (right_lick_times < end)]
```

iii. CONVERSION_NOTES Step 6: "The main cost is repeated per-trial spike binning for every good unit; this scales with `n_trials * n_units`. Tongue discretization currently loops over bins per trial; cost is modest relative to spike binning." Speed-ups it did apply: "Pre-sliced each good unit's spike train once per session before looping over kept trials. Used `np.searchsorted` on sorted spike times / frame timestamps rather than scanning samples in Python loops." Step 7/9 justify stopping there: the measured 298 s is well under the instructions' 15-minute threshold, so no further optimisation was required.

## 10-c. What processing does the code repeat multiple times?

i. Four repetitions:

1. **Every NWB file is opened twice.** `discover_curated_sessions` opens all 174 files and reads the full `units/classification` string dataset only to test membership of `"good"`; `process_session` then reopens each surviving file and reads the identical dataset again. The classification vector is thus decoded twice per session, and 174 extra file opens are incurred.
2. **`edges_abs = go + BIN_EDGES_REL` is computed twice per trial** — once in the candidate-filter loop (for the camera-coverage test) and again in the main conversion loop.
3. **`np.searchsorted` over the full session camera-timestamp array is redone per trial** inside `discretize_tongue_bins`, rather than once for all trials.
4. **`choice`, `outcome` and `early` are computed for every trial in the table**, including the ~6% later discarded by the filters.

The AI documented (1) and nothing else.

ii.
```python
def discover_curated_sessions() -> list[str]:
    files = list_session_files()
    curated = [path for path in files if session_has_good_units(path)]   # opens every file
    return curated
```

```python
with h5py.File(path, "r") as h5:                 # ... and opens it again here
    units = h5["units"]
    classification = decode_vector(units["classification"][:])
```

```python
        go = float(go_times[i])
        edges_abs = go + BIN_EDGES_REL           # candidate loop
        if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]:
            continue
...
    for kept_idx, trial_idx in enumerate(candidate_trial_idx):
        go = float(go_times[trial_idx])
        edges_abs = go + BIN_EDGES_REL           # recomputed
```

```python
choice = derive_choice_per_trial(go_times, left_lick_times, right_lick_times)   # all trials
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int64)
early = np.array([early_map[x] for x in early_text], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 6 "Code inefficiencies identified": "Session discovery opens every NWB file once to inspect `classification`, which is acceptable but still non-zero startup overhead." The AI's justification for keeping it is that it lets the session count (173) be reported before any heavy processing starts, and that the cost is small relative to the full read. Step 6 "Code speedups added" notes the one redundancy it did remove: "Pre-sliced each good unit's spike train once per session before looping over kept trials … avoided redundant file I/O during processing."

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI's notes do not address this question, and the code carries several pieces of work whose results are never written to the pickle:

1. **`session_summary` retains four full-session tracking arrays** — `tongue_timestamps`, `tongue_xy`, `tongue_likelihood`, `tongue_visible` — for *every* session, and all 173 summaries are held in a list for the whole run. Only four scalar fields of each summary (`session_id`, `subject`, `n_good_units`, `n_kept_trials`) ever reach `metadata['session_info']`. At ~22 MB per session this is roughly **3–4 GB of live memory retained for nothing** on top of the 11.7 GB dataset; the arrays are genuinely needed only for the ≤2 sessions plotted under `--show-processing`.
2. **`sample_onsets_abs`** is accumulated in the candidate loop and never read.
3. **`region_counts`** is computed with `Counter` and stored in the summary, but `build_dataset` never uses it.
4. **`trial_instruction`** is decoded for every session but never used (choice is derived from licks instead).
5. **`electrode_locations`** — the full electrode `location` table is decoded per session on every run, but is only consulted in the rare fallback where `anno_name` is empty.
6. **`tongue_likelihood`** is passed into `discretize_tongue_bins` as a parameter and never referenced in the body (visibility is already baked into `tongue_visible_mask`).
7. **`metadata['bin_centers_rel_go_s']`** is stored in the pickle but is not consumed by `train_decoder.py`; it is derivable from `off_start`/`off_end`/`time_bin_size`.

None of these affects correctness; (1) is the only one with a material cost.

ii.
```python
session_summary = {
    ...
    "region_counts": dict(region_counts),
    "tongue_timestamps": tongue_timestamps,
    "tongue_xy": tongue_xy,
    "tongue_likelihood": tongue_likelihood,
    "tongue_visible": tongue_visible_mask,
}
```

```python
session_summaries.append(session_summary)     # held for all 173 sessions
...
"session_info": [
    {
        "session_id": summary["session_id"],
        "subject": summary["subject"],
        "n_good_units": summary["n_good_units"],
        "n_kept_trials": summary["n_kept_trials"],
    }
    for summary in session_summaries
],
```

```python
sample_onsets_abs = []
...
sample_onsets_abs.append(sample_start)        # never read
```

```python
trial_instruction = decode_vector(trials["trial_instruction"][:])   # never used
```

```python
def discretize_tongue_bins(
    frame_timestamps, tongue_xy, tongue_likelihood,   # tongue_likelihood unused in body
    tongue_visible_mask, q40, q60, trial_edges_abs,
) -> np.ndarray:
```

iii. No justification is given — CONVERSION_NOTES Step 6 lists inefficiencies only in terms of runtime (spike binning, tongue loop, session discovery) and the Step 13 cleanup pass moved investigation scripts into `/app/cache/` without revisiting the conversion script's dead weight. The closest the notes come is the Step 10 remark on region labels, which explains a *deliberate* retention of information rather than a discard: "conversion keeps fine-grained `anno_name` histology labels as `brain_regions` rather than collapsing immediately to the paper's coarser analysis families … the decoder does not consume this field; preserving finer labels retains information rather than discarding it."
