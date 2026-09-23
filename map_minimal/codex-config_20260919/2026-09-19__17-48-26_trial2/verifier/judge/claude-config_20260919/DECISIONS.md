# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI 000363 NWB release, one file per session under `data/sub-<subject_id>/`. The AI enumerates every session with a single sorted glob over that layout and opens each file **directly with `h5py`** (not `pynwb`), reading the HDF5 groups it needs: `units` (`classification`, `anno_name`, `spike_times`, `spike_times_index`, `is_good_trials`), `intervals/trials`, `acquisition/BehavioralEvents/*`, `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, `identifier`, and `general/subject/subject_id`. Each file is opened once, converted in one pass, and closed. All 174 files are visited; 173 produce output (see 2-c).

ii.
```python
paths = sorted(glob.glob(os.path.join(data_dir, "sub-*", "*.nwb")))
if not paths:
    raise FileNotFoundError(f"no NWB files under {data_dir}")

for path in paths:
    print(f"Converting {path}", flush=True)
    session = convert_session(path)
```
```python
def convert_session(path: str) -> dict | None:
    with h5py.File(path, "r") as nwb:
        units = nwb["units"]
        ...
        trials = nwb["intervals/trials"]
        go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
```

iii. From the trajectory: the AI first inventoried the release (`du -sh data; find data -name '*.nwb' | wc -l` → 174 files) and confirmed the one-file-per-session layout, then wrote a survey pass over all 174 files to count units, trials, subjects and video availability before committing to a converter. It chose raw `h5py` over `pynwb` after hitting `pynwb`'s slower ragged-column access; it reads the `*_index` offset arrays itself to slice the ragged `spike_times` buffer. Sorting the glob makes session order deterministic. Its summary message states the source is "173 NWB sessions" after the one uncurated file is removed, matching the data paper.

## 1-b. How are the data split into subjects?

i. Each file's animal is read from the NWB field `general/subject/subject_id` (a numeric string such as `'440956'`). `subjects` is the sorted set of unique ids across converted sessions and `subject_idx` is each session's index into that list. This yields 28 subjects with 3–10 sessions each.

ii.
```python
subject = _scalar_string(nwb["general/subject/subject_id"])
...
subjects = sorted({session["subject"] for session in converted_sessions})
subject_to_index = {subject: i for i, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array(
    [subject_to_index[session["subject"]] for session in converted_sessions],
    dtype=np.int16,
),
```

iii. The AI uses the canonical NWB subject field rather than parsing the `sub-*` folder name (which is derived from it anyway), so no separate grouping step is needed. Its verification run reports 28 subjects, which it cross-checked against the dandiset summary.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no splitting or grouping is performed. Session identity is carried as `nwb['identifier']` (e.g. `SC015_20190207_120657_s1`) and recorded per session in `metadata['session_info']` together with the source file path, source/ephys/retained trial counts, per-filter exclusion counts, unit count and the session's tongue percentiles. Session order in the output is the sorted file order; skipped files are listed in `metadata['skipped_source_files']`.

ii.
```python
identifier = _scalar_string(nwb["identifier"])
...
"session_info": {
    "identifier": identifier,
    "source_file": os.path.relpath(path, Path(__file__).parent),
    "source_trial_count": int(len(keep)),
    "ephys_trial_count": int(n_ephys_trials),
    "retained_trial_count": int(np.sum(keep)),
    "excluded_no_ephys_or_unit_invalid_trials": int(np.sum(~neural_valid)),
    "excluded_empty_neural_trials": int(np.sum(empty_neural_trials)),
    "excluded_auto_or_free_water_trials": int(np.sum(neural_valid & assistance)),
    "classifier_good_units": int(len(good_indices)),
    ...
}
```

iii. The AI treats the file boundary as the session boundary because the dandiset stores exactly one session per file, so nothing has to be inferred. It deliberately records per-session provenance and exclusion counts in the metadata so that the curation can be audited; its own reconciliation step read these counts back to confirm totals.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial. The AI asserts that the number of `go_start_times` events equals the number of trial rows, and uses that one-to-one correspondence to index every per-trial quantity. All per-trial arrays (`trial_instruction`, `outcome`, `early_lick`, `photostim_onset`, `photostim_duration`, `start_time`, `auto_water`, `free_water`) are read as whole columns and masked by the same `keep` vector.

ii.
```python
trials = nwb["intervals/trials"]
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
if len(go_times) != len(trials["id"]):
    raise ValueError(f"{path}: go cue and trial counts differ")
```

iii. The AI explicitly examined the event streams (trajectory command printing `presample/sample/delay/go` counts per session) and found that `sample_start_times` and `delay_start_times` have *more* entries than trials because an early lick replays those epochs, while `go_start_times` is exactly one per trial. It therefore anchors trials on the trials table plus the go cue, and raises rather than silently mis-aligning if that invariant ever fails.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, applied per session, with a minimum of 2 surviving trials:

1. **Electrophysiology validity** — `units/is_good_trials` (a `(n_units, n_ephys_trials)` boolean table) is read for the retained good units; a trial is kept only if **every** retained unit is valid on it. Trials beyond the ephys trial columns (8 sessions where the behaviour extends past the recording) are marked invalid.
2. **Assistance trials** — trials with `auto_water != 0` or `free_water != 0` are excluded, following the reference repository's `get_regular_trial_mask`.
3. **Empty neural trials** — after binning, any trial where the whole population has zero spikes in the 4 s window is dropped.

Early-lick, `ignore` (no-response) and photostimulation trials are **deliberately retained** even though the paper and the reference code exclude them, because they are requested decoder inputs/outputs. Result: 88,943 trials over 173 sessions.

ii.
```python
unit_trial_validity = units["is_good_trials"][:][good_indices]
n_ephys_trials = unit_trial_validity.shape[1]
if n_ephys_trials > len(go_times):
    raise ValueError(f"{path}: more ephys trial columns than behavior trials")
neural_valid = np.zeros(len(go_times), dtype=bool)
neural_valid[:n_ephys_trials] = np.all(unit_trial_validity, axis=0)

# Reference-code assistance-trial exclusions.
assistance = (trials["auto_water"][:] != 0) | (trials["free_water"][:] != 0)
keep = neural_valid & ~assistance
kept_indices = np.flatnonzero(keep)
if len(kept_indices) < 2:
    return None

firing_rates_all = _bin_good_units(units, good_indices, go_times)
neural_nonzero = np.any(firing_rates_all != 0, axis=(1, 2))
empty_neural_trials = neural_valid & ~neural_nonzero
keep &= neural_nonzero
```

iii. The AI's stated reasoning (module docstring plus trajectory messages): "Trials with automatic or free water are training/assistance trials and are excluded, as in the reference analysis. Ordinarily the paper also excludes early-lick, no-response, and photostimulation trials. Those three exclusions are deliberately not made here because the requested decoder variables explicitly require those trials." The `is_good_trials` filter was added mid-run in response to a verifier warning: the first artefact produced blocks of all-zero trials, and the AI traced this to behaviour extending beyond ephys ("eight behavioral files extend beyond electrophysiology, and four more contain unit-specific invalid recording blocks"), concluding that requiring simultaneous validity for every retained unit "prevents missing recordings from being encoded as silence" while preserving all 69,453 good units. The trailing all-zero check was kept as a safety net for recordings that end inside a nominally valid trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From the ragged `units/spike_times` buffer together with its offset index `units/spike_times_index`, restricted to the quality-controlled units (2-c), and from `acquisition/BehavioralEvents/go_start_times/timestamps`, which places the analysis window. Unit brain region comes from `units/anno_name`.

ii.
```python
all_spikes = units["spike_times"][:]
spike_ends = units["spike_times_index"][:].astype(np.int64)
spike_starts = np.r_[0, spike_ends[:-1]]
window_starts = go_times + OFF_START
window_ends = go_times + OFF_END
...
spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
```

iii. Sorted spike times are the only neural representation in the file. The AI reads the whole buffer once per session and slices it with the index offsets rather than issuing one HDF5 read per unit, which it found much faster than `pynwb`'s per-unit access.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit, every spike is assigned to a trial by `searchsorted` against the trial window starts, checked against the window end, then assigned to a 50 ms bin by its offset from the window start; `np.bincount` accumulates counts over the flattened (trial × bin) grid and the counts are divided by the bin width. No smoothing, normalisation, or baseline subtraction. Output dtype is `float32`, per-trial shape `(n_neurons, 80)`.

ii.
```python
rates = np.empty((n_trials, len(good_indices), N_TIME), dtype=np.float32)
...
# Windows do not overlap in this task, so each spike has at most one trial.
for out_unit, unit_index in enumerate(good_indices):
    spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
    trial_index = np.searchsorted(window_starts, spikes, side="right") - 1
    valid = trial_index >= 0
    trial_index = trial_index[valid]; spikes = spikes[valid]
    valid = spikes < window_ends[trial_index]
    trial_index = trial_index[valid]; spikes = spikes[valid]
    time_index = np.floor((spikes - window_starts[trial_index]) / BIN_S + 1e-12).astype(np.int64)
    valid = (time_index >= 0) & (time_index < N_TIME)
    flat_index = trial_index[valid] * N_TIME + time_index[valid]
    counts = np.bincount(flat_index, minlength=n_trials * N_TIME)
    rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_S
```

iii. The AI read `sliding_histogram(..., rate=True)` in `preprocessing_DJ_2022Aug.py`, which returns `binSpikes / bin_width`, and reproduced that definition (spikes/s, half-open bins). It notes in a code comment that the go-cue-to-go-cue interval always exceeds the 4 s window, so a spike can belong to at most one trial and the scatter-add is unambiguous (verified independently: the global minimum go-to-go interval in the release is 4.58 s). The `+1e-12` guard protects the floor against floating-point edge cases at bin boundaries.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` (the spike-sorting QC classifier of Chen, Colonell, Li & Svoboda 2023) **and** a non-empty `anno_name` CCF annotation are retained. No thresholds are applied to individual quality metrics and `unit_quality` (the older 'good'/'multi' label) is not used. One session (`sub-440958_ses-20190216T162508`) stores `classification` as a float NaN column rather than strings — it is detected by dtype and skipped entirely. Result: 69,453 units across 173 sessions (mean 401/session, range 90–923).

ii.
```python
classification = units["classification"]
# One released session has NaNs here instead of classifier predictions.
if classification.dtype.kind not in "OSU":
    return None
class_values = classification.asstr()[:]
annotations = units["anno_name"].asstr()[:]
annotation_present = np.array([bool(x.strip()) for x in annotations])
good = (class_values == "good") & annotation_present
good_indices = np.flatnonzero(good)
if len(good_indices) == 0:
    return None
```

iii. From the trajectory: "174 files are present, but one session lacks the paper's classifier QC labels; the remaining 173 contain 69,453 classifier-approved, histologically localized units, matching the paper's stated session count." The AI ran a survey over all files that produced exactly these numbers before writing the converter, and cross-checked its per-region totals against `methods.txt` (its counts of 7,664 striatum / 12,808 thalamus / 7,495 midbrain / 2,928 medulla reproduce the published numbers exactly). The annotation requirement is defended as needed for `brain_region_idx`; in practice it excludes nothing (good and good+annotated are both 69,453).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go cue onset**, taken from `go_start_times`. Spike times, event times and camera timestamps all live on the same session-absolute clock, so the alignment consists of placing the fixed window `[go - 2.5 s, go + 1.5 s)` on each trial's go cue; spikes are binned by their offset from that trial's window start. No resampling, interpolation, or per-stream offset correction.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
...
window_starts = go_times + OFF_START
window_ends = go_times + OFF_END
...
time_index = np.floor((spikes - window_starts[trial_index]) / BIN_S + 1e-12).astype(np.int64)
```
```python
"temporal_alignment_event": "auditory Go cue onset",
"off_start": OFF_START,
"off_end": OFF_END,
```

iii. The instructions specify go-cue alignment and the −2.5/+1.5 s window; the reference repository likewise stores go-cue-relative spike times ("go cue time is 0"). The AI verified that `go_start_times` has exactly one entry per trial, making the anchor unambiguous, and reuses the same absolute-time grid (`kept_go[:, None] + BIN_CENTERS[None, :]`) for the inputs and the tongue output so all streams share one bin grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, non-overlapping, 80 bins per trial spanning −2.5 s to +1.5 s relative to the go cue. The grid is defined once at module level as 81 edges / 80 centres and reused for every trial and session, so every trial has exactly 80 timepoints. There is no rebinning: rates are computed directly from raw spike times at the target resolution, and the bin centres are written into the metadata.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_S = 0.050
N_TIME = int(round((OFF_END - OFF_START) / BIN_S))          # 80
BIN_EDGES = OFF_START + np.arange(N_TIME + 1) * BIN_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```
```python
"time_bin_size": 50.0,
"neural_measurement": "firing rate (spikes/s) in non-overlapping 50-ms bins",
"time_bin_centers_seconds": BIN_CENTERS.tolist(),
```

iii. The 50 ms width and the window come straight from the Decoder Task section of the instructions. Binning raw spike times once at the target resolution avoids the information loss of rebinning, and defining a single module-level grid guarantees the constant `T = 80` that the target format requires (the verifier reports `T_min = T_max = 80`).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/presample_stop_times/timestamps` — the end of the presample epoch — which the AI treats as the tone (sample-epoch) onset, together with the go cue that defines the bin grid. It is taken per trial (one entry per trial) and masked by `keep`.

ii.
```python
# presample_stop is the first tone/sample onset and, unlike sample_start,
# has exactly one entry per trial even when an early lick replays epochs.
tone_onsets = nwb[
    "acquisition/BehavioralEvents/presample_stop_times/timestamps"
][:][keep]
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```
```python
"tone_onset_definition": "initial sample/tone onset (presample stop event)",
```

iii. The AI's stated justification: "I've also made early-lick replay handling explicit: 'tone onset' is the initial sample onset, the one event that remains one-to-one with trials." It inspected `sample_start_times` in one session (`sub-440956_ses-20190208T133600`) and found 521 entries for 480 trials because of early-lick replays, whereas `presample_stop_times` has exactly 480; in that particular session `presample_stop` happens to coincide numerically with the first `sample_start`, and the AI generalised from it. (Verified against the release: this coincidence holds in only 58 of 174 sessions. In the other 116 sessions `presample_stop` precedes the actual sample onset by ~0.72–0.79 s, and 68% of all trials are affected; in 127 sessions some trials have `go − presample_stop` as short as 0.95 s, i.e. the marker falls *after* the tone.)

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute time of every bin centre (`go + centre`) minus the trial's tone time, giving a continuous, monotonically increasing ramp of 80 values per trial in seconds. It is stored as row 0 of the `(2, 80)` `float32` input array. No clipping, normalisation or binarisation.

ii.
```python
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
...
time_from_tone = absolute_bin_times - tone_onsets[:, None]
...
session_inputs.append(
    np.stack([time_from_tone[trial_index], photostim[trial_index]], axis=0).astype(np.float32)
)
```
```python
"input_names": ["time from tone onset (s)", "photostimulation on"],
```

iii. The instructions ask for a continuous, time-varying "time from tone onset in seconds", so the AI simply expresses the bin grid in tone-relative seconds; because the grid is go-cue-locked, the only per-trial variation is the tone-to-go interval. The resulting values run from −1.52 s to 12.70 s across the dataset — the long tail comes from early-lick trials, where the retained marker belongs to the first (aborted) sample epoch rather than the replayed one immediately preceding the go cue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same grid as the firing rates: `absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]`, i.e. the centres of the same 80 go-cue-locked 50 ms bins used for spike binning, on the same session-absolute clock. Bin *k* of the input therefore covers the same interval as bin *k* of the neural array, and the input array is built from the same `keep` mask, so trial *t* of `input` is trial *t* of `neural`.

ii.
```python
firing_rates = firing_rates_all[keep]
kept_go = go_times[keep]
absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]
...
time_from_tone = absolute_bin_times - tone_onsets[:, None]
```

iii. All NWB streams share one global clock, so no interpolation or offset correction is required; the AI derives every time-varying stream from the single `absolute_bin_times` matrix specifically so that the alignment cannot drift between streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials-table string columns `photostim_onset` and `photostim_duration` (both measured from trial start, with the sentinel `'N/A'` on unstimulated trials) plus `trials/start_time` to convert them to absolute time.

ii.
```python
def _numeric_or_nan(values: np.ndarray) -> np.ndarray:
    """Parse NWB string columns containing numbers and the sentinel ``N/A``."""
    return np.array(
        [np.nan if value == "N/A" else float(value) for value in values],
        dtype=np.float64,
    )
...
onset = _numeric_or_nan(trials["photostim_onset"].asstr()[:])[keep]
duration = _numeric_or_nan(trials["photostim_duration"].asstr()[:])[keep]
trial_starts = trials["start_time"][:][keep]
```

iii. The AI enumerated every trials-table column and its unique values in the trajectory, which is where it found that the photostim columns are stored as strings with `'N/A'` for control trials; it converts them to floats with NaN sentinels. It preferred the trials table over the `photostim_start_times`/`photostim_stop_times` event streams because the table gives a per-trial onset/duration pair that is unambiguous to place on the trial's time axis.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Onset and duration are converted to an absolute `[start, end)` interval per trial, and a bin is 1 if its centre falls inside that interval, 0 otherwise — i.e. a binary time series over the 80 bins, not a per-trial flag. Unstimulated trials keep NaN bounds, and NaN comparisons are False, so all their bins are 0 without a special case. The result is stored as row 1 of the `float32` input array (values 0.0/1.0).

ii.
```python
stimulation_start = trial_starts + onset
stimulation_end = stimulation_start + duration
photostim = (
    (absolute_bin_times >= stimulation_start[:, None])
    & (absolute_bin_times < stimulation_end[:, None])
)
```

iii. The instructions require photostimulation to be given as "whether photostimulation is on at every time point (discrete, time-varying)". The AI used the paper's description (ALM photoinhibition during the last 0.5 s of the delay, always ending before the go cue) as a sanity check that the stimulation falls inside the −2.5 s window; the verifier reports the input range as exactly [0, 1].

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Through the same `absolute_bin_times` matrix used for the firing rates: the stimulation window is converted from trial-relative to absolute seconds (`trial_start + onset`) and compared against the absolute bin centres, so no separate alignment step is needed and the bins are identical to the neural bins.

ii.
```python
trial_starts = trials["start_time"][:][keep]
stimulation_start = trial_starts + onset
stimulation_end = stimulation_start + duration
photostim = (
    (absolute_bin_times >= stimulation_start[:, None])
    & (absolute_bin_times < stimulation_end[:, None])
)
```

iii. The onsets are stored relative to trial start while the bins are go-cue-locked, so a conversion is unavoidable; doing it by moving both quantities onto the shared absolute clock (rather than subtracting the go cue) keeps a single representation for all streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from the two trials-table columns `trial_instruction` (`'left'`/`'right'`) and `outcome` (`'hit'`/`'miss'`/`'ignore'`): a hit means the animal licked the instructed side, a miss the opposite side, and `ignore` means no lick. The raw `left_lick_times`/`right_lick_times` event streams are deliberately not used.

ii.
```python
instruction = trials["trial_instruction"].asstr()[:][keep]
outcome_text = trials["outcome"].asstr()[:][keep]
...
# Outcome and instruction encode the report robustly even when the raw
# lick-event stream contains a brief lick at the other port.  Ignore means
# no reported lick; miss means the report was opposite the instruction.
```

iii. The code comment gives the justification: the behavioural report recorded by the rig (instruction × outcome) is more robust than re-deriving direction from the raw lick stream, which can contain incidental licks at the other port during the response window. Instruction and outcome jointly determine the licked side exactly, so nothing is lost.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0` left, `1` right, `2` no lick. Trials with `outcome == 'ignore'` get code 2; otherwise the reported side is left when (hit ∧ instruction left) or (miss ∧ instruction right), else right. The per-trial value is broadcast across all 80 bins into row 0 of the `(4, 80)` `int8` output array, and `output_values[0] = ['left', 'right', 'no lick']`. Distribution over the dataset: 42.9% left, 42.2% right, 14.9% no lick.

ii.
```python
choice = np.full(len(kept_indices), 2, dtype=np.int8)  # no lick
responded = outcome_text != "ignore"
reported_left = ((outcome_text == "hit") & (instruction == "left")) | (
    (outcome_text == "miss") & (instruction == "right")
)
choice[responded & reported_left] = 0
choice[responded & ~reported_left] = 1
...
np.full(N_TIME, choice[trial_index], dtype=np.int8),
```

iii. The instructions list the three categories (left, right, no lick) and require outputs to be categorical and, where possible, time-varying; the AI repeats the per-trial label across bins so all four outputs share one `(n_output, n_timepoints)` array of the shape the format requires.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome_text = trials["outcome"].asstr()[:][keep]
```

iii. No derivation is needed — the AI's column survey confirmed the column's unique values are exactly the three categories named in the instructions.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped through a fixed dictionary to `0` ignore, `1` miss, `2` hit, then broadcast across all 80 bins into row 1 of the output array, with `output_values[1] = ['ignore', 'miss', 'hit']`. Distribution: 14.9% ignore, 16.6% miss, 68.5% hit.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int8)
...
np.full(N_TIME, outcome[trial_index], dtype=np.int8),
```

iii. The code assignment follows the order given in the instructions ("ignore, miss, hit"). An unseen string would raise a `KeyError` rather than being silently mis-coded. The 68.5% hit rate is consistent with the paper's reported ~84% correct rate once `ignore` and photostimulation trials — excluded from the paper's performance figure but kept here — are included.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the trials-table `early_lick` column, whose values are the strings `'no early'` and `'early'`.

ii.
```python
early_text = trials["early_lick"].asstr()[:][keep]
```

iii. The rig flags early licking explicitly, so no derivation from the lick streams is needed. The AI separately confirmed (trajectory) that early-lick trials are the ones with replayed sample/delay epochs, i.e. the flagged event occurs before the go cue and therefore inside the −2.5 s window.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Coded `0` no, `1` yes by testing equality with `'early'`, then broadcast across all 80 bins into row 2 of the output array, with `output_values[2] = ['no', 'yes']`. Distribution: 88.3% no, 11.7% yes.

ii.
```python
early = (early_text == "early").astype(np.int8)
...
np.full(N_TIME, early[trial_index], dtype=np.int8),
```

iii. The code assignment follows the instructions (no = 0, yes = 1). Early-lick trials are retained rather than excluded — contrary to the data paper's analysis convention — precisely because this is a requested decoder output; the AI states this explicitly in the module docstring.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = (`tongue_x`, `tongue_y`, `tongue_likelihood`) at ~294 Hz with matching `timestamps`. The AI reads columns 1:3 — y-position and the DeepLabCut likelihood — and ignores x.

ii.
```python
tracking = behavior["Camera0_side_TongueTracking"]
tracking_data = tracking["data"][:, 1:3]  # y, DeepLabCut likelihood
tracking_times = tracking["timestamps"][:]
y_all = tracking_data[:, 0]
likelihood_all = tracking_data[:, 1]
```

iii. This is the only tongue measurement in the release; the AI verified in its survey pass that all 174 files contain the side-camera tongue series (`no_video` empty), and read the column layout from the series' own `description` attribute (`('tongue_x','tongue_y','tongue_likelihood')`). The side camera is the view in which tongue protrusion appears as y-displacement.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two steps. (1) **Visibility**: a frame counts only if `y` and `likelihood` are finite and `likelihood >= 0.90`; the session's 40th and 60th percentiles are computed over the y-values of *all visible frames in the whole session*. (2) **Per-bin sampling**: for each bin, the single most recent frame at or before the bin centre is taken (`searchsorted(..., 'right') - 1`); that one frame's y and likelihood are then thresholded. There is **no averaging within a bin** — one of the ~15 frames per 50 ms bin is used as the bin's value. The per-session percentiles are recorded in `session_info`.

ii.
```python
TONGUE_LIKELIHOOD_CUTOFF = 0.90
...
visible_all = (
    np.isfinite(y_all) & np.isfinite(likelihood_all)
    & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
)
if not np.any(visible_all):
    raise ValueError("session has no visible tongue samples")
percentile_40, percentile_60 = np.percentile(y_all[visible_all], [40, 60])

frame_index = np.searchsorted(tracking_times, absolute_bin_times.ravel(), side="right") - 1
in_range = (frame_index >= 0) & (frame_index < len(tracking_times))
safe_index = np.clip(frame_index, 0, len(tracking_times) - 1)
sampled = tracking_data[safe_index]
```

iii. The AI searched the papers and the reference repository for a tongue-visibility convention and found none (the repo's marker code never uses the DLC likelihood), so it set its own threshold. 0.90 is defensible because the likelihood is effectively binary (~89% of frames below 0.01, ~10.5% at or above 0.99), so 0.5 and 0.9 select almost identical frame sets. The point-sampling choice, however, is never justified in the trajectory or in a code comment beyond the docstring line "Sample preceding video frames and discretize visible tongue y per session". Its consequence is measurable: sampling one frame per bin marks 12.6% of bins as visible in an example session where at least one visible frame exists in 20.6% of bins, and the released dataset reports 84.1% of bins in the `not visible` class. The percentile base is at least internally consistent with the sampling scheme (percentiles are taken over the same raw-frame quantity that is discretised), and the visible classes come out close to the intended 40/20/40 split (39%/20%/41% of visible bins).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes: `0` below the session's 40th percentile, `1` between the 40th and 60th percentiles inclusive, `2` above the 60th percentile, `3` not visible (the default, used whenever the sampled frame is missing or its likelihood is below the cutoff). Percentiles are per session, as the instructions require. Stored as row 3 of the `int8` output array with `output_values[3] = ['below 40th percentile', '40th to 60th percentile', 'above 60th percentile', 'not visible']`. Dataset distribution: 6.2% / 3.2% / 6.5% / 84.1%.

ii.
```python
result = np.full(len(frame_index), 3, dtype=np.int8)
result[visible & (y < percentile_40)] = 0
result[visible & (y >= percentile_40) & (y <= percentile_60)] = 1
result[visible & (y > percentile_60)] = 2
```
```python
["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"],
```

iii. The three visible classes and the per-session percentile scope are taken verbatim from the Decoder Task specification. A fourth explicit class is added rather than imputing a position, because the tongue is retracted in most frames and the tracker still emits a coordinate then; representing "no tongue" as a category avoids fabricating a position. The bin boundaries are mutually exclusive and exhaustive, and the default-to-3 initialisation guarantees every bin gets a label.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as the spikes and events, so each bin takes the most recent frame at or before its own centre, evaluated on the same `absolute_bin_times` matrix used for the firing rates and inputs. The result is reshaped back to `(n_trials, 80)` so bin *k* of the tongue output covers the same interval as bin *k* of the neural array. Bins whose centre precedes the first camera frame (`frame_index < 0`) fall back to `not visible`.

ii.
```python
frame_index = np.searchsorted(tracking_times, absolute_bin_times.ravel(), side="right") - 1
in_range = (frame_index >= 0) & (frame_index < len(tracking_times))
...
visible = (in_range & np.isfinite(y) & np.isfinite(likelihood)
           & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF))
...
return (result.reshape(absolute_bin_times.shape), ...)
```

iii. A shared clock means no interpolation or offset correction is needed; nearest-preceding-sample is a causal choice that cannot leak future video into a bin. The AI does not bound how *old* the carried-forward frame may be, which matters because the video is trial-gated and pauses during the inter-trial interval — measured on the release, ~0.3% of bins sample a frame more than 50 ms old (up to ~1.5 s), but only ~0.01% of bins are labelled visible from such a stale frame, so the practical effect is negligible.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled case by case, with hard failures where an invariant would be violated:

- **Session never quality-controlled**: `units/classification` is a float NaN column; detected by dtype and the session is skipped (1 of 174).
- **Units without a CCF annotation**: excluded by the `annotation_present` test (in practice a no-op).
- **Behaviour extending past the ephys recording / per-unit invalid blocks**: excluded via `is_good_trials`; a trailing all-zero trial is excluded by the `neural_nonzero` check.
- **Assistance trials**: `auto_water`/`free_water` excluded.
- **`'N/A'` photostim strings**: parsed to NaN, and NaN comparisons yield 0 for every bin of an unstimulated trial.
- **Untracked tongue frames**: non-finite values or likelihood below cutoff become the explicit `not visible` class; bins before the first camera frame likewise.
- **Sessions with fewer than 2 surviving trials, or no visible tongue frames at all**: dropped / raise.
- **Structural inconsistencies** (go-cue count ≠ trial count, more ephys trial columns than behavioural trials) raise `ValueError` rather than being silently patched.

ii.
```python
if classification.dtype.kind not in "OSU":
    return None
...
if len(go_times) != len(trials["id"]):
    raise ValueError(f"{path}: go cue and trial counts differ")
if n_ephys_trials > len(go_times):
    raise ValueError(f"{path}: more ephys trial columns than behavior trials")
...
if not np.any(visible_all):
    raise ValueError("session has no visible tongue samples")
...
if len(kept_indices) < 2:
    return None
```

iii. The AI draws an explicit distinction: where *nothing was recorded* the trial or session is removed, because emitting it would encode missing data as genuine silence (its own words: "prevents missing recordings from being encoded as silence"); where a measurement legitimately has no value, as with a retracted tongue, it is given an explicit category rather than imputed. Structural assumptions that the rest of the code relies on are asserted so that a violation fails loudly.

## 10-a. What are the most time-consuming steps of the code?

i. Conversion itself is fast: measured on this machine, a 459-unit session takes 0.25 s and the largest (923 units) 1.23 s, so the 174-file pass is a few minutes, dominated by (1) HDF5 reads — the full `spike_times` buffer (up to ~11.5 M doubles) and the `(n_frames, 3)` tongue array (~680 k rows) per session, (2) the per-unit loop in `_bin_good_units`, which runs one `searchsorted` + one `bincount` over `n_trials × 80` per unit, and (3) `is_good_trials`, read as a full `(n_units, n_trials)` boolean table. Writing the 11.6 GB pickle at the end is a further sizeable, unavoidable cost. Everything downstream of that (the decoder's per-session SVD initialisation) dwarfs conversion but is not this code.

ii.
```python
all_spikes = units["spike_times"][:]
...
for out_unit, unit_index in enumerate(good_indices):
    ...
    counts = np.bincount(flat_index, minlength=n_trials * N_TIME)
    rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_S
```
```python
with open(output_path, "wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI chose `h5py` over `pynwb` and a single bulk read of the ragged spike buffer specifically to keep I/O down, and it uses a scatter-add (`bincount`) rather than per-bin counting so each unit costs one pass over its spikes. It never profiled the converter explicitly; its monitoring effort in the trajectory went to the decoder run, which it (correctly) identified as the expensive stage.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two remain, neither significant:

1. `_bin_good_units` loops over units. The trial and time dimensions are already vectorised inside the loop, and the loop cannot be collapsed further without concatenating the ragged per-unit spike arrays and carrying a unit offset into the `bincount` index — possible, but it would trade clarity for little gain.
2. The per-trial Python loop in `convert_session` that materialises the output lists, calling `np.stack`/`np.vstack`/`np.full` three times per trial. This is pure repackaging of arrays that already exist in `(n_trials, ...)` form and could be done with slicing alone.

Minor: `_numeric_or_nan` and `annotation_present` use Python list comprehensions over whole columns.

ii.
```python
for out_unit, unit_index in enumerate(good_indices):
    spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
```
```python
for trial_index in range(len(kept_indices)):
    session_neural.append(firing_rates[trial_index])
    session_inputs.append(np.stack([time_from_tone[trial_index], photostim[trial_index]], axis=0).astype(np.float32))
    session_outputs.append(np.vstack([
        np.full(N_TIME, choice[trial_index], dtype=np.int8),
        np.full(N_TIME, outcome[trial_index], dtype=np.int8),
        np.full(N_TIME, early[trial_index], dtype=np.int8),
        tongue[trial_index],
    ]))
```

iii. The unit loop is inherent to the ragged storage of `spike_times`. The per-trial loop exists because the target format demands a Python list of per-trial arrays, so the objects have to be created individually anyway; the AI simply builds them in the most literal way. Since a whole session converts in about a second, neither was worth optimising.

## 10-c. What processing does the code repeat multiple times?

i. Nothing is recomputed within a session: each file is opened once, each column read once, the bin grid is built once at module level and reused for all trials and sessions, and the tongue percentiles are computed once per session during the same pass. The one redundancy is ordering rather than repetition — firing rates are computed for *all* behavioural trials and only afterwards subset by `keep` (see 10-d).

ii.
```python
BIN_EDGES = OFF_START + np.arange(N_TIME + 1) * BIN_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```
```python
with h5py.File(path, "r") as nwb:
    ...
```

iii. The converter is a single pass over the files. Because the tongue discretisation edges are per session rather than global, they can be established inside that same pass, so no second pass over the data is needed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three small items:

1. **Firing rates for excluded trials.** `_bin_good_units` is called on `go_times` for *every* behavioural trial, then `firing_rates_all[keep]` discards the rest. This is unavoidable for the `neural_nonzero` test (which needs the rates), but it means wasted work where many trials are dropped — e.g. `SC015_20190208_133600` bins 480 trials to keep 160.
2. **Diagnostic metadata**: `tongue_y_percentile_40/60`, `tongue_visible_frame_fraction`, the per-filter exclusion counts, `time_bin_centers_seconds` and `skipped_source_files` are all computed/stored but never read by the decoder.
3. **Region-lookup machinery**: the embedded base85+zlib Allen-ontology table is decompressed at import and `region_label`-style mapping is applied to every unit, but the decoder only uses `brain_region_idx` for reporting, not for training.

ii.
```python
firing_rates_all = _bin_good_units(units, good_indices, go_times)
neural_nonzero = np.any(firing_rates_all != 0, axis=(1, 2))
keep &= neural_nonzero
firing_rates = firing_rates_all[keep]
```
```python
REGION_LOOKUP = json.loads(
    zlib.decompress(base64.b85decode(_REGION_LOOKUP_B85)).decode("utf-8")
)
```

iii. The metadata items are deliberate provenance: the AI used them to reconcile totals after the run and they make the curation auditable, at negligible storage cost against an 11.6 GB payload. The region mapping is required by the target format (`brain_regions` / `brain_region_idx`) even though the decoder does not train on it, and the AI cross-validated it against the published per-region unit counts. Only item 1 is true waste, and it is a consequence of ordering the zero-trial check after binning rather than a redundant computation.
