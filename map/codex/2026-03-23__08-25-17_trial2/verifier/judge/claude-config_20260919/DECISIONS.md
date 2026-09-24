# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is a DANDI/NWB archive with one `.nwb` file per session under `/app/data/sub-<id>/`. The AI enumerates every session with a single sorted glob (`sub-*/*.nwb`, 174 files). Instead of `pynwb`, it opens each file directly with `h5py` and reads the HDF5 groups it needs (`units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `general/subject`). Each file is visited twice in a normal run: once in `select_files()`, which opens every file only to test whether it has any `classification == "good"` unit, and again in `process_session()`, which does the real work. `main()` then calls `select_files()` a third time over the full file list when printing the timing summary.

ii.
```python
DATA_DIR = Path("/app/data")

def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))

def select_files(all_files: list[Path], sample_mode: bool) -> tuple[list[Path], list[str]]:
    selected: list[Path] = []
    excluded: list[str] = []
    for path in all_files:
        with h5py.File(path, "r") as f:
            classification = decode_strings(f["units"]["classification"])
            if np.sum(classification == "good") == 0:
                excluded.append(path.name)
                continue
        selected.append(path)
    if sample_mode:
        return selected[:2], excluded
    return selected, excluded
```

```python
def process_session(path: Path, make_plot: bool = False) -> SessionResult | None:
    with h5py.File(path, "r") as f:
        classification = decode_strings(f["units"]["classification"])
        ...
        trials = f["intervals"]["trials"]
        events = f["acquisition"]["BehavioralEvents"]
        tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. From CONVERSION_NOTES Step 2/Step 6: the data directory is "a DANDI/NWB dataset (`dandiset.yaml`) with 174 session files under 28 subject folders", so a glob over `sub-*/*.nwb` is the complete session set. The AI states it "Loads NWB directly with `h5py` instead of PyNWB for speed and lower overhead" and lists "Direct HDF5 reads for ragged spike times and trial tables" as a deliberate speed-up.

## 1-b. How are the data split into subjects?

i. Subject identity is read per session from the NWB metadata field `general/subject/subject_id`, decoded from bytes if necessary, and prefixed with `sub-` (e.g. `sub-440956`). At assembly, `subjects` is built in first-encounter order over the sorted session list and `subject_idx` records each session's index into that list. This yields 28 subjects with 3–10 sessions each.

ii.
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
```

```python
for result in results:
    if result.subject_id not in subject_to_idx:
        subject_to_idx[result.subject_id] = len(subjects)
        subjects.append(result.subject_id)
    subject_idx.append(subject_to_idx[result.subject_id])
```

iii. CONVERSION_NOTES Step 5 maps `subject.subject_id` → `subjects`/`subject_idx` "direct from NWB subject metadata", noting "Session order follows sorted NWB file list after excluding the zero-good-unit session." Step 2 independently counted 28 subjects in the raw archive, and Step 3 quotes the papers ("Mice (n = 28, Table S1)"), so the AI treats 28 as the consistency target and confirms it in Step 9/Step 10.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or inference is needed. The session id is derived from the filename stem with the modality suffix stripped (`sub-440956_ses-20190207T120657`). Session order is the sorted file order, which is chronological within each subject because the filename embeds the acquisition timestamp. One session is excluded up front (the file with no classifier-labelled units), giving 173 sessions.

ii.
```python
session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
```

```python
all_files = load_candidate_files()
session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)
...
"n_sessions": len(results),
"session_ids": [r.session_id for r in results],
"excluded_sessions": excluded_sessions,
```

iii. CONVERSION_NOTES Step 4 resolves the session count explicitly: the archive has 174 files but "Papers repeatedly state `173 behavioral sessions`", so the AI excludes `sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`, "which has `classification == nan` for all 1,852 units and zero classifier-labeled good units. This reconciles the session count to 173." Key Decision 1 records the 173-session base set as preferred over both the raw 174 and the stricter 152-session behavioural subset.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial, with `start_time` / `stop_time` giving the trial bounds. Rather than assuming row-order correspondence with the go-cue event stream, the AI searches `go_start_times` for events falling inside `[start_time, stop_time]` and takes the last one as that trial's go cue; a trial with no go cue raises an error (never triggered). Every downstream quantity is then computed per trial in a Python loop over the trials table.

ii.
```python
for trial_idx in range(len(trial_start)):
    start = trial_start[trial_idx]
    stop = trial_stop[trial_idx]
    go_candidates = interval_values(go_times, start, stop)
    if len(go_candidates) == 0:
        raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
    go_time = float(go_candidates[-1])
```

```python
def interval_values(times: np.ndarray, start: float, stop: float) -> np.ndarray:
    lo = np.searchsorted(times, start, side="left")
    hi = np.searchsorted(times, stop, side="right")
    return times[lo:hi]
```

iii. CONVERSION_NOTES Step 5 states the go cue is found by "matching event timestamp into `[trial.start_time, trial.stop_time]`" and adds "Use event-in-trial matching rather than row order assumptions." Step 2 documents the trials-table columns available (`start_time`, `stop_time`, `trial_instruction`, `early_lick`, `outcome`, `free_water`, `photostim_*`, …), establishing the table as the authoritative trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at removing trials without neural data.

  1. **Observation-interval coverage.** A trial is kept only if its entire go-aligned window `[go − 2.5 s, go + 1.5 s]` lies inside a single interval of `units/obs_intervals` (taken from the first classifier-good unit).
  2. **All-zero guard.** After binning, any trial whose full neural matrix is exactly zero is dropped.

No behavioural quality filter (early lick, ignore, performance, photostim) is applied, because those variables are decoder targets/inputs. This keeps 73,910 of the 94,370 trials in the retained sessions (78%), i.e. 20,460 trials are excluded, and the mean drops to 427 trials/session.

ii.
```python
window_start = go_time + WINDOW_START_S
window_end = go_time + WINDOW_END_S
covered = np.any(
    (window_start >= session_obs_intervals[:, 0])
    & (window_end <= session_obs_intervals[:, 1])
)
if not covered:
    n_trials_dropped_outside_obs += 1
    continue
```

```python
nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
n_trials_dropped_all_zero = int((~nonzero_trial_mask).sum())
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
    input_tensor = input_tensor[nonzero_trial_mask]
    output_tensor = output_tensor[nonzero_trial_mask]
```

iii. The trajectory (steps 166–186) shows this was a reaction to verifier warnings: "a long suffix of trials has zero activity across every neuron. That points to recording coverage ending before the trial table does", followed by "I'm patching `convert_data.py` to drop trials whose full `[-2.5, +1.5] s` go-aligned window falls outside the recorded `obs_intervals`", and then "There's one residual all-zero trial left after the coverage filter … I'm adding a final post-binning guard." CONVERSION_NOTES Step 9 justifies the resulting shortfall as intentional: "The converted trial count is lower than the raw 173-session archive count because trials are excluded unless the full `[-2.5, +1.5] s` window lies inside a good-unit observation interval; this is necessary to avoid invalid all-zero neural windows." Step 1 separately records that the reference code's own trial masks (no early lick, no free water, no ignore, no photostim) are *stricter* than this task needs, since "early lick, outcome, and photostimulation are decoder targets/inputs."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged, with `units/spike_times_index` giving the per-unit end offsets), restricted to units with `units/classification == "good"`. The go-cue times from `acquisition/BehavioralEvents/go_start_times` supply the bin-edge anchor; `units/obs_intervals` is used only for trial curation.

ii.
```python
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
```

```python
def split_ragged(flat: np.ndarray, index: np.ndarray) -> list[np.ndarray]:
    starts = np.concatenate(([0], index[:-1]))
    return [flat[s:e] for s, e in zip(starts, index, strict=True)]
```

iii. CONVERSION_NOTES Step 5 maps "`units.spike_times` for units with `classification == "good"`" → `neural`, citing the reference functions `sliding_histogram` and `process_one_area`. Step 1 notes the reference pipeline is electrophysiology, so "No dF/F computation is relevant here"; spike times are the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit, a `(n_trials, 81)` matrix of absolute bin edges is searched with `np.searchsorted`, and differencing adjacent positions gives spike counts per bin; counts are divided by the 50 ms bin width. No smoothing, normalisation, or baseline subtraction. Rates are stored as `float16` to cut the pickle size (values are exact multiples of 20 Hz, well within half-precision integer range).

ii.
```python
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
...
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. CONVERSION_NOTES Key Decision 3: "Store firing rates, not spike counts, because the task explicitly asks for 50 ms bins 'for computing firing rates'." Step 6 lists the speed-ups: "Vectorized neural binning per unit across **all trials at once** using `np.searchsorted` on a `(n_trials, 81)` edge matrix" and "`float16` neural storage … to keep the full dataset size manageable" (4.6 GB final pickle). Step 10 records an `np.allclose(..., atol=1e-3)` sanity check recomputing rates from raw `units.spike_times` for 3 sessions × 4 trials × 3 units.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are kept — the verdict of the spike-sorting QC classifier. No thresholds are applied to any individual metric (`presence_ratio`, `amplitude_cutoff`, `isi_violation`, `drift_metric`, …), and `units/unit_quality` is deliberately not used. A session with zero such units is dropped entirely. This retains 69,453 units (mean 401.5/session, min 90, max 923).

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

```python
def decode_strings(dataset: h5py.Dataset) -> np.ndarray:
    try:
        return dataset.asstr()[:]
    except Exception:
        arr = dataset[:]
        return np.array(
            [x.decode() if isinstance(x, bytes) else str(x) for x in arr],
            dtype=object,
        )
```

iii. CONVERSION_NOTES Step 4 resolves this explicitly: the reference repo uses external classifier-derived good-unit lists from QC `.mat` files that are not shipped locally, so "Use `units.classification == "good"` as the local equivalent of the external QC lists. `unit_quality == "good"` is too permissive and produces 154,948 units, which is far above the published total." Key Decision 2 repeats this. Step 9 compares 69,453 obtained vs 69,943 reported in the QC white paper and accepts the 0.7% gap as "a likely archive/export-version discrepancy".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so alignment is a lookup: the trial's go-cue timestamp (last `go_start_times` event inside `[trial_start, trial_stop]`) is added to a fixed vector of 81 relative bin edges, and spikes are binned against those absolute edges. No resampling, interpolation, or per-stream offset correction.

ii.
```python
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
```

```python
go_candidates = interval_values(go_times, start, stop)
go_time = float(go_candidates[-1])
```

```python
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
counts = np.diff(edge_idx, axis=1)
```

iii. CONVERSION_NOTES Step 3/Step 4: "Reference analyses are aligned to go cue for neural/video preprocessing and for many figures (`time to go` axis; code explicitly aligns lick/stim/marker data to go cue)". Step 10 Check 6: "Temporal alignment: go-cue alignment matches the reference code's `time to go` convention and `align_markers_between_lims`." Key Decision 9 states the AI keeps reference go-cue alignment and changes only bin width/window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning −2.5 s to +1.5 s relative to the go cue, identical for every trial and session. The grid is defined once at module level as 81 relative edges plus 80 centres. This is a deliberate departure from the reference pipeline's 40 ms width / 3.4 ms stride sliding histogram; the raw spike times are binned once directly onto the 50 ms grid, so there is no rebinning of an already-binned signal. The tongue video (~300 Hz) is not rebinned either — it is point-sampled at the 50 ms bin centres (see 8-d). `metadata['time_bin_size'] = 50.0` ms.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_WIDTH_S / 2.0
N_BINS = len(BIN_CENTERS_REL)
```

```python
"time_bin_size": 50.0,
"temporal_alignment_event": "Go cue onset",
"off_start": WINDOW_START_S,
"off_end": WINDOW_END_S,
```

iii. CONVERSION_NOTES Step 4: "User task requires go-cue alignment with 50 ms bins from `-2.5 s` to `+1.5 s` … Preserve the reference alignment/QC logic but intentionally deviate in final bin width/window to satisfy the decoder task. This is an explicit task-driven modification, not a misunderstanding of the reference pipeline." Step 10 Check 8 verifies "all converted trials have exactly 80 bins".

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times.timestamps` (the sample-epoch tone onsets), combined with the trial's go-cue time and the trial bounds. For each trial the AI takes the **last** sample-start event in `[trial_start, go_time]`. If a trial has none, it falls back to a fixed `go_time − 1.85 s` (nominal sample 0.65 s + delay 1.2 s); this fallback fired 0 times over the full dataset.

ii.
```python
sample_start_times = events["sample_start_times"]["timestamps"][:].astype(np.float64)
...
sample_candidates = interval_values(sample_start_times, start, go_time)
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
else:
    sample_onset = float(sample_candidates[-1])
```

iii. Trajectory step 126: "early-lick trials can contain repeated sample events, so the correct 'tone onset' for alignment is the last sample start before the go cue, not just the first sample event in the trial." CONVERSION_NOTES Step 5: "Early-lick trials can contain multiple sample-start events; the last one is the final replay that leads to the observed go cue," and Key Decision 4 repeats this. Step 3 records the task structure (0.65 s sample, 1.2 s delay, 0.1 s go cue) that motivates the 1.85 s fallback.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: each of the 80 bin centres is converted to absolute time (`go_time + BIN_CENTERS_REL`) and the tone onset is subtracted, giving seconds elapsed since the tone at each bin centre. Stored as `float32` in row 0 of the `(2, 80)` input array. Because the go–tone gap varies per trial (and can be large on replayed trials), the observed range across the dataset is [−1.5, 9.7] s.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
...
input_tensor[keep_idx, 0, :] = rec["time_from_tone"]
```

iii. CONVERSION_NOTES Step 5 maps it as "store `(bin_center_time - sample_onset_time)` in seconds for every bin". Step 9 lists the converted range [−1.5, 9.7] and judges it "Plausible" given "task structure permits long pre-go delays on replayed trials". Step 10 Check 4 verified it against raw events with `np.allclose(..., atol=1e-6)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the input uses the same per-trial go-cue anchor and the same fixed relative grid as the spike binning, so input bin *k* covers exactly the interval of neural bin *k*. The neural value is the count over `[edge_k, edge_{k+1})` while the input value is evaluated at the centre of that same interval.

ii.
```python
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_WIDTH_S / 2.0
```

```python
bin_centers_abs = go_time + BIN_CENTERS_REL      # inputs
...
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL   # neural
```

iii. No separate justification is given beyond the shared go-cue alignment (CONVERSION_NOTES Key Decision 9, Step 10 Check 6). The `--show-processing` plots overlay the tone-onset marker, go cue, inputs and neural raster on one `time to go` axis to demonstrate there is no misalignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Three string columns of the trials table — `photostim_onset`, `photostim_duration`, `photostim_power` — plus `trials.start_time` to convert the trial-relative onset to absolute time and the go cue to place it on the bin grid. Non-stimulated trials carry the literal string `'N/A'`. The AI requires **all three** of onset, duration and power to be present before marking any bin as stimulated. (The separate `photostim_start_times` / `photostim_stop_times` event streams were noted in Step 2 but not used.)

ii.
```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
```

```python
def parse_optional_float(value: str) -> float | None:
    if value in {"N/A", "", "nan", "None"}:
        return None
    return float(value)
```

iii. CONVERSION_NOTES Step 5: "`trials.photostim_onset`, `trials.photostim_duration`, `trials.start_time` → `input[1]` … Trials with `photostim_power == N/A` are all zeros." Step 2 counted "Trials with non-`N/A` photostim power: 18,588 / 94,990 (`19.57%`)", against the paper's "~25% randomly interleaved trials" for the stimulated subset (Step 3). Step 3 also records that photoinhibition is confined to the late delay epoch, ending before the go cue, which the AI used as a shape check.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1, `float32`) time series rather than a per-trial flag. The trial-relative onset is converted to absolute time (`trial_start + onset`), the offset is onset + duration, and a bin is 1 when its centre falls in `[stim_start, stim_stop)`. Trials without stimulation are left as an all-zero row.

ii.
```python
photostim_row = np.zeros(N_BINS, dtype=np.float32)
stim_onset = parse_optional_float(trial_photostim_onset[trial_idx])
stim_dur = parse_optional_float(trial_photostim_duration[trial_idx])
stim_power = parse_optional_float(trial_photostim_power[trial_idx])
if stim_power is not None and stim_onset is not None and stim_dur is not None:
    stim_start_abs = start + stim_onset
    stim_stop_abs = stim_start_abs + stim_dur
    photostim_row = (
        (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
    ).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Convert trial-relative onset/duration to absolute time, then to go-aligned binary series over bins," citing that "reference preprocessing shifts stimulation times relative to go cue". The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)", which the binary-per-bin representation satisfies. Trajectory step 138: "photostimulation starts about 1.2 s before go for 0.5 s", used as a sanity check on the resulting waveform.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Both the stimulation interval and the bin centres are expressed on the session-absolute clock, and the bin centres are derived from the same go-cue anchor as the neural bin edges, so the comparison is direct and shares the neural grid.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
...
stim_start_abs = start + stim_onset
stim_stop_abs = stim_start_abs + stim_dur
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
```

iii. CONVERSION_NOTES Step 10 Check 4 re-derived `photostim_on` from the raw trial-relative onset/duration fields for spot-checked trials and matched with `np.allclose(..., atol=1e-6)`. The `--show-processing` plot draws the photostim step function on the same `time to go` axis as the go cue and the neural raster.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no explicit choice column. The AI derives it from `trials.trial_instruction` (`left`/`right`) crossed with `trials.outcome` (`hit`/`miss`/`ignore`): hit ⇒ instructed side, miss ⇒ opposite side. For `ignore` trials — where no response occurred — it additionally consults the raw lick streams `BehavioralEvents/left_lick_times` and `right_lick_times`, assigning the side of the earliest lick found anywhere in `[trial_start, trial_stop]` (which includes pre-go early licks); if no lick at all is found, it falls back to the instructed side. Only two classes exist in the output (`left=0`, `right=1`); there is no "no lick" class.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}

def infer_choice(instruction, outcome, trial_start, trial_stop,
                 left_lick_times, right_lick_times) -> tuple[int, str]:
    if outcome == "hit":
        return CHOICE_MAP[instruction], "instruction+outcome"
    if outcome == "miss":
        opposite = "right" if instruction == "left" else "left"
        return CHOICE_MAP[opposite], "instruction+outcome"

    left_trial = interval_values(left_lick_times, trial_start, trial_stop)
    right_trial = interval_values(right_lick_times, trial_start, trial_stop)
    if len(left_trial) and len(right_trial):
        side = "left" if left_trial[0] <= right_trial[0] else "right"
        return CHOICE_MAP[side], "ignore:first_lick_in_trial"
    if len(left_trial):
        return CHOICE_MAP["left"], "ignore:first_lick_in_trial"
    if len(right_trial):
        return CHOICE_MAP["right"], "ignore:first_lick_in_trial"
    return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. CONVERSION_NOTES Step 5 documents the hit/miss derivation as "Verified against first post-go lick on sampled files: 0 mismatches in 10,084 responded trials," citing "paper/code derive LL/RR/LR/RL from instruction + correctness". For ignore trials it records "no direct reference equivalent because paper excludes ignore trials" and Key Decision 6: "Because the source does not contain an explicit choice label when no response occurs, use the earliest lick side in the trial if available; otherwise fall back to instructed side. This is a task-driven compromise." Trajectory step 130 shows the AI verified "ignore trials have no post-go lick in the sampled sessions" and then checked `train_decoder.py` "to see whether missing labels are supported", but it never considered adding the third category. Step 12 attributes the relatively low choice accuracy (0.7405) to this: "`17.9%` of kept trials (`13,258 / 73,910`) are ignore trials with no ground-truth choice in the source data, so the required fallback labels inject unavoidable noise."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The inferred side is coded `left=0`, `right=1` and written into row 0 of the per-trial `(4, 80)` `int16` output array, repeated identically across all 80 bins. `output_values[0] = ['left', 'right']`. Across the dataset the distribution is [0.491, 0.509].

ii.
```python
choice_code, choice_source = infer_choice(...)
choice_sources[choice_source] += 1
...
output_row = np.empty((4, N_BINS), dtype=np.int16)
output_row[0, :] = choice_code
```

```python
"output_names": ["choice", "outcome", "early_lick", "tongue_y_bin"],
"output_values": [
    ["left", "right"],
    ...
],
```

iii. CONVERSION_NOTES Key Decision 5: "Make outputs 2D time-varying arrays `(4, 80)` for every trial. Choice/outcome/early-lick will be repeated across bins … This satisfies the decoder format cleanly." The metadata field `choice_note` records the caveat verbatim: "choice inferred from instruction+outcome on hit/miss trials; ignore trials use earliest lick in trial when present, otherwise instructed side as placeholder."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the NWB trials table, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`. No derivation.

ii.
```python
trial_outcome = decode_strings(trials["outcome"])
...
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. CONVERSION_NOTES Step 2 enumerated the raw vocabulary from the data (`outcome`: `hit`, `miss`, `ignore`) and counted `hit=65,254`, `miss=15,641`, `ignore=14,095` across all sessions. Step 5 records the mapping as "direct from raw trial table" and notes it "Matches user specification exactly."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the strings to `ignore=0`, `miss=1`, `hit=2`, and the code is written into row 1 of the output array, repeated across all 80 bins. In the converted dataset the resulting distribution is [0.173, 0.009, 0.818] — the `miss` class is nearly absent because the trial filter in 1-e removes almost all miss trials (their trials are shorter than the 4 s window), versus 16.5% miss in the raw trials table.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
output_row[1, :] = outcome_code
```

```python
"output_values": [
    ...
    ["ignore", "miss", "hit"],
```

iii. CONVERSION_NOTES Step 5: "Map strings to integers: `ignore=0`, `miss=1`, `hit=2`; replicate across all time bins … Matches user specification exactly." Step 9 lists the converted distribution [0.173, 0.009, 0.818] and calls it "Broadly consistent" with the paper's "84% correct control rate"; Step 11 notes outcome decoding is "Above chance despite severe class imbalance (`miss` is rare)". The AI did not investigate why `miss` fell from 16.5% to 0.9%.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of the trials table, which holds `'no early'` / `'early'`.

ii.
```python
trial_early = decode_strings(trials["early_lick"])
...
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. CONVERSION_NOTES Step 2 enumerated the vocabulary (`early_lick`: `early`, `no early`) and counted `early=10,805`, `no early=84,185` in the raw archive. Step 5: "direct from raw trial table … Matches user specification exactly." Step 1 notes the reference analyses exclude early-lick trials, but "For this conversion, those variables still need to be preserved because early lick, outcome, and photostimulation are decoder targets/inputs."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed dictionary maps `'no early' → 0`, `'early' → 1`, written into row 2 of the output array and repeated across all 80 bins. Converted distribution [0.885, 0.115], essentially unchanged from the raw archive fraction (11.4%).

ii.
```python
EARLY_MAP = {"no early": 0, "early": 1}
...
output_row[2, :] = early_code
```

```python
"output_values": [
    ...
    ["no", "yes"],
```

iii. CONVERSION_NOTES Step 5: "Map `no early=0`, `early=1`; replicate across all time bins." Key Decision 5 covers the repeat-across-bins choice. Step 9 confirms the converted distribution matches the raw trial table.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` (~300 Hz, 3.4 ms step). Column 1 (`y`) is the value; column 2 (`likelihood`) gates visibility; column 0 (`x`) is used only inside the velocity-based outlier detector.

ii.
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
processed_tongue_y, tongue_info = process_tongue_trace(tongue_data, tongue_timestamps)
```

```python
xy = tongue_xyzl[:, :2].astype(np.float64, copy=False)
y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
```

iii. CONVERSION_NOTES Step 2 documents the stream layout: "continuous video tracking arrays `Camera0_side_TongueTracking`, … each with shape `(n_frames, 3)` and columns `(x, y, likelihood)`, timestamped every `0.0034 s`", and "All sessions contain tongue tracking". Step 4 resolves the marker-set discrepancy with the reference code: "For this task, use the directly available side-view tongue trajectory from NWB. No paw/whisker reconstruction is needed because the target output only requires tongue y-position."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. A session-level preprocessing pass (`process_tongue_trace`) with three steps, run once per session over the whole recording:

  1. **Velocity outlier rejection**: frame-to-frame 2-D speed is computed, and frames exceeding mean + 5 σ are flagged (typically ~70 frames per session) and replaced by linear interpolation over time.
  2. **Occlusion imputation**: frames with `likelihood < 0.1` (about 89% of all frames, since the tongue is only protruded ~10% of the time) are **replaced by the session mean y of the good frames**, not discarded.
  3. **Percentiles**: the 40th and 60th percentiles are then taken over the *whole imputed trace* (all frames, including the ~89% that now equal the mean).

  Because 89% of the trace is the constant session mean, the 40th and 60th percentiles collapse to exactly that same constant (verified: `p40 == p60 == session_mean_y == 280.7989` for `sub-440956_ses-20190207T120657`).

ii.
```python
dt = np.diff(tongue_timestamps)
dt[dt == 0] = np.nan
velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
vel_threshold = vel_mean + 5.0 * vel_std
outlier_mask = np.zeros(len(y), dtype=bool)
outlier_mask[1:] = np.isfinite(velocity) & (velocity > vel_threshold)
low_likelihood_mask = likelihood < LIKELIHOOD_THRESHOLD      # LIKELIHOOD_THRESHOLD = 0.1

session_mean_y = float(np.mean(y[good_mask]))
processed_y = y.copy()
processed_y[low_likelihood_mask] = session_mean_y
interp_mask = outlier_mask | ~np.isfinite(processed_y)
keep_mask = ~interp_mask
if np.any(interp_mask):
    processed_y[interp_mask] = np.interp(
        tongue_timestamps[interp_mask],
        tongue_timestamps[keep_mask],
        processed_y[keep_mask],
    )

p40, p60 = np.percentile(processed_y, [40.0, 60.0])
```

iii. CONVERSION_NOTES Step 3 quotes the method paper's video preprocessing: "marker outliers removed using a five-sigma velocity threshold and imputed from nearby frames" and "when the tongue is occluded in the mouth, tongue position is set to its mean value". Step 5 therefore plans "5-sigma velocity outlier detection + interpolation; low-likelihood/occluded frames imputed to session mean tongue y." Step 7 already flags the consequence: "The tongue trace preprocessing is conservative: many low-likelihood tongue frames are filled to the session mean, producing a strong middle-bin dominance. This is acceptable for format validation but is a likely place to revisit if decoder accuracy for tongue is weak." The AI did not revisit it (Step 12 found the accuracy acceptable and closed the issue).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes only (`0`, `1`, `2`); the instruction's fourth class `3: not visible` is **not** implemented. Thresholding is by two cascading comparisons against the session `p40` / `p60`: start at 0, set to 1 where `y >= p40`, then set to 2 where `y > p60`. Because `p40 == p60 == session mean` (see 8-b), the effective semantics are: class 0 = visible tongue below the session mean, class 1 = occluded/imputed frames (plus any frame exactly at the mean), class 2 = visible tongue above the session mean. The realised distribution across the dataset is [0.088, 0.824, 0.088] rather than the ~40/20/40 split the instruction's percentile definition implies.

ii.
```python
tongue_y = processed_tongue_y[tongue_frame_idx]
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
...
output_row[3, :] = tongue_bin
```

```python
"output_values": [
    ...
    ["<40th_pct", "40th_to_60th_pct", ">60th_pct"],
],
```

iii. CONVERSION_NOTES Step 5 states the plan as "discretize with session-level 40th and 60th percentiles into classes `0/1/2`", following the Decoder Task percentile spec. The AI recorded the resulting distribution in Step 7 ([0.067, 0.846, 0.087]) and Step 9 ([0.088, 0.824, 0.088]) and labelled it "Expected middle-bin dominance", attributing it to the mean-fill rather than to the percentile collapse. No "not visible" class was considered anywhere in the notes or trajectory.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps share the session-absolute clock with spikes and events, so no offset correction is needed. For each trial the AI **point-samples** the preprocessed trace: `np.searchsorted(tongue_timestamps, bin_centers_abs, side='right') - 1` picks the last camera frame at or before each 50 ms bin centre, and the index is clipped into the valid range. There is no averaging of the ~15 frames that fall inside each bin, and bins whose centre precedes the session's first frame (or falls in a gap in trial-gated video) silently receive the nearest available frame rather than being marked missing.

ii.
```python
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. CONVERSION_NOTES Key Decision 8: "For each 50 ms neural bin, assign tongue y from the processed continuous tongue trace at the bin center (or closest preceding frame). This is closer to the reference marker-alignment logic than averaging over long windows." Step 1 notes the reference `align_markers_between_lims` builds go-cue-relative marker trajectories at `dt=0.0034 s`, and `temporal_alignment_embed_and_ephys` "Aligns marker/video time bases to ephys bin centers", which the AI cites as precedent for centre-sampling. Step 10 Check 5 reports the full time-varying `tongue_y_bin` was recomputed from the raw tracking stream for spot-checked trials and matched exactly.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five distinct cases:

  - **Session never quality-controlled** (`classification` is NaN for all units): `decode_strings` turns the NaNs into non-`"good"` strings, the session has zero good units, and it is excluded both in `select_files` and defensively in `process_session` (returns `None`). One session (`sub-440958_ses-20190216T162508`) is removed this way, reconciling 174 files to 173 sessions.
  - **Trials with no spike coverage**: excluded by the obs-interval filter plus the post-binning all-zero guard (see 1-e).
  - **Missing tone onset**: falls back to `go_time − 1.85 s` and is counted; the counter is 0 for the full dataset.
  - **Occluded / low-confidence tongue frames and tracking outliers**: mean-filled and interpolated respectively (see 8-b), never left as NaN — so no missing-data category reaches the output.
  - **Missing anatomical annotation**: `build_region_labels` raises rather than silently emitting an empty region label.

  Hard failure modes: a trial with no go cue raises `ValueError`, and a session where every trial is filtered out raises `ValueError` instead of being dropped. Neither triggered. There is no explicit "≥ 2 trials per session" guard (the minimum realised is 137).

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

```python
def build_region_labels(anno_name, good_unit_mask) -> list[str]:
    labels = [str(x) for x in anno_name[good_unit_mask].tolist()]
    if any((label == "" or label.lower() == "nan") for label in labels):
        raise ValueError("Good units unexpectedly contain empty anatomical annotations.")
    return labels
```

```python
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
```

```python
n_trials = len(trial_records)
if n_trials == 0:
    raise ValueError(f"{path.name}: all trials were excluded by observation-interval coverage.")
```

iii. CONVERSION_NOTES Step 4 justifies the session exclusion by reconciling to the papers' "173 behavioral sessions". Step 10 Check 8 reports the fallback counters: "`n_missing_sample_onset_fallback = 0`, so no trial needed the `go - 1.85 s` default" and "`n_ignore_choice_fallback = 13,258`; this is expected because ignore trials do not have an explicit behavioral choice label in the raw data." The tongue imputation is justified from the method paper's stated procedure (Step 3). Step 5 justifies the `anno_name` hard check: "All classifier-good units checked so far have non-empty `anno_name`; this avoids mislabeling orbital/other-cortex neurons as coarse probe target labels."

## 10-a. What are the most time-consuming steps of the code?

i. Full conversion took 5.09 min for 173 sessions, ~1.67 s/session (range ~0.5 s to ~8.5 s, scaling with unit count). The costs, largest first: (1) HDF5 reads — the whole ragged `spike_times` buffer, the `(n_frames, 3)` tongue array (~680 k rows), and the trial/unit tables; (2) the per-unit `np.searchsorted` binning loop; (3) pickling the 4.6 GB result; (4) the per-trial Python loop that builds inputs/outputs; and (5) two extra whole-archive passes: `select_files()` opens all 174 files before conversion begins, and `main()` calls it again at the end.

ii.
```python
session_start = time.time()
...
"session_seconds": float(time.time() - session_start),
```

```python
result = process_session(path, make_plot=make_plot)
elapsed = time.time() - t0
session_times.append(elapsed)
print(f"  kept {result.n_good_units} good units, {result.n_trials} trials, {elapsed:.2f}s")
```

```python
mean_session_s = float(np.mean(session_times)) if session_times else 0.0
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

iii. CONVERSION_NOTES Step 6 identifies the bottlenecks it anticipated: "Per-trial Python loops over neurons would be too slow and memory-heavy for the full dataset" and "Full-fidelity float32 storage would make the output pickle unnecessarily large." Step 7 records the projection ("0.70 s/session … 2.01 min projected for 173 sessions") against the instructions' 15-minute budget; the realised 5.09 min stayed within it, so no further optimisation was pursued. The redundant whole-archive passes are not mentioned anywhere.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops remain.

  - **Per-unit neural binning loop** (`for unit_idx, spikes in enumerate(good_spike_times)`): already vectorised across all trials and bins via a `(n_trials, 81)` edge matrix. It cannot be collapsed further because spike arrays are ragged.
  - **Per-trial construction loop** (`for trial_idx in range(len(trial_start))`, ~550 iterations/session): computes go-cue lookup, tone onset, `time_from_tone`, photostim, choice, outcome, early lick and tongue discretisation one trial at a time. Every one of these is expressible as a whole-session array operation (the reference does exactly that), so this is the clearest missed vectorisation.
  - **Per-trial copy loop** (`for keep_idx, rec in enumerate(trial_records)`): unpacks dicts into preallocated tensors; would be unnecessary if the previous loop were vectorised.
  - Minor: `split_ragged` materialises a Python list of *all* units' spike arrays (including the ~75% non-good units) before subsetting to the good ones.

ii. The vectorised part:
```python
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

The non-vectorised part:
```python
for trial_idx in range(len(trial_start)):
    ...
    bin_centers_abs = go_time + BIN_CENTERS_REL
    time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
    ...
    choice_code, choice_source = infer_choice(...)
    ...
    tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
    trial_records.append({...})
```

iii. CONVERSION_NOTES Step 6 claims the relevant speed-up: "Vectorized neural binning per unit across **all trials at once** using `np.searchsorted` on a `(n_trials, 81)` edge matrix," which it credits with reducing "per-session neural binning from projected minute-scale nested loops to sub-second runtime." The per-trial input/output loop is not identified as a vectorisation candidate; the AI's implicit justification is that total runtime (5 min) was already inside the instructions' 15-minute budget.

## 10-c. What processing does the code repeat multiple times?

i. Three repetitions, all file-I/O related:

  - **Every NWB file is opened at least twice.** `select_files()` opens all 174 files and decodes the full `units/classification` column solely to test for good units; `process_session()` then reopens each surviving file and decodes the same column again.
  - **`select_files()` is called a third time** at the end of `main()`, re-opening and re-scanning all 174 files, even in `--full` mode where its result is only used to print a projection in sample mode.
  - **`decode_strings(f["units"]["classification"])`** is therefore evaluated three times per file across a full run.

  Within a session, nothing else is recomputed: the bin grid is module-level, the tongue trace is preprocessed once per session, and each stream is read once.

ii.
```python
def select_files(all_files, sample_mode):
    for path in all_files:
        with h5py.File(path, "r") as f:
            classification = decode_strings(f["units"]["classification"])
            if np.sum(classification == "good") == 0:
```

```python
def process_session(path, make_plot=False):
    with h5py.File(path, "r") as f:
        classification = decode_strings(f["units"]["classification"])
        good_unit_mask = classification == "good"
```

```python
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

iii. CONVERSION_NOTES Step 6 lists "Direct HDF5 reads for ragged spike times and trial tables" and "Session-by-session processing to bound peak memory" as its I/O strategy, and the instructions asked to "Avoid unnecessary file I/O". The duplicated scans are not acknowledged anywhere in the notes; the pre-pass exists so that the excluded-session list and the `--sample` selection can be reported before conversion starts, and `process_session` keeps its own check as a defensive guard (it returns `None`).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are performed and then discarded or rendered moot:

  - **The third `select_files()` call** in `main()`: a full re-scan of all 174 NWB files whose result is used only inside `if sample_mode:`, so in `--full` runs it is pure waste.
  - **Velocity-based outlier detection** in `process_tongue_trace`: computes 2-D speed over the whole session (~680 k frames) and interpolates the flagged frames — but only ~69 frames per session are flagged, and ~89% of the trace is overwritten by the mean fill regardless, so it has essentially no effect on the output.
  - **`xy` (tongue x)** is read and used only for that velocity computation; tongue x never reaches the output.
  - **`trial_photostim_power`** is decoded and parsed only as a redundant gate — `photostim_onset`/`photostim_duration` already carry `'N/A'` on unstimulated trials.
  - **`plot_payload`** retains full-session copies of `tongue_y_raw`, `tongue_y_processed` and `tongue_timestamps` for up to two sessions even though only one trial is plotted.
  - **Per-session `stats`** (`choice_sources`, `tongue_info`, `stim_trials`, timings) are accumulated on every session but only aggregate counters reach `metadata`; the rest is dropped.
  - **`obs_intervals` / `obs_intervals_index` for all units** are read whole even though only the first good unit's slice is used.

ii.
```python
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
...
if sample_mode:
    print(f"  projected full-conversion time: {est_full_s / 60.0:.2f} min")
```

```python
xy = tongue_xyzl[:, :2].astype(np.float64, copy=False)
velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
...
outlier_mask[1:] = np.isfinite(velocity) & (velocity > vel_threshold)
...
processed_y[low_likelihood_mask] = session_mean_y      # overwrites ~89% of frames anyway
```

```python
stim_power = parse_optional_float(trial_photostim_power[trial_idx])
if stim_power is not None and stim_onset is not None and stim_dur is not None:
```

iii. The velocity/outlier machinery is justified in CONVERSION_NOTES Step 3/Step 5 as reproducing the method paper's stated video preprocessing ("marker outliers removed using a five-sigma velocity threshold and imputed from nearby frames"), i.e. it is there for reference fidelity rather than because it materially changes the output — Step 10 Check 6 cites it as evidence that "tongue processing follows the method-paper description". The `stats`/`plot_payload` structures are justified by the instructions' requirements for `--show-processing` plots and for reporting sanity-check counters. The redundant `select_files()` call and the unused `tongue_x` read are not discussed.
