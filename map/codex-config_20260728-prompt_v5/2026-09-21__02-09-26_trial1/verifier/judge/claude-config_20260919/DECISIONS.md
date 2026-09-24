# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is a DANDI/NWB archive with one `.nwb` file per session under `/app/data/sub-<id>/`. The AI enumerates every session with a single sorted glob over that layout and processes each file once. Instead of `pynwb`, it opens each file **directly with `h5py`** and reads the HDF5 paths by hand (`intervals/trials`, `units/...`, `acquisition/BehavioralEvents/...`, `acquisition/BehavioralTimeSeries/...`, `general/extracellular_ephys/electrodes/location`). All 174 files are visited; one is skipped during processing (see 2-c), giving 173 sessions, 28 subjects, 69,453 units and 90,844 trials.

ii.
```python
DATA_DIR = Path("/app/data")
...
all_paths = sorted(DATA_DIR.glob("sub-*/*.nwb"))
chosen_paths = choose_session_paths(all_paths, sample_only=False)
...
for session_path in chosen_paths:
    result = process_session(session_path, make_plot=plots_remaining > 0)
```

```python
with h5py.File(session_path, "r") as nwb:
    subject_id = session_path.parent.name
    classification = read_str_array(nwb["units/classification"])
    ...
    trials = nwb["intervals/trials"]
    trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
    trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
```

iii. From CONVERSION_NOTES Step 2/4: "`/app/data` is a DANDI/NWB dataset (`dandiset.yaml` + one folder per subject)… Each session is stored as one NWB file inside its subject folder, for a total of `174` NWB files." The AI treated NWB as "the authoritative raw format available here" because the reference MATLAB code loads DataJoint `.mat` exports that are not shipped, and mapped the NWB trial/event/unit fields onto the variables the reference code uses. `h5py` was chosen for direct, low-overhead field access (the whole conversion runs in 184 s).

## 1-b. How are the data split into subjects?

i. Each session's subject is taken from the **parent directory name** (`sub-440956`), not from `general/subject/subject_id`. `subjects` is the list of unique ids in order of first appearance (which, because the paths are sorted, is alphabetical), and `subject_idx` indexes each session into that list. Result: 28 subjects with 3–10 sessions each.

ii.
```python
subject_id = session_path.parent.name
```

```python
subjects_order = list(OrderedDict((sess["subject_id"], None) for sess in processed_sessions).keys())
subject_to_idx = {subject: i for i, subject in enumerate(subjects_order)}
...
"subjects": subjects_order,
"subject_idx": np.array(
    [subject_to_idx[sess["subject_id"]] for sess in processed_sessions], dtype=np.int64
),
```

iii. CONVERSION_NOTES Step 5 mapping table: "`general/subject/subject_id` / subject folder name → `subjects`, `subject_idx`… Use strings like `sub-440956` for stability." The folder name is derived from the NWB `subject_id`, so the grouping is identical; the AI preferred the path-derived string as a stable key it could read without opening the subject group.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting or grouping is needed. The session identifier is the **file stem** (`sub-440956_ses-20190207T120657_behavior+ecephys+ogen`) rather than `nwb.identifier`. Session order in the output is the sorted file order, which is chronological within each subject because the filename embeds the acquisition timestamp. Sessions with no QC-`good` units are dropped, and sessions with fewer than 2 surviving trials would also be dropped; in practice exactly one session is lost, leaving 173.

ii.
```python
all_paths = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
session_id = session_path.stem
```

```python
if good_unit_idx.size == 0:
    print(f"  Skipping {session_id}: no good units in NWB classification.")
    return None
...
if int(keep_trials.sum()) < 2:
    print(f"  Skipping {session_id}: fewer than 2 valid trials after filtering.")
    return None
```

```python
"session_ids": [sess["session_id"] for sess in processed_sessions],
"session_info": [sess["stats"] for sess in processed_sessions],
```

iii. CONVERSION_NOTES Step 4: "One NWB session (`sub-440958_ses-20190216T162508...`) has `classification == "nan"` for all `1852` units and no usable good-unit labels. Excluding that session yields `173` analyzable sessions, consistent with the papers." Key decision 1: "Keep sessions with at least one `classification == "good"` unit."

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial. The AI asserts one go-cue event per trial row and hard-fails if this is violated; it also has a defensive fallback if `go_stop_times` has a different length (the fallback value is subsequently never used — see 10-d).

ii.
```python
trials = nwb["intervals/trials"]
trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
n_trials_raw = trial_start.shape[0]
...
if go_start.shape[0] != n_trials_raw:
    raise ValueError(f"{session_id}: expected one go cue per trial.")
if go_stop.shape[0] != n_trials_raw:
    go_stop = np.minimum(go_start + 1.5, trial_stop)
```

iii. CONVERSION_NOTES Step 2 lists the trial columns and Step 5 maps them to the decoder outputs. The AI explicitly rejected re-deriving trial boundaries and also explicitly rejected `units/is_good_trials` as a trial index: Key decision 4 — "in this NWB release the field is not consistently aligned to the trial table across sessions (`9` sessions have widths not equal to the trial table), so using it would create arbitrary shape-dependent behavior."

## 1-e. How are trials filtered based on quality controls?

i. Four filters, no behavioural quality filter:
1. **Neural observation support** — the whole decoder window `[go − 2.5 s, go + 1.5 s]` must lie inside the *intersection* of `units/obs_intervals` across all retained good units (max of per-unit interval starts, min of per-unit interval ends).
2. **Event validity** — the trial must have a finite go cue and at least one `sample_start_times` event inside `[trial_start, go_start]`.
3. **All-zero neural fallback** — after binning, any trial whose entire (units × bins) tensor is zero is dropped.
4. **Session minimum** — a session with fewer than 2 surviving trials is dropped.

Early-lick, miss, ignore and photostim trials are deliberately **kept**. 90,844 of 94,990 raw trials survive (95.6%).

ii.
```python
common_obs_start, common_obs_end = get_common_observation_window(
    np.asarray(nwb["units/obs_intervals"][:], dtype=np.float64),
    np.asarray(nwb["units/obs_intervals_index"][:], dtype=np.int64),
    good_unit_idx.astype(np.int64),
)
...
common_good_trials = (
    np.isfinite(go_start)
    & ((go_start + OFF_START_S) >= common_obs_start)
    & ((go_start + OFF_END_S) <= common_obs_end)
)
...
event_valid = sample_valid & np.isfinite(go_start) & np.isfinite(response_stop)
keep_trials = common_good_trials & event_valid
if int(keep_trials.sum()) < 2:
    return None
```

```python
neural_supported = np.any(neural != 0, axis=(1, 2))
n_zero_neural_trials = int((~neural_supported).sum())
if n_zero_neural_trials:
    keep_idx = keep_idx[neural_supported]
    ...
    neural = neural[neural_supported]
if keep_idx.shape[0] < 2:
    return None
```

iii. CONVERSION_NOTES Key decision 12: "Keep only trials whose full decoder window lies inside the shared `units/obs_intervals` support of retained good units, and drop any residual trial whose binned neural tensor is entirely zero. Rationale: some NWB sessions contain more behavioral trials than the stored spike data actually support; without this guard the sample verifier reports unsupported all-zero late trials." Key decision 3: "Do not exclude early-lick, miss, ignore, or photostim trials globally… these states are required as decoder outputs/inputs. The papers excluded some of them for specific analyses, but the conversion task explicitly needs them represented."

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. `units/spike_times` (ragged, with `units/spike_times_index` offsets), restricted to units with `units/classification == 'good'`, plus `acquisition/BehavioralEvents/go_start_times/timestamps` which places the bin edges. `units/obs_intervals` is used only for trial curation, not for the rates themselves.

ii.
```python
neural = bin_spikes_for_session(
    nwb["units/spike_times"],
    np.asarray(nwb["units/spike_times_index"][:], dtype=np.int64),
    good_unit_idx.astype(np.int64),
    keep_go_start,
)
```

```python
def unit_start_indices(index_array: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], index_array[:-1])).astype(np.int64, copy=False)
```

iii. CONVERSION_NOTES Step 5 mapping: "`units/spike_times` + per-trial go cue timestamps + `units/classification` → `neural`… This preserves the reference go-cue alignment and spike-time processing but changes bin width/window to match the decoder task." Spike times are the only neural representation in the archive.

## 2-b. How is the `neural` data processed?

i. Per good unit, the absolute bin edges of every trial are built as one flat array; `np.searchsorted` gives the running spike count at each edge and differencing adjacent counts gives the spike count per bin; counts are divided by the 0.05 s bin width to give **firing rate in Hz**. No smoothing, normalisation or baseline subtraction. Rates are stored as **`float16`** to shrink the 6 GB pickle. A unit with zero spikes is filled with 0.

ii.
```python
def bin_spikes_for_session(spike_times_dataset, spike_index, good_unit_idx, go_times):
    abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]
    unit_starts = unit_start_indices(spike_index)
    fr = np.empty((go_times.shape[0], good_unit_idx.shape[0], N_BINS), dtype=np.float16)
    for out_i, unit_i in enumerate(good_unit_idx):
        spikes = np.asarray(
            spike_times_dataset[unit_starts[unit_i] : spike_index[unit_i]], dtype=np.float64,
        )
        if spikes.size == 0:
            fr[:, out_i, :] = 0.0
            continue
        counts = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(
            go_times.shape[0], N_BINS + 1
        )
        fr[:, out_i, :] = (np.diff(counts, axis=1) / BIN_SIZE_S).astype(np.float16)
    return fr
```

iii. CONVERSION_NOTES Key decision 5: "Store firing rates, not raw counts. Rationale: the task explicitly asks for `50-ms-width bins for computing firing rates`, and the reference preprocessing also converts spike times to firing rates" (reference `sliding_histogram(..., rate=True)`). On `float16`, the trajectory (step 158) records: "neural trials will be stored as `float16` and outputs as small integers, while leaving the conversion math itself in higher precision. That should materially reduce the full pickle size without changing the decoder inputs semantically."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept; no thresholds are applied to any individual QC metric (`presence_ratio`, `isi_violation`, `amplitude_cutoff`, `drift_metric`, …), and the older `units/unit_quality` label is not used. A session with zero good units is dropped. 69,453 of 272,227 units survive (25.5%), mean 401.5 per session.

ii.
```python
classification = read_str_array(nwb["units/classification"])
good_unit_idx = np.flatnonzero(classification == "good")
n_units_raw = classification.shape[0]
if good_unit_idx.size == 0:
    print(f"  Skipping {session_id}: no good units in NWB classification.")
    return None
```

iii. CONVERSION_NOTES Key decision 2: "Use NWB `units.classification == "good"` as the primary good-unit filter. Rationale: the NWB field description explicitly says this is the paper's single-unit classification label; it is the closest equivalent to the reference `goodunits` lists." Step 4 documents the residual gap against the papers: "Papers report `69943` good units / NWB `classification == "good"` totals `69453`… Most likely explanation is archive/version mismatch: code README cites DANDI `0.231012.2129`, local data are `0.230822.0128`."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams are on one session-absolute clock, so alignment is a lookup: the fixed relative edge grid is added to each trial's `go_start_times` timestamp and the spikes are binned against those absolute edges. No resampling, interpolation or per-stream offset.

ii.
```python
go_start = np.asarray(
    nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:], dtype=np.float64,
)
...
keep_go_start = go_start[keep_idx]
```

```python
abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]
```

iii. CONVERSION_NOTES Step 4: "Align all streams by subtracting per-trial go-cue onset, matching the reference code and decoder task." Step 10 reports raw-vs-converted `np.allclose` spot checks of the neural tensor on three sessions, all passing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins over `[-2.5 s, +1.5 s)` relative to the go cue = exactly **80 bins per trial** for every trial in every session. The grid is built once at module level and reused. Spikes are binned straight from spike times at this resolution, so there is no rebinning of an intermediate representation. The reference codebase's 40 ms / 3.4 ms stride settings were deliberately overridden.

ii.
```python
BIN_SIZE_S = 0.05
OFF_START_S = -2.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))          # 80
BIN_EDGES_REL = OFF_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
```

```python
"time_bin_size": 50.0,
"temporal_alignment_event": "go cue onset",
"off_start": OFF_START_S,
"off_end": OFF_END_S,
```

iii. CONVERSION_NOTES Step 1: "For the current decoder task I will preserve the same source loading, QC logic, and go-cue alignment where applicable, but adapt the neural binning/window to the required `50 ms` bins and `[-2.5, +1.5] s` interval." Step 10 Check 3: "my script changes the bin width/window to the decoder specification (`50 ms`, `[-2.5, 1.5)`), which is an intentional task-driven difference."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the tone/sample-epoch onsets), together with the trial's go cue and `intervals/trials/start_time`. For each trial the AI takes the **last** `sample_start` falling in `[trial_start, go_start]`; a trial with no such event is dropped.

ii.
```python
sample_start_times = np.asarray(
    nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:], dtype=np.float64,
)
sample_start, sample_valid = get_last_events_within(sample_start_times, trial_start, go_start)
```

```python
def get_last_events_within(event_times, lower_bounds, upper_bounds):
    """Return the last event in [lower_bounds, upper_bounds] for each interval."""
    out = np.full(lower_bounds.shape, np.nan, dtype=np.float64)
    idx = np.searchsorted(event_times, upper_bounds, side="right") - 1
    valid = idx >= 0
    ...
    valid &= candidate_times >= lower_bounds
    out[valid] = candidate_times[valid]
    return out, valid
```

iii. CONVERSION_NOTES Key decision 7: "Derive tone onset from event timestamps, not a fixed `-1.85 s`. Rationale: raw NWB trials can contain replay structure; some trials have sample onset substantially earlier than `-1.85 s`." Trajectory step 138: "`sample_start_times` is **not** always a fixed `-1.85 s` from go because some trials replay parts of the task after early licks. So the tone-onset input has to come from the NWB event stream on a trial-by-trial basis."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone time is expressed relative to the go cue (`sample_start_rel = sample_start − go`), and each bin's value is `bin_center − sample_start_rel`, i.e. seconds elapsed from the tone at that bin's centre. Stored as `float32`, continuous and time-varying. Observed range over the full dataset: `[-1.5, 11.9]` s (the long tail comes from replayed sample epochs).

ii.
```python
sample_start_rel = sample_start[keep_idx] - keep_go_start
...
def make_tone_time_matrix(sample_start_rel: np.ndarray) -> np.ndarray:
    return (BIN_CENTERS_REL[None, :] - sample_start_rel[:, None]).astype(np.float32)
```

iii. Step 5 mapping table: "Compute `time_from_tone_onset = bin_center - (sample_start - go_time)` for each neural bin… Must be trial-specific because replays can make tone onset earlier than `-1.85 s`."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on the identical go-cue-relative grid that defines the neural bins: `BIN_CENTERS_REL` are the centres of the same `BIN_EDGES_REL` used for the spike histogram, so input bin *k* covers exactly the interval of neural bin *k*.

ii.
```python
BIN_EDGES_REL = OFF_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
```

```python
abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]       # neural
...
return (BIN_CENTERS_REL[None, :] - sample_start_rel[:, None])  # input
```

iii. CONVERSION_NOTES Step 7: "Tone-onset timing is centered at the expected `-1.85 s` but includes an earlier replay tail down to about `-4.25 s`, confirming that trial-specific event extraction is necessary." Step 10 sanity check 2 verified converted `time_from_tone_onset` against raw events on three trials with `np.allclose`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The **event streams** `acquisition/BehavioralEvents/photostim_start_times` and `photostim_stop_times`, not the trials-table columns `photostim_onset` / `photostim_duration`. For each trial the AI takes the first start and the last stop within `[trial_start, trial_stop]` and expresses both relative to the go cue. Trials with no stim event keep NaN bounds.

ii.
```python
photostim_start_times = np.asarray(
    nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"][:], dtype=np.float64,
)
photostim_stop_times = np.asarray(
    nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"][:], dtype=np.float64,
)
stim_start, _ = get_first_events_within(photostim_start_times, trial_start, trial_stop)
stim_stop, _ = get_last_events_within(photostim_stop_times, trial_start, trial_stop)
...
stim_start_rel = stim_start[keep_idx] - keep_go_start
stim_stop_rel = stim_stop[keep_idx] - keep_go_start
```

iii. CONVERSION_NOTES Step 4: "NWB stores trial-level photostim columns plus absolute event timestamps for photostim start/stop… Reconstruct time-varying photostim from NWB event timestamps relative to trial go cue, which is equivalent in content to reference code after alignment." Step 5 notes "Event counts are 0 or 1 start/stop pair per trial in sampled sessions."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (`float32` 0/1) time series: a bin is 1 if its centre lies in `[stim_start_rel, stim_stop_rel)`. Trials with NaN bounds are left at 0 by an explicit `np.isfinite` mask. 20.0% of retained trials carry stimulation (20.7% restricted to opto sessions).

ii.
```python
def make_photostim_matrix(stim_start_rel, stim_stop_rel):
    mat = np.zeros((stim_start_rel.shape[0], N_BINS), dtype=np.float32)
    valid = np.isfinite(stim_start_rel) & np.isfinite(stim_stop_rel)
    if np.any(valid):
        start = stim_start_rel[valid][:, None]
        stop = stim_stop_rel[valid][:, None]
        active = (BIN_CENTERS_REL[None, :] >= start) & (BIN_CENTERS_REL[None, :] < stop)
        mat[valid] = active.astype(np.float32)
    return mat
```

iii. Step 5 mapping: "Binary vector per trial/bin: `1` if bin center is within the photostim interval relative to go cue, else `0`", following the instruction that a time-varying discrete input be a binary series. Step 7 plot review: "photostim occupies late-delay bins and never extends past the go cue in the sample sessions", matching the papers' statement that late-delay photoinhibition ends before the go cue.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stim onset and offset are converted to go-cue-relative seconds and compared against the same `BIN_CENTERS_REL` grid used for the neural bins, so no separate alignment step is needed.

ii.
```python
stim_start_rel = stim_start[keep_idx] - keep_go_start
stim_stop_rel = stim_stop[keep_idx] - keep_go_start
...
active = (BIN_CENTERS_REL[None, :] >= start) & (BIN_CENTERS_REL[None, :] < stop)
```

iii. Step 4: "Align all streams by subtracting per-trial go-cue onset, matching the reference code and decoder task." Step 5 planned sanity check: "for at least 3 stim and 3 non-stim trials, verify the binary input vector against raw photostim start/stop timestamps" — reported passing in Step 10.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Not from the trials table. Choice is derived from the raw lick event streams `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times`, evaluated in the answer window `[go_start, min(go_start + 1.5 s, trial_stop)]`. The NWB `go_stop_times` field was explicitly rejected as the window end.

ii.
```python
response_stop = np.minimum(go_start + 1.5, trial_stop)
...
left_lick_times = np.asarray(
    nwb["acquisition/BehavioralEvents/left_lick_times/timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(
    nwb["acquisition/BehavioralEvents/right_lick_times/timestamps"][:], dtype=np.float64)
choice = derive_choice_labels(
    left_lick_times, right_lick_times, keep_go_start, keep_response_stop,
)
```

iii. CONVERSION_NOTES Key decision 8: "Derive choice from the first post-go lick in the paper-defined `1.5 s` answer window rather than only from `outcome + instruction` or the raw NWB `go_stop_times`. Rationale: it uses the most direct behavioral measurement and avoids a real NWB inconsistency where `go_stop_times` can mark only the short go-cue duration in later sessions." Step 4: "early sessions use `go_stop - go_start = 1.5 s`, but later sessions often use about `0.05 s`." Step 9 documents the fix on `sub-480928_ses-20210128T133949`, which went from a pathological all-`no lick` distribution to `[0.289, 0.252, 0.460]`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The first left lick and the first right lick in the answer window are found; whichever is earlier sets the label (`0 = left`, `1 = right`); if neither port is licked the trial is `2 = no lick`. The per-trial value is broadcast across all 80 bins and stored as `int8` in row 0 of the output array. Full-dataset distribution `[left 0.430, right 0.421, no lick 0.149]`, with `no lick` tracking `ignore = 0.149`.

ii.
```python
def derive_choice_labels(left_lick_times, right_lick_times, go_start, go_stop):
    left_first, left_valid = get_first_events_within(left_lick_times, go_start, go_stop)
    right_first, right_valid = get_first_events_within(right_lick_times, go_start, go_stop)
    choice = np.full(go_start.shape, 2, dtype=np.int64)  # 2 = no lick
    left_only = left_valid & ~right_valid
    right_only = right_valid & ~left_valid
    both = left_valid & right_valid
    choice[left_only] = 0
    choice[right_only] = 1
    choice[both] = (right_first[both] < left_first[both]).astype(np.int64)
    return choice
```

```python
choice_2d = np.broadcast_to(choice[:, None], (choice.shape[0], N_BINS))
...
output_trials = [
    np.vstack((choice_2d[i], outcome_2d[i], early_2d[i], tongue_bins[i])).astype(np.int8, copy=False)
    for i in range(keep_idx.shape[0])
]
```

iii. Key decision 6: "Store all outputs as time-varying `(n_output, T)` arrays, with per-trial labels broadcast across time… each trial must have a single output array; mixing 1D and 2D outputs inside one trial is not possible." Step 10 quantifies residual disagreement with the trials-table labels: "fraction of hit trials labeled `no lick`: `0.00079`; fraction of ignore trials labeled as a lick choice: `0.00229`. These are very small and consistent with rare edge-case behavioral/event mismatches rather than a systematic bug."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `intervals/trials/outcome` column, which already contains exactly the three strings `hit`, `miss`, `ignore`.

ii.
```python
outcome_raw = read_str_array(trials["outcome"])[keep_idx]
```

iii. Step 2 verified the column's value set across the whole archive: "`outcome`: `hit`, `miss`, `ignore`" with counts `65254 / 15641 / 14095`. Step 5: "Use explicit three-way task labels required by decoder."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the strings to `0 = ignore, 1 = miss, 2 = hit` — the order given in the instructions — and the per-trial value is broadcast across all 80 bins into row 1 of the `int8` output array. Full-dataset distribution `[ignore 0.149, miss 0.166, hit 0.685]`.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_raw], dtype=np.int64)
...
outcome_2d = np.broadcast_to(outcome[:, None], (outcome.shape[0], N_BINS))
```

```python
"output_values": [
    ["left", "right", "no lick"],
    ["ignore", "miss", "hit"],
    ...
]
```

iii. The mapping order follows the Decoder Task spec ("Outcome (ignore, miss, hit)"). The dictionary lookup will raise `KeyError` on any unexpected string, which the AI relied on as an implicit assertion after having enumerated the value set in Step 2. Step 9 cross-checks the derived control-trial correct rate (0.816, restricted to `no early`, no photostim, responded trials) against the paper's 84%.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `intervals/trials/early_lick` column (`'early'` / `'no early'`).

ii.
```python
early_raw = read_str_array(trials["early_lick"])[keep_idx]
```

iii. Step 2 enumerated the column: "`early_lick`: `early`, `no early`" (`10805` / `84185`). Step 5: "Keep these trials instead of excluding them, because early lick is a decoder target" — an explicit departure from the data paper, which excluded early-lick trials from its analyses.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed dictionary maps to `0 = no, 1 = yes`, broadcast across all 80 bins into row 2 of the `int8` output array. Full-dataset distribution `[no 0.885, yes 0.115]`.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int64)
...
early_2d = np.broadcast_to(early[:, None], (early.shape[0], N_BINS))
```

iii. Coding follows the instruction order ("Early lick (no, yes, per-trial)"). It is a per-trial flag, so it is broadcast like the other per-trial outputs; the lick that sets the flag occurs in the sample/delay epoch and therefore inside the −2.5 s window.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` and whose `timestamps` run at ~294 Hz. Column 1 is the y-coordinate and column 2 the DeepLabCut likelihood. (The series' own `description` attribute states the layout as `('tongue_x', 'tongue_y', 'tongue_likelihood')`; the AI inferred the same layout from the shape and the likelihood's bimodality rather than quoting the attribute.)

ii.
```python
tongue_data = np.asarray(
    nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][:], dtype=np.float64)
tongue_times = np.asarray(
    nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][:], dtype=np.float64)
tongue_y = tongue_data[:, 1]
tongue_lik = tongue_data[:, 2]
```

iii. Step 2: "each tracking sample has shape `(n_frames, 3)`, consistent with `x`, `y`, and a confidence/likelihood-like channel… All `174` sessions contain the three side-camera tracking streams." Step 4 records that the method paper's 105-session video subset is not reproduced because "Decoder task only needs tongue y from side-camera tracking, not the exact video-modeling subset."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two steps. (1) **Visibility**: a frame counts as visible only if `y` and likelihood are finite and `likelihood >= 0.9`. (2) **Per-bin sampling**: for each 50 ms bin, the AI takes the **single last frame** whose timestamp falls in `[bin_start, bin_end)` and uses that frame's `y` and likelihood. No averaging over the ~15 frames per bin is performed. A bin with no frame, or whose last frame is not visible, becomes class `3 = not visible`. Full-dataset distribution `[lt40 0.062, 40to60 0.032, gt60 0.065, not visible 0.840]`.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
...
session_visible = (
    np.isfinite(tongue_y) & np.isfinite(tongue_lik) & (tongue_lik >= VISIBILITY_THRESHOLD)
)
```

```python
idx_last = np.searchsorted(frame_times, abs_ends.ravel(), side="left").reshape(
    go_times.shape[0], N_BINS
) - 1
clipped = np.clip(idx_last, 0, frame_times.size - 1)
last_times = frame_times[clipped]
last_y = tongue_y[clipped]
last_lik = tongue_likelihood[clipped]

in_bin = (idx_last >= 0) & (last_times >= abs_starts)
visible = (
    in_bin & np.isfinite(last_y) & np.isfinite(last_lik) & (last_lik >= VISIBILITY_THRESHOLD)
)
```

iii. CONVERSION_NOTES Key decision 9: "Use a high DeepLabCut-likelihood threshold (`>= 0.9`) to define visibility. Rationale: tongue likelihood is strongly bimodal (near zero vs `1.0`), so the exact high threshold has little effect and a strict threshold avoids false visibility." The last-frame rule is attributed in the Step 5 mapping table to "alignment idea from `align_markers.py`", the reference routine that "per bin uses the last frame in `[t-dt, t)`" (Step 1 table).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are computed over the y-values of **all visible frames in that session** (the whole recording, not just trial windows). A bin's last-frame y is then assigned `0` if `< q40`, `1` if `q40 <= y <= q60`, `2` if `> q60`; non-visible bins get `3`. If a session has no visible frames at all, `q40`/`q60` are NaN, all comparisons are False and every bin falls to `3`.

ii.
```python
if np.any(session_visible):
    q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
else:
    q40 = np.nan
    q60 = np.nan
```

```python
out = np.full((go_times.shape[0], N_BINS), 3, dtype=np.int64)
if np.any(visible):
    out[visible & (last_y < q40)] = 0
    out[visible & (last_y >= q40) & (last_y <= q60)] = 1
    out[visible & (last_y > q60)] = 2
```

```python
"output_values": [..., ["lt40", "40to60", "gt60", "not visible"]],
```

iii. Key decision 10: "Compute 40th/60th percentiles from all visible tongue-y frames in the full session before trial binning. Rationale: the task says 'over the session'; using all visible session frames is the most literal interpretation." Step 7 plot review: "visible samples spread cleanly around the session-specific 40th/60th percentile thresholds." The realised split among visible bins (0.062 / 0.032 / 0.065 → 39% / 20% / 41%) confirms the edges are self-consistent with the sampled quantity.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the session-absolute clock with spikes and events, so the same go-cue-relative grid is used: absolute bin starts/ends are formed as `go + BIN_EDGES_REL`, and a vectorised `searchsorted` over the camera timestamps locates the last frame in each of the 80 bins for all trials at once. Bin *k* of the tongue output therefore covers the same interval as bin *k* of the firing rates. Because the video is trial-gated, leading bins of trials whose go cue is <2.5 s after trial start contain no frames and fall to `not visible`.

ii.
```python
abs_starts = go_times[:, None] + BIN_EDGES_REL[:-1][None, :]
abs_ends = go_times[:, None] + BIN_EDGES_REL[1:][None, :]

idx_last = np.searchsorted(frame_times, abs_ends.ravel(), side="left").reshape(
    go_times.shape[0], N_BINS
) - 1
...
in_bin = (idx_last >= 0) & (last_times >= abs_starts)
```

iii. Step 4: "Align all streams by subtracting per-trial go-cue onset." Step 5 planned check: "for at least 3 trials and multiple bins, verify visibility class and percentile bin from raw tongue `y` and likelihood values" — Step 10 reports `np.allclose(raw_output, converted_output) == True` for three reconstructed trials across three sessions.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases, each handled explicitly:
- **Session never quality-controlled** (`classification` all `nan`): string-decoded to `"nan"`, no unit matches `'good'`, session dropped.
- **Behaviour extending past the spike recording**: trials outside the shared `obs_intervals` support are dropped; any residual all-zero neural trial is dropped.
- **Missing/invalid task events**: trials with no `sample_start` in `[trial_start, go]` are dropped; a go-cue count mismatch raises.
- **Inconsistent `go_stop_times`**: ignored; the answer window is rebuilt as `min(go + 1.5, trial_stop)`. A length mismatch also triggers a fallback.
- **Missing histology annotation**: `anno_name` empty/`nan` falls back to the probe-insertion region parsed from the electrode `location` JSON, else the literal `"unknown"`.
- **Untracked tongue**: non-finite or low-likelihood frames are treated as not visible → class `3`; a session with no visible frames yields NaN percentiles and all-`3` output.

ii.
```python
def normalize_region_name(name: str) -> str:
    name = str(name).strip()
    if not name or name.lower() == "nan":
        return ""
    return " ".join(name.split())
```

```python
for unit_i in good_unit_idx:
    region = normalize_region_name(anno_name[unit_i])
    if not region:
        start_idx = unit_electrode_start[unit_i]
        if start_idx < unit_electrodes.shape[0]:
            region = insertion_regions[unit_electrodes[start_idx]]
    unit_region_names.append(region or "unknown")
```

```python
if np.any(session_visible):
    q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
else:
    q40 = np.nan
    q60 = np.nan
```

iii. Step 4 and Step 10 "Issues Found and Resolved": "**Behavior trials extending beyond usable spike support**: Fixed by requiring the full decoder window to lie inside the shared `units/obs_intervals` support and dropping any residual all-zero-neural trial. **Inconsistent `go_stop_times` semantics**: Fixed by deriving choice from the first lick in `[go_start, min(go_start + 1.5 s, trial_stop)]`." Key decision 11 covers the region fallback: "the NWB release does not provide an authoritative coarse-group mapping equivalent to the paper's MATLAB QC files, and preserving the actual unit annotation is more defensible than hand-crafted coarse remapping."

## 10-a. What are the most time-consuming steps of the code?

i. The script instruments every block. Over the full run (173 kept sessions, 184.3 s total, 1.07 s/session): **spike reading + binning dominates at 138.4 s (75%)**, tongue alignment 11.3 s (6%), behavioural labels 1.5 s (<1%); the remaining ~35 s is per-session overhead (full `units/obs_intervals` read, string-column decoding, region lookup) plus pickling the 5.98 GB output. Within `bin_spikes_for_session` the cost is the ragged per-unit HDF5 slice read plus one `searchsorted` over 81 × n_trials edges per unit.

ii.
```python
t_spike = now()
neural = bin_spikes_for_session(...)
print(f"  Neural binning: {now() - t_spike:.2f}s")
...
print(f"  Behavioral labels: {now() - t_labels:.2f}s")
...
print(f"  Tongue alignment: {now() - t_tongue:.2f}s")
...
print(f"Total conversion time: {total_seconds:.2f}s")
print(f"Average time per kept session: {total_seconds / max(1, len(processed_sessions)):.2f}s")
```

iii. Step 6: "Spike binning and tongue alignment are the main expected bottlenecks." Step 9: "The original sample-based runtime estimate (`~99.5 s`) under-predicted the full run because later sessions have much larger neuron-by-trial products than the first two sample sessions. Actual full conversion time was `184.30 s`, still comfortably below the `15 min` threshold." No further optimisation was pursued once the budget was met.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops remain, of which one is essentially irreducible and two are genuinely vectorisable:
- `bin_spikes_for_session`'s **per-unit** loop (irreducible: `spike_times` is ragged, so each unit needs its own sorted search; the per-trial dimension is already flattened into one `searchsorted`).
- `get_common_observation_window`'s **per-good-unit** loop, which slices `obs_intervals` and takes a min/max for each of up to ~900 units per session. This could be a single `np.minimum.reduceat` / `np.maximum.reduceat` over the ragged index, and the full `obs_intervals[:]` read could be avoided.
- The **per-unit region-name** loop and `read_str_array`'s per-element decode loop over up to 3,191 unit labels per session.

The per-trial list comprehensions that build `neural_trials` / `input_trials` / `output_trials` are dictated by the target format. Notably, the tongue per-trial loop *was* vectorised into a single `searchsorted` over all trials and bins.

ii.
```python
for out_i, unit_i in enumerate(good_unit_idx):
    spikes = np.asarray(spike_times_dataset[unit_starts[unit_i] : spike_index[unit_i]], ...)
    counts = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(...)
```

```python
for out_i, unit_i in enumerate(good_unit_idx):
    start = obs_starts[unit_i]
    stop = obs_intervals_index[unit_i]
    unit_obs = np.asarray(obs_intervals[start:stop], dtype=np.float64)
    ...
    good_starts[out_i] = float(np.min(unit_obs[:, 0]))
    good_ends[out_i] = float(np.max(unit_obs[:, 1]))
```

```python
def read_str_array(dataset) -> np.ndarray:
    return np.array([as_str(x) for x in dataset[:]], dtype=object)
```

iii. Step 6 "Code speedups added": "Vectorized spike binning per unit using `np.searchsorted` over all trial bin edges at once. Vectorized tongue-frame alignment using `np.searchsorted`… Filtered out observation-unsupported trials before most downstream processing, which avoids wasted work on unusable late-session trials." The AI did not flag the `get_common_observation_window` or string-decode loops, but at 184 s total they were never a binding constraint.

## 10-c. What processing does the code repeat multiple times?

i. Nothing substantive is recomputed: each NWB file is opened once inside a `with` block, the bin grid is built once at module level, and every derived quantity is computed once. The minor redundancies are: (a) `sample_start`, `stim_start`, `stim_stop` are computed for **all raw trials** and only then indexed by `keep_idx`, so the event alignment is done for trials that are subsequently dropped (up to 320 of 480 in the worst session); (b) `read_str_array` decodes whole string columns before subsetting; (c) `neural[i].astype(np.float16, copy=False)` re-casts an array that is already `float16` (a no-op); (d) `unit_start_indices` is re-derived for each of the three ragged index arrays.

ii.
```python
sample_start, sample_valid = get_last_events_within(sample_start_times, trial_start, go_start)
...
stim_start, _ = get_first_events_within(photostim_start_times, trial_start, trial_stop)
stim_stop, _ = get_last_events_within(photostim_stop_times, trial_start, trial_stop)
...
keep_idx = np.flatnonzero(keep_trials)
sample_start_rel = sample_start[keep_idx] - keep_go_start
```

```python
neural_trials = [neural[i].astype(np.float16, copy=False) for i in range(keep_idx.shape[0])]
```

iii. Step 6 records the intent: "Reused precomputed bin edges/centers and session-wide event arrays" and "Filtered out observation-unsupported trials before most downstream processing." The filter is applied before the expensive spike binning and tongue alignment, so the cheap event alignments being done on all raw trials is deliberate ordering rather than an oversight.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of small items, all negligible relative to the 184 s runtime:
- **`go_stop_times` is read from disk, conditionally overwritten, and then never used** — `derive_choice_labels` is called with `keep_response_stop`, computed independently. This is a dead load and a dead branch.
- **`units/is_good_trials` is opened only to record its `.shape`** into `session_stats`, after the AI decided not to use the field.
- **`load_insertion_regions` parses the JSON `location` string for every electrode of every session unconditionally**, even though it is only consulted for good units whose `anno_name` is missing.
- **`tongue_y_last` and `tongue_visible` are always returned** from `align_tongue_bins` but are only consumed by the optional `--show-processing` plot.
- `make_processing_plot` contains a dead `if False` expression.
- Reading the entire `units/obs_intervals` dataset when only per-unit endpoints are needed.

ii.
```python
go_stop = np.asarray(
    nwb["acquisition/BehavioralEvents/go_stop_times/timestamps"][:], dtype=np.float64,
)
if go_stop.shape[0] != n_trials_raw:
    go_stop = np.minimum(go_start + 1.5, trial_stop)
response_stop = np.minimum(go_start + 1.5, trial_stop)      # go_stop never used again
```

```python
is_good_trials_shape = tuple(nwb["units/is_good_trials"].shape)
...
"is_good_trials_shape": is_good_trials_shape,
```

```python
trial_idx = choose_example_trial(tongue_visible, input_trials[0][1][None, :] if False else np.stack([x[1] for x in input_trials]))
```

iii. These are residue of the investigation documented in Step 4/Step 10: `go_stop_times` was loaded before the AI discovered that "`go_stop_times` are not consistent across sessions: early sessions use `go_stop - go_start = 1.5 s`, but later sessions often use about `0.05 s`" and switched to the reconstructed answer window, and `is_good_trials` was retained as a diagnostic after Key decision 4 ruled it out ("in this NWB release the field is not consistently aligned to the trial table across sessions"). The AI kept the reads for provenance in `session_info` rather than removing them.
