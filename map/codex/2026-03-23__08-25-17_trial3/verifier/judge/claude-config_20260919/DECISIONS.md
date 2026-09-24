# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is a DANDI-style NWB release with one `.nwb` file per session under `data/sub-<subject_id>/`. The AI discovers all sessions with a single sorted glob over that layout, then opens every file once with **`h5py`** (raw HDF5) rather than `pynwb`, reading the NWB groups directly (`intervals/trials`, `units`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`). Each file is processed in one pass by `process_session()`, and the per-session results are stitched together by `build_dataset()`.

Note that discovery itself opens **every** NWB file a first time (`get_nwb_files`) only to test whether the session has any `classification == 'good'` unit; the surviving files are then opened a second time for actual processing.

ii.
```python
DATA_DIR = Path("data")

def get_nwb_files(sample_only: bool) -> list[Path]:
    files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
    valid = []
    for path in files:
        with h5py.File(path, "r") as f:
            good = decode_str_array(f["units/classification"]) == "good"
            if np.any(good):
                valid.append(path)
    if sample_only:
        return valid[:SAMPLE_SESSION_COUNT]
    return valid
```

```python
def process_session(path: Path) -> SessionResult:
    ...
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        n_trials_raw = len(trials["id"])
        classification = decode_str_array(f["units/classification"])
        ...
        go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
```

```python
    for i, path in enumerate(files, start=1):
        print(f"[progress] {i}/{len(files)} {path.name}")
        result = process_session(path)
        results.append(result)
```

iii. From CONVERSION_NOTES.md Step 2 and Step 6: the AI verified the directory layout is `28` subject folders / `174` NWB files, matching `dandiset.yaml`, so the glob is the complete set of sessions and sorting makes session order deterministic. It chose `h5py` over `pynwb` "for lower overhead and explicit access to ragged arrays" (Step 6). Step 10 records that the reference code instead loads author-exported `.mat` session structures, and that the conversion "loads raw NWB files directly with `h5py` and reconstructs the same session/trial organization from source event and unit tables".

## 1-b. How are the data split into subjects?

i. One subject per `sub-*` directory. The AI takes the subject identifier from the **parent folder name** (`path.parent.name`, e.g. `'sub-440956'`) rather than from `nwb.subject.subject_id` (`'440956'`). In `build_dataset`, subjects are accumulated in first-encounter order into `subjects`, and `subject_idx` stores each session's index into that list. Because the file list is sorted, first-encounter order is alphabetical by subject. Result: 28 subjects, 3–10 sessions each.

ii.
```python
    session_id = path.stem
    subject_id = path.parent.name
```

```python
    for session_idx, result in enumerate(results):
        if result.subject_id not in subject_to_idx:
            subject_to_idx[result.subject_id] = len(subjects)
            subjects.append(result.subject_id)
        data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. Step 5 (Variable Mapping) states: "Use exact subject IDs (`sub-xxxxx`) and map each session to its subject index"; "Session order will follow sorted NWB file paths." The AI confirmed in Step 2 that the folder layout is `one folder per subject` and that the 28 folders match the `28` mice reported by the papers and `dandiset.yaml`.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is inferred. The session identifier is the file stem (`sub-440956_ses-20190207T120657_behavior+ecephys+ogen`), not `nwb.identifier`. Session order follows the sorted file list, which (because the filename embeds the acquisition timestamp) is chronological within each subject. One session is dropped: `sub-440958_ses-20190216T162508_...`, which has zero `classification == 'good'` units, giving the 173 sessions the papers report.

ii.
```python
    files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```
```python
    session_id = path.stem
```
```python
ZERO_GOOD_SESSION = "sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb"
...
            "session_filter": (
                "Included sessions with at least one good unit; excluded the single raw session with zero good units."
            ),
            "excluded_sessions": [ZERO_GOOD_SESSION],
            ...
            "session_ids": [r.session_id for r in results],
```

iii. Step 4 "Discrepancies Found" documents the reasoning: the local release has 174 raw sessions while the papers consistently report `173 behavioral sessions`; "Exactly one raw session (`sub-440958_ses-20190216T162508_...`) has zero units with `classification == good`. Excluding that session yields `173` sessions, matching the papers." The AI cross-checked this with probe-insertion counts (659 raw insertions; 655 after excluding that session, matching the STAR Methods' `655 probe insertions`).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial, and the alignment anchor is `acquisition/BehavioralEvents/go_start_times`, which the AI verified has exactly one event per trial in all 174 sessions. The code asserts this equality and raises if it fails. `sample_start_times` / `delay_start_times` are explicitly *not* used as per-trial markers because early licks replay those epochs and produce extra events.

ii.
```python
        trials = f["intervals/trials"]
        n_trials_raw = len(trials["id"])
        ...
        go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
        if len(go_times_all) != n_trials_raw:
            raise ValueError(f"{session_id}: go cue count {len(go_times_all)} does not match trial count {n_trials_raw}")
```

iii. Step 4: "In NWB, `go_start_times` matches trial count exactly, but `sample_start_times` differs in `173/174` sessions and `delay_start_times` differs in `174/174` sessions because replayed epochs are stored explicitly … Use `go_start_times` as the unique per-trial alignment anchor." Step 5, Key Decision 4: "Use `go_start_times` as the unique per-trial alignment event. Do not assume one sample/delay event per trial because replayed epochs are explicitly stored in NWB."

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both motivated as "does this trial have neural data?":

1. **`obs_intervals` containment**: a trial is kept only if its **entire** `[go − 2.5 s, go + 1.5 s]` window is contained inside a *single* observation interval of the **first good unit** (`compute_valid_trial_mask`).
2. **All-zero guard**: after binning, any trial with zero spikes across *all* good units in the window is dropped.

A session raising "no valid trials" aborts; no explicit ≥2-trial rule is coded (all retained sessions happened to have ≥137 trials). No behavioural filter is applied: early-lick, ignore, miss, auto-water and photostim trials are deliberately retained. `free_water` is never referenced in the code. Net result: **73,910 of 94,990 trials retained (77.8%)** — 76,033 passed the `obs_intervals` filter and 2,123 more were dropped as all-zero.

An important consequence the AI did not detect: `obs_intervals` in these files is one interval **per trial**, equal to `[trials.start_time, trials.stop_time]` — it is *not* a record of continuous ephys coverage. Requiring `go + 1.5 s ≤ interval_end` therefore rejects every trial that *ended* less than 1.5 s after the go cue, i.e. trials with a short post-go epoch. Sampling 7 sessions, retention is 97.6% of hit trials and 95.2% of ignore trials but only **6.8% of miss trials**; the dropped trials contain thousands of spikes each (e.g. session `sub-440956_ses-20190207T120657`, trials 15–20, outcome `miss`, ~6,000 spikes in the window, all discarded). Across the full dataset `miss` falls from ~16.5% of raw trials to **0.9%** of converted trials.

ii.
```python
def compute_valid_trial_mask(
    obs_intervals: np.ndarray,
    go_times: np.ndarray,
) -> np.ndarray:
    trial_start = go_times + REL_START_S
    trial_end = go_times + REL_END_S
    starts = obs_intervals[:, 0][None, :]
    ends = obs_intervals[:, 1][None, :]
    return np.any((trial_start[:, None] >= starts) & (trial_end[:, None] <= ends), axis=1)
```
```python
        obs_intervals_index = f["units/obs_intervals_index"][()]
        obs_intervals = get_ragged_row(
            f["units/obs_intervals"],
            obs_intervals_index,
            int(good_unit_indices[0]),
        )
        valid_trial_mask = compute_valid_trial_mask(obs_intervals=obs_intervals, go_times=go_times_all)
        if not np.any(valid_trial_mask):
            raise ValueError(f"{session_id}: no valid trials found from obs_intervals")
```
```python
        nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
        if not np.all(nonzero_trial_mask):
            firing_rates = firing_rates[:, nonzero_trial_mask, :]
            start_times = start_times[nonzero_trial_mask]
            ...
            n_trials = len(go_times)
```

iii. Step 5, Key Decision 2: "Keep all trials from included sessions that have full neural coverage for the requested `[-2.5, 1.5)` go-aligned window according to `units/obs_intervals`; within that valid-coverage set, retain stimulation, early-lick, ignore, and miss trials because they are required by the decoder inputs/outputs. This intentionally differs from the paper's 'regular trial' mask, and the difference is required by the user task." Step 6 records that the filter was added because the first sample conversion produced many all-zero neural trials and verifier warnings. Step 9 justifies the residual 2,123 removals with spot-checks showing those windows are genuinely empty, and attributes the lower trials/session (427 vs the paper's 476) to "the full-window neural coverage requirement and removal of zero-spike windows". The 0.9% `miss` fraction is reported in the Step 9 table and is passed as consistent ("Converted hit fraction is close to paper reward rate given coverage filtering"); the systematic loss of error trials is never investigated.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged, session-absolute seconds) with `units/spike_times_index` for the ragged offsets, restricted to units with `units/classification == 'good'`; plus `acquisition/BehavioralEvents/go_start_times/timestamps` to place the bin edges. `units/anno_name` supplies the per-unit brain-region label.

ii.
```python
        classification = decode_str_array(f["units/classification"])
        good_mask = classification == "good"
        good_unit_indices = np.flatnonzero(good_mask)
        ...
        anno_name = decode_str_array(f["units/anno_name"])
        region_labels = anno_name[good_unit_indices]
        ...
        spike_times_flat = f["units/spike_times"][()]
        spike_times_index = f["units/spike_times_index"][()]
        trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. Step 5 mapping row for `neural`: "`units/spike_times`, `units/classification`, `acquisition/BehavioralEvents/go_start_times` → Keep only units with `classification == "good"` … bin absolute spike times into go-aligned `[-2.5, 1.5)` windows using `50 ms` bins; convert counts to firing rates in Hz." Spike times are the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz with no smoothing, normalisation or baseline subtraction. For each good unit the flat array of absolute bin edges for all trials is built once, `np.searchsorted` gives the running spike count at each edge, adjacent differences give the per-bin count, and counts are divided by the 50 ms bin width. Rates are stored as `float16` (exact here, since counts/0.05 are multiples of 20). Output per trial is `(n_good_units, 80)`.

ii.
```python
def bin_spikes_to_firing_rates(
    spike_times_flat, spike_times_index, good_unit_indices, trial_edges_abs,
) -> np.ndarray:
    n_trials = trial_edges_abs.shape[0]
    n_edges = trial_edges_abs.shape[1]
    flat_edges = trial_edges_abs.reshape(-1)
    start_index = np.concatenate(([0], spike_times_index[:-1]))
    firing_rates = np.empty((len(good_unit_indices), n_trials, n_edges - 1), dtype=np.float16)

    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
        counts = np.diff(edge_idx, axis=1)
        firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
    return firing_rates
```
```python
    neural_trials = [firing_rates[:, i, :].copy() for i in range(n_trials)]
```

iii. Step 1 identifies the reference's `sliding_histogram`, which "converts per-trial spike times into binned spike counts or firing rates". Step 5 mapping specifies "convert counts to firing rates in Hz". Step 6 records the dtype choice: "used `float16` for stored neural/input arrays and `int16` for outputs to reduce pickle size", and "Used search-sorted spike binning on per-unit spike vectors rather than per-bin Python loops". Step 10 Check 2 verified one bin against a direct raw recount (`count / 0.05 s`) with `np.allclose`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept — the output of the region-specific QC classifier described in the spike-sorting white paper. No thresholds are applied to any individual quality metric (`isi_violation`, `presence_ratio`, `amplitude_cutoff`, … are all ignored), and the older `units/unit_quality` ('good'/'multi') label is not used. A session with no good units is excluded at discovery time. Retains **69,453 of 272,227 units (25.5%)**, mean 401.5/session.

ii.
```python
        classification = decode_str_array(f["units/classification"])
        good_mask = classification == "good"
        good_unit_indices = np.flatnonzero(good_mask)
        if len(good_unit_indices) == 0:
            raise ValueError(f"Session {session_id} has no good units")
```
```python
            "unit_filter": "units/classification == 'good'",
```

iii. Step 3 documents the classifier ("Region-specific logistic-regression classifiers … The classifier outputs defined the 'good' units used in the papers"). Step 5, Key Decision 3: "Use NWB `classification == 'good'` as the closest NWB-native equivalent of the reference code's external `goodunits` files." Step 4/Step 10 note and accept the residual mismatch with the papers' `69,943` good units: "the local NWB release contains 69,453 `good` units, while the papers/code report 69,943, which appears to be a release/classifier snapshot mismatch rather than a conversion bug." The AI also used the good-unit fraction (25.5% vs the paper's `25.9% of clusters reported by Kilosort2`) as a corroborating check.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. To go-cue onset, and no alignment step is required: spike times and `go_start_times` are already on the same session-absolute clock. A fixed grid of 81 edges relative to the go cue is added to each trial's go-cue time to produce absolute edges, and spikes are binned against those edges directly. No resampling, interpolation, or per-stream offset correction.

ii.
```python
REL_START_S = -2.5
REL_END_S = 1.5
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
```
```python
        trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```
```python
            "temporal_alignment_event": "Go cue onset",
            "off_start": REL_START_S,
            "off_end": REL_END_S,
```

iii. Required by the Decoder Task ("Temporally align based on Go cue onset", "2.5 s before to 1.5 s after"). Step 4 also confirms this matches the reference: "Reference code and papers are consistently go-cue-centered … Conversion also centers everything on go cue". Step 5 Key Decision 4 selects `go_start_times` as the unique per-trial anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins per trial spanning `[-2.5 s, +1.5 s]` relative to the go cue. The grid is defined once at module level and reused for every trial and session, so every trial has exactly 80 timepoints (verified in the verification log: T min = max = 80). This is a deliberate departure from the method paper's 40 ms window / 3.4 ms stride sliding histogram; spikes are binned once at 50 ms directly from spike times, so there is no rebinning of an intermediate representation. `metadata['time_bin_size'] = 50.0` (ms).

ii.
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))   # 80
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
```
```python
            "time_bin_size": 50.0,
            ...
            "reference_processing_note": (
                "Reference session/unit curation and go-centered alignment preserved where possible; "
                "neural bins intentionally changed from reference 40 ms/3.4 ms to requested 50 ms bins"
            ),
```

iii. Step 5, Key Decision 12: "Use `50 ms` bins because the decoder task explicitly requires this, while preserving the reference session/unit curation and go-cue alignment principles." Step 4 flags it as a deliberate, documented deviation: "The method paper's `40 ms` / `3.4 ms` neural preprocessing describes the reference analysis pipeline, but the requested decoder conversion will intentionally deviate to `50 ms` bins while preserving the same alignment and curation principles where possible."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. **No raw variable.** The AI rejected `acquisition/BehavioralEvents/sample_start_times` and instead hard-codes the tone onset at a **constant −1.85 s relative to the go cue** on every trial of every session, derived from the nominal task structure (0.65 s sample epoch + 1.2 s delay). Only the bin-centre grid and that constant enter the computation; nothing trial-specific is read.

Consequence: `input[0]` is byte-identical on all 73,910 trials (verification log: range `[-0.6, 3.3]` for every session). Measured against the data, the assumption is wrong on ~12% of trials — the true `go − last sample onset` gap reaches 4.25 s — and the deviating trials are disproportionately early-lick trials (a decoder output), because an early lick replays the sample/delay epoch.

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
```
```python
        input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```
```python
            "tone_onset_definition": "Canonical sample onset at -1.85 s relative to go cue from task structure",
```

iii. Step 5, Key Decision 5: "Use the canonical tone onset at `-1.85 s` relative to go cue from the task structure (`0.65 s` sample epoch + `1.2 s` delay), because raw NWB sample-event streams contain replay-related extra events." The trajectory (steps 165–179) shows the AI measured that the last `sample_start_times` event inside each trial sits at exactly −1.85 s on 83,103/94,990 trials (87.5%), tried "pick the event nearest −1.85 s" as an alternative, and concluded: "The raw sample-event stream is too irregular to use as the sole source of tone onset … tone onset will come from the task structure itself, `-1.85 s` relative to go cue, which is what the papers describe and what the clean majority of trials support."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. One vectorised expression: each bin centre (go-cue-relative) minus the constant tone offset, i.e. `centre + 1.85`, producing the fixed ramp `[-0.625 … 3.325]` (stored `float16` as `[-0.6, 3.3]`). That single row is tiled across all trials of the session. There is no per-trial event lookup, so early-lick / replay trials receive the same values as standard trials.

ii.
```python
        input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```
```python
        "input_names": ["time_from_tone_onset_s", "photostim_on"],
```
```python
    for trial_idx in range(n_trials):
        input_trial = np.vstack([input_time[trial_idx], input_stim[trial_idx]]).astype(np.float16)
```

iii. Same justification as 3-a. Step 10's "input sanity check" verifies the vector against "the independent formula from fixed bin centers and canonical tone onset" — i.e. it re-derives the same constant rather than comparing against the raw `sample_start_times` stream, so the check cannot detect the per-trial error.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the values are computed on `REL_CENTERS`, the centres of the same 80 go-cue-relative bins used for the firing rates, so bin *k* of `input[0]` covers exactly the interval of bin *k* of `neural`. The input is emitted as a `(2, 80)` array per trial, matching the neural `(n_neurons, 80)`.

ii.
```python
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
```
```python
        trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]   # neural
        input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))  # input
```

iii. Step 5, Key Decision 6: "Both decoder inputs will be fully time-varying arrays of shape `(2, 80)` per trial." Step 4: everything is centred on the go cue, which is the single alignment anchor shared by all streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` (both stored as strings, `'N/A'` when the trial was not stimulated, and measured **relative to trial start**), together with `intervals/trials/start_time` and the trial's go-cue time to move them onto the go-cue axis. `photostim_power` and the `photostim_start_times` / `photostim_stop_times` event streams are not used.

ii.
```python
        photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
        photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
```
```python
def parse_optional_float_array(strings: np.ndarray) -> np.ndarray:
    out = np.full(strings.shape, np.nan, dtype=np.float64)
    for i, value in enumerate(strings):
        if value == "N/A":
            continue
        out[i] = float(value)
    return out
```

iii. Step 4 documents the coordinate discovery: "NWB trial table stores `photostim_onset` relative to trial start; in example trials, subtracting `(go_time - trial_start)` gives about `-1.2 s`, matching late-delay stimulation", cross-checked against the papers' statement that photoinhibition occupies the last 0.5 s of the delay and ends before the go cue. Step 2 found 168/174 sessions contain at least one non-`N/A` photostim trial.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Onsets and durations are parsed to float (`'N/A'` → NaN), converted from trial-start to go-cue coordinates as `onset − (go − start)`, and the offset is `onset_rel_go + duration`. A bin is set to 1 where its **centre** falls in `[onset, offset)` and 0 elsewhere, giving a binary time series rather than a per-trial flag. Non-stimulated trials keep NaN bounds and are skipped by the `np.isfinite` mask, so their rows stay all-zero. Stored `float16`; verified range `[0, 1]`.

ii.
```python
def build_photostim_matrix(photostim_onset_str, photostim_duration_str, start_times, go_times):
    onset_trial = parse_optional_float_array(photostim_onset_str)
    duration = parse_optional_float_array(photostim_duration_str)
    go_minus_start = go_times - start_times
    onset_rel_go = onset_trial - go_minus_start
    offset_rel_go = onset_rel_go + duration

    stim = np.zeros((len(go_times), N_BINS), dtype=np.float16)
    valid = np.isfinite(onset_rel_go) & np.isfinite(offset_rel_go)
    for trial_idx in np.where(valid)[0]:
        mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
        stim[trial_idx, mask] = 1.0
    return stim
```

iii. Step 5 mapping: "Convert photostim onset/duration from trial-start coordinates to go-centered coordinates; emit binary `0/1` per bin center for photostim on/off … Keep stimulation trials because photostim is an explicit decoder input." This also follows the instruction that a time input should be represented as a binary time series. Step 10's input sanity check re-derived the photostim vector for session `sub-440956_ses-20190207T120657`, trial 44 from the raw `photostim_onset`, `photostim_duration`, `start_time` and `go_time` and matched it with `np.allclose`.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset/offset are re-expressed relative to the go cue — the same event the neural bins are aligned on — and then compared against `REL_CENTERS`, the same bin-centre grid. So no separate alignment step is needed and bin *k* of the photostim input covers bin *k* of the firing rates. The AI's diagnostic plot histograms the go-relative onsets, which cluster near −1.2 s as the papers predict for late-delay inhibition.

ii.
```python
    go_minus_start = go_times - start_times
    onset_rel_go = onset_trial - go_minus_start
    ...
        mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
```
```python
    ax[2, 0].hist(diag["photostim_onsets_rel_go"], bins=30)
    ax[2, 0].set_title("Photostim Onset Relative to Go")
```

iii. Step 4: "Reference code stores stimulation times relative to go cue after subtracting go cue time … Convert photostimulation onset/offset from trial-start coordinates into go-aligned coordinates during conversion." Step 1 notes the reference `process_one_sess` does the same shift ("Raw lick and stimulation times are explicitly shifted into go-cue coordinates by subtracting per-trial go cue time").

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column, so choice is derived from `intervals/trials/trial_instruction` (`'left'`/`'right'`) crossed with `intervals/trials/outcome` (`'hit'`/`'miss'`/`'ignore'`): hit ⇒ instructed side, miss ⇒ opposite side. For `ignore` trials the AI additionally reads the lick event streams `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times` and applies a three-level fallback: (1) first lick in `[go, stop_time]`; (2) else first lick anywhere in `[start_time, stop_time]`; (3) else the **instructed side**.

Crucially, the AI emits only **two** classes (`left`, `right`) — there is no `no lick` class, contrary to the Decoder Task spec. Measured over a 5-session sample, 92% of ignore trials fall through to branch (3) and are labelled with the instructed side, i.e. a fabricated choice for trials on which the animal made none.

ii.
```python
def lick_choice_with_fallback(left_times, right_times, start_time, go_time, stop_time, instructed_choice) -> int:
    left_post = left_times[(left_times >= go_time) & (left_times <= stop_time)]
    right_post = right_times[(right_times >= go_time) & (right_times <= stop_time)]
    post_choice = first_side(left_post, right_post)
    if post_choice is not None:
        return post_choice

    left_any = left_times[(left_times >= start_time) & (left_times <= stop_time)]
    right_any = right_times[(right_times >= start_time) & (right_times <= stop_time)]
    any_choice = first_side(left_any, right_any)
    if any_choice is not None:
        return any_choice
    return instructed_choice
```
```python
        left_lick_times = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()].astype(np.float64)
        right_lick_times = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()].astype(np.float64)
```

iii. Step 5, Key Decision 8: "Because most ignore trials have no post-go lick (`13770 / 14095`), choice must use a documented fallback. I will use lick-derived side when available, otherwise instructed side." The mapping table adds: "for `hit`, use instructed side; for `miss`, use opposite side; for `ignore`, use first post-go lick side if present, else first lick side anywhere in trial if present, else fall back to instructed side." No justification is given anywhere in the notes or trajectory for dropping the `no lick` class that the Decoder Task specifies.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `trial_instruction` is mapped to `left = 0`, `right = 1`; hit trials take the instructed code, miss trials take `1 − instructed`, ignore trials go through `lick_choice_with_fallback`. The hit/miss branches are vectorised; the ignore branch is a Python loop that re-scans the full session lick arrays for each ignore trial. The single per-trial value is then broadcast across all 80 bins into row 0 of the `(4, 80)` output array. `output_values[0] = ['left', 'right']` (two classes). Converted distribution: `left 0.491 / right 0.509`.

ii.
```python
def build_choice_array(trial_instruction, outcome_code, start_times, go_times, stop_times,
                       left_lick_times, right_lick_times) -> np.ndarray:
    choice = np.zeros(len(trial_instruction), dtype=np.int16)
    instructed = np.where(trial_instruction == "left", 0, 1).astype(np.int16)

    hit_mask = outcome_code == 2
    miss_mask = outcome_code == 1
    ignore_mask = outcome_code == 0

    choice[hit_mask] = instructed[hit_mask]
    choice[miss_mask] = 1 - instructed[miss_mask]

    ignore_trials = np.where(ignore_mask)[0]
    for trial_idx in ignore_trials:
        choice[trial_idx] = lick_choice_with_fallback(...)
    return choice
```
```python
        "output_values": [
            ["left", "right"],
            ...
```
```python
        output_trial = np.vstack(
            [
                np.full(N_BINS, choice_code[trial_idx], dtype=np.int16),
                ...
```

iii. Step 5, Key Decision 7: "All outputs will be stored as time-varying arrays of shape `(4, 80)` per trial; choice, outcome, and early-lick labels will be repeated across time, while tongue position varies across bins." Step 10 Check 2 verified hit, miss and ignore example trials against independent recomputation of the *same* rule, and Step 10 Check 5 states "Confirmed ignore-trial choice fallback behaves as documented when there is no post-go lick." Step 12 notes choice validation accuracy (0.742) is marginally below 1.5× chance but concludes there is no conversion bug.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which already stores exactly the three strings `'ignore'`, `'miss'`, `'hit'` the task requires. No derivation from licks or reward events.

ii.
```python
        outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
        outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

iii. Step 2 confirmed by inspection that `outcome` takes exactly `hit`/`miss`/`ignore` in all sessions, with raw counts `hit=65254`, `miss=15641`, `ignore=14095`. Step 5 mapping: "`intervals/trials/outcome` → `output[1]` (`outcome`) | Map `ignore -> 0`, `miss -> 1`, `hit -> 2`". The dict lookup raises a `KeyError` on any unexpected string, so an unhandled value would fail loudly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the three strings to `0 = ignore`, `1 = miss`, `2 = hit` (matching the order given in the Decoder Task), and the per-trial code is broadcast across all 80 bins into row 1 of the output array. `output_values[1] = ['ignore', 'miss', 'hit']`.

The distribution in the converted data is severely distorted by the trial filter of 1-e: `ignore 0.173 / miss 0.009 / hit 0.818`, versus raw proportions of roughly `0.148 / 0.165 / 0.687`. The `miss` class is effectively absent, and 12 sessions contain no `miss` trials at all.

ii.
```python
        outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```
```python
        output_trial = np.vstack(
            [
                np.full(N_BINS, choice_code[trial_idx], dtype=np.int16),
                np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16),
                ...
```
```python
            ["ignore", "miss", "hit"],
```

iii. Step 5 mapping ("Constant across time bins") and Key Decision 7 (per-trial labels repeated across bins so all outputs share one `(4, 80)` array). Step 9's consistency table lists the converted outcome distribution `[0.173, 0.009, 0.818]` and judges it acceptable: "Converted hit fraction is close to paper reward rate given coverage filtering and inclusion of stimulation trials." The near-total loss of `miss` trials is visible in the table but is not flagged.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from `intervals/trials/early_lick`, which stores `'no early'` / `'early'`. No derivation from the lick event streams or from sample/delay replays.

ii.
```python
        early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
        early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
```

iii. Step 2 confirmed the column's values by inspection (`early`/`no early`, raw counts `no early=84185`, `early=10805`). Step 5 mapping cites `get_regular_trial_mask` and the datapaper methods, where the same flag drives the "regular trial" exclusion — but here early-lick trials are kept because the flag is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed dictionary maps `'no early' → 0`, `'early' → 1`, and the per-trial value is broadcast across all 80 bins into row 2 of the output array. `output_values[2] = ['no', 'yes']`. Converted distribution `no 0.885 / yes 0.115`, close to the raw 0.886/0.114. Because the early lick physically occurs during the sample or delay epoch, the event itself lies inside the −2.5 s window even though the label is per-trial.

ii.
```python
        early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
```
```python
                np.full(N_BINS, early_code[trial_idx], dtype=np.int16),
```
```python
            ["no", "yes"],
```

iii. Step 5, Key Decision 7 (per-trial labels repeated across bins). Step 10 Check 2 re-derived the early-lick vectors for three named trials directly from the raw NWB trials table and matched them exactly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps`. Column 1 (`y`) is the value; column 2 (`likelihood`) gates visibility; column 0 (`x`) is used only inside the frame-to-frame velocity outlier test. Present in all 174 sessions.

ii.
```python
        tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
        tongue_data = tongue_ts["data"][()]
        tongue_x = tongue_data[:, 0]
        tongue_y = tongue_data[:, 1]
        tongue_likelihood = tongue_data[:, 2]
        tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. Step 2 records the channel layout from the series' own description attribute (`"('tongue_x', 'tongue_y', 'tongue_likelihood')"`) rather than assuming it, and confirms "Sessions with tongue tracking: `174/174`". Step 3 notes the method paper used only side-view video and DeepLabCut markers including the tongue.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps in `clean_tongue_tracking` / `align_tongue_y`:

1. **Velocity outlier rejection**: frame-to-frame speed `sqrt(dx² + dy²)`; frames above `mean + 5·std` (or non-finite) are replaced by linear interpolation over frame index, for both x and y.
2. **Occlusion imputation**: frames with `likelihood < 0.9` are **replaced by the session's mean visible y**, not marked missing. Since the tongue is visible in only ~10% of frames, ~90% of the trace becomes a single constant.
3. **Sampling**: for each bin the value of the last camera frame at or before the bin centre (last-frame-carried-forward via `searchsorted(..., 'right') - 1`, clipped to the array bounds). No averaging within the bin.
4. **Discretisation**: see 8-c.

ii.
```python
    speed = np.zeros_like(x)
    if len(x) > 1:
        speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
    outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
    ...
        x[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], x[keep_mask])
        y[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], y[keep_mask])
    ...
    visible_mask = np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    mean_y = float(np.nanmean(y[visible_mask])) if np.any(visible_mask) else float(np.nanmean(y))
    occluded_mask = ~visible_mask
    y[occluded_mask] = mean_y
```
```python
def align_tongue_y(timestamps, cleaned_y, go_times) -> np.ndarray:
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

iii. Step 3 extracted the method paper's preprocessing: "Marker outliers were identified using a five-sigma frame-to-frame velocity rule and imputed from nearby frames" and "Tongue position was imputed to its mean when occluded before the response epoch" — the AI mirrors both rules. Step 5, Key Decision 9: "Use last-frame-carried-forward at each bin center to mirror the reference alignment scripts rather than linear interpolation" (confirmed against the reference's `align_markers_between_lims` in trajectory step 157). Metadata summarises it as "5-sigma velocity outlier interpolation, low-likelihood frames set to mean visible y, last-frame-carried-forward alignment to bin centers, per-session 40th/60th percentile discretization".

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles (`q40`, `q60`) are taken over **all aligned bin values of all retained trials**, then `y < q40 → 0`, `q40 ≤ y ≤ q60 → 1`, `y > q60 → 2`. Only **three** classes are emitted; the `3: not visible` class specified in the Decoder Task is **not** implemented, because occluded frames were already overwritten with the mean visible y in 8-b.

Because ~90% of frames are occluded and collapse onto exactly one value, `q40 == q60 == mean_y` in practice, and every imputed bin lands in class 1. The AI hit this directly (its first `np.digitize` version emptied the middle class) and worked around it with explicit `<`/`>` comparisons rather than reconsidering the imputation. The resulting distribution is `[0.082, 0.834, 0.085]` — class 1 holds 83% of bins and no longer means "between the 40th and 60th percentile"; it means "tongue not visible".

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
```
```python
        q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
        tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
        tongue_disc[aligned_tongue_y < q40] = 0
        tongue_disc[aligned_tongue_y > q60] = 2
```
```python
            ["lt_40pct", "40_to_60pct", "gt_60pct"],
```

iii. Step 5, Key Decision 10: "Compute per-session percentiles from the aligned tongue-y samples that actually enter the converted dataset, not from unrelated off-trial video periods." Step 6 documents the fix: "Initial tongue discretization used percentile edges with `np.digitize`, which can collapse the middle class when `q40 == q60` … Replaced percentile-edge discretization with explicit `< q40`, `q40..q60`, `> q60` logic so the middle tongue class remains populated." The trajectory (step 209) confirms the AI understood the tie was caused by imputation: "the 40th and 60th percentiles can tie after imputation", yet kept the imputation and dropped the `not visible` class without comment.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and go cues, so alignment is a direct lookup: the absolute time of each go-cue-relative bin centre is formed, and `searchsorted` picks the last camera frame at or before it. Bin *k* of the tongue output therefore refers to the same instant as bin *k* of the firing rates, giving a genuinely time-varying `(80,)` row 3. Sampling happens before the all-zero-trial filter and the array is masked alongside the other streams, so the trial indexing stays consistent.

One edge case is not handled explicitly: the video is trial-gated (off during the inter-trial interval), and `np.clip` makes bins that precede the first available frame reuse a stale earlier frame rather than being marked missing. In practice this is masked by the mean-imputation of 8-b.

ii.
```python
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```
```python
            aligned_tongue_y = aligned_tongue_y[nonzero_trial_mask]
```
```python
    ax[1, 1].plot(REL_CENTERS, diag["aligned_tongue_y_trial"], label="aligned y")
    ax[1, 1].step(REL_CENTERS, result.output_trials[trial_idx][3], where="mid", label="discrete class")
```

iii. Step 5, Key Decision 9 (mirror the reference's last-frame-carried-forward marker alignment). Step 10 Check 2 independently recomputed "cleaning, alignment, percentile thresholds, and discretization" for session `sub-440956_ses-20190207T120657`, trial 0 and matched the converted tongue classes. The `--show-processing` plot overlays the aligned trace and its discretised class on the go-cue-relative axis to make misalignment visible.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases:

- **Session never quality-controlled** (`classification`/`anno_name` are NaN): `decode_str_array` stringifies them to `'nan'`, so no unit matches `'good'` and `get_nwb_files` excludes the session before processing. This is exactly the one dropped session.
- **Non-stimulated trials** (`photostim_onset == 'N/A'`): parsed to NaN and skipped by an `np.isfinite` mask, so the photostim row stays all-zero (no separate branch, no crash).
- **Trials with no spike data**: removed by the post-binning all-zero guard (see 1-e). `free_water` trials — which the reference identifies as the main cause of empty trials — are never referenced by name; they are caught only incidentally by this guard.
- **Tracking outliers** (non-finite coordinates, 5σ velocity jumps): linearly interpolated from neighbouring frames; if fewer than two clean frames exist, replaced by the nanmean.
- **Occluded tongue** (`likelihood < 0.9`): imputed to the session's mean visible y (see 8-b/8-c).
- **Structural violations**: a go-cue/trial count mismatch, an empty good-unit set, or an unexpected `outcome` / `early_lick` string all raise immediately rather than being silently coerced.

ii.
```python
def decode_str_array(ds: h5py.Dataset) -> np.ndarray:
    ...
        else:
            out.append(str(x))
    return np.array(out, dtype=object)
```
```python
    valid = np.isfinite(onset_rel_go) & np.isfinite(offset_rel_go)
```
```python
    keep_mask = ~outlier_mask
    if np.sum(keep_mask) >= 2:
        x[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], x[keep_mask])
        y[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], y[keep_mask])
    else:
        x[outlier_mask] = np.nanmean(x)
        y[outlier_mask] = np.nanmean(y)
```
```python
        if len(go_times_all) != n_trials_raw:
            raise ValueError(f"{session_id}: go cue count {len(go_times_all)} does not match trial count {n_trials_raw}")
```

iii. Step 4 explains the dropped session (it is the only route to the papers' 173-session count). Step 3 sources the tracking rules from the method paper (five-sigma velocity rule; tongue imputed to its mean when occluded). Step 6 describes the all-zero guard as "a defensive guard against edge-case recording coverage mismatches", and Step 9 spot-checks that the trials it removes are genuinely empty across all good units while neighbouring trials carry ~25–33 k spikes. Step 10 Check 5 confirms "`photostim_on` handles `N/A` values as all-zero vectors" and that all converted trials have 80 bins with no NaNs.

## 10-a. What are the most time-consuming steps of the code?

i. The code instruments itself: it prints a total and a spike-binning time per session. Over the full run, 173 sessions took **430 s of 434 s total (7.24 min)**, of which spike binning accounts for **110 s (26%)**. The remaining ~74% is dominated by HDF5 reads inside `process_session` — `f["units/spike_times"][()]` pulls the entire ragged buffer (up to ~11.5 M float64, ~92 MB) including non-good units, and the tongue `data` array is ~(680 k × 3) — followed by tongue cleaning, the per-trial Python loops that build the input/output lists, and the `np.int16`/`float16` copies. Pickling the 4.6 GB result adds a few seconds. Not counted in the 434 s at all is the `get_nwb_files` pre-scan, which opens and string-decodes the `classification` column of all 174 files before the timer starts.

ii.
```python
        t_neural = time.perf_counter()
        firing_rates = bin_spikes_to_firing_rates(...)
        ...
        neural_time_s = time.perf_counter() - t_neural
```
```python
    print(
        f"[session] {session_id} - done "
        f"({n_trials} trials, {len(good_unit_indices)} good units, "
        f"{diagnostics['session_time_s']:.1f}s total, {neural_time_s:.1f}s binning)"
    )
```

iii. Step 7 estimated ~0.7 s/session from the two sample sessions and projected ~2–4.5 min for the full run; Step 9 reports the actual 7.24 min and concludes it is "below the `15 min` optimization threshold", so no further optimisation was pursued. The notes attribute the main saving to pre-filtering trials with `obs_intervals` ("Removes out-of-recording trials before spike binning; eliminates wasted work"); they do not identify HDF5 reading as the actual dominant cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python-level loops remain, all avoidable except the per-unit one:

- `build_photostim_matrix`: loops over stimulated trials to set a mask; fully vectorisable by broadcasting `REL_CENTERS[None, :]` against the onset/offset columns (which is what the reference does).
- `build_choice_array`: loops over ignore trials, and each iteration re-scans the *entire* session lick arrays with four boolean masks — O(n_ignore × n_licks). Replaceable by two `np.searchsorted` calls.
- The per-trial `input_trials` / `output_trials` construction loop: one `np.vstack` plus four `np.full` allocations per trial; could be a single pre-allocated `(n_trials, 4, 80)` array filled by broadcasting.
- `decode_str_array`: an element-wise Python loop run over seven string columns per session (`classification`, `anno_name`, `trial_instruction`, `outcome`, `early_lick`, `photostim_onset`, `photostim_duration`) — `anno_name`/`classification` alone are ~1,500–3,200 entries per session.
- `parse_optional_float_array`: element-wise Python loop over the photostim strings.
- `build_dataset`: a per-unit Python loop over 69,453 region labels to assign indices.
- `bin_spikes_to_firing_rates`: the per-unit loop is inherent — each unit has its own ragged, separately sorted spike vector — and it is already vectorised over trials by flattening the edge array, so only this one is genuinely unavoidable.

ii.
```python
    for trial_idx in np.where(valid)[0]:
        mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
        stim[trial_idx, mask] = 1.0
```
```python
    for trial_idx in ignore_trials:
        choice[trial_idx] = lick_choice_with_fallback(
            left_times=left_lick_times, right_times=right_lick_times, ...)
```
```python
    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
```

iii. Step 6 claims "Kept all heavy operations vectorized in NumPy" and "Used search-sorted spike binning on per-unit spike vectors rather than per-bin Python loops" — accurate for the spike path, but the notes do not acknowledge the photostim, choice, per-trial-assembly, string-decoding or region loops. No loop was flagged as a remaining inefficiency, because the 7.24 min total was under the 15-minute budget.

## 10-c. What processing does the code repeat multiple times?

i. Three genuine repetitions:

- **Every NWB file is opened and its `classification` column read and string-decoded twice** — once in `get_nwb_files` purely to test for good units, then again in `process_session`. That is a full extra pass over all 174 files (and the result of the first pass is discarded rather than handed to the second).
- `parse_optional_float_array(photostim_onset_str)` and `go_times - start_times` are computed inside `build_photostim_matrix` and then recomputed verbatim after the session loop to build the diagnostics.
- The whole diagnostics dictionary — preview slices of the raw and cleaned tongue traces, the aligned trace, the photostim onset histogram data — is assembled for all 173 sessions even though at most 2 are ever plotted.

The bin grid (`REL_EDGES`, `REL_CENTERS`) is correctly built once at module level and reused, and the tongue percentile thresholds are computed once per session, so those are not repeated.

ii.
```python
def get_nwb_files(sample_only: bool) -> list[Path]:
    for path in files:
        with h5py.File(path, "r") as f:
            good = decode_str_array(f["units/classification"]) == "good"
```
```python
    with h5py.File(path, "r") as f:
        ...
        classification = decode_str_array(f["units/classification"])
        good_mask = classification == "good"
```
```python
    onset_trial = parse_optional_float_array(photostim_onset_str)
    go_minus_start = go_times - start_times
    onset_rel_go = onset_trial - go_minus_start
    photostim_onsets_rel_go = onset_rel_go[np.isfinite(onset_rel_go)]
```

iii. Step 6 lists as a speed-up "Reused aligned bin-center arrays and session-level tongue thresholds instead of recomputing them per trial", and Step 7 credits threshold reuse with avoiding "repeated percentile recomputation". The duplicated file-open pass and the duplicated photostim parsing are not mentioned anywhere in the notes; the pre-scan also sits outside the timing instrumentation, so its cost never appeared in the AI's estimates.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest but real:

- **Tongue x cleaning**: `clean_tongue_tracking` interpolates the outlier frames of `x` as well as `y`, but only `y` is returned — the repaired x values are discarded immediately. (The raw `x` is legitimately needed for the speed computation; only the interpolation of it is wasted.)
- **Diagnostics for unplotted sessions**: the `diagnostics` dict (2,000-frame raw and cleaned tongue previews, the aligned trace, photostim onset array, timing and count fields) is built for every session and carried on every `SessionResult`, while only the first two sessions are ever plotted and only in `--show-processing` mode.
- **`SessionResult.region_names`** (`sorted(set(...))` per session) is computed and stored but never read — `build_dataset` uses `region_labels_per_unit` instead.
- **Firing rates for trials that are then dropped**: the all-zero guard runs *after* binning, so ~2,123 trials are fully binned across all good units and then discarded.
- **Full `spike_times` buffer read**: the entire ragged array is loaded, including the ~75% of units that are not `'good'` and never binned. (This is a deliberate trade — one bulk read beats thousands of small HDF5 slices — and the reference does the same.)

Everything actually written into the output dictionary is used by the decoder; no output field is computed and then thrown away.

ii.
```python
    if np.sum(keep_mask) >= 2:
        x[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], x[keep_mask])
        y[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], y[keep_mask])
    ...
    return y, diagnostics
```
```python
    preview_n = min(2000, len(tongue_timestamps))
    diagnostics = {
        "tracking_preview_time": tongue_timestamps[:preview_n] - go_times[0],
        "tracking_preview_raw": tongue_y[:preview_n],
        "tracking_preview_clean": cleaned_tongue_y[:preview_n],
        ...
    }
```
```python
        region_names=sorted(set(region_labels.tolist())),
```

iii. The notes do not discuss discarded computation. Step 6 frames the all-zero guard as deliberate defensive redundancy ("Added a residual all-zero-trial check after binning as a defensive guard against edge-case recording coverage mismatches"), which implies accepting the wasted binning for the trials it removes. The diagnostics were built unconditionally so that Step 7's processing plots and Step 9/10's per-session statistics could be produced from the same objects.
