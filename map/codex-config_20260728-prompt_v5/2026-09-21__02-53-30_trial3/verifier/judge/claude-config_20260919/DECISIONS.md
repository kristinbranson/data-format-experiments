# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI 000363 Mesoscale Activity Map dandiset: one NWB file per session under `/app/data/sub-<subject_id>/`. The AI discovers every session with a single sorted glob over that layout (hard-coded to `/app/data`, no `--datadir` option), and drops one session by an explicitly hard-coded filename constant (`sub-440958_ses-20190216T162508_...`) before any file is opened. Each remaining file is opened exactly once with **`h5py`** rather than `pynwb`, and every stream (`units/*`, `intervals/trials/*`, `acquisition/BehavioralEvents/*`, `acquisition/BehavioralTimeSeries/*`, `general/subject/subject_id`) is read by raw HDF5 path inside one `with h5py.File(...)` block. Conversion is a single serial pass over 173 files.

ii.
```python
EXCLUDED_SESSION = "sub-440958_ses-20190216T162508_behavior+ecephys+ogen"

def list_valid_session_paths() -> list[str]:
    all_paths = sorted(glob.glob("/app/data/sub-*/*.nwb"))
    valid_paths = []
    for path in all_paths:
        session_name = session_name_from_path(path)
        if session_name == EXCLUDED_SESSION:
            continue
        valid_paths.append(path)
    return valid_paths
```

```python
with h5py.File(path, "r") as h5file:
    subject_id = get_subject_id(h5file, path)
    classification = decode_string_array(h5file["units"]["classification"])
    ...
    spike_times_all = np.asarray(h5file["units"]["spike_times"][()], dtype=np.float64)
    spike_times_index = np.asarray(h5file["units"]["spike_times_index"][()], dtype=np.int64)
    trial_start_times = np.asarray(h5file["intervals"]["trials"]["start_time"][()], dtype=np.float64)
    go_times_abs = np.asarray(
        h5file["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][()],
        dtype=np.float64,
    )
```

iii. From CONVERSION_NOTES Step 2/Step 6: the AI first enumerated the directory and confirmed 174 NWB files / 28 subject folders against `dandiset.yaml` (`numberOfFiles: 174`, `numberOfSubjects: 28`), so the glob is the complete set of sessions. It chose `h5py` over `pynwb` explicitly as a speed-up ("Used `h5py` instead of higher-level NWB table conversion for the main conversion path"), which is listed in the Step 7 speed-up table. The one excluded file is justified in Step 4 and Key Decision 1: it has zero `classification == 'good'` units and all 1,852 units labelled `classification == nan`, and excluding it reproduces the papers' 173-session analysis cohort.

## 1-b. How are the data split into subjects?

i. Subject identity is read per file from the NWB `general/subject/subject_id` field (a numeric string such as `'440956'`), with a fallback that parses the id out of the containing `sub-<id>` directory name if that field cannot be read. At assembly, `subjects` is built in order of first appearance across the session list and `subject_idx` records each session's index into it. No further grouping is done.

ii.
```python
def get_subject_id(h5file: h5py.File, path: str) -> str:
    try:
        subject = h5file["general"]["subject"]["subject_id"][()]
        if isinstance(subject, bytes):
            return subject.decode("utf-8")
        return str(subject)
    except Exception:
        return os.path.basename(os.path.dirname(path)).replace("sub-", "")
```

```python
for session_idx, session in enumerate(processed_sessions):
    subject = session["subject_id"]
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(subjects)
        subjects.append(subject)
    subject_idx[session_idx] = subject_to_idx[subject]
```

iii. CONVERSION_NOTES Step 2 documents the per-subject session census read straight from the directory tree (28 subjects, 3–10 sessions each) and the Step 5 mapping table states "Collect unique subject IDs and map each kept session to subject index... Use all subjects represented among kept sessions." The verification log confirms 28 subjects with the expected per-subject session counts (the excluded session reduces `sub-440958` from 5 to 4).

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no splitting or grouping is inferred. Sessions are identified by the **filename stem** (e.g. `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`) rather than by the NWB `identifier` attribute, and the list is recorded in `metadata['session_names']`. Session order in the output follows the sorted file paths, which is chronological within subject because the filename embeds the acquisition timestamp.

ii.
```python
def session_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    for suffix in (".nwb",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base
```

```python
"session_names": session_names,
```

iii. Step 2 of CONVERSION_NOTES records the observed layout ("one subject folder per mouse and one NWB file per session"), so the file boundary is taken as the session boundary directly. 173 sessions reach the output; the single dropped file is the never-QC'd session described in 1-a.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial, and each row is paired positionally with one entry of `BehavioralEvents/go_start_times`. Rather than asserting that the two agree, the AI defensively truncates **every** per-trial array (including `go_times_abs`) to the minimum common length before anything else happens.

ii.
```python
ntrials = min(
    len(trial_start_times), len(trial_stop_times), len(auto_water), len(free_water),
    len(early_lick), len(outcome), len(trial_instruction),
    len(photostim_onset_trial_rel), len(photostim_duration), len(go_times_abs),
)
trial_start_times = trial_start_times[:ntrials]
...
go_times_abs = go_times_abs[:ntrials]
```

iii. Step 2 documents the trial-table columns and the 94,990 raw trials counted directly from the data. The `min(...)` truncation is the AI's edge-case guard (Step 10 "Check for edge cases") against a length mismatch between the trials table and the go-cue event stream. In practice the two are always equal (verified across all 174 files), so the guard never fires and the split is exactly the trials table.

## 1-e. How are trials filtered based on quality controls?

i. Six filters are applied, in two stages. Before binning: (1) `auto_water == 0`, (2) `free_water == 0`, (3) a tone (sample-start) event must exist inside the trial before its go cue, (4) **common neural coverage** — `units/is_good_trials` must be `True` for the trial across *every* retained good unit, (5) the `[go-2.5, go+1.5]` window must overlap the spike-time support span of the good units. After binning: (6) any trial whose binned matrix is entirely zero is dropped. A session with fewer than 2 surviving trials is dropped. Early-lick, `ignore`, and photostimulated trials are deliberately **kept**. Result: 88,943 of 94,990 trials (93.6%) across 173 sessions.

ii.
```python
def build_common_neural_trial_mask(is_good_trials, good_unit_indices, ntrials):
    """Keep only trials with valid neural coverage for every retained good unit."""
    coverage_mask = np.zeros(ntrials, dtype=bool)
    ...
    coverage_mask[:coverage_trials] = np.all(
        is_good_trials[good_unit_indices, :coverage_trials], axis=0)
    return coverage_mask
```

```python
spike_support_mask = (
    np.isfinite(spike_span_min) & np.isfinite(spike_span_max)
    & ((go_times_abs + WINDOW_END_S) > spike_span_min)
    & ((go_times_abs + WINDOW_START_S) < spike_span_max)
)
keep_trials = (
    (~auto_water) & (~free_water) & np.isfinite(tone_onsets_abs)
    & neural_coverage_mask & spike_support_mask
)
kept_trial_indices = np.flatnonzero(keep_trials)
if len(kept_trial_indices) < 2:
    return None
```

```python
nonzero_neural_mask = np.asarray([np.any(trial) for trial in neural_trials], dtype=bool)
n_zero_neural = int(np.count_nonzero(~nonzero_neural_mask))
if n_zero_neural:
    neural_trials = [t for t, k in zip(neural_trials, nonzero_neural_mask) if k]
    ...
```

iii. Key Decision 4 justifies the water exclusions: "These are not requested outputs, they are explicitly excluded in the reference analyses, and they can confound choice/outcome interpretation" — the reference code's `get_regular_trial_mask` excludes both `auto_water` and `free_water`. Key Decision 5 justifies retaining early-lick / ignore / photostim trials as a deliberate, task-required deviation from the reference "regular trial" subset, since these are the required decoder outputs/inputs. Filters (4)–(6) were added reactively: Step 7 found 321 all-zero-neural trials in one sample session and traced them to good units having real coverage for only the first 160 of 480 behavioural trials, fixed by intersecting with `is_good_trials`; Step 9/10 found a further 127 trials across 3 sessions whose decoder window lay entirely outside the good-unit spike support, fixed by the spike-span mask. After both fixes the all-zero safeguard removed 0 additional trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (the flat ragged buffer of session-absolute spike times) together with `units/spike_times_index` (the per-unit end offsets), restricted to units with `units/classification == 'good'`. The second input is `acquisition/BehavioralEvents/go_start_times/timestamps`, which places the bin grid.

ii.
```python
spike_times_all = np.asarray(h5file["units"]["spike_times"][()], dtype=np.float64)
spike_times_index = np.asarray(h5file["units"]["spike_times_index"][()], dtype=np.int64)
good_unit_indices = np.flatnonzero(classification == "good")
go_times_abs = np.asarray(
    h5file["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][()], dtype=np.float64)
```

iii. Step 5's mapping table: "NWB `units/spike_times` for units with `classification == 'good'` → `neural`". Step 2 established that spike times are the only neural representation in the file (no imaging, no dF/F), and Step 1 confirmed the reference pipeline likewise starts from spike times and calls `sliding_histogram`.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit the flat buffer is sliced with its index bounds, the absolute bin edges for all kept trials are formed as one `(n_trials, 81)` array, `np.searchsorted` gives the running spike count at every edge, and adjacent differences give the per-bin spike count. Counts are divided by the 0.05 s bin width to give Hz. No smoothing, normalisation, baseline subtraction, or low-firing-rate exclusion is applied. Rates are stored as **`float16`**.

ii.
```python
def bin_good_unit_spikes(spike_times_all, spike_times_index, good_unit_indices, go_times_abs):
    """Return a list of (n_neurons, n_timepoints) float16 firing-rate matrices."""
    neural_stack = np.empty((n_units, n_trials, N_BINS), dtype=np.float16)
    bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]
    for out_idx, unit_idx in enumerate(good_unit_indices):
        start = 0 if unit_idx == 0 else int(spike_times_index[unit_idx - 1])
        stop = int(spike_times_index[unit_idx])
        spikes = spike_times_all[start:stop]
        insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
        counts = np.diff(insertion_idx, axis=1)
        neural_stack[out_idx] = (counts.astype(np.float32) / BIN_SIZE_S).astype(np.float16)
    return [neural_stack[:, trial_idx, :].copy() for trial_idx in range(n_trials)]
```

iii. Step 5's mapping specifies "Bin absolute spike times into go-cue-aligned 50 ms bins over `[-2.5, 1.5)` s; divide counts by `0.05` to obtain firing rates", mirroring the reference `sliding_histogram(..., rate=True)`. Key Decision 3 explains why no firing-rate threshold is applied: "The method paper's `< 2 Hz` threshold was specific to video-to-neural prediction analyses, whereas the provided ephys decoding code does not impose one." Step 6 justifies `float16`: "to keep the full converted dataset tractable in memory and on disk; the downstream decoder converts to `float32` during training" (5.5 GB pickle).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are retained; no thresholds are placed on any individual quality metric (`isi_violation`, `presence_ratio`, `amplitude_cutoff`, …), and `unit_quality` ('good'/'multi') is deliberately not used. A session with zero such units is skipped. This retains 69,453 of 272,227 units (25.5%), mean 401.46 per session.

ii.
```python
classification = decode_string_array(h5file["units"]["classification"])
classification_counts = Counter(classification.tolist())
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    print(f"[session] {session_name}: skipping (0 classifier-good units)", flush=True)
    return None
```

iii. Key Decision 2: "Use `units/classification == 'good'` for neuron curation: This is the NWB analogue of the reference code's external `goodunits` classifier output; `unit_quality == good` is much looser and inconsistent with the papers." Step 4's discrepancy table records the quantitative argument: `classification == 'good'` gives 69,453 units against the papers' reported 69,943, whereas `unit_quality == 'good'` gives 154,948. The classifier itself is the region-specific logistic-regression QC described in `ChenLiuEtAl2023_SpikeSortingQC.pdf`, summarised in Step 3.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset, taken from `BehavioralEvents/go_start_times/timestamps`. Spike times and event timestamps share one session-absolute clock, so no resampling or offset correction is needed: the fixed relative edge grid is broadcast onto each trial's go-cue time to produce absolute bin edges, and spikes are binned against those directly.

ii.
```python
BIN_EDGES_REL = WINDOW_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
...
bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]
insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
```

iii. Key Decision 7: "Use go-cue alignment directly from NWB BehavioralEvents: The target dataset is explicitly go-cue aligned, matching both the user request and the reference preprocessing convention." Step 10's reference comparison confirms the reference code aligns spikes, licks and stimulation to `task_cue_time`/`gocue_time`, and verdicts the two "consistent". A Step 10 sanity check (`neural_vector_np_allclose`) recomputed a converted firing-rate vector directly from raw NWB spike times and passed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning −2.5 s to +1.5 s relative to the go cue, identical for every trial and every session. The grid is defined once at module level as 81 relative edges and reused. This **is** a deliberate rebinning relative to the reference pipeline, which uses 40 ms sliding windows at a 3.4 ms stride over [−3, 3] s. `metadata['time_bin_size'] = 50.0` (ms), `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
BIN_EDGES_REL = WINDOW_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
```

```python
"time_bin_size": 50.0,
"temporal_alignment_event": "Go cue onset",
"off_start": WINDOW_START_S,
"off_end": WINDOW_END_S,
```

iii. Step 1's "Relevant mismatch to plan for later" flags the discrepancy explicitly: "Reference preprocessing uses 40 ms sliding windows every 3.4 ms over a larger `[-3, 3]` interval. The requested decoder dataset instead needs go-cue-aligned extraction from `-2.5 s` to `+1.5 s` using 50 ms bins. This will require re-binning while preserving reference alignment logic." Step 10's binning comparison records the verdict: "intentional task-driven re-binning while preserving go-cue alignment and spike-count-to-rate conversion." The instructions mandate the 50 ms / [−2.5, 1.5] specification, so the deviation is required.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch/tone onsets), gated by the trials table's `start_time` and by the trial's go cue. For each trial the AI takes the **last** sample-start event that lies in `[trial_start, go_cue)`. If no such event exists the trial is dropped (see 1-e filter 3).

ii.
```python
def build_tone_onsets_abs(trial_start_times, go_times_abs, sample_start_times_abs):
    """Use the latest sample-start event within each trial before the go cue."""
    start_idx = np.searchsorted(sample_start_times_abs, trial_start_times, side="left")
    end_idx = np.searchsorted(sample_start_times_abs, go_times_abs, side="left")
    tone_onsets_abs = np.full(go_times_abs.shape, np.nan, dtype=np.float64)
    for trial_idx in range(len(go_times_abs)):
        if end_idx[trial_idx] > start_idx[trial_idx]:
            tone_onsets_abs[trial_idx] = sample_start_times_abs[end_idx[trial_idx] - 1]
    return tone_onsets_abs
```

iii. Step 4's discrepancy table: "sample/tone onset cannot be paired to trials by simple event index because replay adds extra sample events. Tone onset must be derived per trial from BehavioralEvents within the trial interval, with explicit handling of replayed early-lick trials." Step 5's mapping adds "This preserves replay-extended trials rather than forcing a fixed 1.85 s sample-to-go interval." Taking the *last* tone before the go cue is the one the animal actually acted on; restricting the search to within the trial prevents borrowing the previous trial's tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone time is converted to go-relative seconds (`tone_rel_s = tone_abs − go`), and each bin's value is its go-relative bin **center** minus that offset, i.e. seconds elapsed since tone onset at that bin center. Values are negative before the tone. The result is a continuous `(n_trials, 80)` float32 array. Observed full-dataset range: `[-1.5, 11.9]` s (long positive values come from replayed early-lick trials).

ii.
```python
tone_rel_s = tone_onsets_abs[kept_trial_indices] - go_keep
...
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]
inputs_2d = np.stack([time_from_tone, photostim_binary], axis=1).astype(np.float32)
```

iii. Key Decision 8: "Represent tone timing as a continuous go-relative trajectory: `time_from_tone_onset_s` will be a continuous per-bin input, with zero at tone onset and negative values before it, because the user explicitly asked for a continuous time-varying variable." The Decoder Task section specifies this input as "continuous, time-varying", which overrides the generic "represent times as a binary time series" guidance. A Step 10 sanity check (`input_time_from_tone_np_allclose`) reconstructed this from raw NWB and passed.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the bin grid used for the neural data: `BIN_CENTERS_REL` are the centers of the same 80 go-cue-relative edges that define the spike bins, and the tone offset is expressed in the same go-relative frame. Bin *k* of the input therefore covers the same interval as bin *k* of the firing rates by construction. No interpolation or independent resampling is done.

ii.
```python
BIN_EDGES_REL = WINDOW_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
...
bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]      # neural
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]     # input
```

iii. Step 10's temporal-alignment comparison: "Conversion aligns all decoder streams to `BehavioralEvents/go_start_times`. Verdict: consistent." The `--show-processing` plots (panel 7, "Aligned Inputs") were reviewed in Step 7 and reported as showing "tone and photostim alignments are go-relative and plausible".

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the trials table columns `photostim_onset` and `photostim_duration` (both stored as strings, with `'N/A'` on unstimulated trials), plus `start_time` and the trial's go cue to re-express them on the go-relative axis. The `BehavioralEvents/photostim_start_times` stream is not used.

ii.
```python
def decode_optional_float_array(dataset) -> np.ndarray:
    values = decode_string_array(dataset)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for idx, item in enumerate(values):
        if item != "N/A":
            out[idx] = float(item)
    return out
```

```python
photostim_onset_trial_rel = decode_optional_float_array(h5file["intervals"]["trials"]["photostim_onset"])
photostim_duration = decode_optional_float_array(h5file["intervals"]["trials"]["photostim_duration"])
```

iii. Step 4's discrepancy table records the empirical check: "NWB `photostim_onset` is stored relative to `trial start`, not to `go cue`. Matching per-trial values to event timestamps confirms the need to convert to go-relative time via `trial_start + photostim_onset - go_time`, mirroring the reference code's subtraction of go cue time." Step 2 counted 18,588 trials with non-`N/A` photostim fields, consistent with the papers' "~25% randomly interleaved trials" in optogenetic mice.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Onset and offset are converted to go-relative seconds, then rasterised into a binary `(n_trials, 80)` float32 array: a bin is `1` if the bin **interval overlaps** `[stim_on, stim_off)` (`bin_start < off and bin_end > on`), `0` otherwise. Unstimulated trials keep NaN bounds and are excluded by an explicit `valid` mask, so all their bins stay 0. Observed range across the dataset: `[0.0, 1.0]`.

ii.
```python
stim_on_rel_s = np.full(len(kept_trial_indices), np.nan, dtype=np.float64)
valid_stim = np.isfinite(photostim_onset_keep) & np.isfinite(photostim_duration_keep)
stim_on_rel_s[valid_stim] = (
    trial_start_keep[valid_stim] + photostim_onset_keep[valid_stim] - go_keep[valid_stim])
stim_off_rel_s = stim_on_rel_s + photostim_duration_keep
```

```python
def build_photostim_binary(stim_on_rel_s, stim_off_rel_s):
    bin_starts = BIN_EDGES_REL[:-1][None, :]
    bin_ends = BIN_EDGES_REL[1:][None, :]
    photostim = np.zeros((len(stim_on_rel_s), N_BINS), dtype=np.float32)
    valid = np.isfinite(stim_on_rel_s) & np.isfinite(stim_off_rel_s)
    if np.any(valid):
        on = stim_on_rel_s[valid][:, None]
        off = stim_off_rel_s[valid][:, None]
        photostim[valid] = ((bin_starts < off) & (bin_ends > on)).astype(np.float32)
    return photostim
```

iii. Key Decision 9: "Convert photostimulation to a binary overlap signal per 50 ms bin: This preserves stimulation timing while adapting the reference trial-level onset/duration representation to the target decoder format." The Decoder Task specifies "Whether photostimulation is on at every time point (discrete, time-varying)", so a per-bin binary series is mandated rather than a per-trial flag. Step 5 notes photostim trials must be retained even though the reference `get_regular_trial_mask` excludes them, because photostim is an explicit decoder input. A Step 10 sanity check (`input_photostim_np_allclose`) passed.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Onset and offset are shifted into the go-cue-relative frame (`trial_start + photostim_onset − go_cue`) before rasterisation, and the rasterisation uses `BIN_EDGES_REL` — the same 81 edges that define the neural bins. The photostim input therefore shares the neural bin grid exactly.

ii.
```python
stim_on_rel_s[valid_stim] = (
    trial_start_keep[valid_stim] + photostim_onset_keep[valid_stim] - go_keep[valid_stim])
```
```python
bin_starts = BIN_EDGES_REL[:-1][None, :]
bin_ends = BIN_EDGES_REL[1:][None, :]
```

iii. Same justification as 4-a/4-b: the per-trial-start reference frame of the raw column had to be converted to the go-cue frame that the whole dataset is aligned on, "mirroring the reference code's subtraction of go cue time" (Step 4). The `--show-processing` plot shades the photostim interval on the go-relative time axis (panel 5) so the alignment is visually checkable.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. **Not** from the trials table. Choice is reconstructed from the raw lick event streams `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times`, together with the go cue: the direction of the **first lick** in the 1.5 s response window `[go, go + 1.5)`. If neither port is licked in that window the trial is labelled `no lick`.

ii.
```python
def reconstruct_choice_labels(go_times_abs, left_lick_times_abs, right_lick_times_abs):
    """Use the first lick in the 1.5 s response window after go cue."""
    labels = np.empty(len(go_times_abs), dtype=object)
    for trial_idx, go_time in enumerate(go_times_abs):
        window_end = go_time + RESPONSE_WINDOW_S
        left_idx = np.searchsorted(left_lick_times_abs, go_time, side="left")
        right_idx = np.searchsorted(right_lick_times_abs, go_time, side="left")
        left_time = np.inf
        right_time = np.inf
        if left_idx < len(left_lick_times_abs) and left_lick_times_abs[left_idx] < window_end:
            left_time = left_lick_times_abs[left_idx]
        if right_idx < len(right_lick_times_abs) and right_lick_times_abs[right_idx] < window_end:
            right_time = right_lick_times_abs[right_idx]
        if left_time == np.inf and right_time == np.inf:
            labels[trial_idx] = "no lick"
        elif left_time <= right_time:
            labels[trial_idx] = "left"
        else:
            labels[trial_idx] = "right"
    return labels
```

iii. Key Decision 6: "Reconstruct choice from actual post-go licking rather than from instruction+outcome: The first lick in the 1.5 s response epoch gives the requested behavior directly, while remaining highly consistent with trial labels (`hit_match ≈ 99.61%`, `miss_opposite ≈ 99.78%`, `ignore_none ≈ 99.77%`)." The AI ran that agreement check in Step 5 before committing (trajectory step ~120). Step 3 records that the papers define the response epoch as 1.5 s, which fixes the window. Step 10 sanity checks verified choice against raw licks on hand-picked ignore / early / hit trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The string labels are mapped through a fixed dictionary to `0` left, `1` right, `2` no lick, and written into row 0 of the per-trial `(4, 80)` int8 output array, repeated across all 80 bins. `output_values[0] = ['left', 'right', 'no lick']`. Full-dataset distribution: `left 0.430, right 0.421, no lick 0.149`.

ii.
```python
CHOICE_TO_INT = {"left": 0, "right": 1, "no lick": 2}
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ...
]
```
```python
choice_labels = reconstruct_choice_labels(go_keep, left_lick_times_abs, right_lick_times_abs)
trial_choice_int = np.asarray([CHOICE_TO_INT[x] for x in choice_labels], dtype=np.int8)
...
outputs_2d = np.empty((len(kept_trial_indices), len(OUTPUT_NAMES), N_BINS), dtype=np.int8)
outputs_2d[:, 0, :] = trial_choice_int[:, None]
```

iii. The instructions list choice as "(left, right, no lick, per-trial)", giving both the three classes and their order. The target format requires a single `(n_output, n_timepoints)` array per trial, so the per-trial scalar is broadcast across the 80 bins; the format spec also says outputs should be time-varying "if at all possible", and broadcasting keeps all four outputs in one array. `int8` is used because all codes are small non-negative integers.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of `intervals/trials`, which already contains exactly the strings `'hit'`, `'miss'`, `'ignore'`.

ii.
```python
outcome = decode_string_array(h5file["intervals"]["trials"]["outcome"])
...
outcome_keep = outcome[kept_trial_indices]
```

iii. Step 2 verified the column's value set directly from the data ("`outcome` values are `hit`, `miss`, `ignore`"), which matches the three categories the instructions ask for, so no derivation is needed. Step 5's mapping notes "Keep ignore trials because they are a required decoder output", overriding the papers' practice of excluding no-response trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to `0` ignore, `1` miss, `2` hit via a fixed dictionary, written into row 1 of the output array and repeated across all 80 bins. Full-dataset distribution: `ignore 0.149, miss 0.166, hit 0.685`.

ii.
```python
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
```
```python
trial_outcome_int = np.asarray([OUTCOME_TO_INT[x] for x in outcome_keep], dtype=np.int8)
outputs_2d[:, 1, :] = trial_outcome_int[:, None]
```

iii. The code order follows the instructions' listing "Outcome (ignore, miss, hit, per-trial)". Step 9's consistency table discusses the resulting 68.5% hit rate against the papers' reported 84% correct rate and explains the gap: the paper statistic is on curated control trials only, whereas the converted set deliberately retains `ignore`, early-lick and photostimulated trials.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of `intervals/trials`, whose values are the strings `'early'` and `'no early'`.

ii.
```python
early_lick = decode_string_array(h5file["intervals"]["trials"]["early_lick"])
...
early_lick_keep = early_lick[kept_trial_indices]
```

iii. Step 2 confirmed the value set directly from the data ("`early_lick` values are `early`, `no early`") and counted 10,805 early trials of 94,990. Step 5's mapping cites `get_regular_trial_mask` as the reference analogue but notes "Keep early-lick trials because early lick is a required decoder output" — a deliberate deviation from the papers, which exclude them.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0` no / `1` yes via a fixed dictionary, written into row 2 of the output array and repeated across all 80 bins. Full-dataset distribution: `no 0.883, yes 0.117`.

ii.
```python
EARLY_TO_INT = {"no early": 0, "early": 1}
```
```python
trial_early_int = np.asarray([EARLY_TO_INT[x] for x in early_lick_keep], dtype=np.int8)
outputs_2d[:, 2, :] = trial_early_int[:, None]
```

iii. Follows the instructions' "(no, yes, per-trial)" ordering. As with the other per-trial outputs, the scalar is broadcast across bins so all four outputs share one `(4, 80)` array. The early lick itself happens during the sample or delay epoch, i.e. inside the −2.5 s pre-go window, so the event is within the extracted data even though the label is per-trial.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: its `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps`. Column 1 is the y value, column 2 is the DeepLabCut likelihood used to decide visibility. The jaw and nose streams are not used.

ii.
```python
tongue_data = np.asarray(
    h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["data"][()],
    dtype=np.float32)
tongue_timestamps_abs = np.asarray(
    h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["timestamps"][()],
    dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
tongue_likelihood_raw = tongue_data[:, 2]
```

iii. Step 2 records the channel layout read from the stream's own description attribute: "`Camera0_side_TongueTracking` is a `TimeSeries` with shape `(680500, 3)` and description `('tongue_x', 'tongue_y', 'tongue_likelihood')`", and confirms all 174 files contain it. Step 5's mapping justifies the side camera: "Use side-view only, consistent with the method paper", matching the reference `align_markers.py` marker keys (`tongue_x`, `tongue_y`, …).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) Per-session class edges: the 40th and 60th percentiles of **raw `tongue_y` over all visible frames in the session**, where visible means `likelihood >= 0.9` and the value is finite. (2) Per bin, a single representative frame is taken — the **last camera frame whose timestamp falls inside the bin** (`[bin_start, bin_end)`), matching the reference `align_markers_between_lims` convention. (3) That frame's y is digitised against the two edges; if the bin contains no frame, or the sampled frame's likelihood is below threshold, the bin gets class `3` (not visible). No averaging, smoothing, or outlier removal is applied. Full-dataset distribution: `lt_p40 0.062, p40_to_p60 0.032, gt_p60 0.066, not_visible 0.840` — i.e. among visible bins the split is ≈39/20/41%, as the percentile definition requires.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
visible_global = (
    np.isfinite(tongue_y_raw) & np.isfinite(tongue_likelihood_raw)
    & (tongue_likelihood_raw >= TONGUE_LIKELIHOOD_THRESHOLD))
if np.count_nonzero(visible_global) == 0:
    tongue_q40 = 0.0
    tongue_q60 = 0.0
else:
    tongue_q40, tongue_q60 = np.quantile(tongue_y_raw[visible_global], [0.4, 0.6])
```

```python
last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
valid = last_idx >= 0
...
    last_times = tongue_timestamps_abs[chosen]
    flat_starts = bin_starts_abs.ravel()[flat_valid]
    keep = last_times >= flat_starts          # frame must be inside the bin
    flat_positions = np.flatnonzero(flat_valid)[keep]
    sampled_y.ravel()[flat_positions] = tongue_y[chosen]
    sampled_likelihood.ravel()[flat_positions] = tongue_likelihood[chosen]
```

iii. Key Decision 10: "Use side-camera tongue tracking with likelihood-based visibility: The method paper uses side-view markers, and the raw likelihood channel is strongly bimodal near 0 and 1, making a visibility threshold of `0.9` defensible for the required `not visible` class" — the AI checked the bimodality empirically first (trajectory step 123: "most frames are near zero confidence, and the visible frames cluster near one"). Key Decision 11: "Discretize tongue y with session-wide percentiles over visible frames only: This matches the user specification and avoids contaminating percentiles with occluded/low-confidence frames." The last-frame-in-bin sampling is justified in Step 5 as "matching reference last-frame logic" — `align_markers.py:48` does `marker_vecs[i, trial, :] = _this_embed_array[:, np.where(mask)[0][-1]]`, i.e. the reference code likewise takes the last frame within each time bin. The papers' own handling (occluded tongue set to the mean position, Step 3) is overridden because the instructions require an explicit "not visible" class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, exactly as the instructions specify: `0` if the sampled y is below the session's 40th percentile, `1` if it lies in `[q40, q60]` inclusive, `2` if above the 60th percentile, `3` if the bin has no visible frame. The array is initialised to `3` and the three visible classes are written over it, so "not visible" is the default. `output_values[3] = ['lt_p40', 'p40_to_p60', 'gt_p60', 'not_visible']`.

ii.
```python
classes = np.full(bin_starts_abs.shape, 3, dtype=np.int8)
visible = (
    np.isfinite(sampled_y)
    & np.isfinite(sampled_likelihood)
    & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD))
classes[visible & (sampled_y < visible_q40)] = 0
classes[visible & (sampled_y >= visible_q40) & (sampled_y <= visible_q60)] = 1
classes[visible & (sampled_y > visible_q60)] = 2
return classes, sampled_y
```

iii. Directly implements the Decoder Task specification ("0: < 40th percentile of y-position over the session; 1: 40th to 60th percentile; 2: > 60th percentile; 3: not visible"), with percentiles taken per session. The Step 7 `--show-processing` plot (panel 4) overlays the two percentile lines on the visible-y histogram so the thresholds can be checked visually, and the Step 10 sanity check `tongue_class_np_allclose` re-derived one bin's class from raw NWB and passed.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as spikes and go cues, so alignment is done by constructing absolute bin boundaries from the *same* `BIN_EDGES_REL` grid used for the firing rates and assigning each frame to the bin containing its timestamp via `searchsorted`. Bin *k* of the tongue output therefore covers the same interval as bin *k* of the neural matrix. No interpolation or offset correction is applied.

ii.
```python
bin_starts_abs = go_times_abs[:, None] + BIN_EDGES_REL[:-1][None, :]
bin_ends_abs = go_times_abs[:, None] + BIN_EDGES_REL[1:][None, :]
last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
...
keep = last_times >= flat_starts
```

iii. Step 10's temporal-alignment check states that all decoder streams are aligned to `BehavioralEvents/go_start_times`, verdict "consistent". Step 7's plot review reports "tongue visibility/discretization matches the expected mostly-occluded side-view distribution" and that the raw tongue track plotted on the go-relative axis (panel 5) showed no misalignment. Because the side video is trial-gated rather than continuous, pre-go bins on short trials simply contain no frames and fall into class 3.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven distinct cases are handled:
- **Never-QC'd session** (`classification`/`anno_name` all NaN): `decode_string_array` turns non-bytes entries into `str(item)` so NaN becomes `'nan'`, which never equals `'good'`; the session is caught both by a hard-coded filename exclusion and by the `len(good_unit_indices) == 0` guard.
- **Length mismatch between per-trial arrays and the go-cue stream**: all arrays are truncated to the minimum common length (a defensive guard; it never fires on this dataset).
- **Trials with no spike coverage**: removed by the `is_good_trials` intersection, the spike-support-span mask, and a final all-zero-matrix safeguard.
- **Trials with no reconstructable tone onset**: `tone_onsets_abs` is NaN and the trial is dropped.
- **`'N/A'` photostim strings**: parsed to NaN and masked out, so those trials get an all-zero photostim input.
- **Sessions with no visible tongue frame at all**: percentiles default to 0.0 (harmless, since every bin is then class 3 anyway).
- **Blank/empty `anno_name`**: relabelled `"Unknown"` rather than creating an empty region name.
- **Unreadable `subject_id`**: falls back to parsing the `sub-<id>` directory name.
- Sessions left with fewer than 2 trials at either filtering stage are dropped entirely.

ii.
```python
def decode_string_array(dataset) -> np.ndarray:
    values = dataset[()]
    out = []
    for item in values:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return np.asarray(out, dtype=object)
```
```python
def decode_optional_float_array(dataset) -> np.ndarray:
    values = decode_string_array(dataset)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for idx, item in enumerate(values):
        if item != "N/A":
            out[idx] = float(item)
    return out
```
```python
region_labels = np.asarray(
    [label if str(label).strip() else "Unknown" for label in region_labels], dtype=object)
```
```python
if len(kept_trial_indices) < 2:
    print(f"[session] {session_name}: skipping (<2 kept trials after filtering)", flush=True)
    return None
```

iii. Step 10 "Issues Found and Resolved" documents the three substantive cases: behaviour trials without full common neural coverage (fixed by the `is_good_trials` intersection), decoder windows outside the actual spike support (127 trials across 3 sessions, fixed by the spike-span filter), and the one unresolved-QC session (excluded, "matching the paper's 173-session analyzed cohort"). The general principle applied is that where nothing was recorded the session or trial is excluded rather than emitted as zeros, while where a measurement legitimately has no value (retracted tongue) it becomes an explicit category. Step 10's edge-case scan confirmed the final dataset has no sessions with <2 trials, no all-zero-neural trials, and no NaNs in any array.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's documented answer (Step 6): "Per-session spike binning remains the dominant cost because every classifier-good unit must be aligned to every kept trial", plus the size of the full-dataset neural storage. Measured behaviour: the full conversion took **2.14 min** for 173 sessions, ~0.70 s/session mean (range ~0.28–1.16 s), scaling with unit and trial count; pickling the 5.5 GB result accounts for roughly the remaining ~7 s. Profiling one representative session confirms the AI's attribution: per-unit spike binning 0.215 s vs. all HDF5 reads combined ~0.17 s (spike buffer 0.109 s for 20.6 M doubles, tongue array 0.054 s, everything else <0.01 s). The AI did not instrument sub-steps within a session; only per-session totals are logged.

ii.
```python
for out_idx, unit_idx in enumerate(good_unit_indices):
    ...
    insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
    counts = np.diff(insertion_idx, axis=1)
    neural_stack[out_idx] = (counts.astype(np.float32) / BIN_SIZE_S).astype(np.float16)
```
```python
print(f"[progress] {session_idx}/{len(valid_paths)} scanned; kept={len(processed_sessions)}; "
      f"mean_per_session={mean_so_far:.2f}s; eta~{eta/60.0:.1f} min", flush=True)
```

iii. Step 7 estimated ~0.5–0.7 s/session → ~1.5–2.0 min for the full run, "well below the 15-minute optimization threshold, so no additional parallelization work was required". The actual 2.14 min matched, so no further optimisation was pursued. The AI attributes the low cost to two choices it made up front: `h5py` instead of `pynwb` table conversion, and one `searchsorted` over a flattened edge array per unit rather than a per-trial loop.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial dimension is vectorised throughout the neural, photostim and tongue paths, but five Python-level loops remain, and the AI does not document them:
- `reconstruct_choice_labels` — one Python iteration per kept trial (~89k total) doing two scalar `searchsorted` calls; fully vectorisable with array `searchsorted` plus `np.where`.
- `build_tone_onsets_abs` — one Python iteration per trial; the two `searchsorted` calls are already vectorised, so the loop body reduces to a single `np.where(end_idx > start_idx, sample[end_idx - 1], np.nan)`.
- `decode_string_array` / `decode_optional_float_array` — Python loops over every element of `classification` and `anno_name` (272,227 units in total across the dataset) and over the photostim string columns; `np.char.decode(arr, 'utf-8')` would do this in C.
- `get_good_unit_spike_span` — a Python loop building `starts`, replaceable with `np.where(idx == 0, 0, spike_times_index[idx - 1])`.
- `nonzero_neural_mask = [np.any(trial) for trial in neural_trials]` — a per-trial pass over the already-materialised list, computable in one reduction on the stacked array.
- The per-unit loop in `bin_good_unit_spikes` is inherent to the ragged spike storage (each unit has a different spike array), so it cannot be collapsed — this matches the reference.

ii.
```python
for trial_idx, go_time in enumerate(go_times_abs):
    window_end = go_time + RESPONSE_WINDOW_S
    left_idx = np.searchsorted(left_lick_times_abs, go_time, side="left")
    right_idx = np.searchsorted(right_lick_times_abs, go_time, side="left")
```
```python
for trial_idx in range(len(go_times_abs)):
    if end_idx[trial_idx] > start_idx[trial_idx]:
        tone_onsets_abs[trial_idx] = sample_start_times_abs[end_idx[trial_idx] - 1]
```
```python
for out_idx in range(1, len(good_unit_indices)):
    unit_idx = good_unit_indices[out_idx]
    starts[out_idx] = 0 if unit_idx == 0 else spike_times_index[unit_idx - 1]
```

iii. Step 6's "Code speedups added" lists only "Vectorized tone/stimulation/tongue-bin alignment across all kept trials within each session" and the `searchsorted`-on-array-shaped-queries pattern; the remaining loops are not identified as inefficiencies anywhere in CONVERSION_NOTES. The implicit justification is the Step 7 runtime estimate: at ~0.70 s/session the conversion is far inside the 15-minute budget, so the instructions' optimisation trigger ("If full conversion time estimate is longer than 15 minutes") never fired.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once and the bin grid is built once at module level, but there is real duplicated work inside a session, none of it documented:
- **Per-unit spike index bounds are computed twice** — once in `get_good_unit_spike_span` and again in `bin_good_unit_spikes`.
- **Three overlapping checks for "does this trial have neural data"**: the `is_good_trials` intersection, the `spike_support_mask`, and the post-binning all-zero safeguard. Each was added in a separate debugging round (Steps 7, 9) rather than replacing the previous one; by the AI's own Step 9 accounting the third removed 0 trials once the second was in place.
- **Trial masking is applied twice**: once via `kept_trial_indices` and then again via `nonzero_neural_mask`, which re-slices twelve separate arrays.
- **String columns are decoded in full** (`classification`, `anno_name`) even though only the `good` subset of `anno_name` is used.

ii.
```python
spike_span_min, spike_span_max = get_good_unit_spike_span(...)   # computes starts/stops
...
neural_trials = bin_good_unit_spikes(...)                        # recomputes start/stop per unit
```
```python
nonzero_neural_mask = np.asarray([np.any(trial) for trial in neural_trials], dtype=bool)
if n_zero_neural:
    neural_trials = [t for t, k in zip(neural_trials, nonzero_neural_mask) if k]
    inputs_2d = inputs_2d[nonzero_neural_mask]
    outputs_2d = outputs_2d[nonzero_neural_mask]
    go_keep = go_keep[nonzero_neural_mask]
    ...                                                          # 12 arrays re-sliced
```
```python
anno_name = decode_string_array(h5file["units"]["anno_name"])
region_labels = anno_name[good_unit_indices].astype(object)
```

iii. CONVERSION_NOTES does not claim these are redundant; the layered trial filters are presented in Step 10 as successive bug fixes ("Issue 1", "Issue 2"), each retained as a belt-and-braces safeguard rather than consolidated. The AI's implicit justification is defensive: keeping the all-zero check guarantees the validator warning about all-zero trials can never reappear even if the upstream metadata filters miss a case. The duplicated index arithmetic and full string decoding are simply not addressed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is a substantial amount, concentrated in what `process_session` returns. The returned per-session dict carries the **full raw tongue arrays** (`tongue_timestamps_abs` float64, `tongue_y_raw` float32, `tongue_likelihood_raw` float32, plus `tongue_y_visible_raw`) for **every** session, and all of these dicts are held in `processed_sessions` for the whole run — but they are only ever read by `plot_processing_steps`, which runs on at most 2 sessions and only under `--show-processing`. Measured on one session this is ~19.6 MB of raw video arrays; across 173 sessions roughly 3 GB of RAM retained for no purpose in the default `--full` path. Additionally:
- `align_tongue_to_bins` builds and returns a full `(n_trials, 80)` `sampled_y` array that the caller discards with `_`.
- It also materialises a parallel `sampled_likelihood` array purely to re-test a threshold already applied when computing `visible_global`.
- `trial_stop_times` is read, truncated, and never used.
- `classification_counts` (a `Counter` over all units) and the seven `n_trials_excluded_*` diagnostics are computed and stored per session but only consumed by the plotting path.
- `float16` firing rates are cast back to `float32` by the decoder at training time.

ii.
```python
return {
    ...
    "tongue_timestamps_abs": tongue_timestamps_abs,
    "tongue_y_raw": tongue_y_raw,
    "tongue_likelihood_raw": tongue_likelihood_raw,
    "tongue_y_visible_raw": tongue_y_raw[visible_global],
    "tongue_classes": tongue_classes,
    "classification_counts": classification_counts,
    "n_trials_excluded_auto_free": int(np.count_nonzero(auto_water | free_water)),
    ...
}
```
```python
tongue_classes, _ = align_tongue_to_bins(...)     # sampled_y discarded
```
```python
trial_stop_times = np.asarray(h5file["intervals"]["trials"]["stop_time"][()], dtype=np.float64)
...
trial_stop_times = trial_stop_times[:ntrials]     # never used again
```

iii. CONVERSION_NOTES identifies no unnecessary processing; the only memory concern it raises is "Full-dataset neural storage is large even after classifier-based unit filtering", which it addresses with `float16` ("Stored neural arrays as `float16` to reduce memory traffic and pickle size", Step 6). The retained raw-tracking arrays appear to be an unintended consequence of using one dict both as the conversion result and as the input to the diagnostic plotting function, rather than a documented decision. In practice the run completed in 2.14 min without an out-of-memory failure, so the waste was never surfaced.
