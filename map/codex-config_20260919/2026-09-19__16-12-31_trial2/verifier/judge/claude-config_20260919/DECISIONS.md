# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI 000363 NWB release, distributed as one HDF5/NWB file per session under `/app/data/sub-<subject_id>/`. The AI discovers every session with a single sorted `glob` over that layout and then reads each file **twice**:

1. A first "inventory" pass (`discover_inventory`) opens all 174 files with `h5py` and collects the global sorted vocabulary of subject IDs (`general/subject/subject_id`) and fine Allen CCF region names (`units/anno_name` for classifier-good units). It also asserts that the NWB `subject_id` equals the containing folder name, and that no curated unit lacks an anatomical annotation.
2. A second conversion pass (`convert_session`) re-opens each file and reads the arrays it needs: `units/classification`, `units/anno_name`, `units/is_good_trials`, `units/obs_intervals_index`, `units/spike_times(_index)`, `intervals/trials/*`, `acquisition/BehavioralEvents/{go_start_times,sample_start_times}/timestamps`, and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`.

Notably the AI reads the NWB files with raw `h5py` rather than `pynwb`, addressing HDF5 paths directly.

ii.
```python
DATA_ROOT = Path("/app/data")
...
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
if not paths:
    raise FileNotFoundError(f"No NWB files under {DATA_ROOT}")
print(f"Discovered {len(paths)} NWB files", flush=True)

inventory_start = time.perf_counter()
subjects, brain_regions = discover_inventory(paths)
subject_lookup = {name: i for i, name in enumerate(subjects)}
region_lookup = {name: i for i, name in enumerate(brain_regions)}
```

```python
def discover_inventory(paths: list[str]) -> tuple[list[str], list[str]]:
    """Return globally sorted subject IDs and fine CCF region annotations."""
    subjects: set[str] = set()
    regions: set[str] = set()
    for path in paths:
        with h5py.File(path, "r") as nwb:
            subject = decode_scalar(nwb["general/subject/subject_id"][()])
            folder_subject = Path(path).parent.name.removeprefix("sub-")
            if subject != folder_subject:
                raise ValueError(f"Subject mismatch in {path}: {subject} vs {folder_subject}")
            subjects.add(subject)
            classification = decode_array(nwb["units/classification"][:])
            good = classification == "good"
            if not np.any(good):
                continue
            annotations = decode_array(nwb["units/anno_name"][:])[good]
            ...
            regions.update(annotations.tolist())
    return sorted(subjects), sorted(regions)
```

```python
with h5py.File(path, "r") as nwb:
    units = nwb["units"]
    classification = decode_array(units["classification"][:])
    ...
    trial_table = nwb["intervals/trials"]
```

iii. From CONVERSION_NOTES Step 2: "174 NWB 2.x HDF5 files are grouped in 28 `sub-<id>/` directories; one file is one recording session", matching `dandiset.yaml` (DANDI 000363 v0.230822.0128, 53.6 GB, 28 mice). Because there is exactly one file per session, the directory listing is the complete inventory and a sorted glob makes session order deterministic (Step 5 Key Decision 7: "lexicographic source path order"). The separate inventory pass is justified as producing a stable, globally sorted subject/region vocabulary independent of which sessions are processed (so `--sample` and `--full` share the same index space). The AI reports the inventory pass costs 4.65 s of a 188.8 s total run.

## 1-b. How are the data split into subjects?

i. Each NWB file names its animal in `general/subject/subject_id` (a numeric string such as `'440956'`). The AI reads this for each file, cross-checks it against the `sub-<id>` folder name and raises if they disagree, builds `subjects` as the globally sorted set of unique IDs during the inventory pass, and stores each session's index into that list as `subject_idx`. Result: 28 subjects with 3–10 sessions each.

ii.
```python
subject = decode_scalar(nwb["general/subject/subject_id"][()])
folder_subject = Path(path).parent.name.removeprefix("sub-")
if subject != folder_subject:
    raise ValueError(f"Subject mismatch in {path}: {subject} vs {folder_subject}")
subjects.add(subject)
```

```python
subject = decode_scalar(nwb["general/subject/subject_id"][()])
...
"subject_idx": subject_lookup[subject],
```

```python
"subjects": subjects,
"subject_idx": np.asarray([x["subject_idx"] for x in converted], dtype=np.int32),
```

iii. Step 5 variable mapping: "`/general/subject/subject_id` / folder ID → `subjects`, `subject_idx`; Stable sorted subject identifiers and integer indices; Folder and NWB IDs will be asserted consistent." The subject ID is the canonical animal identifier in the file, and the DANDI folder layout is derived from it, so the two are redundant and the equality assertion is a free consistency check. The vocabulary is built globally (over all 174 files, including the one later dropped) so that indices are stable regardless of run mode.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session, so no splitting is required. The session identifier is derived from the filename by stripping the modality suffix (`sub-440956_ses-20190207T120657_behavior+ecephys+ogen.nwb` → `sub-440956_ses-20190207T120657`). Session order is the lexicographic path order, which puts sessions in chronological order within each subject because the filename embeds the acquisition timestamp. Per-session bookkeeping (trial counts at each filtering stage, unit counts, tongue percentiles, source file) is written into `metadata['session_info']`. 173 of 174 files reach the output; one is skipped for having no classifier-curated units.

ii.
```python
def session_id_from_path(path: str) -> str:
    name = Path(path).name
    return name.split("_behavior")[0]
```

```python
for path in paths:
    result = convert_session(path, subject_lookup, region_lookup,
                             show_processing=args.show_processing and len(converted) < 2)
    if result is None:
        continue
    converted.append(result)
    if args.sample and len(converted) == 2:
        break
```

```python
session_stats = {
    "session_id": sid,
    "source_file": str(Path(path).relative_to(DATA_ROOT)),
    "subject": subject,
    "n_trials_table": int(n_trials_table),
    "n_trials_recorded": int(n_trials_recorded),
    "n_trials_stable": int(np.sum(stable_mask)),
    ...
}
```

iii. Step 2: "one file is one recording session". Step 5 Key Decision 7: "lexicographic source path order; `--sample` takes the first two usable sessions deterministically." Step 4 resolves the session count discrepancy: "One of 174 files has all-NaN classification/anatomy … Exclude that file; exact session count then matches" the papers' 173 analysed sessions.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial. The AI pairs each trial row **positionally** with the corresponding event in `go_start_times`, after truncating both to the first `n_trials_recorded` rows, where `n_trials_recorded = units/is_good_trials.shape[1]` (the number of trials the ephys recording covers). Every per-trial column (`start_time`, `outcome`, `trial_instruction`, `early_lick`, `photostim_onset`, `photostim_duration`, `auto_water`, `free_water`) is sliced with the same `[:n_trials_recorded]`. The positional pairing is validated indirectly inside `first_tone_onsets`, which requires each trial's tone event to fall inside `[trial_start, go]` — an assertion that would fail loudly if go cues and trial rows were mismatched.

ii.
```python
trial_table = nwb["intervals/trials"]
n_trials_table = len(trial_table["id"])
stability_all = units["is_good_trials"][:]
stability = stability_all[good_indices]
n_trials_recorded = stability.shape[1]
if n_trials_recorded > n_trials_table:
    raise ValueError(f"Neural trials exceed trial table in {path}")
```

```python
go_all = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:n_trials_recorded]
trial_starts = trial_table["start_time"][:n_trials_recorded]
sample_starts = nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:]
tone_all = first_tone_onsets(trial_starts, go_all, sample_starts)
```

```python
def first_tone_onsets(trial_starts, go_times, sample_starts):
    indices = np.searchsorted(sample_starts, trial_starts, side="left")
    ...
    tone = sample_starts[indices]
    valid = (tone >= trial_starts - 1e-9) & (tone <= go_times + 1e-9)
    if not np.all(valid):
        bad = np.flatnonzero(~valid)[:5]
        raise ValueError(f"Tone event outside trial-to-go interval at trials {bad.tolist()}")
    return tone
```

iii. Step 2: "`/intervals/trials` has 264–800 rows/session"; "Go events have one row per NWB trial. Extra sample/delay events can occur, so trial association must be interval-based rather than positional for those streams." The AI therefore keeps positional pairing only for the go cue (where it is one-to-one) and uses interval containment for the sample/tone stream. Step 4 records that the ephys trial extent is defined by `is_good_trials.shape[1]` and the ragged `obs_intervals`, both of which agree: "Eight files contain 1,060 trailing behavior-table rows after every unit's observation intervals; `is_good_trials.shape[1]` and ragged `obs_intervals` both identify the ephys trial count."

## 1-e. How are trials filtered based on quality controls?

i. Three successive filters, all motivated by presence/validity of neural data rather than by behaviour:

1. **Ephys extent (positional)**: only the first `n_trials_recorded` trial rows are considered, where `n_trials_recorded` is the width of `units/is_good_trials`. The AI cross-checks this against the per-unit length of `obs_intervals_index` and raises if the two counts disagree.
2. **Per-trial unit stability**: a trial is kept only if `is_good_trials` is `True` for **every** classifier-curated unit (`np.all(stability, axis=0)`). This removes 509 trials in 4 sessions and guarantees a rectangular, artefact-free session tensor.
3. **All-zero population windows**: after binning, any trial whose entire (n_neurons × 80) firing-rate matrix is exactly zero is dropped as lying beyond real spike coverage. This removes 2,548 trials.

A "coverage" mask on finite tone onset is also applied but is vacuously all-`True`. Deliberately **not** applied are the reference papers' `regular` trial exclusions (photostimulation, early lick, auto/free water, ignore/no-response), because those categories are required decoder inputs/outputs. Notably, `free_water` is **not** filtered explicitly; the AI relies on the all-zero heuristic to remove those trials empirically. Sessions falling below 2 trials raise an error (never triggered). Final count: 94,990 table rows → 94,370 (excluding the dropped session) → 93,310 (ephys extent) → 92,801 (stability) → **90,253** converted trials.

ii.
```python
# The ragged observation-interval count independently verifies how many
# leading trials contain ephys for each curated unit.
obs_ends = units["obs_intervals_index"][:]
obs_lengths = np.diff(np.r_[0, obs_ends])
if not np.all(obs_lengths[good_indices] == n_trials_recorded):
    raise ValueError(f"Observation intervals disagree with stability table in {path}")

stable_mask = np.all(stability, axis=0)
```

```python
# An entirely missing video window is still a valid observation for this
# task: the requested tongue class 3 explicitly means "not visible".
coverage_mask = np.isfinite(tone_all)
keep = stable_mask & coverage_mask
keep_idx = np.flatnonzero(keep)
if len(keep_idx) < 2:
    raise ValueError(f"Fewer than two valid trials in {path}")
```

```python
# A small number of source files have an off-by-one terminal observation
# interval after every unit's spike train has ended. With hundreds of units,
# a completely silent four-second window is a reliable missing-ephys marker.
neural_present = np.any(rates != 0, axis=(1, 2))
if not np.all(neural_present):
    print(f"  {sid}: excluding {np.sum(~neural_present)} all-zero "
          "population trial(s) beyond spike coverage", flush=True)
    rates = rates[neural_present]
    keep_idx = keep_idx[neural_present]
    ...
```

iii. Step 4 discrepancy table: "Video paper `regular` mask removes stimulation, early lick, water, and ignore trials … Do not apply those exclusions because they would delete required decoder classes/inputs. Exclude only neural/alignment invalidity." Step 3 curation rules: "Trials will instead be retained unless they lack the go-aligned window, required behavior stream, or valid neural recording." The stability filter is justified from the QC white paper: "QC paper rejects unstable recordings/periods … Require a trial to be valid for every retained curated neuron, yielding a rectangular, artifact-free session tensor while retaining the published unit population." The all-zero filter was discovered empirically in Step 7: "Direct raw inspection found every curated spike train ended at 1107.44 s while that trial began at 1109.39 s, despite the NWB observation table including it." Step 10 Iteration 2 records that an earlier video-presence filter was removed because it contradicted the "not visible" class policy, recovering 754 neurally valid trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` together with its ragged index `units/spike_times_index`, restricted to units with `units/classification == 'good'`. The go-cue times from `acquisition/BehavioralEvents/go_start_times/timestamps` supply the alignment anchor that places the bin edges. `units/anno_name` supplies the per-neuron brain region, and `units/is_good_trials` / `units/obs_intervals_index` gate which trials are valid.

ii.
```python
spike_ends = units["spike_times_index"][:]
spike_values = units["spike_times"]
rates = np.empty((n_trials, n_units_good, N_TIME), dtype=np.float32)
for out_unit, source_unit in enumerate(good_indices):
    start = 0 if source_unit == 0 else int(spike_ends[source_unit - 1])
    stop = int(spike_ends[source_unit])
    spikes = spike_values[start:stop]
```

```python
classification = decode_array(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
annotations = decode_array(units["anno_name"][:])[good_indices]
```

iii. Step 2: "`/units` contains ragged absolute spike timestamps, unit/probe metadata, scalar QC metrics, `classification` (`good`/`unlabelled`), fine Allen CCF `anno_name`, and a neuron-by-trial `is_good_trials` stability mask." Spike times are the only neural representation in the file. Step 1 notes this is electrophysiology, so "delta-F/F is inapplicable."

## 2-b. How is the `neural` data processed?

i. Spike times are converted into per-bin firing rates in Hz. For each good unit, the absolute bin edges for all trials are flattened into one array, `np.searchsorted(..., side="left")` gives the running spike count at every edge, `np.diff` over the reshaped `(n_trials, 81)` array gives the spike count per bin, and the counts are divided by the 0.05 s bin width. No smoothing, no overlap, no normalisation, no baseline subtraction. Output dtype is `float32`, shaped `(n_trials, n_neurons, 80)` so that each per-trial slice is already C-contiguous `(n_neurons, 80)`. The result is asserted to be non-negative and an exact multiple of 20 Hz (the quantum of one spike in a 50 ms bin).

ii.
```python
# Binning is vectorized over all trials for each unit. searchsorted at all
# edges avoids a Python trial loop and exactly implements [left,right).
absolute_edges = go[:, None] + BIN_EDGES[None, :]
spike_ends = units["spike_times_index"][:]
spike_values = units["spike_times"]
rates = np.empty((n_trials, n_units_good, N_TIME), dtype=np.float32)
for out_unit, source_unit in enumerate(good_indices):
    start = 0 if source_unit == 0 else int(spike_ends[source_unit - 1])
    stop = int(spike_ends[source_unit])
    spikes = spike_values[start:stop]
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
    counts = np.diff(edge_positions.reshape(n_trials, N_TIME + 1), axis=1)
    rates[:, out_unit, :] = counts / BIN_WIDTH_S
```

```python
if np.any(rates < 0) or not np.allclose(rates / 20.0, np.round(rates / 20.0)):
    raise AssertionError(f"Firing rates are not valid 50-ms counts in {path}")
```

iii. Step 1 identified `sliding_histogram` in the reference code, which "Counts spikes in half-open bins centered on requested times and divides by bin width to produce Hz"; Step 10's comparison table records "Same alignment … Vectorized half-open nonoverlapping 50-ms bins" with "Width/stride intentionally follow decoder specification". Step 4: "Use 80 non-overlapping half-open bins, divide counts by 0.05 s." Step 5: "No smoothing/overlap because decoder task specifies 50-ms width." The 2-Hz minimum firing-rate cut used in the method paper was deliberately not applied: "the method paper additionally excludes <2-Hz neurons only for video-prediction analyses; this task decodes behavior from neural activity … so no extra firing-rate cutoff is planned."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are retained — the verdict of the region-specific spike-sorting QC classifier. No individual QC metric is thresholded and no minimum-firing-rate cut is added. A session with zero such units is skipped entirely (returns `None`). Retained: 69,453 of 272,227 units (25.5%) across 173 sessions, mean 401.46 / session, range 90–923. The single dropped file (`sub-440958_ses-20190216T162508`) has `classification` and `anno_name` NaN for all units. Trial-level neural QC (per-unit `is_good_trials` stability and all-zero-window rejection) is described in 1-e.

ii.
```python
classification = decode_array(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
if len(good_indices) == 0:
    print(f"SKIP {sid}: no classifier-curated units", flush=True)
    return None
```

```python
annotations = decode_array(units["anno_name"][:])[good_indices]
if np.any((annotations == "") | (annotations == "nan")):
    raise ValueError(f"Curated unit without anatomy in {path}")
```

```python
"unit_filter": "NWB units.classification == 'good' (published regional classifier QC)",
```

iii. Step 3 neuron curation rules: "Kilosort2 clusters were classified using five region-specific logistic-regression classifiers trained on 15 QC metrics and blinded manual labels. Individual-metric thresholding is specifically discouraged because good and unlabelled distributions overlap. The NWB `classification == 'good'` flag is therefore the authoritative inclusion rule." Step 4 reconciles the count against the papers: "69,943 units, 173 sessions, 25.9% of Kilosort clusters … The 490-unit paper/release discrepancy cannot be corrected from available fields and is attributed to the published snapshot/export; never synthesize or reclassify units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**. All streams in the NWB file (spike times, behavioural events, camera timestamps, trial intervals) share one absolute session clock, so no resampling or per-stream offset correction is needed. The fixed relative edge grid `BIN_EDGES = [-2.5, -2.45, …, +1.5]` is added to each trial's go-cue timestamp to produce the absolute edge times for that trial, and the spikes are binned against those edges directly.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_WIDTH_S = 0.050
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)
```

```python
go_all = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:n_trials_recorded]
...
go = go_all[keep]
...
absolute_edges = go[:, None] + BIN_EDGES[None, :]
```

```python
"temporal_alignment_event": "go cue onset",
"off_start": OFF_START,
"off_end": OFF_END,
```

iii. The decoder task specification requires go-cue alignment over [-2.5, +1.5] s. Step 2: "Spikes, behavioral events, video timestamps, and trial intervals share the same absolute session clock." Step 4: "all clocks in NWB are absolute; go alignment is done by subtracting each trial's go timestamp." Step 3 adds the sanity landmark that "Canonical time landmarks are approximately sample `[-1.85,-1.2]` s, delay `[-1.2,0]` s, and response `[0,1.5]` s", and the `--show-processing` plot overlays the tone marker on the population mean rate to check this visually (Step 7: "tone markers occur at the expected -1.85 s on example trials").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms per bin, 80 non-overlapping half-open `[left, right)` bins spanning -2.5 s to +1.5 s relative to the go cue, identical for every trial and session. `metadata['time_bin_size'] = 50.0` (ms), and `metadata['n_timepoints']` and `metadata['time_bin_centers_seconds']` are also recorded. There is no rebinning from a coarser native resolution: the source neural data are raw spike timestamps, so the 50 ms grid is applied once at binning time. The behavioural video (~294 Hz) is *down*-sampled onto the same grid by point-sampling the nearest preceding frame at each bin centre (see 8-b/8-d), not by averaging. The reference papers' 40 ms / 3.4 ms sliding window is deliberately not reproduced.

ii.
```python
BIN_WIDTH_S = 0.050
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)
```

```python
"time_bin_size": 50.0,
"n_timepoints": N_TIME,
"time_bin_centers_seconds": BIN_CENTERS.astype(float).tolist(),
"neural_measurement": "firing rate (Hz) from non-overlapping 50-ms spike-count bins",
"spike_bin_interval_convention": "half-open [left, right)",
```

iii. Step 4: "Binning — 40-ms sliding window at 3.4-ms stride in method paper code … User requires 50-ms width from -2.5 to +1.5 s → Use 80 non-overlapping half-open bins, divide counts by 0.05 s." Step 1: "This conversion instead must use non-overlapping 50-ms bins over exactly -2.5 to +1.5 s because the decoder specification overrides that binning." Step 5 Key Decision 5: "metadata offsets are bin edges (-2.5,+1.5); sample times are centers (-2.475,...,+1.475); all state sampling and plotted alignment uses centers." The validator independently confirmed 80 bins for every trial.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the auditory sample-epoch/tone onsets), together with `intervals/trials/start_time` and the go-cue times. For each trial the AI takes the **first** sample event at or after that trial's start time, and asserts it falls at or before the go cue. When an early lick causes the sample epoch to be replayed there can be up to 14 sample events in a trial; the AI deliberately selects the earliest (the original instruction onset) rather than the last one before the go cue.

ii.
```python
def first_tone_onsets(
    trial_starts: np.ndarray, go_times: np.ndarray, sample_starts: np.ndarray
) -> np.ndarray:
    """Find the first auditory sample event in each trial without positional pairing."""
    indices = np.searchsorted(sample_starts, trial_starts, side="left")
    if np.any(indices >= len(sample_starts)):
        raise ValueError("A recorded trial has no sample/tone event")
    tone = sample_starts[indices]
    valid = (tone >= trial_starts - 1e-9) & (tone <= go_times + 1e-9)
    if not np.all(valid):
        bad = np.flatnonzero(~valid)[:5]
        raise ValueError(f"Tone event outside trial-to-go interval at trials {bad.tolist()}")
    return tone
```

```python
"tone_onset_definition": "first sample_start_times event between trial start and go cue",
```

iii. Step 4 discrepancy table: "`sample_start_times` can contain 1–14 epochs per trial; extra entries are concentrated in early-lick trials … Early licking replays the sample/delay epoch → Use the first `sample_start_times` event between trial start and go: 'tone onset' denotes the trial's initial instruction onset. Interval matching avoids positional mismatch while preserving the genuinely longer elapsed time on replay trials." Step 10 Check 4 records that the AI inspected the resulting extrema and accepted them: "The extrema (-1.525 to 11.8943 s before validator rounding) were traced to legitimate raw trials: one has tone-to-go -0.95 s and another unusually long -10.4193 s; neither is a mapping error."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A single subtraction, evaluated at every bin. The absolute bin centres for a trial (`go + BIN_CENTERS`) minus that trial's tone onset gives seconds elapsed since the tone at each of the 80 timepoints, stored as `float32` in row 0 of the `(2, 80)` input array. Values are negative before the tone and increase by exactly 0.05 s per bin. Nothing is clipped, normalised, or binarised.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
...
centers = centers_all[keep]
...
inputs = np.empty((n_trials, 2, N_TIME), dtype=np.float32)
inputs[:, 0, :] = centers - tone[:, None]
```

```python
| First `/acquisition/BehavioralEvents/sample_start_times` within trial start-to-go | `input[0]` |
  At every bin center, `absolute_bin_center - tone_onset`; float32 seconds | ... | Continuous, time-varying as explicitly requested. |
```

iii. Step 5 variable mapping specifies "Continuous, time-varying as explicitly requested." Step 10 Check 4 verifies the arithmetic: "elapsed-time traces advance by 0.05 s/bin. Raw input comparisons passed." The AI's independent `cache/raw_sanity_checks.py` reconstructs tone elapsed time directly from the raw events and compares with `np.allclose`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same grid used for the neural bins. `centers_all = go_all[:, None] + BIN_CENTERS[None, :]` uses the same go-cue anchors and the same `BIN_CENTERS` derived from the same `BIN_EDGES` that produce the firing rates, and the same `keep`/`neural_present` masks are applied to `centers` as to `rates`. Bin *k* of the input therefore corresponds to the interval of bin *k* of the neural array (the input is evaluated at the bin centre, the neural rate over the whole bin).

ii.
```python
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
...
absolute_edges = go[:, None] + BIN_EDGES[None, :]
```

```python
rates = rates[neural_present]
keep_idx = keep_idx[neural_present]
go = go[neural_present]
tone = tone[neural_present]
centers = centers[neural_present]
```

iii. Step 5 Key Decision 5: "Window convention: metadata offsets are bin edges (-2.5,+1.5); sample times are centers (-2.475,...,+1.475); all state sampling and plotted alignment uses centers." Step 7's processing plot overlays the input traces on the same time axis as the firing-rate heatmap with the go cue marked, and Step 7 reports "No temporal shift or percentile inversion was observed."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials-table columns `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`, together with `intervals/trials/start_time` to convert them onto the session-absolute clock. Both photostim columns are stored as strings, with `'N/A'` on non-stimulated trials, so they are parsed through a tolerant converter that yields `NaN` where parsing fails. The separate `BehavioralEvents` photostimulation start/stop event streams are not used; the trial columns are preferred.

ii.
```python
def numeric_or_nan(values) -> np.ndarray:
    """Convert NWB string-valued numeric columns (`N/A` included) to float."""
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        try:
            out[i] = float(decode_scalar(value))
        except (TypeError, ValueError):
            pass
    return out
```

```python
onset = numeric_or_nan(trial_table["photostim_onset"][:n_trials_recorded])[keep_idx]
duration = numeric_or_nan(trial_table["photostim_duration"][:n_trials_recorded])[keep_idx]
kept_trial_starts = trial_starts[keep_idx]
stim_start = kept_trial_starts + onset
stim_stop = stim_start + duration
```

iii. Step 2: `/intervals/trials` columns include "photostimulation onset/power/duration". Step 5 variable mapping: "Trial `photostim_onset` + start time and `photostim_duration` → `input[1]` … Uses actual interval, not nominal task timing." Step 3: "Photoinhibition is typically during the last 0.5 s of delay and ends before go cue; representing the actual event intervals preserves exceptional timing rather than hard-coding this nominal window."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series rather than a per-trial flag. `stim_start = trial_start + photostim_onset` and `stim_stop = stim_start + photostim_duration` (both absolute). A bin is 1 when its centre falls in the half-open interval `[stim_start, stim_stop)` and 0 otherwise. Non-stimulated trials carry `NaN` bounds; explicit `np.isfinite` guards force those trials to all-zero. Stored as `float32` in row 1 of the input array. Roughly 19.5% of retained trials are stimulated, and the converted range is exactly [0, 1].

ii.
```python
inputs[:, 1, :] = (
    np.isfinite(stim_start[:, None])
    & np.isfinite(stim_stop[:, None])
    & (centers >= stim_start[:, None])
    & (centers < stim_stop[:, None])
).astype(np.float32)
```

```python
| Trial `photostim_onset` + start time and `photostim_duration` | `input[1]` |
  1 where bin center is in half-open stimulation interval, else 0 | ... |
```

iii. The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)", and the target format states "If an input is a time such as onset of some stimulus, represent it as a binary time series." Step 3 justifies using the actual recorded interval rather than the nominal last-0.5-s-of-delay window. The half-open convention matches the neural binning convention (`metadata['spike_bin_interval_convention']`). Step 10 Check 4 confirms "photostimulation is binary" and that raw reconstruction passed `np.allclose`.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Both the stimulation bounds and the bin centres are expressed on the session-absolute clock, and the bin centres are the same `go + BIN_CENTERS` grid used for the firing rates, so the comparison is made directly on a shared axis with no offset correction. `kept_trial_starts`, `onset` and `duration` are indexed with the final `keep_idx` (after the stability, coverage and all-zero-neural filters), so trial ordering matches `rates` exactly.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
...
centers = centers_all[keep]
...
centers = centers[neural_present]
```

```python
onset = numeric_or_nan(trial_table["photostim_onset"][:n_trials_recorded])[keep_idx]
duration = numeric_or_nan(trial_table["photostim_duration"][:n_trials_recorded])[keep_idx]
kept_trial_starts = trial_starts[keep_idx]
stim_start = kept_trial_starts + onset
stim_stop = stim_start + duration
inputs[:, 1, :] = (... & (centers >= stim_start[:, None]) & (centers < stim_stop[:, None]))
```

iii. Step 4: "all clocks in NWB are absolute", so staying in absolute time and comparing against absolute bin centres avoids an extra go-relative conversion. The `--show-processing` plot draws the stimulation step function against the go-cue-marked time axis (Step 7: "stimulation is binary"), and the independent raw audit re-derives photostimulation state per trial and compares with `np.allclose` (Step 10 Check 2).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from two trials-table columns: `intervals/trials/trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `intervals/trials/outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side, a miss means it licked the opposite side, and an ignore means no lick. The AI additionally cross-validated this derivation against the raw `left_lick_times`/`right_lick_times` event streams (agreeing on 99.64% of trials) but uses the label-based derivation as authoritative.

ii.
```python
def choice_codes(outcomes: np.ndarray, instructions: np.ndarray) -> np.ndarray:
    """Recover the reported lick direction from outcome and instructed direction."""
    if not set(np.unique(outcomes)).issubset(set(OUTCOME_VALUES)):
        raise ValueError(f"Unknown outcomes: {np.unique(outcomes)}")
    if not set(np.unique(instructions)).issubset({"left", "right"}):
        raise ValueError(f"Unknown trial instructions: {np.unique(instructions)}")
    code = np.full(len(outcomes), 2, dtype=np.int8)  # ignore -> no lick
    hit = outcomes == "hit"
    miss = outcomes == "miss"
    code[hit & (instructions == "left")] = 0
    code[hit & (instructions == "right")] = 1
    code[miss & (instructions == "right")] = 0
    code[miss & (instructions == "left")] = 1
    return code
```

```python
outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
instructions = decode_array(trial_table["trial_instruction"][:n_trials_recorded])[keep_idx]
choices = choice_codes(outcomes, instructions)
```

iii. Step 1: "Behavioral conventions in preprocessing: `correctness` is 1 correct/free-water, 0 error, -1 no response; instructed `trial_type` is mapped left=1/right=0. Actual lick choice must come from response/lick fields rather than instructed trial type." Step 5 variable mapping: "`outcome` + `trial_instruction` → `output[0]` choice; ignore → no lick; hit → instructed direction; miss → opposite direction … This agrees with first response-period lick events on 99.64% of retained trials; trial labels are authoritative for exceptional/ambiguous lick trains."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A three-way categorical code (`0` left, `1` right, `2` no lick), one value per trial, broadcast across all 80 bins into row 0 of the `(4, 80)` `int8` output array so it can share one array with the time-varying tongue output. `output_values[0] = ['left', 'right', 'no lick']`. Both the outcome and instruction vocabularies are validated before mapping. Converted distribution: left 0.428 / right 0.422 / no lick 0.150.

ii.
```python
CHOICE_VALUES = ["left", "right", "no lick"]
```

```python
outputs = np.empty((n_trials, 4, N_TIME), dtype=np.int8)
outputs[:, 0, :] = choices[:, None]
```

```python
"output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
"output_values": [CHOICE_VALUES, OUTCOME_VALUES, EARLY_VALUES, TONGUE_VALUES],
```

iii. Step 5 Key Decision 1: "Mixed outputs are all 2D: choice/outcome/early labels are broadcast through time, producing integer `(4,80)` arrays so the time-varying tongue output can share one target array. This preserves their per-trial semantics." Codes follow the instruction ordering (left, right, no lick). Step 10 Check 5: "shapes are `(4,80)`, scalar outputs are constant within trial, requested ranges are exact, and raw label/tongue comparisons passed."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, a string column already containing exactly the three requested categories `'ignore'`, `'miss'`, `'hit'`. No derivation is needed.

ii.
```python
outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
```

```python
if not set(np.unique(outcomes)).issubset(set(OUTCOME_VALUES)):
    raise ValueError(f"Unknown outcomes: {np.unique(outcomes)}")
```

iii. Step 2 lists `outcome` among the `/intervals/trials` columns; Step 5 variable mapping: "`/intervals/trials/outcome` → `output[1]` outcome; Map ignore=0, miss=1, hit=2; Reference `correctness`: -1/0/1." The categories in the file line up one-to-one with the instruction's requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a dictionary built from `OUTCOME_VALUES` to `0` ignore, `1` miss, `2` hit, cast to `int8`, and broadcast across all 80 bins into row 1 of the output array. Converted distribution: ignore 0.150 / miss 0.166 / hit 0.684.

ii.
```python
OUTCOME_VALUES = ["ignore", "miss", "hit"]
```

```python
outcome_lookup = {name: i for i, name in enumerate(OUTCOME_VALUES)}
outcome_codes = np.asarray([outcome_lookup[x] for x in outcomes], dtype=np.int8)
...
outputs[:, 1, :] = outcome_codes[:, None]
```

iii. The code ordering follows the instruction ("Outcome (ignore, miss, hit, per-trial)"). Step 5: "Per-trial label broadcast across 80 timepoints to coexist with tongue time series." Step 4 reconciles the distribution with the papers: the converted 68.4% hit rate is lower than the papers' reported 84% because "Broader counts/rates are expected because required classes cannot be excluded. Reapplying the available regular mask gives pooled 81.6% hit among hit/miss; the paper's 84% reflects its analysis selection/version."

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from `intervals/trials/early_lick`, a string column holding `'no early'` / `'early'`. No derivation is needed; the vocabulary is validated before mapping.

ii.
```python
early_labels = decode_array(trial_table["early_lick"][:n_trials_recorded])[keep_idx]
if not set(np.unique(early_labels)).issubset({"no early", "early"}):
    raise ValueError(f"Unknown early-lick labels in {path}: {np.unique(early_labels)}")
```

iii. Step 2 lists an "early-lick label" among the trials-table columns, and Step 5 maps "`/intervals/trials/early_lick` → `output[2]` early lick; `no early`=0, `early`=1 … `early_lick_trials` in preprocessing". Step 3 notes the papers exclude early-lick trials from their analyses, but Step 4 records that the exclusion cannot be applied here because early lick is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean comparison against `'early'`, cast to `int8` (`0` no, `1` yes), broadcast across all 80 bins into row 2 of the output array. `output_values[2] = ['no', 'yes']`. Converted distribution: no 0.884 / yes 0.116.

ii.
```python
EARLY_VALUES = ["no", "yes"]
```

```python
early_codes = (early_labels == "early").astype(np.int8)
...
outputs[:, 2, :] = early_codes[:, None]
```

iii. The coding follows the instruction ("Early lick (no, yes, per-trial)"). Step 5 Key Decision 1 covers the broadcast-across-time representation. Because the early lick occurs during the sample or delay epoch, the event itself lies inside the -2.5 s pre-go window, so the label is decodable from the extracted neural window; the AI's decoder achieves 0.752 balanced validation accuracy against 0.5 chance.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is an `(n_frames, 3)` array of `(tongue_x, tongue_y, DeepLabCut likelihood)` with matching absolute `timestamps` at ~294 Hz. Column 1 is the y-position and column 2 is the likelihood used to decide visibility. The AI checks the array shape but hard-codes the column indices. Jaw and nose tracks and the three sessions with a duplicate Camera 3 series are deliberately ignored.

ii.
```python
tongue_group = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_t = tongue_group["timestamps"][:]
tongue_data = tongue_group["data"][:]
if tongue_data.ndim != 2 or tongue_data.shape[1] < 3:
    raise ValueError(f"Unexpected tongue data shape in {path}: {tongue_data.shape}")
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. Step 2: "`/acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` exists in every session and contains `(x, y, DeepLabCut likelihood)` at nominal 3.4-ms camera intervals with absolute timestamps. Jaw and nose tracks also exist but are not requested. Camera 3 duplicates exist in three files and are not the canonical side camera." Step 3: "The method paper uses side-view video only. DeepLabCut tracks tongue, jaw, and nose."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:

1. **Visibility**: a frame counts as visible when `tongue_y` and `likelihood` are finite and `likelihood >= 0.9` (the standard DeepLabCut confidence threshold). The AI notes the likelihood is strongly bimodal, so the exact threshold barely matters.
2. **Session percentiles**: the 40th and 60th percentiles of `tongue_y` are computed over **all visible raw frames in the session** (not over bin averages, and not over the extracted trial windows only).
3. **Point-sampling onto the bin grid**: for each bin centre, `np.searchsorted(..., side="right") - 1` finds the nearest *preceding* camera frame; that frame is used if its lag is within 10 ms (i.e. it is genuinely contemporaneous rather than the last frame before a video gap). No averaging over the bin is performed.
4. **Discretisation**: visible samples map to 0/1/2 against the session percentiles; everything else (low likelihood, non-finite, or no contemporaneous frame) maps to class 3 "not visible".

Converted distribution: 0.062 / 0.032 / 0.065 / 0.841. The reference papers' outlier removal and mean-imputation of occluded tongue positions are deliberately not reproduced.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
TONGUE_VALUES = ["< 40th percentile", "40th to 60th percentile",
                 "> 60th percentile", "not visible"]
```

```python
session_visible = (
    np.isfinite(tongue_y)
    & np.isfinite(tongue_likelihood)
    & (tongue_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
)
if np.sum(session_visible) < 2:
    raise ValueError(f"Insufficient visible tongue samples in {path}")
q40, q60 = np.percentile(tongue_y[session_visible], [40, 60])
```

```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
safe_frame_idx = np.clip(frame_idx_all, 0, len(tongue_t) - 1)
frame_lag = centers_all - tongue_t[safe_frame_idx]
# 10 ms is about three 300-Hz frames and robust to the documented 3.4-ms
# nominal spacing. ...
frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
```

```python
sampled_y = tongue_y[frame_idx]
sampled_likelihood = tongue_likelihood[frame_idx]
sampled_visible = (
    np.isfinite(sampled_y)
    & np.isfinite(sampled_likelihood)
    & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    & frame_is_current
)
```

iii. Step 1 identified `align_embedding_vecs_between_lims`, which "Aligns video-derived samples to go cue and uses the most recent frame in each target interval" — the AI reuses this nearest-preceding-frame convention. Step 3: "Reference code … aligns video/markers by go cue and selects the last source frame in each target interval. This supports sampling behavioral state at each decoder bin center from the nearest preceding camera frame." Step 2: "Tongue likelihood is strongly bimodal (example: only 10.5% of frames >=0.9), supporting the standard DLC likelihood threshold to define visibility." Step 4: "Method paper imputes occluded tongue to mean for continuous movement-to-neural regression … Requested output explicitly has class 3 'not visible' → Preserve occlusion as class 3 using DLC likelihood <0.9; compute y percentiles only from visible samples." Step 5: "Percentiles exclude invisible coordinates because those coordinates are tracking artifacts."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session, with class edges at the 40th and 60th percentiles of visible tongue-y over that session. Boundary convention: `y < q40` → 0, `q40 <= y <= q60` → 1, `y > q60` → 2, everything not visible → 3. The array is initialised to 3 and the three visible classes are written over it, so class 3 is the default for any sample that fails the visibility test. The per-session `q40`/`q60` values are recorded in `metadata['session_info']` and drawn as horizontal lines on the `--show-processing` plot.

ii.
```python
tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)
tongue_codes[sampled_visible & (sampled_y < q40)] = 0
tongue_codes[sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
tongue_codes[sampled_visible & (sampled_y > q60)] = 2
```

```python
"tongue_q40": float(q40),
"tongue_q60": float(q60),
...
"tongue_percentile_scope": "all visible Camera0 side-view frames within each session",
```

iii. The 40th/60th split and the per-session scope are specified verbatim by the Decoder Task instructions; the fourth "not visible" class is likewise specified. Step 5: "Boundary convention: `<q40`, `q40<=y<=q60`, `>q60`." Step 7: "tongue classes change only on visible frames and respect the plotted session thresholds. No temporal shift or percentile inversion was observed." Step 10 Check 5: "The session percentile convention and class boundaries were rechecked."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output. Camera timestamps are on the same session-absolute clock as spikes and go cues, so each bin's value is read from the nearest preceding camera frame to that bin's centre (`go + BIN_CENTERS`) — the identical grid used for the firing rates. A 10 ms staleness tolerance guards against video gaps: when the preceding frame is older than 10 ms (as happens in sessions where the camera only runs during trial segments, leaving seconds-long inter-trial gaps), the bin is marked "not visible" rather than being filled from a stale frame. Frame indices and lags are computed for all recorded trials and then subset with the same `keep` and `neural_present` masks as the neural data.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
safe_frame_idx = np.clip(frame_idx_all, 0, len(tongue_t) - 1)
frame_lag = centers_all - tongue_t[safe_frame_idx]
frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
video_has_data = np.any(frame_is_current_all, axis=1)
```

```python
frame_idx = safe_frame_idx[keep]
frame_is_current = frame_is_current_all[keep]
...
frame_idx = frame_idx[neural_present]
frame_is_current = frame_is_current[neural_present]
```

iii. Step 5: "sample nearest preceding frame at each bin center", reusing the reference code's `align_embedding_vecs_between_lims` convention. Step 9: "The first full attempt exposed that several NWB files store camera timestamps only in within-trial segments rather than densely through inter-trial gaps. Treating every gap as invalid caused one session to have fewer than two trials. It was corrected so a source frame is contemporaneous when the preceding frame is within 10 ms and missing/noncontemporaneous frames map to tongue class 3." The inline comment justifies the tolerance: "10 ms is about three 300-Hz frames and robust to the documented 3.4-ms nominal spacing."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five distinct cases, handled by either exclusion (where nothing was recorded) or explicit categorisation (where the measurement legitimately has no value):

- **Session never quality-controlled** (`classification`/`anno_name` all NaN): `decode_scalar` turns the NaN into the string `'nan'`, no unit matches `'good'`, and `convert_session` returns `None` so the session is skipped with a printed message.
- **Trials past the end of ephys coverage** (including a documented off-by-one terminal `obs_intervals` entry, and the `free_water` trials which are covered by `obs_intervals` yet contain no spikes): caught by the all-zero population check and dropped, with the count printed per session.
- **Trials with unstable units**: excluded via `is_good_trials`.
- **`'N/A'` string entries in the numeric photostim columns**: `numeric_or_nan` catches the conversion failure, leaves `NaN`, and the `np.isfinite` guards force those trials to photostimulation 0.
- **Missing / low-confidence / stale tongue frames**: mapped to the explicit "not visible" class 3 rather than imputed or deleted.

Beyond this, the script is deliberately fail-fast: it `raise`s on subject/folder mismatch, curated units without anatomy, obs-interval/stability count disagreement, a tone outside `[trial_start, go]`, unknown outcome/instruction/early-lick vocabularies, non-finite converted values, negative or non-20-Hz-quantised rates, and bad shapes. Note that several of these (fewer than 2 valid trials, insufficient visible tongue samples) `raise` and would abort the whole run rather than skip the session as the session-level `return None` path does.

ii.
```python
classification = decode_array(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
if len(good_indices) == 0:
    print(f"SKIP {sid}: no classifier-curated units", flush=True)
    return None
```

```python
def numeric_or_nan(values) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        try:
            out[i] = float(decode_scalar(value))
        except (TypeError, ValueError):
            pass
    return out
```

```python
neural_present = np.any(rates != 0, axis=(1, 2))
if not np.all(neural_present):
    print(f"  {sid}: excluding {np.sum(~neural_present)} all-zero "
          "population trial(s) beyond spike coverage", flush=True)
```

```python
tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)
```

```python
if not (np.isfinite(rates).all() and np.isfinite(inputs).all() and np.isfinite(outputs).all()):
    raise AssertionError(f"Non-finite converted value in {path}")
if np.any(rates < 0) or not np.allclose(rates / 20.0, np.round(rates / 20.0)):
    raise AssertionError(f"Firing rates are not valid 50-ms counts in {path}")
```

iii. Step 4: the all-NaN session "cannot form a valid neural decoder session"; excluding it makes the session count match the papers' 173. Step 7 documents the discovery of the all-zero trial: "Direct raw inspection found every curated spike train ended at 1107.44 s while that trial began at 1109.39 s, despite the NWB observation table including it. The converter now excludes all-zero population windows as missing neural coverage." The inline comment justifies the heuristic: "With hundreds of units, a completely silent four-second window is a reliable missing-ephys marker." Step 10 Check 7 lists the edge cases checked: "shorter stability matrices, all-NaN unit-QC session, false stability entries, optional numeric `N/A` byte strings, long sample-to-go intervals, terminal/interleaved all-zero neural windows, camera segment gaps, absent video windows, low-likelihood/NaN tongue coordinates, half-open bin endpoints, percentile equality, and sessions with very small trial counts."

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion took 188.8 s wall-clock for 174 files (0.26–2.29 s per session, scaling with unit count), of which the inventory pre-pass was 4.65 s and the final pickle write of the 10.97 GiB result was 13.56 s. Within `convert_session` the cost is dominated by HDF5 I/O and the per-unit binning:

- reading the ragged `spike_times` buffer one unit-slice at a time (up to ~11.5 M doubles per session) and running one `np.searchsorted` per unit over all 81 × n_trials edges (~40 k–65 k edges per session × up to 923 units);
- reading the full tongue-tracking array (`timestamps` plus `(n_frames, 3)` data, ~680 k × 3 per session);
- `decode_array` Python list comprehensions over `classification` and `anno_name` (up to ~4,000 entries each, done once in the inventory pass and again in the conversion pass) and over the per-trial string columns.

The AI instruments this: per-session elapsed time and neural payload size are printed for every session, plus the inventory time, pickle-write time and total.

ii.
```python
started = time.perf_counter()
...
elapsed = time.perf_counter() - started
mib = rates.nbytes / (1024 ** 2)
print(f"DONE {sid}: {n_trials} trials, {n_units_good} neurons, "
      f"{mib:.1f} MiB neural, {elapsed:.2f} s", flush=True)
```

```python
print(f"Inventory: {len(subjects)} subjects, {len(brain_regions)} fine CCF regions "
      f"({time.perf_counter() - inventory_start:.2f} s)", flush=True)
```

```python
print(f"WROTE {output_path}: {len(converted)} sessions, {total_trials} trials, "
      f"{total_neurons} session-units, {size_gib:.3f} GiB; "
      f"pickle write {write_time:.2f} s; total {time.perf_counter() - overall_start:.2f} s", flush=True)
```

iii. Step 6: "Naive unit x trial loops and repeated HDF5 access would dominate runtime; loading all 53.6 GB or materializing per-spike trial assignments would inflate memory. Full output is intrinsically about 12 GB as float32." Step 6 speedups: "For each unit, one vectorized `searchsorted` call bins all trial edges. Small metadata arrays are read once/session, raw spikes are sliced once/unit, and only one session-rate tensor is added at a time." Step 7's estimate ("conservatively 5–7 min") was borne out; Step 9 reports "The full conversion took 177.14 s, well below the 15-minute limit", so no further optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The expensive nested unit × trial loop was collapsed: the per-unit loop issues a single `searchsorted` over *all* trials' edges at once. That remaining per-unit loop cannot be vectorised further because `spike_times` is ragged — each unit has a different number of spikes, so there is no single sorted array to search. The tongue path contains **no** Python loop at all: frame lookup, staleness testing and discretisation are all fully vectorised over `(n_trials, 80)`.

The loops that remain and *could* be vectorised, all minor, are Python-level string/number handling:

- `decode_array` — a per-element list comprehension, called on `classification` and `anno_name` (up to ~4,000 units) in both the inventory and conversion passes, and on the `outcome`, `trial_instruction`, `early_lick` columns (up to 800 trials each). h5py's `.asstr()` or `np.char.decode` would do this in C.
- `numeric_or_nan` — a per-element `try`/`except float()` over the two photostim columns; could be done with `np.char`-based masking plus a single vectorised `astype(float)`.
- `outcome_codes = np.asarray([outcome_lookup[x] for x in outcomes], ...)` — a per-trial dict lookup; `np.searchsorted` over the sorted vocabulary would vectorise it.
- The final `[rates[i] for i in range(n_trials)]` / `[inputs[i] ...]` / `[outputs[i] ...]` comprehensions are required by the target format (a list of per-trial arrays) and are views, not copies.

ii.
```python
# Binning is vectorized over all trials for each unit. searchsorted at all
# edges avoids a Python trial loop and exactly implements [left,right).
absolute_edges = go[:, None] + BIN_EDGES[None, :]
for out_unit, source_unit in enumerate(good_indices):
    ...
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
    counts = np.diff(edge_positions.reshape(n_trials, N_TIME + 1), axis=1)
    rates[:, out_unit, :] = counts / BIN_WIDTH_S
```

```python
def decode_array(values) -> np.ndarray:
    return np.asarray([decode_scalar(x) for x in values])
```

```python
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
safe_frame_idx = np.clip(frame_idx_all, 0, len(tongue_t) - 1)
frame_lag = centers_all - tongue_t[safe_frame_idx]
frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
```

iii. Step 6: "Code speedups added: For each unit, one vectorized `searchsorted` call bins all trial edges." Step 7 run-time table: "Vectorized all-trial `searchsorted` per unit | Avoids about 37 million Python unit/trial loops in full mode". The AI did not document the residual string-decoding loops; they were evidently judged (correctly) not to be the bottleneck, since the measured 188.8 s total was well inside the 15-minute budget stated in the instructions.

## 10-c. What processing does the code repeat multiple times?

i. One substantive repetition: **every NWB file is opened and read twice**. `discover_inventory` opens all 174 files to read `general/subject/subject_id`, `units/classification` and `units/anno_name`; `convert_session` then re-opens each file and re-reads and re-decodes `units/classification` and `units/anno_name`. That is a full extra pass over the dataset's metadata, including duplicate `decode_array` list comprehensions over up to ~4,000 unit labels per file. Measured cost: 4.65 s of 188.8 s (~2.5%).

Smaller repetitions: `go_all`, `trial_starts` and the derived `centers_all` / `frame_idx_all` / `frame_lag` are computed over all `n_trials_recorded` trials and then re-subset twice (by `keep`, then by `neural_present`); `Path(path).name` parsing happens in both passes. Firing rates are also computed for the 2,548 trials that are subsequently discarded as all-zero — unavoidable, since the all-zero test is defined on the binned rates themselves.

Nothing else is recomputed: each session's spikes, video and trial table are read once, and the bin grid (`BIN_EDGES`, `BIN_CENTERS`) is built once at module level and reused for every trial and session.

ii.
```python
subjects, brain_regions = discover_inventory(paths)   # opens all 174 files
subject_lookup = {name: i for i, name in enumerate(subjects)}
region_lookup = {name: i for i, name in enumerate(brain_regions)}

converted = []
for path in paths:
    result = convert_session(path, subject_lookup, region_lookup, ...)  # opens them again
```

```python
# inventory pass
classification = decode_array(nwb["units/classification"][:])
good = classification == "good"
annotations = decode_array(nwb["units/anno_name"][:])[good]
```

```python
# conversion pass — the same two reads and decodes repeated
classification = decode_array(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
annotations = decode_array(units["anno_name"][:])[good_indices]
```

```python
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_WIDTH_S)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The AI's stated rationale for the pre-pass is a stable, globally sorted vocabulary: Step 5 variable mapping requires a "Global sorted list of fine Allen CCF annotations; integer lookup per unit" and "Stable sorted subject identifiers and integer indices", and Key Decision 7 requires `--sample` to be "deterministic" — which means subject and region indices must not depend on which sessions were processed. The pre-pass also front-loads the subject/folder and anatomy consistency assertions so a malformed file fails before any expensive work. Step 6 claims "Small metadata arrays are read once/session, raw spikes are sliced once/unit"; the duplicated inventory read is not acknowledged as redundant anywhere in CONVERSION_NOTES.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small pieces of work whose results never reach the decoder:

- **`coverage_mask = np.isfinite(tone_all)` is dead code.** `first_tone_onsets` raises on any invalid tone, so by construction `tone_all` is always all-finite and `coverage_mask` is always all-`True`; the `& coverage_mask` in `keep` is a no-op. The metadata nevertheless advertises "finite first tone onset" as a trial filter.
- **`video_has_data`** is computed with an `np.any` reduction over the full `(n_trials_recorded, 80)` staleness array but is only used to populate the `n_trials_with_any_video` diagnostic in `session_info`. It was the basis of a filter that Step 10 deliberately removed.
- **`n_trials_window_coverage`** is assigned `n_trials` and reported as a separate statistic, duplicating `n_trials_coverage`/`n_trials_converted` (all three are equal in the output).
- **`auto_water` / `free_water`** columns are read and counted purely for `session_info`; neither is used for filtering.
- **Firing rates are computed for the 2,548 trials later dropped** as all-zero, and `centers_all` / `frame_idx_all` / `frame_lag` are computed for all `n_trials_recorded` trials including the ~509 later dropped for instability.
- **`make_processing_plot`** builds eight panels (including `Counter` over region names and a full `rates.mean(axis=(0,1))`), but only under `--show-processing` for at most two sessions.

None of this is large: together it is a low-single-digit percentage of the 188.8 s runtime, and most of it produces documentation/QC metadata rather than being truly wasted.

ii.
```python
video_has_data = np.any(frame_is_current_all, axis=1)
# An entirely missing video window is still a valid observation for this
# task: the requested tongue class 3 explicitly means "not visible".
coverage_mask = np.isfinite(tone_all)
keep = stable_mask & coverage_mask
```

```python
auto_water = trial_table["auto_water"][:n_trials_recorded][keep_idx]
free_water = trial_table["free_water"][:n_trials_recorded][keep_idx]
session_stats = {
    ...
    "n_trials_window_coverage": int(n_trials_window_coverage),
    "n_trials_with_any_video": int(np.sum(video_has_data[keep_idx])),
    "n_trials_coverage": int(n_trials),
    "n_trials_converted": int(n_trials),
    ...
    "auto_water_trials_converted": int(np.sum(auto_water != 0)),
    "free_water_trials_converted": int(np.sum(free_water != 0)),
}
```

```python
neural_present = np.any(rates != 0, axis=(1, 2))
if not np.all(neural_present):
    ...
    rates = rates[neural_present]
```

iii. Step 5 Key Decision 6 explains the water columns: "auto/free-water flags are retained in per-session metadata counts rather than used as exclusions" — i.e. they are kept deliberately, as documentation of what was *not* filtered, to support the Step 9/10 reconciliation of trial counts against the reference papers. The `video_has_data` residue is a by-product of Step 10 Iteration 2, where the AI removed a video-presence filter it had decided was wrong ("requiring even one video frame still contradicted that policy. Removing this unintended filter recovered 754 neurally valid trials") but left the computation in place as a reported statistic. The vacuous `coverage_mask` and the duplicated `n_trials_window_coverage` field are not discussed in CONVERSION_NOTES.
