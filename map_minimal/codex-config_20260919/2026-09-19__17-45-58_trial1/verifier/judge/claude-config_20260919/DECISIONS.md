# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI "Mesoscale Activity Map" NWB release, one `.nwb` file per session under `/app/data/sub-<subject_id>/`. The AI enumerates every session with a single sorted glob and opens each file exactly once. Unlike the human reference, it does **not** use `pynwb`; it opens the files as plain HDF5 with `h5py` and reads the NWB datasets by their internal paths (`units/...`, `intervals/trials/...`, `acquisition/BehavioralEvents/...`, `acquisition/BehavioralTimeSeries/...`). All 174 files are visited; 173 are converted (one is dropped, see 2-c).

ii.
```python
def convert(data_dir: Path, output_path: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")

    converted_sessions = []
    skipped_sessions = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:3d}/{len(paths)}] {path.name}", flush=True)
        session = _convert_session(path)
```

```python
def _convert_session(path: Path) -> dict | None:
    with h5py.File(path, "r") as nwb:
        classification = _decode_array(nwb["units/classification"])
        ...
        all_go_times = nwb[
            "acquisition/BehavioralEvents/go_start_times/timestamps"
        ][:].astype(np.float64)
        n_source_trials = len(nwb["intervals/trials/id"])
```

iii. The AI first opened a file with `pynwb` to learn the schema (trajectory step 14, which printed the full `NWBFile` field listing), then switched to raw `h5py` reads for the production converter. Its audit (step 20) established the boundaries of the dataset before writing any code: "sessions 174 subjects 28 trials 94990 good 69453 allunits 272227", and it confirmed that every file contains the tongue-tracking series and that the trials table schema is identical across all files. Its summary message (step 39) states: "The NWB audit resolves the dataset boundary cleanly: 174 files are present, but one has no classifier-labeled good units; retaining the remaining 173 sessions yields 69,453 curated units across all 28 mice." Using `h5py` rather than `pynwb` was a speed choice — it avoids building the pynwb object model and lets the ragged `spike_times` buffer be read in one shot; the conversion runs in ~104 s for all 174 files.

## 1-b. How are the data split into subjects?

i. Each session is assigned to the mouse encoded in its containing directory name (`sub-440956` → `'440956'`). The unique ids are sorted to form `subjects`, and `subject_idx` maps each converted session to its index in that list. This yields 28 subjects with 3–10 sessions each.

ii.
```python
subject_id = path.parent.name.removeprefix("sub-")
```

```python
subjects = sorted({session["subject"] for session in converted_sessions})
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray(
    [subject_lookup[session["subject"]] for session in converted_sessions],
    dtype=np.int32,
),
```

iii. The AI counted sessions per subject directly from the directory layout during its audit (step 20: `subjects Counter({'456772': 10, '480927': 10, ..., '456774': 3})`, 28 subjects), confirming that the folder name is a complete and consistent animal grouping. It used the numeric DANDI subject id (identical to the `nwb.subject.subject_id` field the human reference reads) rather than the mouse name used in the papers (e.g. `SC015`).

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is performed. Session order in the output is the sorted file order (which is alphabetical by subject then by acquisition timestamp, so chronological within each mouse). Each session is identified in metadata by its relative file path plus per-session counts.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
```

```python
"session_info": {
    "source_file": str(path.relative_to(path.parents[1])),
    "n_source_trials": int(n_source_trials),
    "n_ephys_mask_trials": int(n_ephys_trials),
    "n_common_good_trials": int(len(common_good_idx)),
    "n_excluded_water_trials": int(np.count_nonzero(water_mask)),
    "n_population_recording_dropouts": n_population_dropouts,
    "n_trials": int(n_trials),
    "n_good_units": int(len(good_rows)),
    "tongue_y_40th_percentile": tongue_q40,
    "tongue_y_60th_percentile": tongue_q60,
},
```

iii. The file boundary is the session boundary in this release, so nothing has to be inferred. The AI's per-session audit (step 20) verified one trials table and one unit table per file. It records the source filename (which embeds `ses-<timestamp>`) rather than the `nwb.identifier` string, plus a per-session curation ledger so that every trial dropped from a session can be traced.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioural trial. The AI asserts that the number of go-cue events equals the number of trial-table rows, so the go cue → trial mapping is one-to-one, and then uses the go cue as the per-trial anchor.

ii.
```python
all_go_times = nwb[
    "acquisition/BehavioralEvents/go_start_times/timestamps"
][:].astype(np.float64)
n_source_trials = len(nwb["intervals/trials/id"])
if len(all_go_times) != n_source_trials:
    raise ValueError(
        f"{path.name}: {len(all_go_times)} go cues for "
        f"{n_source_trials} trials"
    )
```

Each spike is assigned to exactly one trial window by a binary search on the window starts:
```python
# Go-cue separations in this dataset exceed the four-second analysis window,
# so the most recent window start identifies a spike's unique trial.
trial = np.searchsorted(window_starts, spikes, side="right") - 1
```

iii. The agent's exploration (steps 31–32, 52–53) checked the go-cue stream against the trials table across all sessions and found exactly one go cue per row everywhere, which is why the assertion never fires. It also explicitly checked the trial spacing and stated the invariant it relies on in a comment: go-cue separations exceed the 4 s analysis window, so windows never overlap and each spike belongs to at most one trial. (I verified this independently: the minimum go-cue separation anywhere in the dataset is 4.58 s.)

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied in order, all aimed at "does this trial have valid simultaneous ephys, and is it a regular behavioural trial":

1. **Ephys coverage** — a trial is kept only if **every retained good unit** marks it valid in `units/is_good_trials`. In the 8 sessions where the behavioural table is longer than the ephys trial mask, the compact mask is first mapped back onto the source trials via the common `obs_intervals` window, with an assertion that the mapped count matches the mask width.
2. **Water trials** — `auto_water` or `free_water` trials are removed, following the repository's `get_regular_trial_mask`.
3. **Population dropout** — any remaining trial in which *no* good unit fired a single spike is removed as missing acquisition.
4. A session with fewer than 2 surviving trials raises (the format requires ≥2 trials); this never triggered.

Early-lick, no-response (`ignore`), and photostimulation trials are **deliberately retained**, because they are required decoder classes/inputs. Result: 89,068 trials from 94,370 in the retained sessions.

ii.
```python
good_trial_matrix = nwb["units/is_good_trials"][good_rows, :]
n_ephys_trials = good_trial_matrix.shape[1]
...
if n_ephys_trials == n_source_trials:
    ephys_source_idx = np.arange(n_source_trials)
else:
    observation_intervals = nwb["units/obs_intervals"][:]
    observation_ends = nwb["units/obs_intervals_index"][:]
    ...
    common_start = max(unit_starts)
    common_stop = min(unit_stops)
    ephys_source_idx = np.flatnonzero(
        (all_go_times + OFF_START >= common_start - 1e-6)
        & (all_go_times + OFF_END <= common_stop + 1e-6)
    )
    if len(ephys_source_idx) != n_ephys_trials:
        raise ValueError(...)

common_good_mask = np.all(good_trial_matrix, axis=0)
common_good_idx = ephys_source_idx[common_good_mask]
auto_water_all = nwb["intervals/trials/auto_water"][:].astype(bool)
free_water_all = nwb["intervals/trials/free_water"][:].astype(bool)
water_mask = auto_water_all[common_good_idx] | free_water_all[common_good_idx]
valid_trial_idx = common_good_idx[~water_mask]
if valid_trial_idx.size < 2:
    raise ValueError(f"{path.name}: fewer than two common valid trials")
```

```python
# One final recorded trial in the release is marked valid in NWB but has
# no spikes from any cluster. Treat this population-wide dropout as
# missing acquisition, not simultaneous biological silence.
population_recorded = np.any(rates != 0, axis=(1, 2))
n_population_dropouts = int(np.count_nonzero(~population_recorded))
if n_population_dropouts:
    rates = rates[population_recorded]
    valid_trial_idx = valid_trial_idx[population_recorded]
    go_times = go_times[population_recorded]
    n_trials = len(go_times)
```

iii. This filter set was built empirically, driven by the validator's own warnings. The AI first converted with **no** trial filter (94,370 trials), then ran `train_decoder.py --verify-only`, which flagged sessions where "all neural data is zero" (step 70). It then investigated the cause instead of suppressing the warning (step 51: "The validator found an important source-data edge case rather than a format error: some sessions contain long blocks where every retained unit is silent. I'm checking whether those blocks fall outside the units' NWB observation intervals"). Its audit (step 54) tabulated the three coverage regimes across the dataset — 161 files where the mask matches the trials table, 8 where ephys is shorter than behaviour, and 4 where probes disagree — with totals `trials raw/ephys/allgood [94370 93310 92801]`, and it concluded (step 57): "eight NWBs continue the behavioral table after ephys acquisition ends, and four have slightly different probe observation intervals. I've tightened curation to the intersection of `is_good_trials` across retained units. This keeps every requested behavioral class while excluding only trials without common neural coverage."

The water exclusion is justified by direct citation of the reference repository: the AI grepped out `get_regular_trial_mask` (step 77/78), which is `(early_lick == 0) * (auto_water == 0) * (free_water == 0) * (correctness != -1)`, and then deliberately applied only the two water clauses (step 79): "The repository's regular-trial mask excludes both free-water and auto-water trials. I'll apply those two source exclusions while deliberately retaining early-lick, no-response, and photostimulation trials required by this decoder task." It also verified empirically that `free_water` trials contain no spikes at all (steps 71–75: the 24 all-zero trials in `sub-456772_ses-20191119` are all `free_water=1`), and counted the free/auto overlap (step 76: 2,427 free-only, 1,316 auto-only, 23 both).

The population-dropout rule was added last, after a single residual all-zero trial survived every other filter (step 82 examines trial 159 of `sub-440956_ses-20190208` directly). After all four filters the validator reports zero errors and zero warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a ragged concatenated buffer) together with `units/spike_times_index` (the per-unit end offsets), restricted to the rows selected by `units/classification == 'good'`. The go-cue timestamps (`acquisition/BehavioralEvents/go_start_times/timestamps`) supply the window anchor for every trial.

ii.
```python
all_spikes = nwb["units/spike_times"][:]
spike_ends = nwb["units/spike_times_index"][:]
rates = _bin_good_units(all_spikes, spike_ends, good_rows, go_times)
del all_spikes
```

```python
for out_unit, unit_row in enumerate(good_rows):
    start = 0 if unit_row == 0 else int(spike_ends[unit_row - 1])
    stop = int(spike_ends[unit_row])
    spikes = all_spikes[start:stop]
```

iii. Sorted spike times are the only neural representation in the file. The agent (step 10) had already established from the reference code that "spikes are already relative to the go cue, firing rate is spike count divided by bin width, and published analyses retain classifier-labeled 'good' units with histology", and it mapped those conventions onto the NWB fields. It reads the whole ragged buffer once rather than slicing the HDF5 dataset per unit (it checked chunking/compression of `units/spike_times` in step 62 before deciding).

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, converted to firing rate in Hz by dividing by the bin width. No smoothing, no normalisation, no baseline subtraction, no z-scoring. Implementation: for each good unit, every spike is assigned to a trial (binary search on the window starts) and a time bin (floor of the offset from the window start divided by the bin width), the (trial, bin) pair is flattened into a single index, and one `np.bincount` produces the entire (trial × bin) count matrix for that unit.

ii.
```python
        trial = np.searchsorted(window_starts, spikes, side="right") - 1
        in_range = trial >= 0
        trial = trial[in_range]
        spikes = spikes[in_range]
        in_range = (trial < n_trials) & (spikes < window_stops[trial])
        trial = trial[in_range]
        spikes = spikes[in_range]
        if spikes.size == 0:
            continue

        # The tiny epsilon only stabilizes decimal timestamps that are a few ulps
        # below an exact edge. Bins remain left-closed and right-open.
        time_bin = np.floor(
            (spikes - window_starts[trial]) / BIN_SIZE_S + 1e-10
        ).astype(np.int64)
        valid = (time_bin >= 0) & (time_bin < N_BINS)
        flat_bin = trial[valid] * N_BINS + time_bin[valid]
        counts = np.bincount(flat_bin, minlength=n_trials * N_BINS)
        rates[:, out_unit, :] = counts.reshape(n_trials, N_BINS) / BIN_SIZE_S
```

iii. The Hz convention is taken straight from the reference repository's `sliding_histogram`, which returns `binSpikes/float(bin_width)` when `rate=True` (the agent read this file in step 9 and summarised it in step 10: "firing rate is spike count divided by bin width"). The agent cross-validated its own implementation against NumPy histograms before running the full conversion (step 42: "a session-level cross-check matches NumPy histograms exactly for sampled neurons/trials"). I re-ran that check independently on `sub-440956_ses-20190207` over 25 random (unit, trial) pairs and found zero mismatches against `np.histogram`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept — the verdict of the published spike-sorting QC classifier. No individual metric thresholds are applied, and the older `unit_quality` label is not used. A session with no `'good'` units is dropped entirely. The code additionally asserts that every retained good unit carries a non-empty Allen CCF annotation. This retains 69,453 of 272,227 units (25.5%) over 173 sessions, a mean of 401 units per session (min 90, max 923).

ii.
```python
classification = _decode_array(nwb["units/classification"])
good_rows = np.flatnonzero(classification == "good")
if good_rows.size == 0:
    return None
```

```python
annotations = _decode_array(nwb["units/anno_name"])[good_rows]
if np.any(annotations == ""):
    raise ValueError(f"{path.name}: a good unit lacks an Allen annotation")
```

iii. The module docstring states the rule and its provenance: "Keep only units labeled ``good`` by the published region-specific QC classifiers. These are also the units with anatomical ``anno_name`` values." The agent's dataset audit (step 20) enumerated the label distribution — `Counter({'unlabelled': 200922, 'good': 69453, '': 1852})` — showing that `classification` is three-valued and that exactly one session (`sub-440958_ses-20190216T162508`, 1,852 units) has NaN labels, i.e. was never quality-controlled. It reported (step 39): "174 files are present, but one has no classifier-labeled good units; retaining the remaining 173 sessions yields 69,453 curated units across all 28 mice", matching the QC white paper's 173-session figure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed: the per-trial window is simply `[go - 2.5 s, go + 1.5 s)` in absolute time, and each spike's bin index is computed as its offset from that trial's window start.

ii.
```python
window_starts = go_times + OFF_START
window_stops = go_times + OFF_END
...
trial = np.searchsorted(window_starts, spikes, side="right") - 1
...
in_range = (trial < n_trials) & (spikes < window_stops[trial])
...
time_bin = np.floor(
    (spikes - window_starts[trial]) / BIN_SIZE_S + 1e-10
).astype(np.int64)
```

iii. The docstring records the assumption explicitly: "Use the NWB go-cue timestamps as the common clock. Spike timestamps, video timestamps, task events, and trial intervals are all expressed in this clock." The agent verified the shared clock in step 33 by checking that `trials/start_time + photostim_onset` lands exactly on the independent `photostim_start_times` event timestamps. The trial-assignment shortcut is guarded by the documented invariant that go cues are more than 4 s apart, so no spike can fall in two windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, fixed. 80 non-overlapping, left-closed/right-open bins spanning −2.5 s to +1.5 s relative to the go cue, identical for every trial and every session. There is no rebinning step: spikes are binned once, directly at the target resolution, from raw spike times. Bin centres run from −2.475 s to +1.475 s. Note that the reference repository uses *overlapping* sliding bins (40 ms width, 3 ms stride); the AI deliberately uses non-overlapping 50 ms bins because the instructions specify that width and the decoder format requires one value per timepoint.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE_S))
BIN_EDGES = OFF_START + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
"time_bin_size": 50.0,
"binning": "80 non-overlapping, left-closed 50-ms bins",
"bin_centers_relative_to_go_s": BIN_CENTERS.tolist(),
```

iii. The docstring records "Convert non-overlapping 50-ms spike-count bins to Hz by dividing by 0.05 s", and the summary message (step 42) states: "It produces 80 left-closed bins centered from −2.475 to +1.475 s". The window and bin width come directly from the Decoder Task specification. Defining the grid once at module level guarantees a constant 80 timepoints per trial, which the target format requires; the validator confirms `T_min = T_max = 80`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch tone onsets) together with the trial's go-cue time. Because a lick during the sample or delay epoch causes the epoch to be replayed, a single trial can contain several sample-start events; the AI takes the **last** tone onset at or before the go cue.

ii.
```python
def _tone_onsets(go_times: np.ndarray, sample_starts: np.ndarray) -> np.ndarray:
    """Find the final tone onset before each go cue (replays produce extras)."""
    idx = np.searchsorted(sample_starts, go_times, side="right") - 1
    if np.any(idx < 0):
        raise ValueError("A go cue has no preceding sample/tone onset")
    return sample_starts[idx]
```

```python
sample_starts = nwb[
    "acquisition/BehavioralEvents/sample_start_times/timestamps"
][:]
tone_onset = _tone_onsets(go_times, sample_starts)
```

iii. The function's own docstring gives the reason: replays produce extra sample-start events, so the event stream cannot be indexed one-per-trial, and the tone the animal actually used is the last one before the go cue. The agent's exploration of the behavioural event streams (steps 31–33) established that `sample_start_times` is not one-per-trial while `go_start_times` is, which is why the go cue is the trial anchor and the tone is looked up relative to it. The guard raises if any go cue has no preceding tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute time of every bin centre is `go + BIN_CENTERS`; subtracting that trial's tone onset gives seconds elapsed since the tone at each bin. The result is a continuous, time-varying float32 row, stored as row 0 of the `(2, 80)` input array. No clipping, no normalisation.

ii.
```python
for trial in range(n_trials):
    time_from_tone = (
        go_times[trial] + BIN_CENTERS - tone_onset[trial]
    ).astype(np.float32)
    ...
    inputs.append(np.stack((time_from_tone, photo_on), axis=0))
```

```python
"input_names": ["time from tone onset (s)", "photostimulation on"],
```

iii. The Decoder Task specifies this input as "continuous, time-varying", so no discretisation or binarisation is applied; the only real work is locating the correct tone onset (3-a). The realised range in the converted file is [−1.525, 11.894] s, consistent with the nominal 0.65 s sample + 1.2 s delay structure plus replayed sample epochs on early-lick trials.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same go-cue-anchored bin grid used for the firing rates — the shared `BIN_CENTERS` constant. Bin `k` of the input therefore describes the same 50 ms interval as bin `k` of the neural matrix, by construction.

ii.
```python
BIN_EDGES = OFF_START + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

```python
time_from_tone = (
    go_times[trial] + BIN_CENTERS - tone_onset[trial]
).astype(np.float32)
```

iii. No separate alignment step is needed: both streams are defined as offsets from the same `go_times[trial]`, and tone onsets come from the same session-absolute clock as the spikes ("Use the NWB go-cue timestamps as the common clock", module docstring).

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`, which are stored as **strings** measured relative to `intervals/trials/start_time`, with the literal `'N/A'` on unstimulated trials. `start_time` and the go cue are used to move them onto the go-cue-relative bin axis.

ii.
```python
trial_start = nwb["intervals/trials/start_time"][:][valid_trial_idx]
stim_onset_raw = nwb["intervals/trials/photostim_onset"][:][valid_trial_idx]
stim_duration_raw = nwb["intervals/trials/photostim_duration"][:][valid_trial_idx]
stim_onset = np.asarray([_optional_float(x) for x in stim_onset_raw])
stim_duration = np.asarray([_optional_float(x) for x in stim_duration_raw])
```

```python
def _optional_float(value) -> float:
    text = _decode(value)
    return np.nan if text in {"", "N/A", "nan"} else float(text)
```

iii. The agent validated this field pairing directly against the independent `photostim_start_times` event stream (step 33), printing for twelve stimulated trials that `start_time + photostim_onset` equals the nearest `photostim_start_times` event to the digit, and that the resulting onset sits a consistent −1.20 s before the go cue (the end of the sample epoch). That confirmed both the reference frame (trial start, not go cue) and the units (seconds).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series, not a per-trial flag: the stimulation interval `[photo_start, photo_stop)` is converted to absolute time, and every bin whose **centre** falls inside it is set to 1.0, all others 0.0. Trials with `'N/A'` onset become NaN and are skipped by the `np.isfinite` guard, so they stay all-zero. Stored as row 1 of the `(2, 80)` input array, float32.

ii.
```python
photo_on = np.zeros(N_BINS, dtype=np.float32)
if np.isfinite(stim_onset[trial]) and np.isfinite(stim_duration[trial]):
    photo_start = trial_start[trial] + stim_onset[trial]
    photo_stop = photo_start + stim_duration[trial]
    absolute_centers = go_times[trial] + BIN_CENTERS
    photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
inputs.append(np.stack((time_from_tone, photo_on), axis=0))
```

iii. The Decoder Task asks for "Whether photostimulation is on at every time point (discrete, time-varying)", so the AI represents it as a per-bin indicator rather than a trial-level flag. Photostimulation trials are deliberately retained even though the repository's `get_regular_trial_mask` drops them, because photostim is a required decoder input (module docstring: "Keep early-lick, no-response, and photostimulation trials. The papers omit these for many analyses, but they are required classes/inputs in this task."). The realised range in the converted file is [0, 1].

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The comparison is done in absolute session time: the bin centres are mapped to absolute time (`go_times[trial] + BIN_CENTERS`) and tested against the absolute stimulation window derived from `trial_start`. Since the bin centres are the same grid the spikes are binned on, the photostim row is aligned to the neural data bin-for-bin.

ii.
```python
photo_start = trial_start[trial] + stim_onset[trial]
photo_stop = photo_start + stim_duration[trial]
absolute_centers = go_times[trial] + BIN_CENTERS
photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
```

iii. Everything in the file shares one clock, so converting the trial-start-relative onset into absolute time is the only correction required — and the agent verified that conversion against the `photostim_start_times` acquisition stream (step 33) before relying on it.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB. Choice is derived from two trials-table columns: `intervals/trials/trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `intervals/trials/outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side; a miss means it licked the other side; `ignore` means it did not lick.

ii.
```python
instruction = _decode_array(
    nwb["intervals/trials/trial_instruction"]
)[valid_trial_idx]
outcome_text = _decode_array(
    nwb["intervals/trials/outcome"]
)[valid_trial_idx]
```

```python
def _trial_choice(instruction: str, outcome: str) -> int:
    # Output order is left, right, no lick.
    if outcome == "ignore":
        return 2
    if outcome == "hit":
        return 0 if instruction == "left" else 1
    if outcome == "miss":
        return 1 if instruction == "left" else 0
    raise ValueError(f"Unexpected outcome {outcome!r}")
```

iii. The AI did not simply assume the labels were right — it cross-checked them against the raw lick event streams. In step 31 it tabulated, for every trial in the dataset, the joint distribution of (outcome, early_lick, first-lick side within the go window, whether any lick occurred); in step 32 it repeated this over the `[go, go+1.5]` response window and counted disagreements: "mismatch total 334 sessions 90", i.e. ~0.35% of ~95k trials, concentrated in two files. It then chose the trial labels over the event stream, recording the reason in the module docstring: "Define choice from the authoritative instruction/outcome trial labels: a hit has the instructed choice, a miss the opposite choice, and an ignored trial has no lick. This avoids rare missing/mis-timestamped lick events in the NWB." The check also confirmed that every `ignore` trial genuinely has no lick in the response window, so a third "no lick" class is warranted rather than an imputed side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice is encoded `0 = left`, `1 = right`, `2 = no lick`, and written into row 0 of the per-trial `(4, 80)` uint8 output array, repeated across all 80 bins so that all four outputs share one time-varying array. `output_values[0] = ['left', 'right', 'no lick']` names the codes. An unexpected outcome string raises rather than silently mapping to a default.

ii.
```python
trial_output = np.empty((4, N_BINS), dtype=np.uint8)
trial_output[0, :] = _trial_choice(
    instruction[trial], outcome_text[trial]
)
```

```python
"output_names": [
    "lick direction choice",
    "outcome",
    "early lick",
    "tongue y-position",
],
"output_values": [
    ["left", "right", "no lick"],
    ...
],
```

iii. `left`/`right`/`no lick` and their order follow the Decoder Output specification. Choice is a single value per trial, so it is broadcast across bins to satisfy the format's preference for time-varying outputs while keeping all outputs in one `(n_output, n_timepoints)` array. The realised class balance is 42.9% left / 42.2% right / 14.9% no-lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which already stores exactly the three requested strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome_text = _decode_array(
    nwb["intervals/trials/outcome"]
)[valid_trial_idx]
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
```

iii. No derivation is needed — the trials table stores the outcome explicitly with the same three categories the instructions ask for. The agent's step-31/32 audit confirmed the category set is closed (only `hit`/`miss`/`ignore` appear anywhere in the dataset).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a fixed dictionary to `0 = ignore`, `1 = miss`, `2 = hit`, and written into row 1 of the output array, repeated across all 80 bins. A `KeyError` would surface any unexpected value.

ii.
```python
trial_output[1, :] = outcome_map[outcome_text[trial]]
```

```python
"output_values": [
    ["left", "right", "no lick"],
    ["ignore", "miss", "hit"],
    ...
],
```

iii. The code order matches the Decoder Output specification ("Outcome (ignore, miss, hit, per-trial)"). One value per trial, broadcast across bins like the other per-trial outputs. Realised balance: 14.9% ignore / 16.6% miss / 68.5% hit.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `intervals/trials/early_lick`, a string column holding `'early'` / `'no early'`.

ii.
```python
early_text = _decode_array(
    nwb["intervals/trials/early_lick"]
)[valid_trial_idx]
```

iii. The flag is stored explicitly in the trials table, so no derivation from lick events is needed; the agent's step-31 audit included `early_lick` in its joint tabulation and confirmed the column is two-valued across the whole dataset. Early-lick trials, which the source papers exclude from most analyses, are deliberately retained here because "early lick" is a required decoder output (module docstring).

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0 = no`, `1 = yes` by a direct string comparison, and written into row 2 of the output array, repeated across all 80 bins.

ii.
```python
trial_output[2, :] = 1 if early_text[trial] == "early" else 0
```

```python
"output_values": [
    ...
    ["no", "yes"],
    ...
],
```

iii. The coding follows the Decoder Output specification ("Early lick (no, yes, per-trial)"). The early lick itself occurs during the sample or delay epoch, i.e. inside the −2.5 s pre-go window, so the neural data covering the event is present even though the label is per-trial. Realised balance: 88.4% no / 11.6% yes.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: its `data` is an `(n_frames, 3)` DeepLabCut array of `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` on the session-absolute clock. Column 1 (`tongue_y`) is the value; column 2 (`tongue_likelihood`) decides visibility. Column 0 is not used.

ii.
```python
tongue_path = (
    "acquisition/BehavioralTimeSeries/"
    "Camera0_side_TongueTracking"
)
tongue_data = nwb[f"{tongue_path}/data"][:]
camera_times = nwb[f"{tongue_path}/timestamps"][:]
tongue_cat, tongue_q40, tongue_q60 = _tongue_categories(
    tongue_data, camera_times, go_times
)
del tongue_data, camera_times
```

```python
y = tongue_data[:, 1]
likelihood = tongue_data[:, 2]
```

iii. This is the only tongue measurement in the release, and the agent verified it is present in all 174 sessions (step 20: `missing tongue 0`). The channel order matches the reference repository's own marker naming, which the agent read in step 8/9 (`['fs', 'Nframes', 'trialNum', 'nose_x', 'nose_y', 'nose_likelihood', 'tongue_x', 'tongue_y', 'tongue_likelihood', ...]`). It confirmed column 2 is a likelihood by checking its distribution across five sessions (step 30): strongly bimodal, with ~89% of frames below 1e-4 and ~10.5% at ≥0.99.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:
1. A frame counts as showing a visible tongue only if `likelihood >= 0.9` and both `y` and `likelihood` are finite.
2. The session's class edges are the 40th and 60th percentiles of `y` taken over **all visible frames in that session** (frame-level, not bin-averaged).
3. Each neural bin is given the value of the **single camera frame immediately preceding its centre**, provided that frame is no more than 20 ms old (nominal frame spacing is 3.4 ms); older gaps count as no data.
4. Bins whose sampled frame is missing or not visible get the fourth class, `3 = not visible`. If a session has no visible frame at all, the cut points are NaN and every bin becomes class 3.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
```

```python
    visible_session = (
        np.isfinite(y)
        & np.isfinite(likelihood)
        & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    )
    if not np.any(visible_session):
        # All requested samples will be category 3. NaN cut points make this
        # exceptional condition explicit in session metadata.
        q40 = q60 = np.nan
    else:
        q40, q60 = np.percentile(y[visible_session], [40, 60])

    target_times = go_times[:, None] + BIN_CENTERS[None, :]
    frame = np.searchsorted(camera_times, target_times, side="right") - 1
    valid_frame = frame >= 0
    frame = np.clip(frame, 0, len(camera_times) - 1)
    # Mark gaps longer than 20 ms as unavailable rather than carrying a stale
    # video sample forward. Nominal frame spacing is 3.4 ms.
    valid_frame &= (target_times - camera_times[frame]) <= 0.020
    sampled_y = y[frame]
    sampled_likelihood = likelihood[frame]
    visible = (
        valid_frame
        & np.isfinite(sampled_y)
        & np.isfinite(sampled_likelihood)
        & (sampled_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    )
```

iii. The sampling rule is copied from the reference repository's marker-alignment code, which the agent read in steps 8–9: `align_markers_between_lims` builds a time grid and, for each grid point `t`, uses "the embedding at the last time point within the time range [t-dt, t)" — i.e. the frame immediately preceding the grid point, not an average. The module docstring records this: "Sample the DLC tongue trace at each neural-bin center, using the preceding camera frame as in the repository's marker-alignment code. DLC likelihood >= 0.9 denotes a visible tongue. Session percentiles use all visible frames." The likelihood threshold is a necessary task-specific addition (the repository ignores the likelihood channels entirely), and the agent measured that the exact value is immaterial: at thresholds 0.5, 0.8, 0.9, 0.95 and 0.99 the visible fraction moves only between 10.59% and 10.45% in the first session, because the likelihood is effectively binary (step 30). The 20 ms staleness guard prevents the inter-trial video gap — the camera is trial-gated — from being papered over with a stale frame. Percentiles are taken over visible frames only because the tracker still emits a coordinate when the tongue is retracted, which would otherwise contaminate the distribution.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly as specified in the instructions, with a fourth class for missing data: `0` if `y < q40`, `1` if `q40 <= y <= q60`, `2` if `y > q60`, `3` if the tongue is not visible (or no camera frame is available). The percentiles are per-session. Stored in row 3 of the output array as uint8.

ii.
```python
    categories = np.full(target_times.shape, 3, dtype=np.uint8)
    if np.isfinite(q40):
        categories[visible & (sampled_y < q40)] = 0
        categories[visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
        categories[visible & (sampled_y > q60)] = 2
    return categories, float(q40), float(q60)
```

```python
"output_values": [
    ...
    [
        "below session 40th percentile",
        "session 40th to 60th percentile",
        "above session 60th percentile",
        "not visible",
    ],
],
```

iii. The three cut points and the per-session scope are dictated verbatim by the Decoder Output specification; a fourth "not visible" class is required because the instructions list `3: not visible` as a category. The per-session cut points are also written into `session_info` (`tongue_y_40th_percentile`, `tongue_y_60th_percentile`) so the discretisation is auditable. Realised class balance: 6.2% / 3.2% / 6.5% / 84.1% — the three visible classes split 39/20/41, as a 40/20/40 percentile split should.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output. Camera timestamps are on the same session-absolute clock as the spikes and the go cues, so each bin's target time is computed as `go + BIN_CENTERS` — the same go-cue-anchored grid used for the firing rates — and a vectorised `searchsorted` over the camera timestamps finds the frame preceding each target time. Bin `k` of the tongue output therefore refers to the same instant as bin `k` of the neural matrix. There is no interpolation and no per-stream offset correction.

ii.
```python
target_times = go_times[:, None] + BIN_CENTERS[None, :]
frame = np.searchsorted(camera_times, target_times, side="right") - 1
valid_frame = frame >= 0
frame = np.clip(frame, 0, len(camera_times) - 1)
valid_frame &= (target_times - camera_times[frame]) <= 0.020
```

iii. "Use the NWB go-cue timestamps as the common clock. Spike timestamps, video timestamps, task events, and trial intervals are all expressed in this clock" (module docstring). Because the video is trial-gated rather than continuous, the 20 ms guard converts the pre-trial gap into explicit `not visible` bins instead of a carried-forward stale value — the AI's comment states exactly this.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled case by case, with a clear split between "exclude it" and "encode it as a category", plus fail-fast assertions for anything unexpected:

- **Session never quality-controlled** (`classification`/`anno_name` are NaN): `_decode` maps any NaN to `''`, so no unit is `'good'` and the session returns `None` and is recorded in `metadata['skipped_sessions']`. One session is dropped this way.
- **Trials outside ephys coverage**: excluded via the `is_good_trials` intersection, with the compact mask mapped back onto the trials table through `obs_intervals` when the tables differ in length (8 sessions).
- **Trials with no spikes at all** (`free_water`, plus one residual): excluded as missing acquisition rather than treated as biological silence.
- **Unstimulated trials** (`photostim_onset == 'N/A'`): converted to NaN by `_optional_float` and skipped by an `np.isfinite` guard, yielding an all-zero photostim row.
- **Invisible tongue / camera gaps**: encoded as the explicit `not visible` class rather than imputed; a session with no visible frames at all yields NaN cut points recorded in metadata.
- **Anything unexpected** raises: go-cue/trial count mismatch, a unit mask wider than the trials table, a good unit with no Allen annotation, a good unit with no observation intervals, an `ephys_source_idx` mapping whose size disagrees with the mask, a go cue with no preceding tone, an unknown outcome string, or a session left with fewer than 2 trials.

ii.
```python
def _decode(value) -> str:
    """Decode an NWB variable-length string, treating NaN as missing."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return str(value)
```

```python
good_rows = np.flatnonzero(classification == "good")
if good_rows.size == 0:
    return None
```

```python
def _optional_float(value) -> float:
    text = _decode(value)
    return np.nan if text in {"", "N/A", "nan"} else float(text)
```

```python
        if session is None:
            skipped_sessions.append(path.name)
            print("    skipped: no classifier-labeled good units", flush=True)
```

iii. The guiding principle is that data which was never recorded is removed (emitting it would fabricate 4 s of 0 Hz across the whole population), while a measurement that legitimately has no value — a retracted tongue, an unstimulated trial — is represented explicitly rather than imputed. The agent arrived at this by chasing the validator's warnings to their source rather than suppressing them (steps 51, 57, 70–84), and by preferring hard failures over silent defaults everywhere else. The final artifact passes `verify_data_format` with zero errors and zero warnings.

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion of all 174 files took ~104 s wall-clock (22:05:42 → 22:07:26), including writing the 11.6 GB pickle. The dominant costs are, in order:
1. **HDF5 reads** — `units/spike_times` (up to ~11.5 M float64 per session) and the tongue array (`(n_frames, 3)`, up to ~1.1 M frames), both read whole into memory.
2. **The per-unit binning loop** in `_bin_good_units`, which runs one `searchsorted` + one `bincount` per good unit (69,453 units total), each over that unit's full spike train.
3. **`pickle.dump` of the 11.6 GB result** at the end, a single serialisation of ~89k neural arrays.
4. Per-element Python string decoding in `_decode_array`, which runs over **all** 272,227 units' `classification` and `anno_name` strings, not just the good ones.

ii.
```python
all_spikes = nwb["units/spike_times"][:]
spike_ends = nwb["units/spike_times_index"][:]
rates = _bin_good_units(all_spikes, spike_ends, good_rows, go_times)
del all_spikes
```

```python
    with output_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI deliberately optimised the two things it measured. It inspected the HDF5 layout before writing the binning code (step 62 printed shape, chunking and compression of `units/spike_times` and the tongue array), then read the ragged buffer in one call instead of one slice per unit. It also chose a per-unit `bincount` over the full trial set rather than a per-(unit, trial) histogram loop, and freed the two largest temporaries as soon as they were consumed (`del all_spikes`, `del tongue_data, camera_times`). The result is roughly 2.5× faster than the human reference's 247 s + ~40 s pickling for the same 174 files.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
1. **The per-trial Python loop** that builds `inputs` and `outputs` — it constructs two small arrays per trial (~89k iterations) and recomputes `go_times[trial] + BIN_CENTERS` twice per iteration. This is fully vectorisable: `time_from_tone`, `photo_on`, and all four output rows could each be computed as a single `(n_trials, 80)` array and then sliced, exactly as the human reference does.
2. **The per-unit loop** in `_bin_good_units`. Since the spike buffer is already concatenated per unit, the whole session could be done with one global `bincount` by adding `out_unit * n_trials * N_BINS` to the flat index. This is a real (if modest) vectorisation opportunity that the human reference does *not* have, because the reference's `searchsorted`-on-edges approach genuinely requires one sorted array per unit.
3. **The per-element list comprehensions** `_decode_array` and `[_optional_float(x) for x in ...]`, which are Python-level loops over 272k unit labels and over all trials' photostim strings.

ii.
```python
    for trial in range(n_trials):
        time_from_tone = (
            go_times[trial] + BIN_CENTERS - tone_onset[trial]
        ).astype(np.float32)
        photo_on = np.zeros(N_BINS, dtype=np.float32)
        if np.isfinite(stim_onset[trial]) and np.isfinite(stim_duration[trial]):
            photo_start = trial_start[trial] + stim_onset[trial]
            photo_stop = photo_start + stim_duration[trial]
            absolute_centers = go_times[trial] + BIN_CENTERS
            photo_on[(absolute_centers >= photo_start) & (absolute_centers < photo_stop)] = 1.0
        inputs.append(np.stack((time_from_tone, photo_on), axis=0))
```

```python
    for out_unit, unit_row in enumerate(good_rows):
        start = 0 if unit_row == 0 else int(spike_ends[unit_row - 1])
        stop = int(spike_ends[unit_row])
        spikes = all_spikes[start:stop]
```

```python
def _decode_array(dataset) -> np.ndarray:
    return np.asarray([_decode(x) for x in dataset[:]])
```

iii. The AI did not comment on these; it vectorised the two dimensions that dominate the data volume — trials within a unit (one `bincount` per unit covering all trials) and trials × bins for the tongue output (`target_times` is a full `(n_trials, 80)` array in one `searchsorted`) — and left the remaining loops in their more readable per-trial form. Since the trial-level work is a small fraction of the ~104 s runtime, the practical cost is low, and the AI's stated priority was correctness and auditability of curation rather than further speed.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once, and no session is processed twice, but several quantities are computed redundantly inside a session:
- `go_times[trial] + BIN_CENTERS` is computed **twice per trial** — once as `time_from_tone` and again as `absolute_centers` for the photostim comparison.
- `_decode_array` is applied to **all** units' `classification` and `anno_name` even though only the ~25% classified `good` are used (`anno_name` is decoded in full and only then indexed by `good_rows`).
- `np.isfinite(y)` / `np.isfinite(likelihood)` are evaluated once over the whole session (`visible_session`) and again on the sampled subset (`visible`).
- Whole trials-table columns are read with `[:]` and then immediately indexed by `valid_trial_idx`.
- `np.any(rates != 0, ...)` scans the entire rate array once more after it has been built.

ii.
```python
            time_from_tone = (
                go_times[trial] + BIN_CENTERS - tone_onset[trial]
            ).astype(np.float32)
            ...
                absolute_centers = go_times[trial] + BIN_CENTERS
```

```python
        classification = _decode_array(nwb["units/classification"])
        good_rows = np.flatnonzero(classification == "good")
        ...
        annotations = _decode_array(nwb["units/anno_name"])[good_rows]
```

```python
        trial_start = nwb["intervals/trials/start_time"][:][valid_trial_idx]
        stim_onset_raw = nwb["intervals/trials/photostim_onset"][:][valid_trial_idx]
        stim_duration_raw = nwb["intervals/trials/photostim_duration"][:][valid_trial_idx]
```

iii. None of these is a second pass over the expensive data — the spike buffer and the video array are each read and binned once — so the AI treated them as negligible. The tongue percentile edges are per-session, which lets them be computed inside the same single pass rather than requiring a second sweep; and the `is_good_trials` / water / dropout filters are all resolved within the one open-file block. `_decode_array` over all units is the one repeat with a measurable cost, since it is a Python-level loop over 272k strings across the dataset.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, but not nothing:
- **`tongue_x` is read and discarded.** The whole `(n_frames, 3)` DLC array is loaded, but only columns 1 and 2 are used; column 0 is never touched.
- **Tongue percentiles are computed over the entire session**, including inter-trial frames and frames outside every retained trial window, while only the in-window bins are emitted. (This is a deliberate choice — the instructions ask for percentiles "over the session" — but it does process frames that never appear in the output.)
- **Firing rates are computed for trials that are then discarded** by the population-dropout filter, since `population_recorded` is evaluated only after `_bin_good_units` has binned every candidate trial.
- **`bin_centers_relative_to_go_s`** (80 floats) is stored in metadata and is not used by the decoder.
- **`session_info` bookkeeping** (`n_source_trials`, `n_ephys_mask_trials`, `n_common_good_trials`, `n_excluded_water_trials`, `n_population_recording_dropouts`, per-session tongue cut points) is computed and stored purely for auditability.
- `classification`/`anno_name` are decoded for all units, of which ~75% are then dropped.

ii.
```python
tongue_data = nwb[f"{tongue_path}/data"][:]
...
y = tongue_data[:, 1]
likelihood = tongue_data[:, 2]
```

```python
population_recorded = np.any(rates != 0, axis=(1, 2))
n_population_dropouts = int(np.count_nonzero(~population_recorded))
if n_population_dropouts:
    rates = rates[population_recorded]
```

```python
"bin_centers_relative_to_go_s": BIN_CENTERS.tolist(),
```

iii. The AI never states that any of this is wasted; the metadata items are clearly intentional — every curation step it applied writes a counter into `session_info`, so the 94,370 → 89,068 trial reduction can be reconstructed per session without re-running the conversion. That is a deliberate trade of a trivial amount of computation for auditability. The `tongue_x` read is unavoidable in a single contiguous HDF5 read of the `(n, 3)` dataset. Only the ordering of the population-dropout filter (after binning rather than before) does measurable extra work, and it has to be that way because the criterion is defined on the binned rates themselves.
