# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API at all. It reads the brain-wide-map release table shipped with the methods-paper repo (`code/code_zhang2025/data/bwm_release.csv`, 699 probe rows / 459 sessions / 139 subjects), groups the rows by `eid` to get one row per session (carrying `subject`, `date`, `session_number`, `lab` and the tuple of probe names), and then rebuilds the ALF path of each session on disk as `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number:03d>`. Every file is then opened directly with `pandas.read_parquet` / `numpy.load`: the trials table (`alf/**/_ibl_trials.table.pqt`), the wheel (`_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy`), the camera motion energy (`<view>Camera.ROIMotionEnergy.npy`, `_ibl_<view>Camera.times.npy`) and, per probe, `alf/<probe>/pykilosort/<rev>/{spikes.times,spikes.clusters,clusters.channels,channels.brainLocationIds_ccf_2017}.npy` and `clusters.metrics.pqt`. Because a dataset can exist at several revisions (`alf/#2025-03-03#/...`), a helper `pick_latest` globs recursively and takes the lexicographically last match, which for `#YYYY-MM-DD#` revision folders is the most recent revision. Only two ibllib helpers are imported (`interpolate_position`, `velocity_filtered`) plus `iblatlas.regions.BrainRegions`. Sessions are processed one at a time in a single process, in release-table order.

ii.
```python
RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def pick_latest(session_path: Path, pattern: str) -> Path:
    matches = sorted(session_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {session_path}")
    return matches[-1]

def find_session_path(row: pd.Series) -> Path:
    return DATA_ROOT / row["lab"] / "Subjects" / row["subject"] / row["date"] / f"{int(row['session_number']):03d}"

def load_trials_table(session_path: Path) -> pd.DataFrame:
    return pd.read_parquet(pick_latest(session_path, "alf/**/_ibl_trials.table.pqt"))
```
```python
release = pd.read_csv(RELEASE_CSV, index_col=0)
sessions = (
    release.groupby("eid", sort=False)
    .agg({"subject": "first", "date": "first", "session_number": "first",
          "lab": "first", "probe_name": lambda x: tuple(sorted(x))})
    .reset_index()
)
```

iii. From the trajectory (steps 21, 30, 35, 56, 60) the AI first tried the IBL stack, found `brainbox`/`iblatlas` were not on `PYTHONPATH` and that the ONE loaders were built to cache HuggingFace-style datasets for different targets, and then decided a "thinner converter against the cached ALF files" was preferable: "The local cache has the trial tables and motion-energy arrays, so I can bypass the online APIs." It verified the choice at release level (`CONVERSION_NOTES.md`): 459 sessions / 699 probes / 139 subjects from the CSV, matching the data paper, and `label >= 1` reproducing the paper's 75,708 well-isolated units exactly.

## 1-b. How are the data split into subjects?

i. The subject is taken verbatim from the `subject` column of the release table, so no path or filename parsing is needed. After all sessions are converted, `subjects` is built in first-encounter order (not sorted) and `subject_idx` holds, per session, the index into that list. 135 subjects survive over 438 sessions.

ii.
```python
subjects = []
subject_to_idx = {}
for rec in records:
    if rec["subject"] not in subject_to_idx:
        subject_to_idx[rec["subject"]] = len(subjects)
        subjects.append(rec["subject"])
...
"subject_idx": np.array([subject_to_idx[rec["subject"]] for rec in records], dtype=np.int16),
```

iii. No explicit justification is given; the release table already carries a unique subject id per session, so nothing has to be derived. The AI only reports the resulting counts as a sanity check ("139 subjects" in the release, "135" in the converted set).

## 1-c. How are the data split into sessions?

i. A session is the unit of the release table: the CSV has one row per probe insertion, so the AI groups by `eid` (which also collapses the 1-2 probes of a session into a single `probe_name` tuple) and treats each resulting row as one session. Sessions are iterated in release order; each becomes one entry of `neural`/`input`/`output`.

ii.
```python
sessions = (
    release.groupby("eid", sort=False)
    .agg({..., "probe_name": lambda x: tuple(sorted(x))})
    .reset_index()
)
for idx, row in enumerate(sessions.to_dict("records"), start=1):
    session_path = find_session_path(row)
```

iii. No decision to make beyond collapsing the per-probe rows; the AI checked the group-by result against the paper ("699 probes, 459 sessions, 139 subjects").

## 1-d. How are the data split into trials?

i. The trials table `_ibl_trials.table.pqt` already has one row per trial, so the split is given by the data. Each retained row becomes one trial; the trial window is defined as `[stimOn_times - 0.5, stimOn_times + 1.5]` and stored in an `intervals` array shared by the neural and behavioural extraction.

ii.
```python
trials = load_trials_table(session_path)
...
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
```

iii. No justification needed/given; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Four groups of filters. (1) The reference repo's own mask, reproduced by hand: reaction time `firstMovement_times - stimOn_times` in [0.08, 2.0] s; trial length `feedback_times - goCue_times <= 10 s` (the `max_trial_len=10.0` that `prepare_data` passes to `load_trials_and_mask`); `choice != 0`; and none of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType` NaN. Unbiased (`probabilityLeft == 0.5`) blocks are **kept**, matching `exclude_unbiased=False`. (2) Stream coverage: a trial is kept only if both the wheel and the camera have samples spanning the full 2 s window to within one bin, and no NaN inside it. (3) An extra filter not in the reference: trials in which the entire QC-passed population emitted zero spikes in the window are dropped. (4) Sessions with fewer than 2 surviving trials are dropped entirely. 186,245 trials in 438 sessions survive.

ii.
```python
def make_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    trial_len = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    mask &= rt >= 0.08
    mask &= rt <= 2.0
    mask &= trial_len <= 10.0
    mask &= trials["choice"].to_numpy() != 0
    for col in required:
        mask &= ~trials[col].isna().to_numpy()
    return mask
```
```python
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
if final_mask.sum() < 2:
    skip_reasons["too_few_trials_after_stream_alignment"] += 1
    continue
```

iii. `CONVERSION_NOTES.md` §3: "The trial mask matches the reference `load_trials_and_mask(...)` logic used in `ibl_data_utils.py`", listing the six non-NaN events, the 0.08-2.0 s reaction-time range, the 10 s trial-duration cap, `choice != 0` and retention of the unbiased block. The zero-spike filter is justified (trajectory step 85, notes §3) as: "dropping any completely spike-silent trial after QC/alignment so the final dataset validates without avoidable warnings."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` of each probe, restricted to the good clusters. The selection and labelling of those clusters uses `clusters.metrics.pqt` (`cluster_id`, `label`), `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`, but the binned array itself is built only from the two spike arrays.

ii.
```python
metrics = pd.read_parquet(probe_base / "clusters.metrics.pqt", columns=["cluster_id", "label"])
cluster_channels = np.load(probe_base / "clusters.channels.npy")[good_mask].astype(np.int64)
channel_region_ids = np.load(probe_base / "channels.brainLocationIds_ccf_2017.npy")
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
```

iii. Trajectory step 35: the AI inspected one session's ALF contents "to confirm how cluster QC labels, channel-to-region mapping, and wheel/video timestamps are stored" before writing the converter, i.e. it identified these as the minimum set of arrays needed to reproduce the loader output without the loader.

## 2-b. How is the `neural` data processed?

i. Per probe, the good clusters are renumbered to a contiguous 0..n-1 range through a lookup table, with an `offset` so that the second probe's units continue after the first probe's - i.e. the probes of a session are pooled into one population. Spikes belonging to dropped clusters are removed, the two probes' spikes are concatenated and stably sorted by time. For each trial the spikes inside the window are located by `searchsorted`, their time relative to the window start is floored into 20 ms bins, and one `np.bincount` over the flattened (unit, bin) index fills the whole `n_neurons x 100` matrix. **The result is stored as raw spike counts in `float16`, not converted to a firing rate** (the reference divides by the bin width to give Hz in float32). No smoothing is applied.

ii.
```python
cluster_map = np.full(max_cluster_id, -1, dtype=np.int32)
cluster_map[good_cluster_ids] = np.arange(offset, offset + good_cluster_ids.shape[0], dtype=np.int32)
...
offset += good_cluster_ids.shape[0]
spike_times = np.concatenate(all_times)
spike_clusters = np.concatenate(all_clusters)
order = np.argsort(spike_times, kind="stable")
```
```python
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    times = spike_times[lo:hi] - start
    bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
    valid = (bins >= 0) & (bins < NBINS)
    flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]
    counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)
    trials.append(counts.astype(np.float16))
```

iii. `CONVERSION_NOTES.md` §2/§4: "spike counts are binned into 20 ms bins over [-0.5, 1.5] s" and "Merged all probes from the same session into a single pooled population", echoing the data paper ("we did not perform decoding on these probes separately because they are not independent"). No explicit justification is given for keeping counts rather than rates or for the `float16` dtype; the delivered file shows counts (max 5 per 20 ms bin in the first session).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters whose `clusters.metrics.label >= 1.0` are kept - the IBL "stringent quality control" score built from amplitude, noise cut-off and refractory-period violation. Nothing else is filtered: units whose Beryl acronym is `void` (a channel the histology placed outside the brain), `root`, `x` or `y` are all retained deliberately. A probe with no good cluster contributes nothing; a session with no good cluster at all is skipped. The delivered dataset contains 72,757 units in 266 regions, including 250 `void` units (the expert reference excludes `void` and has 72,417 units in 264 regions; `root` = 10,028, `x` = 45 and `y` = 6 are identical in both).

ii.
```python
good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0
good_cluster_ids = cluster_ids[good_mask]
if good_cluster_ids.size == 0:
    continue
...
if len(region_labels) == 0:
    skip_reasons["no_good_units"] += 1
    continue
```

iii. `CONVERSION_NOTES.md` §4-5: "Retained clusters with `clusters.metrics.label >= 1.0`. Sanity check: summing `label >= 1` across all 699 release probes reproduces the paper's `75,708` well-isolated units exactly." On the region labels it states explicitly: "I did **not** apply the paper's region-level filtering step (`>=5` neurons per session-region and grey-matter-only region analyses) because this conversion is a pooled-session decoder dataset rather than a per-region analysis dataset. As a result, metadata still contains labels such as `root`, `void`, `x`, and `y` ... This preserves the release-level good-unit accounting more faithfully for the pooled-neuron setting."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams share one session clock, so alignment is a subtraction. Each trial's window is `[stimOn_times - 0.5, stimOn_times + 1.5]`, and inside `bin_spikes` the spike times have the window start subtracted before being floored into bins, which puts bin 0 at `stimOn_times - 0.5` and bin 25 at stimulus onset. `stim_on`, `intervals` and `spike_times` are all cast to `float32`, which quantises absolute times to roughly 1 ms late in a long session - negligible against a 20 ms bin, but a loss of precision the reference does not incur.

ii.
```python
ALIGN_EVENT = "stimOn_times"
WINDOW = (-0.5, 1.5)
stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
...
start_idx = np.searchsorted(spike_times, starts, side="left")
end_idx = np.searchsorted(spike_times, ends, side="left")
times = spike_times[lo:hi] - start
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` §2: "All trials are aligned to `trials.stimOn_times`. Window: `[-0.5, 1.5]` seconds relative to stimulus onset", which the AI ties to the instruction ("Temporally align based on stimulus onset") and to the methods text ("For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial over the 2 s window, identical for every trial and session; `metadata['time_bin_size']` is 20.0 ms. Spikes are binned once at that resolution - there is no rebinning, resampling or smoothing of the neural data. (The behavioural streams are resampled onto the same 100-bin grid; see 7-b/8-b.)

ii.
```python
WINDOW = (-0.5, 1.5)
BIN_SIZE_S = 0.02
NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))   # 100
...
"time_bin_size": BIN_SIZE_S * 1000.0,
```

iii. `CONVERSION_NOTES.md` §2: "Bin size: 20 ms. Number of bins per trial: 100. This matches the 2 s / 20 ms setup described in the methods text" - i.e. the method paper's "split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `trials.stimOn_times` only, indirectly: the window and bin size are fixed, so the input is a constant 100-element vector identical for every trial and every session. The AI labels each bin by its **end** time rather than its centre, so the vector runs -0.48, -0.46, ..., 1.50 s (the reference uses bin centres, -0.49 ... 1.49).

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)
...
"input_names": ["time_since_stimulus_onset_s", "trial_number_in_block"],
"bin_time_reference": "bin end times relative to stimulus onset",
```

iii. `CONVERSION_NOTES.md` §8: "row 0: time since stimulus onset in seconds - bin end times: `[-0.48, -0.46, ..., 1.50]`". The choice of bin ends over centres is documented but not argued.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None - the vector is constructed analytically from `WINDOW` and `BIN_SIZE_S` and replicated per trial into row 0 of the `(2, 100)` input array, cast to `float32`.

ii.
```python
input_trials = [
    np.vstack([
        time_input,
        np.full(NBINS, trial_number[i], dtype=np.float32),
    ]).astype(np.float32)
    for i in selected_idx
]
```

iii. No justification needed or given; the variable is defined by the conversion, not measured.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the neural binning grid itself, index for index: neural bin `j` covers `[stimOn - 0.5 + 0.02j, stimOn - 0.5 + 0.02(j+1))` and the time input at `j` is `-0.5 + 0.02(j+1)`, i.e. the right edge of that same bin. So the two are aligned to within half a bin (10 ms), with a consistent convention across all trials and sessions.

ii.
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)   # bin ends
bins = np.floor(times / BIN_SIZE_S).astype(np.int64)                                  # same grid
```

iii. Implicit; the metadata field `"bin_time_reference": "bin end times relative to stimulus onset"` is the AI's only comment on the convention.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`. The trials table has no block identifier, so a block boundary is detected as a change of `probabilityLeft` between consecutive rows.

ii.
```python
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
```

iii. Not explicitly argued; `CONVERSION_NOTES.md` §8 states the rule: "row 1: trial number within the current `probabilityLeft` block - resets to `1` whenever `probabilityLeft` changes".

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop over the **full, unfiltered** trials table maintains a counter that starts at 1 and increments while `probabilityLeft` is unchanged (compared with `np.isclose`, so the `float32` cast is harmless), resetting to 1 at every change. Only afterwards is the vector indexed by the surviving trials, so a trial dropped by QC still advances the count and the number reflects the animal's true position in the block. The value is 1-based (range 1-99 in the delivered data; the reference is 0-based, 0-98) and is broadcast constant across the 100 bins.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    trial_num = np.ones(prob_left.shape[0], dtype=np.float32)
    curr = 1.0
    for i in range(1, prob_left.shape[0]):
        if np.isclose(prob_left[i], prob_left[i - 1]):
            curr += 1.0
        else:
            curr = 1.0
        trial_num[i] = curr
    return trial_num
```
```python
trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))
...
np.full(NBINS, trial_number[i], dtype=np.float32)
```

iii. Only the rule is documented (notes §8); the AI does not comment on the 1-based origin or on computing the count before filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, which is +1, -1 or 0. No-response trials (0) are already removed by the trial mask; the remaining values are recoded as **-1 -> 0 ("left") and +1 -> 1 ("right")**, and an exception is raised if any other value appears. The result is broadcast constant across the 100 bins and stored as `int8`. This is the **opposite** polarity to the expert reference (`{1.0: 0, -1.0: 1}`) and to the ibllib convention: `brainbox/behavior/training.py` computes `rightward = trials.choice == -1` and comments "choice == -1 means contrast on right hand side", i.e. `choice == +1` is the leftward report. The delivered class fractions are the mirror image of the reference's (AI 0.4914/0.5086 vs reference 0.5075/0.4925), confirming the flip.

ii.
```python
def map_choice(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(values, -1.0)] = 0
    out[np.isclose(values, 1.0)] = 1
    if np.any(out < 0):
        raise ValueError("Unexpected choice values encountered")
    return out
```
```python
"output_values": [["left", "right"], ...]
```

iii. `CONVERSION_NOTES.md` §9 states the mapping as a fact without evidence: "`choice`: IBL `choice == -1` -> left -> `0`; IBL `choice == +1` -> right -> `1`". Nothing in the trajectory shows the AI checking the IBL sign convention against the data (e.g. against `feedbackType` and `contrastLeft/contrastRight`) or against ibllib.

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the recoding described in 5-a and the dropping of `choice == 0` trials in the trial mask. The scalar is repeated across all 100 bins so that the output array has a uniform `(4, 100)` shape.

ii.
```python
np.full((1, NBINS), choice, dtype=np.int8),
```

iii. The instructions require categorical outputs and allow per-trial values; the AI chose to make every output time-varying-shaped ("repeated across all 100 bins", notes §9) so that all four outputs share one `(4, 100)` array.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes the values 0.2, 0.5 and 0.8, recoded to 0, 1 and 2 exactly as the instructions specify. The values are rounded to one decimal before comparison (defensive against the `float32` cast) and an unexpected value raises.

ii.
```python
def map_probability_left(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values.astype(np.float64), 1)
    out = np.full(values.shape[0], -1, dtype=np.int8)
    out[np.isclose(rounded, 0.2)] = 0
    out[np.isclose(rounded, 0.5)] = 1
    out[np.isclose(rounded, 0.8)] = 2
    if np.any(out < 0):
        raise ValueError("Unexpected probabilityLeft values encountered")
    return out
```

iii. `CONVERSION_NOTES.md` §9 simply restates the mapping required by the instructions (`0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`).

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding. The unbiased 0.5 block at the start of each session is retained (matching `exclude_unbiased=False` in the reference `load_trials_and_mask`), so class 1 is the minority class (14.1% of bins, vs 14.05% in the reference). The value is broadcast constant across the 100 bins as `int8`.

ii.
```python
"exclude_unbiased_block": False,
...
np.full((1, NBINS), prior, dtype=np.int8),
```

iii. `CONVERSION_NOTES.md` §3: "unbiased block retained: `probabilityLeft == 0.5` is kept", justified as matching the reference mask's default.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From the raw wheel encoder arrays `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, turned into a velocity with the ibllib helpers and then made a speed by taking the absolute value. A length mismatch between the two arrays raises and the session is skipped.

ii.
```python
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.timestamps.npy"))
    position = np.load(pick_latest(session_path, "alf/**/_ibl_wheel.position.npy"))
    if timestamps.shape[0] != position.shape[0]:
        raise ValueError("Wheel timestamps/position length mismatch")
    interp_pos, interp_t = interpolate_position(timestamps, position, freq=WHEEL_FS)
    velocity, _ = velocity_filtered(interp_pos, fs=WHEEL_FS, corner_frequency=20, order=8)
    return interp_t.astype(np.float32), np.abs(velocity).astype(np.float32)
```

iii. Trajectory steps 39/50: the AI inspected "the `ibllib` implementation of wheel interpolation/velocity ... so the converter can match the reference processing rather than approximating it", concluding "wheel is linearly interpolated to 1 kHz then low-pass differentiated".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps. (1) `interpolate_position(..., freq=1000)` puts the sparsely sampled encoder position on an even 1 kHz grid and `velocity_filtered(..., fs=1000, corner_frequency=20, order=8)` differentiates it through a 20 Hz Butterworth low-pass - exactly the defaults `SessionLoader.load_wheel` uses, so the trace is the same one the reference gets; the absolute value is the speed in rad/s. (2) The trace is resampled per trial onto the 100-bin grid by `np.interp`. (3) It is discretised into 3 classes (see 7-c). **Step (2) is implemented incorrectly**: the sample times are made relative to the window start (`rel_t = t - beg`, running 0 to 2 s) while the query points are relative to *stimulus onset* (`x_interp`, running -0.48 to 1.50 s). The two coordinate systems differ by 0.5 s, so bins 0-24 fall before the data and are clamped by `np.interp` to the first sample, and bins 25-99 return the trace 0.5 s earlier than the matching neural bin. Both effects are visible in the delivered `converted_data.pkl`: in every session checked, 100% of trials have an identical value in all of the first 25 bins, and the population-average wheel speed rises ~0.5 s after the population firing rate does.

ii.
```python
def interpolate_behavior_into_trials(times, values, intervals):
    starts = intervals[:, 0]        # stimOn - 0.5
    ends = intervals[:, 1]          # stimOn + 1.5
    idx_beg = np.searchsorted(times, starts, side="right")
    idx_end = np.searchsorted(times, ends, side="left")
    x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)   # -0.48 .. 1.50
    for i, (ib, ie, beg, end) in enumerate(zip(idx_beg, idx_end, starts, ends, strict=True)):
        t = times[ib:ie]
        y = values[ib:ie]
        ...
        rel_t = t - beg                      # 0 .. 2.0  -> different origin from x_interp
        interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` §6: "Used the same wheel preprocessing path as the provided IBL code: interpolate wheel position to 1000 Hz; compute filtered velocity with `velocity_filtered(...)` - Butterworth low-pass, corner frequency 20 Hz, order 8; decoder target uses `abs(velocity)` = wheel speed." The resampling step is described only as "computed from continuous aligned wheel speed" (§9); the coordinate-origin mismatch is never mentioned, and the AI's own sanity checks ("confirmed that all retained trials have valid wheel coverage for the full stimulus-aligned window") checked coverage, not the resulting alignment.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 classes by `np.digitize` at the 1/3 and 2/3 quantiles - but the quantiles are computed **globally**, over every time bin of every retained trial of all 438 sessions pooled, not per session. All sessions therefore share one pair of thresholds (0.00756 and 0.17409 rad/s), which are recorded in the metadata. Globally the classes are exactly balanced (0.3333/0.3333/0.3333); per session they are reasonably balanced for the wheel (e.g. 0.39/0.31/0.31 and 0.23/0.37/0.40 in the sample), since rad/s is physically comparable across rigs. The reference instead takes the 33rd/67th percentile of each session's own trace.

ii.
```python
wheel_pool.append(np.concatenate(wheel_cont))
...
wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3]).astype(np.float32)

def digitize_tertiles(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds, right=False).astype(np.int8)
```
```python
"continuous_output_discretization": {
    "method": "global tertiles over all included time bins in the converted dataset", ...}
```

iii. `CONVERSION_NOTES.md` §10: "The paper/code treat wheel and whisker variables as continuous, but the requested decoder format requires categorical outputs. I used a single global discretization over all included time bins ... Reason: global tertiles preserve rank information; avoid session-specific label drift; keep class balance near uniform for decoder training."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The intent is that the wheel trace is sampled at the same 100 per-trial bin positions as the spikes, on the single shared session clock, so that index `j` of the output describes the same instant as index `j` of the neural matrix. In practice the implementation described in 7-b breaks this: the delivered wheel row at bin `j` is the wheel speed at `stimOn - 1.0 + 0.02(j+1)`, i.e. 0.5 s before neural bin `j`, and bins 0-24 are a constant clamp rather than data.

ii.
```python
intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]].astype(np.float32)
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
...
rel_t = t - beg
interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. Notes §2/§9 assert a single stimulus-aligned grid for all streams: "All trials are aligned to `trials.stimOn_times` ... The user request required a single stimulus-aligned dataset containing both static and dynamic outputs. I therefore kept the reference loading/QC machinery but used a single stimulus-aligned 2 s window for all requested variables." No justification is offered for the actual 0.5 s offset, which appears to be unintended.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From the released side-camera motion energy `<view>Camera.ROIMotionEnergy.npy` with frame times `_ibl_<view>Camera.times.npy`. The left camera is used when available and the right camera is the fallback, which retains 6 sessions that a left-only policy would have dropped. If the timestamps are longer than the motion-energy array the *leading* timestamps are trimmed; if they are shorter the stream is treated as unusable and the session is skipped - byte-for-byte the behaviour of `SessionLoader._check_video_timestamps`.

ii.
```python
def _load_camera_stream(session_path: Path, view: str):
    times = np.load(pick_latest(session_path, f"alf/**/*_ibl_{view}Camera.times.npy"))
    values = np.load(pick_latest(session_path, f"alf/**/{view}Camera.ROIMotionEnergy.npy"))
    if times.shape[0] < values.shape[0]:
        raise ValueError(f"{view} camera timestamps shorter than motion-energy array")
    if times.shape[0] > values.shape[0]:
        times = times[-values.shape[0]:]
    return times.astype(np.float32), values.astype(np.float32)

def load_whisker_motion_energy(session_path: Path):
    try:
        times, values = _load_camera_stream(session_path, "left")
        return times, values, "left"
    except Exception:
        times, values = _load_camera_stream(session_path, "right")
        return times, values, "right"
```

iii. `CONVERSION_NOTES.md` §7: "Matched the provided code behavior: prefer left whisker camera motion energy; if left stream is unavailable, fall back to right whisker camera motion energy; if video timestamps are longer than the motion-energy array, trim leading timestamps so lengths match ... This matters materially: 6 included sessions rely on right-camera fallback; a left-only policy would have removed those sessions." This mirrors `bin_behaviors`, which tries `left-whisker-motion-energy` and falls back to `right-whisker-motion-energy`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is - no filtering, smoothing or normalisation. It is resampled onto the per-trial 100-bin grid with the same `interpolate_behavior_into_trials` used for the wheel, and therefore inherits the same defect: a 0.5 s coordinate-origin error plus clamping of the first 25 bins. Trials whose window is not covered, or that contain a NaN in the window or after interpolation, are dropped.

ii.
```python
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
...
        if np.isnan(y).any():
            outputs.append(None); continue
        if np.abs(beg - t[0]) > BIN_SIZE_S or np.abs(end - t[-1]) > BIN_SIZE_S:
            outputs.append(None); continue
        rel_t = t - beg
        interp = np.interp(x_interp, rel_t, y).astype(np.float32)
```

iii. Notes §9: the output row is "computed from continuous aligned whisker motion energy"; no extra processing is claimed, consistent with the method paper treating the released ROI motion energy as the target signal.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: `np.digitize` at the 1/3 and 2/3 quantiles of **all** sessions pooled (thresholds 2.0519 and 6.3315 in arbitrary motion-energy units). Because motion energy is in uncalibrated, camera- and rig-dependent units - and because 6 sessions use the right camera, which is half-resolution at a different frame rate - the pooled thresholds are not comparable across sessions. Globally the three classes are exactly equal, but per session they are severely skewed: in the delivered dataset session 100 is 0.608/0.391/0.001, session 437 is 0.939/0.061/0.000, i.e. whole sessions contain essentially none of the "high" class. The reference's per-session percentiles give every session a 1/3-1/3-1/3 split.

ii.
```python
whisker_pool.append(np.concatenate(whisker_cont))
...
whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3]).astype(np.float32)
whisker_disc = digitize_tertiles(whisker, whisker_thresholds)
```

iii. Same justification as 7-c (notes §10): "global tertiles preserve rank information; avoid session-specific label drift; keep class balance near uniform for decoder training." The AI did not check the per-session consequence of applying one global threshold to a signal in arbitrary units.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. As for the wheel: intended to be the same per-trial 100-bin grid on the shared session clock, but delivered 0.5 s early with the first 25 bins clamped, because `interpolate_behavior_into_trials` mixes window-relative sample times with onset-relative query times. Verified on `converted_data.pkl`: 100% of trials in every session checked have a constant whisker class over bins 0-24, and the average whisker trace rises ~0.5 s after the neural response.

ii.
```python
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
```

iii. Notes §2 ("All trials are aligned to `trials.stimOn_times`"); no justification for the actual offset, which is unintended.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or broken data is skipped rather than patched, at the finest granularity available. Per file: `pick_latest` raises `FileNotFoundError` if no revision of a dataset exists, and the whole session is counted under `missing_required_file` (20 sessions). A catch-all `except Exception` records the exception type in `skip_reasons` and prints the session id, so a single bad session never aborts the run. Per stream: mismatched wheel timestamps/position raise; camera timestamps longer than the motion-energy array are trimmed from the front, shorter ones raise and trigger the right-camera fallback. Per trial: NaNs in any of the six required trial events are excluded by the mask; NaNs in the raw or interpolated behavioural window mark the trial invalid; trials whose wheel/camera coverage does not reach both window edges to within one bin are dropped; trials with no spikes are dropped. Per session: fewer than 2 usable trials, or zero good units, and the session is skipped (1 session). Unexpected `choice` or `probabilityLeft` values raise rather than being silently coerced.

ii.
```python
except FileNotFoundError:
    skip_reasons["missing_required_file"] += 1
except Exception as exc:
    skip_reasons[type(exc).__name__] += 1
    print(f"Skipped session {row['eid']} due to {type(exc).__name__}: {exc}", flush=True)
```
```python
if np.isnan(y).any():
    outputs.append(None)
    continue
if np.abs(beg - t[0]) > BIN_SIZE_S or np.abs(end - t[-1]) > BIN_SIZE_S:
    outputs.append(None)
    continue
```
```python
if trial_mask.sum() < 2:
    skip_reasons["too_few_trials_after_reference_mask"] += 1
    continue
if len(region_labels) == 0:
    skip_reasons["no_good_units"] += 1
    continue
```

iii. `CONVERSION_NOTES.md` documents the exclusions exhaustively (the 21 excluded eids, 20 for missing wheel/whisker files and 1, `f8041c1e-...`, for failing the post-alignment trial count) and reconciles the resulting 438 sessions with the paper's 433: "the provided code explicitly allows right-camera whisker fallback, which retains 6 sessions; the local cache currently has 20 sessions without the required wheel/whisker files and 1 additional alignment-failure session. I kept the code-consistent behavior rather than forcing the paper number."

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion took 1000 s for 459 sessions in a **single process** (the reference uses a 10-worker `ProcessPoolExecutor`). The dominant cost is reading the two large spike arrays of each probe off disk (`spikes.times.npy`, `spikes.clusters.npy`, hundreds of MB per probe), followed by the `np.argsort` that merges the probes' spikes into time order and the `.astype` copies of those arrays. Next are the two per-trial Python loops - `bin_spikes` (one `np.bincount` over an `n_units x 100` grid per trial) and `interpolate_behavior_into_trials` (two `np.interp` calls per trial) - and the per-trial `np.any(trial > 0)` scan that computes `neural_valid`. Finally, `pickle.dump` of the 6.5 GB dictionary and the `np.quantile` over the pooled behavioural traces (hundreds of millions of samples) are one-off costs at the end.

ii.
```python
spikes_times = np.load(probe_base / "spikes.times.npy").astype(np.float32)
spikes_clusters = np.load(probe_base / "spikes.clusters.npy").astype(np.int64)
...
order = np.argsort(spike_times, kind="stable")
```
```python
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
```

iii. No explicit analysis in the notes; the AI only tracked wall-clock progress ("elapsed=1000.0s" in `conversion_full_out.txt`) and observed at step 91 that it would be "a multi-minute pass". It never considered parallelising sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four. (1) `compute_trial_number_in_block` is a pure-Python loop over every trial of the session, doing `np.isclose` on scalars; it is the one clear waste, since the whole thing is `g = (p != shift(p)).cumsum()` plus a `groupby(g).cumcount()` (what the reference uses) or a `np.diff`/`cumsum` construction. (2) `bin_spikes` loops over trials; a single `bincount` with the trial index folded into the flat index would do all trials at once. (3) `interpolate_behavior_into_trials` loops over trials; one concatenated query vector would replace it. (4) `build_output_trials` digitises and `vstack`s per trial, although both traces could be digitised once per session as whole arrays. (2)-(4) are the same per-trial loops the reference keeps for readability and cost little; (1) is gratuitous.

ii.
```python
for i in range(1, prob_left.shape[0]):
    if np.isclose(prob_left[i], prob_left[i - 1]):
        curr += 1.0
    else:
        curr = 1.0
    trial_num[i] = curr
```
```python
for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):
    ...
for i, (ib, ie, beg, end) in enumerate(zip(idx_beg, idx_end, starts, ends, strict=True)):
    ...
for choice, prior, wheel, whisker in zip(record["choice"], record["prior"],
                                         record["wheel_cont"], record["whisker_cont"], strict=True):
```

iii. Not discussed anywhere in the notes or the trajectory.

## 10-c. What processing does the code repeat multiple times?

i. Several small repetitions. `digitize_tertiles` is called once per trial on a 100-sample vector instead of once per session on the stacked array. `np.full((1, NBINS), choice)` / `np.full((1, NBINS), prior)` and the `np.vstack` rebuild the same constant rows for every trial. `time_input` (and `x_interp` inside the interpolation helper) is recomputed for every session and every call although it is a module-level constant. The continuous behavioural traces are materialised twice: once as `wheel_cont`/`whisker_cont` inside each record and again as a concatenated copy pushed onto `wheel_pool`/`whisker_pool`. `pick_latest` re-globs `alf/**` separately for each of the five behavioural datasets. Outside the script, the AI also re-ran the entire 1000 s conversion a second time purely to capture the console transcript into `conversion_full_out.txt` (trajectory steps 153-162).

ii.
```python
wheel_cont = [np.asarray(x, dtype=np.float32) for x in wheel_selected]
whisker_cont = [np.asarray(x, dtype=np.float32) for x in whisker_selected]
wheel_pool.append(np.concatenate(wheel_cont))
whisker_pool.append(np.concatenate(whisker_cont))
```
```python
time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)   # per session
x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)     # per call
```

iii. Not discussed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three things. (1) Spikes are binned and both behavioural traces interpolated for **every** trial in the table, including the ~20-30% that the QC mask has already rejected; only afterwards are the survivors selected with `selected_idx`, so the discarded trials' `bincount`/`interp`/`float16` work is thrown away. The reference applies its mask first and bins only kept trials. (2) The global-tertile decision forces the full-precision continuous wheel and whisker traces for all 186,245 trials to be held in memory for the whole run (in `records` and duplicated in `wheel_pool`/`whisker_pool`), only to be reduced to `int8` classes at the very end and discarded - the reference discretises per session and never keeps them. (3) `neural_valid` scans every binned trial just to detect the handful that are entirely silent, and a large metadata block (per-session paths, probe names, three different trial counts per session, the excluded-session bookkeeping) is written into the pickle although the decoder never reads it. Minor extras: `clusters.channels`/`brainLocationIds` are decoded to Allen and then Beryl acronyms per probe even though the decoder only consumes the region index, and the `right=False` argument to `np.digitize` is the default.

ii.
```python
neural_trials = bin_spikes(spike_times, spike_clusters, intervals, len(region_labels))   # all trials
neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)
wheel_trials, wheel_valid = interpolate_behavior_into_trials(wheel_times, wheel_speed, intervals)
whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)
final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid
selected_idx = np.flatnonzero(final_mask)
neural_selected = [neural_trials[i] for i in selected_idx]
```

iii. Not discussed. The retention of the continuous traces is an indirect consequence of the documented decision to use "a single global discretization over all included time bins" (notes §10), which cannot be applied until every session has been processed.
