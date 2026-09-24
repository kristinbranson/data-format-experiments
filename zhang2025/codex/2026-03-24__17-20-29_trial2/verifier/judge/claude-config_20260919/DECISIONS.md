# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API. It defines the session universe from the frozen brain-wide-map release table shipped with the reference code (`code/code_zhang2025/data/bwm_release.csv`, 699 probe insertions / 459 unique sessions / 139 subjects), grouping the CSV by `eid` to get one `SessionSpec` per session (subject, lab, date, session number, probe names). Every file is then read directly off the local ALF tree at `data/one_cache/<lab>/Subjects/<subject>/<date>/<nnn>/`, with `resolve_latest()` globbing recursively and taking the last sorted match so that the newest `#revision#` folder wins. The streams read per session are: `alf/**/_ibl_trials.table.pqt`, `alf/**/_ibl_wheel.timestamps.npy` + `_ibl_wheel.position.npy`, `alf/**/{left,right}Camera.ROIMotionEnergy.npy` + `_ibl_<view>Camera.times.npy`, and per probe `alf/<probe>/pykilosort/**/{clusters.metrics.pqt, clusters.channels.npy, channels.brainLocationIds_ccf_2017.npy, spikes.times.npy, spikes.clusters.npy}`. The conversion runs in two passes over sessions (pass 1 = trials/wheel/whisker, pass 2 = spikes), parallelised with a `ThreadPoolExecutor` (32 workers on the full run).

ii.
```python
RELEASE_CSV = Path("code/code_zhang2025/data/bwm_release.csv")
DATA_ROOT = Path("data/one_cache")

    @property
    def session_path(self) -> Path:
        return DATA_ROOT / self.lab / "Subjects" / self.subject / self.date / f"{self.session_number:03d}"


def resolve_latest(base: Path, pattern: str) -> Path:
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {pattern} under {base}")
    return matches[-1]


def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    ...
```
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)
```

iii. From CONVERSION_NOTES.md Step 6: *"Direct reuse of `brainbox.io.one.SessionLoader` is not viable in this environment because `brainbox.io.one` imports the unavailable `neuropixel` package, and `ONE.load_object(...)` against the local cache hits `.rest` permission issues. Conversion will therefore use direct ALF file loading plus the bundled `brainbox.behavior.wheel` helpers."* The trajectory confirms this: `import brainbox.io.one` failed with `ModuleNotFoundError: No module named 'neuropixel'` and `pip install neuropixel` found no distribution. The release CSV was chosen over the local `sessions.pqt` snapshots because *"Treat the 459-session / 699-insertion BWM release as authoritative for conversion because it matches both `bwm_release.csv` and the data paper release counts"* (Step 4), the local cache having 461 session folders and `Brainwidemap/sessions.pqt` listing 480.

## 1-b. How are the data split into subjects?

i. The `subject` column of `bwm_release.csv` is carried on each `SessionSpec`; no path parsing is involved. At assembly the subject list is the sorted unique set of subjects over the sessions that survived, and `subject_idx` holds each session's index into that list, appended in the same loop that appends `neural`/`input`/`output`, so the ordering stays consistent. Result: 135 subjects over 438 sessions.

ii.
```python
subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
...
            subject_idx_list.append(subject_to_idx[prepared.spec.subject])
...
        "subjects": subject_names,
        "subject_idx": np.array(subject_idx_list, dtype=np.int64),
```

iii. The release table already carries a unique subject id per session, so nothing has to be derived. CONVERSION_NOTES Step 5 maps *"subject IDs from frozen release metadata → `subjects`, `subject_idx`: unique sorted subject list and per-session index"*.

## 1-c. How are the data split into sessions?

i. A session is the `eid` of the release table. `bwm_release.csv` has one row per probe insertion, so it is grouped by `eid`; the probe names of the group are stored on the spec and later merged into one population. 459 sessions enter, 438 survive filtering.

ii.
```python
    grouped = bwm.groupby("eid", sort=False)
    for eid, df in grouped:
        row = df.iloc[0]
        probe_names = tuple(df["probe_name"].tolist())
        sessions.append(SessionSpec(eid=eid, subject=str(row["subject"]), lab=str(row["lab"]),
                                    date=str(row["date"]), session_number=int(row["session_number"]),
                                    probe_names=probe_names))
```

iii. No decision to make — the release table is organised by session, and grouping insertions by `eid` is the same thing `merge_probes` in the reference code assumes (*"when merging probes we are interested in eids, not pids"*).

## 1-d. How are the data split into trials?

i. The ALF trials table has one row per trial, so the split is given by the data. Everything downstream (mask, choice, prior, block counter, alignment times) indexes that table row-wise.

ii.
```python
    trials = load_trials_table(spec.session_path)
    trial_mask = compute_trial_mask(trials)
    ...
    align_times = trials[ALIGN_EVENT].to_numpy(dtype=np.float64)
```

iii. Not discussed explicitly; the trials table is one row per trial by construction.

## 1-e. How are trials filtered based on quality controls?

i. Two masks, ANDed. (1) `compute_trial_mask()` is a direct re-implementation of the reference `load_trials_and_mask(one, eid, max_trial_len=10.0)` call made in `prepare_data`: non-NaN on `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`; reaction time (`firstMovement_times - stimOn_times`) in [0.08, 2.0] s; trial duration (`feedback_times - goCue_times`) ≤ 10 s; `choice != 0` (no-response excluded). (2) A behavioural-coverage mask, produced as a by-product of `interpolate_behavior_per_trial` for both the wheel and the whisker stream: a trial is dropped if it has no samples in the window, any NaN sample, or if the first/last sample is more than one bin (20 ms) away from the window edge. A session is dropped entirely if fewer than 2 trials survive. 186,261 of the release trials survive across 438 sessions.

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask = (
        ~trials["stimOn_times"].isnull()
        & ~trials["choice"].isnull()
        & ~trials["feedback_times"].isnull()
        & ~trials["probabilityLeft"].isnull()
        & ~trials["firstMovement_times"].isnull()
        & ~trials["feedbackType"].isnull()
        & (rt >= 0.08)
        & (rt <= 2.0)
        & ((trials["feedback_times"] - trials["goCue_times"]) <= 10.0)
        & (trials["choice"] != 0)
    )
    return mask.to_numpy(dtype=bool)
```
```python
        if curr_vals.shape[0] == 0:
            continue
        if np.isnan(curr_vals).any():
            continue
        if abs(t_beg - curr_times[0]) > binsize:
            continue
        if abs(t_end - curr_times[-1]) > binsize:
            continue
```
```python
    keep_mask = trial_mask & wheel_mask & whisker_mask
    if keep_mask.sum() < 2:
        return None
```

iii. CONVERSION_NOTES Step 4: *"Mandatory filters are the shared missing-event and RT filters. Keep `exclude_nochoice=True` and `max_trial_len=10.0` to remain consistent with the provided code path, while noting these are stricter than the minimal text summary."* Step 5 decision 10: *"Drop any trial lacking full valid coverage of neural, wheel, or whisker data on the common window: no padding or fabricated values."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` of every probe of the session. `clusters.metrics.pqt` (`label` column) supplies the quality label, and `clusters.channels.npy` + `channels.brainLocationIds_ccf_2017.npy` supply the anatomy, but the neural matrix itself is built only from the two spike arrays.

ii.
```python
        metrics = pd.read_parquet(metrics_path, columns=["label"])
        good_rows = metrics["label"].to_numpy(copy=False) >= 1
        ...
        spikes_times = np.load(spikes_times_path, mmap_mode="r")
        spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
        ...
        cluster_channels = np.load(clusters_channels_path, mmap_mode="r")[good_rows]
        channel_region_ids = np.load(channels_region_ids_path, mmap_mode="r")
        ...
        cluster_regions = brain_regions.id2acronym(region_ids, mapping="Beryl").astype(str)
```

iii. Step 5 mapping table: *"`spikes.times`, `spikes.clusters`, `clusters.metrics.label` from all probes in a frozen-release session → `neural`"*, with the Beryl mapping taken *"directly from `channels.brainLocationIds_ccf_2017.npy` via `iblatlas.regions.BrainRegions`"* because `SpikeSortingLoader.merge_clusters` was not importable.

## 2-b. How is the `neural` data processed?

i. Spikes of the surviving units are counted into 20 ms bins over the 2 s window around stimulus onset, giving a (n_units, 100) matrix per trial. Units of the two probes of a session are pooled into one population, the second probe's clusters renumbered to continue after the first; the merged spike arrays are then sorted by time so each trial is one contiguous slice. No smoothing is applied, and the counts are **not** divided by the bin width — the stored values are raw spike counts, cast to `float16` to keep the pickle at 6.2 GB.

ii.
```python
        remap = np.full(n_clusters, -1, dtype=np.int32)
        remap[good_rows] = np.arange(int(good_rows.sum()), dtype=np.int32)
        good_spike_times = np.asarray(spikes_times[spike_mask], dtype=np.float64)
        good_spike_clusters = remap[spikes_clusters[spike_mask]] + cluster_offset
        ...
        cluster_offset += cluster_regions.shape[0]
    merged_spike_times = np.concatenate(spike_times_all)
    merged_spike_clusters = np.concatenate(spike_clusters_all)
    order = np.argsort(merged_spike_times)
```
```python
        rel = spike_times[i0:i1] - interval_begs[i]
        bin_idx = np.floor(rel / binsize).astype(np.int64)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        ...
        flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
        counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
        out.append(counts.astype(np.float16))
```

iii. Step 3 notes *"no firing-rate smoothing; data are spike counts per bin"*, and Step 5 specifies *"merge probes by session after filtering to clusters with `label >= 1`, bin spike counts on a common stimulus-onset window `[-0.5, 1.5]` s with 20 ms bins"*, citing `merge_probes` / `bin_spiking_data` / `get_spike_data_per_interval`. Step 6 justifies the dtype: *"Low-precision storage for neural arrays (`float16`) to keep the dense pickle tractable while remaining compatible with the decoder validator."*

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single cut: clusters whose `clusters.metrics.label >= 1` are kept, everything else is dropped before binning. The mask is applied to the spike stream by indexing the per-cluster flag with `spikes.clusters`, with an extra guard for spike cluster ids that fall outside the metrics table. A probe with no good unit is skipped and a session with no good unit is dropped. No anatomical filter is applied — units whose Beryl acronym is `void` (250 neurons, 0.34 % of the total) or `root` are kept. 72,757 units are kept across 438 sessions (166.1 per session).

ii.
```python
        metrics = pd.read_parquet(metrics_path, columns=["label"])
        good_rows = metrics["label"].to_numpy(copy=False) >= 1
        if not np.any(good_rows):
            continue
        ...
        valid_spikes = spikes_clusters < n_clusters
        if np.all(valid_spikes):
            spike_mask = good_rows[spikes_clusters]
        else:
            spike_mask = np.zeros(spikes_clusters.shape[0], dtype=bool)
            spike_mask[valid_spikes] = good_rows[spikes_clusters[valid_spikes]]
```

iii. Step 4 resolves an explicit discrepancy: the reference code *"`load_spiking_data(..., qc=None)` loads all clusters"*, but *"Use `label >= 1` units for the converted decoder dataset. This exactly reproduces the paper's well-isolated-neuron count and keeps the dense per-trial format computationally feasible."* Step 6 records the sanity check: *"Verified that filtering clusters by `label >= 1` across the 699 frozen-release insertions yields exactly 75,708 units, matching the data paper."* `void`/`root` units are never mentioned in the notes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All streams are already on one session clock, so alignment is a subtraction: the window for trial *i* is `[stimOn_times[i] - 0.5, stimOn_times[i] + 1.5]`, the spike array is sliced to that range with `np.searchsorted`, and each spike's bin is `floor((t - window_start) / 0.02)`. Only the trials surviving the mask are aligned and binned.

ii.
```python
ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
```
```python
    interval_begs = align_times + time_window[0]
    interval_ends = align_times + time_window[1]
    ...
    start_idx = np.searchsorted(spike_times, interval_begs, side="left")
    end_idx = np.searchsorted(spike_times, interval_ends, side="left")
    ...
        rel = spike_times[i0:i1] - interval_begs[i]
        bin_idx = np.floor(rel / binsize).astype(np.int64)
```
```python
    kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
```

iii. Step 4: the task instruction *"Temporally align based on stimulus onset"* is taken as binding, and the AI notes it deliberately overrides the method paper's `firstMovement_times` alignment for the dynamic behaviours and its `(-0.6, -0.1)` s / 50 ms window for the prior: *"User explicitly requires 'Temporally align based on stimulus onset.' Therefore, use stimulus-onset alignment for dynamic outputs as a deliberate task override."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins spanning −0.5 to +1.5 s relative to stimulus onset, identical for every trial and session. Spikes are binned once, directly from spike times, so there is no rebinning or resampling of neural data; `metadata['time_bin_size'] = 20.0` (ms) and `off_start`/`off_end` are −0.5/+1.5.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
```
```python
            "time_bin_size": 20.0,
            "temporal_alignment_event": "stimulus onset (stimOn_times)",
            "off_start": TIME_WINDOW[0],
            "off_end": TIME_WINDOW[1],
```

iii. Step 4: *"code uses 20 ms bins and linear interpolation inside each interval … method paper dynamic behavior bins are 20 ms. No substantive discrepancy."* Step 10 reference comparison: *"reference: stimulus-onset alignment with `time_window=(-0.5, 1.5)` and `binsize=0.02` for 100 bins; conversion: … comparison result: matched."*

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` (the alignment event) and the window/bin constants — nothing else. The value is a single 100-element vector reused for every trial of every session, equal to the **right edge** of each bin: `linspace(-0.5 + 0.02, 1.5, 100)` = −0.48 … 1.50 s. This is the same grid the reference `get_behavior_per_interval` uses (`x_interp = linspace(interval_beg + binsize, interval_end, n_bins)`).

ii.
```python
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)
INPUT_NAMES = ["time_since_stimulus_onset_s", "trial_number_in_block"]
```

iii. Step 5: *"common trial grid relative to `stimOn_times` → `input[0]`: 100-length vector of relative times on the same grid as aligned behavior/neural bins; use a common vector for every trial … represent as time-varying continuous input"*, referencing *"`time_window` and `binsize`; behavior interpolation grid from `get_behavior_per_interval`"*.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None — the vector is defined by the window and bin size, computed once at import and broadcast into every trial's input array (cast to float32, stacked with the block counter).

ii.
```python
        inp = np.vstack(
            [
                COMMON_RELATIVE_TIMES,
                np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32),
            ]
        ).astype(np.float32)
```

iii. N/A — no processing is required; the notes only record the resulting range, `[-0.48, 1.50]`, in the Step 7/Step 9 statistics tables.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the same grid the spikes are binned on, carried over from the reference code's convention: neural bin *i* covers `[stimOn − 0.5 + 0.02·i, stimOn − 0.5 + 0.02·(i+1))`, and the input value at column *i* is that bin's right edge, `−0.5 + 0.02·(i+1)`. Column for column the two describe the same 20 ms interval (the label sits at the end rather than the centre of the bin, a 10 ms offset from the human reference's bin-centre labelling). The behavioural outputs are interpolated at exactly the same times, so all four streams share one axis.

ii.
```python
        bin_idx = np.floor(rel / binsize).astype(np.int64)            # neural: left-edge bins from window start
```
```python
        x_interp = np.linspace(t_beg + binsize, t_end, n_bins)        # behaviour + time input: right edges
        y_interp = np.interp(x_interp, curr_times, curr_vals)
```

iii. Step 5 decision 3: *"Use a single stimulus-onset-aligned 20 ms grid for every variable: This satisfies the user's explicit task requirement and simplifies the validator/decoder interface."* Step 10 check 2 verified the reconstructed input row against an independent raw-data script with `np.allclose(...) == True`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table only. It is constant within a block, so a change of value starts a new block; the trials table carries no block identifier.

ii.
```python
    trial_number_in_block = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. Step 5 mapping: *"`probabilityLeft` block structure in raw trial table → `input[1]` = `trial_number_in_block` … reset to 1 whenever `probabilityLeft` changes"*, noting *"unbiased 0.5 block counts as its own block"*.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A running counter over the **unfiltered** trial sequence, reset to 1 whenever `probabilityLeft` changes (NaN handled by mapping to `None` so consecutive NaNs continue a block). It is therefore 1-based (range [1, 99] over the full dataset; the human reference is 0-based, [0, 98]). Counting before the trial mask means a dropped trial still advances the counter, so the number reflects the animal's true position in the block. The scalar is then repeated across all 100 bins of the kept trial. The counter is written with a plain Python loop rather than a vectorised `groupby(...).cumcount()`.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prob_left), dtype=np.int16)
    prev = None
    counter = 0
    for i, val in enumerate(prob_left):
        current = None if pd.isna(val) else float(val)
        if i == 0 or current != prev:
            counter = 1
        else:
            counter += 1
        out[i] = counter
        prev = current
    return out
```
```python
        trial_number_in_block=trial_number_in_block[keep_mask],
```

iii. Step 5 decision 7: *"Compute `trial_number_in_block` on the original trial table before filtering: This preserves the actual behavioural position within a block rather than renumbering after trial exclusion."* Step 10 edge-case review: *"Confirmed `trial_number_in_block` can legitimately reach 99 because biased blocks last 20-100 trials."*

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (left), −1 (right) or 0 (no response). No-response trials have already been removed by the trial mask, and the mapper raises if any value other than ±1 survives.

ii.
```python
        choice_raw=trials.loc[keep_mask, "choice"].to_numpy(),
```
```python
def map_choice(raw_choice: np.ndarray) -> np.ndarray:
    mapped = np.empty(raw_choice.shape[0], dtype=np.int8)
    mapped[raw_choice == 1] = 0
    mapped[raw_choice == -1] = 1
    if not np.all(np.isin(raw_choice, [-1, 1])):
        raise ValueError("Unexpected choice values encountered after filtering.")
    return mapped
```

iii. Step 5: *"raw `choice` in trial table (`-1`, `1`) after filtering → `output[0]`: map left `1 -> 0`, right `-1 -> 1` … verified directly from high-contrast correct trials; no-choice trials are removed before mapping."* Step 12 re-verified the sign convention against the raw table when choice turned out to be the weakest decoded output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the ±1 → {0, 1} recoding. The per-trial scalar is tiled across the 100 bins so that all four outputs share the (4, 100) shape, and stored as `int8`. Resulting distribution: 0.509 left / 0.491 right.

ii.
```python
        out = np.vstack(
            [
                np.full(NBINS, choice[trial_idx], dtype=np.int8),
                np.full(NBINS, prior[trial_idx], dtype=np.int8),
                wheel_bins[trial_idx],
                whisker_bins[trial_idx],
            ]
        ).astype(np.int8)
```

iii. Step 5 decision 5: *"Represent all outputs as 2D arrays of shape `(4, 100)`: static outputs (`choice`, `prior`) are repeated across time so all outputs share a common structure with the dynamic outputs."*

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the values 0.2, 0.5 and 0.8, recoded to 0, 1, 2 as the task specifies.

ii.
```python
        prior_raw=trials.loc[keep_mask, "probabilityLeft"].to_numpy(),
```
```python
def map_prior(raw_prior: np.ndarray) -> np.ndarray:
    out = np.empty(raw_prior.shape[0], dtype=np.int8)
    mapper = {0.2: 0, 0.5: 1, 0.8: 2}
    for i, val in enumerate(raw_prior):
        key = round(float(val), 1)
        if key not in mapper:
            raise ValueError(f"Unexpected probabilityLeft value {val}")
        out[i] = mapper[key]
    return out
```

iii. Step 5: *"raw `probabilityLeft` … map `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`; repeat across all 100 bins … categorical per trial."*

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding (with a `round(val, 1)` guard against float representation and a hard error on unexpected values), then tiling across the 100 bins as `int8`. Resulting distribution 0.419 / 0.141 / 0.441, consistent with sessions spending most trials in biased blocks after the initial 90-trial unbiased block. Note the recoding is a per-trial Python loop rather than a vectorised map.

ii.
```python
    prior = map_prior(prepared.prior_raw)
    ...
                np.full(NBINS, prior[trial_idx], dtype=np.int8),
```

iii. Step 9 consistency table: *"`prior_probability_of_left` distribution … first 90 unbiased, then 0.2 / 0.8 biased blocks … [0.418751, 0.140604, 0.440645] … Yes."*

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, converted to a velocity with the bundled ibllib helpers and then to a speed by taking the absolute value — the same definition the reference uses for its `'wheel-speed'` target.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered  # noqa: E402
...
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    alf_path = session_path / "alf"
    timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
    position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
    if timestamps.shape[0] != position.shape[0]:
        raise ValueError(f"Wheel timestamp/position length mismatch for {session_path}")
    interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
    return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)
```

iii. Step 5: *"wheel trace from `_ibl_wheel.*` via ONE/SessionLoader-equivalent logic → `output[2]`: compute wheel speed as absolute velocity"*, referencing `load_target_behavior('wheel-speed')`. The parameters `freq=1000, corner_frequency=20, order=8` are the defaults of `SessionLoader.load_wheel`, whose source the agent read (trajectory step 152).

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps. (1) The irregularly sampled wheel position is interpolated onto a uniform 1000 Hz grid (`interpolate_position`). (2) It is differentiated into a velocity with a 20 Hz, order-8 Butterworth low pass (`velocity_filtered`), and the speed is its absolute value in rad/s. (3) Per trial, the trace is sliced to the window and linearly interpolated onto the 100 bin times (`np.interp`, i.e. clamping rather than the reference's `fill_value='extrapolate'`), with the coverage/NaN checks of `get_behavior_per_interval`. (4) It is discretized into 3 classes (see 7-c). The interpolation is run over **all** trials of a session, including those the trial mask rejects, and the rejected ones are discarded afterwards.

ii.
```python
    idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
    idxs_end = np.searchsorted(target_times, interval_ends, side="left")
    ...
        curr_times = target_times[idxs_beg[i]:idxs_end[i]]
        curr_vals = target_vals[idxs_beg[i]:idxs_end[i]]
        ...
        x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
        y_interp = np.interp(x_interp, curr_times, curr_vals)
        outputs[i] = y_interp.astype(np.float32)
        mask[i] = True
```
```python
    wheel_interp, wheel_mask = interpolate_behavior_per_trial(wheel_times, wheel_speed, align_times)
    ...
    wheel_cont = np.stack([wheel_interp[i] for i in np.where(keep_mask)[0]], axis=0)
```

iii. Step 5 mapping: *"align/interpolate to the common stimulus-onset grid"* via `load_target_behavior('wheel-speed')` / `get_behavior_per_interval`; Step 6 records that the bundled `brainbox.behavior.wheel` helpers were used because `SessionLoader` itself was unimportable.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes at **global** tertile thresholds: after pass 1, every kept trial's interpolated speed from **every session** is concatenated into one array and its 1/3 and 2/3 quantiles are taken; those two numbers (0.01514 and 0.40301 rad/s on the full run) are then used to `np.digitize` every session. A degenerate fallback picks order statistics if the two quantiles coincide. The human reference instead takes the percentiles of each session's own trace, giving exactly equal class sizes per session. Globally the AI's classes are exactly 1/3 each; per session the largest class ranges up to 0.58 (median 0.41), and no session is degenerate.

ii.
```python
    all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
    all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
    wheel_edges = robust_tertile_edges(all_wheel)
    whisker_edges = robust_tertile_edges(all_whisker)
```
```python
def robust_tertile_edges(values: np.ndarray) -> tuple[float, float]:
    ...
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    if q2 > q1:
        return float(q1), float(q2)
    ...

def digitize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low, high = edges
    if high <= low:
        return np.zeros(values.shape, dtype=np.int8)
    return np.digitize(values, bins=np.array([low, high], dtype=np.float32), right=False).astype(np.int8)
```

iii. Step 5 decision 9: *"Discretize wheel speed and whisker motion energy with global 3-bin thresholds: session-specific thresholds would make class labels inconsistent across sessions; global thresholds keep categories comparable."* The edges are saved to `metadata['dynamic_output_bin_edges']` and the rule to `metadata['dynamic_output_binning'] = 'global tertile bins over all kept timepoints'`.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at exactly the 100 grid times the neural bins define — `align + linspace(-0.5 + 0.02, 1.5, 100)` — measured from the same `stimOn_times`, so column *i* of the output corresponds to neural bin *i* (sampled at that bin's right edge). Trials whose wheel record does not span the window to within one bin are dropped rather than padded.

ii.
```python
    interval_begs = align_times + time_window[0]
    interval_ends = align_times + time_window[1]
    ...
        x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. Step 4: *"No substantive discrepancy. Use 20 ms bins with trial-wise interpolation."* Step 7 plot review: *"wheel speed and whisker motion energy interpolate smoothly onto the common trial grid; discretized wheel / whisker traces visually track the continuous traces."*

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<view>Camera.ROIMotionEnergy.npy` with frame times `_ibl_<view>Camera.times.npy`, taking the **left** camera when the session has both files and the right otherwise. The IBL-released ROI motion energy (a square over the whisker pad) is used as is; sessions with neither side are excluded (20 of 459).

ii.
```python
def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    alf_path = session_path / "alf"
    for view in ("left", "right"):
        me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
        ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
        if not me_candidates or not ts_candidates:
            continue
        motion_energy = np.load(me_candidates[-1])
        timestamps = np.load(ts_candidates[-1])
        timestamps, motion_energy = check_video_timestamps(view, timestamps, motion_energy)
        return timestamps.astype(np.float64), motion_energy.astype(np.float32), view
    raise FileNotFoundError(f"No whisker motion energy stream found for {session_path}")
```

iii. Step 4: *"code prefers left whisker motion energy and falls back to right camera if missing … Prefer left camera when available, else right, matching the provided code and the local coverage pattern"* (434 sessions have left, 421 right, 441 at least one). This mirrors `bin_behaviors`, which tries `'left-whisker-motion-energy'` and falls back to the right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used unfiltered and unnormalised. Camera timestamps longer than the motion-energy array are trimmed from the front — an exact port of ibllib's `SessionLoader._check_video_timestamps`, i.e. the same repair the reference gets for free through `SessionLoader`. The trace is then interpolated onto the same 100 per-trial grid times as the wheel by the same function (same NaN/coverage rejection), and discretized (8-c).

ii.
```python
def check_video_timestamps(view, video_timestamps, video_data):
    if video_timestamps.shape[0] < video_data.shape[0]:
        if video_timestamps.shape[0] == 0:
            raise ValueError(f"Camera times empty for {view}Camera.")
        raise ValueError(f"Camera times are shorter than video data for {view}Camera.")
    if video_timestamps.shape[0] > video_data.shape[0]:
        video_timestamps = video_timestamps[-video_data.shape[0]:]
    return video_timestamps, video_data
```
```python
    whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
```

iii. Step 6: *"Implemented reference-style video timestamp repair where camera timestamps longer than motion-energy arrays are trimmed from the front."* Step 10 edge-case review confirms this matches the raw-data irregularity described in the data-architecture white paper.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly as the wheel: one **global** pair of tertile edges (2.733 and 7.863) computed over the pooled kept timepoints of all 438 sessions, applied with `np.digitize` to every session. Because ROI motion energy is in arbitrary units that depend on camera, ROI placement and lighting, the pooled thresholds are not comparable across sessions: 49 of 438 sessions end up with a class that is essentially empty (<0.1 % of timepoints), 27 sessions put >90 % of their timepoints in a single class and 5 put >99 %, with a median largest-class fraction of 0.554. Globally the three classes are still exactly 1/3 each.

ii.
```python
    whisker_edges = robust_tertile_edges(all_whisker)
    ...
    whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)
```
```python
            "dynamic_output_bin_edges": {
                "wheel_speed_bin": [float(wheel_edges[0]), float(wheel_edges[1])],
                "whisker_motion_energy_bin": [float(whisker_edges[0]), float(whisker_edges[1])],
            },
            "dynamic_output_binning": "global tertile bins over all kept timepoints",
```

iii. Same justification as 7-c (Step 5 decision 9): *"session-specific thresholds would make class labels inconsistent across sessions; global thresholds keep categories comparable."* Step 9/Step 10 only check the **global** distribution (*"dynamic outputs are exactly tertiled by construction"*); the per-session distribution is never examined. Step 12 records whisker as the strongest decoded output (validation balanced accuracy 0.742) and attributes it to *"qualitative expectation is strong decodability"* without considering that session-constant labels are trivially predictable from the session-specific neural projection.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identically to the wheel: the camera trace is linearly interpolated at `align + linspace(-0.5 + 0.02, 1.5, 100)`, on the same session clock as the spikes, so column *i* matches neural bin *i*; trials whose camera record leaves a gap larger than one bin at either window edge are dropped.

ii.
```python
    wheel_interp, wheel_mask = interpolate_behavior_per_trial(wheel_times, wheel_speed, align_times)
    whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)
    keep_mask = trial_mask & wheel_mask & whisker_mask
```

iii. Step 10 sanity check 2 reconstructed a trial's output matrix from the raw wheel/whisker files plus the stored tertile edges and found `np.allclose(...) == True`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything missing is dropped rather than imputed, at the smallest granularity that works:
- **Session level**: each session is processed inside `build_session_behavior_safe`, which converts a missing required stream into `missing_required_stream` and any other exception into a recorded string; the session is excluded and the reason is stored in `metadata['excluded_sessions']`. 21 of 459 sessions were excluded (20 without whisker motion energy, 1 with no good units or <2 usable trials).
- **Probe level**: a probe with no `label >= 1` cluster is skipped; `spikes.clusters` ids outside the metrics table are masked out instead of raising.
- **Trial level**: NaN trial events, out-of-range reaction times, no-response trials, NaN behavioural samples and trials whose wheel/camera record does not span the window are all dropped; sessions with fewer than 2 remaining trials are dropped.
- **Stream repairs**: camera timestamps longer than the motion-energy array are trimmed from the front; a wheel timestamp/position length mismatch raises (and so excludes the session).
- **Zero-spike trials** are deliberately kept: the 16 "all neural data is zero" verifier warnings were each traced back to the raw spike times and confirmed to be genuinely empty windows.
A gap: `map_prior`/`map_choice` raise on unexpected values in pass 2, which is not wrapped in a try/except, so an unexpected `probabilityLeft` would abort the whole run rather than skip the session (this never triggered).

ii.
```python
def build_session_behavior_safe(spec):
    try:
        prepared = build_session_behavior(spec)
        if prepared is None:
            return spec.eid, None, "no_good_units_or_too_few_valid_trials"
        return spec.eid, prepared, None
    except FileNotFoundError:
        return spec.eid, None, "missing_required_stream"
    except Exception as exc:  # pragma: no cover - defensive for raw-data irregularities
        return spec.eid, None, f"{type(exc).__name__}: {exc}"
```
```python
    if cluster_offset == 0:
        raise ValueError(f"No good units remained for {spec.eid}")
```

iii. Step 10 check 1: *"all 16 have exactly 0 spikes in `[-0.5, 1.5]` s … Resolution: preserve these trials. Dropping them would invent an extra curation rule not present in the reference processing."* Step 9: *"The subset mismatch versus the 459-session / 139-mouse release is required by the decoder task because whisker motion energy is a mandatory output and the local cache lacks that stream for 20 release sessions."*

## 10-a. What are the most time-consuming steps of the code?

i. Measured on the full run: pass 1 (trials table + 1 kHz wheel interpolation/filtering + whisker load + per-trial behaviour interpolation + cluster-metrics read) took 144 s; pass 2 (reading `spikes.times`/`spikes.clusters` for 699 insertions and binning them) took 456 s; the 6.2 GB pickle write adds further time, for ~10 min total with 32 threads. The AI identified spike-stream I/O and dtype copying as the dominant cost and optimised it (memory-mapped reads, no-copy dtype handling, dropping the stable sort, capping BLAS threads, 32 workers), cutting a heavy session's payload build from ~28.8 s to ~12.4 s. Parallelism uses threads, not processes, so the pure-Python parts of the work (path globbing, the per-trial loops, the block counter) contend for the GIL.

ii.
```python
        spikes_times = np.load(spikes_times_path, mmap_mode="r")
        spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
```
```python
    print(f"[build] completed in {time.time() - t0:.2f}s")
```

iii. Step 6: *"Initial full-run profiling showed the real bottleneck was spike-stream reload and dtype copying in pass 2, not the trial-binning code."* Step 10: *"Initial full-run implementation was too slow because pass 2 copied large spike arrays and undershot available CPU parallelism. Resolution: switched to no-copy / memory-mapped spike loading … final full conversion completed in about 10 minutes."*

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four. (1) `bin_spikes_by_trial` loops over trials; all trials could be binned in one `np.bincount` by offsetting each spike's flat index by its trial. (2) `interpolate_behavior_per_trial` loops over trials; the 100 query times of every trial could be built as one vector and passed to a single `np.interp`. These two mirror the per-trial loops the human reference also left un-vectorised, and cost ~0.1 s per session. (3) `compute_trial_number_in_block` is a pure-Python per-trial loop where the reference uses `(prob != prob.shift()).cumsum()` + `groupby(...).cumcount()`. (4) `map_prior` is a pure-Python per-trial loop where a vectorised `.map()`/`np.searchsorted` would do. (3) and (4) run over ~186 k trials in Python and hold the GIL against the other worker threads, though they are still small next to spike I/O. The per-neuron `brain_region_idx` list comprehension is a fifth, minor case.

ii.
```python
    for i in range(len(align_times)):
        ...
        counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
```
```python
    for i, val in enumerate(prob_left):
        current = None if pd.isna(val) else float(val)
```
```python
    for i, val in enumerate(raw_prior):
        key = round(float(val), 1)
```

iii. Step 6 lists *"Vectorized per-trial spike binning via `np.searchsorted` + flattened `np.bincount`"* as a speed-up — i.e. the inner work was vectorised while the outer per-trial loop was kept. The remaining scalar loops are not discussed.

## 10-c. What processing does the code repeat multiple times?

i. Several small repeats, mostly a consequence of the two-pass design that the global tertile thresholds force:
- `clusters.metrics.pqt` is read for every probe twice — once in pass 1 by `count_good_units` (only to test for zero good units) and again in pass 2 by `load_good_spikes_and_regions`.
- `resolve_latest` re-globs the ALF tree recursively for every file on every pass (`**/` walks the whole session directory each call, 5 times per probe plus 5 per session).
- `BrainRegions()` is constructed inside `build_session_payload`, i.e. once per session (438 times) rather than once per process, each construction re-loading the Allen/Beryl tables.
- `digitize_three_bins` is recomputed for the plotted trial in `make_processing_plot` even though the same values were already computed in `build_session_payload`.
- `prepared.trials` is re-indexed with `keep_mask` in both passes (`choice_raw`/`prior_raw` in pass 1, `kept_align_times` in pass 2).
The human reference has no such repeats — it loads each session once in a single pass.

ii.
```python
def count_good_units(spec: SessionSpec) -> int:
    total = 0
    for probe_name in spec.probe_names:
        probe_dir = spec.session_path / "alf" / probe_name / "pykilosort"
        metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
        metrics = pd.read_parquet(metrics_path, columns=["label"])
        total += int((metrics["label"].to_numpy() >= 1).sum())
    return total
```
```python
def build_session_payload(prepared, wheel_edges, whisker_edges):
    session_start = time.time()
    brain_regions = BrainRegions()
```

iii. Not documented as repeats; Step 6 only records the intent behind the split (*"Behavior interpolation performed only once per session in pass 1; neural binning deferred until after global discretization thresholds are known"*) and the inefficiency it did flag (*"Repeated recursive path discovery across revisioned ALF files would be slow if done inside inner trial loops"*), which it avoided only at the inner-loop level.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Behaviour interpolated for rejected trials**: `interpolate_behavior_per_trial` is run over *all* trials of a session (`align_times` is the full column) and only afterwards is `keep_mask` applied, so the wheel and whisker traces of every trial that the RT/NaN/no-response mask rejects (~20–30 % of trials) are computed and thrown away.
- **Wheel acceleration**: `velocity_filtered` returns acceleration, which is discarded (`velocity, _ = ...`) — unavoidable given the helper's signature, but it is computed.
- **`count_good_units`**: the exact count is computed in pass 1 but only its `== 0` test is used; the units are re-derived in pass 2.
- **Whole trials DataFrames retained**: every `PreparedSession` keeps the full `trials` table (all ~30 columns) alive for the entire run although only 7 columns are used, alongside the continuous wheel/whisker traces of all sessions (needed only for the global quantiles).
- **Redundant metadata**: `metadata['common_timepoints_relative_to_alignment_s']` duplicates `input[0]`, and `excluded_session_notes` is always an empty list.
- **Storage**: neural data are stored as `float16` spike counts with no unit field, 6.2 GB, which the decoder re-casts anyway.
The human reference reports N/A here — it loads and processes only the trials that survive the mask.

ii.
```python
    align_times = trials[ALIGN_EVENT].to_numpy(dtype=np.float64)
    wheel_interp, wheel_mask = interpolate_behavior_per_trial(wheel_times, wheel_speed, align_times)
    whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)

    keep_mask = trial_mask & wheel_mask & whisker_mask
```
```python
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
```
```python
    n_good_units = count_good_units(spec)
    if n_good_units == 0:
        return None
```

iii. Not documented as waste. The behaviour-for-all-trials pattern is inherited from the reference `get_behavior_per_interval`, which likewise computes a per-trial mask over the whole trials table; the AI needed `wheel_mask`/`whisker_mask` for *every* trial to build `keep_mask`, so the only saving available would have been to intersect with `trial_mask` first. Step 6 does note the memory-driven choices: *"Dense float32 storage would be too large for the full release"* → `float16`.
