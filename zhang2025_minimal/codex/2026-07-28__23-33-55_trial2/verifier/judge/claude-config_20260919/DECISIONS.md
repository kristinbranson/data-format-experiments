# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API at all. It reads the release index CSV shipped with the reference repository (`code/code_zhang2025/data/bwm_release.csv`), which has one row per (session, probe), and groups it by `eid` to get the session list and the probe names of each session. The local path of a session is then reconstructed by hand from the ALF convention `one_cache/<lab>/Subjects/<subject>/<date>/<session_number:03d>/alf/`, and every data object (trials table, wheel, camera motion energy, spikes, clusters, channels) is loaded with `pd.read_parquet` / `np.load` on files found by globbing that directory. Because the cache stores several ALF revisions, a helper `latest_revision_file()` collects all matches of a glob pattern (both at the `alf/` root and inside `#revision#` folders) and takes the lexicographically last revision, which for IBL's ISO-date revision names is the most recent one. Sessions are processed strictly sequentially in a single process; no parallelism is used. 459 sessions are listed, 439 are converted.

ii.
```python
BWM_RELEASE_CSV = Path("/app/code/code_zhang2025/data/bwm_release.csv")
ONE_CACHE_DIR = Path("/app/data/one_cache")
...
release_df = pd.read_csv(BWM_RELEASE_CSV)
for eid, group in release_df.groupby("eid", sort=False):
    first = group.iloc[0]
    session_rows.append({"eid": eid, "subject": first["subject"], "lab": first["lab"],
                         "date": first["date"], "session_number": int(first["session_number"]),
                         "probe_names": list(group["probe_name"])})
```
```python
session_path = (ONE_CACHE_DIR / session["lab"] / "Subjects" / session["subject"]
                / session["date"] / f"{session['session_number']:03d}")
alf_path = session_path / "alf"
```
```python
def latest_revision_file(base: Path, patterns):
    ...
    def sort_key(path: Path):
        revisions = [part for part in path.parts if part.startswith("#") and part.endswith("#")]
        revision = revisions[-1] if revisions else ""
        return revision, str(path)
    matches.sort(key=sort_key)
    return matches[-1]
```

iii. From the trajectory: "The conversion needs to be reproducible offline against the cache that's already here, so I'm checking whether local ONE metadata is sufficient or whether I need to resolve session paths directly from the filesystem", and later "The first version will be self-contained and offline against `/app/data/one_cache`, with the release table driving session/probe discovery". The AI also noted "The cache contains revised ALF objects, so I need to be explicit about revision selection instead of blindly reading the first matching file", which motivated `latest_revision_file`. Using `bwm_release.csv` for discovery mirrors the reference repo's `0_data_caching.py`, which reads the same freeze file.

## 1-b. How are the data split into subjects?

i. The subject of a session is taken verbatim from the `subject` column of `bwm_release.csv`; nothing is parsed out of paths. After conversion, `subjects` is the list of unique subject names in order of first appearance among the retained sessions, and `subject_idx` maps each retained session to its index in that list. 135 subjects survive across 439 sessions (the human reference also gets 135 subjects).

ii.
```python
subjects = ordered_unique(session["subject"] for session in kept_sessions)
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_lookup[session["subject"]] for session in kept_sessions],
                          dtype=np.int64),
```

iii. No explicit justification is given in the trajectory; the release table already carries a unique subject identifier per session, so no derivation is needed. The only deliberate choice is `ordered_unique` (first-appearance order) rather than sorted order, which is irrelevant as long as `subject_idx` is consistent.

## 1-c. How are the data split into sessions?

i. A session is one `eid` in the release table, so no splitting is done: `groupby("eid")` produces one record per session and collects that session's probe names. Sessions keep the release-table order. Each session is then processed independently and appended to the output lists in that order.

ii.
```python
for eid, group in release_df.groupby("eid", sort=False):
    ...
    "probe_names": list(group["probe_name"]),
```
```python
for session_idx, session in enumerate(session_rows, start=1):
    ...
```

iii. Implicit: the AI's plan step was to "Implement an offline loader around the local ONE cache using `bwm_release.csv` to map `eid -> session path / probes`", i.e. the session is the natural unit of the release and the probes of a session are merged rather than kept apart.

## 1-d. How are the data split into trials?

i. Trials come from the rows of the session's `_ibl_trials.table.pqt`; the AI iterates over the rows with `iterrows()` and treats each row as one trial, using `stimOn_times` of that row to define the trial window `[stimOn - 0.5 s, stimOn + 1.5 s]`. Nothing is re-segmented.

ii.
```python
def load_trials_table(alf_path: Path):
    trials_path = latest_revision_file(alf_path, "#*/_ibl_trials.table.pqt")
    if trials_path is None:
        raise FileNotFoundError(f"Missing trials table in {alf_path}")
    return pd.read_parquet(trials_path)
```
```python
for trial_idx, trial_row in trials_df.iterrows():
    if not trial_mask[trial_idx]:
        continue
    trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
    trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
```

iii. No separate justification; the trials table is already one row per trial. The window `(-0.5, 1.5)` around `stimOn_times` is justified from the reference repo: "The repo's `0_data_caching.py` is actually the decisive reference: it uses a 2 s `stimOn`-aligned window with 20 ms bins", quoting `params = {'interval_len': 2, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`.

## 1-e. How are trials filtered based on quality controls?

i. Two layers. (1) A trial mask reimplementing `ibllib.load_trials_and_mask(..., max_trial_len=10.0)` exactly as the reference repo's `prepare_data` calls it: all six default `nan_exclude` fields (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`) plus `goCue_times` must be finite; reaction time `firstMovement_times - stimOn_times` must lie in [0.08 s, 2.0 s]; trial length `feedback_times - goCue_times` must be ≤ 10 s; and no-response trials (`choice == 0`) are dropped. (2) Per-trial stream coverage: a trial is skipped if the wheel or the camera trace does not span its window (the coverage rule of the repo's `get_behavior_per_interval`), and, added late in the run, a trial is also skipped if the whole retained population has zero spikes anywhere in the 2 s window. Sessions left with fewer than two usable trials are dropped. 186,953 trials survive (human reference: 188,740).

ii.
```python
def compute_trial_mask(trials_df: pd.DataFrame):
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType", "goCue_times"]
    mask = np.ones(len(trials_df), dtype=bool)
    for column in required:
        mask &= np.isfinite(trials_df[column].to_numpy())

    rt = trials_df["firstMovement_times"].to_numpy() - trials_df["stimOn_times"].to_numpy()
    trial_len = trials_df["feedback_times"].to_numpy() - trials_df["goCue_times"].to_numpy()
    mask &= rt >= MIN_RT_S
    mask &= rt <= MAX_RT_S
    mask &= trial_len <= MAX_TRIAL_LEN_S
    mask &= trials_df["choice"].to_numpy() != 0
    return mask
```
```python
wheel_trial = interpolate_behavior_trial(wheel_times, wheel_speed, trial_start, trial_end)
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
if wheel_trial is None or whisker_trial is None:
    continue
...
if not np.any(neural_trial):
    continue
...
if len(session_neural) < 2:
    dropped_sessions.append((session["eid"], "fewer_than_two_trials"))
    continue
```

iii. The notes state: "This follows the masking logic in `load_trials_and_mask(...)` plus the `prepare_data(...)` call used by the reference code" — and indeed the repo calls `load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)`. The zero-spike trial filter was added after verification: "The full verifier passed, but it surfaced a small number of all-zero neural trials. That's not a format failure, but it is a curation issue, so I'm dropping those trials and rebuilding once more to remove the warnings entirely."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's `pykilosort` folder supply the neural array itself. Three further objects are used only for curation and labelling: `clusters.metrics.pqt` (the `label` column) for the quality cut, and `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` to give each cluster an anatomical acronym (via `BrainRegions.id2acronym`) for `brain_regions` / `brain_region_idx` and for the `void`/`root` exclusion.

ii.
```python
sorter_base = probe_alf_path / "pykilosort"
metrics_path = latest_revision_file(sorter_base, "#*/clusters.metrics.pqt")
spike_times_path = latest_revision_file(sorter_base, "#*/spikes.times.npy")
spike_clusters_path = latest_revision_file(sorter_base, "#*/spikes.clusters.npy")
cluster_channels_path = latest_revision_file(sorter_base, "#*/clusters.channels.npy")
channel_regions_path = latest_revision_file(sorter_base, "#*/channels.brainLocationIds_ccf_2017.npy")
```
```python
cluster_region_ids[valid_channel] = channel_region_ids[cluster_channels[valid_channel]]
cluster_acronyms = region_ids_to_acronyms(cluster_region_ids, brain_regions)
```

iii. From the trajectory: "cluster QC tables don't carry acronyms directly, so I'm testing whether I can recover Allen/IBL region names cleanly from the cached channel location IDs" — i.e. because the AI reads raw ALF files instead of using `SpikeSortingLoader.merge_clusters`, it has to reconstruct the cluster→channel→Allen-id→acronym chain itself.

## 2-b. How is the `neural` data processed?

i. Spikes of all probes of a session are merged into one population (cluster ids of later probes offset by the number of units already taken, spikes re-sorted by time), then counted into 100 bins of 20 ms over the trial window with a single `bincount` on a flattened (unit, bin) index. The counts are **not** converted to a rate; they are stored as raw spike counts cast to `float16`. No smoothing, no z-scoring, no rebinning.

ii.
```python
def merge_session_probes(probe_infos):
    ...
    merged_clusters.append(info["spike_clusters"] + cluster_offset)
    cluster_offset += info["cluster_acronyms"].shape[0]
    ...
    order = np.argsort(spike_times, kind="stable")
```
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_clusters, trial_start, binsize=BIN_SIZE_S, n_bins=N_BINS):
    trial_end = trial_start + binsize * n_bins
    start_idx = np.searchsorted(spike_times, trial_start, side="left")
    end_idx = np.searchsorted(spike_times, trial_end, side="left")
    ...
    bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    flat_idx = clusters[valid] * n_bins + bin_idx[valid]
    counts = np.bincount(flat_idx, minlength=n_clusters * n_bins).reshape(n_clusters, n_bins)
    return counts.astype(np.float16)
```

iii. Notes: "Probes from the same session were merged before decoding, matching the reference code and the paper description that probes within a session are not treated independently" (the repo's `merge_probes` does exactly this offset-and-resort). On the dtype: "Neural arrays are stored as dense `float16` spike-count matrices to keep the full pickle size manageable" — the full pickle is 6.5 GB even so.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two cuts, applied per probe before any spike is kept. (1) `clusters.metrics.label >= 1`, the IBL "well-isolated / good" label. (2) Anatomical: clusters whose Allen acronym (from the channel their peak sits on) is `void` or `root` are dropped — `void` is a site outside the brain and `root` is a site the histology could not assign to a structure. Clusters with an invalid channel index get region id 0, which maps to `void` and is therefore dropped too. Surviving clusters are renumbered contiguously and only their spikes are kept. Mean units per session is 164.9 (human reference: 164.2). Note the acronyms are kept in the **fine Allen** mapping, not the Beryl summary mapping the papers/repo use, so the dataset reports 538 brain regions where the reference reports 264.

ii.
```python
keep_mask = metrics["label"].to_numpy() >= 1.0
...
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
kept_clusters = np.flatnonzero(keep_mask)
if kept_clusters.size == 0:
    return None

remap = np.full(n_clusters, -1, dtype=np.int64)
remap[kept_clusters] = np.arange(kept_clusters.size, dtype=np.int64)
spike_keep = remap[spike_clusters] >= 0
```

iii. The AI noticed the tension between the repo and the paper: "The Nature paper emphasizes well-isolated neurons, but the Zhang code path I just read bins all merged clusters unless region-filtered later", then checked the numbers: "an example probe has 898 sorted clusters but only 76 `label >= 1` units. That strongly supports using the paper's 'well-isolated neurons' curation for the dense target format". The notes add: "A release-wide scan gave mean retained units per probe of about 108, matching the paper's reported scale" and "The dense target format would be impractically large if all MUA/sorted clusters were retained." Dropping `root`/`void` follows the repo's `3_decode_multi_region.py`, which excludes those two labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one session clock, so alignment is a subtraction: each trial's window is `stimOn_times + (-0.5, 1.5)` in absolute session time, spikes inside it are selected with `searchsorted`, and the bin index of each spike is `floor((t - trial_start)/0.02)`. Bin 0 therefore starts exactly 0.5 s before stimulus onset for every trial, and the event falls on the boundary between bins 24 and 25.

ii.
```python
OFF_START_S = -0.5
OFF_END_S = 1.5
...
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
trial_end = float(trial_row["stimOn_times"] + OFF_END_S)
...
bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)
```
```python
"temporal_alignment_event": "stimulus onset (stimOn_times)",
"off_start": OFF_START_S,
"off_end": OFF_END_S,
```

iii. Notes: "Alignment event: `stimOn_times`; Window: `[-0.5 s, +1.5 s]`", justified as "This matches the defaults used in `0_data_caching.py`" (`align_time: 'stimOn_times'`, `time_window: (-.5, 1.5)`), and the task instruction "Temporally align based on stimulus onset".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial, identical for every trial and session; `metadata['time_bin_size'] = 20.0` ms. Spikes are binned once directly at 20 ms — there is no intermediate resolution and no rebinning/downsampling step. The behavioural streams are not binned but sampled once per 20 ms bin by interpolation (see 7-b, 8-b).

ii.
```python
BIN_SIZE_S = 0.02
OFF_START_S = -0.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))   # 100
```
```python
"time_bin_size": BIN_SIZE_S * 1000.0,
"n_time_bins": N_BINS,
```

iii. Taken directly from the reference repo's caching parameters (`'binsize': 0.02`, `'interval_len': 2`), quoted verbatim in the notes; the method paper describes the same 2 s / 20 ms / T = 100 configuration.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Nothing beyond `stimOn_times` and the fixed window constants: since every trial uses the same window and the same bin grid, the input is one constant vector of 100 values reused for every trial, expressed relative to `stimOn_times`.

ii.
```python
time_since_stim = np.linspace(
    OFF_START_S + BIN_SIZE_S,
    OFF_END_S,
    N_BINS,
    dtype=np.float32,
)
```
```python
"input_names": ["time_since_stimulus_onset_s", "trial_number_in_block"],
```

iii. Implicit in the plan: "inputs: time since stimulus onset, trial number in block" on the `stimOn`-aligned 2 s / 20 ms grid taken from `0_data_caching.py`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None, it is a definition. The values are the **right edges** of the 100 bins, `-0.48, -0.46, …, 1.48, 1.50` s, i.e. `np.linspace(start + binsize, end, n_bins)`. This is the same grid the reference repo uses for its behavioural interpolation (`x_interp = np.linspace(interval_begs + binsize, interval_ends, n_bins)`), not bin centres. The vector is recomputed inside the per-trial loop and stacked with the trial-number-in-block row into a `(2, 100)` float32 array.

ii.
```python
time_since_stim = np.linspace(OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS, dtype=np.float32)
input_trial = np.vstack([
    time_since_stim,
    np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32),
]).astype(np.float32)
```

iii. Notes: "`time_since_stimulus_onset_s`: time-varying, one value per 20 ms bin, represented at the right edge of each bin, from `-0.48 s` to `1.50 s`." The choice of right edges follows the repo's interpolation grid, which the AI reproduced deliberately ("with the same interval coverage checks as `get_behavior_per_interval(...)`").

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. One-to-one with the neural bins: element *k* of the input vector is the time at the end of neural bin *k* (bin *k* covers `[-0.5 + 0.02k, -0.5 + 0.02(k+1))` after `stimOn`, and the input value is `-0.5 + 0.02(k+1)`). Both come from the same `stimOn_times` and the same 100-bin grid, so the alignment is exact up to that fixed half-bin labelling convention (the human reference labels the same bins by their centres, 10 ms earlier).

ii.
```python
trial_start = float(trial_row["stimOn_times"] + OFF_START_S)
...
bin_idx = np.floor((times - trial_start) / binsize).astype(np.int64)   # neural grid
```
```python
time_since_stim = np.linspace(OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS, dtype=np.float32)  # input grid
```

iii. No extra alignment is needed because the input is defined on the neural binning grid itself; the AI's stated rule is one value per 20 ms bin at the bin's right edge.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table only. The trials table has no block identifier, so a block is defined as a maximal run of consecutive trials with the same `probabilityLeft` value; a change of value (or a NaN) starts a new block.

ii.
```python
trial_number_in_block = compute_trial_number_in_block(trials_df["probabilityLeft"].to_numpy())
```
```python
same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
```

iii. Not discussed explicitly in the trajectory beyond the plan item "trial number in block"; the derivation from `probabilityLeft` is forced by the data, which carries the block prior but not a block index.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A single sequential pass over the **unfiltered** trial list: a counter is reset to 1 whenever `probabilityLeft` changes (or is NaN) and incremented otherwise, so the value is the animal's true 1-based position in the block even when intervening trials are later dropped by the trial mask. The per-trial scalar is then broadcast across all 100 time bins as the second input row. Observed range in the converted data is 1–99 (human reference, 0-based: 0–98).

ii.
```python
def compute_trial_number_in_block(probability_left):
    values = np.asarray(probability_left, dtype=np.float64)
    numbers = np.zeros(values.shape[0], dtype=np.float32)
    count = 0
    previous = np.nan
    for idx, value in enumerate(values):
        same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
        count = count + 1 if same_block else 1
        numbers[idx] = float(count)
        previous = value
    return numbers
```
```python
np.full(N_BINS, trial_number_in_block[trial_idx], dtype=np.float32),
```

iii. Notes: "`trial_number_in_block`: 1-based count of the trial index within the current `probabilityLeft` block, computed on the original session trial order before masking, repeated across time within each retained trial." Computing before masking is justified by wanting the animal's real position in the block; broadcasting across time is justified by the format requirement that all inputs share `(d_input, T)`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward turn), −1 (rightward turn) or 0 (no response). No-response trials are already removed by the trial mask, so only ±1 reaches the encoder.

ii.
```python
mask &= trials_df["choice"].to_numpy() != 0
...
choice_class = choice_to_class(float(trial_row["choice"]))
```

iii. Follows the instruction "Choice, binary, per-trial, left = 0, right = 1" together with the IBL sign convention, and the repo's `exclude_nochoice=True` default.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding +1 → 0 (left) and −1 → 1 (right), done with `np.isclose` and raising on any other value; the per-trial scalar is then repeated across the 100 bins and stored as `int8`. `output_values[0] = ['left', 'right']`. The resulting balance is 0.508 / 0.492, essentially identical to the human reference (0.507 / 0.493).

ii.
```python
def choice_to_class(value):
    if np.isclose(value, 1.0):
        return 0
    if np.isclose(value, -1.0):
        return 1
    raise ValueError(f"Unexpected choice value: {value}")
```
```python
output_trial = np.vstack([
    np.full(N_BINS, choice_class, dtype=np.int8),
    np.full(N_BINS, prior_class, dtype=np.int8),
    wheel_bins,
    whisker_bins,
])
```

iii. Notes: "`choice`: mapped from IBL `choice`, `1 -> left -> 0`, `-1 -> right -> 1`" and "Static outputs (`choice`, `prior_probability_left`) were repeated across all 100 time bins so that all outputs share the same `(d_output, T)` shape."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the values 0.2, 0.5 (the unbiased block at the start of a session) and 0.8. Trials with a NaN prior are already excluded by the mask.

ii.
```python
prior_class = probability_left_to_class(float(trial_row["probabilityLeft"]))
```

iii. Directly from the instruction "Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2"; the same column also defines the blocks used for input 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the recoding 0.2 → 0, 0.5 → 1, 0.8 → 2 (float comparison with `np.isclose`, raising on anything else); the scalar is broadcast across the 100 bins as `int8`. Unbiased (0.5) blocks are kept. Class fractions 0.419 / 0.140 / 0.441, matching the human reference (0.418 / 0.141 / 0.442).

ii.
```python
def probability_left_to_class(value):
    mapping = {0.2: 0, 0.5: 1, 0.8: 2}
    for key, encoded in mapping.items():
        if np.isclose(value, key):
            return encoded
    raise ValueError(f"Unexpected probabilityLeft value: {value}")
```

iii. Notes: "`prior_probability_left`: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`", i.e. the mapping prescribed by the instructions; raising on unexpected values is a deliberate fail-fast check.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy` (accepted either at the `alf/` root or in a revision folder). Speed is the absolute value of the velocity derived from those two arrays. The AI does not call `SessionLoader.load_wheel`; it reimplements `brainbox.behavior.wheel.interpolate_position` and `velocity_filtered` locally so the script stays file-based.

ii.
```python
timestamps_path = latest_revision_file(alf_path, ["_ibl_wheel.timestamps.npy", "#*/_ibl_wheel.timestamps.npy"])
position_path = latest_revision_file(alf_path, ["_ibl_wheel.position.npy", "#*/_ibl_wheel.position.npy"])
...
position_interp, time_interp = interpolate_position(timestamps, position, freq=WHEEL_FS)
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
speed = np.abs(velocity).astype(np.float32)
```

iii. Notes: "Decoder output used absolute wheel velocity, i.e. wheel speed", with the same derivation as the repo's `load_target_behavior(target='wheel-speed')`, which returns `np.abs(sess_loader.wheel['velocity'])`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps, mirroring `SessionLoader.load_wheel`. (1) The irregularly sampled wheel position is linearly interpolated onto a uniform 1000 Hz grid (`np.arange(t0, t_end, 1/1000)`, dropping a trailing sample that would fall outside the data). (2) Velocity is the sample-to-sample difference of the zero-phase Butterworth-low-passed position times `fs`, with the ibllib defaults of 20 Hz corner frequency and order 8; speed is `|velocity|` in rad/s. (3) For each trial the trace is linearly interpolated onto the 100 bin times of the trial window, with the repo's coverage checks (see 7-d). Discretisation is then applied globally (7-c).

ii.
```python
def interpolate_position(re_ts, re_pos, freq=WHEEL_FS):
    t = np.arange(re_ts[0], re_ts[-1], 1.0 / freq, dtype=np.float64)
    if t.size and t[-1] > re_ts[-1]:
        t = t[:-1]
    yinterp = np.interp(t, re_ts, re_pos)
    return yinterp, t


def velocity_filtered(pos, fs=WHEEL_FS, corner_frequency=WHEEL_FILTER_CORNER_HZ, order=WHEEL_FILTER_ORDER):
    sos = signal.butter(N=order, Wn=corner_frequency / fs * 2.0, btype="lowpass", output="sos")
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
    acc = np.insert(np.diff(vel), 0, 0.0) * fs
    return vel, acc
```

iii. Notes: "Position was linearly interpolated to 1000 Hz. Velocity was computed with the same Butterworth low-pass filtering used by `SessionLoader.load_wheel(...)`: cutoff `20 Hz`, order `8`." The AI had read `brainbox/behavior/wheel.py` and `brainbox/io/one.py:1336` (`def load_wheel(self, fs=1000, corner_frequency=20, order=8, ...)`) to get these defaults: "I'm now drilling into the exact wheel and video preprocessing so the converter reproduces those traces rather than approximating them."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes by **global tertiles**: after all sessions are converted, every retained wheel-speed sample of the whole dataset (439 sessions × ~426 trials × 100 bins) is pooled, the 33.3rd and 66.7th percentiles are computed once, and those two thresholds are applied to every trial of every session with `np.digitize`. The thresholds are recorded in `metadata['dynamic_output_binning']`. This makes the classes exactly equal-sized over the whole dataset (0.333/0.333/0.333) but not within a session; in the 8-session sample the per-session fractions range from 0.22 to 0.50.

ii.
```python
wheel_flat = np.concatenate(wheel_continuous_all)
_, wheel_edges = discretize_three_bins(wheel_flat)
```
```python
def discretize_three_bins(values):
    values = np.asarray(values, dtype=np.float64)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if q2 <= q1:
        eps = np.finfo(np.float32).eps
        q2 = q1 + eps
    bins = np.digitize(values, [q1, q2], right=False).astype(np.int8)
    return bins, (float(q1), float(q2))
```
```python
wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
```

iii. Notes: "`wheel_speed_bin`: global tertiles across all retained wheel-speed time bins", with `"binning_rule": "global tertiles over all retained session/trial/time bins"` written into the metadata. The instructions only say "Wheel speed discretized into 3 bins"; the AI chose the dataset-wide tertile so that the three labels mean the same physical speed everywhere and the pooled classes are balanced. No discussion of the per-session alternative appears in the trajectory.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The speed trace is evaluated once per neural bin, at the same 100 times used for the *Time since stimulus onset* input, i.e. at the right edge of each 20 ms neural bin measured from that trial's `stimOn_times`. Samples are taken from the slice of the session trace strictly inside the window (`searchsorted` right/left, as the repo does), and the trial is rejected if the first or last available sample is more than one bin away from the window edge — the repo's "target data starts too late / ends too early" test. Both streams live on the same session clock, so no further alignment is needed.

ii.
```python
def interpolate_behavior_trial(sample_times, sample_values, interval_start, interval_end, binsize=BIN_SIZE_S, n_bins=N_BINS):
    start_idx = np.searchsorted(sample_times, interval_start, side="right")
    end_idx = np.searchsorted(sample_times, interval_end, side="left")
    trial_times = sample_times[start_idx:end_idx]
    trial_values = sample_values[start_idx:end_idx]

    if trial_values.size == 0:
        return None
    if abs(interval_start - trial_times[0]) > binsize:
        return None
    if abs(interval_end - trial_times[-1]) > binsize:
        return None

    x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
    y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. Notes: "Trial values were resampled onto the 20 ms decoder bins using linear interpolation, with the same interval coverage checks as `get_behavior_per_interval(...)`." The AI copied the repo's grid (`np.linspace(beg + binsize, end, n_bins)`) and its three rejection conditions verbatim.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<view>Camera.ROIMotionEnergy.npy` (the IBL-released motion energy of a square over the whisker pad) with frame times `_ibl_<view>Camera.times.npy`. The left camera is used when both files are available, otherwise the right camera; a session with neither is dropped (14 of the 20 dropped sessions). 429 sessions use the left camera, 10 the right.

ii.
```python
def load_motion_energy(alf_path: Path):
    for view in ("left", "right"):
        me_path = latest_revision_file(alf_path, f"#*/{view}Camera.ROIMotionEnergy.npy")
        if me_path is None:
            continue
        times_path = latest_revision_file(
            alf_path, [f"_ibl_{view}Camera.times.npy", f"#*/_ibl_{view}Camera.times.npy"])
        if times_path is None:
            continue
        ...
        return timestamps, motion_energy, view
    raise FileNotFoundError("Missing left/right whisker motion energy")
```

iii. Notes: "The left camera whisker motion energy was used when available; otherwise the right camera was used", following the repo's `'left-whisker-motion-energy'` / `'right-whisker-motion-energy'` targets. An early full run kept only 156 sessions; the AI traced this to file layout rather than data: "many sessions store camera timestamps at the `alf/` root while the motion-energy arrays live in revision folders", and after allowing the mixed layout 439 sessions were retained.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as is — no filtering, smoothing or normalisation. The only repair is the ibllib timestamp fix: if there are more camera timestamps than motion-energy frames the **leading** extra timestamps are dropped (`timestamps[-n:]`), which is exactly `SessionLoader._check_video_timestamps`; if timestamps are shorter than the data the session is rejected. The trace is then resampled onto the same 100 per-trial bin times as the wheel, with the same coverage rejection, and discretised globally (8-c).

ii.
```python
if timestamps.shape[0] < motion_energy.shape[0]:
    raise ValueError(f"{view} camera timestamps shorter than motion energy")
if timestamps.shape[0] > motion_energy.shape[0]:
    timestamps = timestamps[-motion_energy.shape[0]:]
```
```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```

iii. Notes: "If timestamps were longer than the motion-energy trace, the extra timestamps at the start were trimmed, matching `_check_video_timestamps(...)`. Values were linearly interpolated onto the 20 ms decoder bins." The ibllib comment the AI relied on explains that for pre-GPIO sessions the first few frames are sometimes not recorded.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Exactly like the wheel: the whisker traces of every retained trial of every session are pooled, one pair of global tertile thresholds is computed, and those two thresholds are applied to all sessions. Globally the three classes are equal (0.333 each), but because motion energy is in uncalibrated, camera- and session-dependent units the per-session distributions are extremely skewed: in the 8-session sample the class fractions are e.g. [0.996, 0.004, 0.000] and [0.961, 0.037, 0.003] for some sessions and [0.275, 0.192, 0.534] for others. Full-dataset validation balanced accuracy for this output is 0.739 versus 0.585 for the human reference, consistent with part of the label being predictable from session identity alone.

ii.
```python
whisker_flat = np.concatenate(whisker_continuous_all)
_, whisker_edges = discretize_three_bins(whisker_flat)
...
whisker_bins = np.digitize(whisker_trial, [whisker_edges[0], whisker_edges[1]], right=False).astype(np.int8)
```
```python
"dynamic_output_binning": {
    "wheel_speed_edges": list(wheel_edges),
    "whisker_motion_energy_edges": list(whisker_edges),
    "binning_rule": "global tertiles over all retained session/trial/time bins",
},
```

iii. Notes: "`whisker_motion_energy_bin`: global tertiles across all retained whisker-motion-energy time bins" and, in the sanity-check section, "`wheel_speed_bin` and `whisker_motion_energy_bin` are ternary and derived from global tertiles." The only stated rationale is the dataset-wide balance of the three classes; the trajectory contains no discussion of whether motion-energy units are comparable across sessions or cameras.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identically to the wheel: `np.interp` of the camera trace at the 100 bin times of the trial (`stimOn + linspace(-0.48, 1.5, 100)`), using only the frames inside the window, and rejecting the trial if the camera coverage of either window edge is worse than one bin. Camera frame times are on the same session clock as the spikes, so subtraction of `stimOn_times` is the only alignment needed.

ii.
```python
whisker_trial = interpolate_behavior_trial(whisker_times, whisker_me, trial_start, trial_end)
```
```python
x_interp = np.linspace(interval_start + binsize, interval_end, n_bins, dtype=np.float64)
y_interp = np.interp(x_interp, trial_times, trial_values).astype(np.float32)
```

iii. Same justification as 7-d: the repo's `get_behavior_per_interval` grid and coverage checks are reproduced so that the behavioural samples sit on the neural bin grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or inconsistent data is always handled by dropping, at four granularities, and every drop is counted and reported. (a) Whole session: the per-session body is wrapped in a `try/except Exception` that records `error:<type>:<msg>` — missing trials table, missing wheel files, missing camera files, wheel length mismatch, camera timestamps shorter than data, or any missing spike-sorting file for **any** probe of the session all drop the entire session (20 of 459 sessions: 14 no camera ME, 5 no wheel, 1 with <2 trials). (b) Session with no unit surviving QC (`merge_session_probes` returns `None`) or fewer than two usable trials is dropped. (c) Trial: NaN task events removed by the mask, wheel/camera window not covered → `None` → skipped, all-zero neural trial → skipped, non-finite interpolation result → skipped. (d) Value level: clusters with an out-of-range channel index get region id 0 (`void`) and are dropped; surplus camera timestamps are trimmed per ibllib; unexpected `choice` / `probabilityLeft` values raise rather than being silently coerced. The final dataset passes `train_decoder.py --verify-only` with no errors and no warnings.

ii.
```python
except Exception as exc:
    dropped_sessions.append((session["eid"], f"error:{type(exc).__name__}:{exc}"))
```
```python
merged = merge_session_probes(probe_infos)
if merged is None:
    dropped_sessions.append((session["eid"], "no_good_units"))
    continue
```
```python
if not np.all(np.isfinite(y_interp)):
    return None
```
```python
"drop_reasons_top20": Counter(reason for _, reason in dropped_sessions).most_common(20),
```

iii. The AI used the drop counts as its main debugging signal — the first full pass kept only 156 sessions and it reacted with "The first full pass exposed a real bug rather than a dataset limitation … I'm debugging the camera file-resolution logic now" — and then cross-checked the final retention against the paper: "439 retained sessions here versus 433 in the paper, which is likely explained by later video data corrections in the newer local release." The zero-spike trial drop was added to clear verifier warnings.

## 10-a. What are the most time-consuming steps of the code?

i. (1) Reading the spike sorting off disk: `spikes.times.npy` and `spikes.clusters.npy` are hundreds of MB per probe and are loaded in full for every probe of all 459 sessions, then concatenated and re-sorted (`np.argsort` over tens of millions of spikes). (2) The per-trial Python loop, which for each of ~187k trials does two `searchsorted`+`np.interp` calls and one dense `bincount` into an `n_clusters × 100` matrix. (3) Serialisation: the result is a 6.5 GB pickle held entirely in RAM and written in one `pickle.dump`. The whole thing runs **single-process and sequentially** over the release (the human reference runs 10 sessions in parallel with a `ProcessPoolExecutor`), which is the largest avoidable cost; the run took on the order of tens of minutes at ~2.5 GB RSS.

ii.
```python
spike_times = np.load(spike_times_path).astype(np.float64)
spike_clusters = np.load(spike_clusters_path).astype(np.int64)
...
order = np.argsort(spike_times, kind="stable")
```
```python
for session_idx, session in enumerate(session_rows, start=1):
    ...
    for trial_idx, trial_row in trials_df.iterrows():
```
```python
with args.output.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI monitored the run rather than optimising it: "The converter is CPU-bound and sitting around 1 GB RSS, which is acceptable", "memory has plateaued under 2.5 GB, so I'm letting it finish rather than rewriting it for streaming output", and later observed the write-out phase separately ("the process is now in the large-file writeout phase"). Parallelism is never considered in the trajectory.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) `compute_trial_number_in_block` is an explicit Python loop over all trials of a session; it is the classic `(p != p.shift()).cumsum()` + `groupby(...).cumcount()` two-liner (which is what the human reference uses). (2) The main `trials_df.iterrows()` loop: `iterrows()` builds a Series per trial, and the spike binning and the two interpolations inside it could be done for all trials at once (one flat `bincount` with a trial offset, one concatenated `np.interp` query vector). (3) The final discretisation loop re-runs `np.digitize` per trial on a 100-value vector for each of ~187k trials, when a single `np.digitize` per session (or per dataset) would do. In addition `time_since_stim` is rebuilt inside the trial loop. None of these dominate the runtime, which is I/O-bound, but (1) and (3) are pure overhead.

ii.
```python
for idx, value in enumerate(values):
    same_block = np.isfinite(value) and np.isfinite(previous) and np.isclose(value, previous)
    count = count + 1 if same_block else 1
```
```python
for trial_idx, trial_row in trials_df.iterrows():
    ...
    time_since_stim = np.linspace(OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS, dtype=np.float32)
```
```python
for static_values, wheel_trial, whisker_trial in zip(...):
    wheel_bins = np.digitize(wheel_trial, [wheel_edges[0], wheel_edges[1]], right=False).astype(np.int8)
    whisker_bins = np.digitize(whisker_trial, [...], right=False).astype(np.int8)
```

iii. Not discussed in the trajectory; the AI's stated performance concerns were memory and wall-clock of the full pass, not loop structure.

## 10-c. What processing does the code repeat multiple times?

i. Several small repetitions. (1) The constant `time_since_stim` vector is recomputed for every trial (~187k times) instead of once at module level. (2) The behavioural traces are carried twice: each trial's interpolated wheel/whisker vector is appended both to the session list and to a global list, so ~150 MB of float32 is duplicated in memory and later re-traversed by `np.concatenate` and `np.quantile`. (3) `np.digitize` is applied per trial after having already been applied to the pooled array inside `discretize_three_bins`. (4) `latest_revision_file` re-globs the session directory once per object (trials, wheel ×2, camera ×2, and five files per probe). (5) The whole conversion was in effect run three times end-to-end during development (once with the camera bug, once clean, once with the zero-spike filter), but that is process rather than code.

ii.
```python
session_wheel.append(wheel_trial)
session_whisker.append(whisker_trial)
wheel_continuous_all.append(wheel_trial)
whisker_continuous_all.append(whisker_trial)
```
```python
time_since_stim = np.linspace(OFF_START_S + BIN_SIZE_S, OFF_END_S, N_BINS, dtype=np.float32)  # inside the trial loop
```

iii. No justification offered; these are incidental. The two-pass structure (collect continuous traces → compute global thresholds → digitize) is however a deliberate consequence of the global-tertile decision in 7-c/8-c: the thresholds cannot be known until every session has been processed, so the traces have to be kept and revisited.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `velocity_filtered` computes the wheel **acceleration** (`np.insert(np.diff(vel), 0, 0) * fs` over a 1000 Hz session-long array) and the caller discards it. (2) `discretize_three_bins` digitizes the full pooled array (~18.7M values, twice) although only the two threshold values are used — the returned `bins` array is thrown away. (3) A second 85 MB `sample_data.pkl` plus a full region/subject remapping (`build_sample_subset`) is produced; it is only a development artefact and no downstream analysis uses it. (4) Per-cluster region acronyms are resolved for every cluster of every probe, including the ones about to be dropped by the QC label. (5) `spikes.times` is upcast to float64 and `spikes.clusters` to int64 on load, doubling the memory of the largest arrays.

ii.
```python
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos)), 0, 0.0) * fs
acc = np.insert(np.diff(vel), 0, 0.0) * fs
return vel, acc
```
```python
velocity, _ = velocity_filtered(position_interp, fs=WHEEL_FS)
```
```python
_, wheel_edges = discretize_three_bins(wheel_flat)
_, whisker_edges = discretize_three_bins(whisker_flat)
```
```python
cluster_acronyms = region_ids_to_acronyms(cluster_region_ids, brain_regions)
keep_mask &= ~np.isin(cluster_acronyms, ["void", "root"])
```

iii. `velocity_filtered` was copied verbatim from `brainbox.behavior.wheel` to guarantee the identical trace ("reproduces those traces rather than approximating them"), which is why the unused acceleration came along. The sample subset was created deliberately as a fast validation path ("I'm testing it on a tiny subset first to catch alignment or file-resolution mistakes before I let it process the full release") and then tightened for readability of the logs.
