# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API at all. It reads the local IBL ONE cache directly off the filesystem. It picks the first session manifest that exists from an ordered list (`2025_Q3_IBL_et_al_BWM/sessions.pqt`, then `Brainwidemap/`, then `2022_Q4_IBL_et_al_BWM/`), which resolves to the 2025_Q3 release table with 459 sessions / 139 subjects. For each manifest row it builds the ALF path `one_cache/<lab>/Subjects/<subject>/<date>/<NNN>/` and keeps the session only if that directory exists (459 candidates were found; 0 missing). Every individual dataset is then located with `rglob` plus a hand-written revision resolver (`version_key` / `pick_one_file`) that sorts candidate paths by `(is_versioned, revision_string)` and takes the last one, i.e. the newest `#YYYY-MM-DD#` revision folder. Files read per session: `alf/**/_ibl_trials.table.pqt`, `alf/**/_ibl_wheel.{position,timestamps}.npy`, `alf/**/{left,right}Camera.ROIMotionEnergy.npy` + `*{left,right}Camera.times.npy`, and per probe `alf/probe*/pykilosort/**/{spikes.times.npy, spikes.clusters.npy, clusters.metrics.pqt, clusters.channels.npy, channels.brainLocationIds_ccf_2017.npy}`. Sessions are processed by a `ThreadPoolExecutor` with up to 8 workers; total runtime 14m 0.6s.

ii.
```python
MANIFEST_FILES = [
    DATA_ROOT / "2025_Q3_IBL_et_al_BWM" / "sessions.pqt",
    DATA_ROOT / "Brainwidemap" / "sessions.pqt",
    DATA_ROOT / "2022_Q4_IBL_et_al_BWM" / "sessions.pqt",
]

def resolve_session_specs() -> tuple[list[SessionSpec], list[str]]:
    manifest = load_session_manifest()
    for eid, row in manifest.iterrows():
        session_path = (DATA_ROOT / row["lab"] / "Subjects" / row["subject"]
                        / str(row["date"]) / f"{int(row['number']):03d}")
        if not session_path.exists():
            missing.append(eid); continue
        specs.append(SessionSpec(eid=eid, lab=..., subject=str(row["subject"]), ...))

def version_key(path: Path) -> tuple[int, str]:
    revision = ""
    for part in path.parts:
        if part.startswith("#") and part.endswith("#"):
            revision = part.strip("#")
    return (1 if revision else 0, revision)

def pick_one_file(base: Path, pattern: str) -> Path | None:
    matches = list(base.rglob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: (version_key(p), str(p)))
    return matches[-1]
```

iii. From CONVERSION_NOTES.md Step 2/Step 6: the cache "uses versioned ALF folders such as `alf/#2025-03-03#/` … so conversion code will need robust path resolution instead of hard-coded filenames", and the script "targets the canonical local `2025_Q3_IBL_et_al_BWM/sessions.pqt` release manifest, which contains 459 sessions from 139 subjects and matches the data paper counts". The trajectory shows it explicitly moved off the smaller reproducible-ephys subset to the full release manifest. Direct file reads were chosen over ONE so that no network/Alyx access is needed and spike arrays can be memory-mapped.

## 1-b. How are the data split into subjects (mice)?

i. The subject name comes straight from the `subject` column of the release manifest row (and is also the directory name in the cache path). It is carried on `SessionSpec.subject` → `ProcessedSession.subject`. At assembly, `subjects` is the sorted unique set over *kept* sessions and `subject_idx` is the index of each session's subject into that list. Result: 135 subjects over 438 kept sessions (139 in the manifest; 4 subjects — `KS045`, `KS052`, `ZM_1897`, `ibl_witten_32` — lost entirely because all of their sessions were dropped for missing wheel/whisker).

ii.
```python
subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16),
```

iii. CONVERSION_NOTES Step 5 mapping table: "Build unique subject list from kept sessions and index each session into it … Deterministic session order required." No derivation needed because the manifest already carries a unique subject id.

## 1-c. How are the data split into sessions?

i. No splitting is performed: one manifest row = one `eid` = one session directory = one `ProcessedSession`. Sessions are the unit of parallelism and of the output lists. The two probes of a two-probe session are *merged* into a single session population rather than treated as separate sessions. 459 candidate sessions → 438 exported.

ii.
```python
specs, missing_eids = resolve_session_specs()
...
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    for session in executor.map(process_session_worker, specs):
        if session is not None:
            processed.append(session)
```

iii. CONVERSION_NOTES Step 4/5: "session-level units of analysis", "merged probes per session", following `prepare_data()` in the reference repo which "merges all probes from an `eid` into one session representation before decoding".

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) already has one row per trial, so the split is given by the data. Each retained row becomes one trial window `[stimOn_times - 0.5 s, stimOn_times + 1.5 s]`. The original row position is preserved through `reset_index(drop=False)` so the block counter (4-b) can be indexed back into the unfiltered table.

ii.
```python
def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    return pd.read_parquet(trial_file)
...
masked_trials = trials.loc[trial_mask].reset_index(drop=False)
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
```

iii. Not discussed as a decision — the notes treat the trials table as authoritative ("Trial table columns: `goCue_times`, `response_times`, `choice`, `stimOn_times`, …", Step 2).

## 1-e. How are trials filtered based on quality controls?

i. Two stages. **Stage 1 (`compute_trial_mask`)** is a direct port of the reference repo's `load_trials_and_mask` defaults: reaction time `firstMovement_times - stimOn_times` in `[0.08, 2.0]` s; trial length `feedback_times - goCue_times <= 10 s`; `choice != 0` (no-response dropped); and non-NaN for the default `nan_exclude` list `['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']`. **Stage 2** drops trials whose 2 s window is not spanned by the wheel *and* by the camera (the `|window_edge - first/last sample| > binsize` test, also ported from the reference `get_behavior_per_interval`), plus trials whose binned neural matrix is entirely zero. A session is skipped if fewer than 2 trials survive either stage. Net: 282,357 raw trials in kept sessions → 186,245 exported (mean 425.2/session).

ii.
```python
def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask &= rt >= TRIAL_MASK_RT[0]          # 0.08
    mask &= rt <= TRIAL_MASK_RT[1]          # 2.0
    mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN   # 10.0
    mask &= trials["choice"] != 0
    for col in required:
        mask &= trials[col].notna().to_numpy()
    return mask
```
```python
        if np.abs(start - ts[0]) > binsize or np.abs(end - ts[-1]) > binsize:
            outputs.append(None); continue
...
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
if combined_mask.sum() < 2:
    print(f"[skip] {spec.eid}: fewer than 2 trials after behavior/neural alignment")
    return None
```

iii. CONVERSION_NOTES Step 4: "Use the full reference-code trial mask, since it contains the paper overlap plus extra curation actually used by the decoder repository." Step 10 Check 3: "`compute_trial_mask()` matches the reference logic in `load_trials_and_mask()` for required events, RT range, no-choice removal, and max trial duration." The all-zero-neural filter was added reactively: "Initial `verification_full_out.txt` had 16 warnings for all-zero neural trials … I traced these to genuinely empty spike windows in the raw data, added an explicit `np.any(trial)` neural-window filter … trial count decreased from 186,261 to 186,245."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's `pykilosort` folder. `clusters.metrics.pqt` (column `label`) supplies the QC filter, and `clusters.channels.npy` + `channels.brainLocationIds_ccf_2017.npy` supply the anatomical acronym for `brain_region_idx`. The neural array itself is built only from the two spike arrays.

ii.
```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
clusters_channels_file = pick_one_file(pykilo_path, "clusters.channels.npy")
channels_ids_file = pick_one_file(pykilo_path, "channels.brainLocationIds_ccf_2017.npy")
...
spikes_times = np.load(spikes_times_file, mmap_mode="r")
spikes_clusters = np.load(spikes_clusters_file, mmap_mode="r")
```

iii. Step 1/Step 5 mapping table: "`spikes.times`, `spikes.clusters` from all probes in one session → `neural`". Memory-mapping was chosen as a speed-up ("Spike arrays are memory-mapped and only QC-passing spikes are materialized").

## 2-b. How is the `neural` data processed?

i. Per probe: keep clusters with `label >= 1`, renumber survivors contiguously, keep only their spikes. Probes of a session are concatenated with a cluster-index offset so probe 1's units continue after probe 0's, and the merged spike train is re-sorted by time (`kind="stable"`). Spikes are then counted into 20 ms bins over each trial's `[-0.5, 1.5] s` window using `iblutil.numerical.bincount2D` with `xlim=[t_beg, t_end]`, truncated to the first 100 bins. **The stored values are raw spike counts, not firing rates** (no division by bin width), cast to `float16`. No smoothing, no z-scoring, no region selection. Exported: 72,757 neurons, mean 166.1/session (min 3, max 524).

ii.
```python
good_mask = cluster_labels >= label_threshold
selected_cluster_ids = np.flatnonzero(good_mask)
spike_keep = good_mask[spikes_clusters]
remap = np.full(cluster_labels.shape[0], -1, dtype=np.int32)
remap[selected_cluster_ids] = np.arange(selected_cluster_ids.size, dtype=np.int32)
selected_clusters = remap[np.asarray(spikes_clusters[spike_keep], dtype=np.int64)]
...
        merged_clusters.append(clusters + cluster_offset)
        cluster_offset += good_here
...
order = np.argsort(spike_times, kind="stable")
```
```python
    for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
        trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
        if idx1 > idx0:
            counts, _, cluster_idx = bincount2D(
                spike_times[idx0:idx1], spike_clusters[idx0:idx1],
                xbin=binsize, xlim=[start, end])
            if counts.size:
                counts = counts[:, :n_bins]
                trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
```
```python
data["neural"].append([trial.astype(np.float16) for trial in session.neural])
```

iii. Step 5: "Neural values will be raw binned spike counts, not standardized z-scores" — the reference repo standardizes only inside its own dataloader (`SingleSessionDataset`), so the AI left the export un-normalized. Step 10 Check 3: "`bin_spikes_for_trials()` … correspond[s] conceptually to `bin_spiking_data()`" — indeed it reproduces the reference's `bincount2D(..., xlim=[t_beg, t_end])[:, :n_bins]` call exactly. Probe merging follows `merge_probes` ("Merge probes within a session, reindex cluster ids, sort spikes by time").

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single cut: `clusters.metrics.label >= 1`, applied per probe before spikes are kept. Clusters whose channel index is out of range get brain-location id 0 and are **kept** (they surface as the `void: 250 neurons` entry in the verification output) — i.e. there is no "outside the brain" exclusion, and no firing-rate, region, or per-session minimum-neuron criterion. 595,576 raw clusters in kept sessions → 72,757 exported (75,708 `label >= 1` across all 459 release sessions, matching the data paper's well-isolated-neuron count exactly).

ii.
```python
GOOD_CLUSTER_LABEL = 1
...
metrics = pd.read_parquet(metrics_file, columns=["label"])
cluster_labels = metrics["label"].to_numpy()
good_mask = cluster_labels >= label_threshold
```
```python
valid_channel = (cluster_channels >= 0) & (cluster_channels < len(channel_ids))
cluster_region_ids = np.zeros(selected_cluster_ids.size, dtype=np.int64)
cluster_region_ids[valid_channel] = channel_ids[cluster_channels[valid_channel]]
cluster_regions = br.id2acronym(cluster_region_ids)
```

iii. Step 4 discrepancy table: "`prepare_data()` loads all spike-sorted clusters; metadata stores `good_clusters = label >= 1` … Data paper main analyses restrict to well-isolated neurons … For this export, keep `clusters.metrics.label >= 1`. This matches the data paper's well-isolated-neuron count exactly in the 459-session local release and keeps the dense pickle tractable." Step 10 flags this as "the main deliberate difference" from the executable reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one session clock, so alignment is just windowing: for each kept trial the interval `[stimOn_times - 0.5, stimOn_times + 1.5]` is cut out of the merged spike train (`searchsorted` on both edges, half-open `[t_beg, t_end)`), and `bincount2D` is called with `xlim=[t_beg, t_end]` so bin 0 starts exactly at `stimOn_times - 0.5`. Bin *i* therefore covers `[-0.5 + 0.02i, -0.48 + 0.02i)` relative to stimulus onset. Metadata records `temporal_alignment_event = "stimulus onset (stimOn_times)"`, `off_start = -0.5`, `off_end = 1.5`.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
...
intervals = np.c_[align_times + window[0], align_times + window[1]]
idx_starts = np.searchsorted(spike_times, intervals[:, 0], side="left")
idx_ends = np.searchsorted(spike_times, intervals[:, 1], side="left")
...
            counts, _, cluster_idx = bincount2D(..., xlim=[start, end])
```

iii. Step 4: the method paper's prose aligns wheel/whisker to first movement, but "the user explicitly requires 'Temporally align based on stimulus onset' and the reference code implements this directly", so the executable settings `align_time='stimOn_times'`, `time_window=(-.5, 1.5)` were used for every stream.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per 2 s trial, identical for every trial and session (`Mean T: 100.0` for all 438 sessions). No rebinning, resampling, or smoothing of the spike data — spikes are binned once, directly at 20 ms. `metadata['time_bin_size'] = 20.0` (ms).

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))   # 100
...
    n_bins = int(np.ceil(interval_len / binsize))
...
"time_bin_size": 20.0,
```

iii. Step 4: "Method paper has internal inconsistency: generic model description says 20 ms, but data-processing prose states 50 ms bins for choice / prior. **Resolution:** Use 20 ms bins. This matches the executable code and the top-level model description (`T = 100` over 2 s)."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Not derived from raw data beyond the alignment event `stimOn_times`: it is a constructed constant grid of 100 values, identical for every trial and session, expressing the time of each bin relative to stimulus onset. The values are the **right edges** of the neural bins: `linspace(-0.5 + 0.02, 1.5, 100)` = `[-0.48, -0.46, …, 1.50]`.

ii.
```python
def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. Step 5 mapping table: "common trial time grid → `input[0]` … Planned values: `[-0.48, -0.46, ..., 1.50]` s **to match behavior interpolation grid**" — i.e. deliberately chosen to be the same grid the reference's `get_behavior_per_interval` uses (`np.linspace(interval_beg + binsize, interval_end, n_bins)`).

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. One vector is built once and reused for every trial in every session; it is stacked with the block counter into the `(2, 100)` per-trial input array and cast to `float32`.

ii.
```python
time_input = make_time_input()
...
    input_trial = np.vstack([
        time_input,
        np.full(N_BINS, block_num, dtype=np.float32),
    ]).astype(np.float32)
    inputs.append(input_trial)
```

iii. Step 10 sanity check: "exported `time_since_stimulus_onset` matched the directly reconstructed `np.linspace(-0.48, 1.5, 100)` grid (`np.allclose == True`)".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It indexes the same bin grid as the neural data, but reports each bin's right edge rather than its centre: neural bin *i* spans `[-0.5 + 0.02i, -0.48 + 0.02i)` while `input[0][i] = -0.48 + 0.02i`. This is a constant 10 ms (half-bin) offset from the bin centre, and it is exactly the convention of the reference repo's behavior interpolation grid, so the time input, the wheel trace and the whisker trace are all sampled at the same instants.

ii.
```python
# neural grid
counts, _, cluster_idx = bincount2D(..., xbin=binsize, xlim=[start, end])
# input / behavior grid
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
time_input = np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS)
```

iii. Step 5: chosen "to match behavior interpolation grid"; the reference `get_behavior_per_interval` uses `x_interp = np.linspace(interval_begs[i] + binsize, interval_ends[i], n_bins)`, which the AI ported verbatim.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the raw (unfiltered) trials table. The trials table has no block id, so a block boundary is inferred from any change of `probabilityLeft` between consecutive rows.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. Step 5 mapping table: "raw `probabilityLeft` block sequence → `input[1]` … Block counter resets whenever `probabilityLeft` changes."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A running counter over the **unfiltered** trial sequence, starting at 1 and resetting to 1 whenever `probabilityLeft` changes. It is computed once per session before any masking, then indexed back with each kept trial's original row position (`masked_keep["index"]`), so trials removed by QC still advance the counter. The scalar is broadcast across all 100 bins as a `float32` row of the input array. Observed range in the export: `[1, 99]`, consistent with the paper's 90-trial unbiased block and 20–100-trial biased blocks.

ii.
```python
def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    counters = np.zeros(len(prob_left), dtype=np.float32)
    count = 1
    counters[0] = count
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            count += 1
        else:
            count = 1
        counters[i] = count
    return counters
```
```python
block_vals = block_trial_number[masked_keep["index"].to_numpy()]
...
    input_trial = np.vstack([time_input, np.full(N_BINS, block_num, dtype=np.float32)])
```

iii. Step 5 Key Decision 8: "Compute trial number in block on the original unfiltered trial table. Excluded trials should not renumber the latent block progression." Step 10 sanity check: "exported `trial_number_in_block` matched a direct block counter computed from the raw `probabilityLeft` sequence (`np.allclose == True`)."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is `+1` / `-1` / `0`. `0` (no response) trials are already removed by the trial mask, so only `±1` reach the mapping; the mapper raises if anything else appears.

ii.
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

iii. Step 5 mapping table: "raw `choice` → `output[0]` … Trials with `choice == 0` are excluded by reference mask". Key Decision 9: "Map choice polarity using rewarded nonzero-contrast trials. Local data confirm `choice == 1` corresponds to left and `choice == -1` to right" — i.e. the sign convention was verified empirically against the raw data rather than assumed.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding `+1 → 0` (left), `-1 → 1` (right), per the Decoder Task spec, with a guard that errors out if any trial is unmapped. The per-trial scalar is broadcast across all 100 bins and stored as `int16`. Exported distribution: left 0.509 / right 0.491.

ii.
```python
def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped = np.full(choice_values.shape, -1, dtype=np.int16)
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right
    if np.any(mapped < 0):
        raise ValueError("Unexpected choice values after masking")
    return mapped
...
    choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. Step 4: "Decoder task requires left = 0, right = 1 … After applying the no-choice mask, map the two remaining raw values to `{0, 1}` in the target export." Step 10 sanity check: "exported choice and prior arrays matched direct raw-trial remapping for trials 0, 5, and 10 … (`np.allclose == True`)".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the three task values 0.2, 0.5, 0.8.

ii.
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

iii. Step 4: "`probabilityLeft` … Raw values are `0.2`, `0.5`, `0.8` … Papers describe unbiased 0.5 block then biased 0.2/0.8 blocks."

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the recoding `0.2 → 0`, `0.5 → 1`, `0.8 → 2` using `np.isclose` (float-safe), with a guard that raises and prints the offending values if any trial falls outside the three levels. Broadcast across 100 bins as `int16`. Exported distribution: `[0.419, 0.141, 0.441]`, i.e. the 0.5 unbiased block is the minority, as expected from a 90-trial unbiased block followed by many biased blocks.

ii.
```python
def map_prior_to_categorical(prob_left: np.ndarray) -> np.ndarray:
    mapped = np.full(prob_left.shape, -1, dtype=np.int16)
    mapped[np.isclose(prob_left, 0.2)] = 0
    mapped[np.isclose(prob_left, 0.5)] = 1
    mapped[np.isclose(prob_left, 0.8)] = 2
    if np.any(mapped < 0):
        vals = np.unique(prob_left[mapped < 0])
        raise ValueError(f"Unexpected probabilityLeft values: {vals}")
    return mapped
```

iii. Step 4: "Export prior as the user-required categorical mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`." Step 5 mapping table: "User requested this categorical coding explicitly."

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, processed with the bundled reference `brainbox.behavior.wheel` functions (imported from `/app/code/ibllib`) rather than through `SessionLoader`. Speed is `|velocity|`.

ii.
```python
sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "ibllib"))
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
...
def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wheel_pos_file = pick_one_file(session_path / "alf", "_ibl_wheel.position.npy")
    wheel_ts_file = pick_one_file(session_path / "alf", "_ibl_wheel.timestamps.npy")
    ...
    return ts_interp, np.abs(vel)
```

iii. Step 5 mapping table: "wheel stream → `output[2]` … Load wheel velocity, take absolute value to obtain speed", referencing `load_target_behavior('wheel-speed')` in the reference repo, which does the same `np.abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages, the first two copied from the IBL/reference library: (1) `interpolate_position(ts, pos, freq=1000)` puts the irregularly sampled wheel position on an even 1000 Hz grid; (2) `velocity_filtered(pos_interp, 1000)` differentiates with the default 20 Hz Butterworth low-pass, giving velocity in rad/s, of which the absolute value is taken; (3) the resulting session-long trace is linearly interpolated (`scipy.interp1d`, `fill_value="extrapolate"`) onto each trial's 100-point grid, with the reference's edge-coverage test applied first. If the timestamps are the 2-column `(index, time)` form they are collapsed by row mean.

ii.
```python
    if ts.ndim == 2 and ts.shape[1] == 2:
        ts = ts.mean(axis=1)
    pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
    vel, _ = velocity_filtered(pos_interp, 1000)
    return ts_interp, np.abs(vel)
```
```python
    idx_beg = np.searchsorted(target_times, align_times + window[0], side="right")
    idx_end = np.searchsorted(target_times, align_times + window[1], side="left")
    for i, align_time in enumerate(align_times):
        vals = target_values[idx_beg[i]: idx_end[i]]
        ts   = target_times[idx_beg[i]: idx_end[i]]
        if len(vals) == 0: outputs.append(None); continue
        if np.abs(start - ts[0]) > binsize or np.abs(end - ts[-1]) > binsize:
            outputs.append(None); continue
        interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
        outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. Step 1/Step 10: `interpolate_behavior_trials()` is documented as the counterpart of the reference `get_behavior_per_interval()`, and it reproduces its searchsorted slicing, its "target data starts too late / ends too early" `> binsize` tests, and its `interp1d(..., fill_value='extrapolate')` call. The 1000 Hz + 20 Hz-filter pipeline is the IBL library default.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 classes by **dataset-wide (global) tertiles**: every kept trial's interpolated speed trace from every session is pooled, the 1/3 and 2/3 quantiles are taken once, and the same two edges are applied to all sessions via `np.digitize`. The edges are recorded in metadata (`0.01514`, `0.40303` rad/s). Degenerate cases (non-finite or `q1 >= q2`) fall back to equal-width thirds of the range, or to `min + eps`/`min + 2 eps` if the trace is constant. Global distribution is `[0.333, 0.333, 0.333]` by construction, but the per-session class-0 fraction ranges from about 0.096 to 0.584.

ii.
```python
def compute_tertile_edges(values):
    concat = np.concatenate(flat_values)
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        ...
    return float(q1), float(q2)

def discretize_three_bins(values, edges):
    return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)
...
wheel_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.wheel_cont)
```

iii. Step 5 Key Decision 7: "Discretize wheel speed and whisker motion energy using global tertiles. Global edges preserve a common categorical meaning across sessions and should keep class balance better than fixed-width bins." Step 4: "dynamic outputs will be discretized after interpolation onto the shared stimulus-onset grid, because the target format requires categorical outputs".

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated on the same 100-point per-trial grid as the time input, `stimOn_times + linspace(-0.48, 1.5, 100)` — i.e. at the right edge of each 20 ms neural bin, on the same session clock, so wheel sample *i* and neural bin *i* correspond. Trials where the wheel does not span the window (edge gap > 20 ms) are dropped from the session entirely, keeping neural, input and output trial lists in lockstep.

ii.
```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
...
        outputs.append(interp(align_time + x_rel).astype(np.float32))
...
combined_mask = wheel_mask & whisk_mask & neural_mask
neural_keep = [neural_trials[i] for i, keep in enumerate(combined_mask) if keep]
wheel_keep  = [wheel_trials[i]  for i, keep in enumerate(combined_mask) if keep]
```

iii. Step 5: all streams share "the shared stimulus-onset-aligned 20 ms grid". Step 10 sanity check: "exported wheel-speed and whisker-motion-energy class arrays matched direct raw interpolation plus discretization at session `ebce500b-…`, trial 5 (`np.allclose == True`)".

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`; if the left camera is unavailable the right camera is used instead (`rightCamera.ROIMotionEnergy.npy` + `*rightCamera.times.npy`). The camera actually used is recorded per session (`whisker_source`) and exercised in practice (e.g. session `0c828385-…` used the right-camera fallback). Sessions with neither are skipped (14 sessions).

ii.
```python
def load_whisker_motion_energy(session_path: Path):
    for camera in ("left", "right"):
        me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
        times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
        if me_file is None or times_file is None:
            continue
        ...
        return times, values, camera
    raise FileNotFoundError(f"Missing whisker motion energy files for {session_path}")
```

iii. Step 4/Step 5 Key Decision 6: "Use left whisker motion energy first, then right as fallback: This exactly matches `bin_behaviors()` and maximizes session retention without inventing a new rule."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, no normalisation, no per-session scaling. Non-finite samples are dropped, and if the motion-energy and times arrays have different lengths both are truncated to the shorter one. The trace is then linearly interpolated onto the same 100-point trial grid by the same `interpolate_behavior_trials` used for the wheel, with the same edge-coverage test.

ii.
```python
        if len(values) != len(times):
            n = min(len(values), len(times))
            values = values[:n]; times = times[:n]
...
    valid_source = np.isfinite(target_times) & np.isfinite(target_values)
    target_times = target_times[valid_source]
    target_values = target_values[valid_source]
...
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
```

iii. Step 3: whisker motion energy is "the mean across pixels of the absolute value of the difference between adjacent frames" in a whisker-pad box — a released quantity, so no recomputation was attempted. Step 5 mapping table: "interpolate onto stimulus-onset grid; discretize into 3 global bins".

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: 3 classes cut at **dataset-wide** 1/3 and 2/3 quantiles of all pooled trial traces from all sessions, applied with `np.digitize`; edges stored in metadata (`2.728`, `7.860`). The global distribution is `[0.333, 0.333, 0.333]`, but because ROI motion energy is in arbitrary camera-dependent units the per-session class-0 fraction ranges from about 0.005 to 0.891 — many sessions are almost entirely one class.

ii.
```python
whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont)
print(f"Whisker motion energy tertile edges: {whisker_edges}")
...
            output_trial = np.vstack([
                choice, prior,
                discretize_three_bins(wheel_cont, wheel_edges),
                discretize_three_bins(whisk_cont, whisker_edges),
            ]).astype(np.int16)
```

iii. Step 5 Key Decision 7 (same rationale as the wheel): "Global edges preserve a common categorical meaning across sessions and should keep class balance better than fixed-width bins." Step 9 records the resulting global distribution as "Yes by construction"; per-session imbalance is not examined anywhere in the notes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: sampled at `stimOn_times + linspace(-0.48, 1.5, 100)` on the shared session clock, one value per neural bin, with trials whose camera coverage has an edge gap > 20 ms dropped from all streams together.

ii.
```python
whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
combined_mask = wheel_mask & whisk_mask & neural_mask
whisk_keep = [whisk_trials[i] for i, keep in enumerate(combined_mask) if keep]
```

iii. Step 10 sanity check: "exported whisker classes also matched direct right-camera interpolation in right-fallback session `0c828385-6dd6-4842-a702-c5075f5f5e81` (`np.allclose == True`)."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered and mostly by dropping. (a) Missing files: a session with no wheel (6 sessions) or no motion energy from either camera (14 sessions) is skipped with a printed reason; a probe missing any of its five required assets raises `FileNotFoundError`; a session with 0 good clusters is skipped. (b) Missing trial fields: NaNs in the six required trial columns remove the trial. (c) Partial coverage: a trial whose wheel or camera trace does not span the window within one bin is dropped. (d) Length mismatch between motion energy and camera times is silently truncated to the shorter array. (e) Non-finite behavior samples are filtered out before interpolation. (f) Out-of-range cluster channel indices default to brain-location id 0 (→ `void`) rather than dropping the unit. (g) Trials whose binned neural matrix is all zeros are dropped (16 trials). (h) A session with < 2 usable trials is skipped (1 session). (i) Unexpected `choice` / `probabilityLeft` values raise rather than being silently coerced. Net loss: 459 → 438 sessions, 139 → 135 subjects, all accounted for in the notes.

ii.
```python
    try:
        wheel_times, wheel_speed = load_wheel_speed(spec.session_path)
        whisk_times, whisk_values, whisk_source = load_whisker_motion_energy(spec.session_path)
    except FileNotFoundError as exc:
        print(f"[skip] {spec.eid}: {exc}")
        return None
...
    if n_clusters_good == 0:
        print(f"[skip] {spec.eid}: no good clusters after QC"); return None
    if len(masked_trials) < 2:
        print(f"[skip] {spec.eid}: fewer than 2 trials after trial mask"); return None
    if combined_mask.sum() < 2:
        print(f"[skip] {spec.eid}: fewer than 2 trials after behavior/neural alignment"); return None
```
```python
    if any(x is None for x in required):
        raise FileNotFoundError(f"Missing spike sorting assets under {probe_path}")
```

iii. Step 9: "Session loss relative to the 459-session release is fully explained by 21 unusable sessions: 6 with missing wheel files, 14 with missing whisker motion energy, 1 with fewer than 2 trials surviving behavior alignment." Step 10: the all-zero neural filter was added specifically to clear verification warnings — "These came from genuine raw windows with zero spikes across all exported neurons. I fixed this by filtering out all-zero neural trial matrices before export."

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk dominates: `spikes.times.npy` / `spikes.clusters.npy` run to ~20 M entries per probe (~330 MB for the pair), and the boolean-mask read `spikes_times[spike_keep]` materialises them despite the memmap. The per-session log shows time scaling with probe count and cluster count (1.8 s for a 17-cluster session up to 19.3 s for a 220-cluster, 1026-trial session). Secondary costs: the per-trial `bincount2D` loop, the per-trial `interp1d` construction, the 1000 Hz wheel interpolation over the whole session, and instantiating `BrainRegions()` once per session. Total full run: 14m 0.6s with 8 threads (just inside the 15-minute budget).

ii.
```python
    spikes_times = np.load(spikes_times_file, mmap_mode="r")
    spikes_clusters = np.load(spikes_clusters_file, mmap_mode="r")
    spike_keep = good_mask[spikes_clusters]
    selected_times = np.asarray(spikes_times[spike_keep], dtype=np.float64)
...
print(f"[ok] {spec.eid}: raw_trials=..., elapsed={format_seconds(elapsed)}")
```

iii. Step 6: "Trial-by-trial spike binning is the main hot path"; speed-ups listed as "Spike arrays are memory-mapped and only QC-passing spikes are materialized", "Spike counting uses `bincount2D` on pre-windowed spike subsets", "Behavior interpolation reuses a shared fixed grid per trial window", plus "Session-level threaded parallelism … capped at 8 for full run" which cut the 2-session sample from 31.0 s to 16.4 s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four. (1) `compute_trial_number_in_block` is a pure-Python loop over every trial; it is exactly `trials.groupby((p != p.shift()).cumsum()).cumcount()` in pandas, or `np.arange - np.maximum.accumulate(boundary_index)` in numpy. (2) The per-trial loop in `bin_spikes_for_trials` calls `bincount2D` once per trial; a single `np.bincount` over `unit * n_bins + bin_index` with a trial offset would do all trials at once. (3) The per-trial loop in `interpolate_behavior_trials` builds a fresh `scipy.interp1d` object per trial — `np.interp` on a single concatenated query vector would avoid ~186k object constructions. (4) The `[np.any(trial) for trial in neural_trials]` list comprehension and the three `[... for i, keep in enumerate(combined_mask) if keep]` comprehensions could be array operations. None of these were vectorized; the notes only acknowledge the spike-binning loop as a hot path.

ii.
```python
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            count += 1
        else:
            count = 1
        counters[i] = count
```
```python
    for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
        ...
            counts, _, cluster_idx = bincount2D(...)
```
```python
    for i, align_time in enumerate(align_times):
        ...
        interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
        outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. Step 6 lists only "Trial-by-trial spike binning is the main hot path" under inefficiencies; the block-counter and `interp1d` loops are not mentioned. The AI's stated remedy was parallelism across sessions rather than vectorization within a session, and it judged the resulting 14-minute runtime acceptable against the 15-minute target.

## 10-c. What processing does the code repeat multiple times?

i. (1) `BrainRegions()` is constructed inside `process_session_worker`, i.e. once per session (459 times), each time re-reading and re-indexing the Allen atlas tables; the reference builds it once at module level. (2) `pick_one_file` calls `Path.rglob` over the whole `alf/` subtree for each of ~7 dataset patterns per session, rescanning the same directory tree repeatedly. (3) `discretize_three_bins` is applied to the wheel and whisker traces again in `plot_processing_summary` after already being applied in `build_data_dict` (only for the ≤2 plotted sessions). (4) Every trial's neural matrix is cast with `.astype(np.float16)` in `build_data_dict` although `bin_spikes_for_trials` already allocated it as `float16`, and inputs are `.astype(np.float32)` twice. (5) `np.asarray(..., dtype=np.float32)` is applied to the wheel/whisker traces that `interpolate_behavior_trials` already returned as `float32`.

ii.
```python
def process_session_worker(spec: SessionSpec) -> ProcessedSession | None:
    return process_session(spec, BrainRegions())
```
```python
def pick_one_file(base: Path, pattern: str) -> Path | None:
    matches = list(base.rglob(pattern))
```
```python
        data["neural"].append([trial.astype(np.float16) for trial in session.neural])
        data["input"].append([trial.astype(np.float32) for trial in session.inputs])
```

iii. Not identified anywhere in CONVERSION_NOTES.md — the Step 6 "Code inefficiencies identified" section lists only serial session processing and trial-by-trial spike binning. The redundant work is real but small relative to spike I/O, and the AI's efficiency review stopped once the full run fit inside the 15-minute budget.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `velocity_filtered` returns acceleration, which is computed and immediately discarded (`vel, _ = ...`). (2) The whole session's wheel position is interpolated to 1000 Hz and filtered even though only 100 samples per trial (≈2 s of each ~10 s trial spacing) are ever read. (3) The continuous `wheel_cont` / `whisker_cont` float32 traces for all 186,245 trials are retained in memory for the entire run purely so global tertiles can be computed at the end, then discarded after discretization — a direct cost of the global-tertile decision. (4) `n_clusters_total` (595,576 raw clusters) and `raw_n_trials` are tracked per session only for log lines. (5) `kept_trial_indices` is stored on every `ProcessedSession` and never used after the block counter is indexed. (6) The `neural_mask` `np.any` scan runs over every trial of every session to remove 16 trials. (7) `cluster_regions` is materialised as an `object`-dtype array and re-stringified (`str(r)`) at assembly.

ii.
```python
    vel, _ = velocity_filtered(pos_interp, 1000)
```
```python
        wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
        whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],
        kept_trial_indices=masked_keep["index"].to_numpy(dtype=np.int32),
        n_clusters_total=n_clusters_total,
        raw_n_trials=raw_n_trials,
```
```python
    neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
```

iii. Not identified in CONVERSION_NOTES.md. Items (1) and (2) are inherited from the reference `brainbox` wheel API and so are arguably unavoidable if the reference pipeline is to be reproduced exactly; items (4)–(5) exist to support the reporting and sanity checks the instructions asked for. Item (3) is a genuine consequence of choosing global rather than per-session discretization thresholds, and is what forces the two-pass structure (all sessions must finish before any output can be discretized).
