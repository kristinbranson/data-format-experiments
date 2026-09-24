# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every NWB file with a single glob over the data directory (`/app/data/sub-*/sub-*_behavior+ophys.nwb`, sorted), giving 152 files = 152 sessions across 11 subject directories. Each file is opened **directly with `h5py`** rather than `pynwb`, and only the groups it needs are read: the ten framewise `BehavioralTimeSeries` streams, the sparse `Reward` data/timestamps, the `ImageSegmentation/PlaneSegmentation` `iscell` and `planeIdx` tables, and every `Deconvolved/plane*` response series. Sessions are processed one at a time and released before the next is loaded. `--sample` restricts the file list to 2 sessions (one single-plane, one multi-plane).

ii.
```python
files = sorted(Path("/app/data").glob("sub-*/sub-*_behavior+ophys.nwb"))
```
```python
def load_session_raw(path):
    with h5py.File(path, "r") as f:
        scene = read_str(f["identifier"]).split("/")[-1]
        subject = read_str(f["general/subject/subject_id"])
        session_id = read_str(f["general/session_id"])

        behavior = {}
        for name in ["environment", "position", "speed", "lick", "reward_zone",
                     "scanning", "trial number", "trial_start", "teleport", "autoreward"]:
            behavior[name] = f[f"{BEHAVIOR_TS_PATH}/{name}/data"][:]
        behavior["timestamps"] = f[f"{BEHAVIOR_TS_PATH}/position/timestamps"][:]
        reward_timestamps = f[f"{BEHAVIOR_TS_PATH}/Reward/timestamps"][:]
        ...
        plane_series = sorted(f[f"{OPHYS_TS_PATH}/Deconvolved"].keys())
```

iii. From CONVERSION_NOTES Step 2/Step 6: "Loader uses direct `h5py` access instead of `pynwb` for speed and lower overhead"; "Process sessions sequentially and release each session before moving to the next"; "Use raw HDF5 slicing and load only the curated deconvolved traces actually used." The AI verified the inventory against the paper: 11 switch-task mice, 14 task days each except m11 (imaging started on day 3), i.e. 152 sessions, and it cross-checked the NWB `identifier` scene strings against the scene names in the reference `sessions_dict.py`.

## 1-b. How are the data split into subjects?

i. Subject identity is read from the NWB metadata field `general/subject/subject_id` (e.g. `m11`), not from the directory name. A dictionary assigns each newly encountered subject the next integer index; `data['subjects']` is the list of subject ids in first-encountered order and `data['subject_idx']` is the per-session index into that list. Result: 11 subjects with 12 (m11) or 14 sessions each.

ii.
```python
subject = session_raw["subject"]
if subject not in subject_lookup:
    subject_lookup[subject] = len(subject_lookup)
    data["subjects"].append(subject)
subject_idx.append(subject_lookup[subject])
...
data["subject_idx"] = np.asarray(subject_idx, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2 lists the 11 `sub-m*` folders and the per-subject session counts; the variable-mapping table records "NWB subject id -> `subjects`, `subject_idx`; normalize to strings like `m3`, `m11`". Step 9 reports the match against the paper's "n = 11 mice" switch cohort.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. No pooling or splitting across files, and no cross-session neuron alignment. `general/session_id` and the file name (`ses-NN`) are kept in `metadata['session_info']` per session; session order in `neural`/`input`/`output` is the sorted file order. A session is dropped only if it yields fewer than 2 valid trials (this never fired: all 152 sessions are retained).

ii.
```python
for session_i, path in enumerate(files):
    session_raw = load_session_raw(path)
    ...
    if len(neural_trials) < 2:
        print(f"Skipping {path.name}: only {len(neural_trials)} valid trials after filtering.")
        continue
    data["neural"].append(neural_trials)
```

iii. CONVERSION_NOTES Step 2: "Session file naming pattern: `sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`"; Step 4 resolves the session inventory against the reference `sessions_dict.py` ("Treat NWB `ses-01..14` as task day / experimental day ... m11 missing days 1-2 is expected"). The <2-trial guard exists because the decoder spec requires "at least two trials within each session".

## 1-d. How are the data split into trials?

i. A trial is the half-open frame interval `[trial_start, teleport)`: every index where the framewise `trial_start` stream is positive is a trial start, and the trial ends at the *next* index where the `teleport` stream is positive. `build_trial_bounds` walks a pointer through the teleport indices so each teleport is consumed once; a start with no following teleport is dropped, as is any pair with `stop <= start`. This deliberately excludes the inter-trial tunnel/teleport period. The stored `trial number` stream is explicitly *not* used to define boundaries. Over the full dataset this yields 12,216 trial bounds.

ii.
```python
def build_trial_bounds(behavior):
    starts = np.flatnonzero(behavior["trial_start"] > 0)
    teleports = np.flatnonzero(behavior["teleport"] > 0)
    bounds = []
    tele_ptr = 0
    for start in starts:
        while tele_ptr < len(teleports) and teleports[tele_ptr] <= start:
            tele_ptr += 1
        if tele_ptr >= len(teleports):
            break
        stop = teleports[tele_ptr]
        tele_ptr += 1
        if stop <= start:
            continue
        bounds.append((int(start), int(stop)))
    return bounds, starts, teleports
```

iii. CONVERSION_NOTES Step 4/Step 5: "Reference code uses `trial_start_inds` and `teleport_inds`, slicing `start-1:stop-1` to exclude teleport/tunnel ... Build trials from `trial_start` to `teleport` (on-track segment only), not from `trial number` alone"; "This matches the reference code's `trial_start_inds` to `teleport_inds` logic and excludes the variable-length tunnel/teleport period." The AI also used this to resolve the `sub-m11_ses-03` edge case: that session has 81 distinct nonnegative `trial number` labels but only 80 start/teleport pairs, and direct inspection showed the extra label is a trailing post-teleport tunnel fragment with no `trial_start`, so start/teleport pairs are treated as authoritative.

## 1-e. How are trials filtered based on quality controls?

i. Two trial-level filters plus one session-level filter:
- **Empty trials** (`pos.size == 0`) are dropped (counter `dropped["empty"]`; never fired).
- **Lick-sensor artifact trials**: a trial is dropped if more than 35% of its frames have a cumulative lick count > 2. 69 of 12,216 trials (0.56%) were dropped this way, leaving 12,147.
- **Sessions with < 2 surviving trials** are skipped entirely (never fired).

No minimum-trial-length filter is applied (the shortest surviving trial is 96 frames).

ii.
```python
LICK_QC_FRACTION = 0.35
...
        if pos.size == 0:
            dropped["empty"] += 1
            continue

        bad_lick = np.mean(lick_raw > 2.0) > LICK_QC_FRACTION
        if bad_lick:
            dropped["lick_qc"] += 1
            continue
```
```python
        if len(neural_trials) < 2:
            print(f"Skipping {path.name}: only {len(neural_trials)} valid trials after filtering.")
            continue
```

iii. CONVERSION_NOTES Step 3 quotes the Methods: "trials with capacitive-sensor artifacts were removed if >30% of 0.0645 s samples had cumulative lick count >2; methods report 81 / 12,376 imaged trials removed (~0.65%)". Step 4 explains the choice of 0.35 over 0.30: "Follow the actual reference code logic (`>35%` of samples in a trial with lick count >2) because the conversion should match released analysis behavior more closely than a rounded prose threshold." Step 5, decision 7: "Because lick is one of the decoder targets, trials with known sensor artifacts should not be preserved as if they were valid." Step 9 reports the resulting 12,216 -> 12,147 accounting.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` array is taken **directly from the NWB `processing/ophys/Deconvolved/plane*` `RoiResponseSeries`** — suite2p's own `spks` output, stored in raw-fluorescence units. `Fluorescence` (F) and `Neuropil` (Fneu) are present in every file but are never read by the conversion script. For multi-plane subjects (m17, m18) each plane's `Deconvolved` series is read and the curated columns are concatenated along the neuron axis.

ii.
```python
        plane_series = sorted(f[f"{OPHYS_TS_PATH}/Deconvolved"].keys())
        ...
        for series_name in plane_series:
            response_group = f[f"{OPHYS_TS_PATH}/Deconvolved/{series_name}"]
            rois = response_group["rois"][:]
            response_iscell = iscell[rois, 0].astype(np.int64) == 1
            response_data = response_group["data"][:common_len, :]
            neural_planes.append(response_data[:, response_iscell].astype(np.float32, copy=False))
        neural = np.concatenate(neural_planes, axis=1)
```
```python
        "neural_signal": "Deconvolved calcium events from NWB ophys/Deconvolved/plane0",
```

iii. CONVERSION_NOTES Step 4, "Neural signal choice": "Reference decoder notebook uses `sess.timeseries['events']` from deconvolution / NWB contains `Fluorescence`, `Neuropil`, and `Deconvolved`, but no explicit dF/F / Paper decoder uses 'deconvolved calcium event timeseries' -> Use NWB `Deconvolved` for neural input, with curated cells only." Step 5, decision 2: "Use curated deconvolved activity rather than recomputing dF/F: The paper's decoder and many downstream analyses use deconvolved events, and the NWB export already contains that signal. Recomputing dF/F from fluorescence/neuropil would add avoidable mismatches." Step 4 final understanding: "The task-relevant neural signal for conversion is curated deconvolved activity, not raw fluorescence and not recomputed dF/F, because the NWB export already contains the deconvolved trace used by the reference decoder analyses." The AI never tested this equivalence assumption.

## 2-b. How is the `neural` data processed?

i. Essentially no processing. The stored `Deconvolved` matrix is sliced to the common neural/behavior length, subset to curated cells, cast to `float32`, concatenated across planes, then per trial sliced `[start:stop]` and transposed to `(n_neurons, n_timepoints)`. There is no neuropil subtraction, no maximin baseline, no dF/F normalisation, no Gaussian smoothing, and no OASIS deconvolution — none of the `preprocessing.dff` pipeline the paper's Methods describe is reimplemented.

ii.
```python
            response_data = response_group["data"][:common_len, :]
            neural_planes.append(response_data[:, response_iscell].astype(np.float32, copy=False))
        neural = np.concatenate(neural_planes, axis=1)
```
```python
        neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. Same justification as 2-a: the AI concluded the NWB export "already contains the deconvolved trace used by the reference decoder analyses", so recomputation "would add avoidable mismatches". CONVERSION_NOTES Step 1 does record the reference `dff` function and its parameters (`neu_coef=0.7`, `maximin` baseline, `keep_teleports=False`, OASIS deconvolution), but Step 4 rules it out on the equivalence assumption above.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: suite2p's manual curation flag, `iscell[:, 0] == 1`, applied per response series through that series' own ROI table region (`rois`) so multi-plane files index the right rows of `PlaneSegmentation`. This yields 138,678 curated cells (mean 912.4/session, range 155-2341). No further neuron filtering is applied — in particular the putative-interneuron exclusion (dF/F vs running-speed Pearson r > 0.5) is not implemented. All curated cells are used, not a place-cell / RR / TR subset.

ii.
```python
        iscell = f[f"{OPHYS_TS_PATH}/ImageSegmentation/PlaneSegmentation/iscell"][:]
        plane_idx_all = f[f"{OPHYS_TS_PATH}/ImageSegmentation/PlaneSegmentation/planeIdx"][:]
        ...
            rois = response_group["rois"][:]
            response_plane_idx = plane_idx_all[rois]
            response_iscell = iscell[rois, 0].astype(np.int64) == 1
            ...
            plane_idx_curated.append(response_plane_idx[response_iscell].astype(np.int64, copy=False))
```

iii. CONVERSION_NOTES Step 4, "Cell filtering": "Reference preprocessing relies on manual Suite2p curation ... Keep only ROIs with `iscell[:,0] == 1`; no extra automatic neuron-quality filter unless later validation exposes a problem." Step 1: "Cell curation is manual upstream through Suite2p `iscell.npy`; I have not yet seen additional automatic neuron-quality filtering inside this repo beyond using the curated `sess` objects and downstream place-cell classifications." Step 5, decision 1: "Use all curated neurons, not only place-cell or RR/TR/non-RR subsets: the user-specified decoder ... needs general-purpose neural inputs for all requested outputs." Step 9 checks the count against the paper's "155-2172 putative pyramidal neurons per session" and flags that the released max (2341) exceeds the paper's max.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires no extra work: the neural and behavioral streams are stored on the same frame grid, so the same index range `[trial_start, teleport)` is used for both. Each trial therefore begins exactly at the trial-start frame (`off_start = 0.0`, no pre-event window) and runs to the frame before teleport. Before trials are cut, all neural and behavior arrays are truncated to a common length so the shared indexing is valid.

ii.
```python
        neural_len = min(neural_lens)
        behavior_len = min(len(v) for v in behavior.values())
        common_len = min(neural_len, behavior_len)
        ...
        for key in behavior:
            behavior[key] = behavior[key][:common_len]
```
```python
    for raw_trial_idx, (start, stop) in enumerate(bounds):
        pos = behavior["position"][start:stop]...
        neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```
```python
        "temporal_alignment_event": "trial start (entry onto the 450 cm track)",
        "off_start": 0.0,
        "off_end": None,
        "trial_window": "Samples from trial_start inclusive to teleport exclusive",
```

iii. CONVERSION_NOTES Step 1/Step 4: "Behavioral data are aligned to imaging frames before most downstream analysis. The aligned framewise table is `sess.vr_data`, with one row per imaging sample"; "Reference analyses treat behavioral and neural streams as synchronized framewise time series." Step 10 sanity checks confirmed with `np.allclose()` that converted neural trials equal the raw curated `Deconvolved` slice at the same frame indices, for one single-plane and two multi-plane sessions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning.** Data are kept at the native imaging/behavior frame grid of ~15.5078 Hz, i.e. 64.4836 ms per bin. `metadata['time_bin_size']` is computed empirically as the median (across sessions) of the per-session median inter-timestamp interval of the behavior clock, times 1000. The AI explicitly rejected the `rate` attribute on the multi-plane (m17/m18) `Deconvolved` series, which reads 31.015625 Hz because that is the scanner rate across two interleaved planes rather than the per-plane sampling rate.

ii.
```python
        dt_behavior = float(np.median(np.diff(behavior["timestamps"][: min(common_len, 1000)])))
```
```python
def build_metadata(session_infos, subjects):
    dt_values = [info["dt_behavior_sec"] for info in session_infos if info["n_trials_kept"] > 0]
    dt_sec = float(np.median(dt_values)) if dt_values else DEFAULT_DT_SEC
    ...
        "time_bin_size": dt_sec * 1000.0,
```

iii. CONVERSION_NOTES Step 3 quotes the Methods: "All behavioral and neural time series were sampled at ~15.5 Hz" and "frames bidirectionally imaged at ~31 Hz interleaved in the scan for a sampling rate of ~15.5 Hz per plane". Step 4: "NWB deconvolved series has `rate=31.015625`, but behavior timestamps advance by ~0.06448 s (15.5078125 Hz) and match sample count ... Treat sample spacing as 0.0644836272 s from behavior timestamps and ignore the misleading deconvolved `rate` attribute for multi-plane sessions." Step 5, decision 4: "Trust behavior timestamps for sample spacing."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attached to the `position` behavioral time series (seconds), stored once per session as `behavior['timestamps']` and truncated to the common neural/behavior length.

ii.
```python
        behavior["timestamps"] = f[f"{BEHAVIOR_TS_PATH}/position/timestamps"][:]
```
```python
        time_raw = behavior["timestamps"][start:stop].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Behavior timestamps (`position/timestamps`) -> `input[0]` (`time_from_trial_start_sec`)", referencing "Reference `sess.vr_data['time']` usage throughout repo". The AI treats every behavior stream as sharing one frame clock (Step 4, "Reference analyses treat behavioral and neural streams as synchronized framewise time series").

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the first timestamp of the trial, so every trial starts at exactly 0.0 s; cast to `float32`. Nothing else. Resulting range over the full dataset is [0.0, 216.5] s.

ii.
```python
        time_from_start = (time_raw - time_raw[0]).astype(np.float32, copy=False)
        ...
        input_trial = np.vstack([
                time_from_start,
                ...
        ])
```

iii. CONVERSION_NOTES Step 5: "For each trial, subtract the trial start timestamp to get elapsed seconds per sample"; listed as a "Time-varying continuous input". Step 10 verified by `np.allclose()` against independently recomputed raw values for three trials.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `[start:stop]` frame slice used for the neural matrix is used for the timestamps, so element *t* of `input[0]` is the same frame as column *t* of the neural matrix. Before slicing, every behavior stream and every neural plane is truncated to `common_len = min(neural_len, behavior_len)`, which fixes the 10 multi-plane sessions whose ophys matrices are one sample longer than behavior; the number of trimmed samples is recorded in `metadata['sessions_trimmed_to_behavior_length']`.

ii.
```python
        neural_len = min(neural_lens)
        behavior_len = min(len(v) for v in behavior.values())
        common_len = min(neural_len, behavior_len)
```
```python
        time_raw = behavior["timestamps"][start:stop]...
        neural_trial = neural[start:stop].T...
```

iii. CONVERSION_NOTES Step 2: "In 10 multi-plane sessions, ophys matrices are one sample longer than behavior matrices"; Step 10: "10 sessions had neural arrays one sample longer than behavior arrays; trimming resolved these without downstream warnings." The AI's Step 10 `np.allclose()` spot checks confirm frame-for-frame agreement between converted trials and raw arrays.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The framewise `environment` behavioral time series (0 = ENV1, 1 = ENV2; negative values mark invalid/tunnel samples).

ii.
```python
            behavior[name] = f[f"{BEHAVIOR_TS_PATH}/{name}/data"][:]   # name == "environment"
...
        env_raw = behavior["environment"][start:stop]
```

iii. CONVERSION_NOTES Step 5 mapping: "`environment` -> `input[1]` (`environment_type`); Map valid values `0 -> ENV1`, `1 -> ENV2`; repeat over trial timepoints", cross-referenced to the reference `behavior.get_trial_types` / `morph` concept. Step 4: "Environment identity is already encoded framewise in NWB `environment` and matches the within-session switch structure."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial the AI discards negative (invalid) samples, takes the **modal** remaining value, and writes that single value into every timepoint of the trial, so environment is constant per trial. It raises if a trial has no valid sample. The output range across the dataset is [0, 1].

ii.
```python
def trial_environment_value(environment_slice):
    valid = environment_slice[environment_slice >= 0]
    if valid.size == 0:
        raise ValueError("No valid environment values within trial slice.")
    values, counts = np.unique(valid, return_counts=True)
    return float(values[np.argmax(counts)])
```
```python
        env_value = trial_environment_value(env_raw)
        ...
                np.full(time_from_start.shape, env_value, dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5, decision 5: "Represent all inputs and outputs as time-varying arrays: Per-trial variables will be repeated across the trial so every trial has a consistent `(n_features, n_timepoints)` structure." The mode-over-valid-samples rule is the AI's robustness handling for the negative sentinel values it found in the `environment` stream (Step 4 notes that environment is constant within a trial and switches between trials on switch days).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The stored framewise `trial number` behavioral time series, sampled at the trial's first frame (`behavior['trial number'][start]`). Trial boundaries themselves still come from `trial_start`/`teleport`; `trial number` is used only as the label.

ii.
```python
        trial_label = int(behavior["trial number"][start])
```

iii. CONVERSION_NOTES Step 5 mapping: "`trial number` -> `input[2]` (`trial_number`); Use per-session trial index (0-based), repeated over timepoints", noting "`glmUtils.get_timeseries_data` writes 0-based `trial_ids`". Step 4: "`trial number` is useful as a contextual label but not as the authoritative trial-boundary definition because it includes tunnel fragments before/after on-track laps" — so it is read at the trial-start frame, where it is unambiguous.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Cast to float and tiled constant across the trial's timepoints. No renumbering after lick-QC drops, so if a trial is dropped the surviving labels keep a gap. Range over the dataset is [0, 99], matching the 80-100 trials/session design.

ii.
```python
                np.full(time_from_start.shape, float(trial_label), dtype=np.float32),
```

iii. Same as 5-a: "Use actual trial index as continuous context input", repeated across timepoints for the uniform `(d_input, n_timepoints)` shape (Step 5, decision 5).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `Reward` event series (its `timestamps`, in seconds). Reward timestamps beyond the end of the behavior clock are discarded, the rest are mapped onto behavior frame indices with `np.searchsorted(..., side="left")` and clipped into range; a trial is "rewarded" if any reward frame index falls in `[start, stop)`.

ii.
```python
        reward_timestamps = f[f"{BEHAVIOR_TS_PATH}/Reward/timestamps"][:]
        ...
        reward_valid = reward_timestamps <= behavior["timestamps"][-1]
        reward_timestamps = reward_timestamps[reward_valid]
        reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps, side="left")
        reward_frame_idx = np.clip(reward_frame_idx, 0, common_len - 1)
```
```python
def trial_reward_outcomes(bounds, reward_frame_idx):
    outcomes = np.zeros(len(bounds), dtype=np.int64)
    for i, (start, stop) in enumerate(bounds):
        has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
        outcomes[i] = int(has_reward)
```

iii. CONVERSION_NOTES Step 4, "Reward representation": "NWB stores `reward_zone` framewise but `Reward` as sparse event timestamps ... Reconstruct framewise/per-trial reward information from `Reward` timestamps aligned to behavior timestamps." Step 4 also validates the reconstruction globally: "across the full dataset, this yields an omission fraction of `0.1535`, consistent with the paper's '~15%'".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial reward outcomes are computed over the **raw** trial-bounds list (before lick-QC dropping) and shifted forward by one: `previous_outcomes[1:] = reward_outcomes_raw[:-1]`, with the first trial of a session set to 0. The value is tiled constant across the trial's timepoints. Because the shift happens on the raw bounds, "previous trial" always means the physically preceding lap even when that lap was later dropped by lick QC.

ii.
```python
    previous_outcomes = np.zeros(len(bounds), dtype=np.int64)
    if len(bounds) > 1:
        previous_outcomes[1:] = reward_outcomes_raw[:-1]
    ...
        previous_outcome = int(previous_outcomes[raw_trial_idx])
        ...
                np.full(time_from_start.shape, float(previous_outcome), dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5 mapping: "Derive reward outcome for each trial from aligned reward events, then shift by one trial; set first trial to 0 ... Consistent with paper's rewarded/omission logic." Step 10 verified `previous_trial_outcome` against independently recomputed raw values with `np.allclose()` for three trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Two things: the framewise `position` stream, and the current trial's reward-zone interval. The zone interval is **not** taken from the framewise `reward_zone` stream (which is loaded but never used); it is derived from the session's scene name, parsed out of the NWB `identifier` field (e.g. `Env1_LocationC_to_B`, `Env2_LocationA`, `Env1_C_to_Env2_A`). The parsed letters are mapped to the paper's fixed coordinates A = 80-130 cm, B = 200-250 cm, C = 320-370 cm. On two-zone (switch) scenes the first zone applies to trials 0-29 and the second to trials 30+.

ii.
```python
REWARD_ZONE_COORDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
SWITCH_TRIAL_INDEX = 30

def parse_scene_reward_sequence(scene):
    match = re.search(r"Location([ABC])_to_([ABC])$", scene)
    if match:
        return [match.group(1), match.group(2)]
    match = re.search(r"Env[12]_([ABC])_to_Env[12]_([ABC])$", scene)
    if match:
        return [match.group(1), match.group(2)]
    match = re.search(r"Location([ABC])$", scene)
    if match:
        return [match.group(1)]
    raise ValueError(f"Unrecognized scene format: {scene}")

def reward_labels_for_trials(scene, n_trials):
    seq = parse_scene_reward_sequence(scene)
    if len(seq) == 1:
        return [seq[0]] * n_trials
    split = min(SWITCH_TRIAL_INDEX, n_trials)
    return [seq[0]] * split + [seq[1]] * max(0, n_trials - split)
```

iii. CONVERSION_NOTES Step 1: "`get_reward_zones` ... Infers reward-zone coordinates and labels (`A/B/C`) from the scene name and switch day structure"; Step 4: "NWB `identifier` suffixes exactly match code scene names for all data sessions checked -> Use NWB `identifier` scene names with the same logic as `behavior.get_reward_zones`". Step 5, decision 6: "Infer reward-zone identity from scene metadata instead of framewise `reward_zone` counts: The framewise `reward_zone` series indicates occupancy/entry events, not the zone identity itself; the scene metadata and reference code provide the intended A/B/C mapping." Step 3 records the coordinates quoted from the Methods ("zone A, 80-130 cm; zone B, 200-250 cm; zone C, 320-370 cm") and the switch rule ("the zone was moved after 30 trials").

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest edge of the current trial's reward-zone interval: negative before the zone (`pos - zone_start`), exactly 0 anywhere inside the zone, positive after the zone (`pos - zone_end`). Computed vectorised over the whole trial, then discretised (7-c).

ii.
```python
def signed_distance_to_zone(position_cm, zone_start, zone_end):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
    return dist
```
```python
        zone_label = reward_labels[raw_trial_idx]
        zone_start, zone_end = REWARD_ZONE_COORDS[zone_label]
        distance = signed_distance_to_zone(pos, zone_start, zone_end)
```

iii. CONVERSION_NOTES Step 5 mapping: "Signed distance to nearest point in current reward-zone interval; negative before zone, 0 inside zone, positive after zone; discretize to 7 bins per task spec." This mirrors the paper's reward-relative position concept (Step 3 notes the paper's RR-position coordinate is centred on reward-zone start).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned with explicit boolean masks rather than `np.digitize`, exactly following the task spec: 0 = `< -50`; 1 = `[-50, -10)`; 2 = `[-10, 0)`; 3 = exactly `0`; 4 = `(0, 10)`; 5 = `[10, 50]`; 6 = `> 50`. `output_values` carries matching labels. The realised distribution over the full dataset is [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243].

ii.
```python
def discretize_distance(distance_cm):
    out = np.full(distance_cm.shape, 6, dtype=np.int64)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm < 10.0)] = 4
    out[(distance_cm >= 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```

iii. CONVERSION_NOTES Step 5 mapping: "discretize to 7 bins per task spec". Class 3 is reserved for exactly 0, which is the "in the reward zone" state produced by `signed_distance_to_zone`. Step 9 lists the resulting distribution as a consistency check ("Derived from raw position + scene metadata").

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No extra alignment: position is sliced with the identical `[start:stop]` frame range as the neural matrix, and the per-trial zone interval is constant, so distance is defined frame-for-frame on the neural grid.

ii.
```python
        pos = behavior["position"][start:stop].astype(np.float32, copy=False)
        ...
        distance = signed_distance_to_zone(pos, zone_start, zone_end)
        ...
        neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 4: behavior and neural are one frame-aligned grid; Step 10 confirms with `np.allclose()` that converted `distance_to_reward_zone` equals independently recomputed raw values for three named trials, including a multi-plane session.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The framewise `position` behavioral time series (cm along the 450 cm virtual corridor), used raw.

ii.
```python
        pos = behavior["position"][start:stop].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 5 mapping: "`position` -> `output[1]` (`absolute_position`); Discretize 0-450 cm into 5 equal 90 cm bins; Methods: 450 cm track."

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice, the `float32` cast, and discretisation. No smoothing, no unwrapping, no rescaling. Because the trial window ends at teleport, the negative-position teleport jitter region is excluded.

ii.
```python
        output_trial = np.vstack([
                discretize_distance(distance),
                discretize_absolute_position(pos),
                ...
        ])
```

iii. CONVERSION_NOTES Step 5, decision 3: slicing "from `trial_start` to `teleport` ... excludes the variable-length tunnel/teleport period from the standard on-track trial representation", so raw position values can be used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins with open ends, assigned by boolean masks: 0 = `< 90` (default, also absorbing any slightly negative sample); 1 = `[90, 180)`; 2 = `[180, 270)`; 3 = `[270, 360]`; 4 = `> 360`. Realised distribution [0.212, 0.177, 0.231, 0.226, 0.154].

ii.
```python
def discretize_absolute_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int64)
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm <= 360.0)] = 3
    out[position_cm > 360.0] = 4
    return out
```

iii. The task spec's five equal bins over the 450 cm track (Step 3: "450 cm linear track"), with open first/last bins so occasional samples marginally outside [0, 450] land in the end classes rather than erroring. `output_values[1]` is labelled `["< 90 cm", "90 to 180 cm", "180 to 270 cm", "270 to 360 cm", "> 360 cm"]`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start:stop]` frame indices as the neural matrix; no resampling or offset.

ii.
```python
        pos = behavior["position"][start:stop]...
        neural_trial = neural[start:stop].T...
```

iii. As in 2-d/7-d: single shared frame clock after truncation to `common_len`, verified by the Step 10 `np.allclose()` spot checks.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The framewise `lick` behavioral time series (cumulative lick counts per frame, can exceed 1).

ii.
```python
        lick_raw = behavior["lick"][start:stop].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 5 mapping: "`lick` -> `output[3]` (`lick`); Apply lick QC, then binarize cumulative counts as `>0 -> 1`", referencing `glmUtils.get_timeseries_data` and the Methods' licking QC.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any frame with a lick count greater than 0 becomes 1, otherwise 0. In addition, whole trials are dropped upstream if more than 35% of their frames have counts > 2 (the capacitive-sensor artifact QC described in 1-e). Realised distribution [0.777 no, 0.223 yes].

ii.
```python
        bad_lick = np.mean(lick_raw > 2.0) > LICK_QC_FRACTION
        if bad_lick:
            dropped["lick_qc"] += 1
            continue
        ...
        lick_binary = (lick_raw > 0.0).astype(np.int64, copy=False)
```

iii. CONVERSION_NOTES Step 3 quotes the Methods on lick QC and binarisation ("Licks are converted to binary after quality control"); Step 4 explains the 0.35 threshold choice; Step 5, decision 7 explains dropping rather than masking artifact trials because lick is itself a decoder target.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop]` frame indices as the neural matrix, no shift or smoothing.

ii.
```python
        lick_raw = behavior["lick"][start:stop]...
        neural_trial = neural[start:stop].T...
```

iii. Single shared frame grid (Step 4); Step 10 confirms converted `lick` matches independently recomputed raw values with `np.allclose()` for three trials.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The session's scene string from the NWB `identifier` field, parsed into one or two zone letters, plus the within-session trial index to pick which letter applies (switch after 30 trials). The framewise `reward_zone` stream is deliberately not used. See 7-a for the parsing code.

ii.
```python
        scene = read_str(f["identifier"]).split("/")[-1]
...
    reward_labels = reward_labels_for_trials(session_raw["scene"], len(bounds))
```

iii. See 7-a. CONVERSION_NOTES Step 4: "NWB `identifier` suffixes exactly match code scene names for all data sessions checked"; Step 5, decision 6.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The letter is mapped `A -> 0`, `B -> 1`, `C -> 2` and tiled constant across all timepoints of the trial, giving a per-trial categorical output stored as a time series. Realised distribution [A 0.332, B 0.336, C 0.332].

ii.
```python
                np.full(pos.shape, {"A": 0, "B": 1, "C": 2}[zone_label], dtype=np.int64),
```
```python
        "output_values": [..., ["A", "B", "C"], ...]
```

iii. CONVERSION_NOTES Step 5 mapping: "Map `A/B/C -> 0/1/2`, repeat across timepoints ... Per-trial categorical output repeated across timepoints for consistent array shapes" (decision 5). Step 9 checks the near-uniform A/B/C split against the paper's balanced-by-design zone assignment.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` event time series — specifically its `timestamps`, mapped onto behavior frame indices. The `Reward` `data` (reward amounts) are read but not used for the outcome. See 6-a for the mapping code.

ii.
```python
        reward_timestamps = f[f"{BEHAVIOR_TS_PATH}/Reward/timestamps"][:]
        reward_valid = reward_timestamps <= behavior["timestamps"][-1]
        reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps[reward_valid], side="left")
        reward_frame_idx = np.clip(reward_frame_idx, 0, common_len - 1)
```

iii. CONVERSION_NOTES Step 4, "Reward representation"; Step 5 mapping: "Sparse `Reward` event times within trial -> `output[5]` (`reward_outcome`)".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, 1 if any mapped reward frame index lies in `[start, stop)`, else 0; the value is tiled constant across the trial's timepoints. Realised distribution [0.158 no, 0.842 yes], i.e. ~15.8% omissions.

ii.
```python
def trial_reward_outcomes(bounds, reward_frame_idx):
    outcomes = np.zeros(len(bounds), dtype=np.int64)
    reward_trials = []
    for i, (start, stop) in enumerate(bounds):
        has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
        outcomes[i] = int(has_reward)
        reward_trials.append(np.flatnonzero((reward_frame_idx >= start) & (reward_frame_idx < stop)))
    return outcomes, reward_trials
```
```python
                np.full(pos.shape, reward_outcome, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping: "`1` if any reward event falls in trial, else `0`; repeat across timepoints", citing the Methods' omission definition. Step 4 and Step 9 both check the resulting omission fraction against the paper's "Reward was randomly omitted on approximately 15% of trials"; Step 12 additionally re-verified three individual trials' reward outcomes directly against raw NWB timestamps.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles six kinds of data irregularity:
- **Neural/behavior length mismatch** (10 multi-plane sessions have ophys one sample longer): all neural planes and all behavior streams are truncated to `common_len = min(neural_len, behavior_len)`; the affected files are recorded in `metadata['sessions_trimmed_to_behavior_length']` and the per-session trim count in `session_info['trimmed_timepoints']`.
- **Multi-plane ROI indexing**: each `Deconvolved` series is masked through its own `rois` table region rather than assuming a 1:1 map to the full `PlaneSegmentation` table.
- **Reward events past the end of the behavior clock**: filtered out, and the remaining indices clipped into range.
- **Trailing/leading `trial number` tunnel fragments** (e.g. the 81st label in `sub-m11_ses-03`): ignored automatically because trials are defined only by `trial_start`/`teleport` pairs.
- **Invalid (negative) `environment` samples**: excluded before taking the modal environment value; an explicit error is raised if a trial has none valid.
- **Degenerate trials/sessions**: empty trials are skipped, and a session with fewer than 2 surviving trials is skipped with a message.

There is no minimum-trial-length filter and no assertion on the reward-timestamp-to-frame alignment error.

ii.
```python
        neural_len = min(neural_lens)
        behavior_len = min(len(v) for v in behavior.values())
        common_len = min(neural_len, behavior_len)
        ...
        for key in behavior:
            behavior[key] = behavior[key][:common_len]

        reward_valid = reward_timestamps <= behavior["timestamps"][-1]
        reward_timestamps = reward_timestamps[reward_valid]
        reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps, side="left")
        reward_frame_idx = np.clip(reward_frame_idx, 0, common_len - 1)
```
```python
        if pos.size == 0:
            dropped["empty"] += 1
            continue
```
```python
        if len(neural_trials) < 2:
            print(f"Skipping {path.name}: only {len(neural_trials)} valid trials after filtering.")
            continue
```

iii. CONVERSION_NOTES Step 2 enumerates the irregularities found during exploration (one-sample ophys overhang in 10 sessions; the `sub-m11_ses-03` extra trial label; the misleading 31 Hz multi-plane `rate`). Step 10 "Issues Found and Resolved" documents each fix, including the loader bug for multi-plane ROI table regions: "Updated loader to iterate across all deconvolved plane series, apply the `iscell` mask within each series' ROI table region, and concatenate curated neurons across planes."

## 13-a. What are the most time-consuming steps of the code?

i. The AI identified HDF5 I/O — reading the large `Deconvolved` matrices and behavior streams — as the dominant cost, and instrumented the script with `time.perf_counter()` per session and in total. Measured: 2.30 s for the 2-session sample, 53.79 s for all 152 sessions. Not accounted for in that number is the final `pickle.dump` of the 9.1 GB output, which in practice is a large share of end-to-end wall time, and is not separately timed.

ii.
```python
    total_start = time.perf_counter()
    for session_i, path in enumerate(files):
        session_start = time.perf_counter()
        session_raw = load_session_raw(path)
        ...
        elapsed = time.perf_counter() - session_start
        print(f"[{session_i + 1:03d}/{len(files):03d}] {path.name}: ... {elapsed:.2f}s", flush=True)
    total_elapsed = time.perf_counter() - total_start
```

iii. CONVERSION_NOTES Step 6/Step 7: "Loader uses direct `h5py` access instead of `pynwb` for speed and lower overhead"; "Use raw HDF5 slicing and load only the curated deconvolved traces actually used"; "Avoid expensive per-trial fancy indexing into HDF5 by loading each session's curated neural matrix once, then slicing in-memory." Step 7 estimated the full run at ~466 s by scaling the sample cost by the total `(timepoints x curated neurons)` workload ratio; the actual run was ~9x faster than that estimate.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
- `build_trial_bounds` — a Python loop over trial starts with an inner `while` over teleport indices. The whole pairing could be done with a single `np.searchsorted(teleports, starts, side='right')`.
- `trial_reward_outcomes` — a Python loop over trials that builds two full boolean masks over all reward events per trial (O(n_trials x n_rewards)); `np.searchsorted` on the sorted reward frame indices would give the same answer in one vectorised pass, and the mask is computed twice per trial.
- The main per-trial loop in `make_trial_arrays` — discretisation of distance, position, speed and lick could be computed once over the whole session array and then sliced, instead of per trial.

The per-frame work inside the loop (distance, all four discretisers) is already fully vectorised over the trial, which is why the loop is cheap in practice.

ii.
```python
    for start in starts:
        while tele_ptr < len(teleports) and teleports[tele_ptr] <= start:
            tele_ptr += 1
```
```python
    for i, (start, stop) in enumerate(bounds):
        has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
        outcomes[i] = int(has_reward)
        reward_trials.append(np.flatnonzero((reward_frame_idx >= start) & (reward_frame_idx < stop)))
```

iii. CONVERSION_NOTES Step 6 lists only the I/O-side speedups; the AI did not enumerate the remaining loops, judging total runtime (53.79 s) well inside the instruction's 15-minute budget, so no further optimisation was pursued.

## 13-c. What processing does the code repeat multiple times?

i. Very little. The design is strictly single-pass: each NWB file is opened exactly once, its arrays read once, and everything (trial bounds, reward outcomes, zone labels, inputs, outputs) derived from that single in-memory copy. There is no separate survey/statistics pass over the dataset.

The small repetitions that do exist: `trial_reward_outcomes` builds the same boolean mask twice per trial (once for `np.any`, once for `np.flatnonzero`); `pick_processing_files` is called both via `pick_sample_files` and again in `convert_dataset`; and `dt_behavior` is recomputed per session and then reduced to a single median.

ii.
```python
        has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
        outcomes[i] = int(has_reward)
        reward_trials.append(np.flatnonzero((reward_frame_idx >= start) & (reward_frame_idx < stop)))
```
```python
def pick_sample_files(all_files):
    return pick_processing_files(all_files)
...
    processing_files = {p.resolve() for p in pick_processing_files(files)} if show_processing else set()
```

iii. CONVERSION_NOTES Step 6: "Process sessions sequentially and release each session before moving to the next"; "Loading all sessions into memory at once would be unnecessary." Because reward-zone identity comes from the scene string rather than from a data-driven segmentation, no pre-pass over the dataset is needed at all.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI did not document any, but the code does contain several pieces of discarded work:
- **`trial_debug` is built unconditionally.** For every trial of every session a dictionary is appended holding the distance, position, speed, raw lick and time arrays plus metadata, even though it is consumed only by `plot_processing_summary`, which runs for at most 2 sessions under `--show-processing`. It keeps every trial's behavior arrays alive for the whole session.
- **`reward_event_groups`** (a `np.flatnonzero` per trial) is computed for all 12,216 trials purely to fill `reward_event_count` in `trial_debug`.
- **Four behavior streams are read in full and never used**: `reward_zone` (superseded by the scene-based labels), `scanning`, `autoreward`, and the `Reward` `data` amounts (`reward_amounts` is filtered and returned from the loader, then never read).
- **`session_info` bookkeeping** such as `n_unique_trial_labels_nonnegative` and `plane_counts_curated` is recomputed per session for metadata only.

None of this affects the converted values; it is wasted I/O and memory.

ii.
```python
        trial_debug.append(
            {
                "raw_trial_idx": raw_trial_idx,
                ...
                "distance_cm": distance,
                "position_cm": pos,
                "speed_cm_s": speed,
                "lick_raw": lick_raw,
                "time_from_start_sec": time_from_start,
                "reward_event_count": int(len(reward_event_groups[raw_trial_idx])),
            }
        )
```
```python
        for name in [
            "environment", "position", "speed", "lick",
            "reward_zone", "scanning", "trial number", "trial_start",
            "teleport", "autoreward",
        ]:
            behavior[name] = f[f"{BEHAVIOR_TS_PATH}/{name}/data"][:]
        ...
        reward_amounts = f[f"{BEHAVIOR_TS_PATH}/Reward/data"][:]
```

iii. CONVERSION_NOTES Step 6 claims the opposite ("Use raw HDF5 slicing and load only the curated deconvolved traces actually used"; "Loading all sessions into memory at once would be unnecessary"), and Step 12 concludes "I did not identify any Step 12 issue that justified changing `convert_data.py`." The unused streams were presumably loaded during exploration (Step 2 documents checking `scanning` and `autoreward`) and never pruned once the scene-based reward-zone decision made `reward_zone` redundant.
