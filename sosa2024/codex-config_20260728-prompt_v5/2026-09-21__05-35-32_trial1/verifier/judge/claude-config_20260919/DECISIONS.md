# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by globbing `/app/data/sub-*/sub-*_behavior+ophys.nwb` and sorting the paths; the sorted file order defines the session order in the exported dictionary. 152 files are found (11 subject directories). Files are opened **directly with `h5py`** rather than with `pynwb`, and the needed groups are read by explicit HDF5 path: `general/subject/subject_id`, `general/session_id`, `processing/behavior/BehavioralTimeSeries/*` (`position`, `environment`, `lick`, `speed`, `trial number`, `trial_start`, `teleport`, `scanning`, `Reward`), and `processing/ophys/{ImageSegmentation/PlaneSegmentation, Deconvolved/plane*}`. In addition, the reference repository's `sessions_dict.py` is imported at runtime with `importlib` to recover each session's experiment-day `scene` string. Each session is loaded, trialized and released before the next file is opened (bounded memory), and everything is concatenated in `build_dataset()`.

ii.
```python
DATA_ROOT = Path("/app/data")
SESSIONS_DICT_PATH = Path("/app/code/src/reward_relative/sessions_dict.py")
...
all_files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
if not all_files:
    raise RuntimeError(f"No NWB files found under {DATA_ROOT}")
selected_files = all_files if args.mode == "full" else select_sample_files(all_files)
sessions_dict = load_sessions_dict()
...
for file_idx, nwb_path in enumerate(selected_files, start=1):
    converted, summary = convert_session(nwb_path, sessions_dict, do_plot)
```
```python
def load_sessions_dict() -> dict[tuple[str, int], SessionMeta]:
    spec = importlib.util.spec_from_file_location("sessions_dict_local", SESSIONS_DICT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapping = {}
    for collection_name in ("single_plane", "multi_plane"):
        collection = getattr(module, collection_name)
        for animal, entries in collection.items():
            for entry in entries:
                meta = SessionMeta(date=entry["date"], scene=entry["scene"], session=int(entry["session"]),
                                   scan=entry["scan"], exp_day=int(entry["exp_day"]))
                mapping[(animal, meta.exp_day)] = meta
    return mapping
```
```python
with h5py.File(nwb_path, "r") as h5:
    subject_id = h5["general/subject/subject_id"][()].decode()
    exp_day = int(h5["general/session_id"][()].decode())
    session_meta = sessions_dict[(subject_to_gcamp(subject_id), exp_day)]
    beh_root = h5["processing/behavior/BehavioralTimeSeries"]
    frame_timestamps = beh_root["position"]["timestamps"][()].astype(np.float64)
    position = beh_root["position"]["data"][()].astype(np.float32)
    ...
```

iii. CONVERSION_NOTES Step 2/Step 4: "`/app/data` contains a DANDI-style NWB release rather than the paper repo's original pickle hierarchy … Treat NWB as the authoritative serialized form of the same aligned content. Map NWB `processing/behavior/BehavioralTimeSeries` to `sess.vr_data`-like variables and NWB `processing/ophys/...` to `sess.timeseries`." The counts were checked against the paper: 11 mice, 152 sessions (10 × 14 days + m11 × 12 days, m11 starting on day 3), which the notes call "Fully consistent". Direct `h5py` access was chosen for speed and to permit column-subset reads of the large ROI matrices; `sessions_dict.py` is imported because the A/B/C reward-zone label is a property of the scene name, not of any framewise NWB variable.

## 1-b. How are the data split into subjects (mice)?

i. The subject of a session is read from the NWB metadata field `general/subject/subject_id` (e.g. `m11`), not from the directory name. The unique set of those ids is sorted to form `subjects` (11 entries), and `subject_idx` holds the index of each session's subject in file order. The NWB id is also mapped to the reference repo's naming convention (`m11 → GCAMP11`) in order to look up the session metadata.

ii.
```python
subject_id = h5["general/subject/subject_id"][()].decode()
...
def subject_to_gcamp(subject_id: str) -> str:
    if subject_id.startswith("m"):
        return f"GCAMP{subject_id[1:]}"
    raise ValueError(f"Unexpected subject id: {subject_id}")
```
```python
subjects = sorted({sess["subject_id"] for sess in converted_sessions})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
data["subjects"] = subjects
data["subject_idx"] = np.asarray([subject_lookup[sess["subject_id"]] for sess in converted_sessions], dtype=np.int16)
```

iii. Key Decision 10: "Encode subjects using NWB IDs (`m3`, `m4`, …): these are the canonical identifiers present in the shared data files and are unambiguous." The Step 4 consistency table confirms the 11 mice in the release are the paper's 11 switch-task mice and that the extra animals in `dayData.define_anim_list` (GCAMP2, 6, 10, fixed-condition cohort) are simply absent from the NWB release.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The session's experiment day comes from `general/session_id` and is used as the key into the reference `sessions_dict` (together with the animal) to obtain the `scene`. All 152 files are kept — no session-level curation is applied (the notes explicitly reject restricting to switch days). Session ordering is the sorted file path order; sessions from the same mouse are therefore contiguous and in day order.

ii.
```python
exp_day = int(h5["general/session_id"][()].decode())
session_meta = sessions_dict[(gcamp, exp_day)]
session_tag = nwb_path.stem.replace("sub-", "").replace("_behavior+ophys", "")
```
```python
data = {"neural": [sess["neural_trials"] for sess in converted_sessions], ...}
```

iii. Key Decision 4: "Keep all complete imaging sessions/days in the export: although several manuscript figures focus on switch days, the user requested the full dataset and the decoder task is well-defined on all sessions." Step 4: "152 NWB files: 10 mice × 14 sessions + m11 × 12 sessions … Fully consistent. Use all 152 NWB files as candidate sessions."

## 1-d. How are the data split into trials?

i. A trial runs from a `trial_start` pulse to the next `teleport` pulse, with the teleport frame itself excluded (`slice(start, end)`). Both events are additionally required to occur while `scanning > 0`. Starts and ends are paired greedily and one-to-one, so unmatched/duplicated pulses cannot create spurious trials, and a trailing `trial_start` with no following `teleport` is dropped. This yields 12,216 complete trial epochs over the release (mean 80.37/session), before quality control.

ii.
```python
def pair_trial_events(trial_start, teleport, scanning):
    start_idx = np.where((trial_start > 0) & (scanning > 0))[0]
    end_idx = np.where((teleport > 0) & (scanning > 0))[0]
    pairs = []
    end_ptr = 0
    for start in start_idx:
        while end_ptr < len(end_idx) and end_idx[end_ptr] <= start:
            end_ptr += 1
        if end_ptr >= len(end_idx):
            break
        end = end_idx[end_ptr]
        if end > start:
            pairs.append((int(start), int(end)))
        end_ptr += 1
    return pairs
...
trial_slice = slice(start, end)  # exclude teleport frame itself
```

iii. Key Decision 1: "Trial boundaries are `trial_start` through the frame immediately before `teleport`: this matches the reference code's use of explicit start/end indices and excludes negative-position tunnel frames plus the corrupted teleport frame." Step 10 Check 3(c) maps this to `glmUtils.get_timeseries_data`, which fills arrays only inside `trial_start_inds`→`teleport_inds` windows. Step 10 Check 5 gives the boundary spot-check for `m11_ses-03` trial 0: position just before `trial_start` = −1.12 cm, first included sample = +0.61 cm, last included = 448.85 cm, excluded teleport frame = 200.79 cm.

## 1-e. How are trials filtered based on quality controls?

i. One trial-level filter is applied: the paper's lick-sensor-fault rule. A trial is discarded if more than 30% of its imaging frames have a cumulative lick count > 2. Zero-length trials are also marked bad. Across the full dataset this removes exactly **81** of 12,216 trials (0.663%), leaving 12,135. Trials are dropped entirely (neural data included) rather than having their lick target set to NaN. Sessions left with fewer than 2 trials would raise an error (none did). No minimum-trial-length filter is applied.

ii.
```python
LICK_ERROR_THRESHOLD = 0.30

def detect_bad_lick_trials(lick, trial_pairs):
    bad = np.zeros(len(trial_pairs), dtype=bool)
    for i, (start, end) in enumerate(trial_pairs):
        trial_lick = lick[start:end]
        if trial_lick.size == 0:
            bad[i] = True
            continue
        bad[i] = np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD
    return bad
...
for trial_idx, ((start, end), bad_lick, reward_label) in enumerate(zip(trial_pairs, bad_lick_trials, reward_labels_all, strict=True)):
    if bad_lick:
        continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"{session_tag}: fewer than 2 valid trials after filtering")
```

iii. Step 3 records the Methods rule ("bad lick-sensor trials removed when >30% of imaging frames in a trial have cumulative lick count >2") and the paper's statistic ("~0.65% of all imaged trials, n = 81 out of 12,376 trials removed"). Step 6: "Trials with lick-sensor faults are removed entirely rather than storing invalid lick targets with NaNs." Step 9 reports the match: "Lick-QC removal removed exactly `81` trials … `81 / 12,216 = 0.663%`, closely matching the manuscript's reported `~0.65%`." Step 10 notes the threshold differs from the helper default `correction_thr=0.5` in `behavior.correct_lick_sensor_error` but "matches the methods text".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is taken **directly from the stored NWB `processing/ophys/Deconvolved/plane*/data`** arrays (suite2p's deconvolved output), restricted to ROIs with `iscell[:,0] > 0.5` and pooled across imaging planes using the ROI table's `planeIdx`. The raw `Fluorescence` (F) and `Neuropil` (Fneu) arrays are read only in ad-hoc checks, never in the conversion.

ii.
```python
seg = h5["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = seg["iscell"][()]
plane_idx = seg["planeIdx"][()].astype(np.int16)
keep_mask = iscell[:, 0] > 0.5
keep_indices = np.flatnonzero(keep_mask)
deconv_root = h5["processing/ophys/Deconvolved"]
...
deconv = np.empty((min_frames, keep_indices.size), dtype=np.float16)
local_index = np.empty(plane_idx.shape[0], dtype=np.int32)
for plane in unique_planes:
    plane_mask = plane_idx == plane
    local_index[plane_mask] = np.arange(np.sum(plane_mask), dtype=np.int32)
kept_plane_idx = plane_idx[keep_indices]
for plane in np.unique(kept_plane_idx):
    kept_positions = np.where(kept_plane_idx == plane)[0]
    local_cols = local_index[keep_indices[kept_positions]]
    plane_data = deconv_root[f"plane{int(plane)}"]["data"][:min_frames, local_cols]
    deconv[:, kept_positions] = plane_data.astype(np.float16)
```

iii. Step 4 discrepancy table: "Code computes `dff` from fluorescence, then deconvolves to `events`; many analyses use `events` / NWB already stores `Fluorescence`, `Neuropil`, and `Deconvolved` / Methods specify trialwise maximin dF/F and OASIS deconvolution → **Prefer NWB `Deconvolved` as the neural signal for export, because it matches the paper's event-like activity and avoids reimplementing dF/F from scratch unless needed for a validation check.**" Key Decision 3: "Use deconvolved activity rather than recomputing dF/F for the main export: this most directly matches the manuscript's RR decoder and place-cell significance computations."

## 2-b. How is the `neural` data processed?

i. Essentially no signal processing is performed. The stored deconvolved traces are (1) subset to `iscell` ROIs, (2) concatenated across planes in ROI-table order, (3) truncated to the common frame count, (4) sliced per trial and transposed to `(n_neurons, T)`, and (5) **cast to `float16`** to keep the pickle to ~4.8 GB. No neuropil subtraction (`F − 0.7·Fneu`), no per-trial maximin baseline over a 20 s window, no `(F − baseline)/|baseline|`, no 2-sample Gaussian smoothing, and no OASIS deconvolution at `tau = 0.7` are applied — i.e. none of the paper's `preprocessing.dff(..., deconvolve=True)` pipeline is reproduced. Consequently the exported values are on the raw-fluorescence scale (hundreds to thousands) rather than the paper's normalized dF/F event scale.

ii.
```python
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```
```python
"neural_signal": (
    "deconvolved calcium activity from NWB processing/ophys/Deconvolved/plane* "
    "datasets, pooled across imaging planes by ROI plane index"
),
```

iii. Step 6: "Neural signal currently uses NWB deconvolved activity directly, filtered to `iscell[:,0] == 1`. To keep the full exported pickle tractable, neural trial matrices are stored as `float16`." The rationale for not recomputing dF/F is the Step 4 resolution quoted in 2-a ("avoids reimplementing dF/F from scratch"), plus the Step 11 pickle-size concern ("~2.42 billion values (~9.67 GB float32, ~4.84 GB float16)"). The agent did record in Step 3 that the Methods prescribe "dF/F baseline computed within each trial independently with a 20 s sliding-window maximin baseline … smoothed with a 2-sample Gaussian kernel … deconvolution performed with OASIS", so the deviation was made knowingly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single neuron filter: suite2p's manual curation flag, `iscell[:,0] > 0.5`. This keeps 138,678 ROIs over 152 sessions (mean 912.4, range 155–2341). The paper's **second** curation step — excluding putative interneurons whose dF/F correlates with running speed at r > 0.5 — was investigated and deliberately **not** applied: the agent correlated the *deconvolved* traces (not dF/F) with speed in the worst-case session `m18_ses-03`, found 0/2341 cells above r = 0.5, and concluded the step could not explain the sessions exceeding the paper's quoted 2172-cell upper bound. No per-neuron activity/SNR filter is applied either.

ii.
```python
iscell = seg["iscell"][()]
keep_mask = iscell[:, 0] > 0.5
keep_indices = np.flatnonzero(keep_mask)
brain_region_idx = np.zeros(len(keep_indices), dtype=np.int16)
...
"roi_filter": "PlaneSegmentation iscell[:,0] == 1",
```

iii. Step 10 Check 4: "The paper text reports `155–2172` putative pyramidal neurons/session, whereas the NWB release contains three pooled multi-plane `m18` sessions above this upper bound (`2281`, `2314`, `2341`). I checked the worst case (`m18_ses-03`) with a direct deconvolved-trace versus speed correlation probe and found `0 / 2341` cells with `r > 0.5`, so the small post-hoc interneuron exclusion described in the paper cannot plausibly explain the NWB upper tail. I therefore retained all manually curated `iscell` ROIs and treat this as a release-versus-manuscript reporting discrepancy rather than a conversion bug." Step 3 had recorded the paper's rule and its expected effect size ("excluding 0.42 ± 0.85% of cells").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start. Because the neural and behavioral streams live on the same imaging-frame grid, alignment is achieved simply by slicing the frame axis from the `trial_start` frame (inclusive) to the `teleport` frame (exclusive); sample 0 of every trial is the `trial_start` frame. `metadata['temporal_alignment_event'] = "trial_start pulse / entry to linear track"`, `off_start = 0.0`, `off_end = None` (trials have variable length). No pre-event baseline window is included.

ii.
```python
trial_slice = slice(start, end)  # exclude teleport frame itself
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
...
"temporal_alignment_event": "trial_start pulse / entry to linear track",
"off_start": 0.0,
"off_end": None,
```

iii. Key Decision 2: "Use imaging-frame-aligned behavioral timestamps as the master clock." Step 4: "NWB behavior streams have the same frame count as neural traces and share imaging-frame timestamps … Treat one imaging frame as the canonical temporal grid for all streams. Reconstruct trials from `trial_start` and `teleport` pulses." The Step 7 plots were used to confirm "no temporal mismatch … between neural heatmaps and behavioral trajectories in the representative trials."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling, smoothing, or trial-time warping. Data stay at the native imaging-frame rate of 15.5078125 Hz, i.e. `time_bin_size = 64.483627 ms`, identical for every trial and session. The two-plane mice (m17, m18) are handled by pooling the two per-plane `Deconvolved` series, each of which is already stored at the per-plane ~15.5 Hz rate; the NWB `rate` attribute of 31.015625 Hz (the scanner rate) is deliberately ignored in favour of the behaviour timestamps. The code *asserts* per session that the median behavioural timestamp step equals 1/15.5078125 s, so any session with a different effective rate would abort the conversion.

ii.
```python
COMMON_FRAME_RATE_HZ = 15.5078125
COMMON_FRAME_DT_S = 1.0 / COMMON_FRAME_RATE_HZ
TIME_BIN_SIZE_MS = COMMON_FRAME_DT_S * 1000.0
...
frame_dt = median_step_seconds(frame_timestamps)
if not np.isclose(frame_dt, COMMON_FRAME_DT_S, atol=1e-6):
    raise RuntimeError(f"{session_tag}: unexpected frame dt {frame_dt}")
...
"time_bin_size": TIME_BIN_SIZE_MS,
"rate_attr_note": "Ignored NWB deconvolved rate attribute; used behavior timestamps as master clock.",
```

iii. Key Decision 2: "all sessions share the same effective frame interval (~64.4836 ms) in behavior timestamps, including m17/m18, whereas the 31.015625 Hz NWB rate metadata for two-plane files is inconsistent with the timestamped behavioral grid." Step 3 cites the Methods: "~0.0645 s imaging frame samples"; "m17 and m18 … ~31 Hz interleaved … ~15.5 Hz per plane". (Note: an earlier, superseded entry in the Step 4 table said the plan was to "downsample time by 2" for m17/m18; the final code does not, because the per-plane series are already at the per-plane rate.)

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the `timestamps` array of the `position` behavioural time series (all behaviour series share the same imaging-frame timestamps), sliced to the trial window.

ii.
```python
beh_root = h5["processing/behavior/BehavioralTimeSeries"]
frame_timestamps = beh_root["position"]["timestamps"][()].astype(np.float64)
...
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
```

iii. Step 5 mapping table: "`processing/behavior/BehavioralTimeSeries/position/timestamps` + `trial_start` → `input[0]` = time from start of trial; For each trial, subtract trial-start timestamp from per-frame timestamps; Imaging-frame-aligned continuous seconds." Step 2 records that "Behavioral time series use explicit timestamps at imaging-frame resolution".

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the trial's first timestamp, then a cast to `float32`. The first sample of every trial is therefore exactly 0.0 s and subsequent samples increment by ~0.0645 s. Range over the full export: [0.0, 216.5] s.

ii.
```python
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
...
input_trial = np.ascontiguousarray(
    np.vstack([
        t_trial.astype(np.float32),
        repeated_row(env_trial, T, np.float32),
        repeated_row(trial_num, T, np.float32),
        repeated_row(prev_reward, T, np.float32),
    ]), dtype=np.float32)
```

iii. Straightforward implementation of the requested "time from start of trial in seconds" with the trial-start alignment. Step 10 Check 2 verifies it: "for the same `m11_ses-03` and `m17_ses-01` trials, independently rebuilt `time_from_trial_start_s` … from the raw behavior streams. `np.allclose(...) == True`."

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No alignment operation is needed: the behaviour timestamps index the same imaging frames as the deconvolved matrix, so both are sliced with the identical `trial_slice`. Before trialization, all streams (behaviour arrays and every plane's neural matrix) are truncated to `min_frames`, the minimum common frame count, which fixes the occasional one-frame mismatch in two-plane files. A shape assertion then checks that neural, input and output have identical `T` per trial.

ii.
```python
plane_frame_counts = [int(deconv_root[f"plane{int(plane)}"]["data"].shape[0]) for plane in unique_planes]
min_frames = min([frame_timestamps.shape[0], *plane_frame_counts])
frame_timestamps = frame_timestamps[:min_frames]
position = position[:min_frames]
... (all behaviour streams truncated identically)
...
if neural_trial.shape[1] != input_trial.shape[1] or neural_trial.shape[1] != output_trial.shape[1]:
    raise RuntimeError(f"{session_tag}: time dimension mismatch in trial {trial_idx}")
```

iii. Step 4: "NWB behavior streams have the same frame count as neural traces and share imaging-frame timestamps … Treat one imaging frame as the canonical temporal grid for all streams." Step 10 Check 5: "Two-plane sessions with one-frame stream mismatches are safely truncated to the minimum common frame count before trialization."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The framewise `environment` behavioural time series (ENV1 = 0, ENV2 = 1; −1 marks pre-synchronization frames).

ii.
```python
environment = beh_root["environment"]["data"][()].astype(np.float32)
...
env_trial_vals = environment[trial_slice]
```

iii. Step 5 mapping table: "`.../environment/data` → `input[1]` = environment type … Encode ENV1=0, ENV2=1", tied to the reference `behavior.env_morph_dict` / trialwise `morph` in `multi_anim_sess_README.md`. Step 2 records the observed convention: "`environment` values are `-1` before synchronization/scanning, then `0` or `1`".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within the trial window, invalid (negative) samples are dropped, the median of the remaining values is taken and rounded to an integer, and that single value is repeated across all `T` timepoints. If no valid sample exists the conversion aborts. Range over the full export is [0, 1], as expected.

ii.
```python
env_trial_vals = environment[trial_slice]
env_trial_vals = env_trial_vals[env_trial_vals >= 0]
if env_trial_vals.size == 0:
    raise RuntimeError(f"{session_tag}: no valid environment values in trial {trial_idx}")
env_trial = int(np.rint(np.median(env_trial_vals)))
...
repeated_row(env_trial, T, np.float32),
```

iii. Key Decision 6: "Represent all decoder inputs and outputs as time-varying matrices `(d, T)`: per-trial variables will be repeated across trial timepoints so the exported structure is uniform." The median-of-valid-samples rule is a robustness measure against the `-1` pre-sync sentinel documented in Step 2; environment is constant within a trial, so the median simply recovers that constant.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The stored `trial number` behavioural time series, sampled at the trial's first frame. (This is equivalent to the 0-based index of the trial within the session — I verified `trial_number[start] == pair index` for all 3,180 trials in the first 40 sessions — but using the stored value means that when a trial is removed by lick QC the remaining trials keep their original session-wide numbering rather than being re-indexed.)

ii.
```python
trial_number = beh_root["trial number"]["data"][()].astype(np.int32)
...
trial_num = int(trial_number[start])
...
repeated_row(trial_num, T, np.float32),
```

iii. Step 5 mapping table: "`.../trial number/data` → `input[2]` = trial number; Take session-local complete-trial index / trial number, repeat across `T` … Use integer-like float values, starting at 0 within session." Step 2 documents the sentinel convention ("`trial number` is `-1` before synchronization and otherwise nonnegative integers") and the one anomaly ("`sub-m11_ses-03` has 81 unique nonnegative `trial number` values but only 80 `trial_start` and 80 `teleport` pulses"), which is handled by only emitting trials that have a paired start/teleport.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond reading the value at the trial-start frame and tiling it across the trial's `T` timepoints as `float32`. Observed range over the export: [0, 99], matching the 80–100 trials/session design.

ii.
```python
trial_num = int(trial_number[start])
input_trial = np.vstack([..., repeated_row(trial_num, T, np.float32), ...])
```

iii. Same as 5-a; Step 9 checks the range against the paper ("80–100 trial sessions, switch after trial 30 … [0, 99] … Yes").

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the sparse `Reward` behavioural time series — specifically its event `timestamps` — combined with the trial boundary timestamps. A per-trial rewarded/omitted flag is computed for **every** paired trial in the session (before lick QC), and the previous entry of that array is used as the input.

ii.
```python
reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)
...
reward_outcomes_all = np.zeros(len(trial_pairs), dtype=np.int16)
for i, (start, end) in enumerate(trial_pairs):
    start_ts = frame_timestamps[start]
    end_ts = frame_timestamps[end]
    reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
```

iii. Step 5 mapping table: "Reward timestamps / previous trial reward outcome → `input[3]` = previous trial outcome; Compute reward outcome per complete trial, shift by one trial within session, fill first trial with 0 … Binary: omitted/no reward = 0, rewarded = 1", referencing `behavior.get_trial_types` and `rewardAnalysis.get_reward_inds`. Step 2 notes `Reward` is stored as sparse timestamped events rather than a framewise array, hence the timestamp-interval test.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *i* > 0 the value is `reward_outcomes_all[i-1]`; for the first trial of a session it is 0. The lookup uses the index into the *complete* (pre-QC) trial list, so if trial *i*−1 was dropped for a lick-sensor fault, trial *i* still reports that dropped trial's true outcome rather than skipping to an earlier trial. The scalar is tiled over `T`.

ii.
```python
prev_reward = int(reward_outcomes_all[trial_idx - 1]) if trial_idx > 0 else 0
...
repeated_row(prev_reward, T, np.float32),
```

iii. The instructions define previous-trial outcome as binary (omitted = 0, rewarded = 1); the first trial has no predecessor and is assigned the "omitted" code. Step 9 confirms the exported range is [0, 1].

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From two sources: (1) the framewise `position` time series, and (2) the active reward-zone interval for that trial, which is **not** read from the framewise `reward_zone` stream but reconstructed from the session's `scene` string in the reference repo's `sessions_dict.py`, combined with the paper's fixed zone coordinates (A: 80–130, B: 200–250, C: 320–370 cm) and the rule that on switch sessions the zone changes after trial 30.

ii.
```python
REWARD_ZONE_COORDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}

def scene_reward_labels(scene, n_trials, switch_trial=30):
    if scene in {"Env1_LocationA", "Env2_LocationA", "Env3_LocationA"}:
        return ["A"] * n_trials
    ...
    transitions = [("A","B"),("B","A"),("A","C"),("C","A"),("B","C"),("C","B")]
    for src, dst in transitions:
        if f"{src}_to" in scene and scene.endswith(dst):
            return [src] * min(switch_trial, n_trials) + [dst] * max(0, n_trials - switch_trial)
    raise ValueError(f"Unsupported scene for reward-zone mapping: {scene}")
...
reward_labels_all = scene_reward_labels(session_meta.scene, len(trial_pairs))
zone_start, zone_end = REWARD_ZONE_COORDS[reward_label]
```

iii. Key Decision 7: "Compute reward-zone identity from reference session metadata rather than from framewise `reward_zone` counts alone: the paper/code define zone A/B/C from scene identity and switch timing; the framewise `reward_zone` signal indicates occupancy, not location label." Step 3 sources the coordinates and the switch rule from the Methods ("zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm"; "Each switch occurred after 30 trials"), and Step 1 sources the scene→zone mapping from `behavior.get_reward_zones`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the active zone interval: `position − zone_start` when before the zone, `position − zone_end` when past it, and exactly 0 while inside. Computed per frame on the trial's position slice, then discretized (7-c). No speed masking, smoothing, or circularization is applied.

ii.
```python
def reward_distance_to_zone(position_cm, zone_start, zone_end):
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist = np.zeros(position_cm.shape, dtype=np.float32)
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
    return dist
...
reward_dist = reward_distance_to_zone(pos_trial, zone_start, zone_end)
reward_dist_bin = discretize_reward_distance(reward_dist)
```

iii. Key Decision 8: "Use nearest-distance-to-interval for reward-zone distance: this satisfies the user's 'distance to any location in the reward zone' requirement while remaining faithful to track geometry." Step 5 mapping table: "Signed distance to nearest point in active reward-zone interval: `<start -> pos-start`, `inside -> 0`, `>end -> pos-end`".

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned with explicit boolean masks, exactly reproducing the bin table in the instructions: 0 = `< −50`; 1 = `[−50, −10)`; 2 = `[−10, 0)`; 3 = exactly 0 (inside the zone); 4 = `(0, 10]`; 5 = `(10, 50]`; 6 = `> 50` (the default fill value). Resulting class fractions over the export: [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243].

ii.
```python
def discretize_reward_distance(distance_cm):
    out = np.full(distance_cm.shape, 6, dtype=np.int16)
    out[distance_cm < -50] = 0
    out[(distance_cm >= -50) & (distance_cm < -10)] = 1
    out[(distance_cm >= -10) & (distance_cm < 0)] = 2
    out[distance_cm == 0] = 3
    out[(distance_cm > 0) & (distance_cm <= 10)] = 4
    out[(distance_cm > 10) & (distance_cm <= 50)] = 5
    return out
```

iii. Directly transcribed from the Decoder Outputs specification; the "0 cm" class is given its own exact-equality test so that in-zone samples form their own category. Step 7 review states that the `--show-processing` plots "confirm … that reward-distance bins transition as expected from pre-zone to in-zone to post-zone."

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[trial_slice]`, the same frame slice used for the neural matrix, so alignment is automatic; the per-trial shape assertion enforces equal `T`.

ii.
```python
trial_slice = slice(start, end)
pos_trial = position[trial_slice]
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
...
if neural_trial.shape[1] != output_trial.shape[1]:
    raise RuntimeError(f"{session_tag}: time dimension mismatch in trial {trial_idx}")
```

iii. Same as 3-c: behaviour and ophys share one imaging-frame grid after truncation to `min_frames`; Step 10 Check 2 verified the reconstructed output bins against raw NWB streams with `np.allclose`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The framewise `position` behavioural time series (cm along the 450 cm virtual corridor).

ii.
```python
position = beh_root["position"]["data"][()].astype(np.float32)
...
pos_trial = position[trial_slice]
```

iii. Step 5 mapping table: "Trial position → `output[1]` = absolute corridor position; Discretize cm position into 5 bins spanning 0–450 cm". Step 2 records "`position` is `-500` before synchronization and otherwise spans the virtual track up to about `450 cm`", and the trial-window rule (1-d) excludes the negative-position inter-trial frames.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial window and discretizing; raw cm values are used as stored (no offsetting, unwrapping or speed masking). Class fractions over the export: [0.212, 0.177, 0.231, 0.226, 0.154].

ii.
```python
pos_trial = position[trial_slice]
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The instruction is to discretize absolute corridor position into 5 equal bins spanning the 450 cm track; the raw stream is already in those units. Step 7's processing plots were used to confirm trials start near 0 cm and end near 450 cm before the teleport.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins with open ends: 0 = `< 90` (the default fill, so it also absorbs any slightly negative sample), 1 = `[90, 180)`, 2 = `[180, 270)`, 3 = `[270, 360]`, 4 = `> 360`.

ii.
```python
def discretize_absolute_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int16)
    out[(position_cm >= 90) & (position_cm < 180)] = 1
    out[(position_cm >= 180) & (position_cm < 270)] = 2
    out[(position_cm >= 270) & (position_cm <= 360)] = 3
    out[position_cm > 360] = 4
    return out
```

iii. Transcribed from the Decoder Outputs specification (5 equal bins over 450 cm = 90 cm each), with open first/last bins so that samples marginally outside [0, 450] still fall in an end class rather than being invalid.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical frame slice as the neural matrix (`position[trial_slice]`); no offset or interpolation.

ii.
```python
pos_trial = position[trial_slice]
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. Same as 3-c/7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The framewise `lick` behavioural time series (cumulative lick count per imaging frame; values can exceed 1).

ii.
```python
lick = beh_root["lick"]["data"][()].astype(np.float32)
...
lick_bin = (lick[trial_slice] > 0).astype(np.int16)
```

iii. Step 1: "`sess.timeseries['licks']` is cumulative per frame and later binarized (`licks[licks > 1] = 1`) for continuous analyses." Step 4: "NWB `lick` values are cumulative-per-frame and can exceed 1 → Match manuscript logic by binarizing licks after detecting and masking faulty trials."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps: (1) the lick-sensor-fault QC from 1-e removes whole trials where >30% of frames have cumulative count > 2; (2) in the surviving trials, any frame with count > 0 is coded 1, otherwise 0. Exported class fractions: [0.777 no, 0.223 yes].

ii.
```python
bad_lick_trials = detect_bad_lick_trials(lick, trial_pairs)
...
if bad_lick:
    continue
...
lick_bin = (lick[trial_slice] > 0).astype(np.int16)
```

iii. The instructions require a binary lick output; the paper's QC ordering (detect faulty trials on the raw cumulative counts, then binarize) is preserved. Step 6: "Trials with lick-sensor faults are removed entirely rather than storing invalid lick targets with NaNs."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame slice as the neural matrix; no shifting or event-window expansion.

ii.
```python
lick_bin = (lick[trial_slice] > 0).astype(np.int16)
```

iii. Same as 3-c.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Not from a raw NWB variable at all: it is derived from session metadata — the NWB `general/subject/subject_id` and `general/session_id` are used to look up the `scene` string in the reference repo's `sessions_dict.py`, which encodes the zone (or the pre→post-switch zone pair) for that experiment day. The framewise `reward_zone` occupancy stream is *not* read by the converter.

ii.
```python
subject_id = h5["general/subject/subject_id"][()].decode()
exp_day = int(h5["general/session_id"][()].decode())
session_meta = sessions_dict[(subject_to_gcamp(subject_id), exp_day)]
...
reward_labels_all = scene_reward_labels(session_meta.scene, len(trial_pairs))
...
RZ_CODE = {"A": 0, "B": 1, "C": 2}
repeated_row(RZ_CODE[reward_label], T, np.int16),
```

iii. Key Decision 7 (quoted in 7-a): the framewise `reward_zone` signal "indicates occupancy, not location label", whereas the paper's own `behavior.get_reward_zones` assigns A/B/C from the scene name plus `change_reward_trial` (default 30). Step 9 reports the resulting balance, `[0.332, 0.336, 0.333]`, "as expected from the experimental design".

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. `scene_reward_labels()` maps the scene string to a per-trial label list: constant `A`/`B`/`C` for fixed-zone scenes; for switch scenes (both within-environment `Env1_LocationA_to_B` style and across-environment `Env1_A_to_Env2_C` style) the source label is assigned to trials 0–29 and the destination label to trials 30+, using a hard-coded `switch_trial = 30`. Unrecognized scenes raise. The label is encoded A = 0, B = 1, C = 2 and tiled across the trial's timepoints; the same label drives the reward-zone interval used for output 0.

ii.
```python
transitions = [("A","B"),("B","A"),("A","C"),("C","A"),("B","C"),("C","B")]
for src, dst in transitions:
    if f"{src}_to" in scene and scene.endswith(dst):
        return [src] * min(switch_trial, n_trials) + [dst] * max(0, n_trials - switch_trial)
raise ValueError(f"Unsupported scene for reward-zone mapping: {scene}")
```

iii. Step 3 quotes the Methods: "On day 3 … the zone was moved after 30 trials … Each switch occurred after 30 trials." Step 1: "Reward-zone labels A/B/C are not read directly from framewise variables; they are inferred from the `scene` name using `behavior.get_reward_zones`. For switch sessions, the code assumes the switch occurs at `change_reward_trial` when present, otherwise default `change_trial=30`." Step 5's planned sanity check was to verify the schedule "for at least three subjects, including a switch session and a non-switch session".

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` behavioural time series' event `timestamps`, compared against the trial's start/end frame timestamps (same array used for the previous-trial-outcome input).

ii.
```python
reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)
...
start_ts = frame_timestamps[start]
end_ts = frame_timestamps[end]
reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
```

iii. Step 5 mapping table: "Reward timestamps within trial → `output[5]` = reward outcome; If a reward event timestamp falls between trial start and teleport, output 1 else 0; repeat across `T`", referencing `behavior.get_trial_types` / `rewardAnalysis.get_reward_inds`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A per-trial binary: 1 if at least one reward event timestamp lies in the half-open window `[t(trial_start), t(teleport))`, else 0. Because the comparison is done in timestamp space against the trial's own boundary frames, no searchsorted/nearest-frame snapping is needed. The scalar is tiled across the trial's timepoints. Exported distribution: [0.158 omitted, 0.842 rewarded].

ii.
```python
reward_outcome = int(reward_outcomes_all[trial_idx])
...
repeated_row(reward_outcome, T, np.int16),
```

iii. Step 3 gives the expected statistic from the paper ("Reward was randomly omitted on approximately 15% of trials"), and Step 9/10 confirm: "Reward rate matches the paper expectation: converted reward outcome `[0.158 no, 0.842 yes]`, consistent with 'approximately 15%' omission." Step 12 additionally re-checked three specific trials against the raw reward stream.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Frame-count mismatches** between behaviour and the per-plane neural matrices (seen in two-plane files, off by one frame): every stream is truncated to `min_frames`, the minimum over the behaviour timestamps and all `Deconvolved/plane*` row counts, before trialization.
- **Dangling/partial trials** (e.g. `sub-m11_ses-03` has a trailing `trial number` label with no `trial_start`/`teleport`): only greedily paired start→teleport events become trials, so unmatched pulses are silently dropped.
- **Events outside scanning**: trial-start/teleport pulses are only accepted where `scanning > 0`.
- **Pre-synchronization sentinels** (`environment == -1`): filtered out before taking the per-trial environment value; an empty result raises.
- **Faulty lick sensor**: whole trials removed (1-e); zero-length trials are also flagged bad.
- **Hard failures**: unexpected frame dt, unknown scene, `T`/neuron shape mismatch, any NaN in the converted arrays, or a session left with <2 trials all raise `RuntimeError` rather than silently emitting bad data.
- No imputation of missing samples is performed anywhere, and no NaNs are written into the export.

ii.
```python
min_frames = min([frame_timestamps.shape[0], *plane_frame_counts])
...
if not np.isclose(frame_dt, COMMON_FRAME_DT_S, atol=1e-6):
    raise RuntimeError(f"{session_tag}: unexpected frame dt {frame_dt}")
...
env_trial_vals = env_trial_vals[env_trial_vals >= 0]
if env_trial_vals.size == 0:
    raise RuntimeError(f"{session_tag}: no valid environment values in trial {trial_idx}")
...
if np.isnan(neural_trial).any() or np.isnan(input_trial).any() or np.isnan(output_trial).any():
    raise RuntimeError(f"{session_tag}: NaN values detected after conversion")
if len(neural_trials) < 2:
    raise RuntimeError(f"{session_tag}: fewer than 2 valid trials after filtering")
```

iii. Step 10 Check 5: "`sub-m11_ses-03` contains one dangling final `trial number` label without a paired `trial_start` / `teleport`; it remains excluded … Two-plane sessions with one-frame stream mismatches are safely truncated to the minimum common frame count before trialization." The min-frames truncation was added in response to a real crash during the first full run (Step "the first real full-run bug is a frame-count mismatch inside one two-plane session … The safe fix is to truncate each session to the minimum shared frame count across behavior and plane series").

## 13-a. What are the most time-consuming steps of the code?

i. Per-session `elapsed` ranges 0.12–1.78 s over 152 sessions (~60 s total), while the whole run takes 104.98 s — so the two dominant costs are (1) reading the `Deconvolved/plane*` matrices out of HDF5 with a fancy column index (the `iscell` subset), which scales with neurons × frames, and (2) the final `pickle.dump` of the ~4.82 GB dataset (~45 s, the difference between total and summed per-session time). Everything else (behaviour streams, trial pairing, discretization) is negligible. No profiling instrumentation beyond per-session and total wall-clock timers is present.

ii.
```python
start_time = time.perf_counter()
...
"elapsed_s": time.perf_counter() - start_time,
...
with args.outpicklefile.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
total_elapsed = time.perf_counter() - t0
print(f"Total elapsed: {total_elapsed:.2f}s")
```

iii. Step 6: "The full dataset is large: rough estimate for complete-trial deconvolved activity at `iscell[:,0] == 1` is ~2.42 billion values (~9.67 GB float32, ~4.84 GB float16). Pickle serialization of many per-trial arrays may be slow on the full dataset." Mitigations listed: "Process sessions one at a time … Read a session's deconvolved matrix once, slice trials from memory, then release it … Store neural data as `float16`."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops remain, all O(n_trials) or O(n_planes) and all cheap relative to I/O:
- `pair_trial_events()` — a two-pointer loop over trial-start pulses; could be done with `np.searchsorted` on the teleport indices.
- `detect_bad_lick_trials()` — per-trial `np.mean(trial_lick > 2)`; could be a single `np.add.reduceat` over the concatenated trial windows.
- the `reward_outcomes_all` loop — per-trial interval test on reward timestamps; could be a single `np.searchsorted` of `reward_event_ts` into the boundary timestamps.
- the main per-trial conversion loop — slicing, discretizing and stacking each trial. Because trials have variable length and must be stored as separate arrays, the discretizations could be hoisted to whole-session arrays (discretize once, then slice), which would be the one worthwhile change.
Also, the per-plane HDF5 reads use fancy column indexing per plane, which is slower than reading contiguous blocks and subsetting in memory.

ii.
```python
for i, (start, end) in enumerate(trial_pairs):
    trial_lick = lick[start:end]
    ...
for i, (start, end) in enumerate(trial_pairs):
    reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
...
for trial_idx, ((start, end), bad_lick, reward_label) in enumerate(zip(trial_pairs, bad_lick_trials, reward_labels_all, strict=True)):
    reward_dist = reward_distance_to_zone(pos_trial, zone_start, zone_end)
    pos_bin = discretize_absolute_position(pos_trial)
    speed_bin = discretize_speed(speed[trial_slice])
```

iii. Not explicitly discussed in CONVERSION_NOTES; the agent's stated efficiency strategy (Step 6) was memory- and I/O-oriented (one session at a time, single read of each matrix, `float16` storage) rather than loop vectorization, and the measured 105 s total made further optimization unnecessary relative to the 15-minute budget in the instructions.

## 13-c. What processing does the code repeat multiple times?

i. Very little: each NWB file is opened exactly once and each array read once (there is no separate survey/scan pass over the dataset). The repetitions that do exist are trivial:
- the `lick` array is scanned twice — once in `detect_bad_lick_trials` and again when binarizing per trial;
- `np.unique(plane_idx)`/`np.unique(kept_plane_idx)` are recomputed rather than reused;
- `local_index` is built for *all* ROIs even though only the `iscell` subset is needed;
- per-trial variables (environment, trial number, previous outcome, reward-zone label, reward outcome) are materialized as full-length rows for every trial, duplicating one scalar `T` times in memory and on disk.

ii.
```python
bad_lick_trials = detect_bad_lick_trials(lick, trial_pairs)   # pass 1 over lick
...
lick_bin = (lick[trial_slice] > 0).astype(np.int16)           # pass 2 over lick
...
unique_planes = np.unique(plane_idx)
...
for plane in np.unique(kept_plane_idx):
...
def repeated_row(value, T, dtype):
    return np.full((T,), value, dtype=dtype)
```

iii. Step 6 explicitly lists the single-pass design as a speed-up: "Read a session's deconvolved matrix once, slice trials from memory, then release it before moving to the next file" and "Direct NWB slicing into per-trial arrays — avoids redundant re-reading / intermediate serialization". Tiling per-trial scalars is a deliberate format choice (Key Decision 6: uniform `(d, T)` matrices for the decoder).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor items only:
- Neural data is cast to `float16` on write and will be cast back to floating point by the decoder — an extra conversion whose only purpose is to halve the 9.7 GB pickle.
- `brain_region_idx` is rebuilt as an all-zeros array per session even though there is exactly one region (`CA1`) for every neuron.
- Per-trial NaN checks are run over every neural matrix (`np.isnan(neural_trial).any()`) on every trial, which touches all 2.4 billion neural values although the source arrays cannot contain NaN.
- `median_step_seconds` recomputes the full timestamp diff per session purely for an assertion.
- Per-trial scalars are tiled to length `T` for five of six outputs and three of four inputs (see 13-c).
- A `summary` dict per session is embedded in `metadata['session_info']`; useful for provenance but unused by the decoder.
- The `--show-processing` plotting path and the `sample_data.pkl` export are validation-only artifacts.
Nothing substantive is computed and thrown away — notably, the converter never reads `Fluorescence`, `Neuropil`, `reward_zone` or `autoreward`, so no expensive unused signal is loaded.

ii.
```python
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
...
if np.isnan(neural_trial).any() or np.isnan(input_trial).any() or np.isnan(output_trial).any():
    raise RuntimeError(f"{session_tag}: NaN values detected after conversion")
...
brain_region_idx = np.zeros(len(keep_indices), dtype=np.int16)
...
"session_info": session_summaries,
```

iii. Step 6: "To keep the full exported pickle tractable, neural trial matrices are stored as `float16`; inputs are `float32` and outputs are integer categorical codes." The validation checks are deliberate: the instructions required sanity checks and shape/type validation at each step, and Step 10's review is built on the per-session summaries stored in `metadata`.
