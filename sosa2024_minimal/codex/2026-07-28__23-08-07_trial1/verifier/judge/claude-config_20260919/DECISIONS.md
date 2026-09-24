# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file in the dataset is discovered with a single glob over `/app/data`
(`sub-*/sub-*_behavior+ophys.nwb`), sorted by (subject number, session number) parsed from the
path, and each file is converted by `build_session()`. Files are opened directly with `h5py`
rather than `pynwb` (the agent used `pynwb` only during exploration). This yields 11 subjects /
152 sessions / 12,216 trials. Within a session the agent reads the behavior streams
(`position`, `speed`, `lick`, `environment`, `reward_zone`, `trial_start`, `Reward`) from
`processing/behavior/BehavioralTimeSeries`, the ophys arrays from
`processing/ophys/{Deconvolved,Fluorescence,Neuropil}/plane<N>`, and the ROI curation table from
`processing/ophys/ImageSegmentation/PlaneSegmentation`. Session-level identity (subject, scene,
date, experiment day) is taken from `general/subject/subject_id`, `identifier` and
`general/session_id`.

ii.
```python
def subject_session_key(path: Path):
    subject = int(path.parent.name.split("-m")[-1])
    session = int(path.stem.split("_ses-")[-1].split("_")[0])
    return subject, session

def build_full_dataset(data_dir: Path):
    session_records = []
    for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
        print(f"Converting {path.relative_to(data_dir.parent)}")
        record = build_session(path)
```
```python
    with h5py.File(path, "r") as f:
        subject = decode_scalar(f["general/subject/subject_id"][()])
        identifier = decode_scalar(f["identifier"][()])
        session_id = decode_scalar(f["general/session_id"][()])
        scene = identifier.split("/")[-1]
        date = identifier.split("/")[-2]

        beh = f["processing/behavior/BehavioralTimeSeries"]
        position = np.asarray(beh["position/data"], dtype=np.float32)
        ...
```

iii. From the trajectory (steps 23–59): the agent first checked that the NWB export has no
`trials` interval table and stores "frame-aligned behavior and ophys directly in processing
modules", so "trial boundaries and reward-zone identity will likely have to be reconstructed
from the aligned behavior streams exactly as the original pipeline did". It then ran a full
inventory over every file (step 53–58) and validated the totals against the manuscript:
"152 sessions across 11 switch-task mice, 12,216 trials total, mean 80.37 trials/session", and
"m11 contributes 12 sessions, consistent with imaging starting on day 3". It switched from
`pynwb` to `h5py` so it could slice curated columns straight out of the HDF5 datasets.

## 1-b. How are the data split into subjects?

i. Subject identity is read from the NWB file itself (`general/subject/subject_id`, e.g. `m3`).
The subject list is built in first-encounter order of the sorted file list (so numeric order
m3, m4, m7, m11, …) and `subject_idx` indexes it per session.

ii.
```python
subject = decode_scalar(f["general/subject/subject_id"][()])
```
```python
    subjects = []
    subject_to_idx = {}
    for record in session_records:
        subject = record["subject"]
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)
    ...
    "subject_idx": np.array([subject_to_idx[record["subject"]] for record in session_records], dtype=np.int64),
```

iii. The agent checked the count against the paper: "Switch-task mice expected: 11. Converted:
11" (`summarize_dataset`, and `CONVERSION_NOTES.md` "Converted mice: `11`, matching the
switch-task cohort in the paper"). Taking the id from the file metadata rather than the
directory name means the subject label is the one the data provider stored; the agent also used
it to map `m<N>` onto the paper's internal `GCAMP<N>` naming when checking day/scene schedules.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are ordered within subject by the `ses-NN` number in the
file name, and the experiment day is taken from `general/session_id` and recorded in
`metadata['session_info']`. No cross-session cell alignment is attempted.

ii.
```python
for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):
    record = build_session(path)
```
```python
        summary = {
            "path": str(path),
            "subject": subject,
            "exp_day": int(session_id),
            "date": date,
            "scene": scene,
            "n_trials": int(ntrials),
            ...
```

iii. The agent confirmed the file-name session number is the experiment day by reading the
NWB identifiers (`/data/InVivoDA/GCAMP3/01_10_2022/Env1_LocationC`, `session_id 01`) and
cross-checking them against `sessions_dict.py` in the paper's repo, which lists one
`{date, scene, exp_day}` entry per session per animal. It used the day number to check the
manuscript claim that imaging for m11 started on day 3 (12 rather than 14 sessions).

## 1-d. How are the data split into trials?

i. Trial starts are the nonzero samples of the `trial_start` behavior stream. The trial end is
**not** taken from the `teleport` stream: instead, within the window from one trial start to the
next, the agent keeps frames up to the *last* frame whose position lies on the corridor,
defined as `0 <= position <= 450.5` cm. The ITI/teleport period is therefore excluded by a
position criterion rather than by the teleport flag.

ii.
```python
def find_trial_segments(position: np.ndarray, trial_start: np.ndarray):
    starts = np.flatnonzero(trial_start > 0)
    next_starts = np.concatenate([starts[1:], [len(position)]])
    ends = []

    for start, next_start in zip(starts, next_starts):
        trial_pos = position[start:next_start]
        valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5))
        if len(valid) == 0:
            continue
        end = start + valid[-1] + 1
        if end <= start:
            continue
        ends.append(end)

    starts = starts[:len(ends)]
    ends = np.array(ends, dtype=np.int64)
    return starts.astype(np.int64), ends
```

iii. From `CONVERSION_NOTES.md`: "Trials are aligned to `trial_start`. For each trial, frames are
kept from that start until the last frame still on the corridor. Corridor frames are defined by
position in `[0, 450]` cm; teleport-zone frames are excluded. This matches the reference focus on
the 450 cm track rather than the teleport period." The agent had inspected `trial_start` /
`teleport` / `position` together (step 46) and seen that position drops to −50 at teleport
("pos start/end 1.26 → −50.0"), which motivated the position-based cutoff. Trial counts agree
exactly with the expert solution (12,216), and mean trials/session (80.37) matches the paper's
80.5 ± 7.4.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. Every trial found by `find_trial_segments` is kept.
The only curation at this level is at the session level: a session raises an error if fewer than
2 trials are found or if no neurons survive the cell filters (neither condition occurs in this
dataset). There is no minimum-duration filter, no lick-sensor-error rejection, and no speed
filter.

ii.
```python
        starts, ends = find_trial_segments(position, trial_start)
        ntrials = len(starts)
        if ntrials < 2:
            raise ValueError(f"{path.name}: expected at least 2 trials, found {ntrials}")
```
```python
        if n_neurons == 0:
            raise ValueError(f"{path.name}: no neurons remain after interneuron exclusion")
```

iii. The agent never states a rationale for omitting a trial filter; its sanity check is
distributional instead — trials/session (80.37), total trials (12,216) and omission fraction
(0.1534 vs "approximately 15%" in the manuscript) all land on the manuscript values, which it
took as evidence that no trials needed to be dropped. It did notice the paper's lick-sensor
correction (`lick_correction_thr: 0.35` in `dayData.py`, step 47) but did not implement it. The
shortest converted trial is 96 frames (~6 s), so no degenerate trials exist.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are the NWB `processing/ophys/Deconvolved/plane<N>/data` arrays — suite2p's
own deconvolution of raw fluorescence as exported to the NWB file — restricted to curated,
non-interneuron ROIs and concatenated across imaging planes. The raw `Fluorescence` and
`Neuropil` arrays *are* read, but only to reconstruct dF/F for the interneuron screen; the dF/F
and any re-deconvolution of it are discarded and never enter `data['neural']`.

ii.
```python
def load_curated_events(
    f: h5py.File,
    plane_masks: dict[str, np.ndarray],
    keep_mask: np.ndarray,
    common_length: int,
):
    events = f["processing/ophys/Deconvolved"]
    plane_names = get_plane_names(events)
    ...
    for plane_name, plane_keep in zip(plane_names, keep_splits):
        if not np.any(plane_keep):
            continue
        curated_mask = plane_masks[plane_name]
        plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)
        arrays.append(plane_data[:, plane_keep])
    ...
    return np.concatenate(arrays, axis=1)
```
```python
        events = load_curated_events(f, plane_masks, keep_mask, common_length)
        n_neurons = events.shape[1]
        ...
            neural = events[start:end].T.astype(np.float32)
```
```python
            "neural_signal": "suite2p OASIS deconvolved calcium events exported in the NWB files",
```

iii. From `CONVERSION_NOTES.md`: "Used the NWB `processing/ophys/Deconvolved` traces as the
decoder neural input… Reconstructed per-trial dF/F from NWB fluorescence and neuropil **only for
the putative-interneuron screen**". At step 62 the agent stated the plan explicitly: "reconstruct
the paper's per-trial dF/F only for the interneuron screen, use the exported deconvolved activity
as the neural signal". No justification is given for preferring the stored array over the
paper's own `events`, even though the agent had already read
`preprocessing.dff(..., deconvolve=True)` and the `make_multi_anim_sess` notebook
(`'ts_key': 'events'`, `baseline_method = 'maximin'`), which document that the paper computes its
own signal.

## 2-b. How is the `neural` data processed?

i. The values placed in `data['neural']` receive **no** processing beyond ROI selection, slicing
to the trial window, casting to `float32` and transposing to (n_neurons, n_timepoints). No
neuropil subtraction, baseline/dF/F normalisation, smoothing or deconvolution is applied by the
conversion, because the stored array is already suite2p's deconvolved output.

Separately (and only for cell curation, question 2-c), a per-trial dF/F is reconstructed in
`compute_interneuron_mask`: subtract `0.7 * Fneu`, add back the per-trial mean neuropil, take a
maximin baseline (Gaussian sigma 15 samples → 300-sample running minimum → 300-sample running
maximum), form `(F - baseline)/|baseline|`, and smooth with a 2-sample Gaussian. This mirrors the
paper's `preprocessing.dff`, except that it always restricts the baseline window to the lap
(the paper's `keep_teleports` per-animal/per-day table is not used) and it stops before the
OASIS deconvolution step.

ii.
```python
        for start, end in zip(starts, ends):
            trial_roi = f_roi[:, start:end] - 0.7 * f_neu[:, start:end]
            trial_roi = trial_roi + 0.7 * np.mean(f_neu[:, start:end], axis=1, keepdims=True)
            baseline = gaussian_filter1d(trial_roi, sigma=15, axis=1, mode="nearest")
            baseline = minimum_filter1d(baseline, size=300, axis=1, mode="nearest")
            baseline = maximum_filter1d(baseline, size=300, axis=1, mode="nearest")
            trial_dff = (trial_roi - baseline) / np.abs(baseline)
            trial_dff = gaussian_filter1d(trial_dff, sigma=2, axis=1, mode="nearest")
            dff[:, start:end] = trial_dff
```
```python
            neural = events[start:end].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` lists the dF/F reconstruction parameters as "matching the reference
logic: neuropil subtraction with coefficient `0.7`, per-trial maximin baseline, Gaussian
smoothing before baseline estimation, final Gaussian smoothing with sigma `2`", and the agent
noted at step 85 that this reconstruction is what makes the converter CPU-bound. The choice to
use it for the interneuron screen but not for the neural signal is asserted, not argued.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, matching the paper. (1) ROIs are restricted to suite2p's manual curation,
`iscell[:, 0] == 1`, split per plane using the `planeIdx` column of the `PlaneSegmentation`
table (with a length assertion against the data columns). (2) Putative interneurons are then
excluded: any remaining cell whose reconstructed dF/F correlates with running speed at r > 0.5
over all in-trial samples. Across the dataset this leaves 910.0 ± 448.1 neurons/session
(min 154, max 2324) out of 912.4 curated, i.e. 0.31% removed.

ii.
```python
def get_curated_plane_masks(f: h5py.File):
    seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
    iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)
    if "planeIdx" in seg:
        plane_idx = np.asarray(seg["planeIdx"]).astype(int)
    ...
        mask = iscell[plane_idx == plane_num]
        ncols = f["processing/ophys/Deconvolved"][plane_name]["data"].shape[1]
        if mask.shape[0] != ncols:
            raise ValueError(...)
```
```python
        dff_valid = dff[:, valid_mask]
        dff_centered = dff_valid - np.nanmean(dff_valid, axis=1, keepdims=True)
        numerator = np.nansum(dff_centered * speed_centered[None, :], axis=1)
        denom = np.sqrt(np.nansum(dff_centered ** 2, axis=1) * speed_ss)
        corr = np.divide(numerator, denom, out=np.zeros_like(numerator, dtype=np.float32), where=denom > 0)
        all_is_int.append(corr > 0.5)
```
```python
        is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
        keep_mask = ~is_int
        events = load_curated_events(f, plane_masks, keep_mask, common_length)
```

iii. `CONVERSION_NOTES.md`: "Kept only manually curated ROIs with `iscell[:, 0] == 1`… Excluded
putative interneurons with `corr(dff, speed) > 0.5`, matching the manuscript criterion and
`dayData.py`." The agent explicitly searched the repo for this step (step 27/30:
`rg -n "exclude_int|interneuron|speed_corr|corrcoef"`) before implementing it, and validated the
result against the manuscript: "Interneuron removal fraction/session: mean `0.0031`… close to
the manuscript statement that speed-correlated interneuron exclusion removed a very small
fraction of cells."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start is achieved by construction: the trial window begins exactly at the
`trial_start` sample, so sample 0 of every neural matrix is the alignment event. Neural and
behavior share one frame index (both streams are frame-aligned in the NWB export), so the same
`[start:end]` slice is applied to both. `metadata['temporal_alignment_event']` is set to
"trial start / entry onto the 0 cm corridor position", with `off_start = 0.0` and
`off_end = None`. No pre-event padding is included.

ii.
```python
        for trial_idx, (start, end) in enumerate(zip(starts, ends)):
            pos_trial = np.clip(position[start:end], 0.0, 450.0)
            speed_trial = speed[start:end]
            lick_trial = (lick[start:end] > 0).astype(np.int64)
            ...
            time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
            neural = events[start:end].T.astype(np.float32)
```
```python
            "temporal_alignment_event": "trial start / entry onto the 0 cm corridor position",
            "off_start": 0.0,
            "off_end": None,
```

iii. The agent verified in exploration (steps 25, 46, 57) that the ophys and behavior series are
sample-for-sample aligned at the same frame rate, so no interpolation or offset is needed; the
only alignment work is trimming all streams to a common length (question 12).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: data are kept at the native imaging/behavior frame rate, one
sample per bin. The bin size is computed per session as the median inter-sample interval of the
behavior timestamps and stored as the dataset-wide median: 64.4836 ms (~15.5 Hz), the per-plane
rate. Note the agent used the timestamps rather than the `rate` attribute on the ophys series,
which is the scanner rate (31.015625 Hz on the two-plane sessions m17/m18) and would have been
wrong by a factor of 2 there. The value is written under the key `time_bin_size_ms` rather than
the `time_bin_size` name given in the target format.

ii.
```python
        dt_seconds = float(np.median(np.diff(position_ts)))
```
```python
    dt_all = [record["session_summary"]["dt_seconds"] for record in session_records]
    ...
            "time_bin_size_ms": float(np.median(dt_all) * 1000.0),
```

iii. `CONVERSION_NOTES.md`: "Median frame interval from the aligned behavior timestamps:
`0.0644836` s / `64.4836` ms. This matches the approximately `15.5 Hz` per-plane sampling
described in the paper." In the trajectory (steps 56–58) the agent explicitly compared
`imaging_rate` / series `rate` (31.015625) against the behavior timestamp spacing (0.064484 s)
on a two-plane animal and against the per-plane row counts, which is how it settled on the
timestamp-derived value.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From `processing/behavior/BehavioralTimeSeries/position/timestamps` (in seconds), sliced to
the trial window.

ii.
```python
        position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)
```
```python
            time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
```

iii. The agent verified during exploration that all behavior series in the module carry the same
timestamp vector and that its spacing is the imaging frame interval (step 57: `dt head
[0.064484]`), so the position timestamps double as the common frame clock for the session; it
reuses them for the time input, the reward-event matching and the time-bin size.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the first timestamp of the trial, giving a time-varying vector starting
at 0 s. Stored as row 0 of the `(4, T)` input array, cast to `float32`.

ii.
```python
            time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)
            inputs = np.vstack([
                time_trial,
                np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
                np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
                np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
            ])
```

iii. No explicit justification is recorded; this is the direct reading of the decoder-input spec
("Time from start of trial in seconds"). The resulting range (0 → 216.6 s) is reported in the
verification log.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is taken from the same frame indices as the neural data, so the alignment is exact by
construction. Before any trial extraction, all dense streams in a session — the seven behavior
arrays plus every ophys plane of `Deconvolved`, `Fluorescence` and `Neuropil` — are truncated to
their common minimum length, so the behavior and neural streams cannot drift apart at the end of
a session.

ii.
```python
        dense_lengths = [len(position), len(position_ts), len(speed), len(lick),
                         len(environment), len(reward_zone), len(trial_start)]
        for grp_name in ("Deconvolved", "Fluorescence", "Neuropil"):
            grp = f[f"processing/ophys/{grp_name}"]
            for plane_name in get_plane_names(grp):
                dense_lengths.append(grp[plane_name]["data"].shape[0])
        common_length = min(dense_lengths)
        clipped_samples = max(dense_lengths) - common_length
        position = position[:common_length]
        position_ts = position_ts[:common_length]
        ...
```

iii. From step 95: "I've confirmed the failing session has 22,790 behavior frames but 22,791
ophys frames in both planes. I'm patching the session loader to trim all dense streams to the
shortest shared length and record how many samples were clipped, since that is the safest way to
preserve the original alignment semantics." The number of clipped samples is recorded per
session in `metadata['session_info']['samples_clipped_to_align_streams']`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Primarily from the session `scene` string parsed out of the NWB `identifier`
(e.g. `Env1_LocationC`, `Env1_B_to_Env2_C`), mapped `Env1 → 0`, `Env2 → 1`. The `environment`
behavior time series is read as well, but only as a cross-check: the per-trial median of the
stream is compared with the scene-derived label and the number of mismatches is recorded.

ii.
```python
ENV_TO_IDX = {"Env1": 0, "Env2": 1}
...
        scene = identifier.split("/")[-1]
```
```python
            env_vals = environment[start:end]
            env_vals = env_vals[env_vals >= 0]
            behavior_env_by_trial.append(int(round(float(np.median(env_vals)))))
        ...
        env_mismatch = int(np.sum(env_by_trial != np.asarray(behavior_env_by_trial, dtype=np.int64)))
```

iii. The agent found the mapping in the paper's own code (`behavior.py`:
`env_morph_dict = {'Env1': 0, 'Env2': 1, 'Env3': 0.5}`) and confirmed at step 52 that "the scene
names in the NWB identifiers encode the same reward-zone and environment transitions used by the
reference code, including day-8 environment switches inside a session". The cross-check came out
clean on the whole dataset: "`0` environment-mismatched trials".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `scene_schedule()` turns the scene string into a per-trial environment vector. For fixed and
within-environment-switch scenes the environment is constant for the session; for the day-8
cross-environment scenes (`Env1_B_to_Env2_C`, `Env2_B_to_Env1_A`, …) the environment changes at
trial 30 (`CHANGE_TRIAL`), the paper's switch trial. The per-trial value is then broadcast
across all timepoints of the trial as row 1 of the input array.

ii.
```python
    fixed_match = re.fullmatch(r"(Env[12])_Location([ABC])", scene)
    same_env_switch_match = re.fullmatch(r"(Env[12])_Location([ABC])_to_([ABC])", scene)
    env_switch_match = re.fullmatch(r"(Env[12])_([ABC])_to_(Env[12])_([ABC])", scene)
    ...
    elif env_switch_match:
        env0 = ENV_TO_IDX[env_switch_match.group(1)]
        ...
        split = min(change_trial, ntrials)
        env_by_trial[:split] = env0
        env_by_trial[split:] = env1
```
```python
                np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
```

iii. `CONVERSION_NOTES.md`: "`environment_type` uses the manuscript/code mapping `Env1 -> 0`,
`Env2 -> 1`… Day-8 environment-switch sessions such as `Env1_C_to_Env2_B` and `Env2_B_to_Env1_A`
are handled explicitly." The scene grammar and the trial-30 switch come from
`reward_relative.behavior.get_reward_zones(..., change_trial=30)`, which the agent read at
step 48; the zero-mismatch check against the `environment` stream validates both the grammar and
the switch trial empirically.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from a raw variable: it is the 0-based index of the trial within the session, i.e. the
position of the trial in the list produced by `find_trial_segments` (which itself comes from the
`trial_start` stream). The NWB `trial number` behavior series is not used.

ii.
```python
        for trial_idx, (start, end) in enumerate(zip(starts, ends)):
            ...
                np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
```

iii. `CONVERSION_NOTES.md`: "`trial_number` is `0`-indexed within session, matching the aligned
behavior stream." The agent had seen at step 46 that the stored `trial number` series does not
line up cleanly with the `trial_start`/`teleport` boundaries (a single start-to-teleport window
could span two values of `trial number`, e.g. "trialnums [2. 3.]"), which makes the loop index
the more reliable definition.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the constant index across the trial's timepoints (row 2 of the input
array). No normalisation and no continuation of the count across sessions; each session restarts
at 0 (range 0–99 across the dataset).

ii.
```python
            inputs = np.vstack([
                time_trial,
                np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32),
                np.full(time_trial.shape[0], trial_idx, dtype=np.float32),
                np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
            ])
```

iii. The spec calls trial number a per-trial continuous input; the agent stores every input as a
time-varying `(4, T)` row so all inputs share one layout ("Inputs are stored as time-varying
`(4, T)` arrays for every trial").

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the sparse `Reward` time series, specifically its `timestamps` (reward delivery events),
compared against the behavior timestamps of each trial window. The per-trial reward flag computed
for output 5 is reused, shifted by one trial.

ii.
```python
        reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
```
```python
            reward_outcome.append(
                int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
            )
```

iii. The agent's inventory scan (steps 53–58) confirmed the `Reward` series is sparse (71 events
in the sample session) with its own timestamps, so membership is tested by timestamp interval
rather than by frame index. `CONVERSION_NOTES.md`: "Reward outcome is derived from the sparse NWB
reward timestamps within each trial."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *k* > 0 the value is the reward flag of trial *k−1* (1 = rewarded, 0 = omitted);
trial 0 is set to 0. The value is broadcast across the trial's timepoints as row 3 of the input
array. "Previous trial" means the previous converted trial in the same session; it does not
carry across sessions.

ii.
```python
            reward_trial = reward_outcome[trial_idx]
            prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0
            ...
                np.full(time_trial.shape[0], prev_reward, dtype=np.float32),
```

iii. `CONVERSION_NOTES.md`: "`previous_trial_outcome` uses the previous imaged trial in the same
session; the first trial is set to `0`." The overall omission rate (0.1534) was checked against
the manuscript's ~15%.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavior stream together with the active reward-zone boundaries for that
trial. The boundaries come from the scene schedule (question 10), i.e. from the session `scene`
string mapped through `ZONE_BOUNDS_CM = {A: (80, 130), B: (200, 250), C: (320, 370)}` — the
paper's `X`/`Y`/`Z` zone coordinates. The `reward_zone` behavior stream is read but used only to
verify those labels, not to set them.

ii.
```python
ZONE_BOUNDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
```
```python
    zone_bounds = np.array([ZONE_BOUNDS_CM[z] for z in zone_labels], dtype=np.float32)
```
```python
            zone_vals = reward_zone[start:end] > 0
            if np.any(zone_vals):
                zone_pos = np.clip(position[start:end][zone_vals], 0.0, 450.0)
                center = float(np.mean(zone_pos))
                inferred = min(ZONE_BOUNDS_CM, key=lambda label: abs(center - np.mean(ZONE_BOUNDS_CM[label])))
                parsed_zone_checks += 1
                parsed_zone_matches += int(inferred == zone_labels[trial_idx])
```

iii. The agent read `behavior.py` (step 48) and found both the zone dictionary
(`'X': [80, 130], 'Y': [200, 250], 'Z': [320, 370]`) and the `map_labels = {'A': 'X', 'B': 'Y',
'C': 'Z'}` translation used by `get_reward_zones`. It then verified the scene-derived labels
against the observed reward-zone occupancy on every trial where the animal entered the zone:
"mean reward-zone agreement `1.0000`".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest point of the active zone: negative
before the zone (`position − zone_start`), exactly 0 while inside it, positive after it
(`position − zone_end`). Position is first clipped to `[0, 450]` cm. The continuous distance is
then discretised in place (question 7-c); the continuous values are not stored.

ii.
```python
def discretize_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
```
```python
            pos_trial = np.clip(position[start:end], 0.0, 450.0)
            ...
                discretize_distance_to_zone(pos_trial, zone_start, zone_end),
```

iii. `CONVERSION_NOTES.md`: "Distance to reward zone is signed distance to the nearest point in
the active 50 cm reward zone: before zone: negative distance to zone start; inside zone: `0`;
after zone: positive distance to zone end." This is the literal reading of the decoder-output
spec ("Distance to any location in the reward zone"), and it reproduces the paper's
reward-relative distance coordinate.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Into the 7 categories of the spec with explicit boolean masks:
0: `< −50`; 1: `[−50, −10)`; 2: `[−10, 0)`; 3: `== 0` (in the zone); 4: `(0, 10]`;
5: `(10, 50]`; 6: `> 50`. Category names are stored in `output_values[0]`. Resulting
distribution: {0.260, 0.100, 0.073, 0.234, 0.020, 0.071, 0.242}.

ii.
```python
    bins = np.empty(position_cm.shape[0], dtype=np.int64)
    bins[dist < -50.0] = 0
    bins[(dist >= -50.0) & (dist < -10.0)] = 1
    bins[(dist >= -10.0) & (dist < 0.0)] = 2
    bins[dist == 0.0] = 3
    bins[(dist > 0.0) & (dist <= 10.0)] = 4
    bins[(dist > 10.0) & (dist <= 50.0)] = 5
    bins[dist > 50.0] = 6
    return bins
```
```python
        "output_values": [
            ["lt_-50cm", "-50_to_-10cm", "-10_to_lt_0cm", "0cm", "gt_0_to_10cm", "10_to_50cm", "gt_50cm"],
```

iii. The edges are copied from the instructions. The masks are written so that the "0 cm"
category is exactly the in-zone samples, which is why the code computes the signed distance as
identically zero inside the zone rather than as a distance to the zone centre.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices as the neural data — `position[start:end]` with the same `start`/`end` used
for `events[start:end]` — after the session-wide common-length truncation. No resampling or
shifting.

ii.
```python
        for trial_idx, (start, end) in enumerate(zip(starts, ends)):
            pos_trial = np.clip(position[start:end], 0.0, 450.0)
            ...
            neural = events[start:end].T.astype(np.float32)
            ...
            outputs = np.vstack([
                discretize_distance_to_zone(pos_trial, zone_start, zone_end),
                ...
```

iii. The behavior and ophys streams are frame-aligned in the NWB export (verified in exploration:
identical sample counts up to the off-by-one cases handled by `common_length`, and a behavior
sample interval equal to the per-plane imaging interval), so a shared index slice is sufficient.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (VR corridor position in cm).

ii.
```python
        position = np.asarray(beh["position/data"], dtype=np.float32)
```
```python
            pos_trial = np.clip(position[start:end], 0.0, 450.0)
```

iii. The agent inspected this stream in exploration (step 46) and saw laps running from ~0 cm at
`trial_start` to ~450 cm before teleport, with −50 cm during the ITI; this is also what defines
its trial window (question 1-d).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Clipping to the track extent `[0, 450]` cm and then binning. The clip handles the handful of
samples that fall marginally outside the corridor; because the trial window already excludes the
teleport period, no −50 cm ITI samples are involved (except in the small number of trials where
the position-based end criterion lets post-teleport samples back in, see 1-d).

ii.
```python
def discretize_position(position_cm: np.ndarray):
    clipped = np.clip(position_cm, 0.0, 450.0)
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
    return bins
```

iii. `CONVERSION_NOTES.md`: "Absolute position uses five equal 90 cm corridor bins." The Methods
give a 450 cm track, which the agent quoted when defining the corridor criterion.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over 0–450 cm, computed as `floor(position / 90)` with the top edge
folded into bin 4 (so position exactly 450 falls in bin 4). Categories are named
`bin0`…`bin4` in `output_values[1]`. Distribution: {0.217, 0.176, 0.229, 0.224, 0.154}.

ii.
```python
    bins = np.floor(clipped / 90.0).astype(np.int64)
    bins[bins > 4] = 4
```
```python
            ["bin0", "bin1", "bin2", "bin3", "bin4"],
```

iii. Directly from the instructions ("Discretized into 5 equal-sized bins spanning the 450 cm
track"); the agent notes the equivalence 450/5 = 90 cm per bin.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical frame slice to the neural data, as in 7-d; no additional alignment step.

ii.
```python
            pos_trial = np.clip(position[start:end], 0.0, 450.0)
            neural = events[start:end].T.astype(np.float32)
```

iii. Same rationale as 7-d: one frame clock for behavior and ophys after common-length trimming.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series (per-frame lick counts from the capacitive sensor).

ii.
```python
        lick = np.asarray(beh["lick/data"], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` describes these as "framewise cumulative lick counts"; the agent
checked the value range during its inventory pass before deciding to threshold.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation only: `lick > 0 → 1`, else 0, per timepoint (row 3 of the output array). No
smoothing, no de-bouncing, and no lick-sensor-error correction (the paper's
`lick_correction_thr = 0.35` trial rejection is not applied). Resulting distribution:
77.2% no-lick / 22.8% lick.

ii.
```python
            lick_trial = (lick[start:end] > 0).astype(np.int64)
```
```python
            ["no", "yes"],
```

iii. `CONVERSION_NOTES.md`: "Licks are binarized from the framewise cumulative lick counts with
`lick > 0`", following the spec's "Lick, time-varying. 0 = no, 1 = yes". The agent located the
paper's lick-sensor correction (step 47) but chose not to port it; no reason is given.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame slice as the neural data; no additional alignment.

ii.
```python
            lick_trial = (lick[start:end] > 0).astype(np.int64)
```

iii. Same rationale as 7-d/8-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the session `scene` string in the NWB `identifier`, parsed by `scene_schedule()` into a
per-trial zone label A/B/C (→ 0/1/2). The `reward_zone` behavior stream and `position` are used
only to audit the label (nearest-zone-centre check on the samples where `reward_zone > 0`).

ii.
```python
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}
...
    zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels], dtype=np.int64)
```
```python
                discretize_position(pos_trial),
                discretize_speed(speed_trial),
                lick_trial,
                np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64),
```

iii. `CONVERSION_NOTES.md`: "Parsed the session `scene` from the NWB identifier and reproduced the
same schedule logic as `reward_relative.behavior.get_reward_zones`." The agent read that function
in full (step 48) and mirrored its scene grammar and default `change_trial=30`. It validated the
result on the data: "mean reward-zone agreement `1.0000`" and near-uniform label counts
(A 0.329 / B 0.337 / C 0.335).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. `scene_schedule()` classifies the scene into one of three grammars and emits a per-trial label:
- `Env<1|2>_Location<A|B|C>` → one zone for the whole session;
- `Env<1|2>_Location<X>_to_<Y>` → zone X for trials 0–29, zone Y from trial 30 on;
- `Env<1|2>_<X>_to_Env<1|2>_<Y>` → zone and environment both switch at trial 30.
The switch index is `min(30, ntrials)`. The per-trial label is broadcast across the trial's
timepoints as row 4 of the output array; the corresponding coordinates feed 7-a.

ii.
```python
    elif same_env_switch_match:
        env = ENV_TO_IDX[same_env_switch_match.group(1)]
        zone0 = same_env_switch_match.group(2)
        zone1 = same_env_switch_match.group(3)
        split = min(change_trial, ntrials)
        env_by_trial[:] = env
        zone_labels[:split] = zone0
        zone_labels[split:] = zone1
    ...
    else:
        raise ValueError(f"Unrecognized scene format: {scene}")
```

iii. "Fixed sessions use a single reward zone. Switch sessions use the first `30` trials before
the switch and the remaining trials after the switch" (`CONVERSION_NOTES.md`), i.e. the paper's
own `change_trial=30`, documented in `multi_anim_sess_README.md` as "set0 is the 30 trials before
the reward switch, set1 is all the trials after the reward switch" — a line the agent read at
step 14. The unrecognised-scene branch raises rather than guessing, and the 100% agreement check
against `reward_zone` occupancy confirms the trial-30 boundary empirically for every session.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the `Reward` time series timestamps, tested against the trial's time window given by the
behavior timestamps.

ii.
```python
        reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)
```
```python
            reward_outcome.append(
                int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9)))
            )
```

iii. As in 6-a: the agent established that `Reward` is a sparse event series with its own
timestamps, so it is matched to trials by time interval. The overall rewarded fraction (0.8466 /
omission 0.1534) was checked against the manuscript's ~15% omission rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial, 1 if any reward timestamp falls in `[t_start, t_end]` (inclusive, with a 1e-9
tolerance on the upper edge), else 0. The scalar is broadcast across the trial's timepoints as
row 5 of the output array. Reward amount is not used, and the paper's extra requirement that the
animal also be in the reward zone (`get_trial_types`: `np.any(reward>0) and np.any(rzone>0)`) is
not applied.

ii.
```python
            reward_trial = reward_outcome[trial_idx]
            ...
                np.full(time_trial.shape[0], reward_trial, dtype=np.int64),
```
```python
            ["omitted", "rewarded"],
```

iii. `CONVERSION_NOTES.md`: "Reward outcome is derived from the sparse NWB reward timestamps
within each trial", with the omission fraction used as the manuscript-level sanity check.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms:
- **Stream length mismatch**: every dense stream in a session (7 behavior arrays + all ophys
  planes of `Deconvolved`/`Fluorescence`/`Neuropil`) is truncated to their common minimum length
  before trial extraction, and the number of discarded samples is stored in the session metadata.
  This was added after a two-plane session was found with 22,790 behavior vs 22,791 ophys frames.
- **Out-of-range positions**: positions are clipped to `[0, 450]` cm before binning; frames
  outside the corridor are excluded from the trial window altogether.
- **Invalid environment samples**: negative `environment` values are dropped before taking the
  per-trial median in the consistency check.
- **Hard failures**: a session with fewer than 2 trials, with no surviving neurons, with an
  `iscell` mask whose length disagrees with the data columns, or with an unparseable scene name
  raises an exception rather than being silently mis-converted. None of these fire on this
  dataset.
A trial that contains no corridor frames is skipped, but the skip is implemented by truncating
the `starts` array (`starts = starts[:len(ends)]`), which would drop the *last* trial rather than
the offending one — a latent misalignment that is never exercised here because every trial
contains corridor frames.

ii.
```python
        common_length = min(dense_lengths)
        clipped_samples = max(dense_lengths) - common_length
```
```python
            "samples_clipped_to_align_streams": int(clipped_samples),
```
```python
            env_vals = environment[start:end]
            env_vals = env_vals[env_vals >= 0]
```
```python
        if mask.shape[0] != ncols:
            raise ValueError(
                f"{plane_name}: iscell mask length {mask.shape[0]} does not match data columns {ncols}"
            )
```

iii. Step 93: "The full pass exposed one real edge case rather than a conceptual problem: a few
session arrays are off by one sample between behavior and ophys. I'm patching the loader to trim
every stream in a session to a common length before trial extraction." Step 95 adds that trimming
to the shortest shared length "is the safest way to preserve the original alignment semantics",
and the clipped-sample count is recorded so the trimming is auditable. The consistency counters
(`env_mismatch_trials`, `zone_match_fraction_when_observed`) exist so that a silently wrong scene
parse would show up in the summary rather than in the decoder.

## 13-a. What are the most time-consuming steps of the code?

i. In order:
1. **The dF/F reconstruction in `compute_interneuron_mask`** — per plane it loads the full
   `Fluorescence` and `Neuropil` arrays for the curated cells and runs, per trial, a Gaussian
   filter plus 300-sample minimum and maximum filters plus a second Gaussian over an
   (n_cells × n_frames) matrix. The agent identified this itself: "The converter is CPU-bound
   rather than stuck, which is what I expected from the per-session dF/F reconstruction"
   (step 85).
2. **HDF5 reads with boolean fancy indexing** (`data[:common_length, curated_mask]`), which h5py
   services far more slowly than a contiguous read, performed three times per plane
   (Fluorescence, Neuropil, Deconvolved).
3. **Pickling the result** — the full dataset is 9.3 GB of float32 neural data, and
   `sample_data.pkl` adds another 664 MB written from the same arrays.

ii. N/A (analysis of the code as a whole; the relevant snippet is the per-trial filter loop
quoted in 2-b).

iii. The agent's own reading of the run: it chose to wait out the CPU-bound pass rather than
restart it ("recomputing the interneuron screen would cost more than waiting out the current
run", step 85), and later avoided recomputing the full conversion when only the sample subset was
needed ("The heavy full conversion already succeeded and wrote `converted_data.pkl`; I don't need
to recompute it", step 114).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
- The per-trial loop inside `compute_interneuron_mask` applies `gaussian_filter1d`,
  `minimum_filter1d` and `maximum_filter1d` one trial at a time. Since the filters are applied
  along the time axis within non-overlapping windows, the whole session could be filtered once on
  a NaN-masked array (as the paper's `dff` does with a single pass per step), or trials could be
  stacked into a padded 3-D array.
- `find_trial_segments` loops over trials to find the last corridor frame; this is a
  `np.maximum.reduceat`/`searchsorted`-style operation on the boolean corridor mask.
- The two per-trial loops in `build_session` (the label/reward loop and the trial-assembly loop)
  are pure slicing and could be merged, and the per-trial `np.full(...)` broadcasts could be
  replaced by writing into a preallocated array.
The discretisation helpers are already fully vectorised.

ii. N/A.

iii. Not discussed by the agent; the per-trial structure is inherited from the paper's own
per-trial baseline logic, which genuinely requires trial-wise windows for the baseline but not
for the smoothing.

## 13-c. What processing does the code repeat multiple times?

i.
- The trials of a session are iterated twice in `build_session` (once to build environment/zone
  checks and reward flags, once to assemble the arrays) and a third time inside
  `compute_interneuron_mask`.
- `next_starts` is computed in `find_trial_segments` and then recomputed in `build_session`,
  where the loop variable `next_start` is then never used.
- Each ophys plane is read three times (Fluorescence, Neuropil, Deconvolved), and the curated
  mask is applied twice per plane (once in the dF/F pass, once when loading events).
- `summarize_dataset` walks the whole dataset once for the full set and again for the sample
  subset.
Notably, the code does **not** repeat the expensive whole-file read: unlike a survey-then-convert
design, each NWB file is opened exactly once.

ii. N/A.

iii. Not discussed by the agent. The single-pass design was a consequence of deriving the trial
schedule from the scene string rather than from a dataset-wide pass over reward-zone positions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **The dF/F reconstruction is almost entirely discarded**: the full (n_cells × n_frames) dF/F
  matrix is computed for every plane of every session, and all that is retained is one correlation
  coefficient per cell, which removes 0.31% of cells (2.3 per session). The dF/F itself — which
  is what the paper actually analyses — is thrown away and does not become the neural signal.
- **`sample_data.pkl`** (664 MB, 8 sessions of m3) is built and written on every run although only
  `converted_data.pkl` is required.
- **The audit statistics** (`behavior_env_by_trial`, the nearest-zone-centre check,
  `parsed_zone_matches`, per-session trial-length statistics) are computed for every trial of
  every session and only ever printed.
- **`metadata['session_info']`** carries a full per-session dictionary that no downstream step
  reads.
- Per-trial constants (environment, trial number, previous outcome, zone label, reward outcome)
  are materialised as full-length rows; this is required by the target format, but it inflates the
  pickle, as does storing neural data as dense float32 rather than a sparser representation of a
  deconvolved signal that is mostly exact zeros (~43% of samples in a spot check).

ii.
```python
        is_int = compute_interneuron_mask(f, plane_masks, starts, ends, speed, common_length)
        keep_mask = ~is_int
        events = load_curated_events(f, plane_masks, keep_mask, common_length)
```
```python
    if args.mode in {"sample", "both"}:
        ...
        sample_data = build_sample_dataset(full_data)
```

iii. The agent describes the sample dataset as a deliberate convenience ("a deterministic
subset… preserves both environments, within-environment reward switches, the day-8 environment
switch, and all three reward-zone labels") used to validate format and decoder behaviour quickly
before the expensive full run, and it treated the audit counters as its main defence against a
silent scene-parsing error. It never remarks on the fact that the dF/F it computes is discarded
rather than used as the neural signal.
