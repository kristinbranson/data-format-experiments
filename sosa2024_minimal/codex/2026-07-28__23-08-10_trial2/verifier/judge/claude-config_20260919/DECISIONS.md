# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file matching `sub-*/sub-*_behavior+ophys.nwb` under `/app/data` is globbed and treated as one session (152 files, 11 subjects). Files are opened directly with `h5py` (not `pynwb`), and datasets are read lazily: behavior streams are read whole, while fluorescence/neuropil are read in column blocks of 256 ROIs so the full run stays memory-bounded (~2.3 GB RSS). Subject id is read from `general/subject/subject_id`, the VR scene and date from the NWB `identifier` string, and the session/day number from the `ses-NN` field of the filename. Sessions are sorted by (numeric subject id, session number). The AI noted in `CONVERSION_NOTES.md` that the released directory holds 152 sessions rather than the 154 implied by the paper (m11 ses-01/02 absent), and did not drop any session, subject, or file.

ii.
```python
def list_nwb_files(data_root: Path) -> list[SessionMeta]:
    session_meta: list[SessionMeta] = []
    for file_path in sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb")):
        with h5py.File(file_path, "r") as f:
            subject = decode_scalar(f["general/subject/subject_id"][()])
            identifier = decode_scalar(f["identifier"][()])
        parts = identifier.strip("/").split("/")
        date = parts[-2]
        scene = parts[-1]
        session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
        session_meta.append(SessionMeta(file_path=file_path, subject=subject,
                                        session_num=session_num, scene=scene, date=date))
    session_meta.sort(key=lambda s: (int(s.subject[1:]), s.session_num))
    return session_meta
```
```python
    with h5py.File(session_meta.file_path, "r") as f:
        behavior = f["processing/behavior/BehavioralTimeSeries"]
        ophys = f["processing/ophys"]
        imgseg = ophys["ImageSegmentation"]["PlaneSegmentation"]
        plane_keys = sorted(ophys["Fluorescence"].keys())
```

iii. From the trajectory: the AI first inspected the dandiset layout and one NWB file with `pynwb` (behavior `TimeSeries` keys, `Fluorescence`/`Neuropil`/`Deconvolved` ROI series, `PlaneSegmentation` columns `pixel_mask`/`iscell`/`planeIdx`), then bulk-audited all sessions for trial counts, curated ROI counts and stream coding before writing the converter. It switched to `h5py` deliberately ("read the NWBs directly with `h5py`, process one session at a time, and materialize trial arrays only after neuron filtering so the full run stays memory-safe"), because the paper's own preprocessing has to be recomputed from raw F/Fneu for ~900 cells x ~30k frames per session.

## 1-b. How are the data split into subjects?

i. One subject per unique `general/subject/subject_id` value (`m3, m4, m7, m11 ... m19`), sorted numerically; `subject_idx` indexes this list per session. The subject in the file content, not the directory name, is authoritative (they agree).

ii.
```python
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
dataset["subject_idx"].append(subject_to_idx[meta.subject])
...
dataset["subject_idx"] = np.asarray(dataset["subject_idx"], dtype=np.int16)
```

iii. Not discussed explicitly beyond the early survey, which confirmed 11 mice with 12-14 sessions each; the count of subjects matches the directory structure and the paper's cohort.

## 1-c. How are the data split into sessions?

i. One session per NWB file. The session number is parsed from the filename `ses-NN` and is retained in `metadata['session_info']` together with the scene, date, frame rate and per-session statistics. No cross-session cell alignment is attempted (each session keeps its own neuron set).

ii.
```python
session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
...
for meta in session_meta:
    print(f"Processing {meta.file_path.name} ({meta.subject}, scene={meta.scene})")
    session_data, stats = process_session(meta, block_size=block_size)
    dataset["neural"].append(session_data["neural"])
```

iii. The AI read `sess_lists.py` in the repo, which maps each animal/date/scene to an `exp_day`, and used the `ses-NN` field as that experiment day (the notes record the 152-vs-154 discrepancy and identify the two missing m11 days). Sessions are the natural unit because each file holds one recording with its own ROI set.

## 1-d. How are the data split into trials?

i. A trial is the half-open sample range `[index of trial_start == 1, index of teleport == 1)`, i.e. from the lap start up to but excluding the teleport sample. Both event streams are taken as impulses: every sample where the stream is >0 is treated as one event, and starts/stops are zipped in order.

ii.
```python
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
...
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The AI's bulk audit checked "trial-start/teleport consistency" across all sessions and found ~80 trials per session, matching the paper's task structure; the repo's own `behavior.py`/`preprocessing.py` use the same `trial_start_inds`/`teleport_inds` pair to slice laps. The notes state explicitly: "For each trial, the kept sample range is `[trial_start_idx : teleport_idx)`. The teleport sample itself is excluded."

## 1-e. How are trials filtered based on quality controls?

i. One trial-level filter: a trial is dropped entirely if more than 30% of its samples have a cumulative lick count > 2, the paper's capacitive lick-sensor error criterion. This removed 81 of 12,216 trials (0.66%), leaving 12,135. No minimum-trial-length filter, no session-level or subject-level filter is applied. Reward outcome and previous-trial outcome are computed on the *original* trial indexing before this exclusion, so a kept trial still sees the true previous trial's outcome.

ii.
```python
def bad_lick_trial_mask(lick_counts, start_idx, stop_idx):
    mask = np.zeros(len(start_idx), dtype=bool)
    for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
        if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
            mask[trial_idx] = True
    return mask
...
lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
kept_trial_indices = np.where(~lick_error_mask)[0]
```

iii. The AI found the criterion in the repo (`behavior.py::correct_lick_sensor_error`, "if >correction_thr (fraction) of samples have a cumulative lick count of >2", and `glmUtils.py` with 0.35, `dayData.py` with `lick_correction_thr=0.3/0.35`), then counted affected trials at each threshold (0.3 -> 81, 0.35 -> 69, 0.5 -> 44) and chose 0.3. Its stated rationale (plan step and notes): "drop lick-sensor-corrupted trials using the paper's >30% threshold" — the lick output would otherwise be an artifact for those trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Raw Suite2p traces: `processing/ophys/Fluorescence/planeN/data` (F) and `processing/ophys/Neuropil/planeN/data` (Fneu), restricted to curated ROIs. The stored `processing/ophys/Deconvolved` array is deliberately **not** used.

ii.
```python
fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
neuropil_ds = ophys["Neuropil"][plane_key]["data"]
...
fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
dff, events = compute_dff_and_events(fluorescence=fluorescence, neuropil=neuropil,
                                     start_idx=trial_start_idx, stop_idx=trial_stop_idx,
                                     frame_rate_hz=frame_rate_hz)
```

iii. The AI prototyped both paths and compared them: it ran its reimplementation of the repo's `dff(..., deconvolve=True)` on the first 10 cells of `sub-m3_ses-01` and correlated the result with the stored `Deconvolved` series, getting r ≈ 0.54-0.79 with very different scales (computed mean ~0.019 vs stored mean ~397) and different sparsity (36% vs 66-72% non-zero). It concluded the stored array is Suite2p's own deconvolution of raw F, not the paper's "events", and wrote in the notes: "The saved neural activity is trial-wise OASIS event activity recomputed from Suite2p fluorescence and neuropil, rather than taking the NWB `Deconvolved` array directly."

## 2-b. How is the `neural` data processed?

i. A reimplementation of the repo's `preprocessing.dff(..., neuropil_method='subtract', baseline_method='maximin', deconvolve=True)`, per plane and per trial:
1. Mask everything outside laps to NaN (samples between teleport and the next trial start are excluded).
2. `F - 0.7 * Fneu`.
3. Per trial, add back `0.7 * mean(Fneu)` of that trial so the ratio is a true dF/F.
4. Per trial maximin baseline: Gaussian smoothing sigma = 15 samples, then a 300-sample running minimum, then a 300-sample running maximum (the Methods' ~20 s window).
5. `dF/F = (F - baseline) / |baseline|`.
6. Per trial Gaussian smoothing of dF/F with sigma = 2 samples.
7. Per trial OASIS deconvolution (`suite2p.extraction.dcnv.oasis`, batch 2000, `tau = 0.7`, rate = per-plane frame rate ≈ 15.508 Hz).
Cells from `plane0` and `plane1` are pooled after processing. Events are stored as `float16`. The baseline window is always restricted to the lap — the repo's `keep_teleports` branch (per-animal/day table in `teleport_metadata.py`) is not implemented. Edge handling uses `mode='nearest'` for the Gaussian/min/max filters (scipy's default in the repo call is `reflect`), and arrays are `float32` rather than `float64`.

ii.
```python
    f -= neu_coef * f_neu
    valid = np.isfinite(f[0])

    baseline = np.full_like(f, np.nan, dtype=np.float32)
    for start, stop in zip(start_idx, stop_idx):
        sl = slice(start, stop)
        f[:, sl] = f[:, sl] + neu_coef * np.nanmean(f_neu[:, sl], axis=1, keepdims=True)
        baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
        baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
        baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")

    dff = np.full_like(f, np.nan, dtype=np.float32)
    dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])

    events = np.full_like(f, np.nan, dtype=np.float32)
    for start, stop in zip(start_idx, stop_idx):
        sl = slice(start, stop)
        dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
        events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)
```

iii. The AI read `preprocessing.dff` in the repo and reproduced it step by step (its plan: "recompute deconvolved activity from fluorescence and neuropil using the repo's trial-wise maximin dF/F procedure"). `neu_coef=0.7` and `baseline_method='maximin'` come from the repo's default dff method dict; `tau=0.7` is the repo default in `dff`. The notes list the seven steps explicitly. The frame rate passed to OASIS is the per-plane rate (see 2-e), which the AI verified against the behaviour timestamps rather than the NWB `rate` attribute.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper: (1) only Suite2p curated ROIs (`iscell[:,0] == 1`) are processed; (2) putative interneurons — cells whose dF/F correlates with running speed at Pearson r > 0.5 over all in-lap samples — are dropped. Result: 909.62 ± 447.96 kept neurons per session, 138,262 total; interneuron exclusion 0.360% ± 0.615% of curated cells. ROI rows of `PlaneSegmentation` are mapped to planes by a running row offset (`planeIdx` is read but unused).

ii.
```python
            plane_iscell = iscell[plane_slice, 0] == 1
            curated_plane_indices = np.where(plane_iscell)[0]
...
                valid_mask = np.isfinite(dff[0])
                speed_valid = speed[valid_mask]
                dff_valid = dff[:, valid_mask]
                block_corr = np.array([
                        np.corrcoef(trace, speed_valid)[0, 1]
                        if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0 else np.nan
                        for trace in dff_valid], dtype=np.float32)
                block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
                keep_block = ~block_interneuron
```

iii. The AI located `spatial.is_putative_interneuron` and the `int_thresh = 0.5` used by `dayData`, and treated the reported exclusion rate as a calibration target: after the one-session smoke test gave 3.8% it explicitly checked whether that was an outlier ("my current interneuron exclusion rate for that test session is higher than the paper's reported average"), ran 5 sessions (1.77%), and accepted the full-run value of 0.360% ± 0.615%, noting in the notes that this "is close to the methods value of about 0.42% ± 0.85%".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which requires no extra work: neural frames and behaviour samples are the same samples in these NWB files (one behaviour sample per imaging frame, shared timestamps), so each trial's neural matrix is just `events[:, trial_start_idx : teleport_idx]`. `metadata['temporal_alignment_event'] = 'trial start'`, `off_start = 0.0`, `off_end = None` (trials have variable length).

ii.
```python
                for out_idx, trial_idx in enumerate(kept_trial_indices):
                    start = trial_start_idx[trial_idx]
                    stop = trial_stop_idx[trial_idx]
                    session_trial_blocks[out_idx].append(
                        kept_events[:, start:stop].astype(np.float16, copy=False))
```

iii. The AI verified in the early exploration that "behavior is already aligned sample-by-sample to imaging frames" (every behaviour `TimeSeries` carries the same 33,913 timestamps as the fluorescence array in the example session), so no resampling or offsetting is needed; sessions where the two differ by one frame are truncated to the common length (see 12).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning: data stay at the native acquisition resolution, 1/15.5078125 s = 64.4836 ms per bin, identical for every session. The rate is *inferred from the behaviour timestamps* (median inter-sample interval) rather than taken from the NWB `rate` attribute, because on two-plane sessions the stored `rate` is the scanner rate (31.015625 Hz) while the per-plane sampling rate is half that. The inferred rate is used both for the OASIS kernel and for `metadata['time_bin_size']` (mean across sessions, all identical).

ii.
```python
def infer_frame_rate(position_timestamps: np.ndarray) -> float:
    diffs = np.diff(position_timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs))
...
    mean_frame_rate = float(np.mean([stats["frame_rate_hz"] for stats in session_stats]))
    dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
```

iii. From the notes: "Frame rate is inferred from the behavior timestamps, not trusted from the NWB `starting_time/rate` metadata. This matters for multi-plane sessions where metadata can report `31.015625` even though the effective imaging/behavior sample spacing is `15.5078125` Hz." The full run reported `Frame rate mean±sd (Hz): 15.507813 ± 0.000000`, confirming a single common bin size across all 152 sessions, so no rebinning was needed to satisfy the "same bin size for all trials and sessions" requirement.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` of the behaviour `position` time series (all behaviour streams share the same timestamps; the AI truncates them to the common session length first).

ii.
```python
        timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
        ...
        timestamps = timestamps[:session_len]
```

iii. The AI's inspection of the NWB showed every behaviour `TimeSeries` carrying an identical timestamp vector of the same length as the imaging frames, so any of them can serve; `position` is the stream it also uses for the position outputs.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the trial's first timestamp, giving a per-sample time in seconds starting at 0. Stored as row 0 of the `(4, T)` input matrix, `float32`.

ii.
```python
            trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
            trial_input = np.vstack([
                    trial_times,
                    np.full(trial_times.shape, float(label.env), dtype=np.float32),
                    np.full(trial_times.shape, trial_number_series[start], dtype=np.float32),
                    np.full(trial_times.shape, prev_reward_outcomes[trial_idx], dtype=np.float32)])
```

iii. Direct reading of the decoder spec ("Time from start of trial in seconds (continuous, time-varying)") combined with the trial-start alignment; the notes state "`input[0]` is `time_from_trial_start_s`, computed from the behavior timestamps relative to the first frame of each trial."

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `[start:stop)` sample index range is used for the timestamps and for the neural events, and every session's streams are truncated to a common length beforehand, so input row 0 has exactly the same number of timepoints as the neural matrix.

ii.
```python
        session_len = min(len(position), len(speed), len(lick_counts), len(env_timeseries),
                          len(trial_number_series), len(trial_start_series),
                          len(teleport_series), len(timestamps), *plane_lengths)
...
            start = trial_start_idx[trial_idx]
            stop = trial_stop_idx[trial_idx]
            trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Same reasoning as 2-d: behaviour and imaging are sample-synchronous in these files, so index-based slicing is the alignment. The decoder's format checker (which errors if input and neural timepoint counts differ) passed with no warnings on all 152 sessions.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Primarily the VR scene name in the NWB `identifier` (e.g. `/data/InVivoDA/GCAMP3/08_10_2022/Env1_C_to_Env2_B`): `Env1 -> 0`, `Env2 -> 1`, with the environment switching at trial index 30 on `EnvX_?_to_EnvY_?` sessions. The behaviour `environment` stream is used as a cross-check on every trial (the run aborts on any mismatch).

ii.
```python
SCENE_LOCATION_RE = re.compile(r"^Env(?P<env>\d+)_Location(?P<zone>[ABC])$")
SCENE_SWITCH_RE = re.compile(r"^Env(?P<env>\d+)_Location(?P<zone0>[ABC])_to_(?P<zone1>[ABC])$")
SCENE_ENV_SWITCH_RE = re.compile(r"^Env(?P<env0>\d+)_(?P<zone0>[ABC])_to_Env(?P<env1>\d+)_(?P<zone1>[ABC])$")
...
        return [TrialLabel(env=env0 if trial_idx < switch_trial_count else env1,
                           zone=zone0 if trial_idx < switch_trial_count else zone1)
                for trial_idx in range(n_trials)]
```
```python
            env_in_trial = env_timeseries[start:stop]
            env_in_trial = env_in_trial[env_in_trial >= 0]
            if env_in_trial.size:
                env_mode = int(round(float(np.median(env_in_trial))))
                if env_mode != label.env:
                    raise RuntimeError(f"Environment mismatch in ... trial {trial_idx + 1}: "
                                       f"scene-derived {label.env}, behavior-derived {env_mode}")
```

iii. The AI found the repo's `env_morph_dict = {'Env1': 0, 'Env2': 1, 'Env3': 0.5}` and the scene-based labelling in `behavior.get_reward_zones`, and separately printed the `environment` stream for every day-8 session, confirming values `{-1, 0, 1}` with `-1` marking inter-trial samples. It therefore used the scene as the authoritative label and the stream as a validation signal; the notes say "The scene-derived environment was cross-checked against the aligned behavior `environment` stream on every kept trial."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `int(EnvN) - 1` gives 0/1; the value is broadcast as a constant across all timepoints of the trial into row 1 of the input matrix. Inter-trial `-1` samples never enter a trial, and are excluded from the cross-check median.

ii.
```python
        env = int(match.group("env")) - 1
        ...
        np.full(trial_times.shape, float(label.env), dtype=np.float32),
```

iii. The instructions require a binary per-trial ENV1/ENV2 input; the repo's `env_morph_dict` fixes the 0/1 coding. Constant-per-trial broadcasting is used for all per-trial variables so every input/output row has `T` timepoints.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The behaviour `trial number` time series, sampled at the trial's start index (i.e. the recorded within-session lap index, 0-based).

ii.
```python
        trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
        ...
        np.full(trial_times.shape, trial_number_series[start], dtype=np.float32),
```

iii. The AI explicitly checked this stream against the `trial_start` events before using it (step 60 of the trajectory): printing `trial number` at the start indices of trials 1, 2, 3, 30, 31, 80 gave 0, 1, 2, 29, 30, 79 — i.e. the recorded trial number at each `trial_start` is exactly the sequential lap index. (Away from laps the stream is `-1`, which is why it is sampled at the start index rather than used wholesale.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond sampling at the trial start and broadcasting the constant across the trial's timepoints (row 2, `float32`, values 0..99 in the released dataset). The number is *not* renumbered after the lick-error exclusion, so the value keeps its meaning as the animal's lap index within the session.

ii.
```python
            trial_input = np.vstack([
                    trial_times,
                    np.full(trial_times.shape, float(label.env), dtype=np.float32),
                    np.full(trial_times.shape, trial_number_series[start], dtype=np.float32),
                    np.full(trial_times.shape, prev_reward_outcomes[trial_idx], dtype=np.float32)])
```

iii. The instructions ask for "Trial number (continuous, per trial)"; the recorded lap index is the direct realisation of that, and the AI verified it coincides with the sequential index (see 5-a).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The timestamps of the `Reward` behaviour time series (a sparse event series with its own timestamps, ~71 events per session), compared against the start/stop *times* of each trial.

ii.
```python
        reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
...
        reward_outcomes = reward_outcome_per_trial(reward_timestamps,
                                                   timestamps[trial_start_idx],
                                                   timestamps[trial_stop_idx])
        prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. The AI's NWB inspection showed `Reward` as a 71-sample series with its own timestamp vector (unit mL, constant 0.004), i.e. event times rather than a per-frame stream, so it is matched to trials in the time domain rather than by sample index.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A per-trial binary reward indicator is computed by sweeping the sorted reward times against the trial windows `[t_start, t_stop)`; the vector is then shifted by one trial, with 0 for the first trial of a session. Crucially the shift is done on the *original* trial indexing, before lick-error trials are dropped, so a kept trial that follows a dropped trial still reports that dropped trial's true outcome. The value is broadcast across the trial (row 3).

ii.
```python
def reward_outcome_per_trial(reward_timestamps, start_times, stop_times):
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    reward_idx = 0
    for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
        while reward_idx < len(reward_timestamps) and reward_timestamps[reward_idx] < start_time:
            reward_idx += 1
        probe_idx = reward_idx
        while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
            outcomes[trial_idx] = 1
            probe_idx += 1
    return outcomes
...
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. The instructions define the input as binary (omitted = 0, rewarded = 1) per trial; the paper omits reward on ~15% of trials, and the converted dataset reproduces that (15.8% omitted). The notes state the ordering decision explicitly: "Reward outcome and previous-trial outcome were computed before removing these trials, so a kept trial still uses the actual previous trial's reward outcome even if that previous trial was later excluded for lick-sensor artifacts."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The behaviour `position` stream together with the reward-zone boundaries for that trial. The zone identity comes from the VR scene name (A/B/C, switching at trial 30 on switch sessions) and the boundaries from a hard-coded dictionary A = [80, 130], B = [200, 250], C = [320, 370] cm. The noisy per-sample `reward_zone` stream is not used.

ii.
```python
ZONE_TO_BOUNDS_CM = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
            label = trial_labels[trial_idx]
            zone_idx = ZONE_TO_INDEX[label.zone]
            zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
...
                    discretize_distance_to_zone(trial_pos, zone_bounds),
```

iii. This mirrors the paper's own function, which the AI read in the repo: `behavior.get_reward_zones(sess, rz_dict=None, change_trial=30)` sets the zone "based on scene ID" with `reward_zone_dict` entries `X: [80,130]`, `Y: [200,250]`, `Z: [320,370]` and `map_labels = {'A':'X','B':'Y','C':'Z'}`. The AI also printed the raw `reward_zone` stream for several trials and saw it fire on only a handful of samples per trial (and not at all on some trials), which is why the scene metadata rather than the stream is used as the label source.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to the 0-450 cm track. Signed distance to the nearest edge of the active zone: negative before the zone (`pos - zone_start`), exactly 0 anywhere inside it, positive past it (`pos - zone_stop`). The continuous distance is then discretized (7-c). Stored as row 0 of the `(6, T)` `int8` output matrix.

ii.
```python
def discretize_distance_to_zone(position_cm, zone_bounds):
    zone_start, zone_stop = zone_bounds
    signed_distance = np.where(position_cm < zone_start, position_cm - zone_start,
                      np.where(position_cm > zone_stop, position_cm - zone_stop, 0.0))
```
```python
            trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The notes describe the rule exactly ("Signed nearest distance to the active reward zone / Before zone: `position - zone_start` / In zone: `0` / After zone: `position - zone_end`"), which implements the instruction's "distance to *any* location in the reward zone" — the 50 cm zone is one class (`0 cm`) rather than a point.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks: `< -50`; `[-50, -10]`; `(-10, 0)`; exactly `0` (inside the zone); `(0, 10]`; `(10, 50]`; `> 50`. Labels are stored in `output_values[0]`.

ii.
```python
    bins = np.zeros_like(position_cm, dtype=np.int8)
    bins[signed_distance < -50.0] = 0
    bins[(signed_distance >= -50.0) & (signed_distance <= -10.0)] = 1
    bins[(signed_distance > -10.0) & (signed_distance < 0.0)] = 2
    bins[signed_distance == 0.0] = 3
    bins[(signed_distance > 0.0) & (signed_distance <= 10.0)] = 4
    bins[(signed_distance > 10.0) & (signed_distance <= 50.0)] = 5
    bins[signed_distance > 50.0] = 6
    return bins
```

iii. Straight transcription of the instruction's bin table; the notes say "Binned into the requested `7` categories". The resulting class distribution over the full dataset is {0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243}.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No separate alignment: position is sliced with the same `[start:stop)` index range as the neural events, after the whole-session truncation to a common length.

ii.
```python
            trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
            ...
            trial_output = np.vstack([discretize_distance_to_zone(trial_pos, zone_bounds), ...])
```

iii. As in 2-d/3-c, behaviour samples and imaging frames are one-to-one in these files, so index slicing is the alignment; the decoder's verifier confirmed equal timepoint counts for neural/input/output on every trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The behaviour `position` stream (cm along the 450 cm VR corridor).

ii.
```python
        position = behavior["position"]["data"][:].astype(np.float32)
        ...
        position = position[:session_len]
```

iii. The stream is in cm and was checked in the survey: inside laps it spans ~0-450 cm, while inter-trial samples sit at -500/-50 (the teleport zone), which is why only in-lap samples are converted.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Clip to [0, 449.999999] then integer-divide by 90 to get one of five equal 90 cm bins (with a redundant guard clamping anything above 4). Row 1 of the output matrix.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins
```

iii. The notes: "Corridor clipped to `0` to `450` cm / Binned into `5` equal bins of `90` cm each". Clipping keeps the handful of samples that fall marginally outside the track (interpolation at lap start/end) inside the first/last bin instead of creating out-of-range classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five bins: `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, `[360,450]`, named in `output_values[1]`; realised distribution {0.212, 0.177, 0.231, 0.226, 0.154}.

ii.
```python
    bins = np.floor(clipped / 90.0).astype(np.int8)
...
        "output_values": [ ...,
            ["0_to_90cm", "90_to_180cm", "180_to_270cm", "270_to_360cm", "360_to_450cm"], ...]
```

iii. Directly from the instructions ("Discretized into 5 equal-sized bins spanning the 450 cm track"), with the track length taken from the paper's Methods.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start:stop)` slice as the neural data; no resampling.

ii.
```python
            trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. See 2-d; sample-synchronous streams.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The behaviour `lick` stream (a per-sample cumulative lick count, integer values 0-6 in the sessions inspected).

ii.
```python
        lick_counts = behavior["lick"]["data"][:].astype(np.float32)
```

iii. The AI inspected the stream's value distribution before using it (`lick unique sample [0..6] max 6.0 n>1 4161 n>2 2717` for `sub-m3_ses-01`), which is also what motivated the sensor-error trial filter in 1-e.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized at `> 0` per sample (row 3, `int8`). Trials flagged as lick-sensor errors are removed altogether (1-e) rather than having their lick trace blanked. Full-dataset distribution: 77.7% no / 22.3% yes.

ii.
```python
            trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The instructions require a binary lick output; because the raw values are cumulative counts (>1 on many samples), thresholding at >0 is the natural binarisation. The notes: "`lick` — Binary `lick_count > 0`".

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop)` index slice as the neural data.

ii.
```python
            trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. See 2-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The VR scene name parsed from the NWB `identifier`, with the zone switching at trial index 30 on switch sessions; encoded A = 0, B = 1, C = 2. The per-sample `reward_zone` behaviour stream is not used.

ii.
```python
ZONE_TO_INDEX = {"A": 0, "B": 1, "C": 2}
...
    match = SCENE_SWITCH_RE.match(scene)
    if match:
        ...
        return [TrialLabel(env=env, zone=zone0 if trial_idx < switch_trial_count else zone1)
                for trial_idx in range(n_trials)]
...
                    np.full(trial_times.shape, zone_idx, dtype=np.int8),
```

iii. The AI copied the paper's own scheme: `behavior.get_reward_zones(..., change_trial=30)` assigns the zone from the scene ID and switches at trial 30, and the paper describes the reward zone being moved mid-session on switch days. The notes record "For switch sessions, the switch point is trial `30`, matching the paper/task structure", and the resulting labels are almost exactly balanced across the dataset (A 0.332, B 0.336, C 0.333), which the AI used as a sanity check.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene string -> regex -> per-trial zone letter -> index 0/1/2, broadcast across the trial's timepoints (row 4). An unrecognised scene raises, so no session can be silently mislabelled.

ii.
```python
    raise ValueError(f"Unrecognized scene format: {scene}")
...
            zone_idx = ZONE_TO_INDEX[label.zone]
...
                    np.full(trial_times.shape, zone_idx, dtype=np.int8),
```

iii. The three regexes cover the 26 distinct scene strings in the dataset (`EnvN_LocationX`, `EnvN_LocationX_to_Y`, `EnvN_X_to_EnvM_Y`). The AI listed the scene of every day-8 file to confirm the environment-switch naming before writing the parser.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event timestamps, the same source as the previous-trial-outcome input.

ii.
```python
        reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
        reward_outcomes = reward_outcome_per_trial(reward_timestamps,
                                                   timestamps[trial_start_idx],
                                                   timestamps[trial_stop_idx])
```

iii. See 6-a: `Reward` is an event series with its own timestamps and a constant 0.004 mL amount, so the only informative content is whether an event fell inside a trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. 1 if at least one reward event time falls in `[t_start, t_stop)` of that trial, else 0, broadcast across the trial (row 5, `int8`). Full dataset: 84.2% rewarded / 15.8% omitted, matching the paper's ~15% omission rate.

ii.
```python
                    np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8),
```

iii. The notes: "Trial-level label from reward timestamps falling between trial start and teleport / Repeated across timepoints in the trial." The AI cited the resulting omission fraction as a check against the paper's stated ~15% random omission.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms:
- **Stream length mismatch**: every behaviour array, the timestamp vector and all fluorescence/neuropil planes are truncated to their common minimum length before anything else. This was added after the first full run crashed on a two-plane session whose behaviour and ophys streams differ by one frame (10 sessions in the dataset are off by one frame).
- **Inter-trial samples**: `environment == -1` samples are excluded from the environment cross-check; only in-lap samples ever enter a trial, so the teleport-zone position values (-500/-50) never reach the outputs.
- **Out-of-range position**: clipped into [0, 450] before discretisation, so the few interpolated samples slightly outside the track fall in the end bins.
- **Hard failures rather than silent corruption**: an unparsable scene, a scene/behaviour environment disagreement, or a session where every neuron was filtered out raises instead of writing bad data.
No NaN ever reaches the output (the decoder's verifier rejects NaN/Inf, and the full dataset passed with no errors or warnings). There is no minimum-trial-length filter; in practice the shortest trial in the released data is 96 samples.

ii.
```python
        session_len = min(len(position), len(speed), len(lick_counts), len(env_timeseries),
                          len(trial_number_series), len(trial_start_series),
                          len(teleport_series), len(timestamps), *plane_lengths)
```
```python
            if not block_list:
                raise RuntimeError(f"No neurons kept for {session_meta.file_path.name}, trial {trial_idx + 1}")
```
```python
                if env_mode != label.env:
                    raise RuntimeError(f"Environment mismatch ...")
```

iii. The truncation was a direct response to a real failure the AI hit mid-run: "The first full pass exposed an off-by-one mismatch in one multi-plane session: the behavior streams and fluorescence streams differ by one frame there ... I'm patching the converter to truncate every session to the common minimum aligned length across behavior and ophys before retrying." The notes document the same. The raising checks are deliberate ("the scene-derived environment was cross-checked against the aligned behavior `environment` stream on every kept trial").

## 13-a. What are the most time-consuming steps of the code?

i. In order: (1) reading F and Fneu out of HDF5 — two ~30k x ~1-5k float arrays per session, read in 256-ROI column blocks, which is I/O bound and dominates; (2) the per-trial dF/F pipeline — the sigma-15 Gaussian and the two 300-sample rank filters plus OASIS deconvolution, run once per trial per block (~80 trials x ~4-20 blocks per session) on all curated cells; (3) the per-cell speed-correlation loop; (4) pickling the 4.8 GB output (and reloading it for verification/decoder training). The full conversion ran for tens of minutes at ~2.3 GB RSS.

ii.
```python
                fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
                neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
                dff, events = compute_dff_and_events(...)
```

iii. The AI explicitly designed around these costs: block-wise reads "so I can recompute the paper's deconvolved signal without holding the full dataset in memory at once", monitored RSS during the run ("sitting around 2.3 GB RSS, so the blockwise approach is doing its job"), and noted that verification is "mostly I/O-bound because it has to load a 4.5 GB pickle".

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Candidates, in rough order of payoff:
- The per-cell Pearson correlation list comprehension in `process_session` — this is a pure matrix operation (centre both, one `dff_valid @ speed_valid` product) and is currently a Python loop over up to a few thousand cells per session.
- `bad_lick_trial_mask` and `reward_outcome_per_trial` — both are Python loops over trials that reduce to `np.add.reduceat` / `np.searchsorted` one-liners (they are cheap, so the gain is small).
- The trial-emission loop inside the block loop, which slices and copies each trial for each block; the copies could be deferred to a single `np.concatenate` per trial over block views.
- The per-trial loops in `compute_dff_and_events` cannot be vectorized straightforwardly, because the baseline, smoothing and OASIS must not run across trial boundaries and trials have unequal length (the repo's own implementation loops the same way).

ii.
```python
                block_corr = np.array([
                        np.corrcoef(trace, speed_valid)[0, 1]
                        if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0 else np.nan
                        for trace in dff_valid], dtype=np.float32)
```

iii. Not discussed in the trajectory or notes; the AI's stated optimisation goal was memory ("materialize trial arrays only after neuron filtering so the full run stays memory-safe"), not per-loop speed, and the conversion finished in one pass without needing further tuning.

## 13-c. What processing does the code repeat multiple times?

i. Little: each NWB file is opened exactly once and each behaviour stream is read once, so there is no survey/convert double pass. What does repeat:
- The trial-slicing loop runs once per ROI block instead of once per session, so each trial's neural data is assembled from `n_blocks` slices that are then concatenated (extra copies, not extra computation).
- Events are materialised as `float32` and then cast to `float16` per block per trial, i.e. two buffers per trial.
- `np.nanstd(speed_valid)` is recomputed inside the per-cell comprehension for every cell.
- The summary statistics are computed at conversion time and again re-derived from `metadata['session_info']` when the sample pickle is built.

ii.
```python
                for out_idx, trial_idx in enumerate(kept_trial_indices):
                    start = trial_start_idx[trial_idx]
                    stop = trial_stop_idx[trial_idx]
                    session_trial_blocks[out_idx].append(
                        kept_events[:, start:stop].astype(np.float16, copy=False))
...
            neural_trial = np.concatenate(block_list, axis=0).astype(np.float16, copy=False)
```

iii. The single-pass design is deliberate and memory-driven (see 13-a); blocking the ROI axis is what forces the per-block trial slicing, which the AI accepted as the cost of bounded memory.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A modest amount:
- The full dF/F trace is computed for every curated cell but is only consumed by the speed-correlation test; it is discarded afterwards (it is, however, needed for that test, so it is not purely wasted).
- OASIS deconvolution is run on all curated cells, including the putative interneurons that are immediately dropped (~0.36% of cells).
- `imgseg["planeIdx"]` is read but never used (plane membership is inferred from a running row offset instead), and `import math` is unused.
- `sample_data.pkl` (33 MB), `sample_trials.png`, and the per-session `stats` dict embedded in `metadata['session_info']` are produced for validation/documentation and are not used by the decoder.
- Reward outcomes, previous-trial outcomes and scene labels are computed for all trials, including the 81 that are then dropped (this is intentional — the previous-trial outcome must see dropped trials).

ii.
```python
        plane_idx = imgseg["planeIdx"][:].astype(np.int16)   # never used again
```
```python
        sample_dataset = make_sample_dataset(full_dataset, SAMPLE_SESSION_KEYS, trials_per_session=10)
        save_pickle(args.sample_output, sample_dataset)
```

iii. The sample dataset and the logged statistics were an explicit part of the AI's workflow ("I'll log sanity checks directly from the conversion run so the full/sample outputs are audit-friendly"; the sample pickle exists so the verifier and decoder could be exercised quickly before and after the full run). The unused `planeIdx` read and `math` import are simple leftovers.
