# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every NWB file under a *relative* `data/` path with `Path('data').rglob('*.nwb')` (152 files, 11 subject directories) and converts one file per session. Files are opened directly with `h5py` rather than with `pynwb`. For each file it reads **every** behavioral time series (data + timestamps) under `processing/behavior/BehavioralTimeSeries`, and **one** neural ROI-response series under `processing/ophys`. Subject/session identity is read from `general/subject/subject_id` and `general/session_id` inside the file. Trials are reconstructed from the behavior streams (there is no NWB `intervals/trials` table).

Two loading gaps: (1) `load_neural_series` iterates over the ROI-response series of `Fluorescence` and `return`s on the **first** one, so for the 28 two-plane sessions (all of `sub-m17` and `sub-m18`) only `plane0` is loaded and `plane1` is silently discarded (e.g. `sub-m18_ses-01`: 1994 of 4857 ROIs loaded). This is confirmed by the per-session `n_neurons` in `verification_full_out.txt`, which match the plane0-only counts exactly. (2) The AI's own Step 2 notes count 312,110 ROIs in the dataset, but the converted file contains 260,091, and the discrepancy is never investigated.

ii.
```python
def load_neural_series(f):
    for base in ['processing/ophys/DfOverF', 'processing/ophys/Fluorescence']:
        if base in f:
            grp = f[base]
            for k in grp.keys():
                g = grp[k]
                if 'data' in g:
                    data = np.asarray(g['data'][:])
                    ts = np.asarray(g['timestamps'][:]) if 'timestamps' in g else None
                    return data, ts, base, k
    raise RuntimeError('No DfOverF or Fluorescence dataset found')

def load_behavior_series(f):
    grp = f['processing/behavior/BehavioralTimeSeries']
    out = {}
    for k in grp.keys():
        g = grp[k]
        if 'data' in g:
            out[k] = np.asarray(g['data'][:])
        if 'timestamps' in g:
            out[k + '__timestamps'] = np.asarray(g['timestamps'][:])
    return out
```
```python
    files = sorted(Path('data').rglob('*.nwb'))
    ...
    for fp in files:
        print('Converting', fp, flush=True)
        sess = convert_session(fp, show_processing=args.show_processing)
```
```python
        subj = decode_scalar(f['general/subject/subject_id'][()])
        sess = decode_scalar(f['general/session_id'][()])
```

iii. From CONVERSION_NOTES Step 2: "Data are organized as NWB files under `data/sub-*/sub-*_ses-*_behavior+ophys.nwb`. Each file corresponds to one subject/session and contains both behavior and optical physiology streams." Step 4 records that "Inspected NWB files do not expose standard `intervals/trials` tables → Reconstruct trials from behavioral time series variables in NWB." `h5py` was chosen implicitly (the trajectory shows the agent probing the files with `h5py` from the start and never trying `pynwb`). No justification is given anywhere for reading only one imaging plane — the notes never mention planes at all.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the `general/subject/subject_id` scalar inside each NWB file (not from the directory name). After all sessions are converted, the unique subject strings are sorted to form `data['subjects']`, and `subject_idx` is the index of each session's subject in that list. This yields the same 11 subjects (m11–m19, m3, m4, m7) as the reference.

ii.
```python
        subj = decode_scalar(f['general/subject/subject_id'][()])
    ...
    return {
        'subject': str(subj),
        ...
```
```python
    subjects = sorted(set(s['subject'] for s in sessions))
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    data = {
        ...
        'subjects': subjects,
        'subject_idx': np.asarray([subj_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 reports the resulting split ("Subjects | 11", "Sessions / subject | {'m11': 12, 'm12': 14, ...}"), and Step 9's consistency table cross-checks it against the paper: "Subjects | 11 mice | ... | 11 | yes". Reading the id from the file metadata rather than the path is not explicitly justified, but is the more authoritative source.

## 1-c. How are the data split into sessions?

i. One session per NWB file; the session label comes from `general/session_id` in the file. Session order in the output is the sorted file-path order. A session is dropped entirely if fewer than 2 trials survive (this never triggers — all 152 sessions are kept).

ii.
```python
        sess = decode_scalar(f['general/session_id'][()])
```
```python
        sess = convert_session(fp, show_processing=args.show_processing)
        if sess['n_trials'] >= 2:
            sessions.append(sess)
        else:
            print('Skipping session with <2 trials after processing:', fp, flush=True)
```

iii. Step 2: "Each file corresponds to one subject/session." The ≥2-trial rule follows the target-format requirement quoted in the instructions ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). Step 9 notes 152 sessions vs. the paper's 77 "switch" sessions and resolves this as "yes for full release; paper subset differs".

## 1-d. Are the data correctly split into trials?

i. Trials are the rising edges of the `trial_start` behavior signal, restricted to samples where the stored `trial number` is ≥ 0. **A trial runs from its own start index to the *next* trial's start index** (and the final trial runs to the end of the recording). The `teleport` signal is loaded into a local variable but is never used to end a trial.

Consequence: every trial contains the inter-trial interval / teleport ("black box") period in addition to the lap. On `sub-m11_ses-03` the median lap (trial_start → teleport) is 171.5 samples while the AI's median trial is 236 samples — ~27% of each trial is non-lap data. Across the dataset this raises mean T to 297.5 samples vs. the reference's 216.8, and the last trial of a session can run to the end of the recording (max T = 10,132 samples = 653 s, vs. reference max 3,359 samples = 217 s). Total trial count nonetheless matches the reference exactly (12,216).

ii.
```python
def rising_edges(x, thr=0.5):
    x = np.asarray(x)
    return np.where((x[1:] > thr) & (x[:-1] <= thr))[0] + 1


def infer_trial_bounds(b):
    trial_num = np.asarray(b['trial number'])
    trial_start = np.asarray(b['trial_start'])
    valid = trial_num >= 0
    starts = rising_edges(trial_start, 0.5)
    starts = starts[valid[starts]]
    if len(starts) == 0:
        changes = np.where(np.diff(trial_num) > 0)[0] + 1
        starts = changes[valid[changes]]
    trial_ids = trial_num[starts].astype(int)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_num)
        if e - s > 1:
            bounds.append((int(trial_ids[i]), int(s), int(e)))
    return bounds
```
```python
    teleport = np.asarray(b['teleport'])   # loaded but never used
```

iii. CONVERSION_NOTES Step 5: "`processing/behavior/BehavioralTimeSeries/trial_start` | trial segmentation | Rising edges define trial starts". Step 4 says "Reconstruct trials from behavioral time series variables in NWB". Step 5 Key Decision 4 states the intent to exclude teleport periods — "Teleport/out-of-track periods and pre-task periods (for example `trial number = -1`, `position = -500`) should be excluded" — but only the `trial number = -1` half of that decision was implemented. The trajectory shows the agent was aware of the `teleport` variable (step 320: "The BehavioralTimeSeries contains ... `teleport`, `trial number`, and `trial_start`") but it never revisited the trial-end definition.

## 1-e. How are trials filtered based on quality controls?

i. Two very weak filters only: a candidate trial is dropped if it spans ≤ 1 behavior sample (`e - s > 1` in `infer_trial_bounds`), and a trial is skipped if it contains fewer than 2 neural samples or fewer than 2 behavior samples. There is no minimum-duration criterion. In practice neither filter ever fires (min T in the converted data is 104 samples ≈ 6.7 s), so the outcome is identical to the reference's `< 50 timepoint` rule.

ii.
```python
        if e - s > 1:
            bounds.append((int(trial_ids[i]), int(s), int(e)))
```
```python
        nmask = (neural_t >= t0) & (neural_t <= t1)
        if np.sum(nmask) < 2 or (e - s) < 2:
            continue
```

iii. No justification is given in CONVERSION_NOTES; the notes' trial-curation section (Step 3) only says "Some session-level analyses in the paper include only sessions meeting convergence/significance criteria for remapping analyses" and "Rewarded vs omission trial labels are important and must be preserved". The ≥2-sample floor appears to be a defensive guard against degenerate slices rather than a documented quality criterion.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. The code prefers `processing/ophys/DfOverF` and falls back to `processing/ophys/Fluorescence`. **No file in this dataset contains `DfOverF`** (verified: 0/152), so in every session the neural data is the **raw suite2p fluorescence trace** `processing/ophys/Fluorescence/plane0/data`. `Neuropil` is never read, `Deconvolved` is never read, and (as noted in 1-a) `plane1` is never read for the 28 two-plane sessions. The array is oriented to (n_neurons, n_timepoints) by comparing its shape with the ROI count from `ImageSegmentation/PlaneSegmentation/id`, falling back to a "shorter axis is neurons" heuristic.

ii.
```python
def load_neural_series(f):
    for base in ['processing/ophys/DfOverF', 'processing/ophys/Fluorescence']:
        if base in f:
            ...
                    return data, ts, base, k
```
```python
    if n_rois is not None:
        if neural.shape[0] == n_rois:
            neural_nt = neural
        elif neural.shape[1] == n_rois:
            neural_nt = neural.T
        else:
            neural_nt = neural if neural.shape[0] < neural.shape[1] else neural.T
    else:
        neural_nt = neural if neural.shape[0] < neural.shape[1] else neural.T
```
```python
        neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "**Use processed imaging activity rather than raw fluorescence when available**: Reference code includes `dff` processing, so NWB `DfOverF` should be preferred if present." The variable-mapping table lists the source as "`processing/ophys/DfOverF/*/data` (or fluorescence-derived activity if needed) ... Prefer NWB `DfOverF` if present to match reference processed activity." The trajectory shows the agent repeatedly listed `DfOverF` among the keys it probed (steps 130, 316, 322, 329) but never recorded that it was absent, and the notes never state that raw fluorescence is what actually ended up in the file. The per-session `neural_source` string the code computes would have revealed this, but it is dropped before the final dictionary is written.

## 2-b. How is the `neural` data processed?

i. **No processing at all.** The raw fluorescence values are sliced per trial and cast to `float32`. There is no neuropil subtraction (`F - 0.7*Fneu`), no maximin baseline, no dF/F normalization, no Gaussian smoothing, and no OASIS deconvolution — i.e. none of the five steps the paper's Methods describe and the reference implements via the paper's own `preprocessing.dff()`.

ii.
```python
        neural_trial = neural_nt[:, nmask].astype(np.float32)
        ...
        session_neural.append(neural_trial)
```
(There is no other operation applied to `neural_nt` anywhere in the script.)

iii. No justification is offered, because the notes assume this step was unnecessary: Step 5 Key Decision 2 treats a stored `DfOverF` as equivalent to the paper's `dff` output. Since `DfOverF` does not exist, the fallback path runs with no substitute processing and this is never flagged. Step 10's "Reference-code comparison" claims only that "imaging preprocessing concepts were matched as closely as possible to the released NWB data" — no per-step comparison against `preprocessing.dff` was performed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neural curation whatsoever.** Every segmented ROI in `plane0` is kept. The suite2p `iscell` column is present in the files (`ImageSegmentation/PlaneSegmentation/iscell`) but is never read — on `sub-m11_ses-03` only 155 of 349 ROIs (44%) are flagged as cells, so the majority of the retained "neurons" are non-cell ROIs. The paper's putative-interneuron exclusion (dF/F vs. speed correlation r > 0.5) is also not applied, even though the AI itself identified `is_putative_interneuron` in Step 1. The result is 260,091 "neurons" (mean 1711/session) vs. the reference's 138,298 (mean 910/session).

ii.
```python
def load_n_rois_and_regions(f):
    n_rois = None
    regions = None
    if 'processing/ophys/ImageSegmentation' in f:
        for k in f['processing/ophys/ImageSegmentation'].keys():
            ps = f['processing/ophys/ImageSegmentation'][k]
            if 'id' in ps:
                n_rois = len(ps['id'])
            if 'location' in ps:
                ...
```
(`iscell` is never indexed; `n_rois` is used only to decide the orientation of the neural matrix.)
```python
    brain_region_idx = np.zeros(neural_nt.shape[0], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 1 lists `is_putative_interneuron` (spatial.py) and `get_cell_classes` (dayData.py) under "CURATION", and Step 3 notes "Reference code suggests cell-type/quality curation exists ... but exact inclusion criteria still need to be reconciled in later consistency checks." Step 4 then resolves the issue by deferring it: "Cell curation | Code contains cell classification/interneuron logic | NWB contains all segmented ROIs | ... | For conversion, start from all valid neural ROIs unless reference code indicates a required exclusion; verify later against methods/code." That verification never happened — Steps 10 and 12 do not revisit cell curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, as required. The `Fluorescence` series carries only `starting_time`/`rate` and no timestamps, so the code **synthesises** a neural time base with `np.linspace(bt[0], bt[-1], n_neural_samples)` over the behavior timestamp range, then selects neural samples with `t0 <= neural_t <= t1` where `t0`/`t1` are the behavior timestamps at the trial's first/last index. Behavioral variables are then re-indexed onto those neural samples with `np.searchsorted`. This is a nearest-neighbour remap rather than the reference's direct index correspondence: on `sub-m11_ses-03`, 6.7% of samples land one index away from the identity mapping (max deviation 1 sample). For the 10 sessions where the neural array has exactly one more frame than the behavior array, the linspace silently stretches the neural grid instead of cropping.

ii.
```python
    if neural_t is None:
        if bt is None:
            raise RuntimeError('No timestamps for neural or behavior')
        neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
```
```python
        t0 = bt[s]
        t1 = bt[e - 1]
        nmask = (neural_t >= t0) & (neural_t <= t1)
        if np.sum(nmask) < 2 or (e - s) < 2:
            continue
        nt = neural_t[nmask]
        bidx = np.searchsorted(bt, nt, side='left')
        bidx = np.clip(bidx, 0, len(bt) - 1)

        neural_trial = neural_nt[:, nmask].astype(np.float32)
        time_from_start = (nt - t0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 4: "temporal synchronization is available (behavior sampled at VR frame rate, imaging at ~15.5 Hz) → Preserve reference loading/curation, but represent each trial in time bins aligned to trial start as required by decoder task." Step 5 Key Decision 3: "**Represent trials in time bins aligned to trial start**: This differs from the paper's main position-binned analyses but is explicitly required by the decoder task." The choice to build the neural clock by interpolating the behavior timestamps (rather than using the stored `rate`) is not discussed.

## 2-e. How is the `neural` data temporally binned/resampled?

i. No rebinning — the data stays at the native per-plane imaging rate (15.5078 Hz, 64.48 ms/bin), which is the same as the behavior sampling rate, so bins are consistent across trials and sessions. However the metadata field that is supposed to report this is left as **NaN**: `'time_bin_size': float(np.nan)`. The reference reports 64.4836 ms. The script also never checks that the rate is actually the same across sessions.

ii.
```python
        'metadata': {
            'task_description': 'Virtual linear track reward-switch task; decode behavior and task variables from hippocampal imaging activity.',
            'time_bin_size': float(np.nan),
            'temporal_alignment_event': 'trial_start',
            'off_start': 0.0,
            'off_end': None,
            'n_sessions': len(sessions),
            'source_format': 'NWB',
        }
```

iii. Step 5 notes "Continuous time in seconds for each bin ... Same bin size for all sessions/trials". The NaN is acknowledged once in the trajectory (step 335: "the script is still provisional: `time_bin_size` is NaN, reward outcome inference is simplistic ...") but is never fixed, and Steps 9–12 of CONVERSION_NOTES do not mention it.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the `position` series' `timestamps` (all behavior series in these files share an identical timestamp vector, so the choice of series is immaterial), via the synthesised neural time grid `nt` described in 2-d. The reference uses the `trial number` series' timestamps.

ii.
```python
    bt = b.get('position__timestamps', None)
    if bt is None:
        any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
        bt = any_ts[0]
```
```python
        nt = neural_t[nmask]
        ...
        time_from_start = (nt - t0).astype(np.float32)
```

iii. Not explicitly justified. Step 5's mapping table lists "time from trial start | input: time from start of trial | Continuous time in seconds for each bin | decoder task requirement". The fallback to "any timestamp series" implies the AI assumed all behavior streams share a time base, which is true in these files.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the trial's first behavior timestamp from each neural sample time, cast to `float32`. Values start at ~0 for every trial. The observed maximum (653.3 s vs. the reference's 216.5 s) is a downstream consequence of the trial-end definition in 1-d, not of this computation.

ii.
```python
        t0 = bt[s]
        ...
        time_from_start = (nt - t0).astype(np.float32)
        ...
        inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
```

iii. Step 5: "Continuous time in seconds for each bin ... Same bin size for all sessions/trials." No further rationale is recorded; this is the obvious implementation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is defined *on* the neural sample times by construction (`nt - t0`), so it is aligned with the neural data by definition. The residual question is whether `nt` itself is faithful to the true imaging clock — see 2-d (linspace-synthesised grid, ≤1-sample jitter relative to index identity).

ii.
```python
        nmask = (neural_t >= t0) & (neural_t <= t1)
        nt = neural_t[nmask]
        neural_trial = neural_nt[:, nmask].astype(np.float32)
        time_from_start = (nt - t0).astype(np.float32)
```

iii. Step 4 records the understanding that the behavior and imaging streams are synchronised in the NWB release; building the input directly on the neural time vector guarantees equal lengths, which is what the format verification checks.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series, via a `map_environment` helper.

ii.
```python
    env = map_environment(np.asarray(b['environment']))
```
```python
        env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
```

iii. Step 5 mapping table: "`processing/behavior/BehavioralTimeSeries/environment` | input: environment type | Convert to binary ENV1 vs ENV2 per trial | ... | Need to map native values (for example ±1) to 0/1 consistently." The trajectory (step 360) confirms the agent checked the per-session distribution: "early sessions are environment 0, session 08 mixes 0 and 1, later sessions are environment 1."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `map_environment` collects the finite unique values in the session; if there are ≥2 distinct non-`-1` values it maps the largest to 1 and everything else to 0, otherwise it falls back to `(x > 0)`. Then, per trial, the **median** of the mapped values over the trial's behavior window is rounded to an integer and broadcast as a constant across all timepoints of the trial. The raw values in this dataset are already 0.0/1.0, so the mapping is the identity; the per-session normalisation is defensive scaffolding that is never exercised. The resulting per-session ranges in `verification_full_out.txt` reproduce the expected pattern (early sessions all 0, one transition session spanning 0→1, later sessions all 1).

ii.
```python
def map_environment(x):
    vals = np.unique(x[np.isfinite(x)])
    vals = [v for v in vals if v >= 0 or v == -1 or v == 1]
    uniq = sorted(set(v for v in vals if v != -1))
    if len(uniq) >= 2:
        lo, hi = uniq[0], uniq[-1]
        return np.where(x == hi, 1, 0)
    return (x > 0).astype(np.int64)
```
```python
        env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
```

iii. Step 5: "Convert to binary ENV1 vs ENV2 per trial ... Need to map native values (for example ±1) to 0/1 consistently." Step 9's consistency table: "Environment range | ENV1/ENV2 | environment variable present in code | NWB environment values 0/1 | 0/1 | yes." Taking the per-trial median (rather than the per-timepoint value) implements the instruction's "per trial" specification for this input.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The stored `trial number` behavior series, sampled at the trial's start index. (The reference deliberately used the loop counter instead, having found the stored variable inconsistent with `trial_start`; in practice the two agree here — the per-session trial-number ranges in `verification_full_out.txt` are identical to the reference's, e.g. [0, 79], [0, 89], [0, 99].)

ii.
```python
    trial_num = np.asarray(b['trial number']).astype(int)
    ...
    trial_ids = trial_num[starts].astype(int)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_num)
        if e - s > 1:
            bounds.append((int(trial_ids[i]), int(s), int(e)))
```

iii. Step 5 mapping table: "`processing/behavior/BehavioralTimeSeries/trial number` | input: trial number | Per-trial scalar broadcast across time bins | `add_trial_dict_info` | Use native trial numbering after reconstructing valid trials." The `trial number >= 0` condition is also used to exclude the pre-task period (Step 5 Key Decision 4).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the scalar `tid` across all timepoints of the trial as `float32`.

ii.
```python
        trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
        ...
        inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
```

iii. "Per-trial scalar broadcast across time bins" (Step 5), and Step 5 Key Decision 5: "**Broadcast per-trial covariates across trial time bins**: Environment, trial number, previous outcome, reward-zone location, and reward outcome can be stored as time-varying constant rows per trial for compatibility."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the `Reward` behavior time series' **event timestamps** (`Reward__timestamps`) — the same source the reference uses. The per-trial reward outcome is computed first (`infer_reward_outcomes_from_events`), and the previous trial's value is then carried forward.

ii.
```python
    reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
    reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. CONVERSION_NOTES Step 10, "Issues Found and Resolved": "Initial reward outcome inference from licks was incorrect (all rewarded). Resolved by using NWB `BehavioralTimeSeries/Reward` event timestamps." The trajectory (step 356) records the diagnosis: "every inspected trial had `teleport` sum 1 and substantial licking, so our reward outcome heuristic based on any lick is indeed too permissive and will classify all trials as rewarded."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A running `prev_out` variable, initialised to 0 before the first trial and updated to the current trial's reward outcome at the end of each loop iteration; broadcast as a constant across the trial. Note that `prev_out` tracks the previous *retained* trial, so if a trial were skipped the value would carry over from two trials back (in practice no trials are skipped).

ii.
```python
    prev_out = 0
    for i, (tid, s, e) in enumerate(trials):
        ...
        prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
        inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
        ...
        prev_out = int(reward_outcomes[i])
```

iii. Step 5: "Infer previous trial rewarded/omitted, encode 0/1, broadcast within current trial ... Must derive reward outcome per trial first." Matches the instruction's "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)"; 0 for the first trial follows the reference's convention.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` series and a **per-trial scalar reward centre** derived from the `reward_zone` series: for each trial, the median `position` over the samples where `reward_zone > 0`. The reference instead maps each trial to one of three *fixed zone intervals* (A = [80,130], B = [200,250], C = [320,370] cm) taken from the paper/reference code, and uses the interval's edges.

Consequences: (a) the target is a point rather than the 50 cm zone; (b) ~14% of trials (231/1600 in a 20-session check) have no `reward_zone > 0` sample at all, so the centre is NaN and the whole trial is silently assigned distance class 0.

ii.
```python
def infer_trial_reward_positions(pos, rz_signal, trials):
    trial_reward_pos = {}
    for tid, s, e in trials:
        sl = slice(s, e)
        nz = rz_signal[sl] > 0
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
    ...
```
```python
        reward_center = float(trial_reward_pos.get(tid, np.nan))
        dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. Trajectory step 366: "`reward_zone` values are not per-trial A/B/C labels; they are time-varying codes ... Therefore, our current use of `reward_zone` for per-trial reward-zone-location is incorrect. A better strategy is to infer the per-trial reward location from ... where nonzero reward_zone codes cluster within each trial." Step 373: "The trial-wise inferred reward positions clearly show a switch around trial 30, with one cluster near ~200 cm before the switch and another near ~80–110 cm after the switch. This strongly supports deriving per-trial reward location from the median position of nonzero `reward_zone` samples." The AI never consulted the paper's/reference code's explicit zone coordinates.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. `dist = position - reward_center`, i.e. signed distance to a single point, computed on the behavior samples re-indexed onto the neural grid. There is no notion of being "inside" a zone of finite width, so the instruction's class 3 (distance exactly 0) is essentially unreachable: it occurs in 0.2% of samples vs. 23.7% in the reference. NaN distances (missing reward centre) are mapped to class 0, the same class as "more than 50 cm before the zone". The resulting distribution is [0.540, 0.064, 0.068, 0.002, 0.077, 0.056, 0.194] vs. the reference's [0.253, 0.102, 0.074, 0.237, 0.021, 0.072, 0.243].

ii.
```python
def discretize_dist_to_reward(pos, reward_center):
    dist = pos - reward_center
    out = np.full(dist.shape, 0, dtype=np.int64)
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[np.isclose(dist, 0)] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    out[np.isnan(dist)] = 0
    return out, dist
```

iii. Step 5 mapping table: "`position` and `reward_zone` | output: distance to reward zone | Compute signed distance from current position to nearest reward-zone location; discretize into 7 bins | reward-zone behavior logic | Need mapping from reward-zone code to zone A/B/C position." The notes describe "distance to nearest reward-zone location", which the point-centre implementation approximates; the zone's spatial extent is never recovered and the near-empty class 3 is never flagged in Steps 10 or 12.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Hard-coded boolean masks on the signed distance, with edges matching the instruction (−50, −10, 0, +10, +50) and `np.isclose(dist, 0)` for the exact-zero class. The default class (used both for `dist < -50` and for NaN) is 0.

ii.
```python
    out = np.full(dist.shape, 0, dtype=np.int64)
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[np.isclose(dist, 0)] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    out[np.isnan(dist)] = 0
```
```python
        'output_values': [
            ['lt_-50', '-50_to_-10', '-10_to_0', '0', '0_to_10', '10_to_50', 'gt_50'],
```

iii. The class boundaries are transcribed directly from the Decoder Task specification in the instructions ("0: < −50 cm, 1: −50 to −10 cm, 2: −10 cm to < 0 cm, 3: 0 cm, 4: >0 cm to +10 cm, 5: +10 to +50 cm, 6: > +50 cm"), and `output_values` mirrors those labels. The NaN→class-0 fallback is not discussed in the notes.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is looked up at `bidx`, the `searchsorted` mapping of the neural sample times onto the behavior timestamps, so the output row has exactly the same length and time base as the neural matrix for that trial. Same caveats as 2-d (≤1-sample jitter; trial window extends through the inter-trial interval).

ii.
```python
        bidx = np.searchsorted(bt, nt, side='left')
        bidx = np.clip(bidx, 0, len(bt) - 1)
        ...
        dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
        out = np.vstack([
            dist_bins,
            ...
```

iii. Step 5 Key Decision 1: "**Use NWB behavior time series as the canonical alignment source**: The NWB files contain synchronized behavior streams." Step 10 reports a spot-check: "Sanity checks: `neural_allclose=True`, `input_allclose=True`, `output_allclose=True`" — though those checks compare the converted file against a fresh call to the AI's own `convert_session`, not against an independent reconstruction from the raw NWB.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series, re-indexed onto the neural grid via `bidx`. Same source as the reference.

ii.
```python
    pos = np.asarray(b['position'])
    ...
    abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
    ...
            abs_pos_bins[bidx],
```

iii. Step 5 mapping table: "`position` | output: absolute corridor position | Discretize corridor position into 5 equal bins | behavior trial matrices | Exclude teleport/out-of-track invalid periods if needed."

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No transformation of the position values themselves — they are used in raw cm. The only preparatory step is the `valid_mask = (trial_num >= 0) & (pos > -400)`, which is used solely to choose the discretisation range (it does **not** exclude those samples from any trial). Because trials run through the inter-trial interval (1-d), the teleport hold position of −50 cm is included in trials and lands in the first position bin.

ii.
```python
    valid_mask = (trial_num >= 0) & (pos > -400)
```
```python
def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges
```

iii. Step 5 Key Decision 4: "**Infer trial validity from behavior state variables**: Teleport/out-of-track periods and pre-task periods (for example `trial number = -1`, `position = -500`) should be excluded." The `pos > -400` clause implements the `-500` half of that (the pre-session sentinel), but the −50 teleport hold value survives the mask and the trial windows.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Into 5 bins of **equal width over the observed per-session position range**, i.e. `np.linspace(min(valid_pos), max(valid_pos), 6)`. On `sub-m11_ses-03` this gives edges [−50.0, 50.2, 150.3, 250.5, 350.6, 450.8] — 100.15 cm bins starting at −50 cm — rather than the instruction's fixed [90, 180, 270, 360] edges over the 450 cm track. The edges are therefore session-dependent (not comparable across sessions), the teleport hold position is pooled with track positions 0–50 cm in bin 0, and the resulting distribution is [0.327, 0.181, 0.185, 0.181, 0.126] vs. the reference's [0.211, 0.178, 0.231, 0.227, 0.154]. The `output_values` labels are the uninformative `pos_bin_0 … pos_bin_4`.

ii.
```python
def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges
```
```python
            [f'pos_bin_{i}' for i in range(5)],
```

iii. Step 5 mapping table says only "Discretize corridor position into 5 equal bins", which the code implements literally — but as equal bins over the *empirical* range rather than the *track*. The instruction's explicit edges ("0: < 90 cm, 1: 90 to 180 cm, 2: 180 to 270 cm, 3: 270 to 360 cm, 4: > 360 cm") are never quoted in CONVERSION_NOTES, and the mismatch is not caught in Steps 10 or 12.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identically to 7-d: the session-level bin array is indexed at `bidx`, giving one value per neural sample within the trial window.

ii.
```python
        bidx = np.searchsorted(bt, nt, side='left')
        bidx = np.clip(bidx, 0, len(bt) - 1)
        out = np.vstack([
            dist_bins,
            abs_pos_bins[bidx],
            ...
```

iii. Same rationale as 7-d — the behavior streams are treated as the canonical synchronised time base (Step 5 Key Decision 1), and the `--show-processing` plots (`processing_m11_03.png` etc.) were produced to visually check position/speed/lick/trial-number traces against the common time axis.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series (integer lick counts per sample, observed values 0–6), re-indexed at `bidx`.

ii.
```python
    lick = np.asarray(b['lick'])
    ...
            (lick[bidx] > 0.5).astype(np.int64),
```

iii. Step 5 mapping table: "`lick` | output: lick | Binary 0/1 time series | lick processing functions | Time-varying." Matches the instruction's "Lick, time-varying. 0 = no, 1 = yes."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation with a `> 0.5` threshold. Since the raw values are non-negative integers, this is exactly equivalent to the reference's `> 0`. The paper's `correct_lick_sensor_error` routine (which the AI itself catalogued in Step 1) is not applied — the reference does not apply it either. The delivered positive rate is 0.150 vs. the reference's 0.230; the gap is a consequence of including the (lick-free) inter-trial interval in every trial (1-d), not of this thresholding.

ii.
```python
            (lick[bidx] > 0.5).astype(np.int64),
```
```python
            ['no', 'yes'],
```

iii. Step 5: "Binary 0/1 time series". The instruction specifies a binary output; a threshold at 0.5 is the natural binarisation of a count signal.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Via the same `bidx` nearest-sample mapping onto the neural time grid; the row is the same length as the trial's neural matrix.

ii.
```python
        bidx = np.searchsorted(bt, nt, side='left')
        ...
            (lick[bidx] > 0.5).astype(np.int64),
```

iii. As in 7-d/8-d. Step 5's planned sanity checks included "Check that a few trial slices from converted `lick`, `speed`, and `position` match raw NWB values with `np.allclose()`", and Step 10 reports `output_allclose=True` (against the AI's own converter, not an independent reconstruction).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From `reward_zone` and `position`, via the same per-trial median in-zone position as 7-a, then clustered. Same two source variables as the reference.

ii.
```python
    trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
    trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
```

iii. Trajectory step 366 (quoted in 7-a) established that `reward_zone` is a time-varying code rather than an A/B/C label, motivating the position-based inference. Step 10: "Initial reward-zone-location mapping from raw `reward_zone` code was incorrect. Resolved by inferring per-trial reward positions from nonzero reward-zone samples and clustering into three canonical locations."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. **Per-session** k-means with `k = min(3, n_unique_values)` on the per-trial median in-zone positions; the cluster centres are sorted ascending and each trial is labelled by the index of the nearest centre; trials with no in-zone samples default to label 0. The label is broadcast across the trial.

Three problems: (1) labels are *session-relative ranks*, not the task's three fixed physical locations, so the same physical zone receives different labels in different sessions; (2) a session normally contains only two distinct reward locations (the zone switches after ~trial 30), yet `k` is effectively always 3 because the medians are continuous-valued, so one true zone is split into two clusters; (3) the ~14% of trials with no in-zone samples silently become "A". The resulting distribution is [0.613, 0.258, 0.128] versus the reference's near-uniform [0.329, 0.337, 0.334], and validation balanced accuracy is 0.511 vs. the reference's 0.841.

ii.
```python
    vals = np.array([v for v in trial_reward_pos.values() if np.isfinite(v)])
    if len(vals) == 0:
        centers = np.array([np.nan, np.nan, np.nan])
    else:
        k = min(3, len(np.unique(np.round(vals, 3))))
        km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
    return trial_reward_pos, centers

def assign_reward_location_labels(trial_reward_pos, centers):
    labels = {}
    finite_centers = [c for c in centers if np.isfinite(c)]
    for tid, rp in trial_reward_pos.items():
        if not np.isfinite(rp) or len(finite_centers) == 0:
            labels[tid] = 0
        else:
            labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
    return labels
```
```python
            np.full(nt.shape, int(trial_reward_loc.get(tid, 0)), dtype=np.int64),
        ...
            ['A', 'B', 'C'],
```

iii. Step 7: "Reward-zone per-trial location is inferred from positions of nonzero `reward_zone` samples and clustered into three canonical locations." Step 9's consistency table rates the result only as "plausible": "Reward zone location distribution | three locations A/B/C in task | ... | inferred from reward-zone/position dynamics | non-degenerate 0/1/2 | plausible." The heavily skewed 0.613/0.258/0.128 distribution was never compared against the task's symmetric A/B/C design, and the paper's fixed zone coordinates were never used to anchor the labels.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` behavior time series' event timestamps — the same source as the reference.

ii.
```python
    reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
    reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. Step 10: "Initial reward outcome inference from licks was incorrect (all rewarded). Resolved by using NWB `BehavioralTimeSeries/Reward` event timestamps." README: "Reward outcomes are derived from NWB `BehavioralTimeSeries/Reward` event timestamps."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, 1 if any reward-event timestamp falls in the closed interval `[bt[start], bt[end-1]]`, else 0; broadcast as a constant across the trial's timepoints. This is a direct time-window test rather than the reference's `searchsorted`-to-index mapping, and there is no assertion that reward times land within half a time bin of a behavior sample. Delivered distribution 0.189/0.811 vs. the reference's 0.157/0.843 (the difference follows from the longer trial windows, which reweight the per-timepoint fractions).

ii.
```python
def infer_reward_outcomes_from_events(trials, behavior_timestamps, reward_event_timestamps):
    outcomes = []
    for _, s, e in trials:
        t0 = behavior_timestamps[s]
        t1 = behavior_timestamps[e - 1]
        rewarded = int(np.any((reward_event_timestamps >= t0) & (reward_event_timestamps <= t1)))
        outcomes.append(rewarded)
    return np.asarray(outcomes, dtype=np.int64)
```
```python
            np.full(nt.shape, int(reward_outcomes[i]), dtype=np.int64),
        ...
            ['no', 'yes'],
```

iii. Step 5: "Rewarded vs omission per trial => 0/1 | `get_omission_trials` and behavior logic | Need exact derivation from NWB variables/signals." Step 9: "Reward outcome distribution | omission trials present | reward/omission handled in code | NWB `Reward` event stream available | non-degenerate 0/1 | yes."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is present but almost entirely **silent**, and in two cases missing data is coerced into a valid category rather than excluded:

- **Missing reward-zone data** (~14% of trials never register `reward_zone > 0`): the reward centre is NaN → the trial's *distance to reward zone* becomes class 0 ("< −50 cm") for every timepoint, and its *reward zone location* becomes 0 ("A"). No warning.
- **Pre-task / out-of-track samples**: `trial_num >= 0` excludes the pre-task block from trial starts, and `pos > -400` excludes the −500 sentinel from the position-binning range only.
- **Neural/behavior length mismatch** (10 sessions have exactly one extra imaging frame): handled implicitly by building `neural_t` as a linspace with `n_neural` points across the behavior timestamp span — i.e. the neural clock is stretched rather than cropped. No warning (the reference prints one and crops).
- **Degenerate trials/sessions**: trials with < 2 samples are skipped and sessions with < 2 trials are dropped, with a print only in the session case.
- **Robust scalar decoding** (`decode_scalar`) and a fallback timestamp source (`any_ts[0]`) guard against metadata format variation.
- **Missing brain-region metadata**: `ImageSegmentation/PlaneSegmentation` has no `location` field, so `regions` is always `None`; the code hard-codes `brain_regions = ['CA1']` (correct — `general/optophysiology/ImagingPlane/location` reads `hippocampus, CA1`).

ii.
```python
    valid_mask = (trial_num >= 0) & (pos > -400)
```
```python
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
```
```python
    out[np.isnan(dist)] = 0
```
```python
        if not np.isfinite(rp) or len(finite_centers) == 0:
            labels[tid] = 0
```
```python
    if neural_t is None:
        if bt is None:
            raise RuntimeError('No timestamps for neural or behavior')
        neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
```
```python
        if np.sum(nmask) < 2 or (e - s) < 2:
            continue
```

iii. Step 5 Key Decision 4 ("Infer trial validity from behavior state variables ... Teleport/out-of-track periods and pre-task periods ... should be excluded") is the stated policy. The NaN→class-0 and NaN→label-"A" fallbacks are not documented or justified anywhere, and Step 10's edge-case review reports no findings on them: "Key statistics comparison: subject/session/brain-region counts and trial structure are broadly consistent with paper/data."

## 13-a. What are the most time-consuming steps of the code?

i. The AI performed **no timing analysis**. The script contains no timing instrumentation of any kind (no `time.time()`, no per-step prints — only `print('Converting', fp)`), and the CONVERSION_NOTES Step 6 template fields are left as literal placeholders ("Code inefficiencies identified: [Note]", "Code speedups added: [Note]"), while Step 7's "Speed-ups Implemented / Time Savings" table is empty and the run-time estimate is the non-quantitative "fast (seconds per 2 sessions) | full dataset likely manageable in minutes". The instructions explicitly asked for timing output and a full-run estimate.

By inspection the bottlenecks are: (1) eagerly materialising each session's full ophys array with `np.asarray(g['data'][:])` (up to ~33k × 4k float32) and every behavior stream including unused ones; (2) `pickle.dump` of a 25 GB dictionary, all held in RAM simultaneously since every session is accumulated in `sessions` before writing; (3) in `--sample` mode, opening every one of the 152 NWB files just to pick two.

ii.
```python
    sessions = []
    for fp in files:
        print('Converting', fp, flush=True)
        sess = convert_session(fp, show_processing=args.show_processing)
        ...
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
```
(No timing calls appear anywhere in `convert_data.py`.)

iii. No justification is recorded. The trajectory shows the agent moved from Step 6 straight to sample conversion without profiling (step 335: "`convert_data.py` was created successfully and passes `py_compile`, so Step 6's basic existence/runnability criterion is met").

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Not analysed or documented by the AI. In the code as written: the per-trial loop in `convert_session` is the natural structure given variable-length trials (the reference makes the same choice); `infer_trial_reward_positions` and `infer_reward_outcomes_from_events` are separate Python loops over the same trial list that could be folded into one pass or vectorised with `np.add.reduceat` / `np.searchsorted` over the trial boundaries; the `--sample` selection loop opens and reads two arrays from every file in the dataset and could stop far earlier or be replaced with a cached lookup. `np.full(nt.shape, ...)` is called five times per trial to materialise constants that could be broadcast.

ii.
```python
    for tid, s, e in trials:
        sl = slice(s, e)
        nz = rz_signal[sl] > 0
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
```
```python
    for _, s, e in trials:
        t0 = behavior_timestamps[s]
        t1 = behavior_timestamps[e - 1]
        rewarded = int(np.any((reward_event_timestamps >= t0) & (reward_event_timestamps <= t1)))
```
```python
        for fp in files:
            with h5py.File(fp, 'r') as f:
                env = f['processing/behavior/BehavioralTimeSeries']['environment']['data'][:]
                tn = f['processing/behavior/BehavioralTimeSeries']['trial number']['data'][:]
```

iii. No justification recorded — the notes' efficiency sections were never filled in.

## 13-c. What processing does the code repeat multiple times?

i. Not documented by the AI. Observed repetition: `load_behavior_series` reads the `timestamps` dataset of **every** behavior series (11 series) even though they are identical and only one (`position__timestamps`) is used; it also loads `autoreward`, `scanning` and `teleport` data that are never consumed. `infer_trial_bounds`, `infer_trial_reward_positions` and `infer_reward_outcomes_from_events` each iterate the trial list separately before the main per-trial loop iterates it a fourth time. In `--sample` mode the whole file set is opened once for selection and the chosen files are then re-opened for conversion. At the whole-workflow level the full conversion was also re-run after each fix (the trajectory shows several full passes).

ii.
```python
    for k in grp.keys():
        g = grp[k]
        if 'data' in g:
            out[k] = np.asarray(g['data'][:])
        if 'timestamps' in g:
            out[k + '__timestamps'] = np.asarray(g['timestamps'][:])
```
```python
    trials = infer_trial_bounds(b)
    trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
    trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
    reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
    reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. No justification recorded.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Not documented by the AI. Present in the code:

- `choose_zone_centers()` is defined and **never called**.
- `discretize_dist_to_reward` returns `dist` alongside the bins; the caller discards it (`dist_bins, _ = ...`).
- `discretize_position` returns `pos_edges`, which is never used or saved — so the session-specific bin edges needed to interpret `pos_bin_0…4` are lost.
- `teleport`, `autoreward` and `scanning` arrays and the 11 near-duplicate timestamp vectors are loaded per session and never used.
- `load_n_rois_and_regions` parses ROI `location` strings that are absent from these files; the result is always `None` and `brain_regions` is hard-coded anyway.
- The per-session dict computes `zone_centers`, `neural_source` and `n_trials`, none of which reach the saved dictionary (`neural_source` in particular would have revealed the raw-fluorescence fallback).
- `speed_bins` and `abs_pos_bins` are discretised over the whole session, including the substantial fraction of samples that fall outside any trial window.
- Most consequentially, the output pickle is 25 GB because it stores raw `float32` fluorescence for all uncurated ROIs across trial windows padded with the inter-trial interval; with the reference's `iscell` + interneuron filtering and lap-only windows the same dataset is far smaller.

ii.
```python
def choose_zone_centers(pos, rz_code):   # never called
    centers = {}
    ...
```
```python
        dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```
```python
    abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```
```python
        'zone_centers': reward_loc_centers.tolist() if hasattr(reward_loc_centers, 'tolist') else reward_loc_centers,
        'neural_source': f'{neural_base}/{neural_key}',
```

iii. No justification recorded; CONVERSION_NOTES Step 13 only states "[x] All files organized".
