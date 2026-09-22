# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything comes from the three subfolders of `/app/data`. `beh/Imaging_Exp_info.npy` is read first as the master index: it is a dict keyed by experiment type, each holding a list of entries with `mname`, `datexp`, `blk`, `stimtype` and `stim_id`. `collect_sessions` walks that index and collapses it into one record per unique `(mouse, date, block)` recording (89 of them from 142 index entries), remembering for each the *first* `(exp_type, beh_key)` it was seen under plus **all** of its `stim_id` vectors. `load_behaviour` then opens each `Beh_<exp_type>.npy` exactly once and retains only the 16 fields listed in `BEH_FIELDS`, so the whole behaviour of the dataset stays small in RAM. Neural traces (`spk/<session>_neural_data.npy`) and the per-neuron area assignment (`retinotopy/<mouse>_<date>_trans.npz`) are loaded lazily, one session at a time, inside worker processes of a `multiprocessing.Pool` (8 by default, 12 for the production run).

ii.
```python
def collect_sessions(root):
    """One entry per unique recording, with merged stimulus-name -> canonical-id map."""
    exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    sessions = OrderedDict()
    for exp_type, db in exp_info.items():
        for d in db:
            kn = session_key(d)
            s = sessions.setdefault(kn, {...'beh_source': (exp_type, beh_key(d))})
```
```python
def load_behaviour(root, sessions):
    """Load each Beh_<exp_type>.npy once and keep only the fields we need."""
    for exp_type, kns in by_exp.items():
        B = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
        for kn in kns:
            b = B[sessions[kn]['beh_source'][1]]
            beh[kn] = {f: b[f] for f in BEH_FIELDS}
        del B
```
```python
    ret = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' %
                               (session['mname'], session['datexp'])), allow_pickle=True)
    spk_path = os.path.join(root, 'spk', '%s_neural_data.npy' % kn)
    spks = np.load(spk_path, allow_pickle=True).item()['spks']
```

iii. The file docstring states the layout and the mapping rule it inferred. From the final report: "`Imaging_Exp_info.npy` lists 142 entries but only 89 unique (mouse, date, block) recordings — matching the paper's '89 recordings in 19 mice' and the 89 spk files. I verified that the behaviour dicts for a recording are byte-identical across the analysis types it appears under, so I de-duplicated to one session per recording." Trajectory steps 40–46 show the agent enumerating the index, checking the duplicate entries and the `stimtype` sub-keys before committing to this scheme. Only the fields actually needed are kept because the full `Beh_*.npy` dicts have 60 fields per session.

## 1-b. How are the data split into subjects?

i. The mouse is `mname` from the index entry; it is carried on the session record and repeated in `session_info`. The subject list is the sorted unique set of mouse names (19), and `subject_idx` is each session's index into that list, appended in the same order the sessions are appended to `neural`/`input`/`output`. No subject is excluded. The output matches the expert exactly: 19 subjects, same names, same sessions-per-subject counts.

ii.
```python
def session_key(d):
    return '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
```
```python
subjects = sorted({s['mname'] for s in sessions.values()})
...
for kn, s in sessions.items():
    ...
    data['subject_idx'].append(subjects.index(s['mname']))
data['subjects'] = subjects
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. No derivation is needed — the index already names the mouse of every recording. The agent's report notes "89 recordings (= 89 sessions), 19 mice ... No sessions/mice excluded," cross-checked against the paper's "We performed 89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` recording, which is also the name of the spike file. The index lists several recordings more than once (33 recordings appear under 2+ experiment types, and 9 appear under two `stimtype` sub-keys `_swap1`/`_swap2`); `setdefault` in `collect_sessions` keeps the first occurrence and merges later occurrences into the same record. 89 sessions result, one per spk file.

ii.
```python
kn = session_key(d)
s = sessions.setdefault(kn, {
    'kn': kn, 'mname': d['mname'], 'datexp': d['datexp'], 'blk': d['blk'],
    'exp_types': [], 'stim_map': {}, 'beh_source': (exp_type, beh_key(d)),
})
if exp_type not in s['exp_types']:
    s['exp_types'].append(exp_type)
s['stim_id_entries'] = s.get('stim_id_entries', []) + \
    [(exp_type, beh_key(d), np.asarray(d['stim_id'], dtype=float))]
```

iii. Docstring: "one session per unique recording (mouse, date, block); the 89 recordings appear several times in Imaging_Exp_info under different analysis names but the behaviour dictionaries are identical, so they are de-duplicated." The list of experiment types a recording appears under is preserved in `metadata['session_info'][i]['experiment_types']` rather than being thrown away.

## 1-d. How are the data split into trials?

i. Trials are the ones the data declares: `ntrials` per session, with every imaging frame labelled by `ft_trInd`. A trial's frames are those frames that (a) carry that trial index, (b) are inside the textured corridor (`ft_CorrSpc`), and (c) are frames during which the virtual reality was advancing, i.e. the mouse was running (`ft_move > 0`). The grouping is done in one vectorised pass (argsort + searchsorted) rather than one scan per trial. Trials are therefore variable length; nothing is padded. Because non-running frames are dropped, a trial in this conversion is the *running* portion of one 4 m traversal, from corridor entry to the grey space: 11 to 178 frames, median 21, 821,579 frames over 38,110 trials (the expert's `ft_CorrSpc`-only selection gives 1,373,170 frames, median 23, max 5,607).

ii.
```python
def trial_frames(beh, nfr):
    """Frame indices per trial: running frames inside the textured corridor."""
    ft_trInd = beh['ft_trInd'][:nfr]
    valid = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr] & ~np.isnan(ft_trInd)
    frames = np.nonzero(valid)[0]
    tr = ft_trInd[frames].astype(np.int64)
    order = np.argsort(tr, kind='stable')
    frames, tr = frames[order], tr[order]
    bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
    return [frames[bounds[i]:bounds[i + 1]] for i in range(int(beh['ntrials']))]
```

iii. From the file docstring: "frames are kept only when the mouse was running, i.e. when the virtual reality was advancing (`ft_move > 0`) and the mouse was inside the textured part of the corridor (`ft_CorrSpc`). This is exactly the frame selection used throughout utils.py (`fr_valid = VRmove & isCorridor`) and described in the paper ('We only considered timepoints during running for analysis')." Both claims check out: `utils.py:431-433` and `methods.txt` line 3. The final report adds the decoder-specific argument: "Because the reference decoder classifies each timepoint independently, dropping stopped frames costs no temporal structure, and the two time inputs are computed from real frame timestamps so they stay correct across the gaps." Trial boundaries are documented in metadata as corridor entry (`Trial_start_time`/`StartFr`) to grey space (`GrayFr`).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. `session_behaviour_arrays` only guards against degenerate trials: fewer than 2 surviving frames, a non-finite `Trial_start_time` or `SoundTime`, or non-finite position/speed. In practice **none of these guards ever fires**: all 38,110 declared trials are kept (the expert keeps 37,728, dropping 382 traversals longer than the 99th percentile). There is no length-outlier filter, no session-level filter and no mouse-level filter; the agent relied on the `ft_move > 0` frame filter to make long trials harmless, which it does for the neural/position/speed streams (the longest surviving trial is 178 running frames), but not for the time inputs — 378 trials still span more than 75 s of wall-clock, up to 1,765 s.

ii.
```python
for trial, fr in enumerate(per_trial):
    if len(fr) < 2:
        continue
    t0 = beh['Trial_start_time'][trial]
    tcue = beh['SoundTime'][trial]
    if not np.isfinite(t0) or not np.isfinite(tcue):
        continue
    ...
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(to_cue))):
        continue
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(s))):
        continue
```

iii. Final report: "All 38,110 trials have ≥11 frames; none were dropped." The agent had measured the distribution first (trajectory step 68 prints `t since start pct [3.8 11.1 60.7 303.4 404.4 1765.2]`, `frac >60s 0.010`) and chose to keep the heavy tail, describing it as the animal's real stopping behaviour: "seconds since corridor entry (real elapsed time, so it carries the animal's stopping behaviour — heavy-tailed, median 3.8 s, max 29 min)". Sessions were not filtered either because every session retains 2,000 neurons and hundreds of trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<mouse>_<date>_<blk>_neural_data.npy`, a list of one `(neurons, frames)` array per imaging plane; neurons are indexed by concatenating the planes in order. The per-neuron visual area comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`. The two are cross-checked: the total neuron count in the spk file must equal the length of `iarea`.

ii.
```python
ret = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' %
                           (session['mname'], session['datexp'])), allow_pickle=True)
region = neu_area_idx(np.asarray(ret['iarea']))
n_total = len(region)
...
spks = np.load(spk_path, allow_pickle=True).item()['spks']
nfr = min(p.shape[1] for p in spks)
n_file = sum(p.shape[0] for p in spks)
if n_file != n_total:
    raise ValueError('%s: %d neurons in spk file but %d in retinotopy' % (kn, n_file, n_total))
```

iii. Docstring of `process_session`: "The spk file stores one array per imaging plane; neurons are indexed by concatenating the planes in order (utils.load_spk)." The count assertion is the agent's own check that the plane concatenation order really does line up with the retinotopy vector.

## 2-b. How is the `neural` data processed?

i. Not processed at all: the deconvolved traces are taken as they are, with no dF/F, no normalisation, no smoothing and no z-scoring. The only transformations are (a) selecting the chosen rows (neurons) and the chosen frame columns, and (b) storing as `float32` in per-trial contiguous blocks. Trials are left at their natural variable length.

ii.
```python
out = np.empty((len(rows), len(frames)), dtype=np.float32)
offset = 0
for plane in spks:
    n = plane.shape[0]
    sel = (rows >= offset) & (rows < offset + n)
    if sel.any():
        out[sel] = plane[rows[sel] - offset][:, frames]
    offset += n
del spks

neural = []
start = 0
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
    start += len(fr)
```

iii. Metadata: `'neural_signal': 'Suite2p deconvolved fluorescence (non-negative deconvolution, 0.75 s decay), one value per imaging frame, no normalisation'`, echoing `methods.txt` line 37 ("All our analyses were based on deconvolved fluorescence traces"). The final report repeats "Raw deconvolved traces, no normalisation."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) **Area filter**, identical to the expert's: a neuron is kept only if `iarea` maps into V1 (8), mHV (0,1,2,9), lHV (5,6) or aHV (3,4); `iarea` of −1 or 7 is dropped (~12.5% of ROIs). (2) **Random subsample**: of the surviving in-area neurons, at most `--max-neurons` (default **2000**) are drawn per session with a per-session seeded RNG, sorted, and kept. Every one of the 89 sessions therefore contributes exactly 2,000 neurons — 178,000 in total, against the expert's 4,105,393 (mean 46,128/session). No other neuron QC is applied.

ii.
```python
def neu_area_idx(iarea):
    """Region index per neuron, following utils.neu_area_ID. -1 = not one of the four
    area groups analysed in the paper (unassigned neurons, iarea==-1, and iarea==7)."""
    idx = np.full(len(iarea), -1, dtype=np.int64)
    idx[iarea == 8] = 0                                              # V1
    idx[np.isin(iarea, [0, 1, 2, 9])] = 1                            # medial HVAs
    idx[np.isin(iarea, [5, 6])] = 2                                  # lateral HVAs
    idx[np.isin(iarea, [3, 4])] = 3                                  # anterior HVAs
    return idx
```
```python
candidates = np.nonzero(region >= 0)[0]
rng = np.random.default_rng(seed)
if max_neurons is not None and len(candidates) > max_neurons:
    rows = np.sort(rng.choice(candidates, size=max_neurons, replace=False))
else:
    rows = candidates
```

iii. For the area filter: "neurons are restricted to the four visual-area groups used in the paper (V1, mHV, lHV, aHV; utils.neu_area_ID)"; the report adds "this drops the 12.5% of ROIs with `iarea` = −1 or 7, which no paper analysis uses". No further neuron QC because the authors already ran the Suite2p cell classifier. For the subsample, the justification is empirical and explicitly tied to the evaluation decoder: "The cap is empirically motivated: I converted a 4-session pilot at 500/1000/2000/3000/8000/20000 neurons — validation accuracy peaks at 2,000 (stimulus 0.96, position 0.85) and collapses above it (3,000 → 0.57/0.43), because `train_decoder` uses an exact SVD initialisation only up to `svd_max_neurons=2000` and a lossy random projection beyond. It also keeps the file at 6.6 GB instead of ~175 GB." The pilot outputs are in trajectory steps 75 and 79; `decoder.py:958-963` does contain the `svd_max_neurons=2000` random-projection branch. The subsample is recorded in `metadata['max_neurons_per_session']` and `metadata['neuron_subsample_seed']`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial's frames start at the first kept frame after `Trial_start_time`/`StartFr` and run to the last kept frame before the grey space, so every trial begins at its own corridor entry; `off_start = 0.0` and `off_end = None` because the traversal length is behaviour-dependent. Trials are stored at their natural length, with no common window, no truncation and no padding. All four streams (neural, inputs, outputs) are indexed with the identical `fr` frame-index array, so alignment is exact by construction.

ii.
```python
beh_arrays = session_behaviour_arrays(session, beh, nfr, day_of_training, speed_edges)
frames = np.concatenate(beh_arrays['frames']) if beh_arrays['frames'] else np.zeros(0, int)
...
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```
```python
'temporal_alignment_event':
    'trial start = entry into the textured virtual-reality corridor '
    '(beh["Trial_start_time"] / "StartFr")',
'off_start': 0.0,
'off_end': None,
'trial_end_event':
    'exit from the textured corridor into the 2 m grey space (beh["GrayFr"]); '
    'trial duration is behaviour-dependent so off_end varies per trial',
```

iii. Docstring: "a trial is one traversal of the 4 m textured corridor, from corridor entry (`Trial_start_time` / `StartFr`) to entry into the 2 m grey space (`GrayFr`)." The report states "`off_end` is None because duration is behaviour-dependent." The agent validated the alignment behaviourally (trajectory step 90): "in the converted data, lick rate is 0.21 in rewarded vs 0.04 in non-rewarded corridors and peaks in the bin containing the sound cue — reproducing the paper's Fig. 1c,d anticipatory-licking result, which confirms the neural/behaviour/cue alignment."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling, smoothing or downsampling is done. One column = one two-photon imaging frame. The bin size is measured from the data: the per-session frame rate is `1 / median(diff(ft))` in seconds, and `time_bin_size` is `1000 / median(frame rates across sessions)` = **314.69 ms** (3.1777 Hz), essentially the expert's 315.46 ms from a hard-coded 3.17 Hz. Note that, because non-running frames are dropped, consecutive columns within a trial are not always adjacent in time: the median inter-column interval is 0.315 s, but 6.5% of intervals exceed 0.5 s and the 99th percentile is 7.6 s.

ii.
```python
frame_rates.append(1.0 / (np.median(np.diff(b['ft'])) * 86400.0))
...
time_bin_size = 1000.0 / float(np.median(frame_rates))
print('time bin: %.2f ms' % time_bin_size, flush=True)
```
```python
'time_bin_size': float(time_bin_size),
'frame_rate_hz': float(np.median(frame_rates)),
```

iii. The imaging frame is the native resolution of the data and all behaviour streams (`ft_*`) are already sampled on the same grid, so there is nothing to resample onto. The rate is derived from `ft` rather than assumed. The report quotes "bin = 314.7 ms".

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the MATLAB datenum timestamp of the sound cue in each trial) and `ft` (the timestamp of every imaging frame). The equivalent frame-index encoding `SoundFr` is not used; the agent checked `SoundTime` for NaNs (trajectory step 52) before relying on it.

ii.
```python
ft = beh['ft'][:nfr]
...
tcue = beh['SoundTime'][trial]
...
to_cue = (tcue - ft[fr]) * 86400.0               # seconds until the sound cue
```

iii. `SoundTime` is the cue time in the same clock as `ft`, so no interpolation from a fractional frame number is needed. Metadata: `'time_to_sound_cue': 'seconds until the sound cue of this trial (positive before the cue, negative after it)'`. (Verified equivalent to the expert's `np.interp(SoundFr, arange(nfr), frame_time)` to within a fraction of a frame.)

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. A per-frame difference converted from days to seconds: `(SoundTime[trial] - ft[frame]) * 86400`. It is positive before the cue and negative after, matching the name "time **to** sound cue". Stored as row 0 of the `(4, T)` float32 input array. No clipping, no normalisation, no binarisation.

ii.
```python
inp = np.empty((len(INPUT_NAMES), len(fr)), dtype=np.float32)
inp[0] = to_cue
```

iii. Metadata `input_descriptions` states the sign convention explicitly. `86400` is the seconds-per-day conversion for MATLAB datenums, which the agent established when inspecting `ft` (trajectory steps 29–38). The observed range is −1763 s to +724 s; the extremes come from the trials in which the mouse stopped for many minutes (see 1-e).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the frames used for the neural columns of that trial: `ft[fr]` with the same `fr` index array, so element *t* of the input row corresponds to column *t* of the neural matrix by construction.

ii.
```python
for trial, fr in enumerate(per_trial):
    ...
    to_cue = (tcue - ft[fr]) * 86400.0
```
```python
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. Every stream in this dataset is indexed by imaging frame number, so using one shared `fr` array guarantees alignment. The agent verified the result empirically by showing that lick rate peaks in the bin containing the cue.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the recording date string `YYYY_MM_DD`) in the index entry, together with `mname` to identify the mouse. The dates are parsed to `datetime.date` and the earliest recording of each mouse is found.

ii.
```python
first_date = {}
for kn, s in sessions.items():
    d = datetime.date(*map(int, s['datexp'].split('_')))
    first_date[s['mname']] = min(first_date.get(s['mname'], d), d)
day_of_training = {kn: (datetime.date(*map(int, s['datexp'].split('_')))
                        - first_date[s['mname']]).days
                   for kn, s in sessions.items()}
```

iii. The date is the only field that orders a mouse's sessions. The report is explicit about the limitation: "day of training = days since that mouse's first recording (the released data has no absolute training-start date)". The agent printed the per-mouse day offsets before committing (trajectory step 56).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar-day difference between the session's date and the mouse's first recorded date, as an integer; a constant per session, broadcast across every bin of every trial of that session. Values run 0–92 (the expert instead counts recording *sessions*, giving 0–7). No normalisation is applied.

ii.
```python
inp[1] = day_of_training
```
```python
'day_of_training': 'days between this recording and the first recording of '
                   'the same mouse',
```

iii. Recording days of a mouse are not consecutive; the agent chose the literal elapsed-days reading ("Day of training, continuous, per-trial") over an ordinal session count, and documented in metadata that the origin is the first *recording*, not the true start of training (the paper says mice were trained ~2 weeks before the first recording).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the datenum timestamp of corridor entry for each trial) and `ft` (the timestamp of every imaging frame). `StartFr`, the fractional frame index of the same event, is not used. The agent checked `Trial_start_time` for NaNs before relying on it (trajectory step 52).

ii.
```python
t0 = beh['Trial_start_time'][trial]
...
t = (ft[fr] - t0) * 86400.0                      # seconds since corridor entry
```

iii. `Trial_start_time` is the corridor-entry time on the same clock as `ft`, so it can be subtracted directly with no interpolation. (I confirmed it agrees with `np.interp(StartFr, ...)` to well under one frame.)

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A per-frame difference converted from days to seconds: `(ft[frame] - Trial_start_time[trial]) * 86400`, positive after corridor entry and starting near 0. Stored as row 2 of the input array as float32, unclipped and unnormalised. It is **wall-clock** elapsed time, so on trials where the mouse stopped it keeps counting through the removed non-running frames: median 3.8 s, 99th percentile 60.7 s, maximum 1,765 s (the expert's maximum is 74.8 s, because those trials were dropped).

ii.
```python
inp[2] = t
```
```python
'time_since_trial_start': 'seconds since entry into the corridor (real time, '
                          'including periods when the mouse was not running)',
```

iii. Metadata spells out the wall-clock choice. The report defends it as informative rather than as an artefact: "seconds since corridor entry (real elapsed time, so it carries the animal's stopping behaviour — heavy-tailed, median 3.8 s, max 29 min)", and argues the time inputs "are computed from real frame timestamps so they stay correct across the gaps" left by the running-frame filter.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same mechanism as 3-c: it is evaluated on `ft[fr]` with the identical per-trial frame-index array `fr` that selects the neural columns, so index *t* matches column *t*.

ii.
```python
t = (ft[fr] - t0) * 86400.0
...
neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. All streams share the imaging-frame grid; using one `fr` array per trial for everything makes alignment automatic.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
inp[3] = 1.0 if beh['isRew'][trial] else 0.0
```

iii. Metadata: `'reward_available': '1 if this corridor is the rewarded one for this mouse (beh["isRew"]), 0 otherwise; always 0 for the unsupervised and naive cohorts'`. The agent cross-checked `isRew` against behaviour (trajectory step 90: lick rate 0.21 in rewarded vs 0.04 in non-rewarded corridors) and recorded per-session `rewarded_trials` and `Reward_Mode` in `session_info`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond casting the boolean to 0.0/1.0 and broadcasting the per-trial constant across all bins of the trial. 12.25% of timepoints carry 1.

ii.
```python
inp = np.empty((len(INPUT_NAMES), len(fr)), dtype=np.float32)
...
inp[3] = 1.0 if beh['isRew'][trial] else 0.0
```

iii. The flag is already exactly the quantity the decoder spec asks for ("1 if in rewarded corridor, 0 if not, discrete, per-trial"), so nothing is derived.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From three things: `WallName` (the wall texture of each trial), `UniqWalls` (the session's ordered list of distinct wall names) and the `stim_id` vectors in `Imaging_Exp_info` (parallel to `UniqWalls`). `build_stim_map` merges the `stim_id` vectors from *all* index entries of a recording — necessary because each analysis entry only labels the stimuli it uses and leaves the rest NaN (e.g. the `_swap1` entry of `TX88_2022_07_22_1` labels `leaf1_swap1` and NaNs `leaf1_swap2`, while the `_swap2` entry does the reverse). `TrialStim` is not used.

ii.
```python
def build_stim_map(session, beh):
    uniq = list(beh['UniqWalls'])
    mapping = {}
    for exp_type, key, stim_id in session['stim_id_entries']:
        if len(stim_id) != len(uniq):
            raise ValueError('stim_id length mismatch for %s (%s)' % (session['kn'], exp_type))
        for wall, sid in zip(uniq, stim_id):
            if not np.isnan(sid):
                sid = int(sid)
                if wall in mapping and mapping[wall] != sid:
                    raise ValueError('conflicting stim_id for %s / %s' % (session['kn'], wall))
                mapping[wall] = sid
    for wall in uniq:
        if wall not in mapping:
            if wall not in EXTRA_STIM:
                raise ValueError('unlabelled wall type %s in %s' % (wall, session['kn']))
            mapping[wall] = len(CANONICAL_STIM) + EXTRA_STIM.index(wall)
    return mapping
```

iii. File docstring: "stim_id in Imaging_Exp_info indexes into this list. Mice trained on the rock/brick/wood texture pairs are mapped onto the same identities by the authors' own stim_id tables." The `CANONICAL_STIM` list is copied verbatim from `utils.py` (`all_stim = ['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']`, lines 511/605/713). The merge and the two `ValueError` guards are the agent's own consistency checks; neither fires on the real data.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's `WallName` is looked up in the session's map, giving an integer index into `STIM_VALUES` = the 7 canonical `all_stim` identities plus `circle3` as an 8th category, and that per-trial constant is broadcast across the trial's bins. So the label space has **8 classes** (the expert collapses the 15 raw wall names into **4** texture families: circle/leaf/rock/wood). Because `stim_id` encodes the stimulus's *role* in the authors' analyses rather than its appearance (`utils.py:472`: `stim = uniqW[stim_id==2][0] # get the name of reward stimulus`), the label is not the texture name for 28 of 89 sessions: in some sessions a `rock1` wall is labelled `circle1` and a `wood1` wall `leaf1`, and in ~8 sessions the labels are swapped outright, e.g. a `leaf1` wall is labelled `circle1` while a `circle1` wall is labelled `leaf1`. The mapping is bijective within each session, so the per-session decoding problem is unaffected, and the raw mapping is preserved in `session_info['wall_name_to_stimulus']`.

ii.
```python
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']
EXTRA_STIM = ['circle3']
STIM_VALUES = CANONICAL_STIM + EXTRA_STIM
```
```python
out = np.empty((len(OUTPUT_NAMES), len(fr)), dtype=np.int64)
out[0] = stim_map[wall]
```
```python
'wall_name_to_stimulus': {w: STIM_VALUES[i] for w, i in r['stim_map'].items()},
```

iii. Report: "Stimulus uses the paper's canonical identities (`utils.all_stim`: circle1/2, leaf1/2/3, leaf1_swap1/2), built by merging the `stim_id` tables across a recording's entries — this is what maps the rock/brick/wood mice onto equivalent stimuli the way the paper pools them. `circle3` is the one wall type the paper never labels; I kept it as an 8th category rather than discarding 390 trials. Raw wall names are preserved per session in `metadata['session_info']`." Resulting class fractions: circle1 0.317, leaf1 0.333, leaf2 0.169, circle2 0.060, leaf3 0.058, leaf1_swap2 0.029, leaf1_swap1 0.027, circle3 0.007.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame number of every lick detected in the session. `LickTime`/`LickPos`/`LickTrind` are not used.

ii.
```python
lick = np.zeros(nfr, dtype=bool)
if len(beh['LickFr']) > 0:
    lf = np.round(np.asarray(beh['LickFr'], dtype=float))
    lf = lf[np.isfinite(lf)].astype(np.int64)
    lick[lf[(lf >= 0) & (lf < nfr)]] = True
```

iii. `LickFr` is already expressed in imaging-frame units, so it needs no clock conversion. The agent sanity-checked lick counts in a few sessions before writing the script (trajectory step 58, which prints selected-frame lick fractions per session).

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length boolean vector is built, set `True` at every frame containing at least one lick; the fractional frame index is rounded to the nearest frame (the expert truncates instead). Empty `LickFr`, non-finite entries, and licks outside `[0, nfr)` are skipped. Per trial the vector is sliced and cast to int, giving a binary time series (`no_lick`/`lick`). Overall 3.47% of bins are licks (expert: 4.14% — the difference is that licks occurring while the mouse was stopped fall on frames this conversion removes).

ii.
```python
out[1] = lick[fr].astype(np.int64)
```
```python
LICK_VALUES = ['no_lick', 'lick']
```
```python
'lick': 'at least one lick was detected in this imaging frame',
```

iii. The spec asks for "Licking, binary, time-varying. 0 = not licking, 1 = licking", and a per-frame occupancy flag is the direct rendering of that on the imaging grid. The agent validated it against the paper's Fig. 1c,d anticipatory-licking result.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the flag vector is already on the neural grid; it is sliced with the same per-trial `fr` array used for the neural columns.

ii.
```python
out[1] = lick[fr].astype(np.int64)
...
neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. One frame-index array per trial drives every stream, so no separate alignment step is needed. Confirmed empirically: lick probability peaks in the bin containing the sound cue and is 5× higher in rewarded than non-rewarded corridors.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position inside the corridor at each imaging frame, together with `Texture_Length` (40 in every session, i.e. 4 m of textured corridor in decimetres).

ii.
```python
pos = beh['ft_Pos'][:nfr]
texture_len = float(beh['Texture_Length'])          # 40 units == 4 m
bin_len = texture_len / len(POSITION_VALUES)
```

iii. `ft_Pos` gives one position per imaging frame, so it is the natural source; the bin width is taken from the session's own `Texture_Length` rather than hard-coded, so the four bins are guaranteed to tile the textured corridor. Metadata records `'corridor_length_m': 4.0`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by the bin length (40/4 = 10 decimetres = 1 m), truncated to an integer and clipped to `[0, 3]`, giving a per-frame category. It is stored as row 2 of the int64 output array, time-varying. The clip never actually fires: within the kept frames `ft_Pos` spans 0 to 39.999999999. Resulting occupancy is essentially uniform: 0.250 / 0.249 / 0.250 / 0.252.

ii.
```python
pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)
out[2] = pos_bin
```
```python
POSITION_VALUES = ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The decoder spec asks for "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins, time varying"; dividing the 4 m texture by 4 is the direct implementation. Metadata: `'position_bin': 'position along the 4 m textured corridor in 1 m bins'`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-width thresholds at 1 m, 2 m and 3 m (i.e. `ft_Pos` of 10, 20, 30), identical in every session and every trial, with a clip so that a frame at exactly the end of the texture lands in the last bin rather than a fifth one. The thresholds are fixed spatial values, not data-dependent quantiles.

ii.
```python
bin_len = texture_len / len(POSITION_VALUES)
pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)
```

iii. The instruction specifies equal-length 1 m bins, so the thresholds follow from the geometry; no data-driven threshold is appropriate. Because the mouse traverses the corridor at roughly constant VR speed, the four bins come out almost exactly equally occupied anyway.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, so it is already on the neural grid, and it is indexed with the same per-trial `fr` array as the neural columns.

ii.
```python
p = pos[fr]
...
out[2] = pos_bin
...
neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. Shared frame indexing; no resampling or interpolation is involved.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the mouse at each imaging frame (cm/s).

ii.
```python
speed = beh['ft_RunSpeed'][:nfr]
...
s = speed[fr]
```
```python
'speed_bin': 'quartile of the running speed (ft_RunSpeed), quartiles '
             'computed over all timepoints in the dataset',
```

iii. `ft_RunSpeed` is the recorded speed on the imaging grid, and the decoder spec asks for running speed discretised, so it is used directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behaviour-only first pass runs `session_behaviour_arrays` over all 89 sessions, concatenates the speeds of every kept frame in the whole dataset, and takes the 25th/50th/75th percentiles as three global cut points (**12.42 / 25.35 / 40.85 cm/s**). Those same three edges are then applied to every session in the second pass. Speeds are not smoothed, rectified or otherwise transformed. Globally the four classes come out at 0.2500 each; per session they can be very unbalanced (one session is 0.956 / 0.042 / 0.002 / 0.000), because a single global threshold cannot balance a session whose mouse ran slowly throughout. (The expert instead ranks within each session, so every session is exactly 25/25/25/25.)

ii.
```python
    speeds = []
    for kn, s in sessions.items():
        b = beh[kn]
        nfr = len(b['ft'])
        arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
        speeds.append(arrays['speeds'])
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed quartile edges (cm/s):', np.round(speed_edges, 3), flush=True)
```

iii. Report: "speed is `ft_RunSpeed` split at global quartiles (10.4 / 20.2 / 32.0 cm/s) computed over all kept timepoints so the classes mean the same thing in every session." (The quoted numbers are from the earlier 4-session pilot; the full-dataset edges written to `metadata['speed_bin_edges_cm_per_s']` are 12.42 / 25.35 / 40.85.) Computing the edges over the frames that survive the running filter also sidesteps the large mass of exactly-zero speeds that would otherwise make quartiles impossible to balance.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.searchsorted(speed_edges, s, side='right')` assigns each frame to 0–3 by comparing its speed against the three global edges; `side='right'` puts a value exactly equal to an edge into the higher bin. When `speed_edges is None` (the behaviour-only first pass) the row is filled with 0 and discarded.

ii.
```python
if speed_edges is not None:
    out[3] = np.searchsorted(speed_edges, s, side='right')
else:
    out[3] = 0
```
```python
SPEED_VALUES = ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']
```
```python
'speed_bin_edges_cm_per_s': [float(x) for x in speed_edges],
```

iii. The spec asks for "Running speed discretized into 4 bins, each corresponding to 25% of the data"; global percentile edges are the literal reading of "25% of the data" and make the class labels comparable across sessions. The edges are written into the metadata so the discretisation is reversible.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is sampled once per imaging frame, so it is already on the neural grid, and it is indexed with the same per-trial `fr` array as the neural columns.

ii.
```python
s = speed[fr]
...
out[3] = np.searchsorted(speed_edges, s, side='right')
...
neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. Shared frame indexing across all streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards, all local and non-fatal:
 - Behaviour can run past the imaging, so every behaviour stream is cut to `nfr = min(plane.shape[1] for plane in spks)`, the number of imaged frames.
 - Frames whose `ft_trInd` is NaN (the pre-trial period, which is `ft_CorrSpc == True` at the start of a session) are excluded from every trial.
 - Licks are filtered to finite values inside `[0, nfr)`; an empty `LickFr` is handled explicitly.
 - Trials with fewer than 2 surviving frames, or with a non-finite `Trial_start_time`/`SoundTime`, or with any non-finite time/position/speed value, are skipped. (None of these fire on this dataset.)
 - A wall type with no `stim_id` in any index entry is only accepted if it is the known-unlabelled `circle3`; anything else raises.
 - The neuron count in the spk file is checked against the length of the retinotopy `iarea` vector.
 There is no per-session `try/except`, so a session that did raise would abort the whole run rather than being skipped.

ii.
```python
nfr = min(p.shape[1] for p in spks)
n_file = sum(p.shape[0] for p in spks)
if n_file != n_total:
    raise ValueError('%s: %d neurons in spk file but %d in retinotopy' % (kn, n_file, n_total))
```
```python
valid = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr] & ~np.isnan(ft_trInd)
```
```python
if len(beh['LickFr']) > 0:
    lf = np.round(np.asarray(beh['LickFr'], dtype=float))
    lf = lf[np.isfinite(lf)].astype(np.int64)
    lick[lf[(lf >= 0) & (lf < nfr)]] = True
```

iii. Code comment: "the number of imaging frames in the recording bounds the behaviour arrays (utils.py slices the behaviour with spk.shape[1] in the same way)". The agent probed for NaNs in `SoundTime`, `Trial_start_time`, `StartFr` and `GrayFr` across several sessions (trajectory step 52) before deciding which guards were needed, and the report confirms the guards never actually remove anything: "All 38,110 trials have ≥11 frames; none were dropped."

## 12-a. What are the most time-consuming steps of the code?

i. Reading and slicing the spike files, which total ~405 GB across the 89 sessions — everything else is negligible. The agent mitigated this by (a) parallelising sessions across a `multiprocessing.Pool` (12 workers for the production run, `maxtasksperchild=1` so each worker's memory is reclaimed), and (b) selecting the ≤2000 chosen rows and the kept frames plane by plane so only the reduced matrix is materialised. The behaviour-only first pass (reading all 23 `Beh_*.npy` files and grouping trial frames for all 89 sessions) is the second cost, on the order of a minute.

ii.
```python
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
```
```python
    for plane in spks:
        n = plane.shape[0]
        sel = (rows >= offset) & (rows < offset + n)
        if sel.any():
            out[sel] = plane[rows[sel] - offset][:, frames]
        offset += n
    del spks
```

iii. Not stated explicitly by the agent, but implied by the design (workers, per-plane slicing, `del spks`, only 16 behaviour fields retained) and by the report's note that the cap "keeps the file at 6.6 GB instead of ~175 GB". `np.load` of a pickled dict still reads each plane in full, so the I/O itself is not avoidable.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loop is the per-trial loop in `session_behaviour_arrays`, which allocates a `(4, T)` input and a `(4, T)` output array per trial and computes four small per-frame expressions; with 38,110 trials this is the bulk of the Python-level work, and it could be done as four whole-session vectors sliced per trial (the way the expert code computes `licking`, `position` and `speed` once per session). `np.searchsorted` for the speed bins, and the `(p / bin_len)` position binning, are called once per trial instead of once per session. The per-plane loop in `process_session` is bounded by the number of imaging planes (a handful) and is not worth vectorising. Notably, the trial-grouping loop is *already* vectorised (one argsort + searchsorted over the session), avoiding the per-trial full-session scan the expert code does.

ii.
```python
    for trial, fr in enumerate(per_trial):
        ...
        pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)
        inp = np.empty((len(INPUT_NAMES), len(fr)), dtype=np.float32)
        ...
        out[3] = np.searchsorted(speed_edges, s, side='right')
```
```python
    order = np.argsort(tr, kind='stable')
    frames, tr = frames[order], tr[order]
    bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
```

iii. The agent did not comment on this. The per-trial loop is unavoidable in part, because the target format is a list of per-trial arrays; only the element-wise computation inside it could be hoisted, and it is negligible next to the 405 GB of neural I/O.

## 12-c. What processing does the code repeat multiple times?

i. `session_behaviour_arrays` is called **twice for every session**: once in the first pass to collect speeds for the global quartile edges, and again inside the worker to produce the real inputs and outputs. Each call re-runs `build_stim_map`, `trial_frames` (including the argsort/searchsorted grouping), the lick-vector construction, and the full per-trial allocation of the `(4, T)` input and output arrays. `np.load` of each `Beh_*.npy` happens once, but the derived behaviour is rebuilt from scratch. In addition, each worker is handed the behaviour dict by pickling it through the `Pool`, so the behaviour is re-serialised per session.

ii.
```python
    # pass 1 (behaviour only): global running-speed quartiles over all kept timepoints
    for kn, s in sessions.items():
        b = beh[kn]
        nfr = len(b['ft'])
        arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
        speeds.append(arrays['speeds'])
```
```python
    beh_arrays = session_behaviour_arrays(session, beh, nfr, day_of_training, speed_edges)
```

iii. Not commented on. A two-pass structure is unavoidable given that the speed quartiles are global, but the first pass only needs `trial_frames` and `ft_RunSpeed`; reusing `session_behaviour_arrays` wholesale was the convenient rather than the cheap choice. The cost is small relative to the neural I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things, all cheap:
 - In the first pass everything except `speeds` is thrown away — the `(4, T)` input arrays, the position binning, the lick vector, the stimulus lookup, and an `out[3] = 0` placeholder row for a speed binning that cannot be done yet.
 - The first pass uses `nfr = len(b['ft'])` (the full behaviour length) rather than the imaged-frame count used in the real pass, so the quartile edges are computed over a slightly different frame set than the one written out.
 - `frame_rates` is computed per session but only its median is used.
 - Rich `session_info` bookkeeping (`Counter` of stimulus trial counts, `wall_name_to_stimulus`, `rewarded_trials`, `reward_mode`, `experiment_types`) is assembled for all 89 sessions and never read by the decoder.
 - Outputs are stored as `int64` and neural as `float32`; `int8` and `float16` would carry the same information (the expert uses exactly those), halving the 6.6 GB file.

ii.
```python
        out = np.empty((len(OUTPUT_NAMES), len(fr)), dtype=np.int64)
        ...
        if speed_edges is not None:
            out[3] = np.searchsorted(speed_edges, s, side='right')
        else:
            out[3] = 0
```
```python
        stim_counts = Counter(STIM_VALUES[int(o[0, 0])] for o in r['output'])
        session_info.append({... 'stimulus_trial_counts': dict(stim_counts), ...})
```

iii. Not commented on. The metadata bookkeeping is deliberate — the instructions ask for descriptive metadata such as `session_info`, and preserving `wall_name_to_stimulus` is what makes the role-based stimulus labels of 7-b recoverable — so it is discarded by the decoder but not wasted. The duplicated first-pass work and the oversized dtypes are genuine waste.
