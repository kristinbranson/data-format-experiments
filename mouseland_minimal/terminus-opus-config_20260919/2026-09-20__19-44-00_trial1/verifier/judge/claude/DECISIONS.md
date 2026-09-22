# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI ignores the master index `beh/Imaging_Exp_info.npy` and instead globs every `beh/Beh_*.npy` file, loading each one fully and iterating over its keys. Each key is a "view" of a recording (`mouse_date_blk`, optionally with a `_swap1`/`_swap2` suffix); views are collapsed onto a base recording id, the first view encountered supplies the behavior dict `d['beh']`, and the `UniqWalls -> stim_id` maps of *all* views are merged into `d['walls']`. This yields 89 unique recordings, which the AI verified to be a 1:1 match with the 89 files in `spk/`. The spike file `spk/<session>_neural_data.npy` (a dict of one neurons x frames array per imaging plane) and the retinotopy file `retinotopy/<mouse>_<date>_trans.npz` (field `iarea`) are then loaded once per session inside the main loop. Behavior is read twice in effect: once in `collect_recordings()` and then held in RAM for a behavior-only first pass that computes global running-speed quartiles, and again (same objects) in the main conversion loop.

ii.
```python
def collect_recordings():
    """Deduplicate the Beh_*.npy views into one entry per recording."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'beh', 'Beh_*.npy'))):
        exp = os.path.basename(f)[4:-4]
        B = np.load(f, allow_pickle=True).item()
        for key, beh in B.items():
            parts = key.split('_')
            base = '_'.join(parts[:5]) if parts[-1].startswith('swap') else key
            d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
            d['exps'].append(exp)
            # merge the wall -> canonical role maps of all views of this recording
            for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
                if not np.isnan(sid):
                    d['walls'][str(wall)] = int(sid)
            if d['beh'] is None:
                d['beh'] = beh
```
```python
        ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mouse, date)),
                      allow_pickle=True)
        iarea = ret['iarea']
        ...
        spkfile = os.path.join(ROOT, 'spk', '%s_neural_data.npy' % name)
        blocks = np.load(spkfile, allow_pickle=True).item()['spks']
        counts = [b.shape[0] for b in blocks]
        assert sum(counts) == len(iarea), (name, sum(counts), len(iarea))
```

iii. From the trajectory (steps 22-27): the AI first surveyed all behavior files and found "99 unique session keys across beh files (some with stimtype suffix), 89 spk files", then established that "after deduplicating the swap1/swap2 views, there are exactly 89 unique recordings, matching the 89 spk files 1:1" and explicitly checked `rec not in spk: []` / `spk not in rec: []`. Its stated reason for deduplicating is that "the Beh files contain multiple views of the same recording (`_swap1`/`_swap2`, and re-listing under several experiment types). Collapsing these yields exactly 89 recordings <-> 89 spk files. Without this, sessions would have been silently duplicated." It verified retinotopy coverage for all 89 recordings ("recordings missing retinotopy: []") and asserted per session that the plane neuron counts sum to `len(iarea)`.

## 1-b. How are the data split into subjects (mice)?

i. The mouse name is the first underscore-separated token of the recording id. Subjects are accumulated in first-encounter order over the alphabetically sorted recording names (which is equivalent to sorted order), and `subject_idx` stores each session's index into that list. The result is 19 mice over 89 sessions.

ii.
```python
        mouse = name.split('_')[0]
        ...
        if mouse not in subjects:
            subjects.append(mouse)
        data['subject_idx'].append(subjects.index(mouse))
```
```python
    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The AI confirmed the split from the data rather than assuming: "n mice 19 ['DR10', 'DR15', 'LZ13', 'LZ16', 'TX104', ...]", matching the paper's "89 recordings in 19 mice". The mouse name is already the leading field of every behavior key and spike filename, so no derivation is needed.

## 1-c. How are the data split into sessions?

i. A session is one unique recording = `mouse_date_block`. Behavior keys that carry a trailing `_swapN` token are mapped onto the 5-token base id, and a recording that appears in several `Beh_<exp_type>.npy` files is kept once (the `setdefault` keeps the first view's behavior, while every view's experiment type is recorded in `session_info['experiment_types']`). 89 sessions result, one per spike file. No session-level quality filter is applied; all 89 are written out.

ii.
```python
            parts = key.split('_')
            base = '_'.join(parts[:5]) if parts[-1].startswith('swap') else key
            d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
            d['exps'].append(exp)
            ...
            if d['beh'] is None:
                d['beh'] = beh
```
```python
    names = sorted(recs)
    print('unique recordings:', len(names), flush=True)
```

iii. Step 26: "'stimulus_of_trial' in TrialStim is a placeholder used in the swap1/swap2 'views' of the SAME recording (identical ft length/ntrials)... So I must (a) dedupe sessions to unique recordings (mname_datexp_blk)". The AI checked that the two swap views of `DR10_2022_07_30_1` have identical `WallName` counts and differ only in which stimuli their `stim_id`/`TrialStim` name, justifying the use of a single behavior record per recording plus a merged label map.

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` traversals the behavior declares. The frames of a trial are those the behavior labels with that trial index (`ft_trInd == t`) **and** that are inside the 4 m textured corridor (`ft_CorrSpc`, positions 0-40 dm) **and** during which the virtual reality was moving, i.e. the mouse was running (`ft_move > 0`). The 2 m of grey space (`ft_GraySpc`, 40-60 dm) is excluded. Frames are returned per trial in increasing frame order, so each trial starts at corridor entry and ends at the end of the texture; trials are variable length (mean 21.6, median 21, min 11, max 178 bins). Because non-running frames are dropped, a trial's frames are not necessarily contiguous in real time.

ii.
```python
def trial_frames(beh, nfr):
    """Frames of each trial: running, inside the textured corridor."""
    tr = beh['ft_trInd'][:nfr]
    corr = beh['ft_CorrSpc'][:nfr].astype(bool)
    moving = beh['ft_move'][:nfr] > 0
    ok = corr & moving & ~np.isnan(tr)
    order = np.argsort(tr[ok], kind='stable')
    idx = np.where(ok)[0][order]
    groups = collections.defaultdict(list)
    for i in idx:
        groups[int(tr[i])].append(i)
    return {t: np.sort(np.array(v)) for t, v in groups.items()}
```
```python
        for t in sorted(frames):
            fr = frames[t]
```

iii. Two justifications, both from the source material. On the corridor window (steps 31-38): the AI measured that `ft_trInd == t & ft_CorrSpc` spans exactly 0-39.97 dm while `ft_GraySpc` spans 40-60 dm, and concluded "frames selected by `ft_trInd==trial & ft_CorrSpc` give exactly the 0-40 dm textured corridor (the paper's 4-m corridor), which maps perfectly onto the required 4x1-m position bins". It rejected `floor(StartFr)` indexing because "flooring StartFr picks the frame *before* corridor entry". On the running filter (step 29 and the module docstring): "only frames while the mouse was running, i.e. the virtual reality was moving (ft_move > 0, the VRmove filter of data_process_script.ipynb). The paper: 'We only considered timepoints during running for analysis'." It measured that only 59-76% of in-corridor frames are running frames, and that raw trial lengths of 14-905 frames collapse to ~21 running frames, "because mice stop".

## 1-e. How are trials filtered based on quality controls?

i. Two filters, applied identically in the behavior pre-pass and the main loop:
 * a trial whose wall texture has no canonical `stim_id` in any view of the recording is dropped (309 trials, all `circle3`, in 4 recordings);
 * a trial with fewer than `MIN_FRAMES_PER_TRIAL = 5` running frames inside the corridor is dropped (0 trials in practice).
37,801 of 38,110 trials survive; minimum 84 trials per session. No outlier-length trial filter is applied, because the running-frame filter already removes the long stationary stretches; the longest surviving trial is 178 bins (~56 s of running). No session is dropped.

ii.
```python
MAX_NEURONS = 2000      # per session; the decoder's SVD init also caps at 2000
MIN_FRAMES_PER_TRIAL = 5
```
```python
            wall = str(beh['WallName'][t])
            if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
                continue    # unassigned stimulus (circle3) or too little running
```
```python
        'excluded': 'trials whose wall texture had no canonical stimulus role '
                    '(circle3) and trials with fewer than %d running frames'
                    % MIN_FRAMES_PER_TRIAL,
```

iii. Step 36: "One wall name ('circle3', 309 trials in one recording) has no canonical stim_id and needs a decision"; step 39: "The 'circle3' trials exist in a few sessions and have no canonical stim_id in any view (except TX109 where circle3->role 4), so they need an explicit decision." The AI chose to drop them rather than invent an eighth category, stating in the final summary that "the 309 excluded 'circle3' trials have no canonical role in any view". The <5-frame rule is a guard against degenerate trials ("only 1 degenerate trial" in its audit); its behavior pass reported "dropped short 0, dropped unmapped stim 309".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<session>_neural_data.npy`, a list of one (neurons x frames) array per imaging plane, which the AI indexes *without* concatenating (it maps the selected global neuron indices back into per-plane offsets). Neuron area identity comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
        iarea = ret['iarea']
        area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
        valid = np.where(area >= 0)[0]
        ...
        blocks = np.load(spkfile, allow_pickle=True).item()['spks']
        counts = [b.shape[0] for b in blocks]
        assert sum(counts) == len(iarea), (name, sum(counts), len(iarea))
        nfr = blocks[0].shape[1]
        offs = np.concatenate([[0], np.cumsum(counts)])
        spk = np.empty((len(sel), nfr), dtype=np.float32)
        for bi, blk_arr in enumerate(blocks):
            m = (sel >= offs[bi]) & (sel < offs[bi + 1])
            if m.any():
                spk[m] = blk_arr[sel[m] - offs[bi]]
```

iii. The AI followed `utils.load_spk` (which concatenates `spks` across planes) and `utils.load_retino`/`neu_area_ID`, and asserted per session that the concatenated plane neuron count equals `len(iarea)`, i.e. that the retinotopy indexes the planes in the same order. It notes in the docstring that these are "Suite2p deconvolved traces", consistent with the paper's "All our analyses were based on deconvolved fluorescence traces".

## 2-b. How is the `neural` data processed?

i. Three steps: (1) restrict to neurons with a valid visual-area assignment; (2) **randomly subsample to at most 2000 neurons per session** with a per-session seeded RNG (`SEED + si`), indices kept sorted; (3) **z-score each retained neuron over the whole recording** (all frames, not just the retained ones), with zero-variance neurons protected. Trials are then stored as `float32` slices of variable length; nothing is padded, smoothed or rebinned. Every session ends up with exactly 2000 neurons (out of 20k-90k recorded, e.g. 2000 of 58,224 for `DR10_2022_07_12_1`), giving 178k neurons total and a 6.6 GB pickle.

ii.
```python
        rng = np.random.default_rng(SEED + si)
        if len(valid) > MAX_NEURONS:
            sel = np.sort(rng.choice(valid, MAX_NEURONS, replace=False))
        else:
            sel = valid
```
```python
        # z-score each neuron over the whole recording (as in utils.py)
        mu = spk.mean(axis=1, keepdims=True)
        sd = spk.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        spk = (spk - mu) / sd
        ...
            neural_s.append(spk[:, fr].astype(np.float32))
```

iii. Docstring: traces are "z-scored per neuron across the whole recording (as utils.get_kfold_reward_response does), then a random subset of at most MAX_NEURONS is kept per session because the full dataset is 405 GB." Step 33 adds the decoder-side reason for z-scoring: "The decoder learns a per-session linear projection to npcs=100 dims, initialized by SVD of the session's neural matrix. It does not z-score, so I should provide sensibly scaled neural data (the paper z-scores deconvolved traces per neuron, which is also numerically well-conditioned here)." Step 41 gives the reason for the 2000 cap: "Neural data is 405 GB, so I must subsample neurons; I'll keep 2000 per session (matching the decoder's svd_max_neurons so SVD init uses real neurons)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only *quality* filter is the visual-area assignment: `iarea` is mapped to V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4) exactly as `utils.neu_area_ID` does, and neurons with `iarea` of -1 or 7 (~7.5% of cells) are dropped. No further curation (the authors' Suite2p cell classifier already curated the cells). On top of this quality filter sits the non-quality random subsample to 2000 neurons per session described in 2-b, which discards roughly 96% of the area-assigned neurons.

ii.
```python
AREA_NAMES = ['V1', 'mHV', 'lHV', 'aHV']
# utils.neu_area_ID: V1 = 8; mHV = 0,1,2,9; lHV = 5,6; aHV = 3,4
IAREA_TO_AREA = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}
```
```python
        area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
        valid = np.where(area >= 0)[0]
```
```python
        'neuron_subsampling':
            'neurons with a retinotopic area assignment (V1/mHV/lHV/aHV); at most '
            '%d randomly chosen per session (full dataset is 405 GB)' % MAX_NEURONS,
```

iii. Docstring: "restricted to neurons with a retinotopic area assignment (V1/mHV/lHV/aHV as in utils.neu_area_ID; iarea -1 and 7 are unassigned and are excluded, as in the paper's analyses)". The AI measured the `iarea` histogram over 12 retinotopy files before deciding (iarea -1: 6.2%, 7: 1.3%). The subsample is justified purely by data volume ("the raw data is 405 GB") plus the belief that it "matches the decoder's svd_max_neurons so SVD init uses real neurons".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start = corridor entry. Each trial's neural matrix is the columns of the z-scored trace at that trial's retained (in-corridor, running) frames, in increasing frame order, so column 0 is the first running frame inside the corridor and the last column is the last running frame before the grey space. Trials keep their own length; nothing is padded or truncated to a common window. `off_start = 0.0`, `off_end = None`.

ii.
```python
            neural_s.append(spk[:, fr].astype(np.float32))
```
```python
        'temporal_alignment_event': 'trial start = entry into the virtual corridor',
        'off_start': 0.0,
        'off_end': None,
        'trial_window':
            'corridor entry to the end of the 4 m textured corridor; only frames '
            'while the mouse was running (virtual reality moving) are kept, so trials '
            'have variable numbers of time bins (median 21)',
```

iii. Step 32: "each trial runs from corridor entry (pos 0) through the 40-dm texture region... Since the decoder spec asks for exactly 4 x 1-m position bins, the trial window should be the 4-m textured corridor (StartFr->GrayFr). Restricting to running frames (ft_move>0, as the paper's own VRmove filter does) yields clean ~20-25-frame trials." All other streams are taken on exactly the same frame index array `fr`, so every stream is aligned by construction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling or position-interpolation is applied: one bin = one two-photon imaging frame. The AI measured the frame rate across all 89 recordings (3.1709-3.1807 Hz, median 3.1777 Hz) and reports `time_bin_size` as the median over sessions of the per-session median inter-frame interval = 314.697 ms. Note that, because non-running frames are removed, consecutive bins within a trial are one frame apart in *running* time but may be separated by arbitrarily long real-time gaps.

ii.
```python
        ft = beh['ft'][:nfr] * 86400.0
        bin_s = float(np.median(np.diff(ft)))
        ...
        bin_ms.append(np.median(np.diff(ft)) * 1000.0)
```
```python
        'time_bin_size': float(np.median(bin_ms)),
```

iii. Step 27/28: "fs min/max/median 3.1709 3.1807 3.1777"; step 60: "Time bin = 314.7 ms (3.177 Hz)." The imaging frame is the native resolution of both the neural data and every behavior stream (all `ft_*` arrays are per frame), so the AI treated the frame grid as the common time base and did not resample. It deliberately did *not* use the authors' `get_interpPos_spk` position-interpolation (60 position bins per corridor), since the decoder task asks for time bins, not position bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `beh['SoundFr']` (the fractional imaging-frame index of the sound cue in each trial), the trial's retained frame index array `fr`, and `beh['ft']` (only to obtain the per-session frame period `bin_s`). The absolute timestamp field `SoundTime` is deliberately *not* used.

ii.
```python
            k = np.arange(len(fr), dtype=np.float32) * bin_s
            rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
            inp = np.empty((4, len(fr)), dtype=np.float32)
            inp[0] = k - rank_cue
```

iii. The AI first used `ft[fr] - SoundTime*86400`, found values of -723 s to +1763 s, and investigated (steps 54-56): it verified that "SoundFr is *always* inside [StartFr, GrayFr] for all 38,110 trials" and that the absolute clock agrees with the frame clock to <1 s in every session, concluding that the extreme values were "genuine pauses (mice stop mid-corridor)" rather than bad cue data. It then switched to the frame-index field, which it had verified to be trustworthy.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue is located by rank: `np.searchsorted(fr, SoundFr[t])` gives the number of retained frames of the trial that precede the cue, and that rank is multiplied by the frame period to place the cue on the trial's accumulated-running-time axis. The input is then `k - rank_cue`, i.e. running time **since** the cue: negative before the cue, zero at the first retained frame at or after the cue, positive after it. Values span about -19.5 s to +51 s. Note this is the opposite sign to the literal reading of the name "time *to* sound cue", and it measures running time rather than wall-clock time, so a pause between a bin and the cue is not counted.

ii.
```python
            # Times are measured on the retained (running) frames, i.e. in
            # accumulated running time since corridor entry.  Only running frames
            # are kept (see module docstring), so this is the clock of the neural
            # sequence the decoder actually sees; wall-clock time would contain
            # large invisible jumps whenever the mouse stopped inside the corridor
            # (pauses of up to ~30 min occur), which are not represented in the data.
            k = np.arange(len(fr), dtype=np.float32) * bin_s
            rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
            inp[0] = k - rank_cue
```
```python
        'time_convention':
            'time_since_trial_start and time_to_sound_cue are accumulated '
            'running time (retained frames x frame period) relative to corridor '
            'entry and to the sound cue; periods when the mouse stopped are '
            'excluded from the data and therefore from the clock',
```

iii. Step 57: "Because the paper keeps only running timepoints, the neural sequence handed to the decoder is the running-only series, so its time base should be accumulated running time, not wall-clock time spanning discarded pauses. I'll define both time inputs on the retained-frame clock (bounded by trial length, max ~56 s), which is consistent with the neural data and avoids pathological input scales." It then validated on 3 sessions that "time_to_sound_cue -10 to +10 s, and the cue crosses zero within the trial as expected".

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built on the same `fr` array as the neural columns, one value per neural bin, so alignment is exact by construction: element `j` of the input corresponds to column `j` of the trial's neural matrix.

ii.
```python
            neural_s.append(spk[:, fr].astype(np.float32))
            k = np.arange(len(fr), dtype=np.float32) * bin_s
            ...
            inp[0] = k - rank_cue
```

iii. Every stream in this dataset is sampled on the imaging-frame grid (`ft_*` arrays, `SoundFr`/`LickFr` given as frame indices), so selecting the same frame set for all streams aligns them. The AI additionally cross-checked the frame-to-trial mapping against `ft_WallID` in all 89 sessions and found 0 mismatches.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The calendar date embedded in the recording id (`mouse_YYYY_MM_DD_blk`), parsed into a `datetime.date`, together with the set of all recording dates for that mouse.

ii.
```python
    def date_of(name):
        p = name.split('_')
        return datetime.date(int(p[1]), int(p[2]), int(p[3]))
    first = {}
    for name in names:
        m = name.split('_')[0]
        first[m] = min(first.get(m, date_of(name)), date_of(name))
```

iii. The date is the only field that orders a mouse's sessions; the AI took the mouse's earliest recording as the origin. (The behavior dicts contain absolute MATLAB datenums too, but the AI had found those fields less convenient and used the id.)

## 4-b. What processing is involved in computing `input` *Day of training*?

i. `day = (date_of(session) - first_date_of_that_mouse).days`, i.e. elapsed **calendar** days since the mouse's first imaging session, so the first session of every mouse is 0. The value is computed over all 89 sessions (not only those converted) and broadcast as a constant across every bin of every trial of the session. Realised values are 0-92 days across 42 distinct values.

ii.
```python
        day = float((date_of(name) - first[mouse]).days)
        ...
            inp[1] = day
```
```python
        session_info.append({'session': name, ..., 'day_of_training': day, ...})
```

iii. The AI did not narrate this choice beyond treating it as the natural continuous reading of "Day of training, continuous, per-trial": elapsed days since the first recording of that mouse. (It does mean the gaps between recordings are counted, and that day 0 is the first *recording*, not the first day of behavioural training.)

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The trial's retained frame array `fr` (whose first element is the first running frame inside the corridor) and `beh['ft']` for the per-session frame period. `StartFr` and `Trial_start_time` are not used in the final version.

ii.
```python
        ft = beh['ft'][:nfr] * 86400.0
        bin_s = float(np.median(np.diff(ft)))
        ...
            k = np.arange(len(fr), dtype=np.float32) * bin_s
            inp[2] = k
```

iii. Same reasoning as 3-a/3-b: the AI verified that the absolute-time fields agree with the frame clock and that the huge wall-clock spans were genuine mid-corridor pauses, then re-based both time inputs on the retained-frame clock so that the input clock "matches the neural sequence".

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It is simply the bin index multiplied by the frame period: 0, 0.3147, 0.6294, ... s. Because trials begin at corridor entry, this is accumulated running time since entry; stationary periods inside the corridor are not counted. Range 0 to 55.7 s over the dataset.

ii.
```python
            k = np.arange(len(fr), dtype=np.float32) * bin_s
            inp[2] = k
```

iii. As quoted in 3-b: wall-clock time "would contain large invisible jumps whenever the mouse stopped inside the corridor (pauses of up to ~30 min occur), which are not represented in the data", so the AI used the clock of the retained sequence. It validated the result: "time_since_trial_start spans 0-12 s" on the test subset, 0-55.7 s on the full dataset.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. By construction: it is indexed by position within the same `fr` array used for the neural columns, so bin `j` of the input is bin `j` of the neural matrix, and bin 0 is the trial's first neural bin.

ii.
```python
            neural_s.append(spk[:, fr].astype(np.float32))
            k = np.arange(len(fr), dtype=np.float32) * bin_s
            inp[2] = k
```

iii. All streams are cut with the same frame index array, which is the AI's general alignment strategy for this dataset.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial flag marking trials run in the rewarded corridor.

ii.
```python
            inp[3] = float(beh['isRew'][t])
```

iii. `isRew` is the field the reference code itself uses for the rewarded corridor (e.g. `utils.get_cat_id`, `get_kfold_reward_response`), so it is taken directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float and broadcasting the per-trial scalar across all bins of the trial. It is 0 for all trials of the unsupervised/naive/grating recordings and 1 for rewarded-corridor trials of the task recordings (dataset-wide range [0, 1]).

ii.
```python
            inp = np.empty((4, len(fr)), dtype=np.float32)
            ...
            inp[3] = float(beh['isRew'][t])
```

iii. No justification was needed or given; the field already encodes exactly what the decoder spec asks for ("1 if in rewarded corridor, 0 if not").

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']` (the texture actually shown on each trial) combined with the recording's `UniqWalls` -> `stim_id` map, merged across all views of that recording. `stim_id` is the authors' canonical stimulus *role* (0 circle1, 1 circle2, 2 leaf1, 3 leaf2, 4 leaf3, 5 leaf1_swap1, 6 leaf1_swap2) as documented in `data_process_script.ipynb`. `TrialStim` is explicitly rejected.

ii.
```python
            for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
                if not np.isnan(sid):
                    d['walls'][str(wall)] = int(sid)
```
```python
            wall = str(beh['WallName'][t])
            if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
                continue    # unassigned stimulus (circle3) or too little running
            ...
            out[0] = walls[wall]
```

iii. Docstring: "Physical texture names differ between mice (leaf/circle, rock/brick, wood), and for some mice the families are swapped, so the role is the cross-mouse label. TrialStim is NOT used: in a swap view it contains the placeholder 'stimulus_of_trial' for trials outside that view's stimulus set." The AI demonstrated this on `DR10_2022_07_30_1_swap1/2`, where `TrialStim` shows 105 x 'stimulus_of_trial' while `WallName` correctly shows 105 x 'leaf1_swap2', and noted from `TX109_2023_05_13_1` (`UniqWalls` circle1/circle2/circle3/leaf1/leaf2 -> `stim_id` 2/3/4/0/1) that "physical wall names are swapped relative to canonical roles for some mice".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall texture of each trial is mapped through the merged `wall -> stim_id` table to one of 7 canonical roles, named `['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']`, and broadcast across all bins of the trial. Different crops of the same photograph (circle1 vs circle2, leaf1 vs leaf2 vs leaf3) are kept as *separate* categories, and the two spatially shuffled leaf1 variants as two more. Because roles rather than textures are used, physically different textures are pooled under one label across mice (`wood1`, `rock1` and `leaf1` all map to role 2 = "leaf1"; for TX109 the physical `circle1` maps to role 2 = "leaf1" and the physical `leaf1` maps to role 0 = "circle1"). Trials whose texture has no role anywhere (309 `circle3` trials) are dropped rather than labelled. Resulting mix: circle1 0.319, leaf1 0.335, leaf2 0.170, circle2 0.060, leaf3 0.059, leaf1_swap2 0.029, leaf1_swap1 0.027.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
```
```python
            out = np.empty((4, len(fr)), dtype=np.int64)
            out[0] = walls[wall]
```

iii. Final summary: "Physical texture names also differ across mice (leaf/circle, rock/brick, wood) and are swapped for some mice, so the role is the correct cross-mouse label"; and for the exclusions, "the 309 excluded 'circle3' trials have no canonical role in any view". The AI also checked frame-to-trial consistency against `ft_WallID` in all 89 sessions ("0 mismatches") before trusting the label.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']`, the (fractional) imaging-frame index of every lick in the session.

ii.
```python
        lick = np.zeros(nfr, dtype=bool)
        lf = np.floor(beh['LickFr']).astype(int)
        lf = lf[(lf >= 0) & (lf < nfr)]
        lick[lf] = True
```

iii. `LickFr` is already expressed on the imaging-frame grid, so it maps directly onto the neural bins with no interpolation; the AI cross-checked lick rates against the rewarded/unrewarded cohorts before accepting the result.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A per-frame boolean is built for the whole session: a frame is 1 if at least one lick falls in it. Fractional lick frames are floored; licks outside `[0, n_imaged_frames)` are discarded. The per-trial output is this flag at the trial's retained frames. Dataset-wide 3.7% of bins are licks, and licking is identically zero in 61 of 89 sessions.

ii.
```python
            out[1] = lick[fr].astype(np.int64)
```
```python
    'output_values': [STIM_NAMES,
                      ['no_lick', 'lick'], ...
```

iii. Step 44: "The 61 zero-lick sessions are exactly the unrewarded (isRew==0) unsupervised/naive recordings - mice that were never water-restricted and so genuinely do not lick. This is a real property of the data, not a mapping bug: the 28 task sessions have lick fractions up to 0.39. So the lick output is correct as computed." The AI flagged this explicitly in its final report.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Through the same frame index array as the neural columns, so it is exactly aligned bin-for-bin. One consequence of the running-only frame selection is that licks that occurred while the mouse was stationary inside the corridor are dropped with their frames: the AI's own measurement showed only 57-74% of licks fall in running frames.

ii.
```python
            neural_s.append(spk[:, fr].astype(np.float32))
            out[1] = lick[fr].astype(np.int64)
```

iii. Same alignment principle as all other streams (frame-index grid). The AI measured "licks in moving frac" per session (0.566-0.736) while deciding on the running filter, and accepted the loss, citing the paper's statement that discarding non-running timepoints "removed time periods when the task mice stopped to collect water rewards".

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the within-corridor position at every imaging frame, in decimetres (0-40 inside the texture, 40-60 in the grey space).

ii.
```python
        pos = beh['ft_Pos'][:nfr]
        ...
            out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. The AI verified the units and the range empirically: for `ft_CorrSpc` frames "corr pos overall 0.0 39.97" and for `ft_GraySpc` frames "gray pos overall 40.0 59.99", concluding that "corridor = 60 dm = 40 dm texture + 20 dm gray, so the requested 4 x 1-m position bins correspond exactly to the 4-m texture region".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimetres is divided by 10 and truncated to an integer, giving metre bins. No smoothing or interpolation. The value is time-varying, one per bin.

ii.
```python
CORRIDOR_BINS = 4       # 4 x 1 m bins over the 4 m textured corridor
```
```python
            out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. Docstring: "Trials: aligned to corridor entry and restricted to the 4 m textured corridor (ft_CorrSpc, position 0-40 dm)... This is the window the 4 x 1 m position bins of the decoder task describe."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length 1 m bins by flooring at whole metres, labelled `['0-1m','1-2m','2-3m','3-4m']`, with a `np.clip` guard at 3 for the frames that sit exactly at 40.0 dm. Because the VR advances at a constant speed while the mouse runs and only in-texture frames are kept, the four bins are almost exactly equally occupied in the converted data (0.250 / 0.249 / 0.250 / 0.252).

ii.
```python
            out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```
```python
                              ['0-1m', '1-2m', '2-3m', '3-4m'],
```

iii. The decoder task prescribes "4 equal-length, 1-m-long spatial bins", which the AI matched to the 4 m texture region it had verified `ft_CorrSpc` to span.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame, sliced with the same `fr` array as the neural columns, hence exactly aligned.

ii.
```python
            neural_s.append(spk[:, fr].astype(np.float32))
            out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. Common frame-index grid, as for every other stream.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the running speed at every imaging frame (cm/s).

ii.
```python
        spd = beh['ft_RunSpeed'][:nfr]
```
```python
        spd = beh['ft_RunSpeed'][:nfr]
        for t, fr in frames.items():
            if len(fr) >= MIN_FRAMES_PER_TRIAL and str(beh['WallName'][t]) in recs[name]['walls']:
                speeds.append(spd[fr])
```

iii. Used directly; no derivation from ball-tracking fields was needed since `ft_RunSpeed` is already on the frame grid.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behaviour-only first pass over **all 89 sessions** collects the speed of every frame that will be retained (same trial and frame filters as the main pass) and computes three **global** quartile edges with `np.percentile(speeds, [25,50,75])` = [12.399, 25.285, 40.753] cm/s. Every frame is then assigned a bin with `np.searchsorted(edges, spd, side='right')`. The edges are the same for all sessions and are stored in metadata. Globally the four bins each hold exactly 25.0% of the bins; per session they can be far from equal (the first DR10 sessions are 46/22/20/11%).

ii.
```python
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed quartile edges (cm/s):', speed_edges.round(3), flush=True)
```
```python
            out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```
```python
        'speed_bin_edges_cm_per_s': speed_edges.tolist(),
```

iii. The decoder task asks for "4 bins, each corresponding to 25% of the data", which the AI read as quartiles of the dataset as a whole; it therefore ran a cheap behaviour-only pass first so that one common set of edges (also reported in metadata) could be applied to every session. The AI noticed and accepted the per-session consequence: "the first DR10 session has 96% of frames in the slowest speed quartile, consistent with its 'VR move when non-run' note" (before the running filter was final).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global thresholds (12.399, 25.285, 40.753 cm/s) with `side='right'`, giving 4 categories named `['Q1_slowest','Q2','Q3','Q4_fastest']`. Since only running frames survive, there is no large pile-up of zero speeds to create ties at an edge.

ii.
```python
                              ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest']],
```
```python
            out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. As in 10-b: quartiles of the pooled retained data, applied uniformly so that a given speed means the same category in every session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is per imaging frame and is sliced with the same `fr` array as the neural columns, so it is exactly aligned.

ii.
```python
            neural_s.append(spk[:, fr].astype(np.float32))
            out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. Common frame-index grid, as for every other stream.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive steps: every behaviour stream is cut to the number of imaged frames (`nfr = blocks[0].shape[1]`); licks outside `[0, nfr)` are dropped and fractional lick frames floored; frames with `NaN` trial index are excluded; textures whose `stim_id` is `NaN` are excluded from the wall map (and their trials dropped); zero-variance neurons get `sd = 1` before z-scoring; trials with fewer than 5 retained frames are dropped; and a hard `assert` checks that the plane neuron counts sum to the retinotopy length. There is no per-session `try/except`, so an unexpected session would abort the run. In practice the dataset proved clean: the behaviour pass and the neural pass agree exactly (815,506 bins in both), 0 NaN event frames, `SoundFr` inside `[StartFr, GrayFr]` for all 38,110 trials, and 0 `ft_WallID`/`WallName` mismatches.

ii.
```python
        nfr = blocks[0].shape[1]
        ...
        lf = np.floor(beh['LickFr']).astype(int)
        lf = lf[(lf >= 0) & (lf < nfr)]
```
```python
    ok = corr & moving & ~np.isnan(tr)
```
```python
        sd = spk.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
```

iii. The AI's audits (steps 35-39, 54-56) were explicitly aimed at finding data problems before writing the converter: "Audit is clean: 89 recordings, no NaN event frames, ~21.6 running frames per 4-m trial, only 1 degenerate trial." When it did find suspicious values (time inputs of +/-1700 s) it traced them to genuine mid-corridor pauses rather than corrupt data, and changed the time convention rather than discarding data.

## 12-a. What are the most time-consuming steps of the code?

i. By far the dominant cost is reading the 405 GB of deconvolved traces: `np.load` of each `spk/<session>_neural_data.npy` (2.2-7.4 GB per session) pulls the whole file into memory even though only 2000 of its rows are kept. Second is the behaviour pre-pass, which loads all 23 `Beh_*.npy` files (5.2 GB) and keeps one behaviour dict per recording resident for the whole run, and which walks every trial of every session in Python. Third is pickling the 6.6 GB result. The per-session z-scoring of a 2000 x ~50,000 matrix and the per-frame Python loop in `trial_frames` are minor by comparison.

ii.
```python
        blocks = np.load(spkfile, allow_pickle=True).item()['spks']
```
```python
    recs = collect_recordings()   # loads every Beh_*.npy and keeps one beh per recording
```
```python
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
```

iii. The AI was aware of the I/O scale from the start ("Large dataset: 89 neural sessions (405GB total!)") and used it as its main argument for subsampling neurons; it ran the conversion with `nohup` in the background and polled, reporting "~15 sessions/minute of wall-clock including the big reads" (an over-estimate; the full run took roughly half an hour).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. `trial_frames` groups frames by trial with an explicit Python loop over every retained frame of the session (hundreds of thousands of iterations per session), after an `argsort`; the whole function could be one `np.argsort` + `np.split`, or simply `np.flatnonzero(ok & (tr == t))` per trial. The area mapping `np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])` is a Python loop over 20k-90k neurons that `np.isin` would do in one pass. The per-trial loops building `inp`/`out` and the behaviour pre-pass's per-trial loop are also per-trial Python. None of this is material next to the 405 GB of file reads.

ii.
```python
    groups = collections.defaultdict(list)
    for i in idx:
        groups[int(tr[i])].append(i)
    return {t: np.sort(np.array(v)) for t, v in groups.items()}
```
```python
        area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
```

iii. Not discussed by the AI; it optimised for getting a correct conversion rather than for CPU time, which is reasonable given that the run is I/O bound.

## 12-c. What processing does the code repeat multiple times?

i. `trial_frames()` is computed twice for every session - once in the speed-quartile pre-pass and again in the main loop - and the behaviour files are traversed twice (once to build `recs`, once for the pre-pass). Inside `trial_frames` the frames are sorted twice: they are gathered in `argsort(tr)` order and then each group is `np.sort`ed again, making the initial `argsort` redundant. The retinotopy `iarea` is re-read per session even for mice with several sessions on the same date (only one date per file, so this is minimal). `np.median(np.diff(ft))` is computed twice per session (once as `bin_s`, once for `bin_ms`).

ii.
```python
        frames = trial_frames(beh, nfr)        # pass 1: speed quartiles
        ...
        frames = trial_frames(beh, nfr)        # pass 2: conversion
```
```python
    order = np.argsort(tr[ok], kind='stable')
    idx = np.where(ok)[0][order]
    ...
    return {t: np.sort(np.array(v)) for t, v in groups.items()}
```
```python
        bin_s = float(np.median(np.diff(ft)))
        ...
        bin_ms.append(np.median(np.diff(ft)) * 1000.0)
```

iii. The duplicate trial-frame computation follows from the decision to use *global* speed quartiles, which cannot be known until all behaviour has been seen; the AI kept the two passes deliberately consistent (same trial filters in both) so the edges describe the retained data. The redundant sorting was not remarked on.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, but some: the pre-pass computes retained frames and speeds for all 89 sessions even when `--limit` converts only a few; whole spike files are read and then 96% of their rows are thrown away (unavoidable given the file format, but the `assert sum(counts) == len(iarea)` and the per-plane index arithmetic exist only to support the subsample); the z-score statistics are computed over all frames of the recording, including the ~40% of in-corridor frames and all grey-space frames that are never written out; `experiment_types` and `n_neurons_recorded` are collected per session for metadata only; `bin_ms` is accumulated per session but only its median is stored, while each session's own `bin_s` is what actually shapes the time inputs; and the `argsort` in `trial_frames` is overwritten by a later `np.sort`.

ii.
```python
    if args.limit:
        names = names[:args.limit]        # applied *after* the global speed pass
```
```python
        mu = spk.mean(axis=1, keepdims=True)
        sd = spk.std(axis=1, keepdims=True)
```
```python
        session_info.append({'session': name, 'mouse': mouse, 'date': date, 'block': blk,
                             'experiment_types': sorted(set(recs[name]['exps'])),
                             ...
                             'n_neurons_recorded': int(len(iarea))})
```

iii. The pre-pass is intentionally run over the whole dataset so that a limited run and a full run share the same speed bin edges. Z-scoring over the whole recording (rather than over the retained frames) is deliberate: it is what `utils.get_kfold_reward_response` does. The extra metadata is retained as provenance for the grader.
