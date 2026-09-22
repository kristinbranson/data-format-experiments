# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from `/app/data`, which holds three subfolders: `beh` (behaviour, one `Beh_<exp_type>.npy` per experiment type plus the master index `Imaging_Exp_info.npy`), `spk` (one `<mouse>_<date>_<blk>_neural_data.npy` per recording, holding a list of per-imaging-plane deconvolved trace matrices) and `retinotopy` (one `<mouse>_<date>_trans.npz` per recording, holding the visual-area code `iarea` of every neuron). `load_sessions()` reads the master index first, then reads **all 23 behaviour files** and, for each entry of the index, pulls out that recording's behaviour dict under the key `<mname>_<datexp>_<blk>` (with `_<stimtype>` appended for swap sessions). All behaviour is therefore held in memory before the main pass. The spike file and the retinotopy file are then read once per session inside the main loop. All 89 recordings / 19 mice are converted; nothing is sub-selected except via the debug flag `--nsessions`.

ii.
```python
def load_sessions():
    exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    sessions = {}
    for exp_type in exp_info:
        Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                      allow_pickle=True).item()
        for db in exp_info[exp_type]:
            key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
            bkey = key + ('_' + db['stimtype'] if 'stimtype' in db else '')
            beh = Beh[bkey]
            s = sessions.setdefault(key, {'beh': beh, 'db': db, 'stim_map': {},
                                          'exp_types': []})
```
```python
        ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                                   % (db['mname'], db['datexp'])), allow_pickle=True)
        aidx = area_index(ret['iarea'])

        spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
                       allow_pickle=True).item()['spks']
```

iii. From the trajectory: the agent dumped `data_process_script.ipynb` and `utils.py` (notably `utils.load_spk` and `utils.neu_area_ID`), confirmed that the master index is grouped by experiment type and that "89 unique sessions from 19 mice, matching the 89 spk files", and that the behaviour dicts of a recording listed under several experiment types are byte-for-byte equivalent ("All duplicate session behavior entries are identical"). The behaviour files are small, so reading all of them up front is cheap; the spike files are ~5 GB each (405 GB in total, ~10 s each) so they are read one at a time and released.

## 1-b. How are the data split into subjects (mice)?

i. The mouse name is read straight off the index entry (`db['mname']`); no splitting has to be derived. `subjects` is the sorted list of unique names over the converted sessions (19 mice) and `subject_idx` is each session's position in that list, in the same order as `neural`/`input`/`output`.

ii.
```python
subjects = sorted(set(sessions[k]['db']['mname'] for k in keys))
...
data['subject_idx'].append(subjects.index(db['mname']))
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=int)
```

iii. No explicit justification beyond the fact that the index names the mouse of every recording; the agent verified 19 mice and 89 sessions against the paper ("We performed 89 recordings in 19 mice").

## 1-c. How are the data split into sessions?

i. A session is one recording: one mouse, one date, one block, keyed as `<mname>_<datexp>_<blk>`, which is also the name of the spike file. The same recording is listed under several experiment types in the index; `sessions.setdefault(key, ...)` keeps the behaviour of the first occurrence and ignores the rest, so 89 unique sessions result. The extra occurrences are still visited, but only to accumulate the wall-name → canonical-stimulus map and the list of experiment types (stored in `session_info`).

ii.
```python
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
bkey = key + ('_' + db['stimtype'] if 'stimtype' in db else '')
beh = Beh[bkey]
s = sessions.setdefault(key, {'beh': beh, 'db': db, 'stim_map': {}, 'exp_types': []})
s['exp_types'].append(exp_type)
```

iii. The docstring states it explicitly: "The same recording appears in several `Beh_<exptype>.npy` files (once per analysis it is used for); the behaviour is identical in all of them (verified), but each file only names the stimuli that belong to that analysis, so the canonical names are collected across all files." The trajectory shows the verification: a signature of `(ntrials, UniqWalls, sum(StartFr))` was compared across every duplicate and none differed.

## 1-d. How are the data split into trials?

i. A trial is one corridor traversal. Frames are assigned to trials with the behaviour's own per-frame trial index `ft_trInd`, restricted to (a) frames inside the 4-m textured section (`ft_CorrSpc`) and (b) frames in which the VR was moving, i.e. the animal was running (`ft_move > 0`). Frames with a NaN trial index (between corridor and grey space) are dropped. Grouping is done in one vectorised pass (`argsort` + `searchsorted`) rather than one scan per trial. Trials keep their own length (11–178 bins, median 21); nothing is padded or truncated, and the grey-space section between corridors is excluded entirely.

The running mask is the substantive difference from the expert solution, which keeps every in-corridor frame: it keeps 825,783 of the 1,378,029 in-corridor frames (59.9%). Because the virtual corridor advances at a fixed 60 cm/s whenever the mouse runs above threshold, removing the non-running frames makes traversals nearly constant length (4 m / 60 cm s⁻¹ ≈ 6.7 s ≈ 21 frames) — but the kept frames of a trial are no longer contiguous in wall-clock time.

ii.
```python
def trial_frames(beh, nfr):
    """Indices of the analysed frames of every trial (running, inside the texture)."""
    ft_tr = beh['ft_trInd'][:nfr]
    keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
    idx = np.where(keep)[0]
    tr = ft_tr[idx]
    ok = ~np.isnan(tr)
    idx, tr = idx[ok], tr[ok].astype(int)
    order = np.argsort(tr, kind='stable')
    idx, tr = idx[order], tr[order]
    bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
    return [idx[bounds[t]:bounds[t + 1]] for t in range(int(beh['ntrials']))]
```

iii. Stated in the module docstring: "Trials are corridor traversals, temporally aligned to corridor entry. Only the 4-m textured part of the corridor is kept (`beh['ft_CorrSpc']`), which is exactly the part that the decoder's position output refers to." and "Only frames in which the animal was running are kept (`beh['ft_move'] > 0`, i.e. the VR was moving): 'We only considered timepoints during running for analysis' (methods.txt); the same mask (VRmove & ft_CorrSpc) is used throughout utils.py." The trajectory confirms the agent read `utils.py` lines 427–433 where `fr_valid = VRmove & isCorridor` and checked the resulting per-trial frame counts before committing.

## 1-e. How are trials filtered based on quality controls?

i. Two trial-level filters, both inside the per-session loop:
 - a trial with fewer than 2 kept (running, in-corridor) frames is dropped — in practice this removes nothing, since the shortest traversal in the dataset has 11 running frames;
 - a trial whose wall name cannot be resolved to one of the seven canonical stimulus names is dropped. This removes the 309 `circle3` trials (0.8% of 38,110), which appear in 4 sessions and are never given a canonical `stim_id` in the index.

No session-level filter is applied (no session ends up empty; the smallest keeps 84 trials), and no duration/outlier filter is applied: unlike the expert solution, there is no cap on trial length. The running mask removes the stationary frames that make the expert's outlier trials pathological, but the residual wall-clock span of such trials survives in the time inputs (`time_since_trial_start` reaches 1,765 s against a 99th percentile of 60 s).

ii.
```python
        for t, idx in enumerate(tidx):
            if len(idx) < 2:
                nskip += 1
                continue
            name = smap.get(str(beh['WallName'][t]), None)
            if name is None or name not in STIM_NAMES:
                nskip += 1
                continue
```

iii. The trajectory shows the decision being taken deliberately: "Decision: keep the paper's 7 canonical stimulus categories and drop the 309 'circle3' trials (a texture that never enters any analysis in the paper and has no canonical id)." The `<2 frames` guard is a defensive minimum so that a trial always has more than one time bin. The agent did measure the long-duration problem ("'time since trial start' and 'time to cue' can reach ±300 s when the mouse stands still for long periods, since only running frames are kept but wall-clock time continues. I want to quantify how extreme this is") and, after printing the percentiles, chose to keep those trials without further comment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session>_neural_data.npy` — a list of one (neurons × frames) float32 matrix per imaging plane, implicitly concatenated in plane order (the same order the retinotopy file indexes neurons in). The per-neuron visual-area label comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`. Nothing else feeds `neural`.

ii.
```python
        ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                                   % (db['mname'], db['datexp'])), allow_pickle=True)
        aidx = area_index(ret['iarea'])

        spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
                       allow_pickle=True).item()['spks']
        nneu_all = sum(s.shape[0] for s in spks)
        assert nneu_all == len(aidx), (k, nneu_all, len(aidx))
```

iii. "Neural data are the raw deconvolved traces ('spks'), one time bin per imaging frame (~315 ms). Planes are concatenated as in `utils.load_spk`." The agent verified the plane structure and the neuron-count/retinotopy correspondence on a sample file (81,473 neurons over 3 planes; `iarea` of length 81,473) and made the equality an assertion in the converter.

## 2-b. How is the `neural` data processed?

i. Not processed at all: the deconvolved traces are used as they are stored, kept as float32, one column per imaging frame. No dF/F, deconvolution, smoothing, normalisation, z-scoring or spatial-position interpolation is applied (the paper's `get_interpPos_spk` resampling onto corridor position is deliberately not used). The only manipulation is a gather: the selected neurons' rows are copied plane by plane into one array, and then the columns of each trial are sliced out and made contiguous.

ii.
```python
        spk = np.empty((len(sel), spks[0].shape[1]), dtype=np.float32)
        off, o2 = 0, 0
        for p in range(len(spks)):
            n = spks[p].shape[0]
            take = sel[(sel >= off) & (sel < off + n)] - off
            spk[o2:o2 + len(take)] = spks[p][take]
            o2 += len(take)
            off += n
        del spks
...
            neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The paper states "All our analyses were based on deconvolved fluorescence traces", so the stored `spks` are already the analysis-ready signal; the metadata records this as `'neural_data': 'suite2p deconvolved calcium activity (non-negative deconvolution, 0.75 s decay), one bin per imaging frame'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters:
 - **Area assignment (matches the reference).** `area_index` reproduces `utils.neu_area_ID` exactly: `iarea == 8` → V1, `{0,1,2,9}` → mHV, `{5,6}` → lHV, `{3,4}` → aHV; any other code (including unassigned) gives −1 and the neuron is dropped. Across the dataset this keeps ~4.1 M of ~4.7 M neurons (median 47,246 of 52,954 per session).
 - **Random subsampling (does not match the reference).** Of the area-assigned neurons, at most `NNEURONS = 1000` per session are kept, drawn without replacement with a fixed seed and then sorted so plane order is preserved. Every session ends up with exactly 1000 neurons, i.e. ~2% of the qualifying neurons (89,000 in total against the expert's ~4.1 M). No other quality control (no activity threshold, no SNR or cell-classifier filter) is applied — Suite2p's cell classifier was already applied by the authors.

ii.
```python
def area_index(iarea):
    """Map suite2p/retinotopy area ids onto the 4 area groups of utils.neu_area_ID."""
    out = -np.ones(len(iarea), dtype=int)
    out[iarea == 8] = 0                                   # V1
    out[np.isin(iarea, [0, 1, 2, 9])] = 1                  # medial HV
    out[np.isin(iarea, [5, 6])] = 2                        # lateral HV
    out[np.isin(iarea, [3, 4])] = 3                        # anterior HV
    return out
```
```python
        valid = np.where(aidx >= 0)[0]
        if len(valid) > args.nneurons:
            sel = np.sort(rng.choice(valid, args.nneurons, replace=False))
        else:
            sel = valid
```

iii. Docstring: "Neurons are kept only if the retinotopy file assigns them to one of the four visual-area groups used in the paper (V1, mHV, lHV, aHV; `utils.neu_area_ID`). For tractability a random subset of at most NNEURONS neurons per session is kept (the raw data are 405 GB)." The trajectory shows the sizing argument: "total kept frames = 825,783 across 89 sessions; with ~50k neurons each that would be >100 TB" — an over-estimate, since the real cost is Σ(neurons × that session's own frames) ≈ 100 GB in float16, not 100 TB. The choice of exactly 1000 is not otherwise justified or tested, and it is recorded transparently in the metadata (`'neuron_selection': 'neurons assigned to V1/mHV/lHV/aHV by the retinotopy mapping; random subset of at most 1000 per session'`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial's neural matrix begins at the first kept frame after corridor entry and runs to the last kept frame before the grey space; trials keep their natural length, nothing is padded or truncated, and no common window is imposed. The metadata declares `temporal_alignment_event = 'trial start = entry into the textured corridor'`, `off_start = 0.0`, `off_end = None`. All other streams are sliced with the same frame index array `idx`, so inputs, outputs and neural data are aligned by construction. Because the running mask removes the stationary frames, bin *k* of a trial is not at a fixed latency from corridor entry — but the mouse's *position* advances at a fixed rate while running, so bin *k* is at a near-fixed distance into the corridor.

ii.
```python
        tidx = trial_frames(beh, nfr)
        ...
        for t, idx in enumerate(tidx):
            ...
            neural_s.append(np.ascontiguousarray(spk[:, idx]))
```
```python
        'temporal_alignment_event': 'trial start = entry into the textured corridor',
        'off_start': 0.0,
        'off_end': None,
```

iii. "Trials are corridor traversals, temporally aligned to corridor entry." The format spec requires only a common bin *size*, and allows `off_end = None`, so variable-length trials are legal; the decoder concatenates all time bins of a session and treats each as an independent sample, so no common window is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling or smoothing: one time bin per two-photon imaging frame. The bin size is measured from the data rather than hard-coded — the median inter-frame interval of each session is collected and the median over sessions (314.69 ms, i.e. 3.177 Hz) is written to `metadata['time_bin_size']` along with `sampling_rate_hz`. All behavioural streams (`ft_Pos`, `ft_RunSpeed`, `ft_move`, `ft_trInd`) are already sampled on this same frame grid, and lick and cue times are expressed in frame units, so nothing has to be resampled.

ii.
```python
        dts.append(np.median(np.diff(ft)) * SEC_PER_DAY)
...
        'time_bin_size': float(np.median(dts) * 1000.0),
...
        'sampling_rate_hz': float(1000.0 / (np.median(dts) * 1000.0)),
```

iii. "Neural data are the raw deconvolved traces ('spks'), one time bin per imaging frame (~315 ms)"; the imaging frame is the finest resolution available and the paper's own analyses are carried out frame by frame.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime`, the MATLAB datenum timestamp of the sound cue on each trial, and `ft`, the datenum timestamp of every imaging frame. (The equivalent fractional frame number `SoundFr` is not used.)

ii.
```python
            tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
            inp[0] = tcue
```

iii. Not argued explicitly; `SoundTime` is the cue's own timestamp on the same clock as `ft`, so no interpolation between frames is needed. The agent verified that the cue always falls inside the corridor ("cue before start 0, cue after gray 0" for every checked session) and that there are no NaNs in `SoundFr`/`SoundTime`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. One subtraction per bin: cue timestamp minus frame timestamp, converted from days to seconds (×86400). The result is positive before the cue and negative after it, matching the name "time **to** sound cue", and it is stored as a time-varying float32 row. No clipping, normalisation or binarisation is applied; values range from −1763 s to +724 s (1st–99th percentile −19.7 s to +21.3 s), the long tails coming from trials in which the mouse stood still for many minutes mid-corridor while wall-clock time kept running.

ii.
```python
SEC_PER_DAY = 86400.0      # beh times are MATLAB datenums (days)
...
            tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
```

iii. Metadata: `'time_to_sound_cue': 'seconds until the sound cue (negative after the cue)'`. The agent noticed the heavy tails at step 49 and quantified them but left the variable unclipped.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the frame timestamps `ft[idx]` of the same index array `idx` that slices the neural columns, so it is a time-varying row of the same length as that trial's neural matrix.

ii.
```python
        for t, idx in enumerate(tidx):
            ...
            T = len(idx)
            tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
            inp = np.empty((4, T), dtype=np.float32)
            inp[0] = tcue
            ...
            neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Every stream in this dataset lives on the imaging-frame grid, so using the same `idx` for all of them guarantees alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the recording date string `datexp` in the index entry (`YYYY_MM_DD`), together with the mouse name `mname`. Nothing in the behaviour file is used.

ii.
```python
        d = datetime.date(*map(int, db['datexp'].split('_')))
        if db['mname'] not in first_date or d < first_date[db['mname']]:
            first_date[db['mname']] = d
```

iii. The agent grepped the paper code for a training-day field, found none, and printed the per-mouse date offsets before deciding (e.g. `TX108 [0, 67, 76, 79, 86, 89, 92]`, `TX119 [0, 1, 2, 11, 12, 25, 26, 28]`).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days elapsed since that mouse's **first imaging session**: a first pass over the behaviour finds `first_date[mouse]`, then each session's value is `(date − first_date).days`, a float in 0…92 that is constant within a session and broadcast across all bins of every trial. It is an elapsed-time measure, not an ordinal session count, so the gaps between recordings (weeks, in several mice) are preserved.

ii.
```python
        day = float((datetime.date(*map(int, db['datexp'].split('_')))
                     - first_date[db['mname']]).days)
...
            inp[1] = day
```

iii. Metadata: `'day_of_training': 'days elapsed since the first imaging session of that mouse'`. No further argument is given; the true start of behavioural training is not in the data, so the first recording is used as the origin.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time`, the datenum timestamp of corridor entry for each trial, and `ft`, the timestamp of each imaging frame. (`StartFr`, the fractional frame number of corridor entry, is not used; the two agree — e.g. `StartFr[0] = 5.729` interpolates to 738314.316183, and `Trial_start_time[0] = 738314.31618297`.)

ii.
```python
            tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
```

iii. Not argued explicitly; `Trial_start_time` is the exact entry time on the same clock as the frame times, so no interpolation is needed.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. One subtraction per bin, converted from days to seconds, positive and starting near zero (the first kept frame is at or after entry). Stored as a time-varying float32 row. No clipping: the median is 3.84 s, the 99th percentile 59.9 s and the maximum 1765 s, the tail again coming from trials in which the mouse stopped for a long time inside the corridor (those stationary frames are removed by the running mask, but the elapsed time they span is not).

ii.
```python
            tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
            inp[2] = tt
```

iii. Metadata: `'time_since_trial_start': 'seconds since entry into the corridor'`. The agent measured the distribution of this variable on a 3-session test before running the full conversion.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Evaluated at the frame timestamps `ft[idx]` of the same `idx` used for the neural columns, so it is a time-varying row of that trial's length whose first element is the latency of the first kept frame after corridor entry.

ii.
```python
            T = len(idx)
            tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
            inp = np.empty((4, T), dtype=np.float32)
            inp[2] = tt
```

iii. Same as 3-c: one frame grid for every stream, one index array per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial boolean flag marking the rewarded corridor. It is true only for task ("sup") mice; unsupervised, grating and naive mice have it false throughout, because they ran the same corridors with no water available.

ii.
```python
            inp[3] = float(beh['isRew'][t])
```

iii. The agent explicitly checked the semantics before committing (step 50): it cross-tabulated `isRew` against `TrialStim` and against whether a reward was actually delivered (`RewTime` non-NaN), and concluded "isRew marks the rewarded corridor (task mice only), matching the required 'reward availability' input" — i.e. it encodes availability, not delivery, which is what the instructions ask for.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast from bool to float and a broadcast across the trial's bins, giving a constant 0/1 row per trial. Over the whole dataset it is 1 in ~11% of bins.

ii.
```python
            inp = np.empty((4, T), dtype=np.float32)
            ...
            inp[3] = float(beh['isRew'][t])
```

iii. The flag is already the required binary per-trial variable, so no processing is needed; it is stored as a time-varying row only because all four inputs are packed into one `(4, T)` array.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim`, keyed by `WallName`. `WallName` gives the physical texture shown on the walls of each trial (15 distinct names across the dataset: `circle1-3`, `leaf1-3`, `leaf1_swap1/2`, `rock1/2`, `wood1/2/5`, `wood1_swap1/2`); `TrialStim` gives the paper's *canonical* — i.e. **functional** — name of that stimulus, which is masked with the placeholder `'stimulus_of_trial'` in behaviour files where that stimulus is not part of the analysis. The converter therefore builds, per session, a `WallName → TrialStim` map from all unmasked entries across all the behaviour files that contain that session, and looks each trial's wall name up in it.

Note what this mapping does: it is not a physical-texture identity. Across the dataset `rock1 → circle1` (4230 trials), `wood1 → leaf1` (4180), `wood2 → leaf2`, `wood5 → leaf3`, and `circle1 → leaf1` in 811 trials of mice for which circle was the trained/rewarded stimulus. So the same physical texture carries different labels in different mice, and physically unrelated textures share a label. Within a session the map is consistent and injective (verified: no wall name maps to two canonical names, and no two wall names collapse onto one canonical name in the same session).

ii.
```python
            # wall name -> canonical stimulus name
            for w, c in zip(beh['WallName'], beh['TrialStim']):
                if str(c) != 'stimulus_of_trial':
                    s['stim_map'][str(w)] = str(c)
```
```python
            name = smap.get(str(beh['WallName'][t]), None)
            if name is None or name not in STIM_NAMES:
                nskip += 1
                continue
```

iii. Docstring: "Stimulus labels use the canonical/functional names of the paper ('circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2'), which are stored per trial in `beh['TrialStim']`. This is essential because the physical textures differ between mice (leaf/circle vs rock/brick...) while their role in the experiment is the same; the decoder is shared across sessions." The trajectory shows the agent discovered the renaming explicitly: "Canonical stimulus labels come from `beh['TrialStim']` (functional renaming: rewarded/trained texture = 'leaf1')".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The canonical name is looked up in the fixed list `STIM_NAMES` of seven categories and stored as its index, broadcast across all bins of the trial as an int64 row. Trials whose name cannot be resolved are dropped rather than given their own category — this is the 309 `circle3` trials (`circle3` is given a canonical name, `leaf3`, in some sessions but is masked in the 4 sessions concerned, and the index gives it `stim_id = nan` there). Resulting distribution over time bins: circle1 0.319, leaf1 0.335, leaf2 0.170, circle2 0.060, leaf3 0.059, leaf1_swap1 0.027, leaf1_swap2 0.029.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
...
            out = np.empty((4, T), dtype=np.int64)
            out[0] = STIM_NAMES.index(name)
```

iii. From the trajectory: "Decision: keep the paper's 7 canonical stimulus categories and drop the 309 'circle3' trials (a texture that never enters any analysis in the paper and has no canonical id)." The agent had first written an 8-category list including `circle3` and edited it out in step 51. Metadata: `'stimulus': 'canonical (functional) identity of the wall texture of the corridor'`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame number of every lick detected in the session. `LickTime`/`LickPos`/`LickTrind` are not used.

ii.
```python
        lickfr = np.round(beh['LickFr']).astype(int) if len(beh['LickFr']) else np.zeros(0, int)
        lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
        is_lick = np.zeros(nfr, dtype=np.int64)
        is_lick[lickfr] = 1
```

iii. `LickFr` is already expressed in imaging-frame units, so it lands on the neural grid with no conversion. The agent sanity-checked it on a task session (3589 licks over 2351 distinct frames, 12% of all frames, 16% of kept frames in that session).

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is built: a bin is 1 if at least one lick falls in it, 0 otherwise. The fractional lick frame is **rounded** to the nearest frame (the paper's own code uses `beh['LickFr'].astype(int)`, i.e. truncation, so this differs by at most one 315 ms bin for licks whose fractional part exceeds 0.5). Licks outside the imaged range are discarded. Over the converted dataset 3.5% of bins are licks — restricting to running frames barely changes this (3.52% over all in-corridor frames vs 3.46% over running in-corridor frames), because most sessions are unsupervised/naive mice that never lick.

ii.
```python
        lickfr = np.round(beh['LickFr']).astype(int) if len(beh['LickFr']) else np.zeros(0, int)
        lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
        is_lick = np.zeros(nfr, dtype=np.int64)
        is_lick[lickfr] = 1
...
            out[1] = is_lick[idx]
```

iii. Metadata: `'lick': '1 if the animal licked during this time bin'`. The instructions ask for a binary time series for an event time, which is what this produces; no justification is given for rounding rather than truncating.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick flag is defined on the imaging-frame grid and indexed with the same `idx` as the neural columns, so it has exactly that trial's length and each element refers to the same frame as the corresponding neural column. Licks that occurred while the mouse was stationary fall on frames that are not kept and are therefore dropped along with them.

ii.
```python
            out[1] = is_lick[idx]
            ...
            neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Same as the other streams: one shared frame index per trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the per-frame position along the corridor in decimetres (0–40 inside the 4 m texture, 40–60 in the 2 m grey space). Only the in-corridor portion is ever used, since the trial frames are restricted to `ft_CorrSpc`.

ii.
```python
        pos = beh['ft_Pos'][:nfr]
...
            out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)  # 4 x 1 m bins
```

iii. The agent checked the units and ranges before using them ("pos in corr range ~0 to 39.98, gray pos range 40.0 to 59.99"), confirming decimetres and a 4 m textured section.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Divide by 10 to convert decimetres to metres, truncate to an integer, clip to 0…3, store as an int64 row per bin. No smoothing or interpolation.

ii.
```python
            out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)  # 4 x 1 m bins
```

iii. Metadata records the bin edges explicitly: `'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0]` and `'position_bin': 'position along the 4 m corridor in 4 x 1 m bins'`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Into the four equal-length 1 m bins the instructions require: [0,1), [1,2), [2,3), [3,4] m, i.e. fixed thresholds at 10, 20 and 30 dm, with the clip folding an exact 40 dm reading into the last bin. The resulting distribution is almost exactly uniform (0.2500 / 0.2488 / 0.2496 / 0.2517), as expected for a corridor traversed at constant VR speed.

ii.
```python
            'output_values': [STIM_NAMES,
                              ['no_lick', 'lick'],
                              ['0-1m', '1-2m', '2-3m', '3-4m'],
                              ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']],
```
```python
            out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. Directly from the Decoder Task spec: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins".

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sampled once per imaging frame, so it already lives on the neural grid; it is indexed with the same `idx` as the neural columns and has that trial's length.

ii.
```python
            out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
            ...
            neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Same as the other streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the per-frame running speed of the mouse (cm/s, from the optical ball sensor; it can be slightly negative). This is the animal's real running speed, not the fixed virtual-corridor speed.

ii.
```python
        speed = beh['ft_RunSpeed'][:nfr]
...
            out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. No separate justification; it is the only running-speed variable in the behaviour dict.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A first pass over all 89 behaviour files pools `ft_RunSpeed` over every frame that will be analysed (running, inside the corridor) and takes the 25th/50th/75th percentiles of that pooled distribution as three global thresholds (12.39, 25.33, 40.83 cm/s). In the second pass each frame's speed is mapped to a bin with `np.searchsorted`. The edges are **global** (one set for the whole dataset), not per session, and are recorded in the metadata.

ii.
```python
    speeds = []
    for k in keys:
        beh, db = sessions[k]['beh'], sessions[k]['db']
        nfr = len(beh['ft'])
        keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
        speeds.append(beh['ft_RunSpeed'][:nfr][keep])
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
```
```python
            out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. Metadata: `'speed_bin': 'running-speed quartile (edges from all analysed frames)'` plus `'speed_bin_edges_cm_per_s'`. The edges are computed from exactly the frames that will be kept, so the four bins each hold 25% of the converted dataset (measured: 0.2498 / 0.2511 / 0.2502 / 0.2489). No per-session normalisation is applied.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins separated by the three global quartile thresholds, `searchsorted` assigning a value exactly on an edge to the lower bin. Because the thresholds are global, the split is exactly 25/25/25/25 over the whole dataset but can be strongly skewed within a session — in the 3-session test run the per-session fractions were e.g. (0.117, 0.202, 0.379, 0.302) for one session and (0.600, 0.313, 0.072, 0.015) for another. Note that the running mask means the zero-speed frames that would otherwise pile up on the first threshold are largely already excluded.

ii.
```python
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed quartile edges (cm/s):', speed_edges)
...
            out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. Decoder Task spec: "Running speed discretized into 4 bins, each corresponding to 25% of the data" — read as 25% of the dataset as a whole, which is what pooling the frames across sessions achieves.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is sampled once per imaging frame, so it is already on the neural grid, and it is indexed with the same `idx` as the neural columns.

ii.
```python
            out[3] = np.searchsorted(speed_edges, speed[idx])
            ...
            neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Same as the other streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several small guards, all of them checked against the data first:
 - **Behaviour longer than imaging.** Every behavioural stream is truncated to `nfr = min(n_spike_frames, len(beh['ft']))` before use, so the frame grid can never be longer than the neural recording.
 - **Frames with no trial.** Frames whose `ft_trInd` is NaN (between corridor and grey space) are dropped in `trial_frames`.
 - **Licks outside the imaged range.** Lick frames are clipped to `0 <= f < nfr`; an empty `LickFr` is handled explicitly.
 - **Neuron/retinotopy mismatch.** An assertion checks that the total number of neurons across planes equals the length of `iarea`, so a misaligned area vector fails loudly rather than silently mislabelling neurons.
 - **Degenerate trials.** Trials with fewer than 2 kept frames and trials with an unresolvable stimulus name are skipped, and the count is printed per session.
 There is no `try/except` around a session, so an unexpected failure would abort the whole run; in practice none occurred. NaNs in `ft_Pos` and `ft_RunSpeed` were checked in advance and there are none.

ii.
```python
        nfr = min(spk.shape[1], len(beh['ft']))
```
```python
    ok = ~np.isnan(tr)
    idx, tr = idx[ok], tr[ok].astype(int)
```
```python
        lickfr = np.round(beh['LickFr']).astype(int) if len(beh['LickFr']) else np.zeros(0, int)
        lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
```
```python
        assert nneu_all == len(aidx), (k, nneu_all, len(aidx))
```

iii. The reference code applies the same truncation (`beh[...][:nfr]` with `nfr = spk.shape[1]`). The trajectory shows the agent ran a NaN audit over all 89 sessions ("no NaNs in trial frame markers", "nan pos 0 nan spd 0") before writing the converter, and hit the NaN `ft_trInd` itself in an exploratory script, which is why it is handled.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files: 89 files, ~1.5–9 GB each, 405 GB in total, measured at ~8–10 s per file. Everything else is negligible — the behaviour pre-pass reads all 23 behaviour files (small), and the per-trial work is a few hundred thousand small array slices. A secondary cost is the plane-by-plane fancy-index gather of the 1000 selected rows out of the full ~50,000-row plane matrices, and finally pickling the 3.3 GB result. The whole conversion took roughly 5–6 minutes, i.e. it is disk-throughput-bound.

ii.
```python
        spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
                       allow_pickle=True).item()['spks']
```

iii. The agent measured this explicitly before designing the converter ("Data is huge (405GB, ~5GB per session)", "load+concat 9.945 s (48824, 27969) float32", "Load speed ~10s/file (~15-20 min total)"). The cost is inherent: the files are pickled lists of arrays, so they cannot be memory-mapped and a partial read is not possible.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python loops are all small relative to the I/O:
 - `for w, c in zip(beh['WallName'], beh['TrialStim'])` in `load_sessions` — one iteration per trial per behaviour file (~200 session-entries × up to 722 trials); `np.unique` on the pair of arrays would do the same work in one call.
 - the per-trial loop `for t, idx in enumerate(tidx)` — ~38,000 iterations, each allocating three small arrays; the inputs/outputs could be computed once for the whole session and split by trial.
 - the plane gather loop and the pass-1 loop over sessions, both of which run 3 and 89 times respectively.
 Notably, the part the expert solution leaves as a per-trial scan — finding each trial's frames — is already vectorised here with `argsort` + `searchsorted` instead of one full-length comparison per trial.

ii.
```python
            for w, c in zip(beh['WallName'], beh['TrialStim']):
                if str(c) != 'stimulus_of_trial':
                    s['stim_map'][str(w)] = str(c)
```
```python
    order = np.argsort(tr, kind='stable')
    idx, tr = idx[order], tr[order]
    bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
```

iii. Not discussed by the agent; the vectorised trial grouping is written that way without comment.

## 12-c. What processing does the code repeat multiple times?

i. Two repetitions, both cheap:
 - The mask `(ft_move > 0) & ft_CorrSpc` is computed once per session in pass 1 (to collect speeds) and again inside `trial_frames` in pass 2. The behaviour arrays are already resident, so this is a few milliseconds per session.
 - `load_sessions` visits a recording once per experiment type it appears under (≈200 visits for 89 sessions) and re-walks its full `WallName`/`TrialStim` arrays each time, even though the behaviour dict is identical; only the first visit's behaviour is retained.
 Behaviour files themselves are read only once each, and each spike file is read exactly once.

ii.
```python
        keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
        speeds.append(beh['ft_RunSpeed'][:nfr][keep])
```
```python
    keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
    idx = np.where(keep)[0]
```

iii. The repeated pass is the price of global speed quartiles, which need every session's speeds before any session can be written out; the agent chose to pay it on behaviour only (no spike file is read twice).

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dominant one is reading ~47,000 neurons per session off disk and then throwing away ~98% of them — 405 GB read to produce 3.3 GB. This is forced by the file format (a pickled list of arrays cannot be memory-mapped or partially read), so it cannot be avoided without re-writing the source files, but it does mean the run is dominated by work that is discarded. Smaller items: pass 1 computes the running mask and pooled speeds over `len(beh['ft'])` frames rather than over the frames that survive the `min(spk_frames, beh_frames)` truncation used in pass 2, so a handful of never-used frames enter the quartile estimate; `nneu_all` is summed only for an assertion and a metadata field; `exp_types`, `cohort`, `reward_type` and `nneurons_recorded` are recorded per session but never used downstream; and `np.ascontiguousarray` forces an extra copy of each trial's neural slice.

ii.
```python
        nneu_all = sum(s.shape[0] for s in spks)
        assert nneu_all == len(aidx), (k, nneu_all, len(aidx))
```
```python
        nfr = len(beh['ft'])                       # pass 1
        keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
```
```python
            neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Not discussed by the agent. The metadata extras are deliberate provenance fields (`session_info`), not accidental work.
