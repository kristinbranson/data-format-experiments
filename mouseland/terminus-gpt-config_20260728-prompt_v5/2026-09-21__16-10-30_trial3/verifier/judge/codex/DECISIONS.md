# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every behavior `.npy` file under `/app/data/beh` into one in-memory dictionary keyed by behavior session id, lists all spike-session files under `/app/data/spk`, matches each spike session to a behavior entry by exact id or prefix, and loads retinotopy per processed session. It then discards all non-rewarded/no-lick sessions and only converts the remaining matched sessions.

ii. 
```python
def load_all_behavior_entries():
    entries = {}
    source_file = {}
    for f in sorted(BEH_ROOT.glob('*.npy')):
        dat = np.load(f, allow_pickle=True).item()
        for sid, sess in dat.items():
            entries[sid] = sess
            source_file[sid] = f.name
    return entries, source_file
```
```python
spike_sessions = sorted(f.name.replace('_neural_data.npy', '') for f in SPK_ROOT.glob('*_neural_data.npy'))
...
for sid in spike_sessions:
    b = choose_behavior_for_spike_session(sid, beh_entries)
    if b is not None:
        matched.append((sid, b))
...
if args.sample:
    matched = rewarded[:2] if len(rewarded) >= 2 else matched[:2]
else:
    matched = rewarded
```

iii. The notes and trajectory justify this as using spike files as the session backbone, matching behavior by session id, and then restricting to rewarded sessions because unsupervised sessions produced degenerate decoder targets. The README explicitly says the final dataset is "filtered to rewarded imaging sessions with lick events."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred purely from the first underscore-delimited token of each spike session id. The converted subject list is built incrementally from the filtered sessions.

ii. 
```python
def parse_session_id(session_id):
    parts = session_id.split('_')
    subj = parts[0]
    date = '_'.join(parts[1:4])
    run = parts[4] if len(parts) > 4 else None
    return subj, date, run
```
```python
subjects = []
subject_to_idx = {}
...
subj, date, run = parse_session_id(sid)
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
```

iii. The Step 4 notes say subject ids are encoded in session ids and should be taken from the prefix before the first underscore.

## 1-c. How are the data split into sessions?

i. Sessions are defined by spike files in `/app/data/spk`. For each spike session, the AI chooses one behavior entry by exact match or by a prefixed variant such as a swap session. Only sessions with reward and at least one lick event are kept in the final full dataset.

ii. 
```python
def choose_behavior_for_spike_session(spike_sid, beh_entries):
    if spike_sid in beh_entries:
        return spike_sid
    candidates = [k for k in beh_entries if k == spike_sid or k.startswith(spike_sid + '_')]
    if candidates:
        for c in candidates:
            if 'swap' not in c:
                return c
        return candidates[0]
    return None
```
```python
rewarded = []
for sid, b in matched:
    sess = beh_entries[b]
    isrew = np.asarray(sess.get('isRew', []))
    lickn = len(np.asarray(sess.get('LickFr', [])))
    if isrew.size and bool(isrew.any()) and lickn > 0:
        rewarded.append((sid, b))
```

iii. The notes say spike files are the "session backbone," behavior may contain extra `swap1`/`swap2` variants, and rewarded sessions were preferred after the initial sample run produced all-zero reward and lick outputs.

## 1-d. How are the data split into trials?

i. Trials are not taken from the reference frame-level trial labels. Instead, the AI infers per-trial start and end frame bounds heuristically from `SoundFr`; if that is unavailable, it divides the session uniformly into `ntrials` contiguous chunks. Each inferred trial is then sliced as one contiguous frame block.

ii. 
```python
def infer_trial_frame_bounds(sess, n_frames):
    ntrials = int(sess['ntrials'])
    sound_fr = np.asarray(sess['SoundFr']).astype(int) if 'SoundFr' in sess else None
    if sound_fr is not None and sound_fr.shape[0] == ntrials:
        starts = np.zeros(ntrials, dtype=int)
        starts[1:] = np.maximum(sound_fr[1:] - np.maximum(np.diff(sound_fr), 1), 0)
        ends = np.empty(ntrials, dtype=int)
        ends[:-1] = np.maximum(starts[1:] - 1, starts[:-1])
        ends[-1] = n_frames - 1
        return starts, ends, {'method': 'soundfr_heuristic', 'candidates': candidates}
    edges = np.linspace(0, n_frames, ntrials + 1).astype(int)
    starts = edges[:-1]
    ends = edges[1:] - 1
    return starts, ends, {'method': 'uniform_fallback', 'candidates': candidates}
```
```python
for tr in range(ntrials):
    s = int(max(0, starts[tr]))
    e = int(min(n_frames - 1, max(starts[tr], ends[tr])))
    trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
```

iii. The notes and metadata both acknowledge that trial boundaries are heuristic and "should be refined from behavior variables if available."

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality filtering step. Once trial bounds are inferred, every trial from `0..ntrials-1` is converted. The code only clips slice bounds to valid frame limits and silently tolerates missing/mismatched arrays elsewhere.

ii. 
```python
for tr in range(ntrials):
    s = int(max(0, starts[tr]))
    e = int(min(n_frames - 1, max(starts[tr], ends[tr])))
    trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
```
```python
if 'isRew' in sess:
    reward_avail = np.asarray(sess['isRew']).astype(int)
    if reward_avail.shape[0] != ntrials:
        reward_avail = np.resize(reward_avail, ntrials)
```

iii. No explicit trial-QC justification was documented. The trajectory instead focuses on filtering sessions to rewarded/no-lick-supervised data and on making heuristic bounds "good enough" for validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived from the spike-session file's `spks` arrays, concatenated across source groups/planes. Brain-region indices are derived from retinotopy `iarea`.

ii. 
```python
def load_spike_session(spike_sid):
    obj = np.load(SPK_ROOT / f'{spike_sid}_neural_data.npy', allow_pickle=True).item()
    spks = obj['spks']
    return np.concatenate([x.astype(np.float32, copy=False) for x in spks], axis=0)
```
```python
def load_retino_for_session(spike_sid):
    rid = normalize_session_for_retino(spike_sid)
    path = RET_ROOT / f'{rid}_trans.npz'
    ...
    return {k: z[k] for k in z.keys()}
```

iii. The notes state that `spks` are already processed deconvolved traces from the reference pipeline and retinotopy should be matched by normalized subject+date id.

## 2-b. How is the `neural` data processed?

i. The AI concatenates `spks`, casts to `float32`, then for each inferred trial extracts a contiguous frame range, downsamples by keeping every 5th frame, and stores the result as `float16`.

ii. 
```python
return np.concatenate([x.astype(np.float32, copy=False) for x in spks], axis=0)
```
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
```

iii. The notes and README justify the stride-5 downsampling as a memory/runtime optimization: the agent reports shrinking a sample output from about 12 GB to about 1.4 GB and making full conversion feasible.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no neuron exclusion step. The AI maps `iarea` codes into five labels (`All`, `V1`, `medial`, `anterior`, `lateral`) and keeps every neuron. If retinotopy is missing, all neurons are assigned to region 0.

ii. 
```python
def area_names_and_idx(iarea):
    area_names = ['All', 'V1', 'medial', 'anterior', 'lateral']
    iarea = np.asarray(iarea)
    idx = np.zeros((len(iarea),), dtype=np.int64)
    idx[(iarea == 7) | (iarea == 8)] = 1
    idx[(iarea == 1) | (iarea == 2)] = 2
    idx[(iarea == 3) | (iarea == 4)] = 3
    idx[(iarea == 5) | (iarea == 6)] = 4
    return area_names, idx
```
```python
if ret is not None and 'iarea' in ret:
    _, bri = area_names_and_idx(ret['iarea'])
    if len(bri) != spk.shape[0]:
        bri = np.resize(bri, spk.shape[0])
else:
    bri = np.zeros((spk.shape[0],), dtype=np.int64)
```

iii. The notes identify retinotopy as the source of area information, but they do not justify keeping all neurons; they only note that region mapping may need to be repeated across concatenated `spks` groups.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI claims alignment to trial start/corridor entry, but operationally alignment is to the inferred trial start index `s` from `infer_trial_frame_bounds`, not to `StartFr` or reference corridor-entry labels.

ii. 
```python
'temporal_alignment_event': 'trial start / corridor entry',
```
```python
starts, ends, bounds_info = infer_trial_frame_bounds(sess, n_frames)
...
s = int(max(0, starts[tr]))
e = int(min(n_frames - 1, max(starts[tr], ends[tr])))
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
```

iii. The justification given in notes is that the decoder task required trial-start alignment, but the agent repeatedly acknowledged that actual boundaries were heuristic.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code decimates time by a stride of 5 original frames and records metadata `time_bin_size` as `5.0`. No interpolation or averaging is done; it simply keeps every 5th frame.

ii. 
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
ts = (np.arange(T, dtype=np.float32) * frame_stride)
```
```python
'time_bin_size': 5.0,
```

iii. The notes say stride-5 temporal downsampling was added to reduce memory use and runtime.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and the inferred trial start frame `s`. The code does not use frame timestamps such as `ft`.

ii. 
```python
sound_fr = np.asarray(sess.get('SoundFr', np.full(ntrials, -1)), dtype=int)
...
cue = sound_fr[tr] - s if tr < len(sound_fr) else -1
```

iii. The notes mention cue-related variables such as `SoundFr`, and the trajectory shows the agent choosing this variable because trial-boundary inference was already based on frame numbers.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each downsampled trial, the AI creates a local time axis `ts = 0, 5, 10, ...` in frame units and subtracts it from the cue's offset-from-start frame index. The output is therefore a frame-count difference, not seconds.

ii. 
```python
ts = (np.arange(T, dtype=np.float32) * frame_stride)
cue = sound_fr[tr] - s if tr < len(sound_fr) else -1
time_to_cue = (cue - ts).astype(np.float32)
```

iii. No explicit justification beyond reuse of the stride-5 frame axis was documented.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by construction to the same downsampled inferred-trial bins used for `trial_spk`; both are length `T` and indexed from the same inferred start frame.

ii. 
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
T = trial_spk.shape[1]
ts = (np.arange(T, dtype=np.float32) * frame_stride)
...
inp = np.vstack([
    time_to_cue,
    np.full(T, float(day_value), dtype=np.float32),
    ts.astype(np.float32),
    np.full(T, float(reward_avail[tr]), dtype=np.float32),
])
```

iii. The agent's general justification is that all decoder variables were built on the same per-trial slices after downsampling.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not derived from any explicit raw behavioral field. The value comes from `enumerate(matched)`, i.e. the order in which filtered sessions are processed after matching.

ii. 
```python
for day_value, (sid, bkey) in enumerate(matched):
    subj, date, run = parse_session_id(sid)
    ...
    neural_trials, input_trials, output_trials, bounds_info, stim_names_local = build_trial_matrices(
        spk, sess, day_value, all_stim_names, frame_stride=5)
```

iii. The notes say training day needed to be inferred from session ordering, but the code never implements per-subject chronological counting.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI broadcasts the single integer `day_value` for the current session across every time bin of every trial in that session. Because `day_value` is just the global session enumeration index, it increases across all kept sessions rather than within subject.

ii. 
```python
inp = np.vstack([
    time_to_cue,
    np.full(T, float(day_value), dtype=np.float32),
    ts.astype(np.float32),
    np.full(T, float(reward_avail[tr]), dtype=np.float32),
])
```

iii. No direct justification was provided for the global enumeration choice.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from inferred trial segmentation only. The code does not use `StartFr` or frame timestamps. It defines time since start as the synthetic local time axis of the sliced trial.

ii. 
```python
s = int(max(0, starts[tr]))
e = int(min(n_frames - 1, max(starts[tr], ends[tr])))
...
ts = (np.arange(T, dtype=np.float32) * frame_stride)
```

iii. The notes frame trial start as the decoder alignment event, but the implementation uses only the inferred start index.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI uses `0, 5, 10, ...` for the kept bins of each downsampled trial and stores that directly as time since trial start. These are frame-step units rather than seconds.

ii. 
```python
ts = (np.arange(T, dtype=np.float32) * frame_stride)
...
np.full(T, float(day_value), dtype=np.float32),
ts.astype(np.float32),
np.full(T, float(reward_avail[tr]), dtype=np.float32),
```

iii. This is another consequence of the stride-5 simplification documented in the notes.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned one-to-one with the same `trial_spk` bins because `ts` is created from the trial's downsampled length `T`.

ii. 
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
T = trial_spk.shape[1]
ts = (np.arange(T, dtype=np.float32) * frame_stride)
```

iii. The agent consistently builds decoder inputs from the same `T` used for the neural slice.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` array when present, otherwise from an all-zero fallback array.

ii. 
```python
if 'isRew' in sess:
    reward_avail = np.asarray(sess['isRew']).astype(int)
    if reward_avail.shape[0] != ntrials:
        reward_avail = np.resize(reward_avail, ntrials)
else:
    reward_avail = np.zeros(ntrials, dtype=int)
```

iii. The notes identify `isRew` as the rewarded-corridor indicator and also explain that unsupervised sessions were removed because they made reward availability constant zero.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Beyond casting to integer, the only processing is resizing to length `ntrials` if the shape does not match and then broadcasting each trial's value across all bins in that trial.

ii. 
```python
if reward_avail.shape[0] != ntrials:
    reward_avail = np.resize(reward_avail, ntrials)
...
np.full(T, float(reward_avail[tr]), dtype=np.float32),
```

iii. No stronger justification than robustness to shape mismatch was documented.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI prefers `TrialStim` if present; otherwise it falls back to `WallName`.

ii. 
```python
def get_stimulus_categories(sess):
    if 'TrialStim' in sess:
        vals = np.asarray(sess['TrialStim']).astype(str)
        ...
        return vals, cats, np.array([mapping[v] for v in vals], dtype=np.int64)
    if 'WallName' in sess:
        vals = np.asarray(sess['WallName']).astype(str)
        ...
        return vals, cats, np.array([mapping[v] for v in vals], dtype=np.int64)
```

iii. The notes recognize both `TrialStim` and `WallName` as candidate stimulus fields, but do not justify preferring `TrialStim`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI takes the unique raw stimulus strings found across the retained sessions, sorts them lexicographically, maps each raw string to an integer category, and broadcasts the per-trial category across all bins of that trial. It does not collapse variants to the four canonical texture classes.

ii. 
```python
vals = np.asarray(sess['TrialStim']).astype(str)
cats = sorted(np.unique(vals).tolist())
mapping = {c: i for i, c in enumerate(cats)}
return vals, cats, np.array([mapping[v] for v in vals], dtype=np.int64)
```
```python
all_stim_names = set()
for _, b in matched:
    sess = beh_entries[b]
    vals, cats, ids = get_stimulus_categories(sess)
    all_stim_names.update(cats)
all_stim_names = sorted(all_stim_names)
...
np.full(T, stim_ids_global[tr], dtype=np.int64),
```

iii. No explicit justification beyond using the strings present in the retained data was documented.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and `LickTrind`.

ii. 
```python
lick_fr = np.asarray(sess.get('LickFr', []), dtype=int)
lick_tr = np.asarray(sess.get('LickTrind', []), dtype=int)
```

iii. The notes identify lick variables as available raw behavior and suitable for time-varying binary outputs.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI finds lick events whose `LickTrind` equals the current trial, subtracts the inferred trial start `s`, integer-divides by `frame_stride`, clips to `[0, T)`, and marks those bins as 1 in a binary vector.

ii. 
```python
lick = np.zeros(T, dtype=np.int64)
if lick_tr.size and lick_fr.size:
    idx = np.where(lick_tr == tr)[0]
    idx = idx[idx < lick_fr.shape[0]]
    if idx.size:
        lf = ((lick_fr[idx] - s) // frame_stride).astype(int)
        lf = lf[(lf >= 0) & (lf < T)]
        if lf.size:
            lick[lf] = 1
```

iii. The notes do not justify the per-trial reindexing beyond the general stride-5/downsampled representation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector has exactly the same downsampled per-trial length `T` as `trial_spk`, with indices computed relative to the same inferred start frame `s`.

ii. 
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
T = trial_spk.shape[1]
...
out = np.vstack([
    np.full(T, stim_ids_global[tr], dtype=np.int64),
    lick,
    pos_bin[s:e+1:frame_stride].astype(np.int64, copy=False),
    speed_bins[s:e+1:frame_stride].astype(np.int64, copy=False),
])
```

iii. The agent's recurring justification is same-slice construction for all modalities.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from the first available one-dimensional session array among `Pos`, `pos`, `AccumPos`, `AccPos`, or `LickPos` that matches `n_frames`. If none exists, the AI synthesizes a linear ramp from 0 to 4 across the session.

ii. 
```python
pos = None
for k in ['Pos', 'pos', 'AccumPos', 'AccPos', 'LickPos']:
    if k in sess and hasattr(sess[k], 'shape') and np.asarray(sess[k]).ndim == 1 and len(sess[k]) == n_frames:
        pos = np.asarray(sess[k], dtype=float)
        break
if pos is None:
    pos = np.linspace(0, 4, n_frames)
```

iii. The notes mention framewise position or accumulated position as possible sources and explicitly acknowledge fallback reconstruction where needed.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The selected position array is converted to four bins by clipping values to below 4 and taking `floor(pos / 1.0)`.

ii. 
```python
pos_bin = np.clip(np.floor(np.minimum(pos, 3.999999) / 1.0).astype(int), 0, 3)
```

iii. The notes justify position decoding as four equal 1 m bins, but do not justify the fallback source-variable choice.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are implicit at `0, 1, 2, 3, 4` in the units of the chosen `pos` array: `[0,1) -> 0`, `[1,2) -> 1`, `[2,3) -> 2`, `[3,4] -> 3` after clipping.

ii. 
```python
pos_bin = np.clip(np.floor(np.minimum(pos, 3.999999) / 1.0).astype(int), 0, 3)
```

iii. The justification in notes is simply that the decoder task asked for four equal 1 m spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by slicing the same inferred trial interval and same stride-5 sampling used for the neural data.

ii. 
```python
out = np.vstack([
    np.full(T, stim_ids_global[tr], dtype=np.int64),
    lick,
    pos_bin[s:e+1:frame_stride].astype(np.int64, copy=False),
    speed_bins[s:e+1:frame_stride].astype(np.int64, copy=False),
])
```

iii. The agent's general alignment rule is shared per-trial slicing across all streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `Run` if present, otherwise `RunFr`, otherwise a zero vector of length `n_frames`.

ii. 
```python
run = np.asarray(sess.get('Run', sess.get('RunFr', np.zeros(n_frames))), dtype=float).reshape(-1)
if run.shape[0] != n_frames:
    run = np.resize(run, n_frames)
```

iii. The notes mention `Run` / `RunFr` as candidate running variables but do not justify preferring them over framewise `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes per-session quartile thresholds with `np.nanquantile` over the chosen running array and assigns every frame to a bin using `np.digitize`.

ii. 
```python
def quartile_bins(x):
    x = np.asarray(x, dtype=float)
    qs = np.nanquantile(x[np.isfinite(x)], [0.25, 0.5, 0.75])
    return np.digitize(x, qs, right=False)
```
```python
speed_bins = quartile_bins(run)
```

iii. The notes say running speed should be discretized into quartiles for the decoder, but they do not discuss ties or reference-style rank binning.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It uses the 25th, 50th, and 75th percentiles of the chosen running array as thresholds; `np.digitize` then returns 0, 1, 2, or 3.

ii. 
```python
qs = np.nanquantile(x[np.isfinite(x)], [0.25, 0.5, 0.75])
return np.digitize(x, qs, right=False)
```

iii. The justification given is only that the task asked for four speed bins corresponding to 25% of the data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by taking the same inferred trial slice and stride-5 sampling as the neural data.

ii. 
```python
out = np.vstack([
    np.full(T, stim_ids_global[tr], dtype=np.int64),
    lick,
    pos_bin[s:e+1:frame_stride].astype(np.int64, copy=False),
    speed_bins[s:e+1:frame_stride].astype(np.int64, copy=False),
])
```

iii. The notes justify alignment through shared trial slicing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or mismatched data by falling back and resizing rather than by strict validation: behavior is matched by prefix when exact ids fail; missing retinotopy yields all-zero regions; mismatched retinotopy lengths are resized; missing `isRew` gives zeros; missing position gives a synthetic linear ramp; mismatched running arrays are resized.

ii. 
```python
if candidates:
    for c in candidates:
        if 'swap' not in c:
            return c
    return candidates[0]
```
```python
if run.shape[0] != n_frames:
    run = np.resize(run, n_frames)
...
if pos is None:
    pos = np.linspace(0, 4, n_frames)
...
if len(bri) != spk.shape[0]:
    bri = np.resize(bri, spk.shape[0])
```

iii. The notes frame these as pragmatic robustness measures to keep conversion running despite imperfect schema matching, but they do not justify them against the reference pipeline.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading all behavior dictionaries, loading and concatenating large spike files, and then constructing trial matrices in Python loops. The code structure makes spike I/O the dominant cost and repeats some session-wide preprocessing per session.

ii. 
```python
for f in sorted(BEH_ROOT.glob('*.npy')):
    dat = np.load(f, allow_pickle=True).item()
```
```python
obj = np.load(SPK_ROOT / f'{spike_sid}_neural_data.npy', allow_pickle=True).item()
spks = obj['spks']
return np.concatenate([x.astype(np.float32, copy=False) for x in spks], axis=0)
```
```python
for tr in range(ntrials):
    ...
    neural_trials.append(trial_spk)
    input_trials.append(inp)
    output_trials.append(out)
```

iii. The notes repeatedly discuss very large spike files, huge output sizes, and the need for stride-5 downsampling to make full conversion feasible.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_trial_matrices`, especially trial-by-trial lick indexing with `np.where`, could have been vectorized or pre-grouped. Trial-bound inference also computes starts/ends procedurally instead of using existing frame labels.

ii. 
```python
for tr in range(ntrials):
    ...
    idx = np.where(lick_tr == tr)[0]
    ...
    neural_trials.append(trial_spk)
    input_trials.append(inp)
    output_trials.append(out)
```

iii. No explicit optimization discussion beyond temporal downsampling was documented.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly parses stimulus categories, repeatedly scans lick indices by trial inside the main trial loop, and repeatedly loads retinotopy once per session even though the mapping is only used to create one session-level region vector. It also computes and returns values that are not later used.

ii. 
```python
for _, b in matched:
    sess = beh_entries[b]
    vals, cats, ids = get_stimulus_categories(sess)
    all_stim_names.update(cats)
```
```python
for tr in range(ntrials):
    ...
    idx = np.where(lick_tr == tr)[0]
```
```python
ret = load_retino_for_session(sid)
```

iii. No explicit justification for this repetition was documented.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `source_file` in `load_all_behavior_entries` but never uses it; returns `stim_names_local` from `build_trial_matrices` but never uses it; constructs `area_names` in `area_names_and_idx` but only uses the indices; and records `candidates` in `bounds_info` only for logging.

ii. 
```python
source_file = {}
...
source_file[sid] = f.name
return entries, source_file
```
```python
return neural_trials, input_trials, output_trials, bounds_info, stim_names_local
```
```python
def area_names_and_idx(iarea):
    area_names = ['All', 'V1', 'medial', 'anterior', 'lateral']
    ...
    return area_names, idx
```

iii. No explicit justification was documented; these look like leftovers from exploratory development.
