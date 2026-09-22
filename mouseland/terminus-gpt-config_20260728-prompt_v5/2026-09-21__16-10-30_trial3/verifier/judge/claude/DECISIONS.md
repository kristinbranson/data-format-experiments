# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by enumerating all `.npy` files in the `beh/` directory (not using the `Imaging_Exp_info.npy` master index), and separately enumerating all spike session files in `spk/`. It matches spike sessions to behavior entries by session ID string matching. Retinotopy is loaded from `retinotopy/` by normalized session ID (subject+date).

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

spike_sessions = sorted(f.name.replace('_neural_data.npy', '') for f in SPK_ROOT.glob('*_neural_data.npy'))
```

iii. The AI decided to use the spike files as the session backbone (89 spike files) and match behavior entries by session ID. The CONVERSION_NOTES.md states: "Use spike files as imaging-session backbone; match behavior by session id and retain only sessions with both neural and behavior data."

## 1-b. How are the data split into subjects?

i. The subject is extracted from the first component of the session ID string (before the first underscore). Subjects are accumulated as sessions are processed.

ii.
```python
def parse_session_id(session_id):
    parts = session_id.split('_')
    subj = parts[0]
    ...
    return subj, date, run

if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
```

iii. The AI noted "Subject count consistent; use subject prefix before first underscore as subject id." However, only 5 subjects appear in the final output because the AI filters to only rewarded sessions.

## 1-c. How are the data split into sessions?

i. Sessions are defined by spike files in the `spk/` directory. Each spike file represents one session. However, the AI filters to only sessions with rewarded trials (`isRew.any()`) and lick events (`LickFr` non-empty), resulting in only 27 of the 89 available sessions across 5 of 19 subjects.

ii.
```python
rewarded = []
for sid, b in matched:
    sess = beh_entries[b]
    isrew = np.asarray(sess.get('isRew', []))
    lickn = len(np.asarray(sess.get('LickFr', [])))
    if isrew.size and bool(isrew.any()) and lickn > 0:
        rewarded.append((sid, b))
if args.sample:
    matched = rewarded[:2] if len(rewarded) >= 2 else matched[:2]
else:
    matched = rewarded
```

iii. The AI's CONVERSION_NOTES.md (Step 10) explains: "Included many unsupervised/naive/grating sessions in the first full conversion, causing 62/89 sessions to have zero lick fraction and 61/89 to have zero reward availability. Resolution: restricted full conversion to sessions with isRew.any() and nonzero LickFr."

## 1-d. How are the data split into trials?

i. The AI uses a heuristic method to infer trial boundaries from `SoundFr` (sound cue frame numbers), not from the `ft_trInd` and `ft_CorrSpc` variables that directly encode trial membership and corridor space. It falls back to uniform splitting if SoundFr is not available.

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
    ...
```

iii. The AI's CONVERSION_NOTES.md acknowledges this as a limitation: "trial boundaries are still inferred heuristically from SoundFr" and notes this likely causes poor position decoding ("likely limited by heuristic trial-boundary inference").

## 1-e. How are trials filtered based on quality controls?

i. The AI does not filter trials based on quality. All trials from matched sessions are kept. The main filtering is at the session level (only rewarded sessions with lick events).

ii.
```python
for tr in range(ntrials):
    s = int(max(0, starts[tr]))
    e = int(min(n_frames - 1, max(starts[tr], ends[tr])))
    trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
    ...
```

iii. No trial-level quality filtering is documented or implemented. The AI's notes do not mention removing outlier-length trials or empty trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike session files (`spk/<session_id>_neural_data.npy`), concatenated across imaging planes.

ii.
```python
def load_spike_session(spike_sid):
    obj = np.load(SPK_ROOT / f'{spike_sid}_neural_data.npy', allow_pickle=True).item()
    spks = obj['spks']
    return np.concatenate([x.astype(np.float32, copy=False) for x in spks], axis=0)
```

iii. The AI correctly identified that `spks` contains already-processed deconvolved fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The neural data is cast to float32 during loading, then temporally downsampled by a stride of 5 (keeping every 5th frame), and stored as float16 per trial.

ii.
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
```
where `frame_stride=5`.

iii. The AI's CONVERSION_NOTES.md explains the stride-5 downsampling was added to reduce dataset size: "Temporal downsampling by stride 5 and float16 neural storage reduced sample size from ~12G to ~1.4G."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons by brain region. All neurons are kept, including those outside the four visual areas (V1, medial/mHV, anterior/aHV, lateral/lHV). Neurons with no assigned area are placed in an "All" category. The area mapping also differs from the reference: iarea=7 is mapped to V1 (the reference only uses iarea=8 for V1), and iarea=0 and iarea=9 are not mapped to any named region.

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

iii. No explicit justification for keeping all neurons or for the different area mapping was provided.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI uses heuristic trial boundaries derived from SoundFr to define trial start/end frames, rather than using the `ft_trInd` and `ft_CorrSpc` variables that directly mark corridor entry. The first trial starts at frame 0 and subsequent trial starts are inferred from differences in SoundFr values.

ii.
```python
starts = np.zeros(ntrials, dtype=int)
starts[1:] = np.maximum(sound_fr[1:] - np.maximum(np.diff(sound_fr), 1), 0)
ends = np.empty(ntrials, dtype=int)
ends[:-1] = np.maximum(starts[1:] - 1, starts[:-1])
ends[-1] = n_frames - 1
```

iii. The AI's notes acknowledge this as an approximation: "Trial boundaries currently inferred heuristically and should be refined from behavior variables if available."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies stride-5 temporal downsampling. The metadata reports `time_bin_size: 5.0`, but this value appears to be in units of frames (not milliseconds). At the native ~3.17 Hz imaging rate (~315 ms/frame), stride-5 gives ~1575 ms bins.

ii.
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
...
'time_bin_size': 5.0,
```

iii. The AI described this as reducing file size. However, the `time_bin_size` in metadata is 5.0 with unclear units, whereas the reference uses `1000.0 / 3.17 = 315.46 ms`.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr`, the frame number of the sound cue for each trial.

ii.
```python
sound_fr = np.asarray(sess.get('SoundFr', np.full(ntrials, -1)), dtype=int)
cue = sound_fr[tr] - s if tr < len(sound_fr) else -1
time_to_cue = (cue - ts).astype(np.float32)
```

iii. The AI uses SoundFr directly.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to cue is computed as the difference between the cue frame (relative to trial start) and the frame index within the trial, in units of frames (multiplied by stride). This is NOT in seconds as the reference computes it.

ii.
```python
ts = (np.arange(T, dtype=np.float32) * frame_stride)
cue = sound_fr[tr] - s if tr < len(sound_fr) else -1
time_to_cue = (cue - ts).astype(np.float32)
```

iii. No explicit justification for using frame units instead of seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to cue is computed from the same trial frame window used for the neural data, so it is aligned.

ii.
```python
ts = (np.arange(T, dtype=np.float32) * frame_stride)
time_to_cue = (cue - ts).astype(np.float32)
```

iii. Aligned by construction since both use the same trial window frames.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The day of training is the enumeration index of the session in the matched list (0, 1, 2, ..., 26), NOT a per-mouse count of training days.

ii.
```python
for day_value, (sid, bkey) in enumerate(matched):
    ...
    np.full(T, float(day_value), dtype=np.float32),
```

iii. No explicit justification for using global enumeration index rather than per-mouse training day count.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day is simply the loop index when iterating over matched sessions. It ranges from 0 to 26 (for 27 sessions). Since sessions from different mice are interleaved, this does NOT represent the actual training day for each mouse.

ii.
```python
for day_value, (sid, bkey) in enumerate(matched):
```

iii. The verification output shows day_of_training ranges [0, 26], with each session having a unique value, confirming it's a global session index rather than per-mouse training day.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the frame index within each trial, scaled by the frame stride.

ii.
```python
ts = (np.arange(T, dtype=np.float32) * frame_stride)
```

iii. No raw data variable is directly used; the time is synthesized from the frame count.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as frame index * frame_stride, giving values in units of frames (not seconds). For the reference solution, actual timestamps from `ft` are used and converted to seconds.

ii.
```python
ts = (np.arange(T, dtype=np.float32) * frame_stride)
...
inp = np.vstack([
    ...
    ts.astype(np.float32),
    ...
])
```

iii. No justification for using frame units instead of seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned by construction since both use the same trial frame indices.

ii.
```python
ts = (np.arange(T, dtype=np.float32) * frame_stride)
```

iii. Same window as neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` in the behavior data.

ii.
```python
reward_avail = None
if 'isRew' in sess:
    reward_avail = np.asarray(sess['isRew']).astype(int)
```

iii. The AI correctly identified `isRew` as the source for reward availability.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to int and broadcast per trial. If `isRew` is missing, defaults to zeros.

ii.
```python
reward_avail = np.asarray(sess['isRew']).astype(int)
if reward_avail.shape[0] != ntrials:
    reward_avail = np.resize(reward_avail, ntrials)
...
np.full(T, float(reward_avail[tr]), dtype=np.float32),
```

iii. The AI adds a fallback for missing isRew and a resize for mismatched lengths.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI uses `TrialStim` first if available, and falls back to `WallName`. It does NOT group stimuli into base texture categories.

ii.
```python
def get_stimulus_categories(sess):
    if 'TrialStim' in sess:
        vals = np.asarray(sess['TrialStim']).astype(str)
        cats = sorted(np.unique(vals).tolist())
        mapping = {c: i for i, c in enumerate(cats)}
        return vals, cats, np.array([mapping[v] for v in vals], dtype=np.int64)
    if 'WallName' in sess:
        vals = np.asarray(sess['WallName']).astype(str)
        cats = sorted(np.unique(vals).tolist())
        ...
```

iii. The AI's CONVERSION_NOTES.md notes that stimulus categories come from corridor/wall metadata. However, the reference explicitly notes that `TrialStim` is masked in swap sessions and uses `WallName` with a mapping to 4 base categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI treats each unique stimulus name as its own category (e.g., circle1, circle2, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3 = 7 categories). The reference groups these into 4 base textures (circle, leaf, rock, wood) using a hard-coded mapping.

ii.
```python
all_stim_names = set()
for _, b in matched:
    sess = beh_entries[b]
    vals, cats, ids = get_stimulus_categories(sess)
    all_stim_names.update(cats)
all_stim_names = sorted(all_stim_names)
```

iii. The verification output confirms 7 stimulus categories: circle1 (0.376), circle2 (0.031), leaf1 (0.373), leaf1_swap1 (0.015), leaf1_swap2 (0.017), leaf2 (0.155), leaf3 (0.035). Rock and wood categories are missing entirely, consistent with only rewarded sessions being included.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (lick frame numbers) and `LickTrind` (lick trial indices).

ii.
```python
lick_fr = np.asarray(sess.get('LickFr', []), dtype=int)
lick_tr = np.asarray(sess.get('LickTrind', []), dtype=int)
```

iii. The AI uses both LickFr and LickTrind to assign licks to trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI uses `LickTrind` to identify which licks belong to each trial, then maps the lick frame to a bin within the trial. This is different from the reference, which creates a session-wide binary lick array from `LickFr` frame indices and then indexes into it.

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

iii. The AI handles edge cases where lick arrays may be inconsistent in length.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are converted to trial-relative indices and adjusted for frame stride, aligning with the strided neural data.

ii.
```python
lf = ((lick_fr[idx] - s) // frame_stride).astype(int)
```

iii. Aligned by converting lick frame numbers to the strided time base.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI searches for position data in multiple possible variable names: `Pos`, `pos`, `AccumPos`, `AccPos`, `LickPos`. It does NOT use `ft_Pos` which is the frame-by-frame position variable used by the reference.

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

iii. No explicit justification for not using `ft_Pos`. The AI falls back to a linear interpolation if no position variable is found.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 1.0 and floored into 4 bins, assuming the position is already in meters. The reference divides `ft_Pos` by 10 (converting from decimeters to meters).

ii.
```python
pos_bin = np.clip(np.floor(np.minimum(pos, 3.999999) / 1.0).astype(int), 0, 3)
```

iii. The AI assumes position is in meters. Whether this produces correct bins depends on which position variable was actually found and its units.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 1.0 m into 4 bins: [0,1), [1,2), [2,3), [3,4].

ii.
```python
pos_bin = np.clip(np.floor(np.minimum(pos, 3.999999) / 1.0).astype(int), 0, 3)
```

iii. The binning logic is similar to the reference in structure but differs in the source variable and unit assumptions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same trial frame range and stride as the neural data.

ii.
```python
pos_bin[s:e+1:frame_stride].astype(np.int64, copy=False),
```

iii. Aligned by using same frame indices and stride.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The AI uses `Run` or `RunFr` (falling back to zeros), NOT `ft_RunSpeed` which is the per-frame running speed used by the reference.

ii.
```python
run = np.asarray(sess.get('Run', sess.get('RunFr', np.zeros(n_frames))), dtype=float).reshape(-1)
if run.shape[0] != n_frames:
    run = np.resize(run, n_frames)
```

iii. No explicit justification for not using `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI uses quantile-based binning with `np.digitize` at the 25th, 50th, and 75th percentiles. This differs from the reference's rank-based quartile method that guarantees equal bin sizes.

ii.
```python
def quartile_bins(x):
    x = np.asarray(x, dtype=float)
    qs = np.nanquantile(x[np.isfinite(x)], [0.25, 0.5, 0.75])
    return np.digitize(x, qs, right=False)
```

iii. The verification output shows highly uneven speed bins for some sessions (e.g., TX60_2021_04_10_1: 0.000, 0.000, 0.732, 0.268), confirming the quantile approach does not achieve equal bin sizes when many values are tied (e.g., many frames at zero speed).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with quantile thresholds [Q25, Q50, Q75], which creates 4 bins. However, when many values are tied at the same value (e.g., zero speed), multiple quantiles can be identical, causing highly unbalanced bins.

ii.
```python
qs = np.nanquantile(x[np.isfinite(x)], [0.25, 0.5, 0.75])
return np.digitize(x, qs, right=False)
```

iii. The reference explicitly addresses this: "a quarter or more of the frames can sit at exactly zero speed, and no threshold can cut a tie like that into equal parts, so the split is on rank rather than on value."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed bins are indexed from the session-wide array using the same trial frame range and stride as neural data.

ii.
```python
speed_bins[s:e+1:frame_stride].astype(np.int64, copy=False),
```

iii. Aligned by using same frame indices and stride.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: missing behavior keys default to zeros or empty arrays; arrays that don't match expected length are resized with `np.resize`; lick indices are clipped to valid ranges; if retinotopy is missing, zeros are used for brain region indices. The AI also handles crashes gracefully by skipping sessions that fail.

ii.
```python
if run.shape[0] != n_frames:
    run = np.resize(run, n_frames)
...
if reward_avail.shape[0] != ntrials:
    reward_avail = np.resize(reward_avail, ntrials)
...
if len(bri) != spk.shape[0]:
    bri = np.resize(bri, spk.shape[0])
```

iii. The AI uses `np.resize` which repeats values cyclically, which could silently introduce incorrect data rather than flagging mismatches.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files is the most time-consuming step. The full conversion of 27 sessions took ~182 seconds.

ii.
```python
spk = load_spike_session(sid)
# loads and concatenates spks arrays
```

iii. The AI's conversion_full_out.txt shows total time of 181.83s for 27 sessions.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_trial_matrices` iterates over all trials sequentially, constructing each trial's neural, input, and output arrays one at a time. The lick assignment per trial (searching `LickTrind` for matching indices) could be vectorized.

ii.
```python
for tr in range(ntrials):
    ...
    if lick_tr.size and lick_fr.size:
        idx = np.where(lick_tr == tr)[0]
        ...
```

iii. N/A

## 12-c. What processing does the code repeat multiple times?

i. The `get_stimulus_categories` function is called twice for each session: once during the initial pass to collect all stimulus names, and once during `build_trial_matrices`.

ii.
```python
# First pass:
for _, b in matched:
    sess = beh_entries[b]
    vals, cats, ids = get_stimulus_categories(sess)
    all_stim_names.update(cats)

# Second pass in build_trial_matrices:
stim_vals, stim_names_local, stim_ids = get_stimulus_categories(sess)
```

iii. N/A

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads all neurons (including those outside the four visual areas) as float32, only to store them all in the output. Since the reference filters to visual area neurons only, the "All" category neurons (those not in V1/medial/anterior/lateral) represent unnecessary data that inflates the dataset size. Additionally, the `infer_trial_frame_bounds` function computes trial boundaries heuristically when direct trial indicators (`ft_trInd`, `ft_CorrSpc`) are available in the data.

ii.
```python
# All neurons kept, including 'All' category
idx = np.zeros((len(iarea),), dtype=np.int64)  # default 'All'
```

iii. N/A
