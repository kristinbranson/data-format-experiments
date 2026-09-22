# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. It reads the master `Imaging_Exp_info.npy`, every `Beh_<exp_type>.npy`, each session's spike file, and its retinotopy file. Duplicate recordings are merged in a session dictionary while stimulus mappings are accumulated.

ii. ```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k), allow_pickle=True).item()['spks']
```

iii. The trajectory established that 89 unique recordings match 89 spike files and verified that duplicate behavior entries were identical.

## 1-b. How are the data split into subjects?

i. Subjects come from `db['mname']`; a sorted unique subject list is made and each session receives its index.

ii. ```python
subjects = sorted(set(sessions[k]['db']['mname'] for k in keys))
data['subject_idx'].append(subjects.index(db['mname']))
```

iii. The trajectory counted 19 unique mice and treated the index's mouse name as authoritative.

## 1-c. How are the data split into sessions?

i. A session key is mouse, date, and block. `setdefault` merges repeated appearances across experiment types, producing 89 unique sessions.

ii. ```python
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
s = sessions.setdefault(key, {'beh': beh, 'db': db, 'stim_map': {}, 'exp_types': []})
```

iii. The agent explicitly checked repeated entries and found their behavior identical.

## 1-d. How are the data split into trials?

i. Trials are grouped by `ft_trInd`, but only frames satisfying both corridor-space and running/VR-moving masks are retained; frame indices are sorted by trial.

ii. ```python
keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
idx = np.where(keep)[0]
tr = ft_tr[idx]
```

iii. The agent cited the methods statement that analyses considered running timepoints and the corresponding paper mask.

## 1-e. How are trials filtered based on quality controls?

i. It drops trials with fewer than two retained running frames and trials without one of seven mapped canonical stimulus names. It does not reject extreme-duration stopped trials directly.

ii. ```python
if len(idx) < 2: continue
name = smap.get(str(beh['WallName'][t]), None)
if name is None or name not in STIM_NAMES: continue
```

iii. The trajectory says the seven paper categories were kept and 309 unmapped `circle3` trials were deliberately dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from per-plane `spks` in each neural-data file; neuron area labels come from retinotopy `iarea`.

ii. ```python
spks = np.load(...).item()['spks']
ret = np.load(...); aidx = area_index(ret['iarea'])
```

iii. The agent inspected `utils.load_spk`, spike-file structure, and neuron-count alignment with retinotopy.

## 2-b. How is the `neural` data processed?

i. It selects at most 1,000 valid neurons, copies them plane by plane into float32, and extracts the selected frame columns without further signal transformation.

ii. ```python
if len(valid) > args.nneurons:
    sel = np.sort(rng.choice(valid, args.nneurons, replace=False))
spk = np.empty((len(sel), spks[0].shape[1]), dtype=np.float32)
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The agent considered subsampling necessary for tractability because raw spikes occupy about 405 GB; it retained native deconvolved traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It removes neurons outside V1/mHV/lHV/aHV and then randomly subsamples valid neurons to at most 1,000 per session with seed 0.

ii. ```python
valid = np.where(aidx >= 0)[0]
sel = np.sort(rng.choice(valid, args.nneurons, replace=False))
```

iii. The visual-area filter follows paper utilities; the extra cap was justified by data size.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is anchored to corridor entry conceptually, has variable length, and uses corridor frames from its trial index, but stationary frames inside the traversal are removed.

ii. ```python
tidx = trial_frames(beh, nfr)
...
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The agent viewed corridor entry as trial start and applied the paper's running mask.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. One neural column is one imaging frame; metadata uses the median measured frame interval, about 315 ms.

ii. ```python
dts.append(np.median(np.diff(ft)) * SEC_PER_DAY)
 'time_bin_size': float(np.median(dts) * 1000.0)
```

iii. The trajectory measured approximately 3.17 Hz and chose native imaging resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Time to cue is derived directly from per-trial `SoundTime` and per-frame timestamps `ft`.

ii. ```python
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
```

iii. The agent inspected timing fields and used timestamps already expressed as MATLAB datenums.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. It subtracts every retained frame time from cue time and multiplies days by 86,400, yielding seconds until cue (positive before, negative after).

ii. ```python
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
inp[0] = tcue
```

iii. The agent explicitly interpreted “time to” as cue minus current time.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It evaluates cue offset at the identical `idx` columns used for neural data.

ii. ```python
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The agent relied on the common imaging-frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Day of training is derived from `db['datexp']` and the earliest selected session date for each mouse.

ii. ```python
d = datetime.date(*map(int, db['datexp'].split('_')))
first_date[db['mname']] = d
```

iii. The agent interpreted day as elapsed calendar days from each mouse's first imaging date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It computes calendar-day difference from the first date and broadcasts that scalar over all bins in a trial.

ii. ```python
day = float((datetime.date(*map(int, db['datexp'].split('_'))) - first_date[db['mname']]).days)
inp[1] = day
```

iii. No explicit trajectory comparison with session ordinal was recorded.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Time since trial start comes from frame timestamps `ft` and per-trial `Trial_start_time`.

ii. ```python
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
```

iii. The agent treated `Trial_start_time` as the corridor-entry timestamp.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It subtracts trial-start time from retained frame times and converts days to seconds.

ii. ```python
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
inp[2] = tt
```

iii. The trajectory checked resulting ranges and retained wall-clock elapsed time even across stopped periods.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The elapsed-time series uses the same masked frame indices as neural data.

ii. ```python
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Alignment was based on shared imaging indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability comes from per-trial `beh['isRew']`.

ii. ```python
inp[3] = float(beh['isRew'][t])
```

iii. The trajectory verified that `isRew` marks rewarded corridors rather than reward delivery.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The binary trial flag is cast to float and broadcast across every retained bin.

ii. ```python
inp[3] = float(beh['isRew'][t])
```

iii. The agent confirmed it is zero for unrewarded contexts and indicates availability throughout the corridor.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Stimulus is derived by accumulating a mapping from physical `WallName` to canonical per-trial `TrialStim` across behavior files.

ii. ```python
for w, c in zip(beh['WallName'], beh['TrialStim']):
    s['stim_map'][str(w)] = str(c)
name = smap.get(str(beh['WallName'][t]), None)
```

iii. The agent found physical textures differ across mice and chose canonical functional identities for shared decoding.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Seven canonical identities are indexed (circle1/2, leaf1/2/3, and two swaps), broadcast over time; unmapped `circle3` trials are skipped.

ii. ```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']
out[0] = STIM_NAMES.index(name)
```

iii. The trajectory deliberately kept the paper's seven categories and excluded 309 `circle3` trials.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`, the session's lick frame positions.

ii. ```python
lickfr = np.round(beh['LickFr']).astype(int) if len(beh['LickFr']) else np.zeros(0, int)
```

iii. The agent identified `LickFr` as directly available lick timing.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are rounded to the nearest integer; in-range frames are marked 1 in an otherwise-zero binary vector.

ii. ```python
lickfr = np.round(beh['LickFr']).astype(int)
is_lick = np.zeros(nfr, dtype=np.int64)
is_lick[lickfr] = 1
```

iii. The agent intended one binary indicator per imaging frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary lick vector is sliced with the same trial indices used by neural data.

ii. ```python
out[1] = is_lick[idx]
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The shared frame grid was the alignment rationale.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from framewise `ft_Pos`.

ii. ```python
pos = beh['ft_Pos'][:nfr]
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. The agent determined positions are in decimeters over a 4 m textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is converted from decimeters to integer meter bins and clipped into labels 0–3.

ii. ```python
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. The requested four equal 1 m bins motivated this conversion.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 dm (0–1, 1–2, 2–3, 3–4 m); division/truncation determines labels and clipping handles endpoints.

ii. ```python
['0-1m', '1-2m', '2-3m', '3-4m']
np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. The corridor geometry and task specification provide these fixed edges.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position values are indexed by the same `idx` as the neural columns.

ii. ```python
out[2] = ... pos[idx] ...
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Alignment uses framewise behavior sampled on imaging frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from framewise `ft_RunSpeed` on frames satisfying the running-and-corridor mask.

ii. ```python
speeds.append(beh['ft_RunSpeed'][:nfr][keep])
speed = beh['ft_RunSpeed'][:nfr]
```

iii. The trajectory identified this as the direct speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. It concatenates selected-frame speeds across all chosen sessions, calculates three global percentile edges, then digitizes trial speeds with those edges.

ii. ```python
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. The agent sought quartile bins across all analyzed frames.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Categories use global 25th, 50th, and 75th percentile speed values with `searchsorted`, yielding labels 0–3.

ii. ```python
speed_edges = np.percentile(speeds, [25, 50, 75])
np.searchsorted(speed_edges, speed[idx])
```

iii. The four requested 25% bins motivated percentile cut points.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Each trial's speed labels use the same frame indices as its neural matrix.

ii. ```python
out[3] = np.searchsorted(speed_edges, speed[idx])
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The agent aligned all behavioral streams on frame number.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. It truncates behavior to the smaller of neural and behavior frame counts; removes NaN trial indices; bounds-checks rounded lick frames; skips trials with fewer than two usable frames or missing stimulus mappings; and asserts neuron/area count equality.

ii. ```python
nfr = min(spk.shape[1], len(beh['ft']))
ok = ~np.isnan(tr)
lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
assert nneu_all == len(aidx)
```

iii. The agent checked key frame-marker NaNs and duplicate consistency, and format validation reported no warnings.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the roughly 405 GB of spike files and copying selected per-plane neuron data dominate runtime; full conversion took several minutes.

ii. ```python
spks = np.load(...).item()['spks']
spk[o2:o2 + len(take)] = spks[p][take]
```

iii. The trajectory measured about eight seconds for one load and explicitly identified the raw neural volume as the bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over sessions, imaging planes, and trials. Trial grouping itself is mostly vectorized by sorting/searching, while per-trial array construction could be partially batched but variable lengths limit this.

ii. ```python
for si, k in enumerate(keys):
for p in range(len(spks)):
for t, idx in enumerate(tidx):
```

iii. The trajectory favored efficient grouped trial indexing and accepted loops needed for variable-length output.

## 12-c. What processing does the code repeat multiple times?

i. Behavior is scanned once while loading/merging sessions and again in a first pass for speed quartiles and earliest dates; retained behavior fields are revisited in the session pass. Duplicate behavior files are all loaded to accumulate mappings.

ii. ```python
sessions = load_sessions()
for k in keys: ... speeds.append(...)
for si, k in enumerate(keys): ...
```

iii. The trajectory intentionally compared and merged duplicate experiment-type entries and used a behavior-only prepass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads every duplicate behavior file and stores metadata not used by decoder training; it also computes/prints summary counts. More materially, it first loads complete spike planes although only up to 1,000 neurons are copied, a consequence of the `.npy` object format.

ii. ```python
s['exp_types'].append(exp_type)
session_info.append({...})
ntr = sum(len(s) for s in data['neural'])
```

iii. The agent viewed duplicate-file loading as necessary for stimulus mappings and full spike loading as unavoidable with the storage format.
