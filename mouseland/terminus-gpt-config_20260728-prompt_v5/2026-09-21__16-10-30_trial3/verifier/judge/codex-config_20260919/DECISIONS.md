# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every behavior `.npy` into one dictionary, enumerates spike files as the session backbone, matches exact or suffixed behavior keys, then deliberately retains only sessions having both a rewarded trial and at least one lick. Spike planes are loaded per retained session and concatenated. Thus it does not convert all imaging data.

ii.
```python
for f in sorted(BEH_ROOT.glob('*.npy')):
    dat = np.load(f, allow_pickle=True).item()
    for sid, sess in dat.items(): entries[sid] = sess
spike_sessions = sorted(f.name.replace('_neural_data.npy', '') for f in SPK_ROOT.glob('*_neural_data.npy'))
if isrew.size and bool(isrew.any()) and lickn > 0: rewarded.append((sid, b))
matched = rewarded
```

iii. Notes say spike files are the 89-session imaging backbone, but the final review restricted the dataset to rewarded sessions with lick events to avoid degenerate decoder targets; the final dataset has 27 sessions and 5 subjects.

## 1-b. How are the data split into subjects?

i. Subject is the prefix before the first underscore in each retained spike-session id. Subjects are added in encounter order and each session receives its index.

ii.
```python
subj = session_id.split('_')[0]
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects); subjects.append(subj)
data['subject_idx'].append(subject_to_idx[subj])
```

iii. The notes justify the session-id prefix as the mouse id, but session filtering reduces the result from the 19 imaging mice to 5.

## 1-c. How are the data split into sessions?

i. Each `*_neural_data.npy` defines a session. It is matched to an exact behavior key when possible, otherwise a key beginning with the spike id (preferring a non-swap key), and only rewarded/licking sessions survive.

ii.
```python
if spike_sid in beh_entries: return spike_sid
candidates = [k for k in beh_entries if k.startswith(spike_sid + '_')]
for c in candidates:
    if 'swap' not in c: return c
```

iii. The notes say behavior includes extra entries and should be matched to imaging sessions. They acknowledge some final matches use swap variants.

## 1-d. How are the data split into trials?

i. The agent ignores the explicit framewise trial index and corridor mask. It heuristically derives starts from consecutive `SoundFr` values and ends immediately before the next inferred start; if unavailable, it partitions the session uniformly. Every declared trial is emitted.

ii.
```python
starts[1:] = np.maximum(sound_fr[1:] - np.maximum(np.diff(sound_fr), 1), 0)
ends[:-1] = np.maximum(starts[1:] - 1, starts[:-1])
ends[-1] = n_frames - 1
# fallback
edges = np.linspace(0, n_frames, ntrials + 1).astype(int)
```

iii. The notes repeatedly call this `soundfr_heuristic` a remaining caveat and associate weak position decoding with it; metadata also says boundaries should be refined.

## 1-e. How are trials filtered based on quality controls?

i. Trials are not quality-filtered. All `range(ntrials)` trials are sliced after clamping inferred bounds, including extremely long/stationary trials.

ii.
```python
for tr in range(ntrials):
    s = int(max(0, starts[tr]))
    e = int(min(n_frames - 1, max(starts[tr], ends[tr])))
```

iii. Notes discuss running-only analysis and the need to exclude stationary periods but the implementation does not do so; filtering is performed at session rather than trial level.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `spks` list in each session's neural file, concatenated across planes/groups. `iarea` is loaded separately for region labels, not neural values.

ii.
```python
obj = np.load(SPK_ROOT / f'{spike_sid}_neural_data.npy', allow_pickle=True).item()
return np.concatenate([x.astype(np.float32, copy=False) for x in obj['spks']], axis=0)
```

iii. The agent correctly notes these are already processed, deconvolved fluorescence traces and should not undergo dF/F or deconvolution again.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, each inferred trial interval is sliced, every fifth frame is retained, and values are stored as float16.

ii.
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
```

iii. The notes justify stride-5 and float16 as size reductions (about 16 GB final output), while acknowledging coarse/heuristic alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. All concatenated `spks` rows remain. Retinotopy codes are mapped to five labels including `All`; unknown/out-of-scope areas default to `All`, and a length mismatch is repaired by `np.resize`.

ii.
```python
idx = np.zeros((len(iarea),), dtype=np.int64)
idx[(iarea == 7) | (iarea == 8)] = 1
if len(bri) != spk.shape[0]: bri = np.resize(bri, spk.shape[0])
```

iii. Notes state source ROIs were already Suite2p-curated and found no further cell threshold. They planned coarse region mapping, but did not apply the reference's visual-area inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Arrays begin at an inferred start, described as corridor entry, but the start is actually reconstructed from sound-frame differences (or uniform partitioning), not `StartFr`/`ft_trInd`/`ft_CorrSpc`. No padding is used.

ii.
```python
starts, ends, bounds_info = infer_trial_frame_bounds(sess, n_frames)
trial_spk = spk[:, s:e+1][:, ::frame_stride]
```

iii. Documentation claims trial-start/corridor-entry alignment but candidly labels it heuristic and needing refinement.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It subsamples every fifth imaging frame without aggregation. Metadata reports `5.0`, although the source rate is about 3.17 Hz, so the actual post-stride interval is about 1.58 s (1577 ms), not 5 ms.

ii.
```python
frame_stride=5
trial_spk = spk[:, s:e+1][:, ::frame_stride]
'time_bin_size': 5.0,
```

iii. Stride-5 was introduced solely to reduce output size. Notes describe it as temporal downsampling but do not correct the units or physical time base.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and the inferred trial start index.

ii.
```python
sound_fr = np.asarray(sess.get('SoundFr', np.full(ntrials, -1)), dtype=int)
cue = sound_fr[tr] - s
```

iii. The mapping plan proposed `SoundFr` or `SoundPos` with a framewise time base; the final code uses `SoundFr` only.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. It subtracts the downsampled frame offset from the cue's frame offset. Values remain in source-frame counts, not seconds.

ii.
```python
ts = np.arange(T, dtype=np.float32) * frame_stride
time_to_cue = (cue - ts).astype(np.float32)
```

iii. Notes intended a continuous time-until-cue value but did not document or implement conversion to seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It has one value for each stride-selected neural column, using offsets `0,5,10,...` from the same inferred start.

ii.
```python
T = trial_spk.shape[1]
ts = np.arange(T, dtype=np.float32) * frame_stride
```

iii. The agent intended to preserve framewise event relationships; internal array lengths align, though the underlying trial alignment is wrong.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not derived per mouse or from dates. It is the global zero-based position of each retained session in sorted matching order.

ii.
```python
for day_value, (sid, bkey) in enumerate(matched):
```

iii. The planning notes said training order should come from session date/file context, but no per-subject calculation was implemented.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The global session index is cast to float and broadcast over all time bins in the trial; it ranges 0–26 in the final dataset.

ii.
```python
np.full(T, float(day_value), dtype=np.float32)
```

iii. The notes call it a continuous per-trial scalar broadcast over time, but provide no justification for global rather than mouse-specific counting.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived only from the number/order of retained neural frames and the fixed stride, based on the heuristic trial start; raw timestamps and `StartFr` are unused.

ii.
```python
T = trial_spk.shape[1]
ts = np.arange(T, dtype=np.float32) * frame_stride
```

iii. The mapping plan called for a trial-relative frame/time index aligned to corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It produces `0, 5, 10, ...` in original-frame units, not elapsed seconds.

ii.
```python
ts = (np.arange(T, dtype=np.float32) * frame_stride)
```

iii. Notes describe a continuous increasing value but never resolve the source frame rate or physical units.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Each selected neural column receives the corresponding stride-multiple offset, so shapes and selected indices align internally.

ii.
```python
trial_spk = spk[:, s:e+1][:, ::frame_stride]
T = trial_spk.shape[1]
ts = np.arange(T) * frame_stride
```

iii. The agent's stated goal was common trial-start alignment, though the start itself is heuristic.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from per-trial `isRew`; missing data becomes zeros and a length mismatch is resized cyclically.

ii.
```python
reward_avail = np.asarray(sess['isRew']).astype(int)
if reward_avail.shape[0] != ntrials: reward_avail = np.resize(reward_avail, ntrials)
```

iii. Notes identify `isRew`/rewarded-corridor identity as the correct source.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The integer trial flag is converted to float and broadcast across every time bin.

ii.
```python
np.full(T, float(reward_avail[tr]), dtype=np.float32)
```

iii. The notes specify a binary per-trial value; no additional scientific processing was intended.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It prefers `TrialStim`; only if absent does it use `WallName`.

ii.
```python
if 'TrialStim' in sess: vals = np.asarray(sess['TrialStim']).astype(str)
elif 'WallName' in sess: vals = np.asarray(sess['WallName']).astype(str)
```

iii. Notes considered either field, despite recognizing behavior variants and swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct raw string across retained sessions is sorted and assigned its own global category, then broadcast over a trial. Crops and swap variants are not collapsed into the four base textures.

ii.
```python
all_stim_names.update(cats)
all_stim_names = sorted(all_stim_names)
stim_ids_global = np.array([stim_map_global[v] for v in stim_vals])
np.full(T, stim_ids_global[tr])
```

iii. The planning notes mention naturalistic categories/test variants; final validation reports seven categories rather than the reference four.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses both `LickFr` (event frame) and `LickTrind` (trial assignment).

ii.
```python
lick_fr = np.asarray(sess.get('LickFr', []), dtype=int)
lick_tr = np.asarray(sess.get('LickTrind', []), dtype=int)
```

iii. The notes planned reconstruction from lick frames/trial indices and later added clipping for inconsistent array lengths.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, matching lick events are converted to local bins by subtracting inferred start and integer-dividing by five; any bin containing a lick is 1.

ii.
```python
idx = np.where(lick_tr == tr)[0]
lf = ((lick_fr[idx] - s) // frame_stride).astype(int)
lick[lf] = 1
```

iii. Notes justify a binary time series and robustly clipping inconsistent lick arrays.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are assigned to the same stride-5 bins as neural columns, using the inferred trial start. Integer division pools any event in a five-frame interval into that retained bin.

ii.
```python
lf = (lick_fr[idx] - s) // frame_stride
lf = lf[(lf >= 0) & (lf < T)]
```

iii. The agent intended framewise alignment; it is internally aligned to its constructed windows but not reliably to true trials.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The code searches `Pos`, `pos`, `AccumPos`, `AccPos`, then `LickPos` for a vector exactly as long as neural frames. It does not use the actual reference field `ft_Pos`. If none matches, it fabricates a session-wide linear ramp from 0 to 4.

ii.
```python
for k in ['Pos', 'pos', 'AccumPos', 'AccPos', 'LickPos']:
    ...
if pos is None: pos = np.linspace(0, 4, n_frames)
```

iii. Notes admit fallback position reconstruction and say it reduces fidelity, yet the final review leaves it in place.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The selected/fabricated values are assumed to be meters; values are floored into unit-width bins after capping at 3.999999.

ii.
```python
pos_bin = np.clip(np.floor(np.minimum(pos, 3.999999) / 1.0).astype(int), 0, 3)
```

iii. The plan asked for four equal 1 m corridor bins but did not validate source units.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are nominally 0, 1, 2, 3, and 4 in assumed meters, yielding codes 0–3; negative values clip to 0 and values at/above 4 clip to 3.

ii.
```python
np.clip(np.floor(np.minimum(pos, 3.999999) / 1.0).astype(int), 0, 3)
```

iii. The agent cites the required four equal-length 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The session-wide position vector is sliced with the same inferred start/end and stride as neural data.

ii.
```python
pos_bin[s:e+1:frame_stride]
```

iii. It was intended as framewise alignment, but the fallback ramp and wrong trial bounds make the semantic alignment invalid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It prefers `Run`, falls back to `RunFr`, and otherwise zeros. A length mismatch is repaired with cyclic `np.resize`. It does not use the reference framewise speed field `ft_RunSpeed`.

ii.
```python
run = np.asarray(sess.get('Run', sess.get('RunFr', np.zeros(n_frames))), dtype=float).reshape(-1)
if run.shape[0] != n_frames: run = np.resize(run, n_frames)
```

iii. Notes identified `Run`/`RunFr` as speed sources, without establishing their semantics or matching the reference field.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. It computes session-wide numeric quartile thresholds using finite values and applies `np.digitize`; selected bins are then sliced per trial.

ii.
```python
qs = np.nanquantile(x[np.isfinite(x)], [0.25, 0.5, 0.75])
return np.digitize(x, qs, right=False)
```

iii. The notes intended quartiles but acknowledge imperfect class coverage.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below the 25th, 50th, and 75th percentile thresholds become categories 0–3. Ties are not split, so bins need not contain 25% each; NaNs also digitize rather than being explicitly handled.

ii.
```python
qs = np.nanquantile(x[np.isfinite(x)], [0.25, 0.5, 0.75])
np.digitize(x, qs, right=False)
```

iii. The agent sought quartiles over valid data, but verification showed strongly unequal classes in several sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The session-wide category vector is sliced with the identical inferred interval and stride used for neural data.

ii.
```python
speed_bins[s:e+1:frame_stride]
```

iii. The agent intended framewise alignment, subject to incorrect source choice, resize behavior, and trial boundaries.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavior matches are skipped; missing reward/run become zeros; missing position becomes a synthetic ramp; retinotopy failures become `All`; mismatched arrays are often cyclically resized; lick index mismatches are clipped; inferred bounds are clamped. These choices prevent crashes but can silently invent or corrupt data.

ii.
```python
run = np.resize(run, n_frames)
pos = np.linspace(0, 4, n_frames)
reward_avail = np.resize(reward_avail, ntrials)
bri = np.resize(bri, spk.shape[0])
idx = idx[idx < lick_fr.shape[0]]
```

iii. Notes specifically document hardening lick indexing after a crash. Other fallbacks were described as caveats/placeholders, not scientifically validated corrections.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating enormous spike arrays, slicing/casting a separate neural array for every trial, retaining all converted trials in memory, and pickling the roughly 16 GB result dominate. Full conversion reportedly took about 182 seconds.

ii.
```python
spk = load_spike_session(sid)
trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
pickle.dump(data, f)
```

iii. Notes focus on dataset size and use downsampling/float16 to make full conversion feasible.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial construction, per-lick trial lookup, stimulus id mapping, region-code assignment, and behavior-entry copying could be vectorized or pre-indexed. The per-trial neural materialization is still required by the target nested format, but repeated event lookup is avoidable.

ii.
```python
for tr in range(ntrials):
    idx = np.where(lick_tr == tr)[0]
stim_ids_global = np.array([stim_map_global[v] for v in stim_vals])
```

iii. The agent's notes mention size optimizations but do not analyze vectorization in detail.

## 12-c. What processing does the code repeat multiple times?

i. Stimulus categories are extracted once to build the global vocabulary and again per session; trial-local lick events are found by scanning `lick_tr` once per trial; neural arrays are first cast/concatenated as float32 and later cast per trial to float16.

ii.
```python
for _, b in matched: vals, cats, ids = get_stimulus_categories(sess)
# later
stim_vals, stim_names_local, stim_ids = get_stimulus_categories(sess)
idx = np.where(lick_tr == tr)[0]
```

iii. No justification is documented for these repetitions.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes unused values (`source_file`, local stimulus ids/names, `candidates`, parsed date/run, returned area names), loads all behavior paradigms before discarding most sessions, and computes outputs for variables that are returned but ignored by callers.

ii.
```python
source_file[sid] = f.name
candidates = [k for k in sess.keys() if 'tr' in k.lower() ...]
stim_vals, stim_names_local, stim_ids = get_stimulus_categories(sess)
subj, date, run = parse_session_id(sid)
```

iii. These appear to be exploration/debugging remnants; the notes provide no downstream purpose.
