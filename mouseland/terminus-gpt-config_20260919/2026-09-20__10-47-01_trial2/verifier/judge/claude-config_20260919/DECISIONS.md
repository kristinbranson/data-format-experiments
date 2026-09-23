# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads everything from the three subfolders of `/app/data`: `beh/` (behaviour), `spk/` (deconvolved calcium traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is treated as the master index; it is grouped by experiment type and contains 142 membership rows. In `load_catalog()` the AI loads **every** `beh/Beh_<exp_type>.npy` file up front, builds a session id `mname_datexp_blk` for each row, resolves the behaviour key (bare id, or id + `_<stimtype>` for the swap sessions), and stores one canonical behaviour record per *physical* session in a dict `beh`, plus a list of the experiment types each session belongs to (`labels`). It additionally cross-checks the set of session ids against the filenames in `spk/` and raises if they differ. Spike files and retinotopy files are then loaded lazily, one per session, inside `convert_session()` / `region_indices()`.

ii.
```python
def load_catalog():
    """Return one canonical behavior record and experiment labels per physical session."""
    info = np.load(DATA/'beh/Imaging_Exp_info.npy', allow_pickle=True).item()
    beh, labels = {}, defaultdict(list)
    for exp, rows in info.items():
        d = np.load(DATA/f'beh/Beh_{exp}.npy', allow_pickle=True).item()
        for r in rows:
            sid = f"{r['mname']}_{r['datexp']}_{r['blk']}"
            keys = [sid, sid + (f"_{r['stimtype']}" if 'stimtype' in r else '')]
            hit = next((k for k in keys if k in d), None)
            if hit is None:
                raise KeyError(f'No behavior key for {sid} in {exp}')
            ...
            else:
                beh[sid] = d[hit]
            if exp not in labels[sid]: labels[sid].append(exp)
    neural_ids = {p.name.removesuffix('_neural_data.npy') for p in (DATA/'spk').glob('*_neural_data.npy')}
    if neural_ids != set(beh):
        raise ValueError(f'Neural/behavior IDs differ: ...')
    return beh, labels
```
```python
raw = np.load(DATA/'spk'/f'{sid}_neural_data.npy', allow_pickle=True).item()['spks']
...
rp = DATA/'retinotopy'/f"{sid.rsplit('_',1)[0]}_trans.npz"
with np.load(rp) as z: ia = np.asarray(z['iarea'])
```

iii. From CONVERSION_NOTES.md Step 1/2/4: "Experiment metadata are loaded from `beh/Imaging_Exp_info.npy`… behavior dictionary keys are built from those fields"; "There are 142 experiment-membership entries but only 89 unique physical recordings; repeated memberships point to identical behavior records. Conversion must deduplicate by mouse/date/block and include each neural file once." The spike loading mirrors reference `utils.load_spk` and the retinotopy loading mirrors `utils.load_retino`. The explicit ID cross-check was added so that "no data is missed during conversion" (Step 2 exploration checks: "Confirmed all 89 neural filenames have metadata-linked behavior and vice versa after physical-session deduplication").

## 1-b. How are the data split into subjects (mice)?

i. The subject is the mouse name, which the AI recovers as the first underscore-delimited field of the session id (equivalently `entry['mname']`, from which the id was constructed). `subjects` is the sorted set of unique mouse names over the sessions actually converted, and `subject_idx` is each session's index into that list, in the same order as `neural`/`input`/`output`. 19 subjects over 89 sessions.

ii.
```python
subjects = sorted({x.split('_')[0] for x in ids}); subidx = {x: i for i, x in enumerate(subjects)}
...
'subject_idx': np.array([subidx[x.split('_')[0]] for x in ids], dtype=np.int16),
```

iii. Step 2 of CONVERSION_NOTES: "Subjects | 19 (`DR10`, `DR15`, …)"; Step 5 mapping table: "`mname` → `subjects`, `subject_idx` | Sorted unique mouse IDs and per-session index | 19 subjects." The index already names the mouse, so no split has to be inferred.

## 1-c. How are the data split into sessions?

i. A session is one physical recording = one mouse × one date × one block (`mname_datexp_blk`), which is also the name of the spike file. Because the master index lists the same recording under several experiment types (and the swap sessions twice, once per `stimtype`), the AI deduplicates: the first occurrence is kept and every later occurrence is checked for consistency (`ntrials` must agree) and otherwise discarded. The set of deduplicated ids is then required to equal the set of `spk/*_neural_data.npy` filenames. Result: 89 sessions; sessions are emitted in `sorted(beh)` order.

ii.
```python
sid = f"{r['mname']}_{r['datexp']}_{r['blk']}"
...
if sid in beh:
    # Exploration established duplicate memberships are identical.
    if int(beh[sid]['ntrials']) != int(d[hit]['ntrials']):
        raise ValueError(f'Conflicting duplicate behavior for {sid}')
else:
    beh[sid] = d[hit]
...
neural_ids = {p.name.removesuffix('_neural_data.npy') for p in (DATA/'spk').glob('*_neural_data.npy')}
if neural_ids != set(beh):
    raise ValueError(...)
```
```python
ids = sorted(beh)
```

iii. Step 4 discrepancy table: "Session identity | Metadata are organized by experiment type | 142 memberships map to 89 physical neural files; duplicate records are identical | A session/stimulus may appear in several analyses | **Deduplicate strictly by mouse/date/block and include each neural file once**." Step 5 Key Decision 1: "Physical-session deduplication: Include all 89 unique neural files once. Experiment dictionaries are views/labels, not independent recordings." The experiment-type memberships are retained in `metadata['session_info'][i]['experiment_types']` so no information is lost.

## 1-d. How are the data split into trials?

i. Trials are the trials the behaviour file declares (`ntrials`, with per-frame labels in `ft_trInd`). For each trial the AI keeps the imaging frames that (a) carry that trial index, and (b) lie inside the 0–4 m textured corridor, i.e. `ft_Pos` finite and in `[0, 40)` decimetres. Inter-trial frames (NaN `ft_trInd`, 10,028 of them) and the 4–6 m grey-space frames are excluded. Nothing is cut from the middle of a trial and no padding is applied, so trials have their own natural length (11–5,607 frames).

I verified numerically on `TX88_2022_07_22_1_swap1` that the AI's mask `(ft_trInd==tr) & isfinite(ft_Pos) & (ft_Pos>=0) & (ft_Pos<40)` is **identical, frame for frame and trial for trial**, to the reference's `(ft_trInd==tr) & ft_CorrSpc` (22,085 frames both ways, 373/373 trials with identical counts).

ii.
```python
for tr in range(int(b['ntrials'])):
    ix = np.flatnonzero((tri==tr) & np.isfinite(pos) & (pos>=0) & (pos<40))
    if len(ix) < 2: raise ValueError(f'{sid} trial {tr}: only {len(ix)} corridor frames')
```

iii. Step 4: "Trial boundary … Extract finite `ft_trInd==trial`; equivalently ceil StartFr through floor EndFr. Exclude 10,028 unassigned inter-trial frames"; "Corridor entry/exit … Restrict decoder trials to native `ft_Pos<40` (0–4 m). `GrayFr` is corridor exit, not entry." Step 3: "The paper's spatial activity analyses use the 0–4 m texture/corridor region." Step 5 Key Decision 4: "No movement filtering: Reference `ft_move>0` is needed only for monotonic spatial interpolation. Retaining stops is necessary to decode temporal behavior and speed."

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level quality filtering is applied at all.** All 38,110 declared trials in all 89 sessions are kept, including trials in which the animal stopped for many minutes (longest retained trial = 5,607 frames ≈ 29 min; mean trial length 40.4 frames vs. median 32.3). The only implicit filter is the corridor restriction of 1-d; the code additionally *aborts* (does not skip) if any trial were to have fewer than 2 corridor frames, but the AI verified that the shortest trial in the dataset has 11 frames so this never fires.

ii.
```python
ix = np.flatnonzero((tri==tr) & np.isfinite(pos) & (pos>=0) & (pos<40))
if len(ix) < 2: raise ValueError(f'{sid} trial {tr}: only {len(ix)} corridor frames')
```
(there is no length cap, no movement criterion and no other rejection rule anywhere in the script)

iii. Step 3 "Trial curation rules": "No global bad-trial exclusion is described. Some figure analyses restrict to moving periods, rewarded trials with first lick after 2 m, or held-out subsets… Such figure-specific restrictions do not justify dropping valid decoder trials." Step 5 Key Decision 10: "All 38,110 trials have at least 11 corridor frames, satisfying the two-trial/session requirement. No source quality flag justifies further filtering." Step 10 Check 5: "Very long trials (maximum 5,607 retained frames) are valid stopped/slow-running behavior and remain included."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` — a list of three float32 (neurons × frames) arrays, one per imaging plane group — concatenated along the neuron axis. The per-neuron visual area comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`, whose length is checked to equal the concatenated neuron count.

ii.
```python
raw = np.load(DATA/'spk'/f'{sid}_neural_data.npy', allow_pickle=True).item()['spks']
nfr = min(a.shape[1] for a in raw)
if any(a.dtype != np.float32 for a in raw): raise TypeError(f'{sid}: expected float32 spks')
spk = np.concatenate(raw, axis=0); del raw
```
```python
with np.load(rp) as z: ia = np.asarray(z['iarea'])
if len(ia) != nneurons: raise ValueError(f'{sid}: retinotopy {len(ia)} != neurons {nneurons}')
out = np.full(nneurons, 4, dtype=np.int16)
out[ia==8]=0; out[np.isin(ia,[0,1,2,9])]=1; out[np.isin(ia,[5,6])]=2; out[np.isin(ia,[3,4])]=3
```

iii. Step 1: "Neural activity is already provided as `spks`. The reference loader performs no fluorescence baseline or delta-F/F calculation. Therefore conversion should not recompute dF/F." Step 4: "Neural representation | `load_spk` concatenates three `spks` arrays … Concatenate on neuron axis; do not calculate dF/F." Step 2: "Every retinotopy vector length exactly matches the corresponding concatenated neural count."

## 2-b. How is the `neural` data processed?

i. No signal processing is done. The concatenated `spks` array is truncated to the imaged frame count, the columns belonging to a trial are extracted, and the result is cast to `float16` purely as a storage encoding. Trials keep their native, variable length; nothing is normalised, smoothed, padded or interpolated.

ii.
```python
nfr = min(nfr, spk.shape[1], len(b['ft_trInd']))
spk = spk[:, :nfr]
...
neural.append(np.asarray(spk[:, ix], dtype=np.float16, order='C'))
```

iii. Step 3: "All paper analyses use the deconvolved fluorescence traces supplied as `spks`; no new dF/F should be computed." Step 5 Key Decision 5: "Neural float16 storage: Required to make the full temporal pickle practical (~135 GiB estimated corridor-only versus ~270 GiB float32). Spot checks found no overflow and negligible error… This is storage encoding, not normalization." Step 4: "Float16 suitability was spot-checked … sampled MAE 0.0004–0.0028 and maximum quantization error <0.64 compared with trace maxima 981–2,520."

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neurons are dropped.** All 4,691,034 Suite2p cells are kept. The retinotopic area is attached as a label: `iarea==8 → V1`, `{0,1,2,9} → mHV`, `{5,6} → lHV`, `{3,4} → aHV`, and everything else (label 7, 585,641 neurons ≈ 12.5%) is assigned to a fifth region called `unmapped` rather than being discarded. No d-prime / selectivity / SNR filter is applied.

ii.
```python
REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unmapped']
...
out = np.full(nneurons, 4, dtype=np.int16)          # default = 'unmapped'
out[ia==8]=0; out[np.isin(ia,[0,1,2,9])]=1; out[np.isin(ia,[5,6])]=2; out[np.isin(ia,[3,4])]=3
```

iii. Step 1: "No generic low-quality-neuron exclusion was found in the reference processing workflow. D-prime thresholds (for example 0.3) … are analysis-specific safeguards against circularity and should not be mistaken for source-data quality curation." Step 2 Region Mapping: "Label 7 is not assigned by that function and must be represented as an unmapped/outside-reference-area class rather than silently dropped." Step 4: "Keep all supplied Suite2p cells and all valid corridor trials. Do not impose figure-specific selection."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start). Each trial's array begins at the **first imaging frame that belongs to that trial and is inside the 0–4 m corridor** (equivalently `ceil(StartFr)`), and ends at the last such frame (corridor exit into grey space). Trials are variable length; there is no common window, no truncation and no padding. `metadata['off_start'] = 0.0`, `metadata['off_end'] = None`, `temporal_alignment_event = 'corridor entry (first imaging frame assigned to trial; ceil StartFr)'`.

ii.
```python
ix = np.flatnonzero((tri==tr) & np.isfinite(pos) & (pos>=0) & (pos<40))
neural.append(np.asarray(spk[:, ix], dtype=np.float16, order='C'))
```
```python
'temporal_alignment_event':'corridor entry (first imaging frame assigned to trial; ceil StartFr)',
'off_start':0.0, 'off_end':None,
```

iii. Step 4: "`StartFr` is fractional between prior/new trial: ceil matches new trial 100% … Align at first assigned frame/ceil StartFr. `GrayFr` is corridor exit, not entry." Step 5 Key Decision 2: "Temporal alignment: Corridor entry is the first finite trial-assigned frame (ceil `StartFr`), not `GrayFr`. Retain native timestamps until `ft_Pos<40` corridor exit." Step 10 Check 5 confirms "ceil StartFr belongs to the new trial 100%; floor EndFr belongs to current trial 99.992%".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. The imaging frame *is* the bin. The AI measures the bin size per session as the median difference of the `ft` timestamps (MATLAB datenums × 86400 s), and writes the median over sessions into metadata: **314.8 ms (~3.177 Hz)**. Every behaviour stream is already on the imaging-frame grid, so nothing needs to be resampled. The paper's 0.1-m spatial interpolation is deliberately *not* used.

ii.
```python
ft = np.asarray(b['ft'], float)[:nfr] * 86400.0
dt = float(np.median(np.diff(ft)))
...
'time_bin_size': float(np.median(dts)*1000),
```

iii. Step 4: "Time bin | Not hard-coded in reference | MATLAB-datenum `ft` median difference is 3.6436e-6 days = 314.81 ms (~3.1765 Hz) | … Retain native frames and record 314.8 ms representative bin; compute time values from actual `ft` timestamps to handle small jitter." Step 5 Key Decision 3: "Native temporal sampling: Do not spatially interpolate because the decoder explicitly requests time-varying cue time, elapsed time, licking, and speed."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the fractional imaging-frame number at which the sound cue was delivered on each trial) together with the frame indices of the trial and the session's median frame interval `dt` (derived from `ft`).

ii.
```python
ft = np.asarray(b['ft'], float)[:nfr]*86400.0
dt = float(np.median(np.diff(ft)))
sound = np.asarray(b['SoundFr'], float)
...
cue = ((sound[tr] - ix.astype(float)) * dt).astype(np.float32)
```

iii. Step 5 mapping table: "`SoundFr`, `StartFr`, `ft` → `input[0]` | Signed seconds until sound cue: `(SoundFr-current_frame)*session_median_frame_dt`; equivalently cue time minus current time | Continuous, time-varying as explicitly requested." Step 3: "Sound cue location is randomized independently by trial and is used instead of reward location for reward-prediction analyses because it is present in every corridor."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame the signed frame offset to the cue is converted to seconds by multiplying by the session's median frame interval: `time_to_cue = (SoundFr − frame_index) × dt`. It is positive before the cue and negative after it, and crosses zero at `SoundFr`. Note this uses a *constant* frame interval rather than the actual per-frame `ft` timestamps (the AI used real timestamps for elapsed time but not here). I measured the resulting discrepancy on real trials: ≤ 0.02 s, i.e. under 7% of one 315 ms bin, so it is immaterial. Because no long trials are dropped, the resulting range is [−1762 s, +722.7 s].

ii.
```python
cue = ((sound[tr] - ix.astype(float)) * dt).astype(np.float32)
inp = np.vstack([cue, np.full(len(ix), day, np.float32), elapsed,
                 np.full(len(ix), float(rew[tr]), np.float32)]).astype(np.float32)
```

iii. README: "**Time to sound cue (s)** — signed, time-varying; positive before and negative after cue." Step 5 planned sanity check: "Verify cue-time input changes by the negative of elapsed time and crosses zero near `SoundFr`" — reported as passing in the Step 10 `np.allclose` checks.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed directly from the same frame-index vector `ix` that selects the neural columns for that trial, so it shares the neural data's grid and length exactly. `SoundFr` is already expressed in imaging-frame coordinates, and the AI verified that cue frames fall inside the declared trial 100% of the time.

ii.
```python
ix = np.flatnonzero((tri==tr) & np.isfinite(pos) & (pos>=0) & (pos<40))
neural.append(np.asarray(spk[:, ix], dtype=np.float16, order='C'))
cue = ((sound[tr] - ix.astype(float)) * dt).astype(np.float32)
```

iii. Step 4: "Cue frame coordinates always belong to the indicated trial." All streams in this dataset live on the imaging-frame grid, so using one shared index vector guarantees alignment; an assertion `n.shape[1]==x.shape[1]==y.shape[1]` is run over every trial before writing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date `datexp`, parsed out of the session id (fields 1–3 of `mname_YYYY_MM_DD_blk`), for every session of the dataset (not only those being converted).

ii.
```python
def day_offsets(session_ids):
    dates = {s: datetime.strptime('_'.join(s.split('_')[1:4]), '%Y_%m_%d') for s in session_ids}
    first = {}
    for s, d in dates.items():
        mouse = s.split('_')[0]; first[mouse] = min(first.get(mouse, d), d)
    return {s: float((d - first[s.split('_')[0]]).days) for s, d in dates.items()}
...
days = day_offsets(sorted(beh))      # always over all 89 sessions, even in --sample mode
```

iii. Step 5 mapping table: "session date (`datexp`) → `input[1]` | Calendar days elapsed from that subject's earliest supplied imaging date; repeat across trial | Continuous per-trial training-day proxy; stage labels retained in metadata."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Per mouse, the earliest supplied imaging date is found and each session is assigned the number of **calendar days elapsed** since it (first session = 0). The value is a float, broadcast across every bin of every trial of that session. Range across the dataset: 0–92 days. The offsets are always computed over all 89 sessions so that `--sample` and `--full` agree. The experiment-stage membership (e.g. `unsup_train1_before_learning`) is additionally kept in `metadata['session_info']`.

ii.
```python
return {s: float((d - first[s.split('_')[0]]).days) for s, d in dates.items()}
...
np.full(len(ix), day, np.float32)
```

iii. Step 5 Key Decision 7: "Training day: Use elapsed calendar days from each subject's first supplied imaging date. This is continuous, objective, and preserves gaps between before/after-learning recordings; metadata also records experiment-stage memberships."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft`, the MATLAB-datenum timestamp of every imaging frame (converted to seconds), evaluated on the trial's own frame indices; the origin is the first retained corridor frame of the trial rather than the fractional `StartFr`.

ii.
```python
ft = np.asarray(b['ft'], float)[:nfr] * 86400.0
...
# Actual timestamps can have small jitter; re-zero at corridor entry.
elapsed = (ft[ix] - ft[ix[0]]).astype(np.float32)
```

iii. Step 5 mapping table: "`ft`, first retained frame → `input[2]` | Actual elapsed seconds from retained corridor-entry frame | synchronized imaging timestamps | Continuous, time-varying."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Elapsed seconds since the first retained frame, so the first sample of every trial is exactly 0.0 and the series increases monotonically with the real (jittered) frame timestamps. It is never negative. Because no long trials are dropped, the range across the dataset is [0, 1765 s]. (The reference instead uses `ft − interp(StartFr)`, giving a small positive first value; I measured that offset at 0.15–0.30 s, i.e. under one bin.)

ii.
```python
elapsed = (ft[ix] - ft[ix[0]]).astype(np.float32)
inp = np.vstack([cue, np.full(len(ix), day, np.float32), elapsed, ...])
```

iii. Step 4: "Retain native frames and record 314.8 ms representative bin; **compute time values from actual `ft` timestamps to handle small jitter**." Step 5 planned sanity check "Verify first time-since-start is zero" — confirmed in Step 7 ("first elapsed-time sample is zero for all inspected trials") and Step 10.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is evaluated on the same `ix` frame-index vector used to slice the neural columns, so it is by construction the same length and the same grid, with sample 0 corresponding to the neural column at corridor entry.

ii.
```python
neural.append(np.asarray(spk[:, ix], dtype=np.float16, order='C'))
elapsed = (ft[ix] - ft[ix[0]]).astype(np.float32)
```

iii. All streams are indexed by imaging frame; a single index vector per trial is used for neural, inputs and outputs, and a per-trial assertion `n.shape[1]==x.shape[1]==y.shape[1]` is checked before pickling.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial flag marking trials run in the rewarded corridor.

ii.
```python
rew = np.asarray(b['isRew'], bool)
```

iii. Step 5 mapping table: "`isRew` → `input[3]` | Boolean to 0/1 and repeat across time | 1 means rewarded corridor, independent of whether reward was delivered."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a bool → float cast and broadcasting across the trial's bins. 11.38% of the 38,110 trials are rewarded; unsupervised/naive sessions are all zero.

ii.
```python
np.full(len(ix), float(rew[tr]), np.float32)
```

iii. Step 2: "Rewarded trials | 4,336 / 38,110 = 11.38%". Step 5: "1 means rewarded corridor, independent of whether reward was delivered." Step 10 Check 4: "Rewarded source fraction is 11.38%; converted reward availability contains both classes and is repeated over valid time bins as intended."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the concrete texture name on the corridor walls of each trial. The AI explicitly rejected `TrialStim` (the reference-normalised category field) because 3,068 trials carry placeholder values there.

ii.
```python
STIMULI = ['circle1','circle2','circle3','leaf1','leaf1_swap1','leaf1_swap2',
           'leaf2','leaf3','rock1','rock2','wood1','wood1_swap1','wood1_swap2','wood2','wood5']
...
wall = np.asarray(b['WallName']).astype(str)
stim_to_idx = {x: i for i, x in enumerate(STIMULI)}
if wall[tr] not in stim_to_idx: raise ValueError(f'Unknown stimulus {wall[tr]}')
```

iii. Step 4: "Stimulus identity | Code uses normalized IDs and wall names | `TrialStim` usually normalized but has 3,068 placeholders; `WallName` is always concrete." Trajectory step 48: "Placeholder `TrialStim` values correspond to legitimate additional concrete stimuli, so `WallName` is the only complete visual-identity source."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's `WallName` is mapped to its index in a hard-coded, sorted list of the **15 concrete wall identities** that occur across the dataset (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`), and broadcast across the trial's bins as int16. The AI deliberately did **not** collapse exemplars/crops/spatial-swap variants into the four base textures (circle / leaf / rock / wood), so the output has 15 classes (chance 0.067) rather than 4. Unknown names raise.

ii.
```python
out = np.vstack([np.full(len(ix), stim_to_idx[wall[tr]], np.int16), lick, pbin, sbin])
...
'output_values': [STIMULI, ['not licking','licking'], ['0-1 m','1-2 m','2-3 m','3-4 m'],
                  ['speed Q1','speed Q2','speed Q3','speed Q4']],
```

iii. Step 5 Key Decision 6: "Visual identity: Use concrete `WallName`, the complete trial variable. Do not use placeholder-containing `TrialStim` **or collapse wood/rock variants**, because the requested output is the presented visual stimulus category." Trajectory step 48: "It should be used directly, preserving all 15 concrete visual categories rather than collapsing or inventing normalized labels."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (fractional imaging-frame number of every lick in the session) together with `LickTrind` (the trial each lick belongs to). 74,483 lick events across 28 sessions.

ii.
```python
lickfr = np.asarray(b['LickFr'], float); licktr = np.asarray(b['LickTrind'], int)
```

iii. Step 5 mapping table: "`LickFr`, `LickTrind` → `output[1]` | Binary vector; each event assigned to nearest retained frame within its declared trial | `get_lick_raster` groups by `LickTrind`." Step 3 notes that the paper's own lick definition (first lick before the cue) is a figure-specific analysis: "The decoder task instead explicitly requests binary time-varying licking, so all lick events must be represented."

## 8-b. What processing is involved in computing `output` *Licking*?

i. A zero vector of the trial's length is created; for each lick event whose `LickTrind` equals the trial, the *nearest* retained corridor frame is found and set to 1, but only if the nearest frame is within 1.0 frame of the fractional event time (so licks that happened in the grey space or between retained frames are dropped). Multiple licks in one bin stay 1. Result: 3.5% of retained bins are lick bins.

ii.
```python
lick = np.zeros(len(ix), dtype=np.int16)
for ev in lickfr[licktr==tr]:
    k = int(np.argmin(np.abs(ix.astype(float)-ev)))
    # Keep only events whose nearest native frame lies in this retained corridor.
    if abs(float(ix[k]) - float(ev)) <= 1.0: lick[k] = 1
```

iii. Step 5 Key Decision 8: "Lick binning: Assign each lick to the nearest retained frame in its declared trial, avoiding fractional-boundary off-by-one errors. Licks outside the requested corridor window are intentionally absent." Trajectory step 48: "Lick event frame coordinates are fractional; rounding matches the declared lick trial for 74,321/74,483 events, better than floor or ceil."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already in imaging-frame coordinates, and the lick vector is built directly on the trial's `ix` frame vector (index `k` is a position within `ix`), so it is the same length and grid as the neural columns.

ii.
```python
neural.append(np.asarray(spk[:, ix], dtype=np.float16, order='C'))
k = int(np.argmin(np.abs(ix.astype(float)-ev)))
lick[k] = 1
```

iii. The trial membership is taken from `LickTrind` rather than inferred, and the ±1-frame tolerance guards the trial boundaries; Step 10 independently re-derived the lick vectors for 9 trials from raw files and matched with `np.allclose`.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the in-corridor position at each imaging frame, in decimetres (0–40 across the texture, continuing to 60 through the grey space).

ii.
```python
pos = np.asarray(b['ft_Pos'], float)[:nfr]
```

iii. Step 4: "Position units | Reference uses 60 bins over length 60 | `ft_Pos` spans 0–60 and `Texture_Length=40` | Native unit is 0.1 m. Convert meters=`ft_Pos/10`; classes are [0,1), [1,2), [2,3), [3,4] m."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. `ft_Pos` is divided by 10 (decimetres → metres) and floored, giving an integer 0–3, stored as int16 and broadcast nowhere (it is genuinely time-varying, one value per bin). Since the trial window already excludes `ft_Pos ≥ 40`, the clip is a no-op safety net.

ii.
```python
pbin = np.clip(np.floor(pos[ix]/10.0), 0, 3).astype(np.int16)
```

iii. Step 5 mapping table: "`ft_Pos` → `output[2]` | Native decimeters /10; bins `[0,1)`, `[1,2)`, `[2,3)`, `[3,4]` m → classes 0–3 | Time-varying, four equal 1 m bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length 1-m bins covering 0–4 m, exactly as the Decoder Task specifies: `[0,1) → 0`, `[1,2) → 1`, `[2,3) → 2`, `[3,4) → 3`, named `'0-1 m'…'3-4 m'`. Full-dataset occupancy: 28.5 / 23.3 / 23.6 / 24.6%.

ii.
```python
pbin = np.clip(np.floor(pos[ix]/10.0), 0, 3).astype(np.int16)
'output_values': [..., ['0-1 m','1-2 m','2-3 m','3-4 m'], ...]
```

iii. Step 9 consistency table: "Position bins | Four requested 1 m bins | 0–4 m texture region | 28.5%, 23.3%, 23.6%, 24.6% | Yes." The first bin is over-represented because animals slow down/stop most often near entry, which the AI notes is genuine behaviour rather than a binning error.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame and is sliced with the same `ix` as the neural columns, so it is on the identical grid and length. It is also the variable that *defines* the trial window, so alignment is exact by construction.

ii.
```python
neural.append(np.asarray(spk[:, ix], dtype=np.float16, order='C'))
pbin = np.clip(np.floor(pos[ix]/10.0), 0, 3).astype(np.int16)
```

iii. Same rationale as the other streams: everything is on the imaging-frame grid. The `--show-processing` plots overlay position bin against elapsed time for 4 trials per session to make the monotonic progression visible.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame (cm/s), truncated to the imaged frames.

ii.
```python
speed = np.asarray(b['ft_RunSpeed'], float)[:nfr]
```

iii. Step 5 mapping table: "`ft_RunSpeed` → `output[3]` | Global quantile edges … | paper interpolates speed to imaging frames | Four empirical quartile classes."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No transformation of the speed values themselves. The AI pre-computed, in an exploratory pass over the **whole dataset** (all valid 0–4 m corridor frames of all 89 sessions after truncation to the imaged frames), the 25/50/75th percentiles of `ft_RunSpeed`: `0, 8.32701545, 30.1567626` cm/s. These three numbers are **hard-coded as a module constant** in `convert_data.py` (they are not recomputed at run time), and are also written to `metadata['speed_quartile_edges_cm_s']`. A single global set of edges is used for every session and trial.

ii.
```python
SPEED_EDGES = np.array([0.0, 8.32701545, 30.15676260], dtype=np.float32)
...
'speed_quartile_edges_cm_s': SPEED_EDGES.tolist(),
```

iii. Step 5 Key Decision 9: "Speed quartiles: Compute once globally over all finite valid corridor frames after behavior-to-neural truncation. Negative speeds and stops are valid behavior and remain in the lowest class." Trajectory step 47: "Global valid-corridor speed quartiles are 0, 8.327, and 30.157 cm/s. About 20.4% of frames are exactly zero and 9.84% are negative."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.searchsorted(SPEED_EDGES, speed, side='right')` gives the class, i.e. the four bins are `speed < 0`, `[0, 8.327)`, `[8.327, 30.157)`, `≥ 30.157` cm/s, labelled `speed Q1…Q4`. Because the 25th percentile is exactly 0 and 20.38% of frames sit at exactly zero, the right-sided convention pushes all the zero-speed (stopped) frames into class 1. The resulting class occupancies are **9.8 / 40.2 / 25.0 / 25.0%**, not the 25% each the Decoder Task asks for; class 0 ends up meaning "negative speed" only. The AI detected this, corrected its written interval notation in Step 10, and argued the imbalance is unavoidable.

ii.
```python
sbin = np.searchsorted(SPEED_EDGES, speed[ix], side='right').astype(np.int16)
```

iii. Step 5 mapping table: "Exact 25% counts are impossible due to 20.38% exact-zero ties; deterministic quantile convention documented." Step 9: "The speed-bin imbalance in the first two classes is not an error: 20.38% of valid frames have speed exactly zero, making exact equal-size value intervals impossible without arbitrarily assigning identical values to different classes. The global quantile edges are reproducible; upper classes are exactly 25% each." Step 10 Check 5: "Corrected documentation interval notation for right-sided speed thresholds; code/data were already correct."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` provides one value per imaging frame and is sliced with the same `ix` as the neural columns, so it shares the grid and the length. No temporal smoothing or shifting is applied.

ii.
```python
neural.append(np.asarray(spk[:, ix], dtype=np.float16, order='C'))
sbin = np.searchsorted(SPEED_EDGES, speed[ix], side='right').astype(np.int16)
```

iii. Step 3: "Paper states running speed was interpolated to imaging-frame timepoints", so the stream is already synchronised; a single per-trial index vector guarantees it stays that way.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures, all fail-loud rather than fail-soft:
- Behaviour runs 1–3 frames past imaging in every session, so **all** behaviour streams are truncated to `nfr = min(min plane frame count, spk.shape[1], len(ft_trInd))`.
- The 10,028 inter-trial frames with NaN `ft_trInd` are excluded by the `tri==tr` comparison, and `np.isfinite(pos)` guards NaN positions.
- Lick events whose nearest retained frame is more than 1 frame away are dropped.
- The retinotopy vector length is asserted equal to the concatenated neuron count.
- The set of behaviour session ids is asserted equal to the set of spike filenames; duplicate behaviour memberships are asserted to have equal `ntrials`.
- `spks` dtype is asserted float32; an unknown `WallName` raises.
- Structural assertions (per-session ≥2 trials; per-trial neural/input/output lengths equal; 4 input and 4 output rows) run before pickling.

Note that anomalies raise exceptions that abort the whole conversion rather than skipping the offending trial or session (`raise ValueError(f'{sid} trial {tr}: only {len(ix)} corridor frames')`, and the `assert len(ns)>=2` after all 89 sessions are processed). In practice nothing fires — the AI verified the shortest trial in the dataset has 11 frames.

ii.
```python
nfr = min(a.shape[1] for a in raw)
...
nfr = min(nfr, spk.shape[1], len(b['ft_trInd']))
spk = spk[:, :nfr]
tri = np.asarray(b['ft_trInd'])[:nfr]; pos = np.asarray(b['ft_Pos'], float)[:nfr]
```
```python
if len(ia) != nneurons: raise ValueError(f'{sid}: retinotopy {len(ia)} != neurons {nneurons}')
if any(a.dtype != np.float32 for a in raw): raise TypeError(f'{sid}: expected float32 spks')
if wall[tr] not in stim_to_idx: raise ValueError(f'Unknown stimulus {wall[tr]}')
```
```python
assert len(neural)==len(inp)==len(out)==len(regions)==len(ids)
for ns, xs, ys in zip(neural, inp, out):
    assert len(ns)==len(xs)==len(ys) and len(ns)>=2
    for n, x, y in zip(ns, xs, ys): assert n.shape[1]==x.shape[1]==y.shape[1] and x.shape[0]==4 and y.shape[0]==4
```

iii. Step 2: "Behavior streams have 1–3 more terminal frames than neural data (59 sessions: +1, 23: +2, 7: +3). This matches reference code that slices behavior to neural `nfr`." Step 4: "Truncate all frame behavior to neural frame count." Step 10 Check 5 enumerates the edge cases checked (fractional `StartFr`/`EndFr`, NaN trial ids, label-7 neurons, lick boundaries) and concludes "Every trial has 11 or more retained corridor frames; no empty/one-frame trials."

## 12-a. What are the most time-consuming steps of the code?

i. The AI identifies neural file I/O as dominant and reports measured timings: 7–38 s per session (89 sessions, ~405 GiB of spike files read), 668.10 s total, of which 135.41 s (20%) is the final `pickle.dump` of the 137.9 GiB output. It mitigated the I/O cost with a 3-worker `ThreadPoolExecutor` (NumPy releases the GIL during file reads and casts). Not identified by the AI, but also non-trivial: `load_catalog()` eagerly loads **all 23 behaviour `.npy` files (~5 GiB)** and holds all 89 behaviour records in RAM for the whole run, including in `--sample` mode where only 2 are needed.

ii.
```python
t0=time.time(); raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
...
print(f'{sid}: neurons={len(ridx):,}, trials={len(neural):,}, timepoints=..., {time.time()-t0:.2f}s',flush=True)
```
```python
        # NumPy I/O, slicing, and casting release the GIL. Three bounded workers reduce
        # wall time without multiprocessing copies of multi-GB return objects.
        from concurrent.futures import ThreadPoolExecutor
        executor=ThreadPoolExecutor(max_workers=3)
        results=executor.map(job,items)
```

iii. Step 6: "The source object format requires materializing each full neural session before selecting corridor frames. Trial-list pickle representation requires copying selected neural columns; full output is expected to be large." Step 9: "Full conversion plus serialization: 668.10 s (11.14 min), below the 15-minute target… Three bounded NumPy worker threads reduced conversion wall time without extra process serialization."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's own answer is that only the lick loop is a Python loop and that it is negligible ("loops only over 74,483 sparse events"). That is correct as far as it goes — but it misses the larger one: the per-trial loop recomputes `np.flatnonzero((tri==tr) & np.isfinite(pos) & (pos>=0) & (pos<40))`, a full scan of the frame axis, once per trial. For a 789-trial × 34,228-frame session that is ~27 M element comparisons × 4 predicates × 789 iterations. Grouping frames by trial in a single `argsort`/`bincount` pass would replace all of it. The lick loop is also quadratic in miniature (`np.argmin` over the whole `ix` per event) and could be a single `np.searchsorted`. Both are indeed dwarfed by the 405 GiB of neural I/O.

ii.
```python
for tr in range(int(b['ntrials'])):
    ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))   # full-array scan per trial
    ...
    for ev in lickfr[licktr==tr]:
        k=int(np.argmin(np.abs(ix.astype(float)-ev)))                  # O(len(ix)) per event
```

iii. Step 6 "Code inefficiencies identified" lists only the materialisation of the neural session and the per-lick loop; "Code speedups added" claims "Vectorized frame masks, output discretization, and direct NumPy slicing."

## 12-c. What processing does the code repeat multiple times?

i. The corridor-validity mask `np.isfinite(pos) & (pos>=0) & (pos<40)` is recomputed inside the per-trial loop, i.e. `ntrials` times per session (up to 789×) even though it is trial-independent and could be computed once. `nfr` is computed twice. `stim_to_idx` is rebuilt for every session. In `--sample` mode, all 23 behaviour files are read even though only 2 sessions are converted. The Step 6 notes claim "Vectorized frame masks … before slicing", which is not what the code does.

ii.
```python
    for tr in range(int(b['ntrials'])):
        ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
```
```python
    nfr=min(a.shape[1] for a in raw)
    ...
    nfr=min(nfr,spk.shape[1],len(b['ft_trInd']))
```
```python
    stim_to_idx={x:i for i,x in enumerate(STIMULI)}   # inside convert_session
```

iii. The AI did not flag any repeated processing; Step 6 asserts the opposite ("Vectorized frame masks, output discretization, and direct NumPy slicing"). Its implicit justification is that everything except neural I/O is negligible: "Per-lick assignment loops only over 74,483 sparse events and is negligible compared with neural I/O."

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reports none. Things it nevertheless computes/stores that downstream decoding does not need:
- 585,641 `unmapped` (retinotopy label 7) neurons are kept and serialised — ≈12.5% of the 137.9 GiB output — although the reference paper analyses only V1/mHV/lHV/aHV.
- Because no trial-length filter is applied, extreme stopped trials are written in full; a single 5,607-bin trial contributes thousands of near-identical stationary samples that carry essentially no new information.
- `labels` / `experiment_types` and per-session `median_frame_interval_ms` are assembled for metadata only.
- A per-session `dt` is computed for every session but only `np.median(dts)` reaches `time_bin_size`.
- `np.clip(..., 0, 3)` on the position bins is a no-op given the `pos < 40` trial window.

ii.
```python
REGIONS = ['V1','mHV','lHV','aHV','unmapped']
out = np.full(nneurons, 4, dtype=np.int16)      # label-7 neurons retained and serialised
```
```python
sinfo.append({'session_id':sid,'experiment_types':sorted(labels[sid]),'training_day_elapsed':days[sid],
              'n_trials':len(n),'n_neurons':len(r),'median_frame_interval_ms':dt*1000})
```
```python
pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
```

iii. The AI's justification for retaining rather than pruning is explicit: Step 2 Region Mapping — label 7 "must be represented as an unmapped/outside-reference-area class rather than silently dropped"; Step 10 Check 5 — "Retinotopy label 7 is retained as `unmapped`; all 4,691,034 neurons have a region index"; Step 10 Check 5 — "Very long trials (maximum 5,607 retained frames) are valid stopped/slow-running behavior and remain included." The metadata fields are kept for traceability of the experiment-type memberships collapsed during deduplication.
