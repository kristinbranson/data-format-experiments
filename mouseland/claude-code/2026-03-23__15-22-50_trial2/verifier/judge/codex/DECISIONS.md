# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent builds a unique session map from `beh/Imaging_Exp_info.npy`, using one `mname_datexp_blk` key per recording. For each session it then loads one behavior dictionary entry from `Beh_<exp_type>.npy`, one spike file from `spk/<session>_neural_data.npy`, and one retinotopy file from `retinotopy/<mouse>_<date>_trans.npz`. Unlike the reference, it reloads behavior per session instead of grouping sessions by behavior file.

ii. ```python
def build_session_map():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    session_map = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
            if spk_key not in session_map or 'stimtype' not in ndb:
                session_map[spk_key] = {
                    'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
                }
    return session_map
```

```python
def load_beh(session_info):
    beh_all = np.load(
        os.path.join(DATA_ROOT, 'beh', f"Beh_{session_info['exp_type']}.npy"),
        allow_pickle=True
    ).item()
    return beh_all[session_info['beh_key']]
```

iii. The recorded justification is that sessions can appear under multiple experiment types but correspond to the same physical recording, so the conversion should use one entry per neural file. In `CONVERSION_NOTES.md` the agent says it will “use one entry per neural file” and that behavior is “the same regardless of which experiment type references it.”

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mname`. The converter tracks the first time each mouse name appears and assigns a `subject_idx` for each processed session from that map.

ii. ```python
mname = info['db']['mname']
...
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
...
subject_idx_list.append(subjects_seen[mname])
...
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. The notes describe the subject mapping as “mouse name -> subjects, subject_idx.” There is no deeper justification beyond using the session metadata field that already names the mouse.

## 1-c. How are the data split into sessions?

i. A session is one unique spike-file key `mname_datexp_blk`. The session map deduplicates repeated entries from `Imaging_Exp_info.npy`, preferring an entry without `stimtype` when duplicates exist.

ii. ```python
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
# Prefer entries without stimtype
if spk_key not in session_map or 'stimtype' not in ndb:
    session_map[spk_key] = {
        'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
    }
```

iii. In Step 4 of `CONVERSION_NOTES.md`, the agent justifies this by stating that 142 metadata entries collapse to 89 unique physical recordings and that sessions with `stimtype` variants have identical underlying trial data.

## 1-d. How are the data split into trials?

i. Trials are split purely by `StartFr`: a trial runs from `StartFr[i]` up to `StartFr[i+1]`, or the end of the session for the last trial. This keeps corridor frames and the following gray-space/inter-trial frames, rather than restricting to `ft_CorrSpc` frames inside the texture corridor.

ii. ```python
StartFr = beh['StartFr'].astype(int)
...
for i in range(ntrials):
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    start = max(0, start)
    end = min(nfr_use, end)
    n_frames = end - start
    if n_frames < 2:
        continue
```

iii. The explicit rationale in `CONVERSION_NOTES.md` is: “Use frames from `StartFr` to start of next trial (or end of session). This captures corridor + gray space.”

## 1-e. How are trials filtered based on quality controls?

i. The code applies only minimal trial QC. It truncates all arrays to the imaged frame count, drops trials shorter than 2 frames, and later drops entire sessions with fewer than 2 surviving trials. It does not apply the reference solution’s long-trial percentile filter.

ii. ```python
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
...
if n_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
```

iii. Step 3 of `CONVERSION_NOTES.md` says “For decoder: use all trials (no filtering).” The implementation keeps that spirit except for the `<2` frame guard and the requirement that sessions retain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `spks` arrays in each session’s neural `.npy` file, and region labels come from `iarea` in the retinotopy `.npz` file.

ii. ```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
planes = spk_data['spks']
...
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The notes explicitly identify the neural source as “Suite2p deconvolved calcium traces” stored in `spks`, with area assignments from retinotopy.

## 2-b. How is the `neural` data processed?

i. The agent filters neurons per plane before concatenation, concatenates the kept planes, truncates the session to `nfr_use`, and stores each per-trial neural matrix as `float16`. It does not compute dF/F or perform additional deconvolution.

ii. ```python
for plane in planes:
    n = plane.shape[0]
    plane_mask = valid_mask[offset:offset+n]
    filtered.append(plane[plane_mask].astype(np.float16))
    offset += n
return np.concatenate(filtered, 0)
```

```python
trial_spk = spk[:, start:end].copy()
```

iii. The notes justify this as a memory optimization and as consistent with the source data already being deconvolved traces: “No additional delta F/F computation needed” and “Store neural data as float16.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is just visual-cortex filtering: neurons with `iarea == -1` or `iarea == 7` are excluded, and the remaining neurons are mapped into `V1`, `mHV`, `lHV`, or `aHV`.

ii. ```python
EXCLUDED_AREAS = {-1, 7}
...
valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
region_idx = np.array([
    BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
    for ia in iarea[valid_mask]
], dtype=np.int64)
```

iii. The notes state this choice repeatedly as matching the reference code’s “inside visual cortex” rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural matrices are aligned to corridor entry only in the sense that each trial starts at `StartFr`. The trial then extends until the next trial start, so alignment is to trial start but not restricted to the corridor segment requested by the reference solution.

ii. ```python
for i in range(ntrials):
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    ...
    trial_spk = spk[:, start:end].copy()
```

iii. The justification recorded in the notes is that the decoder should use trial-start alignment and retain gray space: “Use frames from `StartFr` to start of next trial.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stays at the native imaging-frame resolution, about 315 ms per bin. No temporal rebinning is applied.

ii. ```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
metadata = {
    'time_bin_size': TIME_BIN_MS,
    ...
}
```

```python
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400
fs = 1.0 / dt if dt > 0 else FRAME_RATE
```

iii. The notes explicitly say: “Time bin size: Use native frame rate (~315 ms). No resampling needed.”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Time to sound cue is derived from `SoundFr` together with frame indices and the session frame rate estimated from `ft`.

ii. ```python
SoundFr = beh['SoundFr']
...
ft = beh['ft']
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400
fs = 1.0 / dt if dt > 0 else FRAME_RATE
...
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
```

iii. In the mapping notes the agent describes this as `Time to SoundFr -> (SoundFr - current_frame) / fs`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code subtracts each frame index from the trial’s `SoundFr` and divides by `fs`, producing a continuous per-frame time-to-cue signal. If `SoundFr` is `NaN`, it fills the whole trial with zeros.

ii. ```python
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The notes justify the sign convention directly: “(SoundFr - current_frame) / fs” so the value is time remaining until cue. There is no separate justification for zero-filling missing cues beyond “handle missing data appropriately.”

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time-to-sound vector is built on the same `start:end` frame slice used to extract the neural matrix for that trial.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
frame_idx = np.arange(start, end)
...
time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The implementation itself is the justification here: all trial-wise inputs are generated from the same frame range as the trial-wise neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Day of training is derived from metadata fields in `Imaging_Exp_info.npy`, using `days` if present, otherwise `sess#`, otherwise 0.

ii. ```python
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
```

iii. The notes explicitly state this plan: “Use `sess#` or `days` field from exp_info if available; otherwise use session chronological order within subject.” The fallback to chronological order was not implemented.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code converts the chosen metadata field to `float32` and broadcasts that scalar across every time bin in the trial.

ii. ```python
day = np.float32(get_session_day(db))
...
np.full(n_frames, day, dtype=np.float32)
```

iii. The notes justify this only as a per-trial scalar decoder input. No additional transformation is applied.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Time since trial start is derived from `StartFr`, frame indices within the trial, and the session frame rate estimated from `ft`.

ii. ```python
StartFr = beh['StartFr'].astype(int)
...
ft = beh['ft']
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400
fs = 1.0 / dt if dt > 0 else FRAME_RATE
```

iii. The notes summarize the mapping as `(current_frame - StartFr) / fs`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Within each trial, the code computes elapsed time as `np.arange(n_frames) / fs`, so the first kept frame is 0 s and later bins increase in native frame steps.

ii. ```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. The rationale in `CONVERSION_NOTES.md` is to represent time since corridor entry as a continuous, time-varying decoder input that starts at 0.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Time since trial start is aligned to the neural data because it is built for the same number of bins as `spk[:, start:end]` for each trial.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. The code constructs the input and neural arrays together inside the same trial loop, so alignment is by shared trial slice.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived from the per-trial behavior variable `isRew`.

ii. ```python
isRew = beh['isRew']
```

iii. The mapping notes explicitly list `isRew -> reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation is applied beyond casting to float and broadcasting the per-trial reward flag across all bins of the trial.

ii. ```python
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. The notes describe reward availability as a binary per-trial scalar input.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Visual stimulus category is derived from `WallName`.

ii. ```python
WallName = beh['WallName']
...
stim = standardize_stim_name(str(WallName[i]))
```

iii. The notes say “Visual stimulus: Use `WallName` directly, map to standardized categories.”

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent standardizes wall names through `STIM_CATEGORY_MAP`, but it keeps an 8-category stimulus space rather than collapsing to four base textures. It maps `rock*` and `brick*` onto `circle*` variants and `wood*` onto `leaf*` variants, collects all observed names across sessions, then assigns indices in a second pass.

ii. ```python
STIM_CATEGORY_MAP = {
    'circle1': 'circle1', 'circle2': 'circle2',
    'leaf1': 'leaf1', 'leaf2': 'leaf2', 'leaf3': 'leaf3',
    'leaf1_swap1': 'leaf1_swap1', 'leaf1_swap2': 'leaf1_swap2',
    'rock1': 'circle1', 'rock2': 'circle2',
    'wood1': 'leaf1', 'wood2': 'leaf2',
    'wood1_swap1': 'leaf1_swap1', 'wood1_swap2': 'leaf1_swap2',
    'brick1': 'circle1', 'brick2': 'circle2',
    'brick5': 'leaf3',
    'wood5': 'leaf3',
    'rock5': 'circle3',
}
```

```python
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The recorded justification is that the agent interpreted the decoder output as needing the more specific wall categories. `CONVERSION_NOTES.md` later confirms “8 stimulus categories.”

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`.

ii. ```python
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
```

iii. The notes identify `LickFr/LickTrind` as the source for a binary per-frame licking output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code creates a binary frame-level lick array, truncates lick frame numbers to integers, bounds-checks them against the imaged range, and sets each touched frame to 1.

ii. ```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. The notes justify this as a binary per-frame representation: “check if any lick occurred in that frame’s time window.”

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by slicing the precomputed frame-level lick array with the same `start:end` window used for neural data.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
lick_binary[start:end]
```

iii. The justification is implicit in the implementation: licking is represented on the same imaging-frame grid as the neural data and indexed with the same trial slice.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`, with `ft_CorrSpc` used to distinguish texture-corridor frames from gray-space frames.

ii. ```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. The notes explicitly discuss `ft_Pos` as the source and gray-space handling as an extra design choice.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code turns position into five categories, not four: four 1 m bins inside the texture corridor plus a separate default gray-space category.

ii. ```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3
```

iii. The notes show the agent debating this and then choosing the extra gray class: “I’ll use 4 bins and mark gray space as a 5th bin.”

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are hard-coded at 10-decimeter intervals inside the corridor: `[0,10)`, `[10,20)`, `[20,30)`, `[30,40)` mapped to classes 0-3, with class 4 used for gray-space frames.

ii. ```python
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3
```

iii. The justification is the same gray-space decision recorded in Step 5 of the notes.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned to neural data by taking `pos_bins[start:end]` for the same trial slice used for the neural matrix.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
pos_bins[start:end]
```

iii. The code constructs outputs inside the same loop and with the same `start:end` indices as the neural array.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```

iii. The notes list `ft_RunSpeed` as the source variable for running-speed decoding.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first collects `ft_RunSpeed` from every frame of the selected sessions, computes three global percentile thresholds from positive values only, and then applies those thresholds back to every frame in each session. It does not compute within-session equal-count bins on kept trial frames.

ii. ```python
def collect_speed_quartiles(session_map, keys):
    all_speeds = []
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    valid = all_speeds > 0
    if valid.sum() == 0:
        return np.array([1.0, 2.0, 3.0])
    return np.percentile(all_speeds[valid], [25, 50, 75])
```

iii. The notes justify this as “Discretize into 4 quartile bins computed across ALL running frames in the dataset,” and later note that Q1 becomes much larger because zeros are below all three thresholds.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is thresholded by three scalar cutoffs: values below the first percentile threshold are class 0, between thresholds are classes 1 and 2, and above the last threshold are class 3.

ii. ```python
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The notes describe these as global quartile thresholds rather than rank-based equal-count bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned with neural data by slicing the frame-level `speed_bins` array with the same `start:end` trial window used for neural extraction.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
speed_bins[start:end]
```

iii. The alignment is implicit in the shared per-trial slice.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several small data issues: it truncates behavioral streams to the imaged frame count, drops out-of-range lick indices, zero-fills `time_to_sound` when `SoundFr` is missing, clamps trial starts and ends into range, and skips trials shorter than 2 frames.

ii. ```python
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
```

```python
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
lick_binary[lick_fr[valid_lick]] = 1
```

```python
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
```

iii. The only explicit justification is the general instruction in the notes to handle missing data sensibly; the concrete handling is encoded directly in the script.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading and filtering the large neural spike files session by session. The notes estimate neural loading plus filtering at roughly 15-25 seconds per session and about 21 minutes for the full run.

ii. ```python
# Load retinotopy first to build mask before loading large neural data
iarea = load_retino(mname, datexp)
valid_mask, region_idx = get_brain_region_idx(iarea)
...
spk = load_spk_filtered(mname, datexp, blk, valid_mask)
```

iii. Step 7 of `CONVERSION_NOTES.md` explicitly identifies neural loading/filtering as the dominant runtime cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still contains several Python loops that could be vectorized or reorganized for efficiency, though the agent did not foreground them: per-session behavior loading in `collect_speed_quartiles`, the four-pass position bin loop, the per-trial conversion loop, and the second pass that fills stimulus indices after all sessions are processed.

ii. ```python
for spk_key in keys:
    info = session_map[spk_key]
    beh = load_beh(info)
    ...
```

```python
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
```

```python
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. There is no explicit recorded justification for keeping these loops. The closest notes are the performance bullets about neural I/O being the main bottleneck and behavior-only quartile collection being “fast.”

## 12-c. What processing does the code repeat multiple times?

i. The code repeats some processing. It reloads behavior files during the quartile pass and then reloads them again during session processing, and it stores per-trial stimulus names only to make a second pass that writes the stimulus indices into the output arrays.

ii. ```python
speed_quartiles = collect_speed_quartiles(session_map, keys)
...
result = process_session(spk_key, info, speed_quartiles)
```

```python
def load_beh(session_info):
    beh_all = np.load(...).item()
    return beh_all[session_info['beh_key']]
```

```python
all_stim_per_session.append(result['stim_names'])
all_stim_names.update(result['stim_names'])
...
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The notes emphasize memory savings rather than eliminating repeated work, so no explicit justification is given for these repeated passes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does a few pieces of extra work that are not needed downstream: it carries `stim_names`, `mname`, `nneu_total`, and `ntrials` inside each session result even though only some of them are later used; it performs a whole second pass just to replace placeholder stimulus zeros with final indices; and it supports optional plotting code that is irrelevant to the final dataset unless `--show-processing` is enabled.

ii. ```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'stim_names': stim_names,
    'region_idx': region_idx,
    'mname': mname,
    'nneu': nneu,
    'nneu_total': nneu_total,
    'ntrials': len(neural_trials),
}
```

```python
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),  # placeholder for stim idx
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. The agent did not explicitly discuss these as wasteful. The only stated reason for the extra stimulus pass is implicit: the final category set is discovered only after all sessions are processed.
