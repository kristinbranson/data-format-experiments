# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `data/beh/Imaging_Exp_info.npy`, builds unique session IDs, looks up behavior files by experiment type, and loads each session's spike and retinotopy files. In practice only 76 of 89 sessions were converted because swap-key behavior sessions were not found.

ii.
```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)
spk = load_spk(mname, datexp, blk)
retino = load_retino(mname, datexp)
```

iii. The notes say this follows the reference loaders and concatenates planes, but explicitly acknowledge “76 (13 swap sessions skipped)” because their behavior keys have suffixes.

## 1-b. How are the data split into subjects?

i. Sessions retain `mname`; the final subject list is the sorted unique mouse names among successfully processed sessions, and each session receives its mouse's index.

ii.
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subject_idx.append(mouse_to_idx[result['mname']])
```

iii. The notes report 19 subjects, matching the paper.

## 1-c. How are the data split into sessions?

i. A unique session is the tuple mouse, date, and block. Duplicate index entries are merged and their experiment types accumulated; missing swap behavior causes 13 sessions to be skipped.

ii.
```python
sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if sess_id not in sessions:
    sessions[sess_id] = {..., 'exp_types': []}
sessions[sess_id]['exp_types'].append(exp_type)
```

iii. The agent justified the ID as matching spike filenames, but documented that only 76/89 recordings survived.

## 1-d. How are the data split into trials?

i. Each behavioral trial is sliced from rounded `StartFr` to rounded `GrayFr`, capped at 1,000 frames. Trials shorter than two frames or outside the spike array are dropped.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
end_fr = min(start_fr + n_timepoints, gray_fr)
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The notes describe this as extracting the corridor portion from entry to grey space and mention the 1,000-frame cap.

## 1-e. How are trials filtered based on quality controls?

i. Trials are rejected only if fewer than two rounded slice frames exist or bounds exceed neural data; long trials are truncated at 1,000 rather than filtered as outliers. Sessions with fewer than two valid trials are skipped.

ii.
```python
if avail_frames < 2: return None
if start_fr < 0 or end_fr > spk.shape[1]: return None
end_fr = min(start_fr + n_timepoints, gray_fr)
if len(trial_neural) < 2: return None
```

iii. The notes call the short-trial exclusion and 1,000-frame cap edge-case handling; no data-derived quality threshold is justified.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from per-plane `spks` arrays in each session's neural `.npy`, concatenated over neurons; `iarea` from retinotopy is used only for region labels.

ii.
```python
dat = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
```

iii. The notes identify these as Suite2p deconvolved calcium traces and say no dF/F or further deconvolution is needed.

## 2-b. How is the `neural` data processed?

i. Concatenated traces are sliced by trial and cast to float16; they are neither rebinned nor padded.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
trial_neural.append(neural)
```

iii. Float16 was chosen to reduce the very large output size; existing traces were considered analysis-ready.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. Four visual-area mappings are assigned, while every other neuron is retained as `unassigned`.

ii.
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)
```

iii. The notes state that the reference code performs no explicit neuron-quality filtering and report 483,862 unassigned neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The first neural column is the rounded corridor-entry frame (`StartFr`); data continue to rounded `GrayFr` or 1,000 frames, with variable trial lengths.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The metadata and notes identify trial start/corridor entry as the alignment event and allow variable duration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one time bin at 3.17 Hz, reported as about 315.5 ms; no rebinning is applied.

ii.
```python
FRAME_RATE = 3.17
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. The agent says behavior and neural data already share the imaging-frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, rounded `StartFr`, and the constant 3.17-Hz frame rate.

ii.
```python
sound_fr = beh['SoundFr'][trial_idx]
sound_frame_offset = sound_fr - start_fr
```

iii. The mapping table describes `SoundFr - frame` converted to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each extracted frame, the code computes `(current offset - cue offset) / frame_rate`; this is time since cue, negative before it, despite the name “time_to_sound_cue.”

ii.
```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE
                        for i in range(actual_frames)], dtype=np.float32)
```

iii. A comment explicitly states “negative = before cue, positive = after,” showing the sign was intentional, though it contradicts time-to-cue semantics.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It has one value for each offset of the same rounded trial slice used for neural data, using a fixed-rate synthetic time axis.

ii.
```python
for i in range(actual_frames)
inputs = np.stack([time_to_cue, ...], axis=0)
```

iii. The notes treat all streams as frame-aligned.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It comes from mouse name and the unique `datexp` strings appearing for that mouse in the experiment index.

ii.
```python
if s['mname'] == mname:
    dates.add(s['datexp'])
return sorted(dates).index(datexp)
```

iii. The notes describe it as chronological day index based on `datexp`.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Unique dates are sorted, indexed from zero, and the resulting value is repeated across all frames of a trial.

ii.
```python
training_day = get_training_day(mname, datexp, exp_info)
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. The agent intended to represent chronological training order; the full output reveals gaps after skipped same-date/block sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived only from the integer offset within the extracted trial and the constant frame rate; raw `ft` timestamps are not used.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The mapping plan describes frame minus `StartFr`, in seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Offsets 0 through `actual_frames-1` are divided by 3.17, forcing the first sample to exactly zero.

ii.
```python
np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The agent relied on the nominal constant acquisition rate instead of measured frame timestamps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Its array length and index correspond directly to columns of the same neural slice.

ii.
```python
inputs = np.stack([..., time_since_start, ...], axis=0)
```

iii. The notes consider the common frame offsets sufficient alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `isRew` field.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The mapping table labels `isRew` as the binary source.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is converted to float and repeated at every time point in that trial.

ii.
```python
np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. No further transformation was considered necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` for each trial.

ii.
```python
stim_name = str(wall_names[t])
all_stim_names.add(trial_out['stim_name'])
```

iii. The notes list `WallName` as the categorical source.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All distinct raw wall names are sorted and encoded as 13 separate classes, then repeated across trial frames; crop/swap variants are not collapsed into the four base textures.

ii.
```python
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
out[0, :] = stim_to_idx[trial_out['stim_name']]
```

iii. The notes defend decoder performance with “13 classes,” but provide no scientific justification for departing from broad stimulus categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking comes from `LickFr`, restricted to the trial using `LickTrind`.

ii.
```python
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. The notes describe `LickFr` as a binary-per-frame source.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each selected lick frame is rounded, shifted by rounded trial start, and marks that extracted frame as one; multiple licks in a frame remain binary.

ii.
```python
lf_int = int(np.round(lf))
frame_offset = lf_int - start_fr
if 0 <= frame_offset < actual_frames:
    licking[frame_offset] = 1.0
```

iii. The agent says float frame indices were rounded as edge-case handling.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Rounded lick-frame offsets are placed into an array whose indices match columns in the rounded `StartFr:GrayFr` neural slice.

ii.
```python
licking = np.zeros(actual_frames, dtype=np.float32)
frame_offset = lf_int - start_fr
```

iii. The justification is shared imaging-frame indexing.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos` over the same integer trial slice.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. The notes identify `ft_Pos` and establish 10 VR units per meter.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Raw position is passed to a discretizer with fixed VR-unit edges; no interpolation or smoothing is done.

ii.
```python
pos_bins = discretize_position(trial_data['position'])
```

iii. The stated goal is four equal 1-m spatial bins in the 4-m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `np.digitize` applies edges 10, 20, and 30, then clips results to 0–3, corresponding to 0–1, 1–2, 2–3, and 3–4 m.

ii.
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, N_POSITION_BINS - 1)
```

iii. The notes cite the 0.1-m scale of each VR unit.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position and neural traces use the identical `start_fr:end_fr` slice.

ii.
```python
neural = spk[:, start_fr:end_fr]
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. Shared frame indexing is the alignment rationale.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `ft_RunSpeed` over corridor slices.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. The notes map `ft_RunSpeed` directly to speed bins.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A first pass gathers all corridor speeds across selected sessions, removes NaNs, and computes global 0/25/50/75/100 percentiles. A second pass digitizes each trial using those global thresholds.

ii.
```python
flat_speeds = np.concatenate(all_speeds)
edges = np.percentile(flat_speeds[~np.isnan(flat_speeds)], [0, 25, 50, 75, 100])
```

iii. The agent intended four bins corresponding to 25% of the full data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values are assigned by the three internal global percentile edges using `np.digitize`, then clipped to classes 0–3. Ties, especially zero speeds, make the resulting classes highly unequal.

ii.
```python
bins = np.digitize(speed, bin_edges[1:-1])
bins = np.clip(bins, 0, N_SPEED_BINS - 1)
```

iii. The notes call these quartile bins, although validation reports only 9.9% in the slowest class and 39.4% in the second.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed and neural traces use the same integer frame slice, and discretization preserves its length.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
spd_bins = discretize_speed(trial_data['speed'], speed_bin_edges)
```

iii. The agent relies on the behavior already being sampled per imaging frame.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing behavior skips a session; invalid/short/out-of-neural-bounds trials are dropped; NaN speeds are excluded only when computing edges; sessions with fewer than two valid trials are skipped. Swap-key handling fails, causing 13 sessions to be omitted, and behavior extending beyond imaging is not generally clipped.

ii.
```python
if beh_results is None: return None
if start_fr < 0 or end_fr > spk.shape[1]: return None
flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
```

iii. The notes document these edge cases and the 13 missing swap sessions, but accept the incomplete result after format validation.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the very large spike files, holding/copying trial slices, and saving the 117-GB pickle dominate. The two-pass behavior processing is smaller but still scans all trials twice.

ii.
```python
dat = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
pickle.dump(data, f, protocol=4)
```

iii. The logs report per-session load times and a 117-GB output; float16 was explicitly used to limit size.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-frame time-to-cue/time-since-start construction and per-lick marking could be vectorized. Session/trial loops are structurally appropriate because arrays are ragged and files are separate.

ii.
```python
np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)])
for lf in trial_lick_frs:
    ...
```

iii. The agent supplies no explicit efficiency justification for these loops.

## 12-c. What processing does the code repeat multiple times?

i. Behavioral files are repeatedly loaded for individual sessions and the full dataset is traversed twice for speed collection and conversion. Training-day dates are also rescanned over the entire experiment index for every session.

ii.
```python
for exp_type in exp_types:
    beh_all = np.load(beh_path, allow_pickle=True).item()
for i, sess_id in enumerate(session_ids):
    speeds = process_session(..., collect_speeds=True)
for i, sess_id in enumerate(session_ids):
    result = process_session(...)
```

iii. The first pass is justified as necessary to create global speed thresholds; other repeated loading/scanning is undocumented.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It constructs metadata that is not used in final assembly (`beh_key`, `sound_frame_offset` after input creation, load timing, original trial counts), imports unused SciPy interpolation, and discovers/loads behavior variants before choosing only the first. More materially, it processes and stores unassigned neurons that the reference discards.

ii.
```python
from scipy import interpolate
beh, beh_key = beh_results[0]
'load_time': t_load,
```

iii. These choices are mostly undocumented; retaining all neurons was justified by the agent's interpretation that reference code applied no neuron filter.
