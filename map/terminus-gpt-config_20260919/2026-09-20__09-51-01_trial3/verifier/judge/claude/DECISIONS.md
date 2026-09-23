# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `/app/data/sub-*/` using h5py (not pynwb). It first performs a pre-filtering pass (`eligible_files()`) that opens every NWB to check whether `units/classification` has a string dtype, skipping files where it doesn't. Then each eligible file is processed in `process_session()`.

ii. Pre-filtering:
```python
def eligible_files():
    kept=[]; skipped=[]
    for p in sorted(DATA_ROOT.rglob('*.nwb')):
        with h5py.File(p, 'r') as f:
            d=f['units/classification']
            if d.dtype.kind in 'OSU': kept.append(p)
            else: skipped.append((p.name, 'missing classifier labels'))
    return kept, skipped
```

Loading one session:
```python
with h5py.File(path,'r') as f:
    subject=(f['general/subject/subject_id'].asstr()[()]
             if f['general/subject/subject_id'].dtype.kind in 'OSU'
             else path.parent.name.replace('sub-',''))
    trials=f['intervals/trials']; nt=len(trials['id'])
    go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The AI justified using h5py as a direct NWB/HDF5 reader. The two-pass approach (eligible_files then process_session) means each file is opened twice. The AI documented that one session has NaN classification and is excluded in the pre-filter.

## 1-b. How are the data split into subjects?

i. Subject ID is read from `general/subject/subject_id` in each NWB file. If the dtype is not string, the subject is inferred from the parent directory name. Unique sorted subjects are assembled at the end.

ii.
```python
subject=(f['general/subject/subject_id'].asstr()[()]
         if f['general/subject/subject_id'].dtype.kind in 'OSU'
         else path.parent.name.replace('sub-',''))
```

```python
subjects=sorted(set(x['subject'] for x in sessions)); smap={x:i for i,x in enumerate(subjects)}
```

iii. The AI noted 28 subjects matching the dandiset. The fallback to directory name is a robustness measure not triggered in practice.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The eligible files are sorted by path, and each is processed independently. Session ID is taken from the file stem (path.stem).

ii.
```python
for p in sorted(DATA_ROOT.rglob('*.nwb')):
```

```python
session_info={'session_id':path.stem, ...}
```

iii. The AI documented 173 sessions after excluding the one with missing classifier labels. The session ID uses the filename stem rather than `nwb.identifier`.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials` in the NWB file. The count is taken from `len(trials['id'])` and verified against the number of go cue events.

ii.
```python
trials=f['intervals/trials']; nt=len(trials['id'])
go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
if len(go)!=nt: raise ValueError(f'{path.name}: {len(go)} go events != {nt} trials')
```

iii. The AI verified that go cue count equals trial count in every session.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial filters:
1. **Neural recording coverage**: Maps `obs_intervals` to trial starts to find electrophysiologically recorded trials.
2. **Video coverage**: Requires the full trial window (go-2.5 to go+1.5) to fall within the video timestamp range.
3. **All-zero neural exclusion**: After spike binning, trials where all neurons have zero spikes everywhere are excluded.
4. **Notably absent**: No `free_water` filter is applied.

ii.
```python
recorded_mask=np.zeros(nt,dtype=bool); recorded_mask[recorded_idx]=True
# ...
video_mask=(go+OFF_START>=video_t[0]) & (go+OFF_END<=video_t[-1])
keep_trial=recorded_mask & video_mask
# ...
neural_nonzero=np.any(neural_cube!=0,axis=(1,2))
excluded_all_zero=int(np.sum(~neural_nonzero))
if excluded_all_zero:
    kept=kept[neural_nonzero]
```

iii. The AI justified retaining all trial types (including free_water, early lick, ignore) because they are required decoder inputs/outputs. Video coverage filtering was added to ensure tongue data is available. The all-zero neural filter was added after discovering trials with no spike data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `units/spike_times_index` (the ragged spike time arrays), using only units that pass the neuron quality filter.

ii.
```python
spikes=f['units/spike_times']; ends=f['units/spike_times_index'][:]
# ...
st=np.asarray(unit_spike_slice(spikes, ends, int(u)), dtype=np.float64)
```

iii. Spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins aligned to go cue (-2.5s to +1.5s). For each unit, `np.searchsorted` with `side='left'` is used over all trial bin edges to get cumulative counts, then `np.diff` gives per-bin counts, divided by bin width (0.05s) for Hz.

ii.
```python
edges=go_kept[:,None]+REL_EDGES[None,:]
flat=edges.ravel()
# ...
cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
counts[j]=np.diff(cumulative, axis=1)/BIN_S
```

```python
return counts.transpose(1,0,2)  # trials, neurons, time
```

iii. Same half-open bin counting convention as the reference code. The AI documented matching the `sliding_histogram` logic from the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neuron filters:
1. `classification == 'good'` (the spike-sorting QC classifier)
2. `is_good_trials.all(axis=1)` — the unit must be marked valid on every source trial ("stable" units)

This yields 68,888 units versus the reference's 69,453 (classification-only).

ii.
```python
classification=decode_strings(f['units/classification'])
stable=good_trials.all(axis=1)
unit_ids=np.flatnonzero((classification=='good') & stable)
```

iii. The AI justified the stability filter by noting that the target format requires a fixed neuron matrix shape per session, and using unstable units would mean silently treating invalid periods as zero activity. The CONVERSION_NOTES document this removes 565 additional units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as go cue time plus relative offsets (-2.5 to +1.5 s). Since all NWB timestamps share one global clock, no resampling or offset correction is needed.

ii.
```python
edges=go_kept[:,None]+REL_EDGES[None,:]
flat=edges.ravel()
```

iii. Alignment to go cue onset as required by the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins per trial, spanning -2.5 to +1.5 s relative to go cue. No rebinning — spikes are binned directly from raw spike times.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
N_BINS = int(round((OFF_END-OFF_START)/BIN_S))
REL_EDGES = OFF_START + np.arange(N_BINS+1, dtype=np.float64)*BIN_S
```

iii. Matches the task requirement of 50ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset events) and go cue times. The last sample onset before each trial's go cue is taken as the tone for that trial.

ii.
```python
sample=f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
pos=np.searchsorted(sample, go, side='left')-1
tone=sample[pos]
```

iii. The AI noted that early licks can replay the sample epoch, so a trial can have multiple tone onsets; the last one before the go cue is the causally relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute time of each bin center is computed, then the tone onset time is subtracted to give elapsed time from tone onset in seconds.

ii.
```python
time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
```

Where `centers = go_k[:,None] + REL_CENTERS[None,:]` are absolute bin center times.

iii. Straightforward time difference computation.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for neural data are used for computing time from tone onset. Since `centers = go + REL_CENTERS`, each bin center corresponds to the same time interval as the corresponding neural bin.

ii.
```python
go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]
# ...
time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
```

iii. Aligned by construction — same temporal grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps` — the actual laser on/off event timestamps.

ii.
```python
on=ev['photostim_start_times/timestamps'][:]
off=ev['photostim_stop_times/timestamps'][:]
```

iii. The AI chose to use the event timestamps directly rather than the trial table string fields. This avoids parsing string `'N/A'` values.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series where a bin is 1 if its center falls within any `[stim_on, stim_off)` interval, 0 otherwise. The AI loops over all stimulation intervals and ORs them together.

ii.
```python
result=np.zeros(absolute_centers.shape, dtype=bool)
for a,b in zip(on,off): result |= ((absolute_centers>=a)&(absolute_centers<b))
return result.astype(np.float32)
```

iii. Uses absolute bin center times, consistent with how the reference converts onset/offset to per-bin flags.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The absolute bin centers (go + REL_CENTERS) used for photostimulation marking are the same grid used for neural binning, ensuring alignment.

ii.
```python
centers=go_k[:,None]+REL_CENTERS[None,:]
# ...
stim=photostim_series(f,centers)
```

iii. Aligned by construction.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trials/outcome` and `trials/trial_instruction`. Choice is not stored directly, so it is inferred: ignore = no lick (2), hit = instruction side (left=0, right=1), miss = opposite of instruction side.

ii.
```python
choice=np.where(outcome=='ignore',2,np.where(outcome=='hit',
                 np.where(instruction=='left',0,1),
                 np.where(instruction=='left',1,0))).astype(np.int8)
```

iii. The AI verified that `outcome=='ignore'` means no lick and derived choice from the instruction/outcome combination.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice values (0=left, 1=right, 2=no lick) are expanded across all 80 time bins as a per-trial constant.

ii.
```python
output_cube[:,0,:]=choice[:,None]
```

iii. Per-trial value repeated across time bins, matching the format requirement.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials/outcome` which contains strings 'ignore', 'miss', 'hit'.

ii.
```python
outcome=decode_strings(trials['outcome'])[kept]
outcome_code=np.array([{'ignore':0,'miss':1,'hit':2}[x] for x in outcome],dtype=np.int8)
```

iii. Direct mapping from the existing trial table column.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String values are mapped to integers: ignore=0, miss=1, hit=2. Expanded across all 80 time bins.

ii.
```python
output_cube[:,1,:]=outcome_code[:,None]
```

iii. Same encoding as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials/early_lick` which holds 'no early' and 'early'.

ii.
```python
early=decode_strings(trials['early_lick'])[kept]
early_code=np.array([{'no early':0,'early':1}[x] for x in early],dtype=np.int8)
```

iii. Direct mapping from existing trial table column.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no) and 1 (yes), expanded across 80 time bins.

ii.
```python
output_cube[:,2,:]=early_code[:,None]
```

iii. Same encoding as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `(n_frames, 3)` data columns (tongue_x, tongue_y, tongue_likelihood) with timestamps.

ii.
```python
tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
tongue=np.asarray(tg['data'][:],dtype=np.float64)
video_t=tg['timestamps'][:]  # loaded earlier
```

iii. Same source variable as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a **nearest-frame** approach rather than bin-averaging:
1. Frames with likelihood < 0.9 are marked not visible.
2. Session-wide 40th and 60th percentiles are computed from **raw visible frame y-values** (not bin means).
3. For each bin center, the nearest video frame is found.
4. If that frame is visible, its y-value is discretized: <q40 → 0, q40-q60 inclusive → 1, >q60 → 2.
5. If not visible → 3.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
# ...
visible_session=tongue[:,2]>=VISIBILITY_THRESHOLD
q40,q60=np.percentile(tongue[visible_session,1],[40,60])
vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
y=tongue[vi,1]; likelihood=tongue[vi,2]
tongue_code=np.full(y.shape,3,dtype=np.int8)
vis=likelihood>=VISIBILITY_THRESHOLD
tongue_code[vis & (y<q40)]=0
tongue_code[vis & (y>=q40) & (y<=q60)]=1
tongue_code[vis & (y>q60)]=2
```

iii. The AI justified 0.9 as a "standard conservative DLC confidence threshold" and nearest-frame sampling as sufficient at ~300 Hz. However, this differs from the reference in three ways: (a) threshold 0.9 vs 0.5, (b) nearest frame vs averaging within bins, (c) percentiles on raw frames vs bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses strict inequalities: class 0 is `y < q40`, class 1 is `q40 <= y <= q60` (inclusive on both sides), class 2 is `y > q60`. The reference uses `np.digitize` which gives `[, )` half-open intervals.

ii.
```python
tongue_code[vis & (y<q40)]=0
tongue_code[vis & (y>=q40) & (y<=q60)]=1
tongue_code[vis & (y>q60)]=2
```

iii. The boundary conditions differ slightly from the reference's `np.digitize(m[ok], edges)` which produces 0 for `y < q40`, 1 for `q40 <= y < q60`, 2 for `y >= q60`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI finds the nearest video frame to each absolute bin center using `nearest_indices`, which is a nearest-neighbor lookup.

ii.
```python
vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
```

iii. At ~300 Hz video, the maximum mismatch is ~1.7 ms, which the AI noted is acceptable. However, the reference averages all frames within each 50ms bin rather than using a single nearest frame.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases handled:
- **Session without classifier labels**: Pre-filtered in `eligible_files()` by checking classification dtype.
- **Trials without neural recording**: Mapped via obs_intervals; unrecorded trials excluded.
- **Trials without video coverage**: Excluded via video_mask requiring the full window within video timestamps.
- **All-zero neural trials**: Excluded after spike binning.
- **Unknown brain regions**: Empty or NaN `anno_name` values are replaced with `'Unknown'`.

ii.
```python
if d.dtype.kind in 'OSU': kept.append(p)
else: skipped.append((p.name, 'missing classifier labels'))
```
```python
anno=[x if x and x.lower() not in ('nan','none') else 'Unknown' for x in anno]
```

iii. The AI documented all edge cases thoroughly in CONVERSION_NOTES.md. The 'Unknown' region label fallback is an addition not in the reference.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reported processing time of ~0.86-1.29 s per session for the sample, and total full conversion of 154.6 s for 173 sessions. The eligible_files pre-scan adds an extra pass over all files. Per-session, spike binning (searchsorted per unit) and loading the video/spike arrays dominate.

ii. N/A

iii. The AI provided timing estimates in CONVERSION_NOTES.md and confirmed full conversion was well within the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: (1) per-unit spike binning loop (inherent due to ragged spike arrays), and (2) the photostim interval loop that ORs each interval's mask.

ii.
```python
for j,u in enumerate(unit_ids):
    st=np.asarray(unit_spike_slice(spikes, ends, int(u)), dtype=np.float64)
    cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
    counts[j]=np.diff(cumulative, axis=1)/BIN_S
```
```python
for a,b in zip(on,off): result |= ((absolute_centers>=a)&(absolute_centers<b))
```

iii. The per-unit loop is vectorized over trials (same as reference). The photostim loop is over intervals, not trials.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened twice: once in `eligible_files()` to check classifier dtype, and once in `process_session()` for actual processing. This doubles the file I/O.

ii.
```python
def eligible_files():
    for p in sorted(DATA_ROOT.rglob('*.nwb')):
        with h5py.File(p, 'r') as f:
            d=f['units/classification']
# ...later...
def process_session(path):
    with h5py.File(path,'r') as f:
```

iii. The reference opens each file only once and handles the missing-classifier case inline.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `eligible_files` pre-scan is somewhat redundant since the same check could be done at session processing time. The AI also computes `neural_window_mask` (checking if the full window fits within obs_intervals) but doesn't use it for the final trial filtering — it only uses `recorded_mask & video_mask`.

ii.
```python
neural_window_mask=np.zeros(nt,dtype=bool)
neural_window_mask[recorded_idx]=((go[recorded_idx]+OFF_START>=ref_intervals[:,0]) &
                                  (go[recorded_idx]+OFF_END<=ref_intervals[:,1]))
```

iii. `neural_window_mask` is computed but only used in session_info metadata, not for actual trial selection. The `next_starts` check is also computed but is a validation rather than filtering step.
