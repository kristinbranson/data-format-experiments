# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads all NWB files directly using `h5py` (not the AllenSDK high-level API). It discovers files via `glob.glob('/app/data/**/*.nwb')`, then filters to active session types. Each NWB file is opened individually and its event detection data, trial intervals, running speed, pupil tracking, and stimulus presentation data are extracted from known HDF5 paths.

ii.
```python
DATA_GLOB='/app/data/**/*.nwb'
ACTIVE={'OPHYS_1_images_A','OPHYS_3_images_A','OPHYS_4_images_B','OPHYS_6_images_B'}

files=[]
for f in sorted(glob.glob(DATA_GLOB,recursive=True)):
    with h5py.File(f,'r') as h:
        if text(h['session_description']) in ACTIVE: files.append(f)
```

iii. The agent observed that using the AllenSDK's `BehaviorOphysExperiment.from_nwb_path()` was extremely slow because it eagerly loads full dF/F arrays. By switching to direct h5py access, the agent could efficiently extract only the needed datasets. The agent filtered to active session types (OPHYS 1, 3, 4, 6) based on the paper's methods, which state analyses used active change-detection sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading `general/subject/subject_id` from each NWB file. Unique subject IDs are collected and sorted across all active-session files.

ii.
```python
subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
subjmap={v:i for i,v in enumerate(subjects)}
```

iii. The `subject_id` field in the NWB is the unique mouse identifier. The agent used this directly from the HDF5 metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as a separate session. There is no multi-plane grouping; each experiment (imaging plane) is its own session in the output.

ii.
```python
for fi,f in enumerate(files):
  with h5py.File(f,'r') as h:
    # ... process entire file as one session
    if len(sn)>=2:
        neural.append(sn); inputs.append(si); outputs.append(so)
```

iii. The agent noted that most NWB files have distinct start times (239 out of 247 unique mouse+start combinations are singletons). Since each NWB contains its own neural population and timestamps, treating each as a session avoids the complexity of multi-plane grouping. The agent's trajectory confirms this decision: "Treat each NWB experiment as a recording session, as it has its own neural population and timestamps."

## 1-d. How are the data split into trials?

i. Trials are defined from the `intervals/trials` table in each NWB. Go and Catch trials are retained; aborted and auto-rewarded trials are excluded. Trials must have a finite `change_time`. Each trial is defined by a fixed window of -3.0 to +4.2 seconds around `change_time`, binned at 100 ms (72 time bins).

ii.
```python
DT=0.100
OFF0,OFF1=-3.0,4.2
NB=int(round((OFF1-OFF0)/DT))

go=np.asarray(tr['go']); catch=np.asarray(tr['catch'])
abort=np.asarray(tr['aborted']); auto=np.asarray(tr['auto_rewarded'])
change=np.asarray(tr['change_time'],dtype=float)
keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
```

iii. The agent analyzed trial timing and found that valid trials consistently have stop_time ~4.23s after change_time, while start_time varies from -3.0 to -8.3s before change. A fixed -3.0 to +4.2s window was chosen to preserve the common portion of every valid trial and ensure consistent dimensions across all trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials with finite change_time, excluding aborted and auto-rewarded trials. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
# ...
if len(sn)>=2:
    neural.append(sn)
```

iii. This follows the instructions which say to include Go and Catch trials but exclude Aborted and Auto-rewarded trials. The minimum 2-trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `event_detection` data in the NWB files, accessed at `processing/ophys/event_detection/data`, with timestamps from `processing/ophys/event_detection/timestamps`.

ii.
```python
ev=h['processing/ophys/event_detection/data']
ot=np.asarray(h['processing/ophys/event_detection/timestamps'],dtype=float)
ncell=ev.shape[1]
```

iii. The agent's trajectory explicitly states: "The methods establish the key neural-processing choice: use detected calcium events (not dF/F), matching the paper." The paper methods describe using "detected calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f."

## 2-b. How is the `neural` data processed?

i. Event detection data is temporally binned into 100 ms bins. For each trial, the event data within the -3.0 to +4.2s change-aligned window is read, assigned to time bins, and averaged within each bin.

ii.
```python
edges=change[j]+OFF0+np.arange(NB+1)*DT
centers=(edges[:-1]+edges[1:])/2
a=np.searchsorted(ot,edges[0]); b=np.searchsorted(ot,edges[-1])
slab=np.asarray(ev[a:b,:],dtype=np.float32)
tt=ot[a:b]; mat=np.zeros((ncell,NB),dtype=np.float32)
bi=np.floor((tt-edges[0])/DT).astype(int)
for k in range(NB):
    z=slab[bi==k]
    if len(z): mat[:,k]=z.mean(axis=0)
```

iii. The agent chose 100 ms bins as a balance between preserving calcium-event dynamics and keeping the dataset tractable in size. The mean aggregation within each bin captures event magnitude.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond what the NWB files already provide. All cells in the event_detection dataset are included.

ii. N/A (all cells from `ev.shape[1]` are used)

iii. The NWB files contain curated cells from the Allen SDK pipeline. The agent noted 42,147 curated cells across all files, indicating the data is already quality-controlled.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the scheduled image change time (`change_time`). A fixed window of -3.0 to +4.2 seconds around `change_time` is used, with bin edges computed relative to `change_time`.

ii.
```python
edges=change[j]+OFF0+np.arange(NB+1)*DT
```

iii. The instructions say "Temporally align based on ophys timestamp." The agent chose to align to `change_time` which is the key behavioral event in the change detection task, and used ophys timestamps for determining bin membership. The metadata records `temporal_alignment_event` as "scheduled image change time (Go change or Catch sham-change)".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to 100 ms time bins (DT=0.100). The native ophys frame rate (~31 Hz, ~32 ms per frame) is rebinned by averaging event magnitudes within each 100 ms bin.

ii.
```python
DT=0.100
# ...
'time_bin_size':DT*1000  # = 100.0 ms
```

iii. The agent stated: "100 ms bins preserve calcium-event and behavior dynamics while making the complete dataset tractable." This rebinning aggregates approximately 3 native frames per bin.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation TimeSeries in the NWB, specifically the `data` (image indices) and `timestamps` arrays under `stimulus/presentation/`.

ii.
```python
preskeys=list(h['stimulus/presentation'].keys())
pg=h['stimulus/presentation/'+preskeys[0]]
stim_t=np.asarray(pg['timestamps'],dtype=float); stim_i=np.asarray(pg['data'])
```

iii. The agent used the stimulus presentation TimeSeries to determine which image is on screen at each time point, rather than using trial-level `initial_image_name`/`change_image_name`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is computed by determining which stimulus presentation is active at each time bin center. A global set of image values is predefined: `['gray'] + ['A_image_0',...,'A_image_7','B_image_0',...,'B_image_7']` (17 total). Gray (index 0) is assigned when no image is on screen or during omissions. Image indices are offset by session-dependent set offset (1 for images_A, 9 for images_B).

ii.
```python
image_values=['gray']+[f'{s}_image_{i}' for s in ('A','B') for i in range(8)]
setoff=1 if 'images_A' in stype else 9
pos=np.searchsorted(stim_t,centers,side='right')-1
img=np.zeros(NB,dtype=np.int16)
ok=(pos>=0)
pp=np.clip(pos,0,len(stim_t)-1)
shown=ok&(centers<stim_t[pp]+.25)&(stim_i[pp]<8)
img[shown]=(setoff+stim_i[pp[shown]].astype(int)).astype(np.int16)
```

iii. The agent accounts for the 250 ms stimulus duration (image shown for 250 ms, then 500 ms gray), omissions (stimulus index >= 8 are omitted), and distinguishes between familiar (A) and novel (B) image sets with separate category indices.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100 ms bin centers as the neural data, using `np.searchsorted` to find which stimulus presentation is active at each bin center.

ii.
```python
pos=np.searchsorted(stim_t,centers,side='right')-1
```

iii. Both neural and stimulus data are aligned to the same temporal grid (100 ms bins centered around change_time), ensuring frame-by-frame correspondence.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `change_time` in the trials table.

ii.
```python
change=np.asarray(tr['change_time'],dtype=float)
ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
```

iii. The change_time marks when the image identity switches, which is the defining event of the change detection task.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary variable set to 1 at the single time bin closest to `change_time`, and 0 elsewhere. This applies to both Go and Catch trials.

ii.
```python
ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
```

iii. The agent represents image change as a single-bin pulse rather than a sustained window. This marks the exact moment of change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1), with no thresholding needed. It is 1 at the single bin closest to change_time, 0 elsewhere.

ii.
```python
ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
```

iii. Binary by definition: change or no change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 100 ms bin grid as neural data. The change bin is identified by finding the bin center closest to `change_time`.

ii. See 4-b above.

iii. Aligned via the same temporal grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` with timestamps from `processing/running/speed/timestamps`.

ii.
```python
rt=np.asarray(h['processing/running/speed/timestamps']); rv=np.asarray(h['processing/running/speed/data'])
```

iii. The `running/speed` dataset is the standard processed running speed from the Allen SDK pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated to the 100 ms bin centers using `np.interp` (after cleaning non-finite values), then discretized into 5 quintile bins using within-session percentile edges.

ii.
```python
run=interp_clean(rt,rv,centers)
rq=quintile_reference(rv)
runbin=np.digitize(run,rq).astype(np.int16)
```

Where `quintile_reference` computes edges at [0.2, 0.4, 0.6, 0.8] quantiles and `interp_clean` filters non-finite values before interpolation.

iii. The agent uses within-session quintiles (computed from the full session's running speed data), ensuring balanced bins within each session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using `np.digitize` with edges at the 20th, 40th, 60th, and 80th percentiles of the session's running speed data.

ii.
```python
def quintile_reference(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: return np.array([-np.inf]*4)
    return np.quantile(x,[.2,.4,.6,.8])
runbin=np.digitize(run,rq).astype(np.int16)
```

iii. Five equal percentile bins ensure roughly balanced class counts for decoding, as specified in the instructions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100 ms bin centers as the neural data.

ii.
```python
run=interp_clean(rt,rv,centers)
```

iii. Both share the same temporal grid, ensuring alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width`, with timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pbase='acquisition/EyeTracking/pupil_tracking'
if pbase in h:
    pt=np.asarray(h['acquisition/EyeTracking/eye_tracking/timestamps'])
    pv=np.asarray(h[pbase+'/width'])
else: pt=np.array([]); pv=np.array([])
```

iii. The agent identified `pupil_tracking/width` as the pupil diameter measure by inspecting the HDF5 structure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width is interpolated to the 100 ms bin centers using `np.interp` (after cleaning non-finite values), then discretized into 5 quintile bins using within-session percentile edges.

ii.
```python
pup=interp_clean(pt,pv,centers)
pq=quintile_reference(pv)
pupbin=np.digitize(pup,pq).astype(np.int16)
```

iii. Same approach as running speed: interpolate then quintile-bin within session.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 bins (0-4) using `np.digitize` with edges at the 20th, 40th, 60th, and 80th percentiles of the session's pupil width data.

ii.
```python
pq=quintile_reference(pv)
pupbin=np.digitize(pup,pq).astype(np.int16)
```

iii. Five equal percentile bins as specified in the instructions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100 ms bin centers as the neural data.

ii.
```python
pup=interp_clean(pt,pv,centers)
```

iii. Same temporal grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
hit=np.asarray(tr['hit']); miss=np.asarray(tr['miss'])
fa=np.asarray(tr['false_alarm']); cr=np.asarray(tr['correct_reject'])
outcome=0 if hit[j] else 1 if miss[j] else 2 if fa[j] else 3
```

iii. These are the standard trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes: hit=0, miss=1, false_alarm=2, correct_reject=3. The value is constant across all time bins within a trial.

ii.
```python
outcome=0 if hit[j] else 1 if miss[j] else 2 if fa[j] else 3
out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
```

iii. The outcome is static per trial but broadcast across time bins for a rectangular output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: If `acquisition/EyeTracking/pupil_tracking` doesn't exist, empty arrays are used, resulting in zero-filled pupil bins.
- **Non-finite values**: The `interp_clean` function filters out non-finite timestamps and values before interpolation; if fewer than 2 valid points remain, zeros are returned.
- **Stimulus omissions**: Image indices >= 8 are treated as omissions and mapped to gray (index 0).
- **Few trials**: Sessions with fewer than 2 valid trials are excluded.

ii.
```python
def interp_clean(t,x,q):
    t=np.asarray(t); x=np.asarray(x,dtype=float)
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()<2: return np.zeros(len(q),dtype=np.float32)
    return np.interp(q,t[good],x[good]).astype(np.float32)

if pbase in h:
    pt=np.asarray(h['acquisition/EyeTracking/eye_tracking/timestamps'])
    pv=np.asarray(h[pbase+'/width'])
else: pt=np.array([]); pv=np.array([])
```

iii. The agent handles missing data gracefully by falling back to zeros and checking for sufficient valid data points before interpolation.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading the event detection data from each NWB file using h5py. Each file is ~270-400 MB and contains large neural data arrays. The agent reads only the needed slab per trial (`ev[a:b,:]`) rather than the entire array.

ii.
```python
a=np.searchsorted(ot,edges[0]); b=np.searchsorted(ot,edges[-1])
slab=np.asarray(ev[a:b,:],dtype=np.float32)
```

iii. Reading from disk is I/O bound. The slab-based reading is more efficient than loading the full array.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin averaging loop inside each trial could potentially be vectorized using `np.add.at` or similar approaches instead of iterating over each bin.

ii.
```python
for k in range(NB):
    z=slab[bi==k]
    if len(z): mat[:,k]=z.mean(axis=0)
```

iii. This loop iterates 72 times per trial (one per time bin). While not a major bottleneck compared to I/O, it could be vectorized.

## 9-c. What processing does the code repeat multiple times?

i. The quintile reference computation (`quintile_reference`) is called once per session for each of running speed and pupil diameter, using the full session's data. This is appropriate and not repeated unnecessarily. However, the full running speed and pupil data arrays are loaded per session even though only trial-relevant portions are needed.

ii.
```python
rt=np.asarray(h['processing/running/speed/timestamps']); rv=np.asarray(h['processing/running/speed/data'])
# ... then only centers are used for interpolation
```

iii. Loading full behavioral arrays is simpler but loads more data than strictly needed.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the full running speed and pupil arrays for quintile computation but only uses interpolated values at trial bin centers. The quintile edges are computed from raw session data, not from the trial-relevant time windows.

ii.
```python
rq=quintile_reference(rv)  # uses full session data
pq=quintile_reference(pv)  # uses full session data
```

iii. Using full-session data for quintile computation is a deliberate choice (more representative of the session's behavioral state), not truly unnecessary.
