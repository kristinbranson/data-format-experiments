# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing every `*.nwb` file under `/app/data/sub-*`, then opens each file with `pynwb.NWBHDF5IO`. Inside each file it reads the NWB root object, then uses `n.trials`, `n.units`, and the acquisition time-series groups for behavioral events and tongue tracking.

ii.
```python
files=sorted(glob.glob(os.path.join(ROOT,'sub-*','*.nwb')))
```

```python
with NWBHDF5IO(path,'r',load_namespaces=True) as io:
    n=io.read(); tr=n.trials; u=n.units
```

```python
ev=n.acquisition['BehavioralEvents'].time_series
tongue=n.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
```

iii. In the trajectory, the AI first inspected the data layout and concluded the release consisted of one NWB file per session. It explicitly decided to use PyNWB as the canonical loader for the curated NWB release and to process all sorted files in one pass.

## 1-b. How are the data split into subjects?

i. Subjects are split by `n.subject.subject_id` from each NWB file. The output `subjects` list is built incrementally in first-seen order, and `subject_idx` stores the index of each session's subject in that list.

ii.
```python
subject=s(n.subject.subject_id)
```

```python
if sub not in data['subjects']: data['subjects'].append(sub)
data['subject_idx'].append(data['subjects'].index(sub))
```

iii. The trajectory states that the NWB subject field was taken as the canonical subject identifier, and that this numeric subject id was sufficient without additional regrouping logic.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. The output session order follows the sorted file list. Session metadata is tracked by file basename inside `metadata['session_info']`.

ii.
```python
files=sorted(glob.glob(os.path.join(ROOT,'sub-*','*.nwb')))
```

```python
info={'file':os.path.basename(path),'subject':subject,'n_trials':ntr,
      'n_good_units':len(keep),'tongue_y_percentiles':[float(q40),float(q60)],
      'dlc_visibility_threshold':DLC_VISIBLE}
```

```python
data['metadata']['session_info'].append(info)
```

iii. In the trajectory, the AI repeatedly states that one NWB file corresponds to one session in the release, so no extra within-file session splitting is needed.

## 1-d. How are the data split into trials?

i. Trials are split using the NWB trials table directly. The code sets `ntr = len(tr)`, reads one go-cue timestamp per trial, checks that the go-cue count matches the trial count, and then iterates over trial index `j` to build per-trial neural, input, and output arrays.

ii.
```python
n=io.read(); tr=n.trials; u=n.units
subject=s(n.subject.subject_id)
ntr=len(tr)
```

```python
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
if len(go)!=ntr:
    raise RuntimeError(f'{path}: {len(go)} go events != {ntr} trials')
```

```python
for j,g in enumerate(go):
    fr=frcube[:,j,:].copy()
    ...
    neural.append(fr)
    inputs.append(np.vstack((tone_elapsed,photo)).astype(np.float32))
    outputs.append(out)
```

iii. The trajectory explains that the trials table is the behavioral trial structure and that each trial has a unique go cue, so trial rows were used as the per-trial split.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not filter individual trials by `obs_intervals` or `free_water`. It assumes the release is already behaviorally curated and keeps all trials for sessions that have at least one classifier-good unit. The only exclusion is dropping whole sessions with zero units satisfying `classification == 'good'`.

ii.
```python
cls=np.asarray([s(x).lower() for x in u['classification'][:]])
keep=np.flatnonzero(cls=='good')
if not len(keep):
    return None
```

```python
for z,path in enumerate(files,1):
    result=one_file(path)
    if result is None:
        print(f'[{z}/{len(files)}] SKIP {os.path.basename(path)}: no classifier-good units',flush=True)
        continue
```

iii. The trajectory explicitly says the sessions are "already behaviorally curated" and that the one dropped file should be excluded because it has no classifier-good units. It never adds the reference trial-level filtering; instead it validates the dataset even though the final pickle retains 94,370 trials, including 3,511 all-zero neural trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for units whose `classification` is `'good'`. Go-cue timestamps are used to construct the per-trial bin edges.

ii.
```python
cls=np.asarray([s(x).lower() for x in u['classification'][:]])
keep=np.flatnonzero(cls=='good')
```

```python
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
```

```python
spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
all_edges=go[:,None]+EDGES[None,:]
```

iii. The trajectory says the NWB files provide synchronized spike times on the master session clock, so the agent decided to derive firing rates directly from `spike_times` and align them with `go_start_times`.

## 2-b. How is the `neural` data processed?

i. For each kept unit, spike times are binned into 80 non-overlapping 50 ms bins spanning `[-2.5, 1.5)` seconds around each go cue. The code uses `np.searchsorted` over all trial edges for a unit, differences adjacent counts, and divides by `DT` to convert counts to firing rate.

ii.
```python
DT=.05; OFF0=-2.5; OFF1=1.5
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
all_edges=go[:,None]+EDGES[None,:]
frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. The trajectory first used `np.histogram` and then explicitly optimized that into a vectorized `searchsorted` implementation, arguing that it preserved the same half-open bin counts while making the conversion much faster.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The neural QC filter is `units['classification'] == 'good'`. Sessions with no such units are dropped entirely.

ii.
```python
cls=np.asarray([s(x).lower() for x in u['classification'][:]])
keep=np.flatnonzero(cls=='good')
if not len(keep):
    return None
```

iii. The trajectory says the AI inspected both `unit_quality` and `classification` and decided the classifier label matched the newer QC described in the supplied methods and white paper, so only classifier-good units were retained.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go-cue onset. The code adds the fixed relative edges `EDGES` to each trial's absolute go time, bins spikes against those absolute edges, and stores the resulting per-trial unit-by-time matrix.

ii.
```python
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
all_edges=go[:,None]+EDGES[None,:]
```

```python
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
for j,g in enumerate(go):
    fr=frcube[:,j,:].copy()
```

iii. The trajectory notes that spikes and behavioral streams share the same NWB master clock, so the chosen alignment was simply to center the extraction window on each trial's go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins. The code creates 80 non-overlapping bins covering 4.0 s total from -2.5 s to +1.5 s relative to go cue. No additional smoothing or rebinning is applied.

ii.
```python
DT=.05; OFF0=-2.5; OFF1=1.5
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
'time_bin_size':50.0,'temporal_alignment_event':'go cue onset',
'off_start':-2.5,'off_end':1.5,'n_time_bins':80,
```

iii. The trajectory explicitly says the requested decoder format naturally implies 80 non-overlapping 50 ms bins around the go cue, and that the code should use exactly that grid.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The time-from-tone input is derived from `BehavioralEvents/sample_start_times` together with the per-trial `go_start_times`. For each trial the code chooses the last sample-start event before the go cue.

ii.
```python
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=float)
```

```python
sample_for_trial=np.empty(ntr,float)
for j,g in enumerate(go):
    q=sample[sample < g+1e-9]
    sample_for_trial[j]=q[-1] if len(q) else np.nan
```

iii. The trajectory says extra sample events can occur because the sample epoch is replayed after an early lick, so the agent decided the correct tone for a trial is the last sample onset preceding that trial's go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the code computes the absolute time of each decoder bin center as `go + CENTERS`, then subtracts the selected sample onset for that trial. This produces a continuous time-varying signal in seconds since tone onset.

ii.
```python
times=g+CENTERS
tone_elapsed=(times-sample_for_trial[j]).astype(np.float32)
```

iii. The trajectory frames this variable as "time from sample/tone onset at each decoder bin center" so it is built directly on the same 80-bin grid as the neural data.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone signal is evaluated at exactly the same 80 bin centers used for the go-aligned neural data.

ii.
```python
CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
fr=frcube[:,j,:].copy()
times=g+CENTERS
tone_elapsed=(times-sample_for_trial[j]).astype(np.float32)
```

iii. The trajectory says all streams share the NWB master clock and that the decoder inputs should live on the same go-aligned bin grid as the firing rates.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the session-wide behavioral event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, not from the per-trial `photostim_onset` and `photostim_duration` columns in the trials table.

ii.
```python
pstart=np.asarray(ev['photostim_start_times'].timestamps[:],float)
pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],float)
```

iii. In the trajectory, the AI notes that the NWB acquisition contains explicit photostimulation start/stop events on the master clock. It chose those event streams as the direct source for time-resolved photostimulation state.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code creates a length-80 binary vector and sets bins to `1` when the bin center falls inside any photostimulation interval `[start, stop)`. Trials or bins not covered by any interval stay `0`.

ii.
```python
photo=np.zeros(len(CENTERS),dtype=np.float32)
# State at bin centers for every stimulation interval.
for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The trajectory describes this as converting stimulation into a time-varying on/off state at the decoder bin centers rather than a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation input is aligned on the same absolute bin-center timestamps used for the neural data: `times = go + CENTERS`.

ii.
```python
times=g+CENTERS
for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The trajectory's justification is that all streams use the NWB master clock, so comparing photostim event times to the go-aligned bin centers is sufficient for alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a raw choice column. It is derived from trial-table `trial_instruction` and `outcome`.

ii.
```python
instr=np.asarray([s(x).lower() for x in tr['trial_instruction'][:]])
outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
```

```python
if outcome[j]=='ignore': choice=2
elif outcome[j]=='hit': choice=0 if instr[j]=='left' else 1
elif outcome[j]=='miss': choice=1 if instr[j]=='left' else 0
```

iii. The trajectory explicitly states that actual choice is not stored directly and should be inferred from instructed side plus trial outcome, with `ignore` meaning no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as `0=left`, `1=right`, `2=no lick`, and then repeated across all 80 bins within that trial.

ii.
```python
if outcome[j]=='ignore': choice=2
elif outcome[j]=='hit': choice=0 if instr[j]=='left' else 1
elif outcome[j]=='miss': choice=1 if instr[j]=='left' else 0
```

```python
out=np.vstack((np.full(len(CENTERS),choice,np.int8),
               np.full(len(CENTERS),omap[outcome[j]],np.int8),
               np.full(len(CENTERS),ecat,np.int8),ycat))
```

iii. The trajectory says the decoder expects categorical outputs on the trial grid, so per-trial labels such as choice are broadcast across time to share the same `(n_output, n_timepoints)` layout as tongue position.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` column.

ii.
```python
outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
```

iii. The trajectory treats `outcome` as already present in the NWB trials table with the required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are mapped to integer codes with `ignore=0`, `miss=1`, `hit=2`, then repeated across all 80 bins for that trial.

ii.
```python
omap={'ignore':0,'miss':1,'hit':2}
```

```python
out=np.vstack((np.full(len(CENTERS),choice,np.int8),
               np.full(len(CENTERS),omap[outcome[j]],np.int8),
               np.full(len(CENTERS),ecat,np.int8),ycat))
```

iii. The trajectory says outcome is a categorical per-trial variable and therefore is broadcast across the time axis.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trials-table `early_lick` column.

ii.
```python
early=np.asarray([s(x).lower() for x in tr['early_lick'][:]])
```

iii. The trajectory identifies `early_lick` as an explicit per-trial label in the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code converts `early_lick` into a binary category: `1` if the string starts with `'early'`, otherwise `0`. This value is repeated across all 80 bins.

ii.
```python
ecat=1 if early[j].startswith('early') else 0
```

```python
out=np.vstack((np.full(len(CENTERS),choice,np.int8),
               np.full(len(CENTERS),omap[outcome[j]],np.int8),
               np.full(len(CENTERS),ecat,np.int8),ycat))
```

iii. The trajectory says early lick is a per-trial output, so it is encoded categorically and broadcast over the common time axis.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: the code uses `timestamps`, the second data column as tongue `y`, and the third data column as a visibility/confidence value.

ii.
```python
tongue=n.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tt=np.asarray(tongue.timestamps[:],float)
td=np.asarray(tongue.data[:],dtype=np.float32)
```

```python
visible=np.isfinite(td[:,1]) & np.isfinite(td[:,2]) & (td[:,2]>=DLC_VISIBLE)
```

iii. The trajectory says the NWB acquisition contains synchronized tongue tracking with three columns interpreted as x, y, and DeepLabCut confidence, and it uses that as the sole tongue source.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI marks frames as visible when confidence is at least `0.9`, then computes the session's 40th and 60th percentiles from all visible raw `y` samples. For each decoder time bin, it chooses the nearest camera frame to the bin center and assigns a category from that single frame's `y` value, or class `3` if the frame is not visible.

ii.
```python
DLC_VISIBLE=.9
...
visible=np.isfinite(td[:,1]) & np.isfinite(td[:,2]) & (td[:,2]>=DLC_VISIBLE)
if visible.any(): q40,q60=np.percentile(td[visible,1],[40,60])
else: q40,q60=np.nan,np.nan
```

```python
ix=np.searchsorted(tt,times); ix=np.clip(ix,1,len(tt)-1)
ix-=((times-tt[ix-1]) <= (tt[ix]-times))
vis=(td[ix,2]>=DLC_VISIBLE)&np.isfinite(td[ix,1])
ycat=np.full(len(CENTERS),3,dtype=np.int8)
ycat[vis & (td[ix,1]<q40)]=0
ycat[vis & (td[ix,1]>=q40) & (td[ix,1]<=q60)]=1
ycat[vis & (td[ix,1]>q60)]=2
```

iii. The trajectory explicitly justifies two non-reference choices: using a "conventional" `0.9` DeepLabCut threshold for visibility and assigning tongue state from the nearest video frame at each bin center rather than averaging frames over the 50 ms bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code uses session-wide percentiles of visible raw `tongue_y` values: `< q40` maps to class `0`, `q40 <= y <= q60` to class `1`, `> q60` to class `2`, and non-visible frames to class `3`.

ii.
```python
if visible.any(): q40,q60=np.percentile(td[visible,1],[40,60])
else: q40,q60=np.nan,np.nan
```

```python
ycat=np.full(len(CENTERS),3,dtype=np.int8)
ycat[vis & (td[ix,1]<q40)]=0
ycat[vis & (td[ix,1]>=q40) & (td[ix,1]<=q60)]=1
ycat[vis & (td[ix,1]>q60)]=2
```

iii. The trajectory says it used session-wide 40th/60th percentile thresholds from visible samples because that matched its reading of the instructions, while treating low-confidence frames as "not visible."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is aligned by sampling the nearest camera frame at each of the same go-aligned bin centers used for neural data. It does not average all video frames inside each 50 ms bin.

ii.
```python
times=g+CENTERS
```

```python
ix=np.searchsorted(tt,times); ix=np.clip(ix,1,len(tt)-1)
ix-=((times-tt[ix-1]) <= (tt[ix]-times))
vis=(td[ix,2]>=DLC_VISIBLE)&np.isfinite(td[ix,1])
```

iii. The trajectory justifies this by saying there is no need to extrapolate beyond the video stream and that nearest-frame sampling is a practical way to synchronize tongue labels to the neural bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles one missing-data case by dropping sessions with no classifier-good units. Missing or low-confidence tongue observations are assigned to the `'not visible'` class. It does not remove trials lacking spike observations, so such trials remain in the output as all-zero neural matrices.

ii.
```python
keep=np.flatnonzero(cls=='good')
if not len(keep):
    return None
```

```python
visible=np.isfinite(td[:,1]) & np.isfinite(td[:,2]) & (td[:,2]>=DLC_VISIBLE)
...
vis=(td[ix,2]>=DLC_VISIBLE)&np.isfinite(td[ix,1])
ycat=np.full(len(CENTERS),3,dtype=np.int8)
```

```python
sample_for_trial[j]=q[-1] if len(q) else np.nan
```

iii. The trajectory explicitly mentions dropping the one session whose classifier labels are all missing, and treating low-confidence tongue frames as not visible. It does not add the reference trial-level missing-spike filtering, and the final output retains 3,511 all-zero neural trials.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies full-session NWB reading, dense tongue/video loading, spike-time materialization, and neural binning as the dominant costs. It also notes that serializing the final `converted_data.pkl` is substantial because the file is about 12.31 GB.

ii.
```python
with NWBHDF5IO(path,'r',load_namespaces=True) as io:
    n=io.read(); tr=n.trials; u=n.units
...
tt=np.asarray(tongue.timestamps[:],float)
td=np.asarray(tongue.data[:],dtype=np.float32)
spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
```

```python
frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. The trajectory explicitly calls out the initial per-unit/per-trial histogram loop as too slow, then says the optimized run is dominated by reading large NWB arrays and by per-unit spike binning, with the final pickle size also called out after completion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. In the trajectory, the AI itself identified the original nested unit-by-trial histogram loop as the main vectorization target and replaced it with one `searchsorted` per unit over all trials at once. In the final code, remaining vectorizable loops include the per-trial search for the last sample event and the per-trial scan over all photostimulation intervals.

ii.
```python
sample_for_trial=np.empty(ntr,float)
for j,g in enumerate(go):
    q=sample[sample < g+1e-9]
    sample_for_trial[j]=q[-1] if len(q) else np.nan
```

```python
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

```python
for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The trajectory explicitly says the first version was too slow because it histogrammed every unit on every trial. It then patches that loop away, but leaves the smaller per-trial loops in place.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans the full `sample` event vector once per trial to find the last sample event before `go`, and it rescans every photostimulation interval for every trial to build the binary photostim trace. It also recreates the outcome-code dictionary `omap` inside each trial loop.

ii.
```python
for j,g in enumerate(go):
    q=sample[sample < g+1e-9]
    sample_for_trial[j]=q[-1] if len(q) else np.nan
```

```python
for j,g in enumerate(go):
    ...
    photo=np.zeros(len(CENTERS),dtype=np.float32)
    for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

```python
omap={'ignore':0,'miss':1,'hit':2}
```

iii. The trajectory focuses on one repeated computation, the original per-unit/per-trial spike histogramming, and then removes it. The final script still repeats smaller full-vector scans inside the per-trial loop.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does not compute a large derived data product and then discard it from the saved dataset. Its temporary arrays (`frcube`, `times`, `photo`, `ycat`) are intermediate values used to build outputs, and the per-session `info` dictionary is retained in metadata.

ii.
```python
frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
...
info={'file':os.path.basename(path),'subject':subject,'n_trials':ntr,
      'n_good_units':len(keep),'tongue_y_percentiles':[float(q40),float(q60)],
      'dlc_visibility_threshold':DLC_VISIBLE}
```

```python
data['metadata']['session_info'].append(info)
```

iii. The trajectory does not describe any knowingly discarded computation. Its optimization effort was aimed at speeding up computations that still feed the saved output.
