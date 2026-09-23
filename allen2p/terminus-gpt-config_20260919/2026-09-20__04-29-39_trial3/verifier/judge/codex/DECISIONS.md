# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI code loads data by globbing all local NWB files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, excluding three experiment IDs listed in `NO_PUPIL`, reading per-experiment metadata from `ophys_experiment_table.csv`, and then opening each NWB directly with `h5py`. It does not use the AllenSDK cache or experiment table discovery path from the human reference.

ii. 
```python
paths=sorted(NWB_DIR.glob('*.nwb'),key=exp_id); paths=[p for p in paths if exp_id(p) not in NO_PUPIL]
...
table=pd.read_csv(META_DIR/'ophys_experiment_table.csv').set_index('ophys_experiment_id')
...
for si,p in enumerate(paths):
    eid=exp_id(p); row=table.loc[eid]
    n,i,o,info=process_experiment(p,row,args.show_processing and si<2)
```

```python
with h5py.File(path,'r') as f:
    tr=f['intervals/trials']
    nts=np.asarray(f['processing/ophys/dff/traces/timestamps'][:],float)
    nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)
    rts=np.asarray(f['processing/running/speed/timestamps'][:],float)
    rvs=np.asarray(f['processing/running/speed/data'][:],float)
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI says it used “Direct HDF5 reads of released NWB arrays and metadata CSV joins, avoiding network/cache dependencies.” In Step 5 it also justifies working from the local released subset rather than the full AllenSDK-backed project listing.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values from the retained experiment metadata rows. They are converted to strings, sorted, and later used to build `subject_idx`.

ii. 
```python
eid=exp_id(p); row=table.loc[eid]
...
regions.append(str(row.targeted_structure)); mice.append(str(row.mouse_id))
subjects=sorted(set(mice))
...
'subject_idx':np.asarray([subjects.index(x) for x in mice],dtype=np.int64),
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI explicitly maps metadata `mouse_id` to `subjects` and `subject_idx`, describing them as “Unique sorted string IDs and per-experiment index.”

## 1-c. How are the data split into sessions?

i. Each NWB ophys experiment file is treated as one session. The code does not group multiple experiments that share an `ophys_session_id`; instead, it emits one output session per NWB experiment / imaging plane.

ii. 
```python
for si,p in enumerate(paths):
    eid=exp_id(p); row=table.loc[eid]
    n,i,o,info=process_experiment(p,row,args.show_processing and si<2)
    neural.append(n); inputs.append(i); outputs.append(o); infos.append(info); regions.append(str(row.targeted_structure)); mice.append(str(row.mouse_id))
```

```python
info={'ophys_experiment_id':eid,'ophys_session_id':str(meta_row.ophys_session_id),'mouse_id':str(meta_row.mouse_id),
      'session_type':str(meta_row.session_type),'targeted_structure':str(meta_row.targeted_structure),
```

iii. In Step 5, the AI states: “Session unit: One NWB ophys experiment/imaging plane is one target session.” It justifies this by saying simultaneous planes legitimately repeat behavior but contain distinct cells.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each eligible row, the AI builds a variable-length trial using a uniform 100 ms grid from `start_time` to `stop_time`, rather than native ophys frame indices.

ii. 
```python
tr=f['intervals/trials']
...
starts=np.asarray(tr['start_time'][:],float); stops=np.asarray(tr['stop_time'][:],float)
...
for ii in inds:
    q=trial_grid(starts[ii],stops[ii])
```

```python
def trial_grid(start, stop):
    # Half-open trial, 100 ms centers. Avoid floating endpoint ambiguity.
    n = int(np.floor((stop-start)/DT + 1e-9))
    return start + DT/2 + DT*np.arange(n, dtype=float)
```

iii. In Step 5, the AI says it will “Segment with native trial start/stop boundaries” but use a common 100 ms grid with centers `start_time + 0.05 + 0.1*k`, keeping variable-duration trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. The code then drops trials with fewer than 2 bins, trials whose 100 ms grid extends outside neural or running support, and trials where interpolated running or pupil data are missing/nonfinite. It also excludes three whole experiments up front because they lack pupil data, and requires at least two usable trials per experiment.

ii. 
```python
eligible=(~np.asarray(tr['aborted'][:],bool) & ~np.asarray(tr['auto_rewarded'][:],bool)
          & (np.asarray(tr['go'][:],bool)|np.asarray(tr['catch'][:],bool)))
inds=np.flatnonzero(eligible)
...
if len(q)<2 or q[0]<nts[0] or q[-1]>nts[-1] or q[0]<rts[0] or q[-1]>rts[-1]:
    dropped['support']+=1; continue
run=interp_vector_finite(rts,rvs,q); pup=interp_vector_finite(ets,diam,q)
if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)):
    dropped['nonfinite']+=1; continue
...
if len(records)<2: raise ValueError(f'{eid}: fewer than 2 usable trials')
```

```python
NO_PUPIL = {795953296,806456687,833631914}
...
paths=sorted(NWB_DIR.glob('*.nwb'),key=exp_id); paths=[p for p in paths if exp_id(p) not in NO_PUPIL]
```

iii. In Step 5, the AI justifies keeping exactly go/catch non-aborted non-auto-rewarded rows, requiring support on all streams, and excluding the three no-pupil experiments because “pupil is a required output and no defensible value can be fabricated.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the released dF/F trace matrix in `processing/ophys/dff/traces/data`, using `processing/ophys/dff/traces/timestamps` as the timebase.

ii. 
```python
nts=np.asarray(f['processing/ophys/dff/traces/timestamps'][:],float)
nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)
```

```python
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. In Steps 3-5, the AI repeatedly states that it chose the released dF/F product as the neural signal and would not recompute dF/F from raw fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI linearly interpolates the dF/F traces from native ophys timestamps onto the per-trial 100 ms grid, transposes the time-by-neuron matrix into neuron-by-time, and casts to `float32`. It does not apply additional normalization in the script.

ii. 
```python
def interp_matrix(ts, data, q):
    """Interpolate time x feature data; q must lie within ts."""
    idx = np.searchsorted(ts, q, side='left')
    idx = np.clip(idx, 1, len(ts)-1)
    lo, hi = idx-1, idx
    w = ((q-ts[lo])/(ts[hi]-ts[lo])).astype(np.float32)
    return data[lo] + (data[hi]-data[lo])*w[:,None]
```

```python
n=len(q); neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. Step 5 says: “Neural interpolation: Linear interpolation of released dF/F on absolute ophys timestamps.” The same section explains the 100 ms choice as a task-specific common grid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filter in the conversion script. The neural data are only filtered indirectly because whole experiments without pupil are excluded, and trials are dropped if their 100 ms grid cannot be supported by the available timestamps or if required behavioral streams are nonfinite.

ii. 
```python
if len(q)<2 or q[0]<nts[0] or q[-1]>nts[-1] or q[0]<rts[0] or q[-1]>rts[-1]:
    dropped['support']+=1; continue
...
if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)):
    dropped['nonfinite']+=1; continue
```

```python
paths=sorted(NWB_DIR.glob('*.nwb'),key=exp_id); paths=[p for p in paths if exp_id(p) not in NO_PUPIL]
```

iii. In Steps 3-5, the AI says it will use the released valid cell ROIs and not impose extra cell-level QC. Its explicit exclusions are about missing pupil data and unsupported trial windows rather than neuron quality.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to the native trial start time on the absolute ophys clock. Within each trial, samples are taken at 100 ms grid centers from `start_time + 50 ms` onward until just before `stop_time`.

ii. 
```python
def trial_grid(start, stop):
    # Half-open trial, 100 ms centers. Avoid floating endpoint ambiguity.
    n = int(np.floor((stop-start)/DT + 1e-9))
    return start + DT/2 + DT*np.arange(n, dtype=float)
```

```python
q=trial_grid(starts[ii],stops[ii])
...
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

```python
'time_bin_size':100.0,'temporal_alignment_event':'native trial start time on absolute ophys clock',
'off_start':0.0,'off_end':None,
```

iii. In Step 5, the AI explicitly documents the grid centers and says alignment is to “native trial start time on absolute ophys clock.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 100 ms bins. Yes: the script rebins / resamples all time-varying streams, including neural activity, by interpolation onto this common grid.

ii. 
```python
DT = 0.1
```

```python
def trial_grid(start, stop):
    ...
    return start + DT/2 + DT*np.arange(n, dtype=float)
```

```python
'time_bin_size':100.0,
'grid':'100 ms centers in half-open [trial_start, trial_stop)',
```

iii. The AI’s Step 5 rationale is that 100 ms is close to but not finer than the slowest native ~93 ms sampling, giving one common time bin size across experiments.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the NWB stimulus presentation stream, specifically `stimulus/presentation/.../timestamps` and `.../data`, together with `stimulus/templates/.../control` and `control_description` for mapping stimulus codes to image names. It is not derived from the trial table’s `initial_image_name` and `change_image_name`.

ii. 
```python
def stimulus_info(f):
    pg = next(iter(f['stimulus/presentation'].values()))
    tg = next(iter(f['stimulus/templates'].values()))
    desc = [x.decode() if isinstance(x,bytes) else str(x) for x in tg['control_description'][:]]
    controls = np.asarray(tg['control'][:], int)
    lookup = {int(c): IMAGE_TO_INT[d] for c,d in zip(controls,desc)}
    ...
    return np.asarray(pg['timestamps'][:],float), np.asarray(pg['data'][:],int), lookup
```

iii. Step 5 says the AI will “Decode NWB presentation control indices using template `control_description`” instead of using trial image-name columns.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI adds a dedicated `gray` class and labels each 100 ms bin as an image only when it falls within 250 ms of a stimulus presentation timestamp; otherwise it labels the bin as gray. The labels are integer-coded using a fixed `IMAGE_NAMES` list.

ii. 
```python
IMAGE_NAMES = ['gray','im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_INT = {x:i for i,x in enumerate(IMAGE_NAMES)}
```

```python
def image_labels(q, stim_ts, stim_val, lookup):
    out = np.zeros(len(q), dtype=np.int16)
    j = np.searchsorted(stim_ts, q, side='right')-1
    ok = (j>=0) & ((q-stim_ts[np.clip(j,0,len(stim_ts)-1)]) < 0.250)
    jj = j[ok]
    out[ok] = np.array([lookup.get(int(v),0) for v in stim_val[jj]], dtype=np.int16)
    return out
```

iii. Step 5 justifies this as “actual 250 ms visibility” with “gray during 500 ms inter-image periods.”

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same per-trial 100 ms grid `q` used for neural interpolation, so both streams share the same bin centers and trial lengths.

ii. 
```python
for ii,q,run,pup,outcome in records:
    n=len(q); neu=interp_matrix(nts,nds,q).T.astype(np.float32)
    img=image_labels(q,sts,sval,lookup)
```

```python
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
assert neu.shape[1]==out.shape[1]==n
```

iii. The AI’s Step 5 plan explicitly uses one common grid for all streams so that stimulus and neural data line up exactly in the output arrays.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the trial table’s `change_time` and `is_change` fields. The code uses `is_change` to suppress catch-trial sham changes.

ii. 
```python
changes=np.asarray(tr['change_time'][:],float)
is_changes=np.asarray(tr['is_change'][:],bool)
...
ch=np.zeros(n,dtype=np.int16)
if is_changes[ii] and np.isfinite(changes[ii]):
    k=np.searchsorted(q,changes[ii],side='left')
    if k<n: ch[k]=1
if not is_changes[ii]: assert ch.sum()==0
```

iii. `CONVERSION_NOTES.md` Step 7 says the initial implementation incorrectly pulsed catch trials because catch rows also have a scheduled `change_time`; the AI then “Fixed construction to gate pulses by native `is_change`.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI makes image change a one-bin pulse: it finds the first 100 ms bin at or after `change_time` and sets only that bin to 1 when `is_change` is true.

ii. 
```python
ch=np.zeros(n,dtype=np.int16)
if is_changes[ii] and np.isfinite(changes[ii]):
    k=np.searchsorted(q,changes[ii],side='left')
    if k<n: ch[k]=1
```

iii. Step 5 states: “Change pulse: Set the first grid center at or after native `change_time` to 1. This implements ‘right after’ and avoids anticipatory labeling.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary categorical output with classes `no_change` and `change`.

ii. 
```python
ch=np.zeros(n,dtype=np.int16)
...
'output_values':[IMAGE_NAMES,['no_change','change'],['Q1_lowest','Q2','Q3','Q4','Q5_highest'],
```

iii. The AI does not discuss additional thresholding beyond making the variable binary; the Step 5 mapping table defines it as “Binary pulse.”

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same per-trial 100 ms grid `q` as the neural data, with the pulse placed by `np.searchsorted` into that grid.

ii. 
```python
for ii,q,run,pup,outcome in records:
    ...
    if is_changes[ii] and np.isfinite(changes[ii]):
        k=np.searchsorted(q,changes[ii],side='left')
        if k<n: ch[k]=1
```

```python
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
assert neu.shape[1]==out.shape[1]==n
```

iii. Step 5 explicitly describes the pulse as the first bin at or after `change_time` on the shared grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB `processing/running/speed/timestamps` and `processing/running/speed/data` arrays.

ii. 
```python
rts=np.asarray(f['processing/running/speed/timestamps'][:],float)
rvs=np.asarray(f['processing/running/speed/data'][:],float)
```

```python
run=interp_vector_finite(rts,rvs,q)
```

iii. In Step 5, the AI maps `processing/running/speed` plus timestamps directly to the running-speed output channel.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed to the trial grid `q`, concatenates all retained running samples within an experiment, computes experiment-specific quintile edges, and then slices the resulting integer codes back into individual trials.

ii. 
```python
def interp_vector_finite(ts, values, q):
    ok = np.isfinite(ts) & np.isfinite(values)
    if ok.sum() < 2 or q[0] < ts[ok][0] or q[-1] > ts[ok][-1]:
        return None
    return np.interp(q, ts[ok], values[ok])
```

```python
records=[]; allrun=[]; allpupil=[]
...
run=interp_vector_finite(rts,rvs,q)
...
records.append((ii,q,run,pup,int(np.argmax(flags))))
allrun.append(run)
...
rv=np.concatenate(allrun)
rcodes,redges=quantile_codes(rv,np.ones(rv.size,bool))
```

iii. Step 5 says: “Linear interpolation, then session-wise equal-frequency quintiles,” justified as reducing between-session calibration effects.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into five equal-frequency bins within each experiment using quantiles and `np.searchsorted`. The categories are `Q1_lowest` through `Q5_highest`.

ii. 
```python
def quantile_codes(values, valid_mask, n=5):
    """Session-level equal-frequency bins, robust to repeated quantiles."""
    fit = values[valid_mask]
    ...
    edges = np.quantile(fit, np.arange(1,n)/n)
    return np.searchsorted(edges, values, side='right').astype(np.int16), edges
```

```python
'output_values':[IMAGE_NAMES,['no_change','change'],['Q1_lowest','Q2','Q3','Q4','Q5_highest'],
```

iii. In Step 5, the AI explicitly chooses per-experiment percentile bins and labels them as five quintile classes.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 100 ms trial grid `q` used for neural interpolation, then stacked into the output matrix with matching length.

ii. 
```python
run=interp_vector_finite(rts,rvs,q)
...
rr=rcodes[roff:roff+n]
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
assert neu.shape[1]==out.shape[1]==n
```

iii. Step 5 describes one shared absolute-time grid for all streams, with running aligned by interpolation rather than by native running indices.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/eye_tracking/timestamps` and `acquisition/EyeTracking/pupil_tracking/area`. The code converts area to an equivalent-circle diameter; it does not use a `pupil_width` column.

ii. 
```python
ets=np.asarray(f['acquisition/EyeTracking/eye_tracking/timestamps'][:],float)
area=np.asarray(f['acquisition/EyeTracking/pupil_tracking/area'][:],float)
diam=2*np.sqrt(np.maximum(area,0)/np.pi)
```

iii. In Steps 4-5, the AI argues that the task requires diameter, the released pupil product is area-based, and equivalent-circle diameter is an orientation-invariant way to turn area into one scalar diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code converts area to diameter, linearly interpolates finite values onto the trial grid, rejects trials if interpolation cannot be supported or produces nonfinite values, concatenates retained values within the experiment, and discretizes them into experiment-specific quintiles.

ii. 
```python
diam=2*np.sqrt(np.maximum(area,0)/np.pi)
...
pup=interp_vector_finite(ets,diam,q)
if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)):
    dropped['nonfinite']+=1; continue
...
pv=np.concatenate(allpupil)
pcodes,pedges=quantile_codes(pv,np.ones(pv.size,bool))
```

iii. Step 5 says the AI will “Interpolate only between finite anchors; do not extrapolate outside eye support,” and will exclude the no-pupil experiments rather than fabricate values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into five equal-frequency bins within each experiment using quantiles. The output classes are `Q1_smallest` through `Q5_largest`.

ii. 
```python
pcodes,pedges=quantile_codes(pv,np.ones(pv.size,bool))
...
'output_values':[IMAGE_NAMES,['no_change','change'],['Q1_lowest','Q2','Q3','Q4','Q5_highest'],
                 ['Q1_smallest','Q2','Q3','Q4','Q5_largest'],OUTCOME_KEYS],
```

iii. The AI’s Step 5 mapping table explicitly defines pupil as session-wise equal-frequency quintiles after converting area to diameter.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same per-trial 100 ms grid `q` as the neural data and then sliced into the output matrix with matching time length.

ii. 
```python
pup=interp_vector_finite(ets,diam,q)
...
pp=pcodes[poff:poff+n]
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
assert neu.shape[1]==out.shape[1]==n
```

iii. The AI’s justification is the same shared-grid rationale used for running speed and image identity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
OUTCOME_KEYS = ['hit','miss','false_alarm','correct_reject']
...
outcome_flags=np.vstack([np.asarray(tr[k][:],bool) for k in OUTCOME_KEYS])
```

iii. In Step 5, the AI maps those four flags directly to the four output classes and treats them as mutually exclusive.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code asserts that exactly one of the four outcome flags is true for each retained trial, converts the one-hot flags to an integer class by `argmax` in `OUTCOME_KEYS` order, and repeats that static class across all bins of the trial.

ii. 
```python
flags=outcome_flags[:,ii]
if flags.sum()!=1: raise AssertionError(f'{eid} trial {ii}: outcome not exclusive')
records.append((ii,q,run,pup,int(np.argmax(flags))))
```

```python
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
...
'output_static':[False,False,False,False,True],
```

iii. Step 5 explains that trial outcome is semantically static, but repeated over time if the validator expects a rectangular output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data are handled conservatively. The AI excludes three experiments with no pupil stream, refuses to extrapolate outside finite running/pupil support, drops trials with unsupported or nonfinite interpolated values, asserts mutually exclusive outcomes, and raises an error if fewer than two usable trials remain. It does not impute missing pupil/running bins into a fallback category.

ii. 
```python
NO_PUPIL = {795953296,806456687,833631914}
...
def interp_vector_finite(ts, values, q):
    ok = np.isfinite(ts) & np.isfinite(values)
    if ok.sum() < 2 or q[0] < ts[ok][0] or q[-1] > ts[ok][-1]:
        return None
    return np.interp(q, ts[ok], values[ok])
```

```python
if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)):
    dropped['nonfinite']+=1; continue
...
if flags.sum()!=1: raise AssertionError(f'{eid} trial {ii}: outcome not exclusive')
...
if len(records)<2: raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. In Steps 5 and 10, the AI explicitly justifies not fabricating pupil outputs, treating missingness as a reason to exclude data when a required output cannot be constructed.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify large NWB / dF/F reads and per-trial interpolation as the main expensive steps. It specifically notes that an earlier draft redundantly reread dF/F inside the trial loop and that this was optimized away.

ii. 
```python
nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)
...
for ii in inds:
    q=trial_grid(starts[ii],stops[ii])
    ...
for ii,q,run,pup,outcome in records:
    n=len(q); neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. In Step 6, the AI writes: “Initial draft converted the entire HDF5 dF/F dataset to NumPy once per trial, which would cause severe redundant I/O,” and says it changed this to one dF/F read per experiment plus vectorized interpolation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the two trial loops in `process_experiment` and the list-comprehension-based image lookup inside `image_labels`. The code currently loops over every retained trial once to validate/interpolate running and pupil, then again to build neural and output arrays.

ii. 
```python
for ii in inds:
    q=trial_grid(starts[ii],stops[ii])
    ...
    records.append((ii,q,run,pup,int(np.argmax(flags))))
```

```python
for ii,q,run,pup,outcome in records:
    n=len(q); neu=interp_matrix(nts,nds,q).T.astype(np.float32)
    img=image_labels(q,sts,sval,lookup)
```

```python
out[ok] = np.array([lookup.get(int(v),0) for v in stim_val[jj]], dtype=np.int16)
```

iii. The AI does not spell out all of these in the notes, but Step 6 explicitly discusses performance work and mentions replacing redundant per-trial HDF5 reads with more vectorized handling.

## 9-c. What processing does the code repeat multiple times?

i. The code avoids rereading raw dF/F for each trial, but it still performs a two-pass per-trial workflow: first pass to create `records` and gather continuous running/pupil for quantile fitting, second pass to interpolate neural data and assemble outputs. The trial grid `q` is reused rather than recomputed in the second pass.

ii. 
```python
records=[]; allrun=[]; allpupil=[]
for ii in inds:
    q=trial_grid(starts[ii],stops[ii])
    ...
    records.append((ii,q,run,pup,int(np.argmax(flags))))
    allrun.append(run); allpupil.append(pup)
...
for ii,q,run,pup,outcome in records:
    n=len(q); neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. Step 6 says the AI deliberately removed the worse repeated processing pattern from the initial draft. What remains is a smaller two-pass design driven by the need to compute experiment-level quantile edges before final output assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs some diagnostic work that is not needed by downstream decoding: optional plotting, collecting a `raw_plot` example tuple, detailed `session_info` metadata and edge values, and console timing/summary output. It also carries empty per-trial `input` arrays because the target format requires an `input` field even though the decoder has no inputs.

ii. 
```python
neural=[]; inputs=[]; outputs=[]; raw_plot=None; roff=poff=0
...
if raw_plot is None: raw_plot=(q,neu,img,ch,run,pup,rr,pp)
if make_plot:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ...
    fig.savefig(f'/app/processing_{eid}.png',dpi=130); plt.close(fig)
```

```python
inputs.append(np.empty((0,n),dtype=np.float32))
...
'session_info':infos
```

iii. The notes describe these as diagnostics and audit artifacts rather than part of the core decoder representation. There is no indication that the AI believed they changed the downstream scientific content.
