# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every local `behavior_ophys_experiment_*.nwb`, removes three experiment IDs that have no pupil stream, and reads each file directly with `h5py`. It joins experiment metadata from `ophys_experiment_table.csv`; it does not use the SDK cache or download experiments listed only in the full metadata release.

ii.
```python
paths=sorted(NWB_DIR.glob('*.nwb'),key=exp_id); paths=[p for p in paths if exp_id(p) not in NO_PUPIL]
table=pd.read_csv(META_DIR/'ophys_experiment_table.csv').set_index('ophys_experiment_id')
for si,p in enumerate(paths):
    eid=exp_id(p); row=table.loc[eid]
    n,i,o,info=process_experiment(p,row,args.show_processing and si<2)
```

iii. The notes say the supplied directory is a 284-file local subset while the metadata describes the larger release. Direct NWB reads were chosen to process precisely the supplied files; three files were deliberately excluded because pupil diameter is mandatory and the agent considered fabrication indefensible.

## 1-b. How are the data split into subjects?

i. A subject is the metadata `mouse_id`. Unique mouse IDs are converted to strings, sorted, and each retained experiment gets an index into that list.

ii.
```python
mice.append(str(row.mouse_id))
subjects=sorted(set(mice))
'subject_idx':np.asarray([subjects.index(x) for x in mice],dtype=np.int64)
```

iii. The notes identify `mouse_id` as the animal identifier and report 38 retained mice.

## 1-c. How are the data split into sessions?

i. Each NWB ophys experiment/imaging plane is treated as an independent target session, even when several files share an `ophys_session_id`.

ii.
```python
for si,p in enumerate(paths):
    ...
    neural.append(n); inputs.append(i); outputs.append(o)
```

iii. The agent argues that an SDK `BehaviorOphysExperiment` is one plane, simultaneous planes have distinct cells and native rates, and the paper performs decoding at the imaging-plane level. It explicitly accepts duplicated behavioral streams across simultaneous planes.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. For every eligible row, the agent creates a half-open grid of 100 ms bin centers spanning native `start_time` to `stop_time`; trial lengths therefore vary.

ii.
```python
def trial_grid(start, stop):
    n = int(np.floor((stop-start)/DT + 1e-9))
    return start + DT/2 + DT*np.arange(n, dtype=float)
...
for ii in inds:
    q=trial_grid(starts[ii],stops[ii])
```

iii. The notes say native trials preserve the experiment's behavioral definition, while a common 100 ms grid satisfies the requirement for a uniform bin size across native fast- and slow-rate recordings.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be go or catch and must be neither aborted nor auto-rewarded. Trials are then dropped if they have fewer than two grid points, lie outside neural/running support, lack sufficient finite running or pupil anchors, or produce nonfinite behavior. An experiment must retain at least two trials.

ii.
```python
eligible=(~np.asarray(tr['aborted'][:],bool) & ~np.asarray(tr['auto_rewarded'][:],bool)
          & (np.asarray(tr['go'][:],bool)|np.asarray(tr['catch'][:],bool)))
...
if len(q)<2 or q[0]<nts[0] or q[-1]>nts[-1] or q[0]<rts[0] or q[-1]>rts[-1]: continue
if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)): continue
if len(records)<2: raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. The first predicate directly follows the task. Support and finiteness checks prevent extrapolation or fabricated categories; the notes report that no eligible trial was lost in retained experiments.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the released NWB dF/F trace matrix and its timestamps.

ii.
```python
nts=np.asarray(f['processing/ophys/dff/traces/timestamps'][:],float)
nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)
```

iii. The agent chose released processed dF/F because the SDK already associates it with valid segmented ROIs; recomputing fluorescence processing would diverge from the release.

## 2-b. How is the `neural` data processed?

i. The time-by-cell dF/F matrix is linearly interpolated at every 100 ms trial-grid center, transposed to cells by time, and stored as `float32`. Planes are not merged.

ii.
```python
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. The notes describe linear interpolation as conservative on a 100 ms grid, using absolute ophys timestamps and no extrapolation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra cell-level filter is applied; all dF/F columns in retained experiments are used. Trial windows outside neural timestamp support are dropped, and entire no-pupil experiments are omitted for output completeness rather than neural quality.

ii.
```python
if len(q)<2 or q[0]<nts[0] or q[-1]>nts[-1] ...:
    dropped['support']+=1; continue
```

iii. The notes state that released valid ROI/cell records already embody Allen QC and that electrophysiology-style thresholds would be inappropriate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial begins at its native trial `start_time`; samples are absolute-time 100 ms centers in `[start_time, stop_time)`. There is no fixed window around `change_time`.

ii.
```python
q=trial_grid(starts[ii],stops[ii])
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. The agent records the temporal alignment event as native trial start and uses absolute clocks so neural and outputs share exactly the same query times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 100 ms. Native dF/F at roughly 32 or 93 ms spacing is linearly resampled; this is interpolation, not averaging within bins.

ii.
```python
DT = 0.1
...
'time_bin_size':100.0,
'grid':'100 ms centers in half-open [trial_start, trial_stop)'
```

iii. The agent chose 100 ms to avoid claiming finer resolution than slow multiplane recordings while retaining two or three samples per 250 ms image flash.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation timestamps/data and template `control`/`control_description`, not the trial's initial/change-name fields.

ii.
```python
pg = next(iter(f['stimulus/presentation'].values()))
tg = next(iter(f['stimulus/templates'].values()))
lookup = {int(c): IMAGE_TO_INT[d] for c,d in zip(controls,desc)}
```

iii. The notes say presentation records capture every flash and omission and therefore better represent the genuinely visible stimulus.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Template controls are mapped to a fixed global 17-class vocabulary. At each grid point the latest presentation is used only if it began less than 250 ms earlier; otherwise the class is gray (0).

ii.
```python
j = np.searchsorted(stim_ts, q, side='right')-1
ok = (j>=0) & ((q-stim_ts[np.clip(j,0,len(stim_ts)-1)]) < 0.250)
out[ok] = np.array([lookup.get(int(v),0) for v in stim_val[j[ok]]], dtype=np.int16)
```

iii. This follows the 250 ms image/500 ms gray cadence and uses one stable codebook across image sets A and B.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the identical query vector `q` used to interpolate neural activity.

ii.
```python
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
img=image_labels(q,sts,sval,lookup)
```

iii. Shared absolute query times guarantee matching time columns.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses trial `is_change` and `change_time`.

ii.
```python
changes=np.asarray(tr['change_time'][:],float)
is_changes=np.asarray(tr['is_change'][:],bool)
```

iii. The notes distinguish true identity changes from catch/sham events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is made and, for a true finite change, the first 100 ms bin whose center is at or after `change_time` is set to one.

ii.
```python
ch=np.zeros(n,dtype=np.int16)
if is_changes[ii] and np.isfinite(changes[ii]):
    k=np.searchsorted(q,changes[ii],side='left')
    if k<n: ch[k]=1
```

iii. The agent interpreted “right after a change” as a one-bin event pulse rather than a persistent post-change state.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 means no true change event in that bin and 1 means the first bin after a true change. Catch trials remain all zero.

ii.
```python
if not is_changes[ii]: assert ch.sum()==0
```

iii. No continuous threshold is required; the native boolean and temporal event rule define the categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `change_time` is searched within the same 100 ms center vector used for neural interpolation.

ii.
```python
k=np.searchsorted(q,changes[ii],side='left')
```

iii. Thus the pulse column is the first neural sample at or after the event.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses processed running speed data and its native timestamps.

ii.
```python
rts=np.asarray(f['processing/running/speed/timestamps'][:],float)
rvs=np.asarray(f['processing/running/speed/data'][:],float)
```

iii. The released processed running stream is the SDK-standard locomotion measure.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are linearly interpolated to `q`. All retained-trial values within an experiment are concatenated, four 20/40/60/80% edges are fit once, and values are coded 0–4.

ii.
```python
run=interp_vector_finite(rts,rvs,q)
rv=np.concatenate(allrun)
rcodes,redges=quantile_codes(rv,np.ones(rv.size,bool))
```

iii. Session/experiment-level quintiles were intended to equalize occupancy and reduce apparatus/session scaling effects while avoiding per-trial leakage.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four experiment-specific quantile edges divide running speed into five equal-frequency categories; `side='right'` places values equal to an edge in the higher bin.

ii.
```python
edges = np.quantile(fit, np.arange(1,n)/n)
return np.searchsorted(edges, values, side='right').astype(np.int16), edges
```

iii. The requested “five equal percentile bins” motivated quintiles.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed and neural dF/F are independently interpolated at the same `q` values.

ii.
```python
run=interp_vector_finite(rts,rvs,q)
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. Absolute synchronized timestamps provide alignment without assuming matching source indices.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses processed pupil-tracking `area` and eye-tracking timestamps.

ii.
```python
ets=np.asarray(f['acquisition/EyeTracking/eye_tracking/timestamps'][:],float)
area=np.asarray(f['acquisition/EyeTracking/pupil_tracking/area'][:],float)
diam=2*np.sqrt(np.maximum(area,0)/np.pi)
```

iii. The agent preferred processed area because Allen's blink/outlier handling has already made bad frames NaN; equivalent-circle diameter is orientation invariant.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Nonnegative area is converted to equivalent-circle diameter, finite anchors are linearly interpolated to `q`, and values are converted to experiment-level quintiles.

ii.
```python
diam=2*np.sqrt(np.maximum(area,0)/np.pi)
pup=interp_vector_finite(ets,diam,q)
pcodes,pedges=quantile_codes(pv,np.ones(pv.size,bool))
```

iii. This preserves the scalar size implied by processed ellipse area and bridges blink gaps only between finite surrounding measurements.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four experiment-specific quantile edges split interpolated diameter into five approximately equal-frequency classes, coded 0–4.

ii.
```python
pcodes,pedges=quantile_codes(pv,np.ones(pv.size,bool))
```

iii. The same session-wise equal-occupancy rationale as running speed is used.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Diameter is interpolated at exactly the same trial-grid centers `q` as dF/F.

ii.
```python
pup=interp_vector_finite(ets,diam,q)
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. The shared absolute grid synchronizes eye and ophys streams and prohibits extrapolation outside finite eye support.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the four native trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_KEYS = ['hit','miss','false_alarm','correct_reject']
outcome_flags=np.vstack([np.asarray(tr[k][:],bool) for k in OUTCOME_KEYS])
```

iii. These are the canonical mutually exclusive outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code asserts exactly one flag, takes its index as class 0–3, and repeats that static label across all time bins.

ii.
```python
flags=outcome_flags[:,ii]
if flags.sum()!=1: raise AssertionError(...)
outcome=int(np.argmax(flags))
...
np.full(n,outcome,dtype=np.int16)
```

iii. Repetition makes a static per-trial target fit the rectangular time-varying output matrix; metadata still marks it static.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Three experiments with no eye stream are explicitly excluded. Interpolation removes nonfinite anchors, refuses extrapolation or fewer than two anchors, and drops unsupported/nonfinite trials. Unexpected template mappings, nonexclusive outcomes, fewer than two usable trials, and shape/range violations raise errors rather than being silently repaired.

ii.
```python
NO_PUPIL = {795953296,806456687,833631914}
ok = np.isfinite(ts) & np.isfinite(values)
if ok.sum() < 2 or q[0] < ts[ok][0] or q[-1] > ts[ok][-1]: return None
if flags.sum()!=1: raise AssertionError(...)
```

iii. The agent states that mandatory pupil values should not be fabricated. Finite interpolation bridges reference-marked blink gaps, while explicit counters and assertions expose losses and malformed data.

## 9-a. What are the most time-consuming steps of the code?

i. Reading each experiment's large dF/F HDF5 matrix and interpolating it for hundreds of trials dominate conversion; full decoder training is much slower than conversion but is downstream validation.

ii.
```python
nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)
...
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. The notes explicitly optimize around one dF/F read per experiment and estimate several minutes for the 281 retained experiments plus I/O.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer experiment loop is serial, the trial loop repeatedly searches/interpolates, and image class assignment has a Python comprehension over valid time points. Trials could be concatenated and interpolated in a batched call; experiments could be processed in parallel subject to memory/I/O limits; the lookup could use an array.

ii.
```python
for ii in inds:
    q=trial_grid(starts[ii],stops[ii])
...
out[ok] = np.array([lookup.get(int(v),0) for v in stim_val[jj]], dtype=np.int16)
```

iii. The notes describe interpolation itself as vectorized across time and cells, but do not claim the surrounding trial/experiment loops are fully vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Each trial performs separate timestamp searches for neural, running, pupil, stimulus identity, and change. It also concatenates running/pupil after first storing per-trial arrays, then loops over records again to construct final arrays. Simultaneous-plane files repeat identical behavior processing by design.

ii.
```python
for ii in inds:
    ... records.append((ii,q,run,pup,...)); allrun.append(run); allpupil.append(pup)
...
for ii,q,run,pup,outcome in records:
    neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. Two passes are used so experiment-wide quantile edges can be fit before categorical outputs are emitted; repeated plane behavior is accepted because planes are independent target sessions.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Continuous running and pupil vectors are retained only long enough to fit/code quintiles and are not saved. When requested, diagnostic plotting builds figures and `raw_plot` used only for PNGs. `make_plot=False` avoids that work in ordinary full conversion.

ii.
```python
records.append((ii,q,run,pup,int(np.argmax(flags))))
...
if make_plot:
    ... fig.savefig(...)
```

iii. Continuous values are necessary intermediates for percentile thresholds; plots are optional conversion validation rather than decoder inputs.
