# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK object API entirely and read the released NWB (HDF5) files directly with `h5py`, joined to the release's `project_metadata/ophys_experiment_table.csv` for subject/region/session-type metadata. Every `*.nwb` file in `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` is enumerated (284 files), sorted by numeric experiment id, and processed in a single pass; three hard-coded experiments that contain no eye-tracking series are removed from the file list before processing. No project-code filter is applied, so both `VisualBehavior` (239 single-plane, ~31 Hz) and `VisualBehaviorMultiscope` (45 multi-plane, ~10.7 Hz) experiments are included. Within each file, the arrays read are `processing/ophys/dff/traces/{data,timestamps}`, `processing/running/speed/{data,timestamps}`, `acquisition/EyeTracking/eye_tracking/timestamps`, `acquisition/EyeTracking/pupil_tracking/area`, `stimulus/presentation/*`, `stimulus/templates/*`, and `intervals/trials`. Final result: 281 sessions, 38 mice, 41,871 neurons, 84,313 trials.

ii.
```python
DATA_ROOT = Path('/app/data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_ROOT / 'behavior_ophys_experiments'
META_DIR = DATA_ROOT / 'project_metadata'
NO_PUPIL = {795953296,806456687,833631914}
...
paths=sorted(NWB_DIR.glob('*.nwb'),key=exp_id); paths=[p for p in paths if exp_id(p) not in NO_PUPIL]
...
table=pd.read_csv(META_DIR/'ophys_experiment_table.csv').set_index('ophys_experiment_id')
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
    ets=np.asarray(f['acquisition/EyeTracking/eye_tracking/timestamps'][:],float)
    area=np.asarray(f['acquisition/EyeTracking/pupil_tracking/area'][:],float)
```

iii. From CONVERSION_NOTES Step 6/Step 10: "Direct HDF5 reads of released NWB arrays and metadata CSV joins, avoiding network/cache dependencies" and "(a) Loading: script reads the same NWB products exposed by `BehaviorOphysExperiment.from_nwb`; direct HDF5 is used only for speed and avoids semantic alteration." Step 1 established that dF/F is already computed in the release and must not be recomputed, and that cell/ROI ordering in the dF/F RoiResponseSeries is the curated cell set. Step 4 justified including every supplied experiment: "Convert all supplied experiments because Decoder Task says collect Visual Behavior and explicitly defines trial filtering; paper subset restriction was analysis-specific." The three no-pupil files were excluded because "pupil is a required output and no defensible value can be fabricated."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` values from `ophys_experiment_table.csv`, cast to `str`, sorted lexicographically. `subject_idx[i]` is the index of the mouse owning session `i`. All 38 mice in the supplied release survive conversion.

ii.
```python
mice.append(str(row.mouse_id))
...
subjects=sorted(set(mice))
data={... 'subjects':subjects,
      'subject_idx':np.asarray([subjects.index(x) for x in mice],dtype=np.int64), ...}
```

iii. Step 5 mapping table: "`metadata mouse_id` → `subjects`, `subject_idx` — Unique sorted string IDs and per-experiment index; 38 source mice; final may omit mice only if all their files lack required pupil." `mouse_id` is the release's canonical animal identifier, and the count (38) was cross-checked against the metadata scan in Step 2.

## 1-c. How are the data split into sessions?

i. One target "session" = one NWB file = one *ophys experiment* (a single imaging plane), **not** one `ophys_session_id`. The 284 supplied experiments come from only 247 acquisition sessions: 239 sessions contribute one plane, but 8 Multiscope sessions contribute 3–7 simultaneously-recorded planes each. Each of those planes becomes an independent entry in `neural`/`output`, so the same behavioral trials (same running, pupil, images, outcomes) are emitted up to 7 times with different neuron sets. This is visible in the verification output as runs of identical trial counts (`... 209, 209, 209, 209, 209, 209, 209, ...`). No merging across planes is performed and no `ophys_session_id` grouping exists in the code.

ii.
```python
paths=sorted(NWB_DIR.glob('*.nwb'),key=exp_id)   # one path == one session
for si,p in enumerate(paths):
    eid=exp_id(p); row=table.loc[eid]
    n,i,o,info=process_experiment(p,row,...)
    neural.append(n); inputs.append(i); outputs.append(o); infos.append(info)
    regions.append(str(row.targeted_structure)); mice.append(str(row.mouse_id))
```
```python
info={'ophys_experiment_id':eid,'ophys_session_id':str(meta_row.ophys_session_id),...}
```

iii. Step 4 discrepancy table, "Session unit": "`BehaviorOphysExperiment` is one imaging plane … A few ophys sessions have 3-7 files with identical behavior but distinct cells/rates … Paper decodes each imaging plane → Treat each NWB experiment/plane as one target session, matching SDK and paper decoding unit." And "Simultaneous planes": "Keep each plane as a session; duplicated behavior is expected, not accidental duplication of neurons." The source `ophys_session_id` is preserved in `metadata['session_info']` so the grouping is recoverable.

## 1-d. How are the data split into trials?

i. Trials come from the native `intervals/trials` table. A trial is eligible if `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. Each eligible trial spans its native `start_time` → `stop_time` (variable length, 7.0–12.6 s), sampled on a uniform 100 ms grid of *bin centers* in the half-open interval: `n = floor((stop-start)/0.1)` centers at `start + 0.05 + 0.1k`. This yields 70–126 time points per trial (mean 84.8). 84,313 trials are retained out of 85,230 eligible (the 917 lost are all inside the three excluded no-pupil experiments; zero trials were dropped in retained experiments).

ii.
```python
eligible=(~np.asarray(tr['aborted'][:],bool) & ~np.asarray(tr['auto_rewarded'][:],bool)
          & (np.asarray(tr['go'][:],bool)|np.asarray(tr['catch'][:],bool)))
inds=np.flatnonzero(eligible)
...
def trial_grid(start, stop):
    # Half-open trial, 100 ms centers. Avoid floating endpoint ambiguity.
    n = int(np.floor((stop-start)/DT + 1e-9))
    return start + DT/2 + DT*np.arange(n, dtype=float)
...
for ii in inds:
    q=trial_grid(starts[ii],stops[ii])
```

iii. Step 5 Key Decision 2: "Keep exactly `(go OR catch) AND NOT aborted AND NOT auto_rewarded`," which is the literal instruction ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials") expressed with the native mutually-independent flags. Key Decision 4: "Grid centers are `start_time + 0.05 + 0.1*k`, with only centers strictly before native trial stop. Trials retain their native variable duration … `off_start=0`, `off_end=None` because alignment is to each trial's native start and end varies." Using the full native window (rather than a fixed window around the change) is what makes time-varying image identity/change meaningful.

## 1-e. How are trials filtered based on quality controls?

i. Beyond the go/catch flag filter, three quality gates are applied: (1) every grid point of the trial must lie inside the support of both the ophys timestamps and the running timestamps, otherwise the trial is dropped (`dropped['support']`); (2) interpolated running speed and pupil diameter must be finite at every grid point, otherwise the trial is dropped (`dropped['nonfinite']`); (3) a trial shorter than two bins is dropped. Sessions must retain ≥2 trials (otherwise `ValueError`), and the three experiments with no eye-tracking series at all are removed up front. Outcome exclusivity is asserted per trial. Drop counts are printed per experiment; in the full run every retained experiment reported `dropped={'support': 0, 'nonfinite': 0}`.

ii.
```python
q=trial_grid(starts[ii],stops[ii])
if len(q)<2 or q[0]<nts[0] or q[-1]>nts[-1] or q[0]<rts[0] or q[-1]>rts[-1]:
    dropped['support']+=1; continue
run=interp_vector_finite(rts,rvs,q); pup=interp_vector_finite(ets,diam,q)
if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)):
    dropped['nonfinite']+=1; continue
flags=outcome_flags[:,ii]
if flags.sum()!=1: raise AssertionError(f'{eid} trial {ii}: outcome not exclusive')
...
if len(records)<2: raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. Step 5 Key Decisions 10 and 11: "drop an individual trial only if any grid point lies outside finite eye support or interpolation cannot produce a finite value. Report all losses" and "Require finite interpolated running at every retained grid point; drop only unsupported trials and report." Step 2 had pre-verified that all eligible trial bounds fall inside the ophys support, so the gates were expected to be no-ops — they are defensive. The ≥2-trial rule comes from the target-format requirement that each session have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The released dF/F traces: `processing/ophys/dff/traces/data` (shape time × neuron, float64) with `processing/ophys/dff/traces/timestamps` as the ophys clock. No fluorescence, neuropil, or detected-event arrays are used; dF/F is not recomputed.

ii.
```python
nts=np.asarray(f['processing/ophys/dff/traces/timestamps'][:],float)
nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)
```

iii. Step 1: "Neural activity is calcium imaging, and NWB already contains processed delta-F/F traces. Recomputing dF/F from fluorescence would diverge from the released/reference processing." Step 4 records the deliberate divergence from the cited paper, which analysed detected events: "Use dF/F as the standard SDK continuous calcium-activity property requested by target `neural`; do not recompute. Record the divergence from the paper's specialized event analysis."

## 2-b. How is the `neural` data processed?

i. The only processing is resampling and typing: the full (time × neuron) matrix is read once per experiment as float32, then linearly interpolated at the trial's 100 ms grid centers with a vectorized two-point weight computation, and transposed to (n_neurons, n_timepoints). No z-scoring, smoothing, baseline subtraction, deconvolution, neuron filtering, or cross-plane stacking is applied (planes are separate sessions, see 1-c). Cell ordering is the released ROI response-series order.

ii.
```python
def interp_matrix(ts, data, q):
    """Interpolate time x feature data; q must lie within ts."""
    idx = np.searchsorted(ts, q, side='left')
    idx = np.clip(idx, 1, len(ts)-1)
    lo, hi = idx-1, idx
    w = ((q-ts[lo])/(ts[hi]-ts[lo])).astype(np.float32)
    return data[lo] + (data[hi]-data[lo])*w[:,None]
...
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
```

iii. Step 5 Key Decision 5: "Linear interpolation of released dF/F on absolute ophys timestamps. This is conservative at 100 ms and uses no extrapolation." Step 5 mapping row: "Linear interpolation at 100 ms grid centers; transpose to neuron x time; float32 — Released dF/F, no recomputation; cell order follows ROI response series." Step 6 notes the efficiency fix: "Load each experiment's dF/F matrix once as float32, then vectorized interpolation by index/weight arrays for each trial."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is performed. Every ROI present in the released dF/F traces is kept, and `brain_region_idx` is built as one label per retained neuron. The only neuron loss in the whole conversion is the 276 cells belonging to the three excluded no-pupil experiments (42,147 → 41,871).

ii.
```python
nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)   # all ROIs kept
...
'brain_region_idx':[np.full(sess[0].shape[0],brain_regions.index(r),dtype=np.int64)
                    for sess,r in zip(neural,regions)]
```

iii. Step 3 "Neuron curation rules": "Use released valid cell ROIs/cell specimen rows. The SDK aligns cell/ROI IDs across tables and processed traces. Do not impose electrophysiology criteria." Step 2 validated this choice numerically: "Exact metadata cell count (42,147) equals summed NWB dF/F columns, validating neuron identity/count alignment," i.e. the release has already applied segmentation/classification QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to each trial's native `start_time` on the absolute ophys clock. All streams (neural, image, change, running, pupil) are evaluated at the same absolute times `q = start + 0.05 + 0.1k`, so alignment is by construction — there is no frame-index alignment across clocks and no re-referencing to `change_time`. Metadata records `temporal_alignment_event='native trial start time on absolute ophys clock'`, `off_start=0.0`, `off_end=None` (variable trial length). Trials whose grid falls outside the ophys support are dropped rather than extrapolated.

ii.
```python
q=trial_grid(starts[ii],stops[ii])
if len(q)<2 or q[0]<nts[0] or q[-1]>nts[-1] or ...: dropped['support']+=1; continue
...
neu=interp_matrix(nts,nds,q).T.astype(np.float32)
img=image_labels(q,sts,sval,lookup)
rr=rcodes[roff:roff+n]; pp=pcodes[poff:poff+n]
assert neu.shape[1]==out.shape[1]==n
```
```python
'temporal_alignment_event':'native trial start time on absolute ophys clock',
'off_start':0.0,'off_end':None,
```

iii. Step 10 Check 3(c): "Alignment: all streams use their explicit timestamps, matching `OphysTimestamps`, `RunningSpeed`, and `EyeTrackingTable`; no frame-index alignment across clocks." Step 5 Key Decision 4 sets the half-open grid and explains `off_start`/`off_end`. Step 12 adds: "raw timestamps and converted labels were plotted for fast and slow ophys regimes … asynchronous behavior uses its own timestamps."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100.0 ms uniform bins for every trial and session (`metadata['time_bin_size']=100.0`). Yes, rebinning is applied: the native ophys rate is 32.31 ms (single-plane, ~31 Hz) or 93.23 ms (Multiscope, ~10.7 Hz), and both are resampled onto the common 100 ms grid. The resampling is *point-sampling by linear interpolation at bin centers*, not averaging/integration over the bin, so for 31 Hz sessions roughly two of every three native frames do not directly contribute to the output.

ii.
```python
DT = 0.1
...
def trial_grid(start, stop):
    n = int(np.floor((stop-start)/DT + 1e-9))
    return start + DT/2 + DT*np.arange(n, dtype=float)
...
'time_bin_size':100.0,
'grid':'100 ms centers in half-open [trial_start, trial_stop)',
```

iii. Step 5 Key Decision 3: "Use 100 ms (`metadata.time_bin_size=100.0`). This is close to but not finer than the slowest 93.23 ms native interval, avoids pretending multiplane data have 30 Hz resolution, resolves the target's cross-session uniformity requirement, and still provides 2-3 samples during each 250 ms image flash." Step 10 Check 3(d) flags this as the deliberate, task-required divergence from the paper's 30 Hz resampling: "this task requires one cross-session bin and source includes 10.7 Hz data, so 100 ms is used to avoid invented precision."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the stimulus stream, not the trials table: `stimulus/presentation/<set>/timestamps` (flash onset times) and `/data` (per-flash control index 0–7), decoded through `stimulus/templates/<set>/control` and `/control_description` (the eight image names of that session's image set). The trials-table fields `initial_image_name`/`change_image_name` are not used.

ii.
```python
def stimulus_info(f):
    pg = next(iter(f['stimulus/presentation'].values()))
    tg = next(iter(f['stimulus/templates'].values()))
    desc = [x.decode() if isinstance(x,bytes) else str(x) for x in tg['control_description'][:]]
    controls = np.asarray(tg['control'][:], int)
    lookup = {int(c): IMAGE_TO_INT[d] for c,d in zip(controls,desc)}
    if len(lookup) != 8 or any(v == 0 for v in lookup.values()):
        raise ValueError('Unexpected stimulus template/control mapping')
    return np.asarray(pg['timestamps'][:],float), np.asarray(pg['data'][:],int), lookup
```

iii. Step 4: "SDK reconstructs presentation table and image names … NWB presentation indices map to eight template descriptions … Build image identity from actual presentation timestamps/data, with a dedicated gray class between image presentations." Step 5 Key Decision 6: "Decode NWB presentation control indices using template `control_description`. Mark the actual 250 ms image display after each presentation timestamp, gray otherwise. This honors the experiment rather than labeling the entire 750 ms interval as an image."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A fixed global 17-class code list is hard-coded: `gray` = 0 followed by the 16 image names of sets A and B in lexicographic order. Per bin, the most recent flash onset at or before the bin center is found; the bin takes that flash's image code if the bin is within 250 ms of the onset, otherwise `gray` (0). Omitted flashes simply have no presentation entry, so they fall out as `gray`. Result over the full dataset: gray 0.666, each image ≈0.020–0.022.

ii.
```python
IMAGE_NAMES = ['gray','im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_INT = {x:i for i,x in enumerate(IMAGE_NAMES)}
...
def image_labels(q, stim_ts, stim_val, lookup):
    out = np.zeros(len(q), dtype=np.int16)
    j = np.searchsorted(stim_ts, q, side='right')-1
    ok = (j>=0) & ((q-stim_ts[np.clip(j,0,len(stim_ts)-1)]) < 0.250)
    jj = j[ok]
    out[ok] = np.array([lookup.get(int(v),0) for v in stim_val[jj]], dtype=np.int16)
    return out
```

iii. Step 3/Step 4: the references establish "eight natural images, each displayed for 250 ms and followed by 500 ms gray." Step 5 "Output Classes": "`image_identity`: `gray`, then lexicographically sorted `im000 … im106`." Step 5 note: "Classes: gray plus 16 named images"; the enumeration of exactly 16 images across sets A/B was verified by scanning every file in Step 5 ("Image identities are exactly 16 named images across sets A/B, with no extra trial-only names").

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on the same `q` array of bin centers used for the neural interpolation, so it is aligned by construction. The label transition happens at the first bin center at/after a flash onset, and reverts to gray at the first bin center ≥250 ms after that onset. An assertion enforces equal length with the neural matrix, and a `--show-processing` panel plots image class against the dF/F heatmap on the same time axis.

ii.
```python
n=len(q); neu=interp_matrix(nts,nds,q).T.astype(np.float32)
img=image_labels(q,sts,sval,lookup)
...
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
assert neu.shape[1]==out.shape[1]==n
assert img.min()>=0 and img.max()<len(IMAGE_NAMES) and ...
```

iii. Step 10 Check 2: "Outputs: independently reconstructed image visibility … from raw NWB; every tested array matched with np.allclose." Step 12: "Stimulus identity uses actual 250 ms visibility," cited as the evidence that stimulus/neural temporal alignment is correct (image identity decodes at 6.2× chance).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The trials table's `is_change` flag and `change_time`. `is_change` is used as the gate (it is `True` for all go trials and `False` for all catch trials among eligible trials), and `change_time` gives the moment of the identity switch.

ii.
```python
changes=np.asarray(tr['change_time'][:],float)
is_changes=np.asarray(tr['is_change'][:],bool)
...
if is_changes[ii] and np.isfinite(changes[ii]):
    k=np.searchsorted(q,changes[ii],side='left')
    if k<n: ch[k]=1
if not is_changes[ii]: assert ch.sum()==0
```

iii. Step 7 "Issue Found and Fixed": "Initial sample had one `image_change` pulse on every go and catch trial because native catch `change_time` marks the scheduled sham-change time. This contradicted identity-change semantics. Fixed construction to gate pulses by native `is_change`; added a catch assertion and reconverted/revalidated. Recheck: converted pulse counts 306 and 181 exactly equal raw eligible changes; catch counts 43 and 28 contribute zero pulses."

## 4-b. What processing is involved in computing `output` *Image change*?

i. Essentially none beyond a `searchsorted`: a zero vector of length `n` is created and a single element is set to 1 — the first bin whose center is at or after `change_time` — and only on trials with `is_change=True` and a finite `change_time`. Catch trials stay all-zero (asserted). Over the full dataset this gives 73,733 pulses, exactly the number of retained change trials, i.e. 1.03% of all time bins.

ii.
```python
ch=np.zeros(n,dtype=np.int16)
if is_changes[ii] and np.isfinite(changes[ii]):
    k=np.searchsorted(q,changes[ii],side='left')
    if k<n: ch[k]=1
if not is_changes[ii]: assert ch.sum()==0
```

iii. Step 5 Key Decision 7: "Set the first grid center at or after native `change_time` to 1. This implements 'right after' and avoids anticipatory labeling." Step 9 consistency table records "73,733 pulses (0.010264 of bins)" against "73,733 retained go changes."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Binary, `output_values = ['no_change','change']`. The positive class occupies exactly one 100 ms bin per change trial; there is no extension over the post-change flash or the following grey period and no smoothing/dilation of the label. Resulting class balance: no_change 0.990 / change 0.010; decoder validation balanced accuracy 0.6281 vs 0.5 chance (1.26×).

ii.
```python
'output_names':['image_identity','image_change','running_speed_quintile','pupil_diameter_quintile','trial_outcome'],
'output_values':[IMAGE_NAMES,['no_change','change'],...]
...
if k<n: ch[k]=1
```

iii. Step 12 Check 1: "image-change prevalence is 1.026% of time bins by design because it is a one-bin event … Thus low ratios are not caused by accidental constant outputs." The AI treated the 1.26×-chance result as acceptable after re-verifying raw values and alignment: "no conversion change is justified."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same `q` grid as the neural data; the pulse index is obtained with `np.searchsorted(q, change_time, side='left')`, i.e. the first bin center at or after the true change time (never before it), and it is discarded if it falls past the end of the trial grid. Row 1 of the same `(5, n)` output matrix as the neural `(n_neurons, n)` matrix, with an equal-length assertion.

ii.
```python
k=np.searchsorted(q,changes[ii],side='left')
if k<n: ch[k]=1
...
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
assert neu.shape[1]==out.shape[1]==n
```

iii. Step 10 Check 5: "Initial catch-pulse bug was found in Step 7 and fixed by gating native scheduled `change_time` with `is_change`; full pulse total now equals raw retained changes." Step 5 Key Decision 7 explains the at-or-after convention as avoiding anticipatory labelling.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` (the filtered running speed in cm/s, the same array the SDK exposes as `running_speed`) with its own `processing/running/speed/timestamps`. The unfiltered variant (`speed_unfiltered`) and the raw encoder signal (`dx`) are not used.

ii.
```python
rts=np.asarray(f['processing/running/speed/timestamps'][:],float)
rvs=np.asarray(f['processing/running/speed/data'][:],float)
```

iii. Step 5 mapping table: "`processing/running/speed` + timestamps → `output[...,2,:]` — Linear interpolation, then session-wise equal-frequency quintiles — `RunningSpeed.from_nwb`." Step 1 identified `RunningSpeed.from_nwb` as the SDK's loader for "timestamped running speed in cm/s," so the filtered `speed` array is the SDK-equivalent product.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (restricted to finite timestamp/value pairs, no extrapolation) from the native running clock onto the trial's 100 ms bin centers; then equal-frequency quintile coding. No smoothing, clipping, or absolute-value transform is applied; negative speeds are retained as-is and simply land in the lowest quintile.

ii.
```python
def interp_vector_finite(ts, values, q):
    ok = np.isfinite(ts) & np.isfinite(values)
    if ok.sum() < 2 or q[0] < ts[ok][0] or q[-1] > ts[ok][-1]:
        return None
    return np.interp(q, ts[ok], values[ok])
...
run=interp_vector_finite(rts,rvs,q)
```

iii. Step 5 mapping/Key Decision 11: interpolate onto the common grid and "Require finite interpolated running at every retained grid point." Step 10 Check 3(c): all streams are resampled using their explicit timestamps rather than assumed shared frame indices.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-frequency (quintile) bins, with edges fitted **per experiment/session**, not globally. All retained trials of that experiment are concatenated, `np.quantile` at 0.2/0.4/0.6/0.8 gives four edges, and `np.searchsorted(edges, values, 'right')` assigns codes 0–4. The same edges are applied to every trial of the session (no per-trial fitting), and the edges are saved per session in `metadata['session_info'][i]['running_quintile_edges']`. Verified occupancy is 0.200 ± 0.00002 per class.

ii.
```python
def quantile_codes(values, valid_mask, n=5):
    """Session-level equal-frequency bins, robust to repeated quantiles."""
    fit = values[valid_mask]
    if fit.size == 0 or not np.all(np.isfinite(fit)):
        raise ValueError('No finite values for quantile fit')
    edges = np.quantile(fit, np.arange(1,n)/n)
    return np.searchsorted(edges, values, side='right').astype(np.int16), edges
...
rv=np.concatenate(allrun); pv=np.concatenate(allpupil)
rcodes,redges=quantile_codes(rv,np.ones(rv.size,bool))
...
rr=rcodes[roff:roff+n]; roff+=n
```

iii. Step 5 Key Decision 8: "Compute quintile edges separately per experiment from finite values sampled over retained trials. Use rank/quantile edges with duplicate-edge handling; this minimizes apparatus/session scaling effects and targets equal occupancy. Apply one set of edges to every trial in that session to avoid trial leakage."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated at exactly the same absolute bin centers `q` as the neural data, per trial, so it is aligned by construction; any trial whose grid extends past the running-timestamp support is dropped rather than extrapolated. The quintile code vector is sliced back out in the same trial order used to build the concatenated fitting array (`roff` running offset), then stacked as row 2 of the output matrix.

ii.
```python
if len(q)<2 or ... or q[0]<rts[0] or q[-1]>rts[-1]:
    dropped['support']+=1; continue
run=interp_vector_finite(rts,rvs,q)
...
records.append((ii,q,run,pup,int(np.argmax(flags))))
allrun.append(run)
...
for ii,q,run,pup,outcome in records:
    rr=rcodes[roff:roff+n]; roff+=n
```

iii. Step 4/Step 10: each asynchronous stream is resampled onto the shared absolute-time grid using its own timestamps ("no frame-index alignment across clocks"); the `--show-processing` plots overlay the continuous running trace and its quintile code on the trial time axis to make misalignment visible.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (the *processed* pupil-ellipse area, which the release sets to NaN on `likely_blink` frames) together with `acquisition/EyeTracking/eye_tracking/timestamps`. `area_raw`, `width`, and `height` are not used. Three experiments without any eye-tracking series are excluded from the dataset entirely.

ii.
```python
ets=np.asarray(f['acquisition/EyeTracking/eye_tracking/timestamps'][:],float)
area=np.asarray(f['acquisition/EyeTracking/pupil_tracking/area'][:],float)
diam=2*np.sqrt(np.maximum(area,0)/np.pi)
...
NO_PUPIL = {795953296,806456687,833631914}
paths=[p for p in paths if exp_id(p) not in NO_PUPIL]
```

iii. Step 4: "Processed area is NaN during likely blinks … Convert processed pupil area to equivalent-circle diameter `2*sqrt(area/pi)`, preserving NaN. This is orientation-invariant and uses reference processing." Step 5 Key Decision 9: "Processed area is preferred because reference blink handling has already set bad frames to NaN." Step 5 Key Decision 10 explains the three-file exclusion: "pupil is a required output and no defensible value can be fabricated."

## 6-b. How is `output` *Pupil diameter* processed?

i. Area → equivalent-circle diameter `2*sqrt(max(area,0)/pi)`; then linear interpolation onto the trial's 100 ms bin centers using only finite (non-blink) anchors, so blink gaps are bridged by the surrounding good samples; no extrapolation beyond the finite eye support (such trials are dropped); then equal-frequency quintile coding, identical to running speed.

ii.
```python
diam=2*np.sqrt(np.maximum(area,0)/np.pi)
...
pup=interp_vector_finite(ets,diam,q)
if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)):
    dropped['nonfinite']+=1; continue
...
pcodes,pedges=quantile_codes(pv,np.ones(pv.size,bool))
```

iii. Step 5 Key Decision 9: "Equivalent-circle diameter is a standard scalar diameter preserving ellipse area. Interpolate only between finite anchors; do not extrapolate outside eye support." Step 5 Key Decision 10: "linearly bridge reference-marked blink gaps using surrounding finite samples" (96.45% of samples were found finite in files that have eye data).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical scheme to running speed: five equal-frequency bins with edges fitted per experiment over all retained trials of that experiment, applied uniformly to every trial of the session, edges saved in `metadata['session_info'][i]['pupil_diameter_quintile_edges']`. Occupancy verified at 0.200 ± 0.00002 per class. Labels: `Q1_smallest … Q5_largest`.

ii.
```python
pv=np.concatenate(allpupil)
pcodes,pedges=quantile_codes(pv,np.ones(pv.size,bool))
...
pp=pcodes[poff:poff+n]; poff+=n
...
'output_values':[..., ['Q1_smallest','Q2','Q3','Q4','Q5_largest'], ...]
```

iii. Same rationale as running (Step 5 Key Decision 8): per-session equal-frequency binning "minimizes apparatus/session scaling effects and targets equal occupancy," which is particularly relevant for pupil, where absolute camera-pixel area is not comparable across rigs/sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same mechanism as running speed: evaluated at the same absolute bin centers `q` as the neural interpolation, with trials dropped rather than extrapolated when the grid exceeds the finite eye support. Row 3 of the `(5, n)` output matrix; length asserted equal to the neural matrix.

ii.
```python
pup=interp_vector_finite(ets,diam,q)   # q = same bin centers used for neural
...
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
assert neu.shape[1]==out.shape[1]==n
```

iii. Step 10 Check 3(c) and Step 12: pupil, like running, is aligned via its own timestamps onto the shared absolute-time trial grid; the `--show-processing` panel plots the continuous diameter and its quintile code against the same trial time axis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table, in the fixed order `['hit','miss','false_alarm','correct_reject']`, stacked into a 4 × n_trials flag matrix. Exclusivity is asserted per trial (exactly one flag true), and the conversion raises if violated.

ii.
```python
OUTCOME_KEYS = ['hit','miss','false_alarm','correct_reject']
...
outcome_flags=np.vstack([np.asarray(tr[k][:],bool) for k in OUTCOME_KEYS])
...
flags=outcome_flags[:,ii]
if flags.sum()!=1: raise AssertionError(f'{eid} trial {ii}: outcome not exclusive')
records.append((ii,q,run,pup,int(np.argmax(flags))))
```

iii. Step 4: "SDK exposes mutually exclusive outcomes … Every eligible row has exactly one outcome; totals 15,936 hit, 58,602 miss, 932 false alarm, 9,760 correct reject … Encode four outcome classes in this order and retain only non-aborted, non-auto-rewarded go/catch rows." The exclusivity check was run over all 85,230 eligible rows in Step 4 before being encoded as a runtime assertion.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. `argmax` over the four flags gives an integer code 0–3; because the target format requires a rectangular `(n_output, n_timepoints)` matrix, the static code is broadcast across all time bins of the trial. `metadata['output_static'] = [False,False,False,False,True]` records that the variable is semantically per-trial. Full-dataset (time-weighted) distribution: hit 0.182, miss 0.692, false_alarm 0.010, correct_reject 0.115 — miss-dominated because passive sessions were retained.

ii.
```python
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
...
'output_names':[...,'trial_outcome'],
'output_values':[...,OUTCOME_KEYS],
'output_static':[False,False,False,False,True],
```

iii. Step 5 Key Decision 12: "Outcome is static by specification. If the supplied validator requires one rectangular output matrix, repeat the static class over time while documenting `output_static=[False,False,False,False,True]` in metadata; semantically it remains per-trial." Step 4 justifies keeping passive sessions (and hence the miss-heavy distribution): "Retain explicit eligible rows per task; do not infer trial validity from session name. Document class imbalance."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases are handled, all by exclusion rather than imputation: (1) experiments with no eye-tracking series at all — the three ids are hard-coded in `NO_PUPIL` and skipped before any processing; (2) blink frames — NaN in the processed pupil area, dropped as interpolation anchors so the gap is bridged linearly from surrounding good frames; (3) trials whose 100 ms grid extends beyond the ophys or running timestamp support, or yields any non-finite running/pupil value — dropped and counted in a `dropped` dict printed per experiment; (4) trials shorter than two bins — dropped; (5) structural surprises — hard failures rather than silent fallbacks: unexpected stimulus template mapping (`raise ValueError`), non-exclusive outcome flags (`raise AssertionError`), a change pulse on a catch trial (`assert`), out-of-range image/quintile codes (`assert`), and an experiment left with <2 usable trials (`raise ValueError`). Nothing is imputed or fabricated. In the full run, zero trials were dropped in the 281 retained experiments; the only losses were the 3 no-pupil experiments (276 neurons, 917 eligible trials), fully accounted for in the notes.

ii.
```python
NO_PUPIL = {795953296,806456687,833631914}
paths=[p for p in paths if exp_id(p) not in NO_PUPIL]
...
dropped={'support':0,'nonfinite':0}
if len(q)<2 or q[0]<nts[0] or q[-1]>nts[-1] or q[0]<rts[0] or q[-1]>rts[-1]:
    dropped['support']+=1; continue
if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)):
    dropped['nonfinite']+=1; continue
if flags.sum()!=1: raise AssertionError(...)
if len(records)<2: raise ValueError(f'{eid}: fewer than 2 usable trials')
...
print(f"{eid}: cells=... trials={len(records)}/{eligible.sum()} dropped={dropped} ...")
...
'excluded_no_pupil_experiments':sorted(NO_PUPIL),
```

iii. Step 5 Key Decision 10: "Exclude the three experiments with no pupil series … because pupil is a required output and no defensible value can be fabricated. Within remaining experiments, linearly bridge reference-marked blink gaps using surrounding finite samples; drop an individual trial only if any grid point lies outside finite eye support or interpolation cannot produce a finite value. Report all losses." Step 10: "Three files lack pupil tracking: excluded and fully accounted (276 neurons, 917 eligible trials); fabrication would violate reference handling and required pupil output." Note that these exclusions are hard-coded ids, and that a session-level failure raises out of `main()` (there is no `try/except` around `process_experiment`), so the pipeline aborts rather than skipping such a session — this never triggered because Step 2 had pre-verified that every experiment has ≥39 eligible trials with in-support bounds.

## 9-a. What are the most time-consuming steps of the code?

i. The code instruments itself: `process_experiment` records `time.perf_counter()` per experiment and `main` prints total wall time. Full conversion took 251.8 s for 281 experiments; per-experiment times were 0.27–1.83 s and scale with (#trials × #neurons). The dominant cost is reading each experiment's dF/F matrix out of HDF5 into a float32 NumPy array (the single largest I/O in the file, up to 666 neurons × ~150k frames) plus the per-trial `interp_matrix` gather over that matrix; the remaining ~50 s of the 251.8 s total is pickle serialization of the 4.5 GB output. The AI's own diagnosis (Step 6/Step 10) attributes the cost to HDF5 reads and explicitly notes that an early draft re-read the dF/F dataset inside the trial loop.

ii.
```python
def process_experiment(path, meta_row, make_plot=False):
    eid=exp_id(path); t0=time.perf_counter(); ...
    nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)   # dominant read
    ...
    'conversion_seconds':round(time.perf_counter()-t0,3)}
    print(f"{eid}: cells=... time={info['conversion_seconds']}s",flush=True)
...
print(f'Saved {args.outpicklefile}: ... total_time={time.perf_counter()-start:.2f}s',flush=True)
```

iii. Step 6: "Initial draft converted the entire HDF5 dF/F dataset to NumPy once per trial, which would cause severe redundant I/O" → "Load each experiment's dF/F matrix once as float32, then vectorized interpolation by index/weight arrays for each trial" and "Read only metadata/behavior vectors needed for processing; never load large stimulus image templates." Step 7 estimated ~3–6 min for the full run, and the actual 251.8 s matched.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial Python loops remain in `process_experiment`: the collection loop (grid construction + running/pupil interpolation + outcome extraction per trial) and the emission loop (`interp_matrix` for neural, `image_labels`, change pulse, output stacking per trial). Both could be collapsed by concatenating all trial grids into one long `q` vector, running a single `interp_matrix`/`np.interp`/`image_labels` call over it, and then splitting with `np.split` at trial boundaries — the same trick the code already uses for the quintile codes via the `roff`/`poff` offsets. The change pulse and outcome rows could likewise be built with a single fancy-index assignment over all trials. The dominant inner work is already vectorized: `interp_matrix` computes index/weight arrays for all bins × all neurons at once, and `image_labels` uses a vectorized `searchsorted` (with one residual Python list comprehension over the selected bins to map control indices to image codes, which could be a NumPy lookup-table gather).

ii.
```python
for ii in inds:                                   # loop 1: one trial at a time
    q=trial_grid(starts[ii],stops[ii])
    run=interp_vector_finite(rts,rvs,q); pup=interp_vector_finite(ets,diam,q)
...
for ii,q,run,pup,outcome in records:              # loop 2: one trial at a time
    n=len(q); neu=interp_matrix(nts,nds,q).T.astype(np.float32)
    img=image_labels(q,sts,sval,lookup)
```
```python
out[ok] = np.array([lookup.get(int(v),0) for v in stim_val[jj]], dtype=np.int16)  # residual comprehension
```

iii. The AI documented only the vectorization it did perform (Step 6: "vectorized interpolation by index/weight arrays for each trial"; "Fit quantile edges once per experiment from concatenated retained samples"). It did not flag the remaining per-trial loops, presumably because measured throughput (0.27–1.83 s/experiment, 251.8 s total) was far inside the 15-minute budget set by the instructions, so further optimization had no practical payoff.

## 9-c. What processing does the code repeat multiple times?

i. The notes claim no repeated work ("One dF/F read/session; vectorized interpolation"), and the big items are indeed done once per experiment (dF/F read, area→diameter conversion, stimulus lookup, quantile fit). Two genuine repetitions remain, both inside the per-trial loop and both undocumented: (1) `interp_vector_finite` recomputes `np.isfinite(ts) & np.isfinite(values)` and re-materializes `ts[ok]`/`values[ok]` for the *entire* session-length running (~270k samples) and eye (~136k samples) arrays on every trial — i.e. ~300× per experiment, when the finite mask and filtered arrays could be computed once per experiment; (2) `np.interp`/`np.searchsorted` re-binary-search the full-session arrays per trial (inherent to the per-trial loop, cheap). Additionally the per-trial grid `q` is built once and cached in `records`, so that is not repeated, and each NWB file is opened exactly once.

ii.
```python
def interp_vector_finite(ts, values, q):
    ok = np.isfinite(ts) & np.isfinite(values)        # recomputed for every trial
    if ok.sum() < 2 or q[0] < ts[ok][0] or q[-1] > ts[ok][-1]:
        return None
    return np.interp(q, ts[ok], values[ok])           # ts[ok]/values[ok] rebuilt every trial
...
for ii in inds:
    run=interp_vector_finite(rts,rvs,q); pup=interp_vector_finite(ets,diam,q)
```

iii. The AI's stated position (Step 6, Step 10) is that redundancy was eliminated: the only repetition it identified and fixed was the per-trial re-read of the dF/F dataset. The residual mask recomputation was not identified; it costs O(n_trials × n_samples) per experiment but was immaterial at the observed runtime.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items: (1) the continuous `run`/`pup` float vectors are stored in `records` and concatenated into `allrun`/`allpupil`, doubling that memory, and after the quintile fit they are used only for the optional plot — only the integer codes reach the pickle; (2) `raw_plot` is assembled on every experiment even when `--show-processing` is off; (3) the pupil equivalent-circle conversion `2*sqrt(area/pi)` is computed for the whole session (including frames outside any trial) and is a strictly monotone transform of area, so it cannot change any quintile code — it only affects the human-readable edge values in metadata; (4) all five output rows are stored as `int16` although every value fits in `int8`/`uint8`, and the `(0, n)` float32 input arrays are allocated per trial (required by the target format, but the payload is empty); (5) `change_time`/`is_change` are read for all trials including ineligible ones; (6) per-session `session_info` (including quintile edge lists) is written into metadata and is not consumed by the decoder — though it is valuable for provenance and was used by the AI's own audit scripts. None of this materially affects runtime; the 4.5 GB pickle is dominated by the float32 dF/F payload itself.

ii.
```python
records.append((ii,q,run,pup,int(np.argmax(flags))))   # continuous run/pup kept
allrun.append(run); allpupil.append(pup)               # and copied again
...
if raw_plot is None: raw_plot=(q,neu,img,ch,run,pup,rr,pp)   # built even when not plotting
...
out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])   # int16 for 0-16 values
neural.append(neu); inputs.append(np.empty((0,n),dtype=np.float32))
```

iii. The AI documented memory awareness (Step 5 Key Decision 13: "Neural float32; outputs compact integer arrays; empty inputs shape `(0,T)`. Estimated 7.35M timepoints makes full conversion practical but memory-intensive, so process/write session-by-session in one pass and avoid loading raw image templates") but did not list any residual discarded computation; its Step 6 efficiency review focused solely on the redundant dF/F read.
