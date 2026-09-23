# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively discovers every `.nwb` file under `/app/data`, sorts the paths, and opens each file once with `pynwb.NWBHDF5IO`. Each file supplies its session's trials, units, events, behavioral time series, subject, and identifier. Full mode processes all discovered files; sample mode stops after two retained sessions.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb'))
for p in files:
    res=convert_session(p,make_plot=args.show_processing and len(sessions)<2)
```
```python
with NWBHDF5IO(str(path),'r',load_namespaces=True) as io:
    nwb=io.read(); tr=nwb.trials
```

iii. The notes say there are 174 NWB files, one per session, and emphasize that all access uses the required `pynwb` API, never `h5py`. Sorting makes processing deterministic.

## 1-b. How are the data split into subjects?

i. The subject ID is read from `nwb.subject.subject_id` for each retained session. At assembly, unique IDs are sorted and each session receives an integer index into that list.

ii.
```python
subject=str(nwb.subject.subject_id)
```
```python
subjects=sorted({x['subject'] for x in sessions})
smap={v:i for i,v in enumerate(subjects)}
'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int64)
```

iii. The AI treats the NWB subject field as authoritative and reports 28 unique subjects, consistent with its data exploration and the dataset summary.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. A converted session is appended as one element of `neural`, `input`, and `output`; files with insufficient retained units or trials are skipped.

ii.
```python
res=convert_session(p,...)
if res is None: ...; continue
sessions.append(res)
```
```python
'neural':[x['neural'] for x in sessions]
```

iii. The notes state that the dataset has one electrophysiology/behavior session per NWB. The resulting 173 retained sessions reflect one skipped file with no classifier-good units.

## 1-d. How are the data split into trials?

i. Trial rows are paired positionally with go-cue timestamps up to `min(len(trials), len(go))`. A trial is structurally eligible when its go cue has a preceding sample/tone event. Subsequent QC selects trial-row indices, and each retained row becomes one trial array.

ii.
```python
n=min(len(tr),len(go))
ix=np.searchsorted(sample,go[:n],side='right')-1
structural_idx=np.flatnonzero(ix>=0)
```

iii. The AI describes these as completed trials with an indexed go cue and preceding tone. It found one go cue per trial during exploration, though the code tolerates unequal counts by truncating instead of asserting equality.

## 1-e. How are trials filtered based on quality controls?

i. Starting from structurally valid trials, the AI maps every classifier-good unit's `obs_intervals` to behavioral rows, applies that unit's `is_good_trials`, and intersects valid trial IDs across all selected units. It then removes any retained trial for which no selected unit fired anywhere in the four-second decoder window. It keeps early-lick, stimulation, auto-water, free-water, hit, miss, and ignore trials, and requires at least two survivors.

ii.
```python
common=set(map(int,structurally_valid_trials))
for j in units:
    obs=np.asarray(nwb.units['obs_intervals'][j],dtype=np.float64)
    valid=np.asarray(nwb.units['is_good_trials'][j],dtype=bool)
    ...
    common.intersection_update(map(int,mapped[valid & (mapped>=0)]))
```
```python
neural_valid=np.any(rates>0,axis=(1,2))
trial_idx=trial_idx[neural_valid]
```

iii. The notes justify intersection as avoiding fabricated values and the all-zero removal as eliminating recording gaps. They explicitly retain task categories needed by the decoder. This differs from the reference's use of observed intervals plus explicit `free_water == 0` filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each selected unit's ragged `units['spike_times']`, with `units['classification']` determining selected units and `BehavioralEvents/go_start_times` defining per-trial bin edges.

ii.
```python
spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
edge_matrix=go[:,None]+EDGES_REL[None,:]
```

iii. The AI states that spikes are the NWB neural representation and uses the QC classifier selected by the paper's preprocessing code.

## 2-b. How is the `neural` data processed?

i. For each unit, `searchsorted` obtains cumulative spike indices at all trial/bin edges. Adjacent differences give counts, which are divided by 0.05 seconds to produce firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
for ui,sp in enumerate(spikes):
    rates[:,ui,:]=np.diff(
        np.searchsorted(sp,edge_matrix,side='left'),axis=1
    ).astype(np.float32)/BIN_S
```

iii. The notes say Hz matches the reference's `sliding_histogram(..., rate=True)`, while the mandated non-overlapping 50-ms bins override the paper code's sliding-window parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose NWB `classification` is exactly `'good'` are retained. Sessions with no such units are dropped. The AI also uses selected units' observation/trial-validity metadata to curate trials, but it does not further discard units by metric thresholds.

ii.
```python
cls=arrcol(nwb.units,'classification').astype(str)
units=np.flatnonzero(cls=='good')
if units.size==0:
    return units, np.asarray([],dtype=int)
```

iii. The AI identifies this as the classifier QC mode used by the reference analysis and treats released NWB classifier labels as authoritative; it reports 69,453 good units and one un-QC'd session dropped.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges from 2.5 s before through 1.5 s after each go cue are formed by adding relative edges to each trial's absolute go timestamp. Spikes are counted directly against those edges.

ii.
```python
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=np.float64)
edge_matrix=go[:,None]+EDGES_REL[None,:]
```

iii. The notes explain that NWB spikes and events share one absolute clock, so adding offsets to go-cue timestamps performs the required alignment without interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output contains 80 non-overlapping 50-ms bins spanning edges -2.5 to +1.5 s relative to go onset. Raw spike times are binned directly into this grid; there is no later rebinning.

ii.
```python
BIN_S=0.050
OFF_START=-2.5
OFF_END=1.5
EDGES_REL=np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S)
CENTERS_REL=(EDGES_REL[:-1]+EDGES_REL[1:])/2
```

iii. This is the decoder specification. The AI explicitly notes that it replaces the method paper's 40-ms sliding windows and 3.4-ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, `go_start_times`, and the shared bin-center offsets. Each go cue is paired with the latest sample onset at or before it.

ii.
```python
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
ix=np.searchsorted(sample,go[:n],side='right')-1
tone=sample[ix[trial_idx]]
```

iii. The AI notes that early licks can replay task epochs, so the latest preceding sample onset is the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For every bin, the absolute center time is computed and the selected tone onset is subtracted, producing a continuous elapsed-time vector in seconds.

ii.
```python
centers=g+CENTERS_REL
inp[0]=(centers-to).astype(np.float32)
```

iii. The notes follow the explicit decoder requirement for continuous time from tone onset despite a generic format suggestion about binary onset indicators.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same go-aligned 50-ms bin centers as the neural bins, so element `k` describes the center of neural bin `k`.

ii.
```python
centers=g+CENTERS_REL
inp[0]=(centers-to).astype(np.float32)
```

iii. The AI reports independent comparisons with direct NWB calculations, differing only by float32 rounding.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the absolute timestamp arrays `BehavioralEvents/photostim_start_times` and `photostim_stop_times`.

ii.
```python
pstart=np.asarray(ev['photostim_start_times'].timestamps[:],dtype=np.float64)
pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],dtype=np.float64)
```

iii. The AI chose event timing rather than the trials-table onset/duration strings, describing the event stream as a direct representation of stimulation timing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each bin center, the code finds the latest stimulation start and marks the bin 1 if that center precedes the corresponding stop; otherwise it is 0. Intervals are treated as half-open `[start, stop)`.

ii.
```python
jj=np.searchsorted(pstart,centers,side='right')-1
valid=jj>=0; stim=np.zeros(N_TIME,dtype=bool)
stim[valid]=centers[valid] < pstop[jj[valid]]
inp[1]=stim.astype(np.float32)
```

iii. The code comment states that a center is true when any half-open stimulation interval contains it. Sessions without events receive all zeros.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation events are evaluated at the same absolute go-aligned bin centers used for neural activity.

ii.
```python
centers=g+CENTERS_REL
jj=np.searchsorted(pstart,centers,side='right')-1
```

iii. The notes say the shared NWB clock makes additional resampling or offset correction unnecessary.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is reconstructed from trials-table `trial_instruction` and `outcome`: a hit uses the instructed direction, a miss uses the opposite direction, and ignore means no lick.

ii.
```python
instruction=arrcol(tr,'trial_instruction').astype(str)[trial_idx]
outcome_s=arrcol(tr,'outcome').astype(str)[trial_idx]
```
```python
def trial_choice(instruction,outcome):
    if outcome=='ignore': return 2
    if outcome=='hit': return 0 if instruction=='left' else 1
    if outcome=='miss': return 1 if instruction=='left' else 0
```

iii. The AI found no direct choice column and reports 100% agreement between this reconstruction and the first post-go lick in inspected sessions.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choices are coded left=0, right=1, no lick=2 and repeated over all 80 time bins in output row 0.

ii.
```python
out[0]=trial_choice(instruction[k],outcome_s[k])
```
```python
'output_values':[['left','right','no lick'], ...]
```

iii. The AI repeats static labels because the supplied validator expects a uniform output time axis.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` column.

ii.
```python
outcome_s=arrcol(tr,'outcome').astype(str)[trial_idx]
```

iii. The raw values already have the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to ignore=0, miss=1, hit=2 and the value is repeated across 80 bins in row 1.

ii.
```python
out[1]={'ignore':0,'miss':1,'hit':2}[outcome_s[k]]
```

iii. The mapping follows the requested category order; repetition gives a uniform output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` column.

ii.
```python
early_s=arrcol(tr,'early_lick').astype(str)[trial_idx]
```

iii. The NWB table explicitly stores the requested binary behavioral label, and the AI deliberately retains early-lick trials.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Values are mapped from `no early` to 0 and `early` to 1, then repeated across 80 bins in output row 2.

ii.
```python
out[2]={'no early':0,'early':1}[early_s[k]]
```

iii. The mapping follows the requested no/yes ordering and the uniform output representation.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The preferred source is `BehavioralTimeSeries/Camera0_side_TongueTracking`; if absent, the lexicographically first tongue-tracking series is used. Its timestamps, data column 1 (y), and data column 2 (tracking likelihood) are used.

ii.
```python
keys=sorted(k for k in series if 'TongueTracking' in k)
key='Camera0_side_TongueTracking' if 'Camera0_side_TongueTracking' in keys else keys[0]
times=np.asarray(ts.timestamps[:],dtype=np.float64)
data=np.asarray(ts.data[:],dtype=np.float32)
y=data[:,1]; likelihood=data[:,2]
```

iii. The AI selected the side-camera tongue trace for consistency and uses likelihood to distinguish visible/reliable positions from tracking noise.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI marks raw frames visible when y and likelihood are finite and likelihood is at least 0.90. It computes the session 40th/60th percentiles from all visible raw-frame y values. For each decoder bin center it selects the nearest camera frame; the frame must lie within video coverage and within 10 ms of the center. That frame is then categorized, otherwise the bin is not visible.

ii.
```python
visible=np.isfinite(y)&np.isfinite(likelihood)&(likelihood>=LIKELIHOOD_THRESHOLD)
thresholds=tuple(np.percentile(y[visible],[40,60]).tolist())
```
```python
q=nearest_indices(vt,centers)
covered=(centers>=vt[0])&(centers<=vt[-1])&(np.abs(vt[q]-centers)<=0.010)
vis=covered&...&(vl[q]>=LIKELIHOOD_THRESHOLD)
```

iii. The AI justifies 0.90 from the strongly bimodal likelihood distribution and interprets “over the session” as percentiles of all confidence-valid frames. It uses nearest-frame sampling at bin centers rather than averaging all visible frames within each 50-ms bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session p40 and p60 are computed over visible raw-frame y. Visible values below p40 become 0; values from p40 through p60 become 1; values above p60 become 2. Missing, uncovered, stale-nearest-frame, low-confidence, or nonfinite values become 3.

ii.
```python
tc=np.full(N_TIME,3,dtype=np.int64)
tc[vis & (yy<p40)]=0
tc[vis & (yy>=p40)&(yy<=p60)]=1
tc[vis & (yy>p60)]=2
```

iii. The AI follows the specified inequality boundaries and reserves the fourth category for not visible. Its key interpretive choice is deriving thresholds from raw visible frames, not from session-wide 50-ms visible-frame bin means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera data use the same absolute clock. For each absolute neural-bin center, the nearest camera timestamp is selected, subject to coverage and a 10-ms maximum distance.

ii.
```python
centers=g+CENTERS_REL
q=nearest_indices(vt,centers)
covered=(centers>=vt[0])&(centers<=vt[-1])&(np.abs(vt[q]-centers)<=0.010)
```

iii. The AI says this directly aligns each tongue sample with the corresponding neural-bin center and avoids extrapolation. Unlike the reference, it does not aggregate all camera frames falling inside each neural bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units or fewer than two valid trials are skipped. Missing/misaligned observation metadata raises an error; interval mappings without overlap are rejected. Trials with all-zero selected-unit activity in the requested window are dropped. Missing tongue streams or invalid/uncovered/low-confidence frames become category 3. Blank selected anatomy raises an error. Structural go/tone mismatches are tolerated by truncating to the shorter go/trial count and excluding go cues without a preceding tone.

ii.
```python
if obs.ndim!=2 or obs.shape[0]!=valid.size:
    raise ValueError(...)
```
```python
neural_valid=np.any(rates>0,axis=(1,2))
```
```python
if not keys:
    return None,None,None,None,(np.nan,np.nan)
tc=np.full(N_TIME,3,dtype=np.int64)
```

iii. The AI's principle is to exclude trials/sessions where neural data would otherwise be fabricated, but preserve missing behavioral visibility as an explicit category. Its notes document corrections after discovering local `is_good_trials` indexing and all-zero windows.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies NWB I/O, spike histogramming/searching, accumulation/serialization of the roughly 12-GB result, and full decoder training as the main costs. Full conversion took 209 seconds after optimization.

ii.
```python
spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
for ui,sp in enumerate(spikes):
    rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1)/BIN_S
```

iii. The notes report that replacing per-unit/per-trial histograms with session-wide `searchsorted` reduced a two-session sample from 27.4 s to 2.8 s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural binning still loops over units but vectorizes all trials and bins within a unit. Output construction, input construction, and tongue nearest-frame classification loop over trials. The unit loop is constrained by ragged spike vectors; much of the per-trial output loop could be vectorized, though it is not the principal bottleneck.

ii.
```python
for ui,sp in enumerate(spikes):
    rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1)/BIN_S
```
```python
for k,(g,to) in enumerate(zip(go,tone)):
    ...
```

iii. The AI explicitly optimized away its initial per-unit/per-trial histogram approach. It retained clear per-trial assembly logic after the dominant neural operation was vectorized across trials.

## 10-c. What processing does the code repeat multiple times?

i. The trial loop repeatedly computes absolute centers (`g + CENTERS_REL`), searches the same session photostimulation arrays, searches the same camera timestamp array, allocates trial arrays, and writes static output values across 80 bins. Across sessions, fixed mappings and grids are reused rather than recomputed.

ii.
```python
for k,(g,to) in enumerate(zip(go,tone)):
    centers=g+CENTERS_REL
    jj=np.searchsorted(pstart,centers,side='right')-1
    q=nearest_indices(vt,centers)
```

iii. The notes mainly emphasize single-pass file processing and do not call these per-trial operations problematic; they are repeated because alignment is trial-specific.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal conversion, little derived processing is discarded. `trial_idx` is updated after zero-neural filtering but is used only for its length thereafter. The code also constructs extensive per-session audit metadata that the decoder does not consume. With `--show-processing`, it retains plot payloads and creates diagnostic plots, which are optional and not used for training.

ii.
```python
trial_idx=trial_idx[neural_valid]
```
```python
if make_plot:
    plot_payload=(go,tone,pstart,pstop,vt,vy,vl,p40,p60,neural,...)
```

iii. The AI regards audit metadata and plots as validation/documentation rather than waste. Its notes state that the plotting path is optional and that computed decoder fields are preserved in the output.
