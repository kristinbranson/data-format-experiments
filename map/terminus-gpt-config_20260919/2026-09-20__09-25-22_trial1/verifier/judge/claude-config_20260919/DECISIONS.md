# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB (HDF5) file per session under `/app/data/sub-<id>/`. The AI finds every session with a single recursive glob over the data root and sorts it, giving a deterministic file order (174 files). Rather than using `pynwb`, it opens each file directly with `h5py` and reads the raw HDF5 datasets, reconstructing NWB's ragged (VectorIndex/VectorData) layout by hand via `ragged_bounds`. Each file is opened exactly once inside a `with` block, and everything needed for that session — units (`classification`, `anno_name`, `spike_times`, `obs_intervals`, `is_good_trials`), the trials table (`intervals/trials/*`), behavioural events (`acquisition/BehavioralEvents/*`) and the side-camera tongue tracking (`acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`) — is pulled from that one handle. Byte-string columns are decoded with a helper. Results are accumulated session-by-session in a list and assembled at the end.

ii.
```python
DATA_ROOT = Path('/app/data')
...
files=sorted(DATA_ROOT.rglob('*.nwb')); files=files[:2] if args.sample else files
results=[]
for i,p in enumerate(files):
    r=convert_session(p,args.show_processing and len(results)<2)
    if r is not None: results.append(r); print(...)
    else: print(f'[{i+1}/{len(files)}] skipped {p.name}',flush=True)
```

```python
def decode(a):
    return np.asarray([x.decode(errors='replace') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in a])

def ragged_bounds(index):
    index = np.asarray(index, dtype=np.int64)
    return np.r_[0, index[:-1]], index
```

```python
def convert_session(path, make_plot=False):
    t0=time.time()
    with h5py.File(path,'r') as f:
        good,valid=curate(f)
        ...
        be=f['acquisition/BehavioralEvents']; go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` contains 174 NWB 2.x/HDF5 files organized in subject/session directories... Every NWB has `intervals/trials` with 15 consistent columns... `units` contains raw spike times plus waveform and spike-sorting QC metrics, `unit_quality`, atlas annotation `anno_name`, electrode links, observation intervals, and `is_good_trials`." The AI first inspected the files with `pynwb` during exploration (Step 2) and then chose direct `h5py` access for the converter, noting in Step 6 that "The script uses direct h5py access, reconstructs NWB ragged arrays, validates unit/trial coverage... NWBs are opened once per session; arrays are preallocated." The stated motivation was speed and avoiding redundant I/O. The resulting counts (174 files, 28 subjects, 94,990 trials, 272,227 raw units) were cross-checked against the reference texts in Steps 3-4.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the **file name**, not from the NWB `subject` group: the AI applies the regex `sub-([^_]+)` to `path.name`, yielding numeric ids such as `'440956'`. At assembly, `subjects` is the sorted set of unique ids and `subject_idx` is each session's index into that list. This produces 28 subjects with 3-10 sessions each, matching the dandiset.

ii.
```python
subject=re.search(r'sub-([^_]+)',path.name).group(1)
```

```python
subjects=sorted({r[4] for r in results}); smap={x:i for i,x in enumerate(subjects)}
data={... 'subjects':subjects,'subject_idx':np.array([smap[r[4]] for r in results],np.int64), ...}
```

iii. CONVERSION_NOTES Step 5 maps "NWB subject ID → `subjects`, `subject_idx` | Unique sorted strings and session index | Native NWB metadata | 28 subjects expected." The AI treats the directory/file prefix as the subject id because the DANDI layout derives `sub-<id>` directly from `nwb.subject.subject_id`; the two are identical in this dataset (verified: converted subject list is `440956 … 484677`, the same 28 numeric ids the NWB `subject` group carries).

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is performed. Session order in the output is the sorted recursive glob order, which (because the filename embeds `ses-<YYYYMMDD>T<HHMMSS>`) is chronological within each subject. Per-session provenance is recorded in `metadata['session_info']` as a dict per session carrying the file name, the retained source trial indices, trial/neuron counts, tongue percentile edges and the conversion time. 173 of 174 sessions reach the output; one is dropped (see 2-c).

ii.
```python
files=sorted(DATA_ROOT.rglob('*.nwb')); files=files[:2] if args.sample else files
```

```python
info={'file':path.name,'source_trial_indices':inds.tolist(),'n_source_trials':len(valid),
      'n_retained_trials':len(inds),'n_neurons':len(good),'tongue_p40':float(p40),
      'tongue_p60':float(p60),'seconds':time.time()-t0}
```

```python
'metadata':{... 'session_info':[r[5] for r in results]}
```

iii. Step 2 of CONVERSION_NOTES records the one-file-per-session layout, so the file boundary is the session boundary. Step 4 resolves the 174-vs-173 discrepancy: "174 NWBs; one has no classifier-good units | 173 analyzed sessions | Filter `classification == good`; exclude the one session with zero retained units. This yields exactly 173 sessions."

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`, one row per behavioural trial, with the go cue for each trial taken from `BehavioralEvents/go_start_times/timestamps`. The AI verified during exploration (trajectory Step 34) that "The trial table and go-event count match exactly for all 174 sessions, confirming 94,990 genuine behavioral trials with no duplicated trial rows", so row *i* of the trials table corresponds to go cue *i*. The converter then works with an index array `inds` of surviving trial rows and uses it to slice both the go-cue vector and every trials-table column.

ii.
```python
trial_st=f['intervals/trials/start_time'][()]
go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][()]
...
valid=np.ones(len(go), bool)
```

```python
inds=np.where(valid)[0]
if len(inds)<2: return None
be=f['acquisition/BehavioralEvents']; go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
...
trial=f['intervals/trials']; instruction=decode(trial['trial_instruction'][()])[inds]
outcome_s=decode(trial['outcome'][()])[inds]; early_s=decode(trial['early_lick'][()])[inds]
```

iii. CONVERSION_NOTES Step 4 documents the check: "Trial validity … Ragged `obs_intervals` and `is_good_trials`; some probes cover only a subset". The AI explicitly investigated whether `is_good_trials` widths that are smaller than the trial table implied duplicated trial rows and concluded they did not: "The trial table and go-event count match exactly for all 174 sessions, confirming 94,990 genuine behavioral trials with no duplicated trial rows." Other epoch events (`sample_start_times`, `delay_start_times`) were found to have more entries than trials because early licks replay the sample/delay epoch, so only the go cue is used as the per-trial anchor.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at absence/validity of spike data; no behavioural quality filter is applied.

1. **Observation coverage + manual per-unit validity.** For every classifier-good unit the AI reconstructs that unit's ragged `obs_intervals` rows, maps each interval's start to the nearest trial `start_time` (erroring out if the match is worse than 1e-5 s), and marks those trials valid *only if* the unit's `is_good_trials` flag is also true. The per-unit masks are then **intersected across all good units** (`valid &= uv`), so a trial is kept only if every retained neuron has both an observation row and a manual "good" flag for it.
2. **All-zero population windows.** After binning, any trial where *no* retained neuron fired a single spike anywhere in the 4-s window is dropped as a missing neural segment. (I verified this filter empirically removes exactly the `free_water` trials — in `sub-484674_ses-20210509T131820` all 31 free-water trials have zero spikes across all units and are the only all-zero trials.)

A session is dropped if fewer than 2 trials survive either stage. Early-lick, `ignore`/miss, auto-water and photostimulation trials are all deliberately retained. Net effect: 94,990 raw → 92,801 after (1) → **90,378** after (2).

ii.
```python
def map_observed_trials(trial_starts, intervals):
    j = np.searchsorted(trial_starts, intervals[:, 0])
    j = np.minimum(j, len(trial_starts)-1)
    alt = np.maximum(j-1, 0)
    j = np.where(np.abs(trial_starts[alt]-intervals[:,0]) < np.abs(trial_starts[j]-intervals[:,0]), alt, j)
    if len(j) and np.max(np.abs(trial_starts[j]-intervals[:,0])) > 1e-5:
        raise ValueError('Observation intervals do not map to trial starts')
    return j

def curate(f):
    u=f['units']; good=np.where(decode(u['classification'][()]) == 'good')[0]
    if len(good)==0: return good, np.zeros(len(f['intervals/trials/id']), bool)
    trial_st=f['intervals/trials/start_time'][()]
    go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][()]
    flat=u['obs_intervals'][()]; a,b=ragged_bounds(u['obs_intervals_index'][()]); manual=u['is_good_trials'][()]
    valid=np.ones(len(go), bool)
    for ui in good:
        z=flat[a[ui]:b[ui]]; m=manual[ui]
        if len(z)!=len(m): raise ValueError('Validity/interval length mismatch')
        jj=map_observed_trials(trial_st,z)
        uv=np.zeros(len(go), bool)
        uv[jj]=m  # obs_intervals rows identify recorded trials; their stop times are behavioral trial ends, not recording ends
        valid &= uv
    return good, valid
```

```python
inds=np.where(valid)[0]
if len(inds)<2: return None
...
neural=spike_rates(f,good,go)
# A completely silent population across four seconds indicates a missing neural segment.
has_neural=np.any(neural != 0, axis=(1,2))
inds=inds[has_neural]; go=go[has_neural]; abs_centers=abs_centers[has_neural]
tone=tone[has_neural]; inp=inp[has_neural]; neural=neural[has_neural]
if len(inds)<2: return None
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "reconstruct each unit's ragged `obs_intervals`, map intervals to trial starts, apply `is_good_trials`, require a mapped observation row and manual good flag for every retained unit, and intersect across retained units. This yields 92,801 trials. Observation-interval stop times are behavioral trial ends and must not be treated as neural-recording ends." Step 9 documents an explicit iteration: an earlier version that also required `go+1.5 s ≤ obs_interval stop` left only 766 miss trials, which the AI diagnosed and corrected. Step 10 Key Decision 4 justifies retaining required categories: "early, miss/ignore, auto/free-water, and photostimulation trials remain if streams are valid because they are required outputs/inputs" — i.e. the data paper's exclusion of early-lick and no-response trials is deliberately not applied because those are required decoder outputs here. The all-zero rule is justified in Step 10: "Corrected broad trial retention exposed 2,423 windows with no spikes in any of 90-923 curated neurons… They were excluded."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (with `units/spike_times_index` giving the ragged offsets), restricted to units with `units/classification == 'good'`, together with `BehavioralEvents/go_start_times/timestamps`, which sets the window for each trial.

ii.
```python
u=f['units']; flat=u['spike_times']; a,b=ragged_bounds(u['spike_times_index'][()])
for col,ui in enumerate(units):
    sp=np.asarray(flat[a[ui]:b[ui]])
```

```python
good=np.where(decode(u['classification'][()]) == 'good')[0]
```

iii. Step 1: "The recordings are extracellular electrophysiology. Neural source data are spike times and/or precomputed binned firing rates; delta-F/F is not applicable." Step 5 mapping: "`units/spike_times` → `neural` | Count spikes in 80 non-overlapping 50-ms bins from −2.5 to +1.5 s around each go cue; divide by 0.05 s to Hz… Raw spikes avoid double-smoothing." Spike times are the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit the AI vectorises across all trials at once: `searchsorted` on the sorted trial-window start times assigns every spike to a candidate trial, spikes past the window end are discarded, the within-window offset is floor-divided by the bin width to give a bin index, and a single `np.bincount` over the flattened `trial*80 + bin` code produces the whole trial × bin count matrix for that unit. Counts are accumulated in a `uint16` array, cast to `float32` and divided by 0.05 s. No smoothing, normalisation, or baseline subtraction is applied.

ii.
```python
def spike_rates(f, units, go_valid):
    """Vectorized within each unit; output trial x neuron x time float32."""
    ntr, nn=len(go_valid),len(units)
    counts=np.zeros((ntr, nn, NT), dtype=np.uint16)
    starts=go_valid+OFF_START; ends=go_valid+OFF_END
    u=f['units']; flat=u['spike_times']; a,b=ragged_bounds(u['spike_times_index'][()])
    for col,ui in enumerate(units):
        sp=np.asarray(flat[a[ui]:b[ui]])
        # windows are ordered and non-overlapping in this task
        ti=np.searchsorted(starts, sp, side='right')-1
        ok=(ti>=0)
        ti=ti[ok]; ss=sp[ok]
        ok=ss < ends[ti]
        ti=ti[ok]; ss=ss[ok]
        bi=np.floor((ss-starts[ti])/BIN_S).astype(np.int64)
        ok=(bi>=0)&(bi<NT); code=ti[ok]*NT+bi[ok]
        if len(code): counts[:,col,:]=np.bincount(code,minlength=ntr*NT).reshape(ntr,NT)
    return counts.astype(np.float32).transpose(0,1,2) / np.float32(BIN_S)
```

iii. Step 4 discrepancy table: "Neural temporal processing | Method code uses 40-ms window, 3.4-ms stride | Raw spike times available | Method paper specifies 40 ms / 3.4 ms | Task specification overrides reference here: use 50-ms-width non-overlapping bins from −2.5 to +1.5 s." Step 5: "Raw spikes avoid double-smoothing." Step 6: "Spike assignment is vectorized across all trials within each neuron using searchsorted/bincount." The comment in the code notes the correctness precondition for the `searchsorted` trial assignment ("windows are ordered and non-overlapping in this task"); I confirmed the minimum go-cue spacing across all 174 sessions is 4.58 s > 4 s, so no window overlaps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are retained; no thresholds are applied to any individual QC metric (`isi_violation`, `amplitude_cutoff`, `presence_ratio`, drift metrics, etc. are all ignored). A session with zero such units is dropped. This keeps 69,453 of 272,227 units (25.5%), mean 401.46 per session (range 90-923), and drops exactly one session (`sub-440958_ses-20190216T162508`, whose classification column is NaN for all units), giving 173 sessions.

ii.
```python
u=f['units']; good=np.where(decode(u['classification'][()]) == 'good')[0]
if len(good)==0: return good, np.zeros(len(f['intervals/trials/id']), bool)
```

```python
good,valid=curate(f)
if not len(good): return None
```

iii. Step 4 discrepancy table: "Unit quality field | Method code uses final curated/anatomical units | `unit_quality`: 154,948 good; `classification`: 69,453 good | Region-specific classifiers yielded final good units | Use `classification == good`. Notably, 8,475 classifier-good units have Kilosort label `multi`, confirming these fields are not interchangeable." Step 5 Key Decision 1: "retain only `classification == good`, the final region-specific classifier used by the paper; do not substitute Kilosort `unit_quality`." Step 9 notes the residual 490-unit gap to the white paper's 69,943: "The 490-unit difference is a release/version discrepancy."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**, taken from `BehavioralEvents/go_start_times/timestamps`. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed: the fixed relative window (−2.5 s, +1.5 s) is simply added to each trial's go-cue time to give the absolute window, and spikes are binned against those absolute boundaries.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

```python
go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
abs_centers=go[:,None]+CENTERS_REL[None,:]
```

```python
starts=go_valid+OFF_START; ends=go_valid+OFF_END
...
bi=np.floor((ss-starts[ti])/BIN_S).astype(np.int64)
```

iii. Step 3: "Task alignment is naturally defined by the auditory go cue. The delay begins 1.2 s before go cue; photoinhibition occupies the final 0.5 s of delay and ends before go." Step 10 Check 3: "Alignment: go-event timestamps define zero in both converter and reference analyses." `metadata['temporal_alignment_event']` is set to `'auditory Go cue onset'` with `off_start=-2.5`, `off_end=1.5`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, non-overlapping, 80 bins per trial spanning −2.5 s to +1.5 s relative to the go cue. The edge grid is built once at module level from `OFF_START`/`OFF_END`/`BIN_S` and asserted to have exactly 80 bins; it is reused for every trial and every session, so all trials have identical shape `(n_neurons, 80)`. There is no rebinning of pre-binned data — spike times are binned once, directly, at the target resolution (no 40 ms/3.4 ms sliding-window intermediate). `metadata['time_bin_size']` is 50.0 ms.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
NT = len(CENTERS_REL)
assert NT == 80
```

```python
'metadata':{... 'time_bin_size':50.0,'off_start':OFF_START,'off_end':OFF_END,
            'neural_measure':'50-ms spike-count firing rate (Hz)', ...}
```

iii. Step 4: "Task specification overrides reference here: use 50-ms-width non-overlapping bins from −2.5 to +1.5 s." Step 10 Check 5: "exactly 80 half-open 50-ms bins cover [−2.5,+1.5)". Step 5: binning raw spikes rather than rebinning the paper's sliding rates "avoids double-smoothing".

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times/timestamps` (the sample-epoch tone onsets), together with each trial's `intervals/trials/start_time` and its go cue. Because an early lick replays the sample epoch, a trial may contain several sample-onset events; the AI selects the **last** sample onset that lies within `[trial start, go cue]`, and raises if a trial has none.

ii.
```python
def final_tone_onsets(f, trial_indices, go):
    starts=f['intervals/trials/start_time'][()]
    sample=np.sort(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][()])
    out=np.empty(len(trial_indices), np.float64)
    for k,i in enumerate(trial_indices):
        lo=np.searchsorted(sample, starts[i]-1e-8, 'left'); hi=np.searchsorted(sample, go[i]+1e-8, 'right')
        if hi<=lo: raise ValueError(f'No sample onset for trial {i}')
        out[k]=sample[hi-1]
    return out
```

iii. Trajectory Step 38: "`sample_start_times` can outnumber trials because early licking triggers replay of the sample/delay epoch, exactly as described in methods.txt. Go and trial-end events remain one per trial. Therefore sample/tone onset cannot be matched by index; for each go cue, the relevant onset is the latest sample-start event preceding that go cue. This preserves the final replay that led to the go cue." CONVERSION_NOTES Step 5 Key Decision 5: "Sample events can outnumber trials. Select the final sample-start event between trial start and go cue; no trial lacks such an event."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: for each trial the absolute bin centres (`go + CENTERS_REL`) minus the trial's tone onset. The result is stored as `float32` in row 0 of the `(2, 80)` input array. No discretisation, clipping or normalisation is applied, so the value ranges over roughly [−1.5, 11.9] s across the dataset (large upper values arise on trials where the sample epoch was replayed many times, making `go − tone` up to ~10.4 s; I confirmed this against the raw timestamps).

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
tone=final_tone_onsets(f,inds,go_all)
inp=np.stack([abs_centers-tone[:,None],stim_series(f,abs_centers)],axis=1).astype(np.float32)
```

iii. Step 5 mapping: "Final `sample_start_times` event within each trial → `input[0]` | At each bin center, absolute time minus final sample/tone onset | Continuous seconds from tone onset, as explicitly requested. Final onset handles early-lick replay." Step 10 Check 5: "every final sample onset is selected within trial start/go bounds."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same grid the firing rates are binned on. `abs_centers` is built from the same `go` vector and the same `CENTERS_REL` offsets used to place the spike bin edges, so element *k* of the input row corresponds to the same 50 ms interval as bin *k* of the neural matrix. Both are derived from the one session-absolute clock, so no interpolation or offset correction is required.

ii.
```python
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]     # inputs
...
starts=go_valid+OFF_START                      # neural bin edges, same go vector
```

iii. Step 10 Check 6: "all neural arrays are float32 neuron×80, all inputs float32 2×80". Step 12 Check 5: "Processing plots and raw `np.allclose` checks verify go alignment, replay-aware sample onset, stimulation intervals, and tongue timestamps." The `--show-processing` plots overlay the neural raster and both input rows on a shared −2.5→+1.5 s axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The explicit optogenetic event streams `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps`, rather than the string-valued `photostim_onset`/`photostim_duration` columns of the trials table. (I verified the two are numerically identical to ~5e-13 s and agree on event count in every session, 18,588 stimulated trials in total.)

ii.
```python
def stim_series(f, abs_centers):
    be=f['acquisition/BehavioralEvents']; out=np.zeros(abs_centers.shape, dtype=bool)
    starts=be['photostim_start_times/timestamps'][()]; stops=be['photostim_stop_times/timestamps'][()]
    for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
    return out.astype(np.float32)
```

iii. Step 2: "Photostimulation: 18,588 trials have 0.5-s stimulation; 76,402 use `N/A`. Power/onset/duration are stored as strings and require explicit N/A parsing." Step 5 mapping: "`photostim_start_times` / `photostim_stop_times` → `input[1]` | Binary 1 when bin center lies in any stimulation interval, else 0 | Native BehavioralEvents and reference stimulation metadata | Time-varying." Step 4: photostimulated trials are retained rather than masked out because "photostimulation is a required decoder input, so stimulation trials must be retained when their required streams are valid."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series (not a per-trial flag): each bin is 1 if its centre falls in the half-open interval `[start, stop)` of **any** photostim event in the session, else 0. The mask is accumulated with a logical OR over all events in the session and cast to `float32` into row 1 of the input array. Non-stimulated trials get all-zero rows automatically. On a stimulated trial this produces exactly 10 consecutive 1-bins (0.5 s), ending at bin centre −0.725 s, i.e. before the go cue.

ii.
```python
for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
return out.astype(np.float32)
```
```python
inp=np.stack([abs_centers-tone[:,None],stim_series(f,abs_centers)],axis=1).astype(np.float32)
```

iii. Step 3: "Photostimulation | late delay, final 0.5 s, ending before go cue | `methods.txt`, Photoinhibition." Step 10 Check 5: "stimulation uses [start,stop)". Step 9 records the converted range as binary [0, 1]. The task specification requires "Whether photostimulation is on at every time point (discrete, time-varying)", which the AI implements as a per-bin indicator.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation event times are session-absolute, and they are compared directly against `abs_centers`, the same absolute bin centres that define the neural bins. No per-trial re-expression or offset is needed. Because the stimulation events are compared against the whole trial × bin matrix at once, an event belonging to a neighbouring trial could in principle leak into this trial's window; in practice it cannot, since the minimum go-cue spacing is 4.58 s while stimulation ends at go−0.70 s and starts no earlier than go−2.25 s.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
...
out |= ((abs_centers>=a)&(abs_centers<b))
```

iii. Step 4: "Use go-event timestamps for alignment… and native optogenetic event intervals for the stimulation input." Step 10 Check 2 reports an independent raw-data `np.allclose` check on photostimulation state for three sessions, all passing.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side; a miss means it licked the other side; an `ignore` means it never licked.

ii.
```python
trial=f['intervals/trials']; instruction=decode(trial['trial_instruction'][()])[inds]
outcome_s=decode(trial['outcome'][()])[inds]
choice=np.empty(len(inds),np.int64)
for k,(ins,o) in enumerate(zip(instruction,outcome_s)):
    if o=='ignore': choice[k]=2
    elif o=='hit': choice[k]=0 if ins=='left' else 1
    elif o=='miss': choice[k]=1 if ins=='left' else 0
    else: raise ValueError(o)
```

iii. Step 5 mapping: "`trial_instruction` + `outcome` → `output[0]` choice | hit → instructed side; miss → opposite side; ignore → no lick; repeat across 80 bins | Values: left=0, right=1, no lick=2." Trajectory Step 37: "Choice must represent the actual lick: instructed side on hits, opposite side on misses, and no lick on ignores." Step 10 Check 3: "choice is actual response, derived from instruction/outcome." Unknown outcome strings raise rather than being silently mapped.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived per-trial code (0 = left, 1 = right, 2 = no lick) is repeated across all 80 bins with `np.repeat` and written into row 0 of the `(4, 80)` integer output array, so that all four outputs share one time-varying array shape. `output_names[0]` is `'lick direction choice'` and `output_values[0]` is `['left','right','no lick']`. Converted distribution: left 42.8%, right 42.2%, no lick 14.9%.

ii.
```python
output=np.stack([np.repeat(choice[:,None],NT,1),np.repeat(outcome[:,None],NT,1),
                 np.repeat(early[:,None],NT,1),tongue],axis=1)
```
```python
'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],
                 ['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
```

iii. Step 5 Key Decision 6: "Output temporal form: all outputs are 4×80 for decoder compatibility. Trial-level outputs are repeated; tongue is genuinely time-varying." Trajectory Step 40: "Decoder.py confirms the preferred uniform representation: neural/input float32 and output integer arrays, all shaped variable × time."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `intervals/trials/outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome_s=decode(trial['outcome'][()])[inds]
```

iii. Step 2 records the native distribution ("65,254 hit; 15,641 miss; 14,095 ignore"), confirming the column holds precisely the three categories the instructions ask for, so no derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped through a fixed dictionary to 0 = ignore, 1 = miss, 2 = hit, then repeated across all 80 bins into row 1 of the output array. A string outside the dictionary would raise a `KeyError`. Converted distribution: ignore 14.9%, miss 16.6%, hit 68.5%, closely tracking the raw 14.84/16.47/68.70%.

ii.
```python
omap={'ignore':0,'miss':1,'hit':2}; outcome=np.array([omap[x] for x in outcome_s],np.int64)
```
```python
output=np.stack([..., np.repeat(outcome[:,None],NT,1), ...],axis=1)
```

iii. Step 5 mapping: "`outcome` → `output[1]` | Direct categorical mapping repeated across bins | Native trials | ignore=0, miss=1, hit=2." The code order follows the instructions' listed order (ignore, miss, hit). Step 9 compares converted against raw distributions and finds them consistent.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `intervals/trials/early_lick` column, which holds the strings `'no early'` and `'early'`.

ii.
```python
early_s=decode(trial['early_lick'][()])[inds]
```

iii. Step 2 records the native counts: "Early lick: 84,185 no-early; 10,805 early." Step 3/4 note that the data paper excluded early-lick trials from its own analyses, but Step 4 resolves: "Required labels include early lick, no lick/outcome ignore, and photostimulation… Retain these required categories when neural/video streams are valid; do not apply the analysis-specific regular-trial mask."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 = no, 1 = yes by testing for the literal string `'early'`, with any other value falling through to 0; then repeated across all 80 bins into row 2 of the output array. Converted distribution: no 88.4%, yes 11.6% (raw 11.37%).

ii.
```python
early=np.array([1 if x=='early' else 0 for x in early_s],np.int64)
```
```python
output=np.stack([..., np.repeat(early[:,None],NT,1), ...],axis=1)
```

iii. Step 5 mapping: "`early_lick` → `output[2]` | Direct mapping repeated across bins | Native trials / regular-mask logic | no=0, yes=1." Step 12 Check 6 confirms the class balance is non-degenerate ("early 88.42/11.58%").

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` at ~294 Hz. Column 1 is the y-position used as the value; column 2 (DeepLabCut likelihood) determines visibility.

ii.
```python
ts=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
tt=ts['timestamps'][()]; d=ts['data'][()]; y=d[:,1]; likelihood=d[:,2]
```

iii. Step 2: "`acquisition/BehavioralTimeSeries` contains side-camera jaw, nose, and tongue tracking. Tongue data columns are `(tongue_x, tongue_y, tongue_likelihood)` with explicit timestamps; values are finite, while likelihood is strongly bimodal and therefore carries visibility information." Step 3: "Side and bottom video were recorded at 300 Hz. DeepLabCut tracked tongue, jaw, and nose."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps:
1. Session-level percentiles: the 40th and 60th percentiles of `tongue_y` are computed over **all camera frames of the session whose likelihood ≥ 0.9** (raw frames, not binned values). Sessions with fewer than 2 visible frames raise.
2. Resampling to the bin grid: both `y` and `likelihood` are **linearly interpolated at the 80 absolute bin centres** of each trial (`np.interp`, NaN/0 outside the camera's time range). This is point sampling, not averaging — there are ~15 camera frames per 50 ms bin and only the interpolated value at the bin centre is used.
3. Visibility gating: a bin counts as visible only if its interpolated likelihood ≥ 0.9 and the interpolated y is finite; otherwise it is class 3.

ii.
```python
def tongue_outputs(f, abs_centers):
    ts=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
    tt=ts['timestamps'][()]; d=ts['data'][()]; y=d[:,1]; likelihood=d[:,2]
    visible=likelihood>=0.9
    if visible.sum()<2: raise ValueError('Insufficient visible tongue frames')
    p40,p60=np.percentile(y[visible],[40,60])
    flat=abs_centers.ravel(); yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
    li=np.interp(flat,tt,likelihood,left=0,right=0).reshape(abs_centers.shape)
    cat=np.full(abs_centers.shape,3,dtype=np.int64)
    v=(li>=0.9)&np.isfinite(yi)
    cat[v & (yi<p40)]=0; cat[v & (yi>=p40) & (yi<=p60)]=1; cat[v & (yi>p60)]=2
    return cat,p40,p60
```

iii. Step 4: "Tongue visibility | No explicit confidence threshold found | Likelihood is strongly bimodal; 11.87% frames ≥0.9, very little intermediate mass | DeepLabCut used, threshold unstated | Use likelihood ≥0.9 as visible. This conservative threshold is supported by the native bimodal distribution and is documented explicitly." Trajectory Step 31/33: "Across all 179.3 million tongue frames, only 12.35% have likelihood ≥0.5 and 11.87% ≥0.9… a 0.9 visibility threshold is therefore well justified and consistent with standard conservative DeepLabCut practice." The choice of interpolation at bin centres is justified by reference to the method-paper code's video-alignment convention (Step 1: "Align video embedding frames to a common time grid relative to go cue; nearest preceding frame is used on the requested grid") and Step 5 mapping: "Linear interpolation at bin centers; visible if interpolated likelihood ≥0.9."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, using the session-level `p40`/`p60` computed in 8-b: `y < p40` → 0, `p40 ≤ y ≤ p60` → 1, `y > p60` → 2, and not visible → 3. The percentile edges are per session (stored per session in `metadata['session_info']` as `tongue_p40`/`tongue_p60`) and are computed over the whole session's visible frames, not only over retained trial windows. Converted distribution: 5.7% / 3.2% / 6.5% / 84.6% — i.e. among visible bins, roughly 37/21/42%, close to the intended 40/20/40 split.

ii.
```python
p40,p60=np.percentile(y[visible],[40,60])
...
cat=np.full(abs_centers.shape,3,dtype=np.int64)
v=(li>=0.9)&np.isfinite(yi)
cat[v & (yi<p40)]=0; cat[v & (yi>=p40) & (yi<=p60)]=1; cat[v & (yi>p60)]=2
```
```python
'output_values':[..., ['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
```

iii. Step 5 Key Decision 7: "Tongue thresholds: percentiles are computed from all session frames with likelihood ≥0.9, not merely retained trial windows, matching 'over the session.' Boundary convention: class 0 below p40, class 1 from p40 through p60, class 2 above p60." Step 10 Check 5 restates the boundary convention as an explicit off-by-one/edge check. The fourth class is required because the tongue is retracted in ~88% of frames and the tracker still reports a position there.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and the go cues, and `tongue_outputs` is passed exactly the same `abs_centers` matrix (`go + CENTERS_REL`) used for the neural bins and the inputs, so bin *k* of the tongue output covers the same 50 ms interval as bin *k* of the firing rates. `abs_centers` is filtered by `has_neural` **before** being handed to `tongue_outputs`, so trial indexing stays consistent with the neural array. Where the camera was not running (the video is trial-gated, so leading bins of trials whose go cue falls <2.5 s after trial start have no frames), `np.interp`'s `left`/`right` fill values give NaN y and 0 likelihood, and those bins fall into class 3.

ii.
```python
abs_centers=go[:,None]+CENTERS_REL[None,:]
...
inds=inds[has_neural]; go=go[has_neural]; abs_centers=abs_centers[has_neural]
...
tongue,p40,p60=tongue_outputs(f,abs_centers)
```
```python
flat=abs_centers.ravel(); yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
li=np.interp(flat,tt,likelihood,left=0,right=0).reshape(abs_centers.shape)
```

iii. Step 4: "Use go-event timestamps for alignment." Step 12 Check 5: "Processing plots and raw `np.allclose` checks verify go alignment, replay-aware sample onset, stimulation intervals, and tongue timestamps." The `--show-processing` plot draws the tongue class series on the same −2.5→+1.5 s axis as the neural raster.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases, split between "drop it" and "make it an explicit category", plus hard failures for structural surprises:

- **Session never quality-controlled** (`classification` is NaN for every unit): `decode()` turns the NaN into the string `'nan'`, no unit matches `'good'`, and `convert_session` returns `None`. One session is dropped this way.
- **Trials with no/invalid spike coverage**: excluded via `obs_intervals` + `is_good_trials` (1-e).
- **Trials whose entire population is silent for the full 4 s**: treated as a missing neural segment and dropped (2,423 trials).
- **Sessions left with <2 usable trials**: dropped, satisfying the target format's 2-trial minimum.
- **Bins with no visible tongue** (retracted tongue, or camera not running): assigned the explicit `'not visible'` class 3 rather than imputed.

Structural inconsistencies are not papered over — they raise: an `obs_intervals` start that does not match a trial start to within 1e-5 s, an `is_good_trials` row whose length differs from the unit's interval count, a trial with no sample onset between its start and its go cue, an unrecognised `outcome` string, and a session with <2 visible tongue frames. Finally, shape and finiteness assertions run on every session.

ii.
```python
def decode(a):
    return np.asarray([x.decode(errors='replace') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in a])
```
```python
good,valid=curate(f)
if not len(good): return None
inds=np.where(valid)[0]
if len(inds)<2: return None
```
```python
# A completely silent population across four seconds indicates a missing neural segment.
has_neural=np.any(neural != 0, axis=(1,2))
...
if len(inds)<2: return None
```
```python
if len(j) and np.max(np.abs(trial_starts[j]-intervals[:,0])) > 1e-5:
    raise ValueError('Observation intervals do not map to trial starts')
if len(z)!=len(m): raise ValueError('Validity/interval length mismatch')
if hi<=lo: raise ValueError(f'No sample onset for trial {i}')
if visible.sum()<2: raise ValueError('Insufficient visible tongue frames')
else: raise ValueError(o)
```
```python
assert neural.shape==(len(inds),len(good),NT) and inp.shape==(len(inds),2,NT) and output.shape==(len(inds),4,NT)
assert np.isfinite(neural).all() and np.isfinite(inp).all()
```

iii. Step 9: "Completely zero population windows are excluded as missing neural segments." Step 10 Issues: "Corrected broad trial retention exposed 2,423 windows with no spikes in any of 90-923 curated neurons. These occurred across outcomes and indicate missing neural segments. They were excluded. Full verification then reported no warnings." Step 4: "one [NWB] has no classifier-good units… exclude the one session with zero retained units. This yields exactly 173 sessions." Step 6: "validates unit/trial coverage… and asserts all shapes/ranges."

## 10-a. What are the most time-consuming steps of the code?

i. The script prints a wall-clock time per session but no per-step breakdown. Full conversion of 173 sessions took 130.6 s of processing (mean 0.75 s/session, range 0.19-2.02 s), plus pickling of the 11.17 GiB result. The dominant costs are I/O and per-unit work, both scaling with unit count: (a) in `spike_rates`, one HDF5 slice read per good unit (`flat[a[ui]:b[ui]]`, ~400 reads per session) plus the `searchsorted`/`bincount` pass over that unit's spikes; (b) in `curate`, loading the full `obs_intervals` and `is_good_trials` arrays for **all** units (including the ~75% that are discarded) and then running `map_observed_trials` once per good unit; (c) `tongue_outputs` reading the full `(n_frames, 3)` tracking array (~680k × 3) and interpolating; (d) serialising the output pickle, which is by far the largest single wall-clock item.

ii.
```python
info={'file':path.name, ..., 'seconds':time.time()-t0}
...
print(f'[{i+1}/{len(files)}] {p.name}: trials={r[5]["n_retained_trials"]}, neurons={r[5]["n_neurons"]}, seconds={r[5]["seconds"]:.2f}',flush=True)
```
```python
flat=u['obs_intervals'][()]; a,b=ragged_bounds(u['obs_intervals_index'][()]); manual=u['is_good_trials'][()]
for ui in good:
    ...
```

iii. Step 6: "A naïve neuron × trial × spike histogram loop would be too slow and duplicate I/O. Full float32 payload is expected to be several GiB… Spike assignment is vectorized across all trials within each neuron using searchsorted/bincount; NWBs are opened once per session; arrays are preallocated." Step 7 estimated "generally 0.2-2.0 s/session… comfortably under 15 minutes including serialization", which the full run confirmed. The AI therefore concluded no further optimisation was needed and did not profile individual steps.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops remain:

- `curate`: `for ui in good` — runs `map_observed_trials` (two `searchsorted` calls plus comparisons) once per good unit, ~400 iterations per session. Since units on the same probe share identical `obs_intervals`, this could be collapsed to one mapping per distinct probe/interval array, or the `is_good_trials` intersection could be done as a single `manual[good].all(0)` after one shared interval mapping.
- `final_tone_onsets`: `for k,i in enumerate(trial_indices)` — one iteration per trial; the whole function is a two-line vectorised `searchsorted` over the sorted sample-onset array.
- `stim_series`: `for a,b in zip(starts,stops)` — one full `(n_trials, 80)` boolean comparison and OR per stimulation event (up to ~170 events/session), i.e. O(n_events × n_trials × 80) work where a single `searchsorted` into the interval boundaries would be O(n_trials × 80 × log n_events).
- `spike_rates`: `for col,ui in enumerate(units)` — inherent to the ragged per-unit spike storage (each unit has a different spike count), though the *I/O* part could be collapsed by reading the whole `spike_times` buffer once instead of slicing per unit.

Also `np.repeat(...,NT,1)` materialises three (n_trials, 80) copies of per-trial scalars that broadcasting would avoid.

ii.
```python
for ui in good:
    z=flat[a[ui]:b[ui]]; m=manual[ui]
    ...
    jj=map_observed_trials(trial_st,z)
    uv=np.zeros(len(go), bool); uv[jj]=m
    valid &= uv
```
```python
for k,i in enumerate(trial_indices):
    lo=np.searchsorted(sample, starts[i]-1e-8, 'left'); hi=np.searchsorted(sample, go[i]+1e-8, 'right')
```
```python
for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
```

iii. The AI's notes only claim the spike loop was addressed (Step 6: "Spike assignment is vectorized across all trials within each neuron using searchsorted/bincount") and do not identify the other three. The implicit justification is the runtime budget: Step 7 concludes the conversion is "comfortably under 15 minutes", so no further vectorisation was pursued.

## 10-c. What processing does the code repeat multiple times?

i. Three repetitions:

- **Per-unit re-derivation of the observation-interval → trial mapping** in `curate`. For every one of the ~400 good units the same `map_observed_trials` computation is run, even though `obs_intervals` rows are shared across all units on the same probe (in most sessions they are identical for all units). This is the largest redundancy in the script.
- **`ragged_bounds` is called twice per session** on two different index vectors (`obs_intervals_index` in `curate`, `spike_times_index` in `spike_rates`) — cheap, but the units group is also re-opened and re-read in both functions.
- **`go_start_times` is read twice** — once inside `curate` (only to get `len(go)`) and once in `convert_session`.

Everything else is computed once: the bin grid (`EDGES_REL`, `CENTERS_REL`) is built at module level and reused for every trial and session, each file is opened once, and the per-session tongue percentiles are computed inside the same pass rather than requiring a second sweep.

ii.
```python
def curate(f):
    ...
    go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][()]
    flat=u['obs_intervals'][()]; a,b=ragged_bounds(u['obs_intervals_index'][()]); manual=u['is_good_trials'][()]
    for ui in good:
        ...
        jj=map_observed_trials(trial_st,z)
```
```python
be=f['acquisition/BehavioralEvents']; go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
```

iii. Not identified in CONVERSION_NOTES; the only related claim is Step 6's "NWBs are opened once per session; arrays are preallocated". The per-unit repetition follows from the AI's stated curation rule (Step 5 Key Decision 3: "require a mapped observation row and manual good flag for **every** retained unit, and intersect across retained units"), which is naturally expressed as a per-unit loop even though the interval mapping itself is invariant.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts:

- **Inputs are computed for trials that are then dropped.** `tone`, `inp` (both the tone-time row and the full `stim_series` mask) are computed for all `valid` trials, and only afterwards is the `has_neural` mask applied and ~2.6% of trials discarded. Moving the all-zero filter before input construction would avoid this.
- **A no-op transpose**: `counts.astype(np.float32).transpose(0,1,2)` is the identity.
- **A dtype round-trip**: counts are accumulated in `uint16` and then cast to `float32`, allocating the full trial × neuron × bin array twice.
- **`is_good_trials` and `obs_intervals` are read in full for all units**, including the ~75% of units that are never used.
- **`info['source_trial_indices']`** stores a full per-trial index list for every session in the metadata; it is never used downstream (though it is useful provenance).
- In `--show-processing` mode, `ax[3].hist(neural.ravel(),bins=50)` histograms the entire session's firing-rate array for a diagnostic plot.

Nothing that reaches the output dictionary is unused — all of `neural`, `input`, `output`, `subjects`, `brain_regions`, `input_names`, `output_names`, `output_values` and `metadata` are consumed by the decoder or the format verifier.

ii.
```python
inp=np.stack([abs_centers-tone[:,None],stim_series(f,abs_centers)],axis=1).astype(np.float32)
neural=spike_rates(f,good,go)
has_neural=np.any(neural != 0, axis=(1,2))
inds=inds[has_neural]; ...; inp=inp[has_neural]; neural=neural[has_neural]
```
```python
return counts.astype(np.float32).transpose(0,1,2) / np.float32(BIN_S)
```
```python
info={'file':path.name,'source_trial_indices':inds.tolist(), ...}
```

iii. Not discussed in CONVERSION_NOTES. The ordering (inputs before the all-zero filter) is a consequence of the filter being discovered late — Step 10 records that the all-zero exclusion was added as a fix after the first full run — and the cost is negligible relative to the 0.75 s/session budget the AI was working to.
