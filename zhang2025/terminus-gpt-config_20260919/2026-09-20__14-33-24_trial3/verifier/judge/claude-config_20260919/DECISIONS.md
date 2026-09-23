# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API at all. It takes the reference repository's own cohort freeze file, `code_zhang2025/data/bwm_release.csv` (699 PID rows → 459 unique EIDs, 139 subjects), and uses its `lab / subject / date / session_number / probe_name` columns to build native ALF paths into the staged cache at `/app/data/one_cache/<lab>/Subjects/<subject>/<date>/<nnn>/alf/`. Every object is then loaded by globbing the session's `alf` tree and reading the file directly (`pd.read_parquet` for the trials table and `clusters.metrics.pqt`, `np.load` for spikes, wheel, camera, and channel arrays). Because the staged cache contains several ALF revisions (`#YYYY-MM-DD#` sub-directories) including a partial one-column legacy trials object, the AI wrote explicit revision-resolution rules: for tables, take candidates that contain all required columns and pick the one with the most rows, breaking ties by latest lexical path; for `.npy` objects, take the newest path, and for the camera require a timestamp/value pair of equal length. Conversion is a two-pass design: pass 1 loads behaviour per session and caches it to `/app/cache/conversion_behavior/<eid>.npz`; pass 2 re-reads those caches and memory-maps the spike arrays one probe at a time. Sessions are processed serially in one process; the whole run took 318 s and retained 444/459 sessions, 136 subjects, 188,925 trials.

ii.
```python
ROOT = Path('/app/data/one_cache')
COHORT = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def session_path(r):
    return ROOT / str(r.lab) / 'Subjects' / str(r.subject) / str(r.date) / f'{int(r.session_number):03d}'

def full_trial_table(alf: Path):
    candidates=[]
    for p in alf.glob('**/_ibl_trials.table.pqt'):
        try:
            x=pd.read_parquet(p)
            if REQ_TRIAL.issubset(x.columns): candidates.append((len(x),str(p),p,x))
        except Exception: pass
    if not candidates: raise FileNotFoundError(f'No full trial table in {alf}')
    return max(candidates,key=lambda z:(z[0],z[1]))[2:]
```

```python
rel=pd.read_csv(COHORT,index_col=0); sessions=rel.drop_duplicates('eid',keep='first')
...
for k,(_,r) in enumerate(sessions.iterrows(),1):
    eid=str(r.eid); cf=CACHE/f'{eid}.npz'
    try:
        info=preprocess_behavior(r,eid,cf); kept.append((r,eid,cf))
    except Exception as e: print(f'EXCLUDE {eid}: {type(e).__name__}: {e}',flush=True)
```

iii. From CONVERSION_NOTES.md Step 2/4/5: `bwm_release.csv` is "the authoritative paper cohort" that "maps 699 PIDs to 459 EIDs, 139 subjects" and exactly reproduces the data paper's headline numbers (459 sessions / 699 insertions / 621,733 clusters / 75,708 label-1 units), which the AI used as validation of its path and revision resolution. Direct file access rather than `SessionLoader` was chosen because "generic SessionLoader selected a partial one-column trial object in the staged mixed cache"; the AI documents this as a deliberate, validated divergence ("Explicit validated full-table resolution fixed this; all 459 full tables and 699 cluster tables are found"). It also records that an early `find`/`rglob` inventory missed data because it did not follow the staged directory symlinks, and that Step 2 was reopened and redone.

## 1-b. How are the data split into subjects?

i. Subjects are taken verbatim from the `subject` column of `bwm_release.csv` — one subject string per session row, no parsing of paths. After conversion, the unique subject list is built in first-appearance order with `dict.fromkeys` (not sorted), and `subject_idx` is each session's index into that list. Only subjects with at least one retained session appear: 136 of the 139 cohort subjects survive (3 subjects lost all their sessions to the camera/coverage filters).

ii.
```python
subj.append(str(r.subject))
...
subjects=list(dict.fromkeys(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int32)
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "`bwm_release.csv.subject` → `subjects`, `subject_idx`; Stable first-appearance unique subject list and per-session index … Only subjects with retained sessions appear." The release CSV already carries a unique subject id per session, so nothing has to be derived. Step 9 notes "Retained subjects: 136 … three subjects have no retained session."

## 1-c. How are the data split into sessions?

i. A session is one EID. The cohort CSV has one row per probe insertion, so the AI de-duplicates on `eid` to get the 459 sessions, and later re-selects all rows with that EID to find that session's probes. Nothing is split; each retained session becomes one element of the `neural`/`input`/`output` lists (444 sessions in the final file).

ii.
```python
rel=pd.read_csv(COHORT,index_col=0); sessions=rel.drop_duplicates('eid',keep='first')
if args.sample: sessions=sessions.iloc[:2]
...
probe_rows=rel[rel.eid.astype(str)==eid]
```

iii. Step 2/4: the release file "maps 699 PIDs to 459 EIDs"; the session is the natural unit and the reference pipeline (`prepare_data`) also works per EID and merges that EID's probes. No decision needed beyond de-duplicating the probe-level rows.

## 1-d. How are the data split into trials?

i. The split is given by the data: one row of `_ibl_trials.table.pqt` is one trial, and each surviving row becomes one `(n_neurons, 100)` neural matrix plus one input and one output array. Trial identity is carried through the pipeline as the integer row index into the *raw, unfiltered* trials table (`trial_idx` in the behaviour cache), so filtered trials never renumber the survivors.

ii.
```python
alf=session_path(row)/'alf'; trial_p,x=full_trial_table(alf)
base=trial_mask(x); stim=x.stimOn_times.to_numpy(dtype=float)
...
np.savez(cache_file,trial_idx=common.astype(np.int32), ... ,stim=stim[common], ...)
```

iii. Notes Step 2: "Parquet trial/cluster tables … one row/trial". The AI kept raw row indices explicitly so that block counting and the raw-file sanity checks (Step 10, Check 2) could map a converted trial back to a named raw trial index.

## 1-e. How are trials filtered based on quality controls?

i. Two stages, intersected. Stage 1 (`trial_mask`) reproduces the reference code's `load_trials_and_mask(..., max_trial_len=10.0)` exactly: all of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType` must be non-NaN; reaction time `firstMovement_times - stimOn_times` must be in [0.08, 2.0] s; `choice != 0` (no-response trials dropped); and trial duration `feedback_times - goCue_times <= 10 s`. `probabilityLeft == 0.5` (unbiased) trials are *kept*, matching `exclude_unbiased=False`. Stage 2 requires that both continuous streams cover the whole [-0.5, +1.5] s window: at least two samples inside the window, the first sample within one bin (20 ms) of the window start, the last within one bin of the window end, and all interpolated values finite. The wheel-valid and camera-valid trial sets are then intersected, and a session with fewer than 2 jointly valid trials is dropped entirely. Result: 296,090 raw trials → 195,781 after stage 1 → 188,925 retained after stream coverage.

ii.
```python
def trial_mask(x):
    m=np.ones(len(x),bool)
    for c in ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']:
        m &= x[c].notna().to_numpy()
    rt=x.firstMovement_times.to_numpy()-x.stimOn_times.to_numpy()
    m &= (rt >= .08) & (rt <= 2.)
    m &= x.choice.to_numpy()!=0
    m &= (x.feedback_times.to_numpy()-x.goCue_times.to_numpy() <= 10.)
    return m
```

```python
        tx=times[ib:ie]; vy=values[ib:ie]
        if len(tx)<2 or abs(beg-tx[0])>BIN or abs(end-tx[-1])>BIN or not np.isfinite(vy).all(): continue
```

```python
    common=np.intersect1d(ci,wi,assume_unique=True)
    if len(common)<2: raise ValueError(f'only {len(common)} jointly valid trials')
```

iii. Step 1/4/5: "Actual trial mask defaults plus `max_trial_len=10`: reaction time … 0.08–2.0 s; `feedback_times-goCue_times` must be ≤10 s; required fields … must be non-NaN; choice 0 is excluded. Probability-left 0.5 trials are retained (`exclude_unbiased=False`)." The coverage test is copied from the reference `get_behavior_per_interval` skip reasons ("target data starts too late"/"ends too early", both compared against one `binsize`). The AI also deliberately departs from one reference behaviour: `align_spike_behavior` combines masks with a Python `and` on lists, which silently fails to intersect them; the AI calls this "a clear implementation bug" and explicitly intersects every stream mask instead.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` for every probe listed for the EID in `bwm_release.csv`. `clusters.metrics.pqt` supplies the cluster id universe (and would supply the `label` QC score, which is read but not used as a filter); `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` supply anatomy only. The spike arrays are memory-mapped rather than fully loaded.

ii.
```python
    st=np.load(p['spikes.times.npy'],mmap_mode='r'); sc=np.load(p['spikes.clusters.npy'],mmap_mode='r')
```

```python
    return metrics, {n:local_or_glob(n) for n in ['spikes.times.npy','spikes.clusters.npy',
                     'clusters.channels.npy','channels.brainLocationIds_ccf_2017.npy']}
```

iii. Step 5 mapping table: "Per-probe `spikes.times`, `spikes.clusters` → `neural[session][trial]` … Reference: `load_spiking_data`, `merge_probes`, `get_spike_data_per_interval`, `bin_spiking_data`". Memory-mapping is listed under Step 6 speedups ("Session/probe streaming avoids holding all raw spikes in RAM").

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 half-open 20 ms bins spanning stimOn−0.5 s to stimOn+1.5 s, per cluster, with no smoothing and **no conversion to firing rate** — the stored values are raw integer counts. Each trial's window is located by binary search on the (checked/sorted) spike-time array, cluster ids are remapped to metrics-table row order through a lookup array, and one flattened `np.bincount` fills the whole `(n_clusters, 100)` grid. All probes of a session are binned separately and then concatenated along the neuron axis per trial, so the second probe's units continue after the first probe's, in CSV probe order. Storage dtype is chosen per session: `uint8` if the session's maximum bin count ≤255, else `uint16` (observed maximum across the release was 68).

ii.
```python
    lookup=np.full(maxid+1,-1,dtype=np.int32); lookup[cluster_ids]=np.arange(n,dtype=np.int32)
    ...
    for s in stim:
        beg=s+OFF0; end=s+OFF1; ib=np.searchsorted(st,beg,'left'); ie=np.searchsorted(st,end,'left')
        spike_t=np.asarray(st[ib:ie]); ids=np.asarray(sc[ib:ie],dtype=int)
        ...
        tb=np.floor((spike_t-beg)/BIN).astype(int)
        valid2=(tb>=0)&(tb<NBIN); flat=mapped[valid2]*NBIN+tb[valid2]
        a=np.bincount(flat,minlength=n*NBIN).reshape(n,NBIN)
```

```python
        neural=[np.concatenate([pmats[p][j] for p in range(len(pmats))],axis=0).astype(dtype,copy=False)
                for j in range(ntr)]
```

iii. Step 1: "Spike bins use half-open trial intervals (`times >= start` and `times < end`) and integer spike counts. Returned matrices are neuron by time after transposition." Step 5 decision 3: "Merge every listed probe from the same EID, because simultaneous probes share behavior and the reference explicitly combines them." Step 7/9: counts are kept as integers deliberately — the verifier's "prefers float32" warning is answered with "intentional lossless compact storage of integer spike counts. Full float32 storage would expand the estimated neural payload from ~26.5 GB to ~105.9 GB without adding information; decoder `SessionData` already converts each session to float32."

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron quality filtering is applied at all.** Every cluster row in `clusters.metrics.pqt` is retained, including clusters with `label` 0 and clusters whose Beryl acronym is `void` (histologically outside the brain) or `root`. The `label` column is read only to validate that the release contains 75,708 label-1 units; it never enters the mask. The resulting file has 599,865 neurons summed over 444 sessions (mean ≈1,351/session), of which 85,656 are `root` and 12,827 are `void`, spread over 281 Beryl categories. The only neuron-level checks in the code are structural: cluster-channel array length must equal the metrics row count, and channel indices must be in range.

ii.
```python
def bin_probe(row,stim):
    metrics,p=resolve_probe(row)
    cluster_ids=metrics.cluster_id.to_numpy(dtype=int)
    n=len(cluster_ids)                      # every cluster kept; `label` never used as a filter
```

```python
    br=BrainRegions(); native=br.id2acronym(atlas[ch]); regions=br.acronym2acronym(native,mapping='Beryl').astype(str)
    return mats,regions,max_count
```

iii. Step 4 discrepancy table: "Zhang `prepare_data` passes `qc=None`, bins all clusters, records label metadata … Resolution: Match the methods-paper code for decoder data: retain all clusters. Preserve QC metadata/statistics; do not silently substitute data-paper analysis filtering." Step 5 decision 2 adds: "This yields the paper's 621,733 source clusters … Data-paper well-isolated QC statistics are retained as checks but not imposed because that would change the methods-paper decoder dataset." The AI also explicitly decided not to apply the data paper's gray-matter / ≥5-good-units / ≥2-session region restrictions, arguing those "apply region-level data-paper analyses, not Zhang whole-session decoder caching", and kept `root`/`void` because Beryl assigns them ("Preserve `root` if assigned; one region per retained cluster").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams share one session clock, so alignment is a subtraction. For every retained trial the window is `[stimOn_times − 0.5, stimOn_times + 1.5]`; spikes inside it are found by `searchsorted` and their bin index is `floor((t − window_start)/0.02)`, i.e. time is measured from the window start, which is a fixed offset from stimulus onset. Bin 0 therefore starts exactly at −0.5 s relative to stimulus onset, and the stimulus onset falls exactly on the boundary between bins 24 and 25. Before binning, the code guards against unsorted spike times by sampling the first 100k differences and sorting if any decrease is found.

ii.
```python
        beg=s+OFF0; end=s+OFF1; ib=np.searchsorted(st,beg,'left'); ie=np.searchsorted(st,end,'left')
        ...
        tb=np.floor((spike_t-beg)/BIN).astype(int)
```

```python
    if len(st)>1 and np.any(np.diff(st[:min(len(st),100000)])<0):
        order=np.argsort(st,kind='stable'); st=np.asarray(st)[order]; sc=np.asarray(sc)[order]
```

iii. Step 4/5: "Temporal alignment … Task and executable reference cache take precedence: all variables use stimulus onset and common 100-bin grid", matching the reference `params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5), 'binsize': 0.02}`. Step 10 Check 3 records "Alignment: both use `stimOn_times` plus [-0.5,+1.5]", and Check 2 verified trial-5/neuron-3 counts against an independent histogram of the raw spike files with maximum absolute difference 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, covering a 2 s window; `metadata['time_bin_size'] = 20.0` ms. No rebinning, resampling, or smoothing of the neural data — spikes are histogrammed once directly onto the final grid. The behavioural streams are resampled (by linear interpolation) onto that same 100-point grid, but the neural data itself is never re-binned.

ii.
```python
BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
NBIN = 100
REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)
```

```python
'time_bin_size':20.0,'temporal_alignment_event':'visual stimulus onset (stimOn_times)','off_start':OFF0,'off_end':OFF1,
```

iii. Step 1: "Reference trial parameters are `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)` seconds, and `binsize=0.02` seconds, producing 100 bins per trial." Step 3 notes the method paper's alternative 50 ms per-trial analysis but concludes "Provided caching code uses 20 ms for all requested streams", so 20 ms is used throughout.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from raw data beyond the alignment event `stimOn_times`: it is a fixed, session-independent vector of the 100 bin **end** times relative to stimulus onset, `[-0.48, -0.46, …, 1.48, 1.50]`, broadcast identically to every trial of every session as row 0 of `input`.

ii.
```python
REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)
```

```python
inp=[np.vstack((REL_TIME,np.full(NBIN,tib[j],np.float32))).astype(np.float32) for j in range(ntr)]
```

iii. Step 1: "Interpolation points are `linspace(interval_start + binsize, interval_end, n_bins)`, i.e. bin-end times from −0.48 through +1.50 s relative to stimulus onset" — copied from the reference `get_behavior_per_interval`'s `x_interp`. Step 5 decision 5: "Behavior and time input use bin ends −0.48 through +1.50 s. Spike bins are `[start,start+0.02)`… this deliberate 20 ms offset matches reference behavior interpolation."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. The vector is a constant defined once at module level as float32 and stacked into every trial's input array. It is monotonic, exactly 20 ms apart, and the AI verified the increments and endpoints against `np.arange(-0.48, 1.5001, 0.02)`.

ii.
```python
REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)
```

```python
for n,i,o in zip(neural_all[sidx],inputs_all[sidx],outputs_all[sidx]):
    assert n.shape[1]==i.shape[1]==o.shape[1]==NBIN and np.isfinite(i).all()
```

iii. Step 5 planned sanity check: "Compare converted time input to `np.arange(-0.48,1.5001,0.02)` with `np.allclose`", reported as passing with max absolute difference 0 in Step 10 Check 2; Step 10 Check 5 adds "time increments are exactly 20 ms within tolerance".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It labels the same bins as the neural array, by their right edge: neural bin *k* covers `[−0.5 + 0.02k, −0.5 + 0.02(k+1))` and `REL_TIME[k] = −0.5 + 0.02(k+1)`. So index *k* of the input and index *k* of the neural matrix refer to the same 20 ms interval, with the time value naming the end rather than the centre of that interval (a 10 ms convention offset relative to bin centres). The behavioural outputs are interpolated at exactly these same times, so all four streams share one index axis. `metadata` still reports the window as `off_start=-0.5`, `off_end=1.5`.

ii.
```python
        tb=np.floor((spike_t-beg)/BIN).astype(int)      # neural: bin k = [beg+0.02k, beg+0.02(k+1))
```

```python
        grid=stim[i]+REL_TIME.astype(float)             # behaviour and time input: right edge of bin k
        y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

iii. Step 5 decision 5 (above) states the offset is deliberate and chosen to match the reference behaviour grid rather than to centre the label inside the bin; Step 7 records the resulting verifier-visible range "[-0.48, 1.50] s (verifier rounds to [-0.5,1.5])".

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the raw trials table. There is no block id in the data, so a block boundary is defined as any change in `probabilityLeft` between consecutive rows (the first row always starts a block).

ii.
```python
def trial_in_block(prob):
    prob=np.asarray(prob)
    change=np.r_[True, prob[1:] != prob[:-1]]
```

```python
    probs=x.probabilityLeft.to_numpy(dtype=float)
    tib=trial_in_block(probs)
```

iii. Step 5 mapping: "`trials.probabilityLeft` transitions → `input[...][1,:]`; Number trials consecutively elapsed in current block … Task-specific derived variable." The trials table has no block column, so the block structure is recovered from the prior, which the task holds constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A running position within the block, **1-indexed** (first trial of a block = 1), computed with a vectorised running-maximum of block start indices. It is computed on the **complete, unfiltered** trial sequence and only then indexed by the retained trial rows, so excluded trials still advance the counter and the value is the animal's true position in the block. It is a per-trial scalar broadcast across all 100 bins as row 1 of `input`, stored as float32. Observed range in the converted file is [1, 99] (the reference expert's 0-indexed equivalent is [0, 98]).

ii.
```python
def trial_in_block(prob):
    prob=np.asarray(prob)
    change=np.r_[True, prob[1:] != prob[:-1]]
    starts=np.maximum.accumulate(np.where(change,np.arange(len(prob)),0))
    return (np.arange(len(prob))-starts+1).astype(np.float32)
```

```python
    tib=trial_in_block(probs)   # computed on all raw rows
    np.savez(cache_file, ... ,trial_in_block=tib[common], ...)   # then subset to retained trials
```

iii. Step 5 decision 8: "Compute before filtering, reset whenever `probabilityLeft` changes, and number from 1. This preserves the actual experimental trial position despite excluded trials." Step 10 Check 5 verified the first block edge independently: "trial-in-block 90→1 across raw trial indices 89→90", consistent with the task's 90 unbiased trials at the start of a session.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, whose native values are −1, 0 and +1. Trials with `choice == 0` (no response) have already been removed by `trial_mask`, so only ±1 reach the mapping. The AI mapped **−1 → 0 and +1 → 1**, documenting this as "Native −1→0 (left), +1→1 (right)". The value is a per-trial scalar broadcast over all 100 bins as row 0 of `output`, with `output_values[0] = ['left','right']`.

ii.
```python
        cmap={-1:0,1:1}; pmap={0.2:0,0.5:1,0.8:2}; out=[]
        for j in range(ntr):
            c=cmap[int(z['choice'][j])]
```

```python
    np.savez(cache_file, ... ,choice=choices[common].astype(np.int8), ...)
```

iii. Step 4 discrepancy table: "Choice coding | Native IBL values are −1/0/+1 | Valid counts: −1=96,189; +1=99,592 | Task requires left=0, right=1 | Map −1→0 and +1→1; exclude 0." No evidence is given anywhere in CONVERSION_NOTES.md or the trajectory for *why* −1 is taken to be "left"; the polarity is asserted rather than checked, and the Step 10 raw-file sanity check re-uses the same assumption (`choice=0 if tr.choice.iloc[rawi]==-1 else 1`), so it cannot detect a sign error. The IBL convention in the bundled `ibllib` is the opposite (`brainbox/behavior/training.py`: `rightward = trials.choice == -1`; `brainbox/examples/plot_all_peths.py`: "choice: subject's choice (1=left, −1=right)"), and the raw data confirm it: on correct trials with the stimulus on the left, `choice` is always +1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding above plus the broadcast to 100 timepoints as `uint8`; no other transformation. Because the recoding is `{-1: 0, +1: 1}` while IBL codes +1 as a leftward choice, the stored labels are **inverted** with respect to the declared `output_values` of `['left','right']` and to the task specification "left = 0, right = 1". The effect is visible in the class fractions: the AI's file is 49.21 % class 0 / 50.79 % class 1, exactly the mirror image of the expert reference's 50.75 % / 49.25 %.

ii.
```python
            out.append(np.vstack((np.full(NBIN,c,np.uint8),np.full(NBIN,pmap[key],np.uint8),wc[j],mc[j])))
```

```python
'output_names':['choice','prior probability of left','wheel speed','whisker motion energy'],
'output_values':[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
```

iii. Step 5 decision 6: per-trial values are broadcast over time because "a single trial output array cannot mix scalar and time-varying row shapes and the supplied decoder accepts categorical values at every timepoint." Step 12 re-verified that the stored value is constant across all 100 bins and equals the mapping of the named raw trial — i.e. it verified self-consistency with the assumed polarity, not the polarity itself.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes only the values 0.2, 0.5 and 0.8, recoded to 0, 1 and 2 respectively. Values are matched to the nearest legal key with a 1e−6 tolerance and anything else raises. Broadcast over 100 bins as row 1 of `output`.

ii.
```python
        cmap={-1:0,1:1}; pmap={0.2:0,0.5:1,0.8:2}
        ...
            pv=float(z['prior'][j]); key=min(pmap,key=lambda q:abs(q-pv))
            if abs(key-pv)>1e-6: raise ValueError(f'unexpected prior {pv}')
```

iii. Step 4: "Prior coding | Native `probabilityLeft` is 0.2/0.5/0.8 | Counts are only those three values | Task prescribes 0/1/2 mapping | Map exactly 0.2→0, 0.5→1, 0.8→2." Step 5 adds "Exact protocol values only; unexpected values fail validation" — the tolerance test exists to guard against float representation while still rejecting genuinely unexpected priors. Unbiased (0.5) trials are kept rather than excluded, following `exclude_unbiased=False` in the reference.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond that recoding and the broadcast to 100 `uint8` timepoints. The converted distribution is 78,864 / 26,561 / 83,500 trials (0.417 / 0.141 / 0.442), matching the raw valid-trial counts and the expert reference's fractions to within 0.001.

ii.
```python
            out.append(np.vstack((np.full(NBIN,c,np.uint8),np.full(NBIN,pmap[key],np.uint8),wc[j],mc[j])))
```

iii. Step 9 consistency table: "Prior values | 0.2/0.5/0.8 | same | only these values | 0/1/2 mapping | Yes". Step 10 Check 2 compared the converted prior against the raw parquet value for first/middle/last trials of first/middle/last sessions with `np.allclose`, max difference 0.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy` arrays, loaded directly from the session's `alf` directory (newest revision, lengths required to match and at least 20 samples). Velocity is derived with the same brainbox functions the reference `SessionLoader.load_wheel` uses, and speed is its absolute value.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered

def derive_wheel(alf):
    tp=newest(alf.glob('**/_ibl_wheel.timestamps.npy'))
    pp=newest(alf.glob('**/_ibl_wheel.position.npy'))
    if tp is None or pp is None: raise FileNotFoundError('wheel stream missing')
    t=np.asarray(np.load(tp),dtype=float); p=np.asarray(np.load(pp),dtype=float)
    if len(t)!=len(p) or len(t)<20: raise ValueError('invalid wheel arrays')
    pos,ti=interpolate_position(t,p,freq=1000)
    vel,_=velocity_filtered(pos,1000)
    return np.asarray(ti),np.abs(np.asarray(vel))
```

iii. Step 1: "Wheel speed is `abs(wheel.velocity)`." Step 5 mapping: "Native position → 1 kHz linear interpolation → 20 Hz low-pass filtered velocity → absolute value … Uses reference wheel-speed definition and temporal grid", explicitly calling the brainbox functions rather than re-implementing them so the trace matches the reference bit for bit.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps. (1) `interpolate_position(t, p, freq=1000)` puts the sparsely-sampled wheel position on a uniform 1 kHz grid. (2) `velocity_filtered(pos, 1000)` differentiates it with a 20 Hz Butterworth low pass (brainbox default), giving rad/s. (3) Absolute value → speed. (4) Per trial, the trace is sliced to the window, duplicate timestamps are dropped, and the samples are linearly interpolated (with `fill_value='extrapolate'`) onto the 100 bin-end times, after the coverage/finiteness checks described in 1-e. The float32 per-trial traces are cached to npz before discretisation.

ii.
```python
    pos,ti=interpolate_position(t,p,freq=1000)
    vel,_=velocity_filtered(pos,1000)
    return np.asarray(ti),np.abs(np.asarray(vel))
```

```python
def interp_trials(times, values, stim, base_mask):
    """Reference linear interpolation at bin ends; return values and valid indices."""
    order=np.argsort(times,kind='stable'); times=times[order]; values=values[order]
    keep=np.r_[True,np.diff(times)>0]; times=times[keep]; values=values[keep]
    ...
        grid=stim[i]+REL_TIME.astype(float)
        y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

iii. Step 10 "Issues Found and Resolved": the first version used `np.interp`, which clamps at the endpoints; it was "replaced before sample conversion with reference `interp1d(..., fill_value='extrapolate')`" so the edge bins are extrapolated exactly as the reference does. Step 10 Check 2 independently re-derived wheel speed from the raw position file with brainbox and compared trial 5 to the converted values with `np.allclose` (max difference 0).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes at **global** empirical tertiles: after all sessions' per-trial traces are computed and cached, every retained wheel sample in the whole dataset is concatenated and the 1/3 and 2/3 quantiles are taken once; each value is then assigned `np.searchsorted([q1,q2], v, side='right')` → 0/1/2. The thresholds (0.01512 and 0.40508 rad/s) are stored in `metadata['wheel_speed_thresholds']`. Collapsed or non-finite quantiles raise rather than silently degrade. Because the thresholds are global the dataset-wide classes are exactly balanced (6,297,500 timepoints each), but they are *not* balanced within any individual session.

ii.
```python
def thresholds(values):
    x=np.concatenate([v.reshape(-1) for v in values])
    q=np.quantile(x,[1/3,2/3])
    if not np.isfinite(q).all() or q[0]>=q[1]:
        raise ValueError(f'Collapsed/nonfinite tertiles {q}; rank fallback not implemented ...')
    return q.astype(float)
```

```python
    wz=[np.load(cf)['wheel'] for _,_,cf in kept]; mz=[np.load(cf)['whisker'] for _,_,cf in kept]
    wthr=thresholds(wz); mthr=thresholds(mz); del wz,mz
    ...
        wc=np.searchsorted(wthr,z['wheel'],side='right').astype(np.uint8)
```

iii. Step 5 decision 7: "Use global 1/3 and 2/3 empirical quantiles across all retained finite wheel-speed and whisker values … Global thresholds preserve a common physical class meaning across sessions." The AI notes the discretisation itself is a required divergence from the reference (which treats these targets as continuous regression variables): "Required tertile discretization is the intentional task divergence."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. By construction: the speed trace is evaluated at `stimOn_times + REL_TIME`, the same 100 times that label the neural bins, so index *k* of the wheel output and index *k* of the neural matrix describe the same 20 ms interval (the wheel value being the instantaneous speed at that bin's right edge). A trial is only kept if the wheel stream actually spans the window to within one bin at both ends, so no extrapolation beyond 20 ms can occur.

ii.
```python
        grid=stim[i]+REL_TIME.astype(float)
        y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

```python
        if len(tx)<2 or abs(beg-tx[0])>BIN or abs(end-tx[-1])>BIN or not np.isfinite(vy).all(): continue
```

iii. Step 4/5: the wheel is on the same session clock as the spikes, so "Reproduce reference bin-end linear interpolation". The `--show-processing` plots (`/app/processing_<eid>.png`) overlay the trial-1 spike-count image, the aligned wheel trace with its thresholds, and the resulting class trace on a shared time axis with the onset marked, which the AI used as visual confirmation of alignment (Step 7: "no temporal discontinuities or missing ranges were observed").

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy.npy` with its frame times `_ibl_<side>Camera.times.npy`, taken as released with no further processing. The left camera (≈60 Hz) is preferred and the right (≈150 Hz) is used only if no usable left pair exists. "Usable" is enforced by trying every revision pair, newest first, and requiring the timestamp and value arrays to have equal length >1; if neither side yields a matched pair the session is excluded (14 sessions).

ii.
```python
def paired_stream(alf: Path, side: str):
    times=list(alf.glob(f'**/_ibl_{side}Camera.times.npy'))
    vals=list(alf.glob(f'**/{side}Camera.ROIMotionEnergy.npy'))
    for tp in sorted(times,key=str,reverse=True):
        t=np.load(tp,mmap_mode='r')
        for vp in sorted(vals,key=str,reverse=True):
            v=np.load(vp,mmap_mode='r')
            if len(t)==len(v) and len(t)>1:
                return tp,vp
    return None,None
```

```python
    for sd in ('left','right'):
        tp,vp=paired_stream(alf,sd)
        if tp is not None: side=sd; break
    if side is None: raise FileNotFoundError('no matched camera time/motion stream')
```

iii. Step 1: "Whisker motion energy uses left-camera `whiskerMotionEnergy`, falling back to right camera only if left loading is skipped", mirroring `bin_behaviors`. Step 10 resolved a "whisker source concern" by confirming "from `SessionLoader.load_motion_energy` that left/right `Camera.ROIMotionEnergy` is exactly exposed as `whiskerMotionEnergy`; no source correction required." Step 3 cross-checks the left-camera preference against the method paper's statement that the whisker stream is sampled at 60 Hz.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the trace itself — no filtering, normalisation or baseline subtraction. It goes through exactly the same `interp_trials` path as the wheel: sort, drop duplicate timestamps, window coverage/finiteness check, and linear interpolation with extrapolation onto the 100 bin-end times, then the global tertile split.

ii.
```python
    ct=np.load(tp,mmap_mode='r'); cv=np.load(vp,mmap_mode='r')
    cam,ci=interp_trials(ct,cv,stim,base)
    wt,wv=derive_wheel(alf); wheel,wi=interp_trials(wt,wv,stim,base)
```

iii. Step 5 mapping: "Left/right camera ROI motion energy → prefer left, fallback right; linear interpolation at trial bin ends; global tertile classes 0/1/2 … Require matched timestamp/value arrays and finite interpolated values." Step 10 Check 2 independently interpolated the raw camera arrays for a named trial and compared both the continuous values and the resulting classes with `np.allclose` (max difference 0).

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: one pair of global tertile thresholds (2.7334 and 7.8455 in ROI motion-energy units) computed over every retained whisker sample in the dataset, applied with `searchsorted(..., side='right')`, recorded in `metadata['whisker_motion_energy_thresholds']`. Dataset-wide the classes are within one sample of exact thirds (6,297,499 / 6,297,501 / 6,297,500); within a session they are not balanced, since a session with an overall quiet or active mouse will sit mostly in one class.

ii.
```python
    mthr=thresholds(mz)
    ...
        mc=np.searchsorted(mthr,z['whisker'],side='right').astype(np.uint8)
```

```python
'wheel_speed_thresholds':wthr.tolist(),'whisker_motion_energy_thresholds':mthr.tolist(),
'discretization':'global empirical tertiles across retained session-trial-time samples; searchsorted side=right',
```

iii. Same as 7-c — Step 5 decision 7, "Global thresholds preserve a common physical class meaning across sessions", with the thresholds and method written into metadata and the conversion log for auditability.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: evaluated at `stimOn_times + REL_TIME`, i.e. the right edge of each of the 100 neural bins, on the shared session clock; trials whose camera stream does not span the window to within one bin at both ends are dropped, and the wheel-valid and camera-valid sets are intersected so every retained trial has both streams over the full window.

ii.
```python
        grid=stim[i]+REL_TIME.astype(float)
        y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

```python
    common=np.intersect1d(ci,wi,assume_unique=True)
    cam=np.stack([cam[cmap[i]] for i in common]); wheel=np.stack([wheel[wmap[i]] for i in common])
```

iii. Step 4: "Missing behavior … Explicitly intersect all trial/stream validity masks and exclude undefined trials/sessions; this fixes a clear implementation bug without changing intended processing" (the reference's `align_spike_behavior` mask bug). The camera frame times are on the same clock as the spikes, so evaluating at the bin grid is the whole alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or inconsistent data is detected and the affected trial or session is dropped, never imputed. Specifically: trials with NaN in any required event field are removed by `trial_mask`; trials whose wheel or camera stream does not cover the window, has fewer than two samples, or yields non-finite interpolated values are removed; sessions with no matched camera timestamp/value pair, a missing or short wheel stream, no valid trial table, missing cluster metrics, or fewer than two jointly valid trials raise and are logged as `EXCLUDE <eid>` and skipped (15 sessions: 14 no camera, 1 insufficient coverage). Multiple ALF revisions are handled by explicit selection rules; duplicate wheel/camera timestamps are dropped before interpolation; out-of-range or unknown spike cluster ids are masked out; spike arrays that are not time-sorted are sorted; cluster/channel length or range mismatches raise; an unexpected `probabilityLeft` raises. The whole per-session pass is wrapped in try/except so one bad session cannot abort the run. Trials whose window simply contains no spikes are legitimately kept as all-zero matrices (17 such trials, the same as in the expert reference).

ii.
```python
        try:
            info=preprocess_behavior(r,eid,cf); kept.append((r,eid,cf)); ...
        except Exception as e: print(f'EXCLUDE {eid}: {type(e).__name__}: {e}',flush=True)
```

```python
    keep=np.r_[True,np.diff(times)>0]; times=times[keep]; values=values[keep]
    ...
        valid_id=(ids>=0)&(ids<len(lookup)); spike_t=spike_t[valid_id]; ids=ids[valid_id]
        mapped=lookup[ids]; valid_map=mapped>=0; spike_t=spike_t[valid_map]; mapped=mapped[valid_map]
```

```python
    if len(ch)!=n: raise ValueError(f'cluster channel length {len(ch)} != metrics {n}')
    if np.any((ch<0)|(ch>=len(atlas))): raise ValueError('cluster channel out of range')
```

iii. Step 4/5: "Session inclusion for this task must require all four outputs and at least two jointly valid trials"; Step 10 "Spike mask propagation: corrected handling of unexpected cluster IDs before sample conversion so spike times and mapped IDs always remain aligned." The AI records the per-session exclusion reasons in the conversion log and in CONVERSION_NOTES.md Step 10 Check 6, and argues against deleting sessions merely to reach the method paper's 433: "arbitrary deletion to force 433 would be less reproducible than explicit required-stream criteria."

## 10-a. What are the most time-consuming steps of the code?

i. Measured per session in the conversion log: behaviour loading ≈0.23–0.35 s (dominated by the 1 kHz wheel interpolation/filtering and the camera/trial file reads) and spike binning ≈0.3–1.7 s, scaling with trials × clusters (e.g. 1.69 s for a 1,445-trial, 2,129-neuron session). Full conversion of 444 sessions took 318.7 s. The dominant remaining cost is I/O: reading the per-probe spike arrays (hundreds of MB each) and serialising the 26.7 GB pickle at the end — which is a direct consequence of keeping all 599,865 clusters as 100-bin trial matrices.

ii.
```python
            info=preprocess_behavior(r,eid,cf); ...; info['seconds']=time.time()-ts
            print(f'behavior {k}/{len(sessions)} {eid}: {info}',flush=True)
```

```python
        print(f'convert {si+1}/{len(kept)} {eid}: trials={ntr} neurons={len(preg)} maxcount={maxcount} sec={time.time()-ts:.2f}',flush=True)
```

iii. Step 6/7: "Full output is intrinsically large (~26.5 billion count elements … ) because the applicable methods pipeline retains all clusters"; the run-time table estimates "Behavior alignment ~0.27 s/session … Spike binning + plotting ~0.55 s/session … ~6.5 min plus large-pickle serialization; expected under 15 min", and Step 9 reports the actual 5.31 min, below the 15-minute budget the instructions set.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial Python loops remain, the same two the expert reference keeps: the spike-binning loop in `bin_probe` (one `searchsorted` + `bincount` per trial) and the interpolation loop in `interp_trials` (one `interp1d` per trial). Both could be done in one pass over all trials by offsetting each spike's flat index by its trial index, and by building a single concatenated query vector for the interpolation. There is also a small per-trial loop building the output stacks and a `subjects.index(x)` / `brain_regions.index(x)` linear lookup inside list comprehensions (O(n·k), negligible at 444 sessions but avoidable with a dict). None is a bottleneck: the inner work is already vectorised and the loop bodies are the file-bound part.

ii.
```python
    mats=[]; max_count=0
    for s in stim:
        beg=s+OFF0; end=s+OFF1; ib=np.searchsorted(st,beg,'left'); ie=np.searchsorted(st,end,'left')
```

```python
    for i in idx:
        beg,end=stim[i]+OFF0,stim[i]+OFF1
        ...
        y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

```python
    bri=[np.array([brain_regions.index(x) for x in ss],dtype=np.int32) for ss in session_regions]
```

iii. Step 6: "Python trial loops remain for variable event windows but expensive spike arrays are memory-mapped and each trial uses binary search plus vectorized `bincount`" — i.e. the AI consciously kept the trial loop because each trial is a different slice of the recording, and vectorised the expensive inner operation instead.

## 10-c. What processing does the code repeat multiple times?

i. A few things, none of them documented as such by the AI. (1) `BrainRegions()` — which loads the Allen/Beryl atlas tables — is instantiated inside `bin_probe`, i.e. once per probe (699 times) instead of once per run. (2) The behaviour arrays are written to npz and then read back twice: once in `thresholds()` to compute the global quantiles, and again in the main conversion loop (`z=np.load(cf)`), so the full ~1.5 GB of cached traces is deserialised twice. (3) `resolve_probe` re-globs the probe directory for each file name it needs, and `full_trial_table`/`paired_stream` parse every candidate revision of an object before choosing one. (4) `subjects.index` / `brain_regions.index` re-scan their lists per element. Against that, the two-pass design deliberately avoids the one genuinely expensive repeat — the 1 kHz wheel interpolation and filtering is done once per session and never recomputed during spike binning.

ii.
```python
    br=BrainRegions(); native=br.id2acronym(atlas[ch]); regions=br.acronym2acronym(native,mapping='Beryl').astype(str)
```

```python
    wz=[np.load(cf)['wheel'] for _,_,cf in kept]; mz=[np.load(cf)['whisker'] for _,_,cf in kept]
    wthr=thresholds(wz); mthr=thresholds(mz); del wz,mz
    ...
    for si,(r,eid,cf) in enumerate(kept):
        ts=time.time(); z=np.load(cf); stim=z['stim']
```

iii. The AI documented the intent of the two-pass design (Step 6: "Two-pass behavior caches avoid recomputing wheel filtering/interpolation during spike processing"; "Session/probe streaming avoids holding all raw spikes in RAM") but did not identify the repeated atlas construction or the double read of the behaviour caches. The two-pass structure is itself required by decision 7 (global thresholds need all sessions' traces before any output can be discretised), so the extra disk round-trip is the price of that decision.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest item follows from decision 2-c: all 599,865 clusters are binned and stored, including ~112k label-0 clusters, 85,656 `root` units and 12,827 `void` units that are anatomically outside the brain. That is roughly 8× more neurons than a QC-filtered dataset and produces a 26.7 GB pickle; the supplied decoder then caps SVD initialisation at 2,000 neurons per session and random-projects anything larger, so much of that data is compressed away (or dilutes the signal) rather than used. Smaller items: the camera and wheel traces are interpolated independently for every base-mask trial and only afterwards intersected, so work is done for trials that are dropped because the *other* stream fails; `max_count` is computed over every trial matrix purely to pick between `uint8` and `uint16`; `clusters.metrics.pqt` is fully parsed although only `cluster_id` (and, for validation, `label`) is used; and the npz behaviour caches are written to disk although the run is a single process that could have kept them in memory.

ii.
```python
    cam,ci=interp_trials(ct,cv,stim,base)
    wt,wv=derive_wheel(alf); wheel,wi=interp_trials(wt,wv,stim,base)
    common=np.intersect1d(ci,wi,assume_unique=True)      # work done for trials only one stream covers
```

```python
        max_count=max(max_count,int(a.max(initial=0))); mats.append(a)
    ...
        if maxcount>255: dtype=np.uint16
        else: dtype=np.uint8
```

iii. The AI justifies the size as intrinsic to its curation decision (Step 6: "Full output is intrinsically large … because the applicable methods pipeline retains all clusters") and justifies the integer dtypes as a deliberate 4× saving over float32 (Step 7). It does not flag the redundant interpolation of trials that the other stream will reject, nor that no-QC retention is partly discarded by the decoder's 2,000-neuron SVD cap; it did, however, confirm the cap was exercised ("Large-session SVD initialization used the decoder's intended 50,000-timepoint and 2,000-neuron caps") without connecting that to the low choice accuracy.
