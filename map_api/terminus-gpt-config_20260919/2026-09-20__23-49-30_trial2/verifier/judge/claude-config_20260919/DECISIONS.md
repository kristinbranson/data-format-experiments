# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every NWB file by a recursive glob under the hard-coded root `/app/data`, sorts the paths for determinism, and processes each file exactly once in a single serial pass. Every file is opened with `pynwb.NWBHDF5IO(..., load_namespaces=True)` inside a `with` block (no `h5py` anywhere). Within a session, subjects come from `nwb.subject.subject_id`, trials from `nwb.trials`, units from `nwb.units`, and behavioural streams from `nwb.acquisition['BehavioralEvents']` / `['BehavioralTimeSeries']`. Column access goes through a small helper `arrcol()` that reads `table[name].data[:]` rather than materialising a DataFrame. `--sample` stops after the first 2 *usable* sessions; `--full` processes all 174. Result: 174 files discovered, 173 kept, 28 subjects, 90,378 trials, 69,453 units.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb')); target=2 if args.sample else None
print(f'Discovered {len(files)} NWB files; mode={"sample" if args.sample else "full"}',flush=True)
sessions=[]; t0=time.time()
for p in files:
    st=time.time(); res=convert_session(p,make_plot=args.show_processing and len(sessions)<2)
    if res is None: print(f'SKIP {p.name}: insufficient valid neurons/trials',flush=True); continue
    sessions.append(res)
    ...
    if target and len(sessions)>=target: break
```
```python
def convert_session(path, make_plot=False):
    t0=time.time()
    with NWBHDF5IO(str(path),'r',load_namespaces=True) as io:
        nwb=io.read(); tr=nwb.trials
        ev=nwb.acquisition['BehavioralEvents'].time_series
```
```python
def arrcol(table, name, dtype=None):
    a=np.asarray(table[name].data[:])
    return a.astype(dtype) if dtype is not None else a
```

iii. From CONVERSION_NOTES.md Step 2: "`/app/data` contains 174 NWB files nested in dataset subdirectories; all access and inspection used `pynwb.NWBHDF5IO` with loaded namespaces (never `h5py`)." The AI ran a pynwb-only survey pass over all 174 files first (44 s) to establish subject/session/trial/unit totals before writing the converter, and cross-checked the totals against the papers (173 behavioural sessions, 69,943 good units). It avoided `to_dataframe()` on `units` deliberately as a speed-up ("avoids full Units DataFrame materialization").

## 1-b. How are the data split into subjects (mice)?

i. Each session's subject is read directly from the NWB subject metadata as a string. At assembly time the AI takes the sorted unique set as `subjects` and builds `subject_idx` as an int64 index into that list, one entry per retained session, in session order. This yields the 28 numeric DANDI subject ids (e.g. `440956`), with 3–10 sessions each.

ii.
```python
info=dict(file=path.name,identifier=nwb.identifier,subject=str(nwb.subject.subject_id), ...)
...
return dict(neural=neural,input=inputs,output=outputs,subject=info['subject'],regions=names.tolist(),info=info,plot=plot_payload)
```
```python
subjects=sorted({x['subject'] for x in sessions}); ... smap={v:i for i,v in enumerate(subjects)}
data={... 'subjects':subjects,'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int64), ...}
```

iii. Step 2/Step 5 notes: "Subject identifiers are in `nwbfile.subject.subject_id`" and "Sorted unique string IDs and per-session index … 28 subjects expected." The AI verified the count (28) against its own full-dataset scan and reported per-subject session counts in the verification log. No grouping heuristic is needed because the field is canonical and the `sub-*` directory name is derived from it.

## 1-c. How are the data split into sessions?

i. One NWB file == one session; no splitting or grouping is performed. Session order in the output is the sorted file order (which is chronological within a subject because the filename embeds the acquisition timestamp). Each session records `file`, `identifier` (e.g. `SC015_20190207_120657_s1`), and per-session counts in `metadata['session_info']`. A session is dropped entirely (returns `None`) if it has no classifier-good units or fewer than 2 usable trials.

ii.
```python
files=sorted(Path('/app/data').rglob('*.nwb'))
```
```python
        units,trial_idx=select_units_and_trials(nwb,structural_idx)
        candidates=units
        if units.size==0 or trial_idx.size<2: return None
```
```python
'session_info':[x['info'] for x in sessions]
```

iii. Step 2: "Each NWB represents one electrophysiology/behavior session." Step 9/10: exactly one file (`sub-440958_ses-20190216T162508`, which has NaN `classification` for all units) is excluded, reconciling the released 174 files with the papers' "173 behavioral sessions". The AI explicitly documents this as the reason its session count matches the paper.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB `trials` table, paired one-to-one with the `go_start_times` event stream. The AI first defensively truncates to `n = min(len(trials), len(go))`, then defines "structurally valid" trials as those rows that have an indexed go cue **and** at least one `sample_start_times` (tone) event at or before that go cue. Trials with `ix < 0` (no preceding tone — i.e. the first trial of a session if the tone stream starts late) are discarded. Everything downstream is indexed by the surviving integer trial indices (`trial_idx`). The AI separately verified in three sessions that there is exactly one go cue per trial and that all go cues fall inside the trial bounds.

ii.
```python
        go=np.asarray(ev['go_start_times'].timestamps[:],dtype=np.float64)
        sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
        n=min(len(tr),len(go))
        # Structurally valid completed trials: one indexed go and a preceding tone.
        ix=np.searchsorted(sample,go[:n],side='right')-1
        structural_idx=np.flatnonzero(ix>=0)
        if structural_idx.size<2: return None
```

iii. Step 2: "Go-cue checks in three sessions found exactly one go cue per trial, all within trial bounds." Step 4 resolves the sample/go mismatch: "Sample events can outnumber completed trials … Latest sample start before each go yields one tone/trial." The trial table is used directly rather than re-deriving boundaries because it already encodes one row per behavioural trial with all required labels; the tone requirement exists because `input[0]` (time from tone onset) is undefined without one.

## 1-e. How are trials filtered based on quality controls?

i. Four successive filters, all documented:
1. **Structural**: must have a go cue and a preceding tone (see 1-d).
2. **Ephys observation validity**: for every classifier-good unit, the AI reads that unit's `obs_intervals` and `is_good_trials`, maps each observation interval onto a behavioural trial row (candidate = latest trial starting before the interval end, accepted only if temporal overlap > 0), and keeps the **intersection** across all good units of trials that are both mapped and flagged `is_good_trials`. Results are cached per unique `obs_intervals` array. This removed 1,569 of 94,370 trials in 12 sessions.
3. **All-zero neural windows**: after binning, any trial in which *no* selected unit emitted a single spike anywhere in the 4 s window is dropped as a recording gap. This removed 2,423 trials across 93 sessions.
4. **Minimum session size**: a session with fewer than 2 surviving trials is dropped.

Notably, the AI does **not** apply an explicit `free_water` exclusion, nor does it exclude early-lick / ignore / photostim / auto-water trials. Final: 90,378 trials retained.

ii.
```python
def select_units_and_trials(nwb, structurally_valid_trials):
    cls=arrcol(nwb.units,'classification').astype(str)
    units=np.flatnonzero(cls=='good')
    if units.size==0: return units, np.asarray([],dtype=int)
    starts=arrcol(nwb.trials,'start_time',np.float64); stops=arrcol(nwb.trials,'stop_time',np.float64)
    common=set(map(int,structurally_valid_trials)); map_cache={}
    for j in units:
        obs=np.asarray(nwb.units['obs_intervals'][j],dtype=np.float64)
        valid=np.asarray(nwb.units['is_good_trials'][j],dtype=bool)
        ...
        mapped=np.searchsorted(starts,obs[:,1],side='left')-1
        mapped=np.clip(mapped,0,len(starts)-1)
        overlap=np.minimum(stops[mapped],obs[:,1])-np.maximum(starts[mapped],obs[:,0])
        mapped=np.where(overlap>0,mapped,-1).astype(int)
        common.intersection_update(map(int,mapped[valid & (mapped>=0)]))
        if not common: break
    return units, np.asarray(sorted(common),dtype=int)
```
```python
        # Remove neural recording-gap trials: no selected unit emitted any
        # spike anywhere in the requested four-second window.
        neural_valid=np.any(rates>0,axis=(1,2))
        dropped_zero_neural=int((~neural_valid).sum())
        trial_idx=trial_idx[neural_valid]
        go=go[neural_valid]; tone=tone[neural_valid]; rates=rates[neural_valid]
        ...
        if len(go)<2: return None
```

iii. Step 1/Step 4: "Reference figure analyses often define regular trials by excluding early-lick, auto/free-water, and stimulation trials. Those exclusions are analysis-specific: this task explicitly requires decoding early lick and representing photostimulation, so such trials must not be removed merely for belonging to those classes." Step 13 records the `is_good_trials` correction: "NWB `is_good_trials` is indexed over each unit's local `obs_intervals`, not always over the full behavioral trial table. Direct session-row indexing was incorrect. Fix: map each observation interval to the overlapping behavioral trial … and retain the intersection of valid observed trials across classifier-good units." This was chosen over dropping the 565 drift-affected units so that all 69,453 classifier-good units are preserved with fixed neuron dimensions and no imputation. The all-zero filter was added after the validator flagged 2,423 such trials: "Direct raw checks confirmed correct go alignment and truly zero spikes across all selected units in these windows."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `nwb.units['spike_times']` (ragged, session-absolute seconds), restricted to units with `nwb.units['classification'] == 'good'`, together with `BehavioralEvents/go_start_times` timestamps which place the bin edges. `nwb.units['anno_name']` supplies the per-neuron brain-region label (stored verbatim as the full Allen CCF string, 293 unique labels).

ii.
```python
        cls=arrcol(nwb.units,'classification').astype(str)
        units=np.flatnonzero(cls=='good')
```
```python
        # Read each selected ragged spike vector once; histogramming below is in C.
        spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
```
```python
        names=arrcol(nwb.units,'anno_name').astype(str)[units]
        if np.any(names==''): raise ValueError(f'Blank anatomy among selected units: {path.name}')
```

iii. Step 2: "`nwbfile.units` contains spike times, waveform/spike-sorting metrics, `unit_quality`, classifier `classification`, Allen CCF `anno_name` …". Spike times are the only neural representation in the file. Step 5 maps `units.spike_times → neural` via `sliding_histogram`-equivalent logic from the reference repo.

## 2-b. How is the `neural` data processed?

i. Per-bin spike **counts divided by the 50 ms bin width → firing rate in Hz**, float32. Implementation: build an `(n_trials, 81)` matrix of absolute bin edges, then for each unit do a single `np.searchsorted` over that whole matrix and `np.diff` along the edge axis, which yields counts per bin for all trials at once. No smoothing, no normalisation, no baseline subtraction, no z-scoring. Rates are stored as `(n_trials, n_units, n_time)` and sliced per trial into `(n_units, 80)`.

ii.
```python
        # Vectorize binning across all trials: one searchsorted call per unit.
        # np.diff is along each trial's 81 edges, so no cross-trial bins are introduced.
        edge_matrix=go[:,None]+EDGES_REL[None,:]
        rates=np.empty((len(go),len(units),N_TIME),dtype=np.float32)
        for ui,sp in enumerate(spikes):
            rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1).astype(np.float32)/BIN_S
```

iii. Step 5, decision 5: "firing rates in Hz (count / 0.05 s), matching reference `rate=True`; float32 limits memory." The reference repo's `sliding_histogram(..., rate=True)` returns `binSpikes / bin_width`, so Hz is the reference-consistent representation. Step 10 records an independent check: "Independent pynwb checks for three unit/trial firing-rate vectors matched conversion exactly (`np.allclose`, maximum difference 0)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single unit-level filter: keep units whose `classification` column equals `'good'` — the verdict of the region-specific logistic-regression QC classifier described in the Chen/Liu et al. 2023 white paper. **No individual quality-metric thresholds** are applied, and `unit_quality` ('good'/'multi') is deliberately not used. A session with zero such units is dropped. Per-unit temporal validity (`is_good_trials`) is *not* used to drop units — instead it is pushed onto the trial axis (see 1-e), so that all classifier-good units survive. The AI also asserts that every retained unit has a non-empty anatomical label. Result: 69,453 units retained out of 272,227 raw (25.5%).

ii.
```python
    cls=arrcol(nwb.units,'classification').astype(str)
    units=np.flatnonzero(cls=='good')
    if units.size==0:
        return units, np.asarray([],dtype=int)
```
```python
        if units.size==0 or trial_idx.size<2: return None
```

iii. Step 3: "The QC white paper explicitly warns that thresholding individual metrics causes unacceptable misses/false alarms; therefore NWB `classification == good`, not `unit_quality`, is the reference-consistent filter." Step 1 confirms the reference batch driver runs `qc_mode='classifier'`. Step 9 compares 69,453 retained units and 25.5% retention against the papers' 69,943 units / 25.9% of Kilosort2 clusters, and documents the 490-unit residual as a release/manuscript difference rather than inventing units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**, using the `BehavioralEvents/go_start_times` timestamps. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed: the fixed relative edge vector `EDGES_REL` (−2.5 … +1.5 s) is added to each trial's go-cue time to produce that trial's absolute edges, and spikes are searchsorted against those directly.

ii.
```python
BIN_S=0.050
OFF_START=-2.5
OFF_END=1.5
EDGES_REL=np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
```
```python
        edge_matrix=go[:,None]+EDGES_REL[None,:]
```
```python
'temporal_alignment_event':'go cue onset','off_start':OFF_START,'off_end':OFF_END
```

iii. Step 4 discrepancy table: "Author export says spikes already relative to go / NWB spikes/events are absolute session time → Subtract each NWB trial go timestamp before binning, equivalently histogram absolute spikes with go-shifted edges." Step 10 check 2 independently re-histogrammed raw spike times from pynwb for selected unit/trial pairs and got exactly zero difference, confirming no off-by-one or offset error.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **50 ms non-overlapping bins**, 80 bins per trial spanning −2.5 s to +1.5 s around the go cue; bin centres at −2.475 … +1.475 s. The grid is built once at module level and reused for every trial and every session, so all trials share an identical time axis. No sliding window, no overlap, no post-hoc rebinning or downsampling; the binning is done once, directly from raw spike times. `metadata['time_bin_size'] = 50.0` (ms) and the bin centres are stored in metadata. The AI explicitly departs from the method paper's 40 ms window / 3.4 ms stride over −3 … +3 s.

ii.
```python
BIN_S=0.050
EDGES_REL=np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL=(EDGES_REL[:-1]+EDGES_REL[1:])/2
N_TIME=len(CENTERS_REL)
```
```python
    assert N_TIME==80 and len(EDGES_REL)==81
```
```python
'time_bin_size':50.0,'time_bin_units':'ms', ... 'n_timepoints':N_TIME,'bin_centers_seconds':CENTERS_REL.astype(np.float32)
```

iii. Step 4: "Neural binning — 40-ms sliding windows, 3.4-ms stride, −3 to +3 s … Decoder specification overrides with 50-ms non-overlapping bins from −2.5 to +1.5 s (80 bins)." The AI treats this as one of the sanctioned discrepancies ("required by the Decoder Input and Decoder Output specifications"). The `OFF_END + BIN_S/2` term in `arange` is a guard against floating-point edge loss, and the `N_TIME==80 and len(EDGES_REL)==81` assertion in `validate()` enforces it.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the auditory sample-epoch/tone onsets) together with `BehavioralEvents/go_start_times`. For each trial the AI takes the **last tone at or before that trial's go cue** (`searchsorted(..., side='right') - 1`), which handles the fact that a lick during the sample epoch replays that epoch and produces multiple tone events per trial.

ii.
```python
        sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
        n=min(len(tr),len(go))
        ix=np.searchsorted(sample,go[:n],side='right')-1
        structural_idx=np.flatnonzero(ix>=0)
        ...
        go=go[trial_idx]; tone=sample[ix[trial_idx]]
```

iii. Step 4: "Sample events can outnumber completed trials / Latest sample start before each go yields one tone/trial; median tone-to-go 1.85 s → Pair each go with latest preceding sample start; variable delays are preserved in elapsed-time input." The 1.85 s median matches the task structure (0.65 s sample + 1.2 s delay).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value per bin: the **absolute time of the bin centre minus the trial's tone onset**, in seconds, cast to float32. It is written into row 0 of the `(2, 80)` input matrix. Because the bins are go-aligned and the tone-to-go interval varies by trial, the vector is a ramp whose offset encodes the delay length. Values span [−1.5, 11.9] s over the full dataset (negative early in the window when the tone had not yet occurred; the long tail comes from rare trials with repeated sample replays).

ii.
```python
            centers=g+CENTERS_REL
            inp=np.empty((2,N_TIME),dtype=np.float32)
            inp[0]=(centers-to).astype(np.float32)
```

iii. Step 5, decision 8: "elapsed time from tone onset is represented continuously as explicitly required, despite the generic format note suggesting binary onset series for onset-only inputs." The AI read the Decoder Task spec as unambiguous ("Time from tone onset in seconds (continuous, time-varying)") and let it override the generic "represent onset times as a binary time series" formatting note. Step 10 verified the vector against an independent event-pairing computation with `np.allclose` (max diff < 2.3e-7, float32 rounding only).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same grid used to bin the spikes: `centers = go + CENTERS_REL`, where `CENTERS_REL` is the midpoint vector of the same `EDGES_REL` array used to build the spike-count edges. So bin *k* of the input covers the same interval as bin *k* of the firing rates by construction, with no interpolation and no separate clock.

ii.
```python
CENTERS_REL=(EDGES_REL[:-1]+EDGES_REL[1:])/2
```
```python
        edge_matrix=go[:,None]+EDGES_REL[None,:]      # neural
...
            centers=g+CENTERS_REL                      # inputs/outputs
```

iii. Step 5, decision 4: "80 bins with edges [−2.5, 1.5] and centers [−2.475, 1.475] relative to go; metadata offsets describe edges." Step 12 check 5: "direct raw spike histograms and event-derived inputs passed 108 independent checks."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The `BehavioralEvents/photostim_start_times` and `photostim_stop_times` **event timestamp streams** (session-absolute), not the `trials` table's string-valued `photostim_onset` / `photostim_duration` columns. These two streams are equal-length and index-paired (I verified in `sub-440956_ses-20190207`: 78 starts, 78 stops, all durations 0.5 s, and the start timestamps are numerically identical to `trials.start_time + float(photostim_onset)`).

ii.
```python
        pstart=np.asarray(ev['photostim_start_times'].timestamps[:],dtype=np.float64)
        pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],dtype=np.float64)
```

iii. Step 4: "Photostimulation — Trial flags differ slightly from event overlap in 13 sessions because some stimulation is outside target window … Build time-varying input from start/stop event timestamps; a flagged trial may correctly be all-zero inside the requested window." Step 4 final understanding: "Event timestamps, rather than string-formatted trial photostimulation fields, are authoritative for temporal inputs." Step 3 also notes "ALM photoinhibition occurs during the last 0.5 s of delay and ends before go cue," so the AI expected and accepted all-zero rows on flagged trials whose stim falls outside the extracted window.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A **binary (0/1) time-varying series**, one value per bin, stored as float32 in row 1 of the input matrix. For each bin centre the AI finds the latest stimulation interval starting at or before it (`searchsorted(pstart, centers, 'right') - 1`) and sets 1 if the centre is still before that interval's stop — i.e. membership in a half-open `[start, stop)` interval. Sessions with no photostim events get an all-zero row. Full-dataset input range is [0, 1] as verified.

ii.
```python
            if len(pstart):
                # true at center if any half-open stimulation interval contains center
                jj=np.searchsorted(pstart,centers,side='right')-1
                valid=jj>=0; stim=np.zeros(N_TIME,dtype=bool)
                stim[valid]=centers[valid] < pstop[jj[valid]]
                inp[1]=stim.astype(np.float32)
            else: inp[1]=0
```
```python
    assert set(np.unique(i[1])).issubset({0.,1.})
```

iii. Step 5 mapping table: "`photostim_start_times`, `photostim_stop_times` → `input[1]` — Binary 1 when bin center is within any `[start, stop)` interval, else 0. Uses event timing rather than trial strings." This satisfies the instruction "Whether photostimulation is on at every time point (discrete, time-varying)" and the format note that onset times be represented as a binary series. The `validate()` assertion enforces strict binarity.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Both stimulation events and bin centres are on the same session-absolute clock, and the bin centres are the same go-aligned `g + CENTERS_REL` used for the firing rates, so the comparison is direct — no offset correction, no interpolation. The AI validated the resulting waveform visually in `--show-processing` plots (photostim step function overlaid on the go-relative axis) and numerically in the Step 10 sanity checks.

ii.
```python
            centers=g+CENTERS_REL
            ...
                jj=np.searchsorted(pstart,centers,side='right')-1
                stim[valid]=centers[valid] < pstop[jj[valid]]
```
```python
    ax[1].plot(CENTERS_REL,inputs[ti][1],label='photostim on'); ax[1].axvline(0,color='k',ls='--')
```

iii. Step 10 check 2: "It performed 108 `np.allclose` checks … Every neural histogram, tone input, photostim input, static output, and tongue vector passed." Step 7 plot review: "correctly timed binary photostimulation … No temporal-alignment anomaly was observed."

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not stored in the file, so it is **derived deterministically from `trials.trial_instruction` × `trials.outcome`**: on a `hit` the mouse licked the instructed side, on a `miss` it licked the opposite side, on an `ignore` it did not lick. The AI validated this derivation against the actual first post-go `left_lick_times` / `right_lick_times` event in three sessions and got 100% agreement.

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
    raise ValueError(f'Unknown outcome {outcome!r}')
```

iii. Step 2: "Choice reconstructed from instruction/outcome agreed 100% with first post-go left/right lick: hit=instructed side, miss=opposite side, ignore=no lick." Step 5 mapping table cites the reference repo's `lick_directions` output as the analogue. The unknown-outcome `raise` makes any unanticipated category a hard failure rather than a silent miscoding.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, matching `output_values[0] = ['left','right','no lick']`. It is a per-trial scalar, but it is **broadcast across all 80 time bins** into row 0 of an `(4, 80)` int64 output matrix, so that all four outputs share a single common time axis. Full-dataset distribution: left 0.428, right 0.422, no lick 0.149.

ii.
```python
            out=np.empty((4,N_TIME),dtype=np.int64)
            out[0]=trial_choice(instruction[k],outcome_s[k])
```
```python
'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],[...]],
```
```python
            assert o[0].max()<3 and o[1].max()<3 and o[2].max()<2 and o[3].max()<4
```

iii. Step 5, decision 9 / Step 6: "Repeats static trial labels across 80 time bins because the supplied validator requires every output dimension to share a common time axis." Step 34 of the trajectory: "The validator determines output dimensionality from `shape[0]` and expects consistent time dimensions. Mixed scalar/time-varying outputs therefore cannot be represented as ragged rows. The correct compatible representation is a `(4, 80)` integer matrix per trial." The 0/1/2 coding follows the Decoder Task ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of `nwb.trials`, which already contains exactly the three required strings `'ignore'`, `'miss'`, `'hit'`. No derivation.

ii.
```python
        outcome_s=arrcol(tr,'outcome').astype(str)[trial_idx]
```

iii. Step 2: "Trial labels have the exact required early-lick and outcome categories." Native counts across all 174 files: ignore 14,095; miss 15,641; hit 65,254.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore → 0`, `miss → 1`, `hit → 2` (the order given in the Decoder Task), and the per-trial value is broadcast across all 80 bins into row 1 of the output matrix. Any unexpected string raises `KeyError` rather than being silently coerced. Full-dataset distribution: ignore 0.149, miss 0.166, hit 0.685 — consistent with the native raw counts after curation.

ii.
```python
            out[1]={'ignore':0,'miss':1,'hit':2}[outcome_s[k]]
```
```python
'output_values':[..., ['ignore','miss','hit'], ...]
```

iii. Step 5 mapping table: "`trials.outcome` → `output[1]` — ignore=0, miss=1, hit=2 — Per-trial categorical." Step 12 check 6 re-verified the converted class fractions against the raw label counts.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of `nwb.trials`, whose values are the strings `'no early'` and `'early'`. No derivation from lick event streams.

ii.
```python
        early_s=arrcol(tr,'early_lick').astype(str)[trial_idx]
```

iii. Step 5 mapping table cites the reference repo's `early_lick_trials` field as the direct analogue. Step 1 flagged that reference analyses often *exclude* early-lick trials, and the AI explicitly rejected that here: "this task explicitly requires decoding early lick … so those valid classes must be retained."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Fixed dictionary `'no early' → 0`, `'early' → 1`, broadcast across all 80 bins into row 2 of the output matrix. Full-dataset distribution: no 0.884, yes 0.116 (raw: 84,185 / 10,805 ≈ 0.886 / 0.114).

ii.
```python
            out[2]={'no early':0,'early':1}[early_s[k]]
```
```python
'output_values':[..., ['no','yes'], ...]
```

iii. Step 5 mapping table: "`trials.early_lick` → `output[2]` — no early=0, early=1 — Per-trial categorical; trials retained." Coding order matches the Decoder Task ("Early lick (no, yes)"). The early lick itself occurs during the sample or delay epoch, so the causally relevant neural activity lies inside the −2.5 s pre-go window.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, a DeepLabCut-style `(n_frames, 3)` array `= (tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps` at ~3.4 ms (~294 Hz). Column 1 is the y value, column 2 the tracking likelihood. The AI selects `Camera0_side_TongueTracking` by name when present and otherwise falls back deterministically to the lexicographically first series whose name contains `TongueTracking`; if there is none, it returns all-NaN thresholds so every bin becomes "not visible".

ii.
```python
def get_tongue(nwb):
    series=nwb.acquisition['BehavioralTimeSeries'].time_series
    keys=sorted(k for k in series if 'TongueTracking' in k)
    if not keys:
        return None,None,None,None,(np.nan,np.nan)
    key='Camera0_side_TongueTracking' if 'Camera0_side_TongueTracking' in keys else keys[0]
    ts=series[key]
    times=np.asarray(ts.timestamps[:],dtype=np.float64)
    data=np.asarray(ts.data[:],dtype=np.float32)
    y=data[:,1]; likelihood=data[:,2]
```

iii. Step 2: "`BehavioralTimeSeries` contains side-camera jaw, nose, and tongue x/y/likelihood traces, normally at 3.4-ms sampling. Some sessions additionally contain lick-port tracking or a second camera." Step 5: "Prefer Camera0 for consistency; fall back deterministically to lexicographically first tongue stream only if Camera0 absent."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) **Visibility gating**: a frame counts only if `y` and `likelihood` are finite and `likelihood >= 0.90`; the tracker emits a position even when the tongue is retracted, so low-confidence frames must not be treated as measurements. (2) **Session thresholds**: the 40th and 60th percentiles of `y` are computed over *all visible frames in the whole session* (raw frames, not bin averages). (3) **Per-bin sampling**: for each go-aligned bin centre the AI takes the **nearest video frame** (not a within-bin average), requiring that the centre lie inside the video coverage span *and* that the nearest frame be within 10 ms; the frame's y is then thresholded. Bins with no covered/confident frame get class 3. Full-dataset result: 0.062 / 0.032 / 0.065 / 0.841 — i.e. ~15.9 % of bins visible, split ~39 / 20 / 41 % within the visible fraction.

ii.
```python
    visible=np.isfinite(y)&np.isfinite(likelihood)&(likelihood>=LIKELIHOOD_THRESHOLD)
    thresholds=tuple(np.percentile(y[visible],[40,60]).tolist()) if visible.any() else (np.nan,np.nan)
```
```python
            tc=np.full(N_TIME,3,dtype=np.int64); vis=np.zeros(N_TIME,dtype=bool)
            if vt is not None and len(vt):
                q=nearest_indices(vt,centers)
                # Do not extrapolate beyond video coverage; nearest frame must also be temporally close.
                covered=(centers>=vt[0])&(centers<=vt[-1])&(np.abs(vt[q]-centers)<=0.010)
                vis=covered&np.isfinite(vy[q])&np.isfinite(vl[q])&(vl[q]>=LIKELIHOOD_THRESHOLD)&np.isfinite(p40)&np.isfinite(p60)
```

iii. Step 4: "Likelihood is strongly bimodal; 13 sessions have incomplete video coverage → Use likelihood >= 0.9 as visible; uncovered or low-confidence bins are class 3. Compute session percentiles only from visible y samples. Threshold choice has negligible impact because confidence is near 0 or 1 and will be plot-validated." Step 3 notes the threshold "is not stated clearly in the supplied prose/code, so a conventional 0.9 threshold will be evaluated during mapping and visual validation." Step 7: "High not-visible frequency is expected because tongue confidence is high primarily during licking."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, exactly as specified: `0` below the session 40th percentile, `1` between the 40th and 60th percentiles **inclusive at both edges**, `2` above the 60th percentile, `3` not visible. Percentiles are per-session and computed from visible frames only. Equality at a boundary is assigned to the middle class. `output_values[3] = ['<40th percentile','40th to 60th percentile','>60th percentile','not visible']`. The p40/p60 values are stored per session in `metadata['session_info']` for auditability.

ii.
```python
LIKELIHOOD_THRESHOLD=0.90
```
```python
                yy=vy[q]; tc[vis & (yy<p40)]=0; tc[vis & (yy>=p40)&(yy<=p60)]=1; tc[vis & (yy>p60)]=2
            out[3]=tc
```
```python
'output_values':[..., ['<40th percentile','40th to 60th percentile','>60th percentile','not visible']],
'tongue_discretization':'Per-session visible-frame y percentiles: <p40, p40-p60, >p60; low-confidence/uncovered=not visible'
```

iii. Step 5, decision 7: "compute p40/p60 from all confidence-valid y frames across the session, as 'over the session' requires. Equality is assigned to the middle class." The `--show-processing` plots draw the raw y trace, the likelihood trace, the p40/p60 lines and the resulting class series on one go-relative axis, plus the session class-fraction histogram, specifically to make the discretisation visually checkable.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the same session-absolute clock as spikes and events, so alignment is by nearest-neighbour resampling of the video onto the identical go-aligned bin-centre grid used for the firing rates: `centers = go + CENTERS_REL`, `q = nearest_indices(vt, centers)`. Two guards prevent silent extrapolation: the centre must lie within `[vt[0], vt[-1]]`, and the matched frame must be within 10 ms of the centre (comfortably satisfied at 3.4 ms frame spacing, ≤1.7 ms worst case). Anything outside coverage — including the leading bins of trials where the video is trial-gated and starts after `go − 2.5 s` — becomes class 3.

ii.
```python
def nearest_indices(sorted_times, query):
    """Nearest indices in monotonically increasing sorted_times."""
    j=np.searchsorted(sorted_times, query, side='left')
    j=np.clip(j, 0, len(sorted_times)-1)
    prev=np.maximum(j-1,0)
    use_prev=np.abs(query-sorted_times[prev]) <= np.abs(sorted_times[j]-query)
    return np.where(use_prev,prev,j)
```
```python
                q=nearest_indices(vt,centers)
                covered=(centers>=vt[0])&(centers<=vt[-1])&(np.abs(vt[q]-centers)<=0.010)
```

iii. Step 5 mapping table: "Resample nearest video frame to each bin center; if no coverage or likelihood<0.9 → 3." Step 4: "13 sessions have incomplete tongue-video coverage, so uncovered time bins should be labeled 'not visible', not used for percentile thresholds." Step 10 check 2 independently re-derived three trials' tongue vectors straight from pynwb and compared with `np.allclose`; all passed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six cases, each with an explicit policy:
- **Session never quality-controlled** (`classification` is NaN for all 1,852 units in `sub-440958_ses-20190216T162508`): `.astype(str)` turns NaN into `'nan'`, no unit matches `'good'`, and the session is dropped. This is what reconciles 174 files with the papers' 173 sessions.
- **Trials with no ephys observation / flagged bad by `is_good_trials`**: removed via the obs-interval→trial mapping intersection (1,569 trials).
- **Trials with a recording gap** (no spike from any unit in the whole 4 s window): removed after binning (2,423 trials).
- **Trials with no preceding tone event**: removed by the `ix >= 0` structural filter.
- **Missing/short video coverage or low-confidence tracking**: mapped to the explicit "not visible" class 3 rather than imputed or dropped, so otherwise valid neural trials are kept.
- **Sessions with no photostim events**: the photostim input row is set to all zeros rather than erroring.
- Additionally, unexpected label strings (`outcome`, `early_lick`) raise rather than being silently coerced, and a blank `anno_name` on a selected unit raises `ValueError` (never triggered — all 69,453 good units are labelled).

ii.
```python
    cls=arrcol(nwb.units,'classification').astype(str)
    units=np.flatnonzero(cls=='good')
    if units.size==0: return units, np.asarray([],dtype=int)
```
```python
        if np.any(names==''): raise ValueError(f'Blank anatomy among selected units: {path.name}')
```
```python
            else: inp[1]=0
```
```python
            tc=np.full(N_TIME,3,dtype=np.int64); vis=np.zeros(N_TIME,dtype=bool)
            if vt is not None and len(vt):
```
```python
def validate(data):
    ...
            assert np.isfinite(n).all() and np.isfinite(i).all() and (n>=0).all()
```

iii. The governing principle in Step 5/Step 9: where a measurement was genuinely never recorded, the session or trial is excluded rather than emitted as fabricated zeros ("Direct raw checks confirmed correct go alignment and zero spikes across all selected units; these invalid trials were removed"); where the measurement legitimately has no value (retracted tongue, video not rolling), it is represented as an explicit fourth category, because "missing video coverage maps to the mandated not-visible class rather than dropping otherwise valid neural trials." The AI also chose to *keep* all classifier-good units and push per-unit invalidity onto the trial axis so that no neural values are imputed and neuron dimensions stay fixed.

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion runs 209 s for 174 files (~1.2 s/session, range 0.59–2.38 s), and pickling the 12.0 GB result adds tens of seconds. The dominant costs are HDF5 I/O and per-unit work, both scaling with unit count: (1) `select_units_and_trials` issues **two HDF5 reads per classifier-good unit** (`obs_intervals[j]`, `is_good_trials[j]`) — up to ~1,800 reads per session — and only caches the *mapping* after the read, not the read itself; (2) `spikes=[np.asarray(nwb.units['spike_times'][j]) ...]` performs one ragged `VectorIndex.__getitem__` per unit rather than reading the underlying buffer once; (3) the per-unit `np.searchsorted` over the `(n_trials, 81)` edge matrix; (4) loading the full `(n_frames, 3)` tongue array (~680 k × 3). The AI did not profile at this granularity, but it did identify and eliminate the original dominant cost.

ii.
```python
    for j in units:
        obs=np.asarray(nwb.units['obs_intervals'][j],dtype=np.float64)
        valid=np.asarray(nwb.units['is_good_trials'][j],dtype=bool)
```
```python
        spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
        edge_matrix=go[:,None]+EDGES_REL[None,:]
        for ui,sp in enumerate(spikes):
            rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1).astype(np.float32)/BIN_S
```
```python
        st=time.time(); res=convert_session(p,...)
        ... print(f"[{len(sessions)}] {p.name}: ... {time.time()-st:.2f}s",flush=True)
```

iii. Step 6/Step 7: "Initial implementation uses a per-unit/per-trial `np.histogram`; sample timing will determine whether vectorized session-wide binning is required" → "Vectorized all-trial binning with one `searchsorted` per unit — Sample reduced from 27.4 s to 2.8 s (~9.8x)." Step 10: "Runtime bottleneck: per-unit/per-trial histograms projected >15 minutes. Replaced with one vectorized `searchsorted` operation per unit, yielding a final full runtime of 209 s." The AI also added obs-mapping caching after an interrupted slow dry scan, and reports per-session timing to stdout for bottleneck identification. Runtime is comfortably inside the instructions' 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
1. **`for k,(g,to) in enumerate(zip(go,tone))`** — the per-trial loop that builds inputs and outputs. All of its contents are trivially broadcastable across the trial axis: `inp[0]` is `CENTERS_REL[None,:] + (go-tone)[:,None]`; the photostim `searchsorted` could be done once on a flattened `(n_trials*80,)` centre array; the static outputs are a broadcast of three per-trial vectors; and the tongue `nearest_indices` call could likewise run once over all centres. This is the most vectorizable remaining loop and it re-enters `np.searchsorted`/`nearest_indices` ~90,000 times over the dataset.
2. **`for j in units`** in `select_units_and_trials` — necessarily per-unit because of ragged HDF5 storage, but the intersection could at least short-circuit on identical `obs_intervals` *before* reading, and could operate on boolean masks instead of Python `set` operations.
3. **`for ui,sp in enumerate(spikes)`** — genuinely irreducible, since each unit has a differently sized sorted spike array; the trial dimension is already vectorised by the `(n_trials, 81)` edge matrix.

ii.
```python
        for k,(g,to) in enumerate(zip(go,tone)):
            fr=rates[k]
            centers=g+CENTERS_REL
            inp=np.empty((2,N_TIME),dtype=np.float32)
            inp[0]=(centers-to).astype(np.float32)
            if len(pstart):
                jj=np.searchsorted(pstart,centers,side='right')-1
                ...
            q=nearest_indices(vt,centers)
            ...
            neural.append(fr); inputs.append(inp); outputs.append(out); tongue_visible.append(vis)
```

iii. The AI's own efficiency note is limited to the binning loop it already fixed ("Reads each retained unit spike vector once per session; uses float32 for neural/input matrices and avoids full Units DataFrame materialization"). It did not flag the per-trial input/output loop, presumably because the measured 209 s runtime already met the 15-minute budget, so there was no forcing function to optimise further.

## 10-c. What processing does the code repeat multiple times?

i. Four repetitions, all minor:
1. **Per-unit HDF5 reads of `obs_intervals` / `is_good_trials`** in `select_units_and_trials`. The `map_cache` keyed on `(obs.shape, obs.tobytes())` avoids recomputing the *mapping*, but the read and the `tobytes()` hash still happen for every one of the hundreds of good units, and in most sessions all units share one identical interval array.
2. **Per-trial `np.searchsorted(pstart, centers, ...)`** and **`nearest_indices(vt, centers)`** — the same global `pstart`/`vt` arrays are searched once per trial instead of once per session.
3. **A second full traversal in `validate()`**, which re-walks every trial of every session checking shapes and finiteness after the arrays were already built.
4. **Firing rates are computed for trials that are then discarded** by the `neural_valid` filter (2,423 of 92,801 trials, ~2.6% of the binning work).

Nothing is recomputed across sessions: each file is opened exactly once, the tongue array and percentiles are computed once per session, and the bin grid is built once at module scope.

ii.
```python
        key=(obs.shape,obs.tobytes())
        mapped=map_cache.get(key)
        if mapped is None:
            ...
            map_cache[key]=mapped
```
```python
            if len(pstart):
                jj=np.searchsorted(pstart,centers,side='right')-1
            ...
                q=nearest_indices(vt,centers)
```
```python
def validate(data):
    ...
    for s in range(ns):
        for n,i,o in zip(data['neural'][s],data['input'][s],data['output'][s]):
```

iii. The AI documented the caching decision explicitly in Step 13: "Observation mappings are cached by unique interval arrays for efficiency." `validate()` is a deliberate correctness cost, not an oversight — Step 6 lists "assertions" as part of the design, and the instructions require validating shapes and types at each step. The rate-then-drop ordering is a consequence of the all-zero criterion being defined on the binned rates themselves.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount, none of it on the critical path:
1. **`tongue_visible`** is built for every trial of every session but is only ever consumed inside the `make_plot` branch, so it is pure waste on the 171 non-plotted sessions.
2. **`candidates=units`** is a redundant alias kept only so `info` can report `classifier_good_units` alongside `retained_units` — the two are always equal under the final policy (the AI's earlier plan to drop drift-affected units was abandoned).
3. **Firing rates for the 2,423 trials later dropped** as recording gaps are computed and then thrown away.
4. **Output arrays are `int64`** when the values are all in 0–3; `int8` would be 8× smaller (≈230 MB → ≈29 MB). Neural `float32` dominates the 12.0 GB pickle so this barely matters, but it is unused precision.
5. **`metadata['bin_centers_seconds']`**, several `session_info` fields (`raw_trials`, `structurally_valid_trials`, `conversion_seconds`, …) and the `plot_payload` tuple are stored but not consumed by `train_decoder.py`.

ii.
```python
        neural=[]; inputs=[]; outputs=[]
        tongue_visible=[]
        ...
            neural.append(fr); inputs.append(inp); outputs.append(out); tongue_visible.append(vis)
        ...
        if make_plot:
            plot_payload=(go,tone,pstart,pstop,vt,vy,vl,p40,p60,neural,inputs,outputs,np.asarray(tongue_visible),info)
```
```python
        candidates=units
```
```python
            out=np.empty((4,N_TIME),dtype=np.int64)
```

iii. The AI never flagged these as waste; items 2–5 are deliberate auditability/diagnostic choices consistent with Step 5's "Enables auditability" note for `session_info` and with the `--show-processing` requirement. Item 1 is a genuine (if negligible) oversight. The one memory decision the AI did document is the opposite trade — "uses float32 for neural/input matrices" — which is where the file size actually lives.
