# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads **only the randomized-delay subset** of the dataset. It hard-codes a list of 19 session names (`RAND_SESSIONS`) transcribed from the authors' `loadJEB11/12/23/24_ALMVideo.m` loader scripts, and reads every file from a single folder, `data/RandomizedDelay_Ephys_Behavior/`. The 25 fixed-delay DR/two-context sessions in `data/Ephys_Behavior/` (10 further mice) are never opened, and no other data directory is used. For each session it opens two files: `data_structure_<anm>_<date>.mat` and `motionEnergy_<anm>_<date>.mat`. Both MATLAB formats are supported: `data_structure` is first tried with `h5py` (v7.3/HDF5) and falls back to `scipy.io.loadmat` on `OSError` (v5); `motionEnergy` is always read with `scipy.io.loadmat`. Only a fixed set of fields is pulled out of each file (`bp.L/R/autowater/bitRand/early/hit/miss/no/Ntrials`, `bp.ev.bitStart/sample/delay/goCue/reward`, `clu.trialtm/trial`, `traj[0]` and `traj[1]`); `obj.sglx`, `obj.meta`, `obj.trials` and `clu.quality` are never read. Result: 19 sessions, 4 subjects, 5,805 trials, 3,089 units. The paths are relative (`Path('data/...')`), so the script only runs from `/app`.

ii.
```python
RAND_SESSIONS = [
    'JEB11_2022-05-10','JEB11_2022-05-11',
    'JEB12_2022-05-12','JEB12_2022-05-13',
    'JEB23_2023-10-10', ... ,'JEB24_2023-11-03'
]
...
base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
```

```python
try:
    with h5py.File(data_path, 'r') as f:
        bp = extract_bp_h5(f); trialtm, trial, n_units = extract_clu_h5(f)
        names0, get0 = get_traj_stream_h5(f, 0); names1, get1 = get_traj_stream_h5(f, 1)
        ...
except OSError:
    obj = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)['obj']
    bp = extract_bp_old(obj); trialtm, trial, n_units = extract_clu_old(obj)
    ...
```

iii. From CONVERSION_NOTES.md Step 4/5: the AI noticed that the raw `RandomizedDelay_Ephys_Behavior` folder holds 22 sessions while the paper reports 19, and resolved this by following the authors' loader scripts ("Figure 3 randomized-delay scripts explicitly build `randmeta` from `loadJEB11_ALMVideo`, `loadJEB12_ALMVideo`, `loadJEB23_ALMVideo`, and `loadJEB24_ALMVideo`, confirming the randomized-delay subset uses these four mice"). It gives **no justification for excluding the 25 fixed-delay sessions**; in Step 4 it wrote that the conversion "should likely combine the paper-relevant ephys+behavior session subsets", and in trajectory step 53 it recorded as an open defect that "the script still only handles `RandomizedDelay_Ephys_Behavior`, uses interim session filtering, assumes ALM for all units, and has not yet implemented the full paper-consistent dataset selection". That defect was never fixed; Steps 10 and 12 (the two critical-review steps) are marked NOT STARTED.

## 1-b. How are the data split into subjects?

i. The subject is the part of the session key before the first underscore. Subjects are accumulated in first-encounter order into a list, and `subject_idx` is that list's index for each session. Because only the randomized-delay folder is loaded, this yields 4 subjects (JEB11, JEB12, JEB23, JEB24) with 2/2/7/8 sessions. The animal id inside the file (`obj.meta.anm`) is never consulted.

ii.
```python
subject_names, subject_idx = [], []
for key in keys:
    subj = key.split('_')[0]
    if subj not in subject_names:
        subject_names.append(subj)
    ...
    subject_idx.append(subject_names.index(subj))
...
'subjects': subject_names,
'subject_idx': np.asarray(subject_idx, dtype=int),
```

iii. Not explicitly justified. CONVERSION_NOTES.md Step 2 notes that the loader scripts "enumerate animals/dates and build `meta` entries with `anm`, `date`, `datafn`, `probe`, and `datapth`", i.e. the animal is identified by the filename in the reference code too.

## 1-c. How are the data split into sessions?

i. One session = one entry of `RAND_SESSIONS` = one `data_structure_*.mat` / `motionEnergy_*.mat` pair in one folder. `process_session` returns one list of trials for `neural`, `input`, and `output`, appended in list order, so session order in the output is the order of `RAND_SESSIONS`. There is no session-level quality filter (e.g. the paper's ">= 10 units" rule is not applied, although all 19 sessions pass it anyway), and no handling of two-probe sessions — only `obj.clu{1}` is ever dereferenced (`f['obj/clu'][()].flat[0]`), so if a session has two probes the second is silently discarded.

ii.
```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])   # probe 1 only
    ...
```
```python
all_neural.append(neural); all_input.append(inp); all_output.append(out)
brain_region_idx.append(bri); subject_idx.append(subject_names.index(subj))
```

iii. Same as 1-a: the session list is taken from the authors' `load<ANM>_ALMVideo.m` scripts, which the AI documented as giving "19 sessions total (JEB11:2, JEB12:2, JEB23:7 active with one commented-out session, JEB24:8), matching the paper". The restriction to this one folder is not justified.

## 1-d. How are the data split into trials?

i. The trial count is `bp.Ntrials`, and every per-trial stream is indexed by the same 0-based trial index `ti` in `range(Ntrials)`: `bp.ev.goCue[ti]`, the `ti`-th cell of each `traj` stream, and `motion_data[ti]`. Spikes carry their own 1-based trial number in `clu.trial`, which is converted with `per_trial[trial_idx - 1]`. No trial boundaries are reconstructed. There is no check that `Ntrials` equals the length of the trajectory cell arrays; only the motion-energy array is length-guarded (`if ti < len(motion_data)`).

ii.
```python
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    per_trial = [np.zeros((n_units, n_bins), dtype=np.float32) for _ in range(n_trials)]
    for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
        for trial_idx in range(1, n_trials + 1):
            mask = tr == trial_idx
            if np.any(mask):
                counts, _ = np.histogram(ttm[mask], bins=time_edges)
                per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```
```python
n_trials = bp['Ntrials']
for ti in range(n_trials):
    go = bp['goCue'][ti]
    ...
    if ti < len(motion_data):
        me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 2/5: "Representative session `JEB11_2022-05-10` has 365 trials inferred from `obj.bp.ev` event arrays"; "for each unit, `clu.tm`, `clu.trial`, and `clu.trialtm` are per-spike arrays of equal length ... where `trial` likely indicates trial index and `trialtm` likely gives time within trial. This is enough to reconstruct per-trial spike trains." The planned sanity check "Check that `obj.bp.Ntrials` / event-array lengths match converted trial counts per session" was never executed (Step 10 NOT STARTED).

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both applied after everything has been computed. (1) `bp.early == 0 & bp.no == 0` — early-lick trials **and ignore/no-response trials** are dropped. (2) Trials whose entire neural matrix is exactly zero are dropped. Photostimulation trials (`bp.stim.enable`) are **not** filtered — the field is never read (in practice all 19 randomized-delay sessions have zero stim trials, so this has no effect on the delivered data, but it would matter for the fixed-delay sessions the AI omitted). 5,805 of 6,712 raw trials (86.5%) survive.

ii.
```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
...
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
neural = [neural[i] for i in keep_idx]
input_trials = [input_trials[i] for i in keep_idx]
output_trials = [output_trials[i] for i in keep_idx]
```

iii. Step 3/5 of CONVERSION_NOTES.md: "Early lick and ignore trials are omitted from analyses" and "**Exclude early and ignore/no-go trials from primary analyses**: Methods explicitly omit early lick and ignore trials; outcome should be defined on the remaining valid trials." The all-zero filter was added reactively: the verifier emitted "warnings about all-zero neural data in a block of late trials for session 11 and session 18" and the AI's response (trajectory step 227/249) was to "patch `convert_data.py`" to drop them so the warnings disappear, rather than to diagnose why blocks of trials had no spikes at all.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{1}.trialtm` (per-spike time within its trial) and `obj.clu{1}.trial` (1-based trial number of each spike), for every cluster in probe 1. `obj.bp.ev.goCue` is loaded into `bp['goCue']` but is **never used** for the neural stream. `clu.quality`, `clu.site`, `clu.tm` and `clu.spkWavs` are not read.

ii.
```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial   = [np.asarray(deref(f, r)[()]).ravel().astype(int)        for r in clu['trial'][()].flat]
    n_units = len(trialtm)
    return trialtm, trial, n_units
```

iii. Trajectory step 43: "for each unit, `clu.tm`, `clu.trial`, and `clu.trialtm` are per-spike arrays of equal length (e.g. 8285 spikes for unit 1) ... This is enough to reconstruct per-trial spike trains **without needing further upstream alignment**." That last clause is the origin of the alignment bug in 2-d. `clu.quality` was dropped from the loader after a session was found that lacks the `site` field (trajectory step 69: "patch `convert_data.py` so `extract_clu_trialtm` only requires `trialtm` and `trial`, while treating `site` and `quality` as optional") — but the optional handling was implemented as simply not reading them.

## 2-b. How is the `neural` data processed?

i. Raw **spike counts per 75 ms bin**, stored as float32. There is no conversion to Hz, no smoothing, no baseline subtraction, no normalisation, and no z-scoring. Units from a single probe form the population.

ii.
```python
counts, _ = np.histogram(ttm[mask], bins=time_edges)
per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. No justification is given in CONVERSION_NOTES.md for the absence of smoothing or rate conversion; the notes only record that "`rez.binSize = 75` ms" was found in the reference decoding scripts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron filtering at all.** Every cluster present in `obj.clu{1}` enters the dataset regardless of its manual-curation label (`clu.quality` is never read) and regardless of its firing rate. The result is 3,089 units over 19 sessions (mean 163/session, range 17–560); the paper reports 845 units for the randomized-delay dataset and the human reference keeps 1,954 units over 44 sessions (15–110 per session). Sessions with 543/449/437/465/560 "units" are clearly unfiltered sorter output.

ii. There is no filtering code. The entire cluster list is used:
```python
trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
...
brain_region_idx = np.zeros((n_units,), dtype=int)
```

iii. The AI **documented the correct rules but never implemented them**. CONVERSION_NOTES.md Step 3 "Neuron curation rules" states: "Spike sorting via JRCLUST and/or Kilosort 3 with manual curation in Phy 2 ... Include sessions only if at least 10 units. Include units with firing rates > 1 Hz in most analyses; only well-isolated single units > 1 Hz for subspace alignment and single-unit selectivity analyses." No reason is given for dropping the rules from the script, and the Step 9 consistency table comparing "Total neurons" against the paper was left entirely blank.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **It is not.** The AI's documented decision is go-cue alignment ("Use go cue alignment", Step 5 Key Decision 1; `temporal_alignment_event: 'go cue onset'` in the metadata), but the code histograms the raw `clu.trialtm` — time from **trial start** — into edges spanning −2.5 to +2.5 s, without ever subtracting `bp.ev.goCue`. `bp['goCue']` is subtracted only from the camera streams (`tmid - go`), so neural and behaviour are on two different clocks.

The consequences are directly visible in the delivered data:
- `trialtm` ranges from ≈ −0.5 s to ≈ +13 s, so only 34% of spikes fall in the retained window, and **bins 0–25 of 67 (−2.5 s to −0.55 s) are exactly zero for every neuron in every trial** — verified on `converted_data.pkl` session 0, mean count per bin is 0.000 for the first 26 bins.
- The go cue sits at a median of 2.2 s after trial start in `JEB11_2022-05-10`, and because this is the **randomized**-delay task, it varies from 1.9 s to 7.37 s **across trials of the same session**. So the effective alignment offset is not just wrong by a constant, it is jittered trial-by-trial by several seconds.
- The trials the AI dropped as "all-zero neural" (1-e) are simply trials where the animal took longer than ~2.5 s from trial start, i.e. a symptom of this bug.

ii.
```python
counts, _ = np.histogram(ttm[mask], bins=time_edges)   # ttm is time from TRIAL START; goCue never subtracted
```
Compare with the camera path in the same script, which does subtract the go cue:
```python
go = bp['goCue'][ti]
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The AI repeatedly and correctly identified go cue as the alignment event (Step 1: "`aligntimes = mode(obj(sessix).bp.ev.goCue) - 2.5` ... strongly indicates go-cue-centered timing"; Step 4 resolution: "Use go cue onset as temporal alignment event"). The implementation gap traces to trajectory step 43 — "This is enough to reconstruct per-trial spike trains without needing further upstream alignment" — and was never caught because Step 10 Check 2 ("sanity checks ... using `np.allclose()`" against raw files) and Step 12 ("Check temporal alignment — plot neural + output for a single trial") were never performed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 75 ms bins, 67 bins spanning −2.5 s to +2.5 s (the last edge lands at 2.525 s because `np.arange(-2.5, 2.5+0.075, 0.075)` produces 68 edges). Spikes and camera samples are binned **directly** at 75 ms; there is no intermediate 5 ms binning and therefore no rebinning step. One grid is used for every trial, session and stream, so all streams are nominally on one time axis (though the neural stream's axis means something different — see 2-d). `metadata['time_bin_size']` is written as `0.075`, i.e. in **seconds**, while the target format specifies "length of time bin in ms" (the human reference writes `5.0`).

ii.
```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
...
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
...
'time_bin_size': BIN_SIZE,
```

iii. CONVERSION_NOTES.md Step 1: "Decoding scripts use `rez.binSize = 75` ms and derive sample-step count with `rez.dt = floor(rez.binSize / (params(1).dt*1000))`, indicating a 75 ms analysis bin"; Step 5 Key Decision 4: "**Use 75 ms bins as initial default**: Explicitly present in decoding scripts; verify against neural loading/alignment code during implementation." The verification against the loading code (`params.dt = 1/200`) was never done.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. The input is the analysis grid the AI defined: the centres of the 67 bins spanning −2.5 to 2.5 s. The same 1×67 vector is copied into every trial of every session, so it carries no trial-specific information.

ii.
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2

def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

iii. Step 5 mapping table: "Time from go cue | input[0] | Continuous per-timepoint variable relative to go cue onset | `obj.bp.ev.goCue`, coding-direction plotting/alignment functions | Required decoder input". The mapping table names `bp.ev.goCue` as the source, but the implementation needs no raw variable because the grid is defined by construction — which is correct *provided* every other stream really is expressed relative to the go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond taking bin centres, casting to float32, and replicating the vector once per trial (`.copy()` per trial, so 5,805 identical 67-element arrays are materialised). Range is [−2.5, 2.5] as reported by the verifier.

ii.
```python
return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

iii. Not separately justified.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction the input vector is the centre of the same `time_edges` grid that the spikes are binned into, so bin *k* of `input` and bin *k* of `neural` are the same index. But the **label is wrong**: because the spikes were never shifted by the go cue (2-d), bin *k* of `neural` is time *k* from **trial start**, while the input asserts it is time *k* from the go cue. The two are therefore off by the trial's go-cue latency (median ≈ 2.2 s, varying 1.9–7.4 s within a session). The camera-derived outputs *are* shifted by the go cue, so the input is correctly labelled relative to the outputs but not relative to the neural data.

ii.
```python
counts, _ = np.histogram(ttm[mask], bins=time_edges)          # neural: trial-start clock
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)      # outputs: go-cue clock
time_centers = (time_edges[:-1] + time_edges[1:]) / 2         # input: labelled as go-cue clock
```

iii. No justification; the mismatch is undocumented and unnoticed.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `obj.bp.R` and `obj.bp.L` only — the **instructed** lick port for each trial. The outcome flags `hit`/`miss`, which are what actually determine which port the animal licked, are not used for this output.

ii.
```python
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
...
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. Step 5 mapping table: "`obj.bp.L`, `obj.bp.R` | output[0] lick direction | Map left=0, right=1 | Figure/task condition code, `NeuralChoiceDecoding.m` | Per-trial". The notes do not distinguish instructed side from actual licked side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct relabel of the instructed side: right → 1, otherwise → 0. The value is broadcast across all 67 bins to make it time-varying. Only **two** classes exist; the required third class, "none", is absent because all no-response trials were removed by the `bp['no'] == 0` filter (1-e). Because the value is the instructed rather than the licked port, the label is **wrong on every miss trial** — the animal licked the opposite port — which is 11.2% of the retained trials (the reported `outcome` incorrect fraction). The delivered distribution is {left 0.494, right 0.506}; the human reference gets {left 0.423, right 0.446, none 0.131}.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
...
output_trials.append(np.vstack([
    np.full((len(time_centers),), lick_dir[ti], dtype=np.int64),
    ...
]))
...
'output_values': [['left', 'right'], ...]
```

iii. No justification for using instructed side as lick direction, and none for the missing "none" class. Step 5 Key Decision 2 ("Exclude early and ignore/no-go trials") is the stated reason ignore trials are gone, but the Decoder Task specification explicitly lists "Lick direction (left, right, **none**, per-trial)".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`.

ii.
```python
for k in ['L', 'R', 'autowater', ...]:
    out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
```

iii. Step 4 discrepancy table: "Figure scripts use `autowater` for WC and `~autowater` for DR ... Map WC=autowater, DR=not autowater unless contradicted by deeper inspection." This matches the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabel: autowater → WC (0), otherwise → DR (1), broadcast across all 67 bins. The mapping matches the human reference exactly. The *distribution* however is badly skewed — 1.3% WC overall, and **9 of the 19 sessions contain zero WC trials** (verifier: `1: [1.0, 1.0]` for nine sessions) — because the randomized-delay sessions the AI kept are largely single-context, while the two-context sessions live in the `Ephys_Behavior` folder it never opened. The human reference gets 9.7% WC.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
...
np.full((len(time_centers),), context[ti], dtype=np.int64),
...
'output_values': [..., ['WC', 'DR'], ...]
```

iii. As 5-a. The AI did not flag the near-degenerate context distribution, and the Step 9 consistency table for output distributions was left blank.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` only. `bp.miss` is loaded but not used for this output; `bp.no` is used as a trial filter rather than as a class.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. Step 5 mapping table: "`obj.bp.hit`, `obj.bp.miss`/`obj.bp.no` | output[2] outcome | Map incorrect=0, correct=1; likely hit=1 and miss/no=0 **after trial filtering**".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary relabel: hit → correct (1), everything else → incorrect (0), broadcast across the 67 bins. Since ignore trials were already removed, "everything else" is exactly the miss trials, so the two classes are meaningful — but the required third class, "ignore", is absent. Delivered: {incorrect 0.112, correct 0.888}; human reference: {incorrect 0.120, correct 0.749, ignore 0.131}.

ii.
```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
...
'output_values': [..., ['incorrect', 'correct'], ...]
```

iii. Step 3 / Step 5 Key Decision 2: the paper "omits early lick and ignore trials from analyses", so the AI dropped them. It did not reconcile this with the Decoder Task spec, which lists "Outcome (incorrect, correct, **ignore**, per-trial)".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. `obj.traj` DeepLabCut tracking. `choose_feature` picks a tongue channel from the **bottom** camera (`traj[1]`) first, trying `top_tongue`, `bottom_tongue`, `topleft_tongue`, `bottomleft_tongue`, and falls back to the **side** camera (`traj[0]`: `tongue`, `left_tongue`, `right_tongue`) only if the bottom one yields no finite bins. In practice `top_tongue` is always found, so only the bottom view is ever used and the two views are never combined. Within `ts`, only channels 0 and 1 (x, y) are read; channel 2 (**DeepLabCut likelihood**) is loaded but never used. The v7.3 layout is `(feature, xyl, frame)`; the v5 layout `(frame, xyl, feature)` is transposed to match.

ii.
```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
...
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    if idx is None: continue
    ts, ft = getter(ti)
    spd, tmid = speed_from_ts(ts, ft)
    candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
    if np.any(np.isfinite(candidate)):
        tongue_bin = candidate
        break
```
```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)     # likelihood channel ts[:,2,:] discarded
```

iii. Trajectory step 49: "`obj.traj` has two streams: stream 0 contains tongue/jaw/nose/lickport features, and stream 1 contains tongue and paw features including `top_paw` and `bottom_paw`. Since `ts` is (n_features, 3, n_timepoints), we can use x/y coordinates (first two channels) to compute velocities". The third channel was assumed to be "likelihood/confidence" (step 48) and then simply ignored; no rationale is offered for preferring one camera over the other or for not using the likelihood.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Nominally: first differences of (x, y) across consecutive frames, divided by the frame interval, then averaged into 75 ms bins. In practice the result is **degenerate and does not measure tongue velocity at all**, because of `np.nansum`:

- `top_tongue` x/y are NaN on 98.2% of frames (the authors NaN out low-likelihood frames). `dxy` is therefore NaN for essentially every frame pair.
- `np.nansum(dxy ** 2, axis=1)` treats NaN as **0**, so the speed of an untracked frame is computed as `0 / dt = 0.0` — a finite value, not NaN.
- Consequently the session-wide median (7-c) is **exactly 0.0 in all 19 sessions** (confirmed in `conversion_full_out.txt`: `'tongue_threshold': 0.0` for every session), and every bin containing at least one camera frame is classified `>= 0.0` → class 1.

The delivered `tongue_velocity` is therefore a **binary indicator of whether any camera frame falls in that bin**, not a velocity: in session 0 trial 0 the row is 14 zeros followed by 53 ones, and the transition at bin 14 (t = −1.45 s) is exactly where the camera record begins (frames start at 0.515 s on the video clock, go cue at 1.9 s). Session-wide it is {low 0.140, high 0.860} in every session — the 14% "low" are the pre-camera bins. The human reference gets {<50th 0.062, >=50th 0.062, not visible 0.875}.

The steps performed by the reference and omitted here: likelihood thresholding at 0.9, Gaussian smoothing of x/y within contiguous tracked runs, per-camera normalisation by the 90th percentile, and averaging the two views.

ii.
```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    dt = np.diff(frame_times).astype(np.float32)
    dt[dt <= 0] = np.nan
    dxy = np.diff(xy, axis=2)
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]   # NaN coords -> speed 0.0, not NaN
    tmid = (frame_times[:-1] + frame_times[1:]) / 2
    return spd, tmid
```
```python
def nanbin_mean(times, values, edges):
    out = np.full((len(edges) - 1,), np.nan, dtype=np.float32)
    idx = np.digitize(times, edges) - 1
    for bi in range(len(out)):
        m = idx == bi
        if np.any(m):
            vv = values[m]
            if np.any(np.isfinite(vv)):
                out[bi] = np.nanmean(vv)
    return out
```

iii. The AI's only recorded reasoning is that the outputs became "non-degenerate": Step 6 notes "Sample output distributions are non-degenerate for all six outputs after implementing tongue/paw/motion-energy binning", and trajectory step 53 "all six outputs taking both values 0 and 1. Tongue, paw, and motion-energy outputs are no longer degenerate." Having two distinct values was treated as sufficient evidence of correctness; no check was made that the values reflect tongue motion, and the constant `tongue_threshold: 0.0` printed for all 19 sessions was not investigated.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, across all trials and bins pooled: the threshold is `np.nanmedian` of all finite binned values, and bins are labelled `1` if `>= threshold`, `0` otherwise. Bins that are NaN (no frames at all) are also assigned **`0`** — they are silently merged into the "low" class rather than given the third, "not visible" category the specification requires. So the output has 2 categories instead of 3. Because of the 7-b bug the threshold is 0.0, so the split is vacuous.

ii.
```python
def threshold_session_bins(arr_list):
    finite_chunks = [a[np.isfinite(a)] for a in arr_list if np.any(np.isfinite(a))]
    allv = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    thr = np.nanmedian(allv) if allv.size else np.nan
    out = []
    for a in arr_list:
        b = np.zeros_like(a, dtype=np.int64)          # NaN bins default to 0 = "low"
        if np.isfinite(thr):
            valid = np.isfinite(a)
            b[valid] = (a[valid] >= thr).astype(np.int64)
        out.append(b)
    return out, thr
```

iii. Step 5 mapping table: "Compute velocity, then discretize per session at 50th percentile"; "Session-level thresholding required by task". The median-per-session rule matches the spec; the missing third class is undocumented.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are taken from `traj[stream].frameTimes`, the midpoint of each consecutive pair is formed, the trial's `bp.ev.goCue` is subtracted, and the result is binned into the shared `time_edges`. **The video-to-behaviour clock offset is never applied.** The reference derives it from the bitcode (`sglx.bitcode.bitstart / sglx.fs` minus `bp.ev.bitStart`, per the authors' `findVideoOffset.m`); measured on `JEB11_2022-05-10` it is **0.490 s** — about 6.5 bins of 75 ms. `bp.ev.bitStart` is loaded by `extract_bp_*` but never used, and `obj.sglx` is never opened. So the camera streams are shifted ~0.5 s late relative to the go cue. Separately, they are on a different clock from the neural data altogether (2-d).

ii.
```python
for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
    out[k] = np.asarray(ev[k][()]).astype(np.float32).ravel()   # bitStart loaded, never used
...
go = bp['goCue'][ti]
...
spd, tmid = speed_from_ts(ts, ft)
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)        # no video offset
```

iii. No justification; the AI never identified `findVideoOffset.m` or the bitcode fields. CONVERSION_NOTES.md Step 5 planned sanity check "Check that go-cue-aligned time axis reproduces expected sample/delay offsets from reference code" — which would have exposed this — was never run.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj[1]` (bottom camera), feature `top_paw`, with `bottom_paw` as a fallback if `top_paw` is absent. x and y only; likelihood discarded. This matches the human reference's choice of `top_paw` from the bottom view.

ii.
```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
...
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
    spd, tmid = speed_from_ts(ts, ft)
    paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Trajectory step 49 identified `top_paw` and `bottom_paw` in stream 1 and step 48/49 planned to "select ... a paw feature". No reason is given for preferring `top_paw`; the preference order in `choose_feature` is simply the order the names appear in the file.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same `speed_from_ts` as the tongue: frame-to-frame Euclidean displacement over the frame interval, averaged into 75 ms bins. No likelihood cut, no Gaussian smoothing, no normalisation. Here the pipeline mostly works, because `top_paw` is tracked on ~96.5% of frames, so the thresholds are real physical values (72–175 px/s across sessions). The residual defect is the same `np.nansum` behaviour: the 3.5% untracked frames are assigned speed 0.0 instead of NaN, which biases the distribution slightly downward and, more importantly, removes any possibility of a "not visible" class. Delivered: {low 0.573, high 0.427} — the asymmetry is caused by ties at 0.0 all falling on the `>=` side or below it. Human reference: {<50th 0.407, >=50th 0.407, not visible 0.186}.

ii.
```python
spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
...
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Same as 7-b: the only recorded validation is that the distribution is "non-degenerate".

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identical to 7-c: `threshold_session_bins` takes the session-wide median over all finite binned values and labels `>= median` as 1, `< median` as 0, with NaN bins folded into class 0. Two classes, not the three required. Thresholds are per session and plausible (e.g. 160.8 px/s for `JEB11_2022-05-10`, 72.8 for `JEB12_2022-05-13`).

ii.
```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
...
'paw_threshold': float(paw_thr) if np.isfinite(paw_thr) else None,
```

iii. As 7-c: "discretize per session at 50th percentile".

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: `tmid - goCue[trial]` binned into the shared grid, with **no video-clock offset** (≈0.49 s error) and on a different clock from the neural data. The paw uses the bottom camera's own `frameTimes`, so the paw and the tongue are at least mutually consistent.

ii.
```python
ts, ft = get_stream1(ti)
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Same as 7-d — no justification, and the offset was never discovered.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file, which holds one trace per trial with one value per camera frame. `obj.me` is not consulted. `load_motion_energy` unwraps the several different MATLAB layouts encountered (`me`, `me.data`, `me.data.data`, object arrays of `mat_struct`) via `normalize_motion_elem`. `me.moveThresh` is read and stored in `session_info` but never used. Frame times come from `traj[0]` (the side camera).

ii.
```python
def load_motion_energy(path):
    me = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['me']
    move_thresh = getattr(me, 'moveThresh', None) if hasattr(me, '_fieldnames') else None
    raw = me
    while hasattr(raw, '_fieldnames') and hasattr(raw, 'data'):
        raw = raw.data
    arr = np.asarray(raw, dtype=object)
    out = [normalize_motion_elem(x) for x in arr.flat]
    return out, move_thresh

def normalize_motion_elem(x):
    while True:
        if hasattr(x, '_fieldnames'):
            if hasattr(x, 'data'): x = x.data; continue
            x = getattr(x, x._fieldnames[0]); continue
        if isinstance(x, np.ndarray) and x.dtype == object:
            if x.size == 0: return np.array([], dtype=np.float32)
            x = x.flat[0]; continue
        return np.asarray(x).ravel().astype(np.float32)
```

iii. Step 2: "Representative `motionEnergy_*.mat` file contains struct `me` with fields `data` and `moveThresh`", "`me.data` length 365 and session-level `moveThresh` = 9". Step 6 notes: "Full conversion exposed another motion-energy edge case: some later `motionEnergy_*.mat` files contain `mat_struct` elements inside `me.data` rather than direct numeric arrays, so the loader must normalize multiple internal representations." The unwrapping loop matches the reference's `loadMotionEnergy.m` guard in spirit.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond binning — the value is already one scalar per frame, so it is averaged into the 75 ms bins by `nanbin_mean`. This matches the human reference. There is one guard: if the motion-energy trace length does not equal the number of side-camera frames, the whole trial's motion energy is set to NaN (and hence, after thresholding, to class 0).

ii.
```python
if ti < len(motion_data):
    me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
    _, ft0 = get_stream0(ti)
    me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) \
             else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
else:
    me_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
```

iii. Step 5 mapping table: "`me.data` | output[5] motion energy | Use provided motion-energy time series, discretize per session at 50th percentile".

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `threshold_session_bins`: session-wide median over all finite bins, `>=` → 1, `<` → 0, NaN → 0. Two classes; the required "no video" third class is absent. Thresholds are plausible and vary sensibly across sessions (22.3 to 54.0). Delivered: {low 0.535, high 0.465}; human reference: {<50th 0.479, >=50th 0.483, no video 0.038}.

ii.
```python
me_bin, me_thr = threshold_session_bins(me_vals)
...
'motion_threshold': float(me_thr) if np.isfinite(me_thr) else None,
```

iii. As above; the per-session 50th percentile is taken directly from the Decoder Task specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per side-camera frame, so it is timed by `traj[0].frameTimes`, minus the trial's go cue, and binned into the shared grid. Note this uses the raw frame times (not the midpoints used for the velocities), so motion energy is offset by half a frame (~2 ms) relative to the tongue/paw — negligible at 75 ms. The substantive problems are the same two as 7-d/8-d: **no video-clock offset** (≈0.49 s) and a different clock from the neural data.

ii.
```python
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else ...
```

iii. No justification recorded.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled, all by substituting a default rather than by marking the gap:

- **Mixed MATLAB versions**: `data_structure` is tried as HDF5 and falls back to `scipy.io.loadmat`; the v5 `ts` axis order `(frame, xyl, feature)` is transposed to `(feature, xyl, frame)`.
- **Varying motion-energy wrappers**: `normalize_motion_elem` loops through `mat_struct`/object-array nesting; an empty element returns an empty float32 array.
- **Missing cluster fields** (`site`, `quality` absent in some sessions): handled by never reading them at all.
- **Trial/length mismatches**: `ti >= len(motion_data)` or `len(ft0) != len(me)` → the whole trial's motion energy becomes NaN.
- **Non-monotonic frame times**: `dt[dt <= 0] = np.nan` turns zero/negative intervals into NaN speeds.
- **Untracked DeepLabCut frames**: **not** handled — `np.nansum` silently converts NaN coordinates to a speed of exactly 0.0, which is the root cause of the degenerate tongue output (7-b).
- **NaN bins after binning**: all collapsed to class **0** ("low") by `threshold_session_bins`, so "no data" is indistinguishable from "slow". None of the three movement outputs has the "not visible" / "no video" class the specification requires.
- **Trials with no spikes in the window**: dropped entirely, which masked rather than revealed the alignment bug (2-d).

ii.
```python
dt[dt <= 0] = np.nan
spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
```
```python
b = np.zeros_like(a, dtype=np.int64)          # NaN -> class 0
if np.isfinite(thr):
    valid = np.isfinite(a)
    b[valid] = (a[valid] >= thr).astype(np.int64)
```
```python
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The format/robustness fixes are documented in CONVERSION_NOTES.md Step 6 ("not all `data_structure_*.mat` files are HDF5/v7.3 ... the converter must support both formats"; "some later `motionEnergy_*.mat` files contain `mat_struct` elements inside `me.data`"). The semantic choices — NaN → "low", NaN coordinates → zero speed, dropping all-zero trials — are not justified anywhere; they were introduced to make the verifier's warnings disappear (trajectory steps 227, 249).

## 11-a. What are the most time-consuming steps of the code?

i. The AI printed no timing information at all (the Step 7 "Run Time Estimates" table in CONVERSION_NOTES.md is empty, and `convert_data.py` has no timers), even though the instructions required it. Measured directly on `JEB11_2022-05-10` (63 units, 365 trials): `extract_binned_behavior_generic` **3.6 s**, `bin_unit_spikes_for_trials` **0.5 s**, cluster loading 0.23 s, motion-energy loading and `bp` extraction < 0.05 s. So the behaviour loop dominates for typical sessions; on the five large sessions (437–560 units) spike binning scales up to roughly 4–5 s and becomes comparable. Both bottlenecks are Python-level loops, not I/O — the opposite of the human reference, where file reading dominates. Total runtime is still only a few minutes.

ii. The two hot loops:
```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        mask = tr == trial_idx                      # O(n_units * n_trials) full-array scans
        if np.any(mask):
            counts, _ = np.histogram(ttm[mask], bins=time_edges)
```
```python
def nanbin_mean(times, values, edges):
    idx = np.digitize(times, edges) - 1
    for bi in range(len(out)):                      # 67 Python iterations, each a full boolean scan
        m = idx == bi
```

iii. Not discussed. The instructions asked to "Print timing information to find bottlenecks" and to estimate full-conversion time in Step 7; neither was done.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three clear cases, all left as Python loops:

1. **`bin_unit_spikes_for_trials`** — a nested `n_units × n_trials` loop, each iteration doing a full-length boolean comparison over that unit's spike vector and a `np.histogram` call. For a 560-unit, 400-trial session that is 224,000 iterations and 224,000 full scans. The whole thing collapses to a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, time_edges])` per unit — or one call for all units using a 3-D `histogramdd` / `np.add.at` on precomputed bin indices. This is exactly what the human reference does.
2. **`nanbin_mean`** — loops over the 67 output bins and rescans the full `idx` array each time. `np.bincount(idx, weights=values)` divided by `np.bincount(idx)` (with a finite mask) replaces the whole function with two vectorized calls. It is invoked 3–4 times per trial, so ~20,000 times per session.
3. **The per-trial loop in `extract_binned_behavior_generic`** — unavoidable in part (frame counts differ per trial), but the per-trial `speed_from_ts` could at least be hoisted (see 11-c).

ii.
```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        mask = tr == trial_idx
```
```python
for bi in range(len(out)):
    m = idx == bi
```

iii. Not discussed anywhere in CONVERSION_NOTES.md; the "Code speedups added" field is literally left as the template placeholder `[Note]`.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions:

1. **The bottom camera is read and fully processed twice per trial.** `get_stream1(ti)` is called once inside the tongue loop and again for the paw, and `speed_from_ts` — which computes the speed of **all 10 tracked features** — is run both times. One call would serve both.
2. **The side camera is read again** for motion energy (`get_stream0(ti)`), after possibly already having been read in the tongue fallback branch.
3. **In the v5 path, the trajectory cell array is rebuilt on every access**: `get_trial` executes `np.asarray(stream_trials, dtype=object).ravel()[trial_i]` each call, re-materialising the whole per-trial object array once per trial per stream.
4. **The time grid is rebuilt per session** inside `process_session` rather than once at module level, and the identical 67-element input vector is `.copy()`-ed 5,805 times.

ii.
```python
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    ts, ft = getter(ti)
    spd, tmid = speed_from_ts(ts, ft)            # all 10 features
    ...
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)                     # same data read and reprocessed
    spd, tmid = speed_from_ts(ts, ft)            # all 10 features again
```
```python
def get_trial(trial_i):
    tr = unwrap_obj(np.asarray(stream_trials, dtype=object).ravel()[trial_i])
```

iii. Not discussed.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five categories:

1. **Speeds for unused features.** `speed_from_ts` computes x/y differences and Euclidean speed for **every** tracked feature (10 on the bottom camera, 7 on the side camera), and then a single row is indexed: `spd[idx]`. With the double call in 11-c, roughly 19 feature-speed traces are computed per trial to keep 2.
2. **Unused loaded variables.** `bp.bitRand`, `bp.ev.sample`, `bp.ev.delay`, `bp.ev.reward`, `bp.miss`, `bp.L` (only used in the `R > L` comparison), and — most consequentially — `bp.ev.bitStart`, which is exactly the field needed for the video-clock offset that was never applied (7-d). `me.moveThresh` is loaded and written into `session_info` but never used.
3. **Empty bins.** Bins 0–25 of every trial's neural matrix (39% of the array) are structurally all-zero because of the alignment bug (2-d) and carry no information, yet they are computed, stored and shipped — roughly 107 MB of the 276 MB pickle.
4. **Dead code.** `decode_char`'s `dtype.kind` branch, and the `deref` helper which is a one-line alias for `f[x]`.
5. **`float32` counts.** Spike counts in a 75 ms bin fit in `uint8`/`int16`; storing them as float32 doubles-to-quadruples the largest object in the output (the human reference notes float32 for smoothed *rates*, where it is genuinely needed).

ii.
```python
xy = ts[:, :2, :].astype(np.float32)             # all features
spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
...
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)   # one feature kept
```
```python
for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
    out[k] = np.asarray(ev[k][()]).astype(np.float32).ravel()   # only goCue is used
```

iii. Not discussed. CONVERSION_NOTES.md Step 6's "Code inefficiencies identified" section lists only MATLAB-format edge cases, not actual inefficiencies.
