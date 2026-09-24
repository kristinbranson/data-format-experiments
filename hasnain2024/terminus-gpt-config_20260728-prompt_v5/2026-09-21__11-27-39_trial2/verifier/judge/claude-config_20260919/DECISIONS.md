# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Sessions are **discovered by globbing** the two ephys folders (`Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`) for `data_structure_*.mat`; the animal and date are parsed out of the filename with a regex, and a sibling `motionEnergy_<anm>_<date>.mat` is attached if it exists. The author's `load<ANM>_ALMVideo.m` session lists were never transcribed, so every file on disk in those two folders is taken, including the ones the authors commented out. The behaviour-only folders (`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) are excluded because the decoder needs neural data.

Each session is opened with **h5py only** (`build_session` does `with h5py.File(sess.data_file)`), reading `/obj/bp`, `/obj/clu`, `/obj/traj`, `/obj/meta`. Motion energy is read separately with `scipy.io.loadmat`. A scipy fallback for the v5 (non-HDF5) `data_structure` files was written (`build_session_scipy`, `scipy_obj_to_bp`, `scipy_read_units`, `scipy_align_tongue`) but it is **defined after `if __name__ == '__main__'` and is never called**. Sessions that fail to open are caught by a blanket `try/except` in `main()` and skipped with a printed message.

The result (from `conversion_full_out.txt`) is that 47 files were found, **13 sessions were skipped** (11 for `OSError: file signature not found`, i.e. v5 files, and 2 for a missing `clu` group), leaving **34 sessions / 13 subjects / 11,385 trials / 8,087 units** — against the reference's 44 sessions / 14 subjects / 13,762 trials / 1,954 units. Subject JEB24 is lost entirely.

ii.
```python
def discover_sessions() -> List[SessionInfo]:
    sessions = []
    for task_dir in EPHYS_DIRS:
        d = DATA_ROOT / task_dir
        for data_file in sorted(d.glob('data_structure_*.mat')):
            m = re.match(r'data_structure_([^_]+)_(.+)\.mat', data_file.name)
            ...
            sessions.append(SessionInfo(animal, date, task_dir, data_file, motion_file))
    return sessions
```
```python
def build_session(sess: SessionInfo):
    with h5py.File(sess.data_file, 'r') as h:
        bp = read_bp(h)
        n_trials = int(np.array(bp['Ntrials']).squeeze())
        units = read_clu_units(h)
```
```python
        try:
            neural_trials, input_trials, output_trials, subject, bri = build_session(sess)
        except Exception as e:
            print(f'SKIP {sess.data_file.name}: {e!r}')
            skipped.append((sess.data_file.name, repr(e)))
            continue
```

iii. From CONVERSION_NOTES.md Step 5 Key Decision 1: *"Use only ephys sessions for converted dataset: Decoder input must be neural activity, so behavior-only inhibition sessions without neural data should be excluded."* Step 2 notes correctly record that `data_structure_*.mat` are "v7.3/HDF5 ... should be read with h5py" and that `motionEnergy_*.mat` are "older MATLAB format ... scipy.io.loadmat" — the agent never revised this after discovering that 13 `data_structure` files are also v5. Its Step 9 note admits the gap: *"13 sessions were skipped due to mixed MAT-file formats or missing `clu` ... Additional work is needed to implement scipy fallback for non-HDF5 `data_structure` files."* The agent also found the authors' session-inclusion rule (Step 4 note: *"sessions are filtered to those with more than 40 right-hit and more than 40 left-hit trials"*) but never applied it.

## 1-b. How are the data split into subjects?

i. The subject is the part of the filename before the first underscore (`data_structure_JEB19_2023-04-19.mat` → `JEB19`), captured by the regex in `discover_sessions`. Subjects are appended to `data['subjects']` in first-encountered order as sessions are processed, and `subject_idx` records each session's index into that list. This is the same rule the reference uses (the reference sorts the subject list instead of using encounter order, which is immaterial). Because 13 sessions were dropped, 13 of the 14 subjects survive; JEB24 disappears.

ii.
```python
m = re.match(r'data_structure_([^_]+)_(.+)\.mat', data_file.name)
animal, date = m.groups()
```
```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(data['subjects'])
    data['subjects'].append(subject)
data['subject_idx'].append(subject_to_idx[subject])
```

iii. Not explicitly justified in CONVERSION_NOTES.md; the filename convention is documented in the code README the agent read (*"data_structure_Animal_SessionDate.mat"*), and the agent's Step 2 filename scan tabulated sessions per animal this way.

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file = one session = one element of `neural`, `input`, `output`, `brain_region_idx`, and one entry of `subject_idx`. Both task folders are treated uniformly and concatenated (25 fixed-delay files first, then the randomized-delay files), which matches the reference's treatment. The unit of a session is therefore correct; what differs is *which* sessions end up in the list: 34 rather than 44, including three files the authors excluded (e.g. `JEB23_2023-10-20`, `JEB24_2023-10-03/04`, which were attempted and skipped) and omitting every v5 session.

ii.
```python
EPHYS_DIRS = ['Ephys_Behavior', 'RandomizedDelay_Ephys_Behavior']
...
for sess in sessions:
    ...
    data['neural'].append(neural_trials)
    data['input'].append(input_trials)
    data['output'].append(output_trials)
    data['brain_region_idx'].append(bri)
```

iii. Step 4 note: *"directory counts show 25 DR ephys sessions and 22 randomized-delay ephys sessions in raw files, whereas methods report 19 randomized-delay sessions used in analysis. This strongly suggests additional session-level exclusion beyond raw-file presence"* — the agent identified the discrepancy and the criterion but never implemented an exclusion.

## 1-d. How are the data split into trials?

i. `n_trials = obj.bp.Ntrials` defines the trial count, and every per-trial field is truncated to `[:n_trials]`. Spikes carry a 1-based `clu.trial` index, so trial *t* is the set of spikes with `trial == t`; the code loops `for tr in range(1, n_trials + 1)`. Camera trajectories and motion energy are per-trial cell arrays indexed positionally (`ts_refs[tr]`, `data_arr.flat[tr]`), with a `min(n_trials, len(...))` guard. Every one of the `Ntrials` trials is emitted — no trial is ever dropped.

ii.
```python
n_trials = int(np.array(bp['Ntrials']).squeeze())
...
for ui, u in enumerate(units):
    trials = np.array(u['trial']).astype(int).ravel()
    trialtm = np.array(u['trialtm']).astype(float).ravel()
    # assume MATLAB 1-based trial indices
    for tr in range(1, n_trials + 1):
        mask = trials == tr
```
```python
hit = np.ravel(bp['hit'])[:n_trials] > 0
```

iii. Planned sanity check in Step 5: *"Verify `obj.bp.Ntrials` matches the number of converted trials for representative sessions."* (marked as planned, never checked off). The 1-based assumption is noted in a code comment.

## 1-e. How are trials filtered based on quality controls?

i. **No trial filtering at all.** Early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are read nowhere in the conversion path — `bp.early` appears only in the dead scipy helper's field list, and `bp.stim` is never touched. Trials that run past the end of the ephys recording are also kept; the verifier flags one of them (`Session 27, trial 192: all neural data is zero`). All 11,385 `Ntrials` trials of the 34 loaded sessions are emitted.

ii. There is no filtering code to quote. The full set of per-trial handling is:
```python
for tr in range(n_trials):
    input_trials.append(time[None, :].astype(np.float32))
    ...
    output_trials.append(out)
```

iii. No justification is given. CONVERSION_NOTES.md Step 3 "Trial curation rules" records only the analysis-specific subsampling rules (*"40 right-correct and 40 left-correct trials"*, *"40 DR and 40 WC trials"*) and never mentions the early-lick or photostim exclusions, even though `early` and `stim` were both listed among the `bp` fields the agent enumerated in Step 2.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu`, dereferenced as a cell array with **one entry per probe**, each probe group holding per-cluster cell arrays of `tm`, `trial`, `trialtm`, `quality`, `site` (or `channel`). Spike counts come from `clu.trial` (1-based trial index of each spike) and `clu.trialtm` (spike time relative to trial start); `clu.tm` (session-clock spike times) is read and used only for the firing-rate filter; `clu.quality` is read and never used. **All probes are concatenated** — for two-probe sessions the code takes both, whereas the reference takes only the probe the authors' loading script names (e.g. EKH1 uses probe 2 only; the AI emits all 32+48 = 80 clusters). `obj.bp.ev.goCue` is *not* used for the neural stream.

ii.
```python
def read_clu_units(h: h5py.File) -> List[Dict[str, Any]]:
    clu = h['/obj/clu']
    probe_groups = deref_cell(h, clu)
    units = []
    for probe_idx, grp in enumerate(probe_groups):
        ...
        n_units = fields['trial'].shape[0]
        for i in range(n_units):
            unit = {'probe_idx': probe_idx}
            for key in ['tm', 'trial', 'trialtm', 'quality', 'site']:
                ...
                unit[key] = np.array(target[()]).squeeze()
            units.append(unit)
    return units
```

iii. Step 6 note: *"conversion now reconstructs neural trial matrices from raw cluster `trial` / `trialtm` fields rather than assuming precomputed `trialdat`"*, after discovering (Step 6 note) that *"`/obj/clu` is a 2x1 cell array of per-probe cluster structs ... with fields `tm`, `site`, `quality`, `spkWavs`, `trialtm`, and `trial`"*. Nothing is said about probe selection.

## 2-b. How is the `neural` data processed?

i. Raw **spike counts** per 25 ms bin, cast to `float32`. No conversion to Hz, no Gaussian smoothing, no normalisation, no baseline subtraction. Each unit's spikes for a trial are histogrammed into the fixed edge grid with `np.histogram`. The reference instead divides by the bin width to get Hz and smooths with a 14 ms Gaussian (`gausswin(15)` in the authors' code). Units are stacked in the order they come off the probes into `(n_neurons, n_timepoints)` per trial.

ii.
```python
def build_neural_trials(units, n_trials) -> List[np.ndarray]:
    time = common_time_axis()
    edges = np.concatenate([[time[0] - BIN_SIZE_S / 2], time + BIN_SIZE_S / 2])
    mats = [np.zeros((n_units, n_time), dtype=np.float32) for _ in range(n_trials)]
    for ui, u in enumerate(units):
        ...
        for tr in range(1, n_trials + 1):
            mask = trials == tr
            if np.any(mask):
                mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
    return mats
```

iii. No justification is documented. The header comment marks the parameters as provisional and they were never revisited: *"Initial implementation uses a conservative common window/binning; if reference params indicate otherwise, these will be updated after inspection/validation."* The agent never reached `getDefaultParams.m` (its multi-file dump was truncated, per the Step 1 note *"the multi-file dump was truncated"*), so `params.dt = 1/200` and the smoothing kernel were never found.

## 2-c. How is the `neural` data filtered based on quality controls?

i. One filter, `filter_units`, intended as the paper's >1 Hz rule. It estimates each unit's mean firing rate as `len(clu.tm) / (n_trials × 1.0)` — i.e. it assumes **the session lasts one second per trial**. Real sessions run ~5–10 s per trial, so the estimate is inflated ~5–10× and essentially nothing is removed: session 0 (EKH1) keeps all 80 clusters from both probes, and the dataset totals 8,087 units versus the reference's 1,954. The manual curation label `clu.quality` is read into the unit dict and **never consulted**, so `garbage`/`noisy`/`poor` clusters are all kept. The paper's ≥10 units/session session criterion is not applied either (it happens to be satisfied everywhere).

ii.
```python
def filter_units(units, n_trials):
    kept = []
    session_dur = max(n_trials * 1.0, 1.0)
    for u in units:
        tm = np.array(u['tm']).astype(float).ravel()
        mean_fr = tm.size / session_dur
        if mean_fr > 1.0:
            kept.append(u)
    return kept
```

iii. Step 5 Key Decision 3: *"Use low-FR filtering consistent with reference code: Reference code removes low firing-rate clusters and methods state most analyses include units with firing rate >1 Hz; exact threshold reconciliation will be finalized in implementation/validation."* The reconciliation never happened, and the session-duration proxy is nowhere explained. Step 3 notes correctly record the rules (*">=10 units/session; >1 Hz unit filtering"*, *"only well-isolated single units >1 Hz"* for some analyses).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **It is not.** `build_neural_trials` histograms `clu.trialtm` — spike times relative to **trial start** — directly into the window `[-2.5, 2.5]`, without ever subtracting `bp.ev.goCue`. The go cue is read only for the tongue and motion-energy streams. Consequence, verified against the raw file and the pickle: `trialtm` for EKH1 spans −0.49 to 10.28 s, and the go cue sits at 2.5 s, so the stored "−2.5 → 0 s before the go cue" half of every trial is **exactly zero** (40% of bins in session 0 are identically zero across all 80 units and all trials), and the bins labelled 0 → +2.5 s after the go cue actually contain 0 → +2.5 s after *trial start*, i.e. the sample and delay epochs. The neural data is shifted by the whole go-cue latency (2.5 s in fixed-delay sessions, variable in randomized-delay sessions) and the response epoch is missing entirely.

ii.
```python
for tr in range(1, n_trials + 1):
    mask = trials == tr
    if np.any(mask):
        mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```
(compare the go cue being used for the camera streams:)
```python
align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
...
tv = ft[valid] - align_times[tr]
```

iii. The intent is documented — Step 5 Key Decision 2: *"Align all streams to goCue: This matches the decoder task requirement and the default/reference code alignment (`params.alignEvent = 'goCue'`)"* — but it is not carried out for the neural stream. No check that could have caught it (e.g. plotting a PSTH, or noticing the all-zero half-window) was performed; Step 10 "Critical Review 1" is `NOT STARTED`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **25 ms bins over a fixed [−2.5, +2.5] s window = 200 bins**, identical for every trial and session. Spikes are binned straight from spike times at this resolution, so there is no rebinning of the neural data. The camera streams are resampled onto the same 200-bin grid by linear interpolation (not averaging). The reference uses 5 ms bins (the authors' `params.dt = 1/200`) over the same window, i.e. 1000 bins. The metadata records `'time_bin_size': BIN_SIZE_S` = **0.025**, whereas the target format specifies this field in **milliseconds** (should be 25.0).

ii.
```python
BIN_SIZE_S = 0.025
T_START = -2.5
T_END = 2.5

def common_time_axis() -> np.ndarray:
    edges = np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
    centers = edges[:-1] + BIN_SIZE_S / 2
    return centers.astype(np.float32)
```
```python
'time_bin_size': BIN_SIZE_S,
'temporal_alignment_event': 'go cue onset',
'off_start': T_START,
'off_end': T_END,
```

iii. Only the provisional comment quoted in 2-b. The window is never tied to a source; `getDefaultParams.m` (`params.tmin`/`params.tmax`/`params.dt`) was listed in the Step 1 function table but its contents were never read.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Not derived from raw data: it is the vector of bin centres of the fixed conversion grid, `−2.4875 … +2.4875` in 25 ms steps, identical for every trial and session and tiled as shape `(1, 200)`. Same construction as the reference (which uses the 5 ms grid). Named `time_from_go_cue_s`.

ii.
```python
def common_time_axis() -> np.ndarray:
    edges = np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
    centers = edges[:-1] + BIN_SIZE_S / 2
    return centers.astype(np.float32)
...
input_trials.append(time[None, :].astype(np.float32))
```

iii. Step 5 Key Decision 4: *"Represent time as a continuous time-varying decoder input: This directly matches the task specification."*

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None — the grid is constructed analytically and tiled across trials. `common_time_axis()` is re-evaluated once per session (and once more inside `build_neural_trials`), which is negligible.

ii. See 3-a.

iii. N/A — no processing to justify.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction the input grid *is* the neural binning grid: `build_neural_trials` builds its edges from `common_time_axis()`, so bin *k* of `input` is bin *k* of `neural`. But because the spikes were never shifted by the go cue (2-d), the label attached to bin *k* is wrong: the input says "−2.5 s before the go cue" where the neural data holds "trial start", so the input is offset from the true go-cue-relative time of the neural data by the go-cue latency (≈2.5 s in fixed-delay sessions, trial-varying in randomized-delay sessions). The input is, however, correctly aligned to the tongue-velocity stream's own (also offset) clock, and misaligned from the motion-energy stream by a further ~0.5 s.

ii.
```python
time = common_time_axis()
edges = np.concatenate([[time[0] - BIN_SIZE_S / 2], time + BIN_SIZE_S / 2])
```
```python
input_trials.append(time[None, :].astype(np.float32))
```

iii. The intent is Step 5 Key Decision 2 (align all streams to goCue); the implementation is inconsistent with it for the neural stream.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `obj.bp.R` and `obj.bp.L` — the **instructed** trial type — plus the outcome vector to mark the no-lick class. `bp.hit` / `bp.miss` are used only via the derived `outcomes` array to set class 2; the actual licks (`bp.ev.lickL` / `bp.ev.lickR`) are never read, and hit/miss is never used to flip the direction on error trials. The reference derives direction from instructed side **combined with** hit/miss.

ii.
```python
def infer_lick_direction(bp, outcomes, n_trials) -> np.ndarray:
    R = np.ravel(bp['R'])[:n_trials] > 0
    L = np.ravel(bp['L'])[:n_trials] > 0
    lick = np.full(n_trials, 2, dtype=np.int64)
    lick[L] = 0
    lick[R] = 1
    lick[outcomes == 2] = 2
    return lick
```

iii. The Step 5 mapping table lists the intended sources as *"`obj.bp.R`, `obj.bp.L`, and lick-event logic"* with reference functions *"`firstLickTime`, `getOutcome`, `obj.bp.ev.lickL/lickR`"* and the note *"`none` for ignore/no-lick trials"*. The lick-event half of that plan was never implemented and the shortfall is not recorded anywhere.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Direct relabelling of the instructed side: L → 0 (`left`), R → 1 (`right`), then any trial whose outcome is `ignore` → 2 (`none`). The value is constant across the 200 bins of the trial. Because the direction is the *instructed* port rather than the *licked* port, every error (miss) trial carries the wrong label — from `verification_full_out.txt` these are 13.2% of all trials, so roughly one trial in eight has an inverted lick-direction label. The resulting marginals (left 0.423 / right 0.424 / none 0.153) look plausible, which masks the error.

ii.
```python
lick[L] = 0
lick[R] = 1
lick[outcomes == 2] = 2
```
```python
out = np.vstack([
    np.full((1, time.size), lick[tr], dtype=np.int64),
    ...
```

iii. No justification for equating instructed side with lick direction is offered in CONVERSION_NOTES.md.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `obj.bp.protocol.nums`: the code takes the unique finite values of that per-trial vector and, **only if there are exactly two of them**, labels the larger one class 1. In every session there is one unique value, so the fallback `np.zeros(n_trials)` stands and context is class 0 for all 11,385 trials. `obj.bp.autowater` — the field the reference uses, and which the agent itself listed among the `bp` fields in its Step 2 notes — is never read.

ii.
```python
def infer_context(sess, bp, n_trials) -> np.ndarray:
    # Conservative initial mapping: randomized-delay sessions are DR; standard ephys
    # sessions default DR unless protocol nums show multiple contexts.
    ctx = np.zeros(n_trials, dtype=np.int64)
    protocol = bp.get('protocol', {})
    nums = protocol.get('nums', None) if isinstance(protocol, dict) else None
    if nums is not None:
        vals = np.ravel(nums)[:n_trials]
        uniq = [u for u in np.unique(vals) if np.isfinite(u)]
        if len(uniq) == 2:
            ctx = (vals == uniq.max()).astype(np.int64)
    return ctx
```

iii. Step 7 note: *"Protocol numbers in sampled ephys sessions do not show multi-context variation, so constant DR context in the sample may reflect the chosen sessions rather than a loading bug."* Step 7/52 note: *"The protocol scan found no ephys sessions with more than one unique `protocol.nums`, suggesting the two-context subset is not trivially encoded there."* Step 9 note concedes the failure: *"behavioral_context remained constant DR across all converted sessions ... Additional work is needed ... to identify context labels for the two-context subset."* The agent knew from the methods that 12 two-context sessions exist and still shipped a degenerate output.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. None beyond the relabelling above. The per-trial value is broadcast across all 200 bins. `output_values[1] = ['DR', 'WC']`, so class 0 is DR — the opposite coding from the reference (WC 0, DR 1), which is harmless in itself since the names are carried. The realised distribution is `DR 1.000 / WC 0.000`: a single-valued output that carries no information and produces a degenerate decoding problem (the agent's own sample run reported *"sklearn single-label warnings"* and *"trivially perfect balanced accuracy"*).

ii.
```python
np.full((1, time.size), context[tr], dtype=np.int64),
```
```python
'output_values': [
    ...
    ['DR', 'WC'],
```

iii. Same notes as 5-a; the agent characterised this as *"a major semantic mismatch with the task requirement"* but left it in place.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three mutually exclusive per-trial flags of `obj.bp`: `hit`, `miss`, and `no`, each truncated to `Ntrials`. The reference uses `hit` and `miss` only and infers ignore as the complement; reading `no` explicitly is equivalent (the flags sum to one on every trial).

ii.
```python
def infer_outcomes(bp, n_trials) -> np.ndarray:
    hit = np.ravel(bp['hit'])[:n_trials] > 0
    miss = np.ravel(bp['miss'])[:n_trials] > 0
    no = np.ravel(bp['no'])[:n_trials] > 0
    out = np.full(n_trials, 2, dtype=np.int64)
    out[miss] = 0
    out[hit] = 1
    out[no] = 2
    return out
```

iii. Step 5 mapping table: *"`obj.bp.hit`, `obj.bp.miss`, `obj.bp.no` → output[2] outcome → Map to correct/incorrect/ignore — `getOutcome` — Likely hit->correct, miss->incorrect, no->ignore."*

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling into the prompt's three classes: miss → 0 `incorrect`, hit → 1 `correct`, everything else (and `no`) → 2 `ignore`; broadcast across the 200 bins. Realised distribution 0.132 / 0.715 / 0.153, consistent with a ~72% hit rate on a well-trained animal. Ignore trials are kept as their own class rather than dropped, matching the reference.

ii.
```python
np.full((1, time.size), outcomes[tr], dtype=np.int64),
```
```python
'output_values': [..., ['incorrect', 'correct', 'ignore'], ...]
```

iii. As 6-a; the class codes follow the prompt's ordering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. `obj.traj{1}` — the **side camera only** — specifically `ts` and `frameTimes`, with `bp.ev.goCue` for alignment. Within `ts` the code takes **hard-coded landmark index 0**, reading `ts[0, 0, :]` (x), `ts[0, 1, :]` (y) and `ts[0, 2, :]` (likelihood); it never looks up `featNames`. Index 0 of the side camera happens to be `tongue` in the sessions inspected, so the right feature is picked, but the lookup is positional and would silently take the wrong landmark if the order varied. The bottom camera's `top_tongue` (which, per the reference, tracks the tongue in roughly twice as many frames) is not used.

ii.
```python
def read_traj_trial_refs(h: h5py.File):
    traj = h['/obj/traj']
    arr = np.array(traj[()])
    if arr.dtype != object or arr.size == 0:
        return None
    return h[arr.flatten(order='F')[0]]
```
```python
ts = np.array(h[ts_refs[tr]][()])
...
x = ts[0, 0, :].astype(float)
y = ts[0, 1, :].astype(float)
p = ts[0, 2, :].astype(float)
```

iii. Step 7 note: *"The trajectory `ts` arrays are clearly structured as landmark-by-coordinate/probability over time ... We now know the trajectory arrays have shape (7 landmarks, 3 channels, time), where channel 0/1 are x/y and channel 2 is confidence. The first landmark (tongue) often has NaN x/y with tiny confidence early in trials, so visibility masking is necessary."* The feature list it recovered is *"tongue, left_tongue, right_tongue, jaw, trident, nose, lickport"* — i.e. only camera 0's.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. (1) Frames are kept where x, y and `frameTimes` are finite **and likelihood > 0.5** (the reference uses > 0.9). (2) The invalid frames are *removed from the array* and the speed is then the first difference of the surviving samples, `sqrt(Δx² + Δy²)/Δt`, assigned to the midpoint time — so a difference is taken **across** every untracked gap rather than within contiguous runs. No smoothing of x/y is applied (the reference applies a 5 ms Gaussian within each run). (3) The frame-resolution speed is **linearly interpolated** onto the 25 ms grid with `np.interp`, which fills every bin between the first and last tracked frame, including the long stretches where the tongue is not visible. (4) Only bins outside `[first finite, last finite]` are set to NaN → class 2. Per-session normalisation is not needed since only one camera is used. The net effect is that "not visible" is under-reported and interpolated pseudo-velocities are inserted into bins where the tongue is out of view: the realised split is 0.153 / 0.153 / 0.694, versus the reference's ~88% not-visible.

ii.
```python
valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ft) & (p > 0.5)
if valid.sum() < 3:
    continue
xv = x[valid]; yv = y[valid]; tv = ft[valid] - align_times[tr]
dt = np.diff(tv); good = dt > 0
speed = np.sqrt(np.diff(xv)**2 + np.diff(yv)**2) / dt
tmid = (tv[:-1] + tv[1:]) / 2
...
yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
idx = np.where(np.isfinite(yint))[0]
first, last = idx[0], idx[-1]
yint[:first] = np.nan
yint[last+1:] = np.nan
```

iii. Step 7 note: *"tongue velocity is likely derivable from tongue landmark x/y trajectories ... compute speed from tongue x/y where confidence is above a threshold and coordinates are finite, align to go cue using frameTimes, interpolate to the common time axis, and discretize per session median with missing category 2 elsewhere."* Nothing justifies the 0.5 likelihood cut, the differencing across gaps, or the interpolation across invisible stretches.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. `discretize_session_median` pools **all bins of all trials of the session**, takes the median of the finite entries, and assigns 0 below it, 1 at or above it, 2 where the value is NaN. This is exactly the prompt's per-session 50th-percentile rule and matches the reference (`np.nanpercentile(..., 50)`), giving an even 0/1 split among visible bins (0.153 / 0.153 in the full data).

ii.
```python
def discretize_session_median(arr2d, missing_code: int) -> np.ndarray:
    valid = np.isfinite(arr2d)
    if not np.any(valid):
        return np.full(arr2d.shape, missing_code, dtype=np.int64)
    thr = np.nanmedian(arr2d[valid])
    out = np.full(arr2d.shape, missing_code, dtype=np.int64)
    out[valid & (arr2d < thr)] = 0
    out[valid & (arr2d >= thr)] = 1
    return out
```

iii. Step 5 Key Decision 6: *"Continuous streams will be discretized per session using 50th-percentile thresholds, with explicit missing/not-visible category 2."* Key Decision 7: *"Use no-video / not-visible category rather than imputing: This preserves missingness required by the task and avoids introducing artificial movement signals."*

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. `frameTimes − goCue[trial]`, then interpolated onto the shared 200-bin grid. The **video-clock offset is not applied**. The camera clock leads the behaviour clock by ~0.49 s in these files (computed here from `sglx.bitcode.bitstart/sglx.fs` minus `bp.ev.bitStart`, exactly the reference's `findVideoOffset.m`; the authors' `processME.m` hard-codes the same correction as `frameTimes − 0.5`). So tongue velocity is placed ~0.49 s **later** than it actually occurred. Against the neural stream, which is itself shifted by the go-cue latency (2-d), the two are offset by seconds; against the code's own motion-energy stream, which does subtract 0.5, the tongue is ~1.0 s out of register.

ii.
```python
align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
...
tv = ft[valid] - align_times[tr]
```
(contrast the motion-energy path in the same file:)
```python
old_t = frame_times - 0.5 - align_times[tr]
```

iii. The agent recorded the correction it needed — Step 7 note on `processME.m`: *"use `obj.traj{1}(trix).frameTimes - 0.5 - alignTimes(trix)`"* — and applied it to motion energy but not to the tongue. `findVideoOffset.m` is listed in its own Step 1 function table as *"Align behavior video timing to task timebase"* and was never used.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. **Nothing.** No paw variable is read; the output is hard-coded to class 2 (`not_visible`) for every bin of every trial of every session. The agent concluded from inspecting `obj.traj{1}` (side camera: `tongue, left_tongue, right_tongue, jaw, trident, nose, lickport`) that the dataset has no paw landmarks. That conclusion is false: `obj.traj{2}` (bottom camera) lists `top_tongue, topleft_tongue, bottom_tongue, bottomleft_tongue, top_paw, bottom_paw, lickport, jaw, top_nostril, bottom_nostril` — verified here directly in `data_structure_EKH1_2021-08-07.mat`. The second camera was never opened.

ii.
```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. Step 7/8 note: *"Paw landmarks have not yet been identified in the inspected ephys sessions, so paw may remain missing unless discovered elsewhere."* Step 7 note: *"The feature-name scan across multiple ephys sessions consistently shows only tongue/jaw/nose/lickport/trident landmarks and no paw landmarks, so paw likely is unavailable in these ephys video features. Therefore, leaving paw as all `not_visible` is justified for these sessions."* The scan only ever covered camera 0.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. None. There is no paw velocity computation in the script.

ii.
```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. As 8-a.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It is not thresholded — the constant class 2 is written directly, so `output_values[4] = ['lt50', 'ge50', 'not_visible']` advertises two velocity classes that never occur. The verifier reports `paw_velocity: {not_visible (1.000)}` and a range of `[2.0, 2.0]`, i.e. a second degenerate output (the reference has ~19% not-visible and a balanced 0/1 split over the rest).

ii.
```python
'output_values': [..., ['lt50', 'ge50', 'not_visible'], ...]
```
```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. As 8-a. The agent noted the consequence at Step 8 — *"`behavioral_context` and `paw_velocity` are degenerate in the sample (single-label outputs), producing sklearn warnings and trivially perfect balanced accuracy"* — and did not fix it.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. No alignment is performed; the constant class-2 vector is simply given the same 200-bin length as the other streams.

ii.
```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. As 8-a.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The companion `motionEnergy_<anm>_<date>.mat` beside each data structure, read with `scipy.io.loadmat(..., struct_as_record=False)` and accessed as `me.data`, a per-trial cell of one value per camera frame. Same source as the reference. The loader handles only the common single-wrapped layout: for the doubly-wrapped files (`me.data.data`, e.g. `motionEnergy_JEB15_2022-07-26.mat`) `data_arr` collapses to a single struct element, and for the bare-cell-array files (e.g. `motionEnergy_JEB23_2023-10-10.mat`) `getattr(me, 'data', None)` returns `None` and the function bails out. Verified here: six of the 34 converted sessions (JEB15 07-26, JEB15 07-28, JEB23 10-10/11/12/13) end up with motion energy class 2 on 100% of bins. The reference unwraps in a `while` loop and recovers all 44.

ii.
```python
def load_motion_energy(path):
    if path is None:
        return None
    m = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return m['me'].flat[0]

def align_motion_energy(me, bp, n_trials, time):
    if me is None:
        return None
    data = getattr(me, 'data', None)
    if data is None:
        return None
```

iii. Step 7 note: *"Motion energy is available as per-trial 400 Hz vectors with a session-level `moveThresh`, and the reference helper `processME.m` shows exactly how to align/interpolate it onto the trial time axis."* The layout variation across files is not mentioned anywhere, and the six all-missing sessions are not investigated (Step 9 note only says motion energy is *"entirely missing in some"* sessions).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. No reduction is needed (the file already holds one scalar per frame). The trace is linearly interpolated onto the 25 ms grid with `np.interp`, then the leading/trailing NaNs are **nearest-filled** so the value is extended to the edges of the window. This mirrors `processME.m` (`interp1` + `fillmissing(...,'nearest')`) rather than the reference solution's bin-averaging, and it means that within a session that has video, class 2 never occurs (the realised split is exactly 0.500 / 0.500 / 0.000 for the 28 sessions with usable files).

ii.
```python
y = np.interp(time, old_t[valid], x[valid], left=np.nan, right=np.nan)
...
# nearest fill for edge NaNs to mimic reference helper behavior
idx = np.where(np.isfinite(y))[0]
first, last = idx[0], idx[-1]
y[:first] = y[first]
y[last+1:] = y[last]
```

iii. Step 7 note, quoting `processME.m` directly: *"then fill NaNs with nearest values"*, and the code comment *"mimic reference helper behavior"*. This is the one place the AI faithfully reproduced an authors' helper.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_session_median` as the tongue: session-wide median over all finite bins, 0 below / 1 at-or-above, 2 where NaN. Matches the prompt's 50th-percentile rule and the reference. `output_values[5] = ['lt50', 'ge50', 'no_video']`, following the prompt's wording for the third class.

ii.
```python
motion_disc = discretize_session_median(motion_aligned, missing_code=2) if motion_aligned is not None else None
...
motion = motion_disc[tr] if motion_disc is not None else np.full(time.shape, 2, dtype=np.int64)
```

iii. Step 5 Key Decisions 6 and 7 (quoted in 7-c).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times are **synthesised** as `(1 … n)/400` — the constant 400 Hz fallback — rather than read from `obj.traj{1}(trial).frameTimes`, even though the real times are present in the file and are used for the tongue. Then `− 0.5 − goCue[trial]`, and interpolation onto the shared grid. The 0.5 s is `processME.m`'s video-clock correction, but in that function it is subtracted from *real* frame times, which start at ~0.515 s; subtracting it from times that start at 0.0025 s double-counts the offset. Checked against `data_structure_EKH1_2021-08-07.mat`: the true aligned time of the first frame is `0.5153 − 0.49 = +0.025 s` into the trial, while the code assigns it `−0.4975 s` — motion energy is placed **~0.52 s earlier** than it occurred, and ~1.0 s away from where the same code puts the tongue.

ii.
```python
frame_times = np.arange(1, x.size + 1, dtype=float) / 400.0
old_t = frame_times - 0.5 - align_times[tr]
```

iii. The agent's own Step 7 note records the correct precedence — *"use `obj.traj{1}(trix).frameTimes-0.5-alignTimes(trix)` or fallback `(1:n)/400 - 0.5 - alignTimes(trix)`"* — but only the fallback branch was implemented. No check of the resulting alignment (e.g. motion energy rising at the go cue) was made.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Mostly by skipping rather than by repairing, and always silently at the level of the output. (a) **Whole sessions**: any exception in `build_session` is caught in `main()` and the session is dropped — this swallowed all 11 v5-format files and the 2 files with no `clu` group, i.e. 13 of 47. (b) **Schema variation between sessions**: handled for the cluster fields (`site` vs `channel`) by only reading keys that are present, and by requiring `trial`/`trialtm`. (c) **Missing/short camera data**: guarded by `min(n_trials, len(ts_refs), len(ft_refs))` and by `continue` on too few valid frames — the affected trial then stays NaN → class 2. (d) **Missing motion energy**: `None` → the whole session becomes class 2, which also silently absorbs the two unhandled `.mat` layouts (9-a). (e) **Untracked frames**: not imputed for the tongue in the sense of a class, but effectively imputed by `np.interp` across gaps (7-b); explicitly nearest-filled at trial edges for motion energy (9-b). (f) **Trials past the end of the recording**: not detected; they enter as all-zero neural matrices (one is flagged by the verifier).

ii.
```python
except Exception as e:
    print(f'SKIP {sess.data_file.name}: {e!r}')
    skipped.append((sess.data_file.name, repr(e)))
    continue
```
```python
for key in ['tm', 'trial', 'trialtm', 'quality', 'site']:
    if key not in fields:
        continue
```
```python
n = min(n_trials, len(ts_refs), len(ft_refs))
```

iii. Step 5 Key Decision 7: *"Use no-video / not-visible category rather than imputing: This preserves missingness required by the task and avoids introducing artificial movement signals"* — followed for whole-session gaps, but contradicted by the tongue interpolation and the motion-energy edge fill. The session skipping is acknowledged as unresolved in the Step 9 note rather than defended.

## 11-a. What are the most time-consuming steps of the code?

i. The AI did no timing analysis: Step 7 is marked `NOT STARTED`, its "Run Time Estimates" table is empty, the script prints no timing, and `--show-processing` is accepted as a flag but does nothing. From reading the code, the dominant cost is `build_neural_trials`: for every unit it re-scans the unit's whole spike-trial vector once per trial and calls `np.histogram` — `n_units × n_trials` Python-level iterations (for the 762-unit, 252-trial session that is ~192,000 boolean scans plus histogram calls), against the reference's single `histogram2d` per unit. Secondary costs are the per-trial HDF5 dereferencing of `ts`/`frameTimes` in `align_tongue_speed`, and reading `clu.tm` for every cluster (used only by the broken firing-rate filter). Reading the files, which dominates the reference's 135 s runtime, is a smaller share here because the AI reads only selected sub-trees.

ii.
```python
for ui, u in enumerate(units):
    trials = np.array(u['trial']).astype(int).ravel()
    trialtm = np.array(u['trialtm']).astype(float).ravel()
    for tr in range(1, n_trials + 1):
        mask = trials == tr
        if np.any(mask):
            mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```

iii. No justification exists; the instructions' requirement to *"Print timing information to find bottlenecks"* and to estimate full-run time in Step 7 was not carried out.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three, none of which were vectorised. (1) The `units × trials` double loop in `build_neural_trials` — the entire per-unit block is one `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, edges])` call, as in the reference. (2) The allocation `mats = [np.zeros((n_units, n_time)) for _ in range(n_trials)]` builds `n_trials` separate arrays that are then filled column-block by column-block; one `(n_units, n_trials, n_bins)` array sliced at the end is both faster and contiguous. (3) The per-trial output assembly loop `for tr in range(n_trials)` does six `np.full` allocations and a `np.vstack` per trial, where the reference fills one `(n_trials, 6, n_bins)` array by broadcasting. The per-trial camera loops are genuinely irregular (different frame counts per trial) and reasonably stay as loops.

ii.
```python
mats = [np.zeros((n_units, n_time), dtype=np.float32) for _ in range(n_trials)]
```
```python
for tr in range(n_trials):
    ...
    out = np.vstack([
        np.full((1, time.size), lick[tr], dtype=np.int64),
        ...
    ])
```

iii. Not discussed; CONVERSION_NOTES.md Step 6 "Code inefficiencies identified" and "Code speedups added" are both left as the literal template placeholder `[Note]`.

## 11-c. What processing does the code repeat multiple times?

i. (1) `trials == tr` — the unit's full spike-trial vector is re-compared for every one of the `n_trials` trials, so each spike array is scanned `n_trials` times instead of once. (2) `common_time_axis()` is recomputed in `build_session` and again inside `build_neural_trials`, and the bin edges are rebuilt from it; the reference builds the grid once at module scope. (3) `np.array(u['trial'])`/`np.array(u['trialtm'])` casting happens inside the unit loop, which is fine, but `clu.tm` is read from disk for every cluster purely to feed the firing-rate estimate. (4) The whole `build_session` body is duplicated verbatim in the dead `build_session_scipy`, so any fix has to be made twice. Nothing is cached between sessions, which is correct since nothing is shared.

ii.
```python
for tr in range(1, n_trials + 1):
    mask = trials == tr
```
```python
time = common_time_axis()                # in build_session
...
def build_neural_trials(units, n_trials):
    time = common_time_axis()            # and again here
    edges = np.concatenate([[time[0] - BIN_SIZE_S / 2], time + BIN_SIZE_S / 2])
```

iii. Not discussed.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `read_probe_locations(h)` walks `obj.meta.probe.loc` and decodes the region strings, and the result `probe_locs` is then **never used** — `brain_regions` is hard-coded to `['ALM']` and `brain_region_idx` to zeros. (2) `clu.quality` is dereferenced and stored for every cluster and never consulted (this is also the missed QC filter of 2-c). (3) `clu.site`/`channel` is read and never used. (4) The entire scipy fallback block after `if __name__ == '__main__'` — `scipy_obj_to_bp`, `scipy_read_units`, `scipy_align_tongue`, `build_session_scipy` — is dead code, as are the helpers `mat_fields`, `read_dataset`, and the `extract_numeric_motion_trial` fallback branches. (5) `--show-processing` is parsed and ignored, so the diagnostic plots the instructions ask for are never produced. (6) Conversely, and more costly than any of the above, ~40% of every neural bin is stored as structural zeros because of the alignment error in 2-d.

ii.
```python
probe_locs = read_probe_locations(h)
if not probe_locs:
    probe_locs = ['ALM']
brain_region_idx = np.zeros(len(units), dtype=np.int64)
```
```python
for key in ['tm', 'trial', 'trialtm', 'quality', 'site']:
```
```python
if __name__ == '__main__':
    main()


# --- scipy fallback for non-HDF5 data_structure files ---
def scipy_obj_to_bp(obj_bp):
```

iii. Not discussed. The Step 10 review that would have caught the dead code and the unused `--show-processing` flag is marked `NOT STARTED`.
