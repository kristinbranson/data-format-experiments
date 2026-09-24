# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Sessions are discovered by **globbing a single hard-coded relative directory**, `data/Ephys_Behavior`, for `data_structure_*.mat` files; the subject and date are parsed out of the filename with a regex, and a matching `motionEnergy_<subj>_<date>.mat` sidecar is attached if present. The other three data families in `/app/data` — in particular `RandomizedDelay_Ephys_Behavior`, which the paper counts as 19 of its recording sessions — are never looked at. 25 files are discovered.

Each discovered file is then opened **twice**: once in `select_context_sessions()` (to test whether it has both autowater states) and again in `main()`. `load_mat_obj` sniffs the format: MATLAB v7.3/HDF5 files are returned as an open `h5py.File` handle (never closed), v5 files via `scipy.io.loadmat(..., struct_as_record=False)`.

After the autowater screen (25 → 22) and a runtime skip of `JEB6_2021-04-18` (its `clu` reference resolves to a bare `uint64` dataset rather than a cluster group), **21 sessions / 9 used subjects / 4,837 trials / 5,512 units** enter the output.

ii.
```python
def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
    motion_map = {}
    for mf in sorted(base.glob('motionEnergy_*.mat')):
        m = re.match(r'^motionEnergy_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', mf.name)
        if m:
            motion_map[(m.group(1), m.group(2))] = mf
    ...
        sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
    return sessions


def load_mat_obj(path: Path):
    if is_hdf5_mat(path):
        return h5py.File(path, 'r')
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return x['obj'].flat[0]
```

```python
sessions = discover_sessions()
sessions = select_context_sessions(sessions)      # opens every file a first time
...
for sess in sessions:
    with timed(f'load {sess.data_file.name}'):
        obj = load_mat_obj(sess.data_file)        # opens every file a second time
```

iii. The AI documented in Step 2 that four data families exist and in Step 4 that "the two-context task is encoded via trial labels within session data rather than via a separate top-level data directory name". Its Step 5 Key Decision 1 is therefore: "Use the two-context ephys sessions identified by presence of both DR/non-autowater and WC/autowater trial types". It repeatedly acknowledged in its notes and trajectory that this over-includes relative to the paper (22 vs the paper's 12 two-context sessions) and that "final session selection should likely be derived from those curated meta lists" (`load<ANM>_ALMVideo.m`), but never implemented that; the run ended with the notes reading "reference-matching curation remains unresolved". No justification is given anywhere for excluding `RandomizedDelay_Ephys_Behavior` — it appears to be an unremarked consequence of hard-coding one glob path. Skipping `JEB6` was justified pragmatically: "the pragmatic fix is to skip sessions whose `clu` structure does not match the expected trial-aligned spike format".

## 1-b. How are the data split into subjects?

i. The subject is the first regex group of the filename (`data_structure_JEB13_2022-09-13.mat` → `JEB13`), stored on the `SessionRecord`. `subjects` is the sorted set of unique subject strings, and `subject_idx` appends `subj_to_idx[sess.subject]` for each session actually written.

A bug: `subjects` is computed **before** the per-session loop, so the subject of the session that is later skipped (`JEB6`) remains in `subjects` with zero sessions. The verification log accordingly reports "Number of subjects: 10" while listing only 9 subjects with sessions.

ii.
```python
m = re.match(r'^data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', df.name)
subj, day = m.group(1), m.group(2)
sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
```
```python
subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
...
for sess in sessions:
    ...
    if len(neural) < 2:
        print(f'[warn] skipping ...'); continue      # subjects list already fixed
    data['subject_idx'].append(subj_to_idx[sess.subject])
```

iii. Step 5 mapping table: "Subject metadata from filename / `obj.meta`" → "Extract unique mouse IDs and per-session indices … Session order must match neural/input/output lists". The AI noted that `obj.meta` contains `anm`, but chose the filename. No justification is given for the dangling subject entry (it appears unnoticed — Step 10 was never started).

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file in `Ephys_Behavior` = one session = one element of `neural`/`input`/`output`. A session is dropped if (a) `autowater` is missing or does not take both values 0 and 1, (b) its `clu` group lacks `trial`/`trialtm`, (c) any exception is raised in `build_session`, or (d) fewer than 2 valid trials survive. Only probe 1 is ever read (`clu_root[0,0]`), so the second probe of two-probe sessions is silently dropped.

ii.
```python
def select_context_sessions(sessions):
    keep = []
    for s in sessions:
        ...
        autowater = get_trial_bool(bp, 'autowater')
        if autowater is None: continue
        vals = set(np.unique(autowater.astype(int)).tolist())
        if vals == {0, 1}:
            keep.append(s)
    return keep
```
```python
clu_root = obj['obj']['clu']
clu = obj[clu_root[0,0]]                      # probe 1 only
if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
    raise ValueError('unsupported clu structure')
```
```python
        except Exception as e:
            print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True); continue
        if len(neural) < 2:
            print(f'[warn] skipping {sess.data_file.name}: <2 valid trials', flush=True); continue
```

iii. As in 1-a: the autowater screen implements Step 5 Key Decision 1 (two-context subset identified from trial labels). The `clu` skip is justified in the trajectory as curation-flavoured pragmatism. The probe-1-only choice is never mentioned or justified anywhere in the notes or trajectory.

## 1-d. How are the data split into trials?

i. Trial count comes from the length of `obj.bp.ev.goCue` (`n_trials = go.shape[0]`), not from `bp.Ntrials`. All per-trial behavioural fields are read as flat vectors of the same nominal length. Spikes are assigned to a trial by the 1-based `clu.trial` field (`spikes = tm_arr[tr_arr == tr1]`, `tr1 = tr + 1`). Video frames are assigned to a trial by the per-trial cell index into `traj.frameTimes`/`traj.ts`; motion energy by the per-trial cell index into `me.data`. Surviving trials are the index array `trial_idx = np.where(valid)[0]`.

ii.
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
...
trial_idx = np.where(valid)[0]
for tr in trial_idx:
    ...
    tr1 = tr + 1
    for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
        spikes = tm_arr[tr_arr == tr1]
```

iii. Step 5: "Align to `goCue` … trial direction fields … per-trial categorical". The AI's Step 2 notes record that "`bp/ev` includes event arrays … each with 365 trials in the inspected session", i.e. it treated the event-array length as the trial count. No discussion of `bp.Ntrials` or of fields stored longer than the trial count appears anywhere.

## 1-e. How are trials filtered based on quality controls?

i. A conjunction of five masks, all applied before any computation:
- drop early-lick trials (`~early`);
- **require `hit | miss`** — i.e. ignore/no-response trials are deleted from the dataset entirely;
- drop photostimulation trials (`~stim.enable`) when the field exists;
- require `right ^ left` (exactly one instructed side);
- `np.isin(autowater, [0, 1])`, which is a no-op on a boolean array.

There is **no** filter for trials that run past the end of the recording. Each mask is applied only `if ... is not None`, so a missing field silently disables that filter.

ii.
```python
valid = np.ones(n_trials, dtype=bool)
if early is not None:
    valid &= ~early
if hit is not None and miss is not None:
    valid &= (hit | miss)
if stim_enable is not None:
    valid &= ~stim_enable
if right is not None and left is not None:
    valid &= (right ^ left)
if autowater is not None:
    valid &= np.isin(autowater.astype(int), [0, 1])
trial_idx = np.where(valid)[0]
```

iii. Step 5 Key Decision 4: "Exclude early-lick and ignore trials, following methods text and condition definitions in `getDefaultParams.m`." The paper does omit early-lick and ignore trials from its analyses. But the AI also recorded the direct conflict this creates with the decoder specification, in its own words: "enforcing the exact hit-only/no-stim conditions from `getDefaultParams.m` makes the `outcome` output degenerate (all correct), which conflicts with the decoder task requirement to predict incorrect vs correct outcome. For the decoder-task conversion, miss trials likely need to remain included so `outcome` is nontrivial." It resolved that conflict by keeping miss trials but still dropping ignore trials — even though the Decoder Task section explicitly asks for an `ignore` class in `outcome` and a `none` class in `lick_direction`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two fields of the probe-1 cluster struct in the v7.3 file: `obj.clu{1}.trial` (1-based trial index of each spike) and `obj.clu{1}.trialtm` (spike time relative to that trial's start). Every cluster on probe 1 is read; `obj.clu{1}.quality`, `obj.clu{1}.site`, `obj.clu{2}`, and `obj.bp.ev.goCue` are **not** used in building the neural matrix.

ii.
```python
clu_root = obj['obj']['clu']
clu = obj[clu_root[0,0]]
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    tr_ref = clu['trial'][ci,0]
    tm_ref = clu['trialtm'][ci,0]
    tr_arr = np.asarray(obj[tr_ref][()]).reshape(-1).astype(int)
    tm_arr = np.asarray(obj[tm_ref][()]).reshape(-1).astype(float)
    clu_trial.append(tr_arr)
    clu_trialtm.append(tm_arr)
```

iii. Step 1/Step 2 notes: "`clu` contains 32 clusters in the example session, each with raw spike times `tm`, trial indices `trial`, and trial-aligned times `trialtm`. This means we can directly bin spikes per neuron per trial around the alignment event." The AI also observed that "`quality` is stored as strings via uint16 character codes" but never used it. The v5-file branch has no neural extraction at all (`neural_trials.append(np.zeros((1, time_bins.size)))`), though in practice all surviving sessions were v7.3.

## 2-b. How is the `neural` data processed?

i. Raw spike **counts** per 75 ms bin, `np.float32`. There is no conversion to Hz (no division by the bin width), no smoothing, no normalisation, no baseline subtraction. The stored values are therefore integers up to ~23 counts/bin.

ii.
```python
if clu_trial:
    mat = np.zeros((len(clu_trial), time_bins.size), dtype=np.float32)
    tr1 = tr + 1
    for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
        spikes = tm_arr[tr_arr == tr1]
        if spikes.size:
            mat[ci], _ = np.histogram(spikes, bins=binedges)
    neural_trials.append(mat)
```

iii. Nothing in CONVERSION_NOTES.md or the trajectory justifies leaving the data as counts or omitting smoothing; the notes never discuss it. (The reference code's `params.smooth = 15` / `gausswin(15)` is never mentioned by the AI.)

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control is applied at all.** Every cluster on probe 1 is kept regardless of its `quality` label and regardless of its firing rate. The result is 5,512 units over 21 sessions (mean 262/session, max 740) — versus 1,954 in the human reference over 44 sessions.

ii. There is no filtering code. The full extraction loop is:
```python
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    ...
    clu_trial.append(tr_arr)
    clu_trialtm.append(tm_arr)
```
and every one of those clusters becomes a row of every trial's matrix.

iii. This directly contradicts the AI's own stated plan. Step 3 notes: "For most analyses, include all units with firing rate >1 Hz"; Step 5 Key Decision 5: "Apply low firing-rate filtering consistent with `removeLowFRClusters.m`; target unit inclusion is all units >1 Hz". Step 4 also flags the paper's ≥10-units-per-session session criterion. None of this reached the code, and Step 10 (which would have caught it) was never started.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **It is not.** `clu.trialtm` is spike time relative to *trial start*, and the code histograms it directly against bin edges spanning −1.5375 … +1.5375 s without ever subtracting `goCue`. `goCue` is loaded (`go = get_event(obj, ALIGN_EVENT)`) but is used only to obtain `n_trials` and, separately, inside the trajectory function. In `EKH1_2021-08-07` the go cue sits at 2.5 s after trial start, so the extracted "go-cue-aligned" neural window is in fact the interval from −4.0 s to −1.0 s relative to the go cue — the ITI/sample/delay period, with the entire response epoch excluded. In the randomized-delay-style sessions the go cue also varies trial to trial, so the misalignment is not even constant.

ii. The trajectory path does subtract the go cue:
```python
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
```
The neural path does not:
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
spikes = tm_arr[tr_arr == tr1]
if spikes.size:
    mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The stated decision was correct — Step 5 Key Decision 2: "Align all modalities to `goCue`, matching the decoder task specification"; Step 1: "`alignSpikes` is the likely reference for event alignment of neural data; we need to match its alignment event and trial window during conversion". The metadata written to the pickle also claims `'temporal_alignment_event': 'Go cue onset'`. The implementation simply never did it, and because Steps 10–12 (the sanity-check / spot-check / accuracy-review steps that were specifically designed to catch this) were never started, it went undetected. The sample decoder's near-chance `lick_direction` (0.5071) and `outcome` (0.5005) are the direct symptom.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 75 ms bins, 41 of them, centred on `np.arange(-1.5, 1.5, 0.075)`; bin edges therefore span −1.5375 … 1.5375 s. Spikes are histogrammed **directly** into these 75 ms bins from the raw spike times — there is no intermediate fine binning and no rebinning step. The window is ±1.5 s rather than the reference's ±2.5 s (the paper's `params.tmin`/`tmax`), and the bin is 75 ms rather than the reference's `params.dt = 1/200` = 5 ms. Note also that `metadata['time_bin_size']` is written as `0.075` although the target format specifies **ms**, so the metadata reads as 0.075 ms.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
```
```python
'time_bin_size': BIN_SIZE_S,
```

iii. Step 5 Key Decision 3: "Use 75 ms bins because the reference context decoder scripts explicitly use `rez.binSize = 75` ms." Step 1 notes also spotted the two-level scheme — "`rez.dt = floor(rez.binSize / (params(1).dt*1000))`, implying a finer native time base in `params.dt` that is rebinned for decoding" — but the AI adopted the decoder-level bin rather than the data-level bin. The ±1.5 s window is never justified in the notes.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. It is the fixed bin-centre vector `time_bins`, defined from the module-level constants `T_START = -1.5`, `T_END = 1.5`, `BIN_SIZE_S = 0.075`, identical for every trial of every session.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
...
inp = time_bins[None, :].astype(np.float32)
input_trials.append(inp)
```

iii. Step 5 mapping table: "Trial time relative to go cue → input[0] → Continuous time-from-go-cue vector repeated for each trial/time bin … Decoder input specification requires time from go cue onset."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The vector is constructed once per session and broadcast as a `(1, 41)` array into each trial. Values run −1.5 … 1.5 s in 0.075 s steps, confirmed by the verification log (`time_from_go_cue: [-1.5, 1.5]`).

ii.
```python
inp = time_bins[None, :].astype(np.float32)
```

iii. N/A — the AI treats this as a definitional axis, matching the decoder-input specification.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Formally, by construction: `time_bins` are exactly the centres of the `binedges` used to histogram spikes, so input bin *k* and neural bin *k* are the same interval of the code's internal axis. **Substantively, however, the alignment is wrong**, because the neural axis is time-from-trial-start, not time-from-go-cue (see 2-d). Input bin labelled `0.0 s` therefore holds neural activity from ~2.5 s *before* the go cue in the fixed-delay sessions. The camera-derived outputs use a third, different axis (see 7-d/9-d), so the three streams are not mutually consistent.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
mat[ci], _ = np.histogram(spikes, bins=binedges)      # spikes are relative to trial start
```

iii. The AI's documented intent was a single shared go-cue axis ("Align all modalities to `goCue`"). It never verified it — its own planned sanity check "Verify one trial's binned neural counts against raw spike times aligned to `goCue` using direct recomputation" was written into Step 5 and never performed.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. A single field, `obj.bp.R` (with `obj.bp.L` used only inside the `right ^ left` validity mask). `hit`/`miss` are **not** consulted when assigning direction.

ii.
```python
right = get_trial_bool(bp, 'R')
left  = get_trial_bool(bp, 'L')
...
lick_dir = 1 if bool(right[tr]) else 0
```

iii. Step 5 mapping table: "Trial direction fields (`R`/`L` or equivalent) → output[0] lick_direction → Per-trial categorical: left=0, right=1 … **Use trial labels rather than post-hoc lick timestamps for direction**." That parenthetical is the whole justification: the AI decided to use the instructed side as the label.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct relabelling of the instructed side: `right → 1`, otherwise `0`, broadcast across all 41 bins. Two classes only (`['left', 'right']`). Because ignore trials were removed at 1-e, the required third class `none` never appears; and because `hit`/`miss` are not used, **miss trials are mislabelled** — on a miss the animal licked the port opposite the instructed one, so the stored label is the opposite of the actual lick direction. From the verification log, incorrect trials are 17.0% of the dataset, so roughly 17% of the `lick_direction` labels are inverted. The variable as stored is "instructed direction", not "lick direction".

ii.
```python
lick_dir = 1 if bool(right[tr]) else 0
output_trials.append(np.vstack([
    np.full(time_bins.shape, lick_dir, dtype=np.int64),
    ...
]))
```
```python
'output_values': [
    ['left', 'right'],
    ...
```

iii. No justification is offered for equating instructed side with lick direction, nor for the missing `none` class. The AI's Step 5 planned sanity check "Verify lick-direction labels match raw `R`/`L` trial fields for several manually checked trials" would have confirmed the code but not the semantics; it was never run either way.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`. The same field also gates session inclusion (1-c).

ii.
```python
autowater = get_trial_bool(bp, 'autowater')
```

iii. Step 4 notes: "`NeuralContextDecoding.m` and `DLC_ContextDecoding.m` define context as `afccond = [1 2]` versus `awcond = [3 4]`, i.e. non-autowater (DR/2AFC) versus autowater (WC) trial groupings." This is read straight off the reference code and is correct.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: `autowater → 0` (WC), else `1` (DR), broadcast across all 41 bins. Two classes, `['WC', 'DR']`, matching the codes implied by the Decoder Task ordering.

ii.
```python
context = 0 if bool(autowater[tr]) else 1
```
```python
'output_values': [ ..., ['WC', 'DR'], ... ]
```

iii. Step 5 Key Decision 6: "Convert raw `autowater`-style labels to task-required convention `WC=0`, `DR=1`." The resulting distribution (18.5% WC / 81.5% DR) is the one output the decoder handles well (0.711 balanced accuracy on the sample), which the AI noted.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` only (`obj.bp.miss` is read but used solely in the `hit | miss` validity mask). `obj.bp.no` / ignore flags are not read.

ii.
```python
hit  = get_trial_bool(bp, 'hit')
miss = get_trial_bool(bp, 'miss')
...
outcome = 1 if bool(hit[tr]) else 0
```

iii. Step 5 mapping table: "Outcome fields (`hit`/`miss`, possibly ignore/early exclusions) → output[2] outcome → Per-trial categorical: incorrect=0, correct=1 after excluding early/ignore trials … Need to define incorrect carefully for remaining non-ignored trials."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `hit → 1` (correct), else `0` (incorrect), broadcast across all 41 bins. Two classes, `['incorrect', 'correct']`. Because ignore trials were removed at 1-e, the required third class `ignore` never appears, and within the surviving set `not hit` is exactly `miss`, so the two-class labelling is internally consistent. Resulting distribution: 17.0% incorrect / 83.0% correct.

ii.
```python
outcome = 1 if bool(hit[tr]) else 0
```
```python
'output_values': [ ..., ['incorrect', 'correct'], ... ]
```

iii. The AI explicitly reasoned about the trade-off (quoted in full at 1-e): keeping miss trials so that `outcome` would not be degenerate, while still following the paper in dropping ignore trials. It never reconciled that with the Decoder Task's explicit three-class specification `(incorrect, correct, ignore)`.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. `obj.traj{view}.ts` (x, y, likelihood per feature per frame), `obj.traj{view}.featNames`, `obj.traj{view}.frameTimes`, and `obj.bp.ev.goCue`. The feature is chosen by scanning cameras in file order (outer loop) and, within each camera, candidate names in list order (inner loop). Because the **camera loop is the outer loop**, the side camera (`traj{0}`) always wins as soon as it contains any candidate — and it contains `tongue`. So despite `top_tongue` being first in the candidate list, the code always uses the **side camera `tongue`** marker and never the bottom camera's `top_tongue`. Only one view is used; the two views are never combined.

ii.
```python
for i in range(traj_ds.shape[0]):                 # cameras: side first
    traj = obj[traj_ds[i,0]]
    ...
    feat_idx = None
    for cand in feature_candidates:               # candidate order only breaks ties *within* a camera
        if cand in names:
            feat_idx = names.index(cand); break
    if feat_idx is None:
        continue
    ...
    break
```
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr,
    ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue',
     'tongue', 'left_tongue', 'right_tongue'], time_bins) ...
```

iii. The AI diagnosed the sparsity problem correctly (step 43: "the current extraction chooses entry 0 feature `tongue`, but that feature has extremely sparse support … finite fraction only ~2% to 15% of bins") and stated it had "patched the converter to prefer entry-1 tongue features first". The patch reordered the candidate list but left the camera loop outermost, so the stated fix has no effect. At step 47 it re-diagnosed: "even when searching across all candidates, the best tongue feature in EKH1 is still from traj entry 0" — and moved on without resolving it.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, all inside `extract_hdf5_traj_velocity`:
1. **Frame-to-frame finite difference** of x and y (`np.diff(..., prepend=first)`) divided by `np.diff(frameTimes)`, giving speed = `hypot(dx, dy)/dt`. Non-positive `dt` is replaced by the median positive `dt`. There is **no smoothing** of the position trace and **no likelihood filtering** (the likelihood channel `ts[feat_idx, 2, :]` is never read — the code slices `ts[feat_idx, :2, :]`). Missing tracking survives only because the authors already wrote NaN into x/y.
2. **Linear resampling** onto the 41 bin centres with `np.interp(time_bins, rel_t, speed, left=nan, right=nan)` — a point sample at each bin centre rather than an average over the bin.
3. **Gap filling**: every remaining NaN inside the trial is filled by linear interpolation from the surviving finite bins, with constant extrapolation at the edges. Given that only 7–17% of bins have the tongue in view, this **fabricates a velocity for roughly 85–93% of bins**.
4. No cross-camera normalisation (only one camera is used).

ii.
```python
xy = ts[feat_idx, :2, :]
dx = np.diff(xy[0], prepend=xy[0,0])
dy = np.diff(xy[1], prepend=xy[1,0])
dt = np.diff(ft, prepend=ft[0])
dt[dt <= 0] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0
speed = np.sqrt(dx*dx + dy*dy) / dt
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])       # fabricates the invisible bins
```

iii. The gap fill was introduced deliberately to fix the class balance, not on physiological grounds. Step 54: "We should implement nearest fill across the aligned binned trajectory speed trace prior to thresholding, analogous to motion energy. This may produce a more stable binary tongue variable." Step 56 records the outcome as success: "after nearest-fill interpolation, both tongue_velocity and paw_velocity are now well balanced in the JEB13 sample (tongue ~55.5/44.5 …), which is much more sensible for session-median thresholded outputs." The stated analogy is to `loadMotionEnergy.m`'s `fillmissing(...,'nearest')`, which in the reference fills only a few NaN frames at trial onset — not 90% of the trace. No justification is given for skipping likelihood filtering or smoothing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All 41-bin traces of all surviving trials of the session are concatenated, and the **50th percentile over finite values** is taken as a single per-session threshold. Bins `>= thr` → 1, bins `< thr` → 0, and bins that are still NaN (a trial where no tongue frame was ever finite) → 0. There are therefore **only two classes**; the required class 2 `not visible` is never emitted, and `output_values[3]` is declared as `['low', 'high']`. Because of the gap-filling at 7-b, the "not visible" condition has already been erased before thresholding.

ii.
```python
tongue_all = np.concatenate([np.nan_to_num(extract_hdf5_traj_velocity(obj, tr, [...], time_bins),
                                           nan=np.nan) for tr in trial_idx]) ...
tongue_thr = np.nanpercentile(tongue_all, 50) if tongue_all.size and np.isfinite(tongue_all).any() else np.nan
...
    if np.isfinite(tongue_thr):
        tmp = np.zeros(tong.shape, dtype=np.int64)
        finite = np.isfinite(tong)
        tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
        output_trials[i][3] = tmp
```

iii. Step 5 Key Decision 7: "Convert tongue velocity, paw velocity, and motion energy to binary outputs using per-session median thresholds, as required by the task." The per-session 50th-percentile split is exactly what the Decoder Task asks for; the AI simply never registered that the specification also lists a third value (`2: not visible`). Its step 51 reasoning explains the NaN→0 rule: "the most defensible approach is to threshold only on finite observed bins and assign missing bins to the low class rather than letting sparse observed high-speed bins dominate."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are converted with `rel_t = frameTimes - goCue[trial]` and sampled at the bin centres. **No video-clock offset is applied.** The reference derives the offset once per session from the bitcode (`sglx.bitcode.bitstart/sglx.fs` minus `bp.ev.bitStart`, i.e. `findVideoOffset.m`), and the reference MATLAB `loadMotionEnergy.m` uses the equivalent hard-coded `frameTimes - 0.5 - alignTimes(trix)`. Omitting it leaves a systematic ~0.5 s shift between the camera streams and the go cue. On top of that, the neural stream is itself on a trial-start axis (2-d), so tongue velocity and neural activity are offset from each other by the go-cue latency as well.

ii.
```python
ft = np.asarray(obj[traj['frameTimes'][trial_index0,0]][()]).reshape(-1).astype(float)
...
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go                              # no video-clock offset subtracted
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The AI did read the relevant reference line — Step 1 note: "data are interpolated using `interp1(frameTimes-0.5-alignTimes(trix), ..., taxis)`" — and at step 38 wrote "We need to align trajectories using actual go-cue times, similar to how `loadMotionEnergy.m` uses `frameTimes - 0.5 - alignTimes(trix)`." It then implemented `ft - go` without the `- 0.5`, validating the change only by eyeballing the resulting range at step 39: "`frameTimes - goCue` naturally spans roughly -2 s to +9.5 s, which is much more plausible … than the previous ad hoc alignment." `bitStart`/`sglx.bitcode` appear in its Step 2 inventory but were never used.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` tracking. The side camera has no paw feature, so the camera loop falls through to the bottom camera and the candidate list selects **`top_paw`** — the same single marker the human reference chose. `bottom_paw` is listed as a fallback but is never reached.

ii.
```python
paw_vals = extract_hdf5_traj_velocity(obj, tr,
    ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins) ...
```

iii. Step 38: "paw features do exist, but in `traj` entry 1, not entry 0 … the current paw extraction fails because it only searches for `paw`, `left_paw`, `right_paw`, not `top_paw`/`bottom_paw`. This is fixable." Step 43 records that the fix worked: "paw uses entry 1 `top_paw` and has much better support." No explicit reason is given for preferring `top_paw` over `bottom_paw` — it is simply first in the list — but it is the one that turns out to be well tracked.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue (same function, same code path): finite-difference speed from unsmoothed x/y with no likelihood filtering, point-sampled onto the 41 bin centres, then all interior NaNs linearly filled from the finite bins. No normalisation (only one camera).

ii.
```python
def extract_hdf5_traj_velocity(obj, trial_index0, feature_candidates, time_bins):
    ...
    speed = np.sqrt(dx*dx + dy*dy) / dt
    ...
    vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
    if np.isfinite(vals).any():
        vals = np.interp(idx, idx[good], vals[good])
```

iii. Same justification as 7-b — the fill was added to rebalance the classes (step 56: "paw ~52.8/47.2 … much more sensible"). The paw is well tracked, so the fill does far less damage here than for the tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same scheme as the tongue: one per-session 50th percentile over all finite pooled values, `>= thr` → 1, `< thr` → 0, remaining NaN → 0. Two classes only (`['low', 'high']`); the required class 2 `not visible` is never emitted. The per-session distributions in the verification log are revealing: 13 of 21 sessions come out at exactly 0.500/0.500, which is what a pure median split of an essentially continuous variable gives, i.e. no genuine "no tracking" structure is represented.

ii.
```python
paw_thr = np.nanpercentile(paw_all, 50) if paw_all.size and np.isfinite(paw_all).any() else np.nan
...
    if np.isfinite(paw_thr):
        tmp = np.zeros(paw.shape, dtype=np.int64)
        finite = np.isfinite(paw)
        tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
        output_trials[i][4] = tmp
```

iii. Step 5 Key Decision 7, as at 7-c.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue: `rel_t = frameTimes - goCue[trial]`, no video-clock offset, point sampling at the 41 bin centres. The paw is read from the bottom camera, and the code correctly takes `frameTimes` from *that* camera's traj entry (the same `traj` dict it pulled `ts` from), so it is not affected by cross-camera frame-count mismatches. The ~0.5 s video-offset error and the trial-start-vs-go-cue neural offset both apply.

ii.
```python
traj = obj[traj_ds[i,0]]                                    # the camera that has the feature
ts = np.asarray(obj[traj['ts'][trial_index0,0]][()]).astype(float)
ft = np.asarray(obj[traj['frameTimes'][trial_index0,0]][()]).reshape(-1).astype(float)
...
rel_t = ft - go
```

iii. Same as 7-d; the AI never distinguished paw alignment from tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The sidecar file `motionEnergy_<subj>_<date>.mat`, variable `me`, unwrapped down to the per-trial cell array of traces. `obj.me` (present in some sessions) is not used. If the sidecar is absent the whole output becomes NaN → class 0. Only the *values* are read — the per-trial trace length is used, but no frame times and no `me.moveThresh`.

ii.
```python
def unwrap_me_data(x):
    seen = 0
    while seen < 10:
        seen += 1
        if hasattr(x, 'data'):
            x = x.data; continue
        arr = np.asarray(x)
        if arr.dtype == object and arr.size == 1:
            elem = arr.reshape(-1)[0]
            if hasattr(elem, 'data'):
                x = elem; continue
        return x
    return x

def load_motion_energy(path, n_trials):
    if path is None or not path.exists():
        return None
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    me = x['me'].flat[0]
    data = np.asarray(unwrap_me_data(me)).reshape(-1)
    ...
```

iii. Step 9 iteration note: "full conversion failed at `motionEnergy_JEB15_2022-07-26.mat` because `me.data` entries are not uniformly numeric; at least some are nested MATLAB structs." Step 63: "for JEB15_2022-07-26, `me` itself has `data` and `moveThresh`, but `me.data` is a 1x1 object whose sole element is another `mat_struct` with its own `data` … So some files have an extra nesting level." The recursive unwrap is the fix, and it mirrors the reference's own guard.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling — but the resampling is a **linear time-warp, not an alignment**. The code builds `x_old = np.linspace(T_START, T_END, raw.size)`, i.e. it assumes each trial's motion-energy trace, whatever its length, spans exactly −1.5 … +1.5 s around the go cue, and then interpolates onto the bin centres. In reality the traces are one value per camera frame over the whole trial (~10 s at ~400 Hz) with real `frameTimes`. The effect is that each trial is stretched or squeezed by a different, trial-dependent factor and shifted so that the trial's *midpoint* lands at the go cue. Traces of length ≤ 1 become all-NaN.

ii.
```python
if me_trials is not None and tr < len(me_trials):
    raw = me_trials[tr]
    if raw.size > 1:
        x_old = np.linspace(T_START, T_END, raw.size)
        me_binned = np.interp(time_bins, x_old, raw)
    else:
        me_binned = np.full(time_bins.shape, np.nan)
```

iii. The AI had recorded the correct recipe in its own Step 1 notes — "frame times are assumed at 400 Hz, data are interpolated using `interp1(frameTimes-0.5-alignTimes(trix), ..., taxis)`" — and in Step 5 planned "Interpolate/align to trial time axis, bin to 75 ms". The `linspace` shortcut is nowhere justified; the AI's Step 5 planned sanity check "Verify one trial's motion-energy aligned trace against raw `me.data` interpolation and binning from `loadMotionEnergy.m` logic" was never executed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. All per-trial binned traces are concatenated; NaN bins are first replaced with the global median of the pooled traces, then the 50th percentile of that filled pool becomes the single per-session threshold. Each trial's finite bins are compared to it (`>= thr` → 1); non-finite bins → 0. Two classes, `['low', 'high']` — the required class 2 `no video` is never emitted, and the sessions without a sidecar file would silently come out as all-`low` rather than all-`no video`. The result is 0.500/0.500 in every one of the 21 sessions.

ii.
```python
if len(me_binned_all):
    all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all)))
                             for x in me_binned_all])
    thr = np.nanpercentile(all_me, 50)
    for i, meb in enumerate(me_binned_all):
        tmp = np.zeros(meb.shape, dtype=np.int64)
        finite = np.isfinite(meb)
        tmp[finite] = (meb[finite] >= thr).astype(np.int64)
        output_trials[i][5] = tmp
```

iii. Step 5 Key Decision 7 again. The AI did notice the suspiciously exact balance — step 41: "Motion energy is perfectly balanced because of the current thresholding implementation" — but treated it as expected rather than as evidence that the `no video` case was unrepresented. It also considered using the reference's own `me.moveThresh` (Step 1: "binary movement is created by thresholding motion energy (`me.move = me.data > me.moveThresh`)") but the Decoder Task's 50th-percentile rule took precedence, which is correct.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Not aligned to anything measured. As described at 9-b, `np.linspace(T_START, T_END, raw.size)` places the trace's first sample at exactly −1.5 s and its last at exactly +1.5 s, so alignment is to the trial *midpoint*, with a per-trial time-warp factor. Neither `frameTimes`, nor `goCue`, nor the video offset enters. This is a different (and inconsistent) time base from both the neural stream (trial-start relative) and the tongue/paw streams (go-cue relative, offset-uncorrected).

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. No justification appears anywhere in CONVERSION_NOTES.md or the trajectory — this is the one major stream the AI never revisited after the first implementation, and the `loadMotionEnergy.m` logic it had transcribed in Step 1 was never applied.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Broadly by **silent suppression**: defaults, bare `except: continue`, and fills.
- *Missing behavioural fields*: `get_trial_bool` returns `None`, and the corresponding validity filter is simply skipped (`if early is not None: valid &= ~early`), so a session missing `bp.early` keeps its early-lick trials with no warning.
- *Missing / malformed trajectory data*: `extract_hdf5_traj_velocity` wraps the whole per-camera body in `try/except Exception: continue`, and returns an all-NaN vector if nothing worked → all bins become class 0 (`low`), indistinguishable from genuinely slow bins.
- *Untracked frames*: NaN x/y are not detected at all; the resulting NaN bins are linearly interpolated over (7-b), fabricating values.
- *Missing motion-energy file or short trace*: all-NaN → class 0 (`low`).
- *Short `me.data`*: `while len(out) < n_trials: out.append(np.full(1, np.nan))` pads with NaN trials.
- *Unsupported `clu` layout / any build exception*: the whole session is dropped with a warning (`[warn] skipping ...`).
- *Non-positive frame intervals*: `dt[dt <= 0]` replaced by the median positive `dt`.

ii.
```python
        except Exception:
            continue
    if best is None:
        return np.full(time_bins.shape, np.nan, dtype=float)
```
```python
    while len(out) < n_trials:
        out.append(np.full(1, np.nan))
```
```python
        except Exception as e:
            print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True)
            continue
```
```python
dt[dt <= 0] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0
```

iii. Step 6 instructions asked to "Handle missing data appropriately (consult references, use sensible defaults, document)". The AI's documented rationale is confined to two cases: the recursive motion-energy unwrap (step 63) and the `clu` skip (step 68, "the pragmatic fix is to skip sessions whose `clu` structure does not match … This is defensible as part of curation/documentation"). The mapping of missing data onto the `low` class rather than a dedicated class is a consequence of dropping the third category (7-c/8-c/9-c) and is not discussed.

## 11-a. What are the most time-consuming steps of the code?

i. Per the conversion log, `build_session` dominates at 5.6–19.1 s per session while `load` is 0.00–0.01 s (because `load_mat_obj` only opens the HDF5 handle lazily — the real reads happen inside `build_session`). Within `build_session` the cost is overwhelmingly the per-trial × per-cluster Python loop that does `tm_arr[tr_arr == tr1]` — a full scan of every cluster's whole spike vector for every trial, i.e. O(n_trials × n_clusters × n_spikes) — plus the three redundant passes of `extract_hdf5_traj_velocity`, which re-reads and re-decodes `featNames`, `ts`, and `frameTimes` from HDF5 on every call. Total full-conversion time ≈ 4 minutes for 21 sessions.

ii.
```python
for tr in trial_idx:
    ...
    for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
        spikes = tm_arr[tr_arr == tr1]
        if spikes.size:
            mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The CONVERSION_NOTES.md Step 6 section is left as the unfilled template ("Code inefficiencies identified: [Note] / Code speedups added: [Note]"), and Step 7's "Run Time Estimates" table is empty. The AI's only timing commentary is in the trajectory: "build times range from ~8–18 seconds for richer sessions, so the full run may take several minutes but likely under 15 minutes" — i.e. it judged the runtime acceptable and did no optimisation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Two, both substantial:
- **Spike binning.** The nested `for tr in trial_idx: for ci in clusters:` with a boolean mask and a 1-D `np.histogram` per (trial, cluster) can be replaced by a single `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])` per cluster — which is exactly what the human reference does — collapsing the trial loop entirely.
- **Trajectory resampling.** The per-trial `np.interp` calls in `extract_hdf5_traj_velocity` are unavoidable in principle (frame counts differ per trial), but the HDF5 dereferencing, `featNames` decoding, and camera search inside them are trial-invariant and should be hoisted out of the loop; as written they are repeated for every trial and for each of the three passes.

Also `select_context_sessions` loops over every file opening it just to test one field, duplicating work `main` does again.

ii.
```python
for tr in trial_idx:
    ...
    for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
        spikes = tm_arr[tr_arr == tr1]
        if spikes.size:
            mat[ci], _ = np.histogram(spikes, bins=binedges)
```
```python
for i in range(traj_ds.shape[0]):
    ...
    names_arr = obj[traj['featNames'][trial_index0,0]][()]
    for j in range(names_arr.shape[1]):
        chars = obj[names_arr[0,j]][()]
        names.append(''.join(chr(int(c)) for c in chars.reshape(-1) if int(c) != 0))
```

iii. Not documented — Step 6's inefficiency notes were never filled in. The AI did flag the recomputation issue once in passing (step 42: "tongue and paw values are extracted inside the trial loop but not stored for later thresholding. Later code recomputes them when building thresholds, which is inefficient but not necessarily wrong") and proposed "storing extracted trajectory arrays once per trial to avoid recomputation", but never did it.

## 11-c. What processing does the code repeat multiple times?

i. Several clear repetitions:
1. **Every `.mat` file is opened twice** — once in `select_context_sessions` and once in `main`.
2. **`extract_hdf5_traj_velocity` is called three times per trial per feature** — once in the main trial loop (line 288/289, where the results are assigned to `tongue_vals`/`paw_vals` and then *never used*), once in the `tongue_all`/`paw_all` concatenation, and once in the final discretisation loop. Six full trajectory extractions per trial where two would do.
3. **The camera search and `featNames` string decoding** are redone from HDF5 on every one of those calls even though they are constant across trials.
4. **`np.nanmedian(np.concatenate(me_binned_all))` is recomputed inside a list comprehension**, so the full pooled motion-energy array is concatenated and its median taken once per trial — quadratic in trial count.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins) ...   # result discarded
paw_vals    = extract_hdf5_traj_velocity(obj, tr, [...], time_bins) ...   # result discarded
...
tongue_all = np.concatenate([... extract_hdf5_traj_velocity(obj, tr, [...], time_bins) ... for tr in trial_idx])
paw_all    = np.concatenate([... extract_hdf5_traj_velocity(obj, tr, [...], time_bins) ... for tr in trial_idx])
...
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
    paw  = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
```
```python
all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all)))
                         for x in me_binned_all])
```

iii. Partially acknowledged at step 42 (quoted at 11-b) and then dropped. The double file-open and the quadratic motion-energy median are not mentioned anywhere.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three kinds:
- **Computed and immediately thrown away**: `tongue_vals` and `paw_vals` in the main trial loop are assigned and never read — two full trajectory extractions per trial whose results are discarded outright. Likewise `left` is read only for the `right ^ left` mask, and `miss` only for `hit | miss`.
- **No-ops**: `np.nan_to_num(x, nan=np.nan)` in the `tongue_all`/`paw_all` comprehensions does nothing; `valid &= np.isin(autowater.astype(int), [0, 1])` is always all-True on a boolean array; `to_1d_numeric` is a `reshape(-1)`; `get_bp_field` is defined and never called.
- **Element-wise Python loop where a cast would do**: `get_trial_bool` converts each element in a Python `for` loop with a `try/except` per element.

Separately, `select_context_sessions`'s entire pass exists only to read one field, and the 5,512 unfiltered units mean roughly 3,500 low-rate/garbage-quality channels are binned, stored, and fed to the decoder although the paper's criteria would have removed them.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins) if isinstance(obj, h5py.File) else np.full(time_bins.shape, np.nan)
paw_vals    = extract_hdf5_traj_velocity(obj, tr, [...], time_bins) if isinstance(obj, h5py.File) else np.full(time_bins.shape, np.nan)
output_trials.append(np.vstack([ ... ]))     # tongue_vals / paw_vals never referenced again
```
```python
tongue_all = np.concatenate([np.nan_to_num(extract_hdf5_traj_velocity(...), nan=np.nan) for tr in trial_idx])
```
```python
out = np.zeros(arr.shape[0], dtype=bool)
for i, v in enumerate(arr):
    try:
        out[i] = bool(v)
    except Exception:
        out[i] = False
```

iii. Not documented. The unused `tongue_vals`/`paw_vals` are a leftover from the step-38 → step-43 → step-51 → step-54 sequence of patches, in which the discretisation was progressively moved out of the trial loop into separate passes without removing the original in-loop computation; CONVERSION_NOTES.md Step 6 was never completed and Step 10's code-review checks (which include "Compare your code to the reference code and methods") were never started.
