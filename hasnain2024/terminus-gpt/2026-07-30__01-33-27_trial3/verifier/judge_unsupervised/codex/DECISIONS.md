# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter only discovers `data_structure_*.mat` and matching `motionEnergy_*.mat` files under `data/Ephys_Behavior`, not the other dataset families. It treats filename `(subject, date)` pairs as sessions, then keeps only sessions whose `autowater` field contains both `0` and `1`.

ii.
```python
def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
    motion_map = {}
    for mf in sorted(base.glob('motionEnergy_*.mat')):
        ...
        motion_map[(m.group(1), m.group(2))] = mf
    sessions = []
    for df in data_files:
        ...
        sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
    return sessions

def main():
    sessions = discover_sessions()
    sessions = select_context_sessions(sessions)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent wrote: "Primary task subset: Use the two-context ephys sessions identified by presence of both DR/non-autowater and WC/autowater trial types." Later trajectory entries note that this broad rule still left a mismatch with the paper's reported 12-session / 6-mouse context subset.

## 1-b. How are the data split into subjects?

i. Subjects are split by parsing the subject ID from each filename, then taking the sorted unique subject names from the selected session records. `subject_idx` stores the index of each kept session's subject in that subject list.

ii.
```python
@dataclass
class SessionRecord:
    subject: str
    date: str
    data_file: Path
    motion_file: Optional[Path]

subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_idx[sess.subject])
```

iii. The justification was operational rather than neuroscientific: subject IDs were available in filenames and metadata, so the agent used them to populate `subjects`/`subject_idx`. The trajectory later noted that this was still too broad relative to the curated cohort in the paper/code.

## 1-c. How are the data split into sessions?

i. Each `data_structure_<subject>_<date>.mat` file is treated as one session. The converter iterates one `SessionRecord` per file, loads it, builds per-trial arrays for that session, and appends those arrays as one session entry in the output lists.

ii.
```python
for df in data_files:
    ...
    sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))

for sess in sessions:
    with timed(f'load {sess.data_file.name}'):
        obj = load_mat_obj(sess.data_file)
    ...
    data['neural'].append(neural)
    data['input'].append(inp)
    data['output'].append(out)
```

iii. This follows the raw file organization the agent documented in `CONVERSION_NOTES.md` Step 2, where the `.mat` files were treated as session-level containers.

## 1-d. How are the data split into trials?

i. Trials are split by using the length of the `goCue` event array as the session trial count, then subsetting that trial axis with a boolean `valid` mask. The kept trial indices become the per-session trial list.

ii.
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
...
valid = np.ones(n_trials, dtype=bool)
...
trial_idx = np.where(valid)[0]

for tr in trial_idx:
    ...
    input_trials.append(inp)
    output_trials.append(...)
    neural_trials.append(mat)
```

iii. The notes and trajectory treat `obj.bp.ev.goCue` as the central trial-alignment event, so the agent used its length as the trial axis for all streams.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept if it is not early, is either hit or miss, is not stimulation-enabled, has exactly one lick-direction flag set (`R ^ L`), and has an `autowater` value in `{0,1}`. Ignore trials are excluded indirectly because they are neither hit nor miss.

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
```

iii. `CONVERSION_NOTES.md` cites `getDefaultParams.m` for excluding early and stim trials, but trajectory Steps 88-89 show an explicit tradeoff: the agent temporarily tried hit-only filtering to match context-decoding conditions, then reverted to `hit | miss` because otherwise the required `outcome` output became all-correct.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the `obj.clu` structure, specifically each cluster's `trial` and `trialtm` arrays inside HDF5 sessions.

ii.
```python
if isinstance(obj, h5py.File):
    clu_root = obj['obj']['clu']
    clu = obj[clu_root[0,0]]
    ...
    for ci in range(n_clu):
        tr_ref = clu['trial'][ci,0]
        tm_ref = clu['trialtm'][ci,0]
        tr_arr = np.asarray(obj[tr_ref][()]).reshape(-1).astype(int)
        tm_arr = np.asarray(obj[tm_ref][()]).reshape(-1).astype(float)
        clu_trial.append(tr_arr)
        clu_trialtm.append(tm_arr)
```

iii. In Step 5 notes, the agent mapped "Spike times / trial-aligned spike data from `obj.clu` and trial structures" to `neural`, explicitly referencing `alignSpikes.m`, `getSeq.m`, and `removeLowFRClusters.m`.

## 2-b. How is the `neural` data processed?

i. For each kept trial and each cluster, the code selects the entries in `trialtm` whose `trial` equals that trial number, then bins those spike times directly into 75 ms bins from `-1.5` to `1.5` using `np.histogram`. No smoothing, normalization, time warping, or rebinning from a finer native grid is applied.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
mat = np.zeros((len(clu_trial), time_bins.size), dtype=np.float32)
tr1 = tr + 1
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The agent's justification was that the decoder scripts use `rez.binSize = 75` ms. The notes acknowledge that the reference code also has a finer `params.dt` and explicit alignment functions, but the implemented script simplifies that to direct 75 ms histograms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no per-neuron quality control beyond requiring the HDF5 `clu` object to expose `trial` and `trialtm`. The script does not apply the reference low-firing-rate threshold or any cluster-quality filter.

ii.
```python
if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
    raise ValueError('unsupported clu structure')
...
n_neurons = neural_trials[0].shape[0] if neural_trials else 1
brain_region_idx = np.zeros((n_neurons,), dtype=np.int64)
```

iii. In Step 5 notes the agent planned to "Apply low firing-rate filtering consistent with `removeLowFRClusters.m`," but trajectory and final code show that this was never implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script declares `goCue` as the alignment event, but for spikes it does not subtract the per-trial `goCue` time or otherwise realign `trialtm`. It simply bins the raw per-trial spike times into fixed edges centered on `-1.5:0.075:1.5`.

ii.
```python
ALIGN_EVENT = 'goCue'
...
go = get_event(obj, ALIGN_EVENT)
...
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
spikes = tm_arr[tr_arr == tr1]
if spikes.size:
    mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The notes say the agent intended to match `alignSpikes.m` and align to `goCue`, but the implemented code never uses `go` during spike binning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted trials all use 75 ms bins (`0.075` s) over a `-1.5` to `1.5` window, giving 41 time points. No separate temporal rebinning stage is applied; the code bins directly at that resolution.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The main justification recorded in Step 1 and Step 5 notes is that the reference decoder scripts explicitly set `rez.binSize = 75` ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not computed from a raw per-trial signal. Instead, it is a fixed vector of bin centers derived from the hard-coded window constants and implicitly interpreted as time relative to `goCue`.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
inp = time_bins[None, :].astype(np.float32)
```

iii. The notes describe this as "Trial time relative to go cue" repeated for each trial, motivated by the decoder-task requirement for a continuous time-from-go-cue input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The processing is just `np.arange` over the fixed start, stop, and bin size, followed by broadcasting that same 1 x T vector into every trial.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
...
for tr in trial_idx:
    inp = time_bins[None, :].astype(np.float32)
    input_trials.append(inp)
```

iii. The agent treated this as the simplest representation compatible with the decoder interface, rather than deriving it from a richer trial-state representation.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is aligned with neural data only by using the same fixed `time_bins` array that is also used to define the spike-count bins.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
...
inp = time_bins[None, :].astype(np.float32)
...
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
```

iii. The intended rationale was shared alignment to `goCue`, but because the spike times themselves are not shifted by `goCue`, this is only a shared nominal time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction comes from the trial-level boolean fields `bp.R` and `bp.L`.

ii.
```python
right = get_trial_bool(bp, 'R')
left = get_trial_bool(bp, 'L')
...
lick_dir = 1 if bool(right[tr]) else 0
```

iii. The notes explicitly map trial direction fields (`R`/`L`) to the lick-direction output and cite the reference condition definitions in `getDefaultParams.m`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code enforces exclusive direction labels during trial filtering, maps right to `1` and left to `0`, and then broadcasts that per-trial label across every time bin of the trial.

ii.
```python
if right is not None and left is not None:
    valid &= (right ^ left)
...
lick_dir = 1 if bool(right[tr]) else 0
...
np.full(time_bins.shape, lick_dir, dtype=np.int64)
```

iii. The agent justified this as using trial labels rather than lick timestamps for direction, matching how the reference code defines conditions.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the trial-level `bp.autowater` field.

ii.
```python
autowater = get_trial_bool(bp, 'autowater')
...
context = 0 if bool(autowater[tr]) else 1
```

iii. `CONVERSION_NOTES.md` Step 4 states that the context decoder scripts define context as non-autowater (DR/2AFC) versus autowater (WC), and the agent adopted that mapping.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code requires the session to contain both autowater states, filters trials to `autowater in {0,1}`, then maps autowater `True` to `WC = 0` and autowater `False` to `DR = 1`, broadcasting the label across time bins.

ii.
```python
def select_context_sessions(sessions: List[SessionRecord]) -> List[SessionRecord]:
    ...
    vals = set(np.unique(autowater.astype(int)).tolist())
    if vals == {0, 1}:
        keep.append(s)

...
if autowater is not None:
    valid &= np.isin(autowater.astype(int), [0, 1])
...
context = 0 if bool(autowater[tr]) else 1
```

iii. The notes say "Context coding: Convert raw `autowater`-style labels to task-required convention `WC=0`, `DR=1`."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial-level `bp.hit` and `bp.miss` fields.

ii.
```python
hit = get_trial_bool(bp, 'hit')
miss = get_trial_bool(bp, 'miss')
...
if hit is not None and miss is not None:
    valid &= (hit | miss)
...
outcome = 1 if bool(hit[tr]) else 0
```

iii. The trajectory shows the agent explicitly keeping miss trials so the decoder would have both incorrect and correct outcomes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Trials are limited to hits and misses, early and stimulation trials are excluded, then `hit` is mapped to `correct = 1` and any remaining non-hit trial (therefore miss) is mapped to `incorrect = 0`. The label is broadcast across the trial's time bins.

ii.
```python
if early is not None:
    valid &= ~early
if hit is not None and miss is not None:
    valid &= (hit | miss)
if stim_enable is not None:
    valid &= ~stim_enable
...
outcome = 1 if bool(hit[tr]) else 0
...
np.full(time_bins.shape, outcome, dtype=np.int64)
```

iii. Trajectory Step 88 records the explicit rationale: keeping miss trials was a compromise to satisfy the required outcome output, even though the reference context-decoding conditions are hit-only.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the HDF5 trajectory data: `obj.traj[*].featNames`, `obj.traj[*].ts`, `obj.traj[*].frameTimes`, plus the session `goCue` times.

ii.
```python
traj = obj[traj_ds[i,0]]
names_arr = obj[traj['featNames'][trial_index0,0]][()]
...
ts = np.asarray(obj[traj['ts'][trial_index0,0]][()]).astype(float)
ft = np.asarray(obj[traj['frameTimes'][trial_index0,0]][()]).reshape(-1).astype(float)
...
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
```

iii. Step 5 notes mapped "Tongue trajectories / DLC features from `obj.traj` or processed kin data" to the tongue-velocity output.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code searches trajectory feature names for the first tongue-like feature, takes its x/y coordinates, computes frame-to-frame speed `sqrt(dx^2 + dy^2) / dt`, shifts frame times by subtracting `goCue`, interpolates onto the common `time_bins`, and fills internal NaNs by nearest-index interpolation if any finite values exist.

ii.
```python
for cand in feature_candidates:
    if cand in names:
        feat_idx = names.index(cand)
        break
...
xy = ts[feat_idx, :2, :]
dx = np.diff(xy[0], prepend=xy[0,0])
dy = np.diff(xy[1], prepend=xy[1,0])
dt = np.diff(ft, prepend=ft[0])
...
speed = np.sqrt(dx*dx + dy*dy) / dt
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The notes say the intent was to match `loadKinData.m`/video processing, but the trajectory later notes that `loadKinData.m` was only a loader, so the agent implemented its own raw-marker speed extraction.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The script concatenates all per-bin tongue-speed values across the session, computes the 50th percentile over finite values, and sets each finite time bin to `1` if it is at or above that threshold and `0` otherwise. Non-finite bins remain `0`.

ii.
```python
tongue_all = np.concatenate([... extract_hdf5_traj_velocity(...) ...])
tongue_thr = np.nanpercentile(tongue_all, 50) if tongue_all.size and np.isfinite(tongue_all).any() else np.nan
...
tmp = np.zeros(tong.shape, dtype=np.int64)
finite = np.isfinite(tong)
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
output_trials[i][3] = tmp
```

iii. This follows the task requirement recorded in Step 5 notes: per-session median thresholding for continuous behavior outputs.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue speed is aligned by interpolating it onto the same `time_bins` array used by the neural matrices, after first expressing frame times as `frameTimes - goCue`.

ii.
```python
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
...
output_trials[i][3] = tmp
```

iii. The agent's notes say all modalities should be aligned to `goCue` on the decoder time axis. That intent is clear even though the spike stream itself was not shifted the same way.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the same HDF5 trajectory sources as tongue velocity, but using paw-related feature names.

ii.
```python
paw_vals = extract_hdf5_traj_velocity(
    obj, tr, ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins
)
```

iii. The Step 5 mapping says paw velocity comes from `obj.traj`/DLC features and should be converted to a time-varying binary output.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing is identical to tongue velocity: pick the first matching paw feature, compute x/y speed from consecutive frames, subtract `goCue` from frame times, interpolate onto the common trial bins, and then threshold later at the session median.

ii.
```python
paw_all = np.concatenate([np.nan_to_num(
    extract_hdf5_traj_velocity(obj, tr, ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins),
    nan=np.nan
) for tr in trial_idx]) if len(trial_idx) else np.array([])
```

iii. As with tongue velocity, the justification was pragmatic: the agent could inspect raw trajectory arrays but did not recover the original precomputed kinematic pipeline from the reference repository.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It is thresholded at the session-wide 50th percentile of finite paw-speed bins, with bins above or equal to the threshold labeled `1` and others labeled `0`.

ii.
```python
paw_thr = np.nanpercentile(paw_all, 50) if paw_all.size and np.isfinite(paw_all).any() else np.nan
...
tmp = np.zeros(paw.shape, dtype=np.int64)
finite = np.isfinite(paw)
tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
output_trials[i][4] = tmp
```

iii. This is the same per-session median-threshold rule the agent documented for all continuous outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw speed is placed on the same `time_bins` array as neural data using interpolation from `frameTimes - goCue`.

ii.
```python
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
...
output_trials[i][4] = tmp
```

iii. The stated rationale is the same go-cue-centered common time axis used for other outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the sidecar `motionEnergy_<subject>_<date>.mat` file, specifically the `me.data` field loaded by `load_motion_energy`.

ii.
```python
def load_motion_energy(path: Optional[Path], n_trials: int) -> Optional[List[np.ndarray]]:
    ...
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    me = x['me'].flat[0]
    data = np.asarray(unwrap_me_data(me)).reshape(-1)
```

iii. The Step 5 notes explicitly map `motionEnergy_*.mat` `me.data` to the motion-energy output.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each raw motion-energy vector is reshaped to 1D. For each kept trial, if the raw vector has more than one sample, it is interpolated from a synthetic evenly spaced grid over `[-1.5, 1.5]` onto `time_bins`. After all trials are processed, the session-wide median is computed on the concatenated binned values.

ii.
```python
if me_trials is not None and tr < len(me_trials):
    raw = me_trials[tr]
    if raw.size > 1:
        x_old = np.linspace(T_START, T_END, raw.size)
        me_binned = np.interp(time_bins, x_old, raw)
    else:
        me_binned = np.full(time_bins.shape, np.nan)
else:
    me_binned = np.full(time_bins.shape, np.nan)
me_binned_all.append(me_binned)
```

iii. The notes show the agent knew the reference `loadMotionEnergy.m` uses 400 Hz frame times, subtracts `0.5 + alignTime`, and fills edge NaNs with nearest values. The implemented code is a simplification rather than that reference procedure.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The code concatenates all session motion-energy bins, computes a session median, and labels finite bins as `1` if they are at or above the threshold, else `0`.

ii.
```python
all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all))) for x in me_binned_all])
thr = np.nanpercentile(all_me, 50)
for i, meb in enumerate(me_binned_all):
    tmp = np.zeros(meb.shape, dtype=np.int64)
    finite = np.isfinite(meb)
    tmp[finite] = (meb[finite] >= thr).astype(np.int64)
    output_trials[i][5] = tmp
```

iii. This again follows the task-mandated per-session 50th-percentile discretization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned only by being resampled onto the same nominal `time_bins` array used for neural data. The code does not use trial-specific frame times or `goCue` during this resampling step.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
...
output_trials[i][5] = tmp
```

iii. The agent intended common go-cue alignment, but the final code uses a fixed synthetic source axis instead of the reference per-trial video timing described in the notes.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly handles problems by falling back to NaNs or by skipping entire sessions. Missing trajectory information returns all-NaN vectors. Missing motion-energy files return `None`, and missing motion-energy trials are padded with length-1 NaN arrays. Unsupported cluster structures raise an exception and cause the whole session to be skipped. During discretization, NaN bins usually remain the default category `0`.

ii.
```python
except Exception:
    return np.full(time_bins.shape, np.nan, dtype=float)
...
if path is None or not path.exists():
    return None
...
while len(out) < n_trials:
    out.append(np.full(1, np.nan))
...
if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
    raise ValueError('unsupported clu structure')
...
except Exception as e:
    print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True)
    continue
```

iii. The notes say missing data should be handled "sensibly," but the trajectory shows this remained a pragmatic fallback strategy rather than a reference-matched correction procedure.

## 11-a. What are the most time-consuming steps of the code?

i. The slow path is `build_session`, especially the repeated per-trial trajectory extraction/interpolation and the nested per-trial, per-neuron spike histogramming. The conversion log shows most sessions spending several to ~19 seconds inside `build`.

ii.
```python
with timed(f'build {sess.data_file.name}'):
    neural, inp, out, bri = build_session(obj, sess)
...
for tr in trial_idx:
    ...
    for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
        ...
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The timing wrappers were added specifically so the agent could identify bottlenecks, and the run logs confirm `build_session` dominates runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-element boolean conversion in `get_trial_bool`, the per-neuron histogram loop, the repeated per-trial trajectory extraction calls, and the repeated trialwise threshold-application loops could all have been vectorized or cached more aggressively.

ii.
```python
for i, v in enumerate(arr):
    try:
        out[i] = bool(v)
    except Exception:
        out[i] = False

for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)

tongue_all = np.concatenate([ ... extract_hdf5_traj_velocity(...) ... for tr in trial_idx])
...
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(...)
```

iii. The trajectory repeatedly comments on efficiency problems in these same regions and treats them as likely speed bottlenecks.

## 11-c. What processing does the code repeat multiple times?

i. The biggest repeated work is trajectory extraction. Tongue and paw velocities are computed once inside the first trial loop and thrown away, recomputed again across all trials to get thresholds, and recomputed a third time to assign per-trial binary outputs.

ii.
```python
for tr in trial_idx:
    tongue_vals = extract_hdf5_traj_velocity(...)
    paw_vals = extract_hdf5_traj_velocity(...)
    ...

tongue_all = np.concatenate([ ... extract_hdf5_traj_velocity(...) ... for tr in trial_idx])
paw_all = np.concatenate([ ... extract_hdf5_traj_velocity(...) ... for tr in trial_idx])

for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(...)
    paw = extract_hdf5_traj_velocity(...)
```

iii. This repetition is visible directly in the final script and matches the trajectory's concern that trajectory processing was a major runtime cost.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `tongue_vals` and `paw_vals` are computed inside the first trial loop but never used. Continuous motion-energy traces are also constructed only to be thresholded immediately, and the script stores only the binarized output. In addition, all brain-region assignments are collapsed to a hard-coded all-zero ALM vector instead of using the available cluster metadata.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(...)
paw_vals = extract_hdf5_traj_velocity(...)
output_trials.append(np.vstack([
    ...
    np.zeros(time_bins.shape, dtype=np.int64),
    np.zeros(time_bins.shape, dtype=np.int64),
    np.zeros(time_bins.shape, dtype=np.int64),
]))
...
n_neurons = neural_trials[0].shape[0] if neural_trials else 1
brain_region_idx = np.zeros((n_neurons,), dtype=np.int64)
```

iii. The trajectory repeatedly flags the redundant trajectory extraction as a likely bottleneck, and the final code shows these first-pass trajectory values are indeed discarded.
