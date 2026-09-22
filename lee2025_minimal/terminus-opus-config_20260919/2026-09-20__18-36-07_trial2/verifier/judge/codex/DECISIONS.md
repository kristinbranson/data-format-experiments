# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the seven explicitly named animal files from `/app/data` with `joblib`. Each file is indexed by animal ID and supplies the complete `trace`, `position`, `envs`, and `blocked` arrays. It iterates every recording day for every animal, yielding 207 day-sessions.

ii.
```python
ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
...
for ai, animal in enumerate(ANIMALS):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    traces, positions = dat['trace'], dat['position']
    envs = [str(e[0]) for e in dat['envs']]
    blocked = dat['blocked']
```

iii. The trajectory says the joblib files have content identical to the MATLAB files, and inspection confirmed all seven animals, 207 sessions, 5,413 unique cells, and 69,744 registered cell-sessions, matching the paper.

## 1-b. How are the data split into subjects?

i. A fixed ordered `ANIMALS` list defines subjects. One source file is loaded per ID, and each retained session receives the list index `ai` in `subject_idx`.

ii.
```python
for ai, animal in enumerate(ANIMALS):
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
    subject_idx.append(ai)
...
'subjects': ANIMALS,
```

iii. The trajectory established that each joblib file is a per-animal dictionary. Explicit IDs also make subject ordering deterministic.

## 1-c. How are the data split into sessions?

i. Each recording day is treated as a session. The agent iterates the first dimension of the per-animal trace array and appends a session only if it retains neurons and at least two usable trials.

ii.
```python
n_days = traces.shape[0]
for day in range(n_days):
    ...
    if tr.shape[0] == 0:
        continue
    ...
    if len(sess_neural) < 2:
        continue
    neural_all.append(sess_neural)
```

iii. The trajectory identified the first dimension as recording days and noted that this yields the paper's 207 sessions. The two-trial requirement comes from the downstream format requirements.

## 1-d. How are the data split into trials?

i. After 3-frame pooling, each session is divided into consecutive nominal one-minute windows of 600 bins. The final partial window is considered. Only running bins within each window are retained, so stored trial arrays have variable numbers of samples representing observations from that minute.

ii.
```python
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN))
...
for start in range(0, nbins, BINS_PER_TRIAL):
    sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
    keep = run_b[sl]
    ...
    sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
```

iii. The agent chose one-minute windows as instructed, but retained only samples satisfying the paper's running-speed decoding filter. It verified that the conversion produced 8,163 trials.

## 1-e. How are trials filtered based on quality controls?

i. A nominal one-minute window is dropped if it contains fewer than 30 pooled running bins (three seconds). A whole session is dropped if fewer than two usable trials remain. Within retained trials, non-running bins are removed.

ii.
```python
keep = run_b[sl]
if keep.sum() < MIN_BINS_PER_TRIAL:
    continue
...
if len(sess_neural) < 2:
    continue
```

iii. The running filter follows the paper's within-session position decoder; the minimum-bin safeguard prevents nearly empty trials, and the two-trial safeguard satisfies explicit decoder validation requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each animal's `trace` array. These are binary calcium-transient rising-phase event traces, arranged in the loaded joblib data as cells by time within each day.

ii.
```python
traces, positions = dat['trace'], dat['position']
...
tr = np.asarray(traces[day])  # (n_cells, T)
```

iii. The agent inspected the source arrays and paper code and identified the traces as binarized events, with all-NaN rows on days when a cell was not registered.

## 2-b. How is the `neural` data processed?

i. The agent removes unregistered and low-activity cells, smooths every retained trace with a temporal Gaussian of sigma three original frames, average-pools non-overlapping groups of three frames, casts to `float32`, and then retains the running bins belonging to each trial.

ii.
```python
active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
tr = tr[active]
tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)
...
sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
```

iii. The trajectory traced these settings to `decode_position_within` and `fit_decoder`: sigma-3 smoothing and 3-frame pooling are the paper's decoder preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells not registered on the day are removed using NaN status at the first frame. Among registered cells, only cells with more than five events during frames where smoothed speed exceeds 5 cm/s are retained. Sessions with zero remaining cells are skipped.

ii.
```python
registered = ~np.isnan(tr[:, 0])
tr = tr[registered].astype(np.float32)
...
active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
tr = tr[active]
if tr.shape[0] == 0:
    continue
```

iii. Whole-day NaNs were empirically found to denote unregistered cells. The strict `> 5` event threshold during running matches the original decoding function's `cell_threshold=5` criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event onset. Trials are aligned to consecutive one-minute windows measured from the recording-session start. Neural, position, and speed use the same pooled time grid and identical trial slices/running masks.

ii.
```python
'temporal_alignment_event': (
    'start of the recording session; each session is cut into consecutive '
    '1-minute trials'),
'off_start': 0.0,
'off_end': 60.0,
```

iii. The agent recognized the recordings as continuous free foraging with no stimulus event and documented session start as the only meaningful alignment reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms. Original 30 Hz frames are Gaussian-smoothed and non-overlapping groups of three frames are mean-pooled; trailing frames not divisible by three are discarded.

ii.
```python
TEMPORAL_BIN = 3
...
def pool_mean(x, k):
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(x.shape[:-1] + (n // k, k)).mean(axis=-1)
...
'time_bin_size': 100.0,
```

iii. The agent selected 100 ms because the paper's `fit_decoder` uses `temporal_bin_size=3` at 30 Hz, rather than merely choosing it for storage or runtime.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived primarily from the per-day `blocked` field. The per-day `envs` name is also used to reconstruct the known geometry as a validation cross-check.

ii.
```python
envs = [str(e[0]) for e in dat['envs']]
blocked = dat['blocked']
...
env = envs[day]
inp = blocked_vector(env, blocked[day])
```

iii. The trajectory found that the stored blocked indices use the same `3*y_bin+x_bin` convention needed for outputs, while the repository's display matrix is vertically flipped.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. Blocked indices become a nine-element `float32` binary vector (`1` blocked, `0` open); `-1` means no blocked partitions. It is checked against the vertically flipped repository geometry and copied as a static vector for every retained trial.

ii.
```python
vec = np.zeros(9, dtype=np.float32)
if not (b.size == 1 and b[0] == -1):
    vec[b.astype(int)] = 1.0
from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
assert np.array_equal(vec, from_mat)
...
sess_input.append(inp.copy())
```

iii. The agent empirically checked occupancy and corrected an initially mistaken orientation interpretation. The assertion ensures the raw blocked field and named environment describe the same geometry.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each recording day's two-coordinate `position` array.

ii.
```python
traces, positions = dat['trace'], dat['position']
...
pos = np.asarray(positions[day], dtype=np.float64)  # (2, T)
```

iii. Inspection showed coordinates in centimeters over approximately 0–75 on both axes, with no position NaNs.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The x/y coordinates are mean-pooled over the same three-frame blocks as neural activity and then converted to one categorical grid index. Trial slicing and removal of non-running bins follow.

ii.
```python
pos_b = pool_mean(pos, TEMPORAL_BIN)
out_bins = position_to_bin(pos_b)[np.newaxis, :]
...
sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
```

iii. Pooling position on the same temporal grid as neural data preserves sample-level correspondence. The 3×3 categories implement the requested decoder target.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each 75 cm axis is divided into three 25 cm intervals using floor division conceptually (`floor(coordinate/25)`). Values are clipped to bins 0–2, and the category is `3*y_bin + x_bin`, producing labels 0–8.

ii.
```python
edge = ARENA_SIZE / N_SPACE_BINS
xb = np.clip(np.floor(pos_xy[0] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
yb = np.clip(np.floor(pos_xy[1] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
return 3 * yb + xb
```

iii. The agent verified this convention against the source `blocked` indices and actual occupancy, and added a check that occupancy of blocked categories remains below 2%.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural and position streams originate at the same frame rate, are pooled with the same factor, sliced using the same one-minute boundaries, and indexed by the exact same per-window running mask.

ii.
```python
neural = pool_mean(tr_s, TEMPORAL_BIN)
pos_b = pool_mean(pos, TEMPORAL_BIN)
...
sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
```

iii. The trajectory explicitly investigated temporal handling and used shared pooling/slicing to guarantee frame-for-frame alignment after rebinning and filtering.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN day rows are interpreted as unregistered cells and removed. Pooling drops a remainder shorter than three frames. Empty-neuron sessions, sparse trials, and sessions with fewer than two usable trials are skipped. Assertions detect contradictory geometry or substantial occupancy in blocked areas rather than silently accepting it.

ii.
```python
registered = ~np.isnan(tr[:, 0])
...
n = (x.shape[-1] // k) * k
...
if tr.shape[0] == 0: continue
if keep.sum() < MIN_BINS_PER_TRIAL: continue
if len(sess_neural) < 2: continue
```

iii. The trajectory established that NaNs encode absent registrations rather than sporadic corrupt measurements. The remaining safeguards enforce usable decoder data and expose convention errors early.

## 6-a. What are the most time-consuming steps of the code?

i. Loading large joblib arrays, Gaussian filtering all neural traces, and serializing the multi-gigabyte pickle dominate conversion. The later decoder validation/training is also expensive but is not part of conversion itself.

ii.
```python
dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
...
tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly waited while the seven large animal files were converted and reported a 3.3 GB output. It also observed a lengthy SVD initialization during decoder validation.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The animal/day loop is structurally necessary because sessions differ in cells and metadata. The Python loop that creates trial arrays could potentially be reduced for full equal-sized windows, but variable running masks, sparse-trial rejection, and a partial final window make complete vectorization awkward. `get_env_mat` dictionary creation could be hoisted but is negligible.

ii.
```python
for ai, animal in enumerate(ANIMALS):
    ...
    for day in range(n_days):
        ...
        for start in range(0, nbins, BINS_PER_TRIAL):
```

iii. The trajectory does not claim a missed vectorization. Most heavy numerical work (`diff`, norms, Gaussian filtering, pooling, event sums, and binning) is already vectorized in NumPy/SciPy.

## 6-c. What processing does the code repeat multiple times?

i. Per day it repeatedly computes geometry validation, speed, registration/activity masks, smoothing/pooling, and trial masks. This is appropriate because these quantities differ by day. The environment lookup matrix is reconstructed for each day and could be stored once globally.

ii.
```python
for day in range(n_days):
    inp = blocked_vector(env, blocked[day])
    ...
    speed = gaussian_filter1d(speed, sigma=V_FILT_SIGMA)
    ...
    tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
```

iii. The agent intended day-specific repetition to reproduce per-session curation. It did not identify any costly redundant recomputation after conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `get_env_mat` reconstruction and blocked-occupancy calculations are used only as assertions; their results are discarded. Smoothed speed is retained only long enough to make the running mask. Session metadata is informative but not used by the decoder model.

ii.
```python
from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
assert np.array_equal(vec, from_mat)
...
occ = np.bincount(out_bins[0], minlength=9) / nbins
assert occ[inp.astype(bool)].sum() < 0.02
```

iii. These are deliberate integrity checks prompted by the trajectory's discovery of a vertical-orientation convention mismatch. Their runtime is minor and they prevent a much more consequential labeling error.
