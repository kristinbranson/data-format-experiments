# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the seven per-animal MATLAB v7.3 files in `/app/data/*.mat` directly with `h5py`
(rather than the equivalent `joblib` copies used by `utils.load_dat`). The animal list is
hard-coded from the reference repo's `main.py`. Each file is opened once; the per-day metadata
(`envs`, `blocked`) is read up front by `read_meta`, and `trace` (T, n_cells) and `position`
(T, 2) are read **lazily one session at a time** by `read_session` (each is an HDF5 object
reference, `f['trace'][day, 0]`). All 7 animals x all days (207 sessions) are iterated. Trials do
not exist in the raw data; they are constructed later (see 1-d).

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

def open_animal(animal):
    """Open the MATLAB v7.3 file of one animal (lazy, HDF5)."""
    return h5py.File(os.path.join(DATA_DIR, f'{animal}.mat'), 'r')

def read_meta(f):
    """Return (envs, blocked) for every day of an opened animal file."""
    envs = [''.join(chr(c) for c in f[r][:].ravel()) for r in f['envs'][0]]
    blocked = [np.atleast_1d(f[r][:].ravel()).astype(int) for r in f['blocked'][0]]
    return envs, blocked

def read_session(f, day):
    """Read one session. Returns trace (T, n_cells) and position (T, 2), both 30 Hz."""
    trace = f[f['trace'][day, 0]][:]          # (T, n_cells) float64, NaN if unregistered
    position = f[f['position'][day, 0]][:]    # (T, 2) float64, cm in [0, 75]
    return trace, position

for ai, animal in enumerate(animals):
    f = open_animal(animal)
    envs, blocked = read_meta(f)
    n_days = len(envs)
    days = SAMPLE_SESSIONS[animal] if sample else range(n_days)
    for day in days:
        trace, position = read_session(f, day)
        ...
        del trace, position
    f.close()
```

iii. From CONVERSION_NOTES.md (Steps 1, 2, 6): the reference `load_dat` loads a whole animal
(joblib or `mat73.loadmat`) with the fields `trace`, `position`, `envs`, `blocked`. The AI notes
that "the joblib files load a whole animal (up to 17 GB expanded) at once; the HDF5 route reads one
session (~0.4 s) and keeps peak memory at ~1 GB. Verified byte-identical to the joblib contents."
It also verified that the loaded data reproduce the dataset's own stored derived products
(`maps['sampling']`, `maps['unsmoothed']`), and that the resulting counts match the paper
(7 subjects, 207 sessions, 5,413 unique neurons, 69,744 registered cell-sessions).

## 1-b. How are the data split into subjects?

i. One `.mat` file = one mouse. The 7 subject IDs are taken as the `subjects` list in the order
of the hard-coded `ANIMALS` list, and `subject_idx` for each session is `ANIMALS.index(animal)`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
'subjects': list(ANIMALS), 'subject_idx': [],
...
subj_idx = ANIMALS.index(animal)
...
data['subject_idx'].append(subj_idx)
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. From CONVERSION_NOTES.md Steps 1-3: the reference repo's animal list contains exactly these
7 IDs, and each per-animal file holds all of that animal's recording days. The paper states
"naive male (4) and female (3) mice", i.e. 7 subjects, which the AI confirmed against the data
(7 files, 5,413 unique neurons = 515+875+942+554+862+713+952, mean 773 cells/animal as quoted in
the paper).

## 1-c. How are the data split into sessions?

i. Each recording day within an animal file becomes one output session. The number of days is
taken from the length of the `envs` reference array (31 for six animals, 21 for QLAK-CA1-51 =
207 sessions). A session is dropped only if it yields fewer than 2 usable trials (this never
happens; all 207 are kept). Per-session bookkeeping (`animal`, `day`, `environment`,
`sequence = day // 10`, neuron counts, trial counts, frame counts) is stored in
`metadata['session_info']`.

ii.
```python
envs, blocked = read_meta(f)
n_days = len(envs)
days = SAMPLE_SESSIONS[animal] if sample else range(n_days)
for day in days:
    trace, position = read_session(f, day)
    ...
    sid = f'{animal}_day{day:02d}'
    if len(ntr) < 2:
        print(f'  SKIP {sid}: only {len(ntr)} usable trials', flush=True)
        continue
    data['neural'].append(ntr)
    data['input'].append(inp)
    data['output'].append(outp)
    data['subject_idx'].append(subj_idx)
    data['brain_region_idx'].append(np.zeros(int(keep.sum()), dtype=np.int64))
    session_info.append({'session_id': sid, 'animal': animal, 'day': day,
                         'environment': envs[day], 'sequence': day // 10, ...})
```

iii. CONVERSION_NOTES.md Steps 2-3: "All sessions were 40 min, and one session was recorded per
day"; days per animal are 31,31,31,21,31,31,31 = **207 total**, matching the paper's
"5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps". Each day
has its own geometry, its own cell registration and its own `blocked` vector, so the day is the
natural session unit. No session or animal is excluded ("all pass the quality checks of the
paper").

## 1-d. How are the data split into trials?

i. The experiment is continuous 40-min free exploration, so trials are created artificially, as the
Decoder Task requires: consecutive non-overlapping 60-s blocks. Because the neural/behavioural
streams are first re-binned to 100 ms (see 2-e), a trial is 600 bins. The final partial block is
**kept** if it is at least 50% of a trial (>= 300 bins = 30 s); otherwise it is dropped. In
practice a 71,866-frame session gives 39 full trials plus a 555-bin (55.5 s) remainder, so most
sessions produce 40 trials (8,147 trials total, mean 39.4/session). Within each trial only the
bins in which the mouse is running are retained (see 1-e/2-c), so trials have unequal numbers of
timepoints (mean 315, min 30, max 561 bins) while the *bin size* stays constant at 100 ms.

ii.
```python
TRIAL_SEC = 60.0           # 1 minute trials (decoder task)
MIN_TRIAL_FRAC = 0.5       # keep the final partial block if >= 50% of a trial
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN_FRAMES))   # 600
...
    neural_trials, input_trials, output_trials = [], [], []
    start = 0
    while start < n_bins:
        stop = min(start + BINS_PER_TRIAL, n_bins)
        if (stop - start) < MIN_TRIAL_FRAC * BINS_PER_TRIAL:
            break                      # drop a too-short remainder
        sel = np.where(moving_binned[start:stop])[0] + start
        if sel.size >= MIN_BINS_PER_TRIAL:
            neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
            input_trials.append(input_vec.copy())
            output_trials.append(out_class[sel][None, :].copy())
            trial_slices.append((start, stop, sel))
        start = stop
```

iii. CONVERSION_NOTES.md Step 5, decision 3: "**Trials = consecutive 60 s blocks** of each session
(Decoder Task). The final partial block is kept only if it is at least 30 s long (all sessions have
a ~55 s remainder, so it is kept); this avoids discarding ~2.5% of the data. ~40 trials/session,
well above the minimum of 2." The AI notes there are no native trials ("no trials in the original
experiment (40-min free exploration)"), so the 1-min segmentation comes purely from the task
instructions.

## 1-e. How are trials filtered based on quality controls?

i. Two trial/session-level filters, both consequences of the reference speed criterion:
(1) a trial is dropped if fewer than 30 of its bins (3 s) have smoothed speed > 5 cm/s
(`MIN_BINS_PER_TRIAL`); (2) a session is dropped if it ends up with fewer than 2 trials (needed by
the decoder harness to have train and validation trials). No trial is dropped for any other reason;
no session and no animal is excluded. Empirically every session keeps >= 22 trials (median 40) and
all 207 sessions survive.

ii.
```python
MIN_BINS_PER_TRIAL = 30    # >= 3 s of running data required to keep a trial
...
        sel = np.where(moving_binned[start:stop])[0] + start
        if sel.size >= MIN_BINS_PER_TRIAL:
            neural_trials.append(...)
...
            sid = f'{animal}_day{day:02d}'
            if len(ntr) < 2:
                print(f'  SKIP {sid}: only {len(ntr)} usable trials', flush=True)
                continue
```

iii. CONVERSION_NOTES.md Steps 3, 5 and 10: the reference paper/code define no trials and
therefore no trial QC; the only temporal curation in the reference decoder is the >5 cm/s speed
criterion of `utils.decode_position_within`. The AI added the minimum-running-time rule so that a
"trial" that contains almost no locomotion (and hence almost no valid samples) does not enter the
dataset, and the >= 2 trials/session rule because the target format states "There needs to be at
least two trials within each session in order to evaluate the decoder performance". Step 10's
edge-case table reports "Sessions with < 2 trials: 0 (minimum is 22 trials, median 40)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Only the `trace` field: the authors' binarised rising-phase calcium event vector, shape
(T, n_cells) at 30 Hz, with an entire column NaN when a cell was not registered on that day.
`position` is used indirectly (to build the speed mask used for cell curation and bin selection),
but the neural values themselves come only from `trace`.

ii.
```python
trace = f[f['trace'][day, 0]][:]          # (T, n_cells) float64, NaN if unregistered
...
    tr = trace[:, keep_cells]
```

iii. CONVERSION_NOTES.md Steps 1-3: "Neural signal to use = the **binarized rising-phase vector**
(`trace`), treated as firing rate (per Methods). No dF/F computation needed - already done by the
authors", quoting the paper: "This binary vector was treated as the firing rate in all further
analyses". The AI explicitly checked that `trace` is 0/1 with all-or-none NaN columns and that it
reproduces the dataset's stored `maps['unsmoothed']` rate maps.

## 2-b. How is the `neural` data processed?

i. Four steps, copied from the reference decoding pipeline: (1) cells curated (see 2-c);
(2) `gaussian_filter1d(sigma = 3 frames)` along time; (3) non-overlapping 3-frame average pooling
(the numpy equivalent of `torch.nn.AvgPool1d(kernel_size=3, stride=3)` used by `utils.fit_decoder`)
giving 100 ms bins; (4) multiplication by 30 to express the values in events/s, cast to float32,
and transposed to (n_neurons, n_timepoints). Steps 2-3 are applied to the whole session before
trials are cut and before non-running bins are removed.

ii.
```python
TEMPORAL_BIN_FRAMES = 3    # AvgPool1d(kernel_size=3) in fit_decoder  -> 100 ms
TRACE_SMOOTH_SIGMA = 3     # gaussian_filter1d(sigma=3) in fit_decoder

def pool_mean(x, k=TEMPORAL_BIN_FRAMES):
    """Average non-overlapping blocks of k samples along axis 0 (== AvgPool1d)."""
    n = (x.shape[0] // k) * k
    x = x[:n]
    return x.reshape((n // k, k) + x.shape[1:]).mean(axis=1)
...
    tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SMOOTH_SIGMA, axis=0)
    neural = pool_mean(tr_smooth).astype(np.float32) * FPS     # -> events / s
...
            neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
```

iii. CONVERSION_NOTES.md Step 5, decisions 1-2 and Step 10 Check 3: "`fit_decoder`:
`gaussian_filter1d(traces, sigma=3, axis=0)` then `AvgPool1d(kernel=3, stride=3)`" -> "Use 100 ms
bins ... exactly the reference decoder's `fit_decoder`/`test_decoder` preprocessing. It also removes
the 'neural data is all 0/1' error raised by the verification harness." On the x30 factor: "Neural
data are multiplied by 30 to be in events/s. A pure rescaling; it does not change what a linear
decoder can represent but keeps the values O(1) relative to the L1 penalty." The AI also validated
the whole neural pipeline end-to-end by reproducing the paper's Gaussian-Naive-Bayes 15x15
decoding error from the converted data (13.71 cm vs the stored reference 12.99 cm, ~5%).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two curation steps, both taken from `utils.decode_position_within`:
- **Cells**: a cell is kept only if it is registered on that day (`~isnan(trace[0, :])`) **and**
  has more than 5 events during the running periods (`cell_threshold = 5`). This keeps 68,862 of
  the 69,744 registered cell-sessions (98.7%); mean 332.7 neurons/session (min 112, max 562).
- **Timepoints**: only bins whose mean smoothed speed exceeds 5 cm/s are kept (`v_thresh = 5`,
  speed smoothed with `gaussian_filter1d(sigma = 5 frames)`), which retains ~60% of the bins.
  The speed mask is applied to neural, input and output identically via the index array `sel`.
No place-cell / split-half-reliability selection is applied.

ii.
```python
V_FILT_SIGMA = 5           # v_filt_size in decode_position_within (frames)
V_THRESH = 5.0             # v_thresh in decode_position_within (cm/s)
CELL_THRESHOLD = 5         # cell_threshold in decode_position_within (events)

def compute_speed(position):
    """Smoothed speed in cm/s, exactly as in utils.decode_position_within."""
    speed = np.zeros(position.shape[0])
    speed[1:] = gaussian_filter1d(
        np.linalg.norm(np.diff(position, axis=0) * FPS, axis=1),
        axis=0, sigma=V_FILT_SIGMA)
    return speed
...
    speed = compute_speed(position)
    moving = speed > V_THRESH

    # ---- 2. neuron curation (reference: registered AND > 5 events while moving)
    registered = ~np.isnan(trace[0, :])
    events_moving = np.nansum(trace[moving, :], axis=0)
    keep_cells = registered & (events_moving > CELL_THRESHOLD)
...
    moving_binned = speed_binned > V_THRESH
    sel = np.where(moving_binned[start:stop])[0] + start
```

iii. CONVERSION_NOTES.md Step 5, decisions 4-5: "**Speed filter, >5 cm/s**: timepoints where the
smoothed speed does not exceed 5 cm/s are dropped, exactly as `decode_position_within` does for its
Bayesian position decoder (`v_thresh=5`, `v_filt_size=5` frames). Position coding in CA1 is only
well defined during locomotion, and immobility periods are dominated by replay/SWR activity that is
not about current position." "**Neuron curation** = the reference decoder's rule: keep cells
registered on that day (non-NaN) **and** with >5 events during the movement periods
(`cell_threshold=5`). No place-cell selection - the paper explicitly includes all cells" ("motivated
the inclusion of all cells in subsequent analyses"). Step 9 explains the 69,744 -> 68,862 drop:
"882 cell-sessions (1.3%) are registered on a day but produce <= 5 calcium events while the animal
is running ... Keeping them would add rows that are exactly zero for the whole session."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus/behavioural event to align to: free exploration in a continuous 40-min
session. The alignment "event" recorded in the metadata is the start of each 60-s block, with
`off_start = 0.0` and `off_end = 60.0` s. Neural and behavioural streams are used with a common
frame index and a common bin index with zero lag (they were acquired by the same DAQ at 30 Hz);
within a trial the same `sel` index array selects the running bins of neural, input and output, so
they remain sample-for-sample aligned.

ii.
```python
'temporal_alignment_event':
    'Start of each 60 s trial, measured from the start of the recording '
    'session (free exploration; the task has no discrete events). Behaviour '
    'and calcium were acquired simultaneously at 30 Hz by the same DAQ and '
    'are frame-aligned in the source data.',
'off_start': 0.0,
'off_end': TRIAL_SEC,
...
    pos_binned = pool_mean(position)
    speed_binned = pool_mean(speed[:, None])[:, 0]
    n_bins = neural.shape[0]
    assert pos_binned.shape[0] == n_bins == speed_binned.shape[0]
...
        sel = np.where(moving_binned[start:stop])[0] + start
        neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
        output_trials.append(out_class[sel][None, :].copy())
```

iii. CONVERSION_NOTES.md Step 3/Step 10: "Temporal alignment: behaviour and calcium are acquired by
the same DAQ at 30 Hz and were already frame-aligned by the authors (confirmed by sanity check 2 in
Step 2)" - that sanity check recomputed `maps['unsmoothed']` from `trace` + `position` for every
registered cell and matched the stored maps to within one frame, "-> `trace` and `position` are
frame-aligned with **zero** lag". The `--show-processing` figures additionally overlay the 30 Hz
and 100 ms position traces and the class labels to show there is no shift.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100 ms (`time_bin_size = 100.0`). Yes: the native 30 Hz (33.33 ms) data are rebinned by a factor
of 3 by non-overlapping average pooling, after Gaussian smoothing of the traces (sigma = 3 frames).
Position and speed are pooled on exactly the same grid with the same `pool_mean`, so all streams
share one bin edge set. The trailing at most 2 frames of a session (67 ms of ~40 min) are dropped by
the pooling.

ii.
```python
FPS = 30.0                 # acquisition rate of both streams (methods)
TEMPORAL_BIN_FRAMES = 3    # AvgPool1d(kernel_size=3) in fit_decoder  -> 100 ms
BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS      # 100 ms
...
    neural = pool_mean(tr_smooth).astype(np.float32) * FPS     # -> events / s
    pos_binned = pool_mean(position)
    speed_binned = pool_mean(speed[:, None])[:, 0]
...
    'time_bin_size': BIN_MS,
```

iii. CONVERSION_NOTES.md Steps 3-5: "Temporal binning (reference decoder): `AvgPool1d(kernel=3,
stride=3)` on 30 Hz data = **100 ms bins**, traces first smoothed with
`gaussian_filter1d(sigma=3 frames = 100 ms)`" and Step 4's discrepancy table resolution: "Use
100 ms bins". Decision 2 adds that binning also fixes the harness complaint that raw `trace` is
all 0/1. `pool_mean` is documented as "vectorised equivalent of `torch.nn.AvgPool1d(kernel_size=3,
stride=3)` used by `utils.fit_decoder`".

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Primarily from the per-day geometry name in the `envs` field, which is mapped to the 3x3 binary
geometry matrix of `utils.get_env_mat` and then to a 9-dim "blocked" vector. The raw `blocked`
field (indices 0-8 of blocked partitions, `-1` if none) is also read for every day and is used as
an assertion: the conversion aborts if the vector derived from `envs` disagrees with `blocked`.
So the input is derived from `envs` and validated against `blocked` (I re-ran this check
independently: 0 mismatches over all 207 sessions).

ii.
```python
ENV_MATS = {
    'square':    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    'o':         [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    ...
}

def env_blocked_vector(env_name):
    """9-dim binary vector, 1 = partition blocked, indexed 3*ybin + xbin. ..."""
    m = np.array(ENV_MATS[env_name], dtype=float)
    return (np.flipud(m).ravel() == 0).astype(np.float32)
...
    envs, blocked = read_meta(f)
...
    # consistency check: the geometry name and the `blocked` field must agree
    expected = np.where(env_blocked_vector(envs[day]))[0]
    got = np.sort(blocked[day][blocked[day] >= 0])
    assert np.array_equal(expected, got), \
        f'{animal} day {day}: blocked {got} != {expected} for env {envs[day]}'
```

iii. CONVERSION_NOTES.md Steps 1, 2, 4: `get_env_mat(env)` is the "**Binary 3x3 matrix of
environment geometry** (1 = open partition, 0 = blocked)" and `clean_rate_maps` shows the
orientation used to mask maps (`np.fliplr(get_env_mat(env).T)`), from which the AI derived
`flipud(env_mat)` as the `blocked`-index ordering. Step 4: "`blocked` indices == flat indices of
the zeros of `flipud(get_env_mat(env))` for **all 207 sessions** (0 mismatches)", cross-checked
behaviourally ("occupancy inside `blocked` partitions is ~0 (<=0.6%), whereas the transposed
hypothesis gives up to 71%"). Using `envs` keeps the geometry name available for
`session_info`/plots while the assertion guarantees identity with `blocked`.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3x3 geometry matrix is flipped vertically and flattened, and the zeros (omitted partitions)
become 1s: a 9-dim float32 binary vector indexed `3*ybin + xbin`, i.e. the same indexing as the
dataset's `blocked` field and the same as the output classes. It is 0 everywhere for the `square`
geometry. The vector is constant within a session and is stored as a static `(9,)` array for every
trial of that session (a copy per trial). `input_names` are `blocked_partition_0..8`.

ii.
```python
def env_blocked_vector(env_name):
    m = np.array(ENV_MATS[env_name], dtype=float)
    return (np.flipud(m).ravel() == 0).astype(np.float32)
...
    input_vec = env_blocked_vector(env_name)
...
            input_trials.append(input_vec.copy())
...
'input_names': [f'blocked_partition_{i}' for i in range(9)],
```

iii. CONVERSION_NOTES.md Step 5: "`blocked[day]` (indices 0-8, -1 = none) -> `input[session][trial]`
(9,) float32, binary vector, 1 = partition blocked ... static per trial, as required by the Decoder
Task"; decision 7: "**Input = 9-dim binary 'blocked' vector**, static per trial, exactly as
specified by the Decoder Task". The Step 7 sanity check notes that the classes with zero occupancy
in a `t`-geometry session are exactly the partitions flagged in the input vector - "a strong
cross-check that the input and output share one indexing".

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The `position` field only: DeepLabCut head-tracking (T, 2) in cm, range [0, 75] in both axes,
at 30 Hz, column 0 = x, column 1 = y. (`envs`/`blocked` are used only for the out-of-geometry
cleaning step, and `position` is also used for the speed mask.)

ii.
```python
position = f[f['position'][day, 0]][:]    # (T, 2) float64, cm in [0, 75]
...
    pos_binned = pool_mean(position)
    out_class = position_to_class(pos_binned, blocked_vec=input_vec)
```

iii. CONVERSION_NOTES.md Steps 2-3: "`position` | (n_days, 2, T) float64 | x,y in **cm**, range
[0, 75], no NaNs anywhere (checked all animals)"; "Position data were generated from tracking the
head with DeepLabCut". The AI verified the x/y convention and the absolute 0-75 cm scaling by
reproducing the dataset's own occupancy map `maps['sampling']` to one-frame precision.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. (1) x and y are averaged within each 100 ms bin with the same `pool_mean` used for the neural
data (this mirrors `fit_decoder`, which pools the behavioural stream with the same `AvgPool1d`);
(2) each coordinate is divided by the 25 cm partition size and truncated to an integer bin, clipped
to [0, 2]; (3) the class label is `3*ybin + xbin` (0-8); (4) samples that fall inside a partition
that the session's geometry declares blocked (127 of 2,573,752 samples = 0.005%, DeepLabCut noise
near partition walls) are snapped to the nearest open partition centre; (5) stored as a
`(1, n_timepoints)` int64 array per trial, subset by the running mask. `output_names =
['position_3x3']` and `output_values` names all 9 partitions with their (x, y) indices.

ii.
```python
def position_to_class(position, blocked_vec=None):
    """3x3 partition index for each timepoint: 3*ybin + xbin (`blocked` convention). ..."""
    part = ARENA_SIZE / N_PART
    bins = np.clip((position / part).astype(int), 0, N_PART - 1)
    cls = (N_PART * bins[:, 1] + bins[:, 0]).astype(np.int64)
    if blocked_vec is not None and blocked_vec.any():
        blocked_ids = np.where(blocked_vec > 0)[0]
        bad = np.isin(cls, blocked_ids)
        if bad.any():
            open_ids = np.where(blocked_vec == 0)[0]
            centres = np.stack([(open_ids % N_PART) * part + part / 2,
                                (open_ids // N_PART) * part + part / 2], axis=1)
            d = np.linalg.norm(position[bad][:, None, :] - centres[None], axis=2)
            cls[bad] = open_ids[np.argmin(d, axis=1)]
    return cls
...
    pos_binned = pool_mean(position)
    out_class = position_to_class(pos_binned, blocked_vec=input_vec)
...
            output_trials.append(out_class[sel][None, :].copy())
```

iii. CONVERSION_NOTES.md Step 5: "`position[day]` (2, T) in cm -> `output` ... average x,y within
each 100 ms bin, then bin = `floor(pos/25)` clipped to [0,2]; class = 3*ybin + xbin", justified by
"`fit_decoder` (pools behaviour the same way), `decode_position_within`, `blocked` convention" and
"3x3 = 9 classes as required by the Decoder Task". The snapping was added in Step 10, iteration 1:
"This is exactly the situation the reference `decode_position_within` cleans up by snapping
positions to the nearest bin that belongs to the environment (`true_bins[np.argmin(actual_norms)]`)
... sessions where #classes != 9-#blocked went from 4 to **0** ... The effect on decoder accuracy
was negligible (0.6617 both before and after)." (Note: the Step 5 "Key Decisions" list still says
such samples "are kept as-is rather than being re-assigned"; that sentence was left stale when the
snapping was introduced in Step 10, which is documented.)

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Absolute, equal-width thresholds tied to the physical arena rather than to the observed range:
partition size = 75 cm / 3 = 25 cm, so the bin edges are 25 cm and 50 cm; `int(pos / 25)` is
truncated toward zero and clipped to [0, 2] so that the single boundary value x = 75.0 cm falls in
bin 2. The 9 classes are `3*ybin + xbin`, matching the `blocked` partition indexing. Resulting
global class distribution: [0.095, 0.108, 0.114, 0.088, 0.070, 0.087, 0.109, 0.158, 0.171].

ii.
```python
ARENA_SIZE = 75.0          # cm, side of the square arena (methods)
N_PART = 3                 # 3 x 3 partition grid (decoder task)
...
    part = ARENA_SIZE / N_PART
    bins = np.clip((position / part).astype(int), 0, N_PART - 1)
    cls = (N_PART * bins[:, 1] + bins[:, 0]).astype(np.int64)
...
'output_values': [[f'partition_{i}_(x{i % 3},y{i // 3})' for i in range(9)]],
'partition_size_cm': ARENA_SIZE / N_PART,
'output_encoding': 'class = 3 * ybin + xbin with xbin, ybin in {0,1,2} ...'
```

iii. CONVERSION_NOTES.md Step 4 (discrepancy table) and Step 10 Check 3: the reference
`get_rate_maps` bins by the per-session maximum and `decode_position_within` by the maximum over
all days, but "Position is normalised to exactly [0,75] cm in x and y for **every** animal and day,
so both give the identical 5 cm bin"; the AI therefore uses the absolute 75/n_bins bin size and
"Verified: reproduces `maps['sampling']` and `maps['unsmoothed']` exactly". Decision 6: "**Output =
one categorical variable with 9 values** (3x3 partitions), time-varying, using the same partition
indexing as the dataset's `blocked` field (index = 3*ybin + xbin)". The clip is needed because a
coordinate can equal exactly 75.0.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Frame-for-frame, then bin-for-bin. `position`, `speed` and `trace` all come from the same frame
index; the identical `pool_mean(k=3)` is applied to all three so they share bin edges (an assertion
checks the bin counts are equal); trials are cut on a single bin index and the identical running-bin
index array `sel` is used to subset neural and output, so `neural[t]` and `output[t]` refer to the
same 100 ms bin. The input is static and broadcast over the same bins.

ii.
```python
    pos_binned = pool_mean(position)
    speed_binned = pool_mean(speed[:, None])[:, 0]
    n_bins = neural.shape[0]
    assert pos_binned.shape[0] == n_bins == speed_binned.shape[0]
...
        sel = np.where(moving_binned[start:stop])[0] + start
        if sel.size >= MIN_BINS_PER_TRIAL:
            neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
            input_trials.append(input_vec.copy())
            output_trials.append(out_class[sel][None, :].copy())
```

iii. CONVERSION_NOTES.md Step 10 Check 3: "(c) temporal alignment - reference: frames of `trace` and
`position` are used with the same index, no shift; mine: identical; verified by reproducing
`maps['unsmoothed']`". Step 12: "Temporal alignment verified two ways: by reproducing the dataset's
own rate maps from `trace` + `position` (exact), and visually in panels 4-5 of the
`--show-processing` figures". The harness verification also confirms the neural, input and output
time dimensions are equal for every trial (no warnings).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Unregistered cells** (whole `trace` column NaN): removed by `registered = ~np.isnan(trace[0, :])`
  (verified that NaN is all-or-none per cell/day) and, redundantly, by `np.nansum` + the >5-event
  threshold.
- **Near-silent cells**: removed by the >5-events-while-running criterion (882 cell-sessions).
- **Tracking noise inside blocked partitions**: 127/2,573,752 samples snapped to the nearest open
  partition; verified afterwards that every session's label set is exactly
  `9 - #blocked_partitions` classes.
- **Boundary coordinate 75.0 cm**: clipped into bin 2.
- **Frames that do not fill a 100 ms bin**: the trailing <= 2 frames of a session are dropped by
  `pool_mean`.
- **Session remainder shorter than a trial**: kept if >= 30 s, otherwise dropped.
- **Too little locomotion**: trials with < 30 running bins dropped; sessions with < 2 trials skipped.
- **Metadata inconsistency**: an assertion aborts the conversion if `envs` and `blocked` disagree.
No NaN/Inf remains in the output (harness reports no warnings).

ii.
```python
    registered = ~np.isnan(trace[0, :])
    events_moving = np.nansum(trace[moving, :], axis=0)
    keep_cells = registered & (events_moving > CELL_THRESHOLD)
    tr = trace[:, keep_cells]
    if tr.size == 0:
        return [], [], [], keep_cells
...
    bins = np.clip((position / part).astype(int), 0, N_PART - 1)
...
            cls[bad] = open_ids[np.argmin(d, axis=1)]
...
def pool_mean(x, k=TEMPORAL_BIN_FRAMES):
    n = (x.shape[0] // k) * k
    x = x[:n]
    return x.reshape((n // k, k) + x.shape[1:]).mean(axis=1)
...
    assert np.array_equal(expected, got), \
        f'{animal} day {day}: blocked {got} != {expected} for env {envs[day]}'
```

iii. CONVERSION_NOTES.md Steps 2, 5, 10: NaN structure was characterised first ("If a cell is not
registered on a day, the whole (cell, day) row is NaN (verified: NaN count per cell-day is either 0
or T)"; "position ... no NaNs anywhere (checked all animals)"). Step 10, Check 5's edge-case table
enumerates the remaining cases ("Trailing frames: `pool_mean` drops at most 2 frames (67 ms) of a
40-min session"; "Final partial trial ... kept because it exceeds the 50% threshold"; "All-zero or
non-finite trials: 0"). The out-of-geometry samples are explained as "DeepLabCut tracking noise a
few centimetres inside a partition wall", cleaned by analogy with `decode_position_within`.

## 6-a. What are the most time-consuming steps of the code?

i. The AI instrumented the script (per-session `read`/`proc` timings, plus an extrapolated
full-dataset estimate) and reports two roughly equal costs: reading each session's `trace`
(72,000 x ~340 float64 = ~200 MB) from the HDF5 file (~0.43 s/session) and processing it
(~0.46 s/session), the latter dominated by `gaussian_filter1d` over the full 72,000-frame session
for every kept cell. Pickling the 3.44 GB output is the other notable cost. Full conversion:
245 s (4.1 min) for 207 sessions, well inside the 15-min budget, so no parallelism was added.

ii.
```python
            t0 = time.time()
            trace, position = read_session(f, day)
            t_read = time.time() - t0
...
            t1 = time.time()
            ntr, inp, outp, keep = process_session(trace, position, envs[day],
                                                   diagnostics=diag)
            t_proc = time.time() - t1
...
    if timings:
        tr_ = np.array(timings)
        print(f'Mean per session: read {tr_[:, 0].mean():.2f}s, '
              f'process {tr_[:, 1].mean():.2f}s')
        print(f'Estimated full-dataset time: '
              f'{tr_.sum(axis=1).mean() * 207 / 60:.1f} min')
```

iii. CONVERSION_NOTES.md Steps 6-7 and 9: "Loading whole-animal joblib files (6-40 s each, >10 GB
RAM) -> replaced by per-session HDF5 reads"; the Step 7 timing table gives "read session from .mat
0.43 s -> 1.5 min for 207" and "process session 0.46 s -> 1.6 min for 207", total ~3.1 min, with the
realistic estimate raised to 4-6 min because the sample sessions have fewer cells than average;
Step 9 records the actual 245 s, "close to the 3-4 min estimated in Step 7".

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Essentially none of substance. The only Python loops are: the per-animal / per-session loop
(inherently sequential I/O, 207 iterations of ~0.9 s), the `while` loop that cuts ~40 trials per
session (a slice and a boolean `np.where` per trial), and the list comprehensions in `read_meta`
over the ~31 small metadata references. The frame-level loops of the reference code
(`get_rate_maps`'s `for t, (x, y) in enumerate(position_binned)` and `fit_decoder`'s per-sample
one-hot loop) were deliberately replaced by vectorised numpy: `pool_mean` is a reshape-mean instead
of `torch.nn.AvgPool1d`, discretisation and the out-of-geometry snapping are fully vectorised
(`np.digitize`-style integer division, `np.isin`, a single broadcast distance matrix + `argmin`).
The remaining realistic speedup is not loop vectorisation but parallelism: the 207 independent
sessions (or the 7 animals) could be processed with `multiprocessing`, which the AI considered and
rejected because the total runtime is ~4 min.

ii.
```python
def pool_mean(x, k=TEMPORAL_BIN_FRAMES):
    """Average non-overlapping blocks of k samples along axis 0 (== AvgPool1d)."""
    n = (x.shape[0] // k) * k
    x = x[:n]
    return x.reshape((n // k, k) + x.shape[1:]).mean(axis=1)
...
            d = np.linalg.norm(position[bad][:, None, :] - centres[None], axis=2)
            cls[bad] = open_ids[np.argmin(d, axis=1)]
...
    while start < n_bins:              # ~40 iterations/session, O(1) numpy work each
        stop = min(start + BINS_PER_TRIAL, n_bins)
        ...
        sel = np.where(moving_binned[start:stop])[0] + start
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: ... `AvgPool1d` through torch
tensors -> replaced by a numpy reshape-mean (identical result). Python loops over frames for binning
(as in the reference `get_rate_maps`) -> not needed here; all operations are vectorised." "Total
runtime for all 207 sessions is ~3 min, so no parallelism was necessary."

## 6-c. What processing does the code repeat multiple times?

i. Only small redundancies:
- `env_blocked_vector(envs[day])` is computed twice per session - once in `main` for the
  `blocked`-consistency assertion and again inside `process_session`.
- `(~np.isnan(trace[0, :])).sum()` is computed twice per session (progress print and
  `session_info`), and `sum(t.shape[1] for t in ntr)` likewise.
- The speed threshold is applied twice, once at 30 Hz (`moving`, for cell curation) and once on the
  pooled speed (`moving_binned`, for bin selection); these are different quantities, so this is
  intentional rather than redundant.
- `keep.sum()` is recomputed in three places.
All of these are O(1)-to-O(n_cells) and invisible next to the ~0.9 s/session I/O + filtering. The
expensive operations (HDF5 read, Gaussian smoothing, pooling) each happen exactly once per session,
and the AI explicitly avoided the reference's redundant whole-animal loads.

ii.
```python
            expected = np.where(env_blocked_vector(envs[day]))[0]   # 1st call
            ...
            ntr, inp, outp, keep = process_session(trace, position, envs[day], ...)
# inside process_session:
    input_vec = env_blocked_vector(env_name)                        # 2nd call
...
            session_info.append({..., 'n_cells_registered': int((~np.isnan(trace[0, :])).sum()),
                                 ..., 'n_bins_kept': int(sum(t.shape[1] for t in ntr))})
            print(f'  {sid} env={envs[day]:<10s} cells {keep.sum():4d}/'
                  f'{(~np.isnan(trace[0, :])).sum():4d} reg  trials {len(ntr):3d}  '
                  f'bins {sum(t.shape[1] for t in ntr):6d}  ...')
```

iii. Not discussed explicitly in CONVERSION_NOTES.md beyond "lazy HDF5 session reads, numpy reshape
pooling, in-place float32 casting" (Step 6). The duplicated `env_blocked_vector` call is a
deliberate by-product of the independent consistency assertion in `main`, which the AI describes as
a sanity check ("the geometry name and the `blocked` field must agree", verified for all 207
sessions).

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts of wasted work, all of it cheap or diagnostic:
- **Smoothing and pooling of discarded bins**: `gaussian_filter1d` and `pool_mean` run over the
  whole session, but ~40% of the resulting bins are then dropped by the speed filter. (This
  ordering is deliberate - smoothing before removing immobility avoids mixing non-adjacent frames -
  so the waste is the price of a cleaner signal, not an oversight.) Similarly, the trailing
  remainder bins of a dropped partial trial are computed and discarded.
- **`blocked`** is read for every session but only used in the assertion; the input actually comes
  from `envs`.
- **`speed` at 30 Hz** is kept in full although only `moving` (for cell curation) and its pooled
  version are used.
- **Diagnostics**: `trial_slices` and the `diagnostics` dict are populated only in
  `--show-processing` mode (guarded by `diag is not None`, so nothing is wasted in normal runs).
- **Metadata that the decoder ignores**: `n_cells_file`, `n_cells_registered`, `sequence`,
  `n_frames_raw`, `geometries`, the per-session `session_info` records and the `input_names`
  entry `blocked_partition_7` (partition 7 is never blocked in any geometry, so that input channel
  is constant 0 in the whole dataset). These are documentation, not computation.
- More fundamentally, the whole static 9-dim `input` is constant within a session, and the harness
  trains one model per session, so the input carries no within-session information for the decoder -
  but it is required by the Decoder Task specification.

ii.
```python
    tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SMOOTH_SIGMA, axis=0)
    neural = pool_mean(tr_smooth).astype(np.float32) * FPS     # all bins, ~40% later dropped
...
    envs, blocked = read_meta(f)          # `blocked` used only for the assertion
...
    if diagnostics is not None:
        diagnostics.update(dict(speed=speed, moving=moving, neural=neural, ...))
...
            session_info.append({'session_id': sid, ..., 'n_cells_file': int(trace.shape[1]),
                                 'n_frames_raw': int(trace.shape[0]), ...})
```

iii. CONVERSION_NOTES.md documents the deliberate ones: the `blocked` read is a cross-check
("checked against the dataset's own `blocked` field for all 207 sessions"), the extra metadata is
for provenance ("`metadata['session_info']` - geometry name per session"), and the constant input
channel is noted in the Step 9 statistics table ("[0,1] for all except partition 7, which is never
blocked in any of the 10 geometries"). The smoothing-before-filtering order is justified in Step 10
Check 3 as the faithful analogue of the reference pipeline. The notes do not flag the ~40% of bins
that are smoothed and pooled before being discarded.
