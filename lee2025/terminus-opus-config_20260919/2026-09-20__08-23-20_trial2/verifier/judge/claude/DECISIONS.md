# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the per-animal **joblib** files in `/app/data` (the extension-less `QLAK-CA1-*` files), not the
`.mat` twins. This is the default path of the reference loader `load_dat(animal, p, format="joblib")`
(`utils.py:61`), which does `joblib.load(os.path.join(p_data, f"{animal}"))`. The animal list is hard-coded to the
7 IDs used in the reference `main.py`. One file is loaded per animal, the four needed fields are read out
(`trace`, `position`, `envs`, `blocked`), all days in the file are processed, and the animal dict is deleted before
moving to the next animal to bound peak memory. The AI verified in Step 2 of its notes that the `.mat` and joblib
files hold identical content and chose joblib because it is ~5x faster to load.

ii.
```python
DATA_DIR = "/app/data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
    animals = ANIMALS[3:4] if sample else ANIMALS   # sample: the smallest animal
    for ai, animal in enumerate(animals):
        t0 = time.time()
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        t_load = time.time() - t0
        envs = np.asarray(dat['envs']).ravel()
        n_days = dat['trace'].shape[0]
        ...
        for d in range(n_days):
            res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
        ...
        del dat
```

iii. From CONVERSION_NOTES.md Step 1/2: "`load_dat(animal, p, format='joblib')` -> `joblib.load(data/<animal>)[animal]`
... identical" to the reference. "`<animal>.mat` : identical content in MATLAB v7.3 format (loadable with mat73).
The joblib file is what `load_dat(..., format='joblib')` reads, so I use it (identical data, ~5x faster to load)."
Loading was instrumented; the full run loaded all 7 animals in 88.6 s and confirmed 207 sessions / 5,413 unique
cells / 69,744 registered cell-sessions, matching the paper exactly.

## 1-b. How are the data split into subjects?

i. One subject = one animal = one data file. The 7 animal IDs are the `subjects` list (in the hard-coded order
matching the reference `main.py`), and every session produced from an animal's file gets that animal's index in
`subject_idx`.

ii.
```python
        subject_idx = len(data['subjects'])
        data['subjects'].append(animal)
        ...
        for d in range(n_days):
            ...
            data['subject_idx'].append(subject_idx)
    ...
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. From the notes: the 7 mice `QLAK-CA1-08, -30, -50, -51, -56, -74, -75` are exactly the `animals` list in the
reference `main.py`, and the per-animal unique-cell counts (515, 875, 942, 554, 862, 713, 952; mean 773.3, min 515,
max 952) reproduce the paper's "mean number of cells per animal = 773 ± 68 SE, minimum 515, maximum 952".

## 1-c. How are the data split into sessions?

i. One session = one animal-day. The number of days is taken from the leading axis of `trace`
(`dat['trace'].shape[0]`), and every day is processed and appended as a separate output session. No sessions are
dropped: 31 + 31 + 31 + 21 + 31 + 31 + 31 = **207** sessions.

ii.
```python
        n_days = dat['trace'].shape[0]
        if sample:
            n_days = 2
        ...
        for d in range(n_days):
            res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
            data['neural'].append(res['neural'])
            data['input'].append(res['input'])
            data['output'].append(res['output'])
            data['subject_idx'].append(subject_idx)
            data['brain_region_idx'].append(np.zeros(res['n_neurons'], dtype=np.int64))
            session_id = f'{animal}_day{d:02d}_{envs[d].replace(" ", "-")}'
```
Rich per-session metadata is also stored (`session_info`: session_id, subject, day, environment name, blocked
partitions, n_neurons, n_trials, n_time_bins).

iii. Step 5, decision 1: "**Session = animal-day (207 total)**: the paper's unit of analysis ('one session was
recorded per day'); all 207 kept." The AI cross-checked that 207 sessions is exactly the paper's reported count
("5,413 unique neurons across 207 sessions in 10 geometries"), and that each animal's environment sequence starts
and ends with `square` and repeats the 10-geometry sequence 3 times (2 for QLAK-CA1-51), as the paper describes.

## 1-d. Are the data correctly split into trials?

i. Sessions are continuous ~40 min free-foraging recordings with no native trial structure, so trials are imposed
as non-overlapping 1-minute blocks, as the instructions require. Because the neural/behavioural streams are first
rebinned to 100 ms (see 2-e), a trial is **600 time bins** (= 1800 frames = 60 s). Unlike the human reference, the
AI **keeps the trailing partial block as a final trial when it is at least 30 s long**. All sessions are
71,866–72,219 frames (39.9–40.1 min), so every session yields **39 full 60 s trials plus one 55.5–60 s trial = 40
trials**, 8,280 trials in total.

ii.
```python
TRIAL_SECONDS = 60            # task specification: 1-minute trials
TRIAL_FRAMES = TRIAL_SECONDS * FPS                     # 1800 frames
TRIAL_BINS = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES       # 600 bins
MIN_PARTIAL_TRIAL_BINS = TRIAL_BINS // 2               # keep a trailing trial if >= 30 s

def trial_slices(n_bins):
    """Non-overlapping 1-minute trials; a trailing partial trial is kept if >= 30 s."""
    slices = []
    n_full = n_bins // TRIAL_BINS
    for t in range(n_full):
        slices.append((t * TRIAL_BINS, (t + 1) * TRIAL_BINS))
    rem = n_bins - n_full * TRIAL_BINS
    if rem >= MIN_PARTIAL_TRIAL_BINS:
        slices.append((n_full * TRIAL_BINS, n_bins))
    return slices
...
    for (a, b) in trial_slices(n_bins):
        neural.append(np.ascontiguousarray(rates[:, a:b]))
        output.append(part[a:b][None, :].astype(np.int64))
        inputs.append(blocked_vec.copy())
```
An explicit assertion guarantees the ">= 2 trials per session" format requirement:
```python
        assert len(data['neural'][s]) >= 2, f'session {s} has < 2 trials'
```

iii. Step 5, decision 2: "**Trial = 1 non-overlapping minute** (1800 frames @30 Hz), as required by the task. A
trailing partial minute is kept only if >= 30 s long (all sessions are 39.9-40.1 min, so every session yields 40
trials ...; 8,280 trials in total). Every session therefore has >= 2 trials." The rationale for keeping the
remainder (documented in Step 10, "Issues Found and Resolved") is that it preserves data and "the format allows
variable T". Smoothing is deliberately applied to the continuous session *before* cutting so no filter edge
artefacts are introduced at trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level quality filtering is applied.** All 8,280 trials from all 207 sessions are kept. The AI
explicitly considered and rejected the reference decoder's speed filter (`decode_position_within` drops frames with
smoothed speed <= 5 cm/s) and its per-cell event criterion. The only length-based rule is the >= 30 s minimum for
the trailing partial trial (1-d).

ii.
```python
    # (no filtering step exists; every slice returned by trial_slices becomes a trial)
    for (a, b) in trial_slices(n_bins):
        neural.append(np.ascontiguousarray(rates[:, a:b]))
```

iii. Step 5, decision 6: "**No speed filter**: `decode_position_within` drops frames with speed <= 5 cm/s, but the
paper's rate maps (which I reproduce exactly) use all frames, and the task requires contiguous time-varying outputs
within fixed 1-min trials. Dropping ~40% of the frames would make trials non-contiguous and discard data, so all
frames are kept." Step 3 adds: "the paper's rate maps themselves are built from **all** frames (verified by exact
reproduction), so speed filtering is specific to their Bayesian decoding analysis."

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. Only the `trace` field: `dat[animal]['trace'][day]`, shape `(n_cells, n_frames)`, float64, values strictly in
{0, 1} for registered cells and NaN for the whole session for unregistered cells. The dataset README and the paper
methods state this is the **binarized rising-phase transient vector**, which "was treated as the firing rate in all
further analyses", so no dF/F or deconvolution is required.

ii.
```python
            res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
...
def process_session(trace_day, position_day, env_name, blocked_field):
    """trace_day: (n_cells_all, n_frames) binary events, NaN rows for unregistered cells"""
    registered = ~np.isnan(trace_day[:, 0])
    ...
    tr = trace_day[registered].astype(np.float32)
    assert np.all(np.isin(tr, (0.0, 1.0))), "trace must be binary for registered cells"
```

iii. Step 1/3 notes: "Neural data are already preprocessed: `trace` is the **binarized rising-phase** vector
(1 = significant calcium event) - no dF/F computation needed." The code asserts binarity on every session, so the
assumption is actively verified on all 207 sessions.

## 2-b. How is the `neural` data processed?

i. Three steps, copied from the reference decoder `fit_decoder` (`utils.py:1776`):
1. keep registered cells only (2-c) and cast to float32;
2. `gaussian_filter1d(sigma = 3 frames)` **along time**, applied to the whole continuous session (before trial
   cutting, to avoid filter edge artefacts at trial boundaries);
3. non-overlapping 3-frame average pooling (the vectorised equivalent of the reference's
   `AvgPool1d(kernel_size=3, stride=3)`), then multiplied by `FPS = 30` so the stored values are **events per
   second** rather than events per frame.
Result: `(n_cells, n_bins)` float32 in [0, 30] Hz, mean 0.286 Hz.

ii.
```python
TEMPORAL_BIN_FRAMES = 3       # reference `temporal_bin_size` -> 100 ms bins
SMOOTH_SIGMA_FRAMES = 3       # reference gaussian_filter1d(sigma=temporal_bin_size)

def bin_time_series(x, kernel=TEMPORAL_BIN_FRAMES):
    """Average-pool the last axis in non-overlapping windows of `kernel` frames.
    Equivalent to torch.nn.AvgPool1d(kernel_size=kernel, stride=kernel) used by the
    reference `fit_decoder` / `test_decoder`."""
    n = (x.shape[-1] // kernel) * kernel
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)
...
    # --- neural temporal processing (reference fit_decoder) ---
    tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
    rates = (bin_time_series(tr_s) * FPS).astype(np.float32)      # events / s, (n_cells, n_bins)
```

iii. Step 5, decision 4: "**Neural pre-processing = reference decoder's**: `gaussian_filter1d(trace, sigma=3
frames, axis=time)` then 3-frame average pooling. Smoothing is done once on the **whole continuous session** before
cutting into trials, so there are no filter edge artifacts at trial boundaries. The result is multiplied by 30 to
express the rate in events/s (a constant scaling; makes the values interpretable and avoids the 'all values are 0
or 1' error that raw binary data would trigger in `verify_data_format`)." (That error is real —
`decoder.py:235` raises `"Neural data lacks variability. All values are 0 or 1"`.) An independent sanity check
(`cache/sanity_checks.py`) reloads the raw joblib files and confirms
`np.allclose(converted, pooled(gaussian_filter1d(raw_trace[registered], sigma=3, axis=1))*30)` for randomly chosen
sessions/trials.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron curation is **keep every cell registered in that session**, i.e. every row of `trace` that is
not NaN. Detection uses the first frame plus an assertion that unregistered cells are NaN for the *entire* session
(never partially). This yields 69,744 registered cell-sessions (mean 336.9/session, min 113, max 564), exactly the
"69,744 rate maps" reported in the paper. The AI explicitly declined to apply the reference decoder's
`cell_threshold=5` event criterion.

ii.
```python
    # --- neuron curation: keep cells registered in this session (non-NaN) ---
    registered = ~np.isnan(trace_day[:, 0])
    assert np.all(np.isnan(trace_day[~registered]).all(axis=1)), \
        "unregistered cells must be NaN for the entire session"
    tr = trace_day[registered].astype(np.float32)
```

iii. Step 5, decision 5: "**Neuron curation: keep every cell registered in that session** (non-NaN trace), i.e.
69,744 cell-sessions, the number quoted in the paper. The paper explicitly states its spatial-coding checks
'motivated the inclusion of all cells in subsequent analyses'. I deliberately do **not** apply the
`cell_threshold=5` event criterion used inside `decode_position_within`, because (a) the paper includes all cells,
(b) it would drop only 1.3% of cell-sessions, and (c) that criterion is defined over whole-session running frames,
which is not meaningful for 1-min trials."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is **no task event** to align to — the paradigm is 40 min of continuous free foraging. The AI therefore
defines the alignment event as the start of each 1-minute trial and documents it in the required metadata fields:
`temporal_alignment_event`, `off_start = 0.0`, `off_end = 60.0`. Neural, input and output streams within a trial
share the identical bin index range, so they are aligned by construction; the DAQ acquired behaviour and calcium at
30 Hz with post-hoc timestamp alignment by the original authors, so no shift is applied.

ii.
```python
        'temporal_alignment_event': (
            'start of each 1-minute trial; trial k starts 60*k s after the start of the continuous '
            '40-min recording session (there is no explicit task event - free foraging)'),
        'off_start': 0.0,
        'off_end': float(TRIAL_SECONDS),
...
    n_bins = rates.shape[1]
    assert pos_b.shape[1] == n_bins
    for (a, b) in trial_slices(n_bins):
        neural.append(np.ascontiguousarray(rates[:, a:b]))
        output.append(part[a:b][None, :].astype(np.int64))
```

iii. Step 3: "Temporal alignment: behaviour and calcium are recorded by the same DAQ at 30 Hz and timestamp-aligned
by the authors, so `position[d][:, t]` and `trace[d][:, t]` are already sample-for-sample aligned. No further
alignment is needed (verified by exactly reproducing the stored rate maps)." The `--show-processing` plots include
a dedicated "temporal alignment check" panel overlaying raw 30 Hz event times with the processed rate trace.

## 2-e. How is the `neural` data temporally binned/resampled?

i. The data is **rebinned from 30 Hz (33.3 ms) to 10 Hz (100 ms)** by 3-frame average pooling after Gaussian
smoothing — exactly the reference decoder's `temporal_bin_size=3`. `metadata['time_bin_size'] = 100.0` ms. Each
full trial is 600 bins. Position is pooled over the same bins (4-b), so the streams stay aligned. The 0–2 trailing
frames of a session that cannot fill a whole 100 ms bin are dropped.

ii.
```python
FPS = 30                      # acquisition rate of both behaviour and calcium streams
TEMPORAL_BIN_FRAMES = 3       # reference `temporal_bin_size` -> 100 ms bins
...
        'time_bin_size': 1000.0 * TEMPORAL_BIN_FRAMES / FPS,   # 100.0 ms
...
def bin_time_series(x, kernel=TEMPORAL_BIN_FRAMES):
    n = (x.shape[-1] // kernel) * kernel
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)
```

iii. Step 5, decision 3: "**Time bin = 100 ms (3 frames)**: exactly the reference decoder's `temporal_bin_size=3`
with `AvgPool1d`. 600 bins per trial." Step 1 notes that `fit_decoder`/`test_decoder` perform "Temporal binning:
`AvgPool1d(kernel_size=3, stride=3)` on position & traces (30 Hz -> 10 Hz, 100 ms bins)".

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Primarily the `envs` field (the geometry name string: `square`, `o`, `t`, `u`, `rectangle`, `+`, `i`, `l`,
`bit donut`, `glenn`), mapped through a verbatim copy of the reference `get_env_mat` (`utils.py:215`) into a 3x3
binary matrix. The dataset's own `blocked` field (indices 0–8, or `-1` for none) is used as an independent
cross-check: the script **asserts** that the geometry-derived vector equals the `blocked`-derived vector for every
one of the 207 sessions.

ii.
```python
ENV_MATS = {
    'square':    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    'o':         [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    ... }

def env_blocked_vector(env_name):
    """Binary length-9 vector, 1 where the partition is blocked, using dataset indexing."""
    mat = np.array(ENV_MATS[env_name], dtype=float)
    open_flat = np.flipud(mat).ravel()      # index p -> 1 if open
    return (open_flat == 0).astype(np.float32)
...
    # consistency check against the dataset's own `blocked` field
    bl = np.atleast_1d(np.asarray(blocked_field[0]).ravel()).astype(int)
    from_field = np.zeros(9, dtype=np.float32)
    if not (bl.size == 1 and bl[0] == -1):
        from_field[bl] = 1
    assert np.array_equal(from_field, blocked_vec), \
        f"blocked field {bl} disagrees with env matrix for '{env_name}'"
```

iii. Step 4: "Convention resolved: `blocked` indexes `np.flipud(get_env_mat(env)).ravel()`", derived from the
reference `clean_rate_maps` mask (`np.fliplr(get_env_mat(env).T)`) and verified empirically — "`blocked` matches
low-occupancy partitions with index `3*floor(y/25)+floor(x/25)` in all 207 sessions". Using the geometry name plus
an assertion against `blocked` means the input is built from the reference's own function and independently
validated against the stored field.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The 3x3 open/blocked matrix is flipped vertically and raveled to give a length-9 binary vector in the dataset's
partition indexing (`p = 3*row + col`, row from y), with `1 = blocked`. This static vector (shape `(9,)`,
float32) is attached to every trial of the session, matching "Environment geometry ... Static per-trial."
`input_names` are `blocked_partition_0 ... blocked_partition_8`. Partition 7 is never blocked in any of the 10
geometries, which the AI verified as correct.

ii.
```python
    blocked_vec = env_blocked_vector(env_name)
    ...
    for (a, b) in trial_slices(n_bins):
        ...
        inputs.append(blocked_vec.copy())
...
    'input_names': [f'blocked_partition_{p}' for p in range(9)],
```

iii. Step 5 mapping table: "`dat[animal]['blocked'][day]` (indices of blocked partitions, -1 = none) ->
`input[session][trial]` (9,) static; binary vector b[p]=1 if partition p blocked; `get_env_mat`/`clean_rate_maps`
define the same 3x3 layout; matches 'Decoder Inputs: environment geometry ... static per trial'." Decision 9:
"**Input is static per trial** (shape (9,)) as specified by the task: geometry is constant within a session."

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Only the `position` field: `dat[animal]['position'][day]`, shape `(2, n_frames)`, x and y head position in cm
from DeepLabCut tracking, range 0–75 cm, with no NaNs anywhere in the dataset (checked in Step 2).

ii.
```python
            res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
...
def process_session(trace_day, position_day, env_name, blocked_field):
    """position_day: (2, n_frames) x, y position in cm"""
    ...
    pos_b = bin_time_series(position_day.astype(np.float64))       # (2, n_bins)
```

iii. Step 2: "`position` | (n_days, 2, n_frames) float64 | x,y head position in cm, range [0, 75]; no NaNs
anywhere". Methods: "Position data were generated from tracking the head with DeepLabCut pose-estimation
software."

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. x and y are average-pooled into the same non-overlapping 3-frame (100 ms) bins used for the neural data
(in float64 to avoid precision loss), then discretized (4-c). No smoothing, no speed filtering, no outlier
removal, and no rescaling: the fixed physical 0–75 cm arena grid is used rather than the reference
`get_rate_maps` per-session `nanmax` grid.

ii.
```python
    # --- behaviour: same temporal bins, then spatial discretization ---
    pos_b = bin_time_series(position_day.astype(np.float64))       # (2, n_bins)
    blocked_vec = env_blocked_vector(env_name)
    ...
    part, n_snapped = position_to_partition(pos_b, blocked_vec)
```

iii. Step 5 mapping: "average x,y within each 100 ms bin ... `fit_decoder` pools position the same way". Step 4/10
justify the fixed grid: "the stored rate maps are reproduced exactly by the fixed grid, the arena is physically
75 cm, and a fixed grid is stable across sessions. Per-session maxima differ by <3% so <1% of labels are
affected." The AI verified this by re-deriving the authors' stored `maps['unsmoothed']` from raw `position` and
`trace` on a fixed 5 cm grid and matching them to within a single frame out of 72,219.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The 75 cm arena is cut into a fixed 3x3 grid of 25 cm partitions. Column = `clip(floor(x/25), 0, 2)`,
row = `clip(floor(y/25), 0, 2)`, and the class label is `p = 3*row + col`, giving 9 classes in exactly the same
index space as the `blocked` input vector. Clipping handles the boundary case `x == 75.0` or `y == 75.0` (present
in every animal) instead of creating a 4th bin. Additionally, the AI **snaps the 0.003 % of time bins (152 of
4,967,611) that land inside a partition that is physically walled off** to the nearest *open* partition, by
Euclidean distance from the binned x,y to the open-partition centres. `output_names = ['position_partition']`,
`output_values` name all 9 partitions.

ii.
```python
PART_CENTERS = np.array([[PART_CM * (p % N_PART) + PART_CM / 2,
                          PART_CM * (p // N_PART) + PART_CM / 2] for p in range(9)])

def position_to_partition(pos_binned, blocked_vec):
    """Frames tracked inside a blocked partition (tracking noise; 0.003% of all frames) are
    re-assigned to the nearest open partition, mirroring the reference decoder's step that
    snaps actual/predicted positions to valid spatial bins (`decode_position_within`)."""
    col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
    row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
    p = (N_PART * row + col).astype(np.int64)
    bad = blocked_vec[p] > 0
    if np.any(bad):
        open_ids = np.flatnonzero(blocked_vec == 0)
        d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None], axis=2)
        p[bad] = open_ids[np.argmin(d, axis=1)]
    return p, int(np.sum(bad))
```
and a global assertion that no output label is ever a blocked partition:
```python
            assert np.all(inp[out[0]] == 0), 'output partition must never be a blocked one'
```

iii. Step 5, decision 7: "**Output = single 9-class variable** ... with the same partition indexing as the
dataset's `blocked` field ... This makes input (blocked partitions) and output (occupied partition) use one
consistent 0-8 labelling." Decision 8: "**Blocked-partition tracking noise**: 0.003% of frames are tracked inside a
partition that is walled off. Such time bins are re-assigned to the nearest **open** partition ..., mirroring the
reference decoder's step that snaps actual/predicted positions to valid bins. This avoids creating classes with
1-2 samples that would distort balanced accuracy."

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Both streams are indexed by the same DAQ frame, pooled with the same `bin_time_series` (same kernel, same
truncation of the trailing incomplete bin), and cut with the same `trial_slices` boundaries. An assertion checks
that the binned position and binned rates have the same number of bins, and a per-trial assertion checks
`out.shape == (1, nrl.shape[1])`. No shift or interpolation is applied.

ii.
```python
    n_bins = rates.shape[1]
    assert pos_b.shape[1] == n_bins
    ...
    for (a, b) in trial_slices(n_bins):
        neural.append(np.ascontiguousarray(rates[:, a:b]))
        output.append(part[a:b][None, :].astype(np.int64))
        inputs.append(blocked_vec.copy())
...
            assert out.shape == (1, nrl.shape[1]) and out.dtype == np.int64
```

iii. Step 10, Check 3(c): "Temporal alignment: reference = `position[d][:, t]` and `trace[d][:, t]` share the DAQ
frame index; my code = same index, no shifts — YES." An end-to-end cross-check (`cache/crosscheck_maps.py`)
recomputes per-partition mean firing rate from the *converted* neural + output streams and compares it to the
dwell-time-weighted aggregation of the authors' stored 15x15 rate maps, obtaining r = 0.9995–0.9999 — jointly
testing alignment, partition indexing and map axis order.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Unregistered cells** (NaN for a whole session) are dropped, with an assertion that they are NaN for the
  *entire* session (never partially missing).
- **Non-binary trace values** would raise: `assert np.all(np.isin(tr, (0.0, 1.0)))`.
- **Position exactly on the arena edge** (x or y == 75.0 cm, present in every animal) is clipped into
  row/column 2 rather than spilling into a nonexistent 4th bin.
- **Tracking noise inside walled-off partitions** (152 of 4,967,611 bins = 0.003 %) is snapped to the nearest open
  partition.
- **Frames that cannot fill a whole 100 ms bin** (0–2 per session) are truncated by `bin_time_series`.
- **Trailing partial minute** is kept as a trial if >= 30 s, otherwise dropped.
- **Geometry/`blocked` disagreement** would raise an assertion.
- Final whole-dataset assertions check shapes, dtypes, finiteness (`np.isfinite(nrl).all()`), output range 0–8,
  >= 2 trials/session, and that output partitions are never blocked.
There are no NaNs in `position`, so no interpolation was needed.

ii.
```python
    registered = ~np.isnan(trace_day[:, 0])
    assert np.all(np.isnan(trace_day[~registered]).all(axis=1)), \
        "unregistered cells must be NaN for the entire session"
    tr = trace_day[registered].astype(np.float32)
    assert np.all(np.isin(tr, (0.0, 1.0))), "trace must be binary for registered cells"
...
    col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
    row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
...
        for tr_i in range(len(data['neural'][s])):
            nrl, out, inp = data['neural'][s][tr_i], data['output'][s][tr_i], data['input'][s][tr_i]
            assert nrl.shape[0] == nn and nrl.dtype == np.float32
            assert out.shape == (1, nrl.shape[1]) and out.dtype == np.int64
            assert inp.shape == (9,)
            assert out.min() >= 0 and out.max() <= 8
            assert np.all(inp[out[0]] == 0), 'output partition must never be a blocked one'
            assert np.isfinite(nrl).all()
```

iii. Step 10, Check 5 ("Edge cases") enumerates each of these and states the outcome: unregistered cells never
partial; edge positions clipped; trailing partial trial kept (>= 30 s); tracking noise snapped "so no class is
created from a handful of spurious samples"; no session has < 2 trials; smoothing applied before trial cutting so
no boundary artefacts. The notes also record that "99.997% of recorded frames are present ... no quality
exclusions."

## 6-a. What are the most time-consuming steps of the code?

i. The AI instrumented and reported timings. Full run = **187.2 s** total: **88.6 s joblib loading** (I/O +
decompression of the 7 animal files, the single largest component), **92.4 s session processing** (0.28–0.54 s per
session, dominated by `gaussian_filter1d` over the `(n_cells, ~72,000)` matrix), and **5.2 s** pickling the 6.73 GB
output. Loading is the top cost, and it was already reduced ~5x by choosing joblib over the mat73 `.mat` path.

ii.
```python
    for ai, animal in enumerate(animals):
        t0 = time.time()
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        t_load = time.time() - t0
        ...
        print(f'{animal}: loaded in {t_load:.1f}s, {n_days} sessions, ...')
        t1 = time.time()
        for d in range(n_days):
            ...
        print(f'  processed {n_days} sessions in {time.time() - t1:.1f}s '
              f'({(time.time() - t1) / n_days:.2f}s/session)')
    ...
    print(f'wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')
    print(f'total elapsed {time.time() - t_start:.1f}s')
```

iii. Step 6/7 notes: "Code speedups added: fully vectorised smoothing + pooling ..., one joblib load per animal
(the 15 GB `.mat` files are never touched), and `float32` neural storage." Step 7 estimated "< 10 min (well under
the 15 min budget)" and Step 9 confirmed 187 s, "matching the Step 7 estimate."

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Essentially none of substance remain. The AI replaced the reference's per-frame Python loops
(`get_rate_maps`, `fit_decoder`'s one-hot loop) with fully vectorised operations: `gaussian_filter1d` over the
whole `(n_cells, n_frames)` matrix, a reshape-mean instead of `AvgPool1d`, vectorised `floor`/`clip`
discretization, and a single vectorised nearest-open-partition computation for the snapped bins. The remaining
loops are over animals (7), sessions (207) and trial slices (40/session, pure slicing) and are inherently cheap.
Minor residual candidates, all negligible: the `np.ascontiguousarray` copy per trial, `blocked_vec.copy()` per
trial (8,280 tiny 9-element arrays), and the three separate end-of-run Python passes over all sessions/trials for
assertions, occupancy and rate-range statistics.

ii.
```python
    tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)   # whole session at once
    rates = (bin_time_series(tr_s) * FPS).astype(np.float32)
...
    col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
    row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
    p = (N_PART * row + col).astype(np.int64)
    bad = blocked_vec[p] > 0
    if np.any(bad):
        open_ids = np.flatnonzero(blocked_vec == 0)
        d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None], axis=2)
        p[bad] = open_ids[np.argmin(d, axis=1)]
```

iii. Step 6: "Code inefficiencies identified: the per-frame python loops of the reference `get_rate_maps` /
`get_transition_matrix`; these are not needed for the conversion and were replaced by vectorised reshape/mean."
Step 7 quantifies: "vectorised smoothing/pooling instead of per-frame loops | ~100x on the neural step".

## 6-c. What processing does the code repeat multiple times?

i. Small, validation-driven repeats only:
- `env_blocked_vector(env_name)` is recomputed for every session although there are only 10 distinct geometries
  across 207 sessions (trivial cost).
- The `blocked`-field consistency vector is rebuilt and re-asserted every session (deliberate — it is the
  cross-check).
- `blocked_vec.copy()` is stored once per trial rather than sharing one array.
- At the end of `main()` the converted dataset is traversed three separate times: once for the assertion sweep,
  once to accumulate the occupancy histogram, and once to concatenate first-trial rates for the range printout.
None of these materially affect the 187 s runtime.

ii.
```python
    blocked_vec = env_blocked_vector(env_name)          # recomputed per session
    ...
        inputs.append(blocked_vec.copy())               # one copy per trial
...
    for s in range(ns):                                  # pass 1: assertions
        ...
    occ = np.zeros(9)
    for s in range(ns):                                  # pass 2: occupancy
        for o in data['output'][s]:
            occ += np.bincount(o[0], minlength=9)
    allr = np.concatenate([data['neural'][s][0].ravel() for s in range(ns)])   # pass 3: range
```

iii. The AI did not flag these as problems; its efficiency discussion (Step 6/7) focuses on the vectorisation that
was added and on avoiding the `.mat` files. The repeated end-of-run passes are consequences of the instructions'
requirement to run sanity checks and print summary statistics after conversion.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI states there is essentially none, and the design does avoid the obvious waste (it never touches the
15 GB `.mat` files, never computes rate maps, never runs the reference's Bayesian decoder). Some avoidable work
nevertheless remains:
- `joblib.load` deserialises the **entire** animal file, including `SFPs` (35x35xn_cellsxn_days), `centroids` and
  the three `maps` arrays, none of which are used — this is the bulk of the 88.6 s load time.
- `process_session` returns the full-session diagnostic arrays `rates`, `pos_b`, `part`, `tr` in its result dict
  even when `--show-processing` is off; they are only consumed by the plotting path and are freed by `del res`.
- The binarity assertion `np.all(np.isin(tr, (0.0, 1.0)))` scans every element of every session's trace (~2.4x10^10
  element comparisons across the dataset) purely as a check.
- `np.ascontiguousarray` on each trial slice materialises a copy of the neural data.
- Multiplying the pooled rates by `FPS` is a constant rescaling that cannot change decoder accuracy (it is done for
  interpretability and to avoid the validator's all-0/1 error).

ii.
```python
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]   # also loads SFPs, centroids, maps
...
    return {
        'neural': neural, 'output': output, 'input': inputs,
        ...
        'rates': rates, 'pos_b': pos_b, 'part': part, 'tr': tr,     # diagnostics, plotting only
    }
...
    assert np.all(np.isin(tr, (0.0, 1.0))), "trace must be binary for registered cells"
...
        neural.append(np.ascontiguousarray(rates[:, a:b]))
```

iii. The notes do not list these as unnecessary work; the AI's stated position (Step 6) is that the conversion
avoids the reference's expensive per-frame loops and that the runtime (187 s) is far inside the 15-minute budget,
so no further optimisation was pursued. The assertions are presented as deliberate correctness checks required by
the instructions rather than as overhead.
