# Decisions

Documentation of the decisions made by the agentic AI system in `/app/convert_data.py`,
`/app/CONVERSION_NOTES.md` and its trajectory, for the conversion of the Lee, Keinath, Cianfarano &
Brandon (2025) CA1 geometric-deformation dataset into the decoder-compatible pickle format.

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. **Decisions**

- The AI reads the seven MATLAB v7.3 (`= HDF5`) files `/app/data/QLAK-CA1-{08,30,50,51,56,74,75}.mat`
  with `h5py`, rather than the equivalent joblib dumps that the reference code's `utils.load_dat`
  uses. It established that the two formats hold identical content (the joblib files were produced
  from the `.mat` by the reference `mat2joblib`), and chose HDF5 because it permits **lazy per-day
  reads** (a whole animal decompressed from joblib is 9–25 GB of float64).
- The animal list is hard-coded (`ANIMALS`), taken from the animal list in the reference
  `georepca1/main.py`.
- Each file is opened once; for every day the AI dereferences the object references stored in
  `position`, `trace`, `envs` and `blocked` and reads only that day's arrays.
- HDF5 returns every axis reversed relative to the joblib arrays, so `f[trace_ref]` is
  `(T, n_cells)` and `f[position_ref]` is `(T, 2)` — the orientation
  `decode_position_within` works in. The AI documented and verified this.
- One worker process per animal (`multiprocessing.Pool`, `--nproc` default 7); results are merged
  and re-sorted by `(animal, day)` so the output session order is deterministic.
- Full conversion: 207 sessions in 39.8 s of processing + 4.3 s pickling = **44 s** wall clock.

ii. **Code snippets**

```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

def _h5_str(f, ref):
    """Decode a MATLAB char array stored behind an object reference."""
    return ''.join(chr(c) for c in f[ref][:].ravel())

def read_session(f, day):
    position = f[f['position'][day, 0]][()].astype(np.float64)   # (T, 2)
    trace    = f[f['trace'][day, 0]][()].astype(np.float32)      # (T, n_cells)
    env      = _h5_str(f, f['envs'][0, day])
    blocked  = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
    blocked  = blocked[blocked >= 0].astype(int)   # -1 means "nothing blocked" (the square)
    return position, trace, env, blocked
```

```python
def process_animal(task):
    animal, days, plot_days = task
    with h5py.File(os.path.join(DATA_DIR, f"{animal}.mat"), 'r') as f:
        n_days = f['trace'].shape[0]
        day_list = range(n_days) if days is None else days
        for day in day_list:
            position, trace, env, blocked = read_session(f, day)
            ...

# main()
tasks = [(a, None, []) for a in ANIMALS]
if len(tasks) > 1 and args.nproc > 1:
    with Pool(min(args.nproc, len(tasks))) as pool:
        results = pool.map(process_animal, tasks)
all_sessions = [s for r in results for s in r]
all_sessions.sort(key=lambda s: (s['animal'], s['day']))
```

iii. **Justification**

From CONVERSION_NOTES.md Step 2 / Step 5 decision 9 / Step 10 Check 0: "`/app/data/QLAK-CA1-…`
(joblib) and `.mat` (MATLAB v7.3 = HDF5). The two are identical in content; the `.mat` is used in
this conversion because HDF5 allows **lazy, per-day reads** (a whole animal decompressed from joblib
is 9-25 GB of float64 in RAM)." The equivalence was checked explicitly (sanity Check 0: "HDF5 `.mat`
reads == joblib arrays that `utils.load_dat` returns (position, trace incl. NaNs, envs), days
0/5/30 — PASS"). The number of days is read from the file (`f['trace'].shape[0]`), so the 21-day
animal QLAK-CA1-51 is handled without special-casing. Parallelism over animals is listed under
"Code speedups added".

---

## 1-b. How are the data split into subjects?

i. **Decisions**

- One subject = one animal = one `.mat` file. Seven subjects, matching the paper.
- The subject list is built as `sorted({s['animal'] for s in all_sessions})`, and `subject_idx` is
  the index of each session's animal into that list.
- The animal IDs themselves come from the hard-coded `ANIMALS` constant (copied from the reference
  `main.py`) rather than from globbing the data directory.
- Verified against the paper: 7 animals, 515/875/942/554/862/713/952 cells, mean 773 ± 68 SE,
  min 515, max 952 — matching "mean number of cells per animal = 773 ± 68 SE, minimum cells per
  animal = 515, maximum cells per animal = 952".

ii. **Code snippets**

```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
subjects = sorted({s['animal'] for s in all_sessions})
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subjects.index(s['animal']) for s in all_sessions], dtype=np.int64),
}
```

iii. **Justification**

CONVERSION_NOTES.md Step 2: each `.mat` file is one animal and holds all of that animal's recording
days; Step 3 lists "Subjects | 7 | 7 animal IDs in `main.py`". Step 9's consistency table records
"Subjects | 7 | 7 animal IDs in main.py | 7 files | 7 | YES" and the per-animal cell counts matching
the paper exactly.

---

## 1-c. How are the data split into sessions?

i. **Decisions**

- One session = one recording **day** for one animal (the `day` index into the `trace` / `position` /
  `envs` / `blocked` reference arrays). 207 sessions total = 31+31+31+21+31+31+31.
- Rationale recorded as Key Decision 1: neuron identity is only guaranteed constant within a day in
  the required format (`n_neurons` must be constant within a session), and the geometry — the decoder
  input — changes from day to day.
- No session-level exclusion: all 207 sessions are kept (the AI only defined a fallback rule for
  sessions yielding < 2 usable trials, which never triggered).
- Each session carries `session_info` metadata: session_id, subject, day, environment name, blocked
  partitions, n_neurons, n_registered_cells, n_cells_in_registry, n_frames_30hz, n_trials,
  frac_bins_running, n_samples_snapped_to_open_bin.

ii. **Code snippets**

```python
with h5py.File(os.path.join(DATA_DIR, f"{animal}.mat"), 'r') as f:
    n_days = f['trace'].shape[0]
    day_list = range(n_days) if days is None else days
    for day in day_list:
        position, trace, env, blocked = read_session(f, day)
        sess = process_session(position, trace, env, blocked,
                               session_id=f"{animal}_day{day:02d}", ...)
```

```python
session_info.append({
    'session_id': s['session_id'], 'subject': s['animal'], 'day': int(s['day']),
    'environment': s['env'], 'blocked_partitions': [int(b) for b in s['blocked']],
    'n_neurons': int(s['n_neurons']), 'n_registered_cells': int(s['n_registered']),
    'n_cells_in_registry': int(s['n_cells_total']), 'n_frames_30hz': int(s['n_frames']),
    'n_trials': len(s['neural']), 'frac_bins_running': round(s['frac_moving'], 4),
    'n_samples_snapped_to_open_bin': int(s['n_snapped']),
})
```

iii. **Justification**

CONVERSION_NOTES.md Step 5, Key Decision 1: "**Session = recording day.** 207 sessions. Cell identity
is only stable within a day in the format required here (neuron count must be constant within a
session but may differ between sessions), and the geometry — the decoder input — changes day to
day." Key Decision 8: "**All 7 animals and all 207 sessions are used** — the paper applies no
session- or animal-level exclusion." Consistent with the paper's "5,413 unique neurons across 207
sessions in 10 geometries".

---

## 1-d. How are the data split into trials?

i. **Decisions**

- The source data has **no trial structure** (40 min of continuous free foraging), so trials are
  imposed by the Decoder Task: contiguous **60 s blocks**.
- Because the AI rebinned to 100 ms (see 2-e), a full trial is **600 bins**, not 1800 frames.
- The trailing partial block is **kept** as a shorter trial if it spans ≥ 10 s (`MIN_TAIL_BINS = 100`),
  otherwise dropped. A 71,866-frame session gives 39 full 60 s trials + one 55 s trial = 40 trials.
- Within each block, only bins in which the animal was running are written out (see 1-e / 2-c), so
  **trial lengths are variable**: mean 315.4, min 30, max 561 bins of a possible 600.
- Result: 8,148 trials over 207 sessions (mean 39.4/session, min 22, max 40).

ii. **Code snippets**

```python
TRIAL_SEC = 60.0                                           # Decoder Task: 1-minute trials
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / BIN_FRAMES))  # 600 bins of 100 ms
MIN_TAIL_BINS = 100     # keep the trailing partial block only if it covers >= 10 s
MIN_TRIAL_BINS = 30     # drop a trial left with < 3 s of running
```

```python
starts = list(range(0, n_bins - BINS_PER_TRIAL + 1, BINS_PER_TRIAL))
bounds = [(s, s + BINS_PER_TRIAL) for s in starts]
tail_start = len(starts) * BINS_PER_TRIAL
if n_bins - tail_start >= MIN_TAIL_BINS:
    bounds.append((tail_start, n_bins))

for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
    if idx.size < MIN_TRIAL_BINS:
        continue
    neural_trials.append(np.ascontiguousarray(neural[:, idx]))
    input_trials.append(geometry.copy())
    output_trials.append(pos_bin[idx][np.newaxis, :].copy())
```

iii. **Justification**

CONVERSION_NOTES.md Step 5, Key Decision 2: "**Trial = contiguous 60 s block of the session** (600
bins of 100 ms), as specified in the Decoder Task. ~40 trials per session. The trailing partial block
is kept if it covers >= 10 s of recording, otherwise dropped (nothing else is discarded, so
essentially no data is lost)." Step 3 records that the source has no trials and no stimulus event:
"There is no trial structure and no stimulus event in this experiment — a session is 40 min of free
foraging."

---

## 1-e. How are trials filtered based on quality controls?

i. **Decisions**

Two trial-level rules, both consequences of the timepoint-level running filter:

- A trial is **dropped if fewer than 30 retained bins (3 s of running)** remain in its 60 s block
  (`MIN_TRIAL_BINS = 30`). This is why 46 of 207 sessions have < 40 trials (min 22).
- A **session is dropped if fewer than 2 usable trials** remain, because the provided validator
  requires at least two trials per session to evaluate the decoder. In practice this never fires —
  all 207 sessions are retained.
- The trailing partial block is dropped if shorter than 10 s (see 1-d).
- No other trial-level curation (no rejection based on behaviour, coverage, or neural quality).

ii. **Code snippets**

```python
MIN_TRIAL_BINS = 30      # drop a trial left with < 3 s of running

for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s
    if idx.size < MIN_TRIAL_BINS:
        continue
    ...

if len(neural_trials) < 2:
    return None          # session dropped
```

```python
if sess is None:
    print(f"  [{animal} day {day}] SKIPPED (fewer than 2 usable trials)", flush=True)
    continue
```

iii. **Justification**

CONVERSION_NOTES.md Step 5, Key Decision 4: "Trials retaining < 30 bins (3 s) after this filter are
dropped." Step 10 Check 5 (edge cases): "Trial left with very little running | trials with < 30 bins
(3 s) dropped; this is why 46 sessions have < 40 trials (min 22)"; "Sessions with < 2 usable trials |
would be dropped (the validator needs 2 trials per session) — **none occur**, all 207 sessions are
kept". The underlying running criterion is the reference decoder's own (`v_thresh=5`), documented in
Step 1/Step 3.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. **Decisions**

- `neural` is derived **solely from the `trace` field** of each animal's `.mat` file, shape
  `(T, n_cells)` per day under HDF5 axis-reversal.
- The AI established that `trace` is already the paper's final signal: the **binarised
  transient rising-phase vector** with values in `{0, 1}` (plus all-NaN columns for cells not
  registered that day). No ΔF/F, motion correction, segmentation, or deconvolution is needed — all of
  that was done upstream in MATLAB by the authors.
- `position` is additionally used, but only to build the running mask that curates the neural data
  (cells and timepoints), not to modify the trace values themselves.

ii. **Code snippets**

```python
def read_session(f, day):
    ...
    trace = f[f['trace'][day, 0]][()].astype(np.float32)   # (T, n_cells), binarised rising phases
```

```
metadata['neural_units'] =
    'Rate of binarised calcium-transient rising phases. The distributed `trace` is the binary '
    'event vector the paper treats as the firing rate; ...'
```

iii. **Justification**

CONVERSION_NOTES.md Step 1: "**Calcium processing**: none needed in this conversion. `trace` in the
distributed dataset is **already** the final binarised 'rising phase of transient' vector (values in
{0, 1}, NaN for cells not registered that day). dF/F extraction, motion correction, CNMF-E
segmentation, transient extraction and z-scoring at 2.5 SD were all performed upstream in MATLAB by
the authors (methods.txt, 'Data preprocessing'). **No dF/F needs to be computed.**" Step 4 confirms
empirically: "`trace` values are exactly {0, 1} (or NaN)". The paper states "This binary vector was
treated as the firing rate in all further analyses."

---

## 2-b. How is the `neural` data processed?

i. **Decisions**

The AI reproduced the paper's own within-session position decoder pre-processing
(`fit_decoder` / `decode_position_within` in `georepca1/src/utils.py`), in this order:

1. Drop cells not registered that day (all-NaN columns); assert no registered cell has partial NaNs.
2. Drop cells with ≤ 5 binarised events during running (see 2-c).
3. **Gaussian smoothing along time**, `gaussian_filter1d(trace, sigma=3 frames, axis=0)` —
   `fit_decoder`'s `gaussian_filter1d(traces, sigma=temporal_bin_size=3, axis=0)`.
4. **Average-pool 3 frames** (reshape + `mean`, equivalent to `AvgPool1d(kernel=3, stride=3)`)
   → 100 ms bins, then transpose to `(n_neurons, n_bins)` and cast to `float32`.
5. Keep only the running bins, then slice into 1-minute trials.

A **deliberate ordering difference** from the reference: the reference smooths *after* subsetting to
running frames (and after the CV fold split), which convolves across temporal discontinuities. The AI
smooths and pools the intact, contiguous 30 Hz session and applies the speed criterion afterwards.

ii. **Code snippets**

```python
FPS = 30                 # acquisition rate of both imaging and behaviour streams
BIN_FRAMES = 3           # fit_decoder(temporal_bin_size=3)  ->  100 ms bins
SMOOTH_SIGMA = 3         # fit_decoder: gaussian_filter1d(traces, sigma=temporal_bin_size)
```

```python
# --- neuron curation 1: cells registered on this day (trace is all-NaN otherwise) ---
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.any(np.isnan(trace)), "a registered cell has partial NaNs -- unexpected"

# --- temporal smoothing + binning to 100 ms (fit_decoder) ---
n_bins = n_frames_raw // BIN_FRAMES
n_use = n_bins * BIN_FRAMES
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)
```

iii. **Justification**

CONVERSION_NOTES.md Step 5, Key Decision 3: "**Time bin = 100 ms (3 frames @30 Hz)** with gaussian
pre-smoothing sigma = 3 frames — exactly `fit_decoder`'s `temporal_bin_size=3`. This also makes the
neural values continuous rather than binary, which the validator requires ('Neural data lacks
variability' error if all values are 0/1). *Difference from the reference*: the reference smooths
**after** discarding immobility frames, which convolves across temporal discontinuities. Here
smoothing and pooling are done on the intact 30 Hz session and the speed criterion is applied
afterwards, which is strictly more correct and otherwise identical."

The AI validated the pipeline by copying `fit_decoder`/`test_decoder`/`decode_position_within`
verbatim into `cache/reference_check.py` and reproducing the authors' own stored Figure 1F decoding
errors on 12 sessions to the last printed digit (diff = 0.000 cm), establishing that "the velocity
filter, cell filter, temporal smoothing/pooling and spatial binning used in this conversion are the
reference ones".

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. **Decisions**

Two cell-level rules (both copied from `decode_position_within`) plus one timepoint-level rule:

- **Registration**: a cell is kept for a session only if its trace is not all-NaN, i.e. it was
  registered that day. 5,413 unique cells → 69,744 registered cell-sessions, matching the paper's
  "69,744 rate maps".
- **Activity**: `traces[running].sum(axis=0) > 5` — more than 5 binarised events while the animal is
  running (`cell_threshold=5`). This removed 882 of 69,744 cell-sessions (1.26%), leaving 68,862
  (mean 332.7/session, min 112, max 562).
- **No place-cell selection** — the paper explicitly includes all cells.
- **Timepoint curation**: only bins where running speed > 5 cm/s are written out
  (`v_thresh=5`, velocity smoothed with `sigma=5` frames), the reference decoder's own rule. This
  retains 51.8% of bins on average (min 21.7%, max 75.3%), i.e. 2,569,722 of ~4.96M bins.
  The 30 Hz mask is downsampled to 100 ms bins by majority vote. Frame 0 has no velocity estimate and
  is always excluded, matching the reference's `vel_idx[d, 1:] = ...`.
- No session- or animal-level exclusion.

ii. **Code snippets**

```python
V_FILT_SIZE = 5          # decode_position_within(v_filt_size=5)
V_THRESH = 5.0           # decode_position_within(v_thresh=5) -- cm/s
CELL_THRESH = 5          # decode_position_within(cell_threshold=5) -- events while running

def running_speed(position):
    speed = np.zeros(position.shape[0])
    speed[1:] = gaussian_filter1d(np.linalg.norm(np.diff(position, axis=0) * FPS, axis=1),
                                  sigma=V_FILT_SIZE)
    return speed
```

```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]

speed = running_speed(position)
moving = speed > V_THRESH

# --- neuron curation 2: > 5 binarised events while running ---
events_running = trace[moving].sum(axis=0)
keep_cell = events_running > CELL_THRESH
trace = trace[:, keep_cell]

# immobility mask downsampled by majority vote over the 3 frames of each bin
moving_bins = moving[:n_use].reshape(n_bins, BIN_FRAMES).mean(axis=1) >= 0.5
```

iii. **Justification**

CONVERSION_NOTES.md Step 5, Key Decisions 4 and 5:

> 4. **Exclude immobility (speed <= 5 cm/s)** using the reference's velocity estimate
>    (`gaussian_filter1d(|diff(pos)|*30, sigma=5)` at 30 Hz, downsampled to 100 ms bins by majority
>    vote). This is the reference decoder's own curation rule, and during immobility CA1 activity
>    reflects replay/sharp-wave-ripple content rather than the animal's current location. The
>    provided decoder is memoryless (per-timepoint linear read-out of neural PCs + input), so
>    removing timepoints does not disturb it.
> 5. **Neuron curation = registered on the day AND > 5 events while running** (the reference rule).
>    This removes only ~0.7% of registered cell-sessions. No place-cell selection, per the paper.

Step 1 adds: "**Place-cell filtering is NOT applied** for decoding: the paper states the
spatial-coding quality results 'motivated the inclusion of **all** cells in subsequent analyses'."
The AI also verified the reference's own velocity comparison is algebraically equivalent to comparing
cm/s against 5 ("The reference computes … in *bin units* and compares with `v_thresh / bin_down`;
that is identical to computing the speed in cm/s and comparing with `v_thresh`").

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **Decisions**

- There is **no stimulus or behavioural alignment event**: the experiment is 40 min of continuous
  free foraging with no trial structure. The AI therefore defines the alignment event as the **start
  of each 1-minute trial block**, measured from the start of the continuous session.
- Imaging and behaviour were acquired by the same DAQ at 30 Hz and timestamp-aligned by the authors,
  so `trace[day]` and `position[day]` are frame-for-frame aligned; **no re-alignment is applied**.
  The AI verified this by rebuilding the authors' stored rate maps from raw `position` + `trace`
  (Pearson r = 1.000000).
- Metadata records `temporal_alignment_event`, `off_start = 0.0`, `off_end = 60.0`.

ii. **Code snippets**

```python
'temporal_alignment_event':
    'Start of each 1-minute trial block, measured from the start of the (continuous, 40 min) '
    'recording session. The task is continuous free foraging, so there is no stimulus or '
    'behavioural event to align to; imaging and behaviour were acquired on the same DAQ at '
    '30 Hz and are frame-aligned in the source data.',
'off_start': 0.0,
'off_end': TRIAL_SEC,
```

```python
starts = list(range(0, n_bins - BINS_PER_TRIAL + 1, BINS_PER_TRIAL))
bounds = [(s, s + BINS_PER_TRIAL) for s in starts]
```

iii. **Justification**

CONVERSION_NOTES.md Step 3: "Temporal alignment: imaging and behaviour are acquired **on the same DAQ
at 30 Hz and timestamp-aligned by the authors**; `position[d]` and `trace[d]` have identical frame
counts and are index-for-index aligned. There is no trial structure and no stimulus event in this
experiment — a session is 40 min of free foraging." Step 10 Check 3(c): "no re-alignment; verified by
rate-map reproduction (Check 2)".

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Decisions**

- Raw acquisition is 30 Hz (33.33 ms). The AI **rebins to 100 ms (10 Hz)**, i.e. 3 frames per bin,
  and reports `metadata['time_bin_size'] = 100.0`.
- Rebinning = `gaussian_filter1d(sigma = 3 frames)` then average-pool 3 frames — exactly
  `fit_decoder(temporal_bin_size=3)` at `fps=30`. Implemented as `reshape(..., 3, ...).mean(axis=1)`
  rather than `torch.AvgPool1d`, for speed.
- **Position is pooled over the same 3 frames** before discretisation, and the running mask is
  downsampled to the same bins by majority vote, so all three streams share the 100 ms grid.
- Trailing 1–2 frames that do not fill a 3-frame bin are dropped (≤ 67 ms of a 40 min session).

ii. **Code snippets**

```python
BIN_FRAMES = 3                                # fit_decoder(temporal_bin_size=3) -> 100 ms bins
TIME_BIN_MS = 1000.0 * BIN_FRAMES / FPS       # 100 ms
```

```python
n_bins = n_frames_raw // BIN_FRAMES
n_use = n_bins * BIN_FRAMES
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)

position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
moving_bins = moving[:n_use].reshape(n_bins, BIN_FRAMES).mean(axis=1) >= 0.5
```

iii. **Justification**

CONVERSION_NOTES.md Step 3 lists "Neural data time bin (decoding) | 3 frames = 100 ms |
`fit_decoder(..., temporal_bin_size=3)` with `fps=30`" and "Behaviour data time bin (decoding) | same
100 ms (same `AvgPool1d`)". Key Decision 3 (quoted under 2-b) adds the practical argument that
smoothing/pooling makes the neural values continuous rather than binary, which the provided validator
requires. Step 6 notes "smoothing and pooling done with reshape+`mean` instead of `torch.AvgPool1d`"
as a deliberate speed-up.

---

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. **Decisions**

- The `input` is derived from the per-day **`blocked`** field: the indices (0–8) of the 3 × 3
  partitions walled off that day, with the sentinel `[-1]` meaning "nothing blocked" (the open
  square).
- The per-day **`envs`** string (geometry name: square, o, t, u, rectangle, +, i, l, bit donut,
  glenn) is read as well, but used only (a) as a cross-check of `blocked`, and (b) as metadata in
  `session_info['environment']`.
- The AI copied the reference `get_env_mat` and asserts, for **every** session on every run, that the
  stored `blocked` equals `np.flatnonzero(np.flipud(get_env_mat(env)).ravel() == 0)`.

ii. **Code snippets**

```python
def get_env_mat(env):
    """Binary 3x3 matrix for a geometry name, 0 = omitted (blocked) partition. From utils.get_env_mat."""
    mats = {'square': [[1,1,1],[1,1,1],[1,1,1]], 'o': [[1,1,1],[1,0,1],[1,1,1]], ...}
    ...

def blocked_from_env(env):
    return np.flatnonzero(np.flipud(get_env_mat(env)).ravel() == 0)
```

```python
blocked = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
blocked = blocked[blocked >= 0].astype(int)     # -1 means "nothing blocked" (the square)
...
# cross-check the stored `blocked` list against the geometry name (reference get_env_mat)
expected = blocked_from_env(env)
assert np.array_equal(np.sort(blocked), expected), \
    f"{animal} day {day}: blocked {blocked} != {expected} implied by env '{env}'"
```

iii. **Justification**

CONVERSION_NOTES.md Step 2: "`blocked` | list of `n_days` | Indices (0-8) of blocked partitions in the
3x3 grid, `[[0,1,2],[3,4,5],[6,7,8]]`; `-1` = none blocked (square)", and "`blocked` vs
`get_env_mat`: verified `blocked` lists the flattened zeros of `np.flipud(get_env_mat(env))` for all
10 geometries". Step 6: "`blocked_from_env(env)` — re-derives the blocked list from the reference
`get_env_mat`; asserted against the stored `blocked` for every one of the 207 sessions (a built-in
consistency check that runs on every conversion)."

---

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. **Decisions**

- The geometry is encoded as a **9-dimensional binary vector**, one entry per arena partition,
  `1 = blocked`, `0 = open`. Indexing convention `bin = 3*ybin + xbin`, the same as the output labels,
  so `input[i]` refers to the same partition as output class `i`.
- The vector is **static per trial** (shape `(9,)`, no time axis), as the Decoder Task specifies;
  the same vector is copied into every trial of a session.
- `blocked = [-1]` (open square) → the all-zero vector.
- `input_names = ['blocked_x0y0', 'blocked_x1y0', …, 'blocked_x2y2']`.
- Observed statistics: range [0, 1], mean 2.314 blocked partitions per trial;
  `blocked_x1y2` (index 7) is constant 0 because the bottom-centre partition is open in all 10
  geometries — the AI kept it so the 9 inputs align one-to-one with the 9 output classes.

ii. **Code snippets**

```python
BIN_NAMES = [f"x{i % NBINS}y{i // NBINS}" for i in range(NBINS * NBINS)]
...
# --- decoder input: 9-dim binary geometry vector, 1 = partition blocked ---
geometry = np.zeros(NBINS * NBINS, dtype=np.float32)
geometry[blocked] = 1.0
...
input_trials.append(geometry.copy())
```

```python
'input_names': [f'blocked_{n}' for n in BIN_NAMES],
'input_description':
    'Nine binary values, one per arena partition: 1 = partition blocked off in this session, '
    '0 = partition open. Constant within a session (and therefore within a trial).',
```

iii. **Justification**

CONVERSION_NOTES.md Step 5, Key Decision 7: "**Input = 9-dim binary 'blocked' vector, static per
trial**, exactly as the Decoder Task specifies. It tells the decoder which of the 9 output classes
are reachable in that session." Step 10 Check 1 explains the constant `blocked_x1y0`/`x1y2` input:
"the union of blocked partitions over the 10 geometries is `{0,1,2,3,4,5,6,8}` — the bottom-centre
partition … is open in every geometry. It is kept in the input vector so that the 9 inputs align
one-to-one with the 9 output bins." Sanity Check 3 verified that "input vector flags exactly the
blocked partitions in every session".

---

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. **Decisions**

- `output` is derived **solely from the `position` field**: the DeepLabCut head-tracking `(T, 2)`
  array in centimetres, columns `[x, y]`, spanning exactly 0–75 cm across the dataset.
- The `blocked` field is additionally consulted, but only for the wall-snapping correction (4-c).

ii. **Code snippets**

```python
position = f[f['position'][day, 0]][()].astype(np.float64)   # (T, 2), head position in cm, [x, y]
```

iii. **Justification**

CONVERSION_NOTES.md Step 2: "`position` | `(n_days, 2, T)` | DeepLabCut head position in cm, `[x, y]`,
range exactly 0-75 cm". methods.txt: "Position data were generated from tracking the head with
DeepLabCut pose-estimation software." The AI verified the orientation and units by reproducing the
dataset's own stored occupancy maps `maps['sampling']` from `position` alone (sanity Check 1, max
diff 0.033 s = one frame).

---

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. **Decisions**

- Position is **average-pooled over the same 3 frames** used for the neural data, producing a 100 ms
  position stream, exactly as `fit_decoder` pools `behav` with the same `AvgPool1d(3, 3)`.
- No smoothing is applied to position (the reference does not smooth position either; the gaussian is
  applied to traces only).
- The pooled position is then discretised (4-c), the wall-snapping correction is applied, and the
  same running mask and trial boundaries as the neural data are used to select and split it.
- Output shape per trial is `(1, n_timepoints)` — one categorical variable, `output_names =
  ['position_bin']`, `output_values = [['x0y0', 'x1y0', …, 'x2y2']]`.
- A separate `running_speed` stream is derived from position (`gaussian_filter1d(‖diff(pos)‖·30,
  sigma=5)`), but it is used only as a curation mask, not exported.

ii. **Code snippets**

```python
# position: average-pooled over the same frames, then discretised
position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
pos_bin = position_to_bin(position_binned_cm).astype(np.int64)
n_snapped = int(np.isin(pos_bin, blocked).sum())
pos_bin = snap_to_open_bins(pos_bin, blocked)
```

```python
output_trials.append(pos_bin[idx][np.newaxis, :].copy())   # (1, n_timepoints)
...
'output_names': ['position_bin'],
'output_values': [list(BIN_NAMES)],
```

iii. **Justification**

CONVERSION_NOTES.md Step 5 variable-mapping table: "`position[day]` (2, T) @30 Hz → `output[session]
[trial]` (1, T_trial) | average-pool 3 frames → `bin = 3*clip(floor(y/25),0,2) +
clip(floor(x/25),0,2)` | `fit_decoder` (pooling + one-hot of the 2-D bin), `decode_position_within`
(`bin_down`)". Step 10 Check 3(d) confirms the reference does the same: "position `AvgPool1d(3,3)`
then `int()` … identical, implemented as reshape+`mean`".

---

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. **Decisions**

- The arena is divided into a **3 × 3 grid of 25 cm bins** (`ARENA_CM = 75`, `NBINS = 3`), giving
  9 classes. `bin = 3 * floor(y/25) + floor(x/25)`, clipped to [0, 2] per axis.
- The bin size is **fixed at 75 cm / 3 from the known arena size**, not derived from the per-session
  maximum position, because in walled geometries (e.g. `rectangle`, where `min(x) = 25`) the animal's
  range does not span the arena. The reference's `bin_down = (max + buffer)/n_bins` is computed over
  *all* days of an animal, which equals 75 cm anyway; using a per-session max would be wrong.
- `np.clip` replaces the reference's `1e-15` buffer, which is lost to float rounding at exactly
  75.0 cm.
- **Wall snapping**: samples that land inside a blocked partition (DeepLabCut occasionally tracks the
  head a centimetre past a 25 cm partition wall, and the 3-frame average of a boundary crossing can
  land just inside one) are moved to the nearest open partition by Euclidean distance in
  (xbin, ybin), ties going to the lower index. This affected 152 of 2,569,722 samples (0.006%).
- The grid indexing convention `3*ybin + xbin` was chosen to match the dataset's own
  `[[0,1,2],[3,4,5],[6,7,8]]` layout for `blocked`, and verified: occupancy is exactly zero in every
  blocked partition of all 207 sessions (480/480).

ii. **Code snippets**

```python
ARENA_CM = 75.0
NBINS = 3
BIN_SIZE_CM = ARENA_CM / NBINS      # 25 cm

def position_to_bin(position_cm):
    """Discretise (N, 2) positions in cm into the 3x3 grid index 3*ybin + xbin."""
    b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
    return NBINS * b[:, 1] + b[:, 0]
```

```python
def snap_to_open_bins(pos_bin, blocked):
    if blocked.size == 0:
        return pos_bin
    open_bins = np.setdiff1d(np.arange(NBINS * NBINS), blocked)
    coords = np.stack([np.arange(NBINS * NBINS) % NBINS, np.arange(NBINS * NBINS) // NBINS], axis=1)
    lut = np.arange(NBINS * NBINS)
    for b in blocked:
        d = np.linalg.norm(coords[open_bins] - coords[b], axis=1)
        lut[b] = open_bins[np.argmin(d)]
    return lut[pos_bin]
```

iii. **Justification**

CONVERSION_NOTES.md Step 4 discrepancy table: "Use a **fixed 25 cm bin size from the 75 cm arena**,
not a per-session max. Verified: reproducing the stored occupancy map (`maps['sampling']`) requires
`bin = 75/15`; a per-session max is off by up to 0.77 s of occupancy, while 75/15 matches to 1 frame
(0.033 s)." For the snapping, the in-code docstring says: "The reference decoder does exactly this
cleaning: `decode_position_within` snaps both the true and the predicted bin to the nearest entry of
`temp_maps`, the set of spatial bins with valid rate-map values, before scoring." Step 10 "Issues
Found and Resolved" records that this was discovered by the AI's own sanity check failing (127 of
2,569,722 samples in blocked partitions across 4 sessions), traced to DeepLabCut tracking
`(31.4, 25.1)` cm past a wall, and fixed.

---

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. **Decisions**

- `position` and `trace` are frame-for-frame aligned at 30 Hz in the source data (same DAQ, same
  timestamps); no re-alignment, lag, or interpolation is applied.
- Both streams are pooled over the **same** 3-frame boundaries (`[:n_use].reshape(n_bins, 3, …)`),
  masked by the **same** `moving_bins` index vector, and cut at the **same** trial boundaries, so
  `neural[trial]` and `output[trial]` are element-wise aligned and have identical `n_timepoints`.
- Alignment was verified two ways: (a) rebuilding the authors' stored rate maps from raw
  `position` + `trace` gives Pearson r = 1.000000; (b) the `--show-processing` figure overlays the
  continuous `x/25`, `y/25` traces on the integer output bin and shows the bin stepping exactly at
  the crossings with no lag.

ii. **Code snippets**

```python
for (s, e) in bounds:
    idx = np.flatnonzero(moving_bins[s:e]) + s          # one index vector for both streams
    if idx.size < MIN_TRIAL_BINS:
        continue
    neural_trials.append(np.ascontiguousarray(neural[:, idx]))
    output_trials.append(pos_bin[idx][np.newaxis, :].copy())
```

```python
assert position.shape == (n_frames_raw, 2), "position and trace must have the same number of frames"
```

iii. **Justification**

CONVERSION_NOTES.md Step 3: "`position[d]` and `trace[d]` have identical frame counts and are
index-for-index aligned." Step 10 Check 2: "Rate maps rebuilt from raw `position`+`trace` match the
stored `maps['unsmoothed']`, cells 0/63/143 — PASS, Pearson r = 1.000000 — proves frame-by-frame
neural/behaviour alignment." Step 12: "**Temporal alignment**: `processing_*.png` panel 5 overlays
the continuous `x/25`, `y/25` traces on the integer output bin; the bin steps exactly at the
crossings, no lag."

---

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. **Decisions**

| Issue | Handling |
|---|---|
| Cells not registered on a session (all-NaN trace column) | dropped per session; an assertion checks that a registered cell never has partial NaNs |
| `blocked = -1` (open square) | filtered out by `blocked[blocked >= 0]` → empty list → all-zero input vector |
| Position exactly at the 75.0 cm arena edge | `np.clip(..., 0, NBINS-1)` instead of the reference's `1e-15` buffer, which is lost to float rounding at 75.0 |
| Frame 0 has no velocity estimate | `speed[0] = 0`, so frame 0 is always excluded — matches the reference's `vel_idx[d, 1:] = ...` |
| Session length not a multiple of 3 frames | trailing 1–2 frames dropped (≤ 67 ms of 40 min) |
| Session length not a multiple of 600 bins | trailing partial block kept as a shorter trial if ≥ 10 s, else dropped |
| Trial with almost no running | dropped if < 30 bins (3 s) remain |
| Session with < 2 usable trials | dropped (validator requirement); never occurs |
| Head tracked past a partition wall (impossible position) | 152 samples (0.006%) snapped to the nearest open bin, mirroring the reference's snap-to-`temp_maps` cleaning |
| Animal QLAK-CA1-51 has 21 days, not 31 | handled generically, `n_days` read from the file |
| Geometry name / `blocked` inconsistency | asserted equal for every session on every run |

ii. **Code snippets**

```python
registered = ~np.all(np.isnan(trace), axis=0)
trace = trace[:, registered]
assert not np.any(np.isnan(trace)), "a registered cell has partial NaNs -- unexpected"
```

```python
blocked = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
blocked = blocked[blocked >= 0].astype(int)     # -1 means "nothing blocked" (the square)
```

```python
b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
...
speed = np.zeros(position.shape[0]); speed[1:] = gaussian_filter1d(...)   # frame 0 never running
...
n_bins = n_frames_raw // BIN_FRAMES          # trailing 1-2 frames dropped
...
if n_bins - tail_start >= MIN_TAIL_BINS: bounds.append((tail_start, n_bins))
...
if len(neural_trials) < 2: return None
```

iii. **Justification**

CONVERSION_NOTES.md Step 10 Check 5 tabulates every one of these cases (reproduced above), and the
"Issues Found and Resolved" section documents the one real data defect found: "First full conversion:
sanity Check 3 failed — 4 sessions had 1-120 timepoints in a blocked partition (127 of 2,569,722
samples). Traced to DeepLabCut tracking the head slightly past a 25 cm partition wall in the raw data
(e.g. `(31.4, 25.1)` cm in QLAK-CA1-51 day 4), plus one case created by the 3-frame average of a
boundary crossing. **Fix**: `snap_to_open_bins`, the analogue of the reference's snap-to-`temp_maps`
cleaning in `decode_position_within`."

---

## 6-a. What are the most time-consuming steps of the code?

i. **Decisions / findings**

- The AI instrumented the script with per-session, per-animal and per-phase timers printed to stdout.
- Measured full run: **44.1 s total** = 39.8 s processing (7 parallel animal workers) + 4.3 s writing
  the 3.44 GB pickle.
- Per session, 0.5–1.5 s, scaling with the number of registered cells (112 → 0.51 s, 562 → 1.45 s).
  The dominant costs are (a) the HDF5 read + decompression of the `(T, n_cells)` `trace` array
  (I/O bound; up to 72,000 × 950 floats) and (b) `gaussian_filter1d` over that array.
- The AI identified the joblib whole-animal load as the biggest avoidable cost and eliminated it:
  "Loading a whole animal from joblib materialises 9-25 GB of float64 in RAM"; HDF5 lazy per-day
  reads keep memory at ~0.3 GB per worker.
- Estimated serial time 207 × ~2.0 s ≈ 7 min, reduced to ~1.5 min by the 7 workers; the actual run
  beat the estimate.

ii. **Code snippets**

```python
t_start = time.time()
...
    t0 = time.time()
    position, trace, env, blocked = read_session(f, day)
    ...
    print(f"  [{animal} day {day:2d}] env={env:<10s} cells {sess['n_cells_total']}"
          f" -> reg {sess['n_registered']} -> kept {sess['n_neurons']}"
          f" | {sess['n_bins']} bins, {sess['frac_moving']*100:.1f}% running"
          f" | {len(sess['neural'])} trials | {time.time()-t0:.2f}s", flush=True)
print(f"[{animal}] {len(sessions)} sessions in {time.time()-t_start:.1f}s", flush=True)
```

```python
t0 = time.time()
with open(args.outfile, 'wb') as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t0:.1f}s")
print(f"Total time {time.time()-t_all:.1f}s", flush=True)
```

iii. **Justification**

CONVERSION_NOTES.md Step 6 "Code inefficiencies identified" / "Code speedups added", and the Step 7
timing table: "read + process one session | 1.7-2.1 s | 207 x ~2.0 s = ~7 min serial"; "with 7 animal
workers (longest animal = 31 sessions) | **~1.5 min**"; "**Full conversion estimate** | **< 5 min**
(well under the 15 min budget; no further optimisation needed)". Step 9: "**44 s wall clock** (7
worker processes; 39.8 s processing + 4.3 s pickling), well inside the estimate from Step 7."

---

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. **Decisions / findings**

The AI deliberately vectorised the operations the reference code performs with Python loops, and the
loops that remain are all negligible:

- **Vectorised away**: the reference's `fit_decoder`/`test_decoder` build the one-hot position
  embedding with a per-sample Python loop (`for i in range(n_samples)`), and `get_rate_maps` bins
  frame by frame. The AI replaced both with `np.floor`/`np.clip`/integer arithmetic over the whole
  session. `torch.AvgPool1d` was replaced by `reshape(...).mean(axis=1)`.
- **Remaining loops, all inconsequential**:
  - `for day in day_list` (≤ 31 iterations/animal) — I/O bound, parallelised across animals instead.
  - `for (s, e) in bounds` (≈ 40 iterations/session) — one fancy-index slice each.
  - `for b in blocked` inside `snap_to_open_bins` (≤ 4 iterations) — builds a 9-entry lookup table
    that is then applied vectorised (`lut[pos_bin]`).
  - `_h5_str`'s `''.join(chr(c) for c in ...)` over a short geometry name.
- The one remaining *parallelism* (not vectorisation) opportunity the AI did not take: sessions are
  processed serially within an animal and only 7 of the 128 available CPUs are used. At 44 s total
  this was not worth pursuing.

ii. **Code snippets**

```python
# vectorised replacement for the reference's per-frame one-hot loop
def position_to_bin(position_cm):
    b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
    return NBINS * b[:, 1] + b[:, 0]

# vectorised replacement for torch.AvgPool1d
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)
position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
moving_bins = moving[:n_use].reshape(n_bins, BIN_FRAMES).mean(axis=1) >= 0.5
```

```python
# the only per-element loop left, over <= 4 blocked partitions, then applied vectorised
for b in blocked:
    d = np.linalg.norm(coords[open_bins] - coords[b], axis=1)
    lut[b] = open_bins[np.argmin(d)]
return lut[pos_bin]
```

iii. **Justification**

CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: … Naive per-frame python loops (as in the
reference `get_rate_maps` / `fit_decoder` one-hot construction) are slow." "Code speedups added: …
`trace` cast to float32 at read time; smoothing and pooling done with reshape+`mean` instead of
`torch.AvgPool1d`. Position binning and one-hot-free integer bin indices computed vectorised
(`np.floor`/`np.clip`), no python loop. 7 animal-level worker processes (`--nproc`, default 7)."

---

## 6-c. What processing does the code repeat multiple times?

i. **Decisions / findings**

Very little repetition; what exists is cheap:

- `~np.all(np.isnan(trace), axis=0)` is computed twice when `--show-processing` is active — once
  inside `process_session` and again in `process_animal` to build the raw-trace panel of the figure.
  Only for the ≤ 2 plotted sessions.
- `get_env_mat` / `blocked_from_env` rebuild the 10-entry geometry dictionary on every session (207
  times) purely for the consistency assertion — microseconds each.
- The running mask is derived twice in different resolutions (`moving` at 30 Hz for the cell filter,
  `moving_bins` at 10 Hz for the timepoint filter); both are needed and each is computed once.
- The 9-dim `geometry` vector is `.copy()`-ed once per trial (~40 copies of 9 floats per session)
  rather than shared — a deliberate safety choice, unlike the reference which shares one object
  (`[blocked] * len(neural_trials)`).
- Across the whole workflow (not within `convert_data.py`), the conversion was run twice — once
  before and once after the wall-snapping fix — and the sample conversion re-run; the AI documented
  this as the required iteration protocol.

ii. **Code snippets**

```python
# in process_session
registered = ~np.all(np.isnan(trace), axis=0)

# in process_animal, for the plot only -- same computation again
plot_processing(sess, position, trace[:, ~np.all(np.isnan(trace), axis=0)],
                f"processing_{sess['session_id']}.png")
```

```python
def blocked_from_env(env):          # rebuilds the 10-geometry dict each call, 207 calls
    return np.flatnonzero(np.flipud(get_env_mat(env)).ravel() == 0)
...
expected = blocked_from_env(env)
assert np.array_equal(np.sort(blocked), expected), ...
```

iii. **Justification**

The AI did not list repeated computation as a concern in CONVERSION_NOTES.md; the only relevant
statements are Step 6's "Code inefficiencies identified" (which names whole-animal joblib loads,
float64 filtering, and per-frame Python loops) and the note that the `blocked_from_env` assertion is
"a built-in consistency check that runs on every conversion" — i.e. the repetition is intentional,
traded for per-run validation of all 207 sessions.

---

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **Decisions / findings**

- **Smoothing and pooling are applied to all frames, including the ~48% later discarded** by the
  running filter. This is necessary rather than wasteful: the AI deliberately smooths the contiguous
  30 Hz session so the gaussian kernel does not bridge temporal discontinuities.
- `pos_bin` and `position_binned_cm` are likewise computed for every bin, then subset. Negligible.
- The per-session `blocked_from_env` assertion is pure validation and contributes nothing to the
  output — kept on purpose.
- Bookkeeping fields (`n_registered`, `n_cells_total`, `frac_moving`, `n_snapped`, `trial_bounds`,
  `env`) are computed for every session; `trial_bounds` is used only for the diagnostic figure, the
  rest are surfaced in `metadata['session_info']` and in the sanity checks, so they are not dead.
- `summarise()` concatenates every output array in the dataset to print the class distribution — a
  full extra pass over 2.57 M labels, used only for the log.
- The `_plot` dict (raw position, speed, masks, full `neural`) is retained on the session object
  until the figure is written, then `del`-ed; only for the ≤ 2 plotted sessions.
- One genuinely dead fragment survives in a print statement:
  `f"{f' | {sess[chr(39)+chr(39)]}' if False else ''}"` always evaluates to the empty string.
- Not "processing", but worth noting: the 3.44 GB pickle stores `float32` smoothed rates for every
  trial; the running-filter and 100 ms binning already shrink it ~6× relative to raw 30 Hz.

ii. **Code snippets**

```python
# computed for the whole session, then subset to running bins only
smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)
pos_bin = position_to_bin(position_binned_cm).astype(np.int64)
...
idx = np.flatnonzero(moving_bins[s:e]) + s
neural_trials.append(np.ascontiguousarray(neural[:, idx]))
```

```python
# dead fragment in the progress print
f"{f' | {sess[chr(39)+chr(39)]}' if False else ''}"
```

```python
# validation-only work, executed on every session of every run
expected = blocked_from_env(env)
assert np.array_equal(np.sort(blocked), expected), ...
```

iii. **Justification**

CONVERSION_NOTES.md Step 5, Key Decision 3 explains why smoothing precedes (rather than follows) the
discard of immobility frames: "the reference smooths **after** discarding immobility frames, which
convolves across temporal discontinuities. Here smoothing and pooling are done on the intact 30 Hz
session and the speed criterion is applied afterwards, which is strictly more correct and otherwise
identical." Step 6 lists the assertion as an intentional built-in check. The AI did not identify any
processing as wasteful and did not document the dead print fragment.
