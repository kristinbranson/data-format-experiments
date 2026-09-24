# Decisions

> **Note on the prompt the AI actually received.** The trajectory (`/logs/agent/trajectory.json`, step 1)
> shows the AI's "Decoder Task" section read:
> *"Decode information regarding animal motion from the neural activities recorded from mouse barrel cortex."*
> / Inputs: *"Time elapsed from the beginning of the experiment. Time-varying."*
> / Outputs: *"Motion energy, normalized and discretized into five equal-percentile bins. Time-varying."*
>
> It did **not** contain the sentences "Split sessions into 60-second trials" or "selected per session"
> that appear in `/tests/instruction_reference.md`. Where this matters (1-d, 4-c) it is called out below.

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. **Decisions**

The AI discovers the dataset by walking the `data/` directory tree: subjects are directories whose name
starts with `jm`, sessions are subdirectories of a subject whose first four characters are digits (the
`YYYY-MM-DD_a` date folders). Both levels are sorted, giving a deterministic order. All 6 subjects and all
41 sessions found on disk are loaded; nothing is excluded.

For every session it loads seven arrays in one call: from `suite2p/plane0/` — `F.npy`, `Fneu.npy`,
`ops.npy`, `stat.npy`; from `move_deve/` — `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`.
Only `F`, `Fneu`, `ops` (for `neucoeff`, `win_baseline`, `prctile_baseline`, `fs`) and
`motion_energy_glob` are actually used downstream; `stat`, `tstamps` and `interframe_int` are loaded and
discarded. `iscell.npy` and `spks.npy` are not loaded at all.

Loading is a single sequential pass over sessions; everything is held in memory (as binned blocks) before
the motion-energy discretisation pass, because that pass needs the pooled distribution.

ii. **Code snippets**

```python
def discover_subjects(data_root: Path):
    subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
    sessions_by_subject = {}
    for subj in subjects:
        sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        sessions_by_subject[subj.name] = sessions
    return subjects, sessions_by_subject


def load_session(session_dir: Path):
    pl0 = session_dir / 'suite2p' / 'plane0'
    move = session_dir / 'move_deve'
    F = np.load(pl0 / 'F.npy', allow_pickle=True)
    Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
    ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
    stat = np.load(pl0 / 'stat.npy', allow_pickle=True)
    motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
    tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
    interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
    return F, Fneu, ops, stat, motion, tstamps, interframe
```

```python
selected = []
for subj in subject_names:
    for sess in sessions_by_subject[subj]:
        selected.append((subj, sess))
if args.sample:
    selected = selected[:2]
```

iii. **Justification**

CONVERSION_NOTES.md Step 2 records the structure the AI found: *"`data/` contains 6 subject folders
(`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) … Each subject contains dated session folders (41
sessions total across all subjects). Each session contains `suite2p/plane0/` and `move_deve/`."*
Step 5 Key Decision 6 states *"Use one session per recording day and one subject per mouse: Direct mapping
from native organization."* The data README, which the AI read (trajectory step 15), documents exactly this
convention.

---

## 1-b. How are the data split into subjects?

i. **Decisions**

One subject per top-level `jm*` directory, sorted alphabetically: `['jm031','jm032','jm038','jm039',
'jm040','jm046']`. This list is written verbatim to `data['subjects']`, and every session records its
subject through `subject_names.index(item['subject'])` in `data['subject_idx']`. Note that the full
six-element subject list is written even in `--sample` mode, where only one subject actually contributes
sessions.

ii. **Code snippets**

```python
subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith('jm')])
...
subject_names = [p.name for p in subjects]
...
'subjects': subject_names,
...
data['subject_idx'].append(subject_names.index(item['subject']))
...
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. **Justification**

CONVERSION_NOTES.md Step 4 records the cross-check: *"Subjects | 6 subjects in data | 6 subject folders …
| 6 mice | Match"*, against the methods statement *"we used a full dataset of 6 mice imaged daily for a
minimum of 6 consecutive days"*. The data README states the folder-per-mouse convention explicitly
(`jm031` = mouse A … `jm046` = mouse F).

---

## 1-c. How are the data split into sessions?

i. **Decisions**

One session per dated subdirectory inside a subject folder, sorted alphabetically (which for
`YYYY-MM-DD_a` names is chronological). The AI adds a name filter — `p.name[:4].isdigit()` — so only
date-named folders are treated as sessions. This yields 7/7/7/7/6/7 = 41 sessions, matching the raw data.
Session boundaries are never crossed: each session becomes one entry in `data['neural']`,
`data['input']`, `data['output']`, `data['brain_region_idx']` and `data['subject_idx']`.

ii. **Code snippets**

```python
sessions = sorted([p for p in subj.iterdir() if p.is_dir() and p.name[:4].isdigit()])
```

```python
for item in prepared:
    sess_neural, sess_input, sess_output = [], [], []
    ...
    if len(sess_neural) >= 2:
        data['neural'].append(sess_neural)
        data['input'].append(sess_input)
        data['output'].append(sess_output)
        data['subject_idx'].append(subject_names.index(item['subject']))
        data['brain_region_idx'].append(np.zeros(item['n_neurons'], dtype=np.int64))
```

iii. **Justification**

Step 2 of CONVERSION_NOTES.md: *"Each subject contains dated session folders (41 sessions total)."*
Step 4 confirms against the paper: *"Sessions | Minimum 6 consecutive daily sessions | 41 sessions total;
per-subject counts 7,7,7,7,6,7 | Match."* Step 5 Key Decision 6: *"Use one session per recording day."*
The `isdigit()` guard also excludes the per-subject `ground_truth.csv` files the AI found in Step 2.

---

## 1-d. How are the data split into trials?

i. **Decisions**

The recordings are continuous with no native trial structure, so the AI defines artificial trials as
**consecutive, non-overlapping 120-second (2-minute) blocks**, i.e. `BLOCK_SECONDS = 120.0`,
`BLOCK_FRAMES = 3600`, `BLOCK_BINS = 360` bins after 10-frame binning. Blocking is applied *after*
10-frame binning. Neural and motion streams are first truncated to their common length
(`n_bins = min(...)`), then `n_blocks = n_bins // BLOCK_BINS` blocks are cut and the remainder (< 2 min)
is discarded. This gives 10 trials for the 20-minute sessions (jm031, jm032) and 15 for the 30-minute
sessions (jm038, jm039, jm040, jm046) — 545 trials total, all of length exactly 360 bins.

**This differs from the human reference, which uses 60-second trials (180 bins, 20/30 trials per session,
~1090 trials).** The AI's prompt did not contain the "Split sessions into 60-second trials" sentence that
appears in `/tests/instruction_reference.md`; it chose 120 s from the paper's own analysis protocol.

ii. **Code snippets**

```python
FS = 30.0
BIN_FRAMES = 10
BLOCK_SECONDS = 120.0
BLOCK_FRAMES = int(FS * BLOCK_SECONDS)     # 3600
BLOCK_BINS = BLOCK_FRAMES // BIN_FRAMES    # 360
```

```python
def session_to_blocks(neural_binned, motion_binned):
    n_bins = min(neural_binned.shape[1], motion_binned.shape[0])
    neural_binned = neural_binned[:, :n_bins]
    motion_binned = motion_binned[:n_bins]
    n_blocks = n_bins // BLOCK_BINS
    neural_blocks, motion_blocks, time_blocks = [], [], []
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
        motion_blocks.append(motion_binned[s:e].astype(np.float32))
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
    return neural_blocks, motion_blocks, time_blocks
```

iii. **Justification**

CONVERSION_NOTES.md Step 5 Key Decision 1: *"**Trials = consecutive 2-minute blocks**: Raw recordings are
continuous and the paper's decoding uses consecutive 2-minute blocks, so these are the natural trial units
for the target format."* This is grounded in a direct quote the AI extracted from `methods.txt`:
*"We used 5 fold splits for both the inner and outer loops, splits were done on consecutive 2 minute
blocks of the recording."* Step 3 also records the derived expectation *"10 blocks/session if using
20-minute sessions and 2-minute blocks."*

---

## 1-e. How are trials filtered based on quality controls?

i. **Decisions**

Two defensive filters, both operating on missing motion-energy data rather than on neural quality:

1. **Per-trial**: a block is dropped if fewer than `max(10, 0.8 * 360) = 288` of its 360 bins have a
   finite motion-energy value (i.e. >20 % of the trial's behavioural labels are missing).
2. **Per-session**: a session is written out only if at least 2 trials survive — enforcing the format
   requirement of ≥2 trials per session.

In the delivered full conversion neither filter ever fires: all 41 sessions and all 545 blocks are
retained (`verification_full_out.txt`: 41 sessions, trials per session `[10 ×14, 15 ×27]`). The largest
frame deficit in the dataset is 148 frames ≈ 14.8 bins out of 360 (4 %), well under the 20 % threshold.
No trial is filtered on neural grounds, and no trial is dropped for motion artefacts, saturation, etc.

ii. **Code snippets**

```python
y = np.digitize(mb, edges, right=False).astype(np.int64)
y[np.isnan(mb)] = -1
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
...
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. **Justification**

CONVERSION_NOTES.md gives no explicit rationale for the 80 % threshold; it appears in Step 10 only through
the history of the alignment bug: *"Initial issue: motion alignment incorrectly interpreted `tstamps.npy`
as frame indices, dropping 9 sessions. Resolution: align motion by sample order and pad/truncate to
imaging length."* The filters were introduced as a guard against blocks whose behavioural labels are
largely undefined, and the `>= 2` check is directly traceable to the target-format requirement *"There
needs to be at least two trials within each session in order to evaluate the decoder performance."*
The Step 3 curation notes state that no neuron-level trial curation is implied by the paper, since all
released ROIs are already Suite2p-curated.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. **Decisions**

`neural` is derived from `suite2p/plane0/F.npy` (raw ROI fluorescence, `n_neurons × n_frames`) and
`suite2p/plane0/Fneu.npy` (neuropil fluorescence, same shape), plus scalar parameters read from
`suite2p/plane0/ops.npy` (`neucoeff`, `win_baseline`, `prctile_baseline`, `fs`). `spks.npy` (deconvolved
spikes) is deliberately not used. `stat.npy` is loaded but unused. The AI relies on the data README's
statement that the released `F.npy` already contains only the Track2p-tracked cells, row-matched across
days, so no cross-session neuron matching is performed.

ii. **Code snippets**

```python
F = np.load(pl0 / 'F.npy', allow_pickle=True)
Fneu = np.load(pl0 / 'Fneu.npy', allow_pickle=True)
ops = np.load(pl0 / 'ops.npy', allow_pickle=True).item()
...
neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
```

iii. **Justification**

CONVERSION_NOTES.md Step 5 variable mapping: *"`suite2p/plane0/F.npy` + `Fneu.npy` + `ops.npy` for tracked
cells across days → neural … Data README says rows are already tracked and matched across all days within
each subject; notebook notes `F.npy` are raw fluorescence traces and dF/F should be computed for proper
analysis."* Trajectory step 23: *"the provided helper code simply loads and visualizes `F.npy` traces as
raw fluorescence … this means our conversion should not directly use raw `F.npy` … Instead, we should
derive a dF/F-like signal from Suite2p outputs."* This matches the methods text: *"We used baseline
corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent
analyses."*

---

## 2-b. How is the `neural` data processed?

i. **Decisions**

A hand-written dF/F, in four steps per session:

1. **Neuropil subtraction**: `Fcorr = F − neucoeff·Fneu`, with `neucoeff` read from `ops` (0.7 everywhere).
2. **Baseline estimation**: for each neuron independently, the trace is smoothed with a reflect-padded
   `win = round(win_baseline·fs) = 1800`-frame (60 s) boxcar moving average, then the **8th percentile of
   the whole smoothed trace** is taken as a **single scalar baseline for that neuron**.
3. **Normalisation**: `dff = (Fcorr − baseline) / baseline`, with `baseline` floored at `1e-3`.
4. **Binning**: 10-frame non-overlapping mean (see 2-e), cast to `float32`.

Two properties distinguish this from the reference, which calls suite2p's own
`dcnv.preprocess(baseline='maximin', win_baseline=60, sig_baseline=10, prctile_baseline=8)`:

* The baseline is **constant in time**, so slow drift over the 20–30 min recording is *not* removed. The
  suite2p `maximin` baseline is a rolling (gaussian-filter → minimum-filter → maximum-filter) estimate
  that does remove it.
* The result is **divided** by the baseline (a true dF/F) rather than only baseline-subtracted.

The `np.maximum(baseline, 1e-3)` floor is an unguarded edge case. Neuropil-subtracted baselines are
genuinely negative for a small number of ROIs; for those neurons the divisor collapses from ~ −10² to
`1e-3` and the output explodes. I verified this in the delivered `converted_data.pkl`: **35 of 20 445
neurons, spread over 18 of 41 sessions, contain |dF/F| > 100, with a global maximum of 5.7 × 10⁵** — five
orders of magnitude outside any plausible dF/F range. Neither the verifier nor CONVERSION_NOTES.md flags
this.

ii. **Code snippets**

```python
def moving_average_reflect(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32, copy=True)
    pad = win // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode='reflect')
    kernel = np.ones(win, dtype=np.float32) / win
    y = np.convolve(xp, kernel, mode='valid')
    return y[: x.shape[0]].astype(np.float32)


def compute_dff(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> np.ndarray:
    neucoeff = float(ops.get('neucoeff', NEUCOEFF_DEFAULT))
    Fcorr = F.astype(np.float32) - neucoeff * Fneu.astype(np.float32)
    win_seconds = float(ops.get('win_baseline', 60.0))
    win = max(1, int(round(win_seconds * float(ops.get('fs', FS)))))
    prct = float(ops.get('prctile_baseline', 8.0))
    baseline = np.empty_like(Fcorr, dtype=np.float32)
    for i in range(Fcorr.shape[0]):
        smooth = moving_average_reflect(Fcorr[i], win)
        base = np.percentile(smooth, prct)
        baseline[i] = base
    baseline = np.maximum(baseline, 1e-3)
    dff = (Fcorr - baseline) / baseline
    return dff.astype(np.float32)
```

iii. **Justification**

CONVERSION_NOTES.md Step 3: *"Neural traces for downstream analyses are baseline-corrected Suite2p
fluorescence traces (dF/F using default Suite2p parameters)."* Step 5 mapping: *"Compute
Suite2p-consistent neuropil-corrected / baseline-corrected fluorescence (dF/F-like signal) from released
raw fluorescence traces."* The AI itself flagged the gap in Step 6 — *"Potential inefficiency: baseline
computation loops over neurons and uses a simplified approximation instead of Suite2p internals"* — and in
trajectory step 25 — *"the dF/F computation is only an approximation and may not exactly match Suite2p's
baseline-corrected fluorescence"* — but never returned to it. The trajectory contains **zero** mentions of
`dcnv` or `suite2p.extraction`; suite2p is installed and importable in this environment, so the
approximation was avoidable.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. **Decisions**

No neuron-level filtering at all. Every row of every `F.npy` is kept, so `n_neurons` per session is
221/370/685/746/541/435 and the total is 20 445 — identical to the raw files. `iscell.npy` is not even
loaded by `convert_data.py`. There is also no filtering on ΔF/F range, SNR, or NaN content, which is why
the 35 blown-up neurons described in 2-b survive into the output.

`brain_region_idx` is a zero vector of length `n_neurons` for every session, with
`brain_regions = ['barrel cortex L2/3']`.

ii. **Code snippets**

```python
data['brain_region_idx'].append(np.zeros(item['n_neurons'], dtype=np.int64))
```

```python
'brain_regions': ['barrel cortex L2/3'],
```

(There is no filtering statement to quote — `compute_dff` operates on all rows of `F`.)

iii. **Justification**

CONVERSION_NOTES.md Step 3 records the paper's rule — *"Keep ROIs classified as cells by Suite2p using
default threshold: `iscell` probability > 0.5"* (methods: *"We considered all ROIs above the default
threshold of 0.5 as true cells"*) — and Step 4 records that it is already satisfied by the released data:
*"Cell curation | Suite2p `iscell` probability > 0.5 | All observed `iscell[:,0] == 1` … | Match."*
Trajectory step 13: *"all cells currently have `iscell[:,0] == 1` in these inspected data, so no further
exclusion is implied."* I confirmed this independently: across all 41 sessions, 0 of 20 445 ROIs have
`iscell[:,0] != 1` and the minimum classifier probability is 0.5003. Step 5 Key Decision 2: *"Use the
released tracked-neuron matrices as provided."* The region label comes from methods: *"All recordings were
performed in layer 2/3 … of mouse barrel cortex."*

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **Decisions**

There is no experimental event to align to — the recordings are continuous spontaneous-behaviour sessions.
The AI therefore aligns each trial to the **start of its own 2-minute block**, which are contiguous
segments counted from the start of the session. Metadata records
`temporal_alignment_event = 'start of each 2-minute continuous recording block'`, `off_start = 0.0`,
`off_end = 120.0`. Every trial in every session has exactly 360 bins, so neural, input and output share
one index grid by construction (`verification_full_out.txt`: `T: mean 360.00, min 360, max 360`).

ii. **Code snippets**

```python
'metadata': {
    'task_description': 'Decode spontaneous-motion energy from longitudinal barrel-cortex calcium '
                        'imaging using continuous recordings segmented into 2-minute blocks.',
    'time_bin_size': 1000.0 * BIN_FRAMES / FS,
    'temporal_alignment_event': 'start of each 2-minute continuous recording block',
    'off_start': 0.0,
    'off_end': BLOCK_SECONDS,
    ...
}
```

```python
for b in range(n_blocks):
    s = b * BLOCK_BINS
    e = (b + 1) * BLOCK_BINS
    neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
    motion_blocks.append(motion_binned[s:e].astype(np.float32))
```

iii. **Justification**

CONVERSION_NOTES.md Step 2: *"Trials (total) | Not natively trial-structured; sessions are continuous
recordings"*; Step 4: *"Trial structure | Continuous recordings split into 2-min blocks for decoding |
Continuous recordings; no native trial markers observed | Match."* With no stimulus event, block start is
the only meaningful reference, and `off_start`/`off_end` of 0/120 s are self-consistent with that choice.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Decisions**

Yes — rebinning from the native 30 Hz to **3 Hz**. Both the neural traces and the motion-energy trace are
averaged over non-overlapping bins of `BIN_FRAMES = 10` consecutive frames, giving a
**333.33 ms** time bin (`time_bin_size: 333.3333333333333` in metadata, and a 360-bin trial = 120 s).
Any tail shorter than one full bin is dropped. The neural binning is a plain reshape-mean; the motion
binning is a NaN-aware mean (a bin's value is the mean of its finite entries, NaN only if all 10 frames
are missing). Binning is applied **before** discretisation of the motion energy, which is correct — class
labels cannot be averaged. The bin size is identical for all trials and all sessions.

ii. **Code snippets**

```python
def bin_neural(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (x.shape[1] // bin_frames) * bin_frames
    x = x[:, :n]
    return x.reshape(x.shape[0], -1, bin_frames).mean(axis=2).astype(np.float32)


def nanbin_mean_1d(x: np.ndarray, bin_frames: int) -> np.ndarray:
    n = (len(x) // bin_frames) * bin_frames
    x = x[:n].reshape(-1, bin_frames)
    valid = np.isfinite(x)
    sums = np.where(valid, x, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    out = np.full(x.shape[0], np.nan, dtype=np.float32)
    nz = counts > 0
    out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    return out
```

```python
neural_binned = bin_neural(dff, BIN_FRAMES)
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
...
'time_bin_size': 1000.0 * BIN_FRAMES / FS,
```

iii. **Justification**

CONVERSION_NOTES.md Step 5 Key Decision 3: *"**Use 10-frame averaging before decoding**: Methods
explicitly state that both dF/F and behavior traces were slightly denoised by averaging in bins of 10
consecutive timestamps."* The methods quote the AI extracted is *"For all decoding analysis we slightly
denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."*
Step 3 records *"30 Hz acquisition; decoding uses 10-frame averaging (~0.333 s bins)"* for both streams.

---

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. **Decisions**

No raw data variable. Time is synthesised from the bin index and the constant frame rate:
`t = bin_index × BIN_FRAMES / FS` = `bin_index × 1/3 s`. `tstamps.npy` — the only actual timestamp array
in the dataset — is loaded but not used for this (or anything else). There is exactly one input,
`input_names = ['time_elapsed_s']`, shape `(1, 360)` per trial.

ii. **Code snippets**

```python
time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
...
'input_names': ['time_elapsed_s'],
```

iii. **Justification**

CONVERSION_NOTES.md Step 5 mapping row: *"elapsed time within each 2-minute block → input[0] | Create a
1 × T time series in seconds after 10-frame binning (`t = bin_index * 10 / 30`) | N/A (constructed to
satisfy decoder input specification)."* Step 4 confirms the frame rate is constant and verified:
*"Imaging rate | 30 Hz | `ops.npy['fs'] = 30` in inspected sessions | 30 Hz | Match."*

---

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. **Decisions**

The vector is **reset to zero at the start of every trial**. `time_blocks` appends the *same*
`np.arange(360) * (1/3)` array for every block `b`, with no `+ b * BLOCK_BINS` offset. The delivered input
therefore runs `[0.0 … 119.67]` in every one of the 545 trials — I confirmed this in
`converted_data.pkl` (trial 0 and trial 1 of session 0 are bit-identical, both maxing at 119.667), and
`verification_full_out.txt` reports the input range as `[0.0, 119.7]` for all 41 sessions.

What the input actually encodes is **time within the trial**, not time elapsed from the start of the
experiment/session. Because the vector is identical across trials, it carries no information that
distinguishes one trial from another: a decoder cannot use it to locate a trial in the session, and the
reference's 0–1799.67 s within-session ramp is absent. The human reference computes
`t = (s + arange(trial_frames)) * BIN_FRAMES / FS`, i.e. an offset that accumulates across trials.

ii. **Code snippets**

```python
def session_to_blocks(neural_binned, motion_binned):
    ...
    for b in range(n_blocks):
        s = b * BLOCK_BINS
        e = (b + 1) * BLOCK_BINS
        neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
        motion_blocks.append(motion_binned[s:e].astype(np.float32))
        # NOTE: no `s +` offset — identical vector for every block
        time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

iii. **Justification**

The AI stated this substitution explicitly in CONVERSION_NOTES.md Step 5: *"Decoder input requested by
task is time elapsed from beginning of experiment; operationalize as time within session/block after
alignment."* README.md repeats it: *"`input`: elapsed time within each 2-minute block, shaped
`(1, n_timepoints)`."* No rationale is given for why within-block time is an acceptable operationalisation
of time-from-start-of-experiment, and CONVERSION_NOTES.md Step 7 records the resulting range
(*"time_elapsed_s range | [0.0, 119.7]"*) without remarking that it should have spanned the session.

---

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. **Decisions**

By construction, on the same index grid. `session_to_blocks` builds `neural_blocks`, `motion_blocks` and
`time_blocks` inside a single loop over the same `[s:e]` window, so element `k` of the time vector is the
same 333.33 ms bin as column `k` of the neural matrix. All three are `(·, 360)`. The time value is the
**left edge** of the bin (bin 0 → 0.0 s), the same convention as the reference. No interpolation,
resampling or shifting is applied to the input.

ii. **Code snippets**

```python
for b in range(n_blocks):
    s = b * BLOCK_BINS
    e = (b + 1) * BLOCK_BINS
    neural_blocks.append(neural_binned[:, s:e].astype(np.float32))
    motion_blocks.append(motion_binned[s:e].astype(np.float32))
    time_blocks.append((np.arange(BLOCK_BINS, dtype=np.float32) * (BIN_FRAMES / FS))[None, :])
```

```python
for nb, mb, tb in zip(item['neural_blocks'], item['motion_blocks'], item['time_blocks']):
    ...
    sess_neural.append(nb.astype(np.float32))
    sess_input.append(tb.astype(np.float32))
    sess_output.append(y[None, :].astype(np.int64))
```

iii. **Justification**

CONVERSION_NOTES.md Step 10 Check 3: *"Raw-vs-converted input sanity check: recomputed time vectors matched
converted inputs with `np.allclose()`."* Building all three streams in one loop from one pair of indices
is the structural guarantee that there is no relative offset.

---

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. **Decisions**

Solely from `move_deve/motion_energy_glob.npy` — the authors' pre-computed global motion-energy trace, one
scalar per video frame at 30 Hz. `move_deve/tstamps.npy` and `move_deve/interframe_int.npy` are loaded in
`load_session` and `tstamps` is even passed into `align_motion_to_frames`, but **neither is read anywhere
in the script**. (The reference does use `interframe_int.npy`, to locate dropped frames.) There is one
output, `output_names = ['motion_energy_bin']`, with `output_values = [['bin_0' … 'bin_4']]`.

ii. **Code snippets**

```python
motion = np.load(move / 'motion_energy_glob.npy', allow_pickle=True)
tstamps = np.load(move / 'tstamps.npy', allow_pickle=True)
interframe = np.load(move / 'interframe_int.npy', allow_pickle=True)
```

```python
def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]      # `tstamps` is never referenced
    return aligned
```

iii. **Justification**

CONVERSION_NOTES.md Step 5 mapping: *"`move_deve/motion_energy_glob.npy` aligned to imaging frames →
output[0] | Handle missing camera frames using `tstamps.npy` / `interframe_int.npy`; average in bins of 10
frames; discretize valid values into 5 equal-percentile bins."* Step 3 records the paper's definition of
the signal: *"Global motion energy from squared pixel-wise frame differences"* — i.e. the released file is
already the finished metric and needs no recomputation. Methods: *"We used the global movements of the
mouse as a proxy of its arousal state."*

---

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. **Decisions**

Three steps, in order:

1. **Length reconciliation** (see 4-d): the trace is copied into a NaN-filled array of imaging length,
   prefix-first; a short trace leaves trailing NaNs, a long trace is truncated.
2. **10-frame NaN-aware averaging** to 3 Hz, jointly with the neural data (`nanbin_mean_1d`).
3. **Discretisation** into 5 classes (see 4-c), with NaN bins marked `-1` and then label-filled by
   nearest-valid forward then backward propagation.

Notably, **no normalisation is applied**, despite the AI's own prompt asking for motion energy
*"normalized and discretized into five equal-percentile bins"*. Motion energy is in raw squared-pixel
units (the global quintile edges printed by the run are 7.2e5 / 8.2e5 / 1.0e6 / 1.6e6), and these raw
units differ in scale between mice and sessions. Nothing removes that scale difference before
thresholding.

ii. **Code snippets**

```python
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
motion_binned = nanbin_mean_1d(aligned_motion, BIN_FRAMES)
```

```python
y = np.digitize(mb, edges, right=False).astype(np.int64)
y[np.isnan(mb)] = -1
...
if not np.all(valid):
    yy = y.copy()
    last = None
    for i in range(len(yy)):
        if yy[i] >= 0:
            last = yy[i]
        elif last is not None:
            yy[i] = last
    nxt = None
    for i in range(len(yy)-1, -1, -1):
        if yy[i] >= 0:
            nxt = yy[i]
        elif nxt is not None:
            yy[i] = nxt
    yy[yy < 0] = 0
    y = yy
```

iii. **Justification**

CONVERSION_NOTES.md Step 5 Key Decision 3 covers the binning (paper's 10-frame denoising). Key Decision 4
covers missingness: *"**Align motion to imaging frame index and preserve missingness information**:
`motion_energy_glob.npy` can be shorter than imaging due to missing camera frames; missing values should
be inserted or interpolated according to `tstamps.npy` / `interframe_int.npy`, matching data README
guidance."* The nearest-neighbour label fill is not separately justified in the notes. No justification is
offered anywhere for omitting normalisation.

---

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. **Decisions**

**One global set of quintile edges, computed over the pooled binned motion energy of every session in the
run**, then applied to every trial of every session:

```
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8])
      = [7.212e5, 8.227e5, 1.017e6, 1.646e6]
```

`np.digitize(mb, edges, right=False)` maps each bin to a label in `{0,…,4}`. Because the edges are pooled,
the marginal distribution over the **whole dataset** is exactly 20 % per class
(`verification_full_out.txt`: `{bin_0 (0.200), bin_1 (0.200), bin_2 (0.200), bin_3 (0.200),
bin_4 (0.200)}`) — but the **per-session** distributions are severely skewed, because raw motion-energy
units differ by roughly an order of magnitude across mice. From the delivered pickle:

* session 0 (jm031 day 1): class fractions `[0.000, 0.793, 0.150, 0.042, 0.015]` — class 0 empty, 79 % in
  one class.
* session 40 (jm046 day 7): class fractions `[0.000, 0.000, 0.000, 0.434, 0.566]` — only 2 of 5 classes
  present.
* All 7 jm046 sessions have zero bins in classes 0 and 1; `verification_full_out.txt` shows the output
  range collapsing to `[3.0, 4.0]` for two sessions and `[2.0, 4.0]` for four more.

The human reference computes `np.percentile(me, [0,20,40,60,80,100])` **within each session**, so every
session is balanced 20/20/20/20/20 by construction. The AI's prompt said "normalized and discretized into
five equal-percentile bins" without the words "per session"; `/tests/instruction_reference.md` says
"selected per session".

A side effect: because the class label is largely determined by *which mouse/session* a trial came from,
part of the 0.4443 balanced accuracy reported in `train_decoder_full_out.txt` is attributable to the
decoder recognising session identity from the neural population rather than tracking within-session
movement.

ii. **Code snippets**

```python
all_motion_values = []
...
for mb in motion_blocks:
    all_motion_values.append(mb[np.isfinite(mb)])
...
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
print('motion bin edges:', edges)
```

```python
y = np.digitize(mb, edges, right=False).astype(np.int64)
```

```python
'output_names': ['motion_energy_bin'],
'output_values': [[f'bin_{i}' for i in range(5)]],
```

iii. **Justification**

CONVERSION_NOTES.md Step 5 Key Decision 5: *"**Discretize motion energy into 5 equal-percentile bins using
valid binned samples across the full converted dataset**: This follows the decoder task requirement while
preserving balanced class frequencies as much as possible."* Step 7 acknowledges the consequence without
treating it as a problem: *"Motion bins are globally balanced by construction; per-session fractions vary,
which is expected."* Step 10 Check 4 likewise: *"recomputed global motion-energy bin edges and per-trial
discretized labels matched converted outputs exactly"* — the check verifies self-consistency, not
appropriateness.

---

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. **Decisions**

The two streams are assumed to be frame-synchronous from the first frame (the camera is microscope-
triggered at 30 Hz). `align_motion_to_frames` writes the motion trace into a NaN array of imaging length
starting at index 0, and pads the tail with NaN when the motion trace is short. **The positions of the
dropped camera frames are ignored**, even though `interframe_int.npy` is loaded.

This is a real misalignment, not just a cosmetic one: every motion sample *after* a dropped frame is
shifted one frame earlier relative to the neural data, and the shift accumulates. I measured the deficits
across the raw data:

| session | missing frames | resulting end-of-session shift |
|---|---|---|
| jm031 2023-10-22 | 116 | 3.87 s |
| jm032 2023-10-22 | 148 | 4.93 s |
| jm031 2023-10-21 | 3 | 0.10 s |
| jm031 2023-10-20, jm032 2023-10-20, jm032 2023-10-21 | 2 each | 0.07 s |
| jm039 2024-05-04, jm040 2024-05-04, jm046 2024-09-09 | 1 each | 0.03 s |

So 7 of 41 sessions are misaligned by well under one 333 ms bin (harmless), but 2 sessions drift by ~12–15
bins by the end — the last blocks of those sessions have neural and behavioural data from different
moments. The reference instead inserts an interpolated value at each drop index
(`np.where(dt * 1000 > 0.04)`) and asserts exact length equality.

Also worth flagging: the metadata string shipped inside `converted_data.pkl` says *"Motion energy aligned
to imaging frames using tstamps"*, which the code does not do — `tstamps` is a dead parameter.

ii. **Code snippets**

```python
def align_motion_to_frames(motion: np.ndarray, tstamps: np.ndarray, nframes: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    aligned = np.full(nframes, np.nan, dtype=np.float32)
    # Data README states motion_energy_glob is framewise with occasional missing camera frames.
    # `tstamps.npy` stores timestamps (seconds), not frame indices, so alignment should preserve
    # sample order and pad/truncate to imaging length.
    m = min(nframes, motion.shape[0])
    aligned[:m] = motion[:m]
    return aligned
```

```python
'notes': 'Motion energy aligned to imaging frames using tstamps; missing camera frames remain NaN '
         'before binning. ...'
```

iii. **Justification**

The decision is the product of a bug fix. Trajectory step 64: *"`tstamps.npy` contains timestamps in
seconds, not frame indices. Our current `align_motion_to_frames` rounds these tiny values to integers,
collapsing nearly all motion samples onto index 0 or 1 … The correct alignment is much simpler here: video
is 30 Hz and triggered by the microscope, and `motion_energy_glob.npy` is already framewise with
occasional missing frames. Therefore we should align by sample order … and only use timestamps/interframe
intervals to detect missingness or for possible interpolation—not as direct integer frame indices."*
CONVERSION_NOTES.md Step 10 records the same: *"Resolution: align motion by sample order and pad/truncate
to imaging length, per `data/README.md`."* The AI thus identified the right tool (`interframe_int.npy`) and
the right principle (detect missingness, interpolate) but implemented neither — the data README it cites
says *"The indices of missing frames can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy'
and treated as missing values for motion energy or they can be interpolated over."*

---

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. **Decisions**

Four mechanisms, all narrow:

1. **Short motion trace** → NaN-padded at the tail, never at the true drop indices (4-d). Long motion
   trace → silently truncated.
2. **Partially-missing bins** → `nanbin_mean_1d` averages only the finite frames in a bin; a bin is NaN
   only if all 10 frames are missing.
3. **NaN labels** → set to `-1`, then filled by nearest valid previous label (forward pass), then nearest
   valid next label (backward pass), then any remainder set to `0`. So the last few bins of a short
   session get a *fabricated* label copied from the last real bin rather than being dropped.
4. **Degenerate trials/sessions** → trials with >20 % missing labels are skipped; sessions left with <2
   trials are skipped. Neither triggers in practice.

Session-length heterogeneity (20 min for jm031/jm032 vs 30 min for the rest, against methods.txt's flat
"20 minutes") is handled by simply taking `n_frames // BLOCK_FRAMES` per session, so no data is forced or
lost beyond the sub-block remainder (0 s for 20-min sessions, 0 s for 30-min sessions; both divide evenly
by 120 s).

What is **not** handled: the near-zero/negative baseline case in `compute_dff`, where `np.maximum(baseline,
1e-3)` produces |dF/F| up to 5.7 × 10⁵ for 35 neurons across 18 sessions (see 2-b). There is no assertion
on output length, no range check on the neural data, and no check that the number of NaNs matches the
number of drops reported by `interframe_int.npy`.

ii. **Code snippets**

```python
m = min(nframes, motion.shape[0])
aligned[:m] = motion[:m]
```

```python
valid = np.isfinite(x)
sums = np.where(valid, x, 0.0).sum(axis=1)
counts = valid.sum(axis=1)
out = np.full(x.shape[0], np.nan, dtype=np.float32)
nz = counts > 0
out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
```

```python
y[np.isnan(mb)] = -1
valid = y >= 0
if valid.sum() < max(10, int(0.8 * len(y))):
    continue
if not np.all(valid):
    ...  # forward-fill, then backward-fill, then default to 0
```

```python
baseline = np.maximum(baseline, 1e-3)   # unguarded: negative baselines blow up dff
```

iii. **Justification**

CONVERSION_NOTES.md Step 4 documents the session-length discrepancy and its resolution: *"Session duration
| 20 min | Mixed: 20 min for jm031/jm032 (36000 frames), 30 min for jm038/jm039/jm040/jm046 (54000 frames)
| 20 min stated in methods | Keep observed durations from data; document discrepancy as likely broader
dataset variation or text simplification."* Step 10 / Step 12 both record: *"Initial issue: motion
alignment incorrectly interpreted `tstamps.npy` as frame indices, dropping 9 sessions. Resolution: align
motion by sample order and pad/truncate to imaging length."* Trajectory step 64 adds the motivation for
the ≥2-trial and 80 %-valid filters: *"avoid dropping sessions unnecessarily by interpolating/filling
sparse missing bins."* No rationale is given for the `1e-3` baseline floor.

---

## 6-a. What are the most time-consuming steps of the code?

i. **Decisions**

`compute_dff` dominates. I profiled it on one 685-neuron × 54 000-frame session: **2.62 s in
`compute_dff`, 0.42 s loading the `.npy` files, 0.07 s binning** — so the baseline estimation is ~80 % of
per-session wall time. The cost is the per-neuron `np.convolve` of a 54 000-sample trace with a
1800-sample boxcar kernel (~10⁸ multiply-adds per neuron), executed 20 445 times over the dataset. Total
full-run time was **130.47 s** for 41 sessions (`conversion_full_out.txt`), comfortably under the 15-minute
budget, so the inefficiency never became a blocker.

Everything else is cheap: binning and blocking are reshape-means; discretisation is a single
`np.quantile` plus `np.digitize`; the pickle write of a 414 MB file is I/O-bound but one-off.

The script reports only total elapsed time — there is no per-step timing instrumentation, despite the
instructions asking to "print timing information to find bottlenecks".

ii. **Code snippets**

```python
t0 = time.time()
for idx, (subj, sessdir) in enumerate(selected):
    F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
    dff = compute_dff(F, Fneu, ops)          # ~80% of per-session time
    ...
print(f'n_sessions={len(data["neural"])} total_trials={sum(len(x) for x in data["neural"])} '
      f'elapsed={time.time()-t0:.2f}s')
```

```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)   # np.convolve, 1800-tap kernel
    base = np.percentile(smooth, prct)
    baseline[i] = base
```

iii. **Justification**

CONVERSION_NOTES.md Step 6: *"Code inefficiencies identified: Potential inefficiency: baseline computation
loops over neurons and uses a simplified approximation instead of Suite2p internals. Potential
inefficiency: all sessions are prepared in memory before global motion discretization."* Step 7 estimated
*"~2.1 s/session in sample run … ~1.5–3 min for all 41 sessions"*, which the actual 130 s run matched, so
the AI judged no optimisation necessary.

---

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. **Decisions**

Three:

1. **The per-neuron baseline loop in `compute_dff`** — the dominant cost. The whole loop reduces to two
   vectorised calls: `scipy.ndimage.uniform_filter1d(Fcorr, win, axis=1, mode='reflect')` followed by
   `np.percentile(smooth, prct, axis=1)`. Additionally, `np.convolve` with a 1800-tap boxcar is
   asymptotically the wrong algorithm — a cumulative-sum sliding mean is O(n) instead of O(n·win), which
   alone would cut the dominant cost by ~3 orders of magnitude.
2. **The two label-fill loops** (forward and backward nearest-valid propagation) iterate over 360 bins in
   pure Python per trial. Standard vectorisation: `np.maximum.accumulate` over an index array.
3. **The block loop in `session_to_blocks`** materialises 360-bin slices one at a time; a single
   `reshape(n_neurons, n_blocks, BLOCK_BINS)` would produce all blocks at once (though this loop is cheap
   and the list-of-arrays layout is what the target format wants).

The AI identified (1) in its notes but never acted on it; (2) and (3) are not mentioned.

ii. **Code snippets**

```python
for i in range(Fcorr.shape[0]):
    smooth = moving_average_reflect(Fcorr[i], win)
    base = np.percentile(smooth, prct)
    baseline[i] = base
```

```python
last = None
for i in range(len(yy)):
    if yy[i] >= 0:
        last = yy[i]
    elif last is not None:
        yy[i] = last
nxt = None
for i in range(len(yy)-1, -1, -1):
    if yy[i] >= 0:
        nxt = yy[i]
    elif nxt is not None:
        yy[i] = nxt
```

iii. **Justification**

CONVERSION_NOTES.md Step 6, "Code inefficiencies identified": *"Potential inefficiency: baseline
computation loops over neurons."* Under "Code speedups added" the AI lists only *"Used vectorized frame
binning for neural and motion arrays; Limited plotting to first two sessions in `--show-processing` mode;
Sample mode processes only first two sessions for rapid validation."* Step 7's conclusion that the full
run would take 1.5–3 minutes made further optimisation unnecessary in the AI's judgement.

---

## 6-c. What processing does the code repeat multiple times?

i. **Decisions**

Very little genuinely repeated work. The two items that qualify:

1. **Binned motion is retained twice.** Each block's motion vector is stored in `prepared[...]
   ['motion_blocks']` and simultaneously appended (its finite subset) to `all_motion_values`, which is
   then concatenated into one large array to compute the quantiles. The pooled array duplicates the
   entire binned behavioural dataset in memory.
2. **The whole dataset is preprocessed before anything is written**, because global quantile edges require
   the pooled distribution. That is not recomputation, but it does force a two-pass structure (a
   per-session discretisation, as the reference does, would allow a streaming single pass).

Not repeated: `.npy` files are read exactly once per session; binning is applied once per stream;
`np.quantile` is called once for the whole run.

ii. **Code snippets**

```python
for mb in motion_blocks:
    all_motion_values.append(mb[np.isfinite(mb)])
...
all_motion_values = np.concatenate([x for x in all_motion_values if x.size > 0])
edges = np.quantile(all_motion_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
```

```python
prepared.append({
    'subject': subj, 'session': sessdir.name,
    'neural_blocks': neural_blocks, 'motion_blocks': motion_blocks,
    'time_blocks': time_blocks, 'n_neurons': dff.shape[0],
})
```

iii. **Justification**

CONVERSION_NOTES.md Step 6 flags the memory consequence of the two-pass design: *"Potential inefficiency:
all sessions are prepared in memory before global motion discretization."* The AI accepted it because the
global-edges decision (Step 5 Key Decision 5) requires seeing all sessions before any label can be
assigned.

---

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **Decisions**

1. **The dominant computation is almost entirely wasted.** `moving_average_reflect` builds a full
   54 000-sample smoothed trace for each neuron, and then only a **single scalar** (the 8th percentile) is
   kept from it. The smoothed trace is discarded. ~80 % of the script's runtime produces 20 445 numbers.
2. **Three arrays are loaded and never used**: `stat.npy` (a large pickled object array of per-ROI
   statistics), `tstamps.npy` and `interframe_int.npy`. `tstamps` is additionally threaded through as a
   parameter to `align_motion_to_frames`, where the body ignores it. `stat.npy` alone accounts for a
   noticeable share of the 0.42 s/session load time.
3. **The NaN-aware binning machinery is largely idle**: `nanbin_mean_1d`'s partial-bin logic only matters
   for at most one bin per session (the 9 sessions with drops), yet it runs on every session.
4. **`--show-processing` plots** render a 4-panel plus 1-panel figure per session; these are diagnostics
   and are not consumed by any downstream step (correctly gated to `idx < 2`).
5. **`time_blocks` stores 545 copies of the same 360-element vector** (~780 KB) inside the pickle, since
   the per-trial time input is trial-invariant (see 3-b).

ii. **Code snippets**

```python
F, Fneu, ops, stat, motion, tstamps, interframe = load_session(sessdir)
#                ^^^^                ^^^^^^^  ^^^^^^^^^  never referenced again
dff = compute_dff(F, Fneu, ops)
aligned_motion = align_motion_to_frames(motion, tstamps, dff.shape[1])
```

```python
smooth = moving_average_reflect(Fcorr[i], win)   # 54,000 values computed
base = np.percentile(smooth, prct)               # ... 1 value kept
```

```python
stat = np.load(pl0 / 'stat.npy', allow_pickle=True)   # loaded, never used
```

iii. **Justification**

CONVERSION_NOTES.md does not discuss discarded work; the closest statement is Step 6's *"baseline
computation … uses a simplified approximation instead of Suite2p internals."* The extra loads appear to be
leftovers from the exploratory phase: `tstamps` and `interframe_int` were part of the original
timestamp-indexing alignment scheme that was removed in Step 10 (trajectory step 64), and `stat`/`ops`
were loaded during Step 2 exploration when the AI was still deciding whether `iscell`-based curation was
needed (Step 4: *"All observed `iscell[:,0] == 1`"*).
