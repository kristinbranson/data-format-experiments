# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API. It takes the publication freeze table shipped with the
reference code, `/app/code/code_zhang2025/data/bwm_release.csv` (one row per probe insertion,
with `eid`, `pid`, `lab`, `subject`, `date`, `session_number`, `probe_name`), groups it by `eid`,
and reconstructs each session's ALF directory path by hand
(`<lab>/Subjects/<subject>/<date>/<nnn>/`). Inside that directory it globs for the ALF objects it
needs and reads them directly with `pandas.read_parquet` / `np.load(mmap_mode='r')`:
`_ibl_trials.table.pqt`, `_ibl_wheel.position/timestamps.npy`,
`<side>Camera.ROIMotionEnergy.npy` + `_ibl_<side>Camera.times.npy`, and per probe
`spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`,
`electrodeSites.brainLocationIds_ccf_2017.npy`. Because the cache holds several ONE revision
directories per object (e.g. `#2023-04-20#`, `#2025-03-03#`), it re-implements ONE's
default-revision resolution in `preferred()`: prefer an explicitly revised file over an unrevised
one, and among revised files take the lexicographically largest (= newest dated) path.
Before any conversion it runs `release_preflight()`, which reads every one of the 699 probes'
`clusters.metrics.pqt` and asserts the release reproduces the published
459 sessions / 699 probes / 621,733 raw units / 75,708 `label>=1` units exactly. Only library
helpers are imported from the reference code (`brainbox.behavior.wheel.interpolate_position`,
`velocity_filtered`, `iblatlas.regions.BrainRegions`).

ii.
```python
FREEZE = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def preferred(paths) -> Path | None:
    """Choose the newest explicit revision, or the unrevised file if unique."""
    paths = list(paths)
    if not paths:
        return None
    return sorted(paths, key=lambda p: ("#" in str(p), str(p)))[-1]

def session_path(row: pd.Series) -> Path:
    return (DATA_ROOT / str(row.lab) / "Subjects" / str(row.subject)
            / str(row.date) / f"{int(row.session_number):03d}")
```

```python
freeze = pd.read_csv(FREEZE, index_col=0)
release_stats = release_preflight(freeze)
groups = list(freeze.groupby("eid", sort=False))
```

```python
expected = {"sessions": 459, "probes": 699, "raw_units": 621733, "good_units": 75708}
if observed != expected:
    raise AssertionError(f"Release preflight mismatch: {observed} != {expected}")
```

iii. From CONVERSION_NOTES Step 4/Step 5: the local cache contains 461 session trees and 701
probe trees, i.e. two probes/sessions that are *not* part of the published release. The AI
argues that the release CSV "is the authoritative freeze. It maps without missing probes and
exactly reproduces both published unit totals; exclude the two non-release trees." For revisions
it notes "Explicit revision directories … coexist with unrevised paths; conversion must select
the ONE-default/newest intended revision once per logical object rather than count duplicates",
and cites the data-architecture white paper's ALF invariants as justification for loading
logical ALF objects directly rather than through the API.

## 1-b. How are the data split into subjects?

i. The subject name is read straight from the `subject` column of the release freeze — no path
parsing, no API call. `subjects` is the list of unique subject names in first-appearance
(freeze-file) order, and `subject_idx` is the index of each converted session's subject into
that list. The full run produced 136 subjects for 444 sessions (139 release subjects minus three
whose only sessions were dropped for missing whisker motion energy).

ii.
```python
first = rows.iloc[0]
...
info = {"eid": eid, "subject": str(first.subject), "lab": str(first.lab), ...}
```

```python
subjects = list(dict.fromkeys(x["session_info"]["subject"] for x in converted))
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_lookup[x["session_info"]["subject"]] for x in converted], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping table: "Release subject strings → `subjects`,
`subject_idx`; stable first-appearance unique list and per-session integer index … 139 release
subjects exist; retained set may be smaller after required-stream exclusion." The Step 9
consistency table records 136 vs 139 and attributes the difference to "Three mice only in
excluded sessions."

## 1-c. How are the data split into sessions?

i. A session is the `eid` (experiment id). The release freeze is one row per *probe*, so the AI
groups by `eid` to get one group of 1–4 probe rows per session, and iterates over those groups.
Session identity/order therefore comes straight from the freeze file (`sort=False` preserves
publication order). Probes of the same session are merged into one population rather than
treated as separate sessions.

ii.
```python
groups = list(freeze.groupby("eid", sort=False))
...
for eid, rows in groups:
    result = process_session(str(eid), rows, brain_atlas, want_plot)
```

```python
for _, row in rows.sort_values("probe_name").iterrows():
    probe_names.append(str(row.probe_name))
    probe_path = path / "alf" / str(row.probe_name)
    info = load_probe_units(probe_path, brain_regions)
```

iii. Step 5 Key Decision 9: "Probe merging: Concatenate retained units from simultaneous probes
in deterministic `probe_name` order; do not treat probes as independent sessions, matching both
papers' session-level behavioral dependence rationale." This mirrors the reference
`merge_probes`/`prepare_data`, which also works at the level of `eid` rather than `pid`.

## 1-d. How are the data split into trials?

i. No splitting is needed: `_ibl_trials.table.pqt` has one row per trial. Trial windows are
constructed as `[stimOn_times - 0.5 s, stimOn_times + 1.5 s]` for every trial, and every stream
(spikes, wheel, camera) is sliced into those windows with `np.searchsorted`. Native trial order
and native trial indices are preserved (`original_trial_indices` is stored in metadata).

ii.
```python
def load_trials(path: Path) -> tuple[pd.DataFrame, Path]:
    trial_file = preferred(path.glob("alf/**/_ibl_trials.table.pqt"))
    if trial_file is None:
        raise FileNotFoundError(f"No trial table under {path}")
    return pd.read_parquet(trial_file), trial_file
```

```python
begins_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_START
ends_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_END
```

iii. Step 2 documents the 13 float64 columns of the trials parquet and that it is
`(n_trials, 13)`; Step 4 resolves to "Preserve original trial order so trial number/block
structure remains meaningful."

## 1-e. How are trials filtered based on quality controls?

i. Two stages applied jointly.
**(1) The reference code's trial mask**, re-implemented to reproduce
`load_trials_and_mask(one, eid, max_trial_len=10.0)` as it is called in
`prepare_data`: all of `stimOn_times, choice, feedback_times, probabilityLeft,
firstMovement_times, feedbackType` must be non-NaN; reaction time
`firstMovement_times - stimOn_times` must lie in [0.08, 2.00] s; trial duration
`feedback_times - goCue_times` must be ≤ 10 s; `choice != 0` (no-response trials dropped).
Unbiased 0.5-prior trials are *kept* (`exclude_unbiased=False`, the reference default).
**(2) Stream coverage**: both the wheel trace and the chosen camera motion-energy trace must
span the whole 2 s window, using the reference code's own tolerance — the first in-window
sample must be within one bin (20 ms) of the window start and the last within one bin of the
window end. A per-stage audit (`native → required_events → reaction_time → duration → choice →
motion_coverage → joint_coverage`) is recorded per session. Sessions left with < 2 valid trials
are dropped. Result: 188,925 trials of 296,090 native release trials over 444 sessions.

ii.
```python
def reference_trial_mask(trials):
    """Reproduce the supplied load_trials_and_mask call used by prepare_data."""
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    present = np.ones(len(trials), dtype=bool)
    for name in required:
        present &= trials[name].notna().to_numpy()
    rt = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    rt_ok = (rt >= 0.08) & (rt <= 2.00)
    duration = (trials["feedback_times"] - trials["goCue_times"]).to_numpy()
    duration_ok = duration <= 10.0
    choice_ok = trials["choice"].to_numpy() != 0
    mask = present & rt_ok & duration_ok & choice_ok
```

```python
def coverage_mask(stream, begins, ends):
    """Reference coverage checks used before interpolation."""
    ib = np.searchsorted(stream.times, begins, side="right")
    ie = np.searchsorted(stream.times, ends, side="left")
    ok = (ie - ib) >= 2
    ...
    ok &= np.abs(begins - first) <= BIN_SIZE
    ok &= np.abs(ends - last) <= BIN_SIZE
    return ok
```

```python
valid = base_mask & motion_ok & wheel_ok
trial_indices = np.flatnonzero(valid)
if len(trial_indices) < 2:
    print(f"SKIP {eid}: {len(trial_indices)} jointly valid trials", flush=True)
    return None
```

iii. Step 1/Step 3/Step 4: the mask is "identical event/RT/duration/no-choice mask" to
`load_trials_and_mask(min_rt=.08, max_rt=2, max_trial_len=10, exclude_nochoice=True)`, and the
0.08–2 s bounds are also the data paper's reaction-time truncation. The coverage rule is copied
from the reference `get_behavior_per_interval`, which skips an interval when
`abs(interval_beg - target_time[0]) > binsize` or `abs(interval_end - target_time[-1]) > binsize`.
The AI further notes that the reference `align_spike_behavior` "only preserves the final behavior
availability mask due to Python-list boolean semantics" and deliberately requires *both* the
wheel and the whisker stream to be valid, calling this "an intentional correction, not
replication of that evident bug."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The binned array itself comes from the two per-probe spike arrays `spikes.times.npy` and
`spikes.clusters.npy`. Three further per-probe files are used only for unit selection and
annotation: `clusters.metrics.pqt` (the `label` quality score and `cluster_id`),
`clusters.channels.npy` (each cluster's peak channel), and
`electrodeSites.brainLocationIds_ccf_2017.npy` (fallback
`channels.brainLocationIds_ccf_2017.npy`) for the Allen CCF id of that channel.

ii.
```python
spike_times_path = preferred(probe_path.glob("**/spikes.times.npy"))
spike_clusters_path = preferred(probe_path.glob("**/spikes.clusters.npy"))
```

```python
times = np.load(info["times_path"], mmap_mode="r")
clusters = np.load(info["clusters_path"], mmap_mode="r")
```

```python
metrics = pd.read_parquet(metrics_path)
good_rows = np.flatnonzero(metrics["label"].to_numpy() >= 1)
cluster_channels = np.load(channels_path, mmap_mode="r")
allen = brain_regions.id2acronym(np.asarray(atlas_ids[peak_channels], dtype=np.int64))
beryl = brain_regions.acronym2acronym(allen, mapping="Beryl").astype(str)
```

iii. Step 2: "Neural: per-probe NumPy `spikes.times`, `spikes.clusters` … cluster metrics
Parquet, channel/electrode atlas IDs … Brain location is stored as Allen CCF numeric IDs at
channel/electrode level; neuron acronyms/regions must be obtained by associating each cluster's
peak channel and mapping atlas ID to an Allen acronym, then Beryl as in the reference code."
This reproduces what `SpikeSortingLoader.merge_clusters` does inside the API.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 half-open 20 ms bins over `[stimOn-0.5, stimOn+1.5)`, per unit.
Only `label >= 1` units are counted. Counts are accumulated with a single `np.bincount` per trial
per probe over a flattened `unit * N_BINS + bin` index, built in `uint16` and cast to `float32`
for output. The units of the stored `neural` array are **raw spike counts per 20 ms bin**, not
firing rates — no division by the bin width, no smoothing, no z-scoring. Units of the 1–4
simultaneous probes of a session are concatenated into one population, in `probe_name` order,
with a running offset so unit numbering continues across probes.

ii.
```python
counts = np.zeros((len(trial_indices), n_units, N_BINS), dtype=np.uint16)
offset = 0
for info in unit_infos:
    n = len(info["cluster_ids"])
    bin_probe(info, begins, ends, counts[:, offset : offset + n, :])
    region_labels.extend(info["regions"].tolist())
    offset += n
```

```python
rel_t = np.asarray(times[lo:hi], dtype=np.float64)[keep] - beg
bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)
valid = (bins >= 0) & (bins < N_BINS)
flat = mapped[keep][valid].astype(np.int64) * N_BINS + bins[valid]
destination[trial] = np.bincount(flat, minlength=width).reshape(
    destination.shape[1], N_BINS)
```

```python
neural_trials.append(counts[i].astype(np.float32))
```

iii. Step 1: "Neural representation is raw spike counts, not firing rates and not fluorescence;
delta-F/F is inapplicable to this electrophysiology dataset" — the reference caching script
stores `binned_spikes` as raw counts (`get_sparse_from_binned_spikes`), and
`standardize_spike_data` is applied only inside the reference *decoder*, not in the cached data.
Step 5 mapping: "Raw counts, no smoothing or firing-rate conversion." Probe merging is justified
by Key Decision 9 (simultaneous probes share behaviour, so they are not independent sessions),
matching `merge_probes`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single quality rule: keep clusters whose `clusters.metrics.label >= 1`. The IBL label is
0, 1/3, 2/3 or 1 depending on how many of the amplitude / noise-cutoff / refractory-period
criteria a unit passes, so `>= 1` keeps only units passing all of them. The preflight verifies
this reproduces the data paper's 75,708 of 621,733 units exactly; 73,044 of these fall in the 444
retained sessions. **No anatomical filter is applied** — units whose Beryl acronym is `void`
(outside the brain), `root`, `x` or `y` are all retained, and appear in `brain_regions`
(the verification log reports 246 `void`, 10,053 `root`, 45 `x`, 6 `y` neurons). No
region-level inclusion rule (≥5 neurons/session, ≥2 sessions) is applied either.

ii.
```python
metrics = pd.read_parquet(metrics_path)
good_rows = np.flatnonzero(metrics["label"].to_numpy() >= 1)
cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)[good_rows]
```

```python
remap = np.full(max_id, -1, dtype=np.int32)
remap[good_ids] = np.arange(len(good_ids), dtype=np.int32)
...
keep = mapped >= 0
```

```python
if n_units == 0:
    print(f"SKIP {eid}: no label>=1 units", flush=True)
    return None
```

iii. Step 4 records this as a genuine cross-source conflict: the reference caching script calls
`load_spiking_data(qc=None)` and keeps *all* Kilosort clusters, storing `label>=1` only as
`good_clusters` metadata, while the data paper's analyses use well-isolated units. Resolution
(Step 5 Key Decision 2): "Use `clusters.metrics.label >= 1`. This follows the data-paper's
explicit neural curation, improves biological validity, and keeps the dense target
representation tractable. It intentionally differs from the method caching script's `qc=None`;
all-cluster counts remain checked as a release-integrity invariant." For the anatomy, Step 4
states: "Map each retained unit through its peak channel to Allen acronym then Beryl. Do not
apply the paper's region-level ≥5-neuron/≥2-session inferential filter because target format
needs neuron annotations rather than region-level hypothesis tests." Dropping `void`/`root`
units is never discussed anywhere in the notes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one synchronised session clock in seconds, so alignment is a
subtraction. For each retained trial the window `[stimOn_times - 0.5, stimOn_times + 1.5)` is
located in the (time-sorted) spike array with `np.searchsorted`, the window start is subtracted
from the spike times, and the result is floor-divided by the bin size. Bin 0 therefore starts at
exactly −0.5 s relative to `stimOn_times` and bin 49/50 straddles the event. Indices outside
[0, 99] are discarded rather than clipped.

ii.
```python
begins_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_START
ends_all   = trials["stimOn_times"].to_numpy(dtype=float) + OFF_END
```

```python
left = np.searchsorted(times, begins, side="left")
right = np.searchsorted(times, ends, side="left")
for trial, (beg, lo, hi) in enumerate(zip(begins, left, right)):
    ...
    rel_t = np.asarray(times[lo:hi], dtype=np.float64)[keep] - beg
    bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)
    valid = (bins >= 0) & (bins < N_BINS)
```

iii. Step 4: "Temporal alignment … Caching script: stimulus onset, −0.5/+1.5 s, 20 ms for all
behaviors … Streams share session-clock seconds and cover those windows for most curated
trials … Follow the requested common alignment and the executable reference caching code:
100 bins of 20 ms over [−0.5,+1.5) s." Step 10 check 5 reports that exhaustive edge-case checks
found "No off-by-one mismatch", and the `--show-processing` plots overlay the stimulus at t=0
on raw wheel, behaviour and population-rate traces.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms (`time_bin_size = 20.0` ms in metadata), 100 bins per trial, identical for every trial
and session; the window is −0.5 to +1.5 s about stimulus onset. Spikes are binned once, directly
from the raw spike times — there is no rebinning or resampling of an already-binned array. The
behavioural streams are *resampled* (linear interpolation) onto the same 100-point grid, at the
right edge of each spike bin.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))            # 100
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
```

```python
"time_bin_size": 20.0,
"time_bin_size_units": "ms",
"off_start": OFF_START,
"off_end": OFF_END,
```

iii. Step 3/Step 5 Key Decision 3: "Common window/bin: Use 20 ms and [−0.5,+1.5) s around
stimulus onset for every stream. This is required by the task's stimulus alignment/common time
dimension and matches the executable cache script/overview, despite target-specific windows in
the detailed method-paper analysis." The reference `0_data_caching.py` sets
`{'interval_len': 2, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`,
and the method paper says "2-s trials, each divided into 20-ms bins". The AI explicitly flags
that the detailed STAR Methods use 50 ms for static targets and other windows for dynamic ones,
and overrides that because the task mandates one alignment and one bin size.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines each trial's window. The input itself
is a fixed grid that does not depend on any raw variable beyond the window definition: it is the
**right edge** of each of the 100 spike-count bins, i.e. −0.48, −0.46, …, +1.48, +1.50 s. The
same 100-value vector is written for every trial of every session (`[-0.48, 1.50]` reported by
the verifier).

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
```

```python
inp = np.vstack(
    (TIME_GRID, np.full(N_BINS, block_number[i], dtype=np.float32))
).astype(np.float32, copy=False)
```

iii. Step 5 mapping row: "Reference right-edge bin grid → `input[0]`: Time since stimulus onset
`[-0.48, -0.46, ..., 1.50]` s, repeated identically for each trial. Continuous, time-varying.
Values are right edges of corresponding neural count bins, matching behavioral interpolation."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond defining the grid; it is a constant `np.arange`. The one real decision is the
choice of the bin *right edge* rather than the bin centre or left edge, taken so that the time
axis is exactly the abscissa at which the reference code samples its behavioural signals
(`x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)` in
`get_behavior_per_interval`). It is stored as `float32` and broadcast to every trial.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
```

```python
"time_coordinate": "right edge of each half-open neural spike-count bin, seconds from stimulus onset",
```

iii. Step 1: "Behavior interpolation points are `interval_start + binsize` through
`interval_end`; spike bins cover half-open intervals from the interval start. The one-bin
convention is therefore behavior at the right edge of each corresponding spike-count bin."
Step 5 Key Decision 4: "Continuous sampling: Use the reference code's right-edge interpolation
grid rather than bin centers or per-bin behavior averages, because this is the exact supplied
implementation."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Bin for bin, by construction. `TIME_GRID[k]` is the right edge of the same half-open 20 ms
bin `k` that column `k` of the neural matrix counts spikes in, and both are measured from the
same `stimOn_times`. The identical grid is also the abscissa used to resample wheel speed and
whisker motion energy, so all five rows of `input`/`output` and the neural columns share one
time axis. Column 24/25 is the last bin fully before the stimulus and column 25 is the first bin
containing it.

ii.
```python
begins = begins_all[trial_indices]         # stimOn - 0.5
ends   = ends_all[trial_indices]           # stimOn + 1.5
...
bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)     # neural: left-closed bins from `begins`
```

```python
steps = BIN_SIZE * np.arange(1, N_BINS + 1)
x = beg + steps                                        # behaviour + time input: right edges
```

iii. Step 5 planned sanity check "Shapes/ranges: exactly 100 bins … strict `-0.48` through
`+1.50` time samples"; Step 10 check 2 reports "the 100-point time grid … match inputs" by
`np.allclose` against an independent reconstruction, and check 5 reports no off-by-one errors.
The `--show-processing` panels plot the neural raster, the behaviour traces and the categorical
outputs on the same axis with the stimulus marked at 0.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table only. The trials table carries no block id, so a
block boundary is defined as any trial where `probabilityLeft` differs from the previous trial's
value (compared with `np.isclose`, with NaN treated as a boundary). The counter is computed over
the **complete native trial sequence** of the session, before any QC filtering, and only then
indexed by the retained trial indices.

ii.
```python
def block_trial_numbers(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based native trial number within each probability block."""
    out = np.zeros(len(probability_left), dtype=np.float32)
    count = 0
    for i in range(1, len(probability_left)):
        same = np.isfinite(probability_left[i]) and np.isfinite(probability_left[i - 1])
        same = same and np.isclose(probability_left[i], probability_left[i - 1])
        count = count + 1 if same else 0
        out[i] = count
    return out
```

```python
native_prob = trials["probabilityLeft"].to_numpy(dtype=float)
block_number = block_trial_numbers(native_prob)[trial_indices]
```

iii. Step 5 mapping: "Full raw `probabilityLeft` sequence → `input[1]`: Detect block transitions
before trial filtering; zero-based count since current block began; repeat the per-trial scalar
over 100 bins … excluded trials do not renumber later retained trials."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based running count within each block: the first trial of a block gets 0, the next 1,
and so on. Because it is computed on the native sequence, a trial that is later discarded by QC
still advances the counter, so the stored number is the animal's true position in the block. The
per-trial scalar is then broadcast across all 100 time bins as `float32` (a per-trial quantity
stored in time-varying form so all rows share one shape). The observed range is [0, 98],
matching the human reference exactly.

ii.
```python
count = count + 1 if same else 0
out[i] = count
```

```python
inp = np.vstack(
    (TIME_GRID, np.full(N_BINS, block_number[i], dtype=np.float32))
).astype(np.float32, copy=False)
```

```python
"trial_number_in_block_definition": (
    "zero-based index in native probabilityLeft block, computed before trial filtering"),
```

iii. Step 5 Key Decision 7: "Block counter: Zero-based trial index within each native probability
block, computed on all original trials before QC. This retains true experimental progression
even when invalid trials are removed." Key Decision 6 justifies the broadcasting: "Per-trial
values are repeated in time because NumPy cannot combine scalar and time-varying rows otherwise;
their semantics remain per-trial." Step 10 check 2 verified "three native probability-block
counters match inputs" by `np.allclose` against raw parquet.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of `_ibl_trials.table.pqt`, which is +1, −1 or 0. Trials with 0 are
already removed by the trial mask. The AI states the convention as **native −1 = left, +1 =
right** and maps `−1 → 0` ("left") and `+1 → 1` ("right"). Note that the IBL convention is the
opposite — in `/app/code/ibllib/brainbox/behavior/training.py` line 690 the comment reads
"choice == −1 means contrast on right hand side", `plot_all_peths.py` documents
"subject's choice (1=left, −1=right)", and checking a raw trials table directly (correct trials
only) gives `choice == +1` for every stimulus-on-the-left trial and `choice == −1` for every
stimulus-on-the-right trial. The resulting class fractions are `left 0.492 / right 0.508`,
the mirror image of the human reference's `0.5075 / 0.4925`.

ii.
```python
choice_native = trials["choice"].to_numpy(dtype=float)[trial_indices]
choice = np.where(choice_native == -1, 0, 1).astype(np.int64)
```

```python
"output_values": [
    ["left", "right"],
    ...
```

iii. Step 1: "Choice is native IBL `-1`/`+1` in the cache; the requested target conversion must
remap left/right explicitly." Step 5 mapping row: "`_ibl_trials.choice` → `output[0]`: Native
−1 (left) → 0; +1 (right) → 1 … No-go 0 trials excluded." Step 2 likewise reports "native choice
counts are left (`-1`) 146,529, right (`+1`) 149,824". The polarity is asserted rather than
derived — no check against `contrastLeft`/`contrastRight`/`feedbackType` appears anywhere in the
notes or the trajectory, and the Step 12 "choice debugging" only re-verified that the stored
values equal the mapped raw values, not that the mapping direction is right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nothing beyond the recoding above, plus broadcasting the per-trial scalar over the 100 time
bins and casting to `int64`. No-response trials never reach this point. The binary encoding is
validated before saving (`{0, 1}` only).

ii.
```python
out = np.vstack((
    np.full(N_BINS, choice[i], dtype=np.int64),
    np.full(N_BINS, prior[i], dtype=np.int64),
    wheel_labels[i],
    motion_labels[i],
))
```

```python
assert set(np.unique(np.concatenate([x[0] for x in data["output"][s]]))).issubset({0, 1})
```

iii. Step 5 Key Decision 6 (per-trial values repeated across bins) and Key Decision 10
("output arrays int64"). The task specification fixes left = 0, right = 1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes exactly the three values 0.2,
0.5 and 0.8 (the 0.5 block is the unbiased block at the start of a session, retained). It is
matched with `np.isclose` and recoded 0.2 → 0, 0.5 → 1, 0.8 → 2; any other value raises. The
converted distribution is 0.417 / 0.141 / 0.442, matching the human reference
(0.418 / 0.141 / 0.442).

ii.
```python
prior_native = native_prob[trial_indices]
prior = np.full(len(trial_indices), -1, dtype=np.int64)
for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(prior_native, value)] = label
if np.any(prior < 0):
    raise ValueError(f"Unexpected probabilityLeft in {eid}")
```

```python
"output_values": [..., ["0.2", "0.5", "0.8"], ...]
```

iii. Step 5 mapping row: "`_ibl_trials.probabilityLeft` → `output[1]`: 0.2 → 0; 0.5 → 1; 0.8 → 2;
repeat across 100 bins. This task requests actual block prior probability, not the method
paper's model-derived continuous subjective prior." The mapping is the one given in the
instructions; the note that unbiased 0.5 trials are kept follows `exclude_unbiased=False` in the
reference `load_trials_and_mask` call.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the three-way recoding, float tolerance matching, broadcasting over the 100 bins
and the `int64` cast. A structural assertion confirms only {0, 1, 2} appear.

ii.
```python
np.full(N_BINS, prior[i], dtype=np.int64),
```

```python
for d in (1, 2, 3):
    assert set(np.unique(np.concatenate([x[d] for x in data["output"][s]]))).issubset({0, 1, 2})
```

iii. As above — the AI explicitly rejects the method paper's Bayesian/"subjective" prior in
favour of the literal block probability, because the Decoder Task spec names the three block
values and their integer codes.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw rotary-encoder object: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.
Speed is the absolute value of the filtered wheel velocity derived from those two, computed with
the very functions `SessionLoader.load_wheel` calls internally, imported from the supplied
ibllib (`brainbox.behavior.wheel.interpolate_position` and `velocity_filtered`). The raw
timestamp/position vectors are also returned unchanged for the `--show-processing` plot.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
```

```python
raw_times = np.asarray(np.load(times_path, mmap_mode="r"), dtype=np.float64)
raw_pos = np.asarray(np.load(pos_path, mmap_mode="r"), dtype=np.float64)
if len(raw_times) != len(raw_pos) or len(raw_times) < 20:
    raise ValueError("Invalid wheel position/timestamp dimensions")
```

iii. Step 1 / Step 5: the reference `load_target_behavior(one, eid, 'wheel-speed')` does
`sess_loader.load_wheel()` and takes `np.abs(sess_loader.wheel['velocity'])`; the AI reproduces
exactly this from the raw ALF files because it is not going through `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps. (1) The unevenly sampled wheel position is interpolated onto a uniform 1000 Hz
grid with `interpolate_position(..., freq=1000)`. (2) Velocity is obtained by differentiating
that trace through a zero-phase order-8 Butterworth low pass with a 20 Hz corner
(`velocity_filtered(pos, fs=1000, corner_frequency=20, order=8)`) — these are exactly
`SessionLoader.load_wheel`'s defaults. (3) The absolute value gives speed in rad/s. (4) The trace
is linearly resampled onto the 100 right-edge bin times of each retained trial; the final grid
point coincides with the window end and lies beyond the last in-window sample, so it is linearly
*extrapolated* from the last two in-window samples, reproducing
`scipy.interpolate.interp1d(..., fill_value='extrapolate')` in the reference. Non-finite results
raise.

ii.
```python
pos_1khz, times_1khz = interpolate_position(raw_times, raw_pos, freq=1000)
velocity, _ = velocity_filtered(pos_1khz, fs=1000, corner_frequency=20, order=8)
return (BehaviorStream(times_1khz, np.abs(velocity), "wheel"), raw_times, raw_pos)
```

```python
def interpolate_trials(stream, begins, ends):
    """Fast equivalent of reference per-interval scipy interp1d + extrapolation."""
    steps = BIN_SIZE * np.arange(1, N_BINS + 1)
    ...
        x = beg + steps
        y = np.interp(x, t, v)
        # scipy.interpolate.interp1d(fill_value='extrapolate') in the reference
        # extrapolates the final right-edge sample from the last two in-window points.
        slope = (v[-1] - v[-2]) / (t[-1] - t[-2])
        y[-1] = v[-1] + (end - t[-1]) * slope
```

iii. Step 4: "Wheel preprocessing … SessionLoader resamples position at 1 kHz and computes 20 Hz,
order-8 Butterworth zero-phase velocity; speed is absolute velocity … Reproduce SessionLoader
velocity exactly and sample at the reference right-edge grid." Step 6 lists the vectorised
`np.interp` + explicit end extrapolation as a deliberate speed-up that preserves the reference
numerics. Step 10 check 2 verified the interpolated values against an independent
reimplementation with `np.allclose`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Within-session equal-frequency tertiles. All retained samples of the session
(`n_trials × 100` values) are pooled, the 1/3 and 2/3 quantiles are taken, and each sample is
assigned class 0/1/2 with `np.searchsorted(thresholds, value, side='right')`. If the two
quantiles are equal (a degenerate, heavily tied trace) a deterministic stable-rank fallback
assigns exactly one third of the ordered samples to each class so that all three classes exist.
The thresholds and which method was used are stored per session in `metadata.session_info`.
Global fractions come out at 0.333 / 0.333 / 0.333.

ii.
```python
def discretize_tertiles(values):
    flat = values.ravel()
    thresholds = np.quantile(flat, [1 / 3, 2 / 3])
    if thresholds[0] < thresholds[1]:
        labels = np.searchsorted(thresholds, values, side="right").astype(np.int64)
        return labels, thresholds.astype(float), "value_quantiles"
    # Deterministic equal-frequency fallback for a degenerate/tied signal.
    order = np.argsort(flat, kind="stable")
    ranked = np.empty(flat.size, dtype=np.int64)
    ranked[order] = np.minimum(2, (np.arange(flat.size) * 3) // flat.size)
    return ranked.reshape(values.shape), thresholds.astype(float), "stable_rank_quantiles"
```

```python
wheel_labels, wheel_q, wheel_method = discretize_tertiles(wheel_cont)
```

iii. Step 5 Key Decision 5: "Compute 1/3 and 2/3 quantiles separately within each retained
session over all valid sampled values, then `searchsorted(..., side='right')`. This yields
low/medium/high relative behavior states and mirrors reference per-session scaling. Thresholds
are metadata. If quantiles tie, use deterministic stable rank tertiles as an explicit fallback so
all requested classes exist." The mapping table adds that session-wise thresholds "avoid
between-rig scale confounds".

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at exactly the same 100 time points as `TIME_GRID`, i.e. the right edges of
the neural spike-count bins, measured from the same `stimOn_times`; wheel timestamps are already
on the session clock, so no further synchronisation is needed. A trial is only kept if the wheel
trace covers the window to within one bin at both edges, so no bin is filled by long-range
extrapolation.

ii.
```python
begins = begins_all[trial_indices]   # stimOn - 0.5
...
wheel_cont = interpolate_trials(wheel, begins, ends)
```

```python
steps = BIN_SIZE * np.arange(1, N_BINS + 1)
x = beg + steps
y = np.interp(x, t, v)
```

iii. Step 1's "one-bin convention is therefore behavior at the right edge of each corresponding
spike-count bin", and Key Decision 4. The `--show-processing` figure overlays the continuous
speed and its class labels on `TIME_GRID` with the stimulus at 0; Step 7 records that the plots
show no temporal shift, and Step 10 check 5 found no off-by-one error.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy.npy` together with `_ibl_<side>Camera.times.npy`, the per-frame
motion energy IBL computes over a square covering the whisker pad. The left camera is used when
available and the right is the fallback, matching the reference `load_target_behavior`. Which
side was used is stored per session (`motion_energy_camera`). Sessions with neither usable
stream are dropped (14 of 459 in the release).

ii.
```python
def load_motion_energy(path: Path) -> BehaviorStream | None:
    """Match SessionLoader correction and reference left-first/right-fallback rule."""
    for side in ("left", "right"):
        values_path = preferred(path.glob(f"alf/**/{side}Camera.ROIMotionEnergy.npy"))
        times_path = preferred(path.glob(f"alf/**/_ibl_{side}Camera.times.npy"))
        if values_path is None or times_path is None:
            continue
```

```python
motion = load_motion_energy(path)
if motion is None:
    print(f"SKIP {eid}: no valid left/right whisker motion-energy stream", flush=True)
    return None
```

iii. Step 1: `load_target_behavior` "whisker energy prefers left camera and falls back to right."
Step 4: "437 release sessions have left ME, 8 additional have only right, 14 have neither …
Missing ME cannot be reconstructed and its sessions are excluded."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, smoothing or normalisation. Two data-integrity
corrections are applied first, copied from `SessionLoader._check_video_timestamps`: if there are
more timestamps than frames the *leading* extra timestamps are dropped
(`times = times[-len(values):]`); if there are fewer timestamps than frames (or fewer than two
samples, or non-monotonic timestamps) the stream is rejected and the side falls through to the
other camera. The trace is then linearly resampled onto the same 100 right-edge bin times per
trial, with the same last-point extrapolation as the wheel.

ii.
```python
if len(times) < len(values) or len(values) < 2:
    continue
if len(times) > len(values):
    times = times[-len(values) :]
if not np.all(np.diff(times) > 0):
    continue
return BehaviorStream(times, values, side)
```

```python
motion_cont = interpolate_trials(motion, begins, ends)
```

iii. Step 4: "ME preprocessing … SessionLoader corrects excess leading timestamps, then code
linearly interpolates to behavior-bin right edges … Reproduce timestamp correction,
left-first/right-fallback selection, and linear interpolation; exclude uncovered
trials/sessions." Step 5 mapping row repeats: "if timestamps exceed frames drop leading
timestamps; linearly sample at right-edge grid".

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: the same `discretize_tertiles` function, applied to the pooled
`n_trials × 100` retained samples of that session, splitting at the session's own 1/3 and 2/3
quantiles with `side='right'`, with the same stable-rank fallback for tied quantiles. Thresholds
and method are recorded per session. Global fractions 0.333 / 0.333 / 0.333.

ii.
```python
motion_labels, motion_q, motion_method = discretize_tertiles(motion_cont)
```

```python
thresholds = {"wheel": wheel_q, "motion_energy": motion_q}
...
"whisker_motion_energy_tertile_thresholds": motion_q.tolist(),
"whisker_motion_energy_discretization": motion_method,
```

iii. Same as 7-c — Key Decision 5, plus the mapping-table note that per-session thresholds mirror
the reference decoder's per-session behavioural scaling. Motion energy in particular has an
arbitrary per-rig pixel scale, which the AI cites as the reason not to pool across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: camera frame times are on the session clock, the trace is
evaluated at `stimOn_times - 0.5 + 0.02·k` for k = 1…100, i.e. the right edges of the neural
bins, and a trial is only kept if the camera covers the window to within one bin at each end.

ii.
```python
motion_ok = coverage_mask(motion, begins_all, ends_all)
...
motion_cont = interpolate_trials(motion, begins, ends)
```

```python
ok &= np.abs(begins - first) <= BIN_SIZE
ok &= np.abs(ends - last) <= BIN_SIZE
```

iii. As for the wheel — Key Decision 4 and Step 1's right-edge convention; Step 10 check 2
independently reproduced 600 dynamic class labels from raw camera files with `np.allclose`, and
the processing plots show the ME trace, its class labels and the population rate on one axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/bad data is excluded rather than imputed, at the finest granularity that still makes
sense, and every exclusion is logged:
- **Missing trial events**: NaN in any of the six required trial columns drops the trial (the
  reference `nan_exclude` list). Implausible trials (RT outside 0.08–2 s, duration > 10 s,
  no-response) are dropped too.
- **Missing/short behavioural coverage**: a trial whose window is not spanned by the wheel *and*
  the camera to within one bin is dropped.
- **Camera timestamp/frame mismatch**: excess leading timestamps are trimmed
  (`times[-len(values):]`); if timestamps are *shorter* than frames, or the stream has < 2
  samples, or timestamps are non-monotonic, that camera is rejected and the other side tried.
- **Whole sessions**: a session with no usable camera, with no `label>=1` unit, or with < 2
  jointly valid trials is skipped with a printed `SKIP` line (15 sessions in the full run).
- **Non-release data on disk**: the two extra probe/session trees present in the cache but absent
  from the freeze are never touched.
- **Corrupt indices**: wheel position/timestamp length mismatch, and peak-channel indices outside
  the atlas array, raise explicitly rather than silently producing wrong regions; spikes whose
  cluster id is outside the remap table, or whose bin index falls outside [0, 99], are dropped.
- **Nothing is imputed**: interpolated values are checked finite; 16 trials (0.0085 %) whose
  window genuinely contains no spike from any good unit are *kept*, with the reasoning that
  dropping them would make trial selection depend on neural activity.
- Structural invariants are asserted for every trial of every session before pickling.

ii.
```python
if len(times) < len(values) or len(values) < 2:
    continue
if len(times) > len(values):
    times = times[-len(values) :]
if not np.all(np.diff(times) > 0):
    continue
```

```python
if len(raw_times) != len(raw_pos) or len(raw_times) < 20:
    raise ValueError("Invalid wheel position/timestamp dimensions")
if np.any(peak_channels < 0) or np.any(peak_channels >= len(atlas_ids)):
    raise ValueError(f"Peak-channel index outside atlas array in {probe_path}")
if not np.all(np.isfinite(out)):
    raise ValueError(f"Non-finite interpolated values in {stream.source}")
```

```python
if len(trial_indices) < 2:
    print(f"SKIP {eid}: {len(trial_indices)} jointly valid trials", flush=True)
    return None
...
if n_units == 0:
    print(f"SKIP {eid}: no label>=1 units", flush=True)
    return None
```

iii. Step 5 Key Decision 8: "Missing data: Exclude a trial unless both continuous streams cover
the complete 2-s window; exclude sessions with fewer than two jointly valid trials. Do not impute
outputs." Step 10 check 1 on the all-zero trials: "Direct raw-spike inspection confirmed these
windows truly contain no spikes from the session's good units. Dropping otherwise valid trials
merely to suppress this warning would introduce neural-activity-dependent trial selection, so
they are retained and documented." Step 4 justifies excluding the 14 ME-less sessions: "Missing
ME cannot be reconstructed and its sessions are excluded."

## 10-a. What are the most time-consuming steps of the code?

i. Reading the per-probe spike arrays. The AI profiled this: the first sequential full run
projected to ~19 min and profiling "showed that spike-array reads, rather than computation, were
the bottleneck", so it kept the arrays as memmaps, used `searchsorted` to touch only the 2 s
windows, and overlapped the cold-cache I/O with a 4-thread pool over sessions (the NumPy kernels
and file reads release the GIL). The optimised full run took 349 s, of which 16.7 s is pickling
the 12.6 GiB output. Per session, cost scales with trial count (0.30 s for a 147-trial session up
to 6.9 s for a 1,445-trial session). The secondary costs are `release_preflight` (699 parquet
reads before any conversion) and the whole-session 1 kHz wheel interpolation + order-8
Butterworth filter.

ii.
```python
def bin_probe(info, begins, ends, destination) -> None:
    """Bin one probe directly from sorted spike memmaps into a trial×unit×time view."""
    times = np.load(info["times_path"], mmap_mode="r")
    clusters = np.load(info["clusters_path"], mmap_mode="r")
    ...
    left = np.searchsorted(times, begins, side="left")
    right = np.searchsorted(times, ends, side="left")
```

```python
# Session processing is dominated by independent mounted-file reads plus
# NumPy/SciPy kernels that release the GIL. A small thread pool overlaps
# cold-cache latency without copying multi-megabyte session results through
# multiprocessing pipes. executor.map preserves publication-freeze order.
with ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run_group, groups):
```

iii. Step 6 "Code inefficiencies identified: Loading all 21.1 billion spikes or materializing
full-session spike tables would be wasteful"; Step 9: "The first sequential full run was stopped
after the observed cold-I/O rate projected to approximately 19 minutes (>1.5x the Step 7
estimate) … I added a four-worker `ThreadPoolExecutor` across sessions (with ordered result
collection and no change to numerical processing)."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops remain, all of which could in principle be vectorised, though none is
a real bottleneck:
- `bin_probe` loops over trials, doing one `np.bincount` per trial. It could be a single
  `bincount` over all trials by adding `trial * n_units * N_BINS` to the flat index (which is what
  the human reference notes as the vectorised form).
- `interpolate_trials` loops over trials calling `np.interp` per trial; a single `np.interp` over
  one concatenated query vector would do, since the stream is monotonic.
- `block_trial_numbers` is a scalar Python loop over every native trial of the session
  (~400–1,500 iterations), doing per-element `np.isfinite`/`np.isclose` calls. This is the one
  that is genuinely wasteful and trivially vectorisable — the human reference does it in two
  pandas calls, `(p != p.shift()).cumsum()` then `groupby(...).cumcount()`.
The final assembly loop over trials (building the `(2,100)` and `(4,100)` stacks) is inherent to
the required list-of-arrays output format.
The AI did **not** enumerate any of these as remaining vectorisation opportunities; its notes
describe the loops it kept as already-optimised ("vectorized NumPy interpolation", "vectorized
flat `bincount` per trial/probe").

ii.
```python
for trial, (beg, lo, hi) in enumerate(zip(begins, left, right)):
    ...
    destination[trial] = np.bincount(flat, minlength=width).reshape(...)
```

```python
for i, (beg, end, ib, ie) in enumerate(zip(begins, ends, ibs, ies)):
    t = stream.times[ib:ie]
    v = stream.values[ib:ie]
    y = np.interp(beg + steps, t, v)
```

```python
for i in range(1, len(probability_left)):
    same = np.isfinite(probability_left[i]) and np.isfinite(probability_left[i - 1])
    same = same and np.isclose(probability_left[i], probability_left[i - 1])
    count = count + 1 if same else 0
```

iii. Step 6: "Per-spike Python loops and one SciPy interpolator object per behavior/trial would
be slow" → "Map cluster IDs and use vectorized flat `bincount` per trial/probe"; "Use vectorized
NumPy interpolation with an explicit final-sample extrapolation matching reference `interp1d`
behavior." The AI's stated rationale for keeping the per-trial granularity is that each trial is
a different contiguous slice of the session, and that after the I/O fix the whole conversion runs
in 349 s, so there is nothing material left to gain.

## 10-c. What processing does the code repeat multiple times?

i. Two genuine repeats:
- **`clusters.metrics.pqt` is read twice for every one of the 699 probes**: once in
  `release_preflight` (which reads the `label` column of all 699 probes before any session is
  processed, purely to assert the 621,733/75,708 totals) and again in `load_probe_units` when the
  session is actually converted.
- **`preferred()` re-globs the session directory** once per ALF object (trials, wheel ×2,
  camera ×2 per side, and 5 globs per probe), each a fresh recursive `path.glob("alf/**/...")`
  over the same tree.
Smaller repeats: `np.searchsorted` over the behaviour timestamps is done twice per stream (once
in `coverage_mask`, again in `interpolate_trials`); `stimOn_times` is converted to a NumPy array
twice; `BrainRegions()` lookups are redone per probe.

ii.
```python
def release_preflight(freeze):
    for _, row in freeze.iterrows():
        metrics_path = preferred(probe.glob("**/clusters.metrics.pqt"))
        metrics = pd.read_parquet(metrics_path, columns=["label"])
```
```python
def load_probe_units(probe_path, brain_regions):
    metrics_path = preferred(probe_path.glob("**/clusters.metrics.pqt"))
    ...
    metrics = pd.read_parquet(metrics_path)
```

```python
ib = np.searchsorted(stream.times, begins, side="right")   # coverage_mask
ie = np.searchsorted(stream.times, ends, side="left")
...
ibs = np.searchsorted(stream.times, begins, side="right")  # interpolate_trials
ies = np.searchsorted(stream.times, ends, side="left")
```

iii. The AI does not flag any of this as redundant. The preflight repeat is deliberate and
justified in Step 5 Planned Sanity Checks ("Release membership/totals: 459 EIDs, 699 probes,
621,733 raw clusters, 75,708 `label>=1` units") and Step 10 check 4 — it is a release-integrity
assertion that must run against the raw files independently of the conversion path. It costs a
one-off metadata-only read of 699 small parquet files.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things, all small:
- **`release_preflight`'s raw-unit totals** (621,733) are computed on every run and used only for
  an assertion and a metadata string; the conversion itself never uses non-`label>=1` clusters.
- **Raw wheel timestamps and positions** are returned from `load_wheel_speed` and carried through
  `process_session` on every session, but are only consumed by the `--show-processing` plot, i.e.
  for at most 2 of 444 sessions.
- **Whole-session wheel processing**: `interpolate_position` builds a 1 kHz trace for the entire
  multi-hour recording and `velocity_filtered` filters all of it, although only the ~2 s windows
  around retained stimulus onsets are ever sampled.
- **The continuous traces** `wheel_cont` / `motion_cont` (`n_trials × 100` float32 each) are
  computed and then thrown away after discretisation; only the 0/1/2 labels are saved. This is
  unavoidable given the required categorical outputs, but it is an intermediate that never
  reaches the decoder.
- **Per-trial repetition of per-trial scalars**: `choice`, `prior` and `trial_number_in_block` are
  each stored 100 times per trial, inflating the pickle; the decoder only needs the scalar.
- **Provenance metadata** never read downstream: `original_trial_indices`, `unit_cluster_ids`,
  `unit_probe_names`, `trial_table` path, per-stage `trial_filter_counts`, tertile thresholds and
  `processing_seconds` for all 444 sessions.
- `max_id` in `bin_probe` allocates a remap table of `len(good_rows) + raw_units` entries, roughly
  twice the size actually needed.

ii.
```python
return (BehaviorStream(times_1khz, np.abs(velocity), "wheel"), raw_times, raw_pos)
```

```python
raw += len(metrics)
good += int((metrics["label"] >= 1).sum())
```

```python
"original_trial_indices": trial_indices.astype(np.int64),
"unit_cluster_ids": np.asarray(unit_cluster_ids, dtype=np.int64),
"trial_filter_counts": audit,
"processing_seconds": elapsed,
```

iii. The AI justifies the extra metadata in Step 5 ("Supports audit/reconstruction") and the
per-bin repetition in Key Decision 6 ("Per-trial values are repeated in time because NumPy cannot
combine scalar and time-varying rows otherwise; their semantics remain per-trial"). The preflight
is justified as a mandated sanity check (Step 5, Step 10 check 4). The AI does not itself
identify the whole-session wheel filtering or the unconditional raw-wheel return as waste; its
own Step 6 list of inefficiencies covers only spike loading, per-spike loops, float32
construction and reprocessing ME-less sessions.
