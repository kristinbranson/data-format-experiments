# Decisions

Documentation of the decisions made by the agentic AI system in `/app/convert_data.py`,
`/app/CONVERSION_NOTES.md` and `/logs/agent/trajectory.json`, for the conversion of the IBL
brain-wide map dataset into the decoder format.

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. <Decisions>

The AI deliberately did **not** use the ONE API. It treats the frozen release manifest
`/app/code/code_zhang2025/data/bwm_release.csv` (699 probe insertions / 459 sessions / 139
subjects) as the authoritative inventory, and reads every ALF file directly off the local ONE
cache at `/app/data/one_cache` using paths reconstructed from the CSV columns
(`lab/Subjects/<subject>/<date>/<session_number:03d>/alf/...`).

Per session it locates five file groups with a recursive glob helper `_find_file()` that
resolves ALF *revision* folders by an explicit hard-coded preference string, falling back to
the last lexically-sorted candidate:

- `_ibl_trials.table.pqt` (preferred revision `#2025-03-03#`)
- `_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy`
- `{left,right}Camera.ROIMotionEnergy.npy` + `_ibl_{left,right}Camera.times.npy`
  (preferred revisions `#2025-05-29#` / `#2025-05-31#`)
- per probe: `spikes.times.npy`, `spikes.clusters.npy`, `clusters.channels.npy`,
  `channels.brainLocationIds_ccf_2017.npy` under `alf/<probe>/pykilosort/`
  (preferred revision `#2024-05-06#`)

Spike arrays are opened with `mmap_mode='r'` and never fully materialised. The loader asserts
the reconstructed inventory reproduces the data paper's headline numbers exactly (699 probes,
459 eids, 139 subjects, 621,733 clusters summed from the `clusters.channels` headers) and the
conversion log confirms `good_clusters: 75708` as a cross-check.

Loading is a two-pass design: a sequential behaviour pass over all 459 sessions first (cheap,
needed to compute global discretization thresholds), then a `ThreadPoolExecutor` spike-binning
pass with up to 8 workers.

ii. <Code snippets>

```python
APP = Path("/app")
DATA_ROOT = APP / "data" / "one_cache"
RELEASE_CSV = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

EXPECTED_RELEASE = {
    "subjects": 139, "sessions": 459, "probes": 699,
    "clusters": 621_733, "good_clusters": 75_708,
}
```

```python
def build_release_specs() -> tuple[list[SessionSpec], dict[str, int]]:
    release = pd.read_csv(RELEASE_CSV, dtype={"date": str, "subject": str, "lab": str})
    assert release["pid"].nunique() == EXPECTED_RELEASE["probes"]
    assert release["eid"].nunique() == EXPECTED_RELEASE["sessions"]
    assert release["subject"].nunique() == EXPECTED_RELEASE["subjects"]
    ...
    for eid, rows in release.groupby("eid", sort=False):
        first = rows.iloc[0]
        session_dir = (DATA_ROOT / first["lab"] / "Subjects" / first["subject"] /
                       first["date"] / f"{number:03d}")
        alf = session_dir / "alf"
        for row in rows.itertuples(index=False):
            probe_root = alf / row.probe_name / "pykilosort"
            pf = ProbeFiles(
                name=row.probe_name,
                spikes_times=_find_file(probe_root, "spikes.times.npy", "#2024-05-06#"),
                spikes_clusters=_find_file(probe_root, "spikes.clusters.npy", "#2024-05-06#"),
                ...)
            cluster_inventory += int(np.load(pf.clusters_channels, mmap_mode="r").shape[0])
    assert cluster_inventory == EXPECTED_RELEASE["clusters"]
```

```python
def _find_file(root: Path, name: str, preferred_revision: str | None = None) -> Path:
    candidates = sorted(p for p in root.glob(f"**/{name}") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"Missing {name} beneath {root}")
    if preferred_revision:
        preferred = [p for p in candidates if preferred_revision in p.parts]
        if preferred:
            return preferred[-1]
    return candidates[-1]
```

iii. <Justification>

From CONVERSION_NOTES Step 4: the live ONE cache table indexes 461 sessions / 701 probes /
622,377 clusters, i.e. two *extra* sessions (KS074, KS075) beyond the data-paper freeze. The AI
therefore chose `bwm_release.csv` as "the authoritative freeze" because it "reproduces every
headline data-paper unit statistic and removes the two later cache additions". Direct file reads
are justified in the Step-10 reference-comparison table as "direct files avoid network/API
ambiguity", with the revision policy documented as "Resolve one dataset per attribute
deterministically, favoring the current revision, and never treat revisions as separate
sessions" (five sessions physically carry duplicate trial-table revisions).

---

## 1-b. How are the data split into subjects (mice)?

i. <Decisions>

Subject identity is taken verbatim from the `subject` column of `bwm_release.csv` and carried on
the per-session `SessionSpec`. At assembly time `subjects` is the sorted unique set of subjects
among **retained** sessions and `subject_idx` is the index of each session's subject into that
list. No path parsing or heuristic grouping. Result: 136 subjects over 444 retained sessions
(139 in the freeze; 3 subjects lost entirely with the 15 excluded sessions).

ii. <Code snippets>

```python
specs.append(SessionSpec(
    eid=str(eid), subject=str(first["subject"]), lab=str(first["lab"]), ...))
```

```python
subjects = sorted({s["spec"].subject for s in sessions})
subject_lookup = {s: i for i, s in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_lookup[s["spec"].subject] for s in sessions], dtype=np.int32),
```

iii. <Justification>

Notes Step 5 variable-mapping table: "Frozen CSV subject → `subjects`, `subject_idx`; Sorted
unique subject IDs and integer lookup; Only subjects represented by retained sessions remain."
The release manifest already carries a unique subject id, so nothing has to be derived.

---

## 1-c. How are the data split into sessions?

i. <Decisions>

A session is the `eid`. The release CSV is grouped by `eid` with `sort=False` (preserving release
order); each group becomes one `SessionSpec`, and the probes of that group become that session's
probe list, i.e. simultaneously-recorded probes are pooled into one session rather than treated
as separate recordings. 459 candidate sessions; 444 survive.

ii. <Code snippets>

```python
# groupby(sort=False) preserves the release order.
for eid, rows in release.groupby("eid", sort=False):
    first = rows.iloc[0]
    ...
    probes: list[ProbeFiles] = []
    for row in rows.itertuples(index=False):
        ...
        probes.append(pf)
```

iii. <Justification>

Notes Step 4: "Sessions analyzed — Begin from all 459 frozen sessions, then enforce
requested-stream validity"; and Step 5 decision 1 "Release freeze: use exactly the 699
insertions in `bwm_release.csv`". Pooling probes within a session follows the reference
`merge_probes`, whose docstring/paper rationale is that probes in one session are not
statistically independent.

---

## 1-d. How are the data split into trials?

i. <Decisions>

Trials are the rows of `_ibl_trials.table.pqt`; no splitting logic is invented. Row order is
preserved and the *original* row indices of retained trials are carried through the pipeline as
`source_trial_idx` (and written to metadata as `source_trial_indices`), so that trial-level
provenance and the block counter are never compressed by filtering.

ii. <Code snippets>

```python
trials = pd.read_parquet(spec.trial_table)
raw_n = len(trials)
ref_mask = reference_trial_mask(trials)
raw_idx = np.flatnonzero(ref_mask)
```

```python
"source_trial_idx": raw_idx.astype(np.int32),
"stim_times": trials["stimOn_times"].to_numpy(dtype=np.float64)[raw_idx],
```

iii. <Justification>

Notes Step 2: "Trial data: `alf/#revision#/_ibl_trials.table.pqt`, one row per trial." No
decision to make. The explicit retention of raw indices is justified in Step 5 decision 3:
"drop individual invalid trials while preserving their original indices/block counts".

---

## 1-e. How are trials filtered based on quality controls?

i. <Decisions>

Two stages, both applied per session.

**Stage 1 — reference trial mask** (`reference_trial_mask`), a direct reimplementation of
`load_trials_and_mask(one, eid, max_trial_len=10.0)` as actually invoked by the reference
`prepare_data`:
- all of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times,
  feedbackType` must be non-NaN;
- reaction time `firstMovement_times - stimOn_times` in `[0.08, 2.0]` s;
- trial duration `feedback_times - goCue_times` not `> 10 s` (written as `~(duration > 10)` so
  that NaN durations are *kept*, deliberately reproducing pandas `eval` NaN semantics);
- `choice != 0` (no-response trials dropped).
Trials in the initial unbiased (`probabilityLeft == 0.5`) block are **kept**.

**Stage 2 — stream coverage** (`coverage_mask`), a reimplementation of the reference
`get_behavior_per_interval` boundary checks: for both the wheel and the chosen camera the
trial window `[stimOn-0.5, stimOn+1.5]` must contain ≥2 samples, and the first/last in-window
sample must be within one 20 ms bin of the window edges. Both wheel and camera must pass.

A session is dropped if fewer than two trials survive either stage. Outcome: 296,090 raw →
195,781 reference-valid → 188,925 converted trials.

ii. <Code snippets>

```python
def reference_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    mask = trials[required].notna().all(axis=1).to_numpy().copy()
    rt = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    duration = (trials["feedback_times"] - trials["goCue_times"]).to_numpy()
    mask &= (rt >= 0.08) & (rt <= 2.0)
    mask &= ~(duration > 10.0)  # matches the reference query, including NaN semantics
    mask &= trials["choice"].to_numpy() != 0
    return mask
```

```python
def coverage_mask(times, begins, ends):
    """Implement get_behavior_per_interval boundary validity checks."""
    ib = np.searchsorted(times, begins, side="right")
    ie = np.searchsorted(times, ends, side="left")
    good = (ie - ib >= 2) & (ib < len(times)) & (ie > 1)
    idx = np.flatnonzero(good)
    good[idx] &= np.abs(begins[idx] - times[ib[idx]]) <= BIN_SIZE
    good[idx] &= np.abs(ends[idx] - times[ie[idx] - 1]) <= BIN_SIZE
    return good, ib, ie
```

```python
wheel_good, _, wheel_ie = coverage_mask(wheel_times, begins, ends)
motion_good, _, motion_ie = coverage_mask(motion_times, begins, ends)
stream_good = wheel_good & motion_good
raw_idx = raw_idx[stream_good]
if raw_idx.size < 2:
    return None, "fewer than two trials with complete wheel/whisker coverage"
```

iii. <Justification>

Notes Step 3/Step 4: the data paper states trials are excluded for missing
choice/probabilityLeft/feedbackType/feedback/stimOn/firstMovement events and for RT outside
0.08–2.00 s; "The supplied code additionally excludes no-choice and trial duration >10 s. It
does not exclude the initial 0.5-prior block." Resolution: "Apply the supplied code mask
exactly, then add required behavior-coverage validity." Step 10 check 5 independently
reproduced the raw reward fraction (81.944% vs the paper's 81.4 ± 0.4%), the mean block length
(51.96 vs 51) and the 90-trial initial unbiased run, and showed the added coverage filter shifts
the correct-trial fraction by only 0.031 pp — i.e. it does not bias the retained set.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. <Decisions>

The spike raster itself comes from exactly two arrays per probe: `spikes.times.npy` and
`spikes.clusters.npy`. Two further arrays are read only for anatomy/bookkeeping:
`clusters.channels.npy` (number of clusters, and each cluster's peak channel) and
`channels.brainLocationIds_ccf_2017.npy` (Allen CCF id per channel), which together give each
unit a region label. `clusters.metrics`/`label` is **not** read at conversion time — the
75,708 good-cluster figure is only verified as a release-level sanity check, not used to filter.

ii. <Code snippets>

```python
for probe, n_clusters in zip(spec.probes, n_per_probe):
    spike_times = np.load(probe.spikes_times, mmap_mode="r")
    spike_clusters = np.load(probe.spikes_clusters, mmap_mode="r")
    ...
    cluster_channels = np.load(probe.clusters_channels, mmap_mode="r").astype(np.int64, copy=False)
    channel_ids = np.load(probe.channel_ids, mmap_mode="r")
    native = br.id2acronym(channel_ids[cluster_channels])
    region_names.extend(br.acronym2acronym(native, mapping="Beryl").tolist())
```

iii. <Justification>

Notes Step 5 mapping table: "`spikes.times`, `spikes.clusters` across all frozen-release probes →
`neural`", and "cluster channel CCF ID → `brain_regions`, `brain_region_idx`: cluster → channel
→ Allen ID → native acronym → Beryl acronym". This is the same chain
`SpikeSortingLoader.merge_clusters` performs in the reference, done from the released files
directly.

---

## 2-b. How is the `neural` data processed?

i. <Decisions>

Spikes are counted into 100 half-open 20 ms bins spanning `[stimOn-0.5, stimOn+1.5)`. Per probe,
`np.searchsorted` finds the spike index range for each trial window, and a single `np.bincount`
over the flattened `cluster * 100 + bin` index fills the whole `(n_clusters, 100)` grid for that
trial. Probes of one session are pooled by writing each probe's block into a row offset of the
shared `(n_neurons, 100)` array, which is equivalent to the reference `merge_probes`
re-indexing without physically concatenating and re-sorting the session-wide spike arrays.

Values are stored as **raw spike counts** cast to `float32` (not converted to Hz), with no
smoothing, no normalisation and no baseline subtraction. Trials are stored as a list of
`(n_neurons, 100)` arrays.

ii. <Code snippets>

```python
neural = [np.zeros((n_neurons, N_BINS), dtype=np.float32) for _ in range(n_trials)]
...
left = np.searchsorted(spike_times, begins, side="left")
right = np.searchsorted(spike_times, ends, side="left")
for j, (i0, i1, begin) in enumerate(zip(left, right, begins)):
    if i1 <= i0:
        continue
    clusters = np.asarray(spike_clusters[i0:i1], dtype=np.int64)
    bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
    # Floating-point arithmetic at exact edges may yield N_BINS; the
    # half-open interval guarantees such points belong outside.
    keep = (bins >= 0) & (bins < N_BINS)
    flat = clusters[keep] * N_BINS + bins[keep]
    counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
    neural[j][offset:offset + n_clusters] = counts
offset += n_clusters
```

iii. <Justification>

Notes Step 6: "The reference `get_spike_data_per_interval` applies a full-session Boolean
comparison separately for every trial, which is prohibitively expensive for 21.15 billion source
spike events", so the AI replaced it with "Binary-search each trial boundary in sorted
memory-mapped spike times… count flattened cluster×bin indices with compiled `np.bincount`".
Step 10 reference-comparison: "Same 100 half-open 20-ms spike-count bins; `searchsorted` +
`bincount` is an exact vectorized equivalent." Counts (rather than rates) match the reference
`bin_spiking_data` output, and the reference decoder standardises spike data per bin at training
time (`standardize_spike_data`), which the AI noted is "training preprocessing, not part of
cached data".

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. <Decisions>

**No neuron-level quality filtering at all.** Every Kilosort 2.5 cluster of every insertion in
the frozen release is retained, in cluster-table order — including clusters with `label < 1`,
clusters with zero spikes in the session, and clusters whose Beryl label is `void`, `root`, `x`
or `y` (i.e. sites the histology places outside the brain / outside the summary atlas). The
verification log shows 12,827 `void` neurons, 344 `x` and 18 `y` among the 281 "brain regions".
Total retained: 599,865 session-neurons over 444 sessions (mean 1,351/session), versus the data
paper's 75,708 well-isolated units. The resulting pickle is 98.8 GiB.

No region-based selection is applied either; all Beryl labels are kept because the target format
requires a region for every neuron.

ii. <Code snippets>

```python
n_per_probe = [int(np.load(p.clusters_channels, mmap_mode="r").shape[0]) for p in spec.probes]
n_neurons = int(sum(n_per_probe))
neural = [np.zeros((n_neurons, N_BINS), dtype=np.float32) for _ in range(n_trials)]
```

```python
native = br.id2acronym(channel_ids[cluster_channels])
region_names.extend(br.acronym2acronym(native, mapping="Beryl").tolist())
```

(There is no `label >= 1` test and no `!= 'void'` test anywhere in `convert_data.py`; the only
place `good_clusters` appears is the release-level assertion dictionary `EXPECTED_RELEASE`.)

iii. <Justification>

Notes Step 3 ("Neuron curation rules") and Step 5 decision 4: "Retain every Kilosort cluster
because this exactly matches the supplied methods-paper cache (`qc=None`) and the method paper's
'all neurons' decoder. Quality labels remain described in metadata." Step 4 discrepancy table:
"Caching calls `load_spiking_data` with `qc=None`… For this decoder conversion, retain all
621,733 units as the supplied decoder code does. Do not silently apply the data-paper
analysis-only good-unit filter." On regions: "Preserve every unit's Beryl label because target
format requires a region per retained neuron and the cache's all-region decoder does not apply
reportability filters. The 270 figure is an analyzed-region count, not a unit-loading rule."
In Step 12 the AI explicitly refused to add a QC filter to raise the choice score: "Applying a
good-unit or reportable-region filter solely to raise the score would contradict the supplied
caching code."

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. <Decisions>

Alignment is to `trials.stimOn_times`. Because spikes, trial events, wheel timestamps and camera
frame times are all already expressed in seconds on the same synchronised session clock, aligning
is a pure subtraction: the window is `begins = stimOn - 0.5`, `ends = stimOn + 1.5`, and each
spike's bin is `floor((t - begins)/0.02)`. The window is half-open `[begins, ends)` on both the
`searchsorted` slice (`side="left"` at both ends) and the bin index, with an explicit
`0 <= bin < 100` guard for floating-point edge cases. Metadata records
`temporal_alignment_event = "visual stimulus onset (trials.stimOn_times)"`, `off_start = -0.5`,
`off_end = 1.5`.

ii. <Code snippets>

```python
stim = behavior["stim_times"]
begins, ends = stim + OFF_START, stim + OFF_END
...
left = np.searchsorted(spike_times, begins, side="left")
right = np.searchsorted(spike_times, ends, side="left")
...
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
keep = (bins >= 0) & (bins < N_BINS)
```

```python
"temporal_alignment_event": "visual stimulus onset (trials.stimOn_times)",
"off_start": OFF_START,
"off_end": OFF_END,
```

iii. <Justification>

Notes Step 3/4: "The task here mandates stimulus-onset alignment for all variables. The
compatible common window in the supplied caching code is `[-0.5, +1.5)` s with 20 ms
spike-count bins (100 time bins)." The reference `0_data_caching.py` uses
`align_time='stimOn_times'`, `time_window=(-.5, 1.5)`. Step 10 check 4 states "The half-open
`[start,end)` convention was specifically checked at bin 0 and bin 99", and an in-script spot
check recomputes `neural[0][0,0]` directly from the raw arrays on every session.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. <Decisions>

20 ms bins, 100 bins per trial, over a 2 s window; identical for every trial and session.
`time_bin_size` is recorded as `20.0` ms in metadata. No rebinning, resampling or smoothing is
applied to the neural data — spikes are counted once, directly at 20 ms resolution. The
behavioural streams *are* resampled onto this same 100-point grid (see 7-b/8-b), which is the
only interpolation in the pipeline.

ii. <Code snippets>

```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
# The reference behavior code labels a count bin by its right edge.
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
```

```python
"time_bin_size": 20.0,
"time_bin_size_units": "ms",
"n_time_bins": N_BINS,
```

iii. <Justification>

Notes Step 3: "Requested/common cache: 20 ms over a 2 s window (100 bins)", matching
`params = {'interval_len': 2, 'binsize': 0.02, ..., 'time_window': (-.5, 1.5)}` in
`0_data_caching.py` and the method paper's 2-s / 100-bin tensor. Step 4 notes the paper's
target-specific alternatives (50 ms for prior, first-movement alignment for wheel/whisker)
"cannot coexist in the requested one-tensor, stimulus-aligned target and are superseded by the
explicit Decoder Task".

---

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. <Decisions>

It is not derived from a raw variable at all — it is the bin grid itself, defined by the
constants `OFF_START`, `OFF_END`, `BIN_SIZE`, anchored on `trials.stimOn_times`. The AI chose to
label each bin by its **right edge**, giving the fixed vector `[-0.48, -0.46, …, 1.50]` shared by
every trial of every session (the verifier rounds the displayed range to `[-0.5, 1.5]`).

ii. <Code snippets>

```python
# The reference behavior code labels a count bin by its right edge.
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
```

```python
inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                         np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

iii. <Justification>

Notes Step 5 decision 6: "Bin-end coordinates: Use −0.48…+1.50 s because the reference behavior
interpolator labels each count bin by its right edge. Neural counts remain half-open, avoiding
double counting on edges." This mirrors the reference
`x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)` in
`get_behavior_per_interval`. Metadata records `"time_coordinate": "right edge of each half-open
spike-count bin, seconds from stimulus onset"`.

---

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. <Decisions>

None. A single 100-element `float64` vector is built once at import time and broadcast as row 0
of every trial's `(2, 100)` input array, cast to `float32`. The instruction's guidance to
"represent a time as a binary time series" was not applied — the AI encoded it as a continuous
ramp, which is what the Decoder Task line ("Time since stimulus onset, continuous,
time-varying") specifies.

ii. <Code snippets>

```python
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
```

```python
for j in range(n_trials):
    inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                             np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

iii. <Justification>

Notes Step 5 mapping table: "Bin ending times relative to `stimOn_times` → `input[0, :]`:
`[-0.48, -0.46, ..., 1.50]` seconds, float32 … Continuous time-varying input. These are
bin-ending labels for neural intervals `[-0.50,-0.48),...`." No raw data is involved, so no
processing is required.

---

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. <Decisions>

By construction: it *is* the neural binning grid. Element `k` of the time input is the right edge
of the same bin `k` whose spikes were counted in `neural[:, k]`, with both measured from the same
`stimOn_times`. So the correspondence is exact bin-for-bin; the only nuance is that the label is
the bin's closing edge rather than its centre (a 10 ms offset relative to a centre convention),
and the behavioural outputs are sampled at exactly those same instants, so inputs, outputs and
neural bins are mutually consistent.

ii. <Code snippets>

```python
begins, ends = stim + OFF_START, stim + OFF_END       # neural window
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
```

```python
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]   # behaviour sampled on the same grid
```

iii. <Justification>

Notes Step 10 reference comparison: "Same `stimOn_times`, -0.5 to +1.5 s window and right-edge
behavior grid." The `--show-processing` plots (`processing_<eid>.png`, panel 8) overlay the time
input, the block input and the discretized outputs on the same axis as the spike raster to make
the shared grid visually checkable; Step 7 records "No temporal offset… was found".

---

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. <Decisions>

From `trials.probabilityLeft` on the **unfiltered** trials table. The trials table carries no
block id, so blocks are recovered as maximal runs of constant `probabilityLeft`; a change of
value starts a new block. The initial 90-trial unbiased (0.5) run is treated as a block like any
other, which is why the observed input range is `[0, 89]` for many sessions and up to `[0, 98]`
overall.

ii. <Code snippets>

```python
probs_all = trials["probabilityLeft"].to_numpy(dtype=np.float64)
block_no_all = trial_number_in_block(probs_all)
...
"block_trial": block_no_all[raw_idx].astype(np.float32),
```

iii. <Justification>

Notes Step 5 mapping table: "Raw trial index and `probabilityLeft` run boundaries →
`input[1, :]`: Zero-based count within the original experimental block, reset when
`probabilityLeft` changes… Filtering does not compress block position." Step 10 check 5
validates the recovered blocks against the paper: "Raw probability-left run length is mean 51.964
(paper: 51) and median 47; median initial unbiased run is exactly 90 trials (paper: first 90)."

---

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. <Decisions>

Run-boundary detection followed by a zero-based counter within each run, computed on the full raw
trial sequence *before* any filtering, then indexed by the surviving raw trial indices and
broadcast as a constant across all 100 bins of that trial. So a trial that is later dropped still
advances the counter, and the value is the animal's true position in the block.

ii. <Code snippets>

```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based block position on the unfiltered source sequence."""
    n = probability_left.size
    starts = np.r_[0, np.flatnonzero(probability_left[1:] != probability_left[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    out = np.empty(n, dtype=np.int32)
    for start, end in zip(starts, ends):
        out[start:end] = np.arange(end - start, dtype=np.int32)
    return out
```

```python
inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                         np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

iii. <Justification>

Notes Step 5 decision 3: "drop individual invalid trials while preserving their original
indices/block counts", and Step 7: "The first plotted retained trial correctly has block position
9 because excluded raw trials are not allowed to compress the experimental block counter."
Metadata records `"trial_number_in_block": "zero-based position in the original unfiltered
probabilityLeft run"`. Step 12 additionally argues the input does not leak the block side: "The
trial-number input does not encode block side; it is the explicitly required zero-based
within-block position."

---

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. <Decisions>

From the single column `trials.choice`, which takes values `+1`, `-1` and `0`. Zero (no-response)
trials are already removed by the reference trial mask. The AI maps `choice == +1 → 1` and
`choice == -1 → 0`, and declares `output_values[0] = ["left", "right"]`, i.e. it asserts that
**`-1` is a leftward choice and `+1` is a rightward choice**. The resulting class fractions are
[0.4921, 0.5079].

ii. <Code snippets>

```python
choices = trials["choice"].to_numpy(dtype=np.float64)[raw_idx]
...
if not np.all(np.isin(choices, (-1.0, 1.0))):
    raise ValueError(f"{spec.eid}: unexpected retained choice values")
...
"choice": (choices == 1).astype(np.int8),  # -1 left -> 0; +1 right -> 1
```

```python
"output_values": [
    ["left", "right"],
    ...
],
"choice_mapping": "source -1 (left) -> 0; source +1 (right) -> 1",
```

iii. <Justification>

Notes Step 5 mapping table: "`trials.choice` → `output[0, :]`: source −1 (left) → 0; source +1
(right) → 1; repeat across time… No-choice 0 source trials are already excluded." The same
assignment is stated in Step 2 ("choice counts are left (`-1`) 146,529 … right (`+1`) 149,824")
and Step 12 ("Choice is 49.2116% left / 50.7884% right"). No justification is given for the
direction of the mapping beyond assertion — searching the trajectory shows the AI never checked
`choice` against `contrastLeft`/`contrastRight`/`feedbackType`, nor against the IBL convention
documented in the bundled `ibllib` (`brainbox/behavior/training.py:590` `rightward =
trials.choice == -1`; line 690 `# choice == -1 means contrast on right hand side`).

---

## 5-b. What processing is involved in computing `output` *Choice*?

i. <Decisions>

Only the binary recoding described in 5-a, done with a single vectorised comparison
`(choices == 1).astype(np.int8)`, plus a defensive assertion that no retained choice value is
outside `{-1, +1}`. The per-trial scalar is then replicated across all 100 time bins so that the
static variable can share the `(4, 100)` output array with the two time-varying variables.

ii. <Code snippets>

```python
"choice": (choices == 1).astype(np.int8),  # -1 left -> 0; +1 right -> 1
```

```python
outputs.append(np.vstack((
    np.full(N_BINS, s["choice"][j], dtype=np.int8),
    np.full(N_BINS, s["prior"][j], dtype=np.int8),
    ...
)))
```

iii. <Justification>

Notes Step 5 decision 7: "Mixed static/dynamic fields: A single trial array cannot mix 1D and 2D
rows. Therefore input is 2×100 and output is 4×100; per-trial variables are constant repeats,
preserving their per-trial semantics while satisfying shape consistency." Step 12 adds that this
repetition is the reason the choice score is diluted: "its score includes 0.5 s of pre-stimulus
activity where choice information is weak… The target format cannot represent choice as a
separate 1D row alongside two time-varying rows, making repetition the only shape-consistent
faithful encoding."

---

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. <Decisions>

From `trials.probabilityLeft`, which takes exactly the three values 0.2, 0.5, 0.8. The mapping
0.2→0, 0.5→1, 0.8→2 is implemented with `np.searchsorted` over the sorted value list, guarded by
an assertion that no other value survives. Observed class fractions: [0.4174, 0.1406, 0.4420] —
consistent with 0.5 occurring only in the ~90-trial unbiased run at the start of each session.

ii. <Code snippets>

```python
priors = probs_all[raw_idx]
if not np.all(np.isin(priors, (0.2, 0.5, 0.8))):
    raise ValueError(f"{spec.eid}: unexpected retained prior values")
...
"prior": np.searchsorted(np.array([0.2, 0.5, 0.8]), priors).astype(np.int8),
```

```python
"prior_mapping": {"0.2": 0, "0.5": 1, "0.8": 2},
"output_values": [..., ["0.2", "0.5", "0.8"], ...],
```

iii. <Justification>

Notes Step 5 mapping table: "`trials.probabilityLeft` → `output[1, :]`: 0.2 → 0, 0.5 → 1, 0.8 → 2;
repeat across time… Exact mapping required by Decoder Task." The same column is what the
reference `bin_behaviors` exposes as `block`.

---

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. <Decisions>

None beyond the three-way recoding and the broadcast over 100 bins (same mechanism as choice).
Unbiased 0.5 trials are retained rather than dropped, matching the reference
`load_trials_and_mask(..., exclude_unbiased=False)` default.

ii. <Code snippets>

```python
"prior": np.searchsorted(np.array([0.2, 0.5, 0.8]), priors).astype(np.int8),
```

```python
np.full(N_BINS, s["prior"][j], dtype=np.int8),
```

iii. <Justification>

Notes Step 1 (`load_trials_and_mask` row): "Keeps 0.5-prior trials"; Step 4: the reference code
"does not exclude the initial 0.5-prior block". Step 5 decision 7 covers the constant-across-time
representation.

---

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. <Decisions>

From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy` — the raw, irregularly-sampled
wheel encoder stream. Speed is defined as the absolute value of the filtered velocity derived
from those two arrays; no pre-computed velocity or speed dataset is used.

ii. <Code snippets>

```python
wheel_raw_times = np.asarray(np.load(spec.wheel_times, mmap_mode="r"), dtype=np.float64)
wheel_raw_position = np.asarray(np.load(spec.wheel_position, mmap_mode="r"), dtype=np.float64)
if wheel_raw_times.shape != wheel_raw_position.shape or wheel_raw_times.size < 2:
    return None, "invalid wheel stream shape"
```

```python
wheel_speed = np.abs(wheel_velocity)
```

iii. <Justification>

Notes Step 2: "Wheel: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`… Speed is not
stored directly and must be computed from position/time as in the reference loader." This matches
the reference `load_target_behavior(..., 'wheel-speed')`, which returns
`np.abs(sess_loader.wheel['velocity'])`.

---

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. <Decisions>

Four steps, the first two using the **bundled reference implementation** rather than a
reimplementation (the script inserts `/app/code/ibllib` on `sys.path` and imports the functions
`SessionLoader.load_wheel` itself calls):

1. `interpolate_position(times, position, freq=1000)` — linear interpolation of position onto a
   uniform 1 kHz grid;
2. `velocity_filtered(position, fs=1000, corner_frequency=20, order=8)` — 8th-order Butterworth
   zero-phase low-pass at 20 Hz, then differentiation; the returned acceleration is discarded;
3. `np.abs(velocity)` → speed in rad/s;
4. linear resampling onto the 100 bin-end times of each trial, with the last bin obtained by
   linear *extrapolation* from the last two in-window samples, deliberately reproducing the
   reference's `searchsorted(..., side='left')` exclusion of the interval end combined with
   `fill_value='extrapolate'`.

Finally the trace is discretized (see 7-c). Non-finite aligned values cause the session to be
dropped.

ii. <Code snippets>

```python
# Use the implementation bundled with the reference code, not a reimplementation.
import sys
sys.path.insert(0, "/app/code/ibllib")
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
```

```python
wheel_position, wheel_times = interpolate_position(wheel_raw_times, wheel_raw_position, freq=1000)
wheel_velocity, _ = velocity_filtered(wheel_position, fs=1000, corner_frequency=20, order=8)
wheel_speed = np.abs(wheel_velocity)
...
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
```

```python
def interval_interpolate(times, values, targets, interval_ends, ie):
    """Reference-equivalent linear interpolation, including last-bin extrapolation."""
    result = np.interp(targets.ravel(), times, values).reshape(targets.shape)
    # get_behavior_per_interval excludes samples at/after interval end and uses
    # fill_value='extrapolate'. Reproduce that subtle behavior for the final bin.
    i1 = ie - 1
    i0 = ie - 2
    x0, x1 = times[i0], times[i1]
    y0, y1 = values[i0], values[i1]
    result[:, -1] = y1 + (interval_ends - x1) * (y1 - y0) / (x1 - x0)
    return result
```

iii. <Justification>

Notes Step 5 mapping table: "Uniform 1 kHz position interpolation; order-8, 20 Hz Butterworth
zero-phase filter; absolute differentiated velocity; linearly sample at bin ends… Reference Code
Function(s): `SessionLoader.load_wheel`, `interpolate_position`, `velocity_filtered`,
`load_target_behavior`." Step 10: "exact reference wheel functions… Final-point extrapolation
reproduces reference exclusion of the interval end." The parameters `fs=1000`,
`corner_frequency=20`, `order=8` are exactly `SessionLoader.load_wheel`'s defaults.

---

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. <Decisions>

Three classes split at **global** tertiles: the 1/3 and 2/3 quantiles are computed once over the
concatenation of every aligned wheel sample of every retained session (444 × n_trials × 100
values), then applied uniformly with `np.searchsorted(..., side="right")`. The thresholds
(0.01512 and 0.40508 rad/s) are stored in metadata; the code asserts they are distinct and that
all three classes are non-empty. Globally the classes are exactly balanced (1/3, 1/3, 1/3); the
per-session class-0 fraction in the verification log ranges roughly 0.16–0.44, so sessions remain
reasonably balanced.

ii. <Code snippets>

```python
wheel_values = np.concatenate([s["wheel"].ravel() for s in behavior_sessions])
wheel_q = np.quantile(wheel_values, [1 / 3, 2 / 3])
...
if not (wheel_q[0] < wheel_q[1] and whisker_q[0] < whisker_q[1]):
    raise ValueError(f"Non-distinct discretization thresholds: wheel={wheel_q}, whisker={whisker_q}")
thresholds = {"wheel_speed": [float(x) for x in wheel_q], ...}
```

```python
qwheel = np.asarray(thresholds["wheel_speed"])
...
np.searchsorted(qwheel, s["wheel"][j], side="right").astype(np.int8),
```

iii. <Justification>

Notes Step 5 decision 8: "Behavior discretization: Use deterministic global 1/3 and 2/3 quantiles
over all retained, aligned samples. Global thresholds keep class definitions physically
consistent across sessions and approximately balance classes. Store exact thresholds in metadata
and assert they are distinct and every class is represented." The reference code does not
discretize at all (it regresses continuous wheel speed), so this is an addition required by the
Decoder Task line "Wheel speed discretized into 3 bins".

---

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. <Decisions>

The wheel trace is evaluated at exactly the same 100 time points that label the neural bins —
`stimOn + RELATIVE_BIN_ENDS` — so index `k` of the wheel output and column `k` of the neural
matrix refer to the same trial-relative instant (the closing edge of bin `k`). No independent
alignment step is needed because the wheel timestamps are on the same session clock as the
spikes. Trials whose window is not spanned by the wheel stream (within one bin at each edge) were
already dropped by `coverage_mask`.

ii. <Code snippets>

```python
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
if not (np.all(np.isfinite(wheel)) and np.all(np.isfinite(whisker))):
    return None, "nonfinite aligned behavior"
```

iii. <Justification>

Notes Step 7 "Processing Plots Review": "Wheel position, Butterworth-filtered speed, and 20 ms
samples are synchronized to stimulus onset… No temporal offset, clipping, missing classes, or
anomalous raster artifact was found." Step 12 check 2 re-inspected the plots plus
`sample_trials.png` and `predictions.png` and reported "Neural responses and behavior transitions
are synchronized; predictions track dynamic transitions without a consistent lag."

---

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. <Decisions>

From the released per-frame ROI motion energy of a side camera: `leftCamera.ROIMotionEnergy.npy`
with `_ibl_leftCamera.times.npy`, falling back to the right camera when the left pair is absent.
The trace is used exactly as released (no recomputation from video, no DLC). A session lacking
*both* paired streams is excluded — this accounts for 14 of the 15 excluded sessions. The chosen
view is recorded per session in `metadata['session_info'][i]['motion_energy_view']`.

ii. <Code snippets>

```python
def choose_motion_stream(alf: Path) -> tuple[str, Path, Path] | None:
    for view, revision in (("left", "#2025-05-29#"), ("right", "#2025-05-31#")):
        times = _optional_file(alf, f"_ibl_{view}Camera.times.npy")
        values = _optional_file(alf, f"{view}Camera.ROIMotionEnergy.npy", revision)
        if times is not None and values is not None:
            return view, times, values
    return None
```

```python
motion = choose_motion_stream(spec.session_dir / "alf")
if motion is None:
    return None, "missing paired left and right whisker-motion streams"
```

iii. <Justification>

Notes Step 4: "Camera choice — Code prefers left whisker motion energy and falls back to right…
Use left-first/right-fallback exactly as code; record view in session metadata; omit sessions
with neither." This mirrors the reference `bin_behaviors`, which calls
`load_target_behavior(one, eid, 'left-whisker-motion-energy')` and retries with the right view if
the left one is unavailable. Step 3 records the paper definition ("mean absolute adjacent-frame
difference inside a whisker-pad bounding box anchored between DLC nose and eye locations") as
already applied upstream by IBL.

---

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. <Decisions>

No filtering, smoothing or normalisation. The released trace is validated (matching shapes, ≥2
samples, strictly increasing timestamps, all values finite — otherwise the session is dropped),
then resampled with the same `interval_interpolate` used for the wheel: linear interpolation onto
the 100 bin-end times, with the last bin linearly extrapolated to reproduce the reference's
end-exclusive interval. Then discretized (8-c).

ii. <Code snippets>

```python
motion_times = np.asarray(np.load(motion_times_path, mmap_mode="r"), dtype=np.float64)
motion_values = np.asarray(np.load(motion_values_path, mmap_mode="r"), dtype=np.float64)
if motion_times.shape != motion_values.shape or motion_times.size < 2:
    return None, "invalid whisker-motion stream shape"
if not (np.all(np.diff(motion_times) > 0) and np.all(np.isfinite(motion_values))):
    return None, "nonmonotonic/nonfinite whisker-motion stream"
```

```python
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
```

iii. <Justification>

Notes Step 5 mapping table: "`{left,right}Camera.ROIMotionEnergy` and matching camera times →
`output[3, :]`: Prefer left, fall back right; linearly sample at bin ends; discretize by global
tertiles". Step 3 notes the source rates differ (left ~60 Hz, right ~150 Hz) and that "reference
caching code linearly resamples behavior to 20 ms bins", so resampling is the only processing
required.

---

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. <Decisions>

Identical rule to the wheel: **global** 1/3 and 2/3 quantiles computed once over every aligned
whisker sample of every retained session, giving thresholds 2.7334 and 7.8455 (arbitrary motion
energy units), applied uniformly to all sessions. Globally the three classes are balanced to
within rounding; per session they are not — the verification log's per-session class-0 fractions
for `whisker_motion_energy` span 0.045 to 0.996, and some sessions have an output range of only
`[0, 1]` (class 2 never occurs).

ii. <Code snippets>

```python
whisker_values = np.concatenate([s["whisker"].ravel() for s in behavior_sessions])
whisker_q = np.quantile(whisker_values, [1 / 3, 2 / 3])
```

```python
qwhisker = np.asarray(thresholds["whisker_motion_energy"])
...
np.searchsorted(qwhisker, s["whisker"][j], side="right").astype(np.int8),
```

```python
"continuous_output_discretization": "global tertiles over all retained aligned time samples",
"discretization_thresholds": thresholds,
```

iii. <Justification>

Same as 7-c — Step 5 decision 8: "Global thresholds keep class definitions physically consistent
across sessions and approximately balance classes." The AI's validation checked only the
*global* balance (Step 9: "Whisker tertile fractions [0.333333, 0.33333328, 0.33333339] — Exact
up to ties/count indivisibility") and the presence of all three classes overall
(`internal_validate` asserts `np.all(counts > 0)` on pooled counts); it did not check per-session
balance. The AI reported whisker validation balanced accuracy of 0.7208, markedly higher than the
other three outputs, and attributed this to the variable being intrinsically well decodable
rather than to the thresholding scheme.

---

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. <Decisions>

Same mechanism as the wheel: camera frame times are on the shared session clock, so the trace is
simply evaluated at `stimOn + RELATIVE_BIN_ENDS`, the 100 instants that label the neural bins.
`coverage_mask` has already guaranteed the camera spans the whole window with its first and last
in-window frames within one 20 ms bin of the edges, so no extrapolation beyond one bin is ever
needed.

ii. <Code snippets>

```python
motion_good, _, motion_ie = coverage_mask(motion_times, begins, ends)
stream_good = wheel_good & motion_good
...
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
```

iii. <Justification>

Notes Step 7: "camera samples overlay raw motion energy" in the processing plots, with "no
temporal offset" found; Step 10 reference comparison: "Same `stimOn_times`, -0.5 to +1.5 s window
and right-edge behavior grid."

---

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. <Decisions>

A layered policy, all of it logged:

- **Structural problems raise**: mismatched spike array shapes, non-monotonic spike times (first
  1e6 samples checked), cluster indices out of range, choice/prior values outside the expected
  sets — these raise `ValueError` rather than being silently patched.
- **Session-level missing data → exclude the session, with a recorded reason**: no paired camera
  motion-energy stream; malformed/non-monotonic/non-finite camera stream; malformed wheel stream;
  fewer than two trials surviving either filter stage; any non-finite aligned behaviour. The
  15 exclusions are written to `metadata['excluded_sessions']` and printed to the log.
- **Trial-level missing data → drop the trial**: NaN trial events via the reference mask; wheel
  or camera coverage gaps via `coverage_mask`.
- **Recording-coverage zeros are kept, not patched**: three trials in session 392 have all-zero
  neural windows. The AI traced this to raw data (spike recording ends at 1779.36 s while the
  three stimulus onsets are at 1790.0/1796.5/1799.8 s) and deliberately kept them.
- **Top-level per-session try/except** converts any unexpected exception into an exclusion rather
  than aborting the run.
- `internal_validate` then asserts finiteness, shape and dtype consistency, non-negative
  integer-valued counts, and non-empty categorical classes across the whole dataset.

ii. <Code snippets>

```python
try:
    behavior, reason = prepare_behavior(spec, capture_plot=capture)
except Exception as exc:
    behavior, reason = None, f"behavior processing error: {type(exc).__name__}: {exc}"
if behavior is None:
    excluded.append({"eid": spec.eid, "reason": str(reason)})
    print(f"Excluded {spec.eid}: {reason}", flush=True)
    continue
```

```python
for x, u, y in zip(neural, inputs, outputs):
    assert x.shape[1] == u.shape[1] == y.shape[1] == N_BINS
    assert x.dtype == np.float32 and u.dtype == np.float32 and y.dtype == np.int8
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(u)) and np.all(np.isfinite(y))
    assert np.all(x >= 0) and np.allclose(x, np.round(x))
for counts in class_counts:
    assert np.all(counts > 0), f"Empty categorical class: {counts}"
```

iii. <Justification>

Notes Step 10 check 3: "Session 392 is EID `8c2f7f4d-…`. Direct raw counts across its sole probe
are exactly zero for all three warned windows. The spike recording ends at 1779.359643 s, whereas
the warned stimulus onsets are 1789.999367, 1796.482300, and 1799.791400 s. Thus these zeros are
source recording coverage, not a binning bug. The reference counter also returns zeros and
defines no neural-coverage trial filter, so the three trials remain; the warning cannot be
removed without changing reference curation." Step 5 decision 3 covers the trial-level drops;
Step 4 covers excluding sessions with no whisker stream ("omit sessions with neither").

---

## 10-a. What are the most time-consuming steps of the code?

i. <Decisions>

The AI instrumented every stage (`time.perf_counter` around the behaviour pass, the spike pass,
and the pickle write) and printed per-session timings. On the full run (455.48 s wall clock,
8 threads):

| Stage | Cost |
|---|---|
| `build_release_specs` (recursive globs + opening 699 `clusters.channels` headers) | ~14 s, serial |
| Behaviour pass, 459 sessions serial | ~0.14–0.35 s/session ⇒ ~100 s |
| Spike binning (memory-mapped reads + bincount), 444 sessions on 8 threads | ~0.5–5 s/session of worker time; the dominant CPU/IO cost |
| `pickle.dump` of the 98.8 GiB output | 99.45 s |

So the two real bottlenecks are spike-file I/O and writing the very large pickle; the latter is a
direct consequence of retaining ~600k units (decision 2-c).

ii. <Code snippets>

```python
behavior["spike_seconds"] = time.perf_counter() - t0
print(f"Binned {spec.eid}: {n_trials} trials, {n_neurons} neurons, "
      f"behavior {behavior['behavior_seconds']:.2f}s, spikes {behavior['spike_seconds']:.2f}s",
      flush=True)
```

```python
write_start = time.perf_counter()
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
write_seconds = time.perf_counter() - write_start
...
print(f"Wrote {args.outpicklefile} ({gib:.3f} GiB) in {write_seconds:.2f}s", flush=True)
print(f"Total conversion time: {total_seconds:.2f}s", flush=True)
```

iii. <Justification>

Notes Step 7 "Run Time Estimates" tabulates per-stage times and projects ~5–8 min for the full
run (actual: 7.6 min, inside the instruction's 15-minute budget). Step 6: "Physical data volume
under the lab roots is approximately 571 GB, dominated by 21.15 billion spike events;
memory-mapped loading and session-wise processing are required."

---

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. <Decisions>

The AI identified and removed the reference code's dominant per-trial loop — `compute_spike_count`
in `get_spike_data_per_interval` performs a full-session boolean scan `(times >= t_beg) & (times <
t_end)` once per trial — replacing it with a single `np.searchsorted` for all trial boundaries plus
one `np.bincount` per trial over a flattened `cluster*100 + bin` index.

Loops that remain un-vectorised in the AI's own script, and were *not* flagged in its notes:
- the per-trial loop in `bin_spikes_for_session` (could be a single `bincount` over all trials by
  adding a per-trial offset to the flat index);
- the per-trial loop in `assemble_data` that stacks the `(2,100)` input and `(4,100)` output
  arrays (could be built as whole-session 3-D arrays and sliced);
- the per-run Python loop in `trial_number_in_block` (expressible as
  `arange(n) - repeat(starts, lengths)`);
- the serial 459-session behaviour pass, which is not parallelised even though the spike pass is.

All of these are cheap relative to file I/O.

ii. <Code snippets>

```python
left = np.searchsorted(spike_times, begins, side="left")
right = np.searchsorted(spike_times, ends, side="left")
for j, (i0, i1, begin) in enumerate(zip(left, right, begins)):   # remaining per-trial loop
    ...
    counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
```

```python
for j in range(n_trials):                                        # remaining per-trial loop
    inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                             np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

```python
for start, end in zip(starts, ends):                             # remaining per-run loop
    out[start:end] = np.arange(end - start, dtype=np.int32)
```

iii. <Justification>

Notes Step 6, "Code inefficiencies identified": "The reference `get_spike_data_per_interval`
applies a full-session Boolean comparison separately for every trial, which is prohibitively
expensive for 21.15 billion source spike events." "Code speedups added": "Binary-search each trial
boundary in sorted memory-mapped spike times, process only spikes inside retained windows, count
flattened cluster×bin indices with compiled `np.bincount`, process probes without concatenating
session-wide spike arrays, and parallelize independent sessions with a bounded thread pool."
The notes do not analyse the residual loops.

---

## 10-c. What processing does the code repeat multiple times?

i. <Decisions>

The AI's notes do not identify any repeated processing; its efficiency discussion is confined to
the speedups above. Reading the script, several repeats are present:

- `clusters.channels.npy` is opened once per probe in `build_release_specs` (to count clusters)
  and opened again per probe in `bin_spikes_for_session`.
- `BrainRegions()` — which loads the Allen/Beryl atlas tables — is constructed **inside**
  `bin_spikes_for_session`, i.e. once per session (444 times) rather than once per process.
- The in-script raw spot check at the end of every session re-opens probe 0's `spikes.times` and
  `spikes.clusters` and runs three full-array boolean comparisons over the entire probe's spike
  train, repeating work already done by the binning loop, for every one of the 444 sessions.
- `interval_interpolate` computes the final column twice: once inside the whole-grid `np.interp`
  and again via the explicit extrapolation that overwrites it.
- `_find_file` performs an independent recursive `glob('**/name')` per attribute per session/probe
  (≈6 walks per session plus 4 per probe) over the same directory trees.

ii. <Code snippets>

```python
def bin_spikes_for_session(behavior):
    ...
    n_per_probe = [int(np.load(p.clusters_channels, mmap_mode="r").shape[0]) for p in spec.probes]
    ...
    br = BrainRegions()          # rebuilt for every session
```

```python
# Direct independent spot check against original data for one count.
p0 = spec.probes[0]
st = np.load(p0.spikes_times, mmap_mode="r")
sc = np.load(p0.spikes_clusters, mmap_mode="r")
direct = np.sum((st >= begins[0]) & (st < begins[0] + BIN_SIZE) & (sc == 0))
if not np.allclose(neural[0][0, 0], direct):
    raise AssertionError(f"{spec.eid}: raw neural spot check failed")
```

```python
result = np.interp(targets.ravel(), times, values).reshape(targets.shape)
...
result[:, -1] = y1 + (interval_ends - x1) * (y1 - y0) / (x1 - x0)   # last column recomputed
```

iii. <Justification>

The repeats are all in service of correctness guarantees the AI chose to keep permanently rather
than run once: Step 5 planned sanity check "For at least three trials, independently compute spike
counts from original arrays and require `np.allclose` to converted neural values", and Step 10
check 2 describes the independent audit. The spot check was strengthened during Step 10 ("Initial
weak neural audit points were zero: Strengthened the audit to choose an active neuron
independently in bins 0, 10, and 99"). No note argues for removing it after validation.

---

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. <Decisions>

The notes claim none. In the script the following work is produced and then discarded or unused:

- **Wheel acceleration** — `velocity_filtered` returns `(vel, acc)`; the acceleration is thrown
  away (`wheel_velocity, _ = ...`).
- **Wheel position** — interpolated to 1 kHz and retained only so the `--show-processing` plot can
  draw it; unused in `--full` runs (though it is a necessary intermediate for velocity).
- **The per-session raw spot check** (see 10-c) — a full-probe scan whose only product is an
  assertion.
- **Duplicate last-column interpolation** in `interval_interpolate`.
- **`source_trial_indices`** — the full list of original trial indices for all 188,925 trials is
  serialised into `metadata`; nothing downstream reads it.
- **Replicated constants** — `input[0]` is the *same* 100-value vector for all 188,925 trials, and
  `output[0]`/`output[1]` are per-trial scalars replicated 100×. This is forced by the target
  format, but it means a large fraction of the stored payload is redundant.
- **Retaining ~600k unfiltered units** (decision 2-c) is the single largest contributor: most are
  low-quality or outside the brain (12,827 `void`, 344 `x`, 18 `y`), and it drives the 98.8 GiB
  artifact and the 99 s write.

ii. <Code snippets>

```python
wheel_velocity, _ = velocity_filtered(wheel_position, fs=1000, corner_frequency=20, order=8)
```

```python
"source_trial_indices": s["source_trial_idx"].astype(int).tolist(),
```

```python
inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                         np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

iii. <Justification>

Where the AI does justify these, it is on provenance/auditability grounds: Step 5 decision 3
("preserving their original indices/block counts") and decision 7 (constant repeats are "the only
shape-consistent faithful encoding" given a single `(4, 100)` output array); decision 9 covers the
dtype choices ("float32… and outputs as int8 categorical labels") that keep the size down as far
as the all-unit decision allows. The plotting intermediates are gated by `capture_plot` and only
computed for the first two sessions when `--show-processing` is passed.
