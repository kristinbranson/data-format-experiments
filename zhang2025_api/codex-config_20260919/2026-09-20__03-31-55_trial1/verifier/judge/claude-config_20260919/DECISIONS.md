# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything scientific is read through the ONE API and the `brainbox` loaders; no file in `/app/data` is opened directly. The agent builds a *composite* in-memory ONE index rather than using a single release tag: it instantiates the local-only client `one.api.One(cache_dir=/app/data/one_cache)` twice, loading the `Brainwidemap` release tables into one and the `2025_Q3_IBL_et_al_BWM` release tables into the other, then concatenates the two `datasets` tables and de-duplicates keeping the newer row. The candidate session set is the *intersection* of the eids present in both releases (459 sessions), optionally restricted further by `/app/data/DATALIMIT_SUBSET.csv` if present. Because the frozen Brainwidemap index lists the pre-revision path for the trials table while the file staged on disk is the `#2025-03-03#` revision, the agent rewrites `rel_path` for every `alf/_ibl_trials.table.pqt` row in the in-memory index to the hard-coded revision folder and blanks out `hash`/`file_size` so ONE's integrity check is skipped. From that point everything follows from an `eid`: `SessionLoader` supplies trials/wheel/motion energy and `SpikeSortingLoader` supplies spikes/clusters/channels, once per probe. Probes are discovered from `one.list_collections(eid)` by looking for `alf/probeNN/.../pykilosort` collections rather than by `one.eid2pid` (an Alyx call the local `One` client cannot make).

ii.
```python
one = One(cache_dir=CACHE_ROOT)
one.load_cache(SPIKE_RELEASE)
behavior_one = One(cache_dir=CACHE_ROOT)
behavior_one.load_cache(BEHAVIOR_RELEASE)

datasets = pd.concat([one._cache["datasets"], behavior_one._cache["datasets"]])
datasets = datasets[~datasets.index.duplicated(keep="last")].sort_index()

trial_rows = datasets["rel_path"].eq("alf/_ibl_trials.table.pqt")
datasets.loc[trial_rows, "rel_path"] = (
    f"alf/#{TRIAL_REVISION}#/_ibl_trials.table.pqt"
)
datasets.loc[trial_rows, "hash"] = ""
datasets.loc[trial_rows, "file_size"] = 0
datasets.loc[trial_rows, "default_revision"] = True
one._cache["datasets"] = datasets

spike_eids = set(map(str, one.search()))
behavior_eids = set(map(str, behavior_one.search()))
eids = sorted(spike_eids & behavior_eids)
```

```python
def probe_names(one: One, eid: str) -> list[str]:
    names = set()
    for collection in one.list_collections(eid):
        parts = collection.split("/")
        if len(parts) >= 3 and parts[0] == "alf" and parts[1].startswith("probe"):
            if "pykilosort" in parts:
                names.add(parts[1])
    return sorted(names)
```

```python
loader = SpikeSortingLoader(eid=eid, pname=pname, one=one)
spikes, clusters, channels = loader.load_spike_sorting()
...
sess = SessionLoader(one=one, eid=eid)
sess.load_trials(revision=TRIAL_REVISION)
sess.load_wheel()
sess.load_motion_energy(views=[view])
```

iii. From CONVERSION_NOTES.md Step 4: "Base `_ibl_trials.table.pqt` index entries point to unstaged paths; actual files are revision `#2025-03-03#` … Patch only the in-memory ONE dataset `rel_path` to the known staged revision, then load with ONE. Test loads returned all required columns and expected 0.5 initial blocks. No direct data-file reading is used." And: "Hash/size warnings arise because the old indexed row describes the superseded base file; conversion will disable hash checking for this known revision mapping and perform shape/value checks instead." The 2025 release is merged in because "The 2025 table contains revised video products."

---

## 1-b. How are the data split into subjects?

i. A session's subject name is taken from `one.get_details(eid)` and stored in the per-session info dict. `subjects` is the list of unique subject names in *first-seen* order (not sorted), and `subject_idx` is the index of each retained session's subject into that list. Only sessions that survived conversion contribute subjects; the final dataset has 135 subjects over 441 sessions (4 of the release's 139 mice appear only in excluded sessions).

ii.
```python
details = one.get_details(eid, full=False)
...
info = {"eid": eid, "subject": str(details["subject"]), "lab": str(details["lab"]), ...}
```

```python
subject_names = list(dict.fromkeys(subjects))
subject_lookup = {name: i for i, name in enumerate(subject_names)}
...
"subjects": subject_names,
"subject_idx": np.asarray([subject_lookup[x] for x in subjects], dtype=np.int32),
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "ONE session details `subject` → `subjects`, `subject_idx`; Unique subject strings in first-seen order and per-session integer index; Only retained sessions represented." The subject identity is supplied by the API, so nothing has to be parsed or derived.

---

## 1-c. How are the data split into sessions?

i. No splitting is performed: a session *is* an `eid` in the ONE index. The candidate list is the sorted intersection of eids in the two release tables (459), and each eid is converted independently by `convert_session`. Sessions that raise (missing camera, missing `probabilityLeft`, collapsed tertiles, <2 valid trials) are recorded in `metadata['failed_sessions']` and skipped; 441 sessions remain.

ii.
```python
eids = sorted(spike_eids & behavior_eids)
...
indexed_eids = list(enumerate(eids))
results = executor.map(attempt, indexed_eids)
...
for eid, converted, error in results:
    if error is not None:
        failures.append({"eid": eid, "error": error})
        print(f"SKIP {eid}: {error}", flush=True)
        continue
```

iii. Step 4 of CONVERSION_NOTES.md: "Start from 459 task EIDs; retain only sessions passing trial filters with both requested continuous outputs and ≥2 valid trials. This should approach the method paper's 433 usable sessions." The API returns one eid per session so there is no decision to make about splitting.

---

## 1-d. How are the data split into trials?

i. No splitting is performed: the trials table returned by `SessionLoader.load_trials()` has one row per trial, and the conversion indexes that table with a boolean mask. Each retained row becomes one trial window `[stimOn_times - 0.5, stimOn_times + 1.5]`.

ii.
```python
sess.load_trials(revision=TRIAL_REVISION)
...
stim = trials["stimOn_times"].to_numpy(dtype=float)
...
selected = np.flatnonzero(keep)
```

iii. Not discussed as a decision in CONVERSION_NOTES.md — the trials table is already one row per trial, and the reference code (`load_trials_and_mask`) uses the same table.

---

## 1-e. How are trials filtered based on quality controls?

i. Six filters, ANDed together. (1) All of `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`, `goCue_times` must be finite. (2) Reaction time `firstMovement_times − stimOn_times` must be in [0.08, 2.0] s. (3) Trial duration `feedback_times − goCue_times` ≤ 10 s. (4) `choice ∈ {−1, +1}` (no-response trials dropped). (5) `probabilityLeft ∈ {0.2, 0.5, 0.8}`. (6) Both the wheel trace and the whisker motion-energy trace must cover the full 2 s window with no gap larger than one bin at either edge (enforced by `interpolate_trials` returning `good=False`). A seventh filter is applied after binning: trials whose entire population spike matrix is zero over the 2 s are treated as falling outside a valid recording interval and removed. Sessions left with <2 trials are dropped. The unbiased 0.5 block is *retained*. Net effect: 186,844 of ~295,675 trials kept, mean 424 per session.

ii.
```python
def trial_mask(trials: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(trials), dtype=bool)
    for col in REQUIRED_TRIAL_COLUMNS:
        mask &= np.isfinite(trials[col].to_numpy(dtype=float))
    rt = trials["firstMovement_times"].to_numpy() - trials["stimOn_times"].to_numpy()
    duration = trials["feedback_times"].to_numpy() - trials["goCue_times"].to_numpy()
    choice = trials["choice"].to_numpy()
    prior = trials["probabilityLeft"].to_numpy()
    mask &= (rt >= 0.08) & (rt <= 2.0)
    mask &= duration <= 10.0
    mask &= np.isin(choice, (-1, 1))
    mask &= np.isin(np.round(prior, 6), (0.2, 0.5, 0.8))
    return mask
```

```python
keep = base_mask & wheel_good & whisker_good
selected = np.flatnonzero(keep)
if len(selected) < 2:
    raise RuntimeError(f"only {len(selected)} valid trials")
```

```python
neural_valid = np.asarray([np.any(x) for x in neural], dtype=bool)
if not np.all(neural_valid):
    selected = selected[neural_valid]
    neural = [x for x, valid in zip(neural, neural_valid) if valid]
```

iii. Step 5 Key Decision 3: "Apply the paper/code intersection: required non-NaN events, RT 0.08–2.0 s inclusive, feedback−goCue ≤10 s, choice nonzero, valid prior code, complete wheel and whisker coverage of the full interval, and finite interpolated samples. Keep sessions only with ≥2 surviving trials." These reproduce `load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2., max_trial_len=10.0, exclude_nochoice=True)` as called by the reference `prepare_data`, plus the boundary-coverage requirement of the reference `get_behavior_per_interval`. Step 10 Iteration 1: "three adjacent trials in one session had population-wide zero counts over two seconds. These were treated as invalid recording periods, removed before behavior thresholds were fit." Unbiased trials are kept because "the requested prior output explicitly includes class 0.5" (`exclude_unbiased=False` is also the reference default).

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, one call per pykilosort probe collection. `clusters['acronym']` (after `merge_clusters` attaches histology) is used only for the region labels, and `clusters['label']` only for a metadata count of well-isolated units — neither gates which spikes enter the neural array. `clusters['channels']` is used to get the cluster count for the per-probe index offset.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
n_clusters = len(clusters["channels"])
...
spike_cluster = np.asarray(spikes["clusters"], dtype=np.int64)
valid = (spike_cluster >= 0) & (spike_cluster < n_clusters)
all_times.append(np.asarray(spikes["times"], dtype=np.float64)[valid])
all_clusters.append(spike_cluster[valid] + offset)
all_regions.append(np.asarray(clusters["acronym"], dtype=object).astype(str))
label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
all_good.append(label >= 1)
```

iii. Step 5 mapping table: "`spikes.times`, `spikes.clusters` across all session probes → `neural`". Step 10 Check 6: "both use `SpikeSortingLoader.load_spike_sorting`, `merge_clusters`, `SessionLoader`, and ONE. Scientific files are never opened directly."

---

## 2-b. How is the `neural` data processed?

i. Probes within a session are merged into one population: each probe's cluster ids are offset by the running cluster count, spike times and cluster ids are concatenated and stable-sorted by time (reproducing the reference `merge_probes`). Spikes are then counted into 100 half-open 20 ms bins spanning [stimOn − 0.5, stimOn + 1.5) per trial using one flattened `np.bincount` per trial. The result is stored as **raw spike counts** cast to `float32` — no conversion to a firing rate, no smoothing, no standardisation (the decoder standardises downstream). Shape per trial is (n_neurons, 100).

ii.
```python
times = np.concatenate(all_times)
cluster_ids = np.concatenate(all_clusters)
order = np.argsort(times, kind="stable")
```

```python
def bin_spikes(times, clusters, stim_times, n_clusters):
    edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
    trials = []
    for stim in np.asarray(stim_times, dtype=np.float64):
        edges = stim + edges_rel
        lo = np.searchsorted(times, edges[0], side="left")
        hi = np.searchsorted(times, edges[-1], side="left")
        st = times[lo:hi]
        sc = clusters[lo:hi]
        tb = np.searchsorted(edges, st, side="right") - 1
        valid = (tb >= 0) & (tb < N_BINS)
        flat = sc[valid] * N_BINS + tb[valid]
        count = np.bincount(flat, minlength=n_clusters * N_BINS)
        trials.append(count.reshape(n_clusters, N_BINS).astype(np.float32))
    return trials
```

iii. Step 5 Key Decision 2: "Preserve integer spike counts numerically but store float32 as requested by the validator. Do not convert counts to rates or smooth them; the reference cache and decoder consume counts." Merging probes follows the reference rationale quoted in Step 1 ("data from the probes recorded in the same session are not statistically independent"), which is also the data paper's stated practice ("neurons in the same session and region were combined across probes"). Step 10 Check 8: "Spike bins are half-open to prevent boundary double-counting."

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No quality filter is applied.** Every Kilosort cluster on every probe is retained, including multi-unit clusters and units whose Beryl acronym is `void` (i.e. histologically placed outside the brain) or `root`. The only spikes discarded are those whose cluster index is out of range of the cluster table. `label >= 1` is computed but only recorded as the metadata count `n_good_units`. The converted dataset therefore contains 594,965 session-neurons (mean 1,349 per session) including 85,237 `root` and 12,758 `void` units, versus the 75,708 "well-isolated neurons" the data paper reports after its stringent QC.

ii.
```python
label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
all_good.append(label >= 1)          # metadata only, never used to filter
...
spike_cluster = np.asarray(spikes["clusters"], dtype=np.int64)
valid = (spike_cluster >= 0) & (spike_cluster < n_clusters)
```

```python
"n_good_units": int(np.sum(good_units)),
...
"neuron_filter": "all Kilosort clusters (reference decoder cache qc=None)",
```

iii. Step 4 discrepancy table, "Neuron QC" row: "Supplied cache calls `qc=None`; stores `label >= 1` only as metadata … Method paper says 'all neurons'; data paper anatomical analyses use 75,708 well-isolated units → Retain all clusters, matching the decoder paper and exact supplied cache path. Record quality in metadata statistics; do not silently substitute the stricter anatomical-analysis population." The methods.txt quote relied on is "we bin spike counts using all neurons, sorted by Kilosort 2.5, from each session", and `0_data_caching.py` indeed calls `load_spiking_data(one, pid, ...)` with the default `qc=None`. Step 5 mapping note: "`void`/root values retained explicitly rather than dropping neurons."

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams (spike times, trial event times, wheel timestamps, camera frame times) already live on one synchronised session clock in seconds, so alignment is a subtraction. For each retained trial the bin edge grid `[-0.5, -0.48, …, 1.5]` is added to that trial's `stimOn_times`, and spikes are assigned to bins on that absolute grid. `stimOn_times` is also the event used for the wheel and whisker interpolation windows, so all four streams are aligned to the same instant.

ii.
```python
edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
for stim in np.asarray(stim_times, dtype=np.float64):
    edges = stim + edges_rel
    lo = np.searchsorted(times, edges[0], side="left")
    hi = np.searchsorted(times, edges[-1], side="left")
    tb = np.searchsorted(edges, times[lo:hi], side="right") - 1
```
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
```
```python
"temporal_alignment_event": "visual stimulus onset (trials.stimOn_times)",
"off_start": OFF_START,   # -0.5
"off_end": OFF_END,       #  1.5
```

iii. Step 4: "Cache code: stimulus onset, −0.5:+1.5 s, 20 ms … user explicitly requires stimulus onset and a common grid → Use the supplied cache convention." Step 7 plot review: "t=0 stimulus marker lies at the expected location … no discontinuity or one-bin temporal shift is visible." Step 10 Check 2 reports an independent `np.allclose` reconstruction of the full neuron-by-time count matrix for a spot-checked trial.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (`time_bin_size = 20.0` ms in metadata), 100 bins covering −0.5 to +1.5 s, identical for every trial and session. Spikes are binned once directly from spike times, so there is no rebinning, resampling or interpolation of the neural data. The 50 ms bins that the data paper uses for its static choice/prior decoding were explicitly rejected in favour of the 20 ms cache convention of the reference code.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
```
```python
"time_bin_size": 20.0,
"neural_representation": "unsmoothed spike counts in half-open 20-ms bins",
```

iii. Step 3: "The methods-paper cache representation is a 2-s interval with 100 non-overlapping 20-ms spike-count bins… Because the requested target requires one common temporal grid and explicitly says stimulus-onset alignment, the code's 2-s/20-ms cache convention is the directly applicable reference." This matches `params = {'interval_len': 2, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` in `0_data_caching.py` and the methods.txt sentence "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps."

---

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable at all; it is the fixed relative time grid defined by the alignment event `trials.stimOn_times` and the chosen window. The values are the **bin-end** times of the 100 bins, i.e. −0.48, −0.46, …, +1.50 s, identical for every trial and session, matching the reference code's `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

ii.
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
...
inp = np.vstack([
    BIN_END_TIMES.astype(np.float32),
    np.full(N_BINS, block_num[trial_idx], dtype=np.float32),
])
```

iii. Step 5 mapping table: "Relative bin-end times → `input[0]`; Fixed float32 vector −0.48, −0.46, …, 1.50 s; Reference code function `get_behavior_per_interval`; 'time since stimulus onset', continuous/time-varying." The bin-end convention is taken directly from the reference's interpolation grid rather than bin centres.

---

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the constant vector once at module import and broadcasting it into every trial's input array. It is stored as `float32`.

ii.
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
```

iii. The variable is defined by the conversion, not measured. CONVERSION_NOTES.md Step 7 notes the reported range: "Time since onset range | [-0.48, 1.50] internally (verifier rounds to [-0.5, 1.5])".

---

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the same grid as the neural binning: bin *i* of the neural matrix covers `[stimOn + OFF_START + i·0.02, stimOn + OFF_START + (i+1)·0.02)` and `BIN_END_TIMES[i]` is the right edge of exactly that bin. Both are built from the same `OFF_START`/`BIN_SIZE`/`N_BINS` constants, so there is no possibility of drift between them; the same vector is also the interpolation grid for the wheel and whisker outputs.

ii.
```python
edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START   # neural bin edges
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START   # = edges_rel[1:]
```

iii. Step 10 Check 8: "Conversion matches these and reference `get_behavior_per_interval` bin-end interpolation." Step 10 Check 4: "independently reconstructed bin-end grid and zero-based within-block ordinal; exact `np.allclose` pass."

---

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials['probabilityLeft']` alone. The trials table carries no block identifier, so blocks are recovered as maximal runs of constant `probabilityLeft`, and the input is the position of the trial within its run.

ii.
```python
block_num = block_trial_numbers(trials["probabilityLeft"].to_numpy())
```

iii. Step 5 mapping table: "Run length within consecutive `probabilityLeft` block → `input[1]`". Step 3 records the expected block structure: "First 90 trials at 0.5; then 0.2/0.8 blocks, 20–100 trials, empirical mean 51."

---

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based ordinal that increments while `probabilityLeft` is unchanged from the previous row and resets to 0 whenever it changes, computed with a scalar Python loop over the **unfiltered, chronological** trials table. The value is then broadcast across all 100 time bins and cast to `float32`. Because it is computed before trial exclusions, a trial that is later dropped still advances the count, so the number reflects the animal's real position in the block. The observed full-dataset range is [0, 98], consistent with the 20–100 trial block lengths and the 90-trial unbiased block.

ii.
```python
def block_trial_numbers(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based ordinal within runs of constant block probability."""
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.int32)
    for i in range(1, len(p)):
        out[i] = out[i - 1] + 1 if p[i] == p[i - 1] else 0
    return out
```
```python
block_num = block_trial_numbers(trials["probabilityLeft"].to_numpy())   # before trial_mask
base_mask = trial_mask(trials)
...
np.full(N_BINS, block_num[trial_idx], dtype=np.float32)
```

iii. Step 5 Key Decision 6: "Use zero-based ordinal calculated on the original chronological trial table before trial exclusions, so removing an invalid trial does not renumber later valid trials or invent shorter blocks." Step 5 mapping note: "The initial 0.5 block is treated as a block."

---

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single column `trials['choice']`, which takes values +1, −1 and 0. Trials with `choice == 0` (no response) have already been removed by `trial_mask`, so only ±1 reach the encoder. The agent maps **−1 → 0 ("left") and +1 → 1 ("right")**, on the stated belief that "IBL convention is choice −1 = left and +1 = right". The value is broadcast over all 100 bins as `int8`; `output_values[0] = ['left', 'right']`.

ii.
```python
mask &= np.isin(choice, (-1, 1))
...
choice = 0 if trials.iloc[trial_idx]["choice"] == -1 else 1
```
```python
"output_values": [["left", "right"], ...]
```

iii. Step 4 discrepancy table, "Choice sign" row: "Native IBL choice is −1/0/+1; code removes 0 | Loaded data contain both −1 and +1 | Decoder task requires left=0, right=1 | IBL convention is choice −1 = left and +1 = right; map −1→0, +1→1 after excluding 0/missing."

---

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recoding described above, applied per trial inside the assembly loop, followed by broadcasting to (100,) and stacking into the 4×100 output array as `int8`. No other transform.

ii.
```python
for row, trial_idx in enumerate(selected):
    choice = 0 if trials.iloc[trial_idx]["choice"] == -1 else 1
    out = np.vstack([
        np.full(N_BINS, choice, dtype=np.int8),
        ...
    ])
```

iii. Step 5 mapping table: "`trials.choice` → `output[0]`; Exclude missing/0; map native −1 (left)→0 and +1 (right)→1; broadcast across time; Static trial target represented on common time grid." Step 12 reports that three specific trials were re-loaded from source and "their choice classes exactly matched converted values" — the check validates the code against the agent's own mapping, not the mapping itself.

---

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The single column `trials['probabilityLeft']`, rounded to 6 decimals and mapped through `{0.2: 0, 0.5: 1, 0.8: 2}` exactly as the task specifies. Trials whose value is not one of the three is dropped by `trial_mask`. The latent/inferred continuous "prior belief" that the methods paper also studies was deliberately not used.

ii.
```python
mask &= np.isin(np.round(prior, 6), (0.2, 0.5, 0.8))
...
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
p = float(np.round(trials.iloc[trial_idx]["probabilityLeft"], 6))
```

iii. Step 4, "Prior definition" row: "Cache code stores trial `probabilityLeft` as `block`; methods paper also studies an inferred continuous belief prior … Use categorical `probabilityLeft`, not the separately modeled latent belief, because the requested coding is exact and unambiguous."

---

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the rounding + dictionary lookup, broadcast across the 100 bins as `int8`. The unbiased 0.5 block is kept (it is class 1), which is why the full-dataset class distribution is roughly [0.42, 0.14, 0.44] rather than balanced.

ii.
```python
out = np.vstack([
    np.full(N_BINS, choice, dtype=np.int8),
    np.full(N_BINS, prior_map[p], dtype=np.int8),
    wheel_labels[row],
    whisker_labels[row],
])
```
```python
"output_values": [..., ["0.2", "0.5", "0.8"], ...]
```

iii. Step 10 Check 7: "initial 0.5-prior trials remain because the requested prior output explicitly includes class 0.5" — consistent with the reference default `exclude_unbiased=False`.

---

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. the released `_ibl_wheel.position` / `_ibl_wheel.timestamps`, from which the loader derives an evenly sampled position and a smoothed velocity. Wheel speed is the absolute value of that velocity — identical to the reference code's `'wheel-speed'` target.

ii.
```python
sess.load_wheel()
...
wheel_speed = np.abs(wheel["velocity"].to_numpy(dtype=float))
wheel_interp, wheel_good = interpolate_trials(
    wheel["times"].to_numpy(), wheel_speed, stim
)
```

iii. Step 1 function table: "`load_target_behavior` … wheel speed is absolute smoothed wheel velocity." Step 5 mapping table: "`SessionLoader.wheel.velocity` → `output[2]`; Absolute velocity; …; Reference code function `load_target_behavior`, `get_behavior_per_interval`."

---

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) Inside `SessionLoader.load_wheel`, the event-driven wheel position is interpolated onto a uniform 1 kHz grid and differentiated with a low-pass Butterworth filter to give velocity — the agent uses the loader defaults, so this is identical to the reference. (2) `interpolate_trials` linearly interpolates `|velocity|` onto the 100 bin-end timestamps of each trial, refusing the trial if the available samples do not reach within one bin of either window boundary or if any interpolated value is non-finite. (3) The resulting per-trial × per-bin matrix, restricted to the finally retained trials, is discretised into three classes. No smoothing, normalisation or z-scoring is added on top.

ii.
```python
def interpolate_trials(times, values, stim_times, *, allow_nan=False):
    targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
    ...
    for i, target in enumerate(targets):
        interval_beg = target[0] - BIN_SIZE
        interval_end = target[-1]
        ib = np.searchsorted(times, interval_beg, side="right")
        ie = np.searchsorted(times, interval_end, side="left")
        # Match get_behavior_per_interval's boundary coverage requirement.
        if ib >= len(times) or ie <= 0:
            good[i] = False; result[i] = np.nan; continue
        lo = max(0, ib); hi = min(len(times), ie + 1)
        local_t, local_v = times[lo:hi], values[lo:hi]
        if (local_t[0] - interval_beg > BIN_SIZE + 1e-9 or
                interval_end - local_t[-1] > BIN_SIZE + 1e-9 or len(local_t) < 2):
            good[i] = False; result[i] = np.nan; continue
        result[i] = np.interp(target, local_t, local_v)
```

iii. Step 4, "Continuous behavior sampling" row: "Follow cache code's interpolation onto stimulus-aligned bin-end timestamps." Step 5 Key Decision 4: "Match reference bin-end timestamps and linear interpolation. Reject intervals whose continuous stream begins/ends more than one bin from requested boundaries. Do not impute NaNs because categorical discretization would conceal missing data."

---

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tertiles. After the final trial set is fixed, the 1/3 and 2/3 quantiles of all finite aligned wheel-speed samples *of that session* are computed, and `np.digitize(..., right=False)` assigns each sample to class 0/1/2 (`['slow','medium','fast']`). Because the thresholds come from the same pool that is being labelled, every session's three classes are almost exactly equal in size (verified fractions 0.333/0.333/0.333). If the two thresholds are not strictly increasing the session is rejected rather than rank-split — this removed one session with an all-zero wheel trace. Thresholds are stored per session in `metadata['session_info']`.

ii.
```python
def discretize_tertiles(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise RuntimeError("no finite behavior values")
    thresholds = np.quantile(finite, [1 / 3, 2 / 3])
    if not thresholds[0] < thresholds[1]:
        raise RuntimeError(f"collapsed tertile thresholds {thresholds.tolist()}")
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, thresholds
```
```python
# Thresholds are calculated only from final retained, finite aligned samples.
wheel_labels, wheel_q = discretize_tertiles(wheel_interp[selected])
```

iii. Step 5 Key Decision 5: "Compute 1/3 and 2/3 quantiles separately per session from all valid aligned values. This mirrors the reference's per-session standardization and avoids camera/lab calibration differences. Use `np.digitize(..., right=False)`; if tied thresholds collapse classes, flag/drop the session rather than rank-splitting identical physical values." The reference papers do not discretise wheel speed at all (they use R²), so this is an explicitly task-imposed transform (Step 10 Check 9).

---

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is evaluated at `stimOn_times + BIN_END_TIMES`, i.e. the right edge of each of the same 100 neural bins, using the same `stimOn_times` for the same trial. The interpolation window is `[stimOn − 0.5, stimOn + 1.5]`, identical to the neural window, and a trial is dropped if the wheel does not span it. So the wheel class at index *i* is the wheel speed at the end of neural bin *i*.

ii.
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
...
result[i] = np.interp(target, local_t, local_v)
```

iii. Step 7 plot review: "interpolated wheel and whisker traces align exactly with their categorical step labels; no discontinuity or one-bin temporal shift is visible." Step 10 Check 5: "independently reconstructed raw choice/prior plus continuous behavior interpolation and stored-threshold classes; exact `np.allclose` pass."

---

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The `whiskerMotionEnergy` column of `SessionLoader.load_motion_energy(views=[view])`, i.e. the released `<view>Camera.ROIMotionEnergy` with its frame times `_ibl_<view>Camera.times`. The left camera (60 Hz) is tried first and the right camera (150 Hz) is used as fallback, matching the reference `bin_behaviors` logic. The released trace is used as-is. The chosen camera is recorded per session as `whisker_camera`. Fourteen sessions had neither stream and were dropped.

ii.
```python
whisker_view = None
for view in ("left", "right"):
    try:
        sess.load_motion_energy(views=[view])
        key = f"{view}Camera"
        frame = sess.motion_energy[key]
        if "whiskerMotionEnergy" in frame and len(frame) > 1:
            whisker_view = view
            motion = frame
            break
    except Exception:
        continue
if whisker_view is None:
    raise RuntimeError("no left or right whisker motion-energy stream")
```

iii. Step 4, "Continuous behavior sampling" row: "Use left ROI motion energy and right fallback, matching code." Step 3: "Whisker motion energy is mean absolute adjacent-frame difference within a whisker-pad ROI anchored between nose and eye; the reference code uses left camera with right-camera fallback and linear resampling."

---

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Exactly the same pipeline as the wheel, minus the velocity step: the released motion-energy trace is linearly interpolated onto the 100 bin-end timestamps of each trial (with the same one-bin boundary-coverage requirement), and the result is discretised into per-session tertiles. No filtering, smoothing, normalisation or camera-rate correction is applied, so the 60 Hz left and 150 Hz right traces are treated identically apart from the per-session thresholds absorbing their different scales.

ii.
```python
whisker_interp, whisker_good = interpolate_trials(
    motion["times"].to_numpy(), motion["whiskerMotionEnergy"].to_numpy(), stim
)
...
whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])
```

iii. Step 5 mapping table: "Left `whiskerMotionEnergy` (right fallback) → `output[3]`; Linear interpolation at bin ends; session-wise tertile discretization of finite aligned samples to labels 0/1/2 … Session-wise thresholds handle camera gain/ROI scale differences; store thresholds."

---

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: the 1/3 and 2/3 `np.quantile` of the session's own finite aligned samples (restricted to the finally retained trials), then `np.digitize(..., right=False)` into `['low','medium','high']`. Verified class fractions are 0.333/0.333/0.333 in every session. Sessions with collapsed thresholds are rejected. Thresholds are saved in `metadata['session_info'][i]['whisker_tertiles']`.

ii.
```python
whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])
...
"whisker_tertiles": whisker_q.tolist(),
...
"discretization": "session-wise tertiles over retained aligned samples",
```

iii. Same as 7-c — Step 5 Key Decision 5. Per-session rather than global thresholds are justified because motion energy is in uncalibrated camera units that differ by rig, camera and ROI size.

---

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera trace is evaluated at `stimOn_times + BIN_END_TIMES`, the same 100 bin-end timestamps used for the neural bins and the wheel, from the same `stimOn_times`. The camera frame times are already on the session clock, and a trial is dropped unless the camera spans the whole window to within one bin at each edge.

ii.
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
...
result[i] = np.interp(target, local_t, local_v)
```
```python
keep = base_mask & wheel_good & whisker_good
```

iii. Step 7: the `--show-processing` plots overlay the interpolated trace and its step-function class labels on the same time axis as the population spike count, and were inspected for a one-bin shift; Step 10 Check 5 re-derived the values independently through ONE and compared with `np.allclose`.

---

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is dropped rather than imputed, at whichever granularity it occurs, and every drop is recorded.
 - *Per spike*: cluster indices outside the cluster table are filtered out (`valid` mask).
 - *Per probe*: a probe with zero clusters is skipped; a session with no pykilosort collection at all raises and is skipped.
 - *Per trial*: NaN in any required trial event, RT/duration out of range, `choice == 0`, unexpected `probabilityLeft`, wheel or camera not spanning the window, non-finite interpolated samples, and an all-zero population spike matrix (interpreted as a period outside a valid recording interval) each remove the trial.
 - *Per session*: no whisker stream on either camera, a collapsed tertile threshold, or fewer than 2 surviving trials raise a `RuntimeError`; `attempt` catches every exception, prints `SKIP <eid>: <reason>`, and appends it to `metadata['failed_sessions']`. 18 of 459 sessions were excluded this way (14 no whisker stream, 2 missing `probabilityLeft`, 1 all-zero stream, 1 no valid trials).
 - Known-harmless ONE warnings (revision/hash chatter, `ALFWarning`) are suppressed, and the trials-table hash check is disabled because the index row describes the superseded pre-revision file.
 - `validate_session` asserts shapes, dtypes, finiteness and the allowed categorical value sets for every trial before it is accepted.

ii.
```python
def attempt(item):
    idx, eid = item
    try:
        ...
        return eid, convert_session(worker_one, eid, do_plot), None
    except Exception as exc:
        return eid, None, f"{type(exc).__name__}: {exc}"
    finally:
        gc.collect()
```
```python
if len(selected) < 2:
    raise RuntimeError(f"only {len(selected)} valid trials")
...
neural_valid = np.asarray([np.any(x) for x in neural], dtype=bool)
if not np.all(neural_valid):
    selected = selected[neural_valid]
    neural = [x for x, valid in zip(neural, neural_valid) if valid]
```
```python
def validate_session(neural, inputs, outputs, regions):
    assert len(neural) == len(inputs) == len(outputs) >= 2
    n = len(regions)
    for x, i, o in zip(neural, inputs, outputs):
        assert x.shape == (n, N_BINS) and x.dtype == np.float32
        ...
        assert set(np.unique(o[0])).issubset({0, 1})
        for j in (1, 2, 3):
            assert set(np.unique(o[j])).issubset({0, 1, 2})
```
```python
"failed_sessions": failures,
```

iii. Step 5 Key Decision 4: "Do not impute NaNs because categorical discretization would conceal missing data." Step 10: "Remaining 18 exclusions (not fixable without fabrication) … Because all four outputs are mandatory, imputing these signals or retaining malformed classes would violate the task." Step 10 Iteration 1 also documents a real bug found and fixed: sharing one mutable `ONE` object across 16 threads caused nondeterministic `NoneType` failures that had silently excluded 94 valid sessions; each worker now builds its own client.

---

## 10-a. What are the most time-consuming steps of the code?

i. The agent instrumented `convert_session` with `time.perf_counter()` and printed per-session seconds. Measured cost was ~6 s/session on the 2-session sample (≈46 min sequential for 459), and 1,235.8 s wall clock for the full run at 16 threads. The dominant component is per-session I/O — reading the two spike arrays for each of 1–2 probes through `SpikeSortingLoader.load_spike_sorting()` (note: called *without* `check_hash=False`, so every spike file is additionally re-read and md5'd) — followed by serialising the 98 GiB pickle. `SessionLoader.load_wheel`'s 1 kHz interpolation and Butterworth filtering is the main non-I/O cost. The agent's own notes attribute cost to wheel smoothing, dense neural matrices, and serialisation, but do not name spike-sorting I/O as the single largest term.

ii.
```python
start = time.perf_counter()
...
elapsed = time.perf_counter() - start
print(f"OK {eid}: {len(selected)}/{len(trials)} trials, {len(regions)} neurons, "
      f"{len(probes)} probes, {elapsed:.1f}s", flush=True)
```
```python
spikes, clusters, channels = loader.load_spike_sorting()   # no check_hash=False
```
```python
"processing_seconds": elapsed,
"conversion_seconds": time.perf_counter() - t0,
```

iii. Step 6: "SessionLoader wheel smoothing is intrinsically expensive and produces dense 1-kHz traces. Dense target-format neural matrices dominate memory and output size." Step 7 run-time table: "Behavior + spikes + binning | 5.95 s/session mean | 45.5 min sequential; approximately 3–8 min with 16 workers"; "Serialization | sample 228 MB within total 12.4 s | full estimated ~52 GB; several additional minutes."

---

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain. (1) `block_trial_numbers` is a scalar Python loop over every trial of every session; it is exactly `np.arange(n) - np.maximum.accumulate(np.where(change, np.arange(n), 0))` and could be one vectorised expression, or `trials.groupby(block).cumcount()` as the human reference does. (2) `interpolate_trials` loops once per trial to slice the stream and call `np.interp`; it could build one flat query vector. (3) `bin_spikes` loops once per trial to `bincount`; it could offset each spike's flat index by its trial and issue a single `bincount` for the whole session. The agent did vectorise *within* each trial (one `searchsorted` + one flattened `bincount` instead of per-neuron work) and parallelised *across* sessions with a 16-thread pool, which is where the real win was; the remaining loops are O(n_trials) and cost well under a second per session.

ii.
```python
for i in range(1, len(p)):
    out[i] = out[i - 1] + 1 if p[i] == p[i - 1] else 0
```
```python
for i, target in enumerate(targets):
    ...
    result[i] = np.interp(target, local_t, local_v)
```
```python
for stim in np.asarray(stim_times, dtype=np.float64):
    ...
    count = np.bincount(flat, minlength=n_clusters * N_BINS)
```

iii. Step 6 "Code speedups added": "spike searches use sorted arrays and `searchsorted`; per-trial counts use a single flattened `bincount`; only retained trials are binned; arrays are float32/int8; probes are stable-sorted once after merging; objects are garbage-collected between sessions." The remaining per-trial loops are not enumerated as candidates in CONVERSION_NOTES.md.

---

## 10-c. What processing does the code repeat multiple times?

i. Several things.
 - **The whole ONE index is rebuilt per worker thread.** `build_one()` reads and concatenates both release `datasets` tables and re-applies the revision patch; with 16 threads that is 16 full rebuilds of a large parquet index instead of one shared read-only copy. This was a deliberate fix for a thread-safety bug, but a read-only shared snapshot would have avoided the repetition.
 - **Spike files are read twice.** `load_spike_sorting()` is called without `check_hash=False`, so `SpikeSortingLoader` re-reads and md5-hashes every spike array in addition to loading it.
 - **Behaviour is interpolated for every trial, then most are thrown away.** `interpolate_trials` runs over all raw trials (~646/session) although only the ~424 passing `base_mask` can ever be used.
 - **Motion energy may be loaded twice** for the 14+ sessions where the left camera is tried and fails before the right is used.
 - Per-session metadata calls (`get_details`, `list_collections`) are issued once per session in addition to the loaders' own lookups.

ii.
```python
if not hasattr(worker_state, "one"):
    worker_state.one, _ = build_one(verbose=False)
worker_one = worker_state.one
```
```python
spikes, clusters, channels = loader.load_spike_sorting()
```
```python
wheel_interp, wheel_good = interpolate_trials(wheel["times"].to_numpy(), wheel_speed, stim)
whisker_interp, whisker_good = interpolate_trials(
    motion["times"].to_numpy(), motion["whiskerMotionEnergy"].to_numpy(), stim
)          # `stim` is every trial, not stim[base_mask]
```

iii. The per-thread client is justified in the code comment: "ONE contains mutable loader/cache state and is not thread-safe. Keep a distinct composite ONE instance in each full-conversion worker," and in Step 10 Iteration 1, where the shared object caused 94 spurious session failures. The other repetitions are not discussed in CONVERSION_NOTES.md.

---

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
 - **Interpolating rejected trials** (see 10-c) — roughly a third of the interpolation work is discarded by `base_mask`.
 - **`good_units` / `n_good_units`** — `label >= 1` is computed for every cluster of every probe purely to be counted into metadata; it never affects the data.
 - **`retained_trial_indices`** — the full list of source trial indices is stored per session; useful for auditing, unused downstream.
 - **`plot_processing` receives `wheel_raw` and `whisker_raw`** (the full session dataframes) and never uses them.
 - **`details['lab']` / `details['date']`** are fetched and stored but unused by the decoder.
 - Most consequentially, the decision to keep every Kilosort cluster (2-c) means ~88% of the 594,965 stored units are not well-isolated neurons and ~13% are `root`/`void`; the resulting 98 GiB pickle carries a large amount of data that the reference analyses would have curated away. This is a consequence of the QC decision rather than dead code, but it dominates the wasted I/O and storage.

ii.
```python
label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
all_good.append(label >= 1)
...
"n_good_units": int(np.sum(good_units)),
"retained_trial_indices": selected.astype(int).tolist(),
```
```python
def plot_processing(eid, pop_counts, wheel_raw, whisker_raw, wheel_interp,
                    whisker_interp, wheel_labels, whisker_labels, thresholds):
    # wheel_raw / whisker_raw are never referenced in the body
```
```python
wheel_interp, wheel_good = interpolate_trials(wheel["times"].to_numpy(), wheel_speed, stim)
```

iii. Step 4: "Retain all clusters, matching the decoder paper and exact supplied cache path. Record quality in metadata statistics; do not silently substitute the stricter anatomical-analysis population." The auditing fields are motivated by Step 10's requirement to be able to re-derive any converted value from the raw data ("converted trial 5/raw trial 15"). The redundant interpolation and unused plot arguments are not discussed.
