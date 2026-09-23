# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the ONE API against the staged local cache in `/app/data/one_cache`; no ALF file is opened directly. The list of sessions is taken from the methods repository's release freeze, `code/code_zhang2025/data/bwm_release.csv` (459 sessions, 699 probe insertions), which is the same index `0_data_caching.py` uses. The CSV is grouped by `eid`, so one job = one session and the `pid`/`probe_name` of every insertion of that session travel with the job. An optional `data/DATALIMIT_SUBSET.csv` restricts the eids when the environment ships a subset. Per session, `SessionLoader` provides the trials table, the wheel and the camera motion energy, and `SpikeSortingLoader` is called once per probe for spikes/clusters/channels. Sessions are processed in a `ProcessPoolExecutor` (8 workers), each writing a restartable per-session pickle shard that the parent re-reads once at the end to build the final dictionary. Each worker builds its own read-only ONE client (`One.load_cache(..., clobber=True)` rather than `OneAlyx.load_cache`) so that concurrent workers do not rewrite the cache tables.

ii.
```python
def _one(cache_dir: str):
    """Build a ONE client against the staged, locally cached release."""
    from one.api import ONE, One
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", silent=True, cache_dir=cache_dir)
    One.load_cache(one, tables_dir=Path(cache_dir) / "Brainwidemap", clobber=True)
    return one
```
```python
def _jobs(release_file: Path, cache_dir: Path, shard_dir: Path) -> list[dict]:
    release = pd.read_csv(release_file, index_col=0)
    subset = _read_subset(ROOT / "data" / "DATALIMIT_SUBSET.csv")
    if subset is not None:
        release = release[release["eid"].astype(str).isin(subset)]
    jobs = []
    for eid, rows in release.groupby("eid", sort=False):
        first = rows.iloc[0]
        jobs.append({"eid": str(eid), "subject": str(first["subject"]), ...,
                     "pids": rows["pid"].astype(str).tolist(),
                     "probe_names": rows["probe_name"].astype(str).tolist(), ...})
    return jobs
```
```python
loader = SessionLoader(one=one, eid=eid)
loader.load_trials(); loader.load_wheel()
spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
spikes, clusters, channels = spike_loader.load_spike_sorting()
```

iii. The module docstring states the intent: "The preprocessing follows `code/code_zhang2025/src/0_data_caching.py` and its helpers". The agent first inspected the cache and found "the full 459-session BWM release (699 probes, about 570 GB of source data), not a toy subset", and decided to "build the full release rather than sampling sessions" because the validator "is explicitly engineered for a converted dataset in the hundreds of gigabytes". The shard-per-session design is justified in the docstring as restartability: "Each worker therefore writes one restartable session shard. The parent process only loads those shards once." The read-only `One.load_cache` is justified in a comment after the agent hit a real failure: "The first parallel pass exposed a cache-library race: each worker tried to refresh the same copied Parquet index... the base method is read-only and therefore safe in concurrent workers."

## 1-b. How are the data split into subjects (mice)?

i. No splitting is performed: the `subject` column of `bwm_release.csv` is carried on each session job and stored in the shard. At assembly the unique subject names are sorted into `subjects` and `subject_idx` records each session's index into that list. Result: 136 subjects over 444 sessions.

ii.
```python
jobs.append({"eid": str(eid), "subject": str(first["subject"]), "date": ..., "lab": ..., ...})
```
```python
subjects = sorted({session["subject"] for session in sessions})
subject_map = {name: i for i, name in enumerate(subjects)}
...
"subject_idx": np.array([subject_map[session["subject"]] for session in sessions], dtype=np.int64),
```

iii. Not discussed explicitly; implicitly the release table already carries a unique subject id per session, so nothing has to be derived or parsed.

## 1-c. How are the data split into sessions?

i. A session is the natural unit of the release. The release CSV has one row per probe insertion, so it is grouped by `eid`; every insertion of a session is processed together and its units are pooled into one population (probe merging). One shard file, and therefore one entry of `neural`/`input`/`output`, is produced per session.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    first = rows.iloc[0]
    jobs.append({"eid": str(eid), ..., "pids": rows["pid"].astype(str).tolist(),
                 "probe_names": rows["probe_name"].astype(str).tolist(), ...})
```
```python
offset = 0
for pid, probe_name in zip(job["pids"], job["probe_names"]):
    ...
    spike_parts.append((np.asarray(spikes["times"]), local_ids, offset))
    region_parts.append(mapped)
    offset += n_clusters
```

iii. The docstring says "probes from one recording session are combined", which follows `prepare_data`/`merge_probes` in the methods repository ("data from the probes recorded in the same session are not statistically independent as they have the same underlying behaviour").

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` already has one row per trial, so the split is given by the data. Trials are indexed by row; `candidate` holds the surviving row indices and `stimOn_times` of those rows defines the trial windows.

ii.
```python
loader.load_trials()
trials = loader.trials.copy()
...
candidate = np.flatnonzero(mask)
...
onsets = trials["stimOn_times"].to_numpy(dtype=float)[candidate]
```

iii. No justification given or needed — the ALF trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The agent reimplements `load_trials_and_mask(..., max_trial_len=10.0)` from `code/code_zhang2025/src/utils/ibl_data_utils.py`, i.e. the exact mask `prepare_data` builds, and then adds a behavioural-coverage test. Five criteria combined: (1) all of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType, goCue_times` finite (the repo's default `nan_exclude` list plus `goCue_times`, which the trial-length test needs); (2) reaction time (`firstMovement_times - stimOn_times`) in [0.08, 2.0] s; (3) trial length (`feedback_times - goCue_times`) ≤ 10 s; (4) `choice != 0`, i.e. no-response trials dropped; (5) `probabilityLeft ∈ {0.2, 0.5, 0.8}`. Afterwards, each surviving trial must also have a complete wheel and whisker trace covering [-0.5, 1.5] s around onset (the `get_behavior_per_interval` coverage test, see 9). A session with fewer than two surviving trials raises and is skipped. Result: 188,925 trials over 444 sessions.

ii.
```python
required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
            "firstMovement_times", "feedbackType", "goCue_times"]
...
finite = np.all(np.isfinite(trials[required].to_numpy(dtype=float)), axis=1)
reaction_time = trials["firstMovement_times"].to_numpy(float) - trials["stimOn_times"].to_numpy(float)
trial_length = trials["feedback_times"].to_numpy(float) - trials["goCue_times"].to_numpy(float)
choice_raw = trials["choice"].to_numpy(dtype=float)
mask = (finite
        & (reaction_time >= 0.08) & (reaction_time <= 2.0)
        & (trial_length <= 10.0)
        & (choice_raw != 0)
        & np.isin(probability, [0.2, 0.5, 0.8]))
candidate = np.flatnonzero(mask)
if len(candidate) < 2:
    raise ValueError("fewer than two trials pass the BWM trial criteria")
```
```python
behavior_good = wheel_good & motion_good
candidate = candidate[behavior_good]
```

iii. The docstring says "the BWM trial exclusions are applied", and the metadata field spells the rule out: `"finite required BWM events, choice!=0, first movement latency 0.08-2.00 s, go-cue-to-feedback <=10 s, and complete behavior streams"`. The agent explicitly validated parity against the repository's own function on one session and reported: "A one-session parity check now matches the repository's trial mask exactly: 407 of 565 trials, with the same choice and prior counts."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, one pair per probe. The merged cluster table (`merge_clusters(...).to_df()`) is used only for its row count (to shift probe-local cluster ids) and for its `acronym` column, which becomes `brain_region_idx` after Beryl mapping. No cluster metric (`label`, `amp`, `depths`) is used.

ii.
```python
spikes, clusters, channels = spike_loader.load_spike_sorting()
merged = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
n_clusters = len(merged)
local_ids = np.asarray(spikes["clusters"], dtype=np.int64)
mapped = brain_regions.acronym2acronym(merged["acronym"].to_numpy(), mapping="Beryl").astype(str)
spike_parts.append((np.asarray(spikes["times"]), local_ids, offset))
```

iii. Implicit: these are the two arrays `prepare_data` puts into `neural_dict` (`'spike_times'`, `'spike_clusters'`, `'cluster_regions'`), which `bin_spiking_data` then bins.

## 2-b. How is the `neural` data processed?

i. Spikes of all probes of a session are pooled (probe-local cluster ids shifted by a running offset) and counted into 100 non-overlapping 20 ms bins spanning [-0.5, 1.5] s around `stimOn_times`. Counting is done with one flat index per spike and a single `np.bincount` per probe, giving a `(n_trials, n_neurons, 100)` float32 array that is then split into a list of per-trial `(n_neurons, 100)` matrices. No smoothing, no z-scoring, and **no conversion to firing rate** — the stored values are raw spike counts per 20 ms bin.

ii.
```python
for trial, (begin, end) in enumerate(zip(starts, ends)):
    lo = np.searchsorted(times, begin, side="left")
    hi = np.searchsorted(times, end, side="left")
    if hi <= lo:
        continue
    bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
    clu = clusters[lo:hi] + offset
    valid = (bins >= 0) & (bins < N_BINS) & (clu >= 0) & (clu < n_neurons)
    if np.any(valid):
        flat_parts.append(((trial * n_neurons + clu[valid]) * N_BINS + bins[valid]))
counts = np.bincount(np.concatenate(flat_parts), minlength=size)
return counts.reshape(len(onsets), n_neurons, N_BINS).astype(np.float32)
```
```python
neural = [neural_array[i] for i in range(len(candidate))]
```

iii. Docstring: "spikes are counted in 20 ms bins from 0.5 s before through 1.5 s after stimulus onset"; metadata records `'neural_measure': "Kilosort spike counts in non-overlapping 20 ms bins"`. This follows the method paper's description quoted in `methods.txt` ("Recordings are split into 2-s trials, each divided into 20-ms bins... by aggregating spike counts") and `merge_probes` for pooling probes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control is applied at all.** Every Kilosort cluster of every probe is retained, including clusters the IBL pipeline labels as poor and including clusters whose Beryl acronym is `root` or `void` (i.e. sites the histology placed outside a summary structure or outside the brain). The only cluster-level check is a sanity assertion that probe-local cluster ids are contiguous. This yields a mean of **1351 neurons per session** (min 135, max 3140) and 599,865 units total, versus the 75,708 "well-isolated" units reported by the data paper, and produces a 99 GB pickle.

ii.
```python
# Load and retain every Kilosort cluster, exactly as prepare_data(qc=None)
# in the methods repository.  Probe-local IDs are shifted on merge.
...
if len(local_ids) and (local_ids.min() < 0 or local_ids.max() >= n_clusters):
    raise ValueError(f"non-contiguous cluster IDs for {probe_name}")
mapped = brain_regions.acronym2acronym(merged["acronym"].to_numpy(), mapping="Beryl").astype(str)
```
```python
"neuron_filter": "all sorted clusters (qc=None), matching the methods code",
```

iii. The sole justification is fidelity to the methods repository: `prepare_data` calls `load_spiking_data(one, pid, ...)` without a `qc` argument, and that function documents "If use all available clusters, set qc to None. If use good clusters, set qc to 1." The agent's comment therefore reads "exactly as prepare_data(qc=None) in the methods repository". The trajectory contains no discussion of the data paper's cluster curation, of `label >= 1`, or of `root`/`void` units (which `data_loader_utils.py:252` of the same repository does exclude at decode time).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `stimOn_times`. All IBL streams are already on one synchronised session clock, so alignment is a subtraction: for each trial the window is `[stimOn - 0.5, stimOn + 1.5]`, the spikes inside it are found with `searchsorted`, and the bin index of each spike is `floor((t - (stimOn - 0.5)) / 0.02)`. Bin 0 therefore starts exactly 0.5 s before onset and bin 25 starts at onset. Spikes falling outside 0..99 after rounding are discarded by the `valid` mask (the human reference clips them into the edge bins instead). Windows of successive trials are handled independently, so a spike in an overlap would be counted in both trials.

ii.
```python
starts = onsets + OFF_START      # OFF_START = -0.5
ends = onsets + OFF_END          # OFF_END   = +1.5
...
bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
valid = (bins >= 0) & (bins < N_BINS) & (clu >= 0) & (clu < n_neurons)
```
```python
"temporal_alignment_event": "visual stimulus onset (stimOn_times)",
"off_start": OFF_START, "off_end": OFF_END,
```

iii. The window and the alignment event are taken from the repository's decoding parameters (`'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)`) and from the task instruction "Temporally align based on stimulus onset". The agent noted "The key adaptation is one common stimulus-aligned 2-second window at 20 ms resolution so all four requested outputs coexist on the same trials." The comment in `_bin_spikes` explains the per-trial loop: "Work trial-by-trial so overlapping windows, if any, count a spike in both trials as in the reference implementation."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, identical for every trial and session; `metadata['time_bin_size'] = 20.0` ms. Spikes are binned once at that resolution — there is no rebinning, resampling or smoothing of the neural data. The behavioural streams are interpolated onto the same 100-point grid (at bin right edges), and the per-trial scalars are broadcast over the 100 bins.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```
```python
"time_bin_size": BIN_SIZE * 1000.0,
```

iii. Docstring and comments tie this to the repository config `{'interval_len': 2, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` and to the method paper text in `methods.txt` ("each divided into 20-ms bins, producing T = 100 time steps").

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable beyond the alignment event itself: it is the fixed grid of times relative to `stimOn_times`, `linspace(-0.48, 1.5, 100)`, i.e. the **right edge** of each 20 ms bin. The same vector is written into row 0 of every trial's input.

ii.
```python
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```
```python
inp = np.empty((2, N_BINS), dtype=np.float32)
inp[0] = RELATIVE_TIMES
```

iii. Comment above the constant: "This is exactly the grid used by the reference `get_behavior_per_interval`: behavior values correspond to the right edge of each neural count bin." That matches the repository line `x_interp = np.linspace(interval_begs[i] + binsize, interval_ends[i], n_bins)`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid once at module level and broadcasting it into each trial's `(2, 100)` float32 input array. The value is kept continuous (a ramp from -0.48 s to 1.5 s), not binarised.

ii.
```python
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
...
for row, trial_index in enumerate(candidate):
    inp = np.empty((2, N_BINS), dtype=np.float32)
    inp[0] = RELATIVE_TIMES
    inp[1] = trial_in_block[trial_index]
    input_data.append(inp)
```

iii. The instruction asks for "Time since stimulus onset, continuous, time-varying"; the agent's decision is to reuse the repository's behaviour time grid so that inputs, outputs and neural bins are indexed identically.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Bin for bin. `RELATIVE_TIMES[k]` is the right edge of neural bin `k` (which spans `[-0.5 + 0.02k, -0.5 + 0.02(k+1)]` relative to onset), and the same index `k` is used for the neural counts, the two interpolated behaviour traces and the time input. There is therefore a constant +10 ms offset between the stored time value and the centre of the corresponding neural bin (the human reference uses bin centres instead).

ii.
```python
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS)   # bin right edges
...
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)          # behavior on the same grid
...
bins = np.floor((times[lo:hi] - begin) / BIN_SIZE)   # neural on the same 100-bin partition
```

iii. Same justification as 3-a: the right-edge grid is what `get_behavior_per_interval` uses, so all time-varying streams in the converted file share one convention.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table only: the block prior is constant within a block, so a change of value marks a block boundary. There is no explicit block id in the ALF table.

ii.
```python
probability = trials["probabilityLeft"].to_numpy(dtype=float)
trial_in_block = _trial_number_in_block(probability)
```

iii. Not stated explicitly; the constant-prior-per-block structure is described in `methods.txt` ("in subsequent trials, it appears predominantly on one side in blocks").

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A one-based counter that restarts at 1 whenever `probabilityLeft` changes (and whenever it is non-finite). It is computed on the **full** trials table before any trial exclusion, so a trial dropped by the quality mask still advances the counter and the stored number is the animal's real position in the block. The scalar is then broadcast across the 100 time bins of the trial's input row 1. Observed range across the dataset: 1 to 99.

ii.
```python
def _trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    """Return the task's one-based trial number within each probability block."""
    out = np.empty(len(probability_left), dtype=np.float32)
    number = 0
    previous = None
    for i, value in enumerate(probability_left):
        if i == 0 or not np.isfinite(value) or value != previous:
            number = 1
        else:
            number += 1
        out[i] = number
        previous = value
    return out
```
```python
inp[1] = trial_in_block[trial_index]      # trial_index indexes the unfiltered table
```

iii. Metadata states the rule: `'trial_number_in_block': "one-based count, computed before trial exclusions"`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table (+1 / -1 / 0). Trials with `choice == 0` (no response) are already removed by the trial mask. The remaining values are mapped to a binary label with `choice == 1 → 1` and `choice == -1 → 0`, and `output_values[0]` is declared as `["left", "right"]`. The inline comment states the assumed convention as "left=-1, right=1". This is the opposite of the IBL convention: in the IBL trials table `choice == +1` is a **leftward** choice and `choice == -1` a **rightward** one (`ibllib/brainbox/behavior/training.py`: `rightward = trials.choice == -1`, "choice == -1 means contrast on right hand side"; verified directly on session `6713a4a7…`, where all correct stimulus-on-left trials have `choice == +1`). The stored labels are therefore swapped relative to the required `left = 0, right = 1` encoding — visible in the summary statistics, where the AI's choice fractions (0: 0.492, 1: 0.508) are the mirror image of the human reference's (0: 0.507, 1: 0.493).

ii.
```python
choice_raw = trials["choice"].to_numpy(dtype=float)
...
mask = (... & (choice_raw != 0) & ...)
...
choice = (choice_raw[candidate] == 1).astype(np.int8)  # left=-1, right=1
```
```python
"output_values": [["left", "right"], ...]
```

iii. The only justification is the inline comment asserting the sign convention; the trajectory contains no check of the convention against `contrastLeft`/`contrastRight`/`feedbackType` or against `ibllib`. The agent did verify that its trial mask reproduced the repository's *counts* per choice value (233 `+1` / 174 `-1`), but never verified which side `+1` means.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Nothing beyond the binary recode above, followed by broadcasting the per-trial scalar over all 100 time bins as row 0 of the `(4, 100)` int8 output array.

ii.
```python
choice = (choice_raw[candidate] == 1).astype(np.int8)  # left=-1, right=1
...
out = np.empty((4, N_BINS), dtype=np.int8)
out[0] = choice[row]
```

iii. The instructions ask for a binary per-trial choice; the instructions also say to make outputs time-varying if at all possible, hence the broadcast to 100 bins.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, restricted by the trial mask to the three task values {0.2, 0.5, 0.8} and mapped to {0, 1, 2} with an explicit lookup dictionary.

ii.
```python
probability = trials["probabilityLeft"].to_numpy(dtype=float)
mask = (... & np.isin(probability, [0.2, 0.5, 0.8]))
...
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_lookup[float(x)] for x in probability[candidate]], dtype=np.int8)
```
```python
"output_values": [..., ["0.2", "0.5", "0.8"], ...]
```

iii. The mapping 0.2 → 0, 0.5 → 1, 0.8 → 2 is given verbatim in the task instructions; restricting to those three values also guarantees the lookup cannot raise.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Only the three-way recode, then broadcasting the per-trial value across the 100 bins as output row 1. The unbiased 0.5 block at the start of each session is kept as its own class. Resulting class fractions (0.417 / 0.141 / 0.442) are essentially identical to the human reference (0.418 / 0.141 / 0.442).

ii.
```python
out[1] = prior[row]
```

iii. As above — the mapping is prescribed by the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. `_ibl_wheel.position` / `_ibl_wheel.timestamps` interpolated to a uniform 1 kHz grid and differentiated with the loader's default Butterworth low-pass. Speed is the absolute value of the returned `velocity` column, paired with the loader's `times`.

ii.
```python
loader.load_wheel()
wheel_times = loader.wheel["times"].to_numpy()
wheel_speed = np.abs(loader.wheel["velocity"].to_numpy())
```

iii. This is exactly what `load_target_behavior(one, eid, 'wheel-speed')` does in `ibl_data_utils.py` (`np.abs(sess_loader.wheel['velocity'].to_numpy())`), which the docstring says the conversion follows.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps. (1) The `SessionLoader` interpolation/filtering above. (2) Per trial, the samples strictly inside `[stimOn-0.5, stimOn+1.5]` are selected with `searchsorted(right)/searchsorted(left)` and the repository's coverage tests are applied (at least 2 samples, and the first/last sample within one bin of the window edges), otherwise the trial is dropped. (3) The retained segment is linearly interpolated with `np.interp` onto the 100 right-edge times, giving a `(n_trials, 100)` float32 matrix; a non-finite result also drops the trial. (4) The whole matrix is discretised into session-wise tertiles (see 7-c). Note `np.interp` clamps outside the segment where the repository's `interp1d(..., fill_value='extrapolate')` extrapolates; duplicate timestamps are de-duplicated (last sample kept) so the grid is strictly increasing.

ii.
```python
ib = np.searchsorted(times, begin, side="right")
ie = np.searchsorted(times, end, side="left")
if ib >= ie or ib >= len(times) or ie <= 0:
    good[i] = False; continue
segment_t, segment_v = times[ib:ie], values[ib:ie]
# Same tests as get_behavior_per_interval in the supplied repository.
if (len(segment_t) < 2 or abs(begin - segment_t[0]) > BIN_SIZE or abs(end - segment_t[-1]) > BIN_SIZE):
    good[i] = False; continue
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
```
```python
wheel, wheel_good = _interpolate_trials(wheel_times, wheel_speed, candidate_onsets)
```

iii. The function docstring says "Interpolate one continuous stream using the reference coverage checks", and the inline comment names the source: "Same tests as `get_behavior_per_interval` in the supplied repository." Metadata records `'behavior_sampling': "linear interpolation at neural-bin right edges"`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three equally-populated classes by **within-session tertiles**: the 1/3 and 2/3 quantiles are computed over all retained trial × time samples of that session, and `np.digitize` assigns 0 / 1 / 2. The two cut values are recorded per session in `metadata['session_info']`. Achieved fractions are 0.3333 / 0.3333 / 0.3334.

ii.
```python
def _session_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Discretize a stream into low/middle/high within-session tertiles."""
    cuts = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.digitize(values, cuts, right=False).astype(np.int8)
    return labels, [float(cuts[0]), float(cuts[1])]
```
```python
wheel_cat, wheel_cuts = _session_tertiles(wheel)
...
"continuous_output_discretization": "within-session tertiles computed over all retained trial-time samples",
"wheel_speed_tertile_edges": wheel_cuts,
```

iii. The docstring states "The two continuous decoding targets are converted to session-wise tertiles as required by this task" — i.e. the instruction "Wheel speed discretized into 3 bins" forces a discretisation that the reference code does not perform, and per-session tertiles keep the three classes balanced despite very different wheel-speed scales across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at `stimOn + RELATIVE_TIMES`, the same 100 right-edge times used for the neural bin partition and the time input, on the same session clock as the spikes. Row 2 of the output array therefore corresponds bin for bin to the neural columns.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
...
out[2] = wheel_cat[row]
```

iii. No extra alignment is needed because IBL synchronises wheel and ephys onto one clock upstream; the agent's choice is simply to evaluate on the repository's interval grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its camera timestamps, loaded through `SessionLoader.load_motion_energy(views=[view])` and read from the `whiskerMotionEnergy` column. The left camera is tried first and the right camera is used as a fallback; if neither yields a usable stream the whole session is dropped. The chosen camera is recorded per session as `motion_energy_source`.

ii.
```python
def _load_motion_energy(session_loader):
    """Load left whisker motion energy, falling back to the right camera."""
    errors = []
    for view in ("left", "right"):
        try:
            session_loader.load_motion_energy(views=[view])
            key = f"{view}Camera"
            frame = session_loader.motion_energy[key]
            times = frame["times"].to_numpy()
            values = frame["whiskerMotionEnergy"].to_numpy()
            if len(times) >= 2 and len(times) == len(values):
                return times, values, key
        except Exception as exc:  # fallback is part of the reference pipeline
            errors.append(f"{view}: {exc}")
    raise RuntimeError("no usable whisker motion energy (" + "; ".join(errors) + ")")
```

iii. The comment "fallback is part of the reference pipeline" points at `bin_behaviors`, which does exactly `load_target_behavior(one, eid, 'left-whisker-motion-energy')` and falls back to the right camera when the left is missing. The agent reported that this fallback is what prunes the release: "The exclusions seen so far are almost entirely sessions with no usable whisker-motion stream, which is the expected reason the methods dataset contains fewer sessions than the 459-session release" (15 sessions excluded, 444 kept).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is (no filtering, no normalisation) and passed through the *same* `_interpolate_trials` helper as the wheel: window selection, the repository's coverage tests, linear interpolation onto the 100 right-edge times, then session-wise tertiles. A trial is kept only if **both** the wheel and the whisker trace pass their coverage tests.

ii.
```python
motion_times, motion_values, motion_source = _load_motion_energy(loader)
motion, motion_good = _interpolate_trials(motion_times, motion_values, candidate_onsets)
behavior_good = wheel_good & motion_good
candidate = candidate[behavior_good]
wheel, motion = wheel[behavior_good], motion[behavior_good]
```

iii. Same as 7-b: fidelity to `get_behavior_per_interval`, with the added discretisation required by the task instructions.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: within-session tertiles over all retained trial × time samples, `np.digitize` into 0 / 1 / 2, cut values stored in `session_info`. Achieved fractions 0.3326 / 0.3326 / 0.3348 (slightly off exact thirds because motion energy has ties at its lower bound), which matches the human reference's 0.3326 / 0.3326 / 0.3349 almost exactly.

ii.
```python
motion_cat, motion_cuts = _session_tertiles(motion)
...
"whisker_motion_energy_tertile_edges": motion_cuts,
...
out[3] = motion_cat[row]
```

iii. As in 7-c: the instruction "Whisker motion energy discretized into 3 bins" forces a discretisation; per-session tertiles keep classes balanced across sessions with different camera gains.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: interpolated at `stimOn + RELATIVE_TIMES`, the right edges of the neural bins, on the shared session clock. Camera frame rate (60/150 Hz) is lower than the 50 Hz bin rate only for the right camera, so most bins are interpolated between neighbouring frames rather than sampled.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
```

iii. No extra alignment needed — IBL camera times are already synchronised to the ephys clock; the agent only re-uses the repository's interval grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered dropping, never imputation. (a) Trials with NaN in any required event are removed by the mask. (b) Trials whose wheel or whisker window is not covered — or whose interpolation returns a non-finite value — are removed (`good[i] = False`). (c) A session with fewer than two trials after either stage raises and is skipped. (d) Any other exception in a session (missing wheel, no usable camera, no units, non-contiguous cluster ids, load failure on **any** probe) aborts that whole session; the error and traceback are recorded in `failures.json` and the session is simply absent from the output. (e) At assembly, shards that do not exist or have <2 trials are skipped. (f) Source timestamps that are NaN or non-monotonic are filtered/de-duplicated before interpolation. (g) ONE's ubiquitous revision warnings are silenced in workers. 15 of 459 sessions were dropped this way; the validator reported 3 all-zero-neural trials that were kept.

ii.
```python
good_source = np.isfinite(times)
times, values = times[good_source], values[good_source]
if np.any(np.diff(times) <= 0):
    _, reverse_idx = np.unique(times[::-1], return_index=True)
    keep = np.sort(len(times) - 1 - reverse_idx)
    times, values = times[keep], values[keep]
```
```python
if len(candidate) < 2:
    raise ValueError("fewer than two trials have complete wheel and whisker streams")
if offset == 0:
    raise ValueError("session has no neural units")
```
```python
except Exception as exc:
    return {"eid": eid, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), ...}
...
failure_file.write_text(json.dumps(failures, indent=2))
```
```python
for job in jobs:
    shard = Path(job["shard"])
    if not shard.exists():
        continue
    ...
    if len(payload["neural"]) >= 2:
        sessions.append(payload)
```

iii. The format requirement "There needs to be at least two trials within each session" motivates the <2-trial rule. Dropping rather than imputing follows the repository, which skips intervals failing its coverage tests and wraps each session in `try/except` ("Skipped session {eid} due to unexpected error"). The agent chose to record failures to `failures.json` "in the same spirit as the reference caching script".

## 10-a. What are the most time-consuming steps of the code?

i. Three dominate. (1) Reading spike sorting from disk: `load_spike_sorting()` per probe pulls `spikes.times`/`spikes.clusters` (hundreds of MB per probe, 699 probes, ~570 GB of source data) — this is pure I/O and is why the run uses 8 processes. (2) The spike-binning loop `_bin_spikes`, which for each probe loops over every trial and builds a flat index array; with all clusters retained the output array is `n_trials × ~1351 × 100` float32 per session. (3) Serialising and re-reading the data: each session shard is pickled (99 GB total), then every shard is unpickled again in `_assemble` and the concatenation is pickled a second time into the 99 GB `converted_data.pkl`. The whole conversion took ~25 minutes wall-clock on 8 workers plus several minutes to write the final pickle.

ii.
```python
spikes, clusters, channels = spike_loader.load_spike_sorting()
```
```python
neural_array = _bin_spikes(spike_parts, onsets, offset)
```
```python
with temporary.open("wb") as stream:
    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)   # per session shard
...
with shard.open("rb") as stream:
    payload = pickle.load(stream)                                     # read back
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)           # 99 GB final file
```

iii. The agent's own framing: the shards exist because "The source release is large... Each worker therefore writes one restartable session shard. The parent process only loads those shards once, when producing the required final pickle." The agent monitored memory and file growth throughout the run for this reason.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four. (1) `_bin_spikes` loops over trials inside the probe loop, doing one `searchsorted` pair and one index build per trial; since the flat index already encodes the trial, all trials could be handled in a single vectorised pass (as it is, a `np.concatenate` + one `bincount` is done at the end anyway). (2) `_interpolate_trials` loops over trials calling `np.interp` once per trial; one concatenated query vector would do. (3) `_trial_number_in_block` is a pure-Python loop over every trial of the session; this is a two-line vectorised operation (`(p != p.shift()).cumsum()` then `groupby.cumcount`), as in the human reference. (4) The per-trial assembly loop allocates a fresh `(2, 100)` input and `(4, 100)` output array per trial and the prior lookup is a Python list comprehension; these could be built as whole-session arrays and sliced. None of these is the bottleneck (I/O is), so the practical gain is small.

ii.
```python
for i, value in enumerate(probability_left):          # (3) scalar Python loop
    if i == 0 or not np.isfinite(value) or value != previous:
        number = 1
```
```python
for i, onset in enumerate(onsets):                    # (2) one np.interp per trial
    ...
    row = np.interp(grid, segment_t, segment_v)
```
```python
for trial, (begin, end) in enumerate(zip(starts, ends)):   # (1) one searchsorted pair per trial
    lo = np.searchsorted(times, begin, side="left")
```
```python
for row, trial_index in enumerate(candidate):         # (4) per-trial array allocation
    inp = np.empty((2, N_BINS), dtype=np.float32)
```

iii. Not discussed in the trajectory; the comment in `_bin_spikes` justifies the per-trial loop on semantics rather than speed ("Work trial-by-trial so overlapping windows, if any, count a spike in both trials as in the reference implementation").

## 10-c. What processing does the code repeat multiple times?

i. (1) A ONE client is constructed and the release Parquet tables are re-read (`One.load_cache(..., clobber=True)`) **once per session**, i.e. 459 times, rather than once per worker process — `ProcessPoolExecutor` has no `initializer` here (the human reference uses `initializer=worker_setup`). (2) `BrainRegions()` is instantiated once per session, re-reading the atlas tables. (3) `SessionLoader.load_motion_energy` is attempted twice for every session that lacks a left camera. (4) `acronym2acronym(..., mapping='Beryl')` is applied to the full cluster table of every probe. (5) The identical `RELATIVE_TIMES` vector is written into every one of the 188,925 trials' input arrays, and the same per-trial scalar is written 100 times per output row. (6) Every session's data is pickled, unpickled and pickled again (the shard round-trip of 10-a). Each of these is cheap next to spike I/O except (1) and (6).

ii.
```python
one = _one(job["cache_dir"])          # inside _process_session -> once per session
...
brain_regions = BrainRegions()        # also once per session
```
```python
with ProcessPoolExecutor(max_workers=args.workers) as executor:
    futures = {executor.submit(_process_session, job): job for job in jobs}
```
```python
for row, trial_index in enumerate(candidate):
    inp[0] = RELATIVE_TIMES           # same 100 floats stored 188,925 times
```

iii. The repeated cache load is deliberate, but for safety rather than speed — the comment explains it was the fix for a corrupted index: "OneAlyx.load_cache also checks/downloads remote tables and rewrites these files; the base method is read-only and therefore safe in concurrent workers." The shard round-trip is deliberate restartability, per the module docstring.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) By far the largest: binning and storing **all ~1351 clusters per session** instead of the ~164 that pass the IBL quality label. Most of these are noise/MUA clusters, and the units mapped to `root`/`void` are explicitly filtered out by the methods repository itself at decode time (`data_loader_utils.py`: `[roi for roi in np.unique(...) if roi not in ['root', 'void']]`). This is ~8× more data than any curated analysis uses and is what makes the pickle 99 GB rather than ~13 GB. (2) The 99 GB shard write + read + rewrite cycle, pure I/O that is deleted at the end. (3) Trial columns loaded and checked but otherwise unused (`feedbackType`, `feedback_times`, `goCue_times` serve only the mask). (4) Broadcasting constant per-trial values (choice, prior, trial-in-block) over 100 identical bins, and storing a constant time vector per trial — required by the chosen time-varying layout, but ~4× redundant storage on the input/output side. (5) The cluster-id contiguity check, the tertile edges and rich `session_info` metadata, which nothing downstream reads. (6) `metadata['session_info']` grows with one dict per session including per-session tertile edges — harmless but unused.

ii.
```python
# Load and retain every Kilosort cluster, exactly as prepare_data(qc=None)
# in the methods repository.
```
```python
shard.parent.mkdir(parents=True, exist_ok=True)
temporary = shard.with_suffix(".tmp")
with temporary.open("wb") as stream:
    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
...
if not args.keep_shards:
    shutil.rmtree(args.shard_dir)
```
```python
out = np.empty((4, N_BINS), dtype=np.int8)
out[0] = choice[row]     # one value repeated 100 times
out[1] = prior[row]
```

iii. The agent justified keeping all clusters solely as fidelity to `prepare_data(qc=None)` and never weighed the cost; it did control dtype cost (float32 neural, int8 outputs) and reported the resulting size plainly ("yielding a 99 GB final pickle"), and deleted the shards at the end ("Temporary 99 GB session shards were deleted; they can be regenerated by rerunning the converter").
