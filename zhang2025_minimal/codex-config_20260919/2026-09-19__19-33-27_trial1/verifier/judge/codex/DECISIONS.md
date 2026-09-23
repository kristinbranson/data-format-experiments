# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use `one.search(...)` as the top-level session selector. Instead, it read the release manifest `code/code_zhang2025/data/bwm_release.csv`, optionally restricted it with `data/DATALIMIT_SUBSET.csv`, grouped rows by `eid`, and then processed each session by constructing a `ONE` client and calling `SessionLoader` and `SpikeSortingLoader`. Trials were then loaded from `loader.trials` inside each session worker.

ii.
```python
DEFAULT_RELEASE = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
```

```python
def _jobs(release_file: Path, cache_dir: Path, shard_dir: Path) -> list[dict]:
    release = pd.read_csv(release_file, index_col=0)
    subset = _read_subset(ROOT / "data" / "DATALIMIT_SUBSET.csv")
```

```python
one = _one(job["cache_dir"])
loader = SessionLoader(one=one, eid=eid)
loader.load_trials()
```

iii. In the trajectory, the agent said it was working on “the full 459-session BWM release” and chose to “build the full release rather than sampling sessions.” The script docstring says it follows `0_data_caching.py` and that the source release is large enough to justify per-session shards.

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from the `subject` column in the release CSV. Each session job stores one subject name, and the final dataset builds `subjects` as the sorted unique subject list with `subject_idx` pointing into it.

ii.
```python
jobs.append(
    {
        "eid": str(eid),
        "subject": str(first["subject"]),
```

```python
subjects = sorted({session["subject"] for session in sessions})
subject_map = {name: i for i, name in enumerate(subjects)}
```

iii. The trajectory does not contain a separate explicit defense of this choice; the AI implicitly treated the release manifest as the authoritative session index and reused its subject column throughout processing.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values. The AI groups the release manifest by `eid`, creates one job per `eid`, writes one shard per session, and later assembles one session entry per shard.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
```

```python
"shard": str(shard_dir / f"{eid}.pkl"),
```

iii. The trajectory repeatedly describes the work as a “full 459-session” conversion, so the AI clearly treated `eid` as the session unit.

## 1-d. How are the data split into trials?

i. Trials come from the IBL trials table loaded by `SessionLoader`. After loading `loader.trials`, the AI builds a trial mask and uses the surviving row indices (`candidate`) as the retained trials for that session.

ii.
```python
loader.load_trials()
trials = loader.trials.copy()
```

```python
candidate = np.flatnonzero(mask)
onsets = trials["stimOn_times"].to_numpy(dtype=float)[candidate]
```

iii. The trajectory does not show any extra justification here beyond following the IBL session loader and the BWM reference code structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI kept only trials with finite values in seven required columns, reaction time between 0.08 s and 2.0 s, `feedback_times - goCue_times <= 10 s`, nonzero choice, and `probabilityLeft` in `{0.2, 0.5, 0.8}`. It then further removed trials whose wheel or whisker streams did not fully cover the aligned time window, and dropped sessions left with fewer than two trials.

ii.
```python
required = [
    "stimOn_times",
    "choice",
    "feedback_times",
    "probabilityLeft",
    "firstMovement_times",
    "feedbackType",
    "goCue_times",
]
```

```python
mask = (
    finite
    & (reaction_time >= 0.08)
    & (reaction_time <= 2.0)
    & (trial_length <= 10.0)
    & (choice_raw != 0)
    & np.isin(probability, [0.2, 0.5, 0.8])
)
```

```python
behavior_good = wheel_good & motion_good
candidate = candidate[behavior_good]
if len(candidate) < 2:
    raise ValueError("fewer than two trials have complete wheel and whisker streams")
```

iii. The docstring says “the BWM trial exclusions are applied.” In progress updates the agent also said the 15 excluded sessions mostly lacked usable whisker-motion data, and one had no complete aligned trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` array is built from `spikes["times"]` and `spikes["clusters"]` loaded probe-by-probe with `SpikeSortingLoader`. The merged cluster table is used to recover region acronyms, but not to transform spike counts beyond merge bookkeeping.

ii.
```python
spikes, clusters, channels = spike_loader.load_spike_sorting()
merged = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
local_ids = np.asarray(spikes["clusters"], dtype=np.int64)
```

```python
spike_parts.append((np.asarray(spikes["times"]), local_ids, offset))
```

iii. The trajectory justification is explicit in the code comment: “Load and retain every Kilosort cluster, exactly as `prepare_data(qc=None)` in the methods repository.”

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session by offsetting cluster IDs, then bins spikes into 100 non-overlapping 20 ms bins spanning `[-0.5, 1.5]` s around stimulus onset. Unlike the human reference, it keeps the result as raw spike counts rather than dividing by bin width to convert to firing rate in Hz.

ii.
```python
bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
clu = clusters[lo:hi] + offset
```

```python
counts = np.bincount(flat, minlength=size)
return counts.reshape(len(onsets), n_neurons, N_BINS).astype(np.float32)
```

```python
neural_array = _bin_spikes(spike_parts, onsets, offset)
```

iii. The trajectory justification is that the preprocessing should follow `0_data_caching.py`, with “probes from one recording session combined” and “spikes counted in 20 ms bins from 0.5 s before through 1.5 s after stimulus onset.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not filtered by neuron quality. The AI intentionally kept every sorted Kilosort cluster from every released probe, with no `label >= 1` filter and no exclusion of `void` regions.

ii.
```python
# Load and retain every Kilosort cluster, exactly as prepare_data(qc=None)
# in the methods repository.  Probe-local IDs are shifted on merge.
```

```python
n_clusters = len(merged)
mapped = brain_regions.acronym2acronym(
    merged["acronym"].to_numpy(), mapping="Beryl"
).astype(str)
spike_parts.append((np.asarray(spikes["times"]), local_ids, offset))
region_parts.append(mapped)
offset += n_clusters
```

iii. The main justification visible in the trajectory is the comment above the code and the metadata string `"all sorted clusters (qc=None), matching the methods code"`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window is aligned to `stimOn_times`. For each retained trial, the AI uses a window from `stimOn_times - 0.5` s through `stimOn_times + 1.5` s and bins all spikes falling in that window.

ii.
```python
starts = onsets + OFF_START
ends = onsets + OFF_END
```

```python
for trial, (begin, end) in enumerate(zip(starts, ends)):
    lo = np.searchsorted(times, begin, side="left")
    hi = np.searchsorted(times, end, side="left")
```

iii. The trajectory explicitly says the “key adaptation is one common stimulus-aligned 2-second window at 20 ms resolution.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI used 20 ms bins and exactly 100 bins per trial over the 2 s window. No further temporal rebinning was applied.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

iii. The trajectory repeatedly states “20 ms resolution” and the final report says “100 × 20 ms stimulus-aligned bins per trial.”

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI did not derive this input from a raw sampled signal. It constructed a fixed relative time grid, then interpreted it relative to each trial’s `stimOn_times`.

ii.
```python
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```

```python
onsets = trials["stimOn_times"].to_numpy(dtype=float)[candidate]
```

iii. The justification in the code comment is that this is “exactly the grid used by the reference `get_behavior_per_interval`.”

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI used the 100 right-edge timestamps of the 20 ms bins: `[-0.48, -0.46, ..., 1.50]` s. It then copied that same vector into every retained trial.

ii.
```python
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```

```python
inp = np.empty((2, N_BINS), dtype=np.float32)
inp[0] = RELATIVE_TIMES
```

iii. The agent explicitly justified this with the comment “behavior values correspond to the right edge of each neural count bin.”

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned on the same stimulus-locked bin grid used for trialized neural and behavioral data, but the timestamp assigned to each bin is the right edge of the bin rather than the bin center.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
```

```python
inp[0] = RELATIVE_TIMES
```

iii. The AI’s justification is again the code comment that this is “exactly the grid used by the reference `get_behavior_per_interval`.”

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`. A new block begins whenever `probabilityLeft` changes.

ii.
```python
def _trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
```

```python
if i == 0 or not np.isfinite(value) or value != previous:
    number = 1
```

iii. The trajectory does not contain extra prose justification beyond implementing block changes from the prior-probability sequence.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a one-based within-block count. The count is built before trial exclusions, increments while `probabilityLeft` stays unchanged, and resets to 1 on the first trial, on a prior change, or on a non-finite prior.

ii.
```python
out = np.empty(len(probability_left), dtype=np.float32)
number = 0
previous = None
for i, value in enumerate(probability_left):
    if i == 0 or not np.isfinite(value) or value != previous:
        number = 1
    else:
        number += 1
    out[i] = number
```

```python
trial_in_block = _trial_number_in_block(probability)
```

iii. The metadata explicitly says `"trial_number_in_block": "one-based count, computed before trial exclusions"`, which is the clearest statement of the AI’s intended decision.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is taken directly from the trials table’s `choice` column.

ii.
```python
choice_raw = trials["choice"].to_numpy(dtype=float)
```

iii. No separate trajectory justification was given beyond using the trials table and dropping `choice_raw == 0`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI converted `choice_raw == 1` to category `1` and all retained nonzero alternatives to category `0`. Given the output labels `["left", "right"]`, this effectively treats raw `+1` as right and raw `-1` as left.

ii.
```python
choice = (choice_raw[candidate] == 1).astype(np.int8)  # left=-1, right=1
```

```python
"output_values": [
    ["left", "right"],
```

iii. The only explicit justification is the inline comment `# left=-1, right=1`, which shows the AI’s understanding of the raw coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trials table’s `probabilityLeft` column.

ii.
```python
probability = trials["probabilityLeft"].to_numpy(dtype=float)
```

iii. The trajectory contains no separate argument here; the mapping is straightforward from the task instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, and applies that mapping only on retained trials.

ii.
```python
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_lookup[float(x)] for x in probability[candidate]], dtype=np.int8)
```

iii. The justification is implicit from the user task, which specified exactly those category assignments.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed comes from the wheel stream loaded by `SessionLoader`. The AI uses `loader.wheel["times"]` and the absolute value of `loader.wheel["velocity"]`.

ii.
```python
loader.load_wheel()
wheel_times = loader.wheel["times"].to_numpy()
wheel_speed = np.abs(loader.wheel["velocity"].to_numpy())
```

iii. The trajectory justification is indirect: the agent said it was following the supplied BWM processing and used the session loader’s wheel preprocessing rather than recomputing velocity itself.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI took the absolute wheel velocity, interpolated it onto the trial-aligned 100-bin grid after a coverage check, and then discretized the full session-wide retained trace into tertiles.

ii.
```python
wheel, wheel_good = _interpolate_trials(wheel_times, wheel_speed, candidate_onsets)
```

```python
wheel_cat, wheel_cuts = _session_tertiles(wheel)
```

iii. The docstring says the two continuous decoding targets are converted to “session-wise tertiles,” and the interpolation helper says it uses the “reference coverage checks.”

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is thresholded into three equal-frequency within-session bins using the 1/3 and 2/3 quantiles of the retained wheel-speed samples.

ii.
```python
def _session_tertiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cuts = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.digitize(values, cuts, right=False).astype(np.int8)
```

iii. The trajectory justification is explicit in the top-level docstring: continuous targets are converted to “session-wise tertiles as required by this task.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each retained trial, wheel speed is linearly interpolated at `stimOn_times + RELATIVE_TIMES`, where `RELATIVE_TIMES` are the right edges of the neural bins.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
```

iii. The AI justified this with the code comment that the reference behavior grid corresponds to “the right edge of each neural count bin.”

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from a camera motion-energy stream loaded by `SessionLoader`. The AI prefers the left camera and falls back to the right camera if needed, using each camera’s `times` and `whiskerMotionEnergy` columns.

ii.
```python
for view in ("left", "right"):
    session_loader.load_motion_energy(views=[view])
    key = f"{view}Camera"
    frame = session_loader.motion_energy[key]
    times = frame["times"].to_numpy()
    values = frame["whiskerMotionEnergy"].to_numpy()
```

iii. The trajectory repeatedly notes that excluded sessions were mostly missing usable whisker-motion streams, so this source stream was treated as required for inclusion.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI used the released motion-energy values as-is, interpolated them onto the trial-aligned time grid after the same coverage checks used for wheel, and then discretized the retained session-wide samples into tertiles.

ii.
```python
motion, motion_good = _interpolate_trials(motion_times, motion_values, candidate_onsets)
```

```python
motion_cat, motion_cuts = _session_tertiles(motion)
```

iii. The code and trajectory do not mention any extra filtering or normalization; the visible justification is to reuse the same reference-style interpolation and tertile discretization used for wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded into three equal-frequency within-session bins using the 1/3 and 2/3 quantiles of the retained motion-energy samples.

ii.
```python
motion_cat, motion_cuts = _session_tertiles(motion)
```

iii. The top-level docstring states that both continuous targets are turned into session-wise tertiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. For each retained trial, whisker motion energy is linearly interpolated at `stimOn_times + RELATIVE_TIMES`, where `RELATIVE_TIMES` are the right-edge timestamps of the neural bins.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
```

iii. The AI’s explicit rationale is the same right-edge-grid comment used for wheel speed and time-since-stimulus-onset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly drops bad data rather than repairing it. Missing required trial columns raise an error for the session; malformed continuous streams raise an error; missing left whisker data triggers a right-camera fallback; trials lacking complete wheel or whisker coverage are dropped; sessions with no neurons or fewer than two retained trials are skipped. It also caches successful session shards so interrupted runs can resume.

ii.
```python
missing = [name for name in required if name not in trials]
if missing:
    raise ValueError(f"missing trial columns: {missing}")
```

```python
for view in ("left", "right"):
```

```python
behavior_good = wheel_good & motion_good
candidate = candidate[behavior_good]
```

```python
if offset == 0:
    raise ValueError("session has no neural units")
```

iii. In the trajectory the AI said the restartable shards were deliberate because the source release is large, and later explained that excluded sessions mostly lacked usable whisker-motion data.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading large per-probe spike-sorting data and writing/re-reading large per-session shard files. The code is structured around those costs by parallelizing sessions and making shards restartable.

ii.
```python
spikes, clusters, channels = spike_loader.load_spike_sorting()
```

```python
with temporary.open("wb") as stream:
    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The script docstring says “The source release is large,” and the trajectory repeatedly emphasizes the size of the full release and the need for restartable shards.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious hot loops remain unvectorized: the per-trial interpolation loop in `_interpolate_trials` and the nested per-probe/per-trial spike-binning loop in `_bin_spikes`.

ii.
```python
for i, onset in enumerate(onsets):
```

```python
for times, clusters, offset in spike_parts:
    ...
    for trial, (begin, end) in enumerate(zip(starts, ends)):
```

iii. The trajectory does not discuss vectorization explicitly; this is an implementation choice visible in the final code.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several operations: it converts `RELATIVE_TIMES` to `float64` inside every trial loop, runs nearly the same interpolation procedure separately for wheel and whisker streams, re-creates a `ONE` client in every worker, writes each session to a shard, and later re-reads all shards during assembly.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
```

```python
wheel, wheel_good = _interpolate_trials(wheel_times, wheel_speed, candidate_onsets)
motion, motion_good = _interpolate_trials(motion_times, motion_values, candidate_onsets)
```

```python
one = _one(job["cache_dir"])
```

```python
with temporary.open("wb") as stream:
    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
...
with shard.open("rb") as stream:
    payload = pickle.load(stream)
```

iii. The AI explicitly justified the shard write/read pattern in the docstring: workers write restartable session shards so the parent only assembles once at the end.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores extra per-session metadata that the downstream decoder does not use, such as `date`, `lab`, `probe_names`, `n_source_trials`, and the continuous-variable tertile cutpoints. It also writes `failures.json` and does temporary per-session shard I/O that is not part of the final decoder dataset.

ii.
```python
"info": {
    "eid": eid,
    "subject": job["subject"],
    "date": job["date"],
    "lab": job["lab"],
    "probe_names": list(job["probe_names"]),
    "n_source_trials": int(len(trials)),
    "n_trials": int(len(candidate)),
    "n_neurons": int(offset),
    "motion_energy_source": motion_source,
    "wheel_speed_tertile_edges": wheel_cuts,
    "whisker_motion_energy_tertile_edges": motion_cuts,
},
```

```python
failure_file = args.shard_dir / "failures.json"
failure_file.write_text(json.dumps(failures, indent=2))
```

iii. The trajectory justifies the shard machinery as an engineering choice for restartability on a very large release; it does not claim that the extra metadata is needed by the downstream decoder.
