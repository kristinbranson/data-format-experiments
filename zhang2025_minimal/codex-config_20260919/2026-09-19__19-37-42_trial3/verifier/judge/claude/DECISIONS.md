# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** load data through the ONE API. It reads the reference repository's own release table, `/app/code/code_zhang2025/data/bwm_release.csv` (459 sessions / 699 insertions, columns `eid, pid, subject, probe_name`), and uses that as the authoritative list of sessions and probe insertions. ONE is instantiated only in `mode='local'` and only to resolve `eid -> session directory` via `one.eid2path`. Everything after that is a direct filesystem read of the ALF files under that directory: the trials table (`alf/**/_ibl_trials.table.pqt`), the wheel (`alf/**/_ibl_wheel.position.npy`, `_ibl_wheel.timestamps.npy`), the camera (`alf/**/{view}Camera.ROIMotionEnergy.npy`, `alf/**/_ibl_{view}Camera.times.npy`) and, per probe, `alf/<probe>/pykilosort/**/{spikes.times, spikes.clusters, clusters.channels, channels.brainLocationIds_ccf_2017}.npy`. Because several datasets exist at multiple revisions, a helper `_latest()` reimplements ONE's revision resolution by picking the newest `#YYYY-MM-DD#` folder (an unrevised file sorts as `0000-00-00`). Sessions are converted in parallel with a 16-worker `ThreadPoolExecutor`, and each session is pickled to its own checkpoint in `/app/.converted_sessions/` before a final `_assemble()` pass reads all checkpoints back and writes `/app/converted_data.pkl`. A `DATALIMIT_SUBSET.csv` restriction is supported but that file was absent, so the full release was used.

ii.
```python
BWM_TABLE = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def _load_release_specs(max_sessions: int | None = None) -> list[SessionSpec]:
    release = pd.read_csv(BWM_TABLE, index_col=0)
    ...
    one = ONE(mode="local", cache_dir=CACHE_DIR, silent=True)
    One.load_cache(one, CACHE_DIR / "Brainwidemap")

    specs: list[SessionSpec] = []
    for eid, rows in release.groupby("eid", sort=False):
        session_path = Path(one.eid2path(str(eid)))
        specs.append(SessionSpec(eid=str(eid), subject=str(rows.iloc[0]["subject"]),
                                 session_path=session_path,
                                 probes=tuple(rows["probe_name"].astype(str))))
```

```python
def _latest(paths) -> Path | None:
    candidates = [Path(p) for p in paths if Path(p).is_file()]
    if not candidates:
        return None

    def key(path: Path):
        revisions = re.findall(r"#(\d{4}-\d{2}-\d{2})#", str(path))
        return (revisions[-1] if revisions else "0000-00-00", str(path))

    return max(candidates, key=key)
```

```python
def _spike_files(session_path: Path, probe: str) -> tuple[Path, Path, Path, Path]:
    base = session_path / "alf" / probe / "pykilosort"
    return (
        _required_file(base.glob("**/spikes.times.npy"), f"{probe} spike times"),
        _required_file(base.glob("**/spikes.clusters.npy"), f"{probe} spike clusters"),
        _required_file(base.glob("**/clusters.channels.npy"), f"{probe} cluster channels"),
        _required_file(base.glob("**/channels.brainLocationIds_ccf_2017.npy"),
                       f"{probe} channel atlas ids"),
    )
```

iii. From the trajectory: the AI first exercised the ONE loaders (`SessionLoader`, `SpikeSortingLoader`, `load_spiking_data`) on a test session (steps 28–29, 42–45) and found them working but slow, and it explicitly verified (step 61) that reading `clusters.channels` + `channels.brainLocationIds_ccf_2017` and mapping through `BrainRegions` reproduces the `acronym` column that `SpikeSortingLoader.merge_clusters` attaches. It then chose the direct-read route for speed and for memory-mapped access to the spike files (`np.load(..., mmap_mode="r")`), because it had already estimated the output at ~106 GB float32 (step 47: `estimated float32 GB all 109.85`) and stated at step 65 that "I'm implementing the full converter with resumable per-session checkpoints because the final dense float32 neural payload is about 106 GB." It used `bwm_release.csv` as the session list because that is the table the reference repository itself iterates over in `src/0_data_caching.py`.

## 1-b. How are the data split into subjects?

i. Subjects are taken verbatim from the `subject` column of `bwm_release.csv`, carried on each `SessionSpec` and then on each session's checkpoint payload. At assembly, `subjects` is the sorted set of unique subject names and `subject_idx` is each session's index into that list. Result: 136 subjects over 444 sessions.

ii.
```python
specs.append(SessionSpec(eid=str(eid), subject=str(rows.iloc[0]["subject"]), ...))
```
```python
subjects = sorted({payload["subject"] for payload in payloads})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_to_idx[payload["subject"]] for payload in payloads],
                          dtype=np.int32),
```

iii. No explicit justification is given in the trajectory beyond the release table already carrying a unique subject id per session (the AI inspected the table's columns at step 10), so nothing has to be derived or parsed from paths.

## 1-c. How are the data split into sessions?

i. A session is one `eid`. The AI groups `bwm_release.csv` by `eid` (`release.groupby("eid", sort=False)`), which collapses the per-insertion rows into one `SessionSpec` per session carrying the tuple of probe names for that session. 459 sessions enter; 15 are dropped during conversion (see 9), leaving 444 in the pickle, in release-table order.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    ...
    probes=tuple(rows["probe_name"].astype(str))
```

iii. The AI confirmed the scale at step 48: "The release audit confirms the expected scale: 459 curated BWM sessions and 699 insertions". It treated the session as the natural unit because the reference repo merges all probes of a session into a single population (`merge_probes`), so per-session grouping of insertions is required.

## 1-d. How are the data split into trials?

i. Trials are the rows of `_ibl_trials.table.pqt`; no splitting is performed. The row order is preserved, a boolean keep-mask is built over all rows, and the surviving rows (`keep_idx`) become the per-trial entries of `neural`/`input`/`output`.

ii.
```python
def _load_trials(session_path: Path) -> pd.DataFrame:
    table = _required_file(session_path.glob("alf/**/_ibl_trials.table.pqt"), "trials table")
    trials = pd.read_parquet(table)
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType", "goCue_times"]
    missing = sorted(set(required) - set(trials.columns))
    if missing:
        raise ValueError(f"trials table lacks columns: {missing}")
    return trials
```
```python
stimulus_times = trials["stimOn_times"].to_numpy(dtype=np.float64)
...
keep = paper_mask & wheel_good & motion_good
keep_idx = np.flatnonzero(keep)
```

iii. Nothing to justify: the AI inspected the table at step 30 and confirmed one row per trial with the expected columns.

## 1-e. How are trials filtered based on quality controls?

i. Two filters combined. The first, `_paper_trial_mask`, is a direct reimplementation of the reference repository's `load_trials_and_mask(one, eid, max_trial_len=10.0)` as it is called from `prepare_data`: no NaN in `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`; `choice != 0` (no-response trials dropped); reaction time (`firstMovement_times - stimOn_times`) in [0.08, 2.0] s; and `feedback_times - goCue_times <= 10` s. The second is stream coverage: a trial is kept only if the wheel trace and the camera trace each have a sample within one bin (20 ms) of both the window start and the window end, and the interpolated trace is finite — this reproduces the repo's `get_behavior_per_interval` skip reasons ("target data starts too late" / "ends too early"). There is no explicit `probabilityLeft in {0.2, 0.5, 0.8}` filter; instead the code raises if any *kept* trial has a prior outside that set. Sessions left with fewer than 2 kept trials are dropped entirely. Net effect: 188,925 trials over 444 sessions (audit at step 50: mean 426.5 paper-eligible → 411.6 after coverage per session).

ii.
```python
def _paper_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    """Trial exclusions from the BWM paper and the supplied repository."""
    needed = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
              "firstMovement_times", "feedbackType"]
    reaction_time = trials["firstMovement_times"] - trials["stimOn_times"]
    trial_duration = trials["feedback_times"] - trials["goCue_times"]
    return (
        trials[needed].notna().all(axis=1).to_numpy()
        & (trials["choice"].to_numpy() != 0)
        & (reaction_time.to_numpy() >= 0.08)
        & (reaction_time.to_numpy() <= 2.0)
        & (trial_duration.to_numpy() <= 10.0)
    )
```
```python
    if abs(begins[trial] - times[0]) > BIN_SIZE:
        continue
    if abs(ends[trial] - times[-1]) > BIN_SIZE:
        continue
    ...
    good[trial] = np.isfinite(values[trial]).all()
```
```python
keep = paper_mask & wheel_good & motion_good
keep_idx = np.flatnonzero(keep)
if len(keep_idx) < 2:
    raise ValueError(f"only {len(keep_idx)} aligned valid trials")
```

iii. The AI read `load_trials_and_mask` and `prepare_data` in full (steps 9–10) and stated at step 14: "the reference code ... applies the paper's trial exclusions (missing events, no-choice, 80 ms–2 s first-movement latency, and ≤10 s trial duration)." It also checked (step 59) whether a trial-count threshold could reproduce the methods paper's 433 usable sessions and concluded at step 48 that "the methods paper's reported 433 usable sessions is therefore a downstream availability/alignment subset, not a different release. I'm reproducing that reduction from the actual wheel/camera coverage and finite-window checks rather than hard-coding a session list."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from `alf/<probe>/pykilosort/`, per insertion listed in `bwm_release.csv`. Two further arrays are read only to label the neurons, not to build the matrix: `clusters.channels.npy` (peak channel of each cluster) and `channels.brainLocationIds_ccf_2017.npy` (Allen CCF id of each channel), which are composed and mapped through `iblatlas.regions.BrainRegions.id2acronym(..., mapping="Beryl")` to give `brain_region_idx`.

ii.
```python
times = np.load(times_path, mmap_mode="r")
clusters = np.load(clusters_path, mmap_mode="r")
```
```python
cluster_channels = np.load(cluster_channels_path).astype(np.int64, copy=False)
channel_atlas_ids = np.load(atlas_ids_path)
counts, used = _bin_probe(times_path, clusters_path, interval_begins, len(cluster_channels))
cluster_atlas_ids = channel_atlas_ids[cluster_channels[used]]
regions = atlas.id2acronym(cluster_atlas_ids, mapping="Beryl")
```

iii. Step 61 of the trajectory is an explicit equivalence check: the AI compared `BrainRegions.id2acronym(channel_ids[cluster_channels])` mapped to Beryl against `SpikeSortingLoader.merge_clusters(...).acronym` mapped to Beryl for a test probe and confirmed they agree, which is why it felt safe reading the two small `.npy` files instead of paying for `merge_clusters`.

## 2-b. How is the `neural` data processed?

i. Spikes are counted, per probe, into 100 non-overlapping 20 ms bins covering `[stimOn - 0.5, stimOn + 1.5)` s, using `bin = floor((t - window_start) / 0.02)` with out-of-range bins discarded — a deliberate reimplementation of `iblutil.numerical.bincount2D(times, clusters, xbin=0.02, xlim=[t_beg, t_end])[:, :n_bins]` as used by the repo's `get_spike_data_per_interval`. Only clusters that emit at least one spike anywhere in the probe's spike train are represented (`used = np.unique(clusters)`), matching `bincount2D`'s `ybin=0` behaviour. The per-probe matrices are then concatenated along the neuron axis so a session's probes form one pooled population (the equivalent of the repo's `merge_probes`). **No conversion to firing rate is applied** — values are raw spike counts per 20 ms bin, stored as `float32`. No smoothing. Mean 1,351 neurons per session, max 3,140; 599,865 neurons in total.

ii.
```python
binned = np.zeros((len(interval_begins), len(used), N_BINS), dtype=np.float32)
interval_ends = interval_begins + (OFF_END - OFF_START)
start_idx = np.searchsorted(times, interval_begins, side="left")
end_idx = np.searchsorted(times, interval_ends, side="left")
for trial, (lo, hi) in enumerate(zip(start_idx, end_idx)):
    if hi <= lo:
        continue
    spike_bins = np.floor((np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE).astype(np.int64)
    cluster_ids = np.asarray(clusters[lo:hi], dtype=np.int64)
    if remap is not None:
        cluster_ids = remap[cluster_ids]
    valid = ((cluster_ids >= 0) & (cluster_ids < len(used))
             & (spike_bins >= 0) & (spike_bins < N_BINS))
    flat = cluster_ids[valid] * N_BINS + spike_bins[valid]
    binned[trial].flat[:] = np.bincount(flat, minlength=len(used) * N_BINS).astype(np.float32, copy=False)
```
```python
neural = np.concatenate(probe_bins, axis=1)
regions = np.concatenate(probe_regions)
```

iii. Docstring: "The implementation follows the preprocessing in Zhang et al. (2025): all Kilosort 2.5 clusters from all probes in a session are combined, spikes are counted in non-overlapping 20 ms bins". Step 14: "The reference code combines all probes within a session, uses 20 ms spike-count bins over a 2 s stimulus-aligned window". The AI printed the source of `bincount2D` (step 33) before writing `_bin_probe`, and its in-code comment says "Bin one probe exactly as bincount2D, without rescanning the full spike train", with a second comment explaining the `used`/`remap` logic: "The reference utility only returns cluster IDs represented in the spike train."

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron quality control is applied at all.** Every Kilosort cluster with at least one spike is kept: the IBL `label` metric is never read, no `label >= 1` ("well-isolated neuron") cut is applied, and no anatomical exclusion is applied either — `root` (85,656 units) and `void` (12,827 units, i.e. channels the histology placed outside the brain) are both retained. The only implicit exclusions are clusters that never spike and probes whose files are missing. This yields 599,865 neurons (mean 1,351/session) against the expert's 72,417 (mean 164/session), and a 106 GB / 100 GiB pickle. The choice is recorded in the metadata as `"neuron_filter": "all Kilosort clusters, matching the methods-paper decoder pipeline"`.

ii. There is no filtering code to quote; the absence is the decision. The neuron set is defined purely by which clusters spike:
```python
    used = np.unique(clusters)
    ...
binned = np.zeros((len(interval_begins), len(used), N_BINS), dtype=np.float32)
```
```python
"neuron_filter": "all Kilosort clusters, matching the methods-paper decoder pipeline",
```

iii. The justification is consistency with the methods-paper code rather than the data paper. The AI read `load_spiking_data(one, pid, compute_metrics=False, qc=None, ...)` (step 28, where it printed the `label` column and saw it exists) and `prepare_data`, which calls `load_spiking_data` without a `qc` argument, so the repo's cached datasets contain every sorted cluster. The AI's audit at step 47 reported "ok clusters/trials totals 600174 189315" and "estimated float32 GB all 109.85", so it knew the consequence and accepted it, choosing checkpointed conversion to cope with the size (step 65) rather than reducing the neuron count. It did not comment anywhere in the trajectory on the data paper's statement that stringent quality control "identified 75,708 well-isolated neurons", even though it grepped the data paper for "well-isolated" (steps 17/35).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is on `trials.stimOn_times`, by subtraction on the single shared session clock (IBL synchronises ephys, behaviour and video upstream). For each kept trial the window start is `stimOn - 0.5` s; spike times inside `[start, start + 2.0)` are located with `searchsorted` and the bin index is `floor((t - start) / 0.02)`, so bin 0 begins exactly at the alignment event minus 500 ms. `metadata['temporal_alignment_event']` is `"visual stimulus onset (trials.stimOn_times)"`, with `off_start = -0.5`, `off_end = 1.5`.

ii.
```python
interval_begins = stimulus_times[keep] + OFF_START
```
```python
spike_bins = np.floor((np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE).astype(np.int64)
```
```python
"temporal_alignment_event": "visual stimulus onset (trials.stimOn_times)",
"off_start": OFF_START,
"off_end": OFF_END,
```

iii. Dictated by the instructions ("Temporally align based on stimulus onset") and by the repo's `params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`, which the AI quoted at step 14 ("a 2 s stimulus-aligned window"). No clock correction is needed because all streams share the session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, identical for every trial of every session (`T_min = T_max = 100`). `metadata['time_bin_size'] = 20.0` (ms). No rebinning, resampling or smoothing is applied to the neural data — spikes are counted directly onto the final grid, so there is only ever one binning step. The behavioural traces are sampled onto that same 100-point grid (see 7-d / 8-d).

ii.
```python
OFF_START = -0.5
OFF_END = 1.5
BIN_SIZE = 0.02
N_BINS = 100
```
```python
"time_bin_size": 20.0,
```

iii. Step 14: "uses 20 ms spike-count bins over a 2 s stimulus-aligned window". This is the repo's `'binsize': 0.02, 'interval_len': 2, 'time_window': (-.5, 1.5)`, which the AI read at step 8, and the methods paper's T = 100 time steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable at all — it is the fixed bin grid defined by the alignment event `trials.stimOn_times` and the constants `OFF_START = -0.5`, `BIN_SIZE = 0.02`, `N_BINS = 100`. The same 100-value vector is written into every trial of every session.

ii.
```python
relative_time = (OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE).astype(np.float32)
inputs = np.empty((len(keep_idx), 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = relative_time
```

iii. Implied by the decoder-task specification ("Time since stimulus onset, continuous, time-varying") together with the repo's window and bin size; the AI recorded the convention explicitly in the metadata as `"time_coordinate_convention": "left edge of each 20 ms neural bin"`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. It is a constant `float32` ramp from -0.5 to 1.48 s in 20 ms steps, giving the **left edge** of each neural bin, broadcast across all trials. It is not converted to a binary onset indicator (the instructions' "represent it as a binary time series" clause applies to event-time inputs; the AI kept it continuous as the decoder-task spec requests).

ii.
```python
relative_time = (OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE).astype(np.float32)
```

iii. Nothing to justify; the metadata field `time_coordinate_convention` states the left-edge choice, which matches the repo's `get_spike_data_per_interval`, whose returned time index is also the start/left edge of each bin.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural grid: `relative_time[k]` is by construction the left edge of neural bin `k`, and both come from the same `OFF_START`/`BIN_SIZE`/`N_BINS` constants and the same `stimOn_times`. Bin-for-bin the two describe the same 20 ms interval. (Note that the behavioural outputs are instead sampled at bin *right* edges — see 7-d — so the input time label and the output sample point differ by one bin width.)

ii.
```python
relative_time = (OFF_START + np.arange(N_BINS, dtype=np.float32) * BIN_SIZE).astype(np.float32)
```
```python
interval_begins = stimulus_times[keep] + OFF_START
spike_bins = np.floor((np.asarray(times[lo:hi]) - interval_begins[trial]) / BIN_SIZE).astype(np.int64)
```

iii. Same constants, same alignment event; no additional alignment step is required or claimed.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The trials table carries no block id, so blocks are recovered as maximal runs of a constant `probabilityLeft`.

ii.
```python
probability_left = trials["probabilityLeft"].to_numpy(dtype=np.float64)
trial_in_block = _trial_number_in_block(probability_left)
```

iii. Implicit; the AI inspected the trials table columns at step 30 and confirmed `probabilityLeft` is the only block-related variable available.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based run-length counter over the **full, unfiltered** trial sequence: the counter increments while `probabilityLeft` is unchanged from the previous row and resets to 0 when it changes. Computing it before filtering means a trial later dropped by the quality mask still advances the count, so the value is the animal's true position in the block. The result is cast to `float32` and broadcast constant across the 100 time bins of the trial. Observed range 0–98, identical to the expert's.

ii.
```python
def _trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based trial position in each uninterrupted probability block."""
    out = np.zeros(len(probability_left), dtype=np.float32)
    for trial in range(1, len(probability_left)):
        out[trial] = (out[trial - 1] + 1
                      if probability_left[trial] == probability_left[trial - 1]
                      else 0)
    return out
```
```python
inputs[:, 1, :] = trial_in_block[keep, None]
```

iii. Recorded in the metadata as `"trial_number_in_block_convention": "zero-based run-length within the full unfiltered probabilityLeft sequence"`. The decoder-task spec asks for a per-trial continuous input, so it is held constant over the trial's time bins.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward turn), -1 (rightward turn) or 0 (no response). No-response trials never reach the output because `choice != 0` is part of the trial mask.

ii.
```python
raw_choice = trials["choice"].to_numpy()
# In IBL choice coding, -1 is the rightward response and +1 is leftward.
choice = (raw_choice == -1).astype(np.int64)
```

iii. The in-code comment states the IBL convention; the AI grepped the repo and ibllib for the choice coding at steps 30–31 before writing it. The instructions fix the target encoding (left = 0, right = 1).

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only the recode: `choice == -1` → 1 (right), otherwise 0 (left). Because `choice == 0` trials are already excluded, the "otherwise" branch is exactly `choice == +1`. The scalar is broadcast constant over the 100 time bins and stored as `int64`. `output_values[0] = ['left', 'right']`. Observed class balance 0.508 / 0.492.

ii.
```python
outputs = np.empty((len(keep_idx), 4, N_BINS), dtype=np.int64)
outputs[:, 0, :] = choice[keep, None]
```
```python
"output_values": [["left", "right"], ...]
```

iii. Straight from the instructions ("Choice, binary, per-trial, left = 0, right = 1"); made time-varying by broadcasting because the instructions ask for time-varying outputs "if at all possible".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, the block prior held constant within a block.

ii.
```python
probability_left = trials["probabilityLeft"].to_numpy(dtype=np.float64)
```

iii. Named directly by the instructions ("Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2").

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A three-way recode using `np.isclose` (float-safe): 0.2 → 0, 0.5 → 1, 0.8 → 2. Trials are initialised to -1 and the code raises if any *kept* trial still holds -1, i.e. an unexpected prior value aborts the session rather than silently producing a bogus class. The label is broadcast constant over the 100 bins. Observed fractions 0.417 / 0.141 / 0.442, essentially identical to the expert's 0.418 / 0.141 / 0.442.

ii.
```python
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.full(len(trials), -1, dtype=np.int64)
for value, label in prior_lookup.items():
    prior[np.isclose(probability_left, value)] = label
if np.any(prior[keep] < 0):
    raise ValueError("probabilityLeft contains a value outside {0.2, 0.5, 0.8}")
```
```python
outputs[:, 1, :] = prior[keep, None]
```

iii. The mapping is given by the instructions. `np.isclose` rather than `==` is used because the column is float; the guard converts an unexpected value into a loud failure instead of a silent mislabel.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The raw wheel encoder, read directly as `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. Speed is the absolute value of the velocity derived from them.

ii.
```python
position_path = _required_file(session_path.glob("alf/**/_ibl_wheel.position.npy"), "wheel position")
timestamps_path = _required_file(session_path.glob("alf/**/_ibl_wheel.timestamps.npy"), "wheel timestamps")
position = np.load(position_path)
timestamps = np.load(timestamps_path)
if len(position) != len(timestamps):
    raise ValueError("wheel position/timestamp length mismatch")
```

iii. The reference repo's `load_target_behavior(..., 'wheel-speed')` returns `np.abs(sess_loader.wheel['velocity'])`; the AI printed `SessionLoader.load_wheel`'s source (step 32) and reimplemented it over the raw arrays.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Exactly `SessionLoader.load_wheel`'s defaults, reimplemented inline: the irregularly-sampled encoder position is interpolated onto a uniform 1 kHz grid (`interpolate_position(..., freq=1000)`), differentiated into a velocity with a 20 Hz corner-frequency, order-8 Butterworth low-pass (`velocity_filtered(..., fs=1000, corner_frequency=20, order=8)`), and the absolute value taken (rad/s). That 1 kHz trace is then linearly interpolated onto the 100 per-trial sample points (see 7-d), and finally discretized (see 7-c). No normalisation.

ii.
```python
position_1khz, times_1khz = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(position_1khz, fs=1000, corner_frequency=20, order=8)
return _interpolate_trials(times_1khz, np.abs(velocity), stimulus_times)
```

iii. Docstring: "wheel velocity is computed from the 1 kHz interpolated/low-pass-filtered wheel trace". The AI inspected both `SessionLoader.load_wheel` and `brainbox/behavior/wheel.py` (steps 32–33) and copied the library defaults so the trace is identical to what the repo would obtain through `SessionLoader`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three equally-populated classes by **within-session tertiles**, computed once per session over the pooled retained-trial × time-bin matrix (i.e. over all `n_kept_trials × 100` samples of that session), using `np.quantile(values, [1/3, 2/3])` and `np.digitize(..., right=False)` to give labels 0/1/2 = low/medium/high. The two cut points are recorded per session in `metadata['session_info'][i]['wheel_speed_tertile_edges']`. Resulting global class fractions are 0.3333 / 0.3333 / 0.3333.

ii.
```python
def _three_bins(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Discretize a session's samples into low/middle/high tertiles."""
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
    return labels, (float(q1), float(q2))
```
```python
wheel = wheel[keep]
wheel_labels, wheel_edges = _three_bins(wheel)
...
outputs[:, 2, :] = wheel_labels
```

iii. Step 14 framed this as "the key ambiguity the task introduces: how to discretize the two continuous behaviors consistently across this entire release without leakage or session-specific label drift", and step 65 gives the answer: "The behavioral classes will use within-session tertiles pooled over retained time bins; that preserves three comparable low/medium/high states despite large camera- and rig-dependent scale differences, and avoids one laboratory's absolute units dominating the global thresholds." Recorded in metadata as `"continuous_output_discretization": "Within-session tertiles pooled over all retained trials and time bins"`.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Same alignment event and same window: for each trial the samples strictly inside `[stimOn - 0.5, stimOn + 1.5]` are selected (`searchsorted` with `side='right'` / `side='left'`), coverage is checked to within one bin at each edge, and the trace is linearly interpolated (`interp1d`, `fill_value='extrapolate'`) onto `np.linspace(begin + 0.02, end, 100)` — i.e. the **right edge** of each of the 100 neural bins. So sample *k* is the wheel speed at the end of neural bin *k*; this is one bin width later than the `time_since_stimulus_onset` input, which labels the left edge.

ii.
```python
begins = stimulus_times + OFF_START
ends = stimulus_times + OFF_END
idx_beg = np.searchsorted(sample_times, begins, side="right")
idx_end = np.searchsorted(sample_times, ends, side="left")
...
    target_times = np.linspace(begins[trial] + BIN_SIZE, ends[trial], N_BINS, dtype=np.float64)
    values[trial] = interp1d(times, vals, kind="linear", fill_value="extrapolate")(target_times)
    good[trial] = np.isfinite(values[trial]).all()
```

iii. The function's docstring says it all: "Match repository endpoint interpolation and interval validity checks." This is a line-for-line port of `get_behavior_per_interval` in `ibl_data_utils.py`, which uses `x_interp = np.linspace(interval_begs + binsize, interval_ends, n_bins)` and the same `abs(interval_beg - target_time[0]) > binsize` / `abs(interval_end - target_time[-1]) > binsize` skip rules. The wheel is on the same session clock as the spikes, so no further alignment is needed.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The side-camera ROI motion energy, read directly as `{view}Camera.ROIMotionEnergy.npy` with frame times `_ibl_{view}Camera.times.npy`. The left camera is used when available and the right camera is the fallback. (For side cameras the ROI is the whisker pad, and `SessionLoader.load_motion_energy` merely renames `ROIMotionEnergy` to `whiskerMotionEnergy`, so the array read here is the same one the repo uses.) 437 sessions use the left camera, 8 the right (audit at step 50).

ii.
```python
for view in ("left", "right"):
    try:
        energy_path = _required_file(session_path.glob(f"alf/**/{view}Camera.ROIMotionEnergy.npy"),
                                     f"{view} whisker motion energy")
        times_path = _required_file(session_path.glob(f"alf/**/_ibl_{view}Camera.times.npy"),
                                    f"{view} camera timestamps")
        ...
        return values, good, view
    except Exception as exc:  # left-to-right fallback in the reference code
        errors.append(f"{view}: {exc}")
raise RuntimeError("; ".join(errors))
```

iii. The comment "left-to-right fallback in the reference code" points at `bin_behaviors`, which does exactly this: `target_dict = load_target_behavior(one, eid, 'left-whisker-motion-energy'); if 'skip' in target_dict: ... 'right-whisker-motion-energy'`. The AI verified the file layout and array shapes at step 46.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, smoothing or normalisation. One correction is applied first: `SessionLoader._check_video_timestamps` is reproduced, so if there are more timestamps than motion-energy samples (pre-GPIO sessions where the first frames were not written) the leading timestamps are dropped, and if there are fewer the view is rejected and the fallback tried. The trace is then resampled onto the 100 per-trial sample points by the same `_interpolate_trials` used for the wheel, and discretized (8-c).

ii.
```python
energy = np.load(energy_path)
times = np.load(times_path)
# This is SessionLoader._check_video_timestamps: older camera sessions
# may contain timestamps for a few initial frames that were not saved.
if len(times) < len(energy):
    raise ValueError("camera timestamps are shorter than motion energy")
if len(times) > len(energy):
    times = times[-len(energy):]
values, good = _interpolate_trials(times, energy, stimulus_times)
```

iii. The AI printed the source of `SessionLoader._check_video_timestamps` (step 46) and transcribed it, precisely so that reading the `.npy` files directly would not diverge from what the repo obtains via `SessionLoader`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: within-session tertiles (`np.quantile(..., [1/3, 2/3])` + `np.digitize`) over the pooled retained-trial × time-bin matrix, giving labels 0/1/2 = low/medium/high, with the cut points stored per session as `whisker_motion_energy_tertile_edges`. Global class fractions 0.3326 / 0.3326 / 0.3348 (not exactly a third because motion energy has ties/flat stretches).

ii.
```python
motion = motion[keep]
motion_labels, motion_edges = _three_bins(motion)
...
outputs[:, 3, :] = motion_labels
```

iii. Same reasoning as 7-c, and the AI noted the scale problem explicitly at step 65 ("large camera- and rig-dependent scale differences") — motion energy in particular is in arbitrary units that differ between the 60 Hz full-resolution left camera and the 150 Hz half-resolution right camera, so a global threshold would be meaningless. It also pre-audited for degenerate sessions where the tertile edges coincide (step 51) and found only one.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Exactly as the wheel (7-d): same `_interpolate_trials` helper, same stimulus-onset-anchored window, same one-bin coverage tolerance at both edges, same 100 target times `linspace(stimOn - 0.48, stimOn + 1.5, 100)` — the right edge of each neural bin. The camera frame times are on the same session clock as the spikes, so subtraction is all that is needed.

ii.
```python
values, good = _interpolate_trials(times, energy, stimulus_times)
```
```python
target_times = np.linspace(begins[trial] + BIN_SIZE, ends[trial], N_BINS, dtype=np.float64)
values[trial] = interp1d(times, vals, kind="linear", fill_value="extrapolate")(target_times)
```

iii. As in 7-d: "Match repository endpoint interpolation and interval validity checks." Note this also means the camera is upsampled from 60 Hz (left) or 150 Hz (right) to the 50 Hz bin grid by linear interpolation, the same as the repo.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several layers, mostly "detect and drop, and record why":
- **Missing columns** in the trials table → `ValueError` naming them; the session is skipped.
- **NaN task events** → excluded by `_paper_trial_mask` (`notna().all(axis=1)` over the six required columns). NaN `stimOn_times` also produce `begins = NaN`, which makes `searchsorted` return equal begin/end indices, so the trial is marked not-good rather than crashing.
- **Wheel / camera windows not covered or non-finite** → per-trial `good` flag false, trial dropped (0.004 wheel-bad and 0.85 motion-bad trials per session on average).
- **Camera timestamp/frame-count mismatch** → repaired by trimming leading timestamps when there are too many, rejected when there are too few (the `_check_video_timestamps` port).
- **Length mismatch** between wheel position and timestamps, or spike times and spike clusters → `ValueError`, session skipped.
- **Missing camera entirely** → both views fail, `RuntimeError`, session skipped (14 sessions).
- **Fewer than 2 usable trials** → `ValueError('only N aligned valid trials')`, session skipped (1 session).
- **Unexpected `probabilityLeft`** → hard error rather than a silent mislabel.
- Every skipped session is written with its exception text to `/app/converted_data.failures.json`, and per-session bookkeeping (`source_trial_count`, `paper_eligible_trial_count`, `retained_trial_count`, `excluded_for_stream_alignment_or_nonfinite`, tertile edges, `n_neurons`) goes into `metadata['session_info']`.
- Three all-zero neural trials were detected by the validator and deliberately kept.
- Not handled: a probe listed in `bwm_release.csv` whose spike sorting is absent raises `FileNotFoundError` and kills the **whole session** rather than just that probe (no session hit this case in practice).

ii.
```python
    except Exception as exc:
        failures[spec.eid] = repr(exc)
        print(f"[{completed}/{len(specs)}] SKIP {spec.eid}: {exc}", file=sys.stderr, flush=True)
...
failure_path = args.output.with_suffix(".failures.json")
with open(failure_path, "w") as stream:
    json.dump(failures, stream, indent=2, sort_keys=True)
```
```python
keep = paper_mask & wheel_good & motion_good
keep_idx = np.flatnonzero(keep)
if len(keep_idx) < 2:
    raise ValueError(f"only {len(keep_idx)} aligned valid trials")
```
```python
"excluded_for_stream_alignment_or_nonfinite": int((paper_mask & ~(wheel_good & motion_good)).sum()),
```

iii. Step 84: "All 459 candidates have now been processed; 15 were excluded because whisker motion energy was absent or could not cover any eligible trial window." Step 97 on the zero-activity trials: "It reports only three all-zero neural trials late in one recording; I'm retaining them because the supplied reference binning code produces zero-filled windows and specifies no neural-coverage exclusion."

## 10-a. What are the most time-consuming steps of the code?

i. Two dominate, both I/O:
1. **Reading the spike sorting.** `spikes.times.npy` and `spikes.clusters.npy` are the largest inputs (165 MB + 83 MB on the one probe the AI measured at step 55), read for every one of the 699 insertions. They are memory-mapped and only the per-trial slices are materialised, which is the main saving the AI engineered, but the underlying page-ins still dominate per-session CPU time.
2. **Serialising the output.** Because no neuron QC is applied, the neural payload is ~106 GB. The code writes every session to a checkpoint pickle, reads all 444 checkpoints back in `_assemble`, writes the 100 GiB final pickle, then deletes the checkpoints — roughly 320 GB of disk traffic for a 106 GB result, plus holding all payloads in RAM at once during assembly (the trajectory reports ~108–113 GB resident).

Lesser costs: `BrainRegions()` is constructed once per session (it loads the Allen atlas tables), and `interpolate_position` resamples the whole session's wheel — up to 4.7 M samples at 1 kHz (step 45) — to 1 kHz before any trial slicing.

ii.
```python
times = np.load(times_path, mmap_mode="r")
clusters = np.load(clusters_path, mmap_mode="r")
```
```python
tmp_path = checkpoint_path.with_suffix(".tmp")
with open(tmp_path, "wb") as stream:
    pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
os.replace(tmp_path, checkpoint_path)
```
```python
    with open(path, "rb") as stream:
        payloads.append(pickle.load(stream))
...
with open(tmp_path, "wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 65: "I'm implementing the full converter with resumable per-session checkpoints because the final dense float32 neural payload is about 106 GB. " The mmap and the `searchsorted`-then-slice pattern are described in `_bin_probe`'s docstring as binning "without rescanning the full spike train", i.e. an explicit avoidance of the repo's `(times >= t_beg) & (times < t_end)` full-array scan per trial.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level per-item loops:
1. `_bin_probe`'s `for trial, (lo, hi) in enumerate(zip(start_idx, end_idx))` — could be one `np.bincount` over all trials by adding `trial * n_units * N_BINS` to the flat index.
2. `_interpolate_trials`'s `for trial, (beg_idx, end_idx) in ...` — could be a single `np.interp` over one concatenated query vector, since the traces are already uniformly sampled and monotone.
3. `_trial_number_in_block`'s `for trial in range(1, len(probability_left))` — a genuinely avoidable scalar Python loop; the run-length count is `trials.groupby((p != p.shift()).cumsum()).cumcount()` in one vectorised expression. The expert's code does exactly that; this is the one loop the AI added that the reference does not have.

None of the three matters in practice: they cost order 0.1 s per session against tens of seconds of spike-file I/O. The genuine parallelism limitation is elsewhere — `ThreadPoolExecutor` rather than processes means the pure-Python parts of these loops contend on the GIL across the 16 workers (the numpy and file-read portions do release it).

ii.
```python
for trial, (lo, hi) in enumerate(zip(start_idx, end_idx)):
```
```python
for trial, (beg_idx, end_idx) in enumerate(zip(idx_beg, idx_end)):
```
```python
    for trial in range(1, len(probability_left)):
        out[trial] = (out[trial - 1] + 1
                      if probability_left[trial] == probability_left[trial - 1]
                      else 0)
```
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
```

iii. Not discussed in the trajectory. The per-trial structure of loops 1 and 2 is inherited from the reference repo, which also loops per interval (there via `multiprocessing.Pool`); the AI's version replaces the pool with a plain loop and instead parallelises at the session level.

## 10-c. What processing does the code repeat multiple times?

i.
- **`BrainRegions()` is instantiated inside `_convert_session`**, i.e. once per session (444 times), re-reading and re-building the Allen atlas region tables each time. It is stateless here and could be a module-level singleton, as it is in the expert's code.
- **The checkpoint round-trip**: every session's arrays are pickled to disk and then immediately unpickled by `_assemble` in the same process run, so all 106 GB are serialised, written, read and re-serialised. Justified only as crash-resumption insurance.
- **Wheel and camera traces are interpolated for every trial in the session**, including the ~4 % later dropped for coverage and the ~35 % dropped by the paper mask, because `_load_wheel_speed` / `_load_whisker_energy` are called with the unfiltered `stimulus_times` before `keep` is known.
- **Repeated directory globbing**: `_required_file(session_path.glob("alf/**/..."))` walks the session tree separately for each of the ~4 + 4×n_probes datasets.

ii.
```python
    probe_bins: list[np.ndarray] = []
    probe_regions: list[np.ndarray] = []
    atlas = BrainRegions()
```
```python
wheel, wheel_good = _load_wheel_speed(spec.session_path, stimulus_times)
motion, motion_good, camera_view = _load_whisker_energy(spec.session_path, stimulus_times)
keep = paper_mask & wheel_good & motion_good
```

iii. Not discussed. The trace-before-mask ordering is in fact *required* by the design, since `wheel_good`/`motion_good` are themselves inputs to `keep`; only the interpolation (not the coverage test) is genuinely wasted work on trials the paper mask already rejected.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **The largest item is not processing but retention**: 599,865 neurons are binned and stored where the paper's own curation keeps ~75,708. That includes 12,827 units mapped to `void` (histologically outside the brain) and 85,656 mapped to `root` (no Beryl summary structure) — the reference repo's own `data_loader_utils.py` filters `['root', 'void']` out before decoding. Roughly 8× the neural volume is carried through binning, pickling and decoder training for no downstream use.
- **Dtype choices inflate the file**: spike counts (small non-negative integers, max a few tens per 20 ms bin) are stored as `float32`, and the four categorical outputs, which take values in {0,1,2}, are stored as `int64` — 8 bytes per label where `int8` would do.
- **Wheel/whisker interpolation for trials the paper mask already rejects** (see 10-c), ~35 % of trials.
- **`_load_trials` requires `feedbackType` to be present** and `_paper_trial_mask` tests it for NaN, but the variable is otherwise unused (it is only part of the repo's NaN-exclusion list), and `goCue_times`/`feedback_times` are read only to form the ≤10 s duration test.
- **`clusters.channels` and `channels.brainLocationIds_ccf_2017` are loaded in full** for every probe although only the rows in `used` are needed.
- **The `--keep-session-cache` default deletes the checkpoints** after assembly, so the 106 GB of intermediate writes is pure overhead in the successful path.

ii.
```python
binned = np.zeros((len(interval_begins), len(used), N_BINS), dtype=np.float32)
```
```python
outputs = np.empty((len(keep_idx), 4, N_BINS), dtype=np.int64)
```
```python
"neuron_filter": "all Kilosort clusters, matching the methods-paper decoder pipeline",
```
```python
    if not args.keep_session_cache:
        for path in checkpoint_paths.values():
            path.unlink(missing_ok=True)
```

iii. Not discussed as waste. The AI knew the size in advance (step 47: "estimated float32 GB all 109.85") and treated it as a constraint to engineer around rather than to reduce, choosing checkpointing (step 65) and reporting the finished artifact at step 265 as "100 GiB". It never revisited whether retaining all clusters was the right call once the cost was known.
