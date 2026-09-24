# Decisions

> Note on provenance: the decisions below are reconstructed from `/app/convert_data.py`,
> `/app/CONVERSION_NOTES.md`, `/app/verification_full_out.txt` and
> `/logs/agent/trajectory.json`. One thing the trajectory makes clear and that matters for
> reading the rest of this document: the Decoder Task section the agent actually received
> (trajectory step 3) specified **two** classes for every output — `Lick direction (left = 0,
> right = 1)`, `Outcome (incorrect = 0, correct = 1)`, and movement variables "discretized
> into two bins" — with no `none` / `ignore` / `not visible` classes. The judge-side
> `instruction_reference.md` and the human reference solution specify **three** classes for
> five of the six outputs. Wherever that difference drives a divergence it is called out
> explicitly.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not glob the data folders. It reads the authors' own MATLAB session-loader
scripts in `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`, parses each
`meta(end).date = '...'` / `meta(end).probe = ...` / `datapth = fullfile(...)` triple out of
them, and strips `%` comments first so sessions the authors commented out are not picked up.
Ten loader files are assigned to `Ephys_Behavior` (fixed delay) and four to
`RandomizedDelay_Ephys_Behavior` (randomized delay), producing 25 + 19 = 44 `SessionSpec`
records with the probe(s) each session uses. Each session's `data_structure_<anm>_<date>.mat`
is then read once with `pymatreader.read_mat`, which transparently handles both MATLAB v7.3
(HDF5) and older v5 files. Motion energy is read from the companion
`motionEnergy_<anm>_<date>.mat`.

ii.
```python
FIXED_DELAY_LOADERS = [
    "loadJEB6_ALMVideo.m", "loadJEB7_ALMVideo.m", "loadEKH1_ALMVideo.m", ...
]
RANDOMIZED_DELAY_LOADERS = [
    "loadJEB11_ALMVideo.m", "loadJEB12_ALMVideo.m", "loadJEB23_ALMVideo.m", "loadJEB24_ALMVideo.m",
]

def parse_loader_file(loader_name: str, folder: str, task: str) -> List[SessionSpec]:
    date_re = re.compile(r"meta\(end\)\.date = '([^']+)'")
    probe_re = re.compile(r"meta\(end\)\.probe = (\[[^\]]+\]|[0-9]+)")
    for raw_line in path.read_text().splitlines():
        line = strip_comment(raw_line).strip()
        ...
        if "datapth = fullfile" in line and current_date is not None and current_probe is not None:
            sessions.append(SessionSpec(subject=subject, date=current_date,
                                        probe=current_probe, folder=folder, task=task))
```

```python
obj = read_mat(spec.data_path)["obj"]
```

iii. From CONVERSION_NOTES Step 4/5: the raw folders contain more files than the paper
analyzed (22 randomized-delay files vs 19 analyzed sessions; two `JEB24` files have no `clu`
field at all), so "the analyzed randomized-delay set is the 19 sessions explicitly selected by
the loader files, not every raw file in the folder." Using `pymatreader` was chosen because
"most `data_structure_*.mat` files are MATLAB v7.3/HDF5; at least the older `EKH1` and `EKH3`
sessions are non-HDF5 MATLAB files."

## 1-b. How are the data split into subjects?

i. The subject is the animal id embedded in the loader filename (`load([A-Z0-9]+)_ALMVideo.m`)
and carried on every `SessionSpec`. At assembly time `subjects` is built in order of first
appearance across sessions and `subject_idx` is each session's index into that list. Result:
14 subjects over 44 sessions (JEB6 1, JEB7 2, EKH1 1, EKH3 1, JGR2 2, JGR3 1, JEB13 5, JEB14 4,
JEB15 4, JEB19 4, JEB11 2, JEB12 2, JEB23 7, JEB24 8) — the same per-subject counts as the
human reference.

ii.
```python
subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
subject = subject_match.group(1)
```

```python
if spec.subject not in subject_order:
    subject_order.append(spec.subject)
subject_idx.append(subject_order.index(spec.subject))
```

iii. Not separately argued in CONVERSION_NOTES beyond Step 4, which notes that `obj.meta.anm`
is unreliable and that the loader files are the authoritative record of which animal each
session belongs to. Step 9 records the paper-vs-code mouse-count discrepancy (paper text
implies 13, code/data give 14) and resolves it in favour of the code/data.

## 1-c. How are the data split into sessions?

i. One session = one `SessionSpec` = one `data_structure_*.mat` file = one element of `neural`,
`input`, `output`, and `brain_region_idx`. The folder is fixed by which loader group the
session came from, so fixed-delay and randomized-delay sessions are pooled into one uniform
44-session dataset rather than kept as two datasets. Sessions with fewer than 2 valid trials or
fewer than 10 surviving units would be skipped; in the full run none were.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str; date: str; probe: Tuple[int, ...]; folder: str; task: str

    @property
    def data_path(self) -> Path:
        return DATA_ROOT / self.folder / f"data_structure_{self.subject}_{self.date}.mat"
```

```python
for spec in session_specs:
    result = process_session(spec, show_processing=do_plot)
    if result is None:
        continue
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "Include only neural sessions represented in the
reference ephys loaders. Fixed-delay set: all 25 sessions in `Ephys_Behavior`. Randomized-delay
set: the 19-session subset encoded by `loadJEB11/12/23/24_ALMVideo`. Behavior-only inhibition
sessions are excluded because they have no neural activity."

## 1-d. How are the data split into trials?

i. A trial is one entry of the per-trial fields of `obj.bp`, with one go cue per trial in
`obj.bp.ev.goCue`. The trial count comes from `obj.bp.Ntrials`. Spikes carry their trial number
in `clu.trial` (1-based) and their within-trial time in `clu.trialtm`; camera frames are stored
per trial in `obj.traj[view][field][trial]`; motion energy is one trace per trial. So no trial
boundary is ever reconstructed. Unlike the human reference, the agent does not defensively
truncate `bp` fields to `Ntrials` — it reads them at full length with `to_vector` and indexes
them by the surviving trial indices.

ii.
```python
n_trials = int(round(float(obj["bp"]["Ntrials"])))
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
...
def get_view_trial(view: dict, trial_index: int) -> dict:
    return {key: view[key][trial_index] for key in view.keys()}
```

```python
R = to_vector(bp["R"], float).astype(int)
autowater = to_vector(bp["autowater"], float).astype(int)
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)
```

iii. CONVERSION_NOTES Step 2: "`obj.bp`: trial labels (`L`, `R`, `hit`, `miss`, `no`,
`autowater`, `early`, `protocol`, `stim`)"; "`obj.traj`: two camera views; each view stores
per-trial lists of `frameTimes`, `ts`, `featNames`"; "`me.data` is one trace per trial". The
trial table is taken as the definition of a trial.

## 1-e. How are trials filtered based on quality controls?

i. Four filters. (1) Photostimulation trials (`bp.stim.enable`) are dropped. (2) Early-lick
trials (`bp.early`) are dropped. (3) **Ignore / no-response trials (`bp.no`) are dropped.**
(4) After neuron selection, trials occurring after the last trial in which any surviving unit
fired are dropped, because in `JEB24_2023-10-23` and `JEB24_2023-11-03` the behaviour continues
past the end of the recording; the neuron selection is then re-run on the trimmed trial set.
Sessions left with fewer than 2 trials are skipped. This keeps 11,955 trials. The human
reference applies (1), (2) and (4) only, and keeps 13,762 trials — the ~1,800-trial gap is
almost entirely the dropped ignore trials.

ii.
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid
```

```python
max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
if max_trial_with_spikes > 0:
    neural_coverage_mask = (np.arange(trial_mask.size, dtype=np.int32) + 1) <= max_trial_with_spikes
    refined_trial_mask = trial_mask & neural_coverage_mask
    if not np.array_equal(refined_trial_mask, trial_mask):
        log(f"{spec.session_id}: dropping {dropped} behavior-valid trials after raw trial "
            f"{max_trial_with_spikes} due to missing neural coverage")
        trial_mask = refined_trial_mask
        keep_trials = np.flatnonzero(trial_mask)
        selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "Use only control/non-stimulation trials with
valid behavioral labels. Exclude `stim.enable`, `early`, and `no` trials to match the reference
analyses and keep output definitions unambiguous." Step 3 backs this from the paper: "Early-lick
trials are omitted from analyses. Ignore / no-response trials are omitted from behavioral
analyses." Step 9/10 documents the coverage trim: "`JEB24_2023-10-23` and `JEB24_2023-11-03`
contained behavior-valid trials after the last trial with neural spikes; those late trials were
excluded to avoid invalid all-zero neural inputs."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu[probe-1]`, the spike-sorted clusters of the probe(s) the loader file names for that
session, using the per-cluster fields `trial` (1-based trial number of each spike), `trialtm`
(spike time relative to that trial's start), and `quality` (manual curation label). The
alignment event `obj.bp.ev.goCue` is the other input. Two-probe sessions (e.g. `JEB15` with
`meta.probe = [1 2]`) have both probes' units concatenated into one population, and each
neuron's region label is read from `obj.ex.probe[i].loc`.

ii.
```python
def binned_neuron_trials(probe, neuron_index, align_times, keep_trials_0based, edges):
    trials = to_vector(probe["trial"][neuron_index], int)
    trialtm = to_vector(probe["trialtm"][neuron_index], float)
    ...
    aligned = trialtm[mask] - align_times[trials[mask] - 1]
```

```python
for probe_num in spec.probe:
    probe = probes[probe_num - 1]
    qualities = [normalize_string(q) for q in probe["quality"]]
    quality_mask = quality_keep_mask(qualities)
    loc = get_probe_loc(obj, probe_zero)
```

iii. CONVERSION_NOTES Step 5 variable mapping: "`obj.clu{probe}.trialtm`, `obj.clu{probe}.trial`,
`obj.bp.ev.goCue` → `neural` … Use the probe(s) selected by the reference loader files;
concatenate probes within a session if code does so." Step 1 traces this to
`loadSessionData -> processData -> findClusters -> alignSpikes -> getSeq`.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 5 ms bins over `[-2.5, +2.5)` s from the go cue with
`np.add.at` on a `(1000, n_kept_trials)` grid, divided by the bin width to give Hz, then
smoothed along time with a **causal** Gaussian. The smoother is a direct Python port of the
authors' `code/utils/mySmooth.m`: `gausswin(15)` with `alpha = 2.5`, the first
`floor(15/2) = 7` taps zeroed to make it causal, normalised to unit sum, applied with
`conv(..., 'same')`, and with `'reflect'` boundary handling that prepends the first 15 samples
and trims them afterwards. No normalisation, baseline subtraction or z-scoring. Stored as
`float32` Hz. (The human reference instead uses a *symmetric* `gaussian_filter1d` with
σ = 14 ms, which approximates the kernel width but not the causality.)

ii.
```python
def gaussian_window(n: int, alpha: float = 2.5) -> np.ndarray:
    m = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    sigma = (n - 1) / (2.0 * alpha)
    return np.exp(-0.5 * (m / sigma) ** 2)

def causal_gaussian_smooth(x, n, bctype=BCTYPE):
    if bctype == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0); trim = n
    ...
    kern = gaussian_window(n)
    kern[: n // 2] = 0.0          # causal, as in mySmooth.m
    kern /= kern.sum()
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    out = out[trim:, :]
```

```python
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts = np.zeros((edges.size - 1, keep_trials_0based.size), dtype=np.float32)
np.add.at(counts, (bin_index[valid], keep_index[valid]), 1.0)
rates = counts / DT
return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
```

iii. CONVERSION_NOTES Step 1: "`getSeq` bins spikes on `[tmin, tmax)` using `histc`, then
smooths counts converted to Hz with `mySmooth`." Step 10, Check 3: "binning/smoothing logic
matches `getSeq` with causal Gaussian smoothing and `reflect` boundary handling." The agent read
`mySmooth.m` directly (trajectory step 188) and reproduced its `kern(1:floor(numel(kern)/2)) = 0;
%causal` line.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters and one session-level filter. (1) The manual curation label
`clu.quality` is lower-cased, stripped, and rejected if it is one of `garbage`, `gabrga`,
`noisy`, `real?` — exactly the drop list in the authors' `findClusters.m` under the `'all'`
option, so multi-units and `poor` units are kept. (2) A unit is kept only if its mean firing
rate over the analysis window on the surviving trials exceeds 1 Hz, computed as raw spike count
in `[-2.5, 2.5)` divided by `n_trials × 5 s`. (3) A session is dropped if fewer than 10 units
survive. This yields 2,455 units, 17–142 per session (mean 55.8). The human reference
additionally drops `poor` units and keeps 1,954.

ii.
```python
def quality_keep_mask(qualities: Sequence[str]) -> np.ndarray:
    cleaned = [normalize_string(q).lower() for q in qualities]
    bad = {"garbage", "gabrga", "noisy", "real?"}
    return np.array([q not in bad for q in cleaned], dtype=bool)
```

```python
LOW_FR = 1.0
MIN_UNITS_PER_SESSION = 10
...
for neuron_index in np.flatnonzero(quality_mask):
    mean_fr = neuron_mean_fr(probe, int(neuron_index), align_times, trial_mask)
    if mean_fr > LOW_FR:
        kept_indices.append(int(neuron_index))
...
if len(selected_neurons) < MIN_UNITS_PER_SESSION:
    log(f"Skipping {spec.session_id}: only {len(selected_neurons)} units after filtering")
    return None
```

iii. CONVERSION_NOTES Step 1: "`findClusters` … `'all'` excludes `garbage`, `gabrga`, `noisy`,
and `real?`"; "`removeLowFRClusters` … typically `lowFR = 1` Hz". Step 3, from the paper:
"Sessions were included only if they had at least 10 units"; "For most analyses, all units with
firing rate >1 Hz were included." Step 5, Key Decision 3 restates all three rules.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By a single subtraction. `clu.trialtm` is already on the behaviour clock and already relative
to its own trial's start, and `bp.ev.goCue` is on the same clock, so each spike's time from the
go cue is `trialtm − goCue[trial]`. Spikes are then floored into the fixed 5 ms grid and those
outside `[-2.5, 2.5)` are discarded by the `valid` mask. No interpolation or per-trial offset is
applied to the neural stream (unlike the camera streams, which need the video-clock correction).

ii.
```python
keep_index = trial_to_keep[trials]
mask = keep_index >= 0
aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
valid = (bin_index >= 0) & (bin_index < edges.size - 1)
```

iii. CONVERSION_NOTES Step 1: "`alignSpikes` — Aligns per-spike trial times to the chosen event
(`goCue`, …)"; "Reference analyses commonly use `params.alignEvent = 'goCue'`, which matches the
decoder task requirement." Step 10, Check 3: "go-cue alignment matches `alignSpikes`." Step 10
also reports a raw-data spot check: an independently rebuilt aligned/binned spike train for
`JEB6_2021-04-18` neuron 0, first kept trial, matched the converted trace with max abs diff 0.0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms, uniform: `DT = 0.005`, `TMIN = -2.5`, `TMAX = 2.5`, giving exactly 1000 bins for every
trial of every session. The grid is built once per session by `make_time_edges()` /
`make_time_centers()` and is identical across sessions, and it is the *only* time axis in the
converted file — spikes are binned directly onto it and the camera streams are interpolated
onto it, so there is no rebinning of an intermediate representation. `metadata['time_bin_size']`
is 5.0 (ms).

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5

def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)

def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0
```

iii. CONVERSION_NOTES Step 4 resolves an explicit discrepancy: "Code examples use both 10 ms
(`dt=1/100`) and 5 ms (`dt=1/200`) depending on figure/script … Paper's decoding methods
explicitly state 5 ms bins → Resolved: for decoder-matched conversion, prioritize 5 ms bins over
generic 10 ms examples." Step 5, Key Decision 4: "Use go-cue alignment and a common 5 ms bin
width over `[-2.5 s, +2.5 s)`."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. The input is defined by the conversion: it is the vector of bin centres of
the analysis window, `-2.4975 … +2.4975` s, identical for every trial and every session. The
only raw quantity implicitly involved is `bp.ev.goCue`, which defines where t = 0 sits for the
neural and camera streams.

ii.
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
...
input_trials.append(time_centers[None, :].astype(np.float32))
...
"input_names": ["time_from_go_cue"],
```

iii. CONVERSION_NOTES Step 5 variable mapping: "Common aligned time axis → `input[0]` —
Continuous time-from-go-cue series, same for every trial, shape `(1, time)` … Input name:
`time_from_go_cue`." Step 10 verification: "independently rebuilt the 1000-bin time axis
`(-2.4975 … 2.4975)` and matched `input[0]` … max abs diff `3.58e-07`."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the grid: `np.arange(-2.5, 2.5 + DT/2, DT)` gives the 1001 edges,
and the input is `edges[:-1] + DT/2`, cast to `float32` and reshaped to `(1, 1000)`. The same
array object contents are emitted for every trial.

ii.
```python
def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)

def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0
```

iii. N/A — the agent treats this as a definitional quantity, not derived data. Verified range in
`verification_full_out.txt`: `time_from_go_cue: [-2.5, 2.5]`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `time_edges` is passed to `binned_neuron_trials` as the
spike-count edges, and `time_centers = time_edges[:-1] + DT/2` is the input, so bin *k* of the
input and bin *k* of the neural matrix are the same 5 ms interval relative to the same go cue.
The camera outputs are interpolated onto `time_centers` as well, so all four streams share one
axis with no relative offset (`ADVANCE_MOVEMENT = 0.0`).

ii.
```python
rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
...
input_trials.append(time_centers[None, :].astype(np.float32))
...
taxis = time_centers + ADVANCE_MOVEMENT      # ADVANCE_MOVEMENT = 0.0
xy_aligned = interp(taxis)
```

iii. CONVERSION_NOTES Step 10, Check 3: "Input construction: decoder input is the common aligned
time axis, derived from the same bin centers as the neural data."

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. A single field, `obj.bp.R` (1 on right-instructed trials, 0 on left-instructed trials). The
outcome flags are **not** used to derive the licked side; they are only used for a consistency
assertion. Because `bp.R` is the *instructed* direction, on error (`miss`) trials it is the
opposite of the side the animal actually licked. The human reference derives the licked side
from `R` combined with `hit`/`miss`.

ii.
```python
R = to_vector(bp["R"], float).astype(int)
...
lick_direction = R[keep_trials]
```

iii. CONVERSION_NOTES Step 5 variable mapping: "`obj.bp.R` / `obj.bp.L` → `output[0]` —
Per-trial categorical label: left=`0`, right=`1` … Exclude early and no-response trials so
choice/outcome are well-defined." Reference function cited: `findTrials`. The notes do not
distinguish instructed direction from executed lick direction anywhere.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A straight subset: `lick_direction = R[keep_trials]`, then tiled across all 1000 time bins so
it is a constant time series per trial. Two classes, `["left", "right"]`. There is no `none`
class, because no-response trials were removed at the trial-filtering stage. Full-dataset
distribution: left 0.498 / right 0.502.

ii.
```python
lick_direction = R[keep_trials]
...
output_trials.append(np.vstack([
    np.full(time_centers.size, lick_direction[tr], dtype=np.int64),
    ...
]))
...
"output_values": [["left", "right"], ...]
```

iii. CONVERSION_NOTES Step 5, Key Decision 2 and the Step 5 mapping table: ignore trials are
removed "to keep output definitions unambiguous", and the label is taken directly from the trial
type. The Decoder Task the agent was given specified `Lick direction (left = 0, right = 1,
per-trial)` with no third class. The agent's Step 10 sanity check compared the converted values
against raw `bp.R` — i.e. it verified the implementation, not the semantics.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial field, `obj.bp.autowater`: 1 where water was delivered at a random port
without any cue (water-cued context), 0 otherwise (delayed-response context).

ii.
```python
autowater = to_vector(bp["autowater"], float).astype(int)
```

iii. CONVERSION_NOTES Step 1: "`autowater` is the code's proxy for behavioral context: delayed
response / 2AFC (`autowater == 0`) versus water-cued (`autowater == 1`)." Step 5 mapping:
"`obj.bp.autowater` → `output[1]`".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A polarity flip and a subset: `context = 1 - autowater[keep_trials]`, so WC = 0 and DR = 1,
matching the codes the Decoder Task asks for. Tiled across the 1000 bins as a per-trial constant.
Full-dataset distribution: WC 0.084 / DR 0.916, reflecting that only the two-context sessions
contain WC blocks (14 of 44 sessions are pure DR).

ii.
```python
context = 1 - autowater[keep_trials]
...
"output_values": [..., ["WC", "DR"], ...]
```

iii. CONVERSION_NOTES Step 5, Key Decision 5: "Represent behavioral context as WC=`0`, DR=`1`,
even though raw `autowater` uses the opposite polarity." Step 9 consistency check calls the
resulting 0.084/0.916 split "Plausible / expected DR dominance".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` and `obj.bp.miss`. `bp.no` is read too, but only in `build_trial_mask` to remove
ignore trials; it never becomes an outcome class. The agent asserts that on the surviving trials
`hit + miss == 1` exactly, i.e. the two flags are mutually exclusive and exhaustive there.

ii.
```python
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)
...
outcome = hit[keep_trials]
if not np.all((outcome == 0) | (outcome == 1)):
    raise ValueError(f"{spec.session_id}: outcome contains values outside hit/miss after filtering")
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(f"{spec.session_id}: hit/miss are not mutually exclusive on kept trials")
```

iii. CONVERSION_NOTES Step 5 mapping: "`obj.bp.hit` / `obj.bp.miss` → `output[2]` — Per-trial
categorical label: miss=`0`, hit=`1` … Ignore/no-response trials excluded rather than forced into
incorrect."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `outcome = hit[keep_trials]`, i.e. 1 on correct trials and 0 on error trials, tiled across the
1000 bins. Two classes, `["incorrect", "correct"]`. The hit/miss exclusivity assertion above runs
on every session. Full-dataset distribution: incorrect 0.138 / correct 0.862, consistent with the
paper's >70 % expert criterion. There is no `ignore` class; those trials are absent from the
dataset entirely.

ii.
```python
outcome = hit[keep_trials]
...
np.full(time_centers.size, outcome[tr], dtype=np.int64),
...
"output_values": [..., ["incorrect", "correct"], ...]
```

iii. CONVERSION_NOTES Step 5, Key Decision 6: "Represent only miss versus hit. Ignore trials are
removed instead of being merged into incorrect." Step 3 supports removal from the paper: "Ignore
/ no-response trials are omitted from behavioral analyses." Step 9: the resulting 0.138/0.862
split is "consistent with expert-performance regime".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, using **seven** tongue keypoints across **both**
cameras: side view (`obj.traj[0]`) `tongue`, `left_tongue`, `right_tongue`; bottom view
(`obj.traj[1]`) `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`. For each
trial it reads `ts` (frames × [x, y, likelihood] × features), `featNames`, `frameTimes`, and
`NdroppedFrames`. Alignment additionally needs `bp.ev.goCue`, `bp.ev.bitStart`,
`sglx.bitcode.bitstart` and `sglx.fs`. (The human reference uses only `tongue` and `top_tongue`.)

ii.
```python
TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}
...
tongue_speed_all, tongue_feats = aggregate_speed(obj, TONGUE_FEATURES, time_centers, align_times, vidshift)
```

```python
view = obj["traj"][view_index_one_based - 1]
feat_index = find_feature_index(view, feat_name)
ts = np.asarray(trial_view["ts"], dtype=np.float32)
xy = ts[:, :2, feat_index]
```

iii. CONVERSION_NOTES Step 5 mapping: "DLC trajectories for tongue-related features →
`output[3]` … Because the decoder spec needs one tongue-velocity output, aggregate across tongue
velocity channels after reference processing." Step 3: "Tongue, jaw, nose tracked in both views;
paws tracked from bottom view only."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per feature: the raw `(x, y)` frame series is linearly interpolated from corrected frame times
onto the 1000-bin grid with `interp1d(..., fill_value=np.nan)` (so nothing outside the frame range
is invented); `np.gradient` is taken along the 5 ms grid index, giving pixels per bin; for tongue
features, and only tongue features, the resulting NaNs are replaced by **zero** velocity and no
baseline subtraction or nearest-fill is done. Speed is `hypot(xvel, yvel)`. The seven per-feature
speeds are then averaged bin-by-bin over the components that are finite, with any remaining NaN
set to 0. **No per-camera scale normalisation is applied**, so side-view and bottom-view pixel
scales are averaged raw (the human reference divides each view by its own 90th percentile before
averaging). No explicit likelihood cut is applied either — the agent relies on the authors having
already set `x`/`y` to NaN below likelihood 0.9.

ii.
```python
def feature_velocity(xpos, ypos, feat_name):
    for trial in range(xpos.shape[1]):
        tsinterp = np.column_stack([xpos[:, trial], ypos[:, trial]]).astype(np.float32)
        base = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
        xv = np.gradient(tsinterp[:, 0]).astype(np.float32)
        yv = np.gradient(tsinterp[:, 1]).astype(np.float32)
        if "tongue" not in feat_name:
            xv = fill_nearest_1d(xv - base[0]); yv = fill_nearest_1d(yv - base[1])
        else:
            xv = np.nan_to_num(xv, nan=0.0); yv = np.nan_to_num(yv, nan=0.0)
```

```python
speed = np.sqrt(np.square(xvel) + np.square(yvel))
speed_components.append(speed)
...
agg = np.divide(summed, np.maximum(count, 1), dtype=np.float32)
agg[count == 0] = np.nan
agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 1: "`findVelocity` — Computes per-trial velocity from interpolated
trajectories; non-tongue features are baseline-subtracted, tongue NaNs become zero velocity" —
the agent is deliberately reproducing the authors' `findVelocity.m` convention. Step 5, Key
Decision 7: "collapse the multi-feature representation to scalar tongue-speed and paw-speed
traces using aggregate speed across the relevant tracked points. This is a required deviation
from the reference feature set because the decoder task asks for exactly one tongue-velocity
output."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Nominally at the per-session 50th percentile. In practice the threshold is computed by
`percentile_threshold(...)` and then **never used**: `discretize_trace` ignores its `threshold`
argument and instead `argsort`s the flattened `(1000, n_trials)` session array with a stable
mergesort and assigns class 1 to the upper half of the ranking, forcing an exact 50/50 split.
The agent switched to this after observing (trajectory step 237) that "tongue bins collapsed to a
single class because many aligned tongue samples sit exactly at the median".

The collapse happens because invisible tongue frames were converted to *exactly zero* velocity in
7-b, so the large majority of values are ties at 0. Under a stable sort, ties are broken by flat
array index, and the array is laid out time-major — so the split among the zeros is made by time
bin. The delivered data confirms the consequence: in `sample_data.pkl` session 0,
`tongue_velocity_bin` is 1 in 99.9 % of bins after t ≈ +0.042 s and in only 1.7 % of bins before
it, i.e. the output encodes *time from the go cue* rather than tongue movement. Since time from
the go cue is the decoder's only input, the reported 0.854 "tongue velocity" accuracy is an
artifact of this. There is no `not visible` class; the human reference assigns one to ~88 % of
bins.

ii.
```python
def percentile_threshold(traces: np.ndarray, percentile: float = 50.0) -> float:
    flat = traces.reshape(-1)
    flat = flat[np.isfinite(flat)]
    return float(np.percentile(flat, percentile))

def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)      # computed
tongue_bin = discretize_trace(tongue_speed, tongue_thr)    # argument ignored
```

iii. Trajectory step 237: "tongue bins collapsed to a single class because many aligned tongue
samples sit exactly at the median. I'm switching the discretization to a rank-based median split
with the 50th-percentile value still recorded as metadata, which preserves the intended
sessionwise median threshold while handling ties sensibly." CONVERSION_NOTES Step 12 records the
50/50 distribution as intentional — "the movement bins are exactly 50/50 by construction" — and
treats the 0.854 accuracy as a genuine result ("Strongest movement decoder").

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a session-constant offset is computed once from
the bitcode pulse recorded on both streams — `mode(sglx.bitcode.bitstart / sglx.fs) −
mode(bp.ev.bitStart)` — which is the authors' `findVideoOffset.m`. Each trial's frame times become
`frameTimes − vidshift − goCue[trial]`, and the `(x, y)` series is then linearly interpolated onto
the shared `time_centers` grid, so the tongue output occupies exactly the same 5 ms bins as the
spikes. `ADVANCE_MOVEMENT = 0.0`, so no extra lead/lag is applied. If the bitcode fields are
missing or non-finite, the offset silently falls back to a hard-coded 0.5 s.

ii.
```python
def compute_video_offset(obj: dict) -> float:
    try:
        bit_start = robust_mode(obj["bp"]["ev"]["bitStart"])
        vid_file_offset = robust_mode(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
        if np.isfinite(bit_start) and np.isfinite(vid_file_offset):
            return float(vid_file_offset - bit_start)
    except Exception:
        pass
    return 0.5
```

```python
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, axis=0, kind="linear",
                  bounds_error=False, fill_value=np.nan, assume_sorted=True)
xy_aligned = interp(taxis)
```

iii. CONVERSION_NOTES Step 1: "`findVideoOffset` — Computes offset between neural and video file
starts for frame-time alignment"; "Video/behavior streams are interpolated to the neural/session
time axis rather than used at native frame rate"; "Motion energy alignment is explicit:
`interp1(frameTimes - video_offset - align_event_time, me.data{trial}, obj.time +
advance_movement)`". The agent reproduced that `interp1` call for the DLC streams as well.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera only (`obj.traj[1]`), using **both** tracked paws, `top_paw` and
`bottom_paw`. Same per-trial fields as the tongue (`ts`, `featNames`, `frameTimes`,
`NdroppedFrames`) and the same clock inputs. The human reference deliberately uses `top_paw`
alone.

ii.
```python
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}
...
paw_speed_all, paw_feats = aggregate_speed(obj, PAW_FEATURES, time_centers, align_times, vidshift)
```

iii. CONVERSION_NOTES Step 3: "paws tracked from bottom view only." Step 5 mapping: "DLC
trajectories for paw-related features → `output[4]` … Paws are tracked in the bottom view only."
Key Decision 7 covers collapsing the multi-feature representation into one scalar. The agent does
not discuss `bottom_paw`'s reliability through the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as the tongue but on the non-tongue branch of `findVelocity`: interpolate `(x, y)`
onto the 5 ms grid, **nearest-fill any NaN positions across the whole window**, take `np.gradient`,
subtract the per-trial median inter-sample displacement as a baseline, nearest-fill the velocity
again, then `speed = hypot(xvel, yvel)`, and average the two paws over the finite components.
Values stay in pixels per 5 ms bin; no cross-view normalisation is needed since both features come
from one camera. The nearest-fill means bins where the paw was untracked receive the velocity of
the nearest tracked bin rather than being marked missing.

ii.
```python
if "tongue" not in feat_name:
    xpos[:, trial] = fill_nearest_1d(xpos[:, trial])
    ypos[:, trial] = fill_nearest_1d(ypos[:, trial])
```

```python
base = np.nanmedian(diffs, axis=0)
xv = np.gradient(tsinterp[:, 0]).astype(np.float32)
yv = np.gradient(tsinterp[:, 1]).astype(np.float32)
if "tongue" not in feat_name:
    xv = xv - base[0]; yv = yv - base[1]
    xv = fill_nearest_1d(xv); yv = fill_nearest_1d(yv)
```

iii. CONVERSION_NOTES Step 1: "`findVelocity` — … non-tongue features are baseline-subtracted,
tongue NaNs become zero velocity." Step 3, from the paper's methods: "Missing values are filled
with nearest values for all features except the tongue." Step 5, Key Decision 7 for the
aggregation to a single scalar.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `discretize_trace` rank split as the tongue: the per-session 50th percentile is
computed and discarded, and the flattened session array is split exactly in half by rank. Paw
speed is genuinely continuous (the nearest-fill removed the NaN ties), so here the rank split is
numerically almost identical to a true median threshold and the result is not degenerate — the
class fraction varies over time in the delivered data. Two classes,
`["below_session_median", "at_or_above_session_median"]`; there is no `not visible` class, and
untracked bins are nearest-filled into whichever class their neighbour falls in. The human
reference assigns class 2 to ~19 % of paw bins.

ii.
```python
paw_thr = percentile_threshold(paw_speed, 50.0)
paw_bin = discretize_trace(paw_speed, paw_thr)
```

```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

iii. Same as 7-c: trajectory step 237 introduced the rank split to handle ties; CONVERSION_NOTES
Step 5, Key Decision 8 and Step 12 treat exactly-50/50 movement classes as intended behaviour.
The Decoder Task the agent received asked for two bins only.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the same session-constant `vidshift` is subtracted from the bottom
camera's `frameTimes`, then that trial's `goCue`, and the positions are interpolated onto the same
`time_centers`. Because `aligned_position` looks up `obj["traj"][view - 1]` per feature, the paw is
timed by its own camera's frame times rather than by the side camera's.

ii.
```python
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, axis=0, kind="linear",
                  bounds_error=False, fill_value=np.nan, assume_sorted=True)
xy_aligned = interp(taxis)
```

```python
def get_frame_times(trial_view: dict, n_frames: int) -> np.ndarray:
    frame_times = trial_view.get("frameTimes")
    if frame_times is None:
        return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
    arr = to_vector(frame_times, float)
    if arr.size != n_frames or not np.isfinite(arr).any():
        return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
    return arr.astype(np.float32)
```

iii. Same justification as 7-d — one video offset, one shared grid, `advance_movement = 0`. The
`get_frame_times` fallback synthesises a nominal 400 Hz frame clock when `frameTimes` is absent,
empty or the wrong length, which CONVERSION_NOTES Step 3 motivates with "High-speed video was
captured (400-Hz frame rate)".

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` beside the session file, read with
`read_mat(...)["me"]` and unwrapped through however many nested `data` levels the file has (some
`JEB15` files wrap twice). If that file is missing, it falls back to the embedded `obj.me`. One
trace per trial, one value per side-camera frame. `moveThresh` is read out of the file but never
used. Frame times come from the side camera (`obj.traj[0]`).

ii.
```python
def load_motion_energy_raw(obj: dict, spec: SessionSpec) -> Optional[dict]:
    if spec.motion_energy_path.exists():
        raw_me = read_mat(spec.motion_energy_path).get("me")
        ...
        return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": move_thresh}
    if "me" in obj:
        ...
```

```python
def unwrap_embedded_motion_energy(raw_me) -> List[np.ndarray]:
    data = raw_me
    seen = set()
    while isinstance(data, dict) and "data" in data and id(data) not in seen:
        seen.add(id(data)); data = data["data"]
```

iii. CONVERSION_NOTES Step 2: "External motion-energy files: `me.data` is one trace per trial;
`moveThresh` is a per-session threshold." Step 9/10: "`JEB15` motion-energy files required
recursive unwrapping of nested `me.data` structs" — found and fixed during critical review. Step
1 cites `loadMotionEnergy.m` as the reference implementation.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment. The value is already one scalar per frame (the paper's per-pixel
temporal median difference reduced to the 99th percentile across pixels), so the agent only
interpolates it onto the 5 ms grid, nearest-fills the resulting NaNs, and replaces any remaining
non-finite value with 0. The paper's manually chosen per-session `moveThresh` is deliberately not
applied.

ii.
```python
aligned[:, trial] = interp(taxis).astype(np.float32)
aligned[:, trial] = fill_nearest_1d(aligned[:, trial])
...
return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping: "Motion energy traces → `output[5]` — Align/interpolate to
neural time axis using reference logic, then bin by per-session median … The paper's manual move
threshold is not used because the decoder task explicitly asks for 50th-percentile binning."
Step 1: "`loadMotionEnergy` … interpolates it onto the neural time axis aligned to
`params.alignEvent`, fills edge NaNs, and thresholds into move/non-move."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `discretize_trace` rank split: the per-session 50th percentile is computed, discarded, and
the flattened session array is split exactly in half by rank. Motion energy is continuous, so the
result is close to a true median threshold and is not degenerate (in the sample session the class
fraction varies from 0.0 in the leading edge bins to ~0.57 mid-trial to ~0.37 late). Two classes;
there is no `no video` class, and bins with no video are zero-filled and therefore land in the
lower class. The human reference assigns class 2 to ~3.8 % of bins.

ii.
```python
me_thr = percentile_threshold(motion_energy, 50.0)
me_bin = discretize_trace(motion_energy, me_thr)
```

```python
"output_values": [..., ["below_session_median", "at_or_above_session_median"]],
```

iii. CONVERSION_NOTES Step 5, Key Decision 8: "Use the reference aligned continuous trace, but
discretize with the task-mandated per-session median instead of the paper's manually chosen move
threshold." Trajectory step 237 explains the switch from `traces >= threshold` to the rank split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per side-camera frame, so it is timed by `obj.traj[0]`'s
`frameTimes`, corrected by the same session `vidshift` and the trial's `goCue`, then interpolated
onto `time_centers`. If a trial's frame count and motion-energy length disagree, the code
substitutes a synthetic 400 Hz frame clock **and a hard-coded 0.5 s offset** instead of the
session's computed `vidshift` — an inconsistency between the two paths.

ii.
```python
view = obj["traj"][0]
...
frame_times = get_frame_times(trial_view, ts.shape[0])
if frame_times.size != me_trial.size:
    frame_times = (np.arange(me_trial.size, dtype=np.float32) + 1.0) / 400.0
    old_time = frame_times - 0.5 - float(align_times[trial])
else:
    old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, me_trial, kind="linear", bounds_error=False,
                  fill_value=np.nan, assume_sorted=True)
aligned[:, trial] = interp(taxis).astype(np.float32)
```

iii. CONVERSION_NOTES Step 1 quotes the reference call the agent is reproducing:
`interp1(frameTimes - video_offset - align_event_time, me.data{trial}, obj.time +
advance_movement)`, then "edge NaNs are filled with nearest values". The 0.5 s constant traces back
to the same Step 1 note: "DLC trajectories also use frame times corrected by a 0.5 s or computed
video offset depending on loader/helper path."

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The general strategy is **impute and continue**, not mark-and-preserve. Specifically:
- Frames where DeepLabCut failed (authors already set `x`/`y` to NaN): tongue → zero velocity;
  paw → nearest-neighbour fill of position and of velocity across the whole window.
- Bins outside the frame range after alignment: `interp1d` yields NaN, then the same rules apply,
  and `aggregate_speed` / `aligned_motion_energy` finish with `np.nan_to_num(..., nan=0.0)`, so no
  NaN reaches the output and no bin is ever labelled "missing".
- Trials whose `NdroppedFrames` is entirely NaN are skipped in `aligned_position`, leaving an
  all-NaN column that becomes an all-zero speed trace.
- Missing or wrong-length `frameTimes`: a synthetic 400 Hz frame clock is fabricated.
- Missing bitcode fields: the video offset silently defaults to 0.5 s.
- Missing motion energy for a session: the whole session's motion energy becomes zeros.
- Nested `me.data` wrappers (`JEB15`): unwrapped in a loop with a cycle guard.
- Behaviour trials past the end of the recording (`JEB24_2023-10-23`, `JEB24_2023-11-03`): detected
  via the last trial in which any kept unit fired, dropped, and the neuron selection re-run.
- Structural guards: sessions with <2 valid trials or <10 units are skipped; `hit`/`miss`
  exclusivity is asserted per session; `verify_data_format` is run before the pickle is written.

ii.
```python
def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    mask = np.isnan(x)
    valid = np.flatnonzero(~mask)
    ...
    x[mask] = x[nearest[mask]]
    return x
```

```python
dropped = trial_view.get("NdroppedFrames")
if dropped is not None:
    dropped_arr = np.asarray(dropped)
    if dropped_arr.size and np.isnan(np.asarray(dropped_arr, dtype=float)).all():
        continue
```

```python
agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 3 grounds the nearest-fill in the paper's methods ("Missing values are
filled with nearest values for all features except the tongue") and in `findVelocity.m` /
`loadMotionEnergy.m`. Step 9/10 documents the two edge cases the agent found and fixed during
critical review (nested `JEB15` motion-energy structs; `JEB24` trials with no neural coverage) and
notes that a `np.nanmean` runtime warning was removed by "an explicit finite-value average". The
agent does not discuss the consequence that imputed bins are indistinguishable from measured ones,
because the two-class output format it was given has no slot for "not visible".

## 11-a. What are the most time-consuming steps of the code?

i. The agent identifies MATLAB file loading as the dominant cost, with kinematic interpolation and
per-neuron spike binning next. Timing is instrumented per session only (`time.perf_counter()`
around `process_session`), not per stage. The measured full run is 479.9 s for 44 sessions
(~2.6–20.8 s/session, tracking file size), against the human reference's ~135 s — roughly 3.5×
slower. The per-session time is dominated by `read_mat`, which materialises the entire `obj` tree,
and by the nine full passes over all trials that `aggregate_speed` performs.

ii.
```python
t0 = time.perf_counter()
obj = read_mat(spec.data_path)["obj"]
...
dt_s = time.perf_counter() - t0
log(f"Processed {spec.session_id}: {n_trials} trials kept, {n_neurons} neurons kept in {dt_s:.2f}s")
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Session loading from large MATLAB
files is the main cost. Kinematic interpolation and per-neuron spike binning dominate runtime."
Step 7 estimated ~10–12 s/session → ~8 min total, "comfortably below the 15-minute optimization
threshold", and Step 9 confirms the estimate held (479.91 s).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several remain, and unlike the reference's loops these operate on *rectangular* arrays (because
everything has already been interpolated onto the common 1000-bin grid), so they were genuinely
vectorizable:
- `causal_gaussian_smooth` convolves column by column with `np.convolve`; one `scipy.signal.fftconvolve`
  or `scipy.ndimage.convolve1d` call along `axis=0` would handle all trials at once.
- `feature_velocity` loops over trials to call `np.gradient` on one column at a time; `np.gradient`
  takes an `axis` argument and the array is already `(1000, n_trials)`.
- The neuron→trial scatter in `process_session` is a nested Python loop over neurons × trials that
  copies one row at a time; it is a transpose.
- `select_neurons` calls `neuron_mean_fr` once per neuron in Python.
What the agent *did* vectorize well: `binned_neuron_trials` bins a whole neuron's spikes across all
trials in one `np.add.at`, and `aggregate_speed` combines feature components with array ops. The
per-trial `interp1d` loop in `aligned_position` is the one that genuinely resists vectorisation,
because frame counts differ per trial.

ii.
```python
for col in range(x_filt.shape[1]):
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
```

```python
for out_idx, selected in enumerate(selected_neurons):
    rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
    for tr in range(n_trials):
        neural_trials[tr][out_idx, :] = rates[:, tr]
```

iii. Not addressed in CONVERSION_NOTES beyond the general statement in Step 6 that loading and
interpolation dominate; the agent stopped optimising once the estimate came in under the
15-minute threshold the instructions set.

## 11-c. What processing does the code repeat multiple times?

i. Four repeats, none of them documented:
- **Spikes are read and scanned twice per neuron** — once in `neuron_mean_fr` to compute the rate
  for the 1 Hz filter, then again in `binned_neuron_trials` to build the trial matrix. The agent
  describes this as a deliberate memory/compute trade-off.
- **`select_neurons` is run twice** on the two `JEB24` sessions whose trial set is trimmed for
  neural coverage, recomputing every quality label and every mean firing rate.
- **Per-trial view dicts are rebuilt once per feature** — `get_view_trial` copies every field of
  every trial, and `aligned_position` is called nine times (7 tongue + 2 paw), plus once more
  inside `aligned_motion_energy`, so each trial's camera fields are re-extracted ten times.
- **Velocities are computed for every raw trial**, including the ~20 % later discarded by the trial
  filter; only afterwards is the result subset with `[:, keep_trials]`.
Correctly computed once: the video offset (`compute_video_offset`), the time grid, and the loader
parse.

ii.
```python
mean_fr = neuron_mean_fr(probe, int(neuron_index), align_times, trial_mask)   # pass 1
...
rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)  # pass 2
```

```python
for trial in range(n_trials):          # ALL trials, not just kept ones
    trial_view = get_view_trial(view, trial)
...
tongue_speed = tongue_speed_all[:, keep_trials]
```

iii. CONVERSION_NOTES Step 6 documents only the intentional one: "Avoided full-session temporary
3D neural arrays before low-FR filtering by using a two-pass approach: pass 1: estimate firing
rate and select neurons; pass 2: build only kept neurons' trial matrices." The other three repeats
are not mentioned.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Most notably, **the three per-session percentile thresholds are computed and then thrown away**:
`percentile_threshold` runs on each of tongue, paw and motion energy, and `discretize_trace`
accepts the value as an argument but never reads it, doing a rank split instead. The thresholds
survive only in `session_info` metadata and the diagnostic plots — where they are drawn as the
"session median" line for classes that were not actually produced by that line. Beyond that:
- `discretize_trace` performs a full `argsort` of the whole session array (≈ n_trials × 1000
  values) when a partition or a comparison would do.
- Kinematics and motion energy are computed for every raw trial and then subset to the kept ones.
- `moveThresh` is parsed out of each motion-energy file and never used.
- `read_mat` materialises the whole `obj` tree, including `clu.spkWavs`, `clu.tm`, the `sglx` index
  arrays, and the DLC features (jaw, nose, etc.) the conversion never touches.
- `feature_velocity` computes `base` (the median inter-sample displacement) even on the tongue
  branch, where it is discarded.

ii.
```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)
paw_thr = percentile_threshold(paw_speed, 50.0)
me_thr = percentile_threshold(motion_energy, 50.0)

tongue_bin = discretize_trace(tongue_speed, tongue_thr)   # threshold unused inside
```

```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")   # full sort
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

```python
diffs = np.diff(tsinterp, axis=0)
base = np.nanmedian(diffs, axis=0)      # discarded when feat_name contains "tongue"
```

iii. CONVERSION_NOTES Step 6 claims the opposite for the video path — "Aggregated output traces
only for the feature groups needed by the decoder task (tongue, paw, motion energy) instead of
reconstructing the full video feature matrix" — which is true as far as it goes, but the dead
threshold computation and the whole-object load are not acknowledged anywhere in the notes or the
trajectory.
