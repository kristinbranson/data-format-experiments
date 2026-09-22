# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** load all of the supplied data. It hard-codes a 12-session cohort — the "two-context" sessions loaded by the authors' `Scripts/Figure 8/Figure8d.m` (`loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo.m`) — together with the single ALM probe each loader specifies. Only `/app/data/Ephys_Behavior` is ever opened; `RandomizedDelay_Ephys_Behavior` (19 further ephys sessions used by the paper) and the 13 remaining fixed-delay sessions (JEB13/JEB14/JEB15) are excluded. Each session is a pair of files: `data_structure_<anm>_<date>.mat`, opened with `h5py` and traversed lazily by HDF5 reference dereferencing (only the datasets actually needed are read, never the whole `obj`), and `motionEnergy_<anm>_<date>.mat`, read with `scipy.io.loadmat`. No v5/`scipy.io` fallback exists for the data-structure file (all 12 selected files are v7.3). Sessions are processed one at a time in a loop, with `gc.collect()` between them.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

# This is the exact two-context session/probe list used by the paper's Figure 8
# scripts and their load<animal>_ALMVideo helpers. Probe numbers are MATLAB 1-based.
CONTEXT_SESSIONS = [
    ("JEB6", "2021-04-18", 2),
    ("JEB7", "2021-04-29", 1),
    ...
    ("JEB19", "2023-04-18", 1),
]
```
```python
data_path = DATA_DIR / f"data_structure_{session_id}.mat"
motion_path = DATA_DIR / f"motionEnergy_{session_id}.mat"
if not data_path.exists() or not motion_path.exists():
    raise FileNotFoundError(f"Missing paired data for {session_id}")
...
with h5py.File(data_path, "r") as handle:
```
```python
def referenced_objects(handle, dataset, *, preserve_empty=False):
    """Dereference a MATLAB cell/struct-field dataset in column-major order."""
    for reference in np.asarray(dataset[()]).ravel(order="F"):
        ...
```

iii. CONVERSION_NOTES Step 4/5: "Figure 8 loads JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, and JEB19 using 12 sessions total… Other fixed-delay sessions sometimes contain a handful of autowater trials but are not in the context-analysis loader… Use the exact 12-session context cohort from Figure 8, not every file with any `autowater=1`." Key decision 1: "Cohort is the 12 author-listed context sessions: This matches Figure 8 and the paper's session/unit statistics; randomized-delay and behavior-only experiments cannot support the required WC/DR neural-decoder task." Step 6: direct HDF5 traversal was chosen "to avoid loading 100–300 MB MATLAB objects wholesale".

## 1-b. How are the data split into subjects?

i. The subject is the animal string in the filename/session spec (the tuple's first element, e.g. `JEB19`). `subjects` is built in order of first appearance across the session list and `subject_idx` indexes into it. The 12 sessions come from 7 animals (JEB6, JEB7 ×2, EKH1, EKH3, JGR2 ×2, JGR3, JEB19 ×4). The paper states 6 mice for this cohort; the AI keeps the 7 native IDs and records the discrepancy in metadata.

ii.
```python
if animal not in subject_lookup:
    subject_lookup[animal] = len(subjects)
    subjects.append(animal)
subject_idx.append(subject_lookup[animal])
```
```python
"native_subject_id_count": len(subjects),
"paper_subject_count_discrepancy": (
    "Author Figure 8 loaders and native filenames identify seven IDs; no "
    "undocumented merge was applied to force the paper's six-mouse statement."
),
```

iii. Step 4: "Loader names imply 7 distinct IDs… Paper reports six mice. Preserve the seven native subject IDs rather than merge undocumented identities. Record the paper/code discrepancy in metadata." Key decision 11: "Native subject IDs are authoritative: No undocumented merging is applied to force the paper's six-mouse statement." Notes also state that some older objects lack `meta`, so subject/date are taken from filenames.

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file = one session = one entry in `neural`/`input`/`output`/`brain_region_idx`. Each session carries exactly one author-designated ALM probe (`alm_probe_matlab_index`), so clusters from simultaneously recorded non-ALM probes never enter the data. Sessions are ordered as listed in `CONTEXT_SESSIONS` (which follows the author loaders, including JEB19's reverse-chronological order). Twelve sessions result, with 210–390 retained trials and 27–67 units each.

ii.
```python
for session_index, (animal, date, probe_number) in enumerate(session_specs):
    result = process_session(animal, date, probe_number)
    neural.append(result.neural)
    inputs.append(result.inputs)
    outputs.append(result.outputs)
    brain_region_idx.append(result.brain_region_idx)
```
```python
probe_cells = referenced_objects(handle, handle["obj/clu"], preserve_empty=True)
cluster_group = probe_cells[probe_number - 1]
```

iii. Step 4: "Per-animal loaders specify one ALM probe: JEB6 p2, JEB7 p1, EKH1 p2, EKH3 p2, JGR2 p1, JGR3 p1, JEB19 p1… Use author loader probe assignments and label all retained neurons `ALM`; do not include simultaneously recorded tjM1/brainstem clusters."

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table: `obj.bp.Ntrials` gives the count, and every per-trial flag (`L`, `R`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `trials.bp.haveEphys`, `trials.bp.haveVid`) plus `bp.ev.goCue` is read as a vector of that length, with an explicit assertion that all lengths agree. Spikes carry their own 1-based `clu.trial` index, which is decremented once and range-checked; DLC trajectories and motion energy are stored one entry per trial and indexed by the same trial number. No trial boundaries are reconstructed.

ii.
```python
n_trials_original = int(np.asarray(handle["obj/bp/Ntrials"][()]).squeeze())
go_cue = np.asarray(handle["obj/bp/ev/goCue"][()]).ravel().astype(np.float64)
if go_cue.size != n_trials_original:
    raise ValueError(f"{session_id}: goCue length mismatch")
flags = trial_flags(handle)
if not all(array.size == n_trials_original for array in flags.values()):
    raise ValueError(f"{session_id}: trial flag length mismatch")
```
```python
trial_number = np.asarray(trial_number_objects[cluster_index][()]).ravel().astype(np.int64) - 1
valid_trial = (trial_number >= 0) & (trial_number < n_trials)
```

iii. Step 10 edge-case review: "Verified MATLAB 1-based spike trial numbers are decremented exactly once, probes remain explicitly MATLAB-indexed in configuration…". The length assertions are listed among the planned sanity checks ("Confirm all trial matrices have 1,000 timepoints and session neuron counts are constant").

## 1-e. How are trials filtered based on quality controls?

i. A single mask, applied after the neural tensor is computed: keep trials with a finite go cue, with `trials.bp.haveEphys` true, that are **not** early-lick trials (`bp.early`) and **not** photostimulation trials (`bp.stim.enable`). Ignore (no-response) trials are deliberately **kept**, because the decoder spec requires `ignore`/`none` classes. Video availability is **not** required — trials without usable video are kept and their movement outputs become class 2. A session is rejected if fewer than 2 trials or fewer than 10 units survive, and the code asserts that `hit`/`miss`/`no` are mutually exclusive and exhaustive on the retained trials. 3,116 of 3,905 trials are retained.

ii.
```python
use = (
    np.isfinite(go_cue)
    & flags["have_ephys"]
    & ~flags["early"]
    & ~flags["stim"]
)
retained_trials = np.flatnonzero(use)
if retained_trials.size < 2:
    raise ValueError(f"{session_id}: fewer than two retained trials")

outcome_count = (flags["hit"].astype(np.int8) + flags["miss"].astype(np.int8)
                 + flags["no"].astype(np.int8))
if not np.all(outcome_count[retained_trials] == 1):
    raise ValueError(f"{session_id}: hit/miss/no are not mutually exhaustive")
```

iii. Step 4: "Author condition strings generally exclude stimulation and early trials; most analyses omit ignores… Exclude `early` and `stim.enable`. Retain ignore trials only because the decoder explicitly requires an ignore/none class." Key decision 5 repeats this. Planned sanity check: "Confirm all retained trials satisfy finite go cue, have ephys, `~early`, and `~stim.enable`; do not require video because class 2 encodes its absence."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the one author-designated ALM probe: the per-cluster `quality` label (curation), `trial` (1-based trial of each spike) and `trialtm` (spike time relative to trial start). `obj.bp.ev.goCue` supplies the alignment times, and the Figure-8 condition flags (`hit`, `miss`, `no`, `autowater`, `stim.enable`, `early`) are used for the low-firing-rate criterion. No waveforms, session-clock times (`clu.tm`), depths or channels are read.

ii.
```python
selected_indices, quality_labels = select_quality_indices(handle, cluster_group)
trial_time_objects = referenced_objects(handle, cluster_group["trialtm"])
trial_number_objects = referenced_objects(handle, cluster_group["trial"])
...
trial_time = np.asarray(trial_time_objects[cluster_index][()]).ravel().astype(np.float64)
trial_number = np.asarray(trial_number_objects[cluster_index][()]).ravel().astype(np.int64) - 1
aligned = trial_time - go_cue[trial_number]
```

iii. Step 1 documents `findClusters`/`alignSpikes`/`getSeq`/`removeLowFRClusters` as the reference chain; Step 5's mapping row is "Selected `obj.clu{probe}.trialtm`, `.trial`, `.quality` + `bp.ev.goCue` → `neural[session][trial]`".

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 5 ms bins on an extended window `[-3.0, +2.5)` s (1,100 bins), divided by the bin width to give Hz, then smoothed with an exact port of the authors' `mySmooth(x, 15, 'reflect')`: a MATLAB `gausswin(15)` (alpha 2.5, i.e. std 2.8 samples) whose **leading half is zeroed** (causal), renormalised, and applied as an FIR filter after prepending the first 15 samples as the reflect boundary. The result is cropped to the 1,000 bins spanning `[-2.5, +2.5)`, so the first reported bin has genuine 0.5 s of filter history rather than a boundary artifact. Values are stored as float32 firing rates in Hz; there is no z-scoring, baseline subtraction or normalisation.

ii.
```python
def causal_gaussian_kernel(length: int = SMOOTH_SAMPLES) -> np.ndarray:
    """Reproduce MATLAB gausswin(N), zeroing its leading half as mySmooth.m."""
    kernel = gaussian(length, std=(length - 1) / (2 * 2.5)).astype(np.float32)
    kernel[: length // 2] = 0
    kernel /= kernel.sum()
    return kernel

CAUSAL_TAPS = CAUSAL_KERNEL[SMOOTH_SAMPLES // 2 :]

def reference_smooth(rates):
    prefix = rates[..., :SMOOTH_SAMPLES]
    padded = np.concatenate((prefix, rates), axis=-1)
    filtered = lfilter(CAUSAL_TAPS, np.array([1.0], dtype=np.float32), padded, axis=-1)
    return filtered[..., SMOOTH_SAMPLES:].astype(np.float32, copy=False)
```
```python
bin_index = np.floor((aligned - FILTER_TMIN) / DT).astype(np.int64)
np.add.at(counts[output_index], (trial_number[valid_bin], bin_index[valid_bin]), 1.0)
...
rates = reference_smooth(counts / np.float32(DT))
```

iii. Step 3/4: "For single-trial analysis the paper bins at 5 ms and smooths with a causal Gaussian kernel of 35 ms half-width. The reference code realizes this as `dt=1/200` and a 15-sample causal `gausswin`… At 5 ms, the 15-sample code kernel matches the stated ~35 ms half-width; port it directly." Key decision 3: "Reference causal firing rates, not raw counts… matching the paper rather than inventing a new preprocessing stream." Step 10: "5-ms histograms, 15-sample causal `gausswin`, reference reflect prefix; crop after filtering… crop avoids boundary artifacts."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Probe: only the author-designated ALM probe is read. (2) Manual curation label: an exact, **case-sensitive** port of `findClusters(..., {'all'})` — labels are stripped and clusters labelled `garbage`, `gabrga`, `noisy`, `real?` are dropped; everything else (including multi-units, `Poor`, unlabeled, and the case variant `Noisy`) is kept, exactly as the MATLAB `ismember` does. (3) Firing rate: an exact port of `removeLowFRClusters` with Figure-8's `lowFR = 1` — the smoothed rates are averaged within each of the seven Figure-8 conditions (empty conditions contribute zeros, as in MATLAB where `psth` stays zero), those seven condition means are averaged, and units with mean ≤ 1 Hz are dropped. Sessions with fewer than 10 surviving units are rejected. Result: 521 units (213 with single-unit labels) versus the paper's 522 (214).

ii.
```python
def select_quality_indices(handle, cluster_group):
    """Port findClusters(..., {'all'}), including its case-sensitive exclusions."""
    for index, quality_object in enumerate(quality_objects):
        label = matlab_char(quality_object).strip() if isinstance(...) else ""
        if label in {"garbage", "gabrga", "noisy", "real?"}:
            continue
        selected.append(index)
```
```python
def reference_conditions(flags):
    """Conditions used by the paper's context analysis before low-FR filtering."""
    return [hit | miss | no,
            hit & ~stim & ~aw, hit & ~stim & aw,
            miss & ~stim & ~aw, miss & ~stim & aw,
            hit & ~stim & ~aw & ~early, hit & ~stim & aw & ~early]
...
for condition_index, mask in enumerate(reference_conditions(flags)):
    trial_indices = np.flatnonzero(mask)
    if trial_indices.size:
        condition_means[:, condition_index] = rates[:, trial_indices, :].mean(axis=(1, 2), dtype=np.float64)
mean_fr = condition_means.mean(axis=1)
keep = mean_fr > LOW_FR_HZ
rates = rates[keep][:, :, OUTPUT_MASK]
```
```python
if rates.shape[0] < 10:
    raise ValueError(f"{session_id}: only {rates.shape[0]} retained neurons")
```

iii. Step 4: "Figure 8 uses `lowFR=1`; the all-unit analysis uses `quality={'all'}`… Port `findClusters`, `getSeq`, `mySmooth`, and `removeLowFRClusters` semantics and verify against 522/214. Use all non-explicitly-bad manual labels for decoder data." Key decision 4: "All scientifically curated ALM units, not singles only: The analogous population decoder includes multiunits as well as well-isolated units." Step 9 records that the exact port yields 521/213 and concludes the one-unit gap is a deposited-archive vs analysis-time difference rather than something to fabricate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is one subtraction: each spike's within-trial time `clu.trialtm` minus `bp.ev.goCue` of that spike's own trial, exactly as `alignSpikes.m` does with `params.alignEvent = 'goCue'`. Both are on the behaviour clock, so no offset or interpolation is needed for spikes (only the video streams need the bitcode offset). The metadata records that in the WC/autowater context this same field is the water-drop onset rather than an auditory cue.

ii.
```python
aligned = trial_time - go_cue[trial_number]
bin_index = np.floor((aligned - FILTER_TMIN) / DT).astype(np.int64)
```
```python
"temporal_alignment_event": (
    "obj.bp.ev.goCue onset (auditory go cue in DR; water-drop onset in WC)"
),
```

iii. Step 4: "All author scripts set `alignEvent='goCue'`; video and motion energy subtract `bp.ev.goCue`… Align every stream to native `bp.ev.goCue`; metadata states that this is water-drop onset in WC trials." Step 10's reference comparison calls this the "exact event and synchronization convention".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (`DT = 0.005`, i.e. the reference `params.dt = 1/200`), 1,000 bins with centres from −2.4975 s to +2.4975 s. Spikes are binned once at this resolution, so there is no rebinning or resampling of the neural data; the only extra step is that filtering is done on 1,100 bins starting at −3.0 s (Figure 8's `params.tmin`) and the first 100 bins are discarded after smoothing. All behavioural streams are interpolated onto this same 1,000-bin axis, so every stream shares one time base. `metadata['time_bin_size'] = 5.0` ms, `off_start = -2.5`, `off_end = 2.5`.

ii.
```python
DT = 0.005
FILTER_TMIN = -3.0  # Figure 8 processing start; output is cropped after smoothing.
OUTPUT_TMIN = -2.5
TMAX = 2.5
FILTER_EDGES = np.arange(FILTER_TMIN, TMAX + DT / 2, DT, dtype=np.float64)
FILTER_TIME = (FILTER_EDGES[:-1] + DT / 2).astype(np.float32)
OUTPUT_MASK = (FILTER_TIME >= OUTPUT_TMIN) & (FILTER_TIME < TMAX)
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
N_TIME = int(OUTPUT_TIME.size)
if N_TIME != 1000:
    raise RuntimeError(f"Expected 1000 output bins, constructed {N_TIME}")
```

iii. Step 4: "Figure 8d and `getDefaultParams` use 5 ms; some plotting scripts use 10 or 30 ms… Single-trial Methods specify 5 ms. Use 5 ms consistently across sessions." And: "Standard loader/default is `[-2.5,+2.5)`; context Figure 8 uses a `-3` s start… Use `[-2.5,+2.5)` as the standard decoder window." Step 1 notes "The bin centers produced by `getSeq` are `tmin + dt/2` through `tmax - dt/2`; the right endpoint is excluded."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. None — it is constructed by the conversion itself. It is the vector of bin centres of the go-cue-aligned grid (the same grid the spikes are binned into), i.e. −2.4975 … +2.4975 s in 5 ms steps. The same `(1, 1000)` float32 array is attached to every trial of a session.

ii.
```python
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]        # bin centres, -2.4975 ... 2.4975
...
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
...
input_trials.append(input_template)
```
```python
"input_names": ["time_from_go_cue_seconds"],
```

iii. Step 5 mapping: "Fixed bin centers relative to alignment → `input[session][trial][0,:]`: Copy continuous seconds-from-go-cue vector to every trial as `(1,1000)` float32… Only decoder input requested; do not leak outputs/context into inputs." Key decision 12 emphasises that no other experimental variable is added to the input to prevent leakage.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The vector is computed once at module import from the window constants and reused; it is kept continuous (seconds) rather than binarised, as the Decoder Task specifies a continuous, time-varying input. To save memory, one array object per session is shared by all of that session's trials.

ii.
```python
input_template = OUTPUT_TIME.reshape(1, -1).astype(np.float32)
```
```python
"time_bin_centers_seconds": OUTPUT_TIME.copy(),
```

iii. Step 6: "Reuses one immutable input time array per session; pickle memoization stores shared arrays efficiently." Planned sanity check: "Independently construct the expected bin-center vector and compare every input with `np.allclose`" — reported passed in Step 10 (atol 1e-7).

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction it *is* the neural time axis: `OUTPUT_TIME` is the set of bin centres of the same edges the spikes are histogrammed into, and every behavioural stream is interpolated onto the identical vector. Bin *k* therefore denotes the same interval in the input, the neural matrix and all six outputs.

ii.
```python
FILTER_EDGES = np.arange(FILTER_TMIN, TMAX + DT / 2, DT, dtype=np.float64)
FILTER_TIME = (FILTER_EDGES[:-1] + DT / 2).astype(np.float32)
OUTPUT_TIME = FILTER_TIME[OUTPUT_MASK]
```
```python
bin_index = np.floor((aligned - FILTER_TMIN) / DT).astype(np.int64)   # neural
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)          # video
```

iii. Step 4/Step 10: "Neural and video data will share 1,000 bins spanning centers from -2.4975 to +2.4975 s at 5 ms resolution around `bp.ev.goCue`." The `--show-processing` plot includes a dedicated "Decoder input and temporal alignment" panel to demonstrate this.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial Bpod flags: `obj.bp.no`, `obj.bp.L` and `obj.bp.R`. These are the **trial-type / instructed** flags (the authors' own tutorial defines `right_correct_trials_mask = obj.bp.hit & obj.bp.R`), not a record of which port the animal actually contacted. The outcome flags `hit`/`miss` are read for the outcome output but are **not** used for lick direction, and the actual lick times in `bp.ev.lickL` / `bp.ev.lickR` are never read.

ii.
```python
flags[name] = np.asarray(handle[f"obj/bp/{name}"][()]).ravel().astype(bool)
#   for name in ("L", "R", "hit", "miss", "no", "early", "autowater")
```
```python
if flags["no"][source_trial_index]:
    lick_direction = 2
elif flags["L"][source_trial_index]:
    lick_direction = 0
elif flags["R"][source_trial_index]:
    lick_direction = 1
else:
    raise ValueError(f"{session_id}: trial {source_trial_index + 1} has no direction")
```

iii. Step 5 mapping row: "`bp.L`, `bp.R`, `bp.no` → `output[...,0,:]` lick direction. Code left=0, right=1, none=2; repeat per-trial code over time… `no` overrides instructed L/R to `none`." Step 10 records an audit iteration on this variable, but only about ignore trials: "The first audit-script draft treated raw L/R instruction flags as a lick on ignore trials. Inspection showed ignore trials retain an instructed side; the independent expectation was fixed to give raw `no` precedence." The notes never address error (miss) trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct relabelling with `no` taking precedence: ignore → 2 (`none`), otherwise the instructed side, `L` → 0 (`left`), `R` → 1 (`right`). The scalar is broadcast across all 1,000 time bins so that all six outputs share one `(6, 1000)` int8 matrix. An exception is raised if a non-ignore trial has neither `L` nor `R`. Resulting distribution: 35.7 % left, 41.8 % right, 22.5 % none.

ii.
```python
output = np.empty((6, N_TIME), dtype=np.int8)
output[0].fill(lick_direction)
```
```python
"output_values": [
    ["left", "right", "none"],
    ...
```

iii. Key decision 6: "Per-trial categorical variables are repeated over time: This creates a single `(6,time)` output matrix alongside time-varying movement outputs and allows the validator/decoder to treat every output consistently." No justification is given for equating the instructed side with the licked side; the notes call the flags "instructed" but use them directly as the lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which is 1 on water-cued (WC) trials where water is delivered from a random port without a cue, and 0 on delayed-response (DR) trials.

ii.
```python
flags["autowater"] = np.asarray(handle["obj/bp/autowater"][()]).ravel().astype(bool)
```

iii. Step 1 lists `NeuralContextDecoding.m` — "Reference context labels distinguish delayed response (`~autowater`) from water-cued (`autowater`) trials" — and the tutorial note "obj.bp.autowater=1 when water was delivered regardless of animal choice… this field can be used as a proxy for obtaining water-cued blocks and delayed-response blocks".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater` → WC = 0, otherwise DR = 1, broadcast over the 1,000 bins. Observed distribution 31.5 % WC / 68.5 % DR, with both classes present in every one of the 12 sessions (26–39 % WC per session).

ii.
```python
context = 0 if flags["autowater"][source_trial_index] else 1
...
output[1].fill(context)
```
```python
"output_values": [..., ["WC", "DR"], ...]
```

iii. Step 5 mapping: "`bp.autowater` → `output[...,1,:]` behavioral context. WC=0 when true, DR=1 when false; repeat over time… WC uses water-drop event stored in `goCue`." The class codes follow the Decoder Task's "(WC, DR)" ordering.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags, `obj.bp.miss` and `obj.bp.hit`, with `obj.bp.no` used both as the fall-through class and as a consistency check (the code asserts `hit + miss + no == 1` on every retained trial).

ii.
```python
outcome_count = (flags["hit"].astype(np.int8) + flags["miss"].astype(np.int8)
                 + flags["no"].astype(np.int8))
if not np.all(outcome_count[retained_trials] == 1):
    raise ValueError(f"{session_id}: hit/miss/no are not mutually exhaustive")
```

iii. Step 1 identifies `getOutcome.m` ("Uses `bp.hit` for correct/incorrect and sets `bp.no` trials to NaN (ignore)"); Step 5 mapping: "`bp.miss`, `bp.hit`, `bp.no` → `output[...,2,:]` outcome… Validate exactly one state per retained trial."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling broadcast over time: `miss` → 0 (incorrect), `hit` → 1 (correct), anything else → 2 (ignore). Ignore trials, which the paper normally discards, are retained specifically so this class exists. Distribution: 10.6 % incorrect, 66.9 % correct, 22.5 % ignore.

ii.
```python
if flags["miss"][source_trial_index]:
    outcome = 0
elif flags["hit"][source_trial_index]:
    outcome = 1
else:
    outcome = 2
...
output[2].fill(outcome)
```
```python
["incorrect", "correct", "ignore"],
```

iii. Step 4: "Correct and error trials are used by the core single-trial subspace analyses. Ignore trials are normally omitted in paper analyses, but the decoder specification explicitly requires an `ignore` outcome and `none` lick direction; retaining valid ignore trials is therefore an intentional downstream exception." The codes follow the Decoder Task's "(incorrect, correct, ignore)" ordering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` (the **side** camera only): `featNames` (to locate the feature named `tongue`), `ts` (x, y, likelihood per frame), `frameTimes`, and `NdroppedFrames` (video validity). Alignment additionally uses `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, `obj.bp.ev.bitStart` and `obj.bp.ev.goCue`. The bottom camera's tongue markers (`top_tongue`, `bottom_tongue`, …) are not used.

ii.
```python
side_group = trajectory_group(handle, 0)     # obj.traj{1}, side camera
...
tx, ty, tongue_raw = aligned_feature_position(
    handle, side_group, int(source_trial_index), "tongue",
    go_cue[source_trial_index], offset,
)
```
```python
ts = trajectory_array(ts_object, len(names))   # frames x 3 x features
feature_index = names.index(feature)
x_raw = ts[:, 0, feature_index]
y_raw = ts[:, 1, feature_index]
```

iii. Key decision 8: "Central side-view tongue and top-paw bottom-view landmarks: These are the canonical named tongue marker and the same paw marker used by paper plotting code, avoiding arbitrary averaging across anatomically distinct landmarks." Step 2 notes "Side-camera features include tongue/jaw/trident/nose; bottom-camera features include tongue landmarks, top/bottom paw…".

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. A port of `findPosition.m` + `findVelocity.m` (tongue branch). (1) Trials whose `NdroppedFrames` is NaN are treated as having no video. (2) Frame times are put on the go-cue clock (see 7-d). (3) x and y are linearly interpolated onto the 1,000 bin centres with no extrapolation, so frames outside the window and DLC gaps (the authors already store x/y as NaN where likelihood is low) propagate NaN. (4) The visibility mask is taken **before** any fill, from the interpolated x/y NaNs. (5) Velocity is `np.gradient` of x and y with no smoothing and no baseline subtraction (the reference explicitly does not smooth or baseline-correct the tongue), NaNs are replaced by 0 exactly as `findVelocity.m` does, and speed is `hypot(vx, vy)`. Units are pixels per bin rather than pixels/s, which is irrelevant because the threshold is a percentile of the same quantity.

ii.
```python
def feature_speed(x, y, *, tongue):
    """Port findVelocity.m and return speed plus pre-fill visibility."""
    visibility = np.isfinite(x) | np.isfinite(y)
    if tongue:
        x_velocity = np.gradient(x)
        y_velocity = np.gradient(y)
        x_velocity[~np.isfinite(x_velocity)] = 0.0
        y_velocity[~np.isfinite(y_velocity)] = 0.0
    ...
    return np.hypot(x_velocity, y_velocity), visibility
```
```python
def interpolate_with_nan(x, y, target):
    valid_x = np.isfinite(x)
    if valid_x.sum() < 2:
        return np.full(target.shape, np.nan, dtype=np.float64)
    # MATLAB interp1 does not extrapolate and propagates NaN-valued coordinate gaps.
    return np.interp(target, x[valid_x], y[valid_x], left=np.nan, right=np.nan)
```

iii. Step 4: "`findPosition` uses measured frame times, computed session video offset, go-cue subtraction, linear interpolation, and nearest fill for non-tongue features… Match alignment and interpolation. Preserve visibility before any fill so requested class 2 can be represented." Key decision 7: "Velocity is 2-D speed from reference x/y derivatives: A percentile of signed velocity would separate direction rather than movement magnitude." Step 5 mapping: "Positive scaling to pixels/s is unnecessary because median classes are scale-invariant; use reference pixels/bin derivative."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session, computed once over all retained trials and all 1,000 bins: the 50th percentile of the speed restricted to samples that are both visible (pre-fill mask) and finite. Samples below it get class 0, samples at or above it class 1, and every non-visible sample class 2. If a session has no visible sample at all the whole session becomes class 2. Across the dataset 91.9 % of tongue bins are `not_visible`, 4.0 % class 0 and 4.2 % class 1. In one session (JEB19_2023-04-19) the median speed is exactly 0, so ties push all visible bins into class 1 and class 0 is absent from that session.

ii.
```python
def discretize_session(values, visibility):
    valid = visibility & np.isfinite(values)
    if not np.any(valid):
        return np.full(values.shape, 2, dtype=np.int8), float("nan")
    threshold = float(np.percentile(values[valid], 50))
    output = np.full(values.shape, 2, dtype=np.int8)
    output[valid & (values < threshold)] = 0
    output[valid & (values >= threshold)] = 1
    return output, threshold
```
```python
tongue_class, tongue_threshold = discretize_session(tongue_speed, tongue_visible)
```

iii. Key decision 9: "Threshold population: Each median is computed once per session from all finite samples in retained trials and the full decoder window. Missing samples are excluded from threshold estimation and assigned class 2." Step 10 records the tie audit: "Exact ties at the median (one session has median tongue speed zero) correctly belong to `>= median`."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A per-session video offset is computed exactly as `findVideoOffset.m` — `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`, using a MATLAB-compatible mode (smallest value on ties) — and each trial's frame times become `frameTimes − offset − goCue[trial]`. The aligned frames are then linearly interpolated onto the same 1,000 bin centres as the spikes, so no separate resampling or lag is introduced.

ii.
```python
def video_offset_seconds(handle) -> float:
    bit_start_neural = np.asarray(handle["obj/sglx/bitcode/bitstart"][()]).ravel()
    sampling_rate = float(np.asarray(handle["obj/sglx/fs"][()]).squeeze())
    bit_start_behavior = np.asarray(handle["obj/bp/ev/bitStart"][()]).ravel()
    return matlab_mode(bit_start_neural) / sampling_rate - matlab_mode(bit_start_behavior)
```
```python
aligned_frames = frames - video_offset - go_cue
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
y = interpolate_with_nan(aligned_frames, y_raw, OUTPUT_TIME)
```

iii. Step 1: "`findVideoOffset` computes neural/video offset as `mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`"; Step 10's reference comparison: "Subtract per-trial `goCue`; video offset from neural/behavior bit-start modes… Exact event and synchronization convention." The offset is computed once per session and stored in `metadata['video_offset_seconds']`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{2}` (the **bottom** camera) for the feature `top_paw` — its `featNames`, `ts` (x, y), `frameTimes` and `NdroppedFrames` — plus the same go cue and video offset. `bottom_paw` is not used.

ii.
```python
bottom_group = trajectory_group(handle, 1)
...
px, py, paw_raw = aligned_feature_position(
    handle, bottom_group, int(source_trial_index), "top_paw",
    go_cue[source_trial_index], offset,
)
```

iii. Step 5 mapping: "Bottom-camera DLC feature `top_paw` x/y and frame times → `output[...,4,:]` paw velocity class… Figure 1 uses `top_paw_yvel_view2`." Key decision 8 explains the choice of the canonical marker used by the paper's own plotting code.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The `findVelocity.m` **non-tongue** branch, ported literally. After the same alignment and interpolation onto the 1,000 bins, (1) the pre-fill visibility mask is stored, (2) x and y positions are nearest-filled (`fillmissing(...,'nearest')`), (3) the baseline drift `median(diff([x y]))` is computed and subtracted from the gradients — including the reference's apparent bug of subtracting the **x** baseline from the y velocity, which the AI reproduces deliberately and flags in a comment, (4) velocities are nearest-filled again, and (5) speed is `hypot(vx, vy)`. If nothing is finite the trial returns all-NaN speed.

ii.
```python
    else:
        x_filled = fill_nearest(x)
        y_filled = fill_nearest(y)
        if not np.any(np.isfinite(x_filled)) or not np.any(np.isfinite(y_filled)):
            return np.full(x.shape, np.nan), visibility
        base_derivative = np.nanmedian(np.diff(np.column_stack((x_filled, y_filled)), axis=0), axis=0)
        x_velocity = np.gradient(x_filled) - base_derivative[0]
        # Preserve the supplied findVelocity.m behavior (it subtracts x baseline here).
        y_velocity = np.gradient(y_filled) - base_derivative[0]
        x_velocity = fill_nearest(x_velocity)
        y_velocity = fill_nearest(y_velocity)
    return np.hypot(x_velocity, y_velocity), visibility
```

iii. Step 5 mapping: "Same alignment; nearest-fill coordinate gaps for speed as reference; derivative/baseline correction; Euclidean x/y speed; session median over originally visible samples… Class 2 uses pre-fill visibility, satisfying requested missingness while numerical speed follows reference." Step 1: "Non-tongue gaps are nearest-filled; tongue nonvisibility is tracked from NaNs before baseline substitution."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: `discretize_session` with the per-session 50th percentile of the visible, finite speed samples pooled over all retained trials and bins; < threshold → 0, ≥ threshold → 1, not visible (pre-fill) → 2. Overall 43.5 % / 43.5 % / 12.9 %, with per-session `not_visible` ranging from 0.7 % to 34.5 %.

ii.
```python
paw_class, paw_threshold = discretize_session(paw_speed, paw_visible)
```
```python
"paw_velocity_median": paw_threshold,
```

iii. Same as 7-c: key decision 9, plus Step 4 "Decoder task explicitly overrides this with a finite-sample per-session 50th percentile". Note the asymmetry the AI chose deliberately: the *value* being thresholded is the nearest-filled reference velocity, but the *class-2 mask* is the pre-fill visibility, so filled samples are still thresholded rather than being forced to class 2.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the bottom camera's own `frameTimes`: `frameTimes − video_offset − goCue[trial]`, then linear interpolation onto the shared 1,000-bin axis. The same session-level offset serves both cameras.

ii.
```python
frames = frame_times_for_trial(handle, group, trial_index, ts.shape[0])
aligned_frames = frames - video_offset - go_cue
x = interpolate_with_nan(aligned_frames, x_raw, OUTPUT_TIME)
```

iii. Step 5/Step 10: alignment is described once for all video streams — "Aligns DLC and motion-energy streams from original frame times using the computed SpikeGLX/video offset", verified by the independent per-stream alignment check in Step 10 Check 4.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The paired standalone file `motionEnergy_<anm>_<date>.mat`, field `me.data` — one variable-length vector per trial, one value per video frame. The author's manual `me.moveThresh` is deliberately ignored. Timing comes from the **side** camera's `frameTimes` (`obj.traj{1}`), matching `loadMotionEnergy.m`, with a `(1:n)/400` fallback if frame times are missing or their length disagrees with the motion-energy vector.

ii.
```python
def load_motion_energy(path: Path) -> list[np.ndarray]:
    motion = loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    data = np.atleast_1d(motion.data).ravel()
    return [np.asarray(item, dtype=np.float64).ravel() for item in data]
```
```python
group = trajectory_group(handle, 0)
...
if frames.size != values.size:
    frames = np.arange(1, values.size + 1, dtype=np.float64) / 400.0
```

iii. Step 2: "Motion-energy files contain `me.data`, a per-trial variable-length vector sampled on video frames, and a scalar author threshold `me.moveThresh`. The requested decoder instead requires a per-session 50th-percentile threshold." Step 5 mapping: "Ignore supplied manual `moveThresh` because decoder explicitly requires 50th percentile."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling — the value is already one scalar per frame (the paper's 99th percentile over pixelwise frame differences). The trace is linearly interpolated onto the 1,000 bin centres and then nearest-filled (`fillmissing(me.data,'nearest')` in `loadMotionEnergy.m`), so edge and interior gaps inside a valid video trial are filled rather than marked missing. Trials with no video, with fewer than 2 samples, or with no frames overlapping the window remain all-NaN.

ii.
```python
interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
interpolated = fill_nearest(interpolated)
return interpolated, {...}
```
```python
if trial_index >= len(motion_data) or not trial_video_is_valid(handle, group, trial_index):
    return np.full(N_TIME, np.nan), {}
```

iii. Step 5 mapping: "Align/interpolate using video offset and go cue; nearest-fill finite-trial edge gaps; session median over finite values; low=0/high=1/no-video=2… `loadMotionEnergy`." Key decision 10: "No-video versus edge gaps: Valid video trials receive nearest-filled edge values as in `loadMotionEnergy`; only trials/streams with no usable frames remain class 2."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `discretize_session` call: the per-session 50th percentile of the finite aligned values over all retained trials and bins; < threshold → 0, ≥ threshold → 1, and no usable video → 2 (`no_video`). The split is almost exactly 50/50 in every session (0.497–0.500); exactly one trial in the whole dataset (1,000 of 3,116,000 bins) is `no_video`.

ii.
```python
motion_class, motion_threshold = discretize_session(motion_energy, motion_visible)
```
```python
motion_visible[output_trial_index] = np.isfinite(me)
```
```python
["below_session_median", "at_or_above_session_median", "no_video"],
```

iii. Step 4: "Data include an author manual threshold… Decoder task explicitly overrides this with a finite-sample per-session 50th percentile; keep author alignment but use requested median." Step 11 explains the rare class: "the dataset contains exactly one no-video trial… Retaining it is mandatory under the requested output schema."

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the DLC streams: `frameTimes(side camera) − video_offset − goCue[trial]`, then `interp1` onto the 1,000 bin centres. Motion energy is computed from the side-camera video, so the side camera's frame times are the correct clock.

ii.
```python
frame_object = referenced_at(handle, group["frameTimes"], trial_index)
if isinstance(frame_object, h5py.Dataset):
    frames = np.asarray(frame_object[()]).ravel().astype(np.float64)
else:
    frames = np.arange(1, values.size + 1, dtype=np.float64) / 400.0
...
aligned_frames = frames - video_offset - go_cue
interpolated = interpolate_with_nan(aligned_frames, values, OUTPUT_TIME)
```

iii. Matches `loadMotionEnergy.m` verbatim (`interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)`); documented in Step 1 and confirmed in Step 10's reference-code comparison table.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, all handled by keeping the trial and encoding the gap rather than dropping data:
- **Invalid video trial** (`NdroppedFrames` absent/NaN): the trial's DLC and motion-energy streams become all-NaN → class 2, exactly as `findPosition.m` skips those trials.
- **Missing or all-NaN `frameTimes`**: the `findPosition.m` convention `(1:nFrames)/400` is synthesised.
- **Frame/motion-energy length mismatch**: the synthetic time base is used instead.
- **Feature absent from `featNames`** for a trial: that stream becomes NaN → class 2.
- **DLC drop-outs** (authors store NaN where likelihood is low): NaN propagates through `np.interp`; tongue keeps them as class 2, paw and motion energy are nearest-filled for the value but class 2 still follows the pre-fill mask for the paw.
- **Structural errors are fatal rather than silently patched**: mismatched `goCue`/flag lengths, a missing probe, non-exclusive `hit/miss/no`, fewer than 10 units or 2 trials all raise.
- A final `validate_converted` pass checks shapes, finiteness of neural/input, and that outputs only contain {0,1,2} (and context only {0,1}) before pickling.
Two robustness gaps remain: the data-structure reader is HDF5-only (no `scipy.io` fallback for the v5 files that exist elsewhere in `/app/data`), and `load_motion_energy` does not implement the reference's `if isstruct(me.data), me.data = me.data.data; end` unwrap. Neither triggers on the 12 selected sessions.

ii.
```python
def trial_video_is_valid(handle, group, trial_index) -> bool:
    if "NdroppedFrames" not in group:
        return True
    dropped_object = referenced_at(handle, group["NdroppedFrames"], trial_index)
    if not isinstance(dropped_object, h5py.Dataset):
        return False
    dropped = np.asarray(dropped_object[()], dtype=np.float64).ravel()
    return dropped.size > 0 and not np.all(np.isnan(dropped))
```
```python
    # This is the exact synthetic time convention used in findPosition.m.
    return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
```
```python
if not np.all(np.isin(outputs, [0, 1, 2])):
    raise ValueError("Output contains invalid categorical values")
```

iii. Key decision 10 and Step 10's audit: "One absent aligned ME trial and per-bin tongue/paw NaNs remain class 2 rather than being imputed"; "The initial no-video audit relied only on `haveVid`; one trial has `haveVid=true` but no ME samples overlapping the aligned window… matching the decoder's semantic `no video` class." Step 6: "Validates data shapes and types at each step… Performs internal shape, finiteness, categorical-domain, trial-count, neuron-count, and region-index validation before saving."

## 11-a. What are the most time-consuming steps of the code?

i. The full conversion takes ~34 s (2.0–4.1 s per session) plus 0.6 s to pickle 552 MiB, so nothing is a practical bottleneck. The AI attributes the cost to file reading and states this in the notes; it prints per-session wall time (`processing_seconds`) but never profiles the sub-steps, so the attribution is asserted rather than measured. In practice the two dominant costs inside a session are (1) the per-cluster spike-binning loop with `np.add.at` into a `(n_clusters, n_trials, 1100)` float32 tensor and the FIR filter over that tensor, and (2) the per-trial HDF5 dereferencing of `obj.traj` (three features × ~300 trials, each re-reading `featNames`, `ts`, `frameTimes`, `NdroppedFrames`).

ii.
```python
elapsed = time.perf_counter() - started
metadata = {..., "processing_seconds": float(elapsed)}
...
print(f"  {result.metadata['n_trials_retained']} trials, "
      f"{result.metadata['n_neurons']} neurons, "
      f"{result.metadata['processing_seconds']:.2f} s", flush=True)
```

iii. Step 6/7: "Uses direct HDF5 reference traversal to avoid loading 100–300 MB MATLAB objects wholesale… Direct selective HDF5 reads and shared input arrays — low I/O overhead". Step 7 estimates "about 2.9 s/session → about 35 s for 12 sessions", "far below the 15-minute optimization threshold", which the full run confirmed.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops are already vectorized away: smoothing is a single `lfilter` over the whole `(units, trials, time)` tensor instead of MATLAB's per-unit/per-trial `conv`, and binning uses indexed accumulation instead of per-trial `histc`. What remains: (1) the per-cluster loop around `np.add.at`, which could be a single `np.histogram2d`/`bincount` over all clusters at once (and `np.add.at` is itself the slow unbuffered path); (2) the per-trial video loop, which is genuinely irregular because each trial has a different number of frames, though the three feature streams of one trial could at least share a single read of `ts`; (3) `matlab_char`, which decodes strings one character at a time in a Python loop and is called for every cluster label and every feature name of every trial.

ii.
```python
for output_index, cluster_index in enumerate(selected_indices):
    ...
    np.add.at(counts[output_index], (trial_number[valid_bin], bin_index[valid_bin]), 1.0)
```
```python
filtered = lfilter(CAUSAL_TAPS, np.array([1.0], dtype=np.float32), padded, axis=-1)
```
```python
for output_trial_index, source_trial_index in enumerate(retained_trials):
    tx, ty, tongue_raw = aligned_feature_position(...)
    px, py, paw_raw = aligned_feature_position(...)
    me, motion_raw = aligned_motion_energy(...)
```

iii. Step 6: "Code inefficiencies identified: MATLAB-style per-unit/per-trial histogram and convolution loops would be expensive in Python… Code speedups added: Uses indexed accumulation for binned counts and a vectorized FIR filter for smoothing."

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed once per session (good), and each file is opened once. But there is real repeated work that the notes do not acknowledge: `reference_conditions(flags)` is called twice per session (once only to size the `condition_means` array); `feature_names()` re-dereferences and re-decodes the camera's feature-name cell array **for every trial and every feature** even though it is constant within a session; `trajectory_group()` re-dereferences `obj/traj` on every motion-energy trial; and `referenced_at()` materialises the entire reference array (`np.asarray(dataset[()])`) on every single call in order to take one element, so per-trial access is O(n_trials) each time. `fill_nearest` is also applied twice to the paw (positions and then velocities), following the reference.

ii.
```python
condition_means = np.zeros((n_quality, len(reference_conditions(flags))), dtype=np.float64)
for condition_index, mask in enumerate(reference_conditions(flags)):
```
```python
def referenced_at(handle, dataset, index):
    values = np.asarray(dataset[()]).ravel(order="F")
    reference = values[index]
```
```python
def aligned_feature_position(handle, group, trial_index, feature, go_cue, video_offset):
    names = feature_names(handle, group, trial_index)
```

iii. Step 6 claims only "Directly reads only requested HDF5 datasets/references" and "Processes sessions sequentially to bound peak memory"; the notes do not list any repeated computation, so these repetitions are undocumented. They are cheap enough that the 34 s runtime was accepted without further optimisation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four kinds. (1) `aligned_feature_position` and `aligned_motion_energy` build a diagnostic dict (float32 copies of `aligned_frames`, `x_raw`, `y_raw`, raw motion energy) on **every** trial, but only trial 0's copy is retained. (2) Spikes are binned and smoothed over 1,100 bins and the first 100 (−3.0 to −2.5 s) are thrown away — deliberate filter warm-up, but still discarded computation. (3) Diagnostic quantities that never reach the pickle's data arrays: `n_well_isolated` (a lower-cased label match used only to compare against the paper's 214), `mean_fr_all_quality_selected`, `keep_mask`, `sample_rates`, and the per-session `tongue_class`/`paw_class`/`motion_class` copies in `diagnostics`, all kept alive until the plotting/`del` at the end of the session. (4) `haveVid` is read into `flags` and carried in diagnostics but is never used for filtering (`trial_video_is_valid` is used instead). Everything else computed — rates, time axis, six output rows — ends up in the output.

ii.
```python
    diagnostic = {
        "aligned_frames": aligned_frames.astype(np.float32),
        "x_raw": x_raw.astype(np.float32),
        "y_raw": y_raw.astype(np.float32),
    }
    return x, y, diagnostic
...
            if output_trial_index == 0:
                diagnostic_trial = {...}
```
```python
rates = rates[keep][:, :, OUTPUT_MASK]        # 1100 bins computed, 1000 kept
```
```python
normalized_quality = np.array([label.strip().lower() for label in quality_labels], dtype=object)
well_isolated = np.isin(normalized_quality, ["excellent", "great", "good", "fair", "ood"])
```

iii. The extra pre-window is justified in Step 10: "crop after filtering… crop avoids boundary artifacts". The diagnostics exist to support `--show-processing` and the consistency tables (Step 9's 521/213 unit comparison against the paper's 522/214); the notes do not note that they are computed for all trials but used for one. Step 6 does record a related saving: "Reuses one immutable input time array per session" rather than materialising 3,116 identical inputs.
