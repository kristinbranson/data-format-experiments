# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is read directly from the DANDI NWB files with a single sorted glob over `/app/data/sub-*/*.nwb` (174 files, one per session). Each file is opened once with **`h5py`** (not `pynwb`) and the needed HDF5 datasets are read by path: `general/subject/subject_id`, `units/classification`, `units/spike_times` + `units/spike_times_index`, `units/electrodes` + `units/electrodes_index`, `general/extracellular_ephys/electrodes/location`, `intervals/trials/*`, `acquisition/BehavioralEvents/{go_start_times,left_lick_times,right_lick_times}`, and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. String columns are decoded by a helper (`decode_array` / `decode_scalar`). Sessions are accumulated in file order into the output lists. Note that although every file is *opened*, only a subset of the data survives: units outside ALM are discarded (see 2-c) and sessions failing a behavioural criterion are discarded (see 1-c / 1-e), leaving 96 of 174 sessions, 23 of 28 mice and 15,845 of 69,453 QC-good units.

ii.
```python
DATA_GLOB = "/app/data/sub-*/*.nwb"
...
for path in sorted(glob.glob(DATA_GLOB)):
    session = process_session(path, region_to_index)
    if session is None:
        continue
```
```python
def process_session(path, region_to_index):
    with h5py.File(path, "r") as f:
        subject = decode_scalar(f["general/subject/subject_id"])
        classification = decode_array(f["units/classification"])
        spike_times_flat = f["units/spike_times"][()]
        spike_times_index = f["units/spike_times_index"][()]
        spike_starts = np.r_[0, spike_times_index[:-1]]
        ...
        trials = f["intervals/trials"]
        go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
```

iii. From the trajectory: the agent first inspected the NWB layout with `pynwb` (step 29: printed `units.colnames`, `trials.colnames`, `subject_id`) and confirmed the global counts (step 37: "sessions 174 all_units 272227 good_units 69453"), which match the dandiset and the white paper. It then switched to raw `h5py` path access for the converter, which is faster than materialising `pynwb` dataframes and avoids loading namespaces. The one-file-per-session layout makes the glob the complete session list.

## 1-b. How are the data split into subjects (mice)?

i. The subject of a session is the scalar `general/subject/subject_id` (e.g. `'440956'`). Subjects are collected in **order of first appearance** over the sorted file list into `data['subjects']`, and `data['subject_idx']` is the index of each retained session into that list. 23 subjects appear in the output (5 mice are lost entirely because none of their sessions pass the session filter).

ii.
```python
subject = decode_scalar(f["general/subject/subject_id"])
...
if subject not in subject_to_index:
    subject_to_index[subject] = len(data["subjects"])
    data["subjects"].append(subject)
subject_idx.append(subject_to_index[subject])
...
data["subject_idx"] = np.asarray(subject_idx, dtype=np.int64)
```

iii. The agent verified in step 29 that `nwb.subject.subject_id` is the canonical animal id and that it matches the `sub-<id>` directory name, so no separate grouping step is needed; the numeric id is used verbatim rather than the mouse name (`SC015`, …) used in the papers.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is performed. Sessions are emitted in sorted-path order (chronological within a mouse). The session identity kept in the output is the **file basename** stored in `metadata['session_info'][i]['file']`, together with subject, trial/unit counts and the behavioural-performance metrics used for the filter.

A session is **dropped** if (a) it has no `classification == 'good'` unit on an ALM probe, or (b) it fails the data paper's behavioural session-selection criterion: control-trial performance > 65 % and ≥ 50 correct left **and** ≥ 50 correct right control trials. 96 of 174 sessions survive.

ii.
```python
session_metadata = {
    "file": os.path.basename(path),
    "subject": subject,
    "n_trials_original": n_trials,
    "n_trials_kept": int(len(kept_trial_indices)),
    "n_neurons": int(keep_units.size),
    ...
}
```
```python
behavior_metrics = session_behavior_metrics(outcome, trial_instruction, early, photostim_onset)
if not behavior_metrics["passes_filter"]:
    return None
```
```python
"passes_filter": (
    n_control_trials > 0
    and performance > MIN_CONTROL_PERFORMANCE      # 0.65
    and left_correct >= MIN_CORRECT_PER_SIDE       # 50
    and right_correct >= MIN_CORRECT_PER_SIDE
),
```

iii. The file boundary is the session boundary in this dandiset, so nothing has to be inferred. For the filter, the agent quoted the methods text it had read (step 26): *"We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each."* It quantified the effect first (step 168: "106 of 174 sessions satisfy the stated `>65%` … criteria"; step 172: "96 sessions, about 15.8k good ALM units") and adopted it because it is "a better match to the reference curation and should train faster". I verified this: the implemented criterion (ignore trials counted in the performance denominator) does select exactly 106 sessions, 96 of which have good ALM units — so the implementation matches the number the agent reasoned about.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`; the number of trials is `len(intervals/trials/id)` and the per-trial go cue is taken positionally from `acquisition/BehavioralEvents/go_start_times/timestamps`, i.e. trial *i* ↔ go cue *i*. Each trial's analysis window is `[go_i - 2.525 s, go_i + 1.525 s)` (81 bin centres from -2.5 to +1.5 s, see 2-e). Spikes and camera frames are assigned to trials by `searchsorted` on the window starts, so each event is assigned to at most one trial.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
trial_start = trials["start_time"][()]
...
go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
```
```python
trial_idx = np.searchsorted(window_starts, event_times, side="right") - 1
valid = trial_idx >= 0
...
valid = event_times < window_ends[trial_idx]
```

iii. The agent dumped the event streams for a session (step 27) and observed that `go_start_times` has exactly one entry per trial (368 go cues for 368 trials) while `sample_start_times` (405) and `delay_start_times` (395) have extras because early licks replay those epochs — so the go cue is the unambiguous per-trial anchor. No explicit assertion `len(go_times) == n_trials` is made in the converter, unlike the reference. The one-window-per-event assignment is only lossless because trial windows do not overlap; I checked a sample of sessions and the minimum inter-go-cue interval is ≈ 4.8 s > the 4.05 s window, so no spikes/frames are lost in practice.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, one at session level and one at trial level:
- **Session level** (see 1-c): the data paper's behavioural criterion (> 65 % control performance, ≥ 50 correct trials per side). 78 of 174 sessions are removed by this plus the no-ALM-unit condition.
- **Trial level**: inside a retained session, a trial is kept only if **at least one kept ALM unit fired at least one spike somewhere in the 4.05 s window**. All trial *types* are kept — early-lick, `ignore`/no-response, photostimulated and `free_water` trials are all retained.

96 sessions and 51,263 trials survive (≈ 5 % of trials in retained sessions dropped by the zero-activity rule).

ii.
```python
keep_trials = np.any(rates != 0, axis=(0, 2))
if not np.any(keep_trials):
    return None
kept_trial_indices = np.flatnonzero(keep_trials)
```
```python
"notes": ("... Sessions are filtered using the behavioral criteria stated in the methods "
          "paper, but all trials within retained sessions are kept because outcome and "
          "early-lick labels are decoder targets in this task."),
```

iii. The agent read the repository's own trial mask (step 23): *"The repository only uses `early_lick==0`, `auto_water==0`, `free_water==0` as its 'regular trial' mask … Since this task explicitly asks me to decode early lick and outcome"*, so it deliberately did **not** apply that mask — the same reasoning the expert gives. The zero-activity rule was added after the validator flagged all-zero trials (step 106): *"some exported trials have zero activity across every kept ALM unit. That's almost certainly post-recording behavioral tail rather than legitimate silence, so I'm patching the converter to drop those trials."* The expert reaches the same set of trials by a different route — `units/obs_intervals` plus `free_water == 0` — i.e. it uses the file's own record of which trials were observed rather than inferring it from the absence of spikes. There is no `>= 2 trials per session` guard (the format requires two); only `>= 1` is enforced.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged buffer) with `units/spike_times_index` for the per-unit offsets, gated by `units/classification` and by the unit's brain region, which is looked up through `units/electrodes` / `units/electrodes_index` into `general/extracellular_ephys/electrodes/location` (a JSON string whose `brain_regions` field is e.g. `"left ALM"`). `acquisition/BehavioralEvents/go_start_times/timestamps` supplies the alignment times.

ii.
```python
classification = decode_array(f["units/classification"])
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
spike_starts = np.r_[0, spike_times_index[:-1]]

unit_regions = session_region_labels(f, np.arange(len(classification)))
keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
keep_units = np.flatnonzero(keep_mask)
```
```python
def session_region_labels(h5_file, units_to_keep):
    electrodes_flat = h5_file["units/electrodes"][()]
    electrodes_index = h5_file["units/electrodes_index"][()]
    electrode_starts = np.r_[0, electrodes_index[:-1]]
    electrode_locs = decode_array(h5_file["general/extracellular_ephys/electrodes/location"])
    electrode_regions = np.asarray(
        [json.loads(loc)["brain_regions"] for loc in electrode_locs], dtype=object)
    unit_regions = np.asarray(
        [electrode_regions[electrodes_flat[electrode_starts[idx]]] for idx in units_to_keep],
        dtype=object)
    return unit_regions
```

iii. Spike times are the only neural representation in the file. For the region label the agent explicitly compared the two candidates (step 33: *"whether brain-region labels should come from the unit annotation strings or the electrode-location metadata"*), inspected `units/electrodes` (step 45: unit → electrode → `'left ALM'`), and enumerated the 14 coarse probe-level labels available (step 63: `['left ALM', 'left BLA', 'left ECT', 'left Medulla', 'left Midbrain', 'left Striatum', 'left Thalamus', 'right …']`). It chose the electrode-location label (coarse, one per probe) rather than the per-unit CCF `anno_name` used by the expert. Each unit's region is taken from its **first** associated electrode.

## 2-b. How is the `neural` data processed?

i. For each kept unit, spikes are histogrammed into the 81 non-overlapping 50 ms bins of each trial window and divided by the bin width to give **firing rate in Hz**, stored as `float16`. No smoothing, normalisation or baseline subtraction. The per-session array `(n_units, n_trials, 81)` is then sliced into one `(n_units, 81)` array per kept trial.

ii.
```python
rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)
for out_idx, unit_idx in enumerate(keep_units):
    spikes = spike_times_flat[spike_starts[unit_idx] : spike_times_index[unit_idx]]
    counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
    rates[out_idx] = (counts / BIN_WIDTH).astype(np.float16)
```
```python
bin_idx = np.floor((event_times - window_starts[trial_idx]) / bin_width).astype(np.int64)
valid = (bin_idx >= 0) & (bin_idx < n_bins)
np.add.at(counts, (trial_idx[valid], bin_idx[valid]), 1)
```
```python
neural_trials = [rates[:, trial_idx, :].copy() for trial_idx in kept_trial_indices]
```

iii. This mirrors the repository's `sliding_histogram(..., rate=True)`, which returns `binSpikes / bin_width`; the agent read that function (step 9) before writing the converter. `float16` was chosen to keep the pickle small (the agent had measured that a full `float32` export would be ≈ 12 GB, step 53); firing rates here are multiples of 20 Hz and well inside `float16`'s exact-integer range, so nothing is lost.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two conditions are ANDed:
1. `units/classification == 'good'` — the spike-sorting QC classifier verdict (69,453 of 272,227 units dataset-wide).
2. The unit's probe region must be **`left ALM` or `right ALM`**; every unit from striatum, thalamus, midbrain, medulla, BLA and ECT is discarded.

A session with no unit satisfying both is dropped. After the session filter this leaves **15,845 units (23 % of the QC-good units, 2 of the 14 recorded areas)**, mean 165 per session.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
REGIONS_TO_KEEP = {"left ALM", "right ALM"}
```
```python
keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
keep_units = np.flatnonzero(keep_mask)
if keep_units.size == 0:
    return None
```
```python
"region_subset": sorted(REGIONS_TO_KEEP),
"notes": ("The NWB release contains multi-area recordings. The reference preprocessing "
          "code analyzes one region at a time, so this export keeps ALM units only to "
          "match the behaviorally central, photoinhibited region. ..."),
```

iii. For the `classification` criterion the agent's reasoning matches the expert's: it compared the two available labels (step 62: `classification good 69453` vs `unit_quality good 154948`) and checked `is_good_trials` (step 36, all-true), then used the classifier verdict with no metric thresholds.

The ALM restriction is justified in the trajectory on **size/trainability** grounds first: step 49 *"There's a scale issue with using every good neuron … the resulting pickle would be extremely large"*; step 52 measured 12.4 GB; step 57 *"not realistic for the downstream trainer … that's the only defensible way to keep the converted dataset trainable"*; step 68 measured that ALM-only gives 4.3 GB; step 72 *"constraining the exported sessions to ALM recordings so the saved dataset stays region-homogeneous and small enough for the provided decoder to train on."* The secondary argument is that the reference repository's own scripts (e.g. `medulla_population_decoding.py`, `load_session(..., area)`) process one area at a time. The agent also considered and rejected an `avg_firing_rate` threshold (step 64).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB times are on one session-absolute clock, so alignment is a window lookup: the window of trial *i* is `go_times[i] + BIN_EDGES[0]` to `go_times[i] + BIN_EDGES[-1]`, and a spike's bin is `floor((t - window_start) / 0.05)`. No resampling or interpolation. The alignment event recorded in metadata is `"Go cue onset"` with `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
```
```python
trial_idx = np.searchsorted(window_starts, event_times, side="right") - 1
...
bin_idx = np.floor((event_times - window_starts[trial_idx]) / bin_width).astype(np.int64)
```

iii. The agent confirmed in step 27 that `go_start_times` is the Go-cue tone onset and has exactly one timestamp per trial (step 28: *"the `go_start_times` field is the Go-cue tone onset"*), and the instructions require go-cue alignment. The one structural caveat, noted in 1-d, is that `searchsorted` gives each spike a single owning trial, which would silently drop spikes if two windows overlapped; empirically they never do.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins (`BIN_WIDTH = BIN_STRIDE = 0.05`), reported as `metadata['time_bin_size'] = 50.0` ms. Binning is done once from raw spike times — there is no rebinning of an already-binned signal. The bin grid is built with the repository's convention in which `begin_time`/`end_time` are the **first and last bin centres**, which yields **81 bins** with centres at -2.5, -2.45, …, +1.5 s and edges from -2.525 to +1.525 s, i.e. 4.05 s rather than the 4.0 s / 80 bins the instructions literally imply. The same grid is used for every trial and session, and for the inputs and the tongue output.

ii.
```python
def compute_bin_centers(begin_time, end_time, stride):
    span = (end_time - begin_time) / stride
    if np.allclose(span, math.floor(span) + 1):
        n_bins = math.floor(span) + 2
    else:
        n_bins = math.floor(span) + 1
    return begin_time + np.arange(n_bins) * stride

BIN_CENTERS = compute_bin_centers(BEGIN_TIME, END_TIME, BIN_STRIDE).astype(np.float32)
BIN_EDGES = np.concatenate(
    [BIN_CENTERS - BIN_WIDTH / 2.0, [BIN_CENTERS[-1] + BIN_WIDTH / 2.0]]).astype(np.float64)
N_BINS = len(BIN_CENTERS)
```

iii. `compute_bin_centers` is a transcription of `sliding_histogram` from `/app/code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py`, which the agent read in step 9 — including the docstring note *"begin_time and end_time are treated as the lower- and upper- bounds for bin centers"* and the `np.allclose` guard against floating-point error. So the extra 81st bin is a deliberate consequence of reusing the paper code's binning convention, although the agent never states this trade-off explicitly in its summary, and `off_start`/`off_end` in the metadata report the centre bounds (-2.5/+1.5) rather than the true window edges.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. **None.** The input is the constant `BIN_CENTERS` vector, i.e. time relative to the **go cue**, identical for every trial and every session. `acquisition/BehavioralEvents/sample_start_times` — the instruction-tone onsets — is never read by the converter. The input is named `time_from_tone_onset_s` and its value range in the exported file is exactly [-2.5, 1.5].

ii.
```python
input_trials.append(
    np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
)
```
```python
"input_names": ["time_from_tone_onset_s", "photostimulation_on"],
```

iii. The agent's final message states the decision: *"The 'time from tone onset' input is encoded as time from the Go-cue tone, which is the unambiguous tone aligned by this task."* It had in fact inspected `sample_start_times` in step 27 and seen 405 sample onsets for 368 trials, and noted in step 28 that *"the sample tones repeat before the delay"* — i.e. it was aware of the instruction tone and of the early-lick replay, and chose the go cue instead of resolving the ambiguity (the expert resolves it by taking the last `sample_start_times` entry before the go cue). Because the sample→go interval varies across trials (fixed 0.65 s sample + 1.2 s delay, but replayed on early-lick trials), the expert's input varies per trial while this one does not carry any trial-specific information.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. None beyond building the bin-centre grid once at import time: the same `(81,)` float32 vector is stacked as row 0 of every trial's input array. No tone time is looked up, no offset is added, no per-trial variation exists.

ii.
```python
BIN_CENTERS = compute_bin_centers(BEGIN_TIME, END_TIME, BIN_STRIDE).astype(np.float32)
...
np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
```
(compare the expert: `time_from_tone = CENTERS[None, :] + (go - tone)[:, None]`)

iii. Follows directly from 3-a: having equated "tone onset" with the go cue, time from tone onset *is* the bin centre, so no processing is needed. The agent gives no further justification.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the vector written into the input is the same `BIN_CENTERS` array from which `BIN_EDGES` — and therefore every trial's spike-binning window — is derived, so input bin *k* covers exactly the same 50 ms interval as neural bin *k*.

ii.
```python
BIN_EDGES = np.concatenate(
    [BIN_CENTERS - BIN_WIDTH / 2.0, [BIN_CENTERS[-1] + BIN_WIDTH / 2.0]]).astype(np.float64)
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
```

iii. Same as the expert's argument: one grid is defined once relative to the alignment event and reused by every stream, so there is no offset to correct and no interpolation.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` (strings, `'N/A'` on non-stimulated trials, otherwise seconds from trial start), plus `intervals/trials/start_time` and the trial's go cue to place them on the go-cue-relative axis.

ii.
```python
photostim_onset = decode_array(trials["photostim_onset"])
photostim_duration = decode_array(trials["photostim_duration"])
...
stim_series = parse_photostim_series(
    trial_start[trial_idx], photostim_onset[trial_idx],
    photostim_duration[trial_idx], go_times[trial_idx])
```

iii. In step 48 the agent cross-checked the trials-table values against the independent `BehavioralEvents/photostim_start_times` / `photostim_stop_times` streams and confirmed they agree (`table onset 1.898 dur 0.5000 … event-go [-1.1999] off-go [-0.6999]`), so it used the per-trial table columns, which are unambiguous about which trial a stimulus belongs to.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary `(81,)` time series per trial: the onset is converted to absolute time (`trial_start + onset`), the offset is `onset + duration`, both are expressed relative to the go cue, and a bin is 1 if its **centre** lies in `[rel_on, rel_off)`. `'N/A'` trials return an all-zero vector.

ii.
```python
def parse_photostim_series(trial_start, photostim_onset, photostim_duration, go_time):
    stim = np.zeros(N_BINS, dtype=np.float32)
    if photostim_onset == "N/A":
        return stim
    onset_abs = trial_start + float(photostim_onset)
    offset_abs = onset_abs + float(photostim_duration)
    rel_on = onset_abs - go_time
    rel_off = offset_abs - go_time
    stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
    return stim
```

iii. The instructions ask for "whether photostimulation is on at every time point", so a per-bin binary series is required rather than a per-trial flag; the half-open `[on, off)` test at bin centres matches the expert's implementation exactly. The `'N/A'` early return is the string-column equivalent of the expert's NaN-compares-false trick.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation bounds are converted into go-cue-relative seconds and then compared against `BIN_CENTERS`, the same grid the firing rates are binned on, so no separate alignment step is needed.

ii.
```python
rel_on = onset_abs - go_time
rel_off = offset_abs - go_time
stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
```

iii. Everything shares the session-absolute clock, so subtracting the trial's go cue puts the light onset on the same axis as the neural bins (verified against the event stream in step 48).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From the **measured lick times**: `acquisition/BehavioralEvents/left_lick_times/timestamps` and `right_lick_times/timestamps`, together with the trial's go cue. The choice is the side of the first lick in the 1.5 s response window after the go cue; if there is none, the trial is "no lick". The `outcome` / `trial_instruction` columns used by the expert are not consulted for this output.

ii.
```python
left_licks = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()]
right_licks = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()]
...
choice = first_choice_after_go(left_licks, right_licks, go_times[trial_idx])
```

iii. The agent flagged this as one of the two decisions it wanted to settle from the data (step 38: *"session inclusion and the exact lick-choice definition"*) and then printed, for `miss`/`ignore` trials, the licks around the go cue (step 39). That dump shows `ignore` trials whose only licks are *before* the go cue (early licks at -1.95 s, -2.33 s, …), which is exactly why the window is restricted to `[go, go + 1.5)`. I checked the two derivations against each other on ~3,000 trials from six sessions: the lick-based choice and the expert's `instruction × outcome` derivation agree on **100 %** of trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `searchsorted` locates the first left and first right lick in `[go, go + 1.5)`; the earlier of the two gives the code `0` (left) or `1` (right), and `2` (no lick) if neither exists. The scalar is broadcast across all 81 bins of row 0 of the trial's output array, and `output_values[0] = ['left', 'right', 'no lick']`.

ii.
```python
def first_choice_after_go(left_licks, right_licks, go_time, response_window=1.5):
    left_start = np.searchsorted(left_licks, go_time, side="left")
    left_end = np.searchsorted(left_licks, go_time + response_window, side="left")
    right_start = np.searchsorted(right_licks, go_time, side="left")
    right_end = np.searchsorted(right_licks, go_time + response_window, side="left")
    first_left = left_licks[left_start] if left_start < left_end else np.inf
    first_right = right_licks[right_start] if right_start < right_end else np.inf
    if np.isfinite(first_left) and first_left < first_right:
        return 0
    if np.isfinite(first_right):
        return 1
    return 2
```
```python
output_trials.append(np.vstack([
    np.full(N_BINS, choice, dtype=np.int64), ...]))
```

iii. The 1.5 s window is the answer period stated in the methods (*"During the response epoch (answer period: 1.5 s) mice reported the instruction by licking one of the two lick ports"*, read in step 26). The left/right coding follows the instructions, and a third class is needed for `ignore` trials. Per-trial values are repeated across bins so that all four outputs share one `(4, 81)` array. The resulting class balance in the export is 47.5 % left / 47.4 % right / 5.1 % no lick, consistent with the 5.06 % `ignore` rate.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which already contains the strings `'hit'`, `'miss'`, `'ignore'`.

ii.
```python
outcome = decode_array(trials["outcome"])
```

iii. The three stored categories are exactly the three the instructions ask for, so no derivation is needed (the agent confirmed the trials-table columns in step 29).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed mapping `ignore → 0, miss → 1, hit → 2`, broadcast across all 81 bins into row 1 of the output array; `output_values[1] = ['ignore', 'miss', 'hit']`.

ii.
```python
outcome_label = {"ignore": 0, "miss": 1, "hit": 2}[outcome[trial_idx]]
...
np.full(N_BINS, outcome_label, dtype=np.int64),
```

iii. The code order follows the instructions' listing ("ignore, miss, hit"); the value is constant within a trial so it is repeated across bins, as for the other per-trial outputs.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `intervals/trials/early_lick`, whose values are the strings `'no early'` and `'early'`.

ii.
```python
early = decode_array(trials["early_lick"])
```

iii. The flag is stored explicitly in the trials table, so nothing has to be re-derived from lick times; the agent had already seen (step 39) that these early licks fall in the sample/delay epochs, i.e. inside the -2.5 s window, which makes the label decodable from the exported neural data.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed mapping `no early → 0, early → 1`, broadcast across all 81 bins into row 2; `output_values[2] = ['no', 'yes']`.

ii.
```python
early_label = {"no early": 0, "early": 1}[early[trial_idx]]
...
np.full(N_BINS, early_label, dtype=np.int64),
```

iii. Follows the instructions' ordering (no, yes); one value per trial repeated across bins. Note that these trials are deliberately retained despite the paper excluding them from its own analyses, because early lick is a required decoder output (stated in step 23 and in the metadata `notes`).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of `data` is `tongue_y`, column 2 is the DeepLabCut `tongue_likelihood`, and `timestamps` gives the frame times on the session clock.

ii.
```python
track = np.asarray(f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"])
track_times = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"])
tongue_bins, p40, p60 = build_tongue_bins(track_times, track[:, 1], track[:, 2], go_times)
```

iii. This is the only tongue measurement in the file; the agent inspected the three columns and the likelihood distribution directly (step 47), which confirmed the `(x, y, likelihood)` layout. It also verified that all 174 sessions have tongue tracking (step 42: `sessions 174 have_tongue 174`).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with `likelihood > 0.9` count as "tongue visible"; all other frames are ignored. The 40th and 60th percentiles of the **raw visible-frame y values over the whole session** define the two class edges (stored per session in `metadata['session_info']` as `tongue_visible_y_40` / `tongue_visible_y_60`). Each visible frame inside a trial window is classified individually and written into its bin; when a bin contains several visible frames the **last one wins** (plain fancy-index assignment, no averaging or voting). Bins with no visible frame keep the initialised value 3.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
...
visible = track_prob > VISIBILITY_THRESHOLD
y_visible = track_y[visible]
p40, p60 = np.percentile(y_visible, [40, 60])
...
classes = np.full(len(ys), 1, dtype=np.int64)
classes[ys < p40] = 0
classes[ys > p60] = 2
flat = tongue_bins.reshape(-1)
flat[trial_idx * N_BINS + bin_idx] = classes
```
```python
"tongue_percentiles_computed_over": "all visible side-view tongue frames within session",
```

iii. The agent measured the likelihood distribution before picking the threshold (step 47): the values are essentially binary (median ≈ 6e-5, 90th percentile ≈ 1.0), and it printed the y-percentiles at both a 0.9 and a 0.5 cut-off — `[273.94, 288.70]` vs `[273.70, 288.53]` — confirming the threshold barely matters. Taking percentiles over frames is a literal reading of the instruction ("percentile of y-position over the session"); the expert instead takes percentiles of the 50 ms bin means, arguing that the edges should be defined on the same quantity that is discretised. The within-bin "last visible frame wins" rule is not documented anywhere in the code or metadata; with a ~300 Hz camera it discards ~14 of the ~15 frames in a bin. In the exported data the visible classes come out at 60 % / 13 % / 27 % of visible bins rather than the 40/20/40 the percentiles would suggest.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the four classes the instructions specify: `0` = y < 40th percentile, `1` = 40th–60th percentile, `2` = y > 60th percentile, `3` = not visible (no frame with likelihood > 0.9 in that bin). Percentiles are computed **per session**. The array is pre-filled with 3 so "not visible" is the default.

ii.
```python
tongue_bins = np.full((len(go_times), N_BINS), 3, dtype=np.int64)
...
classes = np.full(len(ys), 1, dtype=np.int64)
classes[ys < p40] = 0
classes[ys > p60] = 2
```
```python
"output_values": [..., ["<40th percentile", "40th-60th percentile", ">60th percentile", "not visible"]],
```

iii. Directly implements the instruction's discretisation, including the per-session scope, and adds the required fourth "not visible" class. If a session has no visible frame at all the function returns the all-3 array with NaN edges rather than failing. 73 % of all bins end up in class 3, close to the expert's 75 %.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the session-absolute clock with the spikes and events, so frames are assigned to trials and bins with the same `searchsorted` / `floor` scheme used for spikes: `trial = searchsorted(window_starts, t) - 1`, then `bin = floor((t - window_start) / 0.05)`, with the same `window_starts = go + BIN_EDGES[0]`. Bin *k* of the tongue output therefore covers the same interval as bin *k* of the firing rates.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
trial_idx = np.searchsorted(window_starts, track_times, side="right") - 1
valid = trial_idx >= 0
...
valid = times < window_ends[trial_idx]
...
bin_idx = np.floor((times - window_starts[trial_idx]) / BIN_WIDTH).astype(np.int64)
valid = (bin_idx >= 0) & (bin_idx < N_BINS)
```

iii. No interpolation or offset correction is needed because the video timestamps are on the global clock. As with the spikes, a frame can be attributed to only one trial window, which is safe only because windows never overlap (min inter-go interval ≈ 4.8 s vs a 4.05 s window). The tongue classes are computed for **all** trials of the session and only afterwards indexed by the kept trials.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases, all handled by exclusion or by an explicit category rather than imputation:
- **Session never quality-controlled** (`units/classification` stored as all-NaN floats): `decode_array` returns the float array, the `== "good"` comparison yields no matches, `keep_units` is empty and the session returns `None`. I confirmed this on `sub-440958_ses-20190216T162508`, the same session the expert drops.
- **Trials with no ephys** (recording started late, `free_water` trials): removed by the all-zero-activity rule (1-e).
- **Non-stimulated trials**: `photostim_onset == 'N/A'` returns an all-zero stimulation series instead of raising on `float('N/A')`.
- **Untracked tongue**: frames below the likelihood threshold are dropped; bins left with no visible frame become class 3; a session with no visible frame at all yields all-3 with NaN edges.

There is no assertion that `len(go_times) == n_trials`, and no guard that a session retains at least 2 trials (the format requires two); only `>= 1` is enforced.

ii.
```python
def decode_array(dataset):
    arr = np.asarray(dataset)
    if arr.dtype.kind == "S":
        return arr.astype(str)
    if arr.dtype == object:
        ...
    return arr
```
```python
if keep_units.size == 0:
    return None
...
if not np.any(keep_trials):
    return None
```
```python
if photostim_onset == "N/A":
    return stim
...
visible = track_prob > VISIBILITY_THRESHOLD
if not np.any(visible):
    return tongue_bins, np.nan, np.nan
```

iii. Where nothing was recorded the unit/trial/session is excluded rather than emitted as zeros; where the measurement legitimately has no value (retracted tongue, no photostimulation) it is represented as an explicit category or as zero. The all-zero-trial rule was added in direct response to validator warnings (step 106), and the agent re-ran the validator afterwards to confirm the warnings disappeared (step 137: *"The export now validates cleanly with no format warnings."*).

## 10-a. What are the most time-consuming steps of the code?

i. The code carries no instrumentation and the agent never profiled it, but from the trajectory the conversion took several minutes per full pass and was CPU-bound (steps 85, 188: *"compute-bound rather than stuck"*, *"CPU-bound and still active"*). The dominant costs are:
1. Reading the ragged `units/spike_times` buffer in full (`f["units/spike_times"][()]`, up to ~11.5 M doubles per session) — and note this happens **before** the session-level behavioural filter, so it is paid for all 174 sessions even though 78 are then discarded.
2. The per-unit binning loop: for each unit, a `searchsorted` over the window starts plus `np.add.at`, which is an unbuffered scatter-add and is several times slower than a `searchsorted`-and-difference formulation.
3. Per-unit Python/`json` parsing of the electrode location strings for *all* units in the session, not just the ALM good ones.
4. The per-trial Python loop that builds `np.vstack` input/output arrays for ~550 trials per session.
5. Pickling the 1.55 GB result.

ii.
```python
spike_times_flat = f["units/spike_times"][()]          # full read, before the session filter
...
behavior_metrics = session_behavior_metrics(...)
if not behavior_metrics["passes_filter"]:
    return None
```
```python
np.add.at(counts, (trial_idx[valid], bin_idx[valid]), 1)
```

iii. No justification is offered in the trajectory; the agent monitored memory and throughput (steps 85, 92, 118, 130) but accepted the runtime as long as it finished, and the export did complete and train.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
1. **Per-unit spike binning.** The loop itself is hard to remove (spike arrays are ragged), but its body could be: replacing `np.add.at` with `np.searchsorted` on a flattened edge array followed by `np.diff` — the expert's formulation — would give the same counts much faster.
2. **Per-trial output/input construction.** `stim_series`, `choice`, the three label lookups and the `np.vstack` are all done one trial at a time in Python; the expert builds `(n_trials, n_output, n_bins)` arrays for the whole session in a handful of vectorised statements, and only slices per trial at the end.
3. **Per-unit region lookup.** `session_region_labels` uses a Python list comprehension with a `json.loads` per electrode and then a second comprehension per unit; the per-electrode parse could be done once and indexed with fancy indexing, and it only needs to run for the `classification == 'good'` units.

ii.
```python
for out_idx, unit_idx in enumerate(keep_units):
    spikes = spike_times_flat[spike_starts[unit_idx] : spike_times_index[unit_idx]]
    counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
    rates[out_idx] = (counts / BIN_WIDTH).astype(np.float16)
```
```python
for trial_idx in kept_trial_indices:
    stim_series = parse_photostim_series(...)
    input_trials.append(np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False))
    choice = first_choice_after_go(left_licks, right_licks, go_times[trial_idx])
    ...
```
```python
unit_regions = np.asarray(
    [electrode_regions[electrodes_flat[electrode_starts[idx]]] for idx in units_to_keep],
    dtype=object)
```

iii. Not discussed by the agent. The tongue path, by contrast, *is* fully vectorised across trials and frames — the agent wrote a single flat-index scatter there — so the remaining loops appear to be convenience rather than necessity.

## 10-c. What processing does the code repeat multiple times?

i. Little is recomputed across sessions (each file is opened once, and the bin grid is built once at module import), but within a session there is some duplication:
- `window_starts` / `window_ends` are computed twice, once in `build_tongue_bins` and again in `process_session`.
- `session_region_labels` re-derives the electrode → region mapping for every unit including the ~75 % that fail the `classification` test, and re-parses the JSON location string for every electrode.
- `first_choice_after_go` runs four `searchsorted` calls per trial over the full session-length lick arrays, instead of one vectorised pass over all trials.
- The dict literals `{"ignore": 0, "miss": 1, "hit": 2}` and `{"no early": 0, "early": 1}` are rebuilt on every trial iteration.

ii.
```python
window_starts = go_times + BIN_EDGES[0]     # in build_tongue_bins
window_ends = go_times + BIN_EDGES[-1]
...
window_starts = go_times + BIN_EDGES[0]     # again in process_session
window_ends = go_times + BIN_EDGES[-1]
```
```python
outcome_label = {"ignore": 0, "miss": 1, "hit": 2}[outcome[trial_idx]]
early_label = {"no early": 0, "early": 1}[early[trial_idx]]
```

iii. Not discussed. None of this is expensive relative to the spike I/O, and the per-session tongue percentiles are correctly computed inside the single pass rather than requiring a second pass over the data.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things are computed and then thrown away:
- **The whole spike-time buffer of rejected sessions.** `f["units/spike_times"][()]` (and the classification/electrode arrays) is read before `session_behavior_metrics` decides the session fails; 78 of 174 sessions are dropped after that read.
- **Firing rates and tongue classes for dropped trials.** `rates` is built for all `n_trials`, then `keep_trials` selects a subset; `build_tongue_bins` likewise classifies every trial in the session before the filter is known.
- **Region labels for non-good units** — `session_region_labels(f, np.arange(len(classification)))` runs over all ~1,900 units per session when only the few hundred `good` ones can ever be used.
- **`p40`/`p60` and `tongue_visible_*` metadata** for sessions that are subsequently dropped, and the unused local `probs` slice inside `build_tongue_bins`.
- **Storage**: outputs are stored as `int64` where the four label sets need at most 2 bits (the expert uses `int8`), i.e. ~8× more bytes than necessary for that field.

ii.
```python
spike_times_flat = f["units/spike_times"][()]      # read before the session filter runs
...
rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)   # all trials
keep_trials = np.any(rates != 0, axis=(0, 2))                             # then subset
```
```python
unit_regions = session_region_labels(f, np.arange(len(classification)))   # all units
```
```python
np.full(N_BINS, choice, dtype=np.int64),   # int64 for 3-value labels
```

iii. Not discussed by the agent. None of it changes the exported values; the notable one is the ordering issue — moving the behavioural session check ahead of the spike-buffer read would have cut roughly 45 % of the file I/O in the final pass at no cost.
