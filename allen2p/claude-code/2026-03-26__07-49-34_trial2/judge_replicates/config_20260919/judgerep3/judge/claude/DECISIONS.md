# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK loader classes. It reads the local NWB (HDF5) files directly with `h5py`, and uses the project metadata CSV `project_metadata/ophys_experiment_table.csv` to enumerate experiments and attach metadata (`mouse_id`, `targeted_structure`, `cre_line`, `passive`).

Concretely:
- Glob all `*.nwb` in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` (284 files), parse the `ophys_experiment_id` out of each filename.
- Intersect with the experiment table, then keep only rows with `passive == False` → **202 experiments from 38 mice**.
- **No filter on `project_code`** is applied, so both `VisualBehavior` (168 active) and `VisualBehaviorMultiscope` (34 active) experiments are included.
- Each retained experiment is opened once in Pass 1 (to accumulate running/pupil values for the global percentile edges) and a second time in Pass 2 (to build the actual per-trial arrays). A third read of up to 10 files happens earlier to build the image-name vocabulary.
- Within each file it reads: `processing/ophys/dff/traces/{data,timestamps}`, `processing/ophys/image_segmentation/cell_specimen_table/valid_roi`, the natural-images `intervals/<stim_key>` table, `intervals/trials`, `processing/running/speed`, and `acquisition/EyeTracking/*`.

ii.
```python
NWB_DIR = 'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/'
METADATA_DIR = 'data/visual-behavior-ophys-1.1.0/project_metadata/'

def get_experiment_metadata():
    """Load experiment metadata and identify available NWB files."""
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    nwb_map = {}
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        nwb_map[eid] = f

    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    # Only active behavior sessions
    active_exps = our_exps[our_exps['passive'] == False].copy()
    active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
    return active_exps, nwb_map
```

```python
with h5py.File(nwb_path, 'r') as f:
    dff_data = f['processing']['ophys']['dff']['traces']['data'][:]      # (timepoints, neurons)
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    ...
    trials = f['intervals']['trials']
    running_speed = f['processing']['running']['speed']['data'][:]
    running_ts    = f['processing']['running']['speed']['timestamps'][:]
```

iii. From CONVERSION_NOTES.md and the trajectory: the AI first mapped the SDK's loader chain (`BehaviorOphysExperiment.from_nwb_path`, `DFFTraces.from_nwb`, `RunningSpeed.from_nwb`, `EyeTrackingTable.from_nwb`, `Trials.from_nwb`, `Presentations.from_nwb`, `CellSpecimens(exclude_invalid_rois=True)`) and recorded exactly which NWB paths each of those reads, then reproduced those reads directly with `h5py` "for speed". Step 10 Check 3 documents the equivalence claim ("h5py direct NWB read ≡ `BehaviorOphysExperiment.from_nwb_path()`"). Active-only was justified as "Passive sessions have no meaningful trial outcomes (no licking)" — the decoder task requires a hit/miss/FA/CR label. The AI never states a rationale for including the Multiscope experiments; it simply worked from "the 284 NWB files we have".

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained (active) experiments, cast to `str` and sorted. `subject_idx` for each session is the index of that experiment's mouse into this sorted list. Result: 38 subjects.

ii.
```python
subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
...
subject_idx = subjects_list.index(result['mouse_id'])
all_subject_idx.append(subject_idx)
...
'subjects': subjects_list,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```
with `'mouse_id': str(exp_info['mouse_id'])` set inside `process_experiment`.

iii. `mouse_id` is the canonical per-animal identifier in the Allen experiment table; the AI cross-checked the count against the whitepaper (82 mice in the full release vs 38 in the 284-file subset) and recorded the discrepancy in Step 4 as "our 284 NWB files are a subset of the full dataset".

## 1-c. How are the data split into sessions?

i. **Each NWB experiment (= one imaging plane) is treated as one "session".** There is no grouping by `ophys_session_id`. For the 168 single-plane `VisualBehavior` experiments this is a 1:1 mapping. For the 34 `VisualBehaviorMultiscope` experiments, however, those 34 planes come from only **6 real ophys sessions of a single mouse (457841)**, so that mouse appears with 34 "sessions" and the *same* behavioral trials are emitted 3–7 times over, each time paired with a different plane's neurons. Sessions are ordered by `ophys_experiment_id`.

ii.
```python
active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
...
for idx, (_, row) in enumerate(active_exps.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = nwb_map[eid]
    result = process_experiment(nwb_path, row, collect_stats_only=False)
    ...
    all_sessions_neural.append(result['neural_trials'])
    all_sessions_input.append(result['input_trials'])
    all_sessions_output.append(session_output)
```
The visible signature of the duplication in `verification_full_out.txt`:
```
Subject 457841: 34 sessions
... trials per session: ... 209, 209, 209, 209, 209, 209, 209, ... 287 x7, 309 x7, 239 x5, 196 x5 ...
```

iii. The decision is stated only in the agent's internal reasoning (step 31): *"Since each NWB file contains its own set of neurons, I should treat each experiment as a separate session for the decoder rather than grouping by ophys_session."* CONVERSION_NOTES.md records "Each NWB file = one imaging plane from one session" and then reports "Sessions: 202", i.e. it silently equates experiments with sessions. At step 79 the AI noticed "one subject (457841) has 34 sessions … this might be because MESO sessions with multiple imaging planes get split into separate experiments", but did not act on it or document it.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if `(go | catch) & ~aborted & ~auto_rewarded`. The trial's temporal extent is **not** taken from `start_time`/`stop_time`; instead it is the set of stimulus presentations whose `trials_id` equals the trial id. Each such presentation becomes one 750 ms time bin, so trials have variable length (10–17 bins, mean 11.6 ≈ 8.7 s). Trials with zero matching presentations are dropped.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
valid_trial_ids = trial_ids[trial_mask]
...
for trial_idx in np.where(trial_mask)[0]:
    tid = trial_ids[trial_idx]
    # Find stimulus presentations for this trial
    trial_stim_indices = np.where(stim_trials_id == tid)[0]
    if len(trial_stim_indices) == 0:
        continue
    n_bins = len(trial_stim_indices)
```

iii. CONVERSION_NOTES Step 4/5: "Include Go+Catch only per task instructions"; the trial window is built from the stimulus presentations because the AI chose one time bin per image flash, which "naturally aligns with the task structure" and makes the pre-change flashes, the change flash and the post-change response window all explicit time bins. It verified that 39/279/239 trials per session matched a direct count from the raw NWB trials table (Step 10, Check 2).

## 1-e. How are trials filtered based on quality controls?

i. Filtering applied:
- Trial level: drop `aborted` and `auto_rewarded`; keep only `go | catch`; drop a trial with no stimulus presentations.
- Session/experiment level: drop an experiment with 0 valid ROIs; drop an experiment whose natural-images stimulus table cannot be found; drop an experiment yielding `< 2` trials.
- No session-level behavioural QC (e.g. d′ ≥ 1, engagement) is applied, and passive sessions were already removed at load time.
In the full run, 0 experiments were skipped; the smallest session had 39 trials.

ii.
```python
trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
...
if n_neurons == 0:
    print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
    return None
...
stim_key = find_stim_key(f)
if stim_key is None:
    print(f"  WARNING: No stimulus presentations found for {experiment_id}, skipping")
    return None
...
if result is None or result['n_trials'] < 2:
    skipped += 1
    reason = 'no result' if result is None else f'only {result["n_trials"]} trials'
    print(f"  Skipped experiment {eid}: {reason}")
    continue
```

iii. The exclusion of aborted/auto-rewarded trials is taken verbatim from the task instructions (aborted = mouse licked before the change so no change was shown; auto-rewarded = free reward biases behaviour). The ≥ 2 trial rule comes from the format spec ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). The AI noted the whitepaper's session-level QC gates (z-drift, d′ ≥ 1, etc.) in Step 3 but treated them as already applied by the Allen pipeline to the released data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Pre-computed ΔF/F traces: the HDF5 dataset `processing/ophys/dff/traces/data` (shape `(n_timepoints, n_rois)`, float64) with its companion `timestamps` (the ophys frame times). Cell validity comes from `processing/ophys/image_segmentation/cell_specimen_table/valid_roi`. Calcium `events` were available in the NWB but deliberately not used.

ii.
```python
dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]

cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
dff_valid = dff_data[:, valid_roi]
```

iii. Step 1 of the notes: "dF/F is pre-computed in NWB files — no need to compute from raw fluorescence" (baseline via 600 s median filter, neuropil-corrected, detrended). Step 4 records the deliberate divergence from Piet et al., who use L0-deconvolved calcium events: "Use dF/F – standard for decoding, pre-computed"; Step 5 Key Decision 2 adds "events are sparser and may not decode as well for time-varying outputs".

## 2-b. How is the `neural` data processed?

i. The only processing is **temporal averaging into 750 ms bins**. For each stimulus presentation `i` of a trial, the bin value of each neuron is the mean of its ΔF/F over all ophys frames with timestamp in `[stim_start[i], stim_start[i] + 0.75)`. This is done with a precomputed cumulative sum over time so each bin is an O(1) difference. No z-scoring, smoothing, baseline subtraction, or normalisation is applied, and traces from different planes are never merged (each plane is its own "session").

ii.
```python
bin_duration = TIME_BIN_MS / 1000.0  # 0.75s
all_stim_starts = stim_start
all_stim_ends   = all_stim_starts + bin_duration
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends,   side='left')

dff_cumsum = np.cumsum(dff_valid, axis=0)
dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])
...
neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
```

iii. The AI's rationale (Step 5 Key Decision 1) is that ΔF/F is already fully processed by the Allen pipeline, so the only transformation needed is to put every recording on a common time base; averaging within the stimulus period is the natural way to do that. The cumsum formulation is documented in Step 6 as an efficiency optimisation ("avoids per-bin boolean masking").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `valid_roi` boolean column of the cell-specimen table is used: columns of the ΔF/F matrix with `valid_roi == False` are dropped. No SNR, event-rate, or activity-based filtering is added. (Empirically this is a no-op on the released NWB files — in every file checked, `valid_roi` is `True` for all ROIs, because the Allen release already stores only valid cells — so all `dff` columns survive.)

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][:].astype(bool)
n_total_rois = dff_data.shape[1]

dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
n_neurons = dff_valid.shape[1]
```

iii. Step 1 of the notes identifies `CellSpecimens.__init__(exclude_invalid_rois=True)` as the SDK's curation step and Step 3 lists exactly what `valid_roi` removes (ROI unions, >70 % duplicates, edge ROIs, apical dendrites, too small/narrow/dim, ghost/crosstalk cells, negative traces). Key Decision 4: "Use only cells marked valid_roi=True in NWB, matching SDK default behavior." Beyond that the AI treats the Allen segmentation/classification pipeline as the quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything is aligned on the **ophys timestamps**, as the instructions require. Alignment is done at the *bin* level rather than at the trial level: bin `i` of a trial starts at the onset of the `i`-th stimulus presentation of that trial and spans 750 ms forward; `np.searchsorted` on `ophys_timestamps` converts those absolute times to frame indices. Consequently the first bin of a trial is the first image flash of that trial (≈ the trial's `start_time`), and the change flash always lands on its own bin. `metadata['temporal_alignment_event']` is set to `'Stimulus presentation onset (each 750ms image flash)'` with `off_start = off_end = None` (trials are variable length).

ii.
```python
ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
ophys_bin_ends   = np.searchsorted(ophys_ts, all_stim_ends,   side='left')
...
s, e = ophys_bin_starts[si], ophys_bin_ends[si]
neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / (e - s)
```
```python
'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
'off_start': None,
'off_end': None,
```

iii. Rationale in Step 4/5: all data streams in this dataset are hardware-synchronised (NI PCI-6612 at 100 kHz), so absolute times are directly comparable; the ophys frame times are used as the reference clock. Anchoring each bin to a stimulus onset guarantees that neural, image-identity and image-change streams cannot drift relative to one another, and the AI verified this in Step 10 Check 2 by recomputing bin means from the raw NWB (`np.allclose`, atol = 1e-5).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **750 ms per bin — one bin per stimulus presentation (250 ms image + 500 ms grey)**, identical for every trial and session. Yes, substantial rebinning is applied: the native ophys rate is ~30.94 Hz (32.3 ms) for the CAM2P single-plane rigs and ~10.73 Hz (93.2 ms) for the MESO.1 mesoscope, and all frames inside each 750 ms window are averaged. A trial therefore has ~11.6 bins instead of ~90–260 frames. `metadata['time_bin_size'] = 750.0`.

ii.
```python
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
...
'time_bin_size': TIME_BIN_MS,
```

iii. Step 4 and Step 5 Key Decision 1: the format spec demands "Time bins should be the same size for all trials and sessions", but the dataset mixes two acquisition rates (the AI measured 10.73 Hz vs 30.94 Hz directly). It rejected native timestamps (bin size would differ per session) and rejected resampling to 11 Hz or 30 Hz (up/down-sampling artefacts), settling on the stimulus-presentation interval because it is "a natural task unit, consistent across all equipment types (MESO/CAM2P), aligned to image presentations", was confirmed in the data (mean ISI 750.6 ms), and makes each task variable (image identity, change) exactly one value per bin.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The `image_name` column of the natural-images stimulus-presentations table (`intervals/<stim_key>/image_name`), restricted to the presentations belonging to the trial (`trials_id`). The `omitted` presentations carry the literal string `'omitted'` in that column and are handled separately. The trials table's `initial_image_name`/`change_image_name` are *not* used.

ii.
```python
stim = f['intervals'][stim_key]
stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in stim['image_name'][:]])
stim_trials_id = stim['trials_id'][:]
...
images = stim_image_name[trial_stim_indices].copy()
```

iii. The AI chose the per-flash stimulus table rather than the per-trial summary because it had already decided that one bin = one flash, so the stimulus table gives the image on screen for each bin directly, with no need to reason about where `change_time` falls. Step 4 records the cross-check "8 unique image names + 'omitted' per session", "8 natural scene images" in the papers.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Two steps: (a) `'omitted'` flashes are filled with the previous flash's image name (or, if the trial begins with an omission, with the next non-omitted name); (b) names are mapped to integer codes through a single global vocabulary. The vocabulary is built by opening **only the first 10 experiments** (sorted by experiment id), collecting their image names, dropping `'omitted'`, and sorting; the full run produced 16 names (image sets A and B), matching the reference's codebook exactly. Names not in the vocabulary silently map to code 0.

ii.
```python
def collect_all_image_names(active_exps, nwb_map):
    """Collect all unique image names across experiments (sample a few to be fast)."""
    all_images = set()
    # Sample up to 10 experiments to collect image names (they use the same 8 images)
    sample_exps = active_exps.head(min(10, len(active_exps)))
    for _, row in sample_exps.iterrows():
        ...
            for n in names:
                name = n.decode() if isinstance(n, bytes) else str(n)
                if name != 'omitted':
                    all_images.add(name)
    return sorted(all_images)
```
```python
img_to_idx = {name: i for i, name in enumerate(IMAGE_NAMES_GLOBAL)}
...
# Forward-fill omitted presentations
for bi in range(len(images)):
    if images[bi] == 'omitted':
        if bi > 0:
            images[bi] = images[bi - 1]
        else:
            for bj in range(bi + 1, len(images)):
                if images[bj] != 'omitted':
                    images[bi] = images[bj]
                    break

image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
```

iii. Key Decision 5: "Omitted stimuli: Forward-fill image identity from previous presentation. Neural and behavioral data still recorded during omissions" — i.e. an omission is a missing flash, not a change of stimulus context, so the last-seen image is the best label. The global (rather than per-session) codebook was justified in the trajectory because "the decoder needs consistent output dimensions" across sessions that use different image sets. Sampling only 10 files was an explicit speed optimisation ("Sample 10 experiments for image name collection (not all 202)"), with the justification that all sessions draw from the same small image sets; the AI verified post-hoc that 16 names were recovered.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. By construction: the image code for bin `i` is the image of the very stimulus presentation whose onset defines bin `i` of the neural matrix. The same `trial_stim_indices` array indexes both, so the two streams share one index space and cannot be misaligned.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
n_bins = len(trial_stim_indices)
for bi, si in enumerate(trial_stim_indices):          # neural
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / (e - s)
images = stim_image_name[trial_stim_indices].copy()   # image identity, same index array
```

iii. Choosing the stimulus presentation as the time bin was motivated precisely by this: "There are no temporal misalignments" was one of the things the `--show-processing` plots were meant to demonstrate, and Step 10 Check 2 re-derived image identity per bin from raw NWB for three sessions and found exact agreement.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` column of the stimulus-presentations table, subset to the trial's presentations. `is_change` is the SDK's flag that the image on this flash differs from the previous one, so it is `True` exactly once per go trial and never on a catch (sham-change) trial. The trials table's `change_time`/`go` columns are read but not used for this output.

ii.
```python
stim_is_change = stim['is_change'][:]
...
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Step 1 of the notes flags `is_change_event()` in `stimulus_processing.py` as the SDK function that defines change events, and Step 5's mapping table records "is_change → output[1], Binary (0/1), 1 at change flash only". Using the presentation-level flag means the change marker inherits the same bin grid as everything else.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Essentially none: NaN (the first presentation of a session has an undefined `is_change`) is coerced to 0 and the boolean is cast to `int64`. No smoothing or window expansion; the indicator is 1 for exactly one 750 ms bin in every go trial and 0 everywhere in catch trials. Result over the full dataset: 7.5 % of bins are `change`.

ii.
```python
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. The AI's own consistency check (Step 10, Check 4) is the justification: "Go trials all have change — 45,477 with change, 0 without; Catch trials no change — 0 with change, 6,515 without", and "Go/Catch split 87.5 %/12.5 %, exact match with the expected 7/8 and 1/8 of the stimulus transition matrix".

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is natively binary. `output_values[1] = ['no_change', 'change']`, values 0/1.

ii.
```python
change_value_names = ['no_change', 'change']
...
'output_values': [
    image_value_names,
    change_value_names,
    ...
]
```

iii. The instructions specify "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0", and `is_change` is already exactly that indicator at the resolution of one stimulus presentation.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: the flag for bin `i` is `is_change` of the presentation that defines bin `i`, indexed by the same `trial_stim_indices`. Because a neural bin spans `[change_onset, change_onset + 750 ms)`, the `change == 1` bin contains exactly the evoked response to the changed image plus the following grey period.

ii.
```python
trial_stim_indices = np.where(stim_trials_id == tid)[0]
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
```

iii. Same as 3-c — bin-level alignment is guaranteed by construction, and was spot-checked against raw NWB in Step 10 Check 2 ("Change Match: True" for sessions 0, 50, 150).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` and `.../timestamps` — the SDK's *filtered* running speed (10 Hz low-pass Butterworth), sampled at ~60 Hz. The unfiltered variant is not used.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][:]
running_ts = f['processing']['running']['speed']['timestamps'][:]
```

iii. Step 1 identifies `RunningSpeed.from_nwb()` as the SDK accessor and Step 3 notes the filtering that the Allen pipeline applies ("10 Hz lowpass Butterworth filter, z-score ≥ 10 transients set to NaN"); Step 10 Check 3 records "Running speed: Filtered speed from NWB ≡ RunningSpeed.from_nwb() — same source".

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Within-bin averaging, then global percentile discretisation:
1. For each stimulus presentation, average all running samples with timestamp in `[stim_start, stim_start + 0.75)` (O(1) via a cumulative sum over the speed trace).
2. Pass 1 pools these binned values over **all** 202 experiments and computes 5 equal-percentile edges with the outer edges replaced by ±inf.
3. Pass 2 assigns each bin to a level with `np.digitize`.

ii.
```python
run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
run_bin_ends   = np.searchsorted(running_ts, all_stim_ends,   side='left')
run_cumsum = np.concatenate([[0], np.cumsum(running_speed)])
...
running_binned = np.zeros(n_bins, dtype=np.float32)
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
    n_pts = e - s
    if n_pts > 0:
        running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts
```

iii. Averaging (rather than point-sampling) is the natural downsample to the 750 ms grid and matches how the neural data is binned, so the two streams describe the same interval. Pooling across the whole dataset before computing percentiles is Key Decision 7 ("Compute percentile bin edges across ALL valid time points in dataset (2-pass), then assign bins"), which keeps the category definitions identical across sessions and gives globally balanced classes (verified: 0.200 per level).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (quintiles) over the pooled binned speeds. `discretize_values` computes `np.percentile(valid, [0,20,40,60,80,100])`, then overwrites the first/last edge with ∓inf and uses the 4 interior edges in `np.digitize`, clipping to `[0, 4]`. Global edges were `[-inf, 0.0041, 0.788, 15.82, 33.13, inf]` cm/s. Level names `speed_q1 … speed_q5`.

ii.
```python
def discretize_values(values, n_bins, bin_edges=None):
    if bin_edges is None:
        valid = values[~np.isnan(values)]
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(valid, percentiles)
        # Ensure unique edges
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)  # 0 to n_bins-1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned, bin_edges
```

iii. Directly from the instructions: "Running speed, discretized into five equal percentile bins". The ±inf outer edges are a defensive measure so that no value can fall outside the defined range, and the AI confirmed the resulting distribution was exactly uniform (Step 7/Step 9: "Running speed bins: 20 % each").

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The averaging window for bin `i` is the *same* absolute time interval `[stim_start[i], stim_start[i] + 0.75)` used for the neural bin, with `np.searchsorted` applied to the running clock instead of the ophys clock. The two streams therefore describe identical wall-clock intervals; no interpolation or lag correction is applied.

ii.
```python
all_stim_starts = stim_start
all_stim_ends   = all_stim_starts + bin_duration          # shared by all streams
ophys_bin_starts = np.searchsorted(ophys_ts,   all_stim_starts, side='left')
run_bin_starts   = np.searchsorted(running_ts, all_stim_starts, side='left')
```

iii. Step 3: "Temporal alignment: All data streams synchronized via NI PCI-6612 at 100 kHz", so the running timestamps and ophys timestamps live on one common clock and windows can be compared directly. Step 10 Check 2 re-derived the running discretisation from raw NWB for three sessions and found exact agreement.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (the fitted pupil-ellipse area, ~30 Hz) with its `timestamps`, plus `acquisition/EyeTracking/likely_blink/data` as the blink mask. Diameter is computed from area rather than taken from a width/height field. If the `EyeTracking` group is absent (3 of 202 experiments) the session has no pupil signal at all.

ii.
```python
has_eye_tracking = 'EyeTracking' in f['acquisition']
if has_eye_tracking:
    pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
    pupil_ts       = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
    likely_blink   = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)
```

iii. Step 1/Step 3: the SDK's `EyeTrackingTable.from_nwb()` plus `determine_likely_blinks()` (z-score threshold 3.0, 2-frame dilation) define the blink QC, which the AI reproduces by using the stored `likely_blink` flag. Step 4's resolution row: "Pupil tracking … Use area → compute diameter, interpolate NaNs" — area is the ellipse-fit quantity and gives an isotropic diameter estimate.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pipeline: blink frames → NaN; non-positive areas → NaN; area → diameter via `d = 2·sqrt(area/π)`; linear interpolation over the NaN gaps (in sample-index space, with end-clamping); average within each 750 ms stimulus bin (NaN-aware cumsum, dividing by the count of valid samples); then global 5-quantile discretisation with edges pooled across all experiments in Pass 1. A trial whose bins are still all-NaN gets the middle level.

ii.
```python
pupil_area = pupil_area_raw.copy().astype(float)
pupil_area[likely_blink] = np.nan
# Also NaN out negative or zero values
pupil_area[pupil_area <= 0] = np.nan
# Compute diameter from area: d = 2*sqrt(area/pi)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
# Interpolate NaNs
pupil_diameter = interpolate_nans(pupil_diameter)
```
```python
pup_valid = ~np.isnan(pupil_diameter)
pup_filled = np.where(pup_valid, pupil_diameter, 0.0)
pup_cumsum       = np.concatenate([[0], np.cumsum(pup_filled)])
pup_count_cumsum = np.concatenate([[0], np.cumsum(pup_valid.astype(np.float64))])
...
n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
if n_valid > 0:
    pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid
```

iii. Key Decision 6: "Pupil NaN handling: Linear interpolation for blink frames before computing diameter and averaging" — blinks are artefacts, not real pupil constrictions, so filling them preserves the slow arousal signal that is actually of interest, while the ±inf-edged quantile binning makes the absolute scale (pixels, camera-dependent) irrelevant. Zero/negative areas are treated as failed ellipse fits.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same `discretize_values` routine as running speed: five equal-percentile bins over the pooled non-NaN binned diameters of all 202 experiments (edges `[-inf, 73.88, 83.84, 92.87, 105.38, inf]`), `np.digitize` + clip. Missing data are imputed rather than dropped: if a trial's pupil bins are entirely NaN (i.e. the session has no eye-tracking group), every bin gets the **middle** level `N_PERCENTILE_BINS // 2 = 2`; if only some bins are NaN they are linearly interpolated first. Resulting distribution: 0.196 / 0.196 / **0.214** / 0.196 / 0.196 — the excess in level 2 is the imputed no-eye-tracking sessions.

ii.
```python
valid_pupil = all_pupil[~np.isnan(all_pupil)]
if len(valid_pupil) > 0:
    _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
else:
    pupil_bin_edges = np.array([-np.inf, 0, 1, 2, 3, np.inf])
...
nan_mask = np.isnan(pupil_raw)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
else:
    if nan_mask.any():
        pupil_raw = interpolate_nans(pupil_raw)
    pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)
```

iii. "Pupil diameter, discretized into five equal percentile bins" comes straight from the instructions; the global pooling rationale is the same as for running speed (Key Decision 7). The middle-bin fallback is the AI's neutral imputation for a session with no pupil signal — it keeps the trial in the dataset rather than discarding otherwise-good neural data, at the cost of a slightly non-uniform class distribution.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identical to running speed: the pupil clock is searched with the *same* `[stim_start, stim_start + 0.75)` windows that define the neural bins, so pupil bin `i` and neural bin `i` cover the same wall-clock interval.

ii.
```python
pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
pup_bin_ends   = np.searchsorted(pupil_ts, all_stim_ends,   side='left')
...
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
```

iii. Same justification as 5-d — the eye-tracking camera is hardware-synced to the same 100 kHz sync line as the ophys and running streams, so absolute timestamps are directly comparable and no resampling of the neural data is needed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, read for the trial's row index.

ii.
```python
trial_hit = trials['hit'][:].astype(bool)
trial_miss = trials['miss'][:].astype(bool)
trial_fa = trials['false_alarm'][:].astype(bool)
trial_cr = trials['correct_reject'][:].astype(bool)
```

iii. These are the SDK's canonical change-detection outcome labels (Step 1 lists `Trials.from_nwb()` as the accessor, Step 5's mapping table maps `hit/miss/fa/cr → output[4], Categorical (4 classes), Static per trial`). Because aborted and auto-rewarded trials are already excluded, exactly one of the four is true for every retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. An if/elif chain maps the booleans to codes `hit=0, miss=1, false_alarm=2, correct_rejection=3`, with a silent fallback to `1` (miss) if none is set. The scalar code is then broadcast across every time bin of the trial so the output row has the same length as the time-varying rows (the variable is constant within a trial). Full-dataset distribution: hit 30.3 %, miss 57.2 %, FA 1.7 %, CR 10.8 %.

ii.
```python
if trial_hit[trial_idx]:
    outcome = 0  # Hit
elif trial_miss[trial_idx]:
    outcome = 1  # Miss
elif trial_fa[trial_idx]:
    outcome = 2  # False Alarm
elif trial_cr[trial_idx]:
    outcome = 3  # Correct Rejection
else:
    outcome = 1  # Default to Miss
```
```python
output_arr = np.stack([
    ot['image_identity'].astype(np.int64),
    ot['image_change'].astype(np.int64),
    running_disc.astype(np.int64),
    pupil_disc.astype(np.int64),
    np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
], axis=0)  # (5, n_bins)
```

iii. The format spec asks for time-varying outputs "if at all possible" while the task spec declares trial outcome "Static per-trial"; replicating the constant across bins satisfies both (uniform `(5, n_bins)` output arrays) and is what `train_decoder.py` expects. Step 10 Check 4 verifies the structural constraint "Hit+Miss for Go, FA+CR for Catch — Confirmed".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handled cases:
- **Experiment with 0 valid ROIs** → skipped with a warning.
- **No natural-images stimulus table** (`find_stim_key` returns `None`) → experiment skipped.
- **Trial with no stimulus presentations** (`trials_id` never matches) → trial skipped.
- **Session with < 2 usable trials** → session dropped from the output.
- **Empty bins** (no ophys/running/pupil sample inside a 750 ms window) → neural and running left at 0.0, pupil left as NaN.
- **Omitted stimulus flashes** → image identity forward-filled (back-filled if the trial starts with an omission).
- **Blinks / non-positive pupil areas** → NaN then linearly interpolated; a fully-NaN trial → middle quantile; a session with no `EyeTracking` group at all (3 of 202) → every bin middle quantile.
- **`is_change` NaN** (first presentation of a session) → 0.
- **Unknown image name** → code 0 (`img_to_idx.get(img, 0)`).
- **No outcome flag set** → miss.
Not handled: there is no `try/except` around per-experiment processing, so a single corrupt file would abort the whole run; and `run_cumsum = np.cumsum(running_speed)` would propagate a single NaN in the speed trace to every later bin of that session (verified to be latent only — the released `processing/running/speed` arrays contain no NaNs).

ii.
```python
def interpolate_nans(values):
    """Linearly interpolate NaN values in a 1D array."""
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return values  # All NaN, can't interpolate
    if valid.sum() == len(values):
        return values  # No NaN
    result = values.copy()
    x = np.arange(len(values))
    result[~valid] = np.interp(x[~valid], x[valid], values[valid])
    return result
```
```python
if n_neurons == 0: ... return None
if stim_key is None: ... return None
if len(trial_stim_indices) == 0: continue
if result is None or result['n_trials'] < 2: ... continue
change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
if nan_mask.all():
    pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
```

iii. Documented in Key Decisions 5 and 6 and in Step 10 Check 5 ("Check for edge cases"), where the AI reports: minimum trials per session = 39 (> 2), no NaN/Inf in the neural data, all output values valid integers. The general philosophy is to impute conservatively and keep the trial rather than discard neural data, and to drop only at the granularity where data is genuinely absent (no cells, no stimulus table, no trials).

## 9-a. What are the most time-consuming steps of the code?

i. The run is dominated by NWB file I/O and the per-bin Python loops, and the code prints timing for each phase. From `conversion_full_out.txt`: Pass 1 (statistics) 261.6 s, Pass 2 (conversion) 246.6 s, image-name collection 1.0 s, pickling 1.1 s — **510.4 s total**, i.e. Pass 1 and Pass 2 each account for ~50 %. Within a single experiment the cost is dominated by `dff_data = f[...]['data'][:]` (reading a full `(140 000 × n_neurons)` float64 array; sessions have up to 666 neurons), the `np.cumsum` over that array, and the three per-bin loops. The AI's own notes do not single out a bottleneck beyond reporting the per-pass timings; it did not observe that Pass 1 is almost entirely redundant.

ii.
```python
print(f"  Pass 1 completed in {time.time()-t0:.1f}s")
...
if (idx + 1) % 10 == 0 or idx == len(active_exps) - 1:
    elapsed = time.time() - t0
    rate = (idx + 1) / elapsed
    remaining = (len(active_exps) - idx - 1) / rate
    print(f"  Pass 2: {idx+1}/{len(active_exps)} experiments "
          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
```

iii. Step 7 of the notes extrapolated 14.6 s for 2 sessions to "~1453 s (~24 min)" for 202, which exceeded the 15-minute guidance in the instructions, prompting the vectorisation work in Step 6/Step 8 (cumsum-based averaging, `searchsorted` boundaries, sampling 10 files for image names). The realised time was 8.5 min.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python-level loops that are all trivially vectorisable given that the bin boundaries are already precomputed for the whole session:
- The three per-bin loops inside the per-trial loop (`for bi, si in enumerate(trial_stim_indices)` for neural, running and pupil). These could be a single fancy-indexed expression per trial, e.g. `neural = (dff_cumsum[ends] - dff_cumsum[starts]).T / counts`, or even one expression for the entire session followed by slicing per trial.
- The outer `for trial_idx in np.where(trial_mask)[0]` loop.
- The `for bi in range(len(images))` omitted forward-fill, which is a classic `np.maximum.accumulate` over the indices of non-omitted entries.
- `collect_all_image_names` iterating files (already capped at 10).
The heaviest arithmetic (per-bin averaging) *was* reduced to O(1) per bin by the cumsum trick, so these loops are overhead rather than asymptotic cost.

ii.
```python
for bi, si in enumerate(trial_stim_indices):
    s, e = ophys_bin_starts[si], ophys_bin_ends[si]
    n_frames = e - s
    if n_frames > 0:
        neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames
...
for bi, si in enumerate(trial_stim_indices):
    s, e = run_bin_starts[si], run_bin_ends[si]
...
for bi, si in enumerate(trial_stim_indices):
    s, e = pup_bin_starts[si], pup_bin_ends[si]
...
for bi in range(len(images)):
    if images[bi] == 'omitted':
        ...
```

iii. Step 6 of the notes lists the optimisations that *were* made ("Cumulative sum-based bin averaging (avoids per-bin boolean masking)", "np.searchsorted for efficient bin boundary finding") and the trajectory (step 61) shows the AI explicitly reasoning that the per-bin search was the bottleneck and replacing it with `searchsorted` + cumsum. It stopped there once the projected runtime fell under the threshold and did not push the remaining loops into array form.

## 9-c. What processing does the code repeat multiple times?

i. Substantial duplication:
- **The entire per-experiment pipeline runs twice.** `process_experiment(..., collect_stats_only=True)` in Pass 1 and `collect_stats_only=False` in Pass 2 execute the *same* code path; the `collect_stats_only` early-out sits *after* the neural bin averaging. So in Pass 1 every file is re-opened, the full ΔF/F array is re-read, `np.cumsum` over it is recomputed, and the `(n_neurons, n_bins)` matrix is built for every trial — and then thrown away. Only the running/pupil lists are kept. This is why Pass 1 costs 261 s, essentially as much as the real conversion.
- Pupil blink masking, area→diameter conversion, and whole-session NaN interpolation are likewise done twice.
- `np.searchsorted` bin-boundary computation for all three streams is done twice.
- Up to 10 files are opened a third time in `collect_all_image_names`.
- `discretize_values` recomputes/re-validates the same edges on every call.
An in-memory cache of the per-trial running/pupil values from a single pass (as the reference solution does) would have removed ~50 % of the runtime.

ii.
```python
# Pass 1
stats = process_experiment(nwb_path, row, collect_stats_only=True)
...
# Pass 2 — same function, same file, same computation
result = process_experiment(nwb_path, row, collect_stats_only=False)
```
```python
            # --- Neural data: vectorized bin averaging ---
            neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
            for bi, si in enumerate(trial_stim_indices):
                ...                                   # executed in BOTH passes
            ...
            if collect_stats_only:
                continue                              # ...and discarded in pass 1
```

iii. The two-pass structure is deliberate and documented (Key Decision 7 and Step 6: "Pass 1: Collect running speed and pupil statistics for percentile bin computation; Pass 2: Full conversion with discretization using global percentile bins") — global percentile edges genuinely require seeing all data before assigning any bin. What is *not* justified anywhere in the notes is re-reading and re-processing the neural data during Pass 1 instead of caching the small per-trial behavioural vectors; the AI never identified this redundancy.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work performed whose result is never used:
- **Pass 1's neural binning** (see 9-c): `dff_data` read, `valid_roi` filtering, `np.cumsum` over the full trace, and one `(n_neurons, n_bins)` matrix per trial — all discarded.
- **Pass 1's image identity and image-change computation** (`images`, the omitted forward-fill, `image_indices`, `change_flags`) — computed, then discarded at the `if collect_stats_only: continue`.
- **Datasets read but never used**: `trial_start`, `trial_stop`, `trial_change_time`, `stim_stop`, `stim_omitted`, `valid_trial_ids`, `n_total_rois`. The trial-timing columns in particular are read from every file in both passes even though trial extents are derived from `trials_id`.
- **Empty input arrays**: an `np.zeros((0, n_bins), dtype=np.float32)` is allocated per trial (~52 000 allocations) for an input dimension the task defines as empty; the format only needs a placeholder.
- **Raw behavioural copies**: `running_speed_raw` and `pupil_diameter_raw` are stored per trial in the intermediate dicts and are only used for the discretised versions (and the optional plots).
- The `bin_edges` return value of `discretize_values` is discarded at both call sites in Pass 2.

ii.
```python
            # --- Image identity ---
            images = stim_image_name[trial_stim_indices].copy()
            ...
            image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)
            # --- Image change ---
            change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)
            ...
            if collect_stats_only:
                continue          # everything above is thrown away in pass 1
```
```python
        trial_start = trials['start_time'][:]        # never used
        trial_stop = trials['stop_time'][:]          # never used
        trial_change_time = trials['change_time'][:] # never used
        stim_stop = stim['stop_time'][:]             # never used
        stim_omitted = stim['omitted'][:]            # never used
        valid_trial_ids = trial_ids[trial_mask]      # never used
...
            input_trials.append(np.zeros((0, n_bins), dtype=np.float32))
```

iii. None of this is justified in CONVERSION_NOTES.md — the notes claim only that optimisations were added, and Step 10's efficiency discussion does not revisit the point. The unused reads are most plausibly leftovers from the first draft of `process_experiment` (which the trajectory shows was rewritten at step 65 to the cumsum formulation), and the `collect_stats_only` flag was bolted onto the existing function rather than factored into a cheap statistics-only path.
