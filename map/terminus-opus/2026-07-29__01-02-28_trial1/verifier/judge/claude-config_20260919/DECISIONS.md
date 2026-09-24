# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as one NWB file per session under `data/sub-<id>/`. It enumerates subject directories with `os.listdir`, then lists the `.nwb` files inside each, building a flat list of `{subject, path, filename}` records (174 files, 28 subjects). Files are opened **directly with `h5py`** rather than with `pynwb`, and every field (trials table, behavioral events, units, electrodes, camera tracking) is read by raw HDF5 path. Each file is opened once per session and closed at the end of `process_session`. The data directory is the hard-coded relative path `DATA_DIR = 'data'`, so the script must be run from `/app`. Sessions are then processed sequentially in a single pass in `main()`.

ii.
```python
DATA_DIR = 'data'

def get_nwb_files():
    """Get all NWB file paths organized by subject."""
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({'subject': subj,
                              'path': os.path.join(subj_dir, nwb_file),
                              'filename': nwb_file})
    return all_files
```
```python
f = h5py.File(nwb_path, 'r')
trials = f['intervals']['trials']
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
```

iii. From the trajectory (steps 9-15) the AI explored the NWB hierarchy with `h5py` and mapped each field it needed to an HDF5 path; it then kept `h5py` for the conversion itself. CONVERSION_NOTES Step 6 records "Used h5py for NWB file reading" as an efficiency measure ("avoid unnecessary file I/O", pre-extract arrays once per file). Step 2 notes the file/subject layout (28 subjects, 174 files) as the basis for the glob-style enumeration.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the **containing directory name** (e.g. `sub-440956`), not from `nwb.subject.subject_id`. Each session record carries that string; in `main()` the AI appends each new subject string to `all_subjects` in order of first appearance and records `all_subjects.index(subj)` as `subject_idx` for the session. This yields the expected 28 subjects, each with several sessions (verification output lists 2-10 sessions per subject).

ii.
```python
all_files.append({'subject': subj, ...})          # subj == 'sub-440956'
...
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
...
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 documents the one-directory-per-animal layout ("28 subjects (sub-440956 through sub-484677)"), so the directory name is used as the canonical animal id. No further justification is given for preferring the folder name over the NWB `subject_id` field (they are the same number with a `sub-` prefix).

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is done. **However, the AI then discards whole sessions on behavioural-performance grounds**: a session is kept only if (a) it has ≥2 valid trials, (b) it has ≥1 good unit, (c) its correct rate on control (non-photostim), non-early-lick trials is ≥65%, and (d) it has ≥50 correct left *and* ≥50 correct right control non-early trials. This dropped 31 of 174 sessions (23 for correct rate, 8 for trials-per-side), leaving **143 sessions / 57,560 units**, versus the 173 sessions / 69,943 good units quoted in the reference texts. Sessions appear in the output in sorted file order; no session id is stored in the output (metadata has no `session_info`).

ii.
```python
MIN_CORRECT_RATE = 0.65
MIN_CORRECT_TRIALS_PER_SIDE = 50
...
is_control = (photostim_onset_trial == b'N/A') & trial_mask
is_not_early = early_lick == b'no early'
control_non_early = is_control & is_not_early
hits = np.sum(outcome[control_non_early] == b'hit')
misses = np.sum(outcome[control_non_early] == b'miss')
correct_rate = hits / (hits + misses)
correct_left = np.sum((outcome == b'hit') & (trial_instruction == b'left') & control_non_early)
correct_right = np.sum((outcome == b'hit') & (trial_instruction == b'right') & control_non_early)
if correct_rate < MIN_CORRECT_RATE:
    print(f"    Skipping: correct rate too low"); f.close(); return None
if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:
    print(f"    Skipping: insufficient correct trials per side"); f.close(); return None
```

iii. CONVERSION_NOTES Step 3 lists "Performance threshold >65%" and "Min correct trials/side 50 each" as methods-paper criteria, and Step 5 decision list keeps them. The AI noticed the resulting mismatch with the papers (Step 9 table: "Sessions 173 vs 143 — PARTIAL, 31 sessions skipped by criteria"; "Total neurons 69,943 vs 57,560 — PARTIAL, fewer sessions") and explained it as an expected consequence of applying the behavioural criteria, rather than investigating whether the published 173-session dataset had already been curated.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials`), read as parallel arrays (`trial_instruction`, `outcome`, `early_lick`, `auto_water`, `free_water`, `photostim_onset`, `photostim_duration`, `start_time`). Trial *k* is matched positionally to `go_start_times.timestamps[k]`. The equality of the two lengths is assumed, never asserted; a per-trial index array `trial_indices` (the surviving rows) then drives firing-rate, input and output construction so all three streams stay in the same trial order.

ii.
```python
trials = f['intervals']['trials']
n_trials_total = len(trials['id'])
trial_instruction = trials['trial_instruction'][:]
outcome          = trials['outcome'][:]
early_lick       = trials['early_lick'][:]
...
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
...
trial_indices = np.where(trial_mask)[0]
for trial_idx in trial_indices:
    go_time = go_times[trial_idx]
```

iii. Trajectory step 14: "Go cue times are in acquisition/BehavioralEvents/go_start_times/timestamps (368 per session, matching n_trials)" — the AI verified the one-go-cue-per-trial correspondence interactively on an example session and then relied on it. It also noted (step 14) that `sample_start_times` has *more* entries than trials because early licks replay the sample epoch, which is why only the go cue is used for positional indexing.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, combined into one boolean mask:
1. `auto_water == 0` (reward given automatically),
2. `free_water == 0`,
3. a **neural-coverage heuristic**: the trial index must be `< units/is_good_trials.shape[1]` (the number of trials for which ephys exists) **and** the trial window must fall inside the spike-time range with a 1 s slack (`go - 2.5 >= min_spike_time - 1.0` and `go + 1.5 <= max_spike_time + 1.0`).

Sessions with <2 surviving trials are dropped. Early-lick, `ignore` (no-response) and photostim trials are deliberately **kept**, because the decoder spec makes them outputs/inputs. Result: 74,484 trials over 143 sessions (mean 520.9/session). Empirically this coverage heuristic reproduces the `obs_intervals` set almost exactly (checked here on all 9 partially-recorded sessions: identical for 7, losing 1 and 125 trials in two of them, admitting no unobserved trial).

ii.
```python
n_recorded_trials = f['units']['is_good_trials'].shape[1]
max_spike_time = spike_times_flat.max(); min_spike_time = spike_times_flat.min()

neural_valid = np.zeros(n_trials_total, dtype=bool)
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True

trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]
if n_selected < 2:
    print(f"    Skipping: too few valid trials"); f.close(); return None
```

iii. The auto/free-water exclusion is taken from the reference code's `get_regular_trial_mask` (CONVERSION_NOTES Step 1/Step 3: "exclude early lick, auto water, free water, no response, photostim"), minus the exclusions the decoder task requires be kept (trajectory step 35: "the task says I need photostimulation as a decoder INPUT … I need to KEEP early lick trials and no-response trials since those are outputs"). The coverage filter was added after debugging (trajectory steps 54-59): 375/1241 sample trials were all-zero because "spike times only go up to ~1428 s but go cue times go up to ~6697 s" — the ephys stopped mid-session. The AI found `is_good_trials` has shape (n_units, n_recorded_trials) and used its second dimension, plus the spike-range check "more robust than relying on is_good_trials" for trials at the boundary.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged, session-absolute seconds) with `units/spike_times_index` for the per-unit offsets, restricted to units with `units/classification == b'good'`, together with `acquisition/BehavioralEvents/go_start_times/timestamps` which supplies the per-trial alignment time. Brain-region labels for those units come from `general/extracellular_ephys/electrodes/location` via `units/electrodes` + `units/electrodes_index`.

ii.
```python
spike_times_flat  = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```
```python
electrodes_idx = f['units']['electrodes'][:]
electrodes_index = f['units']['electrodes_index'][:]
electrode_locations = f['general']['extracellular_ephys']['electrodes']['location'][:]
region = extract_brain_region(electrode_locations[elec_idx])   # JSON -> 'left ALM'
```

iii. Trajectory steps 10-11 and 45-46: the AI confirmed spike times are a flat array plus index, that they are **absolute** timestamps (not pre-aligned), and that `classification` holds the QC verdict. Step 15: "Brain regions are in electrode location as JSON strings with 'brain_regions' field … anno_name gives fine-grained CCF annotations but many are empty", which is why the electrode JSON was chosen over `anno_name`.

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin converted to firing rate in Hz. For each retained trial and each good unit, `searchsorted` locates the spikes in `[go-2.5, go+1.5)`, those spike times are shifted to be go-cue-relative, `np.histogram` bins them on the fixed 81-edge grid, and counts are divided by the bin width. No smoothing, no normalisation, no baseline subtraction. Output is `float32`, one `(n_good, 80)` array per trial.

ii.
```python
def compute_firing_rates_fast(spike_times_flat, spike_times_index, go_cue_times,
                              good_indices, trial_indices, ...):
    n_bins = int(round((window_end - window_start) / bin_width))
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
    starts = np.zeros(len(spike_times_index), dtype=np.int64); starts[1:] = spike_times_index[:-1]
    ends = spike_times_index.astype(np.int64)
    good_spike_times = [spike_times_flat[starts[u]:ends[u]] for u in good_indices]

    for trial_idx in trial_indices:
        go_time = go_cue_times[trial_idx]
        fr_trial = np.zeros((n_good, n_bins), dtype=np.float32)
        abs_start = go_time + window_start; abs_end = go_time + window_end
        for i, unit_spikes in enumerate(good_spike_times):
            idx_lo = np.searchsorted(unit_spikes, abs_start)
            idx_hi = np.searchsorted(unit_spikes, abs_end)
            if idx_hi > idx_lo:
                aligned = unit_spikes[idx_lo:idx_hi] - go_time
                counts = np.histogram(aligned, bins=bin_edges)[0]
                fr_trial[i, :] = counts / bin_width
        firing_rates_list.append(fr_trial)
```

iii. CONVERSION_NOTES Step 5/6: "Computed firing rates as spike counts / bin_width (Hz)"; the reference pipeline's `sliding_histogram` also returns rate = counts/bin width, but with a 40 ms window and 3.4 ms stride, which the AI deliberately replaced with the task-specified non-overlapping 50 ms bins (Step 4 discrepancy table: "Firing rate params bw=0.04, stride=0.0034 → Decoder task specifies 50 ms bins"). Step 10 Check 2 reports a spot check against the raw NWB file, `np.allclose(manual, converted) = True`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == b'good'` (the spike-sorting QC classifier verdict) are kept; no individual metric (`isi_violation`, `presence_ratio`, `amplitude_cutoff`, …) is thresholded, and `unit_quality` is not used. A session with zero good units is dropped. The per-unit × per-trial `is_good_trials` matrix is *not* used for unit or trial-level masking — only its shape. Total retained: 57,560 units over 143 sessions (mean 402.5/session).

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
n_good = int(np.sum(good_mask))
good_indices = np.where(good_mask)[0]
if n_good == 0:
    print(f"    Skipping: no good units"); f.close(); return None
...
good_regions = [unit_regions[i] for i in range(n_total) if good_mask[i]]
```

iii. Trajectory steps 27-30: after reading the reference preprocessing (`qc_mode='classifier'`, `idx_qc_dict` of classifier-good ids), the AI concluded "In the NWB files, the 'classification' field ('good' vs 'unlabelled') IS the QC classifier output. So I should filter by classification=='good'." CONVERSION_NOTES Step 3 links this to the 5 region-specific logistic-regression classifiers of the QC white paper and to the paper's 25.9% good-unit rate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**, `BehavioralEvents/go_start_times/timestamps[trial_idx]`. Spike times and event times share one session-absolute clock, so alignment is a subtraction: the window `[go-2.5, go+1.5)` is selected with `searchsorted` and the selected spike times have `go_time` subtracted before histogramming on the fixed relative grid. No resampling or interpolation. The same go-cue-relative grid (`bin_centers`) is used for the inputs and for the tongue output, so all streams are aligned bin-for-bin.

ii.
```python
go_time = go_cue_times[trial_idx]
abs_start = go_time + window_start
abs_end   = go_time + window_end
idx_lo = np.searchsorted(unit_spikes, abs_start)
idx_hi = np.searchsorted(unit_spikes, abs_end)
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
```
```python
'temporal_alignment_event': 'Go cue onset',
'off_start': WINDOW_START,   # -2.5
'off_end':   WINDOW_END,     # +1.5
```

iii. Trajectory step 45-46: the AI explicitly checked "Spike times are ABSOLUTE timestamps (need to align to go cue)" before writing the code — the reference `.mat` pipeline stores them already go-cue-relative, so this was flagged as a format difference to handle. The alignment event itself is prescribed by the decoder task ("Temporally align based on Go cue onset").

## 2-e. What is the temporal resolution of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins over `[-2.5, +1.5]` s = exactly 80 bins for every trial in every session; `metadata['time_bin_size'] = 50.0` ms. There is no rebinning of an intermediate representation: spike times are binned once directly at the target resolution. The reference pipeline's 40 ms/3.4 ms sliding histogram is deliberately not reproduced.

ii.
```python
BIN_WIDTH = 0.05                 # 50 ms bins
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))   # 80
...
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
'time_bin_size': BIN_WIDTH * 1000,
```

iii. CONVERSION_NOTES Step 5 key decision 1: "**50 ms non-overlapping bins** (task spec, not 40 ms sliding from reference)" and decision 2: "**[-2.5, 1.5] s window** (task spec), giving 80 time bins". Step 4's discrepancy table records this as an allowed deviation from the reference code, justified by the decoder-task specification. Verification confirms T = 80 for every session (min = max = 80).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the tone/sample-epoch onsets) and the trial's go-cue time. Because an early lick replays the sample epoch, a session has more sample starts than trials, so for each trial the AI takes the **last sample start at or before the go cue**. If no sample start precedes the go cue, a hard-coded fallback of −1.85 s (the nominal sample-to-go interval) is used.

ii.
```python
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
...
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
```

iii. Trajectory step 14: "Sample start times have MORE entries than trials (405 vs 368) — this is because early lick trials cause replays"; step 18: "Sample starts 1.85 s before go cue (3 tones × 150 ms + 2 gaps × 100 ms = 650 ms sample epoch + 1.2 s delay = 1.85 s)", which is both the reason for taking the *last* preceding sample start and the source of the −1.85 s fallback constant.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying input: for each trial, the go-cue-relative bin centres are shifted by the tone-to-go gap, i.e. `value(bin) = bin_center + (go - tone)`. It is stored as row 0 of the `(2, 80)` float32 input array. No clipping or normalisation; values before the tone are negative. Observed range across the dataset is [−1.5, 11.9] s — the large maxima come from trials where repeated early-lick replays put the last sample start many seconds before the go cue.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
time_from_tone = bin_centers - tone_onset_rel          # tone_onset_rel = tone - go  (negative)
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Time from tone onset → input[0], `bin_centers - tone_onset_rel`, continuous, time-varying". Trajectory step 80 shows the AI checking the suspicious 11.9 s maximum and concluding it is real: "If tone_onset_rel = −10.4, then at the last bin centre time_from_tone = 11.875 … it's the time since the tone onset, which could be much earlier if there were early lick replays". Step 10 Check 2 records an independent recomputation from the raw NWB (`np.allclose` = True).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined *on* the neural bin grid: the same `bin_centers` array derived from `WINDOW_START/WINDOW_END/BIN_WIDTH` that defines the firing-rate bins is used, shifted by a per-trial scalar. Bin *k* of the input therefore covers exactly the interval of bin *k* of the firing rates by construction; no interpolation or offset is needed, since sample-start and go-cue timestamps are on the same session-absolute clock as the spikes.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
time_from_tone = bin_centers - tone_onset_rel
```
```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)   # same grid, firing rates
```

iii. Implicit in the design (CONVERSION_NOTES Step 5: all streams "aligned to go cue, [-2.5, 1.5] s, 80 bins"). The `--show-processing` plots overlay the inputs on the same go-cue-relative axis as the mean firing rate, with a dashed line at t = 0 and at −1.85 s, as a visual alignment check.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials-table columns `photostim_onset` and `photostim_duration` (stored as byte strings, `b'N/A'` on unstimulated trials), plus `trials/start_time` (the onset is measured from trial start) and the trial's go-cue time (to re-express it on the bin axis).

ii.
```python
photostim_onset_trial    = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times        = trials['start_time'][:]
...
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ps_onset_float = float(ps_onset_val)
    ps_dur_float   = float(photostim_duration_trial[trial_idx])
    trial_start    = trial_start_times[trial_idx]
```

iii. Trajectory steps 16-17: "the photostim_onset in trials is relative to trial start (about 1.8-2.7 s into the trial), while photostim_start_times in BehavioralEvents are absolute timestamps. The stim onset relative to go cue is about −0.5 s (last 0.5 s of delay), consistent with the methods description. The photostim duration is 0.5 s." The trials-table columns were preferred because they give a per-trial onset *and* duration with an explicit `N/A` marker.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying input (row 1 of the `(2, 80)` input array): the stimulation interval is converted to go-cue-relative seconds (`start_time + onset − go`, end = start + duration) and a bin is set to 1.0 if its centre lies in `[on, off)`. Trials with `b'N/A'` keep an all-zero vector. Dataset-wide the input takes only values {0, 1}; ~20.4% of retained trials contain stimulation.

ii.
```python
photostim_binary = np.zeros(n_bins, dtype=np.float32)
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ps_onset_float = float(ps_onset_val)
    ps_dur_float = float(photostim_duration_trial[trial_idx])
    trial_start = trial_start_times[trial_idx]
    ps_rel_start = (trial_start + ps_onset_float) - go_time
    ps_rel_end = ps_rel_start + ps_dur_float
    photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Photostim on/off → input[1], binary from photostim_onset/duration, binary time-varying", following the decoder spec ("Whether photostimulation is on at every time point (discrete, time-varying)") rather than a per-trial flag. Step 9 compares the resulting 20.4% stimulated fraction with the ~25% quoted in the methods and attributes the gap to the dropped sessions.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. By converting the stimulation window into the same go-cue-relative coordinate system as the neural bins (`(trial_start + onset) - go_time`) and testing the neural grid's bin centres against it. Bin *k* of the photostim input therefore refers to the same 50 ms interval as bin *k* of the firing rates.

ii.
```python
ps_rel_start = (trial_start + ps_onset_float) - go_time
ps_rel_end = ps_rel_start + ps_dur_float
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The AI verified the expected placement before coding (trajectory step 17: stim onset ≈ −0.5 s relative to the go cue, i.e. the last 0.5 s of the delay, matching the methods), and the `--show-processing` panel "Input: Photostim" plots stimulated trials on the go-cue axis to confirm the pulses land just before t = 0.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. **Only** `trials/trial_instruction` (`b'left'` / `b'right'`). The AI equates the instructed lick direction with the animal's choice; `outcome` is not consulted, so trials where the animal licked the *wrong* side (`miss`) or did not lick at all (`ignore`) are still labelled with the instructed side. Consequently there are only two classes and no "no lick" class, and the reported distribution (49.4% left / 50.6% right) is the stimulus distribution rather than the behavioural one — roughly 26% of trials (15.3% miss + 10.9% ignore) carry a label that does not describe what the animal did.

ii.
```python
trial_instruction = trials['trial_instruction'][:]
...
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
...
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. The identification is made in trajectory step 11, where the AI catalogues the trials table: "trial_instruction: 'left'/'right' (**lick direction**)". No later step revisits it; the notes' mapping table (Step 5) simply lists "trial_instruction → output[0] choice, left=0, right=1", and neither CONVERSION_NOTES nor the trajectory ever mentions the required third "no lick" category or the miss/ignore cases.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A per-trial scalar (0 = left, 1 = right) broadcast across all 80 bins of row 0 of the `(4, 80)` int64 output array. `output_values[0] = ['left', 'right']`. No use of the lick-time streams (`left_lick_times` / `right_lick_times` exist in the file but are never read), and no third class.

ii.
```python
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = choice
...
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
'output_values': [['left', 'right'], ['ignore', 'miss', 'hit'], ['no', 'yes'],
                  ['below_p40', 'p40_to_p60', 'above_p60']],
```

iii. CONVERSION_NOTES Step 5: "trial_instruction → output[0] choice, left=0, right=1, per-trial, repeated across time". Repeating per-trial values across the 80 bins is justified by the target format's preference for time-varying outputs in a single `(n_output, n_timepoints)` array. The AI's Step 10 "output spot check" compares the converted `choice` against `trial_instruction` in the raw file — i.e. it validates the implementation of the mapping, not the definition of choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials/outcome`, which already contains exactly the three required strings `b'hit'`, `b'miss'`, `b'ignore'`.

ii.
```python
outcome = trials['outcome'][:]
...
out = outcome[trial_idx]
```

iii. Trajectory step 11: "outcome: 'hit'/'ignore'/'miss'" — the three categories map one-to-one onto the decoder spec's ignore/miss/hit, so no derivation is needed. CONVERSION_NOTES Step 5 mapping table: "outcome → output[1], ignore=0, miss=1, hit=2".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String → integer code with an explicit if/elif chain (`ignore`→0, `miss`→1, `hit`→2, anything else→0), written per trial into row 1 of the output array and repeated across all 80 bins. Resulting distribution: ignore 10.9%, miss 15.3%, hit 73.9%.

ii.
```python
out = outcome[trial_idx]
if out == b'ignore':
    outcome_val = 0
elif out == b'miss':
    outcome_val = 1
elif out == b'hit':
    outcome_val = 2
else:
    outcome_val = 0
...
output_trial[1, :] = outcome_val
```

iii. Code order follows the decoder spec ("Outcome (ignore, miss, hit)"). CONVERSION_NOTES Step 9 compares the 73.9% hit rate with the paper's ~84% correct rate and explains the difference as inclusion of early-lick and ignore trials in the denominator (the paper computes performance on control, non-early trials with ignores excluded).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials/early_lick`, a two-valued string column (`b'early'` / `b'no early'`).

ii.
```python
early_lick = trials['early_lick'][:]
...
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. Trajectory step 11 identifies the column and its two values; the decoder spec asks for early lick (no, yes) per trial, so the column is used as-is. The AI also uses this column for the session-level performance computation (control non-early trials).

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 = no, 1 = yes and broadcast across the 80 bins of row 2; `output_values[2] = ['no', 'yes']`. Note the test is "equals `b'early'` → 1, else 0", so any unexpected value silently becomes 0. Distribution: 88.5% no, 11.5% yes.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
...
output_trial[2, :] = early
```

iii. CONVERSION_NOTES Step 5 mapping table: "early_lick → output[2], no=0, yes=1, per-trial, repeated across time". Keeping early-lick trials (which the reference analysis discards) is justified in trajectory step 35 by the decoder spec requiring early lick as an output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data` of shape `(n_frames, 3)` = (tongue_x, tongue_y, tongue_likelihood) with matching `timestamps` at ~294-300 Hz. Column 1 is the y-position; column 2 (likelihood) gates which frames count. Presence of the series is tested first, and if it is absent all bins are NaN.

ii.
```python
has_tongue = 'Camera0_side_TongueTracking' in f['acquisition']['BehavioralTimeSeries']
if has_tongue:
    tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
    tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
...
    y_slice  = tongue_data[idx_start:idx_end, 1]
    lk_slice = tongue_data[idx_start:idx_end, 2]
```

iii. Trajectory steps 12-13 and 45-46: the AI located the series under `BehavioralTimeSeries` (not directly under `acquisition`) and confirmed the channel order: "Tongue tracking columns are (tongue_x, tongue_y, tongue_likelihood) — y is column index 1 … at ~300 Hz (0.0034 s steps)".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per trial, the frames in `[go-2.5-0.05, go+1.5+0.05]` are selected with `searchsorted`, shifted to go-cue-relative time, and for each of the 80 bins the mean y of the frames in that bin **with likelihood > 0.1** is taken; a bin with no such frame is left NaN. All non-NaN bin means from all retained trials of the session are pooled into `all_tongue_y`, and the 40th and 60th percentiles of that pool become the session's class edges. (Percentiles are therefore over the binned, visible-tongue values within the analysed trial windows, not over raw frames and not over the inter-trial intervals.)

ii.
```python
for trial_idx in trial_indices:
    go_time = go_times[trial_idx]
    abs_start = go_time + window_start - bin_width
    abs_end   = go_time + window_end + bin_width
    idx_start = np.searchsorted(tongue_timestamps, abs_start)
    idx_end   = np.searchsorted(tongue_timestamps, abs_end)
    tongue_y_binned = np.full(n_bins, np.nan)
    if idx_start < idx_end:
        ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
        y_slice  = tongue_data[idx_start:idx_end, 1]
        lk_slice = tongue_data[idx_start:idx_end, 2]
        high_conf = lk_slice > 0.1
        for b in range(n_bins):
            bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
            if np.sum(bin_mask) > 0:
                tongue_y_binned[b] = np.mean(y_slice[bin_mask])
    tongue_y_trials.append(tongue_y_binned)
    valid_y = tongue_y_binned[~np.isnan(tongue_y_binned)]
    if len(valid_y) > 0:
        all_tongue_y.extend(valid_y.tolist())
```
```python
all_tongue_y_arr = np.array(all_tongue_y)
p40 = np.percentile(all_tongue_y_arr, 40)
p60 = np.percentile(all_tongue_y_arr, 60)
```

iii. CONVERSION_NOTES Step 5 decision 8: "Tongue y percentiles: computed per-session from all valid (likelihood>0.1) tracking data", following the decoder spec's per-session 40th/60th percentile discretisation. Using the likelihood channel to gate frames reflects the AI's observation (trajectory step 46) that the tracker reports x/y on every frame with a confidence value; the 0.1 cut-off itself is not justified in the notes.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes only: `y < p40` → 0, `p40 ≤ y ≤ p60` → 1, `y > p60` → 2. The array is **initialised to 1**, so every bin whose mean is NaN — i.e. every bin in which the tongue was not visible, and every bin of a session with no tracking series — is emitted as class 1 ("p40_to_p60"). The required fourth class, `3: not visible`, is not created and `output_values[3]` lists only three names. The consequence is visible in the verification output: the class distribution is 10.9% / **78.2%** / 10.9% instead of roughly 40/20/40 of the *visible* bins, because ~75-80% of all bins have no visible tongue and are folded into the middle class.

ii.
```python
tongue_y = tongue_y_trials[i]
tongue_y_disc = np.ones(n_bins, dtype=np.int64)        # NaN bins stay 1
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
output_trial[3, :] = tongue_y_disc
```
```python
'output_values': [..., ['below_p40', 'p40_to_p60', 'above_p60']],
```

iii. CONVERSION_NOTES Step 10 Check 5 states the rule as an intentional edge-case policy: "Missing tongue tracking: Defaults to middle category (1) for NaN values". The AI did notice the consequence during sample validation (trajectory step 61: "The tongue_y distribution is heavily concentrated in the middle category (77%). This is because many timepoints have NaN tongue data (tongue not visible), and I'm defaulting to middle … the tongue_y percentile discretization might need adjustment") but moved on to decoder training and never revisited it; the final notes present the 10.9/78.2/10.9 distribution without further comment.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. On the same go-cue-relative grid as the firing rates: camera timestamps share the session-absolute clock, so the frame range for a trial is found by `searchsorted(tongue_timestamps, go ± window)` and each frame's bin is determined from `timestamp - go_time` against the identical `bin_edges` array used for spikes. A ±1 bin margin is added to the fetched slice (it only widens the slice; the per-bin masks still use the exact 80-bin grid), so bin *k* of the tongue output covers the same interval as bin *k* of the neural data.

ii.
```python
abs_start = go_time + window_start - bin_width
abs_end   = go_time + window_end + bin_width
idx_start = np.searchsorted(tongue_timestamps, abs_start)
idx_end   = np.searchsorted(tongue_timestamps, abs_end)
ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. The AI verified the camera sampling and timestamp convention in trajectory steps 13-14/46 before writing the code, and the `--show-processing` figure plots the raw binned tongue y (with the p40/p60 lines) and the discretised output on the same go-cue axis as the firing rates as a visual alignment check (CONVERSION_NOTES Step 7: "Processing plots … No anomalies observed").

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases are handled, three of them explicitly:
- **Trials with no ephys coverage** (recording stopped before the behaviour ended, up to 442 trials in one session): detected via `is_good_trials.shape[1]` plus a spike-time-range test with 1 s slack, and excluded (see 1-e). Sessions left with <2 trials are dropped.
- **Sessions with no spikes at all**: `len(spike_times_flat) == 0` → session returns `None`.
- **Missing / low-confidence tongue frames**: frames with likelihood ≤ 0.1 are dropped from the bin mean; bins left empty, and whole sessions lacking the `Camera0_side_TongueTracking` series, are *imputed as class 1* rather than flagged.
- **A session whose `classification`/`anno_name` columns are all NaN** (never quality-controlled, `sub-440958_ses-20190216T162508`): not handled explicitly — `classification == b'good'` on a float array yields a scalar `False` (with the NumPy comparison warning suppressed by the module-level `warnings.filterwarnings('ignore')`), which happens to produce zero good units. In this run the session was in fact dropped one check earlier, by the ≥50-correct-trials-per-side criterion.
- Unknown strings in `outcome` / `early_lick` silently fall through to the `else` branches (0 = ignore, 0 = no early), and a trial with no preceding tone falls back to a hard-coded −1.85 s.

ii.
```python
if len(spike_times_flat) > 0:
    max_spike_time = spike_times_flat.max(); min_spike_time = spike_times_flat.min()
else:
    f.close(); return None
```
```python
has_tongue = 'Camera0_side_TongueTracking' in f['acquisition']['BehavioralTimeSeries']
...
else:
    tongue_y_trials = [np.full(n_bins, np.nan) for _ in trial_indices]
    all_tongue_y = []
if len(all_tongue_y) > 0:
    p40 = np.percentile(all_tongue_y_arr, 40); p60 = np.percentile(all_tongue_y_arr, 60)
else:
    p40, p60 = 0, 0
```
```python
tongue_y_disc = np.ones(n_bins, dtype=np.int64)     # missing -> middle class
```
```python
import warnings
warnings.filterwarnings('ignore')
```

iii. CONVERSION_NOTES Step 10 Check 5 lists the edge cases considered: "Partial neural recording: handled via is_good_trials.shape[1] and spike time range check; Missing tongue tracking: defaults to middle category (1) for NaN values; Sessions without photostim: photostim_binary is all zeros (correct); Auto_water trials: properly excluded". The trial-coverage handling is well motivated by the debugging in trajectory steps 54-59 (375/1241 all-zero trials traced to an ephys recording that ended mid-session). The imputation policy for tongue data is stated but not argued for, and the suppression of all warnings is not discussed anywhere.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented every stage with timers and printed them per session. Aggregated over the 143 converted sessions in `conversion_full_out.txt`: **firing-rate computation 810 s (85% of the 956 s total)**, tongue tracking 71 s, HDF5 data loading 44 s, input and output construction ≈0 s, plus ~28 s for pickling the 9.3 GB result. Per session the firing-rate step ran 3-14 s. Sessions that are later rejected by the performance criteria still pay for the full `spike_times` read, because that array is loaded before the criteria are evaluated. The AI's own estimate before the full run (≈4 s/session, ~12 min) undershot the actual 6.7 s/session and 16 min.

ii.
```python
t1 = time.time(); print(f"    Data loading: {t1-t0:.1f}s")
firing_rates = compute_firing_rates_fast(...)
t2 = time.time(); print(f"    Firing rate computation: {t2-t1:.1f}s")
...
t4 = time.time(); print(f"    Tongue tracking: {t4-t3:.1f}s")
...
print(f"    Total session time: {t6-t0:.1f}s")
```

iii. CONVERSION_NOTES Step 7 records the measured breakdown and the extrapolation ("Firing rate computation 3-14 s/session → ~1000 s; Total ~16 min"), and Step 6 records the two optimisations that were applied: "searchsorted instead of full array masking: ~2x speedup; Pre-extract spike times per unit: ~1.5x speedup". Trajectory step 63: "The optimization with searchsorted reduced firing rate computation from 8 s to 3 s per session … estimated ~12 minutes. That's within the 15 minute limit" — after which no further optimisation was attempted, even though the run took 16 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two nested Python loops remain, and both are vectorisable:
- `compute_firing_rates_fast` loops over trials **and inside that over units**, doing two `searchsorted` calls and one `np.histogram` per (trial, unit) pair — on the order of 30 M `np.histogram` calls across the dataset. The trial dimension can be collapsed entirely by building one flat array of all `n_trials × 81` absolute bin edges and calling `searchsorted` once per unit, then differencing (this is exactly what the reference implementation does, and it is the reason the reference converts all 174 sessions in 247 s versus 956 s here).
- `get_tongue_y_for_trials` loops over trials and then over the 80 bins, building a boolean mask over the whole frame slice for each bin — O(n_bins × n_frames) work that a single `np.floor((t - t0)/BIN)` index plus `np.bincount` would do in one pass.

Minor: the per-trial input-construction loop and the per-trial output-construction loop are also scalar loops, but they are negligible (≈0 s).

ii.
```python
for trial_idx in trial_indices:
    ...
    for i, unit_spikes in enumerate(good_spike_times):
        idx_lo = np.searchsorted(unit_spikes, abs_start)
        idx_hi = np.searchsorted(unit_spikes, abs_end)
        if idx_hi > idx_lo:
            aligned = unit_spikes[idx_lo:idx_hi] - go_time
            counts = np.histogram(aligned, bins=bin_edges)[0]
            fr_trial[i, :] = counts / bin_width
```
```python
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
    n_in_bin = np.sum(bin_mask)
    if n_in_bin > 0:
        tongue_y_binned[b] = np.mean(y_slice[bin_mask])
```

iii. The AI treated the `searchsorted` windowing as the optimisation (CONVERSION_NOTES Step 6) and stopped once the projected runtime fell under the 15-minute guideline in the instructions; it never identified the trial loop as vectorisable, and the notes contain no discussion of the tongue binning loop's cost (71 s).

## 10-c. What processing does the code repeat multiple times?

i. Nothing is recomputed in a way that changes results, but several quantities are recomputed or re-derived:
- `n_bins`, `bin_edges` and `bin_centers` are recreated inside `compute_firing_rates_fast`, inside `get_tongue_y_for_trials`, and again in `process_session`, from the same module-level constants.
- The (trial, unit) loop re-runs `searchsorted` over each unit's spike train once per trial (80 M+ binary searches) instead of once per unit over a flattened edge array.
- `np.histogram` re-derives the bin edges and re-validates them on every call.
- `unit_regions` is built for **all** units (JSON parse per unit) and only afterwards subset to good units.
- `trial_indices` is iterated three separate times (firing rates, inputs, outputs), each time re-reading `go_times[trial_idx]`.
- The full `spike_times` array is read from disk for every one of the 174 files, including the 31 that are then rejected.

ii.
```python
def compute_firing_rates_fast(...):
    n_bins = int(round((window_end - window_start) / bin_width))
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
...
def get_tongue_y_for_trials(...):
    n_bins = int(round((window_end - window_start) / bin_width))
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
...
    n_bins = N_TIMEBINS
    bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
```
```python
unit_regions = []
for i in range(n_total):                      # all units, not just good ones
    start = 0 if i == 0 else int(electrodes_index[i-1])
    elec_idx = electrodes_idx[start]
    region = extract_brain_region(electrode_locations[elec_idx])   # json.loads per unit
    unit_regions.append(region)
good_regions = [unit_regions[i] for i in range(n_total) if good_mask[i]]
```

iii. Not discussed in CONVERSION_NOTES. The one deliberate anti-repetition measure is documented in Step 6 ("Pre-extract spike times per unit: ~1.5× speedup") — the ragged `spike_times` buffer is sliced into per-unit arrays once per session rather than per trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, none affecting correctness:
- `unit_regions` is computed (with a `json.loads` per unit) for every unit in the file, then ~75% of the results are thrown away by the `good_mask` subset.
- `spike_times_flat` (up to ~11.5 M doubles) is read in full before the session-level performance criteria are applied, so 31 sessions were fully loaded and then discarded.
- `correct_rate` is computed and returned in the session result dict but is never written into the output (only printed and used in the plot title).
- `min_spike_time`/`max_spike_time` scans over the whole spike array, and `n_trials_total`/`n_recorded_trials` bookkeeping, exist only for the coverage filter.
- `all_tongue_y` is accumulated as a Python list of floats via `.extend(valid_y.tolist())` and then converted back to an array just to take two percentiles.
- Outputs are stored as `int64` although they only take values 0-3 (8× the necessary memory; ~190 MB of the pickle), and the `input` photostim row is stored as float32 although it is binary.
- Under `--show-processing`, an eight-panel matplotlib figure is generated per session (only for the first 2 sessions, and only in that mode).

ii.
```python
return {
    'neural': firing_rates, 'input': inputs_list, 'output': outputs_list,
    'brain_regions': good_regions, 'subject': subject_id,
    'n_good': n_good, 'n_trials': len(firing_rates),
    'correct_rate': correct_rate,          # never stored in the output dict
}
```
```python
output_trial = np.zeros((4, n_bins), dtype=np.int64)     # values are only 0..3
```
```python
valid_y = tongue_y_binned[~np.isnan(tongue_y_binned)]
if len(valid_y) > 0:
    all_tongue_y.extend(valid_y.tolist())               # python list round-trip
```

iii. Not discussed in CONVERSION_NOTES; the notes' only dtype comment is the opposite change, Step 10 issue 3: "Output dtype: changed from float32 to int64 for proper indexing" (made after the decoder rejected float labels — int8/int32 would have sufficed). The 9.3 GB pickle size is reported but its composition is not analysed.
