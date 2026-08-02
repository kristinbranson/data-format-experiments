# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all subject directories in `/app/data` whose names start with `jm`, then enumerates all subdirectories within each subject as sessions. For each session it loads Suite2p fluorescence arrays (`F.npy`, `Fneu.npy`) and behavioral motion-energy data (`motion_energy_glob.npy`) plus camera timestamps (`tstamps.npy`). Trials are not loaded directly from disk; the whole session is loaded first and later split into artificial trial blocks.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])

sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])

F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset consists of 6 `jm*` subjects with 6-7 daily sessions each, and that motion energy was already precomputed in `move_deve/motion_energy_glob.npy`; the notes also say dropped frames are handled using `tstamps.npy`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined as top-level directories whose names start with `jm`, sorted alphabetically.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('jm')])
```

iii. `CONVERSION_NOTES.md` explicitly lists 6 subjects (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) and treats each as one mouse.

## 1-c. How are the data split into sessions?

i. Sessions are defined as all subdirectories inside each subject directory, sorted alphabetically. Each subject-session pair becomes one dataset session.

ii. 
```python
subj_dir = os.path.join(data_dir, subj)
sessions = sorted([d for d in os.listdir(subj_dir)
                  if os.path.isdir(os.path.join(subj_dir, d))])
```

iii. The notes describe "6-7 daily sessions per subject (41 sessions total)," which is the rationale for treating each subdirectory as a daily recording session.

## 1-d. How are the data split into trials?

i. The AI does not use a natural trial structure from the raw data. Instead, it bins the continuous session into 10-frame time bins first, then splits each session into consecutive 2-minute blocks and treats those blocks as trials.

ii. 
```python
BIN_SIZE = 10
TRIAL_DURATION_S = 120

bins_per_trial = int(trial_duration_s * FS / BIN_SIZE)  # 120 * 30 / 10 = 360
n_bins = dff_binned.shape[1]
n_trials = n_bins // bins_per_trial

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
    time_trials.append(time_bins[start:end])
```

iii. The stated justification in `CONVERSION_NOTES.md` is that the paper’s decoding section used "consecutive 2 minute blocks of the recording" for cross-validation, so the AI used 2-minute blocks as decoder trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply an explicit trial-quality filter. It only keeps complete 2-minute trial blocks after binning; leftover bins at the end of a session are dropped implicitly by integer division.

ii. 
```python
n_bins = dff_binned.shape[1]
n_trials = n_bins // bins_per_trial

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    ...
```

iii. No explicit trial-QC justification was documented in the notes or trajectory. The code only reflects the choice to use complete fixed-length blocks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from Suite2p fluorescence outputs `F.npy` and `Fneu.npy` in `suite2p/plane0`.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
```

iii. `CONVERSION_NOTES.md` says the AI followed the paper’s description of using Suite2p outputs and default Suite2p parameters.

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F manually. It subtracts neuropil (`F - 0.7 * Fneu`), estimates a Suite2p-style "maximin" baseline using Gaussian smoothing then min/max filters, divides by the baseline to form dF/F, and then averages the result in 10-frame bins.

ii. 
```python
Fc = F - neucoeff * Fneu

Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)

Flow = np.maximum(Flow, 1e-6)
dff = (Fc - Flow) / Flow

dff_binned = bin_data(dff, BIN_SIZE)
```

iii. The notes justify this as following the paper’s statement that "baseline corrected fluorescence traces" were used as dF/F with default Suite2p parameters, plus the paper’s statement about averaging in bins of 10 timestamps for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no explicit neuron-level quality filter in `convert_data.py`. It loads whatever ROIs are present in `F.npy`/`Fneu.npy` and carries them through.

ii. 
```python
F = np.load(os.path.join(s2p_dir, 'F.npy'))
Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
...
n_neurons, n_frames = F.shape
...
brain_region_idx_list.append(np.zeros(sess_data['n_neurons'], dtype=int))
```

iii. No direct code-level justification was recorded beyond the notes’ assumption that the Track2p/Suite2p outputs already represent tracked neurons, and the sanity check claim that neuron counts are stable across sessions within each mouse.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI effectively aligns neural data to recording onset: it bins the full session from the start, then chops the binned session into consecutive 2-minute windows. It records the alignment event in metadata as `recording_onset`.

ii. 
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])

'metadata': {
    'temporal_alignment_event': 'recording_onset',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_S),
}
```

iii. The notes justify this by defining the decoder input as "Time elapsed from the beginning of the experiment" and by treating the recording as continuous spontaneous behavior rather than event-triggered trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 10-frame bins at 30 Hz, so each time point is 333.33 ms. Yes, explicit temporal rebinning is applied to both neural and behavioral streams.

ii. 
```python
BIN_SIZE = 10
FS = 30

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)

'time_bin_size': BIN_SIZE / FS * 1000,
```

iii. `CONVERSION_NOTES.md` explicitly cites the paper’s decoding text about "averaging in bins of 10 consecutive timestamps" as the reason for the 333.33 ms time bin.

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. It is not loaded from a raw timestamp variable. The AI derives it from bin indices and the known imaging rate.

ii. 
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
```

iii. The notes say there is no separate raw "time" signal used for the decoder input; instead, the decoder input is "Time in seconds from recording onset."

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. After temporal binning, the AI computes the center time of each 10-frame bin as `(bin_index * 10 + 5) / 30` seconds, then slices those binned times into the same 2-minute trial windows.

ii. 
```python
bin_centers = (np.arange(dff_binned.shape[1]) * BIN_SIZE + BIN_SIZE / 2) / FS
...
time_trials.append(time_bins[start:end])
...
sess_input.append(t_trial.reshape(1, -1))
```

iii. `CONVERSION_NOTES.md` states this explicitly: "Computed as bin center times: `(bin_index * 10 + 5) / 30` seconds."

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. The time input is built on exactly the same binned session grid as the neural data and is sliced with the same trial boundaries, so each time value corresponds one-to-one with a neural time bin.

ii. 
```python
dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
neural_trials, me_trials, time_trials = split_into_trials(
    dff_binned, me_binned, time_bins
)
...
sess_input.append(t_trial.reshape(1, -1))
```

iii. The notes justify this through the general choice to bin both neural and behavioral data identically and to define the decoder input from recording onset on that shared grid.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The motion-energy output is derived from `move_deve/motion_energy_glob.npy` together with `move_deve/tstamps.npy`, which the AI uses to infer dropped video frames.

ii. 
```python
me = np.load(os.path.join(move_dir, 'motion_energy_glob.npy'))
tstamps = np.load(os.path.join(move_dir, 'tstamps.npy'))
```

iii. The notes say the motion-energy signal was already provided in `motion_energy_glob.npy`, and that missing frames would be handled using timestamp gaps from `tstamps.npy`.

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates motion energy to the 2-photon frame grid using camera timestamp gaps, averages motion energy in 10-frame bins, then later discretizes it globally into five percentile bins. The notes also claim motion energy is normalized, although the script itself does not perform an explicit standard-deviation normalization.

ii. 
```python
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
...
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx
...
me_interp = np.interp(all_indices, frame_indices, me)
...
me_binned = bin_data(me, BIN_SIZE)
```

iii. `CONVERSION_NOTES.md` justifies this with the paper’s description of motion-energy computation, the claim that some video frames were dropped, the choice to linearly interpolate missing frames, and the paper’s 10-frame averaging for decoding.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. The AI concatenates motion-energy values from all sessions after binning, computes global quintile thresholds, and converts each value into category 0-4 with `np.digitize`.

ii. 
```python
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)

percentiles = np.linspace(0, 100, n_bins + 1)
thresholds = np.percentile(all_me_values, percentiles)

binned = np.digitize(me_values, thresholds[1:-1])
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes explicitly justify this using the task requirement for "five equal-percentile bins" and say that global thresholds keep bin definitions consistent across sessions.

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy by interpolating it onto the inferred 2-photon frame index grid, then bins motion energy and neural data with the same 10-frame bins and slices them into the same 2-minute trial windows.

ii. 
```python
me = interpolate_motion_energy(me, tstamps, n_frames)

dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)

neural_trials, me_trials, time_trials = split_into_trials(
    dff_binned, me_binned, time_bins
)
```

iii. The notes say videography was synchronized to the microscope trigger, but camera drops required interpolation before placing motion energy on the same grid as neural activity.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing video frames by detecting irregular gaps in `tstamps.npy` and interpolating motion energy onto a full 2-photon frame grid. It also discards incomplete remainders when binning or when dividing sessions into full 2-minute trials.

ii. 
```python
if len(me) == n_frames:
    return me
...
ifi = np.diff(tstamps)
median_ifi = np.median(ifi)
...
me_interp = np.interp(all_indices, frame_indices, me)

n_bins = n // bin_size
n_use = n_bins * bin_size
...
n_trials = n_bins // bins_per_trial
```

iii. `CONVERSION_NOTES.md` explicitly discusses dropped camera frames and says they were repaired by interpolation; no further error-handling rationale was documented.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive work is the per-session neural preprocessing in `compute_dff` (Gaussian smoothing and running min/max filters over every neuron and frame). Secondary costs come from motion-energy interpolation and repeated session-wide binning/splitting.

ii. 
```python
Flow = gaussian_filter1d(Fc.astype(np.float64), sigma=sig, axis=1)
Flow = minimum_filter1d(Flow, size=win, axis=1)
Flow = maximum_filter1d(Flow, size=win, axis=1)
...
me_interp = np.interp(all_indices, frame_indices, me)
...
dff_binned = bin_data(dff, BIN_SIZE)
me_binned = bin_data(me, BIN_SIZE)
```

iii. No explicit efficiency justification was recorded in the notes or trajectory. This answer is inferred from the operations applied to full-session arrays.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the Python loop that builds `frame_indices` in `interpolate_motion_energy`, the trial-splitting loop in `split_into_trials`, and the per-trial loops used later to reshape inputs and discretize outputs.

ii. 
```python
for i in range(len(ifi)):
    n_skipped = int(np.round(ifi[i] / median_ifi))
    cum_idx += n_skipped
    frame_indices[i + 1] = cum_idx

for t in range(n_trials):
    start = t * bins_per_trial
    end = start + bins_per_trial
    neural_trials.append(dff_binned[:, start:end])
    me_trials.append(me_binned[start:end])
    time_trials.append(time_bins[start:end])

for t_trial in sess_data['time_trials']:
    sess_input.append(t_trial.reshape(1, -1))
for me_trial in sess_data['me_trials']:
    me_disc = apply_discretization(me_trial, thresholds)
    sess_output.append(me_disc.reshape(1, -1))
```

iii. No explicit vectorization discussion was documented by the AI; this is inferred from the code structure.

## 6-c. What processing does the code repeat multiple times?

i. The AI makes multiple passes over the same session data: it first loads, processes, bins, and splits sessions to collect motion-energy values for global thresholding, then loops through all sessions again to build the final `neural`, `input`, and `output` structures. Within that second pass it loops trial-by-trial again to reshape inputs and discretize outputs.

ii. 
```python
all_sessions_data = []
all_me_flat = []
...
dff_binned, me_binned, time_bins = load_and_process_session(sess_dir)
neural_trials, me_trials, time_trials = split_into_trials(
    dff_binned, me_binned, time_bins
)
...
all_me_concat = np.concatenate(all_me_flat)
thresholds = discretize_motion_energy(all_me_concat)
...
for sess_data in all_sessions_data:
    ...
    for t_trial in sess_data['time_trials']:
        sess_input.append(t_trial.reshape(1, -1))
    for me_trial in sess_data['me_trials']:
        me_disc = apply_discretization(me_trial, thresholds)
        sess_output.append(me_disc.reshape(1, -1))
```

iii. No explicit justification for these repeated passes was recorded. They appear to arise from the choice to compute global motion-energy thresholds before assembling the final dataset.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs some bookkeeping and extra output work that is not used by downstream decoding: it stores each session name in `all_sessions_data` but never uses it later, keeps unused `sample_mode` and `n_2p_frames` arguments, imports `sys` without using it, and always creates a `sample_data.pkl` copy in addition to the full dataset.

ii. 
```python
import sys
...
def load_and_process_session(session_dir, n_2p_frames=None):
...
def convert_data(data_dir=DATA_DIR, sample_mode=False):
...
all_sessions_data.append({
    'subject': subj,
    'session': sess_name,
    'neural_trials': neural_trials,
    'me_trials': me_trials,
    'time_trials': time_trials,
    'n_neurons': n_neurons,
})
...
sample = create_sample(data)
sample_path = os.path.join('/app', 'sample_data.pkl')
with open(sample_path, 'wb') as f:
    pickle.dump(sample, f)
```

iii. No explicit justification for these extras was documented in the notes or trajectory.
