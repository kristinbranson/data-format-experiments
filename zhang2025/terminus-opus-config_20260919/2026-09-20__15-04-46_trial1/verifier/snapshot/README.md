# IBL Brain-Wide Map -> neural-decoder dataset

`/app/converted_data.pkl` (11.3 GB) is the IBL Brain-Wide Map public release reformatted for
the decoder in `/app/train_decoder.py`.

## Dataset description

Head-fixed mice perform the IBL decision-making task: a Gabor stimulus (contrast 0, 6.25,
12.5, 25 or 100%) appears on the left or right of a screen and the mouse turns a wheel to move
it to the centre. The prior probability that the stimulus appears on the left is 0.5 for the
first 90 trials of a session and then alternates between 0.2 and 0.8 in blocks of 20-100
trials. Neuropixels probes record from across the brain while wheel movement and whisker-pad
motion energy are measured.

Each trial is a 2 s window aligned to **stimulus onset** (`trials.stimOn_times`), spanning
**-0.5 s to +1.5 s** and binned into **100 non-overlapping 20 ms bins**, following
`/app/code/code_zhang2025/src/0_data_caching.py`.

## Key statistics

| | |
|---|---|
| Sessions | 440 (of the 459 in the release freeze) |
| Subjects (mice) | 136 |
| Brain regions (Beryl) | 209 |
| Neurons | 60,483 (mean 137.5 per session, range 5-508) |
| Trials | 187,651 (mean 426.5 per session) |
| Timepoints per trial | 100 |
| Bin size | 20 ms |
| Alignment | visual stimulus onset |
| Window | -0.5 s to +1.5 s |

Neurons are **well-isolated units** (`clusters.label >= 1`, i.e. all three RIGOR single-unit
metrics of the data paper: amplitude > 50 uV, noise cut-off < 20 uV, refractory-period
violation), restricted to grey-matter Beryl regions with at least 5 such units in the session
and present in at least 2 sessions. Loading reproduces the data paper's 889 Kilosort units per
probe (888.7 here) and 108 well-isolated units per probe (108.2 here).

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# session 0, trial 5
neural = data['neural'][0][5]   # (n_neurons, 100) float32 spike counts per 20 ms bin
inp    = data['input'][0][5]    # (2, 100) float32
out    = data['output'][0][5]   # (4, 100) int64 class labels

subject = data['subjects'][data['subject_idx'][0]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][0]]
```

Validate and train the reference decoder with:

```bash
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl
```

Regenerate the dataset with:

```bash
python -u convert_data.py converted_data.pkl --full        # all sessions, ~3 min on 32 cores
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Output format specification

```
data = {
  'neural'           : list[n_sessions] of list[n_trials] of (n_neurons, 100) float32
  'input'            : list[n_sessions] of list[n_trials] of (2, 100) float32
  'output'           : list[n_sessions] of list[n_trials] of (4, 100) int64
  'subjects'         : list[136] of str
  'subject_idx'      : (440,) int64
  'brain_regions'    : list[209] of str      (Beryl acronyms)
  'brain_region_idx' : list[440] of (n_neurons,) int64
  'input_names'      : list[2] of str
  'output_names'     : list[4] of str
  'output_values'    : list[4] of list of str
  'metadata'         : dict
}
```

### Inputs (decoder inputs)
| i | `input_names[i]` | Type | Description |
|---|---|---|---|
| 0 | `time_from_stimulus_onset` | continuous, time-varying | right edge of each 20 ms bin relative to stimulus onset, -0.48 to +1.50 s |
| 1 | `trial_number_in_block` | continuous, per-trial (broadcast over time) | 0-based index of the trial within its constant-`probabilityLeft` block, 0-98 |

### Outputs (to be decoded)
| i | `output_names[i]` | Classes (`output_values[i]`) | Type | Distribution |
|---|---|---|---|---|
| 0 | `choice` | `left` = 0, `right` = 1 | binary, per-trial | 0.508 / 0.492 |
| 1 | `prior_probability_left` | `0.2` = 0, `0.5` = 1, `0.8` = 2 | 3-class, per-trial | 0.417 / 0.140 / 0.442 |
| 2 | `wheel_speed` | `low`, `medium`, `high` | 3-class, time-varying | 1/3 each |
| 3 | `whisker_motion_energy` | `low`, `medium`, `high` | 3-class, time-varying | 1/3 each |

Wheel speed (`abs(wheel.velocity)`, 1 kHz) and whisker motion energy (left camera at 60 Hz,
right camera as fallback) are linearly interpolated onto the right edge of each 20 ms bin and
then split at the **per-session 33.3rd and 66.7th percentiles** of all (trial x timepoint)
samples. The split is per session because motion energy is in uncalibrated, camera-dependent
units, so a global threshold would largely encode session identity.

### Metadata
`task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`, `off_start` (-0.5),
`off_end` (+1.5), `n_timepoints`, `bin_times_s`, `bin_time_convention`, `neural_units`,
`atlas_mapping`, `neuron_curation`, `trial_curation`, `behaviour_sources`, `discretization`,
`source`, `reference_code`, and `session_info` (per-session eid, subject, n_neurons, n_trials).

## Curation

**Trials** (identical to the reference `load_trials_and_mask(..., max_trial_len=10.0)`):
no NaN in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`
or `feedbackType`; 0.08 s <= `firstMovement_times - stimOn_times` <= 2.0 s;
`feedback_times - goCue_times` <= 10 s; `choice != 0`. Additionally the wheel and whisker
traces must cover the trial window, and the trial window must lie inside the span of the spike
sorting and contain at least one spike.

**Sessions**: all 459 release eids are attempted; 19 are dropped (14 have no whisker motion
energy, 4 have no Beryl region with >= 5 well-isolated units, 1 has too few usable trials).

## Reference decoder accuracy

| Output | Chance | Validation balanced accuracy |
|---|---|---|
| choice | 0.500 | 0.614 |
| prior_probability_left | 0.333 | 0.656 |
| wheel_speed | 0.333 | 0.600 |
| whisker_motion_energy | 0.333 | 0.587 |

These are averages over all 100 time bins. Choice is at chance before the stimulus appears and
peaks at 0.82 around 0.24 s after onset (see `CONVERSION_NOTES.md`, Step 12).

## Files

| File | Contents |
|---|---|
| `convert_data.py` | conversion script |
| `converted_data.pkl` | full dataset (440 sessions) |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | every decision, check and validation result |
| `conversion_full_out.txt` / `conversion_sample_out.txt` | conversion logs |
| `verification_full_out.txt` / `verification_sample_out.txt` | format-verification logs |
| `train_decoder_full_out.txt` / `train_decoder_sample_out.txt` | decoder training logs |
| `processing_<eid>.png` | per-step processing checks for 2 sessions |
| `sample_trials.png`, `predictions.png` | decoder diagnostics |
| `cache/` | investigation and sanity-check scripts |
