# CA1 reward-relative coding dataset — decoder-ready conversion

Converted from **DANDI 001361**, the two-photon calcium imaging dataset of
Sosa, Plitt & Giocomo (2025), *A flexible hippocampal population code for experience relative to reward*, Nature Neuroscience 28:1497-1509.

## Dataset description

11 head-fixed mice ran laps on a 450 cm virtual linear track while dorsal CA1 neurons
expressing GCaMP7f were imaged at 15.5 Hz. A hidden 50 cm reward zone (A: 80-130 cm,
B: 200-250 cm, C: 320-370 cm) delivered sucrose operantly when the mouse licked inside
it; reward was randomly omitted on ~15% of trials. On "switch" days the zone moved to a
new location after 30 trials, and on some days the switch coincided with a change of
visual environment (ENV 1 <-> ENV 2). Each session is one day (14 days/mouse; m11 was
imaged from day 3, giving 12).

| Statistic | Value |
|---|---|
| Subjects | 11 (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) |
| Sessions | 152 |
| Trials | 11,983 (mean 78.8/session) |
| Neurons | 138,276 CA1 cells (154-2,323 per session, mean 910) |
| Timepoints | 2,533,868 imaging frames (45.4 h) |
| Time bin | 64.48 ms (15.5078 Hz imaging frame rate) |
| Brain region | dorsal hippocampus CA1 |
| Neural signal | dF/F (neuropil-corrected, per-trial maximin baseline, 2-frame Gaussian smoothing) |
| Alignment | each trial starts at the trial-start frame (track entry at 0 cm) and ends at the teleport frame |

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]      # session 0, trial 0: (n_neurons, n_timepoints) dF/F
inputs = data['input'][0][0]       # (4, n_timepoints)
outputs = data['output'][0][0]     # (6, n_timepoints), integer class labels
print(data['metadata']['session_info'][0])   # subject, day, scene, cell count, ...
```

Re-run the conversion with:

```bash
python -u convert_data.py out.pkl --full                 # all 152 sessions (~2.5 min, 12 workers)
python -u convert_data.py out.pkl --sample               # 2 sessions
python -u convert_data.py out.pkl --sample --show-processing   # + QC figures
python -u convert_data.py out.pkl --full --signal events # use the paper's OASIS-deconvolved events instead of dF/F
```

Validate / train the decoder:

```bash
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl --plot-samples
```

## Output format specification

```
data = {
  'neural':  [session][trial] -> float32 (n_neurons, T)   # dF/F
  'input':   [session][trial] -> float32 (4, T)
  'output':  [session][trial] -> int64   (6, T)
  'subjects': ['m3', 'm4', ...],            'subject_idx': int array (n_sessions,)
  'brain_regions': ['CA1'],                 'brain_region_idx': [int array (n_neurons,)]
  'input_names', 'output_names', 'output_values', 'metadata'
}
```

**Inputs** (time-varying arrays; constants are broadcast across the trial)

| # | Name | Description |
|---|---|---|
| 0 | `time_from_trial_start` | seconds since track entry |
| 1 | `environment` | 0 = ENV 1, 0/1 per trial |
| 2 | `trial_number` | 0-based trial index in the session |
| 3 | `prev_trial_outcome` | 0 = previous trial omitted, 1 = rewarded |

**Outputs** (categorical)

| # | Name | Classes |
|---|---|---|
| 0 | `dist_to_reward_zone` | 0: <-50 cm, 1: -50..-10, 2: -10..<0, 3: 0 (inside the zone), 4: >0..+10, 5: +10..+50, 6: >+50 cm |
| 1 | `position` | 0: <90, 1: 90-180, 2: 180-270, 3: 270-360, 4: >360 cm |
| 2 | `speed` | 0: <2, 1: 2-10, 2: 10-20, 3: 20-40, 4: >40 cm/s |
| 3 | `lick` | 0: no lick, 1: lick in this frame |
| 4 | `reward_zone_location` | 0: A, 1: B, 2: C |
| 5 | `reward_outcome` | 0: omitted, 1: rewarded |

Distances are signed and measured to the nearest edge of the active reward zone (0 while inside it).

## Curation applied

- ROIs: suite2p `iscell == 1` (manually curated pyramidal cells), planes pooled, minus
  putative interneurons with Pearson r(dF/F, speed) > 0.5 (402 cells, 0.29%).
- Trials: on-track laps only (ITI excluded — the laser was blanked there); the first trial
  of each session is dropped (previous-trial outcome undefined); the 81 lick-sensor-error
  trials identified by the paper's criterion are dropped.

## Decoder performance (validation balanced accuracy, full dataset)

| Output | Accuracy | Chance |
|---|---|---|
| dist_to_reward_zone | 0.631 | 0.143 |
| position | 0.772 | 0.200 |
| speed | 0.626 | 0.200 |
| lick | 0.768 | 0.500 |
| reward_zone_location | 0.879 | 0.333 |
| reward_outcome | 0.613 | 0.500 |

See `CONVERSION_NOTES.md` for the full provenance, validation and decision log.
