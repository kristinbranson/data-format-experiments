# Majnik et al. 2025 (Track2p) — decoder-ready dataset

Converted version of the longitudinal two-photon calcium-imaging dataset from

> Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel J-C, Cossart R (2025)
> *Longitudinal tracking of neuronal activity from the same cells in the developing brain
> using Track2p.* eLife 14:RP107540. https://doi.org/10.7554/eLife.107540

## Dataset description

Six mouse pups were imaged daily through the second postnatal week (P7–P14) with a 720×720 µm,
30 Hz two-photon field of view in layer 2/3 of **barrel cortex (S1)**. Pups were head-fixed on a
non-motorised treadmill, in the dark, under sensory-minimised conditions — there is **no task and
no stimulus**; all activity is spontaneous. Track2p was used to identify the neurons that could be
tracked across *every* recording day of a mouse, so the neuron set (and the row order of the
activity matrices) is identical across all sessions of a mouse.

Simultaneously, an infrared camera (30 Hz, hardware-triggered by the microscope) recorded the pup's
spontaneous movement; "motion energy" is the sum of squared pixel-wise differences between
consecutive video frames and is used as a proxy for behavioural state/arousal.

**The decoding task**: predict the animal's motion energy (discretised into five equal-percentile
bins per session) from the population calcium activity, given also the time elapsed since the start
of the session.

## Key statistics

| Statistic | Value |
|---|---|
| Subjects (mice) | 6 — jm031, jm032, jm038, jm039, jm040, jm046 (= mice A–F in the paper) |
| Sessions | 41 (7, 7, 7, 7, 6, 7 consecutive daily recordings) |
| Neurons | 2990 total; 220 / 367 / 682 / 746 / 541 / 434 per mouse (mean 498 ± 198) |
| Trials | 1090 (20 per 20-min session, 30 per 30-min session) |
| Timepoints per trial | 180 |
| Time bin | 333.33 ms (10 imaging frames at 30 Hz) |
| Trial length | 60 s |
| Brain regions | 1 — `S1` (barrel cortex, layer 2/3) |
| Inputs | 1 — `time_from_session_start_s`, range [0.15, 1799.82] s |
| Outputs | 1 — `motion_energy_quintile`, 5 classes, 20 % each |

## How to load and use

```python
import pickle
import numpy as np

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# one session, one trial
session, trial = 0, 0
X = data['neural'][session][trial]   # (n_neurons, 180) float32 — binned dF/F
u = data['input'][session][trial]    # (1, 180) float32      — time from session start (s)
y = data['output'][session][trial]   # (1, 180) int64        — motion-energy quintile, 0..4

mouse = data['subjects'][data['subject_idx'][session]]
region = data['brain_regions'][data['brain_region_idx'][session][0]]

# reconstruct the whole session as a continuous recording
X_sess = np.concatenate(data['neural'][session], axis=1)   # (n_neurons, n_bins)
t_sess = np.concatenate([u[0] for u in data['input'][session]])
```

Train / evaluate the reference decoder:

```bash
python train_decoder.py converted_data.pkl --verify-only   # format check + summary
python train_decoder.py converted_data.pkl --plot-samples  # train and plot
```

Regenerate the data:

```bash
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Output format specification

```
data = {
  'neural':   [ [ (n_neurons, 180) float32, ... x n_trials ], ... x 41 sessions ],
  'input':    [ [ (1, 180)        float32, ... ], ... ],
  'output':   [ [ (1, 180)        int64,   ... ], ... ],

  'subjects':        ['jm031','jm032','jm038','jm039','jm040','jm046'],
  'subject_idx':     (41,) int64,
  'brain_regions':   ['S1'],
  'brain_region_idx':[ (n_neurons,) int64, ... x 41 ],

  'input_names':   ['time_from_session_start_s'],
  'output_names':  ['motion_energy_quintile'],
  'output_values': [['q1','q2','q3','q4','q5']],

  'metadata': {...},
}
```

### Variables

- **`neural`** — dF/F, i.e. raw Suite2p fluorescence minus a Suite2p "maximin" baseline
  (gaussian smoothing with σ = 10 frames, then a 60 s minimum filter, then a 60 s maximum
  filter; no neuropil subtraction), averaged in non-overlapping bins of 10 frames. This is
  exactly the reference implementation
  (`track2p/gui/data_management.py::F_processing`) and the binning specified in the paper's
  Methods. Units are raw fluorescence a.u.
- **`input[0]` `time_from_session_start_s`** — seconds elapsed since the first imaging frame of
  the session, evaluated at the centre of each 333.33 ms bin. Continuous across trial
  boundaries (trial *k*, bin *j* → `((k*180 + j)*10 + 4.5)/30`).
- **`output[0]` `motion_energy_quintile`** — motion energy mapped onto the imaging-frame grid
  (dropped camera frames reconstructed from `tstamps.npy` and excluded from the bin mean),
  averaged in the same 10-frame bins, then assigned to one of five equal-percentile bins whose
  edges are the 20/40/60/80th percentiles **of that session**. `0` = least motion,
  `4` = most motion. Each class holds exactly 20 % of each session's timepoints.

### Metadata highlights

`metadata['session_info']` has one entry per session with `subject`, `paper_mouse` (A–F),
`session` / `date`, `day_index`, `postnatal_day`, `n_neurons`, `n_frames`, `n_bins`,
`n_trials`, `duration_s`, `motion_energy_quintile_edges`, `class_fractions`,
`n_missing_camera_frames` and `frac_frames_with_behaviour`.

Other fields: `task_description`, `time_bin_size` (333.33 ms), `temporal_alignment_event`
(start of the 60 s trial), `off_start` (0.0 s), `off_end` (60.0 s), `sampling_rate_hz` (30),
`neural_signal`, `neuron_curation`, `input_description`, `output_description`,
`brain_region_description`, `species`, `paper`.

## Curation applied

1. The released Suite2p folders already contain only ROIs with Suite2p cell probability > 0.5
   (the paper's threshold) **and** tracked by Track2p across all recording days of that mouse.
2. In addition, 8 of the 2998 ROIs whose trace is identically zero on at least one day (they
   fall outside the imaged FOV that day) are removed from *every* session of that mouse, so the
   tracked population stays matched across days. 2998 → 2990 neurons.
3. No session or trial is excluded: every session yields 20 or 30 complete 60 s trials and every
   imaging frame enters exactly one bin of exactly one trial.

## Decoder performance

| Output | Chance | Training balanced acc. | Validation balanced acc. |
|---|---|---|---|
| `motion_energy_quintile` | 0.200 | 0.575 | 0.309 |

Per-session validation accuracy reproduces the paper's central result: sessions after P11
(mean 0.324) decode significantly better than sessions up to P11 (mean 0.285; Mann–Whitney
p = 0.017), matching the developmental emergence of behavioural-state modulation reported in
Fig. 7 of the paper.

See `CONVERSION_NOTES.md` for the full decision log, sanity checks and validation results.
