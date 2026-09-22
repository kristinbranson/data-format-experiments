# Converted dataset: Majnik et al. 2025 (Track2p) - barrel cortex development

`converted_data.pkl` contains the longitudinal 2-photon calcium imaging dataset from

> Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel J-C, Cossart R (2025)
> *Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p*.
> eLife 14:RP107540. https://doi.org/10.7554/eLife.107540

reformatted for training a neural decoder that predicts the animal's **motion energy**
(discretized into five per-session equal-percentile bins) from population calcium activity.

## Dataset description

- 6 mouse pups (jm031, jm032, jm038, jm039, jm040, jm046 = mice A-F in the paper), imaged daily
  during the second postnatal week (P7-P14), 6-7 consecutive days each -> **41 sessions**.
- Layer 2/3 of barrel cortex (S1), 720 x 720 um FOV, 30 Hz resonant scanning, GCaMP8m.
- Only neurons tracked across **all** days of a mouse (Track2p) and classified as cells by suite2p
  (prob > 0.5) are included; rows are matched across the sessions of a mouse.
  Neurons per mouse: 221, 370, 685, 746, 541, 435 (**2998** unique neurons, 20445 neuron-sessions).
- Spontaneous behaviour in the dark on a non-motorised treadmill; behaviour is quantified by the
  videography **motion energy** (sum of squared pixel differences between consecutive frames),
  recorded by a camera hardware-triggered by the microscope (so video frame i == imaging frame i).

## Processing

| Stream | Processing |
|---|---|
| neural | `suite2p/plane0/F.npy` -> baseline-corrected fluorescence dF using the reference implementation `track2p/gui/data_management.py::F_processing` (neucoeff=0.0, maximin baseline, gaussian sigma = 10 frames, 60 s min/max window) -> averaged in non-overlapping bins of **10 frames** (the paper's decoding preprocessing) = 3 Hz |
| input | elapsed time from the start of the session (s), at the bin centres |
| output | motion energy placed on the imaging-frame grid (dropped camera frames recovered from `move_deve/interframe_int.npy` and linearly interpolated; the always-zero first sample is interpolated too) -> same 10-frame averaging -> discretized into **5 equal-percentile bins with thresholds computed within each session** |
| trials | each session is cut into consecutive non-overlapping **60 s** trials = 1800 frames = 180 bins. 20 trials for 20-min sessions (jm031, jm032), 30 for 30-min sessions -> **1090 trials** |

No neurons, trials or sessions were discarded: the released data is already curated, `badframes` is
empty in every session, and all session lengths are exact multiples of 1800 frames.

## How to load

```python
import pickle, numpy as np
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['neural'][0][0]     # session 0, trial 0: (n_neurons, 180) binned dF, float32
t = data['input'][0][0]      # (1, 180) time from session start, seconds
y = data['output'][0][0]     # (1, 180) motion-energy quintile, int in 0..4
data['subjects'][data['subject_idx'][0]]        # mouse of session 0
data['metadata']['session_info'][0]             # per-session details (date, #neurons, quantile edges, ...)
```

## Format specification

| Key | Content |
|---|---|
| `neural` | list of 41 sessions; each a list of trials; each trial `(n_neurons, 180)` float32 binned dF |
| `input` | same nesting; `(1, 180)` float32; `input_names = ['time_from_session_start_s']` |
| `output` | same nesting; `(1, 180)` int64 in 0..4; `output_names = ['motion_energy_quintile']`, `output_values = [['quintile_1' ... 'quintile_5']]` |
| `subjects` | `['jm031','jm032','jm038','jm039','jm040','jm046']` |
| `subject_idx` | `(41,)` int index into `subjects` |
| `brain_regions` | `['S1 barrel cortex (L2/3)']` |
| `brain_region_idx` | list of 41 arrays of zeros, one entry per neuron |
| `metadata` | task description, `time_bin_size` = 333.33 ms, alignment (session start), `off_start` = 0, `off_end` = 60 s, processing description and per-session `session_info` |

## Key statistics

| Statistic | Value |
|---|---|
| Sessions / subjects / trials | 41 / 6 / 1090 |
| Timepoints per trial | 180 (60 s at 3 Hz) |
| Neurons per session | mean 498.7 (min 221, max 746) |
| Output class distribution | 0.200 per class in every session (quintiles by construction) |
| Input range | 0.15 - 1799.8 s |
| Decoder (provided script) | training balanced accuracy 0.610, validation 0.308 (chance 0.200) |
| Paper-style ridge regression on this data | R2 ~0-0.2 on early days, 0.43-0.79 on the last day of each mouse, reproducing Fig. 7C |

See `CONVERSION_NOTES.md` for the full decision log, sanity checks and validation results, and
`convert_data.py` for the conversion code.
