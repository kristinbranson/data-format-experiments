# Converted dataset: Track2p longitudinal imaging of developing mouse barrel cortex

This directory contains a decoder-ready conversion of the dataset from

> Majnik J., Mantez M., Zangila S., Bugeon S., Guignard L., Platel J.-C., Cossart R. (2025)
> *Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p.*
> eLife 14:RP107540. https://doi.org/10.7554/eLife.107540

## Dataset description

Daily two-photon calcium imaging (GCaMP8m, 30 Hz, 720x720 um FOV, layer 2/3) of the **same**
neurons (tracked across days with Track2p) in the **barrel cortex (S1)** of 6 mouse pups during
the second postnatal week (P7-P14). Mice were head-fixed on a non-motorised treadmill in the dark;
there is no stimulus and no task. Spontaneous behaviour was filmed at 30 Hz (camera triggered by
the microscope) and quantified as **motion energy** (sum of squared pixel-wise differences between
consecutive video frames).

| | |
|---|---|
| Subjects | 6 (jm031=A, jm032=B, jm038=C, jm039=D, jm040=E, jm046=F) |
| Sessions | 41 (7,7,7,7,6,7 daily recordings) |
| Neurons tracked per mouse | 221, 370, 685, 746, 541, 435 (mean 500 +- 198; paper: 526 +- 190) |
| Session length | 20 min (jm031, jm032) or 30 min (others) at 30 Hz |
| Trials | consecutive non-overlapping 60 s blocks: 20 or 30 per session, **1090 total** |
| Time bin | 10 imaging frames = 333.33 ms (as used for all decoding analyses in the paper) |
| Timepoints per trial | 180 |
| Brain region | S1 (barrel cortex), layer 2/3 |

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 180) float32, binned dF
x      = data['input'][session][trial]    # (1, 180) float32, elapsed session time (s)
y      = data['output'][session][trial]   # (1, 180) int64, motion-energy quintile 0..4
```

Train/evaluate the reference decoder:

```bash
python train_decoder.py /app/converted_data.pkl              # train
python train_decoder.py /app/converted_data.pkl --verify-only # format check + summary
```

Regenerate the dataset:

```bash
python -u convert_data.py /app/converted_data.pkl --full        # ~7 s with 6 processes
python -u convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Output format specification

| Key | Content |
|-----|---------|
| `neural` | list over 41 sessions of lists over trials of `(n_neurons, 180)` float32 arrays. Baseline-corrected fluorescence (`F - F0`, suite2p maximin baseline, sigma=10 frames, 60 s window, neuropil coefficient 0 - the reference `F_processing`), averaged in non-overlapping bins of 10 frames. |
| `input` | `(1, 180)` float32 per trial: `time_from_session_start_s`, the elapsed time at each bin centre, continuous across trials (0.15 s ... 1199.85/1799.85 s). |
| `output` | `(1, 180)` int64 per trial: `motion_energy_quintile` in {0,1,2,3,4}, the quintile of the binned motion energy with thresholds computed **per session** (exactly 20% of bins per class). |
| `subjects` / `subject_idx` | 6 mouse ids; index of the mouse for each session. |
| `brain_regions` / `brain_region_idx` | `['S1']`; zeros for every neuron. |
| `input_names` / `output_names` / `output_values` | `['time_from_session_start_s']`, `['motion_energy_quintile']`, `[['q1_lowest','q2','q3','q4','q5_highest']]`. |
| `metadata` | task description, `time_bin_size` (333.33 ms), alignment event (session start), `off_start`=0, `off_end`=60 s, processing descriptions, imaging rate, and a `session_info` entry per session (subject, date, day index, n_neurons, n_frames, n_trials, camera frames, interpolated frames, quintile edges, class counts). |

## Key statistics and decoder performance

- Format verification: valid, **no errors or warnings** (`verification_full_out.txt`).
- Class distribution: exactly 20% per quintile in every session.
- Decoder (provided `train_decoder.py`, 200 epochs, 100 PCs/session, balanced loss):
  **training balanced accuracy 0.599, validation 0.305** (chance 0.200); see `train_decoder_full_out.txt`.
  Restricting to the later (older) sessions raises validation accuracy to 0.340, consistent with the
  paper's finding that behavioural-state coding emerges after P11.
- Replication of the paper's analysis on the converted data (ridge regression, 2-min-block CV):
  R2 ~ 0 at the youngest ages and up to **0.76** on the last days, matching Fig. 7B/C of the paper.

See `CONVERSION_NOTES.md` for the full record of decisions, checks and validation results.
