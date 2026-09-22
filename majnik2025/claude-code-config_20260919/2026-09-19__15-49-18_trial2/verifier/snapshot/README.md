# Decoding mouse movement from developing barrel cortex (Majnik et al. 2025, Track2p)

`converted_data.pkl` holds the longitudinal two-photon dataset of
[Majnik et al., eLife 14:RP107540](https://doi.org/10.7554/eLife.107540.1) reformatted for the
neural-decoder harness in `train_decoder.py`.

## Dataset

Head-fixed mouse pups run spontaneously on a non-motorised treadmill in the dark, under
sensory-minimised conditions, while layer 2/3 of the barrel cortex is imaged at 30 Hz
(GCaMP8m, 720 × 720 µm FOV). There is no task and no stimulus. An infrared camera, triggered
by the microscope, films the mouse at 30 Hz; the squared pixel-wise difference between
consecutive frames, summed over pixels ("motion energy"), is the behavioural read-out.

Each mouse was imaged daily for 6–7 consecutive days during the second postnatal week
(P7–P14). Track2p matched the same neurons across all days of a mouse, so within a subject
every session contains the *same* neurons in the *same* row order.

| | |
|---|---|
| Subjects | 6 (`jm031`…`jm046` = mice A…F of the paper) |
| Sessions | 41 (7, 7, 7, 7, 6, 7 per mouse), one per day, P7–P14 |
| Session length | 20 min (mice A, B) or 30 min (mice C–F) |
| Neurons per mouse | 221, 370, 685, 746, 541, 435 (20 445 session-neurons in total) |
| Trials | 1090 contiguous 60 s blocks (20 or 30 per session) |
| Time bin | 333.33 ms (10 imaging frames), 180 bins per trial |
| Brain region | S1 barrel cortex, layer 2/3 |

## Decoder task

* **input** (`input_names = ['time from session start (s)']`): elapsed time at the centre of
  each bin, 0.17 … 1199.83 s or 1799.83 s. Shape `(1, 180)` per trial.
* **output** (`output_names = ['motion energy quintile']`): motion energy discretised into 5
  equal-percentile bins, thresholds computed **per session**, so each class holds exactly 20 %
  of a session's bins. Values 0–4 with `output_values[0] = ['0-20%', …, '80-100%']`.
  Shape `(1, 180)` per trial.
* **neural**: dF/F of the tracked neurons, shape `(n_neurons, 180)` per trial, `float32`.

## Processing

1. **dF/F** — `(F − 0.7·Fneu)` minus the suite2p *maximin* baseline (Gaussian σ = 10 frames,
   then a 60 s minimum filter followed by a 60 s maximum filter), i.e. the reference
   implementation `track2p/gui/data_management.py:F_processing` with the suite2p defaults
   stored in `ops`. Computed on the whole session, before trials are cut.
2. **Binning** — dF/F and motion energy are averaged in bins of 10 consecutive frames, the
   denoising the paper applies "for all decoding analysis".
3. **Conditioning** — each neuron's session mean is subtracted and the session is divided by
   one scalar (its pooled s.d.). For a linear readout with a bias term both operations are
   information-preserving; they only improve the conditioning of the decoder's optimisation.
   The divisor is stored per session in `metadata['session_info'][i]['dff_scale_divisor']`.
4. **Behaviour alignment** — camera frame *i* corresponds to imaging frame *i*; dropped camera
   triggers are located from `interframe_int.npy` and the following samples shifted
   accordingly, then the (always single-frame) gaps are linearly interpolated. Dropped frames
   are ≤ 0.41 % of a session.
5. **Trials** — the session is cut into consecutive, non-overlapping 60 s blocks.

No neurons, trials or sessions are discarded: the released suite2p folders already contain only
ROIs that passed the suite2p classifier *and* were tracked by Track2p on every day.

## Using the data

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

data['neural'][3][7]           # session 3, trial 7: (n_neurons, 180) float32 dF/F
data['input'][3][7]            # (1, 180) elapsed time in seconds
data['output'][3][7]           # (1, 180) int64 motion-energy quintile, 0..4
data['subjects'][data['subject_idx'][3]]          # 'jm039'
data['metadata']['session_info'][3]               # date, postnatal day, quintile edges, ...
```

`metadata` also carries `task_description`, `time_bin_size` (333.33 ms),
`temporal_alignment_event` (session start), `off_start` (0.0 s), `off_end` (60.0 s) and a
`session_info` entry per session (subject, mouse label, date, postnatal day, n_neurons,
n_frames, fraction of interpolated camera frames, motion-energy quintile edges and range,
dF/F scale divisor).

## Reproducing

```bash
python -u convert_data.py converted_data.pkl --full                 # 6 s, writes 414 MB
python -u convert_data.py sample_data.pkl --sample --show-processing  # 2 sessions + plots
python -u train_decoder.py converted_data.pkl --plot-samples
python3 cache/sanity_checks.py                                       # 30 independent checks
python3 cache/paper_stats_checks.py                                  # reproduces Figs. 5D, 7D
python3 cache/ridge_fig7c.py                                         # reproduces Fig. 7C
```

## Key results

| | |
|---|---|
| Decoder validation balanced accuracy | **0.322** (chance 0.200) |
| Decoder training balanced accuracy | 0.615 |
| Best sessions | jm046 P13 0.44, P14 0.52 — the sessions with the strongest behaviour coupling |
| Paper Fig. 7C (ridge R² of motion energy) reproduced | early ≤P11 median 0.111, late >P11 median 0.453, max 0.767 |
| Paper Fig. 5D (Ca²⁺ event rate, mouse D) reproduced | 3.4 → 6.0 /min from P8 to P14 |

Decodability of movement is near zero before P11 and rises steeply afterwards — the main
biological result of the paper — so the pooled accuracy is dominated by the many early
sessions in which barrel cortex activity is not yet modulated by behavioural state.

See `CONVERSION_NOTES.md` for every decision, check and validation result.
