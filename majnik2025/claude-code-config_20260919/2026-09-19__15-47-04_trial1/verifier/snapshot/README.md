# Track2p developing-barrel-cortex dataset, converted for neural decoding

`converted_data.pkl` contains the longitudinal two-photon dataset of **Majnik et al. 2025**
(*Longitudinal tracking of neuronal activity from the same cells in the developing brain using
Track2p*, eLife 14:RP107540, https://doi.org/10.7554/eLife.107540) reformatted so that a decoder can
predict the animal's **motion energy** from **barrel-cortex population activity**.

## Dataset description

Six GAD67-Cre mouse pups were imaged daily through the second postnatal week (P7–P14) with
two-photon calcium imaging (GCaMP8m, 30 Hz, 720 × 720 µm FOV, layer 2/3 of the barrel cortex).
Recordings are of spontaneous activity: the pups were head-fixed on a non-motorised treadmill, in
the dark, under sensory-minimised conditions — there is no task and no stimulus. Spontaneous
movement was filmed at 30 Hz by a camera triggered by the microscope and summarised as *motion
energy* (summed squared pixel-wise difference between consecutive video frames). Track2p matched
every neuron across all recording days of a mouse, so row *i* of a session's activity matrix is the
same cell on every day of that mouse.

| | |
|---|---|
| Subjects | 6 (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046` = mice A…F of the paper) |
| Sessions | 41 (7, 7, 7, 7, 6, 7 per mouse; one per postnatal day, P7–P14) |
| Neurons | 220, 367, 682, 746, 541, 434 per mouse (20,389 summed over sessions; 497.3 per session) |
| Trials | 1090 (consecutive 60 s blocks; 20 per 20-min session, 30 per 30-min session) |
| Time bin | 333.33 ms (10 imaging frames at 30 Hz), identical for all trials/sessions |
| Timepoints per trial | 180 |
| Brain region | `S1` (barrel cortex, layer 2/3) |
| Decoder input | `time_in_session_s` — time from the first imaging frame (s), time-varying |
| Decoder output | `motion_energy_quintile` — motion energy in 5 equal-percentile bins per session, time-varying |

## How to load and use

```python
import pickle
import numpy as np

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

sess, trial = 21, 0
X = data['neural'][sess][trial]    # (n_neurons, 180) float32  dF/F, 333 ms bins
u = data['input'][sess][trial]     # (1, 180)        float32  seconds since session start
y = data['output'][sess][trial]    # (1, 180)        int8     motion-energy quintile, 0..4

print(data['subjects'][data['subject_idx'][sess]])          # 'jm039'
print(data['metadata']['session_info'][sess]['postnatal_day'])   # 8
print(data['output_values'][0])    # ['q1 (lowest 20%)', 'q2', ..., 'q5 (highest 20%)']
```

Train/validate the provided decoder:

```bash
python train_decoder.py converted_data.pkl --verify-only   # structural checks + summary
python train_decoder.py converted_data.pkl --plot-samples  # train + accuracy + figures
```

Regenerate the file (≈40 s for the whole dataset):

```bash
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing   # 2 sessions + diagnostics
```

## Output format

```python
data = {
  'neural':      [ [ (n_neurons, 180) float32, ... ] , ... ],   # per session, per trial
  'input':       [ [ (1, 180) float32, ... ], ... ],
  'output':      [ [ (1, 180) int8,    ... ], ... ],
  'subjects':          ['jm031','jm032','jm038','jm039','jm040','jm046'],
  'subject_idx':       np.ndarray (41,) int64,
  'brain_regions':     ['S1'],
  'brain_region_idx':  [ np.zeros(n_neurons, int64), ... ],     # one array per session
  'input_names':       ['time_in_session_s'],
  'output_names':      ['motion_energy_quintile'],
  'output_values':     [['q1 (lowest 20%)','q2','q3 (middle 20%)','q4','q5 (highest 20%)']],
  'metadata':          {...},
}
```

`metadata` documents the task, the processing and the curation, and contains `session_info`: one
dict per session with `subject`, `mouse_letter`, `session_name`, `date`, `day_index`,
`postnatal_day`, `n_neurons`, `n_frames`, `n_bins`, `n_trials`, `n_missing_camera_frames`,
`n_failed_rois`, `session_duration_s` and the four `quantile_edges` used to discretise that
session's motion energy. Key scalar fields: `time_bin_size = 333.33` (ms),
`temporal_alignment_event = 'start of the imaging session …'`, `off_start = 0.0`, `off_end = 60.0`.

## Processing summary

* **Neural** — `suite2p/plane0/F.npy` of the Track2p-tracked cells → dF/F with the authors' own
  `F_processing()` (copied from `track2p/gui/data_management.py`: maximin baseline, gaussian
  σ = 10 frames, 60 s min/max filter, neuropil coefficient 0) → mean over 10 consecutive frames,
  the denoising the paper uses for all decoding analyses.
* **Behaviour** — `move_deve/motion_energy_glob.npy` put back on the imaging frame grid using the
  camera timestamps (the camera is triggered by the microscope). Dropped camera triggers
  (≤0.4 % of frames, 8/41 sessions) and the first sample (a frame-differencing artefact) are
  linearly interpolated. Averaged over the same 10-frame bins, then digitised at the session's
  20/40/60/80th percentiles.
* **Trials** — consecutive, non-overlapping 60 s blocks (180 bins) from the session start. The
  experiment has no trial structure, so the alignment event is the start of the session.
* **Curation** — the released data already contains only ROIs with suite2p cell probability > 0.5
  that Track2p matched on every day of the mouse. In addition, 8 ROIs whose fluorescence trace is
  identically zero on at least one day (failed signal extraction) were removed from all sessions of
  the affected mouse.

## Validation

* Structural verification: no errors, no warnings (`verification_full_out.txt`).
* Independent re-derivation of neural, input and output values from the raw `.npy` files agrees
  exactly (`cache/sanity_checks_out.txt`, all checks pass).
* The paper's published analyses were reproduced from the converted data: calcium event rates
  (Fig. 5D: mouse D 3.85 → 7.21 /min vs published 3.0 → 7.4) and same-day ridge-regression decoding
  of motion energy (Fig. 7C: early R² = 0.15, late R² = 0.41, best 0.80).
* Provided decoder, full dataset: validation balanced accuracy **0.315** (chance 0.200), training
  0.596; per-session accuracy rises with age (0.295 at ≤P11, 0.326 at >P11, up to 0.51 at P14),
  matching the paper's finding that motion is only encoded in barrel cortex from ~P11 onwards.

See `CONVERSION_NOTES.md` for the full decision log, discrepancy analysis and validation tables.
