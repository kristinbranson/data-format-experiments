# Track2p Motion-Decoding Conversion

This directory contains a decoder-ready conversion of the Track2p developmental barrel-cortex dataset from Majnik et al. (2025). The converted file is:

- `/app/converted_data.pkl`

The conversion uses the matched-neuron Suite2p exports already present in `/app/data`, reconstructs the paper-consistent neural signal as neuropil-subtracted baseline-corrected fluorescence, aligns motion energy to the imaging frame grid, averages neural and behavior traces in 10-frame bins, and splits sessions into contiguous 60-second trials.

## Dataset summary

- Subjects: 6 mice
- Sessions: 41
- Trials: 1,090 total 60 s trials
- Brain region: barrel cortex layer 2/3
- Imaging rate: 30 Hz
- Converted bin size: 333.33 ms (10 imaging frames)
- Trial length after binning: 180 time bins
- Output target: motion energy discretized into 5 equal-percentile bins per session

## Converted structure

The pickle stores a Python dictionary with these main fields:

- `neural`: list of sessions, each a list of trials, each array shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials, each array shaped `(1, n_timepoints)` containing `time_from_session_start_sec`
- `output`: list of sessions/trials, each array shaped `(1, n_timepoints)` containing categorical `motion_energy_bin`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Loading example

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["input_names"])
print(data["output_names"])
print(len(data["neural"]), "sessions")
print(data["neural"][0][0].shape, "first trial shape")
```

## Output definitions

- Input `time_from_session_start_sec`:
  - Absolute elapsed time from the start of the session
  - Time-varying
  - Expressed at the 10-frame binned time base

- Output `motion_energy_bin`:
  - Time-varying categorical label from 0 to 4
  - Built from session-specific quintiles of 10-frame-averaged motion energy
  - Label names are in `data["output_values"][0]`

## Source-to-converted mapping

- Neural source:
  - `suite2p/plane0/F.npy`
  - `suite2p/plane0/Fneu.npy`
  - `suite2p/plane0/ops.npy`

  Converted as:
  - `dF = F - neucoeff * Fneu`
  - `suite2p.extraction.dcnv.preprocess(...)` with per-session Suite2p parameters
  - 10-frame averaging
  - 60 s trialization

- Behavior source:
  - `move_deve/motion_energy_glob.npy`
  - `move_deve/tstamps.npy`

  Converted as:
  - aligned to the imaging frame grid
  - missing camera frames filled by interpolation
  - 10-frame averaging
  - per-session quintile discretization

## Validation artifacts

The conversion and decoder outputs generated during validation are:

- `/app/conversion_sample_out.txt`
- `/app/verification_sample_out.txt`
- `/app/train_decoder_sample_out.txt`
- `/app/conversion_full_out.txt`
- `/app/verification_full_out.txt`
- `/app/train_decoder_full_out.txt`
- `/app/processing_jm032_2023-10-22_a.png`
- `/app/processing_jm038_2023-04-30_a.png`

Detailed rationale, sanity checks, and review notes are in:

- `/app/CONVERSION_NOTES.md`
