# Neural Decoder Conversion

This repository now includes a Python conversion pipeline for the alternating-context ALM ephys cohort from Hasnain, Birnbaum et al. The main entry point is `convert_data.py`, which reads the provided MATLAB session files, reproduces the reference alignment and filtering logic, and writes decoder-ready pickle files.

Generated files:

- `converted_data.pkl`: full 12-session converted dataset
- `sample_data.pkl`: 4-session subset for quick validation
- `CONVERSION_NOTES.md`: processing details and validation results

Key conversion choices:

- Session cohort: the 12 fixed-delay alternating-context ALM sessions used in the Figure 8 context analyses in the reference code
- Alignment: `goCue`
- Neural binning: `tmin=-3.0 s`, `tmax=2.5 s`, `dt=10 ms`
- Neural smoothing: causal Gaussian window `N=15`, MATLAB-style `reflect` boundary handling
- Unit filtering: all non-garbage/non-noisy clusters, then remove units with mean firing rate `<= 1 Hz`
- Trial filtering for the decoder dataset: exclude `early`, `no`, and `stim.enable` trials
- Decoder input: time from go cue
- Decoder outputs: lick direction, context, outcome, tongue velocity bin, paw velocity bin, motion energy bin

Commands:

```bash
python convert_data.py --dataset both --sample-sessions 4
python train_decoder.py sample_data.pkl --verify-only --cpu
python train_decoder.py sample_data.pkl --cpu
python train_decoder.py converted_data.pkl --verify-only --cpu
python train_decoder.py converted_data.pkl --cpu
```

Observed full-dataset summary:

- 12 sessions
- 7 subject IDs
- 2415 usable trials
- 521 neurons after filtering
- Full decoder validation balanced accuracy:
  - `lick_direction`: `0.6336`
  - `behavioral_context`: `0.7669`
  - `outcome`: `0.6245`
  - `tongue_velocity`: `0.8190`
  - `paw_velocity`: `0.6189`
  - `motion_energy`: `0.7966`
