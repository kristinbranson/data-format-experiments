# Track2p Motion Decoder Conversion

This repository now includes a converted dataset in [`converted_data.pkl`](/app/converted_data.pkl) for decoder training with [`train_decoder.py`](/app/train_decoder.py).

## Dataset summary

- Source dataset: longitudinal 2-photon calcium imaging from mouse barrel cortex development (Majnik et al., 2025)
- Subjects: 6 mice (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`)
- Sessions: 41 total
- Neural signal used: Suite2p baseline-corrected fluorescence reconstructed from `F.npy`, `Fneu.npy`, and `ops.npy`
- Behavioral signal used: global motion energy from `move_deve/motion_energy_glob.npy`
- Time binning: 10 imaging frames at 30 Hz (`333.33 ms`)
- Trial definition: consecutive non-overlapping 2-minute blocks
- Output target: motion energy discretized into 5 global equal-percentile bins

## Converted format

The pickle stores a dictionary with the fields required by the decoder:

- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: one time-varying input per trial, `elapsed_time_sec`, shaped `(1, n_timepoints)`
- `output`: one time-varying categorical output per trial, `motion_energy_bin`, shaped `(1, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

All trials have `360` time bins. Sessions with 20-minute recordings yield `10` trials; sessions with 30-minute recordings yield `15` trials.

## How to use

Verify the converted dataset:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Train the reference decoder:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

Rebuild the dataset:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Sample-mode rebuild:

```bash
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

## Key outputs

- [`converted_data.pkl`](/app/converted_data.pkl): full converted dataset
- [`sample_data.pkl`](/app/sample_data.pkl): 2-session sample dataset
- [`CONVERSION_NOTES.md`](/app/CONVERSION_NOTES.md): detailed audit trail
- [`conversion_full_out.txt`](/app/conversion_full_out.txt): full conversion log
- [`verification_full_out.txt`](/app/verification_full_out.txt): full format-verification log
- [`train_decoder_full_out.txt`](/app/train_decoder_full_out.txt): full decoder training log

## Notes

- The released dataset includes both 20-minute and 30-minute sessions, even though the paper methods text states 20-minute sessions.
- The paper reports continuous-motion decoding with `R^2`; this conversion uses 5-way categorical motion bins because that is the required target format for the provided decoder.
