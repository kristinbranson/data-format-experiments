# Converted IBL Brain-Wide Map Decoder Dataset

This repository contains a converted dataset for decoder training from the IBL brain-wide map local ONE cache.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes
- `train_decoder_full_out.txt`: full decoder training output
- `verification_full_out.txt`, `verification_full_out_v2.txt`: format verification outputs

## Data format
The pickle contains a dictionary with:
- `neural`: sessions -> trials -> `(n_neurons, n_timepoints)` spike-count matrices
- `input`: sessions -> trials -> `(2, n_timepoints)` arrays for time since stimulus onset and trial number in block
- `output`: sessions -> trials -> `(4, n_timepoints)` categorical outputs for choice, prior-left probability, wheel-speed bin, and whisker-motion-energy bin
- subject and brain-region metadata
- descriptive metadata including 20 ms bin size and stimulus-onset alignment

## Conversion choices
- 20 ms bins
- alignment to stimulus onset
- probes merged within session
- strict cluster QC filter using `label >= 1`
- wheel speed and whisker motion energy discretized into 3 bins

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl --plot-samples
```
