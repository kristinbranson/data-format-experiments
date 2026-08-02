# MAP Dataset Neural Decoder

Converts the MAP dataset ("Brain-wide neural activity underlying memory-guided movement") from NWB format into a standardized Python dictionary for training neural decoders.

## Dataset Summary

- **144 sessions** from **28 mice** performing an auditory delayed response task
- **57,935 good units** across **14 brain regions** (Neuropixels recordings)
- **74,769 trials** with 80 time bins each (50ms bins, -2.5s to +1.5s relative to Go cue)

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Conversion script (NWB -> standardized dict) |
| `converted_data.pkl` | Full converted dataset (~10 GB) |
| `sample_data.pkl` | First 5 sessions (~269 MB) |
| `train_decoder.py` | Decoder training and validation script |
| `decoder.py` | Decoder model and data format verification |
| `CONVERSION_NOTES.md` | Detailed conversion decisions and validation |
| `conversion_full_out.txt` | Full conversion log |
| `conversion_sample_out.txt` | Sample conversion log |
| `verification_full_out.txt` | Full data format verification output |
| `verification_sample_out.txt` | Sample data format verification output |
| `train_decoder_full_out.txt` | Full decoder training results |
| `train_decoder_sample_out.txt` | Sample decoder training results |

## Usage

### Convert data
```bash
# Full dataset (all 174 NWB files, outputs 144 valid sessions)
python3 convert_data.py --data-dir /path/to/nwb/files --output converted_data.pkl --sample-output sample_data.pkl

# Sample only (first 5 sessions)
python3 convert_data.py --data-dir /path/to/nwb/files --max-sessions 5 --output sample_data.pkl --sample-output sample_data.pkl
```

### Train decoder
```bash
python3 train_decoder.py converted_data.pkl
```

## Data Format

The output is a pickled Python dictionary with keys:

- `neural_data`: list of arrays, shape `(n_trials, T, n_neurons)` — firing rates in spikes/s
- `decoder_input`: list of arrays, shape `(n_trials, T, 2)` — time_from_tone_onset, photostimulation
- `decoder_output`: list of arrays, shape `(n_trials, 4)` — choice, outcome, early_lick, tongue_y_position
- `neuron_regions`: list of arrays — brain region label per neuron
- `session_info`: list of dicts with subject ID and session metadata

## Decoder Results (Full Dataset)

| Output | Validation Accuracy | Chance |
|--------|-------------------|--------|
| choice | 0.722 | 0.500 |
| outcome | 0.660 | 0.333 |
| early_lick | 0.748 | 0.500 |
| tongue_y_position | 0.537 | 0.333 |
