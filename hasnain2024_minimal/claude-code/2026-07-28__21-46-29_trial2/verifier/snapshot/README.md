# Neural Decoder Data Conversion

Converts neuroscience data from Hasnain, Birnbaum et al. (Nature Neuroscience 2024) into a standardized Python dictionary format for training neural decoders.

**Paper**: "Separating cognitive and motor processes in the behaving mouse"
**Data source**: Zenodo (DOI: 10.5281/zenodo.13941415)

## Data Overview

- **Species**: Mouse (mus musculus)
- **Brain region**: Anterior lateral motor cortex (ALM)
- **Tasks**: Delayed-response (DR) and water-cued (WC) licking
- **Datasets**: 44 sessions from 14 mice
  - 25 sessions from Ephys_Behavior (10 mice)
  - 19 sessions from RandomizedDelay_Ephys_Behavior (4 mice)
- **Total trials**: 11,985
- **Total neurons**: 2,457

## Quick Start

```bash
# Convert all sessions
python convert_data.py --output converted_data.pkl

# Convert sample (3 sessions)
python convert_data.py --output sample_data.pkl --sample 3

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu
```

## Output Format

The output pickle contains a dictionary with:

| Key | Description |
|-----|-------------|
| `neural` | List of 44 session lists, each containing per-trial arrays of shape `(n_neurons, 500)` |
| `input` | List of 44 session lists, each containing per-trial arrays of shape `(1, 500)` (time from go cue) |
| `output` | List of 44 session lists, each containing per-trial arrays of shape `(6, 500)` |
| `subjects` | List of 14 subject names |
| `subject_idx` | Array mapping sessions to subjects |
| `brain_regions` | `['ALM']` |
| `brain_region_idx` | Per-session neuron-to-region mapping |
| `input_names` | `['time_from_gocue']` |
| `output_names` | 6 output variable names |
| `output_values` | Class labels per output |

### Decoder Outputs (6 dimensions)

| Index | Name | Values |
|-------|------|--------|
| 0 | lick_direction | left (0) / right (1) |
| 1 | behavioral_context | WC (0) / DR (1) |
| 2 | outcome | incorrect (0) / correct (1) |
| 3 | tongue_velocity | low (0) / high (1) |
| 4 | paw_velocity | low (0) / high (1) |
| 5 | motion_energy | low (0) / high (1) |

## Processing Pipeline

Neural data processing matches the paper's MATLAB pipeline:

1. **Trial filter**: (hit | miss) & ~stim.enable & ~early
2. **Spike alignment**: Align to Go cue onset
3. **Binning**: 10ms bins, -2.5s to +2.5s (500 time bins)
4. **Smoothing**: Causal Gaussian kernel (N=15, reflect boundary)
5. **Quality filter**: Exclude garbage, gabrga, noisy, real?
6. **FR filter**: Remove neurons with mean firing rate <= 1 Hz
7. **Velocity discretization**: 50th percentile threshold per session (non-zero values for tongue)

## Decoder Results

Validation balanced accuracy (all above 0.50 chance):

| Output | Sample (3 sess) | Full (44 sess) |
|--------|-----------------|----------------|
| lick_direction | 0.689 | 0.682 |
| behavioral_context | 0.800 | 0.884 |
| outcome | 0.569 | 0.681 |
| tongue_velocity | 0.802 | 0.788 |
| paw_velocity | 0.586 | 0.572 |
| motion_energy | 0.794 | 0.814 |

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script |
| `converted_data.pkl` | Full dataset (44 sessions) |
| `sample_data.pkl` | Sample dataset (3 sessions) |
| `CONVERSION_NOTES.md` | Detailed conversion notes |
| `conversion_full_out.txt` | Full conversion output log |
| `conversion_sample_out.txt` | Sample conversion output log |
| `verification_full_out.txt` | Full data verification log |
| `verification_sample_out.txt` | Sample data verification log |
| `train_decoder_full_out.txt` | Full decoder training log |
| `train_decoder_sample_out.txt` | Sample decoder training log |
