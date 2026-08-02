# MAP Dataset Conversion: NWB to Decoder Format

Converts the Mesoscale Activity Map (MAP) dataset (DANDI:000363, Chen et al., 2024, Cell) from NWB electrophysiology format to a decoder-compatible Python dictionary/pickle format.

## Dataset

- **Source**: 174 NWB files, 28 subjects, brain-wide Neuropixels recordings during auditory delayed-response task
- **Output**: 144 sessions (after quality filtering), 57,935 neurons, 74,894 trials, 14 brain regions

## Quick Start

```bash
# Convert NWB data to decoder format
python convert_data.py

# Verify and train decoder
python train_decoder.py converted_data.pkl
```

## Output Format

`converted_data.pkl` contains a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `neural` | list[list[ndarray]] | `neural[session][trial]` = (n_neurons, 80) float32 firing rates |
| `input` | list[list[ndarray]] | `input[session][trial]` = (2, 80) float32: [time_from_tone_onset, photostim_on] |
| `output` | list[list[ndarray]] | `output[session][trial]` = (4,) int: [choice, outcome, early_lick, tongue_y] |
| `subjects` | list[str] | 28 subject IDs |
| `subject_idx` | ndarray | Session-to-subject mapping |
| `brain_regions` | list[str] | 14 brain region names |
| `brain_region_idx` | list[ndarray] | Per-session neuron-to-region mapping |
| `input_names` | list[str] | `['time_from_tone_onset', 'photostim_on']` |
| `output_names` | list[str] | `['choice', 'outcome', 'early_lick', 'tongue_y']` |
| `output_values` | list[dict] | Value labels per output dimension |
| `metadata` | dict | Dataset metadata and processing parameters |

### Output Variable Encoding

| Variable | Values | Description |
|----------|--------|-------------|
| choice | 0=left, 1=right | Lick direction |
| outcome | 0=ignore, 1=miss, 2=hit | Trial outcome |
| early_lick | 0=no, 1=yes | Whether mouse licked during delay |
| tongue_y | 0=low, 1=mid, 2=high | Tongue y-position (per-session 40th/60th percentile) |

## Processing Parameters

- **Time window**: [-2.5, +1.5]s relative to go cue
- **Bin size**: 50ms (80 time bins per trial)
- **Neuron QC**: classification == 'good' (QC classifier-based)
- **Session selection**: >65% correct rate, >=50 correct left and right trials
- **Trial filtering**: Excludes only auto_water and free_water trials

## Decoder Results

| Output | Train Bal. Acc | Val Bal. Acc | Chance |
|--------|---------------|-------------|--------|
| choice | 0.750 | 0.722 | 0.500 |
| outcome | 0.708 | 0.657 | 0.333 |
| early_lick | 0.802 | 0.760 | 0.500 |
| tongue_y | 0.697 | 0.681 | 0.333 |

## File Structure

```
.
├── convert_data.py          # Main conversion script
├── converted_data.pkl       # Converted dataset (~10 GB)
├── decoder.py               # Decoder model code
├── train_decoder.py         # Decoder training script
├── CONVERSION_NOTES.md      # Detailed conversion documentation
├── data/                    # Source NWB files (28 subjects)
├── code/                    # Reference code from paper
├── cache/                   # Intermediate processing cache
└── *.pdf                    # Reference papers
```

## References

- Chen, Liu et al. (2024). "Brain-wide neural activity underlying memory-guided movement." *Cell*.
- DANDI Archive: https://dandiarchive.org/dandiset/000363
