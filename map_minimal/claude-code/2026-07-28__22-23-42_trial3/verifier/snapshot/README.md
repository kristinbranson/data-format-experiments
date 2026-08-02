# Brain-Wide Neural Activity Dataset Conversion

Converts NWB data from the brain-wide neural activity dataset to a standardized decoder format.

## Source Data

- **Data paper**: "Brain-wide neural activity underlying memory-guided movement" (Li et al.)
- **Methods paper**: "Brain-wide analysis reveals movement encoding structured across and within brain areas"
- **Task**: Auditory delayed response task with 3kHz/12kHz tones, go cue, and left/right lick responses

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script |
| `converted_data.pkl` | Full converted dataset (144 sessions) |
| `sample_data.pkl` | Sample dataset (first 4 qualifying sessions) |
| `train_decoder.py` | Decoder training/validation script |
| `decoder.py` | Decoder model and utilities |
| `CONVERSION_NOTES.md` | Detailed conversion decisions and validation |

## Usage

```bash
# Convert full dataset
python convert_data.py --output converted_data.pkl

# Convert sample (first 5 NWB files)
python convert_data.py --sample --output sample_data.pkl

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu --plot-samples
```

## Data Structure

- **Neural**: Firing rates in 50ms bins, aligned to go cue, -2.5s to +1.5s (80 time bins)
- **Inputs**: Time from tone onset (s), photostimulation on/off (binary)
- **Outputs**: Choice (left/right), outcome (ignore/miss/hit), early lick (no/yes), tongue y-position (low/mid/high)
- **Quality control**: Classifier-based unit filtering (classification='good')
- **Session filtering**: >65% performance, >=50 correct trials per direction on control trials
- **Trial filtering**: Excluded auto_water and free_water trials

## Dataset Statistics

- 144 sessions from 28 subjects
- 57,925 total neurons across sessions
- 74,759 total trials
- 14 brain regions (ALM, Thalamus, Striatum, Midbrain, Medulla, OtherCortex, Orbital, Hippocampus, Cerebellum, Olfactory, Pallidum, Hypothalamus, CorticalSubplate, Pons)
