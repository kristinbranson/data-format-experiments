# CA1 Geometric Deformation - Neural Decoder Dataset

## Overview
Converted dataset from Lee, Keinath, Cianfarano & Brandon (2025) for training a neural decoder to predict mouse position from CA1 hippocampal activity during free exploration of geometrically deformed environments.

## Source
- **Paper**: "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping." Neuron 113(2): 307-320.
- **Data**: 7 mice, 207 sessions, 5,413 unique neurons, 10 distinct environment geometries

## Files
- `convert_data.py` - Conversion script
- `converted_data.pkl` - Full converted dataset (207 sessions, 8,187 trials)
- `sample_data.pkl` - Sample dataset (8 sessions, 312 trials)
- `CONVERSION_NOTES.md` - Detailed conversion decisions and validation
- `conversion_full_out.txt` - Full conversion log
- `conversion_sample_out.txt` - Sample conversion log
- `verification_full_out.txt` - Full data verification log
- `verification_sample_out.txt` - Sample data verification log
- `train_decoder_full_out.txt` - Full decoder training log
- `train_decoder_sample_out.txt` - Sample decoder training log

## Data Format
- **Neural**: Binary calcium event counts summed into 1-second time bins (n_neurons, 60)
- **Input**: Environment geometry as flattened 3x3 binary matrix (9,) - static per trial
- **Output**: Mouse position discretized into 3x3 spatial bins (1, 60) - time-varying
- **Trials**: 1-minute segments from 40-minute recording sessions
- **Time bins**: 1 second (1000 ms)

## Usage
```bash
# Create sample dataset
python convert_data.py --sample

# Create full dataset
python convert_data.py

# Verify format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu
```

## Results
- Validation balanced accuracy: 63.4% (chance: 11.1%)
