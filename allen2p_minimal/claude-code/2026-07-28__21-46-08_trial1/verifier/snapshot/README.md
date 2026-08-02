# Allen Brain Observatory Visual Behavior 2P - Neural Decoder Dataset

## Overview

This dataset converts the Allen Brain Observatory Visual Behavior 2-photon calcium imaging data into a format suitable for training neural decoders. The data comes from transgenic mice performing a visual change detection task while being imaged with 2-photon microscopy.

## Source Data

- **Dataset**: Allen Brain Observatory Visual Behavior 2P (version 1.1.0)
- **References**:
  - "Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper" (whitepaper.pdf)
  - "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex" (paper.pdf)
- **SDK**: AllenSDK (code directory)

## Data Selection

- **Sessions**: Active behavior sessions with familiar image sets
- **Equipment**: Both single-plane (Scientifica, ~31 Hz) and multi-plane (Multiscope, ~11 Hz) recordings
- **Cell types**: Excitatory (Slc17a7-IRES2-Cre), Sst inhibitory (Sst-IRES-Cre), Vip inhibitory (Vip-IRES-Cre)
- **Brain regions**: VISp (primary visual cortex), VISl (lateral visual area)
- **Trials**: Go and Catch trials only (excluding Aborted and Auto-rewarded)

## Processing

- **Neural signal**: Detected calcium events (deconvolved from dF/F traces)
- **Temporal alignment**: All signals aligned to ophys timestamps
- **Time bin**: 93.2 ms (~10.7 Hz, matching Multiscope frame rate)
- **Single-plane downsampling**: ~31 Hz data downsampled by 3x averaging to match common bin size

## Output Variables

1. **Image identity** (categorical, 8 classes): Which natural image is currently displayed
2. **Image change** (binary): Whether the image changed at this timepoint
3. **Running speed** (5 percentile bins): Mouse running speed discretized into equal-frequency bins
4. **Pupil diameter** (5 percentile bins): Pupil width discretized into equal-frequency bins
5. **Trial outcome** (4 classes): hit, miss, false_alarm, correct_reject

## Dataset Statistics

- 110 sessions (imaging planes)
- 38 subjects (mice)
- 27,643 trials
- ~14,895 total neurons across all sessions
- Mean trial length: ~8.2 seconds (88 timepoints)

## Files

- `convert_data.py` - Conversion script
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Small sample for quick testing (3 sessions)
- `train_decoder.py` - Decoder training and validation script
- `decoder.py` - Decoder implementation
- `CONVERSION_NOTES.md` - Detailed conversion decisions and validation

## Usage

```bash
# Convert data (full)
python convert_data.py --output converted_data.pkl

# Convert sample
python convert_data.py --sample --output sample_data.pkl

# Verify format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu --plot-samples
```
