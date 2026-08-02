# Neural Decoder Data Conversion

## Source Data
Paper: "Unsupervised pretraining in biological neural networks" (Zhong et al. 2025)

**Data**: Two-photon calcium imaging recordings from 19 mice running through virtual reality corridors with visual texture patterns.

## Task Description
Mice run through 4m virtual corridors with naturalistic texture patterns (leaf, circle, rock, brick). A sound cue is presented at a random position, and in rewarded trials, water is available. Mice learn to discriminate visual patterns through licking behavior.

## Data Processing Pipeline

### 1. Neural Data
- **Source**: Deconvolved calcium fluorescence traces from Suite2p
- **Frame rate**: 3.17 Hz (~315 ms per frame)
- **Neurons**: All visual cortex neurons (V1, mHV, lHV, aHV) kept; neurons outside visual cortex (iarea -1, 7) excluded
- **Brain regions**: Assigned via retinotopy (V1=iarea 8, mHV=0/1/2/9, lHV=5/6, aHV=3/4)

### 2. Trial Alignment
- **Alignment event**: Trial start (corridor entry)
- **Window**: 32 frames from corridor entry (~10.1 seconds)
- **Rationale**: At 60 cm/s constant VR speed, 6m corridor takes ~10s = ~32 frames

### 3. Decoder Inputs (4 variables)
1. **time_to_sound_cue**: Time relative to sound cue onset (seconds), negative before cue
2. **day_of_training**: Days since first recording session for this mouse
3. **time_since_trial_start**: Time from corridor entry (seconds)
4. **reward_availability**: 1 if rewarded corridor, 0 if not

### 4. Decoder Outputs (4 variables)
1. **visual_stimulus**: Stimulus category (e.g., circle1, leaf1, etc.) - per trial
2. **licking**: Binary (0=no lick, 1=lick) - time varying
3. **position**: Position in corridor discretized into 4 x 1m bins - time varying
4. **running_speed**: Running speed discretized into 4 quartile bins - time varying

## Files
- `convert_data.py` - Conversion script
- `converted_data.pkl` - Full converted dataset (all 89 sessions)
- `sample_data.pkl` - Sample dataset (3 sessions)
- `CONVERSION_NOTES.md` - Detailed conversion decisions and validation
- `conversion_full_out.txt` - Full conversion output log
- `conversion_sample_out.txt` - Sample conversion output log
- `verification_full_out.txt` - Full data verification output
- `verification_sample_out.txt` - Sample data verification output
- `train_decoder_full_out.txt` - Full decoder training output
- `train_decoder_sample_out.txt` - Sample decoder training output

## Usage
```bash
# Convert sample (3 sessions)
python convert_data.py --sample --output sample_data.pkl

# Convert all sessions
python convert_data.py --output converted_data.pkl

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu
```

## Key Statistics
- **19 mice**, **89 recording sessions**
- **~20,000-90,000 neurons per session** (visual cortex only)
- **~80-700 trials per session**
- **4 brain regions**: V1, mHV, lHV, aHV
- **15 stimulus categories** across all sessions
