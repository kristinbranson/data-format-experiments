# Neural Decoder: Hippocampal CA1 VR Navigation Task

## Dataset

Data from Sosa, Plitt & Giocomo (2025), "A flexible hippocampal population code for experience relative to reward."

2-photon calcium imaging of CA1 neurons in head-fixed mice performing a virtual reality (VR) navigation task with hidden reward zones.

## Task Description

Mice navigate a 450 cm virtual linear track with a hidden 50 cm reward zone at one of three possible locations:
- Zone A: 80-130 cm
- Zone B: 200-250 cm
- Zone C: 320-370 cm

Reward zone switches occur after 30 trials on designated days. Two visually distinct environments (ENV1, ENV2) are used across 14 days. Reward is randomly omitted on ~15% of trials.

## Data Format

See `CONVERSION_NOTES.md` for detailed documentation of the conversion process.

### Decoder Inputs (4 dimensions)
1. **time_from_trial_start** - seconds from trial onset (time-varying)
2. **environment_type** - 0=ENV1, 1=ENV2 (per-trial)
3. **trial_number** - trial index within session (per-trial)
4. **previous_trial_outcome** - 0=omitted, 1=rewarded (per-trial)

### Decoder Outputs (6 dimensions)
1. **distance_to_reward_zone** - signed distance discretized into 7 bins
2. **absolute_position** - track position in 5 equal bins (90 cm each)
3. **speed** - running speed in 5 bins
4. **lick** - binary licking indicator
5. **reward_zone_location** - 0=A, 1=B, 2=C (per-trial)
6. **reward_outcome** - 0=no, 1=yes (per-trial)

## Files

- `convert_data.py` - Conversion script
- `converted_data.pkl` - Full converted dataset (152 sessions, 12,216 trials)
- `sample_data.pkl` - Sample dataset (2 sessions per subject)
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` / `decoder.py` - Decoder training scripts

## Usage

```bash
# Convert data
python convert_data.py --data-dir data --output converted_data.pkl

# Convert sample
python convert_data.py --sample

# Verify format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu
```

## Statistics

- 11 subjects (switch task mice)
- 152 sessions (12-14 per subject)
- 12,216 total trials (mean 80.4 per session)
- 155-2,337 neurons per session (mean 911)
- ~15.5 Hz imaging rate (~64.5 ms time bins)
- ~15% reward omission rate
