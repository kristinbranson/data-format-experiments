# MAP Dataset - Neural Decoder Format

## Dataset Description
Brain-wide Neuropixels recordings from 28 mice performing an auditory delayed response task.
Source: DANDI:000363 (Chen et al. 2023)

### Task
Mice hear pure tones (3kHz or 12kHz), wait through a 1.2s delay, then lick left or right after Go cue.

### Data Summary
- **173 sessions**, **28 subjects**, **89,532 trials**, **69,453 neurons**, **293 brain regions**
- Aligned to **Go cue onset**, window: **-2.5s to +1.5s**, **50ms bins** (80 bins)

## Loading Data
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Neural: data['neural'][session][trial] -> (n_neurons, 80)
# Inputs: data['input'][session][trial] -> (2, 80)
# Outputs: data['output'][session][trial] -> (4, 80)
```

## Inputs
1. **time_from_tone_onset**: seconds from first tone
2. **photostim_on**: binary photostimulation indicator

## Outputs
1. **choice**: left(0), right(1), no_lick(2)
2. **outcome**: ignore(0), miss(1), hit(2)
3. **early_lick**: no(0), yes(1)
4. **tongue_y_position**: below_40th(0), 40-60th(1), above_60th(2), not_visible(3)

## Decoder Performance
| Output | Val Balanced Acc | Chance |
|--------|-----------------|--------|
| choice | 0.677 | 0.333 |
| outcome | 0.662 | 0.333 |
| early_lick | 0.752 | 0.500 |
| tongue_y | 0.656 | 0.250 |
