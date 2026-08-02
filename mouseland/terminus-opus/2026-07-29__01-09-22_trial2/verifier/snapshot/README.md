# Zhong et al. 2025 - Neural Decoder Dataset

## Dataset Description
This dataset contains calcium imaging data from the paper "Unsupervised pretraining in biological neural networks" (Zhong et al., 2025). The data has been converted to a decoder-compatible format for predicting behavioral and experimental variables from neural activity.

### Experiment
Head-fixed mice ran through linear virtual reality corridors with naturalistic texture patterns (leaf, circle, rock, brick). Mice discriminated between visual patterns in rewarded vs unrewarded corridors. A sound cue indicated the reward zone in rewarded corridors.

### Recording
- Two-photon mesoscope calcium imaging
- Suite2p processed: deconvolved fluorescence traces
- Brain regions: V1, medial higher visual (mHV), lateral higher visual (lHV), anterior higher visual (aHV)
- Frame rate: 3.17 Hz

### Dataset Statistics
- 89 sessions, 19 mice
- 38,110 total trials
- 17,363-78,815 neurons per session (after filtering)
- Neurons outside visual cortex excluded (iarea==-1 or 7)

## How to Load
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Data Format
- `data['neural']`: List of sessions, each containing list of trials with shape (n_neurons, n_timepoints)
- `data['input']`: Decoder inputs (4, n_timepoints) per trial
  - time_to_sound_cue: Time from each frame to sound cue (seconds)
  - day_of_training: Days since first recording for each mouse
  - time_since_trial_start: Elapsed time from corridor entry (seconds)
  - reward_availability: 1 if rewarded corridor, 0 otherwise
- `data['output']`: Decoder outputs (4, n_timepoints) per trial
  - visual_stimulus_category: Stimulus identity (categorical)
  - licking: Binary lick detection per frame
  - position_bin: Position in corridor (4 bins of 1.5m)
  - running_speed_bin: Running speed quartile (4 bins)
- `data['subjects']`: List of mouse names
- `data['subject_idx']`: Subject index per session
- `data['brain_regions']`: ['V1', 'mHV', 'lHV', 'aHV']
- `data['brain_region_idx']`: Brain region index per neuron per session

## Key Processing Steps
1. Neural data loaded from Suite2p deconvolved traces
2. Neurons filtered: exclude iarea==-1 and iarea==7 (outside visual cortex)
3. Frames filtered: only VR-moving frames (ft_move > 0)
4. Trials aligned to corridor entry
5. Position binned into 4 equal bins over 6m corridor
6. Running speed binned into global quartiles

## Files
- `converted_data.pkl`: Full converted dataset
- `sample_data.pkl`: 2-session sample for testing
- `convert_data.py`: Conversion script
- `CONVERSION_NOTES.md`: Detailed conversion documentation
- `train_decoder.py`: Decoder training script
