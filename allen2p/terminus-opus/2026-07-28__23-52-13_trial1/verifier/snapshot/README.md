# Allen Brain Observatory - Visual Behavior 2P: Decoder Dataset

## Dataset Description

This dataset contains neural and behavioral data from the Allen Brain Observatory Visual Behavior 2-Photon (2P) calcium imaging project. Mice performed a visual change detection task where they reported changes in natural image identity by licking to earn water rewards.

### Experimental Task
- **Change Detection**: Mice view a continuous series of flashed natural images (250ms on, 500ms off)
- **Go trials**: Image identity changes; mouse should lick (hit) or not (miss)
- **Catch trials**: No image change; mouse should not lick (correct reject) or lick (false alarm)
- **Images**: 8 natural images per set, 2 image sets (A and B), 16 unique images total
- **Excluded**: Aborted trials (lick before change) and auto-rewarded trials

### Neural Data
- **Calcium events**: Deconvolved from dF/F traces using FastLZeroSpikeInference
- **Brain region**: VISp (primary visual cortex)
- **Cell types**: Excitatory (Slc17a7-IRES2-Cre), Sst inhibitory (Sst-IRES-Cre), Vip inhibitory (Vip-IRES-Cre)
- **Frame rate**: 31 Hz (Scientifica single-plane imaging systems only)

### Data Filtering
- Only active behavior sessions included (passive viewing excluded)
- Only Scientifica (31 Hz) sessions included for consistent time bins (Multiscope 11 Hz excluded)
- Only Go and Catch trials included (Aborted and Auto-rewarded excluded)
- ROI filtering already applied by Allen pipeline

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data
neural_session0_trial0 = data['neural'][0][0]  # (n_neurons, n_timepoints)

# Access outputs
output_session0_trial0 = data['output'][0][0]  # (5, n_timepoints)

# Output dimensions:
# 0: image_identity (0-15, index into output_values[0])
# 1: image_change (0=no change, 1=change)
# 2: running_speed (0-4, percentile bins)
# 3: pupil_diameter (0-4, percentile bins)
# 4: trial_outcome (0=hit, 1=miss, 2=false_alarm, 3=correct_reject)
```

## Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 168 |
| Subjects (mice) | 37 |
| Brain regions | VISp |
| Total trials | 43,387 |
| Total neurons | 29,097 |
| Mean neurons/session | 173.2 |
| Mean trials/session | 258.3 |
| Time bin size | ~32.3 ms (31 Hz) |
| Output dimensions | 5 |

## Output Variables
| Name | Type | Values | Description |
|------|------|--------|-------------|
| image_identity | Time-varying | 16 images | Which natural image is currently shown |
| image_change | Time-varying | Binary (0/1) | 1 during change image presentation |
| running_speed | Time-varying | 5 bins (0-4) | Percentile-binned running speed |
| pupil_diameter | Time-varying | 5 bins (0-4) | Percentile-binned pupil width |
| trial_outcome | Static/trial | 4 classes | hit/miss/false_alarm/correct_reject |

## Files
- `converted_data.pkl` - Full converted dataset (168 sessions)
- `sample_data.pkl` - 2-session sample for testing
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model code

## References
- Paper: "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex" (Neuron 2024)
- Whitepaper: "Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper"
- AllenSDK: https://allensdk.readthedocs.io/
