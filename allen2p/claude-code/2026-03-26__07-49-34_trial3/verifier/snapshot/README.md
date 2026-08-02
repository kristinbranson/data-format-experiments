# Allen Brain Observatory Visual Behavior 2P - Neural Decoder Dataset

## Dataset Description

This dataset contains converted neural and behavioral data from the Allen Institute Visual Behavior 2-Photon Calcium Imaging dataset, formatted for neural decoding.

**Task**: Mice performed a visual change detection task. They viewed flashed natural images (250ms stimulus + 500ms gray = 750ms interval) and were rewarded for licking when the image identity changed. The dataset includes neural calcium events, running speed, pupil tracking, and trial outcomes.

**Source**: Allen Brain Observatory Visual Behavior 2P dataset v1.1.0
- Whitepaper: Garrett et al., "Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper"
- Paper: Piet et al. (2024), "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex", Neuron 112, 1876-1890

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Sessions (experiments) | 202 |
| Subjects (mice) | 38 |
| Total trials | 51,992 |
| Total neurons | 29,444 |
| Brain regions | VISp (V1), VISl (LM) |
| Time bin | 33.33 ms (30 Hz) |
| Mean trial length | ~255 timepoints (~8.5s) |
| Cell types | Excitatory (Slc17a7), SST, VIP |
| Neural signal | Calcium events (FastLZeroSpikeInference) |

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access outputs
output = data['output'][0][0]  # shape: (5, n_timepoints)
# output[0] = image identity (0-15)
# output[1] = image change (0 or 1)
# output[2] = running speed bin (0-4)
# output[3] = pupil diameter bin (0-4)
# output[4] = trial outcome (0=hit, 1=miss, 2=FA, 3=CR)

# Subject info
print(data['subjects'])          # list of mouse IDs
print(data['subject_idx'])       # session -> subject mapping

# Brain region info
print(data['brain_regions'])     # ['VISl', 'VISp']
print(data['brain_region_idx'])  # list of arrays, neuron -> region per session
```

## Output Format

### Decoder Outputs (5 variables)

| Index | Name | Type | Values | Description |
|-------|------|------|--------|-------------|
| 0 | image_identity | Categorical, time-varying | 16 natural image names | Which image is being displayed |
| 1 | image_change | Binary, time-varying | no_change / change | 1 during first 750ms of new image |
| 2 | running_speed | 5 bins, time-varying | bin_0 to bin_4 | Equal percentile bins |
| 3 | pupil_diameter | 5 bins, time-varying | bin_0 to bin_4 | Equal percentile bins |
| 4 | trial_outcome | Categorical, static/trial | hit/miss/false_alarm/correct_reject | Mouse behavioral response |

### Trial Selection
- Included: Go trials (image change) and Catch trials (sham change)
- Excluded: Aborted trials (premature licking) and Auto-rewarded trials (free rewards)
- Session types: Active behavior only (OPHYS_1,3,4,6); passive sessions excluded

### Neural Data
- Calcium events from FastLZeroSpikeInference (pre-computed in NWB files)
- Filtered by valid_roi flag (SVM classifier for cell quality)
- Resampled from native rate (~31 Hz Scientifica, ~11 Hz Multiscope) to 30 Hz

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (8.3 GB) |
| `sample_data.pkl` | 2-session sample for testing |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training/validation script |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |

## Running the Decoder

```bash
# Verify data format
python3 train_decoder.py converted_data.pkl --verify-only

# Train and evaluate decoder
python3 train_decoder.py converted_data.pkl --plot-samples

# Re-run conversion
python3 -u convert_data.py converted_data.pkl --full
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```
