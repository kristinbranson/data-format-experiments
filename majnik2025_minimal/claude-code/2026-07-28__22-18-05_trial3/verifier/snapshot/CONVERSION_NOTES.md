# Conversion Notes: Track2p Barrel Cortex Data

## Source Data
- **Paper**: Majnik et al. 2025 - "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p" (eLife 14:RP107540)
- **Data**: 6 mice (jm031-jm046) with longitudinal 2-photon calcium imaging in barrel cortex (layer 2/3) during postnatal days P7-P14
- **Neural data**: Suite2p-processed fluorescence traces (F.npy, Fneu.npy) for neurons tracked across all sessions by Track2p
- **Behavioral data**: Motion energy from videography (motion_energy_glob.npy)

## Processing Pipeline

### 1. Neural Data (dF/F)
- **Neuropil correction**: Fc = F - 0.7 * Fneu (Suite2p default neucoeff=0.7)
- **Baseline correction**: Suite2p's `maximin` baseline method:
  - Gaussian smooth (sigma=10 frames)
  - Running minimum (window=1800 frames = 60s)
  - Running maximum (window=1800 frames)
  - Subtract baseline from Fc
- Used Suite2p's actual `preprocess` function for exact implementation match
- Paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)"

### 2. Motion Energy (Output)
- Raw motion energy: pixel-wise squared difference between consecutive video frames, summed across pixels
- Missing video frames (when ME length < neural frame count) handled by interpolation using interframe intervals to detect gap locations
- Binned by 10 frames (same as neural data)
- Discretized into 5 equal-percentile bins (quintiles) per session

### 3. Temporal Binning
- Both neural and behavioral data binned by averaging 10 consecutive frames
- Paper: "we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps"
- Imaging rate: 30 Hz -> effective bin rate: 3 Hz (333.33 ms bins)

### 4. Trial Structure
- Sessions split into 2-minute blocks (consecutive, non-overlapping)
- Paper: "splits were done on consecutive 2 minute blocks of the recording"
- Each trial = 360 time bins (120s * 3 Hz)
- jm031/jm032 (36000 frames, 20 min): 10 trials per session
- jm038-jm046 (54000 frames, 30 min): 15 trials per session

### 5. No Neuron Filtering
- All ROIs in the dataset are already filtered (iscell[:,0] == 1.0 for all)
- Track2p provides only successfully tracked neurons across all sessions
- Paper: "We considered all ROIs above the default threshold of 0.5 as true cells"

## Decoder Configuration
- **Input**: Time elapsed from session start (seconds), time-varying
- **Output**: Motion energy discretized into 5 equal-percentile bins (0-4), time-varying

## Data Statistics and Validation

### Neuron Counts (vs Paper: 526 +/- 190 std)
| Mouse | Tracked Neurons |
|-------|----------------|
| jm031 (Mouse A) | 221 |
| jm032 (Mouse B) | 370 |
| jm038 (Mouse C) | 685 |
| jm039 (Mouse D) | 746 |
| jm040 (Mouse E) | 541 |
| jm046 (Mouse F) | 435 |
| **Mean +/- std** | **500 +/- 180** |

Close match to paper's reported 526 +/- 190 std.

### Session Structure
- 6 subjects, 41 total sessions (7+7+7+7+6+7)
- Paper: "6 mice imaged daily for a minimum of 6 consecutive days"
- All mice have >= 6 sessions, matching paper

### Output Distribution
- Each percentile bin contains exactly 20% of data points (by construction)
- Consistent across all sessions

### Decoder Performance
- **Full dataset**: Validation balanced accuracy = 0.302 (chance = 0.200)
- **Sample dataset** (2 mice): Validation balanced accuracy = 0.307 (chance = 0.200)
- Performance above chance is expected: paper shows behavioral representation emerges around P11, with earlier sessions showing weak/no correlation between neural activity and motion

### Sanity Checks
1. All sessions pass data format validation with no errors or warnings
2. Neuron counts are constant within each mouse across sessions (Track2p tracking)
3. Motion energy percentile binning gives exactly 20% per bin per session
4. Session durations match expected (20 min for jm031/jm032, 30 min for others)
5. Frame counts match expected at 30 Hz (36000 or 54000 frames)
6. Missing video frames correctly identified via interframe interval analysis

## Files Produced
- `convert_data.py`: Conversion script
- `converted_data.pkl`: Full dataset (414 MB, 41 sessions, 545 trials)
- `sample_data.pkl`: Sample dataset (60 MB, 14 sessions, 140 trials, first 2 mice)
- `conversion_full_out.txt`: Full conversion output log
- `conversion_sample_out.txt`: Sample data verification output
- `verification_full_out.txt`: Full data verification output
- `verification_sample_out.txt`: Sample data verification output
- `train_decoder_full_out.txt`: Full data decoder training output
- `train_decoder_sample_out.txt`: Sample data decoder training output
