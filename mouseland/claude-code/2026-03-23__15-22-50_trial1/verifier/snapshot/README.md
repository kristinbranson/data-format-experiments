# Zhong et al. 2025 - Decoder-Compatible Data Conversion

Converts calcium imaging data from "Unsupervised pretraining in biological neural networks" (Zhong et al., Nature 2025) into a decoder-compatible Python dictionary format.

## Files

### Scripts
- `convert_data.py` - Main conversion script. Loads neural, behavioral, and retinotopy data; outputs `converted_data.pkl`
- `train_decoder.py` - Decoder training/validation script (uses `decoder.py`)
- `decoder.py` - Decoder module (provided, not modified)

### Data
- `converted_data.pkl` - Full converted dataset (148 GB, 89 sessions, 38,109 trials, 19 subjects)
- `sample_data.pkl` - Sample dataset (2 sessions from DR10, for quick testing)
- `data/` - Raw data directory (spk/, beh/, retinotopy/)
- `code/` - Reference code from the paper
- `cache/` - Cache directory

### Documentation
- `CONVERSION_NOTES.md` - Detailed step-by-step conversion notes with all decisions and checks
- `methods.txt` - Extracted methods from the paper

### Output/Logs
- `train_decoder_full_out.txt` - Full decoder training output
- `verification_full_out.txt` - Full data verification output
- `sample_trials.png` - Sample trial visualizations
- `predictions.png` - Decoder prediction visualizations

## Usage

```bash
# Sample conversion (2 sessions)
python3 convert_data.py sample_data.pkl --sample

# Full conversion (all 89 sessions, ~33 min)
python3 convert_data.py converted_data.pkl --full

# Train decoder
python3 train_decoder.py converted_data.pkl --plot-samples --cpu
```

## Output Format

The converted data is a Python dict with keys:
- `neural`: list of 89 sessions, each a list of trials, each `(n_neurons, T)` float16 array
- `input`: list of sessions/trials, each `(4, T)` float32 array
  - `[0]` time_to_sound_cue (seconds, positive before cue)
  - `[1]` day_of_training (days since mouse's first recording)
  - `[2]` time_since_trial_start (seconds)
  - `[3]` reward_availability (0 or 1)
- `output`: list of sessions/trials, each `(4, T)` int array
  - `[0]` visual_stimulus (15 categories)
  - `[1]` licking (binary)
  - `[2]` position (4 bins of 1m each, 0-4m corridor)
  - `[3]` running_speed (4 quartile bins)
- `brain_region`: list of region index arrays per session (0=V1, 1=mHV, 2=lHV, 3=aHV, 4=other)
- `subject_id`, `session_id`, `input_names`, `output_names`, `output_values`

## Decoder Results

| Output | Val. Balanced Acc | Chance |
|--------|-------------------|--------|
| visual_stimulus | 0.706 | 0.067 |
| licking | 0.840 | 0.500 |
| position | 0.734 | 0.250 |
| running_speed | 0.689 | 0.250 |

All outputs well above chance, confirming correct data conversion.
