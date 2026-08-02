# Allen Visual Behavior Ophys Conversion

This repository contains a conversion of the Allen Brain Observatory Visual Behavior 2P dataset into a decoder-friendly pickle format.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`: full-run logs

## Conversion summary
- Data source: Allen Visual Behavior Ophys NWB experiment files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`
- Session unit: one ophys experiment per converted session
- Neural signal: dF/F traces
- Temporal alignment: ophys timestamps
- Trial inclusion: go and catch only
- Trial exclusion: aborted and auto-rewarded trials
- Passive sessions: excluded

## Outputs in converted data
1. `image_identity`
2. `image_change`
3. `running_speed_bin`
4. `pupil_diameter_bin`
5. `trial_outcome`

## Format
The pickle file stores a dictionary with keys:
- `neural`
- `input`
- `output`
- `subjects`
- `subject_idx`
- `brain_regions`
- `brain_region_idx`
- `input_names`
- `output_names`
- `output_values`
- `metadata`

## Re-run conversion
```bash
python -u convert_data.py converted_data.pkl --full
```

## Verify / train decoder
```bash
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl --plot-samples --cpu
```
