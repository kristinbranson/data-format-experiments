# Converted Switch-Task Dataset

This directory contains a decoder-ready conversion of the Sosa et al. hippocampal imaging dataset from the paper *A flexible hippocampal population code for experience relative to reward*.

## Dataset Summary
- Subjects: `11` mice
- Sessions: `152`
- Trials retained after conversion: `12,147`
- Curated neurons: `118,493`
- Brain region: `CA1`
- Common time bin: `64.483 ms` (`15.5078125 Hz`)
- Temporal alignment: trial start

The converted pickle is [`converted_data.pkl`](/app/converted_data.pkl). A smaller test subset is [`sample_data.pkl`](/app/sample_data.pkl).

## Files
- [`convert_data.py`](/app/convert_data.py): conversion script
- [`converted_data.pkl`](/app/converted_data.pkl): full converted dataset
- [`sample_data.pkl`](/app/sample_data.pkl): 2-session sample dataset
- [`CONVERSION_NOTES.md`](/app/CONVERSION_NOTES.md): detailed step-by-step audit trail

## Conversion Choices
- Neural signal: `processing/ophys/Deconvolved/plane0/data`
- Cell curation: ROI ids from `Deconvolved/plane0/rois`, then `iscell[:,0] > 0.5`
- Trial parsing: reconstructed from `trial_start` and `teleport`
- Multipane sessions (`m17`, `m18`): rebinned from `31.015625 Hz` to the common `15.5078125 Hz`
- Lick-artifact filtering: drop trials where more than `35%` of frames have `lick > 2`, matching the reference code path in `glmUtils.get_timeseries_data`

## Output Structure

```python
data = {
    "neural": [session][trial] -> float32 array (n_neurons, n_timepoints),
    "input": [session][trial] -> float32 array (4, n_timepoints),
    "output": [session][trial] -> int64 array (6, n_timepoints),
    "subjects": list[str],
    "subject_idx": int array (n_sessions,),
    "brain_regions": list[str],
    "brain_region_idx": [session] -> int array (n_neurons,),
    "input_names": list[str],
    "output_names": list[str],
    "output_values": list[list[str]],
    "metadata": dict,
}
```

Inputs:
1. `time_from_trial_start_s`
2. `environment`
3. `trial_number`
4. `previous_trial_rewarded`

Outputs:
1. `distance_to_reward_zone`
2. `absolute_position`
3. `speed`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

## Usage

Convert the full dataset:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Convert a 2-session sample and save processing plots:

```bash
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

Verify format only:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Train the reference decoder:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Validation Summary
- Full-format verification passed with no errors or warnings.
- Full decoder training completed successfully.
- Validation balanced accuracy:
  - `distance_to_reward_zone`: `0.4171`
  - `absolute_position`: `0.5236`
  - `speed`: `0.4662`
  - `lick`: `0.6286`
  - `reward_zone_location`: `0.8227`
  - `reward_outcome`: `0.5482`

For detailed sanity checks, consistency comparisons, and review iterations, see [`CONVERSION_NOTES.md`](/app/CONVERSION_NOTES.md).
