# IBL Decoder Conversion

This directory contains a decoder-ready conversion of the International Brain Laboratory brain-wide map electrophysiology release, reformatted for `train_decoder.py`.

## Dataset Summary
- Source release: frozen 459-session / 699-insertion brain-wide map release used by the bundled reference code
- Converted subset: 438 sessions, 135 subjects, 186,261 valid trials, 72,757 well-isolated neurons
- Alignment: stimulus onset (`stimOn_times`)
- Common trial window: `[-0.5, 1.5]` s
- Bin size: `20 ms`
- Trial shape:
  - neural: `(n_neurons, 100)`
  - input: `(2, 100)`
  - output: `(4, 100)`

The converted subset is smaller than the full public release because this decoder task requires whisker motion energy as a mandatory output, and 20 release sessions in the local cache do not contain the required whisker stream. One additional release session had no usable well-isolated units or fewer than 2 valid trials after masking.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample conversion
- `convert_data.py`: conversion entrypoint
- `CONVERSION_NOTES.md`: full step-by-step audit trail
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`: sample run logs
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`: full run logs

## Conversion Command
Run the converter with:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Useful modes:

```bash
python3 -u convert_data.py sample_data.pkl --sample
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

## Validation Command
Verify structure only:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Train the provided decoder:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples --cpu
```

`--cpu` was required in this environment because the available GPU ran out of memory.

## Output Format
The pickle stores a Python dictionary with:
- `neural`: list of sessions, each a list of trial matrices `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trial matrices `(2, 100)`
- `output`: list of sessions, each a list of trial matrices `(4, 100)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

Inputs:
1. `time_since_stimulus_onset_s`
2. `trial_number_in_block`

Outputs:
1. `choice` (`left=0`, `right=1`)
2. `prior_probability_of_left` (`0.2->0`, `0.5->1`, `0.8->2`)
3. `wheel_speed_bin` (`0/1/2`, global tertiles)
4. `whisker_motion_energy_bin` (`0/1/2`, global tertiles)

## Processing Notes
- Neurons are filtered to `clusters.metrics.label >= 1`, reproducing the paper’s 75,708 well-isolated-unit count before the mandatory whisker-availability subset is applied.
- Trials use the reference mask:
  - required non-null `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
  - reaction time `0.08` to `2.0` s
  - no-choice trials removed
  - maximum trial length `10.0` s
- Wheel speed is absolute wheel velocity.
- Whisker motion energy uses the left camera when available, otherwise the right camera.
- Wheel and whisker outputs are discretized with global tertile thresholds saved in `metadata`.

## Full Decoder Result
Validation balanced accuracy on the full dataset:
- `choice`: `0.6222`
- `prior_probability_of_left`: `0.6689`
- `wheel_speed_bin`: `0.6445`
- `whisker_motion_energy_bin`: `0.7424`
