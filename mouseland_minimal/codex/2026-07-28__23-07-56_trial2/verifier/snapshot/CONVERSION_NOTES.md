# Conversion Notes

## Sources Used

- `paper.pdf`
- `methods.txt`
- `code/utils.py`
- `code/data_process_script.ipynb`

## Core Reconstruction

- The raw spike release contains 89 physical recordings in `data/spk`, which matches the paper’s Methods section.
- The behavior exports contain 99 keys because some recordings are repeated across analysis files or exposed through paired `swap1` and `swap2` views.
- The converter rebuilds sessions at the 89-recording level by stripping `_swap1` and `_swap2` suffixes, merging all behavior views that point to the same physical recording, and verifying that the timing and trial annotations match across those views.

## Neural Processing

- Neural activity comes from the deconvolved traces stored in `*_neural_data.npy`.
- Brain-region labels come from the retinotopy files in `data/retinotopy`.
- The same grouped area definitions used in `code/utils.py` are used here:
  - `V1`: retinotopy code `8`
  - `mHV`: codes `0, 1, 2, 9`
  - `aHV`: codes `3, 4`
  - `lHV`: codes `5, 6`
- Neurons outside these visual-cortex groups are excluded, consistent with the reference analysis code paths that remove neurons outside visual cortex.

## Trial Alignment And Frame Filtering

- Temporal alignment event: corridor entry (`Trial_start_time`, `StartFr`).
- Only frames satisfying `ft_move > 0` and `ft_CorrSpc == True` are used.
- This follows the paper’s statement that analyses only considered timepoints during running and matches the reference code’s repeated use of moving corridor masks.
- Trials without usable running corridor frames are dropped.

## Decoder-Facing Representation

- The decoder task explicitly requires 4 equal 1 m position bins, so each trial is represented by 4 ordered samples corresponding to corridor positions `0-1 m`, `1-2 m`, `2-3 m`, and `3-4 m`.
- Within each trial, neural activity and continuous task variables are interpolated onto the centers of those bins from the retained running-in-corridor frames.
- Inputs:
  - `time_to_sound_cue_s`
  - `day_of_training`
  - `time_since_trial_start_s`
  - `reward_availability`
- Outputs:
  - `visual_stimulus_category`
  - `licking`
  - `position_bin`
  - `running_speed_bin`

## Two Intentional Format-Driven Deviations

- The reference analyses often operate either on raw running frames or on finer position-interpolated activity, but the decoder task here requires exactly 4 spatial bins. The exported trials therefore use those 4 bins directly.
- A deterministic neuron cap is applied per session after visual-cortex filtering so the expanded trial structure remains trainable with the provided decoder harness. The raw neuron counts are preserved in metadata and summarized in the validation notes.

## Sanity Checks

- One-session smoke test:
  - `verify_data_format` passes with no errors or warnings.
  - `train_validate_decoder` runs successfully on a miniature single-session export built from the same code path.
- Full conversion sanity checks from `conversion_full_out.txt`:
  - 89 reconstructed sessions.
  - 19 subjects.
  - Raw neuron count range: `20,547` to `89,577`, matching the Methods text.
  - Selected neuron count range: `512` to `512`.
  - Raw trial count range: `84` to `789`.
  - Kept trial count range: `84` to `789`.
  - Empty running trials dropped: `0`.
  - Median frame period: `0.314694 s`.
  - Global running-speed quartile edges: `16.592`, `28.701`, `43.168`.
  - Full artifact size: `156 MB`.
  - Sample artifact size: `17 MB`.

## Validation Results

- `verification_sample_out.txt`
  - `verify_data_format` passed with no errors or warnings.
  - Sample dataset summary:
    - 8 sessions
    - 4,033 trials
    - 8 subjects
    - 512 neurons per session
    - 4 time bins per trial
- `verification_full_out.txt`
  - `verify_data_format` passed with no errors or warnings.
  - Full dataset summary:
    - 89 sessions
    - 38,110 trials
    - 19 subjects
    - 512 neurons per session
    - 4 time bins per trial
- `train_decoder_sample_out.txt`
  - GPU path (`NVIDIA L4`) ran successfully.
  - Validation balanced accuracy:
    - `visual_stimulus_category`: `0.9219`
    - `licking`: `0.7804`
    - `position_bin`: `0.9388`
    - `running_speed_bin`: `0.6089`
- `train_decoder_full_out.txt`
  - Full training run completed successfully on GPU.
  - Training balanced accuracy:
    - `visual_stimulus_category`: `0.9982`
    - `licking`: `0.9888`
    - `position_bin`: `1.0000`
    - `running_speed_bin`: `0.9500`
  - Validation balanced accuracy:
    - `visual_stimulus_category`: `0.8637`
    - `licking`: `0.8226`
    - `position_bin`: `0.9400`
    - `running_speed_bin`: `0.5958`
