# Conversion Notes

## Sources Used

- Paper: `paper.pdf`
- Extracted methods: `methods.txt`
- Reference code: `/app/code`
- Data: `/app/data`

The conversion follows the public NWB export while mirroring the paper/code logic for:
- reward-zone schedules from scene names
- trial alignment
- curated-cell selection
- extra interneuron exclusion
- per-trial environment and reward labels

## Core Decisions

### Neural signal

- Used the NWB `processing/ophys/Deconvolved` traces as the decoder neural input.
- Kept only manually curated ROIs with `iscell[:, 0] == 1`.
- Reconstructed per-trial dF/F from NWB fluorescence and neuropil only for the putative-interneuron screen, matching the reference logic:
  - neuropil subtraction with coefficient `0.7`
  - per-trial maximin baseline
  - Gaussian smoothing before baseline estimation
  - final Gaussian smoothing with sigma `2`
- Excluded putative interneurons with `corr(dff, speed) > 0.5`, matching the manuscript criterion and `dayData.py`.

### Trial window

- Trials are aligned to `trial_start`.
- For each trial, frames are kept from that start until the last frame still on the corridor.
- Corridor frames are defined by position in `[0, 450]` cm; teleport-zone frames are excluded.
- This matches the reference focus on the 450 cm track rather than the teleport period.

### Trial schedule

- Parsed the session `scene` from the NWB identifier and reproduced the same schedule logic as `reward_relative.behavior.get_reward_zones`.
- Fixed sessions use a single reward zone.
- Switch sessions use the first `30` trials before the switch and the remaining trials after the switch.
- Day-8 environment-switch sessions such as `Env1_C_to_Env2_B` and `Env2_B_to_Env1_A` are handled explicitly.

### Decoder inputs

Inputs are stored as time-varying `(4, T)` arrays for every trial:
- `time_from_trial_start_s`
- `environment_type`
- `trial_number`
- `previous_trial_outcome`

Notes:
- `environment_type` uses the manuscript/code mapping `Env1 -> 0`, `Env2 -> 1`.
- `trial_number` is `0`-indexed within session, matching the aligned behavior stream.
- `previous_trial_outcome` uses the previous imaged trial in the same session; the first trial is set to `0`.

### Decoder outputs

Outputs are stored as time-varying `(6, T)` categorical arrays:
- `distance_to_reward_zone`
- `absolute_position_bin`
- `speed_bin`
- `lick`
- `reward_zone_location`
- `reward_outcome`

Details:
- Distance to reward zone is signed distance to the nearest point in the active 50 cm reward zone:
  - before zone: negative distance to zone start
  - inside zone: `0`
  - after zone: positive distance to zone end
- Absolute position uses five equal 90 cm corridor bins.
- Speed uses the aligned NWB speed trace and the requested bin edges.
- Licks are binarized from the framewise cumulative lick counts with `lick > 0`.
- Reward outcome is derived from the sparse NWB reward timestamps within each trial.

## Sanity Checks

### Manuscript-level checks

- Converted mice: `11`, matching the switch-task cohort in the paper.
- Converted sessions: `152`.
  - `m11` contributes `12` sessions, consistent with imaging starting on day 3.
- Converted trials: `12,216`.
- Mean trials/session: `80.368`, close to the manuscript value `80.5 ± 7.4`.
- Omission fraction: `0.1534`, close to the manuscript target of approximately `15%`.

### Alignment checks

- Environment schedule parsed from the scene name matched the behavior stream on every trial:
  - `0` environment-mismatched trials.
- Reward-zone schedule parsed from the scene name matched the observed reward-zone occupancy:
  - mean agreement `1.0000` when reward-zone occupancy was observed.

### Neuron curation checks

- Curated cells/session before interneuron exclusion:
  - mean `912.355`
  - min `155`
  - max `2341`
- Cells/session after interneuron exclusion:
  - mean `910.013`
  - min `154`
  - max `2324`
- Interneuron removal fraction/session:
  - mean `0.0031`
  - sd `0.0055`

This is close to the manuscript statement that speed-correlated interneuron exclusion removed a very small fraction of cells.

### Timing checks

- Median frame interval from the aligned behavior timestamps:
  - `0.0644836` s
  - `64.4836` ms
- This matches the approximately `15.5 Hz` per-plane sampling described in the paper.

### Verification checks

- `train_decoder.py --verify-only` passed on both:
  - `converted_data.pkl`
  - `sample_data.pkl`
- No format errors or warnings were reported.

## Sample Dataset

`sample_data.pkl` is a deterministic subset:
- subject: `m3`
- experiment days: `1, 3, 5, 7, 8, 10, 12, 14`

This subset preserves:
- both environments
- within-environment reward switches
- the day-8 environment switch
- all three reward-zone labels

## Decoder Results

### Sample dataset

Validation balanced accuracy from `train_decoder_sample_out.txt`:
- `distance_to_reward_zone`: `0.4251` vs chance `0.1429`
- `absolute_position_bin`: `0.5519` vs chance `0.2000`
- `speed_bin`: `0.4437` vs chance `0.2000`
- `lick`: `0.5784` vs chance `0.5000`
- `reward_zone_location`: `0.7515` vs chance `0.3333`
- `reward_outcome`: `0.5308` vs chance `0.5000`

### Full dataset

Training balanced accuracy from `train_decoder_full_out.txt`:
- `distance_to_reward_zone`: `0.4542` vs chance `0.1429`
- `absolute_position_bin`: `0.5325` vs chance `0.2000`
- `speed_bin`: `0.4494` vs chance `0.2000`
- `lick`: `0.5763` vs chance `0.5000`
- `reward_zone_location`: `0.8367` vs chance `0.3333`
- `reward_outcome`: `0.5915` vs chance `0.5000`

Validation balanced accuracy from `train_decoder_full_out.txt`:
- `distance_to_reward_zone`: `0.4091` vs chance `0.1429`
- `absolute_position_bin`: `0.5058` vs chance `0.2000`
- `speed_bin`: `0.4248` vs chance `0.2000`
- `lick`: `0.5681` vs chance `0.5000`
- `reward_zone_location`: `0.8015` vs chance `0.3333`
- `reward_outcome`: `0.5185` vs chance `0.5000`
