# Conversion Notes

## Scope

I converted the paper's two-context ALM ephys cohort into the required decoder format, aligned to the session object's `goCue` field and exported to `converted_data.pkl`.

The session roster follows the paper's Figure 8 context-analysis loader in [Figure8a_thru_c.m](/app/code/Scripts/Figure%208/Figure8a_thru_c.m):

1. `JEB6_2021-04-18` probe 2
2. `JEB7_2021-04-29` probe 1
3. `JEB7_2021-04-30` probe 1
4. `EKH1_2021-08-07` probe 2
5. `EKH3_2021-08-11` probe 2
6. `JGR2_2021-11-16` probe 1
7. `JGR2_2021-11-17` probe 1
8. `JGR3_2021-11-18` probe 1
9. `JEB19_2023-04-21` probe 1
10. `JEB19_2023-04-20` probe 1
11. `JEB19_2023-04-19` probe 1
12. `JEB19_2023-04-18` probe 1

## Source-Matching Decisions

- Neural alignment matches the shared MATLAB pipeline:
  - align to `goCue`
  - bin from `-3.0` s to `2.5` s
  - `dt = 0.01` s
  - causal Gaussian smoothing with `smooth = 15` and the same boundary handling as `mySmooth.m`
- Unit filtering matches the shared code path:
  - start from `quality = {'all'}`
  - drop `garbage`, `gabrga`, `noisy`, and `real?`
  - drop units with mean firing rate `<= 1 Hz`
  - keep sessions only if at least 10 units remain
- Trial export filtering follows the paper text:
  - exclude early-lick trials
  - exclude no-response trials
  - exclude `stim.enable` trials
  - keep hit and miss trials from both DR and WC contexts
- Motion energy loading matches `loadMotionEnergy.m`:
  - load the companion `motionEnergy_*.mat`
  - correct video timing by the same `findVideoOffset` logic
  - interpolate onto the neural time axis
  - fill missing values with nearest samples
- Kinematic alignment matches `findPosition.m` and `findVelocity.m`:
  - interpolate tracked positions onto the neural time axis
  - preserve the paper code's special tongue handling
  - preserve the MATLAB baseline-subtraction behavior for non-tongue velocities

## Decoder-Specific Variable Construction

- `input[trial]` is `time_from_go_cue_s`, shape `(1, 550)`.
- `output[trial]` is shape `(6, 550)`:
  - `lick_direction`: constant over time, `left=0`, `right=1`, taken from `bp.R`
  - `behavioral_context`: constant over time, `WC=0`, `DR=1`, derived from `bp.autowater`
  - `outcome`: constant over time, `incorrect=0`, `correct=1`, derived from `bp.hit`
  - `tongue_velocity_bin`: time-varying
  - `paw_velocity_bin`: time-varying
  - `motion_energy_bin`: time-varying

For the movement outputs, the decoder task required discrete bins that are not present in the paper. I therefore discretized after matching the paper's continuous preprocessing:

- `tongue_velocity`:
  - computed as mean speed across the tracked tongue landmarks from both cameras
  - because the shared MATLAB code uses `0` as a placeholder when the tongue is not visible, I estimated the 50th-percentile threshold from strictly positive tongue-speed samples within each session, then applied that threshold to the full time series
- `paw_velocity`:
  - computed as mean speed across `top_paw` and `bottom_paw`
  - thresholded at the session median over exported trials/timepoints
- `motion_energy`:
  - thresholded at the session median over exported trials/timepoints

## Sanity Checks

- `goCue` is finite on WC trials in all exported sessions, which is important because the paper scripts treat this field as the shared "go cue / water drop" alignment reference.
- All trials have exactly 550 time bins after conversion.
- All exported neurons are labeled `ALM`, consistent with the chosen probe metadata loaders.
- The converted dataset passed `train_decoder.py --verify-only` with no errors or warnings on both the full and sample files.

## Final Dataset Statistics

From `conversion_full_out.txt` and `verification_full_out.txt`:

- Sessions: 12
- Subjects: 7
- Trials: 2415
- Units after filtering: 520
- Trials per session:
  - `260, 209, 157, 214, 322, 185, 195, 193, 168, 166, 146, 200`
- Units per session:
  - `31, 66, 27, 48, 66, 29, 53, 28, 33, 67, 37, 35`

## Validation Results

From `train_decoder_full_out.txt`:

- Validation balanced accuracy:
  - `lick_direction`: `0.6210`
  - `behavioral_context`: `0.7668`
  - `outcome`: `0.6250`
  - `tongue_velocity_bin`: `0.8329`
  - `paw_velocity_bin`: `0.6271`
  - `motion_energy_bin`: `0.7913`

From `train_decoder_sample_out.txt`:

- Validation balanced accuracy:
  - `lick_direction`: `0.4919`
  - `behavioral_context`: `0.8436`
  - `outcome`: `0.4968`
  - `tongue_velocity_bin`: `0.8131`
  - `paw_velocity_bin`: `0.5959`
  - `motion_energy_bin`: `0.7548`

The sample subset is intentionally small and stratified, so the lower `lick_direction` and `outcome` scores there are not surprising. The full dataset performs clearly above chance for every output.

## Paper/Code Discrepancies

- The paper text reports `12 sessions`, `522 units`, and `6 mice` for the two-context cohort.
- The released Figure 8 session loader names 12 sessions from 7 distinct animal IDs.
- My Python port of the released code path yields 520 units after quality and low-FR filtering, which is 2 units below the paper text.

I kept the released-code session roster because it is explicit and task-specific, and the final count is very close to the reported total.
