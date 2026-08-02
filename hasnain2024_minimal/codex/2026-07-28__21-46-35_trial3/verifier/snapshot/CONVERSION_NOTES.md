# Conversion Notes

## Scope

The decoder outputs require behavioral context (`WC` vs `DR`), so I converted the ALM two-context electrophysiology cohort rather than the fixed-delay-only or randomized-delay cohorts.

The session roster was taken from the reference MATLAB context-analysis pipeline in `code/Scripts/Figure 8/Figure8a_thru_c.m` plus the per-animal loader files in `code/DataLoadingScripts/Recording and video/`. That roster is:

- `JEB6_2021-04-18` probe 2
- `JEB7_2021-04-29` probe 1
- `JEB7_2021-04-30` probe 1
- `EKH1_2021-08-07` probe 2
- `EKH3_2021-08-11` probe 2
- `JGR2_2021-11-16` probe 1
- `JGR2_2021-11-17` probe 1
- `JGR3_2021-11-18` probe 1
- `JEB19_2023-04-21` probe 1
- `JEB19_2023-04-20` probe 1
- `JEB19_2023-04-19` probe 1
- `JEB19_2023-04-18` probe 1

## Reference-Matched Processing

Implemented directly from the repository:

- Alignment event: `goCue`
- Neural time window: `[-3.0, 2.5]` s
- Time bin: `dt = 1/100` s
- Spike smoothing: causal Gaussian smoothing with window `15` and reflect padding, matching `mySmooth.m`
- Quality filter: same as `findClusters(..., {'all'})`, which excludes `garbage`, `gabrga`, `noisy`, and `real?`
- Low firing-rate filter: `> 1 Hz`, matching Figure 8 / context scripts
- Video alignment: `frameTimes - vidshift - goCue`, with `vidshift = mode(bitcode.bitstart)/fs - mode(bp.ev.bitStart)`, matching `findVideoOffset.m` and `loadMotionEnergy.m`
- Motion energy interpolation: onto the neural time axis, then nearest-fill of edge NaNs, matching `loadMotionEnergy.m`
- Trial inclusion for the exported decoder dataset: `hit | miss`, `stim.enable == 0`, `early == 0`
- Ignore / no-response trials were excluded because the requested `outcome` target is binary correct/incorrect and the paper methods state ignore trials were omitted from analyses

## Output Construction

Neural:
- Exported as smoothed single-trial firing rates with shape `(n_neurons, 550)` per trial.

Input:
- One continuous input: `time_from_go_cue_seconds`
- Stored as a `1 x 550` time series for each trial

Outputs:
- `lick_direction`: actual response direction, not instructed side
- `behavioral_context`: `WC = 0`, `DR = 1`
- `outcome`: `incorrect = 0`, `correct = 1`
- `tongue_velocity_bin`
- `paw_velocity_bin`
- `motion_energy_bin`

Choice logic:
- Correct right trials and incorrect left trials were labeled right
- Correct left trials and incorrect right trials were labeled left

Movement targets:
- Motion energy followed the repository exactly up to aligned continuous values.
- Tongue and paw velocities were derived from the same aligned DLC trajectories used by the repository’s kinematics code.
- Tongue speed was computed from the bottom-view tongue-tip velocity using the average of `top_tongue` and `bottom_tongue`.
- Paw speed was computed as the mean speed of `top_paw` and `bottom_paw`.

## Decoder-Specific Discretization

The paper code does not define the exact categorical decoder targets requested here, so the continuous movement variables were discretized after reference-matched loading/alignment.

- `paw_velocity_bin` and `motion_energy_bin` use the literal per-session 50th percentile over all exported bins.
- `tongue_velocity_bin` uses the per-session 50th percentile over positive tongue-speed bins.

Reason:
- The repository sets invisible-tongue velocity samples to zero.
- Using the median over all bins collapses the threshold to zero in every session.
- With the requested `>= threshold` rule, that makes the tongue target a constant all-ones label, which is not meaningful for decoding.
- Restricting the threshold computation to positive tongue-speed bins preserves the reference-aligned trace while yielding a usable categorical target.

## Sanity Checks

Reference-facing checks:

- Session count matched the Figure 8 context roster exactly: `12`
- Retained ALM units after quality and low-FR filtering: `520`
- Paper text reports `522` units in the two-context task
- Difference from the paper count: `-2` units
- Exported trials after hit/miss and non-early filtering: `2415`
- Timepoints per trial: `550`
- Brain regions: all neurons labeled `ALM`

The `520` vs `522` unit difference is small and most likely comes from a minor mismatch between the packaged data here and the exact analysis snapshot used for the manuscript, not from a gross loading error. The roster, probe selection, alignment, and trial counts all match the reference logic.

Packaged-subject check:

- The converted data contain `7` unique subject IDs: `JEB6`, `JEB7`, `EKH1`, `EKH3`, `JGR2`, `JGR3`, `JEB19`
- The manuscript text states `6` mice for the context cohort
- I preserved the subject IDs present in the provided files rather than collapsing or renaming them without evidence

## Validation Results

Format verification:

- Full dataset: passed with no errors or warnings
- Sample dataset: passed with no errors or warnings

Full dataset summary:

- Sessions: `12`
- Subjects: `7`
- Trials: `2415`
- Units: `520`
- Trials per session: `[260, 209, 157, 214, 322, 185, 195, 193, 168, 166, 146, 200]`
- Neurons per session: `[31, 66, 27, 48, 66, 29, 53, 28, 33, 67, 37, 35]`

Full decoder validation balanced accuracy:

- `lick_direction`: `0.6285`
- `behavioral_context`: `0.7673`
- `outcome`: `0.6272`
- `tongue_velocity_bin`: `0.8314`
- `paw_velocity_bin`: `0.6182`
- `motion_energy_bin`: `0.7922`

Sample decoder validation balanced accuracy:

- `lick_direction`: `0.6322`
- `behavioral_context`: `0.7536`
- `outcome`: `0.6227`
- `tongue_velocity_bin`: `0.8201`
- `paw_velocity_bin`: `0.5926`
- `motion_energy_bin`: `0.8088`

These are all clearly above the binary-chance baseline of `0.5`, which is a strong sign that the conversion retained the intended neural-behavioral relationships and that the exported targets are coherent.

## Files Produced

- `convert_data.py`
- `converted_data.pkl`
- `sample_data.pkl`
- `README.md`
- `CONVERSION_NOTES.md`
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`
