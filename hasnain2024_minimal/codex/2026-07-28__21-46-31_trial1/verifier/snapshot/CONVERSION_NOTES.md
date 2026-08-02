# Conversion Notes

## Scope

This conversion targets the alternating delayed-response / water-cued ALM electrophysiology cohort used for the context analyses in the reference repository. The session list follows the loader combination used in `code/Scripts/Figure 8/Figure8a_thru_c.m` and `code/Scripts/Figure 8/Figure8d.m`.

Included sessions:

1. `JEB6_2021-04-18`
2. `JEB7_2021-04-29`
3. `JEB7_2021-04-30`
4. `EKH1_2021-08-07`
5. `EKH3_2021-08-11`
6. `JGR2_2021-11-16`
7. `JGR2_2021-11-17`
8. `JGR3_2021-11-18`
9. `JEB19_2023-04-21`
10. `JEB19_2023-04-20`
11. `JEB19_2023-04-19`
12. `JEB19_2023-04-18`

## Reference Processing Reproduced

### Neural data

- Alignment event: `goCue`
- Time window: `[-3.0 s, 2.5 s]`
- Bin size: `10 ms`
- Smoothing: causal Gaussian kernel with `N=15`, matching `mySmooth(..., 15, 'reflect')`
- Cluster inclusion before firing-rate filter: all qualities except `garbage`, `gabrga`, `noisy`, `real?`
- Low-firing-rate exclusion: remove units with mean PSTH firing rate `<= 1 Hz`

The low-FR filter was matched to the MATLAB path:

1. Build single-trial smoothed firing rates on the common aligned time axis.
2. Build the same 7 context conditions used by the Figure 8 scripts.
3. Compute the condition-averaged PSTHs.
4. Drop units whose mean across time and conditions is not above 1 Hz.

### Behavioral and task trial filtering

The decoder dataset excludes:

- `early` trials
- `no` response trials
- `stim.enable` trials

This matches the recurring `~early`, `~no`, and `~stim.enable` filtering used in the repository analyses. The remaining trials include both correct and incorrect trials in both contexts.

### Video / kinematics / motion energy

- Video alignment matches the reference code: `frameTimes - video_offset - goCue`
- `video_offset` matches `findVideoOffset`: `mode(obj.sglx.bitcode.bitstart) / obj.sglx.fs - mode(obj.bp.ev.bitStart)`
- Motion energy comes from `motionEnergy_Animal_Date.mat`, is resampled onto the neural time axis, and is nearest-filled at the edges
- Position interpolation and velocity computation follow the repository logic:
  - non-tongue features are nearest-filled
  - tongue invisibility periods become zero velocity
  - non-tongue baseline subtraction reproduces the MATLAB implementation, including subtracting `basederiv(1)` from both `xvel` and `yvel`

## Decoder Variables

### Input

- `time_from_go_cue_sec`: one time-varying input channel equal to the aligned neural time axis

### Outputs

- `lick_direction`
  - `0 = left`
  - `1 = right`
  - Defined as actual lick direction, not instructed side:
    - right if `R & hit` or `L & miss`
    - left if `L & hit` or `R & miss`
- `behavioral_context`
  - `0 = WC`
  - `1 = DR`
- `outcome`
  - `0 = incorrect`
  - `1 = correct`
- `tongue_velocity`
  - scalar tongue speed from the reference x/y tongue velocity components
  - binarized with a per-session 50th-percentile threshold
  - because the repository sets invisible-tongue velocity to zero, if the all-sample median is zero the threshold is recomputed from positive tongue-speed samples only; otherwise several sessions collapse to a trivial constant label
- `paw_velocity`
  - scalar paw speed from the `top_paw` bottom-view marker
  - binarized with the per-session 50th percentile
- `motion_energy`
  - aligned motion energy
  - binarized with the per-session 50th percentile

All outputs are saved as time-varying arrays so they match the training code’s preferred `(d_output, T)` format.

## Sanity Checks Against the Reference

### Structural checks

- `train_decoder.py --verify-only` passes on both `sample_data.pkl` and `converted_data.pkl`
- All sessions have a fixed `T=550` bins
- All neurons in every session are assigned to `ALM`

### Cohort-level checks

Reference paper / code expectations for the alternating-context ALM cohort:

- 12 sessions
- 522 total units
- 214 well-isolated single units

Converted dataset:

- 12 sessions
- 521 units after filtering
- 213 single-unit-quality units after filtering

This is off by one unit in each count. The conversion follows the Figure 8 loader subset and the repository’s low-FR rule, so this remaining discrepancy is likely due to one borderline unit at the 1 Hz cutoff or a loader-era annotation difference. I left the implementation aligned to the code rather than forcing the count.

### Session-by-session retained trials and neurons

| Session | Trials kept | DR | WC | Neurons | Single-quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| `JEB6_2021-04-18` | 260 | 188 | 72 | 32 | 13 |
| `JEB7_2021-04-29` | 209 | 151 | 58 | 66 | 32 |
| `JEB7_2021-04-30` | 157 | 115 | 42 | 27 | 15 |
| `EKH1_2021-08-07` | 214 | 159 | 55 | 48 | 19 |
| `EKH3_2021-08-11` | 322 | 250 | 72 | 66 | 31 |
| `JGR2_2021-11-16` | 185 | 135 | 50 | 29 | 12 |
| `JGR2_2021-11-17` | 195 | 144 | 51 | 53 | 13 |
| `JGR3_2021-11-18` | 193 | 147 | 46 | 28 | 12 |
| `JEB19_2023-04-21` | 168 | 113 | 55 | 33 | 12 |
| `JEB19_2023-04-20` | 166 | 94 | 72 | 67 | 32 |
| `JEB19_2023-04-19` | 146 | 93 | 53 | 37 | 11 |
| `JEB19_2023-04-18` | 200 | 128 | 72 | 35 | 11 |

## Decoder Validation

### Sample dataset (`sample_data.pkl`)

Validation balanced accuracy:

- `lick_direction`: `0.6322`
- `behavioral_context`: `0.7528`
- `outcome`: `0.6232`
- `tongue_velocity`: `0.8168`
- `paw_velocity`: `0.5798`
- `motion_energy`: `0.8085`

### Full dataset (`converted_data.pkl`)

Validation balanced accuracy:

- `lick_direction`: `0.6336`
- `behavioral_context`: `0.7669`
- `outcome`: `0.6245`
- `tongue_velocity`: `0.8190`
- `paw_velocity`: `0.6189`
- `motion_energy`: `0.7966`

These are all above the binary chance baseline of `0.5`, including the behavioral context and movement outputs that are most sensitive to alignment and label construction.

## Noted Discrepancies

1. The loader subset used by the Figure 8 scripts yields 12 sessions across 7 animal IDs, whereas the copied methods text says 12 sessions from 6 mice.
2. The converted neuron and single-unit-quality totals are each lower than the paper count by one.

I kept the implementation anchored to the repository’s explicit session loaders and filtering logic rather than modifying the cohort to force an exact paper-text match.
