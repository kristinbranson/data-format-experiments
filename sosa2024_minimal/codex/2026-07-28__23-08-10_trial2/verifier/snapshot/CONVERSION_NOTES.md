# Conversion Notes

## Scope

This conversion turns the supplied NWB sessions from the paper "A flexible hippocampal population code for experience relative to reward" into the decoder format required by `train_decoder.py`.

Output files created by the conversion workflow:

- `convert_data.py`
- `converted_data.pkl`
- `sample_data.pkl`
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`

## Source Material Used

- `paper.pdf`
- `methods.txt`
- `code/src/reward_relative`
- `data/sub-*/sub-*_behavior+ophys.nwb`

The paper text and repo code were used together. For neural preprocessing, the repo logic was treated as the operational reference.

## Main Decisions

### Session inventory

- The supplied `data/` directory contains `152` NWB sessions, not `154`.
- The missing two sessions are `m11` `ses-01` and `ses-02`.
- The final full dataset therefore contains `152` sessions across `11` subjects.

### Temporal alignment

- Trials are aligned to `trial start`.
- For each trial, the kept sample range is `[trial_start_idx : teleport_idx)`.
- The teleport sample itself is excluded.
- `input[0]` is `time_from_trial_start_s`, computed from the behavior timestamps relative to the first frame of each trial.

### Trial labels

- Trial environment and reward-zone labels are derived from the session scene name, matching the task structure in the paper.
- The parser handles:
  - `Env1_LocationA`
  - `Env1_LocationA_to_B`
  - `Env1_A_to_Env2_B`
  - `Env2_B_to_Env1_A`
- For switch sessions, the switch point is trial `30`, matching the paper/task structure.
- The scene-derived environment was cross-checked against the aligned behavior `environment` stream on every kept trial.

### Neural signal

The saved neural activity is trial-wise OASIS event activity recomputed from Suite2p fluorescence and neuropil, rather than taking the NWB `Deconvolved` array directly.

Processing implemented in `convert_data.py`:

1. Keep only curated Suite2p ROIs with `iscell[:, 0] == 1`.
2. Use the paper-style neuropil subtraction coefficient `0.7`.
3. Restrict the dF/F baseline computation to trial periods only.
4. Recreate the paper repo's maximin baseline per trial:
   - Gaussian smoothing with `sigma=15` samples
   - Minimum filter size `300`
   - Maximum filter size `300`
5. Compute `dF/F = (F - baseline) / abs(baseline)`.
6. Smooth dF/F with a Gaussian `sigma=2` samples.
7. Run OASIS deconvolution with:
   - `tau=0.7`
   - `rate=frame_rate_hz`

### Frame rate

- Frame rate is inferred from the behavior timestamps, not trusted from the NWB `starting_time/rate` metadata.
- This matters for multi-plane sessions where metadata can report `31.015625` even though the effective imaging/behavior sample spacing is `15.5078125` Hz.
- The final mean frame rate was `15.507813` Hz, so `metadata["time_bin_size"]` is `64.4836` ms.

### Multi-plane handling

- Some sessions contain `plane0` and `plane1`.
- A few of these sessions have behavior and fluorescence streams that differ by one frame.
- All streams in a session were truncated to the common minimum length across behavior arrays, timestamps, and all fluorescence planes before further processing.

### Neuron exclusion

- Putative interneurons were excluded when Pearson correlation between dF/F and running speed exceeded `0.5`, computed over valid in-trial samples.
- This follows the paper's description of excluding speed-correlated putative interneurons.

### Trial exclusion

- A full trial was dropped if more than `30%` of its frame samples had `lick count > 2`.
- This matches the paper's lick-sensor error criterion.
- Reward outcome and previous-trial outcome were computed before removing these trials, so a kept trial still uses the actual previous trial's reward outcome even if that previous trial was later excluded for lick-sensor artifacts.

### Behavioral outputs

Decoder outputs were built exactly as required by the task prompt:

- `distance_to_reward_zone`
  - Signed nearest distance to the active reward zone
  - Before zone: `position - zone_start`
  - In zone: `0`
  - After zone: `position - zone_end`
  - Binned into the requested `7` categories
- `absolute_position`
  - Corridor clipped to `0` to `450` cm
  - Binned into `5` equal bins of `90` cm each
- `speed`
  - Binned into the requested `5` categories
- `lick`
  - Binary `lick_count > 0`
- `reward_zone_location`
  - Trial-level label `A/B/C`
  - Repeated across timepoints in the trial
- `reward_outcome`
  - Trial-level label from reward timestamps falling between trial start and teleport
  - Repeated across timepoints in the trial

### Decoder inputs

Each trial input array has shape `(4, T)` with:

1. `time_from_trial_start_s`
2. `environment_type`
3. `trial_number`
4. `previous_trial_outcome`

Notes:

- `environment_type` is `0` for `ENV1`, `1` for `ENV2`.
- `trial_number` is copied from the aligned NWB `trial number` stream at trial start.
- `previous_trial_outcome` is `0` for the first trial of a session.

### Brain regions

- All neurons are labeled `CA1`.

## Full-Dataset Sanity Checks

### Structural checks

- `train_decoder.py --verify-only` reported: `Data format is valid, no errors or warnings.`
- Sessions: `152`
- Trials kept: `12135`
- Subjects: `11`
- Brain-region count: `CA1: 138262 neurons`

### Conversion summary

From `conversion_full_out.txt`:

- Sessions processed: `152`
- Total trials before lick-error exclusion: `12216`
- Total trials kept: `12135`
- Trials removed by lick-error criterion: `81`
- Trials per session before exclusion: `80.37 ± 6.14`
- Trials per session after exclusion: `79.84 ± 6.86`
- Rewarded trial fraction before exclusion: `0.8466`
- Curated neurons per session: `912.36 ± 448.71`
- Kept neurons per session: `909.62 ± 447.96`
- Interneuron exclusion fraction: `0.360% ± 0.615%`
- Frame rate: `15.507813 ± 0.000000` Hz
- Total kept trial timepoints: `2576026`
- Trial length: `212.28 ± 102.86` frames
- Multi-plane sessions: `28`

### Checks against paper/methods expectations

- Lick-sensor artifact exclusion produced exactly `81` dropped trials, which matches the criterion described in the paper text.
- Interneuron exclusion was very small, `0.360% ± 0.615%`, which is close to the methods value of about `0.42% ± 0.85%`.
- Reward-zone labels were nearly perfectly balanced across the full dataset:
  - `A`: `0.332`
  - `B`: `0.336`
  - `C`: `0.333`
- Reward outcome distribution in the final kept dataset was:
  - `omitted`: `0.158`
  - `rewarded`: `0.842`

### Decoder validation

From `train_decoder_full_out.txt`:

- Training trials: `9702`
- Test trials: `2433`
- Test loss: `0.809040`

Validation balanced accuracy:

- `distance_to_reward_zone`: `0.5495` vs chance `0.1429`
- `absolute_position`: `0.6302` vs chance `0.2000`
- `speed`: `0.5831` vs chance `0.2000`
- `lick`: `0.7440` vs chance `0.5000`
- `reward_zone_location`: `0.8370` vs chance `0.3333`
- `reward_outcome`: `0.5673` vs chance `0.5000`

These are comfortably above chance for all outputs, which is a strong end-to-end sanity check that the alignment and formatting are coherent.

## Sample Dataset

`sample_data.pkl` was generated from the already-built full dataset.

Selected session keys:

- `m3`: `1`, `3`, `8`, `14`
- `m11`: `3`, `8`, `14`
- `m17`: `1`, `3`, `8`, `14`

Selection rule:

- Keep the first `10` kept trials from each selected session.
- Skip any selected session that would end up with fewer than `2` trials after this truncation.

Result:

- Sessions: `11`
- Trials: `110`

Notes:

- The sample subset spans early, switch, environment-switch, and late sessions.
- It includes both single-plane and multi-plane sessions.
- In this particular subset, `reward_zone_location` only takes values `B` and `C`; that is acceptable for the sample file and its validation/training still passed.

Sample decoder validation balanced accuracy:

- `distance_to_reward_zone`: `0.5459`
- `absolute_position`: `0.6764`
- `speed`: `0.5694`
- `lick`: `0.7207`
- `reward_zone_location`: `0.9981`
- `reward_outcome`: `0.9140`

## Reproducibility

Commands used:

```bash
python /app/convert_data.py --full-only > /app/conversion_full_out.txt 2>&1
python /app/convert_data.py --sample-only --sample-from-full /app/converted_data.pkl > /app/conversion_sample_out.txt 2>&1
python /app/train_decoder.py /app/converted_data.pkl --verify-only --cpu > /app/verification_full_out.txt 2>&1
python /app/train_decoder.py /app/sample_data.pkl --verify-only --cpu > /app/verification_sample_out.txt 2>&1
python /app/train_decoder.py /app/converted_data.pkl --cpu > /app/train_decoder_full_out.txt 2>&1
python /app/train_decoder.py /app/sample_data.pkl --cpu > /app/train_decoder_sample_out.txt 2>&1
```

Optional inspection artifact present in the workspace:

- `sample_trials.png`

## Final Output Summary

- The required pickle structure is valid.
- The conversion uses the paper/repo processing logic for ROI curation, dF/F, deconvolution, trial exclusion, and interneuron removal.
- The full dataset passes format verification with no warnings.
- Decoder training succeeds on both full and sample datasets.
