# Allen Visual Behavior 2P -> neural decoder dataset

Converted dataset for decoding experimental and behavioural variables from two-photon
calcium-imaging activity in mouse visual cortex.

## Dataset description

**Source**: Allen Brain Observatory *Visual Behavior — 2P* release `visual-behavior-ophys-1.1.0`
(local NWB files under `/app/data`, read through the AllenSDK `VisualBehaviorOphysProjectCache`).

**Task** (change detection): head-fixed mice view a continuous stream of flashed natural images
(250 ms image, 500 ms grey screen, 750 ms cycle; 5% of image repeats omitted) and earn water
rewards for licking when the image identity changes. Trials are **GO** (a real image change) or
**CATCH** (a sham change), giving four outcomes: hit, miss, false alarm, correct reject.
Aborted trials (premature licking) and auto-rewarded (free-reward) trials are excluded.

**Data selection**
- active (behaving) sessions only — passive sessions have no trial outcomes;
- **familiar** image set only (`OPHYS_1_images_A`, `OPHYS_3_images_A`), matching the analysis
  paper's restriction to familiar stimuli and ensuring the same 8 images in every session;
- all simultaneously recorded imaging planes of one `ophys_session_id` are merged into one
  session (Mesoscope sessions contribute 3–7 planes);
- only valid ROIs (as returned by the AllenSDK) — QC-failed planes/sessions are already
  removed upstream;
- 1 session without eye tracking is dropped; individual trials whose running-speed or pupil
  bins contain no valid sample are dropped (2.6%).

**Neural signal**: the SDK's **detected calcium events** (`BehaviorOphysExperiment.events`),
averaged within each time bin and then z-scored per neuron within the session.

**Alignment / binning**: trials are aligned to `trials.change_time` (real change on GO trials,
sham change on CATCH trials — always a flash onset). Window **[−2.25 s, +3.75 s]** in **750 ms**
bins = **8 bins per trial**, one bin per image-presentation interval; bin 3 is the change
interval, bin 2 the preceding repeat.

## Key statistics

| Statistic | Value |
|---|---|
| Sessions | 91 |
| Subjects (mice) | 38 |
| Neurons | 14,669 (7–666 per session, median 89) |
| Trials | 22,179 (39–392 per session, mean 244) |
| Time bins per trial | 8 (750 ms each) |
| Brain regions | VISp (14,565 neurons), VISl (104 neurons) |
| Cre lines | Slc17a7 (excitatory), Sst, Vip |
| Decoder inputs | none (d_input = 0) |
| Decoder outputs | 5 (see below) |
| Trial outcomes | hit 30.3%, miss 57.1%, false alarm 2.2%, correct reject 10.4% |
| Catch fraction of go+catch | 12.6% (whitepaper: ~12.5%) |

## Output format

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

data['neural'][session][trial]   # float32 (n_neurons, 8) z-scored calcium-event magnitude
data['input'][session][trial]    # float32 (0, 8)  -- no decoder inputs for this task
data['output'][session][trial]   # int64  (5, 8)  categorical labels, see below
```

| # | `output_names` | Classes | `output_values` | Definition |
|---|---|---|---|---|
| 0 | `image_identity` | 8 | im061, im062, im063, im065, im066, im069, im077, im085 | identity of the image presented in that 750 ms interval; omitted flashes keep the repeating image's identity |
| 1 | `image_change` | 2 | no_change, change | 1 in the interval starting at the image change (bin 3 of GO trials), 0 elsewhere; always 0 on CATCH (sham) trials |
| 2 | `running_speed_quintile` | 5 | speed_q1…q5 | mean filtered running speed (cm/s) per bin, discretised at the dataset-wide 20/40/60/80th percentiles (0.016, 3.63, 18.14, 32.51 cm/s) |
| 3 | `pupil_diameter_quintile` | 5 | pupil_q1…q5 | pupil diameter `2*sqrt(pupil_area/pi)` px per bin (blink gaps <= 1 s interpolated), dataset-wide quintiles (74.98, 84.05, 93.43, 108.91 px) |
| 4 | `trial_outcome` | 4 | hit, miss, false_alarm, correct_reject | static per trial, broadcast over the 8 bins |

Other keys: `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`,
`output_names`, `output_values`, and `metadata` (task description, `time_bin_size` = 750 ms,
`temporal_alignment_event`, `off_start` = −2.25, `off_end` = +3.75, per-session `session_info`
with experiment ids / cre line / frame rate / trial counts / quantile edges, and
`session_checks` with the per-session sanity-check results).

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~70 s, 8 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```
Useful switches: `--neural-signal {events,filtered_events}`, `--normalize {none,std,zscore}`,
`--bin-size`, `--off-start`, `--off-end`, `--quantile-scope {global,session}`, `--workers`.

## Decoder performance (provided reference decoder, validation balanced accuracy)

| Output | Chance | Validation |
|---|---|---|
| image_identity | 0.125 | 0.491 |
| image_change | 0.500 | 0.642 |
| running_speed_quintile | 0.200 | 0.439 |
| pupil_diameter_quintile | 0.200 | 0.490 |
| trial_outcome | 0.250 | 0.340 |

For reference, re-running the analysis paper's own decoding analysis on this converted data
(random forest, change interval vs preceding repeat, 5-fold CV, per imaging session) gives
0.653 ± 0.013 for change-vs-repeat and 0.726 ± 0.013 for hit-vs-miss, matching the ranges
reported in the paper (Figures 6A and 6C).

See `/app/CONVERSION_NOTES.md` for every processing decision, all consistency checks against
the reference papers/code, and the validation logs.
