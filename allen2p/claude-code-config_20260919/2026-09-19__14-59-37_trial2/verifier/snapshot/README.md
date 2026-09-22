# Visual Behavior 2P → neural-decoder dataset

`converted_data.pkl` contains the Allen Brain Observatory **Visual Behavior 2P** dataset
(`visual-behavior-ophys-1.1.0`, the copy in `/app/data`) reformatted for the decoder in
`train_decoder.py`. Neural activity (2-photon dF/F) is provided per trial, together with the
stimulus and behavioural variables to be decoded from it.

## Dataset description

Head-fixed mice perform a go/no-go **visual change-detection task** while 2-photon calcium imaging
is performed in visual cortex (V1 = VISp, LM = VISl). Natural images are flashed for 250 ms every
750 ms (5 % of flashes omitted); the mouse earns water by licking within 150–750 ms of a change in
image identity. Trials are **Go** (the image changes) or **Catch** (sham change, image does not
change), giving the four outcomes hit / miss / false alarm / correct rejection. Aborted (early
lick) and auto-rewarded (free reward) trials are excluded, as are passive-viewing sessions.

| | |
|---|---|
| Sessions | 171 (one per `ophys_session_id`; 165 single-plane + 6 Multiscope sessions whose 3–7 simultaneously recorded planes are concatenated) |
| Mice | 38 (2–9 sessions each) |
| Neurons | 29 168 (VISp 29 006, VISl 162); 6–666 per session, mean 170.6 |
| Cre lines | Slc17a7 (excitatory), Sst, Vip |
| Trials | 43 975 Go+Catch trials; 39–409 per session, mean 257 |
| Trial length | variable, 7.0–12.6 s (mean 8.45 s) = 217–389 time bins |
| Time bin | 32.258 ms (1/31 s, the single-plane 2P frame period) |
| Alignment | trial start (`trials.start_time`); every stream sampled at ophys frame times |
| Neural signal | dF/F (`dff_traces.dff` from the AllenSDK, i.e. the Allen pipeline's detrended dF/F) |
| Behaviour | hit rate 36.2 %, false-alarm rate 15.1 %, catch trials 12.5 % of Go+Catch |

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['neural'][s][t]      # (n_neurons, T) float32 dF/F for trial t of session s
Y = data['output'][s][t]      # (5, T) int64 categorical outputs
U = data['input'][s][t]       # (0, T) — this task has no decoder inputs

data['output_names']          # ['image_identity', 'image_change', 'running_speed_quintile',
                              #  'pupil_diameter_quintile', 'trial_outcome']
data['output_values'][0][Y[0, 42]]        # e.g. 'im065' -> image shown in time bin 42
data['subjects'][data['subject_idx'][s]]  # mouse id of session s
[data['brain_regions'][i] for i in data['brain_region_idx'][s]]   # region of each neuron
data['metadata']['session_info'][s]       # ids, cre line, session type, depths, quantile edges, ...
```

Validate / train the reference decoder:

```bash
python train_decoder.py /app/converted_data.pkl --verify-only     # format + summary statistics
python train_decoder.py /app/converted_data.pkl --plot-samples    # train + accuracy + figures
```

Regenerate the dataset (≈80 s on 16 cores):

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full --workers 16
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing   # 2 sessions + diagnostics
```

## Output format specification

`data` is a dict with `neural`, `input`, `output` (lists of sessions, each a list of per-trial
arrays), `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`,
`output_names`, `output_values` and `metadata`.

| # | Output | Classes | Definition |
|---|--------|---------|------------|
| 0 | `image_identity` | 17: `gray` + 16 image names | The image on the monitor in that time bin. `gray` covers the 500 ms inter-flash blank and omitted flashes. Image sets A and B are disjoint, so each session shows 8 of the 16 images. |
| 1 | `image_change` | 2: `no_change`, `change` | 1 for the time bins inside the 250 ms presentation of the changed image on Go trials (i.e. right after the image identity changes), 0 otherwise. Catch trials (sham change) are all 0. |
| 2 | `running_speed_quintile` | 5 | Running speed (cm/s, 60 Hz, 10 Hz low-pass `running_speed`) interpolated onto the bin centres, discretised into five equal-percentile bins **within each session**. |
| 3 | `pupil_diameter_quintile` | 5 | Pupil width (pixels, ~30 Hz `eye_tracking.pupil_width`, NaN on blinks → linearly interpolated), same discretisation. |
| 4 | `trial_outcome` | 4: hit, miss, false_alarm, correct_reject | Constant over the trial. |

Marginal distributions: gray 0.670 / each image ≈0.020; change 0.025; each quintile exactly 0.200;
hit 0.313, miss 0.562, false alarm 0.018, correct rejection 0.107 (per time bin).

## Decoder accuracy (reference decoder, full dataset)

| Output | Validation balanced accuracy | Chance |
|---|---|---|
| image_identity | 0.407 | 0.059 |
| image_change | 0.626 | 0.500 |
| running_speed_quintile | 0.282 | 0.200 |
| pupil_diameter_quintile | 0.267 | 0.200 |
| trial_outcome | 0.297 | 0.250 |

For reference, running the *paper's* decoders (random forest on the 400 ms after image onset, per
session) on this converted data gives 76.8 % for change-vs-repeat and 78.7 % for hit-vs-miss,
matching or exceeding Piet et al. (2024), Fig. 6.

## Curation

Kept: all active-behavior sessions (OPHYS_1/3/4/6) whose NWB files are present locally and that
have eye tracking; all ROIs the AllenSDK returns (the Allen pipeline's ROI quality control is
already applied); all Go and Catch trials. Excluded: passive sessions (no licking/reward → no trial
outcome), 3 sessions with no eye-tracking data (795625712, 805989030, 832881662), aborted and
auto-rewarded trials.

Full rationale, validation and sanity checks: `CONVERSION_NOTES.md`.
