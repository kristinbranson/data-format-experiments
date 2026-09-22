# Allen Visual Behavior 2P → neural-decoder dataset

`converted_data.pkl` holds the Allen Brain Observatory **Visual Behavior 2-photon** dataset
(`visual-behavior-ophys-1.1.0`, change-detection task) reformatted for the decoder in
`train_decoder.py` / `decoder.py`.

## Dataset description

Head-fixed mice perform a go/no-go visual **change-detection** task while 2-photon calcium imaging is
performed in primary visual cortex (VISp). Natural images are flashed for 250 ms every 750 ms (500 ms grey
inter-stimulus interval; 5% of flashes are omitted). Mice earn water by licking within 150–750 ms of a
change in image identity. Trials are **go** (real change → hit / miss) or **catch** (sham change → false
alarm / correct reject); aborted (premature lick) and auto-rewarded (free reward) trials are excluded.

What is included:

| | |
|---|---|
| sessions | 165 (one imaging plane each, single-plane Scientifica rigs, 31 Hz) |
| mice | 37 (2–9 sessions each) |
| neurons | 28,821 (6–666 per session), all VISp |
| trials | 42,470 go/catch trials (37,143 go + 5,327 catch), 257 per session on average |
| timepoints | 11,192,974 ophys frames (217–389 per trial, mean 262 = 8.47 s) |
| time bin | 32.32 ms (the native ophys frame interval) |
| session types | OPHYS_1/3 (familiar image set A) and OPHYS_4/6 (novel image set B); passive sessions excluded |
| cre lines | Slc17a7 (106 sessions), Vip (31), Sst (28) |
| file size | 8.3 GB |

Excluded from the local copy of the release: passive sessions (no trials), the Multiscope/multi-plane
sessions (11 Hz frame rate, incompatible with a single time-bin size; 34 planes from only 6 behaviour
sessions of one mouse), and 3 sessions with no eye-tracking data. See `CONVERSION_NOTES.md` for the full
rationale behind every decision.

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][5][12]        # session 5, trial 12: (n_neurons, T) float32 dF/F
out    = data['output'][5][12]        # (5, T) int8 categorical labels
inp    = data['input'][5][12]         # (0, T) — this task has no decoder inputs

mouse  = data['subjects'][data['subject_idx'][5]]
region = data['brain_regions'][data['brain_region_idx'][5][0]]     # 'VISp'
data['metadata']['session_info'][5]   # ids, cre line, depth, bin edges, trial times, cell ids
```

Reproduce the conversion / validation:

```bash
python -u convert_data.py /app/converted_data.pkl --full          # ~1 min, 24 workers
python -u convert_data.py /app/sample_data.pkl --sample --show-processing
python -u train_decoder.py /app/converted_data.pkl --verify-only
python -u train_decoder.py /app/converted_data.pkl --plot-samples # ~13 min on one GPU
```

`convert_data.py --neural {dff,events,filtered_events} --normalize {none,zscore,noise_std}` switches the
neural signal (see `CONVERSION_NOTES.md` Step 5 decision 3 for the comparison behind the default).

## Output format specification

`data` is a dict with the keys required by `decoder.verify_data_format`:

- `neural[session][trial]`: `(n_neurons, T)` float32 — released **dF/F** traces (baseline-normalised,
  detrended by the Allen pipeline), sampled on the native ophys frame times of that trial.
- `input[session][trial]`: `(0, T)` float32 — this task specifies no decoder inputs; `input_names == []`.
- `output[session][trial]`: `(5, T)` int8, `output_names`:

  | row | name | classes | definition |
  |---|---|---|---|
  | 0 | `image_identity` | 16 image names (8 per session) | identity of the most recent presented image, held over the 750 ms image-presentation interval; omitted flashes carry the ongoing image forward, so identity changes exactly at image changes |
  | 1 | `image_change` | `no_change`, `change` | 1 for the 750 ms interval that begins at a real image change; always 0 on catch (sham-change) trials |
  | 2 | `running_speed_bin` | 5 quintiles | running speed (cm/s) interpolated onto the ophys frames and binned into per-session equal-percentile bins |
  | 3 | `pupil_diameter_bin` | 5 quintiles | pupil diameter `2√(pupil_area/π)` in px, blinks interpolated, same binning |
  | 4 | `trial_outcome` | `hit`, `miss`, `false_alarm`, `correct_reject` | one value per trial, constant over the trial |

- `subjects` (37 mouse ids), `subject_idx` (n_sessions,), `brain_regions` (`['VISp']`),
  `brain_region_idx[session]` (n_neurons,), `output_values` (class names per output).
- `metadata`: `task_description`, `time_bin_size` (32.3193 ms), `temporal_alignment_event`
  (trial start, `off_start = 0.0`, `off_end = None` because trial length varies), the neural-signal and
  discretisation definitions, dataset totals, and `session_info` — one dict per session with the
  experiment/session/container/mouse ids, session type, cre line, depth, rig, frame rate, trial counts,
  the image set, the percentile bin edges, the cell-specimen ids, and the trial start/stop/change times.

## Key statistics (validated against the Allen metadata tables and the papers)

- Per-session go / catch / hit / miss / false-alarm / correct-reject counts and neuron counts are
  **identical to the published `behavior_session_table.csv` and `ophys_cells_table.csv`** in all 165
  sessions.
- Catch rate 12.54% (whitepaper: ~12.5%), omission rate 5%, 8 images per session, mean change time
  4.28 s after trial start (whitepaper: 4.2 s), 31 Hz imaging.
- Output distributions: `image_identity` 0.060–0.065 per class; `image_change` 0.923 / 0.077;
  `running_speed_bin` and `pupil_diameter_bin` exactly 0.200 per class (in every session);
  `trial_outcome` hit 0.316, miss 0.559, false alarm 0.018, correct reject 0.107.

## Decoder performance (reference decoder, validation balanced accuracy)

| output | chance | validation |
|---|---|---|
| image_identity | 0.0625 | 0.418 |
| image_change | 0.5 | 0.607 |
| running_speed_bin | 0.2 | 0.282 |
| pupil_diameter_bin | 0.2 | 0.267 |
| trial_outcome | 0.25 | 0.295 |

Evaluated the way the source paper evaluates its decoders (first 400 ms after an image presentation),
the same trained model gets 0.689 % correct for change-vs-repeat (paper: 0.52–0.65) and 0.604 % correct
for hit-vs-miss (paper: 0.55–0.75).

## Files

| file | contents |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` / `sample_data.pkl` | full / 2-session datasets |
| `CONVERSION_NOTES.md` | every decision, check and validation result |
| `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt` | run logs |
| `processing_<experiment_id>.png` | per-step processing diagnostics |
| `sample_trials.png`, `predictions.png` | decoder sample/prediction plots |
| `cache/` | survey and validation scripts (see `cache/README_CACHE.md`) |
