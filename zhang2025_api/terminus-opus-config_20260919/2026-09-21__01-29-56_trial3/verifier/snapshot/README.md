# IBL brain-wide map, converted for neural decoding

`/app/converted_data.pkl` (11.4 GB) contains the International Brain Laboratory (IBL)
**brain-wide map** public dataset (Nature 2023, "A brain-wide map of neural activity during
complex behaviour") reformatted for the decoder in `/app/train_decoder.py`. Processing follows
the IBL data paper and the reference code of Zhang et al., "Exploiting correlations across trials
and behavioral sessions to improve neural decoding" (`/app/code/code_zhang2025`).

## Dataset description

Mice perform a visual decision task: a Gabor patch of one of five contrasts (100, 25, 12.5, 6.25,
0%) appears on the left or right of a screen and the mouse turns a wheel to bring it to the centre.
The first 90 trials of a session are unbiased (p(left) = 0.5); afterwards the prior probability of a
left stimulus alternates between 0.2 and 0.8 in uncued blocks of 20-100 trials. Neuropixels probes
record spiking activity across the brain; side cameras give whisker-pad motion energy and a rotary
encoder gives wheel movement.

Each trial in the converted data is a **2 s window aligned to stimulus onset**, from 0.5 s before to
1.5 s after onset, binned into **100 non-overlapping 20 ms bins**.

## Key statistics

| Statistic | Value |
|---|---|
| Sessions | 441 (of the 459 released; 18 skipped, see `metadata['skipped_sessions']`) |
| Subjects (mice) | 136 |
| Trials | 187,936 (mean 426 per session, median 392, range 125-1445) |
| Neurons | 62,757 well-isolated units in grey matter (mean 142 per session, range 7-516) |
| Brain regions | 263 Beryl (Allen CCF) acronyms |
| Time bins | 100 x 20 ms per trial |
| Neural values | spike counts per bin (float32) |
| choice | 50.8% left / 49.2% right |
| prior p(left) | 41.7% 0.2, 14.0% 0.5, 44.2% 0.8 |
| wheel speed / whisker ME classes | 1/3 each (per-session tertiles) |

Validation balanced accuracy of the reference decoder: choice 0.615 (chance 0.5),
prior 0.660, wheel speed 0.607, whisker motion energy 0.598 (chance 0.333).

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]      # session 0, trial 0: (n_neurons, 100) spike counts
inputs = data['input'][0][0]       # (2, 100)
outputs = data['output'][0][0]     # (4, 100), integer class labels
subject = data['subjects'][data['subject_idx'][0]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][0]]
```

Re-create the file with:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # all sessions (~4 min, 24 workers)
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing   # 2 sessions + diagnostics
python -u /app/train_decoder.py /app/converted_data.pkl                # train/validate the decoder
```

## Output format specification

- `neural`: list over 441 sessions of lists over trials of `(n_neurons, 100)` float32 spike counts.
- `input`: same nesting, each trial `(2, 100)` float32
  1. `time_from_stim_onset` - bin centre in seconds, -0.49 ... 1.49 (negative = before onset)
  2. `trial_number_in_block` - 0-based index of the trial within its constant-`probabilityLeft`
     block, constant within a trial
- `output`: same nesting, each trial `(4, 100)` int32 class labels
  1. `choice` - 0 = left, 1 = right (`trials.choice` +1/-1)
  2. `prior_prob_left` - 0 = p(left) 0.2, 1 = 0.5, 2 = 0.8
  3. `wheel_speed` - 0/1/2 = low/medium/high (per-session tertiles of |wheel velocity|)
  4. `whisker_motion_energy` - 0/1/2 = low/medium/high (per-session tertiles)
- `subjects` (136 names), `subject_idx` (441,), `brain_regions` (263 acronyms),
  `brain_region_idx` (one int array per session), `input_names`, `output_names`, `output_values`.
- `metadata`: `task_description`, `time_bin_size` (20.0 ms), `temporal_alignment_event`
  (stimulus onset), `off_start` (-0.5), `off_end` (+1.5), `n_time_bins`, curation rules,
  discretisation description, and `session_info` with per-session eid, subject, lab, probe and
  cluster counts, trial counts before/after curation, camera used and tertile edges.

## Curation applied

- **Sessions**: the 459 eids of the public BWM release; skipped if no whisker motion energy is
  available (14), if the only available camera does not cover any curated trial (1), or if fewer
  than 5 well-isolated grey-matter units remain (3).
- **Neurons**: `clusters.label >= 1` (all three IBL RIGOR single-unit metrics passed: amplitude
  > 50 uV, noise cut-off < 20 uV, refractory-period violation), Beryl region not `root`/`void`;
  probes of a session merged.
- **Trials**: the reference `load_trials_and_mask(max_trial_len=10.0)` criteria (no NaN in
  stimOn/choice/feedback/probabilityLeft/firstMovement/feedbackType, first movement 0.08-2.0 s after
  stimulus onset, `choice != 0`, feedback-goCue <= 10 s), plus complete wheel and whisker coverage of
  the 2 s window, plus removal of trials that fall in gaps of the ephys recording.

See `/app/CONVERSION_NOTES.md` for the full decision log and all validation results.
