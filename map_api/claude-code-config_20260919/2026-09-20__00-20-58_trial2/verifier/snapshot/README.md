# MAP brain-wide Neuropixels dataset, converted for neural decoding

`converted_data.pkl` holds the Mesoscale Activity Project (MAP) multi-regional Neuropixels
dataset re-packaged as trial-aligned firing rates plus task inputs and behavioural outputs,
ready for the decoder in `train_decoder.py`.

## Dataset

Mice performed an **auditory delayed-response (memory-guided directional licking) task**. A
pure tone (3 kHz or 12 kHz, three 150 ms pips with 100 ms gaps, 0.65 s total) during the
*sample* epoch instructs the animal to lick left or right. After a 1.2 s *delay* epoch an
auditory go cue (6 kHz, 0.1 s) opens a 1.5 s *answer* period, in which licking the instructed
port yields a water reward. Licking during the sample/delay epoch ("early lick") replays the
epoch. On ~20 % of trials one or both ALM hemispheres were photoinhibited during the delay.
Up to five Neuropixels probes recorded simultaneously across the brain, and the face was filmed
from the side at 300 Hz with DeepLabCut tracking of the tongue, jaw and nose.

- Source: **DANDI:000363** (`/app/data`, 174 NWB files), read with `pynwb`.
- Data paper: Chen, Liu et al., *Brain-wide neural activity underlying memory-guided movement*,
  **Cell** 187, 676–691 (2024).
- Method paper: Wang, Kurgyis et al., *Brain-wide analysis reveals movement encoding structured
  across and within brain areas*, **Nat Neurosci** (2025). Reference code in `/app/code`.

## Key statistics of the converted dataset

| | |
|---|---|
| Sessions | **173** (paper: 173) |
| Subjects (mice) | **28** (paper: 28) |
| Probe insertions | **655** (paper: 655) |
| Neurons (`classification == 'good'`) | **69,453** (paper: 69,943) |
| Neurons / session | mean 401.5, range 90–923 |
| Trials | **89,068** (mean 514.8/session, range 159–796) |
| Trials / session excluding early licks | mean 454.9 (paper: 476) |
| Time bins per trial | 80 (50 ms each, −2.5 … +1.5 s around the go cue) |
| Correct rate on control trials | 81.6 % (paper: 84 %) |
| Photostimulation trials | 20.0 % |
| Brain regions | 14 |
| File size | 11.8 GB |

Neurons per region: Thalamus 12,968 · Orbital 10,223 · ALM 9,367 · Striatum 7,666 ·
Midbrain 7,480 · OtherCortex 6,853 · Olfactory 4,137 · Medulla 2,925 · CorticalSubplate 1,960 ·
Hippocampus 1,944 · Cerebellum 1,823 · Pallidum 1,090 · Hypothalamus 650 · Pons 367.

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 5
fr   = data['neural'][session][trial]   # (n_neurons, 80) firing rate in Hz
inp  = data['input'][session][trial]    # (2, 80)  float32
outp = data['output'][session][trial]   # (4, 80)  int64 class labels

print(data['subjects'][data['subject_idx'][session]])            # mouse id
print([data['brain_regions'][i] for i in data['brain_region_idx'][session]][:5])
print(data['metadata']['session_info'][session])
```

Validate / train the decoder:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only   # format + summary
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples  # train + evaluate
```

Regenerate the pickle from the raw NWB files (~40 s on 16 cores):

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Output format specification

```
data = {
  'neural'          : [session][trial] -> float32 (n_neurons, 80), firing rate in Hz
  'input'           : [session][trial] -> float32 (2, 80)
  'output'          : [session][trial] -> int64   (4, 80)
  'subjects'        : list[str], 28 mouse ids
  'subject_idx'     : int64 (173,), index into 'subjects' for each session
  'brain_regions'   : list[str], 14 coarse Allen-CCF areas
  'brain_region_idx': [session] -> int64 (n_neurons,), index into 'brain_regions'
  'input_names'     : ['time_from_tone_onset', 'photostim_on']
  'output_names'    : ['choice', 'outcome', 'early_lick', 'tongue_y_position']
  'output_values'   : names of each class of each output
  'metadata'        : dict (see below)
}
```

### Inputs (per 50 ms bin)

| # | Name | Type | Description |
|---|------|------|-------------|
| 0 | `time_from_tone_onset` | continuous (s) | time from the onset of the instruction tone at the bin centre. The tone onset used is the **last** sample-epoch start before the go cue, so replays after an early lick are handled. Median tone→go interval 1.85 s = 0.65 s sample + 1.2 s delay. |
| 1 | `photostim_on` | binary | 1 if ALM photoinhibition was on during the bin (>1 ms overlap). Always ends before the go cue. |

### Outputs (per 50 ms bin; the first three are constant within a trial)

| # | Name | Classes | Description |
|---|------|---------|-------------|
| 0 | `choice` | `left`, `right`, `no lick` | lick direction, from `trial_instruction` × `outcome` (hit → instructed side, miss → opposite side, ignore → no lick). Verified to agree with the direction of the first post-go lick on 100 % of checked trials. |
| 1 | `outcome` | `ignore`, `miss`, `hit` | from the trials table: no lick / wrong port / correct port |
| 2 | `early_lick` | `no`, `yes` | the animal licked during the sample or delay epoch |
| 3 | `tongue_y_position` | `low (<40th pct)`, `mid (40–60th pct)`, `high (>60th pct)`, `not visible` | side-view DeepLabCut tongue *y*, averaged over the frames of the bin in which the tongue was visible (likelihood > 0.9), then discretised with the 40th/60th percentiles of all visible bins **of that session**. Bins with no visible frame are `not visible`. |

### Metadata

`task_description`, `temporal_alignment_event` (`'onset of the auditory go cue'`),
`off_start` (−2.5), `off_end` (+1.5), `time_bin_size` (50.0 ms), `n_time_bins` (80),
`bin_centers_s`, `neural_units`, `input_descriptions`, `output_descriptions`,
`neuron_curation`, `trial_curation`, `session_curation`, `neuron_hemisphere`
(0 = left, 1 = right, per neuron), `hemisphere_values`, `caveat`, and `session_info`
(session id, subject, source file, neuron/trial counts, tongue percentiles).
The whole metadata dict is JSON-serialisable.

## Curation applied

- **Neurons**: only `units.classification == 'good'` — the region-specific logistic-regression
  quality-control classifiers of the data paper's spike-sorting white paper. No firing-rate
  threshold.
- **Trials**: `auto_water` and `free_water` trials removed (as in the reference
  `get_regular_trial_mask`); trials not annotated good in `units.is_good_trials` removed;
  trials without ephys coverage (`units.obs_intervals`) removed; trials with no spikes at all
  removed. Early-lick, `ignore` and photostimulation trials are **kept**, because they are
  required decoder outputs/inputs.
- **Sessions**: the one session with no good units is dropped, giving exactly the 173 sessions
  and 655 insertions reported in the paper.

## Decoder performance (validation, balanced accuracy)

| Output | Chance | Whole window | Peak bin |
|---|---|---|---|
| choice | 0.333 | 0.679 | 0.86 at +0.33 s |
| outcome | 0.333 | 0.659 | 0.87 at +1.0 s |
| early_lick | 0.500 | 0.748 | 0.84 at −1.1 s (mid-delay) |
| tongue_y_position | 0.250 | 0.662 | 0.69 |

Choice decoding follows the published time course: at chance before the instruction tone,
rising through the sample and delay epochs, and peaking just after the go cue
(0.92 left-vs-right accuracy).

## Caveat

Spikes are only available inside each trial's recorded interval. On error trials the interval
ends at the error lick, so 17.5 % of trials have no spikes in the last 5 bins (and 1.8 % none in
the first 5). Following the reference preprocessing, the fixed window is binned regardless, so
those bins read 0 Hz. `outcome` is therefore partly decodable from where the zero-padding starts.

## Files

| File | Contents |
|---|---|
| `convert_data.py` | the conversion script |
| `ccf_regions.py` | Allen-CCF annotation → 14 coarse brain areas |
| `sanity_checks.py` | 101 independent checks against the raw NWB files |
| `analyze_accuracy_vs_time.py` | decoder accuracy as a function of time in the trial |
| `CONVERSION_NOTES.md` | full record of every decision, check and validation |
| `converted_data.pkl` / `sample_data.pkl` | converted dataset / 2-session sample |
| `processing_<session>.png` | per-step verification plots |
| `cache/` | intermediate exploration scripts and outputs |
