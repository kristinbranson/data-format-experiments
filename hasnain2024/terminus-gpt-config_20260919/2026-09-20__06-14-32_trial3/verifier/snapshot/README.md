# Converted ALM Neural and Behavioral Dataset

This directory contains a decoder-ready conversion of the data accompanying Hasnain, Birnbaum et al., **“Separating cognitive and motor processes in the behaving mouse”** (Nature Neuroscience, 2024).

## Main files

- `converted_data.pkl` — full converted dataset (44 sessions)
- `sample_data.pkl` — two-session test dataset
- `convert_data.py` — reproducible conversion script
- `CONVERSION_NOTES.md` — detailed decisions, source comparisons, checks, and decoder results
- `verification_full_out.txt` — full format-validation report
- `train_decoder_full_out.txt` — full decoder-training log
- `processing_*.png` — sample processing/alignment visualizations

## Reproduce the conversion

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

`--full` is the default when neither `--full` nor `--sample` is specified.

## Load the dataset

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First trial of first session
neural = data['neural'][0][0]   # neurons x 1000 time bins
inputs = data['input'][0][0]    # 1 x 1000
outputs = data['output'][0][0]  # 6 x 1000
```

## Processing summary

- **Sessions:** 44 active, available electrophysiology sessions: 25 fixed-delay/two-context sessions and 19 randomized-delay sessions.
- **Subjects:** 14 mice.
- **Brain region:** ALM.
- **Neurons:** 2,498 curated units. Author loader probe selections are followed; explicit garbage clusters are excluded; units must exceed 0.5 Hz over the reference window.
- **Trials:** 13,762 valid trials after excluding early trials, stimulation trials, invalid go cues/ephys periods, and 61 native trailing periods with no neural activity. Ignore/no-response trials are retained because they are a required target class.
- **Alignment:** trial-specific go-cue onset.
- **Window:** `[-2.5, +2.5)` seconds in 5-ms bins (1000 bins; centers −2.4975 to +2.4975 s).
- **Neural values:** single-trial firing rates in spikes/s, smoothed with the reference 15-point causal Gaussian window.
- **Video:** raw 400-Hz two-camera DeepLabCut data are aligned by frame time and go cue, then interpolated to the common 200-Hz axis.
- **Thresholds:** tongue speed, paw speed, and motion energy use one median per session, as required by the decoder task.

See `CONVERSION_NOTES.md` for detailed justification and source-code comparisons.

## Dictionary schema

- `neural[session][trial]`: `float32`, shape `(n_neurons, 1000)`
- `input[session][trial]`: `float32`, shape `(1, 1000)`
  - row 0: signed time from go cue in seconds
- `output[session][trial]`: `int8`, shape `(6, 1000)`
  1. lick direction: left, right, none
  2. behavioral context: DR, WC
  3. outcome: incorrect, correct, ignore
  4. tongue velocity: below median, at/above median, not visible
  5. paw velocity: below median, at/above median, not visible
  6. motion energy: below median, at/above median, no video
- `subjects`, `subject_idx`: subject lookup
- `brain_regions`, `brain_region_idx`: region lookup (all ALM)
- `metadata.session_info`: source session IDs, selected probes, source/included trial indices, thresholds, neuron rates, and edge-case exclusions

The first three outputs are trial-level labels repeated over time for a uniform output matrix. The final three are time-varying categorical traces.

## Full dataset statistics

| Statistic | Value |
|---|---:|
| Sessions | 44 |
| Trials | 13,762 |
| Neurons | 2,498 |
| Subjects | 14 |
| Time bins/trial | 1,000 |
| Neurons/session | 17–142 |
| Trials/session | 193–474 |

Output fractions across all timepoints:

| Output | Fractions |
|---|---|
| Lick direction | left 0.4227, right 0.4460, none 0.1313 |
| Context | DR 0.9034, WC 0.0966 |
| Outcome | incorrect 0.1197, correct 0.7490, ignore 0.1313 |
| Tongue velocity | below 0.0467, above 0.0467, not visible 0.9066 |
| Paw velocity | below 0.3662, above 0.3662, not visible 0.2676 |
| Motion energy | below 0.4987, above 0.5012, no video 0.00007 |

## Validation and decoder results

`train_decoder.py --verify-only` reports **valid format with no errors or warnings**.

Full validation balanced accuracies:

| Output | Accuracy |
|---|---:|
| Lick direction | 0.5932 |
| Behavioral context | 0.8312 |
| Outcome | 0.5836 |
| Tongue velocity | 0.5227 |
| Paw velocity | 0.5073 |
| Motion energy | 0.7044 |

All outputs exceed 1.5 times the script-reported chance level. Training loss decreased from 8.9589 to 0.7499.
