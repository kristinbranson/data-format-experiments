# Track2p Developing Barrel-Cortex Decoder Dataset

## Dataset

This directory contains a decoder-compatible conversion of the longitudinal two-photon calcium-imaging dataset from Majnik et al. (2025), *Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p*.

The data comprise layer 2/3 mouse barrel-cortex recordings from six mice over 6–7 consecutive daily sessions. Cells were filtered by Suite2p cell probability (`>0.5`) and already matched across all days within each mouse by Track2p.

## Main files

- `converted_data.pkl`: full converted dataset (41 sessions).
- `sample_data.pkl`: two-session test conversion.
- `convert_data.py`: reproducible converter.
- `CONVERSION_NOTES.md`: detailed decisions, source comparisons, checks, and decoder results.
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: complete run logs.
- `processing_*.png`: sample processing diagnostics.

## Load the data

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First session, first 60-second trial
neural = data['neural'][0][0]   # (n_neurons, 180), float32
inputs = data['input'][0][0]    # (1, 180), elapsed session seconds
output = data['output'][0][0]   # (1, 180), integer motion class 0..4
```

Every time point is 333.333 ms (3 Hz), and every trial has 180 time points (60 seconds).

## Processing summary

1. Load raw fluorescence `F`, neuropil `Fneu`, Suite2p metadata, and the supplied global motion energy.
2. Reproduce the paper/reference implementation: `Fc = F - 0.7*Fneu`; Gaussian smoothing (10-frame sigma); 60-second minimum then maximum baseline filters; subtract the resulting baseline. The reference calls this baseline-corrected signal dF/F but does not divide by baseline.
3. Repair internal missing camera samples only when local timestamp gaps exactly explain a behavior/neural length deficit. Observed behavior samples are preserved and missing values are linearly interpolated.
4. Average neural and behavior streams over the same non-overlapping 10 native frames, matching the paper's decoder preprocessing (30 Hz to 3 Hz).
5. Divide each continuous session into consecutive 60-second trials.
6. Use elapsed time from session start (bin centers) as the single time-varying decoder input.
7. Compute the 20th/40th/60th/80th motion-energy percentiles independently within each session and encode classes 0–4.

## Structure

The pickle is a dictionary with these principal fields:

- `neural[session][trial]`: neuron × time matrix.
- `input[session][trial]`: one × time elapsed-seconds matrix.
- `output[session][trial]`: one × time categorical motion-energy matrix.
- `subjects`, `subject_idx`: mouse identities and session mapping.
- `brain_regions`, `brain_region_idx`: all cells are barrel cortex layer 2/3.
- `input_names`, `output_names`, `output_values`: variable/class labels.
- `metadata`: timing, processing description, and per-session diagnostics/quantile edges.

## Key statistics

| Statistic | Value |
|---|---:|
| Subjects | 6 |
| Sessions | 41 |
| Sessions per subject | 6–7 |
| Unique tracked cells across subjects | 2,998 |
| Neurons per session | 221–746 |
| 60-second trials | 1,090 |
| Trials per session | 20 or 30 |
| Time points per trial | 180 |
| Motion classes | 5 |
| Global fraction per class | 0.200 |
| Converted file size | ~395 MiB |

The supplied release has 20-minute sessions for two subjects and 30-minute sessions for four subjects, although the paper text describes 20-minute sessions. All valid supplied data were retained.

## Reproduce and validate

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full verification reports no errors or warnings. The provided neural decoder achieved balanced accuracy 0.3118 on held-out trials (chance 0.2000); training accuracy was 0.6389.

See `CONVERSION_NOTES.md` for exact raw-file sanity checks, discrepancy resolutions, and reference-code comparisons.
