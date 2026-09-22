# Converted Neural Decoder Dataset

This directory contains a decoder-ready conversion of the ALM electrophysiology + behavior/video dataset from the paper _Separating cognitive and motor processes in the behaving mouse_.

## Dataset Summary
- 44 sessions
- 14 subject IDs
- 2,443 ALM units after quality and low-firing-rate filtering
- 13,762 trials after excluding `stim`, `early`, and trials without neural coverage
- Neural activity aligned to `goCue`
- Common time window: `[-2.5, 2.5] s`
- Common bin size: `10 ms`

## Main Files
- `/app/converted_data.pkl`: full converted dataset
- `/app/sample_data.pkl`: 2-session sample conversion
- `/app/convert_data.py`: conversion script
- `/app/CONVERSION_NOTES.md`: detailed conversion log, validation, and review record
- `/app/train_decoder_full_out.txt`: full decoder training output

## Regeneration
Run the converter:

```bash
python3 -u /app/convert_data.py /app/converted_data.pkl --full
```

Sample mode:

```bash
python3 -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

Validate or train the decoder:

```bash
python3 -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python3 -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Format
`data` is a Python dictionary with the following top-level keys:

- `neural`: list of sessions; each session is a list of trials; each trial is `(n_neurons, n_timepoints)`
- `input`: list of sessions; each trial is `(1, n_timepoints)` and stores time from go cue
- `output`: list of sessions; each trial is `(6, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Inputs
- `time_from_go_cue_s`

## Outputs
- `lick_direction`: `0=left`, `1=right`, `2=none`
- `behavioral_context`: `0=WC`, `1=DR`
- `outcome`: `0=incorrect`, `1=correct`, `2=ignore`
- `tongue_velocity`: `0=low`, `1=high`, `2=not_visible`
- `paw_velocity`: `0=low`, `1=high`, `2=not_visible`
- `motion_energy`: `0=low`, `1=high`, `2=no_video`

## Processing Summary
- Session membership follows the released paper loader scripts, not raw folder enumeration.
- Units are restricted to the loader-selected probe(s).
- Cluster qualities `garbage`, `gabrga`, `noisy`, and `real?` are excluded.
- Units with mean firing rate `<= 1 Hz` after alignment/binning/smoothing are excluded.
- Trials are aligned to `bp.ev.goCue`.
- Saved trials satisfy `(hit | miss | no) & ~early & ~stim & neural_covered`.
- Tongue, paw, and motion-energy outputs are discretized per session by the median of visible/valid values.

## Key Metadata
- `metadata['time_bin_size'] = 10.0`
- `metadata['temporal_alignment_event'] = 'Go cue onset'`
- `metadata['off_start'] = -2.5`
- `metadata['off_end'] = 2.5`
- `metadata['session_info']` contains per-session counts, selected probes, chosen kinematic features, and discretization thresholds

## Validation Snapshot
Full decoder validation/training on the converted dataset completed successfully. Validation balanced accuracies were:

- `lick_direction`: `0.6297`
- `behavioral_context`: `0.8493`
- `outcome`: `0.6153`
- `tongue_velocity`: `0.6303`
- `paw_velocity`: `0.6297`
- `motion_energy`: `0.8452`

See `/app/CONVERSION_NOTES.md` for the detailed rationale, raw-data sanity checks, and paper/code consistency review.
