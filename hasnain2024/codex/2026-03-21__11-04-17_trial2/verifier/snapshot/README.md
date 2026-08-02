# Two-Context ALM Decoder Dataset

This directory contains a converted version of the Hasnain/Birnbaum et al. two-context ALM electrophysiology dataset from the paper *Separating cognitive and motor processes in the behaving mouse*.

The converted dataset is saved in `converted_data.pkl` and is formatted for `train_decoder.py`. Trials are aligned to the stored `bp.ev.goCue` event. In delayed-response (DR) trials this is the auditory go cue; in water-cued (WC) trials the stored field acts as the water-presentation-equivalent event used throughout the conversion.

## Included data

- Sessions: 12
- Subjects: 7 (`EKH1`, `EKH3`, `JEB6`, `JEB7`, `JGR2`, `JGR3`, `JEB19`)
- Trials: 2,415 kept trials after filtering
- Neurons: 519 ALM units after quality and firing-rate curation
- Time bin: 5 ms
- Window: `[-2.5, 2.5]` s around alignment

## Trial filtering

- Keep `hit` or `miss` trials
- Exclude `early` trials
- Exclude `no` / ignore trials
- Exclude stimulation trials
- Require a valid first post-alignment lick direction

## Neural preprocessing

- Source: electrophysiology spike times from the ALM probe selected by the paper’s Figure 8 session loaders
- Binning: 5 ms
- Representation: single-trial firing rates
- Smoothing: causal Gaussian (`smooth = 15`, `bctype = reflect`)
- Unit curation:
  - exclude quality labels `garbage`, `gabrga`, `noisy`, `real?`
  - keep units with mean firing rate `> 1 Hz` in the aligned window across all session trials

## Decoder format

`converted_data.pkl` stores a Python dictionary with these keys:

- `neural`: list of sessions, each a list of trials, each trial `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trials, each trial `(1, n_timepoints)`
- `output`: list of sessions, each a list of trials, each trial `(6, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Inputs

- `time_from_go_cue_seconds`

## Outputs

- `lick_direction`: `left=0`, `right=1`
- `behavioral_context`: `WC=0`, `DR=1`
- `outcome`: `incorrect=0`, `correct=1`
- `tongue_velocity`: session-median split
- `paw_velocity`: session-median split
- `motion_energy`: session-median split

Per-trial categorical outputs are stored as constant traces across time. Movement outputs are time-varying binary traces.

## Loading example

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["input_names"])
print(data["output_names"])
print(data["neural"][0][0].shape)
```

## Regeneration

Run the conversion:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Run format verification only:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Run full decoder training:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Notes

- Full conversion/validation details are documented in `CONVERSION_NOTES.md`.
- Review scripts and their outputs are stored in `cache/`.
