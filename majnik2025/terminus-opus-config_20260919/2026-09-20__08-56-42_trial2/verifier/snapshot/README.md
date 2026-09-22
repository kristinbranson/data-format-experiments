# Converted dataset: Majnik et al. 2025 (Track2p) - decoding motion energy from developing barrel cortex

## Dataset description

Daily two-photon calcium imaging (GCaMP8m, 30 Hz, 720x720 um FOV, layer 2/3) of the same
neurons in mouse barrel cortex (S1) across the second postnatal week (P7-P14), from
Majnik et al. 2025, eLife 14:RP107540 (Longitudinal tracking of neuronal activity from the
same cells in the developing brain using Track2p). Neurons were segmented with Suite2p and
tracked across all days of a mouse with Track2p, so the rows of the activity matrix are matched
across the sessions of a mouse. Head-fixed pups moved spontaneously on a non-motorised treadmill
in the dark; a hardware-triggered 30 Hz camera recorded their behaviour, quantified as
motion energy (sum of squared pixel-wise differences of consecutive video frames).

There is no task and no stimulus: the recordings are continuous spontaneous activity.

| Statistic | Value |
|-----------|-------|
| Subjects (mice) | 6 (jm031=A ... jm046=F) |
| Sessions | 41 (6-7 consecutive days per mouse) |
| Neurons per session | 220-746 (mean 497) |
| Neuron-sessions total | 20389 |
| Session duration | 20 min (jm031, jm032) or 30 min (others) |
| Trials | 1090 consecutive 60 s blocks (20 or 30 per session) |
| Time bin | 333.33 ms (10 imaging frames), 180 bins per trial |
| Brain region | S1 (barrel cortex, L2/3) |

## Processing

1. Neural: F.npy and Fneu.npy give dF/F = maximin baseline-corrected, neuropil-subtracted
   fluorescence (Fc = F - 0.7*Fneu; Gaussian sigma = 10 frames; 60 s minimum then maximum
   filter), exactly as in track2p/gui/data_management.py::F_processing and
   suite2p.extraction.dcnv.preprocess, with the parameters stored in the authors own ops.npy.
   The traces are then averaged in bins of 10 frames (the denoising the paper uses for all
   decoding), z-scored per neuron over the session (as in the track2p raster preprocessing),
   and cut into 60 s trials.
2. Input (time_in_session_s): the time of each bin centre measured from session onset.
3. Output (motion_energy_quintile): motion energy per imaging frame (dropped camera frames
   reinserted by linear interpolation using interframe_int.npy), averaged in bins of 10 frames,
   then discretised into 5 equal-percentile bins computed per session (exactly 20 percent of
   the samples per class in every session).

Curation: all released ROIs already satisfy Suite2p iscell probability > 0.5 and are tracked
across all days; in addition 5 neurons with an identically-zero fluorescence trace on some day
were removed from all sessions of their mouse.

## How to load

    import pickle
    with open('/app/converted_data.pkl', 'rb') as f:
        data = pickle.load(f)

    neural = data['neural'][session][trial]   # (n_neurons, 180) float32, z-scored dF/F
    inp    = data['input'][session][trial]    # (1, 180) float32, seconds since session start
    out    = data['output'][session][trial]   # (1, 180) int64, motion-energy quintile 0..4

    data['subjects'][data['subject_idx'][session]]   # mouse id of the session
    data['metadata']['session_info'][session]        # date, day index, duration, n_trials

## Output format specification

| Key | Content |
|-----|---------|
| neural | list (41 sessions) of list (20/30 trials) of (n_neurons, 180) float32 |
| input | same nesting, (1, 180) float32; input_names = [time_in_session_s] |
| output | same nesting, (1, 180) int64; output_names = [motion_energy_quintile], output_values[0] names the 5 percentile bins |
| subjects, subject_idx | 6 mouse ids; index of the mouse for each session |
| brain_regions, brain_region_idx | [S1]; zeros array of length n_neurons per session |
| metadata | task description, time_bin_size = 333.33 ms, temporal_alignment_event = start of each 60 s block, off_start = 0.0, off_end = 60.0, processing and curation notes, per-session session_info |

## Validation

- Format verification: no errors, no warnings (verification_full_out.txt).
- Value-level checks against the raw npy files (cache/sanity_checks.py): all pass.
- Shared decoder (train_decoder.py): validation balanced accuracy 0.321 vs chance 0.200.
- Replication of the paper analyses from the converted data: same-day ridge-regression R2 on
  continuous motion energy is about 0 on early days and 0.40-0.75 on the latest days, and
  corr(PC1, motion) increases with age, reproducing Fig. 7C-D of the paper.

See CONVERSION_NOTES.md for the full decision log and validation report.
