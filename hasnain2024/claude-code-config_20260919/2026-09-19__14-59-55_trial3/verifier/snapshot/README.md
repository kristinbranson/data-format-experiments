# ALM two-context licking dataset, converted for neural decoding

Converted from the electrophysiology + behaviour release of

> Hasnain, Birnbaum, Ulloa Severino, Zhang, Yartsev, Economo (2025),
> *Separating cognitive and motor processes in the behaving mouse*,
> **Nature Neuroscience 28**, 640–653. Data: Zenodo `10.5281/zenodo.13941415`.

The conversion script is [`convert_data.py`](convert_data.py); every decision made while
converting, and every validation check that was run, is documented in
[`CONVERSION_NOTES.md`](CONVERSION_NOTES.md).

---

## Dataset

Head-fixed mice performed two interleaved directional-licking tasks while activity was
recorded extracellularly from anterior lateral motor cortex (ALM) with Neuropixels 1.0 or
64-channel H2 probes, and behaviour was filmed at 400 Hz from a side and a bottom camera
(tracked with DeepLabCut).

- **Delayed-response (DR) trials** — a 1.3 s auditory tone indicates the rewarded port; the
  mouse must withhold licking through a delay epoch (0.9 s in the fixed-delay sessions,
  drawn from {0.3, 0.6, 1.2, 1.8, 2.4, 3.6} s in the randomized-delay sessions) and respond
  after an auditory go cue.
- **Water-cued (WC) trials** — interleaved in blocks of 10–25 trials in part of the
  sessions; all auditory cues are omitted and ~3 µl of water simply appears at a random
  port at a random time. There is no cue telling the mouse which context (block) it is in.

Everything is aligned to the **go cue** (`obj.bp.ev.goCue`, which holds the water-drop time
on WC trials) and spans **−2.5 s to +2.5 s** in **10 ms** bins (500 timepoints per trial).

### Key statistics
| | |
|---|---|
| Sessions | 44 (25 fixed-delay, 19 randomized-delay) |
| Mice | 14 |
| Units | 2,457 (2,312 ALM, 145 tjM1); 443 well-isolated single units |
| Units / session | 55.8 mean (17–141) |
| Trials | 13,762 (193–474 per session) |
| Timepoints / trial | 500 (10 ms bins, −2.5…+2.5 s) |
| Neural signal | causally smoothed firing rate, spikes/s |
| Correct / incorrect / ignore | 0.749 / 0.120 / 0.131 |
| Left / right / no lick | 0.423 / 0.446 / 0.131 |
| DR / WC trials | 0.903 / 0.097 |

### Curation
- **Sessions**: exactly the sessions and probes listed in the authors'
  `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` files (their commented-out
  sessions are excluded). The two behaviour-only optogenetic directories of the release are
  not used because they contain no neural data.
- **Units**: clusters labelled `garbage`/`gabrga`/`noisy`/`real?` are dropped
  (`findClusters.m`), then units whose mean firing rate over the analysis window across all
  trials is ≤ 1 Hz are dropped (`removeLowFRClusters.m`, `params.lowFR = 1`; the paper:
  "All units with firing rates exceeding 1 Hz were included").
- **Trials**: early-lick trials (`bp.early`) and photoinactivation trials
  (`bp.stim.enable`) are excluded, as in every reference condition string; ignore trials are
  kept because they carry the `none`/`ignore` output classes. 64 trailing trials in two
  sessions whose ephys recording had already stopped (no spikes at all) are excluded.

---

## Loading and using the data

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 0
X = data['neural'][session][trial]   # (n_units, 500)  float32, spikes/s
u = data['input'][session][trial]    # (1, 500)        float32, time from go cue (s)
y = data['output'][session][trial]   # (6, 500)        int8, class labels

data['subjects'][data['subject_idx'][session]]                       # mouse name
[data['brain_regions'][i] for i in data['brain_region_idx'][session]]  # region per unit
data['metadata']['session_info'][session]                            # per-session details
```

Validate / train the reference decoder with:

```bash
python train_decoder.py converted_data.pkl --verify-only   # format + summary
python train_decoder.py converted_data.pkl --plot-samples  # train and evaluate
```

Re-create the dataset from the raw `.mat` files (≈2 minutes for all 44 sessions):

```bash
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing   # 2 sessions + figures
```

---

## Format specification

```
data = {
  'neural':  [session][trial] -> float32 (n_units, 500)   firing rate (spikes/s)
  'input':   [session][trial] -> float32 (1, 500)         time from go cue (s)
  'output':  [session][trial] -> int8    (6, 500)         class labels (see below)
  'subjects':          list[str], 14 mouse names
  'subject_idx':       int64 (44,)  index into 'subjects' for each session
  'brain_regions':     ['ALM', 'tjM1']
  'brain_region_idx':  [session] -> int64 (n_units,)  index into 'brain_regions'
  'input_names':       ['time_from_go_cue_s']
  'output_names':      [...]  see table
  'output_values':     [...]  class names for each output
  'metadata':          dict, see below
}
```

### Inputs
| # | name | description |
|---|------|-------------|
| 0 | `time_from_go_cue_s` | bin-centre time relative to the go cue, −2.495 … +2.495 s |

### Outputs
| # | name | 0 | 1 | 2 | how it is defined |
|---|------|---|---|---|-------------------|
| 0 | `lick_direction` | left | right | none | the port the animal licked: `(R & hit) | (L & miss)` → right, `(L & hit) | (R & miss)` → left, `bp.no` (ignore) → none. Constant within a trial. Verified to agree with the side of the first lickport contact after the go cue on 99.9 % of all trials. |
| 1 | `context` | WC | DR | – | `bp.autowater` (1 = water-cued). Constant within a trial. |
| 2 | `outcome` | incorrect | correct | ignore | `bp.miss` / `bp.hit` / `bp.no`. Constant within a trial. |
| 3 | `tongue_velocity` | < median | ≥ median | not visible | speed of the side-camera `tongue` DeepLabCut feature, interpolated onto the neural time axis, split at the session's 50th percentile over the timepoints where DeepLabCut tracked the tongue. Time-varying. |
| 4 | `paw_velocity` | < median | ≥ median | not visible | same, for the bottom-camera `top_paw` feature. Time-varying. |
| 5 | `motion_energy` | < median | ≥ median | no video | whole-frame motion energy (99th percentile of the frame-differenced pixels, 400 Hz) interpolated onto the neural time axis; class 2 where no video frame covers the timepoint. Time-varying. |

### Metadata
`task_description`, `time_bin_size` (10.0 ms), `temporal_alignment_event`,
`off_start` (−2.5), `off_end` (+2.5), `neural_units`, `neuron_curation`, `trial_curation`,
`video`, `output_discretisation`, `source`, and `session_info` — a list of 44 dicts with,
for each session: animal, date, task, probes and their recorded locations, file format,
trial and unit counts before/after each curation step, number of single units, the
video/ephys clock offset, the median sample and delay durations, the three discretisation
thresholds, the output class fractions, and the indices of the retained trials and clusters
in the original `.mat` object (provenance).

---

## How the conversion was validated

- Every processing step is a port of the authors' MATLAB code (`alignSpikes.m`, `getSeq.m`,
  `mySmooth.m`, `removeLowFRClusters.m`, `findPosition.m`, `findVelocity.m`,
  `loadMotionEnergy.m`, `findVideoOffset.m`), verified numerically against independent
  re-implementations that read the raw `.mat` files directly (`cache/sanity_checks.py`,
  0 failures).
- Behavioural statistics match the Methods: 1.3 s sample epoch and 0.9 s delay in all fixed-
  delay sessions, the six randomized delay durations, 84.8 % mean DR performance (training
  criterion > 70 %), median response latency 186 ms ("typically within 300 ms"), WC blocks of
  ~11 trials beginning after ~100 DR trials.
- Re-running the paper's own decoding analyses on the converted data reproduces the
  published accuracies: choice (left vs right) decoding rises from chance in the ITI,
  inflects exactly at the sample-tone onset, reaches 0.85 in the late delay and peaks at
  **0.95** after the go cue (paper: ≈0.95); context decoding is ≈0.8 throughout the trial
  and peaks at **0.93** (paper: ≈0.9–0.95). See `cache/reproduce_fig3b_4b.png`.
- The shared decoder in `train_decoder.py` reaches, on held-out trials:
  lick_direction 0.62, context 0.86, outcome 0.61, tongue_velocity 0.57, paw_velocity 0.54,
  motion_energy 0.72 (chance 0.33, except context 0.5), with no train/validation gap.
