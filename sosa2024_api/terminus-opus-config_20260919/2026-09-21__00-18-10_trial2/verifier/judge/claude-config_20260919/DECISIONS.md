# Decisions

Documentation of the decisions made by the agentic AI in `/app/convert_data.py`, with
justifications taken from `/app/CONVERSION_NOTES.md` and the agent trajectory
(`/logs/agent/trajectory.json`).

Dataset: Sosa, Plitt & Giocomo, "A flexible hippocampal population code for experience
relative to reward" (DANDI 001361), 152 NWB sessions, 11 mice, 2-photon CA1 imaging in VR.

---

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. **Decisions.** Every `*.nwb` file under every `sub-*` directory of `/app/data` is globbed and
sorted, giving 152 sessions across 11 mice; there is no session-level exclusion (all 152 are
converted). Each file is opened once with `pynwb.NWBHDF5IO` (no `h5py`), and everything needed is
read inside that one context: the `behavior/BehavioralTimeSeries` series (`position`, `speed`,
`lick`, `reward_zone`, `environment`, `trial_start`, `teleport`, `Reward`), the `ophys`
`Fluorescence` and `Neuropil` ROI response series for every imaging plane, and the
`ImageSegmentation/PlaneSegmentation` table (`iscell`, `planeIdx`). Subject id comes from
`nwb.subject.subject_id` and the VR scene/date come from `nwb.identifier`. Sessions are processed
in a 12-worker `multiprocessing` pool, one session per worker task
(`maxtasksperchild=1`), and a per-session `try/except` records a failure rather than aborting
the run (no session failed).

ii. **Code.**
```python
files = sorted(glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
with ctx.Pool(args.nproc, maxtasksperchild=1) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs)):
```
```python
def load_session(fn):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(fn, 'r', load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        scene = nwb.identifier.split('/')[-1]
        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        t = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
        ...
        oph = nwb.processing['ophys'].data_interfaces
        Fseries = oph['Fluorescence'].roi_response_series
        Nseries = oph['Neuropil'].roi_response_series
        plane_keys = sorted(Fseries.keys())
        nframes = len(t)
        F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T
                            for k in plane_keys], axis=0)
        ...
        ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
        iscell = np.asarray(ps['iscell'].data[:])[:, 0] == 1
```

iii. **Justification.** Step 2 of CONVERSION_NOTES documents an exhaustive inventory of the release:
"One NWB file per session ... (152 files, 87 GB)", 11 subject directories, and an enumeration of
every behavioural and ophys field, which the agent cross-checked against the paper (11 switch mice,
80.5 ± 7.4 trials/session). The agent explicitly verified that the total trial count (12,216) and
subject count (11) match the paper, concluding nothing is being missed. The instruction to use
`pynwb` is followed literally. Parallel loading was chosen after timing a sample session
("NWB load 0.5–1.2 s ... dF/F + OASIS 3.3–5.8 s"), yielding a 3.2 min full conversion.

---

## 1-b. How are the data split into subjects?

i. **Decisions.** Subjects are the `subject_id` recorded inside each NWB file (`m3`, `m4`, `m7`,
`m11`…`m19`). The unique set over all successfully converted sessions is sorted numerically by the
integer after the `m`, and `subject_idx` is the index of each session's subject in that list. The
directory names `sub-m*` are used only to find files, not to define the subject list. Result:
11 subjects, 12 sessions for m11 and 14 for every other mouse.

ii. **Code.**
```python
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. **Justification.** CONVERSION_NOTES Step 2 records the 11 subject directories and Step 9's
consistency table checks "Subjects | 11 switch mice (paper) | 11 subject folders | 11 | YES", and
"m11 starts on day 3", which the methods confirm. Taking the id from inside the file rather than
from the path makes the mapping authoritative and guarantees `subject_idx` is aligned with the
session ordering actually written into `data['neural']`.

---

## 1-c. How are the data split into sessions?

i. **Decisions.** One session = one NWB file = one experiment day (`sub-<mouse>_ses-<day>`). All
152 files are kept as separate sessions; sessions are never merged across days or mice and no
cross-day ROI alignment is attempted. Sessions are appended in sorted file order, so
`neural`/`input`/`output`/`subject_idx`/`brain_region_idx` are all in the same order.

ii. **Code.**
```python
files = sorted(glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
data = {'neural': [r['neural'] for r in results],
        'input':  [r['input']  for r in results],
        'output': [r['output'] for r in results],
        ...
        'brain_region_idx': [np.zeros(r['n_neurons'], dtype=np.int64) for r in results]}
```

iii. **Justification.** Step 2: "`ses-NN` = experiment day (1–14); m11 has days 3–14 only
(12 sessions; imaging started day 3 due to low expression — consistent with methods)". The
reference pipeline likewise treats one animal-day as one `sess` object. Note that `pool.imap`
preserves input order, so the session order in the output equals the sorted file order.

---

## 1-d. How are the data split into trials?

i. **Decisions.** A trial is the half-open frame window `[trial_start, teleport)`: the frames from
track entry up to but not including the teleport frame. Both boundaries come from the binary
behaviour series `trial_start` and `teleport`. The inter-trial/teleport period is excluded from
everything (neural and behavioural). A consistency assertion requires an equal number of starts and
teleports and that every teleport follows its start. The agent deliberately used `[start, stop)`
instead of the reference's `f[:, start-1:stop-1]` indexing.

ii. **Code.**
```python
starts = np.where(trial_start > 0)[0]
teles = np.where(teleport > 0)[0]
assert len(starts) == len(teles) and np.all(teles > starts), f'bad trial indices in {fn}'
...
for i, (a, b) in enumerate(zip(starts, teles)):
    sl = slice(a, b)
    T = b - a
```

iii. **Justification.** Step 1: "Trials are defined by `trial_start_inds` (track entry) and
`teleport_inds` (track exit). Frames outside of trials (teleport period) are excluded from dF/F",
matching `sess.trial_start_inds` / `sess.teleport_inds` in the reference. Step 4 records the
off-by-one discrepancy explicitly: "the reference's -1 offset would leave the final frame of each
trial undefined ... Difference is 1 frame (64 ms) in baseline estimation only." Step 10 Check 5
verifies that starts and teleports are equal in number and strictly interleaved in all 152
sessions, and that `trial number` is constant within each derived trial.

---

## 1-e. How are trials filtered based on quality controls?

i. **Decisions.** One trial filter: the paper's lick-sensor-error rule. A trial is dropped if more
than 30% of its frames have a cumulative lick count > 2 (a stuck capacitive sensor). Exactly 81
trials out of 12,216 are removed, leaving 12,135. No minimum-trial-length filter, no speed
filter, and no session filter are applied. The dropped trials are removed entirely (neural,
inputs and outputs), not merely NaN-ed in the lick channel.

ii. **Code.**
```python
LICK_ERROR_FRAC = 0.30    # >30% of frames with cumulative lick > 2 => lick sensor error
LICK_ERROR_COUNT = 2
...
lick_err = np.array([(lick[a:b] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for a, b in zip(starts, teles)])
...
for i, (a, b) in enumerate(zip(starts, teles)):
    if lick_err[i]:
        continue
```

iii. **Justification.** Methods quote recorded in Step 3: "n = 81 out of 12,376 trials removed
across 11 switch mice ... detected by >30% of the 0.0645 s imaging frame samples in the trial
containing a cumulative lick count >2". Step 4 Check 2 reports that the agent's implementation
"flags **exactly 81 trials** across the whole dataset — identical to the paper", which it calls
"strong confirmation of the trial definition and lick variable interpretation". Step 5 decision 5
adds the reason for dropping rather than NaN-ing: "their lick output would otherwise be
corrupt/NaN". Step 10 Check 5 notes the shortest surviving trial is 96 frames (6.2 s) and that
every session still has ≥ 40 trials, well above the two-trial minimum the decoder needs.

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. **Decisions.** From the raw suite2p traces: `ophys/Fluorescence/planeN` (F) and
`ophys/Neuropil/planeN` (Fneu), pooled across planes along the neuron axis (plane order =
`sorted(plane_keys)`, which matches the ordering of the global `PlaneSegmentation` table). The
NWB's stored `ophys/Deconvolved` series is deliberately **not** used.

ii. **Code.**
```python
Fseries = oph['Fluorescence'].roi_response_series
Nseries = oph['Neuropil'].roi_response_series
plane_keys = sorted(Fseries.keys())
F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T
                    for k in plane_keys], axis=0)
Fneu = np.concatenate([np.asarray(Nseries[k].data[:nframes, :], dtype=np.float32).T
                       for k in plane_keys], axis=0)
```

iii. **Justification.** Step 4 Check 8: "the NWB Deconvolved series is suite2p `spks` derived from
raw F, not from the paper's maximin dF/F (correlation with recomputed events ~0.25–0.7).
**Decision: recompute dF/F and OASIS events following the reference code**". Step 5 decision 1
repeats this: "all reference analyses (place cells, decoding, GLM) use events deconvolved from the
maximin dF/F".

---

## 2-b. How is the `neural` data processed?

i. **Decisions.** A direct port of `reward_relative/preprocessing.py::dff(..., deconvolve=True)`.
Per session, with all non-trial frames set to NaN: subtract `0.7 × Fneu`; per trial add back
`0.7 ×` that trial's mean neuropil; compute a *maximin* baseline per trial (NaN-aware Gaussian
smoothing with sigma `[0, 15]` frames, then a 300-frame `minimum_filter1d` followed by a 300-frame
`maximum_filter1d` ≈ the Methods' 20 s window); form `dF/F = (F − F0)/|F0|`; smooth dF/F per trial
with a NaN-aware 2-frame-sigma Gaussian; deconvolve each trial with `suite2p.extraction.dcnv.oasis`
at `tau = 0.7` and `fs = rate / n_planes`. Planes are processed together (the baseline Gaussian has
sigma 0 on the cell axis, so this is per-cell and numerically equivalent to per-plane). Arrays are
float32. The emitted `neural` per trial is the (n_neurons, n_timepoints) event matrix. The agent
used the module default `keep_teleports = False` for **every** animal and day.

ii. **Code.**
```python
def compute_dff_events(F, Fneu, starts, teles, fs):
    from suite2p.extraction import dcnv
    f_ = np.full(F.shape, np.nan, dtype=np.float32); fneu_ = np.full(F.shape, np.nan, np.float32)
    for s, e in zip(starts, teles):
        f_[:, s:e] = F[:, s:e]; fneu_[:, s:e] = Fneu[:, s:e]
    nanmask = ~np.isnan(f_[0, :])
    f_ -= NEU_COEF * fneu_                                  # neuropil subtraction
    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, teles):
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        seg = nansmooth_nd(f_[:, s:e], [0, BASELINE_SMOOTH])
        seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
        seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
        flow[:, s:e] = seg
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, teles):
        dff[:, s:e] = nansmooth1d(dff[:, s:e], DFF_SMOOTH, axis=1)
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, fs)
    return dff, events
```
```python
NEU_COEF = 0.7; BASELINE_WIN = 300; BASELINE_SMOOTH = 15; DFF_SMOOTH = 2; TAU = 0.7
fs = S['rate'] / S['n_planes']
```

iii. **Justification.** Step 3 records the Methods text (maximin baseline, 20 s window, 2-sample
Gaussian, OASIS deconvolution) and Step 1 records the `dff` signature and the
`default_dff_method` constants read out of `utilities.py` (`neuropil_method 'subtract'`,
`baseline_method 'maximin'`, `neu_coef 0.7`, `keep_teleports False`). Step 10 Check 3 line (e)
claims an "identical implementation ... same constants (neu_coef 0.7, tau 0.7, fs 15.5078/plane)".
The `rate/n_planes` division is justified in Step 2: the stored `rate` is 31.0156 for the
two-plane animals (m17, m18) but each plane series still has one sample per behaviour frame.
The one documented deviation is the frame window (`[start, stop)` rather than
`[start-1, stop-1)`), justified as avoiding an undefined last frame per trial. The agent did **not**
find `teleport_metadata.py::teleport_sessions`, the per-animal/per-day table the paper's
`make_multi_anim_sess` notebook uses to set `keep_teleports = True` on the days the laser was not
blanked; it used the library default `False` everywhere.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. **Decisions.** Two filters, both the paper's. (1) ROIs are restricted to suite2p's manually
curated `iscell == 1` (from the global `PlaneSegmentation` table, whose row order matches the
plane-wise concatenation of F). (2) Putative interneurons are then excluded: any cell whose dF/F
correlates with running speed at Pearson r > 0.5, computed over all on-trial frames. The
correlation is done as a single vectorised matrix–vector product rather than a per-cell loop.
Across the dataset 402 of 138,678 cells (0.29%) are removed, leaving 138,276 neurons
(154–2,323 per session).

ii. **Code.**
```python
keep = np.where(S['iscell'])[0]
F = S['F'][keep]; Fneu = S['Fneu'][keep]
...
onmask = ~np.isnan(dff[0, :])
sp = S['speed'][onmask]; D = dff[:, onmask]
Dz = D - D.mean(axis=1, keepdims=True); spz = sp - sp.mean()
denom = (np.sqrt((Dz ** 2).sum(axis=1)) * np.sqrt((spz ** 2).sum()))
corr_speed = np.where(denom > 0, (Dz @ spz) / np.maximum(denom, 1e-12), 0.0)
is_interneuron = corr_speed > INTERNEURON_R          # INTERNEURON_R = 0.5
cells = np.where(~is_interneuron)[0]
```

iii. **Justification.** Step 3 quotes the Methods: "Additional putative interneurons were detected
for exclusion ... by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's
running speed", with the paper's rate of 0.42 ± 0.85% of cells. Step 9's table compares the
achieved 0.29% against that figure and the per-session neuron range 154–2,323 against the paper's
"155–2172", with Step 10 Check 4 explaining the high end as the pooled two-plane animal m18 (the
paper's Ext. Fig. 7 treats the two planes separately) and noting the low end (154 after removing
1 interneuron from 155 `iscell`) matches the paper exactly. No place-cell selection is applied,
justified in Step 10 as "the task asks for all recorded neurons".

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **Decisions.** No resampling or shifting is needed. The NWB behaviour series are already
interpolated 1:1 onto imaging frames (the reference's `vr_align_to_2P` step was done upstream), so
slicing the event matrix with the same frame indices `[trial_start, teleport)` used for the
behaviour puts sample 0 of every trial exactly at trial start. `off_start = 0.0`, `off_end = None`
(trials are variable length). Frames of the ophys arrays beyond the behaviour length are truncated
at load time so the two streams cannot drift apart.

ii. **Code.**
```python
nframes = len(t)
F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], ...).T for k in plane_keys], axis=0)
...
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
assert not np.isnan(neu).any(), f"NaNs in neural data, trial {i} of {S['file']}"
...
'temporal_alignment_event': 'trial start (entry to the linear track at position 0 cm)',
'off_start': 0.0, 'off_end': None,
```

iii. **Justification.** Step 3: "Alignment: VR behavior already interpolated onto imaging frame
times in the NWB files (one behavior sample per imaging frame)"; Step 5 decision 2: "no rebinning
is needed and no alignment error is introduced". Step 5 decision 9 sets the alignment event and
offsets. The assertion that no emitted neural sample is NaN is a direct check that the event trace
is defined over exactly the emitted trial window; the `--show-processing` figures overlay neural
traces, position, distance-to-zone, speed and licks on a single time axis with trial-start and
teleport markers (Step 7/Step 12).

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Decisions.** The native per-plane imaging frame period is kept: 1/15.5078125 s =
**64.484 ms**, identical in every session (the two-plane sessions store `rate = 31.0156` but each
plane series is sampled once per behaviour frame, so the effective per-plane rate is the same
15.5078 Hz). No temporal rebinning, downsampling or smoothing beyond the dF/F pipeline is applied,
and no spatial binning is applied.

ii. **Code.**
```python
fs = S['rate'] / S['n_planes']
...
'time_bin_size': 1000.0 / (15.5078125),  # ms per imaging frame (per plane)
'imaging_rate_hz': 15.5078125,
```

iii. **Justification.** Step 2 verified "dt = 0.06448363 s = 15.5078 Hz, identical in every
session" and Step 5 decision 2 states that keeping the native period satisfies the format
requirement that time bins be the same size for all trials and sessions while avoiding any
resampling error. Step 9's table checks the frame period against the paper's "~15.5 Hz" /
"0.0645 s imaging frame".

---

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. **Decisions.** From the explicit `timestamps` attribute of the behaviour series — specifically
`behavior/BehavioralTimeSeries/position.timestamps`, loaded once per session as `t` and reused for
every stream (the agent verified in Step 2 that all behaviour series share the same timestamps at
one sample per imaging frame).

ii. **Code.**
```python
t = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
```

iii. **Justification.** Step 2 documents that the behaviour series are "sampled **1:1 with imaging
frames** (dt = 0.06448363 s ... identical in every session), each with explicit `timestamps`", so
any of the series gives the same time base. Using the real timestamps rather than
`frame_index × dt` preserves any jitter present in the recording.

---

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. **Decisions.** Subtract the timestamp of the trial's first frame, so every trial starts at
exactly 0 s and the value increases monotonically at ~64.5 ms per sample. Stored as float32 in
row 0 of the (4, T) input array. Observed range over the whole dataset is [0, 216.5] s.

ii. **Code.**
```python
tt = (t[sl] - t[a]).astype(np.float32)
inp = np.empty((4, T), dtype=np.float32)
inp[0] = tt
```

iii. **Justification.** Straightforward implementation of the Decoder Task spec ("Time from start
of trial in seconds, continuous, time-varying"). Step 9 flags the 216.5 s maximum and explains it:
"trials are variable length; max is a session where the mouse paused", cross-referenced in Step 10
Check 5 ("longest 3,359 frames (216 s, mouse paused) ... real behaviour and kept").

---

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. **Decisions.** No alignment step is needed: the identical frame slice `a:b` indexes both the
behaviour timestamps and the neural event matrix, because the NWB behaviour is already on the
imaging frame grid and the ophys arrays are truncated to the behaviour length at load time.

ii. **Code.**
```python
sl = slice(a, b)
tt = (t[sl] - t[a]).astype(np.float32)
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
```

iii. **Justification.** Step 5 decision 2 and Step 10 Check 3 line (a): the NWB "already contains
exactly these aligned series". Step 10 Check 2 independently re-derived the time-from-trial-start
vector for all trials of five sessions straight from the NWB and compared it to the pickle with
`np.allclose` ("0 failures out of ~165 assertions").

---

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. **Decisions.** From the `behavior/BehavioralTimeSeries/environment` series (the reference's
`morph`), where 0 = ENV1 and 1 = ENV2 (−1 marks pre-scan frames, which never fall inside a trial).

ii. **Code.**
```python
env = np.asarray(beh['environment'].data[:], dtype=np.float64)
...
morph = np.array([np.unique(env[a:b])[0] for a, b in zip(starts, teles)])
assert np.all(np.isin(morph, [0.0, 1.0])), f"unexpected environment values in {S['file']}"
```

iii. **Justification.** Step 1 maps this to `behavior.get_trial_types`'s `morph`
("`morph` = env identity (0=Env1, 1=Env2)"), and Step 2 records the −1/0/1 coding. The assertion is
itself the sanity check that no pre-scan (−1) frame leaks into a trial.

---

## 4-b. What processing is involved in computing `input` *Environment type*?

i. **Decisions.** One value per trial (the unique value of `environment` within the trial),
broadcast across all timepoints of the trial into row 1 of the (4, T) input array. No other
transformation.

ii. **Code.**
```python
inp[1] = morph[i]
```

iii. **Justification.** Step 4 Check 6: "`environment` is constant within every trial in every
session (0 = ENV1, 1 = ENV2); on `EnvX_?_to_EnvY_?` days it switches exactly at trial index 30
together with the reward zone" — so collapsing to one value per trial loses nothing, and the
`np.unique(...)[0]` would raise/mis-assert if it were ever non-constant. Step 5 decision 7
explains the broadcast: per-trial variables are repeated over time so that `input` is uniformly
(4, T).

---

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. **Decisions.** The within-session sequential index of the trial, i.e. the enumeration index of
the `(trial_start, teleport)` pairs. The stored `trial number` behaviour series is not used.
Because lick-error trials are skipped but the index is not renumbered, the emitted trial numbers
are the original indices (with gaps where a trial was dropped); range over the dataset is 0–99.

ii. **Code.**
```python
for i, (a, b) in enumerate(zip(starts, teles)):
    if lick_err[i]:
        continue
    ...
    inp[2] = i
```

iii. **Justification.** Step 1/Step 2: the NWB has no `trials` interval table, so "trial structure
must be derived from `trial_start` / `teleport`", exactly as the reference builds
`sess.trial_start_inds`. Step 10 Check 5 verified that the stored `trial number` series "is
constant within each trial and equals the trial index", so the two agree; keeping the index derived
from the same boundaries used for slicing guarantees internal consistency.

---

## 5-b. What processing is involved in computing `input` *Trial number*?

i. **Decisions.** None beyond assigning the loop index; the value is constant across all
timepoints of the trial (row 2 of the (4, T) input array), stored as float32.

ii. **Code.**
```python
inp[2] = i
```

iii. **Justification.** The Decoder Task specifies trial number as a continuous per-trial input;
Step 5 decision 7 justifies the broadcast over time. Keeping the *raw* index (rather than
renumbering after trial removal) preserves the true ordinal position of each trial within the
session, which is what the variable is meant to encode (learning/experience within the session).

---

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. **Decisions.** From the same per-trial `isreward` vector used for the reward-outcome output,
which combines two raw variables: the `behavior/Reward` TimeSeries **timestamps** (mapped onto
frame indices with `np.searchsorted`, clipped to the valid range) and the `reward_zone` behaviour
series. A trial counts as rewarded if a reward event fell inside it **and** the reward zone was
entered — the reference's `behavior.get_trial_types` definition.

ii. **Code.**
```python
reward_t = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
...
reward = np.zeros(nframes, dtype=np.float64)
if len(reward_t):
    ridx = np.searchsorted(t, reward_t)
    ridx = np.clip(ridx, 0, nframes - 1)
    reward[ridx] = 1.0
...
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
```

iii. **Justification.** Step 1 records `get_trial_types`: "`isreward` = reward delivered AND rzone
entered". Step 2 notes that `Reward` is stored as event timestamps plus volume, not as a per-frame
series, hence the `searchsorted` mapping onto the frame grid.

---

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. **Decisions.** For trial *i* > 0 the input is `isreward[i-1]` (using the raw trial index, so the
"previous trial" is the true preceding lap even if it was dropped as a lick-sensor error),
broadcast across the trial's timepoints. For the **first** trial of a session, which has no
predecessor, the value is set to **1 (rewarded)**.

ii. **Code.**
```python
inp[3] = 1.0 if i == 0 else float(isreward[i - 1])
```

iii. **Justification.** Step 5 decision 8: "**First trial's previous outcome = 1 (rewarded)**: the
imaging session is always preceded by 30 warm-up trials with the same reward zone; treating the
unobserved previous trial as rewarded matches the modal outcome (~85%)." This is grounded in the
Methods text quoted in Step 3 ("Before the imaging session, mice were provided 30 'warm-up'
trials using the task and reward zone from the previous day"). Step 10 Check 5 restates it as a
deliberately handled edge case.

---

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. **Decisions.** Two ingredients: the `behavior/position` series, and the per-trial active reward
zone. The reward zone is obtained **from the VR scene name** stored in `nwb.identifier` (e.g.
`Env1_LocationB_to_A`), a port of `behavior.get_reward_zones`: fixed-zone scenes keep one zone all
session, switch scenes use the first zone for the first 30 trials and the second thereafter. Zone
coordinates are the paper's A = 80–130, B = 200–250, C = 320–370 cm. This label is
cross-validated per trial against an **empirical** estimate — the animal's position at the first
frame where `reward_zone > 0`, matched to the nearest zone start — and if the two ever disagree the
script prints a warning and falls back to the empirically observed zone.

ii. **Code.**
```python
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
CHANGE_TRIAL = 30

def scene_reward_zones(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.search(r'^Env\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.search(r'([ABC])_to_(?:Env\d_)?(?:Location)?([ABC])$', scene)
    if m:
        n0 = min(change_trial, ntrials)
        return np.array([m.group(1)] * n0 + [m.group(2)] * (ntrials - n0))
    raise NotImplementedError(f'Reward zones not defined for scene {scene}')

def empirical_reward_zones(pos, rzone, starts, teles):
    starts_cm = np.array([REWARD_ZONES[z][0] for z in 'ABC'])
    labels = []
    for a, b in zip(starts, teles):
        idx = np.where(rzone[a:b] > 0)[0]
        labels.append('?' if len(idx) == 0 else 'ABC'[int(np.argmin(np.abs(starts_cm - pos[a:b][idx[0]])))])
    return np.array(labels)
```
```python
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
emp = empirical_reward_zones(pos, rzone, starts, teles)
valid = emp != '?'
zone_agree = float(np.mean(emp[valid] == zone_labels[valid])) if valid.any() else np.nan
if zone_agree < 1.0:
    print(f"  WARNING ... using observed zones")
    fixed = zone_labels.copy(); fixed[valid] = emp[valid]; zone_labels = fixed
```

iii. **Justification.** Step 1 documents `get_reward_zones` including the X/Y/Z → A/B/C dictionary
and `change_trial = 30`; Step 3 quotes the Methods for the zone coordinates and "Each switch
occurred after 30 trials". Step 4 Check 4 is the validation: "This empirical zone agrees with the
scene-name rule ... for **100% of trials in all 152 sessions**; maximum deviation of first-entry
position from the nominal zone start is 8.5 cm (1 frame of running at ~80 cm/s). Confirms both the
zone dictionary and the change_trial = 30 convention." Step 9 reports the resulting balanced usage
A/B/C = 0.332/0.336/0.333.

---

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. **Decisions.** Per timepoint, the signed linear distance in cm from the animal's position to the
nearest edge of the active 50 cm zone: negative before the zone, exactly 0 anywhere inside it,
positive after it. A linear (not circular) distance is used.

ii. **Code.**
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
...
z0, z1 = REWARD_ZONES[zl]
d = signed_distance_to_zone(pos[sl], z0, z1)
out[0] = discretize_distance(d)
```

iii. **Justification.** Step 5's mapping table ties this to the reference's
`glmUtils.create_design_matrix` relative-position predictor and `behavior.get_reward_zones`, with
the note "linear (not circular) distance, per Decoder Task spec" — the paper's Fig. 3 decoder uses
a circular reward-relative coordinate, but the Decoder Task's bin edges (−50 … +50 cm) are linear.
Step 7's plot review confirms "the distance bin is 3 exactly while the animal is inside the shaded
zone".

---

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. **Decisions.** Seven categories via explicit boolean masks, defaulting to class 3 (in-zone,
d = 0): `d < −50` → 0; `−50 ≤ d < −10` → 1; `−10 ≤ d < 0` → 2; `d == 0` → 3; `0 < d ≤ 10` → 4;
`10 < d ≤ 50` → 5; `d > 50` → 6. Stored as int8. Resulting distribution over the full dataset:
[0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243].

ii. **Code.**
```python
def discretize_distance(d):
    out = np.full(d.shape, 3, dtype=np.int8)        # 3 == inside the zone (d == 0)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. **Justification.** The edges are copied verbatim from the Decoder Task specification, with
class 3 reserved for exactly 0 (the in-zone case) as the spec requires. Step 5's planned sanity
check "dist_to_reward_zone == 0 exactly when position is inside the active zone; bin 3 occupancy >
50/450" is verified in Step 9: "23.8% of samples in zone (animals slow down/consume in zone)",
against the 11% expected from track geometry alone.

---

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. **Decisions.** No separate alignment: `pos[sl]` uses the same `slice(a, b)` as the neural event
matrix, and position is on the imaging frame grid.

ii. **Code.**
```python
sl = slice(a, b)
d = signed_distance_to_zone(pos[sl], z0, z1)
neu = events[np.ix_(cells, np.arange(a, b))]
```

iii. **Justification.** Same as 3-c: the behaviour is pre-aligned to imaging frames in the NWB. The
`--show-processing` figures plot position, the shaded active zone, the raw distance and its
discretisation on the same time axis as the dF/F and event traces with trial boundaries marked
(Step 7 review: "rewards occur inside the zone on rewarded trials only"). Step 10 Check 2
recomputed this output from the raw NWB for 3 trials in each of 5 sessions with `np.allclose`.

---

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. **Decisions.** Directly from the `behavior/position` series (cm along the 450 cm virtual track).

ii. **Code.**
```python
pos = np.asarray(beh['position'].data[:], dtype=np.float64)
...
out[1] = discretize_position(pos[sl])
```

iii. **Justification.** Step 2 identifies `position` as the VR track position in cm (with −500 used
before scan start, and values below 0 only during the teleport period, which is excluded from
trials).

---

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. **Decisions.** No transformation other than clipping the raw value into `[0, 450)` and
discretising. Clipping absorbs the handful of samples that fall marginally outside the nominal
track at the start/end of a lap.

ii. **Code.**
```python
TRACK_LENGTH = 450.0

def discretize_position(pos):
    p = np.clip(pos, 0.0, TRACK_LENGTH - 1e-9)
    return np.floor(p / 90.0).astype(np.int8)
```

iii. **Justification.** Step 3 quotes the Methods' "450 cm virtual linear track"; Step 5's mapping
table says "positions clipped to [0,450]". The clip makes the first and last bins effectively
open-ended so that out-of-range samples join the nearest end bin rather than creating spurious
classes, and the `1e-9` guard prevents position == 450.0 producing a sixth bin.

---

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. **Decisions.** Five equal 90 cm bins spanning the 450 cm track, computed as
`floor(clip(pos, 0, 450) / 90)`: 0 = <90, 1 = 90–180, 2 = 180–270, 3 = 270–360, 4 = ≥360 cm.
Resulting distribution over the full dataset: [0.212, 0.177, 0.231, 0.226, 0.154].

ii. **Code.**
```python
return np.floor(p / 90.0).astype(np.int8)
...
OUTPUT_VALUES[1] = ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm']
```

iii. **Justification.** The Decoder Task specifies "5 equal-sized bins spanning the 450 cm track"
with exactly these edges. Step 5's planned check "Position bins approximately monotonic in time
within a trial (unidirectional track)" is confirmed by the trial × time raster in
`processing_*_trials.png`; Step 9 notes the slightly lower occupancy of the last bin because
"360–450 incl. fast running".

---

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. **Decisions.** Same frame slice as the neural data; no additional alignment.

ii. **Code.**
```python
out[1] = discretize_position(pos[sl])   # sl = slice(a, b), the same window as `neu`
```

iii. **Justification.** As in 3-c/7-d — behaviour and imaging share one frame grid in the NWB, and
the identity was re-verified from the raw files in Step 10 Check 2.

---

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. **Decisions.** From the `behavior/lick` series, which stores a cumulative lick count per imaging
frame (values can exceed 1).

ii. **Code.**
```python
lick = np.asarray(beh['lick'].data[:], dtype=np.float64)
```

iii. **Justification.** Step 2 documents the variable as "`lick` (cumulative count per frame)";
Step 1 records that the reference's `behavior.lickrate` / `glmUtils` binarise it the same way.

---

## 9-b. What processing is involved in computing `output` *Lick*?

i. **Decisions.** Binarised per frame: any positive count → 1, else 0. No smoothing, no conversion
to a rate, no spatial binning. Separately, whole trials whose lick channel is corrupted by the
stuck-sensor rule (>30% of frames with cumulative count > 2) are dropped from the dataset (see
1-e), so no trial contributes a known-bad lick trace. Resulting distribution: [0.777, 0.223].

ii. **Code.**
```python
out[3] = (lick[sl] > 0).astype(np.int8)
```
```python
lick_err = np.array([(lick[a:b] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for a, b in zip(starts, teles)])
```

iii. **Justification.** Step 1: "licks binarized (`licks>0 -> 1`)" in `behavior.lickrate` /
`glmUtils`; Step 3 quotes the Methods, "Remaining lick counts were converted to a binary vector".
The Decoder Task requires a binary 0/1 output, so the reference's subsequent smoothing and
spatial binning into a lick *rate* is deliberately not applied. Step 5's planned check "Lick
fraction ~2–10% of frames" was exceeded (22.3%), which the agent attributes to counting every
frame of a lick bout rather than a rate.

---

## 9-c. How is `output` *Lick* aligned with the neural data?

i. **Decisions.** Same frame slice as the neural data; no additional alignment.

ii. **Code.**
```python
out[3] = (lick[sl] > 0).astype(np.int8)
```

iii. **Justification.** Same as 3-c. The processing figure's bottom panel overlays raw cumulative
licks, the binarised output and reward delivery on the shared time axis, which the Step 7 review
used to confirm licks cluster in/just before the shaded reward zone.

---

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. **Decisions.** Identical source to 7-a: the VR scene string in `nwb.identifier` (parsed by
`scene_reward_zones`, switch at trial index 30), cross-checked per trial against the animal's
position at the first `reward_zone > 0` frame, with fallback to the empirical label on disagreement.

ii. **Code.** See 7-a (`scene_reward_zones`, `empirical_reward_zones`, the `zone_agree` check).

iii. **Justification.** See 7-a. Step 4 Check 4: 100% agreement between the scene-derived and
empirically observed zone across all 152 sessions; Step 10 Check 5 records that no fallback warning
was ever emitted (`zone_agreement == 1.0` for every session).

---

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. **Decisions.** The per-trial letter label is mapped A → 0, B → 1, C → 2 and broadcast across all
timepoints of the trial (row 4 of the (6, T) int8 output array). `output_values[4]` records the
zone coordinates alongside the letters.

ii. **Code.**
```python
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
...
out[4] = ZONE_TO_IDX[zl]
...
OUTPUT_VALUES[4] = ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)']
```

iii. **Justification.** The Decoder Task specifies "Reward zone location, per-trial. 0 = A, 1 = B,
2 = C"; Step 5 decision 7 justifies broadcasting per-trial variables over time. Step 9 verifies the
resulting class balance (0.332 / 0.336 / 0.333), consistent with the counterbalanced design.

---

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. **Decisions.** From the `behavior/Reward` TimeSeries **timestamps** (reward delivery events,
mapped onto frame indices) combined with the `reward_zone` behaviour series, exactly as in 6-a.

ii. **Code.**
```python
reward_t = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
ridx = np.clip(np.searchsorted(t, reward_t), 0, nframes - 1)
reward = np.zeros(nframes); reward[ridx] = 1.0
```

iii. **Justification.** Step 2: `Reward` carries "**timestamps** of reward delivery, data = volume
mL", so it must be discretised onto the frame grid. Step 1 records the reference definition of a
rewarded trial from `get_trial_types`.

---

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. **Decisions.** Per trial: 1 if at least one reward frame falls inside the trial **and** the
reward zone was entered during the trial, else 0 — the reference's `isreward`. The per-trial value
is broadcast across all timepoints (row 5 of the (6, T) array). Result: 84.4% rewarded / 15.6%
omitted.

ii. **Code.**
```python
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
...
out[5] = int(isreward[i])
```

iii. **Justification.** Step 1 documents `get_trial_types`: "`isreward` = reward delivered AND rzone
entered", and Step 10 Check 3 line (g) claims the "same definitions". The rate is validated in
Step 9 against the paper: "Reward omission rate | ~15% (paper) | 15.3% (per-session mean in the
data) | 15.6% of trials | YES". Step 12 Check 1 further probed this output with an independent
per-session logistic regression to confirm the modest decoding accuracy is biological rather than a
conversion bug ("frames before the zone give chance (0.49–0.53) ... rules out label leakage").

---

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. **Decisions.** Several defensive measures:
- **Ophys/behaviour length mismatch**: the ophys arrays are sliced to the behaviour length at read
  time (`data[:nframes, :]`). Ten m17/m18 sessions have one extra trailing imaging frame, which
  falls after the last teleport and is discarded.
- **Trial-boundary integrity**: an assertion requires equal numbers of `trial_start` and `teleport`
  events and that each teleport follows its start.
- **Reward timestamps off the frame grid**: mapped with `np.searchsorted` and `np.clip`-ed into
  range so a reward at the very end of a recording cannot index out of bounds; the `if len(reward_t)`
  guard handles a session with no rewards.
- **Corrupt lick channel**: the 81 stuck-sensor trials are dropped entirely rather than emitted
  with bad labels.
- **Reward-zone label ambiguity**: trials where `reward_zone` never fires are marked `'?'` and
  excluded from the agreement statistic; if the scene rule ever disagreed with the observed zone,
  the script warns and uses the observed zone.
- **Undefined neural samples**: an assertion rejects any NaN in an emitted trial's event matrix.
- **Degenerate correlations**: the interneuron correlation guards against a zero denominator.
- **Session-level failure isolation**: each session runs inside `try/except` in the worker; a
  failing session is reported and skipped instead of killing the 152-session run (no session failed).
No minimum-trial-length or minimum-neuron filter is applied (min observed: 96 frames, 154 neurons).

ii. **Code.**
```python
nframes = len(t)
F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T ...])
assert len(starts) == len(teles) and np.all(teles > starts), f'bad trial indices in {fn}'
if len(reward_t):
    ridx = np.clip(np.searchsorted(t, reward_t), 0, nframes - 1)
    reward[ridx] = 1.0
corr_speed = np.where(denom > 0, (Dz @ spz) / np.maximum(denom, 1e-12), 0.0)
assert not np.isnan(neu).any(), f"NaNs in neural data, trial {i} of {S['file']}"
```
```python
def _worker(args):
    fn, show, plot_dir = args
    try:
        res = process_session(fn, show_processing=show, plot_dir=plot_dir)
    except Exception as e:                      # keep going, report at the end
        traceback.print_exc()
        return dict(error=str(e), file=os.path.basename(fn))
    return res
```

iii. **Justification.** Step 5 decision 10 and Step 10 Check 5 document the trailing-frame issue and
its resolution; Step 10 Check 5 also records the systematic edge-case scan over all 152 sessions
(`/app/cache/edge_scan.py`): equal and strictly interleaved trial boundaries, `scanning == 1` on
every on-track frame, `trial number` constant within trials, and "no NaNs in any behaviour series".
The short/long trial extremes (96 frames; 3,359 frames) were inspected and judged "real behaviour
and are kept", and the three small m4 sessions were checked to still exceed the two-trial minimum.

---

## 13-a. What are the most time-consuming steps of the code?

i. **Decisions / findings.** The script instruments itself: `t_load` (NWB read) and `t_dff`
(dF/F + OASIS) are timed per session and printed with a running ETA. Measured on the full run:
dF/F + OASIS deconvolution dominates at ~3–8.5 s per session, NWB loading is ~0.5–2.1 s per session
(I/O bound, scaling with ROI count), and the final `pickle.dump` of the 9.52 GB dictionary takes
12.9 s. With a 12-process pool the whole 152-session conversion takes 3.2 min.

ii. **Code.**
```python
t0 = time.time(); S = load_session(fn); t_load = time.time() - t0
t0 = time.time(); dff, events = compute_dff_events(F, Fneu, starts, teles, fs); t_dff = time.time() - t0
...
print(f"[{i+1}/{len(files)}] ... t_load={inf['t_load']:.1f}s t_dff={inf['t_dff']:.1f}s "
      f"| elapsed {el/60:.1f} min, eta {el/(i+1)*(len(files)-i-1)/60:.1f} min")
```

iii. **Justification.** Step 6 ("Print timing information to find bottlenecks") and Step 7's run-time
table: "NWB load 0.5–1.2 s (scales with #ROIs) ... dF/F + OASIS 3.3 s (155 cells) – 5.8 s (1022
cells) ... Total (12 workers) ~7 s/session → ~3–5 min for 152 sessions (well under 15 min)". The
prediction matched the observed 3.2 min.

---

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. **Decisions / findings.** The remaining Python-level loops are all over trials (~80 per session),
never over neurons or timepoints:
- the three per-trial loops inside `compute_dff_events` (window masking, maximin baseline, smoothing
  + OASIS) — these mirror the reference's structure and cannot be fully vectorised because each
  trial has a different length and the filters must not cross trial boundaries;
- the per-trial list comprehensions for `isreward`, `morph` and `lick_err` (these could be replaced
  by `np.add.reduceat`-style segment reductions);
- the main per-trial assembly loop, which is inherently ragged because trials have different
  lengths.
The agent explicitly vectorised the one loop that was over neurons: the interneuron speed
correlation is a single matrix–vector product instead of the reference's per-cell
`np.corrcoef` loop.

ii. **Code.**
```python
# vectorised replacement for the reference's per-cell np.corrcoef loop
Dz = D - D.mean(axis=1, keepdims=True); spz = sp - sp.mean()
denom = (np.sqrt((Dz ** 2).sum(axis=1)) * np.sqrt((spz ** 2).sum()))
corr_speed = np.where(denom > 0, (Dz @ spz) / np.maximum(denom, 1e-12), 0.0)
```

iii. **Justification.** Step 6: "per-trial loops are only over trials (~80), all cell operations are
vectorised; interneuron correlation is a single matrix-vector product instead of a per-cell loop",
listed under "Code speedups added" together with float32 arrays and `np.searchsorted` for the
reward frame indices.

---

## 13-c. What processing does the code repeat multiple times?

i. **Decisions / findings.** Each NWB file is opened and read exactly **once**; there is no separate
survey/statistics pass, so no data is loaded twice. What is repeated within a session is minor:
the per-trial iteration over `(starts, teles)` occurs in several separate loops (three inside
`compute_dff_events`, plus the `isreward` / `morph` / `lick_err` comprehensions and the assembly
loop) rather than in one fused pass; the reward-zone label is computed twice by two independent
methods (scene-name rule and empirical entry position) as a deliberate cross-check; and in
`--show-processing` mode `np.where(S['iscell'])[0]` is recomputed inside the plotting loop.
`F`/`Fneu` are read for all ROIs and only then subset to `iscell`, so ~60% of the fluorescence read
from disk is discarded.

ii. **Code.**
```python
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)   # rule-based
emp = empirical_reward_zones(pos, rzone, starts, teles)      # independent re-derivation
zone_agree = float(np.mean(emp[valid] == zone_labels[valid])) if valid.any() else np.nan
```

iii. **Justification.** Step 6 describes the single-pass design ("Avoid unnecessary file I/O" was an
explicit instruction), and Step 5/Step 4 justify the duplicated zone computation as a validation
step: it is what produced the "100% of trials in all 152 sessions" agreement statistic and it
remains in the production script as a guard with a warning and fallback.

---

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **Decisions / findings.** Little, but not nothing:
- `planeIdx` is read from `PlaneSegmentation` and returned by `load_session` but never used (plane
  pooling is implicit in the concatenation order).
- The full-session `dff` array is computed and retained for every curated ROI; only the `events`
  array is written to the pickle. `dff` is genuinely needed for the interneuron speed correlation,
  but dF/F **and** OASIS events are computed for the ~0.29% of cells that are then discarded as
  interneurons (a two-pass design could skip their deconvolution).
- Events are computed for the whole session including the lick-error trials that are later dropped.
- `empirical_reward_zones` runs on every session even though its result is used only when the scene
  rule disagrees (which never happened).
- Session metadata (`date`, `scene`, raw ROI counts, per-session timings, `zone_labels`,
  `trial_ids`) is accumulated into `metadata['session_info']`; it is documentation rather than
  decoder input.
- `--show-processing` plotting is limited to the first two sessions, so it is not a per-session cost.
The agent's own notes do not flag any of these as wasted work.

ii. **Code.**
```python
plane_idx = np.asarray(ps['planeIdx'].data[:]).astype(int)   # returned, never used
...
return dict(..., iscell=iscell, plane_idx=plane_idx, rate=rate, n_planes=n_planes)
```
```python
dff, events = compute_dff_events(F, Fneu, starts, teles, fs)   # dff used only for the
                                                               # interneuron correlation
```

iii. **Justification.** Step 6 states the intent ("whole-session F/Fneu are read as float32 and
processed in one pass ... all cell operations are vectorised") and the measured 3.2 min full-run
time indicates the residual waste is not material. Retaining `dff` is required by the paper's
interneuron-exclusion rule, which is defined on dF/F rather than on events.
