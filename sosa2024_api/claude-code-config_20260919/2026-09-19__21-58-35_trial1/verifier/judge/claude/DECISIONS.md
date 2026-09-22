# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `*.nwb` file under `/app/data/sub-*/` is globbed and sorted, giving 152 files
(11 subject directories, 14 sessions each except `m11` which has 12). Each file is one
session and is opened with `pynwb.NWBHDF5IO` in a context manager. From each file the AI
reads: the `ophys` processing module (`Fluorescence` and `Neuropil` `RoiResponseSeries`
— one per imaging plane — plus the `ImageSegmentation/PlaneSegmentation` table for
`iscell` and `planeIdx`), and the `behavior` processing module
(`BehavioralTimeSeries`: `position`, `speed`, `lick`, `environment`, `reward_zone`,
`scanning`, `trial number`, `trial_start`, `teleport`, and the separate `Reward` series
with its own timestamps). Subject id comes from `nwb.subject.subject_id`, session id
from `nwb.session_id`, and the VR scene name / date are parsed out of `nwb.identifier`.
Sessions are processed in parallel with a `ProcessPoolExecutor` (12 workers); the whole
152-session conversion runs in 60 s. All 152 sessions survive into the output.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, "sub-*", "*.nwb")))
...
with NWBHDF5IO(path, "r", load_namespaces=True) as io:
    nwb = io.read()
    subject = nwb.subject.subject_id
    session_id = nwb.session_id
    ident_parts = nwb.identifier.rstrip("/").split("/")
    scene = ident_parts[-1]
    date = ident_parts[-2]
    ophys = nwb.processing["ophys"]
    plane_seg = (ophys.data_interfaces["ImageSegmentation"]
                 .plane_segmentations["PlaneSegmentation"])
    iscell = np.asarray(plane_seg["iscell"].data)[:, 0].astype(bool)
    plane_idx_all = np.asarray(plane_seg["planeIdx"].data).astype(int)
    fluo = ophys.data_interfaces["Fluorescence"].roi_response_series
    neuro = ophys.data_interfaces["Neuropil"].roi_response_series
    ...
    bts = (nwb.processing["behavior"]
           .data_interfaces["BehavioralTimeSeries"].time_series)
    beh = {k: np.asarray(v.data[:]) for k, v in bts.items() if k != "Reward"}
    timestamps = np.asarray(bts["position"].timestamps[:])
    reward_times = np.asarray(bts["Reward"].timestamps[:])
```

iii. From CONVERSION_NOTES Step 2: "`/app/data/sub-m<N>/sub-m<N>_ses-<DD>_behavior+ophys.nwb`;
`ses-<DD>` is the 1-indexed experiment day (01–14). 11 subjects × 14 days, except `m11`
which starts at day 03 (12 sessions) → **152 sessions**, matching `dandiset.yaml`
(`numberOfFiles: 152`, `numberOfSubjects: 11`)." The AI further verified that the
`GCAMP<N>` animal ids and scene/date pairs in `nwb.identifier` match
`code/src/reward_relative/sessions_dict.py`, i.e. that the released files correspond
one-for-one to the paper's sessions. Everything is read through `pynwb` as required.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the NWB metadata field `nwb.subject.subject_id` of each file
(not from the directory name). The list of unique subjects is built in the order the
sessions are encountered and `subject_idx` records, for each session, the index into
that list. 11 subjects result: m11, m12, m13, m14, m15, m17, m18, m19, m3, m4, m7.

ii.
```python
subject = nwb.subject.subject_id       # in load_session()
...
for n, i_, o, pidx, info in results:
    ...
    if info["subject"] not in subjects:
        subjects.append(info["subject"])
    subject_idx.append(subjects.index(info["subject"]))
...
"subjects": subjects,
"subject_idx": np.array(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 4 checks the subject count against the paper: "n = 11 mice"
for the switch task, and the per-subject session counts (14 each, m11 = 12) against
"for a total of 14 days" and "imaging started on day 3 [for m11]". The verification log
confirms 11 subjects with the expected session counts.

## 1-c. How are the data split into sessions?

i. One session per NWB file — the file name / `nwb.session_id` field gives `ses-<DD>`,
which the AI identified as the 1-indexed experiment day. No cross-session neuron
alignment (the paper's multi-day ROI matching) is attempted; each session keeps its own
neuron set. All 152 sessions are kept; a session would only be dropped if it yielded
fewer than 2 usable trials (this never happens).

ii.
```python
session_id = nwb.session_id
...
info = dict(file=os.path.basename(path), subject=S["subject"],
            session_id=S["session_id"], exp_day=int(S["session_id"]), ...)
...
for n, i_, o, pidx, info in results:
    if len(n) < 2:
        print(f"  WARNING: dropping {info['file']} with {len(n)} trials")
        continue
```

iii. CONVERSION_NOTES Step 2: the 152 files match `dandiset.yaml`'s `numberOfFiles: 152`
and the per-mouse day structure in `sessions_dict.py`. The `len(n) < 2` guard implements
the instruction that "there needs to be at least two trials within each session in order
to evaluate the decoder performance".

## 1-d. How are the data split into trials?

i. A trial runs from a `trial_start` event to the following `teleport` event, i.e. the
half-open frame window `[start, end)` where `start` is the index of a non-zero
`trial_start` sample and `end` is the index of the next non-zero `teleport` sample.
The count of starts and ends is asserted to be equal and every teleport is asserted to
follow its start. The same `[start, end)` window is used for the neural stream, the
inputs and the outputs, and also for the per-trial dF/F baseline, so all streams are
sliced identically. 12,216 raw trials are found across the 152 sessions.

ii.
```python
starts = np.where(beh["trial_start"] > 0)[0]
ends = np.where(beh["teleport"] > 0)[0]
assert len(starts) == len(ends), "trial_start / teleport count mismatch"
assert np.all(ends > starts), "teleport must follow its trial_start"
n_trials_raw = len(starts)
...
for i in np.where(keep_trial)[0]:
    s, e = starts[i], ends[i]
    T = e - s
    neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
```

iii. CONVERSION_NOTES Step 1/Step 5 (decision 5): this is the reference repo's own trial
definition — `behavior.py::get_trial_types` and `define_trial_subsets` slice
`[trial_start_inds[i], teleport_inds[i])`. The AI explicitly noted that the reference
`dff()` uses a window shifted by one frame (`[start-1, stop-1)`) and chose `[start, stop)`
for *both* streams "so the two streams are exactly aligned (the reference is internally
inconsistent here)". Step 10 validates the window with whole-dataset boundary
statistics: position at the first frame of a trial is −0.8 to 10.1 cm (mean 1.95) and at
the last frame 442.6 to 451.8 cm (mean 448.6), no trial contains an intertrial
(`environment == −1`) sample, and the stray `trial number` samples at the end of some
recordings are never used because trials are defined only by start/teleport pairs.

## 1-e. How are trials filtered based on quality controls?

i. Three exclusions, applied to the raw trial list:
1. **Lick-sensor failure** — the paper's rule implemented literally: a trial is dropped
   if more than 30% of its imaging frames have a cumulative lick count > 2. This removes
   exactly **81** trials, the number the paper reports.
2. **Unscanned trials** — any trial overlapping frames with `scanning < 0` (0 such trials).
3. **Too-short trials** — fewer than 2 imaging frames (0 such trials).
12,216 raw → 12,135 kept trials. Note the paper NaNs the lick channel of the 81
lick-error trials rather than dropping the trials; the AI drops the whole trial because
NaN is not allowed in the target format.

ii.
```python
LICK_ERR_COUNT_THRESH = 2      # cumulative lick count per imaging frame
LICK_ERR_FRAC_THRESH = 0.3     # fraction of frames in the trial
...
trial_lick_error[i] = (np.mean(lick[s:e] > LICK_ERR_COUNT_THRESH)
                       > LICK_ERR_FRAC_THRESH)
trial_unscanned[i] = np.any(scanning[s:e] < 0)
...
trial_len = ends - starts
too_short = trial_len < 2
keep_trial = ~(trial_lick_error | trial_unscanned | too_short)
```

iii. CONVERSION_NOTES Step 3 quotes the Methods: "A very small number of trials with
erroneous lick detection ... (~0.65% of all imaged trials, n = 81 out of 12,376 trials)"
and the rule ">30% of the 0.0645 s imaging frame samples in the trial containing a
cumulative lick count >2". Step 4 records the match as an exact consistency check:
"My implementation of the methods rule finds **exactly 81** trials ... Exact match → the
trial-curation rule is implemented correctly." Step 9 gives the bookkeeping identity
`raw trials 12,216 = kept 12,135 + lick-error 81 + unscanned 0 + too-short 0`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `ophys/Fluorescence/plane<k>` (`F`) and
`ophys/Neuropil/plane<k>` (`Fneu`), restricted to the ROIs flagged by
`ImageSegmentation/PlaneSegmentation/iscell[:,0]`. For the two two-plane animals (m17,
m18) each plane is a separate `RoiResponseSeries` and the planes are pooled by
scattering each plane's traces into the rows of the `PlaneSegmentation` table indicated
by `planeIdx`. The NWB's stored `Deconvolved` array is deliberately **not** used.

ii.
```python
fluo = ophys.data_interfaces["Fluorescence"].roi_response_series
neuro = ophys.data_interfaces["Neuropil"].roi_response_series
plane_keys = sorted(fluo.keys(), key=lambda k: int(k.replace("plane", "")))
F = np.empty((n_roi_total, n_frames), dtype=np.float64)
Fneu = np.empty((n_roi_total, n_frames), dtype=np.float64)
for k in plane_keys:
    p = int(k.replace("plane", ""))
    rows = np.where(plane_idx_all == p)[0]
    assert len(rows) == fluo[k].data.shape[1], (
        f"{k}: {len(rows)} ROIs in table vs {fluo[k].data.shape[1]} traces")
    F[rows] = np.asarray(fluo[k].data[:]).T
    Fneu[rows] = np.asarray(neuro[k].data[:]).T
plane_idx = plane_idx_all[iscell]
F = F[iscell]
Fneu = Fneu[iscell]
```

iii. CONVERSION_NOTES Step 2: "`Deconvolved/plane0` — suite2p `spks` computed from **raw
F** (values in raw fluorescence units, e.g. 0–3800), *not* the paper's deconvolution of
dF/F", so the neural signal has to be recomputed from `F`/`Fneu` either way. Step 5
decision 10: the two planes in m17/m18 are deep/superficial CA1 and are "pooled for all
analyses except those in Extended Data Fig. 7". The `iscell` filter is applied before
the dF/F arithmetic as a memory/speed optimisation (every dF/F step is per-cell).

## 2-b. How is the `neural` data processed?

i. dF/F is computed with a port of the reference `reward_relative/preprocessing.py::dff()`
in the `neuropil_method='subtract'`, `baseline_method='maximin'`, `subtract_baseline=True`,
`keep_teleports=False` configuration: (1) NaN everything outside `[trial_start, teleport)`;
(2) `F − 0.7·Fneu`; (3) add the trial-mean neuropil back so the ratio is a true dF/F;
(4) per-trial maximin baseline — NaN-tolerant Gaussian smoothing with σ = 15 samples,
then a 300-sample `minimum_filter1d` followed by a 300-sample `maximum_filter1d`
(300 samples / 15.5 Hz ≈ 19.3 s ≈ the Methods' "20 s window"); (5) `dF/F = (F − F0)/|F0|`;
(6) a 2-sample σ Gaussian smoothing per trial. `nansmooth` is a verbatim port of the
reference helper. The result is stored as float32.

**The deconvolution step is not applied.** The Methods end the pipeline with "The
activity rate was extracted by deconvolving dF/F with a canonical calcium kernel using
the OASIS algorithm"; the AI implemented and tested OASIS (in `cache/test_dff.py`,
`cache/test_dff2.py`) but ships the smoothed dF/F as the `neural` signal. Also,
`keep_teleports` is hard-coded to `False` for every session; the reference repo sets it
per animal and per experiment day from `teleport_metadata.py` (the days on which the
laser was not blanked between trials), which the AI did not pick up.

ii.
```python
def compute_dff(F, Fneu, trial_starts, trial_ends):
    f_ = np.full(F.shape, np.nan); fneu_ = np.full(F.shape, np.nan)
    for s, e in zip(trial_starts, trial_ends):
        f_[:, s:e] = F[:, s:e]; fneu_[:, s:e] = Fneu[:, s:e]
    in_trial = ~np.isnan(f_[0, :])
    f_ -= NEU_COEF * fneu_                                   # NEU_COEF = 0.7
    flow = np.full(F.shape, np.nan)
    for s, e in zip(trial_starts, trial_ends):
        f_[:, s:e] = f_[:, s:e] + NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        tmp = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=1)          # sigma = 15
        tmp = sp.ndimage.minimum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1) # 300
        flow[:, s:e] = sp.ndimage.maximum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
    dff = np.full(F.shape, np.nan)
    dff[:, in_trial] = ((f_[:, in_trial] - flow[:, in_trial]) / np.abs(flow[:, in_trial]))
    for s, e in zip(trial_starts, trial_ends):
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)       # sigma = 2
    return dff, in_trial
```

iii. CONVERSION_NOTES Step 5, decision 2: "**Neural signal = dF/F** (computed exactly as
in `preprocessing.dff`), not the deconvolved trace. Rationale: (a) the NWB's
`Deconvolved` array is suite2p's spks from *raw* F and does **not** correspond to the
paper's deconvolution, so it had to be recomputed either way; (b) the paper itself uses
binned dF/F for its spatial peak identification 'as this signal is the closest to the raw
data'; (c) I tested both signals with a PCA+logistic position decoder on three sessions —
dF/F was equal or better in every case (m11 d3 0.53 vs 0.45, m19 d6 0.77 vs 0.63, m13 d8
0.40 vs 0.41), which is expected because the decoder reads instantaneous activity and
dF/F retains the temporal integration that a sparse event train discards. The
deconvolution step itself is implemented and validated but not used." Step 10 Check 3
tabulates the dF/F pipeline against the reference operation-by-operation and reports it
as matching; the one-frame window difference is argued to be negligible for a 300-sample
maximin window. `keep_teleports=False` is justified in Step 1 only by
`utilities.py::default_dff_method`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) ROIs are restricted to suite2p's manual curation,
`iscell[:,0] == 1`, applied before any dF/F arithmetic. (2) Putative interneurons are
dropped: cells whose dF/F correlates with the animal's running speed at Pearson
r > 0.5, computed over all within-trial samples. The correlation is vectorised over all
cells at once; NaN correlations are mapped to 0 (i.e. kept). 138,678 `iscell` ROIs →
138,276 neurons after removing 402 putative interneurons (0.29% overall,
0.35 ± 0.61% per session). No place-cell selection and no speed threshold are applied.

ii.
```python
iscell = np.asarray(plane_seg["iscell"].data)[:, 0].astype(bool)
F = F[iscell]; Fneu = Fneu[iscell]
...
d = dff[:, in_trial]
sp_ = speed[in_trial]
d0 = d - d.mean(axis=1, keepdims=True)
s0 = sp_ - sp_.mean()
denom = np.sqrt((d0 ** 2).sum(axis=1) * (s0 ** 2).sum())
with np.errstate(invalid="ignore", divide="ignore"):
    speed_corr = (d0 @ s0) / denom
speed_corr = np.nan_to_num(speed_corr, nan=0.0)
keep_cells = speed_corr <= SPEED_CORR_THRESH      # SPEED_CORR_THRESH = 0.5
dff = dff[keep_cells]
```

iii. CONVERSION_NOTES Step 3 quotes the Methods for both rules ("Manual curation
eliminated ROIs containing multiple somata or dendrites ..."; "Additional putative
interneurons were detected for exclusion ... by a Pearson correlation of >0.5 between
their dF/F timeseries and the animal's running speed"). Step 9 checks the exclusion rate
against the paper's "0.42 ± 0.85% of cells" (obtained 0.35 ± 0.61%) and the per-session
neuron count against the paper's "155–2172 putative pyramidal neurons per session"
(obtained min 155 `iscell` / 154 after interneuron removal; the upper bound 2341 is
investigated and attributed to the 3 two-plane m18 sessions). Step 5 decisions 3 and 4
justify *not* applying the paper's <2 cm/s speed mask ("speed is a decoder **output**
whose class 0 is exactly '< 2 cm/s'") and *not* restricting to place cells ("the decoder
should see the full curated pyramidal population").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Nothing beyond slicing: the alignment event is the trial start, and the neural trace
is cut at the `trial_start` frame, so sample 0 of every trial *is* the alignment event.
Behaviour and imaging already live on a single shared frame clock in the NWB (the
reference `vr_align_to_2P` interpolated VR onto the 2P frame times before export), so the
same index window aligns all streams. `off_start = 0.0`, `off_end = None`.

ii.
```python
s, e = starts[i], ends[i]
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
inp[0] = ts[s:e] - ts[s]           # time from trial start (s)
...
"temporal_alignment_event": (
    "trial start = VR 'trial_start' event, i.e. entry onto the linear "
    "track at position 0 cm after the intertrial teleport period"),
"off_start": 0.0,
"off_end": None,
```

iii. CONVERSION_NOTES Step 3/Step 10 Check 3: "VR behaviour is interpolated onto the 2P
frame times (`vr_align_to_2P`); the NWB already stores it that way (one shared timestamp
vector)", so "the NWB *is* the output of the reference alignment step". Step 10 Check 2
verifies per-trial alignment against the raw files (whole-trial `np.allclose` matches,
`max|diff| = 0`) and Step 7 verifies it visually — "the blue `trial_start` and red
`teleport` markers bracket exactly one track traversal" and "the input
`time_from_trial_start_s` starts at 0 for every trial".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 2-photon frame period is kept: **64.4836 ms** (15.5078 Hz per imaging
plane), identical in all 152 sessions. **No rebinning, resampling or interpolation is
applied at all.** The bin size written to metadata is the median of `np.diff(timestamps)`
taken over sessions. For the two-plane animals the stored series `rate` is the 31 Hz
volume rate but the per-plane sampling — and therefore the behaviour timestamp spacing —
is the same 15.5 Hz, so no special handling is needed.

ii.
```python
info["dt"] = float(np.median(np.diff(ts)))
...
dt = float(np.median([i["dt"] for i in infos]))
"time_bin_size": dt * 1000.0,
"sampling_rate_hz": 1.0 / dt,
```

iii. CONVERSION_NOTES Step 5, decision 1: "**Time bin = native imaging frame (64.484 ms).**
Every one of the 152 sessions has exactly the same `dt` (15.5078 Hz per plane), and the
NWB behaviour is already interpolated onto that clock by the reference `vr_align_to_2P`.
Using the native grid therefore satisfies 'same bin size for all trials and sessions'
with **zero resampling**, i.e. no risk of introducing temporal misalignment between the
neural and behavioural streams." Step 9 cross-checks it against the paper's "0.0645 s
imaging frame samples".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attached to the `position` `BehavioralTimeSeries` (in seconds). All
behaviour channels in these files share one timestamp vector, so the choice of channel is
immaterial.

ii.
```python
timestamps = np.asarray(bts["position"].timestamps[:])
...
ts = S["timestamps"]
inp[0] = ts[s:e] - ts[s]
```

iii. CONVERSION_NOTES Step 2: the `BehavioralTimeSeries` channels are "all sampled on the
imaging frame clock (shared `timestamps`), `dt = 0.064484 s` (15.5078 Hz) **identical in
all 152 files**". Step 10 Check 2 re-derives `input[0]` from the raw NWB as
`timestamps[start:end] − timestamps[start]` and reports PASS.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first timestamp is subtracted, so every trial starts at exactly 0 s. No
other processing; stored as float32. Values range 0 to 216.5 s across the dataset.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
inp[0] = ts[s:e] - ts[s]           # time from trial start (s)
```

iii. Direct implementation of the Decoder Task specification ("Time from start of trial in
seconds ... continuous, time-varying"). CONVERSION_NOTES Step 7 confirms
"`time_from_trial_start_s` starts at 0 for every trial" in the per-trial plots, and Step
10 lists the very long laps (18 trials > 60 s, max 216 s) as checked and deliberately kept.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is taken from the same frame window as the neural slice, so alignment is by
construction. The AI additionally guards the one place where the two streams can drift:
in 10 of the two-plane sessions the fluorescence series has exactly one frame more than
the VR-aligned behaviour, and the script asserts the excess is 0 or 1 frames and drops
the trailing frame.

ii.
```python
n_extra = F.shape[1] - len(timestamps)
assert 0 <= n_extra <= 1, (
    f"fluorescence has {F.shape[1]} frames, behaviour {len(timestamps)}")
if n_extra:
    F = F[:, :len(timestamps)]
    Fneu = Fneu[:, :len(timestamps)]
```

iii. CONVERSION_NOTES Step 10, Check 5: "10 two-plane sessions have **one more imaging
frame** than the VR-aligned behaviour (`int(max_idx/n_planes)` truncation in
`vr_align_to_2P`) — the trailing frame is dropped, with an assertion that the excess is
≤ 1 frame ... caught by an assertion during the first full run; all 152 sessions now
pass." The identity of the neural and behavioural clocks is otherwise guaranteed by the
export (Step 1/Step 10 Check 3).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` channel of `BehavioralTimeSeries`, which takes values −1 (intertrial
interval), 0 (ENV1) and 1 (ENV2).

ii.
```python
env_ts = beh["environment"]
...
ev = np.unique(env_ts[s:e])
ev = ev[ev >= 0]
trial_env[i] = int(ev[0]) if ev.size else -1
```

iii. CONVERSION_NOTES Step 4 checks the coding against the reference
`behavior.py::env_morph_dict` (`Env1→0`, `Env2→1`) and against the data: "`environment`
channel is −1 (ITI), 0, 1 ... Match. Each trial has a single environment value (verified
for all 12,216 trials)." Step 10 confirms "No trial contains an intertrial
(`environment == −1`) sample, and every trial has exactly one valid environment value."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique non-negative value within the trial is taken and broadcast as a constant
over all timepoints of the trial (`input` row 1). No recoding is needed, the raw 0/1
values are already ENV1/ENV2.

ii.
```python
inp[1] = trial_env[i]              # 0 = ENV1, 1 = ENV2
```

iii. CONVERSION_NOTES Step 5 decision 9: "Per-trial variables are broadcast to `(d, T)`
rather than stored as 1-D, so that all inputs/outputs share one array per trial and the
time-varying and per-trial variables can coexist in a single matrix." The environment
value is per-trial by design of the task, and the paper's ENV1/ENV2 split (9 mice start
in ENV1, 2 in ENV2) is reproduced (ENV1 51.1%, ENV2 48.9% of trials).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the position of the trial's `trial_start` event within the session, i.e. the
index into the raw `starts` array — not the NWB's `trial number` channel. Because dropped
(lick-error) trials keep their original index, trial numbers in the output can have gaps
and run 0 … 99.

ii.
```python
starts = np.where(beh["trial_start"] > 0)[0]
...
for i in np.where(keep_trial)[0]:
    ...
    inp[2] = i                     # trial number within session
```

iii. CONVERSION_NOTES Step 5 maps `input[2]` to the "0-indexed position of the
`trial_start` event in the session". Step 10 Check 2 validates this against the raw data
from both directions: "`input[2]` equals the raw trial index **and** the NWB `trial
number` channel at the trial-start frame — PASS", i.e. the loop index and the stored
channel agree, so nothing is lost by using the index. Step 10 Check 5 also notes the
stored `trial number` channel has stray trailing values with no matching `trial_start`,
which is a reason not to depend on it.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond assigning the raw trial index and broadcasting it as a constant across the
trial's timepoints. It is kept as a raw count (not normalised).

ii.
```python
inp[2] = i                         # trial number within session
```

iii. The Decoder Task specifies "Trial number (continuous, per trial)". Keeping the raw
index preserves the true ordinal position in the session even after the 81 lick-error
trials are dropped, which matters because the reward zone switches after trial 30.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial reward outcome of the preceding raw trial, which is itself computed
from two channels, following `behavior.py::get_trial_types`: the `Reward` `TimeSeries`
(one timestamp per delivery, mapped onto the frame grid with `searchsorted`) **and** the
`reward_zone` channel. A trial counts as rewarded only if a reward event falls inside the
trial *and* the animal entered the reward zone during it.

ii.
```python
rew_frames = np.searchsorted(ts, S["reward_times"])
...
in_zone = rzone_ts[s:e] > 0
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
...
prev_rewarded = np.empty(n_trials_raw, dtype=np.int64)
prev_rewarded[0] = 1
prev_rewarded[1:] = trial_rewarded[:-1]
```

iii. CONVERSION_NOTES Step 1 documents the reference rule: "`get_trial_types` ... Per-trial
`isreward` = `any(reward>0) and any(rzone>0)` within `[trial_start_inds[i],
teleport_inds[i])`", and Step 5 maps `input[3]` to "`isreward` of trial *i−1*" with
reference function `behavior.get_trial_types`. Step 10 verified the chain on "all 11,945
consecutive kept-trial pairs: **0 mismatches**".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The array of per-trial outcomes is shifted by one raw trial (so a trial whose
predecessor was dropped still refers to that predecessor's real outcome), and the value
is broadcast as a constant over the trial's timepoints. **Trial 0 of every session is set
to 1 (rewarded)** rather than 0, on the grounds that each imaging session is preceded by
~30 rewarded warm-up trials of the same task.

ii.
```python
prev_rewarded[0] = 1
prev_rewarded[1:] = trial_rewarded[:-1]
...
inp[3] = prev_rewarded[i]
```

iii. CONVERSION_NOTES Step 5, decision 7: "**`previous_trial_reward` for trial 0 = 1
(rewarded).** Each imaging session is immediately preceded by ~30 warm-up trials of the
same task with the same reward zone ('Before the imaging session, mice were provided 30
"warm-up" trials using the task and reward zone from the previous day'), and rewards are
omitted on only ~15% of trials, so 'rewarded' is the correct prior for the trial
preceding trial 0." Step 10 confirms "Every session's first kept trial is raw trial 0
with `previous_trial_reward = 1`."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` channel, together with the trial's reward-zone identity. The zone
identity is **not** read from the data but parsed from the VR scene name in
`nwb.identifier` (e.g. `Env1_LocationB_to_A`, `Env1_B_to_Env2_C`), with the switch
occurring after 30 trials — a port of the reference `behavior.py::get_reward_zones`. The
zone coordinates are the paper's: A 80–130, B 200–250, C 320–370 cm. The `reward_zone`
channel is used only as an independent cross-check of the scene-derived label.

ii.
```python
REWARD_ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
CHANGE_TRIAL = 30

def reward_zone_labels_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    for z in ZONE_NAMES:
        if scene.endswith("_Location" + z):
            return [z] * n_trials
    zone1 = scene[-1]
    zone0 = None
    for z in ZONE_NAMES:
        if (z + "_to") in scene:
            zone0 = z
            break
    if zone0 is None or zone1 not in ZONE_NAMES:
        raise NotImplementedError(f"Cannot parse reward zones for scene '{scene}'")
    return [zone0] * min(change_trial, n_trials) + \
           [zone1] * max(n_trials - change_trial, 0)
...
# cross-check against the reward_zone channel
if np.any(in_zone):
    p = pos[s:e][np.where(in_zone)[0][0]]
    observed_zone[i] = int(np.argmin([abs(p - REWARD_ZONES[z][0]) for z in ZONE_NAMES]))
scene_zone = np.array([ZONE_NAMES.index(z) for z in zone_labels])
seen = observed_zone >= 0
n_zone_mismatch = int(np.sum(observed_zone[seen] != scene_zone[seen]))
```

iii. CONVERSION_NOTES Step 5, decision 6: "**Reward zone from the scene name +
`change_trial = 30`** (the reference method), *cross-checked* against the position at
which the `reward_zone` channel fires on every trial where it fires. The data-driven
check cannot be used alone because the channel does not fire on reward-omission trials
(~15%)." Step 9 reports the cross-check result: "**0 mismatches / 10,394**" observed zone
entries, and Step 4 confirms the switch trial ("for every one of the 77 switch sessions,
the last trial with an observed pre-switch zone entry is ≤ 29 and the first with a
post-switch entry is ≥ 30") and the zone coordinates ("Reward-zone entry always occurs at
80 / 200 / 320 cm (±3 cm)").

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint the signed distance to the nearest point of the zone is computed:
`pos − zone_start` when the animal is before the zone (negative), `pos − zone_end` when it
is past the zone (positive), and exactly 0 while it is inside. That continuous distance
is then discretized (7-c). The continuous value itself is not stored.

ii.
```python
def discretize_reward_distance(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    ...
    return out, d
...
zs, ze = REWARD_ZONES[zone_labels[i]]
p = pos[s:e]
out[0], _ = discretize_reward_distance(p, zs, ze)
```

iii. This is the Decoder Task's "Distance to any location in the reward zone", i.e.
distance to the *nearest* point of the 50 cm zone, which is why the value saturates at 0
inside the zone. CONVERSION_NOTES Step 7 verified it graphically: "The signed reward-zone
distance crosses 0 exactly where the position trace enters the shaded reward zone", and
Step 10 Check 2 recomputed `output[0]` from the raw position for spot-checked sessions
(PASS).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned with explicit boolean masks, defaulting to class 3 (inside the
zone, d = 0): 0 for d < −50, 1 for −50 ≤ d < −10, 2 for −10 ≤ d < 0, 3 for d = 0,
4 for 0 < d ≤ 10, 5 for 10 < d ≤ 50, 6 for d > 50. The resulting distribution over the
full dataset is [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243].

ii.
```python
out = np.full(pos.shape, 3, dtype=np.int64)     # 3 == inside the zone
out[d < -50.0] = 0
out[(d >= -50.0) & (d < -10.0)] = 1
out[(d >= -10.0) & (d < 0.0)] = 2
out[(d > 0.0) & (d <= 10.0)] = 4
out[(d > 10.0) & (d <= 50.0)] = 5
out[d > 50.0] = 6
```

iii. The edges are copied verbatim from the Decoder Task specification, with class 3
reserved for "0 cm", i.e. any timepoint inside the zone. CONVERSION_NOTES Step 7: "its
7-way discretization steps 0→1→2→3→4→5→6 in order" in the diagnostic plots; the
`output_values` strings in the pickle spell out each bin.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[s:e]`, the same frame window used for the neural slice,
so no extra alignment step exists.

ii.
```python
s, e = starts[i], ends[i]
neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))
...
p = pos[s:e]
out[0], _ = discretize_reward_distance(p, zs, ze)
```

iii. Behaviour and imaging share one timestamp vector in the NWB (Step 1, Step 10 Check 3),
and the per-trial figure `processing_*_trials.png` plots the neural raster and every
output on the same time axis to show there is no offset (Step 7).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` channel of `BehavioralTimeSeries`, in cm along the 450 cm virtual track.

ii.
```python
pos = beh["position"]
...
p = pos[s:e]
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. CONVERSION_NOTES Step 2 identifies `position` (cm) as the VR track position channel;
Step 10 Check 2 recomputes `output[1]` from the raw channel with `pos // 90` and reports
PASS.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None — the raw per-frame position is used directly and only discretized. Samples
marginally outside [0, 450] (the AI measured −2.7 to 451.8 cm) fall into the open end
bins.

ii.
```python
POS_BIN_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. CONVERSION_NOTES Step 10, Check 5 lists "Position slightly outside [0, 450] (−2.7 to
451.8 cm) — `np.clip` into bins 0 and 4" as a handled edge case, and Step 7 notes the
occupancy is near-uniform over the five bins (0.18–0.21 in the sample; 0.154–0.231 on the
full dataset), "as it must be for an animal that runs the whole track every lap".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm bins spanning the 450 cm track via `np.digitize` with inner edges
[90, 180, 270, 360]; the first and last bins are open-ended. The `np.clip(..., 0, 4)` is a
no-op safeguard since `np.digitize` with 4 edges already returns 0–4.

ii.
```python
POS_BIN_EDGES = [90.0, 180.0, 270.0, 360.0]
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. Directly from the Decoder Task ("Discretized into 5 equal-sized bins spanning the
450 cm track"), with the paper's track length (450 cm) confirmed in Step 3.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame window as the neural data (`pos[s:e]`); nothing further is done.

ii.
```python
p = pos[s:e]
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. Shared timestamp vector (Step 1/Step 10 Check 3); boundary statistics in Step 10
show every trial spans ~0 → ~450 cm, i.e. the window is one full traversal with no
off-by-one.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` channel of `BehavioralTimeSeries`, which holds a cumulative lick **count**
per imaging frame (not a binary flag).

ii.
```python
lick = beh["lick"]
...
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 2 identifies `lick` as a "cumulative count per frame"; Step 5
cites the paper's own treatment ("lick counts were converted to a binary vector").

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised at > 0. In addition, the 81 trials whose lick sensor misfired (>30% of frames
with a count > 2) are removed entirely at the trial-curation stage rather than having
their lick channel NaN'ed as in the paper, because the target format forbids NaN.
Resulting distribution: 77.7% no-lick, 22.3% lick.

ii.
```python
out[3] = (lick[s:e] > 0).astype(np.int64)
...
trial_lick_error[i] = (np.mean(lick[s:e] > LICK_ERR_COUNT_THRESH) > LICK_ERR_FRAC_THRESH)
keep_trial = ~(trial_lick_error | trial_unscanned | too_short)
```

iii. CONVERSION_NOTES Step 5, decision 8: "the 81 lick-sensor-failure trials (the paper
NaNs their lick data; since `lick` is a decoder output and NaNs are not allowed, the
trials are dropped)". Step 10 Check 2 verifies `output[3]` equals `raw lick count > 0`
(PASS) and Step 7 confirms visually that "the binary lick output is 1 exactly where the
lick count is > 0" and that licks concentrate in and just before the reward zone.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame window as the neural data (`lick[s:e]`); the lick channel is already on the
imaging frame clock, so no resampling or event-time matching is needed.

ii.
```python
out[3] = (lick[s:e] > 0).astype(np.int64)
```

iii. Shared timestamp vector for all `BehavioralTimeSeries` channels (Step 2, Step 10
Check 3); alignment shown in the per-trial diagnostic plots (Step 7).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The VR scene name parsed out of `nwb.identifier`, combined with the switch-at-trial-30
convention — i.e. the same derivation as 7-a, using
`reward_zone_labels_from_scene()`. The `reward_zone` behaviour channel plus `position`
serve only as an independent validation of that label.

ii.
```python
ident_parts = nwb.identifier.rstrip("/").split("/")
scene = ident_parts[-1]
...
zone_labels = reward_zone_labels_from_scene(S["scene"], n_trials_raw)
scene_zone = np.array([ZONE_NAMES.index(z) for z in zone_labels])
...
out[4] = scene_zone[i]
```

iii. See 7-a. CONVERSION_NOTES Step 2 established that `nwb.identifier` "gives the
**scene name** (e.g. `Env1_LocationB_to_A`), exactly the string the reference
`get_reward_zones` / `define_trial_subsets` parse", verified against
`sessions_dict.py`. The resulting class balance on the full dataset is A 33.2%,
B 33.6%, C 33.3% — consistent with the paper's counterbalancing.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed: a non-switch scene `Env<k>_Location<Z>` gives zone Z for
all trials; a switch scene gives the pre-switch zone for the first 30 trials and the
post-switch zone (the last character of the scene name) afterwards. The label is mapped
A→0, B→1, C→2 and broadcast as a constant over the trial's timepoints. An unparsable
scene raises `NotImplementedError` rather than silently producing a wrong label.

ii.
```python
return [zone0] * min(change_trial, n_trials) + \
       [zone1] * max(n_trials - change_trial, 0)
...
out[4] = scene_zone[i]
```

iii. Port of `behavior.py::get_reward_zones(..., change_trial=30)`; the paper states
"Each switch occurred after 30 trials". Validated against the data in Step 9:
"Reward-zone label agreement ... **0 mismatches / 10,394**" observed zone entries.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` `TimeSeries` (which carries its own timestamps, one per delivery) and the
`reward_zone` behaviour channel — the reference's `get_trial_types` definition. Reward
timestamps are mapped onto the behaviour/imaging frame grid with `np.searchsorted`.

ii.
```python
reward_times = np.asarray(bts["Reward"].timestamps[:])
...
rew_frames = np.searchsorted(ts, S["reward_times"])
...
in_zone = rzone_ts[s:e] > 0
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
...
out[5] = trial_rewarded[i]
```

iii. CONVERSION_NOTES Step 1 records the reference rule (`isreward = any(reward>0) and
any(rzone>0)` within the trial) and Step 5 maps `output[5]` to it with reference function
`behavior.get_trial_types`. Step 4 notes the `autoreward` channel is all zeros in every
file but that automatically delivered rewards still appear in the `Reward` series, "so
`reward_outcome` is unaffected".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial: 1 if at least one reward frame falls in `[start, end)` **and** the animal
entered the reward zone in that trial, else 0. Broadcast as a constant across the trial's
timepoints. The full-dataset omission rate is 15.8% (15.36% of raw trials), matching the
paper's "~15%".

ii.
```python
rew_frames = np.searchsorted(ts, S["reward_times"])
in_zone = rzone_ts[s:e] > 0
got_reward = np.any((rew_frames >= s) & (rew_frames < e))
trial_rewarded[i] = int(got_reward and np.any(in_zone))
out[5] = trial_rewarded[i]
```

iii. CONVERSION_NOTES Step 4: "Reward omission — `isreward = any(reward>0) & any(rzone>0)`
... 15.34% of trials un-rewarded (1,874/12,216) vs paper '~15% of trials' omitted —
Match." Step 10 Check 2 recomputes `output[5]` independently from the raw NWB as
"`any(Reward in trial) and any(reward_zone > 0)`" and reports PASS. Step 12 analyses the
modest decoding accuracy for this variable (0.60) and argues it is a property of the task
(outcome is unknowable before the animal reaches the zone), not a conversion bug.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The handled cases, all documented:
- **Multi-plane sessions (m17, m18)**: `iscell`/`planeIdx` describe the pooled ROI table
  while `F`/`Fneu` are split across per-plane `RoiResponseSeries`; traces are scattered
  into the right rows via `planeIdx` with an assertion on the ROI counts.
- **One extra imaging frame** in 10 two-plane sessions: asserted to be ≤ 1 frame and
  trimmed from the end.
- **Trial 0 has no predecessor**: `previous_trial_reward` set to 1 (warm-up trials).
- **`reward_zone` never fires on ~15% of trials (omissions)**: zone identity comes from
  the scene name instead, with the channel used only as a cross-check.
- **`autoreward` channel is all zeros**: not used.
- **Position marginally outside [0, 450] cm**: absorbed by open end bins.
- **Lick-sensor failure trials, unscanned trials, trials < 2 frames**: dropped.
- **Sessions yielding < 2 trials**: dropped with a warning (never triggered).
- **Degenerate speed correlations** (NaN, e.g. a flat trace): mapped to 0 so the cell is
  kept rather than silently excluded.

ii.
```python
n_extra = F.shape[1] - len(timestamps)
assert 0 <= n_extra <= 1, (
    f"fluorescence has {F.shape[1]} frames, behaviour {len(timestamps)}")
if n_extra:
    F = F[:, :len(timestamps)]; Fneu = Fneu[:, :len(timestamps)]
...
assert len(rows) == fluo[k].data.shape[1], (
    f"{k}: {len(rows)} ROIs in table vs {fluo[k].data.shape[1]} traces")
...
speed_corr = np.nan_to_num(speed_corr, nan=0.0)
...
ev = np.unique(env_ts[s:e]); ev = ev[ev >= 0]
trial_env[i] = int(ev[0]) if ev.size else -1
...
if len(n) < 2:
    print(f"  WARNING: dropping {info['file']} with {len(n)} trials")
    continue
```

iii. CONVERSION_NOTES Step 10, Check 5 tabulates each edge case with the evidence that it
was found and how it is handled; the two real bugs (multi-plane pooling and the 1-frame
mismatch) are recorded in "Issues Found and Resolved" together with the note that the
full conversion and all checks were re-run after each fix. The AI's general stance is to
assert rather than silently repair: "caught by an assertion during the first full run;
all 152 sessions now pass".

## 13-a. What are the most time-consuming steps of the code?

i. Measured per session and printed by the script itself: (1) reading `F`/`Fneu` and the
behaviour out of the NWB file — 0.4 s to 2.4 s per session, I/O bound; (2) the dF/F
computation, dominated by the per-trial Gaussian smoothing and the two 300-sample
min/max filters — 0.3 s to 7.1 s per session; (3) pickling the 9.63 GB output — 14 s.
With 12 worker processes the whole 152-session conversion takes 60 s (46 s conversion +
14 s write).

ii.
```python
t_load = time.time(); S = load_session(path); t_load = time.time() - t_load
t_proc = time.time(); ...; t_proc = time.time() - t_proc
...
print(f"  {info['file']}: ... load={t_load:.1f}s proc={t_proc:.1f}s", flush=True)
...
print(f"[{done}/{len(files)}] elapsed {el:.0f}s, est. total {el / done * len(files):.0f}s")
...
print(f"Wrote {args.outfile} in {time.time() - t1:.0f}s ...")
```

iii. CONVERSION_NOTES Step 6/Step 7: "Reading the full `(n_frames, n_roi)` `F`/`Fneu`
arrays and transposing is the dominant cost ... Per-trial Python loops over ~80 trials
for the maximin baseline are cheap (~1–3 s/session) because each trial is a vectorised
2-D filter call", with a run-time table giving per-step times and the measured 60 s
total, "well under the 15-minute budget; no further optimisation needed".

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. What remains looped: (1) the three per-trial `for s, e in zip(trial_starts, trial_ends)`
loops in `compute_dff` (masking, baseline, post-smoothing); (2) the per-trial behavioural
loop that computes reward/environment/lick-error/zone flags; (3) the per-trial assembly
loop that slices the arrays and digitizes position, speed and reward distance. All three
run ~80 iterations per session over variable-length trials, and each iteration is itself
a vectorised array operation, so the Python overhead is small. In principle the
behavioural flags could be computed with `np.add.reduceat`-style segment reductions over
the whole session, and the discretizations could be applied once to the full session
array before slicing; the AI instead spent its optimisation effort on the steps that
dominated the profile. It did vectorise what mattered: the speed correlation over all
cells is a single matrix–vector product instead of a per-cell `np.corrcoef` loop, and the
`iscell` subset is taken before the dF/F maths.

ii.
```python
# vectorised (replaces a per-cell loop):
d0 = d - d.mean(axis=1, keepdims=True)
s0 = sp_ - sp_.mean()
denom = np.sqrt((d0 ** 2).sum(axis=1) * (s0 ** 2).sum())
speed_corr = (d0 @ s0) / denom

# still looped, once per trial:
for s, e in zip(trial_starts, trial_ends):
    tmp = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=1)
    tmp = sp.ndimage.minimum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
    flow[:, s:e] = sp.ndimage.maximum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
for i, (s, e) in enumerate(zip(starts, ends)):
    ...
for i in np.where(keep_trial)[0]:
    ...
```

iii. CONVERSION_NOTES Step 6 lists the speed-ups and their measured effect: "`iscell`
filter applied before any dF/F arithmetic — ~2× on the per-session compute and memory";
"Vectorised speed-correlation (one matrix–vector product instead of a per-cell loop) —
~50×"; "`ProcessPoolExecutor` over sessions (12 workers) — ~9× wall-clock". The per-trial
baseline loops are kept because the maximin baseline is defined *within* each trial, so
they cannot be collapsed without padding or masking.

## 13-c. What processing does the code repeat multiple times?

i. Little: each NWB file is opened exactly once and each session's dF/F is computed once.
Within a session the trial list is traversed several times (three times inside
`compute_dff`, once for the behavioural flags, once for assembly), which repeats the
loop overhead but not the arithmetic. The genuinely repeated work is: `nansmooth` is
applied twice (σ = 15 for the baseline, σ = 2 after the ratio, as the reference does);
dF/F is computed for whole sessions including the frames that later fall into dropped
trials; and the reward-zone distance is computed both in the conversion path and again,
for the first six trials, inside `make_processing_plots` when `--show-processing` is on.
Notably the AI avoided a second full pass over the dataset: unlike a survey-then-convert
design, all per-session statistics (`info`) are produced during the single conversion
pass.

ii.
```python
# one open, one pass, statistics collected inline:
with NWBHDF5IO(path, "r", load_namespaces=True) as io:
    ...
info = dict(file=..., n_roi_total=..., n_iscell=..., n_interneuron_excluded=...,
            n_trials_raw=..., n_trials_kept=..., n_zone_mismatch=..., frac_rewarded=...)
...
# repeated only in the optional plotting path:
for j in range(ntr_show):
    cls, dd = discretize_reward_distance(pos[s:e], zs, ze)
```

iii. CONVERSION_NOTES Step 6/Step 7 frame the design around avoiding redundant I/O
("Avoid unnecessary file I/O" was the instruction) and report the resulting 46 s
conversion. The exploratory scans that did re-read every file (`cache/scan_nwb.py`,
`cache/explore3.py`) were kept out of the conversion script and moved to `cache/`.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount, most of it deliberate validation:
- `discretize_reward_distance` returns the continuous distance `d` as well as the class,
  and the conversion path discards it (`out[0], _ = ...`).
- The per-trial reward-zone cross-check (`observed_zone`, `n_zone_mismatch`) and the
  various per-session counters exist only to be reported, not to influence the output.
- `dff_all` (the pre-interneuron-filter dF/F) is retained after filtering so the plotting
  function can use it; it is only needed when `--show-processing` is set.
- dF/F is computed for every in-trial frame of the session, including frames belonging to
  the 81 trials that are later dropped.
- `np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)` is a no-op, since `np.digitize` with four
  edges can only return 0–4 (same for speed).
- `n_roi_total`, `date`, `scene`, per-session timings and the full `session_info` list are
  stored in metadata and not used by the decoder.

ii.
```python
out[0], _ = discretize_reward_distance(p, zs, ze)     # continuous distance discarded
...
observed_zone[i] = int(np.argmin([abs(p - REWARD_ZONES[z][0]) for z in ZONE_NAMES]))
n_zone_mismatch = int(np.sum(observed_zone[seen] != scene_zone[seen]))
...
dff_all = dff        # only used by make_processing_plots
...
out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
```

iii. The AI does not flag these as waste; its efficiency discussion (Step 6/Step 7)
concentrates on I/O and the dF/F arithmetic, and it stops optimising once the full
conversion runs in 60 s ("Well under the 15-minute budget; no further optimisation
needed"). The validation-only computations are intentional — Step 9 and Step 10 cite
`n_zone_mismatch` (0 / 10,394) and the per-session counters as the evidence that the
scene-derived reward zone is correct.
