# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. `list_sessions()` walks every sub-directory of `/app/data` and collects every `*.nwb` file, then
sorts them by `(subject number, session number)` parsed from the file name. This yields 152 files /
152 sessions / 11 mice, which is all of the released data. Each file is opened **directly with
`h5py`** rather than with `pynwb` (the agent's stated reason is speed; it read the NWB layout from
the schema/reference and verified the paths). From each file `load_session()` reads: the raw suite2p
`processing/ophys/Fluorescence/plane*/data` and `processing/ophys/Neuropil/plane*/data`, the
`ImageSegmentation/PlaneSegmentation` `iscell` and `planeIdx` columns, the eight behaviour columns it
needs from `processing/behavior/BehavioralTimeSeries` (`position`, `speed`, `lick`, `reward_zone`,
`environment`, `trial number`, `autoreward`, `scanning`), the behaviour `timestamps`, the
`trial_start` / `teleport` flags, the `Reward` event timestamps, and the session metadata
(`subject_id`, `session_id`, `identifier` → scene + date, `ImagingPlane/location`). Sessions are
converted in parallel with a `ProcessPoolExecutor` (12 workers; whole run = 43.8 s).

ii.
```python
def list_sessions(sample=False, session_list=None):
    paths = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        subdir = os.path.join(DATA_ROOT, sub)
        if not os.path.isdir(subdir):
            continue
        for fn in sorted(os.listdir(subdir)):
            if fn.endswith(".nwb"):
                paths.append(os.path.join(subdir, fn))
    def key(p):
        base = os.path.basename(p)
        sub = int(base.split("_")[0].replace("sub-m", ""))
        ses = int(base.split("_")[1].replace("ses-", ""))
        return (sub, ses)
    paths.sort(key=key)
    ...
    return paths
```
```python
with h5py.File(path, "r") as f:
    subject = f["general/subject/subject_id"][()].decode()
    exp_day = int(f["general/session_id"][()].decode())
    scene   = parse_scene(f["identifier"][()].decode())
    region  = f["general/optophysiology/ImagingPlane/location"][()].decode()
    seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
    iscell = seg["iscell"][:, 0] > 0
    ...
    b = f["processing/behavior/BehavioralTimeSeries"]
    beh = {k: b[k]["data"][:] for k in
           ["position", "speed", "lick", "reward_zone", "environment",
            "trial number", "autoreward", "scanning"]}
    timestamps   = b["position"]["timestamps"][:]
    trial_start  = np.where(b["trial_start"]["data"][:] > 0)[0]
    teleport     = np.where(b["teleport"]["data"][:] > 0)[0]
    reward_times = f["processing/behavior/BehavioralTimeSeries/Reward"]["timestamps"][:]
```

iii. From CONVERSION_NOTES Step 2/4: the directory layout is one level deep, `sub-<id>/sub-<id>_ses-<nn>_behavior+ophys.nwb`; 11 mouse directories match the paper's "n = 11 (switch) mice", and
14 sessions per mouse (12 for m11, "for whom imaging started on day 3") match the paper's 14 task
days. The agent cross-checked the NWB contents against the reference pipeline and concluded "the NWB
*is* the product of `TwoPUtils.sess` + `vr_align_to_2P`: `BehavioralTimeSeries` == `vr_data` columns,
`Fluorescence`/`Neuropil` == `F`/`Fneu`". h5py was chosen over `pynwb` purely for read speed; Step 10
Check 2 re-reads the same files independently and matches the converted arrays to float32 precision.

## 1-b. How are the data split into subjects?

i. Each NWB file carries its own `general/subject/subject_id` (e.g. `m11`); the agent uses that
string as the subject identity rather than the directory name. The unique ids are sorted numerically
into `data['subjects']`, and `data['subject_idx']` is the index of each session's subject.

ii.
```python
subject = f["general/subject/subject_id"][()].decode()
...
subjects = sorted({r["info"]["subject"] for r in results},
                  key=lambda s: int(s.replace("m", "")))
data = {
    ...
    "subjects": subjects,
    "subject_idx": np.array([subjects.index(r["info"]["subject"]) for r in results],
                            dtype=np.int64),
```

iii. Reading the id out of the file (rather than off the directory name) is self-validating: the
result is 11 mice, `['m3','m4','m7','m11','m12','m13','m14','m15','m17','m18','m19']`, matching both
the directory names and the paper's 11 switch mice. Notes Step 3 records the paper's "n = 11 mice"
and "an additional 'fixed-condition' cohort (n = 3 mice)" that is not in the DANDI release.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Files are ordered by `(subject number, session number)` parsed from
the filename so sessions are grouped per animal and in chronological day order; the session number is
also read from `general/session_id` and kept in metadata as `exp_day`, because the experiment day
drives the `teleport_metadata` baseline rule and the reward-zone scene rule. No cross-session neuron
alignment (the paper's `multiDayROIAlign`) is attempted — each session's neurons are independent.

ii.
```python
exp_day = int(f["general/session_id"][()].decode())
scene   = parse_scene(f["identifier"][()].decode())
...
def key(p):
    base = os.path.basename(p)
    sub = int(base.split("_")[0].replace("sub-m", ""))
    ses = int(base.split("_")[1].replace("ses-", ""))
    return (sub, ses)
paths.sort(key=key)
```
```python
info = dict(session_id=os.path.basename(path).replace("_behavior+ophys.nwb", ""),
            subject=S["subject"], exp_day=S["exp_day"], scene=S["scene"], ...)
```

iii. Notes Step 2/4: "Sessions / subject = 14 (m11: from day 3)", verified as 14 for every mouse
except m11 (12). The agent explicitly confirmed that the NWB session number is the paper's experiment
day ("m11 starts at ses-03, and the paper says imaging for m11 started on day 3"), which is what makes
the per-day `teleport_metadata` lookup and the day-3/5/7/8 switch structure line up.

## 1-d. How are the data split into trials?

i. A trial is one lap: the half-open sample range `[trial_start, teleport)` taken from the NWB
`trial_start` and `teleport` behaviour flags. The inter-trial teleport period is not exported.
`load_session` asserts that the two flag counts are equal, that each teleport follows its trial start,
that laps do not overlap, and that the last teleport lies inside the imaging data. The agent
deliberately chose `[trial_start, teleport)` over the reference `dff`'s `[start-1, stop-1)` window
(Key Decision 3), because the `-1` shift would include one pre-track frame (position < 0) and drop the
last on-track frame.

ii.
```python
trial_start = np.where(b["trial_start"]["data"][:] > 0)[0]
teleport    = np.where(b["teleport"]["data"][:] > 0)[0]
...
assert len(trial_start) == len(teleport), "trial_start/teleport count mismatch"
assert np.all(teleport > trial_start), "teleport must follow its trial start"
assert np.all(trial_start[1:] > teleport[:-1]), "trials must not overlap"
assert teleport[-1] <= T, "trial extends past the imaging data"
```
```python
for k, (lo, hi) in enumerate(zip(si, ti)):
    ...
    T = hi - lo
    act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
```

iii. Notes Step 5: "A trial is one lap: NWB sample indices `[trial_start_idx, teleport_idx)`, i.e.
exactly the on-track portion (position ~0 → ~450 cm). This is the reference's `sess.trial_start_inds`
/ `sess.teleport_inds` pair." Step 10 Check 5 verifies the boundaries empirically:
`position[trial_start-1] < 0`, `position[trial_start] ≈ 0–4 cm`, `position[teleport-1] ≈ 448 cm`, and
`position[teleport]` is already interpolated into the teleport zone. Totals: 12,216 laps over 152
sessions, 80.4 ± 6.1 per session vs the paper's 80.5 ± 7.4. The agent also documents why its count is
160 laps short of the paper's 12,376: the NWB contains only complete laps, while
`vr_align_to_2P` force-completes ~1 clipped final lap per session.

## 1-e. How are trials filtered based on quality controls?

i. One filter: laps with a **lick-sensor failure** are dropped entirely — a lap where more than 30% of
its frames carry a cumulative lick count > 2. This is the paper's own criterion
(`behavior.correct_lick_sensor_error`). 81 of 12,216 laps are dropped (0.66%), leaving 12,135. No
other trial exclusion is applied: no minimum-duration filter and no `<2 cm/s` speed mask (the latter
is applied by the paper only to spatial analyses, and speed is a decoder output here whose lowest
class is "< 2 cm/s").

ii.
```python
LICK_ERROR_FRAC = 0.30   # methods: >30% of frames with cumulative lick count > 2
...
# behavior.correct_lick_sensor_error -- trials whose lick sensor was stuck
lick_error = np.array([(lick[lo:hi] > 2).sum() / (hi - lo) > LICK_ERROR_FRAC
                       for lo, hi in zip(si, ti)], dtype=bool)
...
for k, (lo, hi) in enumerate(zip(si, ti)):
    if lick_error[k]:
        continue
```

iii. Methods quote recorded in Notes Step 3: "A very small number of trials with erroneous lick
detection ... were removed ... (~0.65% of all imaged trials, n = 81 out of 12,376 trials) ... detected
by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2". The
agent's criterion reproduces **exactly 81 trials**, which it treats (Notes Step 4) as simultaneous
confirmation of the criterion *and* of the lap segmentation. It notes the one deviation: the reference
sets those laps' licks to NaN rather than dropping the lap, but "since `lick` is one of our decoder
outputs and NaNs are not permitted, dropping the trial is the faithful equivalent" (Key Decision 5).
Step 10 Check 5 separately confirms that no lap needs a duration filter (min 96 frames = 6.2 s, max
3,359 frames = 216.6 s) and that the smallest session still has 40 laps, so every session supports a
train/validation split.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw suite2p traces: `processing/ophys/Fluorescence/plane*/data` (F) and
`processing/ophys/Neuropil/plane*/data` (Fneu), restricted to the ROIs with
`ImageSegmentation/PlaneSegmentation/iscell[:,0] > 0`. The NWB `Deconvolved` array is **not** used.
Planes are concatenated ("planes were pooled for all analyses"); the `planeIdx` and the
`PlaneSegmentation` row index of every exported cell are stored in metadata for traceability.

ii.
```python
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = seg["iscell"][:, 0] > 0
plane_idx_all = seg["planeIdx"][:]

F_parts, Fneu_parts, plane_parts, roi_parts = [], [], [], []
offset = 0
for plane in sorted(f["processing/ophys/Fluorescence"].keys()):
    dset = f["processing/ophys/Fluorescence"][plane]["data"]
    nroi = dset.shape[1]
    keep = np.where(iscell[offset:offset + nroi])[0]
    F_parts.append(dset[:, :][:, keep].T.astype(np.float32))
    Fneu_parts.append(
        f["processing/ophys/Neuropil"][plane]["data"][:, :][:, keep].T.astype(np.float32))
    plane_parts.append(plane_idx_all[offset:offset + nroi][keep])
    roi_parts.append(offset + keep)
    offset += nroi
assert offset == len(iscell), "ROI count mismatch between planes and segmentation"
```

iii. Notes Step 4 discrepancy table: "NWB `Deconvolved` is suite2p's own deconvolution of **raw F**,
not of the paper's dF/F" whereas the Methods say "the activity rate was extracted by deconvolving
dF/F ... using OASIS". Resolution: "Recompute dF/F + OASIS from `Fluorescence`/`Neuropil` rather than
using the stored `Deconvolved`." Subsetting to `iscell` at read time is also an explicit memory
optimisation (Notes Step 6).

## 2-b. How is the `neural` data processed?

i. `compute_dff()` is a port of `reward_relative.preprocessing.dff` (maximin, single channel).
Per baseline segment: subtract `0.7 × Fneu`; add the segment's mean neuropil back so the ratio is a
true dF/F; Gaussian-smooth along time with σ = 15 frames; take a 300-frame running minimum then a
300-frame running maximum (the Methods' 20 s window) as the baseline; form `(F − base)/|base|`;
smooth with a σ = 2-frame Gaussian. Segments are individual laps, except on the animal/day
combinations transcribed from `teleport_metadata.teleport_sessions`, where the laser was not blanked
and the segment spans the inter-trial interval plus the next lap. OASIS deconvolution
(`dcnv.oasis(d, 2000, tau=0.7, 15.5078125)`) is implemented and available via `--signal events`.

**The exported signal is dF/F, not the deconvolved events** (`--signal dff` is the default and is what
the full run used: `Converting 152 sessions (signal=dff, workers=12)`).

ii.
```python
def compute_dff(F, Fneu, segments, deconvolve=True):
    from suite2p.extraction import dcnv
    f_    = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for lo, hi in segments:
        f_[:, lo:hi]    = F[:, lo:hi]
        fneu_[:, lo:hi] = Fneu[:, lo:hi]
    f_ -= NEU_COEF * fneu_                                  # neuropil_method == 'subtract'
    dff  = np.full(F.shape, np.nan, dtype=np.float32)
    spks = np.full(F.shape, np.nan, dtype=np.float32) if deconvolve else None
    for lo, hi in segments:
        x = f_[:, lo:hi] + NEU_COEF * np.nanmean(fneu_[:, lo:hi], axis=1, keepdims=True)
        flow = gaussian_filter1d(x, BASELINE_SMOOTH_SIGMA, axis=1)   # sigma = 15
        flow = minimum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)  # 300 frames
        flow = maximum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
        d = (x - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SMOOTH_SIGMA, axis=1)           # sigma = 2
        dff[:, lo:hi] = d
        if deconvolve:
            spks[:, lo:hi] = dcnv.oasis(np.ascontiguousarray(d), 2000, TAU, FRAME_RATE)
    return dff, spks
```
```python
keep_teleports = S["exp_day"] in TELEPORT_SESSIONS.get(S["subject"], [])
if keep_teleports:
    segments = [(si[0], ti[0])]
    segments += [(ti[k - 1] + 1, ti[k]) for k in range(1, ntrials_raw)]
else:
    segments = list(zip(si.tolist(), ti.tolist()))
...
activity = spks if signal == "events" else dff
```

iii. Notes Step 3 quotes the Methods verbatim for every constant, and Step 10 Check 3 tabulates the
stage-by-stage equivalence with `preprocessing.dff` ("identical, same constants, same slice-wise
application"). Step 10 Check 2 re-implements dF/F independently (stride-trick min/max, explicit
Gaussian convolution, float64) and matches to 9e-8–3.5e-7.

For the dF/F-vs-events choice, Notes Step 7 gives a measured comparison on 2- and 6-session subsets:
"Deconvolved events are temporally sparse (~77% of frames exactly 0), so a single 64 ms frame carries
very little information for [a memoryless per-timepoint linear] model, while dF/F integrates activity
over the indicator decay." dF/F won on every time-varying output (e.g. track_position 0.761 vs 0.697;
distance_to_reward_zone 0.560 vs 0.504 on the 6-session set). The agent frames it as "the deviation
from the paper's *decoding* signal is one the task explicitly allows ('except where ... training a
neural decoder require otherwise')" and keeps `--signal events` as a one-flag alternative.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) suite2p manual curation: only `iscell[:,0] > 0` ROIs are read.
(2) Putative interneurons are removed: the Pearson correlation between each cell's dF/F and the
animal's running speed is computed over all **in-trial** samples of the session, and cells with
r > 0.5 are dropped. The correlation is computed as a single vectorised matrix-vector product rather
than a per-cell loop; cells with an undefined correlation (zero variance) are mapped to r = 0 and
kept. Result: 138,678 `iscell` ROIs → 417 removed (0.30%) → 138,261 exported neurons.

ii.
```python
in_trial = np.zeros(len(pos), dtype=bool)
for lo, hi in zip(si, ti):
    in_trial[lo:hi] = True

d  = dff[:, in_trial]
v  = speed[in_trial].astype(np.float64)
dc = d - d.mean(axis=1, keepdims=True)
vc = v - v.mean()
denom = np.sqrt((dc.astype(np.float64) ** 2).sum(axis=1)) * np.sqrt((vc ** 2).sum())
with np.errstate(invalid="ignore", divide="ignore"):
    r_speed = (dc.astype(np.float64) @ vc) / denom
r_speed = np.nan_to_num(r_speed, nan=0.0)
keep_cells = r_speed <= INTERNEURON_SPEED_R      # INTERNEURON_SPEED_R = 0.5
n_interneurons = int((~keep_cells).sum())
...
activity = activity[keep_cells]
```

iii. Methods (Notes Step 3): manual curation "eliminated ROIs containing multiple somata or dendrites,
lacking visually obvious transients ... typical of putative interneurons", and "Additional putative
interneurons were detected for exclusion ... by a Pearson correlation of >0.5 between their dF/F
timeseries and the animal's running speed, excluding 0.42 ± 0.85% of cells". The agent's measured
0.35% ± 0.60% across sessions (0.30% overall) is reported as consistent. Notes Step 10 Check 4 also
flags the one residual mismatch honestly: max neurons/session is 2,341 curated (2,321 after
interneuron removal) vs the paper's stated 2,172, with the conclusion that "discarding real, curated
cells to reach a number in the text would not be justified".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Nothing beyond the trial split is needed. The alignment event is the start of the trial, and
neural and behaviour already live on the same imaging-frame grid in the NWB (the VR stream was
resampled to the 2P clock by the authors' `vr_align_to_2P`). Every stream for trial *k* is sliced
with the same `[lo, hi) = [trial_start[k], teleport[k])` index range, so sample *j* of the neural
matrix and sample *j* of every input/output are the same frame. Metadata records
`off_start = 0.0`, `off_end = None` (laps have variable length).

ii.
```python
for k, (lo, hi) in enumerate(zip(si, ti)):
    ...
    act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
    p   = pos[lo:hi]
    ...
    out[2] = bin_speed(speed[lo:hi])
    out[3] = (lick[lo:hi] > 0).astype(np.int64)
```
```python
"temporal_alignment_event": (
    "start of trial: the imaging frame at which the mouse enters the linear track at "
    "0 cm (NWB `trial_start` flag). Each trial runs to the `teleport` flag at the end "
    "of the lap, so trials have variable length."),
"off_start": 0.0,
"off_end": None,
```

iii. Notes Step 3: "everything already lives on the imaging frame grid (VR resampled to 2P by
`vr_align_to_2P`). Trials are laps: `[trial_start, teleport)`." The `--show-processing` figures
overlay raw F, dF/F, events, the exported neural raster, position, distance-to-zone, speed, licks and
reward markers on one session time axis to make any misalignment visible; Step 12 notes "reward
markers fall inside the lap they are assigned to, and the exported neural raster has gaps exactly at
the ITIs."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native per-plane imaging frame: 15.5078125 Hz → **64.4836 ms**, identical for every session
and animal including the two-plane mice. No rebinning, resampling or smoothing across time bins is
applied beyond what the dF/F pipeline itself does. The rate is a module-level constant rather than
being read per file; the agent verified separately that the behaviour `dt` is 0.06448363 s in all 152
sessions and that the two-plane animals store the same number of per-plane frames.

ii.
```python
FRAME_RATE   = 15.5078125          # Hz, per plane; identical for every session
FRAME_PERIOD = 1.0 / FRAME_RATE    # s
...
"time_bin_size": 1000.0 / FRAME_RATE,
"sampling_rate_hz": FRAME_RATE,
```

iii. Key Decision 2 (Notes Step 5): "Time bin = the native imaging frame, 64.4836 ms, identical for
all sessions and animals (per-plane rate is 15.5078125 Hz everywhere, including the 2-plane mice).
This is the rate the paper's GLM/decoder use ('All behavioral and neural time series were sampled at
~15.5 Hz'), so no re-binning is applied and no information is discarded." Notes Step 4 records the
cross-check "behaviour dt = 0.06448363 s in every session; 2-plane animals store the same number of
per-plane frames → one common time bin for the whole dataset", and notes that the `rate` stored on
the NWB series is the scanner rate (31 Hz on two-plane sessions), which is why the per-plane value is
the relevant one for both the time bin and the OASIS kernel.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is **synthesised from the frame index**, not read from the behaviour `timestamps`: frame
number within the lap × the constant frame period (1/15.5078125 s). The agent justified this by first
establishing that the behaviour timestamps are an exactly regular grid at that period in all 152
sessions.

ii.
```python
FRAME_PERIOD = 1.0 / FRAME_RATE
...
T = hi - lo
inp = np.empty((4, T), dtype=np.float32)
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. Notes Step 5 variable-mapping table: "frame index → `input[0]` `time_from_trial_start_s`,
`(i - trial_start_idx) / 15.5078125` seconds". The premise is the Step 4 finding that the behaviour
sample interval is 0.06448363 s in every session. Step 10 Check 2 lists "`input[0]` ==
`(frame - trial_start)/15.5078125` — PASS" as an independent re-check.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The lap's first frame is time 0 by construction (`np.arange(T)` starts at 0), so the value is
`0, 0.0645, 0.1290, …`. No offset subtraction, interpolation or smoothing. Stored as float32 in
`input[0]`, time-varying, one value per frame. Observed range over the whole dataset: [0, 216.536] s.

ii.
```python
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. Notes Step 5: per-frame time since track entry; the alignment event is trial start so
`off_start = 0.0`. The maximum of 216.536 s corresponds to the single longest lap (3,359 frames),
which Step 10 Check 5 keeps deliberately: "the paper applies no duration filter, and the decoder
handles variable T".

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the input array is built with the same length `T = hi - lo` as the neural slice,
and index *j* of `input[0]` is the *j*-th frame of the same `[lo, hi)` window used for the neural
matrix. Before any of this, `load_session` truncates fluorescence and every behaviour column to the
common length `T = min(n_frames, n_behaviour_samples)`, which matters for the 10 sessions that store
one extra imaging frame.

ii.
```python
# A handful of sessions store one extra imaging frame; align to the common length.
T = min(F.shape[1], len(beh["position"]))
F, Fneu = F[:, :T], Fneu[:, :T]
beh = {k: v[:T] for k, v in beh.items()}
timestamps = timestamps[:T]
```
```python
T = hi - lo
act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
inp = np.empty((4, T), dtype=np.float32)
inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
```

iii. Notes Step 3: neural and behaviour share the imaging-frame grid, so no resampling is needed.
Notes Step 10 Check 5 records the length mismatch as a real edge case found by scanning all 152 files
("Fluorescence one frame longer than behaviour — 10 sessions — truncate to the common length").

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `processing/behavior/BehavioralTimeSeries/environment` column (the reference's `morph`
variable), 0 = ENV1, 1 = ENV2.

ii.
```python
beh = {k: b[k]["data"][:] for k in
       ["position", "speed", "lick", "reward_zone", "environment", ...]}
...
environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                        for lo, hi in zip(si, ti)], dtype=np.int64)
```

iii. Notes Step 5 maps `behavior/environment` → `input[1]` via "per-trial mode of `morph` (0 = ENV1,
1 = ENV2)", citing `behavior.get_trial_types` (which takes `np.unique(morph[firstI:lastI])` per lap).
Notes Step 4 validates the variable: "73 sessions all ENV1, 68 all ENV2, 11 sessions contain both" —
exactly one cross-environment session per mouse, matching the paper's day-8 environment switch, and
"two mice (m17 and m18) began in ENV 2".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial median of the environment column, rounded to the nearest integer, then broadcast
across all of the lap's timepoints as a constant row of `input`. The environment is constant within a
lap, so the median is just a robust way of reading that constant.

ii.
```python
environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                        for lo, hi in zip(si, ti)], dtype=np.int64)
...
inp[1] = environment[k]
```

iii. Key Decision 9 (Notes Step 5): "Per-trial variables are broadcast across the trial's timepoints
so that `input` is (4, T) and `output` is (6, T)". The measured range over the full dataset is
[0.0, 1.0], reported in the conversion summary and the verification log. Step 10 Check 2 verifies
"`input[1]` == NWB `environment` on that lap — PASS".

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The 0-based index of the lap within the session, i.e. the position of the lap in the
`trial_start`/`teleport` flag sequence. The stored NWB `trial number` column is read but not used as
the source; the agent instead verified that the two agree.

ii.
```python
si, ti = S["trial_start"], S["teleport"]
ntrials_raw = len(si)
...
trial_number = np.arange(ntrials_raw, dtype=np.int64)
...
inp[2] = trial_number[k]
```

iii. Notes Step 5: "trial index in session → `input[2]` `trial_number`: 0-based lap index (verified
equal to NWB `trial number` at trial start)". Notes Step 10 Check 5: "VR `trial number` vs lap index
— equal in **152/152** sessions → `trial_number` input is unambiguous". (This is a point where the
human reference found disagreement between the stored `trial number` and `trial_start` and therefore
also fell back on the loop index.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond the index assignment, broadcast across the lap's timepoints as a float32 constant.
Crucially the numbering is computed over the **raw** lap list *before* the lick-error laps are
dropped, so kept laps retain their true within-session lap number (the sequence has gaps where a lap
was dropped rather than being renumbered). Range over the dataset: 0–99.

ii.
```python
trial_number = np.arange(ntrials_raw, dtype=np.int64)
...
for k, (lo, hi) in enumerate(zip(si, ti)):
    if lick_error[k]:
        continue
    ...
    inp[2] = trial_number[k]
```

iii. Key Decision 10 (Notes Step 5): "`trial_number` and `previous_trial_outcome` are computed before
any trial is dropped, so lap numbering and outcome history stay faithful to what the animal
experienced." This also keeps `trial_number` consistent with the lap-30 reward-zone switch rule, which
is indexed on raw lap number.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the previous lap's reward outcome, which is itself derived from the `Reward` time series'
**timestamps** (mapped onto the frame grid with `np.searchsorted`) combined with the `reward_zone`
flag column — i.e. the same `is_reward` vector used for `output[5]`.

ii.
```python
reward_times = f["processing/behavior/BehavioralTimeSeries/Reward"]["timestamps"][:]
...
reward_idx = np.searchsorted(timestamps, reward_times)
```
```python
# behavior.get_trial_types: rewarded iff a reward was delivered AND the reward zone
# was entered on that lap (an omission lap never sets the rzone flag).
is_reward = np.array([
    bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))
    for lo, hi in zip(si, ti)], dtype=np.int64)
```

iii. Notes Step 5 maps "reward delivery of previous lap → `input[3]` `previous_trial_outcome`:
`isreward[k-1]`", citing `behavior.get_trial_types`. Notes Step 10 Check 5 verifies the timestamp
mapping is exact: "Reward timestamps off the frame grid — none: exact in 152/152 sessions →
`searchsorted` mapping is exact".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `prev_outcome[k] = is_reward[k-1]`, broadcast across the lap's timepoints. For the **first lap of a
session there is no predecessor, and the agent sets the value to 1 (rewarded)** rather than 0. The
shift is applied over the raw lap list, so a dropped lick-error lap still supplies the history for the
lap that follows it.

ii.
```python
# First lap of a session: the immediately preceding (unrecorded) warm-up laps were
# rewarded unless randomly omitted, so 1 is the expected value.
prev_outcome = np.empty(ntrials_raw, dtype=np.int64)
prev_outcome[0] = 1
prev_outcome[1:] = is_reward[:-1]
...
inp[3] = prev_outcome[k]
```

iii. Key Decision 8 (Notes Step 5): "First lap's `previous_trial_outcome` = 1 (rewarded). Each imaging
session was preceded immediately by 30 warm-up laps on the previous day's reward zone, on which reward
was delivered unless randomly omitted (~15%), so 'rewarded' is the expected value. Affects 152 of
12,135 trials." Step 12 Check 1 separately shows the input carries no leakage about the *current* lap:
corr(previous_trial_outcome, reward_outcome) = +0.021.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `behavior/position` column plus the active reward zone for that lap. The reward zone is **not**
inferred from the data: it is read off the VR **scene name** stored in the NWB `identifier` string,
using a port of `reward_relative.behavior.get_reward_zones` — the zone is fixed by the scene, and on a
"switch" scene it changes after lap 30. Zone coordinates are the paper's: A = 80–130, B = 200–250,
C = 320–370 cm. The empirical `reward_zone` flag column is used only as a *validation* signal, not as
the source.

ii.
```python
REWARD_ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
SWITCH_TRIAL = 30                # behavior.get_reward_zones(change_trial=30)

def parse_scene(identifier):
    return identifier.strip("/").split("/")[-1]

def reward_zone_labels(scene, ntrials, change_trial=SWITCH_TRIAL):
    parts = scene.split("_")
    labels = np.empty(ntrials, dtype="<U1")
    if len(parts) == 2 and parts[1].startswith("Location"):          # Env1_LocationA
        labels[:] = parts[1][-1]
    elif len(parts) == 4 and parts[1].startswith("Location") and parts[2] == "to":
        labels[:change_trial] = parts[1][-1]                          # Env1_LocationA_to_B
        labels[change_trial:] = parts[3]
    elif len(parts) == 5 and parts[2] == "to":                        # Env1_B_to_Env2_C
        labels[:change_trial] = parts[1]
        labels[change_trial:] = parts[4]
    else:
        raise ValueError("Unrecognised scene name: %r" % scene)
    ...
    return labels
```
```python
zone_labels = reward_zone_labels(S["scene"], ntrials_raw)
zone_start  = np.array([REWARD_ZONES[l][0] for l in zone_labels])
zone_end    = np.array([REWARD_ZONES[l][1] for l in zone_labels])
```

iii. Notes Step 4: "Reward zone per trial — code says scene name + switch at trial 30 ... data shows
position at the first `reward_zone` flag of each trial is within **8.5 cm** of the predicted zone
start for **every trial of every session** ... paper says 'each switch occurred after 30 trials'.
**Consistent.** Use the reference rule (scene + trial 30); verified empirically." The zone-vs-data
comparison is also drawn in the `--show-processing` summary figure ("SANITY: zone from scene name vs.
data"), and Step 10 Check 2 lists "`output[4]` assigned zone contains the lap's reward-zone entry
position — PASS".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance in cm from the animal's position to the nearest edge of the active zone:
negative before the zone, exactly 0 anywhere inside it, positive past it. Computed per lap from the
raw position samples, then discretised.

ii.
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) to the nearest point of the reward zone; 0 while inside it."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after  = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after]  = pos[after]  - zone_end
    return d
...
out[0] = bin_distance_to_reward(signed_distance_to_zone(p, zone_start[k], zone_end[k]))
```

iii. Notes Step 5: "signed distance to nearest point of the active zone, then 7 bins", referencing
`behavior.get_reward_zones` for the zone coordinates. This is the quantity the paper calls
reward-relative position, in linear rather than circular units as the Decoder Task requires. The
resulting class distribution, [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243], is checked in Step 5
against the prior expectation "~11% of samples in class 3 (inside the 50 cm zone of a 450 cm track,
modulated by occupancy)" — the observed 23.8% reflecting the animals slowing in the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes via explicit boolean masks: 0: d < −50; 1: −50 ≤ d < −10; 2: −10 ≤ d < 0;
3: d == 0 (i.e. inside the zone); 4: 0 < d ≤ 10; 5: 10 < d ≤ 50; 6: d > 50. The masks cover the real
line, so no sample is left unassigned by the `np.empty` allocation.

ii.
```python
def bin_distance_to_reward(dist):
    """7 classes; see the Decoder Task specification.
    0: < -50 | 1: [-50,-10) | 2: [-10,0) | 3: 0 (inside zone) | 4: (0,10] | 5: (10,50] | 6: > 50
    """
    out = np.empty(dist.shape, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. Notes Step 5 restates the Decoder Task bin edges and the agent's mapping of "0 cm" to "d == 0
(inside zone)". The `--show-processing` figure panel 6 plots the continuous signed distance with
dashed lines at −50/−10/0/10/50 and the discretised class on a twin axis, expressly so the
discretisation can be checked by eye; Step 10 Check 2 re-derives the bins independently ("`output[0]`
distance-to-zone bins — PASS").

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No extra alignment. The position slice `p = pos[lo:hi]` uses the identical `[lo, hi)` window as the
neural slice, on the same frame grid, so output sample *j* and neural sample *j* are the same imaging
frame.

ii.
```python
for k, (lo, hi) in enumerate(zip(si, ti)):
    ...
    act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
    p = pos[lo:hi]
    ...
    out = np.empty((6, T), dtype=np.int64)
    out[0] = bin_distance_to_reward(signed_distance_to_zone(p, zone_start[k], zone_end[k]))
```

iii. Same rationale as 2-d: the NWB behaviour stream was already resampled onto the imaging frame
clock by the authors, so identical indexing is identical timing. Verified by the per-session
processing plots and by the independent Step 10 spot checks.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `processing/behavior/BehavioralTimeSeries/position` column (cm along the 450 cm virtual
track), used raw.

ii.
```python
beh = {k: b[k]["data"][:] for k in
       ["position", "speed", "lick", "reward_zone", "environment", ...]}
...
pos, speed, lick, rzone = beh["position"], beh["speed"], beh["lick"], beh["reward_zone"]
...
p = pos[lo:hi]
```

iii. Notes Step 5 maps `behavior/position` → `output[1]` `track_position` with no intermediate
transform. Step 10 Check 5 audits the raw values across all 152 files: 366 laps dip to −2.74 cm at
track entry and the maximum is 451.83 cm, both of which the agent verified land in the open end bins.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the lap's samples and discretising them. No smoothing, no unwrapping, no
clipping of the raw values.

ii.
```python
p = pos[lo:hi]
...
out[1] = bin_position(p)
```

iii. Notes Step 5: the track is 450 cm (Methods) and each lap starts at 0 cm, so the raw column is
already in the units the Decoder Task asks for. The observed class distribution
[0.212, 0.177, 0.231, 0.226, 0.154] is compared in Step 5/9 with the expectation of "roughly
uniform-ish with more mass in slow (pre-reward) segments".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins via `np.digitize(pos, [90, 180, 270, 360])`, wrapped in a `np.clip(..., 0, 4)`
(a no-op for four internal edges, but it makes the end bins explicitly open). Samples slightly below
0 cm fall in class 0 and samples slightly above 450 cm in class 4, rather than forming spurious
classes.

ii.
```python
def bin_position(pos):
    """5 equal 90 cm bins spanning the 450 cm track."""
    return np.clip(np.digitize(pos, [90.0, 180.0, 270.0, 360.0]), 0, 4).astype(np.int64)
```

iii. Notes Step 5 spells out the intended mapping "0: p < 90 | 1: 90 ≤ p < 180 | 2: 180 ≤ p < 270 |
3: 270 ≤ p < 360 | 4: p ≥ 360", matching the Decoder Task's "5 equal-sized bins spanning the 450 cm
track". Step 10 Check 5 confirms the end-bin behaviour on the real out-of-range samples
("Position slightly < 0 at lap start — bin 0, which is correct"; "Position slightly > 450 cm — bin 4,
correct").

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical `[lo, hi)` indexing as the neural matrix; nothing further. Because `load_session` first
truncates fluorescence and behaviour to a common length, the frame indices refer to the same frames in
both streams even in the 10 sessions with an extra imaging frame.

ii.
```python
T = min(F.shape[1], len(beh["position"]))
F, Fneu = F[:, :T], Fneu[:, :T]
beh = {k: v[:T] for k, v in beh.items()}
...
p = pos[lo:hi]
out[1] = bin_position(p)
```

iii. Same as 2-d / 7-d. Additionally, the `--show-processing` figure plots position with the lap
boundaries and the assigned reward zone overlaid on the same time axis as the exported neural raster,
which is the visual check the agent describes for "no temporal misalignments".

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `processing/behavior/BehavioralTimeSeries/lick` column, which holds a per-frame **cumulative
lick count** (not already binary).

ii.
```python
beh = {k: b[k]["data"][:] for k in
       ["position", "speed", "lick", "reward_zone", "environment", ...]}
...
out[3] = (lick[lo:hi] > 0).astype(np.int64)
```

iii. Notes Step 5 maps `behavior/lick` → `output[3]` with the transform "cumulative per-frame lick
count → binary (>0)", citing both the dataset README ("anything >1 gets set to 1") and the Methods
("lick counts were converted to a binary vector"). The same column is what drives the lick-sensor
error trial filter in 1-e.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any frame with a lick count > 0 becomes 1, otherwise 0. No temporal smoothing, no
spatial binning (the paper's lick-rate analyses bin in 10 cm of position, which is not applicable to a
per-frame decoder output), and no separation of anticipatory vs consummatory licks. Laps whose lick
sensor was stuck are removed entirely (1-e) rather than having their licks NaN-ed, since NaN is not a
representable output class. Resulting distribution: 77.7% no-lick / 22.3% lick.

ii.
```python
out[3] = (lick[lo:hi] > 0).astype(np.int64)
```
```python
lick_error = np.array([(lick[lo:hi] > 2).sum() / (hi - lo) > LICK_ERROR_FRAC
                       for lo, hi in zip(si, ti)], dtype=bool)
...
if lick_error[k]:
    continue
```

iii. Key Decision 5 (Notes Step 5) for the trial removal; Notes Step 5 mapping table for the
binarisation. The `--show-processing` panel 8 overlays the raw cumulative count, the exported binary
and the reward markers so the binarisation and the reward alignment can be inspected together.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[lo, hi)` frame window as the neural data; the lick column is part of the behaviour block
already resampled onto the imaging clock and truncated to the common length.

ii.
```python
out[3] = (lick[lo:hi] > 0).astype(np.int64)
```

iii. As for the other time-varying outputs (2-d). Step 12's reward-zone split analysis on `lick`
(balanced accuracy 0.700 before the zone vs 0.807 from the zone onward) is used by the agent as
indirect evidence that lick timing is correctly aligned with neural activity.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The VR scene name in the NWB `identifier` string, via the same `reward_zone_labels()` port of
`behavior.get_reward_zones` described in 7-a. The zone label is mapped A → 0, B → 1, C → 2.

ii.
```python
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}
...
zone_labels = reward_zone_labels(S["scene"], ntrials_raw)
zone_idx = np.array([ZONE_TO_IDX[l] for l in zone_labels], dtype=np.int64)
...
out[4] = zone_idx[k]
```

iii. See 7-a. Additional support in Notes Step 3: "The reward zone was a 'hidden', unmarked 50 cm span
at one of three possible locations ... zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" and
"Each switch occurred after 30 trials". `reward_zone_labels` raises on an unrecognised scene grammar
or an unknown zone letter, so a mis-parse fails loudly rather than silently mislabelling.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Parse the scene name into one of three grammars (`Env<n>_Location<Z>`,
`Env<n>_Location<Z0>_to_<Z1>`, `Env<n>_<Z0>_to_Env<m>_<Z1>`); for the two switch grammars, assign the
first zone to laps 0–29 and the second to laps 30 onwards; map the letter to 0/1/2 and broadcast the
per-lap value across the lap's timepoints. Lap indices are the raw ones, so the switch lands on the
true lap 30 even if an earlier lap was dropped. Resulting distribution:
[0.332, 0.336, 0.333] — essentially balanced across A/B/C.

ii.
```python
    if len(parts) == 2 and parts[1].startswith("Location"):          # Env1_LocationA
        labels[:] = parts[1][-1]
    elif len(parts) == 4 and parts[1].startswith("Location") and parts[2] == "to":
        labels[:change_trial] = parts[1][-1]                          # Env1_LocationA_to_B
        labels[change_trial:] = parts[3]
    elif len(parts) == 5 and parts[2] == "to":                        # Env1_B_to_Env2_C
        labels[:change_trial] = parts[1]
        labels[change_trial:] = parts[4]
    else:
        raise ValueError("Unrecognised scene name: %r" % scene)
    unknown = set(labels) - set(REWARD_ZONES)
    if unknown:
        raise ValueError("Unknown reward zone label(s) %s in scene %r" % (unknown, scene))
```
```python
out[4] = zone_idx[k]
```

iii. Notes Step 10 Check 4: "switch at lap **30** in 77/77 switch sessions", and Step 4's empirical
check that the first `reward_zone` flag of every one of the 12,216 laps falls within 8.5 cm of the
predicted zone start — i.e. the rule was confirmed against the data rather than assumed.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` time series' **timestamps** (mapped to frame indices with `np.searchsorted` against
the behaviour timestamps) **and** the `reward_zone` flag column. A lap counts as rewarded only if a
reward event fell inside it *and* the reward-zone flag was set at some point during it — a direct
port of `behavior.get_trial_types`.

ii.
```python
reward_times = f["processing/behavior/BehavioralTimeSeries/Reward"]["timestamps"][:]
...
reward_idx = np.searchsorted(timestamps, reward_times)
```
```python
# behavior.get_trial_types: rewarded iff a reward was delivered AND the reward zone
# was entered on that lap (an omission lap never sets the rzone flag).
is_reward = np.array([
    bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))
    for lo, hi in zip(si, ti)], dtype=np.int64)
```

iii. Notes Step 5 maps "`behavior/Reward/timestamps` + `behavior/reward_zone` → `output[5]`
`reward_outcome`: `any(reward in lap) AND any(rzone>0 in lap)`", citing `behavior.get_trial_types`.
Notes Step 10 Check 5 quantifies the AND: "Lap with reward-zone flag but no reward — 52 laps (entered
the zone, never licked) → counted as **not** rewarded, per `get_trial_types`"; "Lap with reward but no
zone flag — 0 laps → the reference's AND is therefore never the deciding factor"; "Reward timestamps
off the frame grid — none: exact in 152/152 sessions".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per lap, reduce to a single binary value with `np.any` over the lap's frame window, then broadcast
that constant across all of the lap's timepoints (so `output[5]` is a flat row, not an event marker at
the reward frame). Computed on the raw lap list so it can also feed `previous_trial_outcome`.
Resulting distribution: 84.2% rewarded / 15.8% omitted, against the paper's "~15% of trials" omitted.

ii.
```python
out[5] = is_reward[k]
```
```python
print("fraction rewarded  : %.4f (mean over sessions)" % fr.mean())   # 0.8466
```

iii. Notes Step 4: "Reward rate — `any(reward) AND any(rzone)` per trial — 84.66% rewarded — paper
'~15%' omitted — **Consistent**." Step 12 Check 1 further verifies the schedule is random as the paper
states (P(rewarded) = 0.845 / 0.845 / 0.849 for zones A / B / C; near-zero correlation with
trial number, environment and previous outcome), and explains the modest decoding accuracy for this
output as a task-imposed ceiling (at chance before the animal reaches the zone, 0.654 from the zone
onward) rather than a conversion error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several, discovered by an explicit edge-case scan of all 152 files (Notes Step 10 Check 5):
- **Neural/behaviour length mismatch** (10 sessions have one extra imaging frame): both streams are
  truncated to the common length.
- **Trial structure**: assertions that `#trial_start == #teleport`, that each teleport follows its
  start, that laps do not overlap, and that the last teleport is inside the imaging data. None fired.
- **Lick-sensor failure** (81 laps): laps dropped (see 1-e).
- **Undefined speed correlation** (a zero-variance cell would give r = NaN): `np.nan_to_num(..., 0.0)`
  so such a cell is kept rather than silently dropped.
- **Non-finite neural samples inside a lap**: hard error rather than silent export.
- **Out-of-range behaviour values** (position −2.74 to 451.83 cm, negative smoothed speeds down to
  −6.4 cm/s): absorbed by the open end bins, which the agent verified is the intended class.
- **Laps with no reward-zone flag at all** (1,822 = 14.9%, the omission laps): `reward_outcome = 0`.
- **`autoreward` all zeros in every file**: noted and deliberately unused.
- **No predecessor for the first lap of a session**: `previous_trial_outcome = 1` (Key Decision 8).

ii.
```python
# A handful of sessions store one extra imaging frame; align to the common length.
T = min(F.shape[1], len(beh["position"]))
F, Fneu = F[:, :T], Fneu[:, :T]
beh = {k: v[:T] for k, v in beh.items()}
timestamps = timestamps[:T]

assert len(trial_start) == len(teleport), "trial_start/teleport count mismatch"
assert np.all(teleport > trial_start), "teleport must follow its trial start"
assert np.all(trial_start[1:] > teleport[:-1]), "trials must not overlap"
assert teleport[-1] <= T, "trial extends past the imaging data"
```
```python
r_speed = np.nan_to_num(r_speed, nan=0.0)
...
act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
if not np.all(np.isfinite(act)):
    raise RuntimeError("non-finite activity in %s trial %d" % (path, k))
```

iii. Notes Step 10 Check 5 is an explicit table of every edge case the agent looked for, with the
count found in the data and the handling chosen. Its stated philosophy for the two statistics that
still do not match the paper (12,216 vs 12,376 laps; max 2,341 vs 2,172 neurons/session) is to
document rather than to force: "a lap with no teleport has no defined end", and "discarding real,
curated cells to reach a number in the text would not be justified".

## 13-a. What are the most time-consuming steps of the code?

i. Per-session timing is instrumented and printed (`load_time_s`, `dff_time_s`, `total_time_s`). The
two dominant costs are (1) reading the raw F/Fneu blocks out of HDF5 (0.3–1.8 s per session) and
(2) `compute_dff` — the Gaussian, minimum and maximum filters plus the per-lap loop over an
(n_neurons × T) matrix (0.4–5.2 s per session, scaling with neuron count). After the parallel loop,
pickling the 9.63 GB result takes 9.9 s, which is the largest single serial step. Total wall clock:
**43.8 s** for all 152 sessions on 12 workers (33.8 s conversion + 9.9 s write).

ii.
```python
t_load = time.time(); S = load_session(path); t_load = time.time() - t_load
...
t_dff = time.time()
dff, spks = compute_dff(S["F"], S["Fneu"], segments, deconvolve=...)
t_dff = time.time() - t_dff
```
```python
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for i, res in enumerate(pool.map(_worker, jobs)):
        ...
        print("[%3d/%3d] %-22s %4d neurons  %3d trials  (load %.1fs dff %.1fs "
              "total %.1fs)  elapsed %.0fs" % (...))
```

iii. Notes Step 6/7: "Timing is printed per session (load / dF/F / total) and for the whole run";
"`ProcessPoolExecutor` over sessions (default 10 workers); sessions are independent"; "float32
everywhere (halves memory/IO versus the reference's float64)". The agent also investigated and
rejected one candidate optimisation: "the h5py read `dset[:, :][:, keep]` materialises the whole
(T, n_roi) block before subsetting. Fancy-indexing h5py directly is *much* slower here (the files are
contiguous and unchunked), so the full read is deliberate."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent's claim is that only one Python loop remains and that it is irreducible: "Per-session
work is fully vectorised over neurons; the only Python loop is over laps (~80 per session), which is
unavoidable because baselines are per-lap." That is accurate for the expensive part — `compute_dff`'s
per-segment loop cannot be collapsed because each lap has its own maximin baseline, and the
interneuron correlation is a single matrix–vector product rather than the reference's per-cell loop.

What remains vectorisable, at negligible cost: the four per-lap Python comprehensions that build
`is_reward`, `environment`, `lick_error` and the `in_trial` mask each walk the lap list separately and
could be done in one pass with `np.add.reduceat` / cumulative sums; and the final assembly loop
computes `bin_position`, `bin_speed` and the lick binarisation lap by lap when they could be applied
once to the whole session array before slicing. All of these are O(T) on 1-D arrays of ~10⁴ samples,
so the saving would be milliseconds against the seconds spent in `compute_dff`.

ii.
```python
in_trial = np.zeros(len(pos), dtype=bool)
for lo, hi in zip(si, ti):
    in_trial[lo:hi] = True
...
is_reward   = np.array([... for lo, hi in zip(si, ti)], dtype=np.int64)
environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                        for lo, hi in zip(si, ti)], dtype=np.int64)
lick_error  = np.array([(lick[lo:hi] > 2).sum() / (hi - lo) > LICK_ERROR_FRAC
                        for lo, hi in zip(si, ti)], dtype=bool)
...
for k, (lo, hi) in enumerate(zip(si, ti)):
    ...
    out[1] = bin_position(p)
    out[2] = bin_speed(speed[lo:hi])
```

iii. Notes Step 6, "Code efficiencies". The per-lap structure is dictated by the reference processing
(per-lap baselines, per-lap task variables) and by the output format itself, which requires one array
per trial; the agent's efficiency effort went into parallelising across sessions and into vectorising
across neurons, which is where the time actually is.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and each session is processed in a single pass —
there is no separate survey pass (the reward-zone assignment comes from the scene name, so no
data-driven pre-pass over all files is needed). Within a session, the per-lap loop in `compute_dff` is
entered twice (once to mask F/Fneu into the segments, once to compute baselines and smooth), mirroring
the reference's own structure. The remaining repetition is small: four separate per-lap passes over
the behaviour arrays (13-b), and, under `--show-processing` only, `signed_distance_to_zone` is
recomputed for the plotted laps instead of reusing the values already in `outputs`.

ii.
```python
for lo, hi in segments:
    f_[:, lo:hi]    = F[:, lo:hi]
    fneu_[:, lo:hi] = Fneu[:, lo:hi]
...
for lo, hi in segments:
    x = f_[:, lo:hi] + NEU_COEF * np.nanmean(fneu_[:, lo:hi], axis=1, keepdims=True)
    ...
```
```python
# plot_processing (only under --show-processing)
dist = np.concatenate([signed_distance_to_zone(beh["position"][si[k]:ti[k]],
                                               zone_start[k], zone_end[k])
                       for k in range(ntr)])
```

iii. Notes Step 6 lists the deliberate single-pass design ("ROI selection is applied at read time so
only `iscell` columns are kept in memory"). The two-pass structure inside `compute_dff` is inherited
from `preprocessing.dff` and preserved on purpose, since Step 10 Check 3 required the dF/F to match
the reference "same slice-wise application (so the same `reflect` boundaries)".

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount, all of it cheap:
- Three behaviour columns are read from every file and never used by the conversion: `trial number`
  (used only for an out-of-band verification that the lap index matches), `autoreward` (all zeros in
  every file) and `scanning` (all ones). `timestamps` is read and truncated but never used either,
  since `input[0]` is synthesised from the frame index and reward events are located via
  `searchsorted` — which *does* use it.
- `plane_of_cell` / `roi_index` are carried through the whole pipeline purely so provenance can be
  stored in `metadata.session_info`; they play no role in the decoder.
- Under `--show-processing`, OASIS deconvolution runs for the plotted sessions even when the exported
  signal is dF/F, and the full `r_speed` vector and per-lap empirical zone-entry positions are
  recomputed for the figures.
- The `np.clip(..., 0, 4)` in `bin_position` and `bin_speed` is a no-op given four internal bin edges.
- The `and np.any(rzone[lo:hi] > 0)` term in `is_reward` never changes the answer (0 laps have a
  reward without a zone flag), though it is kept for fidelity to `get_trial_types`.

Notably absent: when `--signal dff` (the default), `compute_dff` is called with `deconvolve=False`, so
the expensive OASIS step is skipped entirely rather than computed and thrown away.

ii.
```python
beh = {k: b[k]["data"][:] for k in
       ["position", "speed", "lick", "reward_zone", "environment",
        "trial number", "autoreward", "scanning"]}
```
```python
dff, spks = compute_dff(S["F"], S["Fneu"], segments,
                        deconvolve=(signal == "events") or show_processing)
```
```python
def bin_position(pos):
    return np.clip(np.digitize(pos, [90.0, 180.0, 270.0, 360.0]), 0, 4).astype(np.int64)
```

iii. The agent does not call these out as waste; Notes Step 6 instead presents the pipeline as already
minimal, and Step 10 records the provenance fields as a deliberate late addition ("*ROI provenance was
not recorded*: added `roi_index` and `plane_of_neuron` per session ... This is what makes the neural
sanity check possible"), i.e. they are unnecessary for the decoder but necessary for validation. The
unused behaviour columns are 1-D arrays of ~10⁴ float samples, so the cost is immaterial against the
(n_neurons × T) fluorescence read.
