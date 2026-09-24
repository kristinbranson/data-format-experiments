# Decisions

> Note on provenance: the decisions below are reconstructed from `/app/convert_data.py`,
> `/app/CONVERSION_NOTES.md`, and the agent trajectory in `/logs/agent/trajectory.json`.
> The prompt actually delivered to the agent (trajectory step 3) specified **two-class**
> discretizations for tongue/paw/motion energy and **two-class** lick direction and outcome
> (no `none` / `ignore` / `not visible` classes). Several differences from the human
> reference follow directly from that.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB v7.3 file `data_structure_<anm>_<date>.mat`, read whole with
`mat73.loadmat(...)['obj']`, plus a companion `motionEnergy_<anm>_<date>.mat` read with
`scipy.io.loadmat`. The session list is **hard-coded**, not globbed: a list of 12
`SessionSpec(animal, date, probe)` entries transcribed from the paper's Figure 8
context-analysis script. Only the `Ephys_Behavior` folder is used; `DATA_DIR` is pinned to
it, so the 22 `RandomizedDelay_Ephys_Behavior` sessions and the 13 other `Ephys_Behavior`
sessions on disk are never opened. Everything (trials, spikes, video, motion energy) for a
session is taken from that one `obj` in a single pass in `convert_dataset`.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

# Session roster from code/Scripts/Figure 8/Figure8a_thru_c.m and the loader files
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
    bp = obj["bp"]
    go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
    ntrials = int(bp["Ntrials"])
    probe = obj["clu"][spec.probe - 1]
```

```python
def load_motion_energy(obj, motion_energy_path, taxis, align_times) -> dict:
    mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
    me = mat["me"]
    raw_data = me.data
    if hasattr(raw_data, "data"):
        raw_data = raw_data.data
```

iii. From the trajectory: "I've identified the likely target cohort: the 12 two-context ALM
ephys sessions"; "The MATLAB metadata confirms the session roster is curated by animal/date,
not by scanning the whole folder"; and "the Figure 8 context analysis explicitly uses the 12
sessions from JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, and JEB19". The agent cross-checked the
resulting unit count (520) against the paper's reported "12 sessions / 522 units / 6 mice"
for the two-context cohort and documented the residual mismatch in `CONVERSION_NOTES.md`.
It reasoned that because context (WC vs DR) is a required decoder output, the two-context
cohort is the right dataset. It did *read* the per-animal `load<ANM>_ALMVideo.m` loaders
(which name the full 44-session set the reference uses) and it did compute that 22 of the 25
`Ephys_Behavior` sessions contain both WC and DR trials, but still froze the roster at 12.

## 1-b. How are the data split into subjects?

i. The animal is carried explicitly on each `SessionSpec` (`spec.animal`, the part of the
filename before the date). `subjects` is built in first-appearance order and `subject_idx`
records each session's index into it. The 12 sessions come from 7 animals (JEB6, JEB7, EKH1,
EKH3, JGR2, JGR3, JEB19).

ii.
```python
subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)
...
full_data["subject_idx"].append(subject_idx)
...
full_data["subject_idx"] = np.asarray(full_data["subject_idx"], dtype=np.int64)
```

iii. The agent noted a discrepancy — "the paper reports six mice while those session loaders
name seven IDs" — and inspected `getAnimalNames.m` / `groupSessionsByAnimal.m` to see how the
authors group animals. It concluded the authors also key on the animal string from the
metadata and kept all seven IDs as distinct subjects, documenting the 7-vs-6 discrepancy.

## 1-c. How are the data split into sessions?

i. One session = one entry of `SESSION_SPECS` = one `data_structure_*.mat` file = one element
of `neural` / `input` / `output` / `brain_region_idx` / `subject_idx`. A single probe is used
per session (the probe recorded in the authors' metadata loaders); no session in the roster is
treated as two-probe. A session is only kept if it yields at least 10 units and at least 2
trials, otherwise the script raises. Result: 12 sessions, 2,415 trials, 520 units.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int  # 1-based probe index, as in the MATLAB metadata loaders

    @property
    def data_path(self) -> Path:
        return DATA_DIR / f"data_structure_{self.stem}.mat"
```
```python
if trialdat.shape[1] < 10:
    raise ValueError(f"{spec.stem}: fewer than 10 units remained after filtering")
...
if kept_trial_indices.size < 2:
    raise ValueError(f"{spec.stem}: fewer than 2 valid trials remained after filtering")
```

iii. Same as 1-a: the roster is the Figure 8 two-context cohort. The agent spent several steps
resolving whether `EKH1` belongs (it is in `Figure8a_thru_c.m` but absent from
`NullPotent/SessionMeta.csv`) and concluded from unit counts that "`EKH1` is very likely part
of the intended 522-unit dataset rather than an accidental extra".

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial fields of `obj.bp`, of which there are
`bp.Ntrials`. Spikes carry their own 1-based `trial` index and a within-trial time `trialtm`;
video frames are stored per trial in `traj[view]['frameTimes'][trial]` / `['ts'][trial]`; and
motion energy is one array per trial. So no trial boundaries are reconstructed — the arrays
are indexed by trial directly, and `bp.ev.goCue[trial]` gives the one go cue of that trial.
Spikes whose trial index falls outside `[0, ntrials)` are dropped.

ii.
```python
ntrials = int(bp["Ntrials"])
...
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
finite = (spike_trials >= 0) & (spike_trials < ntrials) & np.isfinite(spike_times)
```
```python
for trix in range(ntrials):
    dropped = np.asarray(cam["NdroppedFrames"][trix]).reshape(-1)
    ...
    frame_times = get_trial_frame_times(cam, trix)
```
```python
if len(raw_trials) != ntrials:
    raise ValueError(f"Motion energy trial count mismatch for {motion_energy_path.name}")
```

iii. The agent dumped the `obj.bp` layout early ("ev goCue ndarray (416,) float64 …") and
confirmed every per-trial field has `Ntrials` entries and that spikes/video carry trial
indices, so the Bpod table defines trials directly. It added the motion-energy trial-count
assertion as one of its "sanity checks".

## 1-e. How are trials filtered based on quality controls?

i. A single boolean mask per session, applied before any output is built:
`(hit | miss) & ~early & ~no & ~stim.enable`. That is — keep only trials where the animal
responded (hit or miss), drop early-lick trials, drop no-response ("ignore") trials, and drop
photostimulation trials. `~no` is redundant with `hit | miss`. Applied across the 12 sessions
this keeps 2,415 of ~3,626 trials (≈67%). No check is made for trials that run past the end of
the ephys recording.

ii.
```python
def build_keep_trial_mask(bp: dict) -> np.ndarray:
    hit = as_array(bp["hit"], bool).ravel()
    miss = as_array(bp["miss"], bool).ravel()
    early = as_array(bp["early"], bool).ravel()
    no = as_array(bp["no"], bool).ravel()
    stim = get_stim_enable(bp)
    return (hit | miss) & ~early & ~no & ~stim
```
```python
keep_trials = build_keep_trial_mask(bp)
kept_trial_indices = np.flatnonzero(keep_trials)
```

iii. `CONVERSION_NOTES.md`: "Trial export filtering follows the paper text: exclude early-lick
trials, exclude no-response trials, exclude `stim.enable` trials, keep hit and miss trials from
both DR and WC contexts." The agent took the exclusions straight from the authors' condition
strings, which are uniformly of the form `~autowater&~early&~no` / `~stim.enable&~early`.
Dropping `no` trials is also forced by the two-class outcome specification in the prompt it
received (incorrect = 0, correct = 1): an ignore trial has no valid label under that scheme.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu[probe-1]`, the spike-sorted clusters of the one probe named for that session. Three
of its per-cluster fields are used: `trial` (1-based trial of each spike), `trialtm` (spike time
relative to that trial's start), and `quality` (manual curation label). `obj.bp.ev.goCue` is the
other input, since it defines the alignment.

ii.
```python
probe = obj["clu"][spec.probe - 1]
...
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
...
aligned = spike_times - go_cue[spike_trials]
```
```python
qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
```

iii. The agent read `code/DataLoadingScripts/findClusters.m`, `getSeq.m`, and `alignSpikes.m`
and mirrored their field usage; `trialtm` is already on the behaviour clock relative to trial
start, so it is directly comparable to `bp.ev.goCue`.

## 2-b. How is the `neural` data processed?

i. Per unit: spikes are histogrammed into 10 ms bins spanning −3.0 to +2.5 s from the go cue
with `np.add.at`, divided by the bin width to give Hz, then smoothed along time with a **causal**
half-Gaussian — the `gausswin(15, α=2.5)` kernel of the authors' `mySmooth.m` with its first
half zeroed and the remainder renormalised — using `reflect` boundary handling. Nothing else is
done: no normalisation, no baseline subtraction, no z-scoring. Stored as `float32` in Hz, shape
`(n_units, 550)` per trial. Note that because the bin is 10 ms rather than the code's 5 ms, the
15-sample kernel spans 150 ms of real time rather than 75 ms.

ii.
```python
counts = np.zeros((ntrials, NT), dtype=np.float64)
np.add.at(counts, (spike_trials, bins), 1.0)
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```
```python
def my_smooth(x, n, bctype="none"):
    ...
    if bctype.lower() == "reflect":
        arr_filt = np.concatenate([arr[:n, :], arr], axis=0)
        trim = n
    ...
    kernel = gausswin(n)
    kernel[: n // 2] = 0          # causal
    kernel /= kernel.sum()
    for col in range(arr_filt.shape[1]):
        out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")
```

iii. `CONVERSION_NOTES.md`: "causal Gaussian smoothing with `smooth = 15` and the same boundary
handling as `mySmooth.m`", i.e. a deliberate line-by-line port of the authors' smoother
(`params.smooth = 15`, `params.bctype = 'reflect'`) rather than a symmetric filter.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters plus one session-level one. First the manual curation label, stripped
and lower-cased, is matched against `garbage`, `gabrga`, `noisy`, `real?` — everything else is
kept, including multi-units and unlabelled clusters. Then every surviving unit's PSTH is
computed over seven trial conditions, averaged over time and conditions, and units at or below
1 Hz are dropped. Finally a session with fewer than 10 surviving units raises an error. Across
the 12 sessions 522 quality-passing clusters become 520 units after the rate cut (27–67 per
session).

ii.
```python
def get_quality_mask(probe: dict) -> np.ndarray:
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad
```
```python
def apply_low_fr_filter(trialdat, bp):
    conds = build_low_fr_conditions(bp)
    psth = np.zeros((trialdat.shape[0], trialdat.shape[1], len(conds)), dtype=np.float32)
    for cond_idx, cond_mask in enumerate(conds):
        trials = np.flatnonzero(cond_mask)
        if trials.size:
            psth[:, :, cond_idx] = trialdat[:, :, trials].mean(axis=2)
    mean_frs = psth.mean(axis=0).mean(axis=1)
    keep = mean_frs > LOW_FR
    return keep, mean_frs
```

iii. Directly ported: `findClusters.m` with `params.quality = {'all'}` returns
`~ismember(...,'garbage') & ~ismember(...,'gabrga') & ~ismember(...,'noisy') & ~ismember(...,'real?')`,
and `removeLowFRClusters.m` computes `meanFRs = mean(mean(obj.psth,3,'omitnan'),'omitnan'); use = meanFRs > lowFR`
with `params.lowFR = 1` in the Figure 8 script. The agent added case-insensitive matching because
the labels are free text. It used the 520-vs-522 unit count as an explicit sanity check against
the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction, per spike: `trialtm − goCue[trial of that spike]`, then a floor-division into
the fixed bin grid. No interpolation, no offset, no per-trial re-referencing — `trialtm` and
`bp.ev.goCue` are already on the same behaviour clock and both relative to trial start. Spikes
falling outside the −3.0 … +2.5 s window are discarded.

ii.
```python
ALIGN_EVENT = "goCue"
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
...
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
in_window = (bins >= 0) & (bins < NT)
```

iii. This is `alignSpikes.m` with `params.alignEvent = 'goCue'`. The agent also verified as a
sanity check that `goCue` is finite on WC (autowater) trials in every exported session, "which
is important because the paper scripts treat this field as the shared 'go cue / water drop'
alignment reference" — it is recorded per session as `go_cue_wc_is_finite`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT = 1/100`) over a window of −3.0 to +2.5 s, giving 550 bins per trial. The grid
is built once at module level and shared by the neural data, the input, and all three camera
outputs, so no stream is rebinned or resampled relative to another. There is no second binning
pass: spikes are counted straight into the final grid, and the video streams are interpolated
straight onto it. `metadata['time_bin_size'] = 10.0`, `off_start = -3.0`, `off_end = 2.5`.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100

def build_time_axis():
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time

EDGES, TIME = build_time_axis()
NT = TIME.size           # 550
```

iii. `CONVERSION_NOTES.md` states this "matches the shared MATLAB pipeline: bin from −3.0 s to
2.5 s, dt = 0.01 s". The window is taken from `Figure8a_thru_c.m` (`params.tmin = -3;
params.tmax = 2.5;`). The 10 ms bin is *not* from that script — the Figure 8 script sets
`params.dt = 1/200`, as does `getDefaultParams.m`; `dt = 1/100` comes from the repository's
example/tutorial pipeline script (which pairs it with `tmin = -2.5` and a comment that reads
"use a 5 ms bin width", an internal inconsistency in the authors' own code). The agent did not
comment on the mismatch in the trajectory.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. It is the analysis time axis the conversion defines, i.e. the centre of each
of the 550 bins of the −3.0 … +2.5 s window around the go cue. Because the window is the same for
every trial, the input array is literally identical across all trials and sessions.

ii.
```python
EDGES, TIME = build_time_axis()      # TIME = bin centres, -2.995 ... 2.495
...
"input_names": ["time_from_go_cue_s"],
...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The go cue is the alignment event specified by the prompt and by `params.alignEvent`, so the
elapsed time relative to it is the natural continuous, time-varying input; the window bounds come
from the paper's Figure 8 parameter block.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond taking bin centres (`edges[:-1] + DT/2`). It is stored as `float32` with shape
`(1, 550)` per trial, in seconds, signed (negative before the go cue).

ii.
```python
edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. N/A — the axis is defined by the conversion, not derived from data.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spikes are placed into bin
`floor((trialtm − goCue − TMIN)/DT)` of the same `EDGES`, and the input is the centre of those
same bins, so element *k* of the input and column *k* of the neural matrix are the same 10 ms
interval. The three camera outputs are interpolated onto `taxis = TIME + ADVANCE_MOVEMENT` with
`ADVANCE_MOVEMENT = 0`, i.e. onto the identical axis, so all four streams share one time base.
The conversion asserts this in metadata as `all_time_axes_match: True`.

ii.
```python
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
taxis = TIME + ADVANCE_MOVEMENT
me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. `params.advance_movement = 0` in the authors' Figure 8 parameter block, so no lead/lag is
introduced between behaviour and neural streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. A single per-trial field: `obj.bp.R`, the right-trial flag. It is used verbatim as the lick
direction (`left = 0`, `right = 1`). `bp.L`, `bp.hit`, and `bp.miss` are read elsewhere in the
script but are not used to determine which port the animal actually licked.

ii.
```python
def build_output_constants(bp, keep_trials):
    lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
    ...
    return lick_direction, context, outcome
```
```python
"output_values": [
    ["left", "right"],
    ...
```

iii. `CONVERSION_NOTES.md`: "`lick_direction`: constant over time, `left=0`, `right=1`, taken from
`bp.R`". The agent treated `bp.R`/`bp.L` as the left/right trial label and mapped it straight onto
the prompt's `left = 0, right = 1` coding. Its early exploration did tabulate `R & hit`,
`R & miss`, `L & hit`, `L & miss` counts per session, so the four-way breakdown was visible to it,
but the distinction between instructed side and licked side is not discussed anywhere in the
trajectory or the notes.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Boolean-to-int cast, subset to the kept trials, and broadcast constant across all 550 time bins
of the trial. No use of the outcome to infer the licked port, so on error (`miss`) trials — 13.6%
of the exported dataset — the recorded value is the port the animal was *instructed* to lick, which
is the opposite of the port it actually licked. There is no third class, since the trials on which
the animal did not lick (`bp.no`) were already removed in 1-e.

ii.
```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
...
output_trials.append(
    np.vstack([
        np.full((1, NT), lick_direction[out_pos], dtype=np.int16),
        ...
    ])
)
```

iii. As above: the agent read `bp.R` as the lick direction directly. The prompt's requirement
"Can be time-varying or discrete values per trial… if at all possible, make it time-varying"
motivated tiling the per-trial scalar across the 550 bins.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`, which marks the water-cued (WC) trials where water is
delivered at a random port with no cues. Everything else is the delayed-response (DR) context.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. Throughout the authors' code the context split is written as `autowater` vs `~autowater`
(e.g. Figure 8's conditions `~autowater&~early&~no` = DR and `autowater&~early&~no` = WC), so the
flag is read directly. The agent also used the per-session WC/DR trial counts it tabulated as a
sanity check that both contexts are present in every exported session (WC fraction 0.22–0.43).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater → 0 (WC)`, otherwise `1 (DR)`, subset to the kept trials and
broadcast constant over the 550 bins. The negation-then-cast implements the prompt's
`WC = 0, DR = 1` coding.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
...
np.full((1, NT), context[out_pos], dtype=np.int16),
...
"output_values": [..., ["WC", "DR"], ...]
```

iii. Codes follow the prompt's `WC = 0, DR = 1`. Verified in the validator output as
`behavioral_context: {WC (0.289), DR (0.711)}`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. One per-trial flag, `obj.bp.hit`. `bp.miss` and `bp.no` are read in the trial-filter, but the
outcome value itself is just `hit`: after filtering, every surviving trial is either a hit or a
miss, so `hit == 0` implies miss.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. `CONVERSION_NOTES.md`: "`outcome`: constant over time, `incorrect=0`, `correct=1`, derived
from `bp.hit`". The agent verified the three flags partition the trials when it tabulated
`hit/miss/no` counts per session.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Cast to int, subset to kept trials, broadcast over 550 bins. Two classes only:
`incorrect = 0` (miss), `correct = 1` (hit). Ignore trials do not appear because they were dropped
in 1-e. Resulting distribution: 13.6% incorrect / 86.4% correct.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
...
np.full((1, NT), outcome[out_pos], dtype=np.int16),
...
"output_values": [..., ["incorrect", "correct"], ...]
```

iii. The two-class coding is exactly what the prompt the agent received specified
("Outcome (incorrect = 0, correct = 1, per-trial)"), and it forced the removal of the
no-response trials described in 1-e.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, using **seven** tongue landmarks across both cameras:
`tongue`, `left_tongue`, `right_tongue` from the side camera (view 0) and `top_tongue`,
`topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` from the bottom camera (view 1). For each,
`traj[view]['ts'][trial][:, 0:2, feat]` (x and y) and `traj[view]['frameTimes'][trial]` are read,
plus `traj[view]['NdroppedFrames'][trial]` as a trial-level video validity flag. Alignment also
requires `obj.bp.ev.goCue`, `obj.bp.ev.bitStart`, `obj.sglx.bitcode.bitstart`, and `obj.sglx.fs`.

ii.
```python
TONGUE_FEATURES = [
    (0, "tongue"), (0, "left_tongue"), (0, "right_tongue"),
    (1, "top_tongue"), (1, "topleft_tongue"), (1, "bottom_tongue"), (1, "bottomleft_tongue"),
]
...
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```
```python
ts = get_trial_ts(cam, trix)[:, :2, feat_idx]
frame_times = get_trial_frame_times(cam, trix)
```

iii. The landmark list is exactly the tongue subset of the authors'
`params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'}, {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_nostril','bottom_nostril','top_paw','bottom_paw'}}`
in `Figure8a_thru_c.m`; the agent enumerated the actual `featNames` in the data to confirm they
exist. `metadata` records it as "Mean speed across side-camera tongue/left_tongue/right_tongue and
bottom-camera top/bottom tongue landmarks."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. A port of `findPosition.m` → `findVelocity.m`. Per trial and per landmark: (1) skip the trial
entirely if `NdroppedFrames` is NaN; (2) convert frame times to go-cue time (see 7-d) and linearly
interpolate x and y onto the 550-bin axis with `np.interp`, leaving NaN outside the frame range —
for tongue features the NaNs from low-likelihood frames (the authors already NaN out x/y below
likelihood 0.9) are *not* filled; (3) take `np.gradient` of the interpolated x and y (pixels per
bin, no division by dt, no baseline subtraction for tongue); (4) replace NaN velocities with 0,
which is the authors' explicit convention "set tongue velocity to 0 if not visible";
(5) speed = `sqrt(xvel² + yvel²)`; (6) `np.nanmean` the speed across all seven landmarks. There is
no per-camera normalisation, so the two cameras' different pixel scales enter the mean unweighted,
and a landmark that is invisible contributes a hard 0 to the mean.

ii.
```python
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
if not is_tongue:
    xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
```
```python
xvel[:, trix] = np.gradient(tsinterp[:, 0])
yvel[:, trix] = np.gradient(tsinterp[:, 1])
if not is_tongue:
    ...
else:
    xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
    yvel[:, trix] = np.nan_to_num(yvel[:, trix], nan=0.0)
```
```python
speeds.append(np.sqrt(xvel**2 + yvel**2))
stacked = np.stack(speeds, axis=0)
out = np.nanmean(stacked, axis=0)
out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. `CONVERSION_NOTES.md`: "Kinematic alignment matches `findPosition.m` and `findVelocity.m`:
interpolate tracked positions onto the neural time axis; preserve the paper code's special tongue
handling". The agent read both MATLAB functions and reproduced their branch structure, including
the tongue/non-tongue split. The averaging across landmarks is its own addition, needed to reduce
the multi-landmark trajectory set to the single scalar the decoder output requires.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes with a per-session threshold. The threshold is the **50th percentile of the
strictly positive tongue speeds** over kept trials and all time bins of that session — the zeros
that represent "tongue not visible" are excluded from the percentile but are still classified,
landing in class 0. Values `>= threshold` are class 1, everything else class 0. There is no
`not visible` class. This yields roughly 83% class 0 / 17% class 1 rather than a 50/50 split.

ii.
```python
def binarize_session_signal(signal, keep_trials, ignore_zeros_for_threshold=False):
    kept = signal[:, keep_trials]
    threshold_source = kept
    if ignore_zeros_for_threshold:
        threshold_source = kept[kept > 0]
    if np.size(threshold_source) == 0:
        threshold = 0.0
    else:
        threshold = float(np.nanpercentile(threshold_source, 50))
    binary = (kept >= threshold).astype(np.int64)
    return binary, threshold
```
```python
tongue_binary, tongue_threshold = binarize_session_signal(
    tongue_speed, keep_trials, ignore_zeros_for_threshold=True,
)
```

iii. This was a deliberate mid-run correction. The agent first used a plain 50th percentile, found
"zero tongue thresholds 11 ['JEB6_2021-04-18', …]" — i.e. the threshold collapsed to 0 in 11 of 12
sessions — and reasoned: "the MATLAB pipeline uses `0` as a placeholder when the tongue is not
visible, so taking the session median over all samples collapses the threshold to zero in most
sessions. I'm treating those placeholder zeros as 'below threshold' rather than using them to
define the percentile." It patched only the tongue rule and left paw and motion energy on the
plain median.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are on the video clock, which leads the behaviour clock, so a session-constant
offset is removed first: the mode of `sglx.bitcode.bitstart / sglx.fs` minus the mode of
`bp.ev.bitStart`. Each trial's frame times then become
`frameTimes − vidshift − goCue[trial]`, and x/y are linearly interpolated from those times onto the
shared 550-bin axis (`taxis = TIME`), so bin *k* of the tongue output is the same interval as bin
*k* of the neural matrix. Frames outside the window produce NaN via `left=np.nan, right=np.nan`.
The offset is computed once per session and passed through.

ii.
```python
def find_video_offset(obj: dict) -> float:
    bit_start = mode_scalar(obj["bp"]["ev"]["bitStart"])
    bitcode = obj["sglx"]["bitcode"]
    bitstart = as_array(bitcode["bitstart"], np.float64).ravel()
    fs = float(np.asarray(obj["sglx"]["fs"]).reshape(-1)[0])
    vid_file_offset = mode_scalar(bitstart) / fs
    return float(vid_file_offset - bit_start)
```
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
```

iii. A direct port of the authors' `findVideoOffset.m` and of the `interp1(traj(trix).frameTimes -
vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)` line in `findPosition.m`, which the agent quoted
in its exploration.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` DeepLabCut tracking, bottom camera only (view 1), using **both** paw
landmarks: `top_paw` and `bottom_paw`. Same auxiliary fields as the tongue (`frameTimes`,
`NdroppedFrames`, the bitcode offset, `bp.ev.goCue`).

ii.
```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
...
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. Both are in the authors' `params.traj_features` bottom-camera list, and the agent verified
they exist in the data's `featNames` ("cam 2 featnames … ['top_paw'], ['bottom_paw'] …").
`metadata`: "Mean speed across bottom-camera top_paw and bottom_paw landmarks."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same `findPosition`/`findVelocity` port as the tongue but down the **non-tongue** branch:
(1) interpolate x and y onto the 550-bin axis; (2) `fill_nearest_1d` the interpolated positions —
gaps take the nearest tracked value, and a trial with no valid samples at all becomes all zeros;
(3) `np.gradient` for x and y velocity; (4) subtract the trial's baseline derivative
`basederiv = nanmedian(diff([x y]))`, with `basederiv[0]` subtracted from **both** x and y (this
reproduces an apparent index bug in the authors' `findVelocity.m`, which the agent kept
deliberately and flagged in a comment); (5) `fill_nearest_1d` the velocities again;
(6) speed = `sqrt(xvel² + yvel²)`; (7) mean across the two paw landmarks. No normalisation; units
are pixels per 10 ms bin.

ii.
```python
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
if np.isnan(basederiv[0]):
    basederiv = np.array([0.0, 0.0])
xvel[:, trix] = np.gradient(tsinterp[:, 0])
yvel[:, trix] = np.gradient(tsinterp[:, 1])
if not is_tongue:
    # Match the MATLAB implementation exactly, including the y-axis subtraction term.
    xvel[:, trix] = xvel[:, trix] - basederiv[0]
    yvel[:, trix] = yvel[:, trix] - basederiv[0]
    xvel[:, trix] = fill_nearest_1d(xvel[:, trix], fill_value=0.0)
    yvel[:, trix] = fill_nearest_1d(yvel[:, trix], fill_value=0.0)
```

iii. `CONVERSION_NOTES.md`: "preserve the MATLAB baseline-subtraction behavior for non-tongue
velocities". The agent quoted `findVelocity.m` — `xvel(:,i) = xvel(:,i) - basederiv(1); yvel(:,i) =
yvel(:,i) - basederiv(1);` and `fillmissing(...,'nearest')` — and chose fidelity to the released
code over correcting it.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two classes at the per-session 50th percentile of the paw speed over all kept trials and all
time bins, **including** the nearest-filled values (`ignore_zeros_for_threshold` is not set).
`>= threshold` is class 1, else class 0. Thresholds range 0.20–0.63 px/bin across the 12 sessions
and the split is exactly 50/50 in every session. No `not visible` class.

ii.
```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
```
```python
threshold = float(np.nanpercentile(threshold_source, 50))
binary = (kept >= threshold).astype(np.int64)
```

iii. Straight from the prompt's specification ("0: < 50th percentile, 1: >= 50th percentile",
per-session). Unlike the tongue, the paw has no placeholder-zero problem — the agent checked the
thresholds explicitly ("paw min/max 0.2033… 0.6287…") and left the plain median in place.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the same session-constant `vidshift` from `find_video_offset` is
subtracted from `frameTimes`, then that trial's `goCue`, and the result is interpolated onto the
same 550-bin `taxis`. Because both paw landmarks live on the bottom camera, the bottom camera's
own frame times are used (`find_position` always reads `cam = obj["traj"][view_index]` for the
view the feature belongs to).

ii.
```python
cam = obj["traj"][view_index]
...
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
```

iii. Same port of `findPosition.m`/`findVideoOffset.m` as in 7-d; no feature needs separate
handling because the view index is carried in `PAW_FEATURES`/`TONGUE_FEATURES`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The companion file `motionEnergy_<anm>_<date>.mat`, loaded with `scipy.io.loadmat`, field
`me.data` (with one level of unwrapping if `me.data.data` exists), which holds one trace per trial
with one value per side-camera frame. `me.moveThresh` is also read and stored in the returned dict
but never used. The in-object copy `obj.me` is not used. Timing comes from
`obj.traj[0]['frameTimes']` (the side camera) plus the bitcode offset and `bp.ev.goCue`.

ii.
```python
mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
me = mat["me"]
raw_data = me.data
if hasattr(raw_data, "data"):
    raw_data = raw_data.data
raw_trials = list(np.asarray(raw_data, dtype=object).ravel())
...
cam = obj["traj"][0]
```

iii. `CONVERSION_NOTES.md`: "Motion energy loading matches `loadMotionEnergy.m`: load the companion
`motionEnergy_*.mat`". The double-wrap guard mirrors that file's `if isstruct(me.data), me.data =
me.data.data; end`. The agent added the `len(raw_trials) != ntrials` assertion as a sanity check.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling — the value is already one scalar per frame (the paper computes it
upstream as the frame-to-frame pixel difference reduced to its 99th percentile across pixels). Per
trial the trace is linearly interpolated from corrected frame times onto the 550-bin axis, and any
remaining NaN (outside the frame range, or the whole trial if frame times are unusable) is filled
by `fill_nearest_1d`, falling back to 0 when the trial has no valid sample at all.

ii.
```python
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. `CONVERSION_NOTES.md`: "interpolate onto the neural time axis; fill missing values with
nearest samples" — the same `interp1` + `fillmissing('nearest')` pattern the authors use for the
kinematic streams.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two classes at the per-session 50th percentile over kept trials and all time bins, including
nearest-filled values. `>= threshold` is class 1, else class 0. Thresholds run 10.5–20.0 across the
12 sessions and each session splits 50/50. No `no video` class; `me.moveThresh` from the file is
loaded but not used as a threshold.

ii.
```python
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

iii. Directly from the prompt's per-session 50th-percentile rule. The agent checked the spread of
the resulting thresholds ("motion min/max 10.5157… 20.0483…") before accepting them.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking. Motion energy has one value per side-camera frame,
so `obj.traj[0]['frameTimes'][trial]` is used, corrected by the session `vidshift` and the trial's
`goCue`, then interpolated onto `taxis`. The `vidshift` computed inside `load_motion_energy` is the
same value reused for the tongue and paw, so all three camera streams sit on one clock.

ii.
```python
cam = obj["traj"][0]
vidshift = find_video_offset(obj)
for trix, trial_me in enumerate(raw_trials):
    ...
    old_t = frame_times - vidshift - align_times[trix]
    resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
...
return {"data": resampled, "moveThresh": ..., "vidshift": vidshift}
```

iii. Same `findVideoOffset.m` logic as 7-d; motion energy is computed from the side-camera video,
so that camera's frame times are the right index.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The governing philosophy is **fill rather than flag** — every gap is converted into a numeric
value so that the two-class outputs remain well defined — with a small number of hard failures for
structural problems.
- **Whole trial's video invalid** (`NdroppedFrames` is NaN): the trial is skipped in
  `find_position`, leaving its x/y columns all NaN. For the tongue that becomes speed 0 (class 0);
  for the paw `fill_nearest_1d` finds no valid sample and writes 0.0 everywhere (class 0).
- **Frame times missing or all-NaN**: a synthetic 400 fps axis is invented,
  `arange(1, n+1)/400`, and shifted by a hard-coded `0.5` s instead of the session's `vidshift`.
- **Untracked frames** (DeepLabCut likelihood low, x/y already NaN in the file): tongue → velocity
  set to 0; paw and motion energy → nearest-neighbour fill.
- **Too few interpolation points** (`<2` finite samples): `interp_with_nan` returns all-NaN, which
  then falls through to the same fill rules.
- **Degenerate baseline derivative** (all-NaN `diff`): `basederiv` is forced to `[0, 0]`.
- **Residual NaN/inf** in the averaged speed: `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`.
- **Missing `stim` struct** in older sessions: treated as all-false.
- **Hard failures**: motion-energy trial count ≠ `Ntrials`, fewer than 10 units after filtering, or
  fewer than 2 kept trials all raise `ValueError` rather than silently degrading.
No check exists for trials recorded after the probe stops; such trials would enter the dataset as
550 bins of zero firing.

ii.
```python
dropped = np.asarray(cam["NdroppedFrames"][trix]).reshape(-1)
if dropped.size and np.isnan(dropped[0]):
    continue
...
use_fallback = frame_times.size == 0 or np.all(~np.isfinite(frame_times))
if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]
```
```python
def fill_nearest_1d(arr, fill_value=0.0):
    ...
    if not mask.any():
        out[:] = fill_value
        return out
```
```python
def get_stim_enable(bp: dict) -> np.ndarray:
    stim = bp.get("stim")
    if isinstance(stim, dict) and "enable" in stim:
        return as_array(stim["enable"], bool).ravel()
    return np.zeros(int(bp["Ntrials"]), dtype=bool)
```

iii. The fills are taken from the authors' own code — `fillmissing(...,'nearest')` for non-tongue
features, "set tongue velocity to 0 if not visible" for the tongue, and the
`traj(trix).frameTimes = (1:size(traj(trix).ts,1))./400` fallback in `findPosition.m`. The
two-class output specification the agent was given leaves no `not visible` code, so a numeric value
had to be assigned to every bin. The `0.5` s used in place of `vidshift` in the fallback branch is
not explained anywhere in the notes or the trajectory.

## 11-a. What are the most time-consuming steps of the code?

i. Reading dominates: `mat73.loadmat(spec.data_path)` materialises the entire `obj` tree for each
session — including spike waveforms, all `sglx` index arrays, and every tracked feature — and is by
far the largest single cost. After that, in rough order: `build_aligned_trialdat`, which loops over
every quality-passing unit and for each allocates an `(ntrials, 550)` array and runs a 550-tap
`np.convolve` per trial column; the nine `find_position` passes (7 tongue + 2 paw landmarks), each
interpolating x and y for every trial; and `build_sample_dataset`, which `copy.deepcopy`s the whole
in-memory dataset. Total wall time was roughly a minute for the 12 sessions.

ii.
```python
obj = mat73.loadmat(spec.data_path)["obj"]
```
```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...
    trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```
```python
def subset_data(data, session_trial_indices):
    subset = copy.deepcopy(data)
```

iii. Not discussed in the trajectory; the agent never profiled the script. The data has to be read
once regardless, so the loading cost is irreducible; the deepcopy exists only to build the
`sample_data.pkl` artifact its prompt required.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- The per-unit loop in `build_aligned_trialdat`. The spike counting could be a single
  `np.histogram2d`/`bincount` over (unit, trial, bin) for all units at once instead of one
  `np.add.at` per unit.
- The per-column `np.convolve` inside `my_smooth`. `scipy.ndimage.convolve1d` or
  `scipy.signal.fftconvolve` along an axis would replace the Python loop over up to 550 columns.
- The per-trial loops in `find_position`, `find_velocity`, and `load_motion_energy`. These are the
  hardest to remove since each trial has a different number of camera frames, but `find_velocity`
  operates on already-rectangular `(550, ntrials)` arrays and could be fully vectorised — the
  `np.gradient`, the baseline subtraction, and the NaN handling all apply column-wise.
- The per-condition loop in `apply_low_fr_filter`, which builds a full `(550, n_units, 7)` PSTH
  tensor only to reduce it to one scalar per unit; the same number could be obtained by averaging
  `trialdat` over the union of condition trials directly.

ii.
```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...
    counts = np.zeros((ntrials, NT), dtype=np.float64)
    np.add.at(counts, (spike_trials, bins), 1.0)
```
```python
for col in range(arr_filt.shape[1]):
    out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")
```
```python
for trix in range(xpos.shape[1]):
    tsinterp = np.column_stack([xpos[:, trix], ypos[:, trix]])
```

iii. The per-trial video loops mirror the structure of the MATLAB source the agent was porting, so
keeping them made the port auditable. Since loading dominates the runtime, none of these would
change the total materially.

## 11-c. What processing does the code repeat multiple times?

i. Several things are recomputed:
- `get_trial_frame_times(cam, trix)` is re-read once per feature per trial — nine times per trial
  for the same two cameras — and `find_video_offset` results are at least reused via `me["vidshift"]`.
- `get_feat_names(cam)` scans every trial's `featNames` until it finds a non-empty one, and is
  called afresh for each of the nine features.
- The `bp` boolean masks (`hit`, `miss`, `early`, `no`, `autowater`, `stim`) are extracted three
  separate times, in `build_low_fr_conditions`, `build_keep_trial_mask`, and
  `build_output_constants`.
- `basederiv` is computed for every trial of every feature, including tongue features where the
  result is discarded (this is the source of the `All-NaN slice encountered` warning in the
  conversion log).
- `copy.deepcopy(data)` in `subset_data` duplicates the full dataset, and
  `copy.deepcopy(subset["metadata"])` duplicates the metadata again inside it.

ii.
```python
def find_dlc_feat_index(obj, view_index, feat_name):
    cam = obj["traj"][view_index]
    feat_names = get_feat_names(cam)
```
```python
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
if np.isnan(basederiv[0]):
    basederiv = np.array([0.0, 0.0])
```
```python
subset = copy.deepcopy(data)
...
subset["metadata"] = copy.deepcopy(subset["metadata"])
```

iii. Not addressed in the trajectory. The redundancy is cheap relative to file loading, and the
duplicated `bp` extraction keeps each helper self-contained and independently testable.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- `mat73.loadmat` reads the whole `obj`, so spike waveforms (`clu.spkWavs`), `clu.tm`, the full
  `sglx` index arrays, the second probe of two-probe sessions, and every non-tongue/non-paw tracked
  feature (`jaw`, `nose`, `trident`, `nostril`, `lickport`) are materialised and never used.
- `SIDE_FEATURES` and `BOTTOM_FEATURES` are defined at module level and referenced nowhere.
- `me.moveThresh` is parsed and stored in the returned dict but never used for thresholding.
- Smoothed firing rates are computed for **all** quality-passing units before the low-FR filter
  runs, so the two units that fail the 1 Hz cut are fully processed and then thrown away; the
  `(550, n_units, 7)` PSTH tensor built for that filter is discarded after one reduction.
- `basederiv` is computed and discarded for all seven tongue features.
- For the tongue the y-positions are interpolated and differentiated even where x is NaN, and every
  landmark's speed is computed at full resolution before the median split reduces it to one bit.
- `sample_data.pkl` (25 MB) plus its deepcopy and the `conversion_stats.json` summary are extra
  artifacts not consumed by the decoder, though the agent's own prompt required the sample file.

ii.
```python
SIDE_FEATURES = ["tongue", "left_tongue", "right_tongue", "jaw", "trident", "nose"]
BOTTOM_FEATURES = [...]        # never referenced
```
```python
return {
    "data": resampled,
    "moveThresh": float(np.asarray(me.moveThresh).reshape(-1)[0]),   # never used
    "vidshift": vidshift,
}
```
```python
trialdat = build_aligned_trialdat(probe, go_cue, ntrials)   # all quality units
low_fr_keep, mean_frs = apply_low_fr_filter(trialdat, bp)
trialdat = trialdat[:, low_fr_keep, :]                      # then discard
```

iii. Computing rates before the low-FR filter is exactly what the MATLAB pipeline does
(`getSeq` → `removeLowFRClusters` operates on the already-built `trialdat`/`psth`), so it was
retained for fidelity; the unused constants and `moveThresh` are leftovers from the exploration
phase. None of this is discussed in the trajectory.
