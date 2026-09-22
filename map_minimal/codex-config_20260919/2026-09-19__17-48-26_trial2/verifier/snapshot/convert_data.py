#!/usr/bin/env python3
"""Convert the MAP NWB release to the neural-decoder interchange format.

The conversion follows the classifier-QC path used by the accompanying papers:
only units whose NWB ``classification`` is ``good`` and which have a CCF
annotation are retained.  The one NWB file without classifier results is skipped,
giving the 173 curated behavioral sessions described in the data paper.

Trials with automatic or free water are training/assistance trials and are
excluded, as in the reference analysis.  Ordinarily the paper also excludes
early-lick, no-response, and photostimulation trials.  Those three exclusions are
deliberately not made here because the requested decoder variables explicitly
require those trials.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import pickle
import zlib
from pathlib import Path

import h5py
import numpy as np


OFF_START = -2.5
OFF_END = 1.5
BIN_S = 0.050
N_TIME = int(round((OFF_END - OFF_START) / BIN_S))
BIN_EDGES = OFF_START + np.arange(N_TIME + 1) * BIN_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
TONGUE_LIKELIHOOD_CUTOFF = 0.90

# Broad regions used by preprocessing_DJ_2022Aug.py, in the same order.  The
# compressed lookup maps the Allen CCF annotation strings occurring in this NWB
# release to their corresponding ancestor.  It was generated from Allen Mouse
# Brain structure graph 1.  MOp/MOs are called ALM (the paper's medial/lateral
# ALM definition), while orbital cortex remains its own reference-code region.
BRAIN_REGIONS = [
    "ALM", "Medulla", "Midbrain", "Striatum", "Thalamus", "Pons",
    "Cerebellum", "Hypothalamus", "Hippocampus", "Orbital",
    "OtherCortex", "Olfactory", "CorticalSubplate", "Pallidum",
]
_REGION_LOOKUP_B85 = """c-pO7OLN>d68<YJot!GUQg()W$j#AOiE1TBWocEZy@4b&g9-_70dis`mH)mC@Cl*;g2O#|H2A(ApwZ|T;J>au<vCSK$hSn4uOx5ur>rfr>!f0hBxO>PH!81wy!v$*zPi5pbd-d*C6Ni^%KS?tCD%zIqykD!WOLmSlaJs3pvldN${QgY`fVFM-9!Ed*~QMH*|ks&l}yNsmySzFck(wV;7qyM?m7L#F7}q?WBha%8(TW>RYa<{V?hQ!;GP=ENYhqxP|o-fexsqAjB^7!G{YEr7EiYo6#C)G3Gg;AX{(kee*o5!HZ*TQ=mdOVMJ5U315WhQo)FI`P<sU*2Sh~+HX*Eq#=z$b3*gU8weU};jNsmhL1O<CiFi$#L8|owKM^IgiZysLsBvb5&(0$-_0O=JluvC|YudU5`$%6TVxb<7fL=+LFTyV%Y7MmL2&_{!O_V;42@osBNy)V0A4yrV!ko|>(iWm_8&XkjpWT~&xyVFvQg<GspGO>PQ0*pCuMIzEkZPG2^_bZ}vMqm&BWXs<5`Oi^`5!P+k`}bv4Y*rz19~k<Qol)3vBD_cruURY4IEsIgc*Kt$okul%WwZl^OBuu^6usCJ9msX+KC<Cq*iR_{b`cJ_?VXb!s=S&q^ixz7mP}jG&kEG1+PW<RtZ_3eDKaWf&DS#BIo|(;P{geI^Bdu9Ci?%`QViR@Y06P<OqLoD$ja5A865*C7C$djtA&qxkt)b-UXOkRCP(e!3h(R+Ge;>0IL-n#{U$2pD=X-Bf%xW0(sufqW?U;^`OD`EOkYuY`F%jPvB_gjUD+uYf&dd*7Co<_iQ+hL$=xjbT|1k2fxT9dlj<sMO5Up;<WY9@tIrmvc2Wp(CtGMMKfwx@V)^|v+|Zxz1Oq6W3<^@lC!0qF951xM`k4@me09mv@DXFPam8K3?m=CBR}9H#(HkycxZ$C<*gRYWR$m3aGgYf>!G2<8N;PfD8!;}TV8ZFt%m4i1*)If5hB0Pe%725NcC8EVlb8dT#5{&)m?-J)1kvoDSCt$-PqyVfioV2L%brRi&G&Td7y__U>XZVRO&;zr*t$BSg}5Gq+x<PhNI2oSPEsQ&y&(EGjgozj=%PWfB;3p)|nK@&TMV3g)+vC3{wOIZ~`muov$@d0D+6_517Y*45nmN1`c&4?A>WwAxLczPKXQ*ycf8Q!cEz1<2P87ZGn#oTmA^$J{d@ihqTxALH`o)iTzIH``cDgoor3ON5Qp_dtcIz`|1IX%@dpBfIpFp>WbSlI$B-7bFL*0(5VoJ($Pqhv<tLh>CQGm>PL=z@?d=wo3?IXDq1)5f>Odf_4n`0hoKt$4$0<y5*C3IVVo(%pv-Gk_g(RxymJ%cdvKkQ0N>qt2RtJxxqEs4;*RC85pX(N+qfPq-G$G<gQxbhm)@hWOwC*I2RL?#702<n3VIT5>;2%4gs_*evg;2%H-a6HfwmB!Kfz3p(_kw(lC;p+j5RFIUT6OiH<%dQwf;&9*=DlO4sGTaQRu*wy!*{L8DDyp!G5$A?Bl}zzDyeZ3IWd4cGG&k*n0{;m!DW$4SzE?HZJC6^M{1Cq!ni>pP#G+(rYsdoPrSB=v^A2&fF!b0jRy+!mw<B$Y4U!15r>3c1;6hDt25SYo;#WSyWNeXwzm8c}r?<$FDMDO|Sqkt>+FEh41sh4UBXnzRO0o5V8Pa=D}hngusLIF-KQyRji*CR<ECzUc_1~ZL9{Y!r+le(w$T*LG2Hk9(5DmnG4l`SE6#qQ!OyZwOYvO5>AIi^Ucojq%>i8>9sVVZ?<{ze2o}SzZcMEV;VML7P*oJ8kP6cx01IhP*4SGD<B0)3U+3yuT8?DqEb2_ompg3@!0ohL=E*U;}j1hkKmQbe_Ozsdx)bXZ!Tcg?!osMFp{<$LIzU-h~2Tk%S8_msbe08ehboqG8Q};a<2hQVIxF27hAz<U?TL{(aO$^aLi^9^BD0^)Uig}Qk;cMmj&U=B@G0dH=($4O6NyOkzx|rEfj}s77ol1AglB_jcoZ2@UC0cVee<|_S5}C5S2z?%yL#inFCUyA`R52Tutl9hy+a}uezt@zari$0qf*!m2ucDA>AzjH5|W)Qv5;90m&{^ge&2&lb`P7?^Z$JOkn~l(Ke?`)n}0pms8WrsT=iWbGNCA!{3*k**SBoj_2IN#mw|#=0*iu)NT50WXpfQY{C|G<BVHQlwwCtFLtM{gu_mLx|6?K1sk`nt?Qkh*p+bD$xnCkx9ZRp;MZZ)W10XsFgFa1Ohi{qx()Plcjxv<zgm6uS=RV<nL&rM(XYA{r64=zY@AZ$`^HFWdMOdnU@S9>VvbLW#s_!X=N*<2xr7xVajS`Y+8E1>-SD_4jj_x~eW6rDOD9off)#kZ`$TiWi)E=W>8-02La2HD62hfXs96mYgk>{W-H2CkXVPvC&03I8Ojo|1zRXjN_4<K)q~;btzb9)8tmU)LWZm^2dMmZg7BcXwD_0Cp=w<(tw~&UBEw9dWgTZd9_05eSCVXKRN3E^a5=vDK;SD38eJ->r8)A18XAsW7*<72}2+5mK*;62Ku~TyfMX$F^p4!^Ie(y{i3Qc`&ty^=nHchVab(-}n>jK#PX8Zpp=#LX?Xtbp6#v>N3^p~&9yy@9A;XehR{nrHyy>?b0B`T?Yj-s!4vEtPm3?iwAONr^FL?nj9GBb=R$?TpLjU>-R2dBjJ@|Ks7^Oomx_won3$)DPs&$OzI`-6O6fJx|gGpp}%B1rDqo>Bky3@+2$%v%YhzW!fcO+4#MFHmMxvF>u!ccr@U0{=dD_x|qn=F9NhxPOrSen^a>8n`1nt}hvcBRYP$;ED3U>E4BiO;4h?ao5AcLf(#LgsJf}X~y&5UV^_cl~cK1Q~d{AAb4uPHpt5vd%{sA-6{o{#5!kr`u_eOp_)Ic"""
REGION_LOOKUP = json.loads(
    zlib.decompress(base64.b85decode(_REGION_LOOKUP_B85)).decode("utf-8")
)
REGION_TO_INDEX = {name: i for i, name in enumerate(BRAIN_REGIONS)}


def _scalar_string(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _numeric_or_nan(values: np.ndarray) -> np.ndarray:
    """Parse NWB string columns containing numbers and the sentinel ``N/A``."""
    return np.array(
        [np.nan if value == "N/A" else float(value) for value in values],
        dtype=np.float64,
    )


def _bin_good_units(
    units: h5py.Group, good_indices: np.ndarray, go_times: np.ndarray
) -> np.ndarray:
    """Return firing rates with shape (trial, good_unit, time_bin)."""
    n_trials = len(go_times)
    rates = np.empty((n_trials, len(good_indices), N_TIME), dtype=np.float32)
    all_spikes = units["spike_times"][:]
    spike_ends = units["spike_times_index"][:].astype(np.int64)
    spike_starts = np.r_[0, spike_ends[:-1]]
    window_starts = go_times + OFF_START
    window_ends = go_times + OFF_END

    # Windows do not overlap in this task, so each spike has at most one trial.
    for out_unit, unit_index in enumerate(good_indices):
        spikes = all_spikes[spike_starts[unit_index] : spike_ends[unit_index]]
        trial_index = np.searchsorted(window_starts, spikes, side="right") - 1
        valid = trial_index >= 0
        trial_index = trial_index[valid]
        spikes = spikes[valid]
        valid = spikes < window_ends[trial_index]
        trial_index = trial_index[valid]
        spikes = spikes[valid]
        time_index = np.floor(
            (spikes - window_starts[trial_index]) / BIN_S + 1e-12
        ).astype(np.int64)
        valid = (time_index >= 0) & (time_index < N_TIME)
        flat_index = trial_index[valid] * N_TIME + time_index[valid]
        counts = np.bincount(flat_index, minlength=n_trials * N_TIME)
        rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_S
    return rates


def _tongue_classes(
    behavior: h5py.Group, absolute_bin_times: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """Sample preceding video frames and discretize visible tongue y per session."""
    tracking = behavior["Camera0_side_TongueTracking"]
    tracking_data = tracking["data"][:, 1:3]  # y, DeepLabCut likelihood
    tracking_times = tracking["timestamps"][:]
    y_all = tracking_data[:, 0]
    likelihood_all = tracking_data[:, 1]
    visible_all = (
        np.isfinite(y_all)
        & np.isfinite(likelihood_all)
        & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
    )
    if not np.any(visible_all):
        raise ValueError("session has no visible tongue samples")
    percentile_40, percentile_60 = np.percentile(y_all[visible_all], [40, 60])

    frame_index = np.searchsorted(
        tracking_times, absolute_bin_times.ravel(), side="right"
    ) - 1
    in_range = (frame_index >= 0) & (frame_index < len(tracking_times))
    safe_index = np.clip(frame_index, 0, len(tracking_times) - 1)
    sampled = tracking_data[safe_index]
    y = sampled[:, 0]
    likelihood = sampled[:, 1]
    visible = (
        in_range
        & np.isfinite(y)
        & np.isfinite(likelihood)
        & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
    )
    result = np.full(len(frame_index), 3, dtype=np.int8)
    result[visible & (y < percentile_40)] = 0
    result[visible & (y >= percentile_40) & (y <= percentile_60)] = 1
    result[visible & (y > percentile_60)] = 2
    return (
        result.reshape(absolute_bin_times.shape),
        float(percentile_40),
        float(percentile_60),
        float(np.mean(visible_all)),
    )


def convert_session(path: str) -> dict | None:
    with h5py.File(path, "r") as nwb:
        units = nwb["units"]
        classification = units["classification"]
        # One released session has NaNs here instead of classifier predictions.
        if classification.dtype.kind not in "OSU":
            return None
        class_values = classification.asstr()[:]
        annotations = units["anno_name"].asstr()[:]
        annotation_present = np.array([bool(x.strip()) for x in annotations])
        good = (class_values == "good") & annotation_present
        good_indices = np.flatnonzero(good)
        if len(good_indices) == 0:
            return None

        trials = nwb["intervals/trials"]
        go_times = nwb[
            "acquisition/BehavioralEvents/go_start_times/timestamps"
        ][:]
        if len(go_times) != len(trials["id"]):
            raise ValueError(f"{path}: go cue and trial counts differ")

        # The NWB trial table can extend beyond the electrophysiology recording.
        # is_good_trials is indexed only over observed ephys trials.  Requiring all
        # retained units to be valid also avoids treating an unobserved unit as a
        # neuron that genuinely fired zero spikes.
        unit_trial_validity = units["is_good_trials"][:][good_indices]
        n_ephys_trials = unit_trial_validity.shape[1]
        if n_ephys_trials > len(go_times):
            raise ValueError(f"{path}: more ephys trial columns than behavior trials")
        neural_valid = np.zeros(len(go_times), dtype=bool)
        neural_valid[:n_ephys_trials] = np.all(unit_trial_validity, axis=0)

        # Reference-code assistance-trial exclusions.  Other reference exclusions
        # are intentionally overridden by the requested decoder variables.
        assistance = (trials["auto_water"][:] != 0) | (trials["free_water"][:] != 0)
        keep = neural_valid & ~assistance
        kept_indices = np.flatnonzero(keep)
        if len(kept_indices) < 2:
            return None

        firing_rates_all = _bin_good_units(units, good_indices, go_times)
        # A recording can end during the final nominally valid trial.  Such a
        # trial has no population samples at all and is not decoder data.
        neural_nonzero = np.any(firing_rates_all != 0, axis=(1, 2))
        empty_neural_trials = neural_valid & ~neural_nonzero
        keep &= neural_nonzero
        kept_indices = np.flatnonzero(keep)
        if len(kept_indices) < 2:
            return None
        firing_rates = firing_rates_all[keep]
        kept_go = go_times[keep]
        absolute_bin_times = kept_go[:, None] + BIN_CENTERS[None, :]

        # presample_stop is the first tone/sample onset and, unlike sample_start,
        # has exactly one entry per trial even when an early lick replays epochs.
        tone_onsets = nwb[
            "acquisition/BehavioralEvents/presample_stop_times/timestamps"
        ][:][keep]
        time_from_tone = absolute_bin_times - tone_onsets[:, None]

        onset = _numeric_or_nan(trials["photostim_onset"].asstr()[:])[keep]
        duration = _numeric_or_nan(trials["photostim_duration"].asstr()[:])[keep]
        trial_starts = trials["start_time"][:][keep]
        stimulation_start = trial_starts + onset
        stimulation_end = stimulation_start + duration
        photostim = (
            (absolute_bin_times >= stimulation_start[:, None])
            & (absolute_bin_times < stimulation_end[:, None])
        )

        tongue, p40, p60, visible_fraction = _tongue_classes(
            nwb["acquisition/BehavioralTimeSeries"], absolute_bin_times
        )

        instruction = trials["trial_instruction"].asstr()[:][keep]
        outcome_text = trials["outcome"].asstr()[:][keep]
        early_text = trials["early_lick"].asstr()[:][keep]
        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
        outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int8)
        early = (early_text == "early").astype(np.int8)

        # Outcome and instruction encode the report robustly even when the raw
        # lick-event stream contains a brief lick at the other port.  Ignore means
        # no reported lick; miss means the report was opposite the instruction.
        choice = np.full(len(kept_indices), 2, dtype=np.int8)  # no lick
        responded = outcome_text != "ignore"
        reported_left = ((outcome_text == "hit") & (instruction == "left")) | (
            (outcome_text == "miss") & (instruction == "right")
        )
        choice[responded & reported_left] = 0
        choice[responded & ~reported_left] = 1

        session_inputs = []
        session_outputs = []
        session_neural = []
        for trial_index in range(len(kept_indices)):
            session_neural.append(firing_rates[trial_index])
            session_inputs.append(
                np.stack(
                    [time_from_tone[trial_index], photostim[trial_index]], axis=0
                ).astype(np.float32)
            )
            session_outputs.append(
                np.vstack(
                    [
                        np.full(N_TIME, choice[trial_index], dtype=np.int8),
                        np.full(N_TIME, outcome[trial_index], dtype=np.int8),
                        np.full(N_TIME, early[trial_index], dtype=np.int8),
                        tongue[trial_index],
                    ]
                )
            )

        region_names = [REGION_LOOKUP[x.strip()] for x in annotations[good_indices]]
        region_indices = np.array(
            [REGION_TO_INDEX[x] for x in region_names], dtype=np.int16
        )
        identifier = _scalar_string(nwb["identifier"])
        subject = _scalar_string(nwb["general/subject/subject_id"])
        return {
            "neural": session_neural,
            "input": session_inputs,
            "output": session_outputs,
            "subject": subject,
            "brain_region_idx": region_indices,
            "session_info": {
                "identifier": identifier,
                "source_file": os.path.relpath(path, Path(__file__).parent),
                "source_trial_count": int(len(keep)),
                "ephys_trial_count": int(n_ephys_trials),
                "retained_trial_count": int(np.sum(keep)),
                "excluded_no_ephys_or_unit_invalid_trials": int(np.sum(~neural_valid)),
                "excluded_empty_neural_trials": int(np.sum(empty_neural_trials)),
                "excluded_auto_or_free_water_trials": int(np.sum(neural_valid & assistance)),
                "classifier_good_units": int(len(good_indices)),
                "tongue_y_percentile_40": p40,
                "tongue_y_percentile_60": p60,
                "tongue_visible_frame_fraction": visible_fraction,
            },
        }


def convert(data_dir: str, output_path: str, limit_sessions: int | None = None) -> dict:
    paths = sorted(glob.glob(os.path.join(data_dir, "sub-*", "*.nwb")))
    if not paths:
        raise FileNotFoundError(f"no NWB files under {data_dir}")

    converted_sessions = []
    skipped = []
    for path in paths:
        print(f"Converting {path}", flush=True)
        session = convert_session(path)
        if session is None:
            skipped.append(os.path.relpath(path, Path(__file__).parent))
            print("  skipped (no classifier-QC units or fewer than two trials)")
            continue
        converted_sessions.append(session)
        info = session["session_info"]
        print(
            f"  {info['retained_trial_count']} trials, "
            f"{info['classifier_good_units']} units",
            flush=True,
        )
        if limit_sessions is not None and len(converted_sessions) >= limit_sessions:
            break

    subjects = sorted({session["subject"] for session in converted_sessions})
    subject_to_index = {subject: i for i, subject in enumerate(subjects)}
    data = {
        "neural": [session["neural"] for session in converted_sessions],
        "input": [session["input"] for session in converted_sessions],
        "output": [session["output"] for session in converted_sessions],
        "subjects": subjects,
        "subject_idx": np.array(
            [subject_to_index[session["subject"]] for session in converted_sessions],
            dtype=np.int16,
        ),
        "brain_regions": BRAIN_REGIONS,
        "brain_region_idx": [
            session["brain_region_idx"] for session in converted_sessions
        ],
        "input_names": ["time from tone onset (s)", "photostimulation on"],
        "output_names": [
            "lick direction choice", "outcome", "early lick", "tongue y-position"
        ],
        "output_values": [
            ["left", "right", "no lick"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"],
        ],
        "metadata": {
            "task_description": (
                "Auditory delayed-response task: infer lick choice, outcome, "
                "early licking, and discretized tongue position from brain-wide activity."
            ),
            "time_bin_size": 50.0,
            "neural_measurement": "firing rate (spikes/s) in non-overlapping 50-ms bins",
            "temporal_alignment_event": "auditory Go cue onset",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "time_bin_centers_seconds": BIN_CENTERS.tolist(),
            "unit_filter": (
                "NWB classifier-QC classification == 'good' with a non-empty CCF annotation"
            ),
            "trial_filter": (
                "require a recorded/valid is_good_trials entry for every retained unit; "
                "exclude trials with no population spikes and auto-water/free-water assistance trials; retain early-lick, "
                "ignore/no-response, and photostimulation trials required by decoder variables"
            ),
            "tongue_visibility_rule": (
                f"side-camera DeepLabCut likelihood >= {TONGUE_LIKELIHOOD_CUTOFF}; "
                "percentiles computed from visible frames over each full session"
            ),
            "tone_onset_definition": "initial sample/tone onset (presample stop event)",
            "session_info": [session["session_info"] for session in converted_sessions],
            "skipped_source_files": skipped,
        },
    }

    with open(output_path, "wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument(
        "--output", default=str(Path(__file__).parent / "converted_data.pkl")
    )
    parser.add_argument(
        "--limit-sessions", type=int, default=None,
        help="development/testing aid; omit to convert the full curated dataset",
    )
    args = parser.parse_args()
    data = convert(args.data_dir, args.output, args.limit_sessions)
    print(
        f"Saved {len(data['neural'])} sessions and "
        f"{sum(map(len, data['neural']))} trials to {args.output}"
    )


if __name__ == "__main__":
    main()
