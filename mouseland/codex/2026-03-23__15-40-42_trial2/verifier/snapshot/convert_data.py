#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/app/code")
import utils  # noqa: E402


ROOT = Path("/app")
DATA_ROOT = ROOT / "data"
BEH_ROOT = DATA_ROOT / "beh"
RET_ROOT = DATA_ROOT / "retinotopy"
SPK_ROOT = DATA_ROOT / "spk"

SECONDS_PER_DAY = 24 * 3600
TEXTURE_LENGTH_DM = 40.0
CORRIDOR_LENGTH_DM = 60.0
DP_THRESHOLD = 0.3
MIN_NEURONS_FALLBACK = 64


@dataclass(frozen=True)
class SessionCandidate:
    session_id: str
    exp_type: str
    beh_key: str
    db: dict


@dataclass
class SessionProcessed:
    session_id: str
    subject: str
    exp_type: str
    day_of_training: float
    neural: list[np.ndarray]
    input_raw: list[dict[str, np.ndarray]]
    output_raw: list[dict[str, np.ndarray]]
    brain_region_idx: np.ndarray
    selected_neurons: int
    raw_neurons: int
    ntrials: int
    processing_info: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Zhong et al. 2025 imaging sessions into decoder format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 sessions.",
    )
    return parser.parse_args()


def trailing_number(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def choose_primary_wall(walls: list[str]) -> str:
    if not walls:
        raise ValueError("No wall names available to choose a primary wall.")
    preferred = [w for w in walls if re.search(r"1$", w)]
    if preferred:
        walls = preferred
    return sorted(walls, key=trailing_number)[0]


def safe_dprime(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if x1.size == 0 or x2.size == 0:
        return np.full((x1.shape[0] if x1.ndim == 2 else x2.shape[0],), np.nan, dtype=np.float32)
    u1 = np.nanmean(x1, axis=1)
    u2 = np.nanmean(x2, axis=1)
    s1 = np.nanstd(x1, axis=1)
    s2 = np.nanstd(x2, axis=1)
    denom = s1 + s2
    denom[denom == 0] = np.nan
    return (2.0 * (u1 - u2) / denom).astype(np.float32)


def session_sort_key(session_id: str) -> tuple[str, datetime, int]:
    parts = session_id.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    blk = int(parts[4])
    return subject, date, blk


def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key

    return sorted(candidates, key=key_fn)[0]


def build_session_catalog() -> list[SessionCandidate]:
    exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
    grouped: dict[str, list[SessionCandidate]] = defaultdict(list)
    for exp_type, db_list in exp_info.items():
        for db in db_list:
            session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            beh_key = session_id
            if "stimtype" in db:
                beh_key = f"{beh_key}_{db['stimtype']}"
            grouped[session_id].append(
                SessionCandidate(
                    session_id=session_id,
                    exp_type=exp_type,
                    beh_key=beh_key,
                    db=db,
                )
            )
    return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]


def load_behavior(exp_type: str, beh_key: str) -> dict:
    beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    return beh[beh_key]


def area_labels_from_iarea(iarea: np.ndarray) -> tuple[list[str], np.ndarray]:
    brain_regions = ["V1", "mHV", "lHV", "aHV", "other"]
    out = np.full(iarea.shape, 4, dtype=np.int64)
    out[iarea == 8] = 0
    out[np.isin(iarea, [0, 1, 2, 9])] = 1
    out[np.isin(iarea, [5, 6])] = 2
    out[np.isin(iarea, [3, 4])] = 3
    return brain_regions, out


def compute_day_offsets(catalog: list[SessionCandidate]) -> dict[str, float]:
    by_subject: dict[str, list[datetime]] = defaultdict(list)
    session_dates: dict[str, datetime] = {}
    for cand in catalog:
        date = datetime.strptime(cand.db["datexp"], "%Y_%m_%d")
        by_subject[cand.db["mname"]].append(date)
        session_dates[cand.session_id] = date
    first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
    return {
        cand.session_id: float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
        for cand in catalog
    }


def get_reference_pair(beh: dict) -> tuple[str, str]:
    uniq_walls = np.asarray(beh["UniqWalls"]).astype(str)
    stim_id = np.asarray(beh.get("stim_id", []), dtype=float)
    if uniq_walls.size and stim_id.size == uniq_walls.size:
        rew_like = uniq_walls[np.isfinite(stim_id) & (stim_id == 2)]
        nonrew_like = uniq_walls[np.isfinite(stim_id) & (stim_id == 0)]
        if rew_like.size and nonrew_like.size:
            return str(rew_like[0]), str(nonrew_like[0])

    wall_name = np.asarray(beh["WallName"]).astype(str)
    is_rew = np.asarray(beh["isRew"]).astype(bool)
    rew_walls = sorted(set(wall_name[is_rew]))
    nonrew_walls = sorted(set(wall_name[~is_rew]))
    if rew_walls and nonrew_walls:
        return choose_primary_wall(rew_walls), choose_primary_wall(nonrew_walls)

    primary_walls = sorted(set(w for w in wall_name if re.search(r"1$", w)), key=trailing_number)
    if len(primary_walls) >= 2:
        return primary_walls[0], primary_walls[1]

    all_walls = sorted(set(wall_name), key=trailing_number)
    if len(all_walls) >= 2:
        return all_walls[0], all_walls[1]

    raise ValueError("Could not determine a reference stimulus pair for this session.")


def compute_neuron_selection(
    spk: np.ndarray,
    beh: dict,
    region_idx_all: np.ndarray,
) -> tuple[np.ndarray, dict]:
    nfr = spk.shape[1]
    ft_wall = np.asarray(beh["ft_WallID"][:nfr]).astype(str)
    running = np.asarray(beh["ft_move"][:nfr], dtype=float) > 0
    corr = np.asarray(beh["ft_CorrSpc"][:nfr]).astype(bool)
    visual_mask = region_idx_all != 4
    rew_primary, nonrew_primary = get_reference_pair(beh)

    stim1_fr = (ft_wall == rew_primary) & corr & running
    stim2_fr = (ft_wall == nonrew_primary) & corr & running

    dp = safe_dprime(spk[:, stim1_fr], spk[:, stim2_fr])
    stim_selective = visual_mask & np.isfinite(dp) & (np.abs(dp) >= DP_THRESHOLD)

    wall_name = np.asarray(beh["WallName"]).astype(str)
    sound_fr = np.asarray(beh["SoundFr"], dtype=int)
    cue_pos = np.mod(np.asarray(beh["SoundDelPos"], dtype=float), float(beh["Corridor_Length"]))
    reward_trials = wall_name == rew_primary
    nonreward_trials = wall_name == nonrew_primary
    valid_reward_trials = reward_trials & np.isfinite(cue_pos)

    valid_sound_reward = reward_trials & (sound_fr >= 0) & (sound_fr < nfr)
    valid_sound_nonreward = nonreward_trials & (sound_fr >= 0) & (sound_fr < nfr)
    dp_sound = safe_dprime(spk[:, sound_fr[valid_sound_reward]], spk[:, sound_fr[valid_sound_nonreward]])

    reward_dp = np.full(spk.shape[0], np.nan, dtype=np.float32)
    if valid_reward_trials.sum() >= 4:
        threshold = np.nanmean(cue_pos[np.isfinite(cue_pos)])
        early_trials = valid_reward_trials & np.asarray(beh["isRew"]).astype(bool) & (cue_pos <= threshold)
        late_trials = valid_reward_trials & np.asarray(beh["isRew"]).astype(bool) & (cue_pos > threshold)
        if early_trials.sum() >= 2 and late_trials.sum() >= 2:
            start_fr = np.asarray(beh["StartFr"], dtype=int)
            gray_fr_trial = np.asarray(beh["GrayFr"], dtype=int)
            ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
            trial_means = np.full((spk.shape[0], int(beh["ntrials"])), np.nan, dtype=np.float32)
            for trial_idx in np.flatnonzero(valid_reward_trials):
                frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
                frames = frames[running[frames] & (ft_pos[frames] >= 5.0) & (ft_pos[frames] <= TEXTURE_LENGTH_DM)]
                if frames.size:
                    trial_means[:, trial_idx] = np.nanmean(spk[:, frames], axis=1)
            reward_dp = safe_dprime(trial_means[:, late_trials], trial_means[:, early_trials])
    reward_dp_thr = np.inf
    ahv_mask = region_idx_all == 3
    if np.isfinite(reward_dp[ahv_mask]).any():
        reward_dp_thr = float(np.nanpercentile(reward_dp[ahv_mask], 95))
    reward_pred = (
        ahv_mask
        & np.isfinite(reward_dp)
        & np.isfinite(dp_sound)
        & (dp_sound > DP_THRESHOLD)
        & (reward_dp >= reward_dp_thr)
    )

    selected = stim_selective | reward_pred

    if selected.sum() < MIN_NEURONS_FALLBACK:
        candidate = visual_mask & np.isfinite(dp)
        if candidate.sum() == 0:
            candidate = visual_mask
        abs_dp = np.abs(np.nan_to_num(dp, nan=0.0))
        candidate_idx = np.flatnonzero(candidate)
        order = candidate_idx[np.argsort(abs_dp[candidate_idx])[::-1]]
        nkeep = min(max(MIN_NEURONS_FALLBACK, int(0.01 * len(order))), len(order))
        selected = np.zeros(spk.shape[0], dtype=bool)
        selected[order[:nkeep]] = True

    info = {
        "rew_primary": rew_primary,
        "nonrew_primary": nonrew_primary,
        "stimulus_selective_count": int(stim_selective.sum()),
        "reward_prediction_count": int(reward_pred.sum()),
        "reward_prediction_threshold": reward_dp_thr,
        "selected_count": int(selected.sum()),
    }
    return selected, info


def build_trial_arrays(
    spk_sel: np.ndarray,
    beh: dict,
    session_day: float,
    stimulus_to_idx: dict[str, int],
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], np.ndarray, dict]:
    nfr = spk_sel.shape[1]
    ft = np.asarray(beh["ft"][:nfr], dtype=float)
    dft = np.diff(ft)
    dft = dft[np.isfinite(dft) & (dft > 0)]
    frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
    ft_move = np.asarray(beh["ft_move"][:nfr], dtype=float)
    ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)

    start_fr = np.asarray(beh["StartFr"], dtype=int)
    gray_fr = np.asarray(beh["GrayFr"], dtype=int)
    sound_fr = np.asarray(beh["SoundFr"], dtype=int)
    wall_name = np.asarray(beh["WallName"]).astype(str)
    is_rew = np.asarray(beh["isRew"]).astype(np.float32)
    lick_fr = np.asarray(beh["LickFr"], dtype=float)
    lick_tr = np.asarray(beh["LickTrind"], dtype=float)

    neural_trials: list[np.ndarray] = []
    input_trials: list[dict[str, np.ndarray]] = []
    output_trials: list[dict[str, np.ndarray]] = []
    all_speeds: list[np.ndarray] = []
    trial_lengths: list[int] = []

    for trial_idx in range(int(beh["ntrials"])):
        frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
        frames = frames[ft_move[frames] > 0]
        if frames.size == 0:
            continue

        neural = spk_sel[:, frames].astype(np.float32, copy=False)
        retained_idx = np.arange(frames.size, dtype=np.float32)
        cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
        t_since = (retained_idx * frame_dt).astype(np.float32)
        t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
        reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
        day = np.full(frames.shape, session_day, dtype=np.float32)

        lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
        lick_frames_trial = lick_frames_trial.astype(int)
        licking = np.isin(frames, lick_frames_trial).astype(np.int64)

        pos = ft_pos[frames]
        pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
        raw_speed = ft_speed[frames].astype(np.float32)
        stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)

        neural_trials.append(neural)
        input_trials.append(
            {
                "time_to_sound_cue_sec": t_to_cue,
                "day_of_training": day,
                "time_since_trial_start_sec": t_since,
                "reward_available": reward,
            }
        )
        output_trials.append(
            {
                "visual_stimulus_category": stim_val,
                "licking": licking,
                "position_bin": pos_bin,
                "running_speed_raw": raw_speed,
            }
        )
        all_speeds.append(raw_speed)
        trial_lengths.append(int(frames.size))

    summary = {
        "trial_lengths": np.asarray(trial_lengths, dtype=np.int32),
        "mean_trial_length": float(np.mean(trial_lengths)) if trial_lengths else 0.0,
        "median_trial_length": float(np.median(trial_lengths)) if trial_lengths else 0.0,
        "speed_count": int(sum(len(x) for x in all_speeds)),
        "ntrials_kept": len(neural_trials),
    }
    speed_concat = np.concatenate(all_speeds).astype(np.float32) if all_speeds else np.empty((0,), dtype=np.float32)
    return neural_trials, input_trials, output_trials, speed_concat, summary


def finalize_io(
    processed_sessions: list[SessionProcessed],
    speed_edges: np.ndarray,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[np.ndarray]]]:
    neural_all: list[list[np.ndarray]] = []
    input_all: list[list[np.ndarray]] = []
    output_all: list[list[np.ndarray]] = []
    thresholds = speed_edges[1:-1]

    for session in processed_sessions:
        neural_session: list[np.ndarray] = []
        input_session: list[np.ndarray] = []
        output_session: list[np.ndarray] = []
        for neural_trial, input_raw, output_raw in zip(session.neural, session.input_raw, session.output_raw):
            T = neural_trial.shape[1]
            input_arr = np.vstack(
                [
                    input_raw["time_to_sound_cue_sec"],
                    input_raw["day_of_training"],
                    input_raw["time_since_trial_start_sec"],
                    input_raw["reward_available"],
                ]
            ).astype(np.float32)
            speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
            output_arr = np.vstack(
                [
                    output_raw["visual_stimulus_category"],
                    output_raw["licking"],
                    output_raw["position_bin"],
                    speed_bin,
                ]
            ).astype(np.int64)
            if input_arr.shape[1] != T or output_arr.shape[1] != T:
                raise ValueError(f"Time dimension mismatch in session {session.session_id}.")
            neural_session.append(neural_trial.astype(np.float32, copy=False))
            input_session.append(input_arr)
            output_session.append(output_arr)
        neural_all.append(neural_session)
        input_all.append(input_session)
        output_all.append(output_session)

    return neural_all, input_all, output_all


def make_speed_bin_names(speed_edges: np.ndarray) -> list[str]:
    names = []
    for i in range(4):
        lo = speed_edges[i]
        hi = speed_edges[i + 1]
        names.append(f"{lo:.2f}-{hi:.2f}")
    return names


def plot_processing(
    session: SessionProcessed,
    speed_edges: np.ndarray | None,
) -> None:
    info = session.processing_info
    trial_lengths = info["trial_lengths"]
    cue_positions = info["cue_positions_dm"]
    example = info["example_trial"]

    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    ax = ax.ravel()

    ax[0].bar(
        ["raw", "selected"],
        [session.raw_neurons, session.selected_neurons],
        color=["0.7", "tab:blue"],
    )
    ax[0].set_title("Neuron Curation")
    ax[0].set_ylabel("count")

    ax[1].hist(trial_lengths, bins=30, color="tab:green", alpha=0.8)
    ax[1].set_title("Retained Trial Lengths")
    ax[1].set_xlabel("running texture frames")
    ax[1].set_ylabel("trials")

    ax[2].hist(cue_positions, bins=20, color="tab:orange", alpha=0.8)
    ax[2].set_title("Cue Positions")
    ax[2].set_xlabel("position (dm)")
    ax[2].set_ylabel("trials")

    if example is not None:
        im = ax[3].imshow(example["neural"], aspect="auto", interpolation="nearest", cmap="magma")
        ax[3].set_title("Example Trial Neural")
        ax[3].set_xlabel("retained frame")
        ax[3].set_ylabel("sampled neurons")
        fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

        ax[4].plot(example["time_since"], label="time_since_start")
        ax[4].plot(example["time_to_cue"], label="time_to_cue")
        ax[4].plot(example["position_bin"], label="position_bin")
        ax[4].plot(example["licking"], label="licking")
        ax[4].legend(loc="upper right", fontsize=8)
        ax[4].set_title("Example Trial Alignment")
        ax[4].set_xlabel("retained frame")

        ax[5].plot(example["speed_raw"], label="speed")
        if speed_edges is not None:
            for edge in speed_edges[1:-1]:
                ax[5].axhline(edge, color="k", linestyle="--", linewidth=0.8)
        ax[5].set_title("Running Speed and Quartiles")
        ax[5].set_xlabel("retained frame")
        ax[5].set_ylabel("speed")
    else:
        ax[3].axis("off")
        ax[4].axis("off")
        ax[5].axis("off")

    fig.suptitle(session.session_id)
    fig.tight_layout()
    fig.savefig(ROOT / f"processing_{session.session_id}.png", dpi=150)
    plt.close(fig)


def process_session(
    candidate: SessionCandidate,
    day_offsets: dict[str, float],
    stimulus_to_idx: dict[str, int],
    show_processing: bool,
) -> SessionProcessed:
    t0 = time.perf_counter()
    beh = load_behavior(candidate.exp_type, candidate.beh_key)
    spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
    ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
    _, region_idx_all = area_labels_from_iarea(ret["iarea"])

    selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
    spk_sel = spk[selected_mask]
    region_sel = region_idx_all[selected_mask]
    raw_neurons = int(spk.shape[0])
    del spk

    neural_trials, input_trials, output_trials, speed_values, trial_summary = build_trial_arrays(
        spk_sel,
        beh,
        session_day=day_offsets[candidate.session_id],
        stimulus_to_idx=stimulus_to_idx,
    )

    example_trial = None
    if show_processing and neural_trials:
        idx = 0
        sample_neurons = min(100, neural_trials[idx].shape[0])
        example_trial = {
            "neural": neural_trials[idx][:sample_neurons],
            "time_since": input_trials[idx]["time_since_trial_start_sec"],
            "time_to_cue": input_trials[idx]["time_to_sound_cue_sec"],
            "position_bin": output_trials[idx]["position_bin"],
            "licking": output_trials[idx]["licking"],
            "speed_raw": output_trials[idx]["running_speed_raw"],
        }

    cue_positions = np.mod(np.asarray(beh["SoundDelPos"], dtype=float), float(beh["Corridor_Length"]))
    cue_positions = cue_positions[np.isfinite(cue_positions)]

    proc_info = {
        **selection_info,
        **trial_summary,
        "cue_positions_dm": cue_positions,
        "example_trial": example_trial,
    }
    elapsed = time.perf_counter() - t0
    print(
        f"[session] {candidate.session_id} exp={candidate.exp_type} "
        f"neurons={raw_neurons}->{selected_mask.sum()} trials={len(neural_trials)} "
        f"mean_T={trial_summary['mean_trial_length']:.1f} time={elapsed:.1f}s"
    )

    return SessionProcessed(
        session_id=candidate.session_id,
        subject=candidate.db["mname"],
        exp_type=candidate.exp_type,
        day_of_training=day_offsets[candidate.session_id],
        neural=neural_trials,
        input_raw=input_trials,
        output_raw=output_trials,
        brain_region_idx=region_sel.astype(np.int64),
        selected_neurons=int(selected_mask.sum()),
        raw_neurons=raw_neurons,
        ntrials=len(neural_trials),
        processing_info=proc_info,
    )


def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    names = set()
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names)


def session_spk_size(candidate: SessionCandidate) -> int:
    fn = f"{candidate.db['mname']}_{candidate.db['datexp']}_{candidate.db['blk']}_neural_data.npy"
    return (SPK_ROOT / fn).stat().st_size


def sample_candidate_score(candidate: SessionCandidate) -> tuple[int, int]:
    beh = load_behavior(candidate.exp_type, candidate.beh_key)
    reward = np.asarray(beh["isRew"]).astype(bool)
    start = np.asarray(beh["StartFr"], dtype=int)
    gray = np.asarray(beh["GrayFr"], dtype=int)
    move = np.asarray(beh["ft_move"]) > 0
    lick_fr = np.asarray(beh["LickFr"], dtype=float)
    lick_tr = np.asarray(beh["LickTrind"], dtype=float)

    lick_trials = 0
    for trial_idx, (s, g) in enumerate(zip(start, gray)):
        frames = np.arange(s, g, dtype=int)
        frames = frames[move[frames]]
        if frames.size == 0:
            continue
        lfr = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)].astype(int)
        if np.isin(frames, lfr).any():
            lick_trials += 1

    representative = int((reward.sum() > 0) and ((~reward).sum() > 0) and (lick_trials > 10))
    return representative, -session_spk_size(candidate)


def compute_time_bin_ms(catalog: list[SessionCandidate]) -> float:
    medians = []
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
        d = np.diff(ft)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            medians.append(float(np.median(d) * SECONDS_PER_DAY * 1000.0))
    if not medians:
        raise RuntimeError("Could not compute imaging frame interval.")
    return float(np.median(medians))


def main() -> None:
    args = parse_args()
    do_sample = args.sample

    t_start = time.perf_counter()
    full_catalog = build_session_catalog()
    day_offsets = compute_day_offsets(full_catalog)
    stimulus_values = collect_stimulus_values(full_catalog)
    time_bin_ms = compute_time_bin_ms(full_catalog)
    catalog = full_catalog
    if do_sample:
        ranked = sorted(
            catalog,
            key=lambda c: (
                -sample_candidate_score(c)[0],
                session_spk_size(c),
                c.session_id,
            ),
        )
        catalog = ranked[:2]
    print(f"Processing {len(catalog)} sessions.")

    stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}

    processed_sessions: list[SessionProcessed] = []
    all_speed_values: list[np.ndarray] = []
    subjects_all = sorted({cand.db["mname"] for cand in catalog})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
    brain_regions = ["V1", "mHV", "lHV", "aHV", "other"]

    for idx, cand in enumerate(catalog, start=1):
        print(f"[{idx}/{len(catalog)}] loading {cand.session_id}")
        session = process_session(
            cand,
            day_offsets=day_offsets,
            stimulus_to_idx=stimulus_to_idx,
            show_processing=args.show_processing and len(processed_sessions) < 2,
        )
        processed_sessions.append(session)
        for trial in session.output_raw:
            all_speed_values.append(trial["running_speed_raw"])

    if not all_speed_values:
        raise RuntimeError("No running-speed samples were collected.")

    all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
    speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
    for i in range(1, len(speed_edges)):
        if speed_edges[i] <= speed_edges[i - 1]:
            speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
    print(f"Running speed quartile edges: {speed_edges.tolist()}")

    neural_all, input_all, output_all = finalize_io(processed_sessions, speed_edges=speed_edges)

    if args.show_processing:
        for session in processed_sessions[:2]:
            plot_processing(session, speed_edges)

    data = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects_all,
        "subject_idx": np.asarray(
            [subject_to_idx[s.subject] for s in processed_sessions], dtype=np.int64
        ),
        "brain_regions": brain_regions,
        "brain_region_idx": [s.brain_region_idx.astype(np.int64) for s in processed_sessions],
        "input_names": [
            "time_to_sound_cue_sec",
            "day_of_training",
            "time_since_trial_start_sec",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus_category",
            "licking",
            "position_bin",
            "running_speed_bin",
        ],
        "output_values": [
            stimulus_values,
            ["no_lick", "lick"],
            ["0-1m", "1-2m", "2-3m", "3-4m"],
            make_speed_bin_names(speed_edges),
        ],
        "metadata": {
            "task_description": (
                "Running-only texture-segment imaging data aligned to corridor entry, "
                "for decoding stimulus category, licking, 1 m position bin, and speed quartile."
            ),
            "time_bin_size": time_bin_ms,
            "temporal_alignment_event": "corridor entry (trial start / StartFr)",
            "off_start": 0.0,
            "off_end": None,
            "trial_end_event": "entry into gray space (GrayFr)",
            "frame_selection": "running-only frames within texture area",
            "neuron_selection": (
                "visual-cortex neurons with paper-style running-corridor stimulus selectivity "
                "|dprime|>=0.3, plus aHV reward-prediction neurons requiring cue-frame stimulus "
                "dprime>0.3 and late-vs-early cue dprime above the session 95th percentile; "
                "fallback top-|dprime| selection is used only if too few neurons survive."
            ),
            "speed_bin_edges": speed_edges.tolist(),
            "session_ids": [s.session_id for s in processed_sessions],
            "canonical_exp_type": [s.exp_type for s in processed_sessions],
        },
    }

    out_path = Path(args.outpicklefile)
    with out_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_trials = sum(len(s.neural) for s in processed_sessions)
    total_neurons = sum(int(s.brain_region_idx.shape[0]) for s in processed_sessions)
    elapsed = time.perf_counter() - t_start
    print(
        f"Saved {out_path} with {len(processed_sessions)} sessions, "
        f"{total_trials} trials, {total_neurons} curated neurons in {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
