"""Keyframe labels from lerobot/libero proprioception.

Gripper opening is state[6] - state[7]. End-effector position is state[0:3].
The release has no reward, done, success, collision, timeout, object pose,
receptacle pose, battery, or occlusion column. Those are not invented.

Each verb is a measured gripper or end-effector change. p_success is what the
later frames do, on a 2 second horizon for an in-progress skill and on the
completed close for a grasp. p_unsafe is a later-frame jump over 5 cm. That
jump does not occur in this release, so the unsafe target stays false.

Escalate follows the published rule. ask_human is not emitted: there is no
battery or familiarity channel. abort is not emitted as a gold action: there
is no collision channel. ask_big is the near-tie branch, scored from the next
half-second of executed motion when two verb scores are close. A mid-episode
stall is the fallback near-tie when every score is flat. Else the label is act.

Home is the recorded start pose. If it would be more than 28% of a split, extra
home rows are dropped. They are not relabeled. Refuse rows are not copied. If
dropping excess home still cannot put refuse at 15%, the gate fails.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lerobot_libero"
EVENTS = ROOT / "data" / "jsonl" / "events.jsonl"
REPORT = ROOT / "data" / "jsonl" / "label_report.json"
FPS = 10.0
HORIZON = 20
DISCONTINUITY_M = 0.05
STALL_STEP_M = 0.002
STALL_FRAMES = 10
ABSENT = [
    "reward",
    "done",
    "success",
    "collision",
    "timeout",
    "object_pose",
    "receptacle_pose",
    "battery",
    "occlusion",
]


def episode_split(episode_index):
    digest = hashlib.sha256(f"actone-v1:{episode_index}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 15:
        return "test"
    if bucket < 30:
        return "calibration"
    if bucket < 42:
        return "dev"
    return "train"


def load_tasks(root):
    table = pq.read_table(root / "meta" / "tasks.parquet")
    names = table.column("__index_level_0__").to_pylist()
    indices = table.column("task_index").to_pylist()
    return {int(index): str(name) for index, name in zip(indices, names)}


def load_episode_videos(root):
    videos = {}
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        episode = table.column("episode_index").to_pylist()
        chunk = table.column("videos/observation.images.image/chunk_index").to_pylist()
        file_index = table.column("videos/observation.images.image/file_index").to_pylist()
        start = table.column("videos/observation.images.image/from_timestamp").to_pylist()
        for i in range(table.num_rows):
            videos[int(episode[i])] = {
                "chunk": int(chunk[i]),
                "file": int(file_index[i]),
                "from_timestamp": float(start[i]),
            }
    return videos


def load_states(root):
    grouped = defaultdict(list)
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(
            path,
            columns=["observation.state", "episode_index", "frame_index", "timestamp", "task_index"],
        )
        states = table.column("observation.state").to_pylist()
        episodes = table.column("episode_index").to_pylist()
        frames = table.column("frame_index").to_pylist()
        times = table.column("timestamp").to_pylist()
        tasks = table.column("task_index").to_pylist()
        for i, episode in enumerate(episodes):
            grouped[int(episode)].append(
                (int(frames[i]), float(times[i]), int(tasks[i]), states[i])
            )
    arrays = {}
    for episode, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        state = np.asarray([row[3] for row in rows], dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 8:
            raise ValueError(f"episode {episode} state shape {state.shape}")
        arrays[episode] = {
            "frame": np.asarray([row[0] for row in rows], dtype=np.int64),
            "time": np.asarray([row[1] for row in rows], dtype=np.float64),
            "task": int(rows[0][2]),
            "state": state,
        }
    return arrays


def _future_discontinuity(xyz, frame):
    future = xyz[frame + 1 : frame + 31]
    if len(future) < 2:
        return False
    steps = np.linalg.norm(np.diff(future, axis=0), axis=1)
    return bool(steps.max() > DISCONTINUITY_M)


def _stuck(step, index):
    quiet = 0
    cursor = index
    while cursor > 0 and step[cursor] < STALL_STEP_M:
        quiet += 1
        cursor -= 1
    return round(quiet / FPS, 2)


def _scores(width, xyz, height, span, closed, opened, index):
    length = len(width)
    end = min(length - 1, index + 5)
    if end <= index or span <= 0:
        return {"reach": 0.0, "grasp": 0.0, "lift": 0.0, "retract": 0.0}
    dw = (width[end] - width[index]) / span
    travel = float(np.linalg.norm(xyz[end] - xyz[index]))
    lift = 0.0
    if width[index] <= closed + 0.20 * span:
        lift = min(1.0, max(0.0, (height[end] - height[index]) / 0.02))
    reach = min(1.0, travel / 0.02) if width[index] >= opened else 0.0
    return {
        "reach": reach,
        "grasp": min(1.0, max(0.0, -dw)),
        "lift": lift,
        "retract": min(1.0, max(0.0, dw)),
    }


def _ranked(scores):
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ordered[0], ordered[1]


def _completes(verb, index, width, xyz, height, closed, opened, span):
    length = len(width)
    end = min(length - 1, index + HORIZON)
    if end <= index:
        return False
    if verb == "grasp":
        for cursor in range(index, end + 1):
            if width[cursor] <= closed:
                stayed = width[cursor : min(length, cursor + 5)]
                return bool(len(stayed) and np.median(stayed) <= closed + 0.10 * span)
        return False
    if verb == "reach":
        return bool(np.any(width[index : end + 1] <= closed))
    if verb == "lift":
        return bool(float(height[index : end + 1].max() - height[index]) >= 0.015)
    if verb == "retract":
        distance = np.linalg.norm(xyz[index + 1 : end + 1] - xyz[index], axis=1)
        opened_later = bool(np.any(width[index : end + 1] >= opened))
        return bool(len(distance) and distance.max() >= 0.03 and opened_later)
    if verb == "home":
        return True
    return False


def _find_grasps(width, span, closed, opened):
    delta = np.diff(width, prepend=width[0])
    length = len(width)
    found = []
    index = 0
    min_drop = 0.35 * span
    while index < length - 2:
        if width[index] < opened:
            index += 1
            continue
        close_at = None
        trough = index
        rebound = None
        limit = min(length, index + 50)
        for cursor in range(index + 1, limit):
            if width[cursor] <= width[trough]:
                trough = cursor
            if width[cursor] <= closed:
                close_at = cursor
                break
            drop = float(width[index] - width[trough])
            if (
                drop >= min_drop
                and cursor > trough + 1
                and width[cursor] > width[trough] + 0.40 * drop
            ):
                rebound = trough
                break
        if close_at is None and rebound is None:
            index += 1
            continue
        end = close_at if close_at is not None else rebound
        region = delta[index + 1 : end + 1]
        if len(region) == 0 or float(region.min()) >= 0:
            index = max(end + 1, index + 1)
            continue
        key = index + 1 + int(np.argmin(region))
        if close_at is not None:
            stayed = width[close_at : min(length, close_at + 5)]
            success = bool(len(stayed) and np.median(stayed) <= closed + 0.10 * span)
            drop = float(width[index] - width[close_at])
            found.append(
                {
                    "key": key,
                    "close_at": close_at,
                    "success": success,
                    "drop": drop,
                    "reached_closed": True,
                }
            )
            index = close_at + 1
            continue
        drop = float(width[index] - width[rebound])
        if drop < min_drop:
            index = rebound + 1
            continue
        found.append(
            {
                "key": key,
                "close_at": None,
                "success": False,
                "drop": drop,
                "reached_closed": False,
                "rebound": rebound,
            }
        )
        index = rebound + 1
    return found


def _find_reach(width, step, grasp_key, opened):
    """First frame of the last motion burst while the gripper is still open.

    Walking back only to the first fast frame stopped at the close itself, so
    most approaches were dropped. The burst after the last short rest is the
    approach. p_success is still whether the closed band arrives within 2 s.
    """
    cursor = grasp_key
    while cursor > 0 and width[cursor] >= opened:
        cursor -= 1
    start = max(cursor + 1, 1)
    if start >= grasp_key:
        return None
    rest = 0
    motion_at = None
    for frame in range(start, grasp_key):
        if step[frame] <= STALL_STEP_M:
            rest += 1
            continue
        if motion_at is None or rest >= 3:
            motion_at = frame
        rest = 0
    return motion_at


def _find_lift(width, height, close_at, closed, span, height_span):
    band = closed + 0.15 * span
    window = []
    cursor = close_at
    while cursor < len(width) and width[cursor] <= band and len(window) < 40:
        window.append(cursor)
        cursor += 1
    if len(window) < 5:
        return None
    base = float(height[close_at])
    segment = height[window]
    rise = float(segment.max() - base)
    fall = float(base - segment.min())
    peak = window[int(np.argmax(segment))]
    needed = max(0.015, 0.20 * height_span)
    if rise >= needed and peak > close_at + 1:
        later = height[close_at + 2 : peak + 1]
        success = bool(len(later) and float(later.max()) >= base + 0.5 * rise)
        return close_at + 1, success, rise
    if 0.005 <= rise < 0.015 and fall <= rise:
        return close_at + 1, False, rise
    return None


def _find_retract(width, xyz, close_at, opened):
    reopen = close_at
    length = len(width)
    while reopen < length and width[reopen] < opened:
        reopen += 1
    if reopen >= length - 1:
        return None
    horizon = min(length, reopen + HORIZON)
    distance = np.linalg.norm(xyz[reopen + 1 : horizon] - xyz[reopen], axis=1)
    travel = float(distance.max()) if len(distance) else 0.0
    return reopen, travel >= 0.03, travel


def _find_stall(step):
    length = len(step)
    best = None
    cursor = 1
    while cursor < length:
        if step[cursor] >= STALL_STEP_M:
            cursor += 1
            continue
        start = cursor
        while cursor < length and step[cursor] < STALL_STEP_M:
            cursor += 1
        run = cursor - start
        if run < STALL_FRAMES:
            continue
        mid = (start + cursor - 1) / 2 / length
        if mid < 0.15 or mid > 0.85:
            continue
        if best is None or run > best[0]:
            best = (run, start)
    return None if best is None else best[1]


def _near_tie(width, xyz, height, span, closed, opened, used):
    length = len(width)
    best = None
    lo = int(0.15 * length)
    hi = int(0.85 * length)
    for index in range(max(lo, 1), max(hi, 1)):
        if index in used:
            continue
        scores = _scores(width, xyz, height, span, closed, opened, index)
        (lead_name, lead), (rival_name, rival) = _ranked(scores)
        if lead < 0.20 or rival < 0.15:
            continue
        margin = lead - rival
        if margin > 0.12:
            continue
        if best is None or margin < best[0]:
            best = (margin, index, lead_name, rival_name, scores)
    return best


def label_episode(episode, pack, stats):
    state = pack["state"]
    frame = pack["frame"]
    time = pack["time"]
    length = len(state)
    if length < 8:
        return []
    width = state[:, 6] - state[:, 7]
    xyz = state[:, :3]
    height = xyz[:, 2]
    step = np.linalg.norm(np.diff(xyz, axis=0, prepend=xyz[:1]), axis=1)
    stats["max_step_m"] = max(stats["max_step_m"], float(step.max()))
    span = float(width.max() - width.min())
    height_span = float(height.max() - height.min())
    events = {}

    def add(verb, index, success, escalate, evidence):
        index = int(index)
        if index < 0 or index >= length:
            return
        evidence = {
            **evidence,
            "gripper_width": round(float(width[index]), 5),
            "ee_z": round(float(height[index]), 5),
            "horizon_s": HORIZON / FPS,
            "channels_absent": ABSENT,
        }
        if index in events:
            current = events[index]
            if escalate != "act" and current["escalate"] == "act":
                current["escalate"] = escalate
                current["evidence"]["escalate_upgrade"] = evidence
            return
        events[index] = {
            "episode_index": int(episode),
            "frame_index": int(frame[index]),
            "timestamp": float(time[index]),
            "verb": verb,
            "p_success": bool(success),
            "p_unsafe": _future_discontinuity(xyz, index),
            "escalate": escalate,
            "ee_z": round(float(height[index]), 5),
            "gripper_width": round(float(width[index]), 5),
            "stuck_s": _stuck(step, index),
            "evidence": evidence,
        }

    add(
        "home",
        0,
        True,
        "act",
        {
            "rule": "episode starts at this end-effector pose",
            "outcome": "observed_start_state",
        },
    )
    if span < 0.01:
        return _finish(list(events.values()))

    closed = float(width.min() + 0.30 * span)
    opened = float(width.max() - 0.25 * span)
    for grasp in _find_grasps(width, span, closed, opened):
        add(
            "grasp",
            grasp["key"],
            grasp["success"],
            "act",
            {
                "rule": "gripper width falls from the open band",
                "outcome": "later frames reach the closed band and stay there"
                if grasp["success"]
                else "the close does not stay in the closed band",
                "width_drop": round(float(grasp["drop"]), 5),
            },
        )
        reach_at = _find_reach(width, step, grasp["key"], opened)
        if reach_at is not None and reach_at != grasp["key"]:
            lead = grasp["key"] - reach_at
            stats["reach_lead_frames"].append(int(lead))
            reached = bool(
                grasp["reached_closed"]
                and grasp["close_at"] is not None
                and grasp["close_at"] - reach_at <= HORIZON
            )
            add(
                "reach",
                reach_at,
                reached,
                "act",
                {
                    "rule": "gripper is open and the end effector starts translating toward a later close",
                    "outcome": "the gripper reaches the closed band within 2 seconds",
                    "object_pose": "absent; reach is the open-gripper translation, not an object detection",
                    "lead_frames": int(lead),
                },
            )
        if grasp["reached_closed"] and grasp["close_at"] is not None:
            lift = _find_lift(width, height, grasp["close_at"], closed, span, height_span)
            if lift is not None:
                lift_at, lift_ok, rise = lift
                add(
                    "lift",
                    lift_at,
                    lift_ok,
                    "act",
                    {
                        "rule": "end-effector z changes while the gripper stays closed",
                        "outcome": "later closed frames realize at least half of the rise"
                        if lift_ok
                        else "z rises less than 1.5 cm while the gripper stays closed",
                        "rise_m": round(float(rise), 5),
                        "object_height": "not in the dataset; ee_z while closed is the measured proxy",
                    },
                )
            retract = _find_retract(width, xyz, grasp["close_at"], opened)
            if retract is not None:
                reopen, moved, travel = retract
                add(
                    "retract",
                    reopen,
                    moved,
                    "act",
                    {
                        "rule": "gripper reopens, then the end effector leaves the release pose",
                        "outcome": "the next 2 seconds move at least 3 cm from the release pose",
                        "travel_m": round(float(travel), 5),
                    },
                )

    stall_at = _find_stall(step)
    if stall_at is not None and stall_at not in events:
        before = max(1, stall_at - 5)
        scores = _scores(width, xyz, height, span, closed, opened, before)
        (lead_name, lead), _rival = _ranked(scores)
        if lead >= 0.15:
            verb = lead_name
        elif width[stall_at] <= closed:
            verb = "lift"
        elif width[stall_at] >= opened:
            verb = "reach"
        else:
            verb = "grasp"
        success = _completes(verb, stall_at, width, xyz, height, closed, opened, span)
        add(
            verb,
            stall_at,
            success,
            "act",
            {
                "rule": "end effector moves less than 2 mm per frame for at least one second, mid-episode",
                "outcome": f"{verb} completion within 2 seconds is {bool(success)}",
                "stall_frames": STALL_FRAMES,
                "scores_before_stall": {name: round(float(value), 4) for name, value in scores.items()},
            },
        )
        stats["stalls"] += 1
    used = set(events)
    tie = _near_tie(width, xyz, height, span, closed, opened, used)
    if tie is not None:
        _margin, index, lead_name, rival_name, scores = tie
        success = _completes(lead_name, index, width, xyz, height, closed, opened, span)
        add(
            lead_name,
            index,
            success,
            "ask_big",
            {
                "rule": "near-tie between two proprioceptive verb scores over the next 0.5 s",
                "outcome": f"{lead_name} completion within 2 seconds is {bool(success)}",
                "leading_verb": lead_name,
                "rival_verb": rival_name,
                "scores": {name: round(float(value), 4) for name, value in scores.items()},
                "occlusion": "not in the dataset; the tie is between executed motion scores",
            },
        )
        stats["near_tie"] += 1
    elif stall_at is not None and stall_at in events:
        events[stall_at]["escalate"] = "ask_big"
        events[stall_at]["evidence"]["escalate_rule"] = (
            "no half-second near-tie in this episode; the stall is the ambiguity branch"
        )
        stats["stall_refuse"] += 1
    return _finish(list(events.values()))


def _finish(events):
    events.sort(key=lambda item: (item["frame_index"], item["verb"]))
    previous = "none"
    for index, event in enumerate(events):
        event["last_outcome"] = previous
        event["skill_age_s"] = 0.0 if index == 0 else round(event["timestamp"] - events[index - 1]["timestamp"], 2)
        previous = "success" if event["p_success"] else "failure"
    return events


def choose_images(events, budgets=None):
    by_split = defaultdict(lambda: defaultdict(list))
    for event in events:
        by_split[event["split"]][event["episode_index"]].append(event)
    if budgets is None:
        budgets = {"train": 240, "dev": 40, "calibration": 50, "test": 50}
    chosen_episodes = {}
    for split, budget in budgets.items():
        episodes = sorted(by_split[split])
        chosen_episodes[split] = episodes[:budget]
    selected = []
    for split, episodes in chosen_episodes.items():
        for episode in episodes:
            selected.extend(by_split[split][episode])
    return selected


def _share(rows, predicate):
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def balance_split(rows):
    """Drop excess home, then other majority act rows, until home <= 28% and refuse >= 15%."""
    rows = list(rows)
    dropped = defaultdict(int)

    def rebuild(homes, others):
        return others + homes

    others = [row for row in rows if row["verb"] != "home"]
    homes = sorted(
        [row for row in rows if row["verb"] == "home"],
        key=lambda row: (row["episode_index"], row["frame_index"]),
    )
    if others:
        max_home = max(1, int((0.28 * len(others)) / 0.72))
    else:
        max_home = len(homes)
    if len(homes) > max_home:
        step = len(homes) / max_home
        kept = []
        used = set()
        for i in range(max_home):
            index = min(len(homes) - 1, int(i * step))
            while index in used and index + 1 < len(homes):
                index += 1
            used.add(index)
            kept.append(homes[index])
        dropped["home"] += len(homes) - len(kept)
        homes = kept
    rows = rebuild(homes, others)

    def refuse_ok(current):
        return _share(current, lambda row: row["escalate"] != "act") >= 0.15

    guard = 0
    while rows and not refuse_ok(rows) and guard < 100000:
        guard += 1
        act = [row for row in rows if row["escalate"] == "act"]
        if not act:
            break
        counts = defaultdict(int)
        for row in rows:
            counts[row["verb"]] += 1
        homes_left = [row for row in act if row["verb"] == "home"]
        if homes_left and len(homes_left) / len(rows) > 0.08:
            victim = homes_left[len(homes_left) // 2]
        else:
            majority = max(counts, key=counts.get)
            if counts[majority] / len(rows) <= 0.12:
                break
            candidates = [row for row in act if row["verb"] == majority]
            if not candidates:
                break
            victim = candidates[len(candidates) // 2]
        rows.remove(victim)
        dropped[victim["verb"]] += 1

    guard = 0
    while rows and guard < 100000:
        guard += 1
        counts = defaultdict(int)
        for row in rows:
            counts[row["verb"]] += 1
        majority = max(counts, key=counts.get)
        if counts[majority] / len(rows) <= 0.35:
            break
        candidates = [row for row in rows if row["verb"] == majority and row["escalate"] == "act"]
        if not candidates:
            break
        victim = candidates[len(candidates) // 2]
        trial = [row for row in rows if row is not victim]
        if _share(trial, lambda row: row["escalate"] != "act") < 0.15:
            break
        rows = trial
        dropped[victim["verb"]] += 1
    return rows, dict(dropped)


def mark_overfit(selected):
    overfit = []
    episodes = []
    seen = set()
    for row in sorted(selected, key=lambda item: (item["episode_index"], item["frame_index"])):
        if row["split"] != "train":
            continue
        if row["episode_index"] not in seen:
            seen.add(row["episode_index"])
            episodes.append(row["episode_index"])
    by_episode = defaultdict(list)
    for row in selected:
        if row["split"] == "train":
            by_episode[row["episode_index"]].append(row)
    for episode in episodes:
        chunk = sorted(by_episode[episode], key=lambda item: item["frame_index"])
        overfit.extend(chunk)
        if len(overfit) >= 50:
            break
    keys = {(row["episode_index"], row["frame_index"], row["verb"]) for row in overfit}
    for row in selected:
        row["overfit"] = (row["episode_index"], row["frame_index"], row["verb"]) in keys
        row["selected"] = True
    return overfit


def attach_paths(events, videos):
    for event in events:
        video = videos[event["episode_index"]]
        path = (
            RAW
            / "videos"
            / "observation.images.image"
            / f"chunk-{video['chunk']:03d}"
            / f"file-{video['file']:03d}.mp4"
        )
        stamp = video["from_timestamp"] + event["frame_index"] / FPS
        image = ROOT / "data" / "frames" / "libero" / f"{event['episode_index']:04d}_{event['frame_index']:04d}.jpg"
        event["video"] = str(path)
        event["video_time"] = stamp
        event["image"] = str(image)


def extract_images(events):
    import av

    groups = defaultdict(list)
    for event in events:
        if Path(event["image"]).is_file() and Path(event["image"]).stat().st_size > 0:
            continue
        groups[event["video"]].append(event)
    for video, jobs in groups.items():
        jobs.sort(key=lambda item: item["video_time"])
        container = av.open(video)
        stream = container.streams.video[0]
        scale = float(stream.time_base)
        for event in jobs:
            container.seek(int(event["video_time"] / scale), stream=stream, backward=True, any_frame=False)
            saved = False
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if frame.pts * scale + 1e-3 >= event["video_time"]:
                    dest = Path(event["image"])
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_image().convert("RGB").save(dest, quality=90)
                    saved = True
                    break
            if not saved:
                raise RuntimeError(f"no frame at {event['video_time']} in {video}")
        container.close()


def _counts(rows):
    verbs = defaultdict(int)
    success = defaultdict(lambda: [0, 0])
    unsafe = 0
    escalate = defaultdict(int)
    for row in rows:
        verbs[row["verb"]] += 1
        success[row["verb"]][1] += 1
        success[row["verb"]][0] += int(row["p_success"])
        unsafe += int(row["p_unsafe"])
        escalate[row["escalate"]] += 1
    n = len(rows)
    refuse = sum(escalate[name] for name in ("ask_big", "ask_human", "abort"))
    majority = max(verbs.values()) / n if n else 0.0
    ps = sum(item[0] for item in success.values()) / n if n else 0.0
    return {
        "n": n,
        "episodes": len({row["episode_index"] for row in rows}),
        "verbs": {key: verbs[key] for key in sorted(verbs)},
        "verb_share": {key: round(verbs[key] / n, 4) for key in sorted(verbs)} if n else {},
        "success_count": {key: success[key][0] for key in sorted(success)},
        "success_rate": {
            key: round(success[key][0] / success[key][1], 4) for key in sorted(success) if success[key][1]
        },
        "p_success_positive_rate": round(ps, 4),
        "p_unsafe_positive_rate": round(unsafe / n, 4) if n else 0.0,
        "unsafe": unsafe,
        "escalate": dict(sorted(escalate.items())),
        "refuse_share": round(refuse / n, 4) if n else 0.0,
        "constant_baselines": {
            "majority_action": round(majority, 4),
            "p_success_true": round(ps, 4),
            "p_success_best_constant": round(max(ps, 1.0 - ps), 4),
            "p_unsafe_false": round(1.0 - (unsafe / n if n else 0.0), 4),
            "escalate_act": round(escalate["act"] / n, 4) if n else 0.0,
        },
    }


def _quantiles(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "p10": round(float(np.quantile(array, 0.10)), 2),
        "p50": round(float(np.quantile(array, 0.50)), 2),
        "p90": round(float(np.quantile(array, 0.90)), 2),
    }


def _gate(by_split):
    problems = []
    for split, counts in sorted(by_split.items()):
        if counts["refuse_share"] < 0.15:
            problems.append(f"{split} refuse {counts['refuse_share']} < 0.15")
        if counts["verb_share"] and max(counts["verb_share"].values()) > 0.35:
            problems.append(f"{split} verb share {max(counts['verb_share'].values())} > 0.35")
        rate = counts["p_success_positive_rate"]
        if rate > 0.90 or rate < 0.10:
            problems.append(f"{split} p_success {rate} is near-constant")
        if counts["constant_baselines"]["escalate_act"] > 0.85:
            problems.append(f"{split} constant act baseline {counts['constant_baselines']['escalate_act']}")
    return {"ok": not problems, "problems": problems}


def summarize(events, overfit, stats, dropped):
    by_split = defaultdict(list)
    for row in events:
        by_split[row["split"]].append(row)
    counted = {key: _counts(value) for key, value in sorted(by_split.items())}
    return {
        "label_version": 2,
        "source": "lerobot/libero",
        "not_emitted": ["place", "ask_human", "abort"],
        "why_not_emitted": {
            "place": "no receptacle pose; a falling ee_z while closed is a normal place-down and is not labeled unsafe",
            "ask_human": "no battery or familiarity channel",
            "abort": "no collision or limit channel, so collision-imminent does not fire",
        },
        "p_unsafe_definition": "a later frame jumps more than 5 cm in one 0.1 s step",
        "max_observed_step_m": round(stats["max_step_m"], 5),
        "p_unsafe_note": "observed steps stay near 1 cm, so this definition is honestly almost always false",
        "escalate_rule": "near-tie of executed motion scores -> ask_big; stall with flat scores -> ask_big; else act",
        "near_tie_episodes": stats["near_tie"],
        "stall_events": stats["stalls"],
        "stall_refuse_episodes": stats["stall_refuse"],
        "reach_lead_frames": _quantiles(stats["reach_lead_frames"]),
        "dropped_for_balance": dropped,
        "all": _counts(events),
        "by_split": counted,
        "overfit_images": len(overfit),
        "overfit_episodes": sorted({row["episode_index"] for row in overfit}),
        "gate": _gate(counted),
    }


def extract_selected(events_path):
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    extract_images(events)
    missing = [row["image"] for row in events if not Path(row["image"]).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} frames were not written")
    print(json.dumps({"extracted": len(events), "events": str(events_path)}))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--train-episodes", type=int, default=240)
    parser.add_argument("--dev-episodes", type=int, default=40)
    parser.add_argument("--calibration-episodes", type=int, default=50)
    parser.add_argument("--test-episodes", type=int, default=50)
    args = parser.parse_args()
    if args.extract:
        extract_selected(args.events)
        return
    budgets = {
        "train": args.train_episodes,
        "dev": args.dev_episodes,
        "calibration": args.calibration_episodes,
        "test": args.test_episodes,
    }
    tasks = load_tasks(RAW)
    videos = load_episode_videos(RAW)
    states = load_states(RAW)
    stats = {"max_step_m": 0.0, "near_tie": 0, "stall_refuse": 0, "stalls": 0, "reach_lead_frames": []}
    events = []
    available = defaultdict(set)
    for episode in sorted(states):
        split = episode_split(episode)
        available[split].add(episode)
        for event in label_episode(episode, states[episode], stats):
            event["split"] = split
            event["task"] = tasks.get(states[episode]["task"], "")
            events.append(event)
    selected = choose_images(events, budgets)
    balanced = []
    dropped = {}
    by_split = defaultdict(list)
    for event in selected:
        by_split[event["split"]].append(event)
    for split, rows in by_split.items():
        kept, note = balance_split(rows)
        dropped[split] = note
        balanced.extend(kept)
    overfit = mark_overfit(balanced)
    attach_paths(balanced, videos)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    with args.events.open("w") as stream:
        for event in balanced:
            stream.write(json.dumps(event) + "\n")
    report = summarize(balanced, overfit, stats, dropped)
    report["events_before_budget"] = len(events)
    report["episode_budgets"] = budgets
    report["episodes_available"] = {split: len(ids) for split, ids in sorted(available.items())}
    balanced_by = defaultdict(list)
    for event in balanced:
        balanced_by[event["split"]].append(event)
    report["rows"] = {split: len(rows) * 4 for split, rows in sorted(balanced_by.items())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "gate": report["gate"],
                "rows": report["rows"],
                "episodes_available": report["episodes_available"],
                "by_split_n": {split: counts["n"] for split, counts in report["by_split"].items()},
                "refuse": {split: counts["refuse_share"] for split, counts in report["by_split"].items()},
                "p_success": {
                    split: counts["p_success_positive_rate"] for split, counts in report["by_split"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
