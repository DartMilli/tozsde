from collections import defaultdict


def audit_confidence_buckets(rows: list[dict]) -> dict:
    buckets = defaultdict(list)

    for r in rows:
        b = r.get("confidence_bucket", "UNKNOWN")
        if r.get("reward") is not None:
            buckets[b].append(r["reward"])

    summary = {}
    for b, rewards in buckets.items():
        if not rewards:
            continue
        summary[b] = {
            "count": len(rewards),
            "avg_reward": round(sum(rewards) / len(rewards), 6),
            "win_rate": round(
                sum(1 for x in rewards if x > 0) / len(rewards),
                3,
            ),
        }

    return summary


def audit_decision_levels(rows: list[dict]) -> dict:
    levels = defaultdict(list)

    for r in rows:
        lvl = r.get("decision_level", "UNKNOWN")
        if r.get("reward") is not None:
            levels[lvl].append(r["reward"])

    out = {}
    for lvl, rewards in levels.items():
        if not rewards:
            continue
        out[lvl] = {
            "count": len(rewards),
            "avg_reward": round(sum(rewards) / len(rewards), 6),
            "win_rate": round(
                sum(1 for r in rewards if r > 0) / len(rewards),
                3,
            ),
        }

    return out


def detect_overconfidence(rows: list[dict]) -> list[dict]:
    flags = []

    for r in rows:
        conf = r.get("confidence")
        reward = r.get("reward")

        if conf is None or reward is None:
            continue

        if conf >= 0.7 and reward < 0:
            flags.append(
                {
                    "timestamp": r["timestamp"],
                    "confidence": conf,
                    "reward": reward,
                    "decision_level": r.get("decision_level"),
                }
            )

    return flags
