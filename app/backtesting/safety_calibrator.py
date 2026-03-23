from collections import defaultdict


def calibrate_safety_thresholds(training_rows: list[dict]) -> dict:
    """
    P8.5 - Empirical safety calibration
    """

    # --- confidence threshold ---
    by_conf = defaultdict(list)
    for r in training_rows:
        c = r.get("confidence")
        reward = r.get("reward")
        if c is not None and reward is not None:
            by_conf[round(c, 2)].append(reward)

    min_confidence = 0.0
    for c in sorted(by_conf.keys()):
        avg = sum(by_conf[c]) / len(by_conf[c])
        if avg >= 0:
            min_confidence = c
            break

    # --- decision level blocking ---
    by_level = defaultdict(list)
    for r in training_rows:
        lvl = r.get("decision_level")
        reward = r.get("reward")
        if lvl and reward is not None:
            by_level[lvl].append(reward)

    block_levels = [
        lvl
        for lvl, rewards in by_level.items()
        if rewards and (sum(rewards) / len(rewards)) < 0
    ]

    # --- bucket blocking ---
    by_bucket = defaultdict(list)
    for r in training_rows:
        b = r.get("confidence_bucket")
        reward = r.get("reward")
        if b and reward is not None:
            by_bucket[b].append(reward)

    block_buckets = [
        b
        for b, rewards in by_bucket.items()
        if rewards and (sum(1 for r in rewards if r > 0) / len(rewards)) < 0.45
    ]

    return {
        "min_confidence": round(min_confidence, 3),
        "block_decision_levels": sorted(block_levels),
        "block_buckets": sorted(block_buckets),
    }
