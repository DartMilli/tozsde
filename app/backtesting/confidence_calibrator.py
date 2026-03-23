from collections import defaultdict


def calibrate_confidence(training_rows: list[dict]) -> dict:
    """
    P8.4 - Empirical confidence recalibration
    """

    buckets = defaultdict(list)

    for r in training_rows:
        b = r.get("confidence_bucket")
        reward = r.get("reward")

        if b is None or reward is None:
            continue

        buckets[b].append(reward)

    calibration = {}

    for b, rewards in buckets.items():
        if not rewards:
            continue

        win_rate = sum(1 for r in rewards if r > 0) / len(rewards)

        calibration[b] = round(win_rate, 3)

    return calibration
