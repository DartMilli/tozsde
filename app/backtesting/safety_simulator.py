def simulate_safety_policy(
    training_rows: list[dict],
    policy: dict,
) -> dict:
    """
    P8.6 – Counterfactual safety simulation
    """

    total_reward_all = 0.0
    total_reward_safe = 0.0

    trades_all = 0
    trades_safe = 0

    wins_all = 0
    wins_safe = 0

    for r in training_rows:
        reward = r.get("reward", 0.0)
        confidence = r.get("confidence")
        level = r.get("decision_level")
        bucket = r.get("confidence_bucket")

        # --- baseline ---
        trades_all += 1
        total_reward_all += reward
        if reward > 0:
            wins_all += 1

        # --- safety decision ---
        blocked = False

        if confidence is None or confidence < policy["min_confidence"]:
            blocked = True

        if level in policy.get("block_decision_levels", []):
            blocked = True

        if bucket in policy.get("block_buckets", []):
            blocked = True

        if not blocked:
            trades_safe += 1
            total_reward_safe += reward
            if reward > 0:
                wins_safe += 1

    return {
        "baseline": {
            "trades": trades_all,
            "total_reward": round(total_reward_all, 3),
            "avg_reward": (
                round(total_reward_all / trades_all, 4) if trades_all else 0.0
            ),
            "win_rate": round(wins_all / trades_all, 3) if trades_all else 0.0,
        },
        "safety": {
            "trades": trades_safe,
            "total_reward": round(total_reward_safe, 3),
            "avg_reward": (
                round(total_reward_safe / trades_safe, 4) if trades_safe else 0.0
            ),
            "win_rate": round(wins_safe / trades_safe, 3) if trades_safe else 0.0,
        },
        "delta": {
            "reward_gain": round(total_reward_safe - total_reward_all, 3),
            "trades_removed": trades_all - trades_safe,
        },
    }
