import numpy as np


def normalize_dqn_confidence(q_values, action):
    """
    Normalize DQN Q-values to [0, 1] range using softmax.
    """
    q = q_values.detach().cpu().numpy()[0]
    exp_q = np.exp(q - np.max(q))
    probs = exp_q / np.sum(exp_q)
    return float(probs[int(action)])


def normalize_ppo_confidence(action_probs):
    """
    PPO confidence already probability-based.
    """
    return float(action_probs)


def apply_confidence(model_output, confidence: float):
    model_output.confidence = confidence
    return model_output


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(value, max_val))


def normalize_final_confidence(raw_confidence: float) -> float:
    """
    Clamp confidence into [0,1] without implicit floors.
    """
    if not isinstance(raw_confidence, (int, float)):
        return 0.0
    return clamp(raw_confidence, 0.0, 1.0)
