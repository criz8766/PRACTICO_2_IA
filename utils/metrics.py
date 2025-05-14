# utils/metrics.py

import numpy as np
import time

def compute_metrics(reward_history, start_time, end_time, threshold=1.0, window=10):
    rewards = np.array(reward_history)
    avg_reward = np.mean(rewards)
    variance = np.var(rewards)
    training_time = end_time - start_time

    convergence_episode = len(rewards)
    for i in range(len(rewards) - window):
        recent_window = rewards[i:i+window]
        if np.max(recent_window) - np.min(recent_window) < threshold:
            convergence_episode = i + window
            break

    return {
        'avg_reward': avg_reward,
        'variance': variance,
        'training_time': training_time,
        'convergence_episode': convergence_episode
    }
