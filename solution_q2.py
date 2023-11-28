import gymnasium as gym
import numpy as np
from collections import defaultdict


GAMMA = 0.99  
THETA = 1e-5  
MAX_EPISODES = 1000  


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

# Q2.2: Execute a Random Policy
def execute_random_policy(env, episodes=MAX_EPISODES):
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    reward_sums = defaultdict(lambda: defaultdict(float))
    state_action_counts = defaultdict(lambda: defaultdict(int))

    for _ in range(episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        done = False
        while not done:
            action = env.action_space.sample()
            step_result = env.step(action)

            
            next_state = step_result[0][0] if isinstance(step_result[0], tuple) else step_result[0]
            reward = step_result[1]
            done = step_result[2]

            
            transition_counts[state][action][next_state] += 1
            reward_sums[state][action] += reward
            state_action_counts[state][action] += 1
            state = next_state

    
    transition_probabilities = {
        s: {
            a: {
                ns: (transition_counts[s][a][ns] / state_action_counts[s][a])
                for ns in transition_counts[s][a]
            }
            for a in transition_counts[s]
        }
        for s in transition_counts
    }
    average_rewards = {
        s: {
            a: (reward_sums[s][a] / state_action_counts[s][a])
            for a in reward_sums[s]
        }
        for s in reward_sums
    }

    return transition_probabilities, average_rewards

transition_probabilities, average_rewards = execute_random_policy(env)

# Q2.3: Value Iteration
def value_iteration(env, transition_probabilities, average_rewards, gamma=GAMMA, theta=THETA):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = V[state]
            action_values = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                action_value = 0
                if state in transition_probabilities and action in transition_probabilities[state]:
                    for next_state, prob in transition_probabilities[state][action].items():
                        reward = average_rewards[state][action] if state in average_rewards and action in average_rewards[state] else 0
                        action_value += prob * (reward + gamma * V[next_state])
                action_values[action] = action_value
            best_action_value = np.max(action_values)
            delta = max(delta, np.abs(v - best_action_value))
            V[state] = best_action_value
        if delta < theta:
            break
    return V

V = value_iteration(env, transition_probabilities, average_rewards)


# Q2.4: Policy Extraction
def extract_policy(V, transition_probabilities, gamma=GAMMA):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            action_value = 0
            if state in transition_probabilities and action in transition_probabilities[state]:
                for next_state, prob in transition_probabilities[state][action].items():
                    reward = average_rewards[state][action] if state in average_rewards and action in average_rewards[state] else 0
                    action_value += prob * (reward + gamma * V[next_state])
            action_values[action] = action_value
        policy[state] = np.argmax(action_values)
    return policy

policy = extract_policy(V, transition_probabilities)


# Q2.5: Execute the Optimal Policy
def execute_optimal_policy(env, policy, episodes=MAX_EPISODES):
    total_reward = 0
    for _ in range(episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state  # Handle tuple state
        done = False
        while not done:
            action = policy[state]
            step_result = env.step(action)  # Execute the action

            # Flexible unpacking for different lengths of return values
            next_state, reward, done = step_result[:3]
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state  # Handle tuple state

            total_reward += reward
            state = next_state
    return total_reward

total_reward = execute_optimal_policy(env, policy)
print(f"Total reward accrued with the optimal policy over {MAX_EPISODES} episodes: {total_reward}")


# Close the environment
env.close()

