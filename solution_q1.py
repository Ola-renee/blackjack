
import gymnasium as gym
import numpy as np
from collections import defaultdict
import random


def choose_action(state, Q, epsilon):
    
    state_tuple = state[0] if isinstance(state, tuple) else state
    if random.random() < epsilon:
        return env.action_space.sample() 
    else:
        return np.argmax(Q[state_tuple]) 


env = gym.make('Blackjack-v1', natural=False, sab=False)


Q = defaultdict(lambda: np.zeros(env.action_space.n))


alpha = 0.1 
gamma = 1.0 
epsilon = 0.1  
num_episodes = 50000  


for episode in range(num_episodes):

    current_state = env.reset()
    current_state = current_state[0] if isinstance(current_state, tuple) else current_state
    
    done = False
    while not done:
        
        action = choose_action(current_state, Q, epsilon)
        step_result = env.step(action)

        next_state = step_result[0]
        reward = step_result[1]
        done = step_result[2]
        

        next_state = next_state[0] if isinstance(next_state, tuple) else next_state
        next_best = np.max(Q[next_state])
        td_target = reward + gamma * next_best
        
        Q[current_state][action] += alpha * (td_target - Q[current_state][action]

        current_state = next_state

# Close the environment
env.close()


for state, actions in list(Q.items())[:5]:
    print(f"State: {state}, Actions: {actions}")
