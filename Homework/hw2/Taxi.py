import gym
import numpy as np

env = gym.make('Taxi-v2')

# your code start here...
# Sets up the Q value matrix
Q = np.zeros([env.observation_space.n, env.action_space.n])
episodes = 5000
discount = .9

for i in range(episodes):
    state = env.reset()
    done = False
    # For each episode, it tries to iterate until it falls in a hole or finds the goal.
    while not done:
	# Decides the next actions based on the largest Q value for each given state, plus some noise for learning
        action = np.argmax(Q[state] + np.random.rand(1, env.action_space.n) / (i+1))
	
        new_state, reward, done, _ = env.step(action)
	
	# Q value iteration
        Q[state, action] = reward + discount * np.max(Q[new_state,:])
        state = new_state

# For printing purposes
state = env.reset()
count = 0
actionstr = ''
done = False
while not done:
    count += 1
    action = int(np.argmax(Q[state]))
    new_state, reward, done, _ = env.step(action)
    state = new_state
    print('#### ', count, ' action')
    env.render()
