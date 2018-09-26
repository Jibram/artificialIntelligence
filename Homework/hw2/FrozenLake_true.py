import gym
import numpy as np
from gym.envs.registration import register

register(
	id='FrozenLake8x8-v3',
	entry_point='gym.envs.toy_text:FrozenLakeEnv',
	kwargs={'map_name': '8x8',
			'is_slippery':True}
)

env = gym.make('FrozenLake8x8-v3')

# Your code start here...
# Q value matrix
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Constants that were messed around with to get a decent looking result.
episodes = 5000
discount = .99
learning_rate = .9
max_action = 100

for i in range(episodes):
    state = env.reset()
    done = False

    for j in range(max_action):
	# .5 times a random noise that occasionally comes back up to be a bit more adventurous
        action = np.argmax(Q[state] + .3 * (np.random.randn(1, env.action_space.n) * (1. / ((i % 100) + 1))))
        new_state, reward, done, _ = env.step(action)

	# Learning rate affected Q-Iteration
        sample = reward + discount * np.amax(Q[new_state])
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * sample

        if done and not new_state == 63:
            Q[state, action] -= 1

        state = new_state

        if done:
            break

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
