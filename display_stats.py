import numpy as np
from matplotlib import pyplot as plt

learning_data = np.load("/home/john/code/pythonfiles/my_dqn/learning_data2.npy")
time_vals = [learning_data[i][0] for i in range(len(learning_data))]
Q_vals = [learning_data[i][1] for i in range(len(learning_data))]
reward_vals = [learning_data[i][2] for i in range(len(learning_data))]
max_reward_vals = [learning_data[i][3] for i in range(len(learning_data))]
step_vals = [learning_data[i][4] for i in range(len(learning_data))]
loss = [learning_data[i][5] for i in range(len(learning_data))]
prob = [learning_data[i][6] for i in range(len(learning_data))]

weight_data = np.load("/home/john/code/pythonfiles/my_dqn/weight_averages2.npy")
layer1_weights = [weight_data[i][0] for i in range(len(weight_data))]
layer1_bias = [weight_data[i][1] for i in range(len(weight_data))]
layer2_weights = [weight_data[i][2] for i in range(len(weight_data))]
layer2_bias = [weight_data[i][3] for i in range(len(weight_data))]
layer3_weights = [weight_data[i][4] for i in range(len(weight_data))]
layer3_bias = [weight_data[i][5] for i in range(len(weight_data))]
layer4_weights = [weight_data[i][6] for i in range(len(weight_data))]
layer4_bias = [weight_data[i][7] for i in range(len(weight_data))]

plt.ylabel("Q values over 20 episodes")

for i in reward_vals:
    if i > 250:
        print(reward_vals.index(i))




plt.plot(time_vals, Q_vals)
# plt.plot(time_vals, layer1_bias)
plt.show()


