import numpy as np
from matplotlib import pyplot as plt

learning_data = np.load("/home/john/code/pythonfiles/my_dqn/learning_data.npy")
time_vals = [learning_data[i][0] for i in range(len(learning_data))]
Q_vals = [learning_data[i][1] for i in range(len(learning_data))]
reward_vals = [learning_data[i][2] for i in range(len(learning_data))]
max_reward_vals = [learning_data[i][3] for i in range(len(learning_data))]
step_vals = [learning_data[i][4] for i in range(len(learning_data))]
loss = [learning_data[i][5] for i in range(len(learning_data))]
prob = [learning_data[i][6] for i in range(len(learning_data))]
avg_grad = [learning_data[i][7] for i in range(len(learning_data))]

weight_data = np.load("/home/john/code/pythonfiles/my_dqn/weight_averages.npy")
layer1_weights = [weight_data[i][0] for i in range(len(weight_data))]
layer1_bias = [weight_data[i][1] for i in range(len(weight_data))]
layer2_weights = [weight_data[i][2] for i in range(len(weight_data))]
layer2_bias = [weight_data[i][3] for i in range(len(weight_data))]
layer3_weights = [weight_data[i][4] for i in range(len(weight_data))]
layer3_bias = [weight_data[i][5] for i in range(len(weight_data))]
layer4_weights = [weight_data[i][6] for i in range(len(weight_data))]
layer4_bias = [weight_data[i][7] for i in range(len(weight_data))]

plt.ylabel("Average reward over 20 episodes")

plt.plot(time_vals, avg_grad)
# plt.plot(time_vals, layer1_bias)
plt.show()
