import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim

# Define rotation angle
theta = 2 * np.pi / 100  # Rotation angle per step

# Build rotation matrix
A = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# Input matrices
B1 = np.array([[0], [1]])
B2 = np.array([[0], [0.9]])
B3 = np.array([[0], [0.8]])

# Communication topology (adjacency matrix)
adj_matrix = np.array([
    [0, 0, 0],  # Neighbors of agent 1
    [1, 0, 0],  # Neighbors of agent 2
    [0, 1, 0]  # Neighbors of agent 3
])

# Connection weights with the leader
b = np.array([1, 1, 0])  # b1=1, b2=1, b3=0

# Control gain
K = 0.1

# Simulation parameters
steps = 300

# Initial states
state0 = np.array([2.0, 0.0])  # Leader initial state
state1 = np.array([3.0, 1.0])  # Follower 1 initial state
state2 = np.array([2.0, 1.5])  # Follower 2 initial state
state3 = np.array([-1.0, 2.0])  # Follower 3 initial state

# Trajectories
trajectory0 = []  # Leader trajectory
trajectory1 = []  # Follower 1 trajectory
trajectory2 = []  # Follower 2 trajectory
trajectory3 = []  # Follower 3 trajectory

current_state0 = state0
current_state1 = state1
current_state2 = state2
current_state3 = state3

# Tracking errors
tracking_errors = []


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value

    def act(self, state):
        policy, _ = self.forward(state)
        action = torch.multinomial(policy, num_samples=1).item()
        return action

    def compute_td_error(self, state, action, reward, next_state, done):
        def custom_environment(state, action, reward, next_state, done):
            # Update agent states
            next_state1 = np.dot(A, state1) + np.dot(B1, np.array([action]))
            next_state2 = np.dot(A, state2) + np.dot(B2, np.array([action]))
            next_state3 = np.dot(A, state3) + np.dot(B3, np.array([action]))

            # Calculate errors
            e1 = b[0] * (next_state1 - state0)
            e2 = adj_matrix[1, 0] * (next_state2 - next_state1) + b[1] * (next_state2 - state0)
            e3 = adj_matrix[2, 1] * (next_state3 - next_state2)

            # Calculate reward
            reward = [-np.linalg.norm(e1), -np.linalg.norm(e2), -np.linalg.norm(e3)]

            # Check if done
            done = False

            return reward, next_state1, next_state2, next_state3, done

        policy, value = self.forward(state)
        next_policy, next_value = self.forward(next_state)
        reward, next_state1, next_state2, next_state3, done = custom_environment(state, action, reward, next_state,
                                                                                 done)
        td_error = custom_environment(state, action, reward, next_state, done) - value
        return td_error

    def learn(self, state1, state2, state3, action1, action2, action3, reward, next_state1, next_state2, next_state3,
              done):
        policy1, value1 = self.forward(state1)
        policy2, value2 = self.forward(state2)
        policy3, value3 = self.forward(state3)

        td_error1 = self.compute_td_error(state1, action1, reward[0], next_state1, done[0])
        td_error2 = self.compute_td_error(state2, action2, reward[1], next_state2, done[1])
        td_error3 = self.compute_td_error(state3, action3, reward[2], next_state3, done[2])

        actor_loss1 = -torch.log(policy1[action1]) * td_error1.detach()
        actor_loss2 = -torch.log(policy2[action2]) * td_error2.detach()
        actor_loss3 = -torch.log(policy3[action3]) * td_error3.detach()

        critic_loss1 = td_error1 ** 2
        critic_loss2 = td_error2 ** 2
        critic_loss3 = td_error3 ** 2

        loss = actor_loss1 + critic_loss1 + actor_loss2 + critic_loss2 + actor_loss3 + critic_loss3
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Simulation process
agent = ActorCritic(state_dim=6, action_dim=2)

for k in range(steps):
    # Calculate tracking errors
    e1 = b[0] * (current_state1 - state0)
    e2 = adj_matrix[1, 0] * (current_state2 - current_state1) + b[1] * (current_state2 - state0)
    e3 = adj_matrix[2, 1] * (current_state3 - current_state2)
    e = np.concatenate((e1, e2, e3))

    # Use AC network to generate actions
    action1 = agent.act(torch.tensor(e1).float())
    action2 = agent.act(torch.tensor(e2).float())
    action3 = agent.act(torch.tensor(e3).float())

    # Update states
    current_state1 = np.dot(A, current_state1) + np.dot(B1, np.array([action1]))
    current_state2 = np.dot(A, current_state2) + np.dot(B2, np.array([action2]))
    current_state3 = np.dot(A, current_state3) + np.dot(B3, np.array([action3]))

    # Record state trajectories and tracking errors
    trajectory0.append(state0.copy())
    trajectory1.append(current_state1.copy())
    trajectory2.append(current_state2.copy())
    trajectory3.append(current_state3.copy())
    tracking_errors.append([np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)])

    # Train AC network
    state = np.concatenate((e1, e2, e3))
    next_state = np.concatenate((np.array([np.linalg.norm(e1[0]), np.linalg.norm(e1[1])]),
                                 np.array([np.linalg.norm(e2[0]), np.linalg.norm(e2[1])]),
                                 np.array([np.linalg.norm(e3[0]), np.linalg.norm(e3[1])])))
    reward = [-np.linalg.norm(e1), -np.linalg.norm(e2), -np.linalg.norm(e3)]
    done = [False, False, False]
    agent.learn(state1, state2, state3, action1, action2, action3, reward, next_state[1], next_state[2], next_state[3], done)

# Convert to numpy arrays
trajectory0 = np.array(trajectory0)
trajectory1 = np.array(trajectory1)
trajectory2 = np.array(trajectory2)
trajectory3 = np.array(trajectory3)
tracking_errors = np.array(tracking_errors)

# Plot state trajectories
plt.figure(figsize=(8, 6))
ax1 = plt.axes(projection='3d')
ax1.plot3D(trajectory0[:, 0], np.arange(len(trajectory0)), trajectory0[:, 1], 'g*', label='x0', markersize=10)
ax1.plot3D(trajectory1[:, 0], np.arange(len(trajectory1)), trajectory1[:, 1], 'b--', label='x1', markersize=1)
ax1.plot3D(trajectory2[:, 0], np.arange(len(trajectory2)), trajectory2[:, 1], 'r--', label='x2', markersize=1)
ax1.plot3D(trajectory3[:, 0], np.arange(len(trajectory3)), trajectory3[:, 1], 'k:', label='x3', markersize=0.5)
ax1.set_xlabel('x1')
ax1.set_ylabel('Time step')
ax1.set_zlabel('x2')
ax1.view_init(elev=20, azim=-40)
ax1.legend()
ax1.grid(False)
ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.tight_layout()
plt.show()

# Plot tracking errors
plt.figure(figsize=(8, 6))
ax2 = plt.axes()
ax2.plot(tracking_errors[:, 0], 'b-', label='e1')
ax2.plot(tracking_errors[:, 1], 'r--', label='e2')
ax2.plot(tracking_errors[:, 2], 'k:', label='e3')
ax2.set_xlabel('Time step')
ax2.set_ylabel('Tracking Error')
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()