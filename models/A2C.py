import time

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from numpy.random import normal
import numpy as np

# SOURCE : Structure of the code is heavily inspired by
# https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
# we modified the networks to our needs
# we understood the DPG algorithm
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class A2C_agent:
    """class Q_net(nn.Module):
        def __init__(self, state_size, action_dim, hidden_size, n_frames_state):
            super(A2C_agent.Q_net, self).__init__()

            # input size is assumed to be a multiple of 16 (easier for us to think about it)
            self.conv2D_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
            # input_size / 2 ; 64
            self.maxpool2D_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # input_size / 4 ; 64
            self.conv2D_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)
            # input_size / 8 ; 64
            self.maxpool2D_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # input_size / 16 ; 64

            cnn_state_size = int(state_size / 16)
            dense_input_size = ((cnn_state_size **2) * 8)

            self.dense1 = nn.Linear(dense_input_size, hidden_size)
            self.dense2 = nn.Linear(hidden_size, hidden_size)
            self.dense3 = nn.Linear(hidden_size, out_features=81)
        def forward(self, state):
            unstacked_state = state.unbind(2) # unstack into list of size n_frames_state
            unstacked_state = unstacked_state[0]

            x = self.conv2D_1(unstacked_state)
            x = F.relu(x)
            x = self.maxpool2D_1(x)
            x = self.conv2D_2(x)
            x = F.relu(x)
            x = self.maxpool2D_2(x)
            x = torch.flatten(x, 1)

            x = self.dense1(x)
            x = F.relu(x)
            x = self.dense2(x)
            x = F.relu(x)
            x = self.dense3(x)

            return x

    class Policy_net(nn.Module):
        def __init__(self, state_size, action_dim, hidden_size, n_frames_state):
            super(A2C_agent.Policy_net, self).__init__()

            # input size is assumed to be a multiple of 16 (easier for us to think about it)
            self.conv2D_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
            # input_size / 2 ; 64
            self.maxpool2D_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # input_size / 4 ; 64
            self.conv2D_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1)
            # input_size / 8 ; 64
            self.maxpool2D_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # input_size / 16 ; 64

            cnn_state_size = int(state_size / 16)
            dense_input_size = ((cnn_state_size **2) * 8)

            self.dense1 = nn.Linear(dense_input_size, hidden_size)
            self.dense2 = nn.Linear(hidden_size, hidden_size)
            self.dense3 = nn.Linear(hidden_size, out_features=81)

            self.soft_max = nn.Softmax(dim=1)

        def forward(self, state):
            unstacked_state = state.unbind(2) # unstack into list of size n_frames_state
            unstacked_state = unstacked_state[0]

            x = self.conv2D_1(unstacked_state)
            x = F.relu(x)
            x = self.maxpool2D_1(x)
            x = self.conv2D_2(x)
            x = F.relu(x)
            x = self.maxpool2D_2(x)
            x = torch.flatten(x, 1)

            x = self.dense1(x)
            x = F.relu(x)
            x = self.dense2(x)
            x = F.relu(x)
            x = self.dense3(x)
            x = self.soft_max(x)

            return x"""
    class Q_net(nn.Module):
        def __init__(self, state_size, action_dim, hidden_size, n_frames_state):
            super(A2C_agent.Q_net, self).__init__()

            self.n_frames_state = n_frames_state
            self.conv2D_1 = []
            self.maxpool2D_1 = []
            self.batchnorm2d_1 = []
            self.conv2D_2 = []
            self.batchnorm2d_2 = []
            self.maxpool2D_2 = []
            self.conv2D_3 = []
            self.batchnorm2d_3 = []

            filters = 8
            for n_th_frame_net in range(n_frames_state):
                # input size is assumed to be a multiple of 16 (easier for us to think about it)
                self.conv2D_1.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=0).to(device))
                self.batchnorm2d_1.append(nn.BatchNorm2d(num_features=32).to(device))
                # input_size / 2 ; 64
                # self.maxpool2D_1.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                # input_size / 4 ; 64
                self.conv2D_2.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0).to(device))
                self.batchnorm2d_2.append(nn.BatchNorm2d(num_features=64).to(device))

                # input_size / 8 ; 64
                # self.maxpool2D_2.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                # input_size / 16 ; 64
                self.conv2D_3.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0).to(device))
                self.batchnorm2d_3.append(nn.BatchNorm2d(num_features=64).to(device))

                cnn_state_size = int(state_size / 8)
                dense_input_size = 4*4*64#((cnn_state_size ** 2) * filters)

            self.dense_all_1 = nn.Linear(dense_input_size * n_frames_state, hidden_size * n_frames_state).to(device)
            self.batchnormd_dense_1 = nn.BatchNorm1d(num_features=hidden_size * n_frames_state).to(device)
            self.dense_all_2 = nn.Linear(hidden_size * n_frames_state, 25).to(device)

        def forward(self, state):

            unstacked_state = state.unbind(2)  # unstack into list of size n_frames_state

            outs = []
            for f in range(self.n_frames_state):
                x = self.conv2D_1[f](unstacked_state[f])
                x = F.relu(x)
                x = self.batchnorm2d_1[f](x)
                # x = self.maxpool2D_1[f](x)
                x = self.conv2D_2[f](x)
                x = F.relu(x)
                x = self.batchnorm2d_2[f](x)
                x = self.conv2D_3[f](x)
                x = F.relu(x)
                x = self.batchnorm2d_3[f](x)
                # x = self.maxpool2D_2[f](x)
                x = torch.flatten(x, 1)

                outs.append(x)

            x = torch.cat(outs, 1)

            x = self.dense_all_1(x)
            x = F.relu(x)
            x = self.batchnormd_dense_1(x)
            x = self.dense_all_2(x)

            return x


    class Policy_net(nn.Module):
        def __init__(self, state_size, action_dim, hidden_size, n_frames_state):
            super(A2C_agent.Policy_net, self).__init__()

            self.n_frames_state = n_frames_state
            self.conv2D_1 = []
            self.maxpool2D_1 = []
            self.batchnorm2d_1 = []
            self.conv2D_2 = []
            self.batchnorm2d_2 = []
            self.maxpool2D_2 = []
            self.conv2D_3 = []
            self.batchnorm2d_3 = []
            filters = 8
            for n_th_frame_net in range(n_frames_state):
                # input size is assumed to be a multiple of 16 (easier for us to think about it)
                self.conv2D_1.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=0).to(device))
                self.batchnorm2d_1.append(nn.BatchNorm2d(num_features=32).to(device))
                # input_size / 2 ; 64
                # self.maxpool2D_1.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                # input_size / 4 ; 64
                self.conv2D_2.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0).to(device))
                self.batchnorm2d_2.append(nn.BatchNorm2d(num_features=64).to(device))

                # input_size / 8 ; 64
                # self.maxpool2D_2.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                # input_size / 16 ; 64
                self.conv2D_3.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0).to(device))
                self.batchnorm2d_3.append(nn.BatchNorm2d(num_features=64).to(device))

                cnn_state_size = int(state_size / 8)
                dense_input_size = 4*4*64#((cnn_state_size ** 2) * filters)

            self.dense_all_1 = nn.Linear(dense_input_size * n_frames_state, hidden_size * n_frames_state).to(device)
            self.batchnormd_dense_1 = nn.BatchNorm1d(num_features=hidden_size * n_frames_state).to(device)
            self.dense_all_2 = nn.Linear(hidden_size * n_frames_state, 25).to(device)
            self.soft_max = nn.Softmax(dim=1).to(device)

        def forward(self, state):
            unstacked_state = state.unbind(2)  # unstack into list of size n_frames_state

            outs = []
            for f in range(self.n_frames_state):
                x = self.conv2D_1[f](unstacked_state[f])
                x = F.relu(x)
                x = self.batchnorm2d_1[f](x)
                # x = self.maxpool2D_1[f](x)
                x = self.conv2D_2[f](x)
                x = F.relu(x)
                x = self.batchnorm2d_2[f](x)
                x = self.conv2D_3[f](x)
                x = F.relu(x)
                x = self.batchnorm2d_3[f](x)
                # x = self.maxpool2D_2[f](x)
                x = torch.flatten(x, 1)

                outs.append(x)

            y = torch.cat(outs, 1)

            y = self.dense_all_1(y)
            y = F.relu(y)
            y = self.batchnormd_dense_1(y)
            y = self.dense_all_2(y)

            y = self.soft_max(y+1e-8)

            return y

    class ExperienceReplay:
        def __init__(self, max_buffer_size):
            self.max_buffer_size = max_buffer_size
            self.buffer = deque(maxlen=max_buffer_size)

        def push(self, tuple):
            self.buffer.append(tuple)

        def sample(self, n_samples):
            if n_samples > len(self.buffer):
                return random.choices(self.buffer, k=n_samples)
            else:
                return random.sample(self.buffer, n_samples)

    def __init__(self, state_size,
                 n_frames_state,
                 game_engine,
                 discount_factor=0.99,
                 batch_size=32,
                 evaluate_batch_step=20,
                 evaluate_n_agents=5,
                 evaluate_animate=False,
                 max_steps=20000,
                 max_steps_evaluation=1000):

        self.state_size = state_size
        self.game_engine = game_engine
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.evaluate_batch_step = evaluate_batch_step
        self.evaluate_n_agents = evaluate_n_agents
        self.evaluate_animate = evaluate_animate
        self.max_steps = max_steps
        self.max_steps_evaluation = max_steps_evaluation

        self.q_net = A2C_agent.Q_net(state_size, action_dim=2, hidden_size=200, n_frames_state=n_frames_state)
        self.q_net_target = A2C_agent.Q_net(state_size, action_dim=2, hidden_size=200, n_frames_state=n_frames_state)
        self.policy_net = A2C_agent.Policy_net(state_size, action_dim=2, hidden_size=200, n_frames_state=n_frames_state)
        self.policy_net_target = A2C_agent.Policy_net(state_size, action_dim=2, hidden_size=200, n_frames_state=n_frames_state)

        self.q_net_learning_rate = 0.0001
        self.q_net_optimizer = optim.Adam(self.q_net.parameters(), lr=self.q_net_learning_rate)
        self.q_net_loss_function = nn.MSELoss()


        self.policy_net_learning_rate = 0.0001
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_net_learning_rate)

        # We initialize the target networks as copies of the original networks
        for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.policy_net_target.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.replay_buffer = A2C_agent.ExperienceReplay(max_buffer_size=65000)

        self.update_targets = "soft" #"soft"

    def update(self, states, actions, rewards, next_states, is_terminal, indexes=None):
        curr_Q = self.q_net.forward(states).gather(1, actions.view(actions.size(0), 1))
        next_Q = self.q_net_target.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards +  (1 - is_terminal) * self.discount_factor * max_next_Q

        loss = self.q_net_loss_function(curr_Q, expected_Q.detach())

        curr_Q_all_actions = self.q_net(states)
        current_policy_prob_actions = self.policy_net(states)
        mean_current_state = (curr_Q_all_actions * current_policy_prob_actions).sum(1).unsqueeze(1)

        advantages = expected_Q - mean_current_state
        log_probs = torch.log(self.policy_net.forward(states).gather(1, actions.view(actions.size(0), 1)) + 1e-8)
        actor_loss = -(advantages.detach() * log_probs).mean()

        self.q_net_optimizer.zero_grad()
        loss.backward()
        self.q_net_optimizer.step()

        self.policy_net_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_net_optimizer.step()

        if self.update_targets == "soft":
            tau = 0.001
            # update target networks
            for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(self.policy_net_target.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def get_actions(self, states, target_network=False, return_indexes=False):
        with torch.no_grad():
            if target_network is True:
                self.policy_net_target.eval()
                probs = self.policy_net_target(states).detach()
                self.policy_net_target.train()
            else:
                self.policy_net.eval()
                probs = self.policy_net(states).detach()
                self.policy_net.train()

        """action_indexes = np.zeros(probs.shape[0])
        for i, dist in enumerate(probs):
            action_indexes[i] = np.random.choice(list(range(81)), size=1, p=dist)
        action_indexes = action_indexes.astype(int)"""

        m = torch.distributions.categorical.Categorical(probs=probs)
        action_indexes = m.sample()

        #print(probs.std(axis=0))

        if return_indexes is True:
            return action_indexes

        actions = np.zeros((action_indexes.shape[0], 2))

        for batch_index, index in enumerate(action_indexes):
            actions[batch_index, :] = self.index_to_action(index)

        return actions

    def index_to_action(self, index):
        actions_x = [-1, -0.5, 0, 0.5, 1]
        actions_y = [-1, -0.5, 0, 0.5, 1]
        #actions_x = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        #actions_y = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        try:
            input_x = actions_x[index % len(actions_x)]
            input_y = actions_y[int(index/len(actions_x))]
        except:
            print(index)
        return np.array([input_x, input_y])

    def action_to_index(self, input_x, input_y):
        actions_x = [-1, -0.5, 0, 0.5, 1]
        actions_y = [-1, -0.5, 0, 0.5, 1]
        #actions_x = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        #actions_y = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        return actions_y.index(input_y) * len(actions_x) + actions_x.index(input_x)

    def interact(self, states, states_identifiers, steps, epsilon):
        # get actions from policy network
        actions = self.get_actions(states, return_indexes=True)
        actions_float = np.zeros((actions.shape[0], 2))
        for i in range(actions.shape[0]):
            r = np.random.uniform(low=0, high=1)
            if r <= epsilon:
                index = np.random.randint(low=0, high=25)
                actions_float[i, :] = self.index_to_action(index)
            else:
                actions_float[i, :] = self.index_to_action(actions[i])


        # reformat actions for the game engine
        actions_buffer = {}
        for i, action in enumerate(actions):  # if several games in //
            actions_buffer[states_identifiers[i]] = {"input_x": actions_float[i, 0], "input_y": actions_float[i, 1]}

        # act on the envs. and get the results
        tuples = self.game_engine.act_and_observe(actions_buffer)



        current_identifiers = []
        current_rewards = []
        current_is_terminal = []

        # add to the replay memory AND REFORMAT to pytorch
        for tuple in tuples:  # if several games in //.
            tuple["state"] = torch.from_numpy(np.stack(tuple["state"])).float()
            tuple["action"] = torch.tensor(
                self.action_to_index(tuple["action"]["input_x"], tuple["action"]["input_y"]))
            tuple["reward"] = torch.tensor(tuple["reward"]).float()
            tuple["new_state"] = torch.from_numpy(np.stack(tuple["new_state"])).float()
            tuple["is_terminal"] = torch.from_numpy(tuple["is_terminal"]).float()
            current_identifiers.append(tuple["identifier"])
            current_rewards.append(tuple["reward"])
            current_is_terminal.append(tuple["is_terminal"])

            self.replay_buffer.push(tuple)

        for i in range(1):
            samples = self.replay_buffer.sample(self.batch_size)

            states = []
            actions = []
            new_states = []
            rewards = []
            is_terminal = []
            for sample in samples:
                states.append(sample["state"])
                actions.append(sample["action"])
                new_states.append(sample["new_state"])
                rewards.append(sample["reward"])
                is_terminal.append(sample["is_terminal"])

            states = torch.stack(states).unsqueeze(1).to(device)
            actions = torch.stack(actions).unsqueeze(1).to(device)
            rewards = torch.stack(rewards).unsqueeze(1).to(device)
            new_states = torch.stack(new_states).unsqueeze(1).to(device)
            is_terminal = torch.stack(is_terminal).unsqueeze(1).to(device)

            self.update(states, actions, rewards, new_states, is_terminal)

        return current_rewards, current_is_terminal, current_identifiers

    def evaluate_policy(self, num_frames):
        print("###### EVALUATION ###### \n\n\n")

        games_dict = self.game_engine.request_new_games(n_games=self.evaluate_n_agents, name="test", pooled=False)
        identifiers = list(games_dict.keys())

        sum_rewards = {id: 0 for id in identifiers}
        games_alive = {id for id in identifiers}
        terminaison_steps = {id: 0 for id in identifiers}
        remaining_pallets = {id: 0 for id in identifiers}
        step = 0

        if self.evaluate_animate:
            n_columns_windows = 7
            positions_windows = {}
            for index_agent, game in enumerate(games_dict.values()):
                x_window = index_agent % n_columns_windows
                y_window = int(index_agent / n_columns_windows)
                x_window = (x_window * (self.state_size + 50)) + 50
                y_window = (y_window * (self.state_size + 50)) + 50
                positions_windows[game.get_identifier()] = [x_window, y_window]

        while True:
            print(step)
            if step > self.max_steps_evaluation:
                for game in games_dict.values():
                    # self.game_engine.remove_game(game.get_identifier())
                    cv2.waitKey(1)
                    cv2.destroyWindow(str(game.get_identifier()))
                    cv2.waitKey(1)
                break

            # fetch states
            states = []
            for game in games_dict.values():
                states.append(game.get_current_state()["state"])

            states = Variable(torch.from_numpy(np.stack(states)).float().unsqueeze(1)).to(device)
            actions = self.get_actions(states)

            actions_buffer = dict()
            for index, (id, game) in enumerate(games_dict.items()):
                actions_buffer[id] = {"input_x": actions[index][0].astype(float),
                                      "input_y": actions[index][1].astype(float)}

            new_states_list = self.game_engine.act_and_observe(actions_buffer, replace_terminal_game=False,
                                                               max_steps=self.max_steps_evaluation)

            for index_agent, new_state_dict in enumerate(new_states_list):
                if new_state_dict["is_terminal"] == 1.0:
                    games_dict.pop(new_state_dict["identifier"])
                    games_alive.remove(new_state_dict["identifier"])
                    terminaison_steps[new_state_dict["identifier"]] += step
                    remaining_pallets[new_state_dict["identifier"]] = new_state_dict["remaining_pallets"]
                    print("Died after {} steps".format(terminaison_steps[new_state_dict["identifier"]]))

                new_state = new_state_dict["last_frame"]
                if self.evaluate_animate and new_state_dict["is_terminal"] == 0.0:
                    [x_window, y_window] = positions_windows[new_state_dict["identifier"]]
                    self.game_engine.get_game(new_state_dict["identifier"]).animate(frame=new_state,
                                                                                    x_window=x_window,
                                                                                    y_window=y_window)
                sum_rewards[new_state_dict["identifier"]] += new_state_dict["reward"]

            step += 1

            if len(games_alive) == 0:
                break

        print("Agent(s) at n_frames:{} \n  "
              "Terminated at steps: {} \n  "
              "Remaining pallets {}\n  "
              "  Average remaining pallets {} ; Std remainig pallets {}\n"
              "Total rewards: {} \n"
              "  Average rewards: {} ; Std rewards: {}".
              format(num_frames,
                     list(terminaison_steps.values()),
                     list(remaining_pallets.values()),
                     np.mean(list(remaining_pallets.values())),
                     np.std(list(remaining_pallets.values())),
                     list(sum_rewards.values()),
                     np.mean(list(sum_rewards.values())),
                     np.std(list(sum_rewards.values()))))

    def train(self, n_episodes):
        steps = {k: 0 for (k, v) in self.game_engine.games.items()}
        rewards_sum = {k: 0 for (k, v) in self.game_engine.games.items()}
        rewards_episode_end = []
        n_frames = 0
        total_n_episodes = 0
        eps_s = 1
        eps_f = 0.01
        eps_decay = 10000
        epsilon = eps_s
        epsilon_by_frame = lambda frame_idx: eps_f + (eps_s - eps_f) * np.math.exp(-1. * frame_idx / eps_decay)
        iterations = 0
        last_iter = time.time()
        while total_n_episodes < n_episodes:
            if iterations % 1000 == 0 and iterations != 0:
                print("Iteration: {} ; 1000 iterations took {:.2f} s".format(iterations, time.time() - last_iter))
                last_iter = time.time()

            states = self.game_engine.get_current_states()

            for state in states:
                if state["identifier"] not in rewards_sum:
                    rewards_sum[state["identifier"]] = 0
                    steps[state["identifier"]] = 0

            states_buffer = []
            states_identifiers = []
            for state in states:
                states_buffer.append(np.stack(state["state"]))
                states_identifiers.append(state["identifier"])

            states = states_buffer

            states = Variable(torch.from_numpy(np.stack(states)).float().unsqueeze(1)).to(device)
            epsilon = epsilon_by_frame(n_frames)
            rewards, is_terminal, identifiers = self.interact(states, states_identifiers,
                                                              steps, epsilon)  # unsqueeze to make the images one chanel (instead of 0)

            n_frames += self.batch_size

            if n_frames % (self.batch_size * self.evaluate_batch_step) == 0 and n_frames != 0:
                self.evaluate_policy(num_frames=n_frames)

            rewards_mean_buffer = []
            for i, (reward, terminal, identifier) in enumerate(zip(rewards, is_terminal, identifiers)):
                rewards_sum[identifier] += reward
                steps[identifier] += 1
                if terminal is True:
                    rewards_episode_end.append([n_frames, rewards_sum[identifier]])
                    rewards_mean_buffer.append(rewards_sum[identifier])
                    # print("End of episode {} ; Number of frames: {} ; Total reward: {} ; Steps survied: {}".format(total_n_episodes+1, n_frames, rewards_sum[identifier], steps[identifier]))
                    rewards_sum.pop(identifier)
                    steps.pop(identifier)
                    total_n_episodes += 1

                iterations += 1

if __name__ == "__main__":
    import Game
    game_engine = Game.GamesManager(state_size=64, n_frames_state=1, max_steps=2500, n_pallets=20)
    game_engine.request_new_games(n_games=1, name="test")
    agent = A2C_agent(state_size=64,
                       n_frames_state=1,
                       game_engine=game_engine,
                       discount_factor=0.99,
                       batch_size=64,
                       evaluate_batch_step=25000,
                       evaluate_n_agents=100,
                       evaluate_animate=False,
                       max_steps=2500,
                       max_steps_evaluation=200)
    agent.train(n_episodes=50000)