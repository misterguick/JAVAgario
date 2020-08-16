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
from torchvision import models

"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
class OUNoise(object):
    def __init__(self, action_space, seed=0, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.2, decay_period=10):
        self.seed = seed
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space["shape"][0]
        self.low = action_space["low"]
        self.high = action_space["high"]
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        np.random.seed(self.seed)
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)

        return np.clip(action + ou_state/5, self.low, self.high)


class DDPG_Agent:
    class Critic(nn.Module):
        def __init__(self, state_size, action_dim, hidden_size, n_frames_state):
            super(DDPG_Agent.Critic, self).__init__()

            self.n_frames_state = n_frames_state
            self.conv2D_1 = []
            self.maxpool2D_1 = []
            self.batchnorm2d_1 = []
            self.conv2D_2 = []
            self.batchnorm2d_2 = []
            self.maxpool2D_2 = []
            self.conv2D_3 = []
            self.batchnorm2d_3 = []

            for n_th_frame_net in range(n_frames_state):
                self.conv2D_1.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=0).to(device))
                self.batchnorm2d_1.append(nn.BatchNorm2d(num_features=32).to(device))
                self.conv2D_2.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0).to(device))
                self.batchnorm2d_2.append(nn.BatchNorm2d(num_features=64).to(device))
                self.conv2D_3.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0).to(device))
                self.batchnorm2d_3.append(nn.BatchNorm2d(num_features=64).to(device))

                dense_input_size = 4 * 4 * 64

            self.dense_all_1 = nn.Linear(dense_input_size * n_frames_state + action_dim, hidden_size * n_frames_state).to(device)
            self.batchnormd_dense_1 = nn.BatchNorm1d(num_features=hidden_size * n_frames_state).to(device)
            self.dense_all_2  = nn.Linear(hidden_size * n_frames_state, 1).to(device)


        def forward(self, state, action):
            """
            Params state and actions are torch tensors
            """
            unstacked_state = state.unbind(2)  # unstack into list of size n_frames_state
            outs = []
            for f in range(self.n_frames_state):
                x = self.conv2D_1[f](unstacked_state[f])
                x = F.relu(x)
                x = self.batchnorm2d_1[f](x)
                x = self.conv2D_2[f](x)
                x = F.relu(x)
                x = self.batchnorm2d_2[f](x)
                x = self.conv2D_3[f](x)
                x = F.relu(x)
                x = self.batchnorm2d_3[f](x)
                x = torch.flatten(x, 1)

                outs.append(x)

            x = torch.cat(outs, 1)
            y = torch.flatten(action, 1)
            x = torch.cat([x, y], 1)

            x = self.dense_all_1(x)
            x = F.relu(x)
            x = self.batchnormd_dense_1(x)
            x = self.dense_all_2(x)

            return x



    class Actor(nn.Module):
        def __init__(self, state_size, action_dim, hidden_size, n_frames_state):
            super(DDPG_Agent.Actor, self).__init__()

            self.n_frames_state = n_frames_state
            self.conv2D_1 = []
            self.maxpool2D_1 = []
            self.batchnorm2d_1 = []
            self.conv2D_2 = []
            self.batchnorm2d_2 = []
            self.maxpool2D_2 = []
            self.conv2D_3 = []
            self.batchnorm2d_3 = []
            for n_th_frame_net in range(n_frames_state):
                self.conv2D_1.append(
                    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=0).to(device))
                self.batchnorm2d_1.append(nn.BatchNorm2d(num_features=32).to(device))
                self.conv2D_2.append(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0).to(device))
                self.batchnorm2d_2.append(nn.BatchNorm2d(num_features=64).to(device))
                self.conv2D_3.append(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0).to(device))
                self.batchnorm2d_3.append(nn.BatchNorm2d(num_features=64).to(device))

                dense_input_size = 4 * 4 * 64



            self.dense_final_1 = nn.Linear(dense_input_size*n_frames_state, hidden_size * n_frames_state).to(device)
            self.batchnormd_dense_1 = nn.BatchNorm1d(num_features=hidden_size * n_frames_state).to(device)
            self.dense_final_2 = nn.Linear(hidden_size * n_frames_state, action_dim).to(device)
            self.out = nn.Tanh().to(device)

        def forward(self, state):
            unstacked_state = state.unbind(2)  # unstack into list of size n_frames_state

            outs = []
            for f in range(self.n_frames_state):
                x = self.conv2D_1[f](unstacked_state[f])
                x = F.relu(x)
                x = self.batchnorm2d_1[f](x)
                x = self.conv2D_2[f](x)
                x = F.relu(x)
                x = self.batchnorm2d_2[f](x)
                x = self.conv2D_3[f](x)
                x = F.relu(x)
                x = self.batchnorm2d_3[f](x)
                x = torch.flatten(x, 1)

                outs.append(x)

            y = torch.cat(outs, 1)
            y = self.dense_final_1(y)
            y = F.relu(y)
            y = self.batchnormd_dense_1(y)
            y = self.dense_final_2(y)
            y = self.out(y)

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
                 discount_factor = 0.99,
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

        hidden_size = 512
        self.actor = DDPG_Agent.Actor(state_size, action_dim=2, hidden_size=hidden_size, n_frames_state=n_frames_state).to(device)
        self.actor_target = DDPG_Agent.Actor(state_size, action_dim=2, hidden_size=hidden_size, n_frames_state=n_frames_state).to(device)
        self.critic = DDPG_Agent.Critic(state_size, action_dim=2, hidden_size=hidden_size, n_frames_state=n_frames_state).to(device)
        self.critic_target = DDPG_Agent.Critic(state_size, action_dim=2, hidden_size=hidden_size, n_frames_state=n_frames_state).to(device)


        self.learning_rate_critric = 0.0005
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critric)
        self.critic_loss_function = nn.MSELoss()

        self.learning_rate_actor = 0.00001
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)

        # We initialize the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.replay_buffer = DDPG_Agent.ExperienceReplay(max_buffer_size=65000)

        self.update_targets = "soft" #"soft"
        self.update_targets_steps = 5000
        self.update_targets_time_to_live = self.update_targets_steps

    def update(self, states, actions, rewards, next_states, is_terminal, indexes=None):
        Q_current = self.critic.forward(states, actions)
        actions_in_next_state = self.actor_target.forward(next_states).unsqueeze(1)
        Q_new_obs = self.critic_target.forward(next_states, actions_in_next_state.detach()) # detach() to break back prop
        Q_target = (rewards + (1 - is_terminal) * self.discount_factor * Q_new_obs)

        Q_current_final = Q_current
        Q_target_final = Q_target


        policy_actions = self.actor.forward(states).unsqueeze(1)
        actor_loss = -self.critic.forward(states, policy_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = self.critic_loss_function(Q_current_final, Q_target_final)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.update_targets == "soft":
            tau = 0.001
            # update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        else:
            if self.update_targets_time_to_live == 0:
                self.update_targets_time_to_live = self.update_targets_steps
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data)
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data)
            else:
                self.update_targets_time_to_live -= 1

    def get_actions(self, states):

        with torch.no_grad():
            self.actor.eval()
            actions = self.actor.forward(states).detach().cpu().numpy()
            self.actor.train()

        return actions

    def interact(self, states, states_identifiers, steps, noises):
        # get actions from policy network
        actions = self.get_actions(states)
        # reformat actions for the game engine
        actions_buffer = {}
        for i, action in enumerate(actions):  # if several games in //
            action = noises[states_identifiers[i]].get_action(action, steps[states_identifiers[i]])
            actions_buffer[states_identifiers[i]] = {"input_x": action[0].item(), "input_y": action[1].item()}

        # act on the envs. and get the results
        tuples = self.game_engine.act_and_observe(actions_buffer)

        current_identifiers = []
        current_rewards = []
        current_is_terminal = []

        # add to the replay memory AND REFORMAT to pytorch
        for tuple in tuples:  # if several games in //.
            tuple["state"] = torch.from_numpy(np.stack(tuple["state"])).float()
            tuple["action"] = torch.from_numpy(np.array([tuple["action"]["input_x"], tuple["action"]["input_y"]])).float()
            tuple["reward"] = torch.tensor(tuple["reward"]).float()
            tuple["new_state"] = torch.from_numpy(np.stack(tuple["new_state"])).float()
            tuple["is_terminal"] = torch.from_numpy(tuple["is_terminal"]).float()
            current_identifiers.append(tuple["identifier"])
            current_rewards.append(tuple["reward"])
            current_is_terminal.append(tuple["is_terminal"])

            self.replay_buffer.push(tuple)

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

        states = torch.stack(states).to(device).unsqueeze(1)
        actions = torch.stack(actions).to(device).unsqueeze(1)
        rewards = torch.stack(rewards).to(device).unsqueeze(1)
        new_states = torch.stack(new_states).to(device).unsqueeze(1)
        is_terminal = torch.stack(is_terminal).to(device).unsqueeze(1)

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

        drawing_size = self.state_size
        if self.evaluate_animate:
            n_columns_windows = 7
            positions_windows = {}
            for index_agent, game in enumerate(games_dict.values()):
                x_window = index_agent % n_columns_windows
                y_window = int(index_agent / n_columns_windows)
                x_window = (x_window * (drawing_size + 50)) + 50
                y_window = (y_window * (drawing_size + 50)) + 50
                positions_windows[game.get_identifier()] = [x_window, y_window]

        while True:
            print(step)
            if step > self.max_steps_evaluation:
                for game in games_dict.values():
                    #self.game_engine.remove_game(game.get_identifier())
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

            new_states_list = self.game_engine.act_and_observe(actions_buffer, replace_terminal_game=False, max_steps=self.max_steps_evaluation)

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
        env_action_space = {"low": -1, "high": 1, "shape": (2, 1)}
        steps = {k: 0 for (k, v) in self.game_engine.games.items()}
        rewards_sum = {k: 0 for (k, v) in self.game_engine.games.items()}
        noises = {k:OUNoise(env_action_space, seed=int(k)) for (k, v) in self.game_engine.games.items()}
        rewards_episode_end = []
        n_frames = 0
        total_n_episodes = 0
        iterations = 0
        last_iteration = time.time()
        while total_n_episodes < n_episodes:
            if iterations % 1000 == 0 and iterations != 0:
                print("Iteration: {} ; 1000 iterations took {:.2f} s".format(iterations, time.time() - last_iteration))
                last_iteration = time.time()

            states = self.game_engine.get_current_states()

            for state in states:
                if state["identifier"] not in rewards_sum:
                    rewards_sum[state["identifier"]] = 0
                    steps[state["identifier"]] = 0
                    noises[state["identifier"]] = OUNoise(env_action_space, seed=int(state["identifier"]))

            """for state in states:
                self.game_engine.games[state["identifier"]].animate(frame=state["state"])"""

            states_buffer = []
            states_identifiers = []
            for state in states:
                states_buffer.append(state["state"])
                states_identifiers.append(state["identifier"])

            states = states_buffer

            states = Variable(torch.from_numpy(np.stack(states)).float().unsqueeze(1)).to(device)
            rewards, is_terminal, identifiers = self.interact(states, states_identifiers,
                                                              steps, noises)  # unsqueeze to make the images one chanel (instead of 0)

            n_frames += self.batch_size

            #print("Num. of frames drawn from the buffer {}".format(n_frames))
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
                    noises.pop(identifier)
                    total_n_episodes += 1

            iterations += 1

if __name__ == "__main__":
    import Game
    game_engine = Game.GamesManager(state_size=64, n_frames_state=1, max_steps=2500, n_pallets=20)
    game_engine.request_new_games(n_games=1, name="test")
    agent = DDPG_Agent(state_size=64,
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