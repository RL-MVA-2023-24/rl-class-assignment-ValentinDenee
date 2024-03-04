import random
import os
import numpy as np
import torch
import torch.nn as nn
from env_hiv import HIVPatient
from functools import partial
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.index = 0

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self):
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_period = 1000
        self.epsilon_delay_decay = 10
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_period
        self.epsilon = self.epsilon_max
        self.model = self.build_model()  # DQN model
        self.target_model = self.build_model()  #target DQN model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(capacity=100000)

    def build_model(self, state_dim=6, n_actions=4, nb_neurons=256):    
        model = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_actions)
        )
        return model

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        if use_random and np.random.rand() < self.epsilon:  
            return np.random.choice(4)  #action aléatoire / exploration
        else:
            return self.greedy_action(observation)  # Utilisation de self.greedy_action

    def greedy_action(self, observation: np.ndarray) -> int:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            Q_values = self.model(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))
            return torch.argmax(Q_values).item()

   #def load(self, file_path="agent_params.pth"):
   #     if os.path.isfile(file_path):
   #         checkpoint = torch.load(file_path)
   #         self.model.load_state_dict(checkpoint['model_state_dict'])
   #         self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
   #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   #         self.epsilon = checkpoint['epsilon']
   #         self.memory.data = checkpoint['replay_buffer']
   #         print("Agent parameters loaded successfully.")
   #     else:
   #         print("No saved parameters found.")


   # def save(self, file_path="agent_params.pth"):
   #     checkpoint = {
   #         'model_state_dict': self.model.state_dict(),
   #         'target_model_state_dict': self.target_model.state_dict(),
   #         'optimizer_state_dict': self.optimizer.state_dict(),
   #         'epsilon': self.epsilon,
   #         'replay_buffer': self.memory.data
   #     }
   #     torch.save(checkpoint, file_path)
   #     print("Agent parameters saved successfully.")

    def load(self, file_path=os.path.join(os.path.dirname(__file__), "agent_params.pth")):
        checkpoint = torch.load(file_path,map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory.data = checkpoint['replay_buffer']
        print("Agent parameters loaded successfully.")           
    
    def save(self, file_path=os.path.join(os.path.dirname(__file__), "agent_params.pth")):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'replay_buffer': self.memory.data
        }
        torch.save(checkpoint, file_path)
        print("Agent parameters saved successfully.")

    def gradient_step(self):
        if len(self.memory) > 1000:
            batch = self.memory.sample(1000)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.Tensor(states)
            #states = torch.tensor(np.array(states), dtype=torch.float32)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            actions = torch.LongTensor(actions)
            dones = torch.Tensor(dones)
            current_Q = self.model(states).gather(1, actions.unsqueeze(1).to(torch.long)).squeeze()
            max_next_Q = self.target_model(next_states).detach().max(1)[0]
            target_Q = rewards + (1 - dones) * 0.99 * max_next_Q
            loss = torch.nn.functional.smooth_l1_loss(current_Q, target_Q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_returns = []
        incr1 = 0
        for episode in range(max_episode):
            episode_cum_reward = 0
            state, _ = env.reset()
            incr2 = 0
            incr1 += 1
            for _ in range(200):  #maximum episode length
                print(incr2)
                print(incr1)
                incr2 += 1
                if len(self.memory) > 1000:
                    self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)
                action = self.act(state,use_random=True)
                next_state, reward, done, _, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward
                self.gradient_step()

                if done or _ == 199:  # stop if done or if reached max episode length
                    episode_returns.append(episode_cum_reward)
                    print(f"Episode {episode + 1}, Epsilon: {self.epsilon}, Episode Return: {episode_cum_reward}")
                    break
                else:
                    state = next_state
        self.save()

if __name__ == "__main__":
    env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
    agent = ProjectAgent()
    agent.train(env, max_episode=200)
    