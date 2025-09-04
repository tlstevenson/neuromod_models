'''Code to implement the model and run a single simulation.'''
# © Thomas Akam, 2023, released under the GPLv3 licence.
# Modified for Stimulus-Response Task

import os
import json
import pickle
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor as tensor
import torch.nn.functional as F
from collections import namedtuple

Episode = namedtuple('Episode', ['states', 'rewards', 'actions', 'pfc_inputs', 'pfc_states', 'pred_states', 'n_trials'])

# Parameters for Stimulus-Response Task
default_params = {
    # Simulation params.
    'n_episodes'  : 500,
    'episode_len' : 30,    # Trials per episode
    'gamma' : 0.9,          # Discount rate

    # Stimulus-Response Task params
    'cs' : 15,              # Cue size (binary elements)
    'nbcuesrange' : (4, 8),  # Range of cue counts
    'triallen' : 4,          # Steps per trial
    'nbtraintrials' : 20,    # Training trials per episode
    'nbtesttrials' : 10,     # Test trials per episode
    'rew' : 1.0,            # Reward amount

    # Model params
    'n_back': 10,            # Length of history
    'n_pfc' : 32,            # Number of PFC units
    'pfc_learning_rate' : 0.001,
    'n_str' : 16,            # Number of striatum units
    'str_learning_rate' : 0.005,
    'entropy_loss_weight' : 0.01
}

class StimulusResponseTask:
    def __init__(self, params):
        self.params = params
        self.n_actions = 2  # 0: no response, 1: response
        self.reset()
        
    def reset(self):
        """Initialize new episode with random cues"""
        # Generate unique cues
        self.nbcues = np.random.randint(*self.params['nbcuesrange'])
        self.cues = [np.random.choice([-1, 1], size=self.params['cs']) 
                     for _ in range(self.nbcues)]
        
        # Initialize trial state
        self.trial_step = 0
        self.trial_num = 0
        # State = cue1 + cue2 + [go] + 4 aux + prev-action one-hot
        self.state = np.zeros(self.params['cs'] * 2 + 1 + 4 + self.n_actions)
        return self.state.copy()
    
    def step(self, action):
        """Execute one time step"""
        reward = 0
        new_trial = False
        
        # State transition logic
        if self.trial_step == 0:  # Cue presentation
            # Select random cue pair
            cue_pair = np.random.choice(self.nbcues, 2, replace=False)
            self.cue1, self.cue2 = cue_pair
            self.correct_order = int(self.cue1 < self.cue2)
            
            # Set state: two cues + additional inputs, keep constant size by appending prev-action zeros
            self.state = np.concatenate([
                self.cues[self.cue1],
                self.cues[self.cue2],
                [0],  # No go signal
                [1.0, 0, 0, 0],  # Additional inputs
                np.zeros(self.n_actions),  # Prev action one-hot (zeros at trial start)
            ])
            
        elif self.trial_step == 1:  # Response phase
            # Set state: go signal present
            self.state[2*self.params['cs']] = 1  # Go signal
            # Clear previous action window
            self.state[-self.n_actions:] = 0  # Clear previous action
            
            # Store chosen action
            self.action_taken = action
            
        elif self.trial_step == 2:  # Reward delivery
            # Determine reward
            if (self.correct_order and action == 1) or \
               (not self.correct_order and action == 0):
                reward = self.params['rew']
            
            # Set state: reward input
            self.state[2*self.params['cs'] + 3] = reward
            
        elif self.trial_step == 3:  # End of trial
            # Reset for next trial, preserve constant length and set previous action one-hot
            self.state = np.zeros_like(self.state)
            prev_idx = len(self.state) - self.n_actions + int(self.action_taken)
            self.state[prev_idx] = 1  # Previous action one-hot
            self.trial_num += 1
            new_trial = True
        
        # Advance to next step
        self.trial_step = (self.trial_step + 1) % self.params['triallen']
        return self.state.copy(), reward, new_trial

# PFC Model
class PFC_model(nn.Module):
    def __init__(self, pm, input_size):
        super(PFC_model, self).__init__()
        self.hidden_size = pm['n_pfc']
        self.rnn = nn.GRU(input_size, pm['n_pfc'], 1, batch_first=True)
        self.state_pred = nn.Linear(pm['n_pfc'], input_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        pred = self.state_pred(hidden)
        return pred, hidden

# Striatum Model
class Str_model(nn.Module):
    def __init__(self, pm, input_size):
        super(Str_model, self).__init__()
        self.input = nn.Linear(input_size + pm['n_pfc'], pm['n_str'])
        self.actor = nn.Linear(pm['n_str'], 2)  # 2 actions
        self.critic = nn.Linear(pm['n_str'], 1)
        
    def forward(self, obs_state, pfc_state):
        y = torch.cat((obs_state, pfc_state), dim=-1)
        y = F.relu(self.input(y))
        actor = F.softmax(self.actor(y), dim=-1)
        critic = self.critic(y)
        return actor, critic

def run_simulation(save_dir=None, pm=default_params):
    # Initialize environment and models
    task = StimulusResponseTask(pm)
    input_size = len(task.state)
    
    # PFC Model
    pfc_model = PFC_model(pm, input_size)
    pfc_loss_fn = nn.MSELoss()
    pfc_optimizer = torch.optim.Adam(pfc_model.parameters(), lr=pm['pfc_learning_rate'])
    pfc_input_buffer = torch.zeros(pm['n_back'], input_size)
    
    # Striatum Model
    str_model = Str_model(pm, input_size)
    str_optimizer = torch.optim.Adam(str_model.parameters(), lr=pm['str_learning_rate'])
    
    # Training loop
    episode_buffer = []
    state = task.reset()
    
    for e in range(pm['n_episodes']):
        states, rewards, actions = [], [], []
        pfc_inputs, pfc_states, values = [], [], []
        episode_reward = 0
        
        # Initialize PFC state
        _, pfc_state = pfc_model(pfc_input_buffer.unsqueeze(0))
        
        for _ in range(pm['episode_len'] * pm['triallen']):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Choose action
            action_probs, value = str_model(state_tensor, pfc_state.detach())
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            
            # Store data
            states.append(state.copy())
            actions.append(action)
            values.append(value.item())
            pfc_inputs.append(pfc_input_buffer.clone().numpy())
            pfc_states.append(pfc_state.detach().numpy())
            
            # Take step
            next_state, reward, new_trial = task.step(action)
            rewards.append(reward)
            episode_reward += reward
            
            # Update PFC input buffer
            pfc_input_buffer = torch.cat([pfc_input_buffer[1:], 
                                         torch.FloatTensor(state).unsqueeze(0)])
            
            # Update PFC state
            _, pfc_state = pfc_model(pfc_input_buffer.unsqueeze(0))
            
            # Update state
            state = next_state
            
            # Train PFC at trial boundaries
            if new_trial and len(states) > pm['n_back']:
                # Prepare inputs and targets
                inputs = torch.FloatTensor(np.array(pfc_inputs[-pm['n_back']:-1]))
                targets = torch.FloatTensor(np.array(states[-pm['n_back']+1:]))
                
                # PFC prediction
                predictions, _ = pfc_model(inputs)
                pfc_loss = pfc_loss_fn(predictions, targets)
                
                # Update PFC
                pfc_optimizer.zero_grad()
                pfc_loss.backward()
                pfc_optimizer.step()
        
        # Store episode
        episode_buffer.append(Episode(
            np.array(states),
            np.array(rewards),
            np.array(actions),
            np.array(pfc_inputs),
            np.vstack(pfc_states),
            np.zeros(len(states)),  # Placeholder
            pm['episode_len']
        ))
        
        # Striatum training with A2C
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + pm['gamma'] * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(values)
        advantages = returns - values
        
        # Calculate losses
        action_probs, _ = str_model(
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.vstack(pfc_states))
        )
        
        # Actor loss
        log_probs = torch.log(action_probs.gather(1, torch.LongTensor(actions).unsqueeze(1)))
        actor_loss = -(log_probs.squeeze() * advantages).mean()
        
        # Critic loss
        critic_loss = advantages.pow(2).mean()
        
        # Entropy bonus
        entropy = -(action_probs * torch.log(action_probs)).sum(1).mean()
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss - pm['entropy_loss_weight'] * entropy
        
        # Update striatum
        str_optimizer.zero_grad()
        total_loss.backward()
        str_optimizer.step()
        
        print(f'Episode {e}: Reward = {episode_reward:.1f}')
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'params.json'), 'w') as f:
            json.dump(pm, f, indent=4)
        with open(os.path.join(save_dir, 'episodes.pkl'), 'wb') as f:
            pickle.dump(episode_buffer, f)
    
    return episode_buffer