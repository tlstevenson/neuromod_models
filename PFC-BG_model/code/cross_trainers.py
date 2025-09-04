"""
Cross-task adapters and training loops.
- Run Blanco-Pozo model on Stimulus-Response task (Miconi-style)
- Run Miconi-style model on Two-step task (Blanco-Pozo environment)
"""

from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F

# Import existing modules
import model as bp
import modelWithMiconiData as mic
import two_step_task as ts


# ----------------------
# Adapters
# ----------------------

class SRTaskAdapterForBP:
    """Wrap StimulusResponseTask (Miconi) so it looks like a discrete-state task
    to the Blanco-Pozo pipeline (discrete n_states, integer state id, 2 actions).
    """

    # Discrete state ids exposed to BP
    CUE = 0
    GO = 1
    REWARD0 = 2
    REWARD1 = 3
    ITI = 4

    def __init__(self, pm: dict):
        self._sr = mic.StimulusResponseTask(pm)
        self.n_actions = 2
        self.n_states = 5
        self.trial_n = 0
        self.A_good = True  # Placeholder for compatibility
        self._pm = pm
        self._last_reward = 0.0

    def get_obs_dim(self) -> int:
        """Return dimensionality of the SR raw observation vector."""
        # In StimulusResponseTask: state = cue1 (cs) + cue2 (cs) + [go] + 4 aux + prev-action one-hot (2)
        return (self._pm['cs'] * 2) + 1 + 4 + self.n_actions

    def get_obs(self) -> np.ndarray:
        """Return current SR raw observation vector (copy)."""
        # The SR env maintains the current state vector.
        return self._sr.state.copy()

    def _disc_state_from_sr(self, sr_state: np.ndarray, reward: float) -> int:
        cs = self._pm['cs']
        go = sr_state[2 * cs]  # Go signal index
        if reward > 0:
            return self.REWARD1
        if go == 1:
            return self.GO
        # End of trial/reset phase typically has zeros and prev action one-hot at the end
        if self._sr.trial_step == 3:
            return self.ITI
        return self.CUE

    def reset(self) -> int:
        s = self._sr.reset()
        self._last_reward = 0.0
        return self._disc_state_from_sr(s, 0.0)

    def step(self, action01: int) -> Tuple[int, float]:
        s_vec, r, new_trial = self._sr.step(action01)
        if new_trial:
            self.trial_n += 1
        self._last_reward = r
        s_id = self._disc_state_from_sr(s_vec, r)
        return s_id, float(r)


class TwoStepAdapterForMiconi:
    """Wrap Two_step so it looks like the StimulusResponseTask API for Miconi.
    Provides:
      - n_actions = 2
      - state: fixed-length vector encoding of discrete state + simple context
      - step(binary_action): internally maps to valid env action given current state
      - returns (state_vector, reward, new_trial)
    """

    def __init__(self, pm: dict):
        self.env = ts.Two_step(good_prob=pm.get('good_prob', 0.8), block_len=pm.get('block_len', [20, 40]))
        self.n_actions = 2
        # Encoding dims: one-hot over env states (6) + [bias, time, prev_reward] + prev binary action one-hot (2)
        self._state_dim = self.env.n_states + 3 + 2
        self._pm = pm
        self._prev_bin_action = 0
        self._prev_reward = 0.0
        self._step_in_trial = 0
        self._last_env_state = None
        self.state = self._encode_state(self.env.reset())

    def _encode_state(self, env_state: int) -> np.ndarray:
        vec = np.zeros(self._state_dim, dtype=np.float32)
        # One-hot over env discrete state
        vec[env_state] = 1.0
        # Bias, time-within-trial (0..1 over a nominal 4-step trial), prev reward
        base = self.env.n_states
        vec[base + 0] = 1.0
        vec[base + 1] = min(self._step_in_trial / 3.0, 1.0)
        vec[base + 2] = float(self._prev_reward)
        # Previous binary action one-hot
        vec[base + 3 + self._prev_bin_action] = 1.0
        return vec

    def reset(self) -> np.ndarray:
        self._prev_bin_action = 0
        self._prev_reward = 0.0
        self._step_in_trial = 0
        self._last_env_state = None
        s = self.env.reset()
        self.state = self._encode_state(s)
        return self.state.copy()

    def _map_action(self, bin_action: int, env_state: int) -> int:
        # Map 0/1 to a valid env action depending on current state
        if env_state in (ts.initiate, ts.reward_A, ts.reward_B):
            return ts.initiate
        if env_state == ts.choice:
            return ts.choose_A if bin_action == 0 else ts.choose_B
        if env_state == ts.sec_step_A:
            return ts.sec_step_A
        if env_state == ts.sec_step_B:
            return ts.sec_step_B
        return ts.initiate

    def step(self, bin_action: int) -> Tuple[np.ndarray, float, bool]:
        # Determine env action
        current_env_state = getattr(self.env, 'state', ts.initiate)
        env_action = self._map_action(bin_action, current_env_state)
        next_state, reward = self.env.step(env_action)

        # Determine new-trial boundary: when we return to initiate from a non-initiate state
        new_trial = (next_state == ts.initiate) and (self._last_env_state is not None) and (self._last_env_state != ts.initiate)

        # Update trackers
        self._prev_bin_action = int(bin_action)
        self._prev_reward = float(reward)
        self._step_in_trial = 0 if new_trial else (self._step_in_trial + 1)
        self._last_env_state = next_state

        # Encode state vector
        self.state = self._encode_state(next_state)
        return self.state.copy(), float(reward), bool(new_trial)


# ----------------------
# Cross-training loops
# ----------------------

def run_blanco_on_sr(save_dir: Optional[str], pm: dict):
    """Train Blanco-Pozo PFC+Str on the Stimulus-Response task via adapter."""
    np.random.seed(int.from_bytes(os.urandom(4), 'little'))

    task = SRTaskAdapterForBP(pm)

    # PFC input size: concatenate [disc one-hot | SR raw obs | prev action one-hot]
    if pm.get('pred_rewarded_only', False):
        input_size = task.n_states
        pfc_input_buffer = torch.zeros([pm['n_back'], input_size])
    else:
        sr_dim = task.get_obs_dim()
        input_size = task.n_states + sr_dim + task.n_actions
        pfc_input_buffer = torch.zeros([pm['n_back'], input_size])

    pfc_model = bp.PFC_model(pm, input_size, task)
    pfc_loss_fn = torch.nn.MSELoss()
    pfc_optimizer = torch.optim.Adam(pfc_model.parameters(), lr=pm['pfc_learning_rate'])

    def update_pfc_input(a: int, s_disc: int, r: float, sr_obs: Optional[np.ndarray] = None):
        # Shift buffer
        pfc_input_buffer[:-1, :] = torch.detach(pfc_input_buffer[1:, :]).clone()
        pfc_input_buffer[-1, :] = 0
        if pm.get('pred_rewarded_only', False):
            # As in original BP: only expose discrete state one-hot gated by reward
            pfc_input_buffer[-1, s_disc] = r
        else:
            # Compose [disc one-hot | SR raw obs | prev action one-hot]
            # Discrete state one-hot
            pfc_input_buffer[-1, s_disc] = 1.0
            # SR raw observation block
            assert sr_obs is not None, "sr_obs must be provided when pred_rewarded_only is False"
            disc_end = task.n_states
            raw_end = disc_end + task.get_obs_dim()
            pfc_input_buffer[-1, disc_end:raw_end] = torch.from_numpy(sr_obs).float()
            # Previous binary action one-hot appended
            pfc_input_buffer[-1, raw_end + a] = 1.0

    def get_masked(inputs: List[np.ndarray]) -> torch.Tensor:
        x = np.array(inputs)
        # Mask the most recent discrete state one-hot so the PFC must predict it
        x[:, -1, :task.n_states] = 0
        # Also mask the SR raw obs and prev action block of the most recent time step
        if not pm.get('pred_rewarded_only', False):
            disc_end = task.n_states
            raw_end = disc_end + task.get_obs_dim()
            x[:, -1, disc_end:raw_end + task.n_actions] = 0
        return torch.tensor(x, dtype=torch.float32)

    str_model = bp.Str_model(pm, task)
    str_optimizer = torch.optim.Adam(str_model.parameters(), lr=pm['str_learning_rate'])

    # Environment loop
    s = task.reset()
    r = 0.0
    _, pfc_s = pfc_model(pfc_input_buffer[None, :, :])

    episode_buffer: List[bp.Episode] = []

    for e in range(pm['n_episodes']):
        step_n = 0
        start_trial = task.trial_n
        states: List[int] = []
        rewards: List[float] = []
        actions: List[int] = []
        pfc_inputs: List[np.ndarray] = []
        pfc_states: List[np.ndarray] = []
        values: List[float] = []
        task_rew_states: List[bool] = []

        while True:
            step_n += 1
            # Choose action
            action_probs, V = str_model(F.one_hot(torch.tensor(s), task.n_states)[None, :], torch.detach(pfc_s).clone())
            a = int(np.random.choice(task.n_actions, p=np.squeeze(torch.detach(action_probs).numpy())))

            # Store
            states.append(int(s))
            rewards.append(float(r))
            actions.append(a)
            pfc_inputs.append(torch.detach(pfc_input_buffer).clone().numpy())
            pfc_states.append(torch.detach(pfc_s).numpy())
            values.append(float(torch.detach(V).numpy()))
            task_rew_states.append(True)

            # Step env
            s, r = task.step(a)

            # PFC update/input using current SR observation
            sr_obs = task.get_obs()
            update_pfc_input(a, s, r, sr_obs)
            _, pfc_s = pfc_model(pfc_input_buffer[None, :, :])

            n_trials = task.trial_n - start_trial
            if n_trials == pm['episode_len'] or (step_n >= pm.get('max_step_per_episode', 10_000) and s == SRTaskAdapterForBP.CUE):
                break

        # Store episode
        predictions, _ = pfc_model(get_masked(pfc_inputs))
        pred_states = np.argmax(torch.detach(predictions).numpy(), 1)
        episode_buffer.append(bp.Episode(np.array(states), np.array(rewards), np.array(actions), np.array(pfc_inputs),
                                         np.vstack(pfc_states), np.array(pred_states), np.array(task_rew_states), n_trials))

        # A2C update (same as bp)
        returns = np.zeros([len(rewards), 1], dtype='float32')
        returns[-1] = torch.detach(V).numpy()
        for i in range(1, len(returns)):
            returns[-i-1] = rewards[-i] + pm['gamma'] * returns[-i]
        advantages = torch.from_numpy((returns - np.vstack(values)).squeeze())

        action_probs_g, values_g = str_model(F.one_hot(torch.tensor(states), task.n_states),
                                             torch.from_numpy(np.vstack(pfc_states)))
        critic_loss = F.mse_loss(values_g, torch.from_numpy(returns), reduction='sum')
        chosen_probs = torch.gather(action_probs_g, 1, torch.tensor(actions).unsqueeze(1))
        log_chosen_probs = torch.log(chosen_probs.squeeze(1) + 1e-8)
        entropy = -torch.sum(action_probs_g * torch.log(action_probs_g + 1e-8), 1)
        actor_loss = torch.sum(-log_chosen_probs * advantages - entropy * pm['entropy_loss_weight'])
        policy_loss = actor_loss + critic_loss

        str_optimizer.zero_grad()
        policy_loss.backward()
        str_optimizer.step()

        # Train PFC
        if pm.get('pred_rewarded_only', False):
            x = torch.tensor(np.array(pfc_inputs[:-1]), dtype=torch.float32)
            y = F.one_hot(torch.tensor(states[1:]), task.n_states).float() * torch.tensor(np.array(rewards[1:]))[:, None]
            batchloader = [(x, y)]
        else:
            x = get_masked(pfc_inputs)
            y = F.one_hot(torch.tensor(states), task.n_states).float()
            batchloader = [(x, y)]

        for w, z in batchloader:
            inputs = w.reshape(-1, pm['n_back'], input_size)
            outputs, __ = pfc_model(inputs)
            tl = pfc_loss_fn(outputs, z)
            pfc_optimizer.zero_grad()
            tl.backward()
            pfc_optimizer.step()

        print(f"[BP on SR] Episode: {e} Trials: {n_trials} Rew/tr: {np.sum(rewards)/max(n_trials,1):.2f} PFC loss: {tl.item():.3f}")

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'params.json'), 'w') as fp:
            json.dump(pm, fp, indent=4)
        with open(os.path.join(save_dir, 'episodes.pkl'), 'wb') as fe:
            pickle.dump(episode_buffer, fe)



def run_miconi_on_two_step(save_dir: Optional[str], pm: dict):
    """Train Miconi-style PFC+Str on the Two-step task via adapter."""
    task = TwoStepAdapterForMiconi(pm)
    input_size = task.state.shape[0]

    # Models
    pfc_model = mic.PFC_model(pm, input_size)
    pfc_loss_fn = torch.nn.MSELoss()
    pfc_optimizer = torch.optim.Adam(pfc_model.parameters(), lr=pm['pfc_learning_rate'])
    pfc_input_buffer = torch.zeros(pm['n_back'], input_size)

    str_model = mic.Str_model(pm, input_size)  # 2-action head, adapter maps to env actions
    str_optimizer = torch.optim.Adam(str_model.parameters(), lr=pm['str_learning_rate'])

    # Training loop
    episode_buffer: List[mic.Episode] = []
    state = task.reset()

    for e in range(pm['n_episodes']):
        states: List[np.ndarray] = []
        rewards: List[float] = []
        actions: List[int] = []
        pfc_inputs: List[np.ndarray] = []
        pfc_states: List[np.ndarray] = []
        values: List[float] = []
        episode_reward = 0.0

        _, pfc_state = pfc_model(pfc_input_buffer.unsqueeze(0))

        # Conservative upper bound for steps
        max_steps = pm.get('episode_len', 100) * 4
        for _ in range(max_steps):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs, value = str_model(state_tensor, pfc_state.detach())
            dist = torch.distributions.Categorical(action_probs)
            action_bin = int(dist.sample().item())

            # Store
            states.append(state.copy())
            actions.append(action_bin)
            values.append(float(value.item()))
            pfc_inputs.append(pfc_input_buffer.clone().numpy())
            pfc_states.append(pfc_state.detach().numpy())

            # Step
            next_state, reward, new_trial = task.step(action_bin)
            rewards.append(float(reward))
            episode_reward += float(reward)

            # Update buffers/state
            pfc_input_buffer = torch.cat([pfc_input_buffer[1:], torch.from_numpy(state).float().unsqueeze(0)])
            _, pfc_state = pfc_model(pfc_input_buffer.unsqueeze(0))
            state = next_state

            # Train PFC at boundaries
            if new_trial and len(states) > pm['n_back']:
                inputs = torch.from_numpy(np.array(pfc_inputs[-pm['n_back']:-1])).float()  # (K, n_back, input_size)
                targets = torch.from_numpy(np.array(states[-pm['n_back']+1:])).float()    # (K, input_size)
                predictions, _ = pfc_model(inputs)
                pfc_loss = pfc_loss_fn(predictions, targets)
                pfc_optimizer.zero_grad()
                pfc_loss.backward()
                pfc_optimizer.step()

        # Save episode
        episode_buffer.append(mic.Episode(
            np.array(states), np.array(rewards), np.array(actions), np.array(pfc_inputs),
            np.vstack(pfc_states), np.zeros(len(states)), pm.get('episode_len', 100)
        ))

        # A2C on striatum
        returns: List[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = r + pm['gamma'] * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        advantages = returns_t - values_t

        action_probs_all, _ = str_model(torch.from_numpy(np.array(states)).float(), torch.from_numpy(np.vstack(pfc_states)).float())
        log_probs = torch.log(action_probs_all.gather(1, torch.tensor(actions).unsqueeze(1)) + 1e-8).squeeze(1)
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        entropy = -(action_probs_all * torch.log(action_probs_all + 1e-8)).sum(1).mean()
        total_loss = actor_loss + 0.5 * critic_loss - pm['entropy_loss_weight'] * entropy

        str_optimizer.zero_grad()
        total_loss.backward()
        str_optimizer.step()

        print(f"[Miconi on Two-step] Episode {e}: Reward = {episode_reward:.1f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'params.json'), 'w') as f:
            json.dump(pm, f, indent=4)
        with open(os.path.join(save_dir, 'episodes.pkl'), 'wb') as f:
            pickle.dump(episode_buffer, f)
