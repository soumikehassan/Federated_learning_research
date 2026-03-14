"""
utils/rl_client_selector.py
DQN-based Reinforcement Learning client selector for federated learning.

State  : [loss, accuracy, data_size_norm] x num_clients  (flat vector)
Action : which clients to select this round
Reward : accuracy improvement + diversity bonus - staleness penalty
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import logging

logger = logging.getLogger(__name__)

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


class RLClientSelector:
    """
    DQN agent that decides which clients participate each federated round.
    """
    def __init__(
        self,
        num_clients=3, features_per_client=3,
        hidden_dim=128, lr=3e-4, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
        buffer_size=1000, batch_size=32,
        min_clients=2, device="cpu",
    ):
        self.num_clients         = num_clients
        self.features_per_client = features_per_client
        self.state_dim           = num_clients * features_per_client
        self.gamma               = gamma
        self.epsilon             = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay       = epsilon_decay
        self.batch_size          = batch_size
        self.min_clients         = min_clients
        self.device              = torch.device(device)

        self.q_net      = DQNNetwork(self.state_dim, num_clients, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, num_clients, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer      = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer         = ReplayBuffer(buffer_size)
        self._update_count  = 0
        self._last_selected = {i: -1 for i in range(num_clients)}
        self.history        = []

    # ── State ──────────────────────────────────────────────────────────────────
    def build_state(self, client_stats: dict) -> torch.Tensor:
        """
        client_stats = {
            "client_0": {"loss": float, "accuracy": float, "data_size": int},
            ...
        }
        """
        parts = []
        max_ds = max(v["data_size"] for v in client_stats.values()) + 1e-8
        for i in range(self.num_clients):
            s = client_stats.get(f"client_{i}", {"loss": 1.0, "accuracy": 0.0, "data_size": 1})
            parts += [
                min(s["loss"] / 5.0, 1.0),
                min(s["accuracy"], 1.0),
                s["data_size"] / max_ds,
            ]
        return torch.FloatTensor(parts).to(self.device)

    # ── Action ─────────────────────────────────────────────────────────────────
    def select_clients(self, client_stats: dict, current_round: int) -> list:
        state = self.build_state(client_stats)
        if random.random() < self.epsilon:
            selected = sorted(random.sample(range(self.num_clients),
                                            random.randint(self.min_clients, self.num_clients)))
        else:
            with torch.no_grad():
                scores = torch.sigmoid(self.q_net(state.unsqueeze(0)).squeeze(0)).cpu().numpy()
            selected = [i for i, s in enumerate(scores) if s >= 0.5]
            if len(selected) < self.min_clients:
                rest  = [i for i in range(self.num_clients) if i not in selected]
                selected += random.sample(rest, self.min_clients - len(selected))
            selected = sorted(selected)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        for c in selected:
            self._last_selected[c] = current_round
        self.history.append(selected)
        logger.info(f"Round {current_round}: RL selected clients {selected} (ε={self.epsilon:.3f})")
        return selected

    # ── Reward ─────────────────────────────────────────────────────────────────
    def compute_reward(self, prev_acc, curr_acc, selected, current_round,
                       client_accs=None, client_domains=None) -> float:
        """
        Multi-objective reward:
        1. Accuracy Improvement (Fundamental goal)
        2. Selection Diversity (Anti-drift / Fairness)
        3. Staleness Penalty (Ensures all clients eventually contribute)
        4. Task Fairness (Ensures no domain is left behind - Experiment 2)
        """
        # 1. Improvement gain
        acc_reward = (curr_acc - prev_acc) * 10.0

        # 2. Diversity bonus
        # We reward selecting clients that haven't been picked in a while
        diversity_reward = 0.0
        for c in selected:
            gap = current_round - self._last_selected[c] - 1
            if gap > 2:
                diversity_reward += 0.1 * min(gap, 10)  # Capped bonus for long waits
        
        # 3. Staleness penalty for those NOT selected
        staleness_penalty = 0.0
        for c in range(self.num_clients):
            if c not in selected:
                gap = current_round - self._last_selected[c] - 1
                if gap > 5:
                    staleness_penalty += 0.05 * gap

        # 4. Task Fairness (if info available)
        fairness_reward = 0.0
        if client_accs is not None and client_domains is not None:
             fairness_reward = self._compute_task_fairness_reward(
                 client_accs, client_domains, selected
             )

        total_reward = acc_reward + diversity_reward - staleness_penalty + fairness_reward
        
        logger.debug(f"Reward Breakdown: Acc={acc_reward:.3f}, Div={diversity_reward:.3f}, "
                     f"Stale=-{staleness_penalty:.3f}, Fair={fairness_reward:.3f}")
        return float(total_reward)

    def _compute_task_fairness_reward(self, client_accs, client_domains, selected):
        """
        Compute fairness reward based on performance variance across domains.

        Goal: Ensure no single domain (task) is left behind.
        Higher reward when selected clients help balance domain performance.
        """
        # Group accuracies by domain
        domain_accs = {}
        for client_idx, domain_id in client_domains.items():
            if domain_id not in domain_accs:
                domain_accs[domain_id] = []
            if client_idx in client_accs:
                domain_accs[domain_id].append(client_accs[client_idx])

        if len(domain_accs) < 2:
            return 0.0

        # Compute mean accuracy per domain
        domain_means = {d: np.mean(accs) for d, accs in domain_accs.items() if accs}

        if len(domain_means) < 2:
            return 0.0

        # Fairness: reward low variance across domains
        mean_accs = list(domain_means.values())
        domain_variance = np.var(mean_accs)

        # Reward when variance is low (max reward when variance = 0)
        # Penalty when variance is high (one domain struggling)
        fairness_reward = 0.5 * (1.0 - min(domain_variance * 10, 1.0))

        # Bonus for selecting clients from underperforming domains
        min_domain = min(domain_means, key=domain_means.get)
        selected_domains = [client_domains.get(c, -1) for c in selected]
        if min_domain in selected_domains:
            fairness_reward += 0.2  # Bonus for helping the weakest domain

        return fairness_reward

    # ── Learning ───────────────────────────────────────────────────────────────
    def store_transition(self, state, action, reward, next_state, done):
        a_vec = torch.zeros(self.num_clients)
        for c in action:
            a_vec[c] = 1.0
        self.buffer.push(state.cpu(), a_vec, reward, next_state.cpu(), done)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch       = self.buffer.sample(self.batch_size)
        states      = torch.stack([t.state      for t in batch]).to(self.device)
        actions     = torch.stack([t.action     for t in batch]).to(self.device)
        rewards     = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones       = torch.FloatTensor([float(t.done) for t in batch]).to(self.device)

        q_cur    = (self.q_net(states) * actions).sum(1) / (actions.sum(1) + 1e-8)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
            q_tgt  = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_cur, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % 10 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()

    def get_selection_stats(self) -> dict:
        if not self.history:
            return {}
        counts = {i: 0 for i in range(self.num_clients)}
        for sel in self.history:
            for c in sel:
                counts[c] += 1
        n = len(self.history)
        return {
            f"client_{c}": {"times_selected": counts[c], "selection_rate": counts[c] / n}
            for c in range(self.num_clients)
        }

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ck["q_net"])
        self.target_net.load_state_dict(ck["target_net"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.epsilon = ck["epsilon"]
