"""
ddqn.py — Double Deep Q-Network in PyTorch.

Riproduce fedelmente l'architettura descritta in Zaccagnino et al. (2025,
Sez. 4.2.2) e implementata in `deepq_simulation.py`:

    Dense(|S|) -> Dense(24, ReLU, He-uniform) -> Dense(12, ReLU, He-uniform)
               -> Dense(|A|, lineare)
    loss  = Huber
    optim = Adam(lr=1e-3)
    target network aggiornata ogni N step (hard update)
    epsilon-greedy con decadimento esponenziale
        eps = eps_min + (eps_max - eps_min) * exp(-decay * episodio)

Differenza tecnica rilevante: il paper usa il target
    y = r + gamma * max_a' Q_target(s', a')
che e' in realta' il target **DQN**, non DDQN. Qui viene implementato il
disaccoppiamento corretto di van Hasselt et al. (2015),
    y = r + gamma * Q_target(s', argmax_a' Q_main(s', a')),
che riduce la sovrastima dei valori Q. Il flag `double=False` ripristina il
comportamento originale, cosi' da poter riportare entrambi in un'ablation.

PyTorch invece di TensorFlow/Keras: stessa rete, installazione molto piu'
leggera e nessun conflitto con le versioni recenti di NumPy.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- #
@dataclass
class DQNConfig:
    hidden: tuple[int, ...] = (24, 12)     # come nel paper
    lr: float = 1e-3
    gamma: float = 0.618                   # discount factor del paper
    batch_size: int = 128
    replay_size: int = 50_000
    min_replay: int = 1_000
    train_every: int = 4                   # step fra due update della main net
    target_update_every: int = 100         # step fra due copie main -> target
    eps_max: float = 1.0
    eps_min: float = 0.01
    eps_decay: float = 0.01                # per episodio
    double: bool = True
    grad_clip: float = 10.0
    device: str = "cpu"


# ---------------------------------------------------------------------- #
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden=(24, 12)):
        super().__init__()
        layers, prev = [], state_dim
        for h in hidden:
            lin = nn.Linear(prev, h)
            nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU()]
            prev = h
        out = nn.Linear(prev, n_actions)   # output lineare: sono Q-value
        nn.init.kaiming_uniform_(out.weight, nonlinearity="linear")
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------- #
class ReplayMemory:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.asarray(s, dtype=np.float32),
                np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32),
                np.asarray(s2, dtype=np.float32),
                np.asarray(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------- #
class DDQNAgent:
    """Il super-agente: la Pokemon Company che impara la propria policy."""

    def __init__(self, state_dim: int, n_actions: int, cfg: DQNConfig | None = None):
        self.cfg = cfg or DQNConfig()
        self.n_actions = n_actions
        self.device = torch.device(self.cfg.device)

        self.main = QNetwork(state_dim, n_actions, self.cfg.hidden).to(self.device)
        self.target = QNetwork(state_dim, n_actions, self.cfg.hidden).to(self.device)
        self.target.load_state_dict(self.main.state_dict())
        self.target.eval()

        self.optim = torch.optim.Adam(self.main.parameters(), lr=self.cfg.lr)
        self.memory = ReplayMemory(self.cfg.replay_size)

        self.epsilon = self.cfg.eps_max
        self._steps = 0
        self.losses: list[float] = []

    # ------------------------------------------------------------------ #
    def act(self, state, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.main(s).argmax(dim=1).item())

    def q_values(self, state) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return self.main(s).cpu().numpy()[0]

    # ------------------------------------------------------------------ #
    def remember(self, s, a, r, s2, done) -> None:
        self.memory.push(s, a, r, s2, done)

    def decay_epsilon(self, episode: int) -> None:
        c = self.cfg
        self.epsilon = c.eps_min + (c.eps_max - c.eps_min) * math.exp(-c.eps_decay * episode)

    # ------------------------------------------------------------------ #
    def learn(self) -> float | None:
        c = self.cfg
        self._steps += 1
        if len(self.memory) < c.min_replay:
            return None
        if self._steps % c.train_every != 0:
            return None

        s, a, r, s2, d = self.memory.sample(c.batch_size)
        s  = torch.as_tensor(s,  device=self.device)
        a  = torch.as_tensor(a,  device=self.device)
        r  = torch.as_tensor(r,  device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d  = torch.as_tensor(d,  device=self.device)

        q = self.main(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if c.double:
                # van Hasselt: l'argmax lo sceglie la main, lo valuta la target
                best = self.main(s2).argmax(dim=1, keepdim=True)
                q_next = self.target(s2).gather(1, best).squeeze(1)
            else:
                q_next = self.target(s2).max(dim=1).values
            y = r + c.gamma * q_next * (1.0 - d)

        loss = F.huber_loss(q, y)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main.parameters(), c.grad_clip)
        self.optim.step()

        if self._steps % c.target_update_every == 0:
            self.target.load_state_dict(self.main.state_dict())

        val = float(loss.item())
        self.losses.append(val)
        return val

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({"main": self.main.state_dict(),
                    "target": self.target.state_dict(),
                    "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.main.load_state_dict(ck["main"])
        self.target.load_state_dict(ck["target"])


# ---------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
