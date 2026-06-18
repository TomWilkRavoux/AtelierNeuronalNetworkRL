import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    from collections import deque

    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"gymnasium {gym.__version__} | torch {torch.__version__} | device : {DEVICE}"
    )
    return DEVICE, F, Image, SEED, deque, gym, mo, nn, np, plt, random, torch


@app.cell
def _(DEVICE, nn, torch):
    class DQN(nn.Module):
        """CNN + FC. Prédit Q(s,a) pour chaque action discrète. Pas d'activation finale."""

        def __init__(self, n_actions=5):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=8, stride=4),
                nn.ReLU(),  # 42 → 9
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),  # 9  → 3
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 3 * 3, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions),  # Q-valeurs brutes
            )

        def forward(self, x):
            return self.fc(self.conv(x))

    _net = DQN().to(DEVICE)
    _dummy = torch.zeros(8, 1, 42, 42).to(DEVICE)
    with torch.no_grad():
        _q = _net(_dummy)
    print(f"Entrée {tuple(_dummy.shape)} → Q-valeurs {tuple(_q.shape)}")  # (8, 5)
    assert _q.shape == (8, 5), "Dimensions incorrectes !"
    print(f"Paramètres : {sum(p.numel() for p in _net.parameters()):,}")
    return (DQN,)


@app.cell
def _(DEVICE, DQN, F, ReplayBuffer, nn, random, torch):
    class DQNAgent:
        def __init__(
            self,
            n_actions=5,
            gamma=0.95,
            lr=1e-4,
            device=DEVICE,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=0.995,
            buffer_cap=50_000,
        ):
            self.n_actions, self.gamma, self.device = n_actions, gamma, device
            self.policy_net = DQN(n_actions).to(device)
            self.target_net = DQN(n_actions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
            self.buffer = ReplayBuffer(buffer_cap)
            self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay

        def select_action(self, state):
            if random.random() < self.eps:                 # exploration
                return random.randrange(self.n_actions)
            with torch.no_grad():                           # exploitation
                s = torch.tensor(state[None], device=self.device)
                return int(self.policy_net(s).argmax(1).item())

        def store(self, *transition):
            self.buffer.push(*transition)

        def train_step(self, batch_size=64):
            if len(self.buffer) < batch_size:
                return None
            s, a, r, s2, d = self.buffer.sample(batch_size)
            s = torch.tensor(s, device=self.device)
            a = torch.tensor(a, device=self.device, dtype=torch.long).unsqueeze(1)
            r = torch.tensor(r, device=self.device).unsqueeze(1)
            s2 = torch.tensor(s2, device=self.device)
            d = torch.tensor(d, device=self.device).unsqueeze(1)

            q = self.policy_net(s).gather(1, a)  # Q(s,a) prédit
            with torch.no_grad():
                q_next = self.target_net(s2).max(1, keepdim=True)[0]
                target = r + self.gamma * q_next * (1 - d)  # ← équation de Bellman
            loss = F.smooth_l1_loss(q, target)  # Huber Loss
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.opt.step()
            return loss.item()

        def update_epsilon(self):
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

        def update_target(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())

    return (DQNAgent,)


@app.cell
def _(deque, np, random):
    class ReplayBuffer:
        def __init__(self, capacity=50_000):
            self.buf = deque(maxlen=capacity)

        def push(self, s, a, r, s2, done):
            self.buf.append((s, a, r, s2, done))

        def sample(self, batch_size):
            s, a, r, s2, d = zip(*random.sample(self.buf, batch_size))
            return (
                np.stack(s),
                np.array(a),
                np.array(r, dtype=np.float32),
                np.stack(s2),
                np.array(d, dtype=np.float32),
            )

        def __len__(self):
            return len(self.buf)

    return (ReplayBuffer,)


@app.cell
def _(DEVICE, DQNAgent, torch):
    ckpt = torch.load("models/dqn_carracing.pt", map_location=DEVICE)
    agent = DQNAgent(gamma=ckpt["gamma"], n_actions=ckpt["n_actions"])
    agent.policy_net.load_state_dict(ckpt["policy_state"])
    agent.target_net.load_state_dict(ckpt["policy_state"])
    print("Modèle rechargé.")
    return (agent,)


@app.cell
def _(gym):
    class CarRacingWrapper(gym.Wrapper):
        """Saute le zoom d'intro, répète chaque action (frame skip), et termine
           l'épisode quand la voiture reste hors-piste trop longtemps."""
        def __init__(self, env, skip=4, intro=45, off_patience=50):
            super().__init__(env)
            self.skip, self.intro, self.off_patience = skip, intro, off_patience
        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            for _ in range(self.intro):
                obs, _, _, _, info = self.env.step(0)        # sauter le zoom
            self.neg = 0
            return obs, info
        def step(self, action):
            total, term, trunc, info = 0.0, False, False, {}
            for _ in range(self.skip):                        # répéter l'action
                obs, r, term, trunc, info = self.env.step(action)
                total += r
                if term or trunc:
                    break
            self.neg = self.neg + 1 if total < 0 else 0       # reward négatif = hors-piste
            if self.neg >= self.off_patience:
                term = True
            return obs, total, term, trunc, info

    return (CarRacingWrapper,)


@app.cell
def _(mo):
    play_button = mo.ui.run_button(label="▶ Regarder l'agent conduire (politique gloutonne)")
    play_button
    return (play_button,)


@app.cell
def _(Image, SEED, gym, np, plt):
    def preprocess(obs):
        """(96,96,3) RGB uint8 → (1,42,42) grayscale float [0,1]."""
        gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
        img = Image.fromarray(gray.astype(np.uint8)).resize((42, 42), Image.BILINEAR)
        return (np.asarray(img, dtype=np.float32) / 255.0)[None, :, :]  # (1,42,42)

    _env = gym.make("CarRacing-v2", continuous=False)
    print("Action space :", _env.action_space)  # Discrete(5)
    _obs, _ = _env.reset(seed=SEED)
    _state = preprocess(_obs)
    print("Obs brute :", _obs.shape, "→ état prétraité :", _state.shape)
    _env.close()

    fig_pre, _ax = plt.subplots(1, 2, figsize=(7, 3.5))
    _ax[0].imshow(_obs)
    _ax[0].set_title("obs RGB 96×96")
    _ax[0].axis("off")
    _ax[1].imshow(_state[0], cmap="gray")
    _ax[1].set_title("état 42×42 gris")
    _ax[1].axis("off")
    fig_pre.tight_layout()
    fig_pre
    return (preprocess,)


@app.cell
def _(CarRacingWrapper, agent, gym, mo, play_button, plt, preprocess, torch):
    mo.stop(not play_button.value, mo.md("*Entraîne (ou recharge) l'agent d'abord.*"))
    from collections import Counter

    _env = CarRacingWrapper(gym.make("CarRacing-v2", continuous=False, render_mode="human"))
    _obs, _ = _env.reset()
    _state = preprocess(_obs)

    _eps_save = agent.eps
    agent.eps = 0.0                                   # ε=0 → exploitation pure, pas d'aléatoire
    _total, _frames, _acts, _t = 0.0, [], [], 0
    with torch.no_grad():
        for _t in range(400):                          # 400 steps wrappés (×4 frames chacun)
            _a = agent.select_action(_state)
            _acts.append(int(_a))
            _obs, _r, _term, _trunc, _ = _env.step(_a)
            _state = preprocess(_obs); _total += _r
            if _t % 10 == 0: _frames.append(_obs.copy())
            if _term or _trunc: break
    agent.eps = _eps_save
    _env.close()

    _names = {0: "rien", 1: "gauche", 2: "droite", 3: "gas", 4: "frein"}
    _repart = {_names[k]: v for k, v in sorted(Counter(_acts).items())}
    print(f"Récompense totale : {_total:.1f}  ({_t + 1} steps)")
    print("Répartition des actions :", _repart)

    fig_play, _axes = plt.subplots(2, 5, figsize=(14, 6))
    for _ax, _f in zip(_axes.ravel(), _frames):
        _ax.imshow(_f); _ax.axis("off")
    for _ax in _axes.ravel()[len(_frames):]:
        _ax.axis("off")
    fig_play.suptitle(f"Agent DQN — politique gloutonne — reward {_total:.0f}")
    fig_play.tight_layout()
    fig_play
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
