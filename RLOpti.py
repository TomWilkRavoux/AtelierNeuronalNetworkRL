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
def _(torch):
    # Test multi-thread
    import os

    torch.set_num_threads(os.cpu_count())
    print(
        f"Cœurs logiques : {os.cpu_count()} | PyTorch threads : {torch.get_num_threads()}"
    )
    return (os,)


@app.cell
def _(mo):
    mo.md(r"""
    # Phase 4 — Agent autonome (DQN)

    On quitte l'apprentissage supervisé. L'agent n'a **aucune donnée étiquetée** : il agit,
    observe la récompense, et apprend à estimer la **valeur** de chaque action.

    Briques clés :
    - **MDP** : état (frame), action (5 discrètes), récompense, politique π.
    - **Q-Learning** : `Q(s,a)` = récompense attendue si on joue `a` en `s` puis qu'on joue bien.
    - **Équation de Bellman** : `Q(s,a) ← r + γ · max_a' Q_target(s', a')` — la cible d'entraînement.
    - **ε-greedy** : avec proba ε on explore (action aléatoire), sinon on exploite (meilleure Q).
    - **Experience replay** : on stocke les transitions et on ré-échantillonne → décorrèle les données.
    - **Target network** : un réseau figé pour calculer la cible → stabilise l'apprentissage.
    """)
    return


@app.cell
def _():
    # def preprocess(obs):
    #     """(96,96,3) RGB uint8 → (1,42,42) grayscale float [0,1]."""
    #     gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
    #     img = Image.fromarray(gray.astype(np.uint8)).resize((42, 42), Image.BILINEAR)
    #     return (np.asarray(img, dtype=np.float32) / 255.0)[None, :, :]  # (1,42,42)

    # _env = gym.make("CarRacing-v2", continuous=False)
    # print("Action space :", _env.action_space)  # Discrete(5)
    # _obs, _ = _env.reset(seed=SEED)
    # _state = preprocess(_obs)
    # print("Obs brute :", _obs.shape, "→ état prétraité :", _state.shape)
    # _env.close()

    # fig_pre, _ax = plt.subplots(1, 2, figsize=(7, 3.5))
    # _ax[0].imshow(_obs)
    # _ax[0].set_title("obs RGB 96×96")
    # _ax[0].axis("off")
    # _ax[1].imshow(_state[0], cmap="gray")
    # _ax[1].set_title("état 42×42 gris")
    # _ax[1].axis("off")
    # fig_pre.tight_layout()
    # fig_pre
    return


@app.cell
def _(Image, deque, gym, np):
    # 9 actions [steer, gas, brake] — dont des combinaisons frein+braquage
    ACTIONS = [
        np.array([ 0.0, 0.0, 0.0], dtype=np.float32),  # 0 rien
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # 1 gauche
        np.array([ 1.0, 0.0, 0.0], dtype=np.float32),  # 2 droite
        np.array([ 0.0, 1.0, 0.0], dtype=np.float32),  # 3 gas
        np.array([ 0.0, 0.0, 0.8], dtype=np.float32),  # 4 frein
        np.array([-1.0, 0.0, 0.6], dtype=np.float32),  # 5 frein + gauche
        np.array([ 1.0, 0.0, 0.6], dtype=np.float32),  # 6 frein + droite
        np.array([-0.6, 0.4, 0.0], dtype=np.float32),  # 7 gauche + gas
        np.array([ 0.6, 0.4, 0.0], dtype=np.float32),  # 8 droite + gas
    ]

    def _to_gray42(obs):
        gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
        img  = Image.fromarray(gray.astype(np.uint8)).resize((42, 42), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0          # (42,42)

    class CarRacingDQN(gym.Wrapper):
        """Actions combinées + frame skip + empilement de 4 frames + arrêt hors-piste."""
        def __init__(self, env, skip=3, intro=45, off_patience=50, n_stack=4):
            super().__init__(env)
            self.skip, self.intro, self.off_patience, self.n_stack = skip, intro, off_patience, n_stack
            self.frames = deque(maxlen=n_stack)
            self.action_space = gym.spaces.Discrete(len(ACTIONS))
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(n_stack, 42, 42), dtype=np.float32)
        def _state(self):
            return np.stack(self.frames, axis=0)                  # (4,42,42)
        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            for _ in range(self.intro):
                obs, _, _, _, info = self.env.step(np.array([0., 0., 0.], dtype=np.float32))
            g = _to_gray42(obs)
            self.frames.clear()
            for _ in range(self.n_stack):
                self.frames.append(g)
            self.neg = 0
            return self._state(), info
        def step(self, action_idx):
            act, total, term, trunc, info = ACTIONS[int(action_idx)], 0.0, False, False, {}
            for _ in range(self.skip):
                obs, r, term, trunc, info = self.env.step(act)
                total += r
                if term or trunc: break
            self.frames.append(_to_gray42(obs))
            self.neg = self.neg + 1 if total < 0 else 0
            if self.neg >= self.off_patience:
                term = True
            return self._state(), total, term, trunc, info

    return ACTIONS, CarRacingDQN


@app.cell
def _(DEVICE, nn, torch):
    class DQN(nn.Module):
        def __init__(self, n_actions=9):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=8, stride=4), nn.ReLU(),   # 4 frames empilées
                nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 3 * 3, 256), nn.ReLU(),
                nn.Linear(256, n_actions),
            )
        def forward(self, x):
            return self.fc(self.conv(x))

    _net = DQN().to(DEVICE)
    _dummy = torch.zeros(8, 4, 42, 42).to(DEVICE)        # ← 4 canaux (frames empilées)
    with torch.no_grad():
        _q = _net(_dummy)
    print(f"Entrée {tuple(_dummy.shape)} → Q-valeurs {tuple(_q.shape)}")   # (8, 9)
    assert _q.shape == (8, 9), "Dimensions incorrectes !"
    print(f"Paramètres : {sum(p.numel() for p in _net.parameters()):,}")
    return (DQN,)


@app.cell
def _(deque, np, random):
    class ReplayBuffer:
        def __init__(self, capacity=60_000):
            self.buf = deque(maxlen=capacity)
        def push(self, s, a, r, s2, done):
            self.buf.append(((s * 255).astype(np.uint8), a, r, (s2 * 255).astype(np.uint8), done))
        def sample(self, batch_size):
            s, a, r, s2, d = zip(*random.sample(self.buf, batch_size))
            return (np.stack(s).astype(np.float32) / 255., np.array(a),
                    np.array(r, dtype=np.float32),
                    np.stack(s2).astype(np.float32) / 255., np.array(d, dtype=np.float32))
        def __len__(self):
            return len(self.buf)

    return (ReplayBuffer,)


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
            if random.random() < self.eps:  # exploration
                return random.randrange(self.n_actions)
            with torch.no_grad():  # exploitation
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
def _():
    # _agent = DQNAgent()
    # _env = gym.make("CarRacing-v2", continuous=False)
    # _obs, _ = _env.reset(seed=SEED)
    # _st = preprocess(_obs)

    # _a = _agent.select_action(_st)  # ← méthode single-state (pas batched)
    # _obs2, _r, _term, _trunc, _ = _env.step(_a)
    # _agent.store(_st, _a, _r, preprocess(_obs2), float(_term or _trunc))
    # _env.close()

    # print(
    #     f"Action choisie : {_a}  | reward : {_r:.2f}  | buffer : {len(_agent.buffer)}"
    # )
    # print("train_step (buffer trop petit) :", _agent.train_step())  # None attendu
    return


@app.cell
def _(np, torch):
    def select_actions_batched(agent, states, eps):
        n = len(states)
        rand = np.random.randint(agent.n_actions, size=n)  # actions exploratoires
        explore = np.random.random(n) < eps
        with torch.no_grad():
            s = torch.tensor(states, device=agent.device)
            greedy = agent.policy_net(s).argmax(1).cpu().numpy()  # actions exploitantes
        return np.where(explore, rand, greedy)

    return (select_actions_batched,)


@app.cell
def _():
    # class CarRacingWrapper(gym.Wrapper):
    #     """Saute le zoom d'intro, répète chaque action (frame skip), et termine
    #     l'épisode quand la voiture reste hors-piste trop longtemps."""

    #     def __init__(self, env, skip=4, intro=45, off_patience=50):
    #         super().__init__(env)
    #         self.skip, self.intro, self.off_patience = skip, intro, off_patience

    #     def reset(self, **kw):
    #         obs, info = self.env.reset(**kw)
    #         for _ in range(self.intro):
    #             obs, _, _, _, info = self.env.step(0)  # sauter le zoom
    #         self.neg = 0
    #         return obs, info

    #     def step(self, action):
    #         total, term, trunc, info = 0.0, False, False, {}
    #         for _ in range(self.skip):  # répéter l'action
    #             obs, r, term, trunc, info = self.env.step(action)
    #             total += r
    #             if term or trunc:
    #                 break
    #         self.neg = self.neg + 1 if total < 0 else 0  # reward négatif = hors-piste
    #         if self.neg >= self.off_patience:
    #             term = True
    #         return obs, total, term, trunc, info
    return


@app.cell
def _(SEED, np, plt, select_actions_batched):
    def train_dqn_vec(agent, envs, total_steps=300_000, target_every=2000,
                      batch_size=64, eps_end=0.05, explore_frac=0.5, verbose=True):
        import copy
        obs, _ = envs.reset(seed=SEED)
        states = obs.astype(np.float32)
        n_envs = envs.num_envs
        ep_r, completed, gstep, last_target = np.zeros(n_envs), [], 0, 0
        best_ma, best_state = -1e9, None                      # ← suivi du meilleur

        while gstep < total_steps:
            agent.eps = max(eps_end, 1.0 - gstep / (explore_frac * total_steps))
            actions = select_actions_batched(agent, states, agent.eps)
            obs, r, term, trunc, _ = envs.step(actions)
            nstates = obs.astype(np.float32)
            done = np.logical_or(term, trunc)
            for i in range(n_envs):
                agent.store(states[i], int(actions[i]), float(r[i]), nstates[i], float(done[i]))
            states = nstates
            ep_r += r; gstep += n_envs
            agent.train_step(batch_size)
            if gstep - last_target >= target_every:
                agent.update_target(); last_target = gstep
            for i in range(n_envs):
                if done[i]:
                    completed.append(ep_r[i]); ep_r[i] = 0.0

            if len(completed) >= 20:                          # ← snapshot du meilleur
                _ma = np.mean(completed[-20:])
                if _ma > best_ma:
                    best_ma, best_state = _ma, copy.deepcopy(agent.policy_net.state_dict())

            if verbose and len(completed) and gstep % (n_envs * 200) < n_envs:
                print(f"steps={gstep:6d} | épisodes={len(completed):4d} "
                      f"| reward moy(20)={np.mean(completed[-20:]):7.1f} | ε={agent.eps:.3f}")

        if best_state is not None:                            # ← on restaure le meilleur
            agent.policy_net.load_state_dict(best_state)
            print(f"Meilleur reward moy(20) = {best_ma:.1f} — modèle restauré.")
        return completed

    def plot_rewards(rewards, window=10):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rewards, alpha=0.35, color="steelblue", label="reward / épisode")
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                range(window - 1, len(rewards)),
                ma,
                lw=2,
                color="orange",
                label=f"moyenne mobile ({window})",
            )
        ax.set(xlabel="Épisode", ylabel="Récompense", title="Apprentissage DQN")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    return plot_rewards, train_dqn_vec


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="▶ Entraîner le DQN sur tous les cœurs")
    train_button
    return (train_button,)


@app.cell
def _(
    ACTIONS,
    CarRacingDQN,
    DQNAgent,
    SEED,
    gym,
    mo,
    os,
    plot_rewards,
    random,
    torch,
    train_button,
    train_dqn_vec,
):
    mo.stop(not train_button.value, mo.md("*Clique pour l'entraînement v2 (long : 30-60 min CPU).*"))

    N_ENVS = min(os.cpu_count(), 8)
    def _make_env():
        return CarRacingDQN(gym.make("CarRacing-v2", continuous=True))   # continuous=True !

    envs = gym.vector.AsyncVectorEnv([_make_env for _ in range(N_ENVS)])
    torch.manual_seed(SEED); random.seed(SEED)
    agent   = DQNAgent(n_actions=len(ACTIONS), gamma=0.99, lr=1e-4, buffer_cap=60_000)
    rewards = train_dqn_vec(agent, envs, total_steps=300_000)
    envs.close()
    plot_rewards(rewards, window=20)
    return agent, rewards


@app.cell
def _(agent, torch):
    # Sauvegarder uniquement le modèle (policy_net)
    torch.save(agent.policy_net.state_dict(), "dqn_model_v2_entraine.pth")
    print("Modèle sauvegardé sous 'dqn_model_entraine.pth'.")
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="💾 Sauvegarder le modèle entraîné")
    save_button
    return (save_button,)


@app.cell
def _(agent, mo, os, save_button, torch):
    mo.stop(
        not save_button.value, mo.md("*Clique pour sauvegarder le réseau entraîné.*")
    )

    os.makedirs("models", exist_ok=True)
    _path = "models/dqn_v2_carracing.pt"
    torch.save(
        {
            "policy_state": agent.policy_net.state_dict(),
            "gamma": agent.gamma,
            "n_actions": agent.n_actions,
        },
        _path,
    )
    print(f"Modèle sauvegardé → {os.path.abspath(_path)}")
    return


@app.cell
def _(plot_rewards, rewards):
    plot_rewards(rewards, window=10)
    return


@app.cell
def _(mo):
    watch_button = mo.ui.run_button(label="▶ Regarder l'agent entraîné conduire")
    watch_button
    return (watch_button,)


@app.cell
def _(SEED, agent, gym, mo, plt, preprocess, torch, watch_button):
    mo.stop(not watch_button.value, mo.md("*Entraîne d'abord l'agent (cellule 9).*"))

    _env = gym.make("CarRacing-v2", continuous=False)
    _obs, _ = _env.reset(seed=SEED + 999)
    for _ in range(45):
        _obs, *_ = _env.step(0)
    _state = preprocess(_obs)

    _eps_save = agent.eps
    agent.eps = 0.0  # exploitation pure (ε=0)
    _total, _frames, _t = 0.0, [], 0
    with torch.no_grad():
        for _t in range(1000):
            _a = agent.select_action(_state)
            _obs, _r, _term, _trunc, _ = _env.step(_a)
            _state = preprocess(_obs)
            _total += _r
            if _t % 30 == 0:
                _frames.append(_obs.copy())
            if _term or _trunc:
                break
    agent.eps = _eps_save
    _env.close()

    print(f"Récompense totale (politique gloutonne) : {_total:.1f}")
    fig_w, _axes = plt.subplots(2, 5, figsize=(14, 6))
    for _ax, _f in zip(_axes.ravel(), _frames):
        _ax.imshow(_f)
        _ax.axis("off")
    for _ax in _axes.ravel()[len(_frames) :]:
        _ax.axis("off")
    fig_w.suptitle(f"Agent DQN entraîné — reward {_total:.0f}")
    fig_w.tight_layout()
    fig_w
    return


@app.cell
def _(mo):
    gamma_button = mo.ui.run_button(
        label="▶ Comparer γ ∈ {0.90, 0.95, 0.99} (TRÈS long : 3 entraînements)"
    )
    gamma_button
    return (gamma_button,)


@app.cell
def _(
    DQNAgent,
    SEED,
    gamma_button,
    gym,
    mo,
    np,
    plt,
    random,
    torch,
    train_dqn,
):
    mo.stop(
        not gamma_button.value,
        mo.md("*Clique pour 3 entraînements (25 ép. chacun) — compte un bon moment.*"),
    )

    rewards_by_gamma = {}
    fig_g, ax_g = plt.subplots(figsize=(8, 4))
    for _g in [0.90, 0.95, 0.99]:
        torch.manual_seed(SEED)
        random.seed(SEED)
        _ag = DQNAgent(gamma=_g)
        _env = gym.make("CarRacing-v2", continuous=False)
        _rw = train_dqn(_ag, _env, n_episodes=25, verbose=False)
        _env.close()
        rewards_by_gamma[_g] = _rw
        _w = 5
        _ma = np.convolve(_rw, np.ones(_w) / _w, mode="valid")
        ax_g.plot(range(_w - 1, len(_rw)), _ma, label=f"γ={_g}")
        print(f"γ={_g} → reward moyen (10 derniers ép.) = {np.mean(_rw[-10:]):.1f}")

    ax_g.set(xlabel="Épisode", ylabel="Récompense (moy. mobile)", title="Impact de γ")
    ax_g.legend()
    ax_g.grid(alpha=0.3)
    fig_g.tight_layout()
    fig_g
    return


@app.cell
def _(mo):
    play_button = mo.ui.run_button(
        label="▶ Regarder l'agent conduire (politique gloutonne)"
    )
    play_button
    return (play_button,)


@app.cell
def _(CarRacingDQN, SEED, agent, gym, mo, play_button, torch):
    mo.stop(not play_button.value, mo.md("*Entraîne (ou recharge) l'agent d'abord.*"))
    from collections import Counter

    _env = CarRacingDQN(gym.make("CarRacing-v2", continuous=True, render_mode="human"))
    _obs, _ = _env.reset(seed=SEED + 1234)
    _state = _obs
    _eps_save = agent.eps; agent.eps = 0.0
    _total, _acts, _t = 0.0, [], 0
    with torch.no_grad():
        for _t in range(800):
            _a = agent.select_action(_state)
            _acts.append(int(_a))
            _obs, _r, _term, _trunc, _ = _env.step(_a)
            _state = _obs; _total += _r
            if _term or _trunc: break
    agent.eps = _eps_save; _env.close()

    _names = {0:"rien",1:"gauche",2:"droite",3:"gas",4:"frein",
              5:"frein+G",6:"frein+D",7:"gauche+gas",8:"droite+gas"}
    _repart = {_names[k]: v for k, v in sorted(Counter(_acts).items())}
    print(f"Récompense totale : {_total:.1f}  ({_t + 1} steps)")
    print("Répartition des actions :", _repart)
    return


@app.cell
def _(mo):
    record_button = mo.ui.run_button(label="⏺ Enregistrer une vidéo de l'agent")
    record_button
    return (record_button,)


@app.cell
def _(CarRacingDQN, SEED, agent, gym, mo, os, record_button, torch):
    mo.stop(not record_button.value, mo.md("*Enregistre un mp4 dans ./videos (`uv add moviepy`).*"))
    from gymnasium.wrappers import RecordVideo
    os.makedirs("videos", exist_ok=True)
    _rec = RecordVideo(gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array"),
                       video_folder="videos", name_prefix="dqn_v2", episode_trigger=lambda e: True)
    _env = CarRacingDQN(_rec)
    _obs, _ = _env.reset(seed=SEED + 1234); _state = _obs
    _eps_save = agent.eps; agent.eps = 0.0
    _total, _t = 0.0, 0
    with torch.no_grad():
        for _t in range(800):
            _a = agent.select_action(_state)
            _obs, _r, _term, _trunc, _ = _env.step(_a)
            _state = _obs; _total += _r
            if _term or _trunc: break
    agent.eps = _eps_save; _env.close()

    _files = sorted(f for f in os.listdir("videos") if f.endswith(".mp4"))
    print(f"Vidéo écrite — reward {_total:.1f}  ({_t + 1} steps)")
    mo.video(src=os.path.join("videos", _files[-1])) if _files else mo.md("Aucun mp4 trouvé.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
