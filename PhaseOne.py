import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import gymnasium as gym
    from pyvis.network import Network

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # activer cuda si dispo sinon fallback cpu
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"gymnasium {gym.__version__} | torch {torch.__version__} | device : {DEVICE}")
    return (
        DEVICE,
        DataLoader,
        Dataset,
        F,
        Network,
        SEED,
        gym,
        mo,
        nn,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploration de l'env / dataset
    """)
    return


@app.cell
def _(SEED, gym, np, plt):
    _env = gym.make("CarRacing-v2", render_mode=None, continuous=True)
    print("Observation :", _env.observation_space)  # Box(0, 255, (96, 96, 3))
    print("Action      :", _env.action_space)       # Box([-1,0,0], [1,1,1], (3,))

    _obs, _ = _env.reset(seed=SEED)
    for _ in range(45):                              # sauter le zoom d'intro
        _obs, *_rest = _env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    demo_frames = []
    for _ in range(8):
        for _ in range(8):                           # avancer pour varier les vues
            _obs, *_rest = _env.step(_env.action_space.sample())
        demo_frames.append(_obs.copy())
    _env.close()

    fig_frames, axes_frames = plt.subplots(2, 4, figsize=(12, 6))
    for _ax, _frame in zip(axes_frames.ravel(), demo_frames):
        _ax.imshow(_frame)
        _ax.axis("off")
    fig_frames.suptitle("8 frames variées de CarRacing-v2")
    fig_frames.tight_layout()
    fig_frames
    return


@app.cell
def _(np, plt):
    _x = np.linspace(-4, 4, 200)

    def _relu(x):    return np.maximum(0, x)
    def _sigmoid(x): return 1 / (1 + np.exp(-x))
    def _tanh(x):    return np.tanh(x)
    def _leaky(x):   return np.where(x > 0, x, 0.01 * x)

    fig_acts, axes_acts = plt.subplots(1, 4, figsize=(14, 3.2))
    for _ax, (_name, _f) in zip(
        axes_acts,
        [("ReLU", _relu), ("Sigmoid", _sigmoid), ("Tanh", _tanh), ("Leaky ReLU", _leaky)],
    ):
        _ax.plot(_x, _f(_x), color="steelblue", lw=2)
        _ax.set_title(_name)
        _ax.axhline(0, color="gray", lw=0.5)
        _ax.axvline(0, color="gray", lw=0.5)
        _ax.grid(alpha=0.3)
    fig_acts.tight_layout()
    fig_acts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Archi model PerseptionCNN
    """)
    return


@app.cell
def _(F, nn, torch):
    class PerceptionCNN(nn.Module):
        """
        CNN de perception : classifie le type de route (tout droit / virage G / virage D).
        Entrée : (batch, 3, 96, 96) — image RGB normalisée dans [0, 1]
        Sortie : (batch, n_classes) — logits bruts (PAS de softmax : voir note ci-dessous)
        """
        def __init__(self, n_classes=3, dropout=0.3):
            super().__init__()
            # --- Bloc convolutionnel : extraction de features visuelles ---
            self.conv1 = nn.Conv2d(3,  8, kernel_size=8, stride=4)   # 96 -> 23
            self.bn1   = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8,  16, kernel_size=4, stride=2)  # 23 -> 10
            self.bn2   = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)  # 10 -> 8
            self.bn3   = nn.BatchNorm2d(32)

            # --- Tête de classification ---
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)   # 2048 -> 128
            self.fc2 = nn.Linear(128, n_classes)    # 128 -> 3

        def forward(self, x):
            # Schéma standard : Conv -> BatchNorm -> ReLU, répété 3 fois
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.flatten(x, start_dim=1)        # (batch, 2048), on garde la dim batch
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)                          # logits, sans activation finale
            return x

    return (PerceptionCNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    mo.md(
    ## Architecture du `PerceptionCNN`

    Le réseau suit le schéma classique d'un CNN de classification d'images :
    un **bloc convolutionnel** qui extrait des features visuelles, suivi d'une
    **tête fully-connected** qui transforme ces features en décision (le type de route).

    ### Le bloc convolutionnel — *« voir »*

    | Couche | Rôle | Sortie |
    |--------|------|--------|
    | Entrée | Image RGB normalisée dans [0, 1] | `(B, 3, 96, 96)` |
    | Conv1 (8 filtres 8×8, stride 4) + BN + ReLU | Détecte les motifs grossiers (bords route/herbe) et réduit fortement la résolution | `(B, 8, 23, 23)` |
    | Conv2 (16 filtres 4×4, stride 2) + BN + ReLU | Combine les motifs locaux en formes intermédiaires | `(B, 16, 10, 10)` |
    | Conv3 (32 filtres 3×3, stride 1) + BN + ReLU | Affine des features plus abstraites (courbure, orientation) | `(B, 32, 8, 8)` |

    Les dimensions spatiales se calculent avec :

    `sortie = floor( (entrée − kernel) / stride ) + 1`

    - Conv1 : `floor((96 − 8) / 4) + 1 = 23`
    - Conv2 : `floor((23 − 4) / 2) + 1 = 10`
    - Conv3 : `floor((10 − 3) / 1) + 1 = 8`

    On part de gros filtres à grand pas (8×8, stride 4) pour résumer vite une image
    où l'information utile est à grande échelle (la position de la bande de route),
    puis on rétrécit les filtres pour raffiner. Le nombre de canaux augmente
    (8 → 16 → 32) à mesure que la résolution baisse : on troque de la précision
    spatiale contre une représentation plus riche.

    ### La tête de classification — *« décider »*

    | Couche | Rôle | Sortie |
    |--------|------|--------|
    | Flatten | Met les features à plat | `(B, 2048)` |
    | Dropout(0.3) | Régularisation : éteint 30 % des unités à l'entraînement | `(B, 2048)` |
    | FC1 (2048 → 128) + ReLU | Combine toutes les features en un vecteur compact | `(B, 128)` |
    | FC2 (128 → 3) | Produit un *logit* par classe | `(B, 3)` |

    ### Justification des choix

    - **Conv → BatchNorm → ReLU** : la BatchNorm normalise les activations *avant*
      la non-linéarité, ce qui stabilise et accélère l'entraînement. Son argument
      est le nombre de **canaux** (ex. `BatchNorm2d(8)`), pas une taille spatiale.
    - **ReLU** dans les couches cachées : pas de saturation pour `x > 0`, gradients
      simples, calcul rapide — le standard en CNN (vs Sigmoid/Tanh qui saturent).
    - **Dropout** placé juste avant la tête FC, là où le risque de surapprentissage
      est le plus fort (c'est la zone qui contient le plus de paramètres).
    - **Pas d'activation finale** : la sortie ce sont des *logits* bruts.
      `CrossEntropyLoss` applique elle-même un `log_softmax` en interne — ajouter
      un softmax manuel l'appliquerait deux fois et dégraderait l'apprentissage.

    ### Pourquoi un CNN plutôt qu'un MLP ?

    Le réseau totalise **≈ 271 000 paramètres**, dont l'essentiel dans `fc1`
    (2048 × 128 ≈ 262 k). Tout le bloc convolutionnel, qui fait pourtant le gros du
    travail visuel, ne pèse que ~8 k paramètres : c'est l'effet du **partage des
    poids** (un même filtre glisse sur toute l'image). Un MLP qui aplatirait
    directement l'image aurait, rien que pour sa première couche,
    96 × 96 × 3 × 128 ≈ **3,5 millions** de paramètres — bien plus lourd, sans
    exploiter la structure spatiale ni l'invariance à la translation.
    )
    """)
    return


@app.cell
def _(PerceptionCNN, torch):
    _model = PerceptionCNN(n_classes=3)
    _dummy = torch.zeros(4, 3, 96, 96)
    with torch.no_grad():
        _out = _model(_dummy)

    print(f"Entrée {tuple(_dummy.shape)} → Sortie {tuple(_out.shape)}")
    assert _out.shape == (4, 3), "Dimensions incorrectes !"

    _total = sum(p.numel() for p in _model.parameters())
    print(f"Paramètres : {_total:,}")
    return


@app.cell
def _(mo):
    collect_button = mo.ui.run_button(label="▶ Collecter ~1200 frames")
    collect_button
    return (collect_button,)


@app.cell
def _(SEED, collect_button, gym, mo, np):
    mo.stop(
        not collect_button.value,
        mo.md("*Clique sur le bouton ci-dessus pour lancer la collecte (~20-30 s).*"),
    )

    def collect_cyclic(n_per_phase=20, n_cycles=20, gas=0.4, steer=0.6,
                       skip_zoom=45, seed=SEED):
        """
        Agent cyclique : alterne tout droit / gauche / droite.
        Label de chaque frame = manœuvre commandée.
        Classes : 0 = tout droit · 1 = virage gauche · 2 = virage droite
        """
        env = gym.make("CarRacing-v2", render_mode=None, continuous=True)
        obs, _ = env.reset(seed=seed)
        for _ in range(skip_zoom):                       # sauter le zoom d'intro
            obs, *_ = env.step(np.array([0., 0., 0.], dtype=np.float32))

        phases = [
            (np.array([0.0,    gas, 0.0], dtype=np.float32), 0),  # tout droit
            (np.array([-steer, gas, 0.0], dtype=np.float32), 1),  # gauche
            (np.array([ steer, gas, 0.0], dtype=np.float32), 2),  # droite
        ]

        frames, labels = [], []
        for _ in range(n_cycles):
            for action, label in phases:
                for _ in range(n_per_phase):
                    obs, _, terminated, truncated, _ = env.step(action)
                    frames.append(obs.copy())
                    labels.append(label)
                    if terminated or truncated:          # ex. limite de 1000 pas
                        obs, _ = env.reset()
                        for _ in range(skip_zoom):
                            obs, *_ = env.step(np.array([0., 0., 0.], dtype=np.float32))
        env.close()
        return np.array(frames), np.array(labels, dtype=np.int64)

    frames_all, labels_all = collect_cyclic()

    print(f"Frames : {frames_all.shape}   Labels : {labels_all.shape}")
    _u, _c = np.unique(labels_all, return_counts=True)
    print("Répartition classes :", dict(zip(_u.tolist(), _c.tolist())))
    return frames_all, labels_all


@app.cell
def _(DataLoader, Dataset, SEED, frames_all, labels_all, np, torch):
    class CarFrameDataset(Dataset):
        def __init__(self, frames, labels):
            # Gym donne (N, H, W, C) ; PyTorch veut (N, C, H, W). Normalisation /255.
            x = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            self.X = torch.tensor(x)
            self.y = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            return self.X[i], self.y[i]

    full_ds = CarFrameDataset(frames_all, labels_all)

    _n       = len(full_ds)
    _n_train = int(0.70 * _n)
    _n_val   = int(0.15 * _n)
    _n_test  = _n - _n_train - _n_val

    _gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [_n_train, _n_val, _n_test], generator=_gen
    )

    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    print(f"Train : {len(train_ds)} | Val : {len(val_ds)} | Test : {len(test_ds)}")
    return full_ds, test_ds, test_ld, train_ld, val_ld


@app.cell
def _(mo):
    mo.md(r"""
    ### Split train / val / test

    Split **70/15/15 aléatoire**, graine fixée (`SEED=42`). Pour de la classification d'images traitées indépendamment, l'aléatoire est acceptable.

    *Limite assumée* : les frames d'une même phase de l'agent cyclique sont quasi identiques d'un pas à l'autre. Un split aléatoire met donc des frames presque jumelles en train **et** en test → métriques légèrement **optimistes**. C'est précisément ce que la Phase 2 corrigera avec un **split temporel par épisode**.
    """)
    return


@app.cell
def _(frames_all, labels_all, np, plt):
    _names = {0: "tout droit", 1: "gauche", 2: "droite"}
    fig_labels, axes_labels = plt.subplots(1, 3, figsize=(10, 3.5))
    for _cls, _ax in zip([0, 1, 2], axes_labels):
        _idx = np.where(labels_all == _cls)[0][30]   # une frame de cette classe
        _ax.imshow(frames_all[_idx])
        _ax.set_title(f"classe {_cls} — {_names[_cls]}")
        _ax.axis("off")
    fig_labels.tight_layout()
    fig_labels
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrainement model
    """)
    return


@app.cell
def _(DEVICE, nn, torch):
    def train_cnn(model, train_ld, val_ld, n_epochs=20, lr=1e-3, device=DEVICE):
        import copy
        criterion = nn.CrossEntropyLoss()           # softmax + NLL intégrés
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # L'O.G des optimizer
        history = {"tl": [], "vl": [], "ta": [], "va": []}
        best_vl, best_state = float("inf"), None

        for ep in range(1, n_epochs + 1):
            # Entrainement
            model.train()
            tl, correct, total = 0., 0, 0
            for xb, yb in train_ld:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()               # 1. remettre les gradients à zéro
                out  = model(xb)                    # 2. forward
                loss = criterion(out, yb)           # 3. loss
                loss.backward()                     # 4. backward (rétropropagation)
                optimizer.step()                    # 5. mise à jour des poids
                tl += loss.item()
                correct += (out.argmax(1) == yb).sum().item()
                total   += yb.size(0)
            tl /= len(train_ld); ta = correct / total

            # validation
            model.eval()
            vl, vcorrect, vtotal = 0., 0, 0
            with torch.no_grad():                   # pas de gradient en validation
                for xb, yb in val_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    vl += criterion(out, yb).item()
                    vcorrect += (out.argmax(1) == yb).sum().item()
                    vtotal   += yb.size(0)
            vl /= len(val_ld); va = vcorrect / vtotal

            for k, v in zip(["tl", "vl", "ta", "va"], [tl, vl, ta, va]):
                history[k].append(v)

            if vl < best_vl:                        # garder le meilleur modèle (val_loss)
                best_vl, best_state = vl, copy.deepcopy(model.state_dict())

            print(f"Ep {ep:2d} | train: loss={tl:.4f} acc={ta:.3f} "
                  f"| val: loss={vl:.4f} acc={va:.3f}")

        model.load_state_dict(best_state)           # restaurer le meilleur
        return history

    return (train_cnn,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="▶ Entraîner le CNN")
    train_button
    return (train_button,)


@app.cell
def _(DEVICE, PerceptionCNN, mo, train_button, train_cnn, train_ld, val_ld):
    mo.stop(not train_button.value, mo.md("*Clique pour lancer l'entraînement (~30 s).*"))

    model   = PerceptionCNN(n_classes=3).to(DEVICE)
    history = train_cnn(model, train_ld, val_ld, n_epochs=20)
    return history, model


@app.cell
def _(history, plt):
    fig_curves, (_ax_l, _ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    _epochs = range(1, len(history["tl"]) + 1)

    _ax_l.plot(_epochs, history["tl"], label="train")
    _ax_l.plot(_epochs, history["vl"], label="val")
    _ax_l.set(xlabel="Epoch", ylabel="CrossEntropy", title="Loss"); _ax_l.legend(); _ax_l.grid(alpha=0.3)

    _ax_a.plot(_epochs, history["ta"], label="train")
    _ax_a.plot(_epochs, history["va"], label="val")
    _ax_a.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy"); _ax_a.legend(); _ax_a.grid(alpha=0.3)

    _gap = history["ta"][-1] - history["va"][-1]
    if   _gap > 0.10:               print(f"→ Écart train-val = {_gap:.2f} : surapprentissage probable.")
    elif history["ta"][-1] < 0.70:  print("→ Accuracy train faible : sous-apprentissage probable.")
    else:                           print("→ Courbes proches et hautes : bonne convergence ✓")

    fig_curves.tight_layout()
    fig_curves
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Question 3 — Surapprentissage ?

    Train accuracy ~0,95 stable, val accuracy qui **oscille** (~0,6 à ~0,95) avec des pics de val_loss. L'écart final train–val (~0,14) indique un **léger surapprentissage**, mais l'instabilité vient surtout d'une **validation petite (~180 frames) et bruitée** : la CrossEntropy explose sur quelques erreurs confiantes alors que l'accuracy reste stable. La **test accuracy 0,944** confirme une bonne généralisation.

    Régularisations déjà en place : **Dropout(0,3)** + **BatchNorm**. Pistes pour réduire l'écart : plus de Dropout, `weight_decay` (L2) sur Adam, data augmentation, plus de frames. L'**ablation de la BatchNorm** ci-dessous le teste empiriquement.
    """)
    return


@app.cell
def _(DEVICE, model, test_ld, torch):
    model.eval()
    _correct = _total = 0
    with torch.no_grad():
        for _xb, _yb in test_ld:
            _pred = model(_xb.to(DEVICE)).argmax(1).cpu()
            _correct += (_pred == _yb).sum().item()
            _total   += _yb.size(0)
    print(f"Accuracy test : {_correct / _total:.3f}")
    return


@app.cell
def _(DEVICE, model, plt, test_ds, torch):
    _sample, _ = test_ds[0]                          # un exemple, (3, 96, 96)
    model.eval()
    _acts = {}
    _hook = model.conv1.register_forward_hook(
        lambda m, i, o: _acts.update({"conv1": o.detach().cpu()})
    )
    with torch.no_grad():
        model(_sample.unsqueeze(0).to(DEVICE))       # déclenche le hook
    _hook.remove()                                   # toujours retirer le hook

    _fmaps = _acts["conv1"][0]                        # (8, 23, 23)

    fig_fmaps, _axf = plt.subplots(2, 5, figsize=(13, 5.5))
    _axf[0, 0].imshow(_sample.permute(1, 2, 0).numpy())   # (C,H,W) -> (H,W,C)
    _axf[0, 0].set_title("image"); _axf[0, 0].axis("off")
    for _k in range(8):
        _ax = _axf.ravel()[_k + 1]
        _ax.imshow(_fmaps[_k], cmap="RdBu_r")        # rouge/bleu = activations +/−
        _ax.set_title(f"filtre {_k}"); _ax.axis("off")
    _axf.ravel()[9].axis("off")                      # case vide
    fig_fmaps.suptitle("Conv1 — image + 8 feature maps")
    fig_fmaps.tight_layout()
    fig_fmaps
    return


@app.cell
def _(DEVICE, model, plt, test_ds, torch):
    _sample, _ = test_ds[0]
    model.eval()

    # Un hook sur chaque couche conv. Le n=_name fige la variable dans la lambda
    # (sinon les 3 hooks pointeraient tous vers "conv3" — piège classique de closure).
    _acts, _hooks = {}, []
    for _name in ["conv1", "conv2", "conv3"]:
        _hooks.append(
            getattr(model, _name).register_forward_hook(
                lambda m, i, o, n=_name: _acts.update({n: o.detach().cpu()})
            )
        )

    with torch.no_grad():
        model(_sample.unsqueeze(0).to(DEVICE))
    for _hk in _hooks:
        _hk.remove()

    _n_show = 8                                   # nb de filtres affichés par couche
    fig_evo, _axes = plt.subplots(3, _n_show, figsize=(15, 6))
    for _row, _name in enumerate(["conv1", "conv2", "conv3"]):
        _fmaps = _acts[_name][0]                  # (C, H, W)
        for _k in range(_n_show):
            _ax = _axes[_row, _k]
            _ax.imshow(_fmaps[_k], cmap="RdBu_r")
            _ax.set_xticks([]); _ax.set_yticks([])
            if _row == 0:
                _ax.set_title(f"filtre {_k}", fontsize=9)
        _axes[_row, 0].set_ylabel(f"{_name}\n{tuple(_fmaps.shape)}", fontsize=10)

    fig_evo.suptitle("Évolution des feature maps : Conv1 → Conv2 → Conv3")
    fig_evo.tight_layout()
    fig_evo
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Question 2 — Que détectent les filtres ?

    Chaque feature map est la **réponse d'un filtre** (rouge = activation positive, bleu = négative), pas une version colorée de l'image. En **Conv1**, les filtres captent du **bas niveau** : bords, orientations, contraste route/herbe — on reconnaît encore la diagonale de la route. Certains détectent ce bord (avec des polarités opposées d'un filtre à l'autre), d'autres réagissent uniformément (luminosité), quelques-uns restent inactifs sur cette frame.

    De **Conv1 → Conv2 → Conv3**, la résolution chute (23→10→8), les canaux montent (8→16→32) et l'**interprétabilité visuelle s'effondre** : les cartes profondes encodent des features abstraites pour la tête FC, plus pour l'œil. C'est la hiérarchie typique d'un CNN : générique en surface, spécifique à la tâche en profondeur.
    """)
    return


@app.cell
def _(Network, mo):
    _layers = [
        ("Input",   "(3, 96, 96)"),
        ("Conv1",   "8×8 s4 → (8, 23, 23)"),
        ("Conv2",   "4×4 s2 → (16, 10, 10)"),
        ("Conv3",   "3×3 s1 → (32, 8, 8)"),
        ("Flatten", "(2048,)"),
        ("FC1",     "→ 128 + ReLU"),
        ("FC2",     "→ 3 (logits)"),
    ]

    _net = Network(height="300px", width="100%", directed=True)
    _net.toggle_physics(False)
    for _i, (_name, _shape) in enumerate(_layers):
        _net.add_node(_i, label=_name, title=_shape, x=_i * 170, y=0, shape="box")
    for _i in range(len(_layers) - 1):
        _net.add_edge(_i, _i + 1)

    mo.iframe(_net.generate_html())
    return


@app.cell
def _(DEVICE, DataLoader, full_ds, mo, model, torch):
    from pyvis.network import Network as _Network

    # Embeddings 128-D de TOUTES les frames (sortie de fc1) via un hook
    model.eval()
    _store = {}
    _h = model.fc1.register_forward_hook(lambda m, i, o: _store.update({"e": o.detach().cpu()}))

    _embs, _labs = [], []
    with torch.no_grad():
        for _xb, _yb in DataLoader(full_ds, batch_size=64, shuffle=False):
            model(_xb.to(DEVICE))
            _embs.append(_store["e"]); _labs.append(_yb)
    _h.remove()

    embeddings  = torch.cat(_embs)            # (N, 128) — la représentation apprise
    node_labels = torch.cat(_labs).numpy()    # (N,)     — classe de chaque frame

    # Graphe des k plus proches voisins dans l'espace d'embedding ──
    K = 6
    _dist = torch.cdist(embeddings, embeddings)     # (N, N)
    _dist.fill_diagonal_(float("inf"))              # pas de self-loop
    _knn = _dist.topk(K, largest=False).indices.numpy()

    # Rendu pyvis, même style que ton exemple Cora ──
    _palette = ['#3498db', '#e74c3c', '#2ecc71']            # 0=droit 1=gauche 2=droite
    _classes = {0: "tout droit", 1: "virage gauche", 2: "virage droite"}

    _net = _Network(height="900px", width="100%", bgcolor="#1a1a2e", font_color="white")
    for _n in range(len(node_labels)):
        _c = int(node_labels[_n])
        _net.add_node(_n, label="", title=f"Frame {_n} — {_classes[_c]}",
                      color=_palette[_c], size=10)

    _seen = set()
    for _n in range(_knn.shape[0]):
        for _v in _knn[_n]:
            _v = int(_v)
            _e = (min(_n, _v), max(_n, _v))
            if _e not in _seen:
                _seen.add(_e)
                _net.add_edge(_n, _v)

    _net.set_options('''
    {
      "physics": {
        "forceAtlas2Based": {"gravitationalConstant": -80, "centralGravity": 0.005,
                             "springLength": 50, "springConstant": 0.05},
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 300, "fit": true},
        "stopSimulationAfterStabilization": true
      },
      "nodes": {"shape": "dot", "borderWidth": 0},
      "edges": {"smooth": false, "width": 0.3},
      "interaction": {"hideEdgesOnDrag": true, "tooltipDelay": 100}
    }
    ''')
    _net.save_graph("graphe_knn.html")
    mo.iframe(_net.generate_html())
    return


@app.cell
def _(DEVICE, mo, model):
    from torchinfo import summary

    _info = summary(
        model, input_size=(1, 3, 96, 96), device=DEVICE, verbose=0,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
    )
    mo.md(f"```\n{_info}\n```")
    return


@app.cell
def _(DEVICE, model):
    from torchview import draw_graph

    _g = draw_graph(
        model, input_size=(1, 3, 96, 96), device=DEVICE,
        graph_name="PerceptionCNN", expand_nested=True,
    )
    _g.visual_graph        # objet Graphviz → marimo le rend en SVG inline
    return


@app.cell
def _(model, torch):
    import copy
    import ml_dtypes

    _m = copy.deepcopy(model).cpu().eval()
    torch.onnx.export(
        _m, torch.zeros(1, 3, 96, 96), "cnn.onnx",
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print("Exporté -> cnn.onnx glisse-le dans netron.app ou retirer décomenter la cellules d'apres")
    return


@app.cell
def _():
    # import netron; 
    # netron.start("cnn.onnx")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusion — Phase 1

    **Forces.** CNN ~271 k paramètres, **accuracy test 0,944** ; architecture compacte grâce au partage de poids, feature maps cohérentes (détecteurs de bords).

    **Limites.** Labels issus de la *commande* de l'agent (bruit quand on tourne sur une ligne droite) ; split aléatoire sur frames corrélées (métriques optimistes) ; validation petite et bruitée.

    **Améliorations.** Split temporel (Phase 2), data augmentation, plus de frames, labels basés sur la géométrie réelle, `weight_decay` contre le léger surapprentissage.

    **Transition.** Le CNN sait *reconnaître* la route mais pas *quelle action choisir* → objet de la Phase 2 (régression `[steering, gas, brake]`).
    """)
    return


@app.cell
def _(F, nn, torch):
    # Variante SANS BatchNorm
    class PerceptionCNN_noBN(nn.Module):
        def __init__(self, n_classes=3, dropout=0.3):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 8, 4)
            self.conv2 = nn.Conv2d(8, 16, 4, 2)
            self.conv3 = nn.Conv2d(16, 32, 3, 1)
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, n_classes)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    return (PerceptionCNN_noBN,)


@app.cell
def _(mo):
    ablation_button = mo.ui.run_button(label="▶ Comparer avec / sans BatchNorm")
    ablation_button
    return (ablation_button,)


@app.cell
def _(
    DEVICE,
    PerceptionCNN_noBN,
    ablation_button,
    history,
    mo,
    plt,
    train_cnn,
    train_ld,
    val_ld,
):
    mo.stop(not ablation_button.value, mo.md("*Clique pour lancer la comparaison.*"))

    model_nobn = PerceptionCNN_noBN(n_classes=3).to(DEVICE)
    hist_nobn  = train_cnn(model_nobn, train_ld, val_ld, n_epochs=20)

    fig_cmp, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _e = range(1, len(history["va"]) + 1)
    _ax[0].plot(_e, history["vl"], label="avec BN"); _ax[0].plot(_e, hist_nobn["vl"], label="sans BN")
    _ax[0].set(title="Val Loss", xlabel="Epoch"); _ax[0].legend(); _ax[0].grid(alpha=0.3)
    _ax[1].plot(_e, history["va"], label="avec BN"); _ax[1].plot(_e, hist_nobn["va"], label="sans BN")
    _ax[1].set(title="Val Accuracy", xlabel="Epoch"); _ax[1].legend(); _ax[1].grid(alpha=0.3)
    fig_cmp.suptitle("Impact de la BatchNorm")
    fig_cmp.tight_layout()
    fig_cmp
    return (hist_nobn,)


@app.cell
def _(hist_nobn, history, plt):
    def _():
        fig_cmp, (_axb, _axn) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        _e = range(1, len(history["va"]) + 1)

        # Avec BN : train lisse, val qui saute → l'écart révèle le décalage train/eval
        _axb.plot(_e, history["ta"], label="train")
        _axb.plot(_e, history["va"], label="val")
        _axb.set(title="Avec BatchNorm", xlabel="Epoch", ylabel="Accuracy")
        _axb.legend(); _axb.grid(alpha=0.3)

        # Sans BN : train et val se suivent de près (pas de décalage train/eval)
        _axn.plot(_e, hist_nobn["ta"], label="train")
        _axn.plot(_e, hist_nobn["va"], label="val")
        _axn.set(title="Sans BatchNorm", xlabel="Epoch")
        _axn.legend(); _axn.grid(alpha=0.3)

        fig_cmp.suptitle("Train vs Val — l'écart révèle le décalage train/eval de la BatchNorm")
        fig_cmp.tight_layout()
        return fig_cmp


    _()
    return


@app.cell
def _():
    # _class_names = {0: "tout droit", 1: "gauche", 2: "droite"}

    # # 8 frames du test set, tirées au hasard (graine fixée pour la repro)
    # _rng = np.random.default_rng(SEED)
    # _idx = _rng.choice(len(test_ds), size=8, replace=False)

    # _imgs = torch.stack([test_ds[int(i)][0] for i in _idx])   # (8, 3, 96, 96)
    # _true = [int(test_ds[int(i)][1]) for i in _idx]

    # model.eval()
    # with torch.no_grad():
    #     _logits = model(_imgs.to(DEVICE))
    #     _probs  = torch.softmax(_logits, dim=1).cpu()         # (8, 3) probabilités
    # _conf, _pred = _probs.max(dim=1)                           # confiance + classe prédite

    # fig_pred, _axes = plt.subplots(2, 4, figsize=(14, 7))
    # for _k, _ax in enumerate(_axes.ravel()):
    #     _ax.imshow(_imgs[_k].permute(1, 2, 0).numpy())         # (C,H,W) -> (H,W,C)
    #     _p   = int(_pred[_k])
    #     _pct = float(_conf[_k]) * 100
    #     _ok  = (_p == _true[_k])
    #     _ax.set_title(
    #         f"prédit : {_class_names[_p]}  ({_pct:.1f}%)\n"
    #         f"réel : {_class_names[_true[_k]]}",
    #         color="green" if _ok else "red", fontsize=10,
    #     )
    #     _ax.axis("off")

    # fig_pred.suptitle("Prédictions du CNN sur 8 frames du test set", fontsize=13)
    # fig_pred.tight_layout()
    # fig_pred
    return


@app.cell
def _(DEVICE, SEED, model, np, plt, test_ds, torch):
    _class_names = {0: "tout droit", 1: "gauche", 2: "droite"}

    # 100 frames du test set, tirées au hasard (graine fixée pour la repro)
    _rng = np.random.default_rng(SEED)
    _idx = _rng.choice(len(test_ds), size=100, replace=False)

    _imgs = torch.stack([test_ds[int(i)][0] for i in _idx])   # (100, 3, 96, 96)
    _true = [int(test_ds[int(i)][1]) for i in _idx]

    model.eval()
    with torch.no_grad():
        _logits = model(_imgs.to(DEVICE))
        _probs  = torch.softmax(_logits, dim=1).cpu()         # (100, 3) probabilités
    _conf, _pred = _probs.max(dim=1)                           # confiance + classe prédite

    # Grille 10x10 pour 100 images
    fig_pred, _axes = plt.subplots(10, 10, figsize=(20, 20))
    for _k, _ax in enumerate(_axes.ravel()):
        _ax.imshow(_imgs[_k].permute(1, 2, 0).numpy())         # (C,H,W) -> (H,W,C)
        _p   = int(_pred[_k])
        _pct = float(_conf[_k]) * 100
        _ok  = (_p == _true[_k])
        _ax.set_title(
            f"prédit : {_class_names[_p]}  ({_pct:.1f}%)\n"
            f"réel : {_class_names[_true[_k]]}",
            color="green" if _ok else "red", fontsize=8,  # Réduire la taille de la police
        )
        _ax.axis("off")

    fig_pred.suptitle("Prédictions du CNN sur 100 frames du test set", fontsize=16)
    fig_pred.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
