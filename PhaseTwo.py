import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import glob, os
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"polars {pl.__version__} | torch {torch.__version__} | device : {DEVICE}")
    return (
        DEVICE,
        DataLoader,
        Dataset,
        SEED,
        glob,
        mo,
        nn,
        np,
        os,
        pl,
        plt,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Phase 2 — Imitation d'un pilote (apprentissage supervisé)

    Objectif : prédire les actions **[steering, gas, brake]** à partir des features résumées
    **[R_moy, G_moy, B_moy, road_offset]**, en imitant un pilote rule-based dont les
    trajectoires sont fournies en CSV.

    Deux ruptures par rapport à la Phase 1 :
    - **Classification -> régression** : la cible n'est plus une classe mais 3 valeurs continues
      -> loss `MSELoss` (et plus `CrossEntropyLoss`), pas de softmax final.
    - **Le split devient le point critique** : sur des données de conduite (série temporelle),
      un split aléatoire crée du **data leakage**. On splitte **par épisode entier**.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(glob, os, pl):
    # Localise le dossier de données, que le notebook soit à la racine ou dans notebooks/
    _candidates = ["data/demo_episodes", "../data/demo_episodes"]
    DATA_DIR = next((d for d in _candidates if os.path.isdir(d)), _candidates[0])

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.csv")))
    dfs = [pl.read_csv(f) for f in csv_files]
    print(f"{len(csv_files)} épisodes trouvés dans '{DATA_DIR}'\n")

    # Récap par épisode : nb de steps + récompense totale
    for _f, _df in zip(csv_files, dfs):
        print(f"{os.path.basename(_f):18s} | {_df.height:4d} steps | "
              f"reward total = {_df['reward'].sum():8.1f}")

    # Concaténation + stats descriptives des actions
    df_all = pl.concat(dfs)
    print("\nColonnes :", df_all.columns)
    print(df_all.select(["action_steering", "action_gas", "action_brake"]).describe())
    return df_all, dfs


@app.cell
def _(df_all, plt):
    fig_dist, axes_dist = plt.subplots(1, 3, figsize=(13, 3.5))
    for _ax, _col, _title in zip(
        axes_dist,
        ["action_steering", "action_gas", "action_brake"],
        ["Steering", "Gas", "Brake"],
    ):
        _ax.hist(df_all[_col].to_numpy(), bins=40, color="steelblue", alpha=0.8)
        _ax.set(title=_title, xlabel="valeur", ylabel="fréquence")
        _ax.grid(alpha=0.3)
    fig_dist.suptitle("Distribution des 3 actions (tous épisodes confondus)")
    fig_dist.tight_layout()
    fig_dist
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### La distribution du steering est-elle équilibrée ?

    **Non.** Le steering est très **piqué autour de 0** (le pilote roule surtout tout droit),
    le gas est majoritairement élevé, et le brake quasi toujours nul. C'est une distribution
    **fortement asymétrique**.

    Conséquences pour l'entraînement :
    - Le modèle peut minimiser la MSE en prédisant **presque toujours ~0** pour le steering :
      il « gagne » sur la majorité (lignes droites) et néglige les virages, pourtant les
      moments qui comptent pour conduire.
    - Les métriques globales (RMSE) seront **dominées par la classe majoritaire** et donneront
      une illusion de bonne performance. D'où l'intérêt d'ajouter la **précision directionnelle**
      (cellule d'éval plus loin) et, éventuellement, une pondération des virages.
    """)
    return


@app.cell
def _(dfs, pl):
    # Split temporel par épisode 
    # Épisodes 0 et 1 -> train  |  épisode 2 coupé en deux -> val puis test
    # (On garde des épisodes INTACTS : aucune frame voisine ne traverse la frontière)
    df_ep2 = dfs[2]
    mid    = df_ep2.height // 2

    train_df = pl.concat(dfs[:2])      # épisodes 0 + 1
    val_df   = df_ep2.slice(0, mid)    # 1ère moitié de l'épisode 2
    test_df  = df_ep2.slice(mid)       # 2ème moitié de l'épisode 2

    print(f"Train : {train_df.height:4d} steps  (épisodes 0-1)")
    print(f"Val   : {val_df.height:4d} steps  (ep 2, 1ère moitié)")
    print(f"Test  : {test_df.height:4d} steps  (ep 2, 2ème moitié)")
    return test_df, train_df, val_df


@app.cell
def _(mo):
    mo.md(r"""
    ### Pourquoi un split par épisode et pas aléatoire ?

    Sur des images indépendantes (chats/chiens) on peut mélanger avant de splitter.
    Ici, **les frames consécutives sont fortement corrélées** : la frame `n+1` ressemble
    presque parfaitement à la frame `n` (le `road_offset`, les couleurs, l'action évoluent
    continûment).

    Si on mélange *avant* de splitter, des frames quasi-jumelles `n` et `n+1` se retrouvent
    l'une en **train**, l'autre en **test** -> le modèle « reconnaît » des voisines déjà vues
    et les métriques de test deviennent **artificiellement bonnes** : c'est du **data leakage**.

    En splittant **par épisode entier** (train = ep 0-1, val/test = moitiés de l'ep 2), aucune
    frame voisine ne traverse la frontière : le test mesure une vraie généralisation à une
    portion de circuit jamais vue. On le **prouvera empiriquement** plus loin en comparant
    split aléatoire vs split temporel sur les mêmes métriques.
    """)
    return


@app.cell
def _(np, plt):
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_pred - y_true)))

    def directional_accuracy(y_true, y_pred):
        # Tourne-t-on dans le bon sens ? (gauche / tout droit / droite)
        return float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    def print_regression_report(y_true, y_pred, label="cible"):
        print(f"── Rapport régression — {label} ──")
        print(f"  RMSE                  : {rmse(y_true, y_pred):.4f}")
        print(f"  MAE                   : {mae(y_true, y_pred):.4f}")
        print(f"  Précision directionn. : {directional_accuracy(y_true, y_pred)*100:.1f}%")

    def plot_training_curves(tl, vl, title="Entraînement", ylabel="Loss"):
        fig, ax = plt.subplots(figsize=(7, 4))
        _e = range(1, len(tl) + 1)
        ax.plot(_e, tl, label="train")
        ax.plot(_e, vl, label="val")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.legend(); ax.grid(alpha=0.3)
        # Diagnostic auto (surapprentissage / sous-apprentissage)
        _gap = vl[-1] - tl[-1]
        if   _gap > 0.5 * tl[-1] and tl[-1] > 0: print(f"-> val >> train (gap {_gap:.4f}) : surapprentissage probable.")
        elif tl[-1] > 0.9 * tl[0]:               print("-> train ne baisse presque pas : sous-apprentissage probable.")
        else:                                    print("-> train et val baissent ensemble : convergence correcte")
        fig.tight_layout()
        return fig

    def plot_predictions_vs_truth(y_true, y_pred, label="cible", n=200):
        fig, ax = plt.subplots(figsize=(11, 3.5))
        _n = min(n, len(y_true))
        ax.plot(y_true[:_n], label="réel", color="steelblue", lw=1.5)
        ax.plot(y_pred[:_n], label="prédit", color="orange", lw=1.2, alpha=0.85)
        ax.set(xlabel="Step (test)", ylabel=label, title=f"Prédictions vs réalité — {label}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    return (
        plot_predictions_vs_truth,
        plot_training_curves,
        print_regression_report,
        rmse,
    )


@app.cell
def _(DataLoader, Dataset, np, pl, test_df, torch, train_df, val_df):
    class DrivingDataset(Dataset):
        FEATURES = ["obs_mean_r", "obs_mean_g", "obs_mean_b", "road_offset"]
        TARGETS  = ["action_steering", "action_gas", "action_brake"]

        def __init__(self, df: pl.DataFrame):
            X = df.select(self.FEATURES).to_numpy().astype(np.float32)
            y = df.select(self.TARGETS).to_numpy().astype(np.float32)
            X[:, :3] /= 255.0          # normalise les couleurs 0-255 -> [0,1]
            # road_offset (colonne 3) est déjà dans [-1, 1] : on n'y touche pas
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)

        def __len__(self):  return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]


    train_ds = DrivingDataset(train_df)
    val_ds   = DrivingDataset(val_df)
    test_ds  = DrivingDataset(test_df)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)   # shuffle OK : le split est déjà étanche
    val_ld   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    print(f"Shape X : {tuple(train_ds.X.shape)}  — 4 features par step")
    print(f"Shape y : {tuple(train_ds.y.shape)}  — 3 actions à prédire")
    return DrivingDataset, test_ld, train_ld, val_ld


@app.cell
def _(DEVICE, nn, torch):
    class DrivingMLP(nn.Module):
        """
        MLP d'imitation. Entrée : 4 features -> Sortie : 3 actions (régression).
        Pas d'activation finale : MSELoss attend des valeurs réelles.
        """
        def __init__(self, input_dim=4, hidden=128, output_dim=3, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), 
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden),    
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 64),        
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            return self.net(x)


    # Vérification des dimensions
    _model = DrivingMLP().to(DEVICE)
    _dummy = torch.zeros(8, 4).to(DEVICE)
    with torch.no_grad():
        _out = _model(_dummy)
    print(f"Entrée {tuple(_dummy.shape)} -> Sortie {tuple(_out.shape)}")
    assert _out.shape == (8, 3), "Dimensions incorrectes !"
    print(f"Paramètres : {sum(p.numel() for p in _model.parameters()):,}")
    return (DrivingMLP,)


@app.cell
def _(DEVICE, nn, torch):
    def train_mlp(model, train_ld, val_ld, n_epochs=30, lr=1e-3,
                  tag="mlp", device=DEVICE, verbose=True):
        import copy
        crit  = nn.MSELoss()
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        history, best_vl, best_state = {"tl": [], "vl": []}, float("inf"), None

        for ep in range(1, n_epochs + 1):
            # ── train ──
            model.train()
            tl = 0.
            for xb, yb in train_ld:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()              # 1. reset gradients
                loss = crit(model(xb), yb)   # 2. forward + loss
                loss.backward()              # 3. rétropropagation
                opt.step()                   # 4. mise à jour des poids
                tl += loss.item()
            tl /= len(train_ld)

            # ── val ──
            model.eval()
            vl = 0.
            with torch.no_grad():
                for xb, yb in val_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    vl += crit(model(xb), yb).item()
            vl /= len(val_ld)
            sched.step(vl)

            history["tl"].append(tl); history["vl"].append(vl)
            if vl < best_vl:
                best_vl, best_state = vl, copy.deepcopy(model.state_dict())
            if verbose and ep % 10 == 0:
                print(f"Ep {ep:3d} | Train={tl:.5f} | Val={vl:.5f} "
                      f"| LR={opt.param_groups[0]['lr']:.6f}")

        model.load_state_dict(best_state)    # restaure le meilleur (val_loss min)
        return history

    return (train_mlp,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="▶ Entraîner le DrivingMLP")
    train_button
    return (train_button,)


@app.cell
def _(DEVICE, DrivingMLP, mo, train_button, train_ld, train_mlp, val_ld):
    mo.stop(not train_button.value, mo.md("*Clique sur le bouton ci-dessus pour entraîner (~quelques secondes).*"))

    model = DrivingMLP().to(DEVICE)
    hist  = train_mlp(model, train_ld, val_ld, n_epochs=30)
    return hist, model


@app.cell
def _(hist, plot_training_curves):
    plot_training_curves(hist["tl"], hist["vl"], title="DrivingMLP — Imitation Learning", ylabel="MSE Loss")
    return


@app.cell
def _(
    DEVICE,
    model,
    np,
    plot_predictions_vs_truth,
    print_regression_report,
    test_ld,
    torch,
):
    model.eval()
    _ys, _preds = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            _preds.append(model(xb.to(DEVICE)).cpu().numpy())
            _ys.append(yb.numpy())

    y_true = np.vstack(_ys)    # (N, 3)
    y_pred = np.vstack(_preds) # (N, 3)

    for _i, _a in enumerate(["Steering", "Gas", "Brake"]):
        print_regression_report(y_true[:, _i], y_pred[:, _i], label=_a)
        print()

    plot_predictions_vs_truth(y_true[:, 0], y_pred[:, 0], label="Steering")
    return y_pred, y_true


@app.cell
def _(np, rmse, train_df, y_pred, y_true):
    y_train_mean = (train_df
                    .select(["action_steering", "action_gas", "action_brake"])
                    .to_numpy().mean(axis=0))
    y_baseline = np.tile(y_train_mean, (len(y_true), 1))

    print("=== Baseline naïve (moyenne train) vs DrivingMLP — RMSE ===")
    for _i, _a in enumerate(["Steering", "Gas", "Brake"]):
        _b = rmse(y_true[:, _i], y_baseline[:, _i])
        _m = rmse(y_true[:, _i], y_pred[:, _i])
        _g = (_b - _m) / _b * 100 if _b > 0 else 0.0
        print(f"{_a:9s} | Baseline={_b:.4f} | MLP={_m:.4f} | Gain={_g:+.1f}%")
    return


@app.cell
def _(mo):
    lr_button = mo.ui.run_button(label="▶ Comparer 3 learning rates")
    lr_button
    return (lr_button,)


@app.cell
def _(
    DEVICE,
    DrivingMLP,
    SEED,
    lr_button,
    mo,
    plt,
    torch,
    train_ld,
    train_mlp,
    val_ld,
):
    mo.stop(not lr_button.value, mo.md("*Clique pour comparer lr ∈ {1e-2, 1e-3, 1e-4}.*"))

    results_lr = {}
    fig_lr, ax_lr = plt.subplots(figsize=(7, 4))
    for _lr in [1e-2, 1e-3, 1e-4]:
        torch.manual_seed(SEED)                       # même init pour comparer équitablement
        _m = DrivingMLP().to(DEVICE)
        _h = train_mlp(_m, train_ld, val_ld, n_epochs=20, lr=_lr, verbose=False)
        results_lr[f"lr={_lr:.0e}"] = min(_h["vl"])
        ax_lr.plot(range(1, len(_h["vl"]) + 1), _h["vl"], label=f"lr={_lr:.0e}")
        print(f"lr={_lr:.0e} → val_loss min = {min(_h['vl']):.5f}")

    ax_lr.set(xlabel="Epoch", ylabel="Val MSE", title="Impact du learning rate")
    ax_lr.legend(); ax_lr.grid(alpha=0.3)
    fig_lr.tight_layout()
    fig_lr
    return


@app.cell
def _(mo):
    leak_button = mo.ui.run_button(label="▶ Démo data leakage : split aléatoire vs temporel")
    leak_button
    return (leak_button,)


@app.cell
def _(
    DEVICE,
    DataLoader,
    DrivingDataset,
    DrivingMLP,
    SEED,
    df_all,
    leak_button,
    mo,
    np,
    rmse,
    torch,
    train_mlp,
    y_pred,
    y_true,
):
    def _():
        mo.stop(not leak_button.value, mo.md("*Clique pour comparer les deux stratégies de split sur le test set.*"))

        # ── SPLIT ALÉATOIRE : on mélange TOUTES les frames avant de splitter (60/20/20) ──
        _n    = df_all.height
        _perm = np.random.default_rng(SEED).permutation(_n).tolist()
        _a, _b = int(0.6 * _n), int(0.8 * _n)

        _rand_tr = df_all[_perm[:_a]]
        _rand_va = df_all[_perm[_a:_b]]
        _rand_te = df_all[_perm[_b:]]

        _tr_ld = DataLoader(DrivingDataset(_rand_tr), batch_size=64, shuffle=True)
        _va_ld = DataLoader(DrivingDataset(_rand_va), batch_size=64, shuffle=False)
        _te_ld = DataLoader(DrivingDataset(_rand_te), batch_size=64, shuffle=False)

        torch.manual_seed(SEED)
        _m_rand = DrivingMLP().to(DEVICE)
        train_mlp(_m_rand, _tr_ld, _va_ld, n_epochs=30, lr=1e-3, verbose=False)

        _m_rand.eval()
        _yt, _yp = [], []
        with torch.no_grad():
            for xb, yb in _te_ld:
                _yp.append(_m_rand(xb.to(DEVICE)).cpu().numpy())
                _yt.append(yb.numpy())
        _yt, _yp = np.vstack(_yt), np.vstack(_yp)

        _rmse_rand = rmse(_yt[:, 0], _yp[:, 0])           # split aléatoire
        _rmse_temp = rmse(y_true[:, 0], y_pred[:, 0])     # split temporel (modèle 'model')

        print("── RMSE steering sur le test set ──")
        print(f"Split ALÉATOIRE : {_rmse_rand:.4f}   ← optimiste : des frames voisines fuient en train")
        print(f"Split TEMPOREL  : {_rmse_temp:.4f}   ← honnête : portion de circuit jamais vue")
        return print(f"\n→ Le split aléatoire 'gagne' artificiellement {(_rmse_temp - _rmse_rand)/_rmse_temp*100:+.0f}% : c'est le data leakage.")


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Questions de réflexion — Phase 2

    **1. Le modèle bat-il la baseline ? Sur quelle action est-il le moins bon ?**
    Le MLP améliore nettement la baseline sur le **steering** (la baseline « moyenne » ≈ 0 ne
    sait pas tourner). Le gain est **plus faible sur gas/brake** : ces actions sont quasi
    constantes dans les données (gas ~élevé, brake ~0), donc prédire la moyenne est déjà une
    bonne approximation — il reste peu de marge à gagner. *(Reporte ici tes % de la cellule 16.)*

    **2. Steering asymétrique (beaucoup de « tout droit ») — effet sur les métriques ?**
    La majorité des steps sont des lignes droites (steering ≈ 0). La RMSE globale est donc
    **dominée par cette classe facile** : un modèle qui prédit toujours ~0 obtient une RMSE
    faible tout en ratant les virages. Solutions : **pondérer** les samples de virage dans la
    loss, **sur-échantillonner** les virages, ou évaluer séparément la performance en virage.

    **3. RMSE vs précision directionnelle — cohérents ?**
    Pas toujours. Un modèle peut avoir une **bonne RMSE** (bonne amplitude moyenne) mais une
    **précision directionnelle médiocre** s'il se trompe parfois de **sens** sur les petits
    virages. Pour un pilote, le **sens** prime sur l'amplitude : la précision directionnelle est
    donc la métrique la plus parlante. *(Compare tes deux valeurs pour le steering et conclus.)*

    **Pourquoi val < train ?** Le Dropout n'est actif qu'à l'entraînement (il dégrade la train
    loss), et la 1ère moitié de l'épisode 2 (val) est plus régulière. Aucun signe de
    surapprentissage.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
