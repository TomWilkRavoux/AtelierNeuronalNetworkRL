import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import glob, os, time
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 10   # longueur de la fenêtre temporelle on la fera varier plus loin
    print(f"polars {pl.__version__} | torch {torch.__version__} | device : {DEVICE} | SEQ_LEN={SEQ_LEN}")
    return (
        DEVICE,
        DataLoader,
        Dataset,
        SEED,
        SEQ_LEN,
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
    # Phase 3 — Mémoire temporelle (LSTM / GRU)

    Le `DrivingMLP` de la Phase 2 prédit `action(t)` à partir de la seule `observation(t)` :
    **aucune mémoire**. Sur un virage qui s'accentue, il réagit trop tard.

    Ici on donne au modèle une **séquence des T derniers pas** (fenêtre glissante) et on le laisse
    mémoriser le passé via un **LSTM** (3 portes + cellule mémoire) ou un **GRU** (2 portes, plus léger).
    On comparera les deux, puis on les confrontera au MLP.

    Le split reste **par épisode** : avec des fenêtres glissantes, un split ligne-par-ligne
    ferait fuiter des séquences chevauchantes entre train et test.
    """)
    return


@app.cell
def _(glob, np, os, pl, plt):
    # Métriques (mêmes qu'en Phase 2, ré-incluses car nouveau notebook)
    def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    def mae(y_true, y_pred):  return float(np.mean(np.abs(y_pred - y_true)))
    def directional_accuracy(y_true, y_pred):
        return float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    def print_regression_report(y_true, y_pred, label="cible"):
        print(f"── {label} ──  RMSE={rmse(y_true,y_pred):.4f} | MAE={mae(y_true,y_pred):.4f} "
              f"| dir.={directional_accuracy(y_true,y_pred)*100:.1f}%")
    def plot_training_curves(tl, vl, title="Entraînement", ylabel="MSE Loss"):
        fig, ax = plt.subplots(figsize=(7, 4)); _e = range(1, len(tl)+1)
        ax.plot(_e, tl, label="train"); ax.plot(_e, vl, label="val")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); return fig

    # ── Chargement (glob récursif, comme en Phase 2) ──
    _hits = sorted(set(glob.glob("**/episode_*.csv", recursive=True))
                   | set(glob.glob("../**/episode_*.csv", recursive=True)))
    DATA_DIR = os.path.dirname(_hits[0])
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.csv")))
    dfs = [pl.read_csv(f) for f in csv_files]
    print(f"{len(dfs)} épisodes chargés depuis '{DATA_DIR}'")

    # ── Auto-corrélation du steering (justifie le LSTM) ──
    def _autocorr(x, lag):
        x = x - x.mean()
        return float(np.sum(x[:-lag] * x[lag:]) / np.sum(x * x))

    _steer = dfs[0]["action_steering"].to_numpy()
    _lags  = range(1, 31)
    _corrs = [_autocorr(_steer, l) for l in _lags]

    fig_ac, _ax = plt.subplots(1, 2, figsize=(13, 4))
    _ax[0].plot(_steer[:300], lw=1, color="steelblue")
    _ax[0].set(xlabel="Step", ylabel="Steering", title="Série temporelle du steering (ep 0)")
    _ax[0].grid(alpha=0.3)
    _ax[1].bar(list(_lags), _corrs, color="steelblue", alpha=0.8)
    _ax[1].axhline(0, color="gray", lw=0.5)
    _ax[1].set(xlabel="Lag (steps)", ylabel="Corrélation", title="Auto-corrélation du steering")
    _ax[1].grid(alpha=0.3)
    fig_ac.tight_layout()

    for l in (1, 5, 10):
        print(f"Auto-corrélation à lag={l:2d} : {_autocorr(_steer, l):.3f}")
    fig_ac
    return dfs, directional_accuracy, print_regression_report, rmse


@app.cell
def _(mo):
    mo.md(r"""
    ### L'auto-corrélation justifie-t-elle le LSTM ?

    **Oui.** La corrélation à **lag=1** est très élevée (proche de 1) puis décroît lentement :
    `steering(t)` prédit fortement `steering(t+1)`, et l'information utile s'étale sur plusieurs
    pas. Le passé récent est donc **informatif** — un modèle qui le mémorise (LSTM/GRU) peut
    **anticiper** au lieu de réagir.

    Lien avec `seq_len` : tant que la corrélation à lag=T reste non négligeable, allonger la
    fenêtre apporte du signal. Quand elle s'effondre, les pas trop anciens ne sont plus que du
    bruit → inutile (voire nuisible) d'allonger encore. C'est ce qu'on testera en faisant varier
    `seq_len`.
    """)
    return


@app.cell
def _(DataLoader, Dataset, SEQ_LEN, dfs, np, pl, torch):
    class SequenceDataset(Dataset):
        """Transforme une série temporelle en fenêtres glissantes.
           X : (N, T, 4)  — historique  |  y : (N, 3) — action au pas T+1."""
        FEATURES = ["obs_mean_r", "obs_mean_g", "obs_mean_b", "road_offset"]
        TARGETS  = ["action_steering", "action_gas", "action_brake"]

        def __init__(self, df: pl.DataFrame, seq_len: int = 10):
            self.seq_len = seq_len
            X_raw = df.select(self.FEATURES).to_numpy().astype(np.float32)
            y_raw = df.select(self.TARGETS).to_numpy().astype(np.float32)
            X_raw[:, :3] /= 255.0                       # normalise les couleurs

            seqs, targets = [], []
            for i in range(len(X_raw) - seq_len):
                seqs.append(X_raw[i : i + seq_len])     # (T, 4)
                targets.append(y_raw[i + seq_len])      # action au pas suivant
            self.X = torch.tensor(np.array(seqs))
            self.y = torch.tensor(np.array(targets))

        def __len__(self):  return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]


    # Split par épisode — identique à la Phase 2
    df_ep2 = dfs[2]
    mid    = df_ep2.height // 2

    train_ds = SequenceDataset(pl.concat(dfs[:2]),     seq_len=SEQ_LEN)
    val_ds   = SequenceDataset(df_ep2.slice(0, mid),   seq_len=SEQ_LEN)
    test_ds  = SequenceDataset(df_ep2.slice(mid),      seq_len=SEQ_LEN)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    print(f"X shape : {tuple(train_ds.X.shape)}  → (N, T={SEQ_LEN}, features=4)")
    print(f"y shape : {tuple(train_ds.y.shape)}  → (N, 3 actions)")
    return SequenceDataset, df_ep2, mid, test_ld, train_ld, val_ld


@app.cell
def _(DEVICE, SEQ_LEN, nn, torch):
    class LSTMDriver(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=64, n_layers=2, output_dim=3, dropout=0.3):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
            self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.)
            self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, output_dim))
        def forward(self, x):
            batch = x.size(0)
            enc = self.encoder(x.view(-1, x.size(-1))).view(batch, -1, 32)
            _, (h_n, _) = self.lstm(enc)
            return self.head(h_n[-1])          # dernier état caché = résumé de la séquence


    class GRUDriver(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=64, n_layers=2, output_dim=3, dropout=0.3):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
            self.gru = nn.GRU(input_size=32, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.)
            self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, output_dim))
        def forward(self, x):
            batch = x.size(0)
            enc = self.encoder(x.view(-1, x.size(-1))).view(batch, -1, 32)
            _, h_n = self.gru(enc)             # GRU : pas de cellule mémoire
            return self.head(h_n[-1])


    # Vérification — ne pas modifier
    for _Cls, _name in [(LSTMDriver, "LSTM"), (GRUDriver, "GRU")]:
        _m = _Cls().to(DEVICE)
        _dummy = torch.zeros(8, SEQ_LEN, 4).to(DEVICE)
        with torch.no_grad():
            _out = _m(_dummy)
        _p = sum(p.numel() for p in _m.parameters())
        print(f"{_name} | Entrée {tuple(_dummy.shape)} → Sortie {tuple(_out.shape)} | {_p:,} paramètres")
        assert _out.shape == (8, 3), f"{_name} : dimensions incorrectes !"
    return GRUDriver, LSTMDriver


@app.cell
def _(DEVICE, nn, np, plt, torch):
    def train_seq_model(model, train_ld, val_ld, n_epochs=25, lr=1e-3, clip=1.0,
                        device=DEVICE, verbose=True):
        import copy
        crit  = nn.MSELoss()
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        history, best_vl, best_state = {"tl": [], "vl": []}, float("inf"), None

        for ep in range(1, n_epochs + 1):
            model.train(); tl = 0.
            for xb, yb in train_ld:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)   # anti-explosion (RNN)
                opt.step()
                tl += loss.item()
            tl /= len(train_ld)

            model.eval(); vl = 0.
            with torch.no_grad():
                for xb, yb in val_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    vl += crit(model(xb), yb).item()
            vl /= len(val_ld)
            sched.step(vl)

            history["tl"].append(tl); history["vl"].append(vl)
            if vl < best_vl: best_vl, best_state = vl, copy.deepcopy(model.state_dict())
            if verbose and ep % 5 == 0:
                print(f"Ep {ep:3d} | Train={tl:.5f} | Val={vl:.5f}")

        model.load_state_dict(best_state)
        return history


    def evaluate_seq(model, loader, device=DEVICE):
        model.eval(); ys, preds = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds.append(model(xb.to(device)).cpu().numpy()); ys.append(yb.numpy())
        return np.vstack(ys), np.vstack(preds)


    def compare_models(results, metrics=("rmse", "dir_acc", "val_loss")):
        names = list(results.keys())
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
        for ax, met in zip(axes, metrics):
            vals = [results[n][met] for n in names]
            ax.bar(names, vals, color="steelblue", alpha=0.8)
            ax.set(title=met); ax.grid(alpha=0.3, axis="y")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout(); return fig

    return compare_models, evaluate_seq, train_seq_model


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="▶ Entraîner LSTM + GRU")
    train_button
    return (train_button,)


@app.cell
def _(
    DEVICE,
    GRUDriver,
    LSTMDriver,
    SEED,
    mo,
    torch,
    train_button,
    train_ld,
    train_seq_model,
    val_ld,
):
    mo.stop(not train_button.value, mo.md("*Clique pour entraîner les deux modèles séquentiels.*"))

    print("=== LSTMDriver ===")
    torch.manual_seed(SEED)
    lstm_model = LSTMDriver().to(DEVICE)
    hist_lstm  = train_seq_model(lstm_model, train_ld, val_ld)

    print("\n=== GRUDriver ===")
    torch.manual_seed(SEED)
    gru_model = GRUDriver().to(DEVICE)
    hist_gru  = train_seq_model(gru_model, train_ld, val_ld)
    return gru_model, hist_gru, hist_lstm, lstm_model


@app.cell
def _(hist_gru, hist_lstm, plt):
    fig_lg, ax_lg = plt.subplots(figsize=(7, 4))
    ax_lg.plot(range(1, len(hist_lstm["vl"]) + 1), hist_lstm["vl"], label="LSTM")
    ax_lg.plot(range(1, len(hist_gru["vl"]) + 1),  hist_gru["vl"],  label="GRU")
    ax_lg.set(xlabel="Epoch", ylabel="Val MSE", title="LSTM vs GRU — val loss")
    ax_lg.legend(); ax_lg.grid(alpha=0.3)
    fig_lg.tight_layout()
    fig_lg
    return


@app.cell
def _(
    compare_models,
    directional_accuracy,
    evaluate_seq,
    gru_model,
    hist_gru,
    hist_lstm,
    lstm_model,
    print_regression_report,
    rmse,
    test_ld,
):
    y_t_lstm, y_p_lstm = evaluate_seq(lstm_model, test_ld)
    y_t_gru,  y_p_gru  = evaluate_seq(gru_model,  test_ld)

    print_regression_report(y_t_lstm[:, 0], y_p_lstm[:, 0], "LSTM — Steering")
    print_regression_report(y_t_gru[:,  0], y_p_gru[:,  0], "GRU  — Steering")

    results = {
        "LSTM": {"rmse": rmse(y_t_lstm[:, 0], y_p_lstm[:, 0]),
                 "dir_acc": directional_accuracy(y_t_lstm[:, 0], y_p_lstm[:, 0]),
                 "val_loss": min(hist_lstm["vl"])},
        "GRU":  {"rmse": rmse(y_t_gru[:, 0], y_p_gru[:, 0]),
                 "dir_acc": directional_accuracy(y_t_gru[:, 0], y_p_gru[:, 0]),
                 "val_loss": min(hist_gru["vl"])},
    }
    compare_models(results)
    return results, y_p_gru, y_p_lstm, y_t_gru, y_t_lstm


@app.cell
def _(
    DEVICE,
    SEED,
    compare_models,
    directional_accuracy,
    evaluate_seq,
    mo,
    nn,
    print_regression_report,
    results,
    rmse,
    test_ld,
    torch,
    train_button,
    train_ld,
    train_seq_model,
    val_ld,
    y_p_gru,
    y_p_lstm,
    y_t_gru,
    y_t_lstm,
):
    # mo.stop(not train_button.value, mo.md("*Entraîne d'abord les modèles séquentiels (cellule 8).*"))

    # # MLP frame-par-frame (pas de mémoire), même split, pour comparer
    # class _MLP(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.Linear(4, 128), nn.ReLU(), nn.Dropout(0.3),
    #             nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
    #             nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
    #     def forward(self, x): return self.net(x)

    # class _FlatDS(Dataset):
    #     def __init__(self, df):
    #         X = df.select(["obs_mean_r","obs_mean_g","obs_mean_b","road_offset"]).to_numpy().astype(np.float32)
    #         y = df.select(["action_steering","action_gas","action_brake"]).to_numpy().astype(np.float32)
    #         X[:, :3] /= 255.0
    #         self.X, self.y = torch.tensor(X), torch.tensor(y)
    #     def __len__(self): return len(self.y)
    #     def __getitem__(self, i): return self.X[i], self.y[i]

    # _mlp_tr = DataLoader(_FlatDS(pl.concat(dfs[:2])),   batch_size=64, shuffle=True)
    # _mlp_va = DataLoader(_FlatDS(df_ep2.slice(0, mid)), batch_size=64, shuffle=False)
    # _mlp_te = DataLoader(_FlatDS(df_ep2.slice(mid)),    batch_size=64, shuffle=False)

    # torch.manual_seed(SEED)
    # _mlp = _MLP().to(DEVICE)
    # _h   = train_seq_model(_mlp, _mlp_tr, _mlp_va, n_epochs=30, verbose=False)  # clip inoffensif ici
    # _yt, _yp = evaluate_seq(_mlp, _mlp_te)

    # results_all = dict(results)
    # results_all["MLP"] = {"rmse": rmse(_yt[:, 0], _yp[:, 0]),
    #                       "dir_acc": directional_accuracy(_yt[:, 0], _yp[:, 0]),
    #                       "val_loss": min(_h["vl"])}
    # print("── RMSE steering (test) ──")
    # for _n, _r in results_all.items():
    #     print(f"{_n:5s} | RMSE={_r['rmse']:.4f} | dir.={_r['dir_acc']*100:.1f}% | val_loss={_r['val_loss']:.5f}")
    # compare_models(results_all)

    mo.stop(not train_button.value, mo.md("*Entraîne d'abord les modèles séquentiels (cellule 8).*"))

    # MLP équitable : MÊME cible next-step que le LSTM, mais ne voit que la dernière frame
    class _LastFrameMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        def forward(self, x):           # x : (B, T, 4)
            return self.net(x[:, -1, :])  # ne garde que le dernier pas observé

    torch.manual_seed(SEED)
    _lf = _LastFrameMLP().to(DEVICE)
    _h  = train_seq_model(_lf, train_ld, val_ld, n_epochs=30, verbose=False)  # mêmes loaders séquentiels
    _yt, _yp = evaluate_seq(_lf, test_ld)

    print("── Comparaison ÉQUITABLE (même tâche next-step, mêmes données) ──")
    print_regression_report(y_t_lstm[:, 0], y_p_lstm[:, 0], "LSTM (T frames)        ")
    print_regression_report(y_t_gru[:,  0], y_p_gru[:,  0], "GRU  (T frames)        ")
    print_regression_report(_yt[:, 0],      _yp[:, 0],      "MLP  (dernière frame)  ")

    results_fair = {
        "LSTM": results["LSTM"],
        "GRU":  results["GRU"],
        "MLP_1frame": {"rmse": rmse(_yt[:, 0], _yp[:, 0]),
                       "dir_acc": directional_accuracy(_yt[:, 0], _yp[:, 0]),
                       "val_loss": min(_h["vl"])},
    }
    compare_models(results_fair)
    return


@app.cell
def _(mo):
    seqlen_button = mo.ui.run_button(label="▶ Balayer seq_len ∈ {5, 10, 20}")
    seqlen_button
    return (seqlen_button,)


@app.cell
def _(
    DEVICE,
    DataLoader,
    LSTMDriver,
    SEED,
    SequenceDataset,
    df_ep2,
    dfs,
    evaluate_seq,
    mid,
    mo,
    pl,
    plt,
    rmse,
    seqlen_button,
    torch,
    train_seq_model,
):
    mo.stop(not seqlen_button.value, mo.md("*Clique pour entraîner un LSTM par valeur de seq_len.*"))

    rmse_by_seq = {}
    for _T in [5, 10, 20]:
        _tr = DataLoader(SequenceDataset(pl.concat(dfs[:2]),   seq_len=_T), batch_size=64, shuffle=True)
        _va = DataLoader(SequenceDataset(df_ep2.slice(0, mid), seq_len=_T), batch_size=64, shuffle=False)
        _te = DataLoader(SequenceDataset(df_ep2.slice(mid),    seq_len=_T), batch_size=64, shuffle=False)
        torch.manual_seed(SEED)
        _m = LSTMDriver().to(DEVICE)
        train_seq_model(_m, _tr, _va, n_epochs=25, verbose=False)
        _yt, _yp = evaluate_seq(_m, _te)
        rmse_by_seq[_T] = rmse(_yt[:, 0], _yp[:, 0])
        print(f"seq_len={_T:2d} → RMSE steering = {rmse_by_seq[_T]:.4f}")

    fig_seq, ax_seq = plt.subplots(figsize=(6, 4))
    ax_seq.plot(list(rmse_by_seq.keys()), list(rmse_by_seq.values()), "o-", color="steelblue")
    ax_seq.set(xlabel="seq_len (T)", ylabel="RMSE steering (test)", title="Impact de seq_len")
    ax_seq.grid(alpha=0.3)
    fig_seq.tight_layout()
    fig_seq
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questions de réflexion — Phase 3

    **1. L'auto-corrélation justifie-t-elle le LSTM ?**
    Elle justifie de **l'essayer** (0.878 à lag 1, 0.827 à lag 10). Mais attention au piège :
    une forte auto-corrélation du **signal** signifie que la route change lentement — pas que la
    **politique** a besoin de mémoire. Ici le pilote agit surtout en fonction de la frame courante,
    donc le bénéfice réel de la mémoire est faible.

    **2. LSTM vs GRU — performance par paramètre ?**
    RMSE quasi identiques (LSTM 0.0818, GRU 0.0813) mais GRU à 46 115 paramètres contre 60 707
    (~24 % de moins). Le **GRU a donc le meilleur rapport perf/paramètre** sur ce problème à
    dépendances courtes — le LSTM ne se distinguerait que sur des dépendances longues.

    **3. Le LSTM améliore-t-il vraiment le MLP ?**
    La 1ère comparaison (cellule 12) était **biaisée** : le `SequenceDataset` prédit `action(t+T)`
    (une étape dans le futur) alors que le MLP voyait `features(t)` → `action(t)` (même instant,
    tâche plus facile). D'où la « victoire » trompeuse du MLP (0.0669 vs 0.0818). À tâche
    **équitable** (cellule 12-bis : même cible next-step, MLP limité à la dernière frame), l'écart
    LSTM/MLP se réduit fortement *(reporte ici tes chiffres)* — confirmant que sur ces données la
    mémoire n'apporte qu'un gain marginal.

    **Impact de seq_len.** RMSE plate (5 → 0.0802, 10 → 0.0818, 20 → 0.0808) : aucune amélioration
    en allongeant la fenêtre. Cohérent avec le point 1 — les frames passées sont **redondantes**
    (forte auto-corrélation), pas porteuses d'info nouvelle. La mémoire longue ne sert que si le
    passé contient un signal **absent** du présent (ex. dérive progressive non visible sur une seule
    frame), ce qui n'est pas le cas ici.

    **4. `h_n[-1]` vs `lstm_out[:, -1, :]` (2 couches).**
    Les deux désignent l'état de la dernière couche au dernier pas → identiques pour la prédiction.
    La différence n'apparaît que si on veut les états **intermédiaires** : `h_n` indexe par couche,
    `lstm_out` par pas de temps (dernière couche uniquement).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
