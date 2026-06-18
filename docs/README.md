# Documentation — Apprentissage par renforcement (DQN sur CarRacing-v2)

Ce dossier documente les trois notebooks [marimo](https://marimo.io) consacrés à la
**Phase 4** du projet : entraîner un agent autonome à conduire dans l'environnement
`CarRacing-v2` de [Gymnasium](https://gymnasium.farama.org) à l'aide d'un
**Deep Q-Network (DQN)**.

> ⚠️ Ce sont des notebooks marimo : des fichiers `.py` versionnables où chaque cellule
> est une fonction décorée par `@app.cell`. On les ouvre avec `marimo edit <fichier>.py`.

## Les trois fichiers documentés

| Fichier | Rôle | Doc dédiée |
|---------|------|-----------|
| `PhaseFour.py` | **Version de référence** (V1) : DQN « vanilla », mono-environnement, pédagogique. | [PhaseFour.md](./PhaseFour.md) |
| `testPhaseFourMutliThread.py` | **Version vectorisée** : même algo que V1 mais entraînement parallèle multi-cœurs (`AsyncVectorEnv`) + wrapper d'environnement. | [testPhaseFourMutliThread.md](./testPhaseFourMutliThread.md) · [pistes d'amélioration](./testPhaseFourMutliThread-ameliorations.md) |
| `RLOpti.py` | **Version optimisée** (V2) : Double DQN, 9 actions combinées, empilement de 4 frames, buffer compressé, restauration du meilleur modèle. | [RLOpti.md](./RLOpti.md) |

## Tableau comparatif synthétique

| Aspect | `PhaseFour.py` (V1) | `testPhaseFourMutliThread.py` | `RLOpti.py` (V2) |
|--------|:---:|:---:|:---:|
| Environnement | `CarRacing-v2` `continuous=False` | `continuous=False` | `continuous=True` |
| Espace d'actions | `Discrete(5)` | `Discrete(5)` | **9 actions combinées** (steer/gas/brake) |
| Entrée du réseau | 1 frame `(1,42,42)` | 1 frame `(1,42,42)` | **4 frames empilées `(4,42,42)`** |
| Algorithme | DQN vanilla | DQN vanilla | **Double DQN** |
| Parallélisme | 1 env (séquentiel) | N envs (`AsyncVectorEnv`) | N envs (`AsyncVectorEnv`) |
| Frame skip | non (dans la boucle) | **×4** (wrapper) | **×3** (wrapper) |
| Replay buffer | 50 000, `float32` | 50 000, `float32` | 60 000, **`uint8` (÷4 RAM)** |
| γ (discount) utilisé | 0.95 | 0.95 | **0.99** |
| Warmup buffer | non | non | **`learning_starts=5000`** |
| Sélection du modèle final | dernier | dernier | **meilleur snapshot (moy. 20 ép.)** |

## Le système de reward en une phrase

Les trois fichiers **n'inventent pas de récompense** : ils réutilisent la
**récompense native de `CarRacing-v2`** (`+1000/N` par tuile de piste franchie,
`−0.1` par frame). Ce que les versions wrappées ajoutent n'est pas un *reward shaping*
mais deux mécanismes :
1. **frame skip** → les récompenses des frames sautées sont **sommées** ;
2. **terminaison anticipée hors-piste** → l'épisode se coupe après un trop long
   passage en récompense négative (la voiture est sortie du circuit).

Le détail exact de chaque reward est donné dans la doc de chaque fichier.
</content>
</invoke>
