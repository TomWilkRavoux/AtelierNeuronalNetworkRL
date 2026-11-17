# Conduite autonome sur CarRacing : de la perception à l'apprentissage par renforcement

Un parcours en quatre phases pour apprendre à une voiture à conduire seule dans l'environnement [`CarRacing-v2`](https://gymnasium.farama.org/environments/box2d/car_racing/) de Gymnasium : d'abord voir la route, puis imiter un pilote, puis se souvenir du passé, et enfin apprendre seule par essais et erreurs avec un Deep Q-Network.

Projet de cours réalisé en juin 2026 dans le cadre du master IA de l'EPSI, à partir du sujet de l'atelier CarRacing (dont proviennent les épisodes de démonstration de `data/demo_episodes/`).

## Les quatre phases

| Phase | Notebook | Ce que fait le modèle | Type d'apprentissage |
| --- | --- | --- | --- |
| 1. Perception | `PhaseOne.py` | Un CNN classe le type de route à partir de l'image : tout droit, virage à gauche, virage à droite | Supervisé (classification d'images) |
| 2. Imitation | `PhaseTwo.py` | Un réseau prédit les actions d'un pilote (direction, accélération, frein) à partir d'épisodes de démonstration | Supervisé (régression) |
| 3. Mémoire | `phase3.py` | Un LSTM / GRU tient compte des pas précédents, là où la phase 2 décidait à chaque pas sans mémoire | Supervisé (séquences) |
| 4. Renforcement | `PhaseFour.py`, `testPhaseFourMutliThread.py`, `RLOpti.py` | Un agent DQN apprend à conduire sans aucune donnée étiquetée, à partir de la seule récompense de l'environnement | Par renforcement |

**Phase 1** : un CNN de 3 couches convolutives (8, 16 puis 32 filtres) sur des images 96 × 96, avec visualisation de ce que détecte chaque filtre, un graphe des plus proches voisins des représentations apprises (`graphe_knn.html`) et un export ONNX pour inspecter le réseau dans [Netron](https://netron.app).

**Phase 2** : la direction est très concentrée autour de zéro (le pilote roule surtout tout droit), ce qui oriente le choix des métriques. Les données sont découpées par épisode plutôt qu'au hasard, pour ne pas évaluer le modèle sur des images quasi identiques à celles de l'entraînement ; la comparaison des deux découpages est mesurée (RMSE 0,0655 en aléatoire, 0,0650 en temporel).

**Phase 3** : l'auto-corrélation de la direction justifie l'ajout d'une mémoire, testée avec un LSTM et un GRU.

## Phase 4 : trois versions de l'agent DQN

| | `PhaseFour.py` (V1) | `testPhaseFourMutliThread.py` | `RLOpti.py` (V2) |
| --- | --- | --- | --- |
| Algorithme | DQN | DQN | **Double DQN** |
| Actions | 5 actions discrètes | 5 actions discrètes | **9 actions combinées** (direction, accélération, frein) |
| Entrée du réseau | 1 image 42 × 42 | 1 image 42 × 42 | **4 images empilées** |
| Environnements | 1 (séquentiel) | **N en parallèle** (`AsyncVectorEnv`) | N en parallèle |
| Frame skip | non | ×4 | ×3 |
| Replay buffer | 50 000, `float32` | 50 000, `float32` | 60 000, **`uint8` (4 fois moins de mémoire)** |
| Facteur γ | 0,95 | 0,95 | **0,99** |
| Modèle final | dernier | dernier | **meilleur snapshot** (moyenne sur 20 épisodes) |

Les trois versions utilisent la **récompense native** de CarRacing (+1000/N par tuile de piste franchie, −0,1 par image), sans reward shaping. Les versions avec wrapper additionnent les récompenses des images sautées et coupent l'épisode quand la voiture reste trop longtemps hors piste.

`Chargement.py` recharge l'agent entraîné (`models/dqn_carracing.pt`) pour le regarder conduire.

## Installation

Prérequis : Python 3.12 et [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`gymnasium[box2d]` compile `box2d-py` : si l'installation échoue, installer [SWIG](https://www.swig.org/) au préalable (`sudo apt install swig` sous Debian/Ubuntu).

## Lancer les notebooks

Les notebooks sont des fichiers [marimo](https://marimo.io) (`.py` versionnables), à lancer dans l'ordre des phases. Ils utilisent le GPU CUDA si disponible, sinon le CPU.

```bash
uv run marimo edit PhaseOne.py                    # 1. perception (CNN)
uv run marimo edit PhaseTwo.py                    # 2. imitation d'un pilote
uv run marimo edit phase3.py                      # 3. mémoire temporelle (LSTM / GRU)
uv run marimo edit PhaseFour.py                   # 4. DQN de référence
uv run marimo edit testPhaseFourMutliThread.py    # 4. DQN entraîné en parallèle
uv run marimo edit RLOpti.py                      # 4. Double DQN optimisé
uv run marimo edit Chargement.py                  # rejouer l'agent entraîné
```

Les modèles entraînés (`models/`, `*.pth`), exports ONNX et vidéos ne sont pas versionnés.

## Structure

```
├── PhaseOne.py                    # phase 1 : CNN de perception
├── PhaseTwo.py                    # phase 2 : imitation supervisée
├── phase3.py                      # phase 3 : LSTM / GRU
├── PhaseFour.py                   # phase 4 : DQN de référence
├── testPhaseFourMutliThread.py    # phase 4 : DQN vectorisé multi-cœurs
├── RLOpti.py                      # phase 4 : Double DQN optimisé
├── Chargement.py                  # rechargement et démonstration de l'agent
├── data/demo_episodes/            # épisodes de démonstration (sujet de l'atelier)
├── graphe_knn.html                # graphe interactif des représentations (phase 1)
└── docs/                          # documentation détaillée de la phase 4
```

## Documentation

- [Vue d'ensemble de la phase 4](docs/README.md)
- [DQN de référence](docs/PhaseFour.md)
- [DQN vectorisé](docs/testPhaseFourMutliThread.md) · [pistes d'amélioration](docs/testPhaseFourMutliThread-ameliorations.md)
- [Double DQN optimisé](docs/RLOpti.md)

## Stack

Python 3.12 · PyTorch · Gymnasium (Box2D) · marimo · ONNX · pyvis · Polars · matplotlib · uv
