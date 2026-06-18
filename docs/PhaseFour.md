# `PhaseFour.py` — DQN de référence (V1)

Notebook marimo pédagogique. C'est la **version la plus simple** : un seul
environnement, un DQN « vanilla », une frame en entrée. Elle sert de socle conceptuel
aux deux autres notebooks.

## 1. Objectif

Quitter l'apprentissage supervisé : l'agent ne reçoit **aucune donnée étiquetée**. Il
agit dans `CarRacing-v2`, observe la récompense, et apprend à estimer la **valeur** de
chaque action via le Q-Learning profond.

Briques théoriques mobilisées (cf. cellule markdown du notebook) : MDP, Q-Learning,
équation de Bellman, ε-greedy, experience replay, target network.

## 2. Architecture mise en place

### 2.1 Prétraitement de l'observation — `preprocess()`

```
obs RGB (96, 96, 3) uint8
   └─► niveaux de gris (luminance 0.299·R + 0.587·G + 0.114·B)
        └─► redimensionnement bilinéaire en 42×42
             └─► normalisation /255  →  état (1, 42, 42) float32 ∈ [0,1]
```

Une **seule frame** par état (pas de notion de vitesse/direction temporelle).

### 2.2 Le réseau — `class DQN`

CNN suivi d'un perceptron, **sans activation finale** (on prédit des Q-valeurs brutes,
qui peuvent être négatives).

| Couche | Détail | Sortie |
|--------|--------|--------|
| `Conv2d(1, 16, k=8, s=4)` + ReLU | extraction grossière | 42 → 9 |
| `Conv2d(16, 32, k=4, s=2)` + ReLU | extraction fine | 9 → 3 |
| `Flatten` + `Linear(32·3·3, 256)` + ReLU | tête dense | 256 |
| `Linear(256, 5)` | une Q-valeur **par action** | 5 |

Entrée `(B, 1, 42, 42)` → sortie `(B, 5)`. L'espace d'actions est `Discrete(5)`
(rien, gauche, droite, accélérer, freiner).

### 2.3 Mémoire de rejeu — `class ReplayBuffer`

`deque(maxlen=50_000)` de transitions `(s, a, r, s2, done)`. Le tirage aléatoire
(`random.sample`) **décorrèle** les échantillons. Stockage en `float32`.

### 2.4 L'agent — `class DQNAgent`

| Hyperparamètre | Valeur | Rôle |
|----------------|--------|------|
| `gamma` | 0.95 | facteur d'actualisation (horizon ~20 pas) |
| `lr` | 1e-4 | pas d'apprentissage (Adam) |
| `eps_start → eps_end` | 1.0 → 0.05 | ε-greedy |
| `eps_decay` | 0.995 / épisode | décroissance multiplicative |
| `buffer_cap` | 50 000 | taille du replay |

Deux réseaux : `policy_net` (entraîné) et `target_net` (figé, copié périodiquement).

**Cible d'entraînement — DQN « vanilla » (équation de Bellman) :**

```python
q_next = target_net(s2).max(1)[0]          # le MÊME réseau choisit ET évalue
target = r + gamma * q_next * (1 - done)
loss   = smooth_l1_loss(q, target)         # Huber loss
```

Le `(1 - done)` annule la valeur future quand l'épisode se termine. La perte de Huber
(`smooth_l1`) est moins sensible aux valeurs aberrantes que la MSE. Le gradient est
**clippé** à 10.0 (`clip_grad_norm_`) pour éviter les explosions.

### 2.5 Boucle d'entraînement — `train_dqn()`

Séquentielle, **un seul environnement** :

1. `reset` puis **45 pas à vide** pour sauter l'animation de zoom d'intro.
2. À chaque pas : `select_action` (ε-greedy) → `env.step` → `store` → `train_step`
   tous les `train_every=4` pas → `update_target` tous les `target_every=1000` pas.
3. **Early-stop** : si la récompense cumulée ne progresse plus pendant
   `no_progress_patience=80` pas (voiture bloquée/hors-piste), l'épisode est coupé.
4. `update_epsilon()` une fois par épisode.

Le notebook propose aussi : visualisation de l'agent (politique gloutonne, ε=0),
courbe d'apprentissage (`plot_rewards`, moyenne mobile) et comparaison de γ ∈ {0.90,
0.95, 0.99}.

## 3. Système de reward — **précis**

L'agent utilise **exclusivement la récompense native de `CarRacing-v2`**, sans aucune
modification (`r` est repris tel quel de `env.step`).

Récompense native de l'environnement :

| Événement | Récompense |
|-----------|-----------|
| Chaque frame écoulée | **−0.1** (pénalité de temps : pousse à finir vite) |
| Chaque tuile de piste **nouvellement** visitée | **+1000 / N** (N = nombre total de tuiles du circuit) |
| Tour complet (toutes les tuiles visitées) | ≈ **+900** au total, épisode `terminated` |
| Sortie loin du terrain de jeu | épisode `terminated` (forte pénalité de temps) |

> La récompense d'un épisode `ep_r` est la **somme** de ces `r` pas à pas.

### Ce que la boucle ajoute (sans toucher au reward)

- **`no_progress_patience=80`** : ce n'est **pas** un reward shaping. C'est une
  condition d'arrêt — si `ep_r` ne bat plus son record pendant 80 pas, on coupe
  l'épisode (l'agent tourne en rond ou est hors-piste). La récompense reçue n'est pas
  modifiée, seul l'épisode est raccourci.

**Conséquence pédagogique :** l'agent est incité à **rester sur la piste et avancer
vite** ; chaque seconde immobile lui coûte −0.1, et seuls les progrès sur le circuit
rapportent. C'est le signal le plus brut possible — d'où l'intérêt des optimisations
des deux autres versions.
</content>
