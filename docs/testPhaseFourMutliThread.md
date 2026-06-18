# `testPhaseFourMutliThread.py` — DQN vectorisé multi-cœurs

Notebook marimo. **Même algorithme que `PhaseFour.py` (DQN vanilla, 5 actions, 1 frame)**,
mais l'entraînement est **parallélisé sur plusieurs environnements** pour exploiter tous
les cœurs CPU. C'est l'étape intermédiaire entre la V1 et la V2 optimisée (`RLOpti.py`).

## 1. Ce qui change par rapport à `PhaseFour.py`

| Élément | `PhaseFour.py` | Ce fichier |
|---------|----------------|-----------|
| Nb d'environnements | 1 (séquentiel) | **N = min(cpu, 8)** via `AsyncVectorEnv` |
| Threads PyTorch | défaut | **`torch.set_num_threads(os.cpu_count())`** |
| Décroissance ε | multiplicative / épisode | **linéaire / pas** (`explore_frac`) |
| Sélection d'action | une à une | **batchée** (`select_actions_batched`) |
| Frame skip / hors-piste | dans la boucle | **wrapper `CarRacingWrapper`** |

Le réseau (`DQN`, entrée 1×42×42 → 5 Q-valeurs), le `ReplayBuffer` (50 000, `float32`)
et le `DQNAgent` (γ=0.95, lr=1e-4, ε 1.0→0.05, **DQN vanilla**) sont **identiques** à la
V1 — voir [PhaseFour.md](./PhaseFour.md) pour leur détail.

## 2. Architecture mise en place (les ajouts)

### 2.1 Parallélisme CPU

```python
torch.set_num_threads(os.cpu_count())                      # BLAS multi-thread
N_ENVS = min(os.cpu_count(), 8)                            # plafonné à 8
envs = gym.vector.AsyncVectorEnv([_make_env for _ in range(N_ENVS)])
```

`AsyncVectorEnv` fait tourner chaque environnement dans son **propre processus** (la
simulation physique de CarRacing est le goulot d'étranglement, pas le réseau). À chaque
itération on collecte donc **N transitions d'un coup**.

### 2.2 Sélection d'actions batchée — `select_actions_batched()`

ε-greedy vectorisé : un tirage `explore` par environnement, les actions exploratoires
sont aléatoires, les exploitantes viennent d'un **seul passage avant** sur tout le batch
d'états (`policy_net(s).argmax(1)`).

### 2.3 Wrapper d'environnement — `class CarRacingWrapper`

Trois responsabilités, **toutes côté environnement** (le reste de l'agent ne change pas) :

| Paramètre | Valeur | Effet |
|-----------|--------|-------|
| `intro=45` | 45 pas à vide au `reset` | saute l'animation de zoom d'intro |
| `skip=4` | **frame skip ×4** | répète l'action 4 frames, **somme les récompenses** |
| `off_patience=50` | compteur `neg` | termine l'épisode si trop longtemps hors-piste |

### 2.4 Boucle vectorisée — `train_dqn_vec()`

- ε **linéaire** : `eps = max(eps_end, 1 - gstep / (explore_frac · total_steps))`
  avec `explore_frac=0.6` (par pas, car par épisode décroîtrait trop vite en vectorisé).
- `total_steps=300_000`, on stocke les N transitions, **1 pas de gradient par itération**.
- `update_target` tous les `target_every=1000` pas globaux.
- Les récompenses d'épisodes terminés sont collectées dans `completed`.

Le notebook fournit aussi : sauvegarde du modèle (`dqn_model_entraine.pth` et
`models/dqn_carracing.pt`), visualisation, comparaison de γ, lecture en direct
(`render_mode="human"`) et enregistrement vidéo (`RecordVideo`).

## 3. Système de reward — **précis**

Comme la V1, **aucune récompense custom** : on part de la **récompense native de
`CarRacing-v2`** (`+1000/N` par tuile franchie, `−0.1` par frame — voir
[PhaseFour.md §3](./PhaseFour.md#3-système-de-reward--précis)). Mais le wrapper la
**transforme de deux manières** :

### 3.1 Agrégation par frame skip (`skip=4`)

```python
def step(self, action):
    total = 0.0
    for _ in range(self.skip):                 # l'action est répétée 4 fois
        obs, r, term, trunc, info = self.env.step(action)
        total += r                             # ← récompenses SOMMÉES
        if term or trunc:
            break
    ...
    return obs, total, term, trunc, info
```

La récompense renvoyée à l'agent est donc la **somme des 4 récompenses natives**. Une
« transition » de l'agent couvre 4 frames de simulation → l'horizon effectif de γ=0.95
est multiplié par 4, et l'apprentissage est ~4× plus rapide.

### 3.2 Terminaison anticipée hors-piste (`off_patience=50`)

```python
self.neg = self.neg + 1 if total < 0 else 0    # streak de récompenses négatives
if self.neg >= self.off_patience:              # 50 steps wrappés hors-piste
    term = True                                # → on coupe l'épisode
```

Une récompense agrégée **négative** signifie que sur ces 4 frames l'agent n'a franchi
aucune tuile (il n'a encaissé que des `−0.1`) : il est **hors-piste ou immobile**. Au
bout de 50 itérations consécutives dans cet état, l'épisode est terminé.

> ⚠️ Cette terminaison **ne modifie pas la valeur** des récompenses (pas de pénalité
> `−100` ajoutée) ; elle évite seulement de gaspiller des pas de simulation sur un
> épisode perdu, ce qui **densifie** le buffer en transitions utiles.

**En résumé :** récompense = somme des `r` natifs sur 4 frames, et l'épisode est coupé
après 50 paquets de 4 frames consécutifs sans aucune tuile franchie.
</content>
