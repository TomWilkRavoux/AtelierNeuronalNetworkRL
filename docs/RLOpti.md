# `RLOpti.py` — DQN optimisé (V2)

Notebook marimo. Version la plus aboutie : elle reprend la base DQN des deux autres
fichiers et y ajoute les **améliorations classiques** qui font passer l'agent d'« il
tient à peu près la route » à « il fait des tours ». C'est le modèle réellement
sauvegardé (`models/dqn_v2_carracing.pt`, `dqn_model_v2_entraine.pth`).

## 1. Les six optimisations clés

1. **Double DQN** → réduit la surestimation des Q-valeurs.
2. **9 actions combinées** (frein+braquage, gas+braquage) au lieu de 5.
3. **Empilement de 4 frames** → l'agent perçoit la **vitesse** et la **direction**.
4. **Environnement `continuous=True`** piloté par un vocabulaire discret d'actions.
5. **Replay buffer en `uint8`** → 4× moins de RAM (60 000 transitions de 4 frames).
6. **Warmup + restauration du meilleur snapshot** → entraînement plus stable.

## 2. Architecture mise en place

### 2.1 Vocabulaire d'actions — `ACTIONS`

L'environnement est créé en mode **`continuous=True`** : `step` attend un vecteur
`[steer, gas, brake]`. Le wrapper expose à l'agent un **`Discrete(9)`** et traduit
chaque indice en l'un des 9 vecteurs ci-dessous :

| Idx | Vecteur `[steer, gas, brake]` | Signification |
|----:|-------------------------------|---------------|
| 0 | `[ 0.0, 0.0, 0.0]` | rien |
| 1 | `[-1.0, 0.0, 0.0]` | gauche |
| 2 | `[ 1.0, 0.0, 0.0]` | droite |
| 3 | `[ 0.0, 1.0, 0.0]` | accélérer |
| 4 | `[ 0.0, 0.0, 0.8]` | freiner |
| 5 | `[-1.0, 0.0, 0.6]` | frein + gauche |
| 6 | `[ 1.0, 0.0, 0.6]` | frein + droite |
| 7 | `[-0.6, 0.4, 0.0]` | gauche + gas |
| 8 | `[ 0.6, 0.4, 0.0]` | droite + gas |

Les **combinaisons** (freiner en tournant, accélérer en tournant) sont impossibles avec
les 5 actions disjointes des V1/V2-vectorisée — c'est ce qui permet une conduite fluide.

### 2.2 Wrapper — `class CarRacingDQN(gym.Wrapper)`

| Paramètre | Valeur | Effet |
|-----------|--------|-------|
| `intro=45` | pas à vide au reset | saute le zoom d'intro |
| `skip=3` | **frame skip ×3** | répète l'action, **somme les récompenses** |
| `n_stack=4` | **empilement de 4 frames** | état `(4, 42, 42)` |
| `off_patience=50` | compteur `neg` | terminaison hors-piste |

Chaque frame est convertie en gris 42×42 (`_to_gray42`), puis les 4 dernières frames
sont empilées (`deque(maxlen=4)`) pour former l'observation. L'`observation_space` est
donc `Box(0, 1, shape=(4, 42, 42))`.

### 2.3 Le réseau — `class DQN`

Identique en structure à la V1 **sauf l'entrée et la sortie** :

| Couche | Détail | Sortie |
|--------|--------|--------|
| `Conv2d(4, 16, k=8, s=4)` + ReLU | **4 canaux = 4 frames** | 42 → 9 |
| `Conv2d(16, 32, k=4, s=2)` + ReLU | | 9 → 3 |
| `Flatten` + `Linear(32·3·3, 256)` + ReLU | | 256 |
| `Linear(256, 9)` | **9 Q-valeurs** | 9 |

Entrée `(B, 4, 42, 42)` → sortie `(B, 9)`.

### 2.4 Replay buffer compressé — `class ReplayBuffer`

```python
def push(self, s, a, r, s2, done):
    self.buf.append(((s*255).astype(np.uint8), a, r, (s2*255).astype(np.uint8), done))
def sample(self, batch_size):
    ... np.stack(s).astype(np.float32) / 255.0 ...      # reconversion au tirage
```

Les états sont stockés en **`uint8`** (×255) et reconvertis en `float32` au moment de
l'échantillonnage → **RAM divisée par 4**, ce qui rend tenable un buffer de 60 000
états de 4 frames.

### 2.5 L'agent — `class DQNAgent` (Double DQN)

Hyperparamètres : `n_actions=9`, `lr=1e-4`, ε 1.0→0.05. **γ utilisé à l'entraînement :
0.99** (horizon long, ~100 pas — un circuit complet), passé explicitement à la
construction.

**Cible Double DQN** (la différence majeure avec les V1/V2-vectorisée) :

```python
with torch.no_grad():
    next_actions = policy_net(s2).argmax(1, keepdim=True)   # la POLICY choisit a*
    q_next       = target_net(s2).gather(1, next_actions)   # la TARGET évalue a*
    target = r + gamma * q_next * (1 - done)
loss = smooth_l1_loss(q, target)                            # Huber + clip_grad 10.0
```

En **découplant la sélection de l'action et son évaluation**, on supprime le biais
optimiste du `max` du DQN vanilla (qui surestime systématiquement les Q-valeurs et
provoque stagnation/oscillation).

### 2.6 Boucle vectorisée — `train_dqn_vec()`

Reprend la version vectorisée et ajoute :

| Mécanisme | Valeur | Rôle |
|-----------|--------|------|
| `learning_starts=5000` | **warmup** | on remplit le buffer avant d'apprendre (échantillons variés) |
| `grad_steps=1` | replay ratio | nb de pas de gradient par itération (montable, coûteux) |
| `target_every=2000` | maj target | réseau cible plus stable |
| `explore_frac=0.5` | ε linéaire | exploration sur la 1ʳᵉ moitié des `total_steps` |
| **best snapshot** | `deepcopy` | on garde le réseau de **meilleure moyenne (20 ép.)** et on le **restaure** à la fin |

`total_steps=300_000`, `N_ENVS=min(cpu, 8)` via `AsyncVectorEnv`. Le notebook propose
sauvegarde, visualisation, comparaison de γ, lecture live, et enregistrement vidéo.

## 3. Système de reward — **précis**

Toujours **la récompense native de `CarRacing-v2`** (`+1000/N` par tuile franchie,
`−0.1` par frame — voir [PhaseFour.md §3](./PhaseFour.md#3-système-de-reward--précis)),
transformée par le wrapper. La logique est la même que la version vectorisée, **mais
avec `skip=3`** (et non 4).

### 3.1 Agrégation par frame skip (`skip=3`)

```python
def step(self, action_idx):
    act, total = ACTIONS[int(action_idx)], 0.0
    for _ in range(self.skip):                  # action continue répétée 3 frames
        obs, r, term, trunc, info = self.env.step(act)
        total += r                              # ← récompenses SOMMÉES
        if term or trunc:
            break
    self.frames.append(_to_gray42(obs))         # met à jour la pile de 4 frames
    ...
    return self._state(), total, term, trunc, info
```

La récompense d'une transition de l'agent = **somme des 3 récompenses natives** des
frames sautées. L'action passée à l'env est le **vecteur continu** `[steer, gas, brake]`
correspondant à l'indice discret choisi.

### 3.2 Terminaison anticipée hors-piste (`off_patience=50`)

```python
self.neg = self.neg + 1 if total < 0 else 0     # streak de paquets négatifs
if self.neg >= self.off_patience:               # 50 paquets de 3 frames hors-piste
    term = True
```

Identique en esprit à la version vectorisée : un total négatif sur 3 frames = aucune
tuile franchie = hors-piste/immobile. Après 50 itérations consécutives, l'épisode est
coupé. **La valeur des récompenses n'est pas modifiée** (pas de pénalité ajoutée) ;
seule la longueur de l'épisode l'est.

### 3.3 Interaction avec γ=0.99 et le frame skip

Avec `skip=3` et `γ=0.99`, la cible de Bellman propage la valeur sur un horizon
effectif de l'ordre de `3 / (1 − 0.99) ≈ 300` frames de simulation : l'agent peut
« voir loin » et planifier l'entrée d'un virage plusieurs dixièmes de seconde à
l'avance — d'où l'intérêt d'empiler 4 frames (perception de la vitesse) et d'autoriser
les actions combinées frein+braquage.

**En résumé :** récompense = somme des `r` natifs sur 3 frames ; épisode coupé après
50 paquets consécutifs sans tuile franchie ; valeur estimée par Double DQN avec γ=0.99
sur un état de 4 frames empilées.
</content>
