# Pistes d'amélioration — `testPhaseFourMutliThread.py`

Document de suite à [testPhaseFourMutliThread.md](./testPhaseFourMutliThread.md).

Le fichier multi-thread est un **DQN vanilla** (5 actions, 1 frame). La plupart des
améliorations « de base » ont **déjà été faites par toi dans `RLOpti.py`** (la V2). Ce
document est donc en deux temps :

- **Partie 1 — Déjà résolu dans `RLOpti.py`** : rappel court, pour mémoire. Si tu veux
  juste remettre le fichier multi-thread à niveau, c'est cette liste qu'il faut porter.
- **Partie 2 — Au-delà de `RLOpti.py`** : les vraies pistes neuves, qui perfectionneraient
  **même ta meilleure version actuelle**.

## Légende

- 💰 gain attendu · 🔧 effort · 📍 statut : ✅ fait dans `RLOpti.py` · 🆕 nouveau (au-delà de la V2)

---

# Partie 1 — Déjà résolu dans `RLOpti.py` ✅

Ces briques **manquent** au fichier multi-thread mais **existent déjà** dans ta V2. Rien
à réinventer : il suffirait de les recopier depuis `RLOpti.py`. Listées juste pour situer
le point de départ.

| Brique | Ce que ça corrige | Où la copier |
|--------|-------------------|--------------|
| **Empilement de 4 frames** | l'agent perçoit vitesse & direction | `CarRacingDQN`, `_to_gray42` |
| **Double DQN** | surestimation des Q-valeurs | `DQNAgent.train_step` |
| **Warmup (`learning_starts`)** | premiers gradients trop bruités | `train_dqn_vec` |
| **Meilleur snapshot restauré** | la perf DQN oscille | `train_dqn_vec` (best_state) |
| **Buffer `uint8` 60k** | RAM ÷4 | `ReplayBuffer.push/sample` |
| **γ=0.99 + 9 actions combinées** | horizon long + conduite fluide | `ACTIONS`, construction de l'agent |

➡️ **Conclusion :** porter la Partie 1, c'est reconstruire `RLOpti.py`. L'intérêt n'est
pas la perf finale (tu l'as déjà), mais de pouvoir **isoler et mesurer le gain de chaque
brique** grâce à l'entraînement vectorisé rapide de ce fichier. À ne faire que comme
exercice de validation. Le vrai travail neuf est en Partie 2.

---

# Partie 2 — Au-delà de `RLOpti.py` 🆕

Ces pistes ne sont **dans aucun** de tes fichiers. Elles s'appliquent aussi bien au
multi-thread qu'à ta V2 : c'est **ici** que se trouve la marge de progression réelle.

## D. Améliorations algorithmiques du DQN

### D.1 — Dueling DQN 🆕 💰💰 🔧🔧
Séparer la dernière couche en deux têtes : **valeur d'état `V(s)`** et **avantage
`A(s,a)`**, recombinées par `Q = V + (A − mean(A))`. En conduite, beaucoup d'actions ont
une valeur quasi identique sur la ligne droite : isoler `V(s)` stabilise l'apprentissage.
Se combine directement avec ton Double DQN existant (« Dueling Double DQN »).

```python
self.val = nn.Linear(256, 1)            # V(s)
self.adv = nn.Linear(256, n_actions)    # A(s,a)
def forward(self, x):
    h = self.fc(self.conv(x))
    v, a = self.val(h), self.adv(h)
    return v + a - a.mean(1, keepdim=True)
```

### D.2 — Prioritized Experience Replay (PER) 🆕 💰💰 🔧🔧🔧
Aujourd'hui le tirage est **uniforme** (`random.sample`). PER échantillonne en priorité
les transitions à **forte erreur TD** (les plus « instructives »), avec un poids
d'importance-sampling pour corriger le biais. Gain net d'efficacité par échantillon,
particulièrement utile ici où les transitions « franchit une tuile » sont rares face aux
`−0.1` répétitifs. Implémentation via un **sum-tree**.

### D.3 — N-step returns 🆕 💰 🔧🔧
Cible Bellman sur `n` pas au lieu d'un seul :
`R = r_t + γ r_{t+1} + … + γⁿ⁻¹ r_{t+n-1} + γⁿ · Q(s_{t+n}, a*)`.
Propage le signal de récompense bien plus vite — décisif quand les récompenses sont
éparses (cas de CarRacing). `n=3` est un bon point de départ.

### D.4 — Distributional / Rainbow 🆕 💰💰 🔧🔧🔧🔧
Combiner Double + Dueling + PER + N-step + NoisyNets + C51, c'est **Rainbow DQN**,
l'état de l'art côté value-based. Tu as déjà Double ; D.1→D.3 te rapprochent. À voir
comme une cible long terme plutôt qu'un sprint.

## E. Changer de famille d'algorithme

### E.1 — PPO / SAC (contrôle continu natif) 🆕 💰💰💰 🔧🔧🔧🔧
`CarRacing` est **nativement continu** (`[steer, gas, brake]`). Tu le discrétises en 9
actions — un compromis. Un **policy-gradient** (PPO) ou un **acteur-critique** (SAC)
exploite directement l'espace continu, donne un contrôle plus fin et se révèle souvent
plus stable que le DQN sur cet environnement. C'est un **changement de paradigme** (pas
de replay/target pour PPO, structure acteur-critique) : gros effort, mais c'est la piste
au plus fort potentiel si le DQN plafonne.

## F. Représentation & perception

### F.1 — Frame stacking plus riche 🆕 💰 🔧🔧
Au-delà des 4 frames empilées de la V2 : tester **différence de frames** (mouvement
explicite), ou concaténer des **canaux d'info** (vitesse/angle lus dans l'`info` de gym)
en entrée de la partie dense.

### F.2 — Résolution / encodeur 🆕 💰 🔧🔧
42×42 est volontairement petit (rapide sur CPU). Si tu passes sur GPU, tester 84×84
(résolution Atari standard) et un CNN un peu plus profond améliore la lecture des virages.

### G. Reward shaping (prudent) 🆕 💰 🔧🔧
Le reward reste **natif** dans tous tes fichiers. Pistes à tester **avec garde-fous**
(un mauvais shaping fait diverger l'agent) :
- petit bonus proportionnel à la **vitesse** quand on est sur la piste ;
- pénalité douce sur les **changements de braquage** brusques (conduite moins saccadée) ;
- pénalité explicite **−x à la sortie de piste** (au lieu de seulement couper l'épisode).

⚠️ Toujours comparer à la baseline non-shapée, sur plusieurs seeds.

---

# Méthode & outillage (indispensable pour la Partie 2)

Sans mesure fiable, impossible de savoir si D/E/F/G aident vraiment. Ces points
**n'existent dans aucun de tes fichiers** et sont prioritaires avant d'empiler les algos.

### M.1 — Évaluation déterministe périodique 🆕 💰💰 🔧
Toutes les ~20 000 pas, jouer **5 épisodes à ε=0** et logguer la moyenne. C'est la seule
métrique fiable : le reward d'entraînement est pollué par l'exploration.

### M.2 — Multi-seeds 🆕 💰💰 🔧
Le RL est très sensible au seed. Rejouer chaque config sur **3 seeds** et comparer
moyenne ± écart-type, sinon on conclut sur du bruit.

### M.3 — Suivi des métriques 🆕 💰 🔧🔧
Logger **loss**, **Q-valeur moyenne**, **longueur d'épisode**, **ε** (CSV ou TensorBoard).
Une Q-valeur qui diverge trahit une surestimation résiduelle.

### M.4 — Checkpoints réguliers 🆕 💰 🔧
Sauvegarder tous les N pas, pas seulement à la fin — filet de sécurité si la perf
s'effondre, complémentaire du « best snapshot » déjà présent dans la V2.

---

## Plan d'attaque conseillé (pour aller au-delà de la V2)

1. **D'abord M.1 + M.3** sur `RLOpti.py` : se doter d'une mesure fiable.
2. **D.1 (Dueling)** puis **D.3 (N-step)** : gros gain, effort modéré, se branchent sur ton Double DQN.
3. **D.2 (PER)** si tu veux pousser l'efficacité par échantillon.
4. **E.1 (PPO/SAC)** comme alternative de fond si le DQN plafonne malgré D.
5. **G (shaping)** seulement en dernier, toujours validé contre la baseline.

> En clair : la Partie 1 te ramène à ce que tu as déjà ; **c'est la Partie 2 (D → G + M)
> qui fait réellement progresser ton meilleur agent**.
</content>
