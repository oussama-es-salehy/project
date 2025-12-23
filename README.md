# 🧠 BERT avec TensorFlow Hub – Guide Complet

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?logo=python)](https://www.python.org/)
[![BERT](https://img.shields.io/badge/BERT-Transformer-green)](https://arxiv.org/abs/1810.04805)

> Guide détaillé sur BERT (Bidirectional Encoder Representations from Transformers) avec implémentation TensorFlow Hub

---

## 📚 Table des Matières

- [Introduction](#-introduction)
- [Architecture Transformer](#-architecture-transformer)
- [Concepts Fondamentaux](#-concepts-fondamentaux)
- [Composants de BERT](#-composants-de-bert)
- [Pré-entraînement](#-pré-entraînement)
- [Code & Implémentation](#-code--implémentation)
- [Applications](#-applications)
- [Ressources](#-ressources)

---

## 🎯 Introduction

### Qu'est-ce que BERT ?

**BERT (Bidirectional Encoder Representations from Transformers)** est un modèle révolutionnaire de traitement du langage naturel développé par Google AI en 2018.

#### ✨ Caractéristiques principales

| Caractéristique | Description |
|-----------------|-------------|
| **Bidirectionnel** | Lit le texte dans les deux sens (←→) |
| **Contextuel** | Comprend le sens selon le contexte |
| **Pré-entraîné** | Entraîné sur 3,3 milliards de mots |
| **Transfer Learning** | Adaptable à diverses tâches NLP |

#### 💡 Exemple de compréhension contextuelle

```text
Phrase 1: "Je vais à la banque pour déposer de l'argent"
         → BERT comprend: banque = institution financière

Phrase 2: "Je m'assieds sur la banque du parc"
         → BERT comprend: banque = siège

✅ Même mot, sens différent selon le contexte !
```

---

## 🏗️ Architecture Transformer

### Le Transformer : Fondation de BERT

BERT utilise uniquement la partie **Encoder** du Transformer original (Vaswani et al., 2017).

```
┌─────────────────────────────────────────┐
│         TRANSFORMER ORIGINAL            │
├────────────────────┬────────────────────┤
│     ENCODER        │      DECODER       │
│  (utilisé par      │   (non utilisé     │
│     BERT)          │    par BERT)       │
│                    │                    │
│  • Bidirectionnel  │  • Unidirectionnel │
│  • Comprend        │  • Génère          │
└────────────────────┴────────────────────┘
```

### Pourquoi uniquement l'Encoder ?

- **Encoder** : Comprend et analyse le texte (bidirectionnel)
- **Decoder** : Génère du nouveau texte (unidirectionnel)

**BERT se concentre sur la compréhension, pas la génération.**

---

## 🔬 Concepts Fondamentaux

### 1️⃣ Embeddings

Les embeddings transforment les mots en vecteurs numériques de 768 dimensions.

#### Types d'embeddings dans BERT

```python
Embedding Final = Token Embedding + Position Embedding + Segment Embedding
```

#### 📊 Détails de chaque embedding

**A. Token Embedding** (768 dimensions)

```python
"python" → [0.23, -0.45, 0.12, 0.67, ..., 0.89]  # 768 valeurs
```

**B. Position Embedding**

```python
Position 0: [0.1, 0.2, 0.3, ..., 0.8]
Position 1: [0.3, 0.4, 0.5, ..., 0.9]
Position 2: [0.5, 0.6, 0.7, ..., 1.0]
```

→ Indique la position du mot dans la phrase (ordre des mots)

**C. Segment Embedding**

```python
Phrase A: [0, 0, 0, 0, 0, ...]
Phrase B: [1, 1, 1, 1, 1, ...]
```

→ Distingue différentes phrases dans l'input

#### 🎨 Visualisation

```
Input: "Hello World"

Token Emb:    [0.2, 0.3, ...] + [0.5, 0.6, ...]
Position Emb: [0.1, 0.1, ...] + [0.2, 0.2, ...]
Segment Emb:  [0.0, 0.0, ...] + [0.0, 0.0, ...]
              ─────────────────────────────────
Final Emb:    [0.3, 0.4, ...] + [0.7, 0.8, ...]
              
              "Hello"           "World"
```

---

### 2️⃣ Self-Attention Mechanism

Le mécanisme clé qui permet à BERT de comprendre le contexte.

#### 📐 Formule mathématique

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Où:
Q = Query  (requête)   - "Que cherche-t-on ?"
K = Key    (clé)       - "Quelles informations avons-nous ?"
V = Value  (valeur)    - "Quelles sont les informations ?"
d_k = dimension des clés (64 pour BERT)
```

#### 🔍 Exemple concret

```text
Phrase: "Le chat mange la souris"

Self-Attention sur le mot "mange":

Chat    ████████████████████ 0.80  ← Forte attention (sujet)
Le      ████ 0.10
mange   ██ 0.05
la      █ 0.02
souris  ████████ 0.30  ← Attention modérée (objet)

→ BERT comprend que "chat" est le sujet de "mange"
→ BERT comprend que "souris" est l'objet de "mange"
```

---

### 3️⃣ Multi-Head Attention

BERT utilise **12 têtes d'attention** en parallèle.

#### 🎯 Pourquoi plusieurs têtes ?

Chaque tête apprend différents aspects linguistiques :

| Tête | Apprentissage |
|------|---------------|
| **Tête 1** | Relations syntaxiques (sujet-verbe) |
| **Tête 2** | Relations sémantiques (synonymes) |
| **Tête 3** | Dépendances longues distances |
| **Tête 4** | Coréférences (pronoms → noms) |
| **Tête 5-12** | Autres patterns linguistiques |

#### 📊 Architecture

```
                    Input
                      ↓
    ┌─────┬─────┬─────┬─────┬───┬─────┐
    │Head1│Head2│Head3│Head4│...│Head12│
    │     │     │     │     │   │     │
    │ 64d │ 64d │ 64d │ 64d │...│ 64d │
    └─────┴─────┴─────┴─────┴───┴─────┘
      ↓     ↓     ↓     ↓         ↓
      └─────┴─────┴─────┴─────────┘
                  ↓
           Concatenation
          (12 × 64 = 768)
                  ↓
              Output
```

---

### 4️⃣ Feed-Forward Network

Après l'attention, chaque token passe par un réseau neuronal à deux couches.

```
Input (768) → Dense(3072) → GELU → Dense(768) → Output

Expansion 4x ──────────┘          └────── Compression
```

#### ⚡ Fonction d'activation GELU

```python
GELU(x) = x × Φ(x)  # Φ = fonction de distribution normale

Plus douce que ReLU, mieux adaptée au NLP
```

---

### 5️⃣ Layer Normalization & Residual Connections

Architecture d'une couche Encoder complète :

```
┌────────────────────────────────┐
│  Input                         │
│    ↓                           │
│  Multi-Head Attention          │
│    ↓                           │
│  Add & Normalize  ←───────┐    │
│    ↓                      │    │
│  Feed Forward Network     │    │
│    ↓                      │    │
│  Add & Normalize  ←───────┘    │
│    ↓                           │
│  Output                        │
└────────────────────────────────┘

Residual Connections (→) évitent le gradient vanishing
```

---

## 🧩 Composants de BERT

### Configuration BERT-Base

```
┌──────────────────────────────────┐
│     BERT-Base (utilisé ici)      │
├──────────────────────────────────┤
│ L = 12   Layers (Encoders)       │
│ H = 768  Hidden size (dimensions)│
│ A = 12   Attention heads         │
│ Params = 110 millions            │
└──────────────────────────────────┘
```

### Configuration BERT-Large

```
┌──────────────────────────────────┐
│         BERT-Large               │
├──────────────────────────────────┤
│ L = 24   Layers                  │
│ H = 1024 Hidden size             │
│ A = 16   Attention heads         │
│ Params = 340 millions            │
└──────────────────────────────────┘
```

---

### 🏷️ Tokens Spéciaux

| Token | Nom | Rôle | Position |
|-------|-----|------|----------|
| `[CLS]` | Classification | Représente toute la phrase | Début |
| `[SEP]` | Separator | Sépare deux phrases | Entre/Fin |
| `[PAD]` | Padding | Remplit pour égaliser la longueur | Fin |
| `[MASK]` | Mask | Mot masqué (entraînement) | Variable |
| `[UNK]` | Unknown | Mot hors vocabulaire | Variable |

#### 📝 Exemple complet

```text
Entrée brute: "Python is great"

Après tokenisation:
[CLS] python is great [SEP] [PAD] [PAD] [PAD]
  ↑      ↑            ↑      ↑
  │      │            │      └─ Séparateur
  │      └────────────┘         (marqueur de fin)
  └─ Token de classification
     (représentation globale)
```

---

### 📖 Vocabulaire WordPiece

BERT utilise **WordPiece tokenization** avec **30 000 tokens**.

#### 🔪 Découpage des mots

```text
Mot inconnu: "unbreakable"

Tokenization WordPiece:
["un", "##break", "##able"]
 ↑      ↑          ↑
 │      │          └─ Suffixe (## = continuation)
 │      └─────────── Racine (continuation)
 └────────────────── Préfixe

Avantages:
✅ Gère les mots rares
✅ Réduit la taille du vocabulaire
✅ Partage les sous-mots communs
```

---

## 🎓 Pré-entraînement

BERT est pré-entraîné sur **deux tâches non supervisées** :

### 1️⃣ Masked Language Model (MLM)

**Objectif** : Prédire les mots masqués dans une phrase.

#### 🎭 Processus

```text
1. Prendre une phrase
2. Masquer 15% des tokens aléatoirement
3. BERT prédit les mots originaux

Exemple:

Original:   I love python programming
            ↓
Masqué:     I love [MASK] programming
            ↓
Prédiction: I love python programming ✅
```

#### 📊 Stratégie de masquage (pour les 15% choisis)

| Action | Probabilité | Exemple |
|--------|-------------|---------|
| Remplacer par `[MASK]` | 80% | `I love [MASK]` |
| Remplacer par mot aléatoire | 10% | `I love banana` |
| Laisser inchangé | 10% | `I love python` |

**Pourquoi cette stratégie ?**

```text
80% [MASK] : Entraînement principal
10% aléatoire : Évite de trop dépendre de [MASK]
10% inchangé : Apprend les représentations sans masque
```

---

### 2️⃣ Next Sentence Prediction (NSP)

**Objectif** : Prédire si la phrase B suit logiquement la phrase A.

#### 💬 Exemples

**✅ IsNext (label = 1)**

```text
Phrase A: "Je vais au marché"
Phrase B: "J'achète des fruits"

→ Ces phrases se suivent logiquement
```

**❌ NotNext (label = 0)**

```text
Phrase A: "Je vais au marché"
Phrase B: "Les étoiles brillent la nuit"

→ Aucun lien logique entre les phrases
```

#### 📋 Format d'entrée

```text
[CLS] Phrase A [SEP] Phrase B [SEP]
  ↑            ↑              ↑
  │            │              └─ Fin de B
  │            └─ Séparateur A/B
  └─ Classification (IsNext ou NotNext)
```

---

### 📚 Données d'entraînement

| Source | Nombre de mots |
|--------|----------------|
| BooksCorpus | 800 millions |
| Wikipedia (EN) | 2,5 milliards |
| **Total** | **3,3 milliards** |

**Temps d'entraînement** : 4 jours sur 64 TPU v3

---

## 💻 Code & Implémentation

### 📦 Installation

```bash
pip install tensorflow tensorflow-hub tensorflow-text
```

### 🔧 Imports

```python
import tensorflow_hub as hub
import tensorflow_text as text
```

**Rôle des bibliothèques :**

| Bibliothèque | Fonction |
|--------------|----------|
| `tensorflow_hub` | Charge modèles pré-entraînés |
| `tensorflow_text` | Opérations NLP (tokenisation) |

---

### 🌐 URLs des modèles

```python
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
```

#### 🔍 Décomposition de l'URL encoder

```text
bert_en_uncased_L-12_H-768_A-12
│    │  │        │    │     │
│    │  │        │    │     └─ A-12  : 12 Attention heads
│    │  │        │    └─────── H-768 : 768 dimensions cachées
│    │  │        └──────────── L-12  : 12 couches Encoders
│    │  └───────────────────── uncased : minuscules uniquement
│    └──────────────────────── en : anglais
└───────────────────────────── bert : modèle BERT
```

---

### 🔄 Chargement du préprocesseur

```python
bert_preprocess_model = hub.KerasLayer(preprocess_url)
```

**Ce que fait le préprocesseur :**

```
Texte brut
    ↓
1. Tokenisation (WordPiece)
    ↓
2. Conversion en IDs numériques
    ↓
3. Ajout tokens [CLS], [SEP]
    ↓
4. Génération masques et segments
    ↓
Données prêtes pour BERT
```

---

### 📝 Données d'entrée

```python
text_test = ['nice movie indeed', 'I love python programming']
```

**Format :**
- Liste de chaînes de caractères
- Batch de 2 phrases
- Pas de prétraitement manuel nécessaire

---

### ⚙️ Prétraitement

```python
text_preprocessed = bert_preprocess_model(text_test)
print(text_preprocessed.keys())
```

**Sortie :**

```python
dict_keys(['input_word_ids', 'input_mask', 'input_type_ids'])
```

#### 🔍 Visualisation détaillée

```text
Phrase: "nice movie indeed"

┌──────────────┬──────┬───────┬───────┬───────┬──────┬──────┬──────┐
│ Token        │ [CLS]│ nice  │ movie │indeed │ [SEP]│ [PAD]│ [PAD]│
├──────────────┼──────┼───────┼───────┼───────┼──────┼──────┼──────┤
│input_word_ids│ 101  │ 3835  │ 3185  │ 5442  │ 102  │  0   │  0   │
│input_mask    │  1   │  1    │  1    │  1    │  1   │  0   │  0   │
│input_type_ids│  0   │  0    │  0    │  0    │  0   │  0   │  0   │
└──────────────┴──────┴───────┴───────┴───────┴──────┴──────┴──────┘
                 ↑      ↑       ↑       ↑       ↑      ↑      ↑
                 │      └───────┴───────┴───────┘      └──────┘
            Token de                Vrais tokens        Padding
          classification
```

---

### 🔢 input_word_ids

```python
print(text_preprocessed['input_word_ids'])
```

**Exemple de sortie :**

```python
array([[  101,  3835,  3185,  5442,   102,     0,     0],
       [  101,  1045,  2293, 15894,  4730,   102,     0]])
```

**Explication :**

```python
Phrase 1: "nice movie indeed"
┌──────┬───────┬───────┬───────┬──────┬──────┬──────┐
│ 101  │ 3835  │ 3185  │ 5442  │ 102  │  0   │  0   │
└──┬───┴───┬───┴───┬───┴───┬───┴──┬───┴──┬───┴──┬───┘
   │       │       │       │      │      │      │
  [CLS]  nice   movie  indeed  [SEP] [PAD] [PAD]
```

**IDs spéciaux :**
- `101` = `[CLS]`
- `102` = `[SEP]`
- `0` = `[PAD]`

---

### 🎭 input_mask

```python
print(text_preprocessed['input_mask'])
```

**Sortie :**

```python
array([[1, 1, 1, 1, 1, 0, 0],
       [1, 1, 1, 1, 1, 1, 0]])
```

**Rôle :**

```
1 = Token réel (BERT traite)
0 = Padding (BERT ignore)

Phrase 1: [1, 1, 1, 1, 1, 0, 0]
           ↑  ↑  ↑  ↑  ↑  ↑  ↑
           │  │  │  │  │  └──┴─ Padding (ignoré)
           └──┴──┴──┴──┘
           Tokens réels (traités)
```

**Importance :** Évite que BERT n'apprenne des patterns sur le padding.

---

### 🏷️ input_type_ids

```python
print(text_preprocessed['input_type_ids'])
```

**Sortie :**

```python
array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]])
```

**Rôle : Distinguer les segments**

```text
Cas avec deux phrases :

Input: "Hello world [SEP] How are you [SEP]"

input_type_ids:
[0, 0, 0, 1, 1, 1, 1]
 └─────┘  └─────────┘
 Phrase A   Phrase B
```

**Dans notre cas** (une seule phrase) :
- Tous les `input_type_ids` = `0`

**Utilisation :**
- Question Answering (question + contexte)
- Similarité de phrases
- Inférence de langage naturel (NLI)

---

### 🧠 Chargement de l'encodeur BERT

```python
bert_model = hub.KerasLayer(encoder_url)
```

**Contenu du modèle :**

```
┌────────────────────────┐
│   Encodeur BERT        │
├────────────────────────┤
│ • 12 couches Encoder   │
│ • Multi-Head Attention │
│ • Feed-Forward Networks│
│ • Layer Normalization  │
│ • 110M paramètres      │
└────────────────────────┘
```

---

### 🚀 Passage dans BERT

```python
bert_results = bert_model(text_preprocessed)
print(bert_results.keys())
```

**Sortie :**

```python
dict_keys(['pooled_output', 'sequence_output', 'encoder_outputs', 'default'])
```

#### 📊 Vue d'ensemble des sorties

```
text_preprocessed
        ↓
┌───────────────┐
│  BERT Encoder │
│   (12 layers) │
└───────────────┘
        ↓
        ├─→ sequence_output  (tokens individuels)
        ├─→ pooled_output    (phrase entière)
        └─→ encoder_outputs  (couches intermédiaires)
```

---

### 📤 sequence_output

```python
print(bert_results['sequence_output'])
print(bert_results['sequence_output'].shape)
```

**Dimensions :** `(batch_size, sequence_length, 768)`

```
Exemple: (2, 7, 768)
         │  │  └── 768 dimensions d'embedding
         │  └───── 7 tokens par phrase
         └──────── 2 phrases dans le batch
```

#### 🎯 Contenu

```python
Phrase: [CLS] nice movie indeed [SEP] [PAD] [PAD]

Embeddings:
[CLS]    → [0.12, -0.34, 0.56, ..., 0.89]  # 768 valeurs
nice     → [0.23, 0.45, -0.12, ..., 0.34]  # 768 valeurs
movie    → [-0.56, 0.78, 0.23, ..., -0.45] # 768 valeurs
indeed   → [0.67, -0.23, 0.91, ..., 0.12]  # 768 valeurs
[SEP]    → [0.34, 0.12, -0.67, ..., 0.78]  # 768 valeurs
[PAD]    → [0.00, 0.00, 0.00, ..., 0.00]  # 768 valeurs
[PAD]    → [0.00, 0.00, 0.00, ..., 0.00]  # 768 valeurs
```

#### 💼 Utilisation

| Tâche | Description |
|-------|-------------|
| **NER** | Named Entity Recognition |
| **POS Tagging** | Part-of-Speech tagging |
| **Question Answering** | Trouver réponse dans texte |
| **Token Classification** | Classifier chaque mot |

**Exemple d'utilisation :**

```python
# Extraire l'embedding du 2ème mot (nice)
word_embedding = bert_results['sequence_output'][0, 1, :]
print(word_embedding.shape)  # (768,)
```

---

### 🎯 pooled_output

```python
print(bert_results['pooled_output'])
print(bert_results['pooled_output'].shape)
```

**Dimensions :** `(batch_size, 768)`

```
Exemple: (2, 768)
         │  └── 768 dimensions
         └───── 2 phrases
```

#### 🧮 Calcul

```python
pooled_output = tanh(Dense(sequence_output[CLS]))

Étapes:
1. Prendre l'embedding du token [CLS]
2. Passer par une couche Dense
3. Appliquer activation tanh
4. Obtenir représentation de la phrase
```

#### 🎨 Visualisation

```
Phrase complète: "nice movie indeed"
                        ↓
              Passe par BERT (12 layers)
                        ↓
         [CLS] nice movie indeed [SEP]
           ↓
    Embedding [CLS] (768)
           ↓
     Dense Layer
           ↓
    Activation tanh
           ↓
   pooled_output (768)
           ↓
   Représentation de toute la phrase
```

#### 💼 Utilisation

| Tâche | Description |
|-------|-------------|
| **Classification de texte** | Positif/Négatif/Neutre |
| **Analyse de sentiment** | Émotions |
| **Similarité sémantique** | Comparer phrases |
| **Classification binaire** | Spam/Ham, etc. |

**Exemple d'utilisation :**

```python
# Classifier le sentiment
from tensorflow.keras import layers

classifier = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')  # 3 classes: pos/neg/neu
])

sentiment_logits = classifier(bert_results['pooled_output'])
```

---

### 🔍 encoder_outputs

```python
print(len(bert_results['encoder_outputs']))
# Sortie: 12
```

**Contenu :**
- Liste des sorties de **chaque couche Encoder**
- 12 tenseurs de forme `(batch_size, seq_length, 768)`

#### 📚 Structure

```python
encoder_outputs[0]  → Sortie de la couche 1  (après 1er Encoder)
encoder_outputs[1]  → Sortie de la couche 2  (après 2ème Encoder)
encoder_outputs[2]  → Sortie de la couche 3
...
encoder_outputs[11] → Sortie de la couche 12 (dernière couche)
```

#### ✅ Vérification

```python
# La dernière couche Encoder = sequence_output
print(bert_results['encoder_outputs'][-1] == bert_results['sequence_output'])
# Sortie: True
```

#### 🔬 Analyse des couches

```python
# Accéder à une couche intermédiaire
layer_6_output = bert_results['encoder_outputs'][5]
print(layer_6_output.shape)  # (2, 7, 768)
```

#### 💼 Utilisation

| Usage | Description |
|-------|-------------|
| **Analyse linguistique** | Étudier ce que chaque couche apprend |
| **Probing tasks** | Tester la connaissance syntaxique/sémantique |
| **Feature extraction** | Combiner plusieurs couches |
| **Visualisation** | Voir l'évolution des embeddings |

---

### 🌊 Pipeline Complet

```
┌────────────────────────────────────────────────────────────┐
│                    PIPELINE BERT COMPLET                   │
└────────────────────────────────────────────────────────────┘

1. Texte brut
   "nice movie indeed"
        ↓

2. Préprocesseur BERT
   ┌─────────────────────┐
   │ • Tokenisation      │
   │ • WordPiece         │
   │ • Ajout [CLS], [SEP]│
   │ • Génération masques│
   └─────────────────────┘
        ↓

3. Entrées numériques
   ┌──────────────────┐
   │ input_word_ids   │ → [101, 3835, 3185, 5442, 102, 0, 0]
   │ input_mask       │ → [1, 1, 1, 1, 1, 0, 0]
   │ input_type_ids   │ → [0, 0, 0, 0, 0, 0, 0]
   └──────────────────┘
        ↓

4. Embeddings
   Token + Position + Segment
        ↓

5. BERT Encoder (12 couches)
   ┌──────────────────┐
   │ Couche 1         │ → encoder_outputs[0]
   │ Couche 2         │ → encoder_outputs[1]
   │ ...              │
   │ Couche 12        │ → encoder_outputs[11]
   └──────────────────┘
        ↓

6. Sorties finales
   ┌──────────────────┬──────────────────┬──────────────────┐
   │ sequence_output  │  pooled_output   │ encoder_outputs  │
   ├──────────────────#   p r o j e c t  
 