# goatsi

## Description

Ce module s'adresse à des techniciens de la data et leur permet de livrer en seulement quelques minutes un premier jet de modèle optimisé par rapport a leur jeu de données sans avoir à écrire une ligne de code.  
En seulement quelques lignes de commandes, toutes les étapes de la pipeline classique de machine learning (i.e split, training, evaluation, interpretation) sont réalisées.  
Ce qui permet d'avancer rapidement et de passer la phase de prototypage en quelques secondes.  

Tous les modules de cet outil sont indépendants (bien qu'interconnectés) ainsi si tu veux juste séparer ton dataset en 2 jeux d'entrainement et de test puis repartir tu peux, idem si tu veux juste regarder les performances de ton modele déjà entrainé sur ton jeu de test c'est possible, ...  

Le but est d'aller vite sur le prototypage pour te laisser tout le loisir et le temps pour te casser la tête dans les futures étapes.

## Prise en main 

La premiere étape est de récuperer ce paquet : 

```bash
pip install # Si tu utilises pip
uv tool add # Si tu utilises uv
```

Ça y est, le plus dur est fait maintenant on peut s'amuser ! 

### Split

Le premier module de ce paquet est le module **split**.  
Il s'adresse principalement aux développeurs en data ou aux curieux.  
Le but est simple tu as un jeu de données, tu le sépares en 2 jeux : un d'apprentissage et un de test.  
Les 2 jeux seront écrits dans le même répertoire que je jeu de données parent au format train_set.[ext]

Comment s'utilise ce module ? 

```bash
goatsi split <filepath> [-target <col>] [-train-size 0.8] [-usecols "['col1', 'col2']"]
```

Explication : 
- filepath : c'est le chemin de ton jeu de données. Le seul argument obligatoire. 
- train_size (optionnel) : la taille que tu veux que fasse le jeu d'entrainement (typiquement entre 70 et 80% donc respectivement 0.7 et 0.8)
- target (optionnel) : cet argument optionnel permet de spécifier quelle est la variable que tu veux prédire. Le spécifier permet d'être sur que cette variable soit distribuée de la même manière entre les deux jeux de données.
- usecol (optionnel) : ici tu spécifies les colonnes que tu veux utiliser pour prédire ta variable. Si certaines sont inutiles pour la prédiction tu peux les enlever. 

Exemple : 
Si tu veux utiser un jeu de données qui s'appelle truc.csv, faire un jeu d'entrainement qui représente 80% de ce jeu de données, que tu veux t'assurer que ta variable que tu veux prédire (âge) a la même distribution entre tes deux jeux de données et qu'en plus tu veux garder uniquement les colones nom prénom (et âge du coup) ?? Voici comment on fait : 

```bash 
goatsi split truc.csv 0.8 'âge' ['nom', 'prénom', 'âge']
```

Bon par contre entre nous, prédire l'âge avec le nom et le prénom ça risque d'être compliqué ! 

### Fit 

C'est le module qui te permet d'apprendre à ton modèle grâce à ton jeu d'entrainement !  
Ici tout se passe sous le capot : 
1. On prend ta variable cible et on essaye de l'expliquer avec les autres 
2. Pour ce faire par défaut on utilise l'algorithme qui a les meilleures performances actuellement : XGBoost.  
3. On optimise les paramètres internes du modèle pour qu'ils s'adaptent à ton jeu de données.
4. Tu récupéres le modèle et un petit graphique qui te montre si avec plus de données ton modèle aurait été meilleur. Comme ça tu sais si il faut que tu mettes les bouchées doubles sur la collecte de données ou pas ! 

Et le plus magique dans tout ça ?  
Tout se passe sous le capot, toi tu vois une jolie barre de progression qui va plus ou moins vite selon les performances de ta machine et tu récupères un model tout beau tout propre et optimisé ! C'est pas génial ça ? (si)  

Comment on lance le fit ? c'est tout simple 
```bash
goatsi fit <train_path> -target <col> [-p <positive_class>]
```

Explication : 
- trainpath (obligatoire) : ici tu dis "je veux que mon modèle apprenne de mon jeu d'entrainement et voici où il se trouve".
- target (obligatoire) : c'est ici que tu dis "je veux prédire cette variable-là !"
- positiveclass (pas toujours obligatoire) : ici c'est le seul moment où tu devras te creuser la tête cet argument n'est pas toujours obligatoire et pour savoir s'il l'est dans ton cas pose toi la question suivante : "est ce que ce que je veux prédire je peux y répondre par oui ou par non ?" si c'est oui alors écris la valeur de la classe positive. Si la réponse à ta question est un nombre alors ne remplis pas cet argument ! 


### Eval 

Ce module là sert à évaluer ton modèle, les métriques classiques de l'évaluation (AUC-ROC, accuracy, ...) comme ça tu sais si ton modèle est bon.  
Tu vois aussi les distributions de probabilité, les courbes de performances ... le but ici est de voir en un clin d'oeil si le modèle semble bon pour ta tâche ou pas. 

Comment évaluer ton modèle ? 
```bash
goatsi eval <model_path> <test_path> -target <col> [-p <positive_class>]
```

Explication : 
- modelepath (obligatoire) : ici tu dis "je veux que mon modèle apprenne de mon jeu d'entrainement et voici où il se trouve".
- testsetpath (obligatoire) : le chemin de ton jeu de données de test.
- target (obligatoire) : la variable que tu veux prédire 
- positiveclass (pas toujours obligatoire) : ici c'est le seul moment où tu devras te creuser la tête cet argument n'est pas toujours obligatoire et pour savoir s'il l'est dans ton cas pose toi la question suivante : "est ce que ce que je veux prédire je peux y répondre par oui ou par non ?" si c'est oui alors écris la valeur de la classe positive. Si la réponse à ta question est un nombre alors ne remplis pas cet argument ! 


### Explain 

Ce module sert à savoir comment fonctionne ton modèle.  
On voit à partir de quelles features ton modèle fait ses choix et comment elles affectent ses prédictions.  
Et tu vois 2 cas (classe positive et négative tirés aléatoirement si tâche de classification, top/bottom 25% des prédictions si régression) et on regarde comment le modèle a prédit pour ces 2 cas.

Comment l'utiliser ?
```bash
goatsi explain <model_path> <test_path> -target <col> [-p <positive_class>]
```

Explication :
- model_path (obligatoire) : chemin vers le modèle entraîné (.pkl).
- test_path (obligatoire) : le chemin de ton jeu de données de test.
- target (obligatoire) : la variable que tu veux prédire.
- positive_class (optionnel) : valeur de la classe positive si target catégorielle.

