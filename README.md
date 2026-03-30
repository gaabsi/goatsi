# goatsi

## Description
Une seule promesse : **Occupez-vous de réunir les données, je vous offre le modèle quelques minutes (le temps de calculer)**.  
Goatsi est un wrapper qui permet d'enlever la barrière du code et des connaissances en Machine Learning à la production d'un modèle.  

Vous avez un excel, un csv ou tout autre jeu de données qui sert à décrire une relation ?  
Avec Goatsi en 2 lignes de commande dans votre terminal vous avez un modèle tout prêt.  
D'humeur avanturière ? Avec 2 commandes de plus vous aurez un rapport personnalisé sur les performances et comment fonctionne votre modèle.  

Le but de ce projet est de permettre 2 choses : 
- Tu connais le vollet métier mais tu ne sais pas comment modéliser. 
- Tu sais modéliser mais tu veux aller plus vite. 

Si tu t'es reconnu dans ces quelques lignes ce paquet est fait pour toi !  
Si tu es simplement curieux, essaye-donc ;)

## Prise en main 

La premiere étape est de récuperer ce paquet : 

```bash
pip install # Si tu utilises pip
uvx add # Si tu utilises uv
```

Ça y est, le plus dur est fait maintenant on peut s'amuser ! 

### Split
Le premier module de ce paquet est le module **split**.  
Il s'adresse principalement aux développeurs en data ou aux curieux.  
Le but est simple tu as un jeu de données, tu le sépares en 2 jeux : un d'apprentissage et un de test.  
L'idée est la suivante : tu entraines un modèle avec le jeu d'apprentissage à comprendre comment ton jeu de données fonctionne et une fois qu'il a compris tu l'évalue sur le jeu de test avec des données qu'il n'a jamais vu pour voir comment il s'en sort.  
Il faut voir ça comme un cours, avec le jeu de test le modèle apprend ses leçons et avec celui de test il passe un contrôle pour vérifier si il a juste appris bêtement les exercices (auquel cas il aura une mauvaise note au contrôle) ou s'il les a réellement compris.  

Comment s'utilise ce module ? 

```bash 
goatsi split -filepath --train_size --target --usecol 
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
goatsi fit -trainpath -target -positiveclass
```

Explication : 
- trainpath (obligatoire) : ici tu dis "je veux que mon modèle apprenne de mon jeu d'entrainement et voici où il se trouve".
- target (obligatoire) : c'est ici que tu dis "je veux prédire cette variable-là !"
- positiveclass (pas toujours obligatoire) : ici c'est le seul moment où tu devras te creuser la tête cet argument n'est pas toujours obligatoire et pour savoir s'il l'est dans ton cas pose toi la question suivante : "est ce que ce que je veux prédire je peux y répondre par oui ou par non ?" si c'est oui alors écris la valeur de la classe positive. Si la réponse à ta question est un nombre alors ne remplis pas cet argument ! 

