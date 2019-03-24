Requested Libraries : Numpy, Scipy, Matplotlib, Keras, Tensorflow, SKImage

Pour télécharger les bibliothèques avec les bonnes versions et les meilleurs poids que nous avons obtenu, lancer le shell `init.sh`.

Pour télécharger les images de SUN2012, lancer le shell `download_data.sh` dans le dossier data, le dataset MIT peut être télécharger avec le shell `mit-place-data.sh`.

## Présentation du Travail

Nous proposons une implémentation en Python et Keras du papier de recherche ColorNet (Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. "Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification". )
Nous avons divisé notre code en 4 fichiers.
- `train.py` : c'est le script chargé de l'entrainement. Deux arguments peuvent lui être donné : le nom du dataset (par défaut SUN), le nom du fichier de poids sauvegardés pour reprendre l'entraînement à un checkpoint.
- `data.py` : nos données sont trop grosses pour être stocké dans la mémoire vive, le Data Generator sert à généré des batch de données. Il s'occupe de la préparation des données : data augmentation, passage en noir et blanc ...
- `nn.py` : il construit le réseau de neurones de l'entraînement, ainsi que des méthodes pour l'entrainement, le transfert de style, la sauvegarde ...
- `config.py` : paramètres des différents dataset.