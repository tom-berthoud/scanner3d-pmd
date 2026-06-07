# Plan de parole - présentation finale

Durée cible : **4 min 55 s**, vidéo de **35 s comprise**.  
Support : `beamer.tex`, format 16:9.

## Répartition

| Temps | Slide | Personne | Contenu |
|---:|---|---|---|
| 0:00-0:20 | 1. Scanner 3D | Tom | Accroche et contexte |
| 0:20-0:58 | 2. Objectif | Jason | Problème, objectif, chaîne de scan |
| 0:58-1:42 | 3. Architecture | Luc | Matériel, sécurité, interface |
| 1:42-2:40 | 4. Méthode logicielle | Hadrien | Extraction laser, triangulation, fusion |
| 2:40-3:50 | 5. Résultats + vidéo | Tim | Résultats, export, vidéo 35 s |
| 3:50-4:55 | 6. Résumé final | Tom | Conclusion, limites, message final |

## Texte proposé

### Slide 1 - Tom - 20 s

« Notre projet est un scanner 3D par triangulation laser. L'idée est simple : on place un objet dans la boîte, le plateau tourne, le laser projette une ligne, et deux caméras observent cette ligne. La présentation montre la chaîne complète, depuis l'acquisition jusqu'au fichier 3D exploitable. »

### Slide 2 - Jason - 38 s

« L'objectif est de numériser un objet réel avec une solution low-cost et reproductible. Le système doit acquérir un profil laser pour chaque angle du plateau, transformer les pixels de l'image en points 3D, puis exporter un nuage de points et un maillage. Dans la configuration actuelle du code, le scan utilise 100 vues pour un tour complet. Le point important est que chaque étape dépend de la précédente : si l'acquisition ou la calibration est mauvaise, le modèle final se dégrade directement. »

### Slide 3 - Luc - 44 s

« L'architecture combine un plateau motorisé, un laser vert, deux caméras et une interface web. Le moteur est piloté en STEP/DIR avec un NEMA 17 et du microstepping. À chaque position, le laser est allumé uniquement le temps de capturer les images, puis il est coupé. Le code vérifie aussi l'interlock de porte : si la boîte s'ouvre, le scan s'arrête et le laser est éteint. Les deux caméras permettent de récupérer plus de géométrie qu'une seule caméra, en particulier lorsque certaines zones sont cachées. »

### Slide 4 - Hadrien - 58 s

« Côté logiciel, le pipeline est piloté par `run_scan`. Le code lance la capture multi-caméras, puis chaque image passe dans `extract_laser_line`. Cette fonction travaille sur le canal vert, applique un seuil et retourne une position moyenne de la ligne laser pour chaque ligne d'image. Ensuite, `triangulate` reconstruit les points 3D : on part du rayon caméra, on l'intersecte avec le plan laser, puis on compense l'angle de rotation du plateau pour ramener le point dans le repère de l'objet. Enfin, les profils des deux caméras sont fusionnés, les outliers sont filtrés, et le nuage est envoyé vers l'export. »

### Slide 5 - Tim - 70 s, dont 35 s vidéo

« Les résultats montrent les deux sorties principales : le nuage de points et le maillage. Le code exporte le nuage brut en PLY, puis reconstruit une surface avec Open3D et l'exporte en STL ou OBJ. C'est important parce que le projet ne s'arrête pas à une visualisation : il produit un fichier que l'on peut inspecter, partager ou imprimer. Les captures montrent aussi que la qualité dépend beaucoup de l'objet, de la calibration et des occultations. »

Lancer la vidéo ici, durée 35 s.

Après la vidéo : « La vidéo sert à montrer le flux réel : acquisition dans la boîte, suivi dans l'interface, puis visualisation du résultat. »

### Slide 6 - Tom - 65 s

« En résumé, nous avons une chaîne complète : mécanique, sécurité, acquisition, traitement, reconstruction et export. Le coeur technique est la cohérence entre la calibration, la triangulation et la fusion des profils. Le système sait produire un nuage PLY et un maillage STL ou OBJ, donc il va jusqu'à une sortie réellement exploitable. La limite principale reste la qualité de calibration et les zones cachées au laser ou aux caméras. La suite logique serait d'améliorer automatiquement ces zones et de rendre la calibration plus rapide. Le message final est : d'une ligne laser à un objet imprimable. »

## Notes pratiques

- Si la vidéo est disponible, la placer sous `docs/presentation_finale/assets/video_35s.mp4`.
- Si le lecteur PDF ne lit pas la vidéo, garder cette slide ouverte et lancer la vidéo séparément.
- Le texte tient sous 5 minutes avec un débit normal. Répéter une fois avec chronomètre avant la présentation.
