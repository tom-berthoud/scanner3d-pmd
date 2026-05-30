# Approche de passage a deux cameras

## Objectif

Le montage mono-camera actuel perd la ligne laser sur certaines pieces anguleuses, notamment les formes rectangulaires, lorsque la face visible par la camera masque le point d'impact laser. La nouvelle architecture ajoute une seconde camera afin d'observer la meme ligne laser depuis deux directions opposees.

La disposition cible est:

- camera droite du laser en vue de dessus, a 30 degres, position haute, inclinee mecaniquement de 20 degres vers le bas;
- camera gauche du laser en vue de dessus, a 30 degres, position basse, inclinee mecaniquement de 20 degres vers le haut;
- laser unique, moteur et plateau inchanges.

## Faisabilite

La solution est faisable, mais elle transforme le probleme principal: il ne suffit pas de capturer deux images. Chaque camera doit produire des points dans un repere plateau commun. Sinon, deux nuages individuellement corrects peuvent se superposer avec un decalage, ce qui deforme le STL final.

Le code evolue donc vers:

- une configuration `cameras:` contenant les cameras `right` et `left`;
- un driver Pi Camera et un driver USB OpenCV;
- une capture sequentielle par pas moteur: moteur, laser ON, camera droite, camera gauche, laser OFF;
- une calibration separee par camera: intrinseques, plan laser, extrinseques vers le repere plateau;
- une fusion v1 qui conserve tous les points calibres avant filtrage.

Le mode mono-camera reste conserve pour debug, regression et fonctionnement degrade.

## Calibration recommandee

1. Identifier la camera USB recuperee: index OpenCV, resolution reelle, exposition manuelle, gain, stabilite de capture.
2. Calibrer les intrinseques de chaque camera avec le damier imprime ou une mire ChArUco si disponible.
3. Calibrer les extrinseques de chaque camera vers le repere plateau avec une mire rigide placee sur le plateau.
4. Calibrer le plan laser dans le repere de chaque camera.
5. Valider l'alignement avec des pieces etalon: cube/rectangle, cylindre et sphere.

Le diagnostic cle est de comparer le nuage `right` et le nuage `left` avant fusion. L'erreur moyenne, RMS et max doivent rester faibles par rapport a la precision attendue du scanner. Si les deux nuages sont decales, il faut corriger la calibration extrinseque avant de regler le maillage.

## Architecture logicielle

La configuration accepte maintenant un bloc `cameras:`. Les fichiers historiques `camera_intrinsics.yaml` et `laser_plane.yaml` restent utilises comme fallback si les fichiers par camera ne sont pas encore calibres.

Le pipeline devient:

1. charger les modeles de chaque camera;
2. capturer les frames par camera a chaque angle;
3. extraire la ligne laser dans chaque image;
4. trianguler dans le repere camera avec le plan laser correspondant;
5. transformer les points vers le repere plateau avec les extrinseques;
6. deroter selon l'angle du plateau;
7. fusionner tous les profils;
8. filtrer et exporter STL/OBJ comme avant.

## Risques

- La camera USB peut ne pas permettre une exposition manuelle stable.
- Les extrinseques sont le point le plus sensible: une petite erreur angulaire peut produire un decalage visible entre les deux nuages.
- La capture de deux cameras augmente le temps par pas, mais la capture sequentielle est plus fiable qu'une synchronisation logicielle.
- Les angles mecaniques de +/-20 degres doivent rester compatibles avec le champ de vision, la profondeur de champ et la taille maximale 150 mm.
- Toute routine de calibration ou de debug doit conserver le laser OFF par defaut et l'eteindre en cas d'erreur.

## Validation

Les validations minimales sont:

- tests unitaires de triangulation avec extrinseques;
- tests du mock avec deux cameras configurees;
- scan d'un cube pour verifier la reduction des occultations;
- scan d'un cylindre ou d'une sphere pour verifier les dimensions;
- inspection du STL final et du nuage brut PLY.

Un scan ne doit etre considere valide que si les deux vues se recouvrent sans double surface visible dans les zones communes.
