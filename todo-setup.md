# TODO setup - Premier test double camera

Ce fichier liste ce qu'il faut preparer, mesurer ou me fournir avant de lancer
un vrai test double camera, puis quoi verifier pendant les premiers scans.

## 1. Avant le premier test

### Materiel camera

- [x] Confirmer quelle camera correspond a chaque id logiciel:
  - `right` = camera nappe / Pi Camera 3 / CSI.
  - `left` = camera USB Arducam IMX5863 / Arducam B0475.
- [x] Verifier que la camera nappe fonctionne seule.
- [x] Verifier que la camera USB est detectee par le Raspberry Pi.
- [x] Trouver l'index OpenCV de la camera USB (`device_index`).
- [x] Noter le modele exact de la camera USB si possible.
- [x] Trouver ou mesurer sa resolution stable.
- [ ] Verifier si l'exposition manuelle USB fonctionne.
- [ ] Verifier si le gain manuel USB fonctionne.
- [x] Verifier que les deux cameras peuvent capturer dans la meme session.

### Configuration logicielle

- [x] Mettre a jour `config/settings.yaml`.
- [x] Verifier `cameras[0].id`, `type`, `resolution`, `exposure_us`, `gain`.
- [x] Verifier `cameras[1].id`, `type`, `device_index`, `resolution`, `exposure_us`, `gain`.
- [x] Lancer le serveur web.
- [x] Tester:
  - `/manual/camera/frame?camera=right`
  - `/manual/camera/frame?camera=left`
  - `/manual/camera/tuning?camera=right`
  - `/manual/camera/tuning?camera=left`
- [x] Verifier que l'UI affiche bien les etapes:
  - Extraction USB Arducam.
  - Extraction Nappe.
  - Nuage USB Arducam.
  - Nuage Nappe.
  - Nuage combine.
  - STL.

### Securite laser

- [ ] Verifier que le laser est OFF au demarrage.
- [ ] Verifier que le laser s'eteint apres une capture.
- [ ] Verifier que le laser s'eteint en cas d'erreur.
- [ ] Ne pas tester laser actif couvercle ouvert si le rayon peut sortir.
- [ ] Verifier que les nouvelles routines de calibration ne laissent pas le laser ON.

### Position 3D des cameras

Pour un scan precis, il faut connaitre la pose 3D de chaque camera dans un
repere commun lie au plateau.

- [x] Mesurer approximativement la position de la camera nappe:
  - position X/Y/Z par rapport au centre du plateau;
  - angle horizontal par rapport au laser;
  - inclinaison verticale;
  - distance approximative au centre de scan.
- [x] Mesurer approximativement la position de la camera USB:
  - position X/Y/Z par rapport au centre du plateau;
  - angle horizontal par rapport au laser;
  - inclinaison verticale;
  - distance approximative au centre de scan.
- [x] Me fournir ces mesures, meme approximatives.
- [ ] Me fournir des photos du montage:
  - vue de dessus;
  - vue de cote camera nappe;
  - vue de cote camera USB;
  - vue du laser par rapport au plateau.
- [x] Remplacer les extrinseques placeholder dans `settings.yaml`.

### Calibration par camera

- [ ] Calibrer ou preparer la calibration intrinseque camera nappe.
- [ ] Calibrer ou preparer la calibration intrinseque camera USB.
- [ ] Produire ou verifier:
  - `config/camera_intrinsics_right.yaml`
  - `config/camera_intrinsics_left.yaml`
- [ ] Calibrer le plan laser vu par la camera nappe.
- [ ] Calibrer le plan laser vu par la camera USB.
- [ ] Produire ou verifier:
  - `config/laser_plane_right.yaml`
  - `config/laser_plane_left.yaml`
- [x] Confirmer si `calibration.use_checkerboard` doit rester `false` pour un test brut ou passer a `true`.

### Masques d'extraction par camera

Chaque camera aura des zones differentes a ignorer.

- [x] Identifier les zones parasites dans l'image camera nappe:
  - bord de boite;
  - support mecanique;
  - laser direct;
  - reflets fixes;
  - plateau ou fond non utile.
- [x] Identifier les zones parasites dans l'image camera USB.
- [ ] Me fournir pour chaque camera:
  - image brute laser ON sans objet;
  - image brute laser ON avec objet simple;
  - image avec les zones a exclure dessinees si possible.
- [ ] Definir une ROI initiale par camera.
- [x] Definir les rectangles d'exclusion par camera.
- [x] Ajouter ensuite dans le code/config un masque par camera.

### Objet de test

- [ ] Choisir un premier objet simple:
  - sphere ou cylindre pour valider la geometrie;
  - cube/rectangle pour valider les occultations.
- [ ] Mesurer ses dimensions reelles.
- [ ] Noter son orientation de depart sur le plateau.
- [ ] Utiliser un objet mat si possible pour limiter les reflets.

## 2. Pendant le premier test

### Test court d'abord

- [ ] Reduire temporairement `scan.n_steps` a 40 ou 80.
- [ ] Lancer un scan court.
- [ ] Verifier que le scan ne depasse pas les temps attendus.
- [ ] Verifier que le laser s'eteint bien en fin de scan.
- [ ] Verifier que le moteur ne bloque pas et ne perd pas de pas.

### Verification UI et artefacts

Dans l'interface, verifier dans cet ordre:

- [ ] `Extraction USB`:
  - ligne laser visible;
  - ligne continue ou segments coherents;
  - pas trop de reflets parasites;
  - pas de points venant du fond ou du support.
- [ ] `Extraction Nape`:
  - meme controles que pour l'USB.
- [ ] `Nuage USB`:
  - forme plausible;
  - dimensions pas absurdes;
  - pas de grande surface parasite.
- [ ] `Nuage Nape`:
  - forme plausible;
  - dimensions pas absurdes;
  - pas de grande surface parasite.
- [ ] `Nuage combine`:
  - les deux nuages se recouvrent-ils;
  - presence ou non de double surface;
  - decalage visible entre les deux cameras.
- [ ] `STL`:
  - maillage complet ou non;
  - trous;
  - surfaces inventees;
  - deformation due a un mauvais alignement.

### Donnees a garder

Apres chaque scan utile, garder:

- [ ] frames d'extraction USB.
- [ ] frames d'extraction Nape.
- [ ] PLY USB.
- [ ] PLY Nape.
- [ ] PLY combine.
- [ ] STL final.
- [ ] config utilisee.
- [ ] objet scanne et orientation.
- [ ] observations sur le resultat.

## 3. Diagnostic apres les premiers tests

### Si une extraction est mauvaise

- [ ] Ajuster seuil laser.
- [ ] Ajuster exposition camera.
- [ ] Ajuster gain camera.
- [x] Ajouter ou modifier le masque de la camera concernee.
- [ ] Verifier si le laser est trop sature dans l'image.
- [ ] Verifier si la camera voit une reflexion directe.

### Si un nuage camera seul est mauvais

- [ ] Verifier les intrinseques de cette camera.
- [ ] Verifier le plan laser de cette camera.
- [ ] Verifier le point d'axe de rotation.
- [ ] Verifier l'unite des valeurs en millimetres.
- [ ] Verifier que la bonne calibration est associee a la bonne camera.

### Si les deux nuages seuls sont bons mais le combine est mauvais

- [ ] Ne pas regler le STL en premier.
- [ ] Corriger les extrinseques camera vers repere plateau.
- [ ] Comparer le decalage entre `Nuage USB` et `Nuage Nape`.
- [ ] Mesurer translation et rotation approximatives entre les deux nuages.
- [ ] Refaire une calibration avec mire ou objet etalon.

### Si le nuage combine est bon mais le STL est mauvais

- [ ] Ajuster les parametres Poisson.
- [ ] Ajuster le filtrage outlier.
- [ ] Verifier les normales.
- [ ] Verifier si le nuage contient assez de points.
- [ ] Verifier si les trous viennent d'occultations restantes ou du maillage.

## 4. Ameliorations apres les premiers tests

### Masques par camera

- [x] Ajouter officiellement une config `mask` par camera.
- [ ] Supporter:
  - ROI rectangulaire;
  - crop gauche/droite;
  - crop haut/bas;
  - rectangles exclus;
  - sauvegarde par camera.
- [x] Ajouter une page UI pour regler les masques visuellement.

### Calibration extrinseque

- [ ] Ajouter une procedure dediee de calibration camera vers plateau.
- [ ] Utiliser une mire connue posee sur le plateau.
- [ ] Exporter automatiquement:
  - `rotation_matrix`;
  - `translation_mm`.
- [ ] Ajouter un diagnostic d'erreur d'alignement entre cameras.

### Qualite de fusion

- [ ] Garder tous les points au debut.
- [ ] Ajouter ensuite un score de confiance par point:
  - intensite laser;
  - nettete de ligne;
  - distance a la camera;
  - proximite des bords masques.
- [ ] Tester suppression de doublons entre cameras.
- [ ] Tester moyenne locale seulement si cela ne gomme pas les aretes.

### Interface

- [x] Afficher le nombre de points par nuage.
- [x] Afficher l'etat disponible/indisponible de chaque artefact.
- [ ] Ajouter telechargement separe:
  - PLY USB;
  - PLY Nape;
  - PLY combine;
  - STL.
- [x] Ajouter capture d'image de debug depuis l'UI.

## 5. Ce qu'il faut me fournir si tu veux que je continue

- [ ] Photos du montage final ou provisoire.
- [x] Modele exact ou informations disponibles sur la camera USB.
- [x] `device_index` USB trouve.
- [x] Resolution stable des deux cameras.
- [ ] Images brutes de chaque camera:
  - laser OFF;
  - laser ON sans objet;
  - laser ON avec objet simple.
- [ ] Dimensions et photos de l'objet etalon utilise.
- [ ] Les fichiers PLY/STL du premier scan.
- [ ] Capture d'ecran de l'UI sur les six etapes.
- [ ] Observations: extraction mauvaise, nuage decale, STL troue, etc.
