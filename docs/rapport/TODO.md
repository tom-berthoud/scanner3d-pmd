# TODO avant livraison du rapport

Ce fichier regroupe uniquement les informations qui ne peuvent pas être
déduites du code ou qui nécessitent une mesure, une preuve matérielle ou une
nouvelle image. Le code actif (`scanner/` et `config/settings.yaml`) reste la
source de vérité pour le fonctionnement logiciel.

## Mesures et validations

- [ ] Mesurer les dimensions extérieures finales du scanner: largeur,
  profondeur et hauteur en millimètres.
- [ ] Mesurer la masse totale du prototype en kilogrammes.
- [ ] Vérifier le volume utile réel avec un objet ou un gabarit de
  150 x 150 x 150 mm.
- [ ] Mesurer directement la tension reçue sur les broches 5 V du Raspberry Pi:
  au repos, pendant la capture des deux caméras et au maximum du scan. La sortie
  du convertisseur est réglée à 5,5 V pour compenser les chutes de câblage.
- [ ] Photographier l'étiquette ou joindre la fiche fabricant qui confirme la
  classe 1M du laser VLM-520-56.
- [ ] Réaliser au moins trois scans d'une pièce étalon et relever, pour chaque
  axe, la cote de référence, la cote reconstruite, l'erreur absolue et
  l'écart-type.
- [ ] Faire tester le parcours utilisateur par des personnes extérieures à
  l'équipe: nombre d'essais, temps de prise en main, réussites sans aide et
  difficultés rencontrées.
- [ ] Contrôler et documenter l'absence de fuite du faisceau lorsque le caisson
  est fermé.
- [ ] Mesurer le délai maximal entre une ouverture du capot pendant une capture
  et l'extinction effective du laser. Le code interroge actuellement le capteur
  avant et juste après l'activation, sans callback asynchrone continu.

## Images à vérifier ou remplacer

| Fichier | Problème possible | Action recommandée |
|---|---|---|
| `images/boite_vue_ext_ouverte.jpeg` | Photo confirmée obsolète: une seule caméra et câblage intermédiaire. | Ne pas réutiliser comme état final; conserver seulement comme historique explicite. |
| `images/vue_interrieur_dessus_global.jpeg` | Photo confirmée obsolète: une seule caméra et carte perforée. | Ne pas réutiliser comme état final; conserver seulement comme historique explicite. |
| `images/photo_composants.jpg` | Vue des composants, mais pas nécessairement de leur intégration finale. | Conserver comme nomenclature ou remplacer par une photo légendée du câblage monté. |
| `images/boite_finale.jpeg` | Le terme « finale » ne garantit pas que l'interlock, le PCB et les deux caméras soient visibles. | Vérifier la date et le contenu; reprendre une photo extérieure et une intérieure si nécessaire. |
| `images/vue_obsturction_laser.jpeg` | Montage historique mono-caméra. | Conserver seulement avec la légende historique actuelle. |
| `images/montage_2_cameras.jpeg` | Peut montrer une géométrie antérieure aux extrinsèques configurées. | Comparer au montage livré avant réutilisation. |
| `images/ui_accueil_verrouille.png` | Capture susceptible de ne plus correspondre aux routes et artefacts actuels. | Refaire après un scan complet avec nuages PLY et maillage disponibles. |
| `images/ui_accueil_deverrouille.png` | Même risque; image non utilisée actuellement. | Mettre à jour ou supprimer des fichiers du rapport. |
| `images/ui_calibration_intrinseque_laser.png` | L'interface de calibration a évolué vers un plan laser global. | Refaire une capture montrant clairement le damier et la calibration laser actuelle. |
| `images/ui_calibration_extrinseque.png` | Doit correspondre aux réglages et à l'ajustement extrinsèque actuels. | Refaire après validation du workflow final. |
| `images/ui_config_camera.png` | Peut ne pas montrer les résolutions et contrôles actifs. | Vérifier Pi 640x480 et USB 1920x1080. |
| `images/ui_manuel.png` | Doit montrer les deux caméras, masques et seuils actuels. | Refaire si une caméra ou une commande diffère. |
| `images/screen_cylindre_stl.png` | Résultat ancien possible, sans paramètres ni date. | Associer à un scan final reproductible ou indiquer qu'il s'agit d'une illustration. |
| `images/cube_double_camera.png` | Ne prouve ni précision ni répétabilité. | Conserver uniquement comme illustration, ou remplacer par une comparaison mono/double caméra issue du pipeline final. |
| `diagrams/architecture_composants.png` | Était incohérent avec le code et l'écran réel. | Régénéré et contrôlé depuis le fichier PlantUML corrigé. |
| `diagrams/machine_etats.png` | Mentionnait des LED non gérées par la machine d'états. | Régénéré et contrôlé depuis le fichier PlantUML corrigé. |
| `diagrams/sequence_scan.png` | Décrivait 200 pas et une seule caméra. | Régénéré et contrôlé avec 100 positions et deux caméras. |
| `assets/Schéma électrique Scan3D_V02.pdf` | À confronter au PCB final et au câblage réel, notamment au réglage 5,5 V et au capteur de porte. | Corriger le schéma ou ajouter une révision si le montage diffère. |

## Cohérence finale

- [ ] Rechercher dans le rapport les termes `atteint`, `fonctionnel`,
  `démontré`, `corrigé` et `final`; conserver uniquement ceux appuyés par une
  mesure, un test ou le code.
- [ ] Vérifier que toutes les captures d'interface proviennent de la version du
  code livrée.
- [ ] Vérifier que les fichiers de calibration présents dans `config/`
  correspondent au prototype livré et ne contiennent aucune valeur qui ne doit
  pas être publiée.
