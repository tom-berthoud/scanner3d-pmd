# scanner3d-pmd

Scanner 3D à triangulation laser — Projet étudiant HEIG-VD, semestre 4, module PMD.

Le système capture des profils d'un objet en rotation sous un laser ligne, reconstruit un nuage de points 3D, et exporte un fichier STL ou OBJ. Tout se passe dans une boîte fermée pilotée par un Raspberry Pi, accessible via une interface web ou un petit écran local.

---

## Comment ça fonctionne

```
1. L'utilisateur pose l'objet sur le plateau et lance un scan
2. Le plateau tourne par pas (200 pas = 360°, 1.8°/pas)
3. À chaque pas : laser allumé → photo → laser éteint
4. Chaque image est analysée pour extraire la ligne laser verte
5. La triangulation géométrique convertit chaque ligne en profil 3D
6. Les 200 profils sont fusionnés en un nuage de points complet
7. Export en fichier STL ou OBJ, téléchargeable via l'interface web
```

**Géométrie de triangulation :**
- Angle laser/caméra : 30° (cible)
- Plan laser : `0.5·x + 0.866·z = 259.8` (repère caméra, mm)
- Centre plateau : Z = 300 mm depuis la caméra
- Intrinsèques de référence : fx = fy = 800 px @ 640×480

**Contraintes du projet :**
- Objets jusqu'à 150 × 150 × 150 mm
- Temps de scan < 10 min
- Prise en main < 10 min
- Budget ~300 CHF
- Boîte fermée (contrôle de la lumière ambiante)

---

## Structure du projet

```
scanner3d-pmd/
├── scanner/
│   ├── hardware/          Contrôle GPIO : moteur, laser, LEDs, écran, caméra
│   │   └── mock.py        Simulation hardware (sans Raspberry Pi)
│   ├── acquisition/       Pipeline de capture image + sauvegarde frames
│   ├── calibration/       Calibration caméra et plan laser
│   ├── processing/        Extraction ligne laser + triangulation 3D
│   ├── reconstruction/    Fusion des profils en nuage de points
│   ├── export/            Génération STL / OBJ (trimesh convex hull)
│   ├── orchestration/     Machine d'états, boucle de scan
│   └── interface/         Serveur web Flask + affichage écran local
├── config/                Paramètres YAML de scan et calibration
├── docs/                  Documentation technique et images de référence
├── tests/                 Tests unitaires et d'intégration
├── mechanics/             Fichiers CAO (FreeCAD)
└── electronics/           Schémas électroniques (KiCad)
```

---

## Démarrage rapide

### Prérequis

- Python 3.11+
- Sur Raspberry Pi : Raspberry Pi OS 64-bit, Pi Camera Module 3

### Installation

```bash
git clone <repo-url>
cd scanner3d-pmd
pip install -r requirements.txt
```

### Lancer l'interface web

```bash
python -m scanner.interface.web
# Interface disponible sur http://localhost:5000  (ou http://<ip-du-pi>:5000)
```

### Développement sans Raspberry Pi (mode mock)

Le module `scanner/hardware/mock.py` simule tout le hardware et s'active automatiquement si les bibliothèques GPIO ne sont pas disponibles.

Formes disponibles dans `config/settings.yaml` (`camera.mock_shape`) :

| Forme | Description |
|---|---|
| `sphere` | Sphère r = 40 mm |
| `cylinder` | Cylindre r = 35 mm, h = 70 mm |
| `cube` | Cube demi-côté = 35 mm |
| `duck` | Canard (corps ellipsoïde + cou + tête + bec) |
| `mushroom` | Champignon (calotte hémisphérique r = 30 mm + pied) |

Pour visualiser les images captées sans lancer un scan complet :
```
http://localhost:5000/preview              ← slider interactif
http://localhost:5000/preview/frame?angle=0
http://localhost:5000/preview/extraction?angle=0   ← avec overlay ligne laser
```

### Lancer les tests

```bash
pytest tests/                          # tests unitaires uniquement
pytest tests/ -m integration           # inclut les tests nécessitant le vrai hardware
```

---

## Interface web — routes disponibles

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Page principale — statut + viewer 3D |
| `/scan/start` | POST | Démarre un scan (thread background) |
| `/scan/status` | GET | JSON : état, progression, dernier fichier |
| `/scan/stream` | GET | Server-Sent Events pour mises à jour en temps réel |
| `/scan/download` | GET | Télécharge le dernier fichier STL/OBJ |
| `/scan/frame/latest` | GET | Dernière image capturée (JPEG, avec overlay) |
| `/scan/frame/<n>` | GET | Image du pas n (0–199) |
| `/preview` | GET | Visualisation interactive du mock caméra |
| `/preview/frame?angle=<rad>` | GET | Image brute à l'angle donné |
| `/preview/extraction?angle=<rad>` | GET | Image avec ligne laser détectée |
| `/calibration` | GET | Page de calibration |
| `/calibration/camera` | POST | Lance la calibration intrinsèque caméra |
| `/calibration/laser` | POST | Lance la calibration du plan laser |
| `/manual` | GET | Page de commande manuelle (laser, moteur, LEDs) |
| `/manual/laser` | POST | Active/désactive le laser |
| `/manual/motor` | POST | Fait avancer le moteur manuellement |
| `/manual/led` | POST | Contrôle manuel des LEDs |

---

## Calibration

Avant le premier scan sur le vrai hardware, deux calibrations sont nécessaires :

1. **Caméra** (intrinsèques) — via l'interface web `/calibration` ou :
   ```bash
   python -m scanner.calibration.camera
   ```
2. **Plan laser** — via l'interface web `/calibration` ou :
   ```bash
   python -m scanner.calibration.laser_plane
   ```

Mode sans damier:
- Si vous voulez desactiver la calibration camera par damier, mettez
  `calibration.use_checkerboard: false` dans `config/settings.yaml`.
- Le systeme utilisera alors des intrinsèques approximees basees sur la
  resolution et `calibration.approx_focal_scale`.

Les résultats sont stockés dans `config/`. Voir `docs/calibration.md` pour la procédure complète.

**Fichiers de configuration clés :**

| Fichier | Contenu |
|---|---|
| `config/settings.yaml` | Paramètres de scan (résolution, pas moteur, seuils...) |
| `config/camera_intrinsics.yaml` | Matrice caméra + distorsion (**ne pas committer**) |
| `config/laser_plane.yaml` | Équation du plan laser `[a, b, c, d]` (**ne pas committer**) |
| `config/platform.yaml` | Point d'axe de rotation, pas/tour (**ne pas committer**) |

---

## Documentation technique

- **[agents.md](agents.md)** — Architecture complète, décisions techniques, conventions de code, machine d'états, interfaces contractuelles entre modules. Document de référence pour l'équipe.
- `docs/` — Images de référence des formes mock, procédures

---

## Sécurité laser

Le laser est un laser ligne vert VLM-520-56 LPO-D45-F40 (520 nm, angle 45°, focale 40 mm).

- Désactivé par défaut au démarrage
- S'éteint automatiquement en cas d'erreur (état `ERROR`)
- Ne jamais regarder directement dans le faisceau
- La boîte doit être fermée pendant le scan

---

## Équipe

Projet à 5, HEIG-VD, module PMD — 2026
