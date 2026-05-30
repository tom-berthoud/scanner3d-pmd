# scanner3d-pmd

Scanner 3D à triangulation laser — Projet étudiant HEIG-VD, semestre 4, module PMD.

Le système capture des profils d'un objet en rotation sous un laser ligne, reconstruit un nuage de points 3D, et exporte un fichier STL ou OBJ. Tout se passe dans une boîte fermée pilotée par un Raspberry Pi, accessible via une interface web ou un petit écran local.

---

## Comment ça fonctionne

```
1. L'utilisateur pose l'objet sur le plateau et lance un scan
2. Le plateau tourne par pas réguliers sur 360° (nombre de photos = `scan.n_steps`, défaut 100)
3. À chaque pas : laser allumé → photo → laser éteint
4. Chaque image est analysée pour extraire la ligne laser verte
5. La triangulation géométrique convertit chaque ligne en profil 3D
6. Tous les profils sont fusionnés en un nuage de points complet
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
│   ├── export/            Génération STL / OBJ (Open3D Poisson)
│   ├── orchestration/     Machine d'états, boucle de scan
│   └── interface/         Serveur web Flask + affichage écran local
│       ├── templates/     Pages HTML (Jinja2)
│       └── static/        CSS, JS et assets vendorisés (vendor/ — hors-ligne)
├── config/                Paramètres YAML de scan et calibration
├── scripts/               Scripts utilitaires (ex. scanner-eng.sh)
├── docs/                  Documentation technique et images de référence
├── tests/                 Tests unitaires et d'intégration
├── mechanics/             Fichiers CAO (FreeCAD)
└── electronics/           Schémas électroniques (KiCad)
```

---

## Démarrage rapide

### Prérequis

- Python 3.11+
- Sur Raspberry Pi : Raspberry Pi OS 64-bit, Pi Camera Module 3 (+ caméra USB en option)

### Installation

```bash
git clone <repo-url>
cd scanner3d-pmd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Les assets de l'interface (Bootstrap, polices, Three.js) sont **vendorisés**
> dans `scanner/interface/static/vendor/` : aucune connexion internet n'est
> requise au runtime, ni sur le PC, ni sur le Pi.

### Raccourcis (Makefile)

Un `Makefile` regroupe les opérations courantes. `make` (ou `make help`) liste
tout. Le Pi cible est `admin@192.168.55.1` (surchargeable :
`make ssh PI_HOST=…`).

| Catégorie | Commande | Effet |
|---|---|---|
| Local | `make install` / `make test` | venv + deps / lancer les tests |
| Local | `make run` | lancer l'UI en **mode scan seul** (= prod) + navigateur |
| Local | `make run-full` | lancer l'UI avec **toutes les pages** (déverrouillé) + navigateur |
| Réseau | `make ping` / `make net-check` | Pi joignable / diagnostic réseau du Pi |
| Connexion | `make ssh` | session SSH interactive |
| Déploiement | `make pull` / `make pi-run` | `git pull` sur le Pi / lancer l'UI sur le Pi |
| Interfaces | `make open` | ouvrir l'UI du Pi dans le navigateur |
| Ingénierie | `make unlock` / `make lock` / `make eng-status` | (dé)verrouiller les pages techniques **sur le Pi** (SSH) |

---

## Tester en local (sans Raspberry Pi — mode mock)

Le module `scanner/hardware/mock.py` simule tout le hardware et s'active
automatiquement si `gpiozero` n'est pas disponible (donc sur n'importe quel PC).

```bash
source .venv/bin/activate
python -m scanner.interface.web        # http://localhost:5000
```

Sur une machine de dev, débloquez d'emblée les pages d'ingénierie en mettant
`interface.engineering_force: true` dans `config/settings.yaml` (voir
[Mode ingénierie](#mode-ingénierie--accès-aux-pages-de-configuration)).

**Formes simulées** (`config/settings.yaml` → `camera.mock_shape`) :

| Forme | Description |
|---|---|
| `sphere` | Sphère r = 40 mm |
| `cylinder` | Cylindre r = 35 mm, h = 70 mm |
| `cube` | Cube demi-côté = 35 mm |
| `duck` | Canard (corps ellipsoïde + cou + tête + bec) |
| `mushroom` | Champignon (calotte hémisphérique r = 30 mm + pied) |

Visualiser les images captées sans lancer un scan complet :
```
http://localhost:5000/preview                       ← slider interactif
http://localhost:5000/preview/frame?angle=0
http://localhost:5000/preview/extraction?angle=0    ← overlay ligne laser
```

**Tests :**
```bash
pytest tests/ -v                       # tous les tests
pytest tests/test_state_machine.py -v  # machine d'états seule (sans deps lourdes)
pytest tests/ -m integration           # tests nécessitant le vrai hardware
```

---

## Mise en production (Raspberry Pi)

1. **Connexion** PC ↔ Pi (Ethernet + SSH/HTTP) : voir
   [`docs/connection_procedure.md`](docs/connection_procedure.md).
2. **Installer** le projet et les dépendances (cf. *Installation* ci-dessus). Sur
   le Pi, `gpiozero` est présent → les vrais drivers GPIO sont utilisés
   automatiquement.
3. **Calibrer** la caméra et le plan laser au premier déploiement (cf.
   [Calibration](#calibration)).
4. **Lancer** le serveur :
   ```bash
   python -m scanner.interface.web     # écoute sur 0.0.0.0:5000
   ```
   L'interface est alors accessible depuis le PC sur `http://<ip-du-pi>:5000`.
5. **Mode kiosk** (écran tactile sur le Pi) : ouvrir l'interface avec
   `?mode=kiosk` → `http://localhost:5000/?mode=kiosk`. L'UI passe en plein
   écran simplifié (gros bouton de scan, navigation réduite).

### Lancer automatiquement au démarrage (exemple systemd recommandé)

Le projet ne fournit pas de service ; voici un exemple à adapter
(`/etc/systemd/system/scanner.service`) :

```ini
[Unit]
Description=Scanner 3D web interface
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/scanner3d-pmd
ExecStart=/home/pi/scanner3d-pmd/.venv/bin/python -m scanner.interface.web
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now scanner.service
```

> **Port web** : configurable via `interface.web_port` (défaut 5000) et
> `interface.web_host` (défaut `0.0.0.0`) dans `config/settings.yaml`.

---

## Mode ingénierie — accès aux pages de configuration

Pour rester léger en exploitation, **l'interface n'expose que le scan par
défaut**. Les pages techniques (Calibration, Extrinsèque, Cam Config, Manuel)
sont **verrouillées** : leurs onglets sont masqués et leurs routes (GET *et*
POST, donc les actions matérielles laser/moteur/LED comprises) redirigent vers
la page de scan. Il n'y a **aucun mot de passe web** à gérer.

Un développeur, déjà connecté en SSH au Pi, les déverrouille avec le script
fourni :

```bash
# Depuis le PC, en une commande :
ssh pi@<ip-du-pi> '~/scanner3d-pmd/scripts/scanner-eng.sh on'     # déverrouille
ssh pi@<ip-du-pi> '~/scanner3d-pmd/scripts/scanner-eng.sh off'    # reverrouille
ssh pi@<ip-du-pi> '~/scanner3d-pmd/scripts/scanner-eng.sh status' # état courant
```

- Le déverrouillage est pris en compte **immédiatement** (pas de redémarrage).
- Le verrou repose sur un **fichier sentinelle** (`interface.engineering_unlock_file`,
  défaut `/tmp/scanner-engineering.unlock`). Comme il vit dans `/tmp`, le Pi
  **se reverrouille automatiquement au redémarrage** (sûr en prod).
- Sur une machine de dev, `interface.engineering_force: true` débloque tout sans
  script.

---

## Interface web — caractéristiques

- **Thème clair** responsive, lisible sur PC comme sur le petit écran tactile du
  Pi. Trois rendus selon la largeur : desktop, mobile et **kiosk** (`?mode=kiosk`).
- **Hors-ligne** : tous les assets (Bootstrap, icônes, polices, Three.js) sont
  servis localement depuis `static/vendor/` et mis en cache long par le
  navigateur — démarrage rapide même sans internet.
- **Temps réel** : progression du scan poussée par Server-Sent Events ; aperçus
  caméra en flux JPEG à faible latence (polling auto-réamorcé pour ne pas
  saturer le Pi, qualité d'aperçu réduite vs. captures de calibration).
- **Viewer 3D** intégré (Three.js) pour visualiser le nuage / maillage exporté.

## Interface web — routes disponibles

🔒 = nécessite le [mode ingénierie](#mode-ingénierie--accès-aux-pages-de-configuration) déverrouillé.

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Page principale — statut + viewer 3D |
| `/scan/start` | POST | Démarre un scan (thread background) |
| `/scan/status` | GET | JSON : état, progression, dernier fichier |
| `/scan/stream` | GET | Server-Sent Events pour mises à jour en temps réel |
| `/scan/download` | GET | Télécharge le dernier fichier STL/OBJ |
| `/scan/frame/latest` | GET | Dernière image capturée (JPEG, avec overlay) |
| `/scan/frame/<n>` | GET | Image du pas n |
| `/preview` | GET | Visualisation interactive du mock caméra |
| `/preview/frame?angle=<rad>` | GET | Image brute à l'angle donné |
| `/preview/extraction?angle=<rad>` | GET | Image avec ligne laser détectée |
| `/usb/drives`, `/usb/export` | GET/POST | Liste / copie des fichiers vers une clé USB |
| 🔒 `/calibration` | GET | Page de calibration |
| 🔒 `/calibration/camera` | POST | Lance la calibration intrinsèque caméra |
| 🔒 `/calibration/laser` | POST | Lance la calibration du plan laser |
| 🔒 `/extrinsics` | GET | Réglage de la pose caméra (extrinsèques) |
| 🔒 `/camera-config` | GET | Configuration caméra (format, exposition…) |
| 🔒 `/manual` | GET | Commande manuelle (laser, moteur, LEDs) |
| 🔒 `/manual/laser` | POST | Active/désactive le laser |
| 🔒 `/manual/motor` | POST | Fait avancer le moteur manuellement |
| 🔒 `/manual/led` | POST | Contrôle manuel des LEDs |

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
