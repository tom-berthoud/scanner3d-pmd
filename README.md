# scanner3d-pmd

Scanner 3D à triangulation laser — Projet étudiant HEIG-VD, semestre 4, module PMD.

Le système capture des profils d'un objet en rotation sous un laser ligne, vu par deux caméras, reconstruit un nuage de points 3D, et exporte un fichier STL ou OBJ. Tout se passe dans une boîte fermée pilotée par un Raspberry Pi, accessible via une interface web ou un petit écran local.

## Rapport

La dernière version compilée du rapport est publiée automatiquement à chaque
push :
[télécharger le PDF](https://github.com/tom-berthoud/scanner3d-pmd/releases/download/rapport-latest/rapport-scanner3d-latest.pdf).

---

## Comment ça fonctionne

```
1. L'utilisateur pose l'objet sur le plateau et lance un scan
2. Le plateau tourne par pas réguliers sur 360° (nombre de photos = `scan.n_steps`, défaut 100)
3. À chaque pas : laser allumé → photo (les deux caméras) → laser éteint
4. Chaque image est analysée pour extraire la ligne laser verte
5. La triangulation géométrique convertit chaque ligne en profil 3D ; les deux
   vues réduisent les zones éclairées mais masquées
6. Tous les profils sont fusionnés en un nuage de points complet
7. Export en fichier STL ou OBJ, téléchargeable via l'interface web
```

**Géométrie de triangulation (deux caméras) :**
- Azimuts caméra/laser : ≈ −25° (caméra `right`) et +33° (caméra `left`)
- Élévations par rapport au plan XZ : ≈ −20° / +20°
- Plateau : centre à ≈ 400 mm des caméras (repère mm)
- Intrinsèques approximées (mode sans damier) : fx = fy ≈ 800 px @ 640×480 (facteur 1.25)

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
│   ├── hardware/          Contrôle GPIO : moteur, laser, écran, caméra
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
├── tests/                 Tests unitaires et d'intégration
└── docs/                  Documentation, rapport et fichiers de fabrication
    ├── rapport/                 Rapport LaTeX (PDF compilé en Release)
    ├── cahier_des_charges/      Cahier des charges (LaTeX) + diagramme FAST
    ├── conception mécanique/    Modèles SolidWorks (.SLDPRT/.SLDASM),
    │                            DXF de découpe laser et STL d'impression 3D (CAD/)
    ├── schema/                  PCB et schéma électronique (KiCad)
    └── connection_procedure.md  Procédure de connexion PC ↔ Raspberry Pi
```

---

## Démarrage rapide

### Prérequis

- Python 3.11+
- Sur Raspberry Pi : Raspberry Pi OS 64-bit, deux caméras pour l'acquisition à
  double vue (Pi Camera Module 3 via CSI + caméra USB)

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
| Local | `make install` / `make test` / `make test-fast` | venv + deps / tous les tests / tests rapides |
| Local | `make run` | lancer l'UI en **mode scan seul** (= prod) + navigateur |
| Local | `make run-full` | lancer l'UI avec **toutes les pages** (déverrouillé) + navigateur |
| Local | `make unlock-local` / `make lock-local` | (dé)verrouiller les pages techniques **en local** |
| Réseau | `make ping` / `make net-check` | Pi joignable / diagnostic réseau du Pi |
| Connexion | `make ssh` / `make open` | session SSH interactive / ouvrir l'UI du Pi dans le navigateur |
| Déploiement | `make pull` / `make pi-install` / `make pi-run` / `make pi-restart` | `git pull` / installer / lancer / redémarrer l'UI sur le Pi |
| Supervision | `make pi-status` / `make pi-logs` | état du service / logs sur le Pi |
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

**Forme simulée** (`config/settings.yaml` → `camera.mock_shape`) : un seul
objet virtuel est disponible, un **`cube`** posé sur le plateau (demi-côté
30 mm, hauteur 60 mm).

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
5. **Écran tactile du Pi** : la page de scan (`/`) est elle-même conçue comme
   une console simplifiée plein écran (gros bouton de scan, navigation
   réduite). Il n'y a pas de mode kiosk séparé ; les pages techniques restent
   masquées tant que le
   [mode ingénierie](#mode-ingénierie--accès-aux-pages-de-configuration)
   n'est pas déverrouillé.

### Lancer automatiquement au démarrage (exemple systemd recommandé)

Le dépôt ne fournit pas de fichier de service, mais les cibles
`make pi-restart` / `pi-status` / `pi-logs` pilotent un service systemd nommé
**`scanner`**. Voici un exemple à créer sur le Pi
(`/etc/systemd/system/scanner.service`) :

```ini
[Unit]
Description=Scanner 3D web interface
After=network.target

[Service]
User=admin
WorkingDirectory=/home/admin/scanner3d-pmd
ExecStart=/home/admin/scanner3d-pmd/.venv/bin/python -m scanner.interface.web
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
POST, donc les actions matérielles laser/moteur comprises) redirigent vers
la page de scan. Il n'y a **aucun mot de passe web** à gérer.

Un développeur, déjà connecté en SSH au Pi, les déverrouille avec le script
fourni :

```bash
# Depuis le PC, en une commande (équivalents : make unlock / lock / eng-status) :
ssh admin@<ip-du-pi> '~/scanner3d-pmd/scripts/scanner-eng.sh on'     # déverrouille
ssh admin@<ip-du-pi> '~/scanner3d-pmd/scripts/scanner-eng.sh off'    # reverrouille
ssh admin@<ip-du-pi> '~/scanner3d-pmd/scripts/scanner-eng.sh status' # état courant
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
  Pi. La page de scan est pensée comme une console plein écran ; la mise en page
  s'adapte à la largeur (desktop, mobile, petit écran tactile).
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
| `/scan/reset` | POST | Réinitialise l'état du scan (retour à `IDLE`) |
| `/scan/status` | GET | JSON : état, progression, dernier fichier |
| `/scan/stream` | GET | Server-Sent Events pour mises à jour en temps réel |
| `/scan/artifacts` | GET | Liste les fichiers produits (nuage, maillage…) |
| `/scan/artifact/<kind>` | GET | Récupère un artefact précis par type |
| `/scan/download` | GET | Télécharge le dernier fichier STL/OBJ |
| `/scan/frame/latest` | GET | Dernière image capturée (JPEG, avec overlay) |
| `/scan/frame/<n>` | GET | Image du pas n |
| `/preview` | GET | Visualisation interactive du mock caméra |
| `/preview/frame?angle=<rad>` | GET | Image brute à l'angle donné |
| `/preview/extraction?angle=<rad>` | GET | Image avec ligne laser détectée |
| `/usb/drives`, `/usb/export` | GET/POST | Liste / copie des fichiers vers une clé USB |
| 🔒 `/calibration` | GET | Page de calibration (caméra + plan laser, par caméra) |
| 🔒 `/calibration/camera/run` | POST | Calcule la calibration intrinsèque (après captures) |
| 🔒 `/calibration/laser/run` | POST | Calcule le plan laser (après captures) |
| 🔒 `/extrinsics` | GET | Réglage de la pose caméra (extrinsèques) |
| 🔒 `/camera-config` | GET | Configuration caméra (format, exposition…) |
| 🔒 `/manual` | GET | Commande manuelle (laser, moteur) |
| 🔒 `/manual/laser` | POST | Active/désactive le laser |
| 🔒 `/manual/motor` | POST | Fait avancer le moteur manuellement |
| 🔒 `/manual/safe-off` | POST | Coupe laser + moteur (arrêt sécurisé) |

> Les pages techniques (`/calibration`, `/extrinsics`, `/camera-config`,
> `/manual`) exposent aussi des sous-routes de session, capture et application
> non listées ici.

---

## Calibration

Le montage utilise **deux caméras** (`left` et `right`). Avant le premier scan
sur le vrai hardware, chacune doit être calibrée :

1. **Caméra** (intrinsèques) — matrice et distorsion, par caméra.
2. **Plan laser** — équation du plan laser dans le repère de chaque caméra.

Tout se fait depuis l'interface web **`/calibration`** (en mode ingénierie), qui
guide chaque étape (session, captures, calcul). Les résultats sont écrits dans
`config/`.

**Mode sans damier** : pour se passer de la calibration caméra par damier, mettez
`calibration.use_checkerboard: false` dans `config/settings.yaml`. Le système
utilise alors des intrinsèques approximées, déduites de la résolution et de
`calibration.approx_focal_scale`.

**Fichiers de configuration clés :**

| Fichier | Contenu |
|---|---|
| `config/settings.yaml` | Configuration globale : caméras, scan, calibration, interface |
| `config/camera_intrinsics_{left,right}.yaml` | Matrice caméra + distorsion, par caméra |
| `config/laser_plane_{left,right}.yaml` | Plan laser par caméra (gitignorés : propres au montage) |
| `config/platform.yaml` | Axe de rotation et pas/tour du plateau |
| `config/*.example.yaml` | Gabarits d'exemple versionnés (à copier puis calibrer) |

---

## Documentation technique

- **[`docs/rapport/`](docs/rapport/)** — Rapport complet : architecture, conception
  mécanique et électronique, chaîne logicielle, résultats. Document de référence.
- **[`docs/connection_procedure.md`](docs/connection_procedure.md)** — Connexion
  PC ↔ Raspberry Pi (Ethernet, SSH, HTTP).
- **`docs/schema/`** — Schéma et PCB KiCad (+ export `Schema connecteurs PCB.png`).
- **`docs/conception mécanique/CAD/`** — Modèles SolidWorks, DXF de découpe laser
  et STL d'impression 3D.

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
