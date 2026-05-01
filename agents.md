# agents.md — Contexte du projet scanner3d-pmd

Last updated: 2026-03-18 | Implémentation complète : pipeline fonctionnel, interface web, mock avec formes 3D

---

## 1. Identité du projet

Scanner 3D à triangulation laser : une boîte fermée pilotée par un Raspberry Pi qui capture des profils d'un objet en rotation sous un laser ligne, reconstruit un nuage de points, et exporte un fichier STL ou OBJ.

**Cours / contexte :** Projet étudiant HEIG-VD, semestre 4, module PMD, équipe de 5.

**Contraintes dures (non-négociables) :**
- Objets jusqu'à 150 × 150 × 150 mm
- Temps de scan < 10 min
- Temps de prise en main < 10 min
- Export STL ou OBJ obligatoire
- Boîte fermée (contrôle de la lumière ambiante)
- Retour visuel utilisateur (écran + LEDs)
- Communication USB et/ou interface web
- Budget cible ~300 CHF
- Sécurité utilisateur vis-à-vis du laser (règles non-négociables, voir section 8)

---

## 2. Architecture matérielle

| Composant | Statut | Notes |
|---|---|---|
| Raspberry Pi 4 Model B 4GB | **Confirmé** | Quad-core Cortex-A72, 4GB RAM, USB 3.0, Gigabit Ethernet |
| Pi Camera Module 3 **Standard** | **Confirmé** | Sony IMX708, 12MP, F1.8, FOV 66°H/41°V, autofocus I2C **→ verrouillé en manuel** |
| Laser ligne vert VLM-520-56 LPO-D45-F40 | **Confirmé** | 520 nm, angle ligne 45°, focale 40 mm, alim 5V — classe sécurité à vérifier |
| Moteur stepper Generic 17HS3401 | **Confirmé** | NEMA 17, 1.8°/pas, 200 pas/tour, 1.5A/phase, couple 40 N·cm |
| Driver moteur DM320T | **Confirmé** | Microstepping jusqu'à 1/32, 20V max, interface STEP/DIR |
| Écran RB-TFT3.2-V2 | **Confirmé** | 3.2" TFT, interface SPI, compatible Raspberry Pi |
| LEDs | 3 couleurs min. (vert/orange/rouge) | Voir signification section 7 |
| Alimentation | À confirmer (5V/3A USB-C pour le Pi) | Séquence de mise sous tension à définir |

**Point critique caméra — autofocus :**
Le Camera Module 3 a un focus motorisé I2C (Sony IMX708). Il doit être désactivé en mode manuel dès l'initialisation, sinon la calibration est invalidée.

```python
cam.set_controls({"AfMode": 0, "LensPosition": 4.0})  # 0=manuel, ~25cm
```

Ne jamais appeler `cam.autofocus_cycle()` dans le code de scan ou de calibration.

**Réponse spectrale IMX708 — justification laser vert :**
- Canal G : réponse ~95% à 520nm (pic à 530nm) → ligne laser très brillante
- Canal R : réponse ~3% à 520nm → fond quasi noir
- Rapport de contraste G/R ≈ 30:1 → excellente détection
- Technique recommandée : `laser_signal = np.clip(green - red, 0, 255)` pour supprimer la lumière ambiante

**Paramètres géométriques fixés :**
- Angle de triangulation α : **30°** (plage acceptable 25°–35°)
- Distance caméra–centre plateau : **300 mm** (Z_turntable)
- Plan laser (repère caméra) : `0.5·x + 0.866·z = 259.8` → `[a=0.5, b=0, c=0.866, d=-259.8]`
- Intrinsèques de référence (640×480) : fx = fy = 800 px, cx = 320, cy = 240
- Axe de rotation (repère monde) : point (0, 0, 300) mm — stocké dans `config/platform.yaml`

---

## 3. Architecture système

### 3.1 Carte des modules

| Dossier | Responsabilité | Langage |
|---|---|---|
| `scanner/hardware/` | GPIO : moteur, laser, LEDs, écran, caméra | Python |
| `scanner/hardware/mock.py` | Simulation hardware complète pour développement sans Pi | Python |
| `scanner/acquisition/` | Pipeline de capture + sauvegarde frames JPEG | Python |
| `scanner/calibration/` | Calibration intrinsèque caméra + plan laser | Python + OpenCV |
| `scanner/processing/` | Extraction ligne laser, triangulation pixel→3D | Python + OpenCV + NumPy |
| `scanner/reconstruction/` | Fusion des profils, filtrage outliers, nuage de points | Python + NumPy + SciPy |
| `scanner/export/` | Génération STL / OBJ via reconstruction Poisson Open3D | Python + Open3D |
| `scanner/orchestration/` | Machine d'états, boucle de scan principale | Python |
| `scanner/interface/` | Serveur web Flask + SSE + affichage écran local | Python + Flask |
| `config/` | Fichiers YAML de calibration et réglages | YAML |
| `tests/` | Tests unitaires et d'intégration | pytest |
| `docs/` | Documentation technique, images de référence mock | Markdown |
| `mechanics/` | Fichiers CAO, plans mécaniques | FreeCAD (non-code) |
| `electronics/` | Schémas, câblage, BOM | KiCad / PDF (non-code) |

### 3.2 Flux de données — pipeline de scan

```
[Utilisateur → POST /scan/start]
          ↓
orchestration/scan.py — StateMachine: IDLE → SCANNING
  Pour chaque pas i de 0 à 199 :
    1. hardware/motor    → avance d'un pas (+ sync angle → MockCamera)
    2. hardware/laser    → allume le laser
    3. hardware/camera   → capture une image (BGR 640×480)
    4. hardware/laser    → éteint le laser
    5. acquisition       → sauvegarde frame_NNN.jpg + latest.jpg dans /tmp/scan_frames/
          ↓
StateMachine: SCANNING → PROCESSING
  Pour chaque frame i :
    6. processing/laser_line.py → extrait pixels ligne verte (N, 2)
    7. processing/triangulation.py → pixels + calibration + angle → points 3D (N, 3)
          ↓
reconstruction/pointcloud.py — StateMachine: PROCESSING → EXPORTING
  → fusionne tous les profils → nuage de points (M, 3)
  → filtre outliers (KDTree, std_ratio)
          ↓
export/stl.py — StateMachine: EXPORTING → COMPLETE
  → estimation des normales → reconstruction Poisson Open3D → fichier.stl dans /tmp/scans/
          ↓
[SSE push → browser → Three.js STL viewer]
```

### 3.3 Interfaces Python entre sous-systèmes

Ces signatures sont contractuelles. Ne pas les modifier sans mettre à jour ce fichier.

```python
# hardware/__init__.py
def init_hardware(config: dict) -> None: ...
def motor_step(n: int, direction: str) -> None: ...
def laser_set(state: bool) -> None: ...
def camera_capture() -> np.ndarray: ...           # retourne image BGR (H×W×3, uint8)
def led_set(color: str, state: bool) -> None: ...
def led_blink(color: str, duration_s: float) -> None: ...

# acquisition/__init__.py
def run_capture_sequence(
    n_steps: int,
    config: dict,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    save_frames: bool = True,          # sauvegarde /tmp/scan_frames/frame_NNN.jpg
) -> list[np.ndarray]: ...

# processing/__init__.py
def extract_laser_line(
    frame: np.ndarray,
    threshold: int = 180,
    min_pixels: int = 10,
    subpixel: bool = True,
) -> np.ndarray: ...                   # retourne (N, 2) colonnes/lignes pixel

def triangulate(
    line_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    laser_plane: np.ndarray,           # [a, b, c, d] → ax+by+cz+d=0
    rotation_angle_rad: float,
    axis_point: np.ndarray | None = None,  # point sur l'axe de rotation (mm)
) -> np.ndarray: ...                   # retourne (N, 3) points 3D monde

# reconstruction/__init__.py
def merge_profiles(profiles: list[np.ndarray]) -> np.ndarray: ...  # (M, 3)
def filter_outliers(
    cloud: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray: ...

# export/__init__.py
def export_stl(cloud: np.ndarray, path: str) -> None: ...
def export_obj(cloud: np.ndarray, path: str) -> None: ...
```

### 3.4 Routes web Flask

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Page principale — statut + viewer 3D Three.js |
| `/scan/start` | POST | Démarre un scan (thread background) |
| `/scan/status` | GET | JSON : `{state, progress, message, last_file, error}` |
| `/scan/stream` | GET | Server-Sent Events pour mises à jour temps réel |
| `/scan/download` | GET | Télécharge le dernier fichier STL/OBJ |
| `/scan/frame/latest` | GET | Dernière image capturée (JPEG + overlay ligne rouge) |
| `/scan/frame/<n>` | GET | Image du pas n |
| `/preview` | GET | Visualisation interactive du mock caméra |
| `/preview/frame?angle=<rad>` | GET | Image brute mock à l'angle donné |
| `/preview/extraction?angle=<rad>` | GET | Image mock avec ligne laser détectée |
| `/calibration` | GET | Page de calibration |
| `/calibration/camera` | POST | Calibration intrinsèque (upload images damier) |
| `/calibration/laser` | POST | Calibration plan laser |

---

## 4. Décisions technologiques

| Besoin | Choix | Justification |
|---|---|---|
| Traitement image | OpenCV (cv2) | Support calibration caméra, extraction de ligne, compatible Pi |
| Calcul numérique | NumPy | Standard, aucune alternative nécessaire |
| Filtrage outliers | SciPy KDTree | Plus léger qu'Open3D sur Pi, même résultat |
| Export mesh | Open3D Poisson | Reconstruction de surface depuis le nuage de points 3D avec estimation/orientation des normales |
| GPIO | gpiozero (fallback RPi.GPIO) | API haut niveau, plus lisible |
| Interface web | Flask + SSE | Léger, bien connu, SSE pour updates temps réel sans WebSocket |
| Viewer 3D web | Three.js (STLLoader) | Chargement direct du STL binaire, pas de conversion |
| Format config | YAML via PyYAML | Lisible humain, adapté calibration |
| Tests | pytest | Standard Python |
| Version Python | 3.11+ | Type hints modernes |
| Gestion paquets | pip + requirements.txt | Simple, compatible Pi |
| Style code | black (line-length 100) + ruff | Formatage automatique |

---

## 5. Architecture de calibration

### 5.1 Calibration intrinsèque caméra
- **Outil :** damier imprimé, OpenCV `calibrateCamera()`
- **Produit :** matrice caméra 3×3 (fx, fy, cx, cy) + coefficients de distorsion (k1,k2,p1,p2,k3)
- **Stockage :** `config/camera_intrinsics.yaml`
- **Valeurs de référence (mock) :** fx=fy=800, cx=320, cy=240 à 640×480
- **Quand refaire :** si la caméra est déplacée ou le focus modifié

### 5.2 Calibration plan laser
- **Méthode :** surface de référence plane à distance(s) connue(s)
- **Produit :** équation du plan laser dans le repère caméra [a, b, c, d]
- **Valeur implémentée :** `[0.5, 0.0, 0.866, -259.8]` (angle 30°, distance 300mm)
- **Stockage :** `config/laser_plane.yaml`
- **Quand refaire :** si le laser ou la caméra est déplacé

### 5.3 Calibration plateforme
- **Produit :** point sur l'axe de rotation, pas par révolution
- **Valeur implémentée :** `rotation_axis_point_mm: [0, 0, 300]`, `steps_per_revolution: 200`
- **Stockage :** `config/platform.yaml`
- **Quand refaire :** si le moteur ou le plateau est modifié

### 5.4 Règle d'immutabilité
- **Le code de scan ne modifie jamais les fichiers de calibration.**
- La recalibration passe exclusivement par `scanner/calibration/`.
- Les fichiers `*.example.yaml` dans `config/` sont des templates — ne jamais les remplir avec de vraies valeurs.

---

## 6. Conventions de code

### Style
- Formatage : `black --line-length 100`
- Linting : `ruff`
- Type hints obligatoires sur toutes les fonctions et méthodes publiques
- Docstrings Google style sur toutes les fonctions publiques

### Structure des modules
- Chaque dossier de sous-système expose son API via `__init__.py`
- Les helpers internes sont préfixés `_`
- Ordre de dépendance (jamais d'import circulaire) :
  `orchestration` → `acquisition` / `processing` / `reconstruction` / `export` / `hardware` / `interface`

### Gestion des erreurs
- Erreurs hardware → `scanner.hardware.HardwareError`
- Calibration manquante/corrompue → `scanner.calibration.CalibrationError`
- Toutes les erreurs remontées à l'orchestration, jamais swallowed
- Interdit : `except:` sans type d'exception

### Logging
- `import logging; logger = logging.getLogger(__name__)`
- Niveau INFO pour fonctionnement normal, DEBUG pour calibration/diagnostics
- Interdit : `print()` dans le code de production

### Tests
- Tous les appels GPIO mockés dans les tests unitaires
- Images de test dans `tests/fixtures/`
- Tests nécessitant le vrai hardware : marqués `@pytest.mark.integration`

---

## 7. Machine d'états

### États

| État | Description | LEDs |
|---|---|---|
| `IDLE` | Prêt, attente commande utilisateur | Vert fixe |
| `CALIBRATING` | Procédure de calibration en cours | Orange clignotant |
| `SCANNING` | Scan actif (moteur + caméra) | Orange fixe |
| `PROCESSING` | Calcul post-capture (triangulation, fusion) | Orange rapide |
| `EXPORTING` | Écriture fichier STL/OBJ | Orange lent |
| `COMPLETE` | Scan terminé, résultat disponible | Vert clignotant |
| `ERROR` | Erreur non-récupérable | Rouge fixe |

### Transitions valides

```
IDLE        → SCANNING, CALIBRATING
CALIBRATING → IDLE, ERROR
SCANNING    → PROCESSING, ERROR
PROCESSING  → EXPORTING, COMPLETE, ERROR
EXPORTING   → COMPLETE, ERROR
COMPLETE    → IDLE, ERROR          ← reset via StateMachine.reset() pour relancer un scan
ERROR       → IDLE
(tout état) → ERROR
```

**Note implémentation :** `POST /scan/start` appelle `_sm.reset()` si l'état est `COMPLETE` ou `ERROR` avant de lancer le nouveau scan.

---

## 8. Exigences de sécurité (non-négociables)

Ces règles ne peuvent pas être retirées ou affaiblies par un agent.

- **Le laser est désactivé par défaut au démarrage.** Une activation explicite est requise.
- **Le laser se désactive automatiquement si l'état passe à ERROR.**
- **Le laser se désactive si le couvercle de la boîte est ouvert** (si un interlock physique est câblé).
- **Aucune routine de calibration n'active le laser sans confirmation préalable.**
- **Le moteur décélère progressivement** — pas d'arrêt brutal (risque mécanique).
- **Signification des LEDs documentée** dans le manuel utilisateur et ce fichier (voir section 7).
- **Checklist de review :** toute PR touchant `hardware/laser.py` ou `orchestration/` doit inclure la mention "impact laser vérifié".

---

## 9. Workflow de développement

- **Développement sans Pi :** utiliser les classes `Mock*` dans `scanner/hardware/mock.py`
- **Avant tout PR :** `pytest tests/` doit passer à 100%
- **Branches :** `feature/<subsystem>-<description>` ou `fix/<subsystem>-<description>`
- **Commits :** impératif, préfixé par le sous-système — ex: `processing: add RANSAC outlier filter`
- **Jamais de push direct sur `main`** — toujours passer par une PR
- **Nouvelle dépendance :** l'ajouter à `requirements.txt` ET justifier dans la section 4 de ce fichier
- **Calibration :** ne jamais committer des fichiers `config/*.yaml` avec des valeurs réelles — utiliser les `.example.yaml`

---

## 10. Mock hardware — formes disponibles

Le mock (`scanner/hardware/mock.py`) simule la géométrie du scanner par lancer de rayon sur des primitives analytiques. `_surface_y(x_w, z_w, theta)` retourne le Y de la surface supérieure en repère monde.

| `mock_shape` | Géométrie | Dimensions |
|---|---|---|
| `sphere` | Sphère complète | r = 40 mm |
| `cylinder` | Cylindre vertical | r = 35 mm, h = 70 mm |
| `cube` | Cube (face supérieure visible) | demi-côté = 35 mm |
| `duck` | Corps ellipsoïde + cou + tête (sphère) + bec | Corps 26×19×22 mm, tête r=13 mm |
| `mushroom` | Calotte hémisphérique + pied cylindrique | Chapeau r=30 mm, pied r=9 mm h=48 mm |

**Limitation connue :** `_surface_y` modélise uniquement la surface supérieure vue du dessus. Les surfaces strictement verticales (côtés d'un cylindre fin entièrement couvert par un chapeau) ne sont pas capturées. Le chapeau du champignon est bien reconstruit, le pied est partiellement visible selon la géométrie.

**Validation géométrique :** scan sphère → rayon moyen mesuré = 40.00 mm ± 0.09 mm.

---

## 11. Questions ouvertes

| Question | Responsable | Notes |
|---|---|---|
| Classe sécurité du laser VLM-520-56 | Équipe hardware | Modèle confirmé, classe exacte à vérifier sur fiche technique |
| Open3D faisable sur Pi cible | Équipe software | Requis pour l'export STL/OBJ par reconstruction Poisson |
| Export USB : stratégie de montage automatique | Équipe software | Dépend de l'OS choisi |
| Réglage qualité Poisson | Équipe software | Ajuster normales, profondeur et filtrage densité selon le bruit du nuage |
| Alimentation : séquence de mise sous tension | Équipe hardware | Pi + moteur + laser : éviter pic de courant au démarrage |
