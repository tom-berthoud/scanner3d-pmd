# recon_tests — banc d'essai de reconstruction STL

Compare plusieurs algorithmes de maillage sur des nuages de points PLY (XYZ)
issus du scanner, pour choisir la meilleure méthode. **Hors du pipeline de
production** : c'est un bac à sable de comparaison.

## Usage

```bash
source .venv/bin/activate

# Met tes nuages dans input/ (déjà fait avec les 5 scans de test)
cp ~/Téléchargements/scan_*_cloud.ply recon_tests/input/

# Toutes les méthodes sur tous les .ply de input/
python recon_tests/reconstruct.py

# Sous-ensemble de méthodes
python recon_tests/reconstruct.py --methods cylindrical,poisson_radial

# Depuis un autre dossier, sans nettoyage du coeur parasite
python recon_tests/reconstruct.py --input ~/Téléchargements --no-clean
```

Les STL sortent dans `output/<méthode>/<nom_du_scan>.stl`. Visualise-les dans
un viewer (MeshLab, Blender, l'aperçu Three.js de l'interface, `f3d`, …).

## Méthodes comparées

| méthode          | principe                                          | pour / contre |
|------------------|---------------------------------------------------|---------------|
| `poisson`        | Poisson, normales par tangent-plane (actuel)      | lisse mais "patate", normales fragiles, **jamais watertight** ici |
| `poisson_radial` | Poisson, **normales orientées radialement** (axe Y connu) | plus fiable que l'actuel ; toujours pas watertight |
| `cylindrical`    | height-field 2.5D : **rayon max** par (θ, y), tissé en quads | watertight, rapide ; bombe légèrement les faces plates |
| `contour`        | **contours 2D + tissage** : intersection rayon/segment en cartésien | watertight, **faces droites + coins préservés** ; le meilleur ici |
| `bpa`            | Ball Pivoting : interpole les points réels        | garde les arêtes ; laisse des trous |
| `alpha`          | Alpha shape : enveloppe non convexe               | simple ; sensible à la densité |

Toutes appliquent d'abord `clean_cloud()` (sauf `--no-clean`) : retrait du
**coeur parasite** (points sur l'axe / plateau, à petit rayon) + Statistical
Outlier Removal. C'est ce nettoyage qui débloque le plus la qualité.

## Méthode retenue : `contour`

`cylindrical` resample le *rayon* à angle fixe et l'interpole linéairement →
les faces plates se bombent (`r=d/cos(θ-θ₀)` n'est pas linéaire) et les coins
se chanfreinent. `contour` reconstruit, par tranche, le **contour 2D réel** et
l'échantillonne par **intersection rayon/segment en cartésien** : le segment
entre deux points d'une face plate EST la face → faces droites exactes, coins
préservés. Tissage en quads + capuchons → watertight par construction.

Étapes de propreté (par tranche, dans cet ordre) :
1. **Rebord extérieur** (`rim_bins`, `rim_margin`) : par secteur angulaire on
   garde les points à moins de `rim_margin` mm du rayon max. Un mur fin garde
   tous ses points (faces planes), mais une **face horizontale pleine** (dessus
   / dessous, vue comme un disque rempli) est réduite à son rebord → fini le
   *chaos aux capuchons* (sans ça, les tranches haute/basse partaient en
   étoile). C'est la correction la plus visible.
2. **Médian circulaire** (`median_w`) : enlève les pics isolés sans toucher
   aux arêtes.
3. **Lissage bilatéral** (`smooth`, défaut 2.0) : le **bouton « beauté »**. Il
   lisse le bruit de surface (aspect granuleux des pièces rondes) tout en
   **préservant coins et marches** (une marche >> `smooth` mm n'est pas lissée,
   contrairement à une moyenne qui l'écraserait). `smooth=0` = fidèle mais
   facetté ; `smooth≈2` = surfaces belles et lisses, toujours watertight et à
   arêtes nettes.

> Note : les « dents » qu'on croit voir sur une coupe en superposant les
> sommets d'une *bande* de hauteur sont un **artefact de tracé** (on mélange
> plusieurs rangées d'une face en pente). Une coupe d'**une seule rangée** du
> maillage est parfaitement droite — voir `renders/CMP_truecut.png`.

Voir `renders/CMP_crosssection_boxy.png` (coupe pièce carrée),
`renders/CUR_cut_bottom.png` (capuchon bas avant/après rim) et
`renders/contour/` (rendus 3D des 5 scans).

## Métrique de fit

Le récap affiche `fit_moy` / `fit_p95` = distance (mm) des points du nuage à la
surface reconstruite. ⚠️ Elle mesure « la surface passe-t-elle près des
points » mais **ne pénalise ni les trous ni la géométrie inventée** : Poisson
a un excellent `fit` tout en n'étant pas imprimable (bosses dans les zones non
scannées, non vues par cette distance). Le bon critère = **watertight ET
`fit` bas**. Sur ce critère, `contour` ≥ `cylindrical` partout, surtout sur les
pièces carrées / étagées.

Résultats (méthodes watertight, `fit_moy` / `fit_p95` en mm, plus bas = mieux) :

| scan                 | forme        | cylindrical | contour (moy / p95) |
|----------------------|--------------|-------------|---------------------|
| scan_…_233338        | carré/étagé  | 0.98        | **0.99 / 1.63**     |
| scan_…_234244        | L / escalier | 0.59        | **0.50 / 2.05**     |
| scan_…_234722        | rond         | 0.17        | **0.11 / 0.52**     |
| scan_…_230608        | rond         | 0.14        | **0.10 / 0.51**     |
| scan_…_232211        | rond         | 0.14        | **0.10 / 0.44**     |

Le nettoyage des capuchons (rim) fait surtout baisser le **p95** (les pires
écarts étaient sur les faces dessus/dessous) et améliore les pièces rondes.

⚠️ Limite commune (cylindrical, contour) : modèle « height-field » → **une
seule surface par (θ, y)**, donc pas de contre-dépouilles. De toute façon un
scan mono-caméra sur tour ne *voit* pas les contre-dépouilles (occlusion) —
c'est une limite de la donnée, pas seulement de l'algo. Le côté « dentelé » des
faces = bruit réel du nuage ; le lisser verticalement améliore les ronds mais
dégrade les carrés (écrase les marches), donc pas activé par défaut.

L'axe de rotation supposé est **Y** (vertical), comme dans le scanner.

## Vérification visuelle headless

`render.py` rend 4 vues (face/côté/dessus/iso) d'un `.stl`/`.ply` en PNG via
Open3D EGL (sans écran). ⚠️ La vue *dessus* ombrée est trompeuse sur les objets
effilés — préférer la vue *iso* ou une coupe transversale pour juger.

```bash
python recon_tests/render.py 'recon_tests/output/contour/*.stl' --outdir recon_tests/renders/contour
```
