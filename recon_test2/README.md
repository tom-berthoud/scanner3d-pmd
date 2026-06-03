# recon_test2 - reconstruction historique mono-camera

Ce dossier reimplemente l'algorithme utilise autour du 23 avril 2026 pour
generer un STL depuis les nuages PLY du scanner mono-camera.

Le pipeline reprend l'ancien export `mesh_mode: cylindrical`:

1. charge un PLY XYZ;
2. reconstruit des profils 3D ordonnes;
3. trie chaque profil par hauteur `Y`;
4. re-echantillonne les profils sur une grille verticale commune;
5. tisse des quads entre profils voisins;
6. ferme le haut et le bas avec des capuchons;
7. exporte un STL.

Par defaut, les profils sont reconstruits depuis l'ordre du PLY, car l'ancien
export PLY ecrivait le nuage dans l'ordre des profils. Un mode `theta` existe
aussi pour regrouper les points par angle autour de l'axe `Y`.

## Usage

```bash
python recon_test2/reconstruct_legacy.py

# comparer avec un regroupement angulaire
python recon_test2/reconstruct_legacy.py --profile-mode theta

# un seul fichier
python recon_test2/reconstruct_legacy.py --input recon_tests/input/scan_20260503_234722_cloud.ply
```

Les STL sortent dans `recon_test2/output/<mode>/`.
