# Diagrammes PlantUML

Sources texte des diagrammes inclus dans le rapport.

## Rendu

`plantuml.jar` est déjà présent dans ce dossier (v1.2024.7).

```bash
# regénérer les PNG après modification d'un .puml
cd docs/rapport/diagrams
java -jar plantuml.jar -tpng -Sdpi=200 *.puml
```

Export PDF nécessite Apache FOP — PNG à 200 DPI donne un rendu propre
dans le rapport et évite cette dépendance.

## Fichiers

| Source | Cible LaTeX | Section |
|---|---|---|
| `architecture_composants.puml` | `architecture_composants.png` | Software — Architecture |
| `machine_etats.puml` | `machine_etats.png` | Software — Orchestration |
| `sequence_scan.puml` | `sequence_scan.png` | Software — Flux de scan |

Les `.png` générés sont inclus via `\includegraphics{diagrams/...}` dans les chapitres.
