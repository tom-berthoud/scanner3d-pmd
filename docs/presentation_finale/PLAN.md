# Plan de la présentation finale — 5 minutes chrono

Présentation **grand public** (toute la classe + jury), durée **exactement 5 minutes**.
Objectif : repartir de zéro, faire comprendre le projet vite et bien, montrer un résultat.

## Principe d'organisation

- **5 intervenants, ~1 minute chacun.** Chacun parle de son pôle.
- **8 slides** très visuelles, peu de texte : on raconte, on ne lit pas.
- Transitions courtes et nommées (« je passe la parole à… »).
- Tom ouvre (chef de projet) et ferme (interface + démo) : c'est le fil rouge.

## Découpage minute par minute

| # | Slide | Intervenant | Durée | Message à faire passer |
|---|---|---|---|---|
| 1 | Titre | Tom | ~5 s | Annoncer le sujet et l'équipe. |
| 2 | Le projet & cahier des charges | **Tom** | ~50 s | Le besoin (numériser un objet → STL imprimable), pour le MakeLab, les contraintes (150 mm, <10 min, 300 CHF, 2 mm). |
| 3 | Comment ça marche (triangulation) | **Tom** | ~40 s | Plateau qui tourne + laser + caméra → point 3D → nuage → STL. |
| 4 | La mécanique | **Tim** | ~50 s | Caisson noir fermé, plateau motorisé, supports imprimés 3D, CAO. |
| 5 | L'électronique | **Luc** (+ Hadrien) | ~50 s | Raspberry Pi, pilotage moteur/laser, alimentation, câblage, PCB, sécurité laser. |
| 6 | Le logiciel : une caméra | **Hadrien** | ~50 s | Pipeline acquisition→export ; ça marche sur les formes rondes, mais occultation = trous sur les formes anguleuses. |
| 7 | La solution : deux caméras | **Jason** | ~50 s | Deux points de vue, fusion des nuages, nettoyage → couverture améliorée. |
| 8 | Résultats & bilan | **Tom** | ~40 s | Interface web, scan ~8 min, STL/OBJ, budget OK, précision à affiner. Démo + « Merci, questions ? ». |

**Total visé : ~4 min 45 + transitions ≈ 5 min.**

## Temps de parole par personne

| Personne | Slides | Temps cumulé |
|---|---|---|
| Tom | 1, 2, 3, 8 | ~1 min 15 (ouverture + clôture) |
| Tim | 4 | ~50 s |
| Luc | 5 | ~50 s |
| Hadrien | 6 (appui slide 5) | ~50 s |
| Jason | 7 | ~50 s |

## Conseils pour tenir les 5 minutes pile

- **Chronométrer chaque passage** au moins deux fois en répétition.
- Une idée par slide : si on dépasse, c'est qu'on en dit trop.
- Préparer **une phrase de transition fixe** entre chaque intervenant (évite les blancs).
- Garder la **démo très courte** (ou une vidéo de 10 s en secours) : c'est le piège qui fait exploser le temps.
- Slide de secours possible : comparatif mono vs double caméra (déjà dans la version Bornand) si on a de l'avance.

## Notes

- La version **technique détaillée** (pour M. Bornand) est dans `../presentation_finale_bornand/` :
  algorithmes de fusion, calibration d'exposition, validation O1–O9. À ne pas confondre.
- Répartition complète des contributions : `../repartition_taches.md`.
