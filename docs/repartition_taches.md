# Répartition des tâches — Scanner 3D à triangulation laser

Projet multidisciplinaire HEIG-VD (PMD 2026), équipe de 5.

## Vue d'ensemble

| Membre | Pôle principal | Contributions clés |
|---|---|---|
| **Tom Berthoud** | Gestion + Logiciel (interface) | Chef de projet, interface graphique web |
| **Tim Stamm** | Mécanique | Conception de la boîte, impression 3D, montage laser |
| **Luc Marchand** | Électrique | Câblage, alimentation, intégration électrique |
| **Hadrien Fuentes** | Électronique + Logiciel | PCB, code d'acquisition / 1ʳᵉ caméra |
| **Jason Weber** | Logiciel (algorithmes) | Algorithmes de traitement / 2ᵉ caméra |

## Détail par personne

### Tom Berthoud — Chef de projet & interface graphique
- Coordination de l'équipe, planning, suivi du cahier des charges.
- Conception et développement de l'**interface web** (Flask + SSE, visualiseur STL Three.js, UI de calibration).
- Lien entre les pôles mécanique, électrique et logiciel.

### Tim Stamm — Mécanique
- **Conception CAO** de la boîte et des supports.
- **Impression 3D** de l'ensemble des pièces.
- Intégration du **plateau tournant**, fixation des **caméras** et du **laser**.
- Caisson fermé pour isoler le laser de la lumière ambiante.

### Luc Marchand — Électrique
- **Câblage** complet du système dans la boîte.
- **Alimentation** et distribution de puissance.
- Pilotage du moteur pas à pas et du laser, gestion de la **sécurité laser**.

### Hadrien Fuentes — Électronique & logiciel (1ʳᵉ caméra)
- Conception du **PCB** dédié pour fiabiliser les connexions.
- Développement du **code d'acquisition** et du traitement pour la **première caméra**.
- Extraction de la ligne laser et triangulation côté caméra 1.

### Jason Weber — Logiciel (algorithmes, 2ᵉ caméra)
- **Algorithmes** de traitement et de reconstruction.
- Intégration de la **deuxième caméra** : calibration, triangulation.
- **Fusion** des nuages de points (fusion par pas, fusion demi-tour), nettoyage et reconstruction du maillage.

> Note : la répartition est par dominante ; le travail a été partagé et les pôles se recouvrent (Hadrien sur électronique **et** logiciel, Tom sur gestion **et** logiciel).
