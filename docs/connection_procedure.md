# Procedure de connexion et execution (Windows <-> Raspberry Pi)

## Objectif
Connecter un Raspberry Pi au PC Windows via cable Ethernet, garder Internet sur le Pi via le Wi-Fi du PC, puis acceder au scanner en SSH et HTTP.

## Resume rapide
- Utiliser **ICS (Internet Connection Sharing)** sur la carte Wi-Fi Windows.
- **Ne pas** utiliser le point d'acces mobile en meme temps.
- Le Pi et le PC doivent etre dans le **meme sous-reseau** Ethernet.
- La route par defaut du Pi doit pointer vers l'IP Ethernet du PC.

## 1. Branchement
1. Connecter le Raspberry Pi au PC avec un cable Ethernet.
2. Verifier que le Wi-Fi du PC est connecte a Internet.

## 2. Configuration Windows (ICS)
1. Ouvrir `Panneau de configuration > Reseau et Internet > Connexions reseau`.
2. Clic droit sur la carte **Wi-Fi** (celle qui a Internet) > `Proprietes`.
3. Onglet `Partage` :
4. Cocher `Autoriser d'autres utilisateurs du reseau a se connecter...`.
5. Selectionner la carte Ethernet branchee au Pi (ex: `Ethernet 4`).
6. Valider.
7. Verifier que le **Point d'acces mobile** Windows est **desactive**.

## 3. Verification des IP
### Cote Windows
Dans PowerShell:
```powershell
ipconfig
```
Repere l'interface Ethernet connectee au Pi (ex: `Ethernet 4`) et note son IPv4 (ex: `192.168.55.20`).

### Cote Raspberry Pi
En SSH ou terminal local:
```bash
ip a
ip route
```
Attendu:
- `eth0` a une IP du meme reseau (ex: `192.168.55.1/24`)
- route par defaut via l'IP Ethernet du PC (ex: `default via 192.168.55.20 dev eth0`)

## 4. Connexion au Raspberry Pi
Depuis Windows:
```powershell
ssh admin@192.168.55.1
```

## 5. Interface scanner
Dans le navigateur Windows:
```text
http://192.168.55.1:5000/
```

## 6. Mise a jour du code
Sur le Raspberry Pi:
```bash
cd ~/scanner3d-pmd
git pull
```

## 7. Checklist de debug
1. `ipconfig` Windows: l'Ethernet vers le Pi a une IPv4.
2. `ip a` Pi: `eth0` est `UP` et a une IPv4 du meme sous-reseau.
3. `ip route` Pi: `default via <IP_PC_ETHERNET>`.
4. `ping <IP_PC_ETHERNET>` depuis le Pi.
5. `ping 8.8.8.8` depuis le Pi.
6. Point d'acces mobile Windows desactive.
