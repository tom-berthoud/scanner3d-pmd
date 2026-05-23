# Procédure de connexion et d'exécution du code

## Connexion au Raspberry Pi

### Étapes pour se connecter en SSH :
1. **Se connecter avec SSH** :
   ```bash
   ssh admin@192.168.55.1
   ```
   - **Mot de passe** : `le mot de passe`

2. **Accéder à l'interface HTTP** (si nécessaire) :
   - Ouvrir votre navigateur et entrer l'URL suivante :
     ```
     http://192.168.55.1:5000/
     ```

3. **Vérifier les services actifs (optionnel)** :
   - Les services `kiosk` (navigateur) et `scanner` (code Python) sont gérés avec `systemd`. Vous pouvez vérifier leur statut :
     ```bash
     systemctl status kiosk.service
     systemctl status scanner.service
     ```
     En général, il n'est pas nécessaire de modifier leurs configurations.

## Configuration réseau

1. Le Raspberry Pi n'a pas d'accès direct à Internet par défaut. Si besoin de connecter le Raspberry à Internet, il faut :
   - **Partager la connexion réseau depuis le PC** en configurant le NAT :
     ```bash
     sudo iptables -t nat -A POSTROUTING -o TON_INTERFACE_WIFI -j MASQUERADE
     ```
   - Vous devez remplacer `TON_INTERFACE_WIFI` par l'interface WiFi du PC connectée à Internet. Sur Windows, un équivalent via interface graphique est nécessaire.

2. **Configurer l'adresse IP du PC :**
   - Changez l'adresse IP de votre PC en mode manuel :
     ```
     192.168.55.100/24
     ```
   - Cela garantit que la route par défaut pointe vers `192.168.55.1` (le Raspberry Pi).

3. **Valider la configuration réseau sur le Raspberry Pi :**
   - Vérifiez les routes :
     ```bash
     ip route
     ```
     Vous devriez voir quelque chose comme :
     ```
     default via 192.168.55.100 dev eth0 proto static metric 100
     192.168.55.0/24 dev eth0 proto kernel scope link src 192.168.55.1 metric 100
     ```

## Clonage et exécution du code

### Cloner le dépôt ou mettre à jour le code :
- Le Raspberry Pi a déjà accès au dépôt Git configuré. Pour le mettre à jour :
  ```bash
  git pull
  ```
- Il n'est pas nécessaire de reconfigurer Git.

### Exécution des scripts :
1. Placez-vous dans le répertoire de travail :
   ```bash
   cd /chemin/vers/le/projet
   ```
2. Lancez le script principal :
   ```bash
   python3 script_principal.py
   ```

### Notes supplémentaires :
- Si une nouvelle bibliothèque est nécessaire et que le Raspberry n'a pas Internet, installez-la localement sur votre PC et copiez-la via `scp` :
  ```bash
  scp -r chemin/vers/bib admin@192.168.55.1:/chemin/destination
  ```