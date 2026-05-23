# Procédure de connexion et d'exécution du code (Windows)

## Connexion au Raspberry Pi

### Étapes pour se connecter en SSH :
1. **Installer un client SSH (si nécessaire)** :
   - Windows contient déjà un client SSH intégré. Sinon, installez **PuTTY** (ou tout autre client SSH).

2. **Se connecter avec SSH** :
   - Si vous utilisez l'invite de commande ou PowerShell :
     ```cmd
     ssh admin@192.168.55.1
     ```
   - Si vous utilisez PuTTY :
     - Ouvrez PuTTY.
     - Dans "Host Name (or IP address)", entrez : `192.168.55.1`
     - Cliquez sur "Open".
   - **Mot de passe** : `le mot de passe`

3. **Accéder à l'interface HTTP** (si nécessaire) :
   - Ouvrez votre navigateur web et accédez à l'adresse :
     ```
     http://192.168.55.1:5000/
     ```

4. **Vérifier les services actifs (optionnel)** :
   - Les services `kiosk` (navigateur) et `scanner` (code Python) sont gérés automatiquement.

## Configuration réseau

1. **Partager la connexion Internet avec le Raspberry Pi** :
   - Sur Windows, suivez ces étapes :
     1. Allez dans "Paramètres" > "Réseau et Internet" > "Point d'accès mobile".
     2. Activez "Partager ma connexion Internet avec d'autres appareils".
     3. Choisissez "Wi-Fi" comme source de connexion partagée.

   - Ensuite, partagez la connexion avec la connexion Ethernet :
     1. Cliquez sur "Modifier les options de carte réseau".
     2. Faites un clic droit sur l'adaptateur réseau utilisé pour Internet (par exemple, votre Wi-Fi).
     3. Sélectionnez "Propriétés" > "Partage".
     4. Cochez "Autoriser les autres utilisateurs à se connecter via la connexion Internet de cet ordinateur" et sélectionnez la connexion Ethernet connectée au Raspberry Pi.

2. **Configurer une IP statique sur le PC** :
   - Ouvrez "Modifier les options de carte réseau".
   - Trouvez l'adaptateur Ethernet utilisé pour se connecter au Raspberry Pi.
   - Faites un clic droit, sélectionnez "Propriétés", choisissez "Protocole Internet version 4 (TCP/IPv4)", et cliquez sur "Propriétés".
   - Entrez les informations suivantes :
     - **Adresse IP** : `192.168.55.100`
     - **Masque de sous-réseau** : `255.255.255.0`

3. **Vérifiez la connexion au Raspberry Pi** :
   - Pingez l'adresse IP du Raspberry Pi :
     ```cmd
     ping 192.168.55.1
     ```
   - Si le ping fonctionne, la connexion est configurée.

## Clonage et exécution du code

### Cloner le dépôt ou mettre à jour le code :
- Le Raspberry Pi a déjà accès au dépôt Git configuré. Pour mettre à jour le code directement sur le Raspberry Pi :
  ```bash
  git pull
  ```

### Transférer le code depuis le PC :
- Utilisez un logiciel comme WinSCP pour transférer des fichiers de votre PC vers le Raspberry Pi.
  - **Hôte** : `192.168.55.1`
  - **Nom d'utilisateur** : `admin`
  - **Mot de passe** : `le mot de passe`

### Exécution des scripts :
1. Connectez-vous en SSH au Raspberry Pi.
2. Naviguez vers le répertoire du projet :
   ```bash
   cd /chemin/vers/le/projet
   ```
3. Lancez le script principal :
   ```bash
   python3 script_principal.py
   ```