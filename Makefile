# ============================================================================
# Scanner 3D — Makefile
# Gère le dev local, la connexion au Raspberry Pi, le réseau et les interfaces.
# Surcharge possible en ligne de commande, ex. :  make ssh PI_HOST=192.168.1.42
# ============================================================================

# ---- Connexion Raspberry Pi ----
PI_USER ?= admin
PI_HOST ?= 192.168.55.1
PI_DIR  ?= ~/scanner3d-pmd
PORT    ?= 5000

PI   := $(PI_USER)@$(PI_HOST)
SSH  := ssh $(PI)
URL  := http://$(PI_HOST):$(PORT)/

# ---- Environnement Python local ----
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

# Ouvre une URL dans le navigateur (Linux / macOS)
OPEN := $(shell command -v xdg-open >/dev/null 2>&1 && echo xdg-open || echo open)

.DEFAULT_GOAL := help

.PHONY: help
help: ## Affiche cette aide
	@echo "Scanner 3D — Pi cible : $(PI)   |   Interface : $(URL)"
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		/^[a-zA-Z_-]+:.*## / { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST)

##@ Local (dev sur PC — hardware simulé)
.PHONY: install run run-full unlock-local lock-local test test-fast
install: ## Crée le venv et installe les dépendances
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

run: ## Démarre en mode SCAN seul (verrouillé, = prod) + navigateur
	@scripts/scanner-eng.sh off
	@( sleep 1.5; $(OPEN) http://localhost:$(PORT)/ >/dev/null 2>&1 ) &
	$(PY) -m scanner.interface.web

run-full: ## Démarre avec TOUTES les pages (déverrouillé) + navigateur
	@scripts/scanner-eng.sh on
	@( sleep 1.5; $(OPEN) http://localhost:$(PORT)/ >/dev/null 2>&1 ) &
	$(PY) -m scanner.interface.web

unlock-local: ## Déverrouille à chaud en local (serveur déjà lancé → rafraîchir)
	@scripts/scanner-eng.sh on

lock-local: ## Reverrouille à chaud en local (serveur déjà lancé → rafraîchir)
	@scripts/scanner-eng.sh off

test: ## Lance toute la suite de tests
	$(PY) -m pytest tests/ -v

test-fast: ## Tests de la machine d'états seule (sans deps lourdes)
	$(PY) -m pytest tests/test_state_machine.py -v

##@ Connexion au Raspberry Pi
.PHONY: ssh open
ssh: ## Ouvre une session SSH interactive sur le Pi
	$(SSH)

open: ## Ouvre l'interface du Pi dans le navigateur
	$(OPEN) $(URL)

##@ Déploiement / Production (Pi)
.PHONY: pull pi-install pi-run pi-restart
pull: ## Met à jour le code sur le Pi (git pull)
	$(SSH) 'cd $(PI_DIR) && git pull'

pi-install: ## (Ré)installe le venv et les deps sur le Pi
	$(SSH) 'cd $(PI_DIR) && python -m venv .venv && .venv/bin/pip install -r requirements.txt'

pi-run: ## Lance l'interface web sur le Pi (premier plan)
	$(SSH) 'cd $(PI_DIR) && .venv/bin/python -m scanner.interface.web'

pi-restart: ## Redémarre le service systemd 'scanner' sur le Pi
	$(SSH) 'sudo systemctl restart scanner'

##@ Réseau & debug
.PHONY: ping net-check pi-status pi-logs
ping: ## Vérifie que le Pi répond sur le réseau
	ping -c 3 $(PI_HOST)

net-check: ## Diagnostic réseau côté Pi (IP, route, accès internet)
	$(SSH) 'echo "--- ip a ---"; ip -brief a; echo "--- ip route ---"; ip route; echo "--- ping internet ---"; ping -c 2 8.8.8.8'

pi-status: ## État du service systemd 'scanner'
	$(SSH) 'systemctl status scanner --no-pager'

pi-logs: ## Suit les logs du service 'scanner' (Ctrl-C pour quitter)
	$(SSH) 'journalctl -u scanner -f'

##@ Interfaces — mode ingénierie (Calibration / Extrinsèque / Cam / Manuel)
.PHONY: unlock lock eng-status
unlock: ## Déverrouille les pages d'ingénierie sur le Pi
	$(SSH) '$(PI_DIR)/scripts/scanner-eng.sh on'

lock: ## Reverrouille les pages d'ingénierie sur le Pi
	$(SSH) '$(PI_DIR)/scripts/scanner-eng.sh off'

eng-status: ## État du mode ingénierie sur le Pi
	$(SSH) '$(PI_DIR)/scripts/scanner-eng.sh status'
