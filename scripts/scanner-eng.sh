#!/usr/bin/env bash
# (Dé)verrouille les pages d'ingénierie de l'interface web du scanner 3D.
#
# Le scan reste toujours libre ; Calibration / Extrinsèque / Cam Config / Manuel
# ne sont accessibles que lorsque le fichier sentinelle existe. À lancer via SSH
# depuis le PC d'un développeur :
#
#   ssh pi@scanner '~/scanner3d-pmd/scripts/scanner-eng.sh on'   # déverrouille
#   ssh pi@scanner '~/scanner3d-pmd/scripts/scanner-eng.sh off'  # reverrouille
#
# Le chemin doit correspondre à interface.engineering_unlock_file de
# config/settings.yaml. Par défaut /tmp => reverrouillage automatique au reboot.
set -euo pipefail

UNLOCK_FILE="${SCANNER_ENGINEERING_FILE:-/tmp/scanner-engineering.unlock}"

case "${1:-}" in
  on)
    touch "$UNLOCK_FILE"
    echo "Ingénierie DÉVERROUILLÉE ($UNLOCK_FILE)"
    ;;
  off)
    rm -f "$UNLOCK_FILE"
    echo "Ingénierie verrouillée"
    ;;
  status)
    if [ -e "$UNLOCK_FILE" ]; then echo "déverrouillé"; else echo "verrouillé"; fi
    ;;
  *)
    echo "usage: $0 {on|off|status}" >&2
    exit 1
    ;;
esac
