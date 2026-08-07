#!/usr/bin/env bash
set -euo pipefail

# Deploy the committed source tree to UnpackModel LXC 201.
# The container's .env, virtualenv and shared data are intentionally preserved.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXMOX_HOST="${PROXMOX_HOST:-10.20.2.72}"
CONTAINER_ID="${CONTAINER_ID:-201}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-/opt/unpackmodel}"

cd "$PROJECT_DIR"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Ошибка: есть незакоммиченные изменения. Сначала выполните git add/commit." >&2
  exit 1
fi

COMMIT="$(git rev-parse --short HEAD)"
echo "Деплой коммита ${COMMIT} в CT ${CONTAINER_ID} на ${PROXMOX_HOST}"

# git archive contains only tracked files, so .env, .venv and runtime data remain untouched.
git archive --format=tar HEAD \
  | ssh -o StrictHostKeyChecking=accept-new "root@${PROXMOX_HOST}" \
      "pct exec ${CONTAINER_ID} -- tar -x -f - -C '${REMOTE_APP_DIR}'"

ssh -o StrictHostKeyChecking=accept-new "root@${PROXMOX_HOST}" \
  "pct exec ${CONTAINER_ID} -- sh -c 'printf \"%s\\n\" ${COMMIT} > ${REMOTE_APP_DIR}/DEPLOY_VERSION && ${REMOTE_APP_DIR}/.venv/bin/python -m unittest discover -s ${REMOTE_APP_DIR}/tests -q'"

echo "Деплой завершён успешно: ${COMMIT}"
