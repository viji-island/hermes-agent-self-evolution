#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f "$HOME/.hermes/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.hermes/.env"
  set +a
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -n "${ANTHROPIC_TOKEN:-}" ]; then
  export ANTHROPIC_API_KEY="$ANTHROPIC_TOKEN"
fi

source .venv/bin/activate
exec python -m evolution.skills.evolve_skill "$@"
