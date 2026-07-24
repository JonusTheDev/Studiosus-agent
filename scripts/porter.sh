#!/usr/bin/env bash
# The Porter — an always-on `claude remote-control` session, held in tmux, so the
# farmer can reach Brother Claude (grounded on this Studiosus-Agent repo) from the
# Claude mobile app's Code tab at any hour.
#
#   scripts/porter.sh            ensure the studiosus_porter tmux session stands (idempotent)
#   scripts/porter.sh --loop     the foreground restart loop (runs inside the session)
#   scripts/porter.sh --status   say whether the porter stands
#
# Run from WSL. Auto-started at VM boot by the studiosus-porter.service systemd user
# unit (see scripts/porter.service). A faithful sibling of the LuceRenascimur porter —
# a DIFFERENT tmux session name and remote-control name, so both porters can stand on
# the same WSL VM at once without stepping on each other.
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=studiosus_porter
REMOTE_NAME="Studiosus-agent"
REPO="$PWD"

case "${1:-}" in
  --loop)
    # claude lives in ~/.local/bin, which non-login tmux shells don't have.
    export PATH="$HOME/.local/bin:$PATH"
    while true; do
      claude remote-control --name "$REMOTE_NAME" || true
      echo "[porter] remote-control ended ($(date)); the porter returns in 15s..."
      sleep 15
    done
    ;;
  --status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "the porter stands: tmux attach -t $SESSION"
    else
      echo "no porter session — start one with: scripts/porter.sh"
      exit 1
    fi
    ;;
  *)
    loginctl enable-linger "$USER" 2>/dev/null || true
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "the porter already stands ($SESSION)"
    else
      tmux new-session -d -s "$SESSION" "exec bash '$REPO/scripts/porter.sh' --loop"
      echo "the porter takes his post: tmux attach -t $SESSION"
    fi
    ;;
esac
