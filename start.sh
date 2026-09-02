#!/usr/bin/env bash
#
# start.sh - open the soccer dashboard on macOS or Linux.
#
# From a prompt it takes the same arguments as launch.py:
#
#     ./start.sh play my_team.py --against balanced
#     ./start.sh doctor
#
# All it does is find Python and hand over, so there is one implementation of
# the actual work rather than one per platform.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Some unzip tools drop the executable bit, and then `./start.sh` fails with
# "Permission denied" before anything of ours has had a chance to run. Anyone
# reading this got here through `bash start.sh`, which works either way, so
# restore the bit now and the shorter form works from here on.
[ -x "$0" ] || chmod +x "$0" 2>/dev/null || true

find_python() {
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if ! PYTHON="$(find_python)"; then
    cat >&2 <<'EOF'

  Python 3.8 or newer was not found.

  macOS:  brew install python
          (or install it from https://www.python.org/downloads/)
  Linux:  sudo apt install python3     # or your distribution's equivalent

EOF
    exit 1
fi

exec "$PYTHON" launch.py "$@"
