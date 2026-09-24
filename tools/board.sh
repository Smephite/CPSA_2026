#!/usr/bin/env bash
# Run a Python script on the KV260 as root in the PYNQ environment (DPU, HDMI), from the repo folder:
#   ./tools/board.sh tools/depth_demo.py              extra arguments go to the script
# Files it writes under out/ are handed back to the user who ran sudo.
set -eo pipefail
if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi
set -a
. /etc/environment
set +a
source /etc/profile.d/pynq_venv.sh
cd "$(dirname "$(readlink -f "$0")")/.."
status=0
python3 "$@" || status=$?
if [ -n "$SUDO_UID" ] && [ -d out ]; then
    chown -R "$SUDO_UID:$SUDO_GID" out
fi
exit $status
