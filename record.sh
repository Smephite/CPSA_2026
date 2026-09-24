#!/usr/bin/env bash
# Record raw webcam clips on the KV260 with a live view on HDMI (and the web stream): tools/record.py.
#   ./record.sh out/clips/reach.avi                 Ctrl+C to stop; writes reach.avi + reach.csv
#   ./record.sh out/clips/walk.avi --duration 30    extra arguments go to tools/record.py (see --help)
# Replay: ./run.sh --source video --video out/clips/reach.avi [--loop | --fast]. Stop run.sh first.
set -eo pipefail
if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi
set -a
. /etc/environment
set +a
source /etc/profile.d/pynq_venv.sh
cd "$(dirname "$(readlink -f "$0")")"
status=0
python3 tools/record.py "$@" || status=$?
if [ -n "$SUDO_UID" ] && [ -d out ]; then
    chown -R "$SUDO_UID:$SUDO_GID" out       # clips belong to the user who ran sudo, not root
fi
exit $status
