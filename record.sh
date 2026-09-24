#!/usr/bin/env bash
# Record raw clips on the KV260 with a live view on HDMI (and the web stream): tools/record.py.
#   ./record.sh out/clips/reach.avi                 Ctrl+C to stop; writes reach.avi + reach.csv
#   ./record.sh out/clips/close.avi --depth         RealSense F200: + close_depth.u16 / .csv
#   ./record.sh out/clips/walk.avi --duration 30    extra arguments go to tools/record.py (see --help)
# Replay: ./run.sh --source video --video out/clips/reach.avi [--loop | --fast]. Stop run.sh first.
exec "$(dirname "$(readlink -f "$0")")/tools/board.sh" tools/record.py "$@"
