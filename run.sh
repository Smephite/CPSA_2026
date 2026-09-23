#!/usr/bin/env bash
# Guardian Node on the KV260: become root, load the PYNQ environment, run the live pipeline.
#   ./run.sh                       webcam -> DPU (YOLOv3-VOC + MoveNet) -> HDMI dashboard, MJPEG on :8080
#   ./run.sh --power --no-audio    extra arguments go to guardian_main.py (see --help)
# Only one process may own the DPU and the webcam: shut down Jupyter kernels first.
set -eo pipefail
if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi
set -a
. /etc/environment
set +a
source /etc/profile.d/pynq_venv.sh
cd "$(dirname "$(readlink -f "$0")")"
exec python3 guardian_main.py --source webcam --backend dpu --hdmi "$@"
