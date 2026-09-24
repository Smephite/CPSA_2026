"""dashboard: what people see. Draws Snapshots and serves the tuning page. Never decides anything.

Interfaces:
    Dashboard.render(snapshot: Snapshot) -> image (1280x720 BGR)
    Sink          show(image); close()          where the image goes

    views.py      Dashboard: camera overlay | virtual scene (avatars on the joints) | status strip
    sinks.py      MjpegSink (web stream + tuning API), VideoFileSink (mp4), DisplayPortSink (HDMI on the board)
    tuning.py     the live-tunable settings (ranges, help), validation, JSON persistence
    webui.py      the web page (stream + tuning panel)

Upstream CPSA_2026 had this as utils/video_dashboard.py (an OpenCV window; see legacy/).
"""
