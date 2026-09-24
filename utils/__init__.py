"""utils: shared building blocks with no knowledge of the pipeline's flow.

    types.py      the data types the boxes exchange (Frame, Detection, Danger, Decision, Snapshot, ...)
    settings.py   Guardian's settings: defaults (one comment per value) <- config.yaml `guardian:` section
    geometry.py   2-D helpers: hulls, point/segment distances, boxes
    clock.py      RealClock, SimClock
    event_log.py  EventLog: system log + event diary through upstream's logger.py
    config.py, logger.py, lock.py   upstream CPSA_2026 (config.yaml loading, log files)
"""
