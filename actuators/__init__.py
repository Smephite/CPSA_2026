"""actuators: what the node does about a decision. Audiovisual demo: sound, robot link (logged), event diary.

Interface (every actuator):
    configure(cfg)                 re-read settings (called at start and after live tuning)
    update(decision: Decision, t)  called every frame; act on decision.level / decision.changed

    audio.py          AudioOut (tones via aplay on the HDMI screen), NullAudio
    robot_link.py     LogRobotLink: slow / stop / resume, logged only (future: a real message to the robot)
    event_diary.py    EventDiary: WARN / STOP lines in upstream's event diary + system log
    actuator_manager.py  ActuatorManager: fans each decision out to all of the above

Actuators never look at camera frames or poses: they only see the Decision (utils/types.py).
The visual warning is drawn by `dashboard`, from the same decision carried in the Snapshot.
Upstream's physical actuators (MetaMotion, BT speaker, LED strip) are in legacy/actuators/.
"""
