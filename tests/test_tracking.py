import numpy as np

from guardian.tracking import Tracker
from guardian.types import Detection


def det(*box):
    return Detection(box=np.array(box, float), score=0.9)


def test_ids_persist_and_roles_lock_leftmost_robot(cfg):
    tr = Tracker(cfg)
    for i in range(20):                            # robot walks past the standing human at 60 px/s
        x = 50 + 30 * i
        tr.update([det(400, 100, 480, 440), det(x, 100, x + 80, 440)], t=i * 0.5)
        tr.assign_roles()
        if i == 2:
            assert len(tr.tracks) == 2 and tr.robot.center[0] < tr.humans[0].center[0]
            robot_id, human_id = tr.robot.id, tr.humans[0].id
    assert (tr.robot.id, tr.humans[0].id) == (robot_id, human_id)
    assert tr.robot.center[0] > tr.humans[0].center[0]


def test_fall_keeps_one_track_and_resets_velocity(cfg):
    tr = Tracker(cfg)
    tr.update([det(400, 200, 480, 440)], t=0.0)
    tr.update([det(402, 200, 482, 440)], t=0.5)
    tid = next(iter(tr.tracks))
    tr.update([det(320, 390, 560, 440)], t=1.0)   # lying: wide flat box, centre jumps down
    assert list(tr.tracks) == [tid]
    np.testing.assert_allclose(tr.tracks[tid].vel, 0.0)


def test_far_detection_starts_a_new_track_and_stale_tracks_drop(cfg):
    tr = Tracker(cfg)
    tr.update([det(0, 200, 60, 440)], t=0.0)
    tr.update([det(560, 200, 620, 440)], t=0.1)
    assert len(tr.tracks) == 2
    tr.update([det(560, 200, 620, 440)], t=0.1 + cfg["tracker"]["max_age_s"] + 0.1)
    assert len(tr.tracks) == 1
