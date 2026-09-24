# main.py

import time

from triggers.constant_tag import ConstantTagSource
from actuators.actuator_manager import ActuatorManager

from core.actuation_policy import StereotipyActivationPolicy
from core.event_dispatcher import EventDispatcher

from VIDEO_pipeline.YOLO.yolo_thread import YoloDpuThread
from VIDEO_pipeline.MOVENET.movenet_thread import MoveNetDpuThread
from VIDEO_pipeline.shared.person_roi_state import PersonRoiState

from utils.logger import log_system
from utils.video_dashboard import (
    VideoDashboard,
    register_dashboard_console,
    unregister_dashboard_console,
)


def main():
    dashboard = None
    trigger_source = None
    actuator_manager = None
    yolo_thread = None
    movenet_thread = None
    dispatcher = None

    try:
        dashboard = VideoDashboard(
            window_name="CPSA Dashboard",
            fullscreen=False
        )

        register_dashboard_console(dashboard)

        log_system("[MAIN] Initializing STOPme system...")

        actuator_manager = ActuatorManager()

        actuator_manager.scan_actuators()

        actuator_manager.initialize_actuators()

        actuators_list = actuator_manager.get_actuators_ids()

        if not actuators_list:
            log_system("[MAIN] No actuators discovered. Event detection and logging still executing")

        policy = StereotipyActivationPolicy(actuator_ids=actuators_list)

        roi_state = PersonRoiState()

        yolo_thread = YoloDpuThread(roi_state=roi_state)
        movenet_thread = MoveNetDpuThread(roi_state=roi_state)

        dispatcher = EventDispatcher(
            actuator_manager=actuator_manager,
            policy=policy,
            yolo_thread=yolo_thread,
            movenet_thread=movenet_thread,
            roi_state=roi_state,
        )

        yolo_thread.start()
        movenet_thread.start()
        dispatcher.start()

        # Replaces the IMU classifier: constant tag 2 drives the YOLO -> MoveNet branch.
        trigger_source = ConstantTagSource(tag=2)
        trigger_source.start()

        log_system("[MAIN] System is now running. Press Ctrl+C or q to terminate.")

        while True:
            dashboard.render(yolo_thread, movenet_thread)

            key = dashboard.wait_key(1)

            if key == ord("q"):
                log_system("[MAIN] GUI quit requested.")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        log_system("[MAIN] Termination signal received.")

    except Exception as e:
        log_system(f"[MAIN] Unhandled error in main loop: {e}", level="ERROR")

    finally:
        if dispatcher:
            dispatcher.stop()

        if yolo_thread:
            yolo_thread.stop()

        if movenet_thread:
            movenet_thread.stop()

        if trigger_source:
            trigger_source.stop()

        if actuator_manager:
            actuator_manager.stop_all()

        log_system("[MAIN] System shutdown complete.")

        if dashboard:
            unregister_dashboard_console()
            dashboard.close()


if __name__ == "__main__":
    main()