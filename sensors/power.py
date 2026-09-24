"""Power meters: `read() -> watts or None`. On the KV260: the carrier's INA260 on the SOM supply (total SOM power)."""
import os


class NullPower:
    label = "n/a"

    def read(self):
        return None


class PynqRailsPower:
    """Sum of on-board rail power sensors via pynq.get_rails() (board only; names vary, see README)."""

    label = "W"

    def __init__(self, rails=()):
        import pynq                                   # noqa: PLC0415 (board only)
        all_rails = pynq.get_rails()
        names = rails or [n for n, r in all_rails.items() if getattr(r, "power", None) is not None]
        self.rails = [all_rails[n] for n in names if n in all_rails]
        if not self.rails:
            raise RuntimeError(f"no power rails found (have {sorted(all_rails)})")

    def read(self):
        return float(sum(r.power.value for r in self.rails))


class HwmonPower:
    """Power straight from Linux hwmon sysfs (`power1_input`, microwatts), no libsensors needed.

    On the KV260 this is the carrier's INA260 (`ina260_u14`) on the SOM supply: total SOM power.
    PYNQ's libsensors fails to initialise on the board image, so pynq.get_rails() returns nothing there.
    """

    label = "W"

    def __init__(self, names=("ina260",), root="/sys/class/hwmon"):
        self.files = []
        for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            path = os.path.join(root, d)
            try:
                name = open(os.path.join(path, "name")).read().strip()
            except OSError:
                continue
            f = os.path.join(path, "power1_input")
            if any(name.startswith(n) for n in names) and os.path.exists(f):
                self.files.append(f)
        if not self.files:
            raise RuntimeError(f"no hwmon power sensor matching {list(names)} under {root}")

    def read(self):
        return sum(int(open(f).read()) for f in self.files) / 1e6


def board_power(rails=()):
    """pynq.get_rails() if it finds power sensors, else hwmon sysfs (INA260)."""
    try:
        return PynqRailsPower(rails)
    except Exception:
        return HwmonPower()
