import os

import pytest

from sensors import power as io


def _hwmon(tmp_path, name, micro_watts):
    d = tmp_path / f"hwmon{len(os.listdir(tmp_path))}"
    d.mkdir()
    (d / "name").write_text(name + "\n")
    if micro_watts is not None:
        (d / "power1_input").write_text(f"{micro_watts}\n")
    return d


def test_hwmon_power_reads_ina260_in_watts(tmp_path):
    _hwmon(tmp_path, "ams", None)
    _hwmon(tmp_path, "ina260_u14", 5260000)
    assert io.HwmonPower(root=str(tmp_path)).read() == pytest.approx(5.26)


def test_hwmon_power_without_sensor_raises(tmp_path):
    _hwmon(tmp_path, "ams", None)
    with pytest.raises(RuntimeError):
        io.HwmonPower(root=str(tmp_path))
