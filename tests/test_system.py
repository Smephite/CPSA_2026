import numpy as np

from sensors.system import SystemLoad, SystemMonitor, split_power


def _stat(path, cores):
    """cores: [(busy, idle)] jiffies per core -> a /proc/stat."""
    lines = ["cpu  1 1 1 1 1 1 1 1 0 0"] + [f"cpu{i} {b} 0 0 {i_} 0 0 0 0 0 0" for i, (b, i_) in enumerate(cores)]
    path.write_text("\n".join(lines) + "\n")


def _ams(root, ps_raw):
    dev = root / "iio:device0"
    dev.mkdir(parents=True)
    (dev / "name").write_text("ams\n")
    for n, key, raw in (("0", "ps", ps_raw), ("2", "pl", 40000)):
        for suffix, v in (("raw", raw), ("offset", 0), ("scale", 1)):
            (dev / f"in_temp{n}_{key}_temp_{suffix}").write_text(f"{v}\n")


def test_cpu_per_core_dpu_share_and_temps(tmp_path):
    stat = tmp_path / "stat"
    _ams(tmp_path / "iio", 52000)
    (tmp_path / "gpu").write_text("suspended\n")
    _stat(stat, [(0, 0), (0, 0)])
    mon = SystemMonitor(str(stat), str(tmp_path / "iio"), str(tmp_path / "gpu"), window_s=1.0)
    first = mon.read(0.0)
    assert first.cpu == () and first.temps == {"ps": 52.0, "pl": 40.0} and first.gpu == "suspended"
    _stat(stat, [(100, 0), (25, 75)])                 # core 0 fully busy, core 1 a quarter
    assert mon.read(0.5, 200.0) is first              # inside the window: cached
    load = mon.read(1.0, 100.0)
    assert np.allclose(load.cpu, (1.0, 0.25)) and np.isclose(load.cpu_cores, 1.25)
    assert np.isclose(load.dpu, 0.3)                  # 300 ms of DPU in 1 s


def test_off_the_board_degrades_quietly(tmp_path):
    mon = SystemMonitor(str(tmp_path / "none"), str(tmp_path / "none"), str(tmp_path / "none"))
    load = mon.read(0.0)
    assert load.cpu == () and load.temps == {} and load.gpu == "n/a"


def test_split_sums_to_the_measured_total():
    model = {"static_w": 5.0, "cpu_core_w": 0.5, "dpu_w": 4.0}
    parts = split_power(8.0, SystemLoad(cpu=(1.0, 1.0, 0.0, 0.0), dpu=0.25), model)
    assert parts == {"static": 5.0, "cpu": 1.0, "dpu": 1.0, "gpu": 0.0, "rest": 1.0}
    assert split_power(None, SystemLoad(), model) is None
