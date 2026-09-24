"""Distance bands -> which detector runs, how often, and how often poses are estimated.

    no robot/human pair yet -> 'detect' rates
    FAR (gap > far_m)       -> detector only, slow
    APPROACH                -> poses at a low rate
    CLOSE (gap < close_m)   -> poses at the full rate
Closing faster than fast_closing_mps moves the pair one band closer. Band edges have hysteresis so the rate
does not flap. Each band names its detector, so a heavier or lighter detector can be used per distance.
"""
from dataclasses import dataclass
from typing import Optional

from guardian.types import Band

_KEYS = {Band.FAR: "far", Band.APPROACH: "approach", Band.CLOSE: "close"}


@dataclass
class Schedule:
    band: Optional[Band]                 # None: no pair yet
    detector: str
    detector_hz: float
    pose_hz: float


class Scheduler:
    def __init__(self, cfg):
        self.configure(cfg)
        self.band: Optional[Band] = None

    def configure(self, cfg):
        b = cfg["bands"]
        self.far, self.close, self.hyst = b["far_m"], b["close_m"], b["hysteresis_m"]
        self.fast = b["fast_closing_mps"]
        self.rates = b["rates"]

    def _raw(self, gap):
        if gap < self.close:
            return Band.CLOSE
        return Band.APPROACH if gap < self.far else Band.FAR

    def update(self, gap_m, closing_mps=0.0) -> Schedule:
        if gap_m is None:
            self.band = None
            return self._schedule(None)
        h = self.hyst
        if self.band is None:
            band = self._raw(gap_m)
        elif self.band == Band.FAR:
            band = Band.FAR if gap_m >= self.far - h else self._raw(gap_m)
        elif self.band == Band.APPROACH:
            if gap_m > self.far + h:
                band = Band.FAR
            elif gap_m < self.close - h:
                band = Band.CLOSE
            else:
                band = Band.APPROACH
        else:  # CLOSE
            band = Band.CLOSE if gap_m <= self.close + h else (Band.FAR if gap_m > self.far + h else Band.APPROACH)
        self.band = band
        effective = Band(min(int(band) + 1, int(Band.CLOSE))) if closing_mps > self.fast else band
        return self._schedule(effective)

    def _schedule(self, band):
        r = self.rates["detect" if band is None else _KEYS[band]]
        return Schedule(band=band, detector=r["detector"], detector_hz=float(r["detector_hz"]),
                        pose_hz=float(r["pose_hz"]))

    def reset(self):
        self.band = None
