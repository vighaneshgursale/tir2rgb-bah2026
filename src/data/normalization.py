"""Fixed physical normalization for Landsat-9 thermal + optical bands.

Rationale (physics-informed): every tile is normalized against the SAME
brightness-temperature range instead of per-patch min/max stretching. This keeps
the mapping temperature -> pixel value stable across the whole dataset, so the
models learn consistent thermal semantics (water cool, urban hot) and outputs
remain radiometrically interpretable.
"""

import numpy as np

# Landsat Collection-2 Level-2 ST_B10 scaling: Kelvin = DN * 0.00341802 + 149.0
ST_B10_SCALE = 0.00341802
ST_B10_OFFSET = 149.0

# Fixed brightness-temperature window covering Indian scenes across seasons
# (winter Himalaya ~270 K to pre-monsoon peak land-surface temps ~345 K).
K_MIN = 265.0
K_MAX = 345.0

# Landsat C2 L2 SR_B* scaling: reflectance = DN * 0.0000275 - 0.2
SR_SCALE = 0.0000275
SR_OFFSET = -0.2


def dn_to_kelvin(dn):
    """Raw ST_B10 digital numbers -> surface temperature in Kelvin."""
    return dn.astype(np.float32) * ST_B10_SCALE + ST_B10_OFFSET


def kelvin_to_unit(kelvin):
    """Kelvin -> [0, 1] using the fixed window. Same for every tile."""
    return np.clip((kelvin.astype(np.float32) - K_MIN) / (K_MAX - K_MIN), 0.0, 1.0)


def unit_to_kelvin(unit):
    return unit.astype(np.float32) * (K_MAX - K_MIN) + K_MIN


def dn_to_reflectance(dn):
    """Raw SR_B2/B3/B4 digital numbers -> surface reflectance, clipped to [0, 1]."""
    return np.clip(dn.astype(np.float32) * SR_SCALE + SR_OFFSET, 0.0, 1.0)


def reflectance_to_uint8(refl, gain=3.0):
    """Reflectance [0,1] -> display uint8. Fixed gain (typical land reflectance
    is < 0.3), NOT a per-image stretch."""
    return (np.clip(refl * gain, 0.0, 1.0) * 255.0).astype(np.uint8)


def tir_unit_to_uint8(unit):
    return (np.clip(unit, 0.0, 1.0) * 255.0).astype(np.uint8)
