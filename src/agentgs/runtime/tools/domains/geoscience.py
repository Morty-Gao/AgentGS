import math

from pydantic import Field, model_validator

from agentgs.contracts.models import Model


class Points(Model):
    lat1: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lon1: float = Field(ge=-180, le=180, allow_inf_nan=False)
    lat2: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lon2: float = Field(ge=-180, le=180, allow_inf_nan=False)


class Magnitude(Model):
    magnitude: float = Field(ge=-5, le=12, allow_inf_nan=False)


class BBox(Model):
    west: float = Field(ge=-180, le=180, allow_inf_nan=False)
    south: float = Field(ge=-90, le=90, allow_inf_nan=False)
    east: float = Field(ge=-180, le=180, allow_inf_nan=False)
    north: float = Field(ge=-90, le=90, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("BBox must be ordered; split antimeridian-crossing regions")
        return self


class CRSList(Model):
    values: list[str] = Field(min_length=1)


class DecimalDegrees(Model):
    value: float = Field(ge=-180, le=180, allow_inf_nan=False)


def distance(p):
    a, b = math.radians(p.lat1), math.radians(p.lat2)
    value = (
        math.sin((b - a) / 2) ** 2
        + math.cos(a) * math.cos(b) * math.sin(math.radians(p.lon2 - p.lon1) / 2) ** 2
    )
    return {
        "distance_km": 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, value)))),
        "method": "spherical_haversine",
    }


def bearing(p):
    a, b = math.radians(p.lat1), math.radians(p.lat2)
    d = math.radians(p.lon2 - p.lon1)
    if p.lat1 == p.lat2 and p.lon1 == p.lon2:
        raise ValueError("Bearing is undefined for identical points")
    return {
        "bearing_degrees": (
            math.degrees(
                math.atan2(
                    math.sin(d) * math.cos(b),
                    math.cos(a) * math.sin(b) - math.sin(a) * math.cos(b) * math.cos(d),
                )
            )
            + 360
        )
        % 360
    }


def energy(p):
    return {"energy_joules": 10 ** (1.5 * p.magnitude + 4.8), "formula": "10^(1.5 Mw + 4.8)"}


def area(p):
    return {
        "area_km2": 6371.0088**2
        * math.radians(p.east - p.west)
        * (math.sin(math.radians(p.north)) - math.sin(math.radians(p.south))),
        "method": "spherical_bbox",
    }


def crs(p):
    values = [v.strip().upper().replace(" ", "") for v in p.values]
    if not all(values):
        raise ValueError("CRS values cannot be empty")
    return {
        "consistent": len(set(values)) == 1,
        "values": values,
        "method": "normalized_label_comparison",
    }


def dms(p):
    degrees = int(abs(p.value))
    minutes_float = (abs(p.value) - degrees) * 60
    minutes = int(minutes_float)
    return {
        "sign": -1 if p.value < 0 else 1,
        "degrees": degrees,
        "minutes": minutes,
        "seconds": (minutes_float - minutes) * 60,
    }
