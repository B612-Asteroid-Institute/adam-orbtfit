import math
import pickle
import random
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from adam_core.coordinates import SphericalCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observers import Observers
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    OrbitDeterminationPhotometry,
)
from adam_core.time import Timestamp

from ..orbfit_orbit_fitter import OrbfitOrbitFitter


@pytest.fixture
def real_data():
    # Actual observations for "2009 JY22"
    obstimes = Timestamp.from_kwargs(
        days=[54952, 54952, 54952, 54952, 54977, 54977, 54977, 54977],
        nanos=[
            15930432000000,
            16879968000000,
            17813088000000,
            18760032000000,
            14122080000000,
            14906592000000,
            15680736000000,
            16459200000000,
        ],
        scale="utc",
    )
    obscodes = ["G96", "G96", "G96", "G96", "G96", "G96", "G96", "G96"]
    lon = [
        173.174080,
        173.173500,
        173.173170,
        173.172420,
        174.067330,
        174.068380,
        174.069290,
        174.070500,
    ]
    lat = [
        7.762110,
        7.760780,
        7.759580,
        7.758310,
        4.412810,
        4.411390,
        4.410170,
        4.408890,
    ]
    obsids = [f"KG0CNl00000055470100001d{i}" for i in range(8)]
    bands = ["V", "V", "V", "V", "V", "V", "V", "V"]
    mags = [21.1, 20.5, 21.2, 21.1, 21.9, 21.2, 21.8, 21.4]

    coords = SphericalCoordinates.from_kwargs(
        lon=lon,
        lat=lat,
        time=obstimes,
        origin=Origin.from_kwargs(code=["SUN"] * 8),
        frame="equatorial",
    )
    observers = Observers.from_codes(codes=obscodes, times=obstimes)

    photometry = OrbitDeterminationPhotometry.from_kwargs(
        mag=mags,
        band=bands,
    )

    observations = OrbitDeterminationObservations.from_kwargs(
        id=obsids,
        coordinates=coords,
        observers=observers,
        photometry=photometry,
    )
    return observations


def test_pickle():
    results_dir = "/some/path"
    fitter = OrbfitOrbitFitter(work_dir=results_dir)
    saved = pickle.dumps(fitter)

    new_fitter = pickle.loads(saved)
    assert new_fitter.work_dir == results_dir


def test_success(real_data):
    observations = real_data
    out_dir = tempfile.TemporaryDirectory(dir=".")
    fitter = OrbfitOrbitFitter(work_dir=out_dir.name, timeout=1000)
    # fitter = OrbfitOrbitFitter(work_dir="test_data2") # out_dir.name)
    object_id = "2009 JY22"
    # Make sure it works with PA class as well
    fitted_orbit, fitted_members = fitter.initial_fit(
        pa.scalar(object_id, type=pa.large_string()), observations
    )
    assert len(fitted_orbit) == 1
    assert fitted_orbit.object_id[0].as_py() == object_id
    assert len(fitted_members) == len(observations)
    outliers = fitted_members.outlier
    solution = fitted_members.solution
    assert pc.invert(outliers) == solution
    outlier_count = np.sum(outliers.to_pylist())
    # Orbfit rejects one of the observations
    assert outlier_count > 0, f"Num outliers {outlier_count}"
    assert outlier_count < len(observations)

    assert fitted_orbit.arc_length[0].as_py() > 0
    assert fitted_orbit.num_obs[0].as_py() == len(observations) - outlier_count


def test_not_enough_data(real_data):
    observations = real_data[:2]
    out_dir = tempfile.TemporaryDirectory()
    fitter = OrbfitOrbitFitter(work_dir=out_dir.name)
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, observations)
    assert len(fitted_orbit) == 0
    assert len(fitted_members) == 0


def test_ra_to_string():
    # 15 degrees per hour == 4 minutes per degree == 0.25 degree per 1 min == ,
    # 'HH MM SS.ddd' format, degrees are [0, 360)
    assert OrbfitOrbitFitter._degrees_to_hms(0.0) == "00 00 00.000"
    assert OrbfitOrbitFitter._degrees_to_hms(307.5) == "20 30 00.000"
    assert OrbfitOrbitFitter._degrees_to_hms(45.0 + 2 + 0.125) == "03 08 30.000"
    assert (
        OrbfitOrbitFitter._degrees_to_hms(45.0 + 2 + 0.125 + 0.25 / 30000)
        == "03 08 30.002"
    )
    # Now a bunch of random ones to make sure we don't hit the '60 seconds' issue again
    random.seed(42)
    for _ in range(10000):
        hours = random.randrange(0, 24)
        minutes = random.randrange(0, 60)
        seconds = random.randrange(0, 60_000) / 1000.0
        degrees = hours * 15 + (minutes + seconds / 60.0) / 4
        assert (
            OrbfitOrbitFitter._degrees_to_hms(degrees)
            == f"{hours:02d} {minutes:02d} {seconds:06.3f}"
        )


def test_dec_to_string():
    # Declination in 'sDD MM SS.dd' format, degrees are [-90, 90]
    assert OrbfitOrbitFitter._degrees_to_dms(-90) == "-90 00 00.00"
    assert OrbfitOrbitFitter._degrees_to_dms(90) == "+90 00 00.00"
    assert OrbfitOrbitFitter._degrees_to_dms(0) == "+00 00 00.00"
    assert OrbfitOrbitFitter._degrees_to_dms(0.00000001) == "+00 00 00.00"
    # If we have a nagative number smaller than precition, print it as +0.
    assert OrbfitOrbitFitter._degrees_to_dms(-0.00000001) == "+00 00 00.00"
    # Now stress it
    random.seed(42)
    for _ in range(10000):
        deg = random.randrange(-90, 90)
        if abs(deg) == 90:
            minutes, seconds = 0, 0.0
        else:
            minutes = random.randrange(0, 60)
            seconds = random.randrange(0, 6000) / 100.0
        degrees = deg + math.copysign(minutes / 60 + seconds / 3600.0, deg)
        assert (
            OrbfitOrbitFitter._degrees_to_dms(degrees)
            == f"{deg:+03d} {minutes:02d} {seconds:05.2f}"
        )
