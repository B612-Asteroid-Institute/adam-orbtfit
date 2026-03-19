import pickle
import tempfile

import numpy as np
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
    fitter = OrbfitOrbitFitter(work_dir=out_dir.name)
    # fitter = OrbfitOrbitFitter(work_dir="test_data2") # out_dir.name)
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, observations)
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
    # assert False


def test_not_enough_data(real_data):
    observations = real_data[:2]
    out_dir = tempfile.TemporaryDirectory()
    fitter = OrbfitOrbitFitter(work_dir=out_dir.name)
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, observations)
    assert len(fitted_orbit) == 0
    assert len(fitted_members) == 0
