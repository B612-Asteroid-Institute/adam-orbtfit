import logging
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import astropy
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin
from adam_core.coordinates.covariances import _upper_triangular_to_full
from adam_core.orbit_determination.evaluate import (
    FittedOrbitMembers,
    FittedOrbits,
    OrbitDeterminationObservations,
)
from adam_core.orbit_determination.orbit_fitter import OrbitFitter
from adam_core.time import Timestamp
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)

STANDARD_OPTIONS = """! Input file for neocp run
neocp.
       .ons_name=.T.                ! NEOCP or provisional/number designation
       .obsdir='mpcobs'             ! directory \of observations
       .error_model='gaiaDR2_mix'   ! error model file name
       .precob=.FALSE.              ! Precedence of rwo file
       .background=.FALSE.           ! Only on\e grid
       .neocp=.FALSE.               ! No variant orbits
       .t_std= 59800.0            ! Standard epoch
       .tno= .F.            ! TNO 
propag.
       .iast=0                    ! 0=no asteroids with mass n=no. of massive asteroids
       .filbe=' '              ! massive asteroids file
       .ephem_file='JPLDE431'      ! ephemerides file
       .npoint=600                 ! minimum number of data points for a deep close appr
       .dmea=0.2d0                 ! min. distance for control close-app. to Earth only
       .dter=0.05d0                ! min. distance for control close-app. to M, V, M
reject.
       .rejopp = .F.    ! reject entire oppositions
       .rej_fudge= .F.  ! fudge not used
IERS.
       .extrapolation=.T. ! extrapolation of Earth rotation
"""


class OrbfitOrbitFitter(OrbitFitter):
    """Implementation of OrbitFitter using Orbfit."""

    def __init__(self, work_dir: str):
        """Constructor for OrbfitOrbitFitter

        Parameters:
        -----------
        work_dir: str
          path to directory to be used for input and output of Orbfit
        """
        self.work_dir = work_dir

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def initial_fit(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        clean_id = object_id.replace(" ", "")
        path, executable = self._setup_work_dir(clean_id)
        self._post_observations(path, clean_id, observations)
        result = subprocess.run(
            f"docker run -v {os.path.abspath(path)}:/Workspace --name orbfit --entrypoint='' --rm minorplanetcenter/orbfit:latest bash {executable}",
            shell=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error(f"Process failed with {result}")
            return FittedOrbits.empty(), FittedOrbitMembers.empty()
        # Extract members first, so that we can set arc length and number of observations in the fitted orbits
        members, selected_times = self._extract_members(path, clean_id, observations)
        orbits = self._extract_orbit(path, clean_id, object_id, selected_times)
        # Fitted members would still have data even if no orbit was found, so check here
        if len(orbits) == 0:
            members = FittedOrbitMembers.empty()
        return orbits, members

    def _setup_work_dir(self, clean_id: str) -> Tuple[Path, str]:
        """Create and setup input/output directory

        Parameters:
        -----------
        clean_id: str
          object id with spaces removed, used in paths

        Returns:
        --------
        Path to the input/output directory outside of the Orbfit container.
        Absolute path to the executable shell script inside the container
        """
        dir = Path(self.work_dir) / clean_id
        logger.debug(f"Make dir {dir}")
        os.makedirs(dir, exist_ok=True)
        with open(dir / "input", "w", encoding="utf-8") as infile:
            infile.write(clean_id)
        # Make an output directory, otherwise Orbfit complains
        os.makedirs(dir / "epoch", exist_ok=True)
        # Add the correct option file
        options = dir / "neocp.nopt"
        if not os.path.exists(options):
            with open(options, "w") as file:
                file.write(STANDARD_OPTIONS)
        # Create executable shell script to run inside the container
        script_path = dir / "run_this.sh"
        with open(script_path, "w") as file:
            file.write("cd /Workspace\n")
            file.write("ln -sf /sa/god_fit/bin/neocp_prelim.x ./neocp_prelim.x\n")
            file.write("./neocp_prelim.x < input\n")
        return dir, "/Workspace/run_this.sh"

    def _post_observations(
        self, path: Path, clean_id: str, observations: OrbitDeterminationObservations
    ):
        dir = path / "mpcobs"
        os.makedirs(dir, exist_ok=True)
        # Orbfit will create this file, but if this file is already there, Orbfit is not happy
        rwo_file = dir / f"{clean_id}.rwo"
        if os.path.exists(rwo_file):
            os.remove(rwo_file)

        mpc1992 = self._mpc1992(clean_id, observations)
        with open(dir / f"{clean_id}.obs", "w", encoding="utf-8") as infile:
            infile.write(mpc1992)

    def _extract_orbit(
        self, path: Path, clean_id: str, object_id: str, selected_times: List[float]
    ) -> FittedOrbits:
        postfit_file = path / "epoch" / f"{clean_id}.eq0_postfit"
        if not postfit_file.exists():
            logger.error(f"Solution file {postfit_file} not found")
            return FittedOrbits.empty()
        with open(postfit_file, "r") as f:
            lines = f.readlines()
        # The file contains several coordinate options. Skip to Cartesian
        curr_line = 0
        while curr_line < len(lines) and not lines[curr_line].startswith(
            "! Cartesian position and velocity vectors"
        ):
            curr_line += 1
        if curr_line >= len(lines):
            logger.error(f"Cartesian data not found in file {postfit_file}")
            return FittedOrbits.empty()

        # The next three lines should be cartesian data, MJD, and MAG
        assert curr_line + 3 < len(lines)
        assert lines[curr_line + 1].startswith(" CAR")
        state_vector = [float(x) for x in lines[curr_line + 1].split()[1:]]
        assert len(state_vector) == 6
        assert lines[curr_line + 2].startswith(" MJD")
        time_line = lines[curr_line + 2].split()
        assert len(time_line) == 3, f"Time line: {time_line}"
        scale = "tt" if time_line[2] == "TDT" else time_line[2].lower()
        time = Timestamp.from_mjd([float(time_line[1])], scale=scale)

        # Collect upper-trianglular COV matrix
        curr_line += 3
        cov_matrix = []
        seen_covariance = False
        while (
            not seen_covariance or lines[curr_line].strip().startswith("COV")
        ) and curr_line < len(lines):
            line = lines[curr_line].strip()
            if line.startswith("COV"):
                cov_matrix.extend([float(x) for x in line.split()[1:]])
                seen_covariance = True
            elif seen_covariance:
                break
            curr_line += 1
        assert len(cov_matrix) == 21, f"Length of cov_matrix {len(cov_matrix)}"
        covariances = _upper_triangular_to_full(np.array(cov_matrix))

        cartesian_coordinates = CartesianCoordinates.from_kwargs(
            x=[state_vector[0]],
            y=[state_vector[1]],
            z=[state_vector[2]],
            vx=[state_vector[3]],
            vy=[state_vector[4]],
            vz=[state_vector[5]],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
            covariance=CoordinateCovariances.from_matrix(
                np.reshape(covariances, (1, 6, 6))
            ),
        )
        orbit = FittedOrbits.from_kwargs(
            orbit_id=[object_id],
            object_id=[object_id],
            coordinates=cartesian_coordinates,
            arc_length=[np.max(selected_times) - np.min(selected_times)],
            num_obs=[len(selected_times)],
            chi2=[0],  # not nullable
            reduced_chi2=[0],  # not nullable
        )
        return orbit

    def _extract_members(
        self, path: Path, clean_id: str, observations: OrbitDeterminationObservations
    ) -> Tuple[FittedOrbitMembers, List[float]]:
        """Extract fitted members and MJD dates for selected members."""
        rwo_file = path / "mpcobs" / f"{clean_id}.rwo"
        if not rwo_file.exists():
            logger.error(f"RWO file {rwo_file} is not found")
            return FittedOrbitMembers.empty(), []
        records = []
        done_header = False
        selected_times = []
        with open(rwo_file, "r") as f:
            for line in f.readlines():
                if line.startswith("! Design"):
                    done_header = True
                    continue
                if done_header:
                    # The date is in "YYYY MM DD.dddddddddd" format
                    # datetime parser doesn't pick the day fraction anyway
                    date, fraction = line[17:38].split(".")
                    parsed = Timestamp.from_astropy(
                        astropy.time.Time(date_parser.parse(date))
                    )
                    parsed = parsed.add_fractional_days(float("0." + fraction))
                    stn = line[180:183]
                    selected = line[194:195] in ["1", "2"]
                    records.append(
                        {
                            "time": parsed.mjd()[0].as_py(),
                            "stn": stn,
                            "selected": selected,
                        }
                    )
                    if selected:
                        selected_times.append(parsed.mjd()[0].as_py())

        solution = []
        second = 1.0 / 86400
        for obs in observations:
            stn = obs.observers.code[0].as_py()
            time = obs.coordinates.time.mjd()[0].as_py()
            matches = [
                rec
                for rec in records
                if rec["stn"] == stn and abs(time - rec["time"]) < second
            ]
            assert (
                len(matches) == 1
            ), f"No unique match for {stn}, {observations.coordinates.time.days[0]}:{observations.coordinates.time.nanos[0]} -> {matches}"
            solution.append(matches[0]["selected"])

        # Obsid doesn't seem to make it to the rwo file, so match by date and STN.
        # Could also check RA and DEC
        od_orbit_members = FittedOrbitMembers.from_kwargs(
            orbit_id=np.full(len(observations), clean_id, dtype="object"),
            obs_id=observations.id,
            # not setting residuals here
            solution=solution,
            outlier=pc.invert(solution),
        )
        return od_orbit_members, selected_times

    def _degrees_to_hms(self, degrees: float) -> str:
        """Print degrees in 'HH MM SS.ddd' format."""
        total_hours = degrees / 15.0
        hours = int(total_hours)
        remaining_minutes = (total_hours - hours) * 60
        minutes = int(remaining_minutes)
        seconds = (remaining_minutes - minutes) * 60
        return f"{hours:02d} {minutes:02d} {seconds:06.3f}"

    def _degrees_to_dms(self, degrees: float) -> str:
        """Print degrees in 'sDD MM SS.dd' format."""
        sign = "+" if degrees > 0 else "-"
        whole_degrees = int(degrees)
        remaining_minutes = (degrees - whole_degrees) * 60
        minutes = int(remaining_minutes)
        seconds = (remaining_minutes - minutes) * 60
        return f"{sign}{whole_degrees:02d} {minutes:02d} {seconds:05.2f}"

    def _mpc1992(
        self, clean_id: str, observations: OrbitDeterminationObservations
    ) -> str:
        """Create string representation of observations in MPC1992 format.

        See https://www.minorplanetcenter.net/iau/info/OpticalObs.html
        """
        lines = []
        for obs in observations:
            # In all example files, it's just 5 spaces
            zero_padded_right_justified_number = "     "
            # Provisional designation, 7-char packed form
            designation = f"{clean_id:<{7}}"[:7]
            # Asterisk is only in the first line for new (or unidentified) objects
            asterisk = " " if len(lines) > 0 else "*"
            # Publishable Notes For Observations Of Minor Planets and Comets
            # https://www.minorplanetcenter.net/iau/info/ObsNote.html
            publishable_note = " "
            # How observation was made. 'C' is CCD
            how_made = "C"
            # Date of observation
            obs_time = obs.coordinates.time.rescale("utc")
            date = obs_time.to_astropy().to_datetime()[0].strftime("%Y %m %d")
            date += f"{obs_time.fractional_days()[0].as_py():.6f}"[
                1:
            ]  # "0.fraction", drop "0"
            # RA, format is "HH MM SS.ddd" with 2 or 3 decimals. Assume J2000
            ra = self._degrees_to_hms(obs.coordinates.lon[0].as_py())
            # Declination is "sDD MM SS.dd" with 1 or 2 decimals. Assume J2000
            decl = self._degrees_to_dms(obs.coordinates.lat[0].as_py())
            # Magnitude
            mag = obs.photometry.mag[0]
            if pc.is_valid(mag):
                mag = f"{mag.as_py():02.2f}{obs.photometry.band[0].as_py()[0]}"
            else:
                mag = "      "
            # Doc says this field "must be blank", examples look like part of obsid
            # Interestingly, putting something here affects selection in Orbfit, but
            # the ids are still black in the .rwo file.
            obsid = "      "
            # Station code
            stn = obs.observers.code[0].as_py()

            line = f"{zero_padded_right_justified_number}{designation}{asterisk}{publishable_note}{how_made}{date}{ra}{decl}         {mag}{obsid}{stn}"
            lines.append(line)
        return "\n".join(lines)
