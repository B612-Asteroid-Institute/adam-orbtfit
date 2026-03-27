# adam_orbfit: ADAM wrapper for Orbfit library
#### A Python package by the Asteroid Institute, a program of the B612 Foundation

A Python wrapper for Orbfit orbit determination software, designed to work seamlessly with adam_core.

The Orbfit software is downloaded and run as a Docker container.
The first time the wrapper is executed, it may take a while to download the container.
Afterwards the cached image will be used.
There are two versions of the container provided by MPC:
`minorplanetcenter/orbfit:latest` and `minorplanetcenter/orbfit:slim`.
The former includes DE441, which by itself is almost 3Gb inside. The slim container
is several times smaller, so it's used by default.
An alternative container can be specified via a constructor parameter.

The executable inside the container is `neocp_prelim.x`.
It takes options via a file. This wrapper will populate this options file.
Default values seem to work fine for our examples, but if desired they can be
overridden via constructor parameters.

Neocp input includes the type of the instrument used to make an observation.
Our input doesn't contain this information, so we default to CCD, represented as 'C'
per [documentation](https://www.minorplanetcenter.net/iau/info/OpticalObs.html).
This default can be overridden via a constructor parameter.

Orbfit occasionally shows strange behavior.
Sometimes for whatever reason it throws `div0` for `[a, c)`, even thought `[a, b)` and `[b, c)`
both complete successfully with similar results. Example is object `2009 NA`.
Further, the runtime seems to grow with the number of observations, but not always in a predictable
way.
While most of our test runs complete within a minute or less, some runs are much, MUCH longer (infinite?)
To address both these issues, the constructor parameters include a `timeout` in seconds (default None)
and a list of `[start fraction, end fraction]` pairs.
This fitter always passes the whole set of input observations to Orbfit first, but if that fails, it
tries to slice the set of observations according to the `fractions` list.
The first option that succeeds is returned as the answer.
The default set of slices includes the first and last 75% and the first, middle, and last 50% of the data.
All observations ignored during this slicing are marked as rejected, in addition to observations
rejected by Orbfit itself.

Orbfit supports a selection of color bands. Unknown to it colors are rejected (just the magnitude part
of the observation). Further, occasionally the `.rwo` file, which provides per-observation information
about which observations were used for the orbit determination, is missing some records.
We allow for up to a third of observations to be missing before rejecting the run.