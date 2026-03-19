# adam_orbfit: ADAM wrapper for Orbfit library
#### A Python package by the Asteroid Institute, a program of the B612 Foundation

A Python wrapper for Orbfit orbit determination software, designed to work seamlessly with adam_core.

The Orbfit software is downloaded and run as a Docker container.
The first time the wrapper is executed, it may take a while to download the container.
Afterwards the cached image will be used.

