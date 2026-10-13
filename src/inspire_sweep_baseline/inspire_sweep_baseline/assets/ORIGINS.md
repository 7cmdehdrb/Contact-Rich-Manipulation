# Asset origins

- `data/ur5e_inspire_usd/` is an unchanged copy of the workspace's preconverted
  UR5e/Inspire asset. All four referenced configuration USD layers are included;
  meshes and tactile pad geometry are embedded in those layers. The conversion
  metadata is preserved in `config.yaml`. The asset keeps its twelve movable
  hand joints and seventeen authored pad bodies; this package observes only the
  palm pad body.
- `robot.py` copies the workspace's Axia80/Inspire assembly helpers into this
  package and changes its factory to the original Sweep-Policy UR5e position
  controller gains and fully open hand pose. `mass_cylinder.py` copies the
  procedural bottom-origin, bottom-heavy cylinder from Sweep-Policy. Neither
  implementation imports or inherits from those source packages.
- The original UR5e controller configuration carries the Isaac Lab developers'
  BSD-3-Clause notice. That notice and license text are retained in
  `ISAACLAB_LICENSE.txt`. The repository license is copied as
  `REPOSITORY_LICENSE.txt`. The copied USD directory contains no additional
  asset-specific license file; this note does not assign a new license to its
  underlying robot meshes.
- `OmniPBR.mdl` is a standard Isaac Sim material module supplied by the simulator;
  it is the only nonlocal material dependency in the copied robot USD layers.
