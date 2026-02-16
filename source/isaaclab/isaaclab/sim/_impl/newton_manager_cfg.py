# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .solvers_cfg import MJWarpSolverCfg, NewtonSolverCfg


@configclass
class SDFCfg:
    """Configuration for SDF-based collision detection and hydroelastic contacts.

    SDF (Signed Distance Field) enables accurate mesh-mesh collision by representing
    geometry as a volumetric distance field. Hydroelastic contacts use SDF to compute
    distributed contact forces over the contact surface.

    .. note::
        SDF requires GPU (CUDA) as it uses Warp's Volume API.
    """

    max_resolution: int = 64
    """Maximum dimension for the sparse SDF grid (must be divisible by 8).

    Common values: 32 (fast), 64 (balanced), 128 (accurate), 256 (very accurate).
    """

    narrow_band_range: tuple[float, float] = (-0.01, 0.01)
    """Narrow band distance range (inner, outer) in meters.

    Defines the region around the mesh surface where SDF values are computed.
    Smaller ranges are faster but may miss contacts for thin objects.
    """

    contact_margin: float = 0.01
    """Contact detection margin in meters.

    Distance at which contacts are generated. Larger margins help prevent tunneling.
    """

    is_hydroelastic: bool = False
    """Enable hydroelastic contact model.

    When True, shapes use distributed contact forces over the contact surface.
    Both colliding shapes must have is_hydroelastic=True for hydroelastic contacts.
    """

    k_hydro: float = 1.0e10
    """Hydroelastic contact stiffness coefficient.

    Controls the force-to-penetration ratio for hydroelastic contacts.
    Only used when is_hydroelastic is True.
    """


@configclass
class NewtonCfg:
    """Configuration for Newton-related parameters.

    These parameters are used to configure the Newton physics simulation.
    """

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    If set to False, the simulation performance will be severely degraded.
    """

    solver_cfg: NewtonSolverCfg = MJWarpSolverCfg()

    sdf_cfg: SDFCfg | None = None
    """SDF and hydroelastic contact configuration. If None (default), SDF is disabled."""

    sdf_overrides: dict[str, SDFCfg] | None = None
    """Per-shape SDF overrides keyed by shape key substring.

    After the default ``sdf_cfg`` is applied to all shapes, these overrides are applied
    to shapes whose key contains the given substring. Later entries override earlier ones.

    Example::

        NewtonCfg(
            sdf_cfg=SDFCfg(max_resolution=64),
            sdf_overrides={
                "finger": SDFCfg(max_resolution=256, is_hydroelastic=True),
                "hand": SDFCfg(max_resolution=128),
            },
        )
    """
