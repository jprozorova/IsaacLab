# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test SDF override workflow via NewtonCfg."""

import pytest
import warp as wp
from newton import GeoType

from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg, SDFCfg
from isaaclab.sim.simulation_cfg import SimulationCfg

from isaaclab_assets import FRANKA_PANDA_CFG


@pytest.mark.parametrize("device", ["cuda:0"])
def test_sdf_via_newton_cfg(device):
    """Set sdf_max_resolution in NewtonCfg, verify override is applied and simulation runs."""
    newton_cfg = NewtonCfg(
        sdf_cfg=SDFCfg(max_resolution=64),
    )
    sim_cfg = SimulationCfg(device=device, create_stage_in_memory=False, newton_cfg=newton_cfg)
    with build_simulation_context(auto_add_lighting=True, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None

        robot = Articulation(cfg=FRANKA_PANDA_CFG.replace(prim_path="/World/Robot"))
        sim.reset()

        assert robot.is_initialized

        model = NewtonManager.get_model()
        assert model is not None
        assert model.shape_sdf_data is not None

        # Verify the builder override was applied (sdf_max_resolution set on all shapes)
        builder = NewtonManager._builder
        for i in range(builder.shape_count):
            assert builder.shape_sdf_max_resolution[i] == 64

        for _ in range(5):
            sim.step(render=False)
            robot.update(sim.cfg.dt)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_sdf_overrides_per_shape(device):
    """Use sdf_overrides to set different SDF config on finger shapes."""
    newton_cfg = NewtonCfg(
        sdf_cfg=SDFCfg(max_resolution=32),
        sdf_overrides={
            "finger": SDFCfg(max_resolution=128, is_hydroelastic=True, k_hydro=1e11),
        },
    )
    sim_cfg = SimulationCfg(device=device, create_stage_in_memory=False, newton_cfg=newton_cfg)
    with build_simulation_context(auto_add_lighting=True, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None

        robot = Articulation(cfg=FRANKA_PANDA_CFG.replace(prim_path="/World/Robot"))
        sim.reset()

        assert robot.is_initialized

        builder = NewtonManager._builder
        from newton import ShapeFlags

        finger_count = 0
        non_finger_count = 0
        for i in range(builder.shape_count):
            if "finger" in builder.shape_key[i]:
                assert builder.shape_sdf_max_resolution[i] == 128, (
                    f"Finger shape {builder.shape_key[i]} should have resolution 128"
                )
                # Hydroelastic only applies to supported types (MESH, primitives), not CONVEX_MESH
                is_mesh = builder.shape_type[i] == int(GeoType.MESH)
                if is_mesh:
                    assert builder.shape_flags[i] & int(ShapeFlags.HYDROELASTIC), (
                        f"Finger mesh shape {builder.shape_key[i]} should be hydroelastic"
                    )
                finger_count += 1
            else:
                assert builder.shape_sdf_max_resolution[i] == 32, (
                    f"Non-finger shape {builder.shape_key[i]} should have resolution 32"
                )
                non_finger_count += 1

        assert finger_count > 0, "Should have matched at least one finger shape"
        assert non_finger_count > 0, "Should have non-finger shapes too"

        for _ in range(5):
            sim.step(render=False)
            robot.update(sim.cfg.dt)
