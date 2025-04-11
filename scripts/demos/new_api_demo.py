# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demo showcasing advanced joint friction and drive property manipulation using IsaacLab's new APIs.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/new_api_demo.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from enum import Enum
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="This script demonstrates usage of friction and drive APIs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_assets.robots.fixedArticulation import ARTICULATION_CFG
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation



class SimulationPhase(Enum):
    """Enumeration of simulation phases"""
    SETUP = 0
    APPLY_FRICTION = 1
    REMOVE_FRICTION = 2
    APPLY_DRIVE_ENVELOPE = 3


def design_scene() -> tuple[Articulation, list[float]]:
    """Designs the scene with a single robot."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    origin = [0.0, 0.0, 0.0]
    prim_utils.create_prim("/World/Origin", "Xform", translation=origin)
    robot = Articulation(ARTICULATION_CFG)
    return robot, origin


def configure_joint_properties(robot: Articulation, phase: SimulationPhase):
    """Configures joint friction and drive properties based on simulation phase.
    
    Args:
        robot: Target articulation to modify
        phase: Current simulation phase from SimulationPhase enum
    """
    dof_axis = 2  # Corresponding to rotZ axis
    
    if phase == SimulationPhase.APPLY_FRICTION:
        # Apply high friction to lock joint
        friction_props = robot.root_physx_view.get_dof_friction_properties()
        friction_props[:, dof_axis, 0] = 1e6  # Static friction
        friction_props[:, dof_axis, 1] = 1e6  # Dynamic friction
        friction_props[:, dof_axis, 2] = 1e3  # Viscous friction
        robot.root_physx_view.set_dof_friction_properties(friction_props, robot._ALL_INDICES.cpu())
        
    elif phase == SimulationPhase.REMOVE_FRICTION:
        # Remove friction to allow free movement
        friction_props = robot.root_physx_view.get_dof_friction_properties()
        friction_props[:, dof_axis, :] = 0.0  # Clear all friction parameters
        robot.root_physx_view.set_dof_friction_properties(friction_props, robot._ALL_INDICES.cpu())
        
    elif phase == SimulationPhase.APPLY_DRIVE_ENVELOPE:
        # Configure drive model parameters
        drive_props = robot.root_physx_view.get_dof_drive_model_properties()
        drive_props[:, dof_axis, 0] = 2.0  # Speed effort gradient
        drive_props[:, dof_axis, 1] = 1.0  # Max actuator velocity
        drive_props[:, dof_axis, 2] = 2.0  # Velocity-dependent resistance
        robot.root_physx_view.set_dof_drive_model_properties(drive_props, robot._ALL_INDICES.cpu())


def run_simulation_loop(sim: sim_utils.SimulationContext, robot: Articulation):
    """Runs the simulation loop."""

    # Simulation parameters
    sim_dt = sim.get_physics_dt()
    phase_steps = {
        SimulationPhase.APPLY_FRICTION: 500,
        SimulationPhase.REMOVE_FRICTION: 1000,
        SimulationPhase.APPLY_DRIVE_ENVELOPE: 1500
    }
    
    root_state = robot.data.default_root_state.clone()
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # Handle phase transitions
            current_phase = next(
                (phase for phase, step in phase_steps.items() if count == step),
                None
            )
            if current_phase:
                configure_joint_properties(robot, current_phase)

        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene with one quadruped
    robot, scene_origin = design_scene()
    scene_origin = torch.tensor(scene_origin, device=sim.device)
    # Play the simulator
    sim.reset()
        
    # Run main simulation loop
    run_simulation_loop(sim, robot)
    simulation_app.close()


if __name__ == "__main__":
    main()
    simulation_app.close()