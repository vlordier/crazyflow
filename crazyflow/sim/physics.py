"""Physics models for the simulation."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from drone_models.core import load_params
from drone_models.first_principles import dynamics as first_principles_dynamics
from drone_models.so_rpy import dynamics as so_rpy_dynamics
from drone_models.so_rpy_rotor import dynamics as so_rpy_rotor_dynamics
from drone_models.so_rpy_rotor_drag import dynamics as so_rpy_rotor_drag_dynamics
from flax.struct import dataclass
from jax import Array

if TYPE_CHECKING:
    from jax import Device

    from crazyflow.sim.data import SimData


class Physics(str, Enum):
    """Physics mode for the simulation."""

    first_principles = "first_principles"
    so_rpy = "so_rpy"
    so_rpy_rotor = "so_rpy_rotor"
    so_rpy_rotor_drag = "so_rpy_rotor_drag"
    fixed_wing = "fixed_wing"
    default = first_principles


@dataclass
class FirstPrinciplesData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    L: Array  # (N, M, 1)
    """Arm length of the drone."""
    prop_inertia: Array  # (N, M, 1)
    """Inertia of the propeller."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    rpm2thrust: Array  # (N, M, 1)
    """Force constant of the drone."""
    rpm2torque: Array  # (N, M, 1)
    """Torque constant of the drone."""
    mixing_matrix: Array  # (N, M, 3, 4)
    """Mixing matrix of the drone."""
    drag_matrix: Array  # (N, M, 3, 3)
    """Drag matrix of the drone."""
    rotor_dyn_coef: Array  # (N, M, 4)
    """Rotor speed dynamics time constant of the drone."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, drone_model: str, device: Device
    ) -> FirstPrinciplesData:
        """Create a default set of parameters for the simulation."""
        p = load_params("first_principles", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return FirstPrinciplesData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            L=jnp.asarray(p["L"], device=device),
            prop_inertia=jnp.asarray(p["prop_inertia"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            rpm2thrust=jnp.asarray(p["rpm2thrust"], device=device),
            rpm2torque=jnp.asarray(p["rpm2torque"], device=device),
            mixing_matrix=jnp.asarray(p["mixing_matrix"], device=device),
            drag_matrix=jnp.asarray(p["drag_matrix"], device=device),
            rotor_dyn_coef=jnp.asarray(p["rotor_dyn_coef"], device=device),
        )


def first_principles_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the first principle physics model."""
    params: FirstPrinciplesData = data.params
    vel, _, acc, ang_acc, rotor_acc = first_principles_dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.rotor_vel,
        rotor_vel=data.states.rotor_vel,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        mass=params.mass,
        L=params.L,
        prop_inertia=params.prop_inertia,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        rpm2thrust=params.rpm2thrust,
        rpm2torque=params.rpm2torque,
        mixing_matrix=params.mixing_matrix,
        drag_matrix=params.drag_matrix,
        rotor_dyn_coef=params.rotor_dyn_coef,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )
    return data.replace(states_deriv=states_deriv)


@dataclass
class SoRpyData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    acc_coef: Array  # (N, M, 1)
    """Coefficient for the acceleration."""
    cmd_f_coef: Array  # (N, M, 1)
    """Coefficient for the collective thrust."""
    rpy_coef: Array  # (N, M, 1)
    """Coefficient for the roll pitch yaw dynamics."""
    rpy_rates_coef: Array  # (N, M, 1)
    """Coefficient for the roll pitch yaw rates dynamics."""
    cmd_rpy_coef: Array  # (N, M, 1)
    """Coefficient for the roll pitch yaw command dynamics."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, drone_model: str, device: Device) -> SoRpyData:
        """Create a default set of parameters for the simulation."""
        p = load_params("so_rpy", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return SoRpyData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            acc_coef=jnp.asarray(p["acc_coef"], device=device),
            cmd_f_coef=jnp.asarray(p["cmd_f_coef"], device=device),
            rpy_coef=jnp.asarray(p["rpy_coef"], device=device),
            rpy_rates_coef=jnp.asarray(p["rpy_rates_coef"], device=device),
            cmd_rpy_coef=jnp.asarray(p["cmd_rpy_coef"], device=device),
        )


def so_rpy_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the so_rpy physics model."""
    params: SoRpyData = data.params
    vel, _, acc, ang_acc, _ = so_rpy_dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.attitude.cmd,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        mass=params.mass,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        acc_coef=params.acc_coef,
        cmd_f_coef=params.cmd_f_coef,
        rpy_coef=params.rpy_coef,
        rpy_rates_coef=params.rpy_rates_coef,
        cmd_rpy_coef=params.cmd_rpy_coef,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc
    )
    return data.replace(states_deriv=states_deriv)


@dataclass
class SoRpyRotorData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    thrust_time_coef: Array  # (N, M, 1)
    """Rotor coefficient of the drone."""
    acc_coef: Array  # (N, M, 1)
    """Acceleration coefficient of the drone."""
    cmd_f_coef: Array  # (N, M, 1)
    """Collective thrust coefficient of the drone."""
    rpy_coef: Array  # (N, M, 1)
    """Roll pitch yaw coefficient of the drone."""
    rpy_rates_coef: Array  # (N, M, 1)
    """Roll pitch yaw rates coefficient of the drone."""
    cmd_rpy_coef: Array  # (N, M, 1)
    """Roll pitch yaw command coefficient of the drone."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, drone_model: str, device: Device) -> SoRpyRotorData:
        """Create a default set of parameters for the simulation."""
        p = load_params("so_rpy_rotor", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return SoRpyRotorData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            thrust_time_coef=jnp.asarray(p["thrust_time_coef"], device=device),
            acc_coef=jnp.asarray(p["acc_coef"], device=device),
            cmd_f_coef=jnp.asarray(p["cmd_f_coef"], device=device),
            rpy_coef=jnp.asarray(p["rpy_coef"], device=device),
            rpy_rates_coef=jnp.asarray(p["rpy_rates_coef"], device=device),
            cmd_rpy_coef=jnp.asarray(p["cmd_rpy_coef"], device=device),
        )


def so_rpy_rotor_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the so_rpy_rotor physics model."""
    params: SoRpyRotorData = data.params
    vel, _, acc, ang_acc, rotor_acc = so_rpy_rotor_dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        rotor_vel=data.states.rotor_vel,
        cmd=data.controls.attitude.cmd,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        mass=params.mass,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        thrust_time_coef=params.thrust_time_coef,
        acc_coef=params.acc_coef,
        cmd_f_coef=params.cmd_f_coef,
        rpy_coef=params.rpy_coef,
        rpy_rates_coef=params.rpy_rates_coef,
        cmd_rpy_coef=params.cmd_rpy_coef,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )
    return data.replace(states_deriv=states_deriv)


@dataclass
class SoRpyRotorDragData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    thrust_time_coef: Array  # (N, M, 1)
    """Rotor coefficient of the drone."""
    acc_coef: Array  # (N, M, 1)
    """Acceleration coefficient of the drone."""
    cmd_f_coef: Array  # (N, M, 1)
    """Collective thrust coefficient of the drone."""
    rpy_coef: Array  # (N, M, 1)
    """Roll pitch yaw coefficient of the drone."""
    rpy_rates_coef: Array  # (N, M, 1)
    """Roll pitch yaw rates coefficient of the drone."""
    cmd_rpy_coef: Array  # (N, M, 1)
    """Roll pitch yaw command coefficient of the drone."""
    drag_matrix: Array  # (N, M, 3, 3)
    """Linear drag coefficient matrix of the drone."""

    @staticmethod
    def create(
        n_worlds: int, n_drones: int, drone_model: str, device: Device
    ) -> SoRpyRotorDragData:
        """Create a default set of parameters for the simulation."""
        p = load_params("so_rpy_rotor_drag", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return SoRpyRotorDragData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            thrust_time_coef=jnp.asarray(p["thrust_time_coef"], device=device),
            acc_coef=jnp.asarray(p["acc_coef"], device=device),
            cmd_f_coef=jnp.asarray(p["cmd_f_coef"], device=device),
            rpy_coef=jnp.asarray(p["rpy_coef"], device=device),
            rpy_rates_coef=jnp.asarray(p["rpy_rates_coef"], device=device),
            cmd_rpy_coef=jnp.asarray(p["cmd_rpy_coef"], device=device),
            drag_matrix=jnp.asarray(p["drag_matrix"], device=device),
        )


def so_rpy_rotor_drag_physics(data: SimData) -> SimData:
    """Compute the forces and torques from the so_rpy_rotor_drag physics model."""
    params: SoRpyRotorDragData = data.params
    vel, _, acc, ang_acc, rotor_acc = so_rpy_rotor_drag_dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.attitude.cmd,
        rotor_vel=data.states.rotor_vel,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        mass=params.mass,
        gravity_vec=params.gravity_vec,
        J=params.J,
        J_inv=params.J_inv,
        thrust_time_coef=params.thrust_time_coef,
        acc_coef=params.acc_coef,
        cmd_f_coef=params.cmd_f_coef,
        rpy_coef=params.rpy_coef,
        rpy_rates_coef=params.rpy_rates_coef,
        cmd_rpy_coef=params.cmd_rpy_coef,
        drag_matrix=params.drag_matrix,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )
    return data.replace(states_deriv=states_deriv)


@dataclass
class FixedWingData:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    # Geometry
    S: Array  # (N, M, 1)
    """Wing reference area [m^2]."""
    b: Array  # (N, M, 1)
    """Wingspan [m]."""
    c: Array  # (N, M, 1)
    """Mean aerodynamic chord [m]."""
    # Longitudinal lift coefficients
    CL0: Array  # (N, M, 1)
    """Zero-alpha lift coefficient."""
    CL_alpha: Array  # (N, M, 1)
    """Lift curve slope [1/rad]."""
    CL_de: Array  # (N, M, 1)
    """Elevon lift effectiveness [1/rad]."""
    # Drag coefficients
    CD0: Array  # (N, M, 1)
    """Zero-lift drag coefficient."""
    k: Array  # (N, M, 1)
    """Induced drag factor."""
    # Lateral coefficients
    CY_beta: Array  # (N, M, 1)
    """Sideforce sideslip derivative [1/rad]."""
    Cl_beta: Array  # (N, M, 1)
    """Dihedral effect [1/rad]."""
    Cl_p: Array  # (N, M, 1)
    """Roll damping coefficient."""
    Cl_r: Array  # (N, M, 1)
    """Roll due to yaw rate."""
    Cl_da: Array  # (N, M, 1)
    """Differential elevon effectiveness [1/rad]."""
    # Pitch moment coefficients
    Cm0: Array  # (N, M, 1)
    """Zero-alpha pitch moment."""
    Cm_alpha: Array  # (N, M, 1)
    """Pitch stability derivative [1/rad]."""
    Cm_q: Array  # (N, M, 1)
    """Pitch damping coefficient."""
    Cm_de: Array  # (N, M, 1)
    """Elevon pitch effectiveness [1/rad]."""
    # Yaw moment coefficients
    Cn_beta: Array  # (N, M, 1)
    """Weathercock stability [1/rad]."""
    Cn_r: Array  # (N, M, 1)
    """Yaw damping coefficient."""
    # Propulsion
    T_max: Array  # (N, M, 1)
    """Maximum thrust [N]."""
    rho: Array  # (N, M, 1)
    """Air density [kg/m^3]."""
    # Stall model
    CL_max: Array  # (N, M, 1)
    """Maximum lift before stall."""
    CD_stall: Array  # (N, M, 1)
    """Extra drag coefficient at stall."""
    stall_k: Array  # (N, M, 1)
    """Stall drag rise sharpness."""
    # Autopilot inner-loop gains (0 = disabled)
    k_pitch_attitude: Array  # (N, M, 1)
    """Pitch attitude feedback gain. 0 = open-loop."""
    k_pitch_rate: Array  # (N, M, 1)
    """Pitch rate damping gain. 0 = open-loop."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, drone_model: str, device: Device) -> FixedWingData:
        """Create default parameters from drone-models data."""
        p = load_params("fixed_wing", drone_model)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return FixedWingData(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            # Geometry
            S=jnp.full((n_worlds, n_drones, 1), p["S"], device=device),
            b=jnp.full((n_worlds, n_drones, 1), p["b"], device=device),
            c=jnp.full((n_worlds, n_drones, 1), p["c"], device=device),
            # Longitudinal lift
            CL0=jnp.full((n_worlds, n_drones, 1), p["CL0"], device=device),
            CL_alpha=jnp.full((n_worlds, n_drones, 1), p["CL_alpha"], device=device),
            CL_de=jnp.full((n_worlds, n_drones, 1), p["CL_de"], device=device),
            # Drag
            CD0=jnp.full((n_worlds, n_drones, 1), p["CD0"], device=device),
            k=jnp.full((n_worlds, n_drones, 1), p["k"], device=device),
            # Lateral
            CY_beta=jnp.full((n_worlds, n_drones, 1), p["CY_beta"], device=device),
            Cl_beta=jnp.full((n_worlds, n_drones, 1), p["Cl_beta"], device=device),
            Cl_p=jnp.full((n_worlds, n_drones, 1), p["Cl_p"], device=device),
            Cl_r=jnp.full((n_worlds, n_drones, 1), p["Cl_r"], device=device),
            Cl_da=jnp.full((n_worlds, n_drones, 1), p["Cl_da"], device=device),
            # Pitch moment
            Cm0=jnp.full((n_worlds, n_drones, 1), p["Cm0"], device=device),
            Cm_alpha=jnp.full((n_worlds, n_drones, 1), p["Cm_alpha"], device=device),
            Cm_q=jnp.full((n_worlds, n_drones, 1), p["Cm_q"], device=device),
            Cm_de=jnp.full((n_worlds, n_drones, 1), p["Cm_de"], device=device),
            # Yaw moment
            Cn_beta=jnp.full((n_worlds, n_drones, 1), p["Cn_beta"], device=device),
            Cn_r=jnp.full((n_worlds, n_drones, 1), p["Cn_r"], device=device),
            # Propulsion
            T_max=jnp.full((n_worlds, n_drones, 1), p["T_max"], device=device),
            rho=jnp.full((n_worlds, n_drones, 1), p["rho"], device=device),
            # Stall
            CL_max=jnp.full((n_worlds, n_drones, 1), p["CL_max"], device=device),
            CD_stall=jnp.full((n_worlds, n_drones, 1), p["CD_stall"], device=device),
            stall_k=jnp.full((n_worlds, n_drones, 1), p["stall_k"], device=device),
            # Autopilot (default 0 = open-loop, no inner-loop stabilization)
            k_pitch_attitude=jnp.full((n_worlds, n_drones, 1), p.get("k_pitch_attitude", 0.0), device=device),
            k_pitch_rate=jnp.full((n_worlds, n_drones, 1), p.get("k_pitch_rate", 0.0), device=device),
        )


def fixed_wing_physics(data: SimData) -> SimData:
    """6-DOF fixed-wing aerodynamics.

    The control pipeline sets:
      - states.force: thrust vector in world frame (from body-x rotation)
      - states.torque: [elevon_roll, elevon_pitch, 0] control moments

    This function computes:
      1. Lift, drag, and sideforce from aerodynamic state
      2. Roll, pitch, yaw moments from state + control surfaces
      3. Gravity in world frame
      4. Total acceleration and angular acceleration
    """
    params: FixedWingData = data.params
    states = data.states

    # Quaternion components [x, y, z, w] — Crazyflow scalar-last convention
    qx, qy, qz, qw = (
        states.quat[..., 0:1],
        states.quat[..., 1:2],
        states.quat[..., 2:3],
        states.quat[..., 3:4],
    )

    # Rotation matrix: world to body frame
    # Standard formula for scalar-last quaternion [x, y, z, w]:
    # R = [[1-2(y²+z²), 2(xy+wz), 2(xz-wy)],
    #      [2(xy-wz), 1-2(x²+z²), 2(yz+wx)],
    #      [2(xz+wy), 2(yz-wx), 1-2(x²+y²)]]
    #
    # This R maps body→world vectors. R^T = R_w2b maps world→body.
    # We construct R^T (world→body) directly:
    R_w2b = (
        jnp.concatenate(
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx),
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2),
            ],
            axis=-1,
        )
        .reshape(*states.vel.shape[:-1], 3, 3)
    )

    # Body-frame velocity: v_body = R_w2b @ v_world
    v_body = jnp.einsum("...ij,...j->...i", R_w2b, states.vel)  # (N, M, 3)
    u, v, w_b = v_body[..., 0:1], v_body[..., 1:2], v_body[..., 2:3]

    # Airspeed, angle of attack, sideslip
    V = jnp.sqrt(jnp.sum(v_body**2, axis=-1, keepdims=True) + 1e-8)
    alpha = jnp.arctan2(w_b, jnp.abs(u) + 1e-8)  # AoA [rad]
    beta = jnp.arcsin(v / (V + 1e-8))  # Sideslip [rad]

    # Dynamic pressure: qbar = 0.5 * rho * V^2
    qbar = 0.5 * params.rho * V**2

    # ── Autopilot inner loop ──
    # Extract pitch angle from quaternion [x, y, z, w]:
    # pitch = arcsin(2 * (qw * qy - qz * qx))
    pitch_angle = jnp.arcsin(jnp.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    pitch_rate = states.ang_vel[..., 1:2]  # body-frame pitch rate
    autopilot_active = (jnp.abs(params.k_pitch_attitude) + jnp.abs(params.k_pitch_rate)) > 1e-8
    target_pitch = jnp.zeros_like(pitch_angle) + 0.126  # 7.2° nose-up for level flight at 50 m/s [rad]
    elevon_pitch_trim = autopilot_active * (
        params.k_pitch_attitude * (pitch_angle - target_pitch)
        + params.k_pitch_rate * pitch_rate
    )

    # ── Longitudinal forces (body frame) ──
    # Control surface deflections from torque command + autopilot trim
    de = states.torque[..., 1:2] / params.T_max + elevon_pitch_trim  # symmetric elevon

    # Lift coefficient
    CL = params.CL0 + params.CL_alpha * alpha + params.CL_de * de

    # Drag coefficient: drag polar + stall model
    CD = params.CD0 + params.k * CL**2
    # Stall: sharp drag rise past CL_max
    stall_excess = jnp.maximum(0.0, jnp.abs(CL) - params.CL_max) * params.stall_k
    CD = CD + params.CD_stall * jnp.tanh(stall_excess)

    # Lift and drag magnitudes
    L_mag = CL * qbar * params.S  # (N, M, 1) lift force
    D_mag = CD * qbar * params.S  # (N, M, 1) drag force

    # Decompose into body-frame force vector.
    # For z-up body frame at angle-of-attack α:
    #   Drag opposes velocity:      [-cos(α),  0, -sin(α)]
    #   Lift perp to velocity, +z at α=0:  [-sin(α),  0,  cos(α)]
    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    F_aero_body = jnp.concatenate(
        [
            -D_mag * cos_alpha - L_mag * sin_alpha,  # body-x: aft + forward lift projection
            jnp.zeros_like(D_mag),  # body-y: zero (sideforce handled below)
            -D_mag * sin_alpha + L_mag * cos_alpha,  # body-z: up (lift) + down (drag projection)
        ],
        axis=-1,
    )

    # Sideforce: CY = CY_beta * beta
    CY = params.CY_beta * beta
    F_side = CY * qbar * params.S  # body-y
    F_aero_body = F_aero_body.at[..., 1:2].set(F_side)

    # ── Moments ──
    p_body = states.ang_vel[..., 0:1]  # roll rate
    q_body = states.ang_vel[..., 1:2]  # pitch rate
    r_body = states.ang_vel[..., 2:3]  # yaw rate

    # Control surface deflections from torque command (normalized)
    da = states.torque[..., 0:1] / params.T_max  # differential elevon (roll)
    dr = states.torque[..., 2:3] / params.T_max  # rudder (yaw), likely unused

    # Roll moment: L = qbar * S * b * Cl
    Cl = (
        params.Cl_beta * beta
        + params.Cl_p * (p_body * params.b / (2 * V))
        + params.Cl_r * (r_body * params.b / (2 * V))
        + params.Cl_da * da
    )
    roll_moment = qbar * params.S * params.b * Cl

    # Pitch moment: M = qbar * S * c * Cm
    Cm = (
        params.Cm0
        + params.Cm_alpha * alpha
        + params.Cm_q * (q_body * params.c / (2 * V))
        + params.Cm_de * de
    )
    pitch_moment = qbar * params.S * params.c * Cm

    # Yaw moment: N = qbar * S * b * Cn
    Cn = params.Cn_beta * beta + params.Cn_r * (r_body * params.b / (2 * V))
    yaw_moment = qbar * params.S * params.b * Cn

    # Total aerodynamic moment in body frame
    M_aero = jnp.concatenate([roll_moment, pitch_moment, yaw_moment], axis=-1)

    # ── Gravity (world frame) ──
    F_gravity_world = params.mass * params.gravity_vec

    # ── Thrust (world frame, set by control function) ──
    # Force from control is unitless throttle [0,1]; scale by T_max to get Newtons
    F_thrust_world = states.force * params.T_max

    # ── Convert aero forces from body to world frame ──
    # R_b2w = R_w2b^T (body→world is transpose of world→body)
    R_b2w = jnp.transpose(R_w2b, axes=(0, 1, 3, 2))
    F_aero_world = jnp.einsum("...ij,...j->...i", R_b2w, F_aero_body)

    # Total force in world frame
    F_total = F_aero_world + F_gravity_world + F_thrust_world

    # Linear acceleration
    acc = F_total / params.mass

    # Angular acceleration: α = J_inv @ M_aero
    # (Aerodynamic moments already include control surface effects via de/da/dr;
    #  no separate control torque addition needed.)
    ang_acc = jnp.einsum("...ij,...j->...i", params.J_inv, M_aero)

    # State derivatives for the integrator.
    # Convention (matching all other physics backends):
    #   vel      = current body angular velocity (used as rotation vector for quaternion update)
    #   ang_vel  = current body angular velocity (drot in integration = states.ang_vel)
    #   acc      = linear acceleration
    #   ang_acc  = angular acceleration
    #   rotor_acc = zeros (non-rotorcraft)
    states_deriv = data.states_deriv.replace(
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        acc=acc,
        ang_acc=ang_acc,
        rotor_acc=jnp.zeros_like(data.states.rotor_vel),
    )
    return data.replace(states_deriv=states_deriv)
