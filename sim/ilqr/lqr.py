import os

import numpy as np

from sim.ilqr.lqr_solver import ILQRSolver, ILQRSolverParameters, ILQRWarmStartParameters


def _max_solve_time_from_env():
    raw = os.environ.get("HUGSIM_ILQR_MAX_SOLVE_TIME_S", "0.05").strip().lower()
    if raw in {"", "none", "off", "disabled"}:
        return None
    value = float(raw)
    if value <= 0.0:
        raise ValueError("HUGSIM_ILQR_MAX_SOLVE_TIME_S must be positive or 'none'")
    return value


def _max_iterations_from_env():
    value = int(os.environ.get("HUGSIM_ILQR_MAX_ITERATIONS", "100"))
    if value <= 0:
        raise ValueError("HUGSIM_ILQR_MAX_ITERATIONS must be positive")
    return value

solver_params = ILQRSolverParameters(
    discretization_time=0.5,
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],
    input_cost_diagonal_entries=[1.0, 10.0],
    state_trust_region_entries=[1.0] * 5,
    input_trust_region_entries=[1.0] * 2,
    max_ilqr_iterations=_max_iterations_from_env(),
    convergence_threshold=1e-6,
    max_solve_time=_max_solve_time_from_env(),
    max_acceleration=3.0,
    max_steering_angle=np.pi / 3.0,
    max_steering_angle_rate=0.4,
    min_velocity_linearization=0.01,
    wheelbase=2.7
)

warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=0.5,
    k_steering_angle_error_feedback=0.05,
    lookahead_distance_lateral_error=15.0,
    k_lateral_error=0.1,
    jerk_penalty_warm_start_fit=1e-4,
    curvature_rate_penalty_warm_start_fit=1e-2,
)

lqr = ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params)

def plan2control(plan_traj, init_state, discretization_time=None):
    global lqr, solver_params
    if discretization_time is not None:
        discretization_time = float(discretization_time)
        if discretization_time <= 0.0:
            raise ValueError("discretization_time must be positive")
        if abs(discretization_time - solver_params.discretization_time) > 1e-9:
            solver_params = ILQRSolverParameters(
                discretization_time=discretization_time,
                state_cost_diagonal_entries=solver_params.state_cost_diagonal_entries,
                input_cost_diagonal_entries=solver_params.input_cost_diagonal_entries,
                state_trust_region_entries=solver_params.state_trust_region_entries,
                input_trust_region_entries=solver_params.input_trust_region_entries,
                max_ilqr_iterations=solver_params.max_ilqr_iterations,
                convergence_threshold=solver_params.convergence_threshold,
                max_solve_time=solver_params.max_solve_time,
                max_acceleration=solver_params.max_acceleration,
                max_steering_angle=solver_params.max_steering_angle,
                max_steering_angle_rate=solver_params.max_steering_angle_rate,
                min_velocity_linearization=solver_params.min_velocity_linearization,
                wheelbase=solver_params.wheelbase,
            )
            lqr = ILQRSolver(solver_params=solver_params,
                             warm_start_params=warm_start_params)
    current_state = init_state
    solutions = lqr.solve(current_state, plan_traj)
    optimal_inputs = solutions[-1].input_trajectory
    accel_cmd = optimal_inputs[0, 0]
    steering_rate_cmd = optimal_inputs[0, 1]
    return accel_cmd, steering_rate_cmd

if __name__ == '__main__':
    # plan_traj = np.zeros((6,5))
    # plan_traj[:, 0] = 1
    # plan_traj[:, 1] = np.ones(6)
    # plan_traj = np.cumsum(plan_traj, axis=0)
    # print(plan_traj)
    plan_traj = np.array([[-0.18724936,  2.29100776,  0.,          0.,          0.,        ],
                        [-0.29260731,  2.2971828 ,  0.,          0.,          0.        ],
                        [-0.46831554,  2.55596018,  0.,          0.,          0.        ],
                        [-0.5859955 ,  2.73183298,  0.,          0.,          0.        ],
                        [-0.62684   ,  2.84659386,  0.,          0.,          0.        ],
                        [-0.67761713,  2.80647802,  0.,          0.,          0.        ]])
    plan_traj = plan_traj[:, [1,0,2,3,4]]
    init_state = np.array([0.00000000e+00, 3.46944695e-17, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    print(plan_traj.shape, init_state.shape)
    acc, steer = plan2control(plan_traj, init_state)
    print(acc, steer)
