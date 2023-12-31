# How to train your human baby: 2D Biped w/ Constrained Hybrid Collocation
## Modular Gait Optimization: From Unit Moves to Full Trajectory in Bipedal Systems

- Running
Full trajecotry optimization with our Gait Modularization Optimization Technique (GMOT)
![media/run.gif](media/run.gif)
- Jumping
Full trajecotry optimization with GMOT
![media/full_traj_jump.gif](media/full_traj_jump.gif)
Gait unit trajectory
![media/unit_traj_jump.gif](media/unit_traj_jump.gif)
- Walking
Full trajecotry optimization with GMOT
![media/full_traj_walk.gif](media/full_traj_walk.gif)
Gait unit trajectory
![media/unit_traj_walk.gif](media/unit_traj_walk.gif)

## Overview
This project addresses the computational challenges in hybrid direct collocation (HDC) for multi-step bipedal systems trajectory optimization. We introduce the Gait Modularization and Optimization Technique (GMOT), which utilizes unit gait trajectories as initialization for multi-step trajectory optimization, leading to significant improvements in efficacy and efficiency over naive HDC methods. We evaluated our results against three gaits: bunny hopping, walking, and running. 

## Running Environments
If you are in Apple Sillicon, `venv` is recommended for PyDrake environment. 
```bash
python -m venv drake_env
source drake_env/bin/activate
pip install drake
```

or you could refer to [docker image](https://drake.mit.edu/docker.html) of PyDrake.

## Math Details
### Cost
The cost function aims to minimize the sum of the squared control inputs (u) and the squared difference between the current state (x) and the desired state (p).

Cost Function:
$$
\min_{x, u} \sum \Vert u \Vert^2 + \Vert x - p \Vert^2
$$

### Dynamics Constraints
The dynamics constraints ensure that the state derivatives (x_s) at each time step satisfy certain conditions. Specifically, the state derivatives at the midpoint of each time step are calculated based on the current state (x_c), control inputs (u_c), and slack impuluses (λ_k).

Dynamics Constraints:
$$
\dot{x_s}(t_k + .5h) = \begin{bmatrix} 
v_c + J(q_c)^T \gamma_k \\
f(x_c, u_c, \bar{\lambda}_k)
\end{bmatrix} 
$$

### Contact Points Constraints
The contact points constraints enforce that certain contact points (phi and psi) on the biped have zero values. This ensures that the biped maintains stable contact with the ground.

Contact Points Constraints:
$$
\phi(q_k) = \psi(q_k) = \alpha = 0
$$

### Guard Function
The guard function defines a condition (g) that must be satisfied at specific time steps. The condition is such that g(x_i) is greater than or equal to zero, and g(x_k) is equal to zero. This condition helps in determining when a transition or event should occur.

Guard Function:
$$
g(x_i) >= 0  \quad g(x_k) = 0
$$

### Hybrid Collocation
The hybrid collocation method is used to update the velocity (v_p) of the biped. It takes into account the current velocity (v_m), the inverse of the mass matrix (M), the transpose of the Jacobian matrix (J), and the Lagrange multipliers (λ).

Hybrid Collocation:
$$
v_p = v_m + M^{-1}J^T\lambda
$$

### Reset Map
The reset map defines the update rule for the state variables (x) at each time step. It ensures that the state variables after a transition (x^-_k) are equal to the state variables before the transition (x^+_k).

Reset Map:
$$
x^-_k  = x^+_k
$$

## Reference
> Posa, Michael, Cecilia Cantu and Russ Tedrake. “A direct method for trajectory optimization of rigid bodies through contact.” The International Journal of Robotics Research 33 (2014): 69 - 81.