import matplotlib.pyplot as plt
import numpy as np
import importlib
from typing import List, Tuple
import time

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve
)
from pydrake.multibody.all import JacobianWrtVariable
import pydrake.math
from pydrake.math import RigidTransform
from pydrake.autodiffutils import AutoDiffXd

from fsm_utils import LEFT_STANCE, RIGHT_STANCE, SPACE_STANCE, DOUBLE_SUPPORT, PointOnFrame

def read_csv(csv_path, sample_num):
    tmp = np.genfromtxt(csv_path, delimiter=',')
    # map = np.random.choice(np.arange(tmp.shape[0]), size = sample_num, replace = False)
    map = np.arange(0, tmp.shape[0], (tmp.shape[0]-0)//sample_num)
    return tmp[map][:sample_num]

class TrajectoryOptimizationSolution:
  def __init__(self):   

    self.builder = DiagramBuilder()
    self.plant = self.builder.AddSystem(MultibodyPlant(0.0))
    file_name = "planar_walker.urdf"
    Parser(plant=self.plant).AddModels(file_name)

    self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName("base").body_frame(),
            RigidTransform.Identity()
        )

    self.plant.Finalize()
    self.planar_arm = self.plant.ToAutoDiffXd()

    self.plant_context = self.plant.CreateDefaultContext()
    self.context = self.planar_arm.CreateDefaultContext()

    ''' Assign contact frames '''
    self.contact_points = {
        LEFT_STANCE: PointOnFrame(
            self.planar_arm.GetBodyByName("left_lower_leg").body_frame(),
            #np.array([AutoDiffXd(0), AutoDiffXd(0), AutoDiffXd(-0.5)])
            np.array([0, 0, -0.5])
        ),
        RIGHT_STANCE: PointOnFrame(
            self.planar_arm.GetBodyByName("right_lower_leg").body_frame(),
            #np.array([AutoDiffXd(0), AutoDiffXd(0), AutoDiffXd(-0.5)])
            np.array([0, 0, -0.5])
        ),
      }

    self.contact_points_nodiff = {
        LEFT_STANCE: PointOnFrame(
            self.plant.GetBodyByName("left_lower_leg").body_frame(),
            #np.array([AutoDiffXd(0), AutoDiffXd(0), AutoDiffXd(-0.5)])
            np.array([0, 0, -0.5])
        ),
        RIGHT_STANCE: PointOnFrame(
            self.plant.GetBodyByName("right_lower_leg").body_frame(),
            #np.array([AutoDiffXd(0), AutoDiffXd(0), AutoDiffXd(-0.5)])
            np.array([0, 0, -0.5])
        ),
    }

    self.swing_foot_points = {
        LEFT_STANCE: self.contact_points[RIGHT_STANCE],
        RIGHT_STANCE: self.contact_points[LEFT_STANCE]
    }

    # Dimensions specific to the planar_arm
    self.n_q = self.planar_arm.num_positions()
    self.n_v = self.planar_arm.num_velocities()
    self.n_x = self.n_q + self.n_v
    self.n_u = self.planar_arm.num_actuators()


    # Store the actuator limits here
    self.effort_limits = np.zeros(self.n_u)
    for act_idx in range(self.n_u):
      self.effort_limits[act_idx] = \
        self.planar_arm.get_joint_actuator(JointActuatorIndex(act_idx)).effort_limit()
    self.joint_limits = np.pi * np.ones(self.n_q)
    self.vel_limits = 15 * np.ones(self.n_v)

  def _CalFootPoints(self, x, pt_to_track:PointOnFrame) -> np.ndarray:
    self.planar_arm.SetPositionsAndVelocities(self.context, x.reshape(-1,1))
    return self.planar_arm.CalcPointsPositions(self.context, 
                                    pt_to_track.frame,
                                    pt_to_track.pt, 
                                    self.planar_arm.world_frame()).ravel()
  
  def _CalBothFootHeight(self, x) -> np.ndarray:
    return self._CalFootPoints(x, self.contact_points[LEFT_STANCE])[2],\
        self._CalFootPoints(x, self.contact_points[RIGHT_STANCE])[2] 

  def AddSwingFootHeightConstraint(self, prog, x, fsm, lb, ub) -> np.ndarray:
    def footHeightHelper(vars):
        return self._CalFootPoints(vars, self.swing_foot_points[fsm])[2:]
    
    vars = x
    prog.AddConstraint(footHeightHelper, 
                          np.array(lb).reshape(-1, 1), 
                          np.array(ub).reshape(-1, 1), 
                          vars)

  
  def AddBothFootHeightConstraint(self, prog, x, lb, ub) -> np.ndarray:
    def bothFootHeightHelper(vars):
        out = np.concatenate(
            [
            np.array([self._CalFootPoints(vars, self.contact_points[LEFT_STANCE])[2],
            self._CalFootPoints(vars, self.contact_points[RIGHT_STANCE])[2]])
            ]).reshape(-1, 1)
    
    vars = x
    prog.AddConstraint(bothFootHeightHelper, 
                            np.array(lb).reshape(-1, 1), 
                            np.array(ub).reshape(-1, 1),  
                          vars)

  def CalculateContactJacobian(self, x, fsm: int, autodiff=True) -> Tuple[np.ndarray, np.ndarray]:
    """
        For a given finite state, LEFT_STANCE or RIGHT_STANCE, calculate the
        Jacobian terms for the contact constraint, J and Jdot * v.

        As an example, see CalcJ and CalcJdotV in PointPositionTrackingObjective

        use contact_points to get the PointOnFrame for the current stance foot
    """    
    plant = self.planar_arm if autodiff else self.plant
    context = self.context if autodiff else self.plant_context
    contact_points = self.contact_points if autodiff else self.contact_points_nodiff

    plant.SetPositionsAndVelocities(context, x)

    J = np.zeros((3, self.n_v))
    JdotV = np.zeros((3,))

    if fsm  == SPACE_STANCE:
      J = np.zeros((3, self.n_v))
      JdotV = np.zeros((3,))
    if fsm in [LEFT_STANCE, RIGHT_STANCE]:
      J = plant.CalcJacobianTranslationalVelocity(
          context, 
          JacobianWrtVariable.kV, 
          contact_points[fsm].frame, 
          contact_points[fsm].pt, 
          plant.world_frame(), 
          plant.world_frame()
      )
      JdotV = plant.CalcBiasTranslationalAcceleration(
          context, 
          JacobianWrtVariable.kV, 
          contact_points[fsm].frame, 
          contact_points[fsm].pt, 
          plant.world_frame(), 
          plant.world_frame()
      ).ravel()
    if fsm == DOUBLE_SUPPORT:
      raise NotImplementedError("Double support not implemented")
    return J, JdotV

  def AddImpulseConstraint(self, prog, xminus, xplus, lambda_c, fsm2):
    # TODO: Do Not Support Double Support Case Now
    ### q should be the same 
    diffq = xminus[:self.n_q] - xplus[:self.n_q]
    prog.AddLinearEqualityConstraint(diffq, np.zeros((self.n_q,)))

    ### vp = vm + M^{-1}*J^T*Lambda
    def impulseHelper(vars):
        xplus = vars[:self.n_x]
        xminus = vars[self.n_x: 2 * self.n_x]
        lambda_c = vars[2 * self.n_x: ]
        self.planar_arm.SetPositionsAndVelocities(self.context, xminus.reshape(-1, 1))
        M = self.planar_arm.CalcMassMatrixViaInverseDynamics(self.context)
        J, _ = self.CalculateContactJacobian(xplus, fsm2)
        
        diffv = M @ (xplus[self.n_q:] - xminus[self.n_q:]) - J.T @ lambda_c
        return diffv
    
    vars = np.hstack((xplus,xminus, lambda_c))
    prog.AddConstraint(impulseHelper, 
                          np.zeros((self.n_v,)), 
                          np.zeros((self.n_v,)), 
                          vars)

  def EvaluateDynamics(self, x, u, J_c, lambda_c, autodiff=True):
    # Computes the dynamics xdot = f(x,u)
    # M @ vdot + Cv + G - B @ u - J_c.T @ lambda_c = 0
    # vdot = M_inv @ (B @ u + J_c.T @ lambda_c - Cv - G)

    plant = self.planar_arm if autodiff else self.plant
    context = self.context if autodiff else self.plant_context
    
    plant.SetPositionsAndVelocities(context, x.reshape(-1, 1))

    M = plant.CalcMassMatrixViaInverseDynamics(context)
    B = plant.MakeActuationMatrix()
    G = -plant.CalcGravityGeneralizedForces(context)
    Cv = plant.CalcBiasTerm(context)

    M_inv = np.zeros((self.n_v, self.n_v)) 
    if(M.dtype == AutoDiffXd):
      M_inv = pydrake.math.inv(M)
    else:
      M_inv = np.linalg.pinv(M)
    v_dot = M_inv @ (B @ u + J_c.T @ lambda_c - Cv - G)
    return np.hstack((x[-self.n_v:], v_dot))

  def CollocationConstraintEvaluator(self, dt, 
                                     x_i, u_i, x_ip1, u_ip1, 
                                     lambda_c_i, lambda_c_ip1, 
                                     bar_lambda, gamma_i,
                                     fsm):
    h_i = np.zeros(self.n_x,)
    # Add a dynamics constraint using x_i, u_i, x_ip1, u_ip1, dt
    # You should make use of the EvaluateDynamics() function to compute f(x,u)
    J_i, _ = self.CalculateContactJacobian(x_i, fsm)
    J_ip1, _ = self.CalculateContactJacobian(x_ip1, fsm)

    f_i = self.EvaluateDynamics(x_i, u_i, J_i, lambda_c_i)
    f_ip1 = self.EvaluateDynamics(x_ip1, u_ip1, J_ip1, lambda_c_ip1)

    s_dot_i = 1.5 / dt * (x_ip1 - x_i) - 0.25 * (f_i + f_ip1)
    s_i = 0.5 * (x_ip1 + x_i) - 0.125 * dt * (f_ip1 - f_i)

    J_col, _ = self.CalculateContactJacobian(s_i, fsm)
    dynam = self.EvaluateDynamics(s_i, 0.5 * (u_i + u_ip1), J_col, bar_lambda)

    # J(q_c) 
    J, _ = self.CalculateContactJacobian(s_i, fsm)

    dynam[:self.n_v] += J.T @ gamma_i

    h_i = dynam - s_dot_i
    return h_i

  def AddCollocationConstraints(self, prog, N, repeat, x, u, lambda_c, gamma, bar_lambda, timesteps, fsm):
    for k in range(repeat):
      for i in range(N - 1):
        def CollocationConstraintHelper(vars):
          x_i = vars[:self.n_x]
          u_i = vars[self.n_x:self.n_x + self.n_u]
          x_ip1 = vars[self.n_x + self.n_u: 2*self.n_x + self.n_u]
          u_ip1 = vars[2*self.n_x + self.n_u: 2*self.n_x + 2*self.n_u]
          lambda_c_i = vars[2*self.n_x + 2*self.n_u: 2*self.n_x + 2*self.n_u + 3]
          lambda_c_ip1 = vars[2*self.n_x + 2*self.n_u + 3: 2*self.n_x + 2*self.n_u + 6]
          gamma_i = vars[2*self.n_x + 2*self.n_u + 6: 2*self.n_x + 2*self.n_u + 6 + 3]
          bar_lambda = vars[2*self.n_x + 2*self.n_u + 6 + 3: 2*self.n_x + 2*self.n_u + 6 + 3 + 3]
          return self.CollocationConstraintEvaluator(timesteps[i+1] - timesteps[i], 
                                                    x_i, u_i, x_ip1, u_ip1, 
                                                    lambda_c_i, lambda_c_ip1, 
                                                    bar_lambda, gamma_i,
                                                    fsm)
        
      # Within this loop add the dynamics constraints for segment i (aka collocation constraints)
      #       to prog
      # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
      # where vars = hstack(x[i], u[i], ...)
        vars = np.hstack((x[i], u[i], x[i+1], u[i+1], 
                          lambda_c[i], lambda_c[i+1], 
                          gamma[i], bar_lambda[i]))
        prog.AddConstraint(CollocationConstraintHelper, 
                          np.zeros(self.n_x,), 
                          np.zeros(self.n_x,), 
                          vars)

  # Contact Contraints, accelaration = 0
  def ContactConstraintEvaluator(self, x_i, mode):
    h_i = np.zeros(3,)
    J_c, J_c_dot_v = self.CalculateContactJacobian(x_i, mode)

    vdot = x_i[self.n_q:]
    h_i = J_c @ vdot + J_c_dot_v
    return h_i

  def AddContactConstraints(self, prog, x, repeat, N, mode):
    for k in range(repeat):
        for i in range(N):
            def AddContactConstraintsHelper(vars):
                x_i = vars[:self.n_x]
                return self.ContactConstraintEvaluator(x_i, mode)
            
            vars = np.hstack(x[:, k*N + i])
            prog.AddConstraint(AddContactConstraintsHelper, 
                        np.zeros(3,), 
                        np.zeros(3,), 
                        vars)


  def solve(self, N, seqs, repeat, initial_states, tf):
    '''
    Parameters:
      N - number of knot points
      seqs - minimal sequence of modes which produce a gait, e.g. [LEFT_FOOT, RIGHT_FOOT]
      repeat - number of times to repeat the sequence
      initial_states - starting configurations
      distance - target distance to throw the ball

    '''
    initial_state = initial_states[0].reshape(-1,1)
    n_mode = len(seqs)

    # Create the mathematical program
    prog = MathematicalProgram()
    x = np.zeros((n_mode, N * repeat, self.n_x), dtype="object")
    u = np.zeros((n_mode, N * repeat, self.n_u), dtype="object")
    lambda_c  = np.zeros((n_mode, N  * repeat, 3), dtype="object")

    ## Slack var
    gamma = np.zeros((n_mode, (N - 1) * repeat, 3), dtype="object")
    bar_lambda_c = np.zeros((n_mode, (N - 1) * repeat, 3), dtype="object")

    for m, mode in enumerate(seqs):
      for i in range(N * repeat):
        x[m, i] = prog.NewContinuousVariables(self.n_x, "x_{}_{}".format(m, i) )
        u[m, i] = prog.NewContinuousVariables(self.n_u, "u_{}_{}".format(m, i) )
        if mode != SPACE_STANCE: 
          lambda_c[m, i] = prog.NewContinuousVariables(3, "lambdac_{}_{}".format(m, i) )
        
    for m, mode in enumerate(seqs):
      for i in range((N-1) * repeat):
          gamma[m, i] = prog.NewContinuousVariables(3, "gamma_{}_{}".format(m, i) )
          if mode != SPACE_STANCE: 
            bar_lambda_c[m, i] = prog.NewContinuousVariables(3, "bar_lambdac_{}_{}".format(m, i) )

    print("Whole x shape: ", x.shape)

    t0 = 0.0
    timesteps = np.linspace(t0, tf, N * repeat * n_mode)


    # 1.Add the kinematic constraints (initial state, final state)?
    ## Torso angle almost upright, within -45 to 45 degrees
    joint_pos_idx = self.plant.GetJointByName("planar_roty").position_start()
    angle = np.pi / 4
    lb = -angle * np.ones((n_mode * N * repeat, 1))
    ub = angle * np.ones((n_mode * N * repeat, 1))
    prog.AddBoundingBoxConstraint(lb.flatten(), ub.flatten(), x[:, :, joint_pos_idx].reshape(-1, 1))

    ## Add constraints on the initial state
    x0 = x[0,0]
    prog.AddLinearEqualityConstraint(x0.reshape(-1, 1), initial_state)


    # 2. Add collocation dynamics constraints
    for m, fsm in enumerate(seqs):
    # Add the collocation aka dynamics constraints
      self.AddCollocationConstraints(prog, N, repeat, 
                                     x[m], u[m], 
                                     lambda_c[m], gamma[m], bar_lambda_c[m],
                                     timesteps, fsm)

    # 3. Add contact point constraints, velocity and acceleration be 0
    for m in seqs:
      if m == SPACE_STANCE:
        continue
      elif m in [LEFT_STANCE, RIGHT_STANCE]:
        self.AddContactConstraints(prog, x, repeat, N, m)
      elif m == DOUBLE_SUPPORT:
        raise NotImplementedError("Double support not implemented")

    # 4. Reset map and guard function
    for m, mode in enumerate(seqs):
      for k in range(repeat):
        for i in range(N):
          if mode in [LEFT_STANCE, RIGHT_STANCE]:
            #TODO: Wrapper function
            if i == N - 1:
              ## Guard Function
              self.AddSwingFootHeightConstraint(prog, x=x[m, k*N + i], fsm=mode, lb=[0], ub=[0])
              
              ## Reset Constraint
              if not(k == repeat - 1 and m == n_mode -1):
                xminus, xplus = x[m, k*N + i], x[seqs[(m+1) % n_mode], k*N + i]
                self.AddImpulseConstraint(prog, xminus, xplus, lambda_c[m, k*N +i], fsm2=seqs[(m+1) % n_mode])
            else:
              self.AddSwingFootHeightConstraint(prog, x=x[m, k*N + i], fsm=mode, lb=[0], ub=[np.inf])
          # For Double Support and Space Stance
          else:
            if i == N - 1:
              self.AddSwingFootHeightConstraint(prog, x=x[m, k*N + i], fsm=mode, lb=[0], ub=[0])
              
              ## Reset Constraint
              if not(k == repeat - 1 and m == n_mode -1 ):
                xminus, xplus = x[m, k*N + i], x[seqs[(m+1) % n_mode], k*N + i]
                self.AddImpulseConstraint(prog, xminus, xplus, lambda_c[m, k*N +i], fsm2=seqs[(m+1) % n_mode])
            else:
              self.AddBothFootHeightConstraint(prog, x=x[m, k*N + i], lb=np.zeros(2), ub=np.ones(2)*np.inf)



    # 5. Add the cost function here
    cost = 0
    u_flat = u.reshape(-1, 4)

    for i in range(u_flat.shape[0] - 1):
      delta_t = timesteps[i+1] - timesteps[i]
      cost += delta_t * 0.5 * (u_flat[i].T @ u_flat[i] + u_flat[i+1].T @ u_flat[i+1])
    prog.AddQuadraticCost(cost)

    # 6. Add bounding box constraints on the inputs and qdot 
    lb = np.ones((n_mode * N * repeat, self.n_q + self.n_v + self.n_u))
    ub = np.ones((n_mode * N * repeat, self.n_q + self.n_v + self.n_u))
    lb[:, :self.n_q] = -self.joint_limits
    ub[:, :self.n_q] = self.joint_limits
    lb[:, self.n_q:self.n_q+self.n_v] = -self.vel_limits
    ub[:, self.n_q:self.n_q+self.n_v] = self.vel_limits
    lb[:, self.n_x:] = -self.effort_limits
    ub[:, self.n_x:] = self.effort_limits

    #print('lb,ub shape:', lb.shape, ub.shape)
    #print("x.shape", np.concatenate([x, u], -1).reshape(-1, self.n_x + self.n_u).shape)
    prog.AddBoundingBoxConstraint(  lb.reshape(-1, 1),
                                    ub.reshape(-1, 1), 
                                    np.concatenate([x, u], -1).reshape(-1, 1)
                                    #np.hstack((x, u))
                                    )

    # Friction Constraints for 2D
    zero_lm = np.zeros((n_mode * N * repeat, 1))
    zero_bar = np.zeros((n_mode * (N-1) * repeat, 1))

    # Add constraint that out of plane contact force is zero
    prog.AddLinearEqualityConstraint(lambda_c[..., 1].reshape(-1, 1), zero_lm)
    prog.AddLinearEqualityConstraint(bar_lambda_c[..., 1].reshape(-1, 1), zero_bar)

    # Add Friction Cone Constraint assuming mu = 1
    mu = 1
    
    lambda_c = lambda_c.reshape(-1, 3)
    bar_lambda_c = bar_lambda_c.reshape(-1, 3)
    for i in range(len(lambda_c)):
        prog.AddLinearConstraint(lambda_c[i, 0] - mu * lambda_c[i, 2] <= 0)
        prog.AddLinearConstraint(-lambda_c[i, 0] - mu * lambda_c[i, 2] <= 0)
    for i in range(len(bar_lambda_c)):
        prog.AddLinearConstraint(bar_lambda_c[i, 0] - mu * bar_lambda_c[i, 2] <= 0)
        prog.AddLinearConstraint(-bar_lambda_c[i, 0] - mu * bar_lambda_c[i, 2] <= 0)
    
    #neginf_lm = -np.inf * np.ones((n_mode * N * repeat, ))
    #neginf_bar = -np.inf * np.ones((n_mode * (N-1) * repeat, ))
    #prog.AddBoundingBoxConstraint(neginf_lm, zero_lm, (lambda_c[..., 0] - mu * lambda_c[..., 2]).flatten())
    #prog.AddBoundingBoxConstraint(neginf_lm, zero_lm,  (-lambda_c[..., 0] - mu * lambda_c[..., 2].flatten()))
    #prog.AddBoundingBoxConstraint(neginf_bar, zero_bar,  (bar_lambda_c[..., 0] - mu * bar_lambda_c[..., 2]).flatten() )
    #prog.AddBoundingBoxConstraint(neginf_bar, zero_bar, (-bar_lambda_c[..., 0] - mu * bar_lambda_c[..., 2]).flatten() )
    

    # TODO: give the solver an initial guess for x and u using prog.SetInitialGuess(var, value)
    
    #initial_traj = np.zeros_like(x)
    #for i in range(n_mode):
    #    for k in range(repeat):
    #        initial_traj[i, ] = initial_states[(i+1) * k * N  : (i+2)*(k+1) * N]

    prog.SetInitialGuess(x.flatten(), initial_states.flatten())
    prog.SetInitialGuess(u.flatten(), np.zeros((n_mode * N * repeat * self.n_u)))


    # Set up solver
    
    s = time.perf_counter()
    result = Solve(prog)
    print(f"Time taken: {time.perf_counter() - s} seconds")
    
    x_sol = result.GetSolution(x.reshape(-1, self.n_x)).reshape(n_mode, -1, self.n_x)
    u_sol = result.GetSolution(u.reshape(-1, self.n_u)).reshape(n_mode, -1, self.n_u)
    lambda_sol = result.GetSolution(lambda_c.reshape(-1, 3)).reshape(n_mode, -1, 3)

    print('optimal cost: ', result.get_optimal_cost())
    print('x_sol: ', x_sol)
    print('u_sol: ', u_sol)

    print(result.get_solution_result())

    #return result, x_sol, u_sol, lambda_sol

    # Reconstruct the trajectory
    xdot_sol = np.zeros(x_sol.shape)
    for m in seqs:
        for k in range(repeat):
            for i in range(N):
                J_i, _ = self.CalculateContactJacobian(x_sol[m, k * N + i], fsm, False)
                xdot_sol[m, k*N + i] = self.EvaluateDynamics(x_sol[m, k * N + i],
                                                     u_sol[m, k * N + i],
                                                     J_i, 
                                                     lambda_sol[m, k * N + i],
                                                     False)
    
    x_sol = x_sol.reshape(-1, self.n_x)
    xdot_sol = xdot_sol.reshape(-1, self.n_x)
    u_sol = u_sol.reshape(-1, self.n_u)
    print(x_sol.shape)
    print(xdot_sol.shape)
    x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
    u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

    return x_traj, u_traj, prog #, prog.GetInitialGuess(x), prog.GetInitialGuess(u)
  
# if __name__ == '__main__':
#   N = 5
#   initial_state = np.zeros(4)
#   final_configuration = np.array([np.pi, 0])
#   tf = 3.0
#   distance = 15.0

#   """
#     q = np.zeros((plant.num_positions(),))
#     q[1] = 0.8
#     theta = -np.arccos(q[1])
#     q[3] = theta
#     q[4] = -2 * theta
#     q[5] = theta
#     q[6] = -2 * theta
#   """

#   x_traj, u_traj, prog, _, _ = solve(N, initial_state, final_configuration, distance, tf)