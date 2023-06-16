# Importing Librairies
import pyro
# Generic python tools
import numpy as np


# Defining the robot arm
from pyro.dynamic import manipulator

robot = manipulator.TwoLinkManipulator()
robot.l1 = 0.5  # length of first rigid link
robot.l2 = 0.3  # length of second rigid link

# Showing the robot natural behavior with no controllers
# robot initial states [ joint 1 angle (rad),  joint 2 angle (rad), joint 1 velocity (rad/sec),  joint 1 velocity (rad/sec)]
robot.x0 = np.array([ 0.1, 0.1, 0.0, 0.0])

# run the simulation
robot.compute_trajectory( tf = 6 )

# Animate and display the simulation
robot.animate_simulation( is_3d = True )

# Joint-space impedance control
from pyro.control  import robotcontrollers

# Target
q_desired = np.array([0.5,0.5]) # this set the desired joint angles vector

# Joint PID
joint_pd      = robotcontrollers.JointPD( dof = 2 )
joint_pd.rbar = q_desired
joint_pd.kp   = np.array([5, 5 ]) # This set the diagonal values of the Kp matrix
joint_pd.kd   = np.array([ 0, 0 ]) # This set the diagonal values of the Kd matrix
# This shows the commanded torque as a function of the configuration (at zero velocity)
#joint_pd.plot_control_law( sys = robot , n = 100 )

# Simulating the robot in closed-loop
# Create the closed-loop system
robot_with_joint_pd = joint_pd + robot

# robot initial states [ joint 1 angle (rad),  joint 2 angle (rad), joint 1 velocity (rad/sec),  joint 1 velocity (rad/sec)]
#closed_loop_robot.x0 = np.array([0.1,0.1,0,0])

# Run the simulation
robot_with_joint_pd.compute_trajectory( tf = 5 )
# Animate the simulation
robot_with_joint_pd.animate_simulation()
# Plot systems states
robot_with_joint_pd.plot_trajectory('x')
# Plot control inputs
robot_with_joint_pd.plot_trajectory('u')

# Analyzing the results
# Frequency domain point of view
robot_with_joint_pd.plot_linearized_bode()

# Task-space impedance controller
# "https://www.youtube.com/embed/videoseries?list=PL6adNeJ0A8UtKTsdDwfei6sKDKJt25ZBO"

# End-effector desired position
r_desired = robot.forward_kinematic_effector( q_desired )
print('Target joint angles [q_1,q_2] =', q_desired )
print('Target effector [x,y] =', r_desired )

effector_pd      = robotcontrollers.EndEffectorPD( robot )
effector_pd.rbar = r_desired
effector_pd.kp   = np.array([100, 100 ]) # This set the diagonal values of the Kp matrix
effector_pd.kd   = np.array([  0,   0 ]) # This set the diagonal values of the Kd matrix

# This shows graphically the computed torque for joint 1 as a function of the configuration
effector_pd.plot_control_law( sys = robot , n = 100 )

# Create the closed-loop system
robot_with_effector_pd = effector_pd + robot

# robot initial states [ joint 1 angle (rad),  joint 2 angle (rad), joint 1 velocity (rad/sec),  joint 1 velocity (rad/sec) ]
#closed_loop_robot.x0 = np.array([0.1,0.1,0,0,0,0])

# Run the simulation
robot_with_effector_pd.compute_trajectory( tf = 5 )
# Animate the simulation
robot_with_effector_pd.animate_simulation()
robot_with_effector_pd.plot_trajectory('x')
robot_with_effector_pd.plot_trajectory('u')


# Custom controller
class CustomController(robotcontrollers.RobotController):

    ############################
    def __init__(self, robot_model):
        """ """

        super().__init__(2)

        self.robot_model = robot_model

    ############################
    def c(self, y, r, t=0):
        """
        y = [  q0,  q1, dq0 , dq1 ] : Feedback signal  y = Robot joint angles and joint velocities
        r = [ qd0, qd1]             : Reference signal r = Robot desired joint angles
        """
        q = y[0:2]  # Joint position vector
        dq = y[2:4]  # Join velocity vector
        r_d = r  # Desired effector position vector

        r = self.robot_model.forward_kinematic_effector(q)  # End-effector actual position
        J = self.robot_model.J(q)  # Jacobian
        g = self.robot_model.g(q)  # Gravity vector
        H = self.robot_model.H(q)  # Inertia matrix
        C = self.robot_model.C(q, dq)  # Coriolis matrix

        u = np.array([0.0, 0.0])  # Place-holder to overwrite with your control law

        ##############################
        # YOUR CODE BELLOW !!
        ##############################

        Kp = np.diag([50, 50])
        Kd = np.diag([0, 0])

        # u = Kp @ ( q_d - q ) + Kd @ ( - dq )                 # Joint impedance law
        u = J.T @ (Kp @ (r_d - r) + Kd @ (- J @ dq))  # End-effector impedance law

        return u

custom_controller = CustomController(robot)
custom_controller.rbar = np.array([0.5, 0.5])  # Desired robot position [ x , y ]

# Create the closed-loop system
robot_with_custom_controller = custom_controller + robot

# Run the simulation
traj = robot_with_custom_controller.compute_trajectory()

# Animate the simulation
robot_with_custom_controller.animate_simulation()

robot_with_custom_controller.plot_trajectory('x')
robot_with_custom_controller.plot_trajectory('u')

# Evaluation of the custom controller
ef = traj.x[-1,0:2] - custom_controller.rbar
print('Final error=',ef)

if np.linalg.norm(ef) > 0.2 :
  print('The final error is large, try an approach to reduce the steady state error.')

max_torque_joint_1 = traj.u[:,0].max()
max_torque_joint_2 = traj.u[:,1].max()

print('Maximum torque of joint 1 = ', max_torque_joint_1 )
print('Maximum torque of joint 2 = ', max_torque_joint_2 )

if (max_torque_joint_1 > 9) or (max_torque_joint_2 > 9) :
  print('The required torques are very large, try to reduce the feedback gains.')

