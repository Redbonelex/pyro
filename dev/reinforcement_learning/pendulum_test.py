#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.control  import controller
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
from pyro.planning import discretizer

sys  = pendulum.SinglePendulum()

sys.x_ub[0] =  3.0
sys.x_lb[0] = -5.0
sys.x_lb[1] = -5.0
sys.x_ub[1] = 5.0

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [301,301] , [11] , 0.05)

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 300

qcf.R[0,0] = 2.0

qcf.S[0,0] = 10.0
qcf.S[1,1] = 10.0


# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.solve_bellman_equation( tol = 0.5 )

ctl = dp.get_lookup_table_controller()

dp.plot_cost2go()
dp.plot_cost2go_3D()

ctl.plot_control_law( sys = sys , n = 100)


#asign controller
cl_sys = controller.ClosedLoopSystem( sys , ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([0,0.1])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()