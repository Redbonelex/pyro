#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.control  import controller
import dynamic_programming as dprog
import discretizer
import costfunction

sys  = pendulum.SinglePendulum()

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [501,501] , [21] )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 1000000


# DP algo
#dp = dprog.DynamicProgramming( grid_sys, qcf )
dp = dprog.DynamicProgrammingWithLookUpTable2( grid_sys, qcf)
#dp = dprog.DynamicProgrammingFast2DGrid(grid_sys, qcf)


#dp.interpol_method = 'nearest' #12 sec
#dp.interpol_method = 'linear'  #18 sec
#dp.interpol_method =  'linear' #

#dp.plot_dynamic_cost2go = False
dp.compute_steps(200)
dp.save_latest('test_hidef')


#grid_sys.plot_grid_value( dp.J_next )

ctl = dprog.LookUpTableController( grid_sys , dp.pi )

ctl.plot_control_law( sys = sys , n = 100)


#asign controller
cl_sys = controller.ClosedLoopSystem( sys , ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([0,0])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()