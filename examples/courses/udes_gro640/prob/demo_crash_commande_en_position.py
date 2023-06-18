#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import LaserRobot

from pyro.control.robotcontrollers import EndEffectorKinematicController
from examples.courses.udes_gro640.prob.dura1101 import CustomPositionController

# Model cinématique du robot
sys = LaserRobot()

# Contrôleur en position de l'effecteur standard
#ctl = EndEffectorKinematicController( sys )
ctl = CustomPositionController( sys )

# Cible de position pour l'effecteur
ctl.rbar = np.array([0,-1])

# Dynamique en boucle fermée
clsys = ctl + sys

# Configurations de départs
clsys.x0 =  np.array([0,0.5,0]) # crash
#clsys.x0 =  np.array([0,0.7,0]) # fonctionne

# Simulation
clsys.compute_trajectory( solver = 'odeint'  )
#clsys.compute_trajectory( solver = 'solve_ivt' )

clsys.plot_trajectory('xu')
clsys.animate_simulation()
