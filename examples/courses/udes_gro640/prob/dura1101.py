#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
------------------------------------


Fichier d'amorce pour les livrables de la problématique GRO640'


"""

import numpy as np

from pyro.control import robotcontrollers
from pyro.control.robotcontrollers import EndEffectorPD
from pyro.control.robotcontrollers import EndEffectorKinematicController


###################
# Part 1
###################

def dh2T(r, d, theta, alpha):
    """

    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1
    
    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """
    ###################
    # Votre code ici
    ###################

    T = np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
                  [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
                  [0, np.sin(alpha), np.cos(alpha), d],
                  [0, 0, 0, 1]])
    return T


def dhs2T(r, d, theta, alpha):
    """

    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1
    
    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil

    """
    
    WTT = np.zeros((4,4))
    
    ###################
    # Votre code ici
    n = r.size
    for i in range(n):
        T = dh2T(r[i], d[i], theta[i], alpha[i])
        if i == 0:
            WTT = T
        else:
            WTT = WTT @ T

    ###################
    
    return WTT


def f(q):
    """
    

    Parameters
    ----------
    q : float 6x1
        Joint space coordinates

    Returns
    -------
    r : float 3x1 
        Effector (x,y,z) position

    """
    r = np.zeros((3,1))
    
    ###################
    # Votre code ici
    ###################
    l = 0

    r = np.array([0.039, 0.155, 0.135, 0, 0, -0.006])
    d = np.array([-0.147, 0, 0, 0.081, 0.137, 0.0095 + q[5]])
    theta = np.array([q[0] - np.pi / 2, q[1] + np.pi / 2, q[2], q[3] + np.pi / 2, q[4] + np.pi / 2, 0])
    alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
    WTT = dhs2T(r, d, theta, alpha)
    r = np.array([[WTT[0, 3]], [WTT[1, 3]], [WTT[2, 3]]])
    return r


###################
# Part 2
###################
    
class CustomPositionController( EndEffectorKinematicController ) :
    
    ############################
    def __init__(self, manipulator ):
        """ """
        
        EndEffectorKinematicController.__init__( self, manipulator, 1)
        
        ###################################################
        # Vos paramètres de loi de commande ici !!
        ###################################################


    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback law: u = c(y,r,t)
        
        INPUTS
        y = q   : sensor signal vector  = joint angular positions      dof x 1
        r = r_d : reference signal vector  = desired effector position   e x 1
        t       : time                                                   1 x 1
        
        OUPUTS
        u = dq  : control inputs vector =  joint velocities             dof x 1
        
        """
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e = r_desired - r_actual
        
        ################
        dq = np.zeros( self.m )  # place-holder de bonne dimension
        
        ##################################
        # Votre loi de commande ici !!!
        ##################################
        gain = 0.5
        dq = np.linalg.inv((J.T @ J + gain**2*np.identity(self.dof))) @ J.T @ e
        
        return dq
    
    
###################
# Part 3
###################
        

        
class CustomDrillingController( robotcontrollers.RobotController ) :
    """ 

    """
    
    ############################
    def __init__(self, robot_model ):
        """ """
        
        super().__init__( dof = 3 )
        
        self.robot_model = robot_model
        
        # Label
        self.name = 'Custom Drilling Controller'
        
        
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        # Ref

        #f_e = r
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Robot model
        r = self.robot_model.forward_kinematic_effector( q ) # End-effector actual position
        J = self.robot_model.J( q )      # Jacobian matrix
        g = self.robot_model.g( q )      # Gravity vector
        H = self.robot_model.H( q )      # Inertia matrix
        C = self.robot_model.C( q , dq ) # Coriolis matrix
        ##################################
        # Votre loi de commande ici !!!
        ##################################

        r_d = np.array([0.25, 0.25, 0.45])
        f_d = np.array([0, 0, -200])
        r_e = r_d - r
        q_d = np.linalg.inv(J) @ r
        J_T = J.T

        Kp = np.diag([20, 20, 20])
        Kd = np.diag([10, 10, 10])

        if np.linalg.norm(r_e) > 0.01:
            u = J.T @ (Kp @ r_e + Kd @ (-J @ dq)) + g
        else:
            u = J_T @ f_d + g

        return u
        
    
###################
# Part 4
###################
        
    
def goal2r(r_0,r_f, t_f):
    """
    
    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time 

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    #################################
    # Votre code ici !!!
    ##################################

    dir_vect = r_f - r_0
    v = dir_vect / t_f
    t_increment = t_f / l
    dir_increment = v * t_increment

    for i in range(l):
        r[:,i] = r_0 + i * dir_increment
        dr[:,i] = v
        ddr[:,i] = np.zeros(3)
    print(r[:,0])
    print(r[:,l-1])
    return r, dr, ddr


def r2q( r, dr, ddr , manipulator ):
    """

    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    # Time discretization
    l = r.shape[1]

    # Number of DoF
    n = manipulator.dof
    # arm length
    l1 = manipulator.l1
    l2 = manipulator.l2
    l3 = manipulator.l3

    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))

    #################################
    # Votre code ici !!!
    ##################################

    for i in range(l):  # Loop over time steps
        # Calculate joint angles using inverse kinematics equations
        q[0, i] = np.arctan2(r[1, i], r[0, i])
        q[1, i] = np.arctan2(r[2, i] - l2, np.sqrt(r[0, i] ** 2 + r[1, i] ** 2))
        q[2, i] = np.arctan2(np.sqrt(r[0, i] ** 2 + r[1, i] ** 2) - l3, r[2, i] - l2)

        # Calculate joint velocities using numerical differentiation
        # if i > 0:
        #     dq[:, i] = (q[:, i] - q[:, i - 1]) / (1.0 / dt)
        #
        # # Calculate joint accelerations using numerical differentiation
        # if i > 1:
        #     ddq[:, i] = (dq[:, i] - dq[:, i - 1]) / (1.0 / dt)

    
    return q, dq, ddq



def q2torque( q, dq, ddq , manipulator ):
    """

    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return tau