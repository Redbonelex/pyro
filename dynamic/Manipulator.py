# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:09:20 2015

@author: agirard
"""

import matplotlib.animation as animation
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from AlexRobotics.dynamic import DynamicSystem as RDDS


'''
#################################################################
##################       2DOF Manipulator Class          ########
#################################################################
'''



class TwoLinkManipulator( RDDS.DynamicSystem ) :
    """ 2DOF Manipulator Class """
    
    
    ############################
    def __init__(self, n = 4 , m = 2 ):
        
        RDDS.DynamicSystem.__init__(self, n , m )
        
        self.state_label = ['Angle 1','Angle 2','Speed 1','Speed 2']
        self.input_label = ['Torque 1','Torque 2']
        
        self.state_units = ['[rad]','[rad]','[rad/sec]','[rad/sec]']
        self.input_units = ['[Nm]','[Nm]']
        
        self.x_ub = np.array([ 6, 6, 6, 6])    # States Upper Bounds
        self.x_lb = np.array([-6,-6,-6,-6])    # States Lower Bounds
        
        tmax = 1
        
        self.u_ub = np.array([ tmax, tmax])      # Control Upper Bounds
        self.u_lb = np.array([-tmax,-tmax])      # Control Lower Bounds
        
        # Default State and inputs        
        self.xbar = np.zeros(n)
        self.ubar = np.array([0,0])
        
        self.setparams()
        
        
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        self.l1  = 1 
        self.l2  = 1
        self.lc1 = 1
        self.lc2 = 1
        
        self.m1 = 1
        self.I1 = 1
        self.m2 = 1
        self.I2 = 1
        
        self.g = 9.81
        
        self.d1 = 1
        self.d2 = 1
        
        
    ##############################
    def trig(self, q = np.zeros(2)):
        """ Compute cos and sin """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        
        return [c1,s1,c2,s2,c12,s12]
        
        
    ##############################
    def fwd_kinematic(self, q = np.zeros(2)):
        """ Compute [x;y] end effector position given angles q """
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        # Three robot points
        
        x0 = 0
        y0 = 0
        
        x1 = self.l1 * s1
        y1 = self.l1 * c1
        
        x2 = self.l1 * s1 + self.l2 * s12
        y2 = self.l1 * c1 + self.l2 * c12
        
        return np.array([[x0,y0],[x1,y1],[x2,y2]])
    
    
    ##############################
    def jacobian_endeffector(self, q = np.zeros(2)):
        """ Compute jacobian of end-effector """
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        J = np.zeros((2,2))
        
        J[0,0] =  self.l1 * c1 + self.l2 * c12
        J[1,0] = -self.l1 * s1 - self.l2 * s12
        J[0,1] =  self.l2 * c12
        J[1,1] = -self.l2 * s12
        
        return J
        
        
    ##############################
    def H(self, q = np.zeros(2)):
        """ Inertia matrix """  
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        H = np.zeros((2,2))
        
        H[0,0] = self.m1 * self.lc1**2 + self.I1 + self.m2 * ( self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * c2 ) + self.I2
        H[1,0] = self.m2 * self.lc2**2 + self.m2 * self.l1 * self.lc2 * c2 + self.I2
        H[0,1] = H[1,0]
        H[1,1] = self.m2 * self.lc2 ** 2 + self.I2
        
        return H
        
        
    ##############################
    def C(self, q = np.zeros(2) ,  dq = np.zeros(2) ):
        """ Corriolis Matrix """  
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        h = self.m2 * self.l1 * self.lc2 * s2
        
        C = np.zeros((2,2))
        
        C[0,0] = - h  * dq[1]
        C[1,0] =   h  * dq[0]
        C[0,1] = - h * ( dq[0] + dq[1] )
        C[1,1] = 0
        
        return C
        
        
    ##############################
    def D(self, q = np.zeros(2) ,  dq = np.zeros(2) ):
        """ Damping Matrix """  
               
        D = np.zeros((2,2))
        
        D[0,0] = self.d1
        D[1,0] = 0
        D[0,1] = 0
        D[1,1] = self.d2
        
        return D
        
        
    ##############################
    def G(self, q = np.zeros(2) ):
        """Gravity forces """  
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1 ) * self.g
        g2 = self.m2 * self.lc2 * self.g
        
        G = np.zeros(2)
        
        G[0] = - g1 * s1 - g2 * s12
        G[1] = - g2 * s12
        
        return G
        
    
    ##############################
    def F(self, q = np.zeros(2) , dq = np.zeros(2) , ddq = np.zeros(2)):
        """ Computed torques given a trajectory (inverse dynamic) """  
        
        H = self.H( q )
        C = self.C( q , dq )
        D = self.D( q , dq )
        G = self.G( q )
        
        F = np.dot( H , ddq ) + np.dot( C , dq ) + np.dot( D , dq ) + G
        
        return F
        
        
    ##############################
    def ddq(self, q = np.zeros(2) , dq = np.zeros(2) , F = np.zeros(2)):
        """ Computed accelerations given torques"""  
        
        H = self.H( q )
        C = self.C( q , dq )
        D = self.D( q , dq )
        G = self.G( q )
        
        ddq = np.dot( np.linalg.inv( H ) ,  ( F - np.dot( C , dq ) - np.dot( D , dq ) - G ) )
        
        return ddq
        
        
    #############################
    def fc(self, x = np.zeros(4) , u = np.zeros(2) , t = 0 ):
        """ 
        Continuous time function evaluation
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vectror n x 1
        
        """
        
        dx = np.zeros(self.n) # State derivative vector
        
        q  = x[0:2]
        dq = x[2:4]
        
        ddq = self.ddq( q , dq , u )
        
        dx[0:2] = dq
        dx[2:4] = ddq
        
        return dx
        
        
    ##############################
    def e_kinetic(self, q = np.zeros(2) , dq = np.zeros(2) ):
        """ Compute kinetic energy of manipulator """  
        
        e_k = 0.5 * np.dot( dq , np.dot( self.H( q ) , dq ) )
        
        return e_k
        
        
    ##############################
    def e_potential(self, q = np.zeros(2) ):
        """ Compute potential energy of manipulator """  
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1 ) * self.g
        g2 = self.m2 * self.lc2 * self.g
        
        e_p = g1 * c1 + g2 * c12        
        
        return e_p
        
    
    ##############################
    def energy_values(self, x = np.zeros(4)  ):
        """ Compute energy values of manipulator """ 
        
        q  = x[0:2]
        dq = x[2:4]
        
        e_k = self.e_kinetic( q , dq )
        e_p = self.e_potential( q )
        e   = e_k + e_p
        
        return [ e , e_k , e_p ]
        
        
    #############################
    def show(self, q):
        """ Plot figure of configuration q """
        
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax.grid()
        
        pts = self.fwd_kinematic( q )
        
        line = ax.plot( pts[:,0], pts[:,1], 'o-', lw=(self.l1+self.l2) )
        
        plt.show()
        
        return fig , ax, line
        
        
    #############################
    def show_jaco(self, q ):
        """ Plot figure of configuration q """
        
        self.show_matrix( self.jacobian_endeffector(q)  , q )
        
        
    #############################
    def show_matrix(self, J , q , plotvector = False ):
        """ Plot figure of configuration q """
        
        # Plot manipulator
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax.grid()
        
        pts = self.fwd_kinematic( q )
        
        line = ax.plot( pts[:,0], pts[:,1], 'o-', lw=(self.l1+self.l2) )
        
        u,s,v = np.linalg.svd( J )
        
        theta = np.arctan2( u[1,0] , u[0,0] ) * 180/3.1416
        
        V1 = u[:,0] * s[0]
        V2 = u[:,1] * s[1]
        
        #print theta,V1,V2,s[0],s[1]
        if plotvector:
            line1 = ax.plot( (pts[2,0],pts[2,0]+V1[0]), (pts[2,1],pts[2,1]+V1[1]), '-')
            line2 = ax.plot( (pts[2,0],pts[2,0]+V2[0]), (pts[2,1],pts[2,1]+V2[1]), '-' )
        
        e = matplotlib.patches.Ellipse(xy=(pts[2,0],pts[2,1]), width=s[0]+0.01, height=s[1]+0.01, angle=theta)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.1)
        e.set_facecolor('b')
        
        
        plt.show()
        
        return fig , ax, line
        
    
    #############################
    def plotAnimation(self, x0 = np.array([0,0,0,0]) , tf = 10 , n = 201 , solver = 'ode' ,  save = False , file_name = 'RobotSim'  ):
        """ Simulate and animate robot """
        
        
        # Integrate EoM
        self.Sim    = RDDS.Simulation( self , tf , n , solver )
        self.Sim.x0 = x0
        
        self.Sim.compute()
        
        # Compute pts localization
        self.PTS = np.zeros((3,2,n))
        
        for i in xrange(n):
            self.PTS[:,:,i] = self.fwd_kinematic( self.Sim.x_sol_CL[i,0:2] ) # Forward kinematic
            
            
        # figure
            
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        self.ax.grid()
        
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.time_template = 'time = %.1fs'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
            
        inter = int( n / 8. )
        #interval=25 with n=201
                    
        self.ani = animation.FuncAnimation( self.fig, self.__animate__, n, interval = inter, blit=True, init_func=self.__ani_init__)
        
        if save:
            self.ani.save( file_name + '.mp4' ) # , writer = 'mencoder' )
            
        plt.show()
        
        return self.PTS
        
        
        
    def __ani_init__(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text
    
    
    def __animate__(self,i):
        thisx = self.PTS[:,0,i]
        thisy = self.PTS[:,1,i]
    
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (i*0.05))
        return self.line, self.time_text
        
    def __animateStop__(self,i):
        
        if i > 198: # To close window
            plt.close()
        thisx = self.PTS[:,0,i]
        thisy = self.PTS[:,1,i]
    
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (i*0.05))
        return self.line, self.time_text

            
'''
#################################################################
##################       1DOF Manipulator Class          ########
#################################################################
'''


class OneLinkManipulator( RDDS.DynamicSystem ) :
    """ 2DOF Manipulator Class """
    
    
    ############################
    def __init__(self, n = 2 , m = 1 ):
        
        RDDS.DynamicSystem.__init__(self, n , m )
        
        self.state_label = ['Angle 1','Speed 1']
        self.input_label = ['Torque 1']
        
        self.state_units = ['[rad]','[rad/sec]']
        self.input_units = ['[Nm]']
        
        self.x_ub = np.array([ 2*np.pi , 2*np.pi])    # States Upper Bounds
        self.x_lb = np.array([-2*np.pi,-2*np.pi])    # States Lower Bounds
        
        tmax = 10
        
        self.u_ub = np.array([ tmax])      # Control Upper Bounds
        self.u_lb = np.array([-tmax])      # Control Lower Bounds
        
        # Default State and inputs        
        self.xbar = np.zeros(n)
        self.ubar = np.array([0])
        
        self.setparams()
        
        
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        self.l1  = 1 
        self.lc1 = 1
        
        self.m1 = 1
        self.I1 = 1
        
        # load
        self.M  = 1
        
        self.g = 9.81
        
        self.d1 = 0
        
        
    ##############################
    def trig(self, q ):
        """ Compute cos and sin """
        
        c1  = np.cos( q )
        s1  = np.sin( q )

        
        return [c1,s1]
        
        
    ##############################
    def fwd_kinematic(self, q = np.zeros(1) ):
        """ Compute [x;y] end effector position given angles q """
        
        [c1,s1] = self.trig( q )
        
        # Three robot points
        
        x0 = 0
        y0 = 0
        
        x1 = self.l1 * s1
        y1 = self.l1 * c1
        
        return np.array([[x0,y0],[x1,y1]])
    
    
    ##############################
    def jacobian_endeffector(self, q = np.zeros(1)):
        """ Compute jacobian of end-effector """
        
        [c1,s1] = self.trig( q )
        
        J = np.zeros((2,1))
        
        J[0] =  self.l1 * c1 
        J[1] = -self.l1 * s1 
        
        return J
        
        
    ##############################
    def H(self, q = np.zeros(1)):
        """ Inertia matrix """  
        
        H = self.m1 * self.lc1**2 + self.I1 + self.M * ( self.l1**2 )
        
        return H
        
        
    ##############################
    def C(self, q = np.zeros(1) ,  dq = np.zeros(1) ):
        """ Corriolis Matrix """  
        
        C = 0

        
        return C
        
        
    ##############################
    def D(self, q = np.zeros(1) ,  dq = np.zeros(1) ):
        """ Damping Matrix """  
               
        D = 0
        
        return D
        
        
    ##############################
    def G(self, q = np.zeros(1) ):
        """Gravity forces """  
        
        [c1,s1] = self.trig( q )
        
        g1 = (self.m1 * self.lc1 + self.M * self.l1 ) * self.g
        
        G = - g1 * s1 
        
        return G
        
    ##############################
    def e_kinetic(self, q = np.zeros(1) , dq = np.zeros(1) ):
        """ Compute kinetic energy of manipulator """  
        
        e_k = 0.5 * np.dot( dq , np.dot( self.H( q ) , dq ) )
        
        return e_k
        
        
    ##############################
    def e_potential(self, q = np.zeros(2) ):
        """ Compute potential energy of manipulator """  
        
        [ c1 , s1 ] = self.trig( q )
        
        g1 = (self.m1 * self.lc1 + self.M * self.l1 ) * self.g
        
        e_p = g1 * c1   
        
        return e_p
        
    ##############################
    def energy_values(self, x = np.zeros(1)  ):
        """ Compute energy values of manipulator """ 
        
        q  = x[0]
        dq = x[1]
        
        e_k = self.e_kinetic( q , dq )
        e_p = self.e_potential( q )
        e   = e_k + e_p
        
        return [ e , e_k , e_p ]
        
    
    ##############################
    def F(self, q = np.zeros(1) , dq = np.zeros(1) , ddq = np.zeros(1)):
        """ Computed torques given a trajectory (inverse dynamic) """  
        
        H = self.H( q )
        C = self.C( q , dq )
        D = self.D( q , dq )
        G = self.G( q )
        
        F = np.dot( H , ddq ) + np.dot( C , dq ) + np.dot( D , dq ) + G
        
        return F
        
        
    ##############################
    def ddq(self, q = np.zeros(1) , dq = np.zeros(1) , F = np.zeros(1) ):
        """ Computed accelerations given torques"""  
        
        H = self.H( q )
        C = self.C( q , dq )
        D = self.D( q , dq )
        G = self.G( q )
        
        ddq = np.dot( 1./H ,  ( F - np.dot( C , dq ) - np.dot( D , dq ) - G  ) )
        
        return ddq
        
        
    #############################
    def fc(self, x = np.zeros(2) , u = np.zeros(1) , t = 0 ):
        """ 
        Continuous time function evaluation
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vectror n x 1
        
        """
        
        dx = np.zeros(self.n) # State derivative vector
        
        q  = x[0]
        dq = x[1]
        
        ddq = self.ddq( q , dq , u[0] )
        
        dx[0] = dq
        dx[1] = ddq
        
        return dx
        
        
    #############################
    def show(self, q):
        """ Plot figure of configuration q """
        
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax.grid()
        
        pts = self.fwd_kinematic( q )
        
        line = ax.plot( pts[:,0], pts[:,1], 'o-', lw=(self.l1) )
        
        plt.show()
        
        return fig , ax, line
        
  
    
    #############################
    def plotAnimation(self, x0 = np.array([0,0]) , tf = 10 , n = 201 ,  solver = 'ode' , save = False , file_name = 'RobotSim' ):
        """ Simulate and animate robot """
        
        
        # Integrate EoM
        self.Sim    = RDDS.Simulation( self , tf , n  , solver  )
        self.Sim.x0 = x0
        
        self.Sim.compute()
        
        
        
        # Compute pts localization
        self.PTS = np.zeros((2,2,n))
        
        for i in xrange(n):
            self.PTS[:,:,i] = self.fwd_kinematic( self.Sim.x_sol_CL[i,0] ) # Forward kinematic
            
            
        # figure
            
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        self.ax.grid()
        
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.time_template = 'time = %.1fs'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

        inter = int( n / 8. )
        #interval=25 with n=201
                    
        self.ani = animation.FuncAnimation( self.fig, self.__animate__, n, interval = inter , blit=True, init_func=self.__ani_init__)
        
        if save:
            self.ani.save( file_name + '.mp4' ) #, writer = 'mencoder' )
            
        plt.show()
        
        return self.PTS
        
    def __ani_init__(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text
    
    
    def __animate__(self,i):
        thisx = self.PTS[:,0,i]
        thisy = self.PTS[:,1,i]
    
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (i*0.05))
        return self.line, self.time_text
        
    def __animateStop__(self,i):
        
        if i > 198: # To close window
            plt.close()
        thisx = self.PTS[:,0,i]
        thisy = self.PTS[:,1,i]
    
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (i*0.05))
        return self.line, self.time_text
        
        




'''
#################################################################
##################       3DOF Manipulator Class          ########
#################################################################
'''



class ThreeLinkManipulator( TwoLinkManipulator ) :
    """
    3DOF Manipulator Class 
    -------------------------------
    
    base:     revolute arround z
    shoulder: revolute arround y
    elbow:    revolute arround y
    
    see Example 4.3 in
    http://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-manipdyn_v1_2.pdf
    
    
    """
    
    
    ############################
    def __init__(self, n = 6 , m = 3 ):
        
        RDDS.DynamicSystem.__init__(self, n , m )
        
        self.state_label = ['Angle 1','Angle 2','Angle 3','Speed 1','Speed 2','Speed 3']
        self.input_label = ['Torque 1','Torque 2','Torque 3']
        
        self.state_units = ['[rad]','[rad]','[rad]','[rad/sec]','[rad/sec]','[rad/sec]']
        self.input_units = ['[Nm]','[Nm]','[Nm]']
        
        self.x_ub = np.array([ 6, 6, 6, 6, 6, 6])    # States Upper Bounds
        self.x_lb = np.array([-6,-6,-6,-6,-6,-6])    # States Lower Bounds
        
        tmax = 1
        
        self.u_ub = np.array([ tmax, tmax, tmax])      # Control Upper Bounds
        self.u_lb = np.array([-tmax,-tmax,-tmax])      # Control Lower Bounds
        
        # Default State and inputs        
        self.xbar = np.zeros(n)
        self.ubar = np.array([0,0,0])
        
        self.setparams()
        
        
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        # Kinematic
        self.l1  = 1 
        self.l2  = 1
        self.l3  = 1
        self.lc1 = 1
        self.lc2 = 1
        self.lc3 = 1
        
        #Inertia
        self.m1 = 1
        self.I1 = 1
        self.m2 = 1
        self.I2 = 1
        self.m3 = 1
        self.I3 = 1
        
        # Gravity
        self.g = 9.81
        
        # Joint damping
        self.d1 = 1
        self.d2 = 1
        self.d3 = 1
        
        
    ##############################
    def trig(self, q = np.zeros(3) ):
        """ Compute cos and sin """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )
        c3  = np.cos( q[2] )
        s3  = np.sin( q[2] )
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        c23 = np.cos( q[2] + q[1] )
        s23 = np.sin( q[2] + q[1] )
        
        return [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23]
        
        
    ##############################
    def fwd_kinematic(self, q = np.zeros(3) ):
        """ Compute [x;y;z] end effector position given angles q """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        # Three robot points
        
        #Base of the robot
        p0 = [0,0,0]
        
        #Shperical point 
        p1 = [ 0, 0, self.l1 ]
        
        #elbow
        z2 = self.l1 - self.l2 * s2
        
        r2 = self.l2 * c2
        x2 = r2 * c1
        y2 = r2 * s1
        
        p2 = [ x2, y2, z2 ]
        
        # end-effector
        z3 = self.l1 - self.l2 * s2 - self.l3 * s23
        
        r3 = self.l2 * c2 + self.l3 * c23
        x3 = r3 * c1
        y3 = r3 * s1
        
        p3 = [ x3, y3, z3 ]
        
        return np.array([p0,p1,p2,p3])
        
        
    #############################
    def show(self, q, plane = 'xy' ):
        """ Plot figure of configuration q """
        
        pts = self.fwd_kinematic( q )
        lw=(self.l1+self.l2+self.l3)
        
        if plane == '3D':
            
            #import matplotlib as mpl
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            line = ax.plot( pts[:,0], pts[:,1], pts[:,2], 'o-', lw)         
            plt.show()
            
        else:
        
            fig = plt.figure()
            ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
            ax.grid()
            
            if plane == 'xy':
                line = ax.plot( pts[:,0], pts[:,1], 'o-', lw )
            elif plane == 'xz':
                line = ax.plot( pts[:,0], pts[:,2], 'o-', lw )
            elif plane == 'yz':
                line = ax.plot( pts[:,1], pts[:,2], 'o-', lw )
            else:
                pass
            
            plt.show()
            
            #return fig , ax, line
            
    ##############################
    def jacobian_endeffector(self, q = np.zeros(3)):
        """ Compute jacobian of end-effector """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        J = np.zeros((3,3))
        
        J[0,0] =  -( self.l2 * c2 + self.l3 * c23 ) * s1
        J[0,1] =  -( self.l2 * s2 + self.l3 * s23 ) * c1
        J[0,2] =  - self.l3 * s23 * c1
        J[1,0] =   ( self.l2 * c2 + self.l3 * c23 ) * c1
        J[1,1] =  -( self.l2 * s2 + self.l3 * s23 ) * s1
        J[1,2] =  - self.l3 * s23 * s1
        J[2,0] =  0
        J[2,1] =  -( self.l2 * c2 + self.l3 * c23 )
        J[2,2] =  - self.l3 * c23
        
        return J
        
        


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    
    
    R1 = OneLinkManipulator()
    R2 = TwoLinkManipulator()
    R3 = ThreeLinkManipulator()

            