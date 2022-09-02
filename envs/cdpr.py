import numpy as np
import scipy as scp
from scipy import linalg
import math as m
from matplotlib import pyplot as plt
from numpy.core.memmap import dtype


import gym
from gym import logger, spaces
# from gym.envs.classic_control import utils
# from gym.error import DependencyNotInstalled
# from gym.utils.renderer import Renderer
# from typing import Optional, Union

#base model
class PRPRmodel():
  def __init__(self):
    self.base_length = 0.860
    self.base_span = 0.570
    self.phiA = 0
    self.phiB = np.pi/2
    self.phiC = np.pi
    self.phiD = 3*np.pi/2

    self.A1 = np.array([self.base_length/2-self.base_span/2,0]).T
    self.A2 = np.array([self.base_length/2+self.base_span/2,0]).T

    self.B1 = np.array([self.base_length,self.base_length/2-self.base_span/2]).T
    self.B2 = np.array([self.base_length,self.base_length/2+self.base_span/2]).T

    self.C1 = np.array([self.base_length/2+self.base_span/2,self.base_length]).T
    self.C2 = np.array( [self.base_length/2-self.base_span/2,self.base_length]).T

    self.D1 = np.array([0,self.base_length/2+self.base_span/2]).T
    self.D2 = np.array([0,self.base_length/2-self.base_span/2]).T
        
    self.b = 0.16
    self.h = 0.04

    self.EA_O = np.array([-self.b/2, -self.h/2]).T
    self.EB_O = np.array([self.b/2, -self.h/2]).T
    self.EC_O = np.array([self.b/2, self.h/2]).T
    self.ED_O = np.array([-self.b/2, self.h/2]).T

    self.P = np.array([0,0]).T

    self.m = 0.1
    self.dx = 0.5
    self.dy = 0.5
    self.dz = 0.01
    self.dtdyn = 0.001
    self.Ts = 0.05

    self.Tf = 10 #self.Ts*self.nsteps

    self.M = np.array([[self.m, 0, 0], [0, self.m, 0], [0, 0, self.b*self.h*(self.b**2+self.h**2)/12]])
    self.D = np.array([[self.dx, 0, 0],[0, self.dy, 0], [0, 0, self.dz]])

    self.l_dot_min = np.array([-0.05,-0.05,-0.05,-0.05]).T
    self.l_dot_max = np.array([0.05,0.05,0.05,0.05]).T

    self.tau_dot_min = np.array([-0.05,-0.05,-0.05,-0.05]).T
    self.tau_dot_max = np.array([0.05,0.05,0.05,0.05]).T

  def plotCDPR(self,X,l):
    EA_P,EB_P,EC_P,ED_P = self.platformCoordinates(X)   
    A3,B3,C3,D3 = self.sliderPositions(l)


    plt.xlim(-0.2, 1)
    plt.ylim(-0.2, 1)

    coords = [self.A1,self.A2,self.B1,self.B2,self.C1,self.C2,self.D1,self.D2,self.A1]
    xr,yr = zip(*coords)
    plt.plot(xr,yr,'b-')

    coords = [EA_P,EB_P,EC_P,ED_P,EA_P]
    xp,yp = zip(*coords)
    plt.plot(xp,yp,'b-')

    plt.plot([A3[0],EA_P[0]],[A3[1],EA_P[1]],'k-')
    plt.plot([B3[0],EB_P[0]],[B3[1],EB_P[1]],'k-')
    plt.plot([C3[0],EC_P[0]],[C3[1],EC_P[1]],'k-')
    plt.plot([D3[0],ED_P[0]],[D3[1],ED_P[1]],'k-')
    plt.axis('equal')

    plt.show()

  def sliderPositions(self,l):
    A3 = self.P + self.A1 + np.matmul(np.array([[m.cos(self.phiA), -m.sin(self.phiA)],[m.sin(self.phiA), m.cos(self.phiA)]]),np.array([l[0],0]).T)
    B3 = self.P + self.B1 + np.matmul(np.array([[m.cos(self.phiB), -m.sin(self.phiB)],[m.sin(self.phiB), m.cos(self.phiB)]]),np.array([l[1],0]).T)
    C3 = self.P + self.C1 + np.matmul(np.array([[m.cos(self.phiC), -m.sin(self.phiC)],[m.sin(self.phiC), m.cos(self.phiC)]]),np.array([l[2],0]).T)
    D3 = self.P + self.D1 + np.matmul(np.array([[m.cos(self.phiD), -m.sin(self.phiD)],[m.sin(self.phiD), m.cos(self.phiD)]]),np.array([l[3],0]).T)
    return A3,B3,C3,D3;

  def platformCoordinates(self,X):
    pGo = [[m.cos(X[2]), -m.sin(X[2]), X[0]], [m.sin(X[2]), m.cos(X[2]), X[1]], [0,0,1]]

    t1 = np.matmul(pGo,np.array([self.EA_O[0],self.EA_O[1],1]).T)
    t2 = np.matmul(pGo,np.array([self.EB_O[0],self.EB_O[1],1]).T)
    t3 = np.matmul(pGo,np.array([self.EC_O[0],self.EC_O[1],1]).T)
    t4 = np.matmul(pGo,np.array([self.ED_O[0],self.ED_O[1],1]).T)

    EA_P = t1[0:2]
    EB_P = t2[0:2]
    EC_P = t3[0:2]
    ED_P = t4[0:2]

    return EA_P,EB_P,EC_P,ED_P

  def prismaticLength(self,X,l):
    EA_P,EB_P,EC_P,ED_P = self.platformCoordinates(X)   
    A3,B3,C3,D3 = self.sliderPositions(l)

    la = np.sqrt((A3[0]-EA_P[0])**2+(A3[1]-EA_P[1])**2)
    lb = np.sqrt((B3[0]-EB_P[0])**2+(B3[1]-EB_P[1])**2)
    lc = np.sqrt((C3[0]-EC_P[0])**2+(C3[1]-EC_P[1])**2)
    ld = np.sqrt((D3[0]-ED_P[0])**2+(D3[1]-ED_P[1])**2)

    l_p = [la,lb,lc,ld]

    return l_p

  def structureMatrix(self,X,l):

    A3,B3,C3,D3 = self.sliderPositions(l); 
    
    phi_e = X[2]
    #print(phi_e, "phi_e")
    # print(m.cos(phi_e))
    # print(-m.sin(phi_e))
    # print(m.sin(phi_e))
    # print(m.cos(phi_e))
    pRo = np.array([[m.cos(phi_e), -m.sin(phi_e)],[m.sin(phi_e), m.cos(phi_e)]]) #rot_z
            
    la = A3 - X[0:1] - np.matmul(pRo,self.EA_O)
    lb = B3 - X[0:1] - np.matmul(pRo,self.EB_O)
    lc = C3 - X[0:1] - np.matmul(pRo,self.EC_O)
    ld = D3 - X[0:1] - np.matmul(pRo,self.ED_O)
    
    ua = np.array([np.array(la/np.linalg.norm(la))]).T
    ub = np.array([np.array(lb/np.linalg.norm(lb))]).T
    uc = np.array([np.array(lc/np.linalg.norm(lc))]).T
    ud = np.array([np.array(ld/np.linalg.norm(ld))]).T
    
    r1 = np.array([np.matmul(pRo,self.EA_O)]).T
    r2 = np.array([np.matmul(pRo,self.EB_O)]).T
    r3 = np.array([np.matmul(pRo,self.EC_O)]).T
    r4 = np.array([np.matmul(pRo,self.ED_O)]).T
   
    baxua = np.cross(r1.T,ua.T).tolist()
    bbxub = np.cross(r2.T,ub.T).tolist()
    bcxuc = np.cross(r3.T,uc.T).tolist()
    bdxud = np.cross(r4.T,ud.T).tolist()

    #print(np.concatenate((baxua,bbxub,bcxuc,bdxud), axis=0),'baxua')

    Jw = np.vstack((np.hstack((ua, ub, uc, ud)),[baxua[0],bbxub[0],bcxuc[0],bdxud[0]]))

    return Jw;

  def fwd_dyn(self,X_dyn,l_des,l0,tau_des,tau0,Ts):
    j = 1
    l = l0
    tau = tau0
    l_dot = (l_des - l0)/self.dtdyn;
    l_dot_min = np.array([-0.05,-0.05,-0.05,-0.05]).T
    l_dot_max = np.array([0.05,0.05,0.05,0.05]).T
    l_dot = np.array([-0.01,2,-0.1,4]).T
    l_dot = [max(x, y) for x, y in zip(l_dot, l_dot_min)] 
    l_dot = [min(x, y) for x, y in zip(l_dot, l_dot_max)]
    tau_dot = (tau_des - tau0)/self.dtdyn;
    tau_dot_min = np.array([-0.05,-0.05,-0.05,-0.05]).T
    tau_dot_max = np.array([0.05,0.05,0.05,0.05]).T
    tau_dot = np.array([-0.01,2,-0.1,4]).T
    tau_dot = [max(x, y) for x, y in zip(tau_dot, tau_dot_min)] 
    tau_dot = [min(x, y) for x, y in zip(tau_dot, tau_dot_max)]

    for j in range(1,int(Ts/self.dtdyn)):
      P_x = self.structureMatrix(X_dyn[0:3],l)
      l_p_x = self.prismaticLength(X_dyn[0:3],l)
      x_dot = X_dyn[3:6];
      x_ddot = np.matmul(np.linalg.inv(self.M),np.matmul(P_x,tau)) - np.matmul(np.linalg.inv(self.D),x_dot)
      X_dyn[0:3] = X_dyn[0:3] + x_dot*self.dtdyn
      X_dyn[3:6] = x_dot + x_ddot*self.dtdyn

      l = l + np.array(l_dot)*self.dtdyn;
      tau = tau + np.array(tau_dot)*self.dtdyn;

    return X_dyn,l,tau

  def bezier_curve(self,P0,P1,P2,Ts,Tf):
    n = int(np.floor(Tf/Ts))
    t = np.linspace(0,1,n).reshape((1,n))   
    B = (1.-t) * (P0 * (1.-t) + P1 *t) + t*((1.-t) * P1 + (t * P2))
    B_dot = t * (2*P0 - 4*P1 + 2*P2) + (-2*P0 + 2*P1); 
    B_dot = B_dot/Tf
    return B,B_dot

  def poseQuality(self,X,l):
    Jw = self.structureMatrix(X,l)
    z = linalg.null_space(Jw)
    if min(z) > 0.:
      kappa = min(z)/max(z)
    elif max(z) < 0.:
      kappa = max(z)/min(z)
    else:
      kappa = min(z)/max(z)
    
    s = linalg.svdvals(Jw[0:2,:])
    mom = np.array([min(s)/max(s)], dtype=np.float32)
    if kappa < 0.:
      mom = np.array([0.], dtype=np.float32)

    return min(kappa,mom)


class CDPRenv(PRPRmodel,gym.Env):
  def __init__(self):
    super().__init__()
    self.nsteps = 200
    self.Tf = self.Ts*self.nsteps
    self.high = np.array(
        [np.finfo(np.float32).max, #x_e
         np.finfo(np.float32).max, #y_e
         np.finfo(np.float32).max, #phi_e
         np.finfo(np.float32).max, #\dot x_e
         np.finfo(np.float32).max, #\dot y_e
         np.finfo(np.float32).max, #\dot phi_e
         np.finfo(np.float32).max, #x_e^t
         np.finfo(np.float32).max, #y_e^t
         np.finfo(np.float32).max, #\dot x_e^t
         np.finfo(np.float32).max, #\dot y_e^t
         2., # kappa+mon
         1., # tensions
         1., # tensions
         1., # tensions
         1., # tensions
         0.57, # slider positions
         0.57, # slider positions
         0.57, # slider positions
         0.57 # slider positions
         ])
    self.low = np.array(
        [-np.finfo(np.float32).max, #x_e
         -np.finfo(np.float32).max, #y_e
         -np.finfo(np.float32).max, #phi_e
         -np.finfo(np.float32).max, #\dot x_e
         -np.finfo(np.float32).max, #\dot y_e
         -np.finfo(np.float32).max, #\dot phi_e
         -np.finfo(np.float32).max, #x_e^t
         -np.finfo(np.float32).max, #y_e^t
         -np.finfo(np.float32).max, #\dot x_e^t
         -np.finfo(np.float32).max, #\dot y_e^t
         0., # kappa+mon
         0.1, # tensions
         0.1, # tensions
         0.1, # tensions
         0.1, # tensions
         0., # slider positions
         0., # slider positions
         0., # slider positions
         0. # slider positions
         ])
    self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    # action space
    self.l_min = np.array([0.,0.,0.,0.])
    self.l_max = np.array([self.base_span,self.base_span,self.base_span,self.base_span])
    self.T_min = np.array([0.1,0.1,0.1,0.1])
    self.T_max = np.array([1.,1.,1.,1.])
    self.action_low = np.array(np.concatenate([self.T_min, self.l_min]), dtype=np.float32)
    self.action_high = np.array(np.concatenate([self.T_max, self.l_max]), dtype=np.float32)
    _act = np.array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)
    self.action_space = spaces.Box(low=-_act, high=_act, dtype=np.float32)

    self.l0 = self.l_min
    self.T0 = self.T_min
    
    self.reset()

  def reset(self):
    # random trajectory:
    self.P0 = 0.4*np.random.rand(1,2).reshape((2,1)) + np.array([0.23,0.23]).reshape((2,1))
    self.P1 = 0.4*np.random.rand(1,2).reshape((2,1)) + np.array([0.23,.23]).reshape((2,1))
    self.P2 = 0.4*np.random.rand(1,2).reshape((2,1)) + np.array([0.23,0.23]).reshape((2,1))

    B,B_dot = self.bezier_curve(self.P0,self.P1,self.P2,self.Ts,self.Tf)
    phi = (np.pi/4)*np.ones((1,len(B.T)))
    phi_dot = np.zeros((1,len(B.T)))
    self.X_des = np.vstack((B,phi,B_dot,phi_dot)) #6 by 100 

    # state reset
    self.state = np.concatenate((self.X_des[:,0],np.array([0.,0.,0.,0.,0.,0.1,0.1,0.1,0.1,0.,0.,0.,0.])),axis=0).T # Observable state info
    self.steps = 0
    self.done = False
    return self.state

  
  def action_scaling(self, action):
    a_v, b_v = self.T_min, self.T_max
    a_w, b_w = self.l_min, self.l_max

    v_s, w_s = action[0:4], action[4:8]
    v = a_v + 0.5*(v_s+1.)*(b_v-a_v)
    w = a_w + 0.5*(w_s+1.)*(b_w-a_w)
    return v, w #np.array([v, w], dtype=np.float32).reshape(1,8)

  def reward(self):

    R0 = np.linalg.norm(self.position_error)
    R1 = 0.
    R2 = 0.
    R3 = np.linalg.norm(self.velocity_error) 
    R4 = np.matmul((np.array([0.1,0.1,0.1,0.1]) - self.T0).T, (np.array([0.1,0.1,0.1,0.1]) - self.T0))
    R5 = (1 - self.kappa[0])

    weights = np.array([1e4,0.,0.,1e3,1e1,1e1]).reshape(1,6)
    reward = np.array([R0,R1,R2,R3,R4,R5]).reshape(1,6)

    total_reward = np.matmul(weights,reward.T)
   
    return -total_reward[0][0]

  def step(self, action, plot = False):
    T_des,l_des = self.action_scaling(action)
    X_dyn0 = self.state[:6]

    if self.steps == 1:
      self.T0 = T_des
      self.l0 = l_des
  
    X_dyn,l1,T1 = self.fwd_dyn(X_dyn0,l_des.T,self.l0,T_des.T,self.T0,self.Ts)
    self.kappa = self.poseQuality(X_dyn[0:3],l1)

    self.T0 = T1
    self.l0 = l1

    

    
    if plot:
      self.plotCDPR(X_dyn[0:3],l1)

    self.position_error = X_dyn[0:2] - self.X_des[0:2,self.steps]
    self.velocity_error = X_dyn[3:5] - self.X_des[3:5,self.steps]

    self.state = np.concatenate((X_dyn,self.position_error,self.velocity_error,self.kappa,T1,l1))

    

    if self.steps == self.nsteps-1:
      self.done = True

    self.steps = self.steps + 1
    return self.state, self.reward(), self.done, {}


