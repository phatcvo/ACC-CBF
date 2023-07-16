import numpy as np
from matplotlib import pyplot as plt
from utils import rk4
import osqp
from scipy import sparse
import copy

#parameters for lead and ego vehicle
c0 = 0.1
c1 = 5.0
c2 = 0.25

m_ego = 1650.0
m_lead = 1650.0

ca = 0.4
cd = 0.4
g = 9.81

v_max = 30.0
v_min = 0.0

v_lead = 13.89
v_des = 24

#parameters for HOCBF
epsilon = 10.0
p = 1
delta = 10.0

#dynamics for lead and ego vehicles
def f(t,x,u):
    u_ego = u[0]
    v_lead = u[1]

    x_ego = x[0]
    v_ego = x[1]
    x_lead = x[2]

    Fr_ego = c0 * np.sign(v_ego) + c1 * v_ego + c2 * v_ego ** 2
    return np.array([v_ego,
            1/m_ego * (-Fr_ego + u_ego),
            v_lead])

def get_state_dependent_matrices_for_qp(x):
    #note that we need to change this method to update only the data in A
    #not create a new sparse csc object in order to use OSQP's update function
    x_lead = copy.copy(x[2])
    x_ego = copy.copy(x[0])
    v_ego = copy.copy(x[1])
    Fr_ego = c0 * np.sign(v_ego) + c1 * v_ego + c2 * v_ego ** 2
    q = np.array([-2 * Fr_ego / m_ego ** 2,0])

    #lyapunov control
    mLgVacc = 2*(v_ego-v_des)
    mLfVacc = -2*(v_ego-v_des) * Fr_ego
    Vacc = (v_ego-v_des)**2

    #safety barrier
    bdot = v_lead-v_ego
    b = x_lead-x_ego-delta


    A = sparse.csc_matrix([[1,0],[0,1],
                           [mLgVacc,-m_ego],
                           [1,0],
                           [-1,0],
                           [1,0]])
    u = np.array([ca*m_ego*g,np.inf,
                  -mLfVacc-m_ego*epsilon*Vacc,
                  Fr_ego+m_ego*(v_max-v_ego),
                  -Fr_ego+m_ego*(v_ego-v_min),
                  Fr_ego+m_ego*(2*p*bdot+p**2*b)])
    return q, A, u

#setup OSQP
pacc = 1.0
prob = osqp.OSQP()
P = sparse.csc_matrix([[2/m_ego**2,0],[0,2*pacc]])

l = np.concatenate((np.array([-cd*m_ego*g,0]),-np.inf * np.ones(4)))
x0 = [0.0,20.0,100.0]


#simulation parameters
T = 30
dt = 0.1

tvec = np.arange(0,T+dt,dt)
N = len(tvec)
X = np.zeros((N,3))
X[0,:] = x0
U = np.zeros((N-1,2))
delta_acc = []
for i in range(len(tvec)-1):
    #note that we need to change the below method to update only the data in A
    #not create a new sparse csc object in order to use OSQP's update function

    q_new, A_new, u_new = get_state_dependent_matrices_for_qp(X[i,:])

    prob = osqp.OSQP()
    prob.setup(P, q_new, A_new, l, u_new)
    res = prob.solve()

    delta_acc.append(res.x[1])
    u_ego = res.x[0]

    U[i,:] = [u_ego,v_lead]

    X[i+1,:] = rk4(f,tvec[i],X[i,:],U[i,:],dt)


#plot results
fig,ax = plt.subplots(nrows=2,ncols=2)

ax[0,0].plot(tvec,X[:,2]-X[:,0])
ax[0,0].set_ylabel('xlead-xego [m]')
ax[0,0].plot(tvec,delta*np.ones(N),'r--')

ax[1,0].plot(tvec,v_lead-X[:,1]+p*(X[:,2]-X[:,0]-delta))
ax[1,0].set_ylabel('$\psi$(x(t))$_1$')
ax[1,0].plot(tvec,delta*np.zeros(N),'r--')

ax[0,1].plot(tvec,X[:,1])
ax[0,1].set_ylabel('v [m/s]')
ax[0,1].plot(tvec,v_max*np.ones(N),'r--')
ax[0,1].plot(tvec,v_min*np.ones(N),'r--')


ax[1,1].plot(tvec[:-1],U[:,0]/m_ego)
ax[1,1].plot(tvec,ca*g*np.ones(N),'r--')
ax[1,1].plot(tvec,-cd*g*np.ones(N),'r--')
ax[1,1].set_xlabel('t [s]')
ax[1,1].set_ylabel('u/m [m/s$^2$]')

fig.tight_layout()

fig2,ax2 = plt.subplots(1)
ax2.plot(delta_acc)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('$\delta$ acc')
ax2.set_title('Slack variable for CLF')
fig2.tight_layout()
plt.show()




