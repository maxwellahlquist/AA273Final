from numpy import sin, cos, tan
import numpy as np
import matplotlib.pyplot as plt
# from numpy import sin as sin

m = 30 # kg
g = 9.81 # m/s^2
lcm = 1.25 * 0.3048 # ft-> m
L = 2.5 * 0.3048    # ft -> m
r = 8 / 12 * 0.3048 # in -> ft -> m
ICxx = 0.25*m*r**2 + 1/12*m*L**2
ICyy = 0.25*m*r**2 + 1/12*m*L**2
ICzz = 0.5*m*r**2

# Spring-damper coefficients
k_psi = 500
k_theta = 30
b_psi = 50
b_theta = 5

dt = 0.005    # sec

# make the process noise dt dependent
# see how the iinstructors do it

# get the names for the 6 k values correct...

def specifiedMotion(t):
    # angular rate wheelchair
    w = 2 # rad/sec

    v = 5
    dv = 0
    phi = 1.570796 + 2.403318*np.sin(w*t)
    dphi = 2.403318*w*np.cos(w*t)
    return v, dv, phi, dphi

def get_Ty(x, t):
#     Tp = k_psi*psi + b_psi*psi'
# Ty = k_theta*theta + b_theta*theta'
    theta, psi, wx, wy = x
    v, dv, phi, dphi = specifiedMotion(t)

    dtheta = -dphi -wx/sin(psi)
    Ty = k_theta*theta + b_theta*dtheta
    return Ty


# Combine this together with dwy, dwx
def dw(x, t):
    
    theta, psi, wx, wy = x
    
    v, dv, phi, dphi = specifiedMotion(t)

    # The other method got rid of the theta' values even though that is what force is based off... maybe it got clobbered and caused problems

    # print(x)
    # print((v, dv, phi, dphi))
    # exit()
    # have a get specfied motion function
    dwx = (m*lcm*(v*dphi*np.cos(theta)-dv*np.sin(theta)-lcm*np.cos(psi)*wx*wy/np.sin(psi))+np.cos(psi)*(ICzz*wx*wy+np.sin(psi)**2*(k_theta*theta+ICzz*wx*wy-ICxx*wx*wy-b_theta*(dphi+wx/np.sin(psi))))/np.sin(psi)**3)/(ICxx+m*lcm**2+ICzz/np.tan(psi)**2)
    # dwx = (m*lcm*(v*dphi*cos(theta)-dv*sin(theta)-lcm*cos(psi)*wx*wy/sin(psi))+cos(psi)*(k_theta*theta+2*ICzz*wx*wy+ICzz*wx*wy/tan(psi)**2-ICxx*wx*wy-b_theta*(dphi+wx/sin(psi)))/sin(psi))/(ICxx+m*lcm**2+ICzz/tan(psi)**2)
    # Updated questionable ^^^ found no diff to the original

    # Ty = get_Ty(x,t)
    # dtheta = -dphi -wx/sin(psi)
    # temp = dwx
    # dwx = (cos(psi)*(Ty+ICzz*(1+1/tan(psi)**2)*wx*wy)/sin(psi)+ICxx*cos(psi)*wy*(dphi+dtheta)-ICzz*cos(psi)*wy*(dphi+dtheta)-m*lcm*(dv*sin(theta)-v*dphi*cos(theta)-lcm*cos(psi)*wy*(2*dphi+wx/sin(psi)+2*dtheta)))/(ICxx+m*lcm**2+ICzz/tan(psi)**2)
    # found no diff to the original even when including Ty
    
    # Include the Ty and Tp ^^^
    # print(temp - dwx)
    

    dwy = -(k_psi*psi+b_psi*wy-m*g*lcm*np.sin(psi)-np.cos(psi)*((ICxx-ICzz)*wx**2/np.sin(psi)-m*lcm*(dv*np.cos(theta)+v*dphi*np.sin(theta)-lcm*wx**2/np.sin(psi))))/(ICyy+m*lcm**2)
    # Do I need this, see what happens if I resubstitute in the Tp, Ty equations
    # if (abs(psi) < 1e-16): 
    #     # the pendulum is vertical, then wx should equal wy
    #     # wz = dtheta' - dphi'
    #     dwz = 0
    dpsi = wy
    dtheta = -dphi - wx/np.sin(psi)
    dwz = -(dphi*dpsi+dpsi*dtheta+np.cos(psi)*dwx)/np.sin(psi)
    return np.array([dwx, dwy, dwz])
    # This one has dwz in it vvv :(
    # dwx  = (m*lcm*(v*dphi*np.cos(theta)-dv*np.sin(theta)-lcm*np.cos(psi)*wx*wy/np.sin(psi))-np.cos(psi)*(ICxx*wx*wy+b_theta*(dphi+wx/np.sin(psi))-k_theta*theta-ICzz*wx*wy-ICzz*dwz)/np.sin(psi))/(ICxx+m*lcm**2)



# jacobian of wx', wy' with respect to theta, psi, wx, wy
def omegaJacobian(x, t):

    Result = np.zeros((2, 4))

    theta, psi, wx, wy = x
    v, dv, phi, dphi = specifiedMotion(t)

    dwx, dwy, dwz = dw(x, t)

    # fixed to remove Ty and Tp
    Result[0,0] = (k_theta*np.cos(psi)/np.sin(psi)-m*lcm*(dv*np.cos(theta)+v*dphi*np.sin(theta)))/(ICxx+m*lcm**2)
    Result[0,1] = -(k_theta*theta+ICzz*wx*wy+ICzz*dwz-ICxx*wx*wy-b_theta*(dphi+wx/np.sin(psi))-2*b_theta*np.cos(psi)**2*wx/np.sin(psi)**3-m*lcm**2*(1+1/np.tan(psi)**2)*wx*wy-(b_theta*dphi+ICxx*wx*wy-k_theta*theta-ICzz*wx*wy-ICzz*dwz)/np.tan(psi)**2)/(ICxx+m*lcm**2)
    Result[0,2] = (ICzz*wy-b_theta/np.sin(psi)-ICxx*wy-m*lcm**2*wy)/((ICxx+m*lcm**2)*np.tan(psi))
    Result[0,3] = (ICzz-ICxx-m*lcm**2)*wx/((ICxx+m*lcm**2)*np.tan(psi))
    Result[1,0] = m*lcm*np.cos(psi)*(dv*np.sin(theta)-v*dphi*np.cos(theta))/(ICyy+m*lcm**2)
    Result[1,1] = -(k_psi+(ICxx-ICzz)*wx**2-m*g*lcm*np.cos(psi)-(ICzz-ICxx-m*lcm**2)*wx**2/np.tan(psi)**2-m*lcm*(np.sin(psi)*(dv*np.cos(theta)+v*dphi*np.sin(theta))-lcm*wx**2))/(ICyy+m*lcm**2)
    Result[1,2] = -2*(ICzz-ICxx-m*lcm**2)*wx/((ICyy+m*lcm**2)*np.tan(psi))
    Result[1,3] = -b_psi/(ICyy+m*lcm**2)
    return Result

def getA(x, t):
    theta, psi, wx, wy = x

    A = np.eye(4)
    A[0, 1] = 1 / (np.tan(psi)*np.sin(psi))*wx*dt
    A[0, 2] = -dt/np.sin(psi)
    A[1, 3] = dt
    A[2:, :] += dt*omegaJacobian(x, t)
    return A

def getReducedC(x):
    C = getC(x)[:2,:]
    return C

def getC(x):
    theta, psi, wx, wy = x

    C = np.zeros((3,4))
    C[0, 2] = 1
    C[1, 3] = 1
    C[2, 2] = -1 / tan(psi)
    return C


# def dynamicsStep(X, t):
#     '''X is an 11 vector as defined in the documentation of generateTrajectory'''
#     pass

def noiselessDynamicsStep(x, t):
    theta, psi, wx, wy = x
    v, dv, phi, dphi = specifiedMotion(t)

    dtheta = -dphi - wx/np.sin(psi)
    dpsi = wy
    
    theta_plus1 = dtheta*dt + theta
    psi_plus1 = dpsi*dt + psi

    # get change in angular velocity scalars based on the state
    dwx, dwy, _ = dw(x, t)
    wx_plus1 = dwx*dt + wx
    wy_plus1 = dwy*dt + wy

    x_plus1 = np.array([theta_plus1, psi_plus1, wx_plus1, wy_plus1])
    return x_plus1

# In other words, this is the "noiseless measurement step"
def h(x):
    theta, psi, wx, wy = x
    wz = -wx/tan(psi)
    w_measure = np.array([[wx],[wy],[wz]])
    return w_measure


def getAugmentedState(x, t):
    ''' X is an 8 vector as defined in the documentation of generateTrajectory
        Doesn't do the math for the wheelchair position'''
    X = np.zeros(8)
    X[:4] = x # dynamical state
    
    theta, psi, wx, wy = x
    v, dv, phi, dphi = specifiedMotion(t)

    X[4] = phi
    X[5] = dphi
    X[6] = lcm*np.sin(psi)*np.cos(theta) # relative x of torso ccm
    X[7] = lcm*np.sin(psi)*np.sin(theta) # relative y of torso ccm

    return X

def stepWheelchair(prevWh, t):
    whx, why = prevWh
    v, dv, phi, dphi = specifiedMotion(t)
    whx_plus1 = v*np.cos(phi)*dt + whx
    why_plus1 = v*np.sin(phi)*dt + why
    return np.array((whx_plus1, why_plus1))


# def getUpdatedAugmentedState(x, t):
    

def generateTrajectory(initialX, tInitial = 0, tFinal = np.pi/2, tStep= dt):
    '''States of interest:
        t       time (sec)
        [0] theta   torso yaw 
        [1] psi     torso pitch
        [2] wx      torso cx> angular velocity measure
        [3] wy      torso cy> angular velocity measure
        [4] phi     wheelchair heading angle
        [5] phi'    wheelchair heading angle rate
        [6] x       torso CoM x position from pivot
        [7] y       torso CoM y position from pivot
        [8] wh_x    wheelchair x position
        [9] wh_y    wheelchair y position
        [10]wz      torso cz> angular velocity measure (auxillary)

        X makes up the values other than t (time is the column)
    '''
    N = int((tFinal -tInitial)/tStep)+1
    T = np.linspace(tInitial, (N-1)*tStep, num=N) # time discretization
    T = np.append(T, tFinal) # Allows for non even step size by making it mostly even
    N += 1

    # Augmented/full state for plotting's sake
    X = np.zeros((11, T.shape[0]))
    
    X[:8, 0] = getAugmentedState(initialX, tInitial)
    X[8:10, 0] = np.zeros(2)
    X[10, 0] = -initialX[2] / tan(initialX[1]) # wz = -wx/tan(psi)

    for i in range(N-1): # make sure this is correct
        x = X[:4, i] # the state for dynamics step
        t = T[i]
        x_tplus1 = noiselessDynamicsStep(x, t) # get the next dynamic step
        
        tp1 = T[i+1]
        X[:8, i+1] = getAugmentedState(x_tplus1, tp1) # get the augmented state from dynamic state
        # could this be off by 1 on time ^^^
        
        prevWh = X[8:10, i] # previous wheelchar position
        X[8:10, i+1] = stepWheelchair(prevWh, t)
        
        # Solve for auxillary wz value
        wx = x[2]
        psi = x[1]
        wz = -wx/tan(psi)
        X[-1, i+1] = wz
        
    return T, X


# def generateTrajectory(initialX, tInitial = 0, tFinal = np.pi, tStep= dt):
#     '''States of interest:
#         t       time (sec)
#         [0] theta   torso yaw 
#         [1] psi     torso pitch
#         [2] wx      torso cx> angular velocity measure
#         [3] wy      torso cy> angular velocity measure
#         [4] phi     wheelchair heading angle
#         [5] phi'    wheelchair heading angle rate
#         [6] x       torso CoM x position from pivot
#         [7] y       torso CoM y position from pivot
#         [8] wh_x    wheelchair x position
#         [9] wh_y    wheelchair y position

#         X makes up the values other than t (time is the column)
#     '''
#     N = int((tFinal -tInitial)/tStep)+1
#     T = np.linspace(tInitial, (N-1)*tStep, num=N) # time discretization
#     T = np.append(T, tFinal) # Allows for non even step size by making it mostly even
#     N += 1

#     # Augmented/full state for plotting's sake
#     X = np.zeros((12, T.shape[0]))
    
#     X[:8, 0] = getAugmentedState(initialX, tInitial)
#     X[8:10, 0] = np.zeros(2)

#     for i in range(N-1): # make sure this is correct
#         x = X[:4, i] # the state for dynamics step
#         t = T[i]
#         x_tplus1 = noiselessDynamicsStep(x, t) # get the next dynamic step
        
#         tp1 = T[i+1]
#         X[:8, i+1] = getAugmentedState(x_tplus1, tp1) # get the augmented state from dynamic state
#         # could this be off by 1 on time ^^^
        
#         prevWh = X[8:10, i] # previous wheelchar position
#         X[8:10, i+1] = stepWheelchair(prevWh, t)

#         X[10:, i+1] = dw(x_tplus1, tp1)[:2]
        
#     return T, X
    

# I can just generate the full noiseless trajectory in MG and then add noise after, however
# the control law is not responding the to the noise

# Also consider comparing the noisy versus un-noisy behavior for the presentation tomorrow

def GenerateAngularVelocityMeasurements(X, R):
    ''' Generate the sensor readings based on the trajectory of the system'''
    N = X.shape[1]
    Y = np.zeros((3, N-1))
    for i in range(N-1): # 1 less measurement than the duration, start by measuring after the starting state
        noise = np.random.multivariate_normal(np.zeros(3), R)
        wx = X[2, i+1]
        wy = X[3, i+1]
        wz = X[10,i+1]

        w_measure = np.array([wx,wy,wz]) + noise
        Y[:, i] = w_measure
    return Y


def generate_EKF(T, X, Y, mu0, Sigma0, Q, R):
    
    N = X.shape[1]
    n= X.shape[0]
    SIGMA = np.zeros((n,n, N))
    Mu = np.zeros((n, N))

    # Init tensors
    SIGMA[:,:, 0] = Sigma0
    Mu[:, 0] = mu0
    for i in range(N-1):
        t = T[i]
        x = Mu[:,i]
        Sigma = SIGMA[:,:, i]
        A = getA(x, t)
        
        # Prediction step of from t-1 measurements and priors
        mu_t_tm1 = noiselessDynamicsStep(x, t) # Prediction mean
        # mu_t_tm1 = A@Mu[:, i-1] + B@U[:, i-1]
        Sigma_t_tm1 = A@Sigma@A.T + Q # Prediction Covariance

        # Kalman Gain Matrix
        C = getC(mu_t_tm1)
        K_t = Sigma_t_tm1@C.T@(np.linalg.inv(C@Sigma_t_tm1@C.T+R))

        # # Measurement Update Step
        y_t = Y[:, i]
        Mu[:, i+1] = mu_t_tm1 + K_t@(y_t - h(mu_t_tm1))
        SIGMA[:,:, i+1] = Sigma_t_tm1 - K_t@C@Sigma_t_tm1
    return Mu, SIGMA

    
    
theta = 20 * np.pi/180 # deg -> rad
psi = 80 * np.pi/180 # deg -> rad
dtheta = 0 * np.pi/180 # deg/sec -> rad/sec
dpsi = 0 * np.pi/180 # deg/sec -> rad/sec

v, dv, phi, dphi = specifiedMotion(0)

wx = -(dtheta + dphi)*np.sin(psi)
wy = dpsi

t, X = generateTrajectory(initialX=np.array((theta, psi, wx, wy)))

Q = 0.1*np.eye(4)
R = 0.0001*np.eye(3)
Y = GenerateAngularVelocityMeasurements(X, R)

mu0 = X[:4, 0]
Sigma0 = 0.1*np.eye(4)
Mu, SIGMA = generate_EKF(t, X[:4, :], Y, mu0, Sigma0, Q, R)

# equivalent figure 8, phi(t), 
# different x(t), theta(t), psi(t)
# psi goes below zero (underdamped for MG, but doesn't reach zero for python)
# consider plotting out the wx', wy' values over time
whx = X[8,:]
why = X[9,:]
x = X[6,:]
phi = X[4, :]
psi = X[1, :]
theta = X[0, :]
dwx = X[10,:]
# print(np.vstack((t, dwx)).T)

# plt.plot(whx, why)
# plt.plot(t, x)
# # plt.plot(t, phi)
# # plt.plot(t, theta)
# # plt.plot(t, psi)
# # print(np.vstack((t, X[4,:])).T)
# plt.ylim([-0.05, 0.4])
# # plt.axis("equal")
# plt.show()

# Save this for making the plots
plt.plot(t, x)
plt.ylim([-0.05, 0.4])
plt.show()

'''
        [0] theta   torso yaw 
        [1] psi     torso pitch
        [2] wx      torso cx> angular velocity measure
        [3] wy      torso cy> angular velocity measure
        [4] phi     wheelchair heading angle
        [5] phi'    wheelchair heading angle rate
        [6] x       torso CoM x position from pivot
        [7] y       torso CoM y position from pivot
        [8] wh_x    wheelchair x position
        [9] wh_y    wheelchair y position
    '''
# if __name__ == "main":
#     theta = 20 * np.pi/180 # deg -> rad
#     psi = 80 * np.pi/180 # deg -> rad
#     wx = 0 * np.pi/180 # deg/sec -> rad/sec
#     wy = 0 * np.pi/180 # deg/sec -> rad/sec
    
#     t, X = generateTrajectory(initialX=np.array((theta, psi, wx, wy)))

#     whx = X[8,:]
#     why = X[9,:]
#     plt.plot(whx, why)
#     plt.show()