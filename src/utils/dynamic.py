import numba
import numpy as np


@numba.jit("double(double)", nopython=True)
def Uocv(x):
    '''
    Function approximation of Uocv

    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0].  Uses Horner's Method.

    The coefficient were obtained by fitting empirical measurements.
    '''
    result = 0
    # Ns = 46, Np = 3
    for coeff in np.array([ 122.64757413, -385.70243081,  465.42626555, -239.01264954,70.06917868,  156.71967607]):
        result = x * result + coeff
    return result


@numba.jit("double(double)", nopython=True)
def Rd(x):
    '''
    Function approximation of Rd. Uses Horner's Method.
    '''
    result = 0
    # Ns = 46, Np = 3
    for coeff in np.array([-0.07360154, 0.25351614, -0.34503675, 0.22146337, -0.06354403, 0.02619238]):
        result = x * result + coeff
    return result


@numba.jit("double(double)", nopython=True)
def Rc(x):
    '''
    Function approximation of Rc. Uses Horner's Method.
    '''
    result = 0
    # Ns = 46, Np = 3
    for coeff in np.array([-0.09039246,  0.27020625, -0.31658673,  0.18057255, -0.04953949,  0.02487341]):
        result = x * result + coeff
    return result


@numba.jit("double(double)", nopython=True)
def Pcmax(x):
    '''
    Function approximation of Pcmax. Uses Horner's Method.
    '''
    result = 0
    # Ns = 46, Np = 3
    for coeff in np.array([ 19.70484435, -62.18360399,  73.60498321, -36.97765868, 10.46088535,  24.00144186]):
        result = x * result + coeff
    return result


@numba.jit("double(double)", nopython=True)
def Pdmax(x):
    '''
    Function approximation of Pdmax. Uses Horner's Method.
    '''
    result = 0
    # Ns = 46, Np = 3
    for coeff in np.array([  44.16476606, -143.18070983,  176.21286896,  -94.66219997, 27.3193923 ,   43.72946199]):
        result = x * result + coeff
    return result



@numba.jit(numba.typeof((1.0,1.0))(numba.double, numba.double, numba.int16, numba.double, numba.double, numba.double),nopython=True)
def dynamic(x, u, Qnom, Pconso, Ppv, rho_d):
    '''
    Return the dynamic
    '''
    Pcbat = min(Pcmax(x) * 1000, u * (u > 0))
    Pdbat = min(Pdmax(x) * 1000, -u * (u < 0))

    if u >= 0:
        Ibat = (-Uocv(x) + np.sqrt(Uocv(x) ** 2 + 4 * Rc(x) * Pcbat)) / (2 * Rc(x))
    else:
        Ibat = (-Uocv(x) + np.sqrt(Uocv(x) ** 2 - 4 * Rd(x) * Pdbat)) / (2 * Rd(x))

    dx = Ibat / (Qnom * 60)  # Qnom [As] is multiplied by 60 to obtain dx [-/minute]

    # Pconso and Ppv are expressed in kW, Pcbat and Pdbat in W -> we uniformize the units in kW  
    P_meter = Pconso - Ppv + Pcbat / 1000 - rho_d * Pdbat / 1000

    return dx, P_meter


@numba.jit(numba.typeof((1.0,1.0))(numba.double, numba.double, numba.double, numba.int16, numba.double, numba.double, numba.double),nopython=True)
def RK4(x, h, u, Qnom, Pconso, Ppv, rho_d):
    '''
    RK4 integration method
    '''
    k1, Pmeter1 = dynamic(x, u, Qnom, Pconso, Ppv, rho_d)
    k2, Pmeter2 = dynamic(x + h / 2 * k1, u, Qnom, Pconso, Ppv, rho_d)
    k3, Pmeter3 = dynamic(x + h / 2 * k2, u, Qnom, Pconso, Ppv, rho_d)
    k4, Pmeter4 = dynamic(x + h * k3, u, Qnom, Pconso, Ppv, rho_d)
    new_x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We divide P_meter by 60 to obtain a result in kWh
    P_meter = (Pmeter1 + 2*Pmeter2 + 2*Pmeter3 + Pmeter4) / 6 * (h / 60)
    
    return new_x, P_meter


# @numba.jit()
def env_step(x, u, dt, purchase_price, selling_price, Qnom, rho_d, Pconso, Ppv):
    '''
    Return the new state, the incurred cost, and the P_meter value as a dict.
    detailed_P_meter contains the intermediate values of P_meter measured over smaller time intervals.
    Indeed, when the battery cannot be charged (or discharged) anymore,
    the sign of P_meter may change in the middle of a time step.
    Depedning on the sign, electricity is purchased or sold.
    '''

    next_x, f, P_meter, detailed_P_meter = virtual_step(x, u, dt, purchase_price, selling_price, Qnom, rho_d, Pconso, Ppv)
    return next_x, f, {'P_meter': P_meter, 'detailed_P_meter': detailed_P_meter}


@numba.jit(numba.typeof((1.0,1.0,1.0, np.zeros(100)))(numba.double, numba.double, numba.int16, numba.double, numba.double, numba.int16, numba.double, numba.double, numba.double),nopython=True)
def virtual_step(x, u, dt, purchase_price, selling_price, Qnom, rho_d, Pconso=0.0, Ppv=0.0):
        """
        Perform one step based on the input u, without updating the system.
        We obtain the associated transition.
        """
        h = dt/100
        tmp_f = 0
        total_P_meter = 0

        tmp_x, P_meter = RK4(x, dt, u, Qnom, Pconso, Ppv, rho_d)
        detailed_P_meter = np.zeros(100)

        # If the new SOC is outside of [0, 1]
        if tmp_x < 0 or tmp_x > 1:
            # We compute the integration on smaller intervals
            for t in range(100):
                tmp_x, P_meter = RK4(x, h, u, Qnom, Pconso, Ppv, rho_d)
                # If the new SOC leaves [0, 1], we apply the action 0: we don't do anything
                if tmp_x < 0 or tmp_x > 1:
                    h = dt - h * t
                    tmp_x, P_meter = RK4(x, h, 0, Qnom, Pconso, Ppv, rho_d)
                    f_purchase = purchase_price * max(P_meter, 0)
                    f_sell =  selling_price * min(P_meter, 0)
                    tmp_f += (f_purchase + f_sell) 
                    total_P_meter += P_meter
                    detailed_P_meter[t:] = P_meter / h
                    break
                # If the new SOC is in [0, 1], we update the temporary state and continue the integration
                else:
                    x = tmp_x
                    f_purchase = purchase_price * max(P_meter, 0)
                    f_sell =  selling_price * min(P_meter, 0)
                    tmp_f += (f_purchase + f_sell) 
                    total_P_meter += P_meter
                    detailed_P_meter[t] = P_meter
        # No issue with the next SOC: we update the temporary state and continue the integration
        else:
            f_purchase = purchase_price * max(P_meter, 0)
            f_sell =  selling_price * min(P_meter, 0)
            tmp_f += (f_purchase + f_sell) 
            total_P_meter += P_meter
            detailed_P_meter[:] = P_meter/100
        
        # # We round the SOC so that it stays on the predetermined grid of 1001 states.
        # next_x = np.round(tmp_x, 3)
        next_x = tmp_x
        f = tmp_f

        return next_x, f, total_P_meter, detailed_P_meter