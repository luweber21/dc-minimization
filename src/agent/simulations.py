import numpy as np
from src.utils.dynamic import *

    

# @numba.jit(forceobj=True)
def simulation_basicAgent(env, table, agent, x0=0.5, policy=None):
    '''
    Simulate an episode using an agent whose state is (SOC, t).
    The policy is given by table, containing the Q-values
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty and the measured Pmeter
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices

    x = x0
    res = []

    for step in range(agent.H-1): # loop over control intervals   
        X = int(x * (agent.nb_states-1))
        if policy is None:
            A = agent.argmin(table[step, X, :])
        else:
            A = policy[step, X]
        u = (2 * A / (agent.nb_actions-1) - 1)
        if u > 0:
            u *= agent.a_max
        else:
            u *= -agent.a_min
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])
        res.append((x, u, f, 0, info["P_meter"]))
        x = next_x
            
    res.append((x, 0, 0, 0, 0))        
    res = np.array(res)
    
    return res[:, 2].sum(), res, False


# @numba.jit(forceobj=True)
def simulation_signAgent(env, table, agent, x0=0.5, policy=None):
    '''
    Simulate an episode using an agent whose state is (SOC, t, sign(Pcons-Ppv)).
    The policy is given by table, containing the Q-values
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter,
    and the number of times a visited state has not been seen during the training
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    issues = np.zeros(agent.H)

    x = x0
    res = []

    for step in range(agent.H-1): # loop over control intervals   
        X = int(x * (agent.nb_states-1))
        d = env.Pconso[step] - env.Ppv[step]
        # element of state cooresponding to the sign of Pcons-Ppv
        D = 2 * (d > 0) + 1 * (d==0)
        if policy is None:
            A = agent.argmin(table[step, D, X, :])
        else:
            A = policy[step, D, X]
        # The real action is something between -20000 and 20000
        u = (2 * A / (agent.nb_actions-1) - 1)
        if u > 0:
            u *= agent.a_max
        else:
            u *= -agent.a_min
        if table[step, D, X, :].sum() == 0:
            issues[step] = 1
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])
        
        res.append((x, u, f, 0, info["P_meter"]))

        x = next_x
            
    res.append((x, 0, 0, 0, 0))
        
    res = np.array(res)
    
    return res[:, 2].sum(), res, issues


# @numba.jit(forceobj=True)
def simulation_preciseAgent(env, table, agent, x0=0.5, policy=None):
    '''
    Simulate an episode using an agent whose state is (SOC, t, sign(Pcons-Ppv)).
    The policy is given by table, containing the Q-values
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter,
    and the number of times a visited state has not been seen during the training
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    # issues = np.zeros(agent.H)

    x = x0
    res = []

    for step in range(agent.H-1): # loop over control intervals   
        X = int(x * (agent.nb_states-1))
        d = env.Pconso[step] - env.Ppv[step]
        # element of state cooresponding to the discretization of Pcons-Ppv
        D = np.clip(d, agent.delta_min, agent.delta_max)
        D = int((D - agent.delta_min)/(agent.delta_diff))
        if policy is None:
            A = agent.argmin(table[step, D, X, :])
        else:
            A = policy[step, D, X]
        u = (2 * A / (agent.nb_actions-1) - 1)
        if u > 0:
            u *= agent.a_max
        else:
            u *= -agent.a_min
        # if table[step, D, X, :].sum() == 0:
        #     issues[step] = 1

        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])
        res.append((x, u, f, 0, info["P_meter"] ))
        x = next_x

    res.append((x, 0, 0, 0, 0))
    res = np.array(res)
    
    return res[:, 2].sum(), res


# @numba.jit(forceobj=True)
def simulation_idleAgent(env, nb_times, x0=0.5, peak=0, crit="month"):
    '''
    Simulate an episode using an idle agent.
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    issue = False
    if crit == "month":
        coeff = 10
    else:
        coeff = 10/20

    x = x0
    res = []

    for step in range(nb_times-1): # loop over control intervals   
        u = 0
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])
        peak = max(peak, np.round(6 * info["P_meter"]).astype(int))

        res.append((x, u, f, 0, info["P_meter"] ))

        x = next_x

    res.append((x, 0, 0, coeff * peak, 0))      
    res = np.array(res)
    
    return res[:, 2].sum(), res, issue



# @numba.jit(forceobj=True)
def simulation_heuristic(env, nb_times, x0=0.5):
    '''
    Simulate an episode using an agent applying a simple heuristic.
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    issue = False

    x = x0
    res = []

    for step in range(nb_times-1): # loop over control intervals  
        delta = Ppv[step] - Pconso[step] 
    
        if env.constraint and step >= nb_times - 7:
            if x < env.xT:
                # u is computed by inverting the dynamic of the linearized system
                # We need to charge
                u = ((120 * Qnom * Rc(x) * (env.xT-x)/dt + Uocv(x))**2 - Uocv(x)**2) / (4* Rc(x))
            else:
                # We want to discharge
                u = -((120 * Qnom * Rd(x) * (env.xT-x)/dt + Uocv(x))**2 - Uocv(x)**2) / (4* Rd(x))
        else:
            if delta > 0:
                # We want to store the excessive energy
                u = min(20000, delta * 1000)
            else:
                # We use the stored energy
                u = max(-20000, delta * 1000)
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])
        
        res.append((x, u, f, 0, info["P_meter"] ))

        x = next_x
    res.append((x, 0, 0, 0, 0))    
    res = np.array(res)
    
    return res[:, 2].sum(), res, issue


def get_u_for_advanced_heuristic(x, delta, step, env, nb_times, purchase_prices, peak=None):
    '''
    Return the action used by a more advanced heuristic.
    '''
    
    magnitude = 20000
    steps_to_empty = 7

    if (step >= nb_times - steps_to_empty):
        # Discharge
        u = max(((120 * env.Qnom * Rd(x) * (0-x)/env.dt + Uocv(x))**2 - Uocv(x)**2) / (4* Rd(x)), -magnitude)
    else:
        if delta > 0:
            # We charge as much as possible
            u = min(magnitude, delta * 1000)
            if peak is not None:
                u = np.clip(u, a_min=-np.inf, a_max=min((peak - max(delta, 0)) * 1000, magnitude))
        elif purchase_prices[step] > 0.14:
            # Electricity is expensive: we don't want to buy, so we discharge the battery
            u = max(delta * 1000, -magnitude)
        else:
            # Do nothing
            u = 0
    return u




# @numba.jit(forceobj=True)
def simulation_advanced_heuristic(env, nb_times, x0=0.5, peak=0, crit="month"):
    '''
    Simulate an episode using an agent applying a more advanced heuristic.
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    issue = False
    if crit == "month":
        coeff = 10
    else:
        coeff = 10/20

    x = x0
    res = []

    for step in range(nb_times-1): # loop over control intervals  
        delta = Ppv[step] - Pconso[step] 
        u = get_u_for_advanced_heuristic(x, delta, step, env, nb_times, purchase_prices)
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])
        peak = max(peak, np.round(6 * info["P_meter"]).astype(int))
        res.append((x, u, f, 0, info["P_meter"] ))

        x = next_x
            

    res.append((x, 0, 0, coeff*peak, 0))        
    res = np.array(res)
    
    return res[:, 2].sum(), res, issue



# @numba.jit(forceobj=True)
def simulation_preciseAgent_with_heuristic(env, table, agent, verbose=False, x0=0.5, crit="month", policy=None):
    '''
    Simulate an episode using an agent whose state is (SOC, t, sign(Pcons-Ppv)).
    The policy is given by table, containing the Q-values
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter,
    and the number of times a visited state has not been seen during the training

    If a state has not been seen during the training, the advanced heuristic is used to determine the action to perform
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    # issues = np.zeros(agent.H)
    # T = env.H
    if crit == "month":
        coeff = 10
    else:
        coeff = 10/20

    x = x0
    res = []
    # à enlever !!!!!!!!!!!!!!!!!!!!
    peak=0

    for step in range(agent.H-1): # loop over control intervals   
        X = int(x * (agent.nb_states-1))
        d = env.Pconso[step] - env.Ppv[step]
        D = np.clip(d, agent.delta_min, agent.delta_max)
        D = int((D - agent.delta_min)/(agent.delta_diff))
        if policy is None:
            A = agent.argmin(table[step, D, X, :])
        else:
            A = policy[step, D, X]
        u = (2 * A / (agent.nb_actions-1) - 1)
        if u > 0:
            u *= agent.a_max
        else:
            u *= -agent.a_min
        if table[step, D, X, :].sum() == 0:
            # issues[step] = 1
            delta = Ppv[step] - Pconso[step] 
            u = get_u_for_advanced_heuristic(x, delta, step, env, agent.H, purchase_prices)

        if verbose:
            print('X', X, 'x', x, 'A', A, 'u', u)
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])
        # print(step, D, peak, X, A)
        # print(table[step, D, X, :])

        res.append((x, u, f, 0, info["P_meter"] ))

        x = next_x
        peak = max(peak, np.round(6 * info["P_meter"]).astype(int))
            
    # if env.constraint:
    #     pen, additional_Pmeter = penalty(x, env.Qnom, env.xT, env.dt, env.max_purchase_price, env.rho_d)
    #     pen += peak * coeff
    #     res.append((x, 0, 0, pen, additional_Pmeter ))
    # else:
    pen = peak * coeff
    res.append((x, 0, 0, pen, 0))
    
    res = np.array(res)

    return res[:, 2].sum(), res, peak


def simulation_preciseAgent_with_heuristic_and_peak(env, table, agent, verbose=False, x0=0.5, peak=0, crit="month", policy=None):
    '''
    Simulate an episode using an agent whose state is (SOC, t, sign(Pcons-Ppv)).
    The policy is given by table, containing the Q-values
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter,
    and the number of times a visited state has not been seen during the training

    If a state has not been seen during the training, the advanced heuristic is used to determine the action to perform
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    issues = np.zeros(agent.H)
    if crit == "month":
        coeff = 10
    else:
        coeff = 10/20
    x = x0
    res = []

    for step in range(agent.H-1): # loop over control intervals   
        X = int(x * (agent.nb_states-1))
        d = env.Pconso[step] - env.Ppv[step]
        D = np.clip(d, agent.delta_min, agent.delta_max)
        D = int((D - agent.delta_min)/(agent.delta_diff))

        if policy is None:
            # print('t:', step, 'D:', D, 'peak:', peak, 'X:', X)
            # print(table[step, D, peak, X, :])
            A = agent.argmin(table[step, D, peak, X, :])
            # print('A:', A)
        else:
            A = policy[step, D, peak, X]

        u = (2 * A / (agent.nb_actions-1) - 1)
        if u > 0:
            u *= agent.a_max
        else:
            u *= -agent.a_min
        if table[step, D, peak, X, :].sum() == 0:
            issues[step] = 1
            delta = Ppv[step] - Pconso[step] 
            u = get_u_for_advanced_heuristic(x, delta, step, env, agent.H, purchase_prices,peak=peak)

        if verbose:
            print('X', X, 'x', x, 'A', A, 'u', u)
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])

        peak = max(peak, np.round(6 * info["P_meter"]).astype(int))
        
        res.append((x, u, f, 0, info["P_meter"] ))

        x = next_x
            

    
    pen = peak * coeff
    res.append((x, 0, 0, pen, 0))
    
    res = np.array(res)

    return res[:, 2].sum(), res, peak



def simulation_preciseAgent_with_peak(env, table, agent, verbose=False, x0=0.5, peak=0, crit="month"):
    '''
    Simulate an episode using an agent whose state is (SOC, t, sign(Pcons-Ppv)).
    The policy is given by table, containing the Q-values
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter,
    and the number of times a visited state has not been seen during the training

    If a state has not been seen during the training, the advanced heuristic is used to determine the action to perform
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    issues = np.zeros(agent.H)
    if crit == "month":
        coeff = 10
    else:
        coeff = 10/20

    x = x0
    res = []

    for step in range(agent.H-1): # loop over control intervals   
        X = int(x * (agent.nb_states-1))
        d = env.Pconso[step] - env.Ppv[step]
        D = np.clip(d, agent.delta_min, agent.delta_max)
        D = int((D - agent.delta_min)/(agent.delta_diff))
        A = agent.argmin(table[step, D, peak, X, :])

        u = (2 * A / (agent.nb_actions-1) - 1)
        if u > 0:
            u *= agent.a_max
        else:
            u *= -agent.a_min

        if table[step, D, peak, X, :].sum() == 0:
            issues[step] = 1
            u = 0

        if verbose:
            print('X', X, 'x', x, 'A', A, 'u', u)
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])

        peak = max(peak, np.round(6 * info["P_meter"]).astype(int))
        
        res.append((x, u, f, 0, info["P_meter"] ))

        x = next_x
            

    pen = peak * coeff
    res.append((x, 0, 0, pen, 0))
    
    res = np.array(res)

    return res[:, 2].sum(), res, peak

def simulation_preciseAgent_with_recoupling(env, table, agent, verbose=False, x0=0.5, peak=0, crit="month", policy=None):
    '''
    Simulate an episode using an agent whose state is (SOC, t, sign(Pcons-Ppv)).
    The policy is given by table, containing the Q-values
    Returns the total cost, a table containg the SOC, costs, actions, eventual penalty, and the measured Pmeter,
    and the number of times a visited state has not been seen during the training

    If a state has not been seen during the training, the advanced heuristic is used to determine the action to perform
    '''
    # Data
    Pconso = env.Pconso
    Ppv = env.Ppv
    dt = env.dt
    rho_d = env.rho_d
    Qnom = env.Qnom
    purchase_prices = env.purchase_prices
    selling_prices = env.selling_prices
    issues = np.zeros(agent.H)
    if crit == "month":
        coeff = 10
    else:
        coeff = 10/20
    x = x0
    res = []

    for step in range(agent.H-1): # loop over control intervals   
        X = int(x * (agent.nb_states-1))
        d = env.Pconso[step] - env.Ppv[step]
        D = np.clip(d, agent.delta_min, agent.delta_max)
        D = int((D - agent.delta_min)/(agent.delta_diff))
        if policy is None:
            A = agent.argmin(table[step, D, peak, X, :])
        else:
            A = policy[step, D, peak, X]

        u = (2 * A / (agent.nb_actions-1) - 1)
        if u > 0:
            u *= agent.a_max
        else:
            u *= -agent.a_min
        if table[step, D, peak, X, :].sum() == 0 or step >= agent.H-16:
            issues[step] = 1
            delta = Ppv[step] - Pconso[step] 
            u = get_u_for_advanced_heuristic_and_recoupling(x, delta, step, env, agent.H, purchase_prices,peak=peak)

        if verbose:
            print('X', X, 'x', x, 'A', A, 'u', u)
            
        next_x, f, info = env_step(x, u, dt, purchase_prices[step], selling_prices[step], Qnom, rho_d, Pconso[step], Ppv[step])

        peak = max(peak, np.round(6 * info["P_meter"]).astype(int))
        
        res.append((x, u, f, 0, info["P_meter"] ))

        x = next_x
            

    
    pen = peak * coeff
    res.append((x, 0, 0, pen, 0))
    
    res = np.array(res)

    return res[:, 2].sum(), res, peak

def get_u_for_advanced_heuristic_and_recoupling(x, delta, step, env, nb_times, purchase_prices, peak=None):
    '''
    Return the action used by a more advanced heuristic.
    '''
    
    magnitude = 20000
    steps_to_charge = 7

    if (step >= nb_times - steps_to_charge):
        # Charge
        t = nb_times - step - 1
        u = np.min([((120 * env.Qnom * Rc(x) * (0.3-x)/(t*env.dt) + Uocv(x))**2 - Uocv(x)**2) / (4* Rc(x)), magnitude, min((peak - max(delta, 0)) * 1000, magnitude)])
    else:
        if delta > 0:
            # We charge as much as possible
            u = min(magnitude, delta * 1000)
            if peak is not None:
                u = np.clip(u, a_min=-np.inf, a_max=min((peak - max(delta, 0)) * 1000, magnitude))
        elif purchase_prices[step] > 0.14:
            # Electricity is expensive: we don't want to buy, so we discharge the battery
            u = max(delta * 1000, -magnitude)
        else:
            # Do nothing
            u = 0
    return u