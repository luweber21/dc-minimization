from src.explorer.interface import ExplorerInterface
import numpy as np
import itertools


class S1Explorer(ExplorerInterface):
    '''
    class implementing the agent whose state is (SOC, t, Ppv-Pcons)
    '''
    def __init__(self, params):
        super().__init__(params)
        self.nSOC = params["nSOC"]
        self.nA = params["nA"]
        self.nT = params["H"] 
        self.Ppvs = params["Ppvs"]
        self.Pconsos = params["Pconsos"]
        self.a_min = params["a min"]
        self.a_max = params["a max"]
        self.initialization()


    def initialization(self):
        '''
        Collect all transitions for a virtual day with no production and no consumption.
        No need to explore all time steps, as only the production and consumption change.
        '''

        # Initialize a matrix to store new states
        new_SOCs = np.zeros((self.nSOC, self.nA), dtype=int)
        # Initialize a matrix to store power data
        Pmeters = np.zeros((self.nSOC, self.nA, 100), dtype=float)


        # Temporarily set Ppv and Pconso to zero for exploration
        tmp_Ppv, tmp_Pconso = self.env.Ppv, self.env.Pconso
        self.env.Ppv, self.env.Pconso = np.zeros(self.nT), np.zeros(self.nT)

        T=0
        state_loop = zip(np.linspace(0, 1, self.nSOC), np.arange(self.nSOC))
        action_list = np.concatenate((np.linspace(self.a_min, 0, self.nA // 2 + 1), np.linspace(0, self.a_max, self.nA // 2 + 1)[1:]))
        
        # Set the environment to the desired state
        for (x, X) in state_loop:
            self.env.reset(x, 0.0)
            # Generate action-value pairs
            action_loop = zip(action_list, np.arange(self.nA))

            for (a, A) in action_loop:
                # Perform action and collect transition information
                new_x, _, info = self.collect_transition(x, a, T)
                
                # Map new state to a discrete value
                new_X = int(new_x * (self.nSOC - 1))
                Pmeters[X, A] = info["detailed_P_meter"] # Store power data
                new_SOCs[X, A] = new_X # Store new state

        # Restore original Ppv and Pconso values
        self.env.Ppv, self.env.Pconso = tmp_Ppv, tmp_Pconso

        # Store the collected data
        self.new_SOCs = new_SOCs
        self.Pmeters = Pmeters

        # Create tuples to represent possible state-action-time combinations
        self.tuples = np.array([[0, X, A] for X, A in itertools.product(range(0, self.nSOC), range(0, self.nA))]) 


    def explore(self, day, t):
        '''
        Adapt the transitions to the given day.
        '''

        # Set solar panel power and consumption data for the day
        self.env.Ppv, self.env.Pconso = self.Ppvs[day], self.Pconsos[day]

        # Calculate the day's power balance (unit conversion)
        day_balance = (self.env.Pconso[t] - self.env.Ppv[t]) / 600 # / 6 for unit conversion (mean of kW over 10 minutes -> kWh), / 100 to correspond to detailed_P_meter
        Pmeters = self.Pmeters + day_balance
        costs = np.sum(Pmeters * ((Pmeters > 0) * (self.env.purchase_prices[t]-self.env.selling_prices[t]) + self.env.selling_prices[t]), axis=-1)
        self.tuples[:, 0] = t
        return costs


    def get_costs(self, Pmeters):
        f_purchase = self.purchase_price * max(Pmeters, 0) # Calculate purchase cost
        f_sell =  self.selling_price * min(Pmeters, 0) # Calculate selling cost
        return (f_purchase + f_sell).sum(axis=-1) 
        

class S2Explorer(ExplorerInterface):
    '''
    class implementing the agent whose state is (SOC, t, sign(Ppv-Pcons))
    '''
    def __init__(self, params):
        super().__init__(params)
        self.nSOC = params["nSOC"]
        self.nA = params["nA"]
        self.nT = params["H"] 
        self.Ppvs = params["Ppvs"]
        self.Pconsos = params["Pconsos"]
        self.a_min = params["a min"]
        self.a_max = params["a max"]
        self.initialization()


    def initialization(self):
        '''
        Collect all transitions for a virtual day with no production and no consumption.
        No need to explore all timesteps, as only the production and consumption change.
        '''
        new_SOCs = np.zeros((self.nSOC, self.nA), dtype=int)
        Pmeters = np.zeros((self.nSOC, self.nA, 100), dtype=float)


        # Update Ppv and Pconso to those of the specified day 'day'
        tmp_Ppv, tmp_Pconso = self.env.Ppv, self.env.Pconso
        self.env.Ppv, self.env.Pconso = np.zeros(self.nT), np.zeros(self.nT)

        T=0
        state_loop = zip(np.linspace(0, 1, self.nSOC), np.arange(self.nSOC))
        action_list= np.concatenate((np.linspace(self.a_min, 0, self.nA // 2 + 1), np.linspace(0, self.a_max, self.nA // 2 + 1)[1:]))
        for (x, X) in state_loop:
            self.env.reset(x, 0.0)
            action_loop = zip(action_list, np.arange(self.nA))
            # action_loop = zip(np.linspace(-40000, 40000, self.nA), np.arange(self.nA))

            for (a, A) in action_loop:
                # Perform action
                new_x, _, info = self.collect_transition(x, a, T)
                
                # new_x, cost, done, info = env.step(a)
                new_X = int(new_x * (self.nSOC - 1))
                Pmeters[X, A] = info["detailed_P_meter"]
                new_SOCs[X, A] = new_X

        self.env.Ppv, self.env.Pconso = tmp_Ppv, tmp_Pconso

        self.new_SOCs = new_SOCs
        self.Pmeters = Pmeters

        self.tuples = np.array([[0, 0, X, A] for X, A in itertools.product(range(0, self.nSOC), range(0, self.nA))]) 


    def explore(self, day, t):
        self.env.Ppv, self.env.Pconso = self.Ppvs[day], self.Pconsos[day]
        D = self.env.Pconso[t] - self.env.Ppv[t]

        day_balance = D / 600 # / 6 for unit conversion
        Pmeters = self.Pmeters + day_balance
        costs = np.sum(Pmeters * ((Pmeters > 0) * (self.env.purchase_prices-self.env.selling_prices)[t] + self.env.selling_prices[t]), axis=-1)
        D = np.sign(D) + 1
        self.tuples[:, 1] = D
        self.tuples[:, 0] = t
        return costs


    def get_costs(self, Pmeters):
        f_purchase = self.purchase_price * max(Pmeters, 0)
        f_sell =  self.selling_price * min(Pmeters, 0)
        return (f_purchase + f_sell).sum(axis=-1) 
        

    

class BigEnvExplorer(ExplorerInterface):
    '''
    class implementing the agent whose state is (SOC, t, Ppv-Pcons)
    '''
    def __init__(self, params):
        super().__init__(params)
        self.nSOC = params["nSOC"]
        self.nA = params["nA"]
        self.nT = params["H"]
        self.Ppvs = params["Ppvs"]
        self.Pconsos = params["Pconsos"]
        self.delta_diff = params["delta diff"]
        self.delta_min = params["delta min"]
        self.delta_max = params["delta max"]
        self.a_min = params["a min"]
        self.a_max = params["a max"]
        self.nP = params["nP"]
        self.peak = self.nP is not None
        
        self.initialization()


    def initialization(self):
        '''
        Collect all transitions for a virtual day with no production and no consumption.
        No need to explore all timesteps, as only the production and consumption change.
        '''
        new_SOCs = np.zeros((self.nSOC, self.nA), dtype=np.int16)
        Pmeters = np.zeros((self.nSOC, self.nA, 100), dtype=float)


        # Update Ppv and Pconso to those of the specified day 'day'
        tmp_Ppv, tmp_Pconso = self.env.Ppv, self.env.Pconso
        self.env.Ppv, self.env.Pconso = np.zeros(self.nT), np.zeros(self.nT)
        T=0
        state_loop = zip(np.linspace(0, 1, self.nSOC), np.arange(self.nSOC))
        action_list = np.concatenate((np.linspace(self.a_min, 0, self.nA // 2 + 1), np.linspace(0, self.a_max, self.nA // 2 + 1)[1:]))
        for (x, X) in state_loop:
            self.env.reset(x, 0.0)
            action_loop = zip(action_list, np.arange(self.nA))            

            for (a, A) in action_loop:
                # Perform action
                new_x, _, info = self.collect_transition(x, a, T)
                
                # new_x, cost, done, info = env.step(a)
                new_X = int(new_x * (self.nSOC - 1))
                Pmeters[X, A] = info["detailed_P_meter"]
                new_SOCs[X, A] = new_X

        self.env.Ppv, self.env.Pconso = tmp_Ppv, tmp_Pconso

        self.new_SOCs = new_SOCs
        self.Pmeters = Pmeters

        if self.peak:
            self.tuples = np.array([[0, 0, P, X, A] for P, X, A in itertools.product(range(0, self.nP), range(0, self.nSOC), range(0, self.nA))], dtype=np.int16) 
        else:
            self.tuples = np.array([[0, 0, X, A] for X, A in itertools.product(range(0, self.nSOC), range(0, self.nA))], dtype=np.int16) 


    def explore(self, day, t):
        self.env.Ppv, self.env.Pconso = self.Ppvs[day], self.Pconsos[day]

        D = self.env.Pconso[t] - self.env.Ppv[t]

        day_balance = D / 600 # / 6 for unit conversion
        Pmeters = self.Pmeters + day_balance
        costs = np.sum(Pmeters * ((Pmeters > 0) * (self.env.purchase_prices-self.env.selling_prices)[t] + self.env.selling_prices[t]), axis=-1)

        D = np.clip(D, self.delta_min, self.delta_max)
        D = (D - self.delta_min) // self.delta_diff
        self.tuples[:, 1] = D
        self.tuples[:, 0] = t
        return costs


    def get_costs(self, Pmeters):
        f_purchase = self.purchase_price * max(Pmeters, 0)
        f_sell =  self.selling_price * min(Pmeters, 0)
        return (f_purchase + f_sell).sum(axis=-1) 
        