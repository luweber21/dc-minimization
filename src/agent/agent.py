from abc import ABC, abstractmethod
from pathlib import Path
import glob
import os
import numpy as np
from src.utils.dynamic import *
from src.explorer.explorers import S1Explorer, S2Explorer, BigEnvExplorer
from tqdm import tqdm
from numba import njit


class AbstractRLAgent(ABC):
    '''
    Abstract class implementing all the common methods and attributes
    '''
    def __init__(self,
                 lr = lambda n : 0.5 * n,
                 H: int = 1,
                 nb_actions: int = 21,
                 nb_states : int = 101,
                 a_min = -20000,
                 a_max = 20000,
                 save_agent=False,
                 dir_name=None,
                 optim_type='price'
                 ):
        self.H = H
        # learning rate as a lambda function
        self.lr = lr
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.a_min = a_min
        self.a_max = a_max
        self.save_agent = save_agent
        self.count = None
        # optimisation type: price or peak shaving
        
        # If the option is true, then a new directory is created
        if self.save_agent:
            if dir_name is None:
                path = Path(os.getcwd())
                experiments_directory = os.path.join(path, "data",  "experiments")
                n = len(glob.glob(os.path.join(experiments_directory, "*")))

                # Directory
                dir_name = f"exp_{n + 1:03d}"
                path = os.path.join(experiments_directory, dir_name)
                try:
                    os.makedirs(path, exist_ok=True)
                    print('Directory has been created:', path)
                except OSError:
                    print('Directory already exists')
                self.path = path
            else:
                path = Path(os.getcwd())
                experiments_directory = os.path.join(path, "data", "experiments")

                # Directory
                exp_dir = os.path.join(experiments_directory, dir_name)
                n = len(glob.glob(os.path.join(exp_dir, "*")))
                dir_name = f"agent_{n + 1:03d}"
                path = os.path.join(exp_dir, dir_name)
                try:
                    os.makedirs(path, exist_ok=True)
                    print('Directory has been created:', exp_dir)
                except OSError:
                    print('Directory already exists')
                self.path = path

        if self.save_agent:
            import inspect
            from src.utils.dynamic import Uocv, Pcmax, Pdmax, Rc, Rd
            path = os.path.join(self.path, "dynamic")
            with open(path, 'a') as f:
                f.write(inspect.getsource(Uocv))
                f.write(inspect.getsource(Pcmax))
                f.write(inspect.getsource(Pdmax))
                f.write(inspect.getsource(Rc))
                f.write(inspect.getsource(Rd))


    def argmin(self, T, axis=0):
        '''
        If several values are equal, returns the closest argmin to the middle of the table.
        I.e., returns the closest action to "do nothing"

        Not used or at most once after the training.
        '''
        mask = T == T.min(axis=axis, keepdims=True)[0]
        value = np.zeros(T.shape)
        n = T.shape[-1]
        value[...,] = 1 + (n-1)/2 - np.abs((n-1)/2 - np.arange(n))  

        res = mask * value

        return np.argmax(res, axis=axis)
            

    @abstractmethod
    def collect_transitions(self,
                            env,
                            Ppvs, 
                            Pconsos,
                            print_tqdm=False):
        pass
    
    
    @abstractmethod
    def update_Q(self,
                 tmp_cost,
                 s,
                 sp):
        pass
    
        
    def train(self,
            env,
            nb_epochs: int = int(1e4),
            Ppvs = None,
            Pconsos = None,
            print_tqdm = False):

        '''
        Train the agent
        '''

        for e in tqdm(range(nb_epochs)) if print_tqdm else range(nb_epochs):
            S = self.update_Q(e)
            self.count[S] += 1

        if self.save_agent:
            self.save_logs()

    def save_logs(self):
        # Save logs
        np.save(os.path.join(self.path, 'q-table.npy'), self.q_table)


class RLAgent(AbstractRLAgent):
    '''
    class implementing the agent
    '''
    def __init__(self, *args, state_space='S1', delta_min=None, delta_max=None, delta_diff=None, nP=None, **kwargs):
        super().__init__(*args, **kwargs)
         # Parameters
        self.delta_diff = delta_diff
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.nP = nP
        self.state_space = state_space

        if state_space == 'S1':
            self.q_table = np.zeros((self.H, self.nb_states, self.nb_actions))
            self.Explorer = S1Explorer
        elif state_space == 'S2':
            self.q_table = np.zeros((self.H, 3, self.nb_states, self.nb_actions))
            self.Explorer = S2Explorer
        elif state_space == 'S3':
            self.q_table = np.zeros((self.H, int((self.delta_max-self.delta_min)/delta_diff) + 1, self.nb_states, self.nb_actions))
            self.Explorer = BigEnvExplorer
        else:
            self.q_table = np.zeros((self.H, int((self.delta_max-self.delta_min)/delta_diff) + 1, self.nP, self.nb_states, self.nb_actions), dtype=np.float32)
            self.Explorer = BigEnvExplorer
            self.price_rate = 0.5

        self.count = np.ones(self.q_table.shape)
        

    def collect_transitions(self,
                            env,
                            Ppvs, 
                            Pconsos):
        params={"env": env,
                "nSOC": self.nb_states,
                "nA": self.nb_actions,
                "a min": self.a_min,
                "a max": self.a_max,
                "H": self.H,
                "Ppvs": Ppvs,
                "Pconsos": Pconsos,
                "delta diff": self.delta_diff,
                "nP": self.nP,
                "delta min": self.delta_min,
                "delta max":  self.delta_max}
        self.explorer = self.Explorer(params)
        if self.save_agent:
            self.explorer.save_params(self.path)
    
    
    def update_Q(self, day):
        '''
        Update the Q-function by Dynamic Programming.
        '''

        costs = self.explorer.explore(day)
        
        if self.state_space != 'S4':
            S = tuple(self.explorer.tuples.T)
            Q = np.zeros((self.H, self.nb_states, self.nb_actions))
            Q[self.H - 1] = costs[self.H - 1]        
            for t in range(self.H-2, -1, -1):
                Q[t] = Q[t+1, self.explorer.new_SOCs].min(axis=2) + costs[t]

            self.q_table[S] += self.lr(self.count[S]) * (Q.reshape(-1) - self.q_table[S])
            
        else:
            # S = tuple(self.explorer.tuples.T)
            # Q = np.zeros((self.H, self.nP, self.nb_states, self.nb_actions), dtype=np.float32) 
            # Pmeters =  self.explorer.Pmeters.sum(axis=-1)
            # Pmeters = (Pmeters[np.newaxis, :, :]*6 + (( self.explorer.env.Pconso -  self.explorer.env.Ppv))[:, np.newaxis, np.newaxis])
            # Q = loop(Pmeters, Q, self.nb_states, self.nb_actions, self.nP, self.H, self.explorer.new_SOCs, self.price_rate, costs)
            
            Pmeters = self.explorer.Pmeters.sum(axis=-1)
            Pmeters = (Pmeters[np.newaxis, :, :]*6 + ((self.explorer.env.Pconso - self.explorer.env.Ppv))[:, np.newaxis, np.newaxis])
            S = tuple(self.explorer.tuples.T)
            Q = np.zeros((self.H, self.nP, self.nb_states, self.nb_actions), dtype=np.float32)        
            # states = np.array([[p, x, a] for p in range(self.nP) for x in range(self.nb_states) for a in range(self.nb_actions)])
            states = np.array([[x, a] for x in range(self.nb_states) for a in range(self.nb_actions)])
            
            V = np.zeros((self.nP, self.nb_states))
            for t in range(self.H-1, -1, -1):
                pmin, pmax = np.round(Pmeters[t]).astype(int).min(), np.round(Pmeters[t]).astype(int).max()  

                # For p <= pmin
                new_P = np.round(Pmeters[t]).astype(int).reshape(-1) 
                new_states = np.stack([new_P, self.explorer.new_SOCs.reshape(-1)], axis=1)
                for p in range(pmin+1):
                    Q[t, p][tuple(states.T)]=V[tuple(new_states.T)] + costs[t].reshape(-1) + (new_P - p) * self.price_rate 
                # For p >= pmax
                for p in range(pmax, self.nP):
                    new_P = np.full_like(new_P, p)
                    new_states = np.stack([new_P, self.explorer.new_SOCs.reshape(-1)], axis=1) 
                    Q[t, p][tuple(states.T)]=V[tuple(new_states.T)] + costs[t].reshape(-1) 
                for p in range(self.nP):
                    new_P = np.maximum(np.round(Pmeters[t]).astype(int).reshape(-1), p)
                    new_states = np.stack([new_P, self.explorer.new_SOCs.reshape(-1)], axis=1)
                    Q[t, p][tuple(states.T)]=V[tuple(new_states.T)] + costs[t].reshape(-1) + (new_P - p) * self.price_rate

                V = Q[t].min(axis=-1)

                # # Calculate Q-values for values between pmin and pmax using advanced indexing
                # new_P = np.maximum(np.round(Pmeters[t]).astype(int).reshape(-1)[np.newaxis, :], np.arange(0, self.nP)[:, np.newaxis])
                # new_states = np.column_stack((new_P.reshape(-1), np.tile(self.explorer.new_SOCs.reshape(-1), self.nP)))

               
                # # print(Q[t, :][tuple(states.T)].shape)
                # # print(V[tuple(new_states.T)].reshape(Q[t, :][tuple(states.T)].shape).shape)
                # # print(np.tile(costs[t].reshape(-1), self.nP).shape)
                # # print(((new_P - np.arange(0, self.nP)[:, np.newaxis]).reshape(-1) * self.price_rate).shape)

                # Q[t, :][tuple(states.T)] = V[tuple(new_states.T)].reshape(Q[t, :][tuple(states.T)].shape)+ np.tile(costs[t].reshape(-1), self.nP) + (new_P - np.arange(0, self.nP)[:, np.newaxis]).reshape(-1) * self.price_rate
                # # Update V values without for loop
                # V = np.min(Q[t], axis=-1)

            self.q_table[S] += self.lr(self.count[S]) * (Q.reshape(-1) - self.q_table[S])
        
        return S

# @njit
# def loop(Pmeters, Q, nb_states, nb_actions, nP, H, new_SOCs, price_rate, costs):
#     # states = np.array([[p, x, a] for p in range(self.nP) for x in range(self.nb_states) for a in range(self.nb_actions)])
#     states = np.array([[x, a] for x in range(nb_states) for a in range(nb_actions)])
    
#     V = np.zeros((nP, nb_states))
#     for t in range(H-1, -1, -1):
#         pmin = int(round(Pmeters[t].min()))
#         pmax = int(round(Pmeters[t].max()))

#         # For p <= pmin
#         new_P = np.round(Pmeters[t]).astype(int).reshape(-1) 
#         new_states = np.stack([new_P, new_SOCs.reshape(-1)], axis=1)
#         for p in range(pmin+1):
#             Q[t, p][tuple(states.T)]=V[tuple(new_states.T)] + costs[t].reshape(-1) + (new_P - p) * price_rate 
#         # For p >= pmax
#         for p in range(pmax, nP):
#             new_P = np.full_like(new_P, p)
#             new_states = np.stack([new_P, new_SOCs.reshape(-1)], axis=1) 
#             Q[t, p][tuple(states.T)]=V[tuple(new_states.T)] + costs[t].reshape(-1) 
#         for p in range(nP):
#             new_P = np.maximum(np.round(Pmeters[t]).astype(int).reshape(-1), p)
#             new_states = np.stack([new_P, new_SOCs.reshape(-1)], axis=1)
#             Q[t, p][tuple(states.T)]=V[tuple(new_states.T)] + costs[t].reshape(-1) + (new_P - p) * price_rate

#         V = Q[t].min(axis=-1) 
#     return Q


