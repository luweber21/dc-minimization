from abc import ABC, abstractmethod
import src.utils.dynamic as utils
import json
import os

class ExplorerInterface(ABC):
    '''
    Class implementing the interface of the explorers.
    Explorers are used to collect all the possible transitions for the discretized state and action spaces.
    '''
    
    def __init__(self, params):
        super().__init__()
        self.env = params["env"]
        self.params = params


    def is_done(self, T):
        # Check if episode is done based on a time condition
        return T * self.env.dt >= self.env.T


    def collect_transition(self, x, a, T):
        '''
        The transition corresponding to action a at time T with SOC x is collected for null Ppv and Pcons.
        The transition can be easily adapted to correspond to any value of Ppv and Pcons.
        '''
        # Collect a transition for the given state (x), action (a), and time (T)
        # It uses utility function step() from module utils.
        return  utils.env_step(x, a, self.env.dt, self.env.purchase_prices[T], self.env.selling_prices[T], self.env.Qnom, self.env.rho_d, 0.0, 0.0)


    def save_params(self, directory):
        # Prepare a dictionary to save the explorer's parameters
        dict_to_save = self.params.copy()
        dict_to_save.pop("env")
        dict_to_save.pop("Pconsos")
        dict_to_save.pop("Ppvs")

        dict_to_save["rho d"] = self.env.rho_d
        dict_to_save["Qnom"] = self.env.Qnom

        # Specify the path to save the parameters
        path = os.path.join(directory, "parameters")
        with open(path, 'w') as f:
            # Serialize the dictionary and save it to a JSON file
            f.write(json.dumps(dict_to_save)) 

    @abstractmethod
    def explore():
        # Declare the explore() method as abstract; concrete subclasses must implement it
        raise NotImplementedError

        

   