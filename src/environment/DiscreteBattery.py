import numpy as np
import pandas as pd
import gymnasium as gym
import os
import typing
from scipy import interpolate as interp
from src.utils.dynamic import *

class DiscreteBattery(gym.Env):
    """
    Custom Environment that follows gym interface
    
    Environment describing a battery.
    Its state and the time are discret.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 discrete_action_space = False,
                 H: int=144,
                 data_file='Data_cryolite.csv', 
                 days=[0]) -> None:
        """
        Initialize the environment.

        Args:
            H (int): Horizon, i.e., the number of time steps.
            data_file (str): Path to the data file containing battery-related data.
        """
        super(DiscreteBattery, self).__init__()
        
        # Initialize horizon (number of time steps)
        self.H = H
        self.occuring_days = days
        self.day_index = 0
        
        self.observation_space = gym.spaces.Dict(
            {
                "time": gym.spaces.Discrete(H+1),
                "SOC": gym.spaces.Discrete(101),
                "delta": gym.spaces.Discrete(52)
            }
        )
        self.discrete_action_space = discrete_action_space
        if discrete_action_space:
            self.action_space = gym.spaces.Discrete(41)
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float32)

        # Load data from the specified file
        self.load_data(data_file)
        
        # Initialize state variables
        self.x, self.time, self.n_step = 0.0, 0.0, 0

        # Load functions related to state of charge
        self.Uocv = lambda x : Uocv(x)
        self.Rd = lambda x : Rd(x)
        self.Rc = lambda x : Rc(x)
        self.Pcmax = lambda x : Pcmax(x)
        self.Pdmax = lambda x : Pdmax(x)


    def load_data(self, data_file):
        """
        Load data from a CSV file and preprocess it.

        Args:
            data_file (str): Path to the data file.
        """
        
        # Load data from the CSV file
        data = pd.read_csv(os.path.join(os.getcwd(), 'src', 'environment', data_file), sep=';')
        
        # Select production and consumption data for working days only
        self.full_Ppv = data[data['travail']==1]['Production'].values
        self.full_Pconso = data[data['travail']==1]['Consommation'].values
        
        # Set the length of a time step
        self.dt = 10
        
        # Select production and consumption of the first day in the dataset
        day = self.occuring_days[self.day_index]
        self.Ppv = self.full_Ppv[day * self.H : (day+1) * self.H]
        self.Pconso = self.full_Pconso[day * self.H : (day+1) * self.H]

        # Create a time index for the data
        time_index = self.dt * np.arange(0, self.H)
        
        # Load other parameters from a separate file
        data_parameters = pd.read_csv(os.path.join(os.getcwd(), 'src', 'environment', 'params_prod_conso.csv'), sep=';')
        parameters = data_parameters.iloc[0, 0:2].to_dict()
        prices = data_parameters.iloc[0:24, 2:5]

        # Extract and store relevant parameters
        self.rho_d, self.Qnom = map(float, parameters.values())

        time_price = np.fromiter(prices['hour'].values, dtype=float)
        selling_prices = prices['selling_price'].values
        purchase_prices = prices['purchase_price'].values
   
        # Interpolate price data to obtain values for each time step
        self.purchase_prices = interp.interp1d(time_price, purchase_prices, kind='linear',
                                               fill_value='extrapolate')(time_index // 60)

        self.selling_prices = interp.interp1d(time_price, selling_prices, kind='linear',
                                              fill_value='extrapolate')(time_index // 60)
        
        # Find the maximum purchase price
        self.max_purchase_price = self.purchase_prices.max()


    def step(self, u: float):
        """
        Perform one step in the environment.

        Args:
            u (float): Action taken by the agent.

        Returns:
            Tuple: New state, cost, done flag, and additional info.
        """

        if self.discrete_action_space:
            u = (u - 20) * 1000
        else:
            u = u.item() * 20000
        # Perform one step and get the result
        new_x, cost, info = env_step(self.x, u, self.dt, self.purchase_prices[self.n_step], self.selling_prices[self.n_step], self.Qnom, self.rho_d, self.Pconso[self.n_step], self.Ppv[self.n_step])
        
        # Update environment variables
        self.n_step += 1
        info['time'] = self.n_step * self.dt
        done = self.n_step >= self.H
        self.x = new_x
        self.time = info['time']
        obs = {"time": self.n_step,
               "SOC": int(self.x*100),
               "delta": 0 if done else int((self.Pconso[self.n_step] - self.Ppv[self.n_step] + 50) / 2 )} 
        reward = -cost
        terminated = done
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


    def reset(self, init_x=None, init_t=None, change=True) -> typing.Tuple[float, int, int]:
        """
        Reset the environment to its initial state.

        Args:
            init_x (float): Initial state of charge (optional).
            init_t (float): Initial time (optional).

        Returns:
            np.array: Initial observation.
        """
        if change:
            if self.day_index == len(self.occuring_days) - 1:
                self.day_index = 0
            else:
                self.day_index += 1
            day = self.occuring_days[self.day_index]
            self.Ppv = self.full_Ppv[day * self.H : (day+1) * self.H]
            self.Pconso = self.full_Pconso[day * self.H : (day+1) * self.H] 
        
        if init_x is not None:
            self.x = init_x
        else:
            self.x = 0.0

        if init_t is not None:
            self.time = init_t
            self.n_step = int(init_t // self.dt)
        else:
            self.time = 0.0
            self.n_step = 0

        obs = {"time": self.n_step,
               "SOC": int(self.x*100),
               "delta": int((self.Pconso[self.n_step] - self.Ppv[self.n_step] + 50) / 2 )}
        info = {"info": None}
        return (obs, info)


    def render(self, mode='human', close=False) -> None:
        """
        Print the current state of the environment.

        Args:
            mode (str): Rendering mode (not used).
            close (bool): Flag indicating whether to close the rendering (not used).
        """
        print(self.x, self.time)

if __name__ == "__main__":
    """
    We check that the environment is compatible with gym
    """
    from stable_baselines3.common.env_checker import check_env

    env = DiscreteBattery()
    # Check compatibility with Gym and output any warnings
    check_env(env)
