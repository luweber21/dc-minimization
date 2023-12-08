import sys
import numpy as np
from src.agent.utils import *
from src.agent.agent import RLAgent
from time import time
import os
from src.agent.simulations import *

      
def run(dir_name, state_space, datafile='Data_cryolite.csv'):
    '''
    Train an agent over N_train episodes.
    Test it over N_test episodes.
    '''
    N_train = 300
    N_test = 100

    # We can choose to enforce the final state constraint or not
    env = get_env(datafile=datafile)

    # Call the agent class corresponding to the input
    nP = 93 if state_space == 'S4' else None
    Agent = RLAgent(lr = lambda n : 1/n,
                    H = env.H,
                    nb_actions=41,
                    nb_states=101,
                    a_min=-20000,
                    a_max=20000,
                    delta_diff = 2,
                    delta_min = -50,
                    delta_max = 52,
                    nP=nP,
                    save_agent=True,
                    dir_name=dir_name,
                    optim_type='',
                    state_space=state_space
                    )


    # Get train and test sets for Ppv and Pconso 
    train_Ppvs, train_Pconsos, test_Ppvs, test_Pconsos = get_train_test_sets(env, N_train=N_train, N_test=N_test)
    # train_Ppvs, train_Pconsos, test_Ppvs, test_Pconsos = get_train_test_sets(env)
    # Get the mean bills and all the bills for the trained and untrained (reference) agent
    train_agent(env, Agent, train_Ppvs, train_Pconsos)

    def estimate_Q(day, t, V, policy):
        costs = Agent.explorer.explore(day, t)
        if Agent.state_space != 'S4':
            S = tuple(Agent.explorer.tuples.T)
            Q = V[1, Agent.explorer.new_SOCs] + costs
            if Agent.state_space != 'S1':
                if Agent.state_space == 'S2':
                    D = env.Pconso[t] - env.Ppv[t]
                    D = np.sign(D) + 1
                elif Agent.state_space == 'S3':
                    D = env.Pconso[t] - env.Ppv[t]
                    D = np.clip(D, Agent.delta_min, Agent.delta_max)
                    D = (D - Agent.delta_min) // Agent.delta_diff
                actions = policy[t, D]
            else:
                actions = policy[t]
            V[0] = Q[np.arange(Agent.nb_states), actions]
                
            Agent.q_table[S] += Agent.lr(Agent.count[S]) * (Q.reshape(-1) - Agent.q_table[S])
            Agent.count[S]+= 1
        else:
            Pmeters = Agent.explorer.Pmeters.sum(axis=-1)
            Pmeters = np.round(6* Pmeters + Agent.explorer.env.Pconso[t] - Agent.explorer.env.Ppv[t]).astype(int).reshape(-1)

            S = tuple(Agent.explorer.tuples.T)
            Q = np.zeros((Agent.nP, Agent.nb_states, Agent.nb_actions), dtype=np.float32)     
            D = env.Pconso[t] - env.Ppv[t]
            D = np.clip(D, Agent.delta_min, Agent.delta_max)
            D = (D - Agent.delta_min) // Agent.delta_diff

            states = np.array([[x, a] for x in range(Agent.nb_states) for a in range(Agent.nb_actions)])
            for p in range(Agent.nP):
                new_P = np.maximum(Pmeters, p)
                new_states = np.stack([new_P, Agent.explorer.new_SOCs.reshape(-1)], axis=1)
                Q[p][tuple(states.T)]=V[1][tuple(new_states.T)] + costs.reshape(-1) + (new_P - p) * Agent.price_rate
                actions = policy[t, D, p]
                V[0, p] = Q[p][np.arange(Agent.nb_states), actions]

            Agent.q_table[S] += Agent.lr(Agent.count[S]) * (Q.reshape(-1) - Agent.q_table[S])
            Agent.count[S]+= 1
    

    if Agent.state_space != 'S4':
        V = np.zeros((N_train, 2, Agent.nb_states))
    else:
        V = np.zeros((N_train, 2, Agent.nP, Agent.nb_states)) 
    policy = Agent.q_table.argmin(axis=-1)
    
    for t in tqdm(range(Agent.H-1, -1, -1)):
        stop = False
        while not stop:
            Agent.q_table[t].fill(0)
            Agent.count[t].fill(1)
            for day in range(N_train):
                estimate_Q(day, t, V[day], policy)
            
            new_policy = Agent.q_table.argmin(axis=-1)
            stop = np.array_equal(new_policy, policy)
            policy = new_policy
        V[:, 1] = V[:, 0]

    np.save(os.path.join(Agent.path, 'PI-q-table.npy'), Agent.q_table)



if __name__ == '__main__':
    # Only three possibilities for the agent
    state_spaces = ['S1', 'S2', 'S3', 'S4']
    if len(sys.argv) == 2 and sys.argv[1] in state_spaces:
        # Store the result data in the specified directory
        run('test', sys.argv[1])
    else:
        print("Enter 'S1', 'S2', 'S3' or 'S4'")
        

