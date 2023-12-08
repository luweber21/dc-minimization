import sys
from src.agent.utils import *
from src.agent.agent import RLAgent
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



if __name__ == '__main__':
    # Only three possibilities for the agent
    state_spaces = ['S1', 'S2', 'S3', 'S4']
    if len(sys.argv) == 2 and sys.argv[1] in state_spaces:
        # Store the result data in the specified directory
        run('bill_min', sys.argv[1])
    else:
        print("Enter 'S1', 'S2', 'S3' or 'S4'")
        
