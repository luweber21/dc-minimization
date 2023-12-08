import numpy as np
from tqdm import tqdm
from src.agent.simulations import *
from src.environment.DiscreteBattery import DiscreteBattery

def get_train_test_sets2(env):
    train_Ppvs = []
    train_Pconsos = []
    test_Ppvs = []
    test_Pconsos = []
    for day in range(env.full_Ppv.shape[0] // 144):
        Ppv = env.full_Ppv[144*day:144*(day+1)]
        Pconso = env.full_Pconso[144*day:144*(day+1)]
        # if day % 2 == 0:
        train_Ppvs.append(Ppv)
        train_Pconsos.append(Pconso)
        # else:
        if day % 2 == 1:
            test_Ppvs.append(Ppv)
            test_Pconsos.append(Pconso)


    return train_Ppvs, train_Pconsos, test_Ppvs, test_Pconsos


def get_train_test_sets(env, N_train=300, N_test=100):
    '''
    Retrieve the data corresponding N_train training days and N_test test days.
    Every second day is used for the test set, the other one is used for the training.

    This separation attempts to prevent seasonality issues

    If N_train > N_test (resp N_train < N_test), then first odd (resp even) days will be used to train (resp test) as well.
    '''
    train_Ppvs = []
    train_Pconsos = []
    test_Ppvs = []
    test_Pconsos = []

    # days = list(range(0, N_train + N_test, 2)) + list(range(1, N_train + N_test, 2))
    training_set = list(range(N_train))
    test_set = list(range(N_train, N_train + N_test))


    for day in training_set:
        train_Ppvs.append(env.full_Ppv[144*day:144*(day+1)])
        train_Pconsos.append(env.full_Pconso[144*day:144*(day+1)])
        
    for day in test_set:
        test_Ppvs.append(env.full_Ppv[144*day:144*(day+1)])
        test_Pconsos.append(env.full_Pconso[144*day:144*(day+1)])
    
    return train_Ppvs, train_Pconsos, test_Ppvs, test_Pconsos


def train_agent(env, Agent, train_Ppvs, train_Pconsos):
    N_train = len(train_Ppvs)

    # Collect all the transitions corresponding to the N_train days, so that we don't have to collect them anymore.
    # These transitions can be stored
    print("Collecting transitions for the training...")
    Agent.collect_transitions(env,
                              train_Ppvs,
                              train_Pconsos,)

    print("Training the main agent...")
    # Train the agent. The agent has an attribute containing the datas of the N_train days.
    # Agent.train(env, N_train, print_tqdm=True)


def get_env(datafile='Data_cryolite.csv'):
    '''
    Function returning a new environment
    '''
    env = DiscreteBattery(H=144, data_file=datafile, discrete_action_space = False)

    return env


