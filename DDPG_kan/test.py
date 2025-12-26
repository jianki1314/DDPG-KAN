##############测试代码##########
####作用######
'''
主要检测环境代码是否能执行，并且通过仿真测试，该代码能够实现
车辆的随机生成，并且能够实现每轮仿真
'''
from sumolib import checkBinary
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
import xml.etree.ElementTree as ET
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from env import SumoEnv  
import torch as tt
import CustomNetwork as network
import traci
import os,sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
cfg_path = "networks/sumoconfig1.sumo.cfg"

def get_data(data,i):
    sys.stdout.flush()
    output_folder = os.path.join("netwroks/output_data/output_default")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    filename_output = f"data_default-{i}.csv"
    try:
        data.to_csv(os.path.join(output_folder, filename_output))
    except:
        print("Error saving file!")

##对输出文件更名##
def modify_output_prefix(cfg_file, new_prefix):
    tree = ET.parse(cfg_file)
    root = tree.getroot()
    output_prefix = root.find('.//output-prefix')
    if output_prefix is not None:
        output_prefix.set('value', new_prefix)
    tree.write(cfg_file)

def main_test():
    env = SumoEnv()
    steps = 1000
    count = 0
    for i in range (steps):
        modify_output_prefix(cfg_path,f'{i}-')
        env.reset(gui=False)
        done = False
        while not done:
            state = env.get_state()
            traci.simulationStep()
            done = env.check_terminated()
        env.close()
        print("########    break    #######")
        count += 1
        print("count is ===========>>",count)      
if __name__ == "__main__":

    main_test()
    
    print('----------------ALL ---------END-----------------------')
