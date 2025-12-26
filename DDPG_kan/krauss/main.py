##############测试代码##########
####作用######
'''
主要检测环境代码是否能执行，并且通过仿真测试，该代码能够实现
车辆的随机生成，并且能够实现每轮仿真
获取默认数据的代码
注意:普通车辆数更改为 80 辆
'''

# from env_test_LSTM import SumoEnv  
from env_test import SumoEnv
import torch as tt
import traci
import os,sys


def main_test():
    env = SumoEnv(render_mode="rgb_array",max_episodes=200,flash_episode=True)
    steps = 200
    for i in range (steps):
        obs = env.reset()
        done = False
        while not done:
            traci.simulationStep()
            state = env.get_state()

            done = env.check_terminated()
        print("回合==============>>>>>>:",i)   
    # env.colse()   
if __name__ == "__main__":

    main_test()##获取默认数据
    
    print('----------------ALL ---------END-----------------------')
