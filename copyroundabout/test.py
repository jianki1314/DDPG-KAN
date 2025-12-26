##############测试代码##########
####作用######
'''
主要检测环境代码是否能执行，并且通过仿真测试，该代码能够实现
车辆的随机生成，并且能够实现每轮仿真
获取默认数据的代码
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
from env_test import SumoEnv  
import torch as tt
import CustomNetwork as network
import traci
import os,sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

###DDPG算法 td3 最新的奖励函数，能耗较高、时间较低
def compute_reward(self):
    #安全奖励函数
    L_width = 3.2
    L_lateral = traci.vehicle.getLateralLanePosition(self.vehicle)
    # R_lc = 1 - math.pow(L_lateral/L_width,2)
    R_lc = -abs(L_lateral) / L_width 

    leader = traci.vehicle.getLeader(self.vehicle)
    if leader is not None:
        leader_id, gap = leader
        leader_speed = traci.vehicle.getSpeed(leader_id)
        speed_diff = self.speed_x - leader_speed
        # 仅当本车速度大于领导车速度且gap大于0时，计算ttc
        if speed_diff > 0 and gap > 0:
            ttc = gap / speed_diff
        else:
            # 当本车速度小于等于领导车速度或gap不大于0时，设置ttc为一个很大的值
            ttc = 10.0
    else:
        # 如果没有领导车，设置ttc为一个很大的值
        ttc = 10.0
    R_ttc = 1 / ( 1 + ttc )
    R_safe = 100 * ( R_lc + R_ttc )
    #效率奖励 
    # R_efficient =  1 - (abs(self.speed_limit - self.speed_x) / self.speed_limit)
    # R_efficient *= 10
    speed_diff = abs(self.speed_limit - self.speed_x)
    R_efficient = 50 * (1 - speed_diff / self.speed_limit)  # 调整系数

    #舒适度j奖励计算
    alpha = 5
    jerk = self.compute_jerk()
    # print("jerk is ===========>>>",jerk)
    R_comfort = -alpha * jerk



    #能耗奖励计算
    beta = 100
    E_consumed = float(traci.vehicle.getParameter(self.vehicle, "device.battery.totalEnergyConsumed"))##消耗的能量
    E_max = float(traci.vehicle.getParameter(self.vehicle, "device.battery.maximumBatteryCapacity"))
    E_regenerated= float(traci.vehicle.getParameter(self.vehicle, "device.battery.totalEnergyRegenerated"))##回收的能量
    # distance_traveled = traci.vehicle.getDistance(self.vehicle)  # 获取车辆行驶距离
    # R_energy = -beta * ( (E_consumed - E_regenerated) / distance_traveled )
    print("消耗能量------>>>",E_consumed)
    print("回收能量------>>>",E_regenerated)
    if E_consumed >= E_regenerated:
        ratio = (E_regenerated / E_consumed)
    else:
        # if E_regenerated > 0:
        #     ratio = E_regenerated
        # else:
        ratio = -2
    R_energy = beta * ratio



    #到达目的奖励
    arrive_flash = self.arrive_reward()
    if arrive_flash:
        R_arrive = 50
    else: 
        R_arrive = 0
    
    #怠速惩罚
    idle_speed = 0.1
    if self.speed_x < idle_speed:  # 低于阈值则进行惩罚
        R_idle_penalty = -100 * (idle_speed - self.speed_x)
    else:
        R_idle_penalty = 0

    R_total = 0.2 * R_safe + 0.3 * R_efficient + 0.2 * R_comfort + 0.3 * R_energy + R_arrive + R_idle_penalty
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("R_safe is ===========>>>",R_safe)
    print("R_efficient is ===========>>>",R_efficient)
    print("R_comfort is ===========>>>",R_comfort)
    print("R_energy is ===========>>>",R_energy)
    print("E_arrive is ===========>>>",R_arrive)
    print("E_idle_penalty is ===========>>>",R_idle_penalty)
    print("R_total =====================>>>",R_total)
    print("+++++++++++++++++++++++++++++++++++++++++++++")

    return R_total

def main_test():
    env = SumoEnv(render_mode="rgb_array",max_episodes=500,flash_episode=True)
    steps = 500
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


