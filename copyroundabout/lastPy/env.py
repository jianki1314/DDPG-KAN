'''
初始版本env
获取默认数据的代码的环境
'''
import gym
from gym import spaces
import traci
import random
import sumolib
import numpy as np
import pandas as pd
from math import atan2, degrees
from collections import deque
import math
import os
import sys

np.set_printoptions(suppress=True, precision=3)

class SumoEnv(gym.Env):
    def __init__(self):
        #车辆信息相关变量
        self.vehicle = 'rl_0'#智能车辆id
        self.name = "rl_0"
        self.vehicleNum = 0 #车辆数量
        self.presence = False #车辆存在性判断
        self.step_length = 0.4
        self.pos = (0,0)
        self.posx = 0 #x轴位置
        self.posy = 0 #y轴位置
        self.speed_x = 0 #x轴速度
        self.speed_y = 0 #y轴速度
        self.lane_heading_difference = 0 #车道航向差
        self.curr_lane = '' #车辆当前车道名称
        self.lane_distance = 0 #车辆与车道中心线的距离
        self.throttle = 0 #加速度输入
        self.steering  = 0 #方向盘输入
        self.acc = 0 #车辆加速度
        self.acc_history = deque([0, 0], maxlen=2) #存储加速度值
        self.state_dim = 6 #状态维度
        self.angle = 0 #车辆的角度
        self.gui = False #仿真界面启动开关值
        self.done = False

        ##定义状态空间和动作空间##
        self.action_space = spaces.Box(low=np.array([-3.5]), high=np.array([4.0]), shape=(1,),dtype=np.float32)
        self.low = np.array([-80, -80, 0, 0, -180, -3.2])
        self.high = np.array([80, 80, 14, 14,180, 3.2])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)

    
    def start(self,gui=False, vehicleNum=36, network_cfg="networks/sumoconfig1.sumo.cfg", network_net="networks/rouabout.net.xml",  vtype1="rlAnget", vtype2="human"):
        self.gui = gui
        self.vehicleNum = vehicleNum
        self.sumocfg = network_cfg
        self.net_xml = network_net
        self.rlvType = vtype1
        self.humanvType = vtype2
        self.done = False

        #starting sumo
        #home = os.getenv("HOME") or os.getenv("USERPROFILE")  # 兼容Windows环境
        home = "/home/jian/sumo"
        print("home is:",home)
        if self.gui:
            sumoBinary = os.path.join(home,"bin/sumo-gui")
        else:
            sumoBinary = os.path.join(home,"bin/sumo")
        sumoCmd = [sumoBinary, "-c", self.sumocfg,"--lateral-resolution","3.2",
         "--start", "True", "--quit-on-end", "True","--no-warnings","True", "--no-step-log", "True", "--step-method.ballistic", "true"]
        traci.start(sumoCmd)
        print("Using SUMO binary:", sumoBinary)

        self.generate_vehicles(self.vehicleNum)
        traci.simulationStep()

        self.update_params()
    
    
    ######车辆生成#########
    def generate_vehicles(self, vehicleNum):
        # 定义所有可能的路线ID列表
        possible_routes = ["route_0", "route_1", "route_2", "route_3","route_4", "route_5", "route_6", "route_7", "route_8", "route_9", "route_10", "route_11"]
        # 为每辆车随机分配路径
        random_routes = [random.choice(possible_routes) for _ in range(vehicleNum)]
        depart_times = []
        for i in range(vehicleNum):
            veh_name = 'vehicle_' + str(i)  # 普通车辆名称
            route_id = random_routes[i]  # 为每辆车随机分配路线
            depart_time = random.uniform(0, 30)  # 在前30秒内随机发车
            depart_times.append(depart_time)  # 收集发车时间
            traci.vehicle.add(veh_name, routeID=route_id, typeID=self.humanvType, departLane='free', departSpeed='random',depart=depart_time)
        if depart_times:
            avg_depart_time = sum(depart_times) / len(depart_times) + 5     
        # 智能车辆生成
        smart_vehicle_route_id = random.choice(possible_routes)  # 智能车辆随机选择一个路线
        traci.vehicle.add(self.vehicle, routeID=smart_vehicle_route_id, typeID=self.rlvType, departLane='random', departSpeed='random',depart=avg_depart_time)

    def update_params(self):
        # 获取基本车辆信息
        self.pos = traci.vehicle.getPosition(self.vehicle)
        self.posx,self.posy = self.pos
        self.curr_lane = traci.vehicle.getLaneID(self.vehicle)
        self.angle = traci.vehicle.getAngle(self.vehicle)
        
        if self.curr_lane != '':
            self.speed_limit = traci.lane.getMaxSpeed(self.curr_lane)
            lane_angle = traci.lane.getShape(self.curr_lane)[-1][-1]  # 假设航向为车道形状末点的角度
            self.lane_heading_difference = lane_angle - self.angle
            self.lane_distance = traci.vehicle.getLateralLanePosition(self.vehicle) 
            # 将航向差调整到-180到180的范围内
            if self.lane_heading_difference > 180:
                self.lane_heading_difference -= 360
            elif self.lane_heading_difference < -180:
                self.lane_heading_difference += 360 
            # 车道的子标签（如果需要）
            self.curr_sublane = int(self.curr_lane.split("_")[1]) if "_" in self.curr_lane else 0
        else:
            self.speed_limit = 14
            self.lane_heading_difference = 0
            self.lane_distance = 0

        # 获取车辆速度及其他信息
        #self.speed_limit = traci.lane.getMaxSpeed(self.curr_lane)
        #self.target_speed = traci.vehicle.getAllowedSpeed(self.vehicle)
        self.speed_x = traci.vehicle.getSpeed(self.vehicle)
        self.speed_y = traci.vehicle.getLateralSpeed(self.vehicle)  # 如果需要横向速度
        self.acc = traci.vehicle.getAcceleration(self.vehicle)
        self.acc_history.append(self.acc)  # 您可能需要初始化 acc_history作为一个列表
        #self.angle = traci.vehicle.getAngle(self.vehicle)

    ####打印状态函数####
    def log_state_to_file(self,state):
        with open("vehicle_states.long","a") as file:
            file.write(f"{state}\n")
            
    def get_state(self):
        all_vehicle  =  traci.vehicle.getIDList()
        state = None
        states = []
        if self.vehicle not in all_vehicle:
            state = np.zeros(self.observation_space.shape)
        else:
            self.update_params() #更新车辆信息
            state = np.array([
                round(self.posx, 3),
                round(self.posy, 3),
                round(self.speed_x, 3),
                round(self.speed_y, 3),
                round(self.lane_heading_difference, 3),
                round(self.lane_distance, 3)         
            ])
        #保存打印信息
        # if state is not None:
        #     states.append(state)
        #     self.log_state_to_file(states)
        return state
        
    def reset(self,gui=False):
        print("Resetting environment with GUI set to:", gui)
        self.start(gui)
        return self.get_state()
    
    def check_terminated(self):
        if traci.vehicle.getIDCount() == 0:
            return True
        else:
            return False
    def step(self,action):
        #将action中的加速度值输入到sumo中
        traci.vehicle.setAcceleration(self.vehicle, action, self.step_length)
        traci.simulationStep()
        reward = self.compute_reward()
        self.update_params()
        next_state = self.get_state()
        done = self.check_terminated()
        info = {}          # 可选的附加信息
        
        return next_state, reward, done, info
    
    ###加速度积分###
    def compute_jerk(self):
        return (self.acc_history[1] - self.acc_history[0])/self.step_length
    
    def compute_reward(self):
        #安全奖励函数
        #lane_id = traci.vehicle.getLaneID(self.vehicle)
        L_width = 3.2
        L_lateral = traci.vehicle.getLateralLanePosition(self.vehicle)
        R_lc = 1 - math.pow(L_lateral/L_width,2)

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
                ttc = float('inf')
        else:
            # 如果没有领导车，设置ttc为一个很大的值
            ttc = float('inf')
        # 如果ttc是无穷大，R_ttc应该设置为最大奖励值1
        if ttc == float('inf'):
            R_ttc = 1
        else:
            R_ttc = 1 - 3/ttc if ttc > 0 else -1 
        R_safe = 0.7 * R_lc + 0.3 * R_ttc

        #效率奖励
        speed_max = traci.vehicle.getMaxSpeed(self.vehicle)
        if self.speed_x > self.speed_limit:
            R_efficient = self.speed_x / self.speed_limit
        else:
            R_efficient = 1 - ( (self.speed_x - self.speed_limit) / (speed_max - self.speed_limit) )

        #舒适度计算
        jerk = self.compute_jerk()
        R_comfort = 1 - jerk/2

        #能耗奖励计算
        E_consumed = float(traci.vehicle.getParameter(self.vehicle, "device.battery.totalEnergyConsumed"))
        E_max = float(traci.vehicle.getParameter(self.vehicle, "device.battery.maximumBatteryCapacity"))
        E_regenerated= float(traci.vehicle.getParameter(self.vehicle, "device.battery.totalEnergyRegenerated"))
        R_energy = 1 - ( (E_consumed - E_regenerated) / E_max ) 
        
        R_total = 0.3*R_safe + 0.25*R_efficient + 0.1*R_comfort + 0.35*R_energy

        return R_total
    
    def close(self):
        traci.close()
