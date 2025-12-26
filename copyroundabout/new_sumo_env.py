import gymnasium as gym
import xml.etree.ElementTree as ET
from typing import Callable, Optional, Tuple, Union
from gymnasium import spaces
import traci
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import math
import os
import random
import torch

np.set_printoptions(suppress=True, precision=3)
##状态空间定义##
def creat_observation():
    state_space_list = ['speed_x', 'speed_y','acc','front_vehicle_distance','front_vehicle_to_speed','rear_vehicle_distance','rear_vehicle_to_speed']
    state_space_low = np.array([0, 0, -10, 0, -14, 0, -14],dtype=np.float32)
    state_space_high = np.array([14, 14, 10, 50, 14, 50, 14],dtype=np.float32)
    obs = spaces.Box(low=state_space_low,high=state_space_high,shape=(7,),dtype=np.float32)
    return obs

##动作空间定义##
def creat_action():
    action_space = spaces.Box(low=np.array([-3.0],dtype=np.float32), high=np.array([4.0],dtype=np.float32),shape=(1,),dtype=np.float32)
    return action_space

class SumoEnv(gym.Env):
    metadata = {"render_modes": ["", "human", "rgb_array"], "render_fps": 4}
    def __init__(
            self, 
            render_mode: Optional[str] = None,
            max_episodes: int = 3,
            flash_episode: bool = False,#训练模型时为False，仿真测试时为True
        ) -> None:
        super(SumoEnv, self).__init__()
        #车辆信息相关变量
        self.vehicle = "rl_0"#智能车辆id
        self.name = "rl_0"
        self.vehicleNum = 0 #车辆数量
        self.arrive_flash= False #车辆到达目的地奖励标志
        self.step_length = 0
        self.speed_x = 0 #x轴速度
        self.speed_y = 0 #y轴速度
        ##
        self.front_vehicle_distacne = 0 #前车距离
        self.front_vehicle_to_speed = 0 #前车速度差
        self.rear_vehicle_distance = 0 #后车距离
        self.rear_vehicle_to_speed = 0 #后车速度差
        ##
        self.lane_heading_difference = 0 #车道航向差
        self.curr_lane = '' #车辆当前车道名称
        self.lane_distance = 0 #车辆与车道中心线的距离
        self.throttle = 0 #加速度输入
        self.acc = 0 #车辆加速度
        self.acc_rl = 0 #存储算法的加速度值
        self.acc_history = deque([0, 0], maxlen=2) #存储加速度值
        self.consumed_history = 0
        self.regenerated_history = 0
        self.prev_distance = 0
        self.state_dim = 6 #状态维度
        self.angle = 0 #车辆的角度
        self.gui = False #仿真界面启动开关值
        self.terminated = False 
        self.truncated = False
        self.low_speed_count = 0 #怠速累积计数
        self.low_speed_thershold = 10 #怠速累积阈值
        self.speed_zero_count = 0 #初始化速度计数器
        self.simulation_running = False #仿真是否开标志
        self.add_reward = False #额外奖励值添加标志

        self.episode_test = 0 #仿真回合设置
        self.episode_train = 0 #仿真回合设置
        self.metrics = [] #记录每回合的其他信息
        self.max_episodes = max_episodes#最大回合数设置
        self.flash_episode = flash_episode #是否开启回合数限制
        self.sim_max_time = 150 #最大仿真时间

        self.should_terminate = False

        self.idle_speed_flash = False #怠速标志
        self.save_csv_flash = False #csv判断输出标志
        self.collision_count = 0 #碰撞警示累积数
 
        #动作空间和状态空间
        self.action_space =  creat_action()
        self.observation_space = creat_observation()

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode =render_mode
        print(self.render_mode)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            super().reset(seed=seed)
        print("Resetting environment with GUI !!!!!!!!! reset")
        next_state = None
        self.should_terminate = False
        self.idle_speed_flash = False
        self.collision_count = 0
        self.low_speed_count = 0
        self.speed_zero_count = 0
        self.speed_x = 0 #x轴速度
        self.speed_y = 0 #y轴速度
        self.acc_rl = 0
        self.acc = 0
        self.acc_history = deque([0, 0], maxlen=2) #存储加速度值
        info1 = {}

        ##仿真测试模式
        if self.flash_episode:
            if self.episode_test < self.max_episodes:
                if self.episode_test != 0:
                    self.save_csv(self.episode_test - 1)
                    self.close()
                self.modify_output_prefix(f'{self.episode_test}-')
                print("episode================================================>>>   测试",self.episode_test)
                self.episode_test += 1
                self.metrics = [] #重置metrics列表
                
                self.save_csv_flash = False
                self.terminated = False 
                self.truncated = False
                self.add_reward = False
                self.start()
                next_state= self.get_state()
                info1 = self._getInfo()     
            else:
                print("$$$$$$$$$$$  已到达最大回合数  ￥￥￥￥￥￥￥￥￥")
                self.should_terminate = True
                next_state = np.zeros(self.observation_space.shape)
                info1 = {"message": "True"}
        else:
            ##训练模型模式
            if self.episode_train != 0:
                 self.close()
            self.episode_train += 1
            print("episode================================================>>>  训练 ",self.episode_train)
            self.metrics = []

            self.terminated = False 
            self.truncated = False
            self.add_reward = False
            self.start()
            next_state= self.get_state()
            info1 = self._getInfo()
        if next_state is not None and isinstance(info1,dict):
            return next_state, info1

         
    ##对输出文件更名##
    def modify_output_prefix(self,new_prefix):
        cfg_file = "networks/sumoconfig_DDPG.sumo.cfg"
        tree = ET.parse(cfg_file)
        root = tree.getroot()
        output_prefix = root.find('.//output-prefix')
        if output_prefix is not None:
            output_prefix.set('value', new_prefix)
        tree.write(cfg_file)

    def save_csv(self, episode):
        base_path = "networks/output_data/output_ddpg/state"
        self.label = "state"
        file_name = f"{episode}-{self.label}.csv"
        full_path = Path(base_path) / file_name
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.metrics)
        df.to_csv(full_path, index=False)
        print(f"#########Data saved to {full_path}########")
        # if self.save_csv_flash:
        #     print("#########数据异常抛弃本回合数据############")
        # else:
        #     df.to_csv(full_path, index=False)
        #     print(f"#########Data saved to {full_path}########")

    ##仿真启动##
    def start(self,gui=True, vehicleNum=80, network_cfg="networks/sumoconfig_DDPG.sumo.cfg", network_net="networks/rouabout.net.xml",  vtype1="rlAnget", vtype2="human"):
        self.gui = gui
        self.vehicleNum = vehicleNum
        self.sumocfg = network_cfg
        self.net_xml = network_net
        self.rlvType = vtype1
        self.humanvType = vtype2

        #starting sumo
        #home = os.getenv("HOME") or os.getenv("USERPROFILE")  # 兼容Windows环境
        home = "/home/jian/sumo"
        # print("home is:",home) #打印当前地址
        if self.render_mode=="human":
            sumoBinary = os.path.join(home,"bin/sumo-gui")
        else:
            sumoBinary = os.path.join(home,"bin/sumo")
        #sumoCmd = [sumoBinary, "-c", self.sumocfg ,"--no-step-log", "true" , "-W"]
        sumoCmd = [sumoBinary, "-c", self.sumocfg,"--lateral-resolution","3.2",
         "--start", "True", "--quit-on-end", "True","--no-warnings","True", "--no-step-log", "True", "--step-method.ballistic", "true"]
        traci.start(sumoCmd)
        # print("Using SUMO binary:", sumoBinary)  #输出当前信息
        self.step_length = traci.simulation.getDeltaT()##获取仿真单位时间步
        
        self.generate_vehicles(self.vehicleNum)#生成车辆
        done1 = False
        while not done1:
            all_vehicles = traci.vehicle.getIDList()
            if self.vehicle not in all_vehicles:
                done1 = False
                traci.simulationStep()
                #print(f"Vehicle '{self.vehicle}' was not added to simulation properly.")
            else:
                done1 = True
                print(f"Vehicle '{self.vehicle}' was  added to simulation properly !!!")

        self.update_params()#更新车辆参数
    
    
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

    #更新车辆参数
    def update_params(self):
        # 获取基本车辆信息
        self.speed_x = traci.vehicle.getSpeed(self.vehicle)
        self.speed_y = traci.vehicle.getLateralSpeed(self.vehicle)  # 横向速度
        self.acc = traci.vehicle.getAcceleration(self.vehicle)
        self.acc_history.append(self.acc)  # 您可能需要初始化 acc_history作为一个列表
        self.curr_lane = traci.vehicle.getLaneID(self.vehicle)
        if self.curr_lane != '':
            self.speed_limit = traci.lane.getMaxSpeed(self.curr_lane)
        else:
            self.speed_limit = 14

        # 获取目标车前车信息
        leader = traci.vehicle.getLeader(self.vehicle, dist=50)  # 查看xxx米范围内的前车
        if leader:
            leader_id, leader_distance = leader
            if leader_id:
                leader_speed = traci.vehicle.getSpeed(leader_id)
                self.front_vehicle_distacne = leader_distance
                self.front_vehicle_to_speed = leader_speed - self.speed_x
                self.get_safe_info(leader_distance,self.speed_x) #获取易碰撞累计数
        else:
            self.front_vehicle_distacne = 0
            self.front_vehicle_to_speed = 0

        # 获取后车（跟随车）信息
        follower = traci.vehicle.getFollower(self.vehicle, dist=50)  # 查看xxx米范围内的后车
        if follower:
            follower_id, follower_distance = follower
            if follower_id:
                follower_speed = traci.vehicle.getSpeed(follower_id)
                self.rear_vehicle_distance = follower_distance
                self.rear_vehicle_to_speed = follower_speed - self.speed_limit
        else:
            self.rear_vehicle_distance = 0
            self.rear_vehicle_to_speed = 0

    #获取状态空间状态      
    def get_state(self):
        # all_vehicle  =  traci.vehicle.getIDList()
        # if self.vehicle in all_vehicle:
                
        state = None
        states = []
        self.update_params() #更新车辆信息
        state = np.array([
            round(self.speed_x, 3),
            round(self.speed_y, 3),
            round(self.acc, 3),
            round(self.front_vehicle_distacne,3),
            round(self.front_vehicle_to_speed,3),
            round(self.rear_vehicle_distance,3),
            round(self.rear_vehicle_to_speed,3)     
        ])
        '''
        #打印状态信息#
        labels = [
            "Speed X (m/s):",
            "Speed Y (m/s):",
            "Acceleration (m/s^2):",
            "Front Vehicle Distance (m):",
            "Front Vehicle Speed Difference (m/s):",
            "Rear Vehicle Distance (m):",
            "Rear Vehicle Speed Difference (m/s):"
        ]
        # 打印每个状态
        for label, value in zip(labels, state):
            print(f"{label} {value}")
        '''
        state_space_low = np.array([0, 0, -10, 0, -14, 0, -14],dtype=np.float32)
        state_space_high = np.array([14, 14, 10, 50, 14, 50, 14],dtype=np.float32)
         # 归一化处理
        normalized_state = 2 * ((state - state_space_low) / (state_space_high - state_space_low)) - 1
    
        return normalized_state
    
    #执行动作
    def applyAction(self, action):
        # print("aciton is ------->>",action)
        # print(f"Action space range: {self.action_space.low} to {self.action_space.high}")
        if not self.action_space.contains(action):
            # 如果不在，则将控制权交给SUMO，默认使用跟随模型
            traci.vehicle.setSpeed(self.vehicle, speed=-1)
        else:
            action = float(action)
            self.acc_rl = action
            # print("self.acc_rl is =============>>",self.acc_rl)
            speed_limit = self.speed_limit
            v_current = traci.vehicle.getSpeed(self.vehicle)
            v_next = max(min(speed_limit, v_current + action), 0)
            if v_current <= 0.1 and action <= 0:
                traci.vehicle.setSpeed(self.vehicle, speed=-1)
                traci.simulationStep()
            if v_current <= 0.1 and action > 0:
                traci.vehicle.setSpeed(self.vehicle, speed=-1)
                traci.simulationStep()
            else:
                traci.vehicle.setSpeed(self.vehicle, speed=v_next)
                traci.simulationStep()
            # print("Modified Action is ------------->", action)
            # traci.vehicle.slowDown(self.name, v_next, self.step_length)
            # traci.vehicle.setSpeed(self.vehicle, speed=v_next)
            # print("current speed is===========>>",v_current)
            # traci.vehicle.setAcceleration(self.name, action, self.step_length)

    ##根据reset返回额外值，这里打印一个训练回合开始##
    def _getInfo(self):
        return {"current_episode":0}  
    
    def check_truncated(self, action):
        min_speed = 1e-5
        if self.speed_x <= min_speed and action < 0:
            self.low_speed_count += 1
        else:
            self.low_speed_count = 0

        if self.speed_x <= min_speed and action > 0:
            self.speed_zero_count += 1
        else:
            self.speed_zero_count = 0

        if self.speed_zero_count >= 30:
            print("####### 车辆长时间速度为0，终止回合 ############ ")
            return True
        if self.low_speed_count >= 10:
            print("####### 车辆长时间怠速，终止回合 ############ ")
            return True

        return False
    
    #检查车辆是否到达了它的路线终点
    def check_terminated(self):
        sim_step = self.sim_step
        if sim_step >= self.sim_max_time:
            print("####### 仿真时间达到最大值 ############ ")
            return True  # 仿真时间达到最大值，终止回合
        else:
            return False

        # all_vehicles = traci.vehicle.getIDList()
        # if self.vehicle not in all_vehicles:
        #     print("####### 车辆已不在仿真中，终止回合 ############ ")
        #     return True  # 车辆已不在仿真中，终止回合

        # route = traci.vehicle.getRoute(self.vehicle)
        # last_edge = route[-1]
        # current_edge = traci.vehicle.getRoadID(self.vehicle)
        # last_edge_lane = last_edge + "_0"  # 假定索引为0的车道

        # if current_edge == last_edge and traci.vehicle.getLanePosition(self.vehicle) >= traci.lane.getLength(last_edge_lane):
        #     print("####### 到达目的地 ############ ")
        #     return True

        # if self.speed_x <= 0.1 and self.speed_zero_count >= 30:
        #     print("####### 车辆长时间速度为0，终止回合 ############ ")
        #     return True

        return False

    ##检查车辆仿真是否结束 或车辆是否怠速##
    def check_truncated1(self,action):
        min_speed = 1e-5
        all_vehicle = traci.vehicle.getIDList()
        route = traci.vehicle.getRoute(self.vehicle)
        last_lane = route[-1]
        # print("last edge is ---->>",last_lane)
        current_edge = traci.vehicle.getRoadID(self.vehicle)
        # print("current_edge is ---->>",current_edge)
        last_edge_lane = last_lane + "_0"  # 假定索引为0的车道

        if self.speed_x <= min_speed and action < 0:
            self.low_speed_count += 1
            # print("vehicle's speed is -------->>",self.speed_x)
            # print(f"Idle steps: {self.low_speed_count}")
        else:
            self.low_speed_count = 0

        if self.speed_x <= min_speed and action > 0:
            self.speed_zero_count += 1
            print("########## 车辆等待前车中 ############")
        else:
            self.speed_zero_count = 0
        
        ######################怠速情况：跟随模型速度########################
        # if self.speed_zero_count >=10:
        #     print("##########车辆怠速，使用默认跟随模型##########")
        #     traci.vehicle.setSpeed(self.vehicle, speed=-1)
        #     # self.low_speed_count = 0
        #     self.speed_zero_count = 0
        ##################################################################
        
        if traci.vehicle.getIDCount() == 0:
            print("￥￥￥￥￥￥ 车辆无了  1 ￥￥￥￥￥￥￥")
            return True
        if self.speed_zero_count >= 30:
            print("￥￥￥￥￥￥ 速度归 000 ￥￥￥￥￥￥")
            self.save_csv_flash = True
            return True
        if self.low_speed_count >= 10:
            print("￥￥￥￥￥￥ 车辆怠速 ￥￥￥￥￥￥￥")
            self.save_csv_flash = True
            return True
        if current_edge == last_lane and traci.vehicle.getLanePosition(self.vehicle) >= traci.lane.getLength(last_edge_lane):
            # 车辆处于最后一段路且已经超过该路段的长度，可以认为到达目的地
            print("#######  到达目的地1   ############ ")
            return True
        else:
            return False

    @property
    def sim_step(self) -> float:
        return traci.simulation.getTime()#返回仿真时间

    def step(self,action):
        all_vehicle = traci.vehicle.getIDList()
        # print("action is ============>>>",action)
        if self.vehicle not in all_vehicle:
            self.terminated = False
            if traci.vehicle.getIDCount() == 0:
                self.truncated = True
                print("￥￥￥￥￥￥ 车辆无了  2 ￥￥￥￥￥￥￥")
            else:
                self.truncated = False
            turncated = self.truncated
            reward = 0.0
            next_state = np.zeros(self.observation_space.shape)
        else:
            assert self.action_space.contains(action), f"Invalid action: {action}"
            #将action中的加速度值输入到sumo中
            self.applyAction(action)
            # traci.simulationStep()
            all_vehicle = traci.vehicle.getIDList()
            if self.vehicle in all_vehicle:
                reward = self.reward()
                next_state = self.get_state()
                # self.terminated = self.check_terminated()##检测智能车辆是否到达终点
                self.truncated = self.check_truncated(action)##检查路网上是否还有车辆
                turncated = self.truncated
                ####
                other_state = self.get_system_info()
                self.metrics.append(other_state)
                ####
            else:
                print("车辆在最后一步离开离开仿真区域")
                reward = 0.0  
                next_state = np.zeros(self.observation_space.shape)  # 状态设为初始状态或其他合适值
                turncated = True
            
        terminated = self.check_terminated()
        if self.should_terminate:
            info = {"message": "True"}
        else:
            # info = self.compute_info()
            info = {}
        return (next_state, reward, terminated, turncated, info)
    
    def compute_info(self,reward):
        info = {"step":self.sim_step}
        info.update(self.get_system_info(reward))
        self.metrics.append(info.copy())
        return info
    
    def get_system_info(self):
        vehicle = self.vehicle
        sim_time = self.sim_step
        speed = self.speed_x
        acc = self.acc_rl
        collision = self.collision_count
        # print("info_acc is ==========>>",acc)
        waiting_times =traci.vehicle.getWaitingTime(vehicle)
        # reward = self.compute_reward()
        jeck = self.compute_jerk()
        return {
            "sim_time": sim_time,
            "agent_id": vehicle,
            "agent_speed": speed,
            "agent_waiting": waiting_times,
            "agent_acc": acc,
            "agent_jeck":jeck,
            "agent_collision": collision
        }
    
    def get_safe_info(self,diatance,speed):
        min_acc = abs(self.action_space.low[0])
        D = 10
        x = pow(speed , 2) / (2 *min_acc)
        if diatance < (D + x):
            self.collision_count += 1

    ###加速度积分###
    def compute_jerk(self):
        return (self.acc_history[1] - self.acc_history[0])/self.step_length
    
    def arrive_reward(self):#检查车辆是否到达目的地，并给予奖励
        route = traci.vehicle.getRoute(self.vehicle)
        last_lane = route[-1]
        current_edge = traci.vehicle.getRoadID(self.vehicle)
        last_edge_lane = last_lane + "_0"  # 假定索引为0的车道

        if current_edge == last_lane and traci.vehicle.getLanePosition(self.vehicle) >= (3 * traci.lane.getLength(last_edge_lane) ) / 4 :
            print("#######  到达目的地2   ############ ")
            return True
        else:
            return False

    
    def reward(self):
        idle_speed = 0.1
        
        # 安全奖励函数
        L_width = 3.2
        L_lateral = traci.vehicle.getLateralLanePosition(self.vehicle)
        R_lc = -pow(L_lateral / L_width, 2)  # 使用平方来增强对偏离中心的惩罚

        leader = traci.vehicle.getLeader(self.vehicle)
        if leader is not None:
            leader_id, gap = leader
            leader_speed = traci.vehicle.getSpeed(leader_id)
            speed_diff = self.speed_x - leader_speed
            ttc = gap / speed_diff if speed_diff > 0 and gap > 0 else 10.0
        else:
            ttc = 10.0
        R_ttc = 1 / (1 + ttc)
        R_safe = 10 * (R_lc + R_ttc)
        
        # 效率奖励
        speed_diff = abs(self.speed_limit - self.speed_x)
        R_efficient = 40 * (1 - speed_diff / self.speed_limit)
        
        # 舒适度奖励
        alpha = 1
        R_comfort = -alpha * pow(self.compute_jerk(), 2)
        
        # 能耗奖励
        delta_distance = traci.vehicle.getDistance(self.vehicle) - self.prev_distance
        self.prev_distance = traci.vehicle.getDistance(self.vehicle)
        electricity_consumption = traci.vehicle.getElectricityConsumption(self.vehicle) * self.step_length
        R_energy = -15 * (electricity_consumption / delta_distance) if delta_distance > 0.01 else -50
        
        # 到达目的奖励
        arrive_flash = self.arrive_reward()
        R_arrive = 150 if arrive_flash else 0
        
        # 怠速惩罚
        R_idle_penalty = -120 * (idle_speed - self.speed_x) if self.speed_x < idle_speed else 0

        # 调整奖励的权重
        R_total = 0.3 * R_safe + 0.2 * R_efficient + 0.2 * R_comfort + 0.2 * R_energy + R_arrive + R_idle_penalty
        return R_total
    
    def render(self, mode='human'):
        if self.render_mode == 'human':
            # 对于'human'渲染模式, SUMO GUI自身即视为渲染器.
            # 不需要编写额外代码, 因为使用'sumo-gui'使得环境可视.
            pass  # 可选择性地实现, 比如调整视角、采集屏幕截图等.
        elif self.render_mode == 'rgb_array':
            # 实现截取当前视图作为pixel array返回 (此处需第三方工具支持).
            pass

    #结束仿真
    def close(self):
        traci.close()
