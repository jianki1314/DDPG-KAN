import gymnasium as gym
from gymnasium import spaces
# import gym
# from gym import spaces
import traci
import numpy as np
from collections import deque
import math
import os
import random

np.set_printoptions(suppress=True, precision=3)
##状态空间定义##
def creat_observation():
    state_space_list = ['pos_x', 'pos_y', 'speed_x', 'speed_y', 'lane_heading_difference', 'lane_distance']
    state_space_low = np.array([-80, -80, 0, 0, -180, -3.2],dtype=np.float32)
    state_space_high = np.array([80, 80, 14, 14,180, 3.2],dtype=np.float32)
    obs = spaces.Box(low=state_space_low,high=state_space_high,shape=(6,),dtype=np.float32)
    return obs

##动作空间定义##
def creat_action():
    action_space = spaces.Box(low=np.array([-3.0],dtype=np.float32), high=np.array([4.0],dtype=np.float32),shape=(1,),dtype=np.float32)
    return action_space

class SumoEnv(gym.Env):
    metadata = {"render_modes": ["", "human", "rgb_array"], "render_fps": 4}
    def __init__(self):
        super(SumoEnv, self).__init__()
        #车辆信息相关变量
        self.vehicle = "rl_0"#智能车辆id
        self.name = "rl_0"
        self.vehicleNum = 0 #车辆数量
        self.presence = False #车辆存在性判断
        self.step_length = 1
        self.pos = (0,0)
        self.posx = 0 #x轴位置
        self.posy = 0 #y轴位置
        self.speed_x = 0 #x轴速度
        self.speed_y = 0 #y轴速度
        self.lane_heading_difference = 0 #车道航向差
        self.curr_lane = '' #车辆当前车道名称
        self.lane_distance = 0 #车辆与车道中心线的距离
        self.throttle = 0 #加速度输入
        self.acc = 0 #车辆加速度
        self.acc_history = deque([0, 0], maxlen=2) #存储加速度值
        self.state_dim = 6 #状态维度
        self.angle = 0 #车辆的角度
        self.gui = False #仿真界面启动开关值
        self.terminated = False 
        self.truncated = False

        #动作空间和状态空间
        self.action_space =  creat_action()
        self.observation_space = creat_observation()
        # assert render_mode is None or render_mode in self.metadata['render_modes']
        # self.render_mode =render_mode
        # print(self.render_mode)

    ##仿真启动##
    def start(self,gui=True, vehicleNum=36, network_cfg="networks/sumoconfig.sumo.cfg", network_net="networks/rouabout.net.xml",  vtype1="rlAnget", vtype2="human"):
        self.gui = gui
        self.vehicleNum = vehicleNum
        self.sumocfg = network_cfg
        self.net_xml = network_net
        self.rlvType = vtype1
        self.humanvType = vtype2

        #starting sumo
        #home = os.getenv("HOME") or os.getenv("USERPROFILE")  # 兼容Windows环境
        home = "D:/sumo"
        print("home is:",home)
        if self.gui:
            sumoBinary = os.path.join(home,"bin/sumo-gui")
        else:
            sumoBinary = os.path.join(home,"bin/sumo")
        #sumoCmd = [sumoBinary, "-c", self.sumocfg ,"--no-step-log", "true" , "-W"]
        sumoCmd = [sumoBinary, "-c", self.sumocfg,"--lateral-resolution","3.2",
         "--start", "False", "--quit-on-end", "true","--no-warnings","True", "--no-step-log", "True"]
        traci.start(sumoCmd)
        print("Using SUMO binary:", sumoBinary)
        
        self.generate_vehicles(vehicleNum)#生成车辆
        traci.simulationStep()
        all_vehicles = traci.vehicle.getIDList()
        #print(all_vehicles)
        if self.vehicle not in all_vehicles:
            print(f"Vehicle '{self.vehicle}' was not added to simulation properly.")
        else:
            print(f"Vehicle '{self.vehicle}' was  added to simulation properly !!!")

        self.update_params()#更新车辆参数
    
    
    ######车辆生成#########
    '''
    def generate_vehicles(self):
        # 生成37个车辆，其中1个为智能控制车辆，36个为普通车辆
        for i in range(self.vehicleNum):
            veh_name = 'vehicle_' + str(i)  # 普通车辆名称
            route_id = 'route_' + str(i % 12)  # 通过取余的方式分配12条路线
            traci.vehicle.add(veh_name, routeID=route_id, typeID=self.humanvType, departLane='random', departSpeed='random')
                    
        traci.vehicle.add(self.vehicle, routeID='route_0', typeID=self.rlvType, departLane='random', departSpeed='random')
        for _ in range(22):
            traci.simulationStep()
    '''
    def generate_vehicles(self, vehicleNum):
        # 定义所有可能的路线ID列表
        possible_routes = ["route_0", "route_1", "route_2", "route_3","route_4", "route_5", "route_6", "route_7", "route_8", "route_9", "route_10", "route_11"]
        # 先为每辆车随机分配路径
        random_routes = [random.choice(possible_routes) for _ in range(vehicleNum)]

        for i in range(vehicleNum):
            veh_name = 'vehicle_' + str(i)  # 普通车辆名称
            route_id = random_routes[i]  # 为每辆车随机分配路线
            traci.vehicle.add(veh_name, routeID=route_id, typeID=self.humanvType, departLane='random', departSpeed='random')
                    
        # 下面是智能车辆的生成，注意确保路线是随机的
        smart_vehicle_route_id = random.choice(possible_routes)  # 智能车辆随机选择一个路线
        traci.vehicle.add(self.vehicle, routeID=smart_vehicle_route_id, typeID=self.rlvType, departLane='random', departSpeed='random')
        
        for _ in range(22):
            traci.simulationStep()

    #更新车辆参数
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

    #获取车辆状态      
    def get_state(self):
        #all_vehicle  =  traci.vehicle.getIDList()
        state = None
        states = []
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
    
    #执行动作
    def applyAction(self, action):
        # print("vehicle's speed is -------->>",self.speed_x)
        traci.vehicle.setAcceleration(self.name, action, self.step_length)
    ##根据reset返回额外值，这里打印一个训练回合开始##
    def _getInfo(self):
        return {"current_episode":0}  
    
    def reset(self, seed=None, options=None):
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass
        super().reset(seed=seed)
        print("Resetting environment with GUI !!!!!!!!! reset")
        self.start()
        self.terminated = False 
        self.truncated = False
        next_state= self.get_state()
        info1 = self._getInfo()
        return next_state, info1
    
    #检查车辆是否到达了它的路线终点
    def check_terminated(self):
        ##reset截断状态函数返回 检查rl车辆是否到达目的地##
        route = traci.vehicle.getRoute(self.vehicle)
        last_lane = route[-1]
        # print("last edge is ---->>",last_lane)
        current_edge = traci.vehicle.getRoadID(self.vehicle)
        # print("current_edge is ---->>",current_edge)
        last_edge_lane = last_lane + "_0"  # 假定索引为0的车道
        if current_edge == last_lane and traci.vehicle.getLanePosition(self.vehicle) > (traci.lane.getLength(last_edge_lane) / 2):
            # 车辆处于最后一段路且已经超过该路段的长度，可以认为到达目的地
            print("########---->>>> The vehicle has arrived ")
            traci.simulationStep()
            return True
        else:
            return False
        
    ##检查车辆仿真是否结束 或车辆是否怠速##
    def check_truncated(self,action):
        min_speed = 1e-10
        if traci.vehicle.getIDCount() == 0:
            return True
        if self.speed_x <= min_speed and action < 0:#检测车辆是否怠速
            return True
        else:
            return False

    def step(self,action):
        all_vehicle = traci.vehicle.getIDList()
        if self.vehicle not in all_vehicle:
            self.terminated = False 
            self.truncated = False
            info = {}
            # info['terminated'] = self.terminated
            info['turncated'] = self.truncated
            reward = 0.0
            next_state = np.zeros(self.observation_space.shape)
        else:
            # print("Action is ------------->",action)
            assert self.action_space.contains(action), f"Invalid action: {action}"
            #将action中的加速度值输入到sumo中
            self.applyAction(action)
            traci.simulationStep()
            self.update_params()#更新一下车辆参数
            reward = self.compute_reward()
            next_state = self.get_state()
            self.terminated = self.check_terminated()##检测智能车辆是否到达终点
            self.truncated = self.check_truncated(action)##检查路网上是否还有车辆
            info = {}  # 确保这是一个字典#,可以在info中添加其他有用的信息
            # info['terminated'] = self.terminated
            info['turncated'] = self.truncated
            if reward is None:
                reward = 0.0
        done = self.terminated or self.truncated
        done1 = self.truncated
        # result = (next_state, float(reward), done, done1, info)
        # print("Step return values:", result)  # 添加调试信息
        return (next_state, float(reward), done, done1,  info)
    
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
            #R_efficient = 1 - ( (self.speed_x - self.speed_limit) / (speed_max - self.speed_limit) )
            R_efficient = self.speed_x / speed_max

        #舒适度计算
        jerk = self.compute_jerk()
        R_comfort = 1 - jerk/4

        #速度惩罚，不要让车辆停在原地
        min_speed = 1e-5
        R_speed_penalty = 0 if self.speed_x > min_speed else -1.0

        # 加速度稳定性奖励
        R_acc_stability = -abs(self.acc)
        # 奖励靠近零的加速度，惩罚过大的加速度

        #能耗奖励计算
        E_consumed = float(traci.vehicle.getParameter(self.vehicle, "device.battery.totalEnergyConsumed"))
        E_max = float(traci.vehicle.getParameter(self.vehicle, "device.battery.maximumBatteryCapacity"))
        E_regenerated= float(traci.vehicle.getParameter(self.vehicle, "device.battery.totalEnergyRegenerated"))
        R_energy = 1 - ( (E_consumed - E_regenerated) / E_max ) 
        
        R_total = 0.2*R_safe + 0.25*R_efficient + 0.25*(R_comfort + R_acc_stability) + 0.30*R_energy + R_speed_penalty

        return R_total

    #结束仿真
    def close(self):
        traci.close()
