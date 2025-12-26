'''
获取专家数据集
'''
import gymnasium as gym
import xml.etree.ElementTree as ET
from typing import Optional
from gymnasium import spaces
import traci
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import os
import random
import h5py

np.set_printoptions(suppress=True, precision=3)

# 定义状态空间
def creat_observation():
    """
    定义状态空间，包含速度、加速度、前后车辆距离和速度差。
    """
    state_space_low = np.array([0, 0, -10, 0, -14, 0, -14], dtype=np.float32)
    state_space_high = np.array([14, 14, 10, 50, 14, 50, 14], dtype=np.float32)
    obs = spaces.Box(low=state_space_low, high=state_space_high, shape=(7,), dtype=np.float32)
    return obs

# 定义动作空间
def creat_action():
    """
    定义动作空间，包含加速度的范围。
    """
    action_space = spaces.Box(low=np.array([-3.0], dtype=np.float32), high=np.array([4.0], dtype=np.float32), shape=(1,), dtype=np.float32)
    return action_space

class SumoEnv(gym.Env):
    metadata = {"render_modes": ["", "human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, max_episodes: int = 3, flash_episode: bool = False) -> None:
        """
        初始化SUMO仿真环境，包括车辆信息、状态空间和动作空间的定义。
        """
        super(SumoEnv, self).__init__()
        self.vehicle = "rl_0"  # 智能车辆id
        self.step_length = 0
        self.speed_x = 0  # x轴速度
        self.speed_y = 0  # y轴速度
        self.front_vehicle_distacne = 0  # 前车距离
        self.front_vehicle_to_speed = 0  # 前车速度差
        self.rear_vehicle_distance = 0  # 后车距离
        self.rear_vehicle_to_speed = 0  # 后车速度差
        self.acc = 0  # 车辆加速度
        self.acc_rl = 0  # 存储算法的加速度值
        self.acc_history = deque([0, 0], maxlen=2)  # 存储加速度值
        self.prev_distance = 0  # 车辆上一步的距离
        self.terminated = False 
        self.truncated = False
        self.simulation_running = False  # 仿真是否开标志
        self.episode_test = 0  # 仿真回合设置
        self.episode_train = 0  # 仿真回合设置
        self.metrics = []  # 记录每回合的其他信息
        self.max_episodes = max_episodes  # 最大回合数设置
        self.flash_episode = flash_episode  # 是否开启回合数限制
        self.sim_max_time = 150  # 最大仿真时间
        self.collision_count = 0  # 碰撞警示累积数
        self.action_space = creat_action()
        self.observation_space = creat_observation()
        # 奖励值最小值和最大值
        self.min_values = {
            'R_safe': -0.25,
            'R_efficient': 0.0,
            'R_comfort': -18.75,
            'R_energy': -5,
            'R_arrive': -1
        }
        self.max_values = {
            'R_safe': 1,
            'R_efficient': 1,
            'R_comfort': 18.75,
            'R_energy': 0,
            'R_arrive': 10
        }

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        print(self.render_mode)

    def reset(self, seed: Optional[int] = None):
        """
        重置环境，初始化车辆状态和参数，并根据模式选择进行仿真测试或训练。
        """
        if seed is not None:
            super().reset(seed=seed)
        print("Resetting environment with GUI !!!!!!!!! reset")
        self.terminated = False
        self.truncated = False
        self.prev_distance = 0  # 车辆上一步的距离
        self.collision_count = 0 # 碰撞警示累积数
        self.speed_x = 0  # x轴速度
        self.speed_y = 0  # y轴速度
        self.acc_rl = 0   # 存储算法的加速度值到csv文件 
        self.acc = 0
        self.acc_history = deque([0, 0], maxlen=2)  # 存储加速度值
        info1 = {}

        # 仿真测试模式
        if self.flash_episode:
            if self.episode_test < self.max_episodes:
                if self.episode_test != 0 and self.simulation_running:
                    self.save_csv(self.episode_test - 1) # 保存数据到CSV文件
                    self.close()
                self.modify_output_prefix(f'{self.episode_test}-')
                print("episode================================================>>>   测试", self.episode_test)
                self.episode_test += 1
                self.metrics = []  # 重置metrics列表
                self.simulation_running = True  # 仿真运行中
                self.start()
                next_state = self.get_state()
                info1 = self._getInfo()     
            else:
                print("$$$$$$$$$$$  已到达最大回合数  ￥￥￥￥￥￥￥￥￥")
                self.terminated = True
                next_state = np.zeros(self.observation_space.shape)
                info1 = {"message": "True"}
        else:
        # 训练模型模式
            if self.episode_train != 0 and self.simulation_running:
                self.close()
            self.episode_train += 1
            print("episode================================================>>>  训练 ", self.episode_train)
            self.metrics = [] # 重置metrics列表
            self.simulation_running = True  # 仿真运行中
            self.start()
            next_state = self.get_state()
            info1 = self._getInfo()
        return next_state, info1

    def modify_output_prefix(self, new_prefix):
        """
        修改SUMO配置文件中的输出前缀。
        """
        cfg_file = "../networks/sumoconfig_EIDM.sumo.cfg"
        tree = ET.parse(cfg_file)
        root = tree.getroot()
        output_prefix = root.find('.//output-prefix')
        if output_prefix is not None:
            output_prefix.set('value', new_prefix)
        tree.write(cfg_file)

    def save_csv(self, episode):
        """
        保存当前回合的仿真数据到CSV文件。
        """
        base_path = "../networks/output_data/output_eidm/state"
        file_name = f"{episode}-state.csv"
        full_path = Path(base_path) / file_name
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.metrics)
        df.to_csv(full_path, index=False)
        print(f"#########Data saved to {full_path}########")

    def start(self, gui=True, vehicleNum=80, network_cfg="../networks/sumoconfig_EIDM.sumo.cfg", network_net="networks/rouabout.net.xml", vtype1="rlAnget", vtype2="human"):
        """
        启动SUMO仿真环境并生成车辆。
        """
        self.gui = gui
        self.vehicleNum = vehicleNum
        self.sumocfg = network_cfg
        self.net_xml = network_net
        self.rlvType = vtype1
        self.humanvType = vtype2

        home = os.environ.get("SUMO_HOME")
        sumoBinary = os.path.join(home, "bin/sumo-gui") if self.render_mode == "human" else os.path.join(home, "bin/sumo")
        sumoCmd = [sumoBinary, "-c", self.sumocfg, "--lateral-resolution", "3.2", "--start", "True", "--quit-on-end", "True", "--no-warnings", "True", "--no-step-log", "True", "--step-method.ballistic", "true", "--delay", "0"]
        traci.start(sumoCmd)
        self.step_length = traci.simulation.getDeltaT()  # 获取仿真单位时间步
        self.generate_vehicles(self.vehicleNum)  # 生成车辆
        while self.vehicle not in traci.vehicle.getIDList():
            traci.simulationStep()
        print(f"Vehicle '{self.vehicle}' was added to simulation properly !!!")
        self.update_params()  # 更新车辆参数

    def generate_vehicles(self, vehicleNum):
        """
        生成普通和智能车辆，并为每辆车随机分配路径和发车时间。
        """
        possible_routes = ["route_0", "route_1", "route_2", "route_3", "route_4", "route_5", "route_6", "route_7", "route_8", "route_9", "route_10", "route_11"]
        random_routes = [random.choice(possible_routes) for _ in range(vehicleNum)]

        for i in range(vehicleNum):
            veh_name = f'vehicle_{i}'  # 普通车辆名称
            route_id = random_routes[i]  # 为每辆车随机分配路线
            depart_time = round(random.uniform(0, 30) / 0.4) * 0.4  # 在前30秒内随机发车，确保发车时间是时间步长的整数倍
            traci.vehicle.add(veh_name, routeID=route_id, typeID=self.humanvType, departLane='free', departSpeed='random', depart=depart_time)
        
        # 智能车辆生成
        # smart_vehicle_route_id = possible_routes  # 智能车辆随机选择一个路线
        smart_vehicle_route_id = random.choice(possible_routes)
        smart_vehicle_depart_time = round(random.uniform(0, 8) / 0.4) * 0.4  # 智能车辆在0到x秒内随机发车
        traci.vehicle.add(self.vehicle, routeID=smart_vehicle_route_id, typeID=self.rlvType, departLane='random', departSpeed='random', depart=smart_vehicle_depart_time)

    def update_params(self):
        """
        更新车辆的速度、加速度、前后车距离及速度差等参数。
        """
        self.speed_x = traci.vehicle.getSpeed(self.vehicle)
        self.speed_y = traci.vehicle.getLateralSpeed(self.vehicle)
        self.acc = traci.vehicle.getAcceleration(self.vehicle)
        self.acc_history.append(self.acc)
        self.curr_lane = traci.vehicle.getLaneID(self.vehicle)
        self.speed_limit = traci.lane.getMaxSpeed(self.curr_lane) if self.curr_lane else 14

        # 获取前车信息
        leader = traci.vehicle.getLeader(self.vehicle, dist=50)
        if leader:
            leader_id, leader_distance = leader
            if leader_id:
                leader_speed = traci.vehicle.getSpeed(leader_id)
                self.front_vehicle_distacne = leader_distance
                self.front_vehicle_to_speed = leader_speed - self.speed_x
                self.get_safe_info(leader_distance, self.speed_x)
            else:
                self.front_vehicle_distacne = 50
                self.front_vehicle_to_speed = 0
        else:
            self.front_vehicle_distacne = 50
            self.front_vehicle_to_speed = 0

        # 获取后车信息
        follower = traci.vehicle.getFollower(self.vehicle, dist=50)
        if follower:
            follower_id, follower_distance = follower
            if follower_id:
                follower_speed = traci.vehicle.getSpeed(follower_id)
                self.rear_vehicle_distance = follower_distance
                self.rear_vehicle_to_speed = follower_speed - self.speed_limit
            else:
                self.rear_vehicle_distance = 50
                self.rear_vehicle_to_speed = 0
        else:
            self.rear_vehicle_distance = 50
            self.rear_vehicle_to_speed = 0

    def get_state(self):
        """
        获取当前车辆的状态，包括速度、加速度、前后车距离及速度差，并进行归一化处理。
        """
        if self.vehicle not in traci.vehicle.getIDList():
            # print(f"Vehicle {self.vehicle} is not known get.")
            self.terminated = True  # 设定终止标志
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self.update_params()
            system_state = self.get_system_info() # 获取车辆的系统信息
            self.metrics.append(system_state)   # 记录每回合的系统信息
            state = np.array([
                self.speed_x,
                self.speed_y,
                self.acc,
                self.front_vehicle_distacne,
                self.front_vehicle_to_speed,
                self.rear_vehicle_distance,
                self.rear_vehicle_to_speed
            ], dtype=np.float32)
            state_space_low = np.array([0, 0, -10, 0, -14, 0, -14], dtype=np.float32)
            state_space_high = np.array([14, 14, 10, 50, 14, 50, 14], dtype=np.float32)
            normalized_state = 2 * ((state - state_space_low) / (state_space_high - state_space_low)) - 1
            action = traci.vehicle.getAcceleration(self.vehicle)
            action_space_low = -10
            action_space_high = 10
            normalized_action = 2 * ((action - action_space_low) / (action_space_high - action_space_low)) - 1
            self.store_data_to_hdf5(normalized_state, normalized_action)
            return normalized_state
        
    def store_data_to_hdf5(self, state, action):
        """
        将状态和动作存储到HDF5文件中，每个回合存储一个文件。
        """
        episode_num = self.episode_train if not self.flash_episode else self.episode_test
        base_path = "../KAN_data/hdf5"
        file_name = f"{episode_num}_episode.h5"
        full_path = os.path.join(base_path, file_name)
        os.makedirs(base_path, exist_ok=True)

        with h5py.File(full_path, 'a') as f:
            if 'states' not in f:
                f.create_dataset('states', data=[state], maxshape=(None, state.shape[0]), chunks=True, compression="gzip")
            else:
                f['states'].resize((f['states'].shape[0] + 1), axis=0)
                f['states'][-1] = state
            
            if 'actions' not in f:
                f.create_dataset('actions', data=[action], maxshape=(None,), chunks=True, compression="gzip")
            else:
                f['actions'].resize((f['actions'].shape[0] + 1), axis=0)
                f['actions'][-1] = action


    def applyAction(self, action):
        """
        根据给定的动作调整车辆的速度。
        """
        if self.vehicle not in traci.vehicle.getIDList():
            print(f"Vehicle {self.vehicle} is not known !!!!.")
            self.terminated = True  # 终止标志
        else:
            if not self.action_space.contains(action):
                traci.vehicle.setSpeed(self.vehicle, speed=-1)
            else:
                action = float(action)
                self.acc_rl = action
                v_current = traci.vehicle.getSpeed(self.vehicle)
                v_next = max(min(self.speed_limit, v_current + action), 0)
                if v_current <= 0.1:
                    traci.vehicle.setSpeed(self.vehicle, speed=-1)
                else:
                    traci.vehicle.setSpeed(self.vehicle, speed=v_next)
                traci.simulationStep()

    def _getInfo(self):
        """
        返回当前回合的信息。
        """
        return {"current_episode": 0}

    @property
    def sim_step(self) -> float:
        """
        返回仿真时间。
        """
        return traci.simulation.getTime()

    def step(self, action):
        """
        执行动作并返回下一个状态、奖励、是否终止和裁剪标志，以及额外信息。
        """
        truncated = False
        all_vehicle = traci.vehicle.getIDList()
        if self.vehicle not in all_vehicle:
            truncated = (self.sim_step >= self.sim_max_time)
            self.terminated = True
            reward = 0.0
            next_state = np.zeros(self.observation_space.shape)
        else:
            assert self.action_space.contains(action), f"Invalid action: {action}"
            self.applyAction(action)
            if self.terminated:
                truncated = True
                reward = 0.0
                next_state = np.zeros(self.observation_space.shape)
            else:
                reward = self.reward()
                next_state = self.get_state()
                system_state = self.get_system_info() # 获取车辆的系统信息
                self.metrics.append(system_state)   # 记录每回合的系统信息
                truncated = (self.sim_step >= self.sim_max_time)
                self.terminated = (not traci.simulation.getMinExpectedNumber())
        info = {"message": "True"} if self.simulation_running else {}
        return next_state, reward, self.terminated, truncated, info
    
    def check_terminated(self):
        """
        检查车辆是否终止。
        """
        truncated = (self.sim_step >= self.sim_max_time)
        terminated = (not traci.simulation.getMinExpectedNumber())
        
        return terminated or truncated

    def compute_info(self):
        """
        计算并返回当前仿真步骤的信息。
        """
        info = {"step": self.sim_step}
        info.update(self.get_system_info())
        self.metrics.append(info.copy())
        return info

    def get_system_info(self):
        """
        获取车辆的系统信息，包括速度、加速度、碰撞次数等。
        """
        vehicle = self.vehicle
        sim_time = self.sim_step
        speed = self.speed_x
        acc = self.acc
        collision = self.collision_count
        jerk = self.compute_jerk()
        return {
            "sim_time": sim_time,
            "agent_id": vehicle,
            "agent_speed": speed,
            "agent_acc": acc,
            "agent_jeck": jerk,
            "agent_collision": collision
        }

    def get_safe_info(self, distance, speed):
        """
        根据距离和速度更新碰撞警示累积数。
        """
        min_acc = abs(self.action_space.low[0])
        D = 10
        x = pow(speed, 2) / (2 * min_acc)
        if distance < (D + x):
            self.collision_count += 1

    def compute_jerk(self):
        """
        计算并返回车辆的加速度变化率jerk。
        """
        return (self.acc_history[1] - self.acc_history[0]) / self.step_length

    def arrive_reward(self):
        """
        判断车辆是否到达目的地，并返回相应的奖励。
        """
        route = traci.vehicle.getRoute(self.vehicle)
        last_lane = route[-1]
        current_edge = traci.vehicle.getRoadID(self.vehicle)
        last_edge_lane = last_lane + "_0"
        if current_edge == last_lane and traci.vehicle.getLanePosition(self.vehicle) >= (7 * traci.lane.getLength(last_edge_lane)) / 8:
            print("#######  到达目的地   ############ ")
            R_arrive = 10
            return R_arrive 
        else:
            R_arrive = -1
            return R_arrive 

    def calculate_safety_reward(self):
        """
        计算并返回安全奖励。
        """
        L_width = 3.2
        L_lateral = traci.vehicle.getLateralLanePosition(self.vehicle)
        R_lc = -pow(L_lateral / L_width, 2)

        leader = traci.vehicle.getLeader(self.vehicle)
        if leader is not None:
            leader_id, gap = leader
            leader_speed = traci.vehicle.getSpeed(leader_id)
            speed_diff = self.speed_x - leader_speed
            ttc = gap / speed_diff if speed_diff > 0 and gap > 0 else 10.0
        else:
            ttc = 10.0
        R_ttc = 1 / (1 + ttc)
        R_safe = (R_lc + R_ttc)
        return R_safe

    def calculate_efficiency_reward(self):
        """
        计算并返回效率奖励。
        """
        speed_diff = abs(14 - self.speed_x)
        R_efficient = (1 - speed_diff / 14)
        return R_efficient

    def calculate_comfort_reward(self):
        """
        计算并返回舒适度奖励。
        """
        R_comfort = -self.compute_jerk()
        return R_comfort

    def calculate_energy_reward(self):
        """
        计算并返回能耗奖励。
        """
        delta_distance = traci.vehicle.getDistance(self.vehicle) - self.prev_distance
        self.prev_distance = traci.vehicle.getDistance(self.vehicle)
        electricity_consumption = traci.vehicle.getElectricityConsumption(self.vehicle) * self.step_length
        # 确保行驶距离差足够大以避免数值不稳定性
        if delta_distance > 0.1:
            # 如果电能消耗为负值，表示能量回收，奖励应为正值
            if electricity_consumption < 0:
                R_energy = 0
            else:
                R_energy = -(electricity_consumption / delta_distance)
        else:
            R_energy = -5  # 固定的负奖励，表示高能耗或低效率行驶

        return R_energy
    
    
    def normalize(self, value, min_value, max_value):
        """
        对奖励值进行归一化处理。
        """
        # return (value - min_value) / (max_value - min_value)  # [0,1]
        return 2 * ((value - min_value) / (max_value - min_value)) - 1 # [-1,1]

    def reward(self):
        """
        计算并返回总奖励。
        """
        if self.vehicle not in traci.vehicle.getIDList():
            print(f"Vehicle {self.vehicle} is not known reward.")
            return 0.0

        try:
            R_safe = self.calculate_safety_reward()
            R_efficient = self.calculate_efficiency_reward()
            R_comfort = self.calculate_comfort_reward()
            R_energy = self.calculate_energy_reward()
            R_arrive = self.arrive_reward()
            # 归一化
            R_safe = self.normalize(R_safe, self.min_values['R_safe'], self.max_values['R_safe'])
            R_efficient = self.normalize(R_efficient, self.min_values['R_efficient'], self.max_values['R_efficient'])
            R_comfort = self.normalize(R_comfort, self.min_values['R_comfort'], self.max_values['R_comfort'])
            R_energy = self.normalize(R_energy, self.min_values['R_energy'], self.max_values['R_energy'])
            R_arrive = self.normalize(R_arrive, self.min_values['R_arrive'], self.max_values['R_arrive'])
            # self.log_rewards(R_safe, R_efficient, R_comfort, R_energy)
            # print(f"R_safe: {R_safe}, R_efficient: {R_efficient}, R_comfort: {R_comfort}, R_energy: {R_energy}")
                    # 定义权重
            weights = {
                'R_safe': 1,
                'R_efficient': 2.5,
                'R_comfort': 2,
                'R_energy': 2.5,
                'R_arrive': 2
            }

            # 计算加权总奖励
            R_total = (
                weights['R_safe'] * R_safe +
                weights['R_efficient'] * R_efficient +
                weights['R_comfort'] * R_comfort +
                weights['R_energy'] * R_energy +
                weights['R_arrive'] * R_arrive
            )
            R_total = self.normalize(R_total, -10, 10)  # 根据实际奖励范围调整
            assert not np.isnan(R_total), "Reward contains NaN values"
            return R_total
        except Exception as e:
            print(f"Error calculating reward: {e}")
        return 0.0
    
    def render(self, mode='human'):
        """
        渲染仿真环境。
        """
        if self.render_mode == 'human':
            # 对于'human'渲染模式, SUMO GUI自身即视为渲染器.
            # 不需要编写额外代码, 因为使用'sumo-gui'使得环境可视.
            pass  # 可选择性地实现, 比如调整视角、采集屏幕截图等.
        elif self.render_mode == 'rgb_array':
            # 实现截取当前视图作为pixel array返回 (此处需第三方工具支持).
            pass

    def close(self):
        """
        关闭仿真环境。
        """
        if self.simulation_running:
            try:
                traci.close()
            except traci.FatalTraCIError:
                pass
        self.simulation_running = False