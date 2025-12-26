from sumolib import checkBinary 
import os  
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from new_sumo_env import SumoEnv
from DDPG import DDPG
import copy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define hyperparameters
ALPHA = 0.0001  # Actor learning rate
BETA = 0.00015  # Critic learning rate
GAMMA = 0.99    # Discount factor
TAU = 0.005     # Soft update coefficient
ACTION_NOISE = 0.1  # Action noise
BATCH_SIZE = 64     # Batch size
MAX_SIZE = 100000   # Replay buffer size
CKPT_DIR = 'models/'  # Model save path

# Global normalization parameters for state (used in data collection)
STATE_SPACE_LOW = np.array([0, 0, -10, 0, -14, 0, -14], dtype=np.float32)
STATE_SPACE_HIGH = np.array([14, 14, 10, 50, 14, 50, 14], dtype=np.float32)

def denormalize_value(norm_val, low, high):
    """Convert normalized value back to original scale."""
    return ((norm_val + 1) / 2) * (high - low) + low

class DDPGDataCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        
    def collect(self, state, action, reward, next_state):
        if isinstance(state, np.ndarray):
            state = state.flatten()
        if isinstance(next_state, np.ndarray):
            next_state = next_state.flatten()
        
        self.states.append(state)
        self.actions.append(action.flatten())
        self.rewards.append(reward)
        self.next_states.append(next_state)
        
    def get_dataset(self):
        return (np.array(self.states), 
                np.array(self.actions), 
                np.array(self.rewards),
                np.array(self.next_states))

class InterpretableDDPG:
    def __init__(self, max_depth=3):  # Set tree max depth to 3
        # Use regression trees (squared error) for continuous action values
        self.state_decision_tree = DecisionTreeRegressor(max_depth=max_depth)
        self.transition_decision_tree = DecisionTreeRegressor(max_depth=max_depth)
        # If classification with Gini impurity is needed, one should discretize actions and use:
        # self.state_decision_tree = DecisionTreeClassifier(max_depth=max_depth, criterion='gini')
        # self.transition_decision_tree = DecisionTreeClassifier(max_depth=max_depth, criterion='gini')
        
        self.feature_names = None
        # Remove Chinese font settings; use default English fonts
        
        # Save normalization parameters for later reverse transformation
        self.state_space_low = STATE_SPACE_LOW
        self.state_space_high = STATE_SPACE_HIGH
        
    def train(self, states, actions, next_states, feature_names=None):
        """Train both decision trees with an 80:20 train-test split."""
        self.feature_names = feature_names  # Set feature names
        
        # Split data for state->action regression
        X_train, X_test, y_train, y_test = train_test_split(
            states, actions, test_size=0.2, random_state=42)
        self.state_decision_tree.fit(X_train, y_train)
        train_score = self.state_decision_tree.score(X_train, y_train)
        test_score = self.state_decision_tree.score(X_test, y_test)
        print(f"State->Action Tree: Training score: {train_score:.3f}, Test score: {test_score:.3f}")
        
        # Split data for state-action->next_state regression
        state_action_pairs = np.hstack([states, actions])
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
            state_action_pairs, next_states, test_size=0.2, random_state=42)
        self.transition_decision_tree.fit(X_train_t, y_train_t)
        train_score_t = self.transition_decision_tree.score(X_train_t, y_train_t)
        test_score_t = self.transition_decision_tree.score(X_test_t, y_test_t)
        print(f"State-Action->Next State Tree: Training score: {train_score_t:.3f}, Test score: {test_score_t:.3f}")
    
    def _denormalize_tree(self, tree_obj, is_transition=False):
        """
        Create a deep copy of the tree object and reverse the normalization of thresholds.
        For state-action tree, only the first 7 features (state features) are denormalized.
        """
        tree_copy = copy.deepcopy(tree_obj)
        n_nodes = tree_copy.tree_.node_count
        for i in range(n_nodes):
            # Skip leaf nodes: feature == -2
            if tree_copy.tree_.feature[i] != -2:
                feat_index = tree_copy.tree_.feature[i]
                norm_thresh = tree_copy.tree_.threshold[i]
                # For transition tree, the last feature (index 7) is action, so skip denormalization for it.
                if is_transition and feat_index == 7:
                    continue
                # Denormalize based on corresponding low and high values.
                orig_thresh = denormalize_value(norm_thresh, 
                                                self.state_space_low[feat_index],
                                                self.state_space_high[feat_index])
                tree_copy.tree_.threshold[i] = orig_thresh
        return tree_copy

    def visualize(self, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
            
        # Create denormalized copies for visualization
        state_tree_plot = self._denormalize_tree(self.state_decision_tree, is_transition=False)
        combined_features = self.feature_names + ['Action']
        transition_tree_plot = self._denormalize_tree(self.transition_decision_tree, is_transition=True)
        
        # Plot State->Action Decision Tree in English
        plt.figure(figsize=(40, 20), dpi=300)
        plot_tree(state_tree_plot, 
                  feature_names=self.feature_names,
                  filled=True,
                  rounded=True)
        plt.title('State to Action Decision Tree')
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('state_to_action_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot State-Action->Next State Decision Tree in English
        plt.figure(figsize=(40, 20), dpi=300)
        plot_tree(transition_tree_plot,
                  feature_names=combined_features,
                  filled=True,
                  rounded=True)
        plt.title('State-Action to Next State Decision Tree')
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('state_action_to_next_state_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def explain_decision(self, state):
        """Explain the decision for a given state (input remains normalized)."""
        if isinstance(state, np.ndarray):
            state = state.flatten()
        
        if self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(len(state))]
            
        prediction = self.state_decision_tree.predict([state])[0]
        feature_importance = self.state_decision_tree.feature_importances_
        
        explanation = {
            'prediction': prediction,
            'feature_importance': dict(zip(self.feature_names, feature_importance))
        }
        return explanation
    
    def plot_feature_importance(self):
        """Plot bar charts for feature importance of both trees."""
        if self.feature_names is None:
            return
        
        # Feature importance for state->action tree
        importances = self.state_decision_tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importance Analysis for State to Action')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                   [self.feature_names[i] for i in indices], 
                   rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('state_to_action_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance for state-action->next state tree
        combined_features = self.feature_names + ['Action']
        importances_t = self.transition_decision_tree.feature_importances_
        indices_t = np.argsort(importances_t)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importance Analysis for State-Action to Next State')
        plt.bar(range(len(importances_t)), importances_t[indices_t])
        plt.xticks(range(len(importances_t)), 
                   [combined_features[i] for i in indices_t], 
                   rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('state_action_to_next_state_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
def extract_rules(tree, feature_names, target_label="Action"):
    """Extract human-readable rules from a decision tree.
       For multi-output targets (e.g. next state vector), the leaf prediction will be printed as a vector.
    """
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value
    
    rules = []
    
    def recurse(node, path):
        if feature[node] != -2:  # not a leaf node
            name = feature_names[feature[node]]
            thresh = threshold[node]
            left_path = path + [f"{name} <= {thresh:.2f}"]
            recurse(children_left[node], left_path)
            right_path = path + [f"{name} > {thresh:.2f}"]
            recurse(children_right[node], right_path)
        else:
            pred = values[node].flatten()
            # 如果预测值为多维向量，则格式化为列表字符串；否则按标量格式输出
            if pred.size == 1:
                pred_str = f"{pred[0]:.3f}"
            else:
                pred_str = "[" + ", ".join(f"{x:.3f}" for x in pred) + "]"
            rules.append(f"{' AND '.join(path)} THEN {target_label} = {pred_str}")
    
    recurse(0, [])
    return rules

def main(num_episodes):
    print("Starting testing...")
    env = SumoEnv(render_mode="rgb_array", max_episodes=num_episodes, flash_episode=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(alpha=ALPHA, beta=BETA, state_dim=state_dim, action_dim=action_dim, ckpt_dir=CKPT_DIR,
                 observation_space=env.observation_space, action_space=env.action_space)
    agent.load_models(666)

    # Data collector
    collector = DDPGDataCollector()
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, train=False)
            next_state, rewards, terminated, truncated, info = env.step(action)
            collector.collect(state, action, rewards, next_state)
            done = terminated or truncated
            state = next_state
        print(f"Episode {episode} ended.")
    env.close()
    
    # Retrieve collected data
    states, actions, rewards, next_states = collector.get_dataset()
    
    # Define feature names (adjust if necessary)
    feature_names = [
        'speed_X',           # x-axis speed
        'speed_y',           # y-axis speed
        'acc',               # vehicle acceleration
        'front_vehicle_distance',  # front vehicle distance
        'front_vehicle_to_speed',  # front vehicle speed difference
        'rear_vehicle_distance',   # rear vehicle distance
        'rear_vehicle_to_speed'    # rear vehicle speed difference
    ]
    
    # Create and train the interpretable model
    interpretable_model = InterpretableDDPG(max_depth=3)
    interpretable_model.train(states, actions, next_states, feature_names)
    
    # Visualize decision trees (with thresholds shown in true scale)
    interpretable_model.visualize(feature_names)
    
    # Plot feature importance
    print("\nFeature Importance Analysis:")
    interpretable_model.plot_feature_importance()
    total_steps = len(states)
    print(f"Total collected steps: {total_steps}")
    
    # Explain a sample decision
    example_state = states[0]
    explanation = interpretable_model.explain_decision(example_state)
    
    print("\nSample Decision Explanation:")
    print(f"State (normalized): {example_state}")
    print(f"Predicted Action: {explanation['prediction']:.3f}")
    print("\nFeature Importance:")
    for feature, importance in explanation['feature_importance'].items():
        print(f"{feature}: {importance:.3f}")
    
    # Extract and print rules for state->action tree
    print("\nState to Action Decision Rules:")
    rules = extract_rules(interpretable_model._denormalize_tree(interpretable_model.state_decision_tree), feature_names, target_label="Action")
    for i, rule in enumerate(rules, 1):
        print(f"Rule {i}: IF {rule}")

    # Extract and print rules for state-action->next state tree (target label改为 Next_State)
    combined_features = feature_names + ['Action']
    print("\nState-Action to Next State Decision Rules:")
    transition_rules = extract_rules(interpretable_model._denormalize_tree(interpretable_model.transition_decision_tree, is_transition=True), combined_features, target_label="Next_State")
    for i, rule in enumerate(transition_rules, 1):
        print(f"Rule {i}: IF {rule}")
    
    print('---------------- Analysis Completed ----------------')

if __name__ == "__main__":
    main(num_episodes=200)






'''
from sumolib import checkBinary
import os  
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import os
from matplotlib.font_manager import FontProperties
from new_sumo_env import SumoEnv
import matplotlib.pyplot as plt
from DDPG import DDPG
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 定义超参数
ALPHA = 0.0001  # Actor 学习率
BETA = 0.00015  # Critic 学习率
GAMMA = 0.99    # 折扣因子
TAU = 0.005     # 软更新系数
ACTION_NOISE = 0.1  # 动作噪声
BATCH_SIZE = 64  # 采样 batch 大小
MAX_SIZE = 100000  # 经验回放缓冲区大小
CKPT_DIR = 'models/'  # 模型保存路径
###此函数为  训练 +  测试  主函数### 最新的


class DDPGDataCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        
    def collect(self, state, action, reward, next_state):
        if isinstance(state, np.ndarray):
            state = state.flatten()
        if isinstance(next_state, np.ndarray):
            next_state = next_state.flatten()
        
        self.states.append(state)
        self.actions.append(action.flatten())
        self.rewards.append(reward)
        self.next_states.append(next_state)
        
    def get_dataset(self):
        return (np.array(self.states), 
                np.array(self.actions), 
                np.array(self.rewards),
                np.array(self.next_states))

class InterpretableDDPG:
    def __init__(self, max_depth=5):
        self.state_decision_tree = DecisionTreeRegressor(max_depth=max_depth)
        self.transition_decision_tree = DecisionTreeRegressor(max_depth=max_depth)
        self.feature_names = None
        # 设置中文字体
        self.font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def train(self, states, actions, next_states, feature_names=None):
        """训练两个决策树模型，并设置特征名称"""
        self.feature_names = feature_names  # 在train时设置feature_names
        self.state_decision_tree.fit(states, actions)
        state_action_pairs = np.hstack([states, actions])
        self.transition_decision_tree.fit(state_action_pairs, next_states)
    
    def visualize(self, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
            
        # 可视化状态到动作的决策树
        plt.figure(figsize=(40, 20), dpi=300)
        plot_tree(self.state_decision_tree, 
                 feature_names=self.feature_names,
                 filled=True,
                 rounded=True)
        plt.title('状态到动作的决策树', fontproperties=self.font)
        plt.subplots_adjust(bottom=0.2)  # 调整底部边距
        plt.savefig('state_to_action_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 可视化状态-动作到下一状态的决策树
        combined_features = self.feature_names + ['Action']
        plt.figure(figsize=(40, 20), dpi=300)
        plot_tree(self.transition_decision_tree,
                 feature_names=combined_features,
                 filled=True,
                 rounded=True)
                #  fontsize=12,
                #  proportion=True,  # 增加 proportion 参数
                #  precision=2,  # 增加 precision 参数
                #  impurity=False)  # 隐藏 impurity 信息
        plt.title('状态-动作到下一状态的决策树', fontproperties=self.font)
        plt.subplots_adjust(bottom=0.2)  # 调整底部边距
        plt.savefig('state_action_to_next_state_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def explain_decision(self, state):
        """解释特定状态下的决策"""
        if isinstance(state, np.ndarray):
            state = state.flatten()
        
        if self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(len(state))]
            
        prediction = self.state_decision_tree.predict([state])[0]
        feature_importance = self.state_decision_tree.feature_importances_
        
        explanation = {
            'prediction': prediction,
            'feature_importance': dict(zip(self.feature_names, feature_importance))
        }
        return explanation
    
    def plot_feature_importance(self):
        """绘制特征重要性条形图"""
        if self.feature_names is None:
            return
        
        # 获取状态到动作决策树的特征重要性
        importances = self.state_decision_tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('状态到动作的特征重要性分析', fontsize=14, fontproperties=self.font)
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                [self.feature_names[i] for i in indices], 
                rotation=45, 
                ha='right')
        plt.tight_layout()
        plt.savefig('state_to_action_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 获取状态-动作到下一状态决策树的特征重要性
        combined_features = self.feature_names + ['Action']
        importances = self.transition_decision_tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('状态-动作到下一状态的特征重要性分析', fontsize=14, fontproperties=self.font)
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                [combined_features[i] for i in indices], 
                rotation=45, 
                ha='right')
        plt.tight_layout()
        plt.savefig('state_action_to_next_state_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
def extract_rules(tree, feature_names):
    """从决策树中提取可理解的规则"""
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value
    
    rules = []
    
    def recurse(node, depth, path):
        if tree.tree_.feature[node] != -2:  # 不是叶节点
            name = feature_names[feature[node]]
            threshold_val = threshold[node]
            
            # 左子树的规则
            left_path = path + [f"{name} <= {threshold_val:.2f}"]
            recurse(children_left[node], depth + 1, left_path)
            
            # 右子树的规则
            right_path = path + [f"{name} > {threshold_val:.2f}"]
            recurse(children_right[node], depth + 1, right_path)
        else:
            # 到达叶节点，添加预测值
            prediction = values[node][0][0]
            rules.append(f"{' AND '.join(path)} THEN Action = {prediction:.3f}")
    
    recurse(0, 1, [])
    return rules

def main(num_episodes):
    print("Starting testing...")
    env = SumoEnv(render_mode="rgb_array", max_episodes=num_episodes, flash_episode=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(alpha=ALPHA, beta=BETA, state_dim=state_dim, action_dim=action_dim, ckpt_dir=CKPT_DIR,
                 observation_space=env.observation_space, action_space=env.action_space)
    agent.load_models(666)

    # 创建数据收集器
    collector = DDPGDataCollector()
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, train=False)
            next_state, rewards, terminated, truncated, info = env.step(action)
                    # 收集数据
            collector.collect(state, action, rewards, next_state)
            done = terminated or truncated
            state = next_state
        print(f"结束本回合，回合数为: {episode}")
    env.close()
    

    # 获取收集的数据
    states, actions, rewards, next_states = collector.get_dataset()
    

    
    # 定义特征名称（根据你的环境状态空间进行修改）
    feature_names = [
        'speed_X',# x轴速度
        'speed_y',# y轴速度
        'acc', # 车辆加速度
        'front_vehicle_distacne', # 前车距离
        'front_vehicle_to_speed', # 前车速度差
        'rear_vehicle_distance', # 后车距离
        'rear_vehicle_to_speed'# 后车速度差
        # 添加其他特征名称...
    ]
    # 创建和训练解释性模型
    interpretable_model = InterpretableDDPG(max_depth=5)
    interpretable_model.train(states, actions, next_states, feature_names)  
    
    # 可视化决策树
    interpretable_model.visualize()
    
    # 分析特征重要性
    print("\n特征重要性分析:")
    interpretable_model.plot_feature_importance()
    total_steps = len(states)
    print(f"总采样步数: {total_steps}")
    
    # 选择一个示例状态进行解释
    example_state = states[0]
    explanation = interpretable_model.explain_decision(example_state)
    
    print("\n决策示例:")
    print(f"状态: {example_state}")
    print(f"预测的动作: {explanation['prediction']:.3f}")
    print("\n特征重要性:")
    for feature, importance in explanation['feature_importance'].items():
        print(f"{feature}: {importance:.3f}")
    
    # 提取决策规则
    print("\n状态到动作的决策规则:")
    rules = extract_rules(interpretable_model.state_decision_tree, feature_names)
    for i, rule in enumerate(rules, 1):
        print(f"规则 {i}: IF {rule}")

    print("\n状态-动作到下一状态的决策规则:")
    combined_features = feature_names + ['Action']
    transition_rules = extract_rules(interpretable_model.transition_decision_tree, combined_features)
    for i, rule in enumerate(transition_rules, 1):
        print(f"规则 {i}: IF {rule}")
    
    print('----------------分析完成-----------------------')

if __name__ == "__main__":
    main(num_episodes=200)  

'''