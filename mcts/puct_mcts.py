from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node:MCTSNode):
       # select the best action based on PUCB when expanding the tree
        
        ########################
        # TODO: your code here #
        ######################## 
        assert not node.done
        # 计算当前节点所有子节点的访问总次数
        total_visits = np.sum(node.child_N_visit)
        # 选择未被发现的动作
        undiscovered_mask = (node.child_N_visit == 0) & (node.action_mask == 1)
        undiscovered_indices = np.flatnonzero(undiscovered_mask)
        if len(undiscovered_indices) > 0:
            return np.random.choice(undiscovered_indices)
        # 计算PUCB值
        exploitation = np.zeros(node.n_action)
        mask = node.child_N_visit > 0
        exploitation[mask] = node.child_V_total[mask] / node.child_N_visit[mask]
        pucb_scores = exploitation + self.config.C * node.child_priors * np.sqrt(total_visits) / (1 + total_visits)
        pucb_scores = np.where(node.action_mask == 1, pucb_scores, -INF)
        # 返回最佳动作
        return np.argmax(pucb_scores)
        ########################

    def backup(self, node:MCTSNode, value):
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        
        ########################
        # TODO: your code here #
        ########################

        current = node
        while current.parent is not None:
            action = current.action
            parent = current.parent
            parent.child_N_visit[action] += 1
            parent.child_V_total[action] += value
            # 切换视角
            value = -value
            current = parent 
        
        ########################   
    
    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        
        ########################
        # TODO: your code here #
        ########################
        
        node = self.root
        while True:
            if node.done:
                return node
            
            legal_actions = np.where(node.action_mask)[0]
            unexpanded = [a for a in legal_actions if not node.has_child(a)]
            if unexpanded:
                # 扩展第一个未探索动作
                action = unexpanded[0]
                return node.add_child(action)
            else:
                # 通过PUCT选择最佳动作
                action = self.puct_action_select(node)
                node = node.get_child(action)

        ########################
    
    def get_policy(self, node:MCTSNode = None):
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
        ########################
        # TODO: your code here #
        ########################
        
        if node is None:
            node = self.root

        policy = np.zeros(node.n_action, dtype=np.float32)
        legal_actions = np.where(node.action_mask)[0]
        visits = node.child_N_visit[legal_actions]
        total_visits = visits.sum()

        if total_visits == 0:
            # 均匀策略
            policy[legal_actions] = 1.0 / len(legal_actions)
        else:
            # 温度参数平滑策略
            temp = self.config.temperature
            if temp > 0:
                scaled_visits = (visits / (total_visits)) ** (1.0 / temp)
                policy[legal_actions] = scaled_visits / scaled_visits.sum()
            else:
                policy[legal_actions] = visits / total_visits

        return policy
    
        ########################

    def search(self):
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                ########################
                # TODO: your code here #
                ########################

                # 直接使用节点的奖励
                value = leaf.reward

                ########################
            else:
                ########################
                # TODO: your code here #
                ########################
                # NOTE: you should compute the policy and value 
                #       using the value&policy model!

                # Use the value model to predict the value
                obs = leaf.env.compute_canonical_form_obs(leaf.env.observation, leaf.env.current_player)
                _, value = self.model.predict(obs)
                # Expand the leaf node by setting its prior
                child_prior, _ = self.model.predict(obs)
                leaf.set_prior(child_prior)

                ########################
            self.backup(leaf, value)
            
        return self.get_policy(self.root)