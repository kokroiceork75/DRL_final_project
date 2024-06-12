#!/usr/bin/env python

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        # 設定模型路徑，替換為模型檔案的實際路徑
        self.modelPath = self.modelPath.replace('/home/user/drl_ws/src/PPO-SAC-DQN-DDPG/PPO',
                                                '/home/user/drl_ws/src/PPO-SAC-DQN-DDPG/PPO/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()  # 讀取模型檔案
        self.stage = 2  # 設定階段
        self.goal_position = Pose()  # 初始化目標位置
        self.init_goal_x = 0.6  # 初始目標位置 X 座標
        self.init_goal_y = 0.0  # 初始目標位置 Y 座標
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'  # 模型名稱
        # 障礙物位置
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)  # ROS 訂閱模型狀態(透過topic)
        self.check_model = False  # 檢查模型是否存在
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":  # 如果模型名稱為 'goal'
                self.check_model = True  # 模型存在

    def respawnModel(self):
        while True:
            if not self.check_model:  # 如果模型不存在
                rospy.wait_for_service('gazebo/spawn_sdf_model')  # 等待服務可用
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)  # 創建服務代理
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")  # 生成模型
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:  # 如果模型存在
                rospy.wait_for_service('gazebo/delete_model')  # 等待服務可用
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)  # 創建服務代理
                del_model_prox(self.modelName)  # 刪除模型
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()  # 刪除現有模型

        if self.stage != 4:
            while position_check:  # 隨機生成目標點，避開障礙物
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
        else:
            while position_check:  # 在第四階段，從預設的目標點列表中選取
                goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

                self.index = random.randrange(0, 13)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        time.sleep(0.5)
        self.respawnModel()  # 生成新的目標點模型

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y  # 返回新的目標點座標
