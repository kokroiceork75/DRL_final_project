#! /usr/bin/python2.7
# coding:utf-8

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True  # 初始化目標點
        self.get_goalbox = False  # 是否到達目標點
        self.position = Pose()
        self.obstacle_min_range = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)  # 發布速度指令
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)  # 訂閱里程計數據
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)  # 重置仿真環境
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)  # 暫停仿真物理
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)  # 繼續仿真物理
        self.respawn_goal = Respawn()  # 初始化目標點生成類

    def getGoalDistace(self):
        # 計算目標點與當前機器人位置的距離
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance

    def getOdometry(self, odom):
        # 獲取里程計數據
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.1  # 碰撞距離
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)  # 獲取最小的雷達信息
        self.obstacle_min_range = obstacle_min_range
        obstacle_angle = np.argmin(scan_range)  # 獲取最小值的角度

        if obstacle_min_range < 0.15:
            done = True  # 碰撞

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)  # 計算機器人到目標點的距離
        if current_distance < 0.2:  # 如果距離小於0.2，表示到達目標點
            self.get_goalbox = True

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done  # 返回狀態

    def setReward(self, state, done, action):
        yaw_reward = []  # 設置角度獎勵
        obstacle_min_range = state[-2]  # 獲取雷達信息中的最小值
        self.obstacle_min_range = obstacle_min_range
        current_distance = state[-3]  # 獲取當前距離
        heading = state[-4]  # 獲取機器人朝向角

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2  # 計算角度
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])  # 計算角度獎勵
            yaw_reward.append(tr)

        if obstacle_min_range <= 0.2:
            scan_reward = -1 / (obstacle_min_range + 0.3)  # 設置激光雷達獎勵，範圍在-3.33到-2.5之間
        else:
            scan_reward = 2

        distance_rate = 2 ** (current_distance / self.goal_distance)  # 計算距離比率

        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + scan_reward  # 總獎勵

        if done:
            rospy.loginfo("Collision!!")
            reward = -500 + scan_reward  # 碰撞懲罰
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)  # 重新生成目標點
            self.pub_cmd_vel.publish(Twist())  # 停止機器人運動

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000 + scan_reward  # 到達目標點獎勵
            self.pub_cmd_vel.publish(Twist())  # 停止機器人運動
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)  # 重新生成目標點
            self.goal_distance = self.getGoalDistace()  # 更新目標距離
            self.get_goalbox = False  # 重置目標點標誌

        return reward

    def step(self, action):
        max_angular_vel = 1.5  # 最大角速度
        ang_vel = ((self.action_size - 1) / 2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.angular.z = ang_vel
        vel_cmd.linear.x = 0.2  # 設置線速度

        self.pub_cmd_vel.publish(vel_cmd)  # 發布速度指令

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)  # 獲取雷達數據
            except:
                pass

        state, done = self.getState(data)  # 獲取當前狀態
        reward = self.setReward(state, done, action)  # 設置獎勵

        return np.asarray(state), reward, done  # 返回狀態、獎勵和是否完成標誌

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()  # 重置仿真環境
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)  # 獲取激光雷達數據
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()  # 初始化目標點
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()  # 計算目標距離
        state, done = self.getState(data)  # 獲取當前狀態

        return np.asarray(state)  # 返回初始狀態
