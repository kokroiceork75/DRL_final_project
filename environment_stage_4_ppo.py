#! /usr/bin/python2.7
#coding:utf-8
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

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
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.obstacle_min_range =0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
#获取目标点距离
    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance
#获取里程计信息
    def getOdometry(self, odom):
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
        min_range = 0.1 #碰撞距离
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)#选择最小的激光雷达信息
        self.obstacle_min_range = obstacle_min_range
        obstacle_angle = np.argmin(scan_range)#数组里面最小的值
        #min_range>激光雷达信息即为碰撞
        if obstacle_min_range< 0.15 :
            done = True #碰撞

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)#计算小车里程计到目标点的距离
        if current_distance < 0.2:#小车距离目标点0.2即为到达目标点
            self.get_goalbox = True#到达目标点

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done#返回state28个数据

    def setReward(self, state, done, action):#传入state,done,action
        yaw_reward = []#角度奖励
        obstacle_min_range = state[-2]#获取激光雷达信息最小的数据
        self.obstacle_min_range = obstacle_min_range#
        current_distance = state[-3]#获取当前数据
        heading = state[-4]#小车的朝向角


        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2#角度分解
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])#角度计算
            yaw_reward.append(tr)#储存角度奖励

        if obstacle_min_range <= 0.2:#激光雷达最小数据小于0.1
            scan_reward = -1/(obstacle_min_range+0.3)#奖励范围-3.33到-2.5
        else:
            scan_reward =2
        # reward = scan_reward
        # return scan_reward
        distance_rate = 2 ** (current_distance / self.goal_distance)#距离比

        # if obstacle_min_range < 0.5:
        #     ob_reward = -5
        # else:
        #     ob_reward =1

        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) +scan_reward
        # reward =scan_reward 

#碰撞
        if done:
            rospy.loginfo("Collision!!")
            reward = -500+scan_reward
            self.goal_x,self.goal_y = self.respawn_goal.getPosition(True,delete=True)
            self.pub_cmd_vel.publish(Twist())
#到达目标点
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000+scan_reward
            self.pub_cmd_vel.publish(Twist())#停止运动
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)#删除模型
            self.goal_distance = self.getGoalDistace()#获得目标点
            self.get_goalbox = False#置False

        return reward


    def step(self, action):
        # obstacle_min_range = state[-2]
        max_angular_vel = 1.5#最大角速度
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        # global obstacle_min_range
        vel_cmd = Twist()
        # vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        # self.obstacle_min_range =obstacle_min_range
        # if self.obstacle_min_range <0.2:
        #     vel_cmd.linear.x =self.obstacle_min_range*0.1
        # else:
        vel_cmd.linear.x = 0.2


        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)