import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)
import rcognita

import numpy as np

from rcognita.obstacles_parser import Obstacles_parser
from shapely.geometry import Point

# ------------------------------------imports for interaction with ROS

import rospy
import threading

import os

from nav_msgs.msg import Odometry
import tf.transformations as tftr
from geometry_msgs.msg import Twist
import math
from numpy import cos, sin, cos
import time as time_lib
from sensor_msgs.msg import LaserScan
from scipy.signal import medfilt


class ROSHarness:
    def __init__(
        self,
        control_mode,
        state_init,
        state_goal,
        my_ctrl_nominal,
        my_sys,
        my_ctrl_benchm,
        action_manual,
        stage_objective,
        my_logger=None,
        datafiles=None,
        dt=0.05,
        pred_step_size=1.0,
    ):
        self.accum_obj_val = 0
        self.stage_objective = stage_objective
        self.action_manual = action_manual
        self.RATE = rospy.get_param("/rate", 1.0 / dt)

        self.odom_lock = threading.Lock()
        self.lidar_lock = threading.Lock()
        self.lidar_lock.acquire()
        # initialization
        self.state_init = state_init
        self.state_goal = state_goal
        self.system = my_sys

        self.ctrl_nominal = my_ctrl_nominal
        self.ctrl_benchm = my_ctrl_benchm

        self.time_start = 0.0

        self.constraints = []
        self.polygonal_constraints = []
        self.line_constrs = []
        self.circle_constrs = []
        self.ranges = []
        self.ranges_t0 = None
        self.ranges_t1 = None
        self.ranges_t2 = None

        # connection to ROS topics
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=False)
        self.sub_odom = rospy.Subscriber(
            "/odom", Odometry, self.odometry_callback, queue_size=1
        )
        self.sub_laser_scan = rospy.Subscriber(
            "/scan", LaserScan, self.laser_scan_callback, queue_size=1
        )

        self.state = np.zeros((3))
        self.dstate = np.zeros((3))
        self.new_state = np.zeros((3))
        self.new_dstate = np.zeros((3))

        self.datafiles = datafiles
        self.logger = my_logger
        self.control_mode = control_mode

        self.rotation_counter = 0
        self.prev_theta = 0
        self.new_theta = 0

        theta_goal = self.state_goal[2]

        self.rotation_matrix = np.array(
            [
                [cos(theta_goal), -sin(theta_goal), 0],
                [sin(theta_goal), cos(theta_goal), 0],
                [0, 0, 1],
            ]
        )

        self.obstacles_parser = Obstacles_parser(safe_margin_mult=1.5)

    def upd_accum_obj(self, observation, action, delta):

        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize objective instead).
        
        """

        self.accum_obj_val += self.stage_objective(observation, action) * delta

    def odometry_callback(self, msg):

        self.odom_lock.acquire()

        # Read current robot state
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        current_rpy = tftr.euler_from_quaternion((q.x, q.y, q.z, q.w))
        theta = current_rpy[2]

        dx = msg.twist.twist.linear.x
        dy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        self.state = [x, y, theta]
        self.dstate = [dx, dy, omega]

        # Make transform matrix from `robot body` frame to `goal` frame
        theta_goal = self.state_goal[2]

        rotation_matrix = np.array(
            [
                [cos(theta_goal), -sin(theta_goal), 0],
                [sin(theta_goal), cos(theta_goal), 0],
                [0, 0, 1],
            ]
        )
        self.rotation_matrix = rotation_matrix.copy()

        state_matrix = np.array([[self.state_goal[0]], [self.state_goal[1]], [0]])

        t_matrix = np.block(
            [[rotation_matrix, state_matrix], [np.array([[0, 0, 0, 1]])]]
        )

        inv_t_matrix = np.linalg.inv(t_matrix)

        if self.prev_theta * theta < 0 and abs(self.prev_theta - theta) > np.pi:
            if self.prev_theta < 0:
                self.rotation_counter -= 1
            else:
                self.rotation_counter += 1

        self.prev_theta = theta
        theta = theta + 2 * math.pi * self.rotation_counter

        new_theta = theta - theta_goal

        # POSITION transform
        temp_pos = [x, y, 0, 1]
        new_state = np.dot(inv_t_matrix, np.transpose(temp_pos))
        self.new_state = np.array([new_state[0], new_state[1], new_theta])

        inv_R_matrix = inv_t_matrix[:3, :3]
        inv_R_matrix = np.linalg.inv(rotation_matrix)
        new_dstate = inv_R_matrix.dot(np.array([dx, dy, 0]).T)
        new_omega = omega
        self.new_dstate = [new_dstate[0], new_dstate[1], new_omega]

        cons = []
        for constr in self.constraints:
            cons.append(constr(self.new_state[:2]))
        f1 = np.max(cons) if len(cons) > 0 else 0

        if f1 > 0:
            print("COLLISION!!!")
            print("ranges", self.ranges)
            print("state", self.new_state)

        self.lidar_lock.release()

    def laser_scan_callback(self, dt):
        self.lidar_lock.acquire()
        timer = rospy.get_time()
        # dt.ranges -> parser.get_obstacles(dt.ranges) -> get_functions(obstacles) -> self.constraints_functions
        try:
            self.ranges_t0 = np.array(dt.ranges)
            if self.ranges_t1 is None and self.ranges_t2 is None:
                self.ranges_t1 = self.ranges_t0
                self.ranges_t2 = self.ranges_t0
                self.ranges = medfilt(
                    np.array([self.ranges_t0, self.ranges_t1, self.ranges_t2]), [3, 3]
                )[2, :]
            else:
                self.ranges = medfilt(
                    np.array([self.ranges_t0, self.ranges_t1, self.ranges_t2]), [3, 3]
                )[2, :]
                self.ranges_t2 = self.ranges_t1
                self.ranges_t1 = self.ranges_t0
            new_blocks, LL, CC, x, y = self.obstacles_parser.get_obstacles(
                np.array(dt.ranges), fillna="else", state=self.new_state
            )
            self.lines = LL
            self.circles = CC
            self.line_constrs = [
                self.obstacles_parser.get_buffer_area(
                    [[i[0].x, i[0].y], [i[1].x, i[1].y]],
                    self.obstacles_parser.safe_margin,
                )
                for i in LL
            ]

            self.circle_constrs = [
                Point(i.center[0], i.center[1]).buffer(i.r) for i in CC
            ]
            self.constraints = self.obstacles_parser(
                np.array(dt.ranges), np.array(self.new_state)
            )

            self.polygonal_constraints = self.line_constrs + self.circle_constrs

        except ValueError as exc:
            print("Exception!", exc)
            self.constraints = []

        timer = rospy.get_time() - timer
        print("time of lidar callback:", timer)
        self.odom_lock.release()

    def spin(self, is_print_sim_step=False, is_log_data=True):
        rospy.loginfo("ROS-preset has been activated!")
        start_time = time_lib.time()
        rate = rospy.Rate(self.RATE)
        self.time_start = rospy.get_time()
        t = t_prev = 0

        while not rospy.is_shutdown() and (rospy.get_time() - self.time_start) < 240:
            timer = rospy.get_time()
            t = rospy.get_time() - self.time_start
            self.t = t

            delta_t = t - t_prev

            t_prev = t

            velocity = Twist()
            action = self.ctrl_benchm.compute_action_sampled(
                self.t, self.new_state, self.constraints
            )

            self.system.receive_action(action)

            xCoord = self.new_state[0]
            yCoord = self.new_state[1]
            alpha = self.new_state[2]

            stage_obj = self.stage_objective(self.new_state, action)
            self.upd_accum_obj(self.new_state, action, delta_t)
            accum_obj = self.accum_obj_val

            if is_print_sim_step:
                self.logger.print_sim_step(
                    t, xCoord, yCoord, alpha, stage_obj, accum_obj, action
                )

            if is_log_data:
                self.logger.log_data_row(
                    self.datafiles[0],
                    t,
                    xCoord,
                    yCoord,
                    alpha,
                    stage_obj,
                    accum_obj,
                    action,
                )

            velocity.linear.x = action[0]
            velocity.angular.z = action[1]

            if (
                np.sqrt(
                    (xCoord - self.state_init[0]) ** 2
                    + (yCoord - self.state_init[1]) ** 2
                )
                < 0.1
                and (
                    (np.abs(np.degrees(alpha - self.state_init[2])) % 360 < 15)
                    or (
                        345 < np.abs(np.degrees(alpha - self.state_init[2])) % 360 < 360
                    )
                )
            ) and t > 10.0:
                print("FINAL RESULTS!!!")
                print(t, xCoord, yCoord, alpha, stage_obj, accum_obj, action)
                velocity.linear.x = 0
                velocity.angular.z = 0
                break

            rate.sleep()
            self.pub_cmd_vel.publish(velocity)

            print("loop time", rospy.get_time() - timer)

        velocity = Twist()
        velocity.linear.x = 0.0
        velocity.angular.z = 0.0
        self.pub_cmd_vel.publish(velocity)

        rospy.loginfo("ROS-preset has finished working")
