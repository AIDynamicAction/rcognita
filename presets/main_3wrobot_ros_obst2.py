#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator) 
with connection to ROS.

"""

import os, sys
PARENT_DIR = os.path.abspath(__file__ + '/../..')
sys.path.insert(0, PARENT_DIR)
import rcognita

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = f"this script is being run using "+f"rcognita ({rcognita.__version__}) " + f"located in cloned repository at '{PARENT_DIR}'. " + f"If you are willing to use your locally installed rcognita, " + f"run this script ('{os.path.basename(__file__)}') outside " + f"'rcognita/presets'."
else:
    info = f"this script is being run using " + f"locally installed rcognita ({rcognita.__version__}). " + f"Make sure the versions match."
print("INFO:", info)

import pathlib
    
import warnings
import csv
from datetime import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from rcognita import simulator
from rcognita import systems
from rcognita import controllers
from rcognita import loggers
from rcognita import visuals
from rcognita.utilities import on_key_press

import argparse

#------------------------------------imports for interaction with ROS

import rospy
import threading

import os

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import tf.transformations as tftr
from geometry_msgs.msg import Point, Twist
from math import atan2, pi
import math
from numpy import inf, matrix, cos, arctan2, sqrt, pi, sin, cos
from numpy.random import randn
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, BFGS, basinhopping
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import time as time_lib
from sensor_msgs.msg import LaserScan

#------------------------------------define helpful classes
class Point:
    def __init__(self, R, x, y, angle):
        self.R = R
        self.x = x
        self.y = y
        self.angle = angle

class Circle:
    def __init__(self, center, r, convex):
        self.center = center
        self.r = r
        self.convex = convex

#------------------------------------define obstacles parser class
class Obstacles_parser:

    def __init__(self, W=0.178, T_d=10, safe_margin_mult=0.5):
        self.W = W
        self.T_d = T_d
        self.safe_margin = self.W * safe_margin_mult

    def __call__(self, l):
        self.new_blocks, self.lines, self.circles, self.x, self.y = self.get_obstacles(l)
        self.constraints = self.get_functions()
        return self.constraints

    def d(self, R_i, R_i1, delta_alpha=np.radians(1)):
        answ = np.sqrt(R_i**2 + R_i1**2 - 2*R_i*R_i1*np.cos(delta_alpha))
        return answ

    def k(self, R_i, R_i1):
        cond = (R_i + R_i1) / 2 > self.T_d
        if cond:
            k = (self.W * R_i * R_i1) / (100 * (R_i + R_i1))
            return k
        else:
            return 0.15

    def get_d_aver(self, block1, block2):
        ds = [self.d(block1[i].R, block1[i+1].R) for i in range(len(block1) - 1)]
        ds += [self.d(block2[i].R, block2[i+1].R) for i in range(len(block2) - 1)]
        if len(ds) < 1:
            return 0
        return np.mean(ds)

    def nan_helper(self, y, mode=1):
        if mode == 1:
            return np.isnan(y), lambda z: z.nonzero()[0]
        elif mode == 2:
            return np.isinf(y), lambda z: z.nonzero()[0]

    def segmentation(self, blocks):
        new_blocks = []
        changed = False
        for block in blocks:
            segmented = False
            for i, point in enumerate(block[:-1]):
                dd = self.d(block[i].R, block[i+1].R)
                kk = self.k(block[i].R, block[i+1].R)
                if dd > kk * self.W:
                    new_blocks.append(block[:i+1])
                    new_blocks.append(block[i+1:])
                    segmented = True
                    changed = True
                    break
            if not segmented:
                new_blocks.append(block)
        
        return changed, new_blocks

    def merging(self, blocks):
        new_blocks = []
        merged = False
        for i in range(len(blocks)):
            block1 = blocks[i]
            next_i = (i + 1) % len(blocks)
            block2 = blocks[next_i]
            p = block1[-1]
            q = block2[0]
            L = np.sqrt((p.x - q.x)**2 + (p.y - q.y)**2)
            if L < self.W:
                d_aver = self.get_d_aver(block1, block2)
                if np.isclose(d_aver, 0.):
                    N = 0
                else:
                    N = int(L // d_aver)
                    R_diff = (q.R - p.R) / N
                    angle_diff = (q.angle - p.angle) / N

                new_block = block1.copy()
                for j in range(N):
                    new_R = p.R + j * R_diff
                    new_angle = p.angle + j * angle_diff
                    new_x = new_R * np.cos(new_angle)
                    new_y = new_R * np.sin(new_angle)
                    new_block.append(Point(new_R, new_x, new_y, new_angle))
                new_block += block2
                merged = True
                new_blocks.append(new_block)
                break
            else:
                new_blocks.append(blocks[i])
        if merged:
            new_blocks += blocks[i+2:]
            if i == len(blocks) - 1:
                new_blocks.pop(0)
        return merged, new_blocks

    def get_D_m(self, block):
        p1 = np.array([block[0].x, block[0].y])
        p2 = np.array([block[-1].x, block[-1].y])
        ab_norm = np.linalg.norm(p2 - p1)
        D_m = 0
        k = 0
        for i, b in enumerate(block):
            p3 = np.array([b.x, b.y])
            D = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / ab_norm
            if D > D_m:
                D_m = D
                k = i
                
        return k, D_m

    def get_convexity(self, o_l, T_io, block):
        N = len(block)
        p_1 = np.array([block[-1].x, block[-1].y])
        p_N = np.array([block[0].x, block[0].y])
        V_i = []
        V_o = []
        for i in range(N):
            p_i = np.array([block[i].x, block[i].y])
            T_1 = np.cross(o_l - p_i, p_1 - p_i)
            T_2 = np.cross(p_1 - p_i, p_N - p_i)
            T_3 = np.cross(p_N - p_i, o_l - p_i)
            if ((T_1 > 0 and T_2 > 0 and T_3 > 0) or 
                (T_1 < 0 and T_2 < 0 and T_3 < 0)):
                V_i.append(p_i)
            else:
                V_o.append(p_i)
                
            r_io = len(V_i) / len(V_o)
            
            if r_io > T_io:
                return True    # means block is convex relative to the origin of the lidar
            else:
                return False    # means block is concave relative to the origin of the lidar

    def get_circle(self, block):
        x_center = (block[0].x + block[-1].x) / 2
        y_center = (block[0].y + block[-1].y) / 2
        center = np.array([x_center, y_center])
        k_, D_m = self.get_D_m(block)
        S = np.sqrt((block[0].x - block[-1].x)**2 + (block[0].y - block[-1].y)**2)
        
        ro = 0
        ind = 0
        for i, point in enumerate(block):
            p_coords = np.array([point.x, point.y])
            ro_cur = np.linalg.norm(p_coords - center)
            if ro_cur > ro:
                ro = ro_cur
                ind = i

        if D_m < 0.2 * S:
            return Circle(center, ro + self.safe_margin, None)
        else:
            return Circle(center, ro + self.safe_margin, self.get_convexity(o_l=np.array([0.0, 0.0]), T_io=2., block=block))

    def splitting(self, block, L, C, T_n=12, T_min=4, prints=False):
        R_mean = np.mean([pnt.R for pnt in block])
        N = int(len(block) * R_mean * 0.5)
        if N < T_min:
            return []
        if N < T_n:
            C.append(self.get_circle(block))
            return [block]

        N = len(block)
        a, b = np.array([block[0].x, block[0].y]), np.array([block[1].x, block[1 ].y])
        S = np.sqrt((block[0].x - block[-1].x)**2 + (block[0].y - block[-1].y)**2)
        k, D_m = self.get_D_m(block)
        d_p = 0.00614
        d_split = 0.15
        if prints:
            print(D_m, 0.2 * S, d_split + block[k].R * d_p)
        #if D_m > 0.2 * S:
        if D_m > d_split + block[k].R * d_p:
            B_1 = self.splitting(block[:k+1], L, C)
            B_2 = self.splitting(block[k:], L, C)
            return B_1 + B_2
        else:
            L.append((block[0], block[-1]))
            return [block]

    def get_obstacles(self, l, rng=360, fillna='const'):
        if fillna == 'const':
            fill_const = 1000
            nans = np.isnan(l)
            l[nans] = fill_const
            nans = np.isinf(l)
            l[nans] = fill_const
        elif fillna == 'interp':
            nans, idx_fun = self.nan_helper(l, mode=1)
            l[nans]= np.interp(idx_fun(nans), idx_fun(~nans), l[~nans]) + np.random.rand(nans.sum()) * 0.01

            nans, idx_fun = self.nan_helper(l, mode=2)
            l[nans]= np.interp(idx_fun(nans), idx_fun(~nans), l[~nans]) + np.random.rand(nans.sum()) * 0.01
            
        degrees = np.arange(rng)
        angles = np.radians(degrees)
        x = l * np.cos(angles)
        y = l * np.sin(angles)
        
        points = []

        for R, x_r, y_r, angle in zip(l, x, y, angles):
            point = Point(R, x_r, y_r, angle)
            points.append(point)
            
        p = [(l[i], np.radians(i)) for i in range(rng)]
        
        blocks = [points]
        
        flag = True
        while flag:
            flag, blocks = self.segmentation(blocks)
            
        flag = True
        while flag:
            flag, blocks = self.merging(blocks)
            
        LL = []
        CC = []
        
        new_blocks = []
        for block in blocks:
            #print(len(block))
            if len(block) > 5:
                new_blocks += self.splitting(block, LL, CC)
            
        return new_blocks, LL, CC, x, y

    def get_buffer_area(self, line, buf = 1):
        p1, p2 = line
        if (p1[0] > p2[0]):
            p_ = p1
            p1 = p2
            p2 = p_
        else:
            if (p1[1] > p2[1]):
                p_ = p1
                p1 = p2
                p2 = p_
            
        v_norm = np.linalg.norm([p2[0] - p1[0], p2[1] - p1[1]])
        unit_v = [(p2[0] - p1[0]) / v_norm, (p2[1] - p1[1]) / v_norm]
        if p2[0] - p1[0] != 0:
            coefs = [(p2[1] - p1[1]) / (p2[0] - p1[0]), p1[1] - (p2[1] - p1[1]) / (p2[0] - p1[0]) * p1[0]]
        else:
            coefs = [0, 0]
        
        p_intersect = [p1[0] - buf * unit_v[0], p1[1] - buf * unit_v[1]] 
        if coefs[0] != 0:
            coefs_ortho = [-1 / coefs[0], p_intersect[1] + 1 / coefs[0] * p_intersect[0]]
            p_another = [p_intersect[0] - buf, coefs_ortho[0] * (p_intersect[0] - buf) + coefs_ortho[1]]
        else:
            if p2[0] - p1[0] == 0:
                p_another = [p_intersect[0] - buf, p_intersect[1]]
            else:
                p_another = [p_intersect[0], p_intersect[0] - buf]
        
        v_ortho_norm = np.linalg.norm([p_intersect[0] - p_another[0], p_intersect[1] - p_another[1]])
        unit_v_par = [(p_intersect[0] - p_another[0]) / v_ortho_norm,
                    (p_intersect[1] - p_another[1]) / v_ortho_norm]
        
        sq_1 = [p_intersect[0] - buf * unit_v_par[0], p_intersect[1] - buf * unit_v_par[1]]   
        sq_2 = [sq_1[0] + unit_v[0] * (2 * buf + v_norm), sq_1[1] + unit_v[1] * (2 * buf + v_norm)]
        sq_4 = [p_intersect[0] + buf * unit_v_par[0], p_intersect[1] + buf * unit_v_par[1]] 
        sq_3 = [sq_4[0] + unit_v[0] * (2 * buf + v_norm), sq_4[1] + unit_v[1] * (2 * buf + v_norm)]
        buffer = [sq_1, sq_2, sq_3, sq_4]
        return buffer

    def get_functions(self):
        inequations = []
        figures = []
        
        def get_constraints(x):
            return np.max([np.min([coef[0] * x[0] + coef[1] * x[1] + coef[2] for coef in ineq_set]) 
                                        for ineq_set in inequations])
                
        def get_constraint(ineq_set):
            return lambda x: np.min([coef[0] * x[0] + coef[1] * x[1] + coef[2] for coef in ineq_set])
    
        def get_circle_constraints(x):
            centers = []
            radiuses = []
            for circle in self.circles:
                centers.append(circle.center)
                radiuses.append(circle.r)
            return np.max([(r**2 - (a - x[0])**2 - (b - x[1])**2) for [a, b], r in zip(centers, radiuses)])
        
    
        def get_straight_line(p1, p2):
            """
            return: array [a, b, c] for a* x + b * y + c
            """
            return [-(p2[1] - p1[1]), (p2[0] - p1[0]), - p2[0]* p1[1] + p1[0] * p2[1]]
        
        
        def check_ineq_sign(line, ctrl_point):
            straight_line = lambda x, y: line[0] * x + line[1] * y + line[2]
            if straight_line(ctrl_point[0], ctrl_point[1]) >= 0:
                return line
            else:
                for i in range(len(line)):
                    line[i] = -line[i]
                return line
            

        def get_figure_inequations(figure):
            hull = ConvexHull(figure)
            ans = []
            for simplex in hull.simplices:
                coefs = get_straight_line(figure[simplex[0]], figure[simplex[1]])
                for i in range(len(figure)):
                    if i not in simplex:
                        coefs = check_ineq_sign(coefs, figure[i])
                        break
                ans.append(coefs)
            return ans
        

        if len(self.lines) > 0:
            for line in self.lines:
                fig = self.get_buffer_area([[line[0].x, line[0].y], [line[1].x, line[1].y]], self.W / 2)
                figures.append(np.array(fig))
                inequations.append(get_figure_inequations(fig))

        constraints = []
        constraints.append(get_constraints)
        
        if len(self.circles) > 0:
            constraints.append(get_circle_constraints)

        return constraints

#------------------------------------define ROS-preset class
class ROS_preset:

    def __init__(self, ctrl_mode, state_goal, state_init, my_ctrl_nominal, my_sys, my_ctrl_benchm, my_logger=None, datafiles=None):
        self.RATE = rospy.get_param('/rate', 50)
        self.lock = threading.Lock()
        # initialization
        self.state_init = state_init
        self.state_goal = state_goal
        self.system = my_sys

        self.ctrl_nominal = my_ctrl_nominal
        self.ctrl_benchm = my_ctrl_benchm

        self.dt = 0.0
        self.time_start = 0.0

        self.constraints = []

        # connection to ROS topics
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=False)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odometry_callback)
        self.sub_laser_scan = rospy.Subscriber("/scan", LaserScan, self.laser_scan_callback)

        self.state = np.zeros((3))
        self.dstate = np.zeros((3))
        self.new_state = np.zeros((3))
        self.new_dstate = np.zeros((3))

        self.datafiles = datafiles
        self.logger = my_logger
        self.ctrl_mode = ctrl_mode

        self.rotation_counter = 0
        self.prev_theta = 0
        self.new_theta = 0

        theta_goal = self.state_goal[2]

        self.rotation_matrix = np.array([
            [cos(theta_goal), -sin(theta_goal), 0],
            [sin(theta_goal),cos(theta_goal), 0],
            [0, 0, 1]
        ])

        self.obstacles_parser = Obstacles_parser()
        #l = np.array((inf, inf, inf, 3.0861682891845703, 3.0325357913970947, 3.02504301071167, 2.010859966278076, 1.9776132106781006, 1.9460697174072266, 1.9390788078308105, 1.9310704469680786, 1.951540470123291, 1.9880226850509644, 2.0127334594726562, inf, inf, 1.030858039855957, 0.9641232490539551, 0.940187931060791, 0.9351522326469421, 0.9298539757728577, 0.9265134930610657, 0.9089210629463196, 0.9214298129081726, 0.8953527212142944, 0.9075207114219666, 0.9075554609298706, 0.9286268353462219, 0.9008835554122925, 0.9352240562438965, 0.9638814926147461, 0.972695529460907, 1.021325707435608, 2.4508416652679443, 2.434957504272461, 2.439626455307007, 2.4602088928222656, 2.515049934387207, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 1.7449512481689453, 1.7160905599594116, 1.7175880670547485, 1.7130509614944458, 1.6924936771392822, 1.70518958568573, 1.7248340845108032, 1.7293537855148315, 1.7895138263702393, 3.352675199508667, 3.3166370391845703, 3.3063015937805176, 3.2597343921661377, 3.2459218502044678, 3.2332046031951904, 3.2143239974975586, 3.181187629699707, 3.066962718963623, 2.8718676567077637, 2.682319164276123, 2.5715138912200928, 2.5700981616973877, 2.5573182106018066, 2.552452325820923, 2.5447068214416504, 2.5202600955963135, 2.51522159576416, 2.5173258781433105, 2.4995245933532715, 2.4168968200683594, 2.3383233547210693, 2.236551284790039, 2.175374984741211, 2.12975811958313, 2.066340923309326, 1.9802098274230957, 1.931720495223999, 1.8902508020401, 1.8316709995269775, 1.7821555137634277, 1.7443459033966064, 1.7134734392166138, 1.6665699481964111, 1.6329622268676758, 1.593448281288147, 1.5696789026260376, 1.5180320739746094, 1.5255087614059448, 1.4724040031433105, 1.4581708908081055, 1.4084149599075317, 1.3855719566345215, 1.3721964359283447, 1.3594599962234497, 1.34696364402771, 1.3276340961456299, 1.3083076477050781, 1.2606889009475708, 1.2622536420822144, 1.2477402687072754, 1.220187783241272, 1.2207741737365723, 1.2065539360046387, 1.196658730506897, 1.1684935092926025, 1.166418194770813, 1.1634021997451782, 1.1536171436309814, 1.155808925628662, 1.1352661848068237, 1.1265285015106201, 1.1049842834472656, 1.1149781942367554, 1.1156255006790161, 1.0915851593017578, 1.0709306001663208, 1.078560709953308, 1.0717359781265259, 1.0562193393707275, 1.0618635416030884, 1.0464541912078857, 1.0468363761901855, 1.0550421476364136, 1.0370537042617798, 1.0502249002456665, 1.0514434576034546, 1.0400515794754028, 1.035658836364746, 1.0286237001419067, 1.0316685438156128, 1.0358165502548218, 1.01632821559906, 1.0397361516952515, 1.0348272323608398, 1.0281643867492676, 1.0201276540756226, 1.0014697313308716, 0.9790696501731873, 0.9490427374839783, 0.9001302719116211, 0.9025514125823975, 0.8979737758636475, 0.8525041937828064, 0.824347734451294, 0.8220825791358948, 0.8142399787902832, 0.8025095462799072, 0.7514674067497253, 0.7544713616371155, 0.7565903663635254, 0.7404968738555908, 0.7141371965408325, 0.6917865872383118, 0.709116518497467, 0.6953718662261963, 0.6753686666488647, 0.6683269143104553, 0.6704203486442566, 0.6682959794998169, 0.6490408182144165, 0.6454023122787476, 0.6243855357170105, 0.6076543927192688, 0.6202757954597473, 0.5933164358139038, 0.6020201444625854, 0.6185680031776428, 0.6073765754699707, 0.5895431637763977, 0.5976154208183289, 0.5967515110969543, 0.5834906101226807, 0.568354606628418, 0.5575466752052307, 0.5679559111595154, 0.5458931922912598, 0.5569720268249512, 0.5660908818244934, 0.5704879760742188, 0.5407575368881226, 0.5620211362838745, 0.5495848655700684, 0.5612789392471313, 0.5461738109588623, 0.5374715328216553, 0.5425431132316589, 0.5223098993301392, 0.5350351333618164, 0.5479685068130493, 0.5320639610290527, 0.5366302132606506, 0.5173152089118958, 0.5428702235221863, 0.537230372428894, 0.5487949848175049, 0.5325515270233154, 0.5284975171089172, 0.5575615167617798, 0.5331605076789856, 0.5394099354743958, 0.5281550288200378, 0.5208730101585388, 0.544990062713623, 0.538831353187561, 0.5244802832603455, 0.526938796043396, 0.5349071025848389, 0.5664812326431274, 0.5715964436531067, 0.5634715557098389, 0.5471823811531067, 0.5346543192863464, 0.5488866567611694, 0.5521693825721741, 0.5617856383323669, 0.5642616152763367, 0.5810156464576721, 0.566067099571228, 0.5802873373031616, 0.5925429463386536, 0.5769492983818054, 0.6122656464576721, 0.6166649460792542, 0.6032747626304626, 0.6331508755683899, 0.6096622943878174, 0.6518257856369019, 0.6360791921615601, 0.6434801816940308, 0.6440712809562683, 0.6763648390769958, 0.6885241270065308, 0.668234646320343, 0.6885436773300171, 0.6910346150398254, 0.7126830816268921, 0.7157810926437378, 0.7463980317115784, 0.7452514171600342, 0.7664777636528015, 0.7687014937400818, 0.7888899445533752, 0.8124525547027588, 0.8159955143928528, 0.8400493860244751, 0.8485746383666992, 0.8809640407562256, 0.8813065886497498, 0.9069647192955017, 0.9248022437095642, 0.9501989483833313, 0.9905684590339661, 1.0354976654052734, 1.0568702220916748, 1.0902408361434937, 1.1097322702407837, 1.1646097898483276, 1.2034186124801636, 1.243091106414795, 1.3056540489196777, 1.3482526540756226, 1.3997080326080322, 1.4600743055343628, 1.5392513275146484, 1.5222216844558716, 1.5365455150604248, 1.5418308973312378, 1.5543769598007202, 1.5784618854522705, 1.5625876188278198, 1.5641851425170898, 1.581931233406067, 1.591413140296936, 1.584258794784546, 1.6096030473709106, 1.6249223947525024, 1.6428356170654297, 1.8563947677612305, 2.217372417449951, 2.260394334793091, 2.290161371231079, 2.3138082027435303, 2.3379597663879395, 2.3418986797332764, 2.3731155395507812, 2.4139623641967773, 2.431980609893799, 2.4462809562683105, 2.475482225418091, 2.5201022624969482, 2.5622761249542236, 2.5811896324157715, 2.627504348754883, 2.661574363708496, 2.7129135131835938, 2.7449240684509277, 2.770678997039795, 2.8333489894866943, 2.869255304336548, 2.92659330368042, 2.9639205932617188, 3.043680429458618, 3.101245164871216, 1.0509991645812988, 1.0017465353012085, 0.9950759410858154, 0.9702335596084595, 0.9499964714050293, 0.9654521346092224, 0.9504379630088806, 0.9405674934387207, 0.9491595029830933, 0.9678528308868408, 0.9619454741477966, 0.9994099736213684, 0.9796149134635925, 1.0094549655914307, 1.0262320041656494, 1.0835461616516113, inf, inf, inf, inf, 2.053907632827759, 2.0082805156707764, 2.0094406604766846, 1.960039496421814, 1.9746379852294922, 1.9873132705688477, 1.9843428134918213, 2.051220655441284, 3.0572266578674316, 3.0352628231048584, 3.0337300300598145, 3.0728259086608887, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf))
        #self.constraints = self.obstacles_parser(l)
        #for example in [[2, 1.5], [0, -1.3], [-0.8, 0.5], [1, 1.4]]:
        #    for constr in self.constraints:
        #        rospy.loginfo(f"RESULT OF CONSTRAINTS FUNCTION FOR EXAMPLE {example}: {constr(example)}")


    def odometry_callback(self, msg):
        self.lock.acquire()
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

        rotation_matrix = np.array([
            [cos(theta_goal), -sin(theta_goal), 0],
            [sin(theta_goal),cos(theta_goal), 0],
            [0, 0, 1]
        ])
        self.rotation_matrix = rotation_matrix.copy()

        state_matrix = np.array([
            [self.state_goal[0]],
            [self.state_goal[1]],
            [0]
        ])

        t_matrix = np.block([
            [rotation_matrix, state_matrix],
            [np.array([[0, 0, 0, 1]])]
        ])

        inv_t_matrix = np.linalg.inv(t_matrix)

        if math.copysign(1, self.prev_theta) != math.copysign(1, theta) and abs(self.prev_theta) > 3:
            if math.copysign(1, self.prev_theta) == -1:
                self.rotation_counter -= 1
            else:
                self.rotation_counter += 1

        self.prev_theta = theta
        theta = theta + 2 * math.pi * self.rotation_counter
        self.new_theta = theta

        new_theta = theta - theta_goal

        # POSITION transform
        temp_pos = [x, y, 0, 1]
        self.new_state = np.dot(inv_t_matrix, np.transpose(temp_pos))
        self.new_state = np.array([self.new_state[0], self.new_state[1], new_theta])

        inv_R_matrix = inv_t_matrix[:3, :3]
        zeros_like_R = np.zeros(inv_R_matrix.shape)
        inv_R_matrix = np.linalg.inv(rotation_matrix)
        self.new_dstate = inv_R_matrix.dot(np.array([dx, dy, 0]).T)
        new_omega = omega
        self.new_dstate = [self.new_dstate[0], self.new_dstate[1], new_omega]

        self.lock.release()

    def laser_scan_callback(self, dt):
        self.lock.acquire()
        # dt.ranges -> parser.get_obstacles(dt.ranges) -> get_functions(obstacles) -> self.constraints_functions
        try:
            self.constraints = self.obstacles_parser(np.array(dt.ranges))
            self.constraints = list(({'type': 'ineq', 'fun': constr}) for constr in self.constraints)
        except ValueError:
            self.constraints = []
        print('RANGES:', np.array(dt.ranges))
        print('CONSTRAINTS:', self.constraints)
        self.lock.release()

    def spin(self, is_print_sim_step=False, is_log_data=False):
        rospy.loginfo('ROS-preset has been activated!')
        start_time = time_lib.time()
        rate = rospy.Rate(self.RATE)
        self.time_start = rospy.get_time()
        while not rospy.is_shutdown() and time_lib.time() - start_time < 100:
            t = rospy.get_time() - self.time_start
            self.t = t
            print(t)

            velocity = Twist()

            action = controllers.ctrl_selector(self.t, self.new_state, action_manual, self.ctrl_nominal, 
                                                self.ctrl_benchm, self.ctrl_mode, self.constraints)
            action = np.clip(action, [-0.22, -2.0], [0.22, 2.0])
            
            self.system.receive_action(action)
            self.ctrl_benchm.receive_sys_state(self.system._state)
            self.ctrl_benchm.upd_accum_obj(self.new_state, action)

            xCoord = self.new_state[0]
            yCoord = self.new_state[1]
            alpha = self.new_state[2]

            stage_obj = self.ctrl_benchm.stage_obj(self.new_state, action)
            accum_obj = self.ctrl_benchm.accum_obj_val

            if is_print_sim_step:
                self.logger.print_sim_step(t, xCoord, yCoord, alpha, stage_obj, accum_obj, action)
            
            if is_log_data:
                self.logger.log_data_row(self.datafiles[0], t, xCoord, yCoord, alpha, stage_obj, accum_obj, action)

            self.ctrl_benchm.receive_sys_state(self.new_state)


            velocity.linear.x = action[0]
            velocity.angular.z = action[1]
            self.pub_cmd_vel.publish(velocity)

            rate.sleep()

        rospy.loginfo('ROS-preset has finished working')


if __name__ == "__main__":
    rospy.init_node('ROS_preset_node')

    #----------------------------------------Set up dimensions
    dim_state = 3
    dim_input = 2
    dim_output = dim_state
    dim_disturb = 2

    dim_R1 = dim_output + dim_input
    dim_R2 = dim_R1

    description = "Agent-environment preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator)."

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                        choices=['manual',
                                'nominal',
                                'MPC',
                                'RQL',
                                'SQL',
                                'JACS'],
                        default='nominal',
                        help='Control mode. Currently available: ' +
                        '----manual: manual constant control specified by action_manual; ' +
                        '----nominal: nominal controller, usually used to benchmark optimal controllers;' +                     
                        '----MPC:model-predictive control; ' +
                        '----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; ' +
                        '----SQL: stacked Q-learning; ' + 
                        '----JACS: joint actor-critic (stabilizing), system-specific, needs proper setup.')
    parser.add_argument('--dt', type=float, metavar='dt',
                        default=0.01,
                        help='Controller sampling time.' )
    parser.add_argument('--t1', type=float, metavar='t1',
                        default=150.0,
                        help='Final time of episode.' )
    parser.add_argument('--state_init', type=str, nargs="+", metavar='state_init',
                        default=['2', '2', 'pi'],
                        help='Initial state (as sequence of numbers); ' + 
                        'dimension is environment-specific!')
    parser.add_argument('--is_log_data', type=bool,
                        default=False,
                        help='Flag to log data into a data file. Data are stored in simdata folder.')
    parser.add_argument('--is_visualization', type=bool,
                        default=True,
                        help='Flag to produce graphical output.')
    parser.add_argument('--is_print_sim_step', type=bool,
                        default=True,
                        help='Flag to print simulation data into terminal.')
    parser.add_argument('--is_est_model', type=bool,
                        default=False,
                        help='Flag to estimate environment model.')
    parser.add_argument('--model_est_stage', type=float,
                        default=1.0,
                        help='Seconds to learn model until benchmarking controller kicks in.')
    parser.add_argument('--model_est_period_multiplier', type=float,
                        default=1,
                        help='Model is updated every model_est_period_multiplier times dt seconds.')
    parser.add_argument('--model_order', type=int,
                        default=5,
                        help='Order of state-space estimation model.')
    parser.add_argument('--prob_noise_pow', type=float,
                        default=8,
                        help='Power of probing (exploration) noise.')
    parser.add_argument('--action_manual', type=float,
                        default=[0.22, 0.0], nargs='+',
                        help='Manual control action to be fed constant, system-specific!')
    parser.add_argument('--Nactor', type=int,
                        default=3,
                        help='Horizon length (in steps) for predictive controllers.')
    parser.add_argument('--pred_step_size_multiplier', type=float,
                        default=6.0,
                        help='Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.')
    parser.add_argument('--buffer_size', type=int,
                        default=200,
                        help='Size of the buffer (experience replay) for model estimation, agent learning etc.')
    parser.add_argument('--stage_obj_struct', type=str,
                        default='quadratic',
                        choices=['quadratic',
                                'biquadratic'],
                        help='Structure of stage objective function.')
    parser.add_argument('--R1_diag', type=float, nargs='+',
                        default=[1, 10, 1, 0, 0],
                        help='Parameter of stage objective function. Must have proper dimension. ' +
                        'Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')
    parser.add_argument('--R2_diag', type=float, nargs='+',
                        default=[[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                        help='Parameter of stage objective function . Must have proper dimension. ' + 
                        'Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, ' +
                        'where diag() is transformation of a vector to a diagonal matrix.')
    parser.add_argument('--Ncritic', type=int,
                        default=50,
                        help='Critic stack size (number of temporal difference terms in critic cost).')
    parser.add_argument('--gamma', type=float,
                        default=1.0,
                        help='Discount factor.')
    parser.add_argument('--critic_period_multiplier', type=float,
                        default=5.0,
                        help='Critic is updated every critic_period_multiplier times dt seconds.')
    parser.add_argument('--critic_struct', type=str,
                        default='quad-nomix', choices=['quad-lin',
                                                    'quadratic',
                                                    'quad-nomix',
                                                    'quad-mix'],
                        help='Feature structure (critic). Currently available: ' +
                        '----quad-lin: quadratic-linear; ' +
                        '----quadratic: quadratic; ' +
                        '----quad-nomix: quadratic, no mixed terms; ' +
                        '----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations).')
    parser.add_argument('--actor_struct', type=str,
                        default='quad-nomix', choices=['quad-lin',
                                                    'quadratic',
                                                    'quad-nomix'],
                        help='Feature structure (actor). Currently available: ' +
                        '----quad-lin: quadratic-linear; ' +
                        '----quadratic: quadratic; ' +
                        '----quad-nomix: quadratic, no mixed terms.')

    args = parser.parse_args()

    #----------------------------------------Post-processing of arguments
    # Convert `pi` to a number pi
    for k in range(len(args.state_init)):
        args.state_init[k] = eval( args.state_init[k].replace('pi', str(np.pi)) )

    args.state_init = np.array(args.state_init)
    args.action_manual = np.array(args.action_manual)

    pred_step_size = args.dt * args.pred_step_size_multiplier
    model_est_period = args.dt * args.model_est_period_multiplier
    critic_period = args.dt * args.critic_period_multiplier

    R1 = np.diag(np.array(args.R1_diag))
    R2 = np.diag(np.array(args.R2_diag))

    assert args.t1 > args.dt > 0.0
    assert args.state_init.size == dim_state

    globals().update(vars(args))

    #----------------------------------------(So far) fixed settings
    is_disturb = 0

    # Disturbance
    sigma_q = 1e-3 * np.ones(dim_disturb)
    mu_q = np.zeros(dim_disturb)
    tau_q = np.ones(dim_disturb)

    is_dyn_ctrl = 0

    x0 = np.zeros(dim_state)

    t0 = 0
    Nruns = 1

    x0 = np.zeros(dim_state)
    action_init = np.zeros(dim_input)
    q0 = np.zeros(dim_disturb)

    # Solver
    atol = 1e-5
    rtol = 1e-3

    # xy-plane
    xMin = -10
    xMax = 10
    yMin = -10
    yMax = 10

    # Model estimator stores models in a stack and recall the best of model_est_checks
    model_est_checks = 0

    # Control constraints
    v_min = -0.22
    v_max = 0.22
    omega_min = -2
    omega_max = 2
    ctrl_bnds=np.array([[v_min, v_max], [omega_min, omega_max]])

    state_goal = state_init.copy()

    #----------------------------------------Initialization : : system
    my_sys = systems.Sys3WRobotNI(sys_type="diff_eqn",
                                        dim_state=dim_state,
                                        dim_input=dim_input,
                                        dim_output=dim_output,
                                        dim_disturb=dim_disturb,
                                        pars=[],
                                        ctrl_bnds=ctrl_bnds,
                                        is_dyn_ctrl=is_dyn_ctrl,
                                        is_disturb=is_disturb,
                                        pars_disturb=[sigma_q, mu_q, tau_q])

    observation_init = my_sys.out(state_init)

    xCoord0 = state_init[0]
    yCoord0 = state_init[1]
    alpha0 = state_init[2]
    alpha_deg_0 = np.degrees(alpha0)

    #----------------------------------------Initialization : : model

    #----------------------------------------Initialization : : controller
    my_ctrl_nominal = controllers.CtrlNominal3WRobotNI(ctrl_gain=0.02, ctrl_bnds=ctrl_bnds, t0=t0, sampling_time=dt)

    # Predictive optimal controller
    my_ctrl_opt_pred = controllers.CtrlOptPred(dim_input,
                                            dim_output,
                                            ctrl_mode,
                                            ctrl_bnds = ctrl_bnds,
                                            action_init = [],
                                            t0 = t0,
                                            sampling_time = dt,
                                            Nactor = Nactor,
                                            pred_step_size = pred_step_size,
                                            sys_rhs = my_sys._state_dyn,
                                            sys_out = my_sys.out,
                                            state_sys = x0,
                                            prob_noise_pow = prob_noise_pow,
                                            is_est_model = is_est_model,
                                            model_est_stage = model_est_stage,
                                            model_est_period = model_est_period,
                                            buffer_size = buffer_size,
                                            model_order = model_order,
                                            model_est_checks = model_est_checks,
                                            gamma = gamma,
                                            Ncritic = Ncritic,
                                            critic_period = critic_period,
                                            critic_struct = critic_struct,
                                            stage_obj_struct = stage_obj_struct,
                                            stage_obj_pars = [R1],
                                            observation_target = [])

    # Stabilizing RL agent
    my_ctrl_RL_stab = controllers.CtrlRLStab(dim_input,
                                            dim_output,
                                            ctrl_mode,
                                            ctrl_bnds = ctrl_bnds,
                                            action_init = action_init,
                                            t0 = t0,
                                            sampling_time = dt,
                                            Nactor = Nactor,
                                            pred_step_size = pred_step_size,
                                            sys_rhs = my_sys._state_dyn,
                                            sys_out = my_sys.out,
                                            state_sys = state_init,
                                            prob_noise_pow = prob_noise_pow,
                                            is_est_model = is_est_model,
                                            model_est_stage = model_est_stage,
                                            model_est_period = model_est_period,
                                            buffer_size = buffer_size,
                                            model_order = model_order,
                                            model_est_checks = model_est_checks,
                                            gamma = gamma,
                                            Ncritic = Ncritic,
                                            critic_period = critic_period,
                                            critic_struct = critic_struct,
                                            actor_struct = actor_struct,
                                            stage_obj_struct = stage_obj_struct,
                                            stage_obj_pars = [R1],
                                            observation_target = [],
                                            safe_ctrl = my_ctrl_nominal,
                                            safe_decay_rate = 1e-4)

    if ctrl_mode == 'JACS':
        my_ctrl_benchm = my_ctrl_RL_stab
    else:
        my_ctrl_benchm = my_ctrl_opt_pred

    #----------------------------------------Initialization : : logger
    if os.path.basename( os.path.normpath( os.path.abspath(os.getcwd()) ) ) == 'presets':
        data_folder = '../simdata'
    else:
        data_folder = 'simdata'

    pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True) 

    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    datafiles = [None] * Nruns

    for k in range(0, Nruns):
        datafiles[k] = data_folder + '/' + 'ROS__' + my_sys.name + '__' + ctrl_mode + '__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
        
        if is_log_data:
            print('Logging data to:    ' + datafiles[k])
                
            with open(datafiles[k], 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(['System', my_sys.name ] )
                writer.writerow(['Controller', ctrl_mode ] )
                writer.writerow(['dt', str(dt) ] )
                writer.writerow(['state_init', str(state_init) ] )
                writer.writerow(['is_est_model', str(is_est_model) ] )
                writer.writerow(['model_est_stage', str(model_est_stage) ] )
                writer.writerow(['model_est_period_multiplier', str(model_est_period_multiplier) ] )
                writer.writerow(['model_order', str(model_order) ] )
                writer.writerow(['prob_noise_pow', str(prob_noise_pow) ] )
                writer.writerow(['Nactor', str(Nactor) ] )
                writer.writerow(['pred_step_size_multiplier', str(pred_step_size_multiplier) ] )
                writer.writerow(['buffer_size', str(buffer_size) ] )
                writer.writerow(['stage_obj_struct', str(stage_obj_struct) ] )
                writer.writerow(['R1_diag', str(R1_diag) ] )
                writer.writerow(['R2_diag', str(R2_diag) ] )
                writer.writerow(['Ncritic', str(Ncritic) ] )
                writer.writerow(['gamma', str(gamma) ] )
                writer.writerow(['critic_period_multiplier', str(critic_period_multiplier) ] )
                writer.writerow(['critic_struct', str(critic_struct) ] )
                writer.writerow(['actor_struct', str(actor_struct) ] )   
                writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'stage_obj', 'accum_obj', 'v [m/s]', 'omega [rad/s]'] )

    # Do not display annoying warnings when print is on
    if is_print_sim_step:
        warnings.filterwarnings('ignore')
        
    my_logger = loggers.Logger3WRobotNI()

    ros_preset_task = ROS_preset(ctrl_mode, [0, 0, 0], state_goal, my_ctrl_nominal, my_sys, my_ctrl_benchm, my_logger, datafiles)
    
    ros_preset_task.spin(is_print_sim_step=is_print_sim_step, is_log_data=is_log_data)