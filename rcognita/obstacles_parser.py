import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)
import rcognita

import numpy as np


import os
from scipy.spatial import ConvexHull


class myPoint:
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


class Obstacles_parser:
    def __init__(self, W=0.178, T_d=10, safe_margin_mult=2):
        self.W = W
        self.T_d = T_d
        self.safe_margin = self.W * safe_margin_mult

    def __call__(self, l, state):
        self.new_blocks, self.lines, self.circles, self.x, self.y = self.get_obstacles(
            l, fillna="else", state=state
        )
        self.constraints = self.get_functions()
        return self.constraints

    def d(self, R_i, R_i1, delta_alpha=np.radians(1)):
        answ = np.sqrt(R_i ** 2 + R_i1 ** 2 - 2 * R_i * R_i1 * np.cos(delta_alpha))
        return answ

    def k(self, R_i, R_i1):
        cond = (R_i + R_i1) / 2 > self.T_d
        if cond:
            k = (self.W * R_i * R_i1) / (100 * (R_i + R_i1))
            return k
        else:
            return 0.15

    def get_d_aver(self, block1, block2):
        ds = [self.d(block1[i].R, block1[i + 1].R) for i in range(len(block1) - 1)]
        ds += [self.d(block2[i].R, block2[i + 1].R) for i in range(len(block2) - 1)]
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
                dd = self.d(block[i].R, block[i + 1].R)
                kk = self.k(block[i].R, block[i + 1].R)
                if dd > kk * self.W:
                    new_blocks.append(block[: i + 1])
                    new_blocks.append(block[i + 1 :])
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
            L = np.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2)
            if L < self.W:
                d_aver = self.get_d_aver(block1, block2)
                if np.isclose(d_aver, 0.0):
                    N = 0
                else:
                    N = int(L // d_aver)
                    R_diff = (q.R - p.R) / N
                    angle_diff = (q.angle - p.angle) / N

                new_block = block1.copy()
                new_block += block2
                merged = True
                new_blocks.append(new_block)
                break
            else:
                new_blocks.append(blocks[i])
        if merged:
            new_blocks += blocks[i + 2 :]
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
            if (T_1 > 0 and T_2 > 0 and T_3 > 0) or (T_1 < 0 and T_2 < 0 and T_3 < 0):
                V_i.append(p_i)
            else:
                V_o.append(p_i)

            r_io = len(V_i) / len(V_o)

            if r_io > T_io:
                return True  # means block is convex relative to the origin of the lidar
            else:
                return (
                    False  # means block is concave relative to the origin of the lidar
                )

    def get_circle(self, block):
        x_center = (block[0].x + block[-1].x) / 2
        y_center = (block[0].y + block[-1].y) / 2
        center = np.array([x_center, y_center])
        k_, D_m = self.get_D_m(block)
        S = np.sqrt((block[0].x - block[-1].x) ** 2 + (block[0].y - block[-1].y) ** 2)

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
            return Circle(
                center,
                ro + self.safe_margin,
                self.get_convexity(o_l=np.array([0.0, 0.0]), T_io=2.0, block=block),
            )

    def splitting(self, block, L, C, T_n=9, T_min=3):
        R_mean = np.mean([pnt.R for pnt in block])
        N = int(len(block) * R_mean * 0.8)
        if N < T_min:
            return []
        if N < T_n:
            C.append(self.get_circle(block))
            return [block]

        N = len(block)
        a, b = np.array([block[0].x, block[0].y]), np.array([block[1].x, block[1].y])
        S = np.sqrt((block[0].x - block[-1].x) ** 2 + (block[0].y - block[-1].y) ** 2)
        k, D_m = self.get_D_m(block)
        d_p = 0.00614
        d_split = 0.10

        if D_m > d_split + block[k].R * d_p:
            B_1 = self.splitting(block[: k + 1], L, C)
            B_2 = self.splitting(block[k:], L, C)
            return B_1 + B_2
        else:
            L.append((block[0], block[-1]))
            return [block]

    def get_obstacles(self, l, rng=360, fillna="else", state=np.array([2, 2, np.pi])):
        state_coord = state[:2].reshape(-1, 1)
        state_angle = state[2]
        rot_mat = np.array(
            [
                [np.cos(state_angle), -np.sin(state_angle)],
                [np.sin(state_angle), np.cos(state_angle)],
            ]
        )

        degrees = np.arange(rng) + np.degrees(state_angle)
        angles = np.radians(degrees)

        if fillna == "const":
            fill_const = 1000
            nans = np.isnan(l)
            l[nans] = fill_const
            nans = np.isinf(l)
            l[nans] = fill_const
        elif fillna == "interp":
            nans, idx_fun = self.nan_helper(l, mode=1)
            l[nans] = (
                np.interp(idx_fun(nans), idx_fun(~nans), l[~nans])
                + np.random.rand(nans.sum()) * 0.01
            )

            nans, idx_fun = self.nan_helper(l, mode=2)
            l[nans] = (
                np.interp(idx_fun(nans), idx_fun(~nans), l[~nans])
                + np.random.rand(nans.sum()) * 0.01
            )
        else:
            nans = np.isnan(l)
            l = l[~nans]
            degrees = degrees[~nans]
            angles = angles[~nans]
            nans = np.isinf(l)
            l = l[~nans]
            degrees = degrees[~nans]
            angles = angles[~nans]
            nans = np.isclose(l, 0.0)
            l = l[~nans]
            degrees = degrees[~nans]
            angles = angles[~nans]
            mask = l < 2.0
            l = l[mask]
            degrees = degrees[mask]
            angles = angles[mask]

        if len(l) == 0:
            return None, [], [], None, None

        x = l * np.cos(angles)
        y = l * np.sin(angles)

        all_coords = np.column_stack([x, y]).T + state_coord
        x, y = all_coords[0, :], all_coords[1, :]
        self.x = x
        self.y = y

        points = []

        for R, x_r, y_r, angle in zip(l, x, y, angles):
            point = myPoint(R, x_r, y_r, angle)
            points.append(point)

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
            if len(block) > 5:
                new_blocks += self.splitting(block, LL, CC)

        return new_blocks, LL, CC, x, y

    def get_buffer_area(self, line, buf=0.178):
        p1, p2 = line
        if p1[0] > p2[0]:
            p_ = p1
            p1 = p2
            p2 = p_
        else:
            if p1[1] > p2[1]:
                p_ = p1
                p1 = p2
                p2 = p_

        v_norm = np.linalg.norm([p2[0] - p1[0], p2[1] - p1[1]])
        unit_v = [(p2[0] - p1[0]) / v_norm, (p2[1] - p1[1]) / v_norm]
        if p2[0] - p1[0] != 0:
            coefs = [
                (p2[1] - p1[1]) / (p2[0] - p1[0]),
                p1[1] - (p2[1] - p1[1]) / (p2[0] - p1[0]) * p1[0],
            ]
        else:
            coefs = [0, 0]

        p_intersect = [p1[0] - buf * unit_v[0], p1[1] - buf * unit_v[1]]
        if coefs[0] != 0:
            coefs_ortho = [
                -1 / coefs[0],
                p_intersect[1] + 1 / coefs[0] * p_intersect[0],
            ]
            p_another = [
                p_intersect[0] - buf,
                coefs_ortho[0] * (p_intersect[0] - buf) + coefs_ortho[1],
            ]
        else:
            if p2[0] - p1[0] == 0:
                p_another = [p_intersect[0] - buf, p_intersect[1]]
            else:
                p_another = [p_intersect[0], p_intersect[0] - buf]

        v_ortho_norm = np.linalg.norm(
            [p_intersect[0] - p_another[0], p_intersect[1] - p_another[1]]
        )
        unit_v_par = [
            (p_intersect[0] - p_another[0]) / v_ortho_norm,
            (p_intersect[1] - p_another[1]) / v_ortho_norm,
        ]

        sq_1 = [
            p_intersect[0] - buf * unit_v_par[0],
            p_intersect[1] - buf * unit_v_par[1],
        ]
        sq_2 = [
            sq_1[0] + unit_v[0] * (2 * buf + v_norm),
            sq_1[1] + unit_v[1] * (2 * buf + v_norm),
        ]
        sq_4 = [
            p_intersect[0] + buf * unit_v_par[0],
            p_intersect[1] + buf * unit_v_par[1],
        ]
        sq_3 = [
            sq_4[0] + unit_v[0] * (2 * buf + v_norm),
            sq_4[1] + unit_v[1] * (2 * buf + v_norm),
        ]
        buffer = [sq_1, sq_2, sq_3, sq_4, sq_1]

        return np.array(buffer)

    def get_functions(self):
        inequations = []
        figures = []

        def get_constraints(x):
            return np.max(
                [
                    np.min(
                        [coef[0] * x[0] + coef[1] * x[1] + coef[2] for coef in ineq_set]
                    )
                    for ineq_set in inequations
                ]
            )

        def get_circle_constraints(x):
            centers = []
            radiuses = []
            for circle in self.circles:
                centers.append(circle.center)
                radiuses.append(circle.r)
            return (
                np.max(
                    [
                        (r ** 2 - (a - x[0]) ** 2 - (b - x[1]) ** 2)
                        for [a, b], r in zip(centers, radiuses)
                    ]
                )
                if len(centers) > 0
                else 0
            )

        def get_straight_line(p1, p2):
            """
            return: array [a, b, c] for a* x + b * y + c
            """
            return [-(p2[1] - p1[1]), (p2[0] - p1[0]), -p2[0] * p1[1] + p1[0] * p2[1]]

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

        constraints = []

        if len(self.lines) > 0:
            for line in self.lines:
                x1, y1 = line[0].x, line[0].y
                x2, y2 = line[1].x, line[1].y
                fig = self.get_buffer_area([[x1, y1], [x2, y2]], self.safe_margin)
                figures.append(np.array(fig))
                inequations.append(get_figure_inequations(fig))

            constraints.append(get_constraints)

        if len(self.circles) > 0:
            constraints.append(get_circle_constraints)

        return constraints
