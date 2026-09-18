#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:22:46 2024

@author: zzha
"""
import numpy as np

from scipy.optimize import minimize
from functions import print_control as prc # the shared function of print control


# add min speed limit 2024.6.6    
class GetBVCurve2:
    def __init__(self, v0, t, dis=200, min_speed=0, max_speed=25):
        self.dis = dis
        self.v0 = v0
        self.t = t
        self.max_speed = max_speed
        self.min_speed = min_speed  # Minimum speed constraint
    
    def objective(self, x):
        t1, a1, t3, a3 = x
        vt = self.v0 + a1 * t1 + a3 * t3
        return -vt
    
    def constraint1(self, x):
        t1, a1, t3, a3 = x
        t2 = self.t - t1 - t3
        v2 = self.v0 + a1 * t1
        return self.dis - (self.v0 * t1 + 0.5 * a1 * t1**2 + v2 * t2 + v2 * t3 + 0.5 * a3 * t3**2)
    
    def constraint2(self, x):
        t1, a1, t3, a3 = x
        return self.t - (t1 + t3)
    
    def constraint3(self, x):
        t1, a1, t3, a3 = x
        vt = self.v0 + a1 * t1 + a3 * t3
        return self.max_speed - vt
    
    def constraint4(self, x):
        t1, a1, t3, a3 = x
        v1 = self.v0 + a1 * t1
        v3 = v1 + a3 * t3
        return min(v1, v3) - self.min_speed  # Ensure velocities are above the minimum speed
    
    def optimize(self):
        cons = (
            {'type': 'eq', 'fun': self.constraint1},
            {'type': 'eq', 'fun': self.constraint2},
            {'type': 'ineq', 'fun': self.constraint3},
            {'type': 'ineq', 'fun': self.constraint4}  # Velocity non-negative constraint
        )
        bnds = ((0, self.t), (-4.5, 0), (0, self.t), (0, 2.6))
        x0 = [self.t / 3, -2, self.t / 3, 1]
        sol = minimize(self.objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        v_rem = -sol.fun  # velocity of reach moment
        prc.print_message("\n*Optimiser result*")
        prc.print_message(f'Final speed: {v_rem:.2f}')
        prc.print_message(f'Optimal solution: {np.round(sol.x, 2)}')  # t1, a1, t3, a3
        return sol, v_rem

if __name__ == '__main__':
    dis = 200  # The specific value for displacement to set
    t = 16.9   # The specific value for total time to set
    v0 = 6.276  # The specific value for initial velocity to set
    optm = GetBVCurve2(v0=20, t=12)
    r, v_rem = optm.optimize() # r.x = [t1, a1 (dec), t3, a3 (acc)]

