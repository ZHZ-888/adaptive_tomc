#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:22:46 2024

@author: zzha
"""
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.optimize import minimize
from functions import print_control as prc # the shared function of print control


# add min speed limit 2024.6.6    
class GetBVCurve2:
    def __init__(self, min_speed=5, max_speed=25,
                 max_accel=2.6, max_decel=4.5,
                 arrival_speed_ratio=1.0):
        self.min_speed = min_speed  # Minimum speed constraint
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.arrival_speed_ratio = arrival_speed_ratio
    
    def objective(self, x, v0):
        t1, a1, t3, a3 = x
        vt = v0 + a1 * t1 + a3 * t3
        return -vt
    
    def _dis_constr(self, x, v0, dis, t): # Distance constraint
        t1, a1, t3, a3 = x
        t2 = t - t1 - t3
        v2 = v0 + a1 * t1
        return dis - (v0 * t1 + 0.5 * a1 * t1**2 + v2 * t2 + v2 * t3 + 0.5 * a3 * t3**2)
    
    def _dur_constr(self, x, t):
        t1, a1, t3, a3 = x
        return t - (t1 + t3)
    
    def _arrival_speed_constr(self, x, v0):
        t1, a1, t3, a3 = x
        vt = v0 + a1 * t1 + a3 * t3
        return self.max_speed - vt
    
    def _min_speed_constr(self, x, v0):
        t1, a1, t3, a3 = x
        v1 = v0 + a1 * t1
        v3 = v1 + a3 * t3
        return min(v1, v3) - self.min_speed  # Ensure velocities are above the minimum speed
    
    def optimise(self, v0, dis, t):
        cons = (
            {'type': 'eq', 'fun': self._dis_constr, 'args': (v0, dis, t)},
            {'type': 'eq', 'fun': self._dur_constr, 'args': (t,)},
            {'type': 'ineq', 'fun': self._arrival_speed_constr, 'args': (v0,)},
            {'type': 'ineq', 'fun': self._min_speed_constr, 'args': (v0,)}  # Velocity non-negative constraint
        )
        bnds = ((0, t), (-self.max_decel, 0), (0, t), (0, self.max_accel))
        x0 = [t / 3, -2, t / 3, 1]
        sol = minimize(self.objective, x0, args=(v0,), method='SLSQP',
                       bounds=bnds, constraints=cons)
        v_arrival = -sol.fun  # velocity of reach moment
        prc.print_message("\n*Optimiser result*")
        prc.print_message(f'Final speed: {v_arrival:.2f}')
        prc.print_message(f'Optimal solution: {np.round(sol.x, 2)}')  # t1, a1, t3, a3
        return sol, v_arrival

    def _plot_yield_profile(self, dis, dur_estimated,
                            v0, dur_max,
                            t1, v1,
                            t3, v_arrival):

        max_delta_t = dur_max - dur_estimated
        # Original profile:
        # maximum acceleration to max_speed, then constant speed.
        # This is consistent with estimate_travel_time().
        max_speed = self.max_speed

        if v0 < max_speed:
            t_acc = (max_speed - v0) / self.max_accel
            d_acc = v0 * t_acc + 0.5 * self.max_accel * t_acc ** 2

            if dis < d_acc:
                # Reach MS before reaching max_speed
                v_orig_arrival = v0 + self.max_accel * dur_estimated

                t_orig = np.array([0.0, dur_estimated])
                v_orig_profile = np.array([v0, v_orig_arrival])

            else:
                # Accelerate to max_speed, then cruise
                t_orig = np.array([0.0, t_acc, dur_estimated])
                v_orig_profile = np.array([v0, max_speed, max_speed])

        else:
            # Already travelling at max_speed
            t_orig = np.array([0.0, dur_estimated])
            v_orig_profile = np.array([v0, v0])

        # Max-delay profile
        if t3 > 1e-8:
            t_adj = np.array([0.0, t1, t1 + t3])
            v_adj = np.array([v0, v1, v_arrival])

        else:
            t_adj = np.array([0.0, t1])
            v_adj = np.array([v0, v_arrival])

        fig, ax = plt.subplots(figsize=(8, 5))

        # Original profile
        ax.plot(t_orig, v_orig_profile, linewidth=2,
                color='gray', label='Original profile')
        ax.fill_between(t_orig, v_orig_profile, 0,
                        color='gray', alpha=0.20)

        # Max-delay profile
        ax.plot(t_adj, v_adj, linewidth=2,
                color='blue', label='Max-delay profile')
        ax.fill_between(t_adj, v_adj, 0,
                        color='blue', alpha=0.18)

        # Arrival times
        ax.axvline(dur_estimated, color='gray',
                   linestyle='--', linewidth=1.3)
        ax.axvline(dur_max, color='blue',
                   linestyle='--', linewidth=1.3)

        # Minimum process speed
        ax.scatter([t1], [v1], color='blue', zorder=5)
        ax.annotate(
            f'Min. process speed = {v1:.2f} m/s',
            xy=(t1, v1), xytext=(t1 + 0.5, v1 + 2.0),
            arrowprops=dict(arrowstyle='->', color='black'))

        # Arrival speed
        ax.scatter([dur_max], [v_arrival],
                   color='blue', zorder=5)
        ax.annotate(
            f'Arrival speed = {v_arrival:.2f} m/s',
            xy=(dur_max, v_arrival),
            xytext=(dur_max - 4.0, v_arrival + 2.0),
            arrowprops=dict(arrowstyle='->', color='black'))

        # Maximum additional time
        y_arrow = max(np.max(v_orig_profile), v0, v_arrival) + 2.0

        ax.annotate(
            '', xy=(dur_estimated, y_arrow),
            xytext=(dur_max, y_arrow),
            arrowprops=dict(
                arrowstyle='<->', color='red',
                linewidth=1.5))

        ax.text(
            (dur_estimated + dur_max) / 2,
            y_arrow + 0.4,
            rf'$\Delta t_{{max}}$ = {max_delta_t:.2f} s',
            ha='center', color='red')

        ax.text(
            0.03, 0.98,
            f'Distance = {dis:.1f} m\n'
            f'Estimated duration = {dur_estimated:.2f} s\n'
            f'Max. duration = {dur_max:.2f} s',
            transform=ax.transAxes, va='top')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')

        ax.set_xlim(0, max(dur_estimated, dur_max) * 1.08)
        ax.set_ylim(0, y_arrow + 2.0)

        ax.grid(alpha=0.25)
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.show()

    def get_max_delta_t(self, v0, dis, dur_estimated=None,
                        plot=False):
        """
        Calculate the maximum additional travel time that can be assigned
        to a mainline leader.

        Assumptions:
            1. Stage 1: deceleration with a1 >= -max_decel.
            2. Stage 3: acceleration with a3 <= max_accel.
            3. No constant-speed Stage 2.
            4. Process speed cannot fall below min_speed.
            5. Arrival speed must be at least
               arrival_speed_ratio * max_speed.

        Parameters
        ----------
        dis : float
            Remaining distance to the merging section.
        v0 : float
            Current speed.
        dur_estimated : float
            Original expected travel duration.
        min_speed : float
            Minimum allowed process speed.
        arrival_speed_ratio : float
            Minimum arrival speed ratio relative to max_speed.

        Returns
        -------
        max_delta_t : float
            Maximum additional travel duration.
        """
        if dur_estimated is None:
            dur_estimated = self._estimate_travel_time(v0, dis)

        min_arrival_speed = self.arrival_speed_ratio * self.max_speed
        if v0 < self.min_speed:
            return 0.0

        # Case 1:
        # Distance is too short to decelerate from v0 to
        # min_arrival_speed, even using maximum deceleration.
        # Only a deceleration stage is used.
        if v0 > min_arrival_speed:
            t1 = (v0 - min_arrival_speed) / self.max_decel
            d1 = v0 * t1 - 0.5 * self.max_decel * t1 ** 2

            if dis <= d1:
                v_arrival = np.sqrt(v0 ** 2 - 2 * self.max_decel * dis)
                t1 = (v0 - v_arrival) / self.max_decel
                t3 = 0.0
                v1 = v_arrival
                dur_max = t1
                max_delta_t = max(0.0, dur_max - dur_estimated)
                if plot:
                    self._plot_yield_profile(
                        dis, dur_estimated, v0, dur_max,
                        t1, v1, t3, v_arrival)
                return max_delta_t
        # If v0 < min_arrival_speed, first check whether the vehicle
        # can recover to min_arrival_speed before reaching MS.
        if v0 < min_arrival_speed:
            d_acc = (min_arrival_speed ** 2 - v0 ** 2) / (2 * self.max_accel)
            if dis < d_acc:
                return 0.0

        # Distance required for:
        # v0 -> min_speed using maximum deceleration
        # min_speed -> min_arrival_speed using maximum acceleration
        t1_min = (v0 - self.min_speed) / self.max_decel
        d1_min = v0 * t1_min - 0.5 * self.max_decel * t1_min ** 2
        t3_max = (min_arrival_speed - self.min_speed) / self.max_accel
        d3_max = self.min_speed * t3_max + 0.5 * self.max_accel * t3_max ** 2

        # Case 2:
        # Distance is insufficient to reach min_speed.
        # Maximum deceleration followed by maximum acceleration.
        # Solve the lowest feasible intermediate speed v1.
        if dis < d1_min + d3_max:
            v1_sq = ((self.max_accel * v0 ** 2
                     + self.max_decel * min_arrival_speed ** 2
                     - 2 * self.max_accel * self.max_decel * dis)
                     / (self.max_accel + self.max_decel))
            v1 = np.sqrt(max(0.0, v1_sq))
            t1 = (v0 - v1) / self.max_decel
            t3 = (min_arrival_speed - v1) / self.max_accel
            dur_max = t1 + t3

        # Case 3:
        # Distance is sufficient to reach min_speed.
        # Maximum deceleration to min_speed, followed by a constant
        # acceleration that reaches min_arrival_speed exactly at MS.
        else:
            v1 = self.min_speed
            t1 = (v0 - v1) / self.max_decel
            d1 = v0 * t1 - 0.5 * self.max_decel * t1 ** 2
            d3 = dis - d1
            t3 = 2 * d3 / (v1 + min_arrival_speed)
            dur_max = t1 + t3
        max_delta_t = max(0.0, dur_max - dur_estimated)
        if plot:
            self._plot_yield_profile(
                dis, dur_estimated, v0, dur_max,
                t1, v1, t3, min_arrival_speed)
        # print(f'max_delta_t: {max_delta_t:.2f} s')
        return max_delta_t

    def _estimate_travel_time(self, v0, D): # get_ts_a
        """
        estimate travel time according to remaining distance and current speed
        :param v0: current speed
        :param D: distance
        :return: travel time
        """
        max_speed = self.max_speed
        t_acc = (max_speed - v0)/self.max_accel # reach max_speed's spending time
        d_acc = v0*t_acc + 0.5*self.max_accel*t_acc**2 # reach max_speed's needing dis
        if D == d_acc:
            travel_time = t_acc
        elif D < d_acc:
            # a bug here already fixed: self.amax*d_acc => self.amax*D
            travel_time = (-v0 + math.sqrt(v0**2 + 2 * self.max_accel * D)) / self.max_accel
        else: # D > d_acc
            t_uni = (D-d_acc)/max_speed
            travel_time = t_uni+t_acc
        return travel_time

if __name__ == '__main__':
    optm = GetBVCurve2(arrival_speed_ratio=1.0)
    optm.get_max_delta_t(v0=20,
                         dis=200,
                         plot=True)
    r, v_arrival = optm.optimise(v0=20,
                                 dis=200,
                                 t=8.39) # r.x = [t1, a1 (dec), t3, a3 (acc)]

