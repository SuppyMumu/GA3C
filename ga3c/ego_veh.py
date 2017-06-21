import os,sys,time
pl_tbx_root = '/home/etienneperot/workspace/pl_tbx/src-build/pl_tbx_python/'
sys.path.append(pl_tbx_root)
import libpl_tbx_python as rec
import numpy as np
from bicycle_relative import *

K_steer_to_wheel = 15

#Units are SI (rad, m, s)
class EgoVehicle:
    def __init__(self, first_time=0, wheelbase = 2.7):
        self.speed = 0
        self.len_front = 0
        self.len_rear = wheelbase
        self.delta_x = 0
        self.delta_y = 0
        self.delta_phi = 0
        self.ref_wheel_angle = 0
        self.ref_speed = 0
        self.ego_wheel_angle = 0
        self.ego_speed = 0
        self.time = first_time
        self.dt = 0
        self.blinker = 0

        #to parse rec for passat bob3
        self.selection = [
            'LW1_LRW',          # steering angle
            'LW1_LRW_Sign',     # sign(steering angle)
            'BR8_Laengsbeschl', # accel longi
            'BR1_Lichtschalt',  # brake can
            'BR1_Rad_kmh',      # speed
            'GK1_Blinker_li',   # blinker left
            'GK1_Blinker_re',   # blinker right
            'BR5_Giergeschw',   # yaw rate
            'BR5_Vorzeichen',   # sign(yaw_rate)
        ]

    def reset_delta(self):
        self.delta_x = 0
        self.delta_y = 0
        self.delta_phi = 0

    def fill_ref(self, recmovie, time):
        self.dt = (time  - self.time) * 1e-6
        self.time = time
        try:
            self.blinker = recmovie.getCanData(time, "GK1_Blinker_li") or recmovie.getCanData(time, "GK1_Blinker_re")
            self.ref_speed = recmovie.getCanData(time, "BR1_Rad_kmh") / 3.6 #kmh -> mps
            self.ref_steering_deg = recmovie.getCanData(time, "LW1_LRW")
            self.ref_steering_rad = self.ref_steering_deg * (np.pi /180)
            sign = recmovie.getCanData(time, "LW1_LRW_Sign")
            if sign == 1:
                self.ref_steering_deg *= -1
                self.ref_steering_rad *= -1
            self.ref_wheel_angle = self.ref_steering_rad / K_steer_to_wheel


        except Exception as inst:
            print(inst)


    def update_with_steer_angle(self, steer_rad):
        self.ego_wheel_angle = steer_rad / K_steer_to_wheel

        #print('DIFF EGO / REF Steer ', steer_rad - self.ref_steering_rad)
        #print('DIFF EGO / REF ', self.ego_wheel_angle-self.ref_wheel_angle)

        dx_old, dy_old, dphi_old = self.delta_x, self.delta_y, self.delta_phi

        dx, dy, dphi = bicycle_relative_integrate(dx_old, dy_old, dphi_old,
                                                  self.ego_wheel_angle,
                                                  self.ref_wheel_angle,
                                                  self.ref_speed,
                                                  self.ref_speed,
                                                  self.len_front,
                                                  self.len_rear,
                                                  self.dt)
        self.delta_x = dx
        self.delta_y = dy
        self.delta_phi = dphi
