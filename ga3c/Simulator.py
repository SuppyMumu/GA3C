import os,sys,time as timer
pl_tbx_root = '/home/etienneperot/workspace/pl_tbx/src-build/pl_tbx_python/'
gcam_root = '/home/etienneperot/workspace/proj1356_camera_utils/build/'
sys.path.append(pl_tbx_root)
sys.path.append(gcam_root)
import libpl_tbx_python as rec
import libgcam as gcam

import ego_veh

import numpy as np
import cv2
import copy

import tensorflow as tf

sys.path.append('../NvidiaLike')
import model


def preprocess_cameras(im_front, im_right, im_left, rotation_deg):
	fish_eye_fov = 170
	camera_fov = 90
	fisheyeHeight = 400
	fisheyeWidth = 1200

	cameraWidth = (fisheyeWidth * camera_fov) / fish_eye_fov

	frontCameraHeight = 160
	rightCameraHeight = 220
	leftCameraHeight = 220

	finalCameraHeight = 66
	finalCameraWidth = 200

	start_left_pixel_number = int((fisheyeWidth / fish_eye_fov) * ((fish_eye_fov - camera_fov) / 2 - rotation_deg))
	#print("start_left_pixel_number = ", start_left_pixel_number)

	im_front = im_front[fisheyeHeight - frontCameraHeight::,start_left_pixel_number:start_left_pixel_number + cameraWidth, :]
	im_right = im_right[fisheyeHeight - rightCameraHeight::,start_left_pixel_number:start_left_pixel_number + cameraWidth, :]
	im_left = im_left[fisheyeHeight - leftCameraHeight::, start_left_pixel_number:start_left_pixel_number + cameraWidth,:]

	im_front = cv2.resize(im_front, (finalCameraWidth, finalCameraHeight), interpolation=cv2.INTER_AREA)
	im_right = cv2.resize(im_right, (finalCameraWidth, finalCameraHeight), interpolation=cv2.INTER_AREA)
	im_left = cv2.resize(im_left, (finalCameraWidth, finalCameraHeight), interpolation=cv2.INTER_AREA)

	return im_front, im_right, im_left

def show_flow_diff(A,B):
	Ag = cv2.cvtColor(A,cv2.COLOR_BGR2GRAY)
	Bg = cv2.cvtColor(B,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(A)
	hsv[..., 1] = 255
	flow = cv2.calcOpticalFlowFarneback(Ag, Bg, 0.5, 3, 15, 3, 5, 1.2, 0)
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return bgr

def run(recmovie, net):
	from scipy import signal
	period = 100
	freq = 10
	t = np.linspace(0, 1, period)
	triangle = signal.sawtooth(2 * np.pi * freq * t, 0.5)

	frame0 = 5500
	time = recmovie.getTimeByFrameRef(frame0)

	car_ref = ego_veh.EgoVehicle(first_time=time, wheelbase = 2.7)

	car = ego_veh.EgoVehicle(first_time=time, wheelbase = 2.7)

	front_movie_number = 1  # We use the front camera movie as reference
	right_movie_number = 0
	left_movie_number = 3
	old_time = time
	for i in range(frame0+1,recmovie.nbFrames(0),1):
		time = recmovie.getTimeByFrameRef(i)
		car.fill_ref(recmovie, time)
		if car.blinker:
			print('Warning: Blinking! (maybe a lane change)')
			continue

		# (left,front,rear,right)
		im_front_raw = recmovie.getImageByFrameIndex(front_movie_number,i)
		im_right_raw = recmovie.getImageByFrameIndex(right_movie_number,i)
		im_left_raw = recmovie.getImageByFrameIndex(left_movie_number,i)

		start = timer.time()
		#car.delta_x = 0
		delta = (car.delta_x, car.delta_y, car.delta_phi)
		trans_front, trans_right, trans_left = cocoon.move_cam(im_front_raw, im_right_raw, im_left_raw, delta)
		real_front, real_right,  real_left = cocoon.proj_cam(im_front_raw, im_right_raw, im_left_raw)


		degrees = car.delta_phi * 180 / np.pi
		tf, tr, tl = preprocess_cameras(trans_front, trans_right, trans_left, degrees)
		#print((timer.time() - start) * 1000, ' ms @cam lutting')

		start = timer.time()
		predict, vizu = dnn.run(tf, tr, tl)
		steer_net_rad_predict = predict[0,0]
		vizu = vizu[0]
		#print((timer.time() - start)*1000, ' ms')

		m1, m2 = vizu.min(), vizu.max()
		vizu = (vizu - m1) / (m2 - m1) * 255.0
		vizu = vizu.astype(np.uint8)

		vf, vr, vl = vizu[0], vizu[1], vizu[2]

		gb = cocoon.display_cams(vf, vr, vl, resize=1.0)


		#steer_net_rad_predict = car.ref_steering_rad - 0.1
		steer_deg_predict = steer_net_rad_predict * 180.0 / np.pi

		#print('net_predict_degree', steer_deg_predict, 'car_angle_degree', car.ref_steering_deg)
		#print('DELTA ANGLE = ', steer_deg_predict-car.ref_steering_deg)

		car.update_with_steer_angle(steer_net_rad_predict)
		print('Delta: ', car.delta_x, car.delta_y, car.delta_phi)


		if abs(car.delta_y) > 1:
			life_time = (time-old_time) * 1e-6
			old_time = time
			car.reset_delta()
			print('+==============================+')
			print('car is too far!!, reset delta')
			print('car lived during ',str(life_time), ' seconds! ')
			print('+==============================+')

		#display
		simulation = cocoon.display_cams(trans_front,trans_right,trans_left)
		reality = cocoon.display_cams(real_front,real_right,real_left)
		diff = show_flow_diff(simulation, reality)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(simulation, 'dy: '+str(car.delta_y), (10, simulation.shape[0]-20), font, 0.75, (0, 0, 255), 2)

		#cv2.imshow('raw', display_total)
		cv2.imshow('gb', gb)
		cv2.imshow('simulation',simulation)
		#cv2.imshow('reality',reality)
		#cv2.imshow('diff', diff)
		cv2.waitKey(10)


run(rec, None)