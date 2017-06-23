from Cocoon import *
from RecParser import *
import ego_veh
import numpy as np
import cv2

front_movie_number = 1  # We use the front camera movie as reference
right_movie_number = 0
left_movie_number = 3
cam_file_dir = '/home/etienneperot/libs/DAR_DEMOCAR/trunk/passat_bob_3/camera/'

def sigmoid(x):
	return 1./ (1. + np.exp(x))

class DrivingSimulator:
	def __init__(self, rec_directory):
		self.dir = rec_directory
		recs = [x[0] for x in os.walk(dir)]
		recfile = recs[np.random.randint(0,len(recs))]
		self.recmovie = rec.RecMovies(recfile)
		self.recmovie.setCanReading(can_channel, dbc_passat_bob3, selection)
		self.nb_frames = self.recmovie.nbFrames(0)
		self.car = ego_veh.EgoVehicle(first_time=0, wheelbase=2.7)
		self.cocoon =  Cocoon(cam_file_dir)
		self.skipframes = 1

	def reset(self):
		self.index = np.random.randint(0, self.nb_frames) #get it from recfile
		self.time = self.recmovie.getTimeByFrameRef(self.index)
		self.car.time = self.time
		self.car.reset_delta()

	def step(self, action):
		time = self.recmovie.getTimeByFrameRef(self.index)
		self.car.fill_ref(self.recmovie, time)
		self.car.update_with_steer_angle(action)

		#compute
		im_front_raw = self.recmovie.getImageByFrameIndex(front_movie_number, self.index)
		im_right_raw = self.recmovie.getImageByFrameIndex(right_movie_number, self.index)
		im_left_raw = self.recmovie.getImageByFrameIndex(left_movie_number, self.index)

		delta = (self.car.delta_x, self.car.delta_y, self.car.delta_phi)
		self.trans_front, self.trans_right, self.trans_left = self.cocoon.move_cam(im_front_raw, im_right_raw, im_left_raw, delta)
		degrees = self.car.delta_phi * 180 / np.pi
		tf, tr, tl = DrivingSimulator.preprocess_cameras(self.trans_front, self.trans_right, self.trans_left, degrees)

		h,w,c = tf.shape
		obs = np.zeros((3,h,w,c))
		obs[0] = tf
		obs[1] = tl
		obs[2] = tr

		penalty = sigmoid(abs(self.car.delta_y))
		reward = 1 - penalty
		info = {}
		done = self.car.delta_y > 1.0
		self.index += self.skipframes
		return obs, reward, info, done

	def render(self):
		#plot trajectory

		#plot cameras
		display = self.cocoon.display_cams(self.trans_front,self.trans_right,self.trans_left)
		cv2.imshow('simu',display)
		cv2.waitKey(5)

	@staticmethod
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
		# print("start_left_pixel_number = ", start_left_pixel_number)

		im_front = im_front[fisheyeHeight - frontCameraHeight::,
				   start_left_pixel_number:start_left_pixel_number + cameraWidth, :]
		im_right = im_right[fisheyeHeight - rightCameraHeight::,
				   start_left_pixel_number:start_left_pixel_number + cameraWidth, :]
		im_left = im_left[fisheyeHeight - leftCameraHeight::,
				  start_left_pixel_number:start_left_pixel_number + cameraWidth, :]

		im_front = cv2.resize(im_front, (finalCameraWidth, finalCameraHeight), interpolation=cv2.INTER_AREA)
		im_right = cv2.resize(im_right, (finalCameraWidth, finalCameraHeight), interpolation=cv2.INTER_AREA)
		im_left = cv2.resize(im_left, (finalCameraWidth, finalCameraHeight), interpolation=cv2.INTER_AREA)

		return im_front, im_right, im_left

if __name__ == '__main__':
	dir = '/media/etienneperot/TOSHIBA EXT/10000km/'
	d = DrivingSimulator(dir)