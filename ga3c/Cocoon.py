#====================#
#					 #
#		 gcam        #
#					 #
#====================#
import os,sys,time as timer
gcam_root = '/home/etienneperot/workspace/proj1356_camera_utils/build/'
sys.path.append(gcam_root)
import libgcam as gcam
import numpy as np
import cv2

def cam_make_lut(cam, delta=(0,0,0)):
	h = cam.imgHeight()
	w = cam.imgWidth()
	lut = np.zeros((h, w, 2), np.float32)
	cam.makelut_delta(lut, delta, False)
	return lut

def proj_cam(image, lut):
	h = lut.shape[0]
	w = lut.shape[1]
	u = lut[..., 0]
	v = lut[..., 1]
	im_out = np.zeros((h, w, 3), dtype=np.uint8)
	cv2.remap(image, u, v, cv2.INTER_CUBIC, im_out)
	return im_out

class Cocoon:
	def __init__(self,cam_file_dir):
		cam_file_right = cam_file_dir + 'sample_config_c4_tjc_right.yml' 
		cam_file_rear = cam_file_dir + 'sample_config_c4_tjc_rear.yml'
		cam_file_front = cam_file_dir + 'sample_config_c4_tjc_front.yml'
		cam_file_left = cam_file_dir + 'sample_config_c4_tjc_left.yml'

		self.cam_front = gcam.GCam()
		self.cam_right = gcam.GCam()
		self.cam_left = gcam.GCam()

		self.cam_front.init_forward(cam_file_front)
		self.cam_right.init_forward(cam_file_right)
		self.cam_left.init_forward(cam_file_left)

		self.lut_front = cam_make_lut(self.cam_front)
		self.lut_left = cam_make_lut(self.cam_left)
		self.lut_right = cam_make_lut(self.cam_right)


	def gen_dy_luts(self, min=-1.0, max=1.0, step=0.01):
		h = self.cam_front.imgHeight()
		w = self.cam_front.imgWidth()
		r = max-min
		n = int( r/step )
		#front, left, right
		mega_lut = np.zeros((n, 3, h, w, 2), dtype=np.float32)
		for i in range(0, n):
			dy = min + step * i
			mega_lut[i,0] = cam_make_lut(self.cam_front, delta=(0,dy,0))
			mega_lut[i,1] = cam_make_lut(self.cam_left, delta=(0, dy, 0))
			mega_lut[i,2] = cam_make_lut(self.cam_right, delta=(0, dy, 0))
		np.save('mega_lut',mega_lut)

	def load_dy_luts(self):
		self.megalut = np.load('mega_lut.npy')

	def find_lut(self, dy):
		s = self.megalut.shape[0]
		max = 1
		min = -1
		a = (max-min)/float(s)
		i = int( (dy - min) / a )
		return self.megalut[i]

	def proj_cam(self, image_front, image_right, image_left, pps=False):
		im_front = proj_cam(image_front, self.lut_front)
		im_left = proj_cam(image_left, self.lut_left)
		im_right = proj_cam(image_right, self.lut_right)

		if pps:
			im_front, im_right, im_left = preprocess_cameras(im_front, im_right, im_left, 0)
		return im_front, im_right, im_left

	def move_cam(self, image_front, image_right, image_left, delta, pps=False):
		dx, dy, dphi = delta
		if hasattr(self, 'megalut'):
			luts = self.find_lut(dy)
			lut_front = luts[0]
			lut_left = luts[1]
			lut_right = luts[2]
		else:
			lut_front = cam_make_lut(self.cam_front, (dx,dy,0))
			lut_left = cam_make_lut(self.cam_left, (dx,dy,0))
			lut_right = cam_make_lut(self.cam_right, (dx,dy,0))
		im_front = proj_cam(image_front, lut_front)
		im_left = proj_cam(image_left, lut_left)
		im_right = proj_cam(image_right, lut_right)

		if pps:
			degrees = dphi * 180 / np.pi
			im_front, im_right, im_left = preprocess_cameras(im_front, im_right, im_left, degrees)
		return im_front, im_right, im_left

	def display_cams(self, image_front, image_right, image_left, resize=0.25):
		h,w,c = image_front.shape
		hh = int(h * resize)
		ww = int(w * resize)
		imf = cv2.resize(image_front, (ww,hh))
		imr = cv2.resize(image_right, (ww, hh))
		iml = cv2.resize(image_left, (ww, hh))

		h, w, c = imf.shape
		display = np.zeros((h,w*3,3), np.uint8)
		display[:,:w] = iml
		display[:,w:w*2] = imf
		display[:, w*2:w*3] = imr
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(display, 'left', (10, 10), font, 0.5, (0, 0, 255), 1)
		cv2.putText(display, 'front', (w+10, 10), font, 0.5, (0, 0, 255), 1)
		cv2.putText(display, 'right', (2*w+10, 10), font, 0.5, (0, 0, 255), 1)

		display = cv2.pyrUp(display)
		return display

if __name__ == '__main__':
	cam_file_dir = '/home/etienneperot/libs/DAR_DEMOCAR/trunk/passat_bob_3/camera/'
	cocoon = Cocoon(cam_file_dir)
	print('GENERATE LUTS')
	cocoon.gen_dy_luts(-1,1,0.01)
	print('done!')