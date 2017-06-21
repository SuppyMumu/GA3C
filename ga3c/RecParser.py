#====================#
#					 #
#		pltbx        #
#					 #
#====================#
import os,sys,time as timer
pl_tbx_root = '/home/etienneperot/workspace/pl_tbx/src-build/pl_tbx_python/'
sys.path.append(pl_tbx_root)
import libpl_tbx_python as rec
import numpy as np

dbc_passat_bob3 = "/home/etienneperot/libs/DAR_DEMOCAR/trunk/passat_bob_3/dbc/can_vehicle_passat.dbc"

selection = [
				'LW1_LRW', 					#steering angle
				'LW1_LRW_Sign',				#sign(steering angle) 
				'BR8_Laengsbeschl',  		#accel longi
				'BR1_Lichtschalt', 	  		#brake can
				'BR1_Rad_kmh', 				#speed
				'GK1_Blinker_li', 			#blinker left
				'GK1_Blinker_re', 			#blinker right
				'BR5_Giergeschw', 			#yaw rate
				'BR5_Vorzeichen', 			#sign(yaw_rate)
			]

dic = 	{ 		
			selection[0]:'steering angle',
			selection[1]:'steering angle (sign)',
			
			selection[2]:'accel longi',
			selection[3]:'brake can',
			selection[4]:'speed',

			selection[5]:'blinker left',
			selection[6]:'blinker right',

			selection[7]:'yaw rate',
			selection[8]:'yaw rate (sign)'
		}

can_channel = 'Can_Car.output'



if __name__ == '__main__':
	recfile = "/home/etienneperot/Videos/20150612_140542_RecFile_1/RecFile_1_20150612_140542.rec"
	rec = rec.RecMovies(recfile)
	rec.setCanReading(can_channel, dbc_passat_bob3, selection)
