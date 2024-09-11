#!/usr/bin/env python3

#!/usr/bin/bash

import matplotlib.pyplot as plt
import shutil
from osgeo import gdal, osr

import subprocess
from custom_msg_python.msg import custom




from cgi import print_directory
from ctypes.wintypes import PCHAR
from tkinter import N
#from msilib.schema import SelfReg
from typing_extensions import Self
import csv
from edrone_client.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time
import tf
import roslib
from time import process_time
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import argparse
import os
import pandas 



class Edrone():
	
	def callback(self,data):
 
 		 # Used to convert between ROS and OpenCV images
		br = CvBridge()
		kernel = np.ones((5, 5), np.uint8)
  # Output debugging information to the terminal
		#rospy.loginfo("receiving video frame")
   
  # Convert ROS Image message to OpenCV image
		image = br.imgmsg_to_cv2(data)

		

		
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		orignal = image.copy()
		

		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		

		lower = np.array([26, 70, 0], dtype="uint8") #22,93,0     #lower threshold size for yellow box detected
		upper = np.array([30, 255, 255], dtype="uint8")           #upper threshold size for yellow box detected
		mask = cv2.inRange(image, lower, upper)
		mask = cv2.erode(mask, kernel, iterations=1)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.dilate(mask, kernel, iterations=1)

		(height,width) = image.shape[:2] #w:image-width and h:image-height
		
		center_x = height/2
		center_y = width/2
		#print(center_x,center_y)
		
		cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		
		
		for c in cnts:

			x1,y,w,h = cv2.boundingRect(c)
			cv2.rectangle(orignal, (x1, y), (x1 + w, y + h), (36,255,12), 0)
			area = h*w
			
			middle_x = (x1+x1+w)/2
			middle_y = (y+y+h)/2
			midpoint = [middle_x,middle_y]
			dist = ((int(middle_x) - width/2)**2 + (int(middle_y) - height/2)**2)**0.5
			
			
			# complete box visible from top 
			# note that due to perspective error, when drone is directly on top of the box, only then it falls within the threshold

			if 3850>area >3720 : 

				cv2.imwrite('/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/sframe%d.jpg' %self.count,orignal)
		
				print('img saved',self.count)

				self.box_pix.append(midpoint)		
				self.distance.append(int(dist))
				self.count += 1
	
					
			else:
				pass


		#if the box is found within the threshold

		while self.success:

			print(self.distance)
			shortest = min(self.distance)
			print(shortest)
			index= self.distance.index(int(shortest)) + int(self.last_len)
			print('last_len',self.last_len)
			print(self.distance.index(int(shortest)),'+',int(self.last_len))

			self.block_midpoint = self.box_pix[index]
			distances =  self.distance
			print('index to keep:',index)
			print(int(shortest),'shortest dist')
			

		# to create a buffer, a serires of images are clicked and stored 
		# so the best image i.e. image taken from top can be chosen to find the coordinates of the centre of box

			for j in range(int(len((distances)))):
				
				if self.nn == index:
					print('saved image sfram',self.nn)
					self.nn+=1

				else: 
					os.remove('/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/sframe%d.jpg'%self.nn)
					if self.nn == int(len(distances))+int(self.last_len):
						print('end searching')
						print('updated_last_len',self.last_len)
						break
					else:
						self.nn=self.nn+1
			
			
			#the best clicked image is then used for feature matching using SIFT and locate it with respect to TIF image
			self.last_len += int(len(distances)) 
			print('updated_last_len',self.last_len)
			
			img11 = cv2.imread(f"/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/sframe{index}.jpg", cv2.IMREAD_GRAYSCALE)
			
			img1 = cv2.normalize(img11, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
			img2 = cv2.imread("/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/task2d.tif", cv2.IMREAD_GRAYSCALE)

			#cv2.imshow("sat", img2)
			sift = cv2.SIFT_create()

			kp1,des1 = sift.detectAndCompute(img1, None)
			kp2,des2 = sift.detectAndCompute(img2, None)

			#brute force
			print(len(kp1),len(kp2))
			bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

			matches = bf.match(des1,des2)


			#print(len(matches))

			matches = sorted(matches, key = lambda x:x.distance)

			matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:70], img2, flags=2)

			scale_percent = 25 # percent of original size
			width = int(matching_result.shape[1] * scale_percent / 100)
			height = int(matching_result.shape[0] * scale_percent / 100)
			dim = (width, height)

			resized = cv2.resize(matching_result, dim, interpolation = cv2.INTER_AREA)

			# Initialize lists
			ist_kp1 = []
			ist_kp2 = []

			list_kp1 = []
			list_kp2 = []

			# For each match...
			for mat in matches:

				# Get the matching keypoints for each of the images
				img1_idx = mat.queryIdx
				img2_idx = mat.trainIdx

				# x - columns
				# y - rows
				# Get the coordinates
				p1 = kp1[img1_idx].pt
				p2 = kp2[img2_idx].pt
				
				

				# Append to each list
				ist_kp1.append(p1)
				list_kp1 = [item for t in ist_kp1 for item in t]
				ist_kp2.append(p2)
				list_kp2 = [item for t in ist_kp2 for item in t]

			#print(ist_kp1)
			print('kp2:',len(list_kp2))

			# Open tif file
			ds = gdal.Open('/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/task2d.tif')
			# GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
			xoff, a, b, yoff, d, e = ds.GetGeoTransform()
			
			def pixel2coord(x, y):
				"""Returns global coordinates from pixel x, y coords"""
				xp = a * x + b * y + xoff
				yp = d * x + e * y + yoff
				return(xp, yp)
			
			# get columns and rows of your image from gdalinfo
			rows = 3994+1
			colms = 4001+1

			ist_coord = []  
			list_coord = []


			for i in range(0,len(list_kp2),2):
				ist_coord.append(pixel2coord(list_kp2[i],list_kp2[i+1]))
				list_coord = [item for t in ist_coord for item in t]


			ds= None

			print('coord:',len(list_coord))

			_gcps=[]
			gcps=[]
			# Enter the GCPs
			#   Format: [map x-coordinate(longitude)], [map y-coordinate (latitude)], [elevation],
			#   [image column index(x)], [image row index (y)]

			def gcp(i):
			
				x=(f'{list_kp1[i]}', f'{list_kp1[i+1]}', f'{list_coord[i]}', f'{list_coord[i+1]}')
				gcps.append(x)
				
			for i in range(0,30,2):
				gcp(i)

			georef = ['gdal_translate' ]
			#print((gcps))

			n=0
			for j in range(int(len(gcps))+1):
				if j < len(gcps):
					n+=1
					georef.append('-gcp')
					for p in range(4):

						georef.append(gcps[j][p])
				else:
					#georef.append(gcps[j])
					
					georef.append('-of')
					georef.append('GTiff')
					georef.append(f'/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/sframe{index}.jpg')
					georef.append(f'/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/sframe{index}f.tif')
				
			print(len(georef))

			subprocess.run(georef)

			SRC_METHOD= 'NO_GEOTRANSFORM'

			from_SRS = "EPSG:4326"

			to_SRS = "EPSG:4326"

			src=f'/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/sframe{index}f.tif'

			dest= f'/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/sframe{index}fin.tif'

			cmd_list = ["gdalwarp","-r", 'bilinear', "-s_srs", from_SRS, "-t_srs", to_SRS, "-overwrite", src, dest]

			subprocess.run(cmd_list)

			print('searching')

			ds = gdal.Open(f'/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/sframe{index}fin.tif')
			# GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
			xoff, a, b, yoff, d, e = ds.GetGeoTransform()
			
			def pixel2coord(x, y):
				"""Returns global coordinates from pixel x, y coords"""
				xp = a * x + b * y + xoff
				yp = d * x + e * y + yoff
				self.long_lat.append([xp, yp])
			
			# get columns and rows of your image from gdalinfo
			rows = 632+1
			colms = 593+1
			print(self.block_midpoint)
			please=pixel2coord(self.block_midpoint[0],self.block_midpoint[1])
			
			self.x[0] = str(f"Obj{self.obj}")
			self.x[1] = float(self.long_lat[0][0])
			self.x[2] = float(self.long_lat[0][1])

			geo=self.x.copy()

			self.rows.append(geo)

			print('csv file',self.rows)
			
			self.msg.lat.data = self.x[2]
			self.msg.long.data = self.x[1]
			self.msg.objectid.data = self.x[0]
			
			# name of csv file 
			filename = "/home/piyaansh/catkin_ws/src/sentinel_drone/sentinel_drone/scripts/geolocation.csv"
				
			# writing to csv file 
			with open(filename, 'w') as csvfile: 
				# creating a csv writer object 
				csvwriter = csv.writer(csvfile) 
					
				# writing the fields 
				#csvwriter.writerow(fields) 
					
				# writing the data rows 
				csvwriter.writerows(self.rows)
			print('FINAL ANSWER:',self.long_lat)

			#reset the variables to default for next yellow box
			ds= None
			self.block_midpoint.clear()
			self.long_lat.clear()
			self.distance.clear()
			self.obj+= 1
			self.success = False
			
		cv2.imshow('orignal', orignal)
		cv2.waitKey(1)
   
	def __init__(self):
		
		rospy.init_node('drone_control')	# initializing ros node with name drone_control

		# This corresponds to your current position of drone. This value must be updated each time in your whycon callback
		# [x,y,z]
		self.drone_position = [0.0,0.0,0.0]	
		#self.drone_position = [self.x, self.y, self.z]
		# [x_setpoint, y_setpoint, z_setpoint]
		

		#boundary points of whycon camera's field of view
		self.setpoint_list=[[0,0,23],[0,0,23],[0,0,23],[0,0,23],[0,0,23],[0,0,23],[0,0,23],[2,0,23],[2,2,23],[-2,2,23],[-2,-2,23],[2,-2,23],[2,0,23],[0,0,23]]

		#navigates drone to center of the whycon camera's field of view
		self.waypoint=[10.2,10.2,20]
		
		self.blockpoint = [0,0,20.5]

		self.distance = []

		self.cnts=[]

		self.success = False

		self.last_len = 0

		self.box_pix =[]

		self.long_lat= []
		
		#Declaring a cmd of message type edrone_msgs and initializing values
		self.cmd = edrone_msgs()
		self.cmd.rcRoll = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcThrottle = 1500
		self.cmd.rcAUX1 = 1500
		self.cmd.rcAUX2 = 1500
		self.cmd.rcAUX3 = 1500
		self.cmd.rcAUX4 = 1500


		#initial setting of Kp, Kd and ki for [roll, pitch, throttle]. eg: self.Kp[2] corresponds to Kp value in throttle axis
		#after tuning and computing corresponding PID parameters, change the parameters
		self.Kp = [12.3,12.3,13.6]#55.96]
		self.Ki = [0,0,0.8] #0.001
		self.Kd = [350,350,390]#1453.1]

		self.error = [0,0,0]
		self.prev_error = [0,0,0]
		self.sum_error = [0,0,0]
		self.delta_error = [0,0,0]

		self.msg = custom()

		self.msg.objectid.data = 'none'
		self.msg.long.data= 0
		self.msg.lat.data= 0

		self.x = [0,0,0]
		self.block_midpoint = []
		self.n= True 

		self.block= False
        

		self.max_values = [1800,1800,1800]
		self.min_values = [1200,1200,1200]
		

		# # This is the sample time in which you need to run pid. Choose any time which you seem fit. Remember the stimulation step time is 50 ms
		self.sample_time = 0.050 # in seconds

		self.count= 0
		self.nn= 0
		
		self.rows=[]
		self.obj=0



		# Publishing /drone_command, /alt_error, /pitch_error, /roll_error
		self.command_pub = rospy.Publisher('/drone_command', edrone_msgs, queue_size=1)
		self.throttle_error_pub = rospy.Publisher('/alt_error', Float64, queue_size=1)
		self.roll_error_pub = rospy.Publisher('/roll_error', Float64, queue_size=1)
		self.pitch_error_pub = rospy.Publisher('/pitch_error', Float64, queue_size=1)
		self.geolocation_pub = rospy.Publisher('geolocation', custom , queue_size=1)

		# Subscribing to /whycon/poses, /pid_tuning_altitude, /pid_tuning_pitch, pid_tuning_roll
		rospy.Subscriber('/whycon/poses', PoseArray, self.whycon_callback)
		rospy.Subscriber('/pid_tuning_altitude',PidTune,self.altitude_set_pid)
		rospy.Subscriber('/pid_tuning_pitch',PidTune,self.pitch_set_pid)
		rospy.Subscriber('/pid_tuning_roll',PidTune,self.roll_set_pid)
		rospy.Subscriber("/edrone/camera_rgb/image_raw", Image, self.callback)

		self.arm() # ARMING THE DRONE

	def show_image(img):
		cv2.imshow("Image Window", img)
		cv2.waitKey(3)


	# Disarming condition of the drone
	def disarm(self):
		self.cmd.rcAUX4 = 1100
		self.command_pub.publish(self.cmd)
		rospy.sleep(1)


	# Arming condition of the drone : Best practise is to disarm and then arm the drone.
	def arm(self):

		self.disarm()

		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1500
		self.cmd.rcAUX4 = 1500
		#self.command_pub.publish(self.cmd)	# Publishing /drone_command
		rospy.sleep(1)



	# Whycon callback function
	# The function gets executed each time when /whycon node publishes /whycon/poses 
	def whycon_callback(self,msg):
		self.drone_position[0] = msg.poses[0].position.x
		self.drone_position[1] = msg.poses[0].position.y
		self.drone_position[2] = msg.poses[0].position.z



	# Callback function for /pid_tuning_altitude
	 #This function gets executed each time when /tune_pid publishes /pid_tuning_altitude
	def altitude_set_pid(self,alt):
		self.Kp[2] = alt.Kp * 0.06# This is just for an example. You can change the ratio/fraction value accordingly
		self.Ki[2] = alt.Ki * 0.0005
		self.Kd[2] = alt.Kd 
		#print(self.Kd[2])

	def pitch_set_pid(self,pitch):
		self.Kp[1] = pitch.Kp * 0.06
		self.Ki[1] = pitch.Ki * 0.001
		self.Kd[1] = pitch.Kd * 0.3

	def roll_set_pid(self,roll):
		self.Kp[0] = roll.Kp * 0.06
		self.Ki[0] = roll.Ki *0.001
		self.Kd[0] = roll.Kd * 0.3



	def func_wrapper(func):	
		
		def inner_func(self):
			while self.block == True:
				
				coord = self.blockpoint
				print('coord',coord)
				func(self,coord=coord)

			else:
				coord=self.waypoint

				print(coord)
				func(self,coord=coord)

				if coord[0] < -10.25 or coord[0] > 10.25:
					self.waypoint[1] -= 3.4
					if len(self.distance)!=0:
						self.success = True

					if self.n == True:
						self.waypoint[0] = -10.2

					else:
						self.waypoint[0] =  10.2

					self.n = not self.n
					
					print('changing column, updated waypoint', self.waypoint)



				else :
					print (self.n)
					if self.n == True:
						self.waypoint[0] -= 4.091 #3.41
					else:
						self.waypoint[0] += 4.091 #3.41

					print('updated waypoint', self.waypoint)

		return inner_func

	def pid(self,coord):

		

		x_setpoint = coord[0]
		y_setpoint = coord[1]
		z_setpoint = coord[2]

		x_max = x_setpoint + 0.2
		y_max = y_setpoint + 0.2
		z_max = z_setpoint + 0.2

		x_min = x_setpoint - 0.2
		y_min = y_setpoint - 0.2
		z_min = z_setpoint - 0.2

		# PID while loop which controls it's approach and postition to setpoint in X,Y,Z
		while (x_max < self.drone_position[0] or self.drone_position[0] < x_min) or (y_max < self.drone_position[1] or y_min > self.drone_position[1]) or (z_max < self.drone_position[2] or z_min > self.drone_position[2]):

			self.error[2] = -(coord[2] - self.drone_position[2])
			self.error[0] = (coord[0] - self.drone_position[0])
			self.error[1] = -(coord[1] - self.drone_position[1])
			self.delta_error[2] = self.error[2] - self.prev_error[2]
			self.delta_error[1] = self.error[1] - self.prev_error[1]
			self.delta_error[0] = self.error[0] - self.prev_error[0]

			P2 = self.Kp[2] * self.error[2]
			F = 0

			if self.error[2] < 0.6 and self.error[2] > -0.6:
				if self.sum_error[2]>3:
					self.sum_error[2] = 3
				elif self.sum_error[2]<-3:
					self.sum_error[2] = -3
				else:
					self.sum_error=self.sum_error
			
			I2 = self.Ki[2] * self.sum_error[2]
			D2 = self.Kd[2] * self.delta_error[2]

			self.cmd.rcThrottle = int(1500 + P2 + I2 + D2)  # (P2 + I2 + D2)
	
			if self.cmd.rcThrottle > 1800:
					self.cmd.rcThrottle = self.max_values[2]
			if self.cmd.rcThrottle < 1200:
					self.cmd.rcThrottle = self.min_values[2]

			self.prev_error[2] = self.error[2]
			
			self.sum_error[2] += self.error[2]

			P1 = self.Kp[1] * self.error[1]

			
			if self.error[1] < 3 and self.error[1] > -3:
			
				I1 = self.Ki[1] * self.sum_error[1]
			else:
				I1 = 0

			D1 = self.Kd[1] * self.delta_error[1]

			self.cmd.rcPitch = int(1500 + P1 + I1 + D1)

			if self.cmd.rcPitch > 1800:
				self.cmd.rcPitch = self.max_values[1]
			if self.cmd.rcPitch < 1200:
				self.cmd.rcPitch = self.min_values[1]

			
			self.prev_error[1] = self.error[1]
			self.sum_error[1] += self.error[1]

			P = self.Kp[0] * self.error[0]
			if self.error[0] < 1.5 and self.error[0] > -1.5:
				I = self.Ki[0] * self.sum_error[0]
			else:
				I = 0

			D = self.Kd[0] * self.delta_error[0]

			self.cmd.rcRoll = int(1500 + P - I + D)

			if self.cmd.rcRoll > 1800:
				self.cmd.rcRoll = self.max_values[0]
			if self.cmd.rcRoll < 1200:
				self.cmd.rcRoll = self.min_values[0]

			self.prev_error[0] = self.error[0]
			self.sum_error[0] += self.error[0]

			self.command_pub.publish(self.cmd)
			self.throttle_error_pub.publish(self.error[2])
			self.pitch_error_pub.publish(self.error[1])
			self.roll_error_pub.publish(self.error[0])
			self.geolocation_pub.publish(self.msg)

			t=rospy.Rate(18)
			t.sleep()
				
	
	pid_tuned = func_wrapper(pid)

if  __name__ == '__main__':

	e_drone = Edrone()
	r = rospy.Rate(18) #specify rate in Hz based upon your desired PID sampling time, i.e. if desired sample time is 33ms specify rate as 30Hz #12
	while not rospy.is_shutdown():
		e_drone.pid_tuned()
		r.sleep()
		
	
	
	
		
	