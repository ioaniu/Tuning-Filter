# -*- coding: utf-8 -*-
"""
.....enviroment.....
Cavity filter class launches the cavity filter tuning and manages the communication with it
@author: Jingfeng Yang 2016-03-10 create
@author: Jingfeng Yang 2016-03-25 modify

"""

import scipy.io
import numpy as np
import numpy.matlib

##
import matplotlib.pyplot as plt


class Filter:
	# Start sampling frequency in MHz
	start_freq = 740
	# End sampling frequency in MHz
	end_freq = 1100
	# Frequency of the left marker in Mhz
	markerl = 883
	# Frequency of the right marker in Mhz
	markerr = 956
	# Design specification for return loss in dB
	threshold = -21
	# Sampling size
	state_size = 401
	# tuning screws number
	screws_nbr = 2
	# tuning actions number
	actions_nbr = screws_nbr * 2
	# tuning action
	actions = range(actions_nbr)
	#	actions = [np.uint8(0), np.uint8(1), np.uint8(2), np.uint8(3)]

	# tuning screw range
	screw_range = 500.000
	# tuning screw start point 
	screw_min = 500.00
	# tuning screw end pint
	screw_max = 24500.00

	# check state position
	start_piont = (markerl - start_freq) * (state_size - 1) / (end_freq - start_freq)  + 1    
	end_piont = (markerr - start_freq) * (state_size - 1)/ (end_freq - start_freq)  + 1  

	def __init__(self):
		"""
		Initialize Filter class.         
		"""
		# read the files to load sample and simulate model
		sample = scipy.io.loadmat('cavityfilter/simulatedata.mat')
		simulatenet = scipy.io.loadmat('cavityfilter/simulatenet.mat')

		# fetch imensionality reduction eigens matrix

		self.pcaeigens = sample.get('U')

		# fetch net model parameters 
		self.x1_step1_xoffset = simulatenet.get('x1_step1_xoffset')
		self.x1_step1_gain = simulatenet.get('x1_step1_gain')
		self.x1_step1_ymin = simulatenet.get('x1_step1_ymin')

		self.b1 = simulatenet.get('b1')
		self.IW1_1 = simulatenet.get('IW1_1')

		self.b2 = simulatenet.get('b2')
		self.LW2_1 = simulatenet.get('LW2_1')

		self.y1_step1_ymin = simulatenet.get('y1_step1_ymin')
		self.y1_step1_gain = simulatenet.get('y1_step1_gain')
		self.y1_step1_xoffset = simulatenet.get('y1_step1_xoffset')

	""" simulation networks"""
	def mynetfunction(self,x):
		# sample Dim
		Q = x.shape[0]

		# 
		xp1 = self.maxmin_apply(x, self.x1_step1_xoffset, self.x1_step1_gain, self.x1_step1_ymin)

		a1 = self.tansig_apply( np.matlib.repmat(self.b1, 1, Q) + np.dot(self.IW1_1, xp1));
		a2 = np.matlib.repmat(self.b2, 1,Q) + np.dot(self.LW2_1, a1);
		y1 = self.maxmin_reverse(a2, self.y1_step1_xoffset, self.y1_step1_gain, self.y1_step1_ymin)

		return y1
       
    
	def tansig_apply(self, n):
		""" tan sigmoid function"""
		a =  2 / (1 + np.exp(-2 * n)) - 1
		return a

	# maxmin data
	def maxmin_apply(self, x, x1_step1_xoffset, x1_step1_gain, x1_step1_ymin):
		"""
		#  Matlab Code 
		#  y = bsxfun(@minus,x,settings_xoffset);
		#  y = bsxfun(@times,y,settings_gain);
		#  y = bsxfun(@plus,y,settings_ymin);
		"""        
		y = x - x1_step1_xoffset.transpose()        
		for i in range(y.shape[0]):
			for j in range(y.shape[1]):
				y[i][j] = y[i][j] * x1_step1_gain[j]
		y = (y + x1_step1_ymin).transpose()
		return y

	# reverse maxmin data
	def maxmin_reverse(self, y, y1_step1_xoffset, y1_step1_gain, y1_step1_ymin):
		"""
		#  Matlab Code        
		#  x = bsxfun(@minus,y,settings_ymin);
		#  x = bsxfun(@rdivide,x,settings_gain);
		#  x = bsxfun(@plus,x,settings_xoffset);
		"""
		x = y - y1_step1_ymin

		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				x[i][j] = x[i][j] / y1_step1_gain[i]        
		x = x + y1_step1_xoffset
		return x

	# dimensionality reduction based on PCA
	def dimreduction_pca(self, state):
		dr_state = np.dot(state, self.pcaeigens[:,0:20]).transpose()
		
		#Ｎormalization
		dr_state = dr_state - self.y1_step1_xoffset
		for i in range(dr_state.shape[0]):
			for j in range(dr_state.shape[1]):
				dr_state[i][j] = dr_state[i][j] * self.y1_step1_gain[i]
		dr_state = dr_state + self.y1_step1_ymin		
   
		return dr_state.transpose()

	# reconstruction 
	def reconstruction_pca(self,dr_state):
		return np.dot(self.pcaeigens[:,0:20],dr_state)

	"""-----start a tuning"""
	def new_tuning(self,screws):
		"""
		Start a new tuning when tuning is lost or finished
		"""
		#: simulation tuning, and obtain current state (DR data)      
		dr_state = self.mynetfunction(screws)
		# Reconstruction data 
		self.current_state = self.reconstruction_pca(dr_state)

		# return current state
		return self.current_state

	def tuning_check(self,state):
		"""
		Check tuning reached 
		When tuning is reached, end_tuning adds last state to moeory resets the system
		""" 

		# Send reset command to Filter
		tuning_finished = False

		dist = np.zeros((self.state_size,1))

		tuningcost = 0

		# calculate the distance between current state and aim state
		for i in range(self.start_piont,self.end_piont):
			if state[i] < self.threshold:
				dist[i] = 0
			else:
				dist[i] = np.abs(state[i] - self.threshold)
		#
		tuningcost = np.linalg.norm(dist)

		# when the distance equal zeros, tuning reached
		if(tuningcost == 0):
			tuning_finished = True            

		# return tuning flag and cost
		return tuning_finished, tuningcost

	def tuning(self,action_index,screws):
		"""
		Sends action to Filter adn reads responds
		@param action_index: int,
		"""
		#： Convert index to action
		action = self.actions[action_index]

		#: change screws position
		if action == 0:
			if (screws[0][0] >= self.screw_min):
				screws[0][0] = screws[0][0] - self.screw_range
		elif action == 1:
			if (screws[0][0] <= self.screw_max):
				screws[0][0] = screws[0][0] + self.screw_range
		elif action == 2:
			if (screws[0][1] >= self.screw_min):
				screws[0][1] = screws[0][1] - self.screw_range
		else:
			if (screws[0][1] <= self.screw_max):
				screws[0][1] = screws[0][1] + self.screw_range

		#: simulation tuning, and obtain current state        
		dr_state = self.mynetfunction(screws)
		# Reconstruction data
		state = self.reconstruction_pca(dr_state)

		# check state 
		tuning_finished, tuning_cost = self.tuning_check(state)

		# return parameters
		return screws, state, tuning_finished, tuning_cost

		# visualization tuning process
	def plot_state(self,state):
		
		plt.plot(state,color = 'g',linewidth = 1)
		plt.xlim(0,400)
		plt.ylim(-50,5)
		plt.xlabel('Frequency (MHz)')
		plt.ylabel('Return Loss (dB)')
		plt.hold(True)
		plt.plot([self.start_piont, self.end_piont],[self.threshold, self.threshold], linewidth = 3,color = 'r')
  #		plt.xticks(['740','780','820','860','900','940','980','1020','1060','1100'])
#           plt.xticks(['740','780','820','860','900','940','980','1020','1060','1100'])
#		ax.set_xticklabels(['740','780','820','860','900','940','980','1020','1060','1100'])
		plt.grid()

#---------the end-----------#