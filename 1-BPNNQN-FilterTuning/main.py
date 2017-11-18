# -*- coding: utf-8 -*-
"""
This is the main class where all thing are put together

@author: Jingfeng Yang 2016-03-09, Create

@author: Jingfeng Yang 2016-03-24, Modify

"""


# add the path of our project
#import sys
# sys.path.append('G:/test/python/Filter-tuning-BPNNQN-master')

import numpy as np
import random
import time
from os import linesep as NL

import matplotlib.pyplot as plt

#-------------------------------- import my function
# Networks
from ai.NeuralNetwork import Nnet
# Data memory
from memory.memoryd import MemoryD
# Enviroment-----cavity-filter
from cavityfilter.cavityfilter import Filter




""" Main class """
class Main:
	#-------Initialization Parameters
	#--Memory Parameters
	memory_size = 50000 # Home many data keep in memory

	total_states_nbr = 0 # Total states tuned, only incremented during training

	data_memory = None # include: state, reward, action, count

	#--Networks Parameters
	minibatch_size = 50 # Size of the minibatch

	test_epsilon = 0.01 # Epsilon during testing

	discount_factor = 0.99 # Discount factor for future rewards

	epsilon_total_nbr = 20000 # Exploration rate annealing speed

	net_models = None # networks models: BPNN

	train_net = None # book the networks which tuning reached
	#--Enviroment Parameters
	state_nbr = 1
	
	tuned_reached = False

	current_state = None

	myfilter = None

	#--Others testing demonstration --random and fixed
	random_nr = 100 # Number of random sample to use for training
	random_demo = []


	def __init__(self):
		"""
		configure
		@param memory: store tuning information
		@param enviroment: implement enviroment----filter
		@param networks: AI
		"""
		# configure the enviroment
		self.myfilter = Filter()

		# configure the networks
		self.net_models = Nnet()

		# configure the memory
		self.data_memory = MemoryD(self.memory_size,self.myfilter.state_size, self.state_nbr)

		"""
		random demostration for calculate the avg qvaluse
		""" 
		random_screws = np.zeros((self.random_nr,2), dtype = np.float64)
		for i in range(self.random_nr):
			random_screws[i] = ([random.uniform(self.myfilter.screw_min, self.myfilter.screw_max),random.uniform(self.myfilter.screw_min, self.myfilter.screw_max)])

		# Fetch random states
		random_states = self.myfilter.new_tuning(random_screws) 
		# dimensionality reduction
		self.dr_random_states = self.myfilter.dimreduction_pca(random_states.transpose())


	def compute_epsilon(self,epsilon_nbr, state_tuned):
		"""
		From the paper: "The behavior policy during training was epsilon-greedy
		with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
		@param states_tuned: How far are we with our learning
		"""
		return max(0.99 - state_tuned / epsilon_nbr, 0.1)
	def compute_reward(self,cost_old, cost):
		
		# Changing rewards
		if cost < cost_old:
			return 1
		else:
			return -1

	def predict_best_action(self,state):
		# Dimensionality reduction
		dr_state = self.myfilter.dimreduction_pca(state.transpose())

		# use networks to predict q-values for all actions
		qvalues = self.net_models.nnet.predict(dr_state)		

		# return action index with maximum Q-values
		return np.argmax(qvalues)

	def tuning_filter(self, tuning_steps, training, epsilon = None):
		"""
		start a tuning
		@param tuning_steps: total steps of the filter allowed to tune
		@param train: true or false, whether to training 
		"""

		# initialization some parameters
		
		tuning_cost = 0 # 
		tuning_cost_old = 0;

		tuning_finished = False # when tuning reached, set True
		tuned_steps = [] # when tuning reached, save tuned steps


		screws_position = np.zeros((1, 2), dtype = np.float64)

		#-----------start a new tuning, get current state
		current_state = self.myfilter.new_tuning(screws_position)

		# If training we add to memory, else pass
		if training:
			self.data_memory.add_first(current_state)
			nnet_now = []
		else:
			pass
		
		# Check tuning reached 
		tuning_finished, tuning_cost = self.myfilter.tuning_check(current_state)
		tuning_cost_old = tuning_cost;

		#-----------loop untill reached or over steps limit
		tuning_count = 0 # use to count the total tuning steps of this cycle
		tuning_step = 0	# use to count the tuning steps, when tuning is finished, clear it.

		while tuning_count < tuning_steps:
			
			# ---Epsilon decrease over time only when training
			if training:
				epsilon = self.compute_epsilon(self.epsilon_total_nbr, self.total_states_nbr)

			# ---Predict action
			# Some time random action
			if random.uniform(0,1) < epsilon:
				action = random.choice(range(self.myfilter.actions_nbr))

			# Usually Net chooses the best action
			else:
				action = self.predict_best_action(self.current_state)

#			print "actions %d " % action

			# tuning, return screws position and the new state
			screws_position, next_state, tuning_finished, tuning_cost = self.myfilter.tuning(action, screws_position)

			# Calculate Rewards

			reward = self.compute_reward(tuning_cost_old,tuning_cost)


			# Book keeping
			tuning_count += 1
			tuning_step += 1
			self.current_state = next_state
			tuning_cost_old = tuning_cost

			# display current state
			plt.subplot(1,2,1)
			self.myfilter.plot_state(next_state)
			plt.text(20,-25,('Action : ' + str(action)))        
			plt.text(20,-28,('screw position: ' + str(screws_position[0])))
			plt.text(20,-31,('Cost: ' + str(tuning_cost)))
			plt.text(20,-34,('Now Count : ' + str(tuning_count-1)))
			plt.text(20,-37,('Total Count : ' + str(self.data_memory.count)))
			plt.hold(False)
			plt.hold(False)
			plt.show()
			plt.draw()
			plt.pause(0.000001)

			#---If training
			if training:
				# Increase total state number
				self.total_states_nbr += 1

				# Store new information to memory
				self.data_memory.add(action, reward, next_state)				

				# Fetch random minibatch from memory
				minibatch = self.data_memory.get_minibatch(self.minibatch_size)

				# Train net with the minibatch
				nnet_now = self.net_models.net_train(minibatch, self.myfilter, self.discount_factor)

			# When tuning reached, if tuning count is not reached limit, tuning continue.
			if tuning_finished:				
				print " Tuning finished, steps = %d " % tuning_step

				# Book tuning count
				tuned_steps.append(tuning_step)

				# Book current networks 
				if training:
					self.train_net.append(nnet_now)


				# Clear tuning step, finishe flag, anc cost.
				tuning_step = 0
				tuning_finished = False

				# Start New tuning
				screws_position = np.zeros((1, 2), dtype = np.float64)

				#-----------start a new tuning, get current state
				current_state = self.myfilter.new_tuning(screws_position)

				# If training we add to memory, else pass
				if training:
					self.data_memory.add_first(current_state)
				else:
					pass

		# avoid null information
		if len(tuned_steps) == 0:
			tuned_steps.append(tuning_steps)

		#return tuned steps
		if training: 
			return tuned_steps, self.train_net
		else:
			return tuned_steps 


	def run(self, epochs, training_steps, testing_steps):
		# -------------Open log files and write headers
		timestamp = time.strftime("%Y-%m-%d-%H-%M")
		if training_steps > 0:
			# training log file open 
			log_training = open("log/training_" + timestamp + ".txt", "w")
			# training log file writes header
			log_training.write("epoch, training tuned steps, total training steps, epsilon, momery count")

		if testing_steps > 0:
			# testing log file open 
			log_testing = open("log/testing" + timestamp + ".txt", "w")
			# testing log file writes header
			log_testing.write("epoch, testing tuned steps, avg qvaule, epsilon, momory count")


		plt.figure(1)
		avg_qvalues = []
		training_tuned_steps = []
		testing_tuned_steps = []
		# --------start loop------------
		for epoch in range(1, epochs + 1):
			# print epoch now
			print "Epoch %d:   " % epoch

			#---------------training
			if training_steps > 0:
				print "Training for %d steps" % training_steps
				# tuning filter
				tuned_steps, train_net = self.tuning_filter(training_steps, training = True)

				# save training log
				# epoch, training steps, avg qvlues, epsilon, memory count
				log_training.write(','.join(map(str, (epoch, tuned_steps, self.total_states_nbr, self.compute_epsilon(self.epsilon_total_nbr, self.total_states_nbr), self.data_memory.count))) + NL)
				log_training.flush()

				# use random state to calculate avg qvalues
				random_qvalues = self.net_models.nnet.predict(self.dr_random_states)
				avg_qvalue = np.mean(np.max(random_qvalues, axis=1))
				print " Avg Q Value : %d  "  %avg_qvalue
				avg_qvalues.append(avg_qvalue)
				plt.subplot(3,2,2)
				plt.plot(avg_qvalues)
				plt.xlabel('Epochs')
				plt.ylabel('Q-values')
				plt.hold(False)
				plt.show()
				plt.draw()
				plt.pause(0.000001)

				# show training tuned steps
				training_tuned_steps.append(tuned_steps)
				plt.subplot(3,2,4)
				plt.plot(training_tuned_steps)
				plt.xlabel('Epochs')
				plt.ylabel('Trianing Steps')
				plt.hold(False)
				plt.show()
				plt.draw()
				plt.pause(0.000001)


			#---------------testing
			if testing_steps > 0:
				print "Testing for %d steps" % testing_steps

				# tuning filter
				tuned_steps = self.tuning_filter(testing_steps, training = False, epsilon = self.test_epsilon)

				# save testing log
				# epoch, testing steps, avg qvalues, epsilon, memory count
				log_testing.write(','.join(map(str, (epoch, tuned_steps, avg_qvalue, self.test_epsilon, self.data_memory.count))) + NL)
				log_testing.flush()

				# show testing tuned steps
				testing_tuned_steps.append(tuned_steps)
				plt.subplot(3,2,6)
				plt.plot(testing_tuned_steps)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Steps')
				plt.hold(False)
				plt.show()
				plt.draw()
				plt.pause(0.000001)

		# close log files
		if training_steps > 0:
			log_training.close()


		if testing_steps > 0:
			log_testing.close()

		# return data 
		return train_net

#----------------------------------------------------------------------------
# main 
#----------------------------------------------------------------------------
if __name__ == '__main__':
	# defaults some parameters

	epochs = 100

	# training steps limit
	training_steps = 500

	# testing steps limit
	testing_steps = 200
	mymain = Main()
	# run the main loop 
	train_net = mymain.run(epochs, training_steps, testing_steps)

	train_net.save_weights('simulate_weigths.h5')
	open('simulate_architechture.json', 'w').write(train_net.to_json())
	open('simulate_architechture.yaml', 'w').write(train_net.to_yaml())

# ---- The End -----#