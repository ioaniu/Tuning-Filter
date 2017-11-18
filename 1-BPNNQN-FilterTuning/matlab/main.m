%close all;
clear all;
clc;

%%  =================== Initialization===============

global number_of_screws minibatch_size epsilon_frames total_frames_trained
global discount_factor memoryD Qnet

memory_size = 50000; % The largest number of transitions in memory
minibatch_size = 50; %Size of the mini-batch used for training DQN
number_of_screws = 2; % Total number of tuning screws of the filter
frame_size = 401; %The number of frequency points of S paramters
state_size = 1*frame_size; % Size of one state is one screens
discount_factor = 0.9; %Discount factor for future rewards in DQN
epsilon_frames = 10000.0; % Exploration rate annealing speed 
test_epsilon = 0.1; %Epsilon during testing 
total_frames_trained = 0; %Total S curves during training
nr_random_states = 10000; % Number of random states to use for calculating Q-values
random_states = []; %Random states that we use to calculate Q-values
memoryD = []; % memory itself
Qnet = []; % neural net itself
current_state = []; % The last states has seen;
total_training_step = [];

%-----initialize the memory
memoryD.state = zeros(frame_size,1);
memoryD.action = 0;
memoryD.reward = 0;
memoryD.count = 0;
memoryD.time = 0;
memoryD.size = memory_size;

%------initialize neural net
% Convolution + Convolution + Full Connection + Full Connection
% 8x8x16 + 4x4x32 + 256 + 4
% Relu + Relu + Relu + linear
%
init_Qnet;
%--------------------------------------------------------------------------------
 for i = 1 : 100
 temp_screw_pos = rand(number_of_screws,1) * 20000;
 hold_frame(:,i) = get_current_frame(round(temp_screw_pos));
 end
 global hold_frame;
%% 
%epochs = input('Please assign the number of epochs: ');
epochs = 100;
training_frames = 1000;
testing_frames = 200;

%% =======================DQN training and testing
%     global aviobj
    
global epoch

for epoch = 1 : epochs
	disp(['Epoch: ',num2str(epoch)]);

	% training
	if training_frames > 0
		disp(['Training for ', num2str(training_frames), 'frames']);
        filter.threshold = -21;  % design specification for return loss in dB
        
		training_scores{epoch} = tune_filters(training_frames,1,test_epsilon);
        total_training_step(epoch) = training_scores{1,epoch}.frames_tuned
%         subplot(2,2,3);
%         bar(total_training_step);
%         xlabel('Training epochs');
%         ylabel('Tuning steps');
%         drawnow;

		%log training scores
		%log aggregated training data
		%get weights from nnet
		%log weights data

	end
		%save network state

	%testing
	if testing_frames > 0
		disp(['Testing for ', num2str(testing_frames), 'frames']);
        filter.threshold = -21;  % design specification for return loss in dB
		testing_scores{epoch} = tune_filters(testing_frames,0,test_epsilon);        
        total_testing_step(epoch) = testing_scores{1,epoch}.frames_tuned;
%         subplot(2,2,4);
%         bar(total_testing_step);
%         xlabel('Testing epochs');
%         ylabel('Tuning steps');
%         drawnow;
		%log testing scores
		% Pick random states to calculate Q-values for
		%log aggregated testing data		

    end	

end


%figure,bar(testing_scores);
