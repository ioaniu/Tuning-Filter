function total_stores = tune_filters(nr_frames,train,epsilon)
	% tune filters 
    global epoch
    if train
       aviobj = VideoWriter(strcat('train',num2str(epoch)));
    else
       aviobj = VideoWriter(strcat('test',num2str(epoch)));
    end
     open(aviobj);

	global number_of_screws memoryD minibatch_size total_frames_trained epsilon_frames
    global Qnet hold_frame
    %regular_frame

    frames_tuned = 0;
    total_stores.Qvalue = [];
    total_stores.action = [];
    total_stores.screw_pos = [];
      total_stores.state = [];
    
	% Start a new tuning    
    screw_pos =rand(number_of_screws,1) * 5000;
%	screw_pos = rand(number_of_screws,1) * 5000;%Initialize all the tuning screws randomly
    current_frame = get_current_frame(round(screw_pos));

	% Compute the distance between current curve and the desired one
    [cost,filter_tuned ] = compute_cost(current_frame);
    
	% If training add the first S parameters curve to memory
	if train
        if memoryD.count < 1 
            memoryD.state(:,mod(memoryD.count + 1,memoryD.size)) = current_frame;
            memoryD.count = memoryD.count + 1;
        end
	end

	% Initialize/ update the current state

	% tune filter maximum number is reached
	while frames_tuned < nr_frames
		if train
			epsilon = compute_epsilon(total_frames_trained,epsilon_frames);
        end

 		% Epsilon-greedy strategy: some times random action is chosen
		num_of_actions = 2 * number_of_screws;
		if rand(1) < epsilon
	%		action = randsrc(1,1,[-num_of_actions/2 : -1 1 :num_of_actions/2]);
            Qvalue = 1 * randn(1,4);
            actionlist = [-number_of_screws : -1 1:number_of_screws];
            action_idx = find(Qvalue==max(Qvalue));
            action = actionlist(action_idx);
        else
            if train
    			[action,Qvalue] = predict_best_action(memoryD.state(:,mod(memoryD.count,memoryD.size)));
            else
                [action,Qvalue] = predict_best_action(current_frame');
            end
        end
        
		% Execute the tuning. Returns distance and the new state
		[new_cost,next_frame,screw_pos,filter_tuned,action] = tune(screw_pos,action,Qvalue);
 %       next_frame = mapminmax('apply',next_frame,regular_frame);
        total_stores.Qvalue = [total_stores.Qvalue;Qvalue];
        total_stores.action = [total_stores.action;action];
        total_stores.screw_pos = [total_stores.screw_pos;screw_pos'];

		% Obtain rewards according to change of cost
		if new_cost < cost
			reward = 1;
		else
			reward = 0;
        end
        
        current_frame = next_frame;
        if train 
        else
            total_stores.state = [total_stores.state;current_frame];
        end
		cost = new_cost;
		frames_tuned = frames_tuned + 1;
%plot S-para
        figure(1);
%        subplot(2,2,1);
        plot(next_frame,'LineWidth',1.5);
        ylim([-45 8]);
        xlim([0 410]);       
%        text(10,-30,strcat('Q1:',num2str(Qvalue(1))));
%        text(100,-30,strcat('Q2:',num2str(Qvalue(2))));
%        text(200,-30,strcat('Q3:',num2str(Qvalue(3))));
%        text(300,-30,strcat('Q4:',num2str(Qvalue(4))));
        if(action == -1)
            text(40,-30,['tuning action: ','screw 1 counterclockwise'],'FontSize',10);
        elseif (action == 1)
            text(40,-30,['tuning action: ','screw 1 clockwise'],'FontSize',10);
        elseif (action == -2)
            text(40,-30,['tuning action: ','screw 2 counterclockwise'],'FontSize',10);
        else
            text(40,-30,['tuning action: ','screw 2 clockwise'],'FontSize',10);
        end
%        text(80,-35,strcat('action:',num2str(action)),'FontSize',15);
        text(180,-30,strcat('cost:',num2str(cost)),'FontSize',10);
%        text(100,-35,['reward: ',num2str(reward)],'FontSize',10);
%        text(250,-35,['timestep: ',num2str(frames_tuned)],'FontSize',10);
        text(20,-35,strcat('screw 1 position : ',num2str(screw_pos(1))),'FontSize',10);
        text(160,-35,strcat('screw 2 position : ',num2str(screw_pos(2))),'FontSize',10);

%        text(20,-35,strcat('screw 1 position (r): ',num2str(screw_pos(1)/2000)),'FontSize',10);
%        text(160,-35,strcat('screw 2 position (r): ',num2str(screw_pos(2)/2000)),'FontSize',10);
        text(300,-35,strcat('meoery number: ',num2str(memoryD.count)),'FontSize',10);
        text(100,-40,num2str(frames_tuned-1));
        text(200,-40,num2str(train));
        text(200,-40,num2str(reward));
        text(300,-40,num2str(epsilon));
%        line([162,240],[-21,-21],'LineWidth',1.7,'color','r');
        line([164,238],[-21,-21],'LineWidth',1.7,'color','r');
        xlabel('Frequency(MHz)');
        ylabel('Return loss(dB)');
    %    title('Reinforcement Learning Approach to Learning Human Experience in Tuning Cavity');
        set(gca,'XTickLabel',{'740','784','828','872','916','960','1004','1048','1092','1136'}) 
        set(gca,'FontSize',9);
        legend('current state','tuning target');
        grid;
        drawnow;

		if train
			% Store new information to memory
			memoryD.action(mod(memoryD.count,memoryD.size)) = action;
			memoryD.reward(mod(memoryD.count,memoryD.size)) = reward;
			memoryD.state(:,mod(memoryD.count+1,memoryD.size)) = next_frame;
			memoryD.count = memoryD.count + 1;
			% Increase total frames only when training
			total_frames_trained = total_frames_trained + 1;
			%Fetch random minibatch from memory
			minibatch = get_minibatch(minibatch_size);
			%train neural network with the minibatch
			Qnet = train_minibatch(minibatch,Qnet);
            figure(2);
%            bar(mycnnprepredict(Qnet,memoryD.state(:,1)));
%            drawnow;
            %show hold Q_Value;
%             subplot(2,2,2);
            total_stores.hold_Qvalue(frames_tuned) = compute_frame_Qvlaue(hold_frame,Qnet);
            plot(total_stores.hold_Qvalue);
            ylim([0 12]);
            xlabel('tuning steps');
            ylabel('Average Q-value(fixed set of states)');
            drawnow;
        else
%             subplot(2,2,2);
%             total_stores.hold_Qvalue(frames_tuned) = compute_frame_Qvlaue(current_frame',Qnet);
%             plot(total_stores.hold_Qvalue);
%             ylim([0 12]);
%             xlabel('tuning steps');
%             ylabel('Q-value(current state)');
		end

		if filter_tuned == 1;           
            total_stores.frames_tuned = frames_tuned;
            break;
        end
%        frame = getframe(gcf);
%        im = frame2im(frame);   
%        writeVideo(aviobj,im);
        
    end
     total_stores.frames_tuned = frames_tuned;     
    close(aviobj);
 

end