function q_net = train_minibatch(minibatch,q_net)
    global discount_factor number_of_screws
    % Load form minibatch
    prestates = minibatch.prestate;
    actions = minibatch.action;
    rewards = minibatch.reward;
    poststates = minibatch.poststate;
    
    
    
    % Predice Q-values for prestates
%    preQvalues = mycnnprepredict(q_net,prestates);
    preQvalues = mybpnnprepredict(q_net,prestates')';
    
    % Predice Q-values for poststates
%    postQvalues = mycnnprepredict(q_net,poststates);
   postQvalues = mybpnnprepredict(q_net,poststates')';
    % Take maximum Q-value of all actions
    maxQvalues = max(postQvalues);
    
    % Update the Q-value for actions we actually preformed
    for i = 1: size(actions,2)
        if(actions(i))<0
            index = number_of_screws + actions(i) + 1;
        else
            index = number_of_screws + actions(i);
        end
        
        preQvalues(index,i) = rewards(i) + discount_factor * maxQvalues(i);
    end
    
    % Train the Q-network
%    q_net = mycnntrain(prestates,preQvalues,q_net);
   q_net = mybpnntrain(prestates',preQvalues',q_net);
    
end