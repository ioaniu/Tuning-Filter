function [res,Qvalues] = predict_best_action(frame_now)
    global Qnet number_of_screws
    
%    Qvalues = mycnnprepredict(Qnet,frame_now);
    Qvalues = mybpnnprepredict(Qnet,frame_now');
    
    actionlist = [-number_of_screws : -1 1:number_of_screws];
    action_idx = find(Qvalues==max(Qvalues));
    res = actionlist(action_idx);
end