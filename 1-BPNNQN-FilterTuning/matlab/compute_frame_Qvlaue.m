function result = compute_frame_Qvlaue(hold_frame,qnet)
    res = mybpnnprepredict(qnet,hold_frame');
%    res = filtercnnDQN(hold_frame, cnn.w, cnn.b, cnn.theta1, cnn.beta1, cnn.theta2, cnn.beta2);
%    Qvalues = reshape(res.x5,4,size(hold_frame,2));
%    res = mycnnpredict(qnet,hold_frame);
%    res = mycnn(hold_frame,net);
%     Qvalues = res.outputy;
    result = mean(max(res'));
end