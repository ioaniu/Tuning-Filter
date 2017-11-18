function res = get_current_frame(screw_pos)
    load NNet.mat;
    temp_frame = net(screw_pos);
    res = recoverdata(temp_frame);  

end