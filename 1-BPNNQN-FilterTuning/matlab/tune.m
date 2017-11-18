function [new_cost,next_frame,screw_pos,filter_tuned,action] = tune(screw_pos,action,Qvalue)
 %   global number_of_screws
    load NNet.mat;

    step_pitch = 500;

    if action < 1
        direction = -1;
    else
        direction = 1;
    end
%    [b,action_idx] = sort(Qvalue);
%    actionlist = [-number_of_screws : -1 1:number_of_screws];
%    if(abs(action) == 1)
        minscrew = 0;
        maxscrew = 25000;
%    else
%        minscrew = 193;
%        maxscrew = 11335;
%    end


if (((screw_pos(abs(action))+ step_pitch * direction) > minscrew) & ((screw_pos(abs(action))+ step_pitch * direction) < maxscrew))
    screw_pos(abs(action)) = screw_pos(abs(action)) + step_pitch * direction; 
%else
%
%    screw_pos(abs(action)) = screw_pos(abs(action)) - 10* step_pitch * direction;
%
end
% else
%     action = actionlist(action_idx(3));  
%     if(abs(action) == 1)
%         minscrew = -1893;   
%         maxscrew = 18230;
%     else
%         minscrew = 193;
%         maxscrew = 11335;
%     end
%     if action < 1
%         direction = -1;
%     else
%         direction = 1;
%     end
%     if(((screw_pos(abs(action))+ step_pitch * direction) > minscrew) & ((screw_pos(abs(action))+ step_pitch * direction) < maxscrew))
%         screw_pos(abs(action)) = screw_pos(abs(action)) + step_pitch * direction;  
%     else
%         action = actionlist(action_idx(2));  
%         if(abs(action) == 1)
%             minscrew = -1893;
%             maxscrew = 18230;
%         else
%             minscrew = 193;
%             maxscrew = 11335;
%         end
%         if action < 1
%             direction = -1;
%         else
%             direction = 1;
%         end
%         if(((screw_pos(abs(action))+ step_pitch * direction) > minscrew) & ((screw_pos(abs(action))+ step_pitch * direction) < maxscrew))
%             screw_pos(abs(action)) = screw_pos(abs(action)) + step_pitch * direction;  
%         else
%             action = actionlist(action_idx(1));  
%             if(abs(action) == 1)
%                 minscrew = -1893;
%                 maxscrew = 18230;
%             else
%                 minscrew = 193;
%                 maxscrew = 11335;
%             end
%             if action < 1
%                 direction = -1;
%             else
%                 direction = 1;
%             end
%             if(((screw_pos(abs(action))+ step_pitch * direction) > minscrew) & ((screw_pos(abs(action))+ step_pitch * direction) < maxscrew))
%                 screw_pos(abs(action)) = screw_pos(abs(action)) + step_pitch * direction;  
%             else
%                 screw_pos(abs(action)) = screw_pos(abs(action)) - step_pitch * direction * 10 ;
%             end
%         end
%     end
% end
    
%     if(screw_pos(abs(action)) == 1)
%         if(screw_pos(1) + 500 * direction) > 20000
%             if(screw_pos(2) + 500 * direction) > 20000
%                 screw_pos(1) = screw_pos(1) - 500 * direction * 5;
%             else
%                 screw_pos(2) = screw_pos(2) + 500 * direction;
%             end            
%         elseif (screw_pos(1) + 500 * direction) < 0
%             if (screw_pos(2) + 500 * direction) < 0
%                 screw_pos(1) = screw_pos(1) - 500 * direction * 5;
%             else
%                 screw_pos(2) = screw_pos(2) + 500 * direction;
%             end
%         else
%            screw_pos(1) = screw_pos(1) + 500 * direction;
%         end
%     else
%         if(screw_pos(2) + 500 * direction) > 20000
%             if(screw_pos(1) + 500 * direction) > 20000
%                 screw_pos(2) = screw_pos(2) - 500 * direction * 5;
%             else
%                 screw_pos(1) = screw_pos(1) + 500 * direction;
%             end            
%         elseif (screw_pos(2) + 500 * direction) < 0
%             if (screw_pos(1) + 500 * direction) < 0
%                 screw_pos(2) = screw_pos(2) - 500 * direction * 5;
%             else
%                 screw_pos(1) = screw_pos(1) + 500 * direction;
%             end
%         else
%            screw_pos(2) = screw_pos(2) + 500 * direction;
%         end
%     end
% %    screw_pos(abs(action)) = screw_pos(abs(action)) + 500 * direction;

 
    
%     if(screw_pos(abs(action))) > 20000
%         screw_pos(abs(action)) = 19000;
%     elseif(screw_pos(abs(action)) < 0)
%         screw_pos(abs(action)) = 1000;
%     end
    next_frame = get_current_frame(round(screw_pos));
    [new_cost, filter_tuned] = compute_cost(next_frame);
    
end
