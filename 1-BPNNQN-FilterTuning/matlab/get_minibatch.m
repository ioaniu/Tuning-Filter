function res = get_minibatch(size)

global memoryD
    res.prestate = [];
    res.action = [];
    res.reward = [];
    res.poststate = [];
    j = 1;
   while j < size+1
       i = randi(min(memoryD.count-1,memoryD.size));
       while memoryD.action(i)== 100
           i = randi(min(memoryD.count,memoryD.size));
       end
       
       % add a state into batch unless it is an end state
       if memoryD.action(i) ~= 100
           res.prestate(:,j) = memoryD.state(:,i);
           res.action(j) = memoryD.action(i);
           res.reward(j) = memoryD.reward(i);
           res.poststate(:,j) = memoryD.state(:,i+1);
           j = j+1;
       end
   end   
 
end