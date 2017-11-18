function pre_qvalue = mybpnnprepredict(qnet, prestates)
    load U.mat
    pcastates = prestates * U(:,1:qnet.innum);
    nr_states = size(prestates,1);
   
    %
    for i = 1 : nr_states
        x1(i,:) = pcastates(i,:) * qnet.bpnnW1 + qnet.bpnnB1;
        for n = 1 : qnet.midnum
            x2(i,n) = 1 / (1 + exp(-x1(i,n)));
        end        
        pre_qvalue(i,:) = x2(i,:) *  qnet.bpnnW2 + qnet.bpnnB2;       
    end       
end