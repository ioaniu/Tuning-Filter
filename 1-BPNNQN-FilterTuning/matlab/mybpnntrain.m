function qnet = mybpnntrain(prestates,targetQvalue,qnet)
    load U.mat
    pcastates = prestates * U(:,1:qnet.innum);
    nr_states = size(prestates,1);
    
    EE = zeros(qnet.iteration);
    for j = 1 : qnet.iteration
        %FW
        for i = 1 : nr_states
            x1(i,:) = pcastates(i,:) * qnet.bpnnW1 + qnet.bpnnB1;
            for n = 1 : qnet.midnum
                x2(i,n) = 1 / (1 + exp(-x1(i,n)));
            end        
            pre_qvalue(i,:) = x2(i,:) *  qnet.bpnnW2 + qnet.bpnnB2;

            error = targetQvalue(i,:) - pre_qvalue(i,:);        
            EE(j) = EE(j) + sum(abs(error));
            %loss
            dw2 = (error' * x2(i,:))';
            db2 = error;
            for k = 1 : qnet.midnum
                SS = 1 / (1 + exp(-x1(k)));
                FI(k) = SS * (1-SS);
            end
            
            dw1 = (FI' * pcastates(i,:) * sum(error * qnet.bpnnW2')')';
            db1 = FI * sum(error * qnet.bpnnW2')';
            %update             
            qnet.bpnnW2 = qnet.bpnnW2 + qnet.bpnnlr * dw2;        
            qnet.bpnnB2 = qnet.bpnnB2 + qnet.bpnnlr * db2;
            qnet.bpnnW1 = qnet.bpnnW1 + qnet.bpnnlr * dw1;
            qnet.bpnnB1 = qnet.bpnnB1 + qnet.bpnnlr * db1;            
        end                
    end
end