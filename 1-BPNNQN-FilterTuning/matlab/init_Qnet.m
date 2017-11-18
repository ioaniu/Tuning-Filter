% initialize Q-network

initW = 0.001;
initB = 0.01;
Qnet.iteration = 50;
%% BPNN
Qnet.innum = 20;
%Qnet.midnum = 10;
Qnet.midnum = 30;
Qnet.outnum = 4;

Qnet.bpnnW1 = rands(Qnet.innum,Qnet.midnum);
Qnet.bpnnB1 = rands(1,Qnet.midnum);
Qnet.bpnnW2 = rands(Qnet.midnum,Qnet.outnum);
Qnet.bpnnB2 = rands(1,Qnet.outnum);

%Qnet.bpnnlr = 0.01;
Qnet.bpnnlr = 0.01;

