function [A, B, C, D, x0est] = mySSest_simple(ys, us, dt, modelOrder)
% ys and us are (signal dim) × (# of data samples)
% Returns state-space estimated model of modelOrder

dataset = iddata(ys', us', dt);

% Additional conversion to double needed to avoid type confusion when
% calling from outside (like Python)
[mySS, x0est] = ssest(dataset, double(modelOrder), 'Ts', dt);
    
A = mySS.A;
B = mySS.B;
C = mySS.C;
D = mySS.D;

end

