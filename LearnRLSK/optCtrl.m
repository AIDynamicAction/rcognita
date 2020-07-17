function [u, J] = optCtrl(y, Uinit, S, R, gamma, N, W, A, B, C, D, x0, mode, u_min, u_max, dt, modelPars, critStruct)
%optCtrl  Predictive optimal controller
%
%   Input:
%
%   y               - Measured system output
%   modelPars       - Vector of parameters passed to model-based control (see definition of costFnc below)
%
%   --- Algorithm ---
%
%       Controller seeks to minimize cost
%
%                   N
%           J = sum     r
%                   1
%
%           where (running cost)
%
%               r = y' * S * y + u' * R * u
%
%
%           and N horizon length
%
%          In RL/ADP, N is interpreted as infinity and so, e.g., in Q-learning, r is substituted for Q-function approximate
%
%          Mode
%             1 - model-predictive control (MPC)
%             2 - MPC with estimated model, a.k.a. adaptive MPC, or AMPC
%             3 - RL/ADP (as N-step roll-out Q-learning) using true model for prediction
%             4 - RL/ADP (as N-step roll-out Q-learning) using estimated model for prediction 
%             5 - RL/ADP (as normalized stacked Q-learning with horizon N) using true model for prediction
%             6 - RL/ADP (as normalized stacked Q-learning with horizon N) using estimated model for prediction 
%
%             Methods 1, 3, 5 are model-based and use a built-in model (provide your specification below, function f) if N > 1
%             Methods 2, 4, 6 use estimated state-space model described by (A, B, C, D, x0est)
%             All nethods are model-free if N=1
%
% This function is effectively the actor part of RL
% Q-function regressor is determined by critStruct (see definition below)
%
% Author: P. Osinenko, 2020
% Contact: p.osinenko@gmail.com

%% Initialization
m = modelPars(1);
I = modelPars(2);

[l, ~] = size(Uinit);

p = length(y);

Umin = repmat(u_min, N, 1);
Umax = repmat(u_max, N, 1);

myUinit = reshape( Uinit, numel(Uinit), 1);

options = optimset('fmincon');

if N == 1
    %     options = optimset(options,'TolX', 1e-7, 'TolFun', 1e-7, 'MaxIter', 300, 'MaxFunEvals', 5000, 'Display', 'iter', 'GradObj', 'on', 'Algorithm', 'interior-point');
    options = optimset(options,'TolX', 1e-7, 'TolFun', 1e-7, 'MaxIter', 300, 'MaxFunEvals', 5000, 'Display', 'off', 'GradObj', 'on', 'Algorithm', 'interior-point');
    
else
    %     options = optimset(options,'TolX', 1e-7, 'TolFun', 1e-7, 'MaxIter', 300, 'MaxFunEvals', 5000, 'Display', 'iter', 'Algorithm', 'interior-point');
    options = optimset(options,'TolX', 1e-7, 'TolFun', 1e-7, 'MaxIter', 300, 'MaxFunEvals', 5000, 'Display', 'off', 'Algorithm', 'sqp');
end

Ntrials = 1;

eps = 1e-2;

%% Actor

Ubest = myUinit;
Jbest = costFncCtrl(myUinit, y);

% Mutli-trial fmincon
for J = 1:Ntrials
    [U, Jval] = fmincon(@(U) costFncCtrl(U, y), myUinit, [],[],[],[], Umin, Umax, [], options);
    if Jval <= Jbest - eps
        Ubest = U;
        Jbest = Jval;
    end
    myUinit = myUinit + ( max(u_max) - min(u_min) )/4 * randn(numel(Uinit), 1);
end

%DEBUG=====================================================================
% myU = reshape(Ubest, l, N);
% Yt = zeros(p, N);
% Yt(:, 1) = y;
% for k0 = 2:N
%     Yt(:, k0) = Yt(:, k0-1) + dt * f(Yt(:, k0-1), myU(:,k0-1));
% end
% modEstDebugger(A, B, C, D, y, N, myU, Yt)
%/DEBUG====================================================================

u = Ubest(1:l);   % Output 1st action, cost fnc val
J = Jval;

%% Actor cost function
    function [J, gradJ] = costFncCtrl(U, y)
        
        gradJ = 0;
        
        myU = reshape(U, l, N);
        
        Y = zeros(p, N);
        
        if (mode == 1) || (mode == 3) || (mode == 5)          % Using true model for prediction
            
            Y(:, 1) = y;
            for k = 2:N
                Y(:, k) = Y(:, k-1) + dt * f(Y(:, k-1), myU(:,k-1)); % Euler scheme. May be improved to more advanced numerical integration
            end
            
        elseif (mode == 2) || (mode == 4) || (mode == 6)          % Using estimated model for prediction
            
            Y(:, 1) = y;
            x = x0;
            for k = 2:N
                x = A*x + B*myU(:, k-1);
                Y(:, k) = C*x + D*myU(:, k-1);
            end
            
        end
        
        J = 0;
        
        %DEBUG=====================================================================
        %         Yt = zeros(p, N);
        %         Yt(:, 1) = y;
        %         for k = 2:N
        %             Yt(:, k) = Yt(:, k-1) + dt * f(Yt(:, k-1), myU(:,k-1));
        %         end
        %         modEstDebugger(A, B, C, D, y, N, myU, Yt)
        %/DEBUG====================================================================
        
        if (mode == 1) || (mode == 2)       % MPC
            
            for k = 1:N
                J = J + double(gamma) * (Y(:,k)' * S * Y(:,k) + myU(:,k)' * R * myU(:,k));
            end
            
        elseif (mode == 3) || (mode == 4)   % RL/ADP (N roll-outs of running cost)
            
            for k = 1:N-1
                J = J + double(gamma) * (Y(:,k)' * S * Y(:,k) + myU(:,k)' * R * myU(:,k));
            end
            
            % Q-function as terminal cost
            Q = W'*phi(Y(:,N), myU(:,N));
            J = J + Q;
            
        elseif  (mode == 5) || (mode == 6)   % RL/ADP (stacked Q-learning)
 
             for k = 1:N
                % Q-function
                Q = W'*phi(Y(:,k), myU(:,k));
                J = J + 1/N * Q;
            end           
   
        end
        
        %%%%%%%%% Supply gradient if RL/ADP with N = 1
        if (N == 1) && ( (mode == 3) || (mode == 4) || (mode == 5) || (mode == 6) ) && (nargout > 1)
            gradJ = ( grad_u_phi(Y(:,1), myU(:,1)) )' * W;
        end
        
        % Estimate initial state for state-space model
        %         function x0 = myFindInitState(A, B, C, D, us, ys, dt)
        %
        %             dataset = iddata(ys', us', dt);
        %
        %             % sys = ss(A, B, C, D, dt);
        %             % sys = idss(sys);
        %
        %             [p, n] = size(C);
        %
        %             sys = idss(A,B,C,D, zeros(n,p), zeros(n,1), dt);
        %
        %             x0 = findstates(sys, dataset);
        %
        %         end
        
        function y = uptria2vec(X)
            % Convert upper triangular square sub-matrix to a column vector
            
            [n, ~] = size(X);
            
            y = zeros(n*(n+1)/2, 1);
            
            j = 1;
            for ii = 1:n
                for jj = ii:n
                    y(j) = X(ii, jj);
                    j = j + 1;
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Regressor (RL/ADP)
        function phiVal = phi(y, u)
            
            z = [y; u];
            
            if critStruct == 1 % Quadratic-linear approximator
                phiVal = [ uptria2vec( kron(z, z') ); z ];
                
            elseif critStruct == 2 % Quadratic approximator
                phiVal = uptria2vec( kron(z, z') );
                
            elseif critStruct == 3 % Quadratic approximator, no mixed terms
                phiVal = z.*z;
                
            elseif critStruct == 4 % W(1) y(1)^2 + ... W(p) y(p)^2 + W(p+1) y(1) u(1) + ... W(...) u(1)^2 + ...
                phiVal = [y.^2; kron(y, u) ;u.^2];
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Gradient of regressor (RL/ADP) w.r.t. u
        function grad_u_phiVal = grad_u_phi(y, u)
            
            if critStruct == 1 % Quadratic-linear approximator-------------
                
                z = [y; u];
                dz_du = zeros(p+l, l);
                dz_du(p+1:end, :) = eye(l);
                
                L = ( (p+l) + 1 ) * (p+l) /2 + (p+l);
                grad_phi = zeros(L, l+p);
                
                % Quadratic part
                line_idx = 1;
                for ii = 1:p+l
                    for jj = ii:p+l
                        if ii == jj
                            grad_phi(line_idx, ii) = 2*z(ii);
                        else
                            grad_phi(line_idx, ii) = z(ii);
                            grad_phi(line_idx, jj) = z(jj);
                            
                        end
                        
                        line_idx = line_idx + 1;
                        
                    end
                end
                
                % Linear part
                grad_phi( L-(p+l):L-1,  :) = eye(p+l);
                
                grad_u_phiVal = grad_phi * dz_du;
                
            elseif critStruct == 2 % Quadratic approximator
                
                z = [y; u];
                dz_du = zeros(p+l, l);
                dz_du(p+1:end, :) = eye(l);
                
                L = ( (p+l) + 1 ) * (p+l) /2;
                grad_phi = zeros(L, l+p);
                
                % Quadratic part
                line_idx = 1;
                for ii = 1:p+l
                    for jj = ii:p+l
                        if ii == jj
                            grad_phi(line_idx, ii) = 2*z(ii);
                        else
                            grad_phi(line_idx, ii) = z(ii);
                            grad_phi(line_idx, jj) = z(jj);
                            
                        end
                        
                        line_idx = line_idx + 1;
                        
                    end
                end
                
                grad_u_phiVal = grad_phi * dz_du;
                
            elseif critStruct == 3 % Quadratic approximator, no mixed terms
                grad_u_phiVal = zeros(p + l, l);
                
                grad_u_phiVal(p + 1:end, :) = 2*diag(u);
                
            elseif critStruct == 4 % W(1) y(1)^2 + ... W(p) y(p)^2 + W(p+1) y(1) u(1) + ... W(...) u(1)^2 + ...
                grad_u_phiVal = zeros(p + p*l +l, l);
                
                grad_u_phiVal(p + 1:p + p*l, :) = kron(eye(l), y);
                
                grad_u_phiVal(p + p*l + 1:end, :) = 2*diag(u);
            end
            
        end
        
    end

% Plant model
    function Dx = f(x, u)
        Dx = zeros(5, 1);
        
        Dx(1) = x(4) * cos( x(3) );
        Dx(2) = x(4) * sin( x(3) );
        Dx(3) = x(5);
        Dx(4) = 1/m * u(1);
        Dx(5) = 1/I * u(2);
    end

end