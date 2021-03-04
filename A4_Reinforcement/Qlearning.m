%% Initialization
%  Initialize the world, Q-table, and hyperparameters

world = 4;              % World chosen
episodes = 1000;
actions = [1 2 3 4];    % Possible actions
prob_a = [1 1 1 1];
eps = 0.9;              % Exploration factor
eta = 0.5;              % Learning rate
gamma = 0.9;            % Discount factor (prio onlong-/short-term reward
eps_update = eps/episodes;

s = gwinit(world);
% imagesc(Q(:,:,1));
%surf(Q(:,:,1));

Q = zeros(s.ysize, s.xsize, 4);
Q(1,:,2) = -inf;
Q(s.ysize,:,1) = -inf;
Q(:,1,4) = -inf;
Q(:,s.xsize,3) = -inf;
%imagesc(Q(:,:,1));
%% Training loop
%  Train the agent using the Q-learning algorithm.

for k = 1:episodes
    s = gwinit(world);
%     k
    while s.isterminal == 0
        [a, opt_a] = chooseaction(Q, s.pos(1), s.pos(2), actions, prob_a, eps);
    
        s_next = gwaction(a); % Next state
        
%        s_next.pos
        
         if s_next.isvalid == 1
         V = getvalue(Q);
         r = s_next.feedback;
            
         Q(s.pos(1),s.pos(2),a) = (1-eta) * Q(s.pos(1),s.pos(2),a) + ...
                                     eta*(r + gamma*V(s_next.pos(1), s_next.pos(2)));
         end
         
%          gwdraw();
         s = s_next;
    end
    eps = eps-eps_update;
end


%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.


epsilon = 0;
e = 1;
% for e = 1:10
    s = gwinit(world);
%     k
    while s.isterminal == 0
        [a, opt_a] = chooseaction(Q, s.pos(1), s.pos(2), actions, prob_a, eps);
    
        s_next = gwaction(opt_a); % Next state
         
        P = getpolicy(Q);
        gwdraw("Policy", P, "Episode", e, "ArrowStyle", "Fast");
        s = s_next;
    end
% end