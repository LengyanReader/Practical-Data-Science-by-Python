%% Initialization
%  Initialize the world, Q-table, and hyperparameters
actions = [1 2 3 4]; %1=down, 2=up, 3=right and 4=left
prob_a = [1 1 1 1]; %probability of taking each action: 1/(1+1+1+1) = 1/4
state = gwinit(1); %change this depending on the world (choose world)

% hyperparameters
n = 0.5; %learning rate %0.1 for 4
gamma = 0.5; %discount factor
episodes = 500; %1:600, 2: 1000 3: 1500, 4: 

r = 0;
Q = zeros(state.ysize, state.xsize, 4);  % each Q(:,:,k) corresponds to value of moving to next state taking one action(up, down, left, right)
P = zeros(state.ysize, state.xsize, episodes); % P(:,:,1) store policy after 1 episode
%% Training loop
%  Train the agent using the Q-learning algorithm.
for e = 1:episodes
     state = gwinit(1); %change this depending on the world
     x=state.pos(2); %get current x state(position)
     y=state.pos(1); %get current y state(position)
     while state.isterminal == 0 % if it is not in invalid area, then move to next state
         %choose action
         [a, oa] = chooseaction(Q, state.pos(1), state.pos(2), actions, prob_a, getepsilon(e,episodes)); % choose action. see  help function
         %make action
         state = gwaction(a);  %update state after taking action
         xnew = state.pos(2); %get new x state(position)
         ynew = state.pos(1); %get new y state(position)
         if state.isvalid == 0
             r = -100; % if new state is in invalid area then give -100 reward         will break in world 4 with -inf
         else  
             r = state.feedback; % if new state is not in invalid area then give reward
         end      
         %updating values
         V = getvalue(Q); 
         Q(y,x,a) = (1-n)*Q(y,x,a) + n*(r + gamma*V(ynew,xnew)); 
         x = xnew;
         y = ynew;
    end
    e
    P(:,:,e) = getpolicy(Q);
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.
state.isvalid = 0;
while state.isvalid == 0
    state = gwinit(1); %change this depending on the world
end
figure(1);
gwdraw('Policy',P(:,:,e));

while state.isterminal == 0
    %choose action
    a = P(state.pos(1),state.pos(2),episodes); % choose action according to the last policy which is the optimal one
    %make action
    state = gwaction(a); % 
    if state.isvalid == 0
    	disp('exited the world');
        break;
    end 
    figure(1);
    gwdraw('Policy',P(:,:,e)); % see gwfunctions
end
figure(1);
gwdraw('Policy',P(:,:,e));
figure(2);
imagesc(V); % plot V-function
title('V-function');
