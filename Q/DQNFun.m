function [MRe,Route,Q] = DQNFun(S,D,Layer)

global alpha gamma epsion SNRMat N

maxL = Layer(S);
M = maxL - 1;

if maxL == 1
    return;
end

%% save node by layer根据分层情况存储节点
L = cell(1,M);
for m = 1:M
    L{m} = find(Layer == maxL - m);
end
L = [S L D];


%% Initialization
Q = zeros(N,N); R = zeros(N,N);
for m = 1:M+1
    
    % R table
    R(L{m}, L{m+1}) = SNRMat(L{m}, L{m+1});
    
    % Q table
    if m == M+1
        Q(L{m}, L{m+1}) = R(L{m}, L{m+1});
    end
end

%% Train part
n = 1; Re = []; MRe = []; 
s(1) = S;
Route = [];
while true
    QPre = Q;
    for m = 1:M
        
        % get am randomly         
        idx_am = randperm(length(L{m+1}),1); % am position am节点在表中的位置
        
        am = L{m+1}(idx_am);
        
        % get maxQ(m+1)and a(m+1)
        [Qm1,idx_am1] = max(Q(am,L{m+2})); % idx_am1为a(m+1)节点在表中的位置
        am1 = L{m+2}(idx_am1);
        
        rm1 = R(am,am1); %  a(m+1)节点的r值
        
%         if am == 19
%             disp(1)
%         end
        

        % update Q table 
        
        if m <= M && R(s(m),am) > rm1
            Q(s(m),am) = (1-alpha)*Q(s(m),am) + alpha*(rm1 + gamma*Qm1);
        elseif m <= M && R(s(m),am) < rm1
            Q(s(m),am) = (1-alpha)*Q(s(m),am) + alpha*(R(s(m),am) + gamma*Qm1);
        elseif m == M + 1
            Q(s(m),am) = R(s(m),am);
        end
        
        % update S
        s(m+1) = am;        
    end
    
    RoutePre = Route;
    Route = [];
    Route(1) = S;
    for m = 1:M
        [~,idx_a] = max(Q(Route(m),L{m+1}));
        Route(m+1) = L{m+1}(idx_a);
    end
%     Route = [Route D];
    
    SNRTemp = [];
    for m = 2:length(Route)
        SNRTemp = [SNRTemp SNRMat(Route(m-1),Route(m))];
    end

%     RTemp = [];
%     for m = 2:length(Route)
%         RTemp = [RTemp log(SNRMat(Route(m-1),Route(m)) + 1)];
%     end

      
    % Compute end-to-end rate Rn
    Re(n) = 1/(M+1)*log(sum(SNRTemp) + 1);
%     Re(n) = mean(RTemp);
    
    if n > 1 && Re(n) < Re(n-1) %if smaller, cancel
        Re(n) = Re(n-1);
        %Q = QPre;
        %Route = RoutePre;
    end
    
    MRe(n) =  mean(Re);
        
    if n > N*10 && abs(MRe(n) - MRe(n-9)) < epsion
        break;
    else
        n = n + 1;
    end
    
end

Route = [];
Route(1) = S;
for m = 1:M
    [~,idx_a] = max(Q(Route(m),L{m+1}));
    Route(m+1) = L{m+1}(idx_a);
end
Route = [Route D];




