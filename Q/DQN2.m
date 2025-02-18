function [MRe,Route] = DQN2(S,D,Layer)

global alpha gamma epsion SNRMat N

maxL = Layer(S);
M = maxL - 1;

if maxL == 1
    return;
end

%% 根据分层情况存储节点
L = cell(1,M);
for m = 1:M
    L{m} = find(Layer == maxL - m);
end
L = [S L D];


%% 初始化
Q = zeros(N,N); R = zeros(N,N);
for m = 1:M+1
    row = length(L{m}); col = length(L{m+1});
    
    % 计算R
    for i = 1:row
        for j = 1:col
            R(L{m}(i), L{m+1}(j)) = SNRMat(L{m}(i), L{m+1}(j));
        end
    end
    
    % 计算Q
    if m == M+1
        Q(L{m}(i), L{m+1}(j)) = R(L{m}(i), L{m+1}(j));
    end
end

%% 训练
n = 1; Re = []; MRe = []; Route = []; SNRTemp = [];
s(1) = S;
while true
    QPre = Q;
    for m = 1:M
        
        % 随机选择am
        idx_am = randperm(length(L{m+1}),1); % am节点在表中的位置
        am = L{m+1}(idx_am);
        
        % 获取最大Q(m+1)及其对应节点a(m+1)
        [Qm1,idx_am1] = max(Q(am,L{m+2})); % idx_am1为a(m+1)节点在表中的位置
%         if Qm1 == 0
%             idx_am1 = randperm(length(L{m+2}),1);
%         end
        am1 = L{m+2}(idx_am1);
        
        rm1 = R(am,am1); %  a(m+1)节点的r值
        

        % 更新Qm-公式(8)
        if m <= M && R(s(m),am) > rm1
            Q(s(m),am) = (1-alpha)*Q(s(m),am) + alpha*(rm1 + gamma*Qm1);
        elseif m <= M && R(s(m),am) < rm1
            Q(s(m),am) = (1-alpha)*Q(s(m),am) + alpha*(R(s(m),am) + gamma*Qm1);
        elseif m == M + 1
            Q(s(m),am) = R(s(m),am);
        end
        
        % 更新S
        s(m+1) = am;        
    end
    
    RoutePre = Route;
    
    
    Route(1) = S;
    for m = 1:M
        [~,idx_a] = max(Q(Route(m),L{m+1}));
        Route(m+1) = L{m+1}(idx_a);
    end
    
    SNRTempPre = SNRTemp;
    
    
    
    SNRTemp = [];
    for m = 2:length(Route)
        SNRTemp = [SNRTemp SNRMat(Route(m-1),Route(m))];
    end
      
    % Compute end-to-end rate Rn
    Re(n) = 1/(M+1)*log(max(SNRTemp) + 1);
    
    if n > 1 && Re(n) < Re(n-1)
        Re(n) = Re(n-1);
        Q = QPre;
    end
    
    MRe = [MRe mean(Re)];
    
    
    if n > 20 && abs(MRe(n) - MRe(n-3)) < epsion
%     if n >100
        break;
    else
        n = n + 1;
    end
    
end
Route = [Route D];

% figure
% plot(Re,'-o')
% 
% disp(1)


