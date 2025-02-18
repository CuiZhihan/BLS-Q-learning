function [Re,Route] = DQN(S,D,Layer)

global alpha gamma epsion SNRMat

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
QTable = cell(1,M+1); RTable = cell(1,M+1);
for m = 1:M+1
    row = length(L{m}); col = length(L{m+1});
    R = zeros(row,col); Q = zeros(row,col);
    
    % 计算R
    for i = 1:row
        for j = 1:col
            R(i,j) = SNRMat(L{m}(i), L{m+1}(j));
        end
    end
    
    % 计算Q
    if m == M+1
        Q = R;
    end
    QTable{m} = Q; 
    RTable{m} = R;
end

%% 训练
n = 1; Re = []; MRe = [];
s(1) = S;
while true
    for m = 1:M
        idx_S = find(L{m} == s(m)); % Sm节点在表中的位置
        
        % 随机选择am
        idx_am = randperm(length(L{m+1}),1); % am节点在表中的位置
        am = L{m+1}(idx_am);
        
        % 获取最大Q(m+1)及其对应节点a(m+1)
        [Qm1,idx_am1] = max(QTable{m+1}(idx_am,:)); % idx_am1为a(m+1)节点在表中的位置
        if Qm1 == 0
            idx_am1 = randperm(size(QTable{m+1},2),1);
        end
        rm1 = RTable{m+1}(idx_am,idx_am1); %  a(m+1)节点的r值
        am1 = L{m+2}(idx_am1);

        % 更新Qm-公式(8)
        if m <= M && RTable{m}(idx_S,idx_am) > rm1
            QTable{m}(idx_S,idx_am) = (1-alpha)*QTable{m}(idx_S,idx_am) + alpha*(rm1 + gamma*Qm1);
        elseif m <= M && RTable{m}(idx_S,idx_am) < rm1
            QTable{m}(idx_S,idx_am) = (1-alpha)*QTable{m}(idx_S,idx_am) + alpha*(RTable{m}(idx_S,idx_am) + gamma*Qm1);
        elseif m == M + 1
            QTable{m}(idx_S,idx_am) = RTable{m}(idx_S,idx_am);
        end
        
        % 更新S
        s(m+1) = am;        
    end
    
    
    Route(1) = S;
    for m = 1:M
        idx_S = find(L{m} == Route(m)); % Sm节点在表中的位置
        [~,idx_a] = max(QTable{m}(idx_S,:));
        Route(m+1) = L{m+1}(idx_a);
    end
    
    SNRTemp = [];
    for m = 2:length(Route)
        SNRTemp = [SNRTemp SNRMat(Route(m-1),Route(m))];
    end
      
    % Compute end-to-end rate Rn
    Re(n) = 1/(M+1)*log(min(SNRTemp) + 1);
    
    
    MRe(n) = mean(Re);
    
    
%     if n > 10 && abs(Re(n) - Re(n-9)) < epsion
    if n >800
        break;
    else
        n = n + 1;
    end
    
end

figure
plot(Re)

%% 预测
Route(1) = S;
for m = 1:M
    idx_S = find(L{m} == Route(m)); % Sm节点在表中的位置
    [~,idx_a] = max(QTable{m}(idx_S,:));
    Route(m+1) = L{m+1}(idx_a);
end

end

