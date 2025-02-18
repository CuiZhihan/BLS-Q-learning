function [Re,Route] = DQN(S,D,Layer)

global alpha gamma epsion SNRMat

maxL = Layer(S);
M = maxL - 1;

if maxL == 1
    return;
end

%% ���ݷֲ�����洢�ڵ�
L = cell(1,M);
for m = 1:M
    L{m} = find(Layer == maxL - m);
end
L = [S L D];


%% ��ʼ��
QTable = cell(1,M+1); RTable = cell(1,M+1);
for m = 1:M+1
    row = length(L{m}); col = length(L{m+1});
    R = zeros(row,col); Q = zeros(row,col);
    
    % ����R
    for i = 1:row
        for j = 1:col
            R(i,j) = SNRMat(L{m}(i), L{m+1}(j));
        end
    end
    
    % ����Q
    if m == M+1
        Q = R;
    end
    QTable{m} = Q; 
    RTable{m} = R;
end

%% ѵ��
n = 1; Re = []; MRe = [];
s(1) = S;
while true
    for m = 1:M
        idx_S = find(L{m} == s(m)); % Sm�ڵ��ڱ��е�λ��
        
        % ���ѡ��am
        idx_am = randperm(length(L{m+1}),1); % am�ڵ��ڱ��е�λ��
        am = L{m+1}(idx_am);
        
        % ��ȡ���Q(m+1)�����Ӧ�ڵ�a(m+1)
        [Qm1,idx_am1] = max(QTable{m+1}(idx_am,:)); % idx_am1Ϊa(m+1)�ڵ��ڱ��е�λ��
        if Qm1 == 0
            idx_am1 = randperm(size(QTable{m+1},2),1);
        end
        rm1 = RTable{m+1}(idx_am,idx_am1); %  a(m+1)�ڵ��rֵ
        am1 = L{m+2}(idx_am1);

        % ����Qm-��ʽ(8)
        if m <= M && RTable{m}(idx_S,idx_am) > rm1
            QTable{m}(idx_S,idx_am) = (1-alpha)*QTable{m}(idx_S,idx_am) + alpha*(rm1 + gamma*Qm1);
        elseif m <= M && RTable{m}(idx_S,idx_am) < rm1
            QTable{m}(idx_S,idx_am) = (1-alpha)*QTable{m}(idx_S,idx_am) + alpha*(RTable{m}(idx_S,idx_am) + gamma*Qm1);
        elseif m == M + 1
            QTable{m}(idx_S,idx_am) = RTable{m}(idx_S,idx_am);
        end
        
        % ����S
        s(m+1) = am;        
    end
    
    
    Route(1) = S;
    for m = 1:M
        idx_S = find(L{m} == Route(m)); % Sm�ڵ��ڱ��е�λ��
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

%% Ԥ��
Route(1) = S;
for m = 1:M
    idx_S = find(L{m} == Route(m)); % Sm�ڵ��ڱ��е�λ��
    [~,idx_a] = max(QTable{m}(idx_S,:));
    Route(m+1) = L{m+1}(idx_a);
end

end

