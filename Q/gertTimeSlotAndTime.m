function [TimeSlot,Time] = gertTimeSlotAndTime(RouteAll,PMat,maxRecNum)
global eta B N L

%% Divide Time slot ʱ϶����
TimeSlot = {};
currentIdx = ones(size(RouteAll));
while true
    SendNode =zeros(1,N); %��¼��Ϊ���ͽڵ�Ĵ���
    RecNode = zeros(1,N); %��¼��Ϊ���սڵ�Ĵ���
    Path = [];
    for i = 1:length(RouteAll)
        if currentIdx(i) == length(RouteAll{i})
            continue;
        end
        for j = currentIdx(i):length(RouteAll{i})-1
            s = RouteAll{i}(j);
            d = RouteAll{i}(j+1);
            if SendNode(s) < maxRecNum && RecNode(s) == 0 &&  SendNode(d) == 0 && RecNode(d) < maxRecNum
                currentIdx(i) = currentIdx(i) + 1;
                SendNode(s) = SendNode(s) + 1;
                RecNode(d) = RecNode(d) + 1;
                Path = [Path; [s d]];
            end
        end
    end
    if ~isempty(Path)
        TimeSlot = [TimeSlot Path];
    else
        break;
    end    
end
    
%% Transmission time ���㴫��ʱ��
Time = [];
for i = 1:length(TimeSlot)
    rData = [];
    % get Prx ��ȡ����
    P = zeros(1, size(TimeSlot{i},1));
    for j = 1:size(TimeSlot{i},1)
        P(j) = PMat(TimeSlot{i}(j,1), TimeSlot{i}(j,2));
    end
    RecNode = unique(TimeSlot{i}(:,2));
    for j = 1:length(RecNode)
        ind = find(TimeSlot{i}(:,2) == RecNode(j)); 
        P1 = sum(P(ind)); P2 = sum(P) - P1;
        sinr = P1/(P2 + eta*B);
        rData = [rData B*(log2(1+sinr))];
    end
    Time = [Time max(L./rData)];
end
end

