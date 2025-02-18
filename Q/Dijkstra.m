% ���룺
% almMatrix�����þ���
% sp�����
function [FG] = Dijkstra(almMatrix,sp)
    nodeNums = size(almMatrix,1);
    father = zeros(nodeNums,1);             % ����ڵ�ĸ��ڵ㣬·���ķ���Ϊ�����ڵ�->�ӽڵ㣬û�и��ڵ�Ľڵ�ֵΪ0
    cost = zeros(nodeNums,1)+inf;           % ���ñ�������ʾ��㵽��ǰ�ڵ�ķ���
    mark = zeros(nodeNums,1);               % ��Ǳ�����1 ��ʾ�������ٷ��ʣ�0 ��ʾ������Ҫ����
    cost(sp,1) = 0;                         % ��㵽����ķ���Ϊ 0
    visitedNums = 0;                        % �ѷ��ʵĽڵ���
    while visitedNums ~= nodeNums
        % ��δȷ���ڵ����ҵ���ǰ����ֵ��С�Ľڵ㣬�����Ϊ��ǰ�ڵ㣬����ɨ��
        minCostNode = 0;
        for i = 1:nodeNums
            if mark(i,1)==0 && (minCostNode==0 || cost(i,1)<cost(minCostNode,1))
                minCostNode = i;
            end
        end
        if minCostNode == 0 % û��δȷ���Ľڵ�
            break;
        else
            mark(minCostNode,1) = 1;
            visitedNums  = visitedNums + 1;
        end
        % �������㵽��ǰȷ���ڵ���ڽڵ�ķ���ֵ���������������ڵ�֮�䶼��һ����
        for i = 1:nodeNums
            % ֻ��δȷ���ڵ���м���
            if mark(i,1)==0
                curCost = cost(minCostNode,1) + almMatrix(minCostNode,i); % ���㵱ǰ����ֵ
                if curCost < cost(i,1)
                    % �ҵ��˴���㵽��ǰ��δȷ���ڵ���̵�·��
                    father(i,1) = minCostNode;  % ���ĵ�ǰ�ڵ�ĸ��ڵ�
                    cost(i,1) = curCost;        % ���ķ���
                end
            end 
        end
    end
    FG = father;
end

