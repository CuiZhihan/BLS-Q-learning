% 输入：
% almMatrix，费用矩阵
% sp，起点
function [FG] = Dijkstra(almMatrix,sp)
    nodeNums = size(almMatrix,1);
    father = zeros(nodeNums,1);             % 保存节点的父节点，路径的方向为：父节点->子节点，没有父节点的节点值为0
    cost = zeros(nodeNums,1)+inf;           % 费用变量，表示起点到当前节点的费用
    mark = zeros(nodeNums,1);               % 标记变量，1 表示后续不再访问，0 表示后续还要访问
    cost(sp,1) = 0;                         % 起点到自身的费用为 0
    visitedNums = 0;                        % 已访问的节点数
    while visitedNums ~= nodeNums
        % 在未确定节点中找到当前费用值最小的节点，并标记为当前节点，线性扫描
        minCostNode = 0;
        for i = 1:nodeNums
            if mark(i,1)==0 && (minCostNode==0 || cost(i,1)<cost(minCostNode,1))
                minCostNode = i;
            end
        end
        if minCostNode == 0 % 没有未确定的节点
            break;
        else
            mark(minCostNode,1) = 1;
            visitedNums  = visitedNums + 1;
        end
        % 计算从起点到当前确定节点的邻节点的费用值，这里任意两个节点之间都有一条边
        for i = 1:nodeNums
            % 只对未确定节点进行计算
            if mark(i,1)==0
                curCost = cost(minCostNode,1) + almMatrix(minCostNode,i); % 计算当前花费值
                if curCost < cost(i,1)
                    % 找到了从起点到当前的未确定节点更短的路径
                    father(i,1) = minCostNode;  % 更改当前节点的父节点
                    cost(i,1) = curCost;        % 更改费用
                end
            end 
        end
    end
    FG = father;
end

