function [g] = CalculateG(FG,almMatrix)
    nodeNums = size(FG,1);
    %% find leaf node 找出叶子节点
    leaf = [];
    count = 0;
    for i = 1:nodeNums
        if FG(i,1)==0
            root = i; % and root node 顺便找到根节点
        end
        %  Determine whether the current node is a parent node 判断当前节点是否为父节点
        isFather = 0;
        for j = 1:nodeNums
            if i == FG(j,1)
                isFather = 1;
                break;
            end
        end
        if isFather == 0
            count = count + 1;
            leaf(count,1) = i;
        end
    end
    %% calculate G 计算 g
    % 计算方式，父节点的值等于子节点值与节点与父节点的边的值的积之和
    % 节点值初始化
    %Node value initialization
    value = zeros(nodeNums,1); 
    for i = 1:size(leaf,1)
        value(leaf(i,1),1) = 1;
    end
    lastLeaf = leaf;
    newLeaf = [];
    isBreak = 0;
    while 1
        count = 0;
        for i = 1:size(lastLeaf,1)
            % 判断叶子节点中是否有根节点，如果有，跳出循环
            %Determine whether there is a root node in the leaf node,
            %if so, jump out of the loop
            if lastLeaf(i,1) == root 
                isBreak = 1;
                break;
            end
            leafNode = lastLeaf(i,1);
            fatherNode = FG(leafNode,1);
            %sum-product
            value(fatherNode,1) = value(fatherNode,1) + value(leafNode,1)*almMatrix(fatherNode,leafNode);
            % 父节点将称为新的叶节点
            %The parent node will be called the new leaf node
            isExist = 0;
            for k = 1:count
                if newLeaf(k,1) == fatherNode
                    isExist = 1;
                    break;
                end
            end
            if isExist == 0
                count = count + 1;
                newLeaf(count,1) = fatherNode;
            end
        end
        if isBreak == 1
            break;
        end
        lastLeaf = newLeaf;
        newLeaf = [];
    end
    g = value(root,1);
end

