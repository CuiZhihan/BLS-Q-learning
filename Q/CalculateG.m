function [g] = CalculateG(FG,almMatrix)
    nodeNums = size(FG,1);
    %% find leaf node �ҳ�Ҷ�ӽڵ�
    leaf = [];
    count = 0;
    for i = 1:nodeNums
        if FG(i,1)==0
            root = i; % and root node ˳���ҵ����ڵ�
        end
        %  Determine whether the current node is a parent node �жϵ�ǰ�ڵ��Ƿ�Ϊ���ڵ�
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
    %% calculate G ���� g
    % ���㷽ʽ�����ڵ��ֵ�����ӽڵ�ֵ��ڵ��븸�ڵ�ıߵ�ֵ�Ļ�֮��
    % �ڵ�ֵ��ʼ��
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
            % �ж�Ҷ�ӽڵ����Ƿ��и��ڵ㣬����У�����ѭ��
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
            % ���ڵ㽫��Ϊ�µ�Ҷ�ڵ�
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

