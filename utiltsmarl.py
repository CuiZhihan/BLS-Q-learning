import numpy as np


def q_learning_training(routes_dict, Q_table, num_epochs, pathloss_log, transmission_power, noise):
    """
    使用 Q-learning 更新路由表。

    参数：
        routes_dict (dict): 包含每个节点初始路径的信息。
        Q_table (dict): Q 表，存储 (当前节点, 下一个节点) 对应的 Q 值。
        num_epochs (int): 训练的总轮数。
        pathloss_log (function): 路径损耗计算函数。
        transmission_power (float): 发送功率。
        noise (float): 噪声功率。
    """
    gamma = 0.9  # 折扣因子
    alpha = 0.1  # 学习率
    epsilon = 0.2  # 探索率

    for epoch in range(num_epochs):
        for current_node in routes_dict:
            if current_node not in routes_dict:  # 跳过非有效节点
                continue

            path, coordinates, _ = routes_dict[current_node]
            current_position = coordinates

            # 随机选择下一个节点（带有探索策略）
            if np.random.rand() < epsilon:
                # 随机选择一个可用的下一跳节点
                next_node = np.random.choice([key for key in routes_dict if key != current_node])
            else:
                # 使用当前的 Q 表选择最优的下一跳节点
                next_node = max(
                    [key for key in routes_dict if key != current_node],
                    key=lambda x: Q_table.get((current_node, x), -np.inf),
                    default=None,
                )

            # 如果没有可用的下一跳节点，则跳过
            if next_node is None:
                continue

            # 计算路径损耗和信号功率
            distance = np.sqrt(
                (routes_dict[next_node][1][0] - current_position[0])**2 +
                (routes_dict[next_node][1][1] - current_position[1])**2
            )
            path_loss = pathloss_log(distance)
            signal_power = transmission_power * path_loss

            # 计算干扰功率和奖励
            interference_power = sum(
                transmission_power * pathloss_log(
                    np.sqrt(
                        (routes_dict[other_node][1][0] - current_position[0])**2 +
                        (routes_dict[other_node][1][1] - current_position[1])**2
                    )
                )
                for other_node in routes_dict if other_node != current_node and other_node != next_node
            )
            SINR = signal_power / (interference_power + noise)
            reward = calculate_reward(SINR)

            # 更新 Q 表
            max_future_q = max(
                [Q_table.get((next_node, key), 0) for key in routes_dict if key != next_node],
                default=0
            )
            Q_table[(current_node, next_node)] = Q_table.get((current_node, next_node), 0) + alpha * (
                reward + gamma * max_future_q - Q_table.get((current_node, next_node), 0)
            )

            # 更新路径
            routes_dict[current_node][0].append(next_node)


def update_q_table(Q_table, current_node, next_node, reward, learning_rate=0.1, discount_factor=0.9):
    """
    更新 Q 表。

    Args:
        Q_table (dict): Q 表。
        current_node (int): 当前节点。
        next_node (int): 下一节点。
        reward (float): 奖励值。
        learning_rate (float): 学习率。
        discount_factor (float): 折扣因子。

    Returns:
        None: 更新 Q 表。
    """
    current_q = Q_table.get((current_node, next_node), 0)
    max_future_q = max(
        [Q_table.get((next_node, neighbor), 0) for neighbor in Q_table.keys() if neighbor[0] == next_node],
        default=0,
    )
    updated_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
    Q_table[(current_node, next_node)] = updated_q

def calculate_reward(current_node, next_node, routes_dict):
    """
    根据链路性能计算奖励。

    Args:
        current_node (int): 当前节点。
        next_node (int): 下一节点。
        routes_dict (dict): 路径信息。

    Returns:
        float: 奖励值。
    """
    sinr = calculate_sinr(current_node, next_node, routes_dict)
    return sinr  # 或者其他奖励逻辑

def q_learning_training(routes_dict, Q_table, num_epochs, pathloss_log, transmission_power, noise):
    """
    执行 Q 学习。

    Args:
        routes_dict (dict): 路径信息。
        Q_table (dict): Q 表。
        num_epochs (int): 训练次数。
        pathloss_log (function): 路径损耗函数。
        transmission_power (float): 发射功率。
        noise (float): 噪声功率。

    Returns:
        None: 更新 Q 表。
    """
    for epoch in range(num_epochs):
        for node, (path, _, _) in routes_dict.items():
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                reward = calculate_reward(current_node, next_node, routes_dict)
                update_q_table(Q_table, current_node, next_node, reward)

def optimize_routes_with_q_table(routes_dict, Q_table):
    """
    使用 Q 表优化路径。

    Args:
        routes_dict (dict): 原始路径信息。
        Q_table (dict): Q 表。

    Returns:
        dict: 优化后的路径。
    """
    optimized_routes = {}
    for node, (path, coordinates, packet_length) in routes_dict.items():
        optimized_path = [path[0]]
        current_node = path[0]
        while current_node != path[-1]:
            next_node = max(
                [neighbor for neighbor in Q_table if neighbor[0] == current_node],
                key=lambda x: Q_table.get((current_node, x[1]), 0),
                default=None,
            )
            if next_node:
                optimized_path.append(next_node[1])
                current_node = next_node[1]
            else:
                break
        optimized_routes[node] = [optimized_path, coordinates, packet_length]
    return optimized_routes
