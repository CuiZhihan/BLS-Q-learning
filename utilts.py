import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json
B = 40 * 10 ** 6  # Hz  带宽
d0 = 1  # decorrelation distance m
attenuation_constant = 3.5
shadowing = 4 # dB
wall_attenuation = 0 # dB
noise_level = 3.9810717055 * 10 ** -18 # mW/Hz, -174 dBm/Hz
transmission_power = 200  # mW
noise = B * noise_level

def mw_to_dbm(power_mw):
    return 10 * np.log10(power_mw)

def calculate_interference(global_table, node_positions, server_positions, transmission_power):
    """
    计算网络中每条链路的干扰功率，并返回干扰总和和平均值。

    参数:
        global_table: DataFrame, 包含节点和路径信息。
        node_positions: dict, 节点坐标，{Node_ID: {'Node_X': x, 'Node_Y': y}}。
        server_positions: dict, 服务器坐标，{Server_ID: {'Node_X': x, 'Node_Y': y}}。
        transmission_power: float, 节点的发射功率。

    返回:
        tuple:
            - dict: {Node_ID: [干扰功率链表]}。
            - float: 总干扰功率。
            - float: 平均干扰功率。
    """
    # 标准化服务器坐标格式
    for key, value in server_positions.items():
        server_positions[key] = {'Node_X': value['Server_X'], 'Node_Y': value['Server_Y']}

    interference_results = {}
    total_interference_power = 0
    total_links = 0

    for index, row in global_table.iterrows():
        if pd.isna(row['route']):
            continue  # 跳过无路径的节点

        # 解析路径
        route = json.loads(row['route']) if isinstance(row['route'], str) else row['route']
        interference_levels = []

        for i in range(len(route) - 1):
            src, dst = route[i], route[i + 1]

            # 获取目标节点的坐标
            if dst < 0:  # 如果目标节点是服务器
                dst_coords = server_positions[dst]
            else:
                dst_coords = node_positions[dst]

            link_interference = 0

            for other_index, other_row in global_table.iterrows():
                if pd.isna(other_row['route']) or other_index == index:
                    continue  # 跳过自身和无路径的节点

                other_route = json.loads(other_row['route']) if isinstance(other_row['route'], str) else other_row['route']
                if len(other_route) > i:
                    other_src = other_route[i]
                    if other_src < 0:  # 如果干扰节点是服务器
                        other_coords = server_positions[other_src]
                    else:
                        other_coords = node_positions[other_src]

                    # 计算距离
                    distance = np.sqrt((other_coords['Node_X'] - dst_coords['Node_X']) ** 2 +
                                       (other_coords['Node_Y'] - dst_coords['Node_Y']) ** 2)

                    # 添加对距离为零的处理
                    if distance == 0:
                        interference_power = 0  # 或者设置为一个很小的值
                    else:
                        interference_power = pathloss_log(distance) * transmission_power

                    # 检查是否为 inf 或 NaN
                    if np.isinf(interference_power) or np.isnan(interference_power):
                        interference_power = 0

                    link_interference += interference_power

            # 存储当前链路的干扰功率
            interference_levels.append(link_interference)

            # 更新总干扰功率和链路数
            total_interference_power += link_interference
            total_links += 1

        # 保存每个节点的干扰功率列表
        interference_results[row['Node_ID']] = interference_levels

    # 计算平均干扰功率
    average_interference_power = total_interference_power / total_links if total_links > 0 else 0

    return interference_results, total_interference_power, average_interference_power

class NodeAgent:
    def __init__(self, node_id, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.node_id = node_id
        self.q_table = {}  # Q 表，键为 (state, action)，值为 Q 值
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.num_actions = num_actions  # 动作数量

    def choose_action(self, state):
        # ε-greedy 策略
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # 随机动作
        q_values = self.q_table.get(state, np.zeros(self.num_actions))
        return np.argmax(q_values)  # 贪婪策略

    def update_q(self, state, action, reward, next_state):
        # 初始化 Q 表
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)
        # Q 学习更新
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

def calculate_path_sinr_and_rates(routes_dict, pathloss_log, transmission_power, noise, B):
    """
    计算每条路径中每一段链路的 SINR 和传输速率，并返回每条路径的总传输速率。

    Args:
        routes_dict (dict): 节点路径字典，键是节点 ID，值是包含路径和位置列表的路径信息。
        pathloss_log (function): 计算路径增益的函数。
        transmission_power (float): 发射功率。
        noise (float): 噪声功率。
        B (float): 带宽（Hz）。

    Returns:
        dict: 包含每个节点路径的 SINRs、Rates 和总 PathRate 的字典。
    """
    # 初始化存储结果的字典
    path_sinr_rates = {}

    # 遍历每个节点及其路径
    for node, (path, coordinates, pack_len) in routes_dict.items():
        sinr_list = []  # 存储当前路径中每一段链路的 SINR
        rates_list = []  # 存储当前路径中每一段链路的速率
        time_list = []
        # 遍历路径中的每一段链路
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]

            # 获取源节点和目标节点的坐标
            src_x, src_y = coordinates if src == node else (routes_dict[src][1][0], routes_dict[src][1][1])
            dst_x, dst_y = coordinates if dst == node else (routes_dict[dst][1][0], routes_dict[dst][1][1])

            # 计算信号功率 S
            distance = np.sqrt((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2)
            G = pathloss_log(distance)
            S = G * transmission_power

            # 计算噪声功率 N
            N = noise

            # 计算干扰功率 I
            I = 0  # 初始化干扰功率

            for other_node, (other_path, other_coordinates, _) in routes_dict.items():
                if other_node == node:
                    continue  # 跳过当前节点

                # 检查其他路径是否在同一时隙干扰当前链路
                # if len(other_path) > i + 1 and other_path[i + 1] == dst:
                if len(other_path) > i + 1:
                    # 其他路径在同一时隙向同一目标发送信号
                    other_src = other_path[i]
                    other_src_x, other_src_y = routes_dict[other_src][1]
                    other_distance = np.sqrt((other_src_x - dst_x) ** 2 + (other_src_y - dst_y) ** 2)
                    other_G = pathloss_log(other_distance)
                    I += other_G * transmission_power

            # 计算 SINR
            SINR = S / (I + N)
            sinr_list.append(SINR)

            # 计算当前链路的传输速率
            rate = B * np.log2(1 + SINR)
            rates_list.append(rate)
            if rate == 0:
                link_time = 0
            else:
                link_time = pack_len / rate
            time_list.append(link_time)
        # 计算路径的总传输速率
        path_rate = np.average(rates_list) if rates_list else 0
        if path_rate == 0:
            path_transmit_time = 0
        else:
            path_transmit_time = pack_len / path_rate
        path_sinr_rates[node] = {'SINRs': sinr_list, 'Rates': rates_list, 'PathRate': path_rate,
                                 'Time': path_transmit_time}

    return path_sinr_rates

def calculate_path_sinr_and_rates_2(routes_dict, pathloss_log, transmission_power, noise, B):
    """
    计算无线多跳网络的总时隙个数、每个时隙的时间、总时间和总干扰功率。
    """
    total_capacity = 0
    unfinished_links = {}  # 未完成的链路 {node: 当前链路索引}
    node_results = {node: {'total_time': 0, 'slots': 0} for node in routes_dict}
    timeslot_times = []  # 记录每个时隙的时间
    total_interference = 0  # 记录总干扰功率

    # 初始化未完成的链路
    for node in routes_dict:
        unfinished_links[node] = 0

    while unfinished_links:
        successful_links = []  # 本时隙成功传输的链路
        receiving_nodes = {}
        max_link_time = 0  # 当前时隙的最大链路传输时间

        # 遍历所有未完成的链路
        for node, link_index in list(unfinished_links.items()):
            path, _, pack_len = routes_dict[node]
            if link_index >= len(path) - 1:
                del unfinished_links[node]  # 链路传输完成
                continue

            src, dst = path[link_index], path[link_index + 1]
            if dst == path[-1]:  # 服务器无限接收
                successful_links.append((node, src, dst))
                continue

            receiving_nodes.setdefault(dst, []).append((node, src, dst))

        # 接收节点容量限制处理
        for dst, links in receiving_nodes.items():
            if len(links) > 2:  # 随机选择两个，其他等待
                selected_links = random.sample(links, 2)
            else:
                selected_links = links
            successful_links.extend(selected_links)

        # 计算干扰和链路的传输时间
        for node, src, dst in successful_links:
            _, coordinates, pack_len = routes_dict[node]
            src_x, src_y = routes_dict[src][1] if src != node else coordinates
            dst_x, dst_y = routes_dict[dst][1] if dst != node else coordinates

            # 信号功率计算
            distance = np.sqrt((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2)
            G = max(pathloss_log(distance), 1e-6)
            S = G * transmission_power

            # 干扰功率计算
            I = 0
            for other_node, other_src, _ in successful_links:
                if other_node == node:
                    continue  # 排除当前链路
                if other_src == dst:  # 排除发送节点和接收节点相同的情况
                    continue
                other_src_x, other_src_y = routes_dict[other_src][1]
                interference_distance = np.sqrt((other_src_x - dst_x) ** 2 + (other_src_y - dst_y) ** 2)
                I += max(pathloss_log(interference_distance), 1e-6) * transmission_power

            # 计算 SINR 和速率
            SINR = S / (I + noise)
            rate = B * np.log2(1 + SINR)
            link_time = pack_len / rate if rate > 0 else float('inf')
            max_link_time = max(max_link_time, link_time)

            # 累加总干扰功率
            total_interference += I

            # 更新链路进度
            unfinished_links[node] += 1

        # 记录当前时隙的时间
        if max_link_time > 0:
            timeslot_times.append(max_link_time)

        # 更新节点传输时间
        for node, _, _ in successful_links:
            node_results[node]['total_time'] += max_link_time
            node_results[node]['slots'] += 1

    # 计算网络总容量
    total_capacity = sum(routes_dict[node][2] / node_results[node]['total_time']
                         for node in node_results if node_results[node]['total_time'] > 0)
    total_time = sum(timeslot_times)  # 总时间

    return total_capacity, total_time, total_interference

def calculate_sinr_and_rate(route, routes_dict, pathloss_log, transmission_power, noise):
    """
    计算路径上的 SINR 和链路速率。

    参数:
        route (list): 当前节点的路径。
        routes_dict (dict): 所有节点的路径信息。
        pathloss_log (function): 路径损耗计算函数。
        transmission_power (float): 传输功率。
        noise (float): 噪声功率。

    返回:
        tuple: (sinr, rate) 当前路径的总信噪比和链路速率。
    """
    sinr_list = []
    rate_list = []
    for i in range(len(route) - 1):
        src = route[i]
        dst = route[i + 1]

        # 获取链路距离
        src_x, src_y = routes_dict[src][1]
        dst_x, dst_y = routes_dict[dst][1]
        distance = ((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2) ** 0.5

        # 计算信号功率
        path_gain = pathloss_log(distance)
        signal_power = path_gain * transmission_power

        # 计算干扰功率
        interference_power = 0
        for other_node, (other_path, _, _) in routes_dict.items():
            if other_node == src or len(other_path) <= i:
                continue
            other_src = other_path[i]
            other_src_x, other_src_y = routes_dict[other_src][1]
            other_distance = ((other_src_x - dst_x) ** 2 + (other_src_y - dst_y) ** 2) ** 0.5
            if other_distance > 0:
                other_path_gain = pathloss_log(other_distance)
                interference_power += other_path_gain * transmission_power

        # 计算 SINR
        sinr = signal_power / (interference_power + noise)
        sinr_list.append(sinr)

        # 计算链路速率
        rate = np.log2(1 + sinr)
        rate_list.append(rate)

    # 平均 SINR 和速率
    avg_sinr = np.mean(sinr_list) if sinr_list else 0
    avg_rate = np.mean(rate_list) if rate_list else 0

    return avg_sinr, avg_rate

def check_completion(routes_dict):
    """
    检查是否所有节点都完成了传输任务。

    参数:
        routes_dict (dict): 所有节点的路径信息。

    返回:
        bool: True 如果所有节点已完成任务；否则 False。
    """
    for node, (path, _, _) in routes_dict.items():
        if len(path) > 1:  # 如果路径长度大于1，说明还有未完成的路径
            return False
    return True

def simulate_action(node, action, routes_dict, pathloss_log, transmission_power, noise):
    """
    模拟节点执行动作后，返回下一状态、奖励和是否完成。
    """
    # 执行动作（如选择路径）
    current_route = routes_dict[node][0]
    new_route = current_route[:action + 1]  # 假设动作表示跳数或路径更新
    routes_dict[node][0] = new_route

    # 计算奖励
    sinr, rate = calculate_sinr_and_rate(new_route, routes_dict, pathloss_log, transmission_power, noise)
    reward = rate  # 奖励可以是路径速率或其他指标

    # 生成下一状态
    next_state = (sinr, rate)  # 示例：下一状态可以是 SINR 和速率
    done = check_completion(routes_dict)  # 判断是否所有任务完成

    return next_state, reward, done

def calculate_interference_for_path(path, routes_dict, pathloss_log, transmission_power, noise):
    """
    计算路径中的干扰功率（仅针对成功发送的链路）。

    Args:
        path (list): 当前节点到目标的路径。
        routes_dict (dict): 所有节点路径和相关信息。
        pathloss_log (function): 用于计算路径损耗的函数。
        transmission_power (float): 发送功率（mW）。
        noise (float): 噪声功率（mW）。

    Returns:
        float: 当前路径的总干扰功率。
    """
    total_interference = 0

    # 遍历路径的每一段链路
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]

        # 当前链路的干扰
        link_interference = 0

        # 遍历所有其他路径，计算干扰
        for other_node, (other_path, _, _) in routes_dict.items():
            if other_node == src:
                continue  # 跳过自己

            # 确保干扰发生在同一个时隙
            if len(other_path) > i:
                other_src = other_path[i]
                other_dst = other_path[i + 1]

                # 干扰发生在目标节点
                if other_dst == dst:
                    distance = np.sqrt((routes_dict[other_src][1][0] - routes_dict[dst][1][0])**2 +
                                       (routes_dict[other_src][1][1] - routes_dict[dst][1][1])**2)

                    # 计算路径增益
                    if distance > 0:
                        path_gain = pathloss_log(distance)
                        interference_power = path_gain * transmission_power
                    else:
                        interference_power = 0

                    link_interference += interference_power

        # 更新当前链路的总干扰功率
        total_interference += link_interference

    return total_interference

def update_q_table(Q_table, current_node, next_node, reward, learning_rate=0.1, discount_factor=0.9):
    """
    更新 Q 表，基于 Q 学习更新规则。

    Args:
        Q_table (dict): Q 表，记录状态-动作对的 Q 值。
        current_node (int): 当前节点。
        next_node (int): 下一步选择的节点。
        reward (float): 当前动作的奖励。
        learning_rate (float): 学习率，用于控制 Q 值更新的速度。
        discount_factor (float): 折扣因子，用于权衡未来奖励的重要性。

    Returns:
        None: 直接更新 Q 表。
    """
    # 获取当前 Q 值
    current_q = Q_table.get((current_node, next_node), 0)

    # 获取下一个状态的最大 Q 值
    max_future_q = max([Q_table.get((next_node, neighbor), 0) for neighbor in Q_table.keys() if neighbor[0] == next_node], default=0)

    # Q 学习更新规则
    updated_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)

    # 更新 Q 表
    Q_table[(current_node, next_node)] = updated_q


def marl_training(routes_dict, num_epochs, pathloss_log, transmission_power, noise):
    """
    基于 MARL 的训练算法，优化路径选择以实现多跳通信。
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for node, (current_path, coordinates, packet_length) in routes_dict.items():
            # 如果已经到达目标节点，则跳过
            if current_path[-1] == len(routes_dict) - 1:  # 假设服务器是最后一个节点
                continue

            # 获取当前节点的最后一跳
            current_node = current_path[-1]

            # 计算当前节点的 SINR
            sinr_matrix = calculate_sinr_mat(routes_dict, pathloss_log, transmission_power, noise)
            sinr_values = sinr_matrix[current_node, :]

            # 选择最佳下一跳
            next_node = select_next_hop_marl(current_node, sinr_values, current_path)

            # 更新路径
            if next_node not in current_path:  # 防止循环
                current_path.append(next_node)

            # 更新奖励
            interference = calculate_interference_for_path(current_path, routes_dict, pathloss_log, transmission_power)
            sinr = sinr_values[next_node]
            reward = marl_reward_function(node, current_path, interference, sinr, len(routes_dict) - 1)

            # 更新 Q 表（或者其他策略）
            update_q_table(node, current_node, next_node, reward)

        # 可视化每轮的路径
        print(f"Routes after epoch {epoch + 1}:")
        for node, (path, _, _) in routes_dict.items():
            print(f"Node {node}: {path}")

def select_next_hop_marl(current_node, sinr_values, current_path):
    """
    基于 SINR 和当前路径选择下一跳节点。
    """
    # 排除已经在路径中的节点，避免循环
    valid_nodes = [i for i in range(len(sinr_values)) if i not in current_path]

    # 如果没有可选节点，返回自己（异常处理）
    if not valid_nodes:
        return current_node

    # 选择 SINR 最大的节点作为下一跳
    next_node = max(valid_nodes, key=lambda x: sinr_values[x])
    return next_node

def marl_reward_function(node, path, interference, sinr, goal_node):
    """
    根据路径的干扰和 SINR 分配奖励。
    """
    if path[-1] == goal_node:
        reward = 10  # 到达目标节点的高奖励
    else:
        reward = -0.1 * interference + 0.5 * np.log2(1 + sinr)  # 平衡干扰和 SINR
    return reward

def calculate_network_metrics(routes_dict, pathloss_log, transmission_power, noise, B):
    """
    计算无线多跳网络的总时隙个数、每个时隙的时间、总时间、总干扰功率、平均路径速率和平均干扰功率。
    """
    total_capacity = 0
    unfinished_links = {}  # 未完成的链路 {node: 当前链路索引}
    node_results = {node: {'total_time': 0, 'slots': 0, 'link_rates': []} for node in routes_dict}
    timeslot_times = []  # 记录每个时隙的时间
    total_interference = 0  # 记录总干扰功率
    successful_link_count = 0  # 记录成功传输的链路数

    # 初始化未完成的链路
    for node in routes_dict:
        unfinished_links[node] = 0

    while unfinished_links:
        successful_links = []  # 本时隙成功传输的链路
        receiving_nodes = {}
        max_link_time = 0  # 当前时隙的最大链路传输时间

        # 遍历所有未完成的链路
        for node, link_index in list(unfinished_links.items()):
            path, _, pack_len = routes_dict[node]
            if link_index >= len(path) - 1:
                del unfinished_links[node]  # 链路传输完成
                continue

            src, dst = path[link_index], path[link_index + 1]
            if dst == path[-1]:  # 服务器无限接收
                successful_links.append((node, src, dst))
                continue

            receiving_nodes.setdefault(dst, []).append((node, src, dst))

        # 接收节点容量限制处理
        for dst, links in receiving_nodes.items():
            if len(links) > 2:  # 随机选择两个，其他等待
                selected_links = random.sample(links, 2)
            else:
                selected_links = links
            successful_links.extend(selected_links)

        # 计算干扰和链路的传输时间
        for node, src, dst in successful_links:
            _, coordinates, pack_len = routes_dict[node]
            src_x, src_y = routes_dict[src][1] if src != node else coordinates
            dst_x, dst_y = routes_dict[dst][1] if dst != node else coordinates

            # 信号功率计算
            distance = np.sqrt((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2)
            G = pathloss_log(distance)
            S = G * transmission_power

            # 干扰功率计算
            I = 0
            for other_node, other_src, _ in successful_links:
                if other_node == node:
                    continue  # 排除当前链路
                if other_src == dst:  # 排除发送节点和接收节点相同的情况
                    continue
                other_src_x, other_src_y = routes_dict[other_src][1]
                interference_distance = np.sqrt((other_src_x - dst_x) ** 2 + (other_src_y - dst_y) ** 2)
                I += pathloss_log(interference_distance) * transmission_power
                #print(I)
            # 计算 SINR 和速率
            SINR = S / (I + noise)
            rate = B * np.log2(1 + SINR)
            link_time = pack_len / rate if rate > 0 else float('inf')
            max_link_time = max(max_link_time, link_time)

            # 累加总干扰功率和成功链路数
            total_interference += I
            successful_link_count += 1

            # 存储链路速率
            node_results[node]['link_rates'].append(rate)

            # 更新链路进度
            unfinished_links[node] += 1

        # 记录当前时隙的时间
        if max_link_time > 0:
            timeslot_times.append(max_link_time)

        # 更新节点传输时间
        for node, _, _ in successful_links:
            node_results[node]['total_time'] += max_link_time
            node_results[node]['slots'] += 1

    # 计算网络总容量
    total_capacity = sum(routes_dict[node][2] / node_results[node]['total_time']
                         for node in node_results if node_results[node]['total_time'] > 0)
    total_time = sum(timeslot_times)  # 总时间

    # 计算平均路径速率
    path_rates = [np.mean(result['link_rates']) for result in node_results.values() if result['link_rates']]
    average_path_rate = np.mean(path_rates) if path_rates else 0

    # 计算平均干扰功率
    average_interference = total_interference / successful_link_count if successful_link_count > 0 else 0

    return total_capacity, total_time, total_interference, average_interference, average_path_rate

def calculate_network_performance(global_table, node_positions, server_positions, transmission_power, pathloss_log, noise, B):
    """
    计算网络中时隙调度、干扰功率、容量和时间等指标。

    参数:
        global_table: DataFrame, 包含所有节点和路径信息。
        node_positions: dict, 节点坐标，{Node_ID: {'Node_X': x, 'Node_Y': y}}。
        server_positions: dict, 服务器坐标，{Server_ID: {'Node_X': x, 'Node_Y': y}}。
        transmission_power: float, 发射功率。
        pathloss_log: function, 路径损耗计算函数。
        noise: float, 噪声功率。
        B: float, 带宽。

    返回:
        tuple:
            - 总网络容量 (float)
            - 总干扰功率 (float)
            - 平均干扰功率 (float)
            - 总时间 (float)
            - 每个节点的路径总时间和网络容量 (dict)
            - 平均路径速率 (float)
    """
    # 标准化服务器坐标格式
    for key, value in server_positions.items():
        server_positions[key] = {'Node_X': value['Server_X'], 'Node_Y': value['Server_Y']}

    total_interference_power = 0  # 总干扰功率
    total_capacity = 0  # 总网络容量
    total_time = 0  # 总网络时间
    successful_links = 0  # 成功传输的链路数
    node_results = {}  # 每个节点的总时间和容量
    path_rates = []  # 存储每条路径的平均传输速率

    unfinished_links = {}  # 每个节点未完成的链路索引 {Node_ID: 当前链路索引}
    waiting_priority = {}  # 每个节点的等待优先级
    current_timeslot = 0  # 当前时隙
    timeslot_times = []  # 每个时隙的时间

    # 初始化未完成的链路
    for index, row in global_table.iterrows():
        if not pd.isna(row['route']):
            route = json.loads(row['route']) if isinstance(row['route'], str) else row['route']
            unfinished_links[row['Node_ID']] = 0
            waiting_priority[row['Node_ID']] = 0
            node_results[row['Node_ID']] = {'total_time': 0, 'capacity': 0, 'link_rates': []}

    while unfinished_links:
        current_timeslot += 1
        receiving_nodes = {}  # 本时隙内接收节点和发送者 {Dst_ID: [(Node_ID, Src_ID)]}
        max_time_in_slot = 0  # 当前时隙的最大传输时间

        # 遍历未完成的链路
        for node, link_index in list(unfinished_links.items()):
            row = global_table[global_table['Node_ID'] == node].iloc[0]
            route = json.loads(row['route']) if isinstance(row['route'], str) else row['route']
            pack_len = row['Packet_Length']

            if link_index >= len(route) - 1:
                del unfinished_links[node]  # 路径传输完成
                continue

            src, dst = route[link_index], route[link_index + 1]
            dst_coords = server_positions[dst] if dst < 0 else node_positions[dst]

            # 添加到接收队列
            receiving_nodes.setdefault(dst, []).append((node, src))

        # 分配接收节点的发送者
        for dst, senders in receiving_nodes.items():
            if len(senders) > 2 and dst >= 0:  # 接收节点最多接收两个发送者（非服务器）
                senders = sorted(senders, key=lambda x: waiting_priority[x[0]], reverse=True)
                selected_senders = senders[:2]
                for sender, _ in senders[2:]:
                    waiting_priority[sender] += 1  # 增加等待优先级
            else:
                selected_senders = senders  # 全部接收

            # 计算每个成功发送链路的传输时间和干扰
            for node, src in selected_senders:
                src_coords = server_positions[src] if src < 0 else node_positions[src]
                distance = np.sqrt((src_coords['Node_X'] - dst_coords['Node_X']) ** 2 +
                                   (src_coords['Node_Y'] - dst_coords['Node_Y']) ** 2)
                signal_power = pathloss_log(distance) * transmission_power

                # 计算干扰功率
                interference = 0
                for other_node, other_src in selected_senders:
                    if other_node == node:
                        continue  # 排除当前链路
                    if other_src == dst:  # 排除发送节点和接收节点相同的情况
                        continue
                    other_coords = server_positions[other_src] if other_src < 0 else node_positions[other_src]
                    other_distance = np.sqrt((other_coords['Node_X'] - dst_coords['Node_X']) ** 2 +
                                             (other_coords['Node_Y'] - dst_coords['Node_Y']) ** 2)
                    if other_distance == 0:
                        continue
                    interference += pathloss_log(other_distance) * transmission_power
                    # print(other_distance)
                    # print(interference)
                # 计算 SINR 和速率
                SINR = signal_power / (interference + noise)
                rate = B * np.log2(1 + SINR)
                print(rate)
                link_time = pack_len / rate if rate > 0 else float('inf')

                # 更新干扰功率和时隙时间
                total_interference_power += interference
                max_time_in_slot = max(max_time_in_slot, link_time)
                successful_links += 1

                # 更新节点状态
                unfinished_links[node] += 1
                node_results[node]['total_time'] += link_time
                node_results[node]['link_rates'].append(rate)

        # 记录时隙时间
        if max_time_in_slot > 0:
            timeslot_times.append(max_time_in_slot)
            total_time += max_time_in_slot

    # 计算网络总容量和路径速率
    for node, result in node_results.items():
        pack_len = global_table[global_table['Node_ID'] == node]['Packet_Length'].iloc[0]
        result['capacity'] = pack_len / result['total_time'] if result['total_time'] > 0 else 0
        total_capacity += result['capacity']
        path_rate = np.mean([rate for rate in result['link_rates'] if np.isfinite(rate)]) if result['link_rates'] else 0
        path_rates.append(path_rate)

    average_interference = total_interference_power / successful_links if successful_links > 0 else 0
    average_path_rate = np.mean(path_rates) if path_rates else 0

    return total_capacity, total_interference_power, average_interference, total_time, node_results, average_path_rate

def calculate_sinr_mat(routes_dict, pathloss_log, transmission_power, noise, B):
    """
    计算每条路径中每一段链路的 SINR 和传输速率，并返回每条路径的总传输速率，同时返回节点之间的 SINR 矩阵。

    Args:
        routes_dict (dict): 节点路径字典，键是节点 ID，值是包含路径和位置列表的路径信息。
        pathloss_log (function): 计算路径增益的函数。
        transmission_power (float): 发射功率。
        noise (float): 噪声功率。
        B (float): 带宽（Hz）。

    Returns:
        dict: 包含每个节点路径的 SINRs、Rates 和总 PathRate 的字典。
        np.array: 节点之间的 SINR 矩阵。
    """

    # 获取节点数量，用于创建 SINR 矩阵
    num_nodes = len(routes_dict)
    sinr_matrix = np.zeros((num_nodes, num_nodes))

    # 遍历每个节点及其路径

    # 计算每个节点之间的 SINR（不包括最后一个节点）
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst:
                continue  # 跳过自身

            # 获取源节点和目标节点的坐标
            src_x, src_y = routes_dict[src][1]
            dst_x, dst_y = routes_dict[dst][1]

            # 计算信号功率 S
            distance = np.sqrt((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2)
            G = pathloss_log(distance)
            S = G * transmission_power

            # 计算噪声功率 N
            N = noise

            # 计算干扰功率 I（只考虑路径中的第一条链路的干扰节点）
            I = 0  # 初始化干扰功率
            for other_node, (other_path, other_coordinates, _) in routes_dict.items():
                if other_node == src:
                    continue  # 跳过当前节点

                # 检查其他路径是否在同一时隙干扰当前链路
                if len(other_path) > 1 and other_path[1] == dst:
                    # 其他路径在同一时隙向同一目标发送信号
                    other_src = other_path[0]
                    other_src_x, other_src_y = routes_dict[other_src][1]
                    other_distance = np.sqrt((other_src_x - dst_x) ** 2 + (other_src_y - dst_y) ** 2)
                    other_G = pathloss_log(other_distance)
                    I += other_G * transmission_power

            # 计算 SINR
            SINR = S / (I + N)
            sinr_matrix[src, dst] = SINR

    return sinr_matrix

def calculate_path_sinr_rates_and_interference_level(routes_dict, pathloss_log, transmission_power, noise, B):
    path_sinr_rates_interference = {}

    # 遍历每个节点及其路径
    for node, (path, coordinates, pack_len) in routes_dict.items():
        sinr_list = []  # 存储当前路径中每一段链路的 SINR
        rates_list = []  # 存储当前路径中每一段链路的速率
        interference_list = []  # 存储当前路径中每一段链路的干扰
        # 遍历路径中的每一段链路
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]

            # 获取源节点和目标节点的坐标
            src_x, src_y = coordinates if src == node else (routes_dict[src][1][0], routes_dict[src][1][1])
            dst_x, dst_y = coordinates if dst == node else (routes_dict[dst][1][0], routes_dict[dst][1][1])

            # 计算信号功率 S
            distance = np.sqrt((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2)
            G = pathloss_log(distance)
            S = G * transmission_power

            # 计算噪声功率 N
            N = noise

            # 计算干扰功率 I
            I = 0  # 初始化干扰功率

            for other_node, (other_path, other_coordinates, _) in routes_dict.items():
                if other_node == node:
                    continue  # 跳过当前节点

                # 检查其他路径是否在同一时隙干扰当前链路
                # if len(other_path) > i + 1 and other_path[i + 1] == dst:
                if len(other_path) > i + 1:
                    # 其他路径在同一时隙向同一目标发送信号
                    other_src = other_path[i]
                    other_src_x, other_src_y = routes_dict[other_src][1]
                    other_distance = np.sqrt((other_src_x - dst_x) ** 2 + (other_src_y - dst_y) ** 2)
                    other_G = pathloss_log(other_distance)
                    I += other_G * transmission_power

            # 计算 SINR
            SINR = S / (I + N)
            sinr_list.append(SINR)

            # 计算当前链路的传输速率
            rate = B * np.log2(1 + SINR)
            rates_list.append(rate)

            #储存干扰
            interference_list.append(I)
        # 计算路径的总传输速率
        path_rate = np.average(rates_list) if rates_list else 0
        if path_rate == 0:
            path_transmit_time = 0
        else:
            path_transmit_time = pack_len / path_rate
        path_sinr_rates_interference[node] = {'SINRs': sinr_list, 'Rates': rates_list, 'PathRate': path_rate, 'Time': path_transmit_time,'Interfernece levels': interference_list}
    return path_sinr_rates_interference

def calculate_interference_level_old(routes_dict, pathloss_log, transmission_power, noise, B):
    interference_level = {}

    # 遍历每个节点及其路径
    for node, (path, coordinates, pack_len) in routes_dict.items():
        sinr_list = []  # 存储当前路径中每一段链路的 SINR
        rates_list = []  # 存储当前路径中每一段链路的速率
        interference_list = []  # 存储当前路径中每一段链路的干扰
        # 遍历路径中的每一段链路
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]

            # 获取源节点和目标节点的坐标
            src_x, src_y = coordinates if src == node else (routes_dict[src][1][0], routes_dict[src][1][1])
            dst_x, dst_y = coordinates if dst == node else (routes_dict[dst][1][0], routes_dict[dst][1][1])

            # 计算信号功率 S
            distance = np.sqrt((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2)
            G = pathloss_log(distance)
            S = G * transmission_power

            # 计算噪声功率 N
            N = noise

            # 计算干扰功率 I
            I = 0  # 初始化干扰功率

            for other_node, (other_path, other_coordinates, _) in routes_dict.items():
                if other_node == node:
                    continue  # 跳过当前节点

                # 检查其他路径是否在同一时隙干扰当前链路
                # if len(other_path) > i + 1 and other_path[i + 1] == dst:
                if len(other_path) > i + 1:
                    # 其他路径在同一时隙向同一目标发送信号
                    other_src = other_path[i]
                    other_src_x, other_src_y = routes_dict[other_src][1]
                    other_distance = np.sqrt((other_src_x - dst_x) ** 2 + (other_src_y - dst_y) ** 2)
                    other_G = pathloss_log(other_distance)
                    I += other_G * transmission_power
            # 储存干扰
            interference_list.append(I)
        interference_level[node] = {'Interfernece levels': interference_list}
    return interference_level

def Qlearning_NLPS_routing(S, server_index, Layer,global_index, SNRMat, N, max_distance, alpha=0.9, gamma=0.5, epsion=1e-3, ):
    #print(SNRMat)
    maxL = int(Layer[S])
    M = maxL - 1
    server_number = N - 1
    # 如果节点在最高层（Layer 1），直接连接到服务器
    if maxL == 1:
        # print(f"Node {S} is in the highest layer (1). Directly connecting to server.")
        Route = [S, server_index]
        return [], Route, np.zeros((N, N)),[global_index[S], server_index]

    # 根据分层情况存储节点
    L = []
    for m in range(1, M + 1):
        L.append(np.where(Layer == maxL - m)[0])
    L = [np.array([S])] + L + [np.array([server_number])]
    # 初始化 Q 和 R 表
    Q = np.zeros((N, N))
    R = np.zeros((N, N))
    #R=SNRMat

    for m in range(M + 1):
        R[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)] = SNRMat[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)]
        # print(f"R table after layer {m}: \n{R}")

        if m == M:
            Q[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)] = R[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)]
            # print(f"Q table initialized for the final layer {m}: \n{Q}")

    # 训练部分
    n = 1
    Re = []
    MRe = []
    s = [S]
    Route = []

    while True:
        QPre = Q.copy()
        for m in range(M):
            # print(f"Current Node: {s[m]}, Layer: {m + 1}")
            # 仅选择下一层的节点
            potential_nodes = []
            if m + 1 <= M:
                for node in L[m + 1]:
                    distance = np.linalg.norm(np.array([SNRMat[s[m], node]]))
                    # print(f"Checking Node {node} from Node {s[m]}, Distance: {distance}")
                    if distance <= max_distance:
                        potential_nodes.append(node)

            # 添加服务器节点到潜在节点列表中，确保路径终点是服务器
            if m == M - 1:
                potential_nodes = [server_number]
            # print(f"Adding server to potential nodes for final layer.")

            # print(f"Potential nodes for next step: {potential_nodes}")

            # 随机选择下一个节点
            if potential_nodes:
                am = np.random.choice(potential_nodes)
                # print(f"Selected next node: {am}")
            else:
                # print(f"No available nodes for Node {s[m]}, ending path selection.")
                break  # 如果没有可用节点，结束当前路径选择

            # 继续更新 Q 表
            if am == server_index:
                Qm1 = 0
                idx_am1 = -1
            else:
                Qm1, idx_am1 = Q[am, L[m + 2]].max(), Q[am, L[m + 2]].argmax()
            rm1 = R[s[m], am]

            if R[s[m], am] > rm1:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (rm1 + gamma * Qm1)
            else:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (R[s[m], am] + gamma * Qm1)

            # print(f"Updated Q[{s[m]}, {am}] = {Q[s[m], am]}")
            s.append(am)

        RoutePre = Route
        Route = [S]
        for m in range(M):
            idx_a = Q[Route[m], L[m + 1]].argmax()
            Route.append(L[m + 1][idx_a])
            # print(f"Route step {m+1}: Current Route: {Route}")

        SNRTemp = [SNRMat[Route[m - 1], Route[m]] for m in range(1, len(Route))]
        # print(f"SNR values along the route: {SNRTemp}")

        Re.append(1 / (M + 1) * np.log(sum(SNRTemp) + 1))
        # print(f"Re[{n}] = {Re[-1]}")

        if n > 1 and Re[-1] < Re[-2]:
            Re[-1] = Re[-2]

        MRe.append(np.mean(Re))
        # print(f"Mean Reward after iteration {n}: {MRe[-1]}")

        if n > N * 10 and abs(MRe[-1] - MRe[-10]) < epsion:
            break
        else:
            n += 1

    Route = [S]
    global_route=[global_index[S]]
    for m in range(M):
        idx_a = Q[Route[m], L[m + 1]].argmax()
        Route.append(L[m + 1][idx_a])
        global_route.append(global_index[L[m + 1][idx_a]])
    Route.append(server_index)
    global_route.append(server_index)
    return MRe, Route, Q, global_route

def Qlearning_NLPS2_routing(S, server_index, Layer, SNRMat, N, max_distance, alpha=0.9, gamma=0.5, epsion=1e-3, ):
    maxL = int(Layer[S])
    M = maxL - 1

    # 如果节点在最高层（Layer 1），直接连接到服务器
    if maxL == 1:
        # print(f"Node {S} is in the highest layer (1). Directly connecting to server.")
        Route = [S, server_index]
        return [], Route, np.zeros((N, N))

    # 根据分层情况存储节点
    L = []
    for m in range(1, M + 1):
        L.append(np.where(Layer == maxL - m)[0])
    L = [np.array([S])] + L + [np.array([server_index])]
    # 初始化 Q 和 R 表
    Q = np.zeros((N, N))
    R = np.zeros((N, N))
    for m in range(M + 1):
        R[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)] = SNRMat[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)]
        # print(f"R table after layer {m}: \n{R}")

        if m == M:
            Q[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)] = R[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)]
            # print(f"Q table initialized for the final layer {m}: \n{Q}")

    # 训练部分
    n = 1
    Re = []
    MRe = []
    s = [S]
    Route = []

    while True:
        QPre = Q.copy()
        for m in range(M):
            # print(f"Current Node: {s[m]}, Layer: {m + 1}")
            # 仅选择下一层的节点
            potential_nodes = []
            if m + 1 <= M:
                for node in L[m + 1]:
                    distance = np.linalg.norm(np.array([SNRMat[s[m], node]]))
                    # print(f"Checking Node {node} from Node {s[m]}, Distance: {distance}")
                    if distance <= max_distance:
                        potential_nodes.append(node)

            # 添加服务器节点到潜在节点列表中，确保路径终点是服务器
            if m == M - 1:
                potential_nodes = [server_index]
            # print(f"Adding server to potential nodes for final layer.")

            # print(f"Potential nodes for next step: {potential_nodes}")

            # 随机选择下一个节点
            if potential_nodes:
                am = np.random.choice(potential_nodes)
                # print(f"Selected next node: {am}")
            else:
                # print(f"No available nodes for Node {s[m]}, ending path selection.")
                break  # 如果没有可用节点，结束当前路径选择

            # 继续更新 Q 表
            if am == server_index:
                Qm1 = 0
                idx_am1 = -1
            else:
                Qm1, idx_am1 = Q[am, L[m + 2]].max(), Q[am, L[m + 2]].argmax()
            rm1 = R[s[m], am]

            if R[s[m], am] > rm1:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (rm1 + gamma * Qm1)
            else:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (R[s[m], am] + gamma * Qm1)

            # print(f"Updated Q[{s[m]}, {am}] = {Q[s[m], am]}")
            s.append(am)

        RoutePre = Route
        Route = [S]
        for m in range(M):
            idx_a = Q[Route[m], L[m + 1]].argmax()
            Route.append(L[m + 1][idx_a])
            # print(f"Route step {m+1}: Current Route: {Route}")

        SNRTemp = [SNRMat[Route[m - 1], Route[m]] for m in range(1, len(Route))]
        # print(f"SNR values along the route: {SNRTemp}")

        Re.append(1 / (M + 1) * np.log(sum(SNRTemp) + 1))
        # print(f"Re[{n}] = {Re[-1]}")

        if n > 1 and Re[-1] < Re[-2]:
            Re[-1] = Re[-2]

        MRe.append(np.mean(Re))
        # print(f"Mean Reward after iteration {n}: {MRe[-1]}")

        if n > N * 10 and abs(MRe[-1] - MRe[-10]) < epsion:
            break
        else:
            n += 1

    Route = [S]
    for m in range(M):
        idx_a = Q[Route[m], L[m + 1]].argmax()
        Route.append(L[m + 1][idx_a])
    Route.append(server_index)

    # print(f"Final route for Node {S} to Server: {Route}")
    return MRe, Route, Q

def Qlearning_INLPS_routing(S, server_index, Layer,global_index, SNRMat, Q, N, max_distance, alpha=0.9, gamma=0.5, epsion=1e-3, ):
    maxL = int(Layer[S])
    M = maxL - 1
    server_number = N - 1
    # 如果节点在最高层（Layer 1），直接连接到服务器
    if maxL == 1:
        # print(f"Node {S} is in the highest layer (1). Directly connecting to server.")
        Route = [S, server_index]
        return [], Route, np.zeros((N, N)),[global_index[S], server_index]

    # 根据分层情况存储节点
    L = []
    for m in range(1, M + 1):
        L.append(np.where(Layer == maxL - m)[0])
    L = [np.array([S])] + L + [np.array([server_number])]
    # 初始化 Q 和 R 表
    #Q = np.zeros((N, N))
    R = np.zeros((N, N))
    for m in range(M + 1):
        R[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)] = SNRMat[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)]
        # print(f"R table after layer {m}: \n{R}")

        if m == M:
            Q[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)] = R[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)]
            # print(f"Q table initialized for the final layer {m}: \n{Q}")

    # 训练部分
    n = 1
    Re = []
    MRe = []
    s = [S]
    Route = []

    while True:
        QPre = Q.copy()
        for m in range(M):
            # print(f"Current Node: {s[m]}, Layer: {m + 1}")
            # 仅选择下一层的节点
            potential_nodes = []
            if m + 1 <= M:
                for node in L[m + 1]:
                    distance = np.linalg.norm(np.array([SNRMat[s[m], node]]))
                    # print(f"Checking Node {node} from Node {s[m]}, Distance: {distance}")
                    if distance <= max_distance:
                        potential_nodes.append(node)

            # 添加服务器节点到潜在节点列表中，确保路径终点是服务器
            if m == M - 1:
                potential_nodes = [server_number]
            # print(f"Adding server to potential nodes for final layer.")

            # print(f"Potential nodes for next step: {potential_nodes}")

            # 随机选择下一个节点
            if potential_nodes:
                am = np.random.choice(potential_nodes)
                # print(f"Selected next node: {am}")
            else:
                # print(f"No available nodes for Node {s[m]}, ending path selection.")
                break  # 如果没有可用节点，结束当前路径选择

            # 继续更新 Q 表
            if am == server_index:
                Qm1 = 0
                idx_am1 = -1
            else:
                Qm1, idx_am1 = Q[am, L[m + 2]].max(), Q[am, L[m + 2]].argmax()
            rm1 = R[s[m], am]

            if R[s[m], am] > rm1:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (rm1 + gamma * Qm1)
            else:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (R[s[m], am] + gamma * Qm1)

            # print(f"Updated Q[{s[m]}, {am}] = {Q[s[m], am]}")
            s.append(am)

        RoutePre = Route
        Route = [S]
        for m in range(M):
            idx_a = Q[Route[m], L[m + 1]].argmax()
            Route.append(L[m + 1][idx_a])
            # print(f"Route step {m+1}: Current Route: {Route}")

        SNRTemp = [SNRMat[Route[m - 1], Route[m]] for m in range(1, len(Route))]
        # print(f"SNR values along the route: {SNRTemp}")

        Re.append(1 / (M + 1) * np.log(sum(SNRTemp) + 1))
        # print(f"Re[{n}] = {Re[-1]}")

        if n > 1 and Re[-1] < Re[-2]:
            Re[-1] = Re[-2]

        MRe.append(np.mean(Re))
        # print(f"Mean Reward after iteration {n}: {MRe[-1]}")

        if n > N * 10 and abs(MRe[-1] - MRe[-10]) < epsion:
            break
        else:
            n += 1

    Route = [S]
    global_route=[global_index[S]]
    for m in range(M):
        idx_a = Q[Route[m], L[m + 1]].argmax()
        Route.append(L[m + 1][idx_a])
        global_route.append(global_index[L[m + 1][idx_a]])
    Route.append(server_index)
    global_route.append(server_index)
    return MRe, Route, Q, global_route

def Qlearning_INLPS2_routing(S, D, Layer, SINRMat, Q, N, alpha=0.5, gamma=0.9, epsion=1e-3, ):
    maxL = Layer[S]
    M = maxL - 1

    if maxL == 1:
        Route = [S, D]
        return [], Route, Q

    L = []
    for m in range(1, M + 1):
        L.append(np.where(Layer == maxL - m)[0])
    L = [np.array([S])] + L + [np.array([D])]

    R = np.zeros((N, N))
    for m in range(M + 1):
        # print(f"Shape of L[m]: {L[m].shape}")
        # print(f"Shape of L[m + 1]: {L[m + 1].shape}")
        # print(f"Shape of SINRMat: {SINRMat.shape}")
        R[np.ix_(L[m], L[m + 1])] = SINRMat[np.ix_(L[m], L[m + 1])]

    n = 1
    Re = []
    MRe = []
    s = [S]
    Route = []

    while True:
        QPre = Q.copy()
        for m in range(M):
            idx_am = np.random.choice(len(L[m + 1]))
            am = L[m + 1][idx_am]

            if m + 2 <= M + 1:
                Qm1 = Q[am, L[m + 2]].max()
                am1 = L[m + 2][Q[am, L[m + 2]].argmax()]
            else:
                Qm1 = 0
                am1 = D
            rm1 = R[am, am1]

            if R[s[m], am] > rm1:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (rm1 + gamma * Qm1)
            else:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (R[s[m], am] + gamma * Qm1)

            s.append(am)

        RoutePre = Route
        Route = [S]
        for m in range(M):
            idx_a = Q[Route[m], L[m + 1]].argmax()
            Route.append(L[m + 1][idx_a])

        SINRTemp = [SINRMat[Route[m - 1], Route[m]] for m in range(1, len(Route))]

        Re.append(1 / (M + 1) * np.log(sum(SINRTemp) + 1))

        if n > 1 and Re[-1] < Re[-2]:
            Re[-1] = Re[-2]

        MRe.append(np.mean(Re))

        if n > N * 10 and abs(MRe[-1] - MRe[-10]) < epsion:
            break
        else:
            n += 1

    Route = [S]
    for m in range(M):
        idx_a = Q[Route[m], L[m + 1]].argmax()
        Route.append(L[m + 1][idx_a])
    Route.append(D)

    return MRe, Route, Q

def pathloss_log(distance):
    PL0 = 20.0 * np.log10(d0)
    PL1 = PL0 + (10.0 * attenuation_constant * np.log10(distance / d0)) - wall_attenuation + shadowing
    G = 1.0 / (10.0 ** (PL1 / 10.0))
    return G

def adaptively_label_dataset(file_path):
    # 加载数据集，没有表头
    data = pd.read_csv(
        file_path,
        header=None,
        sep=',',
        na_values=[''],
        on_bad_lines='warn'  # or 'skip' to silently skip bad lines
    )

    # 确定列数
    num_columns = data.shape[1]

    # 根据列数确定服务器的数量
    num_servers = (num_columns - 4) // 2

    # 动态生成列名
    column_names = ['Node_ID', 'Packet_Length', 'Node_X', 'Node_Y']

    for i in range(num_servers):
        column_names.append(f'Server{i}_X')
        column_names.append(f'Server{i}_Y')

    column_names.append('Assigned_Server')

    # 为数据分配列名
    data.columns = column_names

    # 按 'Node_ID' 和 'Assigned_Server' 组合去重
    data = data.drop_duplicates(subset=['Node_ID', 'Assigned_Server'])

    return data

def plot_nodes_and_servers_corrected(data):
    plt.figure(figsize=(10, 8))

    # Define a colormap for different servers
    colormap = plt.cm.get_cmap('tab10', len(data['Assigned_Server'].unique()))

    # Plot nodes and their assigned servers using the same color
    for server_id in data['Assigned_Server'].unique():
        # Plot nodes assigned to the server
        assigned_nodes = data[data['Assigned_Server'] == server_id]
        plt.scatter(assigned_nodes['Node_X'], assigned_nodes['Node_Y'],
                    s=100, c=[colormap(server_id)], label=f'Nodes (Server {server_id})', marker='o')

        # Plot the server with a larger icon
        server_x = data[f'Server{int(server_id)}_X'].iloc[0]
        server_y = data[f'Server{int(server_id)}_Y'].iloc[0]
        plt.scatter(server_x, server_y, s=400, c=[colormap(server_id)], label=f'Server {server_id}', marker='^')

    # Labeling the plot
    plt.title('Node and Server Positions with Larger Server Icons')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='best')
    plt.grid(True)

    # Show plot
    plt.show()

def filter_nodes_by_distance(labeled_data, max_distance=510):
    filtered_rows = []
    for index, row in labeled_data.iterrows():
        server_id = int(row['Assigned_Server'])
        server_x = row[f'Server{server_id}_X']
        server_y = row[f'Server{server_id}_Y']
        node_x = row['Node_X']
        node_y = row['Node_Y']

        # 计算节点到服务器的距离
        distance = np.sqrt((node_x - server_x) ** 2 + (node_y - server_y) ** 2)

        # 如果距离小于等于510，则保留该节点
        if distance <= max_distance:
            filtered_rows.append(row)

    # 使用 pd.concat 而不是 pd.DataFrame.append
    filtered_data = pd.concat(filtered_rows, axis=1).T
    return filtered_data

def extract_server_node_data(labeled_data):
    # Create an empty dictionary to store server node data
    server_node_data = {}

    # Get the unique server IDs
    unique_servers = labeled_data['Assigned_Server'].unique()

    # Iterate over each server ID
    for server_id in unique_servers:
        # Extract the data corresponding to the current server
        server_data = labeled_data[labeled_data['Assigned_Server'] == server_id]
        # Store the data in the dictionary with server ID as the key
        server_node_data[server_id] = server_data

    return server_node_data

def extract_server_coordinates(labeled_data):
    # Create an empty dictionary to store server coordinates
    server_coordinates = {}

    # Get the unique server IDs
    unique_servers = labeled_data['Assigned_Server'].unique()

    # Iterate over each server ID
    for server_id in unique_servers:
        # Extract the X and Y coordinates for the server
        server_x = labeled_data[f'{server_id}_X'].iloc[0]
        server_y = labeled_data[f'{server_id}_Y'].iloc[0]
        # Store the coordinates as a tuple in the dictionary
        server_coordinates[server_id] = (server_x, server_y)

    return server_coordinates

def divide_nodes_by_fixed_distance1(server_node_data, server_coordinates, server_id, max_radius):
    # Get the server coordinates
    server_x, server_y = server_coordinates[server_id]

    # Extract the nodes assigned to the server and make a copy
    nodes_data = server_node_data[server_id].copy()

    # Calculate the distance of each node from the server
    nodes_data['Distance'] = np.sqrt((nodes_data['Node_X'] - server_x) ** 2 +
                                     (nodes_data['Node_Y'] - server_y) ** 2)

    # Define fixed thresholds based on the max_radius parameter
    thresholds = [max_radius, 2 * max_radius, 3 * max_radius]

    # Assign distance category based on fixed thresholds
    conditions = [
        (nodes_data['Distance'] <= thresholds[0]),
        (nodes_data['Distance'] > thresholds[0]) & (nodes_data['Distance'] <= thresholds[1]),
        (nodes_data['Distance'] > thresholds[1]) & (nodes_data['Distance'] <= thresholds[2])
    ]

    choices = ['Zone_1', 'Zone_2', 'Zone_3']

    nodes_data['Distance_Category'] = np.select(conditions, choices, default=np.nan)

    # Remove nodes with a distance greater than 3 * max_radius
    nodes_data = nodes_data.dropna(subset=['Distance_Category'])

    return nodes_data[['Node_ID', "Packet_Length", 'Node_X', 'Node_Y', 'Distance', 'Distance_Category']]

def divide_nodes_by_fixed_distance(server_node_data, server_coordinates, server_id, max_radius):
    # Get the server coordinates
    server_x, server_y = server_coordinates[server_id]

    # Extract the nodes assigned to the server and make a copy
    nodes_data = server_node_data[server_id].copy()

    # Calculate the distance of each node from the server
    nodes_data['Distance'] = np.sqrt((nodes_data['Node_X'] - server_x) ** 2 +
                                     (nodes_data['Node_Y'] - server_y) ** 2)

    # Define fixed thresholds based on the max_radius parameter for 4 zones
    thresholds = [max_radius, 2 * max_radius, 3 * max_radius, 4 * max_radius, 5 * max_radius]

    # Assign distance category based on fixed thresholds
    conditions = [
        (nodes_data['Distance'] <= thresholds[0]),
        (nodes_data['Distance'] > thresholds[0]) & (nodes_data['Distance'] <= thresholds[1]),
        (nodes_data['Distance'] > thresholds[1]) & (nodes_data['Distance'] <= thresholds[2]),
        (nodes_data['Distance'] > thresholds[2]) & (nodes_data['Distance'] <= thresholds[3]),
        (nodes_data['Distance'] > thresholds[3]) & (nodes_data['Distance'] <= thresholds[4])
    ]

    choices = ['Zone_1', 'Zone_2', 'Zone_3', 'Zone_4', 'Zone_5']

    nodes_data['Distance_Category'] = np.select(conditions, choices, default=np.nan)


    return nodes_data[['Node_ID', "Packet_Length", 'Node_X', 'Node_Y', 'Distance', 'Distance_Category']]
def delete_nonedata(nodes_data):
    zones = ['Zone_1', 'Zone_2', 'Zone_3', 'Zone_4', 'Zone_5']
    valid_zones = set()
    for zone in zones:
        if (nodes_data['Distance_Category'] == zone).any():
            valid_zones.add(zone)
        else:
            # If a zone has no nodes, all further zones are invalid
            break

    # Keep only nodes in valid zones
    nodes_data = nodes_data[nodes_data['Distance_Category'].isin(valid_zones)]
    return nodes_data[['Node_ID', "Packet_Length", 'Node_X', 'Node_Y', 'Distance', 'Distance_Category']]
def plot_server_zones(nodes_data, server_coordinates, server_id, max_radius):
    server_x, server_y = server_coordinates[server_id]

    plt.figure(figsize=(10, 8))

    colors = {'Zone_1': 'red', 'Zone_2': 'orange', 'Zone_3': 'yellow', 'Zone_4': 'green', 'Zone_5': 'blue'}

    for zone, color in colors.items():
        zone_data = nodes_data[nodes_data['Distance_Category'] == zone]
        plt.scatter(zone_data['Node_X'], zone_data['Node_Y'], color=color, label=zone)

    plt.scatter(server_x, server_y, color='black', s=200, marker='x', label='Server Location')

    # Draw circles with the given max_radius
    for i in range(1, 6):
        circle = plt.Circle((server_x, server_y), i * max_radius, color='gray', fill=False, linestyle='--',
                            label=f'Radius {i * max_radius}')
        plt.gca().add_patch(circle)

    # Add labels and legend
    plt.title(f'Node Distribution and Zones for Server {server_id}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)

    # Ensure the circles and scatter plot are visible
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(server_x - 6 * max_radius, server_x + 5 * max_radius)
    plt.ylim(server_y - 6 * max_radius, server_y + 5 * max_radius)

    # Show plot
    plt.show()

def calculate_snr_matrix(nodes_data):
    # 获取节点的数量
    num_nodes = len(nodes_data)

    # 初始化 SNR 矩阵
    snr_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 计算节点 i 和节点 j 之间的距离
            distance = np.sqrt((nodes_data.iloc[i]['Node_X'] - nodes_data.iloc[j]['Node_X']) ** 2 +
                               (nodes_data.iloc[i]['Node_Y'] - nodes_data.iloc[j]['Node_Y']) ** 2)

            # 根据距离计算 SNR
            G = pathloss_log(distance)
            S = G * transmission_power
            N = noise
            snr = S / N

            # 对称地填充 SNR 矩阵
            snr_matrix[i, j] = snr
            snr_matrix[j, i] = snr

    return snr_matrix

def calculate_snr_to_server(nodes_data, server_coord):
    distances = np.sqrt((nodes_data['Node_X'] - server_coord[0]) ** 2 +
                        (nodes_data['Node_Y'] - server_coord[1]) ** 2)
    G = pathloss_log(distances)
    S = G * transmission_power
    N = noise
    snr_to_server = S / N
    return snr_to_server, distances
# 验证路径层级顺序
def verify_path_layers(routes_dict, nodes_data, server_index):
    for node, path in routes_dict.items():
        # 获取路径中各节点的层级
        layers = [nodes_data.iloc[node_id]['Layer'] if node_id != server_index else 0 for node_id in path]

        # 检查层级是否按递减顺序排列
        if layers != sorted(layers, reverse=True):
            print(f"Path for Node {node} does not follow a decreasing layer order: {layers}")
        else:
            print(f"Path for Node {node} is valid: {layers}")

def gertTimeSlotAndTimeSINR(RouteAll, PMat, maxRecNum, eta, B, N, L):
    # Initialize SINR matrix and TimeSlot list
    SINR = np.zeros((N, N))
    TimeSlot = []
    currentIdx = np.ones(len(RouteAll), dtype=int)

    while True:
        SendNode = np.zeros(N)  # Record how many times a node is a sender
        RecNode = np.zeros(N)  # Record how many times a node is a receiver
        Path = []

        for i in range(len(RouteAll)):
            if currentIdx[i] == len(RouteAll[i]):
                continue
            for j in range(currentIdx[i], len(RouteAll[i]) - 1):
                s = RouteAll[i][j]
                d = RouteAll[i][j + 1]
                if SendNode[s] < maxRecNum and RecNode[s] == 0 and SendNode[d] == 0 and RecNode[d] < maxRecNum:
                    currentIdx[i] += 1
                    SendNode[s] += 1
                    RecNode[d] += 1
                    Path.append([s, d])

        if Path:
            TimeSlot.append(np.array(Path))
        else:
            break

    # Calculate Transmission Time and update SINR matrix
    Time = []
    for slot in TimeSlot:
        rData = []
        P = np.array([PMat[s, d] for s, d in slot])

        RecNode = np.unique(slot[:, 1])
        for rec in RecNode:
            ind = np.where(slot[:, 1] == rec)[0]
            P1 = np.sum(P[ind])
            P2 = np.sum(P) - P1
            sinr = P1 / (P2 + eta * B)

            for k in ind:
                s, d = slot[k]
                sinrtemp = SINR[s, d]
                if sinr >= sinrtemp:
                    SINR[s, d] = sinr
                    SINR[d, s] = sinr

            rData.append(B * np.log2(1 + sinr))

        Time.append(max(L / np.array(rData)))

    return TimeSlot, Time, SINR

def Qlearning_INLPS_routing_old(S, server_index, Layer, SNRMat, N, max_distance, Q_NLPS, alpha=0.9, gamma=0.5, epsion=1e-3, ):
    maxL = int(Layer[S])
    M = maxL - 1

    # 如果节点在最高层（Layer 1），直接连接到服务器
    if maxL == 1:
        # print(f"Node {S} is in the highest layer (1). Directly connecting to server.")
        Route = [S, server_index]
        return [], Route, np.zeros((N, N))

    # 根据分层情况存储节点
    L = []
    for m in range(1, M + 1):
        L.append(np.where(Layer == maxL - m)[0])
    L = [np.array([S])] + L + [np.array([server_index])]
    # 初始化 Q 和 R 表
    Q = Q_NLPS
    R = np.zeros((N, N))
    for m in range(M + 1):
        R[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)] = SNRMat[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)]
        # print(f"R table after layer {m}: \n{R}")

        if m == M:
            Q[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)] = R[L[m].reshape(-1, 1), L[m + 1].reshape(1, -1)]
            # print(f"Q table initialized for the final layer {m}: \n{Q}")

    # 训练部分
    n = 1
    Re = []
    MRe = []
    s = [S]
    Route = []

    while True:
        QPre = Q.copy()
        for m in range(M):
            # print(f"Current Node: {s[m]}, Layer: {m + 1}")
            # 仅选择下一层的节点
            potential_nodes = []
            if m + 1 <= M:
                for node in L[m + 1]:
                    distance = np.linalg.norm(np.array([SNRMat[s[m], node]]))
                    # print(f"Checking Node {node} from Node {s[m]}, Distance: {distance}")
                    if distance <= max_distance:
                        potential_nodes.append(node)

            # 添加服务器节点到潜在节点列表中，确保路径终点是服务器
            if m == M - 1:
                potential_nodes = [server_index]
            # print(f"Adding server to potential nodes for final layer.")

            # print(f"Potential nodes for next step: {potential_nodes}")

            # 随机选择下一个节点
            if potential_nodes:
                am = np.random.choice(potential_nodes)
                # print(f"Selected next node: {am}")
            else:
                # print(f"No available nodes for Node {s[m]}, ending path selection.")
                break  # 如果没有可用节点，结束当前路径选择

            # 继续更新 Q 表
            if am == server_index:
                Qm1 = 0
                idx_am1 = -1
            else:
                Qm1, idx_am1 = Q[am, L[m + 2]].max(), Q[am, L[m + 2]].argmax()
            rm1 = R[s[m], am]

            if R[s[m], am] > rm1:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (rm1 + gamma * Qm1)
            else:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (R[s[m], am] + gamma * Qm1)

            # print(f"Updated Q[{s[m]}, {am}] = {Q[s[m], am]}")
            s.append(am)

        RoutePre = Route
        Route = [S]
        for m in range(M):
            idx_a = Q[Route[m], L[m + 1]].argmax()
            Route.append(L[m + 1][idx_a])
            # print(f"Route step {m+1}: Current Route: {Route}")

        SNRTemp = [SNRMat[Route[m - 1], Route[m]] for m in range(1, len(Route))]
        # print(f"SNR values along the route: {SNRTemp}")

        Re.append(1 / (M + 1) * np.log(sum(SNRTemp) + 1))
        # print(f"Re[{n}] = {Re[-1]}")

        if n > 1 and Re[-1] < Re[-2]:
            Re[-1] = Re[-2]

        MRe.append(np.mean(Re))
        # print(f"Mean Reward after iteration {n}: {MRe[-1]}")

        if n > N * 10 and abs(MRe[-1] - MRe[-10]) < epsion:
            break
        else:
            n += 1

    Route = [S]
    for m in range(M):
        idx_a = Q[Route[m], L[m + 1]].argmax()
        Route.append(L[m + 1][idx_a])
    Route.append(server_index)

    # print(f"Final route for Node {S} to Server: {Route}")
    return MRe, Route, Q

def DQNMIPS(S, D, Layer, SINRMat, Q, alpha=0.5, gamma=0.9, epsion=1e-3, N=100):
    maxL = Layer[S]
    M = maxL - 1

    if maxL == 1:
        # print(f"Node {S} is in the highest layer (1). Directly connecting to server.")
        Route = [S, D]
        return [], Route, Q

    # 根据分层情况存储节点
    L = []
    for m in range(1, M + 1):
        L.append(np.where(Layer == maxL - m)[0])
    L = [np.array([S])] + L + [np.array([D])]

    # 初始化 R 表
    R = np.zeros((N, N))
    for m in range(M + 1):
        R[np.ix_(L[m], L[m + 1])] = SINRMat[np.ix_(L[m], L[m + 1])]
        # print(f"R table after layer {m}: \n{R}")

    # 训练部分
    n = 1
    Re = []
    MRe = []
    s = [S]
    Route = []

    while True:
        QPre = Q.copy()
        for m in range(M):
            # print(f"Current Node: {s[m]}, Layer: {m + 1}")
            # 随机选择下一层的节点
            idx_am = np.random.choice(len(L[m + 1]))
            am = L[m + 1][idx_am]
            # print(f"Selected next node: {am}")

            # 获取 maxQ(m+1) 和 a(m+1)
            if m + 2 <= M + 1:
                Qm1 = Q[am, L[m + 2]].max()
                am1 = L[m + 2][Q[am, L[m + 2]].argmax()]
            else:
                Qm1 = 0
                am1 = D
            rm1 = R[am, am1]

            # 更新 Q 表
            if R[s[m], am] > rm1:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (rm1 + gamma * Qm1)
            else:
                Q[s[m], am] = (1 - alpha) * Q[s[m], am] + alpha * (R[s[m], am] + gamma * Qm1)

            # print(f"Updated Q[{s[m]}, {am}] = {Q[s[m], am]}")
            s.append(am)

        RoutePre = Route
        Route = [S]
        for m in range(M):
            idx_a = Q[Route[m], L[m + 1]].argmax()
            Route.append(L[m + 1][idx_a])
            # print(f"Route step {m+1}: Current Route: {Route}")

        SINRTemp = [SINRMat[Route[m - 1], Route[m]] for m in range(1, len(Route))]
        # print(f"SINR values along the route: {SINRTemp}")

        Re.append(1 / (M + 1) * np.log(sum(SINRTemp) + 1))
        # print(f"Re[{n}] = {Re[-1]}")

        if n > 1 and Re[-1] < Re[-2]:
            Re[-1] = Re[-2]

        MRe.append(np.mean(Re))
        # print(f"Mean Reward after iteration {n}: {MRe[-1]}")

        if n > N * 10 and abs(MRe[-1] - MRe[-10]) < epsion:
            break
        else:
            n += 1

    Route = [S]
    for m in range(M):
        idx_a = Q[Route[m], L[m + 1]].argmax()
        Route.append(L[m + 1][idx_a])
    Route.append(D)

    # print(f"Final route for Node {S} to Server: {Route}")
    return MRe, Route, Q