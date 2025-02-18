
# 过滤数据
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

B = 40 * 10 ** 6  # Hz  带宽
d0 = 1  # decorrelation distance m
attenuation_constant = 3.5
shadowing = 4 # dB
wall_attenuation = 0 # dB
noise_level = 3.9810717055 * 10 ** -18 # mW/Hz, -174 dBm/Hz
transmission_power = 200  # mW
noise = B * noise_level

def Qlearning_NLPS_routing(S, server_index, Layer, SNRMat, N, max_distance, alpha=0.9, gamma=0.5, epsion=1e-3, ):
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

def Qlearning_INLPS_routing(S, server_index, Layer, SNRMat, N, max_distance, Q_NLPS, alpha=0.9, gamma=0.5, epsion=1e-3, ):
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
        server_x = labeled_data[f'Server{int(server_id)}_X'].iloc[0]
        server_y = labeled_data[f'Server{int(server_id)}_Y'].iloc[0]
        # Store the coordinates as a tuple in the dictionary
        server_coordinates[server_id] = (server_x, server_y)

    return server_coordinates
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
    for node, (path, coordinates,pack_len) in routes_dict.items():
        sinr_list = []  # 存储当前路径中每一段链路的 SINR
        rates_list = []  # 存储当前路径中每一段链路的速率
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

            for other_node,(other_path,other_coordinates,_) in routes_dict.items():
                if other_node == node:
                    continue  # 跳过当前节点

                # 检查其他路径是否在同一时隙干扰当前链路
                if len(other_path) > i + 1 and other_path[i + 1] == dst:
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

        # 计算路径的总传输速率
        path_rate = np.average(rates_list) if rates_list else 0
        if path_rate == 0:
            path_transmit_time = 0
        else:
            path_transmit_time = pack_len / path_rate
        path_sinr_rates[node] = {'SINRs': sinr_list, 'Rates': rates_list, 'PathRate': path_rate, 'Time': path_transmit_time}

    return path_sinr_rates
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
            for other_node, (other_path, other_coordinates,_) in routes_dict.items():
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

def divide_nodes_by_fixed_distance(server_node_data, server_coordinates, server_id, max_radius=170):
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

def plot_server_zones(nodes_data, server_coordinates, server_id, max_radius=170):
    server_x, server_y = server_coordinates[server_id]

    plt.figure(figsize=(10, 8))

    colors = {'Zone_1': 'blue', 'Zone_2': 'green', 'Zone_3': 'red'}

    for zone, color in colors.items():
        zone_data = nodes_data[nodes_data['Distance_Category'] == zone]
        plt.scatter(zone_data['Node_X'], zone_data['Node_Y'], color=color, label=zone)

    plt.scatter(server_x, server_y, color='black', s=200, marker='x', label='Server Location')

    # Draw circles with the given max_radius
    for i in range(1, 4):
        circle = plt.Circle((server_x, server_y), i * max_radius, color='gray', fill=False, linestyle='--',
                            label=f'Radius {i * max_radius}')
        plt.gca().add_patch(circle)

    # Add labels and legend
    plt.title(f'Node Distribution and Zones for Server {server_id}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)

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
# 使用前面更新的Q表进行优化后的DQNMIPS计算
def INLPS(S, D, Layer, SINRMat, Q, alpha=0.5, gamma=0.9, epsion=1e-3, N=100):
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