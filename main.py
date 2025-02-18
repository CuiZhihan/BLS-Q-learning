import pandas as pd
import matplotlib.pyplot as plt
from utilts import *
from IPython.display import clear_output
import json
import pprint
import os
import pandas as pd
import numpy as np
import datetime
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行

# 超参数
max_radius = 30 #传输距离
CPNC = False #True or False
B = 40 * 10 ** 6  # Hz  带宽
d0 = 1  # decorrelation distance m
attenuation_constant = 3.5
shadowing = 4 # dB
wall_attenuation = 0 # dB
noise_level = 3.9810717055 * 10 ** -18 # mW/Hz, -174 dBm/Hz
transmission_power = 200  # mW
noise = B * noise_level

# 定义输入根目录
#root_directory = "./data/server"
root_directory = "./data/device"
# 遍历根目录下的所有子文件夹
for subfolder in sorted(os.listdir(root_directory)):
    input_directory = os.path.join(root_directory, subfolder)
    if not os.path.isdir(input_directory):  # 跳过非文件夹
        continue
    # 生成输出文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #output_file = f"results_summary_{subfolder}_200D_{timestamp}.csv"
    output_file = f"results_summary_10server_{subfolder}_{timestamp}.csv"

    # 定义结果存储
    results = []

# # 定义输入目录和输出文件名
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# input_directory = "./data/server/5server"  # 默认路径为程序所在目录
# #output_directory = "./data/server"  # 新的保存路径
# output_file = f"results_summary_5S_200D_{timestamp}.csv"
    for file_name in sorted(os.listdir(input_directory)):
        if file_name.startswith("task_info_with_decision_") and file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)
            #file_path = 'task_info_with_decision_10servers_300devices.csv'
            print(file_path)
            labeled_data = adaptively_label_dataset(file_path)
            labeled_data["Node_ID"]=labeled_data["Node_ID"].astype("int")
            labeled_data["Packet_Length"]=labeled_data["Packet_Length"].astype("int")
            labeled_data["Assigned_Server"]=labeled_data["Assigned_Server"].astype("int")

            #画图，可以注释掉
            #_nodes_and_servers_corrected(labeled_data)

            # 处理标记数据集
            labeled_data = filter_nodes_by_distance(labeled_data)
            labeled_data["Node_ID"] = labeled_data["Node_ID"].apply(lambda x: int(x))
            labeled_data["Assigned_Server"] = labeled_data["Assigned_Server"].apply(lambda x: f"Server{int(x)}")
            # 应用函数到标记数据集
            server_node_data = extract_server_node_data(labeled_data)
            # print(server_node_data)

            # 显示每个服务器的节点数量摘要
            server_node_data_summary = {server_id: data.shape[0] for server_id, data in server_node_data.items()}
            # 应用函数到标记数据集
            server_coordinates = extract_server_coordinates(labeled_data)
            # 一个字典取每个server对应的坐标 dict[服务器id]：该服务器对应的坐标
            # print("服务器字典：", server_node_data_summary)
            # print("server坐标：", server_coordinates)
            selected_columns = ['Node_ID', 'Packet_Length', 'Node_X', 'Node_Y', 'Assigned_Server']
            # 选择所需的列
            global_table = labeled_data[selected_columns]
            #print(global_table.head())


            #这里开始遍历：
            all_map_nodes_transmission_dict={}
            server_information_1=[]
            server_information_2=[]
            new_ids = [-1 * int(key[-1])-1 for key in server_coordinates.keys()]

            # Create the new table
            data = {
            'Serer_ID': new_ids,
            'Server_X': [coords[0] for coords in server_coordinates.values()],
            'Server_Y': [coords[1] for coords in server_coordinates.values()]
            }
            server_table = pd.DataFrame(data)
            global_table_NLPS = global_table.copy()
            global_table_INLPS = global_table.copy()

            for current_sever_id in server_coordinates.keys():
                # 过滤节点数据
                print(f"###################  {current_sever_id}   ####################")
                # if int(current_sever_id[-1]) % 2 == 1:
                #     clear_output(wait=True)
                server_nodes_data_filtered1 = divide_nodes_by_fixed_distance(server_node_data, server_coordinates, server_id=current_sever_id, max_radius=max_radius)
                #_server_zones(server_nodes_data_filtered1, server_coordinates, server_id=current_sever_id, max_radius=max_radius)


                server_nodes_data_filtered=delete_nonedata(server_nodes_data_filtered1)

                server_nodes_data_filtered=server_nodes_data_filtered.replace('nan', np.nan).dropna()
                # 绘制区域
                #plot_server_zones(server_nodes_data_filtered, server_coordinates, server_id=current_sever_id, max_radius=max_radius)

                # 使用之前的数据集 server_nodes_data_filtered
                nodes_data = server_nodes_data_filtered.copy()
                server_id = int(current_sever_id[-1])*-1 -1
                #print(server_id)
                # 创建 Layer 信息
                layer_mapping = {'Zone_1': 1, 'Zone_2': 2, 'Zone_3': 3, 'Zone_4': 4, 'Zone_5': 5}
                nodes_data['Layer'] = nodes_data['Distance_Category'].map(layer_mapping)

                # 获取服务器的坐标
                server_coord = server_coordinates[current_sever_id]  # 假设 server_id 是服务器的唯一标识符
                D = len(nodes_data)  # 服务器节点索引放在最后一个
                server_index = len(nodes_data)  # 服务器作为最后一个节点

                # 计算每个节点到服务器的距离和 SNR
                nodes_data['SNR_to_Server'], nodes_data['Distance_to_Server'] = calculate_snr_to_server(nodes_data, server_coord)

                # 计算 SNR 矩阵
                snr_matrix = calculate_snr_matrix(nodes_data)

                # 更新节点数量，包括服务器节点
                N = len(nodes_data) + 1

                # 构建扩展后的 SNR 矩阵
                extended_snr_matrix = np.zeros((N, N))
                extended_snr_matrix[:-1, :-1] = snr_matrix  # 原始 SNR 矩阵
                extended_snr_matrix[:-1, -1] = nodes_data['SNR_to_Server'].values  # 最后一列是到服务器的 SNR
                extended_snr_matrix[-1, :-1] = nodes_data['SNR_to_Server'].values  # 最后一行是到服务器的 SNR
                # 存储每个节点路径的字典
                routes_dict = {}
                ##########################################    NLPS    ##############################
                # 遍历所有节点，找到到服务器的最优路径
                # print(nodes_data)
                for S in range(len(nodes_data)):
                    # 定义 Layer 和 SNR 矩阵，包含服务器节点的信息
                    Layer = nodes_data['Layer'].values
                    Layer = np.append(Layer, 0)  # 服务器在层级 0（可以认为它是层级最高的）
                    # 计算路径
                    # 大表
                    global_index_NLPS=nodes_data['Node_ID'].values
                    _, _, _, global_route_NLPS = Qlearning_NLPS_routing(S, server_id, Layer,global_index_NLPS, extended_snr_matrix, N, max_radius)

                    global_table_NLPS.loc[global_table_NLPS['Node_ID'] == global_route_NLPS[0], 'route'] = str(global_route_NLPS)

                    # 计算路径
                    MRe, Route, Q = Qlearning_NLPS2_routing(S, server_index, Layer, extended_snr_matrix, N, max_radius)


                    # 储存路径
                    routes_dict[S] = [Route,nodes_data.iloc[S][['Node_X', 'Node_Y']].tolist(),int(nodes_data.iloc[S][['Packet_Length']])]
                routes_dict[S+1] = [[S+1], list(server_coordinates[current_sever_id]),0]

                #计算传输速率
                capacity_NLPS, time_NLPS, total_interference_NLPS, average_interference_NLPS, average_path_rate_NLPS = calculate_network_metrics(routes_dict, pathloss_log, transmission_power, noise, B)

                path_sinr_rates_NLPS2 = calculate_path_sinr_and_rates(routes_dict=routes_dict, pathloss_log=pathloss_log, transmission_power=transmission_power, noise=noise, B=B)

                # print(capacity_NLPS)
                # print(time_NLPS)
                # print(total_interference_NLPS)
                # print(average_interference_NLPS)
                # print(average_path_rate_NLPS)
                #计算SINR表
                sinr_mat = calculate_sinr_mat(routes_dict=routes_dict, pathloss_log=pathloss_log, transmission_power=transmission_power, noise=noise, B=B)

                #打印路径和速率
                # for node, (path, coordinates,_) in routes_dict.items():
                #     sinr_info = path_sinr_rates.get(node, {})
                #     sinrs = sinr_info.get('SINRs', [])
                #     rates = sinr_info.get('Rates', [])
                #     path_rate = sinr_info.get('PathRate', 0)
                #     path_transmit_time = sinr_info.get('Time', 0)
                #     #print(f"Node {node} to Server: Path = {path}, Path Rate = {path_rate / 1e6:.2f} Mbps, Path Time = {path_transmit_time} s.")
                #print("##################  NLPS  #################")

                # 计算 path_rate （排除为 0 的值）
                path_rates_NLPS2 = [info['PathRate'] for info in path_sinr_rates_NLPS2.values() if info['PathRate'] > 0]
                avg_path_rate_NLPS2 = np.average(path_rates_NLPS2) if path_rates_NLPS2 else 0
                #print(f"Average Path Rate: {avg_path_rate / 1e6:.2f} Mbps")
                sum_path_rate_NLPS2 = np.sum(path_rates_NLPS2) if path_rates_NLPS2 else 0
                #print(f"Sum Path Rate: {sum_path_rate / 1e6:.2f} Mbps")


                # 计算 path_transmit_time 的最大值（排除为 0 的值）
                # path_times = [info['Time'] for info in path_sinr_rates.values() if info['Time'] > 0]
                # max_path_time = np.max(path_times) if path_times else 0
                #print(f"Max Path Transmit Time: {max_path_time:.10f} s")

                optimized_routes_dict = {}

                ###########################   INLPS2   ############################################
                # 从现有的 routes_dict 中提取所有路径集
                RouteAll = list(routes_dict.values())

                # 是预先定义好的功率矩阵
                PMat = np.random.rand(len(nodes_data) + 1, len(nodes_data) + 1)  # 示例功率矩阵

                #初始化 Q 表，假设已经有一个初始的 Q 表
                Q_initial = Q
                Q = np.copy(Q_initial)

                #遍历所有节点来计算到服务器节点的最优路径

                for S in range(len(nodes_data)):
                    # 定义 Layer 和 SNR 矩阵，包含服务器节点的信息
                    Layer = nodes_data['Layer'].values
                    Layer = np.append(Layer, 0)  # 服务器在层级 0（可以认为它是层级最高的）
                    #大表
                    global_index_INLPS=nodes_data['Node_ID'].values
                    _, _, _, global_route_INLPS = Qlearning_INLPS_routing(S, server_id, Layer, global_index_INLPS, sinr_mat, Q, N, max_radius)

                    global_table_INLPS.loc[global_table_INLPS['Node_ID'] == global_route_INLPS[0], 'route'] = str(global_route_INLPS)

                    # 使用之前更新的 Q 表进行路径优化
                    MRe, optimized_route, Q_INLPS2 = Qlearning_INLPS2_routing(S, D, nodes_data['Layer'].values, sinr_mat, Q, N)
                    # 储存路径
                    #是否根据CPNC优化路径

                    # if CPNC:
                    #     if len(optimized_route) > 2:
                    #         optimized_route= [optimized_route[0], *optimized_route[2:]]
                    #         optimized_routes_dict[S]=[optimized_route, nodes_data.iloc[S][['Node_X', 'Node_Y']].tolist(),
                    #                             int(nodes_data.iloc[S][['Packet_Length']])]
                    #     else:
                    #         optimized_routes_dict[S] = [optimized_route, nodes_data.iloc[S][['Node_X', 'Node_Y']].tolist(),
                    #                             int(nodes_data.iloc[S][['Packet_Length']])]
                    # else:
                    #     optimized_routes_dict[S] = [optimized_route, nodes_data.iloc[S][['Node_X', 'Node_Y']].tolist(),
                    #                             int(nodes_data.iloc[S][['Packet_Length']])]

                    optimized_routes_dict[S] = [optimized_route, nodes_data.iloc[S][['Node_X', 'Node_Y']].tolist(),
                                                int(nodes_data.iloc[S][['Packet_Length']])]
                optimized_routes_dict[S + 1] = [[S + 1], list(server_coordinates[current_sever_id]), 0]

                # 计算传输速率
                capacity_INLPS, time_INLPS, total_interference_INLPS, average_interference_INLPS, average_path_rate_NLPS = calculate_network_metrics(routes_dict, pathloss_log, transmission_power, noise, B)

                path_sinr_rates_INLPS2 = calculate_path_sinr_and_rates(optimized_routes_dict, pathloss_log,
                                                                transmission_power, noise, B)

                # 调用 MARL 训练
                num_epochs = 100
                marl_training(routes_dict, num_epochs, pathloss_log, transmission_power, noise)
                # 可视化每个节点的路径
                for node, (path, _, _) in routes_dict.items():
                    print(f"Node {node} Path after MARL training: {path}")

                # 使用 MARL 训练后的路径
                # 更新 MARL 优化后的路径
                optimized_routes_dict_marl = {
                    node: [route, info[1], info[2]]
                    for node, (route, info) in routes_dict.items()
                }

                # 打印 MARL 优化后的多跳路径
                print("Optimized Routes (MARL):")
                for node, (path, coordinates, packet_length) in optimized_routes_dict_marl.items():
                    print(f"Node {node}: Path = {path}, Coordinates = {coordinates}, Packet Length = {packet_length}")

                # 计算SINR表
                #sinr_mat_INLPS2 = calculate_sinr_mat(routes_dict=optimized_routes_dict, pathloss_log=pathloss_log, transmission_power=transmission_power, noise=noise, B=B)
                # 打印路径和速率
                # for node, (path, coordinates, _) in optimized_routes_dict.items():
                #     sinr_info = path_sinr_rates_INLPS2.get(node, {})
                #     sinrs_INLPS2 = sinr_info.get('SINRs', [])
                #     rates_INLPS2 = sinr_info.get('Rates', [])
                #     path_rate_INLPS2 = sinr_info.get('PathRate', 0)
                #     path_transmit_time_INLPS2 = sinr_info.get('Time', 0)
                    #print(f"Optimized Node {node} to Server: Path = {path}, Path Rate = {path_rate_INLPS2 / 1e6:.2f} Mbps, Path Time = {path_transmit_time_INLPS2} s.")

                # 计算 path_rate 的平均值（排除为 0 的值）
                path_rates_INLPS2 = [info['PathRate'] for info in path_sinr_rates_INLPS2.values() if info['PathRate'] > 0]
                avg_path_rate_INLPS2 = np.average(path_rates_INLPS2) if path_rates_INLPS2 else 0
                # print(f"Average Path Rate: {avg_path_rate_INLPS2 / 1e6:.2f} Mbps")
                sum_path_rate_INLPS2 = np.sum(path_rates_INLPS2) if path_rates_INLPS2 else 0
                # print(f"Sum Path Rate: {sum_path_rate_INLPS2 / 1e6:.2f} Mbps")

                # 计算 path_transmit_time 的最大值（排除为 0 的值）
                path_times_INLPS2 = [info['Time'] for info in path_sinr_rates_INLPS2.values() if info['Time'] > 0]
                max_path_time_INLPS2 = np.max(path_times_INLPS2) if path_times_INLPS2 else 0
                # print(f"Max Path Transmit Time: {max_path_time_INLPS2:.10f} s")

                server_information_1.append(
                    [capacity_NLPS, time_NLPS, total_interference_NLPS, average_interference_NLPS, average_path_rate_NLPS,
                     capacity_INLPS, time_INLPS, total_interference_INLPS, average_interference_INLPS, average_path_rate_NLPS])

                server_information_2.append(
                    [avg_path_rate_NLPS2, sum_path_rate_NLPS2, max_path_time_INLPS2, avg_path_rate_INLPS2,
                     sum_path_rate_INLPS2, max_path_time_INLPS2])

            #循环结束
            node_positions_NLPS = global_table_NLPS[['Node_ID', 'Node_X', 'Node_Y']].set_index('Node_ID').to_dict(orient='index')
            node_positions_INLPS = global_table_INLPS[['Node_ID', 'Node_X', 'Node_Y']].set_index('Node_ID').to_dict(orient='index')
            # 从 server_table 提取服务器坐标
            server_positions_NLPS = server_table.set_index('Serer_ID')[['Server_X', 'Server_Y']].to_dict(orient='index')
            server_positions_INLPS = server_table.set_index('Serer_ID')[['Server_X', 'Server_Y']].to_dict(orient='index')
            # 调用干扰计算函数
            total_capacity_NLPS_3, total_interference_power_NLPS_3, average_interference_NLPS_3, total_time_NLPS_3, node_results_NLPS_3, average_path_rate_NLPS_3 = calculate_network_performance(global_table_NLPS, node_positions_NLPS, server_positions_NLPS, transmission_power, pathloss_log, noise, B)
            total_capacity_INLPS_3, total_interference_power_INLPS_3, average_interference_INLPS_3, total_time_INLPS_3, node_results_INLPS_3, average_path_rate_INLPS_3  = calculate_network_performance(global_table_INLPS, node_positions_INLPS, server_positions_INLPS, transmission_power, pathloss_log, noise, B)
            #to dBm
            total_interference_power_NLPS_3_dBm = mw_to_dbm(total_interference_power_NLPS_3)
            average_interference_NLPS_3_dBm = mw_to_dbm( average_interference_NLPS_3)
            total_interference_power_INLPS_3_dBm = mw_to_dbm(total_interference_power_INLPS_3)
            average_interference_INLPS_3_dBm = mw_to_dbm(average_interference_INLPS_3)
            #interference_results_NLPS, total_interference_power_NLPS, average_interference_power_NLPS = calculate_interference(global_table_NLPS, node_positions_NLPS, server_positions_NLPS, transmission_power)
            #interference_results_INLPS, total_interference_power_INLPS, average_interference_power_INLPS = calculate_interference(global_table_INLPS, node_positions_INLPS, server_positions_INLPS, transmission_power)

            # Case 1
            server_information_1 = np.array(server_information_1)
            total_network_capacity_ave_NLPS_1 = np.mean(server_information_1[:, 0]) / 1e6  # 转换为 Mbps
            total_network_capacity_sum_NLPS_1 = np.sum(server_information_1[:, 0]) / 1e6  # 转换为 Mbps
            max_transmission_time_NLPS_1 = np.max(server_information_1[:, 1])
            ave_transmission_time_NLPS_1 = np.mean(server_information_1[:, 1])
            total_interference_power_NLPS_1_mW = np.sum(server_information_1[:, 2])  # mW
            ave_interference_power_NLPS_1_mW = np.mean(server_information_1[:, 3])  # mW
            total_interference_power_NLPS_1_dBm = mw_to_dbm(total_interference_power_NLPS_1_mW)  # 转换为 dBm
            ave_interference_power_NLPS_1_dBm = mw_to_dbm(ave_interference_power_NLPS_1_mW)  # 转换为 dBm
            ave_path_rate_NLPS_1 = np.mean(server_information_1[:, 4]) / 1e6  # 转换为 Mbps
            total_network_capacity_ave_INLPS_1 = np.mean(server_information_1[:, 5]) / 1e6  # 转换为 Mbps
            total_network_capacity_sum_INLPS_1 = np.sum(server_information_1[:, 5]) / 1e6  # 转换为 Mbps
            max_transmission_time_INLPS_1 = np.max(server_information_1[:, 6])
            ave_transmission_time_INLPS_1 = np.mean(server_information_1[:, 6])
            total_interference_power_INLPS_1_mW = np.sum(server_information_1[:, 7])  # mW
            ave_interference_power_INLPS_1_mW = np.mean(server_information_1[:, 8])  # mW
            total_interference_power_INLPS_1_dBm = mw_to_dbm(total_interference_power_INLPS_1_mW)  # 转换为 dBm
            ave_interference_power_INLPS_1_dBm = mw_to_dbm(ave_interference_power_INLPS_1_mW)  # 转换为 dBm
            ave_path_rate_INLPS_1 = np.mean(server_information_1[:, 9]) / 1e6  # 转换为 Mbps

            # Case 2
            server_information_2 = np.array(server_information_2)
            total_network_capacity_ave_NLPS_2 = np.mean(server_information_2[:, 0]) / 1e6  # 转换为 Mbps
            total_network_capacity_sum_NLPS_2 = np.sum(server_information_2[:, 1]) / 1e6  # 转换为 Mbps
            max_transmission_time_NLPS_2 = np.max(server_information_2[:, 2])
            total_network_capacity_ave_INLPS_2 = np.mean(server_information_2[:, 3]) / 1e6  # 转换为 Mbps
            total_network_capacity_sum_INLPS_2 = np.sum(server_information_2[:, 4]) / 1e6  # 转换为 Mbps
            max_transmission_time_INLPS_2 = np.max(server_information_2[:, 5])

            #Case 3

            # Display the table
            # print(server_table)
            # print(global_table)
            # 输出结果
            # print("################## Conclutions #################")
            # if CPNC:
            #     print("With CPNC")
            # else:
            #     print("No CPNC")

            # print(f"Total interference power for NLPS: {total_interference_power_NLPS} mW")
            # print(f"Total interference power for INLPS: {total_interference_power_INLPS} mW")
            # print(f"Average interference power for NLPS: {average_interference_power_NLPS} mW")
            # print(f"Average interference power for INLPS: {average_interference_power_INLPS} mW")
            # print(f"Maximum transmission time for NLPS: {max_transmission_time_NLPS} s")
            # print(f"Maximum transmission time for INLPS: {max_transmission_time_INLPS} s")
            # print(f"Total network capacity for NLPS: {total_network_capacity_sum_NLPS / 1000000}  Mbps")
            # print(f"Total network capacity for INLPS: {total_network_capacity_sum_INLPS / 1000000}  Mbps")
            # print(f"Average link capacity for NLPS: {total_network_capacity_ave_NLPS / 1000000}  Mbps")
            # print(f"Average link capacity for INLPS: {total_network_capacity_ave_INLPS / 1000000}  Mbps")

            #run many times to obtain the actual average of all values for scenarios.
            # 记录结果
            results.append({
                "File Name": file_name,
                "Total Interference Power NLPS 1 (dBm)": total_interference_power_NLPS_1_dBm,
                "Average Interference Power NLPS 1 (dBm)": ave_interference_power_NLPS_1_dBm,
                "Total Interference Power INLPS 1 (dBm)": total_interference_power_INLPS_1_dBm,
                "Average Interference Power INLPS 1 (dBm)": ave_interference_power_INLPS_1_dBm,
                "Total Network Capacity (Ave NLPS 1, Mbps)": total_network_capacity_ave_NLPS_1,
                "Total Network Capacity (Sum NLPS 1, Mbps)": total_network_capacity_sum_NLPS_1,
                "Total Network Capacity (Ave INLPS 1, Mbps)": total_network_capacity_ave_INLPS_1,
                "Total Network Capacity (Sum INLPS 1, Mbps)": total_network_capacity_sum_INLPS_1,
                "Average Path Rate NLPS 1 (Mbps)": ave_path_rate_NLPS_1,
                "Average Path Rate INLPS 1 (Mbps)": ave_path_rate_INLPS_1,
                "Max Transmission Time NLPS 1 (ms)": max_transmission_time_NLPS_1,
                "Average Transmission Time NLPS 1 (ms)": ave_transmission_time_NLPS_1,
                "Max Transmission Time INLPS 1 (ms)": max_transmission_time_INLPS_1,
                "Average Transmission Time INLPS 1 (ms)": ave_transmission_time_INLPS_1,

                "Total Network Capacity (Ave NLPS 2, Mbps)": total_network_capacity_ave_NLPS_2,
                "Total Network Capacity (Sum NLPS 2, Mbps)": total_network_capacity_sum_NLPS_2,
                "Max Transmission Time NLPS 2 (ms)": max_transmission_time_NLPS_2,
                "Total Network Capacity (Ave INLPS 2, Mbps)": total_network_capacity_ave_INLPS_2,
                "Total Network Capacity (Sum INLPS 2, Mbps)": total_network_capacity_sum_INLPS_2,
                "Max Transmission Time INLPS 2 (ms)": max_transmission_time_INLPS_2,

                "Total Interference Power NLPS 3 (dBm)": total_interference_power_NLPS_3_dBm,
                "Total Interference Power INLPS 3 (dBm)": total_interference_power_INLPS_3_dBm,
                "Average Interference Power NLPS 3 (dBm)" : average_interference_NLPS_3_dBm,
                "Average Interference Power INLPS 3 (dBm)": average_interference_INLPS_3_dBm,
                "Total Network Capacity (Sum NLPS 3, Mbps)": total_capacity_NLPS_3 / 1e6,
                "Total Network Capacity (Sum INLPS 3, Mbps)": total_capacity_INLPS_3 / 1e6,
                "Max Transmission Time NLPS 3 (ms)": total_time_NLPS_3,
                "Max Transmission Time INLPS 3 (ms)": total_time_INLPS_3,
                "Average Path Rate NLPS 3 (Mbps)": average_path_rate_NLPS_3 / 1e6,
                "Average Path Rate INLPS 3 (Mbps)": average_path_rate_INLPS_3 / 1e6

            })
            # 将结果逐条打印出来
            print(f"Results for file: {file_name}")
            print(f"Total Interference Power NLPS 1 (dBm): {total_interference_power_NLPS_1_dBm}")
            print(f"Average Interference Power NLPS 1 (dBm): {ave_interference_power_NLPS_1_dBm}")
            print(f"Total Interference Power INLPS 1 (dBm): {total_interference_power_INLPS_1_dBm}")
            print(f"Average Interference Power INLPS 1 (dBm): {ave_interference_power_INLPS_1_dBm}")
            print(f"Total Network Capacity (Ave NLPS 1, Mbps): {total_network_capacity_ave_NLPS_1}")
            print(f"Total Network Capacity (Sum NLPS 1, Mbps): {total_network_capacity_sum_NLPS_1}")
            print(f"Total Network Capacity (Ave INLPS 1, Mbps): {total_network_capacity_ave_INLPS_1}")
            print(f"Total Network Capacity (Sum INLPS 1, Mbps): {total_network_capacity_sum_INLPS_1}")
            print(f"Average Path Rate NLPS 1 (Mbps): {ave_path_rate_NLPS_1}")
            print(f"Average Path Rate INLPS 1 (Mbps): {ave_path_rate_INLPS_1}")
            print(f"Max Transmission Time NLPS 1 (ms): {max_transmission_time_NLPS_1}")
            print(f"Average Transmission Time NLPS 1 (ms): {ave_transmission_time_NLPS_1}")
            print(f"Max Transmission Time INLPS 1 (ms): {max_transmission_time_INLPS_1}")
            print(f"Average Transmission Time INLPS 1 (ms): {ave_transmission_time_INLPS_1}")

            print(f"Total Network Capacity (Ave NLPS 2, Mbps): {total_network_capacity_ave_NLPS_2}")
            print(f"Total Network Capacity (Sum NLPS 2, Mbps): {total_network_capacity_sum_NLPS_2}")
            print(f"Max Transmission Time NLPS 2 (ms): {max_transmission_time_NLPS_2}")
            print(f"Total Network Capacity (Ave INLPS 2, Mbps): {total_network_capacity_ave_INLPS_2}")
            print(f"Total Network Capacity (Sum INLPS 2, Mbps): {total_network_capacity_sum_INLPS_2}")
            print(f"Max Transmission Time INLPS 2 (ms): {max_transmission_time_INLPS_2}")

            print(f"Total Interference Power NLPS 3 (dBm): {total_interference_power_NLPS_3_dBm}")
            print(f"Total Interference Power INLPS 3 (dBm): {total_interference_power_INLPS_3_dBm}")
            print(f"Average Interference Power NLPS 3 (dBm): {average_interference_NLPS_3_dBm}")
            print(f"Average Interference Power INLPS 3 (dBm): {average_interference_INLPS_3_dBm}")
            print(f"Total Network Capacity (Sum NLPS 3, Mbps): {total_capacity_NLPS_3 / 1e6}")
            print(f"Total Network Capacity (Sum INLPS 3, Mbps): {total_capacity_INLPS_3 / 1e6}")
            print(f"Max Transmission Time NLPS 3 (ms): {total_time_NLPS_3}")
            print(f"Max Transmission Time INLPS 3 (ms): {total_time_INLPS_3}")
            print(f"Average Path Rate NLPS 3 (Mbps): {average_path_rate_NLPS_3 / 1e6}")
            print(f"Average Path Rate INLPS 3 (Mbps): {average_path_rate_INLPS_3 / 1e6}")
            print("=" * 80)  # 分隔线

    # 保存结果到 CSV 文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")

