import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
from utilts_3_1 import *
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

    for file_name in sorted(os.listdir(input_directory)):
        if file_name.startswith("task_info_with_decision_") and file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)
            print(file_path)
            labeled_data = adaptively_label_dataset(file_path)
            labeled_data["Node_ID"]=labeled_data["Node_ID"].astype("int")
            labeled_data["Packet_Length"]=labeled_data["Packet_Length"].astype("int")
            labeled_data["Assigned_Server"]=labeled_data["Assigned_Server"].astype("int")

            #画图，可以注释掉
            #plot_nodes_and_servers_corrected(labeled_data)

            # 处理标记数据集
            labeled_data = filter_nodes_by_distance(labeled_data)
            labeled_data["Node_ID"] = labeled_data["Node_ID"].apply(lambda x: int(x))
            labeled_data["Assigned_Server"] = labeled_data["Assigned_Server"].apply(lambda x: f"Server{int(x)}")
            # 应用函数到标记数据集
            #server_node_data = extract_server_node_data(labeled_data)
            # print(server_node_data)

            # 显示每个服务器的节点数量摘要
            #server_node_data_summary = {server_id: data.shape[0] for server_id, data in server_node_data.items()}
            # 应用函数到标记数据集
            server_coordinates = extract_server_coordinates(labeled_data)
            # 一个字典取每个server对应的坐标 dict[服务器id]：该服务器对应的坐标
            # print("服务器字典：", server_node_data_summary)
            # print("server坐标：", server_coordinates)
            selected_columns = ['Node_ID', 'Packet_Length', 'Node_X', 'Node_Y', 'Assigned_Server']
            # 选择所需的列
            global_table = labeled_data[selected_columns]
            #print(len(global_table))
            #这里开始遍历：
            all_map_nodes_transmission_dict={}
            new_ids = [-1 * int(key[-1])-1 for key in server_coordinates.keys()]

            # Create the new table
            data = {
            'Server_ID': new_ids,
            'Server_X': [coords[0] for coords in server_coordinates.values()],
            'Server_Y': [coords[1] for coords in server_coordinates.values()]
            }
            server_table = pd.DataFrame(data)

            server_mapping = {f"Server{-server_id - 1}": server_id for server_id in server_table['Server_ID']}
            global_table = global_table.copy()  # 显式复制以避免潜在问题
            global_table.loc[:, 'Mapped_Server_ID'] = global_table['Assigned_Server'].map(server_mapping)
            #print(global_table.head())
            global_table_normal = global_table.copy()
            global_table_NLPS = global_table.copy()
            global_table_INLPS = global_table.copy()
            print("################## Normal #################")
            routes, total_hops, average_hops = q_learning_multi_hop_routing(global_table, server_table, max_radius)
            # plot_routes_with_colorss(global_table, server_table, routes, "./Normal Q-learning Routes.svg")
            print("################## NLPS #################")
            routes_NLPS, total_hops_NLPS, average_hops_NLPS = q_learning_with_snr(global_table, server_table, max_radius, transmission_power, noise, B, pathloss_log)
            # plot_routes_with_colorss(global_table, server_table, routes_NLPS, "./NLPS Routes.svg")
            print("################## INLPS #################")
            routes_INLPS, total_hops_INLPS, average_hops_INLPS = q_learning_with_sinr(global_table, server_table, max_radius, transmission_power, noise, B, pathloss_log)
            # plot_routes_with_colorss(global_table, server_table, routes_INLPS, "./INLPS Routes.svg")

            # 找出两种算法的路径覆盖节点差异
            # nodes_with_inlps_routes = set(routes_INLPS.keys())
            # nodes_with_nlps_routes = set(routes_NLPS.keys())
            #
            # # 两种算法均覆盖的节点
            # common_nodes = nodes_with_inlps_routes & nodes_with_nlps_routes
            # for node in common_nodes:
            #     if routes_INLPS[node] != routes_NLPS[node]:
            #         print(f"Node {node}: Different paths")
            #         print(f"INLPS Path: {routes_INLPS[node]}")
            #         print(f"NLPS Path: {routes_NLPS[node]}")
            # 计算跳数的统计指标
            hops_list = [len(path) - 1 for path in routes.values()]  # 获取所有路径的跳数
            # print(len(hops_list))
            total_hops = sum(hops_list)
            average_hops = total_hops / len(hops_list) if hops_list else 0
            median_hops = np.median(hops_list) if hops_list else 0
            mode_hops = max(set(hops_list), key=hops_list.count) if hops_list else None

            hops_list_NLPS = [len(path) - 1 for path in routes_NLPS.values()]  # 获取所有路径的跳数
            # print(len(hops_list_NLPS))
            total_hops_NLPS = sum(hops_list_NLPS)
            average_hops_NLPS = total_hops_NLPS / len(hops_list_NLPS) if hops_list_NLPS else 0
            median_hops_NLPS = np.median(hops_list_NLPS) if hops_list_NLPS else 0
            mode_hops_NLPS = max(set(hops_list_NLPS), key=hops_list_NLPS.count) if hops_list_NLPS else None

            hops_list_INLPS = [len(path) - 1 for path in routes_INLPS.values()]  # 获取所有路径的跳数
            total_hops_INLPS = sum(hops_list_INLPS)
            # print(len(hops_list_INLPS))
            average_hops_INLPS = total_hops_INLPS / len(hops_list_INLPS) if hops_list_INLPS else 0
            median_hops_INLPS = np.median(hops_list_INLPS) if hops_list_INLPS else 0
            mode_hops_INLPS = max(set(hops_list_INLPS), key=hops_list_INLPS.count) if hops_list_INLPS else None


            # 将 routes 合并到 global_table
            global_table_normal['route'] = global_table_normal['Node_ID'].map(lambda node: routes.get(node, []))
            global_table_normal['route'] = global_table_normal['route'].apply(lambda x: json.dumps(x) if x else "[]")
            global_table_NLPS['route'] = global_table_NLPS['Node_ID'].map(lambda node: routes_NLPS.get(node, []))
            global_table_NLPS['route'] = global_table_NLPS['route'].apply(lambda x: json.dumps(x) if x else "[]")
            global_table_INLPS['route'] = global_table_INLPS['Node_ID'].map(lambda node: routes_INLPS.get(node, []))
            global_table_INLPS['route'] = global_table_INLPS['route'].apply(lambda x: json.dumps(x) if x else "[]")
            # # 打印结果
            # for node, path in routes.items():
            #     print(f"Node {node} to Server: Path = {path}")
            node_positions_normal = global_table_normal[['Node_ID', 'Node_X', 'Node_Y']].set_index('Node_ID').to_dict(
                orient='index')
            node_positions_NLPS = global_table_NLPS[['Node_ID', 'Node_X', 'Node_Y']].set_index('Node_ID').to_dict(
                orient='index')
            node_positions_INLPS = global_table_INLPS[['Node_ID', 'Node_X', 'Node_Y']].set_index('Node_ID').to_dict(
                orient='index')

            # 从 server_table 提取服务器坐标
            server_positions_normal = server_table.set_index('Server_ID')[['Server_X', 'Server_Y']].to_dict(orient='index')
            server_positions_NLPS = server_table.set_index('Server_ID')[['Server_X', 'Server_Y']].to_dict(orient='index')
            server_positions_INLPS = server_table.set_index('Server_ID')[['Server_X', 'Server_Y']].to_dict(orient='index')

            # 调用干扰计算函数
            total_capacity_normal_3, total_interference_power_normal_3, average_interference_normal_3, total_time_normal_3, node_results_normal_3, average_path_rate_normal_3 = calculate_network_performance(
                global_table_normal, node_positions_normal, server_positions_normal, transmission_power, pathloss_log, noise,
                B)
            total_capacity_NLPS_3, total_interference_power_NLPS_3, average_interference_NLPS_3, total_time_NLPS_3, node_results_NLPS_3, average_path_rate_NLPS_3 = calculate_network_performance(
                global_table_NLPS, node_positions_NLPS, server_positions_NLPS, transmission_power, pathloss_log, noise,
                B)
            total_capacity_INLPS_3, total_interference_power_INLPS_3, average_interference_INLPS_3, total_time_INLPS_3, node_results_INLPS_3, average_path_rate_INLPS_3 = calculate_network_performance(
                global_table_INLPS, node_positions_INLPS, server_positions_INLPS, transmission_power, pathloss_log,
                noise, B)

            # to dBm
            total_interference_power_normal_3_dBm = mw_to_dbm(total_interference_power_normal_3)
            total_interference_power_NLPS_3_dBm = mw_to_dbm(total_interference_power_NLPS_3)
            total_interference_power_INLPS_3_dBm = mw_to_dbm(total_interference_power_INLPS_3)

            print(f"Results for file: {file_name}")
            print(f"Total hops distance: {total_hops}")
            print(f"Total hops NLPS: {total_hops_NLPS}")
            print(f"Total hops INLPS: {total_hops_INLPS}")
            print(f"Average hops distance: {average_hops}")
            print(f"Average hops NLPS: {average_hops_NLPS}")
            print(f"Average hops INLPS: {average_hops_INLPS}")
            # print(f"Media hops distance: {median_hops}")
            # print(f"Media hops NLPS: {median_hops_NLPS}")
            # print(f"Media hops INLPS: {median_hops_INLPS}")
            # print(f"Mode hops distance: {mode_hops}")
            # print(f"Mode hops NLPS: {mode_hops_NLPS}")
            # print(f"Mode hops INLPS: {mode_hops_INLPS}")

            print(f"Total Interference Power distance 3 (dBm): {total_interference_power_normal_3_dBm}")
            print(f"Total Interference Power NLPS 3 (dBm): {total_interference_power_NLPS_3_dBm}")
            print(f"Total Interference Power INLPS 3 (dBm): {total_interference_power_INLPS_3_dBm}")
            print(f"Total Network Capacity (Sum distance 3, Mbps): {total_capacity_normal_3 / 1e6}")
            print(f"Total Network Capacity (Sum NLPS 3, Mbps): {total_capacity_NLPS_3 / 1e6}")
            print(f"Total Network Capacity (Sum INLPS 3, Mbps): {total_capacity_INLPS_3 / 1e6}")
            print(f"Max Transmission Time distance 3 (ms): {total_time_normal_3}")
            print(f"Max Transmission Time NLPS 3 (ms): {total_time_NLPS_3}")
            print(f"Max Transmission Time INLPS 3 (ms): {total_time_INLPS_3}")
            print(f"Average Path Rate distance 3 (Mbps): {average_path_rate_normal_3 / 1e6}")
            print(f"Average Path Rate NLPS 3 (Mbps): {average_path_rate_NLPS_3 / 1e6}")
            print(f"Average Path Rate INLPS 3 (Mbps): {average_path_rate_INLPS_3 / 1e6}")
            print("=" * 80)  # 分隔线
            print("=" * 80)  # 分隔线
            results.append({
                "File Name": file_name,
                "Total hops distance": total_hops,
                "Total hops NLPS": total_hops_NLPS,
                "Total hops INLPS": total_hops_INLPS,
                "Average hops distance":average_hops,
                "Average hops NLPS": average_hops_NLPS,
                "Average hops INLPS": average_hops_INLPS,
                "Media hops distance": median_hops,
                "Media hops NLPS": median_hops_NLPS,
                "Media hops INLPS": median_hops_INLPS,
                "Mode hops distance": mode_hops,
                "Mode hops NLPS": mode_hops_NLPS,
                "Mode hops INLPS": mode_hops_INLPS

                # "Total Interference Power distance 3 (dBm)": total_interference_power_normal_3_dBm,
                # "Total Interference Power NLPS 3 (dBm)": total_interference_power_NLPS_3_dBm,
                # "Total Interference Power INLPS 3 (dBm)": total_interference_power_INLPS_3_dBm,
                # "Total Network Capacity(Sum distance 3, Mbps)": total_capacity_normal_3 / 1e6,
                # "Total Network Capacity (Sum NLPS 3, Mbps)": total_capacity_NLPS_3 / 1e6,
                # "Total Network Capacity (Sum INLPS 3, Mbps)": total_capacity_INLPS_3 / 1e6,
                # "Max Transmission Time distance 3 (ms)": total_time_normal_3,
                # "Max Transmission Time NLPS 3 (ms)": total_time_NLPS_3,
                # "Max Transmission Time INLPS 3 (ms)": total_time_INLPS_3,
                # "Average Path Rate distance 3 (Mbps)": average_path_rate_normal_3 / 1e6,
                # "Average Path Rate NLPS 3 (Mbps)": average_path_rate_NLPS_3 / 1e6,
                # "Average Path Rate INLPS 3 (Mbps)": average_path_rate_INLPS_3 / 1e6
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")