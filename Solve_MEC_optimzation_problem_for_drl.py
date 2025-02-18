from gurobipy import *
import csv
import pathlib
import numpy as np
import random
import math
import os
import time
#import goto
import pandas as pd
#from goto import with_goto
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
_parent = pathlib.Path(__file__).parent  # 获取当前文件的子目录
import time  # 在文件开始处导入time模块
# 生成一个随机数以确保文件名唯一

#802.11ax 5GHz

#@with_goto
def test():
    global file_name1
    global file_name2
    #label.begin
    # 创建模型
    model = Model("MEC-Offloading")
    # ******************************************************************************
    # ******************************************************************************
    # 定义变量，计算矩阵
    # ******************************************************************************
    # ******************************************************************************

    server_number = 10
    device_number = 200
    output_directory = f"./data/device/{device_number}device"
    #output_directory = f"./data/server/{server_number}server"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    bandwidth = 40 * 10 ** 6  # Hz
    area_length = 200  # m
    area_width = 200
    # 干扰模型参数
    d0 = 1  # decorrelation distance m
    attenuation_constant = 3.5
    shadowing = 4 # dB
    wall_attenuation = 0 # dB
    noise_level = 3.9810717055 * 10 ** -18 # mW/Hz, -174 dBm/Hz
    # 传输参数
    beta_0 = -60  # dB
    sigma2 = -96  # dBm
    transmission_power = 200  # mW
    noise = bandwidth * noise_level
    # 全局参数
    time_slot = 5  # s
    arrival_rate = 100  # 每秒100个任务
    total_time = 100  # s
    timer = 0   # 记录系统时间
    # 任务参数的取值范围
    task_cpu_cycles_per_bit_max = 50000
    task_cpu_cycles_per_bit_min = 10000
    #task_volume_max = 3 * 10 ** 7  # bit
    #task_volume_min = 1 * 10 ** 5  # bit
    task_volume_max = 1500 * 8  # bit
    task_volume_min = 1400 * 8 # bit
    task_tolerable_time_max = 4000  # 秒
    task_tolerable_time_min = 2000  # 秒
    server_capability = [8 * 10 ** 10 for i in range(server_number)]  # 每个服务器80GHz的总算力
    server_antenna_height = 0
    server_location = []
    server_range = 2000  # m
    max_distance = 120
    def pathloss_log(distance):
        PL0 = 20.0 * np.log10(d0)
        PL1 = PL0 + (10.0 * attenuation_constant * np.log10(distance / d0)) - wall_attenuation + shadowing
        G = 1.0 / (10.0 ** (PL1 / 10.0))
        return G



    # 服务器参数

    for i in range(server_number):
        x = random.uniform(0, area_length)  # 在0到area_length之间随机选择x坐标
        y = random.uniform(0, area_width)  # 在0到area_width之间随机选择y坐标

        server_location.append([x, y])

    server_remained_capability = [8 * 10 ** 10 for i in range(server_number)]


    # 设备参数
    device_capability = []
    device_location = []
    distance_victor_between_i_and_j = [[0]*server_number for _ in range(device_number)]
    for i in range(device_number):
        device_location.append([random.uniform(0, area_length), random.uniform(0, area_width)])
        # 设备处理能力 0.1-1GHz
        c = random.uniform(1e7, 1e8)
        device_capability.append(c)
    transmission_rate_of_device_and_servers = [[0]*server_number for _ in range(device_number)]
    # print(transmission_rate_of_device_and_servers)

    # 初始化存储设备间距离的矩阵
    distance_matrix_between_devices = [[0] * device_number for _ in range(device_number)]
    G_matrix_between_devices = np.zeros((device_number, device_number))

    # 计算每对设备间的距离
    for i in range(device_number):
        for j in range(device_number):
            if i != j:  # 确保不计算设备自身的距离
                distance_matrix_between_devices[i][j] = ((device_location[i][0] - device_location[j][0]) ** 2 +
                                                         (device_location[i][1] - device_location[j][1]) ** 2) ** 0.5
                G_matrix_between_devices[i][j] = pathloss_log(distance_matrix_between_devices[i][j])
            else:
                distance_matrix_between_devices[i][j] = 0  # 设备自身的距离为零
    #G值计算
    G_matrix_between_devices_and_servers = np.zeros((device_number, server_number))
    SINR_matrix = np.zeros((device_number, server_number))
    SINR_transmission_rate_matrix = np.zeros((device_number, server_number))
    for j in range(device_number):
        for i in range(server_number):
            # 计算服务器和设备距离
            distance_of_ji = ((server_location[i][0] - device_location[j][0]) ** 2 +
                              (server_location[i][1] - device_location[j][1]) ** 2) ** 0.5
            distance_victor_between_i_and_j [j][i]= distance_of_ji
            G_matrix_between_devices_and_servers[j][i] = pathloss_log(distance_victor_between_i_and_j[j][i])
            # 计算SINR
            S = G_matrix_between_devices_and_servers[j][i] * transmission_power
            N = noise
            I = sum(G_matrix_between_devices_and_servers[k][i] * transmission_power for k in range(device_number) if k != j)
            #I = sum(G_matrix_between_devices[j][k] * transmission_power for k in range(device_number) if k != j)
            SINR = S / (N + I)
            SINR_matrix[j][i] = SINR
            if SINR > 0:  # 避免对数中的负数
                SINR_transmission_rate = bandwidth * math.log2(1 + SINR)
            else:
                SINR_transmission_rate = 0
            SINR_transmission_rate_matrix[j][i] = SINR_transmission_rate

            if distance_of_ji > server_range:
                transmission_rate_of_device_and_servers[j][i] = 0
            else:
                transmission_rate_of_ji = bandwidth * math.log2(
                    1 + (transmission_power * beta_0) / (sigma2 * (distance_of_ji ** 2 + server_antenna_height ** 2)))
                # print(i, j, distance_of_ji, transmission_rate_of_ji)
                transmission_rate_of_device_and_servers[j][i] = transmission_rate_of_ji
    # print(transmission_rate_of_device_and_servers)



    #for row in G_matrix_between_devices_and_servers:
    #    print(row)
    # with open('distance_victor_between_i_and_j.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # 写入矩阵的每一行
    #     for row in distance_victor_between_i_and_j:
    #         writer.writerow(row)
    #
    #
    # with open('G_matrix_between_devices_and_servers.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # 写入矩阵的每一行
    #     for row in G_matrix_between_devices_and_servers:
    #         writer.writerow(row)
    #
    # #for row in SINR_matrix:
    # #    print(row)
    # with open('SINR_matrix.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # 写入矩阵的每一行
    #     for row in SINR_matrix:
    #         writer.writerow(row)

    #for row in SINR_transmission_rate_matrix:
    #    print(row)
    # with open('SINR_transmission_rate_matrix.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # 写入矩阵的每一行
    #     for row in SINR_transmission_rate_matrix:
    #         writer.writerow(row)
    # # 任务参数
    # 任务向量存储的信息：[任务号，设备号，所需算力，数据量，截止时间，设备排队数据量，设备算力，设备到各服务器的距离，各服务器剩余算力]
    # 直接生成100个任务
    task_tolerable_time = []
    device_i_queued_task_volume = []
    task_volume = []
    task_cpu_cycles_per_bit = []

    for j in range(device_number):
        task_cpu_cycles_per_bit.append(random.randint(task_cpu_cycles_per_bit_min, task_cpu_cycles_per_bit_max))
        task_volume.append(random.randint(task_volume_min, task_volume_max))
        task_tolerable_time.append(random.randint(task_tolerable_time_min, task_tolerable_time_max))
        device_i_queued_task_volume.append(random.randint(task_volume_min, task_volume_max))

    # 本地计算时延
    delay_local = []
    # 排队时延 = 排队计算量/设备算力
    # 计算时延 = 任务计算量/设备算力
    for j in range(device_number):
        t_j_local = (device_i_queued_task_volume[j] + task_volume[j]) * task_cpu_cycles_per_bit [j] / device_capability[j]
        delay_local.append(t_j_local)
    # print(delay_local)

    # 远程计算
    transmission_delay = [[0]*server_number for _ in range(device_number)]
    computing_delay = [[0]*server_number for _ in range(device_number)]
    total_delay_remote = [[0]*server_number for _ in range(device_number)]
    # 传输时延 = 任务数据量/传输速率
    for j in range(device_number):
        for i in range(server_number):

            if not transmission_rate_of_device_and_servers[j][i] == 0:
                transmission_delay[j][i] = task_volume[j] / transmission_rate_of_device_and_servers[j][i]
            else:
                transmission_delay[j][i] = 100000000000
    # print(transmission_delay)

    # 计算时延 = 任务计算量/分配的算力
    for j in range(device_number):
        for i in range(server_number):
            computing_delay[j][i] = task_volume [j] * task_cpu_cycles_per_bit[j] * 100 / server_remained_capability[i]
    #print(computing_delay)

    # 总时延 = 传输时延 + 计算时延
    for j in range(device_number):
        for i in range(server_number):
            total_delay_remote[j][i] = transmission_delay[j][i] + computing_delay[j][i]
    print("%%%%%%%%%%%%%%%")
    # # print(total_delay_remote)
    # csv_file_path = _parent / 'total_delay_remote001.csv'  # 使用_parent路径确保文件保存在正确的位置
    # with open(csv_file_path, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(total_delay_remote)
    print("%%%%%%%%%%%%%%%")
    # 添加决策变量
    decision = model.addVars(device_number, server_number, vtype=GRB.BINARY, name="decision")
    a_j = []
    for j in range(device_number):
        a_j.append(sum(decision[j, i] for i in range(server_number)))
    # print(decision)
    #  print(a_j)

    # 更新变量环境
    #model.update()
    # 标准化SINR，距离
    SINR_matrix = np.array(SINR_matrix)  # 确保SINR_matrix是一个NumPy数组
    distance_victor_between_i_and_j = np.array(distance_victor_between_i_and_j)
    # 计算最大值和最小值
    SINR_min = SINR_matrix.min()
    SINR_max = SINR_matrix.max()
    distance_victor_between_i_and_j_min = distance_victor_between_i_and_j.min()
    distance_victor_between_i_and_j_max = distance_victor_between_i_and_j.max()
    # 进行标准化
    normalized_SINR_matrix = (SINR_matrix - SINR_min) / (SINR_max - SINR_min)
    normalized_distance_victor_between_i_and_j = (distance_victor_between_i_and_j - distance_victor_between_i_and_j_min) / (distance_victor_between_i_and_j_max - distance_victor_between_i_and_j_min)
    # 初始化传输时间矩阵，用于存储每个设备到各服务器的传输时间
    SINR_transmission_time_matrix = np.zeros((device_number, server_number))

    # 计算传输时间
    for j in range(device_number):
        for i in range(server_number):
            if SINR_transmission_rate_matrix[j][i] > 0:  # 确保传输速率大于0，避免除以0
                SINR_transmission_time_matrix[j][i] = task_volume[j] / SINR_transmission_rate_matrix[j][i]
            else:
                SINR_transmission_time_matrix[j][i] = float('inf')  # 如果传输速率为0，设置传输时间为无穷大
    # print("shape:", transmission_time_matrix.shape)

    # 目标函数--定义优化问题，系统在决策下的总时间
    obj = QuadExpr()
    for j in range(device_number):
        for i in range(server_number):
            #obj += decision[j, i] * total_delay_remote[j][i]
            #obj += decision[j, i] * (1 - normalized_SINR_matrix[j][i])
            obj += decision[j, i] * SINR_transmission_time_matrix[j][i]
            #obj += decision[j, i] * distance_victor_between_i_and_j[j][i]
        obj += (1 - a_j[j]) * delay_local[j]
    model.setObjective(obj, GRB.MINIMIZE)

    # 约束
    # 单服务器约束
    #model.addConstrs(((quicksum(decision[j, i] for i in range(server_number)) <= 1) for j in range(device_number)), name='Single Server Constraint')
    model.addConstrs(((quicksum(decision[j, i] for i in range(server_number)) == 1) for j in range(device_number)), name='Single Server Constraint')

    # 时间约束
    #model.addConstrs(((quicksum(decision[j, i] * total_delay_remote[j][i] for i in range(server_number)) + (1-a_j[j]) * delay_local[j] <= task_tolerable_time[j]) for j in range(server_number)), name='Time Constraint')

    #model.update()
    # 求解
    model.Params.LogToConsole=True  # 显示求解过程
    model.Params.MIPGap=0.0001  # 百分比界差
    model.Params.TimeLimit=10000  # 限制求解时间为 100s

    # try:
    #     if model.status == GRB.INFEASIBLE or model.status == GRB.UNBOUNDED or model.status == GRB.ITERATION_LIMIT \
    #             or model.status == GRB.TIME_LIMIT or model.status == GRB.INF_OR_UNBD or model.status == GRB.CUTOFF or \
    #             model.status == GRB.NODE_LIMIT:
    #         goto.begin
    # except:
    #     pass
    try:
        if model.Status == GRB.OPTIMAL:
            pass
    except:
        goto.begin

    print("模型有解")
    # 在优化前记录开始时间
    start_time = time.time()
    # 继续求解模型
    if model.status == GRB.LOADED:
        model.update()
        model.optimize()
    else:
        goto.begin

    # 在优化后记录结束时间
    end_time = time.time()

    # 计算并打印Gurobi优化器的运行时间
    optimization_time = end_time - start_time
    #print("%%%%%%%%%%%%%%Gurobi Optimization Time: {} seconds".format(optimization_time))

    # 输出模型结果
    try:
        #print("Obj:", model.objVal)
        #print(decision)
        print("xxxxxxxx")
    except:
        goto.begin
    # 决策变量写入文件
    decision_list = [[0]*server_number for _ in range(device_number)]
    for j in range(device_number):
        for i in range(server_number):
            decision_list[j][i] = decision[j, i]

    # with open(_parent / 'decision_list.csv', 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for row in decision_list:
    #         writer.writerow(row)

    # 把环境变量和决策结果组合为数据集，写入文件
    # 任务向量存储的信息：[任务号，设备号（可能影响模型精度），所需算力A，数据量B，截止时间C，设备排队数据量D，设备算力E，设备到各服务器的距离FGHJ，各服务器剩余算力KLMNO，决策P]
    task_info = []
    for j in range(device_number):
        current_device_number = j
        #task_info.append([current_device_number, task_cpu_cycles_per_bit[j], task_volume[j], task_tolerable_time[j], device_i_queued_task_volume[j], device_capability[j]])
        task_info.append([current_device_number, task_volume[j]])
    #转为np array
    task_info_array = np.asarray(task_info)
    device_location_array = np.asarray(device_location)
    server_locations_expanded = np.tile(server_location, (device_number, 1))
    server_location_array = np.asarray(server_location)
    server_location_row = server_location_array.reshape(1, -1)
    server_location_repeated = np.tile(server_location_row, (device_number, 1))
    server_location_repeated_array = np.asarray(server_location_repeated )
    #print(server_location_array)
    #print(server_location_repeated_array)
    print(server_location_array.shape)
    print(server_location_repeated_array.shape)
    distance_victor_between_i_and_j_array = np.asarray(distance_victor_between_i_and_j)
    server_remained_capability_array = np.asarray(server_remained_capability)
    server_remained_capability_array = np.tile(server_remained_capability_array, (device_number, 1))
    SINR_transmission_time_matrix_array = np.asarray(SINR_transmission_time_matrix)
    SINR_matrix_array = np.asarray(SINR_matrix)
    G_matrix_between_devices_and_servers_array = np.array(G_matrix_between_devices_and_servers)
    # 查看array形状
    # print(task_info_array.shape)
    # print(distance_victor_between_i_and_j_array.shape)
    # print(server_remained_capability_array.shape)
    # array合并
    #task_info_array = np.hstack((task_info_array, distance_victor_between_i_and_j_array, server_remained_capability_array))
    #task_info_array = np.hstack((task_info_array, distance_victor_between_i_and_j_array, SINR_transmission_time_matrix_array))
    #task_info_array = np.hstack((task_info_array, distance_victor_between_i_and_j_array, SINR_transmission_time_matrix_array, G_matrix_between_devices_and_servers_array))
    task_info_array = np.hstack((task_info_array, device_location_array, server_location_repeated_array))

    # print(task_info_array.shape)

    # 给任务作标记
    # decision列表里的字符串进行过滤，只提取0和1
    decision_list = []
    for v in model.getVars():
        decision_list.append(v.x)
    decision_list_array = np.asarray(decision_list)
    decision_list_array = decision_list_array.reshape(device_number, -1)
    # print(decision_list_array)
    # 把决策变量转化为标签--服务器号
    labels = []
    for j in range(device_number):
        flag = 0
        for i in range(server_number):
            if decision_list_array[j][i] == 1:
                flag = 1
                labels.append(i)
        if flag == 0:
            labels.append(server_number)
    # print(labels)
    label_array = np.asarray(labels)
    label_array = label_array.reshape(-1, 1)
    # print(label_array.shape)
    task_info_array = np.hstack((task_info_array, label_array))

    # 初始化保存每个设备传输速率的列表
    task_transmit_rates = []
    SINR_task_transmit_rates = []
    SINR2_task_transmit_rates = []
    SINR_transmit_times = []
    SINR2_transmit_times = []
    for task in task_info_array:
        device_id, server_id, data_length = task[0], task[-1], task[2]
        new_interference = 0
        # 检查设备是否选择发送
        if server_id < server_number:
            device_id = int(device_id)
            server_id = int(server_id)
            signal_strength = G_matrix_between_devices_and_servers[device_id][server_id] * transmission_power
            transmit_time = transmission_delay[device_id][server_id]
            SINR_task_transmit_rate = SINR_transmission_rate_matrix[device_id][server_id]
            task_transmit_rate = data_length / transmit_time if transmit_time > 0 else 0
            task_transmit_rate = transmission_rate_of_device_and_servers[device_id][server_id]
            SINR_transmit_time = SINR_transmission_time_matrix[device_id][server_id]
            for other_device_id in range(device_number):
                # 只考虑与该设备发送到不同服务器的设备
                if decision_list_array[other_device_id][server_id] == 0:  # 这里检查其他设备是否发送到不同的服务器
                    new_interference += G_matrix_between_devices_and_servers[other_device_id][server_id] * transmission_power

                # 使用新的干扰值计算SINR
            new_SINR = signal_strength / (noise + new_interference)

            # 计算新的SINR传输速率，确保SINR大于0
            if new_SINR > 0:
                SINR2_transmission_rate = bandwidth * np.log2(1 + new_SINR)
            else:
                SINR2_transmission_rate = 0
        else:
            task_transmit_rate = 0  # 设备不发送数据
            SINR_task_transmit_rate = 0
            SINR_transmit_time = 0
            SINR2_transmission_rate = 0
        task_transmit_rates.append(task_transmit_rate)
        SINR_task_transmit_rates.append(SINR_task_transmit_rate)
        SINR_transmit_times.append(SINR_transmit_time)
        SINR2_task_transmit_rates.append(SINR2_transmission_rate)

    for index, task in enumerate(task_info_array):
        device_id, server_id, task_volumes = int(task[0]), int(task[-1]), task[1]

        # 获取对应的新的SINR传输速率
        SINR2_rate = SINR2_task_transmit_rates[index]

        # 如果SINR2传输速率大于0，则计算传输时间，否则设置为无穷大
        if SINR2_rate > 0:
            SINR2_transmit_time = task_volumes / SINR2_rate
        else:
            SINR2_transmit_time = float('inf')  # 使用float('inf')表示无穷大

        # 添加到SINR2传输时间列表
        SINR2_transmit_times.append(SINR2_transmit_time)
    # 将 task_transmit_rates 保存到 CSV 文件
    # 计算
    task_volume_value = task_volume_max / 8  # 计算除以8后的值
    # 格式化字符串
    file_name1 = f"total_capacity_10w_{server_number}fixedS_{device_number}D_{int(task_volume_value)}Vtest.csv"
    file_name2 = f"total_latency_10w_{server_number}fixedS_{device_number}D_{int(task_volume_value)}Vtest.csv"

    # csv_file_path = 'task_transmit_rate_10w_5fixedS_100D_1500VAB.csv'  # 注意根据实际路径修改
    # with open(csv_file_path, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['device_id', 'task_transmit_rate'])
    #     for i, rate in enumerate(task_transmit_rates):
    #         writer.writerow([i, rate])

    # csv_file_path = 'SINR_task_transmit_rate_10w_5fixedS_100D_1500VAB.csv'  # 注意根据实际路径修改
    # with open(csv_file_path, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['device_id', 'SINR_task_transmit_rate'])
    #     for i, rate in enumerate(SINR_task_transmit_rates):
    #         writer.writerow([i, rate])

    # 计算平均传输速率
    average_link_rate = np.mean([rate for rate in task_transmit_rates if rate > 0])
    average_SINR_link_rate = np.mean([rate for rate in SINR_task_transmit_rates if rate > 0])
    average_SINR_transmit_times = np.mean([rate for rate in SINR_transmit_times if rate > 0])
    # 计算task_transmit_rates的总和，并命名为capacity
    #capacity = sum(task_transmit_rates)
    total_task_volume = sum(task_volume)
    SINR_capacity = sum(SINR_task_transmit_rates)
    SINR_latency = total_task_volume/ SINR_capacity
    #total_latency = sum(SINR2_transmit_times)
    total_latency = sum(SINR_transmit_times)
    total_capacity = total_task_volume / total_latency
    # 保存平均传输速率为CSV文件
    # print("TASK_INFO%%%%%%%%%%%%%%%")
    # csv_file_path =  'average_capacity_10w_12fixedS_100D_1500VAB.csv'
    # with open(csv_file_path, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     #writer.writerow(['average_link_rate', 'capacity'])
    #     writer.writerow([average_link_rate, capacity])
    #数据集写入文件
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    random_number = random.randint(1000, 9999)
    output_file_path = os.path.join(output_directory, f"task_info_with_decision_{server_number}servers_{device_number}devices_{timestamp}_{random_number}.csv")
    with open(output_file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in task_info_array:
            writer.writerow(row)
    # #print(os.getcwd())


    # print("CAPACITY%%%%%%%%%%%%%%%")
    # # csv_file_path =  'SINR_capacity_10w_5fixedS_150D_1500V.csv'
    # # with open(csv_file_path, 'a', newline='') as file:
    # #     writer = csv.writer(file)
    # #     writer.writerow([average_SINR_link_rate, SINR_capacity])
    # csv_file_path =  file_name1
    # with open(csv_file_path, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([average_SINR_link_rate, total_capacity])
    # print("LATENCY%%%%%%%%%%%%%%%")
    # csv_file_path =  file_name2
    # with open(csv_file_path, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     #writer.writerow(['average_SINR_link_rate', 'SINR_capacity'])
    #     writer.writerow([average_SINR_transmit_times, total_latency])
    # # csv_file_path =  'SINR_latency_10w_5fixedS_150D_1500V.csv'
    # # with open(csv_file_path, 'a', newline='') as file:
    # #     writer = csv.writer(file)
    # #     #writer.writerow(['average_SINR_link_rate', 'SINR_capacity'])
    # #     writer.writerow([average_SINR_transmit_times, SINR_latency])

# def calculate_column_average(file_path, column_index):
#     try:
#         # 读取CSV文件
#         data = pd.read_csv(file_path)
#         # 计算指定列的平均值
#         average_value = data.iloc[:, column_index].mean()
#         return average_value
#     except Exception as e:
#         print(f"Error reading the file or calculating the average: {e}")
#         return None


if __name__ == "__main__":
    for i in range(90):
        test()
        print("times:",i)
    #计算，打印
    # file_path1 = f"{file_name1}"  # 使用全局变量file_name1构建文件路径
    # column_index = 1  # 我们想计算第二列的平均值
    # average_value = calculate_column_average(file_path1, column_index)
    # print(f"capacity: {average_value}")
    # file_path2 = f"{file_name2}"  # 使用全局变量file_name1构建文件路径
    # column_index = 1  # 我们想计算第二列的平均值
    # average_value = calculate_column_average(file_path2, column_index)
    # print(f"latency: {average_value}")
    #删除 设备号
    # df = pd.read_csv('task_info_with_decision_10w_5fixedS_100D_1500V1.csv')
    # df = df.drop(df.columns[0], axis=1)  # axis=1 表示操作列，0 表示第一列
    # df.to_csv('task_info_with_decision_10w_5fixedS_100D_1500V.csv', index=False)
