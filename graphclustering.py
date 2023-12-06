# -*-coding: utf-8 -*-
# @Time    : 2023/11/27 16:05
# @Author  : Liangliang
# @File    : graphclustering.py
# @Software: PyCharm


import os

os.system("pip install networkx")
os.system("pip install datasketch")

import math
import random
import datetime
import argparse
import time
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
from datasketch import MinHash


# 记录每个类簇中的信息
class Cluster:
    def __init__(self, count=0, vector=0, point=defaultdict(int)):
        self.count = count
        self.vector = vector
        self.point = point


def read_graph(args):
    '''
    输入有三份数据，第一份数据为图的边(左右端点的id)；
    第二份数据为图中节点的属性特征(第一列为节点的ID,第二列开始才是属性特征,大小为n*(d+1))
    第三份数据为图中所有节点的属性特征均值(无节点ID，大小:1*d)
    '''
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    data = pd.DataFrame().astype(object)
    for file in input_files:
        count += 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, os.path.join(path, file)))
        # 读取边结构数据
        data = pd.concat([data, pd.read_csv(os.path.join(path, file), sep=',', header=None)], axis=0)
    print("开始读取属性特征信息! {}".format(datetime.datetime.now()))
    # 读取属性特征信息
    path = args.data_input.split(',')[1]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    data_attr = pd.DataFrame().astype(object)
    for file in input_files:
        # 读取属性特征数据
        data_attr = pd.concat([data_attr, pd.read_csv(os.path.join(path, file), sep=',', header=None)], axis=0)
    # 读取属性特征均值信息, 这个数据只有一行
    path = args.data_input.split(',')[2]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    data_attr_avg = pd.read_csv(os.path.join(path, input_files[0]), sep=',', header=None)
    g = nx.Graph(feat=data_attr_avg.to_numpy().astype(np.float32))
    n, m = data_attr.shape
    # 添加节点
    print("开始添加节点. {}".format(datetime.datetime.now()))
    for i in range(n):
        if i % 1000000 == 0:
            print("一共有{}个节点, 当前添加{}个节点. {}".format(n, i, datetime.datetime.now()))
        g.add_node(data_attr.iloc[i, 0], feat=data_attr.iloc[i, 1::].values.astype(np.float32))
    print("完成节点添加. {}".format(datetime.datetime.now()))
    # 添加边
    g.add_edges_from(data.values.tolist())
    return g


# 执行minhash算法集合的交集大小
def estimate_intersection(set1, set2, n_set1, n_set2, args):
    m1, m2 = MinHash(num_perm=args.num_perm), MinHash(num_perm=args.num_perm)
    for d in set1:
        m1.update(d.encode('utf8'))
    for d in set2:
        m2.update(d.encode('utf8'))
    jaccard = m1.jaccard(m2)
    return jaccard * (n_set1 + n_set2) / (1 + jaccard)


# 执行聚类过程
def execution(args):
    g = read_graph(args)
    y_avg = g.graph["feat"]
    m = len(y_avg)
    N = g.number_of_nodes()
    result = []
    for i in range(args.k):
        p = Cluster(count=0, vector=np.zeros((1, m))[0], point=defaultdict(int))
        result.append(p)
    C_max = math.ceil(N / args.k)
    nodes = g.nodes()
    num = 0
    for node in nodes:
        num += 1
        if random.random() < 0.00001:
            print("一共有{}个节点,当前正在处理第{}个节点. {}".format(N, num, datetime.datetime.now()))
        tem_values = np.zeros((1, args.k))[0]
        neighbors = set(g.adj[node])
        n_neighbors = len(neighbors)
        for i in range(args.k):
            point_set = set(result[i].point.keys())
            n_point_set = len(point_set)
            # args.set_num这个参数数取值是多次实验试出来的结果
            if num < args.set_num:
                value1 = len(neighbors & point_set) * (1 - n_point_set / C_max)
            else:
                value1 = estimate_intersection(neighbors, point_set, n_neighbors, n_point_set, args) * \
                         (1 - n_point_set / C_max)
            value2 = np.prod(result[i].vector - y_avg) * np.prod(y_avg - g.nodes[node]["feat"])
            tem_values[i] = value1 * value2 / m
        # 找出分配的类簇
        j = np.argmax(tem_values, 0)
        result[j].vector = (result[j].vector * result[j].count + g.nodes[node]["feat"]) / (result[j].count + 1)
        result[j].count += 1
        result[j].point[node] += 1
    if args.is_fraction:
        print("开始remove非p-fraction节点. {}".format(datetime.datetime.now()))
        result = fraction(g, result, args)
    transfer_array(result, args)


# 过滤掉邻域节点处于同一类簇比例小于p%的节点
def fraction(g, result, args):
    for i in range(args.k):
        points = list(result[i].point.keys())
        for node in points:
            neighbors = set(g.adj[node])
            prob = len(neighbors & set(result[i].point.keys())) / len(neighbors)
            if prob < args.p:
                del result[i].point[node]
    return result


# 判断是否是二维列表
def is_list(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return True
    else:
        return False


# 把数据结果写入文件系统中
def write(data, args, count, file_num):
    # 注意在此业务中data是一个二维list
    # 数据的数量
    print("开始写入第{}个类簇第{}个文件数据. {}".format(count, file_num, datetime.datetime.now()))
    n = len(data)
    flag = is_list(data[0])
    line = ""
    if n > 0:
        start = time.time()
        with open(os.path.join(args.data_output, 'pred_{}_{}.csv'.format(count, file_num)), mode="a") as resultfile:
            if flag:
                for i in range(n):
                    line += ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
            else:
                line = ",".join(map(str, data)) + "\n"
                resultfile.write(line)
        cost = time.time() - start
        print("第{}个类簇第{}个数据文件已经写入完成,写入数据的行数{} 耗时:{}  {}".format(count, file_num,  n, cost,
                                                                                         datetime.datetime.now()))


# 转为数组类型以便于处理保存
def transfer_array(result, args):
    print("开始转换list为数组. {}".format(datetime.datetime.now()))
    for i in range(args.k):
        index = str(i)
        n = len(result[i].point)
        point_list = list(result[i].point.keys())
        result[i].point.clear()
        if n <= 0:
            continue
        else:
            n_files = math.ceil(n / args.file_nodes_num_max)
            for file_num in range(n_files):
                temp_list = point_list[file_num * args.file_nodes_num_max:min((file_num + 1) *
                                                                                    args.file_nodes_num_max, n)]
                n_list = len(temp_list)
                temp_array = np.zeros((n_list, 2)).astype(object)
                for j in range(n_list):
                    temp_array[j, 0] = str(temp_list[j])
                    temp_array[j, 1] = index
                write(temp_array.tolist(), args, i, file_num)


if __name__ == "__main__":
    start_time = time.time()
    # 设置参数
    args = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--p", help="同一个类簇内的邻域占比", type=float, default=0.7)
    parser.add_argument("--is_fraction", help="是否考虑使用邻域占比筛选", type=bool, default=False)
    parser.add_argument("--num_perm", help="随机排列函数的数目", type=int, default=64)
    parser.add_argument("--set_num", help="采用集合计算的最大节点数", type=int, default=205301)
    parser.add_argument("--file_nodes_num_max", help="单个输出文件最大的节点数目", type=int, default=150000)
    parser.add_argument("--k", help="类簇的数目", type=int, default=100)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    execution(args)
    end_time = time.time()
    print("节点划分耗时:{}".format(end_time - start_time))
