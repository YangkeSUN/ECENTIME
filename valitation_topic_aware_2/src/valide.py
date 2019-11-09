# -*- coding: UTF-8 -*-
import json
import copy
import random
import argparse
import logging
from collections import Counter
#from functools import reduce

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from logzero import setup_logger
from sklearn import preprocessing

import .valide_math as formula
import .valide_graph as ns_graph
import .valide_greedy as ns_greedy
import .valide_util as ns_util

logger=setup_logger(name='mylogger',logfile='main.log',level=logging.INFO)

#def generate_PR_relevence(G, dict_gamma): double_edge_label logger.info
#def userid_group_json(rating_filename): pd.read_csv 
#def nodelta_prop(links_filename,genres,infogroup):
#def chose_item_sender(inter_info,ratings,k):
#def choose_gamma(m,k_size,wait_items,genres):
#def initial(k):
#def run_one_time(config, k,alpha,beta):
#def validation(config, nb_k,alpha,beta):
#def load_config(config_path):
#def main():

def generate_PR_relevence(G, dict_gamma):
    '''
    :param G:a network associated weight
    :param dict_gamma: every item's vector gamma
    :return: every user's personal interests ：relevence(u,i)=relevence(u,t)*gamma_i
    '''
    V = list(G.nodes)
    edge_labels = ns_graph.double_edge_label(G)
    rel_user_item = {}
    for v in V:
        C = G[v]  # find node v's all neigbors , type: dict
        list_pp = np.array([edge_labels[(v, i)] for i in C.keys()])
        max_pp = list_pp.max(axis=0)
        rel_u_t = [random.uniform(pp, 0.99) for pp in max_pp]
        rel = []
        for key, gamma_i in dict_gamma.items():
            rel.append(np.dot(rel_u_t, dict_gamma[key]))
        rel_user_item[str(v)] = list(rel)
    logger.info("rel_user_item={}".format(rel_user_item))
    return rel_user_item
#end def generate_PR_relevence 


###################################关于计算pr##########################################
def userid_group_json(k_ratings):
    d = dict()
    for _, row in k_ratings.iterrows():
        user_id, movie_id, rating,time = row
        d.setdefault(user_id, {}).update({movie_id: [rating,time]})
    jsObj = json.dumps(d)
    with ns_util.smart_open('user_infogroup.json.gz', 'w') as fileObject
        fileObject.write(jsObj)
    return d
#end def userid_group_json

def nodelta_prop(links_filename, genres, infogroup):
    edges = text_read(links_filename)
    prob = {}
    useful_edges_info={}
    number=0
    for edge in edges:
        number=number+1
        logger.info('the number is ={}'.format(number))
        edge = edge.split()
        logger.info('this edge ={}'.format(edge))
        u = edge[0]
        v = edge[1]
        try:
            info_u = infogroup[u]
        except:
            # logger.info('user u={} didnt rate any movie'.format(u))
            #useless_edges.append(edge)
            continue
        try:
            info_v = infogroup[v]
        except:
            #useless_edges.append(edge)
            #logger.info('user v={} didnt rate any movie'.format(v))
            continue
        movie_u = list(info_u.keys())
        movie_v = list(info_v.keys())
        # 两个用户评论电影的并集
        total = list(set(movie_u).union(set(movie_v)))
        # logger.info('all movies as denominator={}'.format(total))
        n1 = len(movie_u)
        n2 = len(movie_v)
        # print('the denominator is ',n)
        # 两个用户评论电影的交集
        movie_for_u_v = list(set(movie_u).intersection(set(movie_v)))
        # logger.info('the intersection between u and v is={}'.format(movie_for_u_v))
        if len(movie_for_u_v) == 0:
            #useless_edges.append(edge)
            continue
        list_id_u_v = []
        list_id_v_u = []
        for i in movie_for_u_v:
            # print('we see the movie:',i)
            rating_u = info_u[i][0]
            rating_v = info_v[i][0]
            time_u = info_u[i][1]
            time_v = info_v[i][1]
            delta_u_v = abs(rating_u - rating_v)
            if rating_u <= rating_v and time_u <= time_v:
                list_id_u_v.append(i)
            elif rating_v <= rating_u and time_v <= time_u:
                list_id_v_u.append(i)
        logger.info('u to v={}'.format(list_id_u_v))
        logger.info('v to u={}'.format(list_id_v_u))
        #logger.info('we begin to translate topic')
        sum1 = np.zeros(29)
        for movie in list_id_u_v:
            try:
                data1 = np.array(genres[movie])
            except:
                # logger.info('we dont have this movie={}'.format(movie))
                continue
            sum1 = sum1 + data1

        prob1 = sum1 / n1
        sum2 = np.zeros(29)
        for movie2 in list_id_v_u:
            try:
                data2 = np.array(genres[movie2])
            except:
                # logger.info('we dont have this movie={}'.format(movie2))
                continue
            sum2 = sum2 + data2

        prob2 = sum2 / n2
        # logger.info('sum1={}'.format(sum1))
        # logger.info('sum2={}'.format(sum2))
        logger.info('prob1={}'.format(prob1))
        logger.info('prob2={}'.format(prob2))

        useful_edges_info[str((u, v))]=movie_for_u_v
        prob[str((u, v))] = list(prob1)
        prob[str((v, u))] = list(prob2)

    # text_save(useless_edges, 'useless_edges.txt')

    jsObj1 = json.dumps(prob)
    fileObject = ns_util_smart_open('prob_nodelta.json', 'w')
    fileObject.write(jsObj1)
    fileObject.close()

    jsObj2 = json.dumps(useful_edges_info)
    fileObject = ns_util_smart_open(links_filename+'_interaction_info.json', 'w')
    fileObject.write(jsObj2)
    fileObject.close()

    return prob
#end def nodelta_prop

def chose_item_sender(inter_info,ratings,k):
    # The process of picking one time: picking the items waiting to be detected, and the senders that have all the ratings of these items
    exist_items=list(set(list(ratings['movieid'])))
    exist_items=[str(i) for i in exist_items]
    #print('exist_items',exist_items)
    flag=True
    while flag:
        info1 = random.sample(inter_info.items(), k)
        list_items = []
        for key, value in info1:
            item=random.choice(value)
            if item in exist_items:
                list_items.append(item)
        if len(list_items)==k:
            flag=False

    df = ratings[ratings['movieid'].isin(list_items)]
    senders=[]
    groups=df.groupby('userid')
    for name,group in groups:
        movies=group['movieid']
        if len(movies)==k:
            senders.append(name)
    items=list(set(list(df['movieid'])))
    return items,senders
#end def chose_item_sender


def choose_gamma(m,k_size,wait_items,genres):
    #挑选m个items，并与gamma做成一一对应的字典，其中包含带检测的wait_items
    item_gamma = {}
    records = {}

    # 随机选取m-k个item
    item_gamma0 = random.sample(genres.items(), m - k_size)

    # 将item与gamma一一对应做成字典
    for i, (movie, g) in enumerate(item_gamma0):
        item_gamma[i] = np.array(g) / sum(np.array(g))
        records[i] = movie
    #将wait_item加入到字典中
    for i in range(m - k_size, m):
        records[i] = wait_items[i % k_size]
    new_records = {value: key for key, value in records.items()}

    for item in wait_items:
        index = new_records[item]
        item_gamma[index] = np.array(genres[str(item)]) / sum(np.array(genres[str(item)]))

    #将字典转化为dataframe并保存在input下
    df = pd.DataFrame.from_dict(records, orient='index')
    df['item_id'] = df.index
    df = df.rename(columns={0: 'movie_id'})
    df1 = df[['item_id', 'movie_id']]
    # print(df1)
    item_gamma2={key:str(list(value)) for key,value in item_gamma.items()}
    dff = pd.DataFrame.from_dict(item_gamma2, orient='index')
    dff['item_id'] = dff.index
    dff = dff.rename(columns={0: 'vec_genres'})
    dff1 = dff[['item_id', 'vec_genres']]

    DF=pd.merge(df1,dff1)
    DF.to_csv('output/Records_movie_id'+str(k_size), sep='\t', index=False)

    return item_gamma
#end def choose_gamma

def initial(genres, inter_info, ratings, k):
    #做一些准备工作，得到待观察的items以及sender group，并记录号cut掉关系的ratings

    # print('all logs=',len(ratings))
    # print('all items=',len(set(list(ratings['movieid']))))

    exist_items = list(genres.keys())
    #print(len(exist_items))

    ####################################### 选择 k 个 待观察的items，以及 senders#################################
    items, users = chose_item_sender(inter_info, ratings, k)

    while len(items) != k or len(users) < 100:
        print('not ok')
        items, users = chose_item_sender(inter_info, ratings, k)

    senders = random.sample(users, 100)

    logger.info('wait_items ={}'.format(items))
    logger.info('senders ={}'.format(senders))
    text_save([items],'output/wait_info.txt')
    text_save([senders],'output/wait_info.txt')

    #######################################删除item与senders间的评价关系##########################################
    # 有了待评价的items以及senders之后，删除senders和items之间的rating——log。
    ratings_after_cut = ratings.drop(
        ratings.loc[(ratings['userid'].isin(senders)) & (ratings['movieid'].isin(items))].index)
    #print(len(ratings))
    #print(len(ratings_after_cut))
    ratings_after_cut.to_csv(str(k)+'ratings_after_cut', sep='\t', index=False)

    return items,senders
#end def initial

def run_one_time(config, genres, inter_info, ratings, G, k, alpha, beta, links_filename):
    # step1：确定观察对象个数k-size 为k

    #step2：选择k个观察对象与senders，并切断他们之间的rating记录
    wait_items,list_S=initial(genres, inter_info, ratings, k)

    #step3：根据cut后的rating以及links信息，计算社交图里，每条边的传播概率（topic-aware），生成prob的json文件储存相关信息
    k_rating_after_cut_filepath = str(k) + 'ratings_after_cut'
    rnames = ['user_id', 'movie_id', 'rating', 'time']
    k_ratings = pd.read_csv(k_rating_after_cut_filepath, skiprows=[0], sep='\t', header=None, names=rnames,engine='python')
    infogroup = userid_group_json(k_ratings)

    ###################计算出pr的值，写入prob_nodelta,json##################################################
    nodelta_prop(links_filename,genres,infogroup)

    # 准备greedy算法的input
    n = len(list_S)
    m = 10*len(list_S)
    #####################################建立social network 社交图###########################################
    #print('G is ok')

    ###################################挑选m 个 item 及其对应的 gamma,组成字典##################################
    item_gamma = choose_gamma(m, k, wait_items, genres)
    #print("gamma for every item is  ", item_gamma)

    ##########################得到rel（u,i）##############################################################
    PR_rel = generate_PR_relevence(G, item_gamma)
    # print("PR_rel=", PR_rel)
    #print('rel is ok')

    ###########################初始化####################################################
    nb_topic = 29
    User, Item, X = init_reset(n, m)
    #print("User=", User)
    #print("Item=", Item)
    #print("X=", X)
    gamma = np.zeros(nb_topic)
    reco, benefit = ns_greedy.greedy(G, list_S, gamma, item_gamma, Item, k, PR_rel, X, alpha, beta)
    #print("our recommendations' list is :", reco)
    #print("there are ", len(reco), "items")
    print(benefit)
    x = [j for j in range(1, k + 1)]
    y = benefit
    return benefit
#end def run_one_time

def top_n_most_seen_film(ratings, n):
    return Counter(list(ratings['movie_id'])).most_common(n)

def top_n_most_seen_film_within_group(group, ratings, n):
    return Counter(map(lambda x: x['movie_id'], filter((lambda x: x['user_id'] in group), ratings.itertuples(index=False)))).most_common(n)

#def create_k_core_group(links):


def validation(config, nb_k,m,x,alpha,beta):
    # step1：确定观察对象个数k-size
    genres = ns_util.smart_json_load(config['input']['genres'], 'r')
    prob = ns_util.smart_json_load(config['input']['prob_nodelta'], 'r')
    inter_info = ns_util_smart_open(config['input']['useful_edges_interaction_info'], 'r')
    links = pd.read_csv(config['input']['links_after_2wash_path'], skiprows=[0], sep='\t', header=None, names=mnames, engine='python')
    ratings = pd.read_csv(config['input']['ratings_after_3wash'], sep='\t', engine='python')
    G = ns_graph.topic_network(links, prob, nb_topic=29)
    
    y = []
    for k in range(1, nb_k):
        items = [i for i in range(m - k, m)]
        recommendation = run_one_time(config, genres, inter_info, ratings, G, k, alpha, beta, config['input']['links_after_2wash_path'])
        #valide = reduce((lambda count, x: count+1), filter(lambda x: x in item), recomendation)
        valide = 0
        for item in recommendation:
            if item in items:
                valide = valide + 1
        #print(valide)
        p = valide / k
        y.append(p)
    x = [j for j in range(1, 6)]
    print(111111111)
    print(y)
    print(22222222)
    plt.plot(x, y, linestyle="-", marker="^", linewidth=1)
    plt.xlabel("k-size items")
    plt.ylabel("precision validation")
    plt.xticks(x)
    plt.savefig('validation_result.png')
#end def validation

def load_config(config_path):
    with ns_util_smart_open(config_path) as config_file:
        return json.load(config_file)
#end def load_config

def main():
    parser = argparse.ArgumentParser(description='recommend films to groups of user')
    parser.add_argument('--config', nargs=1, metavar='FILE', type=str, required=False, default='config.json'
            , help='configuration file containing input file locations and default parameters (default=config.json)')
    parser.add_argument('--nb_k', nargs=1, metavar='K', type=float, required=False, help='number of links in group of user')
    parser.add_argument('--alpha', nargs=1, metavar='A', type=float, required=False, help='Social welfare coeficient (between 0.0 and 1.0)')
    parser.add_argument('--beta', nargs=1, metavar='B', type=float, required=False, help='faireness coeficient (between 0.0 and 1.0)')
    parser.add_argument('--m', nargs=1, metavar='M', type=float, required=False, help="number of users in a group")
    parser.add_argument('--x', nargs=1, metavar='X', type=float, required=False, help="minimum number of links from one user to other users in a group")
    args = parser.parse_args()
    
    config = load_config(args.config[0])
    if (args.alpha != None):
        config["alpha"] = args.alpha[0]
    if (args.beta != None):
        config["beta"] = args.beta[0]
    if (args.m != None):
        config["m"] = args.m[0]
    if (args.x != None):
        config["x"] = args.x[0]
    if (args.nb_k != None):
        config["nb_k"] = args.nb_k[0]
    validation(config, config["nb_k"], config["m"], config["x"], config["alpha"], config["beta"])
#end def main

if __name__ == '__main__':
    main()
