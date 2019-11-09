# -*- coding: UTF-8 -*-
import json

#random
import networkx as nx

import pandas as pd
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import logging
from logzero import setup_logger
logger=setup_logger(name='mylogger',logfile='main.log',level=logging.INFO)


#######################################greedy model#################################

def U(u, I, rel_u, X):
    '''
    The definition of Individual Utility in paper <Xiao et al. - 2017 - Fairness-Aware Group Recommendation with Pareto-Ef-converted>
    :param u: the user u
    :param I: a set of items
    :param rel_u: a vector of the relvences between user u and all items : rel(u,*)
    :param X: a m*1 vector ,where xj={0,1} denotes whether item j is recommended to group
    :return:Individual Utility
    '''
    '''

    max = np.max(rel_u)
    max=len(I)*max
    if max==0:
        return 0
    '''
    product = np.dot(rel_u, X)
    # aver = product
    # propor=np.sum(PR[u])/np.sum()
    return product


def SW(I, PR, X):
    '''
    Calculate the Social Welfare

    :param I: a set of items
    :param PR: the dict={user :rel(u,*)}
    :param X:
    :return:Social Welfare
    '''
    sum = 0
    for key, values in PR.items():
        sum += U(key, I, values, X)

    return sum / len(PR)


################################################################################

# 4 methodes to calculate the Fairness
def F(I, PR, X, name):
    '''
    :param g:the group of users
    :param I:the group of items
    :param PR:the matrix of Pr(i,j)
    :param X:xj={0,1} denotes whether item j is recommended to group
    :param name: the method's name
    :return:fairness
    '''
    if name == "Least Misery":
        l = []
        for key, values in PR.items():
            l.append(U(key, I, values, X))
        return np.min(l)
    if name == "Variance":
        l = []
        for key, values in PR.items():
            l.append(U(key, I, values, X))
        return (1 - np.var(l))
    if name == "Jain's Fairness":
        l = []
        ll = []
        for key, values in PR.items():
            l.append(U(key, I, values, X))
            ll.append(U(key, I, values, X) ** 2)
        return np.sum(l) / (len(l) * np.sum(ll))
    if name == "Min_Max Ratio":
        l = []
        for key, values in PR.items():
            l.append(U(key, I, values, X))
        return np.min(l) / np.max(l)


def init_reset(n, m):  # reset User , Item and X
    '''
    :param n: the number of users
    :param m: the number of items
    :return: vector User=[1,2,3,...n]
             vector Item=[1,2,3,...,m]
             Vector X=[0,0,0,...,0]
    '''
    User = np.zeros((n,), dtype=np.int)  # users
    Item = np.zeros((m,), dtype=np.int)  # items
    X = np.zeros(m, dtype=np.int)  # Xj={0,1} to denote whether item j is recommended to the group
    for i in range(n):
        User[i] = i
    for j in range(m):
        Item[j] = j
    return User, Item, X

##################################network###################################################

def topic_network(link_file, pr_file, nb_topic):
    # random pp associated each edge
    mnames = ['user_id', 'friend_id']  # genres means tags
    links = pd.read_csv(link_file, skiprows=[0], sep='\t', header=None, names=mnames, engine='python')
    with open(pr_file, 'r') as file:
        prob = json.load(file)
    a = list(links['user_id'])
    b = list(links['friend_id'])
    # l1=list(set(a).union(set(b)))
    edges1 = list(zip(a, b))

    G = nx.Graph()
    G.add_edges_from(edges1)
    n = G.number_of_edges()
    print(n)
    base_pr = np.random.randint(1, 5, nb_topic) / 10000
    for u, v in G.edges():
        edge1 = str((u, v))
        edge2 = str((v, u))
        # print(edge1)
        # print(edge2)

        try:
            pr1 = prob[str(edge1)]
            pr1 = np.array(pr1)
            if pr1 == np.zeros(nb_topic):
                pr1 = base_pr

        except:
            # pr1= np.random.dirichlet(np.ones(nb_topic), size=1)[0]
            pr1 = base_pr
        try:
            pr2 = prob[str(edge2)]
            pr2 = np.array(pr2)
            if pr2 == np.zeros(nb_topic):
                pr2 = base_pr
        except:
            # pr2 = np.random.dirichlet(np.ones(nb_topic), size=1)[0]
            pr2 = base_pr
        # pr = [np.random.dirichlet(np.ones(nb_topic), size=1)[0] for i in range(nb_topic)]
        G.add_edge(u, v, weight=[pr1, pr2])
        # df2 = pd.DataFrame([(u, v, pr)], columns=['user_id', 'friend_id', 'pr'])
        # df = df.append(df2)
    # df.to_csv('socialNetwork.txt', sep='\t', index=False)
    return G


#########################################################################################################################
def double_edge_label(G):
    edge_labels1 = dict([((u, v,), t["weight"][0]) for u, v, t in G.edges(data=True)])
    edge_labels2 = dict([((v, u,), t["weight"][1]) for u, v, t in G.edges(data=True)])
    edge_labels = edge_labels1.copy()
    edge_labels.update(edge_labels2)
    # print(edge_labels)
    return edge_labels


def proba(node1, node2, gamma, G):
    # edge_labels = dict([((u, v,), t["weight"]) for u, v, t in G.edges(data=True)])
    # print(edge_labels)
    edge_labels = double_edge_label(G)
    pp = edge_labels[(node1, node2)]
    # print("pp:",pp,"and the edge is ",node1,node2)
    # print("the proba is : ",np.dot(pp, gamma))
    # print("pp=",pp)
    # print("gamma=",gamma)
    return np.dot(pp, gamma)


#####################################################sampling_ap###############################
def activation_neib(w, neigb_w, actived, gamma, edge_labels):
    print('in the situation:', w, '\'s neigbours=', neigb_w)

    # print('actived=',actived)
    # print('neigb=',neigb_w)
    # in b But not in A
    # retD = list(set(listB).difference(set(listA)))
    rest = list(set(neigb_w).difference(set(actived)))
    print('rest=neigb-actived=', rest)
    new_sender = []
    if rest == []:
        print('rest has no element')
        return new_sender

    for ww in rest:
        pr = np.dot(edge_labels[tuple([w, ww])], gamma)
        print('pr=', pr)
        r = random.random()
        print('r=', r)
        if pr >= r:
            new_sender.append(ww)
        else:
            print(ww, 'is not actived')

    return new_sender


def sampling_one_time(G, list_S, gamma, edge_labels):
    actived = copy.deepcopy(list_S)
    new_sender = copy.deepcopy(list_S)

    flag = 0
    while new_sender != []:
        print()
        flag = flag + 1
        print('this is the', flag, 'action')

        senders = copy.deepcopy(new_sender)
        new_sender[:] = []
        # print('actived=', actived)
        # print('senders=', senders)
        # print('new_senders=', new_sender)

        print()
        for sender in senders:
            # print('sender=', sender)
            neib_sender = list(G[sender])
            # print('neib=', neib_sender)

            actived_s = activation_neib(sender, neib_sender, actived, gamma, edge_labels)
            print('we should add new actived=', actived_s)

            actived.extend(actived_s)

            new_sender.extend(actived_s)

            new_sender = list(set(new_sender))
            # print('now,total actived nodes:', actived)
        print('new_sender', new_sender)
    return actived


def calculate_sampling_ap(G, list_S, gamma, edge_labels, R):
    # R = 100
    total = []

    for i in range(R):
        activated = sampling_one_time(G, list_S, gamma, edge_labels)

        total.extend(activated)
        # print(total)
        print(i)
    #print(len(total))

    result = pd.value_counts(total)
    dic_result = dict(result)
    # print(dic_result)
    for key, value in dic_result.items():
        dic_result[key] = int(value) / R
    #print(dic_result)

    return dic_result


def new_ap(v, list_S, gamma, G, dict_simulate_ap):
    # this is the new method : simulation S/R
    if v in list_S:
        # print("ap=1,because",v,"is in the list_S")
        return 1
    else:
        C = G[v]  # find node v's all neigbors , type: dict
        # print("neigbors:",C)
        neib_nodes = list(C.keys())
        # print("nb_neib_nodes:",len(neib_nodes))
        produit = 1
        for i in neib_nodes:
            try:
                sampling_ap_i = dict_simulate_ap[i]
            except:
                sampling_ap_i = 0
            produit = produit * (1 - sampling_ap_i * proba(i, v, gamma, G))
            # print(produit)
        print(v, "\'s ap=", 1 - produit)

        return 1 - produit


def text_save(content, filename, mode='a'):
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()


def influence_spread(G, list_S, gamma):
    edge_labels = double_edge_label(G)
    dict_simulate_ap = calculate_sampling_ap(G, list_S, gamma, edge_labels, R=500)
    '''
        with open('dic_result.json', 'r') as file:
        dict_simulate_ap = json.load(file)

    '''
    text_save([gamma, dict_simulate_ap], 'dic_all_result.json', 'a')

    V = list(G.nodes)
    # print("all nodes in G are:",V)
    sum = 0
    for i in V:
        # print("we begin from the node:",i)
        # logger.info('ap(={}) begin'.format(i))
        sum = sum + new_ap(i, list_S, gamma, G, dict_simulate_ap)
        # logger.info('ap finish')
    print("influence sum=", sum)
    print("influence spread is ", sum / len(V))
    return sum / len(V)


def select(Graph, Sender, gamma, dict_gamma, Item, seeds, PR_relevence, X, alpha, beta, benefit):
    new_rest = list(set(Item) - set(seeds))
    nb_seeds = len(seeds)
    nb_new_seeds = nb_seeds + 1
    # max_benefit=benefit
    max_benefit = 0
    item = 0
    better_gamma = 0

    for i in new_rest:
        new_seeds = copy.deepcopy(seeds)
        new_seeds.append(i)
        # print("number of seeds=", len(seeds))
        # print("number of new_seeds=", len(new_seeds))
        new_X = copy.deepcopy(X)
        new_X[i] = 1
        sw = SW(new_seeds, PR_relevence, new_X)
        f = F(new_seeds, PR_relevence, new_X, "Least Misery")
        benefit1 = alpha * sw + beta * f
        logger.info("SW={}".format(sw))
        logger.info("f={}".format(f))
        logger.info("benefit1={}".format(benefit1))
        new_gamma = (gamma * len(seeds) + dict_gamma[i]) / (len(new_seeds))
        print("new_gamma=", new_gamma)

        logger.info('influence begin')
        influence = influence_spread(Graph, Sender, new_gamma)
        logger.info('influence finish')
        benefit2 = (1 - alpha - beta) * influence
        logger.info("benefit2={}".format(benefit2))

        total = benefit1 + benefit2

        if total >= max_benefit:
            max_benefit = total
            item = i
            better_gamma = new_gamma
    logger.info("benefit={}".format(max_benefit))

    X[item] = 1
    # print("X=",np.sum(X))
    return item, max_benefit, better_gamma


def greedy(Graph, Sender, gamma, dict_gamma, Item, k, dic_PR, X, alpha, beta):
    seeds = []  # Storage selected items
    list_m = []  # Storage selected item's benefit
    benefit = 0
    while len(seeds) < k:
        # print("list_item=", seeds)
        logger.info("list_item={}".format(seeds))
        item, new_benefit, new_gamma = select(Graph, Sender, gamma, dict_gamma, Item, seeds, dic_PR, X, alpha, beta,
                                              benefit)
        # print("we selsct ", item)
        logger.info("we selsct item={}".format(item))
        seeds.append(item)
        gamma = new_gamma
        benefit = new_benefit
        list_m.append(benefit)
    return seeds, list_m


def generate_PR_relevence(G, dict_gamma):
    '''
    :param G:a network associated weight
    :param dict_gamma: every item's vector gamma
    :return: every user's personal interests ：relevence(u,i)=relevence(u,t)*gamma_i
    '''
    V = list(G.nodes)
    edge_labels = double_edge_label(G)
    rel_user_item = {}
    for v in V:
        print()
        C = G[v]  # find node v's all neigbors , type: dict
        # print(v,"'s neigbors=:",C)

        list_pp = np.array([edge_labels[(v, i)] for i in C.keys()])
        # print('list_pp',list_pp)
        max_pp = list_pp.max(axis=0)
        # print('max_pp=',max_pp)
        # rel_u_t=random.uniform(max_pp, 1)#this is our relevence_user_topic
        rel_u_t = [random.uniform(pp, 0.99) for pp in max_pp]
        # print('rel_u_t=',rel_u_t)

        rel = []
        for key, gamma_i in dict_gamma.items():
            rel.append(np.dot(rel_u_t, dict_gamma[key]))
        # print("rel=",rel)
        # item_gamma = {i: np.random.dirichlet(np.ones(3), size=1)[0] for i in range(m)}
        rel_user_item[str(v)] = list(rel)
    logger.info("rel_user_item={}".format(rel_user_item))
    return rel_user_item
    # PR_rel = {i: np.array([round(random.uniform(0, 0.9), 2) for i in range(m)]) for i in range(n)}


def text_read(filename):
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

###################################关于计算pr##########################################
def userid_group_json(rating_filename):
    rnames = ['user_id', 'movie_id', 'rating', 'time']
    ratings = pd.read_csv(rating_filename,skiprows=[0], sep='\t', header=None, names=rnames,engine='python')
    #ratings= pd.read_csv('links_and_ratingtime_info.txt',skiprows=[0], sep='\t', header=None, names=rnames,engine='python')
    print(ratings[:10])


    d = dict()
    for _, row in ratings.iterrows():
        user_id, movie_id, rating,time = row
        d.setdefault(user_id, {}).update({movie_id: [rating,time]})
        #print(row)

    jsObj = json.dumps(d)
    fileObject = open('user_infogroup.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()
    return d
def nodelta_prop(links_filename,genres,infogroup):

    edges = text_read(links_filename)
    print(type(edges))

    prob = {}
    useful_edges_info={}
    #useless_edges = []
    number=0
    for edge in edges:
        number=number+1
        logger.info('the number is ={}'.format(number))
        edge = edge.split()

        logger.info('this edge ={}'.format(edge))

        u = edge[0]
        v = edge[1]
        print('user u is ', u, 'and user v is ', v)
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

        # print('info_u=',info_u)
        # print('info_v=', info_v)
        movie_u = list(info_u.keys())
        movie_v = list(info_v.keys())
        # logger.info('movie_u={}'.format(movie_u))
        # logger.info('movie_v={}'.format(movie_v))
        # 两个用户评论电影的并集
        total = list(set(movie_u).union(set(movie_v)))
        # logger.info('all movies as denominator={}'.format(total))
        n1 = len(movie_u)
        n2 = len(movie_v)
        # print('the denominator is ',n)
        # 两个用户评论电影的交集
        movie_for_u_v = list(set(movie_u).intersection(set(movie_v)))
        # logger.info('the intersection between u and v is={}'.format(movie_for_u_v))
        print(len(movie_for_u_v))
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
    fileObject = open('prob_nodelta.json', 'w')
    fileObject.write(jsObj1)
    fileObject.close()


    jsObj2 = json.dumps(useful_edges_info)
    fileObject = open(links_filename+'_interaction_info.json', 'w')
    fileObject.write(jsObj2)
    fileObject.close()
    return prob


################################关于prepare一些文件结果##########################################
def prepare_input():
    # based on link filter ratings，得到ratings after 2wash
    links = pd.read_csv('input/links_after_2wash.txt', sep='\t', engine='python')
    print(len(links))
    a = list(links['user_id'])
    b = list(links['friend_id'])

    users = list(set(a).union(set(b)))
    print(len(users))

    ratings = pd.read_csv('input/cleaning_ratings_timed', sep='\t', engine='python')
    df = ratings[ratings['userid'].isin(users)]
    print(df)
    df.to_csv('ratings_after_2wash.txt', sep='\t', index=False)

def wash3():
    #exist_items are items having topics, delete all others ratings after 3wash
    ratings = pd.read_csv('ratings_after_2wash.txt', sep='\t', engine='python')
    print('2=', len(ratings))

    with open('input/genres.json', 'r') as file:
        genres = json.load(file)
    exist_items = list(genres.keys())
    print(len(exist_items))

    ratings_after_3wash = ratings.drop(ratings.loc[(~ratings['movieid'].isin(exist_items))].index)
    print('3=', len(ratings_after_3wash))

    ratings_after_3wash.to_csv('ratings_after_3wash.txt', sep='\t', index=False)

def chose_item_sender(inter_info,ratings,k):
    #The process of picking one time: picking up the items waiting for the test, and all the senders that have rated these items
    exist_items=list(set(list(ratings['movieid'])))
    exist_items=[str(i) for i in exist_items]
    #print('exist_items',exist_items)
    flag=True
    while flag:
        info1 = random.sample(inter_info.items(), k)
        list_items = []
        for key, value in info1:
            print(value)
            item=random.choice(value)
            if item in exist_items:
                list_items.append(item)
        print(len(list_items))
        if len(list_items)==k:
            flag=False

    df = ratings[ratings['movieid'].isin(list_items)]
    #print(df)
    senders=[]
    groups=df.groupby('userid')
    for name,group in groups:
        #print(group)
        movies=group['movieid']
        #print(list(movies))
        if len(movies)==k:
            print(len(movies))
            senders.append(name)
    #print('senders=',senders)

    items=list(set(list(df['movieid'])))
    print('items=',items)

    return items,senders

def choose_gamma(m,k_size,wait_items,genres):
    #Select m items and make a one-to-one dictionary with gamma, including wait_items with detection
    item_gamma = {}
    records = {}

    # randomly select m-k item
    item_gamma0 = random.sample(genres.items(), m - k_size)

    # Make a dictionary one by one with item and gamma
    for i, (movie, g) in enumerate(item_gamma0):
        item_gamma[i] = np.array(g) / sum(np.array(g))
        records[i] = movie
    #Add wait_item to the dictionary
    for i in range(m - k_size, m):
        records[i] = wait_items[i % k_size]
    new_records = {value: key for key, value in records.items()}

    for item in wait_items:
        index = new_records[item]
        item_gamma[index] = np.array(genres[str(item)]) / sum(np.array(genres[str(item)]))
    #print(records)
    #print(item_gamma)


    #Convert the dictionary to a dataframe and save it under input
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
    DF.to_csv('output/Records_movie_id'+str(k), sep='\t', index=False)

    return item_gamma

def initial(k):
    #Do some preparatory work, get the items to be observed and the sender group, and record the number of the relationship
    with open('input/useful_edges_interaction_info.json', 'r') as file:
        inter_info = json.load(file)

    ratings = pd.read_csv('ratings_after_3wash.txt', sep='\t', engine='python')
    # print('all logs=',len(ratings))
    # print('all items=',len(set(list(ratings['movieid']))))

    with open('input/genres.json', 'r') as file:
        genres = json.load(file)
    exist_items = list(genres.keys())
    print(len(exist_items))

    ####################################### Select k items to be observed, and senders #################################
    items, users = chose_item_sender(inter_info, ratings, k)

    while len(items) != k or len(users) < 100:
        print('not ok')
        items, users = chose_item_sender(inter_info, ratings, k)

    senders = random.sample(users, 100)

    logger.info('wait_items ={}'.format(items))
    logger.info('senders ={}'.format(senders))
    text_save([items],'output/wait_info.txt')
    text_save([senders],'output/wait_info.txt')

    #######################################Delete the evaluation relationship between item and senders##############################################
    # With the items to be evaluated and senders, delete the rating between the senders and items - log
    ratings_after_cut = ratings.drop(
        ratings.loc[(ratings['userid'].isin(senders)) & (ratings['movieid'].isin(items))].index)
    print(len(ratings))
    print(len(ratings_after_cut))
    ratings_after_cut.to_csv(str(k)+'ratings_after_cut', sep='\t', index=False)

    return items,senders

def run_one_time(k):
    # step1: Determine the number of observation objects k-size is k
    
    

     #step2: Select k observation objects and senders, and cut off the rating record between them
    wait_items,list_S=initial(k)
    rating_filename = str(k) + 'ratings_after_cut'
    links_filename = 'links_after_2wash.txt'

    #step3: According to the rating and links information after the cut, calculate the propagation probability (topic-aware) of each side in the social graph, and generate the json file of the prob to store related information

    infogroup = userid_group_json(rating_filename)
    '''
    
    with open('user_infogroup.json', 'r') as file:
        infogroup = json.load(file)
    '''

    with open('input/genres.json', 'r') as file:
        genres = json.load(file)

    ###################Calculate the value of pr, write prob_nodelta, json##################################################
    nodelta_prop(links_filename,genres,infogroup)

    #print(prob)

    # Prepare the input of the greedy algorithm
    m = 10
    n = len(list_S)
    #####################################social network ###########################################
    G = topic_network(link_file=links_filename, pr_file='prob_nodelta.json', nb_topic=29)
    print('G is ok')
   
    ###################################Select m items and their corresponding gammas to form a dictionary##################################

    item_gamma = choose_gamma(m, k, wait_items, genres)
    print("gamma for every item is  ", item_gamma)

    ##########################得到rel（u,i）##############################################################
    PR_rel = generate_PR_relevence(G, item_gamma)
    # print("PR_rel=", PR_rel)
    print('rel is ok')

    ###########################初始化####################################################
    
    
    nb_topic = 29
    alpha = 0.5
    beta = 0.1
    User, Item, X = init_reset(n, m)
    print("User=", User)
    print("Item=", Item)
    print("X=", X)
    gamma = np.zeros(nb_topic)
    reco, benefit = greedy(G, list_S, gamma, item_gamma, Item, k, PR_rel, X, alpha, beta)
    print("our recommendations' list is :", reco)
    #print("there are ", len(reco), "items")
    print(benefit)
    x = [j for j in range(1, k + 1)]
    y = benefit
    plt.plot(x, y, linestyle="-", marker="^", linewidth=1)
    plt.xlabel("k-size items")
    plt.ylabel("benefit")
    plt.xticks(x)
    plt.savefig(str(k)+'simulation_result.png')
    
    #plt.show()
    return reco

if __name__ == '__main__':
    
    #step1：k-size
    m=10
    y=[]
    for k in range(1,6):
        items=[i for i in range(m-k,m)]
        recommendation=run_one_time(k)

        valide = 0
        for item in recommendation:
            if item in items:
                valide=valide+1
        print(valide)
        p=valide/k
        
        y.append(p)
    x = [j for j in range(1, 6)]
    print(y)
    plt.plot(x, y, linestyle="-", marker="^", linewidth=1)
    plt.xlabel("k-size items")
    plt.ylabel("precision validation")
    plt.xticks(x)
    plt.savefig('validation_result.png')
    
        
                
        
        













