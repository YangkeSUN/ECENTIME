import numpy as np
import pandas as pd
import networkx as nx

#def init_reset(n, m):  # reset User , Item and X
#def topic_network(link_file, pr_file, nb_topic):
#def double_edge_label(G):
#def proba(node1, node2, gamma, G):

##################################network###################################################

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
#end def init_reset

def topic_network(links, prob, nb_topic):
    # random pp associated each edge
    mnames = ['user_id', 'friend_id']  # genres means tags
    a = list(links['user_id'])
    b = list(links['friend_id'])
    edges1 = list(zip(a, b))
    G = nx.Graph()
    G.add_edges_from(edges1)
    n = G.number_of_edges()
    base_pr = np.random.randint(1, 5, nb_topic) / 10000
    for u, v in G.edges():
        edge1 = str((u, v))
        edge2 = str((v, u))
        try:
            pr1 = prob[str(edge1)]
            pr1 = np.array(pr1)
            if pr1 == np.zeros(nb_topic):
                pr1 = base_pr
        except:
            pr1 = base_pr
        try:
            pr2 = prob[str(edge2)]
            pr2 = np.array(pr2)
            if pr2 == np.zeros(nb_topic):
                pr2 = base_pr
        except:
            pr2 = base_pr
        G.add_edge(u, v, weight=[pr1, pr2])
    return G
#end def topic_network

#########################################################################################################################

def double_edge_label(G):
    edge_labels1 = dict([((u, v,), t["weight"][0]) for u, v, t in G.edges(data=True)])
    edge_labels2 = dict([((v, u,), t["weight"][1]) for u, v, t in G.edges(data=True)])
    edge_labels = edge_labels1.copy()
    edge_labels.update(edge_labels2)
    # print(edge_labels)
    return edge_labels
#end def double_edge_label


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
#end def proba

