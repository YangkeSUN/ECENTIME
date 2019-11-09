import numpy as np

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
#end def U

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
#end def SW

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
#end def F
