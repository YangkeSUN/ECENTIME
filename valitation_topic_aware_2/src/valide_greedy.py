import .valide_graph as graph

#def activation_neib(w, neigb_w, actived, gamma, edge_labels):
#def sampling_one_time(G, list_S, gamma, edge_labels):
#def calculate_sampling_ap(G, list_S, gamma, edge_labels, R):
#def proximate_sampling_ap(G, list_S, gamma, edge_labels, R):
#def influence_spread(G, list_S, gamma):
#def select(Graph, Sender, gamma, dict_gamma, Item, seeds, PR_relevence, X, alpha, beta, benefit):
#def greedy(Graph, Sender, gamma, dict_gamma, Item, k, dic_PR, X, alpha, beta):

#####################################################sampling_ap###############################
def activation_neib(w, neigb_w, actived, gamma, edge_labels):
    print('in the situation:', w, '\'s neigbours=', neigb_w)

    # print('actived=',actived)
    # print('neigb=',neigb_w)
    # in B but not in A
    # retD = list(set(listB).difference(set(listA)))
    rest = list(set(neigb_w).difference(set(actived)))
    # print('rest=neigb-actived=', rest) check
    new_sender = []
    if rest == []:
        print('rest has no element')
        return new_sender

    for ww in rest:
        pr = np.dot(edge_labels[tuple([w, ww])], gamma)
        # print('pr=', pr) check
        r = random.random()
        # print('r=', r) check
        if pr >= r:
            new_sender.append(ww)
        else:
            print(ww, 'is not actived')

    return new_sender
#end def activation_neib

def sampling_one_time(G, list_S, gamma, edge_labels):
    actived = copy.deepcopy(list_S)
    new_sender = copy.deepcopy(list_S)

    flag = 0
    while new_sender != []:
        # print()
        flag = flag + 1
        #print('this is the', flag, 'action')

        senders = copy.deepcopy(new_sender)
        new_sender[:] = []
        # print('actived=', actived)
        # print('senders=', senders)
        # print('new_senders=', new_sender)

        #print()
        for sender in senders:
            # print('sender=', sender)
            neib_sender = list(G[sender])
            # print('neib=', neib_sender)

            actived_s = activation_neib(sender, neib_sender, actived, gamma, edge_labels)
            #print('we should add new actived=', actived_s)

            actived.extend(actived_s)

            new_sender.extend(actived_s)

            new_sender = list(set(new_sender))
            # print('now,total actived nodes:', actived)
        #print('new_sender', new_sender)
    return actived
#end def sampling_one_time

def proximate_sampling_ap(G, list_S, gamma, edge_labels, R):
    # R = 1000
    total = []
    process_activated=[]#Save the activated points in each smapling and generate a probability of R activations
    for i in range(R):
        activated = sampling_one_time(G, list_S, gamma, edge_labels)
        process_activated.append(len(activated))

        total.extend(activated)
        #print(total)
        #print(i)
    #print(len(total))
    print('process_activated=',process_activated)

    V = list(G.nodes)
    #print("all nodes in G are:", V)
    avg_activated=np.mean(process_activated)
    prob_activated=avg_activated/len(V)

    result = pd.value_counts(total)
    dic_result = dict(result)
    # print(dic_result)
    for key, value in dic_result.items():
        dic_result[key] = int(value)
    print(dic_result)

return dic_result,prob_activated
#end def proximate_sampling_ap


def influence_spread(G, list_S, gamma):
    edge_labels = double_edge_label(G)
    dict_simulate_ap,prob_active = lib_graph_sap.proximate_sampling_ap(G, list_S, gamma, edge_labels, R=20)
    #text_save([gamma, dict_simulate_ap], 'dic_all_result.json', 'a')

    V = list(G.nodes)
    #print("all nodes in G are:",V)
    #print('average activated prob= ', prob_active)

    return prob_active
#end def influence_spread

def select(Graph, Sender, gamma, dict_gamma, Item, seeds, PR_relevence, X, alpha, beta, benefit):
    new_rest = list(set(Item) - set(seeds))
    nb_seeds = len(seeds)
    nb_new_seeds = nb_seeds + 1
    # max_benefit=benefit
    max_benefit = 0
    item = 0
    better_gamma = 0
    list_sw = []
    list_f = []
    list_influence = []
    benefit2 =[]
    benefit1=[]
    total =[]
    xy = 0
    j=0


    for i in new_rest:
        new_seeds = copy.deepcopy(seeds)
        new_seeds.append(i)
        new_gamma = (gamma * len(seeds) + dict_gamma[i]) / (len(new_seeds))
        # print("number of seeds=", len(seeds))
        # print("number of new_seeds=", len(new_seeds))
        new_X = copy.deepcopy(X)
        new_X[i] = 1
        sw = SW(new_seeds, PR_relevence, new_X)
        f = F(new_seeds, PR_relevence, new_X, "Least Misery")
        influence = influence_spread(Graph, Sender, new_gamma)
        list_sw.append(sw)
        list_f.append(f)
        list_influence.append(influence)
        xy += 1
    normalized_sw = preprocessing.normalize([list_sw])
    normalized_f = preprocessing.normalize([list_f])
    normalized_influence = preprocessing.normalize([list_influence])

    benefit1 = alpha * normalized_sw[j] + beta * normalized_f[j]
    #logger.info("benefit1[j]={}".format(benefit1))
    benefit2 = (1 - alpha - beta) * normalized_influence[j]
    total = benefit1 + benefit2
    max_benefit = max(total)
    #logger.info("benefit={}".format(max_benefit))
    X[item] = 1
    return item, max_benefit, better_gamma
#end def select

def greedy(Graph, Sender, gamma, dict_gamma, Item, k, dic_PR, X, alpha, beta):
    seeds = []  # Storage selected items
    list_m = []  # Storage selected item's benefit
    benefit = 0
    while len(seeds) < k:
        # print("list_item=", seeds)
        #logger.info("list_item={}".format(seeds))
        item, new_benefit, new_gamma = select(Graph, Sender, gamma, dict_gamma, Item, seeds, dic_PR, X, alpha, beta,
                                              benefit)
        # print("we selsct ", item)
        #logger.info("we selsct item={}".format(item))
        seeds.append(item)
        gamma = new_gamma
        benefit = new_benefit
        list_m.append(benefit)
    return seeds, list_m
#end def greedy
