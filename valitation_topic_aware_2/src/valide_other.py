import json

import pandas as pd

#def wash3():
#def prepare_input():

def wash3():
    # Exist_items is an item with a topic record, delete the rating-log not in this item list, get ratings after 3wash
    ratings = pd.read_csv('ratings_after_2wash.txt', sep='\t', engine='python')
    #print('2=', len(ratings))

    with open('input/genres.json', 'r') as file:
        genres = json.load(file)
    exist_items = list(genres.keys())
    #print(len(exist_items))

    # Exist_items is an item with a topic record, delete rating-log not in this item list
    ratings_after_3wash = ratings.drop(ratings.loc[(~ratings['movieid'].isin(exist_items))].index)
    #print('3=', len(ratings_after_3wash))

    ratings_after_3wash.to_csv('ratings_after_3wash.txt', sep='\t', index=False)
#end def wash3

################################about prepare file result##########################################
def prepare_input():
    # According to the information of links, clean the ratings information and get ratings after 2wash
    links = pd.read_csv('input/links_after_2wash.txt', sep='\t', engine='python')
    #print(len(links))
    a = list(links['user_id'])
    b = list(links['friend_id'])

    users = list(set(a).union(set(b)))
    #print(len(users))

    ratings = pd.read_csv('input/cleaning_ratings_timed', sep='\t', engine='python')
    df = ratings[ratings['userid'].isin(users)]
    #print(df)
    df.to_csv('ratings_after_2wash.txt', sep='\t', index=False)
#end def prepare_input


