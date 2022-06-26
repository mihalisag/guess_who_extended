import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import pickle
from PIL import Image

pre_file = open('att_to_pre', 'rb')
att_to_pre = pickle.load(pre_file)
pre_file.close()

sent_file = open('att_to_sent', 'rb')
att_to_sent = pickle.load(sent_file)
sent_file.close()

imsize = lambda im: [len(im[0]), len(im)]
binary_entropy = lambda p: -p*math.log2(p) - (1-p)*math.log2(1-p) if (p>0 and p!=1) else 0

df = pd.read_csv('faces.csv')
df = df[df.columns.difference(['Unnamed: 0'])]

def dict_print(d):
    '''
        Print question dictionary without curly braces
    '''
    if type(d) == dict:
        for key, value in d.items():
            print(key.replace(' = ', ''), ':', round(value, 3), ' bits')


def cross_plot(im):
    '''
        Plots an X
    '''
    [x_max, y_max] = imsize(im)
    
    a = [0, x_max]
    b = [0, y_max]
    
    plt.ion()
    plt.axis('off')
    plt.plot(a, b, 'r', a, b[::-1], 'r', linewidth=2.5)
    
    return plt


def name_adder(df):
    '''
        Add names to DataFrame
    '''
    names_df = pd.read_csv('names.csv')
    female_names = list(names_df[names_df['male'] == 0]['Name'])
    male_names = list(names_df[names_df['male'] == 1]['Name'])

    for i in df.index:
        if df.at[i, 'male'] == 1:
            df.at[i, 'name'] = male_names.pop(1)
        else:
            df.at[i, 'name'] = female_names.pop(1)
            
    return df


def table_plotter(images, table_size, name_list, crossed_list=[]):
    '''
        Plots the game table of faces with names and puts crosses
        on the images included in the crossed_list
    ''' 
    rows = cols = math.ceil(math.sqrt(table_size))
    [x_pixels, y_pixels] = imsize(images[1])
    x_pos = x_pixels//4
    y_pos = y_pixels + 30 

    for i in range(1, table_size+1):
        name = name_list[i-1]
        plt.subplot(rows, cols, i)
        plt.axis('off')
        plt.text(x_pos, y_pos, name, bbox=dict(facecolor='black', alpha=1), color='white', fontsize='small')
        if i in crossed_list:
            cross_plot(images[i])
        plt.imshow(images[i])
         
    plt.ion()
    plt.show()

    
def question_maker(att, att_to_pre, att_to_sent):
    '''
        Makes a question from the dictionaries
    '''
    action = att_to_pre[att]
    if action == 'is':
        first_word = action
        action = ''
    else:
        first_word = 'does'
        action += ' '
    
    return first_word.title() + ' the person ' + action + att_to_sent[att] + '? = '


def answer_eliminator(df, att, att_to_pre, att_to_sent):
    '''
        Eliminates faces depending on answer **and if all column's elements are zero**
        Output: DataFrame and indices of eliminated faces 
    '''  
    df = df[df.columns.difference(['name'])]    
    orig_indices = list(df.index)
    att_check = int(input('Question: ' + question_maker(att, att_to_pre, att_to_sent)))
    df = df[df[att] == att_check]
    elim_ind = [ind for ind in orig_indices if ind not in df.index]
    rem_att = pd.Series(df.any())
    rem_att = list(rem_att[rem_att == True].index)[:-1]
    
    df = df[rem_att]
    
    return df, elim_ind


def optimal_attribute_finder(df):
    '''
        Finds attribute with closest to half ones
    '''
    m = 1
    total = len(df)
    att_ones = df.sum(numeric_only=True)
    att_list = []
    att_to_prob = dict()
    att_to_diff = dict()
    pref_att = '' 
    
    for att in att_ones.keys():
        ones = att_ones[att]
        prob = ones/total
        diff = abs(prob - 0.5)
        att_to_prob[att] = prob
        att_to_diff[att] = diff
        
        if ones != total and diff < m:    
            att_list.append(att)
            pref_att = att
            m = diff
            
    return pref_att, att_to_prob, att_to_diff


def best_five(att_to_pre, att_to_sent, df):
    '''
        Prints five best questions to ask
    '''
    i = 0
    att_list = []
    pref_att, att_to_prob, att_to_diff = optimal_attribute_finder(df)
    diff_to_att = {}
    
    for key, value in att_to_diff.items():
        diff_to_att.setdefault(value, []).append(key)
    
    for diff in sorted(diff_to_att):
        for att in diff_to_att[diff]:
            att_list.append(att)
            if len(att_list) == 5:
                question_dict = {question_maker(att, att_to_pre, att_to_sent) : 
                                 binary_entropy(att_to_prob[att]) for att in att_list}

                return question_dict


def game(df):
    '''
        Main game
    '''
    NUM = int(input('Select number of faces = ')) 
    
    df = df.iloc[range(NUM)]
    df.index = range(1, NUM+1)
    
    table_size = len(df) 
    images = {i : img.imread('./faces/'+df.loc[i]['filename']) for i in range(1, table_size + 1)}              
    first_image_index = list(images.keys())[0]

    image_filenames = df['filename'] 
    df = name_adder(df)
    df = df[df.columns.difference(['filename'])]
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print('Please wait...')

    exist_att = []
    crossed_list = []
    key = True
    name_list = list(df['name'])
    new_columns = list(df.columns)
    
    plt.axis('off')
    table_plotter(images, table_size, name_list, crossed_list)

    while key:  
        try:
            att, att_to_prob, att_to_diff = optimal_attribute_finder(df)
        except IndexError:
            return 'Wrong choice'
        if att not in exist_att:
            exist_att.append(att)
            df, elim_ind = answer_eliminator(df, att, att_to_pre, att_to_sent)
            print(2*'------------------------')
            print('Entropy of each question')
            print(2*'------------------------')
            dict_print(best_five(att_to_pre, att_to_sent, df))
            print(2*'------------------------')

            if len(df.index) == 1:
                final_image = image_filenames[df.index[0]]
                im = Image.open('./faces/' + final_image)
                im.show()
                print('The person you picked is ' + name_list[df.index[0]-1])
                return df
            
            crossed_list += elim_ind
            table_plotter(images, table_size, name_list, crossed_list)
            new_columns = list(df.columns)

    return df
    
game(df)
