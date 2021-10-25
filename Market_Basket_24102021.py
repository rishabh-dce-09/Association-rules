########################################################################################################################################
#                                                      *** MARKET BASKET ANALYSIS ***                                                  #
#                                                                                                                                      #
# Author(s): Rishabh Kumar                                                                                                             #
# Creation Date: 19/10/2021                                                                                                            #
# Description: hich products will an Instacart consumer purchase again?                                                                #
# Weblink: https://www.kaggle.com/c/instacart-market-basket-analysis                                                                   #
#                                                                                                                                      #
# Other important weblinks are:                                                                                                        #                                                                                                                                    #
# (i) https://www.kaggle.com/foolwuilin/market-basket-analysis-with-apriori                                                            #                                                                                                                                   #
# (ii) https://pbpython.com/market-basket-analysis.html                                                                                #                                                                                                                                  #
# (iii) https://medium.com/@jihargifari/how-to-perform-market-basket-analysis-in-python-bd00b745b106                                   #                                                                                                                                 #
# (iv) https://medium.com/swlh/a-tutorial-about-market-basket-analysis-in-python-predictive-hacks-497dc6e06b27                         #                                                                                                                                #
# (v) https://nabeel-oz.github.io/qlik-py-tools/docs/Association-Rules.html                                                            #                                                                                                                               #
# (vi) https://towardsdatascience.com/the-apriori-algorithm-5da3db9aea95                                                               #
#                                                                                                                                      #
#                                                                                                                                      #
#                                                             *** EDIT LOG ***                                                         #
# Last Edit Date: xx/xx/20xx                                                                                                           #
# Last Edit by: Rishabh Kumar                                                                                                          #
#                                                                                                                                      #
#        EDIT ID     |     EDITED BY      |     EDIT DATE     |      EDIT DESCRIPTION                                                  #  
#        <xxx001>          <xxx>                xx/xx/20xx           <To be filed>                                                     # 
#                                                                                                                                      #
########################################################################################################################################

########################################################################################################################################
#                                                             *** Steps Followed ***                                                   #
#                                                                                                                                      #
# Step 1: Import required libraries and read the Data                                                                                  #
# Step 2: Basic Data Exploration                                                                                                       #
# Step 3: Exploratory Data Analysis                                                                                                    #
# Step 4: Data Preparation                                                                                                             #
# Step 5: Build Association Rules                                                                                                      #
# Step 6: Analysis of Association Rules                                                                                                #
# Step 7: Output the Data                                                                                                              #
#                                                                                                                                      #
########################################################################################################################################


########################################################### START OF CODE ##############################################################


######################################################################################################################################## 
#                                                                                                                                      #
###################               STEP 1: IMPORT THE REQUIRED LIBRARIES AND READ THE DATA                  ############################# 
#                                                                                                                                      #
########################################################################################################################################

# ------ Import required Libraries ------

import pandas as pd
import numpy as np

from datetime import date
import time as time

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
# from efficient_apriori import apriori

pd.set_option('display.max_columns', None)

# ------ Create required variables ------

input_path = 'C:/Rishabh/Dropbox/2. Self Learning/1. Data Science and Analytics/2. Project(s)/7. Market Basket Analysis/2. Grocery/2. Raw Data'
# input_file = ''

output_path = 'C:/Rishabh/Dropbox/2. Self Learning/1. Data Science and Analytics/2. Project(s)/7. Market Basket Analysis/2. Grocery/3. Output Data'

input_file = 'Groceries data'

read_data = 'YES'
# read_data = 'NO'

# output_dataset = 'YES'
output_dataset = 'NO'

article_desc_col = 'itemDescription'

today = date.today()
date_stamp = today.strftime('%d_%m_%Y') # Format: dd_mm_yyyy

fig_size = (9, 6)
fig_dpi = 200

# plt.style.use('default')

# plt.style.use('classic')
# plt.style.use('seaborn-dark-palette')
plt.style.use('dark_background')

# ------ Read the data ------

start_time = time.time()

if read_data == 'YES':
    
    df = pd.read_csv(f'{input_path}/{input_file}.csv', parse_dates=['Date'])
    time_to_run = time.time() - start_time
    print(f'Time to read the data: {round(time_to_run,2)} second(s)')

else:
    print(f'Input dta file ({input_file} not read')

######################################################################################################################################## 
#                                                                                                                                      #
###################                            STEP 2: BASIC DATA EXPLORATION                              ############################# 
#                                                                                                                                      #
########################################################################################################################################

# df.head()
print(f'\nTop 5 rows of the {input_file} daatset are as follows: \n{df.head()}') # Check

# df.info()
print(f'\nInformation of {input_file} dataset is as follows: \n')
df.info() # Check

# df.shape() 
print(f'\nThe shape aka (rows,columns) of the {input_file} dataset is as follows: {df.shape}') # Check

# df.describe() # Check
print(f'\nThe description of the {input_file} dataset is as follows: \n {df.describe()}') # Check

# df.dtypes # Check
print(f'\nThe data types of the {input_file} dataset are as follows: \n{df.dtypes}') # Check

# len(df.columns) # Check
print(f'\nThe number of columns in the {input_file} dataset are: {len(df.columns)}') # Check

df_main = df.copy()

######################################################################################################################################## 
#                                                                                                                                      #
###################                           STEP 3: EXPLORATORY DATA ANALYSIS                            ############################# 
#                                                                                                                                      #
########################################################################################################################################

df_main[article_desc_col].nunique()

# Create a data column
# df_main['date'] = pd.to_datetime(df_main['year']*10000+df_main['month']*100+df_main['day'],format='%Y%m%d')

# ------ Top N items bought ------
item_num = 10
df_item = df_main.groupby(by = article_desc_col)['Member_number'].agg(['count']).sort_values(['count'], ascending = False).reset_index()
print(f'\nTop {item_num} Items are as follows: \n{df_item.head(item_num)}') # Check

# plt.style.use('default')
figure(figsize=fig_size, dpi=fig_dpi)

plt.bar(df_item[article_desc_col][:item_num],df_item['count'][:item_num], color = 'darkgreen')
plt.title(f'Top {item_num} items by number of purchases')
plt.xlabel('Item Description')
plt.ylabel('Number of itmes')
plt.xticks(rotation=90)
plt.show()

df_year = df_main.groupby(['year'])['Member_number'].agg(['count'])
print(f'\nNumber of items bought by year is as followss: \n{df_year.head(10)}') # Check

# plt.style.use('dark_background')

# ------ Trend of items bought ------
df_month_year = df_main.groupby(by=['year','month'], as_index = False)['Member_number'].agg(['count']).sort_values(['year','month'], ascending = True).reset_index()
print(f'\nNumber of items bought by year-month is as follows: \n{df_month_year.head(10)}') # Check

df_month_year['year-month'] = df_month_year['year'].astype(str) + "_" + df_month_year['month'].astype(str)

figure(figsize=fig_size, dpi=fig_dpi)

plt.plot(df_month_year['year-month'],df_month_year['count'], color='red', marker='o')
plt.title('Trend of number of items bought')
plt.xlabel('Month-Year')
plt.ylabel('Number of Items')
plt.xticks(rotation=90)
# plt.grid(True)
plt.show()


# ------ Trend of Top Items bought ------
top_items = min(5, item_num)
list_items = list(df_item[article_desc_col][:top_items])
df_items_mon_year = df_main[df_main[article_desc_col].isin(list_items)].groupby(by=[article_desc_col,'year','month'], as_index = False)['Member_number'].agg(['count']).sort_values([article_desc_col,'year','month'], ascending = True).reset_index()
print(f'\nNumber of Top items bought by year-month is as follows: \n{df_items_mon_year.head(10)}') # Check

df_items_mon_year['year-month'] = df_items_mon_year['year'].astype(str) + "_" + df_items_mon_year['month'].astype(str)

figure(figsize=fig_size, dpi=fig_dpi)

for item in list_items:
    plt.plot(df_items_mon_year[df_items_mon_year[article_desc_col] == item]['year-month'], df_items_mon_year[df_items_mon_year[article_desc_col] == item]['count'], label=item) # , marker='o' , color='skyblue')

plt.title(f'Trend of Top {top_items} items bought')
plt.xlabel('Month-Year')
plt.ylabel('Number of Items')
plt.xticks(rotation=90)
plt.legend(loc='upper left',ncol=2, fancybox=True, shadow=True)
plt.show()


######################################################################################################################################## 
#                                                                                                                                      #
###################                               STEP 4: DATA PREPARATION                                 ############################# 
#                                                                                                                                      #
########################################################################################################################################

df_main[article_desc_col] = df_main[article_desc_col].str.strip()
df_main_1a = df_main[['Member_number', article_desc_col]]
df_main_1a['qty'] = 1

df_basket = df_main_1a.pivot_table(index='Member_number', columns=article_desc_col, values='qty', aggfunc='count',fill_value=0) 

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
df_basket_sets = df_basket.applymap(encode_units)

######################################################################################################################################## 
#                                                                                                                                      #
###################                            STEP 5: BUILD ASSOCIATION RULES                             ############################# 
#                                                                                                                                      #
########################################################################################################################################

start_time = time.time()

frequent_itemsets = apriori(df_basket_sets, min_support = 0.01, use_colnames = True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x:len(x))

# frequent_itemsets_filtered = frequent_itemsets[frequent_itemsets['support'] >= 0.1]

time_to_run = time.time() - start_time
print(f'Time to run apriori algorithm: {round(time_to_run,2)} second(s)')

df_rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 1)
df_rules.sort_values('lift', ascending = False, inplace = True)


# ------ Convert "Frozen Sets" to tuples ------

# cols_remove_frozen = ['antecedents', 'consequents']
# df_rules[cols_remove_frozen] = df_rules[cols_remove_frozen].applymap(lambda x: tuple(x))

a_c_list = ['antecedents', 'consequents']
for items in a_c_list:
    df_rules[items] = df_rules[items].apply(lambda a: ', '.join(list(a)))


top_rules = 10
print(f'\nTop {top_rules} rules are: \n',df_rules.head(top_rules))

######################################################################################################################################## 
#                                                                                                                                      #
###################                    STEP 6: ANALYSIS OF ASSOCIATION RULES                               ############################# 
#                                                                                                                                      #
########################################################################################################################################


# ------ Quick look at the distribution of the product combination ------

sns.set_context("notebook")

sns.relplot(x='antecedent support', y='consequent support', data=df_rules, size='lift', hue='confidence', height=6, aspect=1.5)
plt.title("Antecedent Support v/s. Consequent Support", fontsize=16, y=1.02)
plt.xlabel('Antecedent Support', fontsize=13)
plt.ylabel('Consequent Support', fontsize=13)
plt.show()

# ------ Analysis of Selected Item ------

select_item = 'whole milk'

rules_sel = df_rules[df_rules['antecedents'].apply(lambda x: select_item in x)]
rules_sel.sort_values('confidence', ascending=False)

rules_support = rules_sel['support'] >= rules_sel['support'].quantile(q = 0.95)
rules_confi = rules_sel['confidence'] >= rules_sel['confidence'].quantile(q = 0.95)
rules_lift = rules_sel['lift'] > 1

rules_best = rules_sel[rules_support & rules_confi & rules_lift]

print(f'\nBest association rules for {select_item} are:\n', rules_best)


######################################################################################################################################## 
#                                                                                                                                      #
###################                          STEP 7: OUTPUT ASSOCIATION RULES                              ############################# 
#                                                                                                                                      #
########################################################################################################################################

if output_dataset == 'YES':
    df_rules.to_csv(f'{output_path}/Association_rules_{date_stamp}.csv', index = False)
    print('Association rules dataframe was output to csv')

else:
    print('Association rules dataframe was *NOT* output')

