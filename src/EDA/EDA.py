import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import plotly.express as px
import sys
import os
from sklearn.metrics import mean_squared_error

def eda_report(data):
    '''Te EDA report will create some files to analyze the in deep the variables of the table.
    The elements will be divided by categoric and numeric and some extra info will printed'''
    
    
    describe_result=data.describe()
    eda_path = './files/modeling_output/figures/'
    reports_path='./files/modeling_output/reports/'
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    if not os.path.exists(eda_path):
        os.makedirs(eda_path)
    # Exporting the file
    with open(reports_path+f'describe.txt', 'w') as f:
        f.write(describe_result.to_string())
    # Exporting general info
    with open(reports_path+f'info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
        
    balance=data['exited'].value_counts(normalize=True)
    fig , ax  = plt.subplots()
    ax.bar(balance.index.astype(str),balance)
    ax.set_title('Balance of clases')
    fig.savefig(eda_path+f'fig_1.png')
