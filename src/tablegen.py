import matplotlib.pyplot as plt
from pandas.tools.plotting import table
import pandas as pd

def tablegen(dict):
    df = pd.DataFrame(dict,index=['MSE','QL'])

    fig, ax = plt.subplots(figsize=(14, 2)) # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
    tabla = table(ax, df.round(3), loc='center', colWidths=[0.17]*len(df.columns))  # where df is your data frame
    tabla.auto_set_font_size(False) # Activate set fontsize manually
    tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
    tabla.scale(1, 1)
