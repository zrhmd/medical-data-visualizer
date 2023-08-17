import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  df_cat = pd.melt(df, id_vars=['cardio'],  value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  
  df_cat = pd.DataFrame(df_cat.groupby(['variable', 'value', 'cardio'])['value'].count()).rename(columns = {'value': 'total'}).reset_index()
  #df_cat = df_cat.rename(columns = {0: 'total'})
  # Draw the catplot with 'sns.catplot()'
  graph = sns.catplot(data = df_cat, kind = 'bar', x = 'variable', y = 'total', hue = 'value', col = 'cardio')

  # Get the figure for the output
  fig = graph.fig
  
  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig


# Draw Heat Map
def draw_heat_map():
  # Clean the data
  df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

  # Calculate the correlation matrix
  corr = df_heat.corr()

  # Generate a mask for the upper triangle
  mask = np.triu(np.ones_like(df_heat.corr(), dtype = bool))

  # Set up the matplotlib figure
  fig, ax = plt.subplots()

  # Draw the heatmap with 'sns.heatmap()'
  sns.heatmap(corr, annot = True, mask = mask, square = True, linewidth = 0.5, fmt = '0.1f', vmax = 0.3, center = 0)
  
  # Do not modify the next two lines
  fig.savefig('heatmap.png')
  return fig
