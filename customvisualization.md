### Assignment 3 - Building a Custom Visualization
In this assignment you must choose one of the options presented below and submit a visual as well as your source code for peer grading. The details of how you solve the assignment are up to you, although your assignment must use matplotlib so that your peers can evaluate your work. The options differ in challenge level, but there are no grades associated with the challenge level you chose. However, your peers will be asked to ensure you at least met a minimum quality for a given technique in order to pass. Implement the technique fully (or exceed it!) and you should be able to earn full grades for the assignment.

      Ferreira, N., Fisher, D., & Konig, A. C. (2014, April). Sample-oriented task-driven visualizations: allowing users to make better, more confident decisions.       In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 571-580). ACM. (video)

In this paper the authors describe the challenges users face when trying to make judgements about probabilistic data generated through samples. As an example, they look at a bar chart of four years of data (replicated below in Figure 1). Each year has a y-axis value, which is derived from a sample of a larger dataset. For instance, the first value might be the number votes in a given district or riding for 1992, with the average being around 33,000. On top of this is plotted the 95% confidence interval for the mean (see the boxplot lectures for more information, and the yerr parameter of barcharts).


Figure 1

        Figure 1 from (Ferreira et al, 2014).


A challenge that users face is that, for a given y-axis value (e.g. 42,000), it is difficult to know which x-axis values are most likely to be representative, because the confidence levels overlap and their distributions are different (the lengths of the confidence interval bars are unequal). One of the solutions the authors propose for this problem (Figure 2c) is to allow users to indicate the y-axis value of interest (e.g. 42,000) and then draw a horizontal line and color bars based on this value. So bars might be colored red if they are definitely above this value (given the confidence interval), blue if they are definitely below this value, or white if they contain this value.


Figure 1

Figure 2c from (Ferreira et al. 2014). Note that the colorbar legend at the bottom as well as the arrows are not required in the assignment descriptions below.



Easiest option: Implement the bar coloring as described above - a color scale with only three colors, (e.g. blue, white, and red). Assume the user provides the y axis value of interest as a parameter or variable.

Harder option: Implement the bar coloring as described in the paper, where the color of the bar is actually based on the amount of data covered (e.g. a gradient ranging from dark blue for the distribution being certainly below this y-axis, to white if the value is certainly contained, to dark red if the value is certainly not contained as the distribution is above the axis).

Even Harder option: Add interactivity to the above, which allows the user to click on the y axis to set the value of interest. The bar colors should change with respect to what value the user has selected.

Hardest option: Allow the user to interactively set a range of y values they are interested in, and recolor based on this (e.g. a y-axis band, see the paper for more details).

Note: The data given for this assignment is not the same as the data used in the article and as a result the visualizations may look a little different.
_______________________________________________________________

### Use the following data for this assignment:
```
%matplotlib notebook
import pandas as pd
import numpy as np

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
```

I had to transpose the data to use it in my graph.
```
df = df.T
df.columns = df.columns.astype(str)
```
I calculated the mean and standard deviation for each graph per the requirements.
```
avg = []
std = []
for year in range(1992,1996):
    avg.append(df[str(year)].mean())
    std.append(df[str(year)].std())
```
The next step was creating the confidence intervals.
```
import math

n = len(df)
z = 1.96

lower = []
upper = []
ci = []
for year in range(1992, 1996):
    lower.append(avg[year-1992] - (z*(std[year-1992]/math.sqrt(n))))
    upper.append(avg[year-1992] + (z*(std[year-1992]/math.sqrt(n))))
    ci.append(upper[year-1992] - lower[year-1992])
```

I normalized the curve and used a colormap for the data.
```
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

norm = Normalize(vmin=-1.96, vmax=1.96)
cmap = get_cmap('seismic')
df_c = pd.DataFrame(index = [0,1,2,3], columns = ['Value', 'Color'])

# We had to pick a value to showcase the full range of colors in our colormap.
y=40000

for i in range(0,4):
    df_c['Value'][i] = norm((avg[i]-y)/std[i])

df_c['Color'] = [cmap(x) for x in df_c['Value']]
```
Now to plot the data and add labels.
```
import matplotlib.pyplot as plt

generic = [0,1,2,3]
x = ['1992', '1993', '1994', '1995']

plt.figure(figsize=(10,5))
plt.bar(generic, avg, yerr=ci, color=df_c['Color'])
plt.axhline(y=y, color = 'black', alpha=.3)

plt.text(3.65, y, 'y=%d' %y)
plt.xticks(generic, x)
plt.xlabel('Year')
plt.ylabel('Mean Votes')
plt.title('Average Votes per Year')
```
