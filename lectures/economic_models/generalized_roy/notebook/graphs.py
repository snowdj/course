''' Graphs of the presentation.

    * Distribution of Individual Treatment Effect
    * ATE
    * TT
    * TUT
    * Return to Marginal Agent

'''

import pylab as pl
import numpy as np
import matplotlib.mlab as mlab

mean = 0.2
sd = 2.0
cost = 1.5
tt = 2.5
tut = -0.5
x = np.linspace(-5.5, 5.5, 120)
y = mlab.normpdf(x, mean, sd)
pl.xlim(x.min(), x.max())
pl.ylim(y.min(), y.max() * 1.1)
pl.xticks([min(x),tut,mean,cost,tt,max(x)])
pl.yticks(np.arange(0, max(y)+0.1, 0.05))
pl.axvline(x = mean, color = "red", linewidth = 2.5, linestyle = "-")
pl.axvline(x = cost, color = "black", linewidth = 2.5, linestyle = "dashed")
pl.axvline(x = tt, color = "blue", linewidth = 2.5, linestyle = "dashed")
pl.axvline(x = tut, color = "green", linewidth = 2.5, linestyle = "dashed")
pl.plot(x, y, color = "darkblue", linewidth = 2.5, linestyle = "-")
pl.annotate('TT', fontsize=13, xy=(tt-0.1, max(y)+0.0406), xytext=(tut-1.5, max(y)+0.0384),
            arrowprops=dict(arrowstyle="->",linewidth=1.0))
pl.annotate('Return to Marginal Agent', fontsize=13, xy=(cost-0.1, max(y)+0.030), xytext=(tut-4.90, max(y)+0.0284),
            arrowprops=dict(arrowstyle="->",linewidth=1.0))
pl.annotate('ATE', fontsize=13, xy=(mean-0.1, max(y)+0.0196), xytext=(tut-1.67, max(y)+0.0174),
            arrowprops=dict(arrowstyle="->",linewidth=1.0))
pl.annotate('TUT', fontsize=13, xy=(tut-0.1, max(y)+0.0096), xytext=(tut-1.65, max(y)+0.0074),
            arrowprops=dict(arrowstyle="->",linewidth=1.0))
#pl.show()
pl.savefig("treatment_effects.png")