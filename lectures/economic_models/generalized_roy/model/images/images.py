''' Graphs of the presentation.

    * Distribution of Individual Benefits
    * Marginal Treatment Effects
    * Joint Distribution of Potential Outcomes

'''

import pylab as pl
import numpy as np
import matplotlib.mlab as mlab
import scipy.stats
import matplotlib.cm as cm

# Distribution of Individual Benefits

'''setup.
'''
mean = 0.2
sd = 2.0
mte = 1.5
tt = 2.5
tut = -0.5
x = np.linspace(-5.5, 5.5, 120)
y = mlab.normpdf(x, mean, sd)

fig = pl.figure().add_subplot(111)
fig.set_ylim(y.min(), y.max() * 1.1)
fig.set_xlim(x.min(), x.max())
pl.yticks(np.arange(0, max(y)+0.1, 0.05)[1:])
fig.vlines(x = mean, ymin = y.min(), ymax = max(y)+0.1,
          color = "red", linewidth = 2.5, linestyle = "-", label = "ATE")
fig.vlines(x = mte, ymin = y.min(), ymax = max(y)+0.1,
           color = "black", linewidth = 2.5, linestyle = "dashed", label = "MTE")
fig.vlines(x = tt, ymin = y.min(), ymax = max(y)+0.1,
            color = "blue", linewidth = 2.5, linestyle = "dashed", label = "TT")
fig.vlines(x = tut, ymin = y.min(), ymax = max(y)+0.1,
            color = "green", linewidth = 2.5, linestyle = "dashed", label = "TUT")
fig.plot(x, y, color = "darkblue", linewidth = 2.5, linestyle = "-")

box = fig.get_position()

fig.legend(loc = 'upper center')
fig.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.88])
fig.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.10),
    fancybox = False, frameon = False, shadow = False, ncol = 3)

pl.savefig("treatment_effects.png")

# Marginal Treatment Effects

''' Setup.
'''
bmteColor = 'blue'
cmteColor = 'darkgreen'
smteColor = 'red'

# Level and Slopes.
bmteLevel  = 0.10
cmteLevel  = 0.10
smteLevel  = bmteLevel - cmteLevel

bmteSlopes =  -0.05
cmteSlopes =   0.05
smteSlopes =  -1.00

# Construct marginal effects of treatment.
evalPoints = np.round(np.arange(1, 100)/100.0, decimals = 2)
quantile   = scipy.stats.norm.ppf(evalPoints, loc = 0, scale = 1.0)

bmte = bmteLevel + bmteSlopes*quantile
cmte = cmteLevel + cmteSlopes*quantile
smte = bmte      - cmte

# Parametrization.
yLower = -0.30
yUpper =  0.30

''' Graph.
'''
fig = pl.figure().add_subplot(111)

fig.set_ylim(yLower, yUpper)

x = np.linspace(0.01, 0.99, 99)

fig.plot(x, bmte,  color = bmteColor, linewidth = 4, \
             label = r''' $B^{MTE}$''', linestyle = '--')

fig.plot(x, cmte,  color = cmteColor, linewidth = 4, \
             label = r''' $C^{MTE}$''', linestyle = '-.')

fig.plot(x, smte,  color = smteColor, linewidth = 4, \
             label = r''' $S^{MTE}$''', linestyle = ':')

pl.plot([0.50, 0.50], [0.0, 0.1], 'o', color = 'black', linewidth = 40)

# Labeling..
pl.xlabel(r''' $u_S$''', fontsize = 20)
pl.ylabel(''' Marginal Effects of Treatment ''', fontsize = 12)

box = fig.get_position()

fig.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.95])

fig.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.10),
    fancybox = False, frameon = False, shadow = False, ncol = 3)

# Margin.
fig.vlines(x = 0.50,ymin = yLower, ymax = 0.10, color = 'k', linestyle = '--')

# Zero Surplus.
fig.hlines(y = 0.0, xmin = 0.00, xmax = 0.50, color = 'k', linestyle = '--')

# Positive Benefits of Marginal Agents.
fig.hlines(y = 0.1, xmin = 0.00, xmax = 0.50, color = 'k', linestyle = '--')

# Text.
fig.text(0.42, 0.15, r'$P(x,z) = 0.50$', fontsize = 15)

pl.savefig('marginalRelationships.png')


# Joint Distribution of Potential Outcomes

'''Setup.
'''

# para
beta1 = np.array([0.5, 0.2, 0.5])
beta0 = np.array([0.1, 0.2, 0.3])
var1 = 0.5
var2 = 0.2
cov = 0.1
V = [[var1, cov], [cov, var2]]
mean = (0, 0)

# simulate data
np.random.seed(123)
n_sim = 5000
X1 = np.random.normal(1.5, 1, n_sim)
X2 = np.random.normal(1.3, 1, n_sim)
Eps = np.random.multivariate_normal(mean, V, n_sim)

Y1 = beta1[0] + beta1[1]*X1 + beta1[2]*X2 + Eps[:, 0]
Y0 = beta0[0] + beta0[1]*X1 + beta0[2]*X2 + Eps[:, 1]

xmin, xmax = -1, 3
ymin, ymax = -1, 4

# Peform the kernel density estimate
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([Y0, Y1])
kernel = scipy.stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = pl.figure().add_subplot(111)
fig.set_xlim(xmin, xmax)
fig.set_ylim(ymin, ymax)

# Contour plot
pl.contour(xx, yy, f)
cset = fig.contour(xx, yy, f)
pl.clabel(cset, inline=1, fontsize=10)
fig.set_xlabel('$Y_0$', fontsize=15, labelpad=20)
fig.set_ylabel('$Y_1$', fontsize=15)

pl.yticks(np.arange(-1, 5, 1)[1:])
pl.xticks(np.arange(-1, 4, 1))
pl.plot([-1, 4], [-1, 4], color = "black", linewidth = 1)

box = fig.get_position()
fig.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.88])

pl.savefig("joint_dist.png")



