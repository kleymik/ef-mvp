# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
# #%auto
#typeset_mode(True, display=true)
print("hello") # python3

# %% [markdown]
# \begin{align}
# \dot{x} & = \sigma(y-x) \\
# \dot{y} & = \rho x - y - xz \\
# \dot{z} & = -\beta z + xy
# \end{align}

# %% [markdown]
# efficient frontier:
#
# re-arrange efficient frontier equation for risk($\sigma$) and for return($\mu$)
#
# calculate intermediate scalars A,B,C,D from matrix of covariance and vector of returns, numerically and symbolically
#
# plot hyperbola of risk/return
#
# derive mimimum risk = hyperbola and apex
#
# "Markowitz Review Paper" href="http://ramanujan.math.trinity.edu/tumath/research/studpapers/s21.pdf" 

# %% [markdown]
#

# %% [markdown]
# ### 1: hyperbola equation
#
# efficient frontier hyperbola:
#
#
# $ \frac{\sigma^2}{1/C} - \frac{(\mu - A/C)^2}{D/C^2} = 1 $
#
# where:
#
# $ A = \mathbf{1}^T V^{-1} e = e^T V^{-1}\mathbf{1} $
#
# $ B = e^T V^{-1} e $
#
# $ C = \mathbf{1}^T V^{-1} \mathbf{1} $
#
# $ D = BC - A^2 $
#
# where $e$=expected returns, $V$=covariance
#
#
#

# %% [markdown]
# ### 2: solve for sigma
#
# $ \frac{\sigma^2}{(1/C)} - \frac{(\mu - A/C)^2}{(D/C^2)} = 1 $
#
# $ \frac{\sigma^2}{(1/C)} = 1 + \frac{(\mu - A/C)^2}{(D/C^2)} $
#
# divide by C:
#
# $ \frac{\sigma^2}{(C/C)} = \frac{1}{C} + \frac{(\mu - A/C)^2}{(DC/C^2)} $
#
# $ \sigma^2 = \frac{1}{C} + \frac{(\mu - A/C)^2}{(D/C)} $
#
# $ \sigma^2 = \frac{1}{C} + \frac{(\mu - A/C)^2C}{D} $
#
# $ \sigma^2 = \frac{1}{C} + \frac{\mu^2C - 2\mu A + A^2/C}{D} $
#
# $ \sigma^2 = \frac{D + \mu^2C^2 - 2\mu AC + A^2}{CD} $
#
# $ \sigma^2 = \frac{D + (\mu C -A)^2}{CD} $
#

# %% [markdown]
# ### 3: solve for mu
# $ \frac{\sigma^2}{(1/C)} - \frac{(\mu - A/C)^2}{(D/C^2)} = 1 $
#
# $ \frac{\sigma^2}{(1/C)} -1 = \frac{(\mu - A/C)^2}{(D/C^2)} $
#
# multiply through by $ D/C^2 $
#
# $ \frac{D \sigma^2}{(C^2/C)} - \frac{D}{C^2}  =  (\mu - A/C)^2 $
#
# $ \sqrt{\frac{D \sigma^2}{C} - \frac{D}{C^2}} =  (\mu - A/C) $
# $ \sqrt{\frac{D(C \sigma^2-1)}{C^2}} =  (\mu - A/C) $
# $ \mu = \frac{\sqrt{D(C\sigma^2-1)}}{C} + A/C $
# $ \mu = \frac{\sqrt{D(C\sigma^2-1)}+A}{C} $

# %% [markdown]
# $ \frac{\sigma^2}{1/C} - \frac{(\mu - A/C)^2}{D/C^2} = 1$, where $D = BC - A^2 $

# %%
import sympy as sym
import IPython.display as disp
import numpy as np
from numpy import linalg as LA

# %%
u, s, A, B, C, D = sym.symbols('u s A B C D')
res = sym.solve(sym.Eq( (s**2 / (1/C)) - ((u - A/C)**2 / (D / C**2)), 1),s)
disp.display(res[0])
disp.display(res[1])
#dir(display)

# %% [markdown]
# ### 4: efficient frontier minimum:
# efficient frontier hyperbola coordinates of minimum:
# $ (\sigma, \mu) = ( \sqrt{1/C},  ( A/C )) $

# %% [markdown]
# ### 5: (sigma, mu) coordinates at minimum
# $ \sigma^2 = \frac{D + (\mu C -A)^2}{CD} $
# let $ \mu=A/C $
# $ \sigma^2 = \frac{D + ((A/C)C - A)^2}{CD} $
# $ \sigma^2 = \frac{D + (A - A)^2}{CD} $
# $ \sigma^2 = \frac{1}{C} $

# %%
# 0 numerical example

# %%
# 1 sample expected returns
mu3 = np.matrix([0.1, 0.05, 0.03]).T; sym.Matrix(mu3)

# %%
# 2 sample correlations
cor3 = np.matrix([[ 1.,          0.61229076, -0.13636468],
                  [ 0.61229076,  1.,         -0.29579264],
                  [-0.13636468, -0.29579264,  1.        ]]); sym.Matrix(cor3)

# %%
# 3 sample vols (stdev)
# vol3 = matrix([ 1.27241802,  1.37165177,  1.39931956])*3; vol3
vol3 = np.matrix([ 0.05,  0.08,  0.02]); sym.Matrix(vol3)

# %%
# 4 cov from vol and cor (element-wise)
cov3 = np.matrix(np.multiply(vol3.T * vol3, cor3)); sym.Matrix(cov3)

# %%
# 5 check risk = sqrt(variance)
vol3chk = np.sqrt(cov3.diagonal()); sym.Matrix(vol3chk) # OK!

# %%
# 6 check: from cov back to cor
cor3chk = np.reciprocal(vol3)  * cov3 * np.reciprocal(vol3).T; sym.Matrix(cor3chk) # OK!

# %%
# sample covariance matrix
cov3B = np.matrix([[1.61904762, 1.52285714, 0.90285714],
               [1.52285714, 1.88142857, 1.39309524],
               [0.90285714, 1.39309524, 1.95809524]]); sym.Matrix(cov3B)

# %%
# for A,B,C,D calcs
ones3 = np.matrix([1,1,1]).T; sym.Matrix(ones3)

# %%
cov3Inv = cov3**-1; sym.Matrix(cov3Inv)

# %%
LA.cond(cov3Inv) # check inverted matrxi condition number

# %% [markdown]
# $ matrix\ A = \mathbf{1}^T V^{-1} e == e^T V^{-1}\mathbf{1} $
#

# %%
# for A,B,C,D calcs
ones3 = np.matrix([1,1,1]).T; sym.Matrix(ones3)

# %%
a = (ones3.T * cov3Inv * mu3)[0,0]# ; sym.Matrix(a)
a
#print([q for q in dir() if not q.startswith('_')])

# %% [markdown]
# $ matrix\ B = e^T V^{-1} e $

# %%
b = (mu3.T * cov3Inv * mu3); sym.Matrix(b)

# %% [markdown]
# $ matrix\ C = \mathbf{1}^T V^{-1} \mathbf{1} $

# %%
c = (ones3.T * cov3Inv * ones3); sym.Matrix(c)

# %% [markdown]
# $ matrix\ D = BC - A^2  $

# %%
d = (b*c - a**2); sym.Matrix(d)

# %% [markdown]
# from above
#
# $ \sigma =   \sqrt{\frac{D + (\mu C - A)^2}{CD}} $
#
# $ \mu    =    \frac{\sqrt{D(C\sigma^2-1)}+A}{C}    $

# %%
# mimimum: sigma, mu
(np.sqrt(1/c), a/c)

# %%
mu = symbol('mu')
fsigma(mu) = sqrt( (d + (mu*c-a)^2)/(c*d) ); fsigma

# %%
#plot(fsigma,(mu,0.03,0.06), figsize=[4,4], legend_label='$\sigma(\mu)$') # rotated version
from sympy.plotting import plot as symplot
#mu = symbol('mu')
#fsigma(mu) = sqrt( (d + (mu*c-a)^2)/(c*d) ); fsigma

nu = sym.symbols('nu')
#x = 0.05*nu + 0.2/((nu - 5)**2 + 2)
x = sym.sqrt(nu)
#x = ( (d + (nu*c-a)) / (c*d) )**0.5
symplot(x)

# %%
# symplot?

# %%
var('sigma')
fmu(sigma) = ( sqrt( d * (c*sigma^2 - 1) ) + a ) / c; fmu

# %%
plot(fmu,(sigma,0.001,0.02), figsize=[4,4],legend_label='$\mu(\sigma)$')


# %%
def fmew(sigma,side):
    if side=='pos': return ( a + sqrt(d*(c*sigma^2-1)) ) / c
    else          : return ( a - sqrt(d*(c*sigma^2-1)) ) / c

#def fmew2(sigma):
#    if            : return ( a + sqrt(d*(c*sigma^2-1)) ) / c
#    else          : return ( a - sqrt(d*(c*sigma^2-1)) ) / c

for sgm in srange(0.01,0.02,0.001):
    print sgm, fmew(sgm, 'neg'), fmew(sgm,'pos')

# %%
v = [(x, fmew(x,'neg')) for x in srange(1.125,1.119515028,-1e-6)]+[(x, fmew(x,'pos')) for x in srange(1.119515028,1.125,1e-6)]
#v = [(x, fmew(x,'neg')) for x in srange(0.02,0.01,-0.001)] + [(x, fmew(x,'pos')) for x in srange(0.01,0.02,0.001)]
line(v, figsize=[4,4])

# %%
# symbolic form of hyperbola in terms of asset variances and returns
var('u,s,A,B,C,D, E, s1,s2,s3, cv12,cv13,cv23, r0,r1,r2')
V = matrix([[s1^2, cv12, cv13],
            [cv12, s2^2, cv12],
            [cv13, cv12, s3^2]]);V

# %%
V.inverse()[0,1]-V.inverse()[1,0]

# %%
V.I.is_symmetric()

# %%
V.I

# %%
var('vi00,vi11,vi22, vi01,vi02,vi12')
Vi = matrix([[vi00, vi01, vi02],
             [vi01, vi11, vi12],
             [vi02, vi12, vi22]]);Vi

# %%
E = matrix([r0,r1,r2]).T;E

# %%
# A
html('$ A = \mathbf{1}^T V^{-1} e = e^T V^{-1}\mathbf{1} $')
A = (ones3.T * Vi * E)[0,0]; A

# %%
# B
html('$ B = e^T V^{-1} e $')
B = (E.T * Vi * E)[0,0]; B

# %%
# C
html('$ C = \mathbf{1}^T V^{-1} \mathbf{1} $')
C = (ones3.T * Vi * ones3)[0,0]; C

# %%
# D
html('$ D = BC - A^2  $')
D = (B*C - A^2); D

# %%
fsigmaSymb(u) = sqrt(D+(u*C-A)^2/(C*D)); fsigmaSymb

# %% language="latex"
# \begin{align}
# \frac{\partial u}{\partial t} + \nabla \cdot \left( \boldsymbol{v} u - D\nabla u \right) = f
# \end{align}
#

# %%
fmuSymb(s) =  ( sqrt(D*(C*s^2-1)) + A) / C; fmuSymb

# %%
# %html
\mbox{vi}_{00} + 2 \, \mbox{vi}_{01} + 2 \, \mbox{vi}_{02} + \mbox{vi}_{11} + 2 \, \mbox{vi}_{12} + \mbox{vi}_{22}}</script></html>
}}}

# %%
fmuSymb.diff(s)

# %%

# %%
plot

# %%
