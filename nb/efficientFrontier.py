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
# %autosave 60
# #%auto
#typeset_mode(True, display=true)
print("hello") # python3

# %%
# all imports: for symbolic manipulation and for numerical example
import sympy as sym
import IPython.display as disp
import numpy as np
from numpy import linalg as LA

# %% [markdown]
# ## 1. Efficient Frontier Formulae Manipulations
#
# This note shows some manipulations for the hyperbola curve of the efficient frontier.
#
# 1. Re-arrange efficient frontier equation for risk($\sigma$) and for return($\mu$)
#
# 2. Calculate intermediate scalars A,B,C,D from matrix of covariance and vector of returns, numerically and symbolically
#
# 3. Plot hyperbola of risk/return
#
# 4. Derive mimimum risk = hyperbola and apex
#
# Reference (using their notation): Beste, Leventhal, Williams, & Dr. Qin Lu "Markowitz Review Paper" http://ramanujan.math.trinity.edu/tumath/research/studpapers/s21.pdf 

# %% [markdown]
# ### 1.1: hyperbola equation
#
# efficient frontier hyperbola:
#
# $$ \frac{\sigma^2}{1/C} - \frac{(\mu - A/C)^2}{D/C^2} = 1 $$
#
#
# where:
#
# $$ A = \mathbf{1}^T V^{-1} e = e^T V^{-1}\mathbf{1} $$
#
# $$ B = e^T V^{-1} e $$
#
# $$ C = \mathbf{1}^T V^{-1} \mathbf{1} $$
#
# $$ D = BC - A^2 $$
#
#
# $e$ = expected returns vector
#
# $V$ = covariance matrix
#
# $\mathbf{1}$ = Identity Matrix
#
#
#

# %% [markdown]
# ### 1.2: solve for sigma: $\sigma$
#
# $$ \frac{\sigma^2}{(1/C)} - \frac{(\mu - A/C)^2}{(D/C^2)} = 1 $$
#
# $$ \frac{\sigma^2}{(1/C)} = 1 + \frac{(\mu - A/C)^2}{(D/C^2)} $$
#
# divide by $C$:
#
# $$ \frac{\sigma^2}{(C/C)} = \frac{1}{C} + \frac{(\mu - A/C)^2}{(DC/C^2)} $$
#
# $$ \sigma^2 = \frac{1}{C} + \frac{(\mu - A/C)^2}{(D/C)} $$
#
# $$ \sigma^2 = \frac{1}{C} + \frac{(\mu - A/C)^2C}{D} $$
#
# $$ \sigma^2 = \frac{1}{C} + \frac{\mu^2C - 2\mu A + A^2/C}{D} $$
#
# $$ \sigma^2 = \frac{D + \mu^2C^2 - 2\mu AC + A^2}{CD} $$
#
# $$ \sigma^2 = \frac{D + (\mu C -A)^2}{CD} $$
#
# $$ \sigma = \sqrt{\frac{D + (\mu C -A)^2}{CD}} $$

# %% [markdown]
# ### 1.3: solve for mu: $\mu$
#
# $$ \frac{\sigma^2}{(1/C)} - \frac{(\mu - A/C)^2}{(D/C^2)} = 1 $$
#
# $$ \frac{\sigma^2}{(1/C)} -1 = \frac{(\mu - A/C)^2}{(D/C^2)} $$
#
# multiply through by $ D/C^2 $
#
# $$ \frac{\sigma^2 D}{(C^2/C)} - \frac{D}{C^2}  =  (\mu - A/C)^2 $$
#
# $$ \sqrt{\frac{\sigma^2 D}{C} - \frac{D}{C^2}} =  (\mu - A/C) $$
#
# $$ \sqrt{\frac{D(\sigma^2 C - 1)}{C^2}} =  (\mu - A/C) $$
#
# $$ \mu = \frac{\sqrt{D(\sigma^2 C - 1)}}{C} + A/C $$
#
# $$ \mu = \frac{\sqrt{D(\sigma^2 C - 1)}+A}{C} $$

# %% [markdown]
# ### 1.4: efficient frontier minimum:
# efficient frontier hyperbola coordinates of minimum:
#
# $$ (\sigma, \mu) = ( \sqrt{1/C},  ( A/C )) $$
#

# %% [markdown]
# ### 1.5: (sigma, mu) coordinates at minimum
# $$ \sigma^2 = \frac{D + (\mu C -A)^2}{CD} $$
#
# let $\mu = A/C$
#
# $$ \sigma^2 = \frac{D + ((A/C)C - A)^2}{CD} $$
#
# $$ \sigma^2 = \frac{D + (A - A)^2}{CD} $$
#
# $$ \sigma^2 = \frac{1}{C} $$

# %% [markdown]
# ## 2 manipulate equation using sympy

# %%
u, s, A, B, C, D = sym.symbols('u s A B C D')
res = sym.solve(sym.Eq( (s**2 / (1/C)) - ((u - A/C)**2 / (D / C**2)), 1),s)
#disp.display(res[0])
sym.Eq(s,res[1])

# %%
sym.Eq(s,sym.factor(res[1]))

# %% [markdown]
# ## 3 numerical example
# Calculate A, B, C, & D, hence $\sigma$ and $\mu$ for small example

# %%
# 1 sample expected returns, in perunit
mu3 = np.matrix([0.1, 0.05, 0.03]).T; sym.Matrix(mu3)

# %%
# 2 sample correlations
cor3 = np.matrix([[ 1.,          0.61229076, -0.13636468],
                  [ 0.61229076,  1.,         -0.29579264],
                  [-0.13636468, -0.29579264,  1.        ]]); sym.Matrix(cor3)

# %%
# 3 sample vols (stdev)
vol3 = np.matrix([ 0.05,  0.08,  0.02]); sym.Matrix(vol3)

# %%
# 4 covariances from vols and correlation (element-wise!)
cov3 = np.matrix(np.multiply(vol3.T * vol3, cor3)); sym.Matrix(cov3)

# %%
# 5 check: risk = sqrt(variance)
vol3chk = np.sqrt(cov3.diagonal()); sym.Matrix(vol3chk) # OK!

# %%
# 6 check: from covariances back to correlations
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
LA.cond(cov3Inv) # check condition number of the inverted matrix

# %% [markdown]
# $$ matrix\ A = \mathbf{1}^T V^{-1} e == e^T V^{-1}\mathbf{1} $$
#

# %%
# for A:a, B:b, C:c, D:d numerical example
ones3 = np.matrix([1,1,1]).T; sym.Matrix(ones3)

# %%
a = (ones3.T * cov3Inv * mu3)[0,0]; a

# %% [markdown]
# $ matrix\ B = e^T V^{-1} e $

# %%
b = (mu3.T * cov3Inv * mu3)[0,0]; b

# %% [markdown]
# $$ matrix\ C = \mathbf{1}^T V^{-1} \mathbf{1} $$

# %%
c = (ones3.T * cov3Inv * ones3)[0,0]; c

# %% [markdown]
# $ matrix\ D = BC - A^2  $

# %%
d = (b*c - a**2); d

# %% [markdown]
# hece using, from above
#
# $ \sigma =   \sqrt{\frac{D + (\mu C - A)^2}{CD}} $
# and
# $ \mu    =    \frac{\sqrt{D(\sigma^2 C - 1)}+A}{C} $

# %%
# mimimum: sigma, mu
(np.sqrt(1/c), a/c)

# %% [markdown]
# ## 4: Symbolic Plot

# %%
#fsigma = exp( (d + (mu*c-a)**2) / (c*d) ),0.5)
#fsigma = exp(( (d + (nu*c-a)**2) / (c*d) ) ,1)

# %% [markdown]
# $ \sigma =   \sqrt{\frac{D + (\mu C - A)^2}{CD}} $

# %%
#plot(fsigma,(mu,0.03,0.06), figsize=[4,4], legend_label='$\sigma(\mu)$') # rotated version
from sympy.plotting import plot as symplot
sgma = sym.sqrt( (d + (u*c - a)**2) / (c*d) )
symplot(sgma, (u, -0.0, 0.3), axis_center=(0.0,0.0), xlabel='$\mu$ mu', ylabel='$\sigma$ sigma')

# %% [markdown]
# $ \mu    =    \frac{\sqrt{D(\sigma^2 C - 1)}+A}{C} $

# %%
mu = ( sym.sqrt(d * (s**2 * c - 1) ) + a ) / c; mu

# %%
symplot(mu,(s,0.0,0.06), axis_center=(0.0,0.04), ylabel='$\mu$ mu', xlabel='$\sigma$ sigma')


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

# %% [markdown]
# ## 5: explcit formula 
# for small portfolio (3 assets)

# %%
# symbolic form of hyperbola in terms of asset variances and returns
u,s,A,B,C,D,E, s1,s2,s3, cv12,cv13,cv23, r0,r1,r2 = sym.symbols(
    'u s A B C D E  s1 s2 s3 cv12 cv13 cv23  r0 r1 r2')

V = sym.Matrix([[s1**2, cv12,  cv13],
                [cv12,  s2**2, cv23],
                [cv13,  cv23,  s3**2]]);V

# %%
V.inv()

# %%
sym.simplify(V.inv()[0,1]-V.inv()[1,0]) # check inverse

# %%
#sym.ask(sym.Q.symmetric(V.inv()))

# %%
vi00,vi11,vi22, vi01,vi02,vi12 = sym.symbols('vi00,vi11,vi22, vi01,vi02,vi12')
Vi = sym.Matrix([[vi00, vi01, vi02],
             [vi01, vi11, vi12],
             [vi02, vi12, vi22]]);Vi

# %%
E = sym.Matrix([r0,r1,r2]).T;E

# %% [markdown]
# $$ A = \mathbf{1}^T V^{-1} e = e^T V^{-1}\mathbf{1} $$

# %%
# A
#A = (ones3 * Vi * E.T)[0,0]; A
sym.Matrix(ones3.T) @ Vi @ E.T

# %% [markdown]
# $ B = e^T V^{-1} e $

# %%
# B
B = (E.T * Vi * E)[0,0]; B

# %% [markdown]
# $$ C = \mathbf{1}^T V^{-1} \mathbf{1} $$

# %%
# C
C = (ones3.T * Vi * ones3)[0,0]; C

# %% [markdown]
# $ D = BC - A**2  $

# %%
# D
D = (B*C - A**2); D

# %%
fsigmaSymb(u) = sqrt(D+(u*C-A)^2/(C*D)); fsigmaSymb

# %%
fmuSymb(s) =  ( sqrt(D*(C*s^2-1)) + A) / C; fmuSymb

# %% [markdown]
# \begin{align}
# \mbox{vi}_{00} + 2 \,
# \mbox{vi}_{01} + 2 \, 
# \mbox{vi}_{02} + 
# \mbox{vi}_{11} + 2 \, 
# \mbox{vi}_{12} + 
# \mbox{vi}_{22}
# \end{align}

# %%
fmuSymb.diff(s)

# %%
plot

# %%
