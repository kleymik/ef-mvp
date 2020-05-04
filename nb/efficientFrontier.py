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

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Efficient Frontier  / Minimum Variance Portfolio Stuff
#
# 1 [Formulae](#1)  
#  1.1 [Hyperbolae](#1.1)  
#  1.2 [Solve for sigma](#1.2)    
#  1.3 [Solve for mu](#1.3)  
#  1.4 [Efficient frontier minimum](#1.4)  
#  1.5 [sigma & mu at minimum](#1.5)
#
# 2 [Solve for sigma & mu using sympy](#2)
#
# 3 [Numerical example](#3)
#
# 4 [Plot symbolic equation](#4)
#
# 5 [Explicit formulae - for 3 assets](#5)
#
#
#
# [//]: # (table styling via markdown)
# <style> table.dataframe { font-size:70%; }   body { font-size:70%} /style>

# %% [markdown]
# do imports, keep variables in a dict as a workspace

# %%
import datetime
import sympy as sym
from sympy.matrices import matrix_multiply_elementwise as mme
from sympy.plotting import plot as symplot
import IPython.display as disp
import numpy as np
from numpy import linalg as LA
print("imports done")

ws = {} # keep variables in a dict as a workspace
ws['dateStart'] = datetime.datetime.now().isoformat()[:16].replace(':','')
ws


# %% [markdown]
# ### 1 Efficient Frontier Formulae Manipulations <a class="anchor" id="1"></a>
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
#
#
# #### 1.1 hyperbola equation <a class="anchor" id="1.1"></a>
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
# $e$ = expected returns vector
#
# $V$ = covariance matrix
#
# $$\mathbf{1} = Identity Matrix $$
#
# $e^{T}$ = $e$ transpose
#
# #### 1.2: solve for sigma: $\sigma$ <a class="anchor" id="1.2"></a>
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
#
# #### 1.3: solve for mu: $\mu$ <a class="anchor" id="1.3"></a>
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
#
# #### 1.4: efficient frontier minimum $\sigma$ <a class="anchor" id="1.4"></a>
#
# efficient frontier hyperbola coordinates of minimum:
#
# $$ (\sigma, \mu) = ( \sqrt{1/C},  ( A/C )) $$
#
# #### 1.5: (sigma, mu) coordinates at minimum $\sigma$ <a class="anchor" id="1.5"></a>
# $$ \sigma^2 = \frac{D + (\mu C -A)^2}{CD} $$
#
# let $\mu = A/C$
#
# $$ \sigma^2 = \frac{D + ((A/C)C - A)^2}{CD} $$
#
# $$ \sigma^2 = \frac{D + (A - A)^2}{CD} $$
#
# $$ \sigma^2 = \frac{1}{C} $$
#
#

# %% [markdown]
# ### 2: manipulate equation using sympy - solve for $\sigma$ <a class="anchor" id="2"></a>
#
# take positive solution only

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
mu, sigma, A, B, C, D = sym.symbols('mu sigma A B C D')
ws['sigmaEqn'] = sym.solve(sym.Eq( (sigma**2 / (1/C)) - ((mu - A/C)**2 / (D / C**2)), 1) , sigma)[1] # [1]=> +ve soln
ws['sigmaEqn']

# %% [markdown] pycharm={"is_executing": false, "name": "#%% md\n"}
# simplify:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
sym.factor(sym.Eq(sigma, ws['sigmaEqn']))

# %% [markdown] pycharm={"name": "#%% md\n"}
# check: [original form] minus [simplified ("factored") form]:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%% \n"}
ws['sigmaEqn'] - ws['sigmaEqn']

# %% [markdown] pycharm={"name": "#%% md\n"}
# gives:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%% \n"}
sym.factor(ws['sigmaEqn'] - sym.factor(ws['sigmaEqn']))

# %% [markdown]
# ### 3 numerical example <a class="anchor" id="3"></a>
# Calculate A, B, C, & D, hence $\sigma$ and $\mu$, for a small example of 3 assets
#
# sample annualized expected returns, in percent:

# %%
ws['prec'] = 4 # number of digits of precision to display numerical values

ws['mu3'] = sym.Matrix(np.array([5.1, 7.0, 0.9]).T) # mu3 = sym.Matrix(mu3)
ws['mu3']

# %% [markdown] pycharm={"name": "#%% md\n"}
# *cor*: sample correlations:

# %%
ws['cor3'] = sym.Matrix([[  1.0,  0.5,  0.4],
                         [  0.5,  1.0, -0.1],
                         [  0.4, -0.1, 1.0]])
sym.N(ws['cor3'], ws['prec'])

# %% [markdown] pycharm={"name": "#%% md\n"}
# *vol*: sample vols (stdev):

# %%
ws['vol3'] = sym.Matrix([ 3.5,  4.2,  1.1])
sym.N(ws['vol3'], ws['prec'])

# %% [markdown] pycharm={"name": "#%% md\n"}
# *cov*: compose to make covariance matrix:

# %%
ws['cov3'] = mme(ws['vol3'] * ws['vol3'].T, ws['cor3']) # mme = element-wise multiply
sym.N(ws['cov3'], ws['prec'])

# %% [markdown] pycharm={"name": "#%% md\n"}
# check that $${variance} = vol^2$$, $$diag(cov) = vol^2$$

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%% \n"}
sym.diag(*ws['vol3'])**2   # how to do sqrt of diagonal matrix in sympy?

# %% [markdown] pycharm={"is_executing": false, "name": "#%% md\n"}
# check: get correlations back from covariance
#
# in index subscript form: $$ r_{ij} = \frac{c_{ij}}{c_{ii}*c_{jj}}$$
#
# in matrix form: $$ cor = vol^{-1}  \times cov \times  vol^{-1} $$
#
# where $$ vol = \sqrt{diag(cov)} $$ as a diagonal matrix

# %% [markdown] pycharm={"name": "#%% md\n"}
# *vol*:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
ws['oneOverVol'] = sym.diag(*ws['cov3'].diagonal())**(-0.5) # works!!!! using sym.sqrt doesn't evaluate fully
ws['oneOverVol'] * ws['cov3'] * ws['oneOverVol']            # oneOverVol is diagonal matrix so it's equal to its transpose

# %% [markdown] pycharm={"name": "#%% md\n"}
# #### 3.1 sample values for calculating sample hyperbolae scalars A,B,C,D
#
# $A,B,C,D$ calculated numerically as variables a,b,c,d:
#

# %% [markdown] pycharm={"name": "#%% md\n"}
# vector of 3 ones:

# %%
ws['ones3'] = sym.Matrix([1,1,1])
ws['ones3']

# %% [markdown] pycharm={"name": "#%% md\n"}
# inverse of *cov*

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
sym.N(ws['cov3']**(-1), ws['prec'])

# %% [markdown] pycharm={"name": "#%% md\n"}
# check condition number of *cov* inverse:

# %%
sym.N(LA.cond(np.array(ws['cov3']**(-1), dtype=float)), ws['prec']) 

# %% [markdown]
# calculate $A$ from covariance $V (=cov)$, and $e (=mu)$ : $$ A = \mathbf{1}^T V^{-1} e = e^T V^{-1}\mathbf{1} $$

# %%
ws['a'] = ws['ones3'].T @ ws['cov3']**(-1) @ ws['mu3'] # = (mu3.T @ cov3**(-1) @ ones3.T)
sym.N(ws['a'], ws['prec'])

# %% [markdown]
# calculate $B$ from covariance $V (=cov)$, and $e (=mu) : B = e^T V^{-1} e$

# %%
ws['b'] = ws['mu3'].T @ ws['cov3']**(-1) @ ws['mu3'] 
sym.N(ws['b'], ws['prec'])

# %% [markdown]
# calculate $C$ from covariance $$V (=cov) : C = \mathbf{1}^T V^{-1} \mathbf{1}$$

# %%
ws['c']  = ws['ones3'].T @ ws['cov3']**(-1) @ ws['ones3']
sym.N(ws['c'], ws['prec'])

# %% [markdown]
# calculate $D$ from covariance: $D = BC - A^2$

# %%
ws['d'] = ws['b'] * ws['c'] - ws['a']**2
sym.N(ws['d'], ws['prec'])

# %% [markdown]
# #### 3.2 hence calculate $\sigma$ or $\mu$
# using $A, B, C, D$ calculate $\sigma$ and $\mu$ from each other
#
# $$ \sigma =   \sqrt{\frac{D + (\mu C - A)^2}{CD}} $$
# and
# $$ \mu    =    \frac{\sqrt{D(\sigma^2 C - 1)}+A}{C} $$
#
# e.g. for $\mu = 0.3$, $\sigma =$ 

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
ws['sgma'] = ( (ws['d'] + (0.03 * ws['c'] - ws['a'])**2) / (ws['c']*ws['d']) )**(0.5)
sym.N(ws['sgma'], ws['prec'])

# %% [markdown] pycharm={"name": "#%% md\n"}
# e.g. for $\sigma = 3.086$, $\mu =???$

# %%
print('mu from sigma)')
# check calculate back mu from sigma
((ws['d'] * ((2**2 * ws['c']) + sym.Matrix([[1]])) )**(0.5) + ws['a'] ) / ws['c']

#sym.N(m, sPrec)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# mimimum: 
print("sigma, mu")
( (sym.Matrix([1]) / ws['c'])**(0.5), ws['a'] / ws['c'] )

# %% [markdown]
# ### 4: Symbolic Plot <a class="anchor" id="4"></a>
#
# sigma vs mu:
#
# $$ \sigma =   \sqrt{\frac{D + (\mu C - A)^2}{CD}} $$
#

# %% [markdown]
# mu vs sigma
#  
# $$ \mu = \frac{A+\sqrt{D(\sigma^2 C - 1)}}{C} $$

# %%
symOne = sym.Matrix([[1]]) # just 1 in a sympy 1x1 Matrix
ws['fmuA'] = ( ws['a'] + ws['d'] * (sigma**2 * ws['c'] - symOne)**0.5 ) / ws['c']  # upper half of hyperbola
ws['fmuB'] = ( ws['a'] - ws['d'] * (sigma**2 * ws['c'] - symOne)**0.5 ) / ws['c']  # lower half of hyperbola

p0 = symplot(ws['fmuA'][0], (sigma, 0, 5.0), axis_center=(0.0,0.0), ylabel='% $\mu$ mu', xlabel='$\sigma$ sigma %', show=False)
p1 = symplot(ws['fmuB'][0], (sigma, 0, 2.0), axis_center=(0.0,0.0), ylabel='% $\mu$ mu', xlabel='$\sigma$ sigma %', show=False)
p0.append(p1[0])
p0.show()


# %% [markdown]
# ## 5: Closed-form formulae <a class="anchor" id="5"></a>
#
# for small portfolio (3 assets) - covariance $V$
#
# symbolic form of hyperbola in terms of asset covariances and returns:
#
# (note the symmetric off-diagonal entries)

# %%
u,s,A,B,C,D,E, s1,s2,s3, cv12,cv13,cv23, r0,r1,r2 = \
    sym.symbols('u s A B C D E  s1 s2 s3 cv12 cv13 cv23  r0 r1 r2')

V = sym.Matrix([[s1**2, cv12,  cv13],
                [cv12,  s2**2, cv23],
                [cv13,  cv23,  s3**2]])
V

# %% [markdown] pycharm={"name": "#%% md\n"}
# inverse of covariance matrix $V$:

# %%
V**(-1)

# %% [markdown] pycharm={"is_executing": false, "name": "#%% md\n"}
# check multiply: $V \times V^{-1}$

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
sym.MatMul(V, V.inv(),doit=False) 

# %% [markdown] pycharm={"name": "#%% md\n"}
# **=** 

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
sym.simplify(V @ V.inv())

# %%
#sym.ask(sym.Q.symmetric(V.inv())) # only avaulable in SageMath/Cocalc?

# %%
vi00,vi11,vi22, vi01,vi02,vi12 = sym.symbols('vi00, vi11, vi22, vi01, vi02, vi12')
sA,sB,sC,sD = sym.symbols('sA, sB, sC, sD') # keep separate from symbols A,B,C,D above
Vi = sym.Matrix([[vi00, vi01, vi02],
                 [vi01, vi11, vi12],
                 [vi02, vi12, vi22]])
Vi

# %% [markdown] pycharm={"name": "#%% md\n"}
# returns: $E$ =

# %%
E = sym.Matrix([r0,r1,r2])
E

# %% [markdown]
# hence closed form formulas for $A,B,C,D$:

# %% [markdown] pycharm={"name": "#%% md\n"}
# $$ A = \mathbf{1}^T V^{-1} e = e^T V^{-1}\mathbf{1} =$$

# %%
sA = ws['ones3'].T @ V.inv() @ E
sA

# %% [markdown]
# $ B = e^T V^{-1} e = $

# %%
# B
sB = E.T @ V.inv() @ E
sB

# %% [markdown]
# $$ C = \mathbf{1}^T V^{-1} \mathbf{1} = $$

# %%
sC = ws['ones3'].T @ V.inv() @ ws['ones3']
sC

# %% [markdown]
# $ D = BC - A^2  $

# %%
sD = sB @ sC - sA**2
sD

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# hmmmm
sym.factor(sD)


# %% [markdown] pycharm={"name": "#%% md\n"}
# workspace final contents:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
ws['dateEnd'] = datetime.datetime.now().isoformat()[:16].replace(':','')

for k in sorted(ws): print("%10s" % k, type(ws[k]))





# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}


