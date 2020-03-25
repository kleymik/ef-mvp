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
# #%matplotlib inline
# #%autosave 60
# #%auto
#typeset_mode(True, display=true)
print("hello") # python3

# %% [markdown]
# [//]: # (table styling via markdown) 
# <style> table.dataframe { font-size:70%; } 
#         body            { font-size:70%}
# </style>

# %% [markdown]
# # Efficient Frontier Stuff
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

# %%
# all imports: for symbolic manipulation and for numerical example
import sympy as sym
from sympy.matrices import matrix_multiply_elementwise as mme
from sympy.plotting import plot as symplot
import IPython.display as disp
import numpy as np
from numpy import linalg as LA
print("imports done")

# %% [markdown]
# ## 1 Efficient Frontier Formulae Manipulations <a class="anchor" id="1"></a>
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
# ### 1.1 hyperbola equation <a class="anchor" id="1.1"></a>
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
# $\mathbf{1}$ = Identity Matrix
#
# ### 1.2: solve for sigma: $\sigma$ <a class="anchor" id="1.2"></a>
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
# ### 1.3: solve for mu: $\mu$ <a class="anchor" id="1.3"></a>
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
# ### 1.4: efficient frontier minimum $\sigma$ <a class="anchor" id="1.4"></a>
#
# efficient frontier hyperbola coordinates of minimum:
#
# $$ (\sigma, \mu) = ( \sqrt{1/C},  ( A/C )) $$
#
# ### 1.5: (sigma, mu) coordinates at minimum $\sigma$ <a class="anchor" id="1.5"></a>
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
# ## 2 manipulate equation using sympy - solve for $\sigma$ <a class="anchor" id="2"></a>
#
# take positive solution only:

# %%
mu, sigma, A, B, C, D = sym.symbols('mu sigma A B C D')
res = sym.solve(sym.Eq( (sigma**2 / (1/C)) - ((mu - A/C)**2 / (D / C**2)), 1), sigma)
#disp.display(res[0])
sym.Eq(sigma, res[1])

# %% [markdown]
# simplify:

# %%
sym.Eq(sigma, sym.factor(res[1]))

# %% [markdown]
# check: subtract original form from simplified ("factored") form:

# %%
res[1] - sym.factor(res[1])

# %% [markdown]
# gives:

# %%
sym.factor(res[1] - sym.factor(res[1]))

# %% [markdown]
# ## 3 numerical example <a class="anchor" id="3"></a>
# Calculate A, B, C, & D, hence $\sigma$ and $\mu$ for a small example of 3 assets
#
# sample annualized expected returns, in perone units:

# %%
sPrec = 4 # number of digits of precision to display numerical values

mu3 = sym.Matrix(np.array([0.1, 0.05, 0.03]).T) # mu3 = sym.Matrix(mu3)
mu3

# %% [markdown]
# cor3: sample correlations:

# %%
cor3 = sym.Matrix([[ 1.,          0.61229076, -0.13636468],
                   [ 0.61229076,  1.,         -0.29579264],
                   [-0.13636468, -0.29579264,  1.        ]])
sym.N(cor3, sPrec)

# %% [markdown]
# vol3: sample vols (stdev):

# %%
vol3 = sym.Matrix([ 0.05,  0.08,  0.02]) 
sym.N(vol3, sPrec)

# %% [markdown]
# cov3: compose to make covariance matrix:

# %%
cov3 = mme(vol3 * vol3.T, cor3) # mme = element-wise multiply
sym.N(cov3, sPrec)

# %% [markdown]
# check that $${variance} = vol^2, diag(cov3) = vol3^2$$ 

# %%
sym.diag(*vol3)**2   # how to do sqrt of diagonal matrix in sympy???

# %% [markdown]
# check: get correlations back from covariance
#
# in index subscript form:
# $$ r_{ij} = \frac{C_{ij}}{C_{ii}*C_{jj}}$$
#
# in matrix algebra form: $$ cor3 = vol3^{-1}  \times cov \times  vol3^{-1} $$ 
#
# where $$ vol3 = \sqrt{diag(cov3)} $$ as a diagonal matrix
#

# %%
oneOverVol = sym.diag(*cov3.diagonal())**(-0.5) # works!!!! using sym.sqrt doesn't evaluate fully
oneOverVol * cov3 * oneOverVol                  # oneOverVol is diagonal matrix so equal to it's transpose

# %% [markdown]
# ### 3.1 sample values for calculating sample hyperbolae scalars A,B,C,D 
#
# $A,B,C,D$ calculated numerically as variables a,b,c,d:

# %%
# sample covariance matrix
cov3B = sym.Matrix([[1.61904762, 1.52285714, 0.90285714],
                    [1.52285714, 1.88142857, 1.39309524],
                    [0.90285714, 1.39309524, 1.95809524]])
sym.N(cov3B, sPrec)

# %%
ones3 = sym.Matrix([1,1,1])
ones3

# %%
sym.N(cov3B**(-1), sPrec)

# %% [markdown]
# check condition number of the matrix inverse:

# %%
sym.N(LA.cond(np.array(cov3B**(-1), dtype=float)), sPrec) 

# %% [markdown]
# calculate $$A$$ from covariance $$V$$ = cov3, and $$e$$ = mu3
#
# $$ matrix\ A = \mathbf{1}^T V^{-1} e == e^T V^{-1}\mathbf{1} $$

# %%
a = (ones3.T @ cov3B**(-1) @ mu3) # = (mu3.T @ cov3B**(-1) @ ones3.T)
sym.N(a, sPrec)

# %% [markdown]
# calculate $$B$$ from covariance $$V$$=cov3, and $$e$$=mu3: $ matrix\ B = e^T V^{-1} e $

# %%
b = (mu3.T * cov3B**(-1) * mu3) 
sym.N(b, sPrec)

# %% [markdown]
# calculate $$C$$ from covariance $$V$$=cov3: $$ matrix\ C = \mathbf{1}^T V^{-1} \mathbf{1} $$

# %%
c  = ones3.T @ cov3B**(-1) @ ones3
sym.N(c, sPrec)

# %% [markdown]
# calculate $$D$$ from covariance: $ matrix\ D = BC - A^2  $

# %%
d = (b*c - a**2); 
sym.N(d, sPrec)

# %% [markdown]
# ### 3.2 hence calculate $\sigma$ or $\mu$ 
# using $A, B, C, D$ calculate $\sigma$ and $\mu$ from each other
#
# $ \sigma =   \sqrt{\frac{D + (\mu C - A)^2}{CD}} $
# and
# $ \mu    =    \frac{\sqrt{D(\sigma^2 C - 1)}+A}{C} $
#
# e.g. for $\mu = 0.3$, $\sigma =$ 

# %%
sgma = ( (d + (0.3*c - a)**2) / (c*d) )**(0.5)
sym.N(sgma, sPrec)

# %% [markdown]
# e.g. for $\sigma = 3.086$, $\mu =???$

# %%
#m = ( (d * (1.351**2 * c - 1))**(1/2) + a ) / c
((d * ((sgma**2 * c) - sym.Matrix([[1]])) )**(0.5) + a ) / c


#sym.N(m, sPrec)

# %%
# mimimum: sigma, mu
(np.sqrt(1/c), a/c)

# %% [markdown]
# ## 4: Symbolic Plot <a class="anchor" id="4"></a>
#
# sigma vs mu

# %%
#fsigma = exp( (d + (mu*c-a)**2) / (c*d) ),0.5)
#fsigma = exp(( (d + (nu*c-a)**2) / (c*d) ) ,1)

# %%
#plot(fsigma,(mu,0.03,0.06), figsize=[4,4], legend_label='$\sigma(\mu)$') # rotated version

fsgma = (( (d + (mu*c - a)**2) / (c*d) )**2)[0]
symplot(fsgma, (mu, -0.0, 0.3), axis_center=(0.0,0.0), xlabel='$\mu$ mu', ylabel='$\sigma$ sigma')
fsgma

# %% [markdown]
# mu vs sigma
#  
# $ \mu    =    \frac{\sqrt{D(\sigma^2 C - 1)}+A}{C} $

# %%
fmu = (( (d * (sigma**2 * c - sym.Matrix([[1]]) ) )**0.5 + a ) / c)[0]
symplot(fmu, (sigma, 0.1, 5), axis_center=(0.0,0.0), ylabel='$\mu$ mu', xlabel='$\sigma$ sigma')
fmu


# %%
def fmew(sigma,side):
    if side=='pos': return ( a + sqrt(d*(c*sigma^2-1)) ) / c
    else          : return ( a - sqrt(d*(c*sigma^2-1)) ) / c

#def fmew2(sigma):
#    if            : return ( a + sqrt(d*(c*sigma^2-1)) ) / c
#    else          : return ( a - sqrt(d*(c*sigma^2-1)) ) / c

for sgm in srange(0.01,0.02,0.001): print(sgm, fmew(sgm, 'neg'), fmew(sgm,'pos'))

# %%
v = [(x, fmew(x,'neg')) for x in srange(1.125,1.119515028,-1e-6)]+[(x, fmew(x,'pos')) for x in srange(1.119515028,1.125,1e-6)]
#v = [(x, fmew(x,'neg')) for x in srange(0.02,0.01,-0.001)] + [(x, fmew(x,'pos')) for x in srange(0.01,0.02,0.001)]
line(v, figsize=[4,4])

# %% [markdown]
# ## 5: explcit formula 
# for small portfolio (3 assets) - covariance $V$

# %%
# symbolic form of hyperbola in terms of asset variances and returns
u,s,A,B,C,D,E, s1,s2,s3, cv12,cv13,cv23, r0,r1,r2 = sym.symbols(
    'u s A B C D E  s1 s2 s3 cv12 cv13 cv23  r0 r1 r2')

V = sym.Matrix([[s1**2, cv12,  cv13],
                [cv12,  s2**2, cv23],
                [cv13,  cv23,  s3**2]]);V

# %%
V**(-1)

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



