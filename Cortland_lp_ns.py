#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:18:31 2017

@author: Caitlin Spence
"""

import os
import math
from scipy.optimize import brentq as root
from rhodium import *
from rhodium.config import RhodiumConfig
from platypus import MapEvaluator
import matplotlib.pyplot as plt
#
RhodiumConfig.default_evaluator = MapEvaluator()

from math import log, sqrt
import numpy as np

os.chdir('')

#------------------------------------------------------------------------------
# Make lake dynamics plots
#------------------------------------------------------------------------------

def lake_eval(pollution_limit,
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean = 0.02,     # mean of natural inflows
         stdev = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100): # monte carlo sampling of natural inflows)
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(pollution_limit)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,nsamples))
    decisions = np.array(pollution_limit)
    reliability = 0.0
    natural_inflows_record = np.empty((nvars, nsamples))
    daily_P_record = np.empty((nvars, nsamples))
    daily_utility = np.zeros((nvars, nsamples))

    for k in range(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
#        daily_utility = np.zeros((nvars,))
        natural_inflows_record[:,k] = natural_inflows
        
        for t in range(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[k,t] += X[t]/float(nsamples)
            daily_utility[k,t] = daily_utility[k,t-1] + alpha*decisions[t]*np.power(delta,float(t))
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
        daily_P_record[:,k] = X
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    # intertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return (average_daily_P, daily_utility, natural_inflows_record, max_P, utility, reliability)

# Make a nonstationary version of the lake problem
def lake_eval_ns(pollution_limit,
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean_base = 0.02,     # mean of natural inflows
         stdev_base = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100): # monte carlo sampling of natural inflows)
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(pollution_limit)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,nsamples))
    decisions = np.array(pollution_limit)
    reliability = 0.0
    natural_inflows_record = np.empty((nvars, nsamples))
    daily_P_record = np.empty((nvars, nsamples))
    daily_utility = np.zeros((nvars, nsamples))
    
    time = np.arange(0, 100, 1)
    mean = mean_base + 0.00025*time    # Somewhat arbritrary trend
    stdev = stdev_base + 0.00003*time  # Somewhat arbitrary trend

    for k in range(nsamples):
        X[0] = 0.0
        natural_inflows = np.empty(nvars)
        
        for t in range(0,nvars):
            natural_inflows[t] = np.random.lognormal(
                math.log(mean[t]**2 / math.sqrt(stdev[t]**2 + mean[t]**2)),
                math.sqrt(math.log(1.0 + stdev[t]**2 / mean[t]**2)),
                size = 1)
#        daily_utility = np.zeros((nvars,))
        natural_inflows_record[:,k] = natural_inflows
        
        for t in range(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[k,t] += X[t]/float(nsamples)
            daily_utility[k,t] = daily_utility[k,t-1] + alpha*decisions[t]*np.power(delta,float(t))
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
        daily_P_record[:,k] = X
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    # intertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return (average_daily_P, daily_utility, natural_inflows_record, max_P, utility, reliability)


test1 = np.full((100), 0.01)
test2 = np.full((100), 0.05)
test3 = np.full((100), 0.07)
test4 = np.full((100), 0.025)

(dP1_stat, dU1_stat, nI1_stat, pho1_stat, uti1_stat, rel1_stat) = lake_eval(test1)
(dP2_stat, dU2_stat, nI2_stat, pho2_stat, uti2_stat, rel2_stat) = lake_eval(test2)
(dP3_stat, dU3_stat, nI3_stat, pho3_stat, uti3_stat, rel3_stat) = lake_eval(test3)
(dP4_stat, dU4_stat, nI4_stat, pho4_stat, uti4_stat, rel4_stat) = lake_eval(test4)

(dP1_ns, dU1_ns, nI1_ns, pho1_ns, uti1_ns, rel1_ns) = lake_eval_ns(test1)
(dP2_ns, dU2_ns, nI2_ns, pho2_ns, uti2_ns, rel2_ns) = lake_eval_ns(test2)
(dP3_ns, dU3_ns, nI3_ns, pho3_ns, uti3_ns, rel3_ns) = lake_eval_ns(test3)
(dP4_ns, dU4_ns, nI4_ns, pho4_ns, uti4_ns, rel4_ns) = lake_eval_ns(test4)

#------------------------------------------------------------------------------
# Make stationary plots
# -----------------------------------------------------------------------------


ind = np.arange(1,99,1)

import seaborn as sns
import matplotlib.animation as animation
from IPython.display import HTML

sns.set_style("white")

# Begin plotting stationary stuff

# Plot test case 1 discharge
fig1 = plt.figure(figsize=(9,6), dpi=150)
line1 = plt.plot(range(ind[0]), test1[0:ind[0]], '-', linewidth=2)
plt.xlim((0,100))
plt.ylim((0,0.10))
plt.ylabel('Phosphate/year')
plt.xlabel('Year since implementation')
plt.title('Constant release of 0.01 units P/year')


# Animate it
# Change to make an animated discharge strategy
def update1(frame_number):
    # update the data

    line1 = plt.plot(range(ind[frame_number]), test1[0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    return line1

base = 'Figure 1'
animated1 = animation.FuncAnimation(fig1, update1, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated1.save('./{}.mp4'.format(base))
plt.close(animated1._fig)
HTML(animated1.to_html5_video())



#------------------------------------------------------------------------------
# Plot Natural phosphorous input
fig1a = plt.figure(figsize=(9,6), dpi=150)
line1 = plt.plot(range(ind[0]), nI1_stat[0,0:ind[0]], '-', linewidth=2)
plt.xlim((0,100))
plt.ylim((0,np.max(nI1_stat)*1.1))
plt.ylabel('Phosphate/year')
plt.xlabel('Year since implementation')
plt.title('Natural phosphorous inputs')


# Animate it
# Change to make an animated discharge strategy
def update1a(frame_number):
    # update the data

    line1 = plt.plot(range(ind[frame_number]), nI1_stat[0,0:ind[frame_number]], 'o', color=plt.cm.viridis(0/4), linewidth=2)
    return line1

base = 'Figure 1a'
animated1a = animation.FuncAnimation(fig1a, update1a, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated1a.save('./{}.mp4'.format(base))
plt.close(animated1a._fig)
HTML(animated1a.to_html5_video())

#------------------------------------------------------------------------------
# Plot Natural P and Factory discharge together
fig1b = plt.figure(figsize=(9,6), dpi=150)
line1a = plt.plot(range(ind[0]), nI1_stat[0,0:ind[0]], '-', linewidth=2)
line1b = plt.plot(range(ind[0]), test1[0:ind[0]], '-', linewidth=2)
plt.xlim((0,100))
plt.ylim((0,np.max(nI1_stat)*1.1))
plt.ylabel('Phosphate/year')
plt.xlabel('Year since implementation')
plt.title('Natural phosphorous inputs')


# Animate it
# Change to make an animated discharge strategy
def update1b(frame_number):
    # update the data

    line1a = plt.plot(range(ind[frame_number]), nI1_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    line1b = plt.plot(range(ind[frame_number]), test1[0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    return line1a, line1b

base = 'Figure 1b'
animated1b = animation.FuncAnimation(fig1b, update1b, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated1b.save('./{}.mp4'.format(base))
plt.close(animated1b._fig)
HTML(animated1b.to_html5_video())

#------------------------------------------------------------------------------
# Plot Total P 
fig1c = plt.figure(figsize=(9,6), dpi=150)
line1 = plt.plot(range(ind[0]), (test1[0:ind[0]] + nI1_stat[0,0:ind[0]]), '-', linewidth=2)
plt.xlim((0,100))
plt.ylim((0,np.max(test1 + nI1_stat)*1.1))
plt.ylabel('Phosphate/year')
plt.xlabel('Year since implementation')
plt.title('Total phosphorous inputs')


# Animate it
# Change to make an animated discharge strategy
def update1c(frame_number):
    # update the data

    line1 = plt.plot(range(ind[frame_number]), (test1[0:ind[frame_number]] + nI1_stat[0,0:ind[frame_number]]), '-', color=plt.cm.viridis(0/4), linewidth=2)
    
    return line1

base = 'Figure 1c'
animated1c = animation.FuncAnimation(fig1c, update1c, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated1c.save('./{}.mp4'.format(base))
plt.close(animated1c._fig)
HTML(animated1c.to_html5_video())

#------------------------------------------------------------------------------
# Plot Taccumulating utility
fig1d = plt.figure(figsize=(9,6), dpi=150)
line1 = plt.plot(range(ind[0]), dU1_stat[0,0:ind[0]], '-', linewidth=2)
plt.xlim((0,100))
plt.ylim((0,np.max(dU1_stat)*1.1))
plt.ylabel('Economic Benefits')
plt.xlabel('Year since implementation')
plt.title('Cumulative Economic Benefits')


# Animate it
# Change to make an animated discharge strategy
def update1d(frame_number):
    # update the data

    line1 = plt.plot(range(ind[frame_number]), dU1_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    
    return line1

base = 'Figure 1d'
animated1d = animation.FuncAnimation(fig1d, update1d, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated1d.save('./{}.mp4'.format(base))
plt.close(animated1d._fig)
HTML(animated1d.to_html5_video())

#-----------------------------------------------------------------------------
# Plot test case 1 lake phosphorous with natural phosphorous

fig1e = plt.figure(figsize=(9,6), dpi=150)
# First: Stochastic natural discharge
ax1 = fig1e.add_subplot(211, autoscale_on='false')
line1 = ax1.plot(range(ind[0]), (test1[0:ind[0]] + nI1_stat[0,0:ind[0]]), '-', color=plt.cm.viridis(0/4), linewidth=2)
ax1.set_xlim((0,100))
ax1.set_ylim((0,np.max(test1 + nI1_stat)*1.1))
ax1.set_ylabel('Phosphate/year')
ax1.set_title('Total Phosphorous Inputs')

# Second: Actual lake phosphorous concentration
ax2 = fig1e.add_subplot(212, autoscale_on='false')
line2 = ax2.plot(range(ind[0]), dP1_stat[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
ax2.set_xlim((0,100))
ax2.set_ylim((0,np.max(dP1_stat)*1.1))
ax2.set_ylabel('Phosphate/year')
ax2.set_xlabel('Year since implementation')
ax2.set_title('Lake Phosphorous Concentration')

# Animate it
def update1e(frame_number):
    # update the data
    line1 = ax1.plot(range(ind[frame_number]), (test1[0:ind[frame_number]] + nI1_stat[0,0:ind[frame_number]]), '-', color=plt.cm.viridis(0/4), linewidth=2)
    line2 = ax2.plot(range(ind[frame_number]), dP1_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    return line1, line2

base = 'Figure 1e'
animated1e = animation.FuncAnimation(fig1e, update1e, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated1e.save('./{}.mp4'.format(base))
plt.close(animated1e._fig)
HTML(animated1e.to_html5_video())
#------------------------------------------------------------------------------
# Reliability bar

left = (1)
height = (rel1_stat)
tick_label = ('0.01')
bar_colors = ('darkslateblue')

fig1f = plt.figure(figsize=(9,6), dpi=150)
# Strategy 1: Still
line = plt.bar(left=left, height=height, tick_label=tick_label, color=bar_colors)
#ax.set_xlim((0,100))
plt.ylim((0,np.max((rel1_stat, rel2_stat, rel3_stat, rel4_stat))*1.1))
plt.ylabel('Ecological Reliability')
plt.xlabel('Factory phosphorous discharge/year')

plt.ylim((0,1.0))
plt.ylabel('Reliability')
plt.xlabel('0.01 Phosphorous/year')

fig1f.savefig('Figure_1f.png', figsize=(9,6), dpi=150, bbox_inches='tight')


#-----------------------------------------------------------------------------
# Plot test case 1, 2, 3, and 4 discharge
fig2 = plt.figure(figsize=(9,6), dpi=150)
line1 = plt.plot(range(ind[0]), test1[0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
line2 = plt.plot(range(ind[0]), test2[0:ind[0]], '-', color=plt.cm.viridis(1/4), linewidth=2)
line3 = plt.plot(range(ind[0]), test3[0:ind[0]], '-', color=plt.cm.viridis(2/4), linewidth=2)
line4 = plt.plot(range(ind[0]), test4[0:ind[0]], '-', color=plt.cm.viridis(3/4), linewidth=2)
plt.xlim((0,100))
plt.ylim((0,pho1_stat*1.1))
plt.ylabel('Discharge (Phosphate/year)')
plt.xlabel('Year since implementation')
plt.title('Four alternative constant discharge strategies')

# Animate it
def update2(frame_number):
    # update the data

    line1 = plt.plot(range(ind[frame_number]), test1[0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    line2 = plt.plot(range(ind[frame_number]), test2[0:ind[frame_number]], '-', color=plt.cm.viridis(1/4), linewidth=2)
    line3 = plt.plot(range(ind[frame_number]), test3[0:ind[frame_number]], '-', color=plt.cm.viridis(2/4), linewidth=2)
    line4 = plt.plot(range(ind[frame_number]), test4[0:ind[frame_number]], '-', color=plt.cm.viridis(3/4), linewidth=2)
    return line1, line2, line3, line4

base = 'Figure 2'
animated2 = animation.FuncAnimation(fig2, update2, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated2.save('./{}.mp4'.format(base))
plt.close(animated2._fig)
HTML(animated2.to_html5_video())

#-----------------------------------------------------------------------------
# Plot test case 1 lake phosphorous
fig3 = plt.figure(figsize=(9,6), dpi=150)
line = plt.plot(range(ind[0]), dP1_stat[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
plt.xlim((0,100))
plt.ylim((0,np.max(dP1_stat)*1.1))
plt.ylabel('Phosphorous concentration')
plt.xlabel('Year since implementation')
plt.title('Lake phosphorous concentration')

# Animate it
def update3(frame_number):
    # update the data
    line = plt.plot(range(ind[frame_number]), dP1_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    return line

base = 'Figure 3'
animated3 = animation.FuncAnimation(fig3, update3, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated3.save('./{}.mp4'.format(base))
plt.close(animated3._fig)
HTML(animated3.to_html5_video())



#-----------------------------------------------------------------------------
# Plot test case 1 lake phosphorous AND natural P with well-characterized uncertainty

fig5 = plt.figure(figsize=(9,6), dpi=150)
# First: Stochastic natural discharge
ax1 = fig5.add_subplot(211, autoscale_on='false')
line1a = ax1.plot(range(ind[0]), nI1_stat[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)

N = np.max(ind)
lines1b = [ax1.plot([], [], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax1.set_xlim((0,100))
ax1.set_ylim((0,np.max(test1 + nI1_stat)*1.1))
ax1.set_ylabel('Phosphate/year')
ax1.set_title('Total Phosphorous Inputs')

# Second: Actual lake phosphorous concentration
ax2 = fig5.add_subplot(212, autoscale_on='false')
line2a = ax2.plot(range(ind[0]), dP1_stat[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
lines2b = [ax2.plot([], [], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax2.set_xlim((0,100))
ax2.set_ylim((0,np.max(dP1_stat)*1.1))
ax2.set_ylabel('Phosphate')
ax2.set_xlabel('Year since implementation')
ax2.set_title('Lake Phosphorous Concentration')


# Animate it
def update5(frame_number):

    # update the data
    x1a = np.arange(0,ind[frame_number],1)
    y1a = test1[0:ind[frame_number]] + nI1_stat[0,0:ind[frame_number]]

    line1a = ax1.plot(x1a, y1a, '-', color=plt.cm.viridis(0/4), linewidth=2)
    
    for j, line in enumerate(lines1b):
        line.set_data(range(ind[frame_number]), (test1[0:ind[frame_number]] + nI1_stat[j,0:ind[frame_number]]))
        
    for j, line in enumerate(lines2b):
        line.set_data(range(ind[frame_number]), dP1_stat[j,0:ind[frame_number]])
    
    x2a = np.arange(0,ind[frame_number],1)
    y2a = dP1_stat[0,0:ind[frame_number]]

    line2a = ax2.plot(x2a, y2a, '-', color=plt.cm.viridis(0/4), linewidth=2)
    
    return line1a, line2a, tuple(lines1b), tuple(lines2b)




base = 'Figure 5'
animated5 = animation.FuncAnimation(fig5, update5, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated5.save('./{}.mp4'.format(base))
plt.close(animated5._fig)
HTML(animated5.to_html5_video())

#-----------------------------------------------------------------------------
# Add test case 2 lake phosphorous AND natural P with well-characterized uncertainty

fig6 = plt.figure(figsize=(9,6), dpi=150)
# First: Stochastic natural discharge
ax1 = fig6.add_subplot(211, autoscale_on='false')
# Strategy 1
line1a, = ax1.plot(range(100), (test1 + nI1_stat[0,:]), '-', color=plt.cm.viridis(0/4), linewidth=2)

N = np.max(ind)
lines1b = [ax1.plot([], [], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]

# Strategy 2
line2a = ax1.plot(range(ind[0]), (test2[0:ind[0]] + nI2_stat[0,0]), '-', color=plt.cm.viridis(1/4), linewidth=2)
N = np.max(ind)
lines2b = [ax1.plot([], [], '-', color=plt.cm.viridis(1/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax1.set_xlim((0,100))
ax1.set_ylim((0,np.max(test2 + nI2_stat)*1.5))
ax1.set_ylabel('Phosphate/year')
ax1.set_title('Total Phosphorous Inputs')


# Second: Actual lake phosphorous concentration
ax2 = fig6.add_subplot(212, autoscale_on='false')
# Strategy 1: Still
line3a = ax2.plot(range(100), dP1_stat[0,:], '-', color=plt.cm.viridis(0/4), linewidth=2)
for k in range(len(ind)):
    line3, = ax2.plot(range(100), dP1_stat[k,:], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)
# Strategy 2: Animated
line4a = ax2.plot(range(ind[0]), dP2_stat[0,0:ind[0]], '-', color=plt.cm.viridis(1/4), linewidth=2)

N = np.max(ind)
lines4b = [ax2.plot([], [], '-', color=plt.cm.viridis(1/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]

ax2.set_xlim((0,100))
ax2.set_ylim((0,np.max(dP2_stat)*1.1))
ax2.set_title('Lake Phosphorous Concentration')
ax2.set_ylabel('Phosphate/year')
ax2.set_xlabel('Year since implementation')
plt.show()

# Animate it
def update6(frame_number):

    # update the data
    x2a = np.arange(0,ind[frame_number],1)
    y2a = test2[0:ind[frame_number]] + nI2_stat[0,0:ind[frame_number]]

    line2a = ax1.plot(x2a, y2a, '-', color=plt.cm.viridis(1/4), linewidth=2)
    
    for j, line in enumerate(lines1b):
        line.set_data(range(ind[frame_number]), (test1[0:ind[frame_number]] + nI1_stat[j,0:ind[frame_number]]))
        
    for j, line in enumerate(lines2b):
        line.set_data(range(ind[frame_number]), (test2[0:ind[frame_number]] + nI2_stat[j,0:ind[frame_number]]))
        

    x4a = np.arange(0,ind[frame_number],1)
    y4a = dP2_stat[0,0:ind[frame_number]]

    line4a = ax2.plot(x4a, y4a, '-', color=plt.cm.viridis(1/4), linewidth=2)
    for j, line in enumerate(lines4b):
        line.set_data(range(ind[frame_number]), dP2_stat[j,0:ind[frame_number]])
    
    return line2a, tuple(lines1b), tuple(lines2b), line4a, lines4b


base = 'Figure 6'
animated6 = animation.FuncAnimation(fig6, update6, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated6.save('./{}.mp4'.format(base))
plt.close(animated6._fig)
HTML(animated6.to_html5_video())

#-----------------------------------------------------------------------------
# All test cases with discharge, lake phosphorous, and natural phosphorous
# Also reliability + accumulating utility to the side

left = (1, 2, 3, 4)
height = (rel1_stat, rel2_stat, rel3_stat, rel4_stat)
tick_label = ('0.01', '0.05', '0.07', '0.025')
bar_colors = ('k', 'b', 'g', 'm')
#bar_colors = np.empty(4)
#bar_colors[0] = plt.cm.viridis(0/4)
#bar_colors[1] = plt.cm.viridis(1/4)
#bar_colors[2] = plt.cm.viridis(2/4)
#bar_colors[3] = plt.cm.viridis(3/4)

fig7 = plt.figure(figsize=(9,6), dpi=150)
# Top left: Stochastic natural discharge with your discharge
ax1 = fig7.add_subplot(221, autoscale_on='false')
# Strategy 1
line1a = ax1.plot(range(100), (test1 + nI1_stat[0,:]), '-', color=plt.cm.viridis(0/4), linewidth=2)
for k in range(len(ind)):
    line1b, = ax1.plot(range(100), (test1 + nI1_stat[k,:]), '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)
# Strategy 2
line1c = ax1.plot(range(100), (test2 + nI2_stat[0,:]), '-', color=plt.cm.viridis(1/4), linewidth=2)
for k in range(len(ind)):
    line1d, = ax1.plot(range(100), (test2 + nI2_stat[k,:]), '-', color=plt.cm.viridis(1/4), linewidth=0.5, alpha=0.05)
# Strategy 3
line1e = ax1.plot(range(ind[0]), (test3[0:ind[0]] + nI3_stat[0,0:ind[0]]), '-', color=plt.cm.viridis(2/4), linewidth=2)
N = np.max(ind)
lines1f = [ax1.plot([], [], '-', color=plt.cm.viridis(2/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
# Strategy 4
line1g = ax1.plot(range(ind[0]), (test4[0:ind[0]] + nI4_stat[0,0:ind[0]]), '-', color=plt.cm.viridis(3/4), linewidth=2)
N = np.max(ind)
lines1h = [ax1.plot([], [], '-', color=plt.cm.viridis(3/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax1.set_xlim((0,100))
ax1.set_ylim((0,np.max(0.075 + nI1_stat)*1.5))
ax1.set_ylabel('Total phosphorous inputs')
#ax1.set_xlabel('Year since implementation')

# Bottom left: Lake phosphorous concentration
ax2 = fig7.add_subplot(223, autoscale_on='false')
# Strategy 1: Still
line2a = ax2.plot(range(100), dP1_stat[0,:], '-', color=plt.cm.viridis(0/4), linewidth=2)
for k in range(len(ind)):
    line2b, = ax2.plot(range(100), dP1_stat[k,:], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)
# Strategy 2: Animated
line2c = ax2.plot(range(100), dP2_stat[0,:], '-', color=plt.cm.viridis(1/4), linewidth=2)
for k in range(len(ind)):
    line2d, = ax2.plot(range(100), dP2_stat[k,:], '-', color=plt.cm.viridis(1/4), linewidth=0.5, alpha=0.05)
# Strategy 3
line2e = ax2.plot(range(ind[0]), dP3_stat[0,0:ind[0]], '-', color=plt.cm.viridis(2/4), linewidth=2)
N = np.max(ind)
lines2f = [ax2.plot([], [], '-', color=plt.cm.viridis(2/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
# Strategy 4
line2g = ax2.plot(range(ind[0]), dP4_stat[0,0:ind[0]], '-', color=plt.cm.viridis(3/4), linewidth=2)
N = np.max(ind)
lines2h = [ax2.plot([], [], '-', color=plt.cm.viridis(3/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax2.set_xlim((0,100))
ax2.set_ylim((0,np.max(dP3_stat)*1.1))
ax2.set_ylabel('Lake Phosphorous Concentration')
#ax2.set_xlabel('Year since implementation')

# Top right: Accumulating utility (animated)
ax3 = fig7.add_subplot(222, autoscale_on='false')
# Strategy 1
line3a = ax3.plot(range(ind[0]), dU1_stat[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
# Strategy 2
line3b = ax3.plot(range(ind[0]), dU2_stat[0,0:ind[0]], '-', color=plt.cm.viridis(1/4), linewidth=2)
# Strategy 2
line3c = ax3.plot(range(ind[0]), dU3_stat[0,0:ind[0]], '-', color=plt.cm.viridis(2/4), linewidth=2)
# Strategy 2
line3d = ax3.plot(range(ind[0]), dU4_stat[0,0:ind[0]], '-', color=plt.cm.viridis(3/4), linewidth=2)
ax3.set_xlim((0,100))
ax3.set_ylim((0,np.max((uti1_stat, uti2_stat, uti3_stat, uti4_stat))*1.1))
ax3.set_ylabel('Accumulated Economic Benefits')
ax3.set_xlabel('Year since implementation')

# Bottom right: Reliability of each strategy
ax4 = fig7.add_subplot(224, autoscale_on='false')
# Strategy 1: Still
line4 = ax4.bar(left=left, height=height, tick_label=tick_label, color=bar_colors)
#ax.set_xlim((0,100))
ax4.set_ylim((0,np.max((rel1_stat, rel2_stat, rel3_stat, rel4_stat))*1.1))
ax4.set_ylabel('Ecological Reliability')
ax4.set_xlabel('Factory phosphorous discharge/year')

# Animate it
def update7(frame_number):
    # update the data
    # Strategy 3
    line1e = ax1.plot(range(ind[frame_number]), (test3[0:ind[frame_number]] + nI3_stat[0,0:ind[frame_number]]), '-', color=plt.cm.viridis(2/4), linewidth=2)
    for j, line in enumerate(lines1f):
        line.set_data(range(ind[frame_number]), (test3[0:ind[frame_number]] + nI3_stat[j,0:ind[frame_number]]))
    
    line1g = ax1.plot(range(ind[frame_number]), (test4[0:ind[frame_number]] + nI4_stat[0,0:ind[frame_number]]), '-', color=plt.cm.viridis(3/4), linewidth=2)
    for j, line in enumerate(lines1h):
        line.set_data(range(ind[frame_number]), (test4[0:ind[frame_number]] + nI4_stat[j,0:ind[frame_number]]))
    
    
    line2e, = ax2.plot(range(ind[frame_number]), dP3_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(2/4), linewidth=2)
    for j, line in enumerate(lines2f):
        line.set_data(range(ind[frame_number]), dP3_stat[j,0:ind[frame_number]])
    
    line2g = ax2.plot(range(ind[frame_number]), dP4_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(3/4), linewidth=2)
    for j, line in enumerate(lines2h):
        line.set_data(range(ind[frame_number]), dP4_stat[j,0:ind[frame_number]])
        
    
    line3a = ax3.plot(range(ind[frame_number]), dU1_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    # Strategy 2
    line3b = ax3.plot(range(ind[frame_number]), dU2_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(1/4), linewidth=2)
    # Strategy 2
    line3c = ax3.plot(range(ind[frame_number]), dU3_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(2/4), linewidth=2)
    # Strategy 2
    line3d = ax3.plot(range(ind[frame_number]), dU4_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(3/4), linewidth=2)
           
    return line1e, lines1f, line1g, lines1h, line2e, lines2f, line2g, lines2h, line3a, line3b, line3c, line3d
base = 'Figure 7'
animated7 = animation.FuncAnimation(fig7, update7, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated7.save('./{}.mp4'.format(base))
plt.close(animated7._fig)
HTML(animated7.to_html5_video())

#------------------------------------------------------------------------------
# Define the Rhodium model for optimization
#------------------------------------------------------------------------------

def lake_problem_2D(pollution_limit,
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean = 0.02,     # mean of natural inflows
         stdev = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100): # monte carlo sampling of natural inflows)
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(pollution_limit)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(pollution_limit)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in range(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)

    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    # intertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return (utility, reliability)

# Make a Rhodium model object

model2D = Model(lake_problem_2D)

# define all parameters to the model that we will be studying
model2D.parameters = [Parameter("pollution_limit"),
                    Parameter("b"),
                    Parameter("q"),
                    Parameter("mean"),
                    Parameter("stdev"),
                    Parameter("delta")]

# define the model outputs
model2D.responses = [Response("utility", Response.MAXIMIZE),
                     Response("reliability", Response.MAXIMIZE)]

# define any constraints (can reference any parameter or response by name)
#model3.constraints = [Constraint("reliability >= 0.80")]

# some parameters are levers that we control via our policy
model2D.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

# some parameters are exogeneous uncertainties, and we want to better
# understand how these uncertainties impact our model and decision making
# process
model2D.uncertainties = [UniformUncertainty("b", 0.1, 0.45),
                       UniformUncertainty("q", 2.0, 4.5),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005),
                       UniformUncertainty("delta", 0.93, 0.99)]


#------------------------------------------------------------------------------
# Begin plotting
#------------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML

# Use Seaborn settings for pretty plots
sns.set_style("white")

# viewing angle
angle = 120
genind = np.linspace(1, 2501, num=51)

trans_value = 0.6
cmap = plt.get_cmap('plasma')
norm = mpl.colors.Normalize(vmin=np.min(genind), vmax=np.max(genind))

gens_cmap = mpl.cm.plasma
gens_smap = mpl.cm.ScalarMappable(cmap=gens_cmap, norm=norm)
gens_smap.set_array([])

heart = '*'

# font sizes
fs = 20
fs2 = 16

utimin = 0
utimax = 2.6
phomin = 0
phomax = 2.3
relmin = 0
relmax = 1.1

utilims = (utimin, utimax)
rellims = (relmin, relmax)

# Random discharge sequences < 0.01
pollution_decisions = np.random.uniform(0, 0.1, size=(100,100))

pho_naive_1 = np.empty((100,1))
uti_naive_1 = np.empty((100,1))
rel_naive_1 = np.empty((100,1))

pho_naive_2 = np.empty((100,1))
uti_naive_2 = np.empty((100,1))
rel_naive_2 = np.empty((100,1))

pho_naive_3 = np.empty((100,1))
uti_naive_3 = np.empty((100,1))
rel_naive_3 = np.empty((100,1))


output_uti_1 = list()
output_rel_1 = list()

for k in range(len(pollution_decisions)):
    (uti_naive_1[k], rel_naive_1[k]) = lake_problem_2D(pollution_decisions[k,:])
    (uti_naive_2[k], rel_naive_2[k]) = lake_problem_2D(pollution_decisions[k,:], b=0.15, q=2.0, delta=0.94, mean=0.02, stdev=0.001)
    (uti_naive_3[k], rel_naive_3[k]) = lake_problem_2D(pollution_decisions[k,:], b=0.42, q=3.0, delta=0.99, mean=0.01, stdev=0.001)

for i in range(len(genind)):
    k = genind[i]
    print(k)
    
    # Perform optimization for specified number of generations
    name_output1 = optimize(model2D, "NSGAII", int(k))
    
    output_uti_1.append(name_output1['utility'])
    output_rel_1.append(name_output1['reliability'])
    
    print(output_uti_1)
    
#------------------------------------------------------------------------------
# Generate 3D plots
#------------------------------------------------------------------------------

# Empty setup
fig = plt.figure(figsize=(9,6), dpi=150)

plt.scatter(range(5), (0.0, 0.001, 0.001, 0.001, 0.001), marker=None, c=None)

plt.xlim(utilims)
plt.ylim(rellims)

plt.yticks([])
plt.xticks([])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

fig.savefig('Fig_2D_0.png', figsize=(9,6), dpi=150, bbox_inches='tight')
plt.show()


# Add one direction of preference
fig = plt.figure(figsize=(9,6), dpi=150)

plt.xlabel('Economic Benefits')

plt.xlim(utilims)
plt.ylim(rellims)

plt.yticks([])
plt.xticks([])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

fig.savefig('Fig_2D_1.png', figsize=(9,6), dpi=150, bbox_inches='tight')
plt.show()

# Add another direction of preference
fig = plt.figure(figsize=(9,6), dpi=150)

plt.annotate('', 
             xy=(np.max(utilims),0.0), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

plt.xlabel('Economic Benefits')

plt.xlim(utilims)
plt.ylim(rellims)

plt.yticks([])
plt.xticks([])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

fig.savefig('Fig_2D_2.png', figsize=(9,6), dpi=150, bbox_inches='tight')
plt.show()

# Add ideal point
fig = plt.figure(figsize=(9,6), dpi=150)

plt.annotate('', 
             xy=(np.max(utilims),0.0), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

plt.annotate('', 
             xy=(0.0,np.max(rellims)), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))


plt.xlabel('Economic Benefits')
plt.ylabel('Reliability')

plt.xlim(utilims)
plt.ylim(rellims)

plt.yticks([])
plt.xticks([])

plt.scatter(2.5,1.0, marker=heart, c='k', s=80, linewidths=4,linestyle='None')

for spine in plt.gca().spines.values():
    spine.set_visible(False)

fig.savefig('Fig_2D_3.png', figsize=(9,6), dpi=150, bbox_inches='tight')
plt.show()

# Add a single alternative
x = 5

fig = plt.figure(figsize=(9,6), dpi=150)

plt.annotate('', 
             xy=(np.max(utilims),0.0), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

plt.annotate('', 
             xy=(0.0,np.max(rellims)), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

plt.xlabel('Economic Benefits')
plt.ylabel('Reliability')

plt.xlim(utilims)
plt.ylim(rellims)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.yticks([])
plt.xticks([])

plt.scatter(2.5,1.0, marker=heart, c='k', s=80, linewidths=4,linestyle='None')

plt.scatter(uti_naive_1[x], rel_naive_1[x], alpha=trans_value, c = plt.cm.plasma(0/4), lw = 0.2)

fig.savefig('Fig_2D_4.png', figsize=(9,6), dpi=150, bbox_inches='tight')
plt.show()

# Add all naive strategies
fig = plt.figure(figsize=(9,6), dpi=150)

plt.annotate('', 
             xy=(np.max(utilims),0.0), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

plt.annotate('', 
             xy=(0.0,np.max(rellims)), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

plt.xlabel('Economic Benefits')
plt.ylabel('Reliability')

plt.xlim(utilims)
plt.ylim(rellims)

plt.yticks([])
plt.xticks([])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.scatter(2.5,1.0, marker=heart, c='k', s=80, linewidths=4,linestyle='None')

plt.scatter(uti_naive_1, rel_naive_1, alpha=trans_value, c = plt.cm.plasma(0/4), lw = 0.2)

fig.savefig('Fig_2D_5.png', figsize=(9,6), dpi=150, bbox_inches='tight')
plt.show()

# Highlight the dominant strategies

def dominates(row, rowCandidate):
    return (row[0] >= rowCandidate[0]) and (row[1] >= rowCandidate[1])

def cull(pts, dominates):
    dominated = list()
    cleared = list()
    for k in range(len(pts)):
        point = pts[k]
        doms = 0
        for j in range(len(pts)):
            if dominates(pts[j], point): 
                doms = doms + 1 # How many other points dominate our point?
                if dominates(point, pts[j]): doms = doms - 1    # Subtract if our point also dominates this point (they are the same)
        # If no other points dominate our point, add the point to the list of non-dominated points.
        if doms == 0: 
            cleared.append(point)
        # If at least one point dominates our point, add this point to the "dominated" list.
        elif doms > 0: 
            dominated.append(point)
        doms = 0
    return(cleared, dominated)

def pareto_sort(uti, rel, dominates):
    output_points = np.column_stack((np.asarray(uti), np.asarray(rel)))
    output_pts = list()
    for k in range(len(uti)):
        output_pts.append(output_points[k,:])
    
    (dominant, dominated) = cull(output_pts, dominates)
    dominant = np.asarray(dominant)
    dominated = np.asarray(dominated)

    if len(dominant) > 0:
        uti_dominant = dominant[:,0]
        rel_dominant = dominant[:,1]
    else:
        uti_dominant = np.asarray(list())
        rel_dominant = np.asarray(list())
        
    if len(dominated) > 0:
        uti_dominated = dominated[:,0]
        rel_dominated = dominated[:,1]
    else:
        uti_dominated = np.asarray(list())
        rel_dominated = np.asarray(list())
    
    return(uti_dominant, rel_dominant, uti_dominated, rel_dominated)
    
(uti_dominant_naive1, rel_dominant_naive1, uti_dominated_naive1, rel_dominated_naive1) = pareto_sort(np.asarray(uti_naive_1),
            np.asarray(rel_naive_1),
            dominates)
    
fig = plt.figure(figsize=(9,6), dpi=150)

plt.annotate('', 
             xy=(np.max(utilims),0.0), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

plt.annotate('', 
             xy=(0.0,np.max(rellims)), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

plt.xlabel('Economic Benefits')
plt.ylabel('Reliability')

plt.xlim(utilims)
plt.ylim(rellims)

plt.yticks([])
plt.xticks([])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.scatter(2.5,1.0, marker=heart, c='k', s=80, linewidths=4,linestyle='None')

plt.scatter(uti_dominated_naive1, rel_dominated_naive1, alpha=trans_value, c = plt.cm.plasma(0/4), lw = 0.2)
plt.scatter(uti_dominant_naive1, rel_dominant_naive1, alpha=trans_value, c = plt.cm.plasma(3/4), lw = 0.2)

fig.savefig('Fig_2D_6.png', figsize=(9,6), dpi=150, bbox_inches='tight')
plt.show()

##------------------------------------------------------------------------------
## Perform optimization
##------------------------------------------------------------------------------
#
#genind = np.linspace(1, 2501, num=101)
#
#output_uti_1 = list(uti_naive_1)
#output_rel_1 = list(rel_naive_1)
#
#for i in range(len(genind)):
#    k = genind[i]
#    
#    print(k)
#    
#    # Perform optimization for specified number of generations
#    name_output1 = optimize(model2D, "NSGAII", int(k))
#    
#    output_uti_1.append(name_output1['utility'])
#    output_rel_1.append(name_output1['reliability'])


#------------------------------------------------------------------------------
# Begin plotting
#------------------------------------------------------------------------------


fig = plt.figure(figsize=(9,6), dpi=150)
ax = fig.add_subplot(111)
ax.annotate('', 
             xy=(np.max(utilims),0.0), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

ax.annotate('', 
             xy=(0.0,np.max(rellims)), 
            xycoords='data', 
            xytext=(0.0,0.0),
            arrowprops=dict(facecolor='black', width=3, headwidth=7,headlength=7, connectionstyle='arc3'))

ax.set_xlabel('Economic Benefits')
ax.set_ylabel('Reliability')

ax.set_xlim(utilims)
ax.set_ylim(rellims)

plt.yticks([])
plt.xticks([])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

ax.scatter(2.5,1.0, marker=heart, color='k', s=80,linestyle='None')

ax.scatter(uti_naive_1, rel_naive_1, alpha=trans_value, color = plt.cm.plasma(0/4), lw = 0.2)

trade_x = output_uti_1[0]
trade_y = output_rel_1[0]
line = ax.scatter(trade_x, trade_y, alpha=trans_value, color = plt.cm.plasma(3/4))

def update(frame_number):
    # update the data
    
    
    trade_x = np.asarray(output_uti_1[frame_number])
    trade_y = np.asarray(output_rel_1[frame_number])
    data = np.c_[trade_x, trade_y]

    line.set_offsets(data)
    return line

base = 'Fig_2D_7'
animated_world1 = animation.FuncAnimation(fig, update, 
                                   len(genind),
                                   blit=False)
animated_world1.save('./{}.mp4'.format(base))
plt.close(animated_world1._fig)
HTML(animated_world1.to_html5_video())

#------------------------------------------------------------------------------
# Compare nonstationary and stationary performance
#------------------------------------------------------------------------------
# Plot Natural phosphorous input
fig1a = plt.figure(figsize=(9,6), dpi=150)
line1 = plt.plot(range(ind[0]), nI1_stat[0:ind[0],1], '-', linewidth=2)
line11 = plt.plot(range(ind[0]), nI1_ns[0:ind[0],1], '-', linewidth=2)
plt.xlim((0,100))
plt.ylim((0,np.max(nI1_ns)*1.1))
plt.ylabel('Phosphate/year')
plt.xlabel('Year since implementation')
plt.title('Natural phosphorous inputs')


# Animate it
# Change to make an animated discharge strategy
def update1a(frame_number):
    # update the data

    line1 = plt.plot(range(ind[frame_number]), nI1_stat[0:ind[frame_number],1], '-', color=plt.cm.viridis(0/4), linewidth=2)
    line11 = plt.plot(range(ind[frame_number]), nI1_ns[0:ind[frame_number],1], '--', color=plt.cm.viridis(0/4), linewidth=2)
    return line1, line11

base = 'Figure 1a'
animated1a = animation.FuncAnimation(fig1a, update1a, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated1a.save('./{}.mp4'.format(base))
plt.close(animated1a._fig)
HTML(animated1a.to_html5_video())



#-----------------------------------------------------------------------------
# Plot test case 1 lake phosphorous AND natural P with well-characterized uncertainty

fig5 = plt.figure(figsize=(9,6), dpi=150)
# First: Stochastic natural discharge
ax1 = fig5.add_subplot(211, autoscale_on='false')
line1a = ax1.plot(range(ind[0]), nI1_stat[0:ind[0],0], '-', color=plt.cm.viridis(0/4), linewidth=2)
line1aa = ax1.plot(range(ind[0]), nI1_ns[0:ind[0],0], '-', color=plt.cm.viridis(0/4), linewidth=2)

N = np.max(ind)
lines1b = [ax1.plot([], [], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
lines1bb = [ax1.plot([], [], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax1.set_xlim((0,100))
ax1.set_ylim((0,np.max(test1 + nI1_ns)*1.1))
ax1.set_ylabel('Phosphate/year')
ax1.set_title('Total Phosphorous Inputs')

# Second: Actual lake phosphorous concentration
ax2 = fig5.add_subplot(212, autoscale_on='false')
line2a = ax2.plot(range(ind[0]), dP1_stat[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
lines2b = [ax2.plot([], [], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
line2aa = ax2.plot(range(ind[0]), dP1_ns[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
lines2bb = [ax2.plot([], [], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax2.set_xlim((0,100))
ax2.set_ylim((0,np.max(dP1_ns)*1.1))
ax2.set_ylabel('Phosphate')
ax2.set_xlabel('Year since implementation')
ax2.set_title('Lake Phosphorous Concentration')


# Animate it
def update5(frame_number):

    # update the data
    x1a = np.arange(0,ind[frame_number],1)
    y1a = test1[0:ind[frame_number]] + nI1_stat[0:ind[frame_number],0]
    x1aa = np.arange(0,ind[frame_number],1)
    y1aa = test1[0:ind[frame_number]] + nI1_ns[0:ind[frame_number],0]

    line1a = ax1.plot(x1a, y1a, '-', color=plt.cm.viridis(0/4), linewidth=2)
    line1a = ax1.plot(x1aa, y1aa, '-', color=plt.cm.viridis(0/4), linewidth=2)
    
    for j, line in enumerate(lines1b):
        line.set_data(range(ind[frame_number]), (test1[0:ind[frame_number]] + nI1_stat[0:ind[frame_number],j]))
        
    for j, line in enumerate(lines2b):
        line.set_data(range(ind[frame_number]), dP1_stat[j,0:ind[frame_number]])
        
    for j, line in enumerate(lines1bb):
        line.set_data(range(ind[frame_number]), (test1[0:ind[frame_number]] + nI1_ns[0:ind[frame_number],j]))
        
    for j, line in enumerate(lines2bb):
        line.set_data(range(ind[frame_number]), dP1_ns[j,0:ind[frame_number]])
    
    x2a = np.arange(0,ind[frame_number],1)
    y2a = dP1_stat[0,0:ind[frame_number]]

    line2a = ax2.plot(x2a, y2a, '-', color=plt.cm.viridis(0/4), linewidth=2)
    x2aa = np.arange(0,ind[frame_number],1)
    y2aa = dP1_ns[0,0:ind[frame_number]]

    line2aa = ax2.plot(x2aa, y2aa, '-', color=plt.cm.viridis(0/4), linewidth=2)
    
    return line1a, line1aa, line2a, line2aa, tuple(lines1b), tuple(lines1bb), tuple(lines2b), tuple(lines2bb)




base = 'Figure 5 ns'
animated5 = animation.FuncAnimation(fig5, update5, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated5.save('./{}.mp4'.format(base))
plt.close(animated5._fig)
HTML(animated5.to_html5_video())


#-----------------------------------------------------------------------------
# All test cases with discharge, lake phosphorous, and natural phosphorous
# Also reliability + accumulating utility to the side

left = (1, 2, 3, 4, 5, 6, 7, 8)
height = (rel1_stat, rel1_ns, rel2_stat, rel2_ns, rel3_stat, rel3_ns, rel4_stat, rel4_ns)
tick_label = ('0.01', '0.01*', '0.05', '0.05*', '0.07', '0.07*', '0.025', '0.025*')
bar_colors = ('k', 'k', 'b', 'b', 'g', 'g', 'm', 'm')
#bar_colors = np.empty(4)
#bar_colors[0] = plt.cm.viridis(0/4)
#bar_colors[1] = plt.cm.viridis(1/4)
#bar_colors[2] = plt.cm.viridis(2/4)
#bar_colors[3] = plt.cm.viridis(3/4)

fig7 = plt.figure(figsize=(9,6), dpi=150)
# Top left: Stochastic natural discharge with your discharge
ax1 = fig7.add_subplot(221, autoscale_on='false')
# Strategy 1
line1a = ax1.plot(range(100), (test1 + nI1_stat[:,0]), '-', color=plt.cm.viridis(0/4), linewidth=2)
line1aa = ax1.plot(range(100), (test1 + nI1_ns[:,0]), '-', color=plt.cm.viridis(0/4), linewidth=2)
for k in range(len(ind)):
    line1b, = ax1.plot(range(100), (test1 + nI1_stat[:,k]), '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)
    line1bb, = ax1.plot(range(100), (test1 + nI1_ns[:,k]), '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)
# Strategy 2
line1c = ax1.plot(range(100), (test2 + nI2_stat[:,0]), '-', color=plt.cm.viridis(1/4), linewidth=2)
line1cc = ax1.plot(range(100), (test2 + nI2_ns[:,0]), '-', color=plt.cm.viridis(1/4), linewidth=2)
for k in range(len(ind)):
    line1d, = ax1.plot(range(100), (test2 + nI2_stat[:,k]), '-', color=plt.cm.viridis(1/4), linewidth=0.5, alpha=0.05)
    line1dd, = ax1.plot(range(100), (test2 + nI2_ns[:,k]), '-', color=plt.cm.viridis(1/4), linewidth=0.5, alpha=0.05)
# Strategy 3
line1e = ax1.plot(range(ind[0]), (test3[0:ind[0]] + nI3_stat[0:ind[0],0]), '-', color=plt.cm.viridis(2/4), linewidth=2)
line1ee = ax1.plot(range(ind[0]), (test3[0:ind[0]] + nI3_ns[0:ind[0],0]), '-', color=plt.cm.viridis(2/4), linewidth=2)
N = np.max(ind)
lines1f = [ax1.plot([], [], '-', color=plt.cm.viridis(2/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
lines1ff = [ax1.plot([], [], '-', color=plt.cm.viridis(2/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
# Strategy 4
line1g = ax1.plot(range(ind[0]), (test4[0:ind[0]] + nI4_stat[0:ind[0],0]), '-', color=plt.cm.viridis(3/4), linewidth=2)
line1gg = ax1.plot(range(ind[0]), (test4[0:ind[0]] + nI4_ns[0:ind[0],0]), '-', color=plt.cm.viridis(3/4), linewidth=2)
N = np.max(ind)
lines1h = [ax1.plot([], [], '-', color=plt.cm.viridis(3/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
lines1hh = [ax1.plot([], [], '-', color=plt.cm.viridis(3/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax1.set_xlim((0,100))
ax1.set_ylim((0,np.max(0.075 + nI1_ns)*1.5))
ax1.set_ylabel('Total phosphorous inputs')
#ax1.set_xlabel('Year since implementation')

# Bottom left: Lake phosphorous concentration
ax2 = fig7.add_subplot(223, autoscale_on='false')
# Strategy 1: Still
line2a = ax2.plot(range(100), dP1_stat[0,:], '-', color=plt.cm.viridis(0/4), linewidth=2)
line2aa = ax2.plot(range(100), dP1_ns[0,:], '-', color=plt.cm.viridis(0/4), linewidth=2)
for k in range(len(ind)):
    line2b, = ax2.plot(range(100), dP1_stat[k,:], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)
    line2bb, = ax2.plot(range(100), dP1_ns[k,:], '-', color=plt.cm.viridis(0/4), linewidth=0.5, alpha=0.05)
# Strategy 2: Animated
line2c = ax2.plot(range(100), dP2_stat[0,:], '-', color=plt.cm.viridis(1/4), linewidth=2)
line2cc = ax2.plot(range(100), dP2_ns[0,:], '-', color=plt.cm.viridis(1/4), linewidth=2)
for k in range(len(ind)):
    line2d, = ax2.plot(range(100), dP2_stat[k,:], '-', color=plt.cm.viridis(1/4), linewidth=0.5, alpha=0.05)
    line2dd, = ax2.plot(range(100), dP2_ns[k,:], '-', color=plt.cm.viridis(1/4), linewidth=0.5, alpha=0.05)
# Strategy 3
line2e = ax2.plot(range(ind[0]), dP3_stat[0,0:ind[0]], '-', color=plt.cm.viridis(2/4), linewidth=2)
line2ee = ax2.plot(range(ind[0]), dP3_ns[0,0:ind[0]], '-', color=plt.cm.viridis(2/4), linewidth=2)
N = np.max(ind)
lines2f = [ax2.plot([], [], '-', color=plt.cm.viridis(2/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
lines2ff = [ax2.plot([], [], '-', color=plt.cm.viridis(2/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
# Strategy 4
line2g = ax2.plot(range(ind[0]), dP4_stat[0,0:ind[0]], '-', color=plt.cm.viridis(3/4), linewidth=2)
line2gg = ax2.plot(range(ind[0]), dP4_ns[0,0:ind[0]], '-', color=plt.cm.viridis(3/4), linewidth=2)
N = np.max(ind)
lines2h = [ax2.plot([], [], '-', color=plt.cm.viridis(3/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
lines2hh = [ax2.plot([], [], '-', color=plt.cm.viridis(3/4), linewidth=0.5, alpha=0.05)[0] for _ in range(N)]
ax2.set_xlim((0,100))
ax2.set_ylim((0,np.max(dP3_ns)*1.1))
ax2.set_ylabel('Lake Phosphorous Concentration')
#ax2.set_xlabel('Year since implementation')

# Top right: Accumulating utility (animated)
ax3 = fig7.add_subplot(222, autoscale_on='false')
# Strategy 1
line3a = ax3.plot(range(ind[0]), dU1_stat[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
line3aa = ax3.plot(range(ind[0]), dU1_ns[0,0:ind[0]], '-', color=plt.cm.viridis(0/4), linewidth=2)
# Strategy 2
line3b = ax3.plot(range(ind[0]), dU2_stat[0,0:ind[0]], '-', color=plt.cm.viridis(1/4), linewidth=2)
line3bb = ax3.plot(range(ind[0]), dU2_ns[0,0:ind[0]], '-', color=plt.cm.viridis(1/4), linewidth=2)
# Strategy 2
line3c = ax3.plot(range(ind[0]), dU3_stat[0,0:ind[0]], '-', color=plt.cm.viridis(2/4), linewidth=2)
line3cc = ax3.plot(range(ind[0]), dU3_ns[0,0:ind[0]], '-', color=plt.cm.viridis(2/4), linewidth=2)
# Strategy 2
line3d = ax3.plot(range(ind[0]), dU4_stat[0,0:ind[0]], '-', color=plt.cm.viridis(3/4), linewidth=2)
line3dd = ax3.plot(range(ind[0]), dU4_ns[0,0:ind[0]], '-', color=plt.cm.viridis(3/4), linewidth=2)
ax3.set_xlim((0,100))
ax3.set_ylim((0,np.max((uti1_ns, uti2_ns, uti3_ns, uti4_ns))*1.1))
ax3.set_ylabel('Accumulated Economic Benefits')
ax3.set_xlabel('Year since implementation')

# Bottom right: Reliability of each strategy
ax4 = fig7.add_subplot(224, autoscale_on='false')
# Strategy 1: Still
line4 = ax4.bar(left=left, height=height, tick_label=tick_label, color=bar_colors)
#ax.set_xlim((0,100))
ax4.set_ylim((0,np.max((rel1_ns, rel2_ns, rel3_ns, rel4_ns))*1.1))
ax4.set_ylabel('Ecological Reliability')
ax4.set_xlabel('Factory phosphorous discharge/year')

# Animate it
def update7(frame_number):
    # update the data
    # Strategy 3
    line1e = ax1.plot(range(ind[frame_number]), (test3[0:ind[frame_number]] + nI3_stat[0:ind[frame_number],0]), '-', color=plt.cm.viridis(2/4), linewidth=2)
    for j, line in enumerate(lines1f):
        line.set_data(range(ind[frame_number]), (test3[0:ind[frame_number]] + nI3_stat[0:ind[frame_number],j]))
    
    line1g = ax1.plot(range(ind[frame_number]), (test4[0:ind[frame_number]] + nI4_stat[0:ind[frame_number],0]), '-', color=plt.cm.viridis(3/4), linewidth=2)
    for j, line in enumerate(lines1h):
        line.set_data(range(ind[frame_number]), (test4[0:ind[frame_number]] + nI4_stat[0:ind[frame_number],j]))
    
    line1ee = ax1.plot(range(ind[frame_number]), (test3[0:ind[frame_number]] + nI3_ns[0:ind[frame_number],0]), '-', color=plt.cm.viridis(2/4), linewidth=2)
    for j, line in enumerate(lines1ff):
        line.set_data(range(ind[frame_number]), (test3[0:ind[frame_number]] + nI3_ns[0:ind[frame_number],j]))
    
    line1gg = ax1.plot(range(ind[frame_number]), (test4[0:ind[frame_number]] + nI4_ns[0:ind[frame_number],0]), '-', color=plt.cm.viridis(3/4), linewidth=2)
    for j, line in enumerate(lines1hh):
        line.set_data(range(ind[frame_number]), (test4[0:ind[frame_number]] + nI4_ns[0:ind[frame_number],j]))
    
    line2e, = ax2.plot(range(ind[frame_number]), dP3_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(2/4), linewidth=2)
    for j, line in enumerate(lines2f):
        line.set_data(range(ind[frame_number]), dP3_stat[j,0:ind[frame_number]])
    
    line2g = ax2.plot(range(ind[frame_number]), dP4_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(3/4), linewidth=2)
    for j, line in enumerate(lines2h):
        line.set_data(range(ind[frame_number]), dP4_stat[j,0:ind[frame_number]])
        
    line2ee, = ax2.plot(range(ind[frame_number]), dP3_ns[0,0:ind[frame_number]], '-', color=plt.cm.viridis(2/4), linewidth=2)
    for j, line in enumerate(lines2ff):
        line.set_data(range(ind[frame_number]), dP3_ns[j,0:ind[frame_number]])
    
    line2gg = ax2.plot(range(ind[frame_number]), dP4_ns[0,0:ind[frame_number]], '-', color=plt.cm.viridis(3/4), linewidth=2)
    for j, line in enumerate(lines2hh):
        line.set_data(range(ind[frame_number]), dP4_ns[j,0:ind[frame_number]])
        
    
    line3a = ax3.plot(range(ind[frame_number]), dU1_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    # Strategy 2
    line3b = ax3.plot(range(ind[frame_number]), dU2_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(1/4), linewidth=2)
    # Strategy 2
    line3c = ax3.plot(range(ind[frame_number]), dU3_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(2/4), linewidth=2)
    # Strategy 2
    line3d = ax3.plot(range(ind[frame_number]), dU4_stat[0,0:ind[frame_number]], '-', color=plt.cm.viridis(3/4), linewidth=2)
    
    line3aa = ax3.plot(range(ind[frame_number]), dU1_ns[0,0:ind[frame_number]], '-', color=plt.cm.viridis(0/4), linewidth=2)
    # Strategy 2
    line3bb = ax3.plot(range(ind[frame_number]), dU2_ns[0,0:ind[frame_number]], '-', color=plt.cm.viridis(1/4), linewidth=2)
    # Strategy 2
    line3cc = ax3.plot(range(ind[frame_number]), dU3_ns[0,0:ind[frame_number]], '-', color=plt.cm.viridis(2/4), linewidth=2)
    # Strategy 2
    line3dd = ax3.plot(range(ind[frame_number]), dU4_ns[0,0:ind[frame_number]], '-', color=plt.cm.viridis(3/4), linewidth=2)
           
    return line1e, lines1f, line1g, lines1h, line2e, lines2f, line2g, lines2h, line3a, line3b, line3c, line3d, line1ee, lines1ff, line1gg, lines1hh, line2ee, lines2ff, line2gg, lines2hh, line3aa, line3bb, line3cc, line3dd
base = 'Figure 7'
animated7 = animation.FuncAnimation(fig7, update7, 
                                   len(ind),
                                   blit=False,
                                   interval=15)
animated7.save('./{}.mp4'.format(base))
plt.close(animated7._fig)
HTML(animated7.to_html5_video())

