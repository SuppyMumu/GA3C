# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:31:13 2017

@author: valeodevbox
"""

from __future__ import division
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)  
    
MAX_EPISODES = 6000

def prepare_time_axis(hour):
    time = []
    sh0 = hour[0].split(' ')[1].split(':')
    t0 = int(sh0[0]) + float(sh0[1])/60.0 + float(sh0[2])/3600.0
    tprev = t0
    total_time = 0
    dt = 0
    for h in hour[:MAX_EPISODES]:
        sh = h.split(' ')[1].split(':')  
        tt = int(sh[0]) + float(sh[1])/60.0 + float(sh[2])/3600.0
        if tt < tprev: 
            dt = total_time
            t0 = tt
        elif (tt-tprev) > 0.5:
            dt = total_time
            t0 = tt
        else:
            total_time += (tt-tprev)
        t = tt - t0 + dt
        time.append(t)
        tprev = tt
        
    return time


def tmp_copy_with_title(infile,outfile):
    with open(infile) as f1:
        with open(outfile, "w") as f2:
            f2.write("date, reward, steps")
            for line in f1:
                f2.write(line)

def addplot(filename,ax,color,label):
    tmp_copy_with_title(filename,'tmp.txt')

    scores = pd.read_csv('tmp.txt', delimiter=', ')
    hour = scores['date']
    reward = scores['reward'][:MAX_EPISODES]
    time = prepare_time_axis(hour)

    mean_window = 100
    std_window = 200
    r_std = reward.rolling(window = std_window).std()
    r = reward.rolling(window = mean_window).mean()

    ax.plot(time, r, color=color,label=label)
    ax.fill_between(time, r-r_std, r+r_std, color=color, alpha=0.2, linewidth=0)
    
import os, glob, re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=str, default='results.txt')
    args = parser.parse_args()
    
    fig, axarr = plt.subplots(1, sharex=True, figsize=(8, 8))

    results = glob.glob('results_*.txt')
    
    for i in range(len(results)):
        file = results[i]
        name = re.split( 'results_' , os.path.splitext(file)[0])[1]
        print(name)
        addplot(file, axarr, tableau20[i%20],name)

    scores = 'results.txt'
    if os.path.isfile(scores):
        addplot(scores, axarr,tableau20[4],'current.lstm')

    plt.xlabel('hours')
    plt.ylabel('PongDeterministic-V0.score')
    plt.legend(loc='best')
    fig_fname = args.scores + '.png'
    plt.savefig(fig_fname)
    plt.show()
    plt.pause(1)
   

if __name__ == '__main__':
    main()
