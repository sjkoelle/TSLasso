import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from collections import OrderedDict
import math
from matplotlib.lines import Line2D
from pylab import rcParams
rcParams['figure.figsize'] = 25, 10

def width(p,w):
    if p > 1.:
        output = 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
    else:
        output = w
    return(output)

def plot_betas(experiments, xaxis, title,filename, gnames,nsel):
    # help(megaman)
    filename = filename+ 'symlog'
    xlabel = r"$\displaystyle \lambda$"
    ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)
    p = experiments[0].p
    n = experiments[0].n
    q = experiments[0].q
    norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
    cmap = plt.cm.rainbow
    nreps = len(experiments.keys())
    maxes = np.zeros(q)
    for k in range(q):
        for l in range(nreps):
            # or j in range(p):
            coeffs = experiments[l].coeffs
            maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    normax = maxes.max()
    if q > 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=cmap(norm(j)), label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[q].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[q].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=cmap(norm(j)), label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
            if k!= q:
                axes[k].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
            if k ==q:
                axes[k].set_title("Combined",fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()
    if q == 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=cmap(norm(j)), label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[q].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[q].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=cmap(norm(j)), label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
            if k!= q:
                axes[k].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
            if k ==q:
                axes[k].set_title("Combined",fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()

    by_label = OrderedDict(zip(labels, handles))
    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
    fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
    plt.suptitle(title, fontsize=55)
    fig.subplots_adjust(right=0.75)
    leg_ax = fig.add_axes([.86, 0.15, 0.05, 0.7])
    leg_ax.axis('off')
    # leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
    leg = leg_ax.legend(by_label.values(), gnames, prop={'size': 25})
    leg.set_title('Torsion', prop={'size': 45})
    for l in leg.get_lines():
        l.set_alpha(1)
    fig.savefig(filename + 'beta_paths_log_n' + str(n) + 'nsel' + str(nsel) + 'nrepss' + str(
        nreps) + 'rigidcombotoohighiter')

def plot_penloss(xs,ys,filename):

    for i in range(ys.shape[0]):
        plt.scatter(xs,ys[i])
    plt.savefig(filename)

def plot_betas_customcolors(experiments, xaxis, title,filename, gnames,nsel,colors, legtitle, color_labels):
    # help(megaman)
    filename = filename+ 'symlog'
    xlabel = r"$\displaystyle \lambda$"
    ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)
    p = experiments[0].p
    n = experiments[0].n
    q = experiments[0].q
    norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
    cmap = plt.cm.rainbow
    nreps = len(experiments.keys())
    maxes = np.zeros(q)
    for k in range(q):
        for l in range(nreps):
            # or j in range(p):
            coeffs = experiments[l].coeffs
            maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    normax = maxes.max()
    if q > 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=colors[j], label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[q].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[q].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=colors[j], label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
            if k!= q:
                axes[k].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
            if k ==q:
                axes[k].set_title("Combined",fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
    fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
    plt.suptitle(title, fontsize=55)
    fig.subplots_adjust(right=0.75)
    leg_ax = fig.add_axes([.86, 0.15, 0.05, 0.7])
    leg_ax.axis('off')
    custom_lines = [Line2D([0], [0], color='red', lw=4), Line2D([0], [0], color='black', lw=4)]
    # leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
    leg = leg_ax.legend(custom_lines, color_labels, prop={'size': 25})
    leg.set_title(legtitle, prop={'size': 45})
    for l in leg.get_lines():
        l.set_alpha(1)
    fig.savefig(filename + 'beta_paths_log_n' + str(n) + 'nsel' + str(nsel) + 'nrepss' + str(
        nreps) + 'rigidcombotoohighiter')


def plot_betas_customcolors2(experiments, xaxis, title,filename, gnames,nsel,colors, legtitle, color_labels):
    # help(megaman)
    filename = filename+ 'symlog'
    xlabel = r"$\displaystyle \lambda$"
    ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)
    p = experiments[0].p
    n = experiments[0].n
    q = experiments[0].q
    norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
    cmap = plt.cm.rainbow
    nreps = len(experiments.keys())
    maxes = np.zeros(q)
    for k in range(q):
        for l in range(nreps):
            # or j in range(p):
            coeffs = experiments[l].coeffs
            maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    normax = maxes.max()
    if q > 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k+1].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k+1].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=colors[j], label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[0].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=colors[j], label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
            if k!= q:
                axes[k+1].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
            if k ==0:
                axes[k].set_title("Combined",fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
    fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
    plt.suptitle(title, fontsize=55)
    fig.subplots_adjust(right=0.75)
    leg_ax = fig.add_axes([.86, 0.15, 0.05, 0.7])
    leg_ax.axis('off')
    custom_lines = [Line2D([0], [0], color='red', lw=4), Line2D([0], [0], color='black', lw=4)]
    # leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
    leg = leg_ax.legend(custom_lines, color_labels, prop={'size': 25})
    leg.set_title(legtitle, prop={'size': 45})
    for l in leg.get_lines():
        l.set_alpha(1)
    fig.savefig(filename + 'beta_paths_log_n' + str(n) + 'nsel' + str(nsel) + 'nrepss' + str(
        nreps) + 'rigidcombotoohighiter')


def plot_betas_combinedfirst(experiments, xaxis, title,filename, gnames,nsel):
    # help(megaman)
    filename = filename+ 'symlog'
    xlabel = r"$\displaystyle \lambda$"
    ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)
    p = experiments[0].p
    n = experiments[0].n
    q = experiments[0].q
    norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
    cmap = plt.cm.rainbow
    nreps = 3
    maxes = np.zeros(q)
    for k in range(q):
        for l in range(nreps):
            # or j in range(p):
            coeffs = experiments[l].coeffs
            maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    normax = maxes.max()
    if q > 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k+1].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k+1].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=cmap(norm(j)), label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[0].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=cmap(norm(j)), label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
                axes[k].set_title("Combined", fontdict={'fontsize': 50})
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
                axes[k].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
    fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
    plt.suptitle(title, fontsize=55)
    fig.subplots_adjust(right=0.75)
    leg_ax = fig.add_axes([.86, 0.15, 0.05, 0.7])
    leg_ax.axis('off')
    # leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
    leg = leg_ax.legend(by_label.values(), gnames, prop={'size': 25})
    leg.set_title('Torsion', prop={'size': 45})
    for l in leg.get_lines():
        l.set_alpha(1)
    fig.savefig(filename + 'beta_paths_log_n' + str(n) + 'nsel' + str(nsel) + 'nrepss' + str(
        nreps) + 'rigidcombotoohighiter')

def plot_betas2(experiments, xaxis, title,filename, gnames,nsel):
    title = ""
    #gnames = np.asarray([r"$\displaystyle g_1$",
    #r"$\displaystyle g_2$",
    #r"$\displaystyle g_3$",
    #r"$\displaystyle g_4$"])
    #def plot_betas(experiments, xaxis, title,filename, gnames,nsel):
    # help(megaman)
    filename = filename+ 'symlog'
    xlabel = r"$\displaystyle \lambda$"
    ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)
    p = experiments[0].p
    n = experiments[0].n
    q = experiments[0].q
    norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
    cmap = plt.cm.rainbow
    nreps = 3
    maxes = np.zeros(q)
    for k in range(q):
        for l in range(nreps):
            # or j in range(p):
            coeffs = experiments[l].coeffs
            maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    normax = maxes.max()
    if q > 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                #print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=cmap(norm(j)), label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[q].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[q].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=cmap(norm(j)), label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
            if k!= q:
                axes[k].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
            if k ==q:
                axes[k].set_title("Combined",fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
    fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
    plt.suptitle(title, fontsize=55)
    fig.subplots_adjust(right=0.75)
    leg_ax = fig.add_axes([.8, 0.15, 0.05, 0.7])
    leg_ax.axis('off')
    # leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
    leg = leg_ax.legend(by_label.values(), gnames, prop={'size': 300/p})
    #leg.set_title('Torsion', prop={'size': Function})
    for l in leg.get_lines():
        l.set_alpha(1)
    #fig.savefig(filename + 'beta_paths_log_n' + str(n) + 'nsel' + str(nsel) + 'nrepss' + str(
    #    nreps) + 'rigidcombotoohighiter')
    fig.savefig(filename + 'beta_paths_n' + str(n) + 'nsel' + str(nsel) + 'nreps' + str(
        nreps))


def plot_betas2reorder(experiments, xaxis, title, filename, gnames, nsel):
    title = ""
    #gnames = np.asarray([r"$\displaystyle g_1$",
    #r"$\displaystyle g_2$",
    #r"$\displaystyle g_3$",
    #r"$\displaystyle g_4$"])
    #def plot_betas(experiments, xaxis, title,filename, gnames,nsel):
    # help(megaman)
    filename = filename+ 'symlog'
    xlabel = r"$\displaystyle \lambda$"
    ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)
    p = experiments[0].p
    n = experiments[0].n
    q = experiments[0].q
    norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
    cmap = plt.cm.rainbow
    nreps = len(experiments.keys())
    maxes = np.zeros(q)
    for k in range(q):
        for l in range(nreps):
            # or j in range(p):
            coeffs = experiments[l].coeffs
            maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    normax = maxes.max()
    if q > 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                #print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k+1].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k+1].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=cmap(norm(j)), label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[0].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=cmap(norm(j)), label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
            if k!= q:
                axes[k+1].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
            if k ==0:
                axes[k].set_title("Combined",fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
    fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
    plt.suptitle(title, fontsize=55)
    fig.subplots_adjust(right=0.75)
    leg_ax = fig.add_axes([.8, 0.15, 0.05, 0.7])
    leg_ax.axis('off')
    # leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
    leg = leg_ax.legend(by_label.values(), gnames, prop={'size': 200/p})
    #leg.set_title('Torsion', prop={'size': Function})
    for l in leg.get_lines():
        l.set_alpha(1)
    #fig.savefig(filename + 'beta_paths_log_n' + str(n) + 'nsel' + str(nsel) + 'nrepss' + str(
    #    nreps) + 'rigidcombotoohighiter')
    fig.savefig(filename + 'beta_paths_n' + str(n) + 'nsel' + str(nsel) + 'nreps' + str(
        nreps))



#def plot_betas3reorder(experiments, xaxis, title,filename, gnames,nsel, legsize = None):

def plot_betas2reorder_tangent(experiments, xaxis, title, filename, gnames, nsel):

    title = ""
    #gnames = np.asarray([r"$\displaystyle g_1$",
    #r"$\displaystyle g_2$",
    #r"$\displaystyle g_3$",
    #r"$\displaystyle g_4$"])
    #def plot_betas(experiments, xaxis, title,filename, gnames,nsel):
    # help(megaman)
    filename = filename+ 'symlog'
    xlabel = r"$\displaystyle \lambda$"
    ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)
    p = experiments[0].p
    legsize = 200 / p 
    n = experiments[0].n
    q = experiments[0].dim
    norm = matplotlib.colors.Normalize(vmin=0, vmax=p)
    cmap = plt.cm.rainbow
    nreps = len(experiments.keys())
    maxes = np.zeros(q)
    for k in range(q):
        for l in range(nreps):
            # or j in range(p):
            coeffs = experiments[l].coeffs
            maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    normax = maxes.max()
    if q > 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                #print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k+1].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k+1].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=cmap(norm(j)), label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[0].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=cmap(norm(j)), label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
            if k!= q:
                axes[k+1].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
            if k ==0:
                axes[k].set_title("Combined",fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()
    if q == 1:
        fig, axes = plt.subplots(1, q + 1, figsize=((15 * q), 15))
        for k in range(q):
            for j in range(p):

                toplot = np.zeros((nreps, len(xaxis)))
                for l in range(nreps):
                    coeffs = experiments[l].coeffs
                    toplot[l, :] = np.linalg.norm(coeffs[:, k, :, j], axis=1)
                    # print('rep = ' ,l ,',' , 'p =',j , coeffs[:,k,:,j].max())
                #print(toplot)
                w = .15
                widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
                # axes[k].boxplot(toplot, positions = xaxis, showfliers=False, patch_artist = True, vert = True ,widths=widths)
                axes[k+1].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
                axes[k+1].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=cmap(norm(j)), label=gnames[j])
                # axes[k].plot(xaxis, , axis = 1), 'go--', linewidth=5, markersize=0, alpha = 1, color = cmap(norm(j)), label = gnames[j])
        for j in range(p):
            toplot = np.zeros((nreps, len(xaxis)))
            for l in range(nreps):
                coeffs = experiments[l].coeffs
                toplot[l, :] = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
            axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[0].plot(xaxis, toplot.mean(axis=0), 'go--', linewidth=5, markersize=0, alpha=1.,
                         color=cmap(norm(j)), label=gnames[j])
        for k in range(1 + q):
            axes[k].tick_params(labelsize=50)
            axes[k].set_xscale('symlog')
            axes[k].set_yscale('symlog')
            axes[k].set_ylim(bottom=0, top=10 * normax)
            #axes[k].set_ylim(bottom=0, top= 2.5*normax)
            if(k ==0):
                tixx = np.hstack([np.asarray([0]),10**np.linspace(math.floor(np.log10(normax)) , math.floor(np.log10(normax)) + 1 ,2)])
                #tixx = 10**np.linspace(math.floor(np.log10(normax)) - 2, math.floor(np.log10(normax)) + 1 ,4)
                #axes[k].set_yticks(tixx)
            if k!=0:
                #axes[k].set_yticks(tixx)
                axes[k].set_yticklabels([])
            if k!= q:
                axes[k+1].set_title(r"$\displaystyle \phi_{{{}}}$".format(k+1),fontdict  = {'fontsize':50})
            if k ==0:
                axes[k].set_title("Combined",fontdict  = {'fontsize':50})
        for k in range(1+q):
            axes[k].grid(True, which="both", alpha = True)

        handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
    fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
    plt.suptitle(title, fontsize=55)
    fig.subplots_adjust(right=0.75)
    leg_ax = fig.add_axes([.8, 0.15, 0.05, 0.7])
    leg_ax.axis('off')
    # leg = leg_ax.legend(by_label.values(), by_label.keys(), prop={'size': 55})
    leg = leg_ax.legend(by_label.values(), gnames, prop={'size': legsize})
    leg = leg_ax.legend(by_label.values(), gnames, prop={'size': 200/p})
    #leg.set_title('Torsion', prop={'size': Function})
    for l in leg.get_lines():
        l.set_alpha(1)
    #fig.savefig(filename + 'beta_paths_log_n' + str(n) + 'nsel' + str(nsel) + 'nrepss' + str(
    #    nreps) + 'rigidcombotoohighiter')
    fig.savefig(filename + 'beta_paths_n' + str(n) + 'nsel' + str(nsel) + 'nreps' + str(
        nreps))