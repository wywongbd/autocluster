import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter


def plot_convergence(data, 
                     iterations='max', 
                     title=None, 
                     x_label='Iterations', 
                     y_label='Scores',
                     x_scale='linear',
                     font_color='default',
                     legends=[], 
                     show_plot=True, 
                     save_plot=True, 
                     file_path=None):
    """
    data : list or np.array (1D or 2D)
           ex) [1, 2, 5] -> one graph
               [[3, 5, 6, 9], [2, 5, 7], [1, 2, 3, 4, 5]] -> three graphs
    iterations : number of iterations
                 iterations='max' : maximum iteration number of datasets
                                    ex) [[3, 5, 6, 9], [2, 5, 7]] -> 4
                 iterations='min' : minimum iteration number of datasets
                                    ex) [[3, 5, 6, 9], [2, 5, 7]] -> 3
                 iterations=(int) : set iteration number manually
    legends : set of legends
              ex) ['SMAC', 'Random Search']
              if legends==[], then legends are named numbers 1, 2, 3, ...
    font_color : color of title, x_label, and y_label
                 font_color='default' : white for script file, black for png file
    x_scale : 'linear' or 'log'
    """
    
    if type(data[0]) != list and type(data[0]) != np.ndarray:
        graph_num = 1
        data_in_2D = [data]
    else:
        graph_num = len(data)
        data_in_2D = data
    
    if type(iterations) == int:
        x_len = iterations
    elif iterations == 'min':
        x_len = min([len(sublist) for sublist in data_in_2D])
    else:
        x_len = max([len(sublist) for sublist in data_in_2D])
        
    if len(legends) > graph_num:
        set_legends = legends[:graph_num]
    elif len(legends) < graph_num:
        set_legends = legends + [str(i) for i in range(len(legends) + 1, graph_num + 1)]
    else:
        set_legends = legends
        
    if show_plot or save_plot:
        lin_sp = range(1, x_len + 1)
        colors = cm.nipy_spectral(np.linspace(0, 1, graph_num + 1))
                
        fig = plt.figure(figsize=(12,7))
        for i in range(graph_num):
            data_len = min(len(data_in_2D[i]), x_len)
            data_temp = [1 if datum == float('inf') else datum for datum in data_in_2D[i][:data_len]]
            plt.plot(lin_sp[:data_len], data_temp, linestyle='-', color=colors[i])
        
        plt.legend(set_legends, loc='best')
        plt.ylim(bottom=0)
        plt.xscale(x_scale)
            
        if save_plot:
            if font_color == 'default':
                font_color_ = 'black'
            else:
                font_color_ = font_color
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
            plt.tick_params(axis='x', colors=font_color_)
            plt.tick_params(axis='y', colors=font_color_)
            plt.title(title, fontsize=20, color=font_color_)
            plt.xlabel(x_label, fontsize=15, color=font_color_)
            plt.ylabel(y_label, fontsize=15, color=font_color_)
            if file_path == None:
                fig.savefig('plots/plotconv-{}.png'.format(timestr), bbox_inches='tight', edgecolor='white')
            else:
                fig.savefig(file_path, bbox_inches='tight')
                
        if show_plot:
            if font_color == 'default':
                font_color_ = 'white'
            else:
                font_color_ = font_color
            plt.tick_params(axis='x', colors=font_color_)
            plt.tick_params(axis='y', colors=font_color_)
            plt.title(title, fontsize=20, color=font_color_)
            plt.xlabel(x_label, fontsize=15, color=font_color_)
            plt.ylabel(y_label, fontsize=15, color=font_color_)
            plt.show()
                
        plt.close(fig)
        
        
        
def plot_convergence_average(data, 
                             iterations='max', 
                             title=None, 
                             x_label='Iterations', 
                             y_label='Average Scores',
                             font_color='default',
                             x_scale='linear',
                             area=True,
                             show_plot=True, 
                             save_plot=True, 
                             file_path=None):
    
    """
    data : dictionary of 2D data
    iterations : number of iterations
                 iterations='max' : maximum iteration number of datasets
                                    ex) [[3, 5, 6, 9], [2, 5, 7]] -> 4
                 iterations='min' : minimum iteration number of datasets
                                    ex) [[3, 5, 6, 9], [2, 5, 7]] -> 3
                 iterations=(int) : set iteration number manually
    font_color : color of title, x_label, and y_label
                 font_color='default' : white for script file, black for png file
    x_scale : 'linear' or 'log'
    area : True for standard deviation area, False for only graph
    """

    data_ave = {}
    data_std = {}
    total_len_ls = []
    legends = list(data.keys())
    graph_num = len(legends)
    
    for key, value in data.items():
        len_ls = [len(sublist) for sublist in value]
        len_max = max(len_ls)
        total_len_ls.append(len_max)
        
        freq_dict = Counter(len_ls)
        num_for_iter = len_max * [0]
        for i, j in freq_dict.items():
            for k in range(i):
                num_for_iter[k] += j

        num_for_iter = np.array(num_for_iter)

        data_temp = [sublist + (len_max - len(sublist)) * [0] if len_max > len(sublist) else sublist for sublist in value]
        data_temp = np.array(data_temp)
        data_temp[data_temp == float('inf')] = 1
        data_sum = np.sum(data_temp, axis=0)
        data_sum = data_sum / num_for_iter
        data_ave[key] = data_sum
        
        if area:
            data_temp = [sublist + list(data_sum[len(sublist):]) if len_max > len(sublist) else sublist for sublist in value]
            data_temp = np.array(data_temp)
            data_var = np.var(data_temp, axis=0) * graph_num
            data_var = np.sqrt(data_var / num_for_iter)
            data_std[key] = data_var
        
            
    if type(iterations) == int:
        x_len = iterations
    elif iterations == 'min':
        x_len = min(total_len_ls)
    else:
        x_len = max(total_len_ls)
        
        
    if show_plot or save_plot:
        lin_sp = range(1, x_len + 1)
        colors = cm.nipy_spectral(np.linspace(0, 1, graph_num + 1))
                
        fig = plt.figure(figsize=(12,7))
        for i, key in enumerate(legends):
            data_len = min(len(data_ave[key]), x_len)
            plt.plot(lin_sp[:data_len], data_ave[key][:data_len], linestyle='-', color=colors[i])
            if area:
                plt.fill_between(lin_sp[:data_len], data_ave[key][:data_len] + data_std[key][:data_len],\
                                data_ave[key][:data_len] - data_std[key][:data_len], facecolor=colors[i], alpha=.1)
        
        plt.legend(legends, loc='best')
        plt.ylim(bottom=0)
        plt.xscale(x_scale)
            
        if save_plot:
            if font_color == 'default':
                font_color_ = 'black'
            else:
                font_color_ = font_color
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
            plt.tick_params(axis='x', colors=font_color_)
            plt.tick_params(axis='y', colors=font_color_)
            plt.title(title, fontsize=20, color=font_color_)
            plt.xlabel(x_label, fontsize=15, color=font_color_)
            plt.ylabel(y_label, fontsize=15, color=font_color_)
            if file_path == None:
                fig.savefig('plots/plotconv-{}.png'.format(timestr), bbox_inches='tight', edgecolor='white')
            else:
                fig.savefig(file_path, bbox_inches='tight')
                
        if show_plot:
            if font_color == 'default':
                font_color_ = 'white'
            else:
                font_color_ = font_color
            plt.tick_params(axis='x', colors=font_color_)
            plt.tick_params(axis='y', colors=font_color_)
            plt.title(title, fontsize=20, color=font_color_)
            plt.xlabel(x_label, fontsize=15, color=font_color_)
            plt.ylabel(y_label, fontsize=15, color=font_color_)
            plt.show()
                
        plt.close(fig)