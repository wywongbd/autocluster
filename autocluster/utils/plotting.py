import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_convergence(data, iterations='max', title=None, x_label='Iterations', y_label='Scores',\
                     legends=[], show_plot=True, save_plot=True, file_path=None):
    
    # data : list or np.array (1D or 2D)
    #        ex) [1, 2, 5] -> one graph
    #            [[3, 5, 6, 9], [2, 5, 7], [1, 2, 3, 4, 5]] -> three graphs
    # iterations : number of iterations
    #              iterations='max' : maximum iteration number of datasets
    #                                 ex) [[3, 5, 6, 9], [2, 5, 7]] -> 4
    #              iterations='min' : minimum iteration number of datasets
    #                                 ex) [[3, 5, 6, 9], [2, 5, 7]] -> 3
    #              iterations=(int) : set iteration number manually
    # legends : set of legends
    #           ex) ['SMAC', 'Random Search']
    #           if legends==[], then legends are named numbers 1, 2, 3, ...
    
    if type(data[0]) == int:
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
        colors = cm.nipy_spectral(np.linspace(0, 1, graph_num))
                
        fig = plt.figure(figsize=(12,7))
        for i in range(graph_num):
            data_len = min(len(data_in_2D[i]), x_len)
            plt.plot(lin_sp[:data_len], data_in_2D[i][:data_len], linestyle='-', color=colors[i])
        
        plt.legend(set_legends, loc='best')
            
        if save_plot:
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
            plt.title(title, fontsize=20)
            plt.xlabel(x_label, fontsize=15)
            plt.ylabel(y_label, fontsize=15)
            if file_path == None:
                fig.savefig('plots/plotconv-{}.png'.format(timestr), bbox_inches='tight', edgecolor='white')
            else:
                fig.savefig(file_path, bbox_inches='tight')
        if show_plot:
            plt.tick_params(axis='x', colors='white')
            plt.tick_params(axis='y', colors='white')
            plt.title(title, color='white', fontsize=20)
            plt.xlabel(x_label, color='white', fontsize=15)
            plt.ylabel(y_label, color='white', fontsize=15)
            plt.show()
                
        plt.close(fig)