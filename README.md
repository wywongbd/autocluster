## autocluster
``autocluster`` is an automated machine learning (AutoML) toolkit for performing clustering tasks.   

Report and presentation slides can be found [here](https://github.com/wywongbd/autocluster/blob/master/reports/autocluster_report.pdf) and [here](https://github.com/wywongbd/autocluster/blob/master/reports/autocluster_ppt.pdf).

## Prerequisites
- Python 3.5 or above
- Linux OS, or [Windows WSL](https://docs.microsoft.com/en-us/windows/wsl/about) is also possible

## How to get started?
1. First, install [SMAC](https://automl.github.io/SMAC3/stable/installation.html):
  - ``sudo apt-get install build-essential swig``
  - ``conda install gxx_linux-64 gcc_linux-64 swig``
  - ``pip install smac==0.8.0``
2. ``pip install autocluster``

## How it works?
- ``autocluster`` automatically optimizes the *configuration* of a clustering problem. By *configuration*, we mean 
    - choice of dimension reduction algorithm
    - choice of clustering model
    - setting of dimension reduction algorithm's hyperparameters
    - setting of clustering model's hyperparameters  
  
- ``autocluster`` provides 3 different approaches to optimize the configuration (with increasing complexity): 
    - random optimization
    - bayesian optimization
	- great [tutorial](http://krasserm.github.io/2018/03/21/bayesian-optimization/) by Martin Krasser 
    - bayesian optimization + meta-learning (warmstarting)

## Algorithms/Models supported
- List of dimension reduction algorithms in ``sklearn`` supported by ``autocluster``'s optimizer.
<img src="images/dim_reduction_algorithms.png" width="600">

- List of clustering models in ``sklearn`` supported by ``autocluster``'s optimizer.
<img src="images/clustering_algorithms.png" width="600">

## Examples
Examples are available in these [notebooks](/autocluster/examples/).

## Experimental results
- This dataset comprises of 16 Gaussian clusters in 128-dimensional space with ``N = 1024`` points. The optimal configuration obtained by ``autocluster`` (SMAC + Warmstarting) consists of a Truncated SVD dimension reduction model + Birch clustering model.
<img src="images/clustering_result_dim128.png" width="600">

- This dataset comprises of 15 Gaussian clusters in 2-dimensional space with ``N = 5000 points``. The optimal configuration obtained by ``autocluster`` (SMAC + Warmstarting) consists of a TSNE dimension reduction model + Agglomerative clustering model.
<img src="images/clustering_result_s2.png" width="600">


## Links  
[Link](https://pypi.org/project/autocluster/) to pypi. 

## Disclaimer
The project is experimental and still under development.
