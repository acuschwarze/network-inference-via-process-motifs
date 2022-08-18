# Network inference via process motifs

Code for producing the results and figures in "Network inference via process motifs" by Alice C Schwarze, Sara M. Ichinaga, and Bingni W. Brunton.

#### Quickstart

If you came here to get started on using LCCF and/or LCRC for network inference ASAP, here is a 3-step guide:

(1) Download the file `qsPEMs.py` and copy it to your working directory. (Not the repository - just that one file!) 
(2) Add the following code snippet to the files in which you want to use LCCF and/or LCRC. 

    from qsPEMs import inf_via_LCCF, inf_via_LCRC

(3) For a time-series data set `TS` given in the form of a 2d numpy array, you can now infer a network structure with `m` edges a given number of edges in one line:

    A = inf_via_LCCF(TS, m)

or 

    A = inf_via_LCRC(TS, m)

The 2d numpy array `A` is the adjacency matrix of the inferred directed, unweighted network. If you set `num_edges` to `None`, `A` is a weighted score matrix. 

You can add the keyword argument `max_lag` to indicate if you expect any tranmission lags on edges. The default value `max_lag=1` indicates that signals take exactly one time step to traverse an edge. Setting `max_lag=2` indicates that you expect signals to traverse edges in either 1 or 2 steps. Larger values for `max_lag` are also possible. However, very large numbers (e.g., `max_lag=100`) may lead to long computation times.

#### Files explained

* `qsPEMs.py` is the lite version of our code library. It lets a user compute the pairwise edge measures LCCF and LCRC and infer networks from them. (The non-lite version (i.e., all other files in this repository) includes functions for comparing LCCF and LCRC to other pairwise edge measures, running parameter sweeps, and plotting results.)
* `notebooks-figures/` includes jupyter notebooks for recreating the figures in our paper.
* `notebooks-other/` includes notebooks that we used to explore some aspects of the stochastic difference model and/or our proposed inference methods. Specifically, it includes a notebook where we derive the simplified expressions for the correction factors $\alpha^{(LCCF)}$ and $\alpha^{(LCRC)}$.
* `utils/` includes several function libraries that we have written to use in our notebooks.
* `libs/` includes the function library `curvygraph` from a previous research project (). We use it here to create drawings of process motifs.
* `data/` includes pre-calculated synthetic data that make it easy to recreate the figures in our paper. To recalculate the data for any figure, change `load=True` to `load=False` in the respective notebook before running it. To recalculate all data (which may take several days or weeks, depending on the available computing resources), delete all files in `data/` before running the notebooks.

#### Dependencies
##### Python libraries available via pip and conda
dill, matplotlib, networkx, netrd, numpy, scipy, seaborn, 

##### Other python libraries
curvygraph (included in libs)

