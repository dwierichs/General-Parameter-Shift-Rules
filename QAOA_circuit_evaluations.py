# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

import lib

# + tags=[]
# Set whether to use precomputed data
reuse_data = False

# Set the number of how often the random projection in the Goemans-Williamson algorithm is performed
# to obtain a valid cut.
sdp_repetitions = 10
# Set the largest size of the graphs to consider. This strongly influences the runtime of the notebook.
max_num_vertices = 20
# Set the circuit depth, influencing the number of parameters for which the gradient would have to be computed.
num_params = 6

graph_types = ["5-regular", "6-regular", "4 edges per vertex", "complete"]

# + tags=[]
# Example for a graph and the difference between decomposition parameter-shift rules and the general parameter-shift rule
num_wires = 10
# num_edges = 3*num_wires
num_edges = (num_wires - 1) * num_wires / 2
g = nx.gnm_random_graph(num_wires, num_edges)
diag = lib.maxcut_hamiltonian(g)
omegas = sorted([val.item() for val in (set(np.array(diag, dtype=int)))])
print(f"omegas: {omegas}")
Omegas = [int(val) for val in lib.get_unique_differences(omegas)]
print(f"Omegas: {Omegas}")
print(
    f"There are {len(g.edges)} edges and {num_wires} vertices.",
    f"Decomposing would yield {2*len(g.edges)} evaluations.",
    f"An unrealistic (i.e. very loose) bound would suggest {2*(omegas[-1]-omegas[0])} evaluations for generalized parameter shifts.",
    f"A realistic bound would suggest {2*(num_wires**2//4-omegas[0])} evaluations for generalized parameter shifts.",
    f"Generalized parameter shifts would actually yield {2*len(Omegas)} evaluations,",
    f"saving {2*len(g.edges)-2*len(Omegas)} evaluations over decomposition",
    sep='\n'
)

# + tags=[]
if reuse_data:
    # Load the circuit execution counts
    circ_evals = pd.read_json("QAOA_evaluations.json")
else:
    # Here we will store the circuit execution counts
    circ_evals = pd.DataFrame()

    for graph_type in graph_types:
        print(f"{graph_type} graphs")
        gen_graph, _nums_wires = lib.graph_generator(graph_type, max_num_vertices)

        for i, num_wires in tqdm(enumerate(_nums_wires), total=len(_nums_wires)):
            g = gen_graph(num_wires)
            if g is None:
                continue
            diag = lib.maxcut_hamiltonian(g)
            omegas = sorted([val.item() for val in diag])
            Omegas = lib.get_unique_differences(omegas)
            mu = omegas[0]
            sdp_bound = np.round(lib.better_sdp_bound(g), 0)
            gw_upper = lib.goemans_williamson_upper_bound(g)
            gw_upper = np.round(gw_upper,0)
            gw_lower = np.round(lib.goemans_williamson_solution(g, number=sdp_repetitions), 0)

            if graph_type.startswith('complete'):
                # Check upper bound to be tight.
                assert omegas[-1]==num_wires**2//4
                phi = num_wires**2//4
                upper_bound = (phi-mu)

            elif 'regular' in graph_type:
                upper_bound = len(g.edges)
                l = int(graph_type.split('-')[0])
                if l%2==0:
                    # Expect Omegas to be even for even degree
                    assert np.allclose(np.round(np.array(Omegas)/2,0), np.array(Omegas)/2)
                    # The following three bounds use the knowledge that for 2k-regular graphs the frequencies
                    # are even and thus the maximum eigenvalue is the number of evaluations.
                    upper_bound /= 2
                    gw_upper /= 2
                    gw_lower /= 2
                    sdp_bound /= 2
            elif 'edges per vertex'==' '.join(graph_type.split()[1:4]):
                # Combine Section 4 in https://www.tandfonline.com/doi/pdf/10.1080/03081088508817681?needAccess=true
                # with Corollary 1.2 in https://core.ac.uk/download/pdf/81106227.pdf
                upper_bound = int(num_wires/4*lib.max_degree_pair(g))

            R_RZZ = {
    #             'Decomposition': (2*len(g.edges), 2*len(g.edges)),
                'Known spectrum': len(Omegas),
                'Upper bound': upper_bound,
                'GW upper bound': gw_upper,
                'GW lower bound': gw_lower,
                'SDP upper bound': sdp_bound,
            }
            for method, R in R_RZZ.items():
                grad_evals = num_params*(R+num_wires)
                hess_evals = num_params*grad_evals-(num_params**2-num_params-2)/2
                circ_evals = circ_evals.append(
                    {
                        'num_wires': num_wires,
                        'method': method,
                        'grad_evals': grad_evals,
                        'hess_evals': hess_evals,
                        'graph_type': graph_type,
                        'graph': g,
                    },
                    ignore_index=True,
                )
            M = len(g.edges)
            grad_evals = num_params*(M+num_wires)
            hess_evals = grad_evals**2/2-num_params*(M**2+num_wires**2-M-num_wires)
            circ_evals = circ_evals.append(
                    {
                        'num_wires': num_wires,
                        'method': 'Decomposition',
                        'grad_evals': grad_evals,
                        'hess_evals': hess_evals,
                        'graph_type': graph_type,
                        'graph': g,
                    },
                    ignore_index=True,
                )

    circ_evals.grad_evals = circ_evals.grad_evals.astype(float)
    circ_evals.hess_evals = circ_evals.hess_evals.astype(float)
    circ_evals.to_json("QAOA_evaluations.json")

# + tags=[]
# %matplotlib notebook
import rsmf
formatter = rsmf.CustomFormatter(
    columnwidth=246 * 0.01389,
    wide_columnwidth=510 * 0.01389,
    fontsizes=11,
    pgf_preamble=r"\usepackage{amssymb}",
)
fig = formatter.figure(aspect_ratio=1.8)
axs = fig.subplots(
    len(graph_types),
    2,
    gridspec_kw={'top':0.86, 'hspace': 0.05, 'wspace': 0.05},
)
# axs = [axs]
palette = {
    'Decomposition': 'xkcd:salmon',
    'Known spectrum': 'xkcd:brick red',
    'Upper bound': 'xkcd:bright blue',
    'GW upper bound': 'xkcd:purple',
    'GW lower bound': 'xkcd:pink',
    'SDP upper bound': 'xkcd:aquamarine',
    'GW approximate sol.': 'xkcd:pink',
    'Exact solution': ''
}
markers = {
    'GW upper bound': 7,
    'GW lower bound': 6,
}
styles = {
    'Decomposition': '',
    'Known spectrum': '',
    'Upper bound': (2, 2),
    'SDP upper bound': (5,2),
}
sdp_based = (circ_evals.method.str.contains('GW'))
msize = 20
for i, graph_type in enumerate(graph_types):
    this_graph = (circ_evals.graph_type==graph_type)
    ax0 = axs[i, 0]
    ax1 = axs[i, 1]
    # Gradient
    sns.lineplot(data=circ_evals.loc[(~sdp_based)&(this_graph)], x='num_wires', y='grad_evals', hue='method',
                 style='method', dashes=styles,
                 palette=palette, ax=ax0)
    sns.scatterplot(data=circ_evals.loc[(sdp_based)&(this_graph)], x='num_wires', y='grad_evals', hue='method',
                    palette=palette, ax=ax0, style='method', s=msize, zorder=100,
                    markers=markers, linewidths=0)
    # gradient+Hessian
    sns.lineplot(data=circ_evals.loc[(~sdp_based)&(this_graph)], x='num_wires', y='hess_evals', hue='method',
                 style='method', dashes=styles, legend=False,
                 palette=palette, ax=ax1)
    sns.scatterplot(data=circ_evals.loc[(sdp_based)&(this_graph)], x='num_wires', y='hess_evals', hue='method',
                    palette=palette, ax=ax1, style='method', s=msize, zorder=100, legend=False,
                    markers=markers, linewidths=0)
    
    [ax.set_yscale('log') for ax in (ax0, ax1)]
    if i==0:
        handles, labels = ax0.get_legend_handles_labels()
        ax0.get_legend().remove()
        xlims = ax0.get_xlim()
        ax0.set_title("$\\nabla E$")
        ax1.set_title("$\\nabla E \& \\nabla^{\\otimes 2}E$")
    else:
        ax0.get_legend().remove()
    if i==len(graph_types)-1:
        [ax.set_xlabel("Number of vertices $N$") for ax in (ax0, ax1)]
    else:
        [ax.set_xlabel("") for ax in (ax0, ax1)]
        [ax.set_xticks([]) for ax in (ax0, ax1)]
    ax0.set_ylabel("Evaluations")
    ax1.set_ylabel("")
    ax1.yaxis.tick_right()
    [ax.set_xlim(xlims) for ax in (ax0, ax1)]

    
fig.legend(
    handles=handles,#[1:5]+handles[6:],
    labels=labels,#[1:5]+labels[6:],
    bbox_to_anchor=(0.5, 0.89),
    loc='lower center',
    ncol=2,
)
plt.tight_layout()
plt.savefig(f'QAOA_evaluations_{max_num_vertices}.pdf')
# -




