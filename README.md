# General parameter-shift rules for quantum gradients

This repository contains the code used to produce the data and figures in our [paper](https://arxiv.org/abs/2107.12390) on general parameter-shift rules for quantum gradients.

- `QAOA_circuit_evaluations.ipynb` computes bounds for the number of required circuit evaluations in 
  MaxCut-QAOA applications as discussed in Section 5.1, and correspondingly produces Figure 3.
- `plot_QAD_landscapes.ipynb` computes the variants of the model cost landscape used in Quantum
  Analytic Descent as discussed in Section 5.3. It outputs Figure 4.

Please note that the circuit evaluation estimation in its current implementation is rather slow, because
multiple semi-definite programs (SDPs) are solved for the various bounds we discuss. It is very well
possible that this code can be optimized to run faster, but in the meantime, we store the precomputed
data in `data/QAOA_circuit_evaluations_20.json` and let the plotting notebook simply access that data.
Recomputing it takes about 5 hours on a proper desktop computer.
Reducing the largest considered system size (`max_num_vertices`) significantly reduces the runtime.

All functionalities and helper functions are stored in `lib.py`.


## Citing this work

If you use this code, please consider citing the paper if applicable:

```bib
@misc{wierichs2021general,
      title={General parameter-shift rules for quantum gradients}, 
      author={David Wierichs and Josh Izaac and Cody Wang and Cedric Yen-Yu Lin},
      year={2021},
      eprint={2107.12390},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
