# fastSparse

<!-- ![example workflow](https://github.com/tynanseltzer/L0Learn/actions/workflows/python.yml/badge.svg) -->

## Introduction
fastSparseGAMs is a Python package that offers an efficient framework for solving L0-regularized learning problems in sparse generalized additive models (GAMs). Leveraging the [L0Learn](https://github.com/TNonet/L0Learn) package, this package introduces two novel algorithms, namely quadratic cuts and dynamic feature ordering, to deliver faster computational speed. Additionally, it comes with a new loss function (exponential loss) for classification.

<img src="https://github.com/tynanseltzer/L0Learn/blob/master/misc/eqs.png" width = 450>

## Package Installation

The latest version can be installed from pip as follows:
```bash
pip install fastsparsegams
```

## Documentation
An example on how to use `fastSparseGAMs` is provided at this [tutorial page](https://github.com/tynanseltzer/L0Learn/blob/master/python/tutorial_example/example.py).

fastSparseGAMs is developed upon the framework of L0Learn, featuring faster and novel algorithms implemented internally.
We do not alter the external Python interface functions.
Therefore, please see L0Learn's python documentation available [here](https://tnonet.github.io/L0Learn/tutorial.html) for the detailed API documentation.
The external function usage is almost idential to L0Learn's API except replacing the module name `l0learn` with `fastsparsegams`.

# Source Code and Installing from Source
Alternatively, `fastSparseGAMs` can be built from source
```bash
git clone https://github.com/ubc-systopia/fastsparsemissing.git --recurse-submodules
cd fastsparsemissing/python/
```

To install, ensure the proper packages are installed from `pyproject.toml` build from source with the following:
```bash
pip install ".[test]"
```

If you modify the source code, make sure to run 
```
pip uninstall fastsparsegams
```
and reinstall as before. Also note that fastsparsegams cannot be imported from the "python" directory -- either use the root directory for the repo or ./python/fastsparsegams.

To test, run the following command:
```bash
python -m pytest
```

## Citing fastSparseGAMs
If you find fastSparseGAMs useful in your research, please consider citing the following papers.

**Paper 1:**
```
@inproceedings{liu2022fast,
  title={Fast Sparse Classification for Generalized Linear and Additive Models},
  author={Liu, Jiachang and Zhong, Chudi and Seltzer, Margo and Rudin, Cynthia},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={9304--9333},
  year={2022},
  organization={PMLR}
}
```

**Paper 2:**
```
@article{doi:10.1287/opre.2019.1919,
author = {Hazimeh, Hussein and Mazumder, Rahul},
title = {Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms},
journal = {Operations Research},
volume = {68},
number = {5},
pages = {1517-1537},
year = {2020},
doi = {10.1287/opre.2019.1919},
URL = {https://doi.org/10.1287/opre.2019.1919},
eprint = {https://doi.org/10.1287/opre.2019.1919}
}
```

**Paper 3:**
```
@article{JMLR:v22:19-1049,
  author  = {Antoine Dedieu and Hussein Hazimeh and Rahul Mazumder},
  title   = {Learning Sparse Classifiers: Continuous and Mixed Integer Optimization Perspectives},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {135},
  pages   = {1-47},
  url     = {http://jmlr.org/papers/v22/19-1049.html}
}
```

**Paper 4:**
```
@article{hazimeh2022l0learn,
      title={L0Learn: A Scalable Package for Sparse Learning using L0 Regularization},
      author={Hussein Hazimeh and Rahul Mazumder and Tim Nonet},
      year={2022},
      eprint={2202.04820},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
