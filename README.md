# libmoran

C++23 header-only library for computing fixation probabilities of the Moran process on graphs, with Python bindings via pybind11.

## Algorithms

| Algorithm | Paper | Notes |
|-----------|-------|-------|
| **Exact** | Lieberman et al. 2005 | Well-mixed formula, isothermal theorem, r=1 |
| **Naive MC** | Diaz et al. 2014, Theorem 13 | Simulates all BD steps, O(n^4) per run |

All MC algorithms use an **epsilon-first API**: specify `(epsilon, delta)` and the library derives sample counts and step limits.

## Build

Requires GCC 13+, CMake 3.25+, Ninja, pybind11, and Python 3.10+.

```bash
./build.sh              # configure + build + run all tests
```

Or manually:

```bash
cmake --preset dev-debug
cmake --build build/debug --parallel
./build/debug/moran_tests       # 37 C++ tests
python -m pytest tests/         # 36 Python tests
```

## Python

```python
import moran
from moran.graph import cycle_graph, star_graph

# Exact formula (regular graphs)
result = moran.fixation_probability(cycle_graph(20), 1.5)

# Naive MC (irregular graphs)
result = moran.fixation_probability(star_graph(20), 1.5, epsilon=0.1)
```

## References

- Díaz, Goldberg, Mertzios, Richerby, Serna & Spirakis (2014). [Approximating fixation probabilities in the generalized Moran process](https://doi.org/10.1007/s00453-012-9722-7). *Algorithmica*, 69, 78–91.
- Lieberman, Hauert & Nowak (2005). [Evolutionary dynamics on graphs](https://doi.org/10.1038/nature03204). *Nature*, 433, 312–316.
