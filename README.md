# libmoran

C++23 header-only library for computing fixation probabilities of the Moran process on graphs, with Python bindings via pybind11.

## Algorithms

| Algorithm | Paper | Notes |
|-----------|-------|-------|
| **Exact** | Lieberman et al. 2005 | Well-mixed formula, isothermal theorem, r=1 |
| **Naive MC** | Diaz et al. 2014, Theorem 13 | Simulates all BD steps, O(n^4) per run |

All MC algorithms use an **epsilon-first API**: specify `(epsilon, delta)` and the library derives sample counts and step limits.

## Build

```bash
cmake --preset dev-debug
cmake --build build/debug --parallel
./build/debug/moran_tests
PYTHONPATH=build-python python -m pytest tests/
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

- Diaz, Goldberg, Mertzios, Richerby, Serna & Spirakis (2014). Approximating fixation probabilities in the generalized Moran process. *Algorithmica*.
- Lieberman, Hauert & Nowak (2005). Evolutionary dynamics on graphs. *Nature*.
