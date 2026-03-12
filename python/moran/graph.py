"""Graph types and construction.

Re-exports C++ bindings for graph creation.
"""

from ._moran import graph as _graph

CSRGraph = _graph.CSRGraph

complete_graph = _graph.complete_graph
cycle_graph = _graph.cycle_graph
star_graph = _graph.star_graph
double_star_graph = _graph.double_star_graph

__all__ = [
    "CSRGraph",
    "complete_graph",
    "cycle_graph",
    "star_graph",
    "double_star_graph",
]
