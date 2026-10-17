# Homological dynamics: Conley–Morse graphs and the order retraction

Three files.

| | |
|---|---|
| `Homological_Dynamics_Leslie_map.ipynb` | The two-stage Leslie map over a region of phase space: a rigorous outer approximation by interval arithmetic, the Morse sets, the Morse graph and the Conley index of each set, at four parameter settings — and, on the last one, the **order retraction**, which draws every box in the color of the Morse set it retracts onto. |
| `Homological_Dynamics_Problem_GP.ipynb` | The same pipeline where the map is not a formula but a Gaussian process fitted to data. |
| `cmgdb_retract.py` | The order retraction itself, as a module that works on what CMGDB already returns. |

Both notebooks run in Google Colab and install what they need in their first
cell.

## The order retraction

CMGDB computes a Morse decomposition: which recurrent sets there are, how they
are ordered, and what their Conley indices prove. It does not say where in
phase space the orbits that do the reaching live. The order retraction does:
it assigns every box of the grid to one Morse set in a way that respects the
order, so the grid is partitioned into the basins of the Morse decomposition.

```python
import CMGDB, cmgdb_retract

morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)

r = cmgdb_retract.order_retraction(morse_graph, map_graph)
print(r)
#   order retraction: all 8040088 components assigned to a Morse set by
#   corollary 2.5 (8040084 outside P), order preserving on every edge -- checked

cmgdb_retract.plot_basins(morse_graph, r, clist)   # every box in its set's color
r.sigma                                            # the Morse set of each box
```

Q is the poset of all strongly connected components of the multivalued map,
ordered by reachability; P is a sub-poset of it — the Morse sets. An order
retraction is a map σ : Q ↠ P that is order preserving and fixes P. Kalies,
Kasti and VanderVorst, *Lattices and order in dynamics*, give the algorithm:
Theorem 2.4 in general, and Corollary 2.5 for the case where every component
outside P has exactly one Morse set immediately below it. Both are
implemented, and nothing is assigned unless a complete retraction exists —
a partial labeling would be read as a decomposition of the domain and it
would not be one. On a domain that is not forward invariant the module
refuses, because there the components whose orbits leave carry no constraint
and would accept any color at all.

It needs only `numpy` and `scipy` beyond CMGDB, and it is linear in the graph:
at 2²³ boxes and 4.7 × 10⁷ edges the retraction takes about 15 seconds, and
reading CMGDB's graph into numpy — one `adjacencies` call per box — takes
rather longer than the retraction does.

## Making the import work in Colab

`cmgdb_retract.py` has to be somewhere Python can find it. The Leslie notebook
mounts Google Drive and does this:

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/AMS Colabs')
import cmgdb_retract
```

so put the file in that folder, or change the path to wherever you keep it.
Fetching it straight from this repository works too:

```python
!wget -q https://raw.githubusercontent.com/USER/homological-dynamics-ams/main/cmgdb_retract.py
import cmgdb_retract
```
