# nx_altair

*Draw NetworkX graphs with Altair*
**nx_altair** offers a similar **draw** API to NetworkX but returns Altair Charts instead.

This is a fork of the (currently) inactive [nx-altair](https://github.com/Zsailer/nx_altair), merging (current) pull requests and acting on some (currently) open issues (see commits).
This fork is NOT meant for a pull request due to the variety of changes (which also include removals, e.g. in this README), and of course due to the (current) inactivity of the original repo.



## Install

This fork is not on any package index, therefore the only install option is a local one:

- Clone the repo
- `pip install -e <PATH TO CLONED REPO>`



## Examples

If you'd like to start playing with nx_altair, download [this notebook](examples/nx_altair-tutorial.ipynb)!

### Simple graph

```python
import networkx as nx
import nx_altair as nxa

# Generate a random graph
G = nx.fast_gnp_random_graph(n = 20, p = 0.25)

# Compute positions for viz.
pos = nx.spring_layout(G)

# Draw the graph using Altair
viz = nxa.draw_networkx(G, pos = pos)

# Show it as an interactive plot!
viz.interactive()
```

<img src = 'docs/_img/readme.png' width = '350'>

### Leverage Altair

<img src = 'docs/_img/interactivity.gif'>


### Customize the visualization

**nx_altair** also supports many of the same arguments from NetworkX for styling your network--with an Altair twist! Map visualization attributes in a declarative manner.

```python
import numpy as np

# Add weights to nodes and edges
for n in G.nodes():
    G.nodes[n]['weight'] = np.random.randn()

for e in G.edges():
    G.edges[e]['weight'] = np.random.uniform(1, 10)


# Draw the graph using Altair
viz = nxa.draw_networkx(
    G, pos = pos,
    node_color = 'weight',
    cmap = 'viridis',
    width = 'weight',
    edge_color = 'black',
)

# Show it as an interactive plot!
viz.interactive()
```
<img src = 'docs/_img/readme2.png' width = '450'>


