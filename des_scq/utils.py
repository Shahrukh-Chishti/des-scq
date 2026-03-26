"""
des_scq.utils
=============
Plotting helpers and miscellaneous utilities.

All visualization functions are built on top of ``plotly_lite`` (a
lightweight Plotly wrapper used throughout the project) and produce
interactive HTML plots.  Several functions also provide a network
visualization of the circuit graph via ``networkx`` / ``pyvis``.

Functions
---------
empty              Create a NaN-filled NumPy array of given shape.
view_pydot         Render a NetworkX graph as a PNG in a Jupyter notebook.
plotVisGraph       Interactive network visualization via PyVis (HTML).
plotMatPlotGraph   Static network visualization via Matplotlib.
plotDOTGraph       Export a DOT-format PNG of a NetworkX graph.
"""

from networkx.drawing.nx_pydot import to_pydot
from IPython.display import Image, display
import time
from pyvis import network as pvnet
import networkx as nx
from numpy import nan, zeros, arange
from plotly_lite import *


# ---------------------------------------------------------------------------
# Numeric utilities
# ---------------------------------------------------------------------------

def empty(shape):
    """Return a NaN-filled NumPy array of the given shape.

    Useful as a pre-allocated buffer for loss landscapes where some entries
    may legitimately remain unfilled.

    Parameters
    ----------
    shape : tuple of int
        Array shape, e.g. ``(20, 20)``.

    Returns
    -------
    ndarray
        Array of the given shape with all entries set to ``nan``.

    Example
    -------
    >>> Loss = empty((len(JJ1_range), len(JJ3_range)))
    >>> Loss[i, j] = loss_value
    """
    E = zeros(shape)
    E.fill(nan)
    return E


# ---------------------------------------------------------------------------
# Network / graph visualization
# ---------------------------------------------------------------------------

def view_pydot(G):
    """Display a NetworkX graph as a PNG image in a Jupyter notebook.

    Converts the graph to DOT format via ``pydot``, renders it as PNG,
    and calls ``IPython.display.display`` to show it inline.

    Parameters
    ----------
    G : networkx.Graph or MultiGraph
        Circuit graph to visualize.

    Notes
    -----
    Requires ``graphviz`` to be installed on the system (used by ``pydot``
    under the hood).
    """
    pdot = to_pydot(G)
    img  = Image(pdot.create_png())
    display(img)


def plotVisGraph(G, filename='temp', height='300px', width='500px'):
    """Create an interactive HTML network visualization via PyVis.

    Parameters
    ----------
    G : networkx.Graph or MultiGraph
        Circuit graph to visualize.
    filename : str, optional
        Base name for the output HTML file (without extension).
        Default ``'temp'``.
    height : str, optional
        CSS height of the visualization canvas.  Default ``'300px'``.
    width : str, optional
        CSS width of the visualization canvas.  Default ``'500px'``.

    Returns
    -------
    str
        Path to the saved HTML file (also opened in the default browser if
        running outside Jupyter).

    Notes
    -----
    A file ``<filename>.html`` is written to the current working directory.
    """
    G   = G.copy()
    net = pvnet.Network(height=height, width=width)
    net.from_nx(G)
    return net.show(filename + '.html')


def plotMatPlotGraph(G, filename):
    """Draw a static Matplotlib visualization of a circuit graph.

    Uses a circular layout.  Nodes are labelled with their integer indices.

    Parameters
    ----------
    G : networkx.Graph
        Circuit graph to draw.
    filename : str
        Unused in the current implementation (kept for API consistency with
        other plot functions).

    Notes
    -----
    Calls ``plt.show()`` — make sure a Matplotlib backend is configured.
    """
    nx.draw(G, with_labels=True, pos=nx.circular_layout(G))
    plt.show()


def plotDOTGraph(G, filename='temp'):
    """Export a DOT-rendered PNG of a NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph
        Circuit graph to render.
    filename : str, optional
        Base name for the output PNG file (without extension).
        Default ``'temp'``.

    Notes
    -----
    Writes ``<filename>.png`` to the current working directory.
    Requires ``graphviz`` to be installed.
    """
    pdot = to_pydot(G)
    pdot.write_png(filename + '.png')
