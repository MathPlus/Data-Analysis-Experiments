import warnings
import numpy as np
from kmapper.visuals import ( colorscale_default ,
                       _scale_color_values ,
                       _format_mapper_data ,
                       _graph_data_distribution ,
                       _format_meta ,
                       _render_d3_vis )

# @deprecated_alias(color_function="color_values")
def my_kmapper_viz(
    graph,
    color_values=None,
    color_function_name=None,
    node_color_function="mean",
    colorscale=None,
    custom_tooltips=None,
    custom_meta=None,
    path_html="mapper_visualization_output.html",
    title="Kepler Mapper",
    save_file=True,
    X=None,
    X_names=None,
    lens=None,
    lens_names=None,
    nbins=10,
    include_searchbar=False,
    verbose=0
):
    """Generate a visualization of the simplicial complex mapper output. Turns the complex dictionary into a HTML/D3.js visualization

    Parameters
    ----------
    graph : dict
        Simplicial complex output from the `map` method.

    color_function : list or 1d array
        .. deprecated:: 1.4.1
           Use `color_values` instead.

    color_values : list or 1d array, or list of 1d arrays
        color_values are sets (1d arrays) of values -- for each set, there should be
        one color value for each datapoint.

        These color values are used to compute the color value of a _node_ by applying `node_color_function` to
        the color values of each point within the node. The distribution of color_values for a given
        node can also be viewed in the visualization under the node details pane.

        A list of sets of color values (a list of 1d arrays) can be passed.
        If this is the case, then the visualization will have a toggle button
        for switching the visualization's currently active set of color values.

        If no color_values passed, then the data points' row positions are used as
        the set of color values.

    color_function_name : String or list
        A descriptor of the functions used to generate `color_values`.
        Will be used as labels in the visualization.
        If set, must be equal to the number of columns in color_values.

    node_color_function : String or 1d array, default is 'mean'
        Applied to the color_values of data points within a node to determine the color of the nodes.
        Will be applied column-wise to color_values.
        Must be a function available on numpy class object -- e.g., 'mean' => np.mean().

        If array, then 1d array of strings of np function names. Each node_color_function
        will be applied to each set of color_values (full permutation), and a toggle button will allow
        switching between the current active node_color_function for the visualization.

        See `visuals.py:_node_color_function()`

    colorscale : list
        Specify the colorscale to use. See visuals.colorscale_default.

    path_html : String
        file name for outputing the resulting html.

    custom_meta: dict
        Render (key, value) in the Mapper Summary pane.

    custom_tooltip: list or array like
        Value to display for each entry in the node. The cluster data pane will display entries for all values in the node. Default is index of data.

    save_file: bool, default is True
        Save file to `path_html`.

    X: numpy arraylike
        If supplied, compute statistics information about the original data source with respect to each node.

    X_names: list of strings
        Names of each variable in `X` to be displayed. If None, then display names by index.

    lens: numpy arraylike
        If supplied, compute statistics of each node based on the projection/lens

    lens_name: list of strings
        Names of each variable in `lens` to be displayed. In None, then display names by index.

    nbins: int, default is 10
        Number of bins shown in histogram of tooltip color distributions.

    include_searchbar: bool, default False
        Whether to include a search bar at the top of the visualization.

        The search functionality performs permits AND, OR, and EXACT
        methods, all against lowercased tooltips.

        * AND: the search query is split by whitespace. A data point's custom tooltip must
          match _each_ of the query terms in order to match overall. The base size of a node
          is multiplied by the number of datapoints matching the searchquery.
        * OR: the search query is split by whitespace. A data point's custom tooltip must
          match _any_ of the query terms in order to match overall. The base size of a node
          is multiplied by the number of datapoints matching the searchquery.
        * EXACT: A data point's custom tooltip must exactly match the query. Any nodes
          with a matching datapoint are set to glow.

        To reset any search-induced visual alterations, submit an empty search query.

    Returns
    --------
    html: string
        Returns the same html that is normally output to `path_html`. Complete graph and data ready for viewing.

    Examples
    ---------

    >>> # Basic creation of a `.html` file at `kepler-mapper-output.html`
    >>> html = mapper.visualize(graph, path_html="kepler-mapper-output.html")

    >>> # Jupyter Notebook support
    >>> from kmapper import jupyter
    >>> html = mapper.visualize(graph, path_html="kepler-mapper-output.html")
    >>> jupyter.display(path_html="kepler-mapper-output.html")

    >>> # Customizing the output text
    >>> html = mapper.visualize(
    >>>     graph,
    >>>     path_html="kepler-mapper-output.html",
    >>>     title="Fashion MNIST with UMAP",
    >>>     custom_meta={"Description":"A short description.",
    >>>                  "Cluster": "HBSCAN()"}
    >>> )

    >>> # Custom coloring data based on your 1d lens
    >>> html = mapper.visualize(
    >>>     graph,
    >>>     color_values=lens
    >>> )

    >>> # Custom coloring data based on the first variable
    >>> cf = mapper.project(X, projection=[0])
    >>> html = mapper.visualize(
    >>>     graph,
    >>>     color_values=cf
    >>> )

    >>> # Customizing the tooltips with binary target variables
    >>> X, y = split_data(df)
    >>> html = mapper.visualize(
    >>>     graph,
    >>>     path_html="kepler-mapper-output.html",
    >>>     title="Fashion MNIST with UMAP",
    >>>     custom_tooltips=y
    >>> )

    >>> # Customizing the tooltips with html-strings: locally stored images of an image dataset
    >>> html = mapper.visualize(
    >>>     graph,
    >>>     path_html="kepler-mapper-output.html",
    >>>     title="Fashion MNIST with UMAP",
    >>>     custom_tooltips=np.array(
    >>>             ["<img src='img/%s.jpg'>"%i for i in range(inverse_X.shape[0])]
    >>>     )
    >>> )

    >>> # Using multiple datapoint color functions
    >>> # Uses a two-dimensional lens, so two `color_function_name`s are required
    >>> lens = np.c_[isolation_forest_lens, l2_norm_lens]
    >>> html = mapper.visualize(
    >>>     graph,
    >>>     path_html="breast-cancer-multiple-color-functions.html",
    >>>     title="Wisconsin Breast Cancer Dataset",
    >>>     color_values=lens,
    >>>     color_function_name=['Isolation Forest', 'L2-norm']
    >>> )

    >>> # Using multiple node color functions
    >>> html = mapper.visualize(
    >>>     graph,
    >>>     path_html="breast-cancer-multiple-color-functions.html",
    >>>     title="Wisconsin Breast Cancer Dataset",
    >>>     node_color_function=['mean', 'std', 'median', 'max']
    >>> )

    >>> # Combining both multiple datapoint color functions and multiple node color functions
    >>> lens = np.c_[isolation_forest_lens, l2_norm_lens]
    >>> html = mapper.visualize(
    >>>     graph,
    >>>     path_html="breast-cancer-multiple-color-functions.html",
    >>>     title="Wisconsin Breast Cancer Dataset",
    >>>     color_values=lens,
    >>>     color_function_name=['Isolation Forest', 'L2-norm']
    >>>     node_color_function=['mean', 'std', 'median', 'max']
    >>> )

    """
    if colorscale is None:
        colorscale = colorscale_default

    if X_names is None:
        X_names = []

    if lens_names is None:
        lens_names = []

    if not len(graph["nodes"]) > 0:
        raise Exception(
            "Visualize requires a mapper with more than 0 nodes. \nIt is possible that the constructed mapper could have been constructed with bad parameters. This can occasionally happens when using the default clustering algorithm. Try changing `eps` or `min_samples` in the DBSCAN clustering algorithm."
        )

    if color_function_name is None:
        color_function_name = []
    elif isinstance(color_function_name, str):
        color_function_name = [color_function_name]

    if isinstance(node_color_function, str):
        node_color_function = [node_color_function]

    for _node_color_function_name in node_color_function:
        try:
            getattr(np, _node_color_function_name)
        except AttributeError as e:
            raise AttributeError(
                "Invalid `node_color_function` {}, must be a function available on `numpy` class.".format(
                    _node_color_function_name
                )
            ) from e

    if color_values is None:
        # We generate default `color_values` based on data row order
        n_samples = np.max([i for s in graph["nodes"].values() for i in s]) + 1
        color_values = np.arange(n_samples)
        if not len(color_function_name):
            color_function_name = ["Row number"]
        else:
            # `color_function_name` was not None, while `color_values` was None
            #
            # This is okay, as long as there's only one entry for `color_function_name`.
            # If this is the case, then that will be used to name the default
            # `color_values` based on row order. But we will raise a warning.

            if len(color_function_name) == 1:
                warnings.warn(
                    "`color_function_name` was set -- however, no `color_values` were passed, so default color_values were computed based on row order, and the passed `color_function_name` will be set as their label. This may be unexpected."
                )
            else:
                raise Exception(
                    "More than one `color_function_name` was set, while `color_values` was not set. If `color_values` was not set, then only one `color_function_name` can be passed. Refusing to proceed."
                )
    else:
        color_values = np.array(color_values)
        # test whether we have a color_function_name for each color_value vector
        if color_values.ndim == 1:
            num_color_value_vectors = 1
        else:
            num_color_value_vectors = color_values.shape[1]
        num_color_function_names = len(color_function_name)
        if num_color_value_vectors != num_color_function_names:
            raise Exception(
                "{} `color_function_names` values found, but {} columns found in color_values. Must be equal.".format(
                    num_color_function_names, num_color_value_vectors
                )
            )

    color_values = _scale_color_values(color_values)

    mapper_data = _format_mapper_data(
        graph,
        color_values,
        node_color_function,
        X,
        X_names,
        lens,
        lens_names,
        custom_tooltips,
        nbins,
        colorscale=colorscale,
    )

    histogram = []
    for _node_color_function_name in node_color_function:
        _histogram = _graph_data_distribution(
            graph, color_values, _node_color_function_name, colorscale
        )
        if np.array(_histogram).ndim == 1:
            _histogram = [_histogram]  # javascript will expect the histogram
            # array to be indexed for the number of
            # node_color_functions first, and second
            # for the number of color_functions
        histogram.append(_histogram)

    mapper_summary = _format_meta(
        graph, color_function_name, node_color_function, custom_meta
    )

    html = _render_d3_vis(
        title, mapper_summary, histogram, mapper_data, colorscale, include_searchbar
    )

    if save_file:
        with open(path_html, "wb") as outfile:
            outfile.write(html.encode("utf-8"))
            if verbose > 0:
                print("Wrote visualization to: %s" % (path_html))

    return html
