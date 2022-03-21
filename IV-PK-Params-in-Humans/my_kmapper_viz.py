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
