import pickle
import sys
import string
import copy

import numpy as np
import pandas as pd
import pymc as pm
from functools import reduce


def flatten(l):
    return reduce(lambda x, y: list(x) + list(y), l)


def pretty_tag(tag):
    return tag[0] if len(tag) == 1 else ", ".join(str(tag))


# COMMENT: ALEX --> Observed strange behavior of the pretty_tag() function that leads to excess commas in plot titles
# Use this as alternative in some cases ?

# def prettier_tag(tag):
#     if len(tag) == 1:
#         return tag[0]
#     else:
#         for tag_tmp in tag:
#             tag_list.append(str(tag_tmp))
#         return '(' + ', '.join(tag_list) + ')'


def load(fname):
    """Load a hierarchical model saved to file via
    model.save(fname)

    """
    with open(fname, "rb") as f:
        model = pickle.load(f)

    return model


def get_traces(model):
    """Returns recarray of all traces in the model.

    :Arguments:
        model : kabuki.Hierarchical submodel or pymc.MCMC model

    :Returns:
        trace_array : recarray

    """
    if isinstance(model, pm.MCMC):
        m = model
    else:
        m = model.mc

    nodes = list(m.stochastics)

    names = [node.__name__ for node in nodes]
    dtype = [(name, np.float) for name in names]
    traces = np.empty(nodes[0].trace().shape[0], dtype=dtype)

    # Store traces in one array
    for name, node in zip(names, nodes):
        traces[name] = node.trace()[:]

    return traces


def logp_trace(model):
    """
    return a trace of logp for model
    """

    # init
    db = model.mc.db
    n_samples = db.trace("deviance").length()
    logp = np.empty(n_samples, np.double)

    # loop over all samples
    for i_sample in range(n_samples):
        # set the value of all stochastic to their 'i_sample' value
        for stochastic in model.mc.stochastics:
            try:
                value = db.trace(stochastic.__name__)[i_sample]
                stochastic.value = value

            except KeyError:
                print("No trace available for %s. " % stochastic.__name__)

        # get logp
        logp[i_sample] = model.mc.logp

    return logp


def interpolate_trace(x, trace, range=(-1, 1), bins=100):
    """Interpolate distribution (from samples) at position x.

    :Arguments:
        x <float>: position at which to evalute posterior.
        trace <np.ndarray>: Trace containing samples from posterior.

    :Optional:
        range <tuple=(-1,1): Bounds of histogram (should be fairly
            close around region of interest).
        bins <int=100>: Bins of histogram (should depend on trace length).

    :Returns:
        float: Posterior density at x.
    """

    import scipy.interpolate

    x_histo = np.linspace(range[0], range[1], bins)
    histo = np.histogram(trace, bins=bins, range=range, density=True)[0]
    interp = scipy.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)

    return interp


def save_csv(data, fname, *args, **kwargs):
    """Save record array to fname as csv.

    :Arguments:
        data <np.recarray>: Data array to output.
        fname <str>: File name.

    :Notes:
        Forwards call to pandas DataFrame.to_csv

    :SeeAlso: load_csv
    """
    pd.DataFrame(data).to_csv(fname, *args, **kwargs)


def load_csv(*args, **kwargs):
    """Load record array from csv.

    :Arguments:
        fname <str>: File name.
        See pandas.read_csv()

    :Optional:
        See pandas.read_csv()

    :Note:
        Forwards call to pandas.read_csv()

    :SeeAlso: save_csv, pandas.read_csv()
    """
    return pd.read_csv(*args, **kwargs)


def set_proposal_sd(mc, tau=0.1):
    for var in mc.variables:
        if var.__name__.endswith("var"):
            # Change proposal SD
            mc.use_step_method(pm.Metropolis, var, proposal_sd=tau)

    return


def stochastic_from_dist(*args, **kwargs):
    return pm.stochastic_from_dist(*args, dtype=np.dtype("O"), mv=True, **kwargs)


def concat_models(models, concat_traces=True):
    """Concatenate traces of multiple identical models into a new
    model containing all traces of the individual models.

    """
    # copy first model
    target_model = copy.deepcopy(models[0])
    target_stochs = target_model.get_stochastics()
    # append traces
    for i, model in enumerate(models[1:]):
        stochs = model.get_stochastics()
        for node, target_node in zip(stochs.node, target_stochs.node):
            assert (
                node.__name__ == target_node.__name__
            ), "Node names do not match. You have to pass identical models."
            if concat_traces:
                target_node.trace._trace[0] = np.concatenate(
                    [target_node.trace[:], node.trace[:]]
                )
            else:
                target_node.trace._trace[i + 1] = node.trace[:]

    target_model.gen_stats()

    return target_model

def plot_ppc_by_cond(
    data,
    or_d=None, # original dataset
    subjs=None,     # subject's index
    conds=None,     # condition
    kind="kde",
    alpha=None,
    mean=True,
    observed=True,
    color=None,
    colors=None,
    grid=None,
    figsize=None,
    textsize=None,
    data_pairs=None,
    var_names=None,
    filter_vars=None,
    coords=None,
    flatten=None,
    flatten_pp=None,
    num_pp_samples=None,
    random_seed=None,
    jitter=None,
    animated=False,
    animation_kwargs=None,
    legend=True,
    labeller=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    group="posterior",
    show=None,
):
    """
    Plot for posterior/prior predictive checks.

    Parameters
    ----------

    """
    from arviz.plots.plot_utils import (
        default_grid,
        filter_plotters_list,
        get_plotting_function,
        _scale_fig_size,
    )
    
    from arviz.labels import BaseLabeller
    from arviz.sel_utils import xarray_var_iter
    from arviz.rcparams import rcParams
    
    from arviz.utils import (
        _var_names,
        _subset_list,
    )
    
#         xarray_sel_iter,
#         _dims,
#         _zip_dims,
#         _scale_fig_size,
        
    from itertools import product
    # _var_names, _subset_list, filter_plotters_list,
    # xarray_sel_iter, xarray_var_iter, _dims, _zip_dims
#     from func4PPCPlot import , , 
#     from func4PPCPlot import 
#     from func4PPCPlot import default_grid, get_plotting_function
    # from rcparams import 
    from numbers import Integral
    import numpy as np
    
    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    for groups in (f"{group}_predictive", "observed_data"):
        if not hasattr(data, groups):
            raise TypeError(f'`data` argument must have the group "{groups}" for ppcplot')

    if kind.lower() not in ("kde", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `cumulative`, or `scatter`")

    if colors is None:
        colors = ["C0", "k", "C1"]

    if isinstance(colors, str):
        raise TypeError("colors should be a list with 3 items.")

    if len(colors) != 3:
        raise ValueError("colors should be a list with 3 items.")

    if color is not None:
        warnings.warn("color has been deprecated in favor of colors", FutureWarning)
        colors[0] = color

    if data_pairs is None:
        data_pairs = {}

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()
    if backend == "bokeh":
        if animated:
            raise TypeError("Animation option is only supported with matplotlib backend.")

    observed_data = data.observed_data

    if group == "posterior":
        predictive_dataset = data.posterior_predictive
    elif group == "prior":
        predictive_dataset = data.prior_predictive

    if coords is None:
        coords = {}

    if labeller is None:
        labeller = BaseLabeller()

    if random_seed is not None:
        np.random.seed(random_seed)

    total_pp_samples = predictive_dataset.sizes["chain"] * predictive_dataset.sizes["draw"]
    if num_pp_samples is None:
        if kind == "scatter" and not animated:
            num_pp_samples = min(5, total_pp_samples)
        else:
            num_pp_samples = total_pp_samples

    if (
        not isinstance(num_pp_samples, Integral)
        or num_pp_samples < 1
        or num_pp_samples > total_pp_samples
    ):
        raise TypeError(
            "`num_pp_samples` must be an integer between 1 and " + f"{total_pp_samples}."
        )

    pp_sample_ix = np.random.choice(total_pp_samples, size=num_pp_samples, replace=False)

    for key in coords.keys():
        coords[key] = np.where(np.in1d(observed_data[key], coords[key]))[0]
        
    obs_plotters = []
    pp_plotters = []

    if conds is None:
        dim_tmp = ['subj_idx']
        level_ls = subjs
    else:
        dim_tmp = ['subj_idx'] + conds
    #     levels = list(chain(or_d[conds].drop_duplicates().values.tolist()))
        levels = or_d[conds].drop_duplicates().values.tolist()
        level_ls = list(product(subjs, levels))


    # convert tuple to a dict
    levels_ls_tmp= {}
    for ii in range(len(level_ls)):
        tmp = level_ls[ii]
        levels_ls_tmp[ii] = []
        if (isinstance(tmp, int)) or (isinstance(tmp, str)):
            levels_ls_tmp[ii].append(tmp)
        else: #  isinstance(tmp, list):
            for jj in tmp:
    #             print(jj)
                if (isinstance(jj, int)) or (isinstance(jj, str)):
                    levels_ls_tmp[ii].append(jj)
                else:
                    levels_ls_tmp[ii]  = levels_ls_tmp[ii] + jj

    # dict's value to list       
    level_ls = list(levels_ls_tmp.values())

    for level in level_ls:
        print(level)
        # combine the subj_idx with conditions
        crit_tmp=[]
        for i, j in zip(dim_tmp, level):
            if isinstance(j, str):
                crit_tmp.append(i + "=='" + str(j) + "'")
            else:
                crit_tmp.append(i + "==" + str(j))

        crit_tmp = " and ".join(crit_tmp) # combine the conditions
        plot_idx = or_d.query(crit_tmp).index

        data_tmp = data.isel(trial_idx=plot_idx)

        observed_data_tmp = data_tmp.observed_data
        predictive_data_tmp = data_tmp.posterior_predictive

        if var_names is None:
            var_names = list(observed_data_tmp.data_vars)
        data_pairs = {}
        var_names = _var_names(var_names, observed_data_tmp, None)
        pp_var_names = [data_pairs.get(var, var) for var in var_names]
        pp_var_names = _var_names(pp_var_names, predictive_data_tmp, None)

        flatten_pp = None
        flatten = None
        num_pp_samples = 50
        coords = {}

        if flatten_pp is None and flatten is None:
            flatten_pp = list(predictive_data_tmp.dims.keys())
        elif flatten_pp is None:
            flatten_pp = flatten
        if flatten is None:
            flatten = list(observed_data_tmp.dims.keys())

        total_pp_samples = predictive_data_tmp.sizes["chain"] * predictive_data_tmp.sizes["draw"]

        pp_sample_ix = np.random.choice(total_pp_samples, size=num_pp_samples, replace=False)

        for key in coords.keys():
            coords[key] = np.where(np.in1d(observed_data_tmp[key], coords[key]))[0]

        obs_plotters_tmp = filter_plotters_list(
            list(
                xarray_var_iter(
                    observed_data_tmp.isel(coords),
                    skip_dims=set(flatten),
                    var_names=var_names,
                    combined=True,
                )
            ),
            "plot_ppc",
        )

        length_plotters_tmp = len(obs_plotters_tmp)
        pp_plotters_tmp = [
            tup
            for _, tup in zip(
                range(length_plotters_tmp),
                xarray_var_iter(
                    predictive_data_tmp.isel(coords),
                    var_names=pp_var_names,
                    skip_dims=set(flatten_pp),
                    combined=True,
                ),
            )
        ]

        for var_idx in range(len(var_names)):
            tmp0 = list(obs_plotters_tmp[0])
            tmp1 = list(pp_plotters_tmp[0])
            for i, j in zip(dim_tmp, level):
                if i == "subj_idx":        
                    tmp0[1][i] = "subj_" + str(j)
                    tmp0[2][i] = "subj_" + str(j)
                    tmp1[1][i] = "subj_" + str(j)
                    tmp1[2][i] = "subj_" + str(j)
                else:
                    tmp0[1][i] = i + "_" + str(j)
                    tmp0[2][i] = i + "_" + str(j)
                    tmp1[1][i] = i + "_" + str(j)
                    tmp1[2][i] = i + "_" + str(j)

            obs_plotters.append(tuple(tmp0))
            pp_plotters.append(tuple(tmp1))
                    
    length_plotters = len(obs_plotters)
    rows, cols = default_grid(length_plotters, grid=grid)

    ppcplot_kwargs = dict(
        ax=ax,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        figsize=figsize,
        animated=animated,
        obs_plotters=obs_plotters,
        pp_plotters=pp_plotters,
        predictive_dataset=predictive_dataset,
        pp_sample_ix=pp_sample_ix,
        kind=kind,
        alpha=alpha,
        colors=colors,
        jitter=jitter,
        textsize=textsize,
        mean=mean,
        observed=observed,
        total_pp_samples=total_pp_samples,
        legend=legend,
        labeller=labeller,
        group=group,
        animation_kwargs=animation_kwargs,
        num_pp_samples=num_pp_samples,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_ppc", "ppcplot", backend)
    axes = plot(**ppcplot_kwargs)
    return axes


###########################################################################
# The following code is directly copied from Twisted:
# http://twistedmatrix.com/trac/browser/tags/releases/twisted-11.1.0/twisted/python/reflect.py
# For the license see:
# http://twistedmatrix.com/trac/browser/trunk/LICENSE
###########################################################################


class _NoModuleFound(Exception):
    """
    No module was found because none exists.
    """


class InvalidName(ValueError):
    """
    The given name is not a dot-separated list of Python objects.
    """


class ModuleNotFound(InvalidName):
    """
    The module associated with the given name doesn't exist and it can't be
    imported.
    """


class ObjectNotFound(InvalidName):
    """
    The object associated with the given name doesn't exist and it can't be
    imported.
    """


def _importAndCheckStack(importName):
    """
    Import the given name as a module, then walk the stack to determine whether
    the failure was the module not existing, or some code in the module (for
    example a dependent import) failing.  This can be helpful to determine
    whether any actual application code was run.  For example, to distiguish
    administrative error (entering the wrong module name), from programmer
    error (writing buggy code in a module that fails to import).

    @raise Exception: if something bad happens.  This can be any type of
    exception, since nobody knows what loading some arbitrary code might do.

    @raise _NoModuleFound: if no module was found.
    """
    try:
        try:
            return __import__(importName)
        except ImportError:
            excType, excValue, excTraceback = sys.exc_info()
            while excTraceback:
                execName = excTraceback.tb_frame.f_globals["__name__"]
                if (
                    execName is None
                    or execName == importName  # python 2.4+, post-cleanup
                ):  # python 2.3, no cleanup
                    raise excType(excValue).with_traceback(excTraceback)
                excTraceback = excTraceback.tb_next
            raise _NoModuleFound()
    except:
        # Necessary for cleaning up modules in 2.3.
        sys.modules.pop(importName, None)
        raise


def find_object(name):
    """
    Retrieve a Python object by its fully qualified name from the global Python
    module namespace.  The first part of the name, that describes a module,
    will be discovered and imported.  Each subsequent part of the name is
    treated as the name of an attribute of the object specified by all of the
    name which came before it.  For example, the fully-qualified name of this
    object is 'twisted.python.reflect.namedAny'.

    @type name: L{str}
    @param name: The name of the object to return.

    @raise InvalidName: If the name is an empty string, starts or ends with
        a '.', or is otherwise syntactically incorrect.

    @raise ModuleNotFound: If the name is syntactically correct but the
        module it specifies cannot be imported because it does not appear to
        exist.

    @raise ObjectNotFound: If the name is syntactically correct, includes at
        least one '.', but the module it specifies cannot be imported because
        it does not appear to exist.

    @raise AttributeError: If an attribute of an object along the way cannot be
        accessed, or a module along the way is not found.

    @return: the Python object identified by 'name'.
    """

    if not name:
        raise InvalidName("Empty module name")

    names = name.split(".")

    # if the name starts or ends with a '.' or contains '..', the __import__
    # will raise an 'Empty module name' error. This will provide a better error
    # message.
    if "" in names:
        raise InvalidName(
            "name must be a string giving a '.'-separated list of Python "
            "identifiers, not %r" % (name,)
        )

    topLevelPackage = None
    moduleNames = names[:]
    while not topLevelPackage:
        if moduleNames:
            trialname = ".".join(moduleNames)
            try:
                topLevelPackage = _importAndCheckStack(trialname)
            except _NoModuleFound:
                moduleNames.pop()
        else:
            if len(names) == 1:
                raise ModuleNotFound("No module named %r" % (name,))
            else:
                raise ObjectNotFound("%r does not name an object" % (name,))

    obj = topLevelPackage
    for n in names[1:]:
        obj = getattr(obj, n)

    return obj


######################
# END OF COPIED CODE #
######################


def centered_half_cauchy_rand(S, size):
    """sample from a half Cauchy distribution with scale S"""
    return abs(S * np.tan(np.pi * pm.random_number(size) - np.pi / 2.0))


def centered_half_cauchy_logp(x, S):
    """logp of half Cauchy with scale S"""
    x = np.atleast_1d(x)
    if sum(x < 0):
        return -np.inf
    return pm.flib.cauchy(x, 0, S) + len(x) * np.log(2)


HalfCauchy = pm.stochastic_from_dist(
    name="Half Cauchy",
    random=centered_half_cauchy_rand,
    logp=centered_half_cauchy_logp,
    dtype=np.double,
)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
