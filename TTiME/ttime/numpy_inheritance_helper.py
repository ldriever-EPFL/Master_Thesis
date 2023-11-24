import numpy as np

"""
Functions to help with the Tensor class inheriting from the numpy.ndarray class 
"""


def adapt_array_ufunc_part_1(child_class, ufunc, *inputs, out=None, **kwargs):
    # Code adapted from https://numpy.org/devdocs/user/basics.subclassing.html
    args = []
    in_no = []
    for i, input_ in enumerate(inputs):
        if isinstance(input_, child_class):
            in_no.append(i)
            args.append(input_.view(np.ndarray))
        else:
            args.append(input_)

    outputs = out
    out_no = []
    if outputs:
        out_args = []
        for j, output in enumerate(outputs):
            if isinstance(output, child_class):
                out_no.append(j)
                out_args.append(output.view(np.ndarray))
            else:
                out_args.append(output)
        kwargs['out'] = tuple(out_args)
    else:
        outputs = (None,) * ufunc.nout

    return args, kwargs, outputs


def adapt_array_ufunc_part_2(child_class, other_dim_class, ufunc, results, outputs):
    # Code adapted from https://numpy.org/devdocs/user/basics.subclassing.html
    # If no such dimensional limit exists, simply set child_class and other_dim_class to the same class
    if results is NotImplemented:
        return NotImplemented

    if ufunc.nout == 1:
        results = (results,)

    results = list((np.asarray(result).view(child_class) if output is None else output) for result, output in zip(results, outputs))
    for i in range(len(results)):
        if type(results[i]) == child_class or type(results[i]) == np.ndarray:
            if results[i].ndim == 3:
                results[i] = child_class(results[i])
            else:
                results[i] = other_dim_class(results[i])

    return results[0] if len(results) == 1 else results


def adapt_array_function(child_class, other_dim_class, func, *args, **kwargs):
    # To allow for the case where the dimension is limited to 3 (i.e., we are handling TTCores) an alternative class can be set
    # If no such dimensional limit exists, simply set child_class and other_dim_class to the same class

    # The kwargs dictionary is always added as an element of the args tuple
    kwargs = args[1]
    args = args[0]
    tmp_args = []

    for arg in args:
        if isinstance(arg, tuple):
            # arg = tuple_extract_arrays(arg)
            arg = list(arg)
            for i in range(len(arg)):
                if isinstance(arg[i], child_class):
                    arg[i] = np.array(arg[i])
        elif isinstance(arg, child_class):
            arg = np.array(arg)
        tmp_args.append(arg)

    results = func(*tmp_args, **kwargs)

    if isinstance(results, tuple):
        results = list(results)
        for i in range(len(results)):
            if isinstance(results[i], np.ndarray):
                if results[i].ndim == 3:
                    results[i] = child_class(results[i])
                else:
                    results[i] = other_dim_class(results[i])
        if len(results) == 1:
            results = results[0]

    else:
        if isinstance(results, np.ndarray):
            if results.ndim == 3:
                results = child_class(results)
            else:
                results = other_dim_class(results)

    return results
