import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
import inspect
import warnings

try:
    import casadi

    CASADI_TYPES = tuple(
        x[1] for x in inspect.getmembers(casadi.casadi, inspect.isclass)
    )
except ModuleNotFoundError:
    warnings.warn_explicit(
        "\nImporting casadi failed. You may still use rcognita, but"
        + " without symbolic optimization capability. ",
        UserWarning,
        __file__,
        42,
    )
    CASADI_TYPES = []
import types


def is_CasADi_typecheck(*args):
    return any([isinstance(arg, CASADI_TYPES) for arg in args])


def decorateAll(decorator):
    class MetaClassDecorator(type):
        def __new__(meta, classname, supers, classdict):
            for name, elem in classdict.items():
                if type(elem) is types.FunctionType and (name != "__init__"):
                    classdict[name] = decorator(classdict[name])
            return type.__new__(meta, classname, supers, classdict)

    return MetaClassDecorator


@decorateAll
def typeInferenceDecorator(func):
    def wrapper(*args, **kwargs):
        is_symbolic = kwargs.get("is_symbolic")
        if not is_symbolic is None:
            del kwargs["is_symbolic"]
        return func(
            is_symbolic=(is_CasADi_typecheck(*args, *kwargs.values()) or is_symbolic),
            *args,
            **kwargs
        )

    return wrapper

