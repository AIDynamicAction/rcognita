import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
import casadi
import types


def decorateAll(decorator):
    class MetaClassDecorator(type):
        def __new__(meta, classname, supers, classdict):
            for name, elem in classdict.items():
                if type(elem) is types.FunctionType and not name is "__init__":
                    classdict[name] = decorator(classdict[name])
            return type.__new__(meta, classname, supers, classdict)

    return MetaClassDecorator


@decorateAll
def typeInferenceDecorator(func):
    def wrapper(*args, **kwargs):

        force = kwargs.get("force")

        if not force is None:
            is_symbolic = force
        else:
            is_symbolic = any(
                [isinstance(arg, (casadi.SX, casadi.DM, casadi.MX)) for arg in args]
            )
        return func(is_symbolic=is_symbolic, *args, **kwargs)

    return wrapper

