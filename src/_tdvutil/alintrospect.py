"""
alintrospect - A very, VERY janky hack of a hack to try to determine the class
hierarchy of a type/object, and figure out what particular data/methods/etc
are defined at each level. I'd hope that there would be a better way to do
this, but I've yet to find one. Suggestions welcome.
"""

import inspect
import itertools
from typing import Iterable, List, Set, Tuple

# FIXME: totally not sure how to type annotate this list of types
# base_types: tuple[type, ...] = (list[Any], tuple[Any, ...], set[Any], dict[Any, Any],
#                                 int, float, complex, str, bytes, bool, set[Any], frozenset[Any])
# base_types = (list, tuple, set, dict, int, float, complex, str, bytes, bool, set, frozenset)

def whatis(obj: object) -> Set[str]:
    """
    Given an object, return a set of strings describing what it is. Intended
    to make it slightly easier to introspect things. Pretty ugly, though.

    :param obj: The object to be introspected
    :type obj: object
    :return: A set of strings describing the provided object
    :rtype: Set[str]
    """
    objis: Set[str] = set()

    if inspect.ismodule(obj):
        objis.add("module")
    if inspect.isclass(obj):
        objis.add("class")
    if inspect.ismethod(obj):
        objis.add("method")
    if inspect.isfunction(obj):
        objis.add("function")
    if inspect.isgeneratorfunction(obj):
        objis.add("generatorfunction")
    if inspect.isgenerator(obj):
        objis.add("generator")
    if inspect.iscoroutinefunction(obj):
        objis.add("coroutinefunction")
    if inspect.iscoroutine(obj):
        objis.add("coroutine")
    if inspect.isawaitable(obj):
        objis.add("awaitable")
    if inspect.isasyncgenfunction(obj):
        objis.add("asyncgenfunction")
    if inspect.isasyncgen(obj):
        objis.add("asyncgen")
    if inspect.istraceback(obj):
        objis.add("traceback")
    if inspect.isframe(obj):
        objis.add("frame")
    if inspect.iscode(obj):
        objis.add("code")
    if inspect.isbuiltin(obj):
        objis.add("builtin")
    if inspect.isroutine(obj):
        objis.add("routine")
    if inspect.isabstract(obj):
        objis.add("abstract")
    if inspect.ismethoddescriptor(obj):
        objis.add("methoddescriptor")
    if inspect.isdatadescriptor(obj):
        objis.add("datadescriptor")
    if inspect.isgetsetdescriptor(obj):
        objis.add("getsetdescriptor")
    if inspect.ismemberdescriptor(obj):
        objis.add("memberdescriptor")
    if isinstance(obj, (list, tuple, set, dict, int, float, complex, str, bytes, bool, set, frozenset)):
        objis.add("base")

    return objis


def is_method(obj: object) -> bool:
    w = whatis(obj)
    if w and "routine" in w:
        return True

    return False


def list_members(cls: object) -> Set[str]:
    return set(x for x, _ in inspect.getmembers(cls))


def list_methods(cls: object) -> Set[str]:
    return set(x for x, _ in inspect.getmembers(cls, is_method))


def list_parent_methods(cls: object) -> Set[str]:
    if not isinstance(cls, type):
        bases = [type(cls)]
    else:
        bases = getattr(cls, "__bases__", [])

    return set(itertools.chain.from_iterable(
        list_members(c).union(list_parent_methods(c)) for c in bases))


def subclass_methods(cls: object) -> Set[str]:
    methods = list_methods(cls)
    parent_methods = list_parent_methods(cls)
    return methods.difference(parent_methods)


# list data (stuff that's not an attribute or method)
def is_not_method(obj: object) -> bool:
    return not is_method(obj)


def list_data(cls: object) -> Set[str]:
    if type(cls) is type:
        return set()

    # return set(x for x in dir(cls) if x not in getattr(cls, "__dict__", []))
    return set(x for x, _ in inspect.getmembers(cls, is_not_method))


def list_parent_data(cls: object) -> Set[str]:
    if not isinstance(cls, type):
        bases = [type(cls)]
    else:
        bases = getattr(cls, "__bases__", [])
    return set(itertools.chain.from_iterable(
        list_members(c).union(list_parent_data(c)) for c in bases))


def subclass_data(cls: object) -> Set[str]:
    datas = list_data(cls)
    parent_datas = list_parent_data(cls)

    return datas.difference(parent_datas)
    # return set(x for x in datas if not (x in parent_data))


def fits_in(length: int, num_colsize: int, stops: int = 3, padding: int = 2) -> int:
    stop_list = [x * num_colsize for x in range(1, stops + 2)]

    with_padding = length + padding
    fits = min([s for s in stop_list if s >= with_padding])

    return fits


def print_columns(cols: List[str], indent: int = 4) -> None:
    if len(cols) == 0:
        return

    num_cols = 3
    screen_width = 80
    padding = 2

    column_width = (screen_width - indent) // num_cols
    all_width = screen_width - indent

    # cols = sorted(list(d))
    first = cols[0]
    cols = cols[1:]

    fit = fits_in(len(first), column_width, num_cols, padding)
    print(f"{' ' * indent}{first:{fit}s}", end="")
    remain = all_width - fit

    for data in cols:
        fit = fits_in(len(data), column_width, num_cols, padding)
        if fit > remain:
            print(f"\n{' ' * indent}{data:{fit}s}", end="")
            remain = all_width - fit
        else:
            print(f"{data:{fit}s}", end="")
            remain -= fit

    print("\n")


def print_thing(name: str, mro: Iterable[str], meth: Iterable[str], data: Iterable[str]) -> None:
    print(f"\n===== {name} =====")
    print("  mro:")
    if mro:
        print("    " + " -> ".join(mro))
    else:
        print("    --None--")

    print("\n  Methods:")
    if meth:
        print_columns(sorted(list(meth)))
    else:
        print("    --None--")

    print("\n  Attrs and data:")
    if data:
        print_columns(sorted(list(data)))
    else:
        print("    --None--")


def getmro(t: type) -> Tuple[type, ...]:
    try:
        return inspect.getmro(t)
    except AttributeError:
        return tuple()


def print_class_hierarchy_r(cls: object, seen: Set[type] = None) -> None:
    if seen is None:
        seen = set()

    if not isinstance(cls, type):
        print_class_hierarchy_r(type(cls), seen)

    mro = getmro(type(cls))

    if mro:
        for t in reversed(mro):
            if t in seen:
                continue

            seen.add(t)

            mro_list = [f"<{x.__name__}>" for x in getmro(t)]
            meth_list = subclass_methods(t)
            data_list = subclass_data(t)

            print_thing(str(t), mro_list, meth_list, data_list)

            print_class_hierarchy_r(t, seen)

    if not isinstance(cls, type):
        meth = subclass_methods(cls)
        data = subclass_data(cls)

        print_thing("<instance>", {}, meth, data)


def alintrospect(cls: object) -> None:
    """
    Print information about an object and its ancestors, trying to identify
    at what point in the hierarchy various methods, attributes, and data are
    defined.

    :param cls: object to be introspected
    :type cls: object
    """
    print_class_hierarchy_r(cls)


if __name__ == "__main__":
    from collections import UserDict
    class TestThing(UserDict[str, str]):
        thing_property: str = "zot"

        def __init__(self) -> None:
            setattr(self, "test_attr", "test_attr_value")

        def thing_method(self) -> str:
            return "method"

    test_obj = TestThing()
    print_class_hierarchy_r(test_obj)
