class Package(object):
    def __init__(self, *children, children_type=None):
        assert len(children) != 0 or children_type
        self.children = children
        self.children_type = children_type if children_type else type(children[0])

    def __add__(self, other): return self.__getattribute__("__add__")(other)
    def __sub__(self, other): return self.__getattribute__("__sub__")(other)
    def __mul__(self, other): return self.__getattribute__("__mul__")(other)
    def __div__(self, other): return self.__getattribute__("__div__")(other)
    def __mod__(self, other): return self.__getattribute__("__mod__")(other)
    def __divmod__(self, other): return self.__getattribute__("__divmod__")(other)
    def __pow__(self, other, modulo): return self.__getattribute__("__pow__")(other, modulo)
    def __lshift__(self, other): return self.__getattribute__("__lshift__")(other)
    def __rshift__(self, other): return self.__getattribute__("__rshift__")(other)
    def __and__(self, other): return self.__getattribute__("__and__")(other)
    def __xor__(self, other): return self.__getattribute__("__xor__")(other)
    def __or__(self, other): return self.__getattribute__("__or__")(other)
    def __radd__(self, other): return self.__getattribute__("__radd__")(other)
    def __rsub__(self, other): return self.__getattribute__("__rsub__")(other)
    def __rmul__(self, other): return self.__getattribute__("__rmul__")(other)
    def __rdiv__(self, other): return self.__getattribute__("__rdiv__")(other)
    def __rmod__(self, other): return self.__getattribute__("__rmod__")(other)
    def __rdivmod__(self, other): return self.__getattribute__("__rdivmod__")(other)
    def __rpow__(self, other): return self.__getattribute__("__rpow__")(other)
    def __rlshift__(self, other): return self.__getattribute__("__rlshift__")(other)
    def __rrshift__(self, other): return self.__getattribute__("__rrshift__")(other)
    def __rand__(self, other): return self.__getattribute__("__rand__")(other)
    def __rxor__(self, other): return self.__getattribute__("__rxor__")(other)
    def __ror__(self, other): return self.__getattribute__("__ror__")(other)
    def __neg__(self): return self.__getattribute__("__neg__")()
    def __pos__(self): return self.__getattribute__("__pos__")()
    def __abs__(self): return self.__getattribute__("__abs__")()
    def __invert__(self): return self.__getattribute__("__invert__")()
    def __complex__(self): return self.__getattribute__("__complex__")()
    def __int__(self): return self.__getattribute__("__int__")()
    def __long__(self): return self.__getattribute__("__long__")()
    def __float__(self): return self.__getattribute__("__float__")()

    def __getattribute__(self, name):
        def wrap_attr(attr_name, element_list):
            def get_attr(*args, **kwargs):
                new_elems = []
                for element in element_list:
                    if isinstance(element, list):
                        new_elem = wrap_attr(attr_name, element)(*args, **kwargs)
                    else:
                        attr = getattr(element, attr_name)
                        new_elem = attr(*args, **kwargs) if callable(attr) else attr
                    new_elems.append(new_elem)
                return Package(*new_elems)
            return get_attr if callable(getattr(self.children_type, name)) else get_attr()

        if name in ("children", "children_type", "__getattribute__"):
            return object.__getattribute__(self, name)
        return wrap_attr(name, self.children)

def flatten_zip(*args):
    args = [flatten(arg) for arg in args]
    return zip(*args)

def nested_map(fn, nested_list):
    return [nested_map(fn, e) if isinstance(e, list) else fn(e) for e in nested_list]

def nested_builder(target, *args):
    for elements in zip(args):
        if isinstance(elements[0], list):
            cb_element_list = []
            yield nested_builder(cb_element_list, *elements)
            target.append(cb_element_list)
        else:
            yield target, elements

def flatten(nested_list):
    items = []
    for elem in nested_list:
        if isinstance(elem, list):
            items.extend(flatten(elem))
        else:
            items.append(elem)
    return items
