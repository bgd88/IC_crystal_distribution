import numpy as np
from contextlib import contextmanager
import colorama
import functools

def are_equal(array_list, N_eps = 10, tol = None):
    ''' Check if x and y are equal to within machine percision.
        If an array is passed, will return False if a single
        element is not within machine percision.
    '''
    # Get floating point machine percision
    if tol is None:
        tol = N_eps * np.finfo(float).eps
    # Function which compares whether all the elements of two arrays
    eq = lambda x, y: ( np.abs(x-y) < np.maximum.reduce([np.abs(x), np.abs(y), \
                                np.ones_like(x)])*tol).all()
    # compare first array with all the rest
    bool_list = map(lambda y: eq(array_list[0], y), array_list)
    # return true if all true, else false
    return all(bool_list)

def zero_threshold(func, N_eps = 1):
    # Get floating point machine percision
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        #TODO: Figure out how to pass arguments to decorators
        M = func(*args, **kwargs)
        tol = N_eps * np.finfo(float).eps
        M_max = np.max(M)
        M[np.abs(M/M_max) < tol] = 0.0
        return M
    return wrapper_decorator

###########################
# Print Arrays with color #
###########################

def color_sign(x):
    if x > 0:
        c = colorama.Fore.GREEN
        x = '+{:2.2E}'.format(x)
    elif x < 0:
        c = colorama.Fore.RED
        x = '{:2.2E}'.format(x)
    else:
        c = colorama.Fore.WHITE
        x = '{:09f}'.format(x)
    return f'{c}{x}'

@contextmanager
def printoptions(**kwds):
    # TO DO: docstring...
    opts = np.get_printoptions()
    np.set_printoptions(**kwds)
    yield
    np.set_printoptions(**opts)
    # TODO: Figure out why this is needed.
    print(colorama.Style.RESET_ALL)

def print_cs(M):
    with printoptions(linewidth=100, formatter={'float': color_sign}):
        print(M)
        print("\n")
