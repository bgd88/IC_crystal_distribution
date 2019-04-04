import errno
import os

def safe_make(dirname):
    try:
        os.mkdir(dirname)
    # except OSError as exc:
    #     if exc.errno != errno.EEXIST:
    #         raise
    except:
        pass

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped
