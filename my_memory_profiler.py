# memory_profiler.py
from memory_profiler import profile

def memory_profiled(func):
    """
    A decorator that uses memory_profiler's @profile.
    To use this, you must run the python script with a special flag:
    
    python -m memory_profiler your_script.py
    
    This decorator is just a placeholder to remind us of the syntax.
    You will typically apply the @profile decorator directly.
    """
    # The actual decorator is @profile from the library itself.
    # This file serves as a documentation placeholder.
    print("To use memory profiling, decorate a function with @profile")
    print("and run: 'python -m memory_profiler your_script.py'")
    return profile(func)

# Example Usage (don't run this file directly)
if __name__ == '__main__':
    
    @profile
    def my_function():
        a = [1] * (10 ** 6) # Allocate 1 million integers
        b = [2] * (2 * 10 ** 7) # Allocate 20 million integers
        del b
        return a

    my_function()