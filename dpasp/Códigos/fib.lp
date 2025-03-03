#python
def fib(X):
    if (X == 0 or X == 1): return 1
    return (fib(X-1) + fib(X-2))

#end.

fib(X) ~ fib(X).
#query fib(2).
#semantics maxent.
