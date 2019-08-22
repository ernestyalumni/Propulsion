# -*- coding: utf-8 -*-
"""
@file decorators.py
@brief Studying decorators in Python with examples.
@ref https://www.python-course.eu/python3_decorators.php
"""
"""
@ref https://www.python-course.eu/python3_decorators.php
@details Decoration occurs in the line before the function header. The "@" is
followed by the decorator function name.

Decorator in Python is a callable Python object that's used to modify a
function, method, or class definition. The original object, the one which is
going to be modified, is passed to a decorator as an argument. The decorator
returns a modified object, e.g. a modified function, which is bound to the name
used in the definition.

EY: 20190420 A decorator is a transformation on a function, and transforms a
function into another function. The "signature", e.g. the "domain" of the
resulting function need not be the same as the original function, nor for the
"range."
"""
from functools import wraps

def our_decorator(function):
    
    def function_wrapper(x):
        print("Before calling " + function.__name__)
        function(x)
        print("After calling " + function.__name__)
        
    return function_wrapper

@our_decorator
def foo(x):
    print("Hi, foo has been called with " + str(x))
    
foo("Hi")


def our_decorator_explicit(function):
    def function_wrapper(x):
        print("Before calling " + function.__name__)
        result = function(x)
        print(result)
        print("After calling " + function.__name__)
    return function_wrapper

@our_decorator_explicit
def succession(n):
    return n + 1

succession(10)

# Use Cases for Decorators

## Argument checking with a decorator; an example of functional programming.

def argument_test_natural_number(f):
    
    def helper(x):
        
        if type(x) == int and x > 0:
            return f(x)
        else:
            raise Exception("Argument is not an integer")
            
    return helper

@argument_test_natural_number
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print("\n factorial \n")    
    
# Expected:
# 1 1
# 2 2
# 3 6 
# 4 24 ...
for i in range(1, 10):
    print(i , factorial(i))

# Expected: Caught an exception: full error: Argument is not an integer
try:   
    print(factorial(-1))
except Exception as e:
    print("Caught an exception: full error: ", e)

## Another example of argument checking with a decorator.

def argument_test_natural_number_2(f):
    def helper(x):

        if type(x) == int and x > 0:
            return f(x)
        else:
            print(x)
            try:
                x = int(x)
                if x < 0:
                    return f(-x)
                else:
                    return f(x)
            except ValueError as e:
                print("Caught a ValueError: ", e)
    return helper

@argument_test_natural_number_2
def factorial2(n):
    if n == 1:
        return 1
    elif n == 0:
        return 0
    else:
        return n * factorial2(n - 1)

for i in range(1, 5):
    print(i, factorial2(i))

print(factorial2(-42))
print(factorial2("Wrong"))
print(factorial2("-41"))

    
## Counting Function Calls with Decorators; an example of adding "state" to a 
## function.

def call_counter(function):
    """
    @details To be precise, we can use this decorator solely for functions with
    exactly one parameter.
    """
    def helper(x):
        helper.calls += 1
        return function(x)
    helper.calls = 0
    
    return helper

def call_counter_multiple_arguments(function):
    """
    @details We'll use the *args and **kwargs notation to write decorators
    which can cope with functions with an arbitrary number of positional and
    keyword parameters.
    """
    def helper(*args, **kwargs):
        helper.calls += 1
        return function(*args, **kwargs)
    
    helper.calls = 0
    
    return helper

@call_counter_multiple_arguments
def succession2(x):
    return x + 1

@call_counter_multiple_arguments
def multiply1(x, y=1):
    return x * y + 1

print(succession2.calls)
for i in range(10):
    succession2(i)

print("\n Run multiply1 \n")    
multiply1(3, 4)
multiply1(4)
multiply1(y=3, x=2)

print(succession2.calls)
print(multiply1.calls)

## Decorators with Parameters: functionals

""" "Old" way or "not smart" way
def evening_greeting(function):
    def function_wrapper(x):
        print("Good evening, " + function.__name__ + " returns:")
        function(x)
    return function_wrapper

def morning_greeting(function):
    def function_wrapper(x):
        print("Good morning, " + function.__name__ + " returns:")
        function(x)
    return function_wrapper

@evening_greeting
def foo42(x):
    print(42)
    
foo42("Hi")
"""

def greeting(expression):
    def greeting_decorator(function):
        def function_wrapper(x):
            print(expression + ", " + function.__name__ + " returns:")
            function(x)
        return function_wrapper
    return greeting_decorator

@greeting("HihiHi")
def foo42(x):
    print(42)

print("\n Demonstrate decorators with parameters, i.e. functionals\n")
    
foo42("Hi")

## Using wraps from functools

"""
@ref https://docs.python.org/2/library/functools.html
@brief functools module is for higher-order functions: functions that act on or
return other functions. In general, any callable object can be treated as a
function for the purposes of this module.
"""

"""
@fn functools.wraps(wrapped[, assigned][, updated])
@ref https://docs.python.org/2/library/functools.html#functools.wraps
@brief This is a convenience function for invoking update_wrapper() as a
function decorator when defining a wrapper function. It's equivalent to

partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated).

functools.partial return a partial object which when called will behave like
func called with the positional arguments args and keyword arguments keywords.

functools.update_wrapper updates a wrapper function to look like the wrapped
function.
"""

def greeting_wrapped(function):
    
    @wraps(function)
    def function_wrapper(x):
        """
        function_wrapper of greeting
        """
        print("Hi, " + function.__name__ + " returns:")
        return function(x)
        
    return function_wrapper

@greeting_wrapped
def f4(x):
    """
    Just some silly funciton
    """
    return x + 4

print("\n Using wraps from functools \n")

f4(10)

# Classes instead of Functions for decorators

"""
@brief __call__ method.

@details A decorator is simply a callable object that takes a function as an
input parameter.

A function is a callable object, but there are other callable objects.
A callable object is an object which can be used and behaves like a function
but might not be a function. It's possible to define classes in a way that the
instances will be callable objects. The __call__ method is called, if the
instance is called "like a function", i.e. using brackets (i.e. parentheses)
"""

print("\n Examples of functors with __call__ method\n")

class A:
    def __init__(self):
        print("An instance of A was initialized")
    def __call__(self, *args, **kwargs):
        print("Arguments are: ", args, kwargs)
        
x = A()
print("Now calling the instance: ")
x(3, 4, x=11, y=10)
print("Let's call it again:")
x(3, 4, x=11, y=10)


class Fibonacci:
    def __init__(self):
        self.cache = {}
        
    def __call__(self, n):
        if n not in self.cache:
            if n == 0:
                self.cache[0] = 0
            elif n == 1:
                self.cache[1] = 1
            else:
                self.cache[n] = self.__call__(n - 1) + self.__call__(n - 2)
        return self.cache[n]
    
fib = Fibonacci()

for i in range(15):
    print(fib(i), end=", ")
    
## Using a Class as a Decorator
    
def decorator1(f):
    def helper():
        print("Decorating", f.__name__)
        f()
    return helper

@decorator1
def foo():
    print("inside foo()")
    
foo()

# Rewrite the following decorator as a class
class decorator2:
    def __init__(self, f):
        self.f = f
        
    def __call__(self):
        print("Decorating", self.f.__name__)
        self.f()
        
@decorator2
def foo_inside():
    print("inside foo()")
    
foo_inside()