"""
@ref https://docs.python.org/3/library/multiprocessing.html

@details In multiprocessing, processes are spawned by creating a Process object
and then calling its start() method.
"""

from multiprocessing import (Pipe, Process)
import multiprocessing as MP
import pytest
import os


def f(name):
    print('hello', name)
    return 'hello' + str(name)

def trivial_multiprocess_example():
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def pids_f(name):
    info('function f')
    print('hello', name)

def pid_example():
    info('main line')
    p = Process(target=pids_f, args=('bob',))
    p.start()
    p.join()

def pipe_f(connection):
    connection.send([42, None, 'hello'])
    connection.close()

def pipe_example():
    parent_connection, child_connection = Pipe()
    p = Process(target=pipe_f, args=(child_connection,))
    p.start()
    received_object = parent_connection.recv()
    print(received_object) # prints "[42, none, 'hello]"
    p.join()
    return received_object

def test_trivial_multiprocess_example(capfd):
    trivial_multiprocess_example()
    captured = capfd.readouterr()

    assert captured.out == "hello bob\n"

def test_pid_example(capfd):
    pid_example()
    captured = capfd.readouterr()
    expected_substring_1 = (
        "main line\n"
            + "module name: test_MultipleProcesses_library\n"
            + "parent process:")
    expected_substring_2 = (
        "function f\n"
            + "module name: test_MultipleProcesses_library\n"
            + "parent process:")

    assert expected_substring_1 in captured.out
    assert expected_substring_2 in captured.out

def test_pipe_example(capfd):
    received_object = pipe_example()
    captured = capfd.readouterr()

    assert captured.out == "[42, None, 'hello']\n"

    assert(type(received_object) == type([]))
    assert(len(received_object) == 3)
    assert(received_object[0] == 42)
    assert(received_object[1] == None)
    assert(received_object[2] == 'hello')
