from multiprocessing import Process, Pipe
from os import getpid
from datetime import datetime


def local_time(counter):
    """
    @brief Print local Lamport timestamp and actual time on machine executing
    the processes.

    @details Note that printing 'actual' time doesn't make sense in a real
    distributed system, since local clocks won't be synchronized with each
    other.
    """
    return ' (LAMPORT_TIME={}, LOCAL_TIME={}'.format(counter, datetime.now())


def calculate_received_timestamp(received_time_stamp, counter):
    """
    @brief Calculates new timestamp when a process received a message.

    @details Function takes the maximum of the received timestamp and its local
    counter, and increments it with one.
    """
    return max(received_time_stamp, counter) + 1


def event(pid, counter):
    """

    @details event event takes local counter and the process id (pid),
    increments the counter by one, prints a line so we know the event took
    place and returns incremented counter.
    """
    counter += 1
    print('Something happened in {} !'.format(pid) + local_time(counter))


def send_message(pipe, pid, counter):
    counter += 1
    pipe.send(('Empty shell', counter))
    print('Message sent from ' + str(pid) + local_time(counter))
    return counter


def receive_message(pipe, pid, counter):
    """
    
    """
    message, timestamp = pipe.recv()
    counter = calculate_received_timestamp(timestamp, counter)
    print('Message received at ' + str(pid) + local_time(counter))
    return counter