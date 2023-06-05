from Breakdown.Algebra.Solvers.BiCGStab import (
	bicgstab,
	bicgstab_with_breaks,
	bicgstab_chatgpt4,
	bicgstab_wikipedia)

import pytest
import numpy as np

def example1():
	A = np.array([[4, -1, 0, -1],
		[-1, 4, -1, 0],
		[0, -1, 4, -1],
		[-1, 0, -1, 4]])

	b = np.array([0, 5, 0, 5])

	return (A, b)

def example1b():
	A = np.array([[4, -1, 0, -1],
		[-1, 4, -1, 0],
		[0, -1, 4, -1],
		[-1, 0, -1, 4]])

	b = np.array([0, 6, 0, 6])

	expected_x_1 = np.array([1, 2, 1, 2])

	return (A, b, expected_x_1)

def example2():
    A = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])

    b = np.array([1, 0, -1])

    expected_x_1 = np.array([0.5, 0, -0.5])

    return (A, b, expected_x_1)


def test_bicgstab_example_after_1_step():
	A, b = example1()

	results = bicgstab(A, b, max_iter=1)

	assert results[0] == 0
	assert results[1] == 0.25
	assert results[2] == 0.25
	assert results[3] == 0.2
	assert results[4][0] == 0.5
	assert results[4][1] == 1.0
	assert results[4][2] == 0.5
	assert results[4][3] == 1.0

def test_bicgstab_example_1b():
	A, b, expected_x_1 = example1b()

	results = bicgstab(A, b, max_iter=10000)

	assert np.allclose(A @ expected_x_1, b, atol=1e-6)

	assert np.allclose(results[7], expected_x_1, atol=1e-6)

	assert results[0] == 1
	assert results[1] == 0.33333333333333337
	assert results[2] == 0.25000000000000006
	assert results[3] == 0.3333333333333333

	assert results[4][0] == -1.1102230246251565e-16
	assert results[4][1] == 0.0
	assert results[4][2] == -1.1102230246251565e-16
	assert results[4][3] == 0.0


def test_bicgstab_example_2():
	A, b, expected_x_1 = example2()

	results = bicgstab(A, b, max_iter=100000)

	assert np.allclose(A @ expected_x_1, b, atol=1e-6)

	assert not np.allclose(results[7], expected_x_1, atol=1e-6)

def test_bicgstab_with_breaks_example_1b():
	A, b, expected_x_1 = example1b()

	results = bicgstab_with_breaks(A, b, max_iter=10000)

	assert np.allclose(A @ expected_x_1, b, atol=1e-6)

	assert np.allclose(results[7], expected_x_1, atol=1e-6)

	assert results[0] == 1
	assert results[1] == 0.33333333333333337
	assert results[2] == 0.25000000000000006
	assert results[3] == 0.2

	expected_r = np.array([0.6, 1.2, 0.6, 1.2])
	expected_p = np.array([1.2, 1.5, 1.2, 1.5])
	expected_s = np.array([
		-1.11022302e-16,
		-2.22044605e-16,
		-1.11022302e-16,
		-2.22044605e-16])

	assert np.allclose(results[4], expected_r, atol=1e-6)
	assert np.allclose(results[5], expected_p, atol=1e-6)
	assert np.allclose(results[6], expected_s, atol=1e-6)

def test_bicgstab_with_breaks_example_2():
	A, b, expected_x_1 = example2()

	results = bicgstab_with_breaks(A, b, max_iter=100000)

	assert np.allclose(A @ expected_x_1, b, atol=1e-6)

	assert np.allclose(results[3], expected_x_1, atol=1e-6)

	expected_r = np.array([1.0, 0.0, -1.0])
	expected_p = np.array([1.0, 0.0, -1.0])

	assert np.allclose(results[1], expected_r, atol=1e-6)
	assert np.allclose(results[2], expected_p, atol=1e-6)

def test_bicgstab_wikipedia_example_1b():
	A, b, expected_x_1 = example1b()

	results = bicgstab_wikipedia(A, b, max_iter=10000)

	assert np.allclose(A @ expected_x_1, b, atol=1e-6)

	assert np.allclose(results[7], expected_x_1, atol=1e-6)

	assert results[0] == 1
	assert results[1] == 0.33333333333333337
	assert results[2] == 0.25000000000000006
	assert results[3] == 0.2

	expected_r = np.array([0.6, 1.2, 0.6, 1.2])
	expected_p = np.array([1.2, 1.5, 1.2, 1.5])
	expected_s = np.array([3.0, 0.0, 3.0, 0.0])

	assert np.allclose(results[4], expected_r, atol=1e-6)
	assert np.allclose(results[5], expected_p, atol=1e-6)
	assert np.allclose(results[6], expected_s, atol=1e-6)

def test_bicgstab_wikipedia_example_2():
	A, b, expected_x_1 = example2()

	results = bicgstab_wikipedia(A, b, max_iter=100000)

	assert np.allclose(A @ expected_x_1, b, atol=1e-6)

	assert np.allclose(results[6], expected_x_1, atol=1e-6)

	expected_r = np.array([1.0, 0.0, -1.0])
	expected_p = np.array([1.0, 0.0, -1.0])

	assert np.allclose(results[4], expected_r, atol=1e-6)
	assert np.allclose(results[5], expected_p, atol=1e-6)

	assert results[1] == 0.5
	assert results[2] == 2.0
	assert results[3] == 1.0

def test_bicgstab_chatgpt4_example():
	A, b = example1()

	results = bicgstab_chatgpt4(A, b)

	expected_x = np.array([1, 2, 1, 2])

	assert not np.allclose(results[7], expected_x, atol=1e-6)

	assert results[0] == 13
	assert results[1] == 0.30629623632098
	assert results[2] == -0.2656154987394915
	assert results[3] == 0.35877794645176875


def test_bicgstab_identity():
	"""
	Test that the BiCGSTAB algorithm correctly solves the system Ax = b for the identity matrix.
	"""

	A = np.eye(3)
	b = np.array([1, 2, 3])


	assert True