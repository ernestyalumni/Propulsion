import numpy as np

def bicgstab(A, b, x0 = None, max_iter=1000, tol=1e-6):
	"""
	Solves the linear system Ax = b using the BiCGSTAB algorithm.

	Parameters
	----------
	A : ndarray
	The coefficient matrix.
	b : ndarray
	The right-hand side vector.
	x0 : ndarray, optional
	Initial guess for the solution. If None, a random vector is used.
	tol : float, optional
	Tolerance for convergence. The algorithm terminates when the norm of the residual is less than
	or equal to tol.
	max_iter : int, optional
	Maximum number of iterations. The algorithm terminates after maxiter iterations even if the
	specified tolerance has not been achieved.

	Returns
	-------
	x : ndarray
	The solution to the linear system.
	"""
	if x0 is None:
		x = np.zeros(A.shape[0])
	else:
		x = x0.copy()

	r = b - A @ x
	rtilde_0 = r.copy()

	p = r.copy()

	for i in range(max_iter):
		rho = np.dot(rtilde_0, r)

		v = A @ p
		alpha_j = rho / np.dot(rtilde_0, v)

		s = r - alpha_j * v

		t = A @ s
		omega_j = np.dot(t, s) / np.dot(t, t)

		x = x + alpha_j * p + omega_j * s
		r = s - omega_j * t

		if np.linalg.norm(r) < tol:
			break

		beta_j = alpha_j / omega_j * np.dot(rtilde_0, r) / rho

		p = r + beta_j * (p - omega_j * v)

	return (i, alpha_j, beta_j, omega_j, r, p, s, x)


def bicgstab_with_breaks(A, b, x0 = None, max_iter=1000, tol=1e-6):
	if x0 is None:
		x = np.zeros(A.shape[0])
	else:
		x = x0.copy()

	r = b - A @ x
	rtilde_0 = r.copy()

	p = r.copy()

	for i in range(max_iter):
		rho = np.dot(rtilde_0, r)

		v = A @ p
		alpha_j = rho / np.dot(rtilde_0, v)

		s = r - alpha_j * v

		if np.linalg.norm(s) < tol:
			x = x + alpha_j * p
			break

		t = A @ s
		omega_j = np.dot(t, s) / np.dot(t, t)

		x = x + alpha_j * p + omega_j * s
		r = s - omega_j * t

		if np.linalg.norm(r) < tol:
			break

		beta_j = alpha_j / omega_j * np.dot(rtilde_0, r) / rho

		p = r + beta_j * (p - omega_j * v)

		if (np.abs(np.dot(r, rtilde_0)) < tol):
			rtilde_0 = r
			p = r

	if (i > 0):
		return (i, alpha_j, beta_j, omega_j, r, p, s, x)
	else:
		return (i, r, p, x)


def bicgstab_wikipedia(A, b, x0 = None, max_iter=1000, tol=1e-6):
	if x0 is None:
		x = np.zeros(A.shape[0])
	else:
		x = x0.copy()

	r = b - A @ x
	rtilde_0 = r.copy()

	rho = 1.0
	alpha = 1.0
	omega = 1.0

	v = np.zeros(A.shape[0])
	p = np.zeros(A.shape[0])

	for i in range(max_iter):

		rho_previous = rho
		rho = np.dot(rtilde_0, r)
		beta = (rho / rho_previous) * (alpha / omega)
		p = r + beta * (p - omega * v)
		v = A @ p
		alpha = rho / np.dot(rtilde_0, v)

		h = x + alpha * p

		# If h is accurate enough, then set x = h and quit.
		if np.linalg.norm(b - A @ h) < tol:
			x = h
			break
		s = r - alpha * v
		t = A @ s
		omega = np.dot(t, s) / np.dot(t, t)
		x = h + omega * s

		# If x_i is accurate enough, then quit.
		if np.linalg.norm(b - A @ x) < tol:
			break

		r = s - omega * t

	if (i > 0):
		return (i, alpha, beta, omega, r, p, s, x)
	else:
		return (i, alpha, beta, omega, r, p, x)

def bicgstab_chatgpt4(A, b, x0 = None, max_iter=1000, tol=1e-6):

	if x0 is None:
		x = np.zeros(A.shape[0])
	else:
		x = x0.copy()

	r = b - np.dot(A, x)
	r_tilde = r.copy()
	p = r.copy()
	rho = 1.0
	alpha = 1.0
	omega = 1.0

	for i in range(max_iter):
	    rho_prev = rho
	    rho = np.dot(r_tilde, r)
	    beta = (rho / rho_prev) * (alpha / omega)

	    r_hat = r - omega * np.dot(A, p)
	    v = np.dot(A, p)
	    alpha = rho / np.dot(r_tilde, v)

	    s = r_hat - alpha * v
	    t = np.dot(A, s)
	    omega = np.dot(t, s) / np.dot(t, t)

	    x += alpha * p + omega * s
	    r = s - omega * t

	    if np.linalg.norm(r) < tol:
	        break

	    p = r + beta * (p - omega * np.dot(A, p))

	return (i, alpha, beta, omega, r, p, s, x)