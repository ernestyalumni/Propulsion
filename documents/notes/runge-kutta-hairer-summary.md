# Runge–Kutta before adaptive stepping — five things to understand

### 1. An RK method samples the differential equation at constructed intermediate states.

Write the problem as

$$
\mathbf y'=\mathbf f(t,\mathbf y),\qquad
\mathbf y_n\approx\mathbf y(t_n).
$$

Here $t$ is Hairer's independent variable $x$.

The exact solution satisfies

$$
\mathbf y(t_n+h)-\mathbf y(t_n)
=\int_{t_n}^{t_n+h}\mathbf f(t,\mathbf y(t))\,dt.
$$

RK approximates this integral using evaluations of $\mathbf f$. Because the intermediate solution values are unknown, it constructs stage states:

$$
\begin{aligned}
\mathbf g_i&=\mathbf y_n+h\sum_{j<i}a_{ij}\mathbf k_j,\\
\mathbf k_i&=\mathbf f(t_n+c_i h,\mathbf g_i),\\
\mathbf y_{n+1}&=\mathbf y_n+h\sum_i b_i\mathbf k_i.
\end{aligned}
$$

You should be able to explain each object: $\mathbf g_i$ is a trial state, $\mathbf k_i$ is a derivative, and $h\mathbf k_i$ is a state increment. Only earlier stages enter an explicit method.

The useful insight from Hairer's midpoint discussion is that relatively crude intermediate states can still produce an accurate final update. Individual stages need not have the order of the completed method.

### 2. The tableau is a compact specification of the algorithm, and notation must translate correctly into code.

In Hairer's notation, $a_{ij}$ controls stage construction, $c_i$ gives the stage's time offset, and $b_i$ weights the final update.

Your earlier notation maps as

$$
\alpha_i\longleftrightarrow c_i,\qquad
\beta_{ij}\longleftrightarrow a_{ij},\qquad
c_i^{\text{earlier}}\longleftrightarrow b_i.
$$

NR's displayed stage variables already include the step:

$$
K_i^{\mathrm{NR}}=h\mathbf k_i^{\mathrm{Hairer}}.
$$

This is exactly the sort of distinction that can introduce an extra or missing $h$ in code.

Understand the usual row-sum condition

$$
c_i=\sum_{j<i}a_{ij}:
$$

it makes the stage state agree through first order with the solution at the stated stage time, when starting from exact data.

For classical RK4, also understand why its two midpoint evaluations are different:

$$
\mathbf k_2=\mathbf f\!\left(t_n+\frac h2,\mathbf y_n+\frac h2\mathbf k_1\right),
\qquad
\mathbf k_3=\mathbf f\!\left(t_n+\frac h2,\mathbf y_n+\frac h2\mathbf k_2\right).
$$

They use the same time and generally different states. The final weights are $(1/6,1/3,1/3,1/6)$.

### 3. Order conditions come from matching Taylor expansions. Work through one example yourself.

My suggested minimum is deriving why explicit midpoint has order two. Let

$$
\mathbf f_0=\mathbf f(t_n,\mathbf y_n),
\qquad \mathbf y_n=\mathbf y(t_n)
$$

for this local calculation. Along the exact solution, the chain rule gives

$$
\mathbf y''=\frac{\partial\mathbf f}{\partial t}+\frac{\partial\mathbf f}{\partial y^j}f^j,
$$

Here $f^j$ is component $j$ of $\mathbf f$. Thus

$$
\mathbf y(t_n+h)
=\mathbf y_n+h\mathbf f_0
+\frac{h^2}{2}
\left(\frac{\partial\mathbf f}{\partial t}+\frac{\partial\mathbf f}{\partial y^j}f_0^j\right)
+O(h^3),
$$

with derivatives evaluated at $(t_n,\mathbf y_n)$.

Midpoint constructs

$$
\begin{aligned}
\mathbf k_2
&=\mathbf f\!\left(t_n+\frac h2,\mathbf y_n+\frac h2\mathbf f_0\right)\\
&=\mathbf f_0+\frac h2\frac{\partial\mathbf f}{\partial t}
+\frac h2\frac{\partial\mathbf f}{\partial y^j}f_0^j+O(h^2).
\end{aligned}
$$

Substitute this into $\mathbf y_{n+1}=\mathbf y_n+h\mathbf k_2$. The numerical and exact expansions agree through $h^2$, leaving a one-step error $O(h^3)$.

That calculation is the basic mechanism behind the higher-order theory. Under the row-sum convention, the first two order conditions are

$$
\sum_i b_i=1,\qquad
\sum_i b_i c_i=\frac12.
$$

Higher orders require matching additional derivative terms. Hairer's rooted trees organize those terms. You can learn that machinery progressively; reproducing all eight fourth-order conditions from memory is unnecessary before adaptation.

### 4. Keep stage count, local error, and global error separate.

The stage count $s$ measures the number of stages; the order $p$ describes accuracy. Four arbitrary stages do not automatically give fourth order.

For a method of order $p$, under the relevant smoothness and stability assumptions:

$$
\begin{aligned}
\text{one-step error from exact starting data}&=O(h^{p+1}),\\
\text{global error over a fixed interval}&=O(h^p).
\end{aligned}
$$

There are roughly $T/h$ steps over a fixed interval, which helps explain the loss of one power; error propagation supplies the rest of the argument.

For RK4, halving $h$ should therefore reduce the fixed-interval global error by approximately $16$ in the asymptotic regime. A single step from the same exact starting point has leading error scaling by approximately $32$.

Order also leaves a separate stability question. High order alone does not make a large step safe. For your orbit work, it likewise does not establish long-term energy conservation or symplecticity.

### 5. The missing ingredient before adaptation is an estimate of this step's error.

Knowing "RK4 has local error $O(h^5)$" does not tell you its numerical size here. The leading coefficient depends on the equation and the current state.

Ordinary classical RK4 supplies a new state without a built-in error estimate. Adaptive methods obtain additional information — for example, by comparing one full step with two half steps, or by combining shared stages with two different sets of output weights.

That estimate lets a solver decide whether to accept the trial and how to choose the next $h$. The power of $h$ governing the estimator will determine the step-size adjustment. Keep track of which error is being estimated when you reach embedded pairs.
