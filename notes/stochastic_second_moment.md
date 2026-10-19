# Steady-state second moment in the frequency domain — additive *and* multiplicative white noise

Short theory note (draft, 2026-10-19). Starting point: Fodor G., PhD dissertation, Sec. 4.3
(PSD-based variance of the SSTM, eqs. 4.18–4.30).

## 1. Setting

Linear time-invariant SDDE in Itô form (dissertation eq. 4.11):

$$
\mathrm d\mathbf x = \big(\mathbf A\mathbf x(t) + \mathbf B\mathbf x(t-\tau) + \mathbf c\big)\,\mathrm dt
 + \big(\boldsymbol\alpha\mathbf x(t) + \boldsymbol\beta\mathbf x(t-\tau) + \boldsymbol\gamma\big)\,\mathrm dW_t .
$$

* First moment: $\mathbb E[\mathrm dW]=0$ and, in Itô, $\mathbf x(t)$ is independent of $\mathrm dW_t$,
  so $\mathbf m=\mathbb E\mathbf x$ obeys the deterministic DDE → stability = the existing Nyquist count.
* Fluctuation $\mathbf y=\mathbf x-\mathbf m$ (stationary $\mathbf m$):

$$
\mathrm d\mathbf y = (\mathbf A\mathbf y+\mathbf B\mathbf y_\tau)\,\mathrm dt
 + \big(\tilde{\boldsymbol\gamma} + \boldsymbol\alpha\mathbf y+\boldsymbol\beta\mathbf y_\tau\big)\mathrm dW,
\qquad \tilde{\boldsymbol\gamma}=\boldsymbol\gamma+(\boldsymbol\alpha+\boldsymbol\beta)\mathbf m .
$$

  For the GSTM, $(\boldsymbol\alpha+\boldsymbol\beta)\mathbf m=0$, so $\tilde{\boldsymbol\gamma}=\boldsymbol\gamma$.

## 2. Key observation: multiplicative white noise is still *white* forcing

Define the forcing $\boldsymbol\eta(t)\,\mathrm dt = \mathbf s(t)\,\mathrm dW_t$ with
$\mathbf s=\tilde{\boldsymbol\gamma}+\boldsymbol\alpha\mathbf y+\boldsymbol\beta\mathbf y_\tau$.
Because $\mathbf s(t)$ is non-anticipating and the increments of $W$ after $t$ are independent of it,

$$
\mathbb E\big[\boldsymbol\eta(t)\boldsymbol\eta(t')^{\mathsf T}\big] = \mathbf Q\,\delta(t-t'),\qquad
\mathbf Q=\mathbb E\big[\mathbf s\,\mathbf s^{\mathsf T}\big]\ \ (\text{constant in steady state}).
$$

So $\boldsymbol\eta$ has a **flat PSD**, exactly as in the additive case. The only difference is that its
intensity $\mathbf Q$ is *unknown*, because it depends on the second moments of the response.
The response is still the deterministic convolution $\mathbf y = \mathbf h_r * \boldsymbol\eta$, and by the
Itô isometry (the same step as dissertation eq. 4.27):

$$
\mathbb E[\mathbf y\mathbf y^{\mathsf T}] = \frac1{2\pi}\int_{-\infty}^{\infty}
 \mathbf H_r(\omega)\,\mathbf Q\,\mathbf H_r(\omega)^{*}\,\mathrm d\omega,
\qquad \mathbf H_r(\omega) = \big(\mathrm i\omega\mathbf I-\mathbf A-\mathbf B e^{-\mathrm i\omega\tau}\big)^{-1}.
$$

**No convolution in frequency appears.** The product $\mathbf s(t)\,\xi(t)$ would normally give
$S_\eta = S_s * S_\xi$. But $S_\xi$ is constant, so the convolution collapses to
$\mathbb E[\mathbf s\mathbf s^{\mathsf T}]$, which is a single number (matrix).

## 3. Self-consistency (closure) — a small linear system

Let the noise-modulating output be $\mathbf z = \boldsymbol\alpha\mathbf y+\boldsymbol\beta\mathbf y_\tau$, with
transfer function from the forcing

$$
\mathbf G(\omega) = \big(\boldsymbol\alpha + \boldsymbol\beta e^{-\mathrm i\omega\tau}\big)\,\mathbf H_r(\omega).
$$

Since $\mathbb E\mathbf z=0$: $\ \mathbf Q = \tilde{\boldsymbol\gamma}\tilde{\boldsymbol\gamma}^{\mathsf T} + \mathbb E[\mathbf z\mathbf z^{\mathsf T}]$, i.e.

$$
\boxed{\ \mathbf Q = \tilde{\boldsymbol\gamma}\tilde{\boldsymbol\gamma}^{\mathsf T} + \mathcal L(\mathbf Q),\qquad
\mathcal L(\mathbf Q) = \frac1{2\pi}\int_{-\infty}^{\infty}\mathbf G(\omega)\,\mathbf Q\,\mathbf G(\omega)^{*}\,\mathrm d\omega\ }
$$

* This is a linear equation in at most $n(n+1)/2$ unknowns. In practice $\mathbf s$ acts through the
  input matrix, so the number of unknowns is the number of noise channels.
* **Mean-square stability** ⇔ deterministic (first-moment) stability **and** spectral radius $\rho(\mathcal L)<1$.
  This is the stochastic small-gain theorem (Willems & Blankenship 1971; Lu & Skelton 2002;
  Bamieh & Filo 2020). The references are from memory and still need checking.
* With several **independent** Wiener processes $W_k$ entering through scalar channels:
  $q_k=\tilde\gamma_k^2+\sum_l M_{kl}q_l$, $M_{kl}=\frac1{2\pi}\int|G_{kl}|^2\mathrm d\omega$.
  This gives $\mathbf q=(\mathbf I-\mathbf M)^{-1}\tilde{\boldsymbol\gamma}^2$, and the MS-stability condition is the Perron root $\rho(\mathbf M)<1$.

### Scalar channel (GSTM turning, dissertation eqs. 4.10–4.12)

Forcing on the acceleration: $K\sigma\,(1 + y_\tau - y)\,\xi$. With
$H_r(\omega) = \big[-\omega^2+2\mathrm i\zeta\omega+1+K(1-e^{-\mathrm i\omega\tau})\big]^{-1}$:

$$
\rho = \frac{K^2\sigma^2}{2\pi}\int_{-\infty}^{\infty}\big|e^{-\mathrm i\omega\tau}-1\big|^2\,|H_r(\omega)|^2\,\mathrm d\omega,
\qquad
\boxed{\ \operatorname{Var}(y)=\frac{\operatorname{Var}_{\text{SSTM}}(y)}{1-\rho}\ },\quad
\operatorname{Var}_{\text{SSTM}}=\frac{K^2\sigma^2}{2\pi}\int|H_r|^2\mathrm d\omega .
$$

* The cost is **two scalar FRF integrals** instead of one, over the same $D(\mathrm i\omega)$ samples.
* $\rho=O(\sigma^2)$. This explains the dissertation's finding that GSTM ≈ SSTM for $\sigma=0.01$
  ($\rho\sim10^{-4}$ away from the boundary).
* The GSTM MS-stability boundary, which the dissertation computes with SSD, is the curve $\rho(\Omega,K)=1$. It is a contour of a
  frequency integral, and MDBM can trace it directly.

## 4. Numerical check (Python, `check_gstm.py`)

| Case | ρ | Additive-only | Formula | Reference |
|---|---|---|---|---|
| No delay, $y''+0.2y'+1.5y=0.6(1-y)\xi$ | 0.600 | 0.600 | **1.5000** | Lyapunov moment eq.: **1.5000** (exact) |
| GSTM, $K=0.2,\ \Omega=1,\ \sigma=1$ | 0.077 | 0.0958 | 0.1039 | Monte Carlo: 0.106 |
| GSTM, $K=0.2,\ \Omega=0.6,\ \sigma=0.7$ | 0.368 | 0.2348 (MC: 0.235) | **0.3716** | Monte Carlo: **0.367 / 0.360** ($\Delta t$=0.002 / 0.001) |
| GSTM, $K=0.2,\ \Omega=0.6,\ \sigma=1.5$ | 1.69 | 1.078 | ρ>1 → MS unstable | MC $E[y^2]$ grows (58 at $T=600$) |

Monte Carlo used symplectic Euler–Maruyama with 2×4000 trajectories. Explicit Euler adds energy to the
lightly damped oscillator and overshoots by 20–50 %. Its estimates still converge towards the formula as $\Delta t\to0$.
Near $\rho\to1$ the stationary distribution is heavy-tailed (the fourth moment diverges first), so
Monte Carlo converges poorly there. The frequency formula has no such problem.

## 5. When it stops being this simple

| Situation | Consequence |
|---|---|
| Stratonovich noise | Convert to Itô first: drift $\mathbf A\to\mathbf A+\tfrac12\boldsymbol\alpha^2$ (no delay part). After that, the same method applies. |
| **Coloured multiplicative noise** (e.g. FOF/SOF filtered models) | $\mathbf s$ and $\xi$ are correlated over time, so $S_\eta=S_s*S_\xi$ is a genuine convolution, and second moments couple to higher ones (bilinear system). This is where the problem really becomes higher order. Options: augment the filter state (then the bilinear moment hierarchy needs closure) or accept an approximation. |
| Coloured **additive** noise | Still trivial: $\int H_r Q(\omega) H_r^*$ with $Q(\omega)=S_\xi(\omega)$. |
| Time-periodic systems (milling, SSV; Hill / HTF) | $\mathbb E[s^2](t)$ becomes $T$-periodic. The closure becomes a linear system over its Fourier harmonics (a Toeplitz block built from harmonic transfer functions). This is a discrete convolution in the harmonic index, but still not a continuous one in $\omega$, so it remains tractable. |
| Non-white multiplicative excitation that is periodic/deterministic | This is not a noise question; it belongs to Hill theory. |

## 6. Fit with the Interpolated Nyquist code

* For a scalar channel, $G(\mathrm i\omega)=N(\mathrm i\omega)/D(\mathrm i\omega)$ with the same characteristic function $D$
  already evaluated along the imaginary axis (with `σ = 0`). Only a numerator (an adjugate entry, e.g.
  $e^{-\mathrm i\omega\tau}-1$ for turning) has to be supplied.
* The integrands $|N/D|^2$ peak where roots approach the axis (stochastic resonance). These are the same
  places the existing quadgk peak repair already handles. Tails decay like $\omega^{-2(\deg D-\deg N)}$, so a
  truncation plus analytic tail term is enough.
* One pass over $\omega$ gives the winding number (first-moment stability), $\int|H_r|^2$ and $\int|G|^2$ together,
  so the variance map and the MS-stability boundary $\rho=1$ come out of the same GPU kernel.

---

# Part II — General n-dimensional LTI delay system: first moment, second-moment spectrum, MS stability, stationary second moment

Scope: time-invariant, retarded, single delay $\tau$; **white** noise, Itô; additive and multiplicative noise,
modulated by both the present and the delayed state. Time-periodic and coloured-noise cases are excluded (see Sec. 5).

## 7. Model

State $\mathbf x\in\mathbb R^N$ (first-order form, $N=2\times$DOF for mechanical systems). There are $m$ independent Wiener processes $W_k$, and the noise
enters through the input matrix $\mathbf E\in\mathbb R^{N\times p}$ ($p$ = noise channels, e.g. force equations):

$$
\mathrm d\mathbf x = \big(\mathbf A\mathbf x+\mathbf B\mathbf x_\tau+\mathbf c\big)\mathrm dt
+\mathbf E\sum_{k=1}^{m}\big(\boldsymbol\gamma_k+\mathbf C_k\mathbf x+\mathbf D_k\mathbf x_\tau\big)\mathrm dW_k ,
\qquad \mathbf x_\tau=\mathbf x(t-\tau).
$$

($\mathbf C_k,\mathbf D_k\in\mathbb R^{p\times N}$; the dissertation's $\boldsymbol\alpha,\boldsymbol\beta,\boldsymbol\gamma$ are $\mathbf E\mathbf C_k,\ \mathbf E\mathbf D_k,\ \mathbf E\boldsymbol\gamma_k$.)

Deterministic objects that the package already evaluates:

$$
\boldsymbol\Delta(\lambda)=\lambda\mathbf I-\mathbf A-\mathbf B e^{-\lambda\tau},\qquad D(\lambda)=\det\boldsymbol\Delta(\lambda).
$$

New transfer matrices. The response to the noise input is $\mathbf H(\lambda)=\boldsymbol\Delta(\lambda)^{-1}\mathbf E$ ($N\times p$). The noise-modulating
output of channel $k$ is

$$
\mathbf G_k(\lambda)=\big(\mathbf C_k+\mathbf D_k e^{-\lambda\tau}\big)\,\boldsymbol\Delta(\lambda)^{-1}\mathbf E\qquad(p\times p).
$$

## 8. Moments

### 8.1 First moment

$\mathbb E[\mathbf E\mathbf s_k\,\mathrm dW_k]=0$ (Itô), so $\mathbf m(t)=\mathbb E\mathbf x$ solves the **deterministic** DDE exactly:

* Spectrum: the roots of $D(\lambda)$. **First-moment stability** is the existing winding count / unwrap march, unchanged.
* Stationary mean: $\mathbf m^*=-(\mathbf A+\mathbf B)^{-1}\mathbf c$.

### 8.2 Fluctuation and the forcing

$\mathbf y=\mathbf x-\mathbf m^*$ satisfies the same equation with $\mathbf c\to0$ and the effective additive part
$\tilde{\boldsymbol\gamma}_k=\boldsymbol\gamma_k+(\mathbf C_k+\mathbf D_k)\mathbf m^*$. Let $\mathbf h(t)$ be the fundamental solution, i.e. the inverse Laplace transform of $\boldsymbol\Delta^{-1}$.
Then (variation of constants):

$$
\mathbf y(t)=\mathbf y_{\text{free}}(t)+\int_0^t\mathbf h(t-u)\,\mathbf E\sum_k\mathbf s_k(u)\,\mathrm dW_k(u),
\qquad \mathbf s_k=\tilde{\boldsymbol\gamma}_k+\mathbf C_k\mathbf y+\mathbf D_k\mathbf y_\tau .
$$

By the Itô isometry, and because the $W_k$ are independent, only one $p\times p$ matrix carries all the noise information:

$$
\mathbf q(t)=\sum_k\mathbb E\big[\mathbf s_k(t)\mathbf s_k(t)^{\mathsf T}\big].
$$

### 8.3 Second-moment dynamics = a matrix renewal (Volterra) equation

With $\mathbf g_k(u)$ the impulse response of $\mathbf G_k$:

$$
\mathbf q(t)=\mathbf q_0(t)+\int_0^t\mathcal K(t-u)[\mathbf q(u)]\,\mathrm du,\qquad
\mathcal K(u)[\mathbf q]=\sum_k\mathbf g_k(u)\,\mathbf q\,\mathbf g_k(u)^{\mathsf T},
$$

$$
\mathbf P(t)=\mathbb E[\mathbf y\mathbf y^{\mathsf T}]=\mathbb E\big[\mathbf y_{\text{free}}\mathbf y_{\text{free}}^{\mathsf T}\big]+\int_0^t\mathbf h(t-u)\mathbf E\,\mathbf q(u)\,\mathbf E^{\mathsf T}\mathbf h(t-u)^{\mathsf T}\mathrm du .
$$

$\mathbf q_0(t)$ contains $\sum_k\tilde{\boldsymbol\gamma}_k\tilde{\boldsymbol\gamma}_k^{\mathsf T}$ plus terms from the initial function, which decay like the deterministic
modes. **The kernel is completely positive**: it maps positive semidefinite (PSD) matrices to PSD matrices.

### 8.4 Spectrum of the second-moment dynamics

Laplace transform: $\big(\mathcal I-\mathbb L(\lambda)\big)\hat{\mathbf q}(\lambda)=\hat{\mathbf q}_0(\lambda)$, with

$$
\mathbb L(\lambda)=\int_0^\infty e^{-\lambda u}\sum_k\mathbf g_k(u)\otimes\mathbf g_k(u)\,\mathrm du
=\frac1{2\pi}\int_{-\infty}^{\infty}\sum_k\mathbf G_k\!\big(\tfrac\lambda2+\mathrm i\omega\big)\otimes\mathbf G_k\!\big(\tfrac\lambda2-\mathrm i\omega\big)\,\mathrm d\omega .
$$

(Split $e^{-\lambda u}=e^{-\lambda u/2}e^{-\lambda u/2}$ and use Parseval.) This holds for $\operatorname{Re}\lambda/2>\max\operatorname{Re}(\text{roots of }D)$.

* **MS characteristic equation:** $\ \Phi(\lambda)=\det\big(\mathbf I_{p^2}-\mathbb L(\lambda)\big)=0$.
  The MS spectrum consists of these roots together with the pairwise sums $\lambda_i+\lambda_j$ of deterministic roots (from the free response).
* For complex $\lambda$, $\mathbb L(\lambda)$ is a **frequency convolution** of $\mathbf G$ with itself. A Nyquist/winding count of
  $\Phi(\mathrm i\nu)$ would need a full $\omega$-integral for every $\nu$, i.e. $O(M^2)$ work. This is the "higher order" problem.
* **But we do not need it.** Because the kernel is positive (Krein–Rutman / renewal-equation argument, as for the Lotka equation):
  1. For real $\sigma$, $r(\sigma):=\rho\big(\mathbb L(\sigma)\big)$ is itself an eigenvalue with a PSD eigenvector.
     It is continuous, strictly decreasing, and $r\to0$ as $\sigma\to\infty$.
  2. $\rho\big(\mathbb L(\sigma+\mathrm i\nu)\big)\le r(\sigma)$ (domination by the positive kernel).
  3. Hence **the rightmost root of $\Phi$ is real**: $\lambda^*$ with $r(\lambda^*)=1$.

  *(Proof sketch only. The domination step 2 for completely positive matrix kernels should be written out properly or
  cited, e.g. from positive-systems/renewal theory.)*

**MS Lyapunov exponent (decay rate of $\mathbb E\|\mathbf y\|^2$):**

$$
\lambda_{\text{MS}}=\max\big(2\max\operatorname{Re}\lambda_{\det},\ \lambda^*\big),\qquad r(\lambda^*)=1,
$$

$$
\mathbb L(\sigma)=\frac1{2\pi}\int\sum_k\overline{\mathbf G_k(\tfrac\sigma2+\mathrm i\omega)}\otimes\mathbf G_k(\tfrac\sigma2+\mathrm i\omega)\,\mathrm d\omega
\quad(\text{a positive matrix, }p^2\times p^2).
$$

The integrand is evaluated on the **shifted line** $\operatorname{Re}s=\sigma/2$, which is exactly what the package's `σ` argument already does.

### 8.5 MS stability limit

$$
\boxed{\ \text{MS stable}\iff \underbrace{D(\lambda)\ \text{has no roots with }\operatorname{Re}\lambda\ge0}_{\text{existing Nyquist count}}\ \ \text{and}\ \ r_0:=\rho\big(\mathbb L(0)\big)<1\ }
$$

* One evaluation, on the imaginary axis, with **no winding number needed** for the second moment.
* The MS boundary in parameter space is the level set $r_0(\mathbf p)=1$. It is a smooth scalar function, so MDBM can trace $\log r_0(\mathbf p)=0$
  just as it traces the deterministic boundary. The full MS-stable domain is the deterministic domain minus $\{r_0\ge1\}$.
* $r_0$ also gives a margin. $1/(1-r_0)$ is the amplification of the stationary variance (Sec. 8.6), i.e. the noise-induced
  "MS resonance", in addition to the deterministic stochastic resonance from roots approaching the axis.

### 8.6 Stationary second moment

For an MS-stable system:

$$
\big(\mathbf I_{p^2}-\mathbb L(0)\big)\operatorname{vec}\mathbf q=\operatorname{vec}\sum_k\tilde{\boldsymbol\gamma}_k\tilde{\boldsymbol\gamma}_k^{\mathsf T},
\qquad
\mathbf P=\frac1{2\pi}\int_{-\infty}^{\infty}\mathbf H(\mathrm i\omega)\,\mathbf q\,\mathbf H(\mathrm i\omega)^{*}\,\mathrm d\omega .
$$

By-products from the same sweep:
* PSD matrix of the response: $\mathbf S_{yy}(\omega)=\frac1{2\pi}\mathbf H\mathbf q\mathbf H^*$.
* Delayed covariance: $\mathbb E[\mathbf y(t)\mathbf y(t-\theta)^{\mathsf T}]=\frac1{2\pi}\int\mathbf H\mathbf q\mathbf H^*e^{\mathrm i\omega\theta}\mathrm d\omega$.
* Second moment of $\mathbf x$: $\mathbf P+\mathbf m^*\mathbf m^{*\mathsf T}$.

Usually only a few output entries are wanted (e.g. tool displacement). Then accumulate only
$\mathbf e_j^{\mathsf T}\mathbf H\mathbf q\mathbf H^*\mathbf e_j$ via $p\times p$ blocks, not the full $N\times N$ matrix.

### 8.7 Numerical verification (`check_exponent.py`, `check_stationary.py`)

| Test | Frequency domain | Exact reference |
|---|---|---|
| 1 DOF, 1 channel, $r_0=0.60$: MS exponent | −0.07949 | −0.07949 (eigenvalue of the moment ODE) |
| 1 DOF, 1 channel, $r_0=1.35$: MS exponent | +0.06856 | +0.06856 |
| 2 DOF, 2 channels, one $W$, $r_0=2.13$: MS exponent | 0.07999 | 0.08000 |
| 2 DOF, 2 channels, additive + multiplicative, $r_0=0.53$: full $\mathbf P$ | rel. error $2\cdot10^{-5}$ | Lyapunov equation (error = truncated $\omega^{-2}$ tail at $\omega_{\max}=3000$) |
| GSTM DDE (Sec. 4) | Var 0.372 | Monte Carlo 0.367 / 0.360 |

The delay-free tests are exact references for the algebra. The DDE is checked only against Monte Carlo so far.
The next step is to compare against stochastic semi-discretization (SSD), e.g. the GSTM MS boundary in dissertation Fig. 4.2 (σ = 1.0).

## 9. Computational cost vs. the deterministic unwrap sweep

Notation: $M$ = number of accepted $\omega$-nodes of the unwrap march (adaptive; set by the phase tolerance `tol`), $N$ = state
dimension, $p$ = noise channels, $m$ = Wiener processes.

| Task | What is evaluated per $\omega$ node | Extra per parameter point | Relative cost |
|---|---|---|---|
| Deterministic stability (current) | $D,\ D'$ (dual number). Closed form: $O(1)$; from a matrix model: LU of $\boldsymbol\Delta$, $O(N^3)$ | — | 1 |
| + stationary moment **and** MS stability ($r_0$) | the same LU, plus back-substitution $\boldsymbol\Delta^{-1}\mathbf E$: $O(N^2p)$; $\mathbf G_k$: $O(mp^2N)$; accumulate $\mathbb L(0)$: $O(mp^4)$; accumulate $\mathbf P$ (wanted entries) | dense $p^2\times p^2$ solve and spectral radius, $O(p^6)$ (negligible for $p\le3$) | **measured (turning, Sec. 10): 2.6–3× for ~10⁻³, 4–5× for ~10⁻⁴** (more nodes; same cost per node) |
| Same, but $D$ supplied in closed form (scalar models) | additional numerators $N_{ij}(\mathrm i\omega)$ (adjugate entries) in the same dual evaluation | — | per node ≈ 1× measured (the numerators share $e^{-\lambda\tau}$) |
| MS exponent / damping $\lambda_{\text{MS}}$ | secant/bisection on $\log r(\sigma)$, each step one sweep on $\operatorname{Re}s=\sigma/2$ | — | 5–10 sweeps (like the σ-shifted counts) |
| MS boundary chart | $r_0(\mathbf p)$ inside MDBM | — | same order as the deterministic MDBM chart |
| Naive MS Nyquist of $\Phi(\mathrm i\nu)$ | $\omega$-integral per $\nu$ | — | $O(M^2)$, so **avoided** by Sec. 8.4 |
| SSD (dissertation) | second-moment mapping of size $\sim(Nr)^2$, $r$ = delay resolution | eigenvalues/solve of a dense $(Nr)^2$ system | hours vs minutes for a 200×200 chart |

The deterministic nodes alone are **not** enough (corrected after the prototype, Sec. 10):
* **Peaks.** A root at distance $d$ from the axis turns the phase by ≈ π over a width ≈ $2d$, so the march does resolve
  each near-axis resonance. Between resonances, however, the phase test lets steps grow ×4 per step, up to $h\approx\omega$.
  A magnitude integral needs more than that: on the deterministic nodes alone the error was 1–16 %. The fix is a second
  acceptance test on the integral. The plain-trapezoid error $\tfrac{h^2}{12}(f'_a-f'_b)$ costs nothing, because $f'$ comes from the dual number
  already computed, and the accumulated value uses the Hermite-corrected trapezoid. This test needs a *global* scale: relative to the running integral it
  wastes most nodes near $\omega=0$, where $e^{-\mathrm i\omega\tau}-1\to0$. Hence the two-pass scheme in Sec. 10.
* **Tails.** $\|\mathbf G\|^2\sim c_\infty\,\omega^{-2r}$, where $r\ge1$ is the relative degree (r = 1 when a velocity modulates the noise of its own
  equation, r = 2 for displacement → force). Add the analytic tail $2c_\infty\,\omega_{\max}^{1-2r}/(2\pi(2r-1))$. For $r=1$ this
  correction is required: without it the error is $\sim1/\omega_{\max}$, which is the $2\cdot10^{-5}$ in the table above.
* **GPU.** Each lane accumulates $mp^4$ extra reals, i.e. one real for the turning model. The register pressure is small, and the kernel and
  memory pattern are unchanged.

**Bottom line:** for a stability chart that shows deterministic stability, MS stability **and** the stationary variance,
the measured cost is **~2.5–3× the deterministic unwrap chart at ~10⁻³ accuracy**, and 4–5× at ~10⁻⁴ (prototype, Sec. 10).
Parameter points that are deterministically unstable cost 1×, because pass 2 is skipped. The mean-square damping exponent costs a few more shifted sweeps.
Nothing requires a frequency-domain convolution as long as the noise is white and the system is time-invariant.

### Limits of this part
* Retarded DDEs only. For neutral DDEs $\mathbf G$ need not decay, $\int\|\mathbf G\|^2$ can diverge, and MS stability fails under any
  multiplicative noise in that channel. This needs separate treatment.
* Distributed delays: replace $e^{-\lambda\tau}$ by the kernel's Laplace transform. Nothing else changes.
* Multiple Wiener processes per channel with correlation $\mathbf R$: replace $\sum_k$ with weights $R_{kl}$, i.e. $\mathcal K[\mathbf q]=\sum_{kl}R_{kl}\mathbf g_k\mathbf q\mathbf g_l^{\mathsf T}$ (still positive).

## 10. Julia prototype on the unwrap march (turning GSTM)

Files (all in `notes/`, run from the repository root with `julia --project=. notes/<file>`):

| File | Content |
|---|---|
| `second_moment_prototype.jl` | `second_moment_unwrap` (the march of `src/unwrap_solver.jl` plus integral accumulation), `second_moment` (two-pass driver), `ms_exponent` (bisection on shifted lines) |
| `second_moment_turning_model.jl` | GSTM model `(D, 1, Kσ(e^{-λτ}-1))`, QuadGK reference |
| `second_moment_turning.jl` | comparison with the Python numbers, accuracy vs `rtol_I`, MS exponent |
| `second_moment_cost.jl` | wall-time and node-distribution study |

**Design.** The model returns `(D, n_1, …, n_J)` from one dual-number evaluation, so the numerators reuse $e^{-\lambda\tau}$.
The phase bookkeeping and step control are identical to `_unwrap_march`, and its internal helpers are reused. A step must also pass
the integral test `rtol_I`. The integrals use the Hermite-corrected trapezoid, plus a power-law tail fitted from $f,f'$ at $\omega_{\max}$.
Two passes:
1. The deterministic march (`rtol_I = Inf`) gives $Z$ and, on the same nodes, a coarse $I$. If $Z>0$, stop.
2. If $Z=0$, re-march with the integral test relative to the coarse $I$.

**Agreement** (rtol_I = 10⁻⁴, single pass):

| $K,\ \Omega,\ \sigma$ | $r_0$ march / QuadGK / Python | Var march / QuadGK / Python |
|---|---|---|
| 0.2, 1.0, 1.0 | 0.077077 / 0.077075 / 0.0771 | 0.103850 / 0.103849 / 0.10385 |
| 0.2, 0.6, 0.7 | 0.367984 / 0.367976 / 0.3680 | 0.371581 / 0.371578 / 0.37158 |
| 0.2, 0.6, 1.5 | 1.6897 / 1.6897 / 1.6897 | MS-unstable (all three) |

**Cost per parameter point** (1 thread; deterministic `calculate_unstable_roots_unwrap` = 34–52 evaluations, 5–7 µs, ≈145 ns/eval;
the second-moment march costs ≈135 ns/eval, i.e. **the same per node**):

| Scheme | rtol_I | evals / deterministic | time / deterministic | rel. error Var | rel. error $r_0$ |
|---|---|---|---|---|---|
| deterministic nodes only | ∞ | 1.0× | 0.9–1.0× | 1–16 % | 0.3–8 % |
| 1-pass (running scale) | 10⁻³ | 6–8× | 5.7–7.3× | 0.5–2·10⁻⁴ | 2–11·10⁻⁴ |
| 1-pass | 10⁻⁴ | 12–16× | 11–14× | ≤1.3·10⁻⁵ | ≤10⁻⁴ |
| **2-pass** | 10⁻³ | **2.8–3.2×** | **2.6–3.0×** | 1.5·10⁻⁵ – 4·10⁻⁴ | 2–10·10⁻⁴ |
| **2-pass** | 10⁻⁴ | 4.1–5.0× | 3.9–4.7× | 0.06–1.2·10⁻⁴ | 0.5–5·10⁻⁴ |

The node histograms (K=0.2, Ω=0.6) show where the nodes go. In 1-pass mode 60–70 % of all nodes sit in $\omega<0.5$, where $|G|^2\to0$.
In 2-pass mode they concentrate at the resonance $\omega\approx1$.

**MS exponent** (bisection, 40 marches on $\operatorname{Re}s=\sigma/2$):
* Delay-free exact references: −0.07949 / +0.06856, both reproduced.
* GSTM (K=0.2, Ω=0.6): $\lambda_{\text{MS}}=-0.00604$ at σ = 0.7 (MS stable, slow) and $+0.00687$ at σ = 1.5 (MS unstable).
  QuadGK confirms $r(\lambda_{\text{MS}})=1.0000$ in both cases.

### 10.1 Peak subtraction and band cap (tested, `second_moment_cost.jl`)

Pass 1 also tracks the 4 deepest $|D|$ minima, as `_unwrap_march` does. They are polished by Newton to roots $\lambda_r$. Pass 2 subtracts the pole pairs
$|{\rm res}_r|^2\big(|\mathrm i\omega-\lambda_r|^{-2}+|\mathrm i\omega-\bar\lambda_r|^{-2}\big)$, ${\rm res}_r=n(\lambda_r)/D_\lambda(\lambda_r)$, and adds their
integral over $[\omega_0,\omega_{\max}]$ back in closed form (arctan). This is exact algebra for any $\lambda_r$. The band cap is $h\le\Omega/4=\pi/(2\tau)$ for $\omega<5$.
Six points were tested, three of them approaching the lobe at Ω = 0.6 ($K_c=0.224735$; $K=0.9K_c,\ 0.99K_c,\ 0.999K_c$).

| Scheme | rtol_I | time vs deterministic, median (max) | max rel. error Var | max rel. error $r_0$ |
|---|---|---|---|---|
| 1-pass | 10⁻³ / 10⁻⁴ | 5.8× (6.1×) / 10.8× (11.9×) | 1.9·10⁻⁴ / 1.7·10⁻⁵ | 1.1·10⁻³ / 9.8·10⁻⁵ |
| **2-pass** | 10⁻³ / 10⁻⁴ | **3.1× (3.4×)** / **4.3× (5.1×)** | 4.3·10⁻⁴ / 1.2·10⁻⁴ | 1.0·10⁻³ / 4.6·10⁻⁴ |
| 2-pass + poles | 10⁻³ / 10⁻⁴ | 4.6× (7.5×) / 5.5× (9.5×) | 1.5·10⁻⁴ / 5.0·10⁻⁵ | 1.4·10⁻³ / 3.1·10⁻⁴ |
| 2-pass + poles + cap | 10⁻³ / 10⁻⁴ | 5.9× (8.6×) / 6.6× (9.8×) | 6.4·10⁻⁵ / 6.6·10⁻⁵ | 8.1·10⁻⁴ / 2.0·10⁻⁴ |

**Result: peak subtraction does not pay off here.** The node count drops by at most ~30 %, and only right at the boundary
($0.999K_c$: 248 → 180 at 10⁻⁴). Elsewhere it stays the same. The phase march already has to resolve every near-axis root to *count* it,
so the resonance peaks were never where the extra nodes went. The extra nodes are spread over the smooth parts of the line,
where the phase test allows long steps but the magnitude integral does not. The pole loop also makes each evaluation slower in this
prototype (a `Tuple` of runtime length, type-unstable). The band cap buys accuracy in $r_0$ at some points for 20–30 % more nodes.
**Default: plain 2-pass** (`subtract = false`; the option stays).

**Open points / next steps**
* Port to the GPU kernel: per lane the extra state is 2 accumulators and the tolerance test; MDBM on $\log r_0=0$ for the MS boundary.
* Independent check of the DDE case against SSD (dissertation Fig. 4.2, σ = 1).
* The error estimate is the plain-trapezoid error, but the Hermite rule is used, so it is conservative in places and
  optimistic in others (the 2-pass $r_0$ error reaches ~5× `rtol_I`). A per-length error budget is untested.
