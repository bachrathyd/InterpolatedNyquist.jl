import numpy as np, sys, time

zeta = 0.1

def freq(K, Om, sig):
    tau = 2*np.pi/Om
    w = np.linspace(-400, 400, 4_000_001)
    Hr = 1.0/(-w**2 + 2j*zeta*w + 1 + K*(1 - np.exp(-1j*w*tau)))
    Gz = K*sig*(np.exp(-1j*w*tau) - 1)*Hr
    dw = w[1]-w[0]
    h2 = np.sum(np.abs(Hr)**2)*dw/(2*np.pi)
    rho = np.sum(np.abs(Gz)**2)*dw/(2*np.pi)
    var_add = (K*sig)**2*h2
    return var_add, rho, var_add/(1-rho)

def mc(K, Om, sig, ntraj=4000, dt=0.002, T=1200.0, seed=1):
    rng = np.random.default_rng(seed)
    tau = 2*np.pi/Om
    nd = int(round(tau/dt))
    n = int(T/dt)
    hist = np.zeros((nd, ntraj))           # ring buffer of y
    y = np.zeros(ntraj); v = np.zeros(ntraj)
    acc = 0.0; cnt = 0; burn = n//3
    sq = np.sqrt(dt)
    for k in range(n):
        yd = hist[k % nd]            # y(t - tau)
        dW = rng.standard_normal(ntraj)*sq
        a = -2*zeta*v - y + K*(yd - y)
        v = v + a*dt + K*sig*(1 + yd - y)*dW   # symplectic Euler (v first)
        y = y + v*dt
        hist[k % nd] = y
        if k >= burn and k % 10 == 0:
            acc += np.mean(y**2); cnt += 1
    return acc/cnt

for (K, Om, sig) in [(0.2, 1.0, 1.0), (0.2, 0.6, 0.7), (0.2, 0.6, 1.5), (0.2, 1.0, 0.01)]:
    va, rho, vm = freq(K, Om, sig)
    print(f"K={K} Om={Om} sig={sig}: additive-only Var={va:.5g}  rho={rho:.4f}  "
          f"multiplicative Var={vm:.5g}", flush=True)
    if sig > 0.1:
        t = time.time(); m = mc(K, Om, sig)
        print(f"   Monte Carlo E[y^2] = {m:.5g}  ({time.time()-t:.0f}s)", flush=True)
