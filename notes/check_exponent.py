# MS Lyapunov exponent: exact (eigenvalue of the second-moment ODE operator) vs
# frequency domain root of rho(L(sigma)) = 1 with L(sigma) = 1/2pi int G(sigma/2+iw) . G* dw
import numpy as np
w = np.linspace(-2000, 2000, 2_000_001); dw = w[1]-w[0]
def case(A, E, C, label):          # dy = A y dt + E (C y) dW, p = E.shape[1] channels, one W
    N = A.shape[0]; I = np.eye(N)
    # exact: dP/dt = A P + P A^T + E C P C^T E^T  -> operator on vec(P)
    Lop = np.kron(I, A) + np.kron(A, I) + np.kron(E @ C, E @ C)
    lam_exact = max(np.linalg.eigvals(Lop).real)
    def rho(s):
        Lm = 0
        for k in range(0, len(w), 200_000):
            ww = w[k:k+200_000]
            lam = s/2 + 1j*ww
            R = np.linalg.solve(lam[:, None, None]*I - A, np.broadcast_to(E, (len(ww),) + E.shape))
            G = C @ R                                   # p x p per frequency
            Lm = Lm + np.einsum('wij,wkl->ikjl', G, G.conj()).reshape(G.shape[1]**2, G.shape[1]**2)*dw/(2*np.pi)
        return max(abs(np.linalg.eigvals(Lm)))
    det_max = max(np.linalg.eigvals(A).real)
    lo, hi = 2*det_max + 1e-6, 5.0          # need sigma/2 > max Re(det. roots)
    if rho(lo) < 1: lam_freq = 2*det_max     # deterministic part dominates
    else:
        for _ in range(50):
            mid = (lo+hi)/2
            lo, hi = (mid, hi) if rho(mid) > 1 else (lo, mid)
        lam_freq = (lo+hi)/2
    print(f"{label}: rho(0)={rho(0.0):.4f}  exact MS exponent={lam_exact:.5f}  frequency root={lam_freq:.5f}")
z, K = 0.1, 0.5
A = np.array([[0, 1], [-(1+K), -2*z]])
case(A, np.array([[0], [1.0]]), K*1.2*np.array([[-1.0, 0]]), "1 channel, s=1.2 (rho<1)")
case(A, np.array([[0], [1.0]]), K*1.8*np.array([[-1.0, 0]]), "1 channel, s=1.8 (rho>1)")
A2 = np.array([[0,0,1,0],[0,0,0,1],[-2,1,-0.1,0],[1,-1.5,0,-0.05]])
E2 = np.array([[0,0],[0,0],[1.,0],[0,1.]]); C2 = np.array([[0.6,-0.3,0.2,0],[0.1,0.5,0,0.1]])
case(A2, E2, C2, "2-DOF, 2 channels, one W")
