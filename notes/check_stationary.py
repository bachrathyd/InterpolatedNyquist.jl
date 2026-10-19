# Stationary second-moment MATRIX, n-DOF, additive + multiplicative (one W): exact Lyapunov vs frequency closure
import numpy as np
A = np.array([[0,0,1,0],[0,0,0,1],[-2,1,-0.1,0],[1,-1.5,0,-0.05]])
E = np.array([[0,0],[0,0],[1.,0],[0,1.]]); C = 0.5*np.array([[0.6,-0.3,0.2,0],[0.1,0.5,0,0.1]])
g = np.array([0.3, -0.2])                    # s = g + C y  (noise amplitudes in the 2 force channels)
N, p = 4, 2; I = np.eye(N)
# exact: 0 = A P + P A^T + E (g g^T + C P C^T) E^T
Lop = np.kron(I, A) + np.kron(A, I) + np.kron(E @ C, E @ C)
P_ex = np.linalg.solve(Lop, -(E @ np.outer(g, g) @ E.T).ravel()).reshape(N, N)
# frequency: q = g g^T + 1/2pi int G q G^*,  P = 1/2pi int H E q E^T H^*
w = np.linspace(-3000, 3000, 3_000_001); dw = w[1]-w[0]
L = np.zeros((p*p, p*p), complex); M = np.zeros((N*N, p*p), complex)
for k in range(0, len(w), 300_000):
    ww = w[k:k+300_000]
    HE = np.linalg.solve(1j*ww[:, None, None]*I - A, np.broadcast_to(E, (len(ww), N, p)))
    G = C @ HE
    L += np.einsum('wij,wkl->ikjl', G, G.conj()).reshape(p*p, p*p)*dw/(2*np.pi)
    M += np.einsum('wij,wkl->ikjl', HE, HE.conj()).reshape(N*N, p*p)*dw/(2*np.pi)
q = np.linalg.solve(np.eye(p*p) - L, np.outer(g, g).ravel())
P_fr = (M @ q).real.reshape(N, N)
print("rho(L) =", round(max(abs(np.linalg.eigvals(L))), 4))
print("max |P_freq - P_exact| / max|P_exact| =", np.abs(P_fr - P_ex).max()/np.abs(P_ex).max())
print("diag P exact:", np.round(np.diag(P_ex), 5)); print("diag P freq :", np.round(np.diag(P_fr), 5))
