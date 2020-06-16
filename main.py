import numpy as np
import scipy.linalg

Qa = 200    # m^3/h
Qb = 300
Qc = 150
Qd = 350

ca = 2      # mg/m^3
cb = 2

E12 = 25    # m^3/h
E23 = 50
E34 = 50
E35 = 25

Ws = 1500   # mg/h
Wg = 2500

'''
0 = Ws + Qa*ca + E12*(c2-c1) - Qa*c1
0 = Qb*cb + Qa*c1 + E12*(c1-c2) + E23*(c3-c2) - (Qa+Qb)*c2
0 = (Qa+Qb)*c2 + E23*(c2-c3) + E34*(c4-c3) + E35*(c5-c3) - Qc*c3 - Qd*c3
0 = Qc*c3 + E34*(c3-c4) - Qc*c4
0 = Wg + Qd*c3 + E35(c3-c5) - Qd*c5
'''
'''
- c1*(E12+Qa) + c2*E12 + c3*0 + c4*0 + c5*0 = - Ws - Qa*ca
c1*(Qa+E12) - c2*(E12+E23+Qa+Qb) + c3*E23 + c4*0 + c5*0 = - Qb*cb
c1*0 + c2*(E23+Qa+Qb) - c3*(E23+E34+E35+Qc+Qd) + c4*E34 + c5*E35 = 0
c1*0 + c2*0 + c3*(E34+Qc) - c4*(E34+Qc) + c5*0 = 0
c1*0 + c2*0 + c3*(E35+Qd) + c4*0 - c5*(E35+Qd) = - Wg
'''
A = [[+E12+Qa, -E12, 0, 0, 0],
     [-Qa-E12, +E12+E23+Qa+Qb, -E23, 0, 0],
     [0, -E23-Qa-Qb, +E23+E34+E35+Qc+Qd, -E34, -E35],
     [0, 0, -E34-Qc, +E34+Qc, 0],
     [0, 0, -E35-Qd, 0, +E35+Qd]]
B = [[+Ws+Qa*ca], [+Qb*cb], [0], [0], [+Wg]]
print('Matrix A:')
print(np.array(A))
print('Array B:')
print(np.array(B))

(P, L, U) = scipy.linalg.lu(A)
print('Matrix L:')
print(L)
print('Matrix U:')
print(U)
r = np.linalg.inv(L).dot(B)
x = np.linalg.inv(U).dot(r)

print('Solution (CO gas concentration array):')
print(x)

B2 = [[800+Qa*ca], [+Qb*cb], [0], [0], [1200]]
r2 = np.linalg.inv(L).dot(B2)
x2 = np.linalg.inv(U).dot(r2)
print('Modified CO gas concentration array:')
print(x2)

d1 = np.linalg.inv(L).dot([[1], [0], [0], [0], [0]])
d2 = np.linalg.inv(L).dot([[0], [1], [0], [0], [0]])
d3 = np.linalg.inv(L).dot([[0], [0], [1], [0], [0]])
d4 = np.linalg.inv(L).dot([[0], [0], [0], [1], [0]])
d5 = np.linalg.inv(L).dot([[0], [0], [0], [0], [1]])
Ainv1 = np.linalg.inv(U).dot(d1)
Ainv2 = np.linalg.inv(U).dot(d2)
Ainv3 = np.linalg.inv(U).dot(d3)
Ainv4 = np.linalg.inv(U).dot(d4)
Ainv5 = np.linalg.inv(U).dot(d5)
Ainv12 = np.concatenate((Ainv1, Ainv2), axis=1)
Ainv123 = np.concatenate((Ainv12, Ainv3), axis=1)
Ainv1234 = np.concatenate((Ainv123, Ainv4), axis=1)
Ainv = np.concatenate((Ainv1234, Ainv5), axis=1)
print('Inverse matrix A (LU decomposition):')
print(Ainv)

s = 100*Ainv[3, 0]*Ws/x[3]
g = 100*Ainv[3, 4]*Wg/x[3]
u = 100*(Qa*ca*Ainv[3,0] + Qb*cb*Ainv[3,1])/x[3]
print('Percentage CO in room for children (4) from:')
print('cigaretes = ' + str(s[0]) + '%')
print('barbecue grill = ' + str(g[0]) + '%')
print('street = ' + str(u[0]) + '%')
