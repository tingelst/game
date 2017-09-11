import sys
sys.path.append('../build')
import versor as vsr
import numpy as np
np.set_printoptions(linewidth=120)
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def create_motor(d_lims=(0, 1), th_lims=(0, np.pi/2)):
    translator = (vsr.Vec(*np.random.random(3)).unit()
                  * np.random.uniform(*d_lims)).trs()
    rotator = vsr.Rot(vsr.Biv(*np.random.uniform(-1, 1, 3)).unit()
                      * np.random.uniform(*th_lims) * -0.5)
    motor = translator * rotator
    return motor


def create_points(motor, gaussian=False, radius=10, n_points=10, points_std=0.8, noise_std=0.09):
    points = []
    for i in range(n_points):
        if gaussian:
            a = vsr.Vec(*np.random.normal(0.0, points_std, 3)).null()
        else:
            a = (vsr.Vec(*np.random.uniform(-1, 1, 3)).unit()
                 * np.random.uniform(0, radius)).null()
        b = a.spin(motor)
        t = vsr.Vec(*np.random.random(3)).unit() * \
            np.random.normal(0.0, noise_std, 1)
        noise_motor = t.trs() * vsr.Rot(1, 0, 0, 0)
        bn = a.spin(noise_motor).spin(motor)
        points.append((a, b, bn))

    return points


def project(Y, M):
    YM2 = Y * M
    YM2[0] = 0.0
    YM2[26] = 0.0
    YM2[27] = 0.0
    YM2[28] = 0.0
    YM2[29] = 0.0
    YM2[30] = 0.0
    return vsr.CGA(vsr.MotRec(YM2 * M.rev()))

def CayleySelig(B):
    Rp = vsr.Mot(1.0, B[0], B[1], B[2], 0.0, 0.0, 0.0, 0.0)
    Rn = vsr.Mot(1.0, -B[0], -B[1], -B[2], 0.0, 0.0, 0.0, 0.0)
    Rninv = Rn.inv()
    eps = vsr.Mot(0,0,0,0,0,0,0,-1)
    b = vsr.Mot(0.0, B[5], -B[4], B[3], 0.0, 0.0, 0.0, 0.0)
    return Rp * Rninv + eps * Rninv * b * Rninv * 2

def main1():

    motor0 = create_motor()
    motor = create_motor()
    points = create_points(motor)

    A, B, _ = points[0]
    dM = vsr.CGA(A) * vsr.CGA(motor0).rev() * vsr.CGA(B) + \
        vsr.CGA(A).rev() * vsr.CGA(motor0).rev() * vsr.CGA(B).rev()

    # print(vsr.CGA(motor0))
    proj = project(dM, vsr.CGA(motor0))
    print(proj)

    dM2 =  vsr.CGA(motor0).rev() * vsr.CGA(A).spin(vsr.CGA(motor0)) * vsr.CGA(B) * 2.0
    print(project(dM2, vsr.CGA(motor0)))

def oexp(B):
    n = np.sqrt(1 + B[0] * B[0] + B[1] * B[1] + B[2] * B[2])
    s = B[0] * B[5] - B[1] * B[4] + B[2] * B[3]
    m = vsr.Mot(1.0, B[0], B[1], B[2], B[3], B[4], B[5], s) * (1.0 / n)
    return m

def CayleyLi(B):
    BB = B * B
    Rp = vsr.Mot(1.0, B[0], B[1], B[2], B[3], B[4], B[5], 0.0)
    R0 = vsr.Mot(1.0 - BB[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    R4 = vsr.Mot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, BB[7])
    Rn = R0 + R4
    Rden = R0 * R0 
    return Rp * Rp * Rn * Rden.inv()

def retr(B, M):
    # print('retr')
    return (B * M + M).retract()



def test(a,b):
    g = (vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm(vsr.CGA(a)) * vsr.CGA(b) * 2.0)[0]
    h = (a[1]* b[0] - a[0] * b[1]) * 2
    print(h)
    print(g)
    i =  (-a[0] * b[0] - a[1] * b[1] ) * 4.0
    j = (vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm(vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm(vsr.CGA(a))) * vsr.CGA(b) * 4.0)[0]
    print(i)
    print(j)

def update(points, mot):

    g = np.zeros(6)
    H = np.zeros((6,6))

    def err(points, mot):
        err = 0.0
        for A, B, _ in points:
            err += (vsr.CGA(A.spin(mot)) * vsr.CGA(B) * -2.0)[0]
        return err

    g0 = 0
    for A, B, _ in points:
        MAM = A.spin(mot)
        # Looks like skew
        g[0] += (MAM[1] * B[0] - MAM[0] * B[1]) * 2
        g[1] += (MAM[2] * B[0] - MAM[0] * B[2]) * 2
        g[2] += (MAM[2] * B[1] - MAM[1] * B[2]) * 2
        g[3] += (MAM[0] - B[0]) * 2
        g[4] += (MAM[1] - B[1]) * 2
        g[5] += (MAM[2] - B[2]) * 2

        # g[0] += (vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm(vsr.CGA(A.spin(mot))) * vsr.CGA(B) * 2.0)[0]
        # g[1] += (vsr.CGA(vsr.Mot(0,0,1,0,0,0,0,0)).comm(vsr.CGA(A.spin(mot))) * vsr.CGA(B) * 2.0)[0]
        # g[2] += (vsr.CGA(vsr.Mot(0,0,0,1,0,0,0,0)).comm(vsr.CGA(A.spin(mot))) * vsr.CGA(B) * 2.0)[0]
        # g[3] += (vsr.CGA(vsr.Mot(0,0,0,0,1,0,0,0)).comm(vsr.CGA(A.spin(mot))) * vsr.CGA(B) * 2.0)[0]
        # g[4] += (vsr.CGA(vsr.Mot(0,0,0,0,0,1,0,0)).comm(vsr.CGA(A.spin(mot))) * vsr.CGA(B) * 2.0)[0]
        # g[5] += (vsr.CGA(vsr.Mot(0,0,0,0,0,0,1,0)).comm(vsr.CGA(A.spin(mot))) * vsr.CGA(B) * 2.0)[0]

        H[0,0] += (MAM[0] * B[0] + MAM[1] * B[1] ) * -4.0
        H[0,1] += MAM[1] * B[2] * -4
        H[0,2] += MAM[0] * B[2] * 4
        H[0,3] += MAM[1] * 4
        H[0,4] += MAM[0] * -4
        # H[0,1] += (vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,1,0,0,0,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[0,2] += (vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,1,0,0,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[0,3] += (vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,1,0,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[0,4] += (vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,0,1,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[0,5] += (vsr.CGA(vsr.Mot(0,1,0,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,0,0,1,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        H[1,0] = H[0,1]
        H[2,0] = H[0,2]
        H[3,0] = H[0,3]
        H[4,0] = H[0,4]
        # H[5,0] = H[0,5]

        H[1,1] += (MAM[0] * B[0] + MAM[2] * B[2] ) * -4.0
        H[1,2] += MAM[0] * B[1] * -4.0
        H[1,3] += MAM[2] * 4.0
        # H[1,4] += 0.0
        H[1,5] += MAM[0] * -4.0
        # H[1,2] += (vsr.CGA(vsr.Mot(0,0,1,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,1,0,0,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[1,3] += (vsr.CGA(vsr.Mot(0,0,1,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,1,0,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[1,4] += (vsr.CGA(vsr.Mot(0,0,1,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,0,1,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[1,5] += (vsr.CGA(vsr.Mot(0,0,1,0,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,0,0,1,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        H[2,1] = H[1,2]
        H[3,1] = H[1,3]
        H[5,1] = H[1,5]
        
        H[2,2] += (MAM[1] * B[1] + MAM[2] * B[2] ) * -4.0
        # H[2,3] += (vsr.CGA(vsr.Mot(0,0,0,1,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,1,0,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        H[2,4] += MAM[2] * 4.0
        H[2,5] += MAM[1] * -4.0
        # H[2,4] += (vsr.CGA(vsr.Mot(0,0,0,1,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,0,1,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[2,5] += (vsr.CGA(vsr.Mot(0,0,0,1,0,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,0,0,1,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[3,2] = H[2,3]
        H[4,2] = H[2,4]
        H[5,2] = H[2,5]

        # H[3,4] += (vsr.CGA(vsr.Mot(0,0,0,0,1,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,0,1,0,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]
        # H[3,5] += (vsr.CGA(vsr.Mot(0,0,0,0,1,0,0,0)).comm((vsr.CGA(vsr.Mot(0,0,0,0,0,0,1,0)).comm(vsr.CGA(A.spin(mot))))) * vsr.CGA(B) * 4.0)[0]

        H[3,3] += -4.0
        H[4,4] += -4.0
        H[5,5] += -4.0

    # print(H)





    B = np.dot(np.linalg.pinv(H), -g)

    # B = g

    # line search
    alpha = 1.0
    beta = 0.01
    err0 = err(points, mot)

    # while err(points, retr(vsr.Dll(*B) * alpha, mot)) > err0 + alpha * beta * np.inner(g, B) :
    # while err(points, CayleyLi(vsr.Dll(*B) * alpha) * mot) > err0 + alpha * beta * np.inner(g, B) :
    # while err(points, CayleySelig(vsr.Dll(*B) * alpha) * mot) > err0 + alpha * beta * np.inner(g, B) :
    while err(points, oexp(alpha * B) * mot) > err0 + alpha * beta * np.inner(g, B) :
    # while err(points, vsr.Dll(*B * alpha).exp() * mot) > err0 + alpha * beta * np.inner(g, B) :
    # while err(points, vsr.Dll(*B * alpha).exp() * mot) > err0 :
        alpha *= 0.5
    # print(alpha)




    # B /= np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)

    # mot = vsr.Dll(*B*alpha).exp() * mot

    # mot = CayleyLi(vsr.Dll(*B) * alpha) * mot
    # mot = CayleySelig(vsr.Dll(*B) * alpha) * mot
    mot = oexp(B * alpha) * mot

    # mot = retr(vsr.Dll(*B) * alpha, mot)

    return mot, err(points, mot), np.linalg.norm(g)


def motor_rotate_point(motor, point):
    m1, m2, m3, m4, m5, m6, m7, m8 = motor
    p1, p2, p3, p5, p4 = point
    q = np.zeros(3)

    R = np.zeros((5,5))
    R[0,0] = m4 * m4 - m3 * m3 - m2 * m2 + m1 * m1
    R[1,0] = -2.0 * m3 * m4 - 2.0 * m1 * m2
    R[2,0] = 2.0 * m2 * m4 - 2.0 * m1 * m3
    R[3,0] = -2.0 * m4 * m8 + 2.0 * m3 * m7 + 2.0 * m2 * m6 - 2.0 * m1 * m5

    R[0,1] = 2.0 * m1 * m2 - 2.0 * m3 * m4
    R[1,1] = - m4 * m4 + m3 * m3 - m2 * m2 + m1 * m1
    R[2,1] = -2.0 * m1 * m4 - 2.0 * m2 * m3
    R[3,1] = 2.0 * m3 * m8 + 2.0 * m4 * m7 - 2.0 * m1 * m6 - 2.0 * m2 * m5

    R[0,2] = 2.0 * m2 * m4 + 2.0 * m1 * m3
    R[1,2] = 2.0 * m1 * m4 - 2.0 * m2 * m3
    R[2,2] = m4 * m4 - m3 * m3 + m2 * m2 + m1 * m1
    R[3,2] = -2.0 * m2 * m8 - 2.0 * m1 * m7 - 2.0 * m4 * m6 - 2.0 * m3 * m5

    R[3,3] = m4 * m4 + m3 * m3 + m2 * m2 + m1 * m1
 
    R[0,4] = -2.0 * m4 * m8 - 2.0 * m3 * m7 - 2.0 * m2 * m6 - 2.0 * m1 * m5
    R[1,4] = 2.0 * m3 * m8 - 2.0 * m4 * m7 - 2.0 * m1 * m6 + 2.0 * m2 * m5
    R[2,4] = -2.0 * m2 * m8 - 2.0 * m1 * m7 + 2.0 * m4 * m6 + 2.0 * m3 * m5
    R[3,4] = 2.0 * m8 * m8 + 2.0 * m7 * m7 + 2.0 * m6 * m6 + 2.0 * m5 * m5
    R[4,4] = m4 * m4 + m3 * m3 + m2 * m2 + m1 * m1

    # print(R[:3,:3])

    return np.dot(R, point.reshape(5,1)).flatten()


def spin(m, a):
    m1, m2, m3, m4, m5, m6, m7, m8 = m
    a1, a2, a3, a5, a4 = a
    residual = np.zeros(5)
    residual[0] = ((((-(2.0 * a4 * m4 * m8)) - 2.0 * a4 * m3 * m7 - 2.0 * a4 * m2 * m6 - 2.0 * a4 * m1 * m5 + a1 * m4 * m4 + (2.0 * a3 * m2 - 2.0 * a2 * m3) * m4) - a1 * m3 * m3 + 2.0 * a3 * m1 * m3) - a1 * m2 * m2 + 2.0 * a2 * m1 * m2 + a1 * m1 * m1)
    residual[1] = (((2.0 * a4 * m3 * m8 - 2.0 * a4 * m4 * m7 - 2.0 * a4 * m1 * m6 + 2.0 * a4 * m2 * m5) - a2 * m4 * m4 + (2.0 * a3 * m1 - 2.0 * a1 * m3) * m4 + a2 * m3 * m3) - 2.0 * a3 * m2 * m3 - a2 * m2 * m2 - 2.0 * a1 * m1 * m2 + a2 * m1 * m1)
    residual[2] = ((((-(2.0 * a4 * m2 * m8)) - 2.0 * a4 * m1 * m7 +
                     2.0 * a4 * m4 * m6 + 2.0 * a4 * m3 * m5) -
                    a3 * m4 * m4 + (2.0 * a1 * m2 - 2.0 * a2 * m1) * m4) -
                   a3 * m3 * m3 + ((-(2.0 * a1 * m1)) - 2.0 * a2 * m2) * m3 +
                   a3 * m2 * m2 + a3 * m1 * m1)

    return residual
                    

def test_motor_rotate_point():
    motor = create_motor()
    # print(np.array(motor))
    motor = vsr.Vec(4,1,2).trs() * vsr.Rot(vsr.Biv(1,1,1).unit() * (np.pi/6))
    point = vsr.Vec(1,2,3).null()
    print(np.array(point.spin(motor)))
    # print(motor.matrix()[:3,:3])
    print(motor_rotate_point(np.array(motor).copy(), np.array(point).copy()))

    # print(spin(np.array(motor), np.array(point)))

def main():
    m = create_motor()
    m = vsr.Mot(1,0,0,0,0,0,0,0)
    print(m)
    motor = create_motor()
    print(motor)
    points = create_points(motor)
    errs = []
    for i in range(100):
        m, err, gnorm = update(points, m)
        if gnorm < 1e-3:
            break
        if err < 1e-6:
            break
        errs.append(err)
        print(i)
    print(m)
    # print(m.rev() * motor)
    plt.semilogy(errs)
    plt.show()


if __name__ == '__main__':
    # main()
    test_motor_rotate_point()
    # test(vsr.Vec(1,2,3).null(), vsr.Vec(4,5,6).null())
