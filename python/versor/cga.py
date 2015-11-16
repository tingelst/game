from libversor import *

def translator(vector):
    return Trs(1.0, -0.5 * vector[0], -0.5 * vector[1], -0.5 * vector[2])

def create_motor(direction, angle, translation):
    d = direction
    a = angle
    t = translation
    B23 =  d[0]
    B13 = -d[1]
    B12 =  d[2]
    cp = cos(angle/2.0)
    sp = sin(angle/2.0)
    t1 = t[0] / 2.0
    t2 = t[1] / 2.0
    t3 = t[2] / 2.0
    m = [cp,
         -B12 * sp,
         -B13 * sp,
         -B23 * sp,
         -t1 * cp + (-B12 * t2 - B13 * t3) * sp,
         -t2 * cp + ( B12 * t1 - B23 * t3) * sp,
         -t3 * cp + (-B13 * t1 + B23 * t2) * sp,
         (B12 * t3 - B13 * t2 + B23 * t1) * sp]
    return Mot(*m)