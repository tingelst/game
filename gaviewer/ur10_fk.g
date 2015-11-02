
function dh(th, d, a, alph) {
  return versor(tv(0,0,d)*versor(exp(-e1^e2 th/2))*tv(a,0,0)*versor(exp(-e2^e3 * alph/2)));
}

function rLog(V) {
	// rigid body motion logarithm,
	// Figure 13.5 in Geometric Algebra for Computer Science, pg 384
	// Created 20060322 LD
	// Modified 20070429 LD
	U = V;
	R = -no.(U ni); 	// rotation part
	t = -2 (no.U)/R;	// translation
	if (abs(-1-grade(R,0)) < 1e-6) { // no unique rotation plane rotation
		if (norm(t) < 1e-6) { // pick arbitrary plane for pi rotation
			return (-e1^e2 pi/2);
		} else {		// rotation plane dual to t
			I = t/I3; grade(R,2); sR2 = sqrt(- I I); I = I/sR2;
			return(grade( - t ni/2 - I pi, 2));
		}
	}
	if (abs(1-grade(R,0)) < 1e-6) { // no rotation
		return(-t ni/2);
	}
	I = grade(R,2); sR2 = sqrt(- I I); I = I/sR2; 	// rotation plane
	phih = -atan2(sR2,grade(R,0)); 	// half angle
	// w = (t^I)/I;			// chasles w
	// v = 1/(1-R R) (t.I)/I;	// chasles v
	return(grade( - ((t^I)/I) ni/2 + (1/(1-R R))(t.I) ni phih - I phih, 2));
}

batch reset_th() {
for (i=0;i<6;i=i+1) {
  th[i] = 0.0;
};
}

batch ur10_fk() {

e1_ = red(e1), e2_ = green(e2), e3_ = blue(e3),

for (i=0;i<6;i=i+1) {
  th[i] = 0.0;
  ctrl_range(th[i], -2 pi, 2 pi);
};
th[1] = -pi/2;
th[3] = -pi/2;


a[0]=0;          a[1]=-0.6127; a[2]=-0.5716; a[3]=0;       a[4]=0;        a[5]=0;
alph[0]=pi/2; alph[1]=0;    alph[2]=0;    alph[3]=pi/2; alph[4]=-pi/2; alph[5]=0;
d[0]=0.118;      d[1]=0;       d[2]=0;       d[3]=0.1639;  d[4]=0.1157;   d[5]=0.0922;

// base plane
plane = e3 + 0 ni,

// Joint motor
dynamic{M0 : M0 = dh(th[0],d[0],a[0],alph[0]);};
dynamic{M1 : M1 = dh(th[1],d[1],a[1],alph[1]);};
dynamic{M2 : M2 = dh(-th[2],d[2],a[2],alph[2]);};
dynamic{M3 : M3 = dh(th[3],d[3],a[3],alph[3]);};
dynamic{M4 : M4 = dh(th[4],d[4],a[4],alph[4]);};
dynamic{M5 : M5 = dh(th[5],d[5],a[5],alph[5]);};

// Motors from origin to each joint
dynamic{M01: M01 = M0 M1;};
dynamic{M02: M02 = M0 M1 M2;};
dynamic{M03: M03 = M0 M1 M2 M3;}
dynamic{M04: M04 = M0 M1 M2 M3 M4;};
dynamic{M05: M05 = M0 M1 M2 M3 M4 M5;}

cog0 = vp(M0, c3ga_point(0.021, 0.000, 0.027))
cog1 = vp(M01, c3ga_point( 0.38, 0.000, 0.158))
cog2 = vp(M02, c3ga_point( 0.24, 0.000, 0.068))

// Tangents at each joint
dynamic{t: t = yellow(weight(          no^e3)),},
dynamic{t00: t00 = yellow(weight(vp(M0,  no^e3))),},
dynamic{t01: t01 = yellow(weight(vp(M01, no^e3))),},
dynamic{t02: t02 = yellow(weight(vp(M02, no^e3))),},
dynamic{t03: t03 = yellow(weight(vp(M03, no^e3))),},
dynamic{t04: t04 = yellow(weight(vp(M04, no^e3))),},
dynamic{t05: t05 = yellow(weight(vp(M05, no^e3))),},

// Error motor
ctrl_range(xerr=0.0, -0.1, 0.1, 0.001);
ctrl_range(yerr=0.0, -0.1, 0.1, 0.001);
ctrl_range(zerr=0.0, -0.1, 0.1, 0.001);
ctrl_range(th12err=0.0, -0.1, 0.1, 0.001);
ctrl_range(th13err=0.0, -0.1, 0.1, 0.001);
ctrl_range(th23err=0.0, -0.1, 0.1, 0.001);
dynamic{Merr:  Merr = tv(xerr,yerr,zerr) * exp(-th12err e1^e2)* exp(-th13err e1^e3)* exp(-th23err e2^e3);};
dynamic{M1err: M1err = exp(vp(Merr,rLog(M1))); };
// dynamic{M01err: M01err = vp(Merr,M1); };
// dynamic{perr1: perr1 = green(vp(M0 M1err, no )),};
// dynamic{terr1: terr1 = green(vp(M0 M1err, no^rcont(e3 ni, no) )),};


// dynamic{L0 : L0 = normalize(rLog(M0)),};
// dynamic{L1 : L1 = normalize(rLog(M01)),};
// dynamic{L2 : L2 = normalize(rLog(M02)),};
// dynamic{L3 : L3 = normalize(rLog(M03)),};
// dynamic{L4 : L4 = normalize(rLog(M04)),};
// dynamic{L5 : L5 = normalize(rLog(M05)),};

dynamic{Merr05 : Merr05 = M0 M1err M2 M3 M4 M5;};

p00 = no,
dynamic{p0: p0 = vp(M0, no),};
dynamic{p1: p1 = vp(M01, no),};
dynamic{p2: p2 = vp(M02, no),};
dynamic{p3: p3 = vp(M03, no),};
dynamic{p4: p4 = vp(M04, no),};
dynamic{p5: p5 = vp(M05, no),};

dynamic{perr05: perr05 = black(vp(Merr05, no)),};
dynamic{xerr05: xerr05 = red(vp(Merr05, no^e1)),};
dynamic{yerr05: yerr05 = green(vp(Merr05, no^e2)),};
dynamic{zerr05: zerr05 = blue(vp(Merr05, no^e3)),};


}
