batch jc () {

e1_ = red(e1), e2_ = green(e2), e3_ = blue(e3),
ctrl_range(th=-pi/3, -pi, pi);

dynamic{R: R = yellow(exp( e3^e1 * -th / 2 )),};

g = -9.81 ^ e3 ^ ni,
p = c3ga_point(1.0, 0.0, 0.0);
dynamic{prot : prot = vp(R,p),};
dynamic{noprot : noprot = dm2(no ^ prot),};


c = 0.01; // [rad/Nm]
m = 1.0; // [kg]

dynamic{Rg : Rg = vp(R,g),};
dynamic{protg: protg = prot ^ g,};
dynamic{noniprotgno : noniprotgno = (no ^ ni) . protg ^ no ,};

dynamic{qc : qc = c * m * noniprotgno; };

dynamic{R2 : R2 = exp(qc),};
dynamic{prot2 : prot2 = vp(R2,p),};
dynamic{noprot2 : noprot2 = no ^ prot2,};


}
