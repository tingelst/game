batch jc () {

e1_ = red(e1), e2_ = green(e2), e3_ = blue(e3),
ctrl_range(th=-pi/3, -pi, pi);
ctrl_range(c=0, 0, 0.01);

dynamic{R: R = yellow(exp( e3^e1 * -th / 2 )),};

// g = no_weight(-9.81 ^ e3 ^ ni),
g = no_weight(-9.81 ^ e3),
p = c3ga_point(1.0, 0.5, 0.0);
dynamic{prot : prot = vp(R,p),};
dynamic{noprot : noprot = dm2(no ^ prot),};


c = 0.01; // [rad/Nm]
m = 1.0; // [kg]

L = (c3ga)(e3^e1),
dynamic{arm : arm = cyan((e3ga)project(prot, L)),}
dynamic{parm : parm = c3ga_point(arm),}

dynamic{Rg : Rg = no_weight(vp(R,g)),};
// dynamic{protg: protg = prot ^ g,};

// Wrong under
// dynamic{noniprotgno : noniprotgno = ((no ^ ni) . protg ^ no) ,};

//dynamic{qc : qc = c * m * noniprotgno; };

dynamic{R2 : R2 = exp(- c * m * arm ^ g),};
dynamic{prot2 : prot2 = vp(R R2,p),};
dynamic{noprot2 : noprot2 = no ^ prot2,};


}
