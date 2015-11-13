batch lds1() {

    inner_product(lcont)

    e1_ = red(e1), e2_ = green(e2), e3_ = blue(e3),

    ctrl_range(th1=0.0, 0, pi/2);
    ctrl_range(th2=pi/4, 0, pi/2);
    ctrl_range(radius=1.0, 0.0, 2.0);


    dynamic{R : R = exp(-e1^e2 th1 / 2) exp(-e3^e1 th2 / 2); };

    t = no ^ -e3;
    dynamic{t_ : t_ = vp(R, t);};

    dynamic{s : s = no - 0.5 * radius * radius * ni;};

    tB = color(dm1(no_weight(no ^ e1 ^ e2)), 0.8, 0.8, 0.8, 0.1),
    plane = dual(tB^ni);

    dynamic{dL : dL = dual(t_ ^ ni),};

    dynamic{pp : pp = dual(s ^ dL);};

    
    dynamic{p1 : p1 = ( pp - sqrt(pp . pp) ) / ( ni . pp);};
    dynamic{p2 : p2 = ( pp + sqrt(pp . pp) ) / (ni . pp);};

    dynamic{Tt_ : Tt_ = cyan(vp(tv(p2), t_)),};
    dynamic{pl : pl = cyan(Tt_ / (ni . Tt_)),};

    // dynamic{xl : xl = dual(dL ^ plane),};

    dynamic{s2 : s2 = wireframe(alpha(black(plane . Tt_),0.5)),};




}
