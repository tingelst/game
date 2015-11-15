batch cog() {

e1_ = red(e1), e2_ = green(e2), e3_ = blue(e3),

p1 = no,
p2 = c3ga_point(e1),
p3 = c3ga_point(e1 + e2),
p4 = c3ga_point(e2),

ctrl_range(m1=1, 0, 10);
ctrl_range(m2=2, 0, 10);
ctrl_range(m3=3, 0, 10);
ctrl_range(m4=4, 0, 10);


dynamic{m : m = m1 + m2 + m3 + m4;};

dynamic{cog_ : cog_ = (1 / m) * (m1 p1 + m2 p2 + m3 p3 + m4 p4),};



}
