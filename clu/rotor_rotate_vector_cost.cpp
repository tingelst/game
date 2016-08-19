void calculate(double a1, double a2, double a3, double b1, double b2, double b3, double r1, double r2, double r3, double r4, double c[8]) {

	c[1] = (((a1 * r4 * r4 + (2.0 * a3 * r2 - 2.0 * a2 * r3) * r4) - a1 * r3 * r3 + 2.0 * a3 * r1 * r3) - a1 * r2 * r2 + 2.0 * a2 * r1 * r2 + a1 * r1 * r1) - b1; // e1
	c[2] = (((-(a2 * r4 * r4)) + (2.0 * a3 * r1 - 2.0 * a1 * r3) * r4 + a2 * r3 * r3) - 2.0 * a3 * r2 * r3 - a2 * r2 * r2 - 2.0 * a1 * r1 * r2 + a2 * r1 * r1) - b2; // e2
	c[3] = (((-(a3 * r4 * r4)) + (2.0 * a1 * r2 - 2.0 * a2 * r1) * r4) - a3 * r3 * r3 + ((-(2.0 * a1 * r1)) - 2.0 * a2 * r2) * r3 + a3 * r2 * r2 + a3 * r1 * r1) - b3; // e3
}
