#include <adept.h>
#include <benchmark/benchmark.h>
#include <ceres/autodiff_cost_function.h>
#include <game/vsr/cga_op.h>
#include <glog/logging.h>

using namespace vsr::cga;

adept::Stack g_stack;

double g_vector[3] = {1.0, 2.0, 3.0};
double g_bivector[3] = {1.0, 2.0, 3.0};
double g_motor[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double g_point[5] = {1.0, 2.0, 3.0, 1.0, 7.0};
double g_point_spin_motor[5];
double g_vec_ip_biv[3] = {0.0, 0.0, 0.0};

template <typename T>
void InnerProductVectorBivector(const T *vec, const T *biv, T *res) {
  Vector<T> vector(vec);
  Bivector<T> bivector(biv);
  Vector<T> result = vector <= bivector;
  for (int i = 0; i < 3; ++i)
    res[i] = result[i];
}

struct InnerProductVectorBivectorFunctor {
  template <typename T>
  bool operator()(const T *vec, const T *biv, T *res) const {
    InnerProductVectorBivector(vec, biv, res);
    return true;
  }
};

template <typename T>
void InnerProductVectorBivector2(const T *a, const T *b, T *res) {
  res[0] = -a[2] * b[1] - a[1] * b[0];
  res[1] = -a[2] * b[2] + a[0] * b[0];
  res[2] = a[1] * b[2] + a[0] * b[1];
}

struct InnerProductVectorBivectorFunctor2 {
  template <typename T>
  bool operator()(const T *vec, const T *biv, T *res) const {
    InnerProductVectorBivector2(vec, biv, res);
    return true;
  }
};

template <typename T> void MotorSpinPoint(const T *mot, const T *pnt, T *res) {
  Motor<T> motor(mot);
  Point<T> point(pnt);
  Point<T> result = point.spin(motor);
  for (int i = 0; i < 5; ++i)
    res[i] = result[i];
}

struct MotorSpinPointFunctor {
  template <typename T>
  bool operator()(const T *mot, const T *pnt, T *res) const {
    MotorSpinPoint(mot, pnt, res);
    return true;
  }
};

template <typename T> void RotorSpinPoint(const T *rot, const T *pnt, T *res) {
  Rotor<T> rotor(rot);
  Vector<T> point(pnt);
  Vector<T> result = point.spin(rotor);
  for (int i = 0; i < 3; ++i)
    res[i] = result[i];
}

struct RotorSpinPointFunctor {
  template <typename T>
  bool operator()(const T *rot, const T *pnt, T *res) const {
    RotorSpinPoint(rot, pnt, res);
    return true;
  }
};

static void BM_InnerProductVectorBivector(benchmark::State &state) {
  while (state.KeepRunning()) {
    InnerProductVectorBivector(g_vector, g_bivector, g_vec_ip_biv);
  }
}

static void BM_InnerProductVectorBivector2(benchmark::State &state) {
  while (state.KeepRunning()) {
    InnerProductVectorBivector2(g_vector, g_bivector, g_vec_ip_biv);
  }
}

static void BM_AdeptJacobianForward(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[9];
    adept::adouble vector[3];
    adept::set_values(vector, 3, g_vector);
    adept::adouble bivector[3];
    adept::set_values(bivector, 3, g_bivector);
    g_stack.new_recording();
    adept::adouble vec_ip_biv[3] = {0.0, 0.0, 0.0};
    InnerProductVectorBivector(vector, bivector, vec_ip_biv);

    g_stack.independent(vector, 3);
    g_stack.dependent(vec_ip_biv, 3);
    g_stack.jacobian_forward(jac);
  }
}

static void BM_AdeptJacobianForward2(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[9];
    adept::adouble vector[3];
    adept::set_values(vector, 3, g_vector);
    adept::adouble bivector[3];
    adept::set_values(bivector, 3, g_bivector);
    g_stack.new_recording();
    adept::adouble vec_ip_biv[3] = {0.0, 0.0, 0.0};
    InnerProductVectorBivector2(vector, bivector, vec_ip_biv);

    g_stack.independent(vector, 3);
    g_stack.dependent(vec_ip_biv, 3);
    g_stack.jacobian_forward(jac);
  }
}

static void BM_AdeptJacobianReverse(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[9];
    adept::adouble vector[3];
    adept::set_values(vector, 3, g_vector);
    adept::adouble bivector[3];
    adept::set_values(bivector, 3, g_bivector);
    g_stack.new_recording();
    adept::adouble vec_ip_biv[3] = {0.0, 0.0, 0.0};
    InnerProductVectorBivector(vector, bivector, vec_ip_biv);

    g_stack.independent(vector, 3);
    g_stack.dependent(vec_ip_biv, 3);
    g_stack.jacobian_reverse(jac);
  }
}

static void BM_AdeptJacobianReverse2(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[9];
    adept::adouble vector[3];
    adept::set_values(vector, 3, g_vector);
    adept::adouble bivector[3];
    adept::set_values(bivector, 3, g_bivector);
    g_stack.new_recording();
    adept::adouble vec_ip_biv[3] = {0.0, 0.0, 0.0};
    InnerProductVectorBivector2(vector, bivector, vec_ip_biv);

    g_stack.independent(vector, 3);
    g_stack.dependent(vec_ip_biv, 3);
    g_stack.jacobian_reverse(jac);
  }
}

static void BM_CeresJacobian(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[9];
    double vector[3] = {1.0, 2.0, 3.0};
    double bivector[3] = {1.0, 2.0, 3.0};
    double vec_ip_biv[3] = {0.0, 0.0, 0.0};
    const double *parameters[2] = {&vector[0], &bivector[0]};
    double *jacobians[2] = {jac, nullptr};

    ceres::AutoDiffCostFunction<InnerProductVectorBivectorFunctor, 3, 3, 3>(
        new InnerProductVectorBivectorFunctor())
        .Evaluate(parameters, &vec_ip_biv[0], jacobians);
  }
}

static void BM_CeresJacobian2(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[9];
    double vector[3] = {1.0, 2.0, 3.0};
    double bivector[3] = {1.0, 2.0, 3.0};
    double vec_ip_biv[3] = {0.0, 0.0, 0.0};
    const double *parameters[2] = {&vector[0], &bivector[0]};
    double *jacobians[2] = {jac, nullptr};

    ceres::AutoDiffCostFunction<InnerProductVectorBivectorFunctor2, 3, 3, 3>(
        new InnerProductVectorBivectorFunctor2())
        .Evaluate(parameters, &vec_ip_biv[0], jacobians);
  }
}

static void BM_MotorSpinPoint(benchmark::State &state) {
  while (state.KeepRunning()) {
    MotorSpinPoint(g_motor, g_point, g_point_spin_motor);
  }
}

static void BM_AdeptMotorSpinPointJacobianReverse(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[5 * 8];
    adept::adouble motor[8];
    adept::set_values(motor, 8, g_motor);
    adept::adouble point[5];
    adept::set_values(point, 5, g_point);
    g_stack.new_recording();
    adept::adouble res[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    MotorSpinPoint(motor, point, res);

    g_stack.independent(motor, 8);
    g_stack.dependent(res, 5);
    g_stack.jacobian_reverse(jac);
  }
}

static void BM_AdeptMotorSpinPointJacobianForward(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[5 * 8];
    adept::adouble motor[8];
    adept::set_values(motor, 8, g_motor);
    adept::adouble point[5];
    adept::set_values(point, 5, g_point);
    g_stack.new_recording();
    adept::adouble res[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    MotorSpinPoint(motor, point, res);

    g_stack.independent(motor, 8);
    g_stack.dependent(res, 5);
    g_stack.jacobian_forward(jac);
  }
}

static void BM_AdeptRotorSpinPointJacobianForward(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[3 * 4];
    adept::adouble motor[4];
    adept::set_values(motor, 4, g_motor);
    adept::adouble point[3];
    adept::set_values(point, 3, g_point);
    g_stack.new_recording();
    adept::adouble res[3] = {0.0, 0.0, 0.0};
    RotorSpinPoint(motor, point, res);

    g_stack.independent(motor, 4);
    g_stack.dependent(res, 3);
    g_stack.jacobian_reverse(jac, true);
  }
}

static void BM_CeresMotorSpinPointJacobian(benchmark::State &state) {
  while (state.KeepRunning()) {
    double jac[5 * 8];
    const double *parameters[2] = {&g_motor[0], &g_point[0]};
    double *jacobians[2] = {jac, nullptr};

    ceres::AutoDiffCostFunction<MotorSpinPointFunctor, 5, 8, 5>(
        new MotorSpinPointFunctor())
        .Evaluate(parameters, &g_point_spin_motor[0], jacobians);
  }
}

static void BM_CeresRotorSpinPointJacobian(benchmark::State &state) {
  double jac[3 * 4];
  const double *parameters[2] = {&g_motor[0], &g_point[0]};
  double *jacobians[2] = {jac, nullptr};
  while (state.KeepRunning()) {
    ceres::AutoDiffCostFunction<RotorSpinPointFunctor, 3, 4, 3>(
        new RotorSpinPointFunctor())
        .Evaluate(parameters, &g_point_spin_motor[0], jacobians);
  }
}

BENCHMARK(BM_InnerProductVectorBivector);
BENCHMARK(BM_AdeptJacobianForward);
BENCHMARK(BM_AdeptJacobianReverse);
BENCHMARK(BM_CeresJacobian);
BENCHMARK(BM_InnerProductVectorBivector2);
BENCHMARK(BM_AdeptJacobianForward2);
BENCHMARK(BM_AdeptJacobianReverse2);
BENCHMARK(BM_CeresJacobian2);
BENCHMARK(BM_MotorSpinPoint);
BENCHMARK(BM_AdeptMotorSpinPointJacobianForward);
BENCHMARK(BM_AdeptMotorSpinPointJacobianReverse);
BENCHMARK(BM_CeresMotorSpinPointJacobian);
BENCHMARK(BM_CeresRotorSpinPointJacobian);
BENCHMARK(BM_AdeptRotorSpinPointJacobianForward);

BENCHMARK_MAIN()
