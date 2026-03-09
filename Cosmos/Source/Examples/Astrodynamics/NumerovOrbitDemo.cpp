/**
 * @file NumerovOrbitDemo.cpp
 * @brief Astrodynamics demo: Numerov integration for orbital propagation
 * 
 * References:
 * - Curtis, "Orbital Mechanics for Engineering Students", §4.7 (J2 effects), §10 (Perturbations)
 * - Hintz, "Orbital Mechanics and Astrodynamics", Ch. 5
 * - Bate, "Fundamentals of Astrodynamics", §9.2-9.6
 * 
 * This demo uses Cowell's method formulation with Numerov integration
 * for second-order ODEs of the form r̈ = f(t, r).
 * 
 * Build: mkdir -p build && cd build && cmake .. && make NumerovOrbitDemo
 * Run: ./NumerovOrbitDemo
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>

// Physical constants (SI units)
constexpr double MU_EARTH = 3.986004418e14;      // m³/s²
constexpr double R_EARTH = 6.378137e6;           // m
constexpr double J2 = 1.08263e-3;                // Earth oblateness

// Simple 3D vector
struct Vector3D {
    double x, y, z;
    
    Vector3D(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
    
    Vector3D operator+(const Vector3D& o) const { return Vector3D(x+o.x, y+o.y, z+o.z); }
    Vector3D operator-(const Vector3D& o) const { return Vector3D(x-o.x, y-o.y, z-o.z); }
    Vector3D operator*(double s) const { return Vector3D(x*s, y*s, z*s); }
    Vector3D operator/(double s) const { return Vector3D(x/s, y/s, z/s); }
    
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
    double norm_sq() const { return x*x + y*y + z*z; }
    
    double dot(const Vector3D& o) const { return x*o.x + y*o.y + z*o.z; }
    Vector3D cross(const Vector3D& o) const {
        return Vector3D(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x);
    }
};

// Acceleration functions
Vector3D twoBodyAccel(const Vector3D& r) {
    double r_norm = r.norm();
    return r * (-MU_EARTH / (r_norm * r_norm * r_norm));
}

Vector3D j2Perturbation(const Vector3D& r) {
    double r_norm = r.norm();
    double r_sq = r_norm * r_norm;
    double z_sq_over_r_sq = (r.z * r.z) / r_sq;
    double factor = -1.5 * MU_EARTH * J2 * R_EARTH * R_EARTH / (r_sq * r_sq * r_norm);
    
    return Vector3D(
        factor * r.x * (1.0 - 5.0 * z_sq_over_r_sq),
        factor * r.y * (1.0 - 5.0 * z_sq_over_r_sq),
        factor * r.z * (3.0 - 5.0 * z_sq_over_r_sq)
    );
}

Vector3D totalAccel(const Vector3D& r, bool use_j2) {
    Vector3D a = twoBodyAccel(r);
    if (use_j2) a = a + j2Perturbation(r);
    return a;
}

// RK4 for starting Numerov
struct State {
    double t;
    Vector3D r, v;
};

State rk4Step(const State& s, double h, bool use_j2) {
    auto accel = [&](const Vector3D& r) { return totalAccel(r, use_j2); };
    
    Vector3D k1v = s.v;
    Vector3D k1a = accel(s.r);
    
    Vector3D k2v = s.v + k1a * (h/2);
    Vector3D k2a = accel(s.r + k1v * (h/2));
    
    Vector3D k3v = s.v + k2a * (h/2);
    Vector3D k3a = accel(s.r + k2v * (h/2));
    
    Vector3D k4v = s.v + k3a * h;
    Vector3D k4a = accel(s.r + k3v * h);
    
    return State{
        s.t + h,
        s.r + (k1v + k2v*2 + k3v*2 + k4v) * (h/6),
        s.v + (k1a + k2a*2 + k3a*2 + k4a) * (h/6)
    };
}

// Numerov integration
class NumerovOrbitPropagator {
public:
    double h;
    bool use_j2;
    std::vector<Vector3D> r_hist;
    std::vector<Vector3D> a_hist;
    std::vector<double> t_hist;
    
    NumerovOrbitPropagator(double step, bool j2) : h(step), use_j2(j2) {}
    
    void init(const State& s0, const State& s1) {
        r_hist.push_back(s0.r);
        r_hist.push_back(s1.r);
        a_hist.push_back(totalAccel(s0.r, use_j2));
        a_hist.push_back(totalAccel(s1.r, use_j2));
        t_hist.push_back(s0.t);
        t_hist.push_back(s1.t);
    }
    
    void step() {
        size_t n = r_hist.size() - 1;
        const Vector3D& r_n = r_hist[n];
        const Vector3D& r_nm1 = r_hist[n-1];
        const Vector3D& a_n = a_hist[n];
        const Vector3D& a_nm1 = a_hist[n-1];
        
        Vector3D r_pred = r_n * 2.0 - r_nm1 + a_n * (h * h);
        Vector3D a_pred = totalAccel(r_pred, use_j2);
        
        Vector3D r_new = r_n * 2.0 - r_nm1 + (a_pred + a_n * 10.0 + a_nm1) * (h * h / 12.0);
        Vector3D a_new = totalAccel(r_new, use_j2);
        
        r_hist.push_back(r_new);
        a_hist.push_back(a_new);
        t_hist.push_back(t_hist.back() + h);
    }
    
    State getState() const {
        size_t n = r_hist.size() - 1;
        Vector3D v = (r_hist[n] - r_hist[n-1]) / h;
        return State{t_hist[n], r_hist[n], v};
    }
};

// Calculate orbital energy
double specificEnergy(const Vector3D& r, const Vector3D& v) {
    return v.norm_sq()/2 - MU_EARTH/r.norm();
}

// Analytical J2 nodal regression (rad/s) - Curtis Eq. 4.52
double j2NodalRegression(double a, double e, double i) {
    double n = std::sqrt(MU_EARTH / (a*a*a));
    return n * -1.5 * J2 * std::pow(R_EARTH / (a * (1 - e*e)), 2) * std::cos(i);
}

// Analytical J2 perigee advance (rad/s) - Curtis Eq. 4.53
double j2PerigeeAdvance(double a, double e, double i) {
    double n = std::sqrt(MU_EARTH / (a*a*a));
    return n * 0.75 * J2 * std::pow(R_EARTH / (a * (1 - e*e)), 2) * (5*std::cos(i)*std::cos(i) - 1);
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Numerov Astrodynamics Demo\n";
    std::cout << "Cowell's method + Numerov integration\n";
    std::cout << "========================================\n\n";
    
    double altitude = 400e3;
    double r0 = R_EARTH + altitude;
    double v_circ = std::sqrt(MU_EARTH / r0);
    double inclination = 28.5 * M_PI / 180.0;
    
    State s0;
    s0.t = 0;
    s0.r = Vector3D(r0, 0, 0);
    s0.v = Vector3D(0, v_circ * std::cos(inclination), v_circ * std::sin(inclination));
    
    double period = 2 * M_PI * std::sqrt(r0*r0*r0 / MU_EARTH);
    
    std::cout << "Initial Conditions:\n";
    std::cout << "  Altitude: " << altitude/1000 << " km\n";
    std::cout << "  Inclination: " << inclination * 180/M_PI << " deg\n";
    std::cout << "  Circular velocity: " << v_circ << " m/s\n";
    std::cout << "  Orbital period: " << period/60 << " min\n\n";
    
    double omega_dot = j2NodalRegression(r0, 0, inclination);
    double argp_dot = j2PerigeeAdvance(r0, 0, inclination);
    std::cout << "Analytical J2 rates (Curtis Eq. 4.52-4.53):\n";
    std::cout << "  Nodal regression: " << omega_dot * 180/M_PI * 86400 << " deg/day\n";
    std::cout << "  Perigee advance:  " << argp_dot * 180/M_PI * 86400 << " deg/day\n\n";
    
    double dt = 60.0;
    int n_orbits = 10;
    int n_steps = static_cast<int>(n_orbits * period / dt);
    
    std::cout << "Simulation: " << n_orbits << " orbits, dt=" << dt << "s\n\n";
    
    for (bool use_j2 : {false, true}) {
        std::cout << "========================================\n";
        std::cout << (use_j2 ? "CASE 2: Two-body + J2" : "CASE 1: Pure Two-body") << "\n";
        std::cout << "========================================\n";
        
        NumerovOrbitPropagator prop(dt, use_j2);
        State s1 = rk4Step(s0, dt, use_j2);
        prop.init(s0, s1);
        
        double E0 = specificEnergy(s0.r, s0.v);
        std::cout << "Initial energy: " << std::scientific << E0 << " J/kg\n";
        
        std::string filename = use_j2 ? "numerov_j2.csv" : "numerov_twobody.csv";
        std::ofstream out(filename);
        out << std::scientific << std::setprecision(15);
        out << "t,orbit,r_x,r_y,r_z,energy,error\n";
        out << s0.t << ",0," << s0.r.x << "," << s0.r.y << "," << s0.r.z << ","
            << E0 << ",0\n";
                
        double max_err = 0;
        int last_orbit = 0;
        
        for (int i = 2; i <= n_steps; ++i) {
            prop.step();
            
            State current = prop.getState();
            double E = specificEnergy(current.r, current.v);
            double err = std::abs((E - E0) / E0);
            max_err = std::max(max_err, err);
            
            int orbit = static_cast<int>(current.t / period);
            
            if (i % 10 == 0) {
                out << current.t << "," << orbit << ","
                    << current.r.x << "," << current.r.y << "," << current.r.z << ","
                    << E << "," << err << "\n";
            }
            
            if (orbit != last_orbit) {
                std::cout << "  Orbit " << orbit << ": r=" << std::fixed << std::setprecision(2) 
                          << current.r.norm()/1000 << " km, err=" << std::scientific << err << "\n";
                last_orbit = orbit;
            }
        }
        
        out.close();
        std::cout << "\nResults:\n";
        std::cout << "  Max energy error: " << std::scientific << max_err << "\n";
        std::cout << "  Output: " << filename << "\n\n";
    }
    
    std::cout << "========================================\n";
    std::cout << "Demo Complete!\n";
    std::cout << "========================================\n";
    
    return 0;
}
