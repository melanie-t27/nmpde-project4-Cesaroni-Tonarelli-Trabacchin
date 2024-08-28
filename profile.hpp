#ifndef PROFILE_H
#define PROFILE_H

struct ProfileData {
    double mesh_init;
    double fe_setup;
    double ode_solve;
    double interpolation;
    double assemble_rhs;
    double fe_solve_tot;
    double fe_precond_init;
    double fe_linear_solve_time;
    double avg_lin_iters;
    double comm_time;
    int N_iters;
};

ProfileData profileData;

#endif