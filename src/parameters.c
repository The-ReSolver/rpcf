#include <string.h>
#include <stdlib.h>
#include <iniparser.h>
#include <math.h>

#include "dbg.h"
#include "parameters.h"

// the reload function only reloads certain parameters that don't break anything
void reloadParametersFromParamsFile(struct Parameters *params) {
    /*  Parse arguments from init file */

    dictionary *dict = iniparser_load("params");

    // parse parameters
    params->Re        = iniparser_getdouble(dict, "params:re", -1);
    if (params->Re == -1) {
        log_err("key Re was not found in params file");
        exit(1);
    }

    params->Ro     = iniparser_getdouble(dict, "params:ro", -1);
    if (params->Ro == -1) {
        log_err("key Ro was not found in params file");
        exit(1);
    }

    params->dt        = iniparser_getdouble(dict, "params:dt", -1);
    if (params->dt == -1) {
        log_err("key dt was not found in params file");
        exit(1);
    }

    params->T         = iniparser_getdouble(dict, "params:t", -1);
    if (params->T == -1) {
        log_err("key T was not found in params file");
        exit(1);
    }
    
    params->n_it_out  = iniparser_getint(dict, "params:n_it_out", -1);
    if (params->n_it_out == -1) {
        log_err("key n_it_out was not found in params file");
        exit(1);
    }

    params->n_it_print = iniparser_getint(dict, "params:n_it_print", params->n_it_out);

    // params->bctype = iniparser_getint(dict, "params:bctype", -1);
    // if (params->bctype == -1) {
    //  log_err("key bctype was not found in params file. 0 -> zero mass flux, 1 -> zero pg");
    //  exit(1);
    // }
    // params->A = iniparser_getdouble(dict, "params:A", -1);
    // if (params->A == -1) {
    //  log_err("key A was not found in params file");
    //  exit(1);
    // }
    // params->eta = iniparser_getdouble(dict, "params:eta", -1);
    // if (params->eta == -1) {
    //  log_err("key eta was not found in params file");
    //  exit(1);
    // }

    iniparser_freedict(dict);
}

struct Parameters *loadParametersFromParamsFile(void) {
    // initialize params struct
    struct Parameters *params = malloc(sizeof(struct Parameters));

    // load variables from params file
    reloadParametersFromParamsFile(params);

    // read data from params file
    dictionary *dict = iniparser_load("params");

    // parse invariant arguments
    params->Ny        = iniparser_getint(dict, "params:ny", -1);
    if (params->Ny == -1) {
        log_err("key Ny was not found in params file");
        exit(1);
    }

    params->Nz        = iniparser_getint(dict, "params:nz", -1);
    if (params->Nz == -1) {
        log_err("key Nz was not found in params file");
        exit(1);
    }

    params->alpha     = 4*asin(1.0)/iniparser_getdouble(dict, "params:L", -1);
    params->L         =             iniparser_getdouble(dict, "params:L", -1);
    if (params->alpha == -1) {
        log_err("key L was not found in params file");
        exit(1);
    }

    params->t_restart = iniparser_getdouble(dict, "params:t_restart", -1);
    if (params->t_restart == -1) {
        log_err("key t_restart was not found in params file");
        exit(1);
    }

    params->t_offset = iniparser_getdouble(dict, "params:t_offset", -1);
    if (params->t_offset == -1){
        log_err("key t_offset was not found in the params file");
        exit(1);
    }

    params->stretch_factor = iniparser_getdouble(dict, "params:stretch_factor", -1);
    if (params->stretch_factor == -1) {
        log_err("key stretch_factor was not found in params file");
        exit(1);
    }

    params->n_threads = iniparser_getint(dict, "params:n_threads", -1);
    if (params->n_threads == -1) {
        log_err("key n_threads was not found in params file"); 
        exit(1);
    }

    params->output_mode = iniparser_getint(dict, "params:output_mode", -1);
    if (params->output_mode == -1){
        params->output_mode = 1;
    }
    if (params->output_mode > 2 || params->output_mode < 1){
        log_err("invalid output mode in params file");
        exit(1);
    }

    // allocate space for grid along y
    double *x = malloc(sizeof(double)*params->Ny);

    // constant spacing along [0, 1]
    double step = 1.0/(params->Ny - 1);

    // create grid along y
    for (int i=0; i<params->Ny; i++) {
        x[i] = tanh(params->stretch_factor*(i*step-0.5)) / tanh(0.5*params->stretch_factor);
    }

    // allocate space for
    params->h = malloc(sizeof(double)*(params->Ny-1));
    for (int i=0; i<params->Ny-1; i++) {
        params->h[i] = x[i+1] - x[i];
    }

    iniparser_freedict(dict);
    free(x);
    return params;
}

void parametersDestroy(struct Parameters *params) {
    /*  Free up memory of parameters structure */
    free(params->h);
    free(params);
}