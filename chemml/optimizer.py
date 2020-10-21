#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize


def ensemble(optimizer, repeat, digest='argmin'):
    optima = [optimizer() for _ in range(repeat)]
    if digest == 'argmin':
        return optima[np.argmin([opt.fun for opt in optima])]
    else:
        return digest(optima)


def sequential_threshold(fun, x0, thresholds, bounds_lower, bounds_upper, **kwargs):
    
    active = np.ones_like(x0, dtype=np.bool_)
    
    while True:
        
        opt = minimize(
            fun,
            x0,
            bounds=np.vstack((
                bounds_lower,
                np.where(active, bounds_upper, bounds_lower)
            )).T,
            **kwargs,
        )
        
        active_next = active & (opt.x > thresholds)
        
        if np.count_nonzero(active_next) == np.count_nonzero(active):
            break
        else:
            active = active_next
            x0 = np.where(active, opt.x, bounds_lower)

    return opt


def l1_regularization(fun, x0, strengths, bounds_lower, bounds_upper, **kwargs):
    
    def ext_fun(x):
        exp_x = np.exp(x)
        val, jac = fun(x)
        return val + np.linalg.norm(strengths * exp_x, ord=1), jac + strengths * exp_x
        
    return minimize(
        ext_fun,
        x0,
        bounds=np.vstack((
            bounds_lower,
            bounds_upper
        )).T,
        **kwargs,
    )


def vanilla_lbfgs(fun, x0, bounds_lower, bounds_upper, **kwargs):
    
    return minimize(
        fun,
        x0,
        bounds=np.vstack((
            bounds_lower,
            bounds_upper
        )).T,
        **kwargs,
    )
