#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Tensor Codes"""
from unittest import TestCase

from createCubicElasticityMatrix import *

# c11 = 1405.9  * 1.e9 # [Pa]
# c12 = 1364.8  * 1.e9 # [Pa]
# c44 =  397.9  * 1.e9 # [Pa]
# rho =   12.98 * 1.e3 # [kg/m^3]
#
# Pressure =  357.5 * 1.e9 # [Pa]
# Temperature = 6000 # [K]
#
# # Create Elasticity 4-Tensor
# C = createCubicElasticityMatrix(c11, c12, c44)
#
# # Display the non-zero components
# M = displayHookeLawMatrix(C)
#
# # Create rotation matrix
# R = rotation_matrix(np.pi/2)
#
# # Rotate Elasticity Tensor
# Cprime = transform_tensor(C, R)
#
# Mprime = displayHookeLawMatrix(Cprime)


def are_equal(x, y):
    ''' Check if x and y are equal to within machine percision.
        If an array is passed, will return False if a single
        element is not within machine percision.
    '''
    epsilon = np.finfo(float).eps
    if (np.abs(x-y) > np.maximum(np.abs(x), np.abs(y))*epsilon).any():
        return False
    else:
        return True

def gen_az():
    return np.random.uniform(0, 2*np.pi)

class testTensorRotations(TestCase):


    def test_right_hand_rule(self):
        """Ensure that rotate Counter Clockwise and sensible"""
        # Rotate Unit Vectors
        x_hat = np.array([1,0,0])
        y_hat = np.array([0,1,0])
        z_hat = np.array([0,0,1])

        # Rotate counter-clockwise az about z-axis
        az = gen_az()
        caz = np.cos(az)
        saz = np.sin(az)

        # Calculate Rotation Matrix
        Rz45 = rotation_matrix(az)
        # check x_hat
        x_prime = transform_tensor(x_hat, Rz45)
        x_true = np.array([caz, saz, 0])
        assert are_equal(x_prime, x_true), "Z rotation of x_hat not correct"

        # check y_hat
        y_prime = transform_tensor(y_hat, Rz45)
        y_true = np.array([-saz, caz, 0])
        assert are_equal(y_prime, y_true), "Z rotation of y_hat not correct"

        # check y_hat
        z_prime = transform_tensor(z_hat, Rz45)
        z_true = np.array([0, 0, 1])
        assert are_equal(z_prime, z_true), "Z rotation of z_hat not correct"

    def test_vector_rotations(self):
        """Ensure that Code Rotates Vectors Correctly, i.e. rank 1 Tensors"""
        u = np.random.uniform(size=[3,])
        R = rotation_matrix(gen_az(), gen_az(), gen_az())
        u_true = np.matmul(R,u)
        u_prime = transform_tensor(u, R)
        assert are_equal(u_prime, u_true), "General rotation of random vec not correct"
