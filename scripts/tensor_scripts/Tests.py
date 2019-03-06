#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Tensor Codes"""
from unittest import TestCase

from createCubicElasticityMatrix import *
from array_utils import are_equal

class testTensorRotations(TestCase):
    def _gen_rand_az(self):
        return np.random.uniform(0, 2*np.pi)

    def _gen_rand_rot(self):
        return rotation_matrix(*[self._gen_rand_az() for i in np.arange(3)])

    def _gen_rand_mat(self, size=[]):
        return np.random.uniform(size=size)

    def test_right_hand_rule(self):
        """Ensure that rotate Counter Clockwise and sensible"""
        # Rotate Unit Vectors
        x_hat = np.array([1,0,0])
        y_hat = np.array([0,1,0])
        z_hat = np.array([0,0,1])

        # Rotate counter-clockwise az about z-axis
        az = self._gen_rand_az()
        caz = np.cos(az)
        saz = np.sin(az)

        # Calculate Rotation Matrix
        R = rotation_matrix(az)
        # check x_hat
        x_prime = transform_tensor(x_hat, R)
        x_true = np.array([caz, saz, 0])
        # Brute force Sum over indices
        x_brute = brute_transform_tensor(x_hat, R)
        assert are_equal([x_true, x_prime, x_brute]), "Z rotation of x_hat not correct"

        # check y_hat
        y_prime = transform_tensor(y_hat, R)
        y_true = np.array([-saz, caz, 0])
        # Brute force Sum over indices
        y_brute = brute_transform_tensor(y_hat, R)
        assert are_equal([y_true, y_prime, y_brute]), "Z rotation of y_hat not correct"

        # check y_hat
        z_prime = transform_tensor(z_hat, R)
        z_true = np.array([0, 0, 1])
        # Brute force Sum over indices
        z_brute = brute_transform_tensor(z_hat, R)
        assert are_equal([z_true, z_prime, z_brute]), "Z rotation of z_hat not correct"

    def test_vector_rotations(self):
        """Ensure that Code Rotates Vectors Correctly, i.e. rank 1 Tensors"""
        v = self._gen_rand_mat([3,])
        R = self._gen_rand_rot()
        # Simple Matrix Multiply
        u_true = np.matmul(R,v)
        # Outer Product
        u_prime = transform_tensor(v, R)
        # Brute force Sum over indices
        u_brute = brute_transform_tensor(v, R)
        assert are_equal([u_true, u_prime, u_brute]), "General rotation of random vec not correct"

    def test_matrix_rotations(self):
        """Ensure that Code Rotates Matrices Correctly i.e. rank 2 tensors"""
        M = self._gen_rand_mat([3,3])
        R = self._gen_rand_rot()
        # Simple Matrix Multiply
        M_true = np.matmul(R,np.matmul(M,R.T))
        # Outer Product
        M_prime = transform_tensor(M, R)
        # Brute force Sum over indices
        M_brute = brute_transform_tensor(M, R)
        assert are_equal([M_true, M_prime, M_brute]), "General rotation of random vec not correct"

    def test_outer_product(self):
        """Ensure that Code Rotates Matrices same as brute force sum i.e. rank 4 tensors"""
        C = self._gen_rand_mat([3,3,3,3])
        R = self._gen_rand_rot()
        C_brute = brute_transform_tensor(C, R)
        C_prime = transform_tensor(C, R)
        assert are_equal([C_prime, C_brute]), "not the same"
