#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Tensor Codes"""
import unittest

#TODO: define __all__ for elasticity_matrix
from elasticity_matrix import *
from array_utils import are_equal
import numpy as np

N_iters = 10
N_eps   = 10
tol = N_eps * np.finfo(float).eps

class testTensorRotations(unittest.TestCase):
    def _gen_rand_az(self):
        return gen_rand_az()

    def _gen_rand_rot(self):
        return gen_rand_rot()

    def _gen_rand_mat(self, size=[]):
        return gen_rand_mat(size)

    def _gen_rand_sym_mat(self, size=[]):
        return gen_rand_sym_mat(size)

    def _are_equal(self, X):
        return are_equal(X, tol=tol)

    def test_right_hand_rule(self):
        """Ensure that rotate Counter Clockwise and sensible"""
        # Rotate Unit Vectors
        x_hat = np.array([1,0,0])
        y_hat = np.array([0,1,0])
        z_hat = np.array([0,0,1])

        for i in np.arange(N_iters):
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
            assert self._are_equal([x_true, x_prime, x_brute]), "Z rotation of x_hat not correct"

            # check y_hat
            y_prime = transform_tensor(y_hat, R)
            y_true = np.array([-saz, caz, 0])
            # Brute force Sum over indices
            y_brute = brute_transform_tensor(y_hat, R)
            assert self._are_equal([y_true, y_prime, y_brute]), "Z rotation of y_hat not correct"

            # check y_hat
            z_prime = transform_tensor(z_hat, R)
            z_true = np.array([0, 0, 1])
            # Brute force Sum over indices
            z_brute = brute_transform_tensor(z_hat, R)
            assert self._are_equal([z_true, z_prime, z_brute]), "Z rotation of z_hat not correct"

    def test_vector_rotations(self):
        """Ensure that Code Rotates Vectors Correctly, i.e. rank 1 Tensors"""
        for i in np.arange(N_iters):
            v = self._gen_rand_mat([3,])
            R = self._gen_rand_rot()
            # Simple Matrix Multiply
            u_true = np.matmul(R,v)
            # Outer Product
            u_prime = transform_tensor(v, R)
            # Brute force Sum over indices
            u_brute = brute_transform_tensor(v, R)
            assert self._are_equal([u_true, u_prime, u_brute]), "General rotation of random vec not correct"

    def test_matrix_rotations(self):
        """Ensure that Code Rotates Matrices Correctly i.e. rank 2 tensors"""
        for i in np.arange(N_iters):
            M = self._gen_rand_mat([3,3])
            R = self._gen_rand_rot()
            # Simple Matrix Multiply
            M_true = np.matmul(R,np.matmul(M,R.T))
            # Outer Product
            M_prime = transform_tensor(M, R)
            # Brute force Sum over indices
            M_brute = brute_transform_tensor(M, R)
            assert self._are_equal([M_true, M_prime, M_brute]), "General rotation of random matrix not correct"

    def test_outer_product(self):
        """Ensure that Code Rotates Matrices same as brute force sum i.e. rank 4 tensors"""
        for i in np.arange(N_iters):
            C = self._gen_rand_mat([3,3,3,3])
            R = self._gen_rand_rot()
            C_brute = brute_transform_tensor(C, R)
            C_prime = transform_tensor(C, R)
            assert self._are_equal([C_prime, C_brute]), \
            "Outer product transfrom and brute force sum transform are not the same"

    def test_stress_strain_rotations(self):
        for i in np.arange(N_iters):
            C = self._gen_rand_mat([3,3,3,3])
            E = np.random.uniform(size=[3,3]); E = 0.5*(E + E.T)
            S = np.tensordot(C, E, ((2,3), (0,1)))
            R = self._gen_rand_rot()
            Cprime = transform_tensor(C, R)
            Eprime = transform_tensor(E, R)
            Sprime = transform_tensor(np.tensordot(Cprime, Eprime, ((2,3), (0,1))), R.T)
            assert self._are_equal([S, Sprime]), "Stress is not the same in different coordinate system!"

    def test_hooke_law_fomulation(self):
        for i in np.arange(N_iters):
            V = self._gen_rand_sym_mat([6,6])
            C = create_Cijkl_from_hooke_law_matrix(V)
            Vprime = get_hooke_law_matrix(C)
            assert self._are_equal([V, Vprime]), "Error in mapping from 4th order tensor to 6x6 matrix"

    def test_isotropic_tensor(self):
        for i in np.arange(N_iters):
            iso_params = self._gen_rand_mat([2,])
            C_iso = create_isotropic_elasticity_tensor(*iso_params)
            R = self._gen_rand_rot()
            Cprime_iso = transform_tensor(C_iso, R)
            assert self._are_equal([C_iso, Cprime_iso]), "Isotropic tensor changed after rotations!"

    def test_cubic_symmetries(self):
        for i in np.arange(N_iters):
            cubic_params = self._gen_rand_mat([3,])
            C = create_cubic_elasticity_tensor(*cubic_params)
            assert self._are_equal([C, brute_create_cubic_elasticity_tensor(*cubic_params)]), "Cubic Tensor not generated Correctly"
            rot_params = [np.random.choice([0,1/2,1,3/2,2])*np.pi for i in range(3)]
            R = rotation_matrix(*rot_params)
            Cprime = transform_tensor(C, R)
            assert self._are_equal([C, Cprime]), "Cubic symmetry not preserved!"

    def test_transverseley_isotropic_symmetry(self):
        for i in np.arange(N_iters):
            tran_iso_params = self._gen_rand_mat([5,])
            A = C = tran_iso_params[0]
            L = N = mu = tran_iso_params[1]
            F = lam = A - 2*N
            C_iso = create_transversely_isotropic_tensor(A,C,F,L,N)
            assert self._are_equal([C_iso, create_isotropic_elasticity_tensor(lam, mu)]), "Doesn't correctly reduce to Isotropic case"
            C = create_transversely_isotropic_tensor(*tran_iso_params)
            R = rotation_matrix(self._gen_rand_az())
            Cprime = transform_tensor(C, R)
            assert self._are_equal([C, Cprime]), "Not transversely Isotropic preserved!"

    def test_christoffel_wavespeed(self):
        N = 50
        for i in np.arange(N_iters):
            cubic_params = 1. + self._gen_rand_mat([3,])
            rho = 1. + np.random.uniform()
            phi, theta, v_analytic = get_cubic_Pwavespeeds(*cubic_params, rho, N)
            C = create_cubic_elasticity_tensor(*cubic_params)
            phi, theta, v_numeric = get_acoustic_Pwavespeeds(C, rho, N)
            assert self._are_equal([v_analytic, v_numeric]), "Numerical solution of the Christoffel equation doesn't match analytic calc. "
