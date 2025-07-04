from unittest import TestCase

import numpy as np
from exps import compute_v_and_o_lower_tri, generate_random_dag_adjacency_matrix, get_random_linear_scm_and_u, hp_cause_linear

class TestDAGGeneration(TestCase):
    def test_generate_random_dag_adjacency_matrix(self):
        n = 5
        adj = generate_random_dag_adjacency_matrix(n, p=0.4, seed=42)
        self.assertEqual(adj.shape, (n, n))
        # assert no self-loops
        self.assertTrue(np.all(np.diag(adj) == 0))

    # def test_get_random_adj(self):
    #     o_thr = 10
    #     n = 10
    #     o_func = lambda v: (v[0] + v[1] + .5 * v[2])**2 + v[3]**2 + v[4]**2
    #     adj_res = get_random_adj(n, o_func, o_thr, edge_prob=0.4)
    #     adj_np = adj_res['adj']
    #     u_np = adj_res['u']
    #     v_np = adj_res['v']
    #     o_np = adj_res['o']
    #     self.assertTrue(o_thr <= o_np)


    def test_hp_cause(self):
        n = 5
        o_func = lambda v: (v[0] + v[1] + .5 * v[2])**2 + v[3]**2 + v[4]**2
        o_thr = 10

        # adj_res = get_random_adj(n, o_func, o_thr, edge_prob=0.4)
        # adj = adj_res['adj']
        # u = adj_res['u']
        adj = np.array([[ 0,  0,  0,  0,  0],
            [-3,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 3,  0, -2,  0,  0],
            [-4, -4,  0,  0,  0]])
        u = np.array([-0.5,  0. , -2. , -2.5,  0. ])
        V = set(range(n))
        
        for res in hp_cause_linear(adj, V, u, o_func, o_thr):
            self.assertIsNotNone(res)
            break
        
        print(repr(adj))
        print(repr(u))
        v_no_int, o_no_int = compute_v_and_o_lower_tri(adj, u, None, o_func)
        self.assertTrue(o_thr <= o_no_int)

        ints = res['cf'] | dict(zip(res['W'], res['w']))
        v_int, o_int = compute_v_and_o_lower_tri(adj, u, ints, o_func)
        self.assertTrue(o_thr > o_int)

    def test_hp_cause_nn(self):
        adj_np = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 3,  3,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  2,  0,  0,  0,  0,  0,  0,  0],
            [ 4,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  2, -4,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  0,  0,  0,  2,  1,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  1,  0,  2,  0,  0,  0],
            [-3, -1,  0,  0, -1,  1, -3, -1,  0,  0],
            [ 0,  0,  2,  0,  0,  0, -2,  0,  0,  0]])

        u_np = np.array([-1.,  0.,  0.,  6.,  0.,  0.,  0.,  0., -5.,  0.])
        # res = one_expr(n, o_func, o_thr, adj_np=adj_np, u_np=u_np)

if __name__ == "__main__":
    import unittest
    unittest.main()