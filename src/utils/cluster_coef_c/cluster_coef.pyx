cimport numpy as np
cdef extern from "_cluster_coef.h":
    cdef double compute_cluster_coef(int n, int n_split, double step, float D[])
    cdef void compute_cluster_coef_batch(int bs, int n, int n_split, double step, float D[], double Coef[])
    cdef void compute_cluster_coef_batch_instance_wise(int bs, int n, int n_split, double step[], float D[], double Coef[])
    cdef double _compute_cluster_coef(int n, int G[])

cdef double _compute_cluster_coef_from_distance_matrix(int n, int n_split, double step, np.ndarray[float, ndim=1] D):
    cdef double ans
    cdef float* _D = <float*> D.data
    ans = compute_cluster_coef(n, n_split, step, _D)
    return ans
cdef void _compute_cluster_coef_from_distance_matrix_batch(int bs, int n, int n_split, double step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    compute_cluster_coef_batch(bs, n, n_split, step, _D, _C)
    return 
cdef void _compute_cluster_coef_from_distance_matrix_batch_instance_wise(int bs, int n, int n_split, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    cdef double* _step = <double*> step.data
    compute_cluster_coef_batch_instance_wise(bs, n, n_split, _step, _D, _C)
    return 
cdef double _compute_cluster_coef_from_adjacent_matrix(int n, np.ndarray[int, ndim=1] G):
    cdef double ans
    cdef int* _G = <int*> G.data
    ans = _compute_cluster_coef(n, _G)
    return ans
def compute_cluster_coef_from_distance_matrix(int n, int n_split, double step, np.ndarray[float, ndim=1] D):
    cdef double ans
    ans = _compute_cluster_coef_from_distance_matrix(n, n_split, step, D)
    return ans
def compute_cluster_coef_from_distance_matrix_batch(int bs, int n, int n_split, double step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch(bs, n, n_split, step, D, C)
    return C
def compute_cluster_coef_from_distance_matrix_batch_instance_wise(int bs, int n, int n_split, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch_instance_wise(bs, n, n_split, step, D, C)
    return C
def compute_cluster_coef_from_adjacent_matrix(int n, np.ndarray[int, ndim=1] G):
    cdef double ans
    ans = _compute_cluster_coef_from_adjacent_matrix(n, G)
    return ans