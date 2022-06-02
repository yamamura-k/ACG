cimport numpy as np
cdef extern from "_cluster_coef_2.hpp":
    cdef double compute_cluster_coef(int n, double step, float D[])
    cdef void compute_cluster_coef_batch(int bs, int n, double step, float D[], double Coef[])
    cdef void compute_cluster_coef_batch_instance_wise(int bs, int n, double step[], float D[], double Coef[])

cdef double _compute_cluster_coef_from_distance_matrix(int n, double step, np.ndarray[float, ndim=1] D):
    cdef double ans
    cdef float* _D = <float*> D.data
    ans = compute_cluster_coef(n, step, _D)
    return ans
cdef void _compute_cluster_coef_from_distance_matrix_batch(int bs, int n, double step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    compute_cluster_coef_batch(bs, n, step, _D, _C)
    return 
cdef void _compute_cluster_coef_from_distance_matrix_batch_instance_wise(int bs, int n, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    cdef double* _step = <double*> step.data
    compute_cluster_coef_batch_instance_wise(bs, n, _step, _D, _C)
    return 

def compute_cluster_coef_from_distance_matrix(int n, double step, np.ndarray[float, ndim=1] D):
    cdef double ans
    ans = _compute_cluster_coef_from_distance_matrix(n, step, D)
    return ans
def compute_cluster_coef_from_distance_matrix_batch(int bs, int n, double step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch(bs, n, step, D, C)
    return C
def compute_cluster_coef_from_distance_matrix_batch_instance_wise(int bs, int n, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch_instance_wise(bs, n, step, D, C)
    return C