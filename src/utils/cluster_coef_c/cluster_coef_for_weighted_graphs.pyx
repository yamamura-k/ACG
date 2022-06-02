cimport numpy as np
cdef extern from "_cluster_coef_for_weighted_graphs.h":
    cdef void compute_cluster_coef_batch_barrat(int bs, int n, float D[], double Coef[])
    cdef void compute_cluster_coef_batch_onnela(int bs, int n, float D[], double Coef[])
    cdef void compute_cluster_coef_batch_zhang(int bs, int n, float D[], double Coef[])
    
    cdef void compute_cluster_coef_batch_instance_wise_barrat(int bs, int n, double step[], float D[], double Coef[])
    cdef void compute_cluster_coef_batch_instance_wise_onnela(int bs, int n, double step[], float D[], double Coef[])
    cdef void compute_cluster_coef_batch_instance_wise_zhang(int bs, int n, double step[], float D[], double Coef[])


cdef void _compute_cluster_coef_from_distance_matrix_batch_onnela(int bs, int n, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    compute_cluster_coef_batch_onnela(bs, n, _D, _C)
    return 
cdef void _compute_cluster_coef_from_distance_matrix_batch_barrat(int bs, int n, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    compute_cluster_coef_batch_barrat(bs, n, _D, _C)
    return 
cdef void _compute_cluster_coef_from_distance_matrix_batch_zhang(int bs, int n, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    compute_cluster_coef_batch_zhang(bs, n, _D, _C)
    return 

cdef void _compute_cluster_coef_from_distance_matrix_batch_instance_wise_onnela(int bs, int n, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    cdef double* _step = <double*> step.data
    compute_cluster_coef_batch_instance_wise_onnela(bs, n, _step, _D, _C)
    return
cdef void _compute_cluster_coef_from_distance_matrix_batch_instance_wise_barrat(int bs, int n, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    cdef double* _step = <double*> step.data
    compute_cluster_coef_batch_instance_wise_barrat(bs, n, _step, _D, _C)
    return
cdef void _compute_cluster_coef_from_distance_matrix_batch_instance_wise_zhang(int bs, int n, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    cdef float* _D = <float*> D.data
    cdef double* _C = <double*> C.data
    cdef double* _step = <double*> step.data
    compute_cluster_coef_batch_instance_wise_zhang(bs, n, _step, _D, _C)
    return


def compute_cluster_coef_from_distance_matrix_batch_barrat(int bs, int n, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch_barrat(bs, n, D, C)
    return C
def compute_cluster_coef_from_distance_matrix_batch_onnela(int bs, int n, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch_onnela(bs, n, D, C)
    return C
def compute_cluster_coef_from_distance_matrix_batch_zhang(int bs, int n, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch_zhang(bs, n, D, C)
    return C

def compute_cluster_coef_from_distance_matrix_batch_instance_wise_barrat(int bs, int n, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch_instance_wise_barrat(bs, n, step, D, C)
    return C
def compute_cluster_coef_from_distance_matrix_batch_instance_wise_onnela(int bs, int n, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch_instance_wise_onnela(bs, n, step, D, C)
    return C
def compute_cluster_coef_from_distance_matrix_batch_instance_wise_zhang(int bs, int n, np.ndarray[double, ndim=1] step, np.ndarray[float, ndim=1] D, np.ndarray[double, ndim=1] C):
    ans = _compute_cluster_coef_from_distance_matrix_batch_instance_wise_zhang(bs, n, step, D, C)
    return C