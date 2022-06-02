double compute_cluster_coef(int n, int n_split, double step, float D[]);
void compute_cluster_coef_batch(int bs, int n, int n_split, double step, float D[], double Coef[]);
void compute_cluster_coef_batch_instance_wise(int bs, int n, int n_split, double step[], float D[], double Coef[]);
double _compute_cluster_coef(int n, int G[]);