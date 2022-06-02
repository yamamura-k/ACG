/*

Reference:
   + Generalizations of the clustering coefficient to weighted complex networks
     (https://journals.aps.org/pre/pdf/10.1103/PhysRevE.75.027105)
   + Intensity and coherence of motifs in weighted complex networks
     (https://journals.aps.org/pre/pdf/10.1103/PhysRevE.71.065103)
   + 
*/
void compute_cluster_coef_batch_instance_wise_barrat(int bs, int n, double step[], float D[], double Coef[]);
void compute_cluster_coef_batch_instance_wise_onnela(int bs, int n, double step[], float D[], double Coef[]);
void compute_cluster_coef_batch_instance_wise_zhang(int bs, int n, double step[], float D[], double Coef[]);

void compute_cluster_coef_batch_barrat(int bs, int n, float D[], double Coef[]);
void compute_cluster_coef_batch_onnela(int bs, int n, float D[], double Coef[]);
void compute_cluster_coef_batch_zhang(int bs, int n, float D[], double Coef[]);