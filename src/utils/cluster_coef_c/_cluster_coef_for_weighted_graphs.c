#include <math.h>

void compute_cluster_coef_batch_instance_wise_barrat(int bs, int n, double step[], float D[], double Coef[])
{
    int i, j, k, b, row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double n_adj_nodes = 0;
    double average_weight = 0;

    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        tmp = 0.0;
        for(i = 0; i < n; ++i)
        {
            row = n * i;
            average_weight = 0.0;
            n_adj_nodes = 0;
            for(j = 0; j < n; ++j)
            {
                if(i == j) continue;
                else if(D[batch_index + row + j] <= step[b])
                {
                    average_weight += D[batch_index + row + j];
                    n_adj_nodes++;
                    for(k = 0; k < n; ++k)
                    {
                        if((k == i) || (k == j)) continue;
                        else if(D[batch_index + row + k] <= step[b])
                        {
                            // 
                            tmp += (D[batch_index + row + j] + D[batch_index + row + k]);
                        }
                    }
                }
            }
            C += tmp / 2 / (average_weight / n_adj_nodes) / n_adj_nodes / (n_adj_nodes - 1);
        }
        Coef[b] = C / n;
        C = 0.0;
    }
    return ;
}
void compute_cluster_coef_batch_instance_wise_onnela(int bs, int n, double step[], float D[], double Coef[])
{
    int i, j, k, b, row, _row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double n_adj_nodes = 0;

    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        tmp = 0.0;
        for(i = 0; i < n; ++i)
        {
            row = n * i;
            n_adj_nodes = 0;
            for(j = 0; j < n; ++j)
            {
                if(i == j) continue;
                else if(D[batch_index + row + j] <= step[b])
                {
                    _row = n * j;
                    n_adj_nodes++;
                    for(k = 0; k < n; ++k)
                    {
                        if((k == i) || (k == j)) continue;
                        else if(D[batch_index + row + k] <= step[b])
                        {
                            // 
                            tmp += pow(D[batch_index + _row + k] * D[batch_index + row + j] * D[batch_index + row + k], 1/3);
                        }
                    }
                }
            }
            C += tmp / n_adj_nodes / (n_adj_nodes-1);
        }
        Coef[b] = C / n;
        C = 0.0;
    }
    return ;
}

void compute_cluster_coef_batch_instance_wise_zhang(int bs, int n, double step[], float D[], double Coef[])
{
    int i, j, k, b, row, _row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double weight_sum = 0.0;
    double square_sum = 0.0;

    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        tmp = 0.0;
        for(i = 0; i < n; ++i)
        {
            row = n * i;
            weight_sum = 0.0;
            square_sum = 0.0;
            for(j = 0; j < n; ++j)
            {
                if(i == j) continue;
                else if(D[batch_index + row + j] <= step[b])
                {
                    _row = n * j;
                    weight_sum += D[batch_index + row + j];
                    square_sum += D[batch_index + row + j] * D[batch_index + row + j];
                    for(k = 0; k < n; ++k)
                    {
                        if((k == i) || (k == j)) continue;
                        else if(D[batch_index + row + k] <= step[b])
                        {
                            // 
                            // D_jk * D_ij * D_ik
                            tmp += D[batch_index + _row + k] * D[batch_index + row + j] * D[batch_index + row + k];
                        }
                    }
                }
            }
            if(weight_sum != 0) C += tmp / (weight_sum * weight_sum - square_sum);
        }
        Coef[b] = C / n;
        C = 0.0;
    }
    return ;
}


/*
。
Reference:
   + Generalizations of the clustering coefficient to weighted complex networks
     (https://journals.aps.org/pre/pdf/10.1103/PhysRevE.75.027105)
   + Intensity and coherence of motifs in weighted complex networks
     (https://journals.aps.org/pre/pdf/10.1103/PhysRevE.71.065103)
   + 
*/
void compute_cluster_coef_batch_barrat(int bs, int n, float D[], double Coef[])
{
    int i, j, k, b, row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double denominator = (n - 1)*(n - 2);//
    double average_weight = 0.0;

    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        tmp = 0.0;
        for(i = 0; i < n; ++i)
        {
            row = n * i;
            average_weight = 0.0;
            for(j = 0; j < n; ++j)
            {
                if(i == j) continue;
                else
                {
                    average_weight += D[batch_index + row + j];
                    for(k = 0; k < n; ++k)
                    {
                        if((k == i) || (k == j)) continue;
                        else
                        {
                            // 
                            tmp += (D[batch_index + row + j] + D[batch_index + row + k]);
                        }
                    }
                }
            }
            C += tmp / 2 / (average_weight / (n - 1)) / denominator;
        }
        Coef[b] = C / n;
        C = 0.0;
    }
    return ;
}
void compute_cluster_coef_batch_onnela(int bs, int n, float D[], double Coef[])
{
    int i, j, k, b, row, _row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double denominator = (n - 1)*(n - 2);//

    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        tmp = 0.0;
        for(i = 0; i < n; ++i)
        {
            row = n * i;
            for(j = 0; j < n; ++j)
            {
                if(i == j) continue;
                else
                {
                    _row = n * j;
                    for(k = 0; k < n; ++k)
                    {
                        if((k == i) || (k == j)) continue;
                        else
                        {
                            // 
                            tmp += pow(D[batch_index + _row + k] * D[batch_index + row + j] * D[batch_index + row + k], 1/3);
                        }
                    }
                }
            }
            C += tmp / denominator;
        }
        Coef[b] = C / n;
        C = 0.0;
    }
    return ;
}

void compute_cluster_coef_batch_zhang(int bs, int n, float D[], double Coef[])
{
    int i, j, k, b, row, _row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double weight_sum = 0.0;
    double square_sum = 0.0;

    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        tmp = 0.0;
        for(i = 0; i < n; ++i)
        {
            row = n * i;
            weight_sum = 0.0;
            square_sum = 0.0;
            for(j = 0; j < n; ++j)
            {
                if(i == j) continue;
                else
                {
                    _row = n * j;
                    weight_sum += D[batch_index + row + j];
                    square_sum += D[batch_index + row + j] * D[batch_index + row + j];
                    for(k = 0; k < n; ++k)
                    {
                        if((k == i) || (k == j)) continue;
                        else
                        {
                            // 
                            tmp += D[batch_index + _row + k] * D[batch_index + row + j] * D[batch_index + row + k];
                        }
                    }
                }
            }
            if(weight_sum != 0) C += tmp / (weight_sum * weight_sum - square_sum);
        }
        Coef[b] = C / n;
        C = 0.0;
    }
    return ;
}