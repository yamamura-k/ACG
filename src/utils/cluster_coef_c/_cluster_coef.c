// #include <stdio.h>
void compute_cluster_coef_batch_instance_wise(int bs, int n, int n_split, double step[], float D[], double Coef[])
{
    int i, j, k, a, b, row, _row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double n_adj_nodes = 0;
    double n_edges = 0;
    double K = 0.0;
    /*// debug
    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        printf("\nbatch index = %d\n", b);
        for(i = 0; i < n; i++)
        {
            for(j = 0; j < n; j++)
            {
                printf("%f ", D[batch_index + n * i + j]);
            }
            printf("\n");
        }
    }
    */
    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        for(a = 0; a < n_split; ++a)
        {
            tmp = 0.0;
            for(i = 0; i < n; ++i)
            {
                n_adj_nodes = 0;
                n_edges = 0;
                row = n * i;
                for(j = 0; j < n; ++j)
                {
                    if(i == j) continue;
                    else if(D[batch_index + row + j] <= K)
                    {
                        n_adj_nodes++;
                        _row = n * j;
                        for(k = j + 1; k < n; ++k)
                        {
                            if((i == k) || (D[batch_index + row + k] > K)) continue;
                            else if(D[batch_index + _row + k] <= K) n_edges++;
                        }
                    }
                }
                if(n_adj_nodes >= 2) tmp += 2 * n_edges / n_adj_nodes / (n_adj_nodes - 1);
            }
            C += tmp / n;
            K += step[b];
        }
        Coef[b] = C / n_split;
        C = 0.0;
        K = 0.0;
    }
    return ;
}

void compute_cluster_coef_batch(int bs, int n, int n_split, double step, float D[], double Coef[])
{
    int i, j, k, a, b, row, _row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double n_adj_nodes = 0;
    double n_edges = 0;
    double K = 0.0;
    /*// debug
    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        printf("\nbatch index = %d\n", b);
        for(i = 0; i < n; i++)
        {
            for(j = 0; j < n; j++)
            {
                printf("%f ", D[batch_index + n * i + j]);
            }
            printf("\n");
        }
    }
    */
    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        for(a = 0; a < n_split; ++a)
        {
            tmp = 0.0;
            for(i = 0; i < n; ++i)
            {
                n_adj_nodes = 0;
                n_edges = 0;
                row = n * i;
                for(j = 0; j < n; ++j)
                {
                    if(i == j) continue;
                    else if(D[batch_index + row + j] <= K)
                    {
                        n_adj_nodes++;
                        _row = n * j;
                        for(k = j + 1; k < n; ++k)
                        {
                            if((i == k) || (D[batch_index + row + k] > K)) continue;
                            else if(D[batch_index + _row + k] <= K) n_edges++;
                        }
                    }
                }
                if(n_adj_nodes >= 2) tmp += 2 * n_edges / n_adj_nodes / (n_adj_nodes - 1);
            }
            C += tmp / n;
            K += step;
        }
        Coef[b] = C / n_split;
        C = 0.0;
        K = 0.0;
    }
    return ;
}

double compute_cluster_coef(int n, int n_split, double step, float D[])
{
    int i, j, k, a, row, _row;
    double C = 0.0;
    double tmp = 0.0;
    double n_adj_nodes = 0;
    double n_edges = 0;
    double K = 0.0;
    /* // debug
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            printf("%f ", D[n * i + j]);
        }
        printf("\n");
    }
    */
    for(a = 0; a < n_split; ++a)
    {
        tmp = 0.0;
        for(i = 0; i < n; ++i)
        {
            n_adj_nodes = 0;
            n_edges = 0;
            row = n * i;
            for(j = 0; j < n; ++j)
            {
                if(i == j) continue;
                else if(D[row + j] <= K)
                {
                    n_adj_nodes++;
                    _row = n * j;
                    for(k = j + 1; k < n; ++k)
                    {
                        if((i == k) || (D[row + k] > K)) continue;
                        else if(D[_row + k] <= K) n_edges++;
                    }
                }
            }
            if(n_adj_nodes >= 2) tmp += 2 * n_edges / n_adj_nodes / (n_adj_nodes - 1);
        }
        C += tmp / n;
        K += step;
    }
    C /= n_split;
    return C;
}

double _compute_cluster_coef(int n, int G[])
{
    double C = 0.0;
    int i, j, k, row, _row;
    double n_adj_nodes = 0;
    double n_edges = 0;
    for(i = 0; i < n; ++i)
    {
        n_adj_nodes = 0;
        n_edges = 0;
        row = n * i;
        for(j = 0; j < n; ++j)
        {
            if(G[row + j] == 0) continue;
            n_adj_nodes++;
            _row = n * j;
            for(k = j; k < n; ++k)
            {
                if(G[row + k] == 0) continue;
                n_edges += G[_row + k];
            }
        }
        if(n_adj_nodes >= 2) C += 2 * n_edges / n_adj_nodes / (n_adj_nodes - 1);
    }
    return C / n;
}