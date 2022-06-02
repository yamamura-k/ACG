#include <vector>
#include <algorithm>

void extract_elements(int bs, int n, double step[], float D[], std::vector<std::vector<float>> &threshould)
{
    for(int b = 0; b < bs; ++b)
    {
        int batch_index = b * n * n;
        std::vector<float> tmp;
        for(int i = 0; i < n; ++i)
        {
            int row = n * i;
            for(int j = 0; j < n; ++j)
            {
                if(i == j) continue;
                float _k = D[batch_index + row + j];
                if (_k > step[b]) continue;
                auto result = std::find(tmp.begin(), tmp.end(), _k);
                if(result == tmp.end()) tmp.push_back(_k);
            }
        }
        tmp.push_back(step[b]);
        std::sort(tmp.begin(), tmp.end());
        threshould[b] = tmp;
    }
    return ;
}

void extract_elements(int bs, int n, double step, float D[], std::vector<std::vector<float>> &threshould)
{
    for(int b = 0; b < bs; ++b)
    {
        int batch_index = b * n * n;
        std::vector<float> tmp;
        for(int i = 0; i < n; ++i)
        {
            int row = n * i;
            for(int j = 0; j < n; ++j)
            {
                if(i == j) continue;
                float _k = D[batch_index + row + j];
                if (_k > step) continue;
                auto result = std::find(tmp.begin(), tmp.end(), _k);
                if(result == tmp.end()) tmp.push_back(_k);
            }
        }
        tmp.push_back(step);
        std::sort(tmp.begin(), tmp.end());
        threshould[b] = tmp;
    }
    return ;
}

void extract_elements(int n, double step, float D[], std::vector<float> &threshould)
{
    for(int i = 0; i < n; ++i)
    {
        int row = n * i;
        for(int j = 0; j < n; ++j)
        {
            if(i == j) continue;
            float _k = D[row + j];
            if (_k > step) continue;
            auto result = std::find(threshould.begin(), threshould.end(), _k);
            if(result == threshould.end()) threshould.push_back(_k);
        }
    }
    threshould.push_back(step);
    std::sort(threshould.begin(), threshould.end());
    return ;
}

void compute_cluster_coef_batch_instance_wise(int bs, int n, double step[], float D[], double Coef[])
{
    int i, j, k, b, row, _row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double n_adj_nodes = 0;
    double n_edges = 0;

    std::vector<std::vector<float>> threshould(bs);
    extract_elements(bs, n, step, D, threshould);
    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        float K_1 = 0;
        for(auto K : threshould[b])
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
            auto w = (K - K_1) / step[b];
            auto coef = tmp / n;
            C += w * coef;
            K_1 = K;
        }
        Coef[b] = C;
        C = 0.0;
    }
    return ;
}

void compute_cluster_coef_batch(int bs, int n, double step, float D[], double Coef[])
{
    int i, j, k, b, row, _row, batch_index;
    double C = 0.0;
    double tmp = 0.0;
    double n_adj_nodes = 0;
    double n_edges = 0;

    std::vector<std::vector<float>> threshould(bs);
    extract_elements(bs, n, step, D, threshould);
    for(b = 0; b < bs; ++b)
    {
        batch_index = b * n * n;
        float K_1 = 0;
        for(auto K : threshould[b])
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
            auto w = (K - K_1) / step;
            auto coef = tmp / n;
            C += w * coef;
            K_1 = K;
        }
        Coef[b] = C;
        C = 0.0;
    }
    return ;
}

double compute_cluster_coef(int n, double step, float D[])
{
    int i, j, k, row, _row;
    double C = 0.0;
    double tmp = 0.0;
    double n_adj_nodes = 0;
    double n_edges = 0;

    std::vector<float> threshould;
    extract_elements(n, step, D, threshould);
    float K_1 = 0.0;
    for(auto K : threshould)
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
    auto w = (K - K_1) / step;
    auto coef = tmp / n;
    C += w * coef;
    K_1 = K;
    }
    return C;
}