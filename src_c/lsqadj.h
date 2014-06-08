/* Forward declarations for matrix operations defined in lsqadj.c */

#ifndef LSQADJ_H
#define LSQADJ_H

void ata(double *a, double *ata, int m, int n);
void ata_v2(double *a, double *ata, int m, int n, int n_large);
void atl(double *a, double *u, double *l, int m, int n);

void matinv(double *a, int n);
void matmul(double *a, double *b, double *c,
    int m, int n, int k);
void mat_transpose(double *mat1, double *mat2, int m, int n);

void atl_v2 (double *a, double *u, double *l, int m, int n, int n_large);
void matinv_v2 (double *a, int n, int n_large);
void matmul_v2 (double *a, double *b, double *c, int m, int n, int k, int m_large, int n_large);


#endif
