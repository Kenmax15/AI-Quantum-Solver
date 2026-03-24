/*
  xxz_dataset_generator.c

  Minimal dataset generator for the disordered XXZ spin-1/2 chain
  in a fixed magnetization sector K_up.

  Hamiltonian:
    H = sum_i [ Jxy/2 (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ ) + Jz S_i^z S_{i+1}^z ]
        + sum_i h_i S_i^z

  Open boundary conditions by default.

  Output:
    CSV file with columns:
    sample_id,N,Kup,Jxy,Jz,W,seed,h_1,...,h_N,E0,E1,gap

  Compile:
    gcc -O3 -std=c11 xxz_dataset_generator.c -llapack -lblas -lm -o xxzgen

  Run:
    ./xxzgen
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* LAPACK dsyev prototype */
extern void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
                   double* w, double* work, int* lwork, int* info);

/* ---------- Bit utilities ---------- */

static inline int popcount_u32(uint32_t x) {
    return __builtin_popcount(x);
}

static inline int bit_at(uint32_t x, int pos) {
    return (x >> pos) & 1U;
}

static inline uint32_t flip_two_bits(uint32_t x, int i, int j) {
    x ^= (1U << i);
    x ^= (1U << j);
    return x;
}

/* ---------- Basis construction ---------- */

typedef struct {
    uint32_t* states;
    int dim;
    int N;
    int Kup;
} BasisSector;

int cmp_u32(const void* a, const void* b) {
    uint32_t x = *(const uint32_t*)a;
    uint32_t y = *(const uint32_t*)b;
    return (x > y) - (x < y);
}

BasisSector build_basis_sector(int N, int Kup) {
    BasisSector basis;
    basis.N = N;
    basis.Kup = Kup;
    basis.dim = 0;
    basis.states = NULL;

    const uint32_t dim_full = (1U << N);

    /* count */
    for (uint32_t s = 0; s < dim_full; ++s) {
        if (popcount_u32(s) == Kup) {
            basis.dim++;
        }
    }

    basis.states = (uint32_t*)malloc((size_t)basis.dim * sizeof(uint32_t));
    if (!basis.states) {
        fprintf(stderr, "Out of memory while allocating basis.\n");
        exit(EXIT_FAILURE);
    }

    int idx = 0;
    for (uint32_t s = 0; s < dim_full; ++s) {
        if (popcount_u32(s) == Kup) {
            basis.states[idx++] = s;
        }
    }

    qsort(basis.states, (size_t)basis.dim, sizeof(uint32_t), cmp_u32);
    return basis;
}

int basis_index(const BasisSector* basis, uint32_t state) {
    uint32_t* found = (uint32_t*)bsearch(
        &state,
        basis->states,
        (size_t)basis->dim,
        sizeof(uint32_t),
        cmp_u32
    );
    if (!found) return -1;
    return (int)(found - basis->states);
}

void free_basis(BasisSector* basis) {
    free(basis->states);
    basis->states = NULL;
    basis->dim = 0;
}

/* ---------- Random helpers ---------- */

double rand_uniform(double a, double b) {
    return a + (b - a) * ((double)rand() / (double)RAND_MAX);
}

/* ---------- Hamiltonian construction ---------- */

double sz_value(int bit) {
    return bit ? 0.5 : -0.5;
}

void build_xxz_sector_hamiltonian(
    const BasisSector* basis,
    double Jxy,
    double Jz,
    const double* h,
    int periodic,
    double* H /* dense dim x dim, column-major for LAPACK */
) {
    const int N = basis->N;
    const int dim = basis->dim;
    const int nbonds = periodic ? N : (N - 1);

    /* zero matrix */
    for (int i = 0; i < dim * dim; ++i) H[i] = 0.0;

    for (int n = 0; n < dim; ++n) {
        uint32_t s = basis->states[n];
        double diag = 0.0;

        /* disorder term */
        for (int i = 0; i < N; ++i) {
            diag += h[i] * sz_value(bit_at(s, i));
        }

        /* bond terms */
        for (int i = 0; i < nbonds; ++i) {
            int j = (i + 1) % N;

            int bi = bit_at(s, i);
            int bj = bit_at(s, j);

            double szi = sz_value(bi);
            double szj = sz_value(bj);

            /* diagonal Ising part */
            diag += Jz * szi * szj;

            /* flip-flop part */
            if (bi != bj) {
                uint32_t sf = flip_two_bits(s, i, j);
                int m = basis_index(basis, sf);
                if (m >= 0) {
                    /* column-major: H[row + col*dim] */
                    H[n + m * dim] += 0.5 * Jxy;
                }
            }
        }

        H[n + n * dim] += diag;
    }

    /* enforce exact symmetry */
    for (int col = 0; col < dim; ++col) {
        for (int row = col + 1; row < dim; ++row) {
            double avg = 0.5 * (H[row + col * dim] + H[col + row * dim]);
            H[row + col * dim] = avg;
            H[col + row * dim] = avg;
        }
    }
}

/* ---------- Diagonalization ---------- */

void diagonalize_dense_symmetric(double* H, int dim, double* evals) {
    char jobz = 'N';  /* eigenvalues only */
    char uplo = 'U';
    int lda = dim;
    int info;

    int lwork = -1;
    double wkopt;
    dsyev_(&jobz, &uplo, &dim, H, &lda, evals, &wkopt, &lwork, &info);
    if (info != 0) {
        fprintf(stderr, "LAPACK workspace query failed, info=%d\n", info);
        exit(EXIT_FAILURE);
    }

    lwork = (int)wkopt;
    double* work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) {
        fprintf(stderr, "Out of memory in LAPACK workspace allocation.\n");
        exit(EXIT_FAILURE);
    }

    dsyev_(&jobz, &uplo, &dim, H, &lda, evals, work, &lwork, &info);
    free(work);

    if (info != 0) {
        fprintf(stderr, "LAPACK dsyev failed, info=%d\n", info);
        exit(EXIT_FAILURE);
    }
}

/* ---------- CSV generation ---------- */

void write_csv_header(FILE* fp, int N) {
    fprintf(fp, "sample_id,N,Kup,Jxy,Jz,W,seed");
    for (int i = 0; i < N; ++i) {
        fprintf(fp, ",h_%d", i + 1);
    }
    fprintf(fp, ",E0,E1,gap\n");
}

int main(void) {
    /* ---------- user parameters ---------- */
    const int N = 12;
    const int Kup = 6;
    const double Jxy = 1.0;
    const double Jz_min = 0.8;
    const double Jz_max = 1.5;
    const double W_min = 0.0;
    const double W_max = 6.0;
    const int nsamples = 1000;
    const int periodic = 0;
    const unsigned int base_seed = 12345;
    const char* outname = "xxz_dataset.csv";

    srand(base_seed);

    BasisSector basis = build_basis_sector(N, Kup);
    printf("Basis built: N=%d, Kup=%d, dim=%d\n", N, Kup, basis.dim);

    double* H = (double*)malloc((size_t)basis.dim * (size_t)basis.dim * sizeof(double));
    double* evals = (double*)malloc((size_t)basis.dim * sizeof(double));
    double* h = (double*)malloc((size_t)N * sizeof(double));

    if (!H || !evals || !h) {
        fprintf(stderr, "Out of memory for working arrays.\n");
        free_basis(&basis);
        free(H);
        free(evals);
        free(h);
        return EXIT_FAILURE;
    }

    FILE* fp = fopen(outname, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file %s\n", outname);
        free_basis(&basis);
        free(H);
        free(evals);
        free(h);
        return EXIT_FAILURE;
    }

    write_csv_header(fp, N);

    for (int sample_id = 0; sample_id < nsamples; ++sample_id) {
        unsigned int seed = (unsigned int)rand();
        srand(seed);

        double Jz = rand_uniform(Jz_min, Jz_max);
        double W = rand_uniform(W_min, W_max);

        for (int i = 0; i < N; ++i) {
            h[i] = rand_uniform(-W, W);
        }

        build_xxz_sector_hamiltonian(&basis, Jxy, Jz, h, periodic, H);
        diagonalize_dense_symmetric(H, basis.dim, evals);

        double E0 = evals[0];
        double E1 = evals[1];
        double gap = E1 - E0;

        fprintf(fp, "%d,%d,%d,%.12g,%.12g,%.12g,%u",
                sample_id, N, Kup, Jxy, Jz, W, seed);

        for (int i = 0; i < N; ++i) {
            fprintf(fp, ",%.12g", h[i]);
        }

        fprintf(fp, ",%.12g,%.12g,%.12g\n", E0, E1, gap);

        if ((sample_id + 1) % 10 == 0) {
            printf("Generated %d / %d samples\n", sample_id + 1, nsamples);
        }

        srand(base_seed + sample_id + 1);
    }

    fclose(fp);
    free_basis(&basis);
    free(H);
    free(evals);
    free(h);

    printf("Dataset saved to %s\n", outname);
    return EXIT_SUCCESS;
}