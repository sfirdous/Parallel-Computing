#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n, m, p; // A is n x m, B is m x p
    vector<double> A, B;
    vector<int> sendcounts(size), displs(size);

    if (rank == 0) {
        ifstream fin("input.txt");
        if (!fin.is_open()) {
            cerr << "Could not open input file\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fin >> n >> m >> p;
        A.resize(n * m);
        B.resize(m * p);

        for (int i = 0; i < n * m; i++) fin >> A[i];
        for (int i = 0; i < m * p; i++) fin >> B[i];
        fin.close();

        int rows_per_proc = n / size;
        int remainder = n % size;

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * m;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) B.resize(m * p);
    MPI_Bcast(B.data(), m * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int recvcount;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<double> A_local(recvcount);

    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_rows = recvcount / m;
    vector<double> C_local(local_rows * p, 0.0);

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += A_local[i * m + k] * B[k * p + j];
            }
            C_local[i * p + j] = sum;
        }
    }

    vector<int> recvcounts(size), recvdispls(size);
    if (rank == 0) {
        int rows_per_proc = n / size;
        int remainder = n % size;

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * p;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
    }

    vector<double> C;
    if (rank == 0) C.resize(n * p);

    MPI_Gatherv(C_local.data(), local_rows * p, MPI_DOUBLE,
                C.data(), recvcounts.data(), recvdispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ofstream fout("output.txt");
        fout << "Resultant Matrix (" << n << " x " << p << "):\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                fout << C[i * p + j] << " ";
            }
            fout << "\n";
        }
        fout.close();
        cout << "Written output matrix to output.txt\n";
    }

    MPI_Finalize();
    return 0;
}
