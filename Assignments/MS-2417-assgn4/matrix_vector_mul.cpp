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

    int n, m;
    vector<double> matrix, vec;
    vector<int> sendcounts(size), displs(size);

    if (rank == 0) {
        ifstream fin("input.txt");
        if (!fin.is_open()) {
            cerr << "Error: Cannot open input file." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fin >> n >> m;
        matrix.resize(n * m);
        vec.resize(m);

        for (int i = 0; i < n * m; i++)
            fin >> matrix[i];
        for (int i = 0; i < m; i++)
            fin >> vec[i];
        fin.close();

        int rows_per_proc = n / size, remainder = n % size;
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * m;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast the vector to all processes
    if (rank != 0) vec.resize(m);
    MPI_Bcast(vec.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter matrix rows
    int recvcount;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<double> local_matrix(recvcount);

    MPI_Scatterv(matrix.data(), sendcounts.data(), displs.data(),
                 MPI_DOUBLE, local_matrix.data(), recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_rows = recvcount / m;
    vector<double> local_result(local_rows, 0.0);

    // Compute local part of matrix-vector product
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < m; ++j)
            local_result[i] += local_matrix[i * m + j] * vec[j];

    // Gather results back to root
    vector<int> recvcounts(size), recvdispls(size);
    if (rank == 0) {
        int rows_per_proc = n / size, remainder = n % size;
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = rows_per_proc + (i < remainder ? 1 : 0);
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
    }

    vector<double> result;
    if (rank == 0) result.resize(n);

    MPI_Gatherv(local_result.data(), local_rows, MPI_DOUBLE,
                result.data(), recvcounts.data(), recvdispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ofstream fout("output.txt");
        for (double val : result)
            fout << val << "\n";
        fout.close();

        cout << "Output written to output.txt\n";
    }

    MPI_Finalize();
    return 0;
}
