#include <iostream>
#include <mpi.h>
#include <vector>
#include <fstream>

using namespace std;

vector<double> back_substitutio(vector<double>& A,vector<double>&b)
{
    int n = b.size();
    vector<double> x(n);

    for(int i = n -1 ; i>= 0 ;--i)
    {
        x[i] = b[i];
        for(int j = i-1 ; j >= 0;--j)
            b[j] -= x[i]*A[j*n+i];
    }

    return x;
}

int read_from_file(string filename, vector<double> &A, vector<double> &b)
{
    ifstream fin(filename);
    if (!fin.is_open())
        return 1;

    int n;
    fin >> n;
    A.resize(n * n);
    b.resize(n);

    for (int i = 0; i < n * n; ++i)
        fin >> A[i];

    for (int i = 0; i < n; i++)
        fin >> b[i];
    return 0;
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<double> A, b;
    vector<int> sendcounts(size), disp(size); // arrays to store # of elements to distribute to each processor and the displacement
    int n;
    if (rank == 0)
    {
        if (read_from_file("input.txt", A, b)) // read coeff matrix and b from file
        {
            cerr << "Error: Cannot open input file." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = b.size(); // dimension of A and b
        if (size != n) // number of processors cannot be greater than # of rows of A
        {
            cerr << "Error: p must be equal to " << n << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int row_per_pro = n / size, rem = n % size;

        int offset = 0;

        for (int i = 0; i < size; ++i) // populate sendcounts and disp array
        {
            sendcounts[i] = (row_per_pro + (i < rem ? 1 : 0)) * n;
            disp[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int recvcount;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<double> row(recvcount);

    MPI_Scatterv(A.data(), sendcounts.data(), disp.data(), MPI_DOUBLE, row.data(), recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_n = recvcount / n;
    vector<double> b_local(local_n);
    if (rank == 0)
    {
        for (int i = 0; i < size; ++i)
        {
            sendcounts[i] /= n;
            disp[i] /= n;
        }
    }
    MPI_Scatterv(b.data(), sendcounts.data(), disp.data(), MPI_DOUBLE, b_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < n; ++i)
    {
        // ith processor perform division
        vector<double> pivot_row(n);
        double pivot_b;
        if (rank == i)
        {
            double pivot = row[i];
            for (int j = i; j < n; ++j)
                row[j] /= pivot; // division for coeff matrix
            b_local[0] /= pivot;
            pivot_row = row;
            pivot_b = b_local[0];
        }

        // then performs one-to-all broadcast of the ith/pivot row)
        MPI_Bcast(pivot_row.data(), n, MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
        

        // i+1 processors perform elimination step
        if (rank > i)
        {
            double factor = row[i];
            for (int j = i; j < n; ++j)
            {
                row[j] -= factor * pivot_row[j];
            }
            b_local[0] -= factor * pivot_b;
        }

        // gather reduced A and b in each step
        MPI_Gather(row.data(),n,MPI_DOUBLE,A.data(),n,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Gather(b_local.data(),1,MPI_DOUBLE,b.data(),1,MPI_DOUBLE,0,MPI_COMM_WORLD);

    }

    if(rank == 0)
    {
        vector<double> sol = back_substitutio(A,b);
        for(int i = 0; i < sol.size() ; ++i)
            cout << "x" << i+1 << " = " << sol[i] << endl; 
    }
    


    MPI_Finalize();
    return 0;
}