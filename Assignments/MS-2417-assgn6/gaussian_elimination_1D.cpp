#include<iostream>
#include<mpi.h>
#include<vector>
#include<fstream>

using namespace std;

void gaussian_elimination(vector<double>& A,vector<double>& b)
{
    int n = b.size();
    for(int i = 0; i < n ; ++i)
    {
        double pivot = A[i*n+i];
        for(int j = i ; j < n;++j )
            A[i*n+j] /= pivot;                          // Division step
        b[i] /=  pivot;

        for(int j = i+1;j < n;++j)
        {
            double factor = A[j*n+i];
            for(int k = i ; k < n ; ++k)
                A[j*n+k] -= factor*A[i*n+k];          // Elimination step
            b[j] -= factor * b[i];

        }
    }
}

int read_from_file(string filename,vector<double>& A,vector<double>& b)
{
    ifstream fin(filename);
    if(!fin.is_open())  return 1;

    int n;
    fin >> n;
    A.resize(n*n);
    b.resize(n);

    for(int i = 0 ; i < n*n ; ++i)
        fin >> A[i];
    
    for (int i = 0; i < n; i++)
        fin >> b[i];
    return 0;
}

int main(int argc,char *argv[])
{

    MPI_Init(&argc,&argv);
    int rank,size;

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    vector<double> A,b;
    if(rank == 0)
    {
        if(read_from_file("input.txt",A,b))
        {
            cerr << "Error: Cannot open input file." << endl;
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        gaussian_elimination(A,b);
        for(int i = 0 ; i < b.size() ; ++i)
        {
            for(int j = 0 ; j < b.size() ; ++j)
                cout << A[i*b.size()+j] << " ";
            cout << "\n";
        }

        for(int element:b)
            cout << element << " ";
        
    }


    MPI_Finalize();
    return 0;
}