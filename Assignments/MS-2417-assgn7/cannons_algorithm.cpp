#include<iostream>
#include<mpi.h>
#include<fstream>
#include<vector>

using namespace std;

int main(int argc,char *argv[])
{
    MPI_Init(&argc,&argv);
    int rank,size;

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int ndims = 2;
    int dims[2] = {0,0};

    MPI_Dims_create(size,ndims,dims);

    int periods[2] = {1,1};

    int reorder = 1;
    MPI_Comm CANNONS_MMUL;

    MPI_Cart_create(MPI_COMM_WORLD,ndims,dims,periods,reorder,&CANNONS_MMUL);

    int cart_rank;
    MPI_Comm_rank(CANNONS_MMUL,&cart_rank);

    int coords[2];
    MPI_Cart_coords(CANNONS_MMUL,cart_rank,ndims,coords);

    cout << "Rank : " << cart_rank <<  " Coords: (" << coords[0] << ", " << coords[1] << ")" << endl; 

    // read matrices from file 
    int n,m,p;
    vector<int> A,B;
    if(rank == 0)
    {
        ifstream fin("input.txt");
        if(!fin.is_open()){
            cerr << "Error: Cannot open file";
            MPI_Abort(CANNONS_MMUL,1);
        }

        fin >> n >> m >> p;
        A.resize(n*m),B.resize(m*p);

        for(int i = 0;i< n*m;++i)   fin >> A[i];
        for(int i = 0;i< m*p;++i)   fin >> B[i];

    }

    MPI_Bcast(&n,1,MPI_INT,0,CANNONS_MMUL);
    MPI_Bcast(&m,1,MPI_INT,0,CANNONS_MMUL);
    MPI_Bcast(&p,1,MPI_INT,0,CANNONS_MMUL);


    int n_per_proc = n / dims[0];
    int m_per_proc = m /dims[1];
    int p_per_proc = p /dims[1];

    int block_A[n_per_proc*m_per_proc],block_B[m_per_proc*p_per_proc];
    vector<int> par_A(n*m),per_B(m*p);
    if(rank == 0)
    {
        int index = 0;
        for(int i = 0 ; i < dims[0];++i)
        {
            for(int j = 0;j < dims[1];++j)
            {
                int start_row_a = i*n_per_proc,
                end_row_a = (i+1)*n_per_proc,
                start_col_a = j*m_per_proc,
                end_col_a = (j+1)*m_per_proc,
                start_col_b = (j)*p_per_proc,
                end_col_b = (j+1)*p_per_proc;

                for(int r = start_row_a;r < end_row_a;++r)
                {
                    for(int c = start_col_a;c < end_col_a;++c)
                    {
                        int global_row = r,
                        global_col = c;

                        par_A[index++] = A[global_row*n+global_col];
                        
                    }
                }
            }
        }
        for(int a: par_A)
            cout << a << " ";
    }

    

    MPI_Finalize();
    return 0;
}