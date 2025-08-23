#include<mpi.h>
#include<iostream>
#include<vector>
using namespace std;

int main(int argc,char *argv[])
{
	MPI_Init(&argc,&argv);
	MPI_Status* status;
	int rank , size;

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);   //rank of the processor
	MPI_Comm_size(MPI_COMM_WORLD,&size);     //number of processor's

	int n = 100,part = n/size;
	//rank 0
	if(rank == 0){
		vector<int> A(n);
		for(int i = 0 ; i <n ; ++i)
			A[i] = i+1;

		MPI_Send(A.data()+part,part,MPI_INT,1,1,MPI_COMM_WORLD);

		int sum = 0;
		for(int i = 0 ; i < part ; ++i)
			sum += A[i];
		cout << "Rank " << rank  << ": " << sum << endl;

		int rem_sum;
		MPI_Recv(&rem_sum,1,MPI_INT,1,2,MPI_COMM_WORLD,status);
		sum += rem_sum;
		cout << "Total sum Sum : " << << endl;
	}

	//rank 1
	if(rank == 1){
		vector<int> B(part);
		MPI_Recv(B.data(),part,MPI_INT,0,1,MPI_COMM_WORLD,status);
		int part_sum = 0;
		for(int i = 0 ; i < part ; ++i)
			part_sum += B[i];
		cout << "Rank " << rank  << ": " << part_sum << endl;
		MPI_Send(&part_sum,1,MPI_INT,0,2,MPI_COMM_WORLD);
	}
	
	

	MPI_Finalize();

	return 0;	
}