// Parallel program for Matrix-vector multiplication

#include <vector>
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc,char *argv[])
{
	MPI_Init(&argc,&argv);
	int size,rank;
	
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	int n=4,m=4;
	vector<int> A(m*n),v(n);	
	if(rank == 0){
		//cout << "Enter matrix elements: \n";
		//for (int i = 0;i <m*n;++i)
		//	cin>>A[i];
		//for(int a: A)
		//	cout << a << " ";
		//cout << "\n";
		//cout << "Enter vector elements: \n";
		//for (int i = 0 ; i < n;++i)
		//	cin >> v[i];

		A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
		v = {10,11,12,13};

	}
	int row_size = (m*n) / size;
	vector<int> row(row_size);
	MPI_Bcast(v.data(),n,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scatter(A.data(),row_size,MPI_INT,row.data(),row_size,MPI_INT,0,MPI_COMM_WORLD);
	//cout << "Rank " << rank << ": " ;	
	//for(auto r : row)
	//	cout << r << " ";
	//cout << "\n";
	vector<int> ans(1);
	int ans_size = row_size / n;
	if(ans_size > 1)
		ans.resize(ans_size);
	for(int i = 0 ; i < ans_size ; ++i){
		ans[i] = 0;
		int idx_v = 0;
		for(int j = i*n ;j < (i+1)*n ;++j){
			ans[i] += v[idx_v++]*row[j];
		}
		cout << "Rank " << rank << ": "; 
		cout << ans[i] << " ";
	}
	
	vector<int> result;
	if (rank == 0) {result.resize(n);
	}
	MPI_Gather(ans.data(),ans_size,MPI_INT,result.data(),ans_size,MPI_INT,0,MPI_COMM_WORLD);
	
	if(rank == 0){
	for(int r:result)
		cout << r << "\n";
	}  
			
	MPI_Finalize();
	return 0;
}