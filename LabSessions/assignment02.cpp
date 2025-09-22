/**Parallelization of simpson 3/8 rule on 4 processors**/

#include<iostream>
#include<mpi.h>

using namespace std;

double f(double x)
{
	return 1/(x*x + 1);
}

int main(int argc,char* argv[])
{
	MPI_Init(&argc,&argv);
	MPI_Status *status;
	
	int rank,size;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	int n = 105;
	double a=1,b=2;
	double h = (b - a) / n;
	int q = (n-1)/size,rem = (n-1)%size;

	if(rank == 0){
		int start = rank*q+1;
		int end = start+q-1;

		float sum = f(a) + f(b);  
		for(int i = start ; i <= end ; ++i)
		{
			if(!(i%3))
				sum += 2*f(a + double(i)*h);
			else
				sum += 3*f(a + double(i)*h);
		}
			
		//cout << "Rank " << rank << ": " << sum << endl;
		float local_sums= 0;
		for(int i = 1 ; i < size ; ++i)
		{
			float rem_sum = 0;
			MPI_Recv(&rem_sum,1,MPI_FLOAT,i,1,MPI_COMM_WORLD,status);
			local_sums += rem_sum;
			//cout << "Sum received from processor : " << i  <<  "  " << rem_sum << endl;
			//cout << "Updated sum : " << local_sums << endl;
		}
		double ans = (3*h / 8) * (sum + local_sums);
		cout << "Answer : " << ans << endl;

	} 
	if(rank == 1){
		int start = rank*q+1;
		int end = start+q-1;
		float partial_sum = 0;  
		for(int i = start ; i <= end ; ++i)
		{
			if(!(i%3))
				partial_sum += 2*f(a + double(i)*h);
			else
				partial_sum += 3*f(a + double(i)*h);
		}
		//cout << "Rank " << rank << ": " << partial_sum << endl;	
		MPI_Send(&partial_sum,1,MPI_FLOAT,0,1,MPI_COMM_WORLD);
		//cout << "Sum send to processor 0 " << partial_sum << " " <<  rank<< endl;
	}
	if(rank == 2){
		int start = rank*q+1;
		int end = start+q-1;
		float partial_sum = 0;  
		for(int i = start ; i <= end ; ++i)
		{
			if(!(i%3))
				partial_sum += 2*f(a + double(i)*h);
			else
				partial_sum += 3*f(a + double(i)*h);
		}
		//cout << "Rank " << rank << ": " << partial_sum << endl;	
		MPI_Send(&partial_sum,1,MPI_FLOAT,0,1,MPI_COMM_WORLD);
		//cout << "Sum send to processor 0 " << partial_sum << " " <<  rank<< endl;
	}
	if(rank == 3){
		int start = rank*q+1;
		int end = start+q-1;
		float partial_sum = 0;  
		for(int i = start ; i <= end ; ++i)
		{
			if(!(i%3))
				partial_sum += 2*f(a + double(i)*h);
			else
				partial_sum += 3*f(a + double(i)*h);
		}
		//cout << "Rank " << rank << ": " << partial_sum << endl;	
		MPI_Send(&partial_sum,1,MPI_FLOAT,0,1,MPI_COMM_WORLD);
		//cout << "Sum send to processor 0 " << partial_sum << " " <<  rank<< endl;
	}

	MPI_Finalize();
	
	return 0;
}
