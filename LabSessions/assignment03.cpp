/**Parallel Program to perform Quick Sort on 4 processor's**/

#include <iostream>
#include <mpi.h>
#include <vector>
#include <algorithm>

using namespace std;

// Partition function for quick sort
int partition(vector<int> &arr,int low,int high)
{
	int pivot = arr[high];
	int i = low -1;
	for(int j = low ; j < high;++j)
	{
		if(arr[j] <= pivot)
		{
			++i;
			swap(arr[i],arr[j]);
		}		
	}
	swap(arr[i+1],arr[high]);
	return i+1;
}

// Quick Sort Algorithm
void quicksort(vector<int> &arr,int low,int high)
{
	if(low < high)
	{
		int pi = partition(arr,low,high);
		quicksort(arr,low,pi-1);
		quicksort(arr,pi+1,high);
	}
}

// Function to merge two sorted array's
vector<int> merge(vector<int> &left,vector<int> &right)
{
	vector<int> result;
	int i = 0,j = 0;
	while(i < left.size() && j < right.size())
	{	
		if(left[i] < right[j])
			result.push_back(left[i++]);
		else
			result.push_back(right[j++]);
	}	
	
	while(i < left.size())	{result.push_back(left[i++]);}
	while(j < right.size()) {result.push_back(right[j++]);}
	
	return result;
}
 
int main(int argc,char *argv[])
{
	MPI_Init(&argc,&argv);

	int size,rank;

	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	vector<int> data;
	int n;
	if(rank == 0)
	{	
		data = {2,9,20,35,40,1,5,3,8,31,21,22,19,12,6,7,15,32,10,12};
		n = data.size();
	
	}	
	
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);	

	int part = n/size;
	vector<int> sub_data(part);

	MPI_Scatter(data.data(),part,MPI_INT,sub_data.data(),part,MPI_INT,0,MPI_COMM_WORLD);

	//cout << "Rank " << rank << ": " << "\n" << "Original Data" << ": ";
	//for(auto s : sub_data)
	//	cout << s << " ";
	//cout << "\n";

	quicksort(sub_data,0,part-1); 
	
	//cout << "Sorted Data" << ": ";
	//for(auto s : sub_data)
	//	cout << s << " ";
	//cout << "\n";
	
	vector<int> gathered;
	if(rank == 0) gathered.resize(n);

	MPI_Gather(sub_data.data(),part,MPI_INT,gathered.data(),part,MPI_INT,0,MPI_COMM_WORLD);

	//if(rank == 0)
	//	for(int g : gathered)
	//		cout << g << " ";

	if(rank == 0)
	{
	  vector<int> result;
	  for(int i = 0 ; i < n ; i+= part)
	  {
	      vector<int> next(gathered.begin()+i,gathered.begin()+i+part);
	      result = merge(result,next);
	  }
          cout << "Sorted Array" << ": ";
          for(int r : result)
		cout << r << " ";
          cout << "\n";
	}	
	MPI_Finalize();
	return 0;
}