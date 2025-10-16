/**Parallel Program to perform Quick Sort on n numbers using p processor's**/

#include <iostream>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
using namespace std;

int read_from_file(string filename,vector<int> &arr)
{
	ifstream infile(filename);
	if(!infile.is_open()){
		cerr << "Error opening file!";
		return 1;
	}

	int num;
	while(infile >> num){
		arr.push_back(num);
	}
	
	infile.close();
	return 0;
}

int write_to_file(string filename,vector<int> &arr)
{
	ofstream outfile(filename);
	if(!outfile){
		cerr << "Error opening file" << endl;
		return 1;
	}
	
	for(int a : arr){
		outfile << a << endl;
	}
	outfile.close();
	cout << "Sorted numbers written in sorted_numbers.txt\n";
	return 0;
}

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
	
	int sendcounts[size],displs[size];	// vector to store # of elements to send each processor and displacement vector
	
	vector<int> data;
	int n;
	if(rank == 0)
	{	
		if(read_from_file("input_numbers.txt",data))
			MPI_Abort(MPI_COMM_WORLD,1);
		n = data.size();
	}
	
	
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);	
	int part = n/size;
	// populate sendcounts and displs vector
	if(rank == 0){
		int rem = n%size;
		for(int i = 0; i < size ; ++i)
			sendcounts[i] = part + (i < rem ? 1 : 0),displs[i] = i * part + min(i,rem);

	}
	
	// Broadcast send counts and displacements to all processors
	MPI_Bcast(sendcounts,size,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(displs,size,MPI_INT,0,MPI_COMM_WORLD);
	
	vector<int> sub_data(sendcounts[rank]);

	MPI_Scatterv(data.data(),sendcounts,displs,MPI_INT,sub_data.data(),sendcounts[rank],MPI_INT,0,MPI_COMM_WORLD);

	quicksort(sub_data,0,sendcounts[rank]-1);
	
	vector<int> gathered;
	if(rank == 0) gathered.resize(n);

	MPI_Gatherv(sub_data.data(),sendcounts[rank],MPI_INT,gathered.data(),sendcounts,displs,MPI_INT,0,MPI_COMM_WORLD);


	if(rank == 0)
	{
	  vector<int> result;
	  for(int i = 0 ; i < size ; ++i)
	  {
	      vector<int> next(gathered.begin()+displs[i],gathered.begin()+displs[i]+sendcounts[i]);
	      result = merge(result,next);
	  }
          write_to_file("sorted_numbers.txt",result);
	}	
	MPI_Finalize();
	return 0;
}