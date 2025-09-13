/**MPI Program for Parallel Distribution and Partial Summation**/

#include<mpi.h>
#include<iostream>
#include<vector>

using namespace std;

int main(int argc,char *argv[])
{
  MPI_Init(&argc,&argv);
  MPI_Status status;
  int rank,size;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); //rank of processor
  MPI_Comm_size(MPI_COMM_WORLD,&size); //number of processor's
  
  
  int n,part,rem;
  //operations performed by rank 0
  if(rank == 0){
    cout << "Enter n to perform their addition: ";
    cin >> n;
    part = n/size;
    rem = n%size;
  }
  MPI_Bcast(&part,1,MPI_INT,0,MPI_COMM_WORLD);
  
  if(rank == 0){
  vector<int> A(n);
  for(int i = 0 ; i < n ; ++i)
    A[i] = i+1;
  int sum = 0;
  
  for(int i = 0 ; i < part ; ++i )
    sum += A[i];
  
  //if rem > 0 processor 0 should calculate sum 0f part+1 elements
  if(rem > 0)
    sum += A[part];
    
  //cout << sum << endl;
  for(int i = 1 ; i < size ; ++i){
    int start = i * part + min(i, rem); // Calculate the starting index for the i-th process's segment in the vector. 
                                        // Each process gets 'part' elements, plus one extra element for each process with rank less than 'rem', for the uneven division remainder.

   int length = part + (i < rem ? 1 : 0); // Calculate the number of elements the i-th process will receive.
                                          // Processes with rank less than 'rem' receive one additional element (part + 1),
                                          // while the rest receive 'part' elements.

    MPI_Send(A.data()+start,length,MPI_INT,i,1,MPI_COMM_WORLD);
  }
  for(int i = 1; i < size ; ++i)
  {
    int indi_sum = 0;
    MPI_Recv(&indi_sum,1,MPI_INT,i,2,MPI_COMM_WORLD,&status);
    sum += indi_sum;
  }
  cout << "Total Sum : " << sum << endl;
  }
  else{
  
  if(rank < rem) 
    part = part+1;
    
  vector<int> sub(part);
  MPI_Recv(sub.data(),part,MPI_INT,0,1,MPI_COMM_WORLD,&status);
  int local_sum = 0;
  for(int i = 0 ; i < part ; ++i){
    local_sum += sub[i];
  }
  //cout << "Rank" << rank << ": " << local_sum << endl;
  MPI_Send(&local_sum,1,MPI_INT,0,2,MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}
