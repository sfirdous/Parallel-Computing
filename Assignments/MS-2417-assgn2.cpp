/**Parallel Program to perform numerical integration using Simpson's 3/8 rule **/

#include<mpi.h>
#include<iostream>
#include<cmath>
#include<vector>

using namespace std;

double f(double x){
  return pow(x,2) * exp(-x);
}

int main(int argc,char *argv[])
{
  MPI_Init(&argc,&argv);
  MPI_Status status;
  int rank,size;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);                                  //rank of processor
  MPI_Comm_size(MPI_COMM_WORLD,&size);                                  //number of processor's
  
  vector<int> intervals = {12,21,42,81};                                // intervals
  
  double a = 0,b = 4;                                                   // lower and upper limits
  
  // perform Simpson's 3/8 rule for each interval 
  for(auto n : intervals){
    double h = (b-a) / n;                                               // Calculate step size
    int part = (n-1)/size,rem = (n-1)%size;

    if(rank == 0){
       
       double sum = f(a) + f(b);                                        // Initialise sum with the function value of end limits
       int start = rank * part + min(rank,rem) + 1;                     // Calculate start index for processor based on its rank
       int end = start + part + (rank < rem ? 0 : -1);                  // Calculate end index for processor based on its rank
      
       for(int i = start ; i <= end ; ++i){                             // Add function values to sum based on index
            if(!(i%3))
                sum += 2*(f(a + double(i)*h));
            else
              sum += 3*(f(a + double(i)*h));
       }
      
       for(int i =1; i < size ; ++i){                                   // Receive local sums from processors and add it to sum
          double local_sum = 0;
          MPI_Recv(&local_sum,1,MPI_DOUBLE,i,1,MPI_COMM_WORLD,&status);
          sum += local_sum;
       }
        
        double result = ((3*h) / 8) * sum;                             // Calculate the result
       cout << "Number of intervals (n): " << n << endl;
       cout << "Computed integral (Simpson's 3/8 rule): " << result << endl;

      }
    else{
       
       int start = rank * part + min(rank,rem) + 1;                   // Calculate start index for processor based on its rank
       int end = start + part + (rank < rem ? 0 : -1);                // Calculate end index for processor based on its rank
       
       double local_sum = 0;
       for(int i = start ; i <= end ; ++i){                           // Add function values to local_sum based on index
            if(!(i%3))
                local_sum += 2*(f(a + double(i)*h));
            else
              local_sum += 3*(f(a + double(i)*h));
       }
      
       MPI_Send(&local_sum,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);          // Send local_sum to processor 0
    }
  }
  MPI_Finalize();
  return 0;
}
