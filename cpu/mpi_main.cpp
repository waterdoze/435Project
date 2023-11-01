#include "mpi_naive.h"

int main(int argc, char* argv[]) {
    int n;
    if (argc == 2) {
        n = atoi(argv[1]);
    }
    else {
        printf("\n Please provide the size of the matrix");
        return 0;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);



    mpi_naive(n, taskid, numtasks);

    MPI_Finalize();
}