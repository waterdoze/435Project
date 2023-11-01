#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#include "../common.h"

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2


void mpi_naive(int n, int taskid, int numtasks) {

    int
        numworkers,                       /* number of worker tasks */
        source,                           /* task id of message source */
        dest,                             /* task id of message destination */
        mtype,                            /* message type */
        rows,                             /* rows of matrix A sent to each worker */
        averow, extra, offset,            /* used to determine rows sent to each worker */
        i, j, k, rc;                      /* misc */
    mat a;                                /* matrix A to be multiplied */
        b,                                /* matrix B to be multiplied */
        c;                                /* result matrix C */
    MPI_Status status;

    if (numtasks < 2)
    {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks - 1;

    // master
    if(taskid == MASTER) {
        get_random_matrix(a);
        get_random_matrix(b);


        // send matrix data to the worker tasks
        averow = n / numworkers;
        extra = n % numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        for(dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows*n, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b, n*n, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        // receive results from worker tasks
        mtype = FROM_WORKER;
        for(i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*n, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        }
    }

    if(taskid > MASTER) {
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows*n, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, n*n, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        // do the work
        for(k = 0; k < n; k++) {
            for(i = 0; i < rows; i++) {
                c[i][k] = 0;
                for(j = 0; j < n; j++) {
                    c[i][k] += a[i][j] * b[j][k];
                }
            }
        }

        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c, rows*n, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    }

}