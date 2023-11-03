#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#include "../../common.h"

// #include "../../linear/lin_naive.h" // TODO: change to cuBlas when implemented

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

void cpu_lin_naive(mat &A, mat &B, mat &C) {
	size_t A_row = A.size();
	size_t A_column = A[0].size();
	size_t B_row = B.size();
	size_t B_column = B[0].size();
	size_t C_row = C.size();
	size_t C_column = C[0].size();
	
	

	for (size_t i = 0; i < A_row; ++i) {
		for (size_t j = 0; j < B_column; ++j) {
			float tmp = 0.0;
			for (size_t k = 0; k < A_column; ++k) {
				tmp += A[i][k] * B[k][j];
			}
			C[i][j] = tmp;
		}
	}
}
int main(int argc, char *argv[]) {

    CALI_CXX_MARK_FUNCTION;

    int n;
    if (argc == 2) {
        n = atoi(argv[1]);
    }
    else
    {
        printf("\n Please provide the size of the matrix");
        return 0;
    }

    int numtasks,                         /* number of tasks in partition */
        taskid,                           /* a task identifier */
        numworkers,                       /* number of worker tasks */
        source,                           /* task id of message source */
        dest,                             /* task id of message destination */
        mtype,                            /* message type */
        rows,                             /* rows of matrix A sent to each worker */
        averow, extra, offset,            /* used to determine rows sent to each worker */
        i, j, k, rc;                      /* misc */
    mat a(n, std::vector<int>(n));                                /* matrix A to be multiplied */
    mat b(n, std::vector<int>(n));                                /* matrix B to be multiplied */
    mat c(n, std::vector<int>(n));                                /* result matrix C */
    MPI_Status status;

    double worker_receive_time,       /* Buffer for worker recieve times */
        worker_calculation_time,      /* Buffer for worker calculation times */
        worker_send_time = 0;         /* Buffer for worker send times */
    double whole_computation_time,    /* Buffer for whole computation time */
        master_initialization_time,   /* Buffer for master initialization time */
        master_send_receive_time = 0; /* Buffer for master send and receive time */
    /* Define Caliper region names */
    const char *comm = "comm";
    const char *comm_large = "comm_large";
    const char *comm_small = "comm_small";
    const char *comp = "comp";
    const char *comp_large = "comp_large";
    const char *comp_small = "comp_small";
    const char *data_init = "data_init";
    const char *correctness = "correctness";

    const char *MPI_send = "MPI_send";
    const char *MPI_recv = "MPI_recv";

    const char *whole_computation = "whole_computation";
    const char *master_initialization = "master_initialization";
    const char *master_send_recieve = "master_send_recieve";
    const char *worker_recieve = "worker_recieve";
    const char *worker_calculation = "worker_calculation";
    const char *worker_send = "worker_send";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    MPI_Comm MPI_COMM_WORKER;
    MPI_Comm_split(MPI_COMM_WORLD, (taskid == MASTER) ? MPI_UNDEFINED : 1, taskid, &MPI_COMM_WORKER);

    if (numtasks < 2)
    {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks - 1;

    // WHOLE PROGRAM COMPUTATION PART STARTS HERE
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    CALI_MARK_BEGIN(whole_computation);
    double whole_computation_start = MPI_Wtime();

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // master
    if(taskid == MASTER) {

        CALI_MARK_BEGIN(data_init);
        CALI_MARK_BEGIN(master_initialization); // Don't time printf
        double master_initialization_start = MPI_Wtime();

        get_random_matrix(a);
        get_random_matrix(b);

        CALI_MARK_END(master_initialization);
        CALI_MARK_END(data_init);
        double master_initialization_end = MPI_Wtime();
        master_initialization_time = master_initialization_end - master_initialization_start;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(master_send_recieve);
        double master_send_receive_start = MPI_Wtime();

        // send matrix data to the worker tasks
        averow = n / numworkers;
        extra = n % numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        CALI_MARK_BEGIN(comm_large)
        CALI_MARK_BEGIN(MPI_send);
        for(dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows*n, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b, n*n, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        CALI_MARK_END(MPI_send);
        CALI_MARK_END(comm_large);

        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(MPI_recv);
        // receive results from worker tasks
        mtype = FROM_WORKER;
        for(i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*n, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        }
        CALI_MARK_END(MPI_recv);
        CALI_MARK_END(comm_large);

        CALI_MARK_END(master_send_recieve);
        CALI_MARK_END(comm);
        double master_send_receive_end = MPI_Wtime();
        master_send_receive_time = master_send_receive_end - master_send_receive_start;
    }

    if(taskid > MASTER) {
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_recieve);
        double worker_recieve_start = MPI_Wtime();

        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows*n, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, n*n, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        CALI_MARK_END(worker_recieve);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);

        double worker_recieve_end = MPI_Wtime();
        worker_receive_time = worker_recieve_end - worker_recieve_start;

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(worker_calculation);
        double worker_calculation_start = MPI_Wtime();

        // do the work
        for(k = 0; k < n; k++) {
            for(i = 0; i < rows; i++) {
                c[i][k] = 0;
                for(j = 0; j < n; j++) {
                    c[i][k] += a[i][j] * b[j][k];
                }
            }
        }
        CALI_MARK_END(worker_calculation);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);
        double worker_calculation_end = MPI_Wtime();
        worker_calculation_time = worker_calculation_end - worker_calculation_start;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        double worker_send_start = MPI_Wtime();

        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c, rows*n, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);

        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
        double worker_send_end = MPI_Wtime();
        worker_send_time = worker_send_end - worker_send_start;
    }

    CALI_MARK_END(whole_computation);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    double whole_computation_end = MPI_Wtime();
    whole_computation_time = whole_computation_end - whole_computation_start;

    // verify
    if(taskid == MASTER) {
        mat c2(n, std::vector<int>(n));
        cpu_lin_naive(a, b, c2); // TODO: change to cuBlas when implemented
        CALI_MARK_BEGIN(correctness);
        bool correct = verify(c, c2);
        CALI_MARK_END(correctness);
        if(correct) {
            printf("Verification successful\n");
        }
        else {
            printf("Verification failed\n");
        }
    }

    adiak::init(NULL);
    adiak::launchdate();                                         // launch date of the job
    adiak::libraries();                                          // Libraries used
    adiak::cmdline();                                            // Command line used to launch the job
    adiak::clustername();                                        // Name of the cluster
    adiak::value("Algorithm", "Naive Matrix Multiplication");                        // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n);                        // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks);                        // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads);                    // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks);                      // The number of CUDA blocks
    // adiak::value("group_num", group_number);                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    double worker_receive_time_max,
        worker_receive_time_min,
        worker_receive_time_sum,
        worker_recieve_time_average,
        worker_calculation_time_max,
        worker_calculation_time_min,
        worker_calculation_time_sum,
        worker_calculation_time_average,
        worker_send_time_max,
        worker_send_time_min,
        worker_send_time_sum,
        worker_send_time_average = 0; // Worker statistic values.

    /* USE MPI_Reduce here to calculate the minimum, maximum and the average times for the worker processes.
    MPI_Reduce (&sendbuf,&recvbuf,count,datatype,op,root,comm). https://hpc-tutorials.llnl.gov/mpi/collective_communication_routines/ */
    if (taskid > MASTER)
    {
        MPI_Reduce(&worker_receive_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORKER);
        MPI_Reduce(&worker_receive_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, MPI_COMM_WORKER);
        MPI_Reduce(&worker_receive_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORKER);
        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORKER);
        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, MPI_COMM_WORKER);
        MPI_Reduce(&worker_calculation_time, &worker_calculation_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORKER);
        MPI_Reduce(&worker_send_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORKER);
        MPI_Reduce(&worker_send_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, MPI_COMM_WORKER);
        MPI_Reduce(&worker_send_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORKER);
    }

    if (taskid == 0)
    {
        // Master Times
        printf("******************************************************\n");
        printf("Master Times:\n");
        printf("Whole Computation Time: %f \n", whole_computation_time);
        printf("Master Initialization Time: %f \n", master_initialization_time);
        printf("Master Send and Receive Time: %f \n", master_send_receive_time);
        printf("\n******************************************************\n");

        // Add values to Adiak
        adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);
        adiak::value("MPI_Reduce-master_initialization_time", master_initialization_time);
        adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);

        // Must move values to master for adiak
        mtype = FROM_WORKER;
        MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_recieve_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&worker_send_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

        adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
        adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
        adiak::value("MPI_Reduce-worker_recieve_time_average", worker_recieve_time_average);
        adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
        adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
        adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
        adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
        adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
        adiak::value("MPI_Reduce-worker_send_time_average", worker_send_time_average);
    }
    else if (taskid == 1)
    { // Print only from the first worker.
        // Print out worker time results.

        // Compute averages after MPI_Reduce
        worker_recieve_time_average = worker_receive_time_sum / (double)numworkers;
        worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
        worker_send_time_average = worker_send_time_sum / (double)numworkers;

        printf("******************************************************\n");
        printf("Worker Times:\n");
        printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
        printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
        printf("Worker Receive Time Average: %f \n", worker_recieve_time_average);
        printf("Worker Calculation Time Max: %f \n", worker_calculation_time_max);
        printf("Worker Calculation Time Min: %f \n", worker_calculation_time_min);
        printf("Worker Calculation Time Average: %f \n", worker_calculation_time_average);
        printf("Worker Send Time Max: %f \n", worker_send_time_max);
        printf("Worker Send Time Min: %f \n", worker_send_time_min);
        printf("Worker Send Time Average: %f \n", worker_send_time_average);
        printf("\n******************************************************\n");

        mtype = FROM_WORKER;
        MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&worker_recieve_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&worker_calculation_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&worker_send_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}