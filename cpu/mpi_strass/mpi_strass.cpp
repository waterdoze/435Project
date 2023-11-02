#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#include "../../common.h"

#include "../../linear/lin_naive.h" // TODO: change to cuBlas when implemented

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

mat strassen(int n, mat m1, mat m2);
void strassen(int n, mat m1, mat m2, mat &m3, int taskid);

/* Define Caliper region names */
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_small";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";
const char *data_init = "data_init";
const char *correctness = "correctness";

const char *strassen_whole_computation = "strassen_whole_computation";
const char *bcast_n = "bcast_n";
const char *bcase_matricies = "bcast_matrix";
const char *split = "split";
const char *addsub = "addsub";
const char *combine = "combine";
const char *strassen = "strassen";
const char *worker_send = "worker_send";
const char *master_receive = "master_receive";
const char *master_combine = "master_combine";
const char *MPI_Barrier = "MPI_Barrier";


int main(int argc, char* argv[]) {

    CALI_CXX_MARK_FUNCTION;
    int taskid, numtasks;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    int n;
    if (argc != 2) {
        std::cout << "include matrix size" << std::endl;
        return 1;
    }
    n = atoi(argv[1]);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(MPI_Barrier)
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPI_Barrier)
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(bcast_n);
    MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END(bcast_n);
    CALI_MARK_END(comm);
    mat a(n, std::vector<int>(n));
    mat b(n, std::vector<int>(n));
    mat c(n, std::vector<int>(n));

    if(taskid == MASTER) {
        CALI_MARK_BEGIN(data_init);
        get_random_matrix(a);
        get_random_matrix(b);
        CALI_MARK_END(data_init);
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(MPI_Barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPI_Barrier);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(bcast_matricies);
    MPI_Bcast(&a[0][0], n*n, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&b[0][0], n*n, MPI_INT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END(bcast_matricies);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    CALI_MARK_BEGIN(strassen_whole_computation);
    strassen(n, a, b, c, taskid);
    CALI_MARK_END(strassen_whole_computation);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(MPI_Barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPI_Barrier);
    CALI_MARK_END(comm);

    if(taskid == MASTER) {
        mat c2(n, std::vector<int>(n));
        cpu_lin_naive(a, b, c2);
        CALI_MARK_BEGIN(correctness);
        bool correct = verify(c, c2); // TODO: change to cuBlas when implemented
        CALI_MARK_END(correctness);
        if(correct) {
            printf("Verification Successful!\n")
        }
        else {
            printf("Verification Failed!\n")
        }
        // print_matrix(c);
    }

    adiak::init(NULL);
    adiak::launchdate();                                      // launch date of the job
    adiak::libraries();                                       // Libraries used
    adiak::cmdline();                                         // Command line used to launch the job
    adiak::clustername();                                     // Name of the cluster
    adiak::value("Algorithm", "Naive Matrix Multiplication"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                  // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n);                             // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads);                    // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks);                      // The number of CUDA blocks
    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    MPI_Finalize();

    return 0;
}

mat strassen(int n, mat m1, mat m2) {
    // mat m3(n, std::vector<int>(n));
    // strassen(n, m1, m2, m3, 0);
    // return m3;

    if(n <= 32) {
        return cpu_naive(m1, m2, n);
    }

    int split_size = n / 2;

    mat a = split(n, m1, 0, 0); // A11
    mat b = split(n, m1, 0, split_size); // A12
    mat c = split(n, m1, split_size, 0); // A21
    mat d = split(n, m1, split_size, split_size); // A22

    mat e = split(n, m2, 0, 0); // B11
    mat f = split(n, m2, 0, split_size); // B12
    mat g = split(n, m2, split_size, 0); // B21
    mat h = split(n, m2, split_size, split_size); // B22

    mat fh_sub = addsub_matricies(split_size, f, h, false);
    mat s1 = strassen(split_size, a, fh_sub); // A11 * (B12 - B22)

    mat ab_add = addsub_matricies(split_size, a, b, true);
    mat s2 = strassen(split_size, ab_add, h); // (A11 + A12) * B22

    mat cd_add = addsub_matricies(split_size, c, d, true);
    mat s3 = strassen(split_size, cd_add, e); // (A21 + A22) * B11

    mat ge_sub = addsub_matricies(split_size, g, e, false);
    mat s4 = strassen(split_size, d, ge_sub); // A22 * (B21 - B11)

    mat ad_add = addsub_matricies(split_size, a, d, true);
    mat eh_add = addsub_matricies(split_size, e, h, true);
    mat s5 = strassen(split_size, ad_add, eh_add); // (A11 + A22) * (B11 + B22)

    mat bd_sub = addsub_matricies(split_size, b, d, false);
    mat gh_add = addsub_matricies(split_size, g, h, true);
    mat s6 = strassen(split_size, bd_sub, gh_add); // (A12 - A22) * (B21 + B22)

    mat ac_sub = addsub_matricies(split_size, a, c, false);
    mat ef_add = addsub_matricies(split_size, e, f, true);
    mat s7 = strassen(split_size, ac_sub, ef_add); // (A11 - A21) * (B11 + B12)

    mat s5_s4_add = addsub_matricies(split_size, s5, s4, true); // S5 + S4
    mat s5_s4_s2_sub = addsub_matricies(split_size, s5_s4_add, s2, false); // S5 + S4 - S2
    mat c11 = addsub_matricies(split_size, s5_s4_s2_sub, s6, true); // P1 = S5 + S4 - S2 + S6

    mat c12 = addsub_matricies(split_size, s1, s2, true); // S1 + S2

    mat c21 = addsub_matricies(split_size, s3, s4, true); // S3 + S4

    mat s5_s1_add = addsub_matricies(split_size, s5, s1, true); // S5 + S1
    mat s5_s1_s3_sub = addsub_matricies(split_size, s5_s1_add, s3, false); // S5 + S1 - S3
    mat c22 = addsub_matricies(split_size, s5_s1_s3_sub, s7, false); // P7 = S5 + S1 - S3 - S7

    return combine_matricies(split_size, c11, c12, c21, c22);
}

void strassen(int n, mat m1, mat m2, mat& m3, int taskid) {

    if (n <= 32) {
        return cpu_naive(A, B, n); // TODO: change to cuBlas when implemented
    }

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(split);
    int split_size = n / 2;

    mat a = split(n, A, 0, 0); // A11
    mat b = split(n, A, 0, split_size); // A12
    mat c = split(n, A, split_size, 0); // A21
    mat d = split(n, A, split_size, split_size); // A22

    mat e = split(n, B, 0, 0); // B11
    mat f = split(n, B, 0, split_size); // B12
    mat g = split(n, B, split_size, 0); // B21
    mat h = split(n, B, split_size, split_size); // B22
    CALI_MARK_END(split);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    mat s1(split_size, std::vector<int>(split_size));
    mat s2(split_size, std::vector<int>(split_size));
    mat s3(split_size, std::vector<int>(split_size));
    mat s4(split_size, std::vector<int>(split_size));
    mat s5(split_size, std::vector<int>(split_size));
    mat s6(split_size, std::vector<int>(split_size));
    mat s7(split_size, std::vector<int>(split_size));

    if(taskid == 0) {
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(master_receive);
        MPI_Recv(&(s1[0][0]), split_size*split_size, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s2[0][0]), split_size*split_size, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s3[0][0]), split_size*split_size, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s4[0][0]), split_size*split_size, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s5[0][0]), split_size*split_size, MPI_INT, 5, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s6[0][0]), split_size*split_size, MPI_INT, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s7[0][0]), split_size*split_size, MPI_INT, 7, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END(master_receive);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    }
    if(taskid == 1) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat fh_sub = addsub_matricies(split_size, f, h, false);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassen);
        s1 = strassen(split_size, a, fh_sub);
        CALI_MARK_END(strassen);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s1[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }
    if(taskid == 2) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat ab_add = addsub_matricies(split_size, a, b, true);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassen);
        s2 = strassen(split_size, ab_add, h);
        CALI_MARK_END(strassen);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s2[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }
    if(taskid == 3) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat cd_add = addsub_matricies(split_size, c, d, true);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassen);
        s3 = strassen(split_size, cd_add, e);
        CALI_MARK_END(strassen);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s3[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }
    if(taskid == 4) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat ge_sub = addsub_matricies(split_size, g, e, false);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassen);
        s4 = strassen(split_size, d, ge_sub);
        CALI_MARK_END(strassen);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s4[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }
    if(taskid == 5) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat ad_add = addsub_matricies(split_size, a, d, true);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat eh_add = addsub_matricies(split_size, e, h, true);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassen);
        s5 = strassen(split_size, ad_add, eh_add);
        CALI_MARK_END(strassen);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s5[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }
    if(taskid == 6) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat bd_sub = addsub_matricies(split_size, b, d, false);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat eg_add = addsub_matricies(split_size, e, g, true);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassen);
        s6 = strassen(split_size, bd_sub, eg_add);
        CALI_MARK_END(strassen);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s6[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }
    if(taskid == 7) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat ac_sub = addsub_matricies(split_size, a, c, false);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        mat eg_add = addsub_matricies(split_size, e, g, true);
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassen);
        s7 = strassen(split_size, ac_sub, eg_add);
        CALI_MARK_END(strassen);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s7[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(MPI_Barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPI_Barrier);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    if(taskid == 0) {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        CALI_MARK_BEGIN(combine);
        mat s5_s4_add = addsub_matricies(split_size, s5, s4, true); // S5 + S4
        mat s5_s4_s2_sub = addsub_matricies(split_size, s5_s4_add, s2, false); // S5 + S4 - S2
        mat c11 = addsub_matricies(split_size, s5_s4_s2_sub, s6, true); // P1 = S5 + S4 - S2 + S6

        mat c12 = addsub_matricies(split_size, s1, s2, true); // S1 + S2

        mat c21 = addsub_matricies(split_size, s3, s4, true); // S3 + S4

        mat s5_s1_add = addsub_matricies(split_size, s5, s1, true); // S5 + S1
        mat s5_s1_s3_sub = addsub_matricies(split_size, s5_s1_add, s3, false); // S5 + S1 - S3
        mat c22 = addsub_matricies(split_size, s5_s1_s3_sub, s7, false); // P7 = S5 + S1 - S3 - S7

        m3 = combine_matricies(split_size, c11, c12, c21, c22);
        CALI_MARK_END(combine);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
    }
}