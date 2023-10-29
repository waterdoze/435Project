# CSCE 435 Group project

## 1. Group members:
1. Kevin Dai
2. An Duong
3. Steven Mao
4. Matthew Wang

---

## 2. _due 10/25_ Project topic

    Running different matrix multiplication algorithms on CPU and GPU (CUDA).  

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

    Our main method of communication will be through Discord, text messages, and in-person meetings.
    
    We will be running 2 different matrix multiplication algorithms using parallel computing: the naive method and Strassen's, on CPU 
    and NVIDIA CUDA GPU. After implementing these algorithms, we will compare the runtimes for the naive method vs Strassen's as 
    well as the algorithm's runtimes on CPU VS GPU. Finally, we'll be using the cuBLAS library to compare our code's runtime to 
    what CUDA should theoretically achieve.
    

- Algorithm 1 (Normal Matrix Multiplication)

    -------Pseudo-------

    Square-Matrix-Multiply-Recursive(A, B)

        n = A.rows
        let C be a new (n x n) matrix

        if n == 1
            c(11) = a(11) * b(11)
        else partition A, B, and C
            C(11) = Square-Matrix-Multiply-Recursive(A(11), B(11))
                    + Square-Matrix-Multiply-Recursive(A(12), B(21))

            C(12) = Square-Matrix-Multiply-Recursive(A(11), B(12))
                    + Square-Matrix-Multiply-Recursive(A(12), B(22))

            C(21) = Square-Matrix-Multiply-Recursive(A(21), B(11))
                    + Square-Matrix-Multiply-Recursive(A(22), B(21))

            C(22) = Square-Matrix-Multiply-Recursive(A(21), B(12))
                    + Square-Matrix-Multiply-Recursive(A(22), B(22))

        return C

    --------------------

- Algorithm 2 (Strassen's Algorithm)

    -------Pseudo-------

    Square-Matrix-Multiply-Recursive(A, B)

        n = A.rows
        let C be a new (n x n) matrix

        if n == 1
            c(11) = a(11) * b(11)
        else make S and P matrices and partition C

            S(1) = B(12) - B(22)
            S(2) = A(11) + A(12)
            S(3) = A(21) + A(22)
            S(4) = B(21) - B(11)
            S(5) = A(11) + A(22)
            S(6) = B(11) + B(22)
            S(7) = A(12) - A(22)
            S(8) = B(21) + B(22)
            S(9) = A(11) - A(21)
            S(10) = B(11) + B(12)

            P(1) = Square-Matrix-Multiply-Recursive(A(11), S(1))
            P(2) = Square-Matrix-Multiply-Recursive(S(2), B(22))
            P(3) = Square-Matrix-Multiply-Recursive(S(3), B(11))
            P(4) = Square-Matrix-Multiply-Recursive(A(22), S(4))
            P(5) = Square-Matrix-Multiply-Recursive(S(5), S(6))
            P(6) = Square-Matrix-Multiply-Recursive(S(7), S(8))
            P(7) = Square-Matrix-Multiply-Recursive(S(9), S(10))

            C(11) = P(5) + P(4) - P(2) + P(6)
            C(12) = P(1) + P(2)
            C(21) = P(3) + P(4)
            C(22) = P(5) + P(1) - P(3) - P(7)

        return C

    --------------------
    
- Algorithm 3 (cuBLAS)

gack