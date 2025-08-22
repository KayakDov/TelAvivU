#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>

#include "deviceArrays.h"
#include "testMethods.cu"
#include "algorithms.cu"

using namespace std;

// Function prototypes
void showHelp();
void processCommandLineArgs(int argc, char const* argv[], string& a_file, int& numDiags, int& width, string& diags_file, string& b_file, string& x_dest_file);
void readInputData(const string& a_file, int numDiags, int width, CuArray2D<float>& A, const string& diags_file, vector<int>& diags_vec, const string& b_file, CuArray1D<float>& b);
void solveAndWriteOutput(CuArray2D<float>& A, const vector<int>& diags_vec, CuArray1D<float>& b, const string& x_dest_file);

/**
 * @file main.cpp
 * @brief This program solves a linear system Ax = b using the unpreconditioned
 * BiCGSTAB method on a CUDA device.
 *
 * It takes the matrix A (stored as packed diagonals), a vector b, and diagonal
 * indices as input from files specified on the command line. The solution x
 * is then written to an output file.
 */
int main(int argc, char const* argv[]) {
    if (argc > 1 && string(argv[1]) == "-h") {
        showHelp();
        return 0;
    }

    if (argc != 7) {
        cerr << "Error: Incorrect number of arguments.\n";
        showHelp();
        return 1;
    }

    checkForDevice();

    try {
        string a_file, diags_file, b_file, x_dest_file;
        int numDiags, width;

        processCommandLineArgs(argc, argv, a_file, numDiags, width, diags_file, b_file, x_dest_file);

        CuArray2D<float> A(numDiags, width);
        CuArray1D<float> b(width);
        vector<int> diags_vec(numDiags);

        readInputData(a_file, numDiags, width, A, diags_file, diags_vec, b_file, b);
        solveAndWriteOutput(A, diags_vec, b, x_dest_file);

    } catch (const exception& e) {
        cerr << "An error occurred: " << e.what() << endl;
        return 1;
    }

    return 0;
}

// Helper function to display the help message.
/**
 * @brief Displays the command-line usage and help information.
 *
 * This function is called when the `-h` flag is passed as a command-line argument.
 * It provides a detailed explanation of the required parameters and their constraints.
 */
void showHelp() {
    cout << "Usage: ./unpreconditionedBiCGSTAB [options] <A_file> <num_diags> <matrix_width> <diags_file> <b_file> <x_dest_file>\n";
    cout << "Options:\n";
    cout << "  -h  Show this help message.\n\n";
    cout << "Arguments:\n";
    cout << "  <A_file>        (1) Path to a file containing the diagonals of the sparse matrix A.\n";
    cout << "  <num_diags>     (2) The number of diagonals, which is the height of the matrix A.\n";
    cout << "  <matrix_width>  (3) The width of the original sparse matrix (and the height of b).\n";
    cout << "  <diags_file>    (4) Path to a file containing the indices of the diagonals (space-separated integers).\n";
    cout << "  <b_file>        (5) Path to a file containing the right-hand side vector b.\n";
    cout << "  <x_dest_file>   (6) Path to a destination file for the solution vector x.\n\n";
    cout << "Constraints:\n";
    cout << "  - The files for A, b, and x should contain space-separated floating-point numbers.\n";
    cout << "  - The file for diagonals should contain space-separated integers.\n";
    cout << "  - The dimensions (num_diags, matrix_width) must be positive integers.\n";
    cout << "  - The number of diagonals in diags_file must match num_diags.\n";
}

/**
 * @brief Processes the command-line arguments and validates them.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @param a_file Reference to a string to store the path to the matrix A file.
 * @param numDiags Reference to an int to store the number of diagonals.
 * @param width Reference to an int to store the width of the original matrix.
 * @param diags_file Reference to a string to store the path to the diagonals file.
 * @param b_file Reference to a string to store the path to the vector b file.
 * @param x_dest_file Reference to a string to store the path to the solution x file.
 * @throws std::runtime_error if numDiags or width are not positive.
 */
void processCommandLineArgs(int argc, char const* argv[], string& a_file, int& numDiags, int& width, string& diags_file, string& b_file, string& x_dest_file) {
    a_file = argv[1];
    numDiags = stoi(argv[2]);
    width = stoi(argv[3]);
    diags_file = argv[4];
    b_file = argv[5];
    x_dest_file = argv[6];

    if (numDiags <= 0 || width <= 0) {
        throw runtime_error("Number of diagonals and matrix width must be positive.");
    }
}

/**
 * @brief Reads input data from the specified files into CuArray objects.
 *
 * @param a_file The path to the matrix A file.
 * @param numDiags The number of diagonals (height of matrix A).
 * @param width The width of the matrix A and vector b.
 * @param A A reference to the CuArray2D to store matrix A.
 * @param diags_file The path to the diagonal indices file.
 * @param diags_vec A reference to a vector<int> to store the diagonal indices.
 * @param b_file The path to the vector b file.
 * @param b A reference to the CuArray1D to store vector b.
 * @throws std::runtime_error if any file cannot be opened or data is invalid.
 */
void readInputData(const string& a_file, int numDiags, int width, CuArray2D<float>& A, const string& diags_file, vector<int>& diags_vec, const string& b_file, CuArray1D<float>& b) {
    ifstream a_fs(a_file);
    if (!a_fs.is_open()) {
        throw runtime_error("Could not open file: " + a_file);
    }
    a_fs >> A;
    a_fs.close();

    ifstream diags_fs(diags_file);
    if (!diags_fs.is_open()) {
        throw runtime_error("Could not open file: " + diags_file);
    }
    for (int i = 0; i < numDiags; ++i) {
        if (!(diags_fs >> diags_vec[i])) {
            throw runtime_error("Error reading diagonal indices from file.");
        }
    }
    diags_fs.close();

    ifstream b_fs(b_file);
    if (!b_fs.is_open()) {
        throw runtime_error("Could not open file: " + b_file);
    }
    b_fs >> b;
    b_fs.close();
}

/**
 * @brief Solves the linear system Ax = b and writes the solution to a file.
 *
 * This function uses the `unpreconditionedBiCGSTAB` algorithm to find the
 * solution vector x and then saves it to the specified destination file.
 *
 * @param A A reference to the CuArray2D containing the matrix A.
 * @param diags_vec A constant reference to the vector of diagonal indices.
 * @param b A reference to the CuArray1D containing the vector b.
 * @param x_dest_file The path to the output file for the solution vector x.
 * @throws std::runtime_error if the output file cannot be opened.
 */
void solveAndWriteOutput(CuArray2D<float>& A, const vector<int>& diags_vec, CuArray1D<float>& b, const string& x_dest_file) {
    CuArray1D<float> x(b.size());
    unpreconditionedBiCGSTAB(A, diags_vec.data(), b, &x, 20, 1e-6f);

    ofstream x_fs(x_dest_file);
    if (!x_fs.is_open()) {
        throw runtime_error("Could not open destination file: " + x_dest_file);
    }
    x_fs << x;
    x_fs.close();
}