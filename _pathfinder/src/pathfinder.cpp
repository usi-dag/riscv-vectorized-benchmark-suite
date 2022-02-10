//#include <stdio.h>
//#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <climits>

using namespace std;

/*************************************************************************
* RISC-V Vectorized Version
* Author: Cristóbal Ramírez Lazo
* email: cristobal.ramirez@bsc.es
* Barcelona Supercomputing Center (2020)
*************************************************************************/

#ifdef VECTORIZE
#include "../../common/vector_defines.h"
#endif
/************************************************************************/

//Enable RESULT_PRINT in order to see the result vector, for instruction count it should be disable
#define RESULT_PRINT
//#define INPUT_PRINT
#define MAXNAMESIZE 1024 // max filename length
#define M_SEED 9
#define MAX_WEIGHT 10
#define NUM_RUNS 100

int rows, cols;
int* wall;
int* result;
string inputfilename;
string outfilename;
//#include "timer.h"

void init(int argc, char** argv);
void run();
void run_vector();
void output_printfile(int *dst, string& filename);
//void init_data(int *data, string& filename );
/*************************************************************************
*GET_TIME
*returns a long int representing the time
*************************************************************************/
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
        return (float) (end_time - start_time) / (1000 * 1000);
}
/*************************************************************************/

void init(int argc, char** argv)
{
    if(argc!=4){
        printf("Usage: pathfiner width num_of_steps input_file output_file\n");
        exit(0);
    }

    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    //inputfilename = argv[3];
    outfilename = argv[3];
    //}else{
    //            printf("Usage: pathfiner width num_of_steps input_file output_file\n");
    //            exit(0);
    //    }
    printf("cols: %d rows: %d \n", cols , rows);

    wall = new int[rows * cols];
    result = new int[cols];
    
    //int seed = M_SEED;
    //srand(seed);
    /*
    init_data(wall, inputfilename );
    */
    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i*cols+j] = rand() % 10;
        }
    }
    
    //for (int j = 0; j < cols; j++)
    //    result[j] = wall[j];

#ifdef INPUT_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i*cols+j]) ;
        }
        printf("\n") ;
    }
#endif
}

void fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int main(int argc, char** argv)
{

// if(argc != 1 ) {
//         cout << "Usage: " << argv[0] << " <input_file> " << endl;
//         exit(1);
//     }   
long long start_0 = get_time();
init(argc,argv);
long long end_0 = get_time();
printf("TIME TO INIT DATA: %f\n", elapsed_time(start_0, end_0));
    //unsigned long long cycles;

#ifndef VECTORIZE
    run();
#else
    run_vector();
#endif

return EXIT_SUCCESS;
}

#ifndef VECTORIZE

void run()
{
    int min;
    int *src,*dst, *temp;
    
    //src = (int *)malloc(sizeof(int)*cols);
    
    printf("NUMBER OF RUNS: %d\n",NUM_RUNS);
    long long start = get_time();

    for (int j=0; j<NUM_RUNS; j++) {
        src = new int[cols];
        for (int x = 0; x < cols; x++){
            result[x] = wall[x];
        }

        dst = result;
        for (int t = 0; t < rows-1; t++) {
            temp = src;
            src = dst;
            dst = temp;
            //#pragma omp parallel for private(min)
            for(int n = 0; n < cols; n++){
              min = src[n];
              if (n > 0)
                min = MIN(min, src[n-1]);
              if (n < cols-1)
                min = MIN(min, src[n+1]);
              dst[n] = wall[(t+1)*cols + n]+min;
            }
        }   
        //delete src;
    }

    long long end = get_time();
    printf("TIME TO FIND THE SMALLEST PATH: %f\n", elapsed_time(start, end));

#ifdef RESULT_PRINT

//    output_printfile(dst, outfilename );

#endif  // RESULT_PRINT
    free(dst);
    free(wall);
    free(src);
}

#else // VECTORIZE

void run_vector()
{
    int *src,*dst, *temp;

    long long start = get_time();
    printf("NUMBER OF RUNS: %d\n",NUM_RUNS);



    int limit = loop_bound(INT32_SPECIES_512, cols-1);

    for (int j=0; j<NUM_RUNS; j++) {
        src = new int[cols];
        for (int x = 0; x < cols; x++){
            result[x] = wall[x];
        }
        dst = result;


        _MMR_i32    xSrc_slideup;
        _MMR_i32    xSrc_slidedown;
        _MMR_i32    xSrc;
        _MMR_i32    xNextrow; 


        for (int t = 0; t < rows-1; t++) 
        {
            int n;

            temp = src;
            src = dst;
            dst = temp;

            for(n = 0; n < limit; n = n + INT32_SPECIES_512)
            {
                xNextrow = _MM_LOAD_i32(&dst[n]);
                xSrc = xNextrow;
                xSrc_slideup = _MM_LOAD_i32(&dst[n+1]);
                if (n > 0) {
                    xSrc_slidedown = _MM_LOAD_i32(&dst[n-1]);
                } else {
                    xSrc_slidedown = _MM_LOAD_i32(&dst[n]);
                    xSrc_slidedown = _MM_LSHIFT_i32(xSrc_slidedown, 1);
                    xSrc_slidedown[0] = INT_MAX;

                }

                xSrc = _MM_MIN_i32(xSrc,xSrc_slideup);
                xSrc = _MM_MIN_i32(xSrc,xSrc_slidedown);
                xNextrow = _MM_LOAD_i32(&wall[(t+1)*cols + n]);
                xNextrow = _MM_ADD_i32(xNextrow,xSrc);
                _MM_STORE_i32(&dst[n],xNextrow);
//                FENCE();
            }

            for (; n < cols; n++) {
                int min = src[n];
                if (n > 0)
                    min = MIN(min, src[n-1]);
                if (n < cols-1)
                    min = MIN(min, src[n+1]);

                dst[n] = wall[(t+1)*cols + n]+min;
            }
        }

//        FENCE();
    }
    long long end = get_time();
    printf("TIME TO FIND THE SMALLEST PATH: %f\n", elapsed_time(start, end));

#ifdef RESULT_PRINT

//    output_printfile(dst, outfilename );

#endif // RESULT_PRINT

    free(wall);
    free(dst);
    free(src);
}
#endif // VECTORIZE

/*
void init_data(int *data,  string&  inputfile ) {

    ifstream fp (inputfile.c_str());
    assert(fp.is_open()); // were there any errors on opening?

    int value;
    int i=0;
    while (!fp.eof()){
        fp >> data[i];
        i++;
    }

    if( !(i-1 == cols*rows) ) {
        printf("error: the number of elements does not match the parameters i=%d , colsxrows = %d \n",i,cols*rows);
        exit(1);
    }
    fp.close();
}
*/
void output_printfile(int *dst,  string& outfile ) {
    ofstream myfile;
    myfile.open(outfile);
    assert(myfile.is_open());

    for (int j = 0; j < cols; j++)
    {
        myfile << dst[j] <<" " ;
    }
    myfile.close();
}