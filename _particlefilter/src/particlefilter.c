/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP 
 */

/*************************************************************************
* RISC-V Vectorized Version
* Author: Cristóbal Ramírez Lazo
* email: cristobal.ramirez@bsc.es
* Barcelona Supercomputing Center (2020)
*************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#ifdef USE_RISCV_VECTOR
#include "../../common/vector_defines.h"


void print512_num(__m512d var)
{
    double val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %f %f %f %f %f %f %f %f \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}
#endif

#include "../../common/riscv_util.h"

//#include <omp.h>
#include <limits.h>

#define PI 3.1415926535897932
/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;

/*****************************
*GET_TIME
*returns a long int representing the time
*****************************/
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
    return (float) (end_time - start_time) / (1000 * 1000);
}

/**
* Takes in a double and returns an integer that approximates to that double
* @return if the mantissa < .5 => return value < input value; else return value > input value
*/
double roundDouble(double value) {
    int newValue = (int) (value);
    if (value - newValue < .5)
        return newValue;
    else
        return newValue + 1;
}

/**
* Set values of the 3D array to a newValue if that value is equal to the testValue
* @param testValue The value to be replaced
* @param newValue The value to replace testValue with
* @param array3D The image vector
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
*/
void setIf(int testValue, int newValue, int *array3D, int *dimX, int *dimY, int *dimZ) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue) {
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
                }
            }
        }
    }
}

/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double randu(int *seed, int index) {
    int num = A * seed[index] + C;
    seed[index] = num % M;
    return fabs(seed[index] / ((double) M));
}

#ifdef USE_RISCV_VECTOR
/*
inline _MMR_f64 randu_vector(int * seed, int index ,unsigned long int gvl)
{
    _MMR_i32    xseed = _MM_LOAD_i32(&seed[index],gvl);
    _MMR_i32    xA = _MM_SET_i32(A,gvl);
    _MMR_i32    xC = _MM_SET_i32(C,gvl);
    _MMR_i32    xM = _MM_SET_i32((int)M,gvl);
    
    xseed =  _MM_MUL_i32(xseed,xA,gvl);
    xseed =  _MM_ADD_i32(xseed,xC,gvl);
    xseed =  _MM_REM_i32(xseed,xM,gvl);

    _MM_STORE_i32(&seed[index],xseed,gvl);
    FENCE();
    _MMR_f64    xResult;
    xResult = _MM_DIV_f64(_MM_VFWCVT_f64_f32(_MM_VFCVT_F_X_f32(xseed,gvl),gvl),_MM_SET_f64((double)M,gvl),gvl);
    xResult = _MM_VFSGNJX_f64(xResult,xResult,gvl);
    return xResult;
}
*/
inline _MMR_f64 randu_vector(long int * seed, int index , double* result, int* num)
{
    /*
    Esta parte del codigo deberia ser en 32 bits, pero las instrucciones de conversion aún no están disponibles,
    moviendo todo a 64 bits el resultado cambia ya que no se desborda, y las variaciones son muchas.
    */
    //double result[256];
    //int num[256];
    for(int x = index; x < index+SPECIES_512; x++){
        num[x-index] = A*seed[x] + C;
        seed[x] = (int) num[x-index] % M;
        result[x-index] = fabs(seed[x]/((double) M));
    }
    _MMR_f64    xResult;
    xResult = _MM_LOAD_f64(&result[0]);
//    FENCE();
    return xResult;
}
#endif // USE_RISCV_VECTOR

/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double randn(int *seed, int index) {
    /*Box-Muller algorithm*/
    double u = randu(seed, index);
    double v = randu(seed, index);
    double cosine = cos(2 * PI * v);
    double rt = -2 * log(u);
    return sqrt(rt) * cosine;
}

#ifdef USE_RISCV_VECTOR
inline _MMR_f64 randn_vector(long int * seed, int index ,double* randu_vector_result,int* randu_vector_num){
    /*Box-Muller algorithm*/
    _MMR_f64    xU = randu_vector(seed,index,randu_vector_result,randu_vector_num);
    _MMR_f64    xV = randu_vector(seed,index,randu_vector_result,randu_vector_num);
    _MMR_f64    xCosine;
    _MMR_f64    xRt;
    
    xV = _MM_MUL_f64(_MM_SET_f64(PI*2.0),xV);

     xCosine =_MM_COS_f64(xV);
    //FENCE();
//    xU = _MM_LOG_f64(xU); // TODO Bug cast returns all NAN

    double val[8];
    memcpy(val, &xU, sizeof(double) * 8);
    for (int i = 0; i < 8; i++) {
        val[i] = log(val[i]);
    }
    xU = _MM_LOAD_f64(&val[0]);

    xRt =  _MM_MUL_f64(_MM_SET_f64(-2.0),xU);

    return _MM_MUL_f64(_MM_SQRT_f64(xRt),xCosine);
}
#endif // USE_RISCV_VECTOR

/**
* Sets values of 3D matrix using randomly generated numbers from a normal distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
void addNoise(int *array3D, int *dimX, int *dimY, int *dimZ, int *seed) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                array3D[x * *dimY * *dimZ + y * *dimZ + z] =
                        array3D[x * *dimY * *dimZ + y * *dimZ + z] + (int) (5 * randn(seed, 0));
            }
        }
    }
}

/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
void strelDisk(int *disk, int radius) {
    int diameter = radius * 2 - 1;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            double distance = sqrt(pow((double) (x - radius + 1), 2) + pow((double) (y - radius + 1), 2));
            if (distance < radius)
                disk[x * diameter + y] = 1;
        }
    }
}

/**
* Dilates the provided video
* @param matrix The video to be dilated
* @param posX The x location of the pixel to be dilated
* @param posY The y location of the pixel to be dilated
* @param poxZ The z location of the pixel to be dilated
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param error The error radius
*/
void dilate_matrix(int *matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
    int startX = posX - error;
    while (startX < 0)
        startX++;
    int startY = posY - error;
    while (startY < 0)
        startY++;
    int endX = posX + error;
    while (endX > dimX)
        endX--;
    int endY = posY + error;
    while (endY > dimY)
        endY--;
    int x, y;
    for (x = startX; x < endX; x++) {
        for (y = startY; y < endY; y++) {
            double distance = sqrt(pow((double) (x - posX), 2) + pow((double) (y - posY), 2));
            if (distance < error)
                matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
        }
    }
}

/**
* Dilates the target matrix using the radius as a guide
* @param matrix The reference matrix
* @param dimX The x dimension of the video
* @param dimY The y dimension of the video
* @param dimZ The z dimension of the video
* @param error The error radius to be dilated
* @param newMatrix The target matrix
*/
void imdilate_disk(int *matrix, int dimX, int dimY, int dimZ, int error, int *newMatrix) {
    int x, y, z;
    for (z = 0; z < dimZ; z++) {
        for (x = 0; x < dimX; x++) {
            for (y = 0; y < dimY; y++) {
                if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
                    dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                }
            }
        }
    }
}

/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
void getneighbors(int *se, int numOnes, double *neighbors, int radius) {
    int x, y;
    int neighY = 0;
    int center = radius - 1;
    int diameter = radius * 2 - 1;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (se[x * diameter + y]) {
                neighbors[neighY * 2] = (int) (y - center);
                neighbors[neighY * 2 + 1] = (int) (x - center);
                neighY++;
            }
        }
    }
}

/**
* The synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
* @param I The video itself
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames of the video
* @param seed The seed array used for number generation
*/
void videoSequence(int *I, int IszX, int IszY, int Nfr, int *seed) {
    int k;
    int max_size = IszX * IszY * Nfr;
    /*get object centers*/
    int x0 = (int) roundDouble(IszY / 2.0);
    int y0 = (int) roundDouble(IszX / 2.0);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

    /*move point*/
    int xk, yk, pos;
    for (k = 1; k < Nfr; k++) {
        xk = abs(x0 + (k - 1));
        yk = abs(y0 - 2 * (k - 1));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if (pos >= max_size)
            pos = 0;
        I[pos] = 1;
    }

    /*dilate matrix*/
    int *newMatrix = (int *) malloc(sizeof(int) * IszX * IszY * Nfr);
    imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for (x = 0; x < IszX; x++) {
        for (y = 0; y < IszY; y++) {
            for (k = 0; k < Nfr; k++) {
                I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
            }
        }
    }
    free(newMatrix);

    /*define background, add noise*/
    setIf(0, 100, I, &IszX, &IszY, &Nfr);
    setIf(1, 228, I, &IszX, &IszY, &Nfr);
    /*add noise*/
    addNoise(I, &IszX, &IszY, &Nfr, seed);
}

/**
* Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
double calcLikelihoodSum(int *I, int *ind, int numOnes) {
    double likelihoodSum = 0.0;
    int y;
    for (y = 0; y < numOnes; y++)
        likelihoodSum += (pow((I[ind[y]] - 100), 2) - pow((I[ind[y]] - 228), 2)) / 50.0;
    return likelihoodSum;
}

/**
* Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
* @note This function uses sequential search
* @param CDF The CDF
* @param lengthCDF The length of CDF
* @param value The value to be found
* @return The index of value in the CDF; if value is never found, returns the last index
*/
int findIndex(double *CDF, int lengthCDF, double value) {
    int index = -1;
    int x;

    // for(int a = 0; a < lengthCDF; a++)
    // {
    // printf("%f ",CDF[a]);
    // }
    // printf("\n");

    // printf("CDF[x] >= value ,%f >= %f \n",CDF[0],value);

    for (x = 0; x < lengthCDF; x++) {
        if (CDF[x] >= value) {
            index = x;
            break;
        }
    }
    if (index == -1) {
        return lengthCDF - 1;
    }
    return index;
}

/**
* Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
* @note This function uses binary search before switching to sequential search
* @param CDF The CDF
* @param beginIndex The index to start searching from
* @param endIndex The index to stop searching
* @param value The value to find
* @return The index of value in the CDF; if value is never found, returns the last index
* @warning Use at your own risk; not fully tested
*/
int findIndexBin(double *CDF, int beginIndex, int endIndex, double value) {
    if (endIndex < beginIndex)
        return -1;
    int middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
    /*check the value*/
    if (CDF[middleIndex] >= value) {
        /*check that it's good*/
        if (middleIndex == 0)
            return middleIndex;
        else if (CDF[middleIndex - 1] < value)
            return middleIndex;
        else if (CDF[middleIndex - 1] == value) {
            while (middleIndex > 0 && CDF[middleIndex - 1] == value)
                middleIndex--;
            return middleIndex;
        }
    }
    if (CDF[middleIndex] > value)
        return findIndexBin(CDF, beginIndex, middleIndex + 1, value);
    return findIndexBin(CDF, middleIndex - 1, endIndex, value);
}

/**
* The implementation of the particle filter using OpenMP for many frames
* @see http://openmp.org/wp/
* @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
* @param I The video to be run
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames
* @param seed The seed array used for random number generation
* @param Nparticles The number of particles to be used
*/
void particleFilter(int *I, int IszX, int IszY, int Nfr, int *seed, int Nparticles) {

    int max_size = IszX * IszY * Nfr;
    long long start = get_time();
    //original particle centroid
    double xe = roundDouble(IszY / 2.0);
    double ye = roundDouble(IszX / 2.0);

    //expected object locations, compared to center
    int radius = 5;
    int diameter = radius * 2 - 1;
    int *disk = (int *) malloc(diameter * diameter * sizeof(int));
    strelDisk(disk, radius);
    int countOnes = 0;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (disk[x * diameter + y] == 1)
                countOnes++;
        }
    }

    //printf("countOnes = %d \n",countOnes); // 69

    double *objxy = (double *) malloc(countOnes * 2 * sizeof(double));
    getneighbors(disk, countOnes, objxy, radius);

    long long get_neighbors = get_time();
    printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start, get_neighbors));
    //initial weights are all equal (1/Nparticles)
    double *weights = (double *) malloc(sizeof(double) * Nparticles);
    //#pragma omp parallel for shared(weights, Nparticles) private(x)
    for (x = 0; x < Nparticles; x++) {
        weights[x] = 1 / ((double) (Nparticles));
    }
    long long get_weights = get_time();
    printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors, get_weights));
    //initial likelihood to 0.0
    double *likelihood = (double *) malloc(sizeof(double) * Nparticles);
    double *arrayX = (double *) malloc(sizeof(double) * Nparticles);
    double *arrayY = (double *) malloc(sizeof(double) * Nparticles);
    double *xj = (double *) malloc(sizeof(double) * Nparticles);
    double *yj = (double *) malloc(sizeof(double) * Nparticles);
    double *CDF = (double *) malloc(sizeof(double) * Nparticles);
    double *u = (double *) malloc(sizeof(double) * Nparticles);
    int *ind = (int *) malloc(sizeof(int) * countOnes * Nparticles);
    //#pragma omp parallel for shared(arrayX, arrayY, xe, ye) private(x)
    for (x = 0; x < Nparticles; x++) {
        arrayX[x] = xe;
        arrayY[x] = ye;
    }
    int k;

    printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, get_time()));
    int indX, indY;
    for (k = 1; k < Nfr; k++) {
        long long set_arrays = get_time();
        //apply motion model
        //draws sample from motion model (random walk). The only prior information
        //is that the object moves 2x as fast as in the y direction
        //#pragma omp parallel for shared(arrayX, arrayY, Nparticles, seed) private(x)
        for (x = 0; x < Nparticles; x++) {
            arrayX[x] += 1 + 5 * randn(seed, x);
            arrayY[x] += -2 + 2 * randn(seed, x);
        }
        long long error = get_time();
        printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
        //particle filter likelihood
        //#pragma omp parallel for shared(likelihood, I, arrayX, arrayY, objxy, ind) private(x, y, indX, indY)
        for (x = 0; x < Nparticles; x++) {
            //compute the likelihood: remember our assumption is that you know
            // foreground and the background image intensity distribution.
            // Notice that we consider here a likelihood ratio, instead of
            // p(z|x). It is possible in this case. why? a hometask for you.        
            //calc ind
            for (y = 0; y < countOnes; y++) {
                indX = roundDouble(arrayX[x]) + objxy[y * 2 + 1];
                indY = roundDouble(arrayY[x]) + objxy[y * 2];
                ind[x * countOnes + y] = fabs(indX * IszY * Nfr + indY * Nfr + k);
                if (ind[x * countOnes + y] >= max_size)
                    ind[x * countOnes + y] = 0;
            }
            likelihood[x] = 0;
            for (y = 0; y < countOnes; y++)
                likelihood[x] +=
                        (pow((I[ind[x * countOnes + y]] - 100), 2) - pow((I[ind[x * countOnes + y]] - 228), 2)) / 50.0;
            likelihood[x] = likelihood[x] / ((double) countOnes);
        }
        long long likelihood_time = get_time();
        printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error, likelihood_time));
        // update & normalize weights
        // using equation (63) of Arulampalam Tutorial
        //#pragma omp parallel for shared(Nparticles, weights, likelihood) private(x)
        for (x = 0; x < Nparticles; x++) {
            weights[x] = weights[x] * exp(likelihood[x]);
        }
        long long exponential = get_time();
        printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time, exponential));
        double sumWeights = 0;
        //#pragma omp parallel for private(x) reduction(+:sumWeights)
        for (x = 0; x < Nparticles; x++) {
            sumWeights += weights[x];
        }
        long long sum_time = get_time();
        printf("TIME TO SUM WEIGHTS TOOK: %f\n", elapsed_time(exponential, sum_time));
        //#pragma omp parallel for shared(sumWeights, weights) private(x)
        for (x = 0; x < Nparticles; x++) {
            weights[x] = weights[x] / sumWeights;
        }
        long long normalize = get_time();
        printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n", elapsed_time(sum_time, normalize));
        xe = 0;
        ye = 0;
        // estimate the object location by expected values
        //#pragma omp parallel for private(x) reduction(+:xe, ye)
        for (x = 0; x < Nparticles; x++) {
            xe += arrayX[x] * weights[x];
            ye += arrayY[x] * weights[x];
        }
        long long move_time = get_time();
        printf("TIME TO MOVE OBJECT TOOK: %f\n", elapsed_time(normalize, move_time));
        printf("XE: %lf\n", xe);
        printf("YE: %lf\n", ye);
        double distance = sqrt(pow((double) (xe - (int) roundDouble(IszY / 2.0)), 2) +
                               pow((double) (ye - (int) roundDouble(IszX / 2.0)), 2));
        printf("%lf\n", distance);
        //display(hold off for now)

        //pause(hold off for now)

        //resampling


        CDF[0] = weights[0];
        for (x = 1; x < Nparticles; x++) {
            CDF[x] = weights[x] + CDF[x - 1];
        }
        long long cum_sum = get_time();
        printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time, cum_sum));
        double u1 = (1 / ((double) (Nparticles))) * randu(seed, 0);
        //#pragma omp parallel for shared(u, u1, Nparticles) private(x)
        for (x = 0; x < Nparticles; x++) {
            u[x] = u1 + x / ((double) (Nparticles));
        }
        long long u_time = get_time();
        printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
        int j, i;

        //#pragma omp parallel for shared(CDF, Nparticles, xj, yj, u, arrayX, arrayY) private(i, j)
        for (j = 0; j < Nparticles; j++) {
            i = findIndex(CDF, Nparticles, u[j]);
            if (i == -1)
                i = Nparticles - 1;
            //printf("%d ", i);     
            xj[j] = arrayX[i];
            yj[j] = arrayY[i];

        }
        //printf("\n"); 

        long long xyj_time = get_time();
        printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n", elapsed_time(u_time, xyj_time));

        //#pragma omp parallel for shared(weights, Nparticles) private(x)
        for (x = 0; x < Nparticles; x++) {
            //reassign arrayX and arrayY
            arrayX[x] = xj[x];
            arrayY[x] = yj[x];
            weights[x] = 1 / ((double) (Nparticles));
            printf("%d x: %f, y: %f, w: %f\n", x, arrayX[x], arrayY[x], weights[x]);
        }
        long long reset = get_time();
        printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
    }
    free(disk);
    free(objxy);
    free(weights);
    free(likelihood);
    free(xj);
    free(yj);
    free(arrayX);
    free(arrayY);
    free(CDF);
    free(u);
    free(ind);
}

#ifdef USE_RISCV_VECTOR
void particleFilter_vector(int * I, int IszX, int IszY, int Nfr, int * seed, long int * seed_64,double* randu_vector_result,int* randu_vector_num, int Nparticles){


    int max_size = IszX*IszY*Nfr;
    long long start = get_time();
    //original particle centroid
    double xe = roundDouble(IszY/2.0);
    double ye = roundDouble(IszX/2.0);

    //expected object locations, compared to center
    int radius = 5;
    int diameter = radius*2 - 1;
    int * disk = (int *)malloc(diameter*diameter*sizeof(int));
    strelDisk(disk, radius);
    int countOnes = 0;
    int x, y;
    for(x = 0; x < diameter; x++){
        for(y = 0; y < diameter; y++){
            if(disk[x*diameter + y] == 1)
                countOnes++;
        }
    }

    //printf("countOnes = %d \n",countOnes); // 69

    double * objxy = (double *)malloc(countOnes*2*sizeof(double));
    getneighbors(disk, countOnes, objxy, radius);

    long long get_neighbors = get_time();
    printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start, get_neighbors));
    //initial weights are all equal (1/Nparticles)
    double * weights = (double *)malloc(sizeof(double)*Nparticles);
    //#pragma omp parallel for shared(weights, Nparticles) private(x)

    for(x = 0; x < Nparticles; x++){
        weights[x] = 1/((double)(Nparticles));
    }

    int limit = loop_bound(SPECIES_512, Nparticles);
    printf("limit: %d (%d)\n", limit, Nparticles);
//    unsigned long int gvl = __builtin_epi_vsetvl(Nparticles, __epi_e64, __epi_m1);

    _MMR_f64    xweights = _MM_SET_f64(1.0/((double)(Nparticles)));
    for(x = 0; x < Nparticles; x=x+SPECIES_512){
//        gvl     = __builtin_epi_vsetvl(Nparticles-x, __epi_e64, __epi_m1);
//        _MM_STORE_f64(&weights[x],xweights);
    }

    for (; x < Nparticles; x++) {
         weights[x] = 1.0/(double)(Nparticles);
    }
    //FENCE();

    long long get_weights = get_time();
    printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors, get_weights));
    //initial likelihood to 0.0
    double * likelihood = (double *)malloc(sizeof(double)*Nparticles);
    double * arrayX = (double *)malloc(sizeof(double)*Nparticles);
    double * arrayY = (double *)malloc(sizeof(double)*Nparticles);
    double * xj = (double *)malloc(sizeof(double)*Nparticles);
    double * yj = (double *)malloc(sizeof(double)*Nparticles);
    double * CDF = (double *)malloc(sizeof(double)*Nparticles);
    double * u = (double *)malloc(sizeof(double)*Nparticles);
    int * ind = (int*)malloc(sizeof(int)*countOnes*Nparticles);
    // Se usa adentro del for, aqui para no repetir
    long int * locations = (long int *)malloc(sizeof(long int)*Nparticles);

    /*
    //#pragma omp parallel for shared(arrayX, arrayY, xe, ye) private(x)
    for(x = 0; x < Nparticles; x++){
        arrayX[x] = xe;
        arrayY[x] = ye;
    }
    */
//    gvl     = __builtin_epi_vsetvl(Nparticles, __epi_e64, __epi_m1);
    _MMR_f64    xArrayX = _MM_SET_f64(xe);
    _MMR_f64    xArrayY = _MM_SET_f64(ye);
    for(int i = 0; i < limit; i=i+SPECIES_512){
//        gvl     = __builtin_epi_vsetvl(Nparticles-i, __epi_e64, __epi_m1);
//        _MM_STORE_f64(&arrayX[i],xArrayX); // TODO malloc top allocation error (malloc but memory is not set before)
//        _MM_STORE_f64(&arrayY[i],xArrayY);
    }
//    FENCE();


    _MMR_f64    xAux;

    int k;
    printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, get_time()));
    int indX, indY;


    for(k = 1; k < Nfr; k++){
        long long set_arrays = get_time();
        //apply motion model
        //draws sample from motion model (random walk). The only prior information
        //is that the object moves 2x as fast as in the y direction
//        gvl     = __builtin_epi_vsetvl(Nparticles, __epi_e64, __epi_m1);
        for(x = 0; x < limit; x=x+SPECIES_512){
//            gvl     = __builtin_epi_vsetvl(Nparticles-x, __epi_e64, __epi_m1);
            xArrayX = _MM_LOAD_f64(&arrayX[x]);
            xAux = randn_vector(seed_64, x,randu_vector_result,randu_vector_num);
            xAux =  _MM_MUL_f64(xAux, _MM_SET_f64(5.0));
            xAux =  _MM_ADD_f64(xAux, _MM_SET_f64(1.0));
            xArrayX = _MM_ADD_f64(xAux, xArrayX);
            _MM_STORE_f64(&arrayX[x],xArrayX);

            xArrayY = _MM_LOAD_f64(&arrayY[x]);
            xAux = randn_vector(seed_64, x,randu_vector_result,randu_vector_num);
            xAux =  _MM_MUL_f64(xAux, _MM_SET_f64(2.0));
            xAux =  _MM_ADD_f64(xAux, _MM_SET_f64(-2.0));
            xArrayY = _MM_ADD_f64(xAux, xArrayY);
//            printf("B: %f, %f, %f, %f, %f, %f, %f, %f\n", arrayY[x], arrayY[x+1], arrayY[x+2], arrayY[x+3], arrayY[x+4], arrayY[x+5], arrayY[x+6], arrayY[x+7]);
            _MM_STORE_f64(&arrayY[x],xArrayY);
//                        printf("A: %f, %f, %f, %f, %f, %f, %f, %f\n", arrayY[x], arrayY[x+1], arrayY[x+2], arrayY[x+3], arrayY[x+4], arrayY[x+5], arrayY[x+6], arrayY[x+7]);

        }
//        FENCE();

        //#pragma omp parallel for shared(arrayX, arrayY, Nparticles, seed) private(x)
        for(; x < Nparticles; x++){
            arrayX[x] += 1 + 5*randn(seed, x);
            arrayY[x] += -2 + 2*randn(seed, x);
        }

        long long error = get_time();
        printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
        //particle filter likelihood
        //#pragma omp parallel for shared(likelihood, I, arrayX, arrayY, objxy, ind) private(x, y, indX, indY)
        for(x = 0; x < Nparticles; x++){
            //compute the likelihood: remember our assumption is that you know
            // foreground and the background image intensity distribution.
            // Notice that we consider here a likelihood ratio, instead of
            // p(z|x). It is possible in this case. why? a hometask for you.
            //calc ind
            for(y = 0; y < countOnes; y++){
                indX = roundDouble(arrayX[x]) + objxy[y*2 + 1];
                indY = roundDouble(arrayY[x]) + objxy[y*2];
                ind[x*countOnes + y] = fabs(indX*IszY*Nfr + indY*Nfr + k);
                if(ind[x*countOnes + y] >= max_size)
                    ind[x*countOnes + y] = 0;
            }
            likelihood[x] = 0;
            for(y = 0; y < countOnes; y++)
                likelihood[x] += (pow((I[ind[x*countOnes + y]] - 100),2) - pow((I[ind[x*countOnes + y]]-228),2))/50.0;
            likelihood[x] = likelihood[x]/((double) countOnes);
        }
        long long likelihood_time = get_time();
        printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error, likelihood_time));
        // update & normalize weights
        // using equation (63) of Arulampalam Tutorial
        //#pragma omp parallel for shared(Nparticles, weights, likelihood) private(x)
        for(x = 0; x < Nparticles; x++){
            weights[x] = weights[x] * exp(likelihood[x]);
        }
        long long exponential = get_time();
        printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time, exponential));
        double sumWeights = 0;
        //#pragma omp parallel for private(x) reduction(+:sumWeights)
        for(x = 0; x < Nparticles; x++){
            sumWeights += weights[x];
        }
        long long sum_time = get_time();
        printf("TIME TO SUM WEIGHTS TOOK: %f\n", elapsed_time(exponential, sum_time));
        //#pragma omp parallel for shared(sumWeights, weights) private(x)
        for(x = 0; x < Nparticles; x++){
            weights[x] = weights[x]/sumWeights;
        }
        long long normalize = get_time();
        printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n", elapsed_time(sum_time, normalize));
        xe = 0;
        ye = 0;
        // estimate the object location by expected values
        //#pragma omp parallel for private(x) reduction(+:xe, ye)
        for(x = 0; x < Nparticles; x++){
            xe += arrayX[x] * weights[x];
            ye += arrayY[x] * weights[x];
        }
        long long move_time = get_time();
        printf("TIME TO MOVE OBJECT TOOK: %f\n", elapsed_time(normalize, move_time));
        printf("XE: %lf\n", xe);
        printf("YE: %lf\n", ye);
        double distance = sqrt( pow((double)(xe-(int)roundDouble(IszY/2.0)),2) + pow((double)(ye-(int)roundDouble(IszX/2.0)),2) );
        printf("%lf\n", distance);
        //display(hold off for now)

        //pause(hold off for now)

        //resampling

        CDF[0] = weights[0];
        for(x = 1; x < Nparticles; x++){
            CDF[x] = weights[x] + CDF[x-1];
        }
        long long cum_sum = get_time();
        printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time, cum_sum));
        double u1 = (1/((double)(Nparticles)))*randu(seed, 0);
        //#pragma omp parallel for shared(u, u1, Nparticles) private(x)
        for(x = 0; x < Nparticles; x++){
            u[x] = u1 + x/((double)(Nparticles));
        }
        long long u_time = get_time();
        printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));

        int j, i;

        _MMR_MASK_i64           xComp;
        _MMR_MASK_i64           xMask;

        _MMR_f64          xCDF;
        _MMR_f64          xU;
        _MMR_f64          xArray;

        long int vector_complete;
        long int valid;
//        gvl     = __builtin_epi_vsetvl(Nparticles, __epi_e64, __epi_m1);

        int * mask = (int *) malloc(SPECIES_512*sizeof(int *));

        for (int mi = 0; mi < SPECIES_512; mi++) {
            mask[mi] = 0;
        }

        for(i = 0; i < limit; i=i+SPECIES_512){
//            gvl     = __builtin_epi_vsetvl(Nparticles-i, __epi_e64, __epi_m1);
            vector_complete = 0;
            xMask   = _MM_VFEQ_f64(xMask, _MM_SET_f64(1), _MM_SET_f64(0));
            xArray  = _MM_SET_f64(Nparticles-1);
            xU      = _MM_LOAD_f64(&u[i]);
            for(j = 0; j < Nparticles; j++){
                xCDF = _MM_SET_f64(CDF[j]);
                xComp = _MM_VFGE_f64(xComp, xCDF,xU);
                xComp = _MM_VMXOR_i64(xComp,xMask);
//                valid = _MM_VMFIRST_i64(xComp); // TODO how to get the first valid bit???
                valid = -1;
                for (int v = 0; v < SPECIES_512; v++) {
                    if (xComp !=0) {
                        valid = v;
                        break;
                    }
                }
                if(valid != -1)
                {
                    _MMR_f64 xJ = _MM_ADD_f64_MASK(xJ, xComp, _MM_SET_f64(0), _MM_SET_f64(j));
                    xArray = _MM_ADD_f64_MASK(xArray, _MM_VMNOT_i64(xComp), xJ, _MM_SET_f64(j));
                    // _MM_MERGE_i64(xArray,_MM_SET_i64(j,gvl),xComp,gvl);
                    xMask = _MM_VMOR_i64(xComp,xMask);
                    vector_complete = _MM_VMPOPC_i64(xMask, SPECIES_512); // TODO no intrinsic for count the number of true values
                }
                if(vector_complete == SPECIES_512){ break; }
            }
            _MM_STORE_i64(&locations[i],xArray);
            //xArray = _MM_MUL_i64(xArray,_MM_SET_i64(8,gvl),gvl); // Position in elements to position in bytes
            //xarrayX = _MM_LOAD_INDEX_f64(&arrayX[i],xArray,gvl);
            //xarrayY = _MM_LOAD_INDEX_f64(&arrayY[i],xArray,gvl);
            //_MM_STORE_f64(&xj[i],xarrayX,gvl);
            //_MM_STORE_f64(&yj[i],xarrayY,gvl);
            // This commented lines corresponds to the scalar code below
        }
//        FENCE();

        for(j = limit; j < Nparticles; j++){
            i = findIndex(CDF, Nparticles, u[j]);
            if(i == -1)
                    i = Nparticles-1;
            //printf("%d ", i);
            xj[j] = arrayX[i];
            yj[j] = arrayY[i];

        }

        //#pragma omp parallel for shared(CDF, Nparticles, xj, yj, u, arrayX, arrayY) private(i, j)
        for(j = 0; j < limit; j++){
            i = locations[j];
            xj[j] = arrayX[i];
            yj[j] = arrayY[i];
        }
        // for(j = 0; j < Nparticles; j++){ printf("%lf ", xj[i]); } printf("\n");
        // for(j = 0; j < Nparticles; j++){ printf("%lf ", yj[i]); } printf("\n");

        long long xyj_time = get_time();
        printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n", elapsed_time(u_time, xyj_time));

        //#pragma omp parallel for shared(weights, Nparticles) private(x)
        for(x = 0; x < Nparticles; x++){
            //reassign arrayX and arrayY
            arrayX[x] = xj[x];
            arrayY[x] = yj[x];
            weights[x] = 1/((double)(Nparticles));

            printf("%d x: %f, y: %f, w: %f\n", x, arrayX[x], arrayY[x], weights[x]);
        }
        long long reset = get_time();
        printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time, reset));
    }
    free(locations);
    free(disk);
    free(objxy);
    free(weights);
    free(likelihood);
    free(xj);
    free(yj);
    free(arrayX);
    free(arrayY);
    free(CDF);
    free(u);
    free(ind);
}
#endif

int main(int argc, char *argv[]) {

    char *usage = "openmp.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
    //check number of arguments
    if (argc != 9) {
        printf("%s\n", usage);
        return 0;
    }
    //check args deliminators
    if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
        printf("%s\n", usage);
        return 0;
    }

    int IszX, IszY, Nfr, Nparticles;

    //converting a string to a integer
    if (sscanf(argv[2], "%d", &IszX) == EOF) {
        printf("ERROR: dimX input is incorrect");
        return 0;
    }

    if (IszX <= 0) {
        printf("dimX must be > 0\n");
        return 0;
    }

    //converting a string to a integer
    if (sscanf(argv[4], "%d", &IszY) == EOF) {
        printf("ERROR: dimY input is incorrect");
        return 0;
    }

    if (IszY <= 0) {
        printf("dimY must be > 0\n");
        return 0;
    }

    //converting a string to a integer
    if (sscanf(argv[6], "%d", &Nfr) == EOF) {
        printf("ERROR: Number of frames input is incorrect");
        return 0;
    }

    if (Nfr <= 0) {
        printf("number of frames must be > 0\n");
        return 0;
    }

    //converting a string to a integer
    if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
        printf("ERROR: Number of particles input is incorrect");
        return 0;
    }

    if (Nparticles <= 0) {
        printf("Number of particles must be > 0\n");
        return 0;
    }
    //establish seed
    int *seed = (int *) malloc(sizeof(int) * Nparticles);
    int i;
    for (i = 0; i < Nparticles; i++) {
        seed[i] = i; //time(0)*i;
    }
    //malloc matrix
    int *I = (int *) malloc(sizeof(int) * IszX * IszY * Nfr); // 128 * 128 * 10 = 163840 * sizeof(int)
    long long start = get_time();
    //call video sequence
    videoSequence(I, IszX, IszY, Nfr, seed);
    long long endVideoSequence = get_time();
    printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));

#ifdef USE_RISCV_VECTOR
    //    unsigned long int gvl = __builtin_epi_vsetvl(Nparticles, __epi_e64, __epi_m1);
        double* randu_vector_result = (double*)malloc(SPECIES_512*sizeof(double));
        int* randu_vector_num = (int*)malloc(SPECIES_512*sizeof(int));
#endif

#ifdef USE_RISCV_VECTOR
    long int * seed_64 = (long int *)malloc(sizeof(long int)*Nparticles);
    for(i = 0; i < Nparticles; i++)
    {
        seed_64[i] = (long int)seed[i];
    }
#endif

    // Start instruction and cycles count of the region of interest
    unsigned long cycles1, cycles2, instr2, instr1;
    instr1 = get_inst_count();
    cycles1 = get_cycles_count();

#ifdef USE_RISCV_VECTOR
    //call particle filter
   particleFilter_vector(I, IszX, IszY, Nfr, seed,seed_64,randu_vector_result,randu_vector_num, Nparticles);
#else
    //call particle filter
    particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
#endif

    // End instruction and cycles count of the region of interest
    instr2 = get_inst_count();
    cycles2 = get_cycles_count();

    long long endParticleFilter = get_time();
    printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
    printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

    // Instruction and cycles count of the region of interest
    printf("-CSR   NUMBER OF EXEC CYCLES :%lu\n", cycles2 - cycles1);
    printf("-CSR   NUMBER OF INSTRUCTIONS EXECUTED :%lu\n", instr2 - instr1);


    for (i = 0; i < Nparticles; i++) {
        printf("seed[%d] = %d\n", i, seed[i]);

    }
#ifdef USE_RISCV_VECTOR
    free(randu_vector_result);
    free(randu_vector_num);
#endif

//    for (i = 0; i < Nparticles; i++) {
//        printf("seed[%d] -> %d\n",i, seed[i]);
//    }
//
//    for (i = 0; i < 50; i++) {
//        printf("%d -> %d\n", i, I[i]);
//    }


    free(seed);
    free(I);
    return 0;
}
