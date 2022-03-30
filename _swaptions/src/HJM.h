#include <assert.h>
#include "HJM_type.h"

#include <cstring>

#include <time.h>
#include <sys/time.h>


FTYPE RanUnif( long *s );
void RanUnif_vector( long *s , int iFactors , int iN ,int  BLOCKSIZE, FTYPE **randZ);

FTYPE CumNormalInv( FTYPE u );
void CumNormalInv_vector( FTYPE* u ,FTYPE* output);

void icdf_SSE(const int N, FTYPE *in, FTYPE *out);
void icdf_baseline(const int N, FTYPE *in, FTYPE *out);
int HJM_SimPath_Forward_SSE(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
			    FTYPE **ppdFactors, long *lRndSeed);
int Discount_Factors_SSE(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath);
int Discount_Factors_opt(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath);


int HJM_SimPath_Forward_Blocking_SSE(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
			    FTYPE **ppdFactors, long *lRndSeed, int BLOCKSIZE);
int HJM_SimPath_Forward_Blocking(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
			    FTYPE **ppdFactors, long *lRndSeed, int BLOCKSIZE);


int Discount_Factors_Blocking(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath, int BLOCKSIZE);
int Discount_Factors_Blocking_vector(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath, int BLOCKSIZE);
int Discount_Factors_Blocking_SSE(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath, int BLOCKSIZE);

int HJM_SimPath_Yield(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdYield, FTYPE **ppdFactors,
                      long *lRndSeed);
int HJM_SimPath_Forward(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdForward, FTYPE *pdTotalDrift,
                        FTYPE **ppdFactors, long *lRndSeed);
int HJM_Yield_to_Forward(FTYPE *pdForward, int iN, FTYPE *pdYield);
int HJM_Factors(FTYPE **ppdFactors,int iN, int iFactors, FTYPE *pdVol, FTYPE **ppdFacBreak);
int HJM_Drifts(FTYPE *pdTotalDrift, FTYPE **ppdDrifts, int iN, int iFactors, FTYPE dYears, FTYPE **ppdFactors);
int HJM_Correlations(FTYPE **ppdHJMCorr, int iN, int iFactors, FTYPE **ppdFactors);
int HJM_Forward_to_Yield(FTYPE *pdYield, int iN, FTYPE *pdForward);
int Discount_Factors(FTYPE *pdDiscountFactors, int iN, FTYPE dYears, FTYPE *pdRatePath);

int HJM_Swaption_Blocking_SSE(FTYPE *pdSwaptionPrice, //Output vector that will store simulation results in the form:
			                              //Swaption Price
			                              //Swaption Standard Error
			      //Swaption Parameters 
			      FTYPE dStrike,				  
			      FTYPE dCompounding,     //Compounding convention used for quoting the strike (0 => continuous,
			      //0.5 => semi-annual, 1 => annual).
			      FTYPE dMaturity,	      //Maturity of the swaption (time to expiration)
			      FTYPE dTenor,	      //Tenor of the swap
			      FTYPE dPaymentInterval, //frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half
		                              //year
			      //HJM Framework Parameters (please refer HJM.cpp for explanation of variables and functions)
			      int iN,						
			      int iFactors, 
			      FTYPE dYears, 
			      FTYPE *pdYield, 
			      FTYPE **ppdFactors,
			      //Simulation Parameters
			      long iRndSeed, 
			      long lTrials, int blocksize, int tid);
 
int HJM_Swaption_Blocking(FTYPE *pdSwaptionPrice, //Output vector that will store simulation results in the form:
			                              //Swaption Price
			                              //Swaption Standard Error
			      //Swaption Parameters 
			      FTYPE dStrike,				  
			      FTYPE dCompounding,     //Compounding convention used for quoting the strike (0 => continuous,
			      //0.5 => semi-annual, 1 => annual).
			      FTYPE dMaturity,	      //Maturity of the swaption (time to expiration)
			      FTYPE dTenor,	      //Tenor of the swap
			      FTYPE dPaymentInterval, //frequency of swap payments e.g. dPaymentInterval = 0.5 implies a swap payment every half
		                              //year
			      //HJM Framework Parameters (please refer HJM.cpp for explanation of variables and functions)
			      int iN,						
			      int iFactors, 
			      FTYPE dYears, 
			      FTYPE *pdYield, 
			      FTYPE **ppdFactors,
			      //Simulation Parameters
			      long *iRndSeed, 
			      long lTrials, int blocksize, int tid);

/*
extern "C" FTYPE *dvector( long nl, long nh );
extern "C" FTYPE **dmatrix( long nrl, long nrh, long ncl, long nch );
extern "C" void free_dvector( FTYPE *v, long nl, long nh );
extern "C" void free_dmatrix( FTYPE **m, long nrl, long nrh, long ncl, long nch );
*/
