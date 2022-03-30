#ifndef __HJM_SECURITIES__
#define __HJM_SECURITIES__


#include "HJM_type.h"

int HJM_SimPath_Yield(FTYPE **ppdHJMPath, int iN, int iFactors, FTYPE dYears, FTYPE *pdYield, FTYPE **ppdFactors);


#endif //__HJM_SECURITIES__
