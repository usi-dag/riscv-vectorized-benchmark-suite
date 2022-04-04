//
// Created by mini on 29.03.22.
//

#include "../common/vector_defines.h"
#include <string.h>
#include <stdio.h>

void print_512d_vector(_MMR_f64 vector) {
    double val[8];
    memcpy(val, &vector, sizeof(val));
    printf("Numerical: %f %f %f %f %f %f %f %f \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}

int main() {

    _MMR_f64 a = _MM_SET_f64(1.0);
    a = _MM_LOG_f64(a);
    print_512d_vector(a);

    return 1;
}

