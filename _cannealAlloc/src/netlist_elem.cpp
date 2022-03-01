// netlist_elem.cpp
//
// Created by Daniel Schwartz-Narbonne on 14/04/07.
//
// Copyright 2007 Princeton University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.


#include <stdlib.h>

#include <iostream>
#include <assert.h>
#include <math.h>

#include "annealer_types.h"
#include "location_t.h"
#include "netlist_elem.h"

using namespace std;

/*************************************************************************
* RISC-V Vectorized Version
* Author: Cristóbal Ramírez Lazo
* email: cristobal.ramirez@bsc.es
* Barcelona Supercomputing Center (2020)
*************************************************************************/

#ifdef USE_RISCV_VECTOR
#include "../../common/vector_defines.h"
#endif

netlist_elem::netlist_elem()
:present_loc(NULL)//start with the present_loc as nothing at all.  Filled in later by the netlist
{
#ifdef USE_RISCV_VECTOR
    fanin_locs_x  = (unsigned long *) malloc(INT32_SPECIES_512 * sizeof(unsigned long *));
    fanin_locs_y  = (unsigned long *) malloc(INT32_SPECIES_512 * sizeof(unsigned long *));
    fanout_locs_x = (unsigned long *) malloc(INT32_SPECIES_512 * sizeof(unsigned long *));
    fanout_locs_y = (unsigned long *) malloc(INT32_SPECIES_512 * sizeof(unsigned long *));
#endif
}

//netlist_elem::~netlist_elem() {
//    free(fanin_locs_x );
//    free(fanin_locs_y );
//    free(fanout_locs_x);
//    free(fanout_locs_y);
//}
#ifdef USE_RISCV_VECTOR
void netlist_elem::add_fanin_x(unsigned long * el) {

    if (fanin_x_size >= fanin_x_capacity) {
        fanin_locs_x = (unsigned long *) realloc(fanin_locs_x, fanin_x_size * 2 * sizeof(unsigned long *));
        fanin_x_capacity = fanin_x_size * 2;
    }

    fanin_locs_x[fanin_x_size++] = *el;
}

void netlist_elem::add_fanin_y(unsigned long * el) {

    if (fanin_y_size >= fanin_y_capacity) {
        fanin_locs_y = (unsigned long *) realloc(fanin_locs_y, fanin_y_size * 2 * sizeof(unsigned long *));
        fanin_y_capacity = fanin_y_size * 2;
    }

    fanin_locs_y[fanin_y_size++] = *el;
}

void netlist_elem::add_fanout_x(unsigned long * el) {

    if (fanout_x_size >= fanout_x_capacity) {
        fanout_locs_x = (unsigned long *) realloc(fanout_locs_x, fanout_x_size * 2 * sizeof(unsigned long *));
        fanout_x_capacity = fanout_x_size * 2;
    }

    fanout_locs_x[fanout_x_size++] = *el;
}

void netlist_elem::add_fanout_y(unsigned long * el) {

    if (fanout_y_size >= fanout_y_capacity) {
        fanout_locs_y = (unsigned long *) realloc(fanout_locs_y, fanout_y_size * 2 * sizeof(unsigned long *));
        fanout_y_capacity = fanout_y_size * 2;
    }

    fanout_locs_y[fanout_y_size++] = *el;
}
#endif
//*****************************************************************************************
// Calculates the routing cost using the manhatten distance
// I make sure to get the pointer in one operation, and then use it
// SYNC: Do i need to make this an atomic operation?  i.e. are there misaligned memoery issues that can cause this to fail
//       even if I have atomic writes?
//*****************************************************************************************
routing_cost_t netlist_elem::routing_cost_given_loc(location_t loc)
{
	routing_cost_t fanin_cost = 0;
	routing_cost_t fanout_cost = 0;
	
	for (int i = 0; i< fanin.size(); ++i){
		location_t* fanin_loc = fanin[i]->present_loc.Get();
		fanin_cost += fabs(loc.x - fanin_loc->x);
		fanin_cost += fabs(loc.y - fanin_loc->y);
	}

	for (int i = 0; i< fanout.size(); ++i){
		location_t* fanout_loc = fanout[i]->present_loc.Get();
		fanout_cost += fabs(loc.x - fanout_loc->x);
		fanout_cost += fabs(loc.y - fanout_loc->y);
	}

	routing_cost_t total_cost = fanin_cost + fanout_cost;
	return total_cost;
}

//*****************************************************************************************
//  Get the cost change of swapping from our present location to a new location
//*****************************************************************************************
#ifdef USE_RISCV_VECTOR
//routing_cost_t netlist_elem::swap_cost_vector(_MMR_i32 xOld_loc ,_MMR_i32 xNew_loc ,int fan_size)
//{
//	routing_cost_t no_swap = 0;
//	routing_cost_t yes_swap = 0;
//
//	_MMR_i32 xLoc2;
//    _MMR_i32 xNo_Swap_i;
//    _MMR_f32 xNo_Swap_aux;
//    _MMR_f32 xNo_Swap;
//    _MMR_f32 xresult_no_swap;
//    _MMR_i32 xYes_Swap_i;
//    _MMR_f32 xYes_Swap_aux;
//    _MMR_f32 xYes_Swap;
//    _MMR_f32 xresult_yes_swap;
//
//    int a_size;
//	a_size = fan_size*2;
//
//    unsigned long int gvl     = __builtin_epi_vsetvl(a_size, __epi_e32, __epi_m1);
//    xresult_no_swap = _MM_SET_f32(0.0f,gvl);
//    xresult_yes_swap = _MM_SET_f32(0.0f,gvl);
//    xNo_Swap = _MM_SET_f32(0.0f,gvl);
//    xYes_Swap = _MM_SET_f32(0.0f,gvl);
//
//    for(int i=0 ; i<a_size ; i = i + gvl)
//    {
//        FENCE();
//        // fan_locs  is a vector which holds the pointers to every input and ouput of the current node,
//        // Then by loading this first vector, it is possible to access to the pointers of the current location of each input and output.
//        gvl     = __builtin_epi_vsetvl((a_size-i)/2, __epi_e64, __epi_m1);
//
//        _MMR_i64   xLoc;
//        xLoc = _MM_LOAD_i64((const long *)&(fan_locs[i/2]),gvl);
//        xLoc = _MM_LOAD_INDEX_i64(0,xLoc,gvl);
//        xLoc = _MM_LOAD_INDEX_i64(0,xLoc,gvl);
//
//        FENCE();
//        gvl     = __builtin_epi_vsetvl(a_size-i, __epi_e32, __epi_m1);
//
//        xLoc2           =   (_MMR_i32)xLoc;
//
//        xNo_Swap_i      = _MM_SUB_i32(xOld_loc,xLoc2,gvl);
//        xNo_Swap_aux    = _MM_VFCVT_F_X_f32(xNo_Swap_i,gvl);
//        xNo_Swap_aux    = _MM_VFSGNJX_f32(xNo_Swap_aux,xNo_Swap_aux,gvl);
//
//        xYes_Swap_i     = _MM_SUB_i32(xNew_loc,xLoc2,gvl);
//        xYes_Swap_aux   = _MM_VFCVT_F_X_f32(xYes_Swap_i,gvl);
//        xYes_Swap_aux   = _MM_VFSGNJX_f32(xYes_Swap_aux,xYes_Swap_aux,gvl);
//
//        gvl     = __builtin_epi_vsetvl(a_size, __epi_e32, __epi_m1);
//        xNo_Swap        = _MM_ADD_f32(xNo_Swap,xNo_Swap_aux,gvl);
//        xYes_Swap       = _MM_ADD_f32(xYes_Swap,xYes_Swap_aux,gvl);
//
//        gvl     = __builtin_epi_vsetvl(a_size-i, __epi_e32, __epi_m1);
//
//    }
//
//    gvl     = __builtin_epi_vsetvl(a_size, __epi_e32, __epi_m1);
//
//    xresult_no_swap = _MM_REDSUM_f32(xNo_Swap,xresult_no_swap,gvl);
//    no_swap = _MM_VGETFIRST_f32(xresult_no_swap);
//
//    xresult_yes_swap = _MM_REDSUM_f32(xYes_Swap,xresult_yes_swap,gvl);
//    yes_swap = _MM_VGETFIRST_f32(xresult_yes_swap);
//    FENCE();
//
//    return (double)(yes_swap - no_swap);
//}
routing_cost_t netlist_elem::swap_cost_vector(location_t* old_loc, location_t* new_loc) {
    int fanin_size = fanin.size();
    int fanout_size = fanout.size();

    routing_cost_t no_swap = 0;
	  routing_cost_t yes_swap = 0;

    int limit_fanin = loop_bound(INT32_SPECIES_512, max(fanin_x_size, INT32_SPECIES_512));
    int limit_fanout = loop_bound(INT32_SPECIES_512, max(fanout_x_size, INT32_SPECIES_512));

    _MMR_i32 no_swap_vector = _MM_SET_i32(0);
    _MMR_i32 yes_swap_vector = _MM_SET_i32(0);
    _MMR_i32 old_loc_x = _MM_SET_i32(old_loc->x);
    _MMR_i32 old_loc_y = _MM_SET_i32(old_loc->y);
    _MMR_i32 new_loc_x = _MM_SET_i32(new_loc->x);
    _MMR_i32 new_loc_y = _MM_SET_i32(new_loc->y);
    _MMR_MASK_i32 mask_fanin;
    _MMR_MASK_i32 mask_fanout;

    int * index_fanin  = (int *) malloc(INT32_SPECIES_512 * sizeof(int));
    int * index_fanout = (int *) malloc(INT32_SPECIES_512 * sizeof(int));
    int * index_zeros  = (int *) malloc(INT32_SPECIES_512 * sizeof(int));


    for (int k = 0; k < INT32_SPECIES_512; ++k) {
        index_zeros[k] = 0;
        if (k < fanin_x_size) index_fanin[k] = 0;
        else index_fanin[k] = 1;
        if (k < fanout_x_size) index_fanout[k] = 0;
        else index_fanout[k] = 1;
    }


    mask_fanin = _MM_VFEQ_f32(mask_fanin, _MM_LOAD_f32((const int *)&index_fanin[0]), _MM_LOAD_f32((const int *)&index_zeros[0]));
    mask_fanout = _MM_VFEQ_f32(mask_fanout, _MM_LOAD_f32((const int *)&index_fanout[0]), _MM_LOAD_f32((const int *)&index_zeros[0]));


    int i;

    for (i = 0; i < limit_fanin; i += INT32_SPECIES_512) {
        _MMR_i32 fanin_loc_xv = _MM_LOAD_i32((const long *)&(fanin_locs_x[i]));
        _MMR_i32 fanin_loc_yv = _MM_LOAD_i32((const long *)&(fanin_locs_y[i]));

        no_swap_vector = _MM_ADD_i32(no_swap_vector, _MM_ABS_i32(_MM_SUB_i32(old_loc_x, fanin_loc_xv)));
        no_swap_vector = _MM_ADD_i32(no_swap_vector, _MM_ABS_i32(_MM_SUB_i32(old_loc_y, fanin_loc_yv)));

        yes_swap_vector = _MM_ADD_i32(yes_swap_vector, _MM_ABS_i32(_MM_SUB_i32(new_loc_x, fanin_loc_xv)));
        yes_swap_vector = _MM_ADD_i32(yes_swap_vector, _MM_ABS_i32(_MM_SUB_i32(new_loc_y, fanin_loc_yv)));
    }

    for (; i < fanin_size; i++) {
        location_t* fanin_loc = fanin[i]->present_loc.Get();

        no_swap += fabs(old_loc->x - fanin_loc->x);
        no_swap += fabs(old_loc->y - fanin_loc->y);

        yes_swap += fabs(new_loc->x - fanin_loc->x);
        yes_swap += fabs(new_loc->y - fanin_loc->y);
    }


    no_swap += _MM_MASK_REDSUM_i32(mask_fanin, no_swap_vector);
    yes_swap += _MM_MASK_REDSUM_i32(mask_fanin, yes_swap_vector);


    no_swap_vector = _MM_SET_i32(0);
    yes_swap_vector = _MM_SET_i32(0);

    for (i = 0; i < limit_fanout; i += INT32_SPECIES_512) {
        _MMR_i32 fanout_loc_xv = _MM_LOAD_i32((const long *)&(fanout_locs_x[i]));
        _MMR_i32 fanout_loc_yv = _MM_LOAD_i32((const long *)&(fanout_locs_y[i]));

        no_swap_vector = _MM_ADD_i32(no_swap_vector, _MM_ABS_i32(_MM_SUB_i32(old_loc_x, fanout_loc_xv)));
        no_swap_vector = _MM_ADD_i32(no_swap_vector, _MM_ABS_i32(_MM_SUB_i32(old_loc_y, fanout_loc_yv)));

        yes_swap_vector = _MM_ADD_i32(yes_swap_vector, _MM_ABS_i32(_MM_SUB_i32(new_loc_x, fanout_loc_xv)));
        yes_swap_vector = _MM_ADD_i32(yes_swap_vector, _MM_ABS_i32(_MM_SUB_i32(new_loc_y, fanout_loc_yv)));
    }

    for (; i < fanout_size; i++) {
        location_t* fanout_loc = fanout[i]->present_loc.Get();
		    no_swap += fabs(old_loc->x - fanout_loc->x);
		    no_swap += fabs(old_loc->y - fanout_loc->y);

		    yes_swap += fabs(new_loc->x - fanout_loc->x);
		    yes_swap += fabs(new_loc->y - fanout_loc->y);
    }

    no_swap += _MM_MASK_REDSUM_i32(mask_fanout, no_swap_vector);
    yes_swap += _MM_MASK_REDSUM_i32(mask_fanout, yes_swap_vector);
    free(index_fanin );
    free(index_fanout);
    free(index_zeros );
    return yes_swap - no_swap;

}
#else // !USE_RISCV_VECTOR
routing_cost_t netlist_elem::swap_cost(location_t* old_loc, location_t* new_loc)
{
	int fanin_size = fanin.size();
	int fanout_size = fanout.size();

	routing_cost_t no_swap = 0;
	routing_cost_t yes_swap = 0;
	
	//printf("fan size: %d\n" , fanin_size + fanout_size);
	
	for (int i = 0; i< fanin_size; ++i){
		location_t* fanin_loc = fanin[i]->present_loc.Get();

		no_swap += fabs(old_loc->x - fanin_loc->x);
		no_swap += fabs(old_loc->y - fanin_loc->y);
		
		yes_swap += fabs(new_loc->x - fanin_loc->x);
		yes_swap += fabs(new_loc->y - fanin_loc->y);
	}

	for (int i = 0; i< fanout_size; ++i){
		location_t* fanout_loc = fanout[i]->present_loc.Get();
		no_swap += fabs(old_loc->x - fanout_loc->x);
		no_swap += fabs(old_loc->y - fanout_loc->y);

		yes_swap += fabs(new_loc->x - fanout_loc->x);
		yes_swap += fabs(new_loc->y - fanout_loc->y);
	}
	return yes_swap - no_swap;
}
#endif //USE_RISCV_VECTOR