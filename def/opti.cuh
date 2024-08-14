#pragma once

#include "mdl.cuh"

#define L2_regularisation 0.01

#define SGD 0
#define SGD_____HISTOIRE 0

#define MOMENT 1
#define MOMENT__HISTOIRE 1

#define RMSPROP 2
#define RMSPROP_HISTOIRE 1

#define ADAM 3
#define ADAM____HISTOIRE 2

void sgd(
	Mdl_t   *  mdl,
	float *** hist,	//[hist][inst][p]
	uint         i,
	float    alpha,
	uint         t
);

void moment(
	Mdl_t   *  mdl,
	float *** hist,	//[hist][inst][p]
	uint         i,
	float    alpha,
	uint         t
);

void rmsprop(
	Mdl_t   *  mdl,
	float *** hist,	//[hist][inst][p]
	uint         i,
	float    alpha,
	uint         t
);

void adam(
	Mdl_t   *  mdl,
	float *** hist,	//[hist][inst][p]
	uint         i,
	float    alpha,
	uint         t
);

//	---------------------------

void opti(
	Mdl_t     *     mdl,
	BTCUSDT_t * btcusdt,
	uint      *   ts__d,
	//
	uint              I,
	uint       tous_les,
	//
	uint        methode,
	float         alpha);