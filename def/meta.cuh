#pragma once

#include "etc.cuh"

//	Grand_t = Mini Batchs
//	Mega_t  = ... t-1, t, t+1 ...
//	Intervs = intervalles (heures, jours, mois ...)

//	Les Parantèses () sont importantes.

#define INTERVS 4

#define GRAND_T 16//16//8//16

#define MEGA_T 1

#define PLUS_DECALAGE 1 //on compare x=bloque[i] et y=bloque[i+plus]

#define t_MODE(t, mega_t) ( (t)*(MEGA_T) + (mega_t) )