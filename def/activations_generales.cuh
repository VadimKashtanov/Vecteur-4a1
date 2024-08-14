#pragma once

#include "meta.cuh"

__device__
static float ACTIVATION(uint activ, float s) {
	float a;
	//
	if      (activ == 0) a = tanh(s);				//	tanh
	else if (activ == 1) a = 1 / (1 + expf(-s));	//	logistique
	else if (activ == 2) a = expf(-s*s);			//	gauss
	else if (activ == 3) a = (s > 0 ? s : 0.0);		//	ReLu
	else if (activ == 4) a = expf(s);				//	exp(x)
	else if (activ == 5) a = s;						//	id(x)
	else if (activ == 6) a = s/(1+expf(-s*1.7));	//	gelu(x)
	else assert(0);
	//
	return a;
};

__device__
static float d_ACTIVATION(uint activ, float s, float a) {
	float da;
	//
	if      (activ == 0) da = 1 - a*a;
	else if (activ == 1) da = a * (1 - a);
	else if (activ == 2) da = -2*s * a;
	else if (activ == 3) da = (s>0);
	else if (activ == 4) da = a;
	else if (activ == 5) da = 1;
	else if (activ == 6) {
		float X = 1.7*s;
		float _e = expf(X);
		float _e_1 = _e + 1;
		da = _e * (X + _e_1) / (_e_1*_e_1);
	}
	else assert(0);
	//
	return da;
};

__device__
static f2 ACTIVATION_f_df(uint activ, float s) {
	float  a =   ACTIVATION(activ, s   );
	float da = d_ACTIVATION(activ, s, a);
	//
	f2 ret = {a, da};
	//
	return ret;
};