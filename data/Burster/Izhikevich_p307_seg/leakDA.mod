: passive leak current

NEURON {
	SUFFIX leakDA
	NONSPECIFIC_CURRENT il
	RANGE il, el, glbar
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	DA_start = 100		             : D1R(Low Affinity) Dopamine Effect after 6 conditioning trials (15*4000) = 60000)
	DA_stop = 600	
	DA_t1 = 0.8 : 0.9 : 1 :  1 : 0.9           : Amount(%) of DA effect- negative value decreases AP threshold / positive value increases threshold of AP
	
	glbar = 2.857142857142857e-05  :3.333333e-5 (siemens/cm2) < 0, 1e9 >
	el = -75 (mV)
}

ASSIGNED {
	v (mV)
	il (mA/cm2)
}

BREAKPOINT { 
	il = glbar*(v - el)*DA1(t)
}
FUNCTION DA1(t) {
	    if (t >= DA_start && t <= DA_stop){DA1 = DA_t1} 									: During conditioning
		else  {DA1 = 1}
	}