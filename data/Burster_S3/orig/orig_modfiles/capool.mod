: Ca2+ pool for IKCa

NEURON {
        SUFFIX capool
	USEION ca READ ica,cai,cao WRITE cai
	RANGE fca, tauca, cainf
}

UNITS {
        (mM) = (milli/liter)
        (mA) = (milliamp)
	(mV) = (millivolt)
	FARADAY = 96485.309 (coul)
}

PARAMETER {
        tauca= 200 (ms)       	: decay time constant
	fca = 1.0		:ca influx factor that goes to ca pool
	cainf = 50e-6 (mM)		: equilibrium ca2+ concentration ( = [ca]i at rest)
	w = 1 (micrometer)	: thickness of shell for ca2+ diffusion
	z = 2			: valence
}

ASSIGNED {
	v (mV)
	ica (mA/cm2)
	cao (mM)
}

STATE { cai(mM)  }

BREAKPOINT { 
	SOLVE states METHOD cnexp
}

INITIAL {
	:cai = cainf
}

DERIVATIVE states {
		cai'= -fca*ica/(z*FARADAY*w*1e-4) + (cainf - cai)/tauca
}



