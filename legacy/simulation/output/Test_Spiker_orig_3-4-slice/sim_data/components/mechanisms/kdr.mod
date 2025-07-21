: potassium delayed rectifier channel (Pyramid)

NEURON {
	SUFFIX kdr
	USEION k READ ek WRITE ik
	RANGE gbar, gkdr, ikd
	RANGE inf, tau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar = 1.860 (siemens/cm2) <0,1e9>
}

ASSIGNED {
	v (mV)
	ek (mV)
	ik (mA/cm2)
	gkdr (siemens/cm2)
	inf
	tau (ms)
	ikd
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gkdr = gbar*n*n*n*n
	ik = gkdr*(v-ek)
	ikd = ik
}

INITIAL {
	rate(v)
	n = inf
}

DERIVATIVE states {
	rate(v)
	n' = (inf-n)/tau
}

PROCEDURE rate(v (mV)) {
	UNITSOFF

	inf = 1.0/(1+ (exp ((v+14.2)/(-11.8))))         
	tau = 7.2-(6.4/(1+(exp (-(v+28.3)/(19.2)))))      
	UNITSON
}