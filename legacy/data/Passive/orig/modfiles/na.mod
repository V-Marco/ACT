: spike-generating sodium channel (Pyramid)

NEURON {
	SUFFIX na
	USEION na READ ena WRITE ina
	RANGE gbar, gna, inaplot
	RANGE minf, hinf, mtau, htau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar = 0.300 (siemens/cm2) <0,1e9>
}

ASSIGNED {
	v (mV)
	ena (mV)
	ina (mA/cm2)
	minf
	hinf
	mtau (ms)
	htau (ms)
	gna (siemens/cm2)
	inaplot
}

STATE {
	m h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gna = gbar*m*m*m*h
	ina = gna*(v-ena)
	inaplot = ina
}

INITIAL {
	rate(v)
	m = minf
	h = hinf
}

DERIVATIVE states {
	rate(v)
	m' = (minf-m)/mtau
	h' = (hinf-h)/htau
}

PROCEDURE rate(v (mV)) {
	UNITSOFF

	minf = (1.0)/(1.0+(exp ((v+24.7)/(-5.29))))       
	mtau = (1.32) - ((1.26)/(1.0+(exp (-(v+120)/(25)))))         
	
	hinf = 1.0/(1.0+(exp ((v+48.9)/(5.18))))       
	htau = (0.67/(1+(exp (-(v+62.9)/(10.0)))))*(1.5+(1.0/(1+(exp ((v+34.9)/(3.6))))))     
	UNITSON
}