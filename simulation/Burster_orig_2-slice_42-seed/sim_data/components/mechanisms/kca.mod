:  ca-dependent potassium current

NEURON {
	SUFFIX kca
	USEION k READ ek WRITE ik
	USEION ca READ cai
        RANGE g, gbar, ik
}

UNITS {
        (mM) = (milli/liter)
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gbar (siemens/cm2)
}

ASSIGNED {
	v (mV)
	ek (mV)
	ik (mA/cm2)
	cinf 
	ctau (ms)
	g (siemens/cm2)
	cai (mM)
}

STATE {
	c
}


BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar*c*c*c*c       
	ik = g*(v-ek)
}


INITIAL {
	rate(v,cai)
	c = cinf
}

DERIVATIVE states {
        rate(v,cai)
	c' = (cinf-c)/ctau
}

PROCEDURE rate(v (mV), cai (mM)) {
	UNITSOFF
	:activation based on internal concentration of capool
	cinf = ((cai)/(cai + 0.003))*((1.0)/(1+ (exp (-(v+28.3)/(12.6)))))        
	ctau = ((180.6)-(150.2)/(1+(exp (-(v+46)/(22.7))))) 
	UNITSON		
}


