: A-type  K+ CHANNEL (KA)

NEURON {
	SUFFIX ka
	USEION k READ ek WRITE ik
	RANGE gbar, g
	RANGE ainf, atau, binf, btau
	RANGE ik
}

UNITS {
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
	ainf
	atau (ms)
	binf
	btau (ms)
	g (siemens/cm2)
}

STATE {
	a b
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gbar*a*a*a*b
	ik = g*(v-ek)
}

INITIAL {
	rate(v)
	a = ainf
	b = binf
}

DERIVATIVE states {
	rate(v)
	a' = (ainf-a)/atau
	b' = (binf-b)/btau
}

PROCEDURE rate(v (mV)) {
	UNITSOFF
	ainf = 1.0/(1.0+(exp ((v+27.2)/(-8.7))))         
	atau = (23.2 - 20.8/(1+(exp ((v+32.9)/(-15.2)))))
	binf = 1.0/(1.0+ (exp ((v+56.9)/(4.9))))        
	btau = (77.2 - 58.4/(1.0+ (exp ((v+38.9)/(-26.5)))))              
	UNITSON
}
