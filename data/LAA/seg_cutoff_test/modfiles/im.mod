: voltage-gated persistent muscarinic channel

NEURON {
	SUFFIX im
	USEION k READ ek WRITE ik
	RANGE gmbar, gm, i
	RANGE inf, tau, lincutoff, vcutoff, slope, intercept
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gmbar = 0.0003 (siemens/cm2) <0,1e9>
	lincutoff
	vcutoff
	slope
	intercept
}

ASSIGNED {
	v (mV)
	ek (mV)
	ik (mA/cm2)
	i  (mA/cm2)
	inf
	tau (ms)
	gm (siemens/cm2)
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gm = gmbar*n*n
	ik = gm*(v-ek)
	i = ik
}

INITIAL {
	rate(v)
	n = inf
}

DERIVATIVE states {
	rate(v)
	n' = (inf-n)/tau
}

FUNCTION alf(v (mV)) (/ms) {
	UNITSOFF
	alf = 0.016/exp(-(v+52.7)/23)
	UNITSON
}

FUNCTION bet(v (mV)) (/ms) {
	UNITSOFF
	bet = 0.016/exp((v+52.7)/18.8)
	UNITSON
}

PROCEDURE rate(v (mV)) {
	LOCAL sum, aa, ab
	UNITSOFF
	aa=alf(v) ab=bet(v) 
	
	sum = aa+ab
	:inf = aa/sum
	inf = 1 / ( 1 + exp( ( - v - 52.7 ) / 10.34 ) )
	:Segregation
	if (v < lincutoff) {
	inf = slope * v + intercept
	}
	if (v < vcutoff) {
	inf = 0
	}
	tau = 1/sum
	: tau = 1.5/sum
	UNITSON
}
