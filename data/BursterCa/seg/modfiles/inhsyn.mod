:Inhibitory Synapse

NEURON {
	POINT_PROCESS inhsyn
	NONSPECIFIC_CURRENT isyn
	RANGE tau1, tau2, norm
	RANGE gsyn, gmax, esyn
	RANGE w
}

UNITS {
	(mV) = (millivolt)
        (nA) = (nanoamp)
	(uS) = (microsiemens)
}

PARAMETER {
	w 	:weight
	tau1 = 10 (ms)
	tau2 = 20 (ms)
	gmax (uS)
	esyn = -80.0 (mV)
}

ASSIGNED {
	v (mV)
	eca (mV)
	isyn (nA)
	gsyn
	norm
}

STATE {
	A
	B
}

INITIAL {
	A=0
	B=0
	norm = -1/((tau1*tau2/(tau2-tau1))*(1/exp(log(tau2/tau1)/(tau1*(1/tau1-1/tau2)))-1/exp(log(tau2/tau1)/(tau2*(1/tau1-1/tau2)))))
}

BREAKPOINT {
	SOLVE states METHOD cnexp
}

DERIVATIVE states {
	A' = -A/tau1
	B' = -B/tau2
	gsyn = w*gmax*norm*(tau1*tau2/(tau2-tau1))*(B-A)
	isyn = gsyn*(v-esyn)
}

NET_RECEIVE(wgt) {
	w = wgt
	state_discontinuity(A,A+1)
	state_discontinuity(B,B+1)
}
