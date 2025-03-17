:  iC   fast Ca2+/V-dependent K+ channel

NEURON {
	SUFFIX sAHPNE
	USEION k READ ek WRITE ik
	USEION cas READ casi VALENCE 2 
        RANGE ik, gk, gsAHPbar
}

UNITS {
        (mM) = (milli/liter)
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {

	NE_start = 100 : 36000		   : NE beta-R(Low Affinity) Norepinephrine Effect after 1 conditioning trials (9*4000 = 36000)
	NE_stop = 600
	NE_t1 = 0.7 : 0.9           : Amount(%) of NE effect

	
	gsAHPbar= 2.318144e-05 : 0.0001	(mho/cm2) : 
}

ASSIGNED {
	v (mV)
	ek (mV)
	casi (mM)
	ik (mA/cm2)
	cinf 
	ctau (ms)
	gk (mho/cm2)
}

STATE {
	c
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gk = gsAHPbar*c       
	ik = gk*(v-ek)*NE1(t)
}

INITIAL {
	rate(v,casi)
	c = cinf
}

DERIVATIVE states {
        rate(v,casi)
	c' = (cinf-c)/ctau
}

UNITSOFF


FUNCTION calf(v (mV), casi (mM)) (/ms) { LOCAL vs, va
	UNITSOFF
	vs=10*log10(1000*casi)
	calf = 0.0048/exp(-0.5*(vs-35))
	UNITSON
}

FUNCTION cbet(v (mV), casi (mM))(/ms) { LOCAL vs, vb 
	UNITSOFF
	  vs=10*log10(1000*casi)
	  cbet = 0.012/exp(0.2*(vs+100))
	UNITSON
}

UNITSON

PROCEDURE rate(v (mV), casi (mM)) {LOCAL  csum, ca, cb
	UNITSOFF
	ca=calf(v, casi) 
	cb=cbet(v, casi)		
	csum = ca+cb
	cinf = ca/csum
	ctau = 48
	UNITSON
}

FUNCTION NE1(t) {
	    if (t >= NE_start && t <= NE_stop){NE1 = NE_t1} 									: During conditioning
		else  {NE1 = 1}
	}	