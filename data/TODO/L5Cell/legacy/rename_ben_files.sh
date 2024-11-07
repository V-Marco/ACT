#!/bin/bash

cd "/home/drfrbc/Neural-Modeling/scripts/output/BenModel/output/"
pwd
mv v_report.h5 Vm_report.h5
mv NaTa_t.ina_report.h5 ina_NaTa_t_data_report.h5
mv NaTa_t.gNaTa_t_report.h5 gNaTa_t_NaTa_t_data_report.h5
mv inmda_report.h5 i_NMDA_report.h5
mv i_membrane__report.h5 i_membrane_report.h5
mv Ih.ihcn_report.h5 ihcn_Ih_data_report.h5
mv igaba_report.h5 i_GABA_report.h5
mv iampa_report.h5 i_AMPA_report.h5
mv Ca_LVAst.ica_report.h5 ica_Ca_LVAst_data_report.h5
mv Ca_HVA.ica_report.h5 ica_Ca_HVA_data_report.h5
mv spikes.h5 spikes_report.h5

