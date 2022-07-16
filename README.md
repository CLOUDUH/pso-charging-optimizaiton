# Battery charging optimization by PSO

2022/07/16 via CLOUDUH in Tianjin University

---

# building…… please wait 220716

## Introduction

Battery charging optimization program.

Use coupling model which include battery 1-RC equivalent circuit model & thermal model & aging model.

Optimization algorithermal_model is particle swarm optimization

You also can find .m program in this repository

## Contents

All processes are implemented through functions

### Battery Model

    battery_model
        equivalent_circuit_model: Li-ion battery equivalent circuit model
        thermal_model: Battery thermal model
        aging_model: Battery semi-empirical model aging model

### Photovoltaic Model

    photovoltaic_model
        irradiation_cal: irradiation model
        photovoltaic_model: Solar cell model

### 

    
