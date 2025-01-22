from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.circuit.library.data_preparation import StatePreparation
from qiskit import *
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit.visualization import circuit_drawer
import qiskit_aer
from qiskit_aer import AerSimulator
import cmath
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
import matplotlib as mat
import pylatexenc
#Am selectat aceleasi librarii de la tema1

#Exercitiul 1:
def create_state_phi1():  
    qc = QuantumCircuit(2, 2)  

    qc.h(0)  
    qc.h(1)  
    qc.s(1)  

    qc.measure(0, 0)  
    qc.measure(1, 1)

    return qc  
 
def create_state_phi2():  
    qc = QuantumCircuit(3, 3)  
     #Incorect
    qc.x(0)
    qc.x(1)
    qc.ry(1.91063, 2)  
    qc.cx(2, 0)  
    qc.cx(2, 1)
   
    qc.measure(0,0)
    qc.measure(1,1)
    qc.measure(2,2)
    return qc

        
def simulate_circuit(circuit, shots=3000):  
    backend = AerSimulator()  
    qc_comp = transpile(circuit, backend)  
    job_sim = backend.run(qc_comp, shots=shots)  
    result_sim = job_sim.result()  
    counts = result_sim.get_counts(qc_comp)  
    return counts  

   
def Testare1():  
    print("Circuit pentru starea |φ1⟩:")  
    circuit_phi1 = create_state_phi1()  
    print(circuit_phi1)  
    counts_phi1 = simulate_circuit(circuit_phi1)  
    print("\nRezultate pentru |φ1⟩:")  
    print(counts_phi1)  

    print("\nCircuit pentru starea |φ2⟩:")  
    circuit_phi2 = create_state_phi2()  
    print(circuit_phi2)  
    counts_phi2 = simulate_circuit(circuit_phi2)  
    print("\nRezultate pentru |φ2⟩:")  
    print(counts_phi2)

#Testare1()



def Exercitiul_2():  
    
    qr = QuantumRegister(3)  
    cr = ClassicalRegister(3)  
    qc = QuantumCircuit(qr, cr)

    #Exercitiul 2 a)
    qc.h(0)  
    qc.s(0)  

    # Exercitiul 2 b)
    qc.h(1)  
    qc.cx(1, 2)  

    # Exercitiul 2 c)
    qc.cx(0, 1)   
    qc.h(0)  
    qc.measure(0, 0) 
    qc.measure(1, 1)  
    with qc.if_test((cr[1], 1)):  qc.x(2)  
    with qc.if_test((cr[0], 1)):  qc.z(2)
    #calcul pe hartie, share la poza la prezentare
    #return qc
     
    qc.measure(2, 2) 
    # Simulare  
    backend = AerSimulator()  
    qc_comp = transpile(qc, backend)  
    job_sim = backend.run(qc_comp, shots=2000)  
    result_sim = job_sim.result()  
    counts = result_sim.get_counts(qc_comp)
    print("Rezultate:")  
    for outcome, count in counts.items():  
        print(f"{outcome}: {count} ({count/2000*100:.1f}%)")   

#Exercitiul_2()


#Exercitiul 3:
def create_state(state_index):  
    qr = QuantumRegister(3, 'q')  
    cr = ClassicalRegister(3, 'c')  
    circuit = QuantumCircuit(qr, cr)  

    theta = 2 * np.pi / 3  
  
    if state_index == 0:  
        
        circuit.x(0)  
        circuit.h([0,1,2])  
        circuit.p(2*theta, 1)  
        circuit.p(theta, 2)    
        circuit.cx(0, 1)  
        circuit.cx(1, 2)  
        circuit.h([0,1,2])  

    else:  
        circuit.x(0)  
        circuit.h([0,1,2])  
        circuit.p(theta, 1)   
        circuit.p(2*theta, 2)    
        circuit.cx(0, 1)  
        circuit.cx(1, 2)  
        circuit.h([0,1,2])  
 
    circuit.measure([0,1,2], [0,1,2])  
    '''
    Pentru |φ₀⟩:

    '100': 1166  
    '010': 351  
    '111': 344  
    '001': 139  

    Pentru |φ₁⟩:

    '100': 1118  
    '111': 399  
    '010': 360  
    '001': 123
    '''
    return circuit

# Starea Phi0
circuit0 = create_state(0)  
backend = AerSimulator()  
qc_comp = transpile(circuit0, backend)  
job_sim = backend.run(qc_comp, shots=9000)  
result_sim = job_sim.result()  
counts0 = result_sim.get_counts(qc_comp)
 

# Starea Phi1 
circuit1 = create_state(1)  
backend = AerSimulator()  
qc_comp = transpile(circuit1, backend)  
job_sim = backend.run(qc_comp, shots=9000)  
result_sim = job_sim.result()  
counts1 = result_sim.get_counts(qc_comp)  

def determine_state(counts):  
   
    total = sum(counts.values())  
    proportions = {state: count/total for state, count in counts.items()}  

    if proportions.get('010', 0) > 0:  
        ratio_111_010 = proportions.get('111', 0) / proportions.get('010', 0) 
        print(ratio_111_010)
    else:  
        ratio_111_010 = float('inf') 

 
    if ratio_111_010 > 1.04 : return 1 #399/360 = 1.11
    else : return 0                   #344/351 = 0.98

def Rezultate3():
    print("Rezultate pentru |φ₀⟩:")  
    print(counts0)  
    print("\nRezultate pentru |φ₁⟩:")  
    print(counts1)
    print(determine_state(counts0))
    print(determine_state(counts1))
Rezultate3()  

'''
Am folosit H pentru a transforma diferentele de faza in diferente de amplitudine
si P pentru a introduce theta si theta^2
Am masurat rezultatele, am vazut cautat un pattern
Frecventele 111 si 010 sunt diferite pentru cele 2 stari 
Raportul lor este pentru phi0 cam 0.98 iar pentru phi1 cam 1.11
Pe baza acestei observatii am facut returnarea

Am observat si ca theta^3=1 din formula Phase Gate
'''