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


def ex1(a, b, c):
   # Calculul |ab>  
    ket_ab = [0] * 4  # Initializare lista cu 4 zerouri  
    index = 0  
    for i in range(2):  
        for j in range(2):  
            ket_ab[index] = a[i][0] * b[j][0]  
            index += 1  

    # Calculul |ca>  
    ket_ca = [0] * 4   
    index = 0  
    for i in range(2):  
        for j in range(2):  
            ket_ca[index] = c[i][0] * a[j][0]  
            index += 1

   # Calculul <ca|  
    ket_ca_conjugat = [0] * 4  
    for i in range(4):  
        ket_ca_conjugat[i] = ket_ca[i].conjugate()  

    # Calculul |ab><ca|  
    ket_ab_bra_ca = [[0 for _ in range(4)] for _ in range(4)] 
    for i in range(4):  
        for j in range(4):  
            ket_ab_bra_ca[i][j] = ket_ab[i] * ket_ca_conjugat[j]  

    return ket_ab_bra_ca

a = np.array([[1], [0]], dtype=complex)  
b = np.array([[0], [1]], dtype=complex)  
c = np.array([[1], [1]], dtype=complex)
print(ex1(a, b, c))


def ex2(A):
    
    rows = len(A)  
    if rows == 0:  return False  
    cols = len(A[0])

    if rows != cols:  return False

  # Calculul adjunctei (A†)  
    transpusa = [[0 for _ in range(rows)] for _ in range(cols)]  
    # for _ in range(cols)] - (Outer loop):controlează numarul de linii din matricea transpusa
    # [[0 for _ in range(rows)] - (Inner loop) : creează o linie din matricea transpusa, umplutu cu rows zerouri
    # Echivalent cu: 
    '''
     transpusa = [] 
     for _ in range(cols):  
     linie = []  
     for _ in range(rows):   
         linie.append(0)  
     transpusa.append(linie) 
    '''
    for i in range(rows):  
        for j in range(cols):  
            transpusa[j][i] = A[i][j]  
    Adjuncta = [[0 for _ in range(rows)] for _ in range(cols)]  
    for i in range(cols):  
        for j in range(rows):  
            Adjuncta[i][j] = transpusa[i][j].conjugate()  

    # Calculul matricei identitate (I)  
    Identitate = [[0 for _ in range(rows)] for _ in range(rows)]  
    for i in range(rows):  
        for j in range(rows):  
            if i == j:  Identitate[i][j] = 1  
            else:  Identitate[i][j] = 0

    # Produsul A†A  
    produs1 = [[0 for _ in range(rows)] for _ in range(rows)]  
    for i in range(rows):  
        for j in range(rows):  
            for k in range(cols):  
                produs1[i][j] += Adjuncta[i][k] * A[k][j]  

    # Produsul AA†  
    produs2 = [[0 for _ in range(rows)] for _ in range(rows)]  
    for i in range(rows):  
        for j in range(rows):  
            for k in range(cols):  
                produs2[i][j] += A[i][k] * Adjuncta[k][j]  

    # Convertim matricele in array-uri pentru a putea calcula norma  
    produs1_np = np.array(produs1)  
    produs2_np = np.array(produs2)  
    Identitate_np = np.array(Identitate)  

    norma1 = np.linalg.norm(produs1_np - Identitate_np)  
    norma2 = np.linalg.norm(produs2_np - Identitate_np)  

    epsilon=1e-6  
    return norma1 <= epsilon and norma2 <= epsilon

B = [  
        [1, 2],  
        [3, 4]  
    ]
C = [  
        [1/np.sqrt(2), 1/np.sqrt(2)],  
        [1/np.sqrt(2), -1/np.sqrt(2)]  
    ]

#print(ex2(C))


def ex3():
    #Cerinta a)
    circ=QuantumCircuit(2)
    circ.h(0)
    circ.h(1)
    circ.cx(0,1)
    circ.h(0)
    circ.h(1)
    circ.swap(0,1)
    print(circ)
    
    # Cerinta b)
    U = Operator(circ)
    U.data
    print(U)
    if ex2(U.data): print("Da, este matrice unitara")
    else: print("Nu, nu este matrice unitara")

    # Cerinta c)
    circ.h(0)
    circ.measure_all()
    print(circ)
    backend = AerSimulator()
    qc_comp= transpile(circ, backend)
    job_sim = backend.run(qc_comp, shots=2000)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc_comp)
    print(counts)
    histogram = plot_histogram(counts, color="red")
    histogram.savefig("histograma.png")

    #Cerinta d)   
    bell_states = ["00", "01", "10", "11"]
    for bell_state in bell_states:  
        circ2=QuantumCircuit(2,2)
        if bell_state == "00":  
            circ2.h(0)  
            circ2.cx(0, 1)  
        elif bell_state == "01":  
            circ2.x(1)  
            circ2.h(0)  
            circ2.cx(0, 1)  
        elif bell_state == "10":  
            circ2.h(0)  
            circ2.cx(0, 1)  
            circ2.z(0)  
        elif bell_state == "11":  
            circ2.x(1)  
            circ2.h(0)  
            circ2.cx(0, 1)  
            circ2.z(0)
        circ2.measure(0, 0)  
        circ2.measure(1, 1)    
        backend = AerSimulator()  
        qc_comp = transpile(circ2, backend)  
        job_sim = backend.run(qc_comp, shots=2000)  
        result_sim = job_sim.result()  
        counts = result_sim.get_counts(qc_comp)  
        print("Rezultate pentru starea Bell   B|",bell_state,">:")  
        for state, count in counts.items():  
            probability = count / 2000  
            print(f"  Starea: {state}, Probabilitate: {probability:.2f}")
        circ2.reset(range(2))          
# ex3()


def ex4(psi):
    psi = np.array(psi).reshape(4, 1)  # Transformare in vector coloana  

    #Calculam matricea densitate ρ = |ψ⟩⟨ψ|  
    rho = np.zeros((4, 4), dtype=complex)  
    for i in range(4):  
        for j in range(4):  
            rho[i, j] = psi[i, 0] * np.conjugate(psi[j, 0])  

    # Calculam ρ²  
    rho2 = np.zeros((4, 4), dtype=complex)  
    for i in range(4):  
        for j in range(4):  
            for k in range(4):  
                rho2[i, j] += rho[i, k] * rho[k, j]  

    # Verificam daca ρ² = ρ (aproximativ, pot fi erorilor de rotunjire)  
    return not np.allclose(rho2, rho)

psi_entangled = [1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2)]  
psi_separable = [1, 0, 0, 0] 
#print(ex4(psi_entangled))
#print(ex4(psi_separable))

#Referinte: (ultimul post de jos)
# https://quantumcomputing.stackexchange.com/questions/2263/how-do-i-show-that-a-two-qubit-state-is-an-entangled-state