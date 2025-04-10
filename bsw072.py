import os
import json
import time
import psutil
import random
import pandas as pd
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
from charm.toolbox.ABEnc import ABEnc
from charm.toolbox.policytree import PolicyParser
from charm.toolbox.secretutil import SecretUtil
from msp import MSP  # Ensure msp.py (containing the MSP class) is in the same directory

# -------------------------------
# Simulated Resource Functions
# -------------------------------
def get_avail_resources(cpu_state, mem_state):
    """Returns simulated available CPU (%) and Memory (%) based on the state."""
    def random_range(state):
        state = state.lower()
        if state == "idle":
            return random.uniform(51, 80)
        elif state == "medium":
            return random.uniform(21, 50)
        elif state == "critical":
            return random.uniform(0, 20)
        else:
            return 50.0  # default value
    return random_range(cpu_state), random_range(mem_state)

# -------------------------------
# Multi-Objective Offloading Decision Function
# -------------------------------
def multi_objective_decision(T_full, T_partial, avail_cpu, avail_mem, num_attributes, file_size):
    """
    Decide whether to offload (partial encryption) or use full encryption locally.
    Returns 0 for Full, 1 for Partial.
    """
    if num_attributes <= 4 or file_size < 2048:
        return 0
    if avail_cpu < 40 or avail_mem < 40:
        return 1
    if avail_cpu >= 60 and avail_mem >= 60:
        return 0 if T_full <= T_partial else 1
    delta = (T_partial - T_full) / T_full
    return 1 if delta > 0.3 else 0

# -------------------------------
# CP-ABE Implementation with Full & Partial Encryption
# -------------------------------
class BSW07(ABEnc):
    """
    Implements the BSW07 CP-ABE encryption scheme.
    Full encryption is performed by encrypt_full.
    Partial encryption (with offloading) is performed by encrypt_partial.
    """
    def __init__(self, group_obj):
        ABEnc.__init__(self)
        self.group = group_obj
        self.util = MSP(group_obj, verbose=False)
        self.parser = PolicyParser()

    def setup(self):
        g1 = self.group.random(G1)
        g2 = self.group.random(G2)
        alpha = self.group.random(ZR)
        beta = self.group.random(ZR)
        h = g2 ** beta  
        f = g2 ** (1 / beta)
        e_gg_alpha = pair(g1, g2) ** alpha 

        pk = {'g1': g1, 'g2': g2, 'h': h, 'f': f, 'e_gg_alpha': e_gg_alpha}
        msk = {'beta': beta, 'alpha': alpha}
        return pk, msk

    def encrypt_full(self, pk, msg, policy_str):
        """Performs Full Encryption locally."""
        start_time = time.time()
        policy_tree = self.util.createPolicy(policy_str)
        mono_span_prog = self.util.convert_policy_to_msp(policy_tree)
        num_cols = self.util.len_longest_row
        u = [self.group.random(ZR) for _ in range(num_cols)]
        s = u[0]
        c0 = pk['h'] ** s
        C = {}
        for attr, row in mono_span_prog.items():
            row_sum = sum(row[i] * u[i] for i in range(len(row)))
            attr_stripped = self.util.strip_index(attr)
            C[attr] = (pk['g1'] ** row_sum, self.group.hash(attr_stripped, G1) ** row_sum)
        c_m = (pk['e_gg_alpha'] ** s) * msg
        exec_time = time.time() - start_time
        return {'policy': policy_str, 'c0': c0, 'C': C, 'c_m': c_m, 'execution_time': exec_time}

    def encrypt_partial(self, pk, msg, policy_str):
        """Performs Partial Encryption on RP1 and offloads the rest to RP2."""
        start_time = time.time()
        dummy_policy = "Dummy AND " + policy_str
        policy_tree = self.util.createPolicy(dummy_policy)
        mono_span_prog = self.util.convert_policy_to_msp(policy_tree)
        num_cols = self.util.len_longest_row
        u = [self.group.random(ZR) for _ in range(num_cols)]
        s = u[0]
        c0 = pk['h'] ** s
        # RP1: Only encrypt Dummy attribute
        C = {"Dummy": (pk['g1'] ** u[0], self.group.hash("Dummy", G1) ** u[0])}
        c_m = (pk['e_gg_alpha'] ** s) * msg
        rp1_time = time.time() - start_time
        # Save RP1 time to a file
        with open("intermediate_params.json", "w") as f:
            json.dump({"RP1_time": rp1_time}, f, indent=4)
        return {'c0': c0, 'C_partial': C, 'c_m': c_m, 'RP1_time': rp1_time}

# -------------------------------
# Core Experiment Runner
# -------------------------------
def run_experiment(file_size, num_attributes, cpu_state, mem_state, network):
    """
    Runs full and partial encryption experiments and determines offloading decision.
    """
    print(f"\n[Experiment] File Size: {file_size} Bytes | Attributes: {num_attributes} | CPU: {cpu_state} | Memory: {mem_state} | Network: {network}")
    
    # Get simulated available resources
    avail_cpu, avail_mem = get_avail_resources(cpu_state, mem_state)

    # Initialize CP-ABE
    cpabe = BSW07(PairingGroup('MNT224'))
    pk, msk = cpabe.setup()
    policy = " AND ".join([f"Attr_{i}" for i in range(1, num_attributes + 1)])
    message = cpabe.group.random(GT)

    # Full Encryption
    full_result = cpabe.encrypt_full(pk, message, policy)
    T_full = full_result["execution_time"]

    # Partial Encryption
    partial_result = cpabe.encrypt_partial(pk, message, policy)
    rp1_time = partial_result["RP1_time"]
    
    # Simulated transmission time
    data_size = os.path.getsize("intermediate_params.json")
    if network.upper() == "LAN":
        bandwidth = 10**7  
        latency = 0.01
    else:
        bandwidth = 5 * 10**6  
        latency = 0.05
    T_trans = data_size / bandwidth + latency

    # Simulate RP2 processing time
    T_rp2 = random.uniform(0.005, 0.02)  
    partial_total_time = rp1_time + T_trans + T_rp2

    # Decision
    decision = multi_objective_decision(T_full, partial_total_time, avail_cpu, avail_mem, num_attributes, file_size)
    
    return {
        "File_Size_Bytes": file_size,
        "Attributes": num_attributes,
        "CPU_State": cpu_state,
        "Memory_State": mem_state,
        "Network": network,
        "Avail_CPU_%": avail_cpu,
        "Avail_Memory_%": avail_mem,
        "Full_Execution_Time_s": T_full,
        "Partial_RP1_Time_s": rp1_time,
        "Transmission_Time_s": T_trans,
        "Partial_RP2_Time_s": T_rp2,
        "Partial_Total_Time_s": partial_total_time,
        "Offload_Decision": decision
    }

# -------------------------------
# Main Experiment Loop
# -------------------------------
if __name__ == "__main__":
    FILE_SIZES = [1, 1024, 10240, 102400, 1048576, 10485760]
    ATTRIBUTES = [2, 4, 5, 9, 12, 20, 30, 50, 100, 250, 500]  
    CPU_STATES = ["Idle", "Medium", "Critical"]
    MEMORY_STATES = ["Idle", "Medium", "Critical"]
    NETWORK_TYPES = ["LAN", "WLAN"]

    experiments = []
    for fs in FILE_SIZES:
        for na in ATTRIBUTES:
            for cs in CPU_STATES:
                for ms in MEMORY_STATES:
                    for nw in NETWORK_TYPES:
                        exp_data = run_experiment(fs, na, cs, ms, nw)
                        experiments.append(exp_data)

    pd.DataFrame(experiments).to_csv("experiment_results.csv", index=False)
    print("\n✅ Experiment complete! Results saved in 'experiment_results.csv'.")

