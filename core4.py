import os
import json
import time
import psutil
import base64
import random
import hashlib
import pandas as pd
from multiprocessing import Pool
from fastecdsa.curve import P256
from fastecdsa.keys import gen_keypair
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# -------------------------------
# Utility: Get Available Resources by State (Randomized)
# -------------------------------
def get_avail_resources(cpu_state, mem_state):
    """
    Returns simulated available CPU (%) and Memory (%) based on the state.
    For both CPU and memory:
      - Idle: random value between 51% and 80%
      - Medium: random value between 21% and 50%
      - Critical: random value between 0% and 20%
    """
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
# Full Encryption (Local Mode)
# -------------------------------
class ECC_CPABE_Full:
    def __init__(self, curve=P256):
        """Initialize ECC-based CP-ABE for full encryption."""
        self.curve = curve
        self.generator = curve.G
        self.params = {}
        self.k_x = None  # AES key
        self.k_y = None  # MAC key

    def setup(self, num_attributes):
        """Generate master public key and attribute public keys."""
        alpha = gen_keypair(self.curve)[0]  # Master secret key
        self.params['MPK'] = alpha * self.generator  # MPK = αB
        self.params['Generator'] = self.generator
        self.params['Attributes'] = [f"Attr_{i}" for i in range(1, num_attributes + 1)]
        self.params['PublicKeys'] = {}
        for attr in self.params['Attributes']:
            hashed_attr = self.hash_attribute(attr)
            attr_secret = gen_keypair(self.curve)[0]
            attr_public = attr_secret * self.generator
            self.params['PublicKeys'][hashed_attr] = attr_public

    def hash_attribute(self, attr):
        """Hash attribute name using SHA-256 and reduce modulo curve order."""
        return int(hashlib.sha256(attr.encode()).hexdigest(), 16) % self.curve.q

    def derive_key(self, sk, key_length, info):
        """Derive a cryptographic key using HKDF from the ECC point's x-coordinate."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=None,
            info=info,
            backend=default_backend()
        )
        return hkdf.derive(str(sk.x).encode())

    def encrypt(self, message, access_tree):
        """
        Encrypt the message using ECC-based CP-ABE.
        Measures execution time.
        """
        #cpu_before = psutil.cpu_percent(interval=0.1)
        #mem_before = psutil.virtual_memory().percent
        start_time = time.time()

        # Generate a random session key k and compute session key point.
        k = gen_keypair(self.curve)[0]
        sk_point = k * self.params['MPK']
        self.k_x = self.derive_key(sk_point, key_length=16, info=b'CP-ABE AES key')
        self.k_y = self.derive_key(sk_point, key_length=16, info=b'CP-ABE MAC key')

        cipher = AES.new(self.k_x, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(pad(message.encode(), AES.block_size))

        # Build a simple access tree: root plus one leaf per attribute.
        polynomials = {}
        for node in access_tree:
            if node == "root":
                polynomials[node] = k
            else:
                parent = access_tree[node]["parent"]
                polynomials[node] = polynomials[parent] + gen_keypair(self.curve)[0]

        #cpu_after = psutil.cpu_percent(interval=0.1)
        #mem_after = psutil.virtual_memory().percent
        exec_time = time.time() - start_time

        return {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "tag": base64.b64encode(tag).decode('utf-8'),
            "execution_time": exec_time,
           # "cpu_usage": (cpu_before + cpu_after) / 2,
           # "memory_usage": (mem_before + mem_after) / 2
        }

# -------------------------------
# Partial Encryption: RP1 (Local Part)
# -------------------------------
class ECC_CPABE_RP1:
    def __init__(self, curve=P256):
        """Initialize RP1 for partial encryption."""
        self.curve = curve
        self.generator = curve.G
        self.params = {}

    def setup(self):
        """Generate master public key for RP1."""
        alpha = gen_keypair(self.curve)[0]
        self.params['MPK'] = alpha * self.generator
        self.params['Generator'] = self.generator

    def partial_encrypt(self, message):
        """
        Partial encryption on RP1: encrypts using only the root and a right leaf (Attr_1).
        Measures execution time and saves intermediate parameters.
        """
        #cpu_before = psutil.cpu_percent(interval=0.1)
        #mem_before = psutil.virtual_memory().percent
        start_time = time.time()

        k = gen_keypair(self.curve)[0]
        polynomials = {"root": k}
        polynomials["Attr_1"] = polynomials["root"] + gen_keypair(self.curve)[0]

        sk_point = polynomials["root"] * self.params['MPK']
        k_x = HKDF(
            algorithm=hashes.SHA256(),
            length=16,
            salt=None,
            info=b'CP-ABE AES key',
            backend=default_backend()
        ).derive(str(sk_point.x).encode())

        cipher = AES.new(k_x, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(pad(message.encode(), AES.block_size))

       # cpu_after = psutil.cpu_percent(interval=0.1)
       # mem_after = psutil.virtual_memory().percent
        exec_time = time.time() - start_time

        intermediate_params = {
            "root_polynomial": str(polynomials["root"]),
            "attr_1_polynomial": str(polynomials["Attr_1"]),
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "tag": base64.b64encode(tag).decode('utf-8'),
          #  "RP1_cpu": (cpu_before + cpu_after) / 2,
           # "RP1_memory": (mem_before + mem_after) / 2,
            "RP1_time": exec_time
        }
        with open("intermediate_params.json", "w") as f:
            json.dump(intermediate_params, f, indent=4)
        return intermediate_params

# -------------------------------
# Partial Encryption: RP2 (Offloaded Part)
# -------------------------------
class ECC_CPABE_RP2:
    def __init__(self, curve=P256):
        """Initialize RP2 for offloaded encryption."""
        self.curve = curve
        self.generator = curve.G

    def full_encrypt(self, num_attributes):
        """
        RP2: Loads intermediate parameters and continues polynomial propagation
        for attributes 2 to num_attributes.
        Measures execution time.
        """
        try:
            with open("intermediate_params.json", "r") as f:
                intermediate_params = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Intermediate parameters not found; run RP1 partial encryption first.")

        root_polynomial = int(intermediate_params["root_polynomial"])
        attr1_polynomial = int(intermediate_params["attr_1_polynomial"])
        
      #  cpu_before = psutil.cpu_percent(interval=0.1)
      #  mem_before = psutil.virtual_memory().percent
        start_time = time.time()
        
        polynomials = {"root": root_polynomial, "Attr_1": attr1_polynomial}
        for i in range(2, num_attributes + 1):
            polynomials[f"Attr_{i}"] = polynomials[f"Attr_{i-1}"] + gen_keypair(self.curve)[0]
        
     #   cpu_after = psutil.cpu_percent(interval=0.1)
    #    mem_after = psutil.virtual_memory().percent
        exec_time = time.time() - start_time
        
        return {
        #    "RP2_cpu": (cpu_before + cpu_after) / 2,
       #     "RP2_memory": (mem_before + mem_after) / 2,
            "RP2_time": exec_time
        }

# -------------------------------
# Revised Multi-Objective Offloading Decision Function Using Simulated States
# -------------------------------
def multi_objective_decision(T_full, T_partial, avail_cpu, avail_mem, num_attributes, file_size):
    """
    Decide whether to offload (partial encryption) or use full encryption locally
    based on multi-objective considerations using simulated available resources.
    
    Parameters:
      T_full      : Execution time for full encryption.
      T_partial   : Total execution time for partial encryption (RP1 + transmission + RP2).
      avail_cpu   : Simulated available CPU (%) based on state.
      avail_mem   : Simulated available Memory (%) based on state.
      num_attributes : Number of attributes in the access policy.
      file_size   : File size in bytes.
      
    Returns:
      0  : Recommend full encryption.
      1  : Recommend partial encryption (offloading).
    """
    # For very small policies or files, prefer full encryption.
    if num_attributes <= 4 or file_size < 2048:
        return 0

    # Strategy:
    # - If available resources are very low (<40%), choose partial encryption.
    # - If available resources are high (>=60%), choose full encryption if T_full is less.
    # - Otherwise, use relative time difference.
    if avail_cpu < 40 or avail_mem < 40:
        return 1
    if avail_cpu >= 60 and avail_mem >= 60:
        return 0 if T_full <= T_partial else 1

    delta = (T_partial - T_full) / T_full
    return 1 if delta > 0.3 else 0

# -------------------------------
# Core Experiment Runner
# -------------------------------
def run_experiment(file_size, num_attributes, cpu_state, mem_state, network):
    """
    Runs full and partial encryption experiments, then computes the offloading decision
    using the multi-objective strategy with simulated available resources.
    
    Parameters:
      file_size: in bytes.
      num_attributes: number of attributes.
      cpu_state: CPU state ("Idle", "Medium", "Critical").
      mem_state: Memory state ("Idle", "Medium", "Critical").
      network: network type ("LAN" or "WLAN") to simulate transmission delay.
      
    Returns a dictionary with all performance metrics and the offload decision.
    """
    print(f"\n[Experiment] File Size: {file_size} Bytes | Attributes: {num_attributes} | CPU State: {cpu_state} | Memory State: {mem_state} | Network: {network}")
    exp_start = time.time()
    
    # Get simulated available resources.
    avail_cpu, avail_mem = get_avail_resources(cpu_state, mem_state)
    
    # --- FULL ENCRYPTION ---
    full_cpabe = ECC_CPABE_Full()
    full_cpabe.setup(num_attributes)
    message = "A" * file_size
    access_tree = {"root": {"type": "node", "threshold": random.randint(2, 10),
                              "children": [f"Attr_{i}" for i in range(1, num_attributes + 1)]}}
    for i in range(1, num_attributes + 1):
        access_tree[f"Attr_{i}"] = {"type": "leaf", "parent": "root", "attribute": f"Attr_{i}"}
    full_result = full_cpabe.encrypt(message, access_tree)
    
    # --- PARTIAL ENCRYPTION (RP1 + Transmission + RP2) ---
    rp1 = ECC_CPABE_RP1()
    rp1.setup()
    rp1_result = rp1.partial_encrypt(message)
    
    if not os.path.exists("intermediate_params.json"):
        raise FileNotFoundError("Intermediate parameters file not found!")
    data_size = os.path.getsize("intermediate_params.json")
    if network.upper() == "LAN":
        bandwidth = 10**7  # bytes/sec
        latency = 0.01
    else:
        bandwidth = 5 * 10**6  # bytes/sec
        latency = 0.05
    T_trans = data_size / bandwidth + latency
    print(f"Transmission Time: {T_trans:.4f} sec (Data Size: {data_size} bytes)")
    
    rp2 = ECC_CPABE_RP2()
    rp2_result = rp2.full_encrypt(num_attributes)
    
    partial_total_time = rp1_result["RP1_time"] + T_trans + rp2_result["RP2_time"]
 #   partial_avg_cpu = (rp1_result["RP1_cpu"] + rp2_result["RP2_cpu"]) / 2
  #  partial_avg_mem = (rp1_result["RP1_memory"] + rp2_result["RP2_memory"]) / 2

    decision = multi_objective_decision(
        T_full=full_result["execution_time"],
        T_partial=partial_total_time,
        avail_cpu=avail_cpu,
        avail_mem=avail_mem,
        num_attributes=num_attributes,
        file_size=file_size
    )

    exp_total_time = time.time() - exp_start
    print(f"Experiment completed in {exp_total_time:.4f} sec.")
    
    return {
        "File_Size_Bytes": file_size,
        "Attributes": num_attributes,
        "CPU_State": cpu_state,
        "Memory_State": mem_state,
        "Network": network,
        "Avail_CPU_%": avail_cpu,
        "Avail_Memory_%": avail_mem,
        "Full_Execution_Time_s": full_result["execution_time"],
        "Partial_RP1_Time_s": rp1_result["RP1_time"],
        "Transmission_Time_s": T_trans,
        "Partial_RP2_Time_s": rp2_result["RP2_time"],
        "Partial_Total_Time_s": partial_total_time,
        "Offload_Decision": decision  # 0 = Full, 1 = Partial
    }

# -------------------------------
# Main Experiment Loop
# -------------------------------
if __name__ == "__main__":
    # File sizes in bytes: 1 Byte, 1KB, 10KB, 100KB, 1MB, 10 MB.
    FILE_SIZES = [1, 1024, 10240, 102400, 1048576, 10485760]
    ATTRIBUTES = list(range(500, 501))       # Attributes from 2 to 100
    #ATTRIBUTES = random.sample(range(2, 101), 2)  # 10 random values between 2 and 100
    CPU_STATES = ["Idle", "Medium", "Critical"]
    MEMORY_STATES = ["Idle", "Medium", "Critical"]
    NETWORK_TYPES = ["LAN", "WLAN"]

    experiments = []
    total_experiments = len(FILE_SIZES) * len(ATTRIBUTES) * len(CPU_STATES) * len(MEMORY_STATES) * len(NETWORK_TYPES)
    exp_count = 1
    for fs in FILE_SIZES:
        for na in ATTRIBUTES:
            for cs in CPU_STATES:
                for ms in MEMORY_STATES:
                    for nw in NETWORK_TYPES:
                        print(f"\n=== Experiment {exp_count}/{total_experiments} ===")
                        exp_data = run_experiment(fs, na, cs, ms, nw)
                        experiments.append(exp_data)
                        exp_count += 1

    df = pd.DataFrame(experiments)
    output_file = "core_encryption_experiment_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Experiment complete! Results saved in '{output_file}'.")

