#!/usr/bin/env python3
"""Qiskit Aer scaling benchmark for the Hisab operator alphabet.
Synthetic circuits only. Exact statevector simulation; no hardware or NLP claims.
"""
from __future__ import annotations
import argparse, csv, json, math, platform, time
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
import qiskit, qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

THETA = math.pi / 4
TOPOLOGIES = ("local", "nearest", "all_to_all")
FIELDS = ["status","n_qubits","topology","logical_layers","seed",
"logical_ry_count","logical_rz_count","logical_cry_count",
"aer_transpile_seconds","aer_compiled_depth","aer_compiled_gate_count","aer_compiled_cry_count",
"basis_transpile_seconds","basis_depth_rz_sx_x_cx","basis_gate_count_rz_sx_x_cx",
"basis_cx_count","basis_sx_count","basis_rz_count","basis_x_count",
"statevector_memory_mb_minimum","elapsed_seconds","state_norm_error",
"central_cut_entropy_bits","central_cut_max_entropy_bits","central_cut_normalized_entropy",
"central_cut_schmidt_rank_numeric","phase_ablation_tvd_z_readout",
"phase_ablation_tvd_hadamard_readout_q0","phase_ablation_fidelity","error"]

def sign(seed, n, layer, q, gate_kind):
    v = seed*11 + n*3 + layer*5 + q*7 + gate_kind*13
    return -1 if v % 7 in (0,2,5) else 1

def all_to_all_rounds(n, seed, layer):
    players = list(range(n)); rounds = []
    for r in range(n-1):
        pairs = []
        for k in range(n//2):
            a,b = players[k], players[n-1-k]
            if ((a*13+b*7+r*2+seed+layer*5)%2)==0: a,b=b,a
            pairs.append((a,b))
        rounds.append(pairs)
        players = [players[0],players[-1]] + players[1:-1]
    return rounds

def build_circuit(n, layers, topology, seed, omit_final_rz_q0=False, hadamard_readout=False):
    if n < 2 or n % 2 or layers < 1:
        raise ValueError("n_qubits must be even and >=2; layers >=1")
    if topology not in TOPOLOGIES: raise ValueError(f"unknown topology: {topology}")
    qc = QuantumCircuit(n)
    for layer in range(layers):
        for q in range(n): qc.ry(sign(seed,n,layer,q,0)*THETA,q)
        for q in range(n):
            if not (omit_final_rz_q0 and layer==layers-1 and q==0):
                qc.rz(sign(seed,n,layer,q,1)*THETA,q)
        if topology=="nearest":
            for start in (0,1):
                for control in range(start,n-1,2):
                    target=control+1
                    qc.cry(sign(seed,n,layer,control+target,2)*THETA,control,target)
        elif topology=="all_to_all":
            for matching in all_to_all_rounds(n,seed,layer):
                for control,target in matching:
                    qc.cry(sign(seed,n,layer,control*17+target,2)*THETA,control,target)
    if hadamard_readout: qc.h(0)
    qc.save_statevector(label="psi")
    return qc

def simulate(qc, backend, seed):
    start_transpile = time.perf_counter()
    compiled = transpile(qc, backend, optimization_level=0, seed_transpiler=seed)
    aer_transpile_seconds = time.perf_counter() - start_transpile

    start = time.perf_counter()
    result = backend.run(compiled, shots=1, seed_simulator=seed).result()
    elapsed = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(result.status)
    state = np.asarray(result.data(0)["psi"], dtype=np.complex128)
    aer_ops = compiled.count_ops()

    # Aer keeps controlled-RY as a native simulator operation. To report a useful
    # hardware-like resource estimate, also transpile the unitary part into a
    # common {rz, sx, x, cx} basis. Omit save_statevector, which is simulator-only.
    unitary = QuantumCircuit(qc.num_qubits)
    for item in qc.data:
        if item.operation.name.startswith("save_"):
            continue
        qindices = [qc.find_bit(q).index for q in item.qubits]
        unitary.append(item.operation, [unitary.qubits[i] for i in qindices])
    start_basis = time.perf_counter()
    basis = transpile(
        unitary, basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0, seed_transpiler=seed
    )
    basis_transpile_seconds = time.perf_counter() - start_basis
    basis_ops = basis.count_ops()

    return state, {
      "elapsed_seconds": elapsed,
      "aer_transpile_seconds": aer_transpile_seconds,
      "aer_compiled_depth": int(compiled.depth()),
      "aer_compiled_gate_count": int(sum(v for k, v in aer_ops.items()
                                          if not str(k).startswith("save_"))),
      "aer_compiled_cry_count": int(aer_ops.get("cry", 0)),
      "basis_transpile_seconds": basis_transpile_seconds,
      "basis_depth_rz_sx_x_cx": int(basis.depth()),
      "basis_gate_count_rz_sx_x_cx": int(sum(basis_ops.values())),
      "basis_cx_count": int(basis_ops.get("cx", 0)),
      "basis_sx_count": int(basis_ops.get("sx", 0)),
      "basis_rz_count": int(basis_ops.get("rz", 0)),
      "basis_x_count": int(basis_ops.get("x", 0)),
    }

def entropy_central_cut(state,n):
    cut=n//2
    matrix=state.reshape((1<<cut,1<<(n-cut)))
    singular=np.linalg.svd(matrix,compute_uv=False)
    weights=singular**2
    weights=weights[weights>1e-15]; weights/=weights.sum()
    entropy=float(-np.sum(weights*np.log2(weights)))
    rank=int(np.count_nonzero(singular>singular.max()*1e-10))
    return entropy,rank

def tvd(p,q): return float(.5*np.sum(np.abs(p-q)))

def configurations():
    for n in (4,8,12,16):
        for topology in TOPOLOGIES:
            for layers in (1,2,4):
                for seed in (11,29): yield n,topology,layers,seed
    for topology in TOPOLOGIES:
        for layers in ((1,2,4) if topology!="all_to_all" else (1,2)):
            yield 20,topology,layers,11
    for n in (22,24):
        for topology in ("local","nearest"):
            for layers in (1,2,4): yield n,topology,layers,11
        yield n,"all_to_all",1,11

def evaluate(n,topology,layers,seed,backend):
    row={k:"" for k in FIELDS}
    logical_cry=(0 if topology=="local" else ((n-1)*layers if topology=="nearest" else n*(n-1)//2*layers))
    row.update(status="ok",n_qubits=n,topology=topology,logical_layers=layers,seed=seed,
      logical_ry_count=n*layers,logical_rz_count=n*layers,logical_cry_count=logical_cry,
      statevector_memory_mb_minimum=(16*(1<<n))/1e6,central_cut_max_entropy_bits=n/2)
    try:
        state,meta=simulate(build_circuit(n,layers,topology,seed),backend,seed)
        row.update(meta); row["state_norm_error"]=abs(float(np.vdot(state,state).real)-1.0)
        if n<=20:
            entropy,rank=entropy_central_cut(state,n)
            row.update(central_cut_entropy_bits=entropy,
              central_cut_normalized_entropy=entropy/(n/2),central_cut_schmidt_rank_numeric=rank)
        if topology=="local" and n<=16:
            ablated,_=simulate(build_circuit(n,layers,topology,seed,omit_final_rz_q0=True),backend,seed)
            mixed,_=simulate(build_circuit(n,layers,topology,seed,hadamard_readout=True),backend,seed)
            mixed_ab,_=simulate(build_circuit(n,layers,topology,seed,omit_final_rz_q0=True,
                                               hadamard_readout=True),backend,seed)
            row["phase_ablation_tvd_z_readout"]=tvd(np.abs(state)**2,np.abs(ablated)**2)
            row["phase_ablation_tvd_hadamard_readout_q0"]=tvd(np.abs(mixed)**2,np.abs(mixed_ab)**2)
            row["phase_ablation_fidelity"]=float(abs(np.vdot(state,ablated))**2)
        return row
    except Exception as exc:
        row["status"]="error"; row["error"]=f"{type(exc).__name__}: {exc}"[:500]
        return row

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path("results/scaling_aer"))
    parser.add_argument("--max-qubits",type=int,default=24)
    args=parser.parse_args()
    backend=AerSimulator(method="statevector",precision="double")
    backend.set_options(max_parallel_threads=2,max_parallel_experiments=1)
    rows=[]; started=time.perf_counter()
    configs=[x for x in configurations() if x[0]<=args.max_qubits]
    for i,(n,topology,layers,seed) in enumerate(configs,1):
        row=evaluate(n,topology,layers,seed,backend); rows.append(row)
        write_csv(args.output_dir/"scaling_results.csv",rows)
        print(json.dumps({"completed":i,"total":len(configs),**{
          k:row.get(k) for k in ("status","n_qubits","topology","logical_layers","seed",
            "elapsed_seconds","central_cut_entropy_bits","error")}},sort_keys=True),flush=True)
    valid=[r for r in rows if r["status"]=="ok"]; failed=[r for r in rows if r["status"]!="ok"]
    elapsed=time.perf_counter()-started; groups={}
    for row in valid: groups.setdefault((row["n_qubits"],row["topology"],row["logical_layers"]),[]).append(row)
    aggregates=[]
    keys=("elapsed_seconds","aer_transpile_seconds","aer_compiled_depth","aer_compiled_gate_count",
      "aer_compiled_cry_count","basis_transpile_seconds","basis_depth_rz_sx_x_cx",
      "basis_gate_count_rz_sx_x_cx","basis_cx_count","basis_sx_count","basis_rz_count","basis_x_count",
      "central_cut_entropy_bits","central_cut_normalized_entropy","central_cut_schmidt_rank_numeric","state_norm_error")
    for (n,topology,layers),group in sorted(groups.items()):
        rec={"n_qubits":n,"topology":topology,"logical_layers":layers,"n_replicates":len(group)}
        for key in keys:
            vals=[float(x[key]) for x in group if x.get(key) not in ("",None)]
            rec[key+"_mean"]=mean(vals) if vals else None
            rec[key+"_sd"]=pstdev(vals) if vals else None
        aggregates.append(rec)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"scaling_aggregated.json").write_text(json.dumps(aggregates,indent=2)+"\n",encoding="utf-8")
    summary={"benchmark":"Hisab Qiskit Aer statevector scaling",
      "backend":"Qiskit AerSimulator(method=statevector), CPU, double precision",
      "qiskit_version":qiskit.__version__,"qiskit_aer_version":qiskit_aer.__version__,
      "python":platform.python_version(),"platform":platform.platform(),"requested_max_qubits":args.max_qubits,
      "max_successful_qubits":max((int(r["n_qubits"]) for r in valid),default=None),
      "circuits_requested":len(configs),"circuits_succeeded":len(valid),"circuits_failed":len(failed),
      "elapsed_seconds_total":elapsed,"max_successful_runtime_seconds":max((float(r["elapsed_seconds"]) for r in valid),default=None),
      "entanglement_entropy_calculated_through_qubits":20,"phase_readout_controls":"local-only, n<=16",
      "synthetic_circuits_only":True,"hardware_execution":False,
      "claims_not_established":["natural-language representational advantage","quantum computational advantage",
       "classical intractability","BQP-hardness of natural-language QA","linguistic validity"],
      "failed_runs":[{"n_qubits":r["n_qubits"],"topology":r["topology"],"layers":r["logical_layers"],"error":r["error"]} for r in failed]}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    report=["# Hisab Qiskit Aer scaling run","",
      f"- Qiskit: {qiskit.__version__}; Aer: {qiskit_aer.__version__}",
      f"- Circuits: {len(valid)} succeeded / {len(configs)} requested",
      f"- Largest successful circuit: {summary['max_successful_qubits']} qubits",
      f"- Total wall time: {elapsed:.1f} s",
      f"- Longest successful simulation: {summary['max_successful_runtime_seconds']:.2f} s",
      "- Backend: exact CPU statevector simulation, double precision",
      "- Entropy SVD computed only through 20 qubits to bound diagnostic overhead",
      "- Synthetic operator circuits only; no hardware or linguistic validation.","",
      "## Slowest successful cases","","| Qubits | Topology | Layers | Aer runtime (s) | Aer depth | Basis depth | Basis CX count |",
      "|---:|---|---:|---:|---:|---:|---:|"]
    for r in sorted(valid,key=lambda x:float(x["elapsed_seconds"]),reverse=True)[:10]:
        report.append(f"| {r['n_qubits']} | {r['topology']} | {r['logical_layers']} | {float(r['elapsed_seconds']):.3f} | {r['aer_compiled_depth']} | {r['basis_depth_rz_sx_x_cx']} | {r['basis_cx_count']} |")
    if failed: report += ["","## Failed cases",""]+[f"- {r['n_qubits']} qubits, {r['topology']}, {r['logical_layers']} layers: {r['error']}" for r in failed]
    report += ["","## Interpretation","",
      "Statevector runtime measures this simulator and circuit family, not quantum advantage. "
      "Central-cut entropy is a state-structure diagnostic, not a complexity proof. Dense-coupling "
      "frontier cases use one logical layer to keep the scale-up progressive and reproducible.",""]
    (args.output_dir/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
