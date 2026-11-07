"""Experimental Java API smoke check, not a validated VEGA batch adapter.

Uses public methods recovered from the installed JAR. Preserves raw results,
status codes and domain assessment rather than guessing classification semantics.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from toxicology import canonicalize_smiles


def main():
    jar = ROOT / "external_tools" / "VEGA GUI 1.2.6" / "Vega-GUI-1.2.6.jar"
    gateway = JavaGateway(gateway_parameters=GatewayParameters(
        port=launch_gateway(classpath=str(jar), die_on_exit=True), read_timeout=90))
    report = {"scope": "Experimental API smoke only; not validated accuracy or regulatory assessment",
              "jar_sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
              "model_class": "insilico.mutagenicity_sarpy.ismMutagenicitySarpy", "results": []}
    try:
        model = gateway.jvm.insilico.mutagenicity_sarpy.ismMutagenicitySarpy()
        info = model.getInfo()
        report["model"] = {"name": info.getName(), "version": info.getVersion(),
                           "key": info.getKey(), "class_values_raw": str(info.getClassValues()),
                           "qmrf": info.getQMRF(), "summary": info.getSummary()}
        report["result_names"] = list(model.GetResultsName())
        report["limitation"] = "Default VEGA assessment may prioritize an experimental training/reference value. This smoke is not independent predictive validation."
        output_class = gateway.jvm.insilico.core.model.InsilicoModelOutput
        report["output_status_constants"] = {
            name: getattr(output_class, name) for name in
            ("OUTPUT_OK", "OUTPUT_OK_AD_NOT_APPLICABLE", "OUTPUT_OK_AD_MISSING",
             "OUTPUT_NOT_CALCULATED", "OUTPUT_ERROR")}
        for name, raw in (("ethanol", "CCO"), ("epichlorohydrin", "ClCC1CO1"),
                          ("NDMA", "CN(C)N=O")):
            smiles = canonicalize_smiles(raw)
            row = {"id": name, "canonical_smiles": smiles}
            try:
                # This converter is referenced by VEGA's bundled mainScriptSarpy example.
                molecule = gateway.jvm.insilico.core.molecule.conversion.SmilesMolecule.Convert(smiles)
                molecule.SetId(name)
                row["parsed_valid"] = molecule.IsValid()
                if not row["parsed_valid"]:
                    row["error"] = molecule.GetErrors().GetMessages()
                else:
                    output = model.Execute(molecule)
                    value = float(output.getMainResultValue())
                    row.update({"status_code": output.getStatus(),
                                "results_raw": list(output.getResults()),
                                "main_value_raw": value if math.isfinite(value) else None,
                                "assessment": output.getAssessment(),
                                "assessment_verbose": output.getAssessmentVerbose(),
                                "error": output.getErrMessage(),
                                "returned_smiles": output.getMoleculeSMILES(),
                                "returned_id": output.getMoleculeId(),
                                "has_experimental": output.HasExperimental(),
                                "experimental_value_raw": output.getExperimentalFormatted() if output.HasExperimental() else None,
                                "adi": None if output.getADI() is None else {
                                    "name": output.getADI().GetIndexName(),
                                    "value": output.getADI().GetIndexValue(),
                                    "assessment": output.getADI().GetAssessment(),
                                    "assessment_class": output.getADI().GetAssessmentClassAsString()},
                                "ad_index_count": output.getADIndexSize()})
            except Exception as exc:
                row["error"] = str(exc)[:3000]
            report["results"].append(row)
    except Exception as exc:
        report["initialization_error"] = str(exc)[:4000]
    finally:
        destination = ROOT / "evaluation" / "results" / "vega_ames_smoke.json"
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        gateway.shutdown()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
