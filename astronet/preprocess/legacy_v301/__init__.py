"""Frozen copy of the astronet v3.0.1 preprocessing ("legacy_v301").

The production AstroNet-Triage model (and the previous vetting model, cshallue_20250429) were
trained on records made by this code, so `generate_input_records.create(mode="triage")`
dispatches here. Copied verbatim from tag v3.0.1; the only edits are import lines, so the
modules resolve inside this package instead of the (since changed) shared light_curve_util /
astronet.preprocess modules. Do not modify: a change here changes production triage inputs.
"""
