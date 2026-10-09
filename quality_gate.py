"""ML quality gate: fail the pipeline if model accuracy is below the threshold."""
import json
import sys

REQUIRED_ACCURACY = 0.8

with open("metrics.json") as f:
    accuracy = json.load(f)["accuracy"]

print(f"Model accuracy: {accuracy}")
print(f"Required accuracy: {REQUIRED_ACCURACY}")

if accuracy >= REQUIRED_ACCURACY:
    print("QUALITY GATE PASSED")
else:
    print("QUALITY GATE FAILED")
    sys.exit(1)
