# Released uniHAR checkpoint evaluation

The authored evaluation scripts load the author checkpoint strictly and retain its original batch/sensor pooling semantics. setup_runtime() fixes file paths, the released CSV dictionary syntax, and compatibility initialization before strict loading. Official released-test and exploratory external-data measurements are separate from the newly trained canonical CDNet adapter; the released IMU checkpoint does not establish canonical held-out training provenance.

The public fetcher retrieves code only by default. Released test arrays and checkpoint files must be obtained separately from the original repository. The canonical launcher builds its runtime from these code files after its required weights are present.
