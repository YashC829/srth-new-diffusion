# How to use RunAI

Frank created a box link for getting started in CLI [here](jhuapl.box.com/s/obi6g6q8zzj0h8245uhkrz7a1vz9mnls)

# This will run the container:

```bash
runai workspace submit surpass-devel-1 \
  -p surpass-2026 \
  -i docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1 \
  -g 1 \
  --existing-pvc claimname=surpass-2026,path=/home/apluser \
  --extended-resource nvidia.com/hostdev=1 \
  --node-pools dgx-h100-80gb \
  --large-shm \
  --run-as-user \
  --allow-privilege-escalation \
  -- \
  sleep infinity
```

# You can connect with:

```bash
kubectl exec -it surpass-devel-1-0-0 -n runai-surpass-2026 -- bash
```
