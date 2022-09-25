## Docker Shenanigans
Build docker image and push to GCR
```bash
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 . --tag gcr.io/$PROJECT_ID/fltk
docker push gcr.io/$PROJECT_ID/fltk
```

## Kubectl Helm
remove orchestrator
```bash
helm uninstall -n test orchestrator
```
remove old experiments
```bash
kubectl delete pytorchjobs.kubeflow.org --all --all-namespaces
```
reinstall orchestrator
```bash
helm install orchestrator ./orchestrator -f fltk-values.yaml -n test
```

delete logged data
```bash
kubectl exec -n test fl-extractor --  rm -rf logging/*
```