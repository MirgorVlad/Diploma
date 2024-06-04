import yaml
from kubernetes import client, config
import os

environment = "seldon"
seldon_deployment = """
    apiVersion: machinelearning.seldon.io/v1alpha2
    kind: SeldonDeployment
    metadata:
      name: wines-classifier
      namespace: seldon
    spec:
      predictors:
      - graph:
          children: []
          implementation: MLFLOW_SERVER
          modelUri: gs://mirgor_model/mnt/models/model.pkl
          name: wines-classifier
        name: model-a
        replicas: 1
        traffic: 100
        componentSpecs:
        - spec:
            containers:
            - name: wines-classifier
              livenessProbe:
                initialDelaySeconds: 60
                failureThreshold: 100
                periodSeconds: 5
                successThreshold: 1
                httpGet:
                  path: /health/ping
                  port: http
                  scheme: HTTP
              readinessProbe:
                initialDelaySeconds: 60
                failureThreshold: 100
                periodSeconds: 5
                successThreshold: 1
                httpGet:
                  path: /health/ping
                  port: http
                  scheme: HTTP
"""

CUSTOM_RESOURCE_INFO = dict(
    group="machinelearning.seldon.io",
    version="v1alpha2",
    plural="seldondeployments",
)

def deploy_model(model_uri: str, namespace: str = "seldon"):
    try:
        # Try to load in-cluster config
        config.load_incluster_config()
        print("Loaded in-cluster config")
    except config.ConfigException:
        # Fall back to kube config
        config.load_kube_config()
        print("Loaded kube config")

    custom_api = client.CustomObjectsApi()

    dep = yaml.safe_load(seldon_deployment)
    dep["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

    try:
        resp = custom_api.create_namespaced_custom_object(
            **CUSTOM_RESOURCE_INFO,
            namespace=namespace,
            body=dep,
        )
        print("Created new Seldon deployment")
    except client.rest.ApiException as e:
        if e.status == 409:
            existing_deployment = custom_api.get_namespaced_custom_object(
                **CUSTOM_RESOURCE_INFO,
                namespace=namespace,
                name=dep["metadata"]["name"],
            )
            existing_deployment["spec"]["predictors"][0]["graph"]["modelUri"] = model_uri

            resp = custom_api.replace_namespaced_custom_object(
                **CUSTOM_RESOURCE_INFO,
                namespace=namespace,
                name=existing_deployment["metadata"]["name"],
                body=existing_deployment,
            )
            print("Updated existing Seldon deployment")
        else:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    deploy_model("gs://mirgor_model/mnt/models/model.pkl")
