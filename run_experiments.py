import json
import os
import subprocess
import sys
import time
from datetime import datetime
import argparse
from os.path import exists

import jinja2


PROJECT_ID="test-bed-fltk-jerrit"  # TODO don't forget to change this
CLUSTER_NAME="fltk-testbed-cluster"
DEFAULT_POOL="default-node-pool"
EXPERIMENT_POOL1="fltk-pool-1"
EXPERIMENT_POOL2="fltk-pool-2"
REGION="us-central1-c"

EXPERIMENT_FILE = "./configs/distributed_tasks/current_experiment.json"
CLUSTER_CONFIG = "./configs/vision_transformer_experiment.json"

SYSTEM_SAMPLE_RATE = 5


env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath='./configs/distributed_tasks/'))
template = env.get_template('single_model_template.json.jinja')


def log(s):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"{current_time} : {s}")


# Class defining an experiment run
class Config:
    def __init__(self, pod_size, core, memory, parallelism, model, dataset, logdir):
        self.pod_size = pod_size
        self.core = core
        self.memory = memory
        self.parallelism = parallelism
        self.model = model
        self.dataset = dataset
        self.logdir = logdir

    def __str__(self):
        return self.logdir


# Manually add or remove the nodes needed
def scale_up_pool(pool_id, num):
    log(f"Scaling pool {pool_id} to {num} nodes.")
    pool = DEFAULT_POOL
    if pool_id == 1:
        pool = EXPERIMENT_POOL1
    elif pool_id == 2:
        pool = EXPERIMENT_POOL2

    cmd_str = f"gcloud container clusters resize {CLUSTER_NAME} --node-pool {pool} --num-nodes {num} --region {REGION} --quiet"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Creates a json file from template for this experiment, overwrites the old one!
def prepare_json(config):
    log(f"Creating configuration json file")
    # use jinja to generate experimental config
    output = template.render(config.__dict__)
    with open('./configs/distributed_tasks/current_experiment.json', 'w') as f:
        f.write(output)


# Adds results logger
def install_extractor():
    log(f"Installing extractor")
    extractor_path = "./charts/extractor"
    vals_path = "./charts/fltk-values.yaml"
    cmd_str = f"helm upgrade --install -n test extractor {extractor_path} -f {vals_path} \
    --set provider.projectName={PROJECT_ID}"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()
    time.sleep(10)  # give extractor time to set up


# Removes results logger
def uninstall_extractor():
    log(f"Uninstalling extractor")
    cmd_str = f"helm uninstall extractor -n test"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Adds experiment orchestrator
def install_experiment():
    log(f"Installing experiment orchestrator")
    orchestrator_dir = "./charts/orchestrator"
    vals_dir = "./charts/fltk-values.yaml"
    cmd_str = f"helm install -n test experiment-orchestrator {orchestrator_dir} -f {vals_dir} \
    --set-file orchestrator.experiment={EXPERIMENT_FILE},orchestrator.configuration={CLUSTER_CONFIG} \
    --set provider.projectName={PROJECT_ID}"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Removes experiment orchestrator
def uninstall_experiment():
    log(f"Uninstalling experiment")
    cmd_str = f"helm uninstall -n test experiment-orchestrator"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


def get_system_metrics(config):
    log(f"  Retrieving system metrics")
    cmd_str = "kubectl get nodes -n test -o=jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}'"
    res = subprocess.run(cmd_str, capture_output=True, shell=True)
    nodes = res.stdout.decode('utf-8')
    i = 0
    for node in nodes.split('\n'):
        if not node.__contains__("fltk-pool"):
            continue
        #print(f"    Getting metrics from {node}")
        cmd_str = f"kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes/{node} | jq '.usage'"
        res = subprocess.run(cmd_str, capture_output=True, shell=True).stdout.decode('utf-8')
        usage_dict = json.loads(res)
        cpu = usage_dict['cpu']
        mem = usage_dict['memory']
        # Log cpu and memory to a log file
        node_file_path = f"{config.logdir}system_metrics.csv"
        if not exists(node_file_path):
            with open(node_file_path, 'w') as f:
                f.write(f"time,node,cpu,mem\n")
        with open(node_file_path, 'a') as f:
            f.write(f"{datetime.now()},{i},{cpu},{mem}\n")
        i += 1

# Semi-busy wait for experiment to finish -_(^_^)_-
def wait_for_finish(config):
    log("Begin waiting for experiment")
    cmd_str = "kubectl get pods -n test fl-server --no-headers -o custom-columns=\":status.phase\""
    unexpected_count = 0
    while True:
        # Wait 60 seconds before checking again, measure system metrics during
        for i in range(0, SYSTEM_SAMPLE_RATE):
            get_system_metrics(config)
            time.sleep(60 / SYSTEM_SAMPLE_RATE)

        process = subprocess.run(cmd_str, capture_output=True, shell=True)
        stdout_as_str = process.stdout.decode("utf-8")
        if "Running" in stdout_as_str:
            log("fl-server is still running.")
            unexpected_count = 0
        elif "Succeeded" in stdout_as_str:
            log("fl-server finished")
            break
        elif "Failed" in stdout_as_str:
            log("fl-server failed")
            break
        else:
            log("Unexpected val!!")
            if unexpected_count >= 2:
                break
            unexpected_count += 1


# Scales up the right node pool and then runs the experiment orchestrator
def run_experiment(config):
    log("Running experiment")
    # Scale up correct node pool, scale down the other
    if config.pod_size == "small":
        scale_up_pool(2, 0)
        scale_up_pool(1, config.parallelism)
    else:
        scale_up_pool(1, 0)
        scale_up_pool(2, min(config.parallelism,3))

    # Install new orchestrator
    install_experiment()


# Calls extractor for tensorflow and kubectl for stdout logs
def collect_results(config):
    log("Collecting results")
    # Get name of extractor pod
    cmd_str = "kubectl get pods -n test -l \"app.kubernetes.io/name=fltk.extractor\" -o jsonpath=\"{.items[0].metadata.name}\""
    pod_name = subprocess.run(cmd_str, capture_output=True, shell=True).stdout.decode("utf-8")
    # Use extractor to get experiment logs
    cmd_str = f"kubectl cp -n test {pod_name}:/opt/federation-lab/logging {config.logdir}"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()
    # Get pod names of all trainjobs
    cmd_str= "kubectl get pods -n test -o=jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}'"
    res = subprocess.run(cmd_str, capture_output=True, shell=True)
    pods = res.stdout.decode('utf-8')
    i = 0
    for pod in pods.split('\n'):
        if pod.__contains__("trainjob"):
            log(f"Retrieving log from {pod}")
            cmd_str = f"kubectl logs -n test {pod} > {config.logdir}node{i}.txt"
            subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()
            i += 1


# Gets rid of trainjobs
def clean_pods():
    log("Removing kubeflow trainjobs")
    cmd_str = f"kubectl delete pytorchjobs.kubeflow.org --all-namespaces --all"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Scales down all node pools and removes any remaining pods
def clean_up(scale_down=True):
    log("Cleaning everything up")
    uninstall_experiment()
    uninstall_extractor()
    clean_pods()
    if scale_down:
        scale_up_pool(0, 0)
        scale_up_pool(1, 0)
        scale_up_pool(2, 0)


def perform_sweep(configs):
    log("Running experiment sweep")
    for i, c in enumerate(configs):
        log(f"Starting experiment: {c.core}/{c.memory} CPU/Ram, {c.parallelism} parallel, {c.model} model, {c.dataset} dataset")
        if not os.path.exists(c.logdir):
            os.makedirs(c.logdir)
        # Prepare new config
        prepare_json(c)
        # Install results logger
        install_extractor()
        # Run the experiment
        run_experiment(c)
        # Wait for it to finish
        wait_for_finish(c)
        # Get any results
        collect_results(c)
        # Clean up
        clean_up(scale_down=False)
        # Wait a bit for above actions to conclude
        time.sleep(10)

"""
0: ViTMNIST_mnist_rgb_small_1/
1: ViTFlowers_flowers_small_1/
2: EfficientNetV2MNIST_mnist_small_1/
3: EfficientNetV2Flowers_flowers_small_1/
4: ViTMNIST_mnist_rgb_small_4/
5: ViTFlowers_flowers_small_4/
6: EfficientNetV2MNIST_mnist_small_4/
7: EfficientNetV2Flowers_flowers_small_4/
8: ViTMNIST_mnist_rgb_big_1/
9: ViTFlowers_flowers_big_1/
10: EfficientNetV2MNIST_mnist_big_1/
11: EfficientNetV2Flowers_flowers_big_1/
12: ViTMNIST_mnist_rgb_big_4/
13: ViTFlowers_flowers_big_4/
14: EfficientNetV2MNIST_mnist_big_4/
15: EfficientNetV2Flowers_flowers_big_4/
"""

# ids of experiments to not run, eg. that have already been completed. Ids are in string above for reference.
#DONT_RUN_MASK = [0,2,4,6,8,10,12,14]
DONT_RUN_MASK = [0,1,2,3,4,5,6,7,8,9,10,11,13,15]

# Parameters
pod_sizes = ['small', 'big']
parallelisms = [1, 4]
models = ['ViT', 'EfficientNetV2']
datasets = ['MNIST', 'Flowers']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', default=False, action='store_true')
    args = parser.parse_args()
    if args.clean:
        log("Only performing a clean!")
        clean_up(scale_down=False)
        sys.exit(0)
    configs = []
    for pod_size in pod_sizes:
        for parallelism in parallelisms:
            for model in models:
                for dataset in datasets:
                    if pod_size == 'small':
                        core = 1
                        memory = '4Gi'
                    else:
                        core = 2
                        memory = '4Gi'
                    # Edge case to choose mnist_rgb for specific ViT, MNIST combination
                    model_combined = model + dataset
                    if dataset == "MNIST":
                        dataset_adj = "MNIST_RGB"
                    else:
                        dataset_adj = dataset
                    logdir = f"./logging/{model_combined}_{dataset_adj.lower()}_{pod_size}_{parallelism}/"
                    configs.append(Config(pod_size, core, memory, parallelism, model_combined, dataset_adj.lower(), logdir))
    clean_up(scale_down=False)
    scale_up_pool(0, 1)  # start default pool
    # Start running from 'resume' index if need to skip some experiments
    to_run = []
    for i,c in enumerate(configs):
        if i not in DONT_RUN_MASK:
            to_run.append(c)
    perform_sweep(to_run)
    clean_up()
