import subprocess
import time
import jinja2

resume = 1

PROJECT_ID="test-bed-fltk-jerrit"
CLUSTER_NAME="fltk-testbed-cluster"
DEFAULT_POOL="default-node-pool"
EXPERIMENT_POOL1="fltk-pool-1"
EXPERIMENT_POOL2="fltk-pool-2"
REGION="us-central1-c"

EXPERIMENT_FILE = "./configs/distributed_tasks/current_experiment.json"
CLUSTER_CONFIG = "./configs/vision_transformer_experiment.json"

# Parameters
node_pools = ['small', 'big']
parallelisms = [1, 4]
models = ['ViT', 'EfficientNetV2']
datasets = ['MNIST', 'Flowers']


env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath='./configs/distributed_tasks/'))
template = env.get_template('single_model_template.json.jinja')


# Class defining an experiment run
class Config:
    def __init__(self, node_pool, core, memory, parallelism, model, dataset):
        self.node_pool = node_pool
        self.core = core
        self.memory = memory
        self.parallelism = parallelism
        self.model = model
        self.dataset = dataset


# Manually add or remove the nodes needed
def scale_up_pool(pool_id, num):
    print(f"Scaling pool {pool_id} to {num} nodes.")
    pool = DEFAULT_POOL
    if pool_id == 1:
        pool = EXPERIMENT_POOL1
    elif pool_id == 2:
        pool = EXPERIMENT_POOL2

    cmd_str = f"gcloud container clusters resize {CLUSTER_NAME} --node-pool {pool} --num-nodes {num} --region {REGION} --quiet"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Creates a json file from template for this experiment, overwrites the old one!
def prepare_json(config):
    print(f"Creating configuration json file")
    # use jinja to generate experimental config
    output = template.render(config.__dict__)
    with open('./configs/distributed_tasks/current_experiment.json', 'w') as f:
        f.write(output)


# Adds results logger
def install_extractor():
    print(f"Installing extractor")
    extractor_path = "./charts/extractor"
    vals_path = "./charts/fltk-values.yaml"
    cmd_str = f"helm upgrade --install -n test extractor {extractor_path} -f {vals_path} \
    --set provider.projectName={PROJECT_ID}"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Removes results logger
def uninstall_extractor():
    print(f"Uninstalling extractor")
    cmd_str = f"helm uninstall extractor -n test"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Adds experiment orchestrator
def install_experiment():
    print(f"Installing experiment orchestrator")
    orchestrator_dir = "./charts/orchestrator"
    vals_dir = "./charts/fltk-values.yaml"
    cmd_str = f"helm install -n test experiment-orchestrator {orchestrator_dir} -f {vals_dir} \
    --set-file orchestrator.experiment={EXPERIMENT_FILE},orchestrator.configuration={CLUSTER_CONFIG} \
    --set provider.projectName={PROJECT_ID}"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Removes experiment orchestrator
def uninstall_experiment():
    print(f"Uninstalling experiment")
    cmd_str = f"helm uninstall -n test experiment-orchestrator"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Semi-busy wait for experiment to finish -_(^_^)_-
def wait_for_finish():
    print("Begin waiting for experiment")
    cmd_str = "kubectl get pods -n test fl-server --no-headers -o custom-columns=\":status.phase\""
    while True:
        time.sleep(60)  # Wait 60 seconds before checking again.

        process = subprocess.run(cmd_str, capture_output=True, shell=True)
        stdout_as_str = process.stdout.decode("utf-8")
        if "Running" in stdout_as_str:
            print("fl-server is still running.")
        elif "Succeeded" in stdout_as_str:
            print("fl-server finished")
            break
        elif "Failed" in stdout_as_str:
            print("fl-server failed")
            break
        else:
            print("Unexpected val!!")
            break


# Scales up the right node pool and then runs the experiment orchestrator
def run_experiment(config):
    print("Running experiment")
    # Scale up correct node pool, scale down the other
    if config.node_pool == "small":
        scale_up_pool(1, config.core)
        scale_up_pool(2, 0)
    else:
        scale_up_pool(2, config.core)
        scale_up_pool(1, 0)

    # Install new orchestrator
    install_experiment()


# Calls extractor for tensorflow and kubectl for stdout logs
def collect_results(config):
    print("Collecting results")
    # Get name of extractor pod
    experiment_logdir = f"./logging/{config.model}_{config.dataset}_{config.node_pool}_{config.parallelism}"
    cmd_str = "kubectl get pods -n test -l \"app.kubernetes.io/name=fltk.extractor\" -o jsonpath=\"{.items[0].metadata.name}\""
    pod_name = subprocess.run(cmd_str, capture_output=True, shell=True).stdout.decode("utf-8")
    # Use extractor to get experiment logs
    cmd_str = f"kubectl cp -n test {pod_name}:/opt/federation-lab/logging {experiment_logdir}"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()
    # Get pod names of all trainjobs
    cmd_str= "kubectl get pods -n test -o=jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}'"
    res = subprocess.run(cmd_str, capture_output=True, shell=True)
    pods = res.stdout.decode('utf-8')
    i = 0
    for pod in pods.split('\n'):
        if pod.__contains__("trainjob"):
            print(f"Retrieving log from {pod}")
            cmd_str = f"kubectl logs -n test {pod} > {experiment_logdir}/node{i}.txt"
            subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()
            i += 1


# Gets rid of trainjobs
def clean_pods():
    print("Removing kubeflow trainjobs")
    cmd_str = f"kubectl delete pytorchjobs.kubeflow.org --all-namespaces --all"
    subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE).communicate()


# Scales down all node pools and removes any remaining pods
def clean_up(scale_down=True):
    print("Cleaning everything up")
    uninstall_experiment()
    uninstall_extractor()
    clean_pods()
    if scale_down:
        scale_up_pool(0, 0)
        scale_up_pool(1, 0)
        scale_up_pool(2, 0)


def perform_sweep(configs):
    print("Running experiment sweep")
    for i, c in enumerate(configs):
        print(f"Starting experiment: {c.core}/{c.memory} CPU/Ram, {c.parallelism} parallel, {c.model} model, {c.dataset} dataset")
        # Prepare new config
        prepare_json(c)
        # Install results logger
        install_extractor()
        # Run the experiment
        run_experiment(c)
        # Wait for it to finish
        wait_for_finish()
        # Get any results
        collect_results(c)
        # Remove the orchestrator
        uninstall_experiment()
        # Remove results logger
        uninstall_extractor()
        # Get rid of trainjobs
        clean_pods()
        if i == 1:
            break # run only first two experiments for testing


if __name__ == "__main__":
    configs = []
    for node_pool in node_pools:
        for parallelism in parallelisms:
            for model in models:
                for dataset in datasets:
                    if node_pool == 'small':
                        core = 1
                        memory = '4Gi'
                    else:
                        core = 2
                        memory = '4Gi'
                    # Edge case to choose mnist_rgb for specific ViT, MNIST combination
                    model_combined = model + dataset
                    if model_combined == "ViTMNIST":
                        dataset_adj = "MNIST_rgb"
                    else:
                        dataset_adj = dataset
                    configs.append(Config(node_pool, core, memory, parallelism, model_combined, dataset_adj.lower()))
    clean_up(scale_down=False)
    scale_up_pool(0, 1)  # start default pool
    install_extractor()
    # Start running from 'resume' index if need to skip some experiments
    perform_sweep(configs[resume:])
    clean_up()
