# :(

# Parameters
node_pools = ['small', 'big']
parallelisms = [1, 4]
models = ['ViT', 'EfficientNetV2']
datasets = ['MNIST', 'Flowers']


class Config:
    def __init__(self, node_pool, parallelism, model, dataset):
        self.node_pool = node_pool
        self.parallelism = parallelism
        self.model = model
        self.dataset = dataset


def prepare_json():
    json_path = ''
    # use jinja to generate experimental config


def run_experiment():
    pass


def perform_sweep():
    for node_pool in node_pools:
        for parallelism in parallelisms:
            for model in models:
                for dataset in datasets:
                    print(f"Starting experiment: {node_pool},{parallelism},{model},{dataset}")
                    pass
                    # clear previous experimental data
                    # setup setting json (with 1 experiment, likely use templating and jinja2)
                    #


if __name__ == "__main__":
    configs = []
    for node_pool in node_pools:
        for parallelism in parallelisms:
            for model in models:
                for dataset in datasets:
                        configs.append(Config(node_pool,parallelism,model,dataset))
    perform_sweep()