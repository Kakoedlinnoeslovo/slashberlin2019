import yaml

class repoConfig:
    def __init__(self):
        with open('/storage/slashberlin2019/app/paths.yml', 'r') as f:
           self.paths = yaml.load(f)


if __name__ == "__main__":
    repo_config = repoConfig()
