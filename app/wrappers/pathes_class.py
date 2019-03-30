import yaml

class repoConfig:
    def __init__(self):
        with open('app/paths.yml', 'r') as f:
           self.paths = yaml.load(f)


if __name__ == "__main__":
    repo_config = repoConfig()
