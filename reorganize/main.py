import pkg_resources
import subprocess

for dist in pkg_resources.working_set:
    print(f"Upgrading {dist.project_name}...")
    subprocess.run(["pip", "install", "--upgrade", dist.project_name])