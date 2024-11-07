import ipdb
output = []

with open("suite2p310_requirements.txt", "r") as file:
    for line in file:
        # Skip comments or empty lines
        if line.startswith("#") or not line.strip():
            continue
        # Extract the package and version (ignoring the build number)
        package_version = line.split('=')[:2]  # ['numpy', '1.23.5']
        output.append("==".join(package_version))

# Print out dependencies for `setup.py` format
print("install_requires=[")
for i,dep in enumerate(output):
    if i%2==0:
        print(f"'{dep}',")
print("]")