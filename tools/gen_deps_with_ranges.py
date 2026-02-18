# gen_deps_with_ranges.py
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError

def upper_bound(ver: str) -> str:
    parts = ver.split(".")
    # 处理 0.x：上界建议卡到 <0.(minor+1)
    major = int(parts[0]) if parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    if major == 0:
        return f"0.{minor+1}"
    return str(major + 1)

req = Path("requirements.in").read_text(encoding="utf-8").splitlines()

deps = []
for line in req:
    name = line.strip()
    if not name or name.startswith("#"):
        continue
    # 去掉任何已有约束（只取包名）
    pkg = name.split(";")[0].strip()
    pkg = pkg.split()[0].strip()
    pkg = pkg.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("!=")[0].strip()

    try:
        v = pkg_version(pkg)
        ub = upper_bound(v)
        deps.append(f"{pkg}>={v},<{ub}")
    except PackageNotFoundError:
        # 当前环境没装到的包：先不加版本范围
        deps.append(pkg)

print("dependencies = [")
for d in deps:
    print(f'  "{d}",')
print("]")
