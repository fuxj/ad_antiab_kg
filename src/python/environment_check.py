# -*- coding: utf-8 -*-
"""环境自检: 逐个 import 关键库并打印版本; 缺失的会汇总在最后。

用法:
    cd $PROJ
    python src/python/environment_check.py

项目根目录的确定顺序:
    1) 环境变量 PROJ
    2) 按脚本自身位置推断 (src/python/xxx.py 的上两级)
"""
import importlib
import os
import pathlib
import sys

PROJ = os.environ.get("PROJ") or str(
    pathlib.Path(__file__).resolve().parents[2]
)

MODULES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("torch_geometric", "torch-geometric"),
    ("rdkit", "rdkit"),
    ("shap", "shap"),
    ("captum", "captum"),
    ("optuna", "optuna"),
    ("esm", "fair-esm"),
    ("sentence_transformers", "sentence-transformers"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("synapseclient", "synapseclient"),
    ("neo4j", "neo4j"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]


def main():
    ok, missing, missing_pkgs = [], [], []

    for mod, pkg in MODULES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok.append("[OK]  {:<24} {}".format(pkg, ver))
        except Exception as e:
            missing.append("[X ]  {:<24} {}: {}".format(pkg, type(e).__name__, e))
            missing_pkgs.append(pkg)

    # GPU 可用性
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU"
        ok.append("[OK]  cuda available={}, device={}".format(cuda, dev))
    except Exception:
        pass

    # token 配置检查
    envp = pathlib.Path(PROJ) / ".env"
    tokens = {}
    if envp.exists():
        for line in envp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                tokens[k.strip()] = bool(v.strip())
    else:
        print("[提示] 未找到 {}, token 未检查".format(envp))

    report = "\n".join(ok + missing)
    if tokens:
        report += "\n--- tokens ---\n" + "\n".join(
            "[{}] {}".format("OK" if v else "空", k) for k, v in tokens.items()
        )
    report += "\n--- python {} ---\n".format(sys.version.replace("\n", " "))
    print(report)

    out = pathlib.Path(PROJ) / "logs" / "env_check_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print("\n已写入 {}".format(out))

    if missing_pkgs:
        print("\n缺失包请执行:")
        print("  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple " + " ".join(missing_pkgs))


if __name__ == "__main__":
    main()
