import json
import random

def random_hex_color():
    return "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])

with open("dependency.txt") as f:
    raw = f.read()

bits = raw.split("\n\n")

known_nodes: list[str] = []
edges: list[tuple[str, str]] = []

for bit in bits:
    lines = bit.split("\n")
    if len(lines) == 0 or len(lines) > 2:
        raise ValueError("Empty bit")
    else:
        if not ": " in lines[0]:
            if lines[0].endswith(":"):
                lines[0] = lines[0] + " "
            else:
                raise ValueError(f"Malformed line, `{lines[0]}`")
        tos, froms = lines[0].split(": ")
        froms: str
        tos: str
        froms: list[str] = [v for v in froms.split(" ") if v != ""]
        tos: list[str] = [v for v in tos.split(" ") if v != ""]

        process = lines[1] if len(lines) > 1 else None

        for node in froms + tos:
            if node not in known_nodes:
                known_nodes.append(node)

        if process is not None:
            process = process.strip()
            if process not in known_nodes:
                known_nodes.append(process)
            for f in froms:
                edges.append((f, process))
            for t in tos:
                edges.append((process, t))
        else:
            for f in froms:
                for t in tos:
                    edges.append((f, t))

if "" in known_nodes:
    raise ValueError("Empty node name")

out_nodes = [{"name": node, "color": random_hex_color()} for node in known_nodes]
out_edges = [{"from": source, "to": target, "type": "direct"} for source, target in edges]

out = {
    "root": "rawtiles", #rawtiles
    "nodes": out_nodes,
    "edges": out_edges
}

with open("dependency.json", "w") as f:
    json.dump(out, f, indent=4)