import numpy as np

PARAMETER_COUNT = 1_040_324
PARAMETER_BYTES = 4 * PARAMETER_COUNT


def flops(image_size, batch):
    s, b = np.asarray(image_size, float), np.asarray(batch, float)
    return 17714 * b * s**2 + 313700 * b


def memory(image_size, batch):
    s, b = np.asarray(image_size, float), np.asarray(batch, float)
    return PARAMETER_BYTES + 68 * b * s**2


def layer_costs(image_size, batch):
    s, b = np.broadcast_arrays(np.asarray(image_size, float), np.asarray(batch, float))
    q = b * s**2

    convs = [(3, 8, 4704, 2352), (2, 4, 51200, 6400),
             (4, 2, 73728, 2304), (2, 4, 32768, 1024),
             (4, 1, 589824, 4608), (1, 2, 131072, 1024)]
    costs = []
    for i, (inp, out, weights, coefficient) in enumerate(convs, 1):
        costs.append(dict(name=f"conv{i}", flops=coefficient*q, bytes=4*((inp+out)*q+weights)))
        costs.append(dict(name=f"relu{i}", flops=np.zeros_like(q), bytes=8*out*q))
        if i == 1:
            costs.append(dict(name="maxpool", flops=np.zeros_like(q), bytes=56*q))
    costs.extend([
        dict(name="global_average_pool", flops=2*q, bytes=8*q+2048*b),
        dict(name="linear1", flops=262400*b, bytes=3072*b+525312),
        dict(name="relu_head", flops=np.zeros_like(q), bytes=2048*b),
        dict(name="linear2", flops=51300*b, bytes=1424*b+102800),
    ])
    return costs


def bytes_moved(image_size, batch):
    s, b = np.asarray(image_size, float), np.asarray(batch, float)
    return PARAMETER_BYTES + 380*b*s**2 + 8592*b


def latency(image_size, batch, theta):
    compute = theta["compute_flops_per_s"]
    bandwidth = theta["bandwidth_bytes_per_s"]
    costs = layer_costs(image_size, batch)
    return theta["overhead_s"] + sum(np.maximum(c["flops"]/compute, c["bytes"]/bandwidth) for c in costs)


def energy(image_size, batch, theta_energy):
    return (theta_energy["offset_j"]
            + theta_energy["base_power_w"] * latency(image_size, batch, theta_energy["latency"])
            + theta_energy["joules_per_flop"] * flops(image_size, batch)
            + theta_energy["joules_per_byte"] * bytes_moved(image_size, batch))
