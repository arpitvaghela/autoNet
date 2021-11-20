from numpy import generic
from darts import search, utils
from darts.models import ops
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import functools


def get_best_from_genotype(genotype, k=3):
    count = {}
    for gene in genotype.normal:
        for x, _ in gene:
            if not x in count:
                count.update({x: 1})
            else:
                count.update({x: count[x] + 1})
    return sorted(count, key=lambda x: count[x], reverse=True)[:k]


def compare_primitives():
    p1 = [
        "max_pool_3x3",
        # "avg_pool_3x3",
        "skip_connect",  # identity
        "sep_conv_3x3",
        # "sep_conv_5x5",
        # "sep_conv_7x7",
        "dil_conv_3x3",
        # "dil_conv_5x5",
        "none",
    ]
    p2 = [
        # "max_pool_3x3",
        "avg_pool_3x3",
        # "skip_connect",  # identity
        # "sep_conv_3x3",
        "sep_conv_5x5",
        "sep_conv_7x7",
        # "dil_conv_3x3",
        "dil_conv_5x5",
        "none",
    ]
    search.set_primitives(p1)
    genotype1 = search.search("catdogs", "custom", 2, epochs=1, dataid="1637157378441")
    p3 = get_best_from_genotype(genotype1)
    search.set_primitives(p2)
    genotype2 = search.search("catdogs", "custom", 2, epochs=1, dataid="1637157378441")
    # new genotype from best for previous
    p3 += get_best_from_genotype(genotype2)
    p3 += ["none"]
    print(p1, p2, p3)
    search.set_primitives(p3)
    search.search("catdogs", "custom", 2, epochs=1, dataid="1637157378441")


if __name__ == "__main__":
    # genotype = search.search("catdogs", "custom", 2, epochs=1, dataid="1637157378441")
    compare_primitives()
