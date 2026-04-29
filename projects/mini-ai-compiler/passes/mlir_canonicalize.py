from __future__ import annotations

import operator

from ir.node import Node
from passes.base import Pass
from passes.rewrite import GreedyPatternRewriteDriver, PatternRewriter, RewritePattern


class FoldConstBinaryPattern(RewritePattern):
    root_op_type: str | None = None
    evaluator = None

    def match_and_rewrite(self, node: Node, rewriter: PatternRewriter) -> bool:
        if len(node.inputs) != 2:
            return False
        if not all(value.is_constant for value in node.inputs):
            return False
        lhs = node.inputs[0].producer.attrs["value"]
        rhs = node.inputs[1].producer.attrs["value"]
        rewriter.replace_op_with_constant(node, self.evaluator(lhs, rhs))
        return True


class FoldAddConstPattern(FoldConstBinaryPattern):
    root_op_type = "add"
    evaluator = staticmethod(operator.add)


class FoldMulConstPattern(FoldConstBinaryPattern):
    root_op_type = "mul"
    evaluator = staticmethod(operator.mul)


class FoldAddZeroPattern(RewritePattern):
    root_op_type = "add"

    def match_and_rewrite(self, node: Node, rewriter: PatternRewriter) -> bool:
        if len(node.inputs) != 2:
            return False
        lhs, rhs = node.inputs
        if lhs.is_constant and _is_zero(lhs.producer.attrs["value"]):
            rewriter.graph.replace_all_uses(node.outputs[0], rhs)
            rewriter.erase_op(node)
            return True
        if rhs.is_constant and _is_zero(rhs.producer.attrs["value"]):
            rewriter.graph.replace_all_uses(node.outputs[0], lhs)
            rewriter.erase_op(node)
            return True
        return False


class FoldMulOnePattern(RewritePattern):
    root_op_type = "mul"

    def match_and_rewrite(self, node: Node, rewriter: PatternRewriter) -> bool:
        if len(node.inputs) != 2:
            return False
        lhs, rhs = node.inputs
        if lhs.is_constant and _is_one(lhs.producer.attrs["value"]):
            rewriter.graph.replace_all_uses(node.outputs[0], rhs)
            rewriter.erase_op(node)
            return True
        if rhs.is_constant and _is_one(rhs.producer.attrs["value"]):
            rewriter.graph.replace_all_uses(node.outputs[0], lhs)
            rewriter.erase_op(node)
            return True
        return False


def _is_zero(value) -> bool:
    try:
        return (value == 0).all()
    except AttributeError:
        return value == 0


def _is_one(value) -> bool:
    try:
        return (value == 1).all()
    except AttributeError:
        return value == 1


class MLIRCanonicalizePass(Pass):
    def __init__(self) -> None:
        self.driver = GreedyPatternRewriteDriver(
            [
                FoldAddConstPattern(),
                FoldMulConstPattern(),
                FoldAddZeroPattern(),
                FoldMulOnePattern(),
            ]
        )

    def run(self, graph) -> bool:
        return self.driver.run(graph)
