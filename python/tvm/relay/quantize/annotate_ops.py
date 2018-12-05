from __future__ import absolute_import
from .. import expr as _expr
from .quantize import QFieldKind, QFieldExpr, register_qfield_rewrite
from .quantize import attach_simulated_quantize, get_current_qconfig


def _forward_op(ref_call, args):
    return _expr.Call(ref_call.op, args,
                      ref_call.attrs, ref_call.type_args)


@register_qfield_rewrite("nn.conv2d")
def conv2d_rewrite(ref_call, new_args, ctx):
    cfg = get_current_qconfig()
    if cfg.counter < cfg.skip_k_conv:
        cfg.counter += 1
        return None
    cfg.counter += 1

    lhs, rhs = new_args
    if isinstance(lhs, QFieldExpr):
        lhs_expr = lhs.expr
        if lhs.kind != QFieldKind.INPUT:
            lhs_expr = attach_simulated_quantize(lhs_expr, QFieldKind.INPUT)
    else:
        lhs_expr = attach_simulated_quantize(lhs, QFieldKind.INPUT)

    assert not isinstance(rhs, QFieldExpr)
    rhs_expr = attach_simulated_quantize(rhs, QFieldKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    return QFieldExpr(expr, QFieldKind.ACTIVATION)


@register_qfield_rewrite("multiply")
def multiply_rewrite(ref_call, new_args, ctx):
    cfg = get_current_qconfig()
    if cfg.counter <= cfg.skip_k_conv:
        return None

    lhs, rhs = new_args
    if not isinstance(lhs, QFieldExpr) and not isinstance(rhs, QFieldExpr):
        return None
    elif lhs.kind == QFieldKind.ACTIVATION and not isinstance(rhs, QFieldExpr):
        lhs_expr = attach_simulated_quantize(lhs.expr, QFieldKind.INPUT)
        rhs_expr = attach_simulated_quantize(rhs.expr, QFieldKind.WEIGHT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QFieldExpr(expr, QFieldKind.ACTIVATION)
    else:
        raise ValueError


@register_qfield_rewrite("add")
def add_rewrite(ref_call, new_args, ctx):
    cfg = get_current_qconfig()
    if cfg.counter <= cfg.skip_k_conv:
        return None

    lhs, rhs = new_args
    if not isinstance(lhs, QFieldExpr) and not isinstance(rhs, QFieldExpr):
        # on float domain
        return None
    elif not isinstance(lhs, QFieldExpr) and rhs.kind == QFieldKind.ACTIVATION:
        # addition for residual, but lhs are calculated on real domain
        lhs_expr = attach_simulated_quantize(lhs, QFieldKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs.expr])
        return QFieldExpr(expr, QFieldKind.ACTIVATION)
    elif lhs.kind == QFieldKind.ACTIVATION and not isinstance(rhs, QFieldExpr):
        # the most common situation, e.g. bias add in bn
        rhs_expr = attach_simulated_quantize(rhs, QFieldKind.WEIGHT)
        expr = _forward_op(ref_call, [lhs.expr, rhs_expr])
        return QFieldExpr(expr, QFieldKind.ACTIVATION)
    elif lhs.kind == QFieldKind.INPUT and rhs.kind == QFieldKind.ACTIVATION:
        # addition for residual, but lhs are muti-refered
        expr = _forward_op(ref_call, [lhs.expr, rhs.expr])
        return QFieldExpr(expr, QFieldKind.ACTIVATION)
    elif lhs.kind == QFieldKind.ACTIVATION and rhs.kind == QFieldKind.ACTIVATION:
        # addition for residual
        expr = _forward_op(ref_call, [lhs.expr, rhs.expr])
        return QFieldExpr(expr, QFieldKind.ACTIVATION)
    else:
        raise ValueError


@register_qfield_rewrite("nn.relu")
def relu_rewrite(ref_call, new_args, ctx):
    cfg = get_current_qconfig()
    if cfg.counter <= cfg.skip_k_conv:
        return None

    x = new_args[0]
    if isinstance(x, QFieldExpr):
        expr = _forward_op(ref_call, [x.expr])
        return QFieldExpr(expr, x.kind)
    else:
        return None
