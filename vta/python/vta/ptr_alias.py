"""Ehanced Utility to support pointer alias."""
import tvm

def reinterpret(src, shape, dtype):
    """Reinterepet the input as tensor shape, dtype

    Parameters
    ----------
    shape : tuple
        The shape of the input

    dtype : str
        The data type

    src : Tensor
        The source tensor

    Returns
    -------
    dst : Tensor
        The result tensor.
    """
    return tvm.extern(
        shape, [src], lambda ins, outs: tvm.call_extern(
            "int32", "vta_ptr_alias", ins[0].data, outs[0].data),
        name="%s_alias" % src.op.name,
        dtype=dtype)


def lower_ptr_alias(stmt):
    """Pass to lower alias"""
    alloc_vars = set()
    alias_list = []

    def _prep(op):
        if isinstance(op, tvm.stmt.Allocate):
            alloc_vars.add(op.buffer_var)
        elif isinstance(op, tvm.expr.Call) and op.name == "vta_ptr_alias":
            alias_list.append((op.args[0], op.args[1]))
            return tvm.const(0, "int32")
        return op

    stmt = tvm.ir_pass.IRTransform(stmt, None, _prep, ["Allocate", "Call"])
    vmap = {}
    alias_set = set()
    # compute the alias map
    for k, v in alias_list:
        assert k not in alias_set
        assert v not in alias_set
        alias_set.add(k)
        alias_set.add(v)
        if not v in alloc_vars:
            vmap[k] = v
        else:
            vmap[v] = k

    def _subst(op):
        if isinstance(op, tvm.stmt.Allocate) and op.buffer_var in vmap:
            return op.body
        elif isinstance(op, tvm.stmt.AttrStmt) and op.node in vmap:
            return op.body
        elif isinstance(op, tvm.expr.Load) and op.buffer_var in vmap:
            return tvm.make.Load(op.dtype, vmap[op.buffer_var], op.index, op.predicate)
        elif isinstance(op, tvm.stmt.Store) and op.buffer_var in vmap:
            return tvm.make.Store(vmap[op.buffer_var], op.value, op.index, op.predicate)
        elif isinstance(op, tvm.expr.Var) and op in vmap:
            return vmap[op]
        return op

    stmt = tvm.ir_pass.IRTransform(
        stmt, None, _subst, ["Allocate",
                             "Load",
                             "Store",
                             "Variable",
                             "AttrStmt"])
    return stmt
