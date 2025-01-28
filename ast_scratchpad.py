from ast import *

FunctionDef(
    name='recursive_test',
    args=arguments(
        posonlyargs=[],
        args=[
            arg(arg='n')],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
    body=[
        If(
            test=Compare(
                left=Name(id='n', ctx=Load()),
                ops=[
                    Gt()],
                comparators=[
                    Constant(value=0)]),
            body=[
                Return(
                    value=BinOp(
                        left=Call(
                            func=Name(id='recursive_test', ctx=Load()),
                            args=[
                                BinOp(
                                    left=Name(id='n', ctx=Load()),
                                    op=Sub(),
                                    right=Constant(value=1))],
                            keywords=[]),
                        op=Mult(),
                        right=Name(id='n', ctx=Load())))],
            orelse=[]),
        Return(
            value=Constant(value=0))],
    decorator_list=[])