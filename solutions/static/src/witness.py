"""Render a JUnit 5 replay for an H2 invalid parameter index finding."""

import jvm
from static import Finding


def render_junit(finding: Finding, inputs: tuple[bool | int, ...] = ()) -> str:
    if finding.kind != "invalid-parameter-index":
        raise ValueError(f"Unsupported replay oracle: {finding.kind}")
    method = finding.pc.method
    params = method.extension.params
    if len(inputs) != len(params):
        raise ValueError("Replay requires a concrete witness input for each parameter")
    args = []
    for typ, value in zip(params, inputs):
        if isinstance(typ, jvm.Boolean) and type(value) is bool:
            args.append(str(value).lower())
        elif isinstance(typ, jvm.Int) and type(value) is int and -(2**31) <= value < 2**31:
            args.append(str(value))
        else:
            raise ValueError(f"Invalid replay input {value!r} for {typ}")
    call = f"{method.classname.dotted()}.{method.extension.name}({', '.join(args)})"
    return f"""import java.sql.SQLException;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class JdbcWitnessTest {{
    @Test
    void invalidParameterIndex() {{
        SQLException error = assertThrows(SQLException.class, () -> {call});
        assertEquals(90008, error.getErrorCode());
        assertTrue(error.getMessage().contains("parameterIndex"), error::getMessage);
    }}
}}
"""
