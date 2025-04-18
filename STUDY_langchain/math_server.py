# https://apidog.com/jp/blog/langchain-mcp-server-jp/

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """
    2つの整数を加算する
    """
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """
    2つの整数を乗算する
    """
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
