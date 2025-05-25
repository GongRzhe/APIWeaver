import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from saas_to_mcp.server import SaasToMCP
from saas_to_mcp.models import APIConfig, APIEndpoint, RequestParam
from fastmcp import Context

# Mock API Configurations
MOCK_API_CONFIG_DICT_VALID = {
    "name": "TestAPI",
    "base_url": "http://mockapi.test",
    "description": "A Test API",
    "endpoints": [
        {
            "name": "get_data",
            "description": "Gets some data",
            "method": "GET",
            "path": "/data",
            "params": [
                {"name": "item_id", "type": "string", "location": "query", "required": True},
                {"name": "limit", "type": "integer", "location": "query", "required": False, "default": 10},
            ],
        },
        {
            "name": "post_data",
            "description": "Posts some data",
            "method": "POST",
            "path": "/data",
            "params": [
                {"name": "content", "type": "string", "location": "body", "required": True},
                {"name": "is_urgent", "type": "boolean", "location": "body", "required": False, "default": False},
            ]
        }
    ],
}

MOCK_API_CONFIG_DICT_NO_PARAMS = {
    "name": "NoParamAPI",
    "base_url": "http://noparam.test",
    "endpoints": [
        {
            "name": "ping",
            "description": "Pings the service",
            "method": "GET",
            "path": "/ping",
            "params": []
        }
    ]
}


@pytest.fixture
async def mcp_server():
    """Fixture to create an SaasToMCP server instance."""
    server = SaasToMCP(name="TestServer")
    # We need to properly close the http_clients created by the server
    yield server
    # Cleanup: Close any httpx clients created during tests
    for client in server.http_clients.values():
        await client.aclose()
    server.http_clients.clear()
    server.apis.clear()
    # Clear registered tools from mcp instance
    # This is a bit of a hack, ideally FastMCP would have a clear_all_tools()
    server.mcp.tools.clear() 


@pytest.fixture
def mock_context():
    """Fixture to create a mock Context object."""
    ctx = MagicMock(spec=Context)
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.debug = AsyncMock()
    return ctx

@pytest.mark.asyncio
async def test_run_valid_api_tool(mcp_server: SaasToMCP, mock_context: Context):
    """Test running a successfully registered and valid API tool."""
    # Register the mock API
    # Mock the actual HTTP request made by the dynamic tool
    with patch('httpx.AsyncClient.request', new_callable=AsyncMock) as mock_http_request:
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = MagicMock(return_value={"message": "success", "item_id": "123", "limit": 5})
        mock_response.text = '{"message": "success", "item_id": "123", "limit": 5}'
        mock_http_request.return_value = mock_response

        await mcp_server.mcp.run_tool("register_api", config=MOCK_API_CONFIG_DICT_VALID, ctx=mock_context)

        tool_name_to_run = "TestAPI_get_data"
        params_to_run_with = {"item_id": "123", "limit": 5}
        
        result = await mcp_server.run_api_tool(
            tool_name=tool_name_to_run,
            tool_params=params_to_run_with,
            ctx=mock_context
        )
        
        assert result == {"message": "success", "item_id": "123", "limit": 5}
        mock_http_request.assert_called_once()
        args, kwargs = mock_http_request.call_args
        assert kwargs['method'] == "GET"
        assert kwargs['url'] == "/data"
        assert kwargs['params'] == params_to_run_with

@pytest.mark.asyncio
async def test_run_non_existent_tool(mcp_server: SaasToMCP, mock_context: Context):
    """Test running a tool that does not exist."""
    with pytest.raises(ValueError) as excinfo:
        await mcp_server.run_api_tool(
            tool_name="ThisToolDoesNotExist",
            tool_params={"param": "value"},
            ctx=mock_context
        )
    assert "Tool 'ThisToolDoesNotExist' not found" in str(excinfo.value)

@pytest.mark.asyncio
@pytest.mark.parametrize("core_tool_name", [
    "register_api",
    "list_apis",
    "unregister_api",
    "test_api_connection",
    "run_api_tool"
])
async def test_run_core_tool_is_forbidden(mcp_server: SaasToMCP, mock_context: Context, core_tool_name: str):
    """Test attempting to run a core tool via run_api_tool."""
    # Ensure core tools are registered (they are in _setup_core_tools)
    assert core_tool_name in mcp_server.mcp.tools

    with pytest.raises(ValueError) as excinfo:
        await mcp_server.run_api_tool(
            tool_name=core_tool_name,
            tool_params={}, # Params don't matter as it should fail before execution
            ctx=mock_context
        )
    assert f"Core tool '{core_tool_name}' cannot be executed via run_api_tool" in str(excinfo.value)

@pytest.mark.asyncio
async def test_run_tool_missing_required_parameter(mcp_server: SaasToMCP, mock_context: Context):
    """Test running an API tool with a missing required parameter."""
    with patch('httpx.AsyncClient.request', new_callable=AsyncMock) as mock_http_request:
        # No need for mock_http_request to do anything as it shouldn't be called
        await mcp_server.mcp.run_tool("register_api", config=MOCK_API_CONFIG_DICT_VALID, ctx=mock_context)

        tool_name_to_run = "TestAPI_get_data"
        # Missing 'item_id' which is required
        params_to_run_with = {"limit": 5} 
        
        with pytest.raises(ValueError) as excinfo:
            await mcp_server.run_api_tool(
                tool_name=tool_name_to_run,
                tool_params=params_to_run_with,
                ctx=mock_context
            )
        # This error comes from the dynamically generated api_tool_function
        assert "Required parameter 'item_id' not provided" in str(excinfo.value)
        mock_http_request.assert_not_called()

@pytest.mark.asyncio
async def test_run_tool_with_no_params(mcp_server: SaasToMCP, mock_context: Context):
    """Test running a tool that takes no parameters."""
    with patch('httpx.AsyncClient.request', new_callable=AsyncMock) as mock_http_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "pong"
        mock_http_request.return_value = mock_response

        await mcp_server.mcp.run_tool("register_api", config=MOCK_API_CONFIG_DICT_NO_PARAMS, ctx=mock_context)
        
        tool_name_to_run = "NoParamAPI_ping"
        result = await mcp_server.run_api_tool(
            tool_name=tool_name_to_run,
            tool_params={}, # No parameters
            ctx=mock_context
        )
        assert result == "pong"
        mock_http_request.assert_called_once_with(method='GET', url='/ping', params=None, headers=None, json=None, timeout=30.0)

# Optional Test: Invalid parameter types
# This is harder to test robustly as Pydantic models used in APIConfig/Endpoint
# might perform coercion, and FastMCP itself might also do so.
# The current dynamic tool generation doesn't add strict runtime type checks beyond what Pydantic/FastMCP provide.
# For a simple case, if the tool's underlying httpx call fails due to bad data, that would be an error.

@pytest.mark.asyncio
async def test_run_tool_invalid_parameter_type_if_possible(mcp_server: SaasToMCP, mock_context: Context):
    """
    Test running an API tool with an invalid parameter type.
    This test's success depends on how type checking/coercion is handled
    by Pydantic, FastMCP, and the dynamically generated tool function.
    The dynamically generated tool converts params to string for path/header/query.
    For 'body' params, if they are part of a JSON structure, httpx might complain.
    """
    with patch('httpx.AsyncClient.request', new_callable=AsyncMock) as mock_http_request:
        # If the call reaches httpx with a type it can't serialize for JSON, it might fail.
        # Let's test a POST request where 'limit' is expected as integer by the API (though our model says string for simplicity for now)
        # but we'll make the dynamic tool expect an integer.
        
        # Modify MOCK_API_CONFIG_DICT_VALID for this test to make 'limit' a body param and type integer
        # This is a bit artificial as 'limit' is usually a query param.
        custom_config = {
            "name": "TypeTestAPI",
            "base_url": "http://typetest.test",
            "endpoints": [
                {
                    "name": "submit_item",
                    "description": "Submits an item with an ID",
                    "method": "POST",
                    "path": "/submit",
                    "params": [
                        # For this test, let's make 'item_id' an integer in the body
                        {"name": "item_id", "type": "integer", "location": "body", "required": True},
                    ]
                }
            ]
        }
        await mcp_server.mcp.run_tool("register_api", config=custom_config, ctx=mock_context)

        tool_name_to_run = "TypeTestAPI_submit_item"
        # Pass 'item_id' as a string, when the tool definition implies it should be an integer for the body.
        # The `_create_endpoint_tool` maps "integer" to `int` annotation.
        # FastMCP's `run_tool` might attempt conversion based on these annotations.
        # If conversion fails, it should raise an error.
        
        # httpx will likely raise an error if it tries to JSON-encode a non-serializable type
        # if the parameter isn't converted by FastMCP or Pydantic before reaching httpx.
        # However, simple types like string for an int field might be coerced by Pydantic/FastMCP.
        # Let's try sending something that is definitely not an int.
        
        # This test relies on FastMCP's type validation based on annotations.
        # If FastMCP successfully converts "not_an_int" to an int, this test would need adjustment.
        # Typically, Pydantic might raise a validation error if such conversion is not possible.
        
        # The call to `self.mcp.run_tool` is where FastMCP would use the type hints.
        # Let's assume FastMCP will try to coerce based on the `int` annotation.
        # If "not_an_int" is passed, it should fail.
        
        with pytest.raises(Exception) as excinfo: # Catching general Exception as it could be TypeError or ValueError from Pydantic/FastMCP
            await mcp_server.run_api_tool(
                tool_name=tool_name_to_run,
                tool_params={"item_id": "this_is_not_an_integer_at_all"}, # This should fail conversion
                ctx=mock_context
            )
        
        # Check that httpx was not called because validation should fail before
        mock_http_request.assert_not_called()
        # The exact error message can vary depending on where the type validation occurs (Pydantic, FastMCP).
        # We expect some form of validation error.
        assert "validation error" in str(excinfo.value).lower() or "invalid literal for int()" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_cleanup_after_unregister(mcp_server: SaasToMCP, mock_context: Context):
    """Test that HTTP client is closed and API config is removed after unregistering."""
    with patch('httpx.AsyncClient.request', new_callable=AsyncMock) as mock_http_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = MagicMock(return_value={"status": "ok"})
        mock_http_request.return_value = mock_response
        
        await mcp_server.mcp.run_tool("register_api", config=MOCK_API_CONFIG_DICT_VALID, ctx=mock_context)

    assert "TestAPI" in mcp_server.apis
    assert "TestAPI" in mcp_server.http_clients
    original_client = mcp_server.http_clients["TestAPI"]
    
    # Patch aclose on the specific client instance to check it's called
    original_client.aclose = AsyncMock()

    await mcp_server.mcp.run_tool("unregister_api", api_name="TestAPI", ctx=mock_context)

    assert "TestAPI" not in mcp_server.apis
    assert "TestAPI" not in mcp_server.http_clients
    original_client.aclose.assert_awaited_once()
    assert "TestAPI_get_data" not in mcp_server.mcp.tools
    assert "TestAPI_post_data" not in mcp_server.mcp.tools

```
