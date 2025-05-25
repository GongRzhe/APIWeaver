# SaasToMCP

A FastMCP server that dynamically creates MCP (Model Context Protocol) servers from web API configurations. This allows you to easily integrate any REST API, GraphQL endpoint, or web service into an MCP-compatible tool that can be used by AI assistants like Claude.

## Features

- 🚀 **Dynamic API Registration**: Register any web API at runtime
- 🔐 **Multiple Authentication Methods**: Bearer tokens, API keys, Basic auth, OAuth2, and custom headers
- 🛠️ **All HTTP Methods**: Support for GET, POST, PUT, DELETE, PATCH, and more
- 📝 **Flexible Parameters**: Query params, path params, headers, and request bodies
- 🔄 **Automatic Tool Generation**: Each API endpoint becomes an MCP tool
- 🧪 **Built-in Testing**: Test API connections before using them
- 📊 **Response Handling**: Automatic JSON parsing with fallback to text

## Installation

```bash
# Clone or download this repository
cd ~/Desktop/SaasToMCP

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Server

There are a couple of ways to run the SaasToMCP server:

**1. After installation (recommended):**

If you have installed the package (e.g., using `pip install .` from the project root after installing requirements), you can use the command-line script:

```bash
saas-to-mcp run [OPTIONS]
```
For example, to start with default settings:
```bash
saas-to-mcp run
```
To start with SSE transport (often useful for web-based tools like Claude Desktop):
```bash
saas-to-mcp run --transport sse --host 127.0.0.1 --port 8000
```

**2. Directly from the repository (for development):**

If you are running the code directly from the cloned repository without installing the package:
```bash
# From the root of the repository
python -m saas_to_mcp.cli run [OPTIONS]
```
Or:
```bash
python saas_to_mcp/cli.py run [OPTIONS]
```
Refer to `saas-to-mcp run --help` (or `python -m saas_to_mcp.cli run --help`) for available options.

Or use with the MCP CLI:

It's generally recommended to run SaasToMCP as a separate server process. However, if your MCP client supports direct installation of local Python scripts or packages as tools:

- If SaasToMCP is installed in your Python environment, you might be able to register it using its package name (this depends on your MCP client's capabilities):
  ```bash
  mcp install saas-to-mcp 
  ```
- Or, you might point to the local `cli.py` file (path may need adjustment):
  ```bash
  mcp install path/to/your/SaasToMCP/saas_to_mcp/cli.py
  ```
Consult your MCP client's documentation for the correct way to add local tools. For most uses with AI assistants, running SaasToMCP as a standalone server (see above) and connecting to its SSE endpoint is the more common approach.

### Core Tools

The server provides these built-in tools:

1. **register_api** - Register a new API and create tools for its endpoints
2. **list_apis** - List all registered APIs and their endpoints
3. **unregister_api** - Remove an API and its tools
4. **test_api_connection** - Test connectivity to a registered API
5. **run_api_tool**
    *   **Purpose**: Executes a previously registered API endpoint tool (i.e., a tool generated from an API endpoint).
    *   **Usage**: This tool allows direct invocation of any dynamically created API tool.
    *   **Arguments**:
        *   `tool_name` (string): The full name of the API tool to execute (e.g., `RegisteredAPI_endpoint_name`).
        *   `tool_params` (object): A dictionary containing the parameters required by the target API tool.
    *   **Restriction**: Cannot be used to execute other core management tools (like `register_api`, `list_apis`, `unregister_api`, `test_api_connection`, or `run_api_tool` itself).
    *   **Example MCP Call (conceptual):**
        ```json
        {
          "tool_name": "run_api_tool",
          "inputs": {
            "tool_name": "weather_get_current_weather",
            "tool_params": {
              "q": "Paris",
              "units": "metric"
            }
          }
        }
        ```

### Using with AI Assistants (like Claude Desktop)

SaasToMCP is designed to expose web APIs as tools for AI assistants that support the Model Context Protocol (MCP). Here's a general guide to using it with an assistant like Claude Desktop:

1.  **Start the SaasToMCP Server:**
    It's recommended to run SaasToMCP as a server with Server-Sent Events (SSE) transport, as this is commonly used for web-based AI tools.
    ```bash
    # After installing the package
    saas-to-mcp run --transport sse --host 127.0.0.1 --port 8000
    ```
    This will start the server, and by default, the MCP endpoint will be available at `http://127.0.0.1:8000/mcp`. You can customize the host, port, and path using the command-line options.

2.  **Configure Your AI Assistant (Claude Desktop):**
    *   Open your Claude Desktop application or interface.
    *   Look for a section related to "Tools," "Plugins," "Custom Tools," or "MCP Servers."
    *   You'll likely need to add a new MCP tool provider or server by providing the URL of the SaasToMCP endpoint (e.g., `http://127.0.0.1:8000/mcp`).
    *   **Note:** The exact steps for adding an MCP tool can vary depending on the version and specific interface of your Claude Desktop or other AI assistant. Please consult the documentation for your specific AI assistant for detailed instructions.

3.  **Register APIs and Use Tools:**
    *   Once Claude Desktop is connected to your SaasToMCP server, you can use the built-in `register_api` tool (usually by instructing the assistant) to define the web APIs you want to access. Provide the JSON configuration for the API you wish to integrate.
    *   After an API is successfully registered, its endpoints will become available as new tools that the AI assistant can use. For example, if you register an API named `mynews` with an endpoint `get_latest_headlines`, a tool like `mynews_get_latest_headlines` should become available.
    *   You can then instruct your AI assistant to use these tools (e.g., "Claude, use mynews_get_latest_headlines to find recent tech news").

By following these steps, you can extend the capabilities of your AI assistant by giving it access to a wide range of web services through SaasToMCP.

### API Configuration Format

```json
{
  "name": "my_api",
  "base_url": "https://api.example.com",
  "description": "Example API integration",
  "auth": {
    "type": "bearer",
    "bearer_token": "your-token-here"
  },
  "headers": {
    "Accept": "application/json"
  },
  "endpoints": [
    {
      "name": "list_users",
      "description": "Get all users",
      "method": "GET",
      "path": "/users",
      "params": [
        {
          "name": "limit",
          "type": "integer",
          "location": "query",
          "required": false,
          "default": 10,
          "description": "Number of users to return"
        }
      ]
    }
  ]
}
```

## Examples

### Example 1: OpenWeatherMap API

```json
{
  "name": "weather",
  "base_url": "https://api.openweathermap.org/data/2.5",
  "description": "OpenWeatherMap API",
  "auth": {
    "type": "api_key",
    "api_key": "your-api-key",
    "api_key_param": "appid"
  },
  "endpoints": [
    {
      "name": "get_current_weather",
      "description": "Get current weather for a city",
      "method": "GET",
      "path": "/weather",
      "params": [
        {
          "name": "q",
          "type": "string",
          "location": "query",
          "required": true,
          "description": "City name"
        },
        {
          "name": "units",
          "type": "string",
          "location": "query",
          "required": false,
          "default": "metric",
          "enum": ["metric", "imperial", "kelvin"]
        }
      ]
    }
  ]
}
```

### Example 2: GitHub API

```json
{
  "name": "github",
  "base_url": "https://api.github.com",
  "description": "GitHub REST API",
  "auth": {
    "type": "bearer",
    "bearer_token": "ghp_your_token_here"
  },
  "headers": {
    "Accept": "application/vnd.github.v3+json"
  },
  "endpoints": [
    {
      "name": "get_user",
      "description": "Get a GitHub user's information",
      "method": "GET",
      "path": "/users/{username}",
      "params": [
        {
          "name": "username",
          "type": "string",
          "location": "path",
          "required": true,
          "description": "GitHub username"
        }
      ]
    },
    {
      "name": "create_issue",
      "description": "Create a new issue in a repository",
      "method": "POST",
      "path": "/repos/{owner}/{repo}/issues",
      "params": [
        {
          "name": "owner",
          "type": "string",
          "location": "path",
          "required": true,
          "description": "Repository owner"
        },
        {
          "name": "repo",
          "type": "string",
          "location": "path",
          "required": true,
          "description": "Repository name"
        },
        {
          "name": "title",
          "type": "string",
          "location": "body",
          "required": true,
          "description": "Issue title"
        },
        {
          "name": "body",
          "type": "string",
          "location": "body",
          "required": false,
          "description": "Issue description"
        },
        {
          "name": "labels",
          "type": "array",
          "location": "body",
          "required": false,
          "description": "Array of label names"
        }
      ]
    }
  ]
}
```

### Example 3: Stripe API

```json
{
  "name": "stripe",
  "base_url": "https://api.stripe.com/v1",
  "description": "Stripe Payment API",
  "auth": {
    "type": "basic",
    "username": "sk_test_your_key_here",
    "password": ""
  },
  "endpoints": [
    {
      "name": "list_customers",
      "description": "List all customers",
      "method": "GET",
      "path": "/customers",
      "params": [
        {
          "name": "limit",
          "type": "integer",
          "location": "query",
          "required": false,
          "default": 10
        },
        {
          "name": "starting_after",
          "type": "string",
          "location": "query",
          "required": false,
          "description": "Cursor for pagination"
        }
      ]
    },
    {
      "name": "create_customer",
      "description": "Create a new customer",
      "method": "POST",
      "path": "/customers",
      "headers": {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      "params": [
        {
          "name": "email",
          "type": "string",
          "location": "body",
          "required": true,
          "description": "Customer email"
        },
        {
          "name": "name",
          "type": "string",
          "location": "body",
          "required": false,
          "description": "Customer name"
        }
      ]
    }
  ]
}
```

## Authentication Types

### Bearer Token
```json
{
  "auth": {
    "type": "bearer",
    "bearer_token": "your-token-here"
  }
}
```

### API Key (Header)
```json
{
  "auth": {
    "type": "api_key",
    "api_key": "your-key-here",
    "api_key_header": "X-API-Key"
  }
}
```

### API Key (Query Parameter)
```json
{
  "auth": {
    "type": "api_key",
    "api_key": "your-key-here",
    "api_key_param": "api_key"
  }
}
```

### Basic Authentication
```json
{
  "auth": {
    "type": "basic",
    "username": "your-username",
    "password": "your-password"
  }
}
```

### Custom Headers
```json
{
  "auth": {
    "type": "custom",
    "custom_headers": {
      "X-Custom-Auth": "custom-value",
      "X-Client-ID": "client-123"
    }
  }
}
```

## Parameter Locations

- **query**: Query string parameters (`?param=value`)
- **path**: Path parameters (`/users/{id}`)
- **header**: HTTP headers
- **body**: Request body (for POST, PUT, PATCH)

## Parameter Types

- **string**: Text values
- **integer**: Whole numbers
- **number**: Decimal numbers
- **boolean**: true/false
- **array**: Lists of values
- **object**: JSON objects

## Advanced Features

### Custom Timeouts
```json
{
  "timeout": 60.0  // Timeout in seconds
}
```

### Enum Values
```json
{
  "name": "status",
  "type": "string",
  "enum": ["active", "inactive", "pending"]
}
```

### Default Values
```json
{
  "name": "page",
  "type": "integer",
  "default": 1
}
```

## Error Handling

The server provides detailed error messages for:
- Missing required parameters
- HTTP errors (with status codes)
- Connection failures
- Authentication errors
- Invalid configurations

## Tips

1. **Test First**: Always use `test_api_connection` after registering an API
2. **Start Simple**: Begin with GET endpoints before moving to complex POST requests
3. **Check Auth**: Ensure your authentication credentials are correct
4. **Use Descriptions**: Provide clear descriptions for better AI understanding
5. **Handle Errors**: The server will report HTTP errors with details

## Troubleshooting

### Common Issues

1. **401 Unauthorized**: Check your authentication credentials
2. **404 Not Found**: Verify the base URL and endpoint paths
3. **Timeout Errors**: Increase the timeout value for slow APIs
4. **SSL Errors**: Some APIs may require specific SSL configurations

### Debug Mode

Run with verbose logging (if installed):
```bash
saas-to-mcp run --verbose
```
Or from the repository:
```bash
python -m saas_to_mcp.cli run --verbose
```

## Contributing

Feel free to extend this server with additional features:
- OAuth2 token refresh
- GraphQL support
- WebSocket endpoints
- Response caching
- Rate limiting
- Request retries

## License

MIT License - feel free to use and modify as needed.
